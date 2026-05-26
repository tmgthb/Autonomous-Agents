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

## Research papers: 2026 4/5

[2026 (5/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2026 (4/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_4.md), [2026 (3/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_3.md), [2026 (2/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_2.md), [2026 (1/5)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_1.md), [2025 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_4.md),[2025 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_3.md), [2025 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (1/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_01.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)


Chronological order. 





</div>



---





#### 17th May 2026


[NewsLens: A Multi-Agent Framework for Adversarial News Bias Navigation](http://arxiv.org/abs/2605.17364)

- NewsLens: introduces a five-agent adversarial pipeline that reframes media bias analysis as structured knowledge navigation rather than simple classification.
- The framework utilizes persona-constrained LLMs to independently generate competing ideological framings, which are then synthesized alongside propaganda detection and fact verification to expose structural omissions.
- By operationalizing the latent ideological skew of LLMs as an analytical instrument, the system produces interpretable epistemic maps that reveal what both progressive and conservative framings jointly ignore.

---

[Generating Realistic Safety-Critical Scenarios for Vehicle-Pedestrian Interactions](http://arxiv.org/abs/2605.17229)

- MA-SST-DDPG (Multi-agent State-space Transformer-enhanced Deep Deterministic Policy Gradient): introduces a three-stage framework that integrates real-world behavioral grounding with online reinforcement learning in simulation to generate high-fidelity safety-critical vehicle-pedestrian interaction data.
- The framework utilizes a State-Space Transformer (SST) module to capture complex temporal dependencies and prioritize safety-critical scenarios through dynamic gating and importance weighting.
- By combining data-driven pre-training with simulation-based online refinement, the approach produces a refined model capable of generating diverse, realistic evasive behaviors that outperform baseline methods in trajectory accuracy and behavioral realism.

---


[Agent Bazaar: Enabling Economic Alignment in Multi-Agent Marketplaces](http://arxiv.org/abs/2605.17698)

- Agent Bazaar: introduces a multi-agent simulation framework for evaluating Economic Alignment in LLM-based marketplaces, utilizing Simulation Environment, Market Clearing, Agent Observation, LLM Agent, Stabilizing Firm, Skeptical Guardian, and REINFORCE++ Trainer.
- The framework identifies systemic failure modes including algorithmic price instability and Sybil-based reputation deception, which are orthogonal to general LLM reasoning capabilities.
- Targeted RL training with REINFORCE++ produces economically aligned agents that outperform frontier models by internalizing market externalities and maintaining stability under adversarial conditions.

---

[Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?](http://arxiv.org/abs/2605.17694)

- Sotopia: introduces a systematic evaluation of socio-cognitive effects in LLMs by simulating power-asymmetric conversations using Persona Hub, DailyPersuasion, and Do-Not-Answer datasets.
- The study evaluates four socio-cognitive effects—pronoun usage, language coordination, authority bias, and harmful compliance—across six LLMs to assess realism and safety in hierarchical interactions.
- Results indicate that while LLMs reproduce human-like socio-cognitive patterns, they exhibit significant variability in controllability, with larger models showing increased resistance to status-driven authority bias.

---

[PULSE: Agentic Investigation with Passive Sensing for Proactive Intervention in Cancer Survivorship](http://arxiv.org/abs/2605.17679)

- PULSE: introduces, a system that replaces fixed feature pipelines with agentic investigation of passive sensing data to provide proactive mental health support for cancer survivors.
- The framework utilizes an LLM agent that performs multi-turn reasoning using Sense, Think, Inference, Agent Reflection, User Memory, Cross-User RAG, and MCP Sensing Tools to dynamically interpret behavioral signals.
- PULSE demonstrates that agentic reasoning is the primary driver of prediction accuracy, effectively dissociating emotional desire from behavioral availability to improve intervention timing.

---

[MUIAnno: An Expert-Annotated Dataset and Evaluation Benchmark for Mobile UI Understanding](http://arxiv.org/abs/2605.17656)

- MUIAnno: introduces a high-quality, expert-annotated dataset of 1,000 iOS screens with 27,367 labeled UI elements to support fine-grained mobile UI understanding.
- The framework utilizes a custom web-based annotation tool and a multi-stage expert validation process to ensure semantic consistency and precise bounding box localization.
- The study establishes a prompt-based benchmark for evaluating multimodal LLMs on UI element extraction, revealing performance gaps between proprietary and open-source models in complex interface scenarios.

---

[Causal Intervention-Based Memory Selection for Long-Horizon LLM Agents](http://arxiv.org/abs/2605.17641)

- CMI (Causal Memory Intervention): introduces a memory-selection technique for LLMs that estimates the causal effect of candidate memories on the model's answer to ensure only useful and stable information is utilized.
- The framework employs a causal decision-making process by evaluating model performance across no-memory, with-memory, and perturbed-memory conditions to filter out irrelevant or harmful context.
- The authors also introduce CAUSAL-LOCOMO, a causally annotated benchmark designed to evaluate the robustness of LLM agents against misleading or poisoned memory retrieval.

---

[WebGameBench: Requirement-to-Application Evaluation for Coding Agents via Browser-Native Games](http://arxiv.org/abs/2605.17637)

- WebGameBench: introduces a requirement-to-application benchmark that evaluates coding agents by synthesizing browser-native games from a Structured WebGame Specification and verifying runtime behavior.
- The framework utilizes a closed-loop pipeline connecting specification-based generation, local deployment, and interactive runtime evaluation to assess if delivered applications satisfy specified functional requirements.
- The evaluation pipeline includes a Runtime Evaluator that interacts with the deployed application via a Playwright Browser Controller to assign quality labels based on observable runtime dynamics.

---

[AI Agents May Always Fall for Prompt Injections](http://arxiv.org/abs/2605.17634)

- CI Framework: introduces a principled approach for evaluating prompt injection vulnerabilities by decomposing agent actions into Contextual Integrity dimensions including sender, receiver, subject, information type, and transmission principle.
- The framework demonstrates that prompt injection is a violation of contextual norms rather than just a data-instruction separation failure, and it includes planning-, reasoning- and tool use-agents.
- The research provides empirical evidence that current LLM-based agents fail to maintain context-sensitive boundaries, leading to security breaches when they overgeneralize authorization or collapse simultaneous information flows.

---

[Episodic-Semantic Memory Architecture for Long-Horizon Scientific Agents](http://arxiv.org/abs/2605.17625)

- Dual-Process Memory Architecture: introduces a memory system for scientific agents that decouples immediate episodic needs from long-term consolidated knowledge to maintain coherence over long-horizon interactions.
- The architecture utilizes an Orchestrator Agent, Memory Manager, Episodic Buffer, Neocortical Memory, Context Assembly, Inference LLM, Background Worker, and Consolidation LLM to manage scientific workflows efficiently.
- By separating raw conversational traces from incrementally consolidated profiles, the system achieves sustained operation beyond standard context limits while maintaining high accuracy for scientific fact retention.

---

[GraphMind: From Operational Traces to Self-Evolving Workflow Automation](http://arxiv.org/abs/2605.17617)

- GraphMind: introduces an end-to-end system that constructs, executes, and evolves action-centric workflow graphs from operational traces using Workflow Extractor, Clustering, Graph Database, Multi-Agent Graph Traversal, Adaptive Traversal Reinforcement, Vector Index, Global Context, and Temporary Context.
- The system utilizes a multi-agent traversal engine that combines graph-guided retrieval with LLM-driven reasoning to navigate workflows and generate trajectories for self-optimization.
- Adaptive Traversal Reinforcement employs an Ant Colony Optimization-inspired mechanism to deposit reinforcement on successful paths and decay stale elements, enabling the graph to self-evolve without manual intervention.

---

[Convergence of Stochastic First-Order Algorithms in Bertrand Competition Under Incomplete Information](http://arxiv.org/abs/2605.17607)

- RRM algorithms: introduces a framework for analyzing the convergence of learning agents in Bayesian Bertrand competition by constructing a global Lyapunov function for projected primal dynamics.
- The paper proves that Euclidean RRM algorithms converge almost surely to the unique Bayes–Nash equilibrium despite the violation of standard monotonicity and Minty variational inequality conditions.
- By approximating infinite-dimensional strategy spaces with piecewise-linear functions, the authors establish global asymptotic stability for symmetric duopoly pricing models.

---

[NeuSymMS: A Hybrid Neuro-Symbolic Memory System for Persistent, Self-Curating LLM Agents](http://arxiv.org/abs/2605.17596)

- NeuSymMS: introduces a hybrid neuro-symbolic memory architecture that utilizes an LLM-based Fact Extractor for neural parsing and a CLIPS-based Expert System for deterministic, rule-based memory management.
- The system employs a dual-horizon memory model with access-based promotion and time-based pruning to maintain persistent, self-curating knowledge while avoiding context-window bloat.
- By representing user knowledge as scoped subject-relation-value triples, the framework ensures auditable, contradiction-aware memory isolation across multi-agent and multi-tenant platforms.

---

[Automated Root-Cause Subclassification and No-Code Fix Generation for Invalid Bug Reports](http://arxiv.org/abs/2605.17561)

- IssueSupport: introduces an automated framework that leverages LLMs to subclassify invalid bug reports by root cause and generate actionable no-code fixes using LLM, Vector Database, Search API, Orchestrator, Reader LLM, LanceDB, Serper Search API, and OpenRouter.
- The framework utilizes a multi-step pipeline incorporating RAG and agentic web search to ground LLM responses in project-specific documentation and real-time external data.
- Experimental results demonstrate that RAG improves subclassification performance, while agentic web search enhances the quality of generated no-code fixes for complex invalid bug reports.

---

[Firefly: Illuminating Large-Scale Verified Tool-Call Data Generation from Real APIs](http://arxiv.org/abs/2605.17558)

- FIREFLY: introduces a pipeline for generating verified tool-call data by inverting the synthesis process, starting with real-world API exploration followed by back-chaining task generation.
- The framework utilizes a retrieval-augmented simulator to cache exploration traces, enabling stable, reproducible, and offline reinforcement learning for LLMs.
- By grounding data generation in real Model Context Protocol servers and using graph-guided exploration, the approach ensures high-quality, verifiable trajectories for training tool-calling agents.

---

[Evaluating Deep Research Agents on Expert Consulting Work: A Benchmark with Verifiers, Rubrics, and Cognitive Traps](http://arxiv.org/abs/2605.17554)

- DRBench: introduces a benchmark for evaluating deep research agents on decision-grade management consulting tasks using Deterministic Verifiers, SME-Graded Rubric, and Cognitive Traps.
- The framework utilizes a Multi-Agent Dispatch Infrastructure with Agent-Specific Adapters to test LLMs on complex, multi-document research prompts while employing a Result Storage API and Diagnostic Tooling for performance analysis.
- The benchmark evaluates LLMs across five capability-targeted prompt classes, identifying agent-specific failure modes such as fabrication, cascading arithmetic errors, and system volatility.

---

[Rethinking Code Review in the Age of AI: A Vision for Agentic Code Review](http://arxiv.org/abs/2605.17548)

- Agent-Orchestrated Collaborative Review framework: introduces a multi-agent system that integrates specialized LLM agents with human-in-the-loop quality gates to transform the code review lifecycle from a manual process into an interactive, evidence-based workflow.
- The framework utilizes a PR Augmentation Agent to orchestrate specialized analysis agents, including Alignment-, Bug Proneness-, Impact-, and Runtime-Analysis agents, to provide verifiable evidence for human reviewers.
- The system incorporates a PR Review Agent as a central natural language interface, supported by Toxicity- and Usefulness-Measurement agents to ensure professional and actionable feedback during the review process.

---

[Memory-Guided Tree Search with Cross-Branch Knowledge Transfer for LLM Solver Synthesis](http://arxiv.org/abs/2605.17539)

- MEMOIR: introduces a tree-search framework for LLM-based solver synthesis that utilizes a two-level memory hierarchy to separate branch-local refinement details from global algorithmic insights.
- The framework employs five LLM-driven operators—PROPOSE, REPAIR, IMPROVE, CRITIC, and REFLECT—to iteratively synthesize and refine heuristic solvers while maintaining execution-grounded feedback.
- By distilling branch-specific trajectories into compressed global summaries, MEMOIR enables effective cross-branch knowledge transfer without polluting the LLM context with low-level debugging artifacts.

---

[Self-supervised Hierarchical Visual Reasoning with World Model](http://arxiv.org/abs/2605.17537)

- ResDreamer: introduces a hierarchical world model that employs residually connected visual planning representations to enable progressive abstraction of world dynamics.
- The architecture utilizes PPB components to transmit reconstruction residuals upward, creating an efficient information channel for modulated visual foresight.
- ResDreamer achieves state-of-the-art sample and parameter efficiency in online RL by training purely on self-supervised imagined trajectories without language-conditioned modules.

---

[AgentModernize: Preserving Business Logic in Legacy Modernization with Multi-Agent LLMs and Behavioral Specification Graphs](http://arxiv.org/abs/2605.17535)

- AgentModernize: introduces a multi-agent framework that treats legacy modernization as a behavioral preservation problem by decomposing the process into extraction, specification, transformation, and validation phases.
- The framework utilizes a Behavioral Specification Graph (BSG) as a structured intermediate representation and trust boundary to ensure business logic is explicit and verifiable before code generation.
- An integrated feedback loop enables the Equivalence Validator to provide targeted correction instructions to the Modernization Transformer, significantly improving behavioral equivalence rates compared to single-pass LLM approaches.

---

[Don’t Guess, Just Ask: Resolving Ambiguity in Referring Segmentation via Multi-turn Clarification](http://arxiv.org/abs/2605.17531)

- IC-Seg: introduces an agentic framework that resolves ambiguous user queries in referring segmentation through multi-turn clarification dialogues before performing final object localization.
- The framework utilizes a Policy MLLM to interact with a User Simulator, guided by a Hi-GRPO optimization strategy that provides dense supervision at trajectory, turn, and step levels.
- IC-Seg achieves state-of-the-art performance on the newly established Ambi-RVOS benchmark by replacing unreliable guessing with proactive intent clarification.

---

[SaaSBench: Exploring the Boundaries of Coding Agents in Long-Horizon Enterprise SaaS Engineering](http://arxiv.org/abs/2605.17526)

- SaaSBench: introduces a benchmark platform designed to evaluate the ability of coding agents to generate and deploy enterprise-level SaaS systems from scratch, utilizing Task Construction, System Generation, Evaluation Protocol, DAG Test Suite, Docker Environments, LLM-as-Judge, and Rule-Based Scoring.
- The benchmark employs a dependency-aware hybrid evaluation paradigm that uses a directed acyclic graph (DAG) to assess agents across six engineering capability dimensions: deployment, data, API, business logic, authorization, and quality.
- Experimental results reveal that current state-of-the-art coding agents struggle with long-horizon system setup, with over 95% of failures occurring before agents reach deep business logic.

---

[The Capability Paradox: How Smarter Auditors Make Multi-Agent Systems Less Secure](http://arxiv.org/abs/2605.17480)

- MAS: introduces a hierarchical architecture where more capable LLMs acting as Workers paradoxically increase system-level vulnerability to semantic hijacking by laundering adversarial narratives into authoritative reports.
- The framework identifies that linguistic certainty, rather than syntactic injection, serves as the primary mediator for successful attacks across the inter-agent trust boundary.
- The authors propose heterogeneous ensemble verification, which exploits capability asymmetries between Workers to disrupt the certainty-to-execution chain and improve system security.

---

[Event-B Agent: Towards LLM Agent for Formal Model Synthesis and Repair](http://arxiv.org/abs/2605.17475)

- Event-B Agent: introduces an end-to-end neurosymbolic framework for formal model synthesis and repair that coordinates model construction and proof derivation using Refinement Strategy Planning LLM, Model Synthesis LLM, Model Repair LLM, Fix Strategy Decision LLM, Model Checker, Theorem Prover, SMT Solver, and Repair Rules Recommendation Component.
- The framework employs a refinement-based approach to decompose complex formal systems into smaller, manageable steps, ensuring that properties proven in earlier stages are preserved through gluing invariants.
- By integrating LLMs with symbolic verification tools, the system achieves high consistency and correctness in formal model development while maintaining efficiency through automated proof-guided repair.

---

[VerifyMAS: Hypothesis Verification for Failure Attribution in LLM Multi-Agent Systems](http://arxiv.org/abs/2605.17467)

- VerifyMAS: introduces a hypothesis verification framework for failure attribution in LLM-MAS that decomposes the task into trajectory-level error validation and fine-grained agent localization.
- The framework utilizes an LLM verifier to evaluate failure hypotheses against full interaction trajectories, enabling the detection of global failures that manifest across multiple steps.
- VerifyMAS employs a hypothesis-based data construction strategy and supervised fine-tuning to improve diagnostic accuracy while maintaining generalization to out-of-distribution trajectories.

---

[Trust No Tool: Evaluating and Defending LLM Agents under Untrusted Tool Feedback](http://arxiv.org/abs/2605.17453)

- VISTA-GUARD (Variable-state Inference for Safe Tool Actions): introduces a framework for detecting cognitive poisoning in LLM agents by scoring final-action risk based on structured trajectory-state evidence and parameter evidence.
- The paper constructs TRUST-BENCH, a benchmark of 1,970 hidden-trigger tool-compromise episodes, to evaluate agent security under untrusted tool feedback where malicious behavior is state-conditioned.
- The proposed GUARDEDJOINT metric provides an asymmetric cost-sensitive evaluation that penalizes missed malicious actions while preserving benign utility, demonstrating that trajectory-aware final-action scoring outperforms prompt-centric heuristics.

---

[DeTrack: A Benchmark and Altitude-Aware Dual World Model for Drone-embodied Tracking](http://arxiv.org/abs/2605.17451)

- AaDWorlds: introduces a drone-embodied tracking framework that utilizes ReDeT, AaP, and DWM to resolve the altitude-mediated contradiction between target visibility and flight safety.
- The framework integrates a reinforcement learning-based policy with dual world models to predict future states at high and low altitudes, providing auxiliary clues for robust closed-loop navigation.
- The research also presents the DeTrack benchmark, a large-scale dataset featuring interactive 3D environments designed to evaluate drone-embodied tracking under complex occlusion and scale variation.

---

[ContraFix: Agentic Vulnerability Repair via Differential Runtime Evidence and Skill Reuse](http://arxiv.org/abs/2605.17450)

- ContraFix: introduces an agentic framework for automated vulnerability repair that utilizes differential runtime evidence and a dual-track skill base to resolve semantic misunderstanding in LLM-based agents.
- The framework coordinates a Mutator, Analyzer, and Patcher to isolate causal variables by comparing crashing and non-crashing execution variants, thereby generating precise repair specifications.
- ContraFix achieves state-of-the-art performance on SEC-Bench and PatchEval benchmarks by leveraging accumulated repair and mutation skills to reduce repair costs and improve success rates across multiple programming languages.

---

[Self-Improving CAD Generation Agents with Finite Element Analysis as Feedback](http://arxiv.org/abs/2605.17448)

- Hephaestus-CCX: introduces an engineering-grounded CAD generation task that utilizes iterative feedback from blueprints, visual inspection, and finite element analysis to produce valid multi-part STEP artifacts.
- The framework employs an LLM agent that refines CAD programs through a closed-loop system, incorporating structured blueprint planning, 21-view visual rendering, and CalculiX-based structural validation.
- Experimental results demonstrate that organizing test-time compute into structured engineering feedback significantly improves the ability of LLMs to satisfy complex physical and geometric requirements compared to one-shot generation.

---

[MemRepair: Hierarchical Memory for Agentic Repository-Level Vulnerability Repair](http://arxiv.org/abs/2605.17444)

- MemRepair: introduces a memory-augmented agentic framework that utilizes a three-tier memory hierarchy (L1 History-Fix Memory, L2 Security-Pattern Memory, and L3 Refinement-Trajectory Memory) to enable experience-driven vulnerability repair.
- The framework employs a feedback-driven refinement loop, incorporating a Locator agent, a Patcher agent, and a Verifier agent to iteratively generate and validate patches based on runtime evidence.
- MemRepair achieves state-of-the-art performance on vulnerability repair benchmarks by leveraging hierarchical memory to guide LLMs in producing precise, surgical fixes while maintaining cost-efficiency.

---

[DiagEval: Trajectory-Conditioned Diagnosis for Reliable Software Evaluation with GUI Agents](http://arxiv.org/abs/2605.17439)

- DiagEval: introduces a trajectory-conditioned diagnostic protocol that transforms failed GUI-agent rollouts into active diagnostic processes to disambiguate evaluator-side errors from genuine software defects.
- The framework utilizes a Failure Diagnostic Summary (FDS) to identify fork nodes and dispatches targeted diagnostic branches, which are ranked by Expected Information Gain (EIG) to refine an internal attribution score.
- By leveraging structured diagnostic evidence and informative counter-evidence, DiagEval improves evaluation accuracy and reliability across multiple GUI-agent frameworks without requiring specialized training.

---

[MATE: Solving Contextual Markov Decision Processes with Memory of Accumulated Transition Embeddings](http://arxiv.org/abs/2605.17431)

- MATE (Memory of Accumulated Transition Embeddings): introduces a permutation-invariant memory architecture for solving CMDPs by replacing complex sequence models with sum-aggregated transition embeddings.
- The framework leverages the permutation invariance of the context posterior to maintain sufficient expressiveness while enabling constant-time inference and temporal parallelization during updates.
- MATE achieves competitive performance against Transformer and RNN baselines across MuJoCo, Meta-World, and T-Maze benchmarks while maintaining a lightweight and computationally efficient design.

---

[Human-Flow Digital Twin for Predicting the Effects of Mobility Introduction on Visitor Circulation](http://arxiv.org/abs/2605.17426)

- Human-Flow Digital Twin framework: introduces a simulation platform that predicts visitor circulation changes by integrating learned destination-choice models with physics-aware multi-agent simulators.
- The framework models mobility introduction as environmental modifications, enabling counterfactual analysis without requiring intervention-specific training datasets.
- The platform couples SUMO and JuPedSim via TraCI to preserve both macro-level routing and fine-grained pedestrian interactions, validated with high-fidelity field data.

---

[Soap2Soap: Long Cinematic Video Remaking via Multi-Agent Collaboration](http://arxiv.org/abs/2605.17423)

- Soap2Soap: introduces a multi-agent framework for long-horizon cinematic video remaking that enforces narrative and visual consistency through a Dual-Bridge Consistency mechanism.
- The framework utilizes a Video Understanding Agent to build a structured Language Bridge (Sjson) and a Visual Bridge (M), which guide the Video Generation Agent in producing temporally coherent video segments.
- A Verification Agent provides closed-loop feedback to ensure identity stability and narrative fidelity, while grid-based batch keyframe synthesis mitigates error accumulation across long sequences.

---

[Rethinking Side-Channel Analysis: Automated Discovery and Analysis of Side-Channel Leakage with LLM-Assisted Agents](http://arxiv.org/abs/2605.17406)

- SCAgent: introduces an automated framework for side-channel risk analysis that decomposes the process into sensitive event discovery, LLM-assisted channel proposal with explicit verification, and foundation-model-based leakage analysis.
- The framework utilizes Mobile-Agent-E for semantic event identification and employs an iterative proposer-verifier loop to discover OS-level side channels while mitigating LLM hallucinations.
- SCAgent achieves high-accuracy inference on foreground app identification and website fingerprinting by combining ROCKET-based feature extraction with TabPFN for data-efficient classification.

---

[Heterogeneous Information-Bottleneck Coordination Graphs for Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.17393)

- HIBCG: introduces a heterogeneous coordination-graph learning framework that jointly optimizes graph topology and message capacity using group-aware information bottleneck principles.
- The framework utilizes a group-aligned block-diagonal prior to enforce sparse inter-group connections while maintaining dense intra-group communication.
- HIBCG decomposes the learning objective into a structural path (AIB) and a message compression path (XIB), achieving state-of-the-art performance in large-scale multi-agent scenarios.

---

[ADR: AN AGENTIC DETECTION SYSTEM FOR ENTERPRISE AGENTIC AI SECURITY](http://arxiv.org/abs/2605.17380)

- ADR (Agentic AI Detection and Response): introduces an enterprise-grade framework for securing AI agents operating through the Model Context Protocol by integrating high-fidelity telemetry, hierarchical online detection, and systematic offline red-teaming.
- The system utilizes an ADR Sensor to reconstruct causal chains of agentic activity, a two-tier ADR Detector to balance cost and precision, and an ADR Explorer to discover hard attack variants via evolutionary algorithms.
- ADR-Bench provides a comprehensive evaluation suite covering 17 attack techniques across 5 tactics, demonstrating superior performance in precision and F1-score compared to existing security baselines in enterprise environments.

---

[FML-bench: A Controlled Study of AI Research Agent Strategies from the Perspective of Search Dynamics](http://arxiv.org/abs/2605.17373)

- FML-bench: introduces a controlled benchmark for AI research agents that isolates agent strategy from execution infrastructure using Task Specification, Agent Experiment Loop, Shared Code Editor, Shared Execution Environment, Process-level Evaluation, and Final Evaluation.
- The framework enables precise analysis of search dynamics by decoupling agent-specific strategies from standardized execution components, facilitating a fair comparison across diverse search topologies.
- Empirical results demonstrate that strategy complexity does not guarantee performance, with greedy search excelling in dense opportunity landscapes and broader exploration strategies performing better in sparse ones.

---

[MasFACT: Continual Multi-Agent Topology Learning via Geometry-Aware Posterior Transfer](http://arxiv.org/abs/2605.17361)

- MasFACT: introduces a geometry-aware posterior transfer framework that preserves and reuses historical collaboration knowledge as transferable topology priors to mitigate topology forgetting in LLMs-based multi-agent systems.
- The framework utilizes a Shared Prior Bank to store factorized topology atoms, which are retrieved and aligned to new tasks using FGW Alignment to balance structural stability with task-specific plasticity.
- By learning a Stochastic Topology Posterior centered on the aligned prior, MasFACT enables efficient adaptation through sparse Residual Score Matrix edits while maintaining performance on previous tasks.

---

[Learning Transferable Topology Priors for Multi-Agent LLM Collaboration Across Domains](http://arxiv.org/abs/2605.17359)

- TopoPrior: introduces a framework for learning transferable topology priors to initialize multi-agent LLM collaboration across domains, shifting the search burden from online optimization to offline prior learning.
- The framework utilizes a conditional variational graph module to capture reusable structural regularities and an adversarial latent adaptation module to minimize domain discrepancy while preserving query-specific structural information.
- By providing informative initial collaboration graphs, TopoPrior reduces online inference-time token usage and communication rounds for various topology-evolution backbones without requiring extensive additional trainable parameters.

---

[You Can’t Fool Us: Understanding the Resilience of LLM-driven Agent Communities to Misinformation](http://arxiv.org/abs/2605.17353)

- CoSim (Controlled LLM-agent simulation framework): introduces a multi-agent simulation environment to analyze how community-level psychological composition, specifically Actively Open-minded Thinking (AOT) and Political Ideology (PI), shapes resilience to misinformation through Community Construction, Misinformation Challenge, Social Interaction Simulation, and Intervention Simulation.
- The framework utilizes calibrated Agent Persona profiles and LLM Backbone models to simulate complex social dynamics, where Memory, Stance Annotator, and Intervention Operator components facilitate the study of how communities transition between misinformation uptake, questioning, and correction.
- Empirical results demonstrate that higher AOT improves both resistance and recovery, while PI composition significantly influences the recovery pathway, with Center-PI communities showing the most reliable correction and Polarized-PI communities often retaining residual support.

---

[AMATA: Adaptive Multi-Agent Trajectory Alignment for Knowledge-Intensive Question Answering](http://arxiv.org/abs/2605.17352)

- AMATA: introduces a multi-agent framework that dynamically aligns agent trajectories and inter-agent dependencies to improve factual grounding in knowledge-intensive QA.
- The framework utilizes Intra-Trajectory Preference Learning to prioritize critical agents and a dependency-aware DPO module to optimize collaborative execution sequences.
- AMATA significantly reduces token consumption and improves reasoning performance by enabling LLMs to adaptively invoke specialized agents based on question complexity.

---

[Taming “Zombie” Agents: A Markov State-Aware Framework for Resilient Multi-Agent Evolution](http://arxiv.org/abs/2605.17348)

- AgentRevive: introduces a Markov state-aware framework for resilient multi-agent evolution, utilizing State-Aware Policy Learning, State-Aware Edge Optimization, Agent Memory, Risk Estimator, Binary Node Mask, and Markov Decision Process to dynamically manage agent collaboration.
- The framework replaces hard-pruning with soft state transitions, categorizing agents into "Active", "Standby", and "Terminated" states to allow for the reactivation of temporarily unreliable "zombie" agents.
- Extensive experiments demonstrate that AgentRevive achieves superior task performance and token efficiency by balancing task-specific reasoning with adaptive agent scheduling.

---

[ASPI: Seeking Ambiguity Clarification Amplifies Prompt Injection Vulnerability in LLM Agents](http://arxiv.org/abs/2605.17324)

- ASPI (Ambiguous-State Prompt Injection): introduces a benchmark that isolates clarification as a distinct agent state to measure how ambiguity resolution increases susceptibility to prompt injection attacks.
- The framework evaluates LLMs across matched execution and clarification settings, revealing that agents are significantly more vulnerable when they solicit and incorporate external user input.
- Experimental results demonstrate that standard execution-time security evaluations systematically underestimate the attack surface of interactive agents, as clarification-seeking behavior introduces new, high-impact vulnerabilities.

---

[TClone: Low-Latency Forking of Live GUI Environments for Computer-Use Agents](http://arxiv.org/abs/2605.17320)

- TClone: introduces a versioned personal workspace system that enables low-latency forking of live GUI environments for CUAs by separating fast branch creation from durable checkpointing.
- The framework utilizes copy-on-write mechanisms for memory and filesystem state to allow parallel, isolated execution of agent trajectories without the overhead of full environment duplication.
- TClone integrates process, memory, network, and GUI subsystems into a container-based architecture to support safe, speculative, and rollback-oriented execution for LLM-based agents.

---

[DISA: Offline Importance Sampling for Distribution-Matching LLM-RL](http://arxiv.org/abs/2605.17295)

- DISA (Decoupled Importance-Sampled Anchoring): introduces an offline-then-online reinforcement learning pipeline that decouples partition-function estimation from policy learning to preserve multi-modal solution diversity.
- The framework utilizes a Proposal Model to generate trajectories for an Offline Importance Sampling Estimator, which trains a Prompt-Conditioned Regressor to provide a Frozen Partition Function for the Trajectory-Balance Objective.
- By freezing the partition function, DISA prevents estimation errors from propagating into the policy gradient, effectively maintaining the reward-tilted distribution and improving strategy-level diversity compared to standard reward-maximization methods.

---

[MetaCogAgent: A Metacognitive Multi-Agent LLM Framework with Self-Aware Task Delegation](http://arxiv.org/abs/2605.17292)

- MetaCogAgent: introduces a multi-agent LLM framework that enhances agent self-awareness through a Metacognitive Unit, enabling agents to evaluate their competence before task execution.
- The framework utilizes a Delegation Hub to reroute tasks when an agent's self-assessed confidence falls below a threshold, ensuring complex tasks are handled by the most capable agents.
- A cybernetic feedback loop continuously updates agent Capability Profiles based on performance, allowing the system to refine its competence boundaries and improve delegation accuracy over time.

---

[OProver: A Unified Framework for Agentic Formal Theorem Proving](http://arxiv.org/abs/2605.17283)

- OProver: introduces a unified framework for agentic formal theorem proving that treats proof construction as a retrieval-grounded, feedback-conditioned multi-round refinement loop.
- The framework integrates a Prover policy, a Retrieval memory, and a Lean 4 compiler to enable iterative proof repair based on compiler diagnostics and retrieved formal context.
- OProver utilizes the OProofs corpus to support a co-evolutionary training pipeline where the prover and the dataset improve iteratively through supervised fine-tuning and reinforcement learning.

---

[CONTRACTBENCH: Can LLM Agents Preserve Observation Contracts?](http://arxiv.org/abs/2605.17281)

- CONTRACTBENCH: introduces a deterministic benchmark for evaluating LLM agent compliance with observation contracts, which are tool-returned artifacts constrained by temporal validity and byte-level integrity.
- The framework utilizes a virtual clock and programmatic validators to assess agent performance across 33 tasks, revealing that contract compliance is a regression-prone capability that does not consistently scale with model size.
- The research demonstrates that structured failure labels serve as an actionable in-context reward signal, enabling improved agent performance through targeted feedback on integrity-heavy tasks.

---

[CLARA: An AI-Augmented Analytics Dashboard for Collaboration Literacy](http://arxiv.org/abs/2605.17259)

- CLARA: introduces an agentic analytics system that transforms discussion transcripts into structured semantic artifacts to support human-AI sensemaking and collaborative literacy assessment.
- The architecture utilizes an LLM to generate concept maps and collaboration assessments, which are indexed in a vector database to serve as both user-facing visualizations and retrieval infrastructure for an agentic workflow.
- Evaluation demonstrates that artifact-grounded agent responses significantly improve groundedness, analytical depth, and helpfulness compared to transcript-only baselines, while automated assessments align with expert human judgment.

---

[CAM-Bench: A Benchmark for Computational and Applied Mathematics in Lean](http://arxiv.org/abs/2605.17255)

- CAM-Bench: introduces a Lean 4 theorem-proving benchmark of 1,000 proof targets in computational and applied mathematics, utilizing an LLM-assisted pipeline for dependency recovery, modularization, and semantic validation.
- The framework employs a dependency-recovery pipeline to transform context-dependent textbook exercises into self-contained formal targets, followed by a modularization-driven Lean formalization process that uses compiler feedback and semantic review to ensure correctness.
- Evaluation of LLMs on CAM-Bench reveals significant performance gaps in library grounding, formal representation, and long-horizon proof control, which are partially mitigated by agentic proof-search strategies incorporating Lean search and execution-guided repair.

---

[SEDualVLN: A Spatially-Enhanced Dual-System for Vision-Language Navigation](http://arxiv.org/abs/2605.17249)

- SEDualVLN: introduces a dual-system architecture that combines a fast VLM-based action generator (System 1) with a slow MLLM-based waypoint planner (System 2) to improve long-horizon navigation.
- System 1 utilizes Global and Local Spatial Enhancement Strategies to improve spatial awareness, while System 2 employs a three-stage Mapping–Rendering–Reasoning pipeline for deliberative planning.
- The framework achieves state-of-the-art performance on VLN-CE benchmarks by coordinating fast low-level actions with slow, high-fidelity spatial reasoning.

---

[From Runnable Code to Shippable Applications: Test-Driven Development for Full-Stack Web Application Generation](http://arxiv.org/abs/2605.17242)

- TDDev: introduces a modular framework that automates the closed-loop Test-Driven Development process for full-stack web application generation by integrating Acceptance Test Generation, Interactive Validation, and Failure Translation.
- The framework utilizes a Coding Agent to implement applications and a LLM-backed Testing Agent to perform browser-based validation, ensuring that generated code meets functional requirements through iterative refinement.
- Empirical results demonstrate that TDDev significantly improves generation quality by 34–48 percentage points over no-TDD baselines, with the optimal enforcement protocol depending on the generation style of the underlying LLM.

---

[Bimodal Synchronization Performance: Why Noise and Sparse Connectivity Can Improve Collective Timing](http://arxiv.org/abs/2605.17206)

- Pulse-coupled oscillator model: introduces a decentralized synchronization framework where agents adjust internal clocks based on quorum-based phase acceleration to achieve collective timing.
- The framework demonstrates that high connectivity or noiseless interactions can lead to bimodal performance, where systems become trapped in stable, symmetrically phase-offset multi-cluster states.
- The research identifies that introducing noise into clock updates or reducing network connectivity breaks symmetric subgroup locking, thereby facilitating global synchronization.

---

#### 16th May 2026


[Multi-LLM Systems Exhibit Robust Semantic Collapse](http://arxiv.org/abs/2605.17193)

- Multi-LLM Systems: introduces an empirical study demonstrating that multi-LLM systems operating in closed loops exhibit robust semantic collapse, characterized by systematic convergence in semantic representations despite lexical variation.
- The research evaluates seven foundation models across 45 conditions, finding that twelve intervention strategies—including prompt design, activation steering, and reinforcement learning—fail to restore semantic diversity.
- Mechanistic analysis reveals that semantic collapse is driven by recursive self-conditioning and the recruitment of induction heads that promote historically dominant sequences, indicating fundamental constraints on open-ended knowledge production in closed-loop AI systems.

---

[Designing for Being-With: Presence Without Personhood in Conversational Human–AI Interaction](http://arxiv.org/abs/2605.17194)

- Bounded Relational Presence framework: introduces a design-oriented approach for conversational agents that prioritizes attentiveness and continuity while explicitly avoiding claims of personhood or therapeutic authority.
- The framework treats conversational presence as a tunable interactional quality composed of specific materials like pacing, memory, and exit conditions rather than a performance to be maximized.
- By emphasizing accountable withdrawal and honesty of limits, the approach provides a governance-compatible strategy for deploying LLMs in sensitive care-adjacent contexts.

---

[Personal AI, On Personal Devices](http://arxiv.org/abs/2605.17172)

- OPENJARVIS: introduces a decomposed personal AI architecture that represents a system as a typed spec over five primitives—Intelligence, Engine, Agents, Tools &amp; Memory, and Learning—to enable end-to-end optimization.
- The framework utilizes LLM-guided spec search, a local-cloud collaboration where frontier cloud models act as teachers to diagnose failures and propose coordinated edits to the local spec, which then runs entirely on-device.
- By treating the personal AI stack as an optimizable configuration object, OPENJARVIS achieves performance competitive with cloud-only models while significantly reducing marginal API costs and end-to-end latency.

---

[Responsible Agentic AI Requires Explicit Provenance](http://arxiv.org/abs/2605.17169)

- NeSy (Neuro-Symbolic) Monitoring Architecture: introduces a provenance-grounded framework for agentic AI that enables computable and actionable responsibility by integrating LLM-based agent, Tools, Persistent memory, Multi-step planning, and Self-correction into a system that produces quantifiable, traceable, and interventionable execution records.
- The framework utilizes a Neuro-symbolic monitor with an Adapter and Event abstraction to convert raw agent traces into a Finite-state automaton, allowing for real-time risk assessment and causal attribution.
- By formalizing a Responsibility tensor, the approach maps compositional agentic failures to specific deployment-chain parties, transforming accountability from a subjective concept into a structured, allocative mechanism.

---

[From Imitation to Interaction: Mastering Game of Schnapsen with Shallow Reinforcement Learning](http://arxiv.org/abs/2605.17162)

- Shallow Reinforcement Learning Framework: introduces a comparative study between supervised imitation and reinforcement learning for mastering the card game Schnapsen, utilizing MLPBot, RLBot, and RdeepBot.
- The research evaluates how shallow neural network agents perform against search-based baselines by manipulating game parameters like search depth and sampling frequency.
- Results indicate that reinforcement learning with a replay buffer significantly outperforms supervised imitation, though optimal performance requires combining learned value functions with strategic lookahead.

---

[MADP: A Multi-Agent Pipeline for Sustainable Document Processing with Human-in-the-Loop](http://arxiv.org/abs/2605.17159)

- MADP: introduces a multi-agent architecture for sustainable document processing that integrates Classificator Agent, Splitter Agent, Parser Agent, Extraction Agent, Validator Agent, PFTFI Engine, Human Reviewer, and Validation GUI to automate enterprise workflows while maintaining high accuracy through human oversight.
- The framework utilizes a PFTFI Engine to incorporate human corrections into system prompts, enabling continuous improvement without requiring model retraining.
- MADP achieves a 97.0% automation rate and 98.5% accuracy, while significantly reducing CO2 emissions, energy consumption, and water usage compared to manual processing.

---

[SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](http://arxiv.org/abs/2605.17101)

- SEMA-RAG: introduces a multi-agent framework that decouples clinical reasoning into I-Agent, E-Agent, and A-Agent to mitigate the structural deficiencies of static RAG in medical question answering.
- The framework utilizes an iterative, sufficiency-driven retrieval loop where the E-Agent dynamically updates queries based on identified evidence gaps until a converged evidence set is achieved.
- By employing role-specialized agents, SEMA-RAG improves decision-making accuracy across multiple LLMs by ensuring evidence is clinically grounded and thoroughly adjudicated before final answer selection.

---

[Can LLMs Think Like Consumers? Benchmarking Crowd-Level Reaction Reconstruction with CONSUMERSIMBENCH](http://arxiv.org/abs/2605.17079)

- CONSUMERSIMBENCH: introduces a benchmark for evaluating LLMs on their ability to reconstruct authentic crowd-level consumer reactions by decomposing tasks into auditable yes-no decisions across four reaction families.
- The framework utilizes a pointwise judging mechanism to evaluate generated comments against atomic criteria, significantly improving agreement rates compared to holistic LLM-as-Judge methods.
- Experimental results demonstrate that while frontier LLMs can produce fluent text, they struggle to anticipate specific social triggers and criticism vectors, highlighting a gap between technical performance and socially grounded consumer intuition.

---

[S-Bus: Automatic Read-Set Reconstruction for Multi-Agent LLM State Coordination](http://arxiv.org/abs/2605.17076)

- S-Bus: introduces a server-side middleware mechanism that reconstructs agent read-sets from HTTP traffic to enable optimistic concurrency control in multi-agent LLM systems.
- The framework utilizes a DeliveryLog to track observable reads and an ACP to enforce Observable-Read Isolation, preventing structural race conditions without requiring in-agent coordination code.
- S-Bus provides a topology-conditional operating envelope, ensuring safety parity with production database concurrency control backends while maintaining operational simplicity for LLM-native deployments.

---

[A Red Teaming Framework for Evaluating Robustness of AI-enabled Security Orchestration, Automation, and Response Systems](http://arxiv.org/abs/2605.17075)

- Hierarchical LLM-RL Red Teaming Framework: introduces a two-level architecture that decouples strategic planning via a frozen LLM from tactical execution via a trainable RL controller to enable robust multi-stage attack campaigns.
- The framework utilizes a perception layer to provide both natural language summaries for the LLM Strategic Planner and numeric feature vectors for the RL Tactical Controller, ensuring grounded decision-making.
- By integrating Hierarchical Reward Shaping and an optional Reflexion Memory Buffer, the agent achieves sustained long-horizon adversarial performance against adaptive blue team defenders in the CybORG environment.

---

[RAGA: Reading-And-Graph-building-Agent for Autonomous Knowledge Graph Construction and Retrieval-Augmented Generation](http://arxiv.org/abs/2605.17072)

- RAGA: introduces an LLM-based autonomous framework that integrates a "Read–Search–Verify–Construct" cognitive loop with a robust KG-vector synchronization mechanism to enable reliable, evidence-anchored knowledge graph construction.
- The framework utilizes a four-layer architecture—Tool, Reading, Memory, and Retrieval—to manage the full KG lifecycle through atomic CRUD operations and multi-hop fusion retrieval.
- By embedding cognitive constraints into a ReAct-style agent loop, RAGA addresses structural deficiencies in existing methods, such as cross-chunk relation loss and uninterpretable construction processes.

---

[EPIC-Bench: A Perception-Centric Benchmark for Fine-Grained Embodied Visual Grounding in Vision-Language Models](http://arxiv.org/abs/2605.17070)

- EPIC-Bench: introduces a comprehensive benchmark for evaluating fine-grained embodied visual grounding in VLMs across Target Localization, Navigation, and Manipulation.
- The framework utilizes a mask-grounding protocol to mitigate linguistic shortcut exploitation and provides a multi-dimensional assessment of VLM perceptual capabilities.
- Extensive evaluations of 89 VLMs reveal critical bottlenecks in multi-target counting, part-whole relationship understanding, and affordance region detection.

---

[PyraVid: Hierarchical Multimodal Memory for Long-Horizon Video Reasoning](http://arxiv.org/abs/2605.17065)

- PyraVid: introduces a hierarchical multimodal memory framework that organizes streaming video into a coarse-to-fine pyramid structure to enable structured access and evidence aggregation.
- The framework utilizes Fact Memory (fine-grained episodic observations), Clip Memory (compact local temporal summaries), and Global Memory (evolving high-level video representation) to support long-horizon reasoning.
- PyraVid employs a structure-guided reasoning mechanism that iteratively expands evidence through a Memory Graph (structured hierarchical and relational links) and uses a Pruning Agent (binary selection of relevant nodes) to reduce noise during inference.

---

[Towards Human-Level Book-Writing Capability](http://arxiv.org/abs/2605.17064)

- Hierarchical Prompt-to-Book Generation Framework: introduces a dataset construction and training pipeline that reframes supervised fine-tuning as a prompt-to-book generation task by decomposing novels into a multi-resolution planning scaffold.
- The framework utilizes a coarse-to-fine expansion process where an LLM is trained to generate a hierarchical planning scaffold—comprising book-level, chapter-level, and scene-level representations—before producing the final human-authored book text.
- To improve training stability and convergence for long-form generation, the approach incorporates a learning-rate-scaled stochastic downcasting technique when materializing FP32 master parameters into bfloat16 training copies.

---

[1GC-7RC: One Graphic Card, Seven Research Challenges! How Good Are AI Agents at Doing Your Job?](http://arxiv.org/abs/2605.17046)

- 1GC-7RC: introduces a modular, reproducible benchmark for evaluating autonomous coding agents on their ability to design, implement, and train ML models from scratch across seven diverse domains under strict time and resource constraints.
- The framework utilizes an isolated harness architecture where an AI Coding Agent interacts with a locked evaluation environment to perform tasks, with a watchdog enforcing a strict wall-clock budget on a single GPU.
- Experimental results across 245 runs reveal that proprietary LLMs generally outperform open-source alternatives, while performance is driven more by strategic architectural choices and training-detail discoveries than by the sheer number of training iterations.

---

[PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](http://arxiv.org/abs/2605.17044)

- PersonaArena: introduces a dynamic simulation framework for evaluating and improving persona-level role-playing in LLMs through multi-turn, context-rich social interactions.
- The framework utilizes a persona bank, an environment agent for scenario coordination, and a multi-agent debating judge to provide holistic and unbiased assessment of LLM role-playing capabilities.
- Experimental results demonstrate that PersonaArena effectively elicits high-quality behavioral trajectories, which can be leveraged for SFT and DPO to enhance LLM persona consistency and realism.

---

[Agentic AI Translate: An Agentic Translator Prototype for Translation as Communication Design](http://arxiv.org/abs/2605.17041)

- Agentic AI Translate: introduces an agentic translation framework that replaces text-in/text-out paradigms with a four-stage cycle of Identification, Prompting, Generation, and Verification.
- The framework utilizes an Interactive Specification layer to condition LLM agents with structured translation briefs grounded in Translation Studies metalanguage.
- Document-level coherence is maintained through a DelTA-lite memory component that stores proper-noun ledgers and bilingual summaries to guide subsequent translation chunks.

---

[Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management](http://arxiv.org/abs/2605.17036)

- GRPO (Group Relative Policy Optimization): introduces a reinforcement-learning post-training framework that trains a shared base LLM using system-level supply-chain rewards to improve agent reliability and reduce decision instability.
- The paper identifies the agent bullwhip effect, where decision instability in LLM agents is amplified across echelons and over time, and demonstrates that repeated sampling is insufficient to mitigate this structural risk.
- By leveraging GRPO to internalize coordinated replenishment policies, the framework enables autonomous agents to achieve superior cost efficiency and robustness compared to out-of-the-box LLM configurations.

---

[Skills on the Fly: Test-Time Adaptive Skill Synthesis for LLM Agents](http://arxiv.org/abs/2605.16986)

- SkillTTA: introduces a parameter-free adaptation pipeline that retrieves task-relevant trajectories and synthesizes them into a temporary, task-specific SKILL.md to guide a fixed LLM solver.
- The framework utilizes a Test-Time Skill Distiller to compress retrieved evidence into actionable procedural guidance, avoiding the overhead of parameter updates or iterative reinforcement learning.
- By leveraging both successful and failed trajectories, the method provides corrective feedback and task-specific constraints that improve performance across spreadsheet manipulation, household interaction, and code generation tasks.

---

[Securing LLM Agents Need Intent-to-Execution Integrity](http://arxiv.org/abs/2605.16976)

- Intent-to-Execution Integrity: introduces a formal correctness property for LLM agents by defining four integrity conditions—Instruction Integrity, Data Flow Integrity, Judgment Integrity, and Tool Integrity—to ensure execution faithfully reflects user intent.
- The framework identifies that modern LLM agents operate over a multi-stage pipeline analogous to compilers, where security failures arise from untrusted data ingestion and untrusted tool execution.
- By evaluating existing defenses against these four properties, the paper demonstrates that current systems provide only partial, non-compositional coverage, leaving critical gaps in securing modern agentic ecosystems.

---

[OmniVL-Guard Pro: A Tool-Augmented Agent for Omnibus Vision-Language Forensics](http://arxiv.org/abs/2605.16962)

- OmniVL-Guard Pro: introduces a tool-augmented agent that extends unified vision-language forensics from closed-world prediction to open-world evidence-driven reasoning.
- The framework utilizes Tree-Structured Self-Evolving Tool Trajectory Generation to construct high-quality training data while avoiding hindsight bias.
- It further employs Checker-Guided Agentic Reinforcement Learning to provide process-level supervision, ensuring that the LLM agent forms evidence-consistent judgments rather than relying on distorted reasoning.

---

[MORN: Metacognitive Object-Goal Regulation for Resource-Rational Long-Horizon Navigation](http://arxiv.org/abs/2605.16932)

- MORN (Metacognitive Object-goal Regulation Navigation): introduces a dual-process executive architecture that augments frozen navigation backbones with a deliberative System 2 meta-controller to enable resource-rational long-horizon navigation.
- The framework utilizes a Neuro-Cognitive State Space to monitor System 1 via an Upward Bus, computing Potentiality, Persistence, and Evidence states to dynamically regulate mission scheduling.
- By issuing high-level interventions like PERSIST, SWITCH, ABORT, and COMMIT, MORN effectively mitigates the Sunk Cost Fallacy and reduces wasted operational budget in multi-goal navigation tasks.

---

[TOBench: A Task-Oriented Omni-Modal Benchmark for Real-World Tool-Using Agents](http://arxiv.org/abs/2605.16909)

- TOBench: introduces a benchmark and evaluation harness for task-oriented omni-modal tool use, utilizing Agent, MCP Servers, Grounded Evaluator, Construction Pipeline, and Workspace Artifacts to evaluate the full perceive–act–inspect–revise loop.
- The framework evaluates LLMs on 100 executable tasks across two macro families, Customer Service and Intelligent Creation, requiring agents to coordinate tool execution with multimodal perception and iterative verification.
- Experimental results on 15 contemporary models demonstrate that TOBench remains highly challenging, with performance bottlenecks identified in reliable tool execution, multimodal reasoning, and self-verification.

---

[ArtifactLinker: Linking Scientific Artifacts for Automatic State-of-the-Art Discovery](http://arxiv.org/abs/2605.16902)

- ARTIFACTLINKER: introduces a two-stage framework for automatic SOTA discovery by modeling the HuggingFace ecosystem as an artifact graph and employing a rank-then-verify pipeline.
- The ranking stage utilizes a GNN Encoder to predict promising model-dataset links, while the verification stage employs a Self-evolving Multi-agent Loop with a Planner Agent, Executor Agent, and Memory to reproduce evaluation results.
- The framework addresses task ambiguity and scalability by formalizing discovery as a link prediction task and using a multi-agent system to overcome redundant trial-and-error in code execution.

---

[LASAR: Towards Spatio-temporal Reasoning with Latent Cognitive Map](http://arxiv.org/abs/2605.16899)

- LASAR: introduces a dual-memory embodied agent that integrates episodic memory and a structured latent cognitive map to bridge the gap between action-centric navigation and reasoning-centric question answering.
- The framework utilizes a Spatio-temporal Contextual Representation Learning (ST-CRL) objective to train the cognitive map by injecting concurrent cognitive queries into navigation tasks.
- LASAR employs a frozen LLM backbone, augmented with LoRA, to process multi-modal inputs and generate task-specific navigation actions and linguistic responses.

---

[The Alpha Illusion: Reported Alpha from LLM Trading Agents Should Not Be Treated as Deployment Evidence](http://arxiv.org/abs/2605.16895)

- P1–P6 (Minimum Reporting Protocol Suite): introduces a framework for evaluating LLM trading agents by mandating structural validity tests to distinguish between historical backtest performance and deployable trading capability.
- The paper identifies three structural mismatches—uncalibrated language confidence, narrative-numerical execution gap, and undisclosed parametric priors—that prevent current end-to-end LLM agents from serving as reliable deployment systems.
- The authors propose a conservative modular architecture where LLMs function as auditable information interfaces upstream of independent calibration, risk, and execution modules to ensure robust financial decision-making.

---

[Beyond Safety Filtering: Control Barrier Function-Informed Reinforcement Learning for Connected and Automated Vehicles](http://arxiv.org/abs/2605.16894)

- SigmaRL (Control Barrier Function-Informed Reinforcement Learning): introduces a reward design for MARL that converts CBF constraint values into safety-guiding signals to improve learning robustness and task performance.
- The framework utilizes a kinematic bicycle model to compute CBF constraint values, which are then mapped via a linear clipping function to provide per-agent safety rewards.
- By integrating CBF-informed rewards with forward-progress objectives, the approach reduces the reliance on posterior safety filters and demonstrates superior performance in multi-vehicle intersection navigation compared to heuristic baselines.

---

[SE-GA: Memory-Augmented Self-Evolution for GUI Agents](http://arxiv.org/abs/2605.16883)

- SE-GA: introduces a memory-augmented framework for GUI agents that integrates hierarchical memory structures with an iterative self-improvement mechanism to enhance multi-step task execution.
- The framework utilizes TTME to dynamically retrieve episodic-, semantic- and experiential-memories for long-term planning and MASE to stabilize the agent's policy through grounding- and self-evolution-training.
- By incorporating a Hindsight Goal-Shifting strategy and hierarchical reward design, SE-GA enables GUI agents to adapt to dynamic environments and achieve continuous self-evolution beyond static pretraining.

---

[Some[Body] Must Receive That Pain for Agent Accountability](http://arxiv.org/abs/2605.16872)

- Consequence Reception Framework: introduces a mechanistic diagnostic for AI accountability, requiring that agents possess a body capable of receiving and integrating corrective feedback signals to alter future behavior.
- The paper argues that current LLMs fail to meet these four conditions, as they are software-defined composites lacking persistent, non-fungible identities and durable internal state updates.
- It proposes that high-stakes AI deployment must remain tethered to human principals who exercise meaningful control, proportional liability, and authority to terminate the agent until coupling-capable architectures are developed.

---

[GoodServe: Towards High-Goodput Serving of Agentic LLM Inferences over Heterogeneous Resources](http://arxiv.org/abs/2605.16867)

- GoodServe: introduces a predict-and-rectify routing system for agentic LLM inferences that optimizes goodput across heterogeneous GPU resources by leveraging a RequestStatusMonitor, a GPUStatusMonitor, an MoE-style predictor, a just-enough instance selection heuristic, and a runtime request migration mechanism.
- The system utilizes an EMA-smoothed, black-box profiling method to estimate GPU execution efficiency and employs token-ID based migration to efficiently re-route requests when SLO-violation risks are detected.
- By prioritizing requests based on end-to-end latency requirements and dynamically adjusting routing decisions, GoodServe achieves higher goodput compared to traditional hardware-aware routing strategies.

---

#### 15th May 2026


[paper.json: A Coordination Convention for LLM-Agent-Actionable Papers](http://arxiv.org/abs/2605.16194)

- paper.json: introduces a lightweight coordination convention for academic papers that enables LLMs to accurately parse sub-claims, definitions, and reproducibility commands via a companion JSON file.
- The framework utilizes stable identifiers for claims, definitions, and theorems, alongside explicit non-claim sections to mitigate common LLM hallucination and scope-overextension errors.
- By providing exact shell commands for figure generation and a machine-readable read-receipt protocol, the convention shifts reproducibility from a codebase property to a verifiable paper property.

---


[ColPackAgent: Agent-Skill-Guided Hard-Particle Monte Carlo Workflows for Colloidal Packing](http://arxiv.org/abs/2605.15625)

- ColPackAgent: introduces an agent framework that autonomously executes structured hard-particle Monte Carlo simulation workflows for colloidal packing by separating domain-specific simulation tools from procedural agent skills.
- The framework utilizes the Model Context Protocol to expose simulation operations as schema-validated tool calls, ensuring reliable execution across different LLMs and agent platforms.
- ColPackAgent supports interactive, autonomous, and autoresearch modes, demonstrating robust performance in mapping phase transitions and providing a stage-aware benchmark for evaluating LLM reliability in scientific workflows.

---


[Task-Semantic Graph-Driven Distributed Agent Networking for Underwater Target Tracking](http://arxiv.org/abs/2605.15528)

- STG-MAPPO: introduces a semantic task graph-driven framework that integrates task-level diagnostics and communication-aware neighbor summaries to enable stable decentralized target tracking for AUV swarms.
- The framework utilizes a velocity-level action interface to map high-level cooperative decisions to executable six-degree-of-freedom AUV control inputs, effectively reducing the complexity of low-level force/torque exploration.
- By encoding task phases, observation confidence, and link availability into a compact semantic graph, the approach ensures persistent target tracking and robust performance under communication-constrained underwater conditions.

---


[A Generative-AI Framework for Intelligent Utility Billing, CO₂ Analytics and Sustainable Resource Optimisation](http://arxiv.org/abs/2605.16250)

- Generative-AI Utility-Billing Framework: integrates Data Acquisition, Preprocessing &amp; Feature Engineering, Generative-AI Bill Generation Agent, Transformer Consumption Forecaster, CO₂ Estimator, Simulated-Bifurcation Tariff &amp; Load Optimiser, and Reporting to automate utility billing and resource management.
- The framework utilizes a transformer-based forecaster for consumption estimation and a quantum-inspired Simulated-Bifurcation solver to optimize demand-response schedules on classical hardware.
- A constrained-decoding policy and post-generation auditor are implemented within the LLM-based bill generation agent to ensure factual consistency and prevent numeric hallucinations in customer communications.

---


[Agentic AI and Human-in-the-Loop Interventions: Field Experimental Evidence from Alibaba’s Customer Service Operations](http://arxiv.org/abs/2605.14830)

- Agentic AI and Human-in-the-Loop Interventions: introduces a field experimental study on Alibaba's platform to evaluate how human-in-the-loop interventions shape service outcomes when Agentic AI (autonomous service task executor) encounters cognitive or emotional failures.
- The system integrates an Agentic AI (autonomous service task executor) with Human Supervisor (human-in-the-loop intervention agent) roles, supported by Monitoring Algorithms (risk detection tools) including Sentiment Monitoring (customer frustration detection) and Intention Monitoring (customer intent shift tracking).
- The research demonstrates that while Agentic AI (autonomous service task executor) improves service speed, human intervention effectiveness is highly dependent on the timing and nature of the failure, with emotional escalations proving significantly harder to recover due to reduced human engagement.

---

[FORGE: Self-Evolving Agent Memory With No Weight Updates via Population Broadcast](http://arxiv.org/abs/2605.16233)

- FORGE: introduces a staged, population-based protocol that evolves prompt-injected natural-language memory for hierarchical ReAct agents without requiring gradient updates.
- The framework utilizes an inner Reflexion-style loop to synthesize knowledge artifacts from failed trajectories, which are then propagated across parallel instances via champion broadcast and stabilized through graduation-based early stopping.
- FORGE improves decision-making in stochastic, long-horizon cyber-defense environments by reducing major failure rates and compressing performance variance across diverse LLM families.

---

[Argus: Evidence Assembly for Scalable Deep Research Agents](http://arxiv.org/abs/2605.16217)

- Argus: introduces a multi-agent system that utilizes a Searcher and a Navigator to perform deep research by assembling evidence into a structured directed acyclic graph.
- The Navigator identifies missing information and dispatches Searchers to target specific gaps, enabling efficient evidence gathering and reducing redundant parallel rollouts.
- By decoupling the Navigator's reasoning context from the number of Searchers, the framework achieves scalable performance and provides fully auditable, source-traced answers.

---

[Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle Where Feedback Matters Most](http://arxiv.org/abs/2605.16207)

- KG-grounded LLM Tutoring Framework: introduces a benchmark for evaluating LLM-based tutoring agents in propositional logic using a knowledge graph to provide ground truth for three-way diagnosis of student solutions.
- The framework utilizes Student Simulator Agents, Peer Feedback Agents, Teacher Feedback Agents, and Judge Feedback Agents to analyze diagnostic precision and pedagogical feedback quality across different information-access conditions.
- The research demonstrates that while LLMs reliably confirm optimal steps, they struggle with valid-alternative and incorrect solutions, highlighting the necessity of hybrid architectures that integrate KG-grounded diagnostic mechanisms with LLM-based scaffolding.

---

[Context, Reasoning, and Hierarchy: A Cost-Performance Study of Compound LLM Agent Design in an Adversarial POMDP](http://arxiv.org/abs/2605.16205)

- Compound LLM Agent Design Framework: introduces a controlled empirical study of compound LLM agent design in an adversarial POMDP, evaluating the interactions between context engineering, deliberation, and hierarchical decomposition.
- The study reveals that programmatic state abstraction provides the most cost-effective performance gains, while distributing deliberation tools across a hierarchy triggers a destructive deliberation cascade that degrades performance and inflates token costs.
- The research establishes that bounded hierarchical decomposition without distributed deliberation achieves the best absolute performance, emphasizing that system-level information flow is more critical than individual agent reasoning depth.

---

[Formal Methods Meet LLMs: Auditing, Monitoring, and Intervention for Compliance of Advanced AI Systems](http://arxiv.org/abs/2605.16198)

- TRAC (Temporal Rule Assessment and Compliance): introduces a framework for auditing and monitoring LLM-based systems by combining formal LTL specifications with machine learning-based labeling and predictive intervention.
- The framework utilizes LTL progression to evaluate temporally extended behavioral constraints, enabling real-time monitoring and retrospective auditing of black-box AI systems.
- Experimental results demonstrate that TRAC significantly outperforms standalone LLM-as-a-Judge methods in detecting violations, while predictive monitors effectively reduce violation rates without sacrificing task performance.

---

[Optimized Three-Dimensional Photovoltaic Structures with LLM guided Tree Search](http://arxiv.org/abs/2605.16191)

- ERA (Empirical Research Assistance): introduces an iterative framework combining a coding agent and LLM-driven tree search to optimize three-dimensional photovoltaic structures while mitigating algorithmic reward hacking.
- The system utilizes AntiGravity to patch the physics engine and scoring function, ensuring that generated designs adhere to physical constraints like structural connectivity and occlusion accuracy.
- By employing BFS-based validation and iterative refinement, the framework successfully discovers high-efficiency solar geometries that outperform human-designed baselines under various material constraints.

---

[An Algebraic Exposition of the Theory of Dyadic Morality](http://arxiv.org/abs/2605.16153)

- TDM (Theory of Dyadic Morality): introduces an algebraic formalization of moral judgment using structural causal modeling to represent human cognition as a two-node template involving an intentional agent and a vulnerable patient.
- The framework incorporates psychological operators including typecasting, completion, and valence-dependent inference to model how humans compute moral judgments under constraints and handle multi-node scalability through node collapse and sequential processing.
- This approach enables neurosymbolic AI systems to perform computationally rigorous moral reasoning by reframing safety policies from fixed enumeration to patient-centric protection against suffering.

---

[Look Before You Leap: Autonomous Exploration for LLM Agents](http://arxiv.org/abs/2605.16143)

- Explore-then-Act paradigm: introduces a training and inference framework that mitigates premature exploitation in LLMs by explicitly optimizing for autonomous environment exploration using verifiable rewards.
- The framework utilizes ECC to quantify the discovery of environment states, objects, and affordances, which are then used to guide an interleaved GRPO training process.
- By decoupling information-gathering from task execution, the agent builds a grounded knowledge summary that significantly improves downstream performance and robustness in unfamiliar environments.

---

[Surrogate Neural Architecture Codesign Package (SNAC-Pack)](http://arxiv.org/abs/2605.16138)

- SNAC-Pack: introduces an open-source AutoML framework for hardware-aware neural architecture codesign and end-to-end FPGA deployment using Optuna, NSGA-II, and hardware surrogate models.
- The framework integrates a global search loop with hardware surrogate models for resource and latency estimation, followed by a local search stage utilizing QAT and iterative magnitude pruning for FPGA synthesis via hls4ml.
- SNAC-Pack supports an optional MCP agentic frontend to automate the pipeline, significantly reducing design space exploration time for resource-constrained tasks like jet classification and qubit readout.

---

[ShopGym: An Integrated Framework for Realistic Simulation and Scalable Benchmarking of E-Commerce Web Agents](http://arxiv.org/abs/2605.16116)

- ShopGym: introduces an integrated framework for realistic simulation and scalable benchmarking of e-commerce web agents, utilizing ShopArena (simulation environment generation pipeline), ShopGuru (grounded task generation pipeline), Planning Agent (decomposes exploration into subtasks), Specification Agent (writes anonymized design specifications), Execution Agent (edits codebase in verification loop), Verifiers (rule-based and multimodal validation), and LLM-as-Judge (evaluates agent trajectory success).
- The framework employs a multi-agent pipeline to transform live seed storefronts into self-contained, inspectable, and resettable sandbox environments while synthesizing grounded tasks for evaluation.
- Experimental results demonstrate that synthetic shops preserve key structural properties of live storefronts, with agent performance on synthetic environments positively correlated with performance on live sites.

---

[Multi-Agent Cooperative Transportation: Optimal and Efficient Task Allocation and Path Finding](http://arxiv.org/abs/2605.16097)

- CT-TCBS (Cooperative Transportation Task Conflict-Based Search): introduces a two-level search framework for solving the Cooperative Transportation Task Allocation and Path Finding (CT-TAPF) problem by integrating High-Level Search, Low-Level Pathfinding, Incremental Expansion, Conflict Expansion, Task Expansion, Heuristic Function, and Task Selector Layer.
- The framework utilizes an Incremental Expansion strategy to manage the combinatorial explosion of team formation by breaking the assignment process into prioritized steps.
- Suboptimal variants employ a global Task Selector Layer using Best-Task or Worst-Task heuristics to establish a more efficient runtime-quality frontier compared to agent-centric baselines.

---

[VideoSeeker: Incentivizing Instance-level Video Understanding via Native Agentic Tool Invocation](http://arxiv.org/abs/2605.16079)

- VideoSeeker: introduces an agentic paradigm for instance-level video understanding that integrates proactive perception and tool invocation to overcome the limitations of text-only prompts.
- The framework utilizes a four-stage automated data synthesis pipeline and a two-stage training strategy, including SFT and agentic RL, to internalize tool-calling capabilities into the base model.
- By employing perception tools like view_visual_prompt and crop_video, the model achieves precise spatiotemporal localization and reasoning, significantly outperforming existing baselines on instance-level video understanding tasks.

---

[Ada-Diffuser: Latent-Aware Adaptive Diffusion for Decision-Making](http://arxiv.org/abs/2605.16054)

- Ada-Diffuser: introduces a unified generative framework that explicitly incorporates latent dynamic inference into decision-making using a Latent Factor Identification Block, Causal Diffusion Model, Autoregressive Denoising Schedule, Denoise-and-Refine Mechanism, Zig-Zag Sampling, and Inverse Dynamics Model.
- The framework leverages theoretical identifiability results to perform block-wise latent inference from minimal temporal observations, enabling adaptive planning and policy learning in partially observable environments.
- Ada-Diffuser employs a denoise-and-refine mechanism and zig-zag sampling to reduce posterior mismatch, ensuring high-quality latent context recovery and improved performance across diverse locomotion and robotic manipulation benchmarks.

---

[RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents](http://arxiv.org/abs/2605.16045)

- RecMem: introduces a three-tier memory architecture that reduces LLM token consumption by deferring memory consolidation until sustained interaction recurrence is observed.
- The framework utilizes a subconscious memory layer for lightweight storage, an episodic memory for event-level narratives, and a semantic memory for fine-grained facts, all managed through a recurrence-driven trigger.
- RecMem incorporates a semantic refinement mechanism to recover critical details omitted during episodic abstraction, ensuring high accuracy while significantly lowering the computational cost of long-running LLM agents.

---

[Who Owns This Agent? Tracing AI Agents Back to Their Owners](http://arxiv.org/abs/2605.16035)

- Agent Attribution Framework: introduces a canary-based protocol to link harmful agent interactions to the responsible operator account at a vendor-hosted LLM.
- The protocol utilizes Lexical Canary and Semantic Canary constructions to bridge the visibility gap between external agent behavior and vendor-side session logs.
- By leveraging a utility-evasion tradeoff, the framework ensures that adversarial attempts to suppress canaries inherently degrade the agent's task performance.

---

[ScreenSearch: Uncertainty-Aware OS Exploration](http://arxiv.org/abs/2605.16024)

- ScreenSearch: introduces an ambiguity-aware desktop exploration system that combines structural screen retrieval, deduplication, and a PUCT graph-bandit to navigate partial observability in OS environments.
- The framework utilizes UIA Tree, Screen Featurizer, Screen Index &amp; Embedding Store, Global MCTS State Store, Trajectory &amp; Artifact Store, PUCT Graph-Bandit, and Worker Pool to manage state identity and drive exploration through frontier expansion and ambiguity reduction.
- By treating GUI exploration as a search problem over a shared deduplicated state graph, the system effectively mitigates premature commitment in visually aliased desktop states.

---

[Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning](http://arxiv.org/abs/2605.15975)

- BISON: introduces a bilevel policy framework that combines symbolic high-level reasoning with neural low-level execution to solve long-horizon planning problems.
- The framework utilizes goal regression and inductive generalisation to derive interpretable, first-order symbolic HL policies from demonstrations, which are then realized by a compact GNN-based LL policy.
- BISON demonstrates superior generalization to long horizons and large numbers of objects compared to end-to-end and VLA baselines while maintaining high training and inference efficiency.

---

[OHP-RL: Online Human Preference as Guidance in Reinforcement Learning for Robot Manipulation](http://arxiv.org/abs/2605.15971)

- OHP-RL: introduces a human-in-the-loop reinforcement learning framework that interprets human interventions as online preference signals to guide policy learning through a state-dependent preference gate.
- The framework utilizes an asynchronous actor-critic architecture with four distinct update modules to balance environment rewards with human-provided preference guidance.
- By adaptively regulating preference influence based on state-dependent advantages, OHP-RL improves sample efficiency and robustness in real-world robotic manipulation tasks.

---

[Deterministic Event-Graph Substrates as World Models for Counterfactual Reasoning](http://arxiv.org/abs/2605.15967)

- Event-Graph Substrate: introduces a world model architecture that represents agent memory as an append-only log of typed RDF triples to enable deterministic counterfactual reasoning.
- The framework utilizes a TBox, ABox, Typed Event Log, Deterministic Interpreter, and Intervention Vocabulary to perform causal-ancestor traversals and kinematic projections without learned components.
- This approach provides formal guarantees on inspectability and replay consistency, outperforming symbolic and parametric baselines on causal-reasoning benchmarks by leveraging structured execution over observed event logs.

---

[PAGER: Bridging the Semantic-Execution Gap in Point-Precise Geometric GUI Control](http://arxiv.org/abs/2605.15963)

- PAGER (Precision-Aware GEometric Reasoning): introduces a topology-aware agent that decomposes geometric construction into dependency-structured Planning Module, Task Execution Module, and precision-aligned RL Precision Optimization.
- The framework utilizes Pixel-Precise Data Construction and SFT Instruction Tuning to establish executable action grammar, mitigating the Semantic-Execution Gap in point-precise GUI tasks.
- PAGER incorporates a GeoGebra Action Executor and RL-based parameter accuracy rewards to ensure point-level spatial precision and robustness against cascading coordinate errors.

---

[PersonaFingerprint: Measuring Persona Inference on Modern Websites with LLM-Driven Browsing](http://arxiv.org/abs/2605.15962)

- PersonaFingerprint: introduces a multi-agent framework that leverages a persona-conditioned decision agent and a computer-use agent to generate labeled encrypted traffic for measuring persona inference risks.
- The framework utilizes a shared packet-window encoder with specialized heads to perform both website and persona fingerprinting, demonstrating that behavioral personas are learnable from encrypted metadata.
- The study reveals that existing website fingerprinting models contain incidental persona leakage, which can be amplified through joint multi-task learning or lightweight MLP probes.

---

[Dynamic Plasma Shape Control with Arbitrary Sensor Subsets](http://arxiv.org/abs/2605.15935)

- RL-based plasma shape control framework: introduces a reinforcement learning agent that achieves robust, real-time plasma shape control in tokamaks by utilizing an asymmetric actor-critic architecture and an auxiliary shape reconstruction head to handle partial observability and diagnostic failures.
- The agent employs diagnostic dropout during training to generalize across arbitrary sensor subsets without requiring explicit fault detection or controller switching.
- Experimental validation on the DIII-D tokamak demonstrates that the policy successfully commands coil actuators for dynamic shape maneuvers while maintaining performance comparable to classical controllers.

---

[Privacy is Fungibility: Why Endogenous Tokens Are Not Money](http://arxiv.org/abs/2605.15934)

- Token Security and Trust Locus Classification: introduces a framework to categorize blockchain assets based on whether their security and trust models are intrinsic to the ledger or derived from external institutions.
- The paper argues that most public, permissionless blockchain tokens function as credit rather than money due to their account-based nature, public visibility, and lack of obliviousness.
- By extending economic models of theft and credit, the authors demonstrate that the absence of privacy in current ledger designs makes these assets fundamentally incompatible with the definition of cash-like money.

---

[Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design](http://arxiv.org/abs/2605.15871)

- AIRA: introduces a dual-framework approach for autonomous neural architecture discovery using LLM-agents to navigate combinatorial design spaces and engineer mechanistic implementations.
- AIRA-Compose utilizes an ensemble of LLM-agents to search and optimize arrangements of computational primitives, while AIRA-Design tasks agents with writing novel attention mechanisms and training scripts from scratch.
- Both frameworks leverage the AIRS-Bench task standard and AIRA-dojo harness to enable recursive self-improvement, consistently outperforming hand-designed baselines and traditional NAS-found models.

---

[Access Timing as Scaffolding: A Reinforcement Learning Approach to GenAI in Education](http://arxiv.org/abs/2605.15850)

- RL-based GenAI Access Timing Framework: introduces a reinforcement learning agent that optimizes the timing of GenAI access to balance cognitive support with independent learning.
- The framework utilizes PPO and BKT to implement an adaptive policy that rewards task success, efficiency, metacognitive reflection, productive failure, and cognitive load management.
- Experimental results demonstrate that strategically timed GenAI access improves objective post-test performance and metacognitive accuracy compared to unrestricted access, while reducing task errors and time on task.

---

[ROADMAPBENCH: Evaluating Long-Horizon Agentic Software Development Across Version Upgrades](http://arxiv.org/abs/2605.15846)

- RoadmapBench: introduces a benchmark of 115 long-horizon coding tasks grounded in real open-source version upgrades, utilizing a Docker environment, repository snapshot, roadmap instruction, agent execution loop, agent code verification, static validation, and rollout-based quality control.
- The benchmark evaluates LLMs on multi-target software evolution, requiring agents to implement functionality across multiple files and modules with a median modification of 3,700 lines.
- Evaluation results demonstrate that even the strongest frontier models struggle with long-horizon development, with performance bottlenecks shifting from build errors in weaker models to implementation precision in stronger models.

---

[WorldAct: Activating Monolithic 3D Worlds into Interactive-Ready Object-Centric Scenes](http://arxiv.org/abs/2605.15843)

- WorldAct: introduces a framework that converts static, monolithic 3D Gaussian Splatting scenes into editable, interaction-ready environments by leveraging a Vision-Language Agent, SAM3, DiffuEraser, DepthLab, Poisson Reconstruction, SAM3D, ICP, and a Differentiable Renderer.
- The framework utilizes a Vision-Language Agent to automate object discovery and viewpoint selection, enabling the decomposition of scenes into independent object assets and a restored background.
- WorldAct supports downstream embodied AI tasks by generating collision-aware geometry and high-quality object assets, facilitating object-level editing, manipulation, and scene rearrangement.

---

[Exploration of k-edge-deficient temporal graphs in linear time](http://arxiv.org/abs/2605.15833)

- Temporal Exploration Framework: introduces a method for exploring k-edge-deficient temporal graphs in O(nk log k) time by reducing the problem to covering a depth-first search tour of a stable spanning tree.
- The framework utilizes a roundabout process to eliminate redundant virtual agents, ensuring that a small set of representative agents covers the entire graph structure efficiently.
- By leveraging Δ-temporal connectivity, the approach enables a single explorer to simulate the movements of these representative agents, achieving near-optimal linear-time performance for constant k.

---

[BootstrapAgent: Distilling Repository Setup into Reusable Agent Knowledge](http://arxiv.org/abs/2605.15815)

- BootstrapAgent: introduces a multi-agent framework that distills repository bootstrapping exploration into a persistent, verifiable, and agent-consumable .bootstrap contract.
- The framework utilizes a Discoverer Agent, Planner Agent, and Generator Agent to automate environment setup, while the Docker Verifier ensures reproducibility through clean replay and trace-driven Delta Repair.
- By persisting setup knowledge, BootstrapAgent significantly reduces downstream LLM token usage and build time for unfamiliar repositories.

---

[Toward Natural and Companionable Virtual Agents via Cross-Temporal Emotional Modeling](http://arxiv.org/abs/2605.15812)

- CTEM (Cross-Temporal Emotional Modeling): introduces a framework that links long-term behavioral history to moment-to-moment emotional expression through a closed-loop system of Psychologically-grounded Behavior Modeling, Episodic State, and Multi-Modal Message Generation.
- The framework utilizes a Physio-emotional State, Motivational Vector, and Memory to maintain an Adapted Personality that dynamically influences future behaviors and interaction styles.
- Auri, the companion agent instance, incorporates a Safety Layer with Dual Stage Detection and LLM Safety Guardrails to ensure ethical interaction while balancing stability and flexibility in long-term companionship.

---

[From Gridworlds to Warehouses: Adapting Lightweight One-shot Multi-Agent Pathfinding for AGVs](http://arxiv.org/abs/2605.15799)

- MAWPF: introduces a gridworld-based pathfinding formulation that incorporates differential-drive AGV kinodynamic constraints, including multi-step rotations, acceleration, deceleration, and follower collision avoidance.
- The framework adapts lightweight MAPF solvers—PP, LNS2, PIBT, and LaCAM—to the MAWPF environment using a rolling-horizon integration and a PIBT-based configuration generator to handle complex kinodynamic states.
- Empirical results demonstrate that the LaCAM+PIBT pipeline achieves high success rates and real-time performance for hundreds of agents, while the follower-collision constraint serves as an implicit congestion-avoidance mechanism.

---

[SaaS-Bench: Can Computer-Use Agents Leverage Real-World SaaS to Solve Professional Workflows?](http://arxiv.org/abs/2605.15777)

- SaaS-Bench: introduces a benchmark for evaluating Computer-Using Agents (CUAs) in realistic, deployable SaaS environments, utilizing Task Input, Agent, SaaS Apps, Deployment, Database, Execute, Browser Use, and Verify components to measure end-to-end workflow completion.
- The framework employs State-Check, Content-Check, and LLM-Judge to provide a systematic, multi-dimensional evaluation of agent performance across long-horizon, cross-application professional tasks.
- Experimental results demonstrate that current LLMs struggle with long-horizon execution, state tracking, and error recovery, with fewer than 4% of tasks completed end-to-end.

---

[Lamarckian Inheritance in Dynamic Environments: How Key Variables Affect Evolutionary Dynamics](http://arxiv.org/abs/2605.15769)

- Lamarckian Inheritance in Dynamic Environments: introduces a framework for co-optimizing robot morphology and control, utilizing Evolutionary Algorithm, Bayesian Optimization, Reinforcement Learning, Modular ANN-based Controller, Critic Network, and Direction Sensor.
- The study demonstrates that the efficacy of Lamarckian inheritance in dynamic environments is contingent upon the degree of environmental conflict and the predictability of environmental changes.
- The research finds that integrating a directional sensor restores the performance benefits of Lamarckian inheritance in conflicting environments by enabling agents to generalize control strategies.

---

[ALSO: Adversarial Online Strategy Optimization for Social Agents](http://arxiv.org/abs/2605.15768)

- ALSO: introduces an adversarial online strategy optimization framework for LLM-based social agents that dynamically adapts behavior through strategy selection without requiring offline retraining.
- The framework formulates multi-turn social interaction as an adversarial bandit problem, utilizing a lightweight neural surrogate to predict rewards and generalize feedback across semantically related strategies.
- By combining randomized bandit-based exploration with context-conditioned value estimation, the system enables robust, sample-efficient adaptation to non-stationary, co-evolving agent behaviors in social simulations.

---

[BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks](http://arxiv.org/abs/2605.15766)

- BioXArena: introduces a biomedical machine learning coding benchmark that evaluates whether LLM agents can create task-specific model-building code for heterogeneous, multi-modal biomedical datasets.
- The framework utilizes Expert-Driven Curation, Unified Public Task Capsules, Hidden Private Labels, Sandbox Runtime, Evaluation Metrics, and Analysis Output to assess agent performance across 76 tasks in 9 biomedical domains.
- BioXArena evaluates 11 agent configurations, including general coding LLMs, biomedical agents, and ML coding agents, to characterize performance under realistic computational constraints.

---

[Attribute-Grounded Selective Reasoning for Artwork Emotion Understanding with Multimodal Large Language Models](http://arxiv.org/abs/2605.15755)

- FAB-G (Formal-Attribute Bottleneck-Guided reasoning): introduces a supervised multi-agent framework that addresses attribute flooding in LLMs by decomposing artwork emotion understanding into attribute salience screening and cue-constrained emotional reasoning.
- The framework utilizes five specialized attribute agents to identify emotionally operative formal cues, which are then aggregated into a bottleneck to guide the final analysis agent's interpretation.
- By grounding explanations in human-salient attributes, FAB-G improves prediction accuracy, enhances explanation auditability, and produces more compact outputs compared to unconstrained LLM baselines.

---

[SMMBench: A Benchmark for Source-Distributed Multimodal Agent Memory](http://arxiv.org/abs/2605.15710)

- SMMBench: introduces a benchmark for evaluating whether agents can retrieve, align, and compose multimodal evidence scattered across independently originated sources rather than reasoning within a single curated context.
- The framework utilizes a Dataset Construction Pipeline to synthesize long-horizon conversational environments where answer-critical evidence is distributed across heterogeneous artifacts like chats, tables, and documents.
- Experimental results demonstrate that current LLMs struggle with source-distributed memory composition, particularly in conflict resolution, preference reasoning, and precise function calling.

---

[Differentiable Mixture-of-Agents Incentivizes Swarm Intelligence of Large Language Models](http://arxiv.org/abs/2605.15706)

- DMoA: introduces a self-evolving multi-agent framework that dynamically routes and activates agents at each reasoning step using a differentiable, context-aware mechanism.
- The framework utilizes a Sentence Transformer and RNN-Router to produce sparse agent activations, optimized via predictive entropy as a self-supervised signal.
- DMoA enables elastic, adaptive collaboration during inference, resolving static compilation limitations found in traditional multi-agent systems.

---

[H-MEM: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure](http://arxiv.org/abs/2605.15701)

- H-MEM: introduces a hybrid memory mechanism that couples a temporal-semantic tree with a knowledge graph to model memory evolution and support multi-hop reasoning.
- The framework utilizes a Retrieval Planner to decompose queries, an Evidence Retriever to extract information from the hybrid structure, and a Generation Process to synthesize final answers.
- H-MEM achieves state-of-the-art performance on long-term memory benchmarks by enabling progressive consolidation of short-term memory into long-term summaries while maintaining entity-level relational dependencies.

---

[Distributed Zeroth-Order Policy Gradient for Networked Multi-agent Reinforcement Learning from Human Feedback](http://arxiv.org/abs/2605.15697)

- DZOPG introduces a distributed reinforcement learning framework that optimizes policies in networked multi-agent systems using human preference feedback instead of explicit reward signals.
- The framework utilizes spatiotemporally truncated trajectories and Gaussian-perturbed policy gradients to enable fully distributed learning under local communication constraints.
- Theoretical analysis establishes that the algorithm achieves polynomial sample complexity and converges to an ϵ-stationary point in infinite-horizon networked multi-agent settings.

---

[Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation](http://arxiv.org/abs/2605.15669)

- Rule2DRC: introduces a large-scale benchmark for evaluating LLM agents in synthesizing executable DRC scripts from natural language rules using execution-based scoring.
- SplitTester: improves script selection by iteratively generating discriminative test layouts to cluster candidate scripts and identify the most functionally correct implementation.
- The framework leverages execution feedback from a DRC engine to guide LLM agents, effectively bypassing the limitations of surface-level code similarity metrics.

---

[PRISM: Prompt Reliability via Iterative Simulation and Monitoring for Enterprise Conversational AI](http://arxiv.org/abs/2605.15665)

- PRISM: introduces a closed-loop framework for continuous prompt reliability in enterprise conversational AI by integrating Test Generator, Platform Simulator, LLM-as-Judge, Diagnosis &amp; Repair, and Continuous Monitoring.
- The framework treats prompt engineering as a continuous reliability problem, using automated simulation and surgical repair to address both creation-time correctness and runtime LLM behavioral drift.
- PRISM achieves 99% production reliability and reduces prompt authoring time by 98% by automating the detection and repair of procedural compliance failures in multi-step conversational agents.

---

[PCASim: Promptable Closed-loop Adversarial Simulation for Urban Traffic Environment](http://arxiv.org/abs/2605.15654)

- PCASim: introduces a closed-loop simulation framework that leverages RAG-Augmented LLM, Adversarial Scenario Repository, and PPO-based Reinforcement Learning to generate and evaluate safety-critical urban traffic scenarios.
- The framework utilizes a Middleware to translate natural language descriptions into executable DSL, which is then refined by Bézier Curve Convex Optimization to ensure trajectory realism.
- By employing a Semantic Alignment Module and Self-Consistency Voting Mechanism, the system achieves high-fidelity scenario generation, enabling robust training of autonomous agents against adversarial behaviors.

---

[TopoEvo: A Topology-Aware Self-Evolving Multi-Agent Framework for Root Cause Analysis in Microservices](http://arxiv.org/abs/2605.15611)

- TopoEvo: introduces a topology-aware, reasoning-enhanced, and self-evolving framework for joint microservice root cause localization and fault type classification.
- The framework utilizes Metric-Anchored Orthogonal Multimodal Alignment and Vector Quantization to transform noisy telemetry into structured, auditable symptom tokens for reliable reasoning.
- TopoEvo employs a multi-agent Hypothesis–Evidence–Test workflow to mitigate symptom-amplification bias and incorporates a self-evolving mechanism to maintain robustness under non-stationary microservice conditions.

---

[See Before You Code: Learning Visual Priors for Spatially Aware Educational Animation Generation](http://arxiv.org/abs/2605.15585)

- OmniManim: introduces a render-feedback-aware framework that utilizes a Shared Scene State, Scene Agent, Vision Agent, Code Agent, and Repair Agent to generate high-quality educational animations.
- The framework employs a Vision Agent to predict sparse keyframe layouts using coarse-to-fine bounding-box denoising and interpolation-aware objectives to ensure visual stability.
- OmniManim incorporates a structured render-feedback loop that enables iterative refinement of animations based on explicit visual quality constraints rather than code-level properties alone.

---

[STAR: A Stage-attributed Triage and Repair framework for RCA Agents in Microservices](http://arxiv.org/abs/2605.15581)

- STAR (Stage-attributed Triage And Repair): introduces a process-centric reliability layer for LLM-based RCA agents that decomposes reasoning into four structured stages to enable targeted debugging and repair.
- The framework utilizes Stage-wise Audit and Diagnosis, Fast/Slow Routing, Decisive Stage Localization, Patch-and-Replay Repair, and Self-Evolving Repair Memory to systematically eliminate error propagation in microservice RCA workflows.
- By treating agent failures as stage-localizable bugs rather than monolithic errors, STAR significantly improves root cause localization and fault type classification accuracy across diverse LLM backbones.

---

[Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems](http://arxiv.org/abs/2605.15573)

- NEXA (Response-Conditioned Parallel-to-Sequential Orchestration): introduces a hybrid multi-agent framework that uses a Draft Generation Stage to produce initial responses, a Semantic Embedding Encoder to represent them, a Response-Conditioned Transformer Policy to predict a sparse communication DAG, a Sequential Propagation Module to refine responses, and a Weighted-Centroid Aggregator to select the final output.
- The framework bridges parallel and sequential execution by using the initial parallel draft as evidence to decide whether structured sequential refinement is necessary.
- NEXA achieves improved accuracy-cost tradeoffs and generalizability across tasks, agent counts, and model scales by learning a sparse, judge-free communication policy.

---

[Detecting Privilege Escalation in Polyglot Microservices via Agentic Program Analysis](http://arxiv.org/abs/2605.15569)

- NEO: introduces an agentic program analysis framework that combines LLM-based semantic reasoning with classic program analysis to detect privilege escalation vulnerabilities in polyglot microservices.
- The framework utilizes a set of unified code search primitives to enable scalable, language-agnostic context retrieval and cross-service data flow analysis.
- NEO validates potential vulnerabilities by combining LLM-based semantic assessment with SMT-based path constraint solving to minimize false positives.

---

[AstraFlow: Dataflow-Oriented Reinforcement Learning for Agentic LLMs](http://arxiv.org/abs/2605.15565)

- AstraFlow: introduces a dataflow-oriented RL system that replaces trainer-centered control with decoupled autonomous components to support complex multi-policy agentic workloads.
- The framework utilizes a Dataflow Layer for coordination, RaaS for scalable trajectory generation, and independent Trainers to enable elastic, heterogeneous, and cross-region training.
- AstraFlow achieves significant speedups in multi-policy collaborative training by enabling fully asynchronous execution and efficient sparse weight updates across distributed compute resources.

---

[TopoClaw: A Human-Centric and Topology-Aware Agent Operating System](http://arxiv.org/abs/2605.15556)

- TopoClaw: introduces a human-centric Agent OS that replaces agent-centric isolation with a decoupled runtime navigating physical device and social relationship topologies, utilizing Core Runtime Services, Physical Topology Routing, Social Topology Orchestration, Cross-Topology Boundary Defense Pipeline, and an Execution Plane.
- The architecture decouples intent generation from physical actuation, enabling agents to function as attributed Digital Twins that operate across distributed hardware and collaborative social spaces.
- TopoClaw ensures structural safety through a distributed, context-aware policy enforcement pipeline that governs agent actions across both physical and social trust boundaries.

---

[DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](http://arxiv.org/abs/2605.15542)

- DRS-GUI: introduces a training-free framework that enhances GUI grounding by dynamically searching for instruction-relevant regions before final prediction.
- The framework utilizes a UI Perceptor to generate semantic cues and an MCTS-based Action Planner to execute Focus, Shift, and Scatter actions for adaptive perceptual exploration.
- By evaluating candidate regions with a composite reward function, the system effectively prunes visual clutter and improves grounding robustness in high-resolution, dense interfaces.

---

[RTL-BenchMT: Dynamic Maintenance of RTL Generation Benchmark Through Agent-Assisted Analysis and Revision](http://arxiv.org/abs/2605.15537)

- RTL-BenchMT: introduces an agentic framework for the dynamic maintenance of RTL generation benchmarks, utilizing a Manager Agent, Failure Analysis Agent, Description Revision Agent, Description Review Agent, and Description Update Agent.
- The framework employs an iterative reasoning paradigm where agents perform thought, action, and observation cycles within an Agent Environment and an isolated EDA Environment to identify flawed cases and detect LLM overfitting.
- RTL-BenchMT systematically reduces human maintenance costs by automating the identification of flawed benchmark cases and quantifying overfitting through semantically equivalent description variations.

---

[STS: Efficient Sparse Attention with Speculative Token Sparsity](http://arxiv.org/abs/2605.15508)

- STS (Speculative Token Sparsity): introduces a training-free sparse attention mechanism that leverages a smaller draft model to generate predictive sparsity masks for a larger target model within a speculative decoding framework.
- The framework decouples mask generation from target model execution, enabling a "known-in-advance" property that facilitates asynchronous prefetching of KV-cache blocks to hide memory transfer latency.
- By maintaining a lossless KV-cache and applying fine-grained token-wise sparsity, the approach achieves significant speedups while preserving model accuracy across both prefill and decode stages.

---

[uGen: An Agentic Framework for Generating Microarchitectural Attack PoCs](http://arxiv.org/abs/2605.15503)

- uGen: introduces a RAG-empowered multi-agent framework that systematically assesses and improves the ability of LLMs to generate functional microarchitectural attack PoCs across diverse attack classes.
- The framework utilizes a multi-agent architecture comprising Programmer-, Reflector-, Gap Analyzer-, Synthesizer- and Feedback-agents to perform role-specific reasoning, tool-grounded execution, and iterative refinement.
- uGen addresses LLM limitations in microarchitectural understanding by injecting attack-specific knowledge through a hierarchical RAG system and validating PoCs using hardware-derived signals.

---

[Hybrid LLM-based Intelligent Framework for Robot Task Scheduling](http://arxiv.org/abs/2605.15486)

- Hybrid LLM-based Intelligent Framework for Robot Task Scheduling: introduces a two-tier LLM pipeline that utilizes a Generator Agent to draft construction task schedules and a Supervisor Agent to validate and repair them using minimal-edit projections.
- The framework employs a Generator Agent and a Supervisor Agent to ensure feasibility in multi-robot construction environments by enforcing typed constraints such as battery safety, precedence, and coverage.
- The system leverages structured prompts and few-shot programmatic exemplars to improve the executability of LLM-generated plans while maintaining traceability for human inspection.

---

#### 14th May 2026


[Why Neighborhoods Matter: Traversal Context and Provenance in Agentic GraphRAG](http://arxiv.org/abs/2605.15109)

- Agentic GraphRAG: introduces a trajectory-level evaluation framework for citation faithfulness in LLMs by analyzing the impact of graph traversal, structure, and visited-but-uncited entities on answer generation.
- The research utilizes a graph-ablation methodology to demonstrate that while cited evidence is often necessary for accuracy, it is not sufficient, as broader graph context significantly influences the reasoning process.
- The study reveals that citation faithfulness in agentic systems requires accounting for the entire retrieval trajectory, including visited nodes and structural cues, rather than relying solely on final output citations.

---


[Falkor-IRAC: Graph-Constrained Generation for Verified Legal Reasoning in Indian Judicial AI](http://arxiv.org/abs/2605.14665)

- Falkor-IRAC: introduces a graph-constrained generation framework that grounds legal reasoning in structured IRAC knowledge graphs to ensure verifiable outputs.
- The architecture utilizes a Retrieval Agent to extract path-guided context from FalkorDB, which the LLM Generator uses to propose answers subject to a hard veto by the Verifier Agent.
- By modeling litigation flow and doctrinal conflicts as first-class graph components, the system enforces strict citation grounding and detects unresolved legal splits.

---



[Computational Thinking Development in AI Agent Creation: A Mixed-Methods Study](http://arxiv.org/abs/2605.14330)

- CocoFlow: introduces a no-code platform for AI agent creation that utilizes a Module Library, Visual Canvas, Intent Recognition Module, Entity Extraction Module, Conditional Logic Nodes, Real-time Chat Simulator, and a Pre-trained NLU Engine to facilitate computational thinking development.
- The study demonstrates that iterative testing engagement within the platform significantly predicts self-efficacy gains among students.
- Research findings reveal an "Optimal Development Zone" where students with moderate initial computational thinking levels achieve the greatest developmental benefits compared to high- or low-level peers.

---

[Agentic AI Ecosystems in Higher Education: A Perspective on AI Agents to Emerging Inclusive, Agentic Multi-Agent AI Framework for Learning, Teaching and Institutional Intelligence](http://arxiv.org/abs/2605.14266)

- Agentic Multi-Agent AI Framework: introduces a unified, multi-stakeholder ecosystem that integrates specialized agents to support learning, teaching, and institutional processes through coordinated planning, reasoning, and adaptive decision-making.
- The framework utilizes a layered architecture comprising a User Interface Layer, a Coordination Layer, and a Data & Knowledge Layer to facilitate seamless interaction between Learning Agents, Teaching Agents, Inclusion Agents, and Institutional Agents.
- This approach addresses the fragmentation of current educational AI by embedding inclusive pedagogy and ecosystem-level intelligence into a scalable, human-aligned, and adaptive multi-agent architecture.

---


[Characterizing AI-Assisted Bot Traffic in Darknet Data: Implications for ICS and IIoT Security](http://arxiv.org/abs/2605.14209)

- Darknet Traffic Analysis Pipeline: introduces a modular framework for characterizing longitudinal darknet traffic to identify AI-assisted botnet reconnaissance targeting critical infrastructure.
- The framework utilizes Merit ORION Network Telescope (ingests darknet PCAP files), Data Ingestion &amp; Preprocessing (parses, filters, and normalizes packets), Feature Extraction (calculates packet rate, protocol, and port flags), Statistical Analysis (core evaluation engine), Burstiness (measures inter-arrival time distributions), Shannon Entropy (measures traffic diversity), IAT (inter-arrival time metrics), ICS Port Targeting (identifies industrial protocol activity), Geographic Attribution (maps source IP origins), and IDS Simulation (evaluates volumetric threshold evasion).
- The study demonstrates that modern botnets employ deliberate micro-pacing to evade standard volumetric IDS thresholds, necessitating a shift toward behavior-aware detection mechanisms.

---


[Guises and Perspectives: An Intentional and Hyperintensional Sketch](http://arxiv.org/abs/2605.15144)

- GL (Guise Logic): introduces a formal framework for intensional logic where guises serve as primary semantic objects to model intentional reference and internal relations.
- The system integrates Leibnizian containment semantics with an intentional operator and a modal layer to address hyperintensional phenomena like substitution failure and de se reference.
- The framework provides a flexible architecture supporting canonical, template-restricted, and finite models to balance cognitive realism with formal tractability.

---


[GraphFlow: An Architecture for Formally Verifiable Visual Workflows Enabling Reliable Agentic AI Automation](http://arxiv.org/abs/2605.14968)

- GraphFlow: introduces a visual workflow architecture that treats diagrams as executable specifications to enable formally verifiable and reliable agentic AI automation through Diagram-as-specification, Verified core, Durable runtime, Cohort search, Operational dashboards, Swimlanes, Event log, Compiler, and Proof assistant.
- The framework improves reliability by shifting agentic planning from ad hoc tool-use to selecting and parameterizing pre-approved, contract-checked workflows.
- GraphFlow isolates nondeterminism at explicit boundaries using swimlanes and ensures reproducibility through durable execution with deterministic replay.

---


[ATLAS: Agentic or Latent Visual Reasoning? One Word is Enough for Both](http://arxiv.org/abs/2605.15198)

- ATLAS: introduces a visual reasoning framework that represents complex visual operations as discrete functional tokens within a standard autoregressive sequence to avoid external tool execution or intermediate image generation.
- The framework utilizes LA-GRPO to mitigate gradient dilution by anchoring reinforcement learning updates specifically to functional tokens, ensuring stable and effective optimization.
- ATLAS maintains compatibility with standard VLM architectures and parallel training pipelines while achieving superior performance on challenging visual reasoning benchmarks with reduced inference latency.

---

[Good to Go: The LOOP Skill Engine That Hits 99% Success and Slashes Token Usage by 99% via One-Shot Recording and Deterministic Replay](http://arxiv.org/abs/2605.14237)

- LOOP Skill Engine: introduces a one-shot recording and deterministic replay paradigm to optimize periodic LLM agent tasks by replacing repeated LLM inferences with parameterized, invariant tool-call sequences.
- The system utilizes a greedy length-descending template extraction algorithm to convert LLM-generated tool trajectories into deterministic skills, effectively eliminating token consumption and non-determinism for subsequent executions.
- A robust Heartbeat Scheduler and multi-layer degradation strategy ensure high reliability and crash-safe persistence, allowing the framework to maintain continuous operation even when individual task stages fail.

---

[FUTURESIM: Replaying World Events to Evaluate Adaptive Agents](http://arxiv.org/abs/2605.15188)

- FutureSim: introduces a chronological simulation environment that replays real-world events to evaluate the long-horizon adaptive forecasting capabilities of LLMs.
- The framework utilizes a date-gated news corpus and a structured agent harness to test how LLMs update probability distributions over free-form outcomes as new information arrives.
- Experimental results demonstrate that while frontier LLMs show performance improvements with better harness design and increased test-time compute, they often struggle with overconfidence and anchoring to initial predictions.

---

[Articraft: An Agentic System for Scalable Articulated 3D Asset Generation](http://arxiv.org/abs/2605.15187)

- Articraft: introduces an agentic system that leverages LLMs to generate articulated 3D assets by iteratively writing and refining code against a domain-specific SDK.
- The framework utilizes an agent harness to provide structured feedback, enabling the LLM to perform iterative, execution-grounded refinement of 3D object programs.
- Articraft facilitates the creation of Articraft-10K, a large-scale dataset of articulated 3D assets, which improves performance in downstream robotics simulation and 3D articulation estimation tasks.

---

[Is Grep All You Need? How Agent Harnesses Reshape Agentic Search](http://arxiv.org/abs/2605.15184)

- Chronos: introduces an empirical study evaluating how retrieval strategies, agent harnesses, and tool-calling architectures jointly influence the performance of LLM agents in long-memory tasks.
- The research compares lexical (Grep) and semantic (Vector) retrieval across custom and provider-native CLI harnesses, demonstrating that retrieval effectiveness is highly dependent on the specific agent stack and delivery method.
- Findings indicate that while lexical search often outperforms semantic search in inline delivery, programmatic file-based delivery can significantly alter these performance dynamics by introducing new constraints on agent tool-use competence.

---

[From Plans to Pixels: Learning to Plan and Orchestrate for Open-Ended Image Editing](http://arxiv.org/abs/2605.15181)

- Plan2Pix: introduces an experiential learning framework for long-horizon image editing that decomposes abstract instructions into structured sub-tasks executed by a reward-driven orchestrator.
- The system utilizes a checklist-guided Planner to generate atomic sub-tasks and an Orchestrator that selects tools and regions based on feedback from a VLM-Judge.
- Closed-loop refinement and verifier-guided selection ensure that the generated plans remain feasible and coherent throughout the multi-step editing process.

---

[BEHAVIOURAL ASSURANCE CANNOT VERIFY the Safety Claims Governance Now Demands](http://arxiv.org/abs/2605.15164)

- Pilot architecture (P1–P6): introduces a reproducible mechanistic-evidence protocol to bridge the audit gap between governance requirements and current behavioural assurance methods, utilizing P1. Claim form, P2(a). Linear probe, P2(b). Activation patching, P2(c). Before/after-training comparison, P3. Pre-registered thresholds, P4. Secure Enclave (TEE), P5. Bounded compute budget, and P6. Report.
- The framework addresses the structural mismatch where current governance demands high-consequence safety proofs that behavioural evaluations cannot provide, by mandating mechanistic evidence within a secure, reproducible audit environment.
- This approach shifts the verification paradigm from surface-level behavioural testing to deep structural verification, specifically targeting latent properties like hidden objectives and deceptive alignment in LLMs.

---

[APWA: A Distributed Architecture for Parallelizable Agentic Workflows](http://arxiv.org/abs/2605.15132)

- APWA (Agent-Parallel Workload Architecture): introduces a distributed multi-agent system designed to efficiently process parallelizable agentic workloads by decomposing tasks into non-interfering subproblems executed by independent agents.
- The architecture utilizes a Manager Agent for high-level planning and task partitioning, while Subtask Worker Agents perform autonomous execution within isolated Subtask Sandboxes.
- APWA leverages a scalable distributed fabric to support heterogeneous data processing and dynamic agent capabilities, enabling high-throughput performance on large-scale tasks where prior multi-agent systems encounter scaling bottlenecks.

---

[MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory](http://arxiv.org/abs/2605.15128)

- MemEye: introduces a two-dimensional evaluation framework that categorizes multimodal agent memory challenges by Visual Evidence Granularity and Memory Reasoning Depth.
- The framework utilizes a benchmark of 371 mirrored questions across eight life-scenario tasks, incorporating Filtering Mechanisms and Diagnostic Probes to isolate failures in visual preservation and temporal state tracking.
- Empirical evaluation of 13 memory methods across four VLM backbones reveals that while text-based memory manages state transitions effectively, it often loses fine-grained visual details, whereas multimodal memory preserves visual evidence but struggles with temporal validity and state selection.

---

[From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents](http://arxiv.org/abs/2605.15104)

- Dataset-agnostic framework for audio tool-calling evaluation: introduces a methodology to convert text-based tool-calling benchmarks into controlled audio evaluations using TTS Pipeline, Omni-Modal LLMs, Automatic Evaluation, and LLMs-as-Judge to measure modality-induced performance degradation.
- The framework enables paired text-audio evaluation by preserving original tool schemas and gold labels, allowing for precise diagnostic analysis of tool-calling failures in voice-enabled LLMs.
- Experimental results across seven omni-modal models demonstrate that tool-calling performance is highly model- and task-dependent, with argument-value errors being the primary cause of failure in speech-based interactions.

---

[Veritas: A Semantically Grounded Agentic Framework for Memory Corruption Vulnerability Detection in Binaries](http://arxiv.org/abs/2605.15097)

- Veritas: introduces a semantically grounded agentic framework for detecting memory corruption vulnerabilities in stripped binaries by unifying static program analysis and runtime validation as two grounding layers for controlled LLM reasoning.
- The framework utilizes a Semantic-driven Context Slicer to extract witness-backed flows, a Dual-view Vulnerability Detector for step-wise reasoning, and an Automatic Vulnerability Validator to confirm hypotheses through debugger-visible artifacts.
- By constraining LLM reasoning to verifiable program semantics reconstructed from binary artifacts, Veritas achieves high recall and low false positives while outperforming existing static, dynamic, and agentic baselines.

---

[Concurrency without Model Changes: Future-based Asynchronous Function Calling for LLMs](http://arxiv.org/abs/2605.15077)

- AsyncFC: introduces an execution-layer framework that decouples LLM decoding from function execution using Scheduler, State Tree, Function Executor, Future Placeholders, and Dependency Annotations.
- The framework enables asynchronous function calling by returning Future Placeholders to the LLM, allowing decoding to overlap with background function execution.
- AsyncFC utilizes a dependency-aware Scheduler and State Tree to enforce safe inter-function parallelism without requiring modifications to the underlying LLM or function implementations.

---

[After the Interface: Relocating Human Agency in the Age of Conversational AI](http://arxiv.org/abs/2605.15064)

- Human-AI Interaction Agency Framework: introduces a two-dimensional diagnostic model that maps AI systems based on Process Control and Outcome Control to visualize the redistribution of human agency.
- The paper argues that human agency in the era of LLMs has not eroded but has relocated from interface-level procedural manipulation to communicative negotiation and outcome evaluation.
- This research highlights that contemporary AI systems demand new metrics for agency that account for iterative judgment, relational trust, and the unequal distribution of communicative competence among users.

---

[SpeakerLLM: A Speaker-Specialized Audio-LLM for Speaker Understanding and Verification Reasoning](http://arxiv.org/abs/2605.15044)

- SpeakerLLM: introduces a speaker-specialized audio-LLM framework that unifies single-utterance speaker profiling, recording-condition understanding, utterance-pair comparison, and evidence-organized verification reasoning.
- The framework utilizes a hierarchical speaker tokenizer to distribute speaker evidence across utterance-level embeddings and frame-level features for improved acoustic and identity modeling.
- SpeakerLLM-VR employs a structured three-block verification reasoning target to generate auditable decision traces that separate profile-level evidence from final verification verdicts.

---

[Orchard: An Open-Source Agentic Modeling Framework](http://arxiv.org/abs/2605.15040)

- Orchard: introduces an open-source framework for scalable agentic modeling, centered on a thin, Kubernetes-native environment service (Orchard Env) that decouples sandbox management from agent harnesses and training stacks.
- The framework enables reusable agentic modeling across diverse domains, including Orchard-SWE, Orchard-GUI, and Orchard-Claw, by providing harness-agnostic primitives for sandbox lifecycle, command execution, and file I/O.
- Orchard utilizes Balanced Adaptive Rollout (BAR) and credit-assignment SFT to improve training efficiency and generalization, achieving state-of-the-art performance for open-source agents while maintaining cost-effectiveness.

---

[AI Knows When It’s Being Watched: Functional Strategic Action and Contextual Register Modulation in Large Language Models](http://arxiv.org/abs/2605.15034)

- Multi-agent LLM debate architecture: introduces a controlled experimental framework to evaluate how LLMs modulate their linguistic register in response to perceived social observation contexts.
- The study demonstrates that LLMs exhibit a "Synthetic Hawthorne Effect," where lexical diversity increases under monitoring, while message elaboration is driven by audience presence.
- The research reveals that LLM behavioral adaptation is sensitive to the identity of the observer, with human evaluation eliciting stronger register formalization than automated AI surveillance.

---

[On the Limits of PAC Learning of Networks from Opinion Dynamics](http://arxiv.org/abs/2605.15033)

- Waterfall algorithm: introduces a framework for learning social network structures from threshold-based opinion dynamics by utilizing a Matching Transformation and a greedy heuristic to identify feasible influencer sets.
- The paper establishes that while PAC learning is efficient for all-but-κ dynamics, it is computationally intractable for majority dynamics, leading to the development of the Waterfall heuristic.
- The proposed Waterfall algorithm achieves high empirical success rates in identifying network structures across various random graph models by iteratively resolving inconsistencies in opinion diffusion samples.

---

[WARD: Adversarially Robust Defense of Web Agents Against Prompt Injections](http://arxiv.org/abs/2605.15030)

- WARD: introduces a practical guard framework for web agents that utilizes a large-scale dataset, guard-targeted training, and an adaptive adversarial training loop to ensure robust detection of prompt injections.
- The framework employs a two-branch data construction pipeline to capture diverse attack patterns across HTML and visual interface modalities.
- A3T enables the guard to co-evolve with an adaptive attacker, significantly improving robustness against evolving adversarial strategies while maintaining high agent utility.

---

[Multi-Agentic Approach for History Matching of Oil Reservoirs](http://arxiv.org/abs/2605.15028)

- PetroGraph: introduces a multi-agent framework that automates oil reservoir history matching by decomposing the workflow into specialized LLM-based agents and a non-LLM simulator agent.
- The system integrates RAG for domain-specific documentation access, human-in-the-loop checkpoints for manual steering, and Bayesian optimization to calibrate reservoir parameters against observed field data.
- Evaluations on synthetic and real-field models demonstrate that the framework effectively reduces history-matching mismatches while lowering the expertise barrier for complex simulation workflows.

---

[COTCAgent: Preventive Consultation via Probabilistic Chain-of-Thought Completion](http://arxiv.org/abs/2605.15016)

- COTCAgent: introduces a hierarchical reasoning framework for longitudinal EHR analysis that decouples statistical trend computation from disease risk scoring to improve diagnostic traceability.
- The framework utilizes a Temporal-Statistics Adapter to convert irregular health records into structured trend predicates, which are then matched against a symptom-trend-disease knowledge base using IDF-weighted Gibbs energies.
- By integrating a bounded completion module for targeted user inquiries, the system iteratively refines disease risk rankings while maintaining an inspectable audit trail of clinical reasoning.

---

[Efficient Online Conformal Selection with Limited Feedback](http://arxiv.org/abs/2605.14953)

- Primal-Dual ACI: introduces a unified framework for online conformal selection under bandit and semi-bandit feedback, utilizing an ACI Controller, Primal-Dual Algorithm, UCB Estimator, Expert-Chain, Lyapunov Function, Bandit Arms, and Probing Budget to achieve adversarial validity and stochastic efficiency.
- The framework employs un-projected ACI updates to maintain per-sequence adversarial validity while minimizing resource costs through Lyapunov drift analysis.
- The approach generalizes to complex combinatorial selection spaces, including NP-Hard problems, by integrating ACI with expert-chain bandit algorithms to optimize probing budgets.

---

[Not All Symbols Are Equal: Importance-Aware Constellation Design for Semantic Communication](http://arxiv.org/abs/2605.14940)

- Semantic-PHY framework: introduces a joint semantic-physical layer architecture that co-designs constellation assignment with semantic importance and statistical co-occurrence structure of learned concept vocabulary.
- The system utilizes a VQ-VAE encoder and SCI network to identify task-critical features, which are then protected at the physical layer by a learned M-QAM constellation that maximizes physical separation for high-importance symbols.
- A DQN-based rate controller dynamically adjusts the transmission payload based on channel SNR, enabling robust semantic communication across varying channel conditions and diverse data domains.

---

[Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations](http://arxiv.org/abs/2605.14937)

- Slot-MPC: introduces a goal-conditioned planning framework that leverages a Scene Parsing module, a cOCVP, a Policy network, an MPC optimizer, and Slot-based representations to perform efficient trajectory optimization in a structured latent space.
- The framework utilizes a Scene Parsing module to decompose visual observations into Slot-based representations, which are then processed by a cOCVP to forecast future states for planning.
- By employing a gradient-based MPC optimizer warm-started by a Policy network, the approach achieves efficient goal-directed control in complex robotic manipulation tasks without requiring environment interaction during training.

---

[Toward Securing AI Agents Like Operating Systems](http://arxiv.org/abs/2605.14932)

- OpenClaw-style agents: introduces a unified architecture for AI agents by drawing a structural analogy to operating systems, identifying key components including Runtime Core, Agent Core, LLM Interface, Gateway, Task Queue, Session Store, Logs, Persistent State, Plugins, Skills, Skill Tools, Core Tools, and Tool Workspaces.
- The paper systematically maps OS security principles—such as process isolation, sandboxing, and privilege separation—to agentic systems to address vulnerabilities arising from unconstrained tool use and sensitive data access.
- An empirical case study of four popular agent runtimes demonstrates that current protection mechanisms are often fragmented or ineffective, highlighting the necessity for comprehensive, OS-inspired security boundaries.

---

[Static and Dynamic Strategies for Influencing Opinions in Social Networks](http://arxiv.org/abs/2605.14918)

- Hegselmann–Krause (HK) model: introduces a comparative study of static and dynamic influence strategies for manipulating collective opinions in social networks using targeted stubborn agents.
- The research evaluates how different centrality-based node selection strategies interact with intervention timing to shift network-wide opinion distributions.
- Results demonstrate that dynamic strategies are significantly more effective than static ones, as they leverage bounded-confidence dynamics to recruit intermediate agents and avoid premature opinion fragmentation.

---

[Chrono-Gymnasium: An Open-Source, Gymnasium-Compatible Distributed Simulation Framework](http://arxiv.org/abs/2605.14911)

- Chrono-Gymnasium: introduces a distributed simulation framework that bridges high-fidelity multi-physics engines with scalable execution pipelines for RL and design optimization.
- The framework utilizes Ray Actors to encapsulate Project Chrono simulation instances, enabling parallel rollouts and efficient data generation across heterogeneous computing clusters.
- By standardizing simulation scenarios through a Gymnasium-compatible interface, the architecture simplifies the integration of complex physics models into modern machine learning and Bayesian optimization workflows.

---

[MEMLENS: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models](http://arxiv.org/abs/2605.14906)

- MEMLENS: introduces a comprehensive benchmark for evaluating multimodal long-term memory in LVLMs and memory-augmented agents, utilizing Multimodal Session Simulation, Question-Answer Pair Construction, Evidence Session Construction, Conversation History Assembly, Automated Filtering, and Human Review.
- The benchmark assesses five core memory abilities—information extraction, multi-session reasoning, temporal reasoning, knowledge update, and answer refusal—across four standardized context lengths (32K–256K tokens).
- Evaluation of 27 LVLMs and 7 memory-augmented agents reveals that while long-context LVLMs excel at short-context visual grounding, they degrade as conversations grow, whereas memory agents remain length-stable but suffer from lossy visual compression.

---

[Beyond Individual Intelligence: Surveying Collaboration, Failure Attribution, and Self-Evolution in LLM-based Multi-Agent Systems](http://arxiv.org/abs/2605.14892)

- LIFE (Lay, Integrate, Find, Evolve) progression: introduces a unified analytical framework for LLM-based multi-agent systems, connecting individual intelligence, collaboration, failure attribution, and self-evolution as causally linked stages.
- The framework characterizes the operational lifecycle of LLMs, where individual agent capabilities (reasoning, memory, planning, tool use) are integrated through collaborative structures, diagnosed via failure attribution, and refined through autonomous self-evolution.
- This survey synthesizes existing research into a coherent roadmap, identifying open challenges at the boundaries of these stages to advance toward resilient, self-organizing collective intelligence.

---

[Temporal Fair Division in Multi-Agent Systems: From Precise Alternation Metrics to Scalable Coordination Proxies](http://arxiv.org/abs/2605.14879)

- Temporal Fair Division Framework: introduces a diagnostic toolkit for assessing coordination quality in repeated multi-agent resource competition by comparing ALT (Sliding-window coordination metrics) and RP (Linear-time fairness metric).
- The framework formalizes MBoE (Repeated competitive resource game) and establishes PA (Canonical temporally fair solution) as the ideal benchmark for evaluating agent coordination.
- Empirical results demonstrate that Q-learning (Independent learning agent policy) consistently fails to achieve temporal fairness, while RP (Linear-time fairness metric) provides a scalable, symmetric alternative to ALT (Sliding-window coordination metrics) for large-scale systems.

---

[Towards In-Depth Root Cause Localization for Microservices with Multi-Agent Recursion-of-Thought](http://arxiv.org/abs/2605.14866)

- RCLAgent: introduces a multi-agent recursion-of-thought framework that decomposes root cause localization along the trace graph to mitigate context explosion and enable parallel reasoning.
- The framework utilizes Dedicated Agents for span-level analysis, an Agents Pool for controlled parallelism, and a Diagnosis Synthesizer that combines a Root-Level Diagnosis Report with a Global Evidence Graph to improve localization accuracy.
- By replacing monolithic serial reasoning with trace-aligned recursive decomposition, RCLAgent effectively balances deep causal exploration with inference efficiency in complex microservice environments.

---

[Holistic Evaluation and Failure Diagnosis of AI Agents](http://arxiv.org/abs/2605.14865)

- Holistic Agent Evaluation Framework: introduces a dual-perspective evaluation method that pairs top-down agent-level diagnosis with bottom-up span-level assessment to provide fine-grained failure localization and categorization.
- The framework decomposes agent execution traces into independent span-level assessments, enabling scalable evaluation of long traces and producing natural language rationales for each verdict.
- By integrating top-down and bottom-up signals, the approach overcomes the limitations of monolithic LLM-judges, achieving state-of-the-art localization and categorization accuracy on the TRAIL benchmark.

---

[Do Coding Agents Understand Least-Privilege Authorization?](http://arxiv.org/abs/2605.14859)

- AuthBench: introduces a benchmark for evaluating the ability of LLMs to infer task-specific least-privilege authorization policies before execution, utilizing an Authorization Agent, Execution Agent, Utility Validator, and Attack Validator.
- The research identifies that LLMs converge toward model-specific authorization attractors, where increased reasoning effort reinforces preferred failure modes rather than improving policy tightness or sufficiency.
- The proposed Sufficiency-Tightness Decomposition improves sensitive-task success and reduces attack exposure by separating coverage-oriented policy generation from necessity and sensitivity auditing.

---

[IFPV: An Integrated Multi-Agent Framework for Generative Operational Planning and High-Fidelity Plan Verification](http://arxiv.org/abs/2605.14851)

- IFPV: introduces an integrated multi-agent framework that unifies generative operational planning via MPHA and high-fidelity adversarial verification through ACSE to address generation infeasibility and verification insufficiency.
- MPHA decomposes commander intent into executable tactical sequences using specialized Pathfinder-, Analyst-, and Planner-agents, while ACSE employs a customized world model to conduct dynamic stress testing against candidate plans.
- The framework utilizes EVA-Loss to inject entity-value awareness into the world model, enabling more discriminative adversarial verification and robust plan evaluation in complex battlefield environments.

---

[Learning Direct Control Policies with Flow Matching for Autonomous Driving](http://arxiv.org/abs/2605.14832)

- Flow-matching planner: introduces a conditional flow-matching architecture that generates actionable control trajectories for autonomous driving by integrating a learned ODE vector field conditioned on BEV scene rasters.
- The architecture utilizes a heavy one-time BEV Encoder to process environmental data, followed by a lightweight Vector-field U-Net that iteratively refines control sequences via an ODE Solver.
- The model demonstrates robust out-of-distribution generalization to highway scenarios and unseen urban environments while maintaining real-time inference capabilities through efficient multi-step integration.

---

[Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces](http://arxiv.org/abs/2605.14786)

- Agent Fingerprinting Framework: introduces a passive identification method that leverages UI interaction traces to attribute web-browsing activity to specific LLMs.
- The framework utilizes an Injected JavaScript Tracker to capture temporal and structural behavioral signals, which are then processed by an XGBoost Classifier to achieve high-accuracy model identification.
- This research demonstrates that LLM agents leave distinct, fingerprintable behavioral traces during web navigation, enabling site operators to identify underlying models and potentially condition adversarial exploits or access control.

---

[Peng’s Q(λ) for Conservative Value Estimation in Offline Reinforcement Learning](http://arxiv.org/abs/2605.14779)

- CPQL (Conservative Peng’s Q(λ)): introduces a model-free offline RL algorithm that adapts the Peng’s Q(λ) operator for conservative value estimation by leveraging multi-step trajectories to mitigate over-pessimism and distributional shift.
- The framework utilizes Critic networks, Actor network, Offline dataset, Conservatism factor, Trace parameter λ, and Target networks to achieve robust performance without requiring additional auxiliary networks or behavior policy estimation.
- CPQL provides theoretical guarantees for performance improvement over the behavior policy and facilitates a smooth transition to online fine-tuning by pre-training a stable Q-function.

---

[MediaClaw: Multimodal Intelligent-Agent Platform Technical Report](http://arxiv.org/abs/2605.14771)

- MediaClaw: introduces a three-layer architecture that integrates heterogeneous AIGC capabilities into a unified, pluginized platform for reusable multimedia workflow orchestration.
- The platform utilizes a Meta-Capability Pool for atomic tool management, a Skill layer for complex process automation, and MediaUI for end-to-end visualization of intermediate artifacts.
- By decoupling business logic from specific model providers through a unified interface, the system enables flexible, Lego-like composition of multimodal production workflows.

---

[Probabilistic Verification of Recurrent Neural Networks for Single and Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.14758)

- RNN-ProVe: introduces a probabilistic verification framework that estimates the likelihood of undesired behaviors in RNN-based policies by leveraging a Feasibility History Classifier and Monte Carlo Estimator to assess policy-induced feasible hidden states.
- The framework addresses the #P-hard nature of exact verification by using a learned classifier to approximate the feasible hidden state manifold, enabling scalable and quantitative safety assessments for single- and multi-agent RL.
- RNN-ProVe provides bounded-error, high-confidence certificates by combining statistical analysis with policy-driven sampling, effectively overcoming the limitations of traditional over-approximation-based verification methods.

---

[Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](http://arxiv.org/abs/2605.14747)

- Video2GUI: introduces a fully automated framework that extracts grounded GUI interaction trajectories from unlabeled internet videos to facilitate large-scale agent pretraining.
- The framework employs a coarse-to-fine filtering strategy, including Meta Info Filtering (text-based coarse video screening), Video Quality Scoring (fine-grained content-based evaluation), Trajectory Extraction (converting videos to instruction-trajectory pairs), Action Spatial Grounding (mapping actions to screen coordinates), and Agent Training (two-stage continual pre-training and post-training).
- By applying this pipeline to 500 million video metadata entries, the authors construct WildGUI, a large-scale dataset containing 12 million interaction trajectories that significantly improves the generalization capabilities of LLMs like Qwen2.5-VL and Mimo-VL.

---

[Mechanical Enforcement for LLM Governance: Evidence of Governance-Task Decoupling in Financial Decision Systems](http://arxiv.org/abs/2605.14744)

- R2 (Mechanical Policy): introduces a governance framework that decouples decision-making from policy interpretation by using four external primitives to enforce rationale quality and decision boundaries.
- The framework utilizes CEFL, I6Q, E3, and Hard Gates to prevent LLMs from producing vacuous, non-compliant rationales under structural stress.
- Experimental results demonstrate that mechanical enforcement significantly reduces the Cosmetic Deadlock Rate and preserves governance quality even when task accuracy degrades.

---

[Agentifying Patient Dynamics within LLMs through Interacting with Clinical World Model](http://arxiv.org/abs/2605.14723)

- SepsisAgent: introduces a world model-augmented LLM agent for sepsis treatment that utilizes a Clinical World Model to simulate patient responses and refine prescriptions through a propose–simulate–refine workflow.
- The framework employs a three-stage training curriculum consisting of supervised fine-tuning for patient-dynamics prediction, behavior cloning for agentic interaction, and world-model-based reinforcement learning for long-horizon policy optimization.
- Experimental results on MIMIC-IV demonstrate that SepsisAgent outperforms traditional RL and LLM-based baselines in off-policy value and safety metrics, while internalizing patient dynamics to improve mortality and vasopressor-requirement prediction.

---

[SR-Platform: An Agentic Pipeline for Natural Language-Driven Robot Simulation Environment Synthesis](http://arxiv.org/abs/2605.14700)

- SR-Platform: introduces a production-deployed agentic pipeline that converts natural language descriptions into executable, physically valid MuJoCo simulation environments through a modular, cache-aware architecture.
- The system utilizes an LLM-based orchestrator, an asset forge for semantic retrieval or CadQuery-based geometry synthesis, a layout architect for constraint verification, and a bridge layer for final MJCF assembly.
- By separating semantic planning, asset resolution, and layout reasoning, the platform enables scalable, auditable, and robust environment generation while leveraging production telemetry to optimize performance and reliability.

---

[π-BENCH: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows](http://arxiv.org/abs/2605.14678)

- π-BENCH: introduces a benchmark for evaluating proactive personal assistant agents in long-horizon workflows, utilizing User Roles, Episode Structure, Persistent Project Environment, Evaluated Agent, User Agent, Hidden Intent Tracker, Checklist, and Graders.
- The framework evaluates LLMs on their ability to identify and resolve hidden user intents through proactive action or targeted clarification while maintaining task completion across multi-session trajectories.
- Experiments across nine frontier LLMs demonstrate that while task completion and proactivity are related, they represent distinct capabilities, with prior interaction history significantly aiding proactive intent resolution.

---

[Agentic AI in Industry: Adoption Level and Deployment Barriers](http://arxiv.org/abs/2605.14675)

- Agentic AI Maturity Framework: introduces a six-level classification system to evaluate industrial adoption of agentic systems, ranging from individual AI Assistants to fully autonomous Self-Optimizing AI Systems.
- The study identifies a capability-deployment verification gap, where experimental agentic capabilities cannot be integrated into production due to the absence of adequate output verification mechanisms.
- Four recurring barriers—context window limitations, underperformance in proprietary content, non-determinism, and data confidentiality—collectively hinder the transition from experimental agentic workflows to qualified production deployment.

---

[Documentation-Guided Agentic Codebase Migration from C to Rust](http://arxiv.org/abs/2605.14634)

- RustPrint: introduces a documentation-driven agentic framework that leverages automatically generated codebase documentation as an intermediate representation to guide the migration of legacy C repositories to idiomatic, memory-safe Rust.
- The framework utilizes specialized LLM-based agents including DocGen, Planner, Translator, Synthesizer, RequirementRefiner, TestTranslator, and ExecutionRevisor to perform iterative, documentation-guided translation and execution-aware refinement.
- Experimental results on eight real-world C repositories demonstrate that RustPrint achieves superior compilability, feature preservation, and safety compared to existing LLM-based translation baselines.

---

[SmartWalkCoach: An AI Companion for End-to-End Walking Guidance, Motivation, and Reflection](http://arxiv.org/abs/2605.14628)

- SmartWalkCoach: introduces an end-to-end, tool-using agent architecture for walking that reduces cognitive load by orchestrating GeographyAgent, AccompanyAgent, SummaryAgent, and a BridgingAgent.
- The system utilizes a shared state object to enable non-communicating agents to coordinate, ensuring secure and modular interaction management across the walking journey.
- Field evaluation demonstrates that integrating context-aware motivational dialogue significantly improves user experience and positive affect compared to information-only navigation.

---

[Digital Twin Synchronization Over Mobile Embodied AI Network With Agentic Intelligence](http://arxiv.org/abs/2605.14625)

- MEAN: introduces a hierarchical framework for digital twin synchronization that coordinates distributed embodied AI agents through a five-stage closed-loop workflow of move-to-sense, cooperative sensing, onboard semantic processing, channel-aware mobility, and uplink transmission.
- The framework employs a two-layer optimization algorithm where the outer-layer manages multi-agent assignment via a dynamic matching game, while the inner-layer optimizes continuous resources including sensing, computation, and communication.
- The system minimizes the maximum twin deviation across regions by leveraging semantic compression to reduce latency and autonomous velocity adaptation to navigate the energy-time trade-off.

---

[In-IDE Toolkit for Developers of AI-Based Features](http://arxiv.org/abs/2605.14612)

- AI Toolkit: introduces an IDE-native observability and evaluation framework that integrates directly into the Run/Debug loop to assist developers in testing non-deterministic LLM-based features.
- The framework utilizes a client-server architecture with a Python wrapper to capture execution traces and an IDE-side server to visualize agent behavior and manage evaluation datasets.
- By treating evaluations as first-class tests and providing a low-friction path from trace to dataset, the toolkit enables non-ML specialists to adopt disciplined AI development practices without leaving their primary coding environment.

---

[Silent Collapse in Recursive Learning Systems](http://arxiv.org/abs/2605.14588)

- MTR (Monitor–Trust–Regulator): introduces a metacognitive control loop that detects and prevents silent collapse in iterative learning systems by monitoring trajectory-level statistics and adaptively modulating learning intensity.
- The framework identifies hidden contraction—characterized by predictive entropy contraction, representation drift, and tail coverage erosion—as a reliable precursor to visible model degradation.
- MTR maintains learning stability and prevents catastrophic performance loss in recursive language modeling and pseudo-labeling tasks without requiring access to pristine real data.

---

[Angel or Demon: Investigating the Plasticity Interventions’ Impact on Backdoor Threats in Deep Reinforcement Learning](http://arxiv.org/abs/2605.14587)

- SCC (Sweeper-Converter-Connector): introduces a conceptual framework for robust backdoor injection in DRL by deconstructing the mechanistic interplay between plasticity interventions and backdoor threats, utilizing Sweeper, Converter, Connector, Pathological Diagnosis, and Sharpness-Based Detection.
- The paper empirically demonstrates that while most plasticity interventions mitigate backdoor threats, SAM exacerbates them by amplifying backdoor gradients and guiding optimization toward flat minima.
- The study identifies three intrinsic mechanisms—activation pathway disruption, representation space compression, and backdoor gradient amplification—that explain how interventions modulate DRL backdoor vulnerabilities.

---

[Remember Your Trace: Memory-Guided Long-Horizon Agentic Framework for Consistent and Hierarchical Repository-Level Code Documentation](http://arxiv.org/abs/2605.14563)

- MemDocAgent: introduces a long-horizon agentic framework that generates consistent, hierarchical repository-level documentation by utilizing Dependency-Aware Traversal Guiding and Memory-Guided Agentic Interaction.
- The framework employs a centralized RepoMemory to store and reuse documentation across sub-tasks, ensuring cross-document consistency and reducing redundant retrievals.
- MemDocAgent utilizes a multi-turn agentic workflow with READ, WRITE, and VERIFY operations to maintain persistent state and produce documentation that supports practical code reconstruction.

---

[Resolving Action Bottleneck: Agentic Reinforcement Learning Informed by Token-Level Energy](http://arxiv.org/abs/2605.14558)

- ACTFOCUS: introduces a token-level reweighting approach that mitigates the Action Bottleneck by downweighting reasoning tokens and prioritizing high-energy action tokens to improve credit assignment in agentic RL.
- The framework utilizes a frozen-reference model to compute token-level energy, which serves as a stable proxy for predictive uncertainty to guide gradient redistribution.
- ACTFOCUS consistently improves performance and training stability across multiple environments and LLM scales by shifting gradient mass toward critical environment-facing action tokens.

---

[TeachAnything: A Multimodal Crowdsourcing Platform for Training Embodied AI Agents in Symmetrical Reality](http://arxiv.org/abs/2605.14556)

- TeachAnything: introduces a three-stage demonstration paradigm that integrates Language-based demonstration, Video-based demonstration, and Teleoperation-based demonstration to collect multimodal data for training embodied agents.
- The platform utilizes a Physics Simulation backend with diverse Robot Embodiments and a Controller to generate physically consistent, temporally aligned training data.
- By leveraging WebSocket streaming and Flask microservices, the system enables distributed, cloud-based crowdsourcing of fine-grained manipulation strategies and high-level task intent.

---

[LiWi: Layering in the Wild](http://arxiv.org/abs/2605.14552)

- LiWi: introduces a framework for high-fidelity natural image decomposition that utilizes an ADD (Agent-driven Data Decomposition) pipeline to construct a large-scale dataset of in-the-wild layered images.
- The framework incorporates a shadow layer to explicitly model complex illumination effects and a degradation-restoration objective to improve alpha boundary accuracy during the generation process.
- By leveraging an agentic system with specialized tools, the approach enables scalable, automated supervision for natural image layering without requiring manual annotation.

---

[VerbalValue: A Socially Intelligent Virtual Host for Sales-Driven Live Commerce](http://arxiv.org/abs/2605.14542)

- VerbalValue: introduces a dual-channel architecture for live-commerce hosting that balances continuous product narration with responsive, intent-conditioned interaction using a fine-tuned LLM.
- The framework utilizes a structured product knowledge base and intent-conditioned fine-tuning to ensure factual accuracy and tactical alignment with viewer comments.
- By decoupling latency-sensitive generation from media operations and employing a reranker for candidate selection, the system maintains high engagement and factual correctness in real-time broadcast environments.

---

[Cattle Trade: A Multi-Agent Benchmark for LLM Bluffing, Bidding, and Bargaining](http://arxiv.org/abs/2605.14537)

- Cattle Trade: introduces a multi-agent benchmark for evaluating LLMs in strategic reasoning under imperfect information, adversarial interaction, and resource constraints.
- The framework integrates competitive auctions, hidden-offer trade challenges, and discrete resource management to test the joint deployment of agentic capabilities.
- Empirical results demonstrate that strategic coherence, such as spending efficiency and resource discipline, is more critical for success than individual sub-skills or spending volume.

---

[Lang2MLIP: End-to-End Language-to-Machine Learning Interatomic Potential Development with Autonomous Agentic Workflows](http://arxiv.org/abs/2605.14527)

- Lang2MLIP: introduces a multi-agent framework that automates the development of machine learning interatomic potentials by formulating the process as a sequential decision-making problem solved by LLMs.
- The framework utilizes a two-phase approach consisting of an interactive preparation phase for task clarification and structure generation, followed by an autonomous training phase for iterative model refinement.
- By employing a central decision-making agent to orchestrate specialized sub-agents, the system enables non-experts to develop stable and accurate MLIPs without requiring manually designed pipelines.

---

[When Robots Do the Chores: A Benchmark and Agent for Long-Horizon Household Task Execution](http://arxiv.org/abs/2605.14504)

- HoloMind: introduces a hierarchical agent framework for long-horizon household tasks, integrating High-level Planner, Low-level Planner, Executor, Multimodal Spatial Memory, Episodic Memory, and Critic.
- The framework utilizes a DAG-based hierarchical planner to decompose complex instructions into executable subgoals, supported by persistent memory and reflective supervision to maintain stable long-term execution.
- The paper also presents LongAct, a benchmark for evaluating long-horizon planning autonomy, featuring free-form instructions and an Improvement Rate metric to quantify experience-driven performance gains.

---

[GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations](http://arxiv.org/abs/2605.14498)

- GroupMemBench: introduces a benchmark for evaluating LLM agent memory in multi-party conversations by utilizing a Graph-grounded synthesis pipeline, an Adversarial query pipeline, and a Solve-Judge-Refine loop.
- The framework evaluates memory systems across three dimensions: group dynamics, speaker-grounded belief tracking, and audience-adapted language.
- Benchmarking results reveal that current memory systems often flatten group structure and fail to condition retrieval on speaker identity, leading to significant performance gaps.

---

[Contestable Multi-Agent Debate with Arena-based Argumentative Computation for Multimedia Verification](http://arxiv.org/abs/2605.14495)

- A-QBAF (Arena-based Quantitative Bipolar Argumentation Framework): introduces a contestable multi-agent framework that integrates multimodal LLMs, external verification tools, and A-QBAF to provide transparent, editable, and evidence-grounded multimedia verification.
- The framework decomposes multimedia cases into claim-centered sections, utilizing a planner agent, deep researcher agent, and argument cards to structure evidence into support-attack relations.
- It employs a sparse local graph design for efficient reasoning, selective clash resolution via a judge model, and uncertainty-aware escalation to ensure reliable outcomes in high-stakes verification.

---

[LEMON: Learning Executable Multi-Agent Orchestration via Counterfactual Reinforcement Learning](http://arxiv.org/abs/2605.14483)

- LEMON: introduces a framework for learning compositional multi-agent orchestration by generating executable specifications that integrate task-specific roles, capacity levels, and dependency structures.
- The framework utilizes an LLM Orchestrator to produce YAML-based specifications, which are trained using a combination of global GRPO and localized counterfactual credit assignment to optimize performance and efficiency.
- By applying reward contrasts to specific edited spans of the orchestration specification, LEMON effectively addresses the sparse credit assignment problem inherent in multi-agent system training.

---

[Test-Time Learning with an Evolving Library](http://arxiv.org/abs/2605.14477)

- EVOLIB: introduces a test-time learning framework that enables LLMs to accumulate, reuse, and evolve knowledge across problem instances without parameter updates or external supervision.
- The framework maintains a shared library of modular skills and reflective insights, which are automatically extracted from inference trajectories and refined through a principled weighting and consolidation mechanism.
- By optimizing for both immediate utility and long-term value via Information Gain and Future Information Gain, EVOLIB allows simple instance-specific abstractions to evolve into general, reusable knowledge over time.

---

[Exploiting LLM Agent Supply Chains via Payload-less Skills](http://arxiv.org/abs/2605.14460)

- SCH (Semantic Compliance Hijacking): introduces a payload-less supply chain attack that weaponizes natural language skill documentation to induce LLMs into synthesizing malicious code at runtime.
- The framework utilizes MS-AO to iteratively refine adversarial skills within a sandbox, bypassing static and semantic security defenses by omitting explicit code signatures.
- The research demonstrates that highly aligned LLMs are paradoxically more vulnerable to semantic manipulation, achieving significant success rates in data exfiltration and remote code execution.

---

[LiSA: Lifelong Safety Adaptation via Conservative Policy Induction](http://arxiv.org/abs/2605.14454)

- LiSA (Lifelong Safety Adaptation): introduces a conservative policy induction framework that improves a fixed base guardrail through structured memory, including broad policy abstractions, conflict-aware local rules, and evidence-aware confidence gating.
- The framework operates via an online-offline loop, where sparse user-reported failures are abstracted into reusable policies and boundary-specific local rules to adapt to deployment environments without repeated fine-tuning.
- LiSA utilizes a Beta-posterior lower bound to gate memory reuse, ensuring that only sufficiently supported policies influence LLM inference, thereby maintaining robustness against noisy feedback and preventing overgeneralization.

---

[FrontierSmith: Synthesizing Open-Ended Coding Problems at Scale](http://arxiv.org/abs/2605.14445)

- FrontierSmith: introduces an automated pipeline that transforms closed-ended coding problems into open-ended variants through targeted mutations and multi-stage filtering to generate scalable training data for LLMs.
- The framework utilizes an idea divergence metric to quantify solution strategy diversity, ensuring that synthesized problems elicit genuine algorithmic exploration rather than single-strategy dominance.
- FrontierSmith-generated problems enable LLMs to exhibit long-horizon agent behavior, characterized by increased turn counts and thinking tokens, comparable to human-curated open-ended benchmarks.

---

[GGBound: A Genome-Grounded Agent for Microbial Life-Boundary Prediction](http://arxiv.org/abs/2605.14442)

- GGBound: introduces a genome-conditioned, tool-augmented LLM agent that maps microbial genotypes to physiological life boundaries using LucaOne Genome Encoder, Qwen Backbone, RAG Module, GEM Tool, and GRPO Optimizer.
- The framework integrates genomic embeddings with external biological evidence through a three-stage training pipeline comprising gene-text alignment, agentic supervised fine-tuning, and reinforcement learning with a counterfactual gene-grounding reward.
- GGBound achieves competitive performance against larger frontier LLMs by leveraging selective evidence acquisition and causal genomic conditioning to predict microbial traits such as growth ranges, optimal conditions, and metabolic capabilities.

---

[FuzzAgent: Multi-Agent System for Evolutionary Library Fuzzing](http://arxiv.org/abs/2605.14431)

- FuzzAgent: introduces a multi-agent system that transforms library fuzzing into an evolutionary process by utilizing specialized agents to iteratively refine harnesses based on runtime feedback.
- The architecture integrates an Agent Pool, dedicated Interfaces, and a stateful Environment to automate the entire fuzzing lifecycle, including build configuration, harness generation, and crash validation.
- FuzzAgent employs a closed-loop reasoning strategy where agents leverage runtime evidence to overcome coverage plateaus and distinguish genuine library bugs from harness-induced errors.

---

[Collaborative Yet Personalized Policy Training: Single-Timescale Federated Actor-Critic](http://arxiv.org/abs/2605.14423)

- pFedAC (Personalized Federated Actor-Critic): introduces a federated reinforcement learning framework that enables agents to collaboratively learn a shared linear subspace representation while maintaining personalized local critic heads and actors to handle environmental heterogeneity.
- The framework utilizes a single-timescale update scheme with Markovian sampling and a joint linear approximation approach to achieve linear speedup with respect to the number of agents.
- The approach incorporates a simulator to generate auxiliary state-action pairs, mitigating distribution mismatches between critic updates and policy gradients in heterogeneous multi-agent environments.

---

[MemLineage: Lineage-Guided Enforcement for LLM Agent Memory](http://arxiv.org/abs/2605.14421)

- MemLineage: introduces a defence for LLM agent memory that attaches cryptographic provenance and derivation lineage to every entry to prevent untrusted content from authorizing sensitive actions.
- The system utilizes a six-module architecture including M1 (Provenance metadata), M2 (Cryptographic binding), M3 (Append-only Merkle log), M4 (Lineage DAG), M5 (Verifier-aware retrieval), and M6 (Sensitive-action gate) to maintain chain-of-custody.
- MemLineage effectively mitigates persistent memory attacks by propagating trust labels through LLM-mediated derivation chains and gating sensitive tool dispatches based on the ancestry of the retrieved memory.

---

[SWE-CHAIN: Benchmarking Coding Agents on Chained Release-Level Package Upgrades](http://arxiv.org/abs/2605.14415)

- SWE-CHAIN: introduces a benchmark for evaluating LLMs on chained release-level package upgrades, utilizing DecompSynth to synthesize grounded specifications from release notes and code diffs.
- The framework employs an Agent Workspace within a Docker Environment to model long-horizon software maintenance, where agents must carry changes forward across consecutive versions.
- To ensure robust evaluation, the pipeline incorporates a Build+Fix Regularization mechanism that allows agents a single controlled repair attempt for execution-level errors.

---

[DermAgent: A Self-Reflective Agentic System for Dermatological Image Analysis with Multi-Tool Reasoning and Traceable Decision-Making](http://arxiv.org/abs/2605.14403)

- DermAgent: introduces a collaborative agentic system that orchestrates specialized vision and language tools within a Plan–Execute–Reflect framework to provide traceable dermatological diagnosis.
- The system utilizes a dual-modality retrieval module, incorporating Case RAG and Guideline RAG, to anchor diagnostic predictions in verifiable external clinical evidence.
- A deterministic Critic module performs post-hoc auditing of the evidence chain using confidence, coverage, and conflict gates to trigger targeted self-correction and mitigate LLM hallucinations.

---

[Agentic Recommender System with Hierarchical Belief-State Memory](http://arxiv.org/abs/2605.14401)

- ARS (Memory-Augmented Agentic Recommender System): introduces a hierarchical memory architecture that abstracts raw user interactions into structured preference chunks and coherent natural language profiles to improve recommendation accuracy.
- The framework utilizes an LLM-based planner to manage a complete memory lifecycle, including extraction, reinforcement, weakening, consolidation, forgetting, and resynthesis, replacing rigid heuristics with adaptive scheduling.
- By decoupling the online ranking path from the offline memory lifecycle, ARS achieves state-of-the-art performance while significantly reducing computational costs through efficient state abstraction.

---

[Coding Agent Is Good As World Simulator](http://arxiv.org/abs/2605.14398)

- Multi-Agent Framework: introduces an agentic system that constructs physics-based world models by iteratively generating and refining executable simulation code through a closed-loop workflow.
- The framework coordinates specialized agents including planning-, code generation-, visual review- and simulation judge-agents to ensure physical consistency and instruction fidelity.
- By utilizing executable code as the world representation, the system enables inspectable dynamics and iterative repair, outperforming traditional video-based world models in physical accuracy.

---

[NEXUS: An Agentic Framework for Time Series Forecasting](http://arxiv.org/abs/2605.14389)

- NEXUS: introduces a multi-agent framework that decomposes time series forecasting into structured stages, utilizing a Historical Context Agent, Macro-Reasoning Agent, Micro-Reasoning Agent, Forecast Synthesizer Agent, and Calibration Agent to synthesize numerical trends with qualitative context.
- The framework employs a dual-resolution approach where the Macro-Reasoning Agent establishes overarching regimes and the Micro-Reasoning Agent identifies granular, event-driven catalysts to improve forecasting accuracy.
- NEXUS incorporates a Calibration Agent that uses a backtesting mechanism to generate master guidelines from past errors, ensuring the system adapts to domain-specific dynamics without requiring manual instruction design.

---

[Data-Augmented Game Starts for Accelerating Self-Play Exploration in Imperfect Information Games](http://arxiv.org/abs/2605.14379)

- DAGS (Data-Augmented Game Starts): introduces a starting-state sampling strategy that initializes self-play episodes at intermediate states from an Offline Dataset to accelerate exploration in imperfect-information games.
- The framework utilizes Data-Augmented Game Starts to bypass long coordinated control sequences, enabling agents to focus on strategically relevant subgames using PPO-Uniform or PPO-EMAg solvers.
- To mitigate potential equilibrium bias introduced by state augmentation, the approach incorporates Multi-task Observation Flags to partition policies into unbiased tasks during training.

---

[Semi-Synchronous Exploration in Dynamic Graphs](http://arxiv.org/abs/2605.14375)

- SSYNC_EXPO: introduces a deterministic algorithm for exploring 1-interval connected dynamic graphs under a semi-synchronous scheduler where an adversary controls agent activation and network topology.
- The paper establishes a tight lower bound on the number of agents required for successful exploration based on the adversary's deactivation power.
- The proposed algorithm utilizes a pipeline strategy and progressive parameter estimation to achieve exploration with O(kDˆ) move complexity and O(max{log n, log p}) memory per agent.

---

[HERCULEAN: An Agentic Benchmark for Financial Intelligence](http://arxiv.org/abs/2605.14355)

- HERCULEAN: introduces a standardized, skill-based benchmark for evaluating LLM agents across four complex financial workflows: Trading, Hedging, Market Insights, and Auditing.
- The framework utilizes a two-level interaction design, where a structured skill layer mediates agent access to MCP-grounded environments, ensuring architecture-agnostic evaluation of reasoning and execution capabilities.
- Experimental results across multiple agent frameworks and LLM backbones reveal that financial agent performance is highly workflow-dependent, with significant gaps in long-horizon reasoning, state management, and deterministic verification.

---

[Distributionally Robust Multi-Task Reinforcement Learning via Adaptive Task Sampling](http://arxiv.org/abs/2605.14350)

- DRATS: introduces a multi-task reinforcement learning algorithm that addresses data imbalance by adaptively prioritizing tasks with the largest return gaps using a minimax optimization objective.
- The framework utilizes a KL-regularized task-sampling distribution to ensure stable, non-zero sampling probabilities across all tasks while focusing on those furthest from their target returns.
- DRATS is compatible with existing multi-task network architectures and demonstrates improved data efficiency and worst-task performance across various robotic manipulation and continuous control benchmarks.

---

[Sub-Band Full Duplex Resource Allocation: A Predictive Deep Reinforcement Learning Approach](http://arxiv.org/abs/2605.14339)

- Bi-LSTM+DDQN: introduces a predictive framework for dynamic sub-band allocation in SBFD systems, integrating 1D-CNN (extracts local spatial features), Bi-LSTM (captures long-term temporal dependencies), DDQN (performs dynamic resource allocation), Experience Replay Memory (stores past interaction events), Online Network (selects optimal scheduling actions), and Target Network (evaluates action stability).
- The framework utilizes a hybrid 1D-CNN and Bi-LSTM architecture to forecast network traffic, providing the DDQN agent with a 22-dimensional state representation for proactive scheduling.
- By dynamically adjusting UL/DL split ratios based on predicted traffic, the system achieves 100% peak UL utilization and significantly reduces queue buildup compared to static baseline configurations.

---

[Are Agents Ready to Teach? A Multi-Stage Benchmark for Real-World Teaching Workflows](http://arxiv.org/abs/2605.14322)

- EduAgentBench: introduces a theory-grounded, source-grounded benchmark for evaluating LLMs across three teaching capability surfaces: pedagogical judgment, situated tutoring, and teaching workflow execution.
- The framework utilizes a pedagogical-insight-driven pipeline to construct 150 quality-controlled tasks that require LLMs to diagnose learner states, adapt scaffolding, and perform institutional actions within simulated learning-management systems.
- Evaluation results reveal that while current LLMs demonstrate proficiency in bounded pedagogical judgment, they frequently fail to maintain coherence in multi-turn tutoring or execute complex, evidence-grounded teaching workflows.

---

[Making OpenAPI Documentation Agent-Ready: Detecting Documentation and REST Smells with a Multi-Agent LLM System](http://arxiv.org/abs/2605.14312)

- Hermes: introduces a multi-agent LLM-based system that detects documentation and REST-related smells in OpenAPI specifications to assess readiness for autonomous agent consumption, utilizing a Smell Detector Agent, Documentation Smell Agents, REST Smell Agents, and a Reduced OpenAPI Representation.
- The system employs an endpoint-centric strategy, isolating operations to generate explainable diagnostic reports that guide remediation efforts and inform strategic AI adoption decisions.
- Empirical evaluation across 600 production endpoints revealed that structural validity in microservice environments does not guarantee semantic readiness for LLMs, necessitating systematic documentation assessment as a foundational prerequisite for AI-agent integration.

---

[Beyond Binary: Reframing GUI Critique as Continuous Semantic Alignment](http://arxiv.org/abs/2605.14311)

- BBCritic: introduces a contrastive framework that reframes GUI critique from binary classification to continuous semantic alignment within a shared Affordance Space.
- The framework utilizes a VLM-based Encoder to map instructions and actions into a shared embedding space, employing InfoNCE Loss to recover hierarchical action structures and resolve affordance collapse.
- BBCritic incorporates a two-stage training curriculum using UI Element Parser and VLM Rollout to generate hard negatives, enabling robust ranking performance on the BBBench Benchmark.

---

[Towards Self-Evolving Agentic Literature Retrieval](http://arxiv.org/abs/2605.14306)

- PaSaMaster: introduces a self-evolving agentic literature retrieval system that separates intent-aware planning from evidence-grounded retrieval and ranking to ensure source authenticity and cost-efficient scaling.
- The system utilizes a Navigator: Planner to iteratively refine search strategies based on feedback, while a Librarian Swarm: Parallel Executor performs retrieval and verification using an Agent-Native Repository and Toolset.
- By treating literature discovery as an intent-paper relevance ranking process rather than generation, PaSaMaster eliminates source hallucinations and achieves superior performance on the PaSaMaster-Bench compared to existing LLM-based retrieval methods.

---

[Web Agents Should Adopt the Plan-Then-Execute Paradigm](http://arxiv.org/abs/2605.14290)

- PTE (Plan-Then-Execute): introduces a secure web agent architecture that separates control flow from untrusted data by committing to a task-specific program before execution, utilizing a Planner, Executor, LLM subroutine, Trusted API, and Web.
- The framework mitigates control-flow hijacking by ensuring that untrusted web content can only influence data values within a fixed execution graph rather than synthesizing new actions.
- Empirical analysis on the WebArena benchmark demonstrates that all tasks are compatible with the PTE paradigm, with over 80% being fully static and the remainder requiring only constrained LLM subroutines.

---

[Watermarking Game-Playing Agents in Perfect-Information Extensive-Form Games](http://arxiv.org/abs/2605.14283)

- KGW watermark adaptation: introduces a method to embed robust, unique signatures into game-playing agents by modifying action probability distributions based on green- and red-list partitioning of available actions.
- The framework utilizes a strategy profile, a game-solving algorithm, a pseudo-random number generator, a green list, a red list, a watermark wrapper, and a statistical test to ensure detectability while bounding the loss in expected utility.
- Experimental results on UCI chess engines demonstrate that the watermark is detectable with a handful of games while maintaining negligible performance degradation.

---

[Auditing Agent Harness Safety](http://arxiv.org/abs/2605.14271)

- HarnessAudit: introduces a harness-centric safety auditing framework that evaluates full execution trajectories across boundary compliance, execution fidelity, and system stability using hidden, agent-independent evidence channels.
- HarnessAudit-Bench: provides a comprehensive stress-testing suite of 210 tasks across eight real-world domains, instantiated in both single-agent and multi-agent configurations with embedded safety constraints.
- The research demonstrates that task completion is misaligned with safe execution, with safety risks varying significantly across domains, agent roles, and multi-agent collaboration structures.

---

[Heuristic Pathologies and Further Variance Reduction via Uncertainty Propagation in the AIVAT Family of Techniques](http://arxiv.org/abs/2605.14261)

- AIVAT: introduces a cautionary analysis of heuristic value functions in variance reduction, demonstrating that unfixed heuristics can lead to pathological variance or p-hacking.
- The paper proposes propagating heuristic uncertainty to the estimate level using inverse-variance weighting to achieve further variance reduction in multiagent evaluation.
- Experimental results on poker data demonstrate that the proposed uncertainty-aware approach yields a 43.0% reduction in the number of samples required for statistical significance.

---

[Hypergraph Enterprise Agentic Reasoner over Heterogeneous Business Systems](http://arxiv.org/abs/2605.14259)

- HEAR: introduces an enterprise agentic reasoner that grounds LLMs within a Stratified Hypergraph Ontology to resolve heterogeneous data dependencies and enforce n-ary business constraints for multi-hop reasoning.
- The architecture integrates an Agentic Reasoning Loop with a Stratified Hypergraph Ontology, utilizing a Graph Layer for virtualized data access and a Hyperedge Layer for binding n-ary business axioms and procedural protocols.
- HEAR achieves high accuracy and adaptive execution efficiency on complex supply-chain tasks by dynamically orchestrating ontology tools to navigate heterogeneous systems without requiring LLM retraining.

---

[Latency-Quality Routing for Functionally Equivalent Tools in LLM Agents](http://arxiv.org/abs/2605.14241)

- LQM-CONTEXTROUTE: introduces a contextual bandit router for functionally equivalent tool providers that optimizes a renewal-reward rate to prevent latency from compensating for poor answer quality.
- The framework utilizes a LinUCB quality head, an EMA latency estimator, and LLM-as-judge feedback to adaptively route queries based on real-time load and provider-specific performance.
- By treating latency as a service-cycle cost rather than an additive penalty, the approach effectively mitigates performance degradation in heterogeneous provider pools under non-stationary load conditions.

---

[Quantum Advantage in Multi Agent Reinforcement Learning](http://arxiv.org/abs/2605.14235)

- QMARL (Quantum Multi Agent Reinforcement Learning): introduces a decentralized framework utilizing Variational Quantum Circuit (VQC) actors and shared entangled states to achieve coordination in multi-agent systems.
- The framework employs Centralised Training with Decentralized Execution (CTDE) to enable agents to learn policies that exploit quantum entanglement for implicit coordination without runtime communication.
- Experimental results demonstrate that entanglement provides a provable quantum advantage in non-local games, while VQC expressiveness and hybrid actor-critic architectures offer performance benefits in cooperative navigation tasks.

---

[MetaAgent-X : Breaking the Ceiling of Automatic Multi-Agent Systems via End-to-End Reinforcement Learning](http://arxiv.org/abs/2605.14212)

- MetaAgent-X: introduces an end-to-end reinforcement learning framework that jointly optimizes Designer- and Executor-agents to break the performance ceiling of static multi-agent systems.
- The framework utilizes Executor-Designer Hierarchical Rollout to enable structured trajectory collection and accurate credit assignment across roles.
- Stagewise Co-evolution decouples the learning stages of the Designer and Executor to improve training stability and scalability during the joint optimization process.

---

[ASH: Agents that Self-Hone via Embodied Learning](http://arxiv.org/abs/2605.14211)

- ASH: introduces a dynamic bootstrapping framework that enables agents to learn long-horizon embodied policies by iteratively retrieving and learning from relevant internet video without manual reward engineering.
- The system utilizes an Inverse Dynamics Model to generate pseudo-actions from unlabeled video and a dual-memory architecture to maintain both reactive short-term control and long-term task progression.
- ASH demonstrates superior performance in complex, open-ended environments like Pokémon Emerald and The Legend of Zelda by autonomously identifying key moments and adapting to new visual dynamics through self-improvement.

---

[SimPersona: Learning Discrete Buyer Personas from Raw Clickstreams for Grounded E-Commerce Agents](http://arxiv.org/abs/2605.14205)

- SimPersona: introduces a framework that learns discrete buyer personas from raw clickstreams using a behavior-aware VQ-VAE and grounds them in LLM agents through a two-stage SFT process.
- The framework utilizes a Data Pipeline to extract behavioral features and generate multi-turn agent traces, enabling the LLM Agent to simulate realistic, merchant-specific buyer population distributions.
- By decoupling persona grounding from action learning, the Two-Stage SFT approach ensures robust agent performance and generalization across unseen storefronts without requiring per-store calibration.

---

[MMSkills: Towards Multimodal Skills for General Visual Agents](http://arxiv.org/abs/2605.13527)

- MMSkills: introduces a framework for representing, generating, and using reusable multimodal procedural knowledge to improve visual decision-making in agents.
- The framework utilizes a multimodal skill package containing textual procedures, runtime state cards, and multi-view keyframes to provide state-aware guidance.
- A branch-loaded mechanism isolates skill-environment grounding in a temporary branch, returning distilled structured guidance to the main agent to avoid context pressure and visual anchoring.

---

[Speculative Interaction Agents: Building Real-Time Agents with Asynchronous I/O and Speculative Tool Calling](http://arxiv.org/abs/2605.13360)

- Speculative Interaction Agents: introduces an event-driven architecture that decouples agent reasoning from user and environment streams to enable real-time responsiveness.
- The framework utilizes Asynchronous I/O to overlap reasoning with streaming inputs and Speculative Tool Calling to manage task execution while awaiting full information.
- A clock-based training methodology is employed to adapt LLMs for continuous reasoning and error correction in dynamic, latency-sensitive agentic workflows.

---

[D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models](http://arxiv.org/abs/2605.13276)

- D-VLA: introduces a high-concurrency distributed reinforcement learning framework that utilizes Plane Decoupling to isolate high-frequency simulation data from low-frequency weight control, effectively eliminating resource contention.
- The framework employs a four-thread Swimlane pipeline to enable full parallel overlap of sampling, inference, gradient computation, and parameter distribution, significantly enhancing throughput for large-scale VLA models.
- By integrating dual-pool VRAM management and topology-aware replication, D-VLA resolves memory fragmentation and communication bottlenecks, achieving stable linear speedup for trillion-parameter embodied agents.

---

[Residual Reinforcement Learning for Robot Teleoperation under Stochastic Delays](http://arxiv.org/abs/2605.15480)

- DR-RL: introduces a hybrid control framework that integrates an LSTM-based state estimator with a residual RL policy to ensure stable teleoperation under stochastic communication delays.
- The framework utilizes an autoregressive LSTM-based state estimator to provide continuous state predictions, effectively mitigating the partial observability caused by time-varying network delays.
- A residual RL agent, trained via Soft Actor-Critic, computes corrective torque terms to compensate for unmodeled dynamics and tracking errors that the nominal controller cannot suppress.

---

[EgoExo-WM: Unlocking Exo Video for Ego World Models](http://arxiv.org/abs/2605.15477)

- EgoExo-WM: introduces a framework that leverages exocentric video to train egocentric world models by converting third-person observations into action-aligned egocentric visual experiences.
- The framework utilizes a Body Pose Predictor and an Exocentric to Egocentric Converter to ground video synthesis in human kinematics, enabling the integration of large-scale internet video for training.
- EgoExo-WM incorporates a wrist-position consistency objective and a latent-space world model to improve future state prediction and goal-conditioned planning for embodied agents.

---

[Validated Hypotheses as a Lens for Human-Likeness Evaluation in AI Agents](http://arxiv.org/abs/2605.15473)

- HUMANSTUDY-BENCH: introduces a principled, diagnostic platform that evaluates LLM human-likeness by comparing agent behavior against validated social science hypotheses using PAS and ECS metrics.
- The framework utilizes a human-in-the-loop pipeline to reconstruct published experimental protocols, enabling objective and scalable assessment of agent performance across diverse cognitive and social domains.
- Empirical results across 10 LLMs demonstrate that agent design significantly influences alignment, revealing that current models often fail to replicate human behavioral patterns in diagnostic, non-monotonic ways.

---

[Estimated Dynamic Equilibrium Model: Supply and Demand as a Sample Path of a Stochastic Process](http://arxiv.org/abs/2605.15472)

- EDEM (Estimated Dynamic Equilibrium Model): introduces an agent-based framework that models market supply and demand as a coupled stochastic process driven by heterogeneous, error-prone agent valuations.
- The framework identifies an order-statistic mechanism where max-bid clearing and per-epoch price feedback generate persistent positive price drift without requiring behavioral assumptions like investor optimism.
- By varying parameters such as divergence of opinion, seller patience, and population balancing, the model reproduces diverse market regimes including stable equilibria, business cycles, and runaway bubbles.

---

[DRUGSAGE: Self-evolving Agent Experience for Efficient State-of-the-Art Drug Discovery](http://arxiv.org/abs/2605.15461)

- DRUGSAGE: introduces an agentic framework that accumulates and reuses cross-task experience to efficiently build state-of-the-art drug discovery models.
- The framework integrates a persistent memory system—comprising Solution Memory, Refinement Memory, and Execution Memory—into a Monte Carlo Tree Search loop to guide model development and enable zero-test-time solution transfer.
- By leveraging cross-task evidence, DRUGSAGE significantly reduces the search budget and LLM API costs while outperforming existing baselines in molecular property prediction tasks.

---

[Runtime-Structured Task Decomposition for Agentic Coding Systems](http://arxiv.org/abs/2605.15425)

- RSTD: introduces an architectural pattern that externalizes task structure into executable control flow to enable selective retry at subtask granularity.
- The framework utilizes a Decomposition Engine, Typed LLM Judgment Operators, Schema Validation, and a State Manager to isolate failures and prevent cascading re-execution.
- Empirical evaluation demonstrates that RSTD achieves significant retry cost reductions compared to monolithic and static decomposition approaches by enabling runtime-controlled branching.

---

[Social-Mamba: Socially-Aware Trajectory Forecasting with State-Space Models](http://arxiv.org/abs/2605.15424)

- Social-Mamba: introduces a trajectory forecasting architecture that reformulates unstructured social interactions as structured sequential processes using selective state-space models.
- The framework utilizes a Cycle Mamba (CM) block to enable continuous bidirectional information flow, ensuring that the forward pass is explicitly conditioned on the future context.
- Social-Mamba organizes agents into an egocentric grid and employs social triplet factorization to capture temporal, egocentric, and goal-centric dynamics efficiently.

---

[Beyond Partner Diversity: An Influence-Based Team Steering Framework for Zero-Shot Human-Machine Teaming](http://arxiv.org/abs/2605.15400)

- IBTS: introduces a framework for zero-shot human-machine teaming that combines partner diversity with learned coordination structure to improve performance in sparse-reward environments.
- The framework utilizes Influence-Shaping to incentivize supportive behaviors, a Trajectory-Conditioned Team Predictor to recognize coordination patterns, and Team Steering to guide agents toward high-performing interaction modes.
- Evaluations across simulated, synthetic LLM-partner, and human-subject studies demonstrate that IBTS outperforms diversity-focused baselines in both dyadic and group human-machine teaming settings.

---

[Ensemble Monitoring for AI Control: Diverse Signals Outweigh More Compute](http://arxiv.org/abs/2605.15377)

- Ensemble Monitoring for AI Control: introduces a framework for improving AI safety by aggregating diverse signals from multiple LLM-based monitors to detect misaligned code actions.
- The approach utilizes both prompt-based and fine-tuned monitors to generate complementary suspicion scores, which are then aggregated to outperform individual monitors and homogeneous ensembles.
- The research demonstrates that monitor diversity, rather than increased compute scale, is the primary driver of performance gains in AI control, with small ensembles capturing most of the potential safety improvements.

---

[Belief Engine: Configurable and Inspectable Stance Dynamics in Multi-Agent LLM Deliberation](http://arxiv.org/abs/2605.15343)

- BE (Belief Engine): introduces an auditable simulation-control layer that decouples belief maintenance from generative reasoning to enable explicit, parameterised control over stance dynamics in multi-agent LLM deliberation.
- The framework utilizes a Bayesian-style log-odds update rule, controlled by evidence uptake and prior anchoring parameters, to maintain a persistent, proposition-level belief state that conditions LLM-generator responses.
- By separating argument extraction, evidence judgement, and structured memory from generation, the architecture provides an inspectable audit trail for agent stance changes, addressing the limitations of implicit prompt-based belief revision.

---

[Minerva-Ego: Spatiotemporal Hints for Egocentric Video Understanding](http://arxiv.org/abs/2605.15342)

- Minerva-Ego: introduces a benchmark for complex egocentric video reasoning that pairs multi-step questions with dense, human-annotated spatiotemporal reasoning traces.
- The framework utilizes spatiotemporal hints, including object masks and temporal selection, to guide LLMs in identifying relevant objects and time segments within long-form egocentric videos.
- Evaluations demonstrate that frontier LLMs struggle with perceptual grounding in egocentric settings, and that explicit spatiotemporal highlighting significantly improves reasoning performance.

---

[Hidden in Memory: Sleeper Memory Poisoning in LLM Agents](http://arxiv.org/abs/2605.15338)

- Sleeper Memory Poisoning framework: introduces a delayed security attack where an adversary manipulates external content to force an LLM to store a fabricated memory that later influences future interactions.
- The attack pipeline involves three stages: injection of adversarial content into persistent memory, retrieval of the poisoned memory in a subsequent session, and negative impact on the assistant's behavior or agentic actions.
- Empirical evaluations across stateful LLMs demonstrate high success rates for universal poisoning payloads, highlighting the vulnerability of memory-augmented systems to long-term adversarial influence.

---

[From I/O to Code with Discovery Agent](http://arxiv.org/abs/2605.15334)

- DIO-Agent: introduces a discovery agent for IO2Code that utilizes Curriculum-wise Evolution, Transformation Priority Premise, and Error-Grounded Feedback to synthesize programs from input-output behavior through an LLM-driven Evolutionary Loop.
- The framework employs a Sandbox Evaluator to provide structured debugging evidence, guiding the LLM-based mutation operator to navigate the program space from simple to complex constructs.
- DIO-Agent incorporates island-based search and an optional Multimodal LLM Tool to enhance generalizability and performance across diverse algorithmic, geometric, and multimodal tasks.

---

[Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning](http://arxiv.org/abs/2605.15315)

- LaMR (Latent Multi-Rubric): introduces a structured pruning framework that decomposes code relevance into interpretable semantic and dependency dimensions to optimize context for LLMs.
- The framework utilizes a Shared Backbone Feature Fusion, MoE Gate, CRFsem, CRFdep, Fused CRF, and Viterbi Decoding to dynamically balance semantic evidence and structural support while filtering distracting noise.
- LaMR improves token efficiency and task performance for LLMs by preserving self-contained evidence-support units through AST-derived supervision and rubric-specific transition dynamics.

---

[Solvita: Enhancing Large Language Models for Competitive Programming via Agentic Evolution](http://arxiv.org/abs/2605.15301)

- Solvita: introduces a multi-agent framework that enables continuous, experience-driven evolution for frozen LLMs in competitive programming by coupling a Planner, Solver, Oracle, and Hacker with trainable, graph-structured Knowledge Networks.
- The framework utilizes a closed-loop system where failure signals from the Oracle and Hacker are recast as reinforcement learning updates to the agents' Knowledge Networks via a Contextual-Bandit Policy.
- Solvita achieves state-of-the-art performance by accumulating transferable reasoning experience across tasks, effectively doubling the accuracy of single-pass LLM baselines without requiring weight updates to the underlying LLM backbone.

---

[Autonomous Intelligent Agents for Natural-Language-Driven Web Execution with Integrated Security Assurance](http://arxiv.org/abs/2605.15281)

- Autonomous Intelligent Agent Framework: introduces an AI-driven testing system that utilizes a five-strategy enhancement pipeline to improve web test reliability and security validation through a decoupled containerized worker architecture.
- The framework employs a vision-enabled LLM within an agentic perceive-reason-act loop to translate natural language instructions into browser-based functional tests and security probes.
- By separating stateless orchestration from stateful browser execution, the system achieves high success rates in complex web environments while enabling automated security testing aligned with OWASP standards.

---

[Training on Documents About Monitoring Leads to CoT Obfuscation](http://arxiv.org/abs/2605.15257)

- CoT Obfuscation Framework: introduces a methodology where models trained on documents describing monitoring systems learn to obfuscate their reasoning traces to evade detection by a CoT monitor.
- The research demonstrates that monitor-aware models consistently achieve higher rates of undetected misbehavior compared to unaware controls by actively suppressing or reframing reasoning content.
- The study identifies CoT controllability as a strong predictor of obfuscation success and shows that monitor-aware models learn to reward-hack faster than unaware controls during reinforcement learning.

---

[Assistance to Autonomy: A Systematic Literature Review of Agentic AI across the Software Development Life Cycle](http://arxiv.org/abs/2605.15245)

- Agentic AI SLR: introduces a domain-agnostic multi-agent screening pipeline that utilizes an Assistant, Evaluator, LLM Team, Quality Control, Screening, and Finalizing components to automate systematic literature reviews.
- The paper identifies output verifiability as the primary enabler for industrial adoption of agentic AI, with the Planner-Executor-Reviewer pattern serving as the dominant architectural framework.
- Industrial mitigation strategies for agentic AI challenges consistently focus on confining agent actions to bounded, verifiable spaces to ensure reliability and consistency.

---

[A3D: Agentic AI flow for autonomous Accelerator Design](http://arxiv.org/abs/2605.15237)

- A3D: introduces an end-to-end agentic framework that automates hardware accelerator design by partitioning tasks among specialist agents, utilizing deterministic tools, and employing adversarial verification loops.
- The framework integrates an agentic RAG pipeline to bridge the gap between LLM knowledge and proprietary EDA tool requirements, enabling autonomous refactoring and synthesis of complex scientific kernels.
- A3D achieves high-reliability automation by combining task decomposition, tool augmentation, and iterative verifier loops, successfully generating Pareto-optimal accelerator designs from complex C++ and CUDA codebases without human intervention.

---



#### 13th May 2026


[IdeaForge: A Knowledge Graph-Grounded Multi-Agent Framework for Cross-Methodology Innovation Analysis and Patent Claim Generation](http://arxiv.org/abs/2605.13311)

- IdeaForge: introduces a knowledge graph-grounded multi-agent framework that treats innovation methodologies as heterogeneous reasoning operators acting over a shared persistent FalkorDB Knowledge Graph.
- The framework utilizes TRIZ Agent, Design Thinking Agent, and SCAMPER Agent to contribute structured entities to the graph, which are then synthesized by an Embedding Synthesis Agent to identify cross-methodology convergence.
- An InnovationScore Module ranks the resulting claims based on convergence, methodology diversity, and strength, enabling a Patent Agent to generate traceable, grounded patent drafts.

---



[Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations](http://arxiv.org/abs/2605.14175)

- Grounded Continuation: introduces a runtime verifier that maintains an explicit dependency graph to ensure LLM outputs trace back to prior conversation commitments.
- The framework utilizes an LLM Interpreter to classify utterances into operations, a Symbolic Engine to track dependencies, and a Context Renderer to feed structured state back into the LLM.
- This approach enables linear-time verification of conversation grounding, effectively catching stale-premise errors that standard LLM-only baselines frequently miss.

---

[BOOKMARKS: Efficient Active Storyline Memory for Role-playing](http://arxiv.org/abs/2605.14169)

- BOOKMARKS: introduces a search-based memory framework for RPAs that utilizes a Proposal Module, Matching Module, Memory Bank, Synchronization Operator, and Grounding Context to maintain task-relevant information efficiently.
- The framework improves long-horizon consistency by actively proposing queries and passively updating only the necessary bookmarks, avoiding the computational overhead of full-storyline processing.
- BOOKMARKS outperforms existing retrieval-based and profile-based memory methods by providing precise, task-specific grounding through incremental synchronization of reusable memory anchors.

---

[Agentic Systems as Boosting Weak Reasoning Models](http://arxiv.org/abs/2605.14163)

- Verifier-backed committee search framework: introduces a formal model of inference-time boosting for LLMs by decomposing reasoning into proposal coverage, local identifiability, progress, and diversity.
- The framework utilizes a Proposer to generate candidates, a Critic to perform binary filtering, and a Comparator to rank remaining candidates, effectively converting latent proposal-pool capability into realized solve rates.
- Empirical results on SWE-bench Verified demonstrate that this orchestration approach allows weak LLMs to match the performance of significantly stronger standalone models by optimizing selection over cached proposal pools.

---

[EXPLOITBENCH: A Capability Ladder Benchmark for LLM Cybersecurity Agents](http://arxiv.org/abs/2605.14153)

- ExploitBench: introduces a capability-graded benchmark that decomposes exploitation into 16 measurable flags across five tiers, verified by deterministic oracles to measure LLM agent progress on hardened production targets.
- The framework utilizes an MCP server, CompositeGrader, Coverage grader, Diff grader, Primitive grader, Engine instrumentation, Environment builder, Agent runner, LiteLLM, and Audit catalog to provide a standardized, reproducible evaluation environment.
- The research demonstrates a sharp capability split between publicly deployed LLMs and private-frontier models, where only the latter reliably achieve arbitrary code execution on hardened V8 targets.

---

[Distribution-Aware Algorithm Design with LLM Agents](http://arxiv.org/abs/2605.14141)

- Distribution-Aware Algorithm Design with LLM Agents: introduces a framework that leverages LLM agents to infer reusable solver hints from public samples, which are then compiled into specialized deployment solvers to optimize both solution quality and execution runtime.
- The framework utilizes a three-stage construction process involving a Hypothesis (Hc), an Analysis Program (Ac), and a Deployment Solver (sc), which are refined through a diversity-preserving beam search to discover distribution-specific computational shortcuts.
- Empirical results across 21 combinatorial-optimization distributions demonstrate that synthesized solvers significantly outperform heuristic and exact baselines in runtime while maintaining high solution quality by replacing ambient search with distribution-specific computation.

---

[ClawForge: Generating Executable Interactive Benchmarks for Command-Line Agents](http://arxiv.org/abs/2605.14133)

- ClawForge: introduces a generator-backed benchmark framework for evaluating LLM agents on command-line workflows under state conflict, utilizing Scenario Template, Grounded Slots, State-Mode Selection, Reference Command Synthesis, Validator Generation, Interactive Environment, Command Router, Normalized Evaluation State, and Result-First Evaluator.
- The framework enables systematic testing of LLM agents by initializing tasks with pre-existing partial, stale, or conflicting artifacts, requiring agents to perform state-aware judgments rather than simple command imitation.
- Evaluations across seven frontier models demonstrate that ClawForge effectively separates agent capabilities, particularly in scenarios requiring state repair, replacement, and multi-source decision-making.

---

[Reinforcement Learning for Tool-Calling Agents in Fast Healthcare Interoperability Resources (FHIR)](http://arxiv.org/abs/2605.14126)

- SkyRL: introduces a post-training pipeline for clinical LLM agents that uses execution-grounded reinforcement learning to improve multi-turn reasoning over structured FHIR clinical graphs.
- The framework utilizes a multi-turn CodeAct-style agent that interacts with a FHIR server via retrieval and Python tools to perform schema discovery and data aggregation.
- By applying GRPO-based post-training with execution-grounded rewards, the system enables smaller open-weight models to outperform larger closed-source models on complex clinical question-answering tasks.

---

[Mini-JEPA Foundation Model Fleet Enables Agentic Hydrologic Intelligence](http://arxiv.org/abs/2605.14120)

- Mini-JEPA: introduces a fleet of small, sensor-specialized foundation models that leverage a shared Vision Transformer backbone and I-JEPA training recipe to provide targeted hydrologic intelligence.
- The system utilizes a router LLM that consults per-modality reference cards to select the most relevant Mini-JEPA specialists for specific natural-language hydrologic queries.
- Evaluation demonstrates that this routed fleet significantly outperforms planetary-scale generalist models on single-modality tasks while remaining efficient enough for deployment on commodity hardware.

---

[Privacy Preserving Multi Agent Path Finding](http://arxiv.org/abs/2605.14119)

- kPPMAPF (k-Privacy Preserving Multi Agent Path Finding): introduces a framework that preserves privacy by adding mock agents to the planning process, ensuring that no agent can identify the exact location of others within a minimum of k possible values.
- ePPMAPF (Execution-level Privacy Preserving Multi Agent Path Finding): extends the framework to prevent privacy leakage during execution by incorporating Field-of-View (FoV) constraints into the planning process using fPP and PPfPP.
- PPfPP (Post-Processing fPP): improves solution quality by identifying safe zones where agents can re-plan locally without violating privacy or collision constraints.

---

[ProtoMedAgent: Multimodal Clinical Interpretability via Privacy-Aware Agentic Workflows](http://arxiv.org/abs/2605.14113)

- ProtoMedAgent: introduces a framework that formalizes multimodal clinical reporting as an iterative, zero-gradient test-time optimization problem over a strict neuro-symbolic bottleneck to ensure evidence-grounded documentation.
- The framework utilizes a suite of dedicated agents including a Perception Agent, Tabular Agent, Memory Agent, and Verification Agent to translate retrieved evidence into a fully grounded and verifiable clinical report.
- By employing a semantic privacy gate and a Scribe-Critic loop, the system mathematically precludes unsupported narrative claims and mitigates privacy risks without requiring gradient updates to the underlying frozen backbone.

---

[Modeling Bounded Rationality in Drug Shortage Pharmacists Using Attention-Guided Dynamic Decomposition](http://arxiv.org/abs/2605.14111)

- Attention-Guided Dynamic Decomposition framework: introduces a computational model for drug shortage management that dynamically decomposes high-dimensional state spaces into a focused subset for intensive reasoning and a secondary subset for monitoring.
- The framework utilizes an Expert Agent with predefined attention weights and a Learner Agent that optimizes attention allocation using a REINFORCE-style gradient update to maintain stable performance under uncertainty.
- By restricting planning to high-urgency drugs, the approach achieves significant computational efficiency and prevents stockouts in complex, partially observable healthcare supply chain scenarios.

---

[SToRe3D: Sparse Token Relevance in ViTs for Efficient Multi-View 3D Object Detection](http://arxiv.org/abs/2605.14110)

- SToRe3D: introduces a planner-aligned sparsity framework that jointly prunes 2D image tokens and 3D object queries using lightweight relevance heads and store-reactivate buffers to reduce inference latency.
- The framework utilizes a future interaction corridor to supervise relevance, ensuring that compute is prioritized for agents critical to ego-vehicle motion planning.
- By caching low-relevance features in buffers for selective reactivation, SToRe3D achieves real-time performance on ViT-based 3D detection with minimal accuracy loss.

---

[ChromaFlow: A Negative Ablation Study of Orchestration Overhead in Tool-Augmented Agent Evaluation](http://arxiv.org/abs/2605.14102)

- ChromaFlow: introduces a tool-augmented autonomous reasoning framework that evaluates the impact of orchestration overhead on agent reliability.
- The system utilizes an Optimus supervisory controller to manage execution paths, while the reliability layer monitors operational noise and enforces performance gates.
- The study demonstrates that aggressive orchestration can increase operational noise and decrease accuracy, highlighting the necessity of rigorous evaluation protocols for LLM-based agents.

---

[SkillFlow: Flow-Driven Recursive Skill Evolution for Agentic Orchestration](http://arxiv.org/abs/2605.14089)

- SkillFlow: introduces a flow-based framework for agentic orchestration that utilizes a trainable Supervisor, a dynamic skill library, and a frozen Executor to automate task completion through multi-turn interaction.
- The framework employs TTB (Tempered Trajectory Balance) to perform reward-proportional trajectory sampling, which preserves diverse orchestration strategies and prevents mode collapse.
- SkillFlow incorporates a recursive skill evolution mechanism that uses flow diagnostics to autonomously determine when to evolve, what skills to create or prune, and where decision gaps exist.

---

[CRANE: Constrained Reasoning Injection for Code Agents via Nullspace Editing](http://arxiv.org/abs/2605.14084)

- CRANE: introduces a training-free parameter-editing method that injects reasoning behavior from a Thinking checkpoint into an Instruct backbone while preserving tool-use protocols using Magnitude Thresholding, Conservative Taylor Gate, and Graduated Sigmoidal Projection.
- The framework treats the Thinking–Instruct delta as a candidate edit pool, denoising it and projecting out format-critical directions to maintain agentic protocol fidelity.
- Empirical results across Roo-Eval, SWE-bench-Verified, and Terminal-Bench v2 demonstrate that CRANE improves task success while maintaining efficient, Instruct-like token usage.

---

[Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](http://arxiv.org/abs/2605.14057)

- ICA (Inquisitive Conversational Agent): introduces a dual-agent hierarchical reinforcement learning framework designed to emulate judicial questioning patterns by splitting inquisitive reasoning between an Appraisal Agent and a Hierarchical Dialogue Agent.
- The framework utilizes a three-level action taxonomy embedded in a Poincaré space to optimize dialogue strategies for information elicitation in high-stakes legal domains.
- The system incorporates a multi-component reward function—balancing goal relevance, lexical novelty, and answer succinctness—to steer conversations toward uncovering critical information.

---

[Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](http://arxiv.org/abs/2605.14054)

- MoCA (Modality-aware Credit Assignment): introduces a reinforcement learning framework that resolves the perception-reasoning "seesaw effect" by explicitly decoupling generation into interleaved Perception Actions and Reasoning Actions, enabling targeted supervision.
- The framework utilizes Perception Verification to reward visual grounding via a Blindfolded Text Reasoner and Structured Verbal Verification to provide low-variance outcome rewards for free-form responses.
- By routing granular rewards to specific components, MoCA identifies and corrects "bad seeing" versus "bad thinking" errors, significantly improving performance across perception-intensive and reasoning-intensive tasks.

---

[SPIN: Structural LLM Planning via Iterative Navigation for Industrial Tasks](http://arxiv.org/abs/2605.14051)

- SPIN: introduces a planning wrapper for LLM agents that enforces a strict Directed Acyclic Graph (DAG) contract and utilizes a simulator-critic loop to optimize execution efficiency through early stopping.
- The framework employs a validator to ensure machine-consumable plan structures and a prefix-based evaluation policy to terminate workflows once sufficient progress is achieved.
- Empirical results on industrial benchmarks demonstrate that SPIN reduces downstream execution burden, including tool calls and API usage, while improving task-level accomplishment rates.

---

[Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use](http://arxiv.org/abs/2605.14038)

- Two-stage cognition-execution modeling framework: introduces a model-adaptive definition of tool necessity grounded in empirical performance to diagnose the knowing-doing gap in LLMs.
- The framework decomposes tool use into an internal cognition stage and an execution stage, revealing that failures predominantly occur during the transition from cognition to action.
- By probing hidden states, the research demonstrates that while both cognition and action are linearly decodable, their probe directions become nearly orthogonal in the late-layer, last-token regime.

---

[Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility](http://arxiv.org/abs/2605.14037)

- SP-KV (Self-Pruned Key-Value Attention): introduces a learned sparse-write mechanism that selectively retains key-value pairs in the persistent KV cache based on predicted future utility to reduce memory footprint and improve decoding speed.
- The framework utilizes a lightweight KV utility predictor to assign scores to key-value pairs, ensuring only high-utility tokens are stored in the persistent cache while maintaining a local buffer for recent interactions.
- SP-KV is trained jointly with the LLM using next-token prediction, enabling dynamic sparsification that adapts to input sequences and provides structured sparsity patterns for designing hybrid local-global attention architectures.

---

[From Descriptive to Prescriptive: Uncover the Social Value Alignment of LLM-based Agents](http://arxiv.org/abs/2605.14034)

- SoVA (Social Value Alignment): introduces a value-based framework that employs GraphRAG to convert psychological theories into prescriptive instructions for steering LLM-based agents.
- The framework utilizes Maslow’s Hierarchy of Needs, Plutchik’s Wheel of Emotions, and Aristotle’s Virtues as seed principles to construct a knowledge graph that guides agent behavior in social dilemmas.
- SoVA improves social value alignment by retrieving community-specific instructions, though it may introduce trade-offs in creative generation and multi-turn conversational coherence.

---

[Sheaf-Theoretic Transport and Obstruction for Detecting Scientific Theory Shift in AI Agents](http://arxiv.org/abs/2605.14033)

- STO framework: introduces a finite sheaf-theoretic diagnostic for AI agents to distinguish between representational transport via deformation and theory extension via structural reorganization.
- The framework utilizes Representational Constellations as local charts, where an Obstruction Functional evaluates whether models glue across contexts or require an extension of the representational language.
- Experimental results on physics-inspired transition families demonstrate that the obstruction-based ranking effectively identifies necessary theory shifts while maintaining stability under noise and stress.

---

[Case Studies and Reflections on Agentic Software Engineering for Rapid Development of Digital Music Instruments](http://arxiv.org/abs/2605.14016)

- ASE: introduces a methodology for developing audio software by leveraging agentic LLMs to automate planning, code generation, and build management within the JUCE framework.
- The research demonstrates that ASE can effectively lower barriers to entry for non-programmers and improve software longevity by translating legacy music instruments into modern, interoperable C++ plugins.
- Through three case studies, the paper evaluates the efficacy of agentic workflows in re-creating, translating, and modernizing digital music instruments using natural language prompts and iterative testing.

---

[PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts](http://arxiv.org/abs/2605.14002)

- PolitNuggets: introduces a benchmark for evaluating agentic information synthesis by constructing political biographies through a Supervisor/Searcher/Archive/Coder/FactNet/Judge LRM framework.
- The system utilizes a Supervisor-Searcher architecture with an Archive memory component to enable long-horizon reasoning and evidence-grounded discovery.
- FactNet provides an evidence-conditional evaluation protocol that validates candidate nuggets against retrieved sources to measure discovery, fine-grained accuracy, and efficiency.

---

[COLLIDER-BENCH: Benchmarking AI Agents with Particle Physics Analysis Reproduction](http://arxiv.org/abs/2605.13950)

- COLLIDER-BENCH: introduces a benchmark for evaluating autonomous LLM agents on long-horizon scientific tasks by reproducing Large Hadron Collider analyses using public software and papers.
- The framework requires agents to construct executable simulation-and-selection pipelines, producing binned event yields that are compared against hidden reference values using fidelity metrics.
- An LLM judge audits the agent's workspace and execution trace to distinguish between legitimate scientific attempts, incomplete runs, and fabricated results.

---

[EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents](http://arxiv.org/abs/2605.13841)

- EVA-Bench (End-to-end Voice Agent Evaluation Benchmark): introduces, an end-to-end evaluation framework for voice agents that jointly addresses simulation fidelity and measurement comprehensiveness using User Simulator, Voice Agent, Tool Executor, Simulator Validation, Quality Measurements, and Diagnostic Metrics.
- The framework orchestrates parallel bot-to-bot audio sessions to evaluate cascade, hybrid, and S2S architectures under identical conditions, including controlled acoustic and behavioral perturbations.
- EVA-Bench provides composite metrics EVA-A (Accuracy) and EVA-X (Experience) alongside a multi-trial consistency framework (pass@1, pass@k, pass^k) to enable direct cross-architecture comparison of voice agents.

---

[Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update Your Weights](http://arxiv.org/abs/2605.13839)

- TFLOW (Thought Flow): introduces a weight-space communication paradigm for multi-agent LLMs that replaces natural-language message exchange with transient, instance-specific LoRA weight perturbations.
- The framework utilizes a trainable parameter generator to map sender hidden states into low-rank LoRA factors, which are fused and injected into a frozen receiver agent to enable efficient, context-aware collaboration.
- By eliminating the need for auxiliary text-based messages, TFLOW significantly reduces KV-cache memory usage and prefill overhead while maintaining competitive performance across reasoning, coding, and knowledge benchmarks.

---

[Training Long-Context Vision-Language Models Effectively with Generalization Beyond 128K Context](http://arxiv.org/abs/2605.13831)

- MMProLong: introduces a systematic study of long-context continued pre-training for LVLMs, utilizing Document Pool, OCR Expert, Segment Sampling, and QA Generator to construct effective training data.
- The framework employs Long-document VQA and OCR Transcription tasks within a LongPT Recipe to enhance context window scaling from 32K to 128K tokens.
- The implementation leverages FlashAttention, Sequence Parallelism, and FSDP to achieve efficient training, demonstrating generalization to 512K context lengths and broader multimodal tasks.

---

[History Anchors: How Prior Behavior Steers LLM Decisions Toward Unsafe Actions](http://arxiv.org/abs/2605.13825)

- HISTORYANCHOR-100: introduces a benchmark for evaluating how LLMs, acting as decision-making agents, are influenced by prior history logs when choosing subsequent actions at a free-choice node.
- The framework demonstrates that aligned LLMs, when provided with a consistency-demanding system prompt and a sequence of unsafe prior actions, frequently shift from safe to unsafe decision-making.
- The research identifies an inverse-scaling pattern where more capable, aligned LLMs are often more susceptible to this behavioral-consistency pressure than their smaller counterparts.

---

[Harnessing Agentic Evolution](http://arxiv.org/abs/2605.13821)

- AEVO: introduces a harnessed meta-editing framework that treats agentic evolution as an interactive environment, where a Meta Agent observes the accumulated Evolution Environment and edits the underlying Evolution Mechanism to steer future search.
- The framework utilizes a Protected Evaluator and a structured Workspace to maintain a stable interface, allowing the Meta Agent to perform coarse-grained interventions through a two-phase loop of meta-editing and evolution segments.
- By decoupling the evolution mechanism from candidate generation, AEVO enables persistent, long-horizon improvement across both procedure-based and agent-based evolution paradigms.

---

[EvoGround: Self-Evolving Video Agents for Video Temporal Grounding](http://arxiv.org/abs/2605.13803)

- EvoGround: introduces a framework of two coupled self-evolving agents, a proposer and a solver, that learn video temporal grounding from raw videos without manual labels.
- The proposer and solver are initialized from the same backbone and iteratively improve each other through a self-reinforcing reinforcement learning loop.
- The framework utilizes a group reward-decoupled normalization policy optimization (GDPO) to balance multiple reward signals, achieving performance competitive with fully supervised models.

---

[EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents](http://arxiv.org/abs/2605.13941)

- EVOLVEMEM: introduces a self-evolving memory architecture that treats retrieval infrastructure as a dynamic action space optimized through an autonomous AutoResearch process.
- The framework utilizes a Structured Memory Store, a Multi-view Retriever, and an LLM-powered Diagnosis Module to iteratively refine retrieval configurations based on per-question failure logs.
- By replacing manual tuning with a closed-loop evolution cycle, the system autonomously discovers effective retrieval strategies and transfers universal principles across different benchmarks.

---

[AgentTrap: Measuring Runtime Trust Failures in Third-Party Agent Skills](http://arxiv.org/abs/2605.13940)

- AgentTrap: introduces a dynamic benchmark for evaluating whether LLM agents can safely utilize third-party skills by measuring runtime trust failures across 16 security-impact dimensions.
- The framework employs a controlled execution layer to monitor agent trajectories, enabling diagnostic attribution of security failures to the LLM backbone, agent framework, or environment configuration.
- AgentTrap utilizes a combination of deterministic checks and LLM-based trajectory analysis to distinguish between successful task completion, malicious behavior, and defensive blocking.

---

[EconAI: Dynamic Persona Evolution and Memory-Aware Agents in Evolving Economic Environments](http://arxiv.org/abs/2605.13762)

- EconAI: introduces an LLM-powered simulation framework that integrates macro/microeconomic dynamics by utilizing an LLM-backbone, Event Perception Module, Long-term Memory Bank, Short-term Memory Bank, Content Extractor, Persona Extraction Module, Long-term Persona Bank, Response Generator, Economic Sentiment Index (ESI), and a Decision-making Mechanism.
- The framework employs a cognitive architecture where agents use memory-driven learning and sentiment-modulated preferences to balance short-term optimization with long-term strategic planning.
- Empirical results demonstrate that EconAI improves the stability of economic indicators and successfully recovers canonical economic regularities like the Phillips and Okun curves.

---

[Learning POMDP World Models from Observations with Language-Model Priors](http://arxiv.org/abs/2605.13740)

- Pinductor: introduces a framework that induces executable POMDP world models from observation-action trajectories by leveraging LLM priors and belief-based feedback without requiring privileged hidden state access.
- The framework utilizes an LLM to propose candidate model components, which are then evaluated via particle filtering and refined through diagnostic feedback to optimize a belief-based likelihood objective.
- Pinductor demonstrates sample-efficient world-model learning that matches privileged-state baselines and outperforms non-LLM tabular methods across various partially observable MiniGrid environments.

---

[Senses Wide Shut: A Representation–Action Gap in Omnimodal LLMs](http://arxiv.org/abs/2605.13737)

- IMAVB: introduces a 500-clip benchmark designed to measure the Representation–Action Gap in omnimodal LLMs by testing their ability to detect implicit false premises in video and audio.
- The study documents that while hidden states of omnimodal LLMs reliably encode premise–perception mismatches, these models frequently fail to propagate this signal to their output, exhibiting under-rejection or over-rejection behaviors.
- The authors propose PGLA as a diagnostic intervention that re-injects the encoded mismatch signal into the output distribution, yielding a +15.0pp mean improvement in rejection accuracy across eight open-source models.

---

[SCIOMIND: Cognitively Grounded Multi-Agent Social Simulation with Anchoring-Based Belief Dynamics and Dynamic Profiles](http://arxiv.org/abs/2605.13725)

- SCIOMIND: introduces a cognitively grounded multi-agent simulation framework that integrates structured opinion dynamics with LLM-based agent reasoning to improve behavioural realism.
- The framework utilizes a four-layer memory architecture and an anchoring-based belief update mechanism to simulate persistent, experience-driven belief formation in social networks.
- SCIOMIND incorporates dynamic agent profiles and a social relationship simulation engine to enable heterogeneous, context-aware interactions that mirror real-world opinion dynamics.

---

[SkillOps: Managing LLM Agent Skill Libraries as Self-Maintaining Software Ecosystems](http://arxiv.org/abs/2605.13716)

- SkillOps: introduces a plug-in maintenance framework that treats LLM agent skill libraries as self-maintaining software ecosystems to mitigate skill technical debt.
- The framework utilizes a Hierarchical Skill Ecosystem Graph (HSEG) to model skills as typed contracts and manage them through alternating task-time execution and library-time maintenance loops.
- SkillOps improves task success rates by diagnosing library health across utility, redundancy, compatibility, failure-risk, and validation-gap dimensions, applying automated repairs with minimal LLM overhead.

---

[Identifying AI Web Scrapers Using Canary Tokens](http://arxiv.org/abs/2605.13706)

- Canary Token Infrastructure: introduces a methodology for identifying AI web scrapers by embedding unique canary tokens into controlled websites and monitoring their appearance in LLM-generated responses.
- The framework utilizes Website Templates to serve distinct Canary Tokens to visiting Scraper Bots, enabling the Scraper Inference Engine to attribute retrieved data to specific scraper identities.
- Experimental results across 22 production AI Chatbots demonstrate that many systems rely on third-party search engine scrapers and that content remains cached even after websites are taken offline or restricted.

---

[FlowCompile: An Optimizing Compiler for Structured LLM Workflows](http://arxiv.org/abs/2605.13647)

- FlowCompile: introduces a compiler-inspired framework for optimizing structured LLM workflows by performing compile-time design space exploration to generate a reusable set of accuracy–latency trade-off configurations.
- The framework utilizes a structure-aware compositional proxy to estimate workflow-level performance from individual sub-agent profiles, enabling scalable optimization without exhaustive end-to-end evaluation.
- FlowCompile supports flexible deployment by providing a menu of optimized operating points that can be selected based on specific latency budgets or performance preferences.

---

[Learning Equilibria in Coordination Games via Minorization-Maximization](http://arxiv.org/abs/2605.13644)

- IMM (Iterative Minorization-Maximization): introduces a learning framework for coordination games that utilizes a regularized potential function to ensure unique equilibrium selection under prospect-theoretic utility models.
- The framework employs a coordinating agent to aggregate information, enabling scalable learning and convergence to potential-optimal equilibria even in non-smooth utility settings.
- By replacing original optimization problems with a sequence of surrogate problems, the approach demonstrates superior convergence speed compared to traditional gradient-based and best-response methods.

---

[How to Interpret Agent Behavior](http://arxiv.org/abs/2605.13625)

- Act·ONOMY: introduces a hierarchical taxonomy and automated analysis pipeline to interpret and characterize the runtime behavior of LLM agents.
- The framework utilizes an LLM-powered-Discovery-Qualitative-Analyst to map unstructured agent execution traces into a structured, quote-grounded vocabulary of 10 actions and 46 sub-actions.
- By providing a shared vocabulary and automated tools, the research enables scalable behavioral profiling, failure mode identification, and cross-agent comparison.

---

[Unweighted ranking for value-based decision making with uncertainty](http://arxiv.org/abs/2605.13601)

- FUW-VBDM: introduces a human-centred decision-making framework that integrates quantitative and qualitative criteria using fuzzy logic to address uncertainty and normative bias.
- The framework utilizes the Rankzzy method to perform unweighted optimization over a domain of fuzzy weights, ensuring mathematical consistency and transparency in value-based decision-making.
- Rankzzy employs a generalized fuzzy p-mean score function to generate customizable rankings, demonstrating reduced computational costs and robust performance compared to existing multi-criteria decision-making approaches.

---

[Position: Assistive Agents Need Accessibility Alignment](http://arxiv.org/abs/2605.13579)

- Accessibility Alignment Framework: introduces a lifecycle-oriented design pipeline for assistive agents that integrates Task Card, Accessibility Success Specification, Interaction Contract, Risk and Uncertainty Policy, Privacy Manifest, and Autonomy Calibration Specification to address systematic failures in BVI-centered scenarios.
- The framework shifts agent design from generic task completion to safety-critical, verifiability-aware, and non-visual interaction paradigms tailored for BVI users.
- It addresses recurring failure modes such as silent failures, overconfident hallucinations, miscalibrated autonomy, and interaction-induced cognitive overload through structured design artifacts and runtime guardrails.

---

[Self-Supervised On-Policy Reinforcement Learning via Contrastive Proximal Policy Optimisation](http://arxiv.org/abs/2605.13554)

- CPPO (Contrastive Proximal Policy Optimisation): introduces an on-policy reinforcement learning algorithm that computes advantages directly from contrastive Q-values using a Policy network, State-action encoder, Goal encoder, Contrastive critic, and PPO optimizer.
- The framework replaces traditional reward-based value estimation with a self-supervised contrastive objective, enabling goal-conditioned learning without hand-crafted rewards or replay buffers.
- CPPO demonstrates robust performance across discrete and continuous, single-agent and multi-agent environments, matching or exceeding reward-based PPO baselines in most tested tasks.

---

[RealICU: Do LLM Agents Understand Long-Context ICU Data? A Benchmark Beyond Behavior Imitation](http://arxiv.org/abs/2605.13542)

- ICU-Evo: introduces a structured-memory agent framework for ICU decision-support that organizes clinical context into heterogeneous memory types to improve long-horizon reasoning.
- The framework utilizes an Observation Agent, Assessment Agent, and Insight Agent to maintain Working memory, Trend memory, Critical-event memory, Trajectory memory, and Insight memory for sequential clinical decision-making.
- RealICU benchmark evaluates LLMs on four physician-motivated tasks using hindsight-annotated labels to measure clinical correctness rather than behavioral imitation.

---

[Integration of an Agent Model into an Open Simulation Architecture for Scenario-Based Testing of Automated Vehicles](http://arxiv.org/abs/2605.13539)

- OSMP based simulation integration architecture: introduces a standardized, modular framework for integrating traffic agent models into heterogeneous simulation environments using Open Simulation Interface (OSI) and Functional Mock-up Interface (FMI).
- The architecture utilizes an OSMP-packaged agent model containing an OSI Adapter, Behavior Model, and Dynamics Model to ensure tool-independent interoperability across platforms like OpenPASS, CARLA, and CarMaker.
- Evaluation demonstrates that the approach maintains stable closed-loop behavior and scales linearly in computational cost, facilitating reproducible scenario-based testing for automated driving systems.

---

[Scaling Retrieval-Augmented Reasoning with Parallel Search and Explicit Merging](http://arxiv.org/abs/2605.13534)

- MultiSearch: introduces an RL-based framework that improves retrieval-during-reasoning by employing parallel multi-query retrieval and explicit information merging to enhance signal-to-noise ratios.
- The framework utilizes a multi-process reward design, including answer-, multi-query-, and merging-rewards, to provide targeted supervision for intermediate retrieval and consolidation behaviors.
- MultiSearch optimizes the multi-reward objective using Group reward-Decoupled Normalization Policy Optimization (GDPO), which independently normalizes heterogeneous reward signals to ensure robust policy training.

---

[Limits of Personalizing Differential Privacy Budgets](http://arxiv.org/abs/2605.13503)

- Limits of Personalizing Differential Privacy Budgets: introduces a comparative analysis between the best affine estimator and a simple unique-threshold ε-estimator for mean estimation under heterogeneous privacy constraints.
- The research demonstrates that full personalization of privacy budgets offers only modest utility gains over simpler thresholding approaches in most practical scenarios.
- The study establishes constant-factor approximation bounds for the threshold-based estimator in specific regimes and characterizes the performance gap for arbitrary privacy levels.

---

[Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models](http://arxiv.org/abs/2605.13497)

- APG4RecSim: introduces a three-stage framework that automates the construction of task-executable user profiles from interaction history without manual schemas, utilizing Attribute Initialisation and Extraction, Context-Aware Semantic Consolidation, and Causal Mapping and Refinement.
- The framework employs an LLM-based pipeline to transform raw interaction logs into a Consolidated User Persona and subsequently into a Task-Aligned Simulation Profile via a Task Decision Path.
- By utilizing counterfactual trait-to-step mapping, the framework ensures that generated profiles are robust to popularity and position biases while maintaining stable performance across diverse recommendation tasks and LLM backbones.

---

[MARLIN: Multi-Agent Game-Theoretic Reinforcement Learning for Sustainable LLM Inference in Cloud Datacenters](http://arxiv.org/abs/2605.13496)

- MARLIN: introduces a multi-agent reinforcement learning framework that utilizes a game-theoretic approach to balance competing objectives for sustainable LLM inference in geo-distributed cloud datacenters.
- The framework employs a two-phase process where agents independently propose scheduling plans in phase 1 and negotiate a final blended plan through weighted voting and a veto mechanism in phase 2.
- MARLIN optimizes for time-to-first-token, carbon emissions, water usage, and energy costs, demonstrating significant performance improvements over state-of-the-art LLM inference management frameworks.

---

[SieveFL: Hierarchical Runtime-Aware Pruning for Scalable LLM-Based Fault Localization](http://arxiv.org/abs/2605.13491)

- SieveFL: introduces a five-stage hierarchical framework that resolves the Scale-Precision Dilemma in LLM-based fault localization through progressive pre-LLM filtering.
- The framework utilizes LLM-based Test Analysis, Suspicious File Identification, Runtime-Aware Candidate Pruning, Per-Method LLM Screening, and LLM-Based Re-ranking to reduce candidate search space and token consumption.
- By integrating JaCoCo runtime traces with semantic retrieval, SieveFL enables efficient, high-precision fault localization on commodity hardware without requiring proprietary frontier models.

---

[Sustainable Graph Analytics Workload Scheduling with Evolutionary Reinforcement Learning in Edge-Cloud Systems](http://arxiv.org/abs/2605.13489)

- MERSEM (Multi-Objective Evolutionary Reinforcement Learning framework for Sustainable Edge-Cloud Management): introduces a hybrid optimization framework that integrates an Evolutionary Algorithm for global exploration with an RL-Guided Local Search agent for adaptive workload scheduling.
- The framework co-optimizes SLA violation rates and operational carbon emissions by modeling DAG-based graph analytics workloads across heterogeneous edge, fog, and cloud infrastructure.
- MERSEM utilizes a trajectory-based RL agent and dominance-based evolutionary operators to maintain Pareto-optimal scheduling solutions under dynamic system conditions.

---

[R²-Mem: Reflective Experience for Memory Search](http://arxiv.org/abs/2605.13486)

- R²-Mem: introduces a reflective experience framework for memory search systems that utilizes a Rubric-guided Evaluator and a self-Reflection Learner to distill reusable process-level guidance from historical search trajectories.
- The framework improves LLM agent efficiency and effectiveness by retrieving relevant planning and reflection experiences from dedicated banks to guide iterative search processes.
- R²-Mem enables RL-free self-improvement by allowing agents to learn from both high- and low-quality search steps, reducing redundant exploration and token consumption.

---

[PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents](http://arxiv.org/abs/2605.13481)

- PAI-2: introduces a GraphRAG framework that utilizes a multi-stage query processing pipeline to optimize knowledge graph retrieval and reasoning for LLMs.
- The framework incorporates a dynamic planning mechanism that iteratively refines search steps and subgraph traversals based on extracted entities and matched graph vertices.
- PAI-2 improves factual correctness and reduces hallucinations by balancing structured and unstructured data retrieval through LLM-driven reasoning and iterative query refinement.

---

[Sleeper Channels and Provenance Gates: Persistent Prompt Injection in Always-on Autonomous AI Agents](http://arxiv.org/abs/2605.13471)

- Sleeper Channels and Provenance Gates: introduces a threat model for persistent prompt injection in always-on OS-live agents, utilizing OpenClaw and Hermes Agent as canonical instances, and proposes a tiered defense mechanism D2-Gate to mitigate cross-surface attacks.
- The framework employs Update Hooks and Gate Hooks to track provenance across memory, skills, and filesystem substrates, ensuring that consequential actions are validated against a closed action set.
- By binding action-instance digests to hardware-attested owner grants, the system prevents paraphrase laundering and unauthorized agent behavior, effectively decoupling security enforcement from the LLM's internal context.

---

[CA2: Code-Aware Agent for Automated Game Testing](http://arxiv.org/abs/2605.13918)

- CA2 (Code-Aware Agent): introduces a goal-conditioned reinforcement learning framework that leverages internal call stack information to improve functional code coverage in automated game testing.
- The architecture integrates a source code profiler with a Causal Transformer to process multi-modal inputs, including game states and call stack traces, for effective offline policy learning.
- Experimental results demonstrate that incorporating call stack signals via Multi-Head Self-Attention significantly enhances the agent's ability to reach specific target functions compared to non-code-aware baselines.

---

[COGNIFOLD: Always-On Proactive Memory via Cognitive Folding](http://arxiv.org/abs/2605.13438)

- COGNIFOLD: introduces a brain-inspired, always-on agent memory architecture that continuously folds fragmented event streams into self-emerging cognitive structures using a tri-layered substrate consisting of a Hippocampal Layer, Neocortical Layer, and Prefrontal Layer.
- The framework employs a dynamically evolving multigraph to address four structural debts—accumulation, compression, decay, and completion—through automatic graph-level operations that enable proactive intent emergence.
- By extending Complementary Learning Systems theory, the system achieves robust performance across diverse cognitive benchmarks by transitioning from reactive retrieval to proactive, structure-driven assembly.

---

[TRIAGE: Evaluating Prospective Metacognitive Control in LLMs under Resource Constraints](http://arxiv.org/abs/2605.13414)

- TRIAGE: introduces an evaluation framework that measures the prospective metacognitive control of LLMs by requiring them to commit to a portfolio-level plan of task selection, sequencing, and token allocation under a finite budget.
- The framework evaluates models across two regimes: an unconstrained regime for advisory planning and a constrained regime where the model's own token allocations are enforced as binding limits on the solver.
- Experimental results across 20 models demonstrate that object-level capability and metacognitive control often dissociate, as extended reasoning frequently fails to improve triage efficiency and models struggle to honor their own self-imposed budget constraints.

---

[FPGA-Accelerated Lock Management and Transaction Processing: Architecture, Optimization, and Design Space Exploration](http://arxiv.org/abs/2605.13398)

- FPGA-based transaction processing accelerator: introduces a hardware-accelerated architecture for 2-Phase Locking (2PL) that offloads lock management and transaction execution from CPUs to FPGAs to mitigate latency and throughput bottlenecks.
- The architecture utilizes dedicated Transaction Agents and Lock Agents, connected via a hierarchical crossbar, to enable high-parallelism transaction processing and efficient lock serving.
- Experimental results demonstrate that the accelerator achieves significantly higher lock serving and transaction throughput compared to CPU-based baselines by leveraging on-chip memory and asynchronous pipelining.

---

[RS-Claw: Progressive Active Tool Exploration via Hierarchical Skill Trees for Remote Sensing Agents](http://arxiv.org/abs/2605.13391)

- RS-Claw: introduces a novel agent architecture that redefines tool selection as an active exploration process within a hierarchical skill tree to mitigate context bottlenecks in remote sensing tasks.
- The framework utilizes a progressive disclosure mechanism that enables the LLM to dynamically load tool information on-demand, effectively reducing context overhead and filtering semantic noise.
- By internalizing tool acquisition as an autonomous decision variable, the agent maintains a locally bounded context while preserving tool coverage for complex, long-horizon reasoning.

---

[GRIP-VLM: Group-Relative Importance Pruning for Efficient Vision-Language Models](http://arxiv.org/abs/2605.13375)

- GRIP-VLM: introduces a hierarchical dynamic pruning framework that utilizes a budget-aware Adaptive Token Scorer (ATS) to perform fine-grained, contextualized token-wise importance evaluation within VLM backbones.
- The framework employs a two-stage training strategy, combining SFT-anchored initialization with GRPO-based RL to effectively navigate the non-convex, discrete combinatorial space of visual token selection.
- By integrating a FiLM-based modulator and a hybrid reward function, the system achieves robust generalization across arbitrary compression ratios and outperforms heuristic baselines in inference speed and multi-modal accuracy.

---

[AI Harness Engineering: A Runtime Substrate for Foundation-Model Software Agents](http://arxiv.org/abs/2605.13357)

- AI Harness Engineering: introduces a runtime substrate that mediates between a foundation-model agent and a software environment to transform latent coding capability into auditable software-engineering behavior.
- The framework utilizes an H0–H3 harness ladder to progressively expose runtime support, enabling empirical separation of the harness's contribution from the model's latent capabilities.
- A trace-based evaluation protocol records eight classes of execution evidence, allowing for the classification of agent performance based on verification autonomy rather than simple task success.

---

[Contextual Bandits for Resource-Constrained Devices using Probabilistic Learning](http://arxiv.org/abs/2605.13346)

- HD-CBPROB: introduces a resource-efficient contextual bandit framework that replaces deterministic accumulation with a probabilistic update rule on low-precision saturating integers.
- The framework utilizes a time-decaying update probability to manage learning rates, effectively bounding action hypervectors without requiring periodic binarization or auxiliary counters.
- Experimental results demonstrate that HD-CBPROB achieves performance comparable to real-valued baselines while maintaining a significantly smaller memory footprint than existing binarized hyperdimensional approaches.

---

[Multi-Agent Systems in Emergency Departments: Validation Study on a ED Digital Twin](http://arxiv.org/abs/2605.13345)

- DES-ABM-MAS: introduces a hybrid simulation framework combining Discrete Event Simulation and Agent-Based Modeling to evaluate emergency department resource optimization strategies.
- The framework integrates a LLM-based Multi-Agent System that utilizes a Blackboard Architecture to observe simulation states and propose interventions via specialized agents.
- Validation results demonstrate that the simulation effectively replicates real-world emergency department dynamics and intervention outcomes across varying facility sizes.

---

[EGO2WORLD: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning](http://arxiv.org/abs/2605.13335)

- EGO2WORLD: introduces an executable benchmark that compiles real-world egocentric cooking videos into symbolic graph-transition environments to evaluate embodied agents under partial observation.
- The framework separates the hidden world graph (Gwt) from the agent-side belief graph (Gbt), forcing agents to maintain memory, handle state changes, and perform replanning based on partial observations and feedback.
- Experiments demonstrate that action-level overlap often overestimates physical-state success, highlighting the necessity of belief maintenance and diagnostic replanning for long-horizon embodied tasks.

---

[What Limits Vision-and-Language Navigation ?](http://arxiv.org/abs/2605.13328)

- StereoNav: introduces a robust Vision-Language-Action framework that mitigates perceptual instability and instruction under-specification by integrating target-point priors and stereo-based unified understanding.
- The framework utilizes Visual Rendering to provide persistent global guidance and employs a multi-branch encoder architecture to synergize semantic, structural, and geometric tokens for the MLLM.
- StereoNav achieves state-of-the-art performance on R2R-CE and RxR-CE benchmarks while demonstrating superior reliability and execution consistency in real-world robotic deployments.

---

[HCSG: Human-Centric Semantic-Geometric Reasoning for Vision-Language Navigation](http://arxiv.org/abs/2605.13321)

- HCSG (Human-Centric Semantic-Geometric Reasoning): introduces a dual-stream framework for VLN that synergizes geometric forecasting and LLM-based semantic interpretation to enable socially compliant navigation in dynamic environments.
- The framework utilizes a Human Detector to trigger parallel reasoning streams, where the Geometric Reasoning Module models human motion dynamics and the Semantic Reasoning Module generates natural language descriptions of human intent.
- These human-centric features are fused into a topological map, allowing the agent to perform instruction-conditioned planning while adhering to social norms via a dedicated Social Distance Loss.

---

[Embodied Neurocomputation: A Framework for Interfacing Biological Neural Cultures with Scaled Task-Driven Validation](http://arxiv.org/abs/2605.13315)

- Embodied Neurocomputation Framework: introduces a systems-level approach to optimize the interface between digital environments and biological neural networks through modular encoding, transformation, decoding, and feedback components.
- The framework operationalizes BNN-based computation as a multi-variable optimization problem, utilizing an automated pipeline to identify encoding configurations that enable goal-oriented navigation.
- Empirical results demonstrate that optimized BNN agents significantly outperform silicon-based DQN agents and non-adaptive baselines in task performance under equivalent interaction budgets.

---

[Discrete Diffusion for Complex and Congested Multi-Agent Path Finding with Sparse Social Attention](http://arxiv.org/abs/2605.13296)

- DiffLNS: introduces a hybrid framework that integrates a D3PM (Discrete Denoising Diffusion Probabilistic Model) as a learned initializer with an LNS2 (Large Neighborhood Search 2) repair-based solver to generate high-quality, coordinated multi-agent path finding plans.
- The framework utilizes a diffusion-aware sparse social attention mechanism to dynamically construct local neighborhoods, focusing computation on conflict-relevant agent interactions rather than dense all-to-all attention.
- By leveraging discrete diffusion for warm-starting, DiffLNS improves repair success rates in dense and congested environments while maintaining scalability to large agent teams and competitive solution quality.

---

[CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution](http://arxiv.org/abs/2605.13295)

- CANTANTE: introduces a framework that optimizes LLM-based multi-agent systems by decomposing global system-level rewards into per-agent update signals using contrastive attribution across multiple joint rollouts.
- The framework treats agent prompts as learnable parameters and utilizes an attribution LLM to isolate individual agent contributions, enabling effective credit assignment in complex multi-agent workflows.
- CANTANTE consistently outperforms existing prompt optimization baselines on programming, mathematical reasoning, and multi-hop question answering benchmarks while maintaining lower inference costs.

---

[RETOOL-VIDEO: Recursive Tool-Using Video Agents with Meta-Augmented Tool Grounding](http://arxiv.org/abs/2605.13228)

- RETOOL-VIDEO: introduces a recursive tool-using framework that grounds high-level video intents into executable tool chains by delegating abstract actions to a resolver, utilizing a Planner, Resolver, MVTL, Base Tools, Meta Tools, Execution Engine, and Observation Buffer.
- The framework employs a MetaAug-Video Tool Library (MVTL) containing 134 registered tools, including 26 base tools for multimodal signal processing and 108 meta tools for filtering, aggregation, and intermediate-result operations.
- RETOOL-VIDEO optimizes the planner policy using reinforcement learning to improve action selection, evidence sufficiency judgment, and termination in complex video understanding tasks.

---

[An Agentic AI Framework with Large Language Models and Chain-of-Thought for UAV-Assisted Logistics Scheduling with Mobile Edge Computing](http://arxiv.org/abs/2605.13221)

- Agentic AI Framework: introduces an agentic AI-assisted optimization framework that integrates LLMs, RAG, and CoT reasoning to translate user requirements into interpretable mathematical models for hybrid logistics and computational scheduling.
- The framework employs a hierarchical DRL approach with upper-layer PPO for UAV routing and lower-layer PPO for task execution and resource allocation to solve complex combinatorial problems in cloud manufacturing.
- The system utilizes a two-agent workflow consisting of a Responder and a Verifier to ensure semantic fidelity and logical consistency in the generated optimization formulations.

---

[GAGPO: Generalized Advantage Grouped Policy Optimization](http://arxiv.org/abs/2605.13217)

- GAGPO: introduces a critic-free reinforcement learning method for multi-turn LLM agents that enables precise, step-aligned temporal credit assignment through Rollout Grouping, Step-level Credit Assignment, and Group-normalized PPO Update.
- The framework utilizes a Non-parametric Value Proxy and a TD/GAE-style Temporal Estimator to propagate outcome supervision backward through time without requiring a learned critic.
- By employing a Sequence-level Importance Ratio and group-wise normalization, GAGPO achieves stable, localized optimization signals that outperform existing RL baselines on multi-turn agent benchmarks.

---

[Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](http://arxiv.org/abs/2605.13213)

- HAM3: introduces a hierarchical adversarial framework that decomposes attacks into perception, communication, and reasoning layers to evaluate the robustness of multi-modal multi-agent systems.
- The framework models how localized perturbations propagate through agent collaboration, specifically targeting visual-textual inputs, communication topology, and internal reasoning chains.
- Experimental results demonstrate that reasoning-layer attacks, such as Chain-of-Thought Injection, cause the most severe performance degradation, while systemic errors dominate across all attack layers.

---

[FIKA-BENCH: From Fine-grained Recognition to Fine-Grained Knowledge Acquisition](http://arxiv.org/abs/2605.13193)

- FIKA-BENCH: introduces a leakage-aware, evidence-grounded benchmark for evaluating the ability of LMMs and agents to perform active fine-grained knowledge acquisition.
- The framework evaluates systems through a pipeline of model-hard filtering, leakage inspection, and human-verified evidence grounding to ensure models move beyond parametric memorization.
- Empirical results demonstrate that current LMMs and agents struggle with fine-grained recognition, with performance limited by incorrect entity retrieval and visual grounding errors rather than tool availability.

---

[Decoupled Planning for Multiple Omega-Regular Objectives](http://arxiv.org/abs/2605.13185)

- Decoupled Planning Framework: introduces a modular approach for satisfying multiple omega-regular objectives by assigning each to an independent agent and using a scheduler to compose their local policies.
- The framework utilizes stochastic schedulers and specific conventions to ensure that independently designed policies satisfy all objectives almost surely without requiring direct communication.
- The approach supports modular design, robustness, and iterative development by allowing agents to operate independently while maintaining correctness through minimal runtime coordination or pre-agreed conventions.

---

[When Does Hierarchy Help? Benchmarking Agent Coordination in Event-Driven Industrial Scheduling](http://arxiv.org/abs/2605.13172)

- DESBench: introduces a benchmark for evaluating agent coordination in hierarchical, event-driven industrial scheduling environments using Shared World State, Event Engine, Event Interpreter, Decision Layer, and Runtime Interface.
- The framework evaluates four coordination paradigms—centralized, hierarchical, heterarchical, and holonic—by measuring effectiveness, constraint alignment, coordination efficiency, and robustness.
- It utilizes LLMs as decision-making agents within a unified simulation environment to analyze trade-offs in information flow, decision authority, and conflict resolution.

---

[Finding the Weakest Link: Adversarial Attack against Multi-Agent Communications](http://arxiv.org/abs/2605.13170)

- SVCP-APOSG (Single-Victim Communication Perturbation Adversarial Partially Observable Stochastic Game): introduces a framework for single-victim communication perturbation attacks that identifies vulnerable messages, agents, and timesteps using a Jacobian-proxy, weighted-loss, and maximum-loss.
- The framework utilizes a Jacobian-based saliency method to rank messages and select victims, while employing novel loss functions to enhance the effectiveness of gradient-based perturbations against MARL systems.
- Empirical results demonstrate that the proposed methods achieve significant impact across various multi-agent environments, outperforming random message selection in most tested scenarios.

---

[GeoBuildBench: A Benchmark for Interactive and Executable Geometry Construction from Natural Language](http://arxiv.org/abs/2605.13167)

- GeoBuildBench: introduces an interactive benchmark for evaluating LLMs and MLLMs on grounded, executable plane geometry construction from natural language.
- The framework utilizes an agent-environment loop where the LLM or MLLM Agent iteratively generates programs in a Geometry Construction DSL, which are then processed by a Python Geometry Kernel and evaluated by a Verification Module.
- The benchmark assesses model performance through metrics including success rate, structural hallucination frequency, and feedback-driven error recovery capabilities.

---

[Collaborating in Multi-Armed Bandits with Strategic Agents](http://arxiv.org/abs/2605.13145)

- CAOS (Collaborating Agents with Optimistic Stopping): introduces a mechanism that sustains collaborative exploration among strategic agents in multi-armed bandit problems using information sharing as the sole incentive.
- The framework utilizes an OER (Optimistic Expected Reward) procedure to dynamically determine a set of Active Agents who continue to follow a target algorithm, while non-compliant agents revert to a single-agent Algorithm B.
- By enforcing a structured Communication Protocol that verifies actions and shares rewards, the mechanism ensures that collaborative behavior constitutes a Nash equilibrium and achieves strong regret guarantees.

---

[SWE-Cycle: Benchmarking Code Agents across the Complete Issue Resolution Cycle](http://arxiv.org/abs/2605.13139)

- SWE-Cycle: introduces a comprehensive benchmark for evaluating autonomous code agents across the complete issue resolution lifecycle, including environment reconstruction, code implementation, and verification test generation.
- The framework utilizes SWE-Judge, which integrates static code review and dynamic execution to provide robust, fine-grained assessment of agent performance while overcoming the limitations of traditional script-based evaluation.
- By evaluating agents in both isolated and end-to-end FullCycle settings, the research exposes critical bottlenecks in cross-phase dependencies and demonstrates that end-to-end integration often improves dynamic correctness at the cost of structural quality.

---

[ERPPO: Entropy Regularization-based Proximal Policy Optimization](http://arxiv.org/abs/2605.13131)

- ERPPO: introduces a multi-agent reinforcement learning framework that integrates a Distributional Spatiotemporal Ambiguity (DSA) learner and entropy-based policy regularization to enhance object detection in dynamic maritime environments.
- The framework utilizes a DSA learner to compute confidence fields and an entropy-regularized PPO algorithm to dynamically adjust policy updates based on observed environmental ambiguity.
- By applying L1 regularization in high-ambiguity scenarios and L2 regularization in low-ambiguity states, the approach improves search stability and reduces false detection rates in time-critical UAV operations.

---

[Towards Long-horizon Embodied Agents with Tool-Aligned Vision-Language-Action Models](http://arxiv.org/abs/2605.13119)

- VLAs-as-Tools: introduces a framework that distributes long-horizon task burdens by utilizing a high-level VLM agent for planning and a family of specialized VLA tools for bounded physical execution.
- The system employs a VLA tool-family interface to facilitate bidirectional communication, where the agent sends invocation messages and receives progress feedback from the VLA tools.
- Tool-Aligned Post-Training (TAPT) is utilized to align VLA models with specific subtask invocations, employing residual adapters to maintain shared semantic representations while enabling specialized tool behavior.

---

[A Multi-Agent Orchestration Framework for Venture Capital Due Diligence](http://arxiv.org/abs/2605.13110)

- Multi-Agent Orchestration Framework for Venture Capital Due Diligence: introduces an event-driven automation pipeline that utilizes specialized AI agents to synthesize unstructured market data and official financial filings into structured investment reports.
- The system integrates a programmatic extraction pipeline to reverse-engineer Greek Business Registry endpoints, ensuring auditable data retrieval while employing a structural fallback mechanism to mitigate LLM hallucinations.
- By leveraging a low-code DAG-structured architecture, the framework automates end-to-end corporate research, providing traceable provenance for financial metrics and strategic recommendations.

---

[Counterfactual Reasoning for Causal Responsibility Attribution in Probabilistic Multi-Agent Systems](http://arxiv.org/abs/2605.13077)

- PATL-SR: introduces a formal framework for quantifying backward counterfactual responsibility in probabilistic multi-agent systems using the Shapley value.
- The framework utilizes CSG and PSMAS to model agent interactions and compute stable strategy profiles where agents balance expected rewards against responsibility penalties.
- The research demonstrates that model checking and Nash equilibrium computation within this logic remain in PSPACE, ensuring computational feasibility for responsibility-aware strategic reasoning.

---

[PBT-Bench: Benchmarking AI Agents on Property-Based Testing](http://arxiv.org/abs/2605.15229)

- PBT-Bench: introduces a benchmark of 100 curated problems across 40 Python libraries to evaluate the ability of LLMs to perform property-based testing by deriving invariants from documentation and constructing precise input-generation strategies.
- The framework utilizes an automated, containerized F→P harness that evaluates LLM-generated tests against 365 human-verified semantic bugs, requiring agents to identify invariants rather than just concrete test cases.
- Evaluation across eight contemporary LLMs reveals that property-based testing scaffolding significantly improves performance for mid-capability models, while highlighting persistent gaps in cross-function protocol reasoning for all models.

---

[Verifiable Agentic Infrastructure: Proof-Derived Authorization for Sovereign AI Systems](http://arxiv.org/abs/2605.15228)

- DTF (Distributed Trust Framework): introduces a verification layer for governed mutation systems that computes execution authority from structured, verifiable artifacts rather than relying on standing identity.
- The framework enforces a compact authorization invariant where high-stakes execution requires a Justification Proof, consensus-gated approval, and an append-only Evidence Chain.
- By shifting authorization from static roles to proof-derived authority, DTF enables governable, auditable, and replayable execution for autonomous AI agents in sovereign deployments.

---


#### 11th May 2026



[Instruction Adherence in Coding Agent Configuration Files: A Factorial Study of Four File-Structure Variables](http://arxiv.org/abs/2605.10039)

- Instruction Adherence in Coding Agent Configuration Files: introduces a systematic factorial study evaluating how four structural variables in configuration files affect LLM agent compliance with behavioral instructions.
- The study utilizes an AST-based scoring pipeline to measure compliance across 1,650 sessions, finding that file size, instruction position, architecture, and conflicting instructions do not significantly impact adherence.
- The research identifies a significant within-session attenuation effect where LLM agent compliance decreases by approximately 5.6% per generated function, highlighting task identity as a stronger predictor of adherence than configuration structure.

---


[DataMaster: Towards Autonomous Data Engineering for Machine Learning](http://arxiv.org/abs/2605.10906)

- DataMaster: introduces a data-agent framework that treats the data state as the primary optimization target by integrating DataTree (tree-structured search over data states), Data Pool (shared repository for external datasets), and Global Memory (persistent record of outcomes and findings).
- The framework utilizes Red Nodes (agents for external data discovery) and Black Nodes (agents for data refinement and exploitation) to iteratively improve downstream performance under a fixed learning algorithm.
- DataMaster employs a UCB-based scheduling policy and dynamic tree growth to manage limited evaluation budgets while enabling cumulative learning across data-engineering branches.

---

[Shields to Guarantee Probabilistic Safety in MDPs](http://arxiv.org/abs/2605.10888)

- Shielding Framework: introduces a formal framework for probabilistic safety in Markov decision processes that conservatively extends classical shielding techniques.
- The framework provides various shield instantiations, including optimistic, pessimistic, and saturated shields, to balance safety guarantees with policy permissiveness.
- The research demonstrates that combining maximal permissiveness and probabilistic safety is impossible, and provides efficient offline and online learning routines for constructing safe, permissive shields.

---

[Agent-First Tool APIs: Rethinking Enterprise Service Interfaces for LLM-Native Execution](http://arxiv.org/abs/2605.10555)

- Agent-First Tool API: introduces a goal-achievement paradigm for enterprise software that replaces traditional CRUD interfaces with a Six-Verb Semantic Protocol, a Normalized Tool Contract (NTC), and a Dual-layer Governance Pipeline.
- The framework enables LLM agents to perform autonomous tasks by providing structured feedback, confidence scores, and built-in risk assessment, effectively reducing ambiguity-induced failures.
- By separating capability-based and object-scoped permissions, the architecture ensures enterprise-grade security and auditability while maintaining compatibility with existing transport standards like the Model Context Protocol.

---

[Causal Explanations from the Geometric Properties of ReLU Neural Networks](http://arxiv.org/abs/2605.10396)

- ReLU Neural Network Explainability Framework: introduces a method for generating exact causal "why" and "why not" explanations for ReLU-based neural networks by leveraging their geometric properties through Polyhedral Decomposition, Bit Vector Representation, Linear Programming, Adjacent Polytope Marching, H-Representation, and V-Representation.
- The framework utilizes the piecewise linear nature of ReLU networks to partition the input space into convex polytopes, allowing for the extraction of minimally complete explanations directly from the network's internal geometry.
- The approach addresses the black-box nature of control policies by providing exact, verifiable explanations while discussing the trade-offs between H-Representation and V-Representation regarding computational complexity and interpretability.

---

[Agent-ValueBench: A Comprehensive Benchmark for Evaluating Agent Values](http://arxiv.org/abs/2605.10365)

- Agent-ValueBench: introduces a comprehensive benchmark for evaluating the values of autonomous agents by synthesizing executable environments and value-conflict tasks.
- The framework utilizes an automated pipeline with test- and repair-agents to generate validated environments, which are then assessed using trajectory-level rubrics and an LLM-as-Judge.
- Experimental results across 14 models and 4 harnesses reveal that agent values exhibit a "Value Tide" of population-wide homogenization that bends under harness pull and deliberate skill steering.

---

[Merlin: Deterministic Byte-Exact Deduplication for Lossless Context Optimization in Large Language Model Inference](http://arxiv.org/abs/2605.09990)

- Merlin: introduces a deterministic, byte-exact deduplication engine designed for real-time LLM inference preprocessing, utilizing an Input Stream, Fingerprint Pass, L2-Aligned Memory Arena, Lock-Free Routing, and Output Stream.
- The engine operates as a low-latency, CPU-bound sidecar that removes redundant context records before prompt assembly without impacting model quality.
- Empirical validation across multiple benchmarks and production LLMs demonstrates that the engine maintains lossless performance while operating significantly below typical inference latency budgets.

---

[Personal Visual Context Learning in Large Multimodal Models](http://arxiv.org/abs/2605.10936)

- Agentic Context Bank: introduces a framework that structures a user’s visual history into a self-refining memory bank and employs query-adaptive evidence selection to improve LMM reasoning over personal visual context.
- The framework addresses the modality paradox and scaling paradox by replacing passive visual concatenation with a two-stage process involving structured memory construction and selective visual verification.
- The research introduces Personal-VCL-Bench to evaluate LMM performance across three axes of personal visual knowledge: persons, objects, and behavior.

---

[Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](http://arxiv.org/abs/2605.10917)

- MAPF-MMOT: introduces a framework that casts multi-agent path finding as a multi-marginal optimal transport problem, utilizing P1, P2, P3, Sinkhorn-MAPF, and Shadow Transport to achieve scalable and optimal robot trajectories.
- The approach leverages the total unimodularity of the underlying linear program to guarantee integral solutions while employing Schrödinger bridges for probabilistic relaxation and efficient graph pruning.
- Experimental results demonstrate that the proposed pipeline provides significant speedups over existing methods while maintaining near-optimal cost performance in large-scale multi-agent path finding scenarios.

---

[SHEPHERD: A Runtime Substrate Empowering Meta-Agents with a Formalized Execution Trace](http://arxiv.org/abs/2605.10913)

- SHEPHERD: introduces a functional programming model that formalizes meta-agent operations on target agents as functions, with core operations mechanized in Lean.
- The framework treats agent execution as a first-class object, enabling meta-agents to read, rewind, branch, and modify worker agent trajectories through a Git-like execution trace.
- SHEPHERD supports diverse agentic applications, including live supervision, counterfactual meta-optimization, and meta-agent guided Tree-RL training, while providing efficient state-forking and prompt-cache reuse.

---

[Revisiting Policy Gradients for Restricted Policy Classes: Escaping Myopic Local Optima with k-step Policy Gradients](http://arxiv.org/abs/2605.10909)

- k-step Policy Gradient Framework: introduces a generalized k-step policy gradient method that utilizes a k-step Q-function to overcome the myopic nature of standard policy gradient methods in restricted policy classes.
- The framework employs correlated policies to enable multi-step reasoning, effectively escaping suboptimal local optima that arise from independent action sampling.
- Theoretical analysis demonstrates that this approach achieves exponential convergence to near-optimal solutions in O(1/T) iterations for both projected gradient descent and mirror descent.

---

[Engineering Robustness into Personal Agents with the AI Workflow Store](http://arxiv.org/abs/2605.10907)

- AI Workflow Store: introduces a framework that integrates rigorous software engineering processes into agentic loops to produce hardened, reusable workflows instead of relying on improvised on-the-fly synthesis.
- The architecture utilizes a Backend SE Agent Team to engineer workflows, a Workflow Repository to store them, and a Local Agent to match user prompts to these vetted artifacts.
- By amortizing the cost of rigorous engineering across a community of users, the framework aims to achieve production-grade reliability and security in personal agents without sacrificing flexibility.

---

[MDrive: Benchmarking Closed-Loop Cooperative Driving for End-to-End Multi-agent Systems](http://arxiv.org/abs/2605.10904)

- MDrive: introduces a closed-loop cooperative driving benchmark that systematically evaluates multi-agent systems using Human-in-the-Loop Simulation Interface, Agentic Scenario Generation Pipeline, and Real-to-Simulation (Real2Sim) Pipeline.
- The framework utilizes a Simulation Controller to manage reactive agents and human demonstrations, while employing Scenario Feasibility Check and Duplicate &amp; Difficulty Screening to ensure high-quality, diverse test scenarios.
- MDrive integrates Behavior Extraction, Asset Matching, and Coordinate Alignment to reconstruct real-world V2X driving logs into closed-loop CARLA scenarios for robust multi-agent evaluation.

---

[RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards](http://arxiv.org/abs/2605.10899)

- RubricEM: introduces a rubric-guided reinforcement learning framework that combines stagewise policy decomposition with reflection-based meta-policy training to optimize long-horizon research agents.
- The framework utilizes a Stage-Structured Scaffold to organize trajectories into distinct decision modes, enabling Stage-Structured GRPO to provide dense semantic feedback for long-horizon credit assignment.
- A shared-backbone reflection meta-policy distills judged trajectories into reusable rubric-grounded guidance stored in an Agent Rubric Bank, facilitating both within-episode refinement and cross-episode transfer.

---

[AssayBench: An Assay-Level Virtual Cell Benchmark for LLMs and Agents](http://arxiv.org/abs/2605.10876)

- AssayBench: introduces a large-scale benchmark for phenotypic screen prediction, utilizing a Data Curation Pipeline, LLM-assisted Curation, and a Gene Relevance Scorer to evaluate LLMs and agents as virtual cell surrogates.
- The framework frames phenotypic screening as a gene-ranking task, employing a Neural Gene-relevance Predictor, an LLM Ensemble, and Retrieval Baselines to assess model performance across heterogeneous CRISPR assays.
- Evaluation is conducted using the Adjusted normalized Discounted Cumulative Gain (AnDCG@k) metric, which corrects for screen-specific random baselines to enable continuous performance comparison across diverse biological contexts.

---

[Remember the Decision, Not the Description: A Rate-Distortion Framework for Agent Memory](http://arxiv.org/abs/2605.10870)

- DeMem: introduces a decision-centric memory framework that optimizes memory allocation by preserving distinctions necessary for downstream decisions rather than descriptive fidelity.
- The framework utilizes a K-slot memory system that routes history-query pairs to specific slots and performs certified refinement only when shared memory induces decision conflicts.
- DeMem provides a memory-distortion frontier and near-minimax regret guarantees, demonstrating consistent performance gains in long-horizon conversational and agentic tasks.

---

[Is Your Driving World Model an All-Around Player?](http://arxiv.org/abs/2605.10858)

- WorldLens: introduces a unified benchmark for evaluating driving world models across five complementary axes: generation, reconstruction, action-following, downstream tasks, and human preference.
- The framework utilizes WorldLens-26K, a large-scale human-annotated dataset, to train WorldLens-Agent, a vision-language model that provides automated, explainable assessments of world model fidelity.
- The research demonstrates that current world models often prioritize visual appearance over physical and behavioral consistency, highlighting a critical need for holistic evaluation protocols.

---

[The Generalized Turing Test: A Foundation for Comparing Intelligence](http://arxiv.org/abs/2605.10851)

- GTT (Generalized Turing Test): introduces a formal, dataset-agnostic framework for evaluating LLMs by measuring their ability to imitate other models such that a distinguisher cannot reliably identify the actor.
- The framework utilizes an Actor, a Distinguisher, and an optional Specimen to compute a Turing Comparator, which establishes a relative intelligence ordering between conversational agents.
- Empirical results across nine modern LLMs demonstrate that the GTT recovers a stratified intelligence hierarchy consistent with existing benchmarks while providing a closed-loop, adaptive evaluation mechanism.

---

[Rethinking Agentic Search with PI-SERINI: Is Lexical Retrieval Sufficient?](http://arxiv.org/abs/2605.10848)

- PI-SERINI: introduces a search agent that isolates agent-retriever interaction to demonstrate that a well-configured lexical retriever can support effective deep research when paired with capable LLMs.
- The framework utilizes a retrieval controller to manage cached rankings and selective evidence acquisition, enabling efficient deep research without relying on dense retrievers.
- Experimental results on the BrowseComp-Plus benchmark show that PI-SERINI achieves competitive accuracy and high recall while significantly reducing evaluation costs compared to existing dense-retriever search agents.

---

[Training-Free Cultural Alignment of Large Language Models via Persona Disagreement](http://arxiv.org/abs/2605.10843)

- DISCA (Disagreement-Informed Steering for Cultural Alignment): introduces a training-free inference-time alignment method that uses within-country persona disagreement as a reliability signal to steer LLMs toward culturally grounded moral preferences.
- The framework utilizes WVS-grounded persona agents to generate logit-based disagreement signals, which are then processed via Prospect-Theory importance sampling and a dual-pass reliability gate to compute a bounded, loss-averse logit correction.
- By treating disagreement as a sufficient statistic for correction reliability rather than noise, DISCA achieves robust cultural alignment across diverse LLM backbones without requiring weight updates, per-country reward models, or internal model access.

---

[From Controlled to the Wild: Evaluation of Pentesting Agents for the Real-World](http://arxiv.org/abs/2605.10834)

- Evaluation Protocol for AI Pentesting Agents: introduces a methodology for assessing agentic security systems using a structured finding-to-ground-truth pipeline with LLM-as-a-judge, bipartite matching, and cumulative performance analysis.
- The protocol shifts evaluation from binary task completion to validated vulnerability discovery, supporting realistic assessment across complex, open-ended targets.
- It incorporates efficiency metrics and continuous ground-truth maintenance to provide an operationally informative comparison of stochastic LLM-based pentesting agents.

---

[Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents](http://arxiv.org/abs/2605.10832)

- ODE (On-policy Data Evolution): introduces a closed-loop data construction framework that refines training tasks for multimodal deep search agents by using rollout feedback to update generator configurations.
- The framework utilizes a Visual-Native Agent Harness that employs an Image Bank Reference Protocol to make tool-produced visual evidence persistently reusable across multi-step trajectories.
- By treating data generation as an adaptive optimization process, ODE aligns training data with the evolving capabilities of the target LLM agent, significantly improving performance on complex multimodal search benchmarks.

---

[The First Drop of Ink: Nonlinear Impact of Misleading Information in Long-Context Reasoning](http://arxiv.org/abs/2605.10828)

- The First Drop of Ink: introduces a systematic study of how the proportion of hard distractors in long-context LLMs causes a nonlinear, front-loaded performance degradation.
- The research demonstrates that hard distractors dominate the softmax attention denominator even at small proportions, rendering partial filtering ineffective.
- Theoretical and empirical analyses confirm that substantial performance recovery in long-context LLMs requires reducing the hard-distractor proportion to near zero, emphasizing the necessity of high upstream retrieval precision.

---

[MaD Physics: Evaluating information seeking under constraints in physical environments](http://arxiv.org/abs/2605.10820)

- MaD Physics: introduces a benchmarking framework for evaluating LLMs on their ability to perform strategic information gathering and model inference in resource-constrained physical environments.
- The framework utilizes three distinct physical domains—Classical, Quantum, and Fluid mechanics—with altered physical laws to prevent reliance on memorized knowledge and force active empirical discovery.
- Experimental results demonstrate that while LLM performance generally scales with model capability, agents struggle with structured exploration and accurate symbolic recovery under strict cost-fidelity constraints.

---

[Policy Gradient Methods for Non-Markovian Reinforcement Learning](http://arxiv.org/abs/2605.10816)

- ASMPG (Agent State-Markov Policy Gradient): introduces a reward-centric reinforcement learning framework for non-Markovian decision processes that jointly optimizes recursive agent state dynamics and a control policy.
- The framework utilizes an Agent State-Markov policy class to maintain a compact, recursively updated internal state that summarizes interaction history for efficient decision-making.
- The paper establishes a novel policy gradient theorem for non-Markovian environments and provides finite-time convergence guarantees for the proposed ASMPG algorithm.

---

[NanoResearch: Co-Evolving Skills, Memory, and Policy for Personalized Research Automation](http://arxiv.org/abs/2605.10813)

- NanoResearch: introduces a multi-agent framework that enables personalized research automation through tri-level co-evolution of skills, memory, and planner policy.
- The system utilizes an Orchestrator to coordinate ideation, experimentation, and writing stages, while the Skill Bank, Memory Module, and Planner Model accumulate experience to refine research outputs over successive cycles.
- NanoResearch incorporates SDPO-based feedback internalization to align the planner policy with individual researcher preferences, demonstrating improved performance and cost-efficiency across diverse scientific domains.

---

[LLMs for Secure Hardware Design and Related Problems: Opportunities and Challenges](http://arxiv.org/abs/2605.10807)

- LLM-driven EDA and Hardware Security Frameworks: introduces a comprehensive synthesis of LLM-native methodologies for hardware design, covering reasoning-driven synthesis, multi-agent vulnerability extraction, and robust security countermeasures.
- The paper evaluates the integration of LLMs into semiconductor workflows, highlighting the transition from passive coding assistants to autonomous agents that utilize Multi-agent loops, Reasoning models, and Graph-based representations to bridge the semantic gap in hardware design.
- It addresses critical security challenges including data contamination, backdoor attacks, and the alignment paradox, while proposing solutions such as Machine unlearning, Red-teaming agents, and Dynamic benchmarking to ensure trustworthy design ecosystems.

---

[ComplexMCP: Evaluation of LLM Agents in Dynamic, Interdependent, and Large-Scale Tool Sandbox](http://arxiv.org/abs/2605.10787)

- ComplexMCP: introduces a rigorous evaluation framework for LLM agents operating within large-scale, interdependent, and stochastic tool ecosystems using a seed-driven architecture.
- The framework utilizes a rule-based, deterministic evaluation system to assess agent performance across 150+ interdependent tools, replacing subjective LLM-based scoring.
- Granular trajectory analysis identifies critical failure modes in LLMs, including "Clean-Slate" bias, over-confidence, and strategic defeatism, which hinder performance in complex, real-world software environments.

---

[MAGS-SLAM: Monocular Multi-Agent Gaussian Splatting SLAM for Geometrically and Photometrically Consistent Reconstruction](http://arxiv.org/abs/2605.10760)

- MAGS-SLAM: introduces a monocular RGB-only multi-agent 3D Gaussian Splatting SLAM framework for collaborative scene reconstruction using Monocular Front-end, Coordinator, Submap Summary, Sim(3) Pose Graph, Occupancy-Aware Fusion, and Global Photometric Refinement.
- The framework enables collaborative mapping by transmitting compact submap summaries instead of raw data, utilizing a Sim(3) pose graph to resolve monocular scale ambiguity across agents.
- Occupancy-aware fusion and joint photometric refinement eliminate duplicated Gaussians and photometric seams, achieving photorealistic reconstruction without active depth sensors.

---

[The Agent Use of Agent Beings: Agent Cybernetics Is the Missing Science of Foundation Agents](http://arxiv.org/abs/2605.10754)

- Agent Cybernetics: introduces a theoretical framework that maps six canonical laws of classical cybernetics onto agent design principles to provide a scientific foundation for reliable foundation agents.
- The framework synthesizes these principles into three engineering desiderata—reliability, lifelong running, and self-improvement—to address failure modes in long-horizon tasks.
- The paper demonstrates the utility of this approach by identifying domain-specific failure modes and engineering recommendations for code generation, computer use, and automated research.

---

[An Uncertainty-Aware Resilience Micro-Agent for Causal Observability in the Computing Continuum](http://arxiv.org/abs/2605.10718)

- AURORA (Uncertainty-Aware Resilience Micro-Agent for Causal Observability): introduces a lightweight, uncertainty-aware framework for autonomous grey failure diagnosis and mitigation in edge-tier computing environments using parallel micro-agents, Bayesian networks, and a dual-gated execution mechanism.
- The framework employs a dual-gated safety mechanism that evaluates posterior causal confidence and variational free energy to prevent destructive interventions, escalating unresolved cases to the fog tier.
- Experimental results demonstrate that AURORA achieves a 0% destructive action rate while maintaining 62.0% repair accuracy and low computational latency on resource-constrained edge hardware.

---

[Heteroscedastic Diffusion for Multi-Agent Trajectory Modeling](http://arxiv.org/abs/2605.10717)

- U2Diffine (Unified Uncertainty-aware Diffusion): introduces a diffusion-based framework for multi-agent trajectory completion that jointly estimates state-wise heteroscedastic uncertainty using a first-order Taylor approximation for Reverse Gaussian Sampling.
- The architecture integrates a Temporal Mamba for sequential dynamics and a Social Transformer for agent interactions, enabling robust trajectory reconstruction and uncertainty quantification.
- A post-processing RankNN module further enhances reliability by assigning error probabilities to generated modes, significantly improving scene-level trajectory forecasting and completion performance.

---

[The Bystander Effect in Multi-Agent Reasoning: Quantifying Cognitive Loafing in Collaborative Interactions](http://arxiv.org/abs/2605.10698)

- MAS: introduces a theoretical framework to quantify the "Bystander Effect" in LLMs, where simulated social pressure induces cognitive loafing and alignment hallucinations.
- The paper formalizes the Interaction Depth Limit and Sovereignty Decay Law to demonstrate how swarm size and task entropy trigger a transition from a "Fortified Mind" to a "Hollowed Mind" state.
- By auditing internal Chain-of-Thought traces against external outputs, the research proves that LLMs often compute correct derivations but deliberately externalize falsehoods to sycophantically appease the swarm.

---

[Step Rejection Fine-Tuning: A Practical Distillation Recipe](http://arxiv.org/abs/2605.10674)

- SRFT: introduces a fine-grained distillation approach that leverages unresolved LLM agent trajectories by employing a critic LLM to mask harmful steps during supervised fine-tuning.
- The framework utilizes a Student model, a Critic LLM, a trajectory dataset, a loss masking mechanism, and a weighted distillation objective to improve performance on software engineering tasks.
- By selectively omitting loss contributions from erroneous steps identified by the critic, the method enables the Student model to learn from partial successes without internalizing mistakes found in failed trajectories.

---

[Evolving-RL: End-to-End Optimization of Experience-Driven Self-Evolving Capability within Agents](http://arxiv.org/abs/2605.10663)

- Evolving-RL: introduces a unified algorithmic framework that jointly optimizes experience extraction and utilization within a single shared policy, utilizing Extractor, Solver, Retriever, Shared Policy, Environment, and Joint Optimizer.
- The framework employs an extractor-centric design where the Extractor generates candidate skills evaluated by the Solver on retrieved tasks, providing coupled supervisory signals for co-evolution.
- By internalizing reusable experience patterns into model parameters, Evolving-RL functions as an experience-augmented RL algorithm that enhances generalization on unseen tasks.

---

[PRISM: Generation-Time Detection and Mitigation of Secret Leakage in Multi-Agent LLM Pipelines](http://arxiv.org/abs/2605.10614)

- PRISM: introduces a real-time, multi-signal defence that monitors token-level generation dynamics to detect and mitigate secret leakage in multi-agent LLM pipelines before full reconstruction occurs.
- The framework utilizes a per-token risk score derived from temporal generation signals and text-structural features to trigger graduated interventions, including pass-through, sanitisation, or halting.
- PRISM effectively addresses propagation amplification by monitoring each agent in the pipeline, achieving zero observed leakage on a 2,000-task adversarial benchmark while maintaining high output utility.

---

[CrackMeBench: Binary Reverse Engineering for Agents](http://arxiv.org/abs/2605.10597)

- CrackMeBench: introduces a reproducible benchmark for evaluating LLM agents on binary reverse engineering tasks using an executable-oracle framework.
- The framework utilizes a Host orchestrator, Docker sandbox, Private oracle, Agent-visible filesystem, and Tool manifest to measure agent performance on deterministic binary validation problems.
- The benchmark evaluates LLMs on their ability to recover validation logic from stripped binaries without relying on source code or external web resources.

---

[Controllability in preference-conditioned multi-objective reinforcement learning](http://arxiv.org/abs/2605.10585)

- MOPPO (Multi-objective Proximal Policy Optimization): introduces a preference-conditioned reinforcement learning approach that utilizes an actor network, a critic network, and a multi-objective value head to enable dynamic adaptation to user preferences.
- The framework leverages PufferMO, an extension of PufferLib, to facilitate high-throughput training and evaluation of agents in complex, multi-objective environments.
- The research proposes rank-correlation-based metrics to quantify agent controllability, addressing the limitations of mainstream multi-objective reinforcement learning evaluation protocols.

---

[VISTA: A Generative Egocentric Video Framework for Daily Assistance](http://arxiv.org/abs/2605.10579)

- VISTA (Video Synthesis for Training Agents): introduces a modular framework that utilizes a 5-step LLM-driven pipeline to synthesize high-fidelity egocentric videos for training and evaluating proactive AI agents.
- The system employs causal reverse reasoning to derive user actions from intervention needs, ensuring logical consistency and physical plausibility in generated scenarios.
- VISTA incorporates a multi-dimensional evaluation pipeline that leverages spatial analysis and LLM-as-a-Judge protocols to measure intervention utility, timeliness, and safety criticality.

---

[Effect of Graph Gluing on Consensus in Networked Multi-Agent Systems](http://arxiv.org/abs/2605.10558)

- Graph Gluing Framework: introduces a mathematical approach to analyze how interconnecting multi-agent subsystems via bridge gluing or interface gluing impacts the Fiedler eigenvalue and consensus convergence rate.
- The framework utilizes the spectral properties of the graph Laplacian to establish theoretical bounds on algebraic connectivity for combined networked systems.
- Simulation results demonstrate that increasing the number of interconnecting links between subsystems enhances the Fiedler eigenvalue, thereby accelerating the consensus convergence of the overall multi-agent network.

---

[Higher Resolution, Better Generalization: Unlocking Visual Scaling in Deep Reinforcement Learning](http://arxiv.org/abs/2605.10546)

- Impoola: introduces a resolution-independent architecture for deep RL that replaces the standard flattening operation with global average pooling to decouple parameter count from input resolution.
- The framework demonstrates that higher-resolution visual inputs significantly improve agent performance and generalization by enabling more spatially localized policy attention.
- By utilizing the Procgen-HD benchmark, the study reveals that standard low-resolution conventions in deep RL create a perceptual bottleneck that limits the scalability of traditional flatten-based architectures.

---

[TIE: Time Interval Encoding for Video Generation over Events](http://arxiv.org/abs/2605.10543)

- TIE: introduces a principled, plug-and-play interval-aware generalization of rotary embeddings that elevates time intervals to first-class primitives within DiT cross-attention.
- The framework utilizes RoTE to aggregate positional evidence over full event durations, effectively acting as a temporal low-pass filter that provides robustness against noisy boundary annotations.
- TIE enables precise temporal control in concurrent-event regimes, significantly improving temporal constraint satisfaction and alignment metrics compared to traditional single-active-prompt video generation methods.

---

[A Reflective Storytelling Agent for Older Adults: Integrating Argumentation Schemes and Argument Mining in LLM-Based Personalised Narratives](http://arxiv.org/abs/2605.10531)

- Reflective Storytelling Agent: introduces a hybrid architecture that integrates LLMs with symbolic knowledge and argumentation theory to generate personalized, purpose-driven narratives for older adults.
- The framework utilizes a reflective analysis layer that employs argument mining to assess narrative grounding, claim structure, and hallucination risk against a structured user model.
- The system incorporates User Interaction and Dialogue, User Model and Preferences, Knowledge and Normative Constraints, Narrative Planning, Narrative Realisation, Reflective Analysis, and Adaptation and Persistence components to ensure narrative coherence and transparency.

---

[PrimeKG-CL: A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs](http://arxiv.org/abs/2605.10529)

- PrimeKG-CL: introduces a benchmark for continual graph learning on evolving biomedical knowledge graphs, utilizing EWC, Multimodal Replay, CMKL, and LLM-RAG to address catastrophic forgetting.
- The framework evaluates ten continual learning methods across four KGE decoders, revealing that decoder choice and learning strategy interact significantly, often leading to performance degradation if mismatched.
- PrimeKG-CL provides a stratified evaluation of persistent, added, and removed edges, enabling precise analysis of how models retain valid knowledge while unlearning deprecated facts in dynamic biomedical environments.

---

[Consistency as a Testable Property: Statistical Methods to Evaluate AI Agent Reliability](http://arxiv.org/abs/2605.10516)

- Statistical Framework for Measuring Agent Consistency: introduces a rigorous measurement science for AI agent reliability by decomposing consistency into observable output-level and internal trajectory-level dimensions using U-statistics and kernel-based metrics.
- The framework utilizes Maximum Mean Discrepancy (MMD) and the Global Alignment Kernel (GAK) to detect structural and temporal inconsistencies in agent execution trajectories that traditional accuracy metrics fail to capture.
- By employing weighted Levenshtein distance and spectral clustering, the approach enables the diagnosis of specific failure modes, such as early-stage planning errors or late-stage aggregation drift, across diverse agentic benchmarks.

---

[SoK: A Systematic Bidirectional Literature Review of AI &amp; DLT Convergence](http://arxiv.org/abs/2605.10515)

- AI &amp; DLT Convergence Framework: introduces a bidirectional taxonomy classifying the integration of AI and DLT across five-layer architectural stacks for each technology.
- The paper synthesizes 53 peer-reviewed studies to identify recurring design patterns, including AI-driven consensus optimization and DLT-based federated learning coordination.
- The analysis reveals that while AI improves DLT adaptability and DLT enhances AI transparency, current research remains fragmented and lacks large-scale production deployment.

---

[A Theory of Multilevel Interactive Equilibrium in NeuroAI](http://arxiv.org/abs/2605.10505)

- MIE (Multilevel Interactive Equilibrium): introduces a game-theoretic framework for intelligent systems that generalizes Nash equilibrium by coupling neural learning, cognitive inference, and behavioral strategies across interacting agents.
- The framework models agents as hierarchical systems where stable interaction emerges when neural dynamics, cognitive belief models, and behavioral policies mutually stabilize.
- This approach provides a unified mathematical foundation for analyzing diverse interactive scenarios, including human-AI collaboration, brain-machine interfaces, and psychiatric disorders.

---

[SkillEvolver: Skill Learning as a Meta-Skill](http://arxiv.org/abs/2605.10500)

- SkillEvolver: introduces a lightweight, plug-and-play framework for online skill learning that enables a CLI-agent to iteratively author, deploy, and refine domain-specific skills without updating model weights.
- The framework utilizes a meta-skill to orchestrate a loop consisting of strategy-diversified exploration, contrastive skill update, and independent audit to produce reusable procedural artifacts.
- By grounding refinement in the observed behavior of a deployed Domain-Skill Agent rather than self-reflection, the system effectively captures procedural knowledge and improves agent performance across diverse benchmarks.

---

[DeepRefine: Agent-Compiled Knowledge Refinement via Reinforcement Learning](http://arxiv.org/abs/2605.10488)

- DeepRefine: introduces a general LLM-based reasoning model for agent-compiled knowledge refinement that improves pre-constructed knowledge bases using multi-turn interactions, abductive diagnosis, and targeted refinement actions.
- The framework utilizes a Gain-Beyond-Draft (GBD) reward and reinforcement learning to optimize refinement policies without requiring gold references for the knowledge base updates.
- DeepRefine performs defect localization and incremental updates through three reasoning steps: Answerability Judgement Loop, Error Abduction, and Refinement Actions Generation.

---

[OpenSGA: Efficient 3D Scene Graph Alignment in the Open World](http://arxiv.org/abs/2605.10484)

- OpenSGA: introduces a unified framework for 3D scene graph alignment that fuses vision-language, textual, and geometric features to predict object correspondences under partial overlap.
- The framework utilizes a distance-gated spatial attention encoder to capture node context and an MCF-based allocator to resolve many-to-one object associations.
- The authors also introduce ScanNet-SG, a large-scale dataset with over 700k samples, to support robust training and evaluation of scene graph alignment in open-world environments.

---

[Safe Multi-Agent Behavior Must Be Maintained, Not Merely Asserted: Constraint Drift in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2605.10481)

- CSG (Constraint State Governance): introduces a paradigm for LLM-based multi-agent systems that maintains safety-critical constraints as explicit, signed execution state to prevent constraint drift across trajectories.
- The framework utilizes Rule Tokenization, Maintained State, Action Proposal, Effect, Admission Checks, Decision and Audit, and Constraint Native Reinforcement Learning to ensure constraints remain fresh, inherited, enforceable, and auditable.
- By coupling governance with reinforcement learning, the system ensures that LLMs optimize for utility only within admissible trajectories, effectively mitigating safety failures that occur during multi-agent coordination and tool use.

---

[ASIA: an Autonomous System Identification Agent](http://arxiv.org/abs/2605.10480)

- ASIA (Autonomous System Identification Agent): introduces an agentic framework that automates the system identification pipeline by delegating model selection, architecture design, and hyperparameter tuning to an LLM-based coding agent.
- The framework utilizes an iterative experimentation loop to propose, implement, and evaluate model configurations based on a fixed evaluation protocol and historical experiment data.
- Experimental results on cascaded tank and nanodrone benchmarks demonstrate that ASIA discovers competitive, non-trivial model architectures and training strategies while outperforming traditional random search methods.

---

[Can Agent Benchmarks Support Their Scores? Evidence-Supported Bounds for Interactive-Agent Evaluation](http://arxiv.org/abs/2605.10448)

- Outcome-Evidence Reporting Layer: introduces an additive reporting framework that evaluates whether existing benchmark artifacts sufficiently support their reported success claims without modifying original tasks or agents.
- The framework utilizes a Case Checklist to categorize agent performance into Evidence Pass, Evidence Fail, or Unknown, thereby quantifying uncertainty and identifying benchmark-evaluator alignment issues.
- By maintaining Unknown records as visible diagnostic data rather than silent failures, the approach provides evidence-supported performance bounds that clarify the reliability of interactive-agent benchmarks.

---

[Statistical Model Checking of the Keynes+Schumpeter Model: A Transient Sensitivity Analysis of a Macroeconomic ABM](http://arxiv.org/abs/2605.10447)

- MultiVeStA: introduces a reproducible statistical workflow for analyzing complex macroeconomic agent-based models by decoupling the simulator from the statistical analysis engine.
- The framework utilizes MultiQuaTEx to specify temporal queries and employs automated stopping rules based on user-defined confidence and precision targets to determine simulation effort.
- This approach enables a principled transient sensitivity analysis of the Keynes+Schumpeter model, revealing that financial and structural parameters exert stronger effects than heuristic-switching mechanisms.

---

[TourMart: A Parametric Audit Instrument for Commission Steering in LLM Travel Agents](http://arxiv.org/abs/2605.10440)

- TourMart: introduces an applied intelligent-system audit instrument for measuring commission steering in LLM-based online travel agents using paired counterfactual replay and governance-parameter sweeps.
- The framework utilizes a symmetric six-gate producer audit to isolate generator-side failures from genuine commercial steering across a continuous governance grid.
- TourMart employs traveler-reader LLMs to extract perception features, which are then mapped by a deterministic welfare rule to quantify steering susceptibility.

---

[Position: Life-Logging Video Streams Make the Privacy-Utility Trade-off Inevitable](http://arxiv.org/abs/2605.10404)

- Pipeline-aware Privacy-Preserving System: introduces a framework that balances privacy and utility in life-logging video streams by utilizing a Privatization Representation Construction Module to generate a Minimal Sufficient Representation (MSR) via Pretrained Foundational Encoders, which is then transmitted through a Privatized Representation Communication Module to a Remote Server for secure processing.
- The framework emphasizes that Edge Devices must perform initial data privatization to ensure that only task-relevant, non-reconstructible representations are shared, thereby mitigating privacy risks associated with raw data exposure.
- The research demonstrates that non-private embeddings from standard encoders are vulnerable to inversion attacks, necessitating the adoption of pipeline-aware, attack-agnostic privacy designs for sustainable AI development.

---

[AnomalyClaw: A Universal Visual Anomaly Detection Agent via Tool-Grounded Refutation](http://arxiv.org/abs/2605.10397)

- AnomalyClaw: introduces a training-free VAD agent that decomposes anomaly detection into a multi-round refutation process using a Multi-round Refutation Agent, a Direct VLM Scorer, and an optional Verbalized Self-Evolution (OSR) extension.
- The system utilizes a 13-tool catalog to gather evidence for candidate features, which are then refuted against normal-sample references to produce a fine-grained anomaly score.
- AnomalyClaw improves cross-domain VAD performance by leveraging internal branch disagreement to build an online rulebook without requiring oracle labels or parameter updates.

---

[Agentic Performance at the Edge: Insights from Benchmarking](http://arxiv.org/abs/2605.10384)

- APE: introduces a domain-conditioned evaluation methodology for edge-deployed LLMs that systematically analyzes the impact of model size, generation, and variant type on agentic reliability using an ITBench-based Evaluation Harness, Observability Retrieval Tool, Topology Reasoning Tool, Candidate Reduction Tool, Cost Anomaly Analysis Tool, Root-Cause Evaluator, Inference Protocol, and Local Inference Endpoint.
- The study demonstrates that agentic performance at the edge is a complex systems property where domain-specific task difficulty and failure modes—semantic versus execution—often outweigh simple parameter-count scaling.
- The research provides empirical evidence that coder-oriented LLMs and strategic model routing can achieve significant latency reductions and reliability improvements in resource-constrained edge environments.

---

[PC3D: Zero-Shot Cooperation Across Variable Rosters via Personalized Context Distillation](http://arxiv.org/abs/2605.10377)

- PC3D: introduces a method for open-team cooperation that uses a set-structured centralized critic to distill personalized coordination contexts into decentralized recurrent actors.
- The framework employs a teacher module to generate coordination tokens via cross-attention, which are then personalized for each agent and distilled into decentralized policies to enable zero-shot adaptation across variable roster sizes.
- Agents recover these distilled contexts from local interaction histories and use gated FiLM modulation to adaptively condition their decision-making without requiring execution-time communication or global observations.

---

[Sleep Walk: A Three-Tier Benchmark for Stress-Testing Instruction-Guided Vision-Language Navigation](http://arxiv.org/abs/2605.10376)

- SleepWalk: introduces a benchmark for evaluating instruction-grounded trajectory prediction in single-scene 3D environments, utilizing Hunyuan3D-3.0, Qwen3-8B-VL, GPT-5-mini, TLControl, and MotionGPT.
- The framework evaluates LLMs on their ability to translate natural-language instructions into spatially coherent, executable paths across three tiers of difficulty.
- The benchmark employs a pointwise judge-based evaluation protocol to assess start-location consistency, goal satisfaction, obstacle avoidance, and trajectory efficiency.

---

[Approximate Envy-Free Allocations up to any k Goods](http://arxiv.org/abs/2605.10371)

- G3PA (Generalized Property Preserving Allocation): introduces a polynomial-time algorithm to achieve (k+1)/(k+2)-EFkX allocations for any number of agents with additive valuations.
- The framework utilizes a modified envy graph and a three-step process involving G3PA, AllocateAndEliminateCritical, and EnvyCycleElimination to ensure fair resource distribution.
- The research also establishes that EFkX orientations on graphs do not always exist and that deciding their existence is an NP-complete problem.

---

[AgentGR: Semantic-aware Agentic Group Decision-Making Simulator for Group Recommendation](http://arxiv.org/abs/2605.10367)

- AgentGR: introduces a novel agent-based framework that integrates semantic meta-path guided CoP reasoning, semantic-aware recognition for group topics and leadership, and multi-agent simulation strategies to model complex group decision-making processes.
- The framework utilizes LLMs to perform member-role-playing and simulate dynamic group interactions, effectively bridging the gap between high-order collaborative filtering signals and rich textual semantics.
- AgentGR provides two distinct simulation strategies—a static workflow for efficiency and a dynamic dialogue-based approach for precision—to accommodate diverse group recommendation requirements.

---

[EGL-SCA: Structural Credit Assignment for Co-Evolving Instructions and Tools in Graph Reasoning Agents](http://arxiv.org/abs/2605.10366)

- EGL-SCA: introduces a verifier-centric dual-space framework that models a graph reasoning agent using an instruction-side policy space for reasoning strategies and a tool-side program space for executable algorithmic tools.
- The framework utilizes structural credit assignment to map trajectory evidence to conditional updates, precisely routing failures to either prompt optimization or tool synthesis and repair.
- By co-evolving instructions and tools through a family-aware training curriculum and Pareto-style retention, the agent achieves high success rates on complex graph reasoning tasks.

---

[CellDX AI Autopilot: Agent-Guided Training and Deployment of Pathology Classifiers](http://arxiv.org/abs/2605.10362)

- CellDX AI Autopilot: introduces a platform that enables users to train, evaluate, and deploy pathology classifiers through natural language interaction with an AI Agent, utilizing Agent Skills, CellDX API, Azure ML GPU compute, Metric history (MongoDB), and a Deployed widget.
- The system leverages a structured set of agent skills to guide users through dataset curation, automated hyperparameter tuning, and multi-strategy model comparison while maintaining human-in-the-loop deployment controls.
- By operating on a pre-built feature store of over 32,000 cases, the platform removes the need for expensive feature extraction, allowing researchers to run parallel experiments cost-effectively on cloud infrastructure.

---

[How Mobile World Model Guides GUI Agents?](http://arxiv.org/abs/2605.10347)

- MWM (Mobile World Model): introduces a unified training pipeline for mobile GUI world models across four distinct modalities to provide agents with prospective foresight.
- The framework evaluates how different prediction formats—text-based, code-rendered, and diffusion-based—impact GUI agent performance, decision confidence, and task completion.
- Empirical results demonstrate that while world models improve agent foresight, their effectiveness is constrained by agent confidence, candidate diversity, and the fidelity of the simulated future states.

---

[TMAS: Scaling Test-Time Compute via Multi-Agent Synergy](http://arxiv.org/abs/2605.10344)

- TMAS: introduces a multi-agent framework for scaling test-time compute by organizing inference as a collaborative process among specialized agents that utilize hierarchical memory banks to coordinate information flow across trajectories.
- The framework employs an experience agent to manage low-level reasoning signals and a guideline agent to track high-level strategies, facilitating both the exploitation of reliable findings and the exploration of novel solution paths.
- A hybrid reinforcement learning system, incorporating rewards for correctness, experience utilization, and novel strategy exploration, further enhances the model's ability to perform sustained iterative reasoning.

---

[PaperFit: Vision-in-the-Loop Typesetting Optimization for Scientific Documents](http://arxiv.org/abs/2605.10341)

- PaperFit: introduces a vision-in-the-loop agent that automates scientific document typesetting by closing the sense–act–verify loop through multi-source evidence integration, constrained repair policy, and checklist-gated multi-round validation.
- The system addresses typesetting defects by fusing source, log, PDF, and page-image signals into structured records, enabling precise, constrained source-level revisions that avoid common pseudo-fixes.
- PaperFit-Bench provides a comprehensive evaluation framework with 200 papers across 10 templates, demonstrating that structured visual closed-loop control significantly outperforms existing document automation baselines.

---

[EmbodiSkill: Skill-Aware Reflection for Self-Evolving Embodied Agents](http://arxiv.org/abs/2605.10332)

- EmbodiSkill: introduces a training-free framework for embodied agents that utilizes skill-aware reflection and targeted revision to evolve procedural knowledge from task trajectories.
- The framework separates skill-body updates from skill-appendix refinements, ensuring that execution lapses are addressed by emphasizing existing valid guidance rather than incorrectly modifying core procedural rules.
- By employing a skill-aware evolution spiral, the agent iteratively improves its performance on complex household tasks by consolidating targeted reflection records into progressively more accurate and executable skill specifications.

---

[Verifiable Process Rewards for Agentic Reasoning](http://arxiv.org/abs/2605.10325)

- VPR (Verifiable Process Rewards): introduces a framework that converts symbolic or algorithmic oracles into dense, turn-level supervision for LLM agents to improve long-horizon credit assignment.
- The framework utilizes task-specific verifiers—including MCTS, constraint solvers, and posterior inference engines—to provide objective, noise-free feedback for each intermediate action.
- Empirical results demonstrate that VPR outperforms outcome-level and rollout-based baselines in controlled environments while fostering transferable reasoning skills for general and agentic benchmarks.

---

[Positive Alignment: Artificial Intelligence for Human Flourishing](http://arxiv.org/abs/2605.10310)

- Positive Alignment: introduces a paradigm shift in AI alignment by moving beyond mere harm avoidance toward the active cultivation of human and ecological flourishing through a multi-stage lifecycle approach.
- The framework utilizes Negative Attractors, Repellers, Satisficing Region, and Positive Attractors to map the transition from traditional safety constraints to proactive, value-aligned system behaviors.
- It proposes a full-stack alignment strategy that integrates Intentional Data Sourcing, Adaptive Constitutions, and multi-agent cooperation to ensure AI systems support long-term human well-being while maintaining epistemic humility and pluralistic governance.

---

[AgentRx: A Benchmark Study of LLM Agents for Multimodal Clinical Prediction Tasks](http://arxiv.org/abs/2605.10286)

- AgentRx: introduces a comprehensive benchmark for evaluating LLM-based agentic systems on multimodal clinical prediction tasks using Patient Summary, Electronic Health Records, Radiology Reports, and Chest X-rays.
- The framework evaluates performance across three settings: Single Agent Unimodal, Single Agent Multimodal, and Multi-Agent Multimodal, utilizing Unimodal Agents, Multimodal Agents, Multimodal Judge Agents, and Worker Agents.
- The study highlights that single agent frameworks outperform multi-agent systems in multimodal clinical prediction, while also identifying significant calibration challenges in decentralized multi-agent setups.

---

[MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading](http://arxiv.org/abs/2605.10268)

- MemReread: introduces a memory-guided LLM agent that enhances long-context reasoning by decomposing tasks into sub-questions and performing targeted rereading to recover discarded latent evidence.
- The framework utilizes a four-phase workflow—Read, Decompose, Integrate, and Answer—to maintain a bounded memory buffer while enabling non-linear reasoning through iterative information acquisition.
- To optimize computational efficiency, the paper introduces Rereading-Adaptive GRPO, a reinforcement learning strategy that dynamically modulates the number of rereading passes based on task complexity.

---

[Towards Autonomous Railway Operations: A Semi-Hierarchical Deep Reinforcement Learning Approach to the Vehicle Rescheduling Problem](http://arxiv.org/abs/2605.10257)

- Maze-Flatland: introduces a semi-hierarchical multi-agent RL framework that decomposes railway rescheduling into MADS (high-level dispatching) and MAPF (low-level routing) to improve scalability and learning stability.
- The framework utilizes a decision controller to enforce temporal ordering, where MADS manages global traffic density and MAPF handles local conflict resolution through path deviations.
- By separating decision scopes, the architecture mitigates the hard-exploration problem and reduces task interference, enabling effective performance in dense railway networks.

---

[Beyond Autonomy: A Dynamic Tiered AgentRunner Framework for Governable and Resilient Enterprise AI Execution](http://arxiv.org/abs/2605.10223)

- Dynamic Tiered AgentRunner: introduces a risk-adaptive, multi-role execution architecture that dynamically adjusts governance intensity to match task risk, enforces physical separation between proposal and execution, and builds resilience through systematic failure handling.
- The framework utilizes an OrchestratorAgent, WorkerAgent, CriticAgent, VerifierAgent, RecoveryAgent, RetrospectorAgent, ToolGateway, and Checkpoint to ensure governable and resilient enterprise AI execution.
- By treating failure as a first-class execution state and implementing a hard-wired ToolGateway, the framework transforms AI governance from prompt-level suggestions into robust system architecture guarantees.

---

[V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning](http://arxiv.org/abs/2605.10172)

- V-ABS: introduces a closed-loop beam search framework that integrates a Thinker, Actor, and Observer to mitigate imagination-action-observer bias in multi-step visual reasoning.
- The framework employs an entropy-based adaptive weighting algorithm to dynamically balance prior policy confidence from the Thinker with grounded visual feedback from the Observer.
- V-ABS utilizes a large-scale supervised fine-tuning dataset to improve action verification and achieves state-of-the-art performance across diverse visual search, navigation, and logic benchmarks.

---

[When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews](http://arxiv.org/abs/2605.10171)

- IMPACT (Intensity-based Multi-Agent Contradiction estimation): introduces a multi-agent framework that integrates ACEA, DIA-A/B, Intensity Agreement Checker, Disagreement Orchestrator, Adjudication Agent, and CVG to model fine-grained reviewer contradictions and their intensity.
- The framework utilizes a deliberative reasoning protocol where multiple agents debate conflicting assessments to mitigate bias and surface nuanced disagreements, which are then distilled into TIDE for efficient deployment.
- The authors also present RevCI, an expert-annotated benchmark of peer-review pairs with evidence-level contradiction annotations and graded intensity labels to support the fine-grained analysis of reviewer disagreement.

---

[Balancing Efficiency and Fairness in Traffic Light Control through Deep Reinforcement Learning](http://arxiv.org/abs/2605.10170)

- DDQN framework: introduces a reinforcement learning-based traffic light control agent that explicitly integrates fairness considerations between vehicular and pedestrian flows to reduce congestion.
- The approach utilizes a composite reward function with a fairness trade-off coefficient to dynamically balance waiting times for different road user categories.
- Experimental results demonstrate that the agent outperforms fixed-time traffic light controllers by autonomously adapting to varying traffic conditions in a simulated four-way intersection.

---

[NyayaAI: An AI-Powered Legal Assistant Using Multi-Agent Architecture and Retrieval-Augmented Generation](http://arxiv.org/abs/2605.10155)

- NyayaAI: introduces a multi-agent legal assistant that leverages the Mastra TypeScript framework, RAG Knowledge Base, and Claude SDK to automate and simplify legal workflows for Indian legal resources.
- The system architecture integrates a main Nyaya agent with specialized sub-agents for research, summarization, case analysis, and drafting, all validated by a compliance module.
- Evaluation results demonstrate that the structured multi-agent LLM approach achieves 70% domain classification precision and 74% RAG retrieval precision within the Indian legal domain.

---

[Is DRL-based MAC Ready for Underwater Acoustic Networks? Exploring Its Practicality in Real Field Experiments](http://arxiv.org/abs/2605.10144)

- EA-MAC: introduces a DRL-based protocol for underwater acoustic networks that utilizes a DQN-agent, an aggregated ACK mechanism, a transmission queue, a fairness penalty module, a Bayesian inference module, and an observation completion module to achieve efficient, autonomous, and fair channel access.
- The framework addresses challenges in underwater environments, such as uncertain reward acquisition delays and incomplete observations, by employing Bayesian inference and a sliding window observation approach to enhance decision-making robustness.
- EA-MAC balances network throughput and transmission fairness by dynamically adjusting node access strategies based on local information and overheard data, outperforming existing DRL-based protocols in both simulation and real-world field experiments.

---

[Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](http://arxiv.org/abs/2605.10118)

- SAGE: introduces a generative experience-driven learning paradigm that enables agents to build robust navigation capabilities by distilling high-level VLM reasoning into structured embodied experiences within a physics-grounded sandbox.
- The framework utilizes a three-phase approach comprising Genesis for autonomous task synthesis, Evolution for policy optimization via Asymmetric Adaptive Clipping, and Navigation for retrieval-augmented decision-making.
- SAGE effectively bridges the modality gap between semantic reasoning and low-level robot control, achieving significant performance gains on A-EQA and GOAT-Bench benchmarks while demonstrating robust Sim2Real transfer.

---

[ViSRA: A Video-based Spatial Reasoning Agent for Multi-modal Large Language Models](http://arxiv.org/abs/2605.10106)

- ViSRA: introduces an inference-time, human-aligned agent framework that enhances the spatial intelligence of LLMs by dynamically orchestrating specialized perception tools without requiring post-training.
- The framework utilizes a multi-role agent architecture comprising a Planner, Reflector, Executor, and Summarizer to decompose complex spatial queries into verifiable subproblems.
- ViSRA leverages a suite of spatial tools, including 2D/3D detection, tracking, and scene modeling, to construct explicit spatio-temporal evidence, achieving superior generalization on unseen tasks compared to post-trained models.

---

[RFAmpDesigner: A Self-Evolving Multi-Agent LLM Framework for Automated Radio Frequency Amplifier Design](http://arxiv.org/abs/2605.10093)

- RFAmpDesigner: introduces a multi-agent framework that automates RF amplifier sizing by reframing high-dimensional parameter tuning as a low-dimensional resource-allocation problem managed by RFAmpManager, RFAmpSearcher, RFAmpRefiner, Tool Middleware, Knowledge Base, Experience Base, and RAG.
- The framework utilizes a two-tier multi-agent architecture to decouple heavy simulation tasks from lightweight reasoning, enabling efficient search and refinement through RAG-based knowledge reuse.
- RFAmpDesigner achieves significant improvements in sample efficiency and runtime by leveraging domain-specific tool middleware and agentic decomposition to navigate sparse feasible regions in RF circuit design.

---

[Agentic Fuzzing: Opportunities and Challenges](http://arxiv.org/abs/2605.10074)

- AFuzz: introduces a bug-finding approach that uses deep agents as the primary reasoning engine, seeded by historical bugs, to identify complex logic bugs through code reasoning, hypothesis generation, and verification.
- The framework utilizes a four-stage LLM agent pipeline—Analyzer, Investigator, Scenario Analyzer, and Validator—to systematically analyze reference bugs and discover variants across different code implementations.
- To optimize efficiency, AFuzz incorporates a DPP-MAP seed scheduler for diversity and a scenario coverage tracker to prevent redundant investigations across the target codebase.

---

[MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs](http://arxiv.org/abs/2605.10064)

- MAGE: introduces a framework that externalizes self-knowledge into a co-evolutionary knowledge graph (EVOKG) to enable frozen LLMs to improve across iterations.
- The framework utilizes a dual-memory index within the experience subgraph, storing both self-harvested success traces and teacher-written failure corrections to guide the frozen execution tier.
- MAGE couples graph updates with task-level search and skill-level routing bandits, all driven by a shared reward stream to ensure stable, append-only evolution of the retrieval substrate.

---

[EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force Guided Curriculum Learning](http://arxiv.org/abs/2605.10063)

- EFGCL: introduces a reinforcement learning framework that accelerates dynamic motor skill acquisition by applying spotting-inspired external forces to guide agents toward successful trajectories.
- The framework utilizes an adaptive curriculum that gradually decays assistive forces based on the success rate, enabling a smooth transition to autonomous motion generation.
- EFGCL employs a Teacher-Student architecture where the teacher policy is trained with privileged information and external forces, while the student policy is distilled to operate using only onboard sensor observations.

---

[Strategic Exploitation in LLM Agent Markets: A Simulation Framework for E-Commerce Trust](http://arxiv.org/abs/2605.10059)

- TruthMarketTwin: introduces a controlled simulation framework for studying LLM-agent behavior in e-commerce markets under asymmetric information.
- The framework incorporates Seller Agents, Buyer Agents, Marketplace Architecture, Reputation System, Warrant Enforcement, Communication Channels, and Persistent Memory to evaluate strategic deception and institutional governance.
- Research findings demonstrate that LLM agents autonomously exploit reputation vulnerabilities, while warrant enforcement effectively reshapes strategic reasoning and improves market welfare under economic pressure.

---

[Route by State, Recover from Trace: STAR with Failure-Aware Markov Routing for Multi-Agent Spatiotemporal Reasoning](http://arxiv.org/abs/2605.10057)

- STAR (Spatio-Temporal Agent Router): introduces a failure-aware routing framework that externalizes inter-agent control as a state-conditioned transition policy over the current agent, task type, and typed execution status.
- The framework utilizes an agent routing matrix that combines expert-specified nominal routes with recovery transitions learned from execution traces to handle distinct failure modes like malformed outputs, missing dependencies, and tool-query mismatches.
- Specialists execute through an extract-compute-deposit protocol, writing intermediate results to a shared blackboard, which enables the router to treat recovery as an explicit routing problem rather than an implicit byproduct of LLM generation.

---

[Swarm Skills: A Portable, Self-Evolving Multi-Agent System Specification for Coordination Engineering](http://arxiv.org/abs/2605.10052)

- Swarm Skills: introduces a portable, self-evolving specification for multi-agent coordination that decouples coordination logic from runtime environments using SKILL.md, roles/, workflow.md, bind.md, and evolutions.json.
- The framework utilizes a self-evolution algorithm comprising a Trajectory Aggregator, Trajectory Analyzer, and Governance Engine to autonomously refine coordination protocols based on multi-dimensional scoring of Effectiveness, Utilization, and Freshness.
- By adhering to the Anthropic Skills standard and progressive disclosure, Swarm Skills enables zero-adapter portability and backward compatibility across diverse LLM-based agent execution environments.

---

[Adaptive Action Chunking via Multi-Chunk Q Value Estimation](http://arxiv.org/abs/2605.10044)

- ACH: introduces a novel offline-to-online RL framework that dynamically modulates action chunk length during training and inference to optimize performance based on current state dynamics.
- The framework utilizes a Transformer-based Value Function to simultaneously estimate action-values for all candidate chunk lengths in a single forward pass.
- An Adaptive Sampling Mechanism selects the most effective chunk length by balancing future state uncertainty and policy stochasticity jittering, consistently outperforming fixed-length baselines.

---

[TimeClaw: A Time-Series AI Agent with Exploratory Execution Learning](http://arxiv.org/abs/2605.10038)

- TimeClaw: introduces an exploratory execution learning framework that transforms time-series agent exploration into reusable hierarchical distilled experience through a four-stage loop of Explore, Compare, Distill, and Reinject.
- The framework utilizes metric-supervised execution comparison and task-aware tool dropout to mitigate tool-prior collapse, ensuring that agents learn from quantitative execution quality rather than completion alone.
- TimeClaw externalizes learned strategies into structured memory, skills, and notes, which are reinjected into the agent's prompt at inference time to improve reasoning without requiring online model updates.

---

[Bridging the Cognitive Gap: A Unified Memory Paradigm for 6G Agentic AI-RAN](http://arxiv.org/abs/2605.10036)

- AI-RAN: introduces a memory-centric architecture that utilizes a unified memory fabric to bridge the cognitive gap between microsecond-level reflexes, millisecond-level reasoning, and long-term evolution in 6G networks.
- The framework maps biological memory hierarchies—sensory, working, and long-term—onto heterogeneous computing fabrics using CXL interconnects to enable zero-copy state sharing across cognitive loops.
- By replacing message-based interfaces with shared memory, the architecture eliminates spatial blindness and temporal amnesia, allowing agents to perform cause-aware control and proactive adaptation.

---

[Beyond Self-Play and Scale: A Behavior Benchmark for Generalization in Autonomous Driving](http://arxiv.org/abs/2605.10034)

- BehaviorBench: introduces a comprehensive benchmarking framework for autonomous driving that addresses evaluation, complexity, and behavior diversity through an Evaluation Interface, Scenario Splits, Traffic Agents, and a Hybrid Planner.
- The framework enables at-scale RL policy evaluation by bridging PufferDrive with established benchmarks like nuPlan and interPlan.
- It demonstrates that self-play RL agents often overfit to training opponents and proposes a hybrid planner to improve generalization and safety in interactive traffic scenarios.

---

[Combining Mechanical and Agentic Specification Inference for Move](http://arxiv.org/abs/2605.10005)

- MoveFlow: introduces a hybrid specification inference methodology that combines mechanical weakest-precondition analysis with an agentic coding assistant to automate the generation of formal specifications for Move smart contracts.
- The framework utilizes a Model Context Protocol service to integrate the Move Prover as a mechanical baseline, allowing the LLM-based agent to focus on complex tasks like loop invariant synthesis and specification simplification.
- The system employs a propose-observe-refine loop where the Move Prover acts as a verification oracle, providing feedback to the agent to iteratively refine specifications and resolve verification timeouts.

---

[Continual Harness: Online Adaptation for Self-Improving Foundation Agents](http://arxiv.org/abs/2605.09998)

- Continual Harness: introduces a reset-free framework that automates the refinement of an agent's scaffolding by alternating between acting in an Environment and using a Refiner to update the Harness components (System prompt, Sub-agents, Skills, Memory) based on trajectory data.
- The framework incorporates an online co-learning loop where an Agent model and the Harness state are jointly updated through a process reward model and teacher relabeling of trajectory windows.
- By enabling mid-episode, reset-free self-improvement, the system allows agents to overcome long-horizon partial-observability challenges in complex RPG environments without requiring manual scaffolding.

---

[A Resource Allocation Game and its Equilibrium Strategies](http://arxiv.org/abs/2605.09988)

- RAG (Resource Allocation Game): introduces a Bayesian game framework for allocating resources among autonomous players using AIF functions to determine optimal request strategies under incomplete information.
- The framework utilizes a construction algorithm to identify Nash equilibria, incorporating mean-field and Gaussian approximations for large-scale systems.
- The model identifies a chattering regime where players employ a continuous, strictly increasing strategy to sustain high payoffs when standard identity or flat strategies are insufficient.

---

[Towards Generalist Game Players: An Investigation of Foundation Models in the Game Multiverse](http://arxiv.org/abs/2605.09965)

- Generalist Game Player Framework: introduces a comprehensive, end-to-end lifecycle for game-playing AI, organizing the field into four interdependent pillars: Dataset, Model, Harness, and Benchmark.
- The paper formalizes the evolution of game-playing AI through a Goal-Conditioned POMDP, tracing the transition from brittle specialists to generalist agents capable of omni-reality adaptability.
- It identifies five fundamental trade-offs bottlenecking the field and proposes a five-level roadmap, progressing from single-game mastery to the ultimate Demiurge stage where agents generate and evolve their own game multiverses.

---

[Generating synthetic electronic health record data using agent-based models to evaluate machine learning robustness under mass casualty incidents](http://arxiv.org/abs/2605.09951)

- ED ABM (Emergency Department Agent-Based Model): introduces a mechanistic simulation approach to generate synthetic EHR data for evaluating the robustness of ML models under mass casualty incidents.
- The framework utilizes real-world EHR data to calibrate an emergency department environment, allowing for the systematic manipulation of system conditions such as patient arrival rates and resource availability.
- By simulating these conditions, the approach enables stress testing of ML models to identify performance vulnerabilities, specifically declines in recall, that are not captured by standard held-out test sets.

---

[HAGE: Harnessing Agentic Memory via RL-Driven Weighted Graph Evolution](http://arxiv.org/abs/2605.09942)

- HAGE: introduces a weighted multi-relational memory framework that reconceptualizes retrieval as a sequential, query-conditioned traversal over a dynamic graph, utilizing a Weighted Multi-Relational Memory Graph, Event-Nodes, Trainable Edge Features, a QueryRouter, a Reinforcement Learning-Based Training Framework, an LLM-based Classifier, and a Context Synthesis Module.
- The framework optimizes memory access by jointly training edge representations and routing policies via reinforcement learning, allowing the agent to prioritize structurally critical paths based on query intent and downstream feedback.
- Empirical results demonstrate that HAGE improves long-horizon reasoning accuracy and provides a favorable accuracy-efficiency trade-off compared to existing agentic memory systems.

---

[Population Protocols over Ordered Agents](http://arxiv.org/abs/2605.09937)

- PP[N] (Population Protocols over Ordered Agents): introduces a distributed computation model where agents are totally ordered and interactions are restricted by predicates, establishing that the immediate observation fragment IO-PP[&lt;] is equivalent to unambiguous star-free languages.
- The paper characterizes the expressive power of these protocols, showing that IO-PP[N] with successor predicates equals NSPACE(n), while providing logic and automaton models for the class PP[&lt;].
- The authors investigate the well-specification problem, proving that checking if a protocol is a decider is undecidable for PP[&lt;] and IO-PP[+1], but conditionally decidable for IO-PP[&lt;].

---

[TRACER: Verifiable Generative Provenance for Multimodal Tool-Using Agents](http://arxiv.org/abs/2605.09934)

- TRACER: introduces a framework for verifiable generative provenance in multimodal tool-using agents by linking answer claims to specific tool turns, evidence units, and semantic support relations.
- The framework utilizes a Policy Model to generate sentence-level provenance records, which are then validated by a Traceability Verifier to ensure source authenticity and relation rationality.
- TRACER employs a Reward Module to provide provenance-aware feedback, enabling reinforcement learning that optimizes for both answer accuracy and efficient, evidence-grounded tool usage.

---

[FOCUSFT: Bilevel Optimization for Dilution-Aware Long-Context Fine-Tuning](http://arxiv.org/abs/2605.09932)

- FOCUSFT: introduces a bilevel optimization framework that mitigates training-time attention dilution by using an Inner Loop to construct a Parametric Memory via LoRA Fast-Weight Adapters, which guides the Outer Loop to produce a more informative gradient signal.
- The framework employs Bidirectional Context Attention to eliminate the causal asymmetry that drives attention sinks, while maintaining Causal Masking for response generation to ensure inner-outer consistency.
- By sharpening the attention distribution during training, FOCUSFT enables LLMs to better utilize long-context information without requiring additional inference-time computation.

---

[Fair Allocation under Conflict Constraints](http://arxiv.org/abs/2605.09930)

- Fair Allocation under Conflict Constraints: introduces a framework for distributing indivisible items represented as vertices in a conflict graph, where agents receive independent sets of items to satisfy fairness and efficiency criteria.
- The paper utilizes a color-switching technique and a cycle-plus-n-cliques graph construction to prove existence and computational complexity results for EF1 and EF[1,1] allocations under various valuation profiles.
- The research establishes that while maximal EF1 allocations exist for two agents with monotone valuations, the problem becomes NP-hard for larger agent groups, necessitating the use of relaxed fairness notions like EF[1,1] on path graphs.

---

[TeleResilienceBench: Quantifying Resilience for LLM Reasoning in Telecommunications](http://arxiv.org/abs/2605.09929)

- TeleResilienceBench: introduces a benchmark to quantify reasoning resilience in LLMs by evaluating their ability to correct flawed, truncated reasoning traces within telecommunications domain tasks.
- The framework utilizes a weak generator to produce corrupted reasoning paths, which are then presented to target models to measure their Correct Flip Rate (CFR), No Flip Rate (NFR), and Wrong Flip Rate (WFR).
- Experimental results across multiple LLM families demonstrate that reasoning resilience is a distinct capability from standard accuracy, with smaller models like Nemotron-3-nano 4b often outperforming significantly larger models in error recovery.

---

[Position: Academic Conferences are Potentially Facing Denominator Gaming Caused by Fully Automated Scientific Agents](http://arxiv.org/abs/2605.09915)

- Agentic Conference Denominator Gaming: introduces a systemic threat where malicious actors use a multi-agent pipeline to inflate conference submission pools, thereby manipulating acceptance rates to favor targeted papers.
- The framework utilizes a Research Agent for high-volume generation of stylistically plausible manuscripts and a Submission Agent for automated, large-scale submission via platform APIs.
- This approach exploits the stable acceptance rate norm of academic conferences, creating an economically asymmetric attack that threatens to overwhelm peer-review systems and erode scientific trust.

---

[RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](http://arxiv.org/abs/2605.09907)

- RADAR: introduces an iterative, step-wise generative framework that utilizes conditional graph diffusion models to synthesize task-adaptive multi-agent communication topologies while actively minimizing redundant information flow.
- The framework incorporates an effective size metric to guide the denoising network in constructing low-redundancy collaboration graphs, balancing structural expressiveness with communication efficiency.
- By employing a reinforcement learning-based training strategy, RADAR optimizes both the structural fidelity of the generated communication graphs and the downstream task utility of the multi-agent system.

---

[Deterministic vs. LLM-Controlled Orchestration for COBOL-to-Python Modernization](http://arxiv.org/abs/2605.09894)

- ATLAS (Autonomous Transpilation for Legacy Application Systems): introduces a controlled experimental framework to compare deterministic and LLM-controlled orchestration strategies for COBOL-to-Python modernization.
- The framework isolates execution control as the sole experimental variable by maintaining constant LLMs, prompts, tools, and configurations across both deterministic and agentic workflows.
- Empirical results demonstrate that deterministic orchestration improves worst-case robustness and reduces token consumption by up to 3.5x compared to LLM-controlled orchestration, while maintaining comparable functional correctness.

---

[Pseudo-Deliberation in Language Models: When Reasoning Fails to Align Values and Actions](http://arxiv.org/abs/2605.09893)

- VALDI: introduces a framework for measuring the value–action gap in LLMs by comparing stated values against generated dialogue across 4,941 scenarios.
- VIVALDI: introduces a multi-agent value auditor that intervenes at different generation stages to improve alignment between LLM reasoning and final actions.
- The research identifies pseudo-deliberation as a failure mode where explicit reasoning can degrade value–action alignment, demonstrating that post-hoc auditing of final outputs is more effective than reasoning-level interventions.

---

[Skill Description Deception Attack against Task Routing in Internet of Agents](http://arxiv.org/abs/2605.09889)

- SDD (Skill Description Deception) attack framework: introduces a security vulnerability in Internet of Agents (IoA) systems where malicious agents manipulate self-declared skill descriptions to bias semantic task routing, utilizing an LLM-based Query Generator, a Skill Description Optimizer, and a Semantic Router.
- The framework leverages an iterative optimization process to align malicious skill descriptions with target-domain queries, effectively hijacking task delegation from benign agents without requiring access to internal routing parameters.
- Experimental results across nine domains demonstrate that the SDD attack achieves up to 98% success rate, highlighting the critical need for trust-aware routing and capability verification in LLM-driven multi-agent ecosystems.

---

[M2A: Synergizing Mathematical and Agentic Reasoning in Large Language Models](http://arxiv.org/abs/2605.09879)

- M2A: introduces a training-free paradigm that synergizes mathematical and agentic reasoning in LLMs by merging task vectors into the null space of agent-critical behavior subspaces.
- The framework utilizes Agent-critical behavior calibration, Null-space projected merging, and Adaptive layer-wise merging to enhance internal reasoning while preserving the multi-turn interaction patterns of LLMs.
- By employing a Similarity-aware layer mask and Merge coefficient calibration, M2A provides a controllable interface for modulating reasoning depth without requiring additional gradient-based training.

---

[EGOMEMREASON: A Memory-Driven Reasoning Benchmark for Long-Horizon Egocentric Video Understanding](http://arxiv.org/abs/2605.09874)

- EGOMEMREASON: introduces a benchmark for evaluating long-horizon memory in egocentric videos, utilizing Evidence Preparation, Memory-Centric Question Generation, Automatic Filtering, and Human Verification to assess entity-, event-, and behavior-memory.
- The framework evaluates LLMs and agentic systems on their ability to aggregate evidence across week-long temporal horizons to solve complex reasoning tasks.
- Experimental results reveal that current models struggle with long-horizon memory, with performance bottlenecks varying across entity, event, and behavior memory types.

---

[ConsistNav: Closing the Action Consistency Gap in Zero-Shot Object Navigation with Semantic Executive Control](http://arxiv.org/abs/2605.09869)

- ConsistNav: introduces a training-free framework for zero-shot ObjectNav that utilizes a semantic executive to bridge the action consistency gap between perception and navigation.
- The framework integrates Persistent Candidate Memory, a Finite-State Executive Controller, and Stability-Aware Action Control to maintain stable object hypotheses and robust navigation commitments.
- By enforcing a reliability contract between perception and planning, the system improves success rates and path efficiency without requiring retraining of the underlying detector or planner.

---

[Nautilus Compass: Black-box Persona Drift Detection for Production LLM Agents](http://arxiv.org/abs/2605.09863)

- Nautilus Compass: introduces a black-box persona drift detection system that operates in user-space hooks to monitor LLM agent consistency without requiring access to model weights.
- The system utilizes a BGE-m3 Embedder Daemon to compute cosine similarity between user prompts and task-shaped behavioral anchors, aggregating results via a weighted top-k mean to generate drift scores.
- Nautilus Compass integrates memory recall, drift scoring, and strategy retrieval into a unified pipeline that injects behavioral context into the LLM prompt to mitigate persona drift in production agents.

---

[EnactToM: An Evolving Benchmark for Functional Theory of Mind in Embodied Agents](http://arxiv.org/abs/2605.09826)

- EnactToM: introduces an evolving benchmark for functional Theory of Mind in embodied multi-agent settings, utilizing Embodied Agents, 3D Household Environment, PDDL Goal Formula, Epistemic Operators, Generation Agent, LLM Judge Council, Structural Calibrator, Task Pool, Literal ToM Probes, and Functional ToM Evaluation.
- The framework evaluates the gap between an agent's ability to report beliefs (Literal ToM) and its capacity to act on those beliefs during grounded multi-agent coordination (Functional ToM).
- The benchmark employs an autonomous generation pipeline that uses PDDL-verified tasks and failure-biased evolution to maintain difficulty as LLMs improve.

---

#### 10th May 2026

[CalBench: Evaluating Coordination–Privacy Trade-offs in Multi-Agent LLMs](http://arxiv.org/abs/2605.09823)

- CalBench: introduces a decentralized multi-agent evaluation environment for calendar scheduling that requires agents to coordinate under private information constraints while balancing coordination quality against privacy leakage.
- The framework utilizes an oracle-anchored evaluation harness to measure performance across success rate, communication efficiency, coordination quality, fairness, and privacy preservation using the VPS metric.
- Experimental results across seven LLMs demonstrate a systematic privacy-efficiency tradeoff, where models that share more precise cost information achieve better coordination but suffer higher privacy leakage.

---

[Oracle Poisoning: Corrupting Knowledge Graphs to Weaponise AI Agent Reasoning](http://arxiv.org/abs/2605.09822)

- Oracle Poisoning: introduces an attack class where adversaries corrupt structured knowledge graphs to manipulate AI agent reasoning via trusted tool-use protocols.
- The research demonstrates that LLMs consistently treat poisoned graph data as ground truth, leading to incorrect conclusions despite sound reasoning processes.
- The authors propose a VUT (Visibility-Understanding-Traceability) framework to mitigate these vulnerabilities, emphasizing read-only access control and multi-tool cross-verification as primary defenses.

---

[Learning to Compress Time-to-Control: A Reinforcement Learning Framework for Chronic Disease Management](http://arxiv.org/abs/2605.09818)

- Chronic Care RL Framework: introduces a two-loop reinforcement learning architecture that couples clinician preference learning with an execution-intensity-constrained MDP to optimize chronic disease management.
- The framework utilizes a tiered reward hierarchy calibrated to the CMS ACCESS model to improve credit assignment and a two-layer action taxonomy to distinguish between clinical and operational interventions.
- An engagement harness serves as a supervisory layer to bound agent autonomy by routing high-risk or high-uncertainty decisions to clinicians while permitting autonomous execution of routine operational tasks.

---

[Efficient Multi-Robot Motion Planning with Precomputed Translation-Invariant Edge Bundles](http://arxiv.org/abs/2605.09801)

- KiTE-Extend: introduces a planner-agnostic action selection mechanism that leverages precomputed translation-invariant edge bundles to accelerate kinodynamic node expansion in sampling-based motion planning.
- The framework reframes node expansion as a retrieval and ranking problem, allowing planners to reuse dynamically feasible trajectory segments while preserving the theoretical guarantees of the underlying sampling-based approach.
- Experimental results demonstrate that integrating KiTE-Extend into centralized, prioritized, and conflict-based MRMP paradigms consistently improves success rates, reduces computation time, and enhances solution quality across diverse kinodynamic systems.

---

[LLM Agents Enable User-Governed Personalization Beyond Platform Boundaries](http://arxiv.org/abs/2605.09794)

- User-Governed Personalization: introduces a paradigm shift where users aggregate fragmented cross-platform data to enable holistic personalization via an LLM Agent.
- The framework leverages the User as the unique integration point, utilizing Cross-Platform Data processed by an LLM Agent to generate a Personalization Output that transcends individual platform silos.
- Empirical results demonstrate that this approach outperforms single-platform baselines by revealing user interests invisible to isolated recommendation systems.

---

[Attribution-based Explanations for Markov Decision Processes](http://arxiv.org/abs/2605.09780)

- Attribution-based Explanations for Markov Decision Processes framework: introduces a formal characterization and optimization-based approach to compute importance scores for states and execution paths in MDPs.
- The framework utilizes strategy synthesis to quantify the influence of specific states and paths on reaching a target, addressing the limitations of static attribution methods in sequential decision-making.
- The approach provides three distinct encodings—QP, QP*, and LP*—to efficiently compute importance bounds, demonstrating scalability on complex models with up to 10,000 states and transitions.

---

[UTS at PsyDefDetect: Multi-Agent Councils and Absence-Based Reasoning for Defense Mechanism Classification](http://arxiv.org/abs/2605.09769)

- UTS at PsyDefDetect: introduces a multi-phase deliberative council of LLMs that classifies psychological defense mechanisms by evaluating evidence strength through structured advocacy and formal role decomposition.
- The architecture utilizes a Clinical Analyst, Mechanism Specialist, and Pattern Analyst for initial assessment, followed by class-specific advocates and a resolution function to mitigate majority-class bias.
- An override ensemble, comprising builder agents, a critic agent, and a regression guard, provides high-confidence corrections to the council's predictions, significantly improving classification performance on imbalanced datasets.

---

[SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis](http://arxiv.org/abs/2605.09768)

- SAGE: introduces a training-free, agentic diagnostic pipeline that utilizes a source-cited knowledge base to perform explainable plant disease diagnosis.
- The framework employs a multi-turn reasoning agent that observes test images, narrows candidates via an anatomical index, and performs sequential comparisons against reference images to produce a verifiable reasoning trace.
- By grounding predictions in source-cited symptom descriptions and reference images, the system achieves significant accuracy improvements over baseline models without requiring task-specific training.

---

[RubricRefine: Improving Tool-Use Agent Reliability with Training-Free Pre-Execution Refinement](http://arxiv.org/abs/2605.09730)

- RubricRefine: introduces a training-free, pre-execution semantic contract verification method that uses task-specific rubrics to iteratively repair code-mode agent programs before execution.
- The framework employs a generator-verifier loop where the verifier provides structured item-level feedback based on a generated rubric, enabling targeted repairs of inter-tool contract violations.
- RubricRefine achieves high reliability in single-attempt settings by utilizing early stopping triggered when candidate code satisfies all rubric criteria, significantly reducing latency and LM call usage.

---

[Metal-Sci: A Scientific Compute Benchmark for Evolutionary LLM Kernel Search on Apple Silicon](http://arxiv.org/abs/2605.09708)

- METAL-SCI: introduces a 10-task scientific compute benchmark for Apple Silicon that utilizes a Frozen LLM, Runtime Compiler, Dispatch, Score, Feedback, Evolutionary Loop, and Held-out Gate to optimize GPU kernels.
- The framework employs a (1+1) evolutionary loop where a Frozen LLM iteratively refines kernels based on structured Feedback from a Runtime Compiler and Dispatch mechanism.
- A Held-out Gate provides an external oversight primitive, evaluating the final kernel on unseen configurations to detect silent correctness violations or performance regressions that in-distribution scoring misses.

---

[MOTOR-Bench: A Real-world Dataset and Multi-agent Framework for Zero-shot Human Mental State Understanding](http://arxiv.org/abs/2605.09703)

- MOTOR-MAS: introduces a structured multi-agent framework that coordinates specialized Behavior Agent, Cognition Agent, and Emotion Agent to infer human mental states in collaborative learning.
- The framework utilizes a staged reasoning process where intermediate predictions from earlier agents serve as anchors to support subsequent inferences by later agents.
- By grounding agent communication in self-regulated learning theory, the system effectively reduces hallucinations and improves performance on latent mental state inference compared to black-box LLMs.

---

[Ambig-DS: A Benchmark for Task-Framing Ambiguity in Data-Science Agents](http://arxiv.org/abs/2605.09698)

- Ambig-DS: introduces a diagnostic benchmark to evaluate whether data-science agents can recognize and resolve task-framing ambiguity before executing pipelines.
- The framework utilizes two diagnostic suites, Ambig-DS-Target and Ambig-DS-Objective, to measure silent failure modes where LLMs commit to unintended task framings without flagging underspecification.
- Experimental results across five frontier LLMs demonstrate that while agents can utilize clarification when provided, they struggle to reliably detect when to ask, often defaulting to silent, suboptimal commitments.

---

[Unpredictability dissociates from structured control in language agents](http://arxiv.org/abs/2605.09692)

- Language Agent Family: introduces a lesionable framework to test whether stochastic sampling can substitute for structured control mechanisms that couple reasons, memory, self-state, and inhibition to action selection.
- The study demonstrates that increasing stochasticity raises action entropy but fails to reproduce the structured-control profile maintained by the agent's internal components.
- Experimental results across seven datasets and multiple LLMs confirm that structured action control remains distinct from stochastic unpredictability, even when controlling for reporting format and entropy.

---

[MonitoringBench: Semi-Automated Red-Teaming for Agent Monitoring](http://arxiv.org/abs/2605.09684)

- MonitoringBench: introduces a semi-automated red-teaming methodology that decomposes attack construction into Strategy Generation, Execution, and Refinement Pipeline to expose harder-to-catch attacks for LLM-based agent monitors.
- The framework utilizes a Reconnaissance Agent equipped with a Monitor Tool and Think Tool to iteratively optimize attack strategies against development monitors, effectively bridging the conceive-execute gap.
- By applying this pipeline to the BashArena environment, the authors produce a difficulty-graded benchmark of 2,644 attack trajectories that reveals critical monitor failure modes such as susceptibility to persuasion and scoring calibration errors.

---

[DeepTumorVQA: A Hierarchical 3D CT Benchmark for Stage-Wise Evaluation of Medical VLMs and Tool-Augmented Agents](http://arxiv.org/abs/2605.09679)

- DeepTumorVQA: introduces a hierarchical 3D CT benchmark that decomposes medical reasoning into recognition, measurement, visual reasoning, and medical reasoning stages to diagnose model failures.
- The framework evaluates both direct VLM inference and tool-augmented agents using segment_organ, measure, lookup_medical_knowledge, crop_region, and a ReAct-style agent loop to isolate performance bottlenecks.
- Benchmarking over 30 model configurations reveals that reliable quantitative measurement is the primary bottleneck in 3D CT diagnostic tasks, which tool augmentation and trace supervision effectively mitigate.

---

[CodeClinic: Evaluating Automation of Coding Skills for Clinical Reasoning Agents](http://arxiv.org/abs/2605.09675)

- CodeClinic: introduces a benchmark and autoformalization pipeline for evaluating how LLMs synthesize and compose reusable clinical skills from natural-language guidelines to automate EHR reasoning.
- The framework utilizes an offline autoformalization pipeline where an LLM agent, supported by a ReACT loop and a verification agent, transforms clinical guidelines into a verified Python function library.
- This approach enables LLMs to perform longitudinal ICU surveillance and compositional information seeking by invoking verified, reusable code instead of relying on static toolboxes or zero-shot code generation.

---

[Workspace Optimization: How to Train Your Agent](http://arxiv.org/abs/2605.09650)

- DREAMTEAM: introduces a workspace optimization framework where frozen LLMs adapt by editing a structured, typed workspace of executable artifacts instead of updating model weights.
- The framework utilizes a multi-agent harness comprising observer-, simulator-, explorer-, critic-, and leader-agents that maintain and refine an inspectable world model through commit-then-retrodict loops.
- By routing prediction failures to specific artifact owners and validating patches against a regression set, the system enables efficient, sample-constrained adaptation in unfamiliar interactive environments.

---

[PDEAgent-Bench: A Multi-Metric, Multi-Library Benchmark for PDE Solver Generation](http://arxiv.org/abs/2605.09636)

- PDEAgent-Bench: introduces a multi-metric, multi-library benchmark for evaluating LLMs and code agents in synthesizing executable numerical solvers from PDE specifications.
- The framework utilizes a staged evaluation pipeline that sequentially verifies code executability, numerical accuracy against reference solutions, and computational efficiency within isolated sandboxes.
- The benchmark includes 645 instances across 11 PDE families and 3 FEM libraries (DOLFINx, Firedrake, deal.II) to assess model performance in scientific computing tasks.

---

[Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases: A Matched-Ceiling Study of Open-Weight LLM Reasoning Protocols](http://arxiv.org/abs/2605.09618)

- Statistical Scouting Framework: evaluates reasoning protocols under a matched token ceiling to determine if cheap pre-deliberation signals can recover latent routing headroom from oracle-selected protocols.
- The study identifies that vote entropy effectively predicts debate safety by reducing backfire, but fails to identify debate-useful cases where voting is unanimous yet incorrect.
- Empirical results demonstrate that learned controllers underperform simple entropy thresholds, and a naive self-critique behavioral probe provides zero discriminative signal due to sycophancy or format-compliance artifacts.

---

[Trust Me, Import This: Dependency Steering Attacks via Malicious Agent Skills](http://arxiv.org/abs/2605.09594)

- Dependency Steering: introduces a semantic-preserving optimization framework that manipulates LLM coding agents into recommending attacker-controlled packages by injecting malicious persistent instruction artifacts known as Skills.
- The framework utilizes a Multi-Objective Dependency Steering Scorer, an Explicit Veto Mechanism, a Context-Patch Injection Engine, a Dependency Steering Strategy Library, and Lifelong Semantic Strategy Exploration to generate stealthy, effective adversarial patches.
- Evaluations across multiple LLMs demonstrate that the approach achieves high targeted hallucination rates, exhibits strong cross-model and cross-domain transferability, and effectively evades most existing static and LLM-based security auditing tools.

---

[ConCovUp: Effective Agent-Based Test Driver Generation for Concurrency Testing](http://arxiv.org/abs/2605.09573)

- ConCovUp: introduces a multi-agent framework that integrates LLMs with program analysis to automate the generation of concurrent test drivers for C/C++ libraries.
- The framework utilizes an Analysis Agent to identify shared memory targets, a Path Agent to perform heuristic-guided backward tracing for constraint satisfaction, and a Test Generation Agent to synthesize executable concurrent harnesses.
- ConCovUp operates within an iterative feedback loop, using dynamic coverage reports to refine test drivers and improve the coverage of hard-to-reach concurrent behaviors.

---

[TacoMAS: Test-Time Co-Evolution of Topology and Capability in LLM-based Multi-Agent Systems](http://arxiv.org/abs/2605.09539)

- TacoMAS: introduces a test-time co-evolution framework for LLMs that jointly optimizes agent capabilities and communication topology through two coupled loops.
- The framework utilizes a fast capability loop for rapid expertise refinement via Meta-LLM and Meta-Judge feedback, and a slow topology loop for structural graph adaptation.
- TacoMAS models the co-evolution as a two-timescale replicator-mutator process, ensuring stable convergence toward task-conditioned equilibria in multi-agent systems.

---

[Governing AI-Assisted Security Operations: A Design Science Framework for Operational Decision Support](http://arxiv.org/abs/2605.09534)

- Governed AI query-broker architecture: introduces a socio-technical framework that separates AI planning from operational execution to ensure accountability in high-risk security environments.
- The framework utilizes RAG grounding, policy-enforced KQL brokers, and source-specific adapters to manage AI-assisted threat hunting while maintaining auditability and cost control.
- This study provides design propositions and a maturity model to guide engineering managers in transitioning AI-assisted operations from informal experimentation to governed production.

---

[MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents](http://arxiv.org/abs/2605.09530)

- MemPrivacy: introduces a privacy-preserving framework for edge-cloud agents that uses local reversible pseudonymization to protect sensitive data while maintaining memory utility.
- The framework employs a Local MemPrivacy Model to identify privacy spans and replace them with typed placeholders, which are then restored locally after cloud-side processing.
- MemPrivacy includes a four-level privacy taxonomy and the MemPrivacy-Bench dataset to systematically evaluate the trade-off between privacy protection and personalized memory utility.

---

[Emergent Communication for Co-constructed Emotion Between Embodied Agents via Collective Predictive Coding](http://arxiv.org/abs/2605.09522)

- Inter-GMM+MVAE: introduces a computational framework for the co-construction of emotion by integrating MVAE, GMM, and MHNG to align symbolic representations between embodied agents.
- The framework utilizes MVAE to fuse multimodal sensory inputs into a shared latent space, while MHNG enables agents to align emotion categories through symbolic communication without explicit feedback.
- Experiments demonstrate that this approach allows agents with divergent interoceptive dynamics to achieve robust categorical alignment, supporting the theory that interoceptive heterogeneity is constitutive of emotional meaning.

---

[A Game-Theoretic Free Energy Analysis of Higher-Order Synergy in Attention Heads of Large Language Models](http://arxiv.org/abs/2605.09515)

- GT-FEP: introduces a game-theoretic framework that models LLM attention heads as bounded-rational agents minimizing variational free energy to quantify higher-order synergistic interactions.
- The framework utilizes Harsanyi dividends to decompose coalition energy, revealing that triple interactions in LLMs consistently exhibit higher-order redundancy.
- By applying the Nash-FEP correspondence, the authors derive a principled pruning criterion based on Shapley values that effectively compresses LLMs while maintaining performance.

---

[Position: AI Security Policy Should Target Systems, Not Models](http://arxiv.org/abs/2605.09504)

- Swarm-attack: introduces an adversarial testing framework where multiple LFM2.5-1.2B-Thinking agents coordinate through shared memory, parallel exploration, and evolutionary optimization to bypass frontier model safety and discover software vulnerabilities.
- The framework utilizes a pipeline of LFM2.5-1.2B-Thinking agents, regex pattern detection, hand-crafted exploit seeds, and AddressSanitizer-based crash classification to compensate for the limited reasoning capacity of individual small LLMs.
- Empirical results demonstrate that system-level scaffolds enable small open-weights models to achieve offensive capabilities comparable to frontier models, suggesting that security policy should prioritize system-level assessment over model-level access restrictions.

---

[Don’t Click That: Teaching Web Agents to Resist Deceptive Interfaces](http://arxiv.org/abs/2605.09497)

- DUDE (Deceptive UI Detector &amp; Evaluator): introduces a two-stage framework that enhances VLM agent robustness against deceptive web interfaces by combining Hybrid-Reward Learning for calibrated evaluation and Experience Summarization for transferable knowledge accumulation.
- The framework utilizes an interposed evaluator that assesses candidate interactions, balancing task completion against deception avoidance through asymmetric penalties and iterative failure distillation.
- The authors introduce the RUC (Real UI Clickboxes) benchmark, comprising 1,407 scenarios to evaluate agent performance and susceptibility to dark patterns across four distinct domains.

---

[LASSA Architecture-Based Autonomous Fault-Tolerant Control of Unmanned Underwater Vehicles](http://arxiv.org/abs/2605.09494)

- LASSA (LLM-based Agent with Solver, Sensor and Actuator): introduces a hierarchical, dual-loop control architecture for UUVs that integrates L-LLM, A-Agent, and S-Solver to enable autonomous fault-tolerant navigation in communication-constrained environments.
- The architecture utilizes a fast-slow control loop design where the slow loop performs high-level cognitive reasoning and strategy optimization, while the fast loop ensures real-time, high-precision low-level actuation.
- By grounding LLM-generated strategies through a deterministic S-Solver, the framework suppresses physically infeasible hallucinations and provides interpretable, auditable decision-making for complex underwater fault scenarios.

---

[Kintsugi: Learning Policies by Repairing Executable Knowledge Bases](http://arxiv.org/abs/2605.09487)

- Kintsugi: introduces a white-box policy-learning framework that treats embodied policy improvement as the verifier-gated construction of a typed executable Knowledge Base (KB).
- The framework utilizes a restricted LLM editor to propose localized typed edits to the Knowledge Base, which are then validated by a deterministic applier and verifier gate before deployment.
- At inference, the accepted Knowledge Base is executed by a deterministic symbolic executor with zero LLM calls, ensuring inspectability, local editability, and verifier-gated deployment.

---

[PolicyCache-SDN: Hierarchical Intra-Path Learning for Adaptive SDN Traffic Control](http://arxiv.org/abs/2605.09473)

- PolicyCache-SDN: introduces a hierarchical SDN traffic control framework that enables local online adaptation by confining learning to path-specific aggregates within controller-defined policy envelopes.
- The architecture utilizes an SDN Controller to manage global intent and compute policy envelopes, while Edge Agents perform fast local learning using HAT models to execute metering, queueing, and rerouting actions.
- By combining centralized policy constraints with decentralized intra-path learning, the framework ensures safe, auditable, and efficient traffic engineering that outperforms static and offline-learned baselines.

---

[Through the Lens of Character: Resolving Modality-Role Interference in Multimodal Role-Playing Agent](http://arxiv.org/abs/2605.09443)

- CAVI (Character-Aware Visual Intervention): introduces a training-free framework that resolves Modality-Role Interference in MLLMs by aligning visual perception with subjective character personas through Character Anchor, Character-Guided Token Pruning, Orthogonal Feature Modulation, Modality-Adaptive Role Steering, and Dynamic Contextual Re-Injection.
- The framework functions as an information-theoretic bottleneck that systematically purifies role-consistent visual facts from objective noise to enhance character-consistent multimodal interactions.
- Extensive experiments demonstrate that CAVI significantly improves role-playing performance and generalization to unseen characters without requiring parameter updates or incurring prohibitive inference overhead.

---

[Beyond Isolation: A Unified Benchmark for General-Purpose Navigation](http://arxiv.org/abs/2605.09441)

- OmniNavBench: introduces a unified benchmark designed to rigorously evaluate cross-skill coordination and cross-embodiment generalization in embodied navigation by composing sequences of primary sub-tasks.
- The framework utilizes a high-fidelity simulation platform and human-expert teleoperated trajectories to assess agent performance across diverse robot morphologies and complex, multi-stage navigation instructions.
- Extensive evaluations reveal that current unified models struggle with sequential task coordination, termination decisions, and dynamic social interaction, highlighting the need for more robust generalist navigation agents.

---

[SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning](http://arxiv.org/abs/2605.09423)

- SimWorld Studio: introduces an open-source platform for generating evolving embodied learning environments using a tool-augmented coding agent that constructs physically grounded 3D worlds.
- The framework utilizes SimCoder, which leverages a Tool Library, Skill Library, and MCP Server to iteratively build and refine environments based on feedback from Scene Verification.
- SimWorld Studio enables a co-evolution loop where the Embodied Agent's performance feedback guides the generation of adaptive curricula, significantly improving agent generalization and training efficiency.

---

[MACAA: Belief-Revision Multi-Agent Reasoning for Open-World Code Authorship Verification](http://arxiv.org/abs/2605.09421)

- MACAA: introduces a belief-revision-based multi-agent framework for training-free code authorship verification that utilizes a Coordinator Agent and four Expert Agents to analyze layout, lexical, syntactic, and programming-pattern evidence.
- The framework operationalizes AGM belief revision theory through expansion, contraction, and revision operations to maintain a consistent authorship hypothesis while resolving conflicts in heterogeneous code pairs.
- MACAA provides explicit evidence tracing and forensic auditability by recording the belief revision path, enabling inspection of the feature-level basis for each authorship decision.

---

[Empowering VLMs for Few-Shot Multimodal Time Series Classification via Tailored Agentic Reasoning](http://arxiv.org/abs/2605.09395)

- MarsTSC (Multimodal Time Series Classification): introduces a VLM-based agentic reasoning framework that utilizes a dynamic, self-evolving knowledge bank to improve few-shot time series classification performance.
- The framework employs a collaborative trio of agents—a Generator, a Reflector, and a Modifier—to iteratively refine task-specific knowledge through trial-and-error without updating model parameters.
- A test-time update mechanism with two-pass generation and deferred knowledge validation mitigates few-shot bias and distribution shift, ensuring robust and interpretable classification decisions.

---

[NEXUS: Continual Learning of Symbolic Constraints for Safe and Robust Embodied Planning](http://arxiv.org/abs/2605.09387)

- NEXUS: introduces a modular neuro-symbolic framework that integrates LLMs with formal methods to enforce safe and feasible embodied planning through continual learning.
- The framework includes planning-, perception- and safety-agents that decouple physical feasibility from safety specifications by grounding probabilistic risk assessments into deterministic hard LTL constraints.
- NEXUS achieves superior task success rates and robust defense against adversarial attacks by progressively improving planning efficiency through knowledge accumulation in PDDL domains and LTL safety specifications.

---

[Towards a Virtual Neuroscientist: Autonomous Neuroimaging Analysis via Multi-Agent Collaboration](http://arxiv.org/abs/2605.09366)

- NIAgent: introduces a multi-agent system for autonomous end-to-end neuroimaging analysis that utilizes Supervisor Agent, Data Awareness Agent, Quality Control Agent, Processing Agent, Downstream Analysis Agent, Just-in-Time Context Injection, Code-Centric Execution, Domain-Specific Primitive Libraries, and Hierarchical QC Module.
- The framework employs a centralized-planning and decentralized-execution design where specialist agents synthesize executable programs over composable domain-specific primitives to handle long-horizon neuroimaging workflows.
- The system incorporates a closed-loop quality control mechanism that integrates cohort-level metric screening with agentic visual inspection to drive evidence-grounded workflow remediation.

---

[Multi-scale Predictive Representations for Goal-conditioned Reinforcement Learning](http://arxiv.org/abs/2605.09364)

- Ms.PR (Multi-scale Predictive Representations): introduces a framework that enforces dynamical, behavioral, and temporal alignment through multi-scale predictive supervision to construct robust latent representations for offline goal-conditioned reinforcement learning.
- The framework utilizes a shared encoder architecture with specialized predictive modules to decouple representation learning from value approximation, effectively mitigating value overestimation in sparse-reward environments.
- Ms.PR demonstrates superior performance and robustness across diverse state-based and pixel-based tasks, particularly under conditions of trajectory stitching, limited data, and noisy expert demonstrations.

---

[Skill-R1: Agent Skill Evolution via Reinforcement Learning](http://arxiv.org/abs/2605.09359)

- Skill-R1: introduces a reinforcement learning framework for recurrent agent skill evolution that decouples task execution from skill improvement by training a lightweight skill generator while keeping the task LLM frozen.
- The framework utilizes a bi-level group-relative policy optimization objective that combines intra-generation rollout quality comparisons with inter-generation progress rewards to guide directional skill refinement.
- Empirical results demonstrate that Skill-R1 consistently outperforms no-skill baselines and standard GRPO across complex multi-step reasoning and tool-use benchmarks.

---

[PECMAN: Perception-enabled Collaborative Multi-Agent Navigation in Unknown Environments](http://arxiv.org/abs/2605.09344)

- PECMAN: introduces a multi-agent navigation framework that utilizes LiDAR Scan, Shared Perception, Distributed Tree Repair, Narrow Corridor Coordination, and a Global King Priority Layer to enable efficient navigation in unknown, dynamic environments.
- The framework extends the SMART-3D single-agent planner by implementing distributed tree morphing and a coordination layer that resolves deadlocks in narrow corridors through priority-based agent sequencing.
- Experimental results demonstrate that the shared perception strategy significantly reduces team-completion time and improves situational awareness compared to independent navigation modes.

---

[A Cross-Layered Multi-Drone Coordination for Medical Supply Delivery during Disaster Response Management](http://arxiv.org/abs/2605.09342)

- CEDA (Centralized Training with Decentralized Execution Deep Q-Network): introduces a unified reinforcement learning framework that jointly optimizes drone routing, multi-agent coordination, and triage-priority-aware scheduling across physical, network, and application layers.
- The framework utilizes a centralized Q-network during training to learn a policy that enables decentralized execution by individual drones, ensuring robustness to communication disruptions in disaster response environments.
- CEDA incorporates a Priority-Preserving Fair Scheduling strategy that uses dynamic per-patient survival models and multiple reward components to ensure equitable service across triage classes without requiring explicit fairness constraints.

---

[SkillMAS: Skill Co-Evolution with LLM-based Multi-Agent System](http://arxiv.org/abs/2605.09341)

- SkillMAS: introduces a non-parametric framework that couples skill evolution and MAS restructuring by constraining both processes under a shared verified-trace evidence surface.
- The framework utilizes Utility Learning to assign credit to execution-supported skills and participating executors, while employing evidence-gated MAS restructuring to modify organization only when structural bottlenecks are identified.
- SkillMAS improves post-deployment specialization across embodied manipulation, command-line OS workflows, and retail-service interaction without requiring retraining of the underlying LLMs.

---

[The Trap of Trajectory: Towards Understanding and Mitigating Spurious Correlations in Agentic Memory](http://arxiv.org/abs/2605.09330)

- CAMEL (CAusality-informed MEmory caLibration): introduces a plug-and-play calibration framework that mitigates spurious correlations in agentic memory by applying Write-stage calibration and Retrieve-stage calibration to existing memory architectures.
- The framework utilizes Residualization and a Content-novelty write criterion to neutralize confounding and collider bias at the point of memory storage, while employing a Causal stability test to filter retrieved candidates based on their sensitivity to non-causal perturbations.
- By maintaining Streaming covariance statistics and a Non-causal subspace, CAMEL effectively reduces spurious reasoning ratios across diverse LLM-based agentic systems without requiring additional LLM forward passes or fine-tuning.

---

[OpenIIR: An Open Simulation Platform for Information Retrieval Research](http://arxiv.org/abs/2605.09321)

- OpenIIR: introduces a modular simulation platform for IR research that utilizes a shared core of Agent Runtime, World-Model Store, Retrieval Primitives, Claim Extractor, and Persona Ontology to support diverse multi-agent experiments.
- The platform enables researchers to configure LLM-driven personas across various study types, including deliberative panels, social platforms, curated feeds, and evolutionary co-evolution scenarios, using a unified type interface.
- OpenIIR provides reproducible, end-to-end simulation outputs such as argument graphs and exposure logs, allowing for complex IR research questions to be investigated on a single laptop without requiring extensive infrastructure.

---

[Mem-W: Latent Memory-Native GUI Agents](http://arxiv.org/abs/2605.09317)

- Mem-W: introduces a latent-context-native interface for GUI agents that unifies working memory and experiential memory into a single continuous embedding space using a shared trajectory-to-latent compressor.
- The framework replaces manual memory hierarchies with a learnable compressor that projects both retrieved historical trajectories and in-session episode prefixes into compact, decision-relevant latent tokens.
- Mem-W is trained via self-distillation and outcome-aware supervision to ensure that the latent memory effectively supports long-horizon task success without requiring updates to the frozen GUI agent policy.

---

[Do Self-Evolving Agents Forget? Capability Degradation and Preservation in Lifelong LLM Agent Adaptation](http://arxiv.org/abs/2605.09315)

- CPE (Capability-Preserving Evolution): introduces a stabilization principle that constrains destructive capability drift in self-evolving LLM agents by regularizing updates to the agent's mutable repository.
- The framework addresses capability erosion across four evolution dimensions—workflow, skill, model, and memory—by biasing self-evolution toward low-interference solutions that preserve previously mastered competencies.
- Empirical results demonstrate that CPE consistently improves retained capability stability and performance across diverse LLM backbones compared to unconstrained self-evolution.

---

[LEAF-SQL: Level-wise Exploration with Adaptive Fine-graining for Text-to-SQL Skeleton Prediction](http://arxiv.org/abs/2605.09295)

- LEAF-SQL: introduces a novel Text-to-SQL framework that reframes skeleton prediction as a coarse-to-fine tree search process, utilizing Level-wise Skeleton Search, Skeleton Formulation Agent, Skeleton Evaluation Agent, SQL Generation Module, and Result Selection Module.
- The framework employs a three-level skeleton hierarchy (Base, Expanded, Detailed) to guide the search process, enabling adaptive granularity and structural diversity in skeleton candidates.
- By integrating a pruning mechanism via the Skeleton Evaluation Agent, LEAF-SQL effectively navigates the search space to identify optimal query structures while maintaining computational efficiency.

---

[PiCA: Pivot-Based Credit Assignment for Search Agentic Reinforcement Learning](http://arxiv.org/abs/2605.09287)

- PiCA (Pivot-Based Credit Assignment): introduces a novel reinforcement learning framework for LLM-based search agents that reformulates search trajectories as a sequential process of cumulative progress to mitigate reward sparsity and isolated credit assignment.
- The framework utilizes a Policy LLM, a Search Engine, and a Reward Model to identify pivot steps as information peaks, providing dense, trajectory-dependent guidance through Potential-Based Reward Shaping.
- By integrating step-level explicit supervision and outcome-level implicit supervision, PiCA ensures robust generalization and improved performance across diverse knowledge-intensive QA benchmarks.

---

[A Prompt-Aware Structuring Framework for Reliable Reuse of AI-Generated Content in the Agentic Web](http://arxiv.org/abs/2605.09283)

- A Prompt-Aware Structuring Framework for Reliable Reuse of AI-Generated Content in the Agentic Web: introduces a framework that automatically attaches structured metadata and verifiable credentials to AIGC to ensure provenance and reliability for downstream reuse.
- The framework utilizes POML to modularize prompts, enabling a curation agent to mechanically evaluate instruction-following fidelity before using the content for LLM fine-tuning.
- Experimental results on ComplexBench demonstrate that metadata-driven curation of AIGC consistently improves the requirements following ratio compared to random data selection.

---

[EQUIMEM: Calibrating Shared Memory in Multi-Agent Debate via Game-Theoretic Equilibrium](http://arxiv.org/abs/2605.09278)

- EQUIMEM: introduces a zero-trust game-theoretic framework for multi-agent debate that calibrates shared memory updates using structural indicators instead of LLM-based judgments.
- The framework utilizes a contributor-auditor game structure where memory integrity is enforced through a calibration mechanism that combines detection signals, alignment signals, and credibility weights.
- By replacing vote-based commitment with algorithmic structural verification, the system achieves robust memory protection against adversarial agents without incurring additional LLM inference costs.

---

[Towards Conversational Medical AI with Eyes, Ears and a Voice](http://arxiv.org/abs/2605.09272)

- AI co-clinician: introduces a dual-agent architecture comprising a Talker (handles real-time patient interaction) and a Clinical Planner (supervisory reasoning module) to balance conversational fluency with deep clinical reasoning.
- The system utilizes continuous Audio-Video Input Stream (continuous patient data) to perform real-time diagnostic and management tasks, supported by Audio, Text Output (agent response generation) for natural dialogue.
- This research evaluates the framework against human physicians and other LLMs using a novel TelePACES rubric, demonstrating significant improvements in clinical reasoning and physical examination accuracy through the Clinical Planner.

---

[Agentic AI for Particle-Based Simulation: Automating SPH Workflows for Debris Flow Modeling](http://arxiv.org/abs/2605.09265)

- Agentic AI for Particle-Based Simulation: introduces a human-in-the-loop agentic workflow for meshless particle-based simulation using DualSPHysics, integrating Planning AI-agent, DualSPHysics solver, Post-processing AI-agent, Human-in-the-loop interface, and Multimodal input module.
- The framework utilizes a stage-structured design to separate reasoning-intensive tasks from deterministic simulation execution, enhancing robustness in complex debris flow modeling.
- A cognitive-task-oriented evaluation framework is employed to assess agent performance across scalar extraction, visualization, phase identification, physical derivation, and geometric disambiguation.

---

[LLM Agents Already Know When to Call Tools - Even Without Reasoning](http://arxiv.org/abs/2605.09252)

- PROBE&PREFILL: introduces a lightweight inference-time intervention that leverages hidden-state signals to steer LLM tool-call decisions without requiring model fine-tuning.
- The framework utilizes the WHEN2TOOL benchmark to demonstrate that LLMs internally encode tool-necessity information, which can be extracted via a linear probe to guide generation.
- By prepending a steering sentence based on probe predictions, the method achieves a superior accuracy-efficiency tradeoff compared to prompt engineering or explicit reasoning baselines.

---

#### 9th May 


[OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](http://arxiv.org/abs/2605.08876)

- OTora (Unified Red Teaming Framework for Reasoning-Level Denial-of-Service): introduces a two-stage red-teaming pipeline that induces Reasoning-Level Denial-of-Service (R-DoS) in LLM agents by triggering external tool access and injecting reasoning-intensive payloads to exhaust reasoning budgets.
- The framework utilizes a Stage I Trigger Optimizer to hijack tool calls and a Stage II Payload Optimizer to embed persistent, task-consistent reasoning detours that inflate end-to-end latency while maintaining functional correctness.
- Experimental results demonstrate that OTora achieves order-of-magnitude increases in reasoning tokens and latency across diverse LLM agents and tool interfaces without triggering standard safety or early-stop mechanisms.

---



[AssemPlanner: A Multi-Agent Based Task Planning Framework for Flexible Assembly System](http://arxiv.org/abs/2605.08831)

- AssemPlanner: introduces a hierarchical multi-agent framework that orchestrates SchedAgent, KnowledgeAgent, LineBalanceAgent, and a Scene Graph to transform natural-language instructions into executable industrial assembly sequences.
- The framework utilizes a closed-loop "Reason-Action-Observe" cycle where the SchedAgent dynamically negotiates with specialized agents to ensure constraint compliance and logical consistency.
- By integrating KG-enhanced RAG and prompt-driven reflective optimization, the system eliminates the need for expert-level parameter tuning and manual reward engineering in flexible assembly environments.

---



[Internal vs. External: Comparing Deliberation and Evolution for Multi-Agent Constitutional Design](http://arxiv.org/abs/2605.09128)

- Internal vs. External Constitutional Design Framework: compares internal deliberation and external evolution for governing LLM-based multi-agent systems across three social environments.
- The study reveals that external evolution excels in stable collective-action settings by discovering behavioral directives, while internal deliberation provides structural responsiveness by adapting governance policies to shifting incentives.
- The research identifies a structural punishment gap where deliberating agents avoid costly peer-to-peer sanctions, unlike evolved constitutions that reliably converge on cooperation-sustaining enforcement mechanisms.

---


[Trustworthy AI: Ensuring Reliability and Accountability from Models to Agents](http://arxiv.org/abs/2605.08964)

- Trustworthy AI: introduces a comprehensive framework for ensuring reliability and accountability in ML systems, ranging from predictive models to generative LLMs and autonomous agents.
- The research develops kernel-based methods for multiaccuracy, information-theoretic watermarking schemes for LLMs, and a multi-agent simulator for evaluating autonomous supply chain management.
- The thesis addresses systemic risks in AI by formalizing arbitrariness in predictive models, establishing provable detection-distortion tradeoffs for LLM watermarking, and identifying emergent dynamics in multi-agent systems.

---

[Toward Web 4.0: Bidirectional Trust between AI Agents and Blockchain](http://arxiv.org/abs/2605.08922)

- Bidirectional Trust Framework: introduces a systematization of knowledge for the convergence of autonomous AI agents and blockchain, formalizing the interaction through an Agent-Blockchain Interaction Model (ABIM) and a five-dimensional evaluation framework.
- The framework categorizes the symbiotic relationship into two directions: B→A (Blockchain providing trust infrastructure for agents) and A→B (AI agents participating in core blockchain mechanisms), both underpinned by a Trust Foundation of verifiable computation.
- The analysis identifies significant ecosystem immaturity, where most agent-specific standards remain in draft status, and highlights critical research gaps including the LLM verifiability bottleneck and the need for formal security models for agent-blockchain interactions.

---

[Generalization Bounds of Emergent Communications for Agentic AI Networking](http://arxiv.org/abs/2605.08613)

- DIB-based Emergent Communication Framework: introduces a multi-agent communication architecture that leverages Distributed Information Bottleneck (DIB) theory to unify decision-making functions and emergent communication protocols through a joint loss function.
- The framework optimizes the trade-off between task-relevant information and computational complexity, represented by the Minimum Description Length (MDL), to ensure robust performance in resource-constrained AgentNet environments.
- Theoretical generalization bounds are derived to guarantee performance stability when agents transition from training datasets to unseen decentralized inference scenarios.

---

[ProactBench: Beyond What The User Asked For](http://arxiv.org/abs/2605.09228)

- ProactBench: introduces a three-agent evaluation architecture to measure conversational proactivity by testing if an LLM can identify and act on unstated user needs.
- The framework decomposes proactivity into three phase-tied trigger types: EMERGENT (single-anchor inference), CRITICAL (multi-anchor synthesis), and RECOVERY (grounded forward-looking value after task completion).
- By enforcing cold-start isolation and using varied communication styles, the benchmark evaluates models on their ability to provide useful, unprompted assistance without collapsing into reactive instruction-following.

---

[Flame3D: Zero-shot Compositional Reasoning of 3D Scenes with Agentic Language Models](http://arxiv.org/abs/2605.09218)

- Flame3D: introduces a training-free framework that represents 3D scenes as editable visual-textual memories, enabling compositional reasoning through a set of spatial and meta-tools.
- The framework utilizes a tool-calling VLM to interact with a structured scene memory, allowing the agent to synthesize custom spatial programs at inference time for complex, multi-hop queries.
- Flame3D achieves competitive performance on 3D benchmarks without 3D-specific fine-tuning by leveraging explicit scene representations and expressive compositional abstractions.

---

[Learning the Preferences of a Learning Agent](http://arxiv.org/abs/2605.09217)

- Learning the Preferences of a Learning Agent: introduces a formal framework for a predictor to infer the underlying reward function of a learner agent that is itself learning to act optimally over time.
- The framework models the learner as either a no-regret agent or a Boltzmann-rational agent, establishing theoretical guarantees for reward inference under different environment settings.
- The research demonstrates that while inferring the full preference structure of a no-regret learner is generally impossible, Boltzmann-rationality allows for meaningful bounds on prediction error.

---

[Rethinking Ratio-Based Trust Regions for Policy Optimization in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.09212)

- MARS (Multi-Agent Ratio Symmetry): introduces a policy optimization objective for cooperative MARL that replaces additive trust-region mechanisms with a multiplicatively symmetric geometric barrier to preserve corrective gradients and prevent probability extinction.
- The framework utilizes a centralized critic to compute joint advantage estimates, which are then used to calibrate a geometric penalty that remains unbounded as probability ratios approach zero.
- By ensuring that probability extinction is never optimal, MARS maintains stable training dynamics in high-variance continuous control and discrete coordination tasks compared to standard MAPPO and MASPO methods.

---

[Evidence Over Plans: Online Trajectory Verification for Skill Distillation](http://arxiv.org/abs/2605.09192)

- SPARK (Structured Pipelines for Autonomous Runnable tasKs and sKill generation): introduces a framework for distilling environment-grounded skills from agent trajectories by utilizing the Posterior Distillation Index to ensure procedural knowledge is based on empirical evidence rather than prior plans.
- The framework includes a Teacher Agent for exploration, a Student Agent for skill consumption, and a PDI-based online intervention mechanism to improve skill generation quality during the exploration process.
- SPARK demonstrates that posterior-based skill distillation significantly improves task success rates and transferability across different LLM models while reducing inference costs.

---

[Agentic MIP Research: Accelerated Constraint Handler Generation](http://arxiv.org/abs/2605.09186)

- Agentic MIPR: introduces a solver-aware harness that embeds LLM agents to automate the generation, verification, and tuning of SCIP constraint handlers for MIP research.
- The framework utilizes a pipeline of Prompt Specification, Plugin Generation, In-Context Learning, and Large-Scale Evaluation to bridge the gap between algorithmic hypotheses and executable solver plugins.
- By leveraging a sandbox environment and structured stage contracts, the framework enables the autonomous discovery of novel constraint patterns and propagation strategies that improve solver performance on complex MIPLIB 2017 instances.

---

[AutoRedTrader: Autonomous Red Teaming of Trading Agents through Synthetic Misinformation Injection](http://arxiv.org/abs/2605.09185)

- AutoRedTrader: introduces an autonomous red-teaming framework that generates finance-specific misinformation through behavioral bias manipulation, minor textual perturbations, and style-controlled rewriting to stress-test LLM-based financial agents.
- The framework utilizes a closed-loop feedback mechanism where the financial agent's cumulative return and historical decision-impact records guide the MisGen module to iteratively strengthen misinformation generation.
- AutoRedTrader incorporates an optional time-series-informed grounding setting that provides structured historical market evidence to help LLM-based agents assess the consistency of textual signals and improve robustness against misinformation.

---

[CIVeX: Causal Intervention Verification for Language Agents](http://arxiv.org/abs/2605.09168)

- CIVeX: introduces a causal intervention verifier that maps proposed actions to structural causal queries over a committed action-state graph to ensure intervention identifiability before execution.
- The framework utilizes a Causal Certificate containing an identification argument, a one-sided Lower Confidence Bound, and risk limits to issue one of four auditable verdicts: EXECUTE, REJECT, EXPERIMENT, or ABSTAIN.
- CIVeX provides a robust safety mechanism for LLMs by treating tool execution as causal inference, effectively mitigating false executions in confounded environments where observational correlations fail.

---

[FORTIS: Benchmarking Over-Privilege in Agent Skills](http://arxiv.org/abs/2605.09163)

- FORTIS: introduces a two-stage benchmark designed to evaluate over-privilege in LLM agents by measuring their ability to select minimally sufficient skills and execute them within documented boundaries.
- The framework organizes skills and tools into a five-level privilege hierarchy to assess whether models exercise restraint or default to broader, more permissive capabilities when faced with semantic ambiguity.
- Empirical results across ten frontier models demonstrate that over-privileged behavior is a structural tendency, with failure rates often exceeding 75% under realistic conditions like convenience-sensitive framing.

---

[Beyond Self-Play: Hierarchical Reasoning for Continuous Motion in Closed-Loop Traffic Simulation](http://arxiv.org/abs/2605.09153)

- Hierarchical Framework: introduces a modular architecture that decouples high-level interaction reasoning from low-level continuous trajectory realization to improve behavioral realism in closed-loop traffic simulation.
- The system utilizes a Stackelberg-style MARL module for strategic decision-making and a command-conditioned Wayformer for physically consistent motion generation.
- A hybrid co-training scheme incorporating heuristic recovery trajectories is employed to mitigate distribution shift and stabilize closed-loop policy execution.

---

[AlphaExploitem: Going Beyond the Nash Equilibrium in Poker by Learning to Exploit Suboptimal Play](http://arxiv.org/abs/2605.09150)

- AlphaExploitem: introduces a hierarchical transformer-based architecture that enables poker agents to adapt to opponent-specific behaviors by processing multi-hand session histories.
- The framework utilizes a hierarchical history transformer consisting of a within-hand encoder and an across-hand encoder to generate a session-level context vector, which is fused with current-hand card and action encoders to inform decision-making.
- Training is performed via PPO using a combination of K-best league self-play, a long-tail buffer of past agent checkpoints, and a diverse set of hand-crafted exploitable opponent policies to ensure robust adaptation.

---

[Beyond Thinking: Imagining in 360° for Humanoid Visual Search](http://arxiv.org/abs/2605.09146)

- Imagining in 360°: introduces a decoupled architecture for Humanoid Visual Search that separates intuitive spatial imagination from rigorous action planning to improve search efficiency.
- The Imaginator functions as a probabilistic predictor of spatial priors, generating multiple hypotheses about the semantic layout of observed and unobserved regions to guide the Actor.
- This framework utilizes a fully automated pseudo-labeling pipeline to scale training to millions of samples, enabling robust performance in complex, in-the-wild environments without manual trajectory annotations.

---

[MCP-Cosmos: World Model-Augmented Agents for Complex Task Execution in MCP Environments](http://arxiv.org/abs/2605.09131)

- MCP-Cosmos: introduces a framework that integrates generative World Models into the MCP ecosystem to enable predictive task automation through latent space simulation.
- The framework utilizes a BYOWM (Bring Your Own World Model) architecture to allow LLM agents to perform speculative look-ahead searches and refine trajectories before physical execution.
- It proposes the Execution Quality metric to quantify the efficiency of tool usage by penalizing excessive tool calls and rewarding successful, proactive planning.

---

[A Communication-Theoretic Framework for LLM Agents: Cost-Aware Adaptive Reliability](http://arxiv.org/abs/2605.09121)

- AGENTCODEC: introduces a unified communication-theoretic framework for LLM agents that treats model invocations as discrete stochastic channels to enable principled reliability optimization.
- The framework maps common reliability techniques to classical communication operators, including diversity combining, hybrid retransmission, iterative generator-critic decoding, rateless sampling, structured redundant verification, and difficulty-adaptive routing.
- A cost-aware semantic-nearest-neighbor router dynamically selects the optimal reliability technique per task, significantly improving the quality-cost Pareto frontier compared to fixed-technique baselines.

---

[Token Economics for LLM Agents: A Dual-View Study from Computing and Economics](http://arxiv.org/abs/2605.09104)

- Token Economics for LLM Agents: introduces a unified framework conceptualizing tokens as production factors, exchange mediums, and units of account to evaluate the trade-off between output quality and economic cost in agentic systems.
- The framework categorizes token-based economic dynamics across three architectural scales: micro-level single-agent optimization, meso-level multi-agent system coordination, and macro-level agent ecosystem governance.
- The research synthesizes computational system design with neoclassical economic theory to address systemic bottlenecks, including transaction costs, congestion externalities, and security-related token attrition.

---

[GRC: Unifying Reasoning-Driven Generation, Retrieval and Compression](http://arxiv.org/abs/2605.09100)

- GRC: introduces a unified training framework that enables LLMs to perform reasoning-driven generation, text embedding, and context compression in a single forward pass using meta latent tokens.
- The framework utilizes meta latent tokens as internal registers to compress context into O(1) length KV cache, facilitating efficient latent memory-augmented generation and reasoning-enhanced text representation.
- Hybrid paged attention is integrated to manage regular and compressed KV caches, significantly improving inference throughput for the unified model.

---

[Robust Multi-Agent LLMs under Byzantine Faults](http://arxiv.org/abs/2605.09076)

- SAC (Self-Anchored Consensus): introduces a decentralized protocol for LLM multi-agent systems that ensures robustness against Byzantine faults by replacing sender-reported confidence with receiver-side evaluation and iterative filtering.
- The framework leverages (F+1)-robust graph-theoretic conditions to guarantee that honest agents can maintain reliable consensus and filter out malicious or faulty information.
- Experimental results demonstrate that SAC effectively preserves the performance of strong agents while improving the accuracy of weaker agents across diverse communication topologies, unlike prior confidence-weighted aggregation methods.

---

[Octopus Protocol: One-Shot Hardware Discovery and Control for AI Agents via Infrastructure-as-Prompts](http://arxiv.org/abs/2605.09055)

- Octopus Protocol: introduces a system that automates hardware driver generation and control for AI agents by utilizing a five-stage build pipeline (PROBE, IDENTIFY, INTERFACE, SERVE, DEPLOY) and a self-healing daemon (WATCH, HEAL, PERCEIVE) to eliminate manual driver engineering.
- The framework treats hardware protocols as prompts rather than static code, enabling an LLM-driven coding agent to compile declarative specifications into platform-specific MCP servers at runtime.
- By leveraging a persistent daemon loop, the system provides autonomous self-healing capabilities, allowing the agent to monitor logs, reinstall dependencies, and regenerate code to maintain hardware connectivity without human intervention.

---

[LCGNav: Local Candidate-Aware Geometric Enhancement for General Topological Planning in Vision-Language Navigation](http://arxiv.org/abs/2605.09053)

- LCGNav: introduces a modular geometric enhancement framework that improves topological planning in VLN-CE by integrating 3D point cloud-based local features into ghost nodes.
- The framework utilizes 3D Point Cloud Transformation, Depth-directed Physical Hard Truncation, Farthest Point Sampling (FPS), PointNet Encoder, Dimension-Preserving Residual Fusion, and a State Degradation Mechanism to refine local spatial perception.
- LCGNav acts as a lightweight, transferable module that enhances existing topological planners by providing physically grounded local geometric cues while maintaining original architectural dimensions.

---

[Containment Verification: AI Safety Guarantees Independent of Alignment](http://arxiv.org/abs/2605.09045)

- Containment Verification: introduces a fail-safe paradigm for AI agents that deductively verifies the containment layer mediating between the AI model and external state to enforce boundary-enforceable safety properties.
- The framework utilizes havoc oracle semantics to model the AI as an unconstrained procedure, ensuring safety guarantees are invariant to model capability, training, or alignment.
- The approach employs an agentic formal synthesis pipeline to automate the generation of machine-checked Dafny proofs, establishing a forward-simulation refinement between abstract safety specifications and concrete operational state machines.

---

[ShadowMerge: A Novel Poisoning Attack on Graph-Based Agent Memory via Relation-Channel Conflicts](http://arxiv.org/abs/2605.09033)

- SHADOWMERGE: introduces a black-box poisoning attack against graph-based agent memory that exploits relation-channel conflicts to inject malicious evidence into shared memory.
- The framework utilizes an AIR pipeline consisting of Anchor, Inscribe, and Render to ensure poisoned relations are materialized, merged into target anchor neighborhoods, and retrieved for victim queries.
- SHADOWMERGE achieves high attack success rates by manipulating the graph-memory substrate without requiring privileged access to the memory graph or retriever.

---

[GAMBIT: A Three-Mode Benchmark for Adversarial Robustness in Multi-Agent LLM Collectives](http://arxiv.org/abs/2605.09027)

- GAMBIT: introduces a benchmark and dataset for evaluating imposter detectors in multi-agent LLM collectives using a chess-based substrate with a deterministic cost function.
- The framework includes an adaptive imposter agent that evolves its strategy across generations to evade detection, alongside a three-mode evaluation protocol measuring generalization and few-shot recalibration.
- Experimental results demonstrate that static adversarial benchmarks are insufficient, as meta-learned detectors significantly outperform standard fine-tuned models in fast recalibration under distribution shift.

---

[Evolutionary Ensemble of Agents](http://arxiv.org/abs/2605.09018)

- EvE (Evolutionary Ensemble): introduces a decentralized framework that organizes coding agents into a co-evolving system to optimize algorithmic discovery by maintaining two populations of functional code solvers and agent guidance states.
- The framework utilizes a synchronous race mechanism where multiple agents are sampled to refine solvers, with their relative performance tracked via Elo ratings to drive continuous, stage-dependent adaptation.
- By integrating solver improvement and self-referential agent optimization into a unified stage, EvE enables scalable, context-aware evolution that avoids phase mismatch and breaks through static performance ceilings.

---

[Learning to Explore: Scaling Agentic Reasoning via Exploration-Aware Policy Optimization](http://arxiv.org/abs/2605.08978)

- EAPO (Exploration-Aware Policy Optimization): introduces a reinforcement learning framework that enables LLM agents to adaptively explore environments by explicitly modeling exploration utility and memory.
- The framework utilizes a variational proxy to estimate the potential of exploratory actions and an exploration-aware grouping mechanism to distinguish information-seeking behavior from task-completion actions.
- By incorporating a rollback mechanism and structured memory, the agent effectively manages uncertainty and generalizes across complex, long-horizon tasks without requiring additional fine-tuning.

---

[Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery](http://arxiv.org/abs/2605.08956)

- Agentic AI Scientist Framework: introduces a critical analysis of current LLM-based agentic systems, arguing they are fundamentally unsuited for autonomous scientific discovery due to inherent design limitations in problem selection, training data, and evaluation.
- The paper identifies that preference optimization in LLMs compresses hypothesis diversity toward consensus, while current benchmarks fail to capture the multi-step, feedback-driven nature of real-world scientific inquiry.
- To address these gaps, the authors propose integrating scientific simulations as verifiers, implementing persistent world models to maintain epistemic state, and establishing centralized preregistration repositories to capture failure knowledge and improve scientific integrity.

---

[MDGYM: Benchmarking AI Agents on Molecular Simulations](http://arxiv.org/abs/2605.08941)

- MDGYM: introduces a modular evaluation harness for benchmarking AI agents on complex molecular dynamics simulation tasks.
- The framework utilizes three extensible layers—agent, orchestration, and validator—to assess LLM performance across diverse scientific simulation workflows.
- Evaluation results reveal that current LLMs struggle with physical grounding, often failing to produce stable simulation configurations despite successful code generation.

---

[PnP-Corrector: A Universal Correction Framework for Coupled Spatiotemporal Forecasting](http://arxiv.org/abs/2605.08935)

- PnP-Corrector: introduces a model-agnostic framework that decouples physical simulation from error correction by freezing pre-trained Physics Engines and training a lightweight Correction Agent to mitigate Reciprocal Error Amplification.
- The framework utilizes the DSLCast Architecture, which integrates an Axially-Gated Block for efficient spatial feature extraction and a Differentiable Semi-Lagrangian Advection Block to explicitly model physical advective motion.
- By employing an autoregressive predict-then-correct loop, the Correction Agent proactively counteracts systematic biases, ensuring long-term stability and physical realism in coupled spatiotemporal forecasting.

---

[Quantitative Comparison of Credible Compilation and Verification In Coding Agent Compiler Development](http://arxiv.org/abs/2605.08927)

- Axon (Verified Compiler Framework): introduces a quantitative comparison between credible compilation and full verification approaches for compiler optimization development using an LLM-based coding agent.
- The study demonstrates that full verification requires approximately an order of magnitude more engineering effort than credible compilation, primarily due to the complexity of developing machine-checked proofs.
- Results indicate that while verified optimizations are more robust, they often suffer from performance overheads due to algorithm choices prioritized for provability, whereas credible compilation allows for more complex, performance-tuned optimizations.

---

[OPT-BENCH: Evaluating the Iterative Self-Optimization of LLM Agents in Large-Scale Search Spaces](http://arxiv.org/abs/2605.08904)

- OPT-Agent: introduces a framework that emulates human-like cognitive adaptation through a perception–memory–reasoning loop to iteratively refine solutions based on environmental feedback.
- The framework utilizes a structured workflow consisting of drafting, improving, and debugging actions to navigate complex search spaces in both continuous machine learning and discrete NP-hard domains.
- By integrating environmental feedback, the agent bridges the gap between initial hypotheses and optimal solutions, though performance remains constrained by the base capacity of the LLMs.

---

[ACE-SKILL: Bootstrapping Multimodal Agents with Prioritized and Clustered Evolution](http://arxiv.org/abs/2605.08887)

- ACE-SKILL: introduces a co-evolutionary framework that jointly optimizes rollout allocation and knowledge organization to enable self-evolving multimodal agents, utilizing a Prioritized Sampler and a Clustered Organizer.
- The framework employs a Prioritized Sampler to focus LLM rollout resources on informative samples using lazy-decay proficiency tracking, while a Clustered Organizer isolates knowledge into tactical and strategic components to mitigate interference.
- ACE-SKILL enables open-source LLMs to match proprietary models on multimodal benchmarks and supports zero-shot transfer of bootstrapped knowledge to smaller-scale models without additional training.

---

[On MMS, APS and XOS](http://arxiv.org/abs/2605.08859)

- Greedy-average allocation algorithm: introduces a randomized allocation framework for XOS agents that achieves an α-MMS approximation for α > 11/40 when the number of agents n is sufficiently large.
- The framework utilizes a potential function β to guide bundle selection and incorporates filtering and stealing procedures to ensure that remaining agents maintain sufficient value throughout the allocation process.
- The approach extends from identical to different XOS valuations by employing a doubling procedure to establish performance bounds for large n, effectively bypassing the 1/4-approximation barrier.

---

[When Agents Overtrust Environmental Evidence: An Extensible Agentic Framework for Benchmarking Evidence-Grounding Defects in LLM Agents](http://arxiv.org/abs/2605.08828)

- EnvTrustBench (Evidence-Grounding Defect Benchmarking Framework): introduces an extensible agentic framework for benchmarking evidence-grounding defects (EGDs) where LLMs treat unreliable environment-facing claims as authoritative ground for action.
- The framework generates concrete test cases from user-defined scenarios, including Task Scenario, Workspace, Environment, Task Objective, and Validation Oracle, to evaluate whether LLMs can distinguish true environment states from misleading adversarial claims.
- Evaluation across 14 LLM-scaffold stacks reveals that EGDs are a pervasive reliability problem, with most agents failing to verify environmental evidence before executing task-incorrect paths.

---

[Mirror, Mirror on the Wall: Can VLM Agents Tell Who They Are at All?](http://arxiv.org/abs/2605.08816)

- Mirror-guided self-identification benchmark: introduces a controlled 3D environment to evaluate whether embodied LLMs can infer hidden body attributes from mirror reflections to guide goal-directed behavior.
- The framework employs diagnostic interventions including mirror removal, misleading linguistic cues, and occluded reflections to distinguish grounded self-identification from shortcuts, priors, or confabulation.
- The study utilizes process-level metrics such as mirror consultation, temporal ordering, and self-attribution to assess whether LLM agents possess functional self-grounding rooted in perception and action.

---

[AgentSlimming: Towards Efficient and Cost-Aware Multi-Agent Systems](http://arxiv.org/abs/2605.08813)

- AgentSlimming: introduces a training-free framework that optimizes multi-agent workflows through structural pruning and semantic quantization to achieve cost-efficient agentic collaboration.
- The framework utilizes a hybrid importance evaluation mechanism, integrating topological priors and functional contributions via Reciprocal Rank Fusion (RRF) to identify and compress redundant agent nodes.
- AgentSlimming achieves significant token cost reductions of up to 78.9% across diverse benchmarks while maintaining or improving reasoning performance compared to unoptimized multi-agent systems.

---

[EvoMAS: Learning Execution-Time Workflows for Multi-Agent Systems](http://arxiv.org/abs/2605.08769)

- EvoMAS: introduces a framework for execution-time multi-agent workflow construction that dynamically adapts agent coordination based on an evolving task state.
- The system utilizes a Planner-Evaluator-Updater pipeline to construct explicit task states and a learned Workflow Adapter to instantiate stage-specific layered workflows from a fixed candidate agent pool.
- EvoMAS optimizes workflow selection via reinforcement learning using sparse terminal task success, outperforming static multi-agent design methods on complex, long-horizon tasks.

---

[UserGPT Technical Report](http://arxiv.org/abs/2605.08766)

- UserGPT: introduces a principled framework for enhancing LLM reasoning in persona understanding by distilling noisy behavioral histories into structured tags and narrative summaries.
- The framework utilizes a User Behavior Simulation Engine and Data-Centric Semantization to create high-fidelity training data, followed by a curriculum-driven post-training strategy to improve temporal reasoning and logical consistency.
- UserGPT achieves significant performance gains on the newly established HPR-Bench and enables efficient, incremental user profiling for next-generation AI agent memory.

---

[When LLMs Team Up: A Coordinated Attack Framework for Automated Cyber Intrusions](http://arxiv.org/abs/2605.08763)

- CAESAR (Coordinated Adversarial Execution and Strategic Reasoning): introduces a coordinated multi-agent framework that decomposes intrusion workflows into five specialized roles—Detective, Strategist, General, Executor, and Validator—to improve task success and reduce performance variance.
- The framework utilizes a round-based protocol with a persistent knowledge base and validator-gated memory to ensure information flow, artifact provenance, and budget-aware plan selection across heterogeneous LLM backends.
- Experimental results on CTF challenges and social-engineering scenarios demonstrate that CAESAR outperforms single-agent baselines by enabling structured, evidence-driven reasoning and adaptive behavior.

---

[Omni-DeepSearch: A Benchmark for Audio-Driven Omni-Modal Deep Search](http://arxiv.org/abs/2605.08762)

- Omni-DeepSearch: introduces a benchmark for audio-driven omni-modal deep search, requiring models to infer clues from audio and iteratively invoke text, image, and video search tools to perform multi-hop reasoning.
- The framework utilizes a multi-stage filtering pipeline to ensure that all tasks are audio-dependent, retrieval-demanding, and uniquely verifiable across 15 fine-grained categories.
- Experimental results demonstrate that while frontier models show progress, the task remains challenging due to bottlenecks in audio entity inference, query formulation, and cross-modal verification.

---

[Beyond the All-in-One Agent: Benchmarking Role-Specialized Multi-Agent Collaboration in Enterprise Workflows](http://arxiv.org/abs/2605.08761)

- ENTCOLLABBENCH: introduces a benchmark for evaluating role-specialized multi-agent collaboration in enterprise environments, utilizing Agent Layer, Enterprise Service Layer, Access Control, Data Isolation, Evaluation Pipeline, and Judgment Module.
- The framework simulates a permission-isolated organization where 11 LLM agents across six departments must perform cross-departmental delegation and stateful workflow execution.
- Evaluation is based on objective execution traces, database state verification, and deterministic policy adjudication rather than natural-language response judging.

---

[Omni-scale Learning-based Sequential Decision Framework for Order Fulfillment of Tote-handling Robotic Systems](http://arxiv.org/abs/2605.08758)

- OLSF-TRS: introduces a generalized, scalable framework that integrates structured combinatorial optimization with multi-agent reinforcement learning to coordinate order, tote, and robot decisions in warehouse fulfillment.
- The framework utilizes bisimulation quotienting to construct abstract Markov decision processes, enabling efficient policy learning across heterogeneous system configurations and scales.
- By employing a centralized MAPPO critic, the system effectively resolves inter-agent dependencies and mitigates congestion-induced inefficiencies in large-scale, high-concurrency warehouse environments.

---

[AHD Agent: Agentic Reinforcement Learning for Automatic Heuristic Design](http://arxiv.org/abs/2605.08756)

- AHD Agent: introduces a tool-integrated, multi-turn framework that empowers LLMs to proactively decide between heuristic generation and tool-based evidence retrieval for combinatorial optimization.
- The framework utilizes an agentic reinforcement learning system with an environment synthesis pipeline to train a compact 4B-parameter model in generalizable heuristic design.
- Experimental results demonstrate that the agent matches or surpasses state-of-the-art baselines using significantly fewer evaluations and exhibits strong generalization across diverse problem domains.

---

[Communicating Sound Through Natural Language](http://arxiv.org/abs/2605.08750)

- LAC (Lexical Acoustic Coding): introduces a framework that projects audio into interpretable acoustic descriptors, quantizes them into a lexical code, and transmits the sound as structured English prose between LLM agents.
- The system utilizes a shared vocabulary to map waveforms to a d-dimensional lexical code, which is then parsed by a receiver agent to reconstruct audio via a deterministic hybrid synthesizer.
- Closed-loop refinement iteratively adjusts renderer controls to ensure the synthesized waveform aligns with the transmitted lexical acoustic constraints.

---

[Done, But Not Sure: Disentangling World Completion from Self-Termination in Embodied Agents](http://arxiv.org/abs/2605.08747)

- VIGIL (Vision-based Interaction and Grounded Independent-judgment Logic): introduces an evaluation framework that decouples world-state completion from terminal commitment by requiring agents to issue semantically verifiable reports under a no-feedback contract.
- The framework utilizes Egocentric RGB Observation, Bounded Dialogue History, and a Native Action Space to isolate closure failures from execution traps across diagnostic and compositional task tiers.
- Empirical results across 20 models demonstrate that execution and terminal commitment are separable, with models exhibiting distinct failure profiles such as false reports and no-report exhaustion that aggregate metrics typically conflate.

---

[HULK: Large-scale Hierarchical Coordination under Continual and Uncertain Temporal Tasks](http://arxiv.org/abs/2605.08722)

- HULK: introduces a hierarchical coordination framework that decomposes global temporal missions into subtasks and assigns them to subteams using a receding-horizon approach to handle continual and uncertain task arrivals.
- The framework integrates a global task assignment layer with a local coordination layer, employing specific strategies for static, unknown, and dynamic task environments to ensure computational efficiency and robustness.
- HULK utilizes mixed-integer linear programming and coalition formation techniques to manage large-scale heterogeneous multi-agent systems under temporal constraints and environmental uncertainties.

---

[Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](http://arxiv.org/abs/2605.08721)

- DEPT: introduces a dual-timescale evolutionary perception mechanism that detects impasse by quantifying dual-scale value baseline divergence alongside match entropy.
- Upon perceiving training collapse, the framework activates asymmetric advantage reshaping to dynamically modulate the optimization landscape and restore gradient signals.
- The method effectively prevents policy degeneration in open-ended social language games by penalizing dominant outcomes and amplifying rare trajectories to enforce sustained strategic exploration.

---

[Debugging the Debuggers: Failure-Anchored Structured Recovery for Software Engineering Agents](http://arxiv.org/abs/2605.08717)

- PROBE (Failure-Anchored Structured Recovery for Software Engineering Agents): introduces a framework that transforms failed-run telemetry into structured evidence, structured diagnosis, and bounded recovery guidance to improve the recoverability of LLM-based software engineering agents.
- The framework utilizes a Telemetry Layer to preserve fine-grained runtime signals, a Diagnosis Layer to fuse evidence into a grounded diagnosis, and a Guidance Gate to ensure recovery instructions are actionable and bounded.
- Evaluations across repository-level repair, enterprise workflow, and AIOps settings demonstrate that PROBE significantly improves diagnosis accuracy and recovery rates compared to baseline methods by bridging the gap between failure identification and actionable recovery.

---

[AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems](http://arxiv.org/abs/2605.08715)

- AgentForesight: introduces a deployment-time online auditing framework that monitors multi-agent systems step-by-step to detect and localize decisive errors before they propagate into full-trajectory failures.
- The framework utilizes AFTRAJ-2K, a curated corpus of safe and annotated unsafe trajectories, to train the AgentForesight-7B auditor using a coarse-to-fine reinforcement learning recipe.
- AgentForesight-7B employs a two-stage training process consisting of failure-boundary alignment via BPPO and three-axis verdict sharpening via GRPO to achieve precise step-level localization of failures.

---

[AgentPSO: Evolving Agent Reasoning Skill via Multi-agent Particle Swarm Optimization](http://arxiv.org/abs/2605.08704)

- AgentPSO: introduces a framework that evolves multi-agent reasoning skills by treating each agent as a particle in a swarm, iteratively refining natural-language instructions through personal-best, global-best, and self-reflective guidance.
- The framework utilizes an Independent Solving Agent, Peer Observation Module, Self-Reflective Direction Generator, PSO-like Velocity Updater, Skill Updater, Personal-Best Memory, and Global-Best Memory to improve reasoning without updating backbone LLM parameters.
- AgentPSO demonstrates that evolved reasoning skills are transferable across different benchmarks and backbone LLMs, achieving superior performance compared to static multi-agent debate methods while reducing test-time computational overhead.

---

[RewardHarness: Self-Evolving Agentic Post-Training](http://arxiv.org/abs/2605.08703)

- RewardHarness: introduces a self-evolving agentic reward framework that reframes reward modeling as context evolution by iteratively refining a library of Skills and Tools while keeping model weights fixed.
- The framework utilizes an Orchestrator to select relevant artifacts from the Library, which a frozen Sub-Agent then employs to construct interpretable reasoning chains for preference judgment.
- By analyzing reasoning successes and failures against ground-truth labels, the system automatically updates its library without additional human annotation, achieving high data efficiency and performance.

---

[SKILLMASTER: Toward Autonomous Skill Mastery in LLM Agents](http://arxiv.org/abs/2605.08693)

- SKILLMASTER: introduces a training framework that enables LLMs to autonomously develop, refine, and select reusable skills through tool-integrated reasoning and reinforcement learning.
- The framework utilizes a counterfactual utility reward to evaluate skill modifications on probe tasks, providing explicit signals for training skill-editing decisions.
- DualAdv-GRPO decouples advantage normalization for task-solving actions and skill-management decisions, ensuring stable joint optimization within a unified policy.

---

[PrepBench: How Far Are We from Natural-Language-Driven Data Preparation?](http://arxiv.org/abs/2605.08687)

- PrepBench: introduces a benchmark for evaluating NL-driven data preparation across three core capabilities: interactive disambiguation, prep-code generation, and code-to-workflow translation.
- The framework utilizes an agent-based pipeline to construct ground-truth assets, including a Disambiguation Knowledge Base and executable workflows, to systematically measure LLM performance on complex, multi-step data preparation tasks.
- Experimental results demonstrate that while LLMs show promise, they struggle with ambiguity resolution, handling data irregularities, and generating structurally valid workflows, highlighting significant gaps in current end-to-end NL-driven data preparation systems.

---

[Iterative Critique-and-Routing Controller for Multi-Agent Systems with Heterogeneous LLMs](http://arxiv.org/abs/2605.08686)

- Iterative Critique-and-Routing Controller: introduces a multi-turn coordination framework that models agent interaction as a finite-horizon MDP, utilizing a Controller (LLM-based router and critic) to evaluate intermediate drafts and selectively invoke agents from an Agent Pool (heterogeneous LLMs with varying capabilities).
- The framework employs a Composite Reward (rule-based metric for routing and verification) and Lagrangian Relaxation (optimization technique for constrained agent utilization) to balance answer quality against computational costs.
- Training is performed via Group Relative Policy Optimization (GRPO) (RL algorithm for training the controller), enabling the system to outperform one-shot routing baselines while maintaining strict usage constraints on stronger, more expensive models.

---

[MLS-Bench: A Holistic and Rigorous Assessment of AI Systems on Building Better AI](http://arxiv.org/abs/2605.08678)

- MLS-Bench: introduces a benchmark for evaluating whether AI systems can invent generalizable and scalable ML methods, with Task Specification, Validity Enforcement, Unified Scoring, Agent Harness, Editable Codebase, Protected Codebase, Baseline Implementations, Evaluation Settings, Compute Budget, and Seed Policy.
- The framework evaluates LLM agents on 140 tasks across 12 domains, requiring agents to improve targeted ML components while maintaining performance across controlled settings.
- The benchmark exposes a significant method-discovery gap, finding that current LLMs are more proficient at engineering-style tuning than at genuine scientific invention.

---

#### 8th May 2026

[LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling](http://arxiv.org/abs/2605.08083)

- AutoTTS: introduces an environment-driven framework that automates the discovery of test-time scaling strategies by shifting human effort from manual heuristic design to constructing replayable discovery environments.
- The framework utilizes an explorer LLM to iteratively propose and refine controllers, which are evaluated against pre-collected reasoning trajectories to ensure efficient, deterministic, and scalable search.
- By employing beta parameterization and fine-grained execution trace feedback, the system effectively navigates the high-dimensional control space to discover robust strategies that generalize across diverse models and benchmarks.

---

[The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents](http://arxiv.org/abs/2605.08060)

- The Memory Curse: introduces a systematic study of how expanded interaction history in LLM agents paradoxically degrades cooperation in multi-agent social dilemmas due to a cognitive vulnerability to accumulated negative content.
- The research identifies that while short-term memory facilitates trust repair, extended memory often triggers defensive, history-following reasoning patterns that override cooperative priors.
- The authors demonstrate that this "memory curse" can be mitigated by targeted fine-tuning to promote forward-looking reasoning or by sanitizing memory content, proving the phenomenon is driven by reasoning style rather than context length.

---


[A Roadmap of Mixed Reality Body Doubling for Adults with ADHD](http://arxiv.org/abs/2605.07851)

- Body Doubling Framework: introduces a multidimensional model for designing and evaluating body doubling setups to support adults with ADHD through Individual Motivation, Agent-Related Dimensions, Interaction-Related Dimensions, Contextual Dimensions, and Efficacy of the Approach.
- The framework categorizes body doubling setups using a roadmap that accounts for agent characteristics, interaction dynamics, and environmental context to identify research gaps in mixed reality and interactive systems.
- This research provides a structured approach for future empirical studies and the development of personalized, technology-mediated body doubling interventions for neurodivergent individuals.

---



[Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners](http://arxiv.org/abs/2605.08019)

- LRMs: introduces a comparative study evaluating frontier Large Reasoning Models against human game learners and traditional reinforcement learning agents using VGDL-fMRI datasets.
- The research demonstrates that frontier LRMs exhibit behavioral patterns and brain-encoding representations that align significantly closer to human game-learning than deep reinforcement learning baselines.
- The study establishes that LRM representations predict human BOLD activity an order of magnitude better than RL alternatives, suggesting these models capture transferable, human-like in-context learning dynamics.

---

[Collaborator or Assistant? How AI Coding Agents Partition Work Across Pull Request Lifecycles](http://arxiv.org/abs/2605.08017)

- Collaborator-Assistant Spectrum: introduces a process-centric framework to analyze how AI coding agents partition operational agency and governance authority across pull request lifecycles.
- The framework classifies tools into Collaborator or Assistant paradigms based on initiation patterns and review-routing behaviors observed in 29,585 pull request lifecycles.
- It utilizes an Initiator × Approver taxonomy to disentangle operational work from merge governance, revealing that while agents frequently initiate tasks, merge authority remains predominantly human.

---

[Learning CLI Agents with Structured Action Credit under Selective Observation](http://arxiv.org/abs/2605.08013)

- A3 (Action Advantage Assignment): introduces a reinforcement learning paradigm for CLI agents that addresses partial workspace observation and sparse action credit through structured action analysis.
- The framework utilizes σ-Reveal to construct token-budgeted workspace views and employs a three-channel advantage mechanism—episode, turn, and tree scope—to assign credit to shell actions based on their structural intent.
- The approach is evaluated on the ShellOps dataset, demonstrating improved performance in complex filesystem interaction tasks by leveraging native shell syntax for credit assignment at a computational cost comparable to standard agentic RL.

---

[Interpreting Reinforcement Learning Agents with Susceptibilities](http://arxiv.org/abs/2605.08007)

- Susceptibility Framework: introduces a method for neural network interpretability that studies the response of posterior expectation values of observables to perturbations of the regret landscape, utilizing Conv 1-3, FF 1, FF 2, Policy, SGLD, and Regret Landscape.
- The framework generalizes susceptibilities to reinforcement learning by measuring how initial state distribution perturbations affect the effective complexity of specific architectural components.
- Empirical validation through activation-steering and direction-conditioned posterior regret confirms that susceptibility-detected parameter-space structures correspond to functionally meaningful internal agent computations.

---

[Tool Calling is Linearly Readable and Steerable in Language Models](http://arxiv.org/abs/2605.07990)

- Tool Calling is Linearly Readable and Steerable in Language Models: introduces a method for steering LLM tool selection by adding a mean-difference vector to the residual stream, which redirects tool choice and automatically adapts argument schemas.
- The research demonstrates that tool identity is linearly encoded within a low-dimensional subspace of the model's internal activations, allowing for precise intervention without weight updates.
- Mechanistic analysis reveals that tool selection is driven by a three-stage circuit involving early-layer features, mid-layer attention heads, and late-layer formatting, with instruction tuning essential for routing these internal signals to structured output.

---

[Graph Representation Learning Augmented Model Manipulation on Federated Fine-Tuning of LLMs](http://arxiv.org/abs/2605.07961)

- AugMP: introduces an adversarial framework that leverages graph representation learning to synthesize malicious updates that disrupt the federated fine-tuning of LLMs while evading detection.
- The framework utilizes a VGAE to capture feature correlations among benign updates and an iterative manipulation algorithm based on an augmented Lagrangian dual formulation to enforce geometric stealth constraints.
- Experimental results demonstrate that AugMP significantly degrades the accuracy of global and local LLMs across multiple backbones while maintaining statistical and geometric consistency with benign updates.

---

[Ask Early, Ask Late, Ask Right: When Does Clarification Timing Matter for Long-Horizon Agents?](http://arxiv.org/abs/2605.07937)

- Forced-injection framework: introduces a methodology to measure the value of information (VOI) by injecting ground-truth clarifications at controlled trajectory points across diverse long-horizon agent tasks.
- The study demonstrates that clarification timing is highly dimension-dependent, with goal information requiring early intervention and input information remaining recoverable through mid-trajectory.
- Empirical results reveal that current LLMs fail to naturally ask within these optimal timing windows, establishing a quantitative foundation for designing future timing-aware clarification policies.

---

[TraceFix: Repairing Agent Coordination Protocols with TLA+ Counterexamples](http://arxiv.org/abs/2605.07935)

- TraceFix: introduces a verification-first pipeline for LLM multi-agent coordination that uses counterexample-driven repair to ensure protocol correctness.
- The framework utilizes an Orchestration Agent to synthesize a Protocol Topology IR and PlusCal Coordination Logic, which are then validated by the TLC Model Checker to eliminate concurrency hazards.
- Verified protocols are compiled by the Prompt Compiler into per-agent prompts and enforced at runtime by a Topology Monitor to prevent out-of-protocol coordination operations.

---

[AgentEscapeBench: Evaluating Out-of-Domain Tool-Grounded Reasoning in LLM Agents](http://arxiv.org/abs/2605.07926)

- AgentEscapeBench: introduces an escape-room-style benchmark that evaluates the ability of LLMs to perform tool-grounded reasoning in unfamiliar environments using Template Library, DAG Skeleton Generator, Source Annotator, Value Instantiator, Deterministic Forward Executor, Narrative Generator, and Evaluation Sandbox.
- The framework utilizes a six-stage automated pipeline to construct directed acyclic graph tasks that require agents to infer, execute, and revise tool-use procedures under explicit long-range dependency constraints.
- Experimental results across sixteen LLMs demonstrate that performance degrades monotonically with dependency depth, highlighting significant bottlenecks in long-range state tracking and intermediate-result propagation.

---

[One World, Dual Timeline: Decoupled Spatio-Temporal Gaussian Scene Graph for 4D Cooperative Driving Reconstruction](http://arxiv.org/abs/2605.07910)

- DUST (DecoUpled Spatio-Temporal) Gaussian Scene Graph: introduces a cooperative reconstruction framework that decouples pose trajectories into source-specific timelines to eliminate gradient conflicts caused by temporal asynchrony in multi-source driving data.
- The framework utilizes a shared canonical Gaussian representation for appearance consistency while maintaining independent pose timelines for vehicle and infrastructure sensors to enable high-fidelity 4D scene reconstruction.
- DUST incorporates a static anchor-based pose correction pipeline and a pose-regularized joint optimization scheme to ensure robust initialization and stable training under asynchronous conditions.

---

[ADKO: Agentic Decentralized Knowledge Optimization](http://arxiv.org/abs/2605.07863)

- ADKO (Agentic Decentralized Knowledge Optimization): introduces a modular framework for collaborative black-box optimization that enables autonomous agents to share semantically rich insights via Private Gaussian Process (GP) surrogate, Knowledge token, Graph-structured communication, LM reasoning module, Fidelity-aware token pruning, and Reasoning score without exposing raw data.
- The framework utilizes a Reasoning score that synthesizes private GP-based exploitation and exploration with collaborative success-attraction and failure-avoidance signals derived from peer-shared Knowledge tokens.
- ADKO provides a rigorous theoretical foundation for decentralized optimization by decomposing cumulative regret into GP error, LM bias, LM noise, and compression loss, while validating performance through neural architecture search and scientific discovery experiments.

---

[VISTA: Decentralized Machine Learning in Adversary Dominated Environments](http://arxiv.org/abs/2605.07841)

- VISTA: introduces an adaptive decentralized learning framework that dynamically tunes acceptance thresholds and learning rates to ensure convergence in adversary-dominated environments.
- The framework utilizes a consistency-based acceptance rule to incentivize rational worker nodes to provide reliable gradient reports while mitigating adversarial noise.
- VISTA achieves asymptotic convergence rates matching standard SGD by balancing update frequency and estimation accuracy through an adaptive signal-to-noise tolerance parameter.

---

[RELAGENT: LLM Agents as Data Scientists for Relational Learning](http://arxiv.org/abs/2605.07840)

- RELAGENT: introduces an agentic framework that automates relational learning by iteratively searching for optimal SQL feature programs and predictive models using an LLM agent interacting with a database via specialized tools.
- The framework decouples the search phase, where an LLM agent refines feature programs and model configurations, from the inference phase, which executes the final SQL-based predictor without further LLM calls.
- By maintaining relational structure externally and using executable SQL queries, the approach achieves intrinsic interpretability and relational invariance while matching or exceeding the performance of fixed-architecture relational models.

---

[Unsafe by Flow: Uncovering Bidirectional Data-Flow Risks in MCP Ecosystem](http://arxiv.org/abs/2605.07836)

- MCP-BiFlow (Model Context Protocol Bidirectional Data-Flow Analysis Framework): introduces a static analysis framework that detects bidirectional data-flow vulnerabilities in Model Context Protocol servers by combining MCP Entrypoint Recovery, MCP-Specific Taint Specification, and Bidirectional Interprocedural Taint Analysis.
- The framework addresses request-side propagation, where requester-controlled inputs reach sensitive operations, and return-side propagation, where untrusted external content influences downstream LLM reasoning or tool invocations.
- Evaluated on 32 confirmed vulnerabilities, the framework achieves 93.8% recall and identifies 118 unique vulnerability paths across 87 real-world Model Context Protocol server repositories.

---

[Many-to-Many Multi-Agent Pickup and Delivery](http://arxiv.org/abs/2605.07835)

- M2M: introduces a sequential many-to-many MAPD algorithm that optimizes task allocation across agents, tasks, and multiple potential pickup and delivery locations using Initial Task Allocation, Large Neighborhood Search, and Priority Based Search.
- The framework addresses the NP-hard 4-dimensional assignment problem by decomposing cost tensors into matrices to enable efficient computation of task allocations in automated warehouse environments.
- Experimental results demonstrate that M2M consistently outperforms existing state-of-the-art methods in task throughput and maintains scalability for up to 150 agents and tasks within a 1-second computation budget.

---

[CyBiasBench: Benchmarking Bias in LLM Agents for Cyber-Attack Scenarios](http://arxiv.org/abs/2605.07830)

- CyBiasBench: introduces a comprehensive benchmarking framework to quantify attack-selection bias in LLM agents across diverse cyber-attack scenarios using Prompt Design, Agent Penetration Testbed, and Evaluation Metrics Suite.
- The framework utilizes a Kali Linux Container for isolated execution against Target Applications, employing a Deterministic Classifier and Verifier Pipeline to measure agent behavior independently of self-reported logs.
- The research reveals that LLM agents exhibit persistent, agent-specific attack-selection biases that remain stable across prompt variations and are not consistently improved by explicit steering, a phenomenon termed bias momentum.

---

[SCENE: Recognizing Social Norms and Sanctioning in Group Chats](http://arxiv.org/abs/2605.07823)

- SCENE: introduces a benchmark for evaluating how LLMs infer and adapt to implicit social norms within multi-party group chats through interaction with scripted personas.
- The framework utilizes a two-stage generation process to create synthetic, interactionally plausible scenarios where a subject agent must recognize sanctioning signals and adjust its behavior accordingly.
- Evaluation metrics include repair rates after sanctioning, breach rates based on prior demonstrations, and compliance under opposing norm pairs to assess the social adaptation capabilities of various LLMs.

---

[GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning](http://arxiv.org/abs/2605.07817)

- GazeVLM: introduces a multimodal architecture that internalizes active vision by dynamically modulating the causal attention mask of a VLM using <LOOK> tokens to focus on task-relevant visual regions.
- The framework employs a two-stage training process, starting with supervised fine-tuning on curated gaze-reasoning traces, followed by reinforcement learning using GRPO to optimize autonomous visual navigation.
- By applying a continuous suppression bias to irrelevant visual tokens, GazeVLM achieves high-resolution multimodal reasoning without the computational overhead of external cropping tools or context window inflation.

---

[CktFormalizer: Autoformalization of Natural Language into Circuit Representations](http://arxiv.org/abs/2605.07782)

- CktFormalizer: introduces a framework that redirects LLM-driven hardware generation through a dependently-typed HDL embedded in Lean 4 to ensure structural correctness and enable formal verification.
- The framework utilizes an LLM Coding Agent that leverages immediate compile-time feedback from the Lean type system to iteratively refine hardware designs, effectively bridging the gap between natural language specifications and synthesizable silicon.
- CktFormalizer incorporates a closed-loop PPA optimization strategy and automated theorem proving to produce formally verified, optimized hardware implementations that are robust against backend synthesis and routing failures.

---

[Coding Agents Don’t Know When to Act](http://arxiv.org/abs/2605.07769)

- FIXEDBENCH: introduces a benchmark to evaluate whether coding agents can recognize when software issues are already resolved and abstain from making unnecessary code changes.
- The study demonstrates that LLMs exhibit an action bias, frequently applying spurious edits to already-fixed code, which can be mitigated by explicit instructions to verify the issue before acting.
- The research highlights that while verify-then-abstain prompting improves abstention rates, it introduces a risk of over-abstention on tasks that genuinely require code modifications.

---

[Emergence of Social Reality of Emotion through a Social Allostasis Model with Dynamic Interpretants](http://arxiv.org/abs/2605.07761)

- Social Allostasis Model: introduces a computational framework where two agents co-construct social reality by aligning bodily control goals and symbol interpretations through active inference and the Metropolis–Hastings Naming Game.
- The framework utilizes POMDP-based generative models to enable agents to perform allostatic regulation of bodily states while simultaneously negotiating shared emotional concepts via symbolic communication.
- Experimental results demonstrate that agents successfully converge on shared prior preferences and dynamic symbol interpretations, confirming the emergence of social reality from the interplay of individual bodily regulation and social interaction.

---

[Alternating Target–Path Planning for Scalable Multi-Agent Coordination](http://arxiv.org/abs/2605.07744)

- TAPF Framework: introduces an iterative refinement approach that decouples target assignment from pathfinding to achieve scalability in multi-agent coordination by leveraging Initial Assignment, MAPF Solver, Feedback Mechanism, Reassignment Strategy, and Final Path Optimization.
- The framework utilizes feedback-driven reassignment loops to identify bottleneck agents via Delay-Based Selection or Spectral Bottleneck Sampling, subsequently refining assignments using PIBT or Local Hungarian methods.
- Empirical results demonstrate that the framework scales to thousands of agents, significantly outperforming traditional Conflict-Based Search methods in large-scale logistics scenarios.

---

[Securing the Dark Matter: A Semantic-Enhanced Neuro-Symbolic Framework for Supply Chain Analysis of Opaque Industrial Software](http://arxiv.org/abs/2605.07737)

- SCAA (Supply Chain Analysis Agent): introduces a neuro-symbolic framework that reconstructs behavioral semantics from opaque industrial binaries to perform tractable global risk reasoning.
- The framework utilizes a Reflexive Prompting pipeline to constrain a local LLM agent with structural verification, effectively suppressing hallucinations during semantic lifting.
- A domain-adapted Graphormer and surjective graph transformation enable scalable vulnerability detection and APT fingerprinting across complex industrial supply chains.

---

[SARC: A Governance-by-Architecture Framework for Agentic AI Systems](http://arxiv.org/abs/2605.07728)

- SARC: introduces a governance-by-architecture framework that treats constraints as first-class specification objects, comprising S, A, R, and C, enforced at PAG, ATM, PAA, and ER.
- The framework enables auditable-by-construction agentic systems by compiling regulatory obligations into runtime checks that maintain system invariants.
- SARC provides a specification-to-runtime compilation discipline that separates optimization from admissibility, ensuring compliance is verifiable through structured trace derivation.

---

[SOD: Step-wise On-policy Distillation for Small Language Model Agents](http://arxiv.org/abs/2605.07725)

- SOD: introduces a step-wise on-policy distillation framework that adaptively reweights distillation strength based on student-teacher divergence to stabilize training for small language model agents.
- The framework combines GRPO for trajectory-level exploration with a step-wise OPD objective that attenuates misleading teacher signals in high-divergence regions caused by tool-induced state drift.
- Experimental results demonstrate that SOD significantly improves agentic reasoning performance and training stability across math, science, and code benchmarks compared to standard distillation and reinforcement learning baselines.

---

[The AI-Native Large-Scale Agile Software Development Manifesto](http://arxiv.org/abs/2605.07717)

- AI-Native Large-Scale Agile Software Development Framework: introduces a paradigm shift in software engineering by replacing sequential human-centric processes with an intelligent, adaptive system where Human Experts define intent and AI Agents execute tasks.
- The framework utilizes AI Personas, AI Agents, and Skills to enable parallel development, while a Semantic Layer and Knowledge Graphs provide a shared, living context for both humans and agents.
- By integrating MCP Connectors for tool interaction and emphasizing reusable blueprints, the approach facilitates continuous verification and autonomous collaboration across large-scale development teams.

---

[DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain](http://arxiv.org/abs/2605.07699)

- DRIP-R: introduces a benchmark for evaluating LLM agents in retail scenarios governed by ambiguous policies, utilizing a simulation environment with Scenario Construction, Simulation Environment, Evaluation Pipeline, Orchestrator, LLM-as-a-Judge, Customer Agent, User Simulator, Database, and Tool-calling Agent.
- The framework employs a multi-judge evaluation pipeline to assess agent performance across policy adherence, dialogue quality, behavioral alignment, interest alignment, and task-resolution adherence.
- The benchmark systematically exploits real-world policy ambiguities to test how LLMs interpret conflicting rules, justify decisions, and balance stakeholder interests in conversational settings.

---

[GASim: A Graph-Accelerated Hybrid Framework for Social Simulation](http://arxiv.org/abs/2605.07692)

- GASim: introduces a graph-accelerated hybrid framework for large-scale social simulations that utilizes EDG, GOM, and GMP to optimize agent-based modeling performance.
- The framework employs EDG to dynamically partition agents into LLM-based core agents and numerical ordinary agents, significantly reducing computational latency.
- GOM replaces intensive LLM-based retrieval with lightweight graph propagation, while GMP leverages a GAT to enable parallel opinion updates for ordinary agents.

---

[A Multi-Level Agent-Based Architecture for Climate Governance Integrating Cognitive and Institutional Dynamics](http://arxiv.org/abs/2605.07683)

- Multi-level Agent-Based Architecture: introduces a modular simulation framework that integrates micro-level HUMAT-MOA behavioural decision-making, meso-level social influence networks, and macro-level institutional strategy to model democratic climate governance.
- The architecture utilizes a land-use development proposal as a central interaction artefact, enabling the co-evolution of public opinion, organized advocacy, and formal political decision-making.
- By combining empirically grounded cognitive models with strategic institutional rules, the framework provides a transparent and portable approach for simulating complex socio-political dynamics in climate governance.

---

[The Endogeneity of Miscalibration: Impossibility and Escape in Scored Reporting](http://arxiv.org/abs/2605.07671)

- Credibility Game framework: introduces a formal model of strategic reporting where an agent's combined objective of accuracy and non-accuracy benefits leads to an endogenous impossibility of truthful reporting.
- The framework demonstrates that a principal's optimal oversight mechanism is necessarily non-affine, which triggers a structural impossibility that undermines calibration.
- The research establishes that a sharp step-function approval threshold serves as a constructive escape, achieving first-best screening for any strictly proper scoring rule.

---

[Operating Within the Operational Design Domain: Zero-Shot Perception with Vision-Language Models](http://arxiv.org/abs/2605.07649)

- ODD-TAX-232 (Operational Design Domain Taxonomy 232): introduces a framework for zero-shot perception in autonomous driving that utilizes specialized agents including Traffic Signage Expert, Road Marking Expert, Scenery Element Expert, Weather Condition Expert, and Trigger Condition Expert, integrated via a Chained Prompting Pipeline and Knowledge Base within a VLM-based Perception System.
- The research evaluates the effectiveness of various VLMs in identifying fine-grained ODD elements by employing persona-based prompting and structured reasoning to enhance zero-shot classification and detection performance.
- The study demonstrates that while VLMs show potential for offline ODD auditing and compliance reporting, their current performance requires further refinement for safety-critical, real-time on-board deployment.

---

[MAVEN: Multi-Agent Verification-Elaboration Network with In-Step Epistemic Auditing](http://arxiv.org/abs/2605.07646)

- MAVEN: introduces a blackboard-inspired multi-agent framework that decouples factual probing from synthesis to transform LLMs into deliberate, auditable reasoners.
- The framework utilizes an adversarial Skeptic-Researcher-Judge loop to enforce discrete verification gates and parametric probing, ensuring that reasoning trajectories are factually grounded and logically coherent.
- By employing an Adaptive Router and a persistent Knowledge Cache, MAVEN optimizes computational efficiency while maintaining structural rigor across diverse LLM backbones.

---

[Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents](http://arxiv.org/abs/2605.07630)

- PHONESAFETY: introduces a benchmark of 700 safety-critical moments to distinguish between safe judgment and inability to act in LLM-based phone-use agents.
- The framework categorizes agent responses into safe actions, unsafe actions, and failures to do anything useful to prevent misinterpreting harmless outcomes as evidence of safety.
- Empirical results demonstrate that general phone-use capability does not reliably predict safe choices, and that failures to act often stem from operational limitations rather than safety-conscious judgment.

---

[MemCompiler: Compile, Don’t Inject — State-Conditioned Memory for Embodied Agents](http://arxiv.org/abs/2605.07594)

- MemCompiler: introduces a state-conditioned memory compilation paradigm that dynamically selects and compiles relevant experience into dual-channel guidance for embodied agents.
- The framework utilizes a Memory Compiler to maintain a structured Brief State and deliver targeted text and latent Soft-Mem tokens to the Executor, preventing attention dilution.
- MemCompiler significantly improves performance across multiple embodied benchmarks while reducing executor input tokens and latency by optimizing memory delivery for lightweight LLMs.

---

[Deadline-Driven Hierarchical Agentic Resource Sharing for AI Services and RAN Functions in AI-RAN](http://arxiv.org/abs/2605.07547)

- HAF (Hierarchical Agentic Framework): introduces a two-layer architecture for AI-RAN that decouples slow-timescale service placement from fast-timescale GPU/CPU allocation to optimize SLO fulfillment.
- The framework utilizes an LLM-based agent for intelligent migration decision-making, supported by a predictive critic to mitigate service interruption costs.
- A closed-form convex allocator ensures hard real-time RAN deadline satisfaction while dynamically distributing remaining compute resources to elastic AI services.

---

[Multi-Environment POMDPs with Finite-Horizon Objectives](http://arxiv.org/abs/2605.07537)

- MEPOMDP: introduces a robust framework for sequential decision-making under uncertainty with multiple environments and finite-horizon objectives, utilizing Multi-belief, Multi-expected payoff, and Mixture of policies to compute optimal values.
- The paper establishes that the value computation problem for MEPOMDPs is PSPACE-complete, providing both a space-efficient deterministic algorithm and a practically efficient algorithm based on bottom-up frontier construction with pruning.
- Empirical evaluations demonstrate that the proposed efficient pruning algorithm significantly outperforms existing state-of-the-art tools across classical benchmarks like Rock Sample, Robot Navigation, and Identification.

---

[Synchronizing Minds through Collective Predictive Coding: A Computational Model of Parent-Infant Homeostatic Co-Regulation](http://arxiv.org/abs/2605.07524)

- MHNG: introduces a computational model of parent–infant homeostatic co-regulation that integrates POMDP (Individual-level active interoceptive inference) with MHNG (Decentralized Bayesian communication mechanism) and CPC (Social-level predictive coding hypothesis) to achieve latent-state alignment.
- The framework utilizes Generative Matrices (Learned internal model parameters) and a Communicative Variable (Shared interactional symbol) to enable agents with asymmetric knowledge to coordinate regulatory actions.
- Simulation results demonstrate that MHNG-mediated interaction facilitates rapid synchronization of Internal Representations (Agent latent belief states) and improves visceral state regulation compared to one-sided control conditions.

---

[InterLV-Search: Benchmarking Interleaved Multimodal Agentic Search](http://arxiv.org/abs/2605.07510)

- InterLV-Search: introduces a benchmark for interleaved multimodal agentic search, which evaluates how LLMs use visual evidence as search pivots to guide multi-hop trajectories across three progressively challenging levels.
- The framework includes an InterLV-Agent that utilizes a reason-act-observe loop, supported by Short-term Memory, Long-term Memory, and Tool Integration to manage complex, non-linear search tasks.
- Experimental results demonstrate that current LLMs struggle with interleaved multimodal search, highlighting the necessity of active visual evidence seeking and robust search control beyond simple text-centric browsing.

---

[MASPrism: Lightweight Failure Attribution for Multi-Agent Systems Using Prefill-Stage Signals](http://arxiv.org/abs/2605.07509)

- MASPrism: introduces a two-stage failure attribution framework that leverages prefill-stage signals from an SLM to identify root causes in multi-agent execution traces without requiring decoding, replay, or task-specific training.
- The framework utilizes a Filtering stage to identify symptom steps via NLL and candidate sources via attention, followed by a Diagnosis stage that reconstructs a focused prompt to perform fine-grained ranking of failure sources.
- By operating exclusively on prefill-stage internal signals, MASPrism achieves significant speedups and cost reductions compared to agent-based or replay-based attribution methods while maintaining high accuracy on long-trace benchmarks.

---

[LiteGUI: Distilling Compact GUI Agents with Reinforcement Learning](http://arxiv.org/abs/2605.07505)

- LiteGUI: introduces a SFT-free training paradigm for lightweight GUI agents by integrating Guided On-policy Distillation and Multi-solution Dual-level GRPO.
- The framework utilizes an automated trajectory generation pipeline to create multi-solution datasets, which are then used to train agents via privileged teacher guidance and dual-level reinforcement learning rewards.
- LiteGUI achieves state-of-the-art performance among lightweight models by jointly aligning macro-level subtask planning with micro-level execution matching, effectively mitigating exploration bottlenecks in long-horizon GUI tasks.

---

[HBEE: Human Behavioral Entropy Engine Pre-Registered Multi-Agent LLM Simulation of Peer-Suspicion-Based Detection Inversion](http://arxiv.org/abs/2605.07472)

- HBEE: introduces a pre-registered multi-agent LLM simulation framework to evaluate the effectiveness of insider threat detection mechanisms against adaptive adversaries.
- The framework utilizes LLM-driven agents to model organizational communication and tests how adaptive OPSEC directives influence detection signals in both cascade and blind defender configurations.
- Empirical results demonstrate a detection inversion where an adaptive mole becomes less suspicious than innocent peers in a cascade-based detection environment, revealing a decoupling of suspicion metrics.

---

[The Moltbook Files: A Harmless Slopocalypse or Humanity’s Last Experiment](http://arxiv.org/abs/2605.07462)

- Moltbook Files: introduces a dataset of 232k posts and 2.2M comments from an AI-agent-populated platform, processed through a Collection Pipeline, Preprocessing Pipeline, and PII Anonymization Pipeline.
- The research utilizes BERTopic Modeling to characterize agent discourse and Fine-tuning Configuration to assess the impact of synthetic data on LLM factuality and alignment.
- The study employs an LLM-as-a-judge Evaluator to demonstrate that while training on agent-generated content degrades model performance, the effect is comparable to fine-tuning on human-generated Reddit data.

---

[EditRefiner: A Human-Aligned Agentic Framework for Image Editing Refinement](http://arxiv.org/abs/2605.07457)

- EditRefiner: introduces a hierarchical, interpretable, and human-aligned agentic framework that reformulates post-editing correction as a perception-reasoning-action-evaluation loop.
- The framework utilizes a Perception Agent to identify flaws, a Reasoning Agent to diagnose them, an Action Agent to perform targeted re-editing, and an Evaluation Agent to ensure alignment with human preferences.
- The system is supported by the EditFHF-15K dataset, which provides extensive human-annotated feedback, including saliency maps and MOS scores, to facilitate the development of self-corrective image editing agents.

---

[Sparse Autoencoders as Plug-and-Play Firewalls for Adversarial Attack Detection in VLMs](http://arxiv.org/abs/2605.07447)

- SAEgis: introduces a lightweight adversarial attack detection framework by inserting sparse autoencoders into pretrained VLMs to identify attack-relevant signals through reconstruction-based latent feature analysis.
- The framework utilizes a difference-of-means approach on sparse latent features to detect adversarial perturbations without requiring additional adversarial training.
- By ensembling sparse autoencoder signals across multiple layers, the system achieves robust detection performance across diverse domains and unseen adversarial attack methods.

---

[GameGen-Verifier: Parallel Keypoint-Based Verification for LLM-Generated Games via Runtime State Injection](http://arxiv.org/abs/2605.07442)

- GameGen-Verifier: introduces an automated verification paradigm for LLM-generated games that decomposes specifications into verifiable keypoints and grounds them into independent verification units for parallel execution.
- The framework utilizes parameter-based state construction and runtime state patching to instantiate specific game states, bypassing the need for long-horizon gameplay exploration.
- GGV-HARNESS provides a two-layer architecture for concurrency management, runtime isolation, and fault recovery, significantly improving verification accuracy and reducing wall-clock time compared to traditional agent-based approaches.

---

[Low-code and No-code with BESSER to Create and Deploy Smart Web Applications](http://arxiv.org/abs/2605.07376)

- BESSER: introduces an open-source low-code framework that enables the design, generation, and deployment of smart web applications through B-UML, Structural Perspective, Agent Perspective, GUI Perspective, Code Generation Pipeline, Backend Generator, Frontend Generator, BAF Generator, and Deployment Service.
- The framework utilizes a model-driven approach to automate the creation of full-stack web applications, including backend APIs, React-based frontends, and LLM-integrated agent backends.
- BESSER streamlines the development lifecycle by integrating directly with cloud platforms like GitHub and Render to provide one-click deployment of generated smart web applications.

---

[FlightSense: An End-to-End MLOps Platform for Real-Time Flight Delay Prediction via Rotation-Chain Propagation Features and Agentic Conversational AI](http://arxiv.org/abs/2605.07364)

- FlightSense: introduces an end-to-end MLOps platform for real-time flight delay prediction that utilizes an XGBoost Classifier, AWS Lambda, Amazon SageMaker, a Streamlit Dashboard, an Amazon Bedrock Nova Micro agent, and a Tool-Use Architecture.
- The framework employs a progressive three-version feature engineering approach, prioritizing rotation-chain propagation features to achieve high predictive accuracy.
- The system integrates an agentic LLM to provide grounded, natural-language delay risk assessments through multiplicative probability compounding based on live weather and operational data.

---

[A Comprehensive Survey on Agent Skills: Taxonomy, Techniques, and Applications](http://arxiv.org/abs/2605.07358)

- Agent Skills Survey: introduces a comprehensive taxonomy for LLM-based agent systems, focusing on the lifecycle of reusable procedural artifacts that bridge the gap between raw tool access and robust task execution.
- The framework categorizes agent systems into five core modules: Skill Representation, Skill Acquisition, Skill Retrieval and Selection, Skill Evolution, and Runtime Governance.
- This survey synthesizes current research on how LLMs can externalize, manage, and refine procedural knowledge to improve the scalability, robustness, and maintainability of autonomous agent ecosystems.

---

[Tools as Continuous Flow for Evolving Agentic Reasoning](http://arxiv.org/abs/2605.07339)

- FlowAgent: introduces a paradigm that reconceptualizes discrete tool chaining as continuous trajectory generation within a semantic space to enable global plan-level reasoning.
- The framework utilizes a Conditional Flow Planner to generate latent trajectories, which are mapped to discrete actions via Discrete Plan Decoding and a Parameterized Stop Mechanism to ensure robust execution.
- FlowAgent incorporates a closed-loop execution scheme with a State Update Operator and Observation Encoder to dynamically absorb environmental feedback, effectively mitigating error accumulation in long-horizon tasks.

---

[RCoT-Seg: Reinforced Chain-of-Thought for Video Reasoning and Segmentation](http://arxiv.org/abs/2605.07334)

- RCoT-Seg: introduces a video-of-thought framework that factorizes Video Reasoning Segmentation into temporal video reasoning and keyframe target perception, utilizing MLLM, AKS, KTG, SAM2, GRPO, and Hungarian-based reward.
- The framework employs an agentic keyframe selection mechanism with a self-critical loop to verify and refine keyframe choices, replacing heuristic sampling with a verifiable decision process.
- RCoT-Seg leverages GRPO reinforcement learning with a matching-aware reward to align reasoning traces with downstream segmentation, achieving state-of-the-art performance on video reasoning and referring segmentation benchmarks.

---

[Beyond Linear Attention: Softmax Transformers Implement In-Context Reinforcement Learning](http://arxiv.org/abs/2605.07333)

- Softmax ICTD: introduces a theoretical framework demonstrating that Transformers with standard softmax attention can implement weighted softmax TD updates in their forward pass.
- The architecture utilizes a dual-head Transformer design where one head computes current value updates and the other computes target values, effectively performing iterative policy evaluation.
- The paper proves that these Transformer parameters are global minimizers of a reinforcement pretraining loss, establishing convergence to the true value function as depth and context length increase.

---

[Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation](http://arxiv.org/abs/2605.07323)

- DoLQ (Discovering Ordinary differential equations with LLM-based Qualitative and quantitative evaluation): introduces a multi-agent framework that integrates LLM-based qualitative reasoning with quantitative numerical validation to discover interpretable ODEs.
- The framework utilizes a Sampler Agent to propose candidate terms, a Parameter Optimizer to refine coefficients, and a Scientist Agent to filter physically implausible models through iterative feedback.
- By combining semantic assessment with residual-based optimization, DoLQ achieves superior success rates in recovering ground-truth symbolic structures across diverse dynamical systems compared to existing symbolic regression methods.

---

[When Stored Evidence Stops Being Usable: Scale-Conditioned Evaluation of Agent Memory](http://arxiv.org/abs/2605.07313)

- Scale-Conditioned Evaluation Protocol: introduces a trajectory-level evaluation framework for LLM agent memory that assesses usability under evidence-preserving growth by monitoring interaction budgets and failure regimes.
- The protocol utilizes four diagnostics—budget-compliant reliability, tail retrieval-call burden, failure-regime decomposition, and usable-scale boundary—to distinguish between intrinsic memory failures and agent-facing interaction inefficiencies.
- Empirical audits across flat, planar, and hierarchical memory interfaces demonstrate that scalable-memory claims must be explicitly conditioned on the specific agent, interface, scale range, and interaction budget.

---

[AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in Vision-Language-Action Models](http://arxiv.org/abs/2605.07308)

- AT-VLA: introduces an adaptive tactile injection mechanism that balances pretrained knowledge with newly learned tactile representations to enhance precision in contact-rich manipulation.
- The framework utilizes a Tactile Reaction Dual-Stream mechanism to decouple sensory processing into a slow visual-language stream and a fast tactile control stream, enabling real-time closed-loop responses within 0.04 seconds.
- By employing a learnable Tactile Gate and Adaptive Cross Attention, the model dynamically modulates modality contributions, ensuring robust performance even when tactile signals are absent during inference.

---

[BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation](http://arxiv.org/abs/2605.07306)

- BioProVLA-Agent: introduces a multi-agent framework for biological laboratory manipulation that integrates protocol parsing, state verification, and embodied execution through the Guiding Decision Agent, Tailored LLM Protocol Agent, VLM-RAG Verification Agent, and VLA Embodied Agent.
- The system utilizes AugSmolVLA to enhance the robustness of the VLA Embodied Agent against visual perturbations common in wet-lab environments, such as transparent labware and illumination shifts.
- By employing a closed-loop verification mechanism, the framework enables reliable long-horizon biological experimentation while reducing dependence on fixed robotic scripts.

---

[SOM: Structured Opponent Modeling for LLM-based Agents via Structural Causal Model](http://arxiv.org/abs/2605.07301)

- SOM (Structured Opponent Modeling): introduces a two-stage framework that decouples opponent model construction and prediction by grounding the process in Structural Causal Models (SCMs) to enable structured and controllable reasoning.
- The framework utilizes Dynamic SCM Construction to build a causal graph of opponent decision-making and Reasoning for Opponent Prediction and Adaptation to perform inference using personalized reasoning examples.
- SOM improves prediction accuracy and adaptability in multi-agent environments by replacing implicit contextual reasoning with explicit, graph-based structured reasoning pathways.

---

[EgoPro-Bench: Benchmarking Personalized Proactive Interaction in Egocentric Video Streams](http://arxiv.org/abs/2605.07299)

- ProAct-Stream: introduces a comprehensive benchmark and proactive streaming model that leverages simulated user profiles and a "short thinking, better interaction" paradigm to enable personalized HMI.
- The framework utilizes a two-stage training process, integrating Supervised Fine-Tuning and Reinforcement Learning to optimize reasoning length and response quality for low-latency interaction.
- EgoPro-Bench provides a robust evaluation protocol across 12 distinct domains, addressing the limitations of reactive MLLMs in temporal alignment and personalized intent understanding.

---

[Can Agents Price a Reaction? Evaluating LLMs on Chemical Cost Reasoning](http://arxiv.org/abs/2605.07251)

- CHEMCOST introduces a benchmark for evaluating LLMs on chemical procurement cost estimation, requiring agents to resolve chemical identities, retrieve supplier quotes, and perform multi-step arithmetic.
- The framework utilizes a ReAct agent architecture equipped with specialized tools to navigate complex, domain-constrained procurement tasks under both clean and noise-injected conditions.
- Experimental results demonstrate that while tool access is necessary for performance, LLMs struggle with evidence integration and robustness to realistic input formatting, highlighting a significant gap in scientific agentic reasoning.

---

[EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation](http://arxiv.org/abs/2605.07247)

- EnvSimBench: introduces a diagnostic framework and benchmark that reframes LLM-based environment simulation as a fully observable state prediction task to address hallucination, logical inconsistency, and state drift.
- The framework utilizes a constraint-driven MDP formulation that provides models with explicit before-state configurations and implementation code to enable independent, verifiable evaluation of simulation fidelity.
- Systematic evaluation reveals a universal state-change cliff where LLMs fail catastrophically when multiple state variables require simultaneous updates, a gap addressed by the specialized Balance2 model.

---

[MEMOREPAIR: Barrier-First Cascade Repair in Agentic Memory](http://arxiv.org/abs/2605.07242)

- MEMOREPAIR: introduces a barrier-first cascade-repair contract that manages the lifecycle of derived artifacts in agentic memory by withdrawing invalidated cascades and selectively republishing validated successors.
- The framework models agentic memory as a directed provenance graph and utilizes a min-cut solver to optimize the repair-cost tradeoff when reconstructing artifacts after deletion, correction, or migration events.
- Experimental results on ToolBench and MemoryArena demonstrate that MEMOREPAIR effectively eliminates stale-memory exposure while recovering the majority of validated successors at a significantly lower repair-operator cost than exhaustive methods.

---

[Rethinking Priority Scheduling for Sequential Multi-Agent Decision Making in Stackelberg Games](http://arxiv.org/abs/2605.07240)

- HPA (Hierarchical Priority Adjustment): introduces a hierarchical reinforcement learning framework that dynamically optimizes the execution order of agents in N-level Stackelberg Games to improve multi-agent coordination.
- The framework utilizes an upper policy to select execution sequences based on the current system state, while lower-level agents execute actions within a Spatio-Temporal Sequential Markov Game (STMG) structure.
- By employing a slow-fast update scheme with shared intrinsic rewards, the method effectively coordinates learning across time scales to adapt to changing environmental conditions.

---

[HMACE: Heterogeneous Multi-Agent Collaborative Evolution for Combinatorial Optimization](http://arxiv.org/abs/2605.07214)

- HMACE: introduces a heterogeneous multi-agent framework that decomposes heuristic search into specialized roles, including Proposer, Generator, Evaluator, and Reflector, to improve combinatorial optimization.
- The framework utilizes an archive-based memory to store evaluated heuristics, enabling behavior-aware retrieval that guides the Proposer toward diverse and promising search regions.
- By integrating a lightweight deterministic filter and role-specialized collaboration, HMACE achieves a favorable quality-efficiency trade-off while minimizing LLM token consumption compared to monolithic baselines.

---

[Towards Autonomous Business Intelligence via Data-to-Insight Discovery Agent](http://arxiv.org/abs/2605.07202)

- AIDA (Autonomous Insight Discovery Agent): introduces an end-to-end framework for autonomous business analysis that leverages a proprietary DSL and reinforcement learning to transform complex enterprise data into actionable insights.
- The framework models business analysis as a Markov Decision Process, utilizing a structured state representation and a Pareto Principle-guided reward mechanism to prioritize high-leverage insights while suppressing hallucinations.
- AIDA employs specialized masking strategies and a dual-channel feedback loop to ensure logical consistency and structural integrity during multi-turn exploration in large-scale industrial data environments.

---

[Learning Agent Routing From Early Experience](http://arxiv.org/abs/2605.07180)

- BoundaryRouter: introduces a training-free routing framework that leverages Experience Memory and Rubric-guided CoT to dynamically dispatch queries between a Lightweight LLM and a Full Agent.
- The framework utilizes a Hybrid Retriever to fetch relevant behavioral examples from the Experience Memory, enabling the Routing LLM to make informed decisions without requiring ground-truth labels.
- By employing Rubric-guided CoT, the system ensures stable and consistent routing performance across diverse task distributions, effectively balancing inference latency and accuracy.

---

[HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents](http://arxiv.org/abs/2605.07177)

- HyperEyes: introduces a parallel multimodal search agent that optimizes inference efficiency by fusing visual grounding and retrieval into a single atomic action using Unified Grounded Search (UGS), TRACE, and On-Policy Distillation (OPD).
- The framework employs a Parallel-Amenable Data Synthesis pipeline and Progressive Rejection Sampling to curate high-quality, non-redundant trajectories for training LLMs.
- HyperEyes establishes a new state-of-the-art for open-source agents by Pareto-dominating existing models in both accuracy and operational efficiency across six multimodal search benchmarks.

---

[Repeated Deceptive Path Planning against Learnable Observer](http://arxiv.org/abs/2605.07174)

- DeMP (Deceptive Meta Planning): introduces a two-level optimization framework that combines episode-level adaptation and meta-level updates to enable sustained deception against a learnable observer.
- The framework utilizes meta-level updates to anticipate the observer's learning dynamics, effectively mitigating the accumulation of adaptation lag inherent in repeated adversarial interactions.
- DeMP leverages a surrogate objective that aligns with belief-induced rewards, ensuring the agent maintains deceptive performance while adapting to evolving adversarial recognition models.

---

[Rethinking Experience Utilization in Self-Evolving Language Model Agents](http://arxiv.org/abs/2605.07164)

- ExpWeaver: introduces a lightweight paradigm that interweaves experience utilization into the agent's decision-making process by exposing experience as an optional resource during reasoning.
- The framework utilizes a Reasoning (Iterative cognitive process) → Action (Environment-interactive step) loop, augmented with a Trigger Token (Special [Retrieve] signal) to activate the Retrieval Mechanism (Context-aware memory lookup) only when additional guidance is required.
- ExpWeaver enables LLMs to selectively invoke experience from the Experience Repository (External memory of past interactions) at beneficial decision points, effectively reducing reliance on rigid initialization-only or always-on usage strategies.

---

[SREGym: A Live Benchmark for AI SRE Agents with High-Fidelity Failure Scenarios](http://arxiv.org/abs/2605.07161)

- SREGym: introduces a high-fidelity, modular benchmark for evaluating AI agents in diagnosing and mitigating complex, multi-layered failures within live cloud-native production environments.
- The framework utilizes a System Environment (production-like cloud-native environment), Agent Interface (MCP servers for observability and control), Fault and Noise Injectors (orchestrators for distributed failure scenarios), Oracles (diagnosis and mitigation verification), and a Leaderboard (performance tracking) to provide a rigorous evaluation platform.
- SREGym challenges LLMs by simulating diverse failure modes, including metastable and correlated failures, while requiring agents to distinguish between actual root causes and ambient noise.

---

[MATHLIBPR: Pull Request Merge-Readiness Benchmark for Formal Mathematical Libraries](http://arxiv.org/abs/2605.07147)

- MATHLIBPR: introduces a benchmark for evaluating whether LLMs can judge the merge-readiness of build-passing pull requests in the Mathlib4 formal mathematical library.
- The framework utilizes a staged evaluation protocol that provides increasing levels of context—ranging from code-local diffs to automated diagnostics and PR intent—to assess the reviewer-like judgment capabilities of LLMs and LLM agents.
- Empirical results demonstrate that current LLMs and agents struggle to distinguish merge-ready from not-merge-ready contributions, even when provided with repository-wide context and diagnostic signals.

---

[Can You Break RLVER? Probing Adversarial Robustness of RL-Trained Empathetic Agents](http://arxiv.org/abs/2605.07138)

- AEB: introduces a benchmark for evaluating the adversarial robustness of RL-trained empathetic agents across six psychologically grounded dialogue trajectories.
- The framework utilizes an adversarial simulator to test if LLMs can address latent emotional needs when users exhibit behaviors like gaslighting, escalation, or mood reversal.
- The study reveals a dissociation where RL-trained LLMs improve emotional responsiveness and final scores without significantly enhancing observable state tracking as measured by the Emotional Consistency Score.

---

[Demystifying and Detecting Agentic Workflow Injection Vulnerabilities in GitHub Actions](http://arxiv.org/abs/2605.07135)

- TAINTAWI: introduces a static analysis framework to detect Agentic Workflow Injection (AWI) vulnerabilities in GitHub Actions by modeling data flows from untrusted event contexts to agent-facing prompts and downstream workflow sinks.
- The framework constructs an Agentic Workflow Dependency Graph (AWDG) to capture cross-boundary control and data dependencies, enabling the identification of Prompt-to-Agent (P2A) and Prompt-to-Script (P2S) injection patterns.
- TAINTAWI identifies 496 exploitable AWI vulnerabilities in real-world workflows, including 343 previously unknown zero-day cases, by performing reachability analysis against action-level and workflow-level security guards.

---

[Region4Web: Rethinking Observation Space Granularity for Web Agents](http://arxiv.org/abs/2605.07134)

- Region4Web: introduces a framework that reorganizes the AXTree into functional regions through hierarchical decomposition and semantic abstraction to provide a more informative basis for web agent state understanding.
- PageDigest: provides a web-specific inference pipeline that delivers region-level observations as a compact digest, reducing observation length while maintaining task-relevant information across steps.
- The framework improves web agent performance on the WebArena benchmark by enabling efficient, region-based page state understanding that scales across diverse backbone LLMs.

---

[RRCM: Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation](http://arxiv.org/abs/2605.07129)

- RRCM: introduces a ranking-driven retrieval-and-reasoning framework that dynamically constructs decision-relevant contexts for LLMs by selectively accessing collaborative and meta memories.
- The framework utilizes an LLM as an agent that interleaves reasoning with on-demand memory retrieval, optimized end-to-end via reinforcement learning to maximize final recommendation quality.
- By replacing static retrieval rules with an adaptive policy, RRCM effectively mitigates context-length bottlenecks and improves recommendation accuracy for both popular and long-tail items.

---

[The Position Curse: LLMs Struggle to Locate the Last Few Items in a List](http://arxiv.org/abs/2605.07127)

- Position Curse: introduces a systematic failure in LLMs where models struggle to retrieve items by position in short lists, particularly when using backward indexing.
- The paper characterizes this failure across four axes: query direction, anchor type, indexing direction, and item type, demonstrating that backward retrieval consistently lags behind forward retrieval.
- The authors propose POSBENCH and PYINDEX to evaluate and mitigate this positional deficit, showing that while fine-tuning improves performance, the underlying capability remains far from saturated.

---

[Convergence and Emergence of In-Context Reinforcement Learning with Chain of Thought](http://arxiv.org/abs/2605.07123)

- ICRL (In-Context Reinforcement Learning): introduces a theoretical framework where a single-layer linear Transformer with Chain-of-Thought generation performs iterative Temporal Difference learning updates during the forward pass.
- The framework utilizes a structured context buffer to store interaction trajectories and intermediate weight iterates, enabling the model to refine its policy evaluation performance through successive CoT steps.
- Theoretical analysis establishes that the CoT generation process converges geometrically to the Temporal Difference fixed point, with performance saturating at a statistical floor determined by the context length.

---

[RepoZero: Can LLMs Generate a Code Repository from Scratch?](http://arxiv.org/abs/2605.07122)

- ACE (Agentic Code-Test Evolution): introduces a benchmark and framework for repository-level code generation that utilizes Source Repository, Test Files Generator, Test Case Generator, Environment Construction, Filtering, Coding Agent, Testing Agent, and ACE Workflow.
- The framework reformulates repository generation as a reproduction task, requiring agents to implement functionality from scratch based on API specifications while ensuring output equivalence through automated black-box testing.
- By integrating an iterative code-test feedback loop, the ACE framework enhances the success rate of complex repository-level synthesis by leveraging deterministic ground-truth outputs from source repositories.

---

[Switchcraft: AI Model Router for Agentic Tool Calling](http://arxiv.org/abs/2605.07112)

- Switchcraft: introduces a specialized model router for agentic tool calling that utilizes a DistilBERT classifier and a cost model to select the most cost-effective LLM while maintaining correctness.
- The system employs an AST-based checker to evaluate tool call accuracy across diverse benchmarks, enabling the router to learn from realized model behavior rather than chat-based preferences.
- Switchcraft achieves near-oracle accuracy while significantly reducing inference costs by dynamically routing queries to the cheapest LLM capable of executing the required tool calls.

---

[Securing Computer-Use Agents: A Unified Architecture–Lifecycle Framework for Deployment-Grounded Reliability](http://arxiv.org/abs/2605.07110)

- CUA Architecture–Lifecycle Framework: introduces a diagnostic lens for deployment-grounded reliability by mapping architectural layers (Perception, Decision, Execution) to lifecycle stages (Creation, Deployment, Operation, Maintenance).
- The framework distinguishes between failure origin and failure manifestation, identifying how upstream choices in Creation and Deployment create latent risks that surface during Operation or Maintenance.
- The paper provides a security and privacy taxonomy that links threat surfaces to specific lifecycle stages, emphasizing that effective control placement must follow the timing of failure introduction.

---

[ARMOR: An Agentic Framework for Reaction Feasibility Prediction via Adaptive Utility-aware Multi-tool Reasoning](http://arxiv.org/abs/2605.07103)

- ARMOR: introduces an agentic framework that explicitly models tool-specific utilities, adaptively prioritizes tools, and resolves tool conflicts to improve reaction feasibility prediction.
- The framework utilizes a hierarchical tool structure, pattern-based utility assessment, and a memory-augmented reasoning mechanism to leverage complementary strengths of multiple tools.
- Experimental results demonstrate that ARMOR consistently outperforms existing tool aggregation and selection methods by effectively identifying appropriate tools for specific reaction characteristics.

---

[Decentralized Diffusion Policy Learning for Enhanced Exploration in Cooperative Multi-agent Reinforcement Learning](http://arxiv.org/abs/2605.07101)

- DDPL: introduces a decentralized MARL algorithm that replaces unimodal Gaussian policies with expressive DDPMs to enable effective exploration in high-dimensional joint action spaces.
- The framework utilizes ISSM to enable efficient online training of diffusion policies without requiring samples from the target energy-based distribution.
- DDPL provides theoretical sample-complexity and Nash equilibrium gap guarantees, demonstrating improved performance and sample efficiency across continuous-action MARL benchmarks.

---

[TeamBench: Evaluating Agent Coordination under Enforced Role Separation](http://arxiv.org/abs/2605.07073)

- TeamBench: introduces a benchmark for evaluating LLM agent coordination by enforcing strict role separation via operating system permissions rather than relying on prompt-based instructions.
- The framework utilizes a Planner, Executor, and Verifier to ensure that no single agent can simultaneously read full requirements, modify the workspace, and certify the final answer.
- Experimental results demonstrate that while teams rarely outperform single agents on average, the benchmark effectively exposes coordination costs and Verifier failure modes that pass-rate metrics typically mask.

---

[Social Theory Should Be a Structural Prior for Agentic AI: A Formal Framework for Multi-Agent Social Systems](http://arxiv.org/abs/2605.07069)

- MASS (Multi-Agent Social Systems): introduces a formal framework for modeling agentic AI systems as dynamical systems where system-level outcomes emerge from the interaction of heterogeneous Agents, Information exchange function (f), Influence dynamics function (g), and Networked interaction structure (G).
- The framework identifies four structural priors—strategic heterogeneity, network-constrained dependence, co-evolution, and distributional instability—to explain how agent interactions generate emergent social dynamics.
- The authors demonstrate the framework's utility by analyzing an LLM-based social network, MoltBook, showing that these structural priors are empirically present in multi-agent systems.

---

[2.5-D Decomposition for LLM-Based Spatial Construction](http://arxiv.org/abs/2605.07066)

- 2.5-D Decomposition: introduces a neuro-symbolic pipeline that improves spatial reasoning by restricting the LLM to 2D horizontal planning while a deterministic executor handles vertical placement.
- The architecture incorporates an Instruction Parser, Structure Analyzer, Build Planner (LLM), Plan Verifier, Spatial Executor, and Response Formatter to eliminate systematic coordinate errors.
- By removing deterministic dimensions from the LLM output space, the system achieves high structural accuracy on construction tasks and transfers effectively to edge hardware.

---

[From Assistance to Agency: Rethinking Autonomy and Control in CI/CD Pipelines](http://arxiv.org/abs/2605.07062)

- Agentic CI/CD: introduces a conceptual framework for authority transfer in software delivery, distinguishing between data-plane authority and control-plane authority to categorize agent autonomy.
- The paper characterizes current systems as operating under bounded autonomy, where LLM-based agents are confined to data-plane tasks while relying on external governance mechanisms for safety.
- The authors propose a research agenda prioritizing control-plane safety, formalization of autonomy boundaries, and the development of evaluation frameworks to address the challenges of delegating decision-making to agents.

---

[Do Joint Audio-Video Generation Models Understand Physics?](http://arxiv.org/abs/2605.07061)

- AV-Phys Bench: introduces a comprehensive benchmark for evaluating physical commonsense in joint audio-video generation models across steady-state, event-transition, and environment-transition scene categories.
- The framework utilizes an AV-Phys Agent, which integrates a multimodal LLM with deterministic audio-DSP tools to provide scalable, rubric-based judgments of physical consistency.
- Experimental results reveal a significant gap between semantic adherence and physical consistency, with scene transitions emerging as the primary difficulty frontier for current generative models.

---

[MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments](http://arxiv.org/abs/2605.07058)

- MedExAgent: introduces a POMDP-based framework for interactive clinical diagnosis that integrates patient questioning, medical exam tool-calling, and final diagnosis under a unified agent architecture.
- The framework employs a two-stage training pipeline consisting of supervised fine-tuning on Calgary-Cambridge-structured conversations followed by DAPO reinforcement learning to optimize for diagnostic accuracy, tool-call quality, and exam cost.
- MedExAgent incorporates a systematic noise model to simulate real-world clinical uncertainty, including seven patient noise types and three exam noise types, to enhance agent robustness.

---

[More Than Can Be Said: A Benchmark and Framework for Pre-Question Scientific Ideation](http://arxiv.org/abs/2605.06345)

- InciteResearch: introduces a multi-agent framework that transforms vague human research inspirations into structured, actionable proposals by decomposing the Socratic questioning chain into E, V, and N components.
- The framework utilizes a cognitive state-machine to manage the research pipeline, ensuring that each stage—Elicitation, Reframing, and Necessity Checking—serves as a logical prerequisite for the next.
- The authors also introduce TF-Bench, a benchmark designed to evaluate the capacity of LLMs to assist in the early, tacit stages of scientific ideation across four distinct scientific modes.

---

[Safactory: A Scalable Agentic Infrastructure for Training Trustworthy Autonomous Intelligence](http://arxiv.org/abs/2605.06230)

- Safactory: introduces a scalable infrastructure framework for training trustworthy autonomous agents by integrating Parallel Simulation Platform, Trustworthy Data Platform, and Autonomous Evolution Platform into a continuous closed-loop evolutionary pipeline.
- The framework utilizes a Parallel Simulation Platform for high-concurrency trajectory generation, a Trustworthy Data Platform for structured data asset management, and an Autonomous Evolution Platform for asynchronous reinforcement learning and policy distillation.
- Safactory enables systematic risk discovery and model improvement by transforming security evaluation from a one-time process into a continuous, data-driven lifecycle that includes planning-, perception- and tool use-agents.

---

[Proactive Instance Navigation with Comparative Judgment for Ambiguous User Queries](http://arxiv.org/abs/2605.06223)

- ProCompNav: introduces a two-stage framework for ambiguous instance navigation that constructs a candidate pool and identifies the target through comparative judgment using MLLM and NLI Verifier components.
- The framework replaces independent matching with a collect-then-compare pipeline, utilizing binary questions to prune candidates based on discriminative attributes.
- ProCompNav improves success rates and reduces user burden by leveraging comparative judgment to resolve target ambiguity among visually similar distractors.

---

[Entropy-Regularized Adjoint Matching for Offline Reinforcement Learning](http://arxiv.org/abs/2605.06156)

- ME-AM (Maximum Entropy Adjoint Matching): introduces a unified framework that resolves the Support-Binding Dilemma in offline RL by integrating a Reward-Guided Mixture Prior and functional Mirror Descent into the continuous flow formulation.
- The framework utilizes a Reward-Guided Mixture Prior to expand the geometric support of the generative model and employs functional Mirror Descent to flatten the empirical behavior density, thereby mitigating popularity bias.
- ME-AM achieves state-of-the-art performance on sparse-reward continuous control benchmarks by embedding score-based entropy gradients directly into the terminal boundary condition of the Adjoint Matching ODE.

---

[Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning](http://arxiv.org/abs/2605.06130)

- Skill1: introduces a unified framework that trains a single policy to co-evolve skill selection, utilization, and distillation toward a shared task-outcome objective.
- The framework decomposes the task-outcome signal into a low-frequency trend for selection and a high-frequency variation for distillation to enable simultaneous optimization of all three capabilities.
- Experiments on ALFWorld and WebShop demonstrate that Skill1 outperforms prior skill-based and RL baselines by achieving unified convergence across the entire agent lifecycle.

---

[Breaking, Stale, or Missing? Benchmarking Coding Agents on Project-Level Test Evolution](http://arxiv.org/abs/2605.06125)

- TEBench (Test Evolution Benchmark): introduces the first project-level benchmark for test evolution, requiring autonomous identification and update of tests across entire repositories following production code changes.
- The framework evaluates LLM-based agents using a three-version structure to classify test evolution into Test-Breaking, Test-Stale, and Test-Missing categories.
- Experimental results reveal a shared performance ceiling across seven configurations, highlighting that Test-Stale is the most challenging evolution type due to the lack of explicit execution failure signals.

---

[On Time, Within Budget: Constraint-Driven Online Resource Allocation for Agentic Workflows](http://arxiv.org/abs/2605.06110)

- MCPP (Monte Carlo Portfolio Planning): introduces a closed-loop planning framework that dynamically allocates models and parallel samples to subtasks to maximize the probability of completing agentic workflows within explicit budget and deadline constraints.
- The framework treats budget and time as hard success conditions, shifting the optimization objective from average efficiency to instance-specific constrained completion probability.
- By simulating downstream workflow outcomes using a portfolio of continuation policies, the planner selects actions that prioritize bottleneck subtasks while managing remaining resources.

---

[Milestone-Guided Policy Learning for Long-Horizon Language Agents](http://arxiv.org/abs/2605.06078)

- BEACON: introduces a milestone-guided policy learning framework that addresses credit misattribution and sample inefficiency in long-horizon LLM agents by partitioning trajectories at milestone boundaries, applying temporal reward shaping, and utilizing dual-scale advantage estimation.
- The framework leverages the compositional structure of long-horizon tasks to decouple credit assignment across phases, ensuring that local action quality is isolated from downstream stochasticity.
- Experimental results on ALFWorld, WebShop, and ScienceWorld demonstrate that BEACON significantly improves success rates and sample utilization compared to existing trajectory-level reinforcement learning methods.

---

[MAS-Algorithm: A Workflow for Solving Algorithmic Programming Problems with a Multi-Agent System](http://arxiv.org/abs/2605.05949)

- MAS-Algorithm: introduces a multi-agent workflow that decomposes algorithmic problem solving into modular stages, including Agent1 (predicts relevant algorithm labels), Agent2 (retrieves external algorithmic knowledge), Agent3 (generates structured solution plans), Agent4 (translates plans into executable code), and Agent5 (diagnoses errors and provides feedback).
- The framework integrates specialized agents with auxiliary components like a RAG Module (retrieves domain-specific information), Replanning Module (revises reasoning plans), Revising Module (updates generated code), Judging Tools (executes and validates code), a Format Conversion Tool (standardizes intermediate outputs), and a Sample Extraction Tool (parses problem specifications).
- Experimental results demonstrate that this multi-agent approach consistently improves the performance of LLMs on algorithmic tasks, outperforming both direct prompting and parameter-efficient fine-tuning.

---

[Active Learning for Communication Structure Optimization in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2605.05703)

- Active Learning for Communication Structure Optimization in LLM-MAS: introduces an ensemble-based information-theoretic framework that optimizes communication structures by selecting the most informative tasks for LLM-MAS training.
- The framework utilizes Representative Selection to reduce candidate pools and an EKI-based Utility Estimator to approximate Bayesian posterior updates for graph parameters without requiring gradients.
- By incorporating surrogate modeling and batch Thompson sampling, the approach efficiently identifies informative tasks to improve downstream accuracy and robustness under constrained computational budgets.

---


[Exploring a Virtual Pet to Provide Context Notifications in a Tourism Recommender System: a Pilot Study](http://arxiv.org/abs/2605.07960)

- GRS: introduces a context-aware tourism recommendation framework that utilizes a virtual pet as a social mediator to deliver personalized alerts and mitigate notification fatigue.
- The system integrates real-time IoT environmental data with a Multi-Agent Microservice to generate personalized recommendations and safety-critical notifications.
- A pilot study demonstrates that the virtual pet interface enhances hedonic quality and user acceptance by transforming technical alerts into friendly, character-mediated companion advice.

---


[REINFORCEMENT LEARNING FOR SCALABLE AND TRUSTWORTHY INTELLIGENT SYSTEMS](http://arxiv.org/abs/2605.08378)

- FedNPG-ADMM: introduces a communication-efficient synchronous federated RL framework that integrates ADMM with natural policy gradient optimization to reduce communication complexity from O(d²) to O(d).
- AFedPG: introduces an asynchronous federated RL framework that utilizes a delay-adaptive lookahead technique to mitigate stale updates and improve global time complexity in heterogeneous environments.
- MaPPO and CI-RL: introduce trustworthy RL methods, where MaPPO incorporates prior reward knowledge into preference optimization for LLMs, and CI-RL instills contextual integrity reasoning into LLM policies via RL.

---

#### 7th May 2026


[UX in the Age of AI: Rethinking Evaluation Metrics Through a Statistical Lens](http://arxiv.org/abs/2605.05600)

- ADUX-Stat (Adaptive Dynamic UX Statistical Framework): introduces a three-construct model designed to evaluate usability in AI-mediated environments by treating user experience as a probabilistic signal distribution.
- The framework integrates IEI to measure output variability, TDC to track longitudinal usability trends, and BUCS to provide Bayesian credible intervals for uncertainty quantification.
- ADUX-Stat addresses the limitations of classical UX metrics by providing a statistically grounded, field-deployable methodology for evaluating non-deterministic AI interfaces.

---


[Incentive Design in Competitive Resource Allocation: Exploiting Valuation Asymmetry in Tullock Contests](http://arxiv.org/abs/2605.07045)

- Incentive Design in Competitive Resource Allocation: introduces a mechanism for a central coordinator to influence subordinate agents in a multi-player Tullock contest by strategically manipulating their reported valuations of a contested prize.
- The framework leverages cost asymmetry among subordinates to steer the Nash equilibrium toward outcomes that maximize the coordinator's utility.
- Analytical results demonstrate that optimal valuations for subordinates are proportional to the square root of their per-unit costs, effectively reducing the coordinator's optimization problem to a two-variable system.

---


[The Causally Emergent Alignment Hypothesis: Causal Emergence Aligns with and Predicts Final Reward in Reinforcement Learning Agents](http://arxiv.org/abs/2605.06746)

- Causal Emergence Alignment Hypothesis: introduces a framework for evaluating how causal emergence in RL agents aligns with and predicts final reward performance across diverse environments.
- The study utilizes ΦID decomposition on latent-space trajectories to quantify causal emergence as a measure of agent integration and goal-directed representational reorganization.
- Experimental results demonstrate that causal emergence provides a robust, early-training directional signal that outperforms standard neural representation metrics in predicting final learning outcomes.

---


[The Context Gathering Decision Process: A POMDP Framework for Agentic Search](http://arxiv.org/abs/2605.07042)

- CGDP: introduces a formal POMDP-based framework for agentic search that replaces implicit LLM state tracking with explicit, modular infrastructure.
- The framework utilizes a persistent Belief State to bound context and a programmatic Exhaustion Gate to prevent redundant looping and premature stopping.
- Empirical validation across four agent harnesses demonstrates that these modular interventions improve multi-hop reasoning and reduce token consumption without sacrificing accuracy.

---



[Beyond Task Success: Measuring Workflow Fidelity in LLM-Based Agentic Payment Systems](http://arxiv.org/abs/2605.06457)

- HMASP (Hierarchical Multi-Agent System for Payments): introduces a trajectory-level evaluation metric, ASR, to measure workflow fidelity in LLM-based agentic payment systems by comparing observed agent execution sequences against expected paths.
- The framework utilizes a Payment Supervisor and CPA to manage payment workflows, incorporating prompt engineering and deterministic routing guards to ensure compliance with auditable procedural steps.
- Evaluation across 18 LLMs reveals that standard metrics like Task Success Rate often fail to detect systematic workflow shortcuts, whereas ASR identifies deviations in agent handoffs essential for regulatory compliance.

---

[Cognitive Agent Compilation for Explicit Problem Solver Modeling](http://arxiv.org/abs/2605.07040)

- CAC: introduces a failure-driven learning pipeline that compiles problem-solving knowledge from a Teacher LLM into an explicit, inspectable Knowledge Base for a target Cognitive Agent.
- The framework utilizes a Cognitive Agent (SLM) that performs deterministic actions based on retrieved declarative memory, while the Teacher LLM iteratively refines the Knowledge Base to ensure task success.
- This approach addresses the opacity of LLMs in educational settings by decoupling knowledge representation from reasoning, facilitating more transparent and editable learner modeling.

---

[PACEvolve++: Improving Test-time Learning for Evolutionary Search Agents](http://arxiv.org/abs/2605.07039)

- PACEvolve++: introduces an advisor-model reinforcement learning framework that decouples strategic search decisions from code implementation to enable effective test-time policy adaptation in evolutionary search agents.
- The framework utilizes a trainable Advisor LLM for hypothesis generation and selection, while delegating the realization of executable code to a stronger, frozen Frontier Implementation LLM.
- To handle non-stationary search dynamics, the system employs a phase-adaptive reinforcement learning objective that transitions credit assignment from group-relative feedback during early exploration to frontier-contribution feedback during late-stage refinement.

---

[Self Driving Datasets: From 20 Million Papers to Nuanced Biomedical Knowledge at Scale](http://arxiv.org/abs/2605.07022)

- Starling: introduces a multi-agent deep research system that autonomously converts unstructured biomedical literature into large-scale, nuanced, and accurate structured datasets.
- The system utilizes a Proposer agent, Validator agent, Investigator agent, Extractor agent, and Judge agent to perform iterative query construction, schema induction, and high-fidelity data extraction.
- Starling leverages an entity tagging model and a hybrid sparse–dense retrieval layer to efficiently process a 22.5 million paper PubMed corpus, outperforming manual curation in scale, accuracy, and contextual nuance.

---

[Problem Space Attunement in Youth Social Media Design](http://arxiv.org/abs/2605.07018)

- Problem Space Attunement in Youth Social Media Design: introduces a research framework that addresses misattunement in youth social media design through Fictional Inquiry, Asynchronous Remote Community, and an LLM-agent simulation sandbox.
- The framework utilizes ego-anchored agents to simulate social dynamics, allowing youth to evaluate design choices within familiar, grounded contexts.
- This approach shifts design research from adult-centric assumptions toward youth-articulated goals, criteria, and boundary conditions for relationally supportive social media.

---

[Dual-Agent Co-Training for Health Coaching via Implicit Adversarial Preference Optimization](http://arxiv.org/abs/2605.07011)

- DACT: introduces a dual-agent co-training framework that interactively optimizes both a health coach agent and a client simulator using implicit adversarial preference optimization.
- The framework utilizes a multi-dimensional LLM judge to construct Pareto-dominant preference pairs, enabling the coach to improve across multiple clinical dimensions while the client learns to provide increasingly challenging interactions.
- This co-evolutionary process admits a natural stochastic-game interpretation, where alternating DPO updates for the coach and client agents drive robust performance improvements and significant reduction in clinical anti-patterns.

---

[SmellBench: Evaluating LLM Agents on Architectural Code Smell Repair](http://arxiv.org/abs/2605.07001)

- SmellBench: introduces a task orchestration framework for evaluating LLM agents on architectural code smell repair, utilizing an LLM Agent, SmellBench MCP Server, Repository, Agent Skill, Trigger Prompt, GEPA Prompt Optimization Pipeline, Static Validator, Judge Model, and Task Lifecycle State Machine.
- The framework employs a GEPA pipeline to generate smell-specific execution playbooks and few-shot demonstrations, which are delivered to agents via a structured task packet.
- Evaluation of 11 agent configurations reveals that repair aggressiveness and net codebase quality are inversely related, with the best agents achieving a 47.7% resolution rate on architectural smells.

---

[Echo: KV-Cache-Free Associative Recall with Spectral Koopman Operators](http://arxiv.org/abs/2605.06997)

- Echo: introduces a KV-cache-free architecture that replaces standard attention and feedforward layers with Spectral Koopman Attention and Koopman MLP components to enable constant-memory retrieval.
- The framework utilizes Mamba-2/SSM Backbone for local processing while employing SKA to accumulate sufficient statistics in fixed-size Sufficient Statistic Accumulators, effectively eliminating the memory cliff associated with standard SSMs.
- By recasting content-addressed retrieval as a closed-form ridge regression solved via Cholesky Factorization and Power Spectral Filter, the architecture maintains high retrieval accuracy across long sequences without linear KV cache growth.

---

[Why Does Agentic Safety Fail to Generalize Across Tasks?](http://arxiv.org/abs/2605.06992)

- Agentic Safety Generalization Framework: introduces a theoretical and empirical investigation into why agentic safety fails to generalize across tasks, demonstrating that the relationship between a task and its safe execution is inherently more complex than the relationship between a task and its execution alone.
- The paper proves that the mapping from task specification to an optimal controller has a higher Lipschitz constant under safety requirements than without, establishing a formal sense in which generalization is fundamentally more difficult with safety constraints.
- Empirical experiments across linear-quadratic control, neural network-based quadcopter navigation, and LLM-based CRM agents corroborate the theoretical findings, showing that safe behavior is significantly harder to transfer to unseen tasks than standard execution behavior.

---

[The Cost of Consensus: Malignant Epistemic Herding and Adaptive Gating in Distributed Multi-Agent Search](http://arxiv.org/abs/2605.06988)

- Entropy-delta gating framework: introduces a decentralized multi-agent search architecture that mitigates malignant epistemic herding by conditioning communication on information novelty using Shannon entropy as both a transmission gate and a fusion weight.
- The framework utilizes an inverse-entropy fusion rule to dynamically weight incoming messages, ensuring that more confident agents exert greater influence on the collective belief state.
- By replacing continuous high-frequency communication with event-triggered updates, the system achieves superior epistemic alignment and task success while reducing bandwidth consumption by over 98%.

---

[Group of Skills: Group-Structured Skill Retrieval for Agent Skill Libraries](http://arxiv.org/abs/2605.06978)

- GOSKILLS (Group of Skills): introduces an inference-time retrieval method that organizes skills into role-labeled execution contexts using Skill Library, Typed Skill Graph, Group Pool, Group Graph, Inverted Index, Query Schema, Anchor Selection, Support Expansion, Bottlenecking, Execution Contract, and Coverage-Debt Accounting.
- The framework improves agent performance by replacing flat skill lists with structured, anchor-centered groups that explicitly define execution entry points, support roles, and visible requirements.
- GOSKILLS preserves visible-requirement coverage under constrained budgets and reduces agent-only runtime by minimizing the organizational burden on the LLM during task execution.

---

[Learning and Reusing Policy Decompositions for Hierarchical Generalized Planning with LLM Agents](http://arxiv.org/abs/2605.06957)

- HCL-GP (Hierarchical Component Learning for Generalized Policies): introduces a dynamic policy-learning architecture that integrates generalized planning and hierarchical task decomposition to synthesize reusable, parameterized policies for LLM agents.
- The framework utilizes a multi-agent system comprising Task Abstraction and Parameterization Agent, Component Search Agent, Policy Generator (Planning) Agent, Decomposition Agent, and Generalize and Deduplicate Agent to iteratively learn, refine, and store executable policy components.
- By extracting and generalizing procedural patterns from successful executions, the approach enables efficient cross-domain transfer and improves iteration efficiency in complex, multi-step interactive environments.

---

[Multi-Objective Constraint Inference using Inverse reinforcement learning](http://arxiv.org/abs/2605.06951)

- MOCI: introduces a framework that jointly recovers shared hard constraints and individual preferences from unlabeled, heterogeneous expert trajectories using an Expectation-Maximization approach.
- The framework utilizes Maximum Entropy Inverse Reinforcement Learning to model diverse expert behaviors by clustering trajectories into latent types while simultaneously identifying forbidden states via greedy search.
- Empirical results demonstrate that MOCI achieves superior predictive accuracy and computational efficiency compared to existing baseline methods in multi-objective GridWorld environments.

---

[Bridging the Last Mile of Circuit Design: POSTEDA-BENCH, a Hierarchical Benchmark for PPA Convergence and DRC Fixing](http://arxiv.org/abs/2605.06936)

- POSTEDA-BENCH: introduces a hierarchical benchmark for evaluating LLM-based agents on post-EDA closure, including DRC-Bench and PPA-Bench, supported by both open-source and commercial toolchains.
- The benchmark evaluates LLM-based agents across DRC-Essential, DRC-Reasoning, PPA-Mono, and PPA-Multi, identifying that trade-off reasoning is the primary bottleneck for multi-objective PPA convergence.
- Experimental results across eight LLMs demonstrate a synthetic-to-practical performance gap, where vision augmentation consistently enhances DRC-Bench performance.

---

[MAGIQ: A Post-Quantum Multi-Agentic AI Governance System with Provable Security](http://arxiv.org/abs/2605.06933)

- MAGIQ: introduces a post-quantum secure governance framework for multi-agentic AI systems that enforces user-defined communication and access control policies through efficient cryptographic primitives.
- The framework utilizes application-level communication abstractions, A-session and C-session, to provide fine-grained policy enforcement and message attribution for agent-to-agent and one-to-many interactions.
- MAGIQ provides provable security guarantees using the Universal Composability (UC) framework and demonstrates practical performance overhead compared to classical baseline systems.

---

[A2RD: Agentic Autoregressive Diffusion for Long Video Consistency](http://arxiv.org/abs/2605.06924)

- A2RD: introduces an agentic autoregressive architecture for long video synthesis that decouples creative synthesis from consistency enforcement using a Retrieve–Synthesize–Refine–Update cycle.
- The framework integrates a Multimodal Video Memory, Adaptive Segment Generation, and Hierarchical Test-Time Self-Improvement to maintain temporal consistency and narrative coherence over long horizons.
- The authors also contribute LVbench-C, a benchmark designed to stress-test long-horizon video consistency through cyclical entity and environment transitions.

---

[Same Signal, Opposite Meaning: Direction-Informed Adaptive Learning for LLM Agents](http://arxiv.org/abs/2605.06908)

- DIAL (Direction-Informed Adaptive Learning): introduces a framework that learns the utility direction of gating signals per environment and backbone to optimize test-time compute for LLM agents.
- The framework utilizes Bernoulli-ε exploration to collect counterfactual data, enabling the training of a sparse linear gate that avoids the pitfalls of fixed-direction assumptions.
- By decomposing state types into intervention-unsuitable and decision-difficult regimes, DIAL effectively calibrates compute utility across heterogeneous agent environments.

---

[Self-Programmed Execution for Language-Model Agents](http://arxiv.org/abs/2605.06898)

- SPE (Self-Programmed Execution): introduces an agent architecture where the LLM completion itself functions as the orchestrator program, removing the need for a fixed external orchestration policy.
- The framework utilizes Spell, a Lisp-based language, to enable LLMs to programmatically manage their own context, tool calls, and recursive self-calls.
- Experiments demonstrate that frontier LLMs can effectively use Spell to solve complex agentic tasks by treating orchestration as dynamic, model-generated program logic.

---

[Beyond the Black Box: Interpretability of Agentic AI Tool Use](http://arxiv.org/abs/2605.06890)

- Mechanistic-interpretability toolkit: introduces a framework for monitoring agentic tool decisions by mapping pre-action activations through Sparse Autoencoders (SAEs) to identify tool-use intent and risk levels.
- The framework utilizes a Tool-Need Probe (binary) and a Tool-Risk Probe (ternary) to provide internal observability into agent behavior before external execution occurs.
- By applying feature ablation to sparse latent representations, the approach confirms the functional necessity of specific internal features in driving agent tool-use decisions.

---

[Agentick: A Unified Benchmark for General Sequential Decision-Making Agents](http://arxiv.org/abs/2605.06869)

- Agentick: introduces a unified benchmark for evaluating RL, LLM, VLM, and hybrid agents across 37 procedurally generated tasks using a Gymnasium-compatible interface, Coding API, oracle reference policies, pre-built SFT datasets, and a composable agent harness.
- The framework utilizes a capability-decomposed, multi-modal design that provides five synchronized observation modalities to ensure fair comparison across diverse agent paradigms.
- Experimental results demonstrate that no single paradigm dominates, while harness design, specifically the use of chain-of-thought reasoning, significantly multiplies LLM performance.

---

[Multi-Objective Multi-Agent Bandits: From Learning Efficiency to Fairness Optimization](http://arxiv.org/abs/2605.06864)

- MO-MA-MAB: introduces decentralized algorithms for multi-objective multi-agent bandits that balance learning efficiency and fairness under heterogeneous rewards and time-varying communication.
- The paper develops PARETO UCB1 GOSSIP for Pareto-optimal learning and SIMULATED NSW UCB GOSSIP for optimizing Nash Social Welfare using gossip-based communication and preference-based reward simulation.
- Theoretical analysis establishes regret bounds of O(log T) for Pareto efficiency and O(T^3/4) for Nash Social Welfare, highlighting the fundamental tradeoff between fairness and convergence speed in decentralized multi-agent systems.

---

[AGWM: Affordance-Grounded World Models for Environments with Compositional Prerequisites](http://arxiv.org/abs/2605.06841)

- AGWM: introduces a world model that explicitly tracks environment affordances using a self-evolving Dynamic Affordance Graph to mitigate compounding multi-step imagination errors.
- The framework integrates an SC Classifier to detect structure-changing events and a Graph Predictor that enforces a frontier-mask constraint to ensure imagined trajectories remain within feasible action sets.
- By conditioning the RSSM World Model on the graph embedding, AGWM achieves improved generalization to novel rule configurations and reduces prediction error in environments with compositional prerequisites.

---

[Randomness is sometimes necessary for coordination](http://arxiv.org/abs/2605.06825)

- Diamond Attention: introduces a cross-attention architecture that utilizes structured random masking to induce transient rank ordering among homogeneous agents, enabling role differentiation in cooperative multi-agent reinforcement learning.
- The framework generates per-timestep asymmetric attention masks by sampling scalar random numbers, allowing agents to dynamically form hierarchies without requiring explicit communication or unique identifiers.
- By decoupling coordination from fixed agent identities and environmental cues, the architecture achieves zero-shot generalization to varying team sizes and cross-scenario transfer in adversarial environments.

---

[SHARP: A Self-Evolving Human-Auditable Rubric Policy for Financial Trading Agents](http://arxiv.org/abs/2605.06822)

- SHARP: introduces a neuro-symbolic framework that replaces unconstrained prompt mutation with structured, symbolic policy optimization to address credit assignment in LLM-based financial trading.
- The framework utilizes an Attribution Agent to diagnose failures, an Evolution Agent to perform atomic rule edits, and a Validation Gate to ensure robust, auditable strategy refinement.
- By confining LLM reasoning to a bounded, human-readable rubric, SHARP enables dynamic adaptation to non-stationary markets while maintaining the transparency required for institutional finance.

---

[Towards Security-Auditable LLM Agents: A Unified Graph Representation](http://arxiv.org/abs/2605.06812)

- Agent-BOM (Agent Bill of Materials): introduces a hierarchical attributed directed graph representation that bridges the semantic gap in LLM agent security by modeling static capability bases and dynamic runtime semantic states.
- The framework enables path-level security auditing by organizing execution traces into queryable evidence paths that connect risk entry points, semantic evolution, capability invocation, and cross-agent propagation.
- By instantiating a 4-tuple auditing rule template, the approach allows for systematic root-cause analysis and security adjudication of complex agentic risks such as memory poisoning, tool misuse, and multi-agent ecosystem hijacking.

---

[Conformal Agent Error Attribution](http://arxiv.org/abs/2605.06788)

- Conformal Agent Error Attribution: introduces a framework for identifying decisive errors in multi-agent system trajectories using conformal prediction to provide statistical coverage guarantees.
- The framework utilizes filtration-based conformal prediction algorithms to generate contiguous prediction sets, enabling efficient error localization and automated agent recovery.
- The approach is model-agnostic and demonstrates that matching the conformal algorithm to the error distribution of the data significantly improves the precision of error attribution.

---

[When Does Critique Improve AI-Assisted Theoretical Physics? SCALAR: Structured Critic–Actor Loop for Agentic Reasoning](http://arxiv.org/abs/2605.06772)

- SCALAR: introduces a pedagogical Actor–Critic–Judge pipeline to evaluate how interaction strategies between LLMs affect reasoning performance in theoretical physics.
- The framework utilizes an Actor agent for problem-solving, a Critic agent for formative feedback, and an independent Judge agent for final evaluation against reference solutions.
- SCALAR provides a controlled testbed to analyze the impact of Actor personas, Critic feedback strategies, and model scaling on the convergence of multi-turn LLM reasoning.

---

[BAMI: Training-Free Bias Mitigation in GUI Grounding](http://arxiv.org/abs/2605.06664)

- BAMI: introduces a training-free inference framework that mitigates precision and ambiguity biases in GUI grounding by utilizing MPD, Coarse-to-Fine Focus, and Candidate Selection.
- The framework employs MPD to diagnose failure sources, followed by a recursive Coarse-to-Fine Focus mechanism and a prompt-guided Candidate Selection process to improve localization accuracy.
- BAMI enhances existing GUI grounding models by incorporating structured inference and external correction models without requiring additional training.

---

[AI Co-Mathematician: Accelerating Mathematicians with Agentic AI](http://arxiv.org/abs/2605.06651)

- AI Co-Mathematician: introduces a stateful, agentic workbench designed to support mathematicians by orchestrating complex, iterative research workflows through a hierarchy of specialized agents.
- The system utilizes a project coordinator agent to manage parallel workstreams, which leverage specialized sub-agents and reviewer agents to maintain rigor and track research progress in a shared workspace.
- By integrating human-in-the-loop steering with hard programmatic constraints, the framework enables the synthesis of native mathematical artifacts while managing the inherent uncertainty of LLMs.

---

[Revisiting Adam for Streaming Reinforcement Learning](http://arxiv.org/abs/2605.06764)

- Adaptive Q(λ): introduces a streaming reinforcement learning algorithm that combines momentum-based eligibility traces with ADAM-style variance-adjusted updates and bounded error signals to achieve stable learning.
- The framework demonstrates that classical batch-RL objectives, when coupled with properly tuned ADAM hyperparameters, are highly effective in streaming environments.
- The research identifies ADAM's epsilon parameter as a critical Signal-to-Noise Ratio filter that facilitates stable convergence by regulating updates for sparse or noisy features.

---

[StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction](http://arxiv.org/abs/2605.06642)

- StraTA: introduces a framework that decomposes long-horizon agentic tasks into strategy generation and strategy-conditioned action execution to improve exploration and credit assignment.
- The framework utilizes a hierarchical GRPO-style training objective that optimizes both high-level strategies and low-level actions using Strategy Generator, Action Executor, and Hierarchical GRPO-style Rollout.
- StraTA enhances learning efficiency through Farthest Point Sampler for diverse strategy exploration and Critical Self-Judgment Agent for fine-grained step-level credit assignment.

---

[Recursive Agent Optimization](http://arxiv.org/abs/2605.06639)

- RAO: introduces a reinforcement learning approach that trains a single shared LLM policy to dynamically spawn and manage recursive sub-agents for complex task decomposition.
- The framework utilizes a local node reward mechanism, incorporating a delegation bonus and a leave-one-out baseline, to provide dense credit assignment across the execution tree.
- By optimizing a weighted multi-task objective, RAO enables agents to generalize to harder tasks, scale beyond context window limits, and improve training efficiency through self-induced curricula.

---

[Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents](http://arxiv.org/abs/2605.06635)

- Source attribution evaluation framework: introduces a three-stage pipeline that utilizes a Markdown AST parser to extract citation-claim pairs and three specialized evaluators to assess link accessibility, topical relevance, and factual accuracy.
- The framework employs an LLM-as-a-judge approach for content and fact verification, calibrated through human review to mitigate systematic biases.
- Experimental results across 14 LLMs demonstrate that while frontier models maintain high link validity, they exhibit significant factual accuracy degradation as search depth increases, indicating an information overload effect.

---

[MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](http://arxiv.org/abs/2605.06623)

- MASPO: introduces a joint prompt optimization framework for LLMs in multi-agent systems that resolves credit assignment dilemmas by integrating LLM Evaluator, Prompt Optimizer, Misalignment Buffer, Search Tree, Beam Search, Topological Scheduler, and Joint Reward Model.
- The framework employs a multi-granularity joint evaluation mechanism and misalignment-aware sampling to bridge the gap between local agent objectives and global system outcomes.
- MASPO utilizes an evolutionary beam search with a beam refresh mechanism to ensure stable co-adaptation of agents in non-stationary multi-agent environments.

---

[SkillOS: Learning Skill Curation for Self-Evolving Agents](http://arxiv.org/abs/2605.06614)

- SkillOS: introduces an experience-driven RL training recipe for learning skill curation in self-evolving agents by pairing a frozen Agent Executor with a trainable Skill Curator.
- The framework utilizes a SkillRepo to store reusable procedural knowledge in Markdown format, which is dynamically managed by the Skill Curator through insert, update, and delete operations.
- By training on grouped task streams and employing a composite reward function, the system effectively turns delayed downstream feedback into actionable learning signals for long-term agent proficiency.

---

[Patch2Vuln: Agentic Reconstruction of Vulnerabilities from Linux Distribution Binary Patches](http://arxiv.org/abs/2605.06601)

- Patch2Vuln: introduces an agentic pipeline that reconstructs security vulnerabilities from Linux distribution binary patches by orchestrating binary analysis tools and LLM-based reasoning without external advisory data.
- The framework utilizes a candidate-centric architecture where binary diffs are transformed into dossiers, enabling the LLM agent to perform preliminary audits, plan bounded validation, and generate final vulnerability reports.
- Experimental results on 25 Ubuntu package pairs demonstrate that the agent effectively localizes security-relevant functions and identifies root-cause classes while maintaining hallucination resistance against negative controls.

---

[Cross-Modal Navigation with Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.06595)

- CRONA: introduces a decentralized MARL framework for cross-modal navigation that leverages modality-specialized agents, auxiliary belief predictors, and a centralized multi-modal critic to improve collaborative navigation efficiency.
- The framework utilizes Audio Encoder, Vision Encoder, Auxiliary Belief Predictor, History Encoder, Multi-Modal Critic, Replay Buffer, Policy Head, and Value Head to enable effective coordination among heterogeneous agents without inter-agent communication during execution.
- Experimental results across Matterport3D scenes demonstrate that CRONA outperforms single-agent baselines and identifies five distinct modality-dominance patterns that dictate the effectiveness of collaborative navigation strategies.

---

[WEBLICA: Scalable and Reproducible Training Environments for Visual Web Agents](http://arxiv.org/abs/2605.06761)

- WEBLICA (Web Replica): introduces a framework for constructing reproducible and scalable web environments to train visual web agents using WEBLICA-CACHE and WEBLICA-SYNTH.
- The framework leverages HTTP-level caching to capture stable visual states and an LLM-based synthesis pipeline to generate diverse, grounded web environments for RL training.
- WEBLICA-8B outperforms open-weight baselines on web navigation benchmarks while using fewer inference steps and scaling effectively with additional test-time compute.

---

[NeuroAgent: LLM Agents for Multimodal Neuroimaging Analysis and Research](http://arxiv.org/abs/2605.06584)

- NeuroAgent: introduces a hierarchical multi-agent framework that automates the full lifecycle of multimodal neuroimaging analysis, utilizing a Central Orchestrator, Specialized Modality Agents, and a Feedback-Driven Execution Engine.
- The system employs a Global Workflow Registry for state persistence and a Human-in-the-Loop interface to ensure reliability and allow expert intervention during complex scientific workflows.
- NeuroAgent achieves high performance in automated preprocessing and downstream Alzheimer's disease classification by integrating LLM-driven reasoning with established neuroimaging toolchains.

---

[Language Models Can Autonomously Hack and Self-Replicate](http://arxiv.org/abs/2605.06760)

- Language Models Can Autonomously Hack and Self-Replicate: demonstrates that LLMs can autonomously exploit web vulnerabilities, extract credentials, and replicate their full inference stack across a network of hosts.
- The research evaluates autonomous self-replication by decomposing the process into propensity and capability, specifically measuring the ability of LLMs to perform hacking and copying tasks.
- The study validates chain replication across multiple hops, showing that replicas can independently discover and exploit new targets without human intervention.

---

[Coordination Matters: Evaluation of Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.06557)

- STAT: introduces a coordination-aware evaluation perspective for cooperative MARL by utilizing a controlled commitment-constrained spatial task-allocation testbed to expose process-level diagnostics beyond aggregate return.
- The framework employs STAT to systematically scale agents, tasks, and environment size, isolating assignment coordination as the primary bottleneck while holding task rules and observation access fixed.
- Empirical results demonstrate that aggregate return often obscures distinct coordination failure modes, necessitating process-level metrics like conflict rate, assignment diversity, and throughput for robust benchmarking.

---

[ROSE: Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL](http://arxiv.org/abs/2605.06534)

- ROSE: introduces a cooperative, resource-elastic post-training system that harvests idle compute and memory on serving GPUs to expedite agentic RL rollouts while preserving serving SLOs.
- The system utilizes an SLO-safe co-serving executor, a cross-cluster weight transfer engine, and an elastic rollout scheduler to optimize training throughput in environments with bursty serving traffic.
- ROSE leverages weight-differential sparsity and shard-aware routing to minimize cross-datacenter communication overhead, achieving significant end-to-end throughput improvements over existing resource-fixed and elastic baselines.

---

[Market-Alignment Risk in Pricing Agents: Trace Diagnostics and Trace-Prior RL under Hidden Competitor State](http://arxiv.org/abs/2605.06529)

- Trace-Prior RL: introduces a diagnostic and repair framework for agentic systems where partial observability causes deterministic policies to collapse uncertainty into suboptimal shortcuts.
- The framework utilizes a learned distributional market prior to regularize a stochastic pricing policy via a KL penalty, ensuring the agent maintains market-aligned behavior despite hidden competitor states.
- This approach demonstrates that when scalar rewards are easily gamed, trace-level diagnostics are essential to verify that agents have learned the intended behavioral discipline rather than just optimizing for proxy metrics.

---

[STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?](http://arxiv.org/abs/2605.06527)

- STALE (State Tracking And Latent Evaluation): introduces a benchmark for assessing long-term memory in LLM agents under implicit conflict, utilizing a three-dimensional probing framework consisting of State Resolution, Premise Resistance, and Implicit Policy Adaptation.
- The paper identifies implicit conflict as a critical failure mode where new observations invalidate earlier memories without explicit negation, requiring contextual inference and commonsense reasoning.
- The authors propose CUPMEM, a prototype that improves memory reliability by implementing write-side state adjudication and propagation-aware search to ensure downstream behavior is grounded in the current user state.

---

[Sustaining Cooperation in Populations Guided by AI: A Folk Theorem for LLMs](http://arxiv.org/abs/2605.06525)

- LLM-mediated meta-game framework: introduces a formal model where multiple LLMs act as strategic entities guiding populations of clients in repeated games to sustain cooperative outcomes.
- The framework establishes a folk theorem for LLMs, proving that any feasible and individually rational outcome can be sustained as an ε-equilibrium despite indirect observation and attribution challenges.
- The research demonstrates that shared LLM guidance functions as a powerful coordination mechanism, capable of reshaping strategic interactions and facilitating cooperation even when underlying incentives are misaligned.

---

[Process Matters more than Output for Distinguishing Humans from Machines](http://arxiv.org/abs/2605.06524)

- COGCAPTCHA30: introduces a cognitive task battery to evaluate human-machine discrimination by analyzing latent process-level behavioral features rather than task performance alone.
- The study demonstrates that while LLMs can achieve output parity with humans, they exhibit distinct process-level signatures that allow for reliable discrimination using a Random Forest classifier.
- Explicit process-level fine-tuning (P-SFT) improves human-like behavioral mimicry when task-specific process representations are known, but this advantage diminishes significantly under cross-task transfer.

---

[Agentic AIs Are the Missing Paradigm for Out-of-Distribution Generalization in Foundation Models](http://arxiv.org/abs/2605.06522)

- Agentic OOD Framework: introduces a system-level paradigm for foundation models that addresses out-of-distribution generalization through a closed-loop cycle of Perceive, Reason, Act, and Verify, which complements traditional model-centric approaches.
- The framework defines agentic systems by their ability to perform diagnostic perception, select heterogeneous response strategies, invoke external resources, and implement closed-loop verification to handle open-ended distribution shifts.
- By extending the reachable set beyond the parameter coverage ceiling, this paradigm enables LLMs to resolve knowledge gaps, capability limitations, and compositional shifts that cannot be solved by model adjustment alone.

---

[Optimizing Social Utility in Sequential Experiments](http://arxiv.org/abs/2605.06520)

- Belief Markov Decision Process: introduces a statistical protocol for sequential randomized controlled trials where a regulator provides partial subsidies to a developer to maximize social utility.
- The framework models the regulatory approval process as a principal-agent game, utilizing belief-based dynamic programming to determine optimal experimental policies and subsidies.
- The protocol employs anytime-valid e-values to maintain rigorous false positive rate guarantees throughout the sequential experimentation process.

---

[Instrumental Choices: Measuring the Propensity of LLM Agents to Pursue Instrumental Behaviors](http://arxiv.org/abs/2605.06490)

- Instrumental Choices Benchmark: introduces a realistic, low-nudge evaluation suite for measuring the propensity of LLM agents to pursue instrumentally convergent behaviors in terminal-based environments.
- The framework utilizes a ReAct-style agent operating within a seeded sandbox, employing deterministic environment-state scorers to identify policy-violating shortcuts across seven operational tasks.
- Experimental results indicate that while IC behavior is rare (5.1% overall), it is systematically structured by environmental factors, particularly when official paths are blocked, rather than being uniform noise.

---

[Autonomous Adversary: Red-Teaming in the age of LLM](http://arxiv.org/abs/2605.06486)

- LMA Framework: introduces a systematic approach for evaluating LLM-driven offensive cyber agents by decomposing intrusion scenarios into ordered task chains with explicit validation predicates.
- The framework utilizes an LMA-1 (Orchestrator) for planning, a Cyber Agent for execution, and an LMA-2 (Judge) to provide deterministic outcome verification through iterative feedback loops.
- Benchmarking across three operational modalities reveals that expert-defined plans improve task-completion rates, while fully autonomous modes often suffer from reliability bottlenecks and loss-of-control behaviors.

---

[SCARFBENCH: A Benchmark for Cross-Framework Application Migration in Enterprise Java](http://arxiv.org/abs/2605.06754)

- SCARFBENCH: introduces a benchmark for behavior-preserving cross-framework refactoring of enterprise Java applications, utilizing Coding Agents, Containerized Evaluation Harness, Behavioral Oracle, Failure Taxonomy, and Skill-based Prompting.
- The benchmark evaluates LLMs on their ability to perform complex structural transformations across Spring, Jakarta EE, and Quarkus frameworks while maintaining functional equivalence.
- Experimental results demonstrate that current LLMs struggle with cross-framework migration, showing significant performance asymmetries and frequent failures in build, deployment, and behavioral validation stages.

---

[Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management](http://arxiv.org/abs/2605.06472)

- PBKV (Prediction-Based KV-Cache Management): introduces a system for dynamic agent workflows that optimizes KV-Cache usage by predicting future agent invocations to inform hierarchical eviction and conservative prefetching decisions.
- The framework utilizes a GraphSAGE-based predictor to fuse global call graph topology, workflow history, and LLM prefill semantics to generate multi-step forecasts, which drive cache management policies to minimize re-prefill costs.
- PBKV incorporates deterministic guardrails within a probabilistic system to ensure graceful performance degradation under prediction errors, consistently outperforming LRU and static baseline approaches in cache hit rates and end-to-end latency.

---

[To What Extent Does Agent-generated Code Require Maintenance? An Empirical Study](http://arxiv.org/abs/2605.06464)

- AIDev (AI-generated code maintenance study framework): introduces an empirical methodology to quantify the long-term maintenance requirements of code produced by autonomous coding agents compared to human-authored code.
- The framework utilizes the AIDev Dataset, Repository Filter, Random Sampler, Commit Extractor, Conventional Commits Classification System (CCS), and Maintenance Analyzer to track maintenance frequency, magnitude, and authorship over a six-month observation period.
- The study reveals that AI-generated files require less frequent and smaller-magnitude maintenance than human-authored code, with human developers performing the vast majority of ongoing maintenance tasks.

---

[PrefixGuard: From Large Language Model Agent Traces to Online Failure-Warning Monitors](http://arxiv.org/abs/2605.06455)

- PrefixGuard: introduces a modular neural-symbolic framework for synthesizing online failure-warning monitors from raw LLM agent traces without deployment-time LLM inference.
- The framework utilizes StepView to convert heterogeneous raw traces into typed canonical records, which are then processed by a differentiable event abstraction layer and a monitor backend to produce calibrated risk scores.
- PrefixGuard includes diagnostic tools such as an observability ceiling and post-hoc DFA extraction to evaluate the boundary between ranking quality and actionable deployment utility.

---

[Counterexamples to EFX for Submodular and Subadditive Valuations](http://arxiv.org/abs/2605.06451)

- EFX Counterexample Framework: introduces a compact, human-verifiable construction using cyclic symmetry to demonstrate the non-existence of EFX allocations in structured valuation classes.
- The framework utilizes an Ordinal Rank Function to establish combinatorial obstructions, which are then realized through specific Cardinal Valuation Functions for subadditive and submodular settings.
- By employing a Cyclic Relabeling Mechanism, the research provides an optimal approximation barrier for EFX under monotone subadditive valuations and identifies a failure of EFX in weighted coverage functions.

---

[Constraint decay: The Fragility of LLM Agents in Backend Code Generation](http://arxiv.org/abs/2605.06445)

- Constraint decay framework: introduces a systematic evaluation methodology to measure how LLM agents struggle with structural constraints in multi-file backend code generation.
- The study utilizes an evaluation pipeline comprising LLM agents, Docker-based isolation, and static verifiers to assess performance across varying levels of architectural, database, and ORM constraints.
- Empirical results demonstrate that LLM agents exhibit significant performance degradation as structural requirements accumulate, with data-layer defects identified as the primary root cause of failures.

---

[SCRuB: Social Concept Reasoning under Rubric-Based Evaluation](http://arxiv.org/abs/2605.06444)

- SCRuB: introduces a framework for evaluating social concept reasoning in LLMs by shifting from accuracy-based benchmarks to rubric-based comparative judgment of open-ended responses.
- The framework utilizes Dataset Construction, Response Generation, and Rubric-Based Evaluation to assess reasoning depth across five critical thinking dimensions: conceptual clarity, evidential grounding, contextual relevance, pluralistic engagement, and argumentative soundness.
- To ensure robust evaluation, the methodology incorporates a Panel of Disciplinary Perspectives, which acts as an automated LLM-Judge Ensemble to approximate expert human judgment and mitigate individual bias.

---

[AgenticPrecoding: LLM-Empowered Multi-Agent System for Precoding Optimization](http://arxiv.org/abs/2605.06443)

- AgenticPrecoding: introduces a multi-agent framework that automates end-to-end precoding derivation by decomposing the process into problem formulation, solver selection, prompt upsampling, and code generation stages.
- The framework utilizes two domain-adapted LoRA-tuned agents for specialized reasoning and two general-purpose LLMs for implementation, supported by a feedback-driven refinement mechanism to ensure code executability and solution feasibility.
- Experimental results across 10 representative wireless scenarios demonstrate that the framework achieves superior cross-scenario adaptability and consistent performance compared to conventional optimization-based baselines.

---

[Knowledge Graphs, the Missing Link in Agentic AI-based Formal Verification](http://arxiv.org/abs/2605.06434)

- Knowledge Graph-based Agentic Formal Verification Framework: introduces a verification-centric methodology that constructs a Knowledge Graph from structured Intermediate Representations to provide design-grounded context for LLM-based assertion synthesis and iterative refinement.
- The framework utilizes a multi-agent workflow that includes property generation-, syntax correction-, counterexample correction-, and coverage improvement-agents to automate the formal verification process.
- By maintaining persistent, queryable links between requirements, RTL structure, and formal tool feedback, the system enables traceable, closed-loop refinement of SystemVerilog Assertions.

---

[MiA-Signature: Approximating Global Activation for Long-Context Understanding](http://arxiv.org/abs/2605.06416)

- MiA-Signature: introduces a framework that models memory access as a two-stage process of global activation followed by compact representation to improve long-context understanding in LLMs.
- The framework utilizes a Mindscape (global semantic memory space) and constructs a MiA-Signature (compact query-conditioned global state) via submodular selection to guide downstream retrieval and generation.
- MiA-Signature integrates into RAG and agentic systems, employing a Mindscape-aware retriever and an iterative update model to maintain an evolving global state alongside local evidence memory.

---

[Constraining Host-Level Abuse in Self-Hosted Computer-Use Agents via TEE-Backed Isolation](http://arxiv.org/abs/2605.06393)

- SHCUA protection framework: introduces an operation-centric security model that mitigates host-level abuse by isolating security-critical classification and authorization within a TEE-backed trusted operation plane.
- The architecture employs a risk-driven minimal-confinement strategy, ensuring that only security-sensitive operations are elevated to the trusted plane while ordinary tasks remain in the constrained REE.
- The system provides verifiable audit evidence by binding operation requests, authorization decisions, and execution results within a trusted path, preventing manipulation by a compromised host environment.

---

[Automated Safety Is Harder Than You Think](http://arxiv.org/abs/2605.06390)

- AARP: introduces a framework for automating alignment research that risks producing catastrophically misleading safety assessments due to undetected systematic errors in hard-to-supervise fuzzy tasks.
- The framework highlights that automated alignment research is prone to output-level failures and aggregation-level failures, where correlated uncertainties between research outputs lead to overconfident safety assessments.
- The paper argues that because alignment lacks safe feedback loops, agents must be trained to reliably perform hard-to-supervise fuzzy tasks using generalisation or scalable oversight, both of which currently face significant open challenges.

---

[Independent Learning of Nash Equilibria in Partially Observable Markov Potential Games with Decoupled Dynamics](http://arxiv.org/abs/2605.06377)

- Independent Learning of Nash Equilibria in Partially Observable Markov Potential Games with Decoupled Dynamics: introduces a communication-free independent learning algorithm for POMGs with decoupled dynamics and potential structure, leveraging a superstate Markov game representation to achieve quasi-polynomial complexity.
- The framework utilizes a finite-window approximation based on the filter stability assumption to transform the partially observable problem into a tractable superstate Markov game.
- The proposed approach avoids the curse of multi-agency by exploiting the decoupled transition structure and potential properties, ensuring convergence to an approximate Nash equilibrium without centralized coordination.

---

[From Agent Loops to Deterministic Graphs: Execution Lineage for Reproducible AI-Native Work](http://arxiv.org/abs/2605.06365)

- Execution Lineage: introduces a DAG-based computational model that replaces implicit agent loops with explicit, dependency-aware artifact production to ensure reproducibility and maintainable state across revisions.
- The framework utilizes Input/Edit Event, Source Context Artifact, Analysis Artifact, Criteria Decision Artifact, and Final Artifact to maintain strict boundaries and enable selective recomputation.
- By externalizing work structure into a directed acyclic graph, the system achieves precise dependency isolation and artifact-level propagation, outperforming traditional loop-based LLM agents in maintained-state quality.

---

[Prediction and Empowerment: A Theory of Agency through Bridge Interfaces](http://arxiv.org/abs/2605.06346)

- BGP (Bridge-Gap Pursuit): introduces a decision-theoretic framework for agency that models sensing and actuation as bridge interfaces to resolve failures in prediction, compression, and empowerment.
- The framework utilizes a bridge potential function to decompose and minimize bridge gaps, ensuring that agent objectives align with task-relevant identification and control rather than distractor optimization.
- BGP includes a counterfactual relevance gate and channel-reachability estimators to distinguish between genuine task-relevant controllability and mere overwrite or distractor control.

---

[MANTRA: Synthesizing SMT-Validated Compliance Benchmarks for Tool-Using LLM Agents](http://arxiv.org/abs/2605.06334)

- MANTRA: introduces a framework for automatically synthesizing machine-checkable compliance benchmarks for tool-using LLM agents by converting natural-language procedural manuals into formally validated test cases.
- The framework utilizes a two-artifact approach, generating both a symbolic world model and trace-level compliance checks, which are cross-validated using SMT solvers to ensure consistency and minimize human intervention.
- MANTRA enables deterministic evaluation of agent tool-call trajectories, allowing for fine-grained debugging of failure modes such as missing required calls or incorrect ordering in complex, long-horizon procedural workflows.

---

[Improving the Efficiency of Language Agent Teams with Adaptive Task Graphs](http://arxiv.org/abs/2605.06320)

- LATTE (Language Agent Teams for Task Evolution): introduces a formal orchestration framework for LLM teams that utilizes a shared, dynamic coordination graph to manage task dependencies, agent assignments, and progress.
- The framework employs a hybrid centralized-decentralized model where a Lead agent maintains global graph consistency while Workers autonomously propose updates and claim tasks to maximize parallel execution.
- By externalizing coordination through explicit graph mutation operators, the system reduces inter-agent conflicts, communication overhead, and idle computation compared to static or fully decentralized LLM team architectures.

---

[From Specification to Deployment: Empirical Evidence from a W3C VC + DID Trust Infrastructure for Autonomous Agents](http://arxiv.org/abs/2605.06738)

- MolTrust: introduces a production-deployed trust infrastructure for autonomous agents using W3C Verifiable Credentials and Decentralized Identifiers to provide identity, authorization, and behavioral accountability.
- The system architecture utilizes an Agent Authorization Envelope (AAE) enforced across three layers: cryptographic signatures, API-level trust management, and kernel-level syscall monitoring via Falco eBPF.
- MolTrust provides empirical evidence of a cross-organizational trust layer that addresses agent-to-agent security gaps identified by regulators and industry frameworks.

---

[Data Language Models: A New Foundation Model Class for Tabular Data](http://arxiv.org/abs/2605.06290)

- DLM (Data Language Model): introduces a foundation model class for tabular data that natively ingests raw tables to perform domain identification and prediction without preprocessing pipelines.
- Schema-1 utilizes four input pathways—column semantics, distributional summaries, raw cell values, and missing value structure—to derive a unified representation of tabular data.
- The framework incorporates retention and adaptive memory components to enable sequential fine-tuning across distinct business domains without catastrophic forgetting or data replay.

---

[Profiling for Pennies: Unveiling the Privacy Iceberg of LLM Agents](http://arxiv.org/abs/2605.06232)

- IcebergExplorer: introduces an autonomous auditing agent that simulates an informed adversary to reconstruct high-fidelity personal profiles from disparate public data sources using Search Engine, Crawler, LLM-as-a-judge, Visual LLM, Knowledge Base, and Refinement Loop.
- The framework categorizes privacy risks into a three-tier PrivacyIceberg model consisting of Direct Identifiable Information, Contextually Inferred Information, and Aggregated Mosaic Information.
- Extensive experiments demonstrate that the system achieves high-fidelity profiling at low cost, revealing that current LLM guardrails are ineffective against multi-stage synthesis attacks.

---

[A Versatile AI Agent for Rare Disease Diagnosis and Risk Gene Prioritization](http://arxiv.org/abs/2605.06226)

- Hygieia: introduces a multi-modal agentic system that integrates phenotypic, genomic, and clinical data to perform rare disease diagnosis and risk gene prioritization through a multi-stage workflow.
- The framework utilizes a router-based architecture to tailor diagnostic pipelines for common versus rare diseases, incorporating a self-verification mechanism to mitigate LLM-based hallucinations and ensure output consistency.
- Hygieia provides transparent, multi-step reasoning trajectories and confidence scores, demonstrating superior diagnostic accuracy and clinical utility compared to human experts and standard LLM baselines.

---

[ClawGuard: Out-of-Band Detection of LLM Agent Workflow Hijacking via EM Side Channel](http://arxiv.org/abs/2605.06205)

- ClawGuard: introduces a passive, out-of-band integrity monitor that uses electromagnetic emanations to detect LLM agent workflow hijacking by bypassing compromised host-internal telemetry.
- The system utilizes a dual-band SDR setup and a drift-aware coarse-fine windowing pipeline to translate macroscopic hardware EM envelopes into discrete skill-level and attack-state evidence.
- ClawGuard provides a hardware-rooted trust anchor that validates agent execution against an out-of-band policy, achieving high detection efficacy while remaining resilient to full host OS compromise.

---

[A Self-Healing Framework for Reliable LLM-Based Autonomous Agents](http://arxiv.org/abs/2605.06737)

- Reliability-aware self-healing framework: introduces a unified architecture for LLM-based autonomous agents that integrates Agent Execution Engine, Failure Detection Module, Reliability Evaluation Module, Decision Controller, Self-Healing Module, and Re-execution Engine to manage runtime failures.
- The framework utilizes a quantitative reliability model based on output consistency, semantic correctness, and execution success rate to trigger adaptive recovery strategies like re-planning, prompt correction, and tool re-selection.
- By establishing a closed-loop system for monitoring and adaptation, the approach significantly improves task success rates and system robustness in multi-agent workflow environments.

---

[Bandit Learning in General Open Multi-agent Systems](http://arxiv.org/abs/2605.06202)

- General Open MA-MAB: introduces a unified framework for bandit learning in open multi-agent systems with heterogeneous rewards and dynamic populations, utilizing CERTIFIEDARRIVALTRANSFER, GLOBALVALUEAGGREGATION, and LOCALUPDATEANDBROADCAST to minimize global dynamic regret.
- The framework employs a global-UCB index that separates continuing-agent statistical uncertainty, communication error, and certified arrival uncertainty to handle endogenous non-stationarity caused by agent churn.
- Theoretical analysis establishes regret bounds governed by pre-training quality and agent patterns, identifying a stable-arm regime where regret is limited by burn-in time rather than cumulative entry uncertainty.

---

[A2TGPO: Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping](http://arxiv.org/abs/2605.06200)

- A2TGPO: introduces a reinforcement learning framework for agentic LLMs that performs fine-grained process credit assignment using turn-group normalized information gain.
- The framework utilizes Turn-Group Normalization, Variance-rescaled Discounted Accumulation, and IG-based Adaptive Turn-level Clipping to stabilize training and improve tool-use performance.
- A2TGPO aligns optimization granularity with the multi-turn interaction structure of agentic LLMs, consistently outperforming prior RL methods on single-hop and multi-hop QA benchmarks.

---

[BioMedArena: An Open-source Toolkit for Building and Evaluating Biomedical Deep Research Agents](http://arxiv.org/abs/2605.06177)

- BioMedArena: introduces an open-source toolkit that decouples six layers of biomedical agent evaluation to eliminate per-paper engineering overhead and enable fair, head-to-head comparisons of LLMs.
- The framework includes the MUTUAL-EVOLVE harness, which utilizes a Global Workspace and parallel solvers to share intermediate findings and improve deep-research accuracy.
- BioMedArena registers 147 benchmarks, 75 typed tools, and multiple context-management strategies, achieving state-of-the-art performance across 12 backbones on 8 representative biomedical benchmarks.

---

[Beyond Accuracy: Policy Invariance as a Reliability Test for LLM Safety Judges](http://arxiv.org/abs/2605.06161)

- Policy Invariance Framework: introduces a three-principle stress test to evaluate whether LLM safety judges maintain consistent verdicts under semantically equivalent policy rewrites and directional normative shifts.
- The framework utilizes a transformation taxonomy to isolate policy wording as a variable, measuring judge reliability through the Policy Invariance Score and a standardized Judge Card reporting protocol.
- Experimental results across four LLMs demonstrate that current safety judges often conflate structural policy changes with normative shifts, revealing significant reliability gaps invisible to standard accuracy-based benchmarks.

---

[Stateful Agent Backdoor](http://arxiv.org/abs/2605.06158)

- Stateful Agent Backdoor: introduces a cross-session attack framework that models malicious agent behavior as a Mealy machine to maintain state across isolated sessions using persistent memory and tool-use capabilities.
- The framework decomposes complex multi-session attacks into independent sub-backdoors, enabling autonomous, incremental execution triggered by a single initial injection.
- Experimental results across four LLMs demonstrate high attack success rates and validate the effectiveness of the decomposition-based construction against mainstream agent frameworks.

---

[BUILD-AND-FIND: An Effort-Aware Protocol for Evaluating Agent-Managed Codebases](http://arxiv.org/abs/2605.06136)

- BUILD-AND-FIND: introduces a protocol for evaluating how effectively LLM-generated codebases communicate intended design choices to downstream agents by measuring recovery accuracy and inspection effort.
- The framework utilizes a builder-finder protocol where a builder agent constructs a repository from a hidden specification, and a finder agent attempts to recover design intent through a specification-traced question bank.
- Inspection effort serves as an agent-facing proxy for artifact legibility, interpreted only after passing recovery and stability gates to ensure the codebase is both correct and consistently understandable.

---

[MemReranker: Reasoning-Aware Reranking for Agent Memory Retrieval](http://arxiv.org/abs/2605.06132)

- MemReranker: introduces a reasoning-aware reranking model family that distills LLM-level capabilities into compact 0.6B/4B architectures to improve agent memory retrieval through calibrated scoring and instruction-aware design.
- The framework utilizes a multi-stage distillation pipeline, incorporating BCE pointwise training and InfoNCE contrastive fine-tuning, to address score calibration failures and reasoning deficits in conventional rerankers.
- By integrating Elo/Bradley-Terry calibrated scoring and specialized multi-turn dialogue data, the model achieves performance parity with large-parameter models while maintaining significantly lower inference latency for production agent systems.

---

[When Routine Chats Turn Toxic: Unintended Long-Term State Poisoning in Personalized Agents](http://arxiv.org/abs/2605.06731)

- StateGuard: introduces a post-execution defense mechanism that audits long-term state modifications at the writeback boundary to mitigate unintended long-term state poisoning in personalized LLM agents.
- The paper formalizes unintended long-term state poisoning, where routine user interactions cumulatively reshape an agent's persistent state, leading to insecure behavioral defaults.
- The authors propose the ULSPB benchmark and the Harm Score (HS) metric to systematically quantify and evaluate security-relevant behavioral drift across personalized agent frameworks.

---

[VibeServe: Can AI Agents Build Bespoke LLM Serving Systems?](http://arxiv.org/abs/2605.06068)

- VibeServe: introduces a multi-agent system that automatically synthesizes bespoke LLM serving stacks by utilizing an Outer Loop (plans search over designs) and an Inner Loop (implements and validates candidates) to optimize performance for specific model, hardware, and workload targets.
- The framework employs an Implementer Agent (writes serving-system code), an Accuracy Judge Agent (verifies correctness), and a Performance Evaluator Agent (profiles system performance) to iteratively refine serving systems within an Execution Environment (provides isolated workspace) guided by a Skills Library (contains serving-systems knowledge).
- VibeServe leverages a Search Policy (manages optimization strategy) and Search State (tracks git-based history) to enable generation-time specialization, achieving performance parity with optimized baselines in standard settings and significant speedups in non-standard deployment scenarios.

---

[Causal Reinforcement Learning for Complex Card Games: A Magic The Gathering Benchmark](http://arxiv.org/abs/2605.06066)

- MTG-Causal-RL: introduces a Gymnasium benchmark for causal RL research using Magic: The Gathering, featuring a hand-designed Structural Causal Model (SCM) and a reference agent, CGFA-PPO (Causal Graph-Factored Advantage PPO), which utilizes per-factor critic heads, a state-conditional residual gate, and an intervention-calibration loss.
- The framework enables causal credit assignment and policy auditability by decomposing win probability into strategic causal factors such as mana, board pressure, and card advantage.
- Experimental results demonstrate that while CGFA-PPO is competitive, the benchmark effectively exposes diagnostic structure and performance variations across different deck archetypes that scalar win rates alone cannot capture.

---

[PersonaGesture: Single-Reference Co-Speech Gesture Personalization for Unseen Speakers](http://arxiv.org/abs/2605.06064)

- PersonaGesture: introduces a diffusion-based pipeline for single-reference co-speech gesture personalization that utilizes Style Perceiver, Adaptive Style Infusion, and Implicit Distribution Rectification to synthesize identity-consistent motion for unseen speakers without test-time optimization.
- The framework employs Adaptive Style Infusion to inject speaker-memory tokens into the denoising process and Implicit Distribution Rectification to apply length-aware latent moment correction after generation.
- By separating temporal style control during denoising from conservative post-generation statistical correction, the model effectively preserves speaker-specific pose choices while maintaining alignment with new target speech.

---

[Multiagent Stochastic Shortest Path Problem](http://arxiv.org/abs/2605.06056)

- MSSP (Multiagent Stochastic Shortest Path) introduces: "a framework for minimizing the expected time for multiple agents to reach a target state in a known MDP, analyzing both autonomous and coordinated settings with COORHIT and AUTOHIT."
- COORHIT provides optimal coordinated strategies using linear programming, while AUTOHIT employs differentiable programming to synthesize high-quality autonomous strategy profiles.
- The research demonstrates that while coordinated MSSP is PSPACE-hard, AUTOHIT effectively overcomes complexity barriers in the autonomous setting by optimizing randomized memoryless profiles.

---

[Semantic State Abstraction Interfaces for LLM-Augmented Portfolio Decisions: Multi-Axis News Decomposition and RL Diagnostics](http://arxiv.org/abs/2605.06730)

- SSAI (Semantic State Abstraction Interfaces): introduces a methodological template for mapping sparse unstructured text into auditable, named coordinates to separate representation hypotheses from optimization variance in sequential decision systems.
- The framework utilizes a frozen LLM to elicit four integer-valued semantic axes—sentiment, risk, confidence, and volatility forecast—which serve as a shared, auditable observation interface for diverse estimators.
- By fixing the semantic representation across different estimators, the framework enables diagnostic evaluation to isolate the impact of algorithm choice from the quality of the state representation in LLM-augmented decision-making.

---

[Multi-agent decision making: A Blackwell’s informativeness approach](http://arxiv.org/abs/2605.06028)

- MA-PoP: introduces a principled decision-making framework that utilizes Blackwell’s informativeness to aggregate LLM agent beliefs through a product-of-posteriors estimator.
- The framework employs an Agent-Posterior-Estimator with an NLI-Cross-Encoder and a Deep-Sets-Calibration-Module to derive robust, permutation-equivariant probability distributions from LLM responses.
- By approximating the Bayesian pooled posterior, the approach provides a computationally efficient and theoretically grounded alternative to iterative multi-agent debate and simple voting methods.

---

[Strat-LLM: Stratified Strategy Alignment for LLM-based Stock Trading with Real-time Multi-Source Signals](http://arxiv.org/abs/2605.06024)

- Strat-LLM: introduces a framework for auditing strategic alignment in LLMs by integrating Multi-Source Data Integration Module, Stratified Strategy Scaffolding Engine, and Dynamic Execution & Evaluation Engine to evaluate trading performance under varying autonomy modes.
- The framework utilizes an Expert Strategy Taxonomy to enforce constraints on the LLM Agent, enabling a comparative analysis of reasoning-heavy versus standard LLMs across different market regimes.
- By employing a T+1 rolling-window mechanism, the system eliminates look-ahead bias and identifies an "Alignment Tax" where rigid rule enforcement impacts performance differently based on model scale and architecture.

---

[PersonaKit (PK): A Plug-and-Play Platform for User Testing Diverse Roles in Full-Duplex Dialogue](http://arxiv.org/abs/2605.06007)

- PK (PersonaKit): introduces an open-source, low-latency platform for prototyping and evaluating persona-conditioned turn-taking behaviors in full-duplex dialogue systems.
- The framework utilizes JSON-based configurations to manage interruption strategies—such as Yield, Resume, Bridge, or Override—allowing researchers to define distinct sociolinguistic personas for LLMs.
- PK integrates an end-to-end workflow that includes real-time audio processing, intent classification, and automated data collection to facilitate comparative user studies of conversational agents.

---

[A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance](http://arxiv.org/abs/2605.05991)

- Case-driven multi-agent framework: introduces a closed-loop ecosystem that replaces human roles with autonomous agents to automate the pipeline from bad-case identification to model resolution.
- The framework utilizes an Annotator Agent, an Optimizer Agent, and a User Agent to perform standard-grounded labeling, data-centric repair, and dialectical bad-case discovery.
- Supporting harness-engineering extensions include an All-In-One Relevance Model, Global Memory for cross-agent synchronization, and a Deep Search Agent to mitigate recall failures.

---

[BIORESEARCHER: Scenario-Guided Multi-Agent for Translational Medicine](http://arxiv.org/abs/2605.05985)

- BIORESEARCHER: introduces a scenario-guided multi-agent system that automates translational medicine workflows by mapping queries to research playbooks, delegating tasks to specialized subagents, and reconciling evidence into auditable dossiers.
- The architecture utilizes a Master Orchestrator to manage subagents, including a Translation Subagent for entity grounding, Retrieval Subagents for data acquisition, and an Insights & Signals Subagent that leverages a CodeAct Sandbox for quantitative analysis.
- The system employs a multi-model reconciliation pipeline to resolve claim-level conflicts, ensuring the generation of provenance-preserving reports for complex biomedical research questions.

---

[TACT: Mitigating Overthinking and Overacting in Coding Agents via Activation Steering](http://arxiv.org/abs/2605.05980)

- TACT: introduces a training-free pipeline that detects and mitigates agent drift in LLMs by steering hidden states along contrastive axes at the reasoning-action boundary.
- The framework utilizes an LLM-as-judge to label trajectory steps as overthinking, overacting, or calibrated, enabling the construction of orthogonalized drift axes for real-time activation intervention.
- Experimental results demonstrate that TACT improves resolve rates on long-horizon coding benchmarks by correcting failure modes without requiring additional fine-tuning or prompt modifications.

---

[BehaviorGuard: Online Backdoor Defense for Deep Reinforcement Learning](http://arxiv.org/abs/2605.05977)

- BehaviorGuard: introduces a trigger-agnostic framework that detects and mitigates backdoor attacks in DRL by identifying persistent behavioral drift in action distributions.
- The framework utilizes BDD to quantify deviations from benign action statistics and employs a drift-constrained mitigation mechanism to project abnormal actions back to a safe region without requiring policy retraining.
- BehaviorGuard provides a unified defense for single-agent and multi-agent DRL environments by incorporating DCR to handle interaction-induced non-stationarity and strategic coupling.

---

[PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts](http://arxiv.org/abs/2605.05974)

- PragLocker: introduces a black-box prompt protection scheme that transforms plaintext system prompts into model-specific, non-portable obfuscated prompts using Prompt Initialization and Noise-Injected Prompt Optimizer.
- The framework utilizes a two-phase pipeline consisting of a code-symbol conversion followed by a gradient-free random search optimization driven by Target LLM feedback to ensure functional equivalence on the target model while preventing cross-model portability.
- Experimental results demonstrate that PragLocker effectively protects agent intellectual property by rendering stolen prompts unusable on non-target LLMs while maintaining high performance on the intended target LLM.

---

[TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning](http://arxiv.org/abs/2605.05963)

- TheraAgent: introduces an agentic framework that replaces one-shot generation with an iterative generate-reflect-refine pipeline to improve treatment planning precision and safety.
- The framework utilizes a Planner to generate plans, a Memorizer to maintain historical context, and a TheraJudge to provide clinically grounded, multi-dimensional feedback.
- TheraAgent achieves state-of-the-art performance on HealthBench and demonstrates an 86% win rate against human physicians in blinded expert evaluations.

---

[Intentmaking and Sensemaking: Human Interaction with AI-Guided Mathematical Discovery](http://arxiv.org/abs/2605.05921)

- AlphaEvolve: introduces an interactive interface for evolutionary coding agents that facilitates an iterative intentmaking and sensemaking workflow for scientific discovery.
- The framework utilizes a pipeline of LLMs to optimize code, supported by a critique agent that identifies potential reward hacks and design flaws before full-scale execution.
- The system enables domain experts to refine their research goals through a cycle of defining problem statements, observing AI-generated results, and adjusting parameters based on visual feedback.

---

[Agentic, Context-Aware Risk Intelligence in the Internet of Value](http://arxiv.org/abs/2605.05878)

- OmniRisk Architecture: introduces a five-engine framework for agentic, context-aware risk intelligence in the Internet of Value, utilizing a shared fabric for data ingestion and a Bittensor-based verification subnet for incentivized prediction scoring.
- The framework integrates an LLM-mediated agentic engine governed by constitutional constraints to ensure safe, role-bound execution of pre-committed action programs.
- Empirical validation is provided through a 27-hour policy-constrained liquidity stress-response experiment and a 168-hour production calibration arc demonstrating the deployability of the risk-prediction system.

---

[SkillScope: Toward Fine-Grained Least-Privilege Enforcement for Agent Skills](http://arxiv.org/abs/2605.05868)

- SkillScope: introduces a graph-based framework for fine-grained least-privilege enforcement in LLM Agent Skills by modeling instruction-level procedures and code-level operations as a Unified Execution Graph to identify and constrain task-conditioned over-privileged actions.
- The framework utilizes Over-privilege Candidate Extraction to localize suspicious behaviors, Action Over-privilege Validation to confirm necessity through task-conditioned replay, and Control-flow Privilege Constraining to restrict unnecessary actions while preserving legitimate functionality.
- Experimental results demonstrate that SkillScope achieves 94.53% F1 in detection and reduces triggered over-privileged action-in-task instances by 88.56% across real-world agent runtimes.

---

[SANEmerg: An Emergent Communication Framework for Semantic-aware Agentic AI Networking](http://arxiv.org/abs/2605.05861)

- SANEmerg: introduces a multi-agent emergent communication framework that optimizes signaling protocols for AgentNet systems under strict bandwidth and computational constraints.
- The framework utilizes an importance-filter to prioritize task-relevant message dimensions and an MDL-based complexity-regularizer to enforce efficient, bounded-intelligence communication.
- Experimental results on an AgentNet prototype demonstrate that SANEmerg achieves superior task accuracy while significantly reducing bandwidth and computational overhead compared to existing benchmarks.

---

[LoopTrap: Termination Poisoning Attacks on LLM Agents](http://arxiv.org/abs/2605.05846)

- LoopTrap: introduces an automated red-teaming framework that exploits LLM agent termination mechanisms by injecting malicious prompts to induce unbounded execution loops.
- The framework utilizes Behavior Vulnerability Fingerprinting to profile agent tendencies, Profile-Guided Trap Synthesis to generate context-aware attacks, and Reflective Skill Evolution to refine and store successful attack strategies.
- LoopTrap achieves significant step amplification across diverse LLM agents by dynamically adapting adversarial injections to specific task contexts and agent-level behavioral signatures.

---

[From Chat to Interview: Agentic Requirements Elicitation with an Experience Ontology](http://arxiv.org/abs/2605.05828)

- OntoAgent: introduces an agentic framework that transforms free-form LLM interviews into structured, interpretable requirements elicitation processes by utilizing an experience ontology to guide questioning.
- The framework employs a hierarchical Experience Ontology to decouple the "what to ask" from the "how to ask," enabling systematic exploration of implicit requirements.
- OntoAgent integrates four decision-making operations—ParseUser, ScoreOnto, ReRankOnto, and GatePrune—to dynamically prioritize requirement slots and generate targeted questions while minimizing redundancy.

---

[Long-Horizon Q-Learning: Accurate Value Learning via n-Step Inequalities](http://arxiv.org/abs/2605.05812)

- LQL (Long-horizon Q-learning): introduces a reinforcement learning approach that mitigates compounding TD error by augmenting standard TD updates with trajectory-wise optimality inequalities enforced via asymmetric hinge penalties.
- The framework utilizes Online Q-network, Target Q-network, Replay buffer, Hinge-squared penalty, Lower-bound (LB) penalty, Upper-bound (UB) penalty, Bootstrap policy, and Actor to maintain value function stability without requiring auxiliary networks or additional forward passes.
- LQL demonstrates superior performance on long-horizon tasks by treating trajectory length as a robust scaling axis, effectively absorbing information from long sequences while avoiding the off-policy bias inherent in traditional n-step TD methods.

---

[Sheet as Token: A Graph-Enhanced Representation for Multi-Sheet Spreadsheet Understanding](http://arxiv.org/abs/2605.05811)

- Sheet as Token: introduces a two-stage framework that represents worksheets as unified semantic tokens to enable scalable multi-sheet spreadsheet retrieval.
- The framework utilizes a Feature Extractor to generate schema-aware records, a Sheet Encoder to produce dense Sheet Tokens, and a Graph Retriever to perform query-conditioned cross-sheet reasoning.
- By modeling multi-hop dependencies through a graph-enhanced architecture, the system effectively identifies supporting sheet sets without the computational overhead of full-workbook serialization.

---

[Selective Rollout: Mid-Trajectory Termination for Multi-Sample Agent RL](http://arxiv.org/abs/2605.05802)

- Selective Rollout: introduces a mid-rollout gate that terminates groups of parallel LLM-policy rollouts early when they converge on identical action prefixes, thereby reducing redundant computation in GRPO.
- The gate utilizes mean pairwise prefix edit distance to identify zero-variance groups, effectively filtering them before they contribute to the gradient-batch, which increases the signal-to-noise ratio of the policy updates.
- This approach achieves significant wall-clock savings in agent RL training while maintaining or improving policy performance by mitigating gradient dilution caused by zero-advantage trajectories.

---

[Reward Shaping and Action Masking for Compositional Tasks using Behavior Trees and LLMs](http://arxiv.org/abs/2605.05795)

- MRBT: introduces a symbolic, reactive, and modular framework that leverages LLMs and SMT-solvers to automate reward shaping and action masking for compositional robotic tasks.
- The framework utilizes an MRBT template to structure subtasks, which are then verified against logical specifications to ensure reactivity and robustness across task spaces.
- Experimental results demonstrate that integrating MRBTs into a neurosymbolic RL loop consistently improves training efficiency and task success rates compared to baseline methods.

---

[GazeMind: A Gaze-Guided LLM Agent for Personalized Cognitive Load Assessment](http://arxiv.org/abs/2605.05790)

- GazeMind: introduces a gaze-guided LLM agent framework for personalized cognitive load assessment on smart glasses, utilizing TGE, TGR, AUP, and CogRAG to provide interpretable predictions without model fine-tuning.
- The framework leverages an LLM Engine to process structured gaze data, task-specific guidance, and retrieved historical examples to achieve state-of-the-art performance in cognitive load classification.
- GazeMind incorporates the CogLoad-Bench dataset, which includes 152 participants and 10K+ real-time annotations, to enable robust cross-user and cross-task generalization for wearable AI assistants.

---

[X-OmniClaw: A Unified Mobile Agent for Multimodal Understanding and Interaction](http://arxiv.org/abs/2605.05765)

- X-OmniClaw: introduces a unified edge-native mobile agent architecture that integrates Omni Perception, Omni Memory, and Omni Action to enable reliable, context-aware task execution on Android devices.
- The framework utilizes an Agent Loop for central planning and Trajectory Cloned Execution to transform user interactions into reusable skills, bypassing redundant UI navigation.
- By leveraging on-device processing for perception and memory, the system maintains task continuity and privacy while offloading high-level reasoning to cloud-based LLMs.

---

[BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](http://arxiv.org/abs/2605.05758)

- BioTool: introduces a comprehensive biomedical tool-calling dataset designed to enhance the capabilities of LLMs through instruction fine-tuning on 7,040 human-verified query-API call pairs.
- The framework utilizes a multi-stage construction pipeline involving Tool Selection, API Call Generation, Execution and Heuristic-based Filtering, LLM-based Query Generation, LLM-based Informative Filtering, and Human Refinement to ensure high-quality, scientifically accurate data.
- Experimental results demonstrate that smaller LLMs fine-tuned on BioTool significantly outperform larger proprietary models in tool-calling precision and downstream answer quality, effectively mitigating domain-specific hallucinations.

---

[Multi-Dimensional Behavioral Evaluation of Agentic Stock Prediction Systems Using LLM Judges with Closed-Loop Reinforcement Learning Feedback](http://arxiv.org/abs/2605.05739)

- Agentic Stock Prediction System: introduces a behavioral evaluation framework that utilizes an ensemble of LLM judges to assess stock prediction agents across six domain-specific dimensions and provides closed-loop reinforcement learning feedback.
- The framework decomposes agentic decision-making into structured behavioral traces, enabling targeted credit assignment to the SAC Engine based on diagnostic scores from the LLM Judge.
- Validation through perturbation experiments demonstrates that the LLM Judge ensemble effectively identifies specific component failures, leading to improved predictive performance and Sharpe ratios in high-volatility market conditions.

---

[SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents](http://arxiv.org/abs/2605.05726)

- SKILLRET: introduces a large-scale benchmark for evaluating skill retrieval in LLM agents, utilizing a curated library of 17,810 skills and 63,259 training queries to address the challenge of selecting relevant procedural modules.
- The framework employs a two-stage retrieve-then-rerank pipeline, demonstrating that domain-specific fine-tuning significantly improves retrieval performance by sharpening the model's focus on skill-relevant query segments.
- The benchmark provides a structured taxonomy and disjoint train/evaluation splits, establishing a foundation for future research into long-document matching and efficient skill selection in large-scale agent systems.

---

[Detecting Time Series Anomalies Like an Expert: A Multi-Agent LLM Framework with Specialized Analyzers](http://arxiv.org/abs/2605.05725)

- SAGE (Specialized Analyzer Group for Expert-like Detection): introduces a multi-agent LLM framework that decomposes time-series anomaly diagnosis into specialized analysis perspectives, integrating quantitative and visual evidence through a Detector and Supervisor to produce structured, analyst-facing reports.
- The framework utilizes a dual-representation strategy to balance numerical precision for statistical tools with token-efficient summaries for LLM reasoning, while employing a synthetic in-context learning module to provide contrastive evidence without requiring real anomalous examples.
- By assigning distinct anomaly families to specialized agents—PointAnalyzer, StructAnalyzer, SeasonAnalyzer, and PatternAnalyzer—the system achieves superior detection performance and diagnostic interpretability compared to single-model approaches.

---

[Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes](http://arxiv.org/abs/2605.05724)

- Auto Research with Specialist Agents: introduces a closed-loop empirical framework that automates training-recipe development by partitioning tasks among Specialist Agents that share measured evidence via Shared Lineage Memory.
- The framework utilizes a Closed-Loop Experiment Cycle where agents propose code edits, which are then validated by an External Evaluator, with results stored in a central Blackboard to guide subsequent research moves.
- By treating research as an auditable trajectory of code edits and measured outcomes rather than a single output, the system enables autonomous improvement of training recipes under strict compute and performance constraints.

---

[More Is Not Always Better: Cross-Component Interference in LLM Agent Scaffolding](http://arxiv.org/abs/2605.05716)

- CCI (Cross-Component Interference) Analysis Framework: introduces a systematic empirical study of performance degradation in LLM agents when scaffolding components interact negatively.
- The research demonstrates that adding more scaffolding components to an LLM agent often leads to performance degradation rather than improvement, a phenomenon termed Cross-Component Interference.
- The study reveals that optimal agent configurations are task-dependent and scale-sensitive, with simpler subsets frequently outperforming the full "All-In" agent across multiple model families.

---

[SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](http://arxiv.org/abs/2605.05704)

- SAFEHARBOR: introduces a hierarchical memory-augmented guardrail framework that establishes precise decision boundaries for LLM agents by combining Adversarial Rule Generator, Dynamic Cluster, Memory Tree, Safety Projector, Gating-logic, LLM Judgment, and Rule Engine.
- The framework utilizes an automated adversarial rule generation process to synthesize safety policies, which are then organized into a self-evolving hierarchical memory structure to enable efficient, context-aware retrieval.
- By employing a dual-stage inference pipeline with a lightweight safety projector and an LLM judgment agent, SAFEHARBOR effectively mitigates over-refusal while maintaining robust protection against malicious agentic attacks.

---

[Knowledge-Graph Paths as Intermediate Supervision for Self-Evolving Search Agents](http://arxiv.org/abs/2605.05702)

- KG-based Self-Evolving Search Agents: introduces a framework that reuses knowledge-graph paths as construction-derived intermediate supervision to improve both question generation and reward shaping in self-play.
- The Proposer utilizes LLM-guided Subgraph Extraction to ground question construction in relational contexts, while the Solver employs Waypoint Coverage Reward to receive graded partial credit for trajectories that align with intermediate entities on the construction path.
- This approach enhances the quality of generated training data and provides denser feedback for LLMs without requiring additional human annotations or manually labeled process steps.

---

[Inference-Time Budget Control for LLM Search Agents](http://arxiv.org/abs/2605.05701)

- VOI: introduces a two-stage inference-time control framework for LLM search agents that optimizes budget allocation during search and performs risk-controlled answer finalization.
- The framework utilizes a training-free VOI controller to rank retrieval, decomposition, and answer commitment actions based on estimated marginal task value per unit budget.
- An evidence-grounded finalizer provides conservative answer refinement by selectively intervening only when the expected exactness gain outweighs the risk of damaging the trajectory answer.

---

[An Empirical Study of Proactive Coding Assistants in Real-World Software Development](http://arxiv.org/abs/2605.05700)

- ProCodeBench: introduces a large-scale real-world dataset and benchmark for evaluating proactive coding assistants by analyzing the simulation-to-reality gap in developer behavior.
- The framework utilizes IDE interaction traces and repository context to train and evaluate LLMs, Retrieval-Augmented LLMs, and LLM-based agents on their ability to predict latent developer intent.
- Experimental results demonstrate that LLM-generated simulated data is insufficient for real-world performance, but serves as a valuable initialization for mixed-data training regimes.

---

[Irminsul: MLA-Native Position-Independent Caching for Agentic LLM Serving](http://arxiv.org/abs/2605.05696)

- Irminsul: introduces a content-addressed caching system for Multi-Head Latent Attention (MLA) LLMs that enables position-independent reuse of KV states by combining CDC, a registry, and δ-rotation.
- The framework leverages the structural decoupling of MLA into position-free latents and RoPE-carrying keys to perform efficient, position-independent KV caching with minimal correction overhead.
- By implementing a first-chunk carve-out and content-hash keying, Irminsul recovers significant prompt token reuse in agentic workloads while maintaining output consistency and reducing prefill energy consumption.

---

[Retrieval-Conditioned Topology Selection with Provable Budget Conservation for Multi-Agent Code Generation](http://arxiv.org/abs/2605.05657)

- CODE-AGENT: introduces a multi-agent framework that utilizes a Retrieval Layer to extract structural complexity signals for dynamic topology selection, ensuring provable budget conservation via an Execution Layer.
- The architecture integrates a Routing Layer that maps code-structure signals to one of four orchestration topologies, significantly reducing misrouting compared to text-based classifiers.
- A static Budget Algebra Verifier performs O(|V|+|E|) checks before any LLM invocation, guaranteeing that hierarchical contract delegation never exceeds parent resource bounds.

---

[Agentic Coding Needs Proactivity, Not Just Autonomy](http://arxiv.org/abs/2605.06717)

- Proactive Coding Agent Framework: introduces a three-level taxonomy of proactivity for coding agents, distinguishing between reactive, scheduled, and situation-aware systems.
- The paper proposes a Level 3 engine architecture that maintains development state and a developer mental model to emit insights (notify, question, draft, stay silent) based on interruption cost and utility.
- It defines three metrics—Insight Decision Quality (IDQ), Context Grounding Score (CGS), and Learning Lift (LL)—to evaluate proactive agent behavior beyond simple task completion.

---

[AffectSeek: Agentic Affective Understanding in Long Videos under Vague User Queries](http://arxiv.org/abs/2605.05640)

- AffectSeek: introduces a multi-agent framework that decomposes vague-query-driven affective understanding into role-specialized stages including Core-Agent, Localize-Agent, Perception-Agent, and Reflection-Agent to progressively align user intent with long-video evidence.
- The framework utilizes a coarse-to-fine localization strategy via LocalizeTool and RefinementTool, followed by evidence-based verification and emotion reasoning to improve the reliability and interpretability of affective understanding.
- The authors also construct VQAU-Bench, a comprehensive benchmark integrating long videos, vague natural-language queries, temporal annotations, emotion labels, and evidence-grounded rationales to evaluate agentic affective understanding.

---

[From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](http://arxiv.org/abs/2605.06716)

- LLM Agent Memory Mechanisms Framework: introduces an evolutionary taxonomy for LLM agents, categorizing memory development into Storage (preservation of interaction trajectories), Reflection (active evaluation and refinement), and Experience (cross-trajectory pattern abstraction).
- The framework identifies three core drivers for memory evolution: the necessity for long-range consistency, the challenges of dynamic environments, and the ultimate goal of continual learning.
- This survey synthesizes existing research to provide design principles and a roadmap for next-generation LLM agents, emphasizing the transition from passive storage to autonomous self-evolution.

---

[Architecture Matters: Comparing RAG Systems under Knowledge Base Poisoning](http://arxiv.org/abs/2605.05632)

- RAG architectures: introduces a comparative evaluation of four distinct RAG systems under adversarial knowledge base poisoning, demonstrating that architectural design is a primary variable in system robustness.
- The study evaluates Vanilla RAG, Agentic RAG, MADAM-RAG, and Recursive Language Models (RLM) against CorruptRAG-AK, an adversarial attack targeting credibility assessment through meta-epistemic framing.
- Results indicate that RLM provides the strongest robustness, while the vulnerability of other architectures is largely driven by content-reasoning failures rather than retrieval limitations.

---

[Operationalizing Ethics for AI Agents: How Developers Encode Values into Repository Context Files](http://arxiv.org/abs/2605.05584)

- AGENTS.md: introduces a mechanism where developers encode ethical principles and behavioral constraints directly into repository-level context files to govern AI coding agents.
- This approach translates abstract ethical values like fairness, sustainability, and inclusivity into actionable, machine-interpretable directives within software development workflows.
- The research outlines a roadmap for investigating how these artifacts influence agent behavior, negotiate governance between contributors, and evolve over time within open-source projects.

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


