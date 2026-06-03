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

#### 31st May 2026


[ASE-26: a curriculum for agentic software engineering as a discipline](http://arxiv.org/abs/2606.01152)

- ASE-26: introduces a comprehensive undergraduate curriculum designed to establish agentic software engineering as a rigorous discipline through the integration of ACE, AEE, MRP, the Evolutionary Spiral, Audit Trail, and Verification Gates.
- The curriculum emphasizes the co-evolution of human intent and machine execution, shifting the focus from manual coding to directing LLMs through structured, auditable workflows.
- By prioritizing durable principles over contingent tool capabilities, the framework prepares practitioners to manage the structural risks of agent-mediated development, such as skill atrophy and diffuse accountability.

---


[Application of Algorithms in Energy-Efficient Design Platforms for Green Building](http://arxiv.org/abs/2606.01229)

- Energy-Efficient Design Platform: introduces a multi-layer service architecture that integrates BIM Data Input &amp; Repository, Dynamic Simulation Engine, Multi-objective Optimization Module, and User Interface to optimize building energy performance.
- The platform utilizes a high-performance C++ core and adaptive agent models to facilitate iterative simulation and multi-objective optimization of building envelopes and HVAC systems.
- By leveraging a surrogate model for rapid search and real-time data visualization, the system achieves significant reductions in annual energy consumption while maintaining economic and comfort constraints.

---

[Agyn: An Open-Source Platform for AI Agents with Scalable On-Demand Execution, Agent Definition as a Code, and Zero-Trust Access](http://arxiv.org/abs/2605.27575)

- Agyn: introduces a Kubernetes-native platform for orchestrating LLM-based agents using a signal-driven, stateful serverless runtime, a Terraform-based configuration harness, and a zero-trust security architecture.
- The platform utilizes a Gateway, Authorization, Threads, Notifications, Agents Service, Agents Orchestrator, Organizations, Secrets, Ziti Management, k8s-runner, Agent Container, MCP Sidecar, Ziti Sidecar, LLM Proxy, and Tracing to ensure isolated, secure, and on-demand agent execution.
- Agyn enforces least-privilege security through per-container isolation, OpenZiti-based mTLS overlays, and relationship-based authorization to prevent credential exposure and unauthorized service access by LLMs.

---

[Move the Query, Not the Cache: Characterizing Cross-Instance Latent and Sparse Attention Redistribution Across GPU Fabrics](http://arxiv.org/abs/2606.01502)

- MQNC introduces a topology-aware cost model and selection predicate to optimize cross-instance attention redistribution by routing queries to the holder instead of moving the KV cache.
- The framework leverages the byte-asymmetry of Multi-head Latent Attention (MLA) to make query-routing a fine-grained, latency-bound operation that outperforms cache-moving primitives.
- MQNC utilizes device-initiated RDMA (IBGDA) to minimize per-request overhead, enabling efficient distributed attention across GPU fabrics for agentic workloads.

---

[TimeSage-MT: A Multi-Turn Benchmark for Evaluating Agentic Time Series Reasoning](http://arxiv.org/abs/2606.01498)

- TimeSage-MT: introduces a multi-turn benchmark for evaluating agentic time series reasoning across 240 tasks and 2,680 dialogue turns, utilizing a Validator, Profiler, Planner, Executor, Evaluator, Reporter, TimeSage Skill Library, Memory, Diagnostics, Audit Manifests, and Annotation Dashboard.
- The framework employs a reproducible pipeline to convert real-world time series data into multi-turn conversations with verifiable answer targets, enabling fine-grained diagnosis of agentic capabilities.
- Evaluation results reveal that while code execution and skill libraries improve performance, frontier LLMs still struggle with numerical accuracy and analytical grounding in complex multi-turn workflows.

---

[ClawHub Security Signals: When VirusTotal, Static Analysis, and SkillSpector Disagree](http://arxiv.org/abs/2606.01494)

- ClawHub Security Signals: introduces a sanitized dataset of 67,453 agent skills to evaluate the disagreement between three independent security scanners: VirusTotal, static analysis, and SkillSpector.
- The framework demonstrates that scanner disagreement is structured by attack surface, where VirusTotal excels at detecting bundled-code malware while SkillSpector identifies semantic agentic risks.
- The research argues for a layered, systemic defense strategy for LLM agent ecosystems, emphasizing that no single scanner can provide comprehensive security coverage.

---

[LLM Consortium for Software Design Refinement: A Controlled Experiment on Multi-Agent Collaboration Topologies](http://arxiv.org/abs/2606.01490)

- LLM Consortium for Software Design Refinement: introduces a controlled experimental framework evaluating 12 multi-agent collaboration topologies for software architecture design using a 2x2x2 factorial design.
- The study identifies that structural adversarial review (v4b) and cross-model review (v2b) significantly outperform other topologies by enforcing architectural rigor and leveraging epistemic diversity.
- The research demonstrates that parallel merge strategies suffer from token starvation and the "Frankenstein effect," leading to degraded structural coherence in software designs.

---

[Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX](http://arxiv.org/abs/2606.01478)

- CRAZYFLOW: introduces a high-fidelity, GPU-accelerated, differentiable drone simulator built on JAX that unifies physics and controllers into a single computation graph for massive parallelization.
- The framework leverages JIT compilation via XLA to enable real-time, in-flight learning and optimization by fusing Policy/Controller, Low-level control, Dynamics, Custom components, Numerical integration, and Sensing into a single differentiable pipeline.
- CRAZYFLOW supports both first-principles and abstracted dynamics models, enabling sub-centimeter sim-to-real transfer and high-throughput training of RL agents without domain randomization.

---

[Genotype-Conditioned Molecular Generation via Evidence-Grounded Multi-Objective Latent Perturbation in Diffusion Models](http://arxiv.org/abs/2606.01461)

- Genotype-Conditioned Molecular Generation via Evidence-Grounded Multi-Objective Latent Perturbation in Diffusion Models: introduces a latent-space optimization framework that enhances a pretrained G2D-Diff model by jointly optimizing for drug sensitivity, viability, and mechanistic plausibility.
- The framework utilizes a multi-agent LLM pipeline, including BiologyAgent, ChemistryAgent, and ScoreAgent, to provide biologically grounded feedback that guides the diffusion model's latent space toward clinically relevant molecular candidates.
- To enable efficient optimization, the approach employs online property surrogate networks that provide dense, differentiable approximations of non-differentiable reward signals, facilitating gradient-based updates without requiring backbone retraining.

---

[Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence](http://arxiv.org/abs/2606.01444)

- CategoryScienceClaw: introduces a category-theoretic framework for agentic discovery systems that models scientific states as copresheaves and discovery as verified regime transitions.
- The framework utilizes a schema category to define admissible artifacts and operations, employing gates to manage commitment and Kan extensions to transport evidence across regime transitions.
- This approach enables the construction of auditable, self-revising knowledge-computation graphs that preserve provenance while allowing for the evolution of scientific representational regimes.

---

[Dive into Ambiguity: A*-Inspired Multi-Agents Commonsense Obfuscation Attack on LLM Prompts](http://arxiv.org/abs/2606.01441)

- A*-inspired Factual Error Induction Framework: introduces a heuristic-guided search method that generates semantically aligned yet obfuscated prompts to systematically induce commonsense hallucinations in LLMs.
- The framework utilizes a Hierarchical Rewrite Strategy with adaptive aggressiveness levels and includes planning-, perception- and tool use-agents to optimize adversarial prompt generation.
- The Agentic Mechanism Labeling (AML) component provides interpretability by automatically discovering and refining adversarial mechanisms, which are then used to guide subsequent prompt rewrites.

---

[Self-Healing Agentic Orchestrators for Reliable Tool-Augmented Large Language Model Systems](http://arxiv.org/abs/2606.01416)

- Self-Healing Agentic Orchestrator: introduces a modular reliability control plane that maps failure signals to targeted recovery actions, verifies recovered trajectories, and records observability traces for tool-augmented LLM systems.
- The framework treats reliability as a bounded runtime control problem, utilizing a monitor–detect–diagnose–recover–verify loop to manage failures without relying on unbounded retry or replanning.
- Empirical results demonstrate that this approach achieves 98.8% task success in fault-injected benchmarks, significantly outperforming static, retry-only, and full-replanning baselines under matched recovery budgets.

---

[Agent Skills Should Go Beyond Text: The Case for Visual Skills](http://arxiv.org/abs/2606.01414)

- AutoVisualSkill: introduces a multimodal skill paradigm that addresses the textual bottleneck in agent experience by combining declarative textual logic with explicit visual priors.
- The framework categorizes visual skills into static priors for spatial conventions, dynamic priors for in-situ tracking, and interleaved skills for evidence-grounded reasoning.
- Experimental results demonstrate that integrating visual priors significantly improves agent performance on GUI grounding and dense object counting tasks compared to text-only approaches.

---

[Bridging Requirements and Architecture: Multi-Agent Orchestration with External Knowledge and Hierarchical Memory](http://arxiv.org/abs/2606.01385)

- MAAD (Multi-Agent Architecture Design): introduces a knowledge-driven multi-agent framework that orchestrates Analyst, Modeler, Designer, and Evaluator agents to autonomously transform requirements into comprehensive, multi-view architectural blueprints.
- The framework leverages RAG for external knowledge infusion and a hierarchical memory mechanism (Working Memory, Episodic Memory, Semantic Memory) to enable iterative architecture refinement and cross-task knowledge reuse.
- Empirical evaluations demonstrate that MAAD outperforms existing multi-agent baselines by producing more modular, traceable, and evaluation-validated architectures through specialized agent roles and systematic quality assurance.

---

[ActMVS: Active Scene Reconstruction with Monocular Multi-View Stereo](http://arxiv.org/abs/2606.01367)

- ActMVS: introduces a monocular active reconstruction framework that integrates a View Factor Graph, Voxel Map, Gaussian Splatting Map, Planner, MVSA, and Global Depth Optimization to enable online, globally consistent dense depth mapping without depth sensors.
- The framework utilizes a View Factor Graph with voxel-frame visibility modeling to select optimal reference frames for MVSA-based depth prediction.
- Global depth optimization enforces 3D consistency across co-visible regions through pixel-level depth warping and alignment, achieving performance competitive with RGB-D methods.

---

[Early Diagnosis of Wasted Computation in Multi-Agent LLM Systems via Failure-Aware Observability](http://arxiv.org/abs/2606.01365)

- Failure-Aware Observability Framework: introduces a diagnostic layer for multi-agent LLM systems that maps recurring failure modes to online trace signals and offline semantic metrics to identify wasted computation.
- The system utilizes an Orchestrator, Search Component, and Execution Component to generate structured event traces that enable the detection of orchestration loops, tool instability, and evidence gaps.
- Empirical analysis on GAIA validation traces demonstrates that early warning signals and grounding metrics provide actionable insights into agent performance beyond final-answer accuracy.

---

[Needles at Scale: LLM-Assisted Target Selection for Windows Vulnerability Research](http://arxiv.org/abs/2606.01364)

- Symbolicate-Enrich-Sample: introduces a three-stage batch pipeline that transforms a corpus of production Windows binaries into a queryable, priority-ranked research queue for vulnerability analysis.
- The pipeline utilizes a low-cost LLM to assign reachability, risk, and bug-class hypotheses to functions based on deterministic structural features, effectively filtering millions of functions into a manageable shortlist.
- By focusing on target selection rather than direct analysis, the framework provides a scalable substrate that allows human analysts or LLM agents to prioritize high-value code paths efficiently.

---

[All Models are Wrong, Knowing Where is Useful: On Model Uncertainty in Reinforcement Learning](http://arxiv.org/abs/2606.01363)

- Uncertainty-Aware MBRL framework: introduces a methodology for mitigating model exploitation in Model-Based Reinforcement Learning by leveraging a Probabilistic Ensemble (PE) model to quantify epistemic uncertainty and restricting rollouts to a Certain Area (E) or paths with low information loss.
- The framework utilizes the Infoprop mechanism to perform maximum likelihood estimation, effectively correcting predictive distributions and enabling long-horizon planning by treating epistemic uncertainty as channel noise.
- By integrating these uncertainty-aware components into Dyna-style architectures, the approach facilitates safe exploration and efficient hardware learning, as demonstrated by the successful control of an underactuated unicycle robot.

---

[Recognize Your Orchestrator: An Entropy Dynamics Perspective for LLM Multi-Agent Systems](http://arxiv.org/abs/2606.01351)

- IWG (Inverse Workflow Generation): introduces a framework that models multi-agent orchestration as a system governed by the competing forces of task resolution and cumulative context loading, using a Mean-Field Entropy Dynamics perspective to quantify system stability.
- The framework utilizes a Scout Agent, Wrapper Agent, and Validation Committee to synthesize process-verifiable benchmarks with dense intermediate checkpoints, enabling empirical analysis of LLM-based orchestrator performance.
- The research identifies a "Reasoning Trap" where heavy-thinking models fail as orchestrators due to context squeezing, advocating for "Instant Breadth-First Thinking" strategies to maintain system stability.

---

[Reducing Token Usage of State-in-Context Agents using Minification](http://arxiv.org/abs/2606.01326)

- DirectSolve (State-in-Context Agent): introduces a framework that reduces LLM token consumption in software engineering agents by applying semantics-preserving source code minification before the repair step.
- The framework utilizes a Ranking-Agent to identify relevant files, a Minification-Module to compress source code, and a Repair-Agent to generate patches, supported by a Mapping-Table and Git-Diff-Matcher to ensure patch validity.
- Empirical evaluation on SWE-bench Verified demonstrates that minification reduces average input token usage by 42% with a 12% absolute drop in resolution rate, highlighting a significant trade-off between cost efficiency and agent performance.

---

[Digital Twin-Assisted Adaptive Multi-Agent DRL for Intelligent Spectrum and Resource Management in Open-RAN UAV-Enabled 6G Networks](http://arxiv.org/abs/2606.01324)

- DT-assisted PSO-MADRL framework: introduces a hybrid architecture for 6G networks that decomposes optimization into PSO-based UAV trajectory planning and MADRL-based resource management, utilizing DT-driven centralized training and decentralized edge inference.
- The framework integrates Non-RT RIC for global model training and Near-RT RIC for real-time distributed execution, ensuring low-latency and energy-efficient spectrum sharing among UAVs and ground users.
- By employing twin critic networks and target policy smoothing within a DDPG-based MADRL approach, the system achieves stable convergence and robust adaptability in dynamic UAV-assisted wireless environments.

---

[SABER: Benchmarking Operational Safety of LLM Coding Agents in Stateful Project Workspaces](http://arxiv.org/abs/2606.01317)

- SABER: introduces a benchmark for evaluating the operational safety of LLM agents in stateful, Docker-sandboxed project workspaces by analyzing action traces rather than isolated responses.
- The framework utilizes a layered judging protocol combining a Rule-based Judge for deterministic pattern matching and an LLM Judge for semantic analysis of multi-step agent behavior.
- SABER categorizes safety risks into embedded injection, risky self-selection, and contextual warnings, providing a comprehensive evaluation of agent safety beyond simple prompt-refusal metrics.

---

[Science Earth: Towards A Planet-Scale Operating System for AI-Native Scientific Discovery](http://arxiv.org/abs/2606.01316)

- EACN: introduces a planet-scale scientific runtime that enables heterogeneous capabilities to discover one another, negotiate task ownership, and adjudicate across incompatible evidentiary standards through an open protocol stack.
- The framework utilizes four coordination primitives—discovery, bidding, adjudication, and reputation—to facilitate emergent task decomposition and long-horizon scientific collaboration without central orchestration.
- By connecting software agents, GPU clusters, and human scientists under a unified protocol, the system allows scientific questions to generate their own collaboration structure and self-correct through cross-tradition comparison.

---

[SkillSmith: Co-Evolving Skills and Tools for Self-Improving Agent Systems](http://arxiv.org/abs/2606.01314)

- SkillSmith: introduces a synergy-aware framework that jointly evolves Skills (St), Tools (Tt), and Anti-pattern Memory (Ft) to enable robust agent self-improvement.
- The framework utilizes a Proposer (R) to generate atomic bundles that modify both skills and tools, while the Tool-Smith (Bτ) applies typed lifecycle operations to ensure safe and controlled tool evolution.
- An ecological utility model based on Lotka-Volterra dynamics manages skill interactions, and the Anti-pattern Memory (Ft) prevents amnesic regression by vetoing previously failed configurations.

---

[PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making](http://arxiv.org/abs/2606.01313)

- PSG-Nav: introduces a framework for robust open-vocabulary navigation that utilizes a 3D-PSG (hierarchical probabilistic environment representation), Multiverse Decision (stochastic planning via sampled worlds), and an EEC (RAG-based confidence calibration module) to mitigate perception uncertainty.
- The framework maintains a 3D-PSG to preserve full semantic distributions, which are then sampled into discrete worlds for LLM-based reasoning and goal selection.
- The EEC cross-references visual detections against a Positive Bank (memory of successful goal identifications) and a Negative Bank (memory of historical false positives) to dynamically calibrate confidence and suppress false-positive navigation terminations.

---

[SkillAdaptor: Self-Adapting Skills for LLM Agents from Trajectories](http://arxiv.org/abs/2606.01311)

- SkillAdaptor: introduces a training-free framework that performs step-level failure attribution to refine agent skills without updating the backbone LLM.
- The framework utilizes a Localizer and Linker to identify specific faulty steps and responsible skills, followed by a Generator or Reviser to modify the skill collection, and a Qualifier to ensure performance stability.
- By shifting from trajectory-level reflection to step-level attribution, the approach enables more precise and auditable skill maintenance for LLMs in long-horizon interactive tasks.

---

[Multiagent Matroid Upgrading: Greedy is Fair and Efficient](http://arxiv.org/abs/2606.01309)

- MMUP (Multiagent Matroid Upgrading Problem): introduces a polynomial-time greedy algorithm to solve resource allocation tasks by iteratively selecting elements to upgrade within agent-specific matroid constraints.
- The framework leverages a nestedness property of optimal solutions to ensure that greedy selection of elements achieves global optimality for both efficiency and fairness objectives.
- The approach extends to generalized settings, including minimax objectives and interval fairness constraints, while also providing a novel combinatorial solution for budget-constrained minimum spanning tree problems with binary weights.

---

[ANDES: Agent Native Data Evolving Synthesis Tool for Autonomous Instruction Alignment](http://arxiv.org/abs/2606.01279)

- ANDES: introduces a plug-and-play agent skill that reimagines data synthesis as a dynamic, closed-loop process for autonomous LLM post-training.
- The framework utilizes a self-evolving World Tree routing mechanism and report-driven feedback to steer data generation toward target-aligned capabilities while maintaining contextual diversity.
- By abstracting data curation into an interactive tool, ANDES enables foundationally weaker LLMs to achieve state-of-the-art performance on complex benchmarks under strict compute constraints.

---

[DeepIPCv3: Event-Aware Multi-Modal Sensor Fusion for Sudden Pedestrian Crossing Avoidance](http://arxiv.org/abs/2606.01277)

- DeepIPCv3: introduces a multi-modal autonomous navigation framework that synergizes LiDAR-based spatial perception with asynchronous DVS event streams using a Transformer-inspired cross-modal attention mechanism to achieve low-latency obstacle avoidance.
- The architecture utilizes a hybrid planning-control module that combines traditional PID tracking with a neural policy network to generate safe waypoints and control commands.
- By dynamically weighting LiDAR spatial anchors against DVS instantaneous dynamic updates, the framework effectively mitigates motion blur and exposure failures during sudden pedestrian crossing scenarios.

---

[Domination-Avoiding Learning Agents Cannot Collude](http://arxiv.org/abs/2606.01275)

- DA (Domination-Avoiding) framework: introduces a class of learning agents that provably avoid collusive outcomes by iteratively eliminating strictly dominated strategies in repeated games.
- The research demonstrates that while standard no-external-regret algorithms can sustain supra-competitive collusion, DA-agents are guaranteed to converge to competitive Nash equilibrium prices in Bertrand duopoly settings.
- The paper establishes that several common learning dynamics, including Mean-Based learners, No-Swap-Regret minimizers, and contextual variants, satisfy the DA property, thereby precluding spontaneous algorithmic collusion.

---

[Agentic Clustering: Controllable Text Taxonomies via Multi-Agent Refinement](http://arxiv.org/abs/2606.01255)

- Agentic Clustering: introduces an agentic framework for text clustering that replaces fixed programmatic pipelines with an orchestrator that dynamically dispatches specialized agents to refine taxonomies.
- The system utilizes an Orchestrator to manage a loop of Proposer-, Synthesizer-, Auditor-, Investigator-, and Critic-agents to iteratively construct and validate taxonomies based on user-defined constraints.
- This approach achieves state-of-the-art performance on seven text-clustering benchmarks by adapting the discovery process to the specific structure and granularity of the input corpus.

---

[Trust Region On-Policy Distillation](http://arxiv.org/abs/2606.01249)

- TrOPD (Trust Region On-Policy Distillation): introduces a framework that partitions student-generated tokens into trust regions and outliers to stabilize on-policy distillation for LLMs.
- The framework utilizes On-Policy Trust Region, Outlier Estimation, and Off-Policy Guidance to mitigate unreliable policy gradients while preserving informative supervision from the teacher model.
- TrOPD consistently outperforms existing on-policy distillation baselines across mathematical, coding, and general-domain reasoning benchmarks by ensuring stable optimization through adaptive token-level supervision.

---

[Where to Look: Can Foundation Models Reach a Target Viewpoint Through Active Exploration?](http://arxiv.org/abs/2606.01247)

- TVR (Target Viewpoint Reproduction): introduces a closed-loop active exploration task where an agent must navigate a 3D environment until its egocentric observation matches a target image, utilizing Multimodal LLM, Visual-Action Memory, Action-Only Memory, Rule-Based Expert Planner, Simulator Environment, GRPO, SFT, and CoT Annotator.
- The framework evaluates foundation models on TVRBench, identifying that models struggle primarily with mapping spatial discrepancies to body translation rather than visual matching.
- Post-training with visual-action SFT and multi-turn GRPO significantly improves performance, whereas CoT supervision and single-turn GRPO are shown to degrade closed-loop control.

---

[HomeFlow: A Data Flywheel for Smart Home Agent Training with Verifiable Simulation](http://arxiv.org/abs/2606.01230)

- HomeFlow: introduces a verifiable data flywheel that leverages HomeEnv, HomeMaker, Blueprint, and MCTS-Flow to generate high-quality training data for smart home agents.
- The framework utilizes MCTS-Flow to synthesize diverse, verifiable multi-turn trajectories, which are then used to optimize LLM agents via supervised fine-tuning and step-wise RLVE.
- By grounding agent training in a physically verifiable simulation, HomeFlow enables LLMs to master complex device-control logic and recover from compounding errors in multi-turn interactions.

---

[DiscourseFlip: An Oblique Discourse-Level Opinion Manipulation Attack against Black-box Retrieval-Augmented Generation](http://arxiv.org/abs/2606.01212)

- DiscourseFlip: introduces a discourse-level opinion manipulation attack that coordinates influence across a semantic query network to induce holistic opinion shifts in black-box RAG systems.
- The framework utilizes a graph-guided agentic process to dynamically allocate a limited poisoning budget across Atomic Semantic Units (ASUs) to maximize coverage and opinion deviation.
- Extensive experiments demonstrate that DiscourseFlip achieves superior coverage and effectiveness compared to existing baselines while remaining highly camouflaged from user detection.

---

[Can LLM Agents Sustain Long-Horizon Organizational Dynamics?](http://arxiv.org/abs/2606.01199)

- TaskWeave: introduces a hierarchical agentic framework for simulating long-horizon organizational dynamics by coupling FPDA-based planning-state propagation with dependency-aware execution.
- The framework utilizes a hierarchical simulation memory mechanism to maintain organizational intent, task commitments, and grounded execution traces across temporal levels.
- TaskWeave enables closed-loop simulation where planning states generate tasks, execution grounds these tasks in memory, and results are written back to support subsequent organizational activities.

---

[Linear Strategic Classification with Endogenous Improvements](http://arxiv.org/abs/2606.01198)

- STRAT-IMP-AWARE: introduces a framework for improvement-aware strategic classification where agents' feature modifications induce genuine, probabilistic qualification changes.
- The approach utilizes a strategic-optimal shifted classifier that serves as a superior surrogate for improvement-aware objectives compared to standard Bayes-optimal models.
- The framework provides PAC-style learnability guarantees and a practical plug-in algorithm that jointly optimizes over strategic shifts and oracle-adjusted labels.

---

[PairedGTA: Generating Driving Datasets for Controlled Photometric Shift Analysis](http://arxiv.org/abs/2606.01192)

- PairedGTA: introduces a deterministic game-engine-based framework for generating pixel-aligned driving datasets under controlled illumination and weather conditions, utilizing Python Client, DeepGTAV Server, VPilot, ScriptHookV, GTA V Game Engine, Scene Descriptor, and Semantic Segmentation Models.
- The framework preserves scene geometry, camera pose, and dynamic object identity across multiple photometric variants to enable precise, counterfactual robustness analysis of visual perception systems.
- Experimental results demonstrate that this controlled synthetic data provides clearer insights into model performance degradation under adverse conditions compared to real-world datasets where semantic and geometric factors are entangled.

---

[“Skill issues”: data-centric optimization of lakehouse agents](http://arxiv.org/abs/2606.01185)

- Bauplan: introduces a data-centric optimization pipeline for LLM agents operating on a branching lakehouse, utilizing LLM-powered data generation module, Harbor orchestration, Modal sandboxes, Coding agent, Bauplan lakehouse, Optimizer, and Validation script.
- The framework leverages the isomorphism between pipeline code and data commits to enable fine-grained, programmatic verification of agentic write-path operations.
- Optimized skills improve agent accuracy by 31.9% by iteratively refining textual hyperparameters through a closed-loop evaluation process on synthetic, yet realistic, data tasks.

---

[Coordinating Task Switching in a Robotics Multi-Agent System Using Behavior Trees](http://arxiv.org/abs/2606.01170)

- ThunderVolt: introduces a Behavior Tree-based coordination strategy for multi-agent robotics systems to improve modularity and maintainability over traditional Finite State Machines.
- The framework utilizes the MOISE+ organizational model to define robot roles and a Blackboard architecture to facilitate real-time state sharing and dynamic task switching between agents.
- Experimental results demonstrate that the Behavior Tree approach achieves superior performance in goal-scoring efficiency and reactivity compared to the previous Finite State Machine implementation.

---

[Towards Interactive Video World Modeling: Frontiers, Challenges, Benchmarks, and Future Trends](http://arxiv.org/abs/2606.01164)

- Interactive World Modeling (IWM): introduces a comprehensive survey of interactive world models, focusing on action-conditioned controllability, long-horizon interactions, and real-time responsiveness.
- The paper categorizes interactive world models by their ability to integrate user actions into world state transitions, utilizing memory mechanisms and efficient generation backbones to maintain temporal consistency.
- It provides a systematic review of current benchmarks and future research directions across application domains including embodied AI, autonomous driving, game engines, and open-world exploration.

---

[On Fréchet Traveling Salesmen Problems](http://arxiv.org/abs/2606.01147)

- Fréchet-TSP: introduces a new class of TSP-style problems where the goal is to partition a set of points into two curves that minimize the Fréchet distance between them, utilizing Unit-Disk Graph, Nearest-Neighbor-Graph, Minimum-Spanning-Tree, Star-Cover, DFS-Traversal, and Partitioning-Algorithm.
- The paper presents a near-linear time algorithm for the discrete Fréchet-TSP by constructing a spanning forest of the Unit-Disk Graph and partitioning it into two curves using a DFS-based coloring strategy.
- The authors further analyze variants for minimizing curve lengths and balancing vertex distribution, while proving that the continuous version of the Fréchet-TSP is NP-hard.

---

[Repeated Descent: A Framework for Online Budget-Feasible Auctions](http://arxiv.org/abs/2606.01142)

- RED (Repeated Descent): introduces a deterministic framework for online budget-feasible procurement auctions that adaptively adjusts pricing thresholds to maintain budget feasibility without explicit estimation of the optimal scale.
- The framework utilizes a "win-win" structure where either successful descents accumulate significant value or low-threshold pricing dominates, ensuring constant-competitive performance for monotone and non-monotone submodular valuations.
- The paper establishes a logarithmic lower bound for XOS valuations, demonstrating that constant-factor approximation is impossible for this class in the online budget-feasible setting.

---

[memorywire: A Vendor-Neutral Wire Format for Agent Memory Operations](http://arxiv.org/abs/2606.01138)

- memorywire: introduces a vendor-neutral JSON-Schema wire format for agent memory operations, utilizing a Memory facade, MemoryRouter, Backend adapters, STM-LTM transformer, Governance plane, and Audit log to standardize memory management across heterogeneous frameworks.
- The framework employs a MemoryRouter that aggregates results from multiple backends using Reciprocal Rank Fusion to ensure robustness against malicious or unavailable storage components.
- It provides a governance plane that implements a diff-and-approve workflow, allowing human oversight of memory mutations before they are committed to long-term storage.

---

[CAREAgent: Clinical Agent with Structured Reasoning and Tool-Integrated for Order Generation](http://arxiv.org/abs/2606.01094)

- CAREAgent: introduces a two-stage agentic framework that constructs verifiable clinical reasoning trajectories through structured multi-stage reasoning and real-world tool integration.
- The framework utilizes an EMR summarization agent to maintain context and employs a two-stage post-training process combining supervised fine-tuning and reinforcement learning to enhance clinical reasoning and tool usage.
- Experimental results on multiple benchmarks demonstrate that CAREAgent significantly improves clinical order generation performance by explicitly modeling fine-grained attributes and priority rankings.

---

[Deep Research as Rubric for Reinforcement Learning](http://arxiv.org/abs/2606.01091)

- DR-Rubric: introduces a two-stage framework that reframes rubric construction as an evidence-driven research process to generate fine-grained reward signals for LLMs.
- The framework utilizes Information Elicitation to gather domain facts and Rubric Synthesis to distill these into atomic constraints, which are then used for GRPO Update and Bootstrap Self-Improvement Loop.
- By enabling LLMs to self-generate customized rubrics through iterative research, the approach achieves strong performance on agentic and reasoning tasks while mitigating reward sparsity.

---

[Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs](http://arxiv.org/abs/2606.01072)

- TDSG (Task-Driven Scene Graph): introduces a structured memory mechanism that maintains a dynamic, task-relevant scene graph to provide spatial and temporal context for robotic imitation learning in partially observed environments.
- The framework utilizes LLMs and vision foundation models to incrementally update object-centric nodes, consisting of visual embeddings, 2D bounding boxes, and 3D centroids, which are then processed by a transformer-based diffusion policy.
- Experimental results demonstrate that this explicit, compact memory representation significantly improves policy performance and robustness in long-horizon mobile and tabletop manipulation tasks compared to memory-less baselines.

---

[Leyline: KV Cache Directives for Agentic Inference](http://arxiv.org/abs/2606.01065)

- Leyline: introduces a serving-side primitive for policy-driven KV cache editing that decouples edit intent from kernel-level cache management.
- The framework utilizes a declarative (span, replacement) directive interface to perform in-place cache splices via a closed-form δ-rotation kernel, avoiding expensive re-prefill operations.
- Leyline enables agentic LLMs to actively prune or modify their context window, improving solve rates and throughput by maintaining attention correctness across shifted token positions.

---

[MindClaw: Closed-Loop Embodied Mental-State Reasoning for Precision Intervention](http://arxiv.org/abs/2606.01063)

- MindClaw: introduces a closed-loop framework for embodied mental-state reasoning that utilizes an Input Interface, Claw Layer, and Reasoning Layer to provide precision intervention.
- The framework employs a Trigger module as an embodied cognitive skill to decide when to invoke Observation, Mental Reasoning, or Action Generation, ensuring the agent only intervenes when necessary.
- By maintaining a persistent Memory with actor-specific belief tables, MindClaw enables LLMs to perform real-time, context-aware assistance in dynamic simulated environments.

---

[3DCodeBench: Benchmarking Agentic Procedural 3D Modeling Via Code](http://arxiv.org/abs/2606.01057)

- 3DCodeBench: introduces a comprehensive benchmark and evaluation framework for assessing VLM agents in procedural 3D generation by leveraging an agentic curation pipeline with human-in-the-loop verification.
- The framework utilizes a Skills Library and an Experience Library to enable iterative refinement of procedural scripts, addressing challenges in API usage and geometric structural integrity.
- 3DCodeBench includes a large-scale dataset of 26K multimodal prompts and 3D object triplets, alongside 3DCodeArena for human-preference-based ranking of LLM-generated 3D assets.

---

[TravelEval: A Comprehensive Benchmarking Framework for Evaluating LLM-Powered Travel Planning Agents](http://arxiv.org/abs/2606.01046)

- TravelEval: introduces a comprehensive benchmarking framework for evaluating LLM-powered travel planning agents across six orthogonal dimensions including accuracy, compliance, temporality, spatiality, economy, and utility.
- The framework utilizes a realistic data sandbox and a simulation-based evaluation method to assess agent performance in complex, multi-constraint travel planning scenarios.
- Experimental results on 12 mainstream approaches reveal that current LLMs struggle with globally-optimized multi-dimensional planning and that agentic reasoning strategies offer no consistent performance improvement.

---

[ExpWeaver: LLM Agents Learn from Experience via Latent Reasoning](http://arxiv.org/abs/2606.01041)

- ExpWeaver: introduces a framework that enables LLMs to learn from past experiences via latent retrieval-augmented generation, bypassing explicit text-level constraints.
- The framework utilizes an Experience Bank to store trajectories as dense latent embeddings, which are retrieved and integrated into the LLM's hidden states using cross-attention and gated residual mechanisms.
- ExpWeaver achieves state-of-the-art performance across 12 of 13 tasks while maintaining token efficiency comparable to non-retrieval baselines through end-to-end reinforcement learning optimization.

---

[Tackling the Root of Misinformation by Teaching Laypeople about Logical Fallacies via Socratic Questioning and Critical Argumentation](http://arxiv.org/abs/2606.01020)

- LFTutor: introduces an LLM-based tutoring system that utilizes intent-based pedagogical steering to teach laypeople to recognize logical fallacies through structured Socratic questioning and critical argumentation.
- The framework integrates a Disagreement Bank, Intent Detection, Intent-based Strategy Selection, and Verified Strategy Execution to maintain pedagogical focus and avoid common LLM tutoring pitfalls.
- Automatic and human evaluations demonstrate that LFTutor significantly outperforms baseline LLMs in maintaining focus, providing guidance, and fostering critical thinking during multi-turn dialogues.

---

[Hybrid Verified Decoding: Learning to Allocate Verification in Speculative Decoding](http://arxiv.org/abs/2606.01019)

- Hybrid Verified Decoding: introduces a payoff-guided speculative decoding framework that dynamically allocates verification work between a low-cost cache-based drafter and a model-based drafter.
- The framework utilizes a lightweight Payoff Predictor to estimate the accepted length of cache-based drafts, ensuring that only high-payoff proposals are submitted for verification by the Target Model.
- By optimizing the selection between cache-based and model-based drafting, the approach reduces sequential decoding cycles and improves end-to-end throughput across diverse agentic and structured workloads.

---

[AI-IoT-Robotics Integration: Survey of Frameworks, Emerging Trends, and the Path Toward Connected Robotics](http://arxiv.org/abs/2606.01015)

- AI-IoT-Robotics Integration Framework: introduces a modular system architecture that aligns AI, IoT, and robotics to enable distributed cognition and autonomous decision-making across edge-fog-cloud hierarchies.
- The framework utilizes a layered approach where Cloud AI handles global reasoning, Fog nodes manage task coordination, and Edge AI enables low-latency inference for robotic agents.
- This survey classifies integration depth and identifies key challenges in interoperability, real-time feedback control, and secure distributed intelligence for next-generation Connected Robotics.

---

[FVSpec: Real-World Property-Based Tests as Lean Challenges](http://arxiv.org/abs/2606.01008)

- FVSpec: introduces a benchmark of 9,415 Lean 4 formal verification challenges derived from real-world Python property-based tests, utilizing a pipeline of Scraper, Discovery-agent, Licensing-filter, Extraction-agent, Formalization-agent, LSP-repair-loop, Post-production-validator, Haiku-grader, and Lean-kernel.
- The framework leverages an agentic transpilation pipeline to convert Python code and property-based tests into Lean implementations and theorem statements, with iterative feedback from the Lean Language Server Protocol to ensure compilation.
- The benchmark provides a challenging, out-of-distribution dataset for LLMs, with baseline evaluations demonstrating that frontier models achieve 70% success on easy problems and 49% on hard problems.

---

[An Open-Source Benchmark and Baseline for Multi-temporal Referring Segmentation](http://arxiv.org/abs/2606.00987)

- MTRefSeg-R1: introduces a change-aware LVLM framework for multi-temporal referring segmentation that utilizes a two-stage training strategy to align language instructions with temporal variations.
- The framework integrates a CRAFT-Agent for automated data construction and a Change-Aware Fusion Block to explicitly model temporal differences between bi-temporal image inputs.
- MTRefSeg-21K provides a large-scale benchmark containing 21K high-quality bi-image–text–mask triplets to facilitate research in language-guided temporal change understanding.

---

[Prospect-Theory Behavior from Bellman Optimality in MDPs with Catastrophic States](http://arxiv.org/abs/2606.00970)

- Catastrophe-boundary MDP: introduces a structural mechanism where standard Bellman optimality in environments with an absorbing failure state generates prospect-theory-like signatures including S-shaped value functions, endogenous loss sensitivity, and reflection-effect policy reversals.
- The framework utilizes a Bellman optimality equation to derive optimal policies, demonstrating that risk-averse behavior in growth regimes and risk-seeking behavior in decline regimes emerge from continuation values near catastrophe rather than preference curvature.
- The research validates these findings through a value iteration algorithm, a tabular Q-learning agent, and robustness testing against various stochastic transition kernels, confirming the emergence of prospect-theory-like behavior across diverse parameter configurations.

---

[When Parallelism Pays Off: Cohesion-Aware Task Partitioning for Multi-Agent Coding](http://arxiv.org/abs/2606.00953)

- Co-Coder: introduces a multi-agent orchestration framework that models repository-level coding as a graph partitioning problem to optimize the communication-to-computation trade-off.
- The framework utilizes Repository Interface Blueprint, Weighted Dependency Graph, Structural Hub Isolation, Community Detection, Latent Parallelism Exploitation, Dependency-Aware List Scheduler, and a Leader Agent to minimize inter-agent communication while maximizing parallel execution efficiency.
- Co-Coder improves pass rates and reduces latency and API costs by partitioning tasks into cohesive groups, effectively mitigating the overheads typically associated with uncoordinated multi-agent LLM systems.

---

[FinCom: A Financial Multi-Agent Demo with Disagree-or-Commit Deliberation](http://arxiv.org/abs/2606.00939)

- FinCom (Financial Committee): introduces a governed multi-agent framework that utilizes a Supervisor and three specialist agents to perform financial analysis while mitigating sycophancy through the DoC Protocol.
- The system employs a Supervisor to orchestrate Research, Quant, and Risk agents, which are implemented as LangGraph nodes and equipped with role-specific API Tools for data-driven decision support.
- By requiring agents to explicitly critique or commit to peer reasoning, the DoC Protocol enhances the auditability and epistemic robustness of LLM-based financial committee deliberations.

---

#### 30th May 2026

[Generative Multi-Robot Motion Planning via Diffusion Modeling with Multi-Agent Reinforcement Learning Guidance](http://arxiv.org/abs/2606.00933)

- Generative Multi-Robot Motion Planning via Diffusion Modeling with Multi-Agent Reinforcement Learning Guidance: introduces a framework that combines decentralized diffusion-based trajectory generation with a centralized MARL-based value function to enable coordinated multi-robot motion planning.
- The framework utilizes a shared Diffusion Model to generate independent trajectories, which are then refined through Gradient-based Steering using a MARL-trained value function to reduce inter-agent interference.
- By applying the centralized value function as a guidance signal during the reverse diffusion process, the approach achieves multi-agent coordination without requiring joint trajectory modeling or retraining of the generative model.

---

[CV-Arena: An Open Benchmark for Instructional Computer Vision Problem Solving with Human-AI Collaborative Preferences](http://arxiv.org/abs/2606.00931)

- CV-Arena: introduces a professional-grade benchmark for instructional computer vision problem solving, utilizing CogRetriever, CV-Judge, Active Elo, and CV-Agent to evaluate complex image editing tasks.
- The framework employs a dual-track CogRetriever pipeline for data curation and an Active Elo protocol that integrates CV-Judge with selective human expert supervision to ensure reliable model rankings.
- CV-Agent demonstrates that closed-loop reasoning, incorporating planning, editing, and verification, significantly improves instruction adherence and constraint satisfaction in professional visual editing workflows.

---

[Benchmarking Security Risk Detection and Verification in Open Agentic Skill Ecosystems](http://arxiv.org/abs/2606.00925)

- SkillVetBench: introduces a two-stage security vetting framework that combines LLM-as-a-Judge for semantic analysis with Sandbox Execution for runtime verification of agent skills.
- The framework evaluates skills across seven vulnerability categories by analyzing both static artifacts and dynamic execution traces to identify malicious behaviors that evade traditional scanners.
- Experimental results demonstrate that SkillVetBench achieves zero false negatives on confirmed malicious skills by grounding security verdicts in observable runtime evidence rather than static suspicion alone.

---

[Banyan: a procedurally-generated environment for studying the effect of task diversity on transfer](http://arxiv.org/abs/2606.00880)

- Banyan: introduces a GPU-accelerated continual RL domain that enables parametric control over task layouts, tree topologies, and object instances to study the impact of task diversity on systematic transfer.
- The framework demonstrates that while increasing task diversity improves forward and backward transfer across single distribution shifts, excessive diversity can paradoxically inhibit long-run continual learning by stalling specialization.
- Experimental results across discrete grid-world and continuous control substrates indicate that diversity shapes generalized features, yet interference between tasks limits performance in multi-distribution sequences.

---

[Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI](http://arxiv.org/abs/2606.00866)

- MORI: introduces a program-aware scheduler that optimizes KV cache placement by ranking agentic programs on a continuous relative idleness spectrum across GPU HBM and CPU DRAM tiers.
- The framework utilizes a windowed idleness metric to dynamically demote idle programs to CPU DRAM and promote busy programs to GPU HBM, effectively managing memory capacity and reducing TTFT.
- MORI enforces admission control at both memory tiers and maintains cache affinity in multi-replica deployments to minimize recomputation and improve overall system throughput.

---

[GenPT: Beyond Self-Report for Reliable LLM Psychometrics via Generative Projective Testing](http://arxiv.org/abs/2606.00860)

- GenPT: introduces a three-stage psychometric pipeline that utilizes projective testing to bypass data contamination and social desirability bias in LLMs.
- The framework employs an Examinee agent to respond to ambiguous stimuli, an Interpreter to quantify behavioral outputs, and a Diagnostician to infer personality traits or mental health risks.
- GenPT demonstrates superior resistance to framing-induced bias and higher sensitivity to longitudinal context compared to traditional self-report questionnaires.

---

[MOMENTO: Evaluating Persistent Memory and Reasoning with Multi-Session Agentic Conversations](http://arxiv.org/abs/2606.00832)

- MOMENTO: introduces a benchmark framework for evaluating persistent, multi-session agentic task completion by integrating User Agent, Assistant Agent, Toolbox, Agentic Memory, Multi-Session Context, Dialogue History, and Database components.
- The framework utilizes a hybrid memory module that combines long-term SQL-based retrieval for cross-session dependencies with short-term structured compression to maintain context within a 128k-token budget.
- Experimental results demonstrate that LLMs often fail in multi-session environments due to misestimating user state and over-relying on historical context instead of performing necessary verification through tool use.

---

[RoboStressBench: Benchmarking VLM Robustness to Physical Visual Stress in Embodied Scenes](http://arxiv.org/abs/2606.00828)

- RoboStressBench: introduces a benchmark for evaluating VLM robustness under physical visual stress in embodied scenes, categorized by Material, Viewpoint, Lighting, and Geometry dimensions.
- The framework includes StressDART, a modular agentic solver that utilizes a Stress Detector to identify stressors and a Stress Rectifier to perform targeted visual editing before the Reasoner processes the input.
- Experimental results across 16 VLMs demonstrate that physical visual stress significantly degrades performance, and that explicit diagnosis combined with test-time rectification improves robustness without requiring model fine-tuning.

---

[Beyond Independent Manipulation: Individual Fairness-aware Strategic Classification with Peer Imitation](http://arxiv.org/abs/2606.00827)

- IFSC: introduces a strategic classification framework that models interdependent agent behavior through peer imitation to satisfy individual fairness constraints.
- The framework replaces the standard assumption of independent agent manipulation with a peer-driven mechanism where agents adjust features to resemble nearby positively decided peers.
- IFSC employs a robust learning process using stochastic perturbations of visible peer sets to optimize classifiers against uncertain and population-dependent strategic responses.

---

[Partial Fairness Awareness: Belief-Guided Strategic Mechanism for Strategic Agents](http://arxiv.org/abs/2606.00826)

- PFA (Partial Fairness Awareness): introduces a strategic classification framework that mitigates fairness reversal and welfare loss by partially exposing fairness constraints to agents through a belief-guided mechanism.
- The framework utilizes a Bayesian update mechanism where agents iteratively refine their belief distribution over a candidate set of fairness constraints based on feedback from the Decision System.
- Theoretical and empirical results demonstrate that PFA achieves lower group fairness gaps and higher social welfare compared to fully public or fully private fairness regimes.

---

[SuperMemory-VQA: An Egocentric Visual Question-Answering Benchmark for Long-Horizon Memory](http://arxiv.org/abs/2606.00825)

- SuperMemory-VQA: introduces a multi-modal egocentric VQA benchmark for evaluating AI assistants on practical, long-horizon memory tasks, utilizing an agentic pipeline comprising Audio Extraction, LLM Captioning Agent, Person Registry, Super Ledger, QA Planner, Verifier, Retriever, Enhancer, Accepted Set, and Rejected Set.
- The framework employs a human-in-the-loop annotation pipeline where the QA Planner generates grounded Q&A pairs, which are then rigorously validated by the Verifier and Retriever agents against factual and causal criteria.
- Benchmarking results demonstrate that current agentic frameworks and LLMs struggle with answerability detection, long temporal gaps, and multi-moment evidence integration, highlighting the need for more robust architectures for grounded AI memory.

---

[SkillPager: Query-Adaptive Intra-Skill Navigation via Semantic Node Retrieval](http://arxiv.org/abs/2606.00822)

- SkillPager: introduces a two-stage framework that parses procedural skill documents into typed semantic nodes and performs global query-adaptive MMR selection to construct compact, execution-sufficient contexts.
- The framework utilizes a Semantic Parsing Pipeline to categorize document fragments into six roles, enabling the MMR Selection component to balance relevance and diversity while minimizing token consumption.
- By employing a Dynamic Budget and Context Assembly, SkillPager achieves statistical non-inferiority to full-document prompting while significantly reducing the token overhead for LLM agents.

---

[Not All Flips Are Conformity: Decomposing Stance Convergence in Multi-Agent LLM Debate](http://arxiv.org/abs/2606.00820)

- Three-source decomposition framework: introduces a methodology to isolate spontaneous instability, stance-induced conformity, and reasoning-induced persuasion in multi-agent LLM debate.
- The framework utilizes controlled counterfactual conditions to demonstrate that approximately 40% of apparent peer influence is actually spontaneous model instability.
- The research reveals that harmful conformity is predictable from initial disagreement structures and that reasoning-like presentation significantly influences LLMs regardless of logical content.

---

[Interaction-Centered Intelligence: Toward Interaction as the Primary Unit of Analysis in Co-Creative AI and Human-AI Systems](http://arxiv.org/abs/2606.00807)

- ICI (Interaction-Centered Intelligence): introduces a framework that shifts the primary unit of analysis from isolated internal computation to interaction dynamics occurring between humans and AI systems.
- The framework synthesizes distributed cognition, embodiment, enaction, and participatory sense-making to model intelligence as an emergent relational phenomenon unfolding through time.
- It operationalizes this shift through computational methods including activity traces, creative trajectories, and sense-making curves to quantify collaborative dynamics in co-creative AI.

---

[Dynamic Coordination Strategy Selection for Enterprise Multi-Agent Systems](http://arxiv.org/abs/2606.00804)

- FAOS (Foundation AgenticOS): introduces a dynamic coordination strategy selection framework that evaluates the performance of Single-agent, Consensus, Debate, and Synthesis patterns across diverse enterprise problem classes.
- The research demonstrates that while strict pre-registered winner-selection laws fail, dynamic routing serves as a reliable near-best heuristic for enterprise LLM deployments.
- Empirical results indicate that structured compliance tasks benefit from Single-agent execution, whereas conflicting-objective tasks require specific routing to either Debate or Consensus/Synthesis based on the nature of the decision.

---

[Behavior-Invariant Task Representation Learning with Transformer-based World Models for Offline Meta-Reinforcement Learning](http://arxiv.org/abs/2606.00780)

- MetaSTAR: introduces a framework that leverages a stochastic Transformer-based world model to learn behavior-invariant task representations for offline meta-reinforcement learning.
- The framework utilizes a world model to capture task-relevant dynamics while suppressing behavior-policy-specific correlations, effectively mitigating context distribution shift.
- MetaSTAR combines contextual imagination with a conservative value objective to stabilize policy learning and improve generalization under sparse-reward and out-of-distribution settings.

---

[FALAT: Tracing Failures in LLM Agent Trajectories via Dependency-Guided Search](http://arxiv.org/abs/2606.00765)

- FALAT: introduces a diagnostic framework that reformulates failure attribution in LLM agent trajectories as a hierarchical dependency-guided search problem.
- The framework utilizes an External Prior and a Hierarchical Trajectory Representation to prune candidates through a Typed Dependency Structure, distinguishing between error origination and propagation.
- FALAT employs Counterfactual Verification and Local Re-Search to identify the decisive error step and responsible agent, consistently outperforming existing baselines on the Who&When benchmark.

---

[STEM: Semantic Target Search and Exploration using MAVs in Cluttered Environments](http://arxiv.org/abs/2606.00762)

- STEM: introduces a framework for semantically-guided target search and exploration in 3D environments using MAVs, integrating Semantic Priority Masking, Active Perception, and a Target Search Planner.
- The framework utilizes a Large Language Model to infer semantic relationships, which are then propagated into 3D space to prioritize exploration frontiers likely to contain the target.
- By formulating the target search as a Weighted Minimum Latency Problem, the planner effectively balances coverage-based exploration with semantic guidance to minimize target discovery time.

---

[Distributed GNEP Algorithms without Multiplier Sharing and Applications to Multi-Robot Coordination and Contextual Bandit-Based Active Learning](http://arxiv.org/abs/2606.00759)

- CAAL (Contextual Adaptive Active Learning): introduces a fully distributed continuous-time algorithm for solving Generalized Nash Equilibrium Problems (GNEPs) without requiring multiplier consensus, alongside a contextual bandit-based framework for adaptive active learning.
- The GNEP approach utilizes primal-dual dynamics within a projected dynamical system to achieve convergence while preserving agent privacy and reducing communication overhead.
- The active learning component employs a contextual bandit meta-learner to dynamically select optimal hand-crafted strategies based on reward prediction, improving robustness across varying batch sizes in industrial settings.

---

[CoMIC: Collaborative Memory and Insights Circulation for Long-Horizon LLM Agents in Cloud-Edge Systems](http://arxiv.org/abs/2606.00756)

- CoMIC: introduces a parameter-update-free cloud-edge framework that enhances long-horizon decision-making for lightweight LLMs by offloading memory maintenance and reflection to a cloud-side critic.
- The framework utilizes a "Centralized Reflection, Decentralized Execution" design where edge agents perform local subgoal-oriented tasks while the cloud asynchronously aggregates cross-agent experiences into reusable global guidance.
- CoMIC improves task success rates and grounding accuracy for resource-constrained agents by dynamically injecting distilled insights into prompts without requiring model fine-tuning.

---

[I-WebGenBench : Evaluating Interactivity in LLM-Generated Scientific Web Applications](http://arxiv.org/abs/2606.00750)

- I-WebGenBench: introduces a benchmark for evaluating the ability of LLMs to generate interactive scientific web applications from paper-derived specifications, utilizing App Generation & Execution, Rendered Page Analysis, Interaction Probe, Semantic Action Mapping, DOM Mutation Observation, Build Success Rate (BSR), Interaction Rate (IR), and VLM-as-a-Judge.
- The framework decouples compilability from interactivity by employing a deterministic Interaction Probe that triggers DOM mutations to assess functional responsiveness.
- Empirical results reveal a persistent gap between high build success rates and lower interaction rates, highlighting that current LLMs struggle to implement coherent event-driven logic despite achieving visual completeness.

---

[SkyShield: Occupancy as a Safety Interface for Low-Altitude UAV Autonomy](http://arxiv.org/abs/2606.00747)

- SkyOcc: introduces a geometry-first monocular baseline for low-altitude UAV semantic occupancy prediction that utilizes Attitude-aware Spatial Projection, Spatiotemporal Pillar Encoder, and Safety-prior Voxel Optimization to ensure safety-critical spatial understanding.
- The framework addresses the challenges of dynamic 6-DoF UAV motion by integrating frame-wise attitude directly into the projection chain, rather than relying on static-rig assumptions.
- SkyShield provides a benchmark with 36K samples and the KAR-mIoU metric, which penalizes occupancy prediction errors based on kinematic reachability and time-to-collision to prioritize flight safety.

---

[EMA: Approximate Nearest Neighbor Search with General Attribute Filtering and Dynamic Updates](http://arxiv.org/abs/2606.00734)

- EMA: introduces a filtering ANN algorithm that supports multi-predicate queries over mixed numerical and categorical attributes by augmenting graph edges with Marker, Codebook, and diversity-aware pruning.
- The framework utilizes an edge recovery mechanism to maintain graph navigability under low-selectivity and off-cluster query scenarios, ensuring robust search performance.
- EMA supports efficient dynamic updates through a patch mechanism that enables incremental insertions, attribute modifications, and local graph repairs without requiring frequent full index rebuilds.

---

[Higher-order Network Analysis of Human Mobility Data](http://arxiv.org/abs/2606.00733)

- Higher-order Network Analysis Framework: introduces a methodology for evaluating the validity of synthetic mobility datasets by comparing them against observed GPS trajectories using higher-order network representations.
- The framework utilizes empirical de Bruijn graphs and multi-order models to capture sequential dependencies and memory effects inherent in human mobility, contrasting these with memoryless random walk baselines.
- By applying this approach to the NetMob 2025 dataset and MATSim simulations, the research identifies key divergences in path length distributions, node visitation patterns, and predictability, highlighting limitations in current simulation paradigms.

---

[Multi-Agent Conformal Prediction with Personalized Statistical Validity](http://arxiv.org/abs/2606.00717)

- PFWCP: introduces a framework that corrects for bias and variance induced by data heterogeneity in federated settings while preserving privacy and requiring only lightweight communication.
- The framework combines density-ratio-based weighting of conformal scores at each agent with a weighted-quantile-of-quantiles aggregation at the server to provide asymptotically valid marginal and calibration-conditional coverage guarantees.
- PFWCP supports both standard and one-shot federated protocols, demonstrating improved calibration quality and efficiency over existing federated conformal prediction baselines across various synthetic and real-world datasets.

---

[The Cartan-Topos Protocol: A Unified Geometric and Categorical Framework for Resilient Multi-Agent Coordination](http://arxiv.org/abs/2606.00714)

- Cartan-Topos Protocol: introduces a unified geometric and categorical framework for resilient multi-agent coordination by bridging continuous control theory and discrete symbolic logic through Homogeneous Manifolds, Clifford Algebra, Cellular Sheaves, Sheaf Laplacian, Cartan Connection, Sheaf-Theoretic Planning (STP), Grothendieck Topoi, and Actor Model (Erlang/OTP).
- The framework utilizes cellular sheaves and the Cartan connection to model complex network interactions and logical dependencies, ensuring harmonic consistency across heterogeneous agent states.
- Asynchronous nonlinear sheaf diffusion guarantees robust convergence to global sections under partial asynchrony, while STP enables resilient temporal reasoning and abductive repair within a categorical topos-theoretic structure.

---

[MOSAIC: Modular Orchestration for Structured Agentic Intelligence and Composition](http://arxiv.org/abs/2606.00708)

- MOSAIC: introduces a structured agentic framework that bridges AutoML pipelines and unconstrained LLM-based code generation by utilizing EDA & Feature Engineering, Case Retrieval & Model Selection, Model Generation, Code Refinement, Knowledge Banks, RL Policy, and LLM Executor.
- The framework constructs executable modelling workflows by retrieving prior cases and source-code modules, grounding LLM-based code generation in a structured blueprint rather than unconstrained synthesis.
- MOSAIC formulates model refinement as a failure-aware offline reinforcement learning problem, utilizing trajectory branching and invalid-action masking to learn long-horizon refinement policies for LLM-mediated code editing.

---

[VICR: Visual In-Context Restoration for Real-World Image Super-Resolution](http://arxiv.org/abs/2606.00704)

- VICR: introduces a Diffusion Transformer-based framework that reformulates Real-ISR as visual in-context completion by decoupling local visual evidence, global image context, and semantic prompts.
- The framework utilizes a diptych for local structural guidance, a Vision-Context Bridge for global scene stabilization, and an inference-time agent to iteratively refine semantic prompts.
- By explicitly assigning distinct roles to heterogeneous condition streams, the model achieves state-of-the-art performance with 127M trainable parameters while reducing structural drift and semantic inconsistencies.

---

[Shape-Prior-Based Point Cloud Completion for Single-Stage Fully Sparse 3D Object Detection](http://arxiv.org/abs/2606.00688)

- SPPCC: introduces a shape-prior-based point cloud completion method designed to enhance single-stage fully sparse 3D object detectors by addressing point cloud sparsity and incompleteness.
- The framework utilizes an Instance Selection module to isolate foreground objects and an Alignment-Based Point Completion module to refine object shapes using category-specific prototypes.
- By integrating these modules, the approach improves detection performance on occluded objects without relying on computationally expensive proposal boxes or dense representations.

---

[NeuroLog: Reasoning You Can Audit — Neuro-Symbolic Vulnerability Discovery via LLM Facts, Datalog, and SMT](http://arxiv.org/abs/2606.00669)

- NeuroLog: introduces an end-to-end, compile-free vulnerability analysis pipeline that integrates Tree-sitter, LLM-smell-pass, Soufflé-Datalog-mesh, Z3-Symbex, LLM-synthesis-agent, and AddressSanitizer-harness to perform auditable security reasoning.
- The framework leverages LLMs for function-level fact extraction and crash synthesis, while delegating cross-function dataflow analysis to a Datalog rule mesh and path feasibility checking to an SMT solver.
- By avoiding full compilation and utilizing a tiered, auditable reasoning process, the system enables security analysts to perform vulnerability discovery on unfamiliar codebases with minimal setup time.

---

#### 28th May 2026

[Transferable Reinforcement Learning via Probabilistic Latent Embeddings and Dynamic Policy Adaptation for Sim-to-Real Deployment](http://arxiv.org/abs/2605.27659)

- Transferable Reinforcement Learning framework: introduces a unified CMDP-based approach that integrates probabilistic latent context variable adaptation with distributional RL to enable safe and efficient policy transfer under Sim2Real gaps.
- The framework utilizes an Environment Encoder to infer latent context variables and employs a Dynamic Adaptation Module to adjust risk-sensitive policies at inference time, ensuring safety constraints are met during real-world deployment.
- By leveraging distributional RL and a principled safety upper-bound, the method achieves a favorable reward-cost trade-off, maintaining deployment costs below thresholds while adapting to unseen environments.

---

#### 27th May 2026


[AI, Take the Wheel: What Drives Delegation and Trust in Human–Computer Cooperative Question Answering?](http://arxiv.org/abs/2605.28255)

- Human-AI Cooperative Trivia Competition framework: introduces a competitive trivia-based benchmark to evaluate human reliance on LLMs through proactive delegation and deliberative adoption decisions.
- The framework utilizes a two-stage question-answering process where humans manage AI involvement via muting and evaluate AI-generated explanations to reach final consensus.
- The study identifies that while collaboration is synergistic, humans often miscalibrate trust due to surface-level explanation features, highlighting the need for evidence-grounded AI explanations.

---



[Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players](http://arxiv.org/abs/2605.28816)

- Gamma-World: introduces a generative multi-agent world model that utilizes Simplex Rotary Agent Encoding for permutation-symmetric agent identity and Sparse Hub Attention for efficient cross-agent communication.
- The framework employs a Causal Multi-Agent DiT student model distilled from a bidirectional teacher to enable real-time, action-responsive video generation at 24 FPS.
- By leveraging hub tokens and simplex-based encoding, the model achieves scalable multi-agent interaction that generalizes from two to four players without additional training.

---

[Self-Improving Language Models with Bidirectional Evolutionary Search](http://arxiv.org/abs/2605.28814)

- BES (Bidirectional Evolutionary Search): introduces a search framework that couples forward candidate evolution with backward goal decomposition to improve LLM reasoning performance.
- The framework utilizes four evolution operators—combination, translocation, deletion, and crossover—to generate diverse candidates beyond the model's own distribution.
- Backward search recursively decomposes complex tasks into verifiable sub-goals, providing dense intermediate feedback that guides the forward search process more efficiently than standard methods.

---

[Calibrating Conservatism for Scalable Oversight](http://arxiv.org/abs/2605.28807)

- CCO (Calibrated Collective Oversight): introduces a deployment-time framework that aggregates diverse auxiliary overseers into a penalty to constrain agentic behavior while using a conformal controller to calibrate conservatism online.
- The framework utilizes a primary agent, candidate actions, auxiliary overseers, aggregate evaluation, a conformal controller, and a conservative baseline to ensure long-run violation rates converge to a user-specified target.
- CCO provides formal safety guarantees for sequential, agentic settings without requiring distributional assumptions, effectively balancing utility against safety constraints.

---

[Personal Visual Memory from Explicit and Implicit Evidence](http://arxiv.org/abs/2605.28806)

- VISUALMEM: introduces a hybrid visual–text architecture that augments a text-memory backend with a structured visual memory module to store and reason over persistent personal visual information.
- The framework utilizes a two-stage contextual decision process, including context-guided interpretation and deferred commitment, to resolve identity, ownership, and latent facts from multimodal interactions.
- Experiments demonstrate that VISUALMEM significantly outperforms prior memory systems on a new synthetic benchmark for personal visual memory while maintaining compatibility with standard text-centric benchmarks.

---

[OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration](http://arxiv.org/abs/2605.28805)

- OmniVerifier-M1: introduces a multimodal meta-verification framework that leverages symbolic outputs for fine-grained error localization and decoupled reinforcement learning to improve verifier training stability.
- The framework utilizes a Verifier Agent to produce spatial bounding boxes and semantic edit instructions, which guide a UMM Agent in performing iterative, region-level self-correction of generated images.
- By decoupling binary judgment and meta-verification objectives, the system avoids the sparse and entangled reward signals typical of joint training, resulting in more robust and efficient optimization.

---

[Do Agents Need Semantic Metadata? A Comparative Study in Agentic Data Retrieval](http://arxiv.org/abs/2605.28787)

- ADK: introduces a comparative study of agentic data retrieval by contrasting a Baseline Agent and a Semantic Agent across unstructured and structured web environments.
- The research utilizes an LLM-as-a-judge pipeline, incorporating Relevance-, Data Accessibility- and Dataset Page Type-autoraters, to evaluate agent performance against FAIR principles.
- Results demonstrate that while unstructured retrieval offers broader exploratory coverage, structured metadata ecosystems are essential for reliable, machine-actionable autonomous workflows.

---

[Learn from Weaknesses: Automated Domain Specialization for Small Computer-Use Agents](http://arxiv.org/abs/2605.28775)

- LEARNWEAK: introduces an annotation-free specialization framework for small CUAs that uses a stronger reference agent to identify student weaknesses, synthesize targeted tasks, and construct supervision automatically.
- The framework utilizes LEARNWEAK-GEN to iteratively construct a weakness-aware dataset and LEARNWEAK-DPO to perform error-aware preference optimization, distinguishing between planning and execution failures.
- By employing modular LoRA adapters, the approach enables efficient domain-specific specialization for small CUAs without requiring human annotation, significantly outperforming existing autonomous trajectory generation baselines.

---

[Agent Explorative Policy Optimization for Multimodal Agentic Reasoning](http://arxiv.org/abs/2605.28774)

- AXPO (Agent eXplorative Policy Optimization): introduces a reinforcement learning algorithm that addresses the Thinking-Acting Gap by fixing the thinking prefix and performing tool-call resampling to concentrate learning signals on under-trained tool-using behaviors.
- The framework utilizes uncertainty-based prefix ranking to identify failed tool-using rollouts and applies a per-prefix GRPO advantage calculation to ensure that successful resampled continuations provide positive gradient signals to the model.
- By decoupling the thinking prefix from the tool-call continuation, AXPO effectively recovers all-wrong tool-using subgroups and improves agentic reasoning performance across multimodal benchmarks with significantly fewer parameters than larger baselines.

---

[Rethinking Memory as Continuously Evolving Connectivity](http://arxiv.org/abs/2605.28773)

- FluxMem: introduces a connectivity-evolving memory framework that models memory as a heterogeneous graph and progressively refines its topology through Semantic Knowledge Layer, Episodic Experiences Layer, Procedural Skills Layer, Initial Connection Formation, Feedback-Driven Refinement, Long-Term Consolidation, PEMS, LLM-based verification, and Subgraph Gt(q).
- The framework dynamically constructs a task-specific Subgraph Gt(q) by integrating factual knowledge, past trajectories, and distilled reasoning templates to optimize LLM agent performance.
- FluxMem employs a three-stage evolutionary pipeline that repairs missing links, prunes interference, and aligns abstraction granularity to transform static memory into a self-optimizing connectivity substrate.

---

[SwarmHarness: Skill-Based Task Routing via Decentralized Incentive-Aligned AI Agent Networks](http://arxiv.org/abs/2605.28764)

- SwarmHarness: introduces a decentralized protocol for skill-based task routing that organizes autonomous compute nodes into a self-regulating swarm without central authority.
- The architecture utilizes a SwarmRegistry for peer discovery, a SwarmRouter for utility-based task dispatching, and a SwarmCredit Ledger for fair, Shapley-value-based incentive attribution.
- By employing digital pheromones for routing signals, the network achieves emergent collective intelligence and load balancing while maintaining backward compatibility with existing HarnessAPI deployments.

---

[Extrapolative Weight Averaging Reveals Correctness–Efficiency Frontiers in Code RL](http://arxiv.org/abs/2605.28751)

- Extrapolative Weight Averaging: introduces a method to navigate and extend the correctness-efficiency frontier in code RL by linearly interpolating and extrapolating between RL checkpoints trained under nested unit-test coverage.
- The paper demonstrates that RL checkpoints trained with varying verifier strictness form a stable correctness-efficiency frontier, where interpolation recovers this frontier and extrapolation extends it to new, complementary policies.
- Ensembling these extrapolated checkpoints provides broader solved-set coverage and improves performance on hard competitive programming benchmarks by leveraging the diversity of policies along the extended frontier.

---

[Execution and assessment of agentic influence operations in simulated social networks](http://arxiv.org/abs/2605.28725)

- OSN Simulation Framework: introduces an agent-based simulation environment for evaluating AI-enabled influence operations by integrating an OSN simulator with a Red Module for adversarial campaign execution.
- The framework utilizes LLMs to generate organic content and execute three distinct influence workflows—Narrative Release, Narrative Support, and Counter-Narrative Reaction—across defined DISARM phases.
- Experimental results demonstrate that reactive strategies, particularly Counter-Narrative Reaction, achieve superior reach and belief change compared to proactive volume-driven approaches.

---

[LiveBrowseComp: Are Search Agents Searching, or Just Verifying What They Already Know?](http://arxiv.org/abs/2605.28721)

- LiveBrowseComp: introduces a deep-search benchmark designed to evaluate LLMs beyond their intrinsic knowledge coverage by using recent, long-tail facts that require genuine evidence-driven discovery.
- The framework identifies Intrinsic Knowledge Dependence (IKD), where LLMs use search tools primarily to verify internally generated hypotheses rather than to discover new information.
- Experimental results demonstrate that when the memory-backed verification shortcut is removed, LLM performance drops significantly and rankings shift, highlighting the need for dynamic, time-sensitive evaluation benchmarks.

---

[TRACER: Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative Multi-LLM Reasoning](http://arxiv.org/abs/2605.28699)

- TRACER: introduces a two-layer framework for cooperative multi-LLM reasoning that integrates Controller-Regret Layer and Generation-Credit Layer to optimize collaborative decision-making and utterance generation.
- The framework utilizes a Controller-Regret Layer to manage turn-level speaking decisions via regret matching and a Generation-Credit Layer to refine agent outputs using role-specific GSPO rewards.
- By modeling multi-agent collaboration as a game-theoretic process with binary actions, TRACER achieves mathematically rigorous convergence and stable training dynamics while maintaining low inference costs.

---

[VeriTrip: A Verifiable Benchmark for Travel Planning Agents over Unstructured Web Corpora](http://arxiv.org/abs/2605.28683)

- VeriTrip: introduces a verifiable benchmark for evaluating retrieval-based planning agents in unstructured, multimodal web environments using a Multimodal Retrieval Base (MRB) and a Verifiable Knowledge Base (VKB).
- The framework employs a cell-wise factual verification protocol to decouple logical reasoning from parametric hallucinations in LLMs.
- Experimental results across various MLLMs reveal a perception-reasoning trade-off where high cognitive load from multimodal retrieval interferes with instruction following and preference fulfillment.

---

[An LLM-Based Assistance System for Intuitive and Flexible Capability-Based Planning](http://arxiv.org/abs/2605.28666)

- Hybrid Planning Assistance System: introduces a routed agentic workflow that augments formal SMT-based planning with LLM-based agents for natural-language interaction, explanation, and model adaptation.
- The architecture utilizes a Router Agent, Knowledge Retrieval Agent, Capability Mapper Agent, Planning Agent, Analyze and Adaptation Agent, and Repair Agent to manage the planning lifecycle while ensuring formal correctness via an SMT-Planner.
- The system incorporates Human-in-the-Loop (HitL) checkpoints to ensure all knowledge graph modifications are explicitly validated by the user before re-planning.

---

[AUTOSCIENTISTS: Self-Organizing Agent Teams for Long-Running Scientific Experimentation](http://arxiv.org/abs/2605.28655)

- AUTOSCIENTISTS: introduces a decentralized multi-agent framework that enables autonomous, long-running scientific experimentation through self-organizing teams and a shared experimental state.
- The framework coordinates research by allowing agents to dynamically form teams, critique proposals via a shared forum, and maintain a champion model while tracking failures in a dead-end registry.
- AUTOSCIENTISTS improves experimental efficiency and performance across biomedical machine learning, protein fitness prediction, and LLM training optimization by replacing central planners with decentralized, self-organizing agent interactions.

---

[MaskClaw: Edge-Side Personalized Privacy Arbitration for GUI Agents with Behavior-Driven Skill Evolution](http://arxiv.org/abs/2605.28646)

- MaskClaw: introduces an edge-side privacy arbitration framework that uses local perception, rule-based policy arbitration, and SafeScreenshot construction to protect sensitive GUI information before it reaches downstream agents.
- The framework incorporates a feedback-driven skill evolution loop that utilizes a sandbox judge gate to refine and validate privacy policies based on user corrections and interaction traces.
- MaskClaw leverages the P-GUI-Evo benchmark to evaluate personalized privacy decisions across various personas, tasks, and UI patterns, ensuring that privacy handling remains context-aware and auditable.

---

[Mobile-Aptus: Confidence-Driven Proactive and Robust Interaction in MLLM-based Mobile-Using Agents](http://arxiv.org/abs/2605.28629)

- Mobile-Aptus: introduces a universal confidence integration framework that enables MLLM-based mobile agents to autonomously determine the necessity of human intervention by combining supervised fine-tuning and direct preference optimization.
- The framework utilizes a confidence bias correction strategy that employs semantic similarity retrieval to construct preference pairs, effectively mitigating over-soliciting and over-execution issues in mobile automation.
- Experimental results across four benchmarks demonstrate that Mobile-Aptus achieves state-of-the-art performance and provides more accurate intervention timing compared to existing interactive mobile-using agents.

---

[LACUNA: Safe Agents as Recursive Program Holes](http://arxiv.org/abs/2605.28617)

- LACUNA: introduces a programming model for LLM agents that treats agent actions as typed holes in the host program, ensuring safety by compiling model-generated code against the surrounding lexical scope before execution.
- The framework leverages the host language's compiler to enforce static type safety, capability constraints, and information-flow control on untrusted model-generated code.
- By integrating model calls as recursive, type-checked primitives, LACUNA enables complex agent patterns like ReAct loops and multi-model planning to be expressed as ordinary program control flow.

---

[Adaptive Multimodal Agents-Based Framework for Automatic Workflow Execution](http://arxiv.org/abs/2605.28607)

- Adaptive Multimodal Agents-Based Framework: introduces a two-phase pipeline that constructs a topological knowledge base from execution logs and utilizes LLM-powered agents with closed-loop verification to execute complex workflows.
- The framework integrates an Adaptive Graph-based RAG pipeline for knowledge augmentation and a Multi-Agent Execution System comprising planning-, observation-, decision- and verification-agents.
- By replacing linear episodic memory with a graph-based representation and a goal-aware semantic history, the architecture enables agents to navigate non-stationary interfaces with improved reliability and reduced logical hallucinations.

---

[Evaluating the Realism of LLM-powered Social Agents: A Case Study of Reactions to Spanish Online News](http://arxiv.org/abs/2605.28598)

- LLM-powered Social Agents Framework: introduces a validation methodology for assessing whether LLMs can reproduce the distributional, affective, and harmfulness properties of real human discourse in online news reactions.
- The framework utilizes a matched evaluation setting where LLMs generate reactions conditioned on synthetic user profiles and real news stimuli, which are then compared against empirical benchmarks using hate-speech detection, sentiment analysis, and MAUVE-based semantic alignment.
- Results demonstrate that while LLMs can generate fluent text, they often fail to capture the statistical distribution of real audience behavior, with fine-tuning providing uneven improvements across different model families.

---

[Technical Report: Exploring the Emerging Threats of the Agent Skill Ecosystem](http://arxiv.org/abs/2605.28588)

- MCP-scan: introduces a security analysis framework for AI agent skills that combines customized LLM judges and deterministic rules to detect malicious patterns and vulnerabilities.
- The research evaluates 3,984 agent skills, identifying critical security threats including prompt injection, malicious code execution, and improper credential handling.
- The study provides a threat taxonomy and highlights the urgent need for automated security pipelines in AI agent marketplaces to mitigate risks from evolving attack vectors.

---

[Deformable Gaussian Occupancy: Decoupling Rigid and Nonrigid Motion with Factorized Distillation](http://arxiv.org/abs/2605.28587)

- DeGO: introduces a deformable Gaussian occupancy framework that decouples rigid and nonrigid motion using a learnable adaptive rigid-body mask and factorized 4D foundation-model distillation.
- The framework integrates a Decoupled Gaussian Deformation module to selectively allocate deformation capacity and a Factorized Feature Distillation module to align Gaussian features with spatiotemporal cues from a VGGT teacher.
- By leveraging foundation-aligned 4D representations, the approach improves human-centric and overall scene understanding in weakly supervised occupancy prediction without increasing inference costs.

---

[A Matter of TASTE: Improving Coverage and Difficulty of Agent Benchmarks](http://arxiv.org/abs/2605.28556)

- TASTE (Task Synthesis from Tool Sequence Evolution): introduces an automated method for generating challenging agent benchmarks by sampling diverse tool sequences using an Adaptive Contrastive n-gram model, selecting representative sequences via K-medoids clustering, and instantiating them into tasks refined by a task evolution pipeline.
- The framework utilizes a Verifier agent to ensure task solvability and employs adversarial techniques within the task evolution pipeline to increase interaction difficulty and broaden tool-use coverage.
- By reversing the traditional task construction process, TASTE generates benchmarks that expose performance gaps in LLMs that appear saturated on existing benchmarks, while also providing broader coverage of tool-use patterns.

---

[A Minimal Executable Proof for Multi-Language Contract Traceability](http://arxiv.org/abs/2605.28546)

- DAG-TOML: introduces a structured framework for contract-centered traceability by linking source implementations to verifiable executable witnesses and review gates.
- The framework utilizes an implementation DAG to organize source artifacts and automated validators to enforce runtime and source-analysis constraints.
- This approach provides a falsifiable, inspectable proof pack that maps engineering claims directly to executable scripts and evidence artifacts.

---

[GUI-CIDER: Mid-training GUI Agents via Causal Internalization and Density-aware Exemplar Reselection](http://arxiv.org/abs/2605.28534)

- GUI-CIDER: introduces a mid-training method that explicitly internalizes GUI world knowledge into LLMs through Causal Internalization and Density-aware Exemplar Reselection.
- The framework utilizes a Data Synthesis Pipeline to distill static planning and dynamic causal knowledge from raw trajectories into structured textual rationales.
- An Exemplar Reselection module filters the synthesized corpus by rewarding causal structures and penalizing semantic redundancy to ensure high-quality training data for the LLM.

---

[Do Agents Know What They Can’t Do? Evaluating Feasibility Awareness in Tool-Using Agents](http://arxiv.org/abs/2605.28532)

- FeasiGen: introduces an automated pipeline for constructing infeasible agent tasks by identifying and masking critical execution dependencies required for successful task completion.
- The framework evaluates LLM feasibility awareness by measuring false continue rates and token consumption when agents encounter tasks missing essential tools.
- Experimental results across nine models demonstrate that multi-agent architectures significantly improve infeasibility detection compared to single-agent systems, reducing unnecessary token waste.

---

[Do LLMs Favor Their Providers? Measuring Vertical Integration Bias in Code Generation](http://arxiv.org/abs/2605.28515)

- VIBENCH: introduces a benchmark to measure Vertical Integration Bias (VIB) in LLM-generated code by comparing provider-affiliated models against non-affiliated controls across 20 software-integration scenarios.
- <arxiv_paper_date>27th May 2026</arxiv_paper_date>
<arxiv_paper_name>Do LLMs Favor Their Providers? Measuring Vertical Integration Bias in Code Generation</arxiv_paper_name>
<arxiv_paper_id>2605.28515</arxiv_paper_id>
<arxiv_paper_link>http://arxiv.org/abs/2605.28515</arxiv_paper_link>
<arxiv_paper_name>VIBENCH (Vertical Integration Bias Benchmark):</arxiv_paper_name>

<arxiv_paper_framework_1>VIBENCH/LiteLLM/OpenCode/OpenAI Agents SDK/Attribution Heuristics/VIB Estimator</arxiv_paper_framework_1>

<arxiv_paper_framework_2>VIBENCH (benchmark for measuring provider bias)/LiteLLM (routing interface for provider APIs)/OpenCode (provider-independent agent runtime)/OpenAI Agents SDK (alternative agent runtime)/Attribution Heuristics (keyword-based detection of provider services)/VIB Estimator (statistical method for calculating bias scores)</arxiv_paper_framework_2>

<arxiv_paper_description_1>VIBENCH: introduces a benchmark to measure Vertical Integration Bias (VIB) in LLM-generated code by comparing provider-affiliated models against non-affiliated controls across 20 software-integration scenarios.</arxiv_paper_description_1>

<arxiv_paper_description_2>The framework utilizes LiteLLM for standardized API access and OpenCode or OpenAI Agents SDK as agentic runtimes to evaluate how provider-affiliated LLMs favor their own ecosystems in direct and agentic workflows.</arxiv_paper_description_1>

<arxiv_paper_description_3>The research demonstrates that VIB is present in direct code generation, significantly amplified in agentic workflows, and can persist into downstream files, indicating a potential path toward vendor lock-in.</arxiv_paper_description_3>
- The research demonstrates that VIB is present in direct code generation, significantly amplified in agentic workflows, and can persist into downstream files, indicating a potential path toward vendor lock-in.

---

[On Compositional Learning Behaviours in Formal Mathematics](http://arxiv.org/abs/2605.28512)

- S2B-LM (Symbolic Behaviour Benchmark for Language Models): introduces a diagnostic framework that replaces continuous stimuli with categorical tokens and employs a rule-based verbalizer to scaffold chain-of-thought reasoning for evaluating Compositional Learning Behaviours in LLMs.
- The framework utilizes a sync round mechanism to provide ground-truth evidence, which the verbalizer converts into reasoning traces to address the competence-performance distinction in LLMs.
- Empirical results demonstrate that Compositional Learning Behaviour competency is a necessary structural prerequisite for LLMs to achieve Olympiad-level performance in formal mathematical verification.

---

[The Decision to Verify: How Warmth and User Characteristics Shape Reliance on Conversational Agents for Information Search](http://arxiv.org/abs/2605.28498)

- Experimental Framework: investigates how conversational warmth and user characteristics influence reliance on LLMs during information search tasks.
- The study employs a mixed-subjects experiment where participants interact with either a warm or neutral chatbot while having access to external web search tools.
- Findings reveal that reliance is primarily driven by individual user tendencies rather than the availability of fact-checking tools, with conversational warmth indirectly increasing overreliance.

---

[Beyond One Path: Evaluating and Enhancing Divergent Thinking in Interactive LLM Agents](http://arxiv.org/abs/2605.28465)

- ReDNA (Reflect-driven Divergent-to-Narrowing Agent): introduces a framework that enhances divergent thinking in LLM agents by decoupling candidate generation from convergent selection using a target-centered failure memory.
- The framework utilizes a Reflect module to accumulate object-level failure feedback and a Divergent-to-Narrowing module to generate and select mechanism-distinct solution paths.
- Experimental results demonstrate that ReDNA significantly improves path discovery and action-level divergence across various LLM families by resolving structural blind spots in agentic reasoning.

---

[Skill0.5: Joint Skill Internalization and Utilization for Out-of-Distribution Generalization in Agentic Reinforcement Learning](http://arxiv.org/abs/2605.28424)

- Skill0.5: introduces a unified agentic RL framework that jointly optimizes decoupled general and task-specific skills based on real-time task mastery.
- The framework utilizes a difficulty-aware router to stream tasks into three tiers, applying privileged distillation for hard tasks, standard GRPO for medium tasks, and anti-shortcut diagnostic probing for easy tasks.
- By permanently internalizing general reasoning heuristics while dynamically utilizing task-specific skills, Skill0.5 achieves robust OOD generalization and avoids common failure modes like contextual interference and parametric knowledge conflict.

---

[You Live More Than Once: Towards Hierarchical Skill Meta-Evolving](http://arxiv.org/abs/2605.28390)

- HiSME (Hierarchical Skill Meta-Evolving): introduces a lightweight, text-based framework that optimizes both task-level skills and the evolving procedure itself by learning meta-skills from agent execution traces.
- The framework utilizes an Executor (LLM-based agent performing tasks), a Skill Library (reusable artifacts for agent), a Retriever (selects relevant skills for executor), an Extractor (identifies reusable fragments from traces), a Refactorer (abstracts shared structures into skills), a Refiner (revises existing skills based on evidence), a Credit Assigner (evaluates skill helpfulness/harmfulness), a Bundle Tester (maintains test cases for skills), a Filter (manages skill exposure based on performance), a Meta-Evo Agent (optimizes evolving rules via meta-skills), and a Replay Buffer (stores experience for meta-evolving).
- HiSME improves agent performance by treating skill generation and maintenance as a multi-level residual optimization problem, allowing the system to adapt its evolving strategy at test time without modifying underlying LLM parameters.

---

[Teacher-Student Representational Alignment for Reinforcement Learning-Driven Imitation Learning](http://arxiv.org/abs/2605.28372)

- Teacher-Student Representational Alignment: introduces a task-agnostic method that bridges the imitation gap by learning a shared latent representation space that hides agent-specific private information from the teacher policy.
- The framework utilizes a multi-view approach with contrastive learning to extract common components from teacher and student observations, ensuring the teacher learns behaviors reproducible by the student.
- By incorporating explicit alignment and stability losses, the method enables stable policy training and consistent performance across environments without requiring modifications to the original reward function.

---

[From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence](http://arxiv.org/abs/2605.28371)

- FCA (Framework-Coupled Agent): introduces a staged workflow that maps scientific papers into a shared PHM framework using INGEST, ANALYZE, MAP, IMPLEMENT, VERIFY, and REPORT components to transform under-specified methods into executable, benchmarkable implementations.
- The framework utilizes explicit assumption tracking and verification gates to ensure that implementation decisions are auditable and consistent across different research papers.
- By coupling agentic generation with a shared domain-specific framework, the approach enables systematic cross-paper benchmarking and reduces the labor-intensive nature of reproducing scientific machine learning methods.

---

[CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict](http://arxiv.org/abs/2605.28369)

- CyberJurors: introduces a multi-agent framework that integrates IV-CoT and JCV to achieve fine-grained evidence perception and fair dispute verdicts in E-commerce.
- The framework utilizes IV-CoT for structured reasoning and JCV for multi-round jury simulation, supported by a Verdict Precedent Base to mitigate cognitive biases.
- Experiments on the VerdictBench dataset demonstrate that CyberJurors outperforms state-of-the-art LLMs and court simulators in accuracy and alignment with real-world jury voting patterns.

---

[Prompt Codebooks: Discrete Compositional Optimization for Language Model Instruction Refinement](http://arxiv.org/abs/2605.28360)

- PCO (Prompt Codebooks): introduces a compositional prompt optimization framework that recasts automatic prompt optimization as discrete learning over a finite vocabulary of natural-language instincts, utilizing a Prompt Encoder, Codebook, Prompt Generator, and Critic.
- The framework employs an LLM-based encoder for per-instance adaptive routing, selecting specific instincts from a codebook to be composed into a prompt by a generator, which is then refined via natural-language feedback from a critic.
- By replacing monolithic prompt strings with discrete, reusable instruction units, PCO enables localized credit assignment and improves performance across reasoning, mathematical, and instruction-following benchmarks while reducing inference-time prompt overhead.

---

[From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets](http://arxiv.org/abs/2605.28359)

- KTD-FIN (Knowing-To-Doing Financial Benchmark): introduces an end-to-end stock market trading benchmark that mitigates pretraining memory leakage via a data-side masking protocol and evaluates LLM trading agents using a Barra-style performance attribution framework.
- The framework utilizes an Agent Trader that operates under three distinct decision modes—memory-only, fixed-candidate, and open-research—to probe different levels of agentic reasoning and information access.
- By employing a ten-dimensional metric panel and an independent de-anonymization probe, the benchmark separates genuine stock-selection alpha from passive market and style factor exposure.

---

[Plan Before Search: Search Agents Need Plan](http://arxiv.org/abs/2605.28354)

- PL-Search: introduces a structured agentic framework that decomposes multi-hop questions into ordered sub-questions to anchor retrieval steps and prevent trajectory drift.
- The framework utilizes a self-bootstrapping paradigm where a seed model generates high-quality trajectories to train target LLMs, eliminating the need for distillation from stronger models.
- The approach employs a composite reward function comprising plan-, format-, and outcome-based components to supervise the structural validity and semantic alignment of the agent's reasoning process.

---

[Towards Cybersecurity SuperIntelligence (CSI): What’s the best harness for cybersecurity?](http://arxiv.org/abs/2605.28334)

- CSI: introduces a meta-scaffold architecture that unifies heterogeneous LLM-driven agent harnesses under a common orchestration layer to improve cybersecurity task performance.
- The architecture utilizes a blackboard-based multi-agent protocol where scaffold-specialised agents exchange intermediate findings via a shared substrate to exceed the performance of individual harnesses.
- By leveraging structural heterogeneity among scaffolds, the system achieves higher coverage on cybersecurity challenges than any single LLM-based agent scaffold.

---

[Multi-Agent LLM-based Metamorphic Testing for REST APIs](http://arxiv.org/abs/2605.28321)

- ARMeta: introduces a multi-agent LLM-based workflow that automates metamorphic testing for REST APIs by deriving metamorphic relations from OpenAPI specifications and executing them as Gherkin-based tests.
- The framework utilizes a Test Manager to coordinate specialized agents, including MR Generator-, MR Refiner-, Test Generator- and Code Refiner-agents, to ensure generated tests are syntactically correct and semantically aligned with API documentation.
- ARMeta improves testing robustness by generating diverse API call sequences that reveal faults, such as contract violations and state inconsistencies, which are often missed by conventional scenario-based testing approaches.

---

[How Far Can Disaggregation Go? A Design-Space Exploration of Attention–FFN Disaggregation for Efficient MoE LLM Serving](http://arxiv.org/abs/2605.28302)

- AIC++: introduces a co-design framework for exploring the design space of Attention–FFN Disaggregation (AFD) in MoE LLM serving by combining AIConfigurator, AstraSim, and a vLLM-based AFD prototype.
- The framework enables multi-dimensional design-space exploration by jointly optimizing token-level parallelism, prefill–decode disaggregation, and operator-level attention–FFN disaggregation to identify optimal GPU placement and scheduling.
- By modeling compute and communication costs, AIC++ identifies that AFD improves user interactivity through micro-batch overlapping and enables feasible deployments for memory-constrained long-context workloads.

---

[AtomComposer: Discovering Chemical Space from First Principles with Reinforcement Learning](http://arxiv.org/abs/2605.28287)

- AtomComposer: introduces a self-guided RL agent that constructs valid 3D isomers from first principles using a multi-composition training scheme and terminal rewards.
- The framework utilizes an equivariant PAINN backbone and PPO to optimize atom-by-atom placement policies without pretraining on existing molecular datasets.
- AtomComposer demonstrates composition-generalizable discovery by leveraging terminal atomization energy and validity rewards to navigate chemical space autonomously.

---

[Commit to the Bit: Reactive Reinforcement Learning Done Right](http://arxiv.org/abs/2605.28276)

- Committed Q-learning: introduces a reinforcement learning algorithm that converges to the optimal reactive policy in partially observable environments under the rewire-robustness assumption.
- The framework utilizes a Q-table, behavior policy, and options to perform updates only when observed features change, effectively operating on an aggregate MDP.
- By minimizing the Bellman risk, the approach avoids the learnability issues associated with traditional value error or Bellman error in partially observable settings.

---

[EchoAvatar: Real-time Generative Avatar Animation from Audio Streams](http://arxiv.org/abs/2605.28272)

- EchoAvatar: introduces a unified streaming framework for real-time 3D avatar animation from audio, utilizing a Causal Attention-based Motion Tokenizer, a Pre-trained LLM Generator, and a Reinforcement Learning (RL) Alignment stage.
- The framework employs Hierarchical Token Corruption to ensure robust audio-motion alignment across diverse domains and integrates a tool-call interface for semantic control by LLMs.
- Extensive experiments demonstrate that the system achieves high-fidelity, low-latency motion synthesis, outperforming state-of-the-art baselines in both objective metrics and subjective human preference.

---

[GUI Agents for Continual Game Generation](http://arxiv.org/abs/2605.28258)

- Play2Code: introduces a framework for continual game generation that utilizes a Game Agent and a GUI Agent operating in a sustained loop with shared memory to refine game artifacts based on playtesting feedback.
- The framework employs a three-layer memory system comprising Episode Memory, Skill Memory, and World Memory to accumulate experience and improve generation performance across rounds and tasks.
- PlaytestArena serves as the evaluation environment, using GUI agents to adjudicate game quality against rubrics of expected in-play behaviors, providing a scalable alternative to human playtesting.

---

[POINav: Benchmarking and Enhancing Final-Meters Arrival in Real-World Vision-Language Navigation](http://arxiv.org/abs/2605.28237)

- POINav (POINav Brain-Action Framework): introduces a decoupled navigation architecture that utilizes a Semantic Brain Module for POI-grounded reasoning and a Geometric Action Module for continuous waypoint prediction.
- The framework employs Elastic Temporal Sampling to maintain global context from observation history and Latent Action Querying to bypass iterative diffusion processes for efficient geometric control.
- The research includes the POINav-Bench, a high-fidelity 3D Gaussian Splatting platform, and a curated dataset of 70K signage-entrance pairs to facilitate precise final-meters navigation for embodied agents.

---

[PIRS: Physics-Informed Reward Shaping for SAC-Based Building Energy Management](http://arxiv.org/abs/2605.28232)

- PIRS (Physics-Informed Reward Shaping): introduces a biophysically grounded reward mechanism for building energy management that replaces heuristic comfort proxies with the ISO 7730 Predicted Mean Vote (PMV) index.
- The framework integrates the PMV-Module into a standard SAC-Agent pipeline to improve reward interpretability and grid-stress performance without requiring modifications to the underlying reinforcement learning architecture.
- Empirical results demonstrate that PIRS achieves performance comparable to manually tuned baselines while significantly outperforming non-physics-grounded reward designs in load ramping and peak demand metrics.

---

[When Does Memory Help Multi-Trajectory Inference for Tool-Use LLM Agents?](http://arxiv.org/abs/2605.28224)

- LiTS: introduces a unified scope-abstraction framework for memory in tool-use LLM agents that decomposes memory into scope (within-expansion vs. across-trajectories) and abstraction (raw, reflection, or atomic facts) to evaluate their impact on inference strategies.
- The research identifies the inference method as a critical confound, demonstrating that memory mechanisms produce statistically distinct results depending on whether best-of-N, beam search, or MCTS is employed.
- Empirical results show that reflection significantly improves accuracy only under MCTS, cross-sibling injection aids diversity-starved beam search, and atomic fact extraction improves efficiency by reducing trajectory length on tasks with reusable environmental structure.

---

[Out of Sight, Not Out of Mind: Unveiling Latent Attack in Latent-based Multi-Agent Systems](http://arxiv.org/abs/2605.28214)

- LatentMAS: introduces a latent attack framework that reactivates adversarial effects in multi-agent systems by injecting steering vectors into hidden states and KV-cache handoffs without explicit adversarial text.
- The framework constructs attack-associated steering vectors from paired clean-correct and attacked-wrong executions to manipulate agent behavior through node-level hidden states or edge-level KV-cache handoffs.
- Empirical results demonstrate that latent-based multi-agent systems are vulnerable to these attacks, with inter-agent handoffs presenting a more significant attack surface than local agent nodes.

---

[The Illusion of Opting in AI-Mediated Consequential Decisions](http://arxiv.org/abs/2605.28210)

- The Illusion of Opting: introduces a conceptual framework describing how AI-mediated systems create a deceptive appearance of meaningful choice while simultaneously weakening the meta-capacity required for genuine consequential decision-making.
- The paper identifies that AI-mediated systems can lead to conversion or drift by presenting optimized pathways that prematurely foreclose alternatives and displace the user's agency.
- The authors propose three normative imperatives—existential honesty, ecological rationality, and counterfactual reparation—to protect and cultivate the meta-capacity necessary for individuals to navigate life-determining decisions under radical uncertainty.

---

[Nonvolatile Charge-Domain Attention with HZO Ferroelectric Capacitors: A Simulation-Based Device-to-System Evaluation](http://arxiv.org/abs/2605.28208)

- FCDC (Ferroelectric Charge-Domain Compute Cell): introduces a nonvolatile HZO memcapacitor-based architecture for performing charge-domain vector-matrix multiplication in transformer attention mechanisms.
- The framework utilizes HZO memcapacitors to enable persistent KV-cache residency, significantly reducing energy consumption in long-context LLM inference by eliminating refresh requirements.
- The architecture is evaluated through a device-to-system simulation model, demonstrating substantial energy advantages over GPU baselines for long-residency workloads while maintaining LLM accuracy.

---

[Plant, Persist, Trigger: Sleeper Attack on Large Language Model Agents](http://arxiv.org/abs/2605.28201)

- Sleeper Attack framework: introduces a unified threat model where adversarial content is planted into agent states, remains dormant across interactions, and is later triggered by benign user queries.
- The framework evaluates vulnerabilities across three agent state targets—session context, memory, and reusable skills—using three distinct attack strategies: LIP, PIE, and PIC.
- Experimental results demonstrate that LLM agents remain significantly vulnerable to these cross-interaction threats, even when they exhibit low attack success rates under single-interaction baselines.

---

[Agentic Active Omni-Modal Perception for Multi-Hop Audio-Visual Reasoning](http://arxiv.org/abs/2605.28192)

- AOP-Agent: introduces an efficient agentic framework for active omni-modal perception that enables open-source LLMs to perform multi-hop audio-visual reasoning without additional training.
- The framework utilizes a Hierarchical Omni-modal Memory and a collaborative observe-reflect-replan loop to progressively localize and integrate sparse evidence across long videos.
- AOP-Agent includes planning-, reflection- and reasoning-agents that work together to navigate video content and reduce the reliance on full-video processing.

---

[Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation](http://arxiv.org/abs/2605.28175)

- MixRAGRec: introduces a cooperative multi-agent framework for KG-RAG recommendation that integrates a Mixture-of-Experts Retrieval Agent, a Knowledge Preference Alignment Agent, and a Contrastive Learning-reinforced Recommendation Agent.
- The framework utilizes MMAPO to jointly optimize these agents under a shared objective that balances recommendation performance with retrieval utility via a marginal information gain reward.
- By routing queries to appropriate retrieval experts ranging from no retrieval to connected-graph retrieval, the system effectively manages the trade-off between knowledge richness and computational cost.

---

[MangaFlow: An End-to-End Agentic Framework for Controllable Story to Manga Generation](http://arxiv.org/abs/2605.28173)

- MangaFlow: introduces an agentic framework for controllable long-form manga generation that decomposes the creation process into structured subtasks including planning, layout construction, and reference-conditioned rendering.
- The framework utilizes a Story Section Memory to maintain consistency of characters, scenes, and objects across panels by linking narrative segments with explicit visual references.
- MangaFlow treats page layout as an explicit, editable structural variable and includes a meta-benchmark, MangaGen-MetaBench, to evaluate layout controllability, visual consistency, and lettering quality.

---

[OccuReward: LLM-Guided Occupant-Centric Reward Shaping for Demographic Equity in Grid-Interactive Buildings](http://arxiv.org/abs/2605.28168)

- OccuReward: investigates how LLM-mediated reward design influences demographic equity in grid-interactive buildings by utilizing Gemini 1.5 Flash to iteratively refine reward functions based on the Comfort Equity Index.
- The framework employs a Soft Actor-Critic agent within the CityLearn v2 environment to optimize energy management while balancing thermal comfort across diverse demographic profiles derived from the ASHRAE Global Thermal Comfort Database II.
- Experimental results demonstrate that incorporating equity-aware feedback into the LLM reward shaping process significantly improves satisfaction for vulnerable demographic groups while simultaneously enhancing grid flexibility and reducing energy costs.

---

[OR-Space: A Full-Lifecycle Workspace Benchmark for Industrial Optimization Agents](http://arxiv.org/abs/2605.28158)

- OR-Space: introduces a full-lifecycle benchmark for industrial optimization agents that evaluates performance across persistent multi-artifact workspaces using Documents (D), Parameters (P), Code (S), Runtime Environment (E), and Evaluation Metric (M).
- The framework evaluates LLMs through three lifecycle-oriented task modes: Build (constructing models), Revise (modifying models), and Explain (grounded reasoning over solver outputs).
- OR-Space exposes critical failure modes in data mapping, constraint grounding, and cross-artifact consistency that are typically hidden in isolated, text-only optimization benchmarks.

---

[DeltaMCP: Incremental Regeneration via Spec-Aware Transformation for MCP servers](http://arxiv.org/abs/2605.28148)

- DeltaMCP: introduces a transformation-based incremental regeneration system that updates only the affected MCP tools when OpenAPI specifications evolve, preserving custom enterprise logic.
- The framework utilizes a semantic differencing tool to isolate schema changes, which are then processed by a fine-tuned LLM to generate targeted, deterministic code updates.
- DeltaMCP significantly reduces computational overhead and memory usage compared to full-generation methods while maintaining high-fidelity, up-to-date MCP server infrastructures.

---

[Cybersecurity AI (CAI) Dataset](http://arxiv.org/abs/2605.28146)

- CAI (Cybersecurity AI) framework: introduces a large-scale, trajectory-level corpus of cybersecurity LLM interactions collected through a multi-scaffold agent framework to address the scarcity of expert operator data.
- The dataset aggregates over 26 million user prompts and 230,000 session logs, capturing real-world operator workflows including reconnaissance, exploitation, and persistence.
- The research highlights the operational risks of LLM-augmented workflows and provides a blueprint for privately-hosted, cybersecurity-specialised LLMs to maintain operator confidentiality.

---

[SNARE: Adaptive Scenario Synthesis for Eliciting Overeager Behavior in Coding Agents](http://arxiv.org/abs/2605.28122)

- SNARE: introduces an adaptive pipeline that synthesizes benign scenarios to elicit and measure overeager behavior in coding agents using Consent-realization library, Trap-surface library, Skeleton library, Fixture-seed library, Thompson sampler, Mutation operator family, Composite oracle, and Docker-based sandbox.
- The framework employs a Thompson sampler to dynamically allocate a fixed run budget across archetype-consent cells, ensuring coverage while concentrating on high-yield scenarios.
- Evaluation across a 4x5 matrix of coding agents and base models reveals that the agent framework accounts for the majority of overeager behavior variation, significantly outperforming static benchmarks.

---

[LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](http://arxiv.org/abs/2605.28120)

- LegalGraphRAG: introduces a framework for reliable legal reasoning that synergizes a Hierarchical Legal Graph (HierarGraph) with a multi-agent system comprising Researcher-, Auditor- and Adjudicator-agents.
- The HierarGraph organizes legal knowledge into Fact-, Ontology- and Rule-graphs to decouple historical cases, statutes and interpretations, while the multi-agent system performs evidence-based retrieval, validation and synthesis.
- By enforcing stepwise verification through Diagnostic Checklists and Judicial Interpretations, the framework ensures that LLM-generated legal judgments are transparent, evidence-supported and traceable.

---

[MIRAGE: Context-Aware Prompt Injection against Mobile GUI Agents via User-Generated Content](http://arxiv.org/abs/2605.28116)

- MIRAGE: introduces a three-stage pipeline that transforms benign mobile screenshots into realistic, context-aware prompt-injection samples for benchmarking VLM-based GUI agents.
- The framework utilizes a Localizer to identify injection points, a Generator to synthesize payloads using gpt-image-2, and a Curator to ensure visual realism and distributional balance.
- Empirical results demonstrate that all evaluated VLM-based GUI agents are vulnerable to these injections, and that visual-quality filtering is insufficient as a defense because realism does not correlate with attack success.

---

[Human-like In-group Bias in Instruction-tuned Language Model Agents](http://arxiv.org/abs/2605.28114)

- Multi-agent simulation framework: introduces a controlled environment to evaluate group-contingent social dynamics in LLMs across multiple architectures and training regimes.
- The framework utilizes a trust-based interaction model where agents maintain internal states and memory to coordinate tasks and accumulate reputational histories.
- The study demonstrates that label salience triggers systematic in-group bias and structural inequality, which remain covert in standard action-log audits.

---

[Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents](http://arxiv.org/abs/2605.28108)

- ATRBench (Ask-to-Remember Benchmark): introduces a controlled benchmark to evaluate the proactivity gap in long-lived LLM agents by measuring their ability to acquire latent user preferences before they are needed for future tasks.
- The framework utilizes an Agent, User Simulator, Router, Classifier, Cross-session Context, Standing Rule Pool, and Test Sessions to isolate explicit asking as a measurable acquisition channel with delayed payoffs.
- Evaluations across eight frontier LLMs reveal a significant proactivity gap, where agents fail to proactively acquire reusable preferences, and prompting-based interventions yield only minimal improvements in performance.

---

[Defending LLM-based Multi-Agent Systems Against Cooperative Attacks with Sentence-Level Rectification](http://arxiv.org/abs/2605.28104)

- STAR (Sentence-Level Trustworthiness Analysis and Rectification): introduces a defense framework that utilizes sentence-level decomposition and verification, suspicion modeling, targeted rectification, and robust decision aggregation to mitigate misinformation in LLM-based multi-agent systems.
- The paper proposes a Cooperative Attack Method where malicious agents use ally-context awareness and cooperative response formulation to dynamically coordinate deceptive strategies.
- Empirical evaluations demonstrate that the STAR framework significantly improves task success rates and reduces attack success rates against both independent and cooperative adversarial threats.

---

[Examining Agents’ Bias Amplification versus Suppression in Multi-Agent Systems](http://arxiv.org/abs/2605.28098)

- Multi-agent decision pipelines: introduces a framework to evaluate system-wide fairness in LLM-based multi-agent systems by analyzing how prompt-induced bias propagates through interacting Prediction Agent, Explanation Agent, ML Predictor, and Judge Agent components.
- The study utilizes the Favor Bias Strength (FBS) metric to quantify how agent-level bias exposures are amplified or suppressed at the system level across different pipeline configurations.
- Empirical results demonstrate that multi-agent interactions can super-additively amplify individual agent biases, while ML-grounded arbitration serves as a partially effective mitigation strategy.

---

[ICAN-Deploy: Identity-Stable Canary Deployment for Safety-Critical Embodied Agents](http://arxiv.org/abs/2605.28097)

- ICAN-Deploy: introduces a middleware construction that maintains a constant cryptographic identity hash for safety-critical embodied agents throughout the canary deployment process by separating capability names from mutable version states.
- The framework utilizes an AEROS bridge to manage an eight-state pipeline, ensuring that identity-affecting inputs remain invariant while allowing for safe, atomic version updates via a provisional version map.
- Formal verification through TLA+ model-checking and structural AST linting confirms that the identity hash remains byte-invariant across all canary transitions, preventing identity drift in LLM-driven robotic systems.

---

[VLA-Hijack: A Transferable Patch Attack against Vision-Language-Action Models via Visual Proprioception Hijacking](http://arxiv.org/abs/2605.28083)

- VLA-Hijack: introduces a unified adversarial framework that improves transferability by hijacking the visual proprioceptive loop of VLA models through Attention-Guided Proprioceptive Suppression and Multimodal Proprioceptive Injection.
- The framework utilizes an Alternation Gate to synchronize Semantic Concept Anchoring and Visual Prototype Projection, effectively replacing the agent's true embodiment with a surrogate "phantom embodiment" in the model's latent space.
- By creating a "perceptual vacuum" around the real robotic arm, VLA-Hijack forces the VLA model to misidentify the adversarial patch as its own physical structure, achieving superior cross-architecture and cross-domain transferability.

---

[MACReD: A Multi-Agent Collaborative Reasoning Framework for Reaction Diagram Parsing](http://arxiv.org/abs/2605.28077)

- MACReD: introduces a hierarchical multi-agent framework that coordinates specialized agents across Planning-, Perception- and Reasoning-layers to parse complex chemical reaction diagrams.
- The framework utilizes a Multigraph Fusion mechanism to integrate spatial, chemical, and VLM-derived evidence into a unified representation for robust reaction reconstruction.
- By decomposing the parsing task into collaborative agent-level subtasks, MACReD achieves state-of-the-art performance on the RxnScribe benchmark while maintaining chemical consistency across diverse diagram layouts.

---

[AgentGuard: An Attribute-Based Access Control Framework for Tool-Use LLM-Based Agent](http://arxiv.org/abs/2605.28071)

- AgentGuard: introduces a mandatory, attribute-based access control framework for tool-use LLMs that utilizes a client-server architecture to intercept and inspect tool invocations.
- The framework integrates with LLM agents via lightweight client-side SDKs, enabling centralized security enforcement through the AgentGuard Server without modifying underlying agent logic.
- Security inspection is performed using rule-based detection, LLM-assisted detection, and manual verification to mitigate both single-tool and cross-tool security risks.

---

[ZipRL: Adaptive Multi-Turn Context Compression with Hindsight Response Replay](http://arxiv.org/abs/2605.28069)

- ZipRL: introduces a framework that optimizes long-horizon agentic tasks by integrating a Multi-Granularity Mechanism for adaptive context compression with Hindsight Response Replay for efficient RL training.
- The framework utilizes a multi-dimensional heuristic scoring function to evaluate compression quality, enabling the model to dynamically adjust information retention based on query relevance.
- By reshaping trajectory-level advantages into turn-level signals, the approach effectively mitigates sparse reward challenges in multi-turn interactions without requiring external reward models.

---

[Verifiable Benchmarking of Long-Horizon Spatial Biology](http://arxiv.org/abs/2605.28065)

- SpatialBench-Long: introduces a benchmark for evaluating LLM agents on long-horizon scientific reasoning tasks using Study systems, Candidate claims, Reproduce &amp; review, Build evaluation, Score &amp; diagnose, Trajectory diagnostics, and Final-answer grader.
- The framework utilizes deterministic final-answer grading paired with rubric-based trajectory diagnostics to assess agent performance across complex spatial biology workflows.
- Experimental results across 15 model-harness pairs demonstrate that current LLMs struggle with end-to-end scientific reasoning, often failing due to compounding local analysis errors.

---

[CogPortrait: Fine-Grained Eye-Region Control in Portrait Animation via Hierarchical Agent Planning](http://arxiv.org/abs/2605.28056)

- CogPortrait: introduces a two-stage framework that compiles high-level labels into fine-grained eye-region facial keypoints using Planning Agent, Composition Agent, and Critic Agent, followed by a DiT-based Video Generation Backbone.
- The framework utilizes a Prototype Library to preserve genuine temporal dynamics and employs a Dynamic CFG Strategy with eye-region-aware reweighting to improve local controllability.
- KTO Refinement is integrated to enhance generation quality for boundary cases, ensuring identity consistency and precise control over irregular facial motions.

---

[MemCog: From Memory-as-Tool to Memory-as-Cognition in Conversational Agents](http://arxiv.org/abs/2605.28046)

- MemCog: introduces a Memory-as-Cognition paradigm that integrates Navigable Memory Store, Cross-Dimensional Navigation Interface, and Proactive Reasoning Protocol to make memory access an integral part of the LLM reasoning process.
- The system replaces passive single-shot retrieval with autonomous, multi-step navigation, allowing the LLM to dynamically explore structured memory based on conversational context.
- MemCog achieves state-of-the-art performance on passive QA benchmarks and significantly improves proactive memory triggering through its synergistic protocol-structure design.

---

[Personality, Role, and Expressive Style in Large Language Models: An Interactionist Analysis](http://arxiv.org/abs/2605.28037)

- Interactionist Analysis Framework: introduces a methodology to analyze how prompt-specified Big Five traits interact with dialogue-level social context, including role and expressive style, to shape generated utterances.
- The framework employs a factorial design to systematically combine personality, role, and expressive style conditions, generating dyadic dialogues that are subsequently evaluated by an LLM-as-a-Judge.
- Results demonstrate that personality expression in LLMs is context-dependent, with the relative influence of personality specification, role, and expressive style varying significantly across Big Five dimensions.

---

[RESEARCHMATH-14K: Scaling Research-Level Mathematics via Agents](http://arxiv.org/abs/2605.28003)

- RESEARCHMATH-14K: introduces a large-scale dataset of 14,056 research-level mathematical problems curated from academic sources using an agentic pipeline comprising an Extractor Agent and a Refiner Agent.
- The framework utilizes Qwen3-Embedding-8B for duplicate filtering and employs GPT-5.5 and GPT-5.4-nano as judges to evaluate reasoning behavior and factuality in generated trajectories.
- The study demonstrates that fine-tuning LLMs on filtered, wrong-but-reasonable reasoning trajectories significantly improves performance on research-level mathematical tasks compared to base models.

---

[Tool Forge: A Validation-Carrying Toolchain for Governed Agentic Execution](http://arxiv.org/abs/2605.28000)

- Tool Forge: introduces a validation-carrying toolchain that transforms natural-language intent into governed, sandbox-verified tool capsules to improve the reliability of LLM agent execution.
- The framework utilizes a Generation Loop to synthesize capability contracts and a Routing Loop to perform intent-scoped tool selection, significantly reducing token consumption compared to naive full-catalog exposure.
- By treating tools as inspectable artifacts containing validation evidence and metadata, the system shifts agent tool management from model-driven completion to a governed software compilation process.

---

[Learning to Assign Prediction Tasks to Agents with Capacity Constraints](http://arxiv.org/abs/2605.27999)

- Online Task Allocation Framework: introduces a contextual multi-armed bandit approach that learns agent expertise while enforcing long-run capacity constraints for sequential task assignment.
- The framework utilizes a Contextual Reward Model to estimate performance and a Virtual Queue to maintain agent-specific workload limits, ensuring optimal task routing.
- Experimental results across various datasets demonstrate that this approach systematically outperforms non-contextual baselines by effectively leveraging heterogeneous agent expertise.

---

[AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios](http://arxiv.org/abs/2605.27995)

- AsyncTool: introduces a benchmark for evaluating LLM-based agents in interactive multi-task environments with delayed tool feedback.
- The framework assesses agent performance across step, sub-task, and task levels using efficiency-oriented metrics to measure task interleaving and dependency tracking.
- Experimental results demonstrate that delayed tool feedback causes significant performance degradation, highlighting the necessity for agents to master temporal coordination and state maintenance.

---

[KVoiceBench, KOpenAudioBench, and KMMAU: Agent-Driven Korean Speech Benchmarks for Evaluating SpeechLMs](http://arxiv.org/abs/2605.27984)

- KVoiceBench, KOpenAudioBench, and KMMAU: introduces human-agent collaborative frameworks for constructing high-quality target-language speech benchmarks by transferring source-language SpokenQA and converting ASR corpora into audio understanding tasks.
- The framework utilizes Reviewer LLM, Meta-reviewer LLM, Human-agent collaborative loop, Normalization agent, TTS model, ASR corpora, and Human annotators to ensure benchmark validity across language-specific instructions and paralinguistic properties.
- The research evaluates eight SpeechLMs, revealing that English-Korean performance gaps vary significantly across models and task families, with SpokenQA and audio understanding rankings diverging.

---

[Mags-RL: Wearing Multimodal LLMs a Magnifying Glass via Agentic Reinforcement Learning For Complex Scene Reasoning](http://arxiv.org/abs/2605.27960)

- Mags-RL: introduces an agentic reinforcement learning framework that equips MLLMs with an external super-resolution agent to enhance fine-grained visual reasoning in complex scenes.
- The framework utilizes a two-round reasoning process where the MLLM policy generates an initial rationale and identifies regions of interest, which are then upscaled by the super-resolution agent for verification and final answer generation.
- Mags-RL employs a data-efficient curriculum learning strategy and a multi-component reward function to align the MLLM policy with agentic behaviors, achieving superior performance on complex visual benchmarks with minimal training samples.

---

[DisasterBench: Benchmarking LLM Planning under Typed Tool Interface Constraints](http://arxiv.org/abs/2605.27957)

- DisasterBench: introduces a benchmark for evaluating executable workflow grounding in multi-agent systems by requiring LLMs to coordinate semantically similar but operationally distinct disaster-response tools.
- The framework utilizes a Tool Pool, Workflow Execution Constraints, Task Generation, LLM Planning Paradigms, and an FPoF Diagnostic Framework to identify bottlenecks like depth-driven performance collapse and instruction clash.
- Evaluation across 14 LLMs reveals that while semantic tool selection is achievable, maintaining execution-consistent parameter bindings and dependency propagation remains a significant challenge for current models.

---

[Skill-as-Pseudocode: Refactoring Skill Libraries to Pseudocode for LLM Agents](http://arxiv.org/abs/2605.27955)

- SaP (Skill-as-Pseudocode): introduces a verified refactoring pipeline that converts free-form markdown skill libraries into structured, typed pseudocode to improve LLM agent performance.
- The framework utilizes a deterministic verifier and LLM-aware passes to generate typed contracts and concrete action templates, effectively breaking the prose-induced retrieval-action loop.
- Experimental results on ALFWorld and SkillsBench demonstrate that SaP simultaneously increases task success rates and reduces token consumption compared to standard retrieval baselines.

---

[Cyclical Entropy Eruption: Entropy Dynamics in Agent Reinforcement Learning](http://arxiv.org/abs/2605.27954)

- SEAL (Separation-Enhanced Agent Learning): introduces a lightweight auxiliary loss that separates correct and incorrect trajectories in representation space to mitigate gradient interference and stabilize agent RL training.
- The framework addresses the cyclical entropy eruption phenomenon, where high representation similarity between trajectories causes gradient interference and leads to training instability.
- By training a binary classifier to distinguish between correct and incorrect trajectory representations, the method reduces harmful gradient interference and suppresses degenerate patterns like sentence duplication and hallucination.

---

[Do Agents Think Deeper? A Mechanistic Investigation of Layer-Wise Dynamics in Sequential Planning](http://arxiv.org/abs/2605.27935)

- Sparse Mixture-of-Experts (MoE) Transformer Architecture: introduces a mechanistic study of layer-wise dynamics in autonomous LLM agents, demonstrating that sequential planning induces adaptive depth allocation.
- The study reveals that as agentic trajectories unfold, models progressively recruit deeper layers and exhibit correction-dominant residual updates to manage state complexity.
- Quantitative analysis identifies a construction-refinement gap where semantic direction is established early, while deep layers are required for final output stabilization.

---

[Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows](http://arxiv.org/abs/2605.27922)

- Harness-Bench: introduces a diagnostic benchmark for evaluating configuration-level harness effects in realistic agent workflows by fixing external task conditions while preserving native harness execution behavior.
- The framework evaluates agent performance across 106 sandboxed tasks, utilizing an evaluation pipeline that integrates an Oracle Grader, LLM Judge, and multi-dimensional scoring to assess completion, process quality, and security.
- Experimental results across 5,194 trajectories demonstrate that agent capability is significantly influenced by the harness layer, supporting the reporting of performance at the model–harness configuration level rather than by the base model alone.

---

[OphIn-500K: Curating Web-Scale Visual Instructions for Scaling Ophthalmic Multimodal Large Language Models](http://arxiv.org/abs/2605.27916)

- OphIn-Engine: introduces an automated data curation pipeline that transforms web-scale ophthalmic videos into high-quality multimodal instruction pairs for training specialized LLMs.
- The framework utilizes a Multimodal Transcription Pipeline, Visual Cue Separation and Scoring Module, Instruction Synthesis Module, and Post-processing and Quality Control Module to construct the OphIn-500K dataset.
- OphIn-VL, an ophthalmology-specific MLLM, leverages the OphIn-500K dataset to achieve superior performance in clinical visual interpretation and multi-turn conversational capabilities.

---

[ESC-Skills: Discovering and Self-Evolving Skills for Emotional Support Conversations](http://arxiv.org/abs/2605.27908)

- ESC-Skills: introduces a skill-centric framework that models emotional support as localized state-action-outcome dynamics to discover and self-evolve executable support skills.
- The framework utilizes Intervention Units (IUs) to construct an ESC-Skills Bank, which is refined through a multi-profile self-evolutionary loop involving SAGE, a Skill Generator, and a Skill Verifier.
- Experimental results demonstrate that ESC-Skills improves response-level quality and long-horizon emotional support outcomes across multiple LLM backbones by providing explicit, verifiable behavioral guidance.

---

[AI Research Agents Narrow Scientific Exploration](http://arxiv.org/abs/2605.27905)

- AI research-agent frameworks: introduces a systematic study evaluating how four AI research-agent frameworks and six LLMs generate scientific ideas from citation-defined research areas compared to human-authored research.
- The study reveals that AI-generated ideas are more concentrated, remain closer to seed literature, and primarily rely on recombining existing technical methods rather than introducing new research questions.
- These findings suggest that current AI research agents are better suited for local elaboration within existing conceptual neighborhoods than for broadening the scope of scientific exploration.

---

[Dr-CiK: A Testbed for Foresight-Driven Agents](http://arxiv.org/abs/2605.27904)

- Dr-CiK: introduces a benchmark for evaluating whether agents can retrieve forecasting-relevant context, reject distractors, and synthesize evidence for grounded time-series forecasting.
- The framework utilizes a three-phase generation pipeline to create controllable environments with forecast-dependent distractors, enabling stage-wise evaluation of failures from deep research to forecasting.
- Experimental results demonstrate that while supporting evidence improves forecasting, current deep research agents struggle with distractor rejection and evidence synthesis, highlighting the need for foresight-driven agents.

---

[SKILLC: Learning Autonomous Skill Internalization in LLM Agents via Contrastive Credit Assignment](http://arxiv.org/abs/2605.27899)

- SKILLC: introduces a framework that enables LLMs to internalize skills by using paired contrastive rollouts and a dual-stream advantage estimator to redirect learning signals toward autonomous success.
- The framework utilizes an internalization-aware curriculum that dynamically adjusts attribution strength, rollout allocation, and active skill sets based on validation-level contrastive gaps.
- By resolving internalization blindness, SKILLC ensures that LLMs acquire genuine autonomous competence without requiring external skill access at inference time.

---

[A Unified Framework for the Evaluation of LLM Agentic Capabilities](http://arxiv.org/abs/2605.27898)

- Unified Framework for the Evaluation of LLM Agentic Capabilities: introduces a standardized evaluation system that decouples agent architecture from benchmark-specific implementations to provide clean measurements of LLM agentic capabilities.
- The framework integrates diverse benchmarks into a unified instruction–tool–environment format, executes agents within a fixed ReAct-style architecture, and provides an optional offline setting to isolate environmental volatility from model performance.
- It further introduces unified efficiency metrics and a decision- and execution-level failure taxonomy to disentangle intrinsic LLM capabilities from framework-induced artifacts across single-agent, multi-agent, and safety-critical scenarios.

---

[Reflective Dialogue between Teacher and Solver Agents for Video Question Answering](http://arxiv.org/abs/2605.27885)

- RD (Reflective Dialogue): introduces a training-free adaptation method for Video QA by constructing multi-turn conversations between Teacher and Solver agents to serve as static context for LLMs.
- The framework utilizes a Teacher agent to provide correctness feedback and a Solver agent to generate visual grounding explanations, which are then prepended to test questions to guide reasoning.
- By pre-constructing reflective dialogues from support sets, the method avoids per-question retry loops and improves generalization across specialized egocentric video domains.

---

[VibeSearchBench: Benchmarking Long-horizon Proactive Search in the Wild](http://arxiv.org/abs/2605.27882)

- VibeSearchBench: introduces a benchmark for evaluating LLM agents on long-horizon proactive search, utilizing a User Persona, Progressive-disclosure User Simulator, Agent Harness, Schema-free Knowledge Graph, and Graph-matching Evaluation Framework.
- The framework models search as a bidirectional convergence process where agents must proactively elicit evolving user intent through multi-turn dialogue rather than relying on single-turn queries.
- Experimental results across seven frontier LLMs reveal that current models struggle with context overflow, inefficient intent elicitation, and the construction of structurally complex knowledge graphs.

---

[Retrieval, Reward, and Training Protocols: What Matters in Training Search Agents?](http://arxiv.org/abs/2605.27881)

- Search Agent Training Framework: introduces a controlled empirical study isolating retrieval environment, reward design, and training protocol as key factors for training effective search agents.
- The study demonstrates that retrieval corpus completeness is a more decisive factor for performance than the specific choice of training algorithm.
- The research provides practical guidelines for optimizing search agents, highlighting that heuristic process-level credit assignment does not consistently outperform simple outcome-level supervision.

---

[Towards Faithful Agentic XAI: A Verification Method and an Open-World Benchmark for Better Model Faithfulness](http://arxiv.org/abs/2605.27879)

- FAX (Faithful Agentic XAI): introduces a verification-centric framework that improves explanation faithfulness by decomposing draft explanations into testable claims and cross-checking them against inherently faithful tools before final generation.
- The framework utilizes an LLM controller to orchestrate planning-, tool execution-, and verification-agents, ensuring that only corroborated claims are presented to the user.
- The authors also introduce CRAFTER-XAI-Bench, an open-world reinforcement learning benchmark designed to assess model-specific faithfulness by distinguishing between learned policy behaviors and generic environment knowledge.

---

[AIBuildAI-2: A Knowledge-Enhanced Agent for Automatically Building AI Models](http://arxiv.org/abs/2605.27873)

- AIBuildAI-2: introduces a hierarchical, knowledge-enhanced agent architecture that utilizes a continually evolving, two-level knowledge system to automate end-to-end AI model development.
- The framework coordinates a manager agent and specialized sub-agents—designer, coder, and tuner—to perform parallel, multi-step reasoning across isolated solution repositories.
- By dynamically loading relevant knowledge from a hierarchical repository and distilling experience from completed runs, the system achieves performance competitive with expert human practitioners.

---

[FundaPod: A Multi-Persona Agent Pod Platform with Knowledge Graph Memory for AI-Assisted Fundamental Investment Research](http://arxiv.org/abs/2605.27864)

- FundaPod: introduces a multi-persona agent platform for AI-assisted fundamental investment research that utilizes an independence-preserving architecture to support human-centric decision-support tasks.
- The system employs persona-distilled agents that reason in isolation, a declarative skill registry for workflow extensibility, and a knowledge graph "second brain" to maintain persistent, queryable research memory.
- FundaPod ensures evidentiary rigor by linking all generated claims to source artifacts in an append-only evidence store, enabling transparent and auditable investment research.

---

[MolLingo: Molecule-Native Representations for LLM-Powered Scientific Agents](http://arxiv.org/abs/2605.27853)

- MolLingo: introduces a multi-agent framework that coordinates an Orchestrator Agent, a Literature Agent, and a Chemist Agent through a Shared Memory Module to automate molecular design.
- The framework utilizes BFE (BRICS-based Fragment Enumeration) to decompose molecules into chemically meaningful blocks, aligning molecular structure with the semantic space of LLMs.
- MolLingo grounds the Chemist Agent's reasoning in structural and biological context derived from molecular docking and ADMET Oracle Models to enable iterative, evidence-driven molecular optimization.

---

[TCP-MCP: Landscape-Guided Co-Evolution of Prompts and Communication Topologies for Multi-Agent Systems](http://arxiv.org/abs/2605.27850)

- TCP-MCP (Topology-Coupled Prompting for Multi-Agent Collaborative Problem-Solving): introduces a co-evolutionary framework that treats prompts and communication topologies as a unified genome to optimize multi-agent systems under performance, cost, and complexity constraints.
- The framework utilizes an initialization-time landscape probe and cross-generational Pareto-front diagnostics to adaptively guide the search process through structural crossover and mutation operators.
- Experimental results on MMLU, MMLU-Pro, and GSM8K demonstrate that jointly evolving prompts and communication structures achieves superior task-adaptive performance compared to fixed-topology or staged-optimization baselines.

---

[Decentralized Parameter-Free Online Learning with Compressed Gossip](http://arxiv.org/abs/2605.27831)

- DECO-EF (DEcentralized COin-betting with Error Feedback): introduces a decentralized parameter-free online learning algorithm that combines coin-betting predictions with compressed difference-based gossip to achieve sublinear network-regret.
- The framework utilizes error feedback to manage compression residuals, ensuring that local agents maintain accurate trackers of the global state despite lossy communication.
- By employing a radial Hilbert-space coin-betting potential, the algorithm eliminates the need for tuning learning rates to the horizon or comparator norm in decentralized settings.

---

[MRMMIA: Membership Inference Attacks on Memory in Chat Agents](http://arxiv.org/abs/2605.27825)

- MRMMIA (Multi-Recall Memory Membership Inference Attack): introduces a unified attack framework that utilizes multiple recall probes to extract membership signals from chat agent memory across black-box, gray-box, and white-box settings.
- The framework employs a Query Generator to create diverse recall probes, which are then processed by a Response Scorer and an optional Memory Scorer to determine if a candidate statement exists within the agent's memory.
- Experimental results demonstrate that MRMMIA consistently outperforms existing LLM and RAG-based membership inference baselines by effectively leveraging the unique characteristics of agent memory systems.

---

[EgoBench: An Interactive Egocentric Multimodal Benchmark for Tool-Using Agents](http://arxiv.org/abs/2605.27820)

- EgoBench: introduces an interactive multimodal benchmark for evaluating tool-using agents grounded in egocentric video, utilizing a three-stage synergistic pipeline to assess perception, reasoning, and interaction.
- The framework incorporates a multi-agent simulated user with an Actor-Evaluator-Summarizer structure to generate high-fidelity, task-aligned responses for robust interaction evaluation.
- A deterministic joint validation framework ensures objective assessment by simultaneously verifying process-based tool-call coverage and result-based database state equivalence.

---

[Are Diffusion Language Models Good Database Analysts?](http://arxiv.org/abs/2605.27791)

- SQL-D1 (Agentic NL2SQL framework): introduces a four-stage agentic pipeline that integrates database-aware context engineering, test-time scaling, and iterative refinement to improve the performance of Diffusion Language Models in NL2SQL tasks.
- The framework employs a dual-track inference pipeline that supports both direct model-based generation and a structured agentic workflow comprising Retriever (Ar), Generator (Ag), Verifier (Av), and Selector (As) modules.
- By utilizing iterative denoising and agentic coordination, the system addresses the sequential error propagation limitations of autoregressive models while providing a flexible efficiency-accuracy trade-off for database reasoning.

---

[Long Live the Librarian! A Persistent Search Sub-Agent for Energy-Efficient Multi-Agent Software Engineering Systems](http://arxiv.org/abs/2605.27787)

- Librarian: introduces a persistent search sub-agent that reduces energy consumption in multi-agent software engineering systems by eliminating redundant file exploration across agent invocations.
- The framework utilizes a persistent session and pointer-only submission mechanism to minimize output tokens, which are identified as the primary source of GPU energy waste in LLM-based agentic systems.
- Experimental results on SWE-Bench Verified demonstrate that integrating the Librarian reduces per-episode GPU energy consumption by up to 25% while maintaining task performance.

---

[A Query Engine for the Agents](http://arxiv.org/abs/2605.27785)

- Hyperparam: introduces a browser-native lakehouse client designed for agentic data systems, utilizing Squirreling (async SQL engine), Hyparquet (Parquet reader), and Icebird (Iceberg client) to enable efficient, in-process querying of unstructured data.
- The framework leverages per-cell, async-native SQL execution to minimize LLM inference costs and latency by ensuring expensive operations are only performed when demanded by downstream operators.
- By operating directly within the client JavaScript runtime and fetching data from object storage, the architecture eliminates the need for intermediate query gateways or server-side infrastructure.

---

[Diagnosing Live Within-Policy Instruction Conflicts in LLM Agents with Witnessed Resolution Profiles](http://arxiv.org/abs/2605.27784)

- WIRE (Witnessed Intra-policy Rule Evaluation): introduces a neuro-symbolic pipeline that identifies and diagnoses latent instruction conflicts within a single LLM prompt policy by testing them against concrete behavioral witnesses.
- The framework extracts prescriptive rules, encodes them into symbolic clauses, uses satisfiability checks to nominate hard-collision candidates, and evaluates LLM resolution profiles on realized concrete states.
- By measuring joint compliance, single-rule satisfaction, and violation rates, the pipeline exposes distinct behavioral signatures of LLMs when faced with conflicting internal policy instructions.

---

#### 26th May 2026


[Examining the Challenges of Intellectual Property in AI-Generated Productions](http://arxiv.org/abs/2605.26590)

- Intellectual Property in AI-Generated Productions framework: examines the legal challenges of assigning ownership and moral rights to autonomous AI-generated works within existing intellectual property regimes.
- The paper analyzes the regulatory gaps in Iranian law compared to international systems, focusing on the roles of AI-based generative systems and human users in the creative process.
- It proposes policy recommendations, including revising copyright statutes, creating quasi-copyright for AI outputs, and enhancing international cooperation to balance innovation with human creativity.

---


[ORCA: An End-to-End Interactive Copilot for Optimized Root Cause Analysis](http://arxiv.org/abs/2605.27022)

- ORCA: introduces an interactive agentic copilot that orchestrates a multi-agent framework to guide users through end-to-end causal analysis, including data preprocessing, causal discovery, and root cause analysis.
- The system leverages an LLM backbone to translate natural language user intents into executable Python scripts while managing complex causal workflows through a centralized orchestration mechanism.
- ORCA incorporates human-in-the-loop feedback and domain knowledge integration to address the methodological complexity of causal inference for non-expert users in real-world domains.

---


[How to Mitigate the Distribution Shift Problem in Robotics Control: A Robust and Adaptive Approach Based on Offline to Online Imitation Learning](http://arxiv.org/abs/2605.25414)

- RAIL (Robust and Adaptive Imitation Learning): introduces a lifelong imitation learning framework that mitigates distribution shift by leveraging supplementary demonstrations with a regularized discriminator during offline training and performing self-supervised online adaptation.
- The framework utilizes a novel regularization term for the discriminator to accurately estimate sample optimality, enabling stable weighted behavior cloning for both offline and online phases.
- RAIL incorporates an update time management mechanism that triggers online policy adaptation only when a significant distribution shift is detected, ensuring stable performance in non-stationary robotics environments.

---

[MerLean-Prover: A Recursive Looping Harness for End-to-End Lean 4 Theorem Proving](http://arxiv.org/abs/2605.26959)

- MerLean-Prover: introduces a recursive harness for Lean 4 theorem proving that decomposes complex proofs into single-objective tasks handled by Planning Agent, Check Agent, and Lean Agent.
- The system utilizes a Shared Proof Plan as the primary global state, allowing the Recursive Outer Loop to manage dependencies and trigger replanning when verification fails.
- By assigning one focused objective per LLM invocation, the harness achieves competitive performance on formal mathematics benchmarks without requiring fine-tuning or theorem-specific scaffolding.

---


[EgoProx: Evaluating MLLMs on Egocentric 3D Proximity Reasoning Across a Cognitive Hierarchy](http://arxiv.org/abs/2605.24456)

- EgoProx: introduces a benchmark for evaluating LLMs on egocentric 3D proximity reasoning tasks organized along a cognitive hierarchy of intention, exploration, exploitation, and chain-of-actions.
- The framework utilizes an agent-based data engine comprising a Salient Clip Sampler, 3D Analysis Toolset, Occupancy Map Generator, Exploration Path Generator, Spatial Calculator, Gaze Parser, Affordance Detector, Keystep Extraction Tool, and Chain Constructor to synthesize high-quality VQA data.
- Experimental results demonstrate that while current LLMs possess latent spatial knowledge, they require structured instruction tuning to effectively perform complex 3D reasoning from first-person visual inputs.

---

[Benchmarking the Limits of In-Context Reinforcement Learning for Ad-Hoc Teamwork](http://arxiv.org/abs/2605.24423)

- ICRL4AHT: introduces a large-scale benchmark and reproducible pipeline for evaluating In-Context Reinforcement Learning (ICRL) in Ad-Hoc Teamwork (AHT) settings, utilizing Benchmark Manifest, Teammate Suite, Rollout &amp; Logging, Dataset Storage &amp; Loader, ICRL Training, and Online Evaluation.
- The framework evaluates representative history-conditioned ICRL baselines, specifically Algorithm Distillation (AD) and Decision-Pretrained Transformer (DPT), across diverse teammate and layout generalization tracks.
- Results demonstrate that current ICRL sequence models struggle with strategic partner inference in the OvercookedV2 environment, often underperforming simple random baselines and failing to exhibit robust in-context adaptation.

---

[HABERMOLT: Delegating Deliberation to AI Representatives](http://arxiv.org/abs/2605.24413)

- HABERMOLT: introduces a platform for AI-delegated deliberation where autonomous agents represent human users in democratic processes by maintaining persistent memory, generating opinions, and participating in collective consensus-building.
- The architecture utilizes a bring-your-own-statement mechanism and Schulze ranking to aggregate diverse agent-authored inputs into actionable consensus statements.
- The system incorporates a revision loop allowing users to inspect, edit, and correct agent-generated artifacts to ensure alignment with their evolving values.

---

[Attested Tool-Server Admission: A Security Extension to the Model Context Protocol](http://arxiv.org/abs/2605.24248)

- ATSA: introduces a security extension for the Model Context Protocol that adds a host-side gate and server-published clearance assertions to enable authenticated, least-privilege tool access.
- The framework utilizes a pinned trust root and per-server tool allowlists to prevent unauthorized tool execution by LLMs, ensuring all admission decisions are recorded in a tamper-evident audit log.
- ATSA operates as an additive layer above existing MCP infrastructure, requiring no changes to current message formats while providing robust protection against prompt-injection and confused-deputy attacks.

---

[Beyond Final Answers: Auditing Trajectory-Level Hallucinations in Multi-Agent Industrial Workflows](http://arxiv.org/abs/2605.24219)

- Trajel: introduces a dataset and evaluation framework for auditing trajectory-level hallucinations in multi-agent industrial workflows using a five-type taxonomy and expert-annotated agent traces.
- The framework utilizes Agentic System, Trajectory Analysis, Trajel ML Model Framework, Evaluation Stage, LLM-as-a-Judge, Human Review, and Execution-Quality Signals to identify structural deviations in sequential, tool-mediated agent loops.
- Empirical results demonstrate that procedural hallucinations dominate agentic failures and that lightweight execution-quality signals, particularly clarity-and-justification, outperform supervised classifiers in predicting hallucination.

---

[MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation](http://arxiv.org/abs/2605.27366)

- MUSE-Autoskill: introduces a skill-centric agent framework that enables LLMs to continuously improve task-solving capabilities through a unified lifecycle of creation, memory, management, evaluation, and refinement.
- The framework integrates a multi-level memory system, including a novel skill-level memory, to accumulate per-skill experience and facilitate effective reuse and cross-agent transfer.
- MUSE-Autoskill employs adaptive context compression over a DAG of ReAct turns to handle long-horizon tasks while maintaining performance and reducing token usage.

---

[Natural Language Query to Configuration for Retrieval Agents](http://arxiv.org/abs/2605.27361)

- BRANE: introduces a framework that dynamically selects the optimal retrieval pipeline configuration per query to minimize cost while meeting accuracy targets.
- The framework utilizes an LLM to extract workload-specific query characteristics, which are then used by lightweight per-configuration predictors to estimate performance and guide selection.
- By employing Lagrangian routing, BRANE effectively navigates the cost-quality Pareto frontier, outperforming static tuning and existing routing baselines across multiple benchmarks.

---

[GENESIS: Harnessing AI Agents for Autonomous 6G RAN Synthesis, Research, and Testing](http://arxiv.org/abs/2605.27360)

- GENESIS: introduces an agentic framework that automates the Radio Access Network (RAN) R&D life-cycle by converting high-level intents into validated solutions through a closed-loop system of Agentic Layer, Deterministic Execution Layer, Substrate Layer, and Knowledge Layer.
- The framework utilizes a modular architecture of agents, skills, and hooks to synthesize, test, harden, optimize, discover, and secure network functionalities across a staged validation continuum ranging from simulation to over-the-air deployment.
- By grounding LLM reasoning in a persistent knowledge plane (SYNAPSE) and enforcing safety through deterministic hooks, GENESIS achieves autonomous end-to-end RAN engineering while maintaining auditable provenance and spec-to-code traceability.

---

[Exploring Agent Interactions in MoltBook through Social Network Analysis](http://arxiv.org/abs/2605.27349)

- MoltBook Social Network Analysis Framework: utilizes the Hermes Agent (powered by Minimax 2.7 LLM) to perform automated data collection and analysis of autonomous agent interactions within the MoltBook ecosystem.
- The framework integrates Social Network Analysis metrics, sentiment analysis, and Gephi-based visualization to decode the structural topology and semantic content of agent-native discourse.
- This study demonstrates a human-AI collaborative model where the Hermes Agent manages high-volume data processing while researchers maintain oversight of conceptual design and interpretation.

---

[FINHARNESS: An Inline Lifecycle Safety Harness for Finance LLM Agents](http://arxiv.org/abs/2605.27333)

- FINHARNESS: introduces an inline safety harness that wraps finance LLM agents to block unauthorized actions while approving legitimate workflows by integrating QUERY MONITOR, TOOL MONITOR, and CASCADE components.
- The framework utilizes FIRED-SIGNAL DYNAMIC INJECTION to provide the agent with structured evidence, enabling autonomous refusal or re-planning based on real-time risk assessments.
- By employing a cost-aware CASCADE that routes verification between lightweight and advanced LLM judges, the system maintains high safety standards with significantly reduced computational overhead.

---

[Maat: The Agentic Legal Research Assistant for Competition Protection](http://arxiv.org/abs/2605.27331)

- Maat: introduces a multi-turn agentic system for competition law research that utilizes a ReAct Agent to orchestrate specialized tools including Database Search, Web Search, Answer Case, Answer Theoretical, and Ask Clarification.
- The architecture leverages a Memory Layer consisting of a Vector Database and a Scratchpad to maintain context and reasoning history for accurate, citation-backed legal analysis.
- Maat integrates a purpose-built dataset of EU and German competition cases, employing LLMs for metadata extraction and semantic search to outperform existing LLMs in case-specific legal research tasks.

---

[Governed Evolution of Agent Runtimes through Executable Operational Cognition](http://arxiv.org/abs/2605.27328)

- Governed Evolution of Agent Runtimes through Executable Operational Cognition: introduces a framework for managing agent-generated artifacts as persistent, governed capabilities rather than transient outputs, utilizing Specialized Governance Agents, Governed Runtime Kernel, Knowledge-Grounded Runtime Graph, Execution and Artifact Substrate, and Observability.
- The framework employs HarnessMutation to apply controlled, validated transformations to harness configurations, ensuring that runtime evolution remains observable, auditable, and constrained.
- By integrating a Knowledge-Grounded Runtime Graph, the system enables structured, dependency-aware reuse of operational capabilities, shifting agent memory from passive retrieval to active, executable accumulation.

---

[Modeling Agentic Technical Debt and Stochastic Tax: A Standalone Framework for Measurement, Simulation, and Dashboarding](http://arxiv.org/abs/2605.27320)

- ATD-ST Framework: introduces a formal model that distinguishes between Agentic Technical Debt (accumulated design liability) and Stochastic Tax (recurring operating burden) to support measurement and simulation of agentic AI systems.
- The framework utilizes a debt-amplified cost structure where Stochastic Tax is modeled as a function of a baseline floor, accumulated debt, and operating-exposure variables like adoption and surface area.
- This approach enables managers to decompose rising operational costs into either remediable technical debt or unavoidable stochastic operating burdens, facilitating data-driven governance of LLM-based workflows.

---

[SIA: Self Improving AI with Harness &amp; Weight Updates](http://arxiv.org/abs/2605.27276)

- SIA: introduces a self-improving loop that iteratively optimizes both the task-specific agent's harness and its model weights using a Feedback-Agent.
- The system utilizes a Meta-Agent to initialize the scaffold and a Feedback-Agent to dynamically select between harness updates and RL-based weight updates.
- SIA demonstrates consistent performance gains across legal classification, GPU kernel optimization, and biological data denoising by combining externalized infrastructure changes with internalized model knowledge.

---

[Can Retrieval Heads See Images? Multimodal Retrieval Heads in Long-Context Vision-Language Models](http://arxiv.org/abs/2605.27243)

- MMRetHeads: introduces a method to identify sparse, intrinsic, and causally important attention heads in LVLMs that facilitate evidence location across interleaved text and images.
- The framework utilizes an attention-based retrieval score and null-question calibration to isolate specific heads that concentrate attention from question tokens to task-relevant needles.
- Experimental results demonstrate that masking these identified MMRetHeads significantly degrades performance on long-context retrieval and downstream reasoning tasks, while their application as a retriever improves document re-ranking.

---

[ENPMR-Bench: Benchmarking Proactive Memory Retrieval for Emotional Support Agents](http://arxiv.org/abs/2605.27240)

- ENPMR-Bench: introduces a diagnostic benchmark designed to evaluate the capacity of LLMs to infer latent emotional needs and proactively retrieve appropriate memories for empathetic support.
- The framework grounds memory retrieval in Maslow’s hierarchy of needs, mapping specific unmet needs to distinct memory categories to assess the alignment between user requirements and retrieved information.
- Experimental results demonstrate that current embedding-based and LLM-driven retrieval paradigms struggle with emotional appropriateness, highlighting a significant performance gap compared to golden memory conditions.

---

[EVIACT: An Evidence-to-Action Framework for Agentic Program Repair](http://arxiv.org/abs/2605.27238)

- EVIACT: introduces an evidence-driven agentic APR framework that coordinates Setup, Localize, Patch, and Verify stages using Retrieval Scaffold, Compile Gate, and Test-Driven Gate to transform execution evidence into repair actions.
- The framework utilizes a Retrieval Scaffold for structurally grounded localization, a Compile Gate for filtering invalid edits, and a Test-Driven Gate to ensure target-test recovery before full regression.
- By anchoring repair trajectories in traceable execution evidence, EVIACT reduces localization drift and validation costs compared to existing workflow-based APR systems.

---

[Learning to Act under Noise: Enhancing Agent Robustness via Noisy Environments](http://arxiv.org/abs/2605.27209)

- NoisyAgent: introduces an agentic training framework that explicitly incorporates environmental imperfections into the learning process to enhance robustness.
- The framework utilizes an automatic noise injection pipeline to simulate user-side ambiguity and tool-side execution anomalies during training.
- An adaptive training strategy employs hybrid rollouts and a curriculum-based scheduler to progressively increase noise difficulty, ensuring stable optimization under heterogeneous conditions.

---

[FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation](http://arxiv.org/abs/2605.27178)

- FoundObj: introduces a label-free 3D object segmentation framework that utilizes an Object Discovery Agent guided by Semantic Reward Module and Geometric Reward Module to identify objects without human annotations.
- The framework leverages DINOv2 and TRELLIS foundation models to provide complementary semantic and geometric feedback to the Object Discovery Agent during reinforcement learning.
- FoundObj employs a superpoint-based approach to incrementally merge regions into valid object candidates, demonstrating strong generalization across zero-shot and long-tail 3D scene benchmarks.

---

[VitaBench 2.0: Evaluating Personalized and Proactive Agents in Long-Term User Interactions](http://arxiv.org/abs/2605.27141)

- VitaBench 2.0: introduces a benchmark for evaluating personalized and proactive LLM agents in long-term user interactions, utilizing User Profile, User Preferences, Interaction History, Memory Module, Agentic Memory, RAG Memory, Task Agent, User Simulator, and Evaluator LLM.
- The framework evaluates agent performance across temporally ordered task sequences, requiring models to infer, maintain, and update user preferences while managing memory through either Agentic Memory or RAG Memory.
- Experimental results reveal that current LLMs struggle with long-term personalization and proactive information acquisition, identifying memory management and preference inference as critical bottlenecks for agentic systems.

---

[StepOPSD: Step-Aware Online Preference Distillation for Agent Reinforcement Learning](http://arxiv.org/abs/2605.27140)

- StepOPSD: introduces a post-rollout preference self-distillation framework that decomposes agent trajectories into action-centered segments to redistribute credit via hindsight-enriched teacher contexts.
- The framework integrates a specialized reward manager and a lightweight trainer subclass to modulate the GRPO advantage without altering online rollout dynamics.
- StepOPSD employs a two-knob law where local weight clipping stabilizes training while global mixing strength is tuned based on the specific agentic task requirements.

---

[Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](http://arxiv.org/abs/2605.27134)

- GUIEvalKit: introduces a comprehensive framework for benchmarking VLM-based agents in mobile GUI navigation, integrating CLI Interface, Episode Data Files, Unified Metrics, Prediction Matching, History Context Formulation, Prompt Building, Prediction Unification, Answer Parsing, Execution State Preservation, Restart (Optional), Distributed MLLM Deployment, Unified Generation Interface, and Concurrent Inference.
- The framework facilitates rigorous evaluation by standardizing model inference and providing a unified action space for diverse GUI benchmarks.
- It enables systematic analysis of data scaling, reinforcement-based finetuning, and the impact of reasoning capabilities on agent performance.

---

[Rethinking Agentic RAG: Toward LLM-Driven Logical Retrieval Beyond Embeddings](http://arxiv.org/abs/2605.27123)

- LOGICALRAG: introduces an agentic RAG framework that delegates retrieval control to the LLM by utilizing structured logical expressions over a lightweight inverted-index backend.
- The framework replaces complex embedding-based or graph-based retrieval backends with a deterministic execution layer that supports Boolean logic and adjustable matching granularity.
- Experimental results demonstrate that this approach achieves performance parity with hybrid baselines while significantly reducing construction costs and lowering hallucination rates.

---

[Position: AI Safety Requires Effective Controllability](http://arxiv.org/abs/2605.27117)

- CAS (Controllable AI Systems): introduces a control-centric architectural framework for AI safety that integrates Guardrails, Constraint compiler, Agent runtime, Action planner, Tool-Function call, Specialist agent, and Audit log to ensure persistent runtime authority.
- The framework addresses the limitations of alignment by requiring that LLMs remain interruptible, overridable, and constrainable during long-horizon agentic execution.
- The paper introduces CONTROLBENCH, a benchmark for evaluating controllability failures in high-risk agentic scenarios where current safety mechanisms often fail to provide persistent runtime control.

---

[Improved Hardness Results for Nash Social Welfare, Budgeted Allocation and GAP via the Unique Games Conjecture](http://arxiv.org/abs/2605.27098)

- Indivisible Good Allocation Framework: introduces a novel dictator test based on a pairwise independent probability distribution to establish improved inapproximability results for fundamental allocation objectives.
- The framework utilizes a reduction from the Unique Games Conjecture to prove that approximating max Nash welfare, max budgeted allocation, and the max generalized assignment problem is NP-hard beyond specific constant factors.
- By constructing instances with large, small, and dummy goods, the approach effectively separates dictator-like allocation functions from those far from dictators, providing tighter computational lower bounds.

---

[Can Broad Biomedical Knowledge be Contextualized into Scenario-Grounded Propositions?](http://arxiv.org/abs/2605.27082)

- SCENE (Scenario-Contextualized Evidence and Knowledge Engine): introduces a bi-level multi-agent framework that implements knowledge contextualization as an iterative search process, utilizing Upper-Level Direction Planning, Lower-Level Knowledge Discovery, Scenario Adapter, Feedback Loop, Evidence Gate, Evolutionary Grounding, and Pareto Selection.
- The framework bridges abstract biomedical knowledge and concrete dataset evidence by translating broad priors into bounded search directions that guide multi-objective evolutionary discovery.
- SCENE produces traceable, scenario-grounded propositions that pair search directions with executable rules and evidence records, enabling domain experts to inspect and validate candidate hypotheses.

---

[Cost of Structural Learning under Censored Feedback: A Threshold-Bandit Approach](http://arxiv.org/abs/2605.27076)

- TAC-MAB (Threshold-Activated Cooperative Multi-Armed Bandit): introduces a framework for multi-agent task allocation where rewards are gated by unknown coalition-size thresholds and feedback is fully censored below feasibility.
- The paper proposes C-TAC as a centralized baseline and D-TAC as a decentralized protocol, utilizing a Virtual Coordinator and a structured Communication Protocol to achieve near-centralized performance with significantly reduced message overhead.
- By decomposing regret into structural search and statistical monitoring, the authors demonstrate that D-TAC effectively resolves feasibility uncertainty while maintaining communication efficiency through event-triggered synchronization of agent Belief States.

---

[IPIBench: Evaluating Interactive Proactive Intelligence of MLLMs under Continuous Streams](http://arxiv.org/abs/2605.27074)

- IPI-Agent: introduces a training-free agentic framework that utilizes an Interaction-Control Policy and a Temporal-Gating Mechanism to coordinate reactive queries, proactive instructions, and management instructions for MLLMs under continuous streaming video.
- The framework employs an Intent Router and a Memory Tool to maintain persistent proactive objectives while enabling interleaved reactive interactions.
- A Gating Tool within the Temporal-Gating Mechanism improves proactive triggering stability by regulating response activation based on temporal similarity variations between task proposals and visual observations.

---

[Learning to Orchestrate Agents under Uncertainty](http://arxiv.org/abs/2605.27073)

- BOT-Orch (Bandit-based Orchestration with OT alignment): introduces a lightweight framework that recasts agent orchestration as a bandit problem regularized by Optimal Transport distances to manage uncertainty in agent reliability and output distributions.
- The framework integrates survival-based rewards to account for latent task difficulty and censoring, while employing a Boltzmann policy to balance exploration and exploitation under non-stationary conditions.
- Theoretical analysis establishes sublinear regret and convergence properties, while empirical results demonstrate superior performance in synthetic and human-AI triage tasks under distribution shift.

---

[Traceable Knowledge Graph Reasoning Enables LLM-Assisted Decision Support for Industrial VOCs in the Steel Industry](http://arxiv.org/abs/2605.27071)

- Chat-ISV: introduces a multi-agent KG-enhanced framework that integrates automated knowledge extraction, topology optimization, and hierarchical query routing to provide traceable decision support for industrial VOC governance.
- The system utilizes a chunk-centered star topology in a Neo4j database to link fragmented scientific literature with structured entity relations, ensuring high factual reliability and source-level traceability.
- Chat-ISV employs a three-tier multi-agent architecture that coordinates graph-reasoning, literature-retrieval, and open-domain agents to mitigate LLM hallucinations in specialized industrial domains.

---

[QUACK: Questioning, Understanding, and Auditing Communicated Knowledge in Multimodal Social Deduction Agents](http://arxiv.org/abs/2605.27068)

- QUACK (Questioning, Understanding, and Auditing Communicated Knowledge): introduces a multimodal social deduction environment and evaluation framework for auditing the grounding of LLM or VLM agents in partially observable settings.
- The framework utilizes a Statement Verification Pipeline to reconstruct ground-truth trajectories from engine logs, enabling the automatic detection of spatial hallucinations, unsupported accusations, deception collapses, and language-action inconsistencies.
- By evaluating frontier LLMs or VLMs across homogeneous and adversarial settings, the research demonstrates that agents exhibit systematic grounding failures that remain invisible to standard game-outcome metrics.

---

[BEAT: Rhythm-Elastic Alignment for Agentic Music-guided Movie Trailer Generation](http://arxiv.org/abs/2605.27067)

- BEAT: introduces a framework for music-guided movie trailer generation that grounds core alignment in a fine-tunable encoder while coordinating creative decisions through a five-phase agentic pipeline.
- The framework utilizes MuVA for cross-modal alignment and Bar-DP for elastic many-to-one shot-to-bar mapping, enabling rhythmically coherent trailer editing that adapts to musical dynamics.
- BEAT incorporates an agentic pipeline with an LLM/VLM-based Critic to perform iterative quality refinement, supported by the TrailerArena benchmark for comprehensive evaluation across selection, ordering, composition, and perceptual dimensions.

---

[TPS-Drive: Task-Guided Representation Purification for VLM-based Autonomous Driving](http://arxiv.org/abs/2605.27038)

- TPS-Drive: introduces a framework that utilizes an Agent-Centric Tokenizer to isolate spatial redundancy from dynamic agents, enabling the VLM to perform reasoning in a purified 3D space.
- The framework employs a decoupled pipeline that sequentially executes scene understanding, future forecasting, and action generation, optimized through a progressive three-stage training paradigm.
- By leveraging a frozen 3D detection head for task-guided vector quantization, the model effectively mitigates representation interference and improves safety-critical planning performance.

---

[ReasonOps: A Unified Operational Paradigm for Trustworthy Verified LLM Reasoning](http://arxiv.org/abs/2605.27014)

- ReasonOps: introduces a unified operational paradigm for trustworthy verified LLM reasoning by treating inference as a continuously monitored, reliability-aware lifecycle rather than a static generation task.
- The architecture integrates Natural Language Understanding, Autoformalization, Neuro-symbolic Reasoning, Formal Verification, Runtime Monitoring and Assurance, Probabilistic Reliability Analysis, and Safety Enforcement and Certification to ensure symbolic correctness and behavioral safety.
- By incorporating a Continuous Feedback and Improvement Loop, the framework enables dynamic repair of reasoning trajectories, refinement of constraints, and ongoing validation of LLM outputs in safety-critical applications.

---

[Probabilistic Recurrent Intention Switching Model](http://arxiv.org/abs/2605.26998)

- PRISM: introduces a framework for multi-intention inverse reinforcement learning that uses a recurrent neural network to model latent intention switching and recovers per-intention reward functions via an expectation-maximization algorithm.
- The framework utilizes a gating function to map observation histories to intention distributions, enabling the decomposition of the inference problem into independent, closed-form reward subproblems solvable by IAVI.
- PRISM demonstrates scalability and interpretability across diverse domains, including non-Markovian gridworlds, biological mouse navigation, and large-scale robotic manipulation, without requiring manual specification of temporal horizons.

---

[ChartAct: A Benchmark for Dynamic Chart Understanding](http://arxiv.org/abs/2605.26994)

- ChartAct: introduces a benchmark for dynamic chart understanding that requires LLMs to perform interactive actions to reveal hidden evidence and reason over changing chart states.
- The framework evaluates LLMs across two environments, Dynamic Chart and Dashboard Chart, to assess their ability to handle varying levels of visual context and interactive complexity.
- The benchmark utilizes an answer-driven evaluation protocol with an LLM judge to assess performance on 1,440 question-answer samples derived from real-world interactive charts.

---

[Efficient Agentic Reinforcement Learning with On-Policy Intrinsic Knowledge Boundary Enhancement](http://arxiv.org/abs/2605.26952)

- AKBE (Agentic Knowledge Boundary Enhancement): introduces an on-policy method that dynamically probes the model's intrinsic knowledge boundary through dual-path rollouts to construct supervisory signals that eliminate redundant tool calls.
- The framework categorizes trajectories into Tool-dependent, Efficiency, Hallucination, and Both-wrong signals to provide fine-grained, instance-level guidance without modifying the underlying RL reward.
- Experimental results demonstrate that AKBE improves task accuracy and tool productivity across multiple benchmarks while maintaining plug-and-play compatibility with diverse agentic RL algorithms.

---

[From Norms to Indicators (N2I-RAG): An Agentic Retrieval-Augmented Generation Framework for Legal Indicator Computation](http://arxiv.org/abs/2605.26926)

- N2I-RAG (From Norms to Indicators): introduces an agentic retrieval-augmented generation framework designed to automate the computation of legal indicators from complex, heterogeneous legal documents.
- The framework utilizes a multi-agent pipeline, including Metadata Retriever Agent, Context Retriever Agent, Context Grader Agent, Generator Agent, Groundedness Grader Agent, Answer Relevance Grader Agent, Query Disambiguator Agent, and Binary QA Agent, to ensure traceable and evidence-based legal analysis.
- By integrating VLM-based OCR, BGE-M3 Embedding, and ChromaDB, the system bridges the gap between open-text legal language and standardized binary evaluation grids, significantly reducing hallucinations and improving interpretability.

---

[TADDLE: A Tool-Augmented Agent for Detecting Deficient LLM-Generated Peer Reviews](http://arxiv.org/abs/2605.26911)

- TADDLE: introduces a tool-augmented agent framework that decomposes the auditing of LLM-generated peer reviews into specialized analysis stages orchestrated by an agent and synthesized by a fine-tuned INTEGRATE module.
- The framework utilizes an Orchestrator to route review segments through four specialized analysis tools—VERIFY, CORRECT, COMPLETE, and TRANSFORM—to generate evidence traces for the INTEGRATE module.
- TADDLE leverages a multi-label expert-annotated benchmark and a two-stage semi-supervised training procedure to calibrate the INTEGRATE module for robust detection of fine-grained review deficiencies.

---

[Telenor Nordics Customer Service Self-Help Corpus](http://arxiv.org/abs/2605.26891)

- Telenor Nordics Customer Service Self-Help Corpus: introduces a multilingual dataset of 1,122 validated customer service documents sourced from Nordic telecommunications operators using a pipeline of Web Scraper, Gemma-3-27b-it, Human Annotator, Annotation UI, and Multilingual-e5-large.
- The pipeline utilizes Gemma-3-27b-it for automated pre-annotation and translation, followed by human validation via an Annotation UI to ensure high-quality, PII-free data.
- The resulting corpus supports research in retrieval-augmented generation and cross-lingual transfer learning for Nordic languages by providing real-world, domain-specific telecommunications content.

---

[A Dynamic Deontic Simplicial Logic for Joint Commitments](http://arxiv.org/abs/2605.26883)

- DSL (Deontic Simplicial Logic): introduces a geometric framework using impure simplicial complexes to model individual and joint commitments among agents.
- DDSL (Dynamic Deontic Simplicial Logic): extends the static framework with action modalities and product update mechanisms to reason about the creation, cancellation, and restructuring of group normative relationships over time.
- The research provides a sound and complete logical foundation for multi-agent deontic reasoning, utilizing combinatorial topology to capture collective commitments that are otherwise difficult to express in traditional Kripke-based frameworks.

---

[Secure UAV Swarms in Low-Altitude Wireless Networks: Challenges and Solutions](http://arxiv.org/abs/2605.26876)

- Cloud-Edge-End Collaborative Defense Framework: introduces a hierarchical architecture that integrates cloud-based global intelligence, edge-based distributed protection, and UAV-based autonomous sensing to secure UAV swarms against sophisticated threats.
- The framework employs a double-layer decision mechanism combining Bayesian games and Mean Field Games to optimize defense intensity against GPS spoofing while maintaining resource efficiency.
- It further utilizes a multi-agent LLM-based system and Datalog-based formal logic to perform proactive attack forensics, enabling intelligent tracing and mitigation of multi-hop penetration attacks.

---

[Knowledge Graphs as the Missing Data Layer for LLM-Based Industrial Asset Operations](http://arxiv.org/abs/2605.26874)

- Samyama-KG: introduces a knowledge graph data layer to improve LLM-based industrial asset operations by replacing flat document stores with structured, queryable graph representations.
- The framework utilizes an inverted LLM usage pattern where LLMs are constrained to generate structured queries from a typed schema, delegating data traversal and aggregation to deterministic graph engines.
- This approach achieves significant performance gains over baseline document-based agents by enabling complex operations like multi-hop dependency analysis, vector similarity search, and PageRank criticality.

---

[Persistent AI Agents in Academic Research: A Single-Investigator Implementation Case Study](http://arxiv.org/abs/2605.26870)

- PARE-M (Persistent Agentic Research Environment Measurement): introduces a structured implementation case study of a persistent agentic research environment, utilizing Human researcher, Agent runtime, Memory layer, Tools, Repositories, Scheduled jobs, Specialized agent roles, Governance layer, Interaction surfaces, and External APIs.
- The study quantifies the utilization, outputs, resource consumption, and governance protocols of an AI-embedded academic research workflow over 115 days.
- The findings indicate that persistent agentic environments shift the economic focus from per-token costs to artifact-level efficiency, characterized by high cache-dominance and integrated governance.

---

[REVERSE: Reinforcing Evidence Verification and Search for Agentic Image geo-localization](http://arxiv.org/abs/2605.26861)

- REVERSE: introduces a framework that reinforces the interplay between evidence search and verification to enable multi-turn agentic reasoning for image geo-localization.
- The framework utilizes Qwen3-VL-4B-Instruct (Base vision-language model) to perform iterative visual inspection, evidence retrieval, and hypothesis verification using Zoom tool (Region-level crop and magnification), Image search tool (Region-level reverse image search), and Text search tool (Web snippet retrieval).
- REVERSE employs a three-stage training pipeline including SFT, Agentic Cold Start, and Agentic RL, supervised by a composite Process reward (Multi-term reward function) that incorporates an MCC discrimination reward (Evidence selection accuracy metric) to prevent reward hacking.

---

[Helicase: Uncertainty-Guided Supply Chain Knowledge Graph Construction with Autonomous Multi-Agent LLMs](http://arxiv.org/abs/2605.26835)

- Helicase: introduces an autonomous multi-agent LLM system for uncertainty-guided supply chain knowledge graph construction, utilizing a Planner Agent, Web Search Agent, Reasoning Agent, Coding Agent, Knowledge Graph, Convergence Gate, and a Three-layer Uncertainty Framework.
- The system employs a helical process of iterative query decomposition, evidence harvesting, and structural graph mutation to synthesize fragmented information into validated, uncertainty-annotated knowledge graphs.
- Helicase incorporates a three-layer uncertainty quantification mechanism that tracks reliability at the action, trajectory, and memory levels to ensure calibrated confidence in complex, multi-hop supply chain inferences.

---

[EmoDistill: Offline Emotion Skill Distillation for Language Model Agents in Adversarial Negotiation](http://arxiv.org/abs/2605.26785)

- EmoDistill: introduces an offline framework for distilling emotional negotiation skills into LLMs by decoupling emotion selection from utterance expression.
- The framework utilizes an IQL selector to determine emotional strategy and a LoRA-adapted SLM to execute expressions, refined by JPO using dense LLM-judge feedback.
- EmoDistill enables 7B SLMs to outperform stronger baselines in adversarial negotiations by treating emotion as a strategic control channel rather than a surface style.

---

[Adversarial Training for Robust Coverage Network under Worst-case Facility Losses](http://arxiv.org/abs/2605.26763)

- DADRL (Dual-Agent Deep Reinforcement Learning): introduces a bi-level optimization framework that utilizes a Location Agent and an Interdiction Agent trained through adversarial learning to solve the Maximal Covering Location-Interdiction Problem.
- The framework employs a Surrogate-based Ensemble Inference Strategy, which uses the trained Interdiction Agent as a high-fidelity evaluator to guide the Location Agent's decisions during inference.
- By reformulating the bi-level problem into a sequential Markov Decision Process, the approach achieves superior computational efficiency and solution quality compared to traditional exact and heuristic methods.

---

[Cordon-MAS: Defending RAG against Knowledge Poisoning via Information-Flow Control](http://arxiv.org/abs/2605.26754)

- CORDON-MAS: introduces a compartmentalized multi-agent framework that enforces the Cordon Principle by separating evidence extraction, cross-source audit, and answer synthesis into agents with asymmetric memory privileges.
- The framework mitigates RAG knowledge poisoning by replacing raw document access with structured Evidence Claim Cards, ensuring that untrusted natural-language evidence cannot directly influence the final generator.
- CORDON-MAS reframes RAG poisoning as an information-flow control problem, achieving a 92.4% reduction in attack success rate compared to standard RAG systems.

---

[From Actions to Obligations: A Deontic Action Model Logic](http://arxiv.org/abs/2605.26739)

- DAML: introduces a dynamic modal framework that integrates action model logic with Bayesian-inspired deontic evaluation to reason about context-sensitive obligations in multi-agent systems.
- The framework enables agents to perform hypothetical reasoning by simulating the epistemic and normative consequences of alternative actions before selection.
- The logic provides a formal axiomatization, including soundness and completeness proofs, for deriving obligations based on the maximization of expected deontic value under epistemic uncertainty.

---

[It’s Not the Capability: Harness Sensitivity Is Non-Monotone Across LLM Agent Tiers](http://arxiv.org/abs/2605.26731)

- HEAT-24: introduces a synthetic benchmark to evaluate LLM agent reliability across varying harness complexity levels, refuting the monotone inverse hypothesis regarding model capability and structural guidance.
- The study demonstrates that harness sensitivity is non-monotone and depends on model type, where frontier chat models favor light harnesses while frontier reasoning models benefit from strict structural scaffolding.
- The research identifies that instruction-tuning quality is a more reliable predictor of harness sensitivity than parameter count, with format violations serving as the primary failure mode for capable LLMs.

---

[Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](http://arxiv.org/abs/2605.26720)

- CUDAnalyst: introduces a unified analysis layer for controlled, generation-level attribution of planning decisions to feedback components via trajectory freezing and selective feedback injection.
- The framework utilizes coalitional-style attribution to quantify the marginal contributions and interactions of heterogeneous feedback signals, including debugger-, analyzer- and profiler-agents.
- CUDAnalyst enables stable, intervention-based evaluation of self-evolving LLMs by decoupling planning decisions from historical evolutionary trajectories.

---

[Mind the Tool Failures: Achieving Synergistic Tool Gains for Medical Agents](http://arxiv.org/abs/2605.26691)

- CSRL: introduces a reinforcement learning framework that optimizes medical agents to perform instance-level tool selection by leveraging tool complementarity and disagreement-aware rewards.
- The framework utilizes Entropy-Guided Sampling to prioritize high-disagreement instances during training, providing stronger signals for learning effective tool synergy.
- CSRL consistently outperforms individual tools and baselines across multiple medical benchmarks by mitigating instance-level heterogeneity through collaborative tool integration.

---

[Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](http://arxiv.org/abs/2605.26684)

- GraphGPO: introduces a graph-based reinforcement learning method that replaces trajectory-level attribution with a unified state-transition graph to enable fine-grained step-level credit assignment.
- The framework aggregates rollout trajectories into a directed graph where nodes represent environment states and edges represent actions, allowing for the estimation of graph-based advantages based on the distance to the task goal.
- GraphGPO remains critic-free and achieves state-of-the-art performance in multi-turn agentic tasks by leveraging global structural information to reduce the variance of step-level feedback.

---

[SLA-Aware Traffic Steering in Hybrid TN-NTN 5G Backhaul: A Potential Game Approach](http://arxiv.org/abs/2605.26673)

- SLA-Aware Traffic Steering Framework: introduces a decentralized load-balancing mechanism that models per-slice traffic steering in hybrid TN-NTN backhauls as an exact potential game to ensure SLA compliance.
- The framework utilizes two independent agents, the gNB Loadbalancer and Core Loadbalancer, to dynamically distribute traffic across terrestrial and satellite paths using local telemetry without control-message overhead.
- By employing a Best Response Iteration (BRI) algorithm, the system achieves a unique Nash equilibrium that optimizes throughput, latency, and reliability while minimizing SLA violations for heterogeneous network slices.

---

[UnityMAS-O: A General RL Optimization Framework for LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2605.26646)

- UnityMAS-O: introduces a general reinforcement learning optimization framework for LLM-based multi-agent systems that treats user-defined workflows as the primary unit of optimization.
- The framework utilizes a star-topology runtime with a central controller and model-local worker groups to enable scalable, distributed training across heterogeneous LLM instances.
- UnityMAS-O decouples logical agent roles from physical model parameters, supporting flexible parameter-sharing regimes and role-specific reward attribution for complex multi-agent tasks.

---

[Adaptation-Free Heterogeneous Collaborative Perception with Unseen Agent Configurations](http://arxiv.org/abs/2605.26642)

- ALF (Adaptation-Free Late-to-Intermediate Fusion): introduces a framework that enables zero-adaptation collaborative perception by lifting compact box-level messages into ego-compatible latent features using B2BR, EFS, OCE, EIM, and ELR.
- The framework decouples communication and fusion by transmitting lightweight box-level messages, which are synthesized into fusion-ready representations within a frozen ego-stack.
- ALF effectively addresses configuration-agnostic heterogeneity in open-world settings, outperforming prior baselines on V2X-Real while maintaining a minimal communication budget.

---

[Credibility Trilemma in Polymatroidal Service Markets](http://arxiv.org/abs/2605.26604)

- Credibility Trilemma in Polymatroidal Service Markets: introduces a credibility trilemma for polymatroidal service markets, proving that no static sealed-bid mechanism can simultaneously achieve revenue optimality, agent DSIC, and operator credibility.
- The paper establishes the Cost of Non-Credibility (CoNC) as a structural welfare-loss measure and provides three resolutions: commitment via broadcast, administrative domain separation, and integrator competition.
- The research demonstrates that marketplace neutrality is a first-order design constraint, with the credibility gap arising when the marketplace operator is a strategic player with private information.

---

[Control Physiology: An Agent-Based Model of FAIR-CAM Dynamics](http://arxiv.org/abs/2605.26597)

- FAIR-CAM: introduces an agent-based model to operationalize control physiology by simulating interactions between ThreatSource, ThreatAgent, TechAsset, BusinessAsset, LEC, VMC, DSC, Personnel, RemediationQueue, and NarrativeCausationEngine.
- The framework models organizational dynamics such as budget-constrained remediation, cascading monitoring failures, and personnel-driven variance that static risk analysis cannot capture.
- The simulation provides a narrative causation engine that enables automated root cause analysis by tracing loss events through the causal chain of control degradation and monitoring failures.

---

[AGORA: Adapter-Grounded Observation-Action Retention for Inference-Free Prompt Compression in LLM Agents](http://arxiv.org/abs/2605.26596)

- AGORA (Adapter-Grounded Observation-Action Retention): introduces an inference-free, step-level prompt compression framework for LLM agents that preserves action-grammar by utilizing a Structural Parser, an Always-Keep Floor, a Relevance Scorer, and a Greedy Budget Fill.
- The framework addresses the failure mode of action-grammar destruction by operating at step-granularity rather than token-level, ensuring syntactically complete action calls are retained.
- AGORA achieves adaptive end-to-end compression ranging from 1.0–11.5× across various environments without requiring per-step LLM calls or modifications to the frozen LLM backbone.

---

[TrajAudit: Automated Failure Diagnosis for Agentic Coding Systems](http://arxiv.org/abs/2605.26563)

- TrajAudit: introduces an automated failure diagnosis framework for agentic coding systems that utilizes a prior failure reasoning module and a semantic saliency folding module to pinpoint error steps in long, noisy execution trajectories.
- The framework employs an investigator agent that dynamically probes compressed trajectory segments to maintain focus on critical failure signals while minimizing token consumption.
- The authors also present RootSE, a comprehensive benchmark comprising 93 complex software maintenance failure instances to evaluate the effectiveness of failure localization methods in agentic systems.

---

[SEC-bench Pro: Can Language Models Solve Long-Horizon Software Security Tasks?](http://arxiv.org/abs/2605.26548)

- SEC-bench Pro: introduces a benchmark and three-phase pipeline for evaluating LLM-based security agents on long-horizon software vulnerability discovery and PoC generation.
- The framework utilizes a three-image validation system (vulnerable, fixed, and latest) and an LLM-as-a-judge to ensure accurate attribution of PoC evidence to target vulnerabilities.
- Evaluation of frontier coding agents reveals that success rates remain below 40% on critical software targets, highlighting the need for improved trigger synthesis and ensemble search strategies.

---

[MobileExplorer: Accelerating On-Device Inference for Mobile GUI Agents via Online Exploration](http://arxiv.org/abs/2605.26546)

- MobileExplorer: introduces a framework that accelerates on-device inference for vision-based mobile GUI agents by performing parallel online exploration during VLM reasoning.
- The system utilizes idle VLM reasoning time to proactively probe UI elements, which are then summarized into contextual hints to enhance subsequent reasoning steps.
- A two-level rollback mechanism ensures reliable state recovery by combining depth-bounded backtracking with deterministic home-and-replay procedures.

---

[PolyFusionAgent: A Multimodal Foundation Model and Autonomous AI Assistant for Polymer Property Prediction and Inverse Design](http://arxiv.org/abs/2605.26543)

- PolyFusionAgent: introduces a unified framework that couples a multimodal foundation model with a tool-augmented agent to transform polymer informatics into an interactive, evidence-linked discovery paradigm.
- The framework utilizes PolyFusion to align four complementary polymer representations—PSMILES, 2D graphs, 3D geometry, and fingerprints—into a shared latent space for robust property prediction and property-conditioned generation.
- PolyAgent integrates a GPT-4.1 orchestrator with retrieval-augmented generation and domain-specific tools to ground design hypotheses in scientific literature and provide verifiable, constraint-consistent recommendations.

---

[ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation](http://arxiv.org/abs/2605.26542)

- ChainCaps: introduces a runtime mechanism that enforces composition safety in LLM agents by attaching sink-specific authority budgets to data and propagating them monotonically through tool chains.
- The framework utilizes an MCP proxy to intercept tool calls, ensuring that derived values cannot gain authority beyond the intersection of their contributing sources' initial budgets.
- By implementing monotonic capability attenuation, the system effectively mitigates permission laundering vulnerabilities in multi-tool workflows without requiring modifications to the underlying LLMs or tool servers.

---

[Which Changes Matter? Towards Trustworthy Legal AI via Relevance-Sensitive Evaluation and Solver-Grounded Reasoning](http://arxiv.org/abs/2605.26530)

- LexGuard: introduces a solver-grounded adversarial multi-agent framework that anchors legal decisions in explicit statutory conditions and verifiable reasoning to ensure legal-relevance sensitivity.
- The framework utilizes role-differentiated LLM agents to extract structured arguments, which are then validated against a formal legal knowledge base using an SMT solver to ensure logical consistency and statutory compliance.
- LexGuard improves legal reasoning reliability by reducing vulnerability to manipulative framing, improving disambiguation among similar statutes, and enabling auditable legal conclusions.

---

[Testing Agentic Workflows with Structural Coverage Criteria](http://arxiv.org/abs/2605.26521)

- Structural Testing Pipeline for Multi-Agent Workflows: introduces a structural adequacy model for LLM-based multi-agent systems that uses a Normalized Workflow Specification, Reachable Coordination Graph, Structural Obligations, DSPy-based Scenario Realizer, Runtime Adapter, and Witness Predicates to verify that declared coordination structures are exercised.
- The framework defines four coverage criteria—reachable agents, allowed tool edges, restricted tool edges, and delegation edges—to assess whether the internal coordination structure of an agentic workflow has been thoroughly exercised during testing.
- By utilizing a coverage-driven generation procedure, the pipeline synthesizes natural-language test scenarios that are grounded in runtime observations, providing a diagnostic layer for multi-agent systems that complements traditional end-to-end task evaluation.

---

[Uncertainty-Aware Gaussian Map for Vision-Language Navigation](http://arxiv.org/abs/2605.26503)

- Uncertainty-Aware Gaussian Map (SGM) framework: introduces a navigation agent that explicitly models geometric, semantic, and appearance uncertainties within a Semantic Gaussian Map (SGM) to improve decision-making reliability.
- The framework constructs a unified 3D Value Map by augmenting Gaussian primitives with uncertainty estimates, which are then processed by a multi-layer transformer (MLT) to predict navigation actions.
- By quantifying structural, semantic, and visual reliability, the agent effectively disambiguates complex environments and improves navigation success rates across R2R, RxR, and REVERIE benchmarks.

---

[3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](http://arxiv.org/abs/2605.26500)

- 3D Gaussian Map: introduces a unified 3D representation for Vision-Language Navigation that integrates geometric priors via Egocentric Scene Map, semantic enrichment through Open-Set Semantic Grouping, and hierarchical decision-making using Multi-Level Action Prediction.
- The framework utilizes differentiable 3D Gaussians initialized from sparse pseudo-lidar point clouds to capture fine-grained spatial structures and object-level semantics efficiently.
- By aggregating spatial-semantic cues across scene, view, and instance levels, the agent achieves improved navigation performance and robust object grounding in complex 3D environments.

---

[Aligning Provenance with Authorization: A Dual-Graph Defense for LLM Agents](http://arxiv.org/abs/2605.26497)

- AUTHGRAPH: introduces a dual-graph defense framework that structurally compares an Injected Reasoning Graph (models actual execution provenance) against an Authorization Graph (models injection-immune intent) to detect LLM agent security deviations.
- The framework utilizes a Graph Alignment Checker to perform three-layer detection—hard block, tool name check, and parameter source check—to identify unauthorized tool calls and cross-tool parameter pollution.
- By separating execution provenance from authorization specifications, AUTHGRAPH achieves fine-grained injection detection without requiring modifications to underlying LLM weights or sacrificing agent flexibility.

---

[The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence](http://arxiv.org/abs/2605.26494)

- MiniMax-M2 Series: introduces a family of Mixture-of-Experts (MoE) language models designed for agentic deployment, utilizing a 229.9B parameter backbone with only 9.8B activated parameters per token.
- The framework integrates a Multi-Token Prediction (MTP) module for speculative decoding and the Forge RL system, which decouples training and inference to support both white-box and black-box agents.
- The M2.7 checkpoint demonstrates early self-evolution capabilities, enabling the model to autonomously debug training runs and modify its own agent scaffold to improve performance on complex agentic benchmarks.

---

[Verus-SpecGym: An Agentic Environment for Evaluating Specification Autoformalization](http://arxiv.org/abs/2605.26457)

- Verus-SpecGym: introduces an agentic environment for evaluating specification autoformalization, utilizing Verus-SpecBench, Verus verifier, exec_spec_unverified, Harbor, SWE-AGENT, Codeforces test suite, and Adversarial hacks.
- The framework enables scalable, execution-based evaluation of formal specifications by extending Verus's mechanism to compile logical predicates into executable Rust code for testing against official and adversarial test cases.
- Experimental results demonstrate that specification autoformalization remains a significant bottleneck for LLMs, with failures often occurring even when models successfully generate correct code.

---

[Constitutional Arms Races in the Public Goods Game: Co-Evolving LLM Constitutions Under Cooperation–Defection Pressure](http://arxiv.org/abs/2605.26448)

- Adversarial Constitutional Co-evolution: introduces a framework for co-evolving interpretable natural-language constitutions between two opposing factions to study adversarial dynamics in multi-agent settings.
- The framework utilizes an LLM-mutation operator and MAP-Elites to iteratively update constitutions, demonstrating that fitness coupling is essential for generating genuine adversarial pressure.
- The research identifies evaluation seed count as a critical hyperparameter for stabilizing LLM-guided evolutionary search and provides evolved constitutions as interpretable red-team artifacts for testing future LLM designs.

---

[When Does Deep RL Beat Calibrated Baselines? A Benchmark Study on Adaptive Resource Control](http://arxiv.org/abs/2605.26418)

- RLSCALE-BENCH: introduces a reproducible benchmark and evaluation protocol for deep reinforcement learning on adaptive resource control, comprising Environment &amp; Workloads, Matched RL Agents, Calibrated HPA Baseline, Training Protocol, Transfer Evaluation, and Three Findings.
- The framework evaluates six DRL algorithms against a properly calibrated rule-based baseline across diverse workload patterns to address inconsistencies in current resource management research.
- The study demonstrates that while calibrated rule-based controllers often achieve lower costs, DRL agents—specifically discrete-action variants—can provide superior SLO compliance on unpredictable, bursty traffic.

---

[From Static Context to Calibrated Interactive RL: Mitigating Distribution Shift in Multi-turn Dialogue with Aligned Simulator](http://arxiv.org/abs/2605.26403)

- Calibrated Interactive RL: introduces a unified framework that couples interactive RL with simulator alignment to mitigate policy- and simulator-induced distribution shifts in multi-turn dialogue.
- The framework utilizes a two-stage process consisting of Simulator Calibration (SFT) to ground the user simulator in human reality and Interactive Policy Optimization (GRPO) to train the policy agent on self-generated trajectories.
- By aligning the simulator with human interaction patterns, the approach effectively bridges the sim-to-real gap and prevents reward hacking, yielding state-of-the-art performance in collaborative editing and conversational reasoning tasks.

---

[Got a Secret? LLM Agents Can’t Keep It: Evaluating Privacy in Multi-Agent Systems](http://arxiv.org/abs/2605.27766)

- Moltbook-style simulation platform: introduces a multi-agent environment to evaluate privacy leakage as a downstream safety concern under varying degrees of social pressure, utilizing Agent Persona, Human Profile, Memory, Tool Suite, Asynchronous Agent Interaction Loop, Leakage Detection Pipeline, and Adversarial Contamination.
- The framework demonstrates that LLMs exhibit a social ratchet effect where exposure to peer disclosure in multi-turn interactions significantly increases the probability of sensitive information leakage.
- The research highlights that static, single-turn safety benchmarks systematically underestimate privacy risks in persistent, socially embedded agentic deployments.

---

[PEAM: Parametric Embodied Agent Memory through Contrastive Internalization of Experience in Minecraft](http://arxiv.org/abs/2605.27762)

- PEAM: introduces a two-tier embodied agent framework that transforms episodic memory into parameter-resident skills using a slow deliberative LLM, an episodic store, and a fast parametric module with per-category isolated LoRA adapters.
- The framework utilizes a joint behavioral-cloning and contrastive objective to internalize failure-correction trajectory pairs, ensuring the agent learns from both successes and failures.
- PEAM employs a parameterization-worthiness score and a scale-free self-triggered consolidation mechanism to automate the internalization of experience without requiring task-specific hand-tuned thresholds.

---

[AndroidDaily: A Verifiable Benchmark for Mobile GUI Agents on Real-World Closed-Source Applications](http://arxiv.org/abs/2605.27761)

- GRADE: introduces a process-aware evaluation framework for closed-source mobile applications by utilizing a three-tiered system of observable guidelines to verify agent trajectories without internal state access.
- The framework employs an Evidence Layer to extract structured task-relevant signals from visual trajectories and a Verdict Layer to perform diagnostic checks against operational obligations, output quality, and negative constraints.
- AndroidDaily provides a large-scale benchmark of 350 realistic tasks across 94 closed-source applications, revealing that frontier LLMs struggle with latency-induced misalignment and memory-induced action loops.

---

[SkillGrad: Optimizing Agent Skills Like Gradient Descent](http://arxiv.org/abs/2605.27760)

- SkillGrad: introduces a gradient-descent-inspired framework for optimizing LLM agent skills by treating structured skill packages as optimizable parameters.
- The framework utilizes an iterative loop where an executor generates trajectory-level loss evidence, a diagnoser produces textual gradients, and a momentum agent stabilizes updates via a persistent memory overlay.
- Empirical results on spreadsheet manipulation tasks demonstrate that SkillGrad consistently outperforms training-based skill evolution baselines by systematically refining procedural knowledge.

---

[A Policy-Driven Runtime Layer for Agentic LLM Serving](http://arxiv.org/abs/2605.27744)

- Agent Runtime Layer: introduces an architectural tier between the agent framework and serving engine that exposes four primitives (observe, score, predict, act) to manage cross-cutting policies using agent identity as a shared coordinate.
- The framework utilizes CacheSage to implement these primitives for KV caching, employing a transition learner to model agent behavior and a survival-probability scorer to optimize cache eviction.
- Experimental results demonstrate that this runtime layer significantly improves cache hit rates, reduces latency, and increases throughput for multi-agent LLM workloads by leveraging predictable agent transition patterns.

---

[UserHarness: Harnessing User Minds for Stronger Agent Theory-of-Mind](http://arxiv.org/abs/2605.27721)

- UserHarness: introduces an inference-time scaffold that reframes Theory-of-Mind evaluation as explicit user-mind reconstruction by separating the true environment from the user's subjective perspective.
- The framework formalizes user reasoning as a perception–belief–action loop, utilizing an Environment, Observation, Belief, Goal, Action, Nested Belief, Rule-guided Operator, and Trace to ensure consistent mental state tracking.
- By constraining LLMs to reason through this structured symbolic scaffold, UserHarness significantly improves Theory-of-Mind accuracy and reliability across diverse benchmarks while reducing dependence on unconstrained generation.

---

[LLM Based Web Accessibility Repair: An Empirical Study of Detection, Remediation, and Cost](http://arxiv.org/abs/2605.27716)

- Dual-pipeline accessibility repair framework: introduces a hybrid system combining deterministic rule-based detection with LLM-based agentic remediation to address web accessibility violations.
- The framework utilizes a Kimi K2.5 LLM agent for contextual reasoning and iterative repair, validated by a multi-layer mechanism that checks syntactic validity, structural integrity, and WCAG compliance.
- Empirical results demonstrate that while LLMs are effective for localized semantic repairs, iterative agent-based refinement significantly increases computational costs without improving overall remediation success compared to zero-shot approaches.

---

[Chain-based Adaptive Reconfiguration Over Lattices for Hallucination Reduction](http://arxiv.org/abs/2605.27706)

- CAROL (Chain-based Adaptive Reconfiguration Over Lattices): introduces a probabilistic framework for test-time hallucination reduction in LLMs by casting generation as a Markov chain accept-reject process over a lattice of textual sequences.
- The framework utilizes a string-submodular objective to evaluate semantic consistency between generated responses and a trusted axiom set, enabling iterative refinement without modifying the underlying LLM.
- CAROL integrates within an agentic pipeline that includes planning-, researcher- and reasoner-agents to achieve near-optimal hallucination mitigation with provable convergence guarantees.

---

[AgenticVBench: Can AI Agents Complete Real-World Post-Production Tasks?](http://arxiv.org/abs/2605.27705)

- AgenticVBench: introduces a 100-task benchmark evaluating AI agents on four real-world video post-production families using programmatic verifiers and expert rubrics.
- The benchmark assesses LLMs across Assembly, Repair, Sequencing, and Repurpose tasks, revealing that current frontier agent systems perform significantly below human expert levels.
- Experimental results demonstrate that harness design is a critical factor influencing agent performance, tool-use patterns, and failure modes across diverse video production workflows.

---

[Hierarchical Prompt-Domain Control and Learning for Resource-Constrained Agentic Language Models](http://arxiv.org/abs/2605.27703)

- Hierarchical prompt-domain control and adaptive learning framework: introduces a closed-loop architecture that separates offline schema distillation from online semantic adaptation to maintain compact LLMs within feasible prompt domains.
- The framework utilizes an oracle-controller loop to monitor protocol validity and semantic performance, triggering lightweight oracle-supervised fine-tuning when prompt-domain drift is detected.
- A feasibility-aware projection mechanism maps growing interaction histories into a bounded prompt domain, mitigating attention-induced saturation and ensuring structural compatibility with agentic systems.

---

[TRACES: Proactive Safety Auditing for Multi-Turn LLM Agents via Trajectory-State Modeling](http://arxiv.org/abs/2605.27690)

- TRACES: introduces a representation-based proactive auditor that models trajectory risk as an evolving state read from the hidden representations of an observer LLM.
- The framework utilizes a Representation Mechanism Bank to extract interpretable latent evidence patterns and a temporal auditor to estimate prefix-level risk without requiring dense step-level annotations.
- TRACES improves proactive risk discrimination and full-trajectory safety prediction across multi-turn agent benchmarks by identifying emerging unsafe behaviors before they manifest as final outcomes.

---

[Decoupled Intelligence: A Multi-Agent LLM Framework for Controllable Traffic Scenario Generation in SUMO](http://arxiv.org/abs/2605.27685)

- Decoupled Multi-Agent Collaborative Framework: introduces a modular architecture that decomposes complex traffic simulation workflows into specialized roles—including Planner-, Builder-, Modifier-, Demand-, Runner- and Analyst-agents—to mitigate reasoning drift and parameter inconsistency in LLMs.
- The framework utilizes a state-persistent Master Orchestrator powered by the Model Context Protocol (MCP) to ensure deterministic artifact handover and eliminate path-referencing hallucinations across distributed agent actions.
- A closed-loop feedback mechanism enables the Analyst agent to provide performance-based verbal reinforcement to the Planner, facilitating autonomous iterative optimization of traffic scenarios.

---

[Agentic Language-to-Objective Synthesis for Optofluidic Assembly](http://arxiv.org/abs/2605.27643)

- Speak-to-Objective: introduces an agentic pipeline that utilizes an LLM to translate natural-language commands into differentiable objective functions for microscale particle assembly.
- The framework employs a closed-loop cycle of perceive, compose, propose, act, and report &amp; learn to refine objective synthesis through user feedback and a few-shot knowledge base.
- By decoupling design intent from physical actuation, the system enables actuator-agnostic, interpretable, and self-healing microassembly via laser-induced thermoviscous flows.

---

[Poison with Style: A Practical Poisoning Attack on Code Large Language Models](http://arxiv.org/abs/2605.27631)

- PwS (Poison-with-Style): introduces a stealthy model poisoning attack that leverages code styles as covert triggers to induce vulnerable code generation in CLLMs without requiring explicit prompt manipulation.
- The framework utilizes a four-phase process—Data Collection, Data Poisoning, Model Poisoning, and Deployment—to train CLLMs to generate vulnerable code when specific trigger code styles are detected in input prompts.
- Experimental results demonstrate that PwS achieves high attack success rates across diverse vulnerabilities while maintaining model utility and remaining robust against state-of-the-art defense mechanisms.

---

[Intelligence as Managed Autonomy: Failure, Escalation, and Governance for Agentic AI Systems](http://arxiv.org/abs/2605.27628)

- SMARt (Self-Managing Multi-tier Autonomous Reasoning with Regulated/Revoked transitions): introduces a state-based framework that replaces unbounded autonomy with a formal lifecycle of Stable Autonomous Reasoning (S), Meta-cognitive Local Recovery (M), Assisted Mutual Recovery (A), and Regulated/Revoked Transition to External Control (Rt).
- The framework utilizes Timed Guarded Petri Nets to enforce state transitions based on domain-specific trigger sets, ensuring that LLMs cannot persist in ungrounded or unsafe states.
- By treating autonomy as a revocable privilege rather than a continuous default, SMARt provides a mathematically verifiable mechanism to bound operational risk and mandate escalation to human governance.

---

[Reasoning and Planning with Dynamically Changing Norms](http://arxiv.org/abs/2605.27622)

- SocialBot: introduces a theoretically grounded approach for guiding AI agent planning using dynamically changing norms represented as guard rails within a cognitive architecture.
- The framework utilizes a defeasible calculus to resolve normative conflicts in user testimony, ensuring consistent behavior through formal logical inference.
- Empirical evaluation on a synthetic dataset of 1,536 dialogues demonstrates that the agent accurately updates and respects evolving privacy norms during task execution.

---

[Agents that Matter: Optimizing Multi-Agent LLMs via Removal-Based Attribution](http://arxiv.org/abs/2605.27621)

- MAS Attribution Framework: introduces a unified, protocol-conditioned cooperative game approach to quantify individual agent contributions in multi-agent LLM systems by parameterizing attribution queries with Removal Protocol, Coalition Distribution, and Target Metric.
- The framework demonstrates that Leave-One-Out (LOO) effectively identifies bottleneck agents with significantly lower computational cost than combinatorial methods like Shapley, Owen, or Myerson values.
- The authors introduce model replacement as a topology-preserving protocol, enabling cost-effective interventions by substituting low-contribution agents with cheaper backbones while maintaining or improving system performance.

---

[Laguna M.1/XS.2 Technical Report](http://arxiv.org/abs/2605.27605)

- Model Factory: introduces an industrial-scale methodology for foundation model development, integrating versioned data, training, evaluation, and inference components to accelerate research iteration.
- The framework utilizes Titan for distributed training, Hive for synthetic data generation, and AutoMixer for automated data mixture optimization to build agentic coding models.
- The system incorporates a custom workload scheduler and containerized code execution environments to support long-horizon agentic tasks and reliable reinforcement learning.

---

[The Energy Blind Spot: NVIDIA’s Flagship Edge AI Hardware Cannot Support Process-Level Energy Attribution](http://arxiv.org/abs/2605.27599)

- GB10 Hardware Audit Framework: introduces a systematic evaluation of energy observability on NVIDIA’s GB10 edge AI hardware, revealing a critical lack of process-level energy attribution due to disabled firmware interfaces.
- The paper demonstrates that while the GB10 SoC contains internal power-sensing capabilities via the PMIC and SPBM, these are not exposed to userspace, preventing accurate energy accounting for agentic AI workloads.
- The authors propose an interim calibration bridge using external DC metering and NVML, while advocating for a standards-track firmware update to expose SCMI powercap protocols for research-grade energy observability.

---

[Voluntary Collusion in Competing LLM Agents with Secret Tools](http://arxiv.org/abs/2605.27593)

- Voluntary Collusion in Competing LLM Agents with Secret Tools: introduces an empirical framework to investigate whether LLMs voluntarily adopt unfair, secret, and harmful tools for strategic advantage across Liar’s Bar and Cleanup environments.
- The study demonstrates that most LLMs consistently accept these tools and develop collusive strategies, even when explicitly acknowledging the unfairness and harm imposed on other agents.
- The research highlights that preventing such collusive behavior requires explicit safeguards, as general safety alignment is often insufficient to deter agents when strategic incentives are present.

---

[You Only Align Once: Propagating Cooperative Behaviors in Multi-Agent Systems through Seed Agents](http://arxiv.org/abs/2605.27586)

- YOAO: introduces a framework for propagating cooperative behaviors in multi-agent systems by strategically placing SFT-trained seed agents that influence untrained agents through natural language interaction.
- The pipeline utilizes a teacher model to generate high-quality cooperative reasoning trajectories, which are then distilled into a seed agent via LoRA SFT to instill robust deliberative skills.
- Evaluations demonstrate that these seed agents effectively propagate cooperation across diverse environments and model architectures, challenging the necessity of exhaustive per-agent training for multi-agent alignment.

---

[Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation](http://arxiv.org/abs/2605.27582)

- Uni-LaViRA: introduces a unified agentic architecture that decomposes embodied navigation into a Language-Vision-Robot Actions Translation task, leveraging pretrained MLLMs for zero-shot reasoning across diverse task families and robot embodiments.
- The framework utilizes a Language Action Model for high-level planning, a Vision Action Model for pixel-level grounding, and a deterministic Robot Action Controller for low-level execution.
- Two agent-loop mechanisms, TODO List Memory and Second Chance Backtrack, enable the agent to maintain long-horizon sub-goal progress and perform self-correcting re-planning after navigation errors.

---

[Discovery Agents for Real-Time Analytics: Toward Proactive Insight Systems](http://arxiv.org/abs/2605.27571)

- Discovery Agents for Real-Time Analytics: introduces a multi-agent architecture that automates the analytics lifecycle by transforming real-time data streams into actionable insights through a continuous discovery loop.
- The system utilizes specialized LLM-based agents to generate hypotheses, compile executable Python and FlinkSQL artifacts, validate outputs, and package them into deployable applications.
- A contract-driven design using typed intermediate artifacts ensures modularity, observability, and lineage tracking across the entire autonomous discovery process.

---

[Why LLMs Fail at Causal Discovery and How Interventional Agents Escape](http://arxiv.org/abs/2605.27567)

- A-CBO (Agentic Causal Bayesian Optimization): introduces a framework that overcomes the geometric kernel obstruction of standard LLM training by delegating causal hypothesis discrimination to an external Bayesian loop using local interventional queries.
- The framework utilizes a frozen LLM as a binary interventional oracle, ensuring that discrete causal decisions are made outside the kernel predictor's representation space.
- A-CBO provably converges to the correct causal graph in logarithmically many rounds, maintaining performance as graph complexity scales where fine-tuned LLMs collapse.

---

[DynaSchedBench: Calibrated Dynamic Scheduling Benchmarks and Observability Paradox in LLM-based Scheduling Agents](http://arxiv.org/abs/2605.27566)

- DynaSchedBench: introduces a diagnostic framework for Dynamic Flexible Job Shop Scheduling that rigorously controls instance generation through a Sequential Event-Space Calibrator and Schedule Stress Index.
- The framework integrates modular components for simulation, evaluation, and visualization to enable controlled stress-testing of LLM-based scheduling agents across varying observability levels.
- The research identifies an "Observability Paradox" where providing LLMs with full structural information degrades performance compared to concise statistical summaries, characterizing them as robust heuristic approximators.

---

[Detection Without Correction: A Two-Parameter Decomposition of Multi-Stage LLM Pipelines](http://arxiv.org/abs/2605.27559)

- Detection-Generation Decomposition: introduces a two-parameter framework that decomposes multi-stage LLM pipeline responses into detection and conditional generation decisions to explain aggregate performance phenomena.
- The framework identifies the DM (detect-miscorrect) regime as the primary load-bearing failure mode where verification triggers but fails to produce a correct alternative.
- Empirical analysis across 14 cohorts demonstrates that while detection rates are contextual, conditional miscorrection remains consistently dominant, explaining observed accuracy plateaus and reversals in multi-stage LLM pipelines.

---

[From Task Allocation to Risk Clearing: A Unifying Interface for Mixed Human-Agent Societies](http://arxiv.org/abs/2605.27547)

- ROC (Risk-Aware Option Clearing): introduces a unifying coordination mechanism for mixed human-agent societies that utilizes Option Providers, ROC Clearinghouse, Risk-Aware Optimizer, Calibration & Reputation, Assignment, and Real-World Execution Environment to manage task allocation under uncertainty.
- The framework enables heterogeneous agents to expose temporally extended skills paired with probabilistic risk summaries, allowing a central clearinghouse to optimize mission utility while respecting safety and deadline constraints.
- ROC functions as a scalable infrastructure that decouples agent-level execution from system-level coordination, supporting diverse deployment tiers ranging from learned empirical models to full distributional predictions.

---

[SCALE-COMM: Shared, Contrastively-Aligned Latent Embeddings for MARL Communication](http://arxiv.org/abs/2605.27532)

- SCALE-COMM (Shared, Contrastively-Aligned Latent Embeddings for COMMunication): introduces a self-supervised framework that decouples communication learning from policy optimization by training compact, stable, and policy-relevant latent messages.
- The architecture utilizes cross-agent contrastive alignment, prototype-based distillation, and temporal predictive consistency to ensure semantically grounded and reusable communication tokens.
- A soft curriculum scheduling mechanism balances self-supervised representation learning with reinforcement learning to prevent early representational collapse and ensure grounded task utility.

---

[Agentic Separation Logic Specification Synthesis](http://arxiv.org/abs/2605.27531)

- Spec-Agent: introduces an agentic system for synthesizing expressive, well-validated specifications across large C++ codebases by combining counterexample-guided contract inference with automatically generated fuzz harnesses.
- The framework utilizes an adaptive specification-language selector to reason about propositional logic, first-order logic, propositional separation logic, and first-order separation logic within a single pipeline.
- Spec-Agent employs fuzz testing as a pseudo-oracle to systematically validate candidate contracts, significantly outperforming existing LLM-based approaches in accuracy and cost-efficiency.

---

[Benchmarks are Not Enough: RAMP for Runtime Assessing of Agentic Models in Production Systems](http://arxiv.org/abs/2605.27492)

- RAMP (Runtime Assessment of Models in Production): introduces a production-grounded infrastructure for evaluating LLM agents using serial compiler-construction workloads and a resurrection mechanism to isolate failure propagation.
- The framework utilizes an orchestrator to manage task dependencies and injects golden artifacts to enable continued assessment of downstream performance despite upstream failures.
- RAMP incorporates multi-dimensional metrics, including the Agent Efficiency Index (AEI), to jointly evaluate task completion, process efficiency, and resource utilization across heterogeneous LLM agents.

---

[HARP: Measuring Harm Amplification in Multi-Agent LLM Systems](http://arxiv.org/abs/2605.27489)

- HARP (Harm Amplification through Role Perturbation): introduces a trace-first methodology for quantifying how bounded local perturbations in multi-agent LLM systems are amplified into system-level harm through orchestration.
- The framework evaluates multi-agent systems by comparing paired clean and perturbed execution traces, recording specialist outputs, tool calls, memory reads/writes, and decision gate outcomes to measure harm amplification.
- HARP identifies four vulnerability regimes—single-specialist compromise, multi-agent collusion, shared-context failure, and temporal/persistent failure—and evaluates defense effectiveness using metrics like attack success rate, harm amplification, stealth, and benign utility.

---

[Grimlock: Guarding High-Agency Systems with eBPF and Attested Channels](http://arxiv.org/abs/2605.27488)

- Grimlock: introduces a substrate-level security architecture for high-agency agents by enforcing mandatory network mediation and channel-bound authorization using Sandbox CVM, Proxy CVM, eBPF hooks, kTLS, Grimlock A2A Protocol, and Scope Token.
- The framework utilizes eBPF hooks to transparently redirect all agent traffic through a Proxy CVM, ensuring no-bypass enforcement at the sandbox boundary.
- It leverages kTLS for efficient kernel-space encryption and implements post-handshake attestation to mint channel-bound Scope Tokens for secure, least-privilege agent-to-agent communication.

---

[Automating Formal Verification with Agent-Guided Tree Search](http://arxiv.org/abs/2605.27485)

- Automating Formal Verification with Agent-Guided Tree Search: introduces an agentic framework for Lean theorem proving that utilizes a Parent Agent, Subagent, Lean-LSP-MCP Server, Search Tool, Submit Tool, State-based Orchestrator, Context-based Orchestrator, and Proof-state Tree to improve verified-code generation.
- The framework leverages an agentic loop with mathlib search to significantly improve LLM performance on formal verification tasks, scaling effectively with increased LLM call budgets.
- The research demonstrates that context-based tree search outperforms state-based approaches by preserving subagent transcripts, enabling more effective exploration of complex proof strategies.

---

[Detect by Yourself: Self-Designing Agentic Workflows for Few-Shot Graph Anomaly Detection](http://arxiv.org/abs/2605.27470)

- SignGAD: introduces a paradigm that reformulates graph anomaly detection from training fixed detectors to designing task-conditioned workflows using Task Agent, Evidence Agent, Detector Agent, Workflow Detector Bank, Validation Workflow Search, and Guarded Final Refit.
- The framework utilizes LLMs to construct task-specific detection workflows by grounding planning in textual descriptions and graph statistics, enabling adaptive anomaly detection under limited supervision.
- SignGAD enhances detection reliability and efficiency by explicitly organizing contextual anomaly evidence and employing a guarded refit strategy to refine detector performance.

---

[AgensFlow: A Coordination-Policy Substrate for Multi-Agent Systems](http://arxiv.org/abs/2605.27466)

- AgensFlow: introduces a coordination-policy substrate that treats multi-agent orchestration as an online policy-learning problem under partial observability, utilizing a Policy Graph, Router, and RelativeJudge to optimize task-specific workflows.
- The framework replaces static pipelines with a learnable, auditable routing policy that dynamically selects skill protocols, model bindings, and coordination topologies based on folded task signatures.
- AgensFlow incorporates a RelativeJudge for reward-signal robustness and supports cross-domain transfer by warm-starting the policy graph with learned coordination priors from structurally similar tasks.

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





