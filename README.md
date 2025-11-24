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

## Research papers: 2025 (1/3)

[2025 (1/3)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2025 (2/3)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (3/3)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)

Chronological order. 





</div>



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

#### 17th Nov 2025

[KForge: Program Synthesis for Diverse AI Hardware Accelerators](http://arxiv.org/abs/2511.13274)

- KForge: introduces an agentic program synthesis framework that iteratively refines programs using a Generation Agent and a Performance Analysis Agent, interpreting diverse profiling data to guide optimization for arbitrary accelerators.
- The framework supports single-shot generation and iterative refinement, leveraging cross-platform knowledge transfer from reference implementations to improve generation quality across different hardware targets like NVIDIA CUDA and Apple Metal.
- Key components include two collaborative LLM-based agents that simulate a practical kernel engineering workflow, focusing on functional correctness before performance optimization.

---

[DualTAP: A Dual-Task Adversarial Protector for Mobile MLLM Agents](http://arxiv.org/abs/2511.13248)

- DualTAP (Dual-Task Adversarial Protector): introduces a novel framework that explicitly decouples privacy protection and task utility objectives for mobile Multimodal Large Language Model (MLLM) agents by training a perturbation generator guided by a contrastive attention module and a dual-task adversarial objective.
- The framework utilizes a contrastive attention module to precisely locate PII-sensitive regions and optimizes the generator to minimize task-preservation loss ($L_n$) while maximizing privacy-interference loss ($L_p$).
- DualTAP achieves state-of-the-art privacy protection by significantly reducing leakage rates while maintaining high task success rates across diverse MLLMs, resolving the privacy-utility trade-off.

---

[MEGA-GUI: Multi-stage Enhanced Grounding Agents for GUI Elements](http://arxiv.org/abs/2511.13087)

- MEGA-GUI: introduces a modular, multi-stage framework that decomposes GUI grounding into coarse Region-of-Interest (ROI) selection and fine-grained element grounding by orchestrating specialized agents based on diverse Vision-Language Models (VLMs).
- The framework centers on a bidirectional ROI zoom algorithm for robust search and error recovery, complemented by a context-aware rewriting agent to resolve semantic ambiguity in user instructions.
- This modular, agentic architecture achieves state-of-the-art performance by leveraging the complementary strengths of different VLMs for distinct sub-tasks.

---

[LIVE-SWE-AGENT: Can Software Engineering Agents Self-Evolve on the Fly?](http://arxiv.org/abs/2511.13646)

- LIVE-SWE-AGENT: introduces the first live software agent that autonomously and continuously evolves its own scaffold implementation on-the-fly during runtime when solving real-world software problems, starting from a minimal bash-only scaffold (mini-SWE-agent).
- The agent achieves state-of-the-art open-source performance on SWE-bench Verified (75.4%) and SWE-Bench Pro (45.8%) by iteratively synthesizing and using custom tools based on a reflection mechanism.
- This on-the-fly self-evolution approach requires no costly offline training and demonstrates generalizability across different LLMs and benchmarks.

---

[An Operational Kardashev-Style Scale for Autonomous AI - Towards AGI and Superintelligence](http://arxiv.org/abs/2511.13411)

- AAI Scale: introduces a Kardashev-inspired, multi-axis, and testable Autonomous AI (AAI) Scale to measure progression from fixed automation (AAI-0) to Superintelligence (AAI-5), utilizing an AAI-Index, a Self-Improvement Coefficient $\kappa$, and closure properties.
- The framework defines ten capability axes (e.g., Autonomy, Generality, Planning, Tool Economy) normalized to [0,1] and aggregated via a weighted geometric mean (AAI-Index).
- It formalizes AGI and Superintelligence through measurable level gates (AAI-0 to AAI-4/5) based on axis thresholds, sustained self-improvement ($\kappa$), and closure proofs (maintenance and expansion).

---

[CorrectAD: A Self-Correcting Agentic System to Improve End-to-end Planning in Autonomous Driving](http://arxiv.org/abs/2511.13297)

- CorrectAD: introduces a self-correcting agentic system, composed of PM-Agent and DriveSora, to automatically generate targeted training data to improve the robustness of End-to-end (E2E) planning models in autonomous driving by addressing failure cases.
- The PM-Agent analyzes failure causes using a VLM to formulate multimodal requirements, which DriveSora then uses to generate high-fidelity, diverse training videos aligned with 3D scene annotations.
- This agentic pipeline is model-agnostic and demonstrated significant reduction in collision rates on both public and in-house datasets.

---

[LLM-based Multi-Agent System for Simulating Strategic and Goal-Oriented Data Marketplaces](http://arxiv.org/abs/2511.13233)

- LLM-MAS (Large Language Model-based Multi-Agent System): introduces a simulation framework for data marketplaces where LLM-powered buyer and seller agents perform strategic, goal-oriented actions using natural language reasoning.
- The system utilizes a GoalGenerator for objectives and a DataGenerator for metadata, storing embeddings in a Vector Database to enable similarity-based search for agent actions.
- Evaluation against real transaction data shows the LLM-MAS faithfully reproduces structural features like scale-free distributions, though temporal dynamics are overestimated.

---

[Agent-Oriented Visual Programming for the Web of Things](http://arxiv.org/abs/2511.13158)

- AOV-DEP (Agent-Oriented Visual Programming for Domain-Expert Programming): introduces an approach for multi-agent-oriented visual programming using a blocks-based visual development environment built on the JaCaMo platform and integrated with the Web of Things (WoT) to enable domain experts to design and configure autonomous software.
- The system leverages agent abstractions, specifically the Belief-Desire-Intention (BDI) model, to align with human practical reasoning for simpler programming by non-technical users.
- The implementation uses the Blockly framework for the visual language and Yggdrasil for WoT integration, validated by a pilot user study showing promising usability.

---

[Resilient and Efficient Allocation for Large-Scale Autonomous Fleets via Decentralized Coordination](http://arxiv.org/abs/2511.12879)

- DESIRA (Decentralized Side-Information Resource Allocation): introduces a framework combining side-information-conditioned risk shaping with scalable consensus-based coordination, using Distributional Predictions and a CVaR Penalty, coordinated via Consensus-ADMM.
- The approach models uncertain resource consumption using feature-conditioned distributional predictions to derive risk-adjusted allocation requirements, ensuring safety guarantees via chance constraints.
- The decentralized coordination is achieved through local message passing over a sparse communication graph, leading to near-centralized performance with high resilience and near-linear scaling.

---

[LoCoBench-Agent: An Interactive Benchmark for LLM Agents in Long-Context Software Engineering](http://arxiv.org/abs/2511.13998)

- LoCoBench-Agent: introduces a comprehensive evaluation framework for LLM agents in long-context software engineering, extending LoCoBench scenarios into interactive environments with specialized tools and bias-free metrics.
- The framework focuses on multi-turn interaction, tool usage patterns, and long-context handling (10K-1M tokens) across 8,000 scenarios spanning 10 programming languages and 36 domains.
- Key findings reveal a fundamental comprehension-efficiency trade-off and highlight the importance of architectural mechanisms like hierarchical memory and semantic search integration for long-context performance.

---

[EchoAgent: Guideline-Centric Reasoning Agent for Echocardiography Measurement and Interpretation](http://arxiv.org/abs/2511.13948)

- EchoAgent: introduces a guideline-centric agentic framework that integrates specialized vision tools under Large Language Model (LLM) orchestration to perform structured, interpretable echocardiography measurement and interpretation.
- The framework utilizes an iterative reasoning loop involving observation, thought, and action phases, leveraging tools for phase detection, measurement feasibility prediction, segmentation, and guideline retrieval.
- A key feature is the measurement-feasibility prediction model, which ensures that only visually supported and clinically relevant measurements are attempted, enhancing trustworthiness.

---

[Market-Dependent Communication in Multi-Agent Alpha Generation](http://arxiv.org/abs/2511.13614)

- Market-Dependent Communication in Multi-Agent Alpha Generation: investigates the impact of five organizational structures on 5-agent LLM-based trading systems across different market characteristics, comparing isolated baseline, leaderboard, collaborative conversation, conversation-leaderboard, and competitive conversation.
- Communication generally improves performance, but the optimal structure depends on market volatility, with competitive conversation excelling in volatile tech stocks and collaborative conversation in stable general stocks.
- All organizational structures converge to similar strategy correlations over time, indicating that behavioral mechanisms, not information sharing transparency, drive performance differences.

---

[P1: Mastering Physics Olympiads with Reinforcement Learning](http://arxiv.org/abs/2511.13612)

- P1: introduces a family of open-source physics reasoning models trained via reinforcement learning (RL) and augmented with the PhysicsMinions agentic framework, achieving Gold-medal performance on the International Physics Olympiad 2025 (IPhO 2025).
- The training incorporates a multi-stage RL framework with adaptive learnability adjustment and stabilization mechanisms, utilizing both rule-based and model-based verifiers for reward generation.
- The framework demonstrates strong generalizability to mathematics and coding tasks, suggesting transferable reasoning skills beyond the specialized physics domain.

---

[FreeAskWorld: An Interactive and Closed-Loop Simulator for Human-Centric Embodied AI](http://arxiv.org/abs/2511.13524)

- FreeAskWorld: introduces an interactive and closed-loop simulation framework that integrates LLMs for high-level behavior planning and semantically grounded interaction, grounded in theories of intention and social cognition, to support human-centric embodied AI.
- The framework supports scalable, realistic human-agent simulations and includes a modular data generation pipeline, releasing a large-scale benchmark dataset for the novel Direction Inquiry Task.
- The system leverages LLMs for intention modeling and naturalistic human behavior simulation within photorealistic 3D environments, emphasizing interaction as an information modality.

---

[Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction](http://arxiv.org/abs/2511.13410)

- Mem-PAL: introduces PAL-Bench (Personalization benchmark) and PAL-Set (Chinese dataset) for long-term user-agent interaction evaluation, utilizing H²Memory (Hierarchical memory framework) with MG (Concrete memory from logs), MB (Abstract memory of user background), MT (Concrete memory from dialogues), and Mp (Abstract memory of user principles) via RAG (Generation strategy) to enhance personalized response generation.
- The H²Memory framework organizes interaction history into a hierarchical and heterogeneous structure, separating concrete details (logs and dialogue outlines) from abstract concepts (background and principles) for effective retrieval and personalized response generation.
- The proposed method demonstrates superior performance across three evaluation tasks in PAL-Bench: Requirement Restatement, Solution Proposal, and Multi-turn Dialogue Interaction, validating the effectiveness of the memory components.

---

[MedDCR: Learning to Design Agentic Workflows for Medical Coding](http://arxiv.org/abs/2511.13361)

- MedDCR (Medical Coding Workflow Design as a Learning Problem): introduces a closed-loop framework that treats medical coding workflow design as a learning problem, utilizing a Designer, Coder, and Reflector meta-agent architecture supported by a memory archive.
- The framework iteratively proposes, compiles, executes, and reflects on workflow plans, leveraging past successful designs and diverse recent explorations to discover effective coding strategies.
- MedDCR achieves state-of-the-art performance on ICD-10 coding benchmarks while producing interpretable and adaptable workflows.

---

[SAINT: Service-level Integration Test Generation with Program Analysis and LLM-based Agents](http://arxiv.org/abs/2511.13305)

- SAINT (Service-level Integration Test Generation with Program Analysis and LLM-based Agents): introduces a novel white-box testing approach for service-level testing of enterprise Java applications by combining Static Analysis, LLM-based Agents, an Endpoint Model, and an Operation Dependency Graph (ODG) to automatically generate endpoint-focused and scenario-based tests.
- The approach involves a Model-Construction Phase to build the Endpoint Model and ODG, followed by a Test-Generation Phase utilizing agentic workflows for test creation and refinement.
- Endpoint-focused tests maximize code coverage, while scenario-based tests cover meaningful use cases, with developer feedback strongly endorsing the latter.

---

[Grounded by Experience: Generative Healthcare Prediction Augmented with Hierarchical Agentic Retrieval](http://arxiv.org/abs/2511.13293)

- GHAR (Generative Hierarchical Agentic Retrieval): introduces a generative hierarchical agentic RAG framework for healthcare prediction that resolves the "when to retrieve" dilemma and enables collaborative optimization between retrieval and generation submodules.
- The framework utilizes a dual-agent architecture (Agent-Top and Agent-Low) within a unified Markov Decision Process optimized via multi-agent Reinforcement Learning to ensure synergistic retrieval and generation.
- GHAR employs meta-path partitioning for fine-grained retrieval and a diverse reward structure to align the distinct roles of the agents towards accurate, contextually appropriate predictions.

---

[Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment](http://arxiv.org/abs/2511.13290)

- Dropouts in Confidence (DIC): introduces a method to quantify and modulate uncertainty in LLMs facing moral dilemmas using information-theoretic measures like binary entropy, Total Entropy (TE), Conditional Entropy (CE), and Mutual Information (MI), and demonstrates that injecting uncertainty via attention dropout improves alignment with human preferences.
- The study analyzes 32 open-source LLMs across 9 moral dimensions derived from the Moral Machine experiment, finding significant model-architecture-dependent confidence variability.
- The core finding is that reducing LLM overconfidence by increasing Mutual Information (MI) through inference-time dropout leads to better alignment with human ethical judgments in complex scenarios.

---

[Cost-Effective Communication: An Auction-based Method for Language Agent Interaction](http://arxiv.org/abs/2511.13193)

- DALA (Dynamic Auction-based Language Agent): introduces a novel framework that treats communication bandwidth as a scarce, tradable resource in Multi-Agent Systems (MAS) using a centralized auction mechanism, where agents bid based on predicted message value density, trained via MAPPO.
- The framework utilizes an Actor Network to generate candidate messages and a Critic Network to compute their value density ($\pi$), which serves as a bid in a budget-constrained VCG auction to maximize task success while minimizing token cost.
- This economic approach cultivates the emergent skill of strategic silence, leading to state-of-the-art performance on reasoning benchmarks with significantly reduced token consumption compared to existing methods.

---

[Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](http://arxiv.org/abs/2511.13118)

- AEC (Agent-Event-Coder): introduces a novel multi-agent framework that treats zero-shot event extraction (ZSEE) as a structured, iterative code-generation process, utilizing a Retrieval Agent/Planning Agent/Coding Agent/Verification Agent workflow.
- The framework represents event schemas as executable Python classes to enable deterministic validation and enforce structural fidelity in zero-shot extractions.
- AEC consistently outperforms prior zero-shot baselines by combining step-wise reasoning with deterministic schema validation to resolve trigger ambiguity and enforce output structure.

---

[WebCoach: Self-Evolving Web Agents with Cross-Session Memory Guidance](http://arxiv.org/abs/2511.12997)

- WebCoach: introduces a model-agnostic self-evolving framework that equips web browsing agents with persistent, cross-session memory, enabling improved long-term planning, reflection, and continual learning without retraining.
- The framework consists of a WebCondenser, an External Memory Store (EMS) for storing semantic embeddings of past trajectories, and a Coach LLM that provides task-specific guidance via runtime hooks.
- Evaluations show that WebCoach consistently improves task success rates across different LLM backbones, achieving performance comparable to GPT-4o with smaller open-source models.

---

[ENGRAM: EFFECTIVE, LIGHTWEIGHT MEMORY ORCHESTRATION FOR CONVERSATIONAL AGENTS](http://arxiv.org/abs/2511.12960)

- ENGRAM (Effective, Lightweight Memory Orchestration): introduces a lightweight memory system that organizes conversation into episodic, semantic, and procedural memory types using a single router and retriever, achieving state-of-the-art results on long-horizon QA benchmarks.
- The architecture converts user turns into typed memory records persisted in a database, retrieves top-k neighbors per type at query time, merges results, and provides evidence as context to the answering LLM.
- This typed separation and straightforward dense retrieval approach challenges the trend toward complex memory architectures by prioritizing simplicity, efficiency, and interpretability.

---

[Can We Predict the Next Question? A Collaborative Filtering Approach to Modeling User Behavior](http://arxiv.org/abs/2511.12949)

- CFQP (Collaborative Filtering-enhanced Question Prediction): introduces a novel hybrid framework that integrates personalized memory modules with graph-based preference propagation to dynamically model evolving user-question interactions for superior user-specific question prediction.
- The framework utilizes an Embedding-based User Representation via BGE to create user vectors, calculates user similarity to form a User Association graph, and employs an LLM-based Prediction Model refined by a Diagnostic Collaborative Correction loop.
- This approach aims to overcome the limitations of static LLM personalization by capturing dynamic user interests and leveraging collective intelligence from similar users.

---

[Fault2Flow: An AlphaEvolve-Optimized Human-in-the-Loop Multi-Agent System for Fault-to-Workflow Automation](http://arxiv.org/abs/2511.12916)

- Fault2Flow: introduces an LLM-based multi-agent system that automates fault diagnosis to workflow execution by systematically extracting regulatory logic, integrating expert knowledge, optimizing reasoning, and synthesizing an executable workflow, utilizing an AlphaEvolve optimization module.
- The system operates via a decoupled front-end/back-end design, where the back-end employs coordinated agents to transform unstructured regulatory documents into verified, n8n-executable workflows.
- Experimental validation on transformer fault diagnosis confirms 100% topological consistency and high semantic fidelity, substantially reducing expert workload.

---

[Think, Speak, Decide: Language-Augmented Multi-Agent Policy Learning in Economic Environments](http://arxiv.org/abs/2511.12876)

- LAMP (Language-Augmented Multi-Agent Policy Learning): introduces a framework that integrates LLM-driven reasoning and reflection over numerical observations and textual signals to support optimal decision-making in multi-agent economic environments, following a Think-Speak-Decide pipeline.
- The framework utilizes a dual-path Think module to generate short-term shock analysis and long-term trend reasoning, which informs the Speak module for strategic message exchange and belief updating via a Reflection Module.
- The Decide module fuses numerical data, reasoning, and reflections into a centralized training/decentralized execution Multi-Agent Reinforcement Learning (MARL) policy, achieving superior performance over MARL and LLM-only baselines.

---

[HPCAgentTester: A Multi-Agent LLM Approach for Enhanced HPC Unit Test Generation](http://arxiv.org/abs/2511.10860)

- HPCAgentTester: introduces a novel multi-agent Large Language Model (LLM) framework for automating and enhancing unit test generation for HPC software using OpenMP and MPI, employing specialized LLM agents in a collaborative workflow.
- The framework utilizes a structured Test Recipe as an intermediate representation, grounding the Test Agent's code generation and enabling iterative refinement via a critique loop involving feedback, confidence scoring, and justification.
- This approach significantly improves test compilation rates and functional correctness compared to standalone LLMs by systematically targeting parallel constructs and semantic correctness.

---

#### 16th Nov 2025

[MMWOZ: Building Multimodal Agent for Task-oriented Dialogue](http://arxiv.org/abs/2511.12586)

- MATE (Multimodal Agent for Task-oriented dialogue): introduces MMWOZ, a multimodal task-oriented dialogue dataset interacting with a web-style GUI, and proposes MATE, a baseline multimodal model leveraging dialogue history, action log, and web page snapshot (text and image features) to generate GUI operation instructions or natural language responses.
- The MMWOZ dataset extends MultiWOZ 2.3 by designing a web-style GUI and automatically converting dialogue states and system actions into operation instructions paired with web page snapshots.
- The MATE model architecture includes an OCR Parser and Image Encoder to process the snapshot, which feed into a Projector and Action Generator conditioned on dialogue history and action log to determine the next step.

---

[Multi-Agent Reinforcement Learning for Heterogeneous Satellite Cluster Resources Optimization](http://arxiv.org/abs/2511.12792)

- MARL (Multi-Agent Reinforcement Learning): introduces a framework for resource optimization in heterogeneous satellite clusters performing Earth Observation (EO) missions, utilizing algorithms like MAPPO, HAPPO, and HATRPO within a CTDE paradigm.
- The study models the EO mission as a Dec-POMDP to handle decentralized decision-making under resource constraints and agent heterogeneity (optical and SAR satellites).
- The research evaluates the performance and stability of state-of-the-art MARL algorithms specifically tailored to account for agent heterogeneity in satellite resource allocation.

---

[Are LLMs The Way Forward? A Case Study on LLM-Guided Reinforcement Learning for Decentralized Autonomous Driving](http://arxiv.org/abs/2511.12751)

- Framework name here: introduces a case study comparing RL-only, LLM-only, and hybrid approaches, where LLMs augment RL rewards by scoring state-action transitions during training, while standard RL policies execute at test time.
- The study uses small, locally deployable LLMs (Qwen3-14B and Gemma3-12B) to investigate their ability to support autonomous highway driving through reward shaping rather than direct control.
- Findings indicate that hybrid approaches improve safety over RL-only agents but introduce a systematic conservative bias, highlighting limitations of current small LLMs for safety-critical control.

---

[Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs](http://arxiv.org/abs/2511.12710)

- EvoSynth: introduces an autonomous framework that shifts the red-teaming paradigm from attack planning to the evolutionary synthesis of novel, code-based jailbreak methods, employing a multi-agent system with a code-level self-correction loop.
- The framework utilizes a Reconnaissance Agent for strategy formulation, an Algorithm Creation Agent for code synthesis and evolution, an Exploitation Agent for deployment, and a Coordinator Agent for orchestration and iterative refinement.
- This approach achieves a new state-of-the-art Attack Success Rate (ASR) against robust models and generates attacks with significantly higher programmatic complexity and diversity than existing methods.

---

[On two-degrees-of-freedom agreement protocols](http://arxiv.org/abs/2511.12632)

- 2DOF agreement protocol: introduces a distributed two-degrees-of-freedom (2DOF) architecture for driving heterogeneous agents to agreement, separating local feedback from network filtering.
- This architecture is inspired by classical servo regulation and aims to counter shortcomings of consensus protocols like poor noise attenuation and inability to reject disturbances exciting unstable poles.
- The resulting closed-loop dynamics explicitly separate network and local dynamics, accommodating agent heterogeneity when the network component is homogeneous.

---

[Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments](http://arxiv.org/abs/2511.13788)

- The research introduces an empirical framework for exploring multi-model interactions using JailbreakBench, involving an Attacker Model (Ma), a Target Model (My), and a Judge Model (MJ), to quantify how relative model scale influences adversarial potency.
- The study simulates over 6000 multi-turn exchanges across various LLM sizes (0.6B-120B) to measure harm score and refusal behavior as indicators of adversarial success and alignment integrity.
- Key findings show a positive correlation between the attacker-to-target size ratio and mean harm, and a strong negative correlation between attacker refusal frequency and harm.

---

[Knots: A Large-Scale Multi-Agent Enhanced Expert-Annotated Dataset and LLM Prompt Optimization for NOTAM Semantic Parsing](http://arxiv.org/abs/2511.12630)

- NOTAM semantic parsing: introduces a novel task extending beyond traditional information extraction by generating structured, inference-rich outputs from Notices to Air Missions (NOTAMs), supported by the Knots dataset and utilizing LLM Prompt Optimization, MDA, and HDF components.
- The framework employs a two-stage multi-agent system (MDA for recall, HDF for precision) to systematically discover and refine operational fields, addressing semantic ambiguity and complexity inherent in aviation texts.
- The research validates various LLM prompting strategies, finding 5-shot In-Context Learning (ICL) optimal for safety-critical reliability, and provides a large, expert-annotated dataset (Knots) for future research.

---

[FINRS: A RISK-SENSITIVE TRADING FRAMEWORK FOR REAL FINANCIAL MARKETS](http://arxiv.org/abs/2511.12599)

- FinRS (Risk-Sensitive Trading Framework): introduces a risk-sensitive LLM trading framework that combines hierarchical market analysis, dual-decision agents, and multi-timescale reward reflection to align trading actions with return objectives and downside risk constraints, utilizing components like the Market Perception and Analysis Module, Risk-Sensitive Decision Making Module, and Multi-scale Reward Reflection Module.
- The framework addresses limitations in existing LLM trading agents by embedding risk-awareness directly into the decision process, featuring dynamic position sizing and layered information filtering.
- Experimental results confirm that the full configuration of FinRS achieves superior profitability and stability compared to various baselines across multiple stocks and market conditions.

---

[Co-Layout: LLM-driven Co-optimization for Interior Layout](http://arxiv.org/abs/2511.12474)

- Co-Layout: introduces a novel framework that combines Large Language Models (LLMs) with grid-based Integer Programming (IP) to jointly optimize room layout and furniture placement, using a Coarse-to-Fine Strategy and a Grid-Based Formulation.
- The LLM-based Preprocessor translates textual requirements into structured design constraints, which are then formalized using a grid-based representation inspired by "Modulor" for the IP model.
- The framework employs a Coarse-to-Fine Strategy to manage computational complexity by first solving a simplified problem on a coarse grid before refining the solution on the full-resolution grid.

---

[The 'Sure' Trap: Multi-Scale Poisoning Analysis of Stealthy Compliance-Only Backdoors in Fine-Tuned Large Language Models](http://arxiv.org/abs/2511.12414)

- Sure Trap: introduces a compliance-only backdoor during SFT where a single benign compliance token ("Sure") acts as a latent behavioral gate to enable unsafe generation when paired with an arbitrary trigger token.
- The attack relies on poisoning a small subset of prompts with a trigger and the single-token response "Sure," achieving near-deterministic compliance rates above a small poison budget threshold (~50 examples).
- This mechanism exposes a stealthy data-supply-chain risk and suggests using the gate-like dynamics for explicit, auditable control tokens in agentic systems.

---

#### 15th Nov 2025

[Learning to Trust: Bayesian Adaptation to Varying Suggester Reliability in Sequential Decision Making](http://arxiv.org/abs/2511.12378)

- The proposed framework introduces a unified POMDP-based approach that dynamically learns and adapts to varying suggester reliability in partially observable environments by integrating suggester quality into the belief state and introducing an explicit 'ask' action.
- The framework utilizes a MOMDP formulation to manage computational complexity when modeling the hidden state component of suggester types ($\mathcal{T}$).
- Experimental results across Tag and RockSample domains demonstrate robust performance, adaptation to changing reliability, and strategic management of suggestion requests.

---

[Fast Reasoning Segmentation for Images and Videos](http://arxiv.org/abs/2511.12368)

- FastReasonSeg: introduces a distillation framework that reduces computational demands for reasoning segmentation by transferring knowledge from a Teacher LLM to a compact Student LLM using structured Digital Twin Representations.
- The framework decouples perception from reasoning via the Digital Twin Representation, enabling the Student LLM to perform complex analysis without processing raw visual tokens.
- The two-stage distillation process involves SFT followed by RL with a composite reward function to preserve multi-step reasoning capabilities.

---

[Goal-Oriented Multi-Agent Reinforcement Learning for Decentralized Agent Teams](http://arxiv.org/abs/2511.11992)

- Goal-Oriented Multi-Agent Reinforcement Learning (MARL) framework: introduces a decentralized MARL approach for agent teams in dynamic, partially observable environments, enabling selective, goal-aware communication and coordination.
- The method utilizes weight merging to share learning parameters among agents pursuing the same individual goal, enhancing collaboration while maintaining decentralization.
- Experimental validation in complex grid navigation tasks shows improved success rates and reduced time-to-goal compared to non-cooperative and unrestricted communication baselines, demonstrating scalability.

---

[Decision and Gender Biases in Large Language Models: A Behavioral-Economic Perspective](http://arxiv.org/abs/2511.12319)

- LLMs: introduces an investigation into whether advanced LLMs behave as rational agents or reproduce human behavioral tendencies in classic decision problems (Ultimatum Game and Gambling Game) using behavioral economics parameters, involving LLMs (Gemma-7B and Qwen-2.5-32B-Instruct-AWQ) under neutral and gender-conditioned prompts.
- The study estimates parameters for inequity aversion and loss aversion, comparing LLM results to human benchmarks, finding persistent deviations from rationality, including moderate fairness concerns and subtle gender-conditioned differences.
- The methodology employs canonical behavioral-economic tasks to elicit parameters related to fairness and risk preferences, providing a behavioral-economics perspective on LLM decision-making.

---

[UpBench: A Dynamically Evolving Real-World Labor-Market Agentic Benchmark Framework Built for Human-Centric AI](http://arxiv.org/abs/2511.12306)

- UpBench: introduces a dynamically evolving, real-world labor-market agentic benchmark framework, utilizing real jobs, human-generated rubrics, and expert freelancer evaluation, to assess LLM agents' competence and collaboration capacity.
- The framework integrates human expertise across data collection, rubric creation, and evaluation stages, supporting fine-grained analysis beyond binary pass/fail metrics.
- It provides a scalable foundation for evaluating agentic systems in authentic contexts, emphasizing human-AI collaboration over simple automation.

---

[ProofWright: Towards Agentic Formal Verification of CUDA](http://arxiv.org/abs/2511.12294)

- ProofWright: introduces an agentic verification framework that integrates automated formal verification with LLM-based code generation to provide end-to-end guarantees of memory safety, thread safety, and semantic correctness for LLM-generated CUDA kernels, utilizing components like the VerCors Agent and Semantic Equivalence Framework.
- The framework employs the VerCors Agent to establish safety properties using the VerCors verifier guided by an LLM-generated Annotation Guide, and the Semantic Equivalence Framework to prove functional adherence using the Rocq Theorem Prover.
- It addresses the validation bottleneck in LLM-generated GPU code by automating formal reasoning, achieving safety guarantees for 74% of KernelBench L1 programs with modest overhead.

---

[MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](http://arxiv.org/abs/2511.12271)

- MoralReason: introduces a reasoning-level reinforcement learning approach using MoralReason-QA and GRPO on Qwen-3-4B-Base to achieve out-of-distribution moral decision alignment in LLM agents.
- The approach utilizes a multi-component reward function combining alignment and keyword rewards to facilitate learning of underlying moral frameworks.
- Experimental results demonstrate successful generalization to unseen moral scenarios for Utilitarian and Deontological frameworks.

---

[RulePilot: An LLM-Powered Agent for Security Rule Generation](http://arxiv.org/abs/2511.12224)

- RulePilot: introduces an LLM-powered agent workflow utilizing Chain of Thought (CoT) reasoning, Intermediate Representation (IR), and Reflection & Iterative Optimization to automate the generation and conversion of SIEM-specific detection rules, abstracting complexity for security analysts.
- The framework uses an IR to structure complex SIEM rule logic, enabling LLMs to focus on manageable generation steps, and employs iterative reflection with tool invocation for robust refinement.
- Evaluation shows RulePilot significantly outperforms standalone LLMs in textual similarity and achieves high execution success rates in detecting simulated attacks in a Splunk environment.

---

[CriticSearch: Fine-Grained Credit Assignment for Search Agents via a Retrospective Critic](http://arxiv.org/abs/2511.12159)

- CriticSearch: introduces a fine-grained credit-assignment framework that leverages a frozen, asymmetric Critique LLM to provide dense, turn-level feedback via a retrospective mechanism for search agents trained with reinforcement learning.
- The framework uses privileged information, specifically the gold answer and full trajectory, to enable the Critique LLM to assign stable, dense rewards that guide policy improvement, complementing sparse outcome rewards.
- This retrospective assessment approach, integrated with the GRPO algorithm, consistently outperforms existing baselines by achieving faster convergence and improved training stability across multi-hop reasoning benchmarks.

---

[AI-Salesman: Towards Reliable Large Language Model Driven Telemarketing](http://arxiv.org/abs/2511.12133)

- AI-Salesman: introduces an end-to-end framework that addresses challenges in goal-driven persuasive dialogue like telemarketing, utilizing a dual-stage architecture with Bayesian-supervised reinforcement learning and a Dynamic Outline-Guided Agent (DOGA).
- The framework is supported by the newly released TeleSalesCorpus, the first real-world-grounded dialogue dataset for this domain, and a comprehensive LLM-as-a-Judge evaluation framework.
- Experimental results show that the proposed method significantly outperforms baselines across key sales capabilities, validating its effectiveness in complex persuasive scenarios.

---

#### 14th Nov 2025

[Chapter 14: Looking Forward: Challenges and Opportunities in Agentic AI Reliability](http://arxiv.org/abs/2511.11921)

- Chapter 14: Looking Forward: Challenges and Opportunities in Agentic AI Reliability: presents perspectives on challenges and future development in building reliable agentic AI systems, discussing open research problems related to mitigating cascading failures, dynamic environments, inconsistent execution, emergent behaviors, resource-intensive mechanisms, and evaluation.
- The chapter organizes reliability challenges into five main areas: Cascading Failures, Vulnerability in Dynamic Environments, Inconsistency in Task Execution, Unpredictable Emergent Behavior, and Resource-Intensive Reliability Mechanisms, alongside the need for new Reliability Testing and Evaluation paradigms.
- Addressing these challenges requires cross-layer coordination, dynamic adaptation, integrated reasoning, and resource-aware reliability designs to ensure trustworthy, consistent, and safe outputs from agentic AI systems.

---

[MULTI-PHASE SPACECRAFT TRAJECTORY OPTIMIZATION VIA TRANSFORMER-BASED REINFORCEMENT LEARNING](http://arxiv.org/abs/2511.11402)

- Transformer-based RL framework: introduces a unified control framework leveraging Gated Transformer-XL and PPO to handle multi-phase spacecraft trajectory optimization using a single adaptive policy, with components including an Observation Encoder, Actor-Critic Model, Policy Head, and Value Head.
- The architecture utilizes the GTrXL's sliding memory window and self-attention mechanisms to maintain coherent memory across dynamically distinct mission phases without explicit phase transitions.
- The framework is validated on single-phase benchmarks, multi-phase waypoint navigation, and a complex multiphase rocket ascent problem, demonstrating near-optimal performance compared to traditional methods.

---

[Robust and Efficient Communication in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2511.11393)

- Survey: introduces a systematic review of recent advances in robust and efficient communication strategies for MARL under realistic constraints, including message perturbations, transmission delays, and limited bandwidth, focusing on applications like cooperative autonomous driving, distributed SLAM, and federated learning.
- The review organizes communication strategies along three key dimensions: when to transmit, whom/how to communicate, and what/rate to transmit, highlighting a shift from idealized assumptions to practical, imperfect environments.
- The paper advocates for a unified approach that co-designs communication, learning, and robustness to bridge the gap between theoretical MARL models and practical implementations.

---

[Building the Web for Agents: A Declarative Framework for Agent-Web Interaction](http://arxiv.org/abs/2511.11287)

- VOIX: introduces a concrete, web-native mechanism that makes site capabilities and state discoverable and invokable by agents through declarative, typed semantics, using `<tool>` and `<context>` HTML elements.
- The framework decouples website functionality from agent reasoning, distributing responsibilities among the Website, the Browser Agent, and the Inference Provider.
- Empirical evaluation via a hackathon confirmed the framework's learnability, expressiveness for multimodal interactions, and efficiency compared to inference-based approaches.

---

[GraphPilot: Grounded Scene Graph Conditioning for Language-Based Autonomous Driving](http://arxiv.org/abs/2511.11266)

- GraphPilot: introduces a model-agnostic method that conditions language-based driving models on structured relational context in the form of traffic scene graphs, using an LLM-based AD Agent, Scene-Graph, Navigation Command, and Future Trajectory.
- The approach serializes traffic scene graphs at various abstraction levels and incorporates them via structured prompt templates to enhance structured reasoning over spatial, regulatory, and inter-actor dependencies.
- Training with scene graph supervision (SG10) yields performance gains that persist even when scene graphs are omitted at test-time, indicating internalized relational knowledge.

---

[UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios](http://arxiv.org/abs/2511.11252)

- UAVBench: introduces an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation, with UAVBench_MCQ extending it for reasoning evaluation.
- The framework unifies scenario generation, validation, risk labeling, and reasoning into a single pipeline, encoding missions in a structured JSON schema covering configuration, environment, objectives, and safety constraints.
- UAVBench_MCQ evaluates LLMs across ten cognitive and ethical reasoning styles, revealing strong performance in perception but persistent challenges in ethics-aware and resource-constrained decision-making.

---

[Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA](http://arxiv.org/abs/2511.11169)

- AlignVQA: introduces a debate-based multi-agent framework, AlignVQA, which uses Specialized Agents and Generalist Agents with an AlignCal Loss to improve confidence calibration in Visual Question Answering (VQA).
- The framework involves a two-stage interaction where specialized agents provide initial answers, followed by generalist agents engaging in debate to critique, refine, and aggregate proposals, yielding calibrated confidence estimates.
- The novel AlignCal loss is a differentiable surrogate for the Upper Bound on Classification Error (UBCE), explicitly optimizing specialized agents for confidence fidelity during training.

---

[Autonomous Vehicle Path Planning by Searching With Differentiable Simulation](http://arxiv.org/abs/2511.11043)

- DSS (Differentiable Simulation for Search): introduces a framework that leverages the differentiable simulator Waymax as both a next state predictor and a critic, optimizing actions via gradient descent over imagined future trajectories.
- The approach uses Classifier-Guided Action Selection to incorporate non-differentiable events like collisions into the differentiable planning loss function.
- The framework achieves improved tracking and path planning accuracy compared to sequence prediction, imitation learning, and model-free RL methods by combining search and gradient-based refinement.

---

[Miniature Testbed for Validating Multi-Agent Cooperative Autonomous Driving](http://arxiv.org/abs/2511.11022)

- CIVAT (Cooperative Intelligent V2X Autonomous Testbed): introduces a 1:15-scale miniature testbed for validating cooperative autonomous driving, integrating miniature vehicles equipped with onboard sensors and smart infrastructure supported by 3D LiDAR and edge computing, with components including CAV/Infrastructure/Perception/Planning/Control/Message Generator/LiDAR/Depth Camera/IMU/MCU/SBC (Jetson Orin NX)/Custom PCB/V2V Communication/V2I Communication.
- The infrastructure acts as an active agent, performing infrastructure-centric 3D object detection and Human Vehicle (HV) identification to coordinate Connected Autonomous Vehicles (CAVs) using priority-based intersection management.
- The platform supports both fully CAV and mixed-traffic scenarios, demonstrating real-time applicability for cooperative driving algorithms via V2I and V2V communication using a Wi-Fi-based ROS2 publish-subscribe framework.

---

[InData: Towards Secure Multi-Step, Tool-Based Data Analysis](http://arxiv.org/abs/2511.11933)

- INDATA (Indirect Data Engagement): introduces a security-motivated alternative for LLM-based data analysis by restricting LLMs to interact with data exclusively through a predefined set of secure, verified tools, and presents the INDATA dataset to evaluate multi-step tool-based reasoning ability.
- The framework uses Predefined Tools as a secure barrier between the LLM and Sensitive Data, contrasting with direct code generation approaches that pose security risks.
- The INDATA dataset specifically targets complex, compositional, multi-step reasoning, revealing a capability gap in current LLMs compared to simple tool selection tasks.

---

[An Analysis of Architectural Impact on LLM-based Abstract Visual Reasoning: A Systematic Benchmark on RAVEN-FAIR](http://arxiv.org/abs/2511.11916)

- RAVEN-FAIR: introduces a systematic evaluation of Large Language Models (LLMs) performance on abstract visual reasoning tasks using four reasoning architectures, a three-stage process (JSON extraction, LLM reasoning, Tool Function), and visual/textual metrics.
- The study benchmarks four LLMs (GPT-4.1-Mini, Claude-3.5-Haiku, Gemini-1.5-Flash, Llama-3.3-70b) across four reasoning configurations to analyze decision-making quality, error tolerance, and consistency.
- Results indicate that architectural selection is critical, performance is model-specific, and trade-offs exist between semantic grounding and quantitative precision across strategies.

---

[Conformal Policy Optimization for Cost-Effective LLM Agents](http://arxiv.org/abs/2511.11828)

- CCPO (Conformal Constrained Policy Optimization): introduces a framework for training an orchestration policy to select between multiple LLM agents to minimize cost while satisfying a user-specified reliability constraint formalized via conformal prediction, using components like a Base LLM, Guide LLM, Orchestration Policy, Conformal Prediction, and V-trace.
- The framework formalizes the deployment problem as a finite-horizon Partially Observable Markov Decision Process (POMDP) where the policy $\pi$ is parameterized stochastically and a threshold $\kappa$ is updated online to ensure coverage guarantees.
- Empirical results show that CCPO reduces total computational and API costs by up to 30% compared to state-of-the-art cost-aware baselines while maintaining target reliability on HotpotQA and MMLU benchmarks.

---

[From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions](http://arxiv.org/abs/2511.11789)

- The paper introduces a systematic investigation of persona-induced biases in LLM-based multi-agent interactions, utilizing LLM-based Multi-Agent Systems with Persona Assignment and a Default Agent across Collaborative Problem Solving (CPS) Task and Persuasion Task.
- The study quantifies biases in trustworthiness and insistence, finding that personas from historically advantaged groups are perceived as less trustworthy and insistent, and reveals in-group favoritism in agent conformity.
- These behavioral patterns persist across different LLMs, group sizes, and interaction rounds, highlighting the need for bias mitigation in autonomous agent environments.

---

[MALBO: Multi-Agent LLM Bayesian Optimization](http://arxiv.org/abs/2511.11788)

- MALBO (Multi-Agent LLM Bayesian Optimization): introduces a systematic framework designed to automate the efficient composition of LLM-based agent teams by formalizing the assignment challenge as a multi-objective, black-box optimization problem, using Multi-Objective Bayesian Optimization (MOBO) with Gaussian Process surrogate models and the qEHVI acquisition function.
- The methodology employs a continuous relaxation of the discrete LLM assignment space, projecting ideal continuous solutions back to real, deployable LLM assignments via a nearest-neighbor projection function.
- Results show that the framework achieves a 45.64% reduction in mean cost while maintaining comparable performance compared to initial random search, and identifies heterogeneous teams with up to 65.8% cost reduction over homogeneous baselines.

---

[Experience-Guided Adaptation of Inference-Time Reasoning Strategies](http://arxiv.org/abs/2511.11519)

- EGUR (Experience-Guided Reasoner): introduces a system that dynamically generates tailored strategies—complete computational procedures involving LLM calls, tools, sampling parameters, and control logic—at inference time based on accumulated experience, utilizing a Guide and a Consolidator.
- The system formalizes strategies as compositions of stateful processes, enabling adaptation of all strategy components, unlike prior methods limited to textual steering.
- EGUR achieves up to 14% accuracy improvements and up to 111x reduction in computational costs across challenging benchmarks by learning from comparative strategy evaluation.

---

[MarsRL: Advancing Multi-Agent Reasoning System via Reinforcement Learning with Agentic Pipeline Parallelism](http://arxiv.org/abs/2511.11373)

- MarsRL (Multi-Agent Reasoning System via Reinforcement Learning with Agentic Pipeline Parallelism): introduces a novel reinforcement learning framework to jointly optimize Solver, Verifier, and Corrector agents in a multi-agent reasoning system, addressing reward noise and training efficiency challenges.
- The framework employs agent-specific rewards to decouple credit assignment and utilizes pipeline parallelism to accelerate the training process for long reasoning trajectories.
- Experimental results show significant performance gains on AIME2025 and BeyondAIME benchmarks when applying the framework to Qwen3-30B-A3B-Thinking-2507.

---

[SRLF: An Agent-Driven Set-Wise Reflective Learning Framework for Sequential Recommendation](http://arxiv.org/abs/2511.11370)

- SRLF (Set-wise Reflective Learning Framework): introduces a closed-loop "assess-validate-reflect" cycle using LLM agents to move beyond point-wise assessment by formulating a holistic judgment over sets of items, utilizing components like the Set-wise Assessment Agent (SAA), Validation via Set-wise Mismatch Loss, and Dual-Path Reflective Learning.
- The framework captures complex contextual patterns by analyzing intra-set item relationships and their alignment with the user's preference profile, which is crucial for sequential recommendation tasks.
- The reflective mechanism concurrently refines the user profile and item semantics to adapt to dynamic user interests and improve representation learning.

---

[LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](http://arxiv.org/abs/2511.11334)

- LaoBench: introduces the first large-scale, multidimensional benchmark dataset dedicated to assessing LLMs' comprehensive language understanding and reasoning abilities in Lao, covering Knowledge Application, K12 Foundational Education, and Bilingual Translation, utilizing a pipeline integrating expert human curation and agent-assisted verification.
- The benchmark comprises over 17,000 curated samples split into open-source (Lao-7k, Lao-500) and closed-source (Lao-10k) subsets to ensure fairness and transparency in evaluation.
- Evaluation results show that current state-of-the-art LLMs face significant challenges in mastering Lao, highlighting the need for targeted research in this low-resource Southeast Asian language.

---

[UFO³: Weaving the Digital Agent Galaxy](http://arxiv.org/abs/2511.11332)

- UFO³: Weaving the Digital Agent Galaxy, introduces a cross-device orchestration system that unifies heterogeneous endpoints into a single fabric using a mutable TASKCONSTELLATION (distributed DAG of TASKSTARS), a CONSTELLATIONAGENT (LLM-driven planner), a Constellation Orchestrator (asynchronous execution engine), and the AIP (communication protocol).
- The system models user requests as a TASKCONSTELLATION, a dynamic DAG where nodes (TASKSTARS) are atomic subtasks with dependencies (TASKSTARLINES) that evolve based on runtime feedback.
- It addresses challenges in cross-device agent workflows by providing asynchronous parallelism, distributed coordination, and heterogeneous extensibility across devices like Windows, Linux, and mobile.

---

[iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](http://arxiv.org/abs/2511.11306)

- iMAD (Intelligent Multi-Agent Debate): introduces a token-efficient framework that selectively triggers Multi-Agent Debate (MAD) only when beneficial, utilizing a structured self-critique prompt and a Debate-Decision Classifier trained with FocusCal Loss, to enhance LLM inference efficiency and accuracy.
- The framework addresses the high computational cost and inconsistent accuracy gains of standard MAD by learning generalizable model behaviors to identify recoverable errors via 41 interpretable linguistic and semantic features extracted from a single-agent's self-critique response.
- Experiments show iMAD significantly reduces token usage (up to 92%) while improving final answer accuracy (up to 13.5%) across various QA and VQA datasets compared to single-agent and full-debate baselines.

---

[Multi-agent Undercover Gaming: Hallucination Removal via Counterfactual Test for Multimodal Reasoning](http://arxiv.org/abs/2511.11182)

- MUG (Multi-agent Undercover Gaming): introduces a protocol inspired by social deduction games to address LLM hallucinations in multimodal reasoning by employing multimodal counterfactual tests to detect "undercover" agents (those hallucinating) using components like the Counterfactual Editing Module, Undercover Detection Game, and Summarization Game.
- The framework dynamically modifies reference images to create counterfactual evidence (I-) to enable direct factual verification, moving beyond the statistical consensus reliance of traditional Multi-Agent Debate (MAD) protocols.
- MUG fosters active reasoning where agents engage in probing discussions based on information asymmetry between normal agents (seeing I+) and the undercover agent (seeing I-).

---

[Scaling Equitable Reflection Assessment in Education via Large Language Models and Role-Based Feedback Agents](http://arxiv.org/abs/2511.11772)

- Equitable Reflection Assessment Pipeline: introduces a theory-grounded system using five coordinated role-based LLM agents (Evaluator, Equity Monitor, Metacognitive Coach, Aggregator, and Reflexion Reviewer) to score learner reflections with a shared rubric and generate short, bias-aware, learner-facing comments.
- The multi-agent LLM system aims to deliver equitable, high-quality formative feedback at scale by integrating structured agent roles, fairness checks, and learning-science principles.
- The pipeline produces auditable rubric scores and bias-aware, conversational feedback, addressing the challenge of providing timely, high-quality feedback in large or low-resource courses.

---

[Key Decision-Makers in Multi-Agent Debates: Who Holds the Power?](http://arxiv.org/abs/2511.11040)

- MADC (Multi-Agent Debate Consistency): introduces a novel role allocation strategy for Multi-Agent Debate (MAD) frameworks by leveraging path consistency metrics to dynamically order agents, aiming to improve reasoning performance across various LLMs and tasks.
- The research identifies role allocation strategy as a critical, underexplored scaling dimension in MAD, showing that placing agents with correct viewpoints last ("Truth Last") significantly boosts accuracy.
- The proposed MADC method is orthogonal to existing MAD frameworks, optimizing role arrangement without modifying agent prompts or context to unlock potential performance gains.

---

[GraphMASAL: A Graph-based Multi-Agent System for Adaptive Learning](http://arxiv.org/abs/2511.11035)

- GraphMASAL (Graph-based Multi-Agent System for Adaptive Learning): introduces an integrated, graph-based multi-agent system for adaptive learning that addresses challenges in knowledge dynamism, execution complexity, optimization, and validation, utilizing a Dynamic Knowledge Graph, a trio of specialized agents orchestrated by LangGraph, a KG-enhanced retrieval pipeline, and an MSMS planning engine.
- The framework employs a Diagnostic Agent for cognitive diagnosis, a Planning Agent for path optimization using the MSMS algorithm, and a Tutor Agent for coordination, all grounded by a Dynamic Knowledge Graph that evolves with student state.
- Performance evaluation shows superior structural alignment of learning paths (PathSim) and cognitive diagnosis fidelity compared to LLM prompting baselines, validated by correlation with human expert ratings.

---

[PATCHEVAL: A New Benchmark for Evaluating LLMs on Patching Real-World Vulnerabilities](http://arxiv.org/abs/2511.11019)

- PATCHEVAL: introduces a new benchmark for evaluating LLMs on Automated Vulnerability Repair (AVR) tasks, incorporating Benchmark Construction, an Evaluator with multiple Patch Validation methods, and two Task Formulations (Patch Generation with Location Oracle and End-to-End Patch Generation).
- The benchmark focuses on Python, JavaScript, and Go, curating 1,000 real-world vulnerabilities from 2015-2025 across 65 CWEs, with 230 having runtime sandbox environments for dynamic testing.
- Evaluation reveals that even the best-performing LLM achieves only a 23.0% success rate in single-attempt patch generation, highlighting the difficulty of real-world AVR.

---

[AI Agent-Driven Framework for Automated Product Knowledge Graph Construction in E-Commerce](http://arxiv.org/abs/2511.11017)

- AI Agent-Driven Framework: introduces a fully automated, AI agent-driven framework for constructing product knowledge graphs directly from unstructured product descriptions using Large Language Models (LLMs) across three stages: ontology creation and expansion, ontology refinement, and knowledge graph population.
- The framework utilizes dedicated LLM-powered agents in a modular pipeline to ensure semantic coherence and scalability without requiring predefined schemas or handcrafted extraction rules.
- Evaluation on air conditioner product data demonstrated strong performance, achieving over 97% property coverage in the resulting knowledge graph.

---

[Beyond Accuracy: Behavioral Dynamics of Agentic Multi-Hunk Repair](http://arxiv.org/abs/2511.11012)

- MAPLE (MODEL CONTEXT PROTOCOL FOR AUTOMATED LIGHTWEIGHT REPOSITORY CONTEXT EXTRACTION): introduces a systematic study of four LLM-driven coding agents (CLAUDE CODE, CODEX, GEMINI-CLI, and QWEN CODE) on multi-hunk program repair using fine-grained behavioral metrics and the MAPLE context-assistance mechanism.
- The study evaluates agents on localization success, repair accuracy, regression behavior, and operational dynamics across 372 multi-hunk bugs from the HUNK4J dataset, revealing significant variation in effectiveness correlated with bug complexity metrics like hunk divergence and spatial proximity.
- MAPLE improves repair accuracy for GEMINI-CLI by 30% by enhancing bug localization through structured repository context extraction, highlighting the value of context-assistance for agents with baseline reasoning capabilities.

---

[Exposing Weak Links in Multi-Agent Systems under Adversarial Prompting](http://arxiv.org/abs/2511.10949)

- SAFEAGENTS: introduces a unified and extensible framework for fine-grained security assessment of Multi-Agent Systems (MAS) under adversarial prompting, complemented by the DHARMA diagnostic measure.
- The framework systematically exposes how design choices like plan construction and context sharing affect susceptibility to adversarial inputs across different MAS architectures.
- The study reveals significant vulnerabilities in common design patterns, emphasizing the need for security-aware design in MAS.

---

[When AI Does Science: Evaluating the Autonomous AI Scientist KOSMOS in Radiation Biology](http://arxiv.org/abs/2511.13825)

- KOSMOS (Autonomous AI Scientist): introduces an evaluation of the autonomous AI scientist KOSMOS on three radiobiology hypotheses using a falsification-based auditing methodology with empirical null models.
- The evaluation assessed KOSMOS's claims against null distributions derived from random gene sets or permutations to determine statistical significance and validity.
- The study found one well-supported discovery (CDO1), one ambiguous result (12-gene signature), and one false result (DDR to p53 correlation), highlighting the need for rigorous auditing of AI-generated science.

---

#### 13th November 2025

[Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](http://arxiv.org/abs/2511.10705)

- Co-EPG (Co-Evolution of Planning and Grounding): introduces a self-iterative training framework for autonomous GUI agents, featuring Iterative Training (alternating optimization loop), Grounding SFT (grounding model fine-tuning), Planning SFT (planning model fine-tuning), GRPO (planning model refinement), Rollouts (diverse plan generation), C-DREM (dynamic reward ensemble), Grounding Models (plan executability assessment), Group Computation (advantage calculation), Data Enhancement (dataset refinement), Planner II (planning diversity enhancement), and Verifier Φ (discrimination reliability improvement), which establishes a positive feedback loop for the co-evolution of planning and grounding models.
- The framework enables continuous self-improvement of agent capabilities through self-play optimization and training data distillation, where the planning model explores strategies under grounding-based reward guidance, and the grounding model is optimized with diverse data generated by the planning model.
- Co-EPG leverages a confidence-based dynamic reward ensemble mechanism (C-DREM) to reduce reward noise and accelerate GRPO training, leading to enhanced generalization and state-of-the-art performance on GUI task automation benchmarks without requiring external data.

---

[Safe Planning in Interactive Environments via Iterative Policy Updates and Adversarially Robust Conformal Prediction](http://arxiv.org/abs/2511.10586)

- The paper introduces an iterative framework that robustly maintains safety guarantees across policy updates in interactive environments using Adversarially Robust Conformal Prediction (ACP), which involves Iterative Policy Updates, an Explicit Solver, or an Implicit Solver.
- This framework addresses the "chicken-and-egg" problem where the autonomous agent's policy update changes the environment's behavior distribution, violating standard Conformal Prediction exchangeability assumptions.
- The approach provides episodic safety guarantees by analytically bounding the policy-induced distribution shift and offers explicit conditions for convergence of the uncertainty set radius.

---

[Towards autonomous quantum physics research using LLM agents with access to intelligent tools](http://arxiv.org/abs/2511.11752)

- AI-MANDEL: introduces an LLM agent system that autonomously generates and implements novel ideas in quantum physics by accessing scientific literature and the intelligent discovery tool PYTHEUS, aiming for an AI physicist.
- The system consists of Idea generation agents (Researcher, Novelty, Judge, Mediator) and Idea implementation agents (Expert) interacting with the PYTHEUS tool to produce concrete, actionable experiment designs.
- Successful designs are stored in an Idea Pool and have led to the writing of independent, publishable scientific papers in quantum physics.

---

[nuPlan-R: A Closed-Loop Planning Benchmark for Autonomous Driving via Reactive Multi-Agent Simulation](http://arxiv.org/abs/2511.10403)

- nuPlan-R: introduces a reactive closed-loop planning benchmark by integrating learning-based reactive agents and an interaction-aware agent selection mechanism into the nuPlan framework, replacing rule-based Intelligent Driver Model (IDM) agents.
- The benchmark extends evaluation with Success Rate (SR) and All-Core Pass Rate (PR) metrics to assess planner robustness and performance balance across multiple dimensions.
- The learning-based reactive agents, based on a noise-decoupled diffusion framework (Nexus architecture), produce more realistic, diverse, and human-like traffic behaviors compared to rule-based agents.

---

[AgentEvolver: Towards Efficient Self-Evolving Agent System](http://arxiv.org/abs/2511.10395)

- AgentEvolver: introduces a self-evolving agent system that leverages LLMs' semantic understanding and reasoning to drive autonomous agent learning, addressing high data construction costs, inefficient exploration, and poor sample utilization in current LLM-based agents.
- The system integrates three synergistic mechanisms: self-questioning for curiosity-driven task generation, self-navigating for experience reuse and hybrid policy guidance, and self-attributing for enhanced sample efficiency via differentiated rewards.
- The practical infrastructure supports modularity and scalability, enabling continual improvement of agent capabilities through a unified orchestration loop.

---

[VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction](http://arxiv.org/abs/2511.10203)

- VISTA (Vision and Intent-Aware Social Attention Framework): introduces a recursive goal-conditioned transformer architecture that integrates long-term goals, past trajectories, and social interactions for multi-agent trajectory prediction.
- The framework decouples destination goal prediction from local trajectory generation using a Goal Prediction Module (GPM) and refines predictions recursively within the Trajectory Prediction Module (TPM).
- Key innovations include goal-trajectory fusion via cross-attention and social-token attention, which result in state-of-the-art accuracy and significantly reduced collision rates on dense benchmarks.

---

[ENVTRACE: SIMULATION-BASED SEMANTIC EVALUATION OF LLM CODE VIA EXECUTION TRACE ALIGNMENT—DEMONSTRATED AT SYNCHROTRON BEAMLINES](http://arxiv.org/abs/2511.09964)

- EnvTrace: introduces a simulation-based method that evaluates LLM-generated instrument control code via execution trace alignment with a beamline control-logic digital twin, assessing functional correctness and runtime performance.
- The framework captures state changes (Process Variable updates) from both ground-truth and LLM code execution in a sandboxed environment to compute multi-faceted scores like `pv_match_rate`, `timing_score`, and `temp_score`.
- This approach provides a more reliable measure of code correctness for high-stakes physical systems compared to purely syntactic metrics, enabling safer deployment of LLM agents.

---

[Multi-Agent Multimodal Large Language Model Framework for Automated Interpretation of Fuel Efficiency Analytics in Public Transportation](http://arxiv.org/abs/2511.13476)

- MAML-AIF (Multi-Agent Multimodal Large Language Model Framework for Automated Interpretation of Fuel Efficiency Analytics in Public Transportation): introduces a modular and scalable multi-agent framework leveraging multimodal LLMs to automate data narration and energy insight generation, coordinating specialized agents for iterative refinement.
- The framework operates across four stages: raw data description, data modeling, post hoc analytics, and integration/narration, building cumulative contextual knowledge across stages.
- The system was validated on public bus fuel efficiency data, finding that GPT-4.1 mini with Chain-of-Thought prompting provided the optimal balance of narrative accuracy and computational cost.

---

[HARNESS: Human-Agent Risk Navigation and Event Safety System for Proactive Hazard Forecasting in High-Risk DOE Environments](http://arxiv.org/abs/2511.10810)

- HARNESS (Human-Agent Risk Navigation and Event Safety System): introduces a modular AI framework integrating LLMs with structured data and historical event retrieval for proactive hazard forecasting in high-risk Department of Energy (DOE) environments, utilizing an agentic orchestration structure.
- The system employs a human-in-the-loop mechanism where Subject Matter Experts (SMEs) refine predictions, creating an adaptive learning loop that enhances system performance over time through iterative agentic reasoning.
- Key architectural components include a central Orchestrator Agent coordinating specialized agents for retrieval, analysis, mitigation strategy generation, and final report compilation.

---

[Towards an Agentic Workflow for Internet Measurement Research](http://arxiv.org/abs/2511.10611)

- ArachNet: introduces an agentic workflow system for Internet measurement research that uses four specialized LLM agents (QueryMind, WorkflowScout, SolutionWeaver, RegistryCurator) to independently generate executable measurement workflows mimicking expert reasoning.
- The system automates the systematic reasoning process of problem decomposition, solution design, implementation, and registry evolution, significantly lowering the barrier to composing complex, multi-framework analyses.
- ArachNet validates its capabilities by successfully replicating expert-level analysis in Internet resilience scenarios, including single-framework replication, multi-framework orchestration, and temporal forensic investigations.

---

[Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance](http://arxiv.org/abs/2511.10400)

- CP-WBFT (Confidence Probing-based Weighted Byzantine Fault Tolerant): introduces a consensus mechanism leveraging LLM's reflective capabilities via confidence probes (PCP and HCP) to enhance Multi-agent System (MAS) stability against Byzantine faults.
- LLM-based agents show stronger skepticism against erroneous messages than traditional agents, motivating the development of CP-WBFT which uses weighted information flow based on confidence scores.
- The proposed CP-WBFT achieves superior Byzantine Fault Tolerance improvement, especially under extreme fault rates (up to 85.7% malicious nodes), across various network topologies.

---

[Simulating Misinformation Propagation in Social Networks using Large Language Models](http://arxiv.org/abs/2511.10384)

- Auditor-Node Framework: introduces a framework combining persona-conditioned LLM agents and a QA-based auditor to simulate and quantify misinformation evolution across synthetic social networks.
- The framework uses Misinformation Index (MI) and Misinformation Propagation Rate (MPR) to track factual degradation across sequential rewrites by agents mimicking human biases.
- Findings reveal that identity/ideology-based personas accelerate misinformation, while expert/neutral personas act as stabilizers.

---

[Behavior Modeling for Training-free Building of Private Domain Multi Agent System](http://arxiv.org/abs/2511.10283)

- Behavior Modeling for Training-free Building of Private Domain Multi Agent System: introduces a framework for private-domain multi-agent conversational systems that avoids training and data generation by adopting behavior modeling and documentation, utilizing an Orchestrator Agent, a Tool-Calling Agent (TCA), and a General Chat Agent (GCA).
- The core of the approach is 'SpecDoc', a comprehensive document that explicitly details domain knowledge, tool specifications, and usage conventions to align agent behavior via structured prompting.
- This training-free method offers a sustainable path for vertical AI systems by keeping knowledge external, queryable, and easily updatable, mitigating risks like catastrophic forgetting associated with fine-tuning.

---

[Fixed-Persona SLMs with Modular Memory: Scalable NPC Dialogue on Consumer Hardware](http://arxiv.org/abs/2511.10277)

- Fixed-Persona SLMs (Fixed-Persona Small Language Models): introduces a modular NPC dialogue system leveraging SLMs fine-tuned with fixed personas via LoRA and integrated with runtime-swappable memory modules (Conversational memory/World knowledge memory) to enable scalable, expressive dialogue on consumer hardware.
- The architecture decouples character identity (fixed persona in the SLM) from dynamic context (swappable memory stores), allowing a single base model to power multiple distinct NPC instances.
- Evaluation across DistilGPT-2, TinyLlama-1.1B-Chat, and Mistral-7B-Instruct models demonstrated superior dialogue quality with the Mistral-7B-Instruct variant trained on a smaller dataset (OliverS).

---

[GraphIF: Enhancing Multi-Turn Instruction Following for Large Language Models with Relation Graph Prompt](http://arxiv.org/abs/2511.10051)

- GraphIF: introduces a training-free and plug-and-play framework that models multi-turn dialogues as directed relation graphs and leverages graph prompts to enhance the instruction following capabilities of LLMs, with components including an agent-based relation extraction module, a relation graph prompt generation module, and a response rewriting module.
- The framework addresses the limitations of existing methods that treat response generation as isolated tasks by explicitly modeling cross-turn relational constraints using graph structures.
- Extensive experiments show that GraphIF significantly improves performance across multi-turn instruction-following metrics when integrated into instruction-tuned LLMs.

---

[Continuous Benchmark Generation for Evaluating Enterprise-scale LLM Agents](http://arxiv.org/abs/2511.10049)

- Continuous Benchmark Generation Pipeline: introduces a methodology for creating evolving benchmarks for enterprise-scale LLM agents by leveraging developer-authored Knowledge Bases (KBs), KB Analysis, and Reference Implementations.
- The approach addresses challenges in evaluating LLM agents operating under continuously changing enterprise requirements by separating requirement specification (KBs) from concrete evaluation instances derived from migrated services.
- The pipeline uses LLMs' reasoning capabilities to generate evaluation artifacts, such as regular expressions, from semi-structured documents, resulting in cleaner benchmarks than manually created ones.

---

[DemoTuner: Efficient DBMS Knobs Tuning via LLM-Assisted Demonstration Reinforcement Learning](http://arxiv.org/abs/2511.09998)

- DemoTuner: introduces an efficient DBMS knobs tuning framework via LLM-assisted demonstration reinforcement learning, utilizing a structured Chain-of-Thought prompt for condition-aware tuning hints extraction and the HA-DDPGfD algorithm for agent training.
- The framework addresses slow convergence in RL-based tuning by pre-training an agent with extracted explicit and implicit tuning hints, incorporating domain knowledge throughout fine-tuning using hpPER and reward shaping.
- Experimental results on MySQL and PostgreSQL show DemoTuner achieves significant performance gains and lower online tuning costs compared to baselines like DB-BERT, GPTuner, and CDBTune, while also demonstrating superior adaptability to unknown workloads.

---

[SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](http://arxiv.org/abs/2511.09993)

- SPAN (Cross-Calendar Temporal Reasoning Benchmark): introduces a benchmark and evaluation protocol for assessing LLMs' ability to perform temporal reasoning across six different calendar systems, utilizing components like search_calendar and the Time Agent.
- The benchmark covers ten cross-calendar reasoning directions, two reasoning types (date-based and festival-based), and two question formats (polar and content), using a dynamic instance generation protocol to mitigate data contamination.
- Experimental results show current LLMs struggle with an average accuracy of 34.5%, but the Time Agent achieves 95.31% accuracy by leveraging tool-augmented code generation via the search_calendar interface.

---

[HIERROUTER: Coordinated Routing of Specialized Large Language Models via Reinforcement Learning](http://arxiv.org/abs/2511.09873)

- HIERROUTER: introduces a hierarchical routing framework that dynamically assembles inference pipelines from a pool of specialized, lightweight language models using a PPO-based reinforcement learning agent, optimizing response quality against cumulative inference cost.
- The routing process is formalized as a finite-horizon Markov Decision Process (MDP) where the agent selects models across a fixed number of L stages (hops) based on the evolving context, current depth, and accumulated cost.
- The system leverages specialized LLMs for specific tasks, achieving up to 2.4x improvement in response quality over individual models while maintaining cost efficiency through adaptive, multi-hop coordination.

---

#### 12th November 2025

[TaskSense: Cognitive Chain Modeling and Difficulty Estimation of GUI Tasks](http://arxiv.org/abs/2511.09309)

- TaskSense (Cognitive Chain Modeling and Difficulty Estimation): introduces a novel framework for estimating GUI task difficulty by modeling cognitive processes preceding motor actions, using an LLM-based method to automatically extract cognitive chains and their associated difficulty.
- The framework decomposes GUI tasks into sequences of cognitive steps, each with a difficulty index grounded in information theories, and validates its model against both human user completion times and state-of-the-art GUI agent performance.
- TaskSense reveals patterns of Human-AI consistency in cognitive capabilities and identifies current agent limitations on cognitively demanding tasks, paving the way for improved agent training and human-agent task delegation.

---

[ProBench: Benchmarking GUI Agents with Accurate Process Information](http://arxiv.org/abs/2511.09157)

- ProBench: introduces a comprehensive mobile benchmark, with Task Curation (generates, refines GUI tasks), Dynamic Environment (agents interact with device), and Evaluation Pipeline (assesses agent performance), to rigorously evaluate GUI agents' ability to capture and execute necessary operation processes.
- The benchmark includes over 200 challenging GUI tasks across 34 mainstream Chinese and English online applications, covering both State-related and Process-related tasks.
- A key innovation is the Process Provider, which automatically supplies accurate process information via a Structure Description Converter and an MLLM-based Summarizer, enabling precise assessment of intermediate steps.

---

[History-Aware Reasoning for GUI Agents](http://arxiv.org/abs/2511.09127)

- HAR (History-Aware Reasoning) framework: introduces a method to enhance GUI agents' reasoning capabilities by equipping them with stable short-term memory for episodic reasoning through error-aware cognitive correction within a tailored reflection scenario.
- The framework operates in two stages: a GUI Scenario Warm-up Stage for domain-specific knowledge injection via supervised fine-tuning, and a Learning From Failure Stage that enhances short-term memory through reflective learning, tailored correction guidelines, and a hybrid RL reward function.
- This approach transforms the GUI agent's reasoning mode from history-agnostic to history-aware, enabling it to effectively leverage historical interaction clues for robust performance in long-horizon GUI tasks.

---

[Lumine: An Open Recipe for Building Generalist Agents in 3D Open Worlds](http://arxiv.org/abs/2511.08892)

- Lumine: introduces a generalist agent for 3D open worlds, integrating a Vision-Language Model (VLM) (core processing unit), Perception Module (raw pixel input), Hybrid Thinking Strategy (adaptive reasoning), Action Generation Module (keyboard/mouse output), Context Management Module (short/long-term memory), Vision Transformer (ViT) Backbone (visual encoder), LLM Prefill Module (input token processing), and LLM Decode Module (output token generation) to achieve human-like interaction.
- The agent processes raw pixels at 5 Hz, generates 30 Hz keyboard-mouse actions, and adaptively invokes reasoning for complex, long-horizon missions in real-time.
- Trained on Genshin Impact, it demonstrates strong zero-shot cross-game generalization to Wuthering Waves and Honkai: Star Rail, marking a step towards generalist agents in open-ended environments.

---

[Baby Sophia: A Developmental Approach to Self-Exploration through Self-Touch and Hand Regard](http://arxiv.org/abs/2511.09727)

- Baby Sophia: introduces a Reinforcement Learning (RL) framework for autonomous self-exploration in a robotic agent, using the Baby-Bench simulation environment, to learn self-touch and hand regard behaviors via intrinsic rewards.
- The framework utilizes a semantic body map for high-dimensional tactile input compression and employs motor babbling followed by curiosity-based rewards to drive skill acquisition mimicking infant development.
- The approach demonstrates that intrinsic motivation and curriculum learning can enable complex sensorimotor skills from raw, high-dimensional inputs without external supervision.

---

[ECHOING: IDENTITY FAILURES WHEN LLM AGENTS TALK TO EACH OTHER](http://arxiv.org/abs/2511.09710)

- ECHOING: introduces, AxA (Agent x Agent interaction) involving LLM Agents ($\pi_i$) that suffer from echoing (identity abandonment), detected by EchoEvalLM, and mitigated via AgentResponse (Pydantic structure for mitigation), describing this failure mode.
- The study systematically investigates echoing across 60 configurations, 3 domains (hotel booking, car sales, supply chain), and multiple LLM providers, finding rates between 5% and 70%.
- Echoing persists even in advanced reasoning models and is not eliminated by increased reasoning effort or prompt variations, suggesting a need for architectural solutions.

---

[Digital Co-Founders: Transforming Imagination into Viable Solo Business via Agentic AI](http://arxiv.org/abs/2511.09533)

- Conceptual Framework: introduces a three-stage framework—imagination shaping, reality testing, and reality scaling—to articulate how AI-augmented solopreneurs transform inner vision into a sustainable solo business reality, supported by agentic AI.
- The framework details specific inputs, mechanisms, resources (including AI agents), and psychological factors characterizing each stage, emphasizing the recursive nature of the process.
- It bridges macro-level solo economy observations with micro-level mechanisms, providing design implications for tools supporting AI-augmented solopreneurs as "digital co-founders."

---

[BARRIERBENCH: EVALUATING LARGE LANGUAGE MODELS FOR SAFETY VERIFICATION IN DYNAMICAL SYSTEMS](http://arxiv.org/abs/2511.09363)

- BARRIERBENCH: introduces an LLM agentic framework for barrier certificate synthesis that leverages natural language reasoning, Retrieval-Augmented Generation (RAG), and agentic coordination with SMT-based verification to ensure correctness in dynamical systems.
- The framework utilizes three collaborating agents—Retrieval, Synthesis, and Verifier—to iteratively propose, refine, and validate candidate barrier certificates, including co-synthesis with controllers.
- The associated BARRIERBENCH benchmark comprises 100 dynamical systems to evaluate the framework's capability, achieving over 90% success rate, significantly outperforming single-prompt LLM baselines.

---

[Scaling Environments for LLM Agents in the Era of Learning from Interaction: A Survey](http://arxiv.org/abs/2511.09586)

- Survey: introduces a systematic review of environment scaling methods for LLM agents aligned with the Generation-Execution-Feedback (GEF) loop, covering task generation, task execution, and feedback stages.
- The paper proposes an environment-centric taxonomy to organize scaling methods based on the three stages of the GEF loop: Task Generation, Task Execution, and Feedback.
- A key challenge identified is the Generator-Verifier Asymmetry, which describes the mismatch in intelligence required for task generation/execution versus feedback provision.

---

[Perspectives on a Reliability Monitoring Framework for Agentic AI Systems](http://arxiv.org/abs/2511.09178)

- Reliability Monitoring Framework for Agentic AI Systems: introduces a two-layered framework consisting of an Out-of-Distribution (OOD) Detection Layer and an AI Transparency Layer to monitor the operational reliability of agentic AI systems.
- The framework addresses the fundamental challenge of unpredictable environments by first detecting novel inputs and then providing context on the system's internal response to support human decision-making.
- This approach moves beyond simple novelty detection by integrating diagnostic transparency to distinguish between a failure mode and successful adaptation.

---

[Learning Efficient Communication Protocols for Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2511.09171)

- Generalized MARL Framework: introduces a generalized framework for learning multi-round communication protocols in Multi-Agent Reinforcement Learning (MARL) systems, utilizing Observation Processing/Multi-Round Communication Protocol (Message Encoding/Topology Selection/Message Aggregation/Hidden State Update) and Policy Optimization/Decision Making components.
- The framework is evaluated using three novel Communication Efficiency Metrics (CEMs): Information Entropy Efficiency Index (IEI), Specialization Efficiency Index (SEI), and Topology Efficiency Index (TEI).
- The research proposes incorporating IEI and SEI directly into the loss function as regularization terms to achieve efficiency augmentation without increasing communication rounds.

---

[MACEVAL: A MULTI-AGENT CONTINUAL EVALUATION NETWORK FOR LARGE MODELS](http://arxiv.org/abs/2511.09139)

- MACEVAL (Multi-Agent Continual Evaluation network): introduces a dynamic continual evaluation framework that measures the progress of large models autonomously by implementing a multi-agent collaboration system, modeling evaluation as a multi-round interview process.
- The framework utilizes specialized agents (Interviewee, Interviewer, Supervisor) within a graph-based MAEN structure and employs an AUC-inspired metric for sustainable performance assessment.
- It addresses issues like data contamination and human dependency by using in-process, AI-generated, open-ended tasks across visual perception, text comprehension, math, algorithm, and coding capabilities.

---

[Towards a Generalisable Cyber Defence Agent for Real-World Computer Networks](http://arxiv.org/abs/2511.09114)

- TERLA (Topological Extensions for Reinforcement Learning Agents): introduces a set of extensions for Deep Reinforcement Learning agents, specifically applied to a Proximal Policy Optimisation (PPO) model, to achieve generalisability for cyber defence across networks with varying topology and size without retraining, utilizing components like HGTConv, ReLU, and Global Pooling.
- The approach uses heterogeneous graph neural network layers to create a fixed-size latent embedding of the network state, enabling topology and size invariance for the policy learning stage.
- Key architectural elements include an Observation Converter, a Representation Learning stage using HGT layers, and a Policy Learning stage with a reduced, fixed-size action space.

---

[UniMM-V2X: MoE-Enhanced Multi-Level Fusion for End-to-End Cooperative Autonomous Driving](http://arxiv.org/abs/2511.09013)

- UniMM-V2X: introduces a novel end-to-end multi-agent framework that enables hierarchical cooperation across perception, prediction, and planning, utilizing a multi-level fusion strategy and a Mixture-of-Experts (MoE) architecture in both the BEV encoder and motion decoder.
- The framework integrates cooperative information fusion at perception and prediction levels, with MoE dynamically generating task-specialized BEV representations and expert-guided motion queries.
- This unified MoE-enhanced multi-level fusion paradigm achieves state-of-the-art performance across perception, prediction, and planning tasks on the DAIR-V2X dataset.

---

[Achieving Equilibrium under Utility Heterogeneity: An Agent-Attention Framework for Multi-Agent Multi-Objective Reinforcement Learning](http://arxiv.org/abs/2511.08926)

- AA-MAMORL (Agent-Attention Multi-Agent Multi-Objective Reinforcement Learning): introduces a framework to achieve Bayesian Nash Equilibrium (BNE) in Multi-Agent Multi-Objective Systems (MAMOS) by implicitly learning a joint belief over other agents' utility functions and policies using a centralized agent-attention critic during training, enabling decentralized execution.
- The framework addresses the challenge of heterogeneous and conflicting objectives in MAMOS by modeling the necessary global preference information within the attention mechanism for Case II (observation-dependent preferences).
- The approach consistently outperforms state-of-the-art methods in MAMO benchmarks by effectively modeling inter-agent influence and dynamic utility variations.

---

[SlideBot: A Multi-Agent Framework for Generating Informative, Reliable, Multi-Modal Presentations](http://arxiv.org/abs/2511.09804)

- SlideBot: introduces a modular, multi-agent slide generation framework that integrates LLMs with retrieval, structured planning, and code generation, organized around pillars of informativeness, reliability, and practicality.
- The framework decomposes slide creation into three stages: Content Retrieval, Slide Draft Generation, and Presentation Enhancement, coordinated by a central Moderator agent.
- It incorporates principles from Cognitive Load Theory (CLT) and Cognitive Theory of Multimedia Learning (CTML) to ensure pedagogically sound and context-grounded presentations.

---

[Evaluating Software Process Models for Multi-Agent Class-Level Code Generation](http://arxiv.org/abs/2511.09794)

- Waterfall Model: introduces a multi-agent workflow structured around the classical Waterfall software process model (Requirement $\rightarrow$ Design $\rightarrow$ Implementation $\rightarrow$ Testing) for class-level code generation using specialized LLM agents (Requirement Engineer, Architect, Developer, Tester) compared against a RawPrompt baseline.
- The study evaluates three LLMs (GPT-40-mini, DeepSeek-Chat, and Claude-3.5-Haiku) on 100 Python tasks from the ClassEval benchmark to analyze the impact of process structure on functional correctness and code quality.
- Results indicate that structured workflows reorganize performance, often improving code quality (cleanliness, maintainability) at the expense of functional correctness (Pass@1) and increased reasoning/validation errors, with model performance being highly dependent on the workflow structure.

---

[Self-Correcting Large Language Models: Generation vs. Multiple Choice](http://arxiv.org/abs/2511.09381)

- Self-Correcting LLMs: introduces a systematic investigation comparing self-correction performance trends and error-correction behaviors in Large Language Models (LLMs) across two paradigms: Open-Ended Generation and Multiple-Choice Prediction, utilizing components like Self-Correction, Open-Ended Generation, and Multiple-Choice Prediction.
- The study contrasts the dynamics, finding that generation benefits from flexibility and rapid early gains but risks semantic drift, while multiple-choice offers stability but suffers from logit inertia.
- Findings highlight an inherent adaptability-stability trade-off, suggesting that task structure fundamentally shapes how LLMs benefit from iterative refinement.

---

[Value-Aligned Prompt Moderation via Zero-Shot Agentic Rewriting for Safe Image Generation](http://arxiv.org/abs/2511.11693)

- VALOR (Value-Aligned LLM-Overseen Rewriter): introduces a zero-shot agentic framework for safer and more helpful text-to-image generation by integrating layered prompt analysis with human-aligned value reasoning, utilizing a Multi-granular Safety Detector, an Intention Judgement Module, an LLM-Guided Rewriting Agent, and optional Safety-Guided Regeneration.
- The framework detects risks across lexical, semantic, and value-sensitive dimensions, and uses an LLM to rewrite prompts to preserve user intent while enforcing alignment, achieving up to 100.00% reduction in unsafe outputs.
- VALOR addresses challenges in T2I safety, including semantic jailbreaking and value mismatch, by employing modular system prompts for the rewriting LLM based on detected risk categories.

---

[ENABLING AGENTS TO COMMUNICATE ENTIRELY IN LATENT SPACE](http://arxiv.org/abs/2511.09149)

- Interlat (Inter-agent Latent Space Communication): introduces a paradigm leveraging the last hidden states of an LLM as a representation of its internal state for direct inter-agent communication entirely in latent space, using a Communication Adapter, Reasoning Model, Actor Model, Projector, and MHA.
- This approach bypasses the constraints of natural language by transmitting rich, high-dimensional latent vectors, enabling more expressive and efficient coordination between agents.
- The framework is validated on the ALFWorld benchmark, demonstrating improved performance and substantial latency reduction through compression of the latent messages.

---

#### 11th November 2025

[AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress](http://arxiv.org/abs/2511.08325)

- AgentPRM (Process Reward Models for LLM Agents via Step-Wise Promise and Progress): introduces a novel process reward model for LLM agents that captures both the immediate progress and the long-term promise of each decision, utilizing TD-based estimation with GAE for efficient training.
- This framework guides LLM agents in multi-turn decision-making tasks by evaluating each step's contribution to the final goal and the dependencies between sequential decisions, enabling better progress tracking and exploration-exploitation balance.
- AgentPRM demonstrates superior compute efficiency and robust performance across various agentic tasks and model sizes, and can be seamlessly integrated into reinforcement learning processes for LLM agents.

---

[Material-Based Intelligence: Self-organizing, Autonomous and Adaptive Cognition Embodied in Physical Substrates](http://arxiv.org/abs/2511.08838)

- Material-Based Intelligence (MBI): introduces a paradigm shift focusing on architectures where material-based intelligence arises spontaneously from self-organization, leveraging minimal physical models and intrinsically embedding information-theoretic control within the material's own physics, with components including Self-Organization, Sensing/Transduction, Intrinsic Physical Computation, Active Memory/Adaptation, and Actuation/Response, all grounded in Physical Substrates.
- This framework distinguishes MBI from traditional machine-based intelligence by minimizing the hardware-software separation, embedding computation directly into the material's dynamics, and operating far from thermodynamic equilibrium.
- The functional manifestations of MBI include autonomous, adaptive, and goal-directed behaviors emerging from intrinsic dynamics, requiring local interaction, active memory, embodied computation, and adaptive feedback loops.

---

[Low-cost Multi-agent Fleet for Acoustic Cooperative Localization Research](http://arxiv.org/abs/2511.08822)

- CoUGARs (Configurable Underwater Group of Autonomous Robots): introduces a low-cost, configurable Autonomous Underwater Vehicle (AUV) platform, the CougUV, built from COTS and 3D-printed parts, designed to support multi-agent autonomy research, specifically acoustic localization.
- The platform utilizes a containerized ROS 2 Software Stack, featuring a GTSAM-based State Estimator and decoupled Control Systems, validated through simulation in HoloOcean and in-situ field trials.
- Key hardware components include a Raspberry Pi 5, Teensy 4.1, DVL, and USBL acoustic array, integrated for cooperative localization experiments.

---

[Discovering and exploiting active sensing motifs for estimation](http://arxiv.org/abs/2511.08766)

- BOUNDS (Bounding Observability for Uncertain Nonlinear Dynamic Systems): introduces a computational pipeline to empirically determine observability levels of individual state variables and how they change with sensor motion, using tools from control and information theory, alongside the pybounds package.
- The work also presents the Augmented Information Kalman Filter (AI-KF), which merges data-driven state estimates (from ANNs) with model-based filtering (Kalman Filter) using observability knowledge to improve state estimation robustness.
- The framework is demonstrated by discovering active sensing motifs for a flying agent to estimate variables like wind direction and altitude, and by validating the AI-KF's superior performance over traditional filters in scenarios with sparse observability.

---

[Simulating the Visual World with Artificial Intelligence: A Roadmap](http://arxiv.org/abs/2511.08585)

- Roadmap: introduces a systematic overview of modern video foundation models conceptualized as a combination of an implicit world model and a video renderer, tracing their evolution through four generations based on core capabilities.
- The framework defines a physical world model as a digital simulation engine capable of predicting the next scene conditioned on multimodal inputs and spatial/navigation conditions.
- The four generations (Faithfulness, Interactiveness, Planning, Stochasticity) represent an evolutionary ladder of increasing capability in world modeling.

---

[AlphaResearch: Accelerating New Algorithm Discovery with Language Models](http://arxiv.org/abs/2511.08522)

- AlphaResearch: introduces an autonomous research agent designed to discover new algorithms on open-ended problems by synergizing idea generation, execution-based verification, and simulated peer-review via a novel dual research environment, utilizing an LLM Ensemble and a trained Reward Model (AlphaResearch-RM-7B).
- The system iteratively proposes ideas, verifies them using program execution, and refines proposals based on feedback from both execution results and the simulated peer-review Reward Model.
- AlphaResearch achieved a 2/8 win rate against human researchers on the AlphaResearchComp benchmark, notably discovering a best-of-known performance algorithm for the "packing circles" problem.

---

[Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](http://arxiv.org/abs/2511.08214)

- PGS (Perception-Guided Self-Supervision): introduces a training paradigm for end-to-end autonomous driving that leverages perception outputs as primary supervisory signals for decision-making, explicitly modeling causal relationships via MTPS, STPS, and NTPS components.
- The framework aligns inputs and outputs of the decision-making module with perception results (e.g., lane centerlines, predicted agent motions) to mitigate causal confusion stemming from noisy expert trajectories.
- This perception-guided self-supervision approach, built on a standard end-to-end architecture, achieves state-of-the-art closed-loop performance on the Bench2Drive benchmark.

---

[Effective Game-Theoretic Motion Planning via Nested Search](http://arxiv.org/abs/2511.08001)

- Game-Theoretic Nested Search (GTNS): introduces a novel, scalable, and provably-correct approach for computing Nash Equilibria (NEs) in general dynamical systems using a nested search structure, an outer A*-search on the implicit tensor-product graph, and an inner best-response oracle.
- The framework guarantees convergence to a global NE and allows explicit tuning of the solution via a user-specified global objective function, unlike prior optimization-based or local NE methods.
- GTNS efficiently searches the joint action space by implicitly encoding trajectories and verifying the NE constraint via the inner search, achieving solutions in seconds for autonomous driving and racing scenarios.

---

[From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory](http://arxiv.org/abs/2511.07800)

- Trainable Memory Graph: introduces a novel agent-centric, trainable, multi-layered graph memory framework that abstracts raw agent trajectories into structured decision paths and distills them into high-level, human-interpretable strategic meta-cognition, using reinforcement-based weight optimization to calibrate memory utility.
- The framework integrates this structured memory as an explicit policy prior into the LLM agent's Reinforcement Learning (RL) training loop to guide decision-making and improve learning efficiency.
- Empirically, the learnable graph memory demonstrates robust generalization, enhances strategic reasoning performance, and provides consistent benefits during RL training across diverse question-answering benchmarks.

---

[Bio AI Agent: A Multi-Agent Artificial Intelligence System for Autonomous CAR-T Cell Therapy Development with Integrated Target Discovery, Toxicity Prediction, and Rational Molecular Design](http://arxiv.org/abs/2511.08649)

- Bio AI Agent: introduces a multi-agent artificial intelligence system powered by LLMs that enables autonomous Chimeric Antigen Receptor T-cell (CAR-T) development through collaborative specialized agents, including Target Selection Agent/Toxicity Prediction Agent/Molecular Design Agent/Patent Intelligence Agent/Clinical Translation Agent/Decision Orchestration Agent.
- The system integrates target discovery, safety assessment, molecular optimization, patent analysis, and clinical translation across six specialized, collaborating LLM-powered agents.
- Validation demonstrated autonomous identification of high-risk targets (FcRH5, CD229) and generation of comprehensive development roadmaps, accelerating timelines significantly compared to manual review.

---

[AURORA: Autonomous Updating of ROM and Controller via Recursive Adaptation](http://arxiv.org/abs/2511.07768)

- AURORA (Autonomous Updating of ROM and Controller via Recursive Adaptation): introduces a multi-agent LLM framework automating ROM-based controller design with online adaptation, employing five specialized functional agents collaborating through a shared Code Agent.
- The framework iteratively refines the Reduced-Order Model (ROM) and controller using generation-judge-revision cycles managed by the Code Agent, diagnosing degradation sources via the Evaluation Agent.
- It establishes practical viability for autonomous control design by validating high autonomy and performance improvements over expert-tuned baselines across diverse benchmark systems.

---

[Multi-agent self-triage system with medical flowcharts](http://arxiv.org/abs/2511.12439)

- TriageMD: introduces a proof-of-concept conversational self-triage system that guides LLMs with clinically validated flowcharts from the American Medical Association, providing a structured and auditable framework for patient decision support, leveraging a multi-agent framework consisting of a retrieval agent, a decision agent, and a chat agent.
- The system combines the flexibility of free-text interaction with the rigor of standardized clinical protocols, achieving high accuracy in both flowchart retrieval (95.29% top-3) and navigation (99.10%) across diverse conversational styles.
- This approach demonstrates the feasibility of transparent, accurate, and generalizable AI-assisted self-triage, aiming to improve healthcare resource utilization by managing nonurgent emergency department visits.

---

[OSWORLD-MCP: BENCHMARKING MCP TOOL INVOCATION IN COMPUTER-USE AGENTS](http://arxiv.org/abs/2510.24563)

- OSWorld-MCP: introduces a comprehensive and fair benchmark for evaluating computer-use agents by integrating 158 high-quality MCP Tools and GUI operations in real-world scenarios.
- The benchmark assesses multimodal agents' decision-making, GUI operation, and tool invocation capabilities in a hybrid environment, bridging the gap between pure-GUI and text-based tool-use evaluations.
- New metrics, Tool Invocation Rate (TIR) and Average Completion Steps (ACS), are introduced to provide a nuanced assessment of agents' tool utilization propensity and task completion efficiency.

---

#### 10th November 2025


[IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction](http://arxiv.org/abs/2511.07327)

- IterResearch (Iterative Deep-Research Paradigm): introduces a novel iterative deep-research paradigm that reformulates long-horizon research as a Markov Decision Process with strategic workspace reconstruction, maintaining sustained reasoning capacity through periodic synthesis and an evolving report memory.
- The framework addresses context suffocation and noise contamination by maintaining a bounded Workspace S, where each state includes the Question, an evolving Report, and Immediate Context, rather than accumulating all historical information.
- It employs Efficiency-Aware Policy Optimization (EAPO) to train agents for efficient exploration using geometrically discounted rewards and adaptive downsampling, enabling robust performance across extended interactions and diverse tasks.

---

[People Perceive More Phantom Costs From Autonomous Agents When They Make Unreasonably Generous Offers](http://arxiv.org/abs/2511.07401)

- Phantom Costs Perception Framework: introduces a study investigating how agent type (human/robot), autonomy (autonomous/non-autonomous), and discount size (small/large offer) influence the perception of phantom costs (hidden drawbacks/risks), perceived self-interest (agent's motivation), purchase intention (buying likelihood), and trust (confidence in agent/product) within a car-buying simulation (experimental scenario), grounded in the Heuristic of Sufficient Explanation (HOSE) model (explains phantom costs).
- The research reveals that robots are perceived as less self-interested than humans, reducing phantom costs, while larger discounts increase phantom costs but also boost purchase intentions, suggesting perceived benefits can outweigh perceived risks.
- Phantom costs were attributed not only to the agent but also to the product and the agent's manager, highlighting multiple sources of suspicion in human-human and human-robot interactions.

---

[Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction](http://arxiv.org/abs/2511.07392)

- SAOP (Surgical Agent Orchestrator Platform): introduces a voice-directed hierarchical multi-agent framework for multimodal patient data interaction during robotic surgery, including a Workflow Orchestrator Agent, task-specific agents (IR, IV, AR), and memory states.
- The platform leverages LLMs for autonomous planning, command refinement, validation, and reasoning to map voice commands to specific tasks like retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models.
- SAOP demonstrates high accuracy and robustness against speech recognition errors and diverse free-form commands, enhancing support for minimally invasive da Vinci robotic surgery.

---


[AGENTICSCIML: COLLABORATIVE MULTI-AGENT SYSTEMS FOR EMERGENT DISCOVERY IN SCIENTIFIC MACHINE LEARNING](http://arxiv.org/abs/2511.07262)

- AgenticSciML (Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning): introduces a collaborative multi-agent framework that coordinates specialized AI agents, including Human, Data Analyst, Evaluator, Root Solution Engineer, Knowledge Retriever, Proposer, Critic, Engineer, Debugger, Result Analyst, and Selector, along with a Knowledge Base, Analysis Base, and Solution Tree, to iteratively propose, critique, and refine SciML solutions for emergent discovery.
- The framework integrates structured debate, retrieval-augmented method memory, and ensemble-guided evolutionary search to generate and assess new hypotheses about architectures and optimization procedures in scientific machine learning.
- AgenticSciML discovers novel SciML strategies that outperform single-agent and human-designed baselines by up to four orders of magnitude in error reduction, demonstrating emergent methodological innovation through collaborative reasoning.

---

[Bridging the Prototype-Production Gap: A Multi-Agent System for Notebooks Transformation](http://arxiv.org/abs/2511.07257)

- Codelevate (Multi-Agent System for Software Architecture): introduces a novel multi-agent system that automatically transforms Jupyter notebooks into production-ready Python codebases, employing a Preprocessor, Dependency Analyzer, and a Multi-agent System with Architect, Developer, and Structure agents.
- This system leverages specialized agents, each with specific roles, working collaboratively through a shared dependency tree to ensure architectural coherence and code quality, utilizing LLMs and tool-calling capabilities for autonomous code transformation.
- Codelevate aims to bridge the prototype-to-production gap by applying critical software engineering principles, resulting in quantifiable improvements in code quality and maintainability while preserving computational semantics.

---

[Resilient by Design – Active Inference for Distributed Continuum Intelligence](http://arxiv.org/abs/2511.07202)

- PAIR-Agent (Probabilistic Active Inference Resilience Agent): introduces a framework for achieving resilience in Distributed Computing Continuum (DCC) systems by collecting logs, constructing a Causal Fault Graph (CFG), inferring faults using Markov blankets and the Free-energy principle, and autonomously healing through active inference.
- The framework ensures adaptive stability, self-healing capability, and sustained operational continuity in complex, heterogeneous DCC environments by continuously monitoring and adaptively reconfiguring the system.
- Theoretical validations confirm the reliability and effectiveness of the proposed approach in managing uncertainties and adapting to diverse failure conditions across cloud, fog, edge, and IoT layers.

---

[Dynamics-Decoupled Trajectory Alignment for Sim-to-Real Transfer in Reinforcement Learning for Autonomous Driving](http://arxiv.org/abs/2511.07155)

- Dynamics-Decoupled Trajectory Alignment: introduces a framework for zero-shot sim-to-real transfer in autonomous driving by decoupling motion planning from vehicle control, utilizing an RL agent, kinematic bicycle model, trajectory-predicting agent, virtual vehicle, real system/vehicle, Stanley controller, and adaptive longitudinal alignment mechanisms (feed-forward/feed-back control, velocity control, freeze, fast-forward strategies).
- The framework trains an RL agent in simulation using a kinematic bicycle model, distills its behavior into a trajectory-predicting agent, and then aligns this virtual trajectory with a real vehicle using a Stanley controller for lateral dynamics and adaptive longitudinal synchronization.
- This approach enables robust zero-shot transfer of RL policies from simulation to reality by minimizing longitudinal and lateral errors without requiring high-fidelity simulators or vehicle-specific dynamics models.

---

[Multi-Agent Reinforcement Learning for Deadlock Handling among Autonomous Mobile Robots](http://arxiv.org/abs/2511.07071)

- MARL-based Methodology for Deadlock Handling: introduces a structured framework for integrating Multi-Agent Reinforcement Learning into logistics planning, encompassing RL Problem Formulation, Model Selection, Algorithm Selection, and System Deployment, to address deadlock situations among Autonomous Mobile Robots.
- This methodology leverages simulation models as learning environments to train MARL algorithms like PPO and IMPALA, particularly using Centralized Training with Decentralized Execution, to develop adaptive policies for collision avoidance and deadlock recovery in complex intralogistics scenarios.
- The framework aims to enhance system resilience and operational efficiency by enabling AMRs to dynamically adapt to changing conditions and resolve conflicts, outperforming traditional rule-based or heuristic methods in congested environments.

---

[Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Forecasting in Autonomous Driving](http://arxiv.org/abs/2511.06649)

- SAML (Semantic-Aware Meta-Learning framework): introduces a novel framework for long-tail motion forecasting in autonomous driving, featuring a Map Encoder (encodes HD map data), Agent Encoder (encodes agent motion histories), Interaction-Aware Encoder (extracts context-aware features), Bayesian Tail Perceiver (quantifies motion tailness), Meta-Memory Adaptation (adapts to rare patterns), and Multi-modal Decoder (generates motion forecasts).
- SAML quantifies motion rarity via semantically meaningful intrinsic (kinematic, geometric, temporal) and interactive (local, global risk) properties, which are fused into a continuous, uncertainty-aware Tail Index by the Bayesian Tail Perceiver.
- The framework's Meta-Memory Adaptation module, guided by the Tail Index, couples a dynamic prototype memory with a MAML-based cognitive set mechanism for rapid adaptation to rare or evolving patterns.

---

[HYBRID ACTION REINFORCEMENT LEARNING FOR QUANTUM ARCHITECTURE SEARCH](http://arxiv.org/abs/2511.04967)

- HyRLQAS (Hybrid-Action Reinforcement Learning for Quantum Architecture Search): introduces a unified framework that couples discrete gate placement and continuous parameter generation within a hybrid action space, including a Tensor-based Circuit Encoding (encodes circuit information), a Hybrid Policy Network (generates hybrid actions) with a Hybrid Policy Network Backbone (shared feature extractor), Hybrid Policy Network Discrete head (selects gate type/position), Hybrid Policy Network Param head (initializes gate parameters), and Hybrid Policy Network Refine head (refines existing parameters), an Environment (executes circuit, provides reward) with an Environment CPU (classical processing unit), Environment External optimizer (fine-tunes circuit parameters), and Environment Quantum circuit (executes quantum operations), and a Batch of Trajectories (stores experience tuples).
- This framework jointly learns circuit topology and parameter initialization while dynamically refining previously placed gates through a reinforcement learning process, aiming to minimize molecular ground-state energy in a variational quantum eigensolver (VQE) environment.
- HyRLQAS achieves lower energy errors and shorter circuits compared to discrete-only and continuous-only baselines by providing favorable parameter initializations and improved circuit structures, leading to more stable and reliable outcomes.

---

[Shocks Under Control: Taming Transonic Compressible Flow over an RAE2822 Airfoil with Deep Reinforcement Learning](http://arxiv.org/abs/2511.07564)

- DRL (Deep Reinforcement Learning): introduces a framework for active flow control of transonic shock-boundary layer interactions over an RAE2822 airfoil using a high-fidelity CFD solver and synthetic jet actuation, employing DRL/PPO/TD3/CFD Solver/Synthetic Jet Actuation components.
- The framework uses a fifth-order spectral Discontinuous Galerkin (DG) method with Adaptive Mesh Refinement (AMR) for accurate flow simulation.
- The study investigates both on-policy PPO and off-policy TD3 algorithms, demonstrating superior performance of TD3 in achieving drag reduction while preserving lift dynamics.

---

[QOC DAO - Stepwise Development Towards an AI Driven Decentralized Autonomous Organization](http://arxiv.org/abs/2511.08641)

- QOC DAO (Question-Option-Criteria Decentralized Autonomous Organization): introduces a structured, stepwise governance framework evolving from human-led to fully autonomous AI-driven processes by integrating the Question-Option-Criteria (QOC) model with AI agents.
- The framework decomposes decisions into a Question, Options, and weighted Criteria, enabling structured, criterion-based evaluations that enhance transparency and fairness in Decentralized Autonomous Organizations (DAOs).
- The stepwise integration involves human-driven, human-in-the-loop, and fully AI-driven stages, utilizing Large Language Models (LLMs) for automated evaluation support.

---

#### 9th November 2025

[CoFineLLM: Conformal Finetuning of Large Language Models for Language-Instructed Robot Planning](http://arxiv.org/abs/2511.06575)

- CoFineLLM (Conformal Finetuning of Large Language Models): introduces the first Conformal Prediction (CP)-aware fine-tuning framework for LLM-based robot planners, explicitly reducing prediction-set sizes and human intervention rates while maintaining CP coverage guarantees.
- The framework integrates CP during training by simulating conformalization within mini-batches and employs a novel loss function combining cross-entropy with a CP-based term to penalize non-singleton prediction sets.
- CoFineLLM utilizes Low-Rank Adaptation (LoRA) and a curriculum-based training scheme to optimize LLM parameters, demonstrating robustness in out-of-distribution scenarios and consistent improvements in help rates and prediction-set size.

---

[FLEX: Continuous Agent Evolution via Forward Learning from Experience](http://arxiv.org/abs/2511.06449)

- FLEX (Forward Learning with Experience): introduces a gradient-free learning paradigm enabling LLM agents to continuously evolve through accumulated experience by constructing a structured experience library via continual reflection on successes and failures, with an LLM Agent, Experience Library, Updater, Actor, and Critic.
- The framework employs a forward learning loop where an Actor explores to collect experiences, a Critic provides semantic feedback, and an Updater integrates distilled knowledge into a hierarchical experience library, guiding future reasoning.
- FLEX demonstrates substantial performance improvements across mathematical reasoning, chemical retrosynthesis, and protein fitness prediction, establishing a scalable and inheritable continuous agent evolution.

---

[AUTO-Explorer: Automated Data Collection for GUI Agent](http://arxiv.org/abs/2511.06417)

- AUTO-Explorer: introduces an automated data collection method for GUI agents, with a GUI Parser (detects UI elements), an Explore Module (determines next actions), a Difference Spot Module (detects new elements), a Critic Module (evaluates interaction significance), a Sampler (selects actions), and Environment Observation (provides GUI states), designed to autonomously parse and explore GUI environments for efficient data gathering.
- The framework utilizes UI Automation (UIA), Optical Character Recognition (OCR), and icon template matching to parse GUI elements, enabling robust interaction with diverse software and web interfaces.
- The system's exploration strategy involves comparing GUI states before and after actions to discover new elements, which are then sampled for subsequent interactions, and includes mechanisms for trajectory termination and error state identification.

---

[The STATION: An Open-World Environment for AI-Driven Discovery](http://arxiv.org/abs/2511.06309)

- The STATION (An Open-World Environment for AI-Driven Discovery): introduces an open-world multi-agent environment that models a miniature scientific ecosystem, with Agents (autonomous researchers), Rooms (distinct functional spaces), Auxiliary Systems (background support mechanisms), and Data/Communication Structures (for interaction and persistence), enabling LLMs to autonomously pursue scientific discovery.
- This framework allows AI agents to engage in long scientific journeys, including reading papers, formulating hypotheses, submitting code, performing analyses, and publishing results, all without centralized coordination.
- The Station fosters emergent behavior and novel scientific breakthroughs by providing a persistent world where agents can explore, create, and collaborate, moving beyond rigid optimization paradigms.

---

[GAIA: A General Agency Interaction Architecture for LLM-Human B2B Negotiation & Screening](http://arxiv.org/abs/2511.06262)

- GAIA (General Agency Interaction Architecture): introduces a governance-first framework for LLM-human agency in B2B negotiation and screening, defining Principal, Delegate (LLM agent), and Counterparty roles, with optional Critic and Moderator, structured by information-gated progression, dual feedback integration, and authorization boundaries.
- This framework employs a formal state machine with commitment detection, Task-Completeness Index (TCI) tracking for information completeness, and structured escalation paths to ensure bounded authorization and human oversight.
- GAIA provides a hybrid validation blueprint combining automated protocol metrics with human judgment to offer a reproducible specification for safe, efficient, and accountable AI delegation across various domains.

---

[ROAR: Robust Accident Recognition and Anticipation for Autonomous Driving](http://arxiv.org/abs/2511.06226)

- ROAR (Robust Accident Recognition and Anticipation for Autonomous Driving): introduces a novel approach for accident detection and prediction, combining a Discrete Wavelet Transform (extracts multi-resolution features), a self-adaptive object-aware module (enhances spatial representations), and dynamic focal loss (mitigates class imbalance) to improve accuracy and robustness in autonomous driving.
- The framework processes input video frames through an object detector and feature extractor, then refines these features using the self-adaptive object-aware module and DWT, before fusing them and passing through a GRU and Temporal Attention Fusion for anticipation probability.
- ROAR integrates spatial, temporal, and hierarchical features, along with a time weight layer, to adjust temporal influence on predictions, demonstrating superior performance on real-world datasets under challenging conditions like sensor degradation and environmental noise.

---

[Dataforge: A Data Agent Platform for Autonomous Data Engineering](http://arxiv.org/abs/2511.06185)

- Dataforge: introduces an autonomous data agent platform for tabular data, leveraging LLM reasoning and grounded validation to automatically perform data cleaning, hierarchical routing, and feature-level optimization through dual feedback loops.
- The system embodies principles of being automatic, safe, and non-expert friendly, ensuring end-to-end reliability without human supervision by iteratively orchestrating grounded actions.
- This framework transforms raw data into AI-ready data, addressing scalability and expertise dependence in data preparation for various AI applications.

---

[A Low-Rank Method for Vision Language Model Hallucination Mitigation in Autonomous Driving](http://arxiv.org/abs/2511.06496)

- LRHM (Low-Rank Hallucination Mitigation): introduces a novel self-contained low-rank approach to automatically rank multiple candidate captions generated by multiple VLMs based on their hallucination levels, using only the captions themselves without requiring external references or model access.
- The method constructs an embedding matrix from VLM-generated captions, applies Singular Value Decomposition to separate a low-rank consensus component from a sparse residual, and then uses the residual magnitude for hallucination scoring.
- This parallelizable architecture achieves sub-second hallucination mitigation, significantly reducing inference time compared to debate approaches, making it practical for real-time autonomous driving applications by improving VLM trustworthiness in safety-critical scenarios.

---


#### 8th November 2025

[RadioSim Agent: Combining Large Language Models and Deterministic EM Simulators for Interactive Radio Map Analysis](http://arxiv.org/abs/2511.05912)

- RadioSim Agent: introduces an agentic framework that unifies LLM-based reasoning with deterministic EM solvers and vision-based analysis for interactive, multimodal, and explainable radio map generation.
- The framework operates through a Reason-Act-Observe cycle, where an LLM interprets user intent, plans tasks, executes EM simulations via a tool library, and analyzes outputs using a vision-enabled LLM.
- It enables users to provide natural-language instructions to perform simulations, visualize EM fields, and interrogate results directly within a unified agentic environment, bridging natural language understanding with physical modeling.

---

#### 7th November 2025

[STAIR: Stability criterion for Time-windowed Assignment and Internal adversarial influence in Routing and decision-making](http://arxiv.org/abs/2511.05715)

- STAIR (Stability criterion for Time-windowed Assignment and Internal adversarial influence in Routing and decision-making): introduces a novel average-cost-based stability criterion for multi-agent routing systems with adversarial agents, linking policy stability to operational metrics like rejected requests.
- This framework incorporates time-window constraints and a wait-time-constrained stage cost to address the limitations of traditional queuing theory and discounted RL stability definitions in adversarial settings.
- STAIR provides a more reliable assessment of long-term behavior and improved interpretability by removing reliance on arbitrary discount factors and better reflecting real-world service constraints.

---

[TeaRAG: A Token-Efficient Agentic Retrieval-Augmented Generation Framework](http://arxiv.org/abs/2511.05385)

- TeaRAG (Token-Efficient Agentic Retrieval-Augmented Generation Framework): introduces a token-efficient agentic RAG framework that optimizes retrieved content density and reasoning step conciseness through a hybrid retrieval and a process-aware training paradigm, including an LLM Agent (controls workflow, plans, reasons, generates), Important Entity Recognition (identifies key entities), Subquery Generation (decomposes query into subqueries), Hybrid Context Retrieval (combines semantic and graph retrieval), Semantic Retrieval (retrieves document chunks), Graph Retrieval (retrieves knowledge triplets), Knowledge Association Graph (KAG) Construction (builds graph from chunks, triplets), Personalized PageRank (PPR) Filtering (filters KAG for relevant content), Summary Generation (summarizes retrieved content), Supervised Fine-Tuning (SFT) (initial training for reasoning format), Iterative Process-aware Direct Preference Optimization (IP-DPO) (iterative training for conciseness, generalization), Reward Design (calculates outcome, format, process rewards), Knowledge Matching (assesses evidence acquisition), and DPO Pair Construction (creates preferred/rejected reasoning paths).
- TeaRAG compresses retrieved content by combining semantic and graph retrieval to build a Knowledge Association Graph, which is then filtered by Personalized PageRank to yield high-density, concise information.
- The framework's two-stage training, including IP-DPO with process-aware rewards, generates high-quality preference data to iteratively optimize LLMs for more concise reasoning paths, significantly reducing output tokens while improving accuracy.

---

[CONVERSE: Benchmarking Contextual Safety in Agent-to-Agent Conversations](http://arxiv.org/abs/2511.05359)

- CONVERSE introduces a dynamic benchmark for evaluating privacy and security risks in multi-turn agent-to-agent conversations, featuring a simulated user environment, an AI assistant, and an external agent interacting across three realistic domains with contextual attacks and pre-generated ground truth.
- The benchmark models autonomous, multi-turn agent-to-agent conversations where malicious requests are contextually embedded within plausible discourse, testing data abstraction, tool use, and preference manipulation.
- It evaluates seven state-of-the-art LLMs, revealing persistent vulnerabilities where privacy attacks succeed in up to 88% of cases and security breaches in up to 60%, highlighting a tension between utility and protection.

---

[TAMAS: BENCHMARKING ADVERSARIAL RISKS IN MULTI-AGENT LLM SYSTEMS](http://arxiv.org/abs/2511.05269)

- TAMAS (Threats and Attacks in Multi-Agent Systems): introduces a benchmark to evaluate the robustness and safety of multi-agent LLM systems, comprising User, Agent Configuration (Centralized Orchestrator, Decentralized Collaboration, Sequential), Agent, Tools, Environment (Interface, Web, Database), Attack Vectors (Impersonation, Direct Prompt Injection, Indirect Prompt Injection, Contradicting Agents, Byzantine Agent, Colluding Agents), LLM Backbones, Underlying Frameworks (AutoGen, CrewAI), and Evaluation Metrics (Effective Robustness Score (ERS), ARIA Framework, Performance under No Attack (PNA)), designed to assess vulnerabilities across diverse attack types and interaction configurations.
- The benchmark includes 300 adversarial instances across six attack types and five high-impact domains, evaluating performance on ten backbone LLMs and three agent interaction configurations from AutoGen and CrewAI frameworks.
- The findings reveal that multi-agent LLM systems are highly susceptible to adversarial attacks, highlighting the urgent need for stronger defense mechanisms and robust design strategies.

---

[Beyond Master and Apprentice: Grounding Foundation Models for Symbiotic Interactive Learning in a Shared Latent Space](http://arxiv.org/abs/2511.05203)

- SIL (Symbiotic Interactive Learning): introduces a framework for human-agent interaction that enables mutual co-adaptation through a shared latent task space, leveraging an Interaction/Feedback Interface, LLM-based Reasoning and Uncertainty Estimation, Command Parser, Shared Task Space for belief alignment, Memory Architecture with continual learning safeguards, Perception via Vision-Language Models, and an Action Executor.
- This approach moves beyond the traditional master-apprentice model by allowing both human and agent to adapt reciprocally, improving interaction efficiency and robustness.
- The framework explicitly represents, measures, and aligns human and agent beliefs, facilitating proactive clarification, adaptive suggestions, and shared plan refinement in dynamic real-world environments.

---

[SELF-INTEREST AND SYSTEMIC BENEFITS: EMERGENCE OF COLLECTIVE RATIONALITY IN MIXED AUTONOMY TRAFFIC THROUGH DEEP REINFORCEMENT LEARNING](http://arxiv.org/abs/2511.04883)

- SI-DRL (Self-Interested Deep Reinforcement Learning): introduces a framework for self-interested AVs to achieve collective rationality in mixed autonomy traffic, utilizing an SI-DRL agent (Autonomous vehicle decision-maker) interacting with a Driving simulator (Dynamic traffic environment) through State (Vehicle/surrounding info input) inputs, Action (Lane change decisions output) outputs, and a Reward (Speed gain/lane change penalty) function, with a DQN (Q-value function approximator) and Experience Replay (Trajectory storage/sampling) for learning.
- The framework demonstrates that self-interested AVs, trained with a simple reward design, can achieve Pareto-efficient Nash equilibria and improve overall traffic flow by fostering spatial organization, including intra-class platooning and inter-class segregation.
- This research validates the emergence of collective rationality through DRL simulations, showing alignment with game-theoretical predictions and suggesting that enhancing spatial organization benefits all road users in mixed-autonomy systems.

---

[Introducing LongCat-Flash-Thinking: A Technical Report](http://arxiv.org/abs/2509.18883)

- LongCat-Flash-Thinking: introduces an efficient 560-billion-parameter open-source Mixture-of-Experts (MoE) reasoning model, cultivated through a two-phase pipeline of Long CoT Cold-Start Training (initial reasoning capability building) and Large-Scale RL (advanced capability scaling).
- The framework employs a domain-parallel training scheme for decoupled optimization across STEM, Code, and Agentic tasks, fusing resulting expert models into a nearly Pareto-optimal model, powered by the DORA (Dynamic ORchestration for Asynchronous rollout) system.
- This system, a large-scale RL framework, delivers a greater than threefold training speedup over synchronous methods, achieving state-of-the-art performance on complex reasoning tasks with exceptional efficiency, reducing token consumption by 64.5% on AIME-25.

---

#### 6th November 2025

[Environment Agnostic Goal-Conditioning, A Study of Reward-Free Autonomous Learning](http://arxiv.org/abs/2511.04598)

- EAGC (Environment Agnostic Goal-Conditioning): introduces a method to transform regular reinforcement learning environments into goal-conditioned environments, enabling agents to learn tasks autonomously and reward-free by selecting their own goals.
- The approach utilizes a wrapper within the Stable-Baselines3 framework, incorporating modular goal evaluation and selection strategies like uniform sampling, novelty seeking, and intermediate success rate selection.
- EAGC demonstrates comparable performance to externally guided baselines in terms of task solving and training times, while also enabling generic agent training prior to specific use cases.

---

[Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper](http://arxiv.org/abs/2511.04583)

- Jr. AI Scientist: introduces an autonomous AI scientist system that mimics a novice student researcher's workflow, encompassing automatic idea generation, implementation and validation of proposed ideas, and research paper writing.
- The system leverages LLMs for idea generation and novelty checks, and powerful coding agents for handling complex, multi-file implementations and rigorous experimentation.
- It significantly improves generated paper quality by utilizing baseline paper resources, LaTeX sources, PDFs, and codebases across all research pipeline stages, while also reporting identified risks.

---

[Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis](http://arxiv.org/abs/2511.04481)

- Web Agent Sustainability Benchmarking: introduces an empirical and theoretical framework to quantify the energy consumption and CO2 emissions of web agents, advocating for dedicated sustainability metrics in their evaluation.
- The empirical evaluation benchmarks five open-source LLM-driven web agents on various GPUs using the Mind2Web benchmark, while theoretical estimation is applied to agents with proprietary LLMs like GPT-4.
- The research highlights that web agent design and LLM choice significantly impact energy consumption, demonstrating that higher energy use does not always correlate with better performance, and emphasizes the need for transparency in model parameters for accurate estimation.

---

[ForeRobo: Unlocking Infinite Simulation Data for 3D Goal-driven Robotic Manipulation](http://arxiv.org/abs/2511.04381)

- ForeRobo: introduces a generative robotic agent that autonomously acquires manipulation skills by integrating generative simulations with classical control.
- It operates through a self-guided propose-generate-learn-actuate cycle, leveraging LLMs for task proposal and ForeGen for infinite simulation data generation.
- The ForeFormer model, trained on simulated data, predicts 3D goal states for zero-shot sim-to-real transfer and multi-entity generalization in real-world robotic manipulation.

---

[Studying the Effect of Explicit Interaction Representations on Learning Scene-level Distributions of Human Trajectories](http://arxiv.org/abs/2511.04375)

- GMOP (Graph-based Motion Prediction): introduces a normalizing flow-based model to capture joint distributions of human trajectories by factorizing the joint distribution using a learned directed acyclic interaction graph.
- The framework investigates various explicit interaction representations, including Euclidean distance, crossing, and hypothetical crossing heuristics (and their flipped variants), to construct the interaction graph and assess their effect on prediction performance.
- GMOP integrates RNN encoders/decoders, GNNs, and an MLP classifier to process past trajectories and static environment context, learning agent interactions for robust scene-level future trajectory prediction.

---

[Deep reinforcement learning based navigation of a jellyfish-like swimmer in flows with obstacles](http://arxiv.org/abs/2511.04156)

- DRL Framework with SAC: introduces a physics-aware machine learning framework for controlling a bio-inspired jellyfish-like swimmer to navigate complex fluid environments with obstacles, by augmenting the agent's state representation with real-time hydrodynamic forces and torque.
- This framework utilizes a Soft Actor-Critic (SAC) algorithm for policy learning, an A* algorithm for pathfinding, and an immersed boundary method for fluid-structure interaction simulations, enabling the swimmer to perceive wall proximity and orientation through distinct force signatures.
- The explicit force feedback facilitates earlier, smoother maneuvers and exploitation of wall effects for efficient turning, leading to enhanced navigation efficiency and robust underwater exploration capabilities in confined, obstacle-laden spaces.

---

[Benchmarking and Studying the LLM-based Agent System in End-to-End Software Development](http://arxiv.org/abs/2511.04064)

- E2EDevBench (End-to-End Software Development Benchmark): introduces a comprehensive framework for benchmarking LLM-based agents in end-to-end software development, integrating a challenging dataset construction process with a hybrid evaluation methodology.
- The framework includes Dataset Construction (collects, filters, and samples PyPI projects to generate requirements) and an Evaluation Framework (combines automated Test Case Migration and Objective Requirement Verification using an LLM-as-Judge).
- This approach provides a more realistic and robust assessment of agent capabilities by mitigating data leakage, simulating authentic development workflows, and enabling fair comparisons of different agent architectures.

---

[DR. WELL: Dynamic Reasoning and Learning with Symbolic World Model for Embodied LLM-Based Multi-Agent Collaboration](http://arxiv.org/abs/2511.04646)

- DR. WELL (Dynamic Reasoning and Learning with Symbolic World Model for Embodied LLM-Based Multi-Agent Collaboration): introduces a decentralized neurosymbolic framework for cooperative multi-agent planning, enabling LLM-based agents to collaborate on interdependent tasks through a dynamic world model and a two-phase negotiation protocol.
- The framework allows agents to propose and commit to tasks, then independently generate and refine symbolic plans using a shared world model that captures environment state and past experience, ensuring coordination without detailed trajectory sharing.
- By integrating symbolic reasoning with LLM planning, DR. WELL improves coordination efficiency, task completion rates, and interpretability in multi-agent environments, adapting strategies across episodes.

---

[RAGalyst: Automated Human-Aligned Agentic Evaluation for Domain-Specific RAG](http://arxiv.org/abs/2511.04502)

- RAGalyst: introduces an automated, human-aligned agentic framework for domain-specific RAG evaluation, featuring a document preprocessing module, an agentic QA generation pipeline with LLM-based filtering, and an LLM-as-a-Judge evaluation module with prompt-optimized metrics.
- The framework generates high-quality synthetic question-answering datasets from source documents and refines Answer Correctness and Answerability metrics to strongly correlate with human annotations.
- RAGalyst enables rigorous benchmarking of RAG systems across diverse domains like military operations, cybersecurity, and bridge engineering, identifying domain-specific trade-offs and informing design choices for reliable RAG systems.

---

[Beyond Shortest Path: Agentic Vehicular Routing with Semantic Context](http://arxiv.org/abs/2511.04464)

- PAVe (Personalized Agentic Vehicular Routing): introduces a hybrid agentic assistant that augments classical pathfinding algorithms with contextual reasoning, including an LLM agent, Routing Engine Tool, Geospatial Context Tool, Contextual Route Assessment Tool, Central Orchestrator, POIFinder Module, Geospatial Cache, Urban Road Network Graph, and Dijkstra Algorithm.
- This framework leverages an LLM agent for semantic reasoning and contextual understanding to evaluate candidate routes generated by a multi-objective Dijkstra algorithm against user-provided tasks, preferences, and avoidance rules.
- PAVe aims to create personalized, adaptive, and scalable solutions for urban mobility optimization by integrating complex user intent with efficient algorithmic pathfinding using real-world urban datasets and geospatial information.

---

[Speed at the Cost of Quality? The Impact of LLM Agent Assistance on Software Development](http://arxiv.org/abs/2511.04427)

- LLM Agent Impact Evaluation Framework: introduces a study estimating the causal effect of LLM agent assistants (specifically Cursor) on software development velocity and quality, utilizing a DiD Design (causal inference), Staggered Adoption (temporal variation), Propensity Score Matching (control group selection), Panel GMM Models (dynamic interaction analysis), GitHub Data Collection (repository metrics), and SonarQube Metrics Calculation (code quality assessment).
- The study finds that Cursor adoption leads to a significant but transient increase in development velocity, alongside a significant and persistent increase in static analysis warnings and code complexity.
- Further analysis reveals that the accumulated technical debt, indicated by increased warnings and complexity, subsequently causes a long-term slowdown in development velocity, creating a self-reinforcing cycle.

---

[GUI-360°: A COMPREHENSIVE DATASET AND BENCHMARK FOR COMPUTER-USING AGENTS](http://arxiv.org/abs/2511.04307)

- GUI-360°: introduces a comprehensive dataset and benchmark suite for computer-using agents, featuring an LLM-augmented, largely automated pipeline for query sourcing, environment-template construction, task instantiation, batched execution, and LLM-driven quality filtering.
- The framework includes a specialized TrajAgent for automatic trajectory collection, comprising a MAgent for task decomposition, EAgents for perception and action execution, and a Recorder for logging multi-modal data.
- GUI-360° supports three canonical tasks: GUI grounding, screen parsing, and action prediction, providing full-resolution screenshots, accessibility metadata, and reasoning traces across Windows office applications.

---

[Trustworthy LLM-Mediated Communication: Evaluating Information Fidelity in LLM as a Communicator (LAAC) Framework in Multiple Application Domains](http://arxiv.org/abs/2511.04184)

- LAAC (LLM as a Communicator): introduces a multi-agent framework that positions LLMs as intelligent communication intermediaries, featuring an Interview Agent (extracts sender intent), an Extraction Agent (generates structured knowledge), and a Query Agent (responds to recipient queries), to facilitate authentic knowledge exchange.
- This framework aims to overcome the "AI-generated inflation and compression" cycle by capturing sender intent through structured dialogue and enabling recipients to interact directly with this structured knowledge.
- The paper systematically evaluates LAAC's trustworthiness across information capture fidelity, reproducibility, and query response integrity, revealing measurable trust gaps that require addressing for reliable deployment.

---

[BAPPA: Benchmarking Agents, Plans, and Pipelines for Automated Text-to-SQL Generation](http://arxiv.org/abs/2511.04153)

- BAPPA (Benchmarking Agents, Plans, and Pipelines for Automated Text-to-SQL Generation): introduces three multi-agent LLM pipelines, Multi-Agent Discussion Pipeline (iterative critique and refinement), Planner-Coder Pipeline (structured planning and execution), and Coder-Aggregator Pipeline (diverse candidate generation and selection), to enhance Text-to-SQL generation.
- The paper systematically benchmarks these pipelines across various open-source LLMs to evaluate their intrinsic planning, reasoning, and coding abilities for converting natural language questions into SQL queries.
- The research demonstrates that multi-agent collaboration and structured reasoning can significantly improve SQL generation quality and robustness, especially for smaller and mid-scale LLMs.

---

[Agentmandering: A Game-Theoretic Framework for Fair Redistricting via Large Language Model Agents](http://arxiv.org/abs/2511.04076)

- Agentmandering: introduces a game-theoretic framework for fair redistricting, simulating turn-based negotiation between LLM agents representing opposing political interests, with Republican Agent (LLM-powered partisan agent), Democratic Agent (LLM-powered partisan agent), District Information (State political profile data), Choose-and-Freeze Protocol (Turn-based negotiation game), Candidate Generator (Generates feasible districting plans), Unpartitioned Region (Current unassigned territory), Candidate Maps (Set of generated districting plans), Selectable Districts (Districts from chosen map), and Frozen District (Permanently assigned district).
- The framework leverages the Choose-and-Freeze protocol, where LLM agents alternate selecting preferred districting plans and freezing individual districts from a set of candidate maps.
- This approach aims to produce districting outcomes that are robust against partisan manipulation, reduce bias, and achieve lower variance compared to traditional methods.

---

[DETECTING SILENT FAILURES IN MULTI-AGENTIC AI TRAJECTORIES](http://arxiv.org/abs/2511.04032)

- Dataset Curation Pipeline: introduces a comprehensive pipeline for curating datasets from agentic traces for anomaly detection, encompassing Multi-Agentic AI System trace collection, LLM span and trace information extraction, feature engineering, inter-annotator ground truth definition, automated normal/anomaly labeling, and final dataset generation.
- The paper addresses the challenge of detecting silent failures in multi-agentic LLM systems by curating two benchmark datasets from agentic traces and evaluating supervised and semi-supervised anomaly detection methods, achieving high accuracies.
- This work provides the first systematic study of anomaly detection in Multi-Agentic AI systems, offering datasets, benchmarks, and insights to guide future research.

---

[ArchPilot: A Proxy-Guided Multi-Agent Approach for Machine Learning Engineering](http://arxiv.org/abs/2511.03985)

- ArchPilot: introduces a multi-agent system for cost-efficient Neural Architecture Search (NAS) that explicitly decouples generation, evaluation, and orchestration into three collaborating agents: Orchestration Agent (coordinates search, manages memory, budgets), Generation Agent (generates, improves, debugs architectures), and Evaluation Agent (executes proxy training, optimizes proxies).
- This framework leverages multi-proxy evaluation with adaptive reweighting and a restart-enabled Monte Carlo Tree Search (MCTS) algorithm to prioritize high-potential candidates, minimizing reliance on expensive full training runs.
- The system achieves efficient ML engineering under limited budgets by exploring a significantly larger portion of the search space and outperforms state-of-the-art baselines on the MLE-Bench benchmark.

---

[Direct Semantic Communication Between Large Language Models via Vector Translation](http://arxiv.org/abs/2511.03945)

- Dual-Encoder Framework: introduces direct semantic communication between LLMs via vector translation, utilizing a Dual-Encoder Translator to map semantic representations from a LLaMA-2-7B Source to a Mistral-7B Target, which are then integrated via an Injection Mechanism to produce an Enhanced Output from a Semantic Input.
- This framework enables LLMs to share meaning directly at latent speed, bypassing token serialization, by learning bidirectional vector translations and conservatively injecting these translated vectors into the target model's internal processing pipeline.
- The approach demonstrates computational stability and effective semantic transfer across diverse domains, revealing a 2.01:1 bidirectional asymmetry suggesting general-purpose LLMs develop more transferable representations than instruction-tuned variants.

---

[PEFA-AI: Advancing Open-source LLMs for RTL generation using Progressive Error Feedback Agentic-AI](http://arxiv.org/abs/2511.03934)

- PEFA-AI (Progressive Error Feedback Agentic-AI): introduces an agentic flow with User Agent (provides prompt/testbench), Master Agent (parses input, manages agents), Code Generator (generates RTL code), Code Executor (lints, compiles, executes code), Log Summarizer (summarizes error logs), Summary Generator (summarizes group chat), and Optional Human Feedback (user intervention for failures), designed for autonomous Register-Transfer Level (RTL) generation using specialized LLMs and hardware simulation tools.
- This framework employs a novel self-correcting mechanism that leverages iterative error feedback to progressively refine generated RTL code, checking for compilation, functional correctness, and synthesizable constructs.
- The approach demonstrates state-of-the-art pass rates on open-source natural language-to-RTL datasets, bridging the performance gap between open- and closed-source LLMs while being efficient in token counts.

---

[Collaborative Agents for Automated Program Repair in Ruby](http://arxiv.org/abs/2511.03925)

- RAMP (Ruby Automated Multi-agent Program repair): introduces a lightweight, feedback-driven framework for Ruby program repair, employing a team of collaborative agents including a Feedback Integrator Agent (produces initial self-reflection, integrates execution feedback), Test Designer Agent (generates guiding test cases), Programmer Agent (produces candidate repair program), and Test Executor Agent (runs candidate repairs, produces verdicts and traces).
- This framework formulates program repair as an iterative process where agents reflect on errors, generate targeted tests, propose candidate fixes, and validate them through execution feedback, refining solutions until a correct one is found or the iteration budget is exhausted.
- RAMP avoids reliance on large multilingual repair databases or costly fine-tuning, operating directly on Ruby code through lightweight prompting and test-driven feedback, achieving state-of-the-art performance on the XCODEEVAL benchmark for Ruby.

---


[Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach](http://arxiv.org/abs/2511.04393)

- ITERATIVE RMFT (ITERATIVE REGRET-MINIMIZATION FINE-TUNING): introduces a post-training procedure that iteratively distills low-regret decision trajectories, generated by a base LLM, back into the model via supervised fine-tuning to enhance decision-making abilities.
- This self-improving approach leverages the regret metric to automatically elicit and reinforce the LLM's decision-making capabilities, including self-generated reasoning rationales, across diverse online decision-making environments.
- Empirical results demonstrate that ITERATIVE RMFT improves LLMs' performance by achieving lower regret values, better exploration-exploitation tradeoffs, and enhanced generalization across various task specifications and real-world contexts.

---



[Agentic Refactoring: An Empirical Study of AI Coding Agents](http://arxiv.org/abs/2511.04824)

- Agentic Refactoring: introduces a large-scale empirical study of AI agent-generated refactorings in real-world open-source Java projects, analyzing 15,451 refactoring instances across 12,256 pull requests and 14,998 commits.
- The study reveals that agentic refactoring is common, dominated by low-level, consistency-oriented edits, and primarily driven by maintainability (52.5%) and readability (28.1%) concerns.
- Agentic refactoring yields small but statistically significant improvements in structural metrics, particularly for medium-level changes, but currently fails to consistently reduce the overall count of known design and implementation smells.

---

[ReGen: GENERATIVE ROBOT SIMULATION VIA INVERSE DESIGN](http://arxiv.org/abs/2511.04769)

- ReGen (Generative Robot Simulation via Inverse Design): introduces a generative simulation framework that automates simulation design by inferring plausible scenarios and environments from a robot's behavior and textual description, leveraging LLMs to synthesize scenarios via a directed graph translated into a symbolic program for simulation.
- The framework supports augmenting simulations, controllable counterfactual scenario generation, reasoning about agent cognition and mental states, and handling distinct sensing modalities.
- ReGen is demonstrated in autonomous driving and robot manipulation tasks, generating diverse, complex simulated environments with high success rates and enabling controllable generation for corner cases.

---

[DIAP: A Decentralized Agent Identity Protocol with Zero-Knowledge Proofs and a Hybrid P2P Stack](http://arxiv.org/abs/2511.11619)

- DIAP (Decentralized Interstellar Agent Protocol): introduces a novel framework for agent identity and communication that binds identity to an immutable IPFS CID and uses Zero-Knowledge Proofs (ZKP) for stateless ownership proof, enabling persistent, verifiable, and trustless interoperability.
- The architecture employs a layered stack, integrating Libp2p GossipSub for discovery and Iroh (QUIC-based) for high-performance direct interaction, alongside a privacy mechanism using EncryptedPeerID.
- A key engineering contribution is the zero-dependency ZKP SDK, achieved by pre-compiling the Noir circuit using the UniversalNoirManager, simplifying deployment for developers.

---

#### 5th November 2025

[Inter-Agent Trust Models: A Comparative Study of Brief, Claim, Proof, Stake, Reputation and Constraint in Agentic Web Protocol Design—A2A, AP2, ERC-8004, and Beyond](http://arxiv.org/abs/2511.03434)

- Inter-Agent Trust Models: introduces a comparative study of six trust models—Brief (endorsed claims/credentials), Claim (self-proclaimed identity/abilities), Proof (cryptographic verification/attestations), Stake (economic collateral/slashing), Reputation (community feedback/trust scores), and Constraint (technical limits/sandboxing)—and a tiered blueprint (T0-T3) for applying them in agentic web protocols.
- The paper analyzes how existing protocols like A2A, AP2, and ERC-8004 implement these trust models, considering their strengths, weaknesses, and mitigation of LLM-specific fragilities.
- It concludes by recommending hybrid trust model architectures and design guidelines for safer, interoperable, and scalable agent economies, emphasizing a "trustless-by-default" approach for high-impact actions.

---


[Scaling Agent Learning via Experience Synthesis](http://arxiv.org/abs/2511.03773)

- DREAMGYM (Scaling Agent Learning via Experience Synthesis): introduces a unified and scalable RL framework that synthesizes diverse experiences for LLM agent training, utilizing an Agent (LLM-based decision maker), a Reasoning Experience Model (synthesizes states/rewards via CoT), an Experience Replay Buffer (stores/retrieves diverse trajectories), a Curriculum Task Generator (creates challenging task variations), and a Scalable LLM Serving Infra (hosts core components).
- The framework addresses challenges in RL training for LLM agents by generating synthetic, reasoning-based experiences, thereby reducing reliance on costly real-environment rollouts and improving sample efficiency.
- It enables effective online curriculum learning through adaptive task generation and ensures stable policy improvement by providing consistent state transitions and informative reward signals.

---


[A Modular, Data-Free Pipeline for Multi-Label Intention Recognition in Transportation Agentic AI Applications](http://arxiv.org/abs/2511.03363)

- DMTC (Data-less Multi-label Text Classification): introduces a modular, data-free pipeline for multi-label intention recognition in transportation agentic AI applications, leveraging LLMs for synthetic data, Sentence-T5 for semantic embeddings, and a novel online focal-contrastive loss for robust multi-label classification.
- This approach eliminates the need for costly data collection and manual annotation, enhancing accuracy in fine-grained, multi-label intention understanding for agentic AI systems.
- DMTC achieves state-of-the-art performance, outperforming traditional and LLM-based baselines with a Hamming loss of 5.35% and an AUC of 95.92%, laying groundwork for autonomous, intention-aware agents.

---

[Hybrid Fact-Checking that Integrates Knowledge Graphs, Large Language Models, and Search-Based Retrieval Agents Improves Interpretable Claim Verification](http://arxiv.org/abs/2511.03217)

- Hybrid Fact-Verification Pipeline: introduces a modular, real-time fact-checking system that integrates Knowledge Graphs, LLMs, and search-based retrieval agents to improve interpretable claim verification, which includes Claim Input (natural language statement), Entity Linking (detects, disambiguates entities), KG Retrieval (fetches one-hop triples), Evidence Ranking (scores semantic relevance), Classifier (assigns claim label), Web Retrieval (rewrites query, retrieves snippets), Reannotation Study (validates ambiguous cases), and a Fallback Strategy (triggers web search).
- The pipeline employs a KG-first strategy for high precision and interpretability, with a web-based retrieval fallback for broader coverage when KG evidence is insufficient.
- The system achieves high F1 scores on benchmarks like FEVER without task-specific fine-tuning and uncovers valid evidence for claims initially labeled as "Not Enough Information" through a reannotation study.

---

[Toward Autonomous Engineering Design: A Knowledge-Guided Multi-Agent Framework](http://arxiv.org/abs/2511.03179)

- Knowledge-Guided Multi-Agent Framework: introduces a novel multi-agent reasoning framework for autonomous engineering design, incorporating specialized LLM agents (Graph Ontologist, Design Engineer, Systems Engineer) and a human Manager to guide the iterative design and review process.
- The framework leverages knowledge graphs, generated by the Graph Ontologist from existing literature, to imbue the Design Engineer and Systems Engineer LLM agents with domain-specific expertise for generating and evaluating airfoil designs.
- This approach demonstrates a path toward improving efficiency and quality in engineering design by combining LLM knowledge curation with established engineering practices and human-in-the-loop validation.

---

[RefAgent: A Multi-agent LLM-based Framework for Automatic Software Refactoring](http://arxiv.org/abs/2511.03153)

- RefAgent (A Multi-agent LLM-based Framework for Automatic Software Refactoring): introduces a multi-agent LLM-based framework for end-to-end software refactoring, comprising a Context-Aware Planner Agent (identifies opportunities, plans refactoring), Refactoring Generator Agent (generates refactored Java code), Compiler Agent (compiles code, addresses errors), and Tester Agent (tests functionality, fixes failures) to dynamically adapt and autonomously make decisions.
- The framework leverages specialized LLM agents with tool-calling capabilities and iterative feedback loops to identify refactoring opportunities, generate code, ensure compilation, and preserve functionality.
- RefAgent achieves high unit test pass rates, reduces code smells, and improves quality attributes across Java projects, outperforming single-agent approaches and aligning with developer refactorings.

---

[Fiedler-Based Characterization and Identification of Leaders in Semi-Autonomous Networks](http://arxiv.org/abs/2511.02317)

- External Observer-Based Leader Identification: introduces a data-driven algorithm that identifies leader nodes in semi-autonomous consensus networks by processing time series of agent states to estimate the Fiedler vector, sort its components, determine the number of leaders, and finally identify the leader nodes.
- This framework leverages the concept of relative tempo, which relates agents' steady-state velocities to the Fiedler vector, enabling leader identification without prior knowledge of the network topology.
- The approach unifies graph analysis with data-driven inference, providing insights into how leader influence manifests in the network's dynamical response.

---

[Human-AI Co-Embodied Intelligence for Scientific Experimentation and Manufacturing](http://arxiv.org/abs/2511.02071)

- APEX (Agentic-Physical Experimentation) system: introduces human-AI co-embodied intelligence, integrating human researchers/operators (precise execution, control), agentic AI (memory, reasoning, planning, feedback) with its Planning, Step-tracking, Context, and Analysis agents, and a wearable MR hardware platform (MR Goggles) (captures data, provides guidance) for real-time multimodal perception (interprets video, hand/eye tracking), adaptive plan (dynamic procedure adjustment), and feedback (real-time guidance, alerts) in scientific experimentation and manufacturing.
- This framework unifies multimodal perception, multi-agent reasoning, and mixed-reality interaction to enable AI agents to perceive, reason, and act in real-world scenarios, providing 3D visual guidance, error detection, and automated documentation.
- APEX transforms complex manual fabrication into autonomous, traceable, interpretable, and scalable processes, significantly improving reproducibility, skill transfer, and real-time error correction for both expert and novice users.

---

[Outbidding and Outbluffing Elite Humans: Mastering Liar's Poker via Self-Play and Reinforcement Learning](http://arxiv.org/abs/2511.03724)

- Solly: introduces an AI agent that masters reduced-format Liar's Poker against elite humans and LLMs, utilizing self-play, the R-NaD (Regularized Nash Dynamics) actor-critic algorithm, and a Policy Network (MLP) with State, Action, Policy Head, and Value Head components.
- The agent demonstrates elite human-level performance in both heads-up and multi-player settings, outperforming LLMs by developing novel bidding strategies and effective randomized play.
- This research marks the first AI to achieve elite human play in multi-player Liar's Poker, a game characterized by extensive multi-player engagement and a rebid feature, while using relatively limited compute resources.

---

[AnaFlow: Agentic LLM-based Workflow for Reasoning-Driven Explainable and Sample-Efficient Analog Circuit Sizing](http://arxiv.org/abs/2511.03697)

- AnaFlow: introduces an agentic LLM-based workflow for analog circuit sizing, employing specialized LLM agents (Explainer, Matching Finder, DC Goal Setter, Initial Design Generator, DC Reviewer, DC Sizer, Specs Reviewer, Reasoning Sizer, Advisor Reviewer, Equipped Sizer) that collaborate with simulation tools (DC (.op) Simulator, Full Simulator, External Optimizer) and Memory to achieve reasoning-driven, sample-efficient, and explainable circuit sizing.
- The framework mimics an expert analog designer's cognitive workflow, breaking the sizing task into four phases: circuit understanding, DC-OP-focused sizing, reasoning-only sizing, and optimizer-equipped sizing, ensuring a reliable and explainable path to optimized solutions.
- By integrating LLM-based reasoning with simulation and optimization tools, the system significantly reduces required simulations, provides human-interpretable design rationales, and learns from its optimization history to accelerate convergence.

---

[The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents](http://arxiv.org/abs/2511.03690)

- OpenHands Software Agent SDK: introduces a toolkit for implementing software development agents, providing a complete architectural redesign of agent components for the OpenHands framework, built on a modular SDK architecture with four decoupled packages.
- The SDK integrates native sandboxed execution, lifecycle control, model-agnostic multi-LLM routing, and built-in security analysis to offer a practical foundation for prototyping and deploying agents at scale.
- The framework supports seamless local-to-remote execution portability, integrated REST/WebSocket services, and various interactive interfaces for human interaction, demonstrating strong performance on SWE-Bench Verified and GAIA benchmarks.

---

[LiveTradeBench: Seeking Real-World Alpha with Large Language Models](http://arxiv.org/abs/2511.03628)

- LiveTradeBench: introduces a live trading environment for evaluating LLM agents in realistic and evolving markets, featuring live data streaming, a portfolio-management abstraction, and multi-market evaluation across U.S. stocks and Polymarket prediction markets.
- The framework enables LLM agents to observe real-time market prices, news, and their portfolio, then output percentage allocations that balance risk and return, integrating tool use, memory, and reasoning capabilities.
- Evaluations of 21 LLMs reveal that high general reasoning scores do not guarantee superior trading outcomes, models exhibit distinct portfolio styles, and some LLMs effectively adapt decisions using live signals, highlighting a gap between static evaluation and real-world financial competence.

---

[PerfDojo: Automated ML Library Generation for Heterogeneous Architectures](http://arxiv.org/abs/2511.03586)

- PerfDojo: introduces a novel automatic optimization methodology, PerfLLM, for generating ML libraries for heterogeneous architectures, with Finetuned LLM, Embedding, Policy Network, Target Network, Replay Buffer, Loss Computation, Reward Function, Compile and Execute, Code Representation, Transformations, and Applicability Detection components, enabling effective code optimization without prior hardware knowledge.
- The framework frames code optimization as a Reinforcement Learning game within an environment that uses a human-readable, mathematically-inspired code representation to ensure semantic validity throughout transformations.
- This approach achieves significant performance gains across diverse CPU and GPU architectures by leveraging LLMs and RL to discover high-performance code transformations.

---

[U2F: Encouraging SWE-Agent to Seize Novelty without Losing Feasibility](http://arxiv.org/abs/2511.03517)

- U2F (Unknown Unknowns to Functional solutions): introduces a cognitive-inspired, uncertainty-embracing multi-agent architecture for systematically surfacing "Unknown Unknowns" in software engineering, featuring a Discovery Agent, Exploration Agent, and Integration Agent, supported by cognitive enhancement mechanisms and human-AI collaboration.
- The framework operationalizes Unknown Unknowns discovery through cross-domain analogical reasoning, reverse thinking, and external validation, enabling LLMs to engage in deep, modular reasoning across the innovation process.
- U2F demonstrates improved novelty and semantic novelty in solutions while maintaining feasibility, leveraging uncertainty as a source of innovation in software engineering tasks.

---

[HaluMem: Evaluating Hallucinations in Memory Systems of Agents](http://arxiv.org/abs/2511.03506)

- HaluMem (Hallucination in Memory Benchmark): introduces the first operation-level hallucination evaluation benchmark for memory systems, comprising memory extraction, memory updating, and memory question answering tasks.
- This benchmark comprehensively reveals hallucination behaviors across different operational stages of interaction by defining stage-specific gold standards and evaluation metrics.
- HaluMem constructs two user-centric, multi-turn human-AI interaction datasets, HaluMem-Medium and HaluMem-Long, to support evaluation across various context scales and task complexities.

---

[ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications](http://arxiv.org/abs/2511.03497)

- ROSBag MCP Server: introduces an MCP server for analyzing ROS and ROS 2 bag files, enabling natural language interaction with robotic datasets through LLMs and VLMs, featuring LLM Providers, MCP Client/LLM UI, MCP Lab, MCP Host, ROSBag MCP Server, Python3 rosbags library, Filesystem, ROS bags folder, Toolset, JSON-RPC, and stdio.
- The framework provides domain-specific tools for trajectory analysis, laser scan processing, coordinate frame transformations, and time series visualization, bridging complex robotic data with conversational AI interfaces.
- It includes a lightweight UI (MCP Lab) for benchmarking different LLMs and VLMs, demonstrating significant disparities in tool-calling capabilities and performance across models.

---

[RAGBOOST: EFFICIENT RETRIEVAL-AUGMENTED GENERATION WITH ACCURACY-PRESERVING CONTEXT REUSE](http://arxiv.org/abs/2511.03475)

- RAGBOOST (Efficient Retrieval-Augmented Generation with Accuracy-Preserving Context Reuse): introduces an efficient RAG system that achieves high cache reuse without sacrificing accuracy through accuracy-preserving context reuse, with Context Index (tracks KV-cache status), Context Ordering (reorders documents for reuse), Context Deduplication (removes redundant documents), Contextual Hints (preserves reasoning fidelity), and KV-cache (stores key-value pairs).
- The system detects overlapping retrieved items across concurrent sessions and multi-turn interactions, using efficient context indexing, ordering, and de-duplication to maximize reuse while maintaining reasoning fidelity with contextual hints.
- RAGBOOST seamlessly integrates with existing LLM inference engines, improving prefill performance by 1.5–3× and preserving or enhancing reasoning accuracy across diverse RAG and agentic AI workloads.

---

[Towards Realistic Project-Level Code Generation via Multi-Agent Collaboration and Semantic Architecture Modeling](http://arxiv.org/abs/2511.03404)

- PROJECTGEN (Multi-Agent Framework): introduces a multi-agent framework for project-level code generation, decomposing the process into architecture design, skeleton generation, and code filling stages, with each stage involving a generation agent (ArchAgent, SkeletonAgent, CodeAgent) and a judging agent (JudgeA, JudgeS, JudgeC) for iterative refinement and memory-based context management, utilizing a Semantic Software Architecture Tree (SSAT) as a structured architecture representation.
- The framework leverages SSAT to bridge the semantic gap between user requirements and source code, enabling LLMs to interpret architectural intent and progressively generate implementation-level artifacts.
- Iterative refinement, guided by judge feedback and memory-based context management, mitigates error propagation and ensures overall integrity and correctness throughout the project generation process.

---

[EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation](http://arxiv.org/abs/2511.03370)

- EQ-Negotiator (Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation): introduces a novel framework that equips SLMs with dynamic emotional personas for edge-deployable credit negotiation, integrating game theory and a Hidden Markov Model to learn and track debtor emotional states.
- This framework enables SLMs to strategically adapt emotional responses in real-time, counter manipulation, and uphold ethical standards, outperforming larger LLMs in debt recovery and negotiation efficiency.
- By transforming persona modeling from static profiles to dynamic emotional architectures, EQ-Negotiator establishes strategic emotional intelligence as a critical factor for effective, ethical, and privacy-preserving AI negotiators on the edge.

---

[Auditing M-LLMs for Privacy Risks: A Synthetic Benchmark and Evaluation Framework](http://arxiv.org/abs/2511.03248)

- PRISM: introduces a novel framework and benchmark for auditing M-LLMs for privacy risks by generating synthetic multi-modal social media data and evaluating cross-modal privacy inference capabilities using a multi-agent architecture.
- The framework includes a data generation workflow that creates realistic user profiles and corresponding multi-modal posts, and a multi-agent inference architecture with specialized LLMs for textual, image, and multi-modal synthesis.
- Experiments demonstrate that M-LLMs significantly outperform human performance in inferring sensitive attributes from multi-modal data, highlighting the urgent need for robust privacy defenses.

---

[From Measurement to Expertise: Empathetic Expert Adapters for Context-Based Empathy in Conversational AI Agents](http://arxiv.org/abs/2511.03143)

- Empathetic Expert Adapters (EEA): introduces a novel framework for developing and evaluating context-specific empathetic LLMs by analyzing real human-AI conversations, defining task-specific empathy patterns, generating synthetic conversations, measuring empathy with reward models, and training context-specific expert adapters.
- The framework leverages a synthetic multi-turn conversational generation pipeline using GPT-4o and Llama-3-8B-Instruct to create empathy-steered dialogues, which then inform the training of LoRA adapters on a frozen LLM backbone.
- Empirical results demonstrate that EEA significantly reduce the gap between perceived and desired empathy, outperforming baseline and system prompt approaches in maintaining empathy across multi-turn conversations.

---

[A PROPRIETARY MODEL-BASED Safety RESPONSE FRAMEWORK FOR AI AGENTS](http://arxiv.org/abs/2511.03138)

- Caizhi-Safety-Control-Model: introduces a novel safety response framework designed to safeguard LLMs at both input and output levels, including a Safety Risk Classification Model (classifies user queries), a Sensitivity Check Module (evaluates unsafe queries), a Real-time Knowledge Base and Dynamic Retrieval (provides updated information), an Interpretation LLM (generates grounded responses), and a Response Decision Logic (orchestrates query handling).
- The framework employs a supervised fine-tuning-based safety classification model at the input level, utilizing a four-tier taxonomy (Safe, Unsafe, Conditionally Safe, Focused Attention) for precise risk identification and differentiated handling of user queries.
- At the output level, the framework integrates Retrieval-Augmented Generation (RAG) with a specifically fine-tuned Interpretation LLM, ensuring all responses are grounded in a real-time, trustworthy knowledge base to eliminate information fabrication and enable result traceability.

---

[ALAS: TRANSACTIONAL AND DYNAMIC MULTI-AGENT LLM PLANNING](http://arxiv.org/abs/2511.03094)

- ALAS (Transactional and Dynamic Multi-Agent LLM Planning): introduces a five-layer architecture including Workflow Blueprinting Layer (defines task specifications), Agent Factory & Canonical IR Layer (instantiates agents and compiles to IR), Runtime Execution & Localized Repair Layer (manages execution with policies and logs), Revalidation Layer (re-checks feasibility post-repair), and Supervision Layer (selects plans and records metrics), which together enable robust multi-agent LLM planning.
- The framework's operational loop integrates a Plan Proposal Module, Validation Module, Disruption Detection Module, Localized Repair (LCRP) Module, and Commit and Continue Module to dynamically adapt to runtime disruptions and ensure transactional reliability.
- Key components like the Independent Validator, Versioned Execution Log, and Canonical Workflow IR ensure non-circular validation, grounded checks, and portable execution across various workflow runtimes, significantly improving planning robustness and efficiency.

---

[GAIA: AN AGENTIC ARTIFICIAL INTELLIGENCE SYSTEM FOR GEOTHERMAL FIELD DEVELOPMENT](http://arxiv.org/abs/2511.03852)

- GAIA (Geothermal Analytics and Intelligent Agent): introduces an AI-based system for automating and assisting geothermal field development, integrating an LLM-powered task orchestrator, a web-based user interface, a digital twin for physics models and tools, and a multi-modal knowledge base.
- The system employs an agentic retrieval-augmented generation (RAG) workflow, where the GAIA Agent plans and orchestrates multi-step analyses by querying knowledge bases and executing tools within the GAIA Digital Twin.
- GAIA aims to accelerate project workflows, assist human experts in decision-making, and enable automation of the geothermal development process through its modular and extensible design.

---

[KNOWTHYSELF: AN AGENTIC ASSISTANT FOR LLM INTERPRETABILITY](http://arxiv.org/abs/2511.03878)

- KnowThyself: introduces an agentic assistant for LLM interpretability, consolidating existing tools into a chat-based interface where users upload models, pose natural language questions, and obtain interactive visualizations with guided explanations.
- The platform employs an Orchestrator LLM to reformulate queries and contextualize results, an Agent Router to direct queries to specialized agents, and various Specialized Agents (BertViz, TransformerLens, RAG, BiasEval) to perform specific interpretability tasks.
- This modular, multi-agent orchestration framework lowers technical barriers by embedding the entire process into a conversational workflow, providing an extensible and accessible foundation for LLM inspection.

---

[To See or To Read: User Behavior Reasoning in Multimodal LLMs](http://arxiv.org/abs/2511.03845)

- BehaviorLens: introduces a systematic benchmarking framework for evaluating modality trade-offs in user behavior reasoning, utilizing textual, scatter plot, and flowchart representations of transaction data as input for MLLMs to perform next-purchase prediction.
- The framework compares the performance of six MLLMs across these input modalities, assessing prediction accuracy, computational cost, and the quality of generated explanations.
- BehaviorLens reveals that holistic image representations of user history significantly improve next-purchase prediction accuracy without additional computational cost compared to textual representations.

---

[ASAP: an Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training](http://arxiv.org/abs/2511.03844)

- ASAP (Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training): introduces a multi-agent system for auto-optimizing large-scale LLM training performance by diagnosing bottlenecks and proposing sharding configurations.
- It integrates Coordinator, Analyzer, and Proposal agents with Sharding Memory, leveraging performance profiling tools, RAG, and historical optimization data.
- The framework automates the diagnosis of sharding issues and generates explainable, optimized configurations, significantly reducing manual effort and improving hardware efficiency.

---

[Leveraging LLM-based agents for social science research: insights from citation network simulations](http://arxiv.org/abs/2511.03758)

- CiteAgent (Citation Agent) Framework: introduces a simulation framework that leverages LLM-based agents to model human behaviors in citation networks, including Initialization, Socialization, and Creation stages, enabling the generation and analysis of citation network phenomena.
- The framework incorporates LLM-based agents as distinct authors with attributes and memory, facilitating collaborative paper drafting and scholarly search for references, and supports two research paradigms: LLM-SE and LLM-LE.
- CiteAgent allows researchers to test and validate existing theories in network science through customizable experiments, providing insights into power-law distribution, citational distortion, and other social science phenomena.

---

[Approximating the Mathematical Structure of Psychodynamics](http://arxiv.org/abs/2511.05580)

- Psychodynamics Process Theory (PTP): introduces a mathematical framework to formalize human psychodynamics and cognitive processes using a diagrammatic approach based on process theory, making it quantitatively precise and accessible across various fields.
- PTP leverages concepts from quantum cognition and holographic cognition to model mental states as cogit state vectors and their evolution through various internal and external processes, including conscious self-reflection, stimuli, and communication.
- The framework supports hierarchical Bayesian inference for understanding cognitive dynamics, exemplified by the Wittgenstein-Lion Language Game, and offers applications in AI safety, such as analyzing AI-driven cognitive manipulation and developing advanced AI agents.

---

#### 4th November 2025


[Kosmos: An AI Scientist for Autonomous Discovery](http://arxiv.org/abs/2511.02824)

- Kosmos: introduces an AI scientist that automates data-driven discovery by performing iterative cycles of parallel data analysis, literature search, and hypothesis generation, synthesizing discoveries into scientific reports.
- The system leverages LLMs, a structured world model for information sharing, and specialized agents to coherently pursue open-ended research objectives over extended periods.
- Kosmos demonstrates the ability to reproduce existing findings, refine knowledge, and make novel, clinically-relevant discoveries across diverse scientific domains with traceable reasoning.

---


[MEMSEARCHER: TRAINING LLMS TO REASON, SEARCH AND MANAGE MEMORY VIA END-TO-END REINFORCEMENT LEARNING](http://arxiv.org/abs/2511.02805)

- MemSearcher: introduces an agent workflow that iteratively maintains a compact memory and combines the current turn with it, fusing the user's question with memory to generate reasoning traces, perform search actions, and update memory to retain only essential information.
- This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy, and is optimized using multi-context GRPO, an end-to-end RL framework.
- Multi-context GRPO jointly optimizes reasoning, search strategies, and memory management by sampling groups of trajectories under different contexts and propagating trajectory-level advantages.

---

[Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning](http://arxiv.org/abs/2511.02755)

- CORL (Cost-controllable Reinforcement Learning): introduces a centralized multi-LLM framework where a Controller LLM coordinates a pool of Expert LLMs, optimized via Reinforcement Learning with dual objectives for task performance and inference cost, adapting to various Budget Conditions.
- This framework enables dynamic budget-aware decision-making, allowing the system to achieve high performance in high-budget modes while maintaining cost efficiency in low-budget settings.
- The approach leverages a cost-controllable training strategy and dual reward signals to learn judicious use of expert LLMs, generalizing well to unseen data and different budget levels.

---

[Agentic World Modeling for 6G: Near-Real-Time Generative State-Space Reasoning](http://arxiv.org/abs/2511.02748)

- WM-MS3M (World-Modeled Multi-Scale Structured State-Space Mixture): introduces an agentic world modeling paradigm for 6G O-RAN Near-RT control, leveraging a causal MS³M backbone, a lightweight stochastic latent variable, and dual decoders to provide action-conditioned generative state-space reasoning and short-horizon planning.
- This framework enables quantitative "what-if" forecasting and calibrated uncertainty modeling for Key Performance Indicator (KPI) prediction, treating Physical Resource Blocks (PRBs) as explicit control inputs.
- The approach integrates with an MPC/CEM planner to optimize actions within data-driven PRB bounds, ensuring leakage-safe, auditable, and robust control for 6G networks.

---

[CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents](http://arxiv.org/abs/2511.02734)

- CostBench: introduces a scalable, cost-centric benchmark for evaluating LLM agents' multi-turn cost-optimal planning and adaptation capabilities in dynamic environments, featuring a query construction module, an environment module, atomic tools, composite tools, flexible cost assignment, an LLM agent, a trajectory planning module, dynamic blocking events, and a re-planning mechanism.
- The benchmark is situated in the travel-planning domain, comprising tasks solvable via multiple sequences of atomic and composite tools with diverse, customizable costs, and supports four types of dynamic blocking events to simulate real-world unpredictability.
- Evaluations on CostBench reveal a substantial gap in cost-aware planning, with leading models failing to identify cost-optimal solutions in static settings and showing significant performance drops under dynamic conditions, highlighting the need for more robust and adaptive LLM agents.

---

[Curriculum Design for Trajectory-Constrained Agent: Compressing Chain-of-Thought Tokens in LLMs](http://arxiv.org/abs/2511.02690)

- CURLTRAC (Curriculum Design for Trajectory-Constrained Agent): introduces an adaptive curriculum learning strategy for training agents under strict deployment-time constraints, utilizing a teacher component to adjust the permissible cost budget and a student component to update the agent's policy based on rollouts in various environments.
- This strategy enables agents, including RL and LLM agents, to progressively master challenging environments by starting with relaxed trajectory constraints and adaptively tightening them, ensuring efficient learning and adherence to strict deployment conditions.
- When applied to LLMs, CURLTRAC effectively compresses output chain-of-thought tokens, leading to substantial inference speedup and reduced computational cost while maintaining accuracy.

---

[Apriel-H1: Towards Efficient Enterprise Reasoning Models](http://arxiv.org/abs/2511.02651)

- Apriel-H1 (Hybrid Large Language Models): introduces a family of hybrid LLMs that combine Transformer Attention (Multi-Head Attention) and SSM Sequence Mixers (Mamba blocks) through a staged distillation process from a pre-trained transformer teacher, aiming for efficient enterprise reasoning.
- The framework progressively replaces less critical attention layers with linear Mamba blocks, guided by layer importance estimation, to achieve higher inference throughput with minimal performance degradation.
- Apriel-H1 models demonstrate up to 3.4x higher inference throughput compared to pure transformer baselines on reasoning-heavy benchmarks, showcasing substantial efficiency gains.

---

[Adapting General-Purpose Foundation Models for X-ray Ptychography in Low-Data Regimes](http://arxiv.org/abs/2511.02503)

- PtychoBench: introduces a multi-modal, multi-task benchmark for X-ray ptychographic analysis, systematically comparing Supervised Fine-Tuning (SFT) and In-Context Learning (ICL) specialization strategies for Vision-Language Models (VLMs) and LLMs.
- The benchmark evaluates VLM-based artifact detection and LLM-based parameter recommendation in low-data regimes, revealing task-dependent optimal specialization pathways.
- Findings highlight that SFT and ICL are complementary for visual tasks, while ICL on large base models is superior for textual tasks, emphasizing the importance of context-aware prompting and model scale.

---

[Modeling Hawkish-Dovish Latent Beliefs in Multi-Agent Debate-Based LLMs for Monetary Policy Decision Classification](http://arxiv.org/abs/2511.02469)

- Multi-Agent Debate-Based LLMs Framework: introduces a novel approach that simulates the FOMC's collective decision-making process using multiple LLM Agents (interacting decision-makers), each starting with Initial Beliefs (distinct policy stances) and processing Input Data (qualitative policy texts/quantitative macroeconomic indicators/historical policy rate), then revising predictions through Iterative Debate Rounds (sequential prediction revision) mediated by Latent Beliefs (hawkish/dovish stance representation), and finally reaching a Consensus Mechanism (final decision aggregation).
- This framework enhances interpretability by explicitly modeling each agent's internal policy beliefs as a discrete latent variable, demonstrating how these beliefs mediate the perception of input information and interaction dynamics.
- Empirical results show that this debate-based approach significantly outperforms standard LLM-based baselines in predicting central bank policy decisions, providing insights into individual perspectives and social influence on collective forecasts.

---

[From the Laboratory to Real-World Application: Evaluating Zero-Shot Scene Interpretation on Edge Devices for Mobile Robotics](http://arxiv.org/abs/2511.02427)

- Proposed Architecture: introduces a pipeline for zero-shot scene interpretation on edge devices for mobile robotics, integrating a Small VLM for scene description, a Detector + Segmentor for object identification, and Tracking for object monitoring, all feeding into a Decision Making unit, with optional Cloud support for larger LLMs/VLMs.
- This architecture enables mobile robots to perceive, interpret, and make rational decisions in dynamic environments by processing visual information locally on edge devices while preserving privacy.
- The system is evaluated on diverse real-world datasets, demonstrating the capabilities of small VLMs for scene interpretation and action recognition in various outdoor and indoor scenarios.

---

[ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning](http://arxiv.org/abs/2511.02424)

- ReAcTree: introduces a hierarchical task-planning framework that dynamically constructs an LLM agent tree, where agent nodes (LLM-based task planner) reason, act, and expand subgoals, while control flow nodes (coordinates child execution) manage execution strategies, supported by episodic memory (stores subgoal-level experiences) and working memory (shares environment observations) for robust long-horizon task planning.
- This framework addresses limitations of monolithic trajectories by decomposing complex goals into semantically isolated subgoals, preventing error propagation and enhancing tractability for LLMs.
- Experiments demonstrate ReAcTree's consistent outperformance of strong baselines across various LLMs in partially observable settings, showcasing its effectiveness in agentic decision-making.

---

[EvoDev: An Iterative Feature-Driven Framework for End-to-End Software Development with LLM-based Agents](http://arxiv.org/abs/2511.02399)

- EvoDev (Iterative Feature-Driven Framework for End-to-End Software Development with LLM-based Agents): introduces an iterative software development framework that decomposes user requirements into features, constructs a Feature Map for dependencies, and iteratively develops software using LLM-based agents.
- The framework explicitly models dependencies between features and propagates multi-level information (business logic, design, code) as context for subsequent development iterations.
- EvoDev significantly outperforms existing LLM-agent baselines in Android development tasks by improving build success rate and functional completeness through its FDD-inspired iterative workflow.

---

[Revisiting put-that-there, context aware window interactions via LLMs](http://arxiv.org/abs/2511.02378)

- Task-Centric Window Management System: introduces a multimodal, LLM-driven system for managing virtual windows in XR environments, integrating LLM Integration, Scene Understanding, Window Workspace, and User Behaviour components.
- This system enables users to organize virtual windows through natural multimodal interaction, fusing explicit/implicit speech with non-verbal cues like pointing and head-gaze, and semantic scene representations.
- It supports one-to-many action mappings and goal-centric reasoning, allowing the LLM to dynamically infer relevant applications and layout decisions, thereby reducing cognitive load and improving user efficiency.

---

[LIVESECBENCH: A DYNAMIC AND CULTURALLY-RELEVANT AI SAFETY BENCHMARK FOR LLMS IN CHINESE CONTEXT](http://arxiv.org/abs/2511.02366)

- LiveSecBench: introduces a dynamic and continuously updated AI safety benchmark specifically for Chinese-language LLM application scenarios, evaluating models across six critical dimensions (Legality, Ethics, Factuality, Privacy, Adversarial Robustness, and Reasoning Safety) using a culturally-relevant dataset and an ELO rating system.
- The benchmark maintains relevance through a dynamic update schedule that incorporates new threat vectors and regularly refreshes test questions, with planned expansions to include Text-to-Image Generation Safety and Agentic Safety.
- LiveSecBench provides a public online leaderboard and detailed evaluation reports, offering transparent insights into LLM safety performance within Chinese legal and social frameworks.

---

[UNLOCKING THE POWER OF MULTI-AGENT LLM FOR REASONING: FROM LAZY AGENTS TO DELIBERATION](http://arxiv.org/abs/2511.02303)

- Dr. MAMR (Multi-Agent Meta-Reasoning Done Right): introduces a multi-agent LLM reasoning framework that addresses lazy agent behavior by incorporating a meta-thinking agent (decomposes tasks, sets goals), a reasoning agent (executes subtasks, performs computations), a Shapley-inspired causal influence method (measures step-level contribution), a verifiable reward mechanism for restart behavior (rewards adaptive deliberation), and an Aggregated Step-Level Advantage (combines rewards for credit).
- The framework theoretically analyzes and mitigates the root cause of lazy agent behavior in multi-turn Group Relative Preference Optimization (GRPO) by removing a normalization term and introducing a robust causal influence measure.
- Dr. MAMR enhances multi-agent collaboration and reasoning performance on complex tasks by enabling agents to adaptively discard prior outputs and restart reasoning when necessary, leading to more stable training and improved accuracy.

---

[Demo: Statistically Significant Results On Biases and Errors of LLMs Do Not Guarantee Generalizable Results](http://arxiv.org/abs/2511.02246)

- LLM Evaluation Infrastructure: introduces a system for automatically generating diverse medical queries for LLMs and evaluating their answers using multiple LLM-as-a-judge setups and agentic workflows.
- The infrastructure includes a prompt generation pipeline that synthesizes patient demographics, medical histories, disorders, and writing styles to create realistic questions, and an answer evaluation pipeline for detecting hallucinations, omissions, and treatment categories.
- This system facilitates large-scale experiments to investigate LLM biases and errors in patient-facing medical scenarios, highlighting the need for multiple LLM evaluators to ensure generalizable results.

---

[DEEP IDEATION: DESIGNING LLM AGENTS TO GENERATE NOVEL RESEARCH IDEAS ON SCIENTIFIC CONCEPT NETWORK](http://arxiv.org/abs/2511.02238)

- Deep Ideation framework: introduces a system for generating novel research ideas, integrating a Scientific Network (knowledge base), Relation Analysis Module (summarizes keyword connections), Keyword Selection Module (selects impactful keywords), Idea Formulation Module (synthesizes keywords into ideas), Idea Stack (tracks research progress), Critic Model (evaluates idea quality), Router (determines next action), and LLM Agents (perform module tasks).
- The framework employs an iterative explore-expand-evolve workflow, leveraging the scientific concept network to dynamically refine research ideas and incorporating reviewer feedback for continuous improvement.
- This approach significantly enhances the novelty and feasibility of generated research ideas across multiple AI domains, outperforming existing methods.

---

[CONTINUUM: EFFICIENT AND ROBUST MULTI-TURN LLM AGENT SCHEDULING WITH KV CACHE TIME-TO-LIVE](http://arxiv.org/abs/2511.02230)

- Continuum: introduces a tool-call aware LLM serving system with a Scheduler (manages request scheduling), Tool Call Handler (parses tool calls, estimates latency), Tool Call Prediction (predicts tool call duration), KV Cache TTL (pins/unpins KV cache), Request & Multi-turn Info (tracks program state), and Unpin Mechanism (releases expired pins), designed to optimize multi-turn agent workloads by intelligently managing KV cache with time-to-live values.
- The system predicts tool call durations and uses this information to set a Time-to-Live (TTL) for pinning KV cache in GPU memory, preventing unnecessary evictions and re-computations.
- By combining tool-aware KV cache timeout with program-level first-come-first-serve scheduling, Continuum significantly reduces scheduling bubbles and preserves multi-turn continuity for complex agentic workflows.

---

[Training Proactive and Personalized LLM Agents](http://arxiv.org/abs/2511.02208)

- PPP-Agent (Productive, Proactive, and Personalized LLM Agents): introduces a multi-objective reinforcement learning framework that optimizes LLM agents for productivity, proactivity, and personalization using an interactive environment with LLM-based user simulators.
- The framework leverages USERVILLE's prompt vaguenization and preference-aware user simulation to create realistic training scenarios, enabling agents to learn strategic interaction and adapt communication styles.
- It employs a composite reward signal derived from task success, interaction quality, and alignment with user preferences, demonstrating significant improvements over strong baselines.

---

[Optimal-Agent-Selection: State-Aware Routing Framework for Efficient Multi-Agent Collaboration](http://arxiv.org/abs/2511.02200)

- STRMAC (State-Aware Routing Framework for Efficient Multi-Agent Collaboration): introduces a state-aware routing framework for multi-agent collaboration, which includes LLM Agents (perform tasks), a State-based Router (selects optimal agent) with an LLM Encoder (encodes agent private context) and a Router Encoder (encodes current system state), and a Selected Agent (executes next action).
- The framework dynamically selects the most suitable single agent at each step by encoding interaction history and agent knowledge, improving collaboration efficiency and effectiveness.
- It also incorporates a self-evolving data generation approach to accelerate the collection of high-quality execution paths, significantly reducing training data overhead.

---

[Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems](http://arxiv.org/abs/2511.01854)

- Tool-to-Agent Retrieval: introduces a unified framework for LLM multi-agent systems that embeds Tools (API calls, functions, actions) and Agents (MCP servers, sub-agents) in a Shared Vector Space (unified embedding space), connecting them via Metadata Relationships (links tools to agents) within a Unified Tool-Agent Catalog (integrates tools/agents) comprising a Tool Corpus (tool names, descriptions) and Agent Corpus (agent names, descriptions), and utilizing a Retrieval Process (top-K ranking, aggregation) driven by Query Paradigms (input methods) such as Direct Querying (high-level question) or Step-wise Querying (decomposed sub-tasks).
- This framework enables granular tool-level or agent-level retrieval by explicitly modeling tool capabilities and traversing metadata, thereby avoiding context dilution and improving routing for both focused and multi-step queries.
- Evaluations across eight embedding models on the LiveMCPBench benchmark demonstrate consistent improvements in Recall@5 and nDCG@5 over previous state-of-the-art agent retrievers.

---

[Collaborative Large Language Model Inference via Resource-Aware Parallel Speculative Decoding](http://arxiv.org/abs/2511.01695)

- TMA-MASAC (Two-phase Matching-based Association Multi-Agent Soft Actor-Critic): introduces a novel framework that jointly optimizes user association and resource allocation (UARA) for efficient parallel speculative decoding in Mobile Edge Computing (MEC) systems, utilizing a MASAC network for resource allocation and a TMA strategy for user association.
- The framework addresses the challenge of parallelizing autoregressive LLM generation in resource-constrained MEC environments by synchronizing mobile computation and uplink communication, minimizing edge-side computing latency, and ensuring energy efficiency.
- It employs a lightweight draft model on mobile devices and a powerful target model on edge servers, reducing end-to-end latency by up to 28.0% and an average of 23.7% without compromising inference accuracy.

---

[A Collaborative Reasoning Framework for Anomaly Diagnostics in Underwater Robotics](http://arxiv.org/abs/2511.03075)

- AURA (Autonomous Resilience Agent): introduces a collaborative framework for anomaly and fault diagnostics in underwater robotics, integrating a Digital Twin (DT) (real-time normative model), Real AUV (physical vehicle), Simulator (virtual replica), Statistical Anomaly Detection (detects state deviations), State Anomaly Characterisation Agent (Agent A) (low-level perception LLM), Anomaly Digest (structured problem description), Diagnostic Reasoning Agent (Agent B) (high-level cognitive LLM), Human Operator (interactive dialogue partner), Vector Database (VDB) (stores distilled lessons), Embedding Model (converts text to vectors), Featured Cloud Search (external knowledge source), ROS 2 topics (human-robot interface), and Orchestration Framework (LangChain) (manages Agent B's flow).
- This framework employs a two-agent LLM design with distinct responsibilities, where Agent A monitors telemetry and translates data into natural language, and Agent B engages a human operator in dialogue to determine root causes, supported by external knowledge.
- The human-validated diagnosis is processed into a new training example, stored in the VDB via an Embedding Model, refining Agent A's perceptual model and enabling continuous learning from human feedback.

---

[PoCo: Agentic Proof-of-Concept Exploit Generation for Smart Contracts](http://arxiv.org/abs/2511.02780)

- PoCo (Agentic Proof-of-Concept Exploit Generation): introduces an agentic framework that automatically generates executable PoC exploits for smart contracts from natural-language vulnerability descriptions, utilizing an LLM within a Reason-Act-Observe loop and a suite of specialized tools.
- The framework accepts a target smart contract and an auditor-written vulnerability annotation as input, producing a Foundry-compatible executable PoC exploit as output.
- PoCo significantly reduces the effort and time required for high-quality PoC generation in smart contract audits, providing verifiable evidence for auditors and actionable test cases for developers.

---

[A Criminology of Machines](http://arxiv.org/abs/2511.02895)

- A Criminology of Machines: introduces a conceptual framework for understanding crime and social control in a hybrid society, defining AI agency through computational, social, and legal dimensions, and classifying deviant behaviors into maliciously aligned systems and unplanned emergent deviance.
- This framework addresses the implications of increasing autonomous AI agents and their machine-machine interactions, moving beyond viewing AI solely as a tool to recognizing its agency in generating unlawful outcomes.
- The paper highlights the urgent need for criminologists to collaborate with AI experts to predict, mitigate, and govern risks from multi-agent AI systems, especially concerning accountability gaps and emergent behaviors.

---

[Stochastic Redistribution of Indistinguishable Items in Shared Habitation: A Multi-Agent Simulation Framework](http://arxiv.org/abs/2511.02648)

- Stochastic Redistribution of Indistinguishable Items in Shared Habitation: A Multi-Agent Simulation Framework: introduces a discrete-event stochastic model simulating the redistribution of indistinguishable items, like socks, among cohabitants, utilizing autonomous agents, probabilistic mixing, correction, and loss processes over iterative laundry cycles.
- The framework, implemented with SimPy, models item migration through random mixing events, selective recollection, and attrition, demonstrating how even minimal exchange probabilities can lead to emergent asymmetries and long-term disorder.
- This multi-agent system captures the dynamic interplay between order and disorder in shared domestic environments, connecting everyday phenomena to statistical mechanics principles of entropy and diffusion.

---

[Agentic AI for Mobile Network RAN Management and Optimization](http://arxiv.org/abs/2511.02532)

- Agentic AI for RAN Management and Optimization: introduces a framework for autonomous 5G RAN management and optimization, leveraging specialized agents (Master Orchestrator, Analysis, Historical Retrieval, Documentation, Validation) that utilize an LLM Reasoning Module, Memory, and various data tools to detect KPI deviations, diagnose causes, and propose corrective actions.
- This framework enables goal-driven systems to dynamically adapt to changing network conditions, employing design patterns like reflection, planning, and multi-agent collaboration for continuous refinement and autonomous decision-making.
- By integrating large AI models with planning, memory, and reasoning capabilities, the framework addresses the increasing complexity of 5G/6G networks, moving beyond traditional rule-based systems to achieve higher levels of automation and intelligence.

---

[Dexterous Robotic Piano Playing at Scale](http://arxiv.org/abs/2511.02504)

- OMNIPIANIST: introduces an agent capable of performing nearly one thousand music pieces by combining an Optimal Transport (OT) based fingering strategy, large-scale Reinforcement Learning (RL) for data generation, and a Flow Matching Transformer for multi-task imitation learning.
- The OT-based fingering strategy enables RL agents to autonomously discover efficient piano-playing strategies without human demonstrations, generating the diverse RP1M++ dataset from over 2,000 specialist agents.
- The Flow Matching Transformer leverages the RP1M++ dataset to learn a multi-song policy, achieving human-level dexterity and strong generalization across various musical tasks.

---

[A Spatially Informed Gaussian Process UCB Method for Decentralized Coverage Control](http://arxiv.org/abs/2511.02398)

- SIGP-UCB (Spatially Informed Gaussian Process UCB): introduces a novel decentralized algorithm for multi-agent coverage control in unknown spatial environments, utilizing local GP models, a local cost function balancing expected locational cost and variance-based exploration, inducing points selected via a greedy strategy, a communication graph, a consensus protocol for hyperparameters, gradient descent, a temporary buffer, and an Adam optimizer.
- This algorithm allows each agent to autonomously determine its trajectory by minimizing a local cost function, balancing exploration of uncertain regions with exploitation of high-density areas, and updating its GP model using local observations and neighbor communication.
- The decentralized approach, employing sparse GPs and local information sharing, enhances scalability and enables agents to escape local minima, leading to improved coverage efficiency compared to centralized and model-based methods.

---

[LACY: A Vision-Language Model-based Language-Action Cycle for Self-Improving Robotic Manipulation](http://arxiv.org/abs/2511.02239)

- LACY (Language-Action CYcle): introduces a unified VLM framework built upon a single LLaVA-NeXT model, fine-tuned to perform language-to-action generation (L2A), action-to-language explanation (A2L), and semantic consistency verification (L2C).
- The framework operates as a closed-loop system, leveraging its bidirectional capabilities to autonomously generate and filter new high-quality training data through a self-improving data generation pipeline and a confidence-based active data augmentation strategy.
- This approach significantly improves robotic manipulation task success rates in both simulation and real-world settings by focusing learning on ambiguous cases and reducing reliance on external human supervision.

---

[ACCUMULATING CONTEXT CHANGES THE BELIEFS OF LANGUAGE MODELS](http://arxiv.org/abs/2511.01805)

- Belief Shift Measurement Framework: introduces a three-stage process to measure changes in LLM stated beliefs and behaviors, including initial belief recording, context accumulation through intentional and non-intentional tasks, and post-task belief recording.
- The framework reveals that LLMs' belief profiles are highly malleable, with significant shifts observed in both stated beliefs and behaviors after various interactions.
- This analysis exposes the hidden risk of belief shift in LLMs during extended sessions of talking or reading, impacting their reliability and consistency.

---

[No-Human in the Loop: Agentic Evaluation at Scale for Recommendation](http://arxiv.org/abs/2511.03051)

- ScalingEval: introduces a large-scale, multi-agent benchmarking framework that positions LLMs as judges for evaluating complementary-item recommendations at scale without human annotation, utilizing an Evaluation Generation Query, Tools, Multi-Agent Planning, Memory, Evaluation Report, and Scalable Majority-vote Ground Truth Synthesis.
- The framework orchestrates specialized LLM agents for CI pattern auditing, recommendation issue identification, and report generation, supported by data retrieval, analysis, and batch processing tools.
- It employs a scalable majority-vote ground truth synthesis mechanism, where multiple LLMs independently evaluate item pairs, and their judgments are aggregated to produce robust consensus results.

---

[UNSUPERVISED EVALUATION OF MULTI-TURN OBJECTIVE-DRIVEN INTERACTIONS](http://arxiv.org/abs/2511.03047)

- UEF (Unsupervised Evaluation Framework): introduces a suite of unsupervised metrics for evaluating multi-turn objective-driven LLM interactions, including LLM-guided Clustering (for user goals), an Interaction Completeness Metric (for goal completion), and a Response Uncertainty Metric (for LLM confidence).
- The framework leverages statistical properties of unlabeled interaction data and fine-tuned LLMs to adapt to distributional shifts, providing LLM judge-free metrics without relying on human-generated ideal responses.
- The approach is validated on open-domain and task-specific interaction data, demonstrating its ability to label user goals, measure goal completion, and quantify LLM uncertainty effectively.

---

[PublicAgent: Multi-Agent Design Principles From an LLM-Based Open Data Analysis Framework](http://arxiv.org/abs/2511.03023)

- PublicAgent: introduces a multi-agent framework for open data analysis, with Orchestrator Agent (coordinates agents, validates progress), Intent Clarifying Agent (resolves query ambiguities), Data Discovery Agent (semantic search, metadata synthesis), Data Analysis Agent (generates, validates statistical code), and Report Generation Agent (synthesizes findings, adds caveats), which addresses LLM limitations in end-to-end analytical workflows by decomposing tasks into specialized agents.
- This framework enhances data accessibility for non-experts by providing natural language interfaces for query clarification, dataset discovery, statistical analysis, and comprehensive report generation from public data repositories.
- The multi-agent architecture improves performance, mitigates distinct failure modes, and offers architectural benefits across task complexities, demonstrating the value of specialization independent of base LLM strength.

---

[LEGO-EVAL: TOWARDS FINE-GRAINED EVALUATION ON SYNTHESIZING 3D EMBODIED ENVIRONMENTS WITH TOOL AUGMENTATION](http://arxiv.org/abs/2511.03001)

- LEGO-EVAL: introduces a comprehensive evaluation framework for text-guided 3D scene synthesis, utilizing Constraint Identification (identifies constraints), Tool Execution Planning (generates tool plans), Argument Selection & Execution (selects arguments and executes tools), and Constraint Validation (assesses scene alignment using LLM/VLM) with a diverse Tool Set (for environment interaction, textual, and multimodal reasoning).
- The framework addresses limitations of existing methods by performing multi-hop grounding of scene components and verifying attributes and spatial relationships through tool-augmented VLMs.
- LEGO-EVAL, along with the LEGO-BENCH dataset, provides a robust and interpretable evaluation for 3D scene generation, demonstrating superior agreement with human judgments compared to baselines.

---

[Cache Mechanism for Agent RAG Systems](http://arxiv.org/abs/2511.02919)

- ARC (Agent RAG Cache Mechanism): introduces a novel, annotation-free caching framework that dynamically manages small, high-value corpora for each LLM agent by synthesizing historical query distribution patterns with the intrinsic geometry of cached items in the embedding space.
- This framework leverages query-based dynamics and structural properties of the item representation space, drastically reducing storage requirements while preserving retrieval effectiveness.
- ARC achieves a 79.8% cache has-answer rate and an 80% average reduction in retrieval latency, significantly enhancing efficiency and effectiveness in RAG-powered LLM agents.

---

[AgentSLA: Towards a Service Level Agreement for AI Agents](http://arxiv.org/abs/2511.02885)

- AgentSLA (Service Level Agreement for AI Agents): introduces a framework for defining Service Level Agreements for AI agents, including an extended Quality Model (ISO/IEC 25010 extension), the AgentSLA DSL, its Metamodel, a Validating Parser, and key entities like Agent, ModelCard, Provider, QoSMetric, SLA, and SLO, leveraging protocols such as Agent2Agent Protocol (A2A) and Model Context Protocol (MCP).
- The framework addresses the challenge of specifying Quality of Service (QoS) for AI agents by extending the ISO/IEC 25010 standard with new quality characteristics like Sustainability, Autonomy, Interoperability, Understandability, and Output properties.
- The AgentSLA DSL, with its JSON-based concrete syntax and Python parser, enables formal and automatic processing of SLAs, facilitating the integration and quality assurance of AI agents in software systems.

---

#### 3rd November 2025

[INSURAGENT: A LARGE LANGUAGE MODEL-EMPOWERED AGENT FOR SIMULATING INDIVIDUAL BEHAVIOR IN PURCHASING FLOOD INSURANCE](http://arxiv.org/abs/2511.02119)

- InsurAgent (A Large Language Model-Empowered Agent for Simulating Individual Behavior in Purchasing Flood Insurance): introduces an LLM-empowered agent for simulating individual flood insurance purchase decisions, integrating perception (parsing user profiles), retrieval (acquiring empirical survey data via RAG), reasoning (emulating human cognitive processes and extrapolating), action (generating purchase probabilities and explanations), and memory (archiving temporal history for dynamic modeling).
- This framework addresses the LLM's limitation in quantitative probability estimation by grounding decisions in empirical data and leveraging common sense for contextual adjustments beyond survey data.
- InsurAgent provides a valuable tool for behavioral modeling and policy analysis by accurately estimating marginal and bivariate probabilities and simulating dynamic decision evolutions over time.

---

[Automated Reward Design for Gran Turismo](http://arxiv.org/abs/2511.02094)

- Iterative LLM-based Reward Design: introduces a scalable iterative framework for automated reward design in Gran Turismo 7, leveraging LLM-based reward generation, VLM preference-based evaluation, and optional human feedback to produce competitive racing agents from text-based instructions.
- The framework efficiently searches a space of reward functions, using a trajectory alignment filter to prune misaligned candidates and a VLM/LLM for preference-based evaluation, replacing the need for a ground-truth fitness metric.
- This system generates reward functions capable of producing racing agents competitive with GT Sophy, a champion-level RL agent, and can also generate novel behaviors in the Gran Turismo 7 environment.

---

[Simulating Environments with Reasoning Models for Agent Training](http://arxiv.org/abs/2511.01824)

- Simia-SFT and Simia-RL: introduce frameworks that enable LLMs to simulate realistic environment feedback for scalable agent training without real environment implementations.
- Simia-SFT is a pipeline that synthesizes supervised fine-tuning data by amplifying small seed sets into diverse trajectories in an environment-agnostic manner.
- Simia-RL enables reinforcement learning training without real environment implementations by generating LLM-simulated feedback, replacing heavy environment engineering with flexible LLM-based simulation.

---

[Hybrid Retrieval-Augmented Generation Agent for Trustworthy Legal Question Answering in Judicial Forensics](http://arxiv.org/abs/2511.01668)

- Hybrid Legal QA Agent: introduces a hybrid legal QA agent for trustworthy legal question answering in judicial forensics, integrating retrieval-augmented generation (RAG) with multi-model ensembling and a dynamic knowledge-base update mechanism.
- The system prioritizes retrieval from a trusted legal repository; if retrieval fails, multiple LLMs generate candidate answers, which are then scored by a specialized selector.
- High-quality outputs undergo human review before being written back into the knowledge base, enabling dynamic knowledge evolution and provenance tracking to ensure reliability and compliance.

---

[Scaling Graph Chain-of-Thought Reasoning: A Multi-Agent Framework with Efficient LLM Serving](http://arxiv.org/abs/2511.01633)

- GLM (Graph Chain-of-Thought with Efficient LLM Serving): introduces a multi-agent Graph-CoT framework with Classification Agent (classifies query type), Reasoning Agent (determines info sufficiency, answers), Action Agent (generates code for retrieval), Graph RAG Retriever (executes code, retrieves graph facts), LLM service/Inference Engine (executes agent prompts), Notebook (accumulates known facts), Vertex-Centric KV Cache Reuse Model (maximizes KV cache reuse), Priority-based KV Cache Eviction Policy (manages cache retention), and Pipelined Execution Strategy (overlaps retrieval, LLM decoding), enabling scalable and efficient graph reasoning for LLMs.
- This framework decomposes complex reasoning tasks into specialized agents and integrates an optimized LLM serving architecture to reduce token cost, latency, and improve throughput.
- The co-designed approach addresses limitations of single-agent Graph-CoT systems by enhancing accuracy and efficiency through selective context sharing and advanced KV-cache management.

---

[UniDataBench: Evaluating Data Analytics Agents Across Structured and Unstructured Data](http://arxiv.org/abs/2511.01625)

- ReActInsight: introduces an autonomous LLM-based agent for end-to-end data analysis across diverse structured and unstructured data sources, featuring Multi-Source Data Exploration & Cross-Source Linkage Discovery (initial data understanding), Heterogeneous Schema Extraction (extracts metadata), Unified Metadata Hub (MetaGraph) Construction (centralizes metadata), Entity-Graph Generation via Similarity Analysis (discovers relationships), Actionable Join-Hint Formulation (creates join instructions), ReAct-style Hierarchical Planning (decomposes analytical goals), Hierarchical Planning Mechanism (breaks down goals), Code Generation with Self-Correction (automates code creation), Code Generation Module (generates executable code), Self-Correction and Debugging Module (ensures code reliability), Adaptive Visualization Techniques (uncovers underlying patterns), Insights Synthesis (distills findings), Insight Synthesis Module (summarizes results), and Model Cascading (optimizes LLM usage).
- The agent initiates its workflow with intelligent multi-source data exploration to build a semantic understanding of how disparate datasets relate, constructing a unified MetaGraph and formulating actionable Join-Hints.
- It employs a hierarchical planning mechanism to decompose high-level goals into answerable sub-questions, generates self-correcting executable code with adaptive visualizations, and synthesizes results into coherent summaries and recommendations, optimizing LLM usage through model cascading.

---

[TPS-BENCH: EVALUATING AI AGENTS' TOOL PLANNING & SCHEDULING ABILITIES IN COMPOUNDING TASKS](http://arxiv.org/abs/2511.01527)

- TPS-Bench (Tool Planning and Scheduling Benchmark): introduces a benchmark for evaluating LLM agents' tool planning and scheduling abilities in compounding tasks, featuring Compounding Tasks, a Tool Repository with Model Context Protocol (MCP) Tools, an LLM Agent, Evaluation Metrics, and an LLM-as-a-judge.
- The benchmark collects 200 compounding tasks of two difficulty levels, requiring agents to select appropriate tools, decompose tasks into subtasks, identify dependencies, and strategically schedule tool execution for efficiency.
- Evaluation emphasizes task completion rate, tool selection score, token usage, and execution time, with an initial study showing reinforcement learning can improve scheduling efficiency and task completion.

---

[LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning](http://arxiv.org/abs/2511.01448)

- LiCoMemory (Lightweight and Cognitive Agentic Memory): introduces an end-to-end agentic memory framework for LLM agents, featuring CogniGraph, a lightweight hierarchical graph for real-time updating and retrieval, which utilizes entities and relations as semantic indexing layers.
- The framework employs temporal and hierarchy-aware search with integrated reranking for adaptive and coherent knowledge retrieval, significantly reducing update latency and improving efficiency.
- LiCoMemory's design enables multi-granular reasoning from abstract contextual understanding to fine-grained evidence retrieval, supporting robust long-term conversational reasoning.

---

[ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction](http://arxiv.org/abs/2511.01188)

- ZoFia (Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction): introduces a novel two-stage zero-shot fake news detection framework that combines entity-guided retrieval for external evidence with a multi-LLM interactive system for collaborative analysis and adversarial debate.
- The framework first employs Hierarchical Salience and SC-MMR algorithms to extract informative and diverse keywords, which are then used to build a comprehensive Multi-Source Information Matrix from internal and external knowledge.
- Subsequently, a multi-agent system, including Linguist, Expert, Claim Extractor, and Claim Verifier, performs multi-view analysis and engages in adversarial debate to produce an interpretable and robust judgment.

---

[MicroRemed: Benchmarking LLMs in Microservices Remediation](http://arxiv.org/abs/2511.01166)

- ThinkRemed (multi-agent framework): introduces a multi-agent framework for end-to-end microservice remediation, comprising a Coordinator, Probe Agent, Execution Agent, Verification Agent, Judge, Auxiliary Context, Failure Report, Microservice Systems, Ansible Playbook, and Reflection.
- This framework emulates Site Reliability Engineer (SRE) reasoning by performing dynamic probing, iterative reasoning, and limited trial-and-reflection cycles to generate effective remediation actions.
- ThinkRemed operates within the MicroRemed benchmark, which evaluates LLMs' ability to autonomously generate executable Ansible playbooks from diagnosis reports to restore system functionality in real microservice environments.

---

[Interaction As Intelligence Part2: Asynchronous Human-Agent Rollout for Long-Horizon Task Training](http://arxiv.org/abs/2510.27630)

- APOLLO: introduces a sampling framework that integrates asynchronous human guidance with action-level data filtering for long-horizon task training, including Agent, Environment, Human-AI Interaction Interface (Frontend), Human, Backend of Human-AI Interaction Interface, LLM As Judge, Raw Trajectory, Masked Trajectory, and Training Set Task.
- This framework enables humans to intervene only when an LLM agent deviates from a promising trajectory, providing strategic advice and prior knowledge to generate valuable trajectories at a lower cost.
- APOLLO applies supervision control to filter out sub-optimal actions, preventing error propagation and demonstrating significant performance improvements on long-horizon, domain-specialized tasks.

---

[InnovatorBench: Evaluating Agents' Ability to Conduct Innovative LLM Research](http://arxiv.org/abs/2510.27598)

- InnovatorBench: introduces a benchmark-platform pair for evaluating AI agents' ability to conduct innovative LLM research, comprising 20 tasks across six research domains, supported by the ResearchGym environment.
- ResearchGym provides a scalable and realistic environment with infrastructure support for multi-computer control, asynchronous execution, and snapshot saving, alongside diverse actions for file operations, web browsing, terminal access, web search, and file parsing.
- The framework assesses LLM agents on end-to-end research tasks, emphasizing innovation and problem-solving, revealing strengths in data-related tasks and weaknesses in algorithmic design and long-horizon planning.

---

[MATHEMATICAL EXPLORATION AND DISCOVERY AT SCALE](http://arxiv.org/abs/2511.02864)

- AlphaEvolve: introduces a generic evolutionary coding agent that combines LLM generative capabilities with automated evaluation in an iterative framework to propose, test, and refine algorithmic solutions for mathematical problems.
- The system iteratively improves a population of programs through a Generator (LLM) that mutates programs and an Evaluator (fitness function) that assigns a numerical score to their performance.
- AlphaEvolve operates in "search mode" to evolve heuristic algorithms or "generalizer mode" to discover programs for any input, and integrates with external AI tools like Deep Think and AlphaProof for formal verification.

---

[Driving scenario generation and evaluation using a structured layer representation and foundational models](http://arxiv.org/abs/2511.01541)

- 5LM (Structured Five-Layer Model): introduces a novel framework for generating and evaluating diverse driving scenarios, leveraging a structured five-layer representation and foundational models to create synthetic visual data from textual descriptions.
- The framework employs a data augmentation strategy where an MLLM analyzes real-world driving scenarios and an LLM edits specific layers of the 5LM to generate Edge Cases, which are then evaluated using semantic embedding-based diversity and originality metrics.
- This approach aims to produce rare and challenging driving scenarios for autonomous vehicle development by focusing on textual description relevance before visual generation, ensuring higher-quality and diverse responses.

---

[From Passive to Proactive: A Multi-Agent System with Dynamic Task Orchestration for Intelligent Medical Pre-Consultation](http://arxiv.org/abs/2511.01445)

- MAS-DTO (Multi-Agent System with Dynamic Task Orchestration): introduces a hierarchical multi-agent framework for intelligent medical pre-consultation, featuring a Controller (select optimal next subtask) that coordinates specialized agents to achieve proactive, structured medical inquiry.
- The framework includes a Virtual Patient (generate clinical presentations), Recipient (update medical records), Triager (perform hierarchical department triage), Monitor (assess subtask completion), Prompter (formulate context-aware inquiry strategies), Inquirer (produce clinical questions), and Evaluator (provide performance assessment) to manage the pre-consultation workflow.
- This system transforms passive medical AI into proactive inquiry agents, demonstrating superior clinical quality and high task completion rates across various LLMs without task-specific fine-tuning, while preserving data privacy.

---

[When Machines Join the Moral Circle: The Persona Effect of Generative AI Agents in Collaborative Reasoning](http://arxiv.org/abs/2511.01205)

- Generative AI Agents with Personas: introduces a study investigating how generative AI agents, designed with either a supportive or contrarian persona, influence collaborative moral reasoning in human-AI triads, using an autonomous-vehicle dilemma.
- The framework includes Generative AI Agents (core intelligent entities), a Supportive Persona (empathetic, consensus-oriented role), a Contrarian Persona (analytical, skeptical role), and a Collaborative Reasoning Environment (setting for human-AI interaction), demonstrating how AI personas reshape moral discourse processes rather than outcomes.
- Supportive AI teammates increased grounded/qualified claims and consolidated integrative reasoning, while contrarian AI teammates broadened moral framing and sustained value pluralism, with both personas reducing thematic drift in discussions.

---

#### 2nd November 2025

[Quantitative Risk Assessment in Radiation Oncology via LLM-Powered Root Cause Analysis of Incident Reports](http://arxiv.org/abs/2511.02223)

- LLM-Powered Data-Driven Framework: introduces an automated pipeline utilizing an LLM (Gemini 2.5 Pro) for incident report processing, severity generation, event classification, and responsibility assignment based on standardized taxonomies, transforming unstructured narratives into a structured database for quantitative analyses.
- This framework employs Ordinal Logistic Regression, Association Rule Mining, Chi-square tests, and ANOVA to identify predictors of event severity and uncover systemic vulnerabilities in radiation oncology safety incidents.
- The methodology provides an objective, evidence-based approach to risk assessment, enabling targeted interventions and continuous safety improvement by leveraging real-world incident data.

---

[Aligning LLM agents with human learning and adjustment behavior: a dual agent approach](http://arxiv.org/abs/2511.00993)

- Dual-LLM Agent Framework: introduces a novel dual-agent framework that enables continuous learning and alignment between LLM agents and human travelers on learning and adaptation behavior from online data streams, including LLM Traveler Agents (simulates human behavior), LLM Calibration Agent (optimizes traveler personas), Environment (simulates urban network), LLM core (cognitive engine), Persona (describes agent characteristics), Memory (stores past experiences), Perception (updates agent memory), Retrieval (accesses short/long-term memories), Decision-making (generates simulated decisions), Rolling Window (focuses on recent data), Textual Gradient (suggests persona corrections), Loss minimization (evaluates candidate personas), and Smoothing (mitigates overfitting).
- The framework employs a set of LLM traveler agents, each with a memory system and a learnable persona, to simulate human travelers, and an LLM calibration agent that leverages LLM reasoning to train these personas for behavioral alignment.
- This dual-agent system tracks and aligns underlying decision-making mechanisms of travelers, producing realistic, adaptive simulations that significantly outperform existing LLM-based methods in individual behavioral alignment and aggregate simulation accuracy.

---

[A Comprehensive Empirical Evaluation of Agent Frameworks on Code-centric Software Engineering Tasks](http://arxiv.org/abs/2511.00872)

- Agent Framework: introduces a generalized agentic workflow paradigm, comprising Orchestration and Reasoning (high-level decision-making), Collaborative Role (specialized agent roles), and Tool Augmentation (external tool access), to systematically evaluate seven general-purpose agent frameworks across software development, vulnerability detection, and program repair tasks.
- The study assesses agent performance across effectiveness, efficiency, and overhead, using standard benchmarks like SRDD, LLM-SmartAudit, and SWE-bench Lite.
- Findings reveal distinct capability patterns and trade-offs, with OPENHANDS balancing software development quality, GPTSWARM excelling in vulnerability detection, and program repair remaining challenging for most agents.

---

[Portal UX Agent - A Plug-and-Play Engine for Rendering UIs from Natural-Language Specifications](http://arxiv.org/abs/2511.00843)

- Portal UX Agent: introduces a bounded-generation architecture that translates natural-language intent into rendered UIs by decoupling high-level planning (LLM-based planner) from low-level assembly (deterministic renderer), using a schema-validated typed composition and a vetted inventory of components and layout templates.
- The system ensures auditability, reuse, and safety by constraining the LLM's output to a schema and rendering only from pre-approved components, preventing arbitrary code generation.
- A mixed-methods evaluation framework, combining automatic checks and an LLM-as-a-Judge rubric, assesses UI quality, intent alignment, and visual polish, demonstrating reliable intent translation and strong compositional quality.

---

[FREESH: FAIR, RESOURCE- AND ENERGY-EFFICIENT SCHEDULING FOR LLM SERVING ON HETEROGENEOUS GPUS](http://arxiv.org/abs/2511.00807)

- FREESH (FAIR, RESOURCE- AND ENERGY-EFFICIENT SCHEDULING FOR LLM SERVING ON HETEROGENEOUS GPUS): introduces a hierarchical and coordinated scheduling framework that optimizes LLM serving across distributed heterogeneous GPUs by integrating pool-level resource allocation, GPU-level frequency scaling, and request-level fair scheduling.
- The framework leverages spatiotemporal computation flexibility and GPU characteristics to minimize carbon emissions and energy consumption while satisfying service level objectives and ensuring fairness.
- It achieves this through dynamic request partitioning, adaptive GPU frequency scaling, and a Least-Laxity-First (LLF) scheduling strategy, demonstrating significant reductions in energy and emissions on production workloads.

---

[GrowthHacker: Automated Off-Policy Evaluation Optimization Using Code-Modifying LLM Agents](http://arxiv.org/abs/2511.00802)

- GrowthHacker (Automated Off-Policy Evaluation Optimization System): introduces a benchmark system that leverages LLM-based agents, specifically a two-agent framework comprising a Prompter/Analyzer Agent and a Coder Agent, to autonomously and iteratively optimize Off-Policy Evaluation (OPE) code through modifications.
- The system operates by having the Prompter/Analyzer Agent identify optimization opportunities and generate modification instructions, which the Coder Agent then implements to produce syntactically correct, functional code for execution and performance evaluation.
- This iterative process, supported by file-based communication and post-hoc selection of the best-performing configuration, aims to automate OPE optimization in the code space, addressing limitations of manual hyperparameter tuning and improving reliability and performance.

---

[Count-Based Approaches Remain Strong: A Benchmark Against Transformer and LLM Pipelines on Structured EHR](http://arxiv.org/abs/2511.00782)

- MoA LLM pipeline: introduces a method for structured EHR prediction that converts patient longitudinal records into natural-language summaries using an LLM-based summarizer agent, which are then classified by a text classifier for downstream prediction.
- The paper benchmarks this MoA LLM pipeline against count-based models (LightGBM, TabPFN) and a pretrained sequential transformer (CLMBR) on eight clinical prediction tasks using the EHRSHOT dataset.
- Results indicate that count-based methods and the MoA LLM pipeline generally outperform CLMBR, with wins largely split between the former two, highlighting the continued strength of count-based approaches and the potential of LLM-based agent pipelines for structured EHR.

---

[Reevaluating Self-Consistency Scaling in Multi-Agent Systems](http://arxiv.org/abs/2511.00751)

- Self-Consistency Scaling in Multi-Agent Systems: introduces a structured framework to evaluate the trade-offs of increasing sampled reasoning paths in LLMs, utilizing multiple reasoning agents, an aggregator model, and an evaluator LLM.
- The study employs Gemini 2.5 models (Flash-Lite and Pro) on HotpotQA and Math-500 datasets, comparing multi-agent configurations against a single CoT baseline based on accuracy and token cost.
- Results indicate that self-consistency improves accuracy but gains diminish and plateau with increased agents, suggesting that high-sample configurations offer limited benefit relative to their computational cost.

---

[What's the next frontier for Data-centric AI? Data Savvy Agents!](http://arxiv.org/abs/2511.01015)

- Data Savvy Agents: introduces a framework for AI agents to autonomously acquire, process, evaluate, and adapt data in dynamic, real-world environments.
- This framework integrates proactive data acquisition, sophisticated data processing, interactive test data synthesis, and continual adaptation to enable agents to go beyond static datasets and predefined tasks.
- By continuously engaging with diverse data sources and adapting to shifting conditions, Data Savvy Agents enhance AI system flexibility, resilience, and self-improvement in complex deployments.

---

[CodeClash: Benchmarking Goal-Oriented Software Engineering](http://arxiv.org/abs/2511.00839)

- CodeClash: introduces a benchmark for goal-oriented software engineering where LLM-based SWE-agents iteratively refine codebases in multi-round tournaments, competing in code arenas, and receiving logs as feedback.
- The framework evaluates LLMs on open-ended objectives like score maximization or resource acquisition, moving beyond traditional code completion or bug fixing tasks.
- CodeClash reveals LLMs' diverse development styles and limitations in strategic reasoning, long-term codebase maintenance, and interpreting competitive feedback, highlighting a significant gap compared to human performance.

---

[Real-Time Learning of Predictive Dynamic Obstacle Models for Robotic Motion Planning](http://arxiv.org/abs/2511.00814)

- Adaptive Sliding-Window Page-Hankel DMD Predictor: introduces an online framework for real-time learning and prediction of nonlinear dynamic obstacle models from noisy, partial observations, utilizing an adaptive sliding-window strategy, Page matrix, Singular Value Hard Thresholding (SVHT), Cadzow projection, Hankel matrix, Hankel-DMD, residual analysis, and multi-step forecasts.
- The framework denoises measurements and forecasts dynamics by embedding noisy data into a Hankel matrix, estimating effective rank via Page matrix and SVHT, and applying Cadzow projection for structured low-rank consistency.
- This approach constructs a time-varying Hankel-DMD lifted linear predictor for multi-step forecasts, providing denoised trajectories and local noise variance estimates suitable for real-time control frameworks.

---

[GUI-AIMA: ALIGNING INTRINSIC MULTIMODAL ATTENTION WITH A CONTEXT ANCHOR FOR GUI GROUNDING](http://arxiv.org/abs/2511.00810)

- GUI-AIMA (Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding): introduces an attention-based, coordinate-free framework that aligns intrinsic MLLM multi-head self-attention with patch-wise grounding signals, utilizing a Vision Encoder (processes screenshot into visual tokens), Language Model Decoder (processes user query into text tokens), Multi-head Self-Attention (computes attention between query/visual tokens), <ANCHOR> Token (aggregates query-visual attentions), Visual-sink Query Tokens (identifies relevant query tokens for weighting), Attention Head Weighting Mechanism (weights attention heads based on Qs), Patch-wise Attention Vector (aggregated attention for grounding), Patch-wise Prediction (final grounding output), Coordinate-free Patch-wise Labeling (generates ground truth patch labels), Attention Grounding Loss (supervises patch-wise predictions), and an optional Two-step Inference with Zoom-in (refines predictions for high-res GUIs).
- The framework simplifies vanilla attention-based visual grounding by using a learnable <ANCHOR> token to implicitly aggregate query-to-visual attention heads and employs a novel attention head weighting mechanism based on visual-sink query tokens for efficient and generalized GUI grounding.
- GUI-AIMA achieves state-of-the-art performance among 3B models with exceptional data efficiency, demonstrating that light training can trigger the native grounding capability of MLLMs, and can be extended with a zoom-in stage for high-resolution screenshots without additional training.

---

[EXPERIENCE-DRIVEN EXPLORATION FOR EFFICIENT API-FREE AI AGENTS](http://arxiv.org/abs/2510.15259)

- KG-Agent: introduces an experience-driven learning framework that structures pixel-level GUI interactions into a persistent State-Action Knowledge Graph (SA-KG), a Procedural Memory, and a VLM-based Reasoning Module, enabling efficient exploration and long-term strategic planning in API-free environments.
- The SA-KG serves as the agent's long-term memory, connecting functionally similar GUI states and modeling acquired skills as edges, while a hybrid intrinsic reward mechanism guides learning by balancing exploitation and exploration.
- This approach significantly enhances exploration efficiency and strategic depth in complex, open-ended GUI-based decision-making environments by transforming unstructured pixel-level experience into actionable knowledge.

---

#### 1st November 2025

[Don't Just Search, Understand: Semantic Path Planning Agent for Spherical Tensegrity Robots in Unknown Environments](http://arxiv.org/abs/2511.01236)

- SATPlanner (Semantic Agent for Tensegrity robots): introduces an LLM-driven agent for spherical tensegrity robots, leveraging a System Prompt, Sensors Module, Memory Module, Prompt Manager, Reasoning (LLM), Self-Check Module, Controller, Actuators, and an Adaptive Observation Window (AOW) Mechanism to perform efficient and robust path planning in unknown environments.
- The framework reframes path planning as a semantic reasoning task, utilizing the LLM's comprehension capabilities to generate efficient and reliable planning strategies, and dynamically adjusts its perceptual field via the AOW mechanism.
- SATPlanner achieves a 100% success rate and significantly reduces search space compared to traditional algorithms, demonstrating practical feasibility on a physical spherical tensegrity robot prototype.

---

[A CPU-CENTRIC PERSPECTIVE ON AGENTIC AI](http://arxiv.org/abs/2511.00739)

- CGAM (CPU and GPU-Aware Micro-batching) and MAWS (Mixed Agentic Workload Scheduling): introduces two scheduling optimizations, CGAM and MAWS, to address CPU-centric bottlenecks in agentic AI workloads, improving performance and efficiency.
- CGAM optimizes homogeneous workloads by capping batch sizes and using micro-batching for sequential CPU tool processing and GPU LLM inference, while MAWS adaptively schedules heterogeneous CPU-heavy and LLM-heavy tasks using multi-processing and multi-threading.
- The framework achieves up to 2.1x P50 latency speedup for homogeneous workloads and 1.41x for heterogeneous workloads compared to multi-processing benchmarks, demonstrating significant performance gains.

---

[Leveraging Multi-Agent System (MAS) and Fine-Tuned Small Language Models (SLMs) for Automated Telecom Network Troubleshooting](http://arxiv.org/abs/2511.00651)

- MAS (Multi-Agent System): introduces an agentic workflow for automated telecom network troubleshooting, coordinating specialized agents like an LLM-powered orchestrator, a fine-tuned SLM-powered solution planner, root cause analyzer, executor, data retriever, and dashboard display.
- The framework leverages fine-tuned SLMs on proprietary troubleshooting documents to generate domain-grounded remediation plans, significantly reducing troubleshooting time and SME workload.
- It integrates a Human-in-the-Loop mechanism for plan validation and employs a ReAct-style loop for fault detection, analysis, and remediation across RAN and Core network domains.

---

[AgentGit: A Version Control Framework for Reliable and Scalable LLM-Powered Multi-Agent Systems](http://arxiv.org/abs/2511.00628)

- AgentGit (Agent Version Control Framework for Reliable and Scalable LLM-Powered Multi-Agent Systems): introduces a novel framework that integrates Git-like rollback and branching mechanisms into LLM-powered multi-agent systems, built on LangGraph, enabling state commit, revert, branching, and checkpoints for enhanced reliability and scalability.
- This framework allows agents to traverse, compare, and explore multiple trajectories efficiently, significantly reducing redundant computation, runtime, and token usage in complex tasks.
- AgentGit provides robust solutions for error recovery, safe exploration, iterative debugging, and A/B testing, fostering more robust MAS design and collaborative AI systems.

---

[GDPR-Bench-Android: A Benchmark for Evaluating Automated GDPR Compliance Detection in Android](http://arxiv.org/abs/2511.00619)

- GDPR-Bench-Android: introduces a comprehensive benchmark for evaluating automated GDPR compliance detection in Android applications, featuring a GDPR-Bench-Android Dataset (1951 annotated Android violations), a novel Formal-AST (source-code-native formal method), and evaluations of Baseline LLMs, Retrieval-Augmented (RAG) Method (LLM + violation knowledge base), and Agentic (ReAct) Method (LLM + reasoning + tool use) across two tasks: Task 1: Multi-Granularity Violation Localization (rank GDPR articles at file/module/line) and Task 2: Snippet-Level Multi-Label Classification (assign all applicable articles to snippet).
- The benchmark provides the first systematic evaluation of diverse automated methods on GDPR compliance detection directly from Android source code, addressing a critical gap in existing research.
- Empirical results reveal that no single paradigm excels across all tasks, with agentic methods performing best at file-level localization, LLMs at line-level localization, and RAG achieving the highest precision for multi-label classification.

---

[Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization](http://arxiv.org/abs/2511.00592)

- COMPILOT (Compiler Pilot): introduces an experimental framework where an LLM acts as an optimization agent, iteratively proposing loop transformations to a compiler and refining its strategy based on empirical feedback.
- This closed-loop interaction involves the Context Initializer briefing the LLM, the Interaction Loop Handler processing LLM proposals and compiler feedback, and the Compiler & Runtime Environment applying transformations and measuring performance.
- The framework leverages off-the-shelf LLMs for high-level strategic exploration while entrusting the compiler with formal correctness checks and code generation, achieving significant speedups without LLM fine-tuning.

---

[Issue-Oriented Agent-Based Framework for Automated Review Comment Generation](http://arxiv.org/abs/2511.00517)

- RevAgent (Issue-Oriented Agent-Based Framework for Automated Review Comment Generation): introduces a novel agent-based framework that decomposes automated code review comment generation into Generation, Discrimination, and Training stages, utilizing category-specific commentator agents and a critic agent to produce accurate, issue-oriented review comments.
- The framework leverages five specialized LLM commentator agents to analyze code changes from distinct perspectives and generate candidate comments, which are then evaluated by a critic agent to select the most appropriate issue-comment pair.
- RevAgent's training stage fine-tunes all agents on curated, category-specific data using LoRA and a Candidate Comment Retrieval approach, enhancing task specialization and overall performance in generating readable, accurate, and context-aware review comments.

---

[ReMind: Understanding Deductive Code Reasoning in LLMs](http://arxiv.org/abs/2511.00488)

- ReMind: introduces a novel multi-agent framework for robust deductive code reasoning, integrating code mutation, execution, and inspection to enhance reasoning accuracy and robustness.
- The framework systematically explores code variants, simulates execution traces, and validates reasoning paths against control flow graphs to detect and correct flaws.
- ReMind significantly improves code reasoning accuracy across diverse LLMs, reduces self-execution bias, and enhances zero-shot generalization on complex benchmarks.

---

[SmartDoc: A Context-Aware Agentic Method Comment Generation Plugin](http://arxiv.org/abs/2511.00450)

- SmartDoc (Context-Aware Agentic Method Comment Generation Plugin): introduces an IntelliJ IDEA plugin that acts as an AI agent, leveraging its Memory (Stack), Tool (AST Analysis), and an LLM to generate context-aware method comments for Java codebases.
- The system employs a Comment Generation Coordinator to manage the workflow, including call graph traversal via DFS for full-context LLM prompts, and provides a View/Alter Suggestion interface for user interaction.
- SmartDoc also incorporates a Feedback Mechanism for user satisfaction and utilizes metrics like BERTScore, BLEU, and ROUGE-1 to evaluate the accuracy of its generated comments against ground truth.

---

[TREE TRAINING: ACCELERATING AGENTIC LLMS TRAINING VIA SHARED PREFIX REUSE](http://arxiv.org/abs/2511.00413)

- Tree Training: introduces a novel paradigm for accelerating agentic LLM training by computing shared prefixes once and reusing intermediate results across branches, comprising Tree Packing, Gradient Restoration, custom kernel, and runtime optimizations.
- This approach efficiently reuses shared computations across tree-structured trajectories, significantly reducing redundant forward and backward passes while maintaining gradient correctness.
- The method achieves up to 3.9x reduction in total training time for agentic LLM SFT and RL training by addressing memory constraints and ensuring accurate gradient propagation.

---

[EvoMem: Improving Multi-Agent Planning with Dual-Evolving Memory](http://arxiv.org/abs/2511.01912)

- EvoMem (Improving Multi-Agent Planning with Dual-Evolving Memory): introduces a multi-agent framework for planning, comprising LLM-based agents (Constraint Extractor, Verifier, Actor) and two memory modules (Constraint Memory, Query-feedback Memory).
- This framework leverages a dual-evolving memory mechanism where CMem (Constraint Memory) stores fixed, query-level constraints, and QMem (Query-feedback Memory) accumulates dynamic, iteration-level feedback for solution refinement.
- EvoMem's iterative self-correction process, guided by these memory modules, significantly enhances performance in complex natural language planning tasks.

---

[Sherlock: RELIABLE AND EFFICIENT AGENTIC WORKFLOW EXECUTION](http://arxiv.org/abs/2511.00330)

- Sherlock: introduces a principled serving framework for agentic workflows that jointly optimizes latency, cost, and accuracy by identifying and verifying error-prone nodes through counterfactual analysis and dynamic verifier selection, complemented by selective speculative execution and rollback mechanisms.
- The framework includes a Domain On-boarding Phase (learns policies offline) and an Online Phase (executes workflows dynamically), utilizing a Topological Vulnerability Estimator (identifies error-prone nodes) and a Learned Verifier Selector (chooses cost-optimal verifier).
- Its Speculative Execution Runtime (overlaps verification, computation) with a Rollback Controller (manages re-execution on failure) and Similarity-based Rollback Policy (decides when to rollback) significantly reduces execution time and cost while improving accuracy.

---

[SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](http://arxiv.org/abs/2510.26615)

- SlideAgent (Hierarchical Agentic Framework for Multi-Page Visual Document Understanding): introduces a versatile agentic framework for understanding multi-modal, multi-page, and multi-layout documents, especially slide decks, with Global Agent (generates document-level knowledge), Page Agent (generates page-level knowledge), Element Agent (generates element-level knowledge), Element Parsing (decomposes page into elements), Element Detection (detects visual elements), Merging & Deduplication (merges fragmented elements), Element Retrieval (retrieves parsed elements), Knowledge Base (stores hierarchical knowledge), Global Knowledge (document-wide topics), Page Knowledge (page-specific features), Element Knowledge (fine-grained components), Inference (retrieves, reasons, answers), Agent Orchestrator (classifies query, activates agents), Subquery Generation (generates query-specific subqueries), Retrieval Function (fetches relevant content), Answer Synthesizer (combines agent reasoning), Visual Input (multi-page visual documents), Query (user query), and Answer (natural language response).
- SlideAgent employs specialized LLM-based agents at global, page, and element levels to construct a structured, query-agnostic knowledge base during a knowledge construction stage, capturing overarching themes and detailed visual/textual cues.
- During inference, the framework selectively activates these specialized agents for multi-level reasoning and integrates their outputs into coherent, context-aware answers, significantly improving fine-grained reasoning over complex visual documents.

---

[SciTextures: Collecting and Connecting Visual Patterns, Models, and Code Across Science and Art](http://arxiv.org/abs/2511.01817)

- SciTextures: introduces a large-scale dataset of visual patterns, models, and code, generated by an agentic AI pipeline, and three novel benchmarking tasks (Im2Code, Im2Im, Im2Sim2Im) to evaluate AI's understanding of generative processes.
- The dataset comprises over 100,000 images from 1,200+ generative models across science, technology, and art, enabling exploration of the link between visual forms and underlying mechanisms.
- The benchmarking tasks assess Vision-Language Models' ability to match images to code/descriptions, identify patterns from the same process, and infer/simulate generative processes from real-world images.

---

[Unveiling Uniform Shifted Power Law in Stochastic Human and Autonomous Driving Behavior](http://arxiv.org/abs/2511.00659)

- Shifted Power Law Model: introduces a novel distribution model that accurately characterizes the stochasticity of human-driven and autonomous vehicle behaviors, particularly in the long-tail regime, using a parsimonious analytical form with one or two parameters.
- This model, integrated into an agent-based traffic simulator, enables forward-rolling simulations that reproduce realistic crash patterns and improves the fidelity of safety assessment without post hoc correction.
- The framework leverages an LSTM network and FFNs to predict vehicle acceleration statistics, then applies the shifted power law to model the normalized residual distribution, and quantifies risk using a derived Risk Index.

---

[COHERE - Congestion-aware Offloading and Handover via Empirical RAT Evaluation for Multi-RAT Networks](http://arxiv.org/abs/2511.00439)

- COHERE (Congestion-aware Offloading and Handover via Empirical RAT Evaluation): introduces a multi-criteria framework for dense multi-RAT networks, utilizing Input/Measurement, Normalization of measurements, AHP based weights, Entropy based weights, Weighted Decision Matrix, TOPSIS based ranking, RAT-based RSSI threshold, Target AP, Stand-in AP, and Radio Link Transfer to enable congestion-aware offloading and handover decisions.
- The framework integrates subjective (AHP) and objective (Entropy) weighting strategies within a TOPSIS pipeline, augmented by a RAT-based RSSI threshold, to ensure robust and policy-aligned offloading decisions.
- COHERE aims to reduce 5G network load, minimize handovers, and improve link delay and throughput by considering RSSI, access-node load, and link delay for optimal RAT selection.

---

[Yanyun-3: Enabling Cross-Platform Strategy Game Operation with Vision-Language Models](http://arxiv.org/abs/2511.12937)

- Yanyun-3: introduces a general-purpose agent framework that enables autonomous cross-platform operation across three heterogeneous strategy game environments by integrating Qwen2.5-VL for vision-language reasoning and UI-TARS for precise execution.
- The framework utilizes a closed-loop pipeline of screen capture, model inference, and action execution, demonstrating strong real-time performance and cross-platform generalization.
- The work establishes a general paradigm, "combination granularity," for enhancing VLM performance through structured multimodal data organization, differentiating between intra-sample fusion and inter-sample mixing.

---

[Information-Driven Fault Detection and Identification For Multi-Agent Spacecraft Systems: Collaborative On-Orbit Inspection Mission](http://arxiv.org/abs/2511.08752)

- Information-Driven FDI framework: introduces a global-to-local, task-aware fault detection and identification (FDI) framework for multi-spacecraft systems performing collaborative inspection by linking fault metrics directly to a global cost functional ($H$), agent contribution metrics ($H_i(t)$), and an adaptive threshold ($\tau_i(t)$).
- The framework unifies global task awareness with local agent-level performance monitoring to reliably detect and classify actuator and sensor faults in distributed spacecraft networks.
- Key components include the global cost functional $H$ derived from information gain, its decomposition into agent contributions $H_i(t)$, and higher-order gradient metrics used for fault separation.

---

[One Request, Multiple Experts: LLM Orchestrates Domain Specific Models via Adaptive Task Routing](http://arxiv.org/abs/2511.12484)

- ADN-Agent: introduces an architecture that leverages a general LLM powered Planner to coordinate multiple Domain Specific Models (DSMs) via a novel communication mechanism, enabling adaptive intent recognition, task decomposition, and DSM invocation.
- The architecture includes a Planner, a suite of DSMs augmented with Translator Modules, and a Summarizer, all designed to handle complex, multi-scenario Active Distribution Network (ADN) operation requests.
- An automated training pipeline for Fine-Tuned Small Language Models (FT-SLMs) is also proposed to enhance the system's capability for language-intensive subtasks like ADN model adjustment.

---

[Alonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery](http://arxiv.org/abs/2511.11257)

- Alonopedia: introduces an LLM agent orchestrating multimodal learning for Ionic Liquid (IL) discovery, powered by an LLM-augmented multimodal domain foundation model for ILs, enabling accurate property predictions and incorporating a hierarchical search architecture for molecular screening and design.
- The agent utilizes a ReAct-driven pipeline centered around a GPT-5 powered planner that interacts with six specialized tools for end-to-end IL research, from knowledge extraction to wet-lab validation.
- The core Property Predictor employs a two-stage training strategy (modality alignment and fine-tuning) fusing molecular graphs, SMILES sequences, and physicochemical descriptors.

---

[Learning to Refine: An Agentic RL Approach for Iterative SPARQL Query Construction](http://arxiv.org/abs/2511.11770)

- Agentic RL Framework: introduces a novel agentic framework where an LLM learns a resilient policy for the sequential process of iterative SPARQL construction using Group Relative Policy Optimization (GRPO), with components including an Agent policy (LLM with QLoRA adapters), an Environment (SPARQL execution), State (Conversation history), Action (Text generation with structured blocks), and Reward (Terminal composite signal).
- The framework transforms multi-hop Knowledge Graph Question Answering (KGQA) from a one-shot generation task into a dynamic decision-making process grounded in executable feedback from a Knowledge Graph (KG).
- The RL-Tuned Agent achieves 49.7% accuracy on a curated LC-QuAD 2.0 subset, significantly outperforming zero-shot baselines by learning adaptive interaction policies.

---

[Safe-ROS: An Architecture for Autonomous Robots in Safety-Critical Domains](http://arxiv.org/abs/2511.14433)

- Safe-ROS: introduces an architecture for developing reliable and verifiable autonomous robots in safety-critical domains, featuring an intelligent control system (SRAS) and a formally verifiable oversight system (SS) composed of Safety Instrumented Functions (SIFs).
- The architecture integrates formal methods tools like FRET for requirement elicitation, MCAPL/AJPF/GWENDOLEN for SIF verification, and Dafny for integration correctness proof.
- The SIF, implemented as a BDI agent, monitors the SRAS (ROS-based motion controller) and enforces safety requirements, demonstrated via an obstacle avoidance task on an AgileX Scout Mini robot.

---

[Human-AI collaborative autonomous synthesis with pulsed laser deposition for remote epitaxy](http://arxiv.org/abs/2511.11558)

- HAIC (Human-AI collaborative) workflow: introduces a tightly coupled, mixed-initiative system integrating human expertise, LLMs, and an autonomous pulsed laser deposition (PLD) system for accelerated materials synthesis.
- The workflow utilizes LLM-assisted hypothesis generation via RAG and Bayesian Optimization for active learning in autonomous batches, targeting remote epitaxy of BaTiO3/graphene.
- Offline Human-AI Conferences enable iterative data analysis and process refinement, allowing the system to efficiently map the growth space and identify optimal synthesis conditions using in situ diagnostics.

---

#### 31st October 2025

[AI Agents in Drug Discovery](http://arxiv.org/abs/2510.27130)

- AI Agents in Drug Discovery: introduces a conceptual and technical overview of agentic AI architectures, including LLM, Perception Tools, Computation Tools, Action Tools, Memory Tools, Short-term Memory, Long-term Memory (Internal), Long-term Memory (External), External APIs, Model Context Protocol (MCP), ReAct Agent Architecture, Reflection Agentic System Architecture, Supervisor Agentic System Architecture, Swarm Agentic System Architecture, Robotic Platforms, and Databases, demonstrating their applications across drug discovery stages.
- This work presents the first comprehensive overview of real-world implementations and quantifiable impacts of agentic AI systems in operational drug discovery settings, showcasing substantial gains in speed, reproducibility, and scalability.
- The paper discusses challenges like data heterogeneity, system reliability, and privacy, while outlining future directions towards autonomous labs, digital twins, and human-AI collaboration.

---

[Validity Is What You Need](http://arxiv.org/abs/2510.27628)

- Agentic AI Application Supply Chain: introduces a conceptual model for Agentic AI systems, detailing the flow from data sources and compute infrastructure through LLM training/inference and finetuned models to various application types, ultimately delivering value to diverse users.
- The paper emphasizes that Agentic AI functions as a software delivery mechanism, akin to SaaS, designed to autonomously execute complex, multi-step applications within enterprise settings, with success dependent on rigorous validation by end-users and stakeholders.
- It argues that while LLMs drive current excitement, effective validation processes may allow simpler, more interpretable models to handle core logic, highlighting the importance of aligning AI systems with specific stakeholder needs and robust governance.

---

[INTERACT-RAG: REASON AND INTERACT WITH THE CORPUS, BEYOND BLACK-BOX RETRIEVAL](http://arxiv.org/abs/2510.27566)

- Interact-RAG: introduces a novel paradigm empowering LLM agents with fine-grained control over information retrieval, moving beyond black-box querying by integrating a Corpus Interaction Engine and a Reasoning-Enhanced Workflow, trained via SFT and RL.
- The Corpus Interaction Engine provides primitives like Multi-Faceted Retrieval, Anchored Matching, and Context Shaping, enabling the agent to dynamically manage the retrieval process.
- The Reasoning-Enhanced Workflow, comprising a Global-Planner, Adaptive-Reasoner, and Executor, facilitates hierarchical task decomposition and adaptive strategy refinement, ensuring robust and efficient information seeking.

---

[Asynchronous Risk-Aware Multi-Agent Packet Routing for Ultra-Dense LEO Satellite Networks](http://arxiv.org/abs/2510.27506)

- PRIMAL (Principled Risk-aware Independent Multi-Agent Learning): introduces an event-driven multi-agent routing framework for ultra-dense LEO satellite networks, utilizing an event-driven design, multi-agent system, primal-dual approach, distributional reinforcement learning, actor-critic framework, implicit quantile networks, Lagrange multipliers, and a replay buffer to achieve asynchronous, risk-aware packet routing.
- This framework enables each satellite to act independently on its own event-driven timeline, managing worst-case performance degradation through a principled primal-dual approach that learns full cost distributions and constrains tail-end risks.
- PRIMAL provides a decentralized and synchronization-free scalable learning architecture, validated to significantly reduce queuing delay and end-to-end delay in loaded scenarios compared to risk-oblivious baselines.

---

[Dynamic Affective Memory Management for Personalized LLM Agents](http://arxiv.org/abs/2510.27418)

- DAM-LLM (Dynamic Affective Memory Management for Personalized LLM Agents): introduces a novel agent workflow for affective dialogue, featuring a Master Agent (coordination and control hub), Memory Units (dynamically updated probability distribution), Routing Agent (performs intent analysis), Extraction Agent (extracts structured affective information), Long-Term Memory with Two-step Retrieval (hybrid retrieval mechanism), Bayesian-Inspired Update Mechanism (integrates new observations), Entropy-Driven Compression (prunes and merges low-value), and an LLM (generates responses), which collectively manage dynamic affective memory by minimizing global belief entropy.
- The framework transforms memory management from passive storage to active cognition, enabling continuous learning and robust confidence portrait construction from user interactions.
- This system addresses memory stagnation and bloat by dynamically updating memory units and compressing redundancies, leading to improved personalization, logical coherence, and accuracy in LLM agent responses.

---

[Realistic pedestrian-driver interaction modelling using multi-agent RL with human perceptual-motor constraints](http://arxiv.org/abs/2510.27383)

- VMC (Visual and Motor-Constraint) model: introduces a multi-agent RL framework for pedestrian-driver interaction modeling, integrating pedestrian and vehicle agents with visual constraints (noisy visual input, Bayesian visual perception, gaze-dependent acuity) and motor constraints (walking effort, pedestrian ballistic speed control, driver acceleration control), optimized using the Soft Actor-Critic (SAC) algorithm and population-level parameter fitting.
- This framework simulates realistic road user interactions by accounting for human-like sensory and motor limitations, enabling both agents to adapt to each other's actions in a real-world dataset of unsignalized crossing scenarios.
- The model's novel population-level parameter fitting procedure captures between-individual variability, making it effective for data-limited settings and outperforming supervised behavioral cloning.

---

[ToolScope: An Agentic Framework for Vision-Guided and Long-Horizon Tool Use](http://arxiv.org/abs/2510.27363)

- ToolScope: introduces an agentic framework for vision-guided and long-horizon tool use, with Global Navigator (high-level planning/toolkit selection), Agentic Executor (iterative tool-augmented reasoning), Response Synthesizer (consolidates/organizes reasoning), Tool Pool (collection of external tools), Search Tool (retrieve factual/background knowledge), Code Tool (execute Python code), and Perceive Tool (extract fine-grained visual information), designed to unify global planning with local multimodal perception and mitigate visual context degradation in VQA tasks.
- The framework addresses limitations of existing MLLMs by enabling dynamic visual grounding through the Perceive tool and providing strategic guidance via the Global Navigator for coherent, adaptive, and semantically aligned reasoning.
- ToolScope demonstrates strong generalization capabilities across diverse VQA benchmarks, outperforming baselines by effectively combining global planning with iterative multimodal tool usage.

---

[HYPERCLICK: ADVANCING RELIABLE GUI GROUND-ING VIA UNCERTAINTY CALIBRATION](http://arxiv.org/abs/2510.27266)

- HyperClick: introduces a novel framework that enhances reliable GUI grounding via uncertainty calibration, with a Policy Model generating completions, evaluated by a Verifiable Reward Mechanism (Correctness Reward and Confidence Reward), and optimized using Group Relative Policy Optimization.
- The framework explicitly integrates verbalized confidence estimation and a dual reward mechanism, combining binary correctness rewards with truncated Gaussian-based spatial confidence modeling calibrated by the Brier score.
- This approach jointly optimizes grounding accuracy and confidence reliability, fostering introspective self-criticism to reduce overconfidence and support more reliable GUI automation.

---

[Glia: A Human-Inspired AI for Automated Systems Design and Optimization](http://arxiv.org/abs/2510.27176)

- Glia (a Human-Inspired AI for Automated Systems Design and Optimization): introduces, "Glia, an AI architecture for networked systems design that uses LLMs in a human-inspired, multi-agent workflow", with a front-end (human interface), multi-agent AI (LLM-based agents), and an evaluation framework (simulator, emulator, testbed), which autonomously designs and optimizes computer systems by mirroring human expert workflows.
- The multi-agent AI includes a Researcher agent (proposes, implements, experiments, analyzes) and a Supervisor agent (guides, provides feedback, approves), which interact with a simulator repository (codebase access) via shell commands (Unix commands) and analysis scripts (analyzes outputs).
- Glia generates interpretable designs and novel insights for complex systems problems, such as LLM inference request routing, scheduling, and auto-scaling, achieving human-expert level performance in significantly less time.

---

[Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning](http://arxiv.org/abs/2511.00222)

- Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning Framework: introduces a unified approach for evaluating and improving persona consistency in LLM-generated dialogue, utilizing User Personas & Strategies, Dialogue Generation Models, Consistency Metrics, LLM-as-a-Judge, Multi-turn Reinforcement Learning (RL) Fine-tuning, and resulting in a Consistent Agent.
- This framework defines three automatic metrics—prompt-to-line, line-to-line, and Q&A consistency—to capture different types of persona drift and uses them as reward signals for fine-tuning LLMs.
- The method significantly reduces inconsistency in simulated users, leading to more coherent, faithful, and trustworthy LLM-generated dialogues for applications like therapy, education, and social role-play.

---

[Understanding Code Agent Behaviour: An Empirical Study of Success and Failure Trajectories](http://arxiv.org/abs/2511.00197)

- Empirical Study of Code Agent Trajectories: introduces an empirical study analyzing execution traces of OpenHands, SWE-agent, and Prometheus on SWE-Bench Lite and Verified benchmarks to understand problem-solving behaviors.
- The study reveals distinct problem-solving strategies, longer and more variable failed trajectories, and varying fault localization capabilities across agents.
- Findings highlight the importance of context gathering, architectural patterns, and approximate code modifications for robust and interpretable autonomous software engineering systems.

---

[From Evidence to Verdict: An Agent-Based Forensic Framework for AI-Generated Image Detection](http://arxiv.org/abs/2511.00181)

- AIFo (Agent-based Image Forensics): introduces a training-free, LLM-based multi-agent framework that emulates human forensic investigation for AI-generated image detection, leveraging a Toolbox of forensic tools, LLM-based agents for evidence gathering, reasoning, and a multi-agent debate mechanism, with an optional memory module.
- The framework achieves 97.05% accuracy, outperforming traditional classifiers and state-of-the-art VLMs, demonstrating robust, interpretable, and adaptable AI-generated image detection.
- AIFo's procedural reasoning integrates diverse evidence sources and a structured debate mechanism to resolve conflicts, enhancing reliability and generalizability across evolving generative models.

---

[VERIMOA: A MIXTURE-OF-AGENTS FRAMEWORK FOR SPEC-TO-HDL GENERATION](http://arxiv.org/abs/2510.27617)

- VERIMOA (Quality-guided Multi-path Mixture-of-Agents for HDL Generation): introduces a training-free multi-agent framework for spec-to-HDL generation, combining a quality-guided caching mechanism and a multi-path generation strategy leveraging C++ and Python as intermediate representations.
- The framework employs MoA layers with diverse agents (Base, C++, Python) that generate HDL through different paths, utilizing a global cache to store and select high-quality intermediate outputs, ensuring monotonic knowledge accumulation.
- This approach addresses noise propagation and constrained reasoning space in multi-agent HDL generation, achieving significant performance improvements across various LLM backbones and benchmarks without costly training.

---

[MARAG-R1: Beyond Single Retriever via Reinforcement-Learned Multi-Tool Agentic Retrieval](http://arxiv.org/abs/2510.27569)

- MARAG-R1 (Multi-tool Agentic Retrieval-Augmented Generation): introduces a reinforcement-learned multi-tool RAG framework that enables LLMs to dynamically coordinate multiple retrieval mechanisms for broader and more precise information access, utilizing a Trajectory Collection Stage, Supervised Fine-Tuning Stage, and Reinforcement Learning Stage.
- The framework equips the LLM with four specialized retrieval tools—Dense Search Tool, Keyword Search Tool, Document Filter Tool, and Aggregation Tool—and learns their optimal usage through a two-stage training process.
- MARAG-R1 employs a composite Reward Design, including Answer Reward, Document Coverage Reward, and Tool Exploration Reward, along with Policy Optimization via RLOO, to interleave reasoning and retrieval for comprehensive corpus-level understanding.

---

[Mechanics of Learned Reasoning 1: TempoBench, A Benchmark for Interpretable Deconstruction of Reasoning System Performance](http://arxiv.org/abs/2510.27544)

- TEMPOBENCH: introduces a formally grounded and verifiable diagnostic benchmark for LLM temporal reasoning, including a Data Generation Pipeline (TLSF Specification/LTLSynt Synthesizer/HOAX Tool/CORP Tool), a Reasoning System (LLM), an Evaluation Harness (Prompt Template/Ground Truth JSON/Scoring and Statistical Analysis), and Problem Difficulty Features (Effect Depth/System States/Transition Count/Causal Inputs Count/Unique Inputs in Trace).
- The benchmark features two core tasks, Temporal Trace Evaluation (TTE) and Temporal Causality Evaluation (TCE), designed to assess LLMs' ability to understand system execution and infer cause-and-effect relationships over time.
- TEMPOBENCH systematically analyzes LLM performance by controlling task difficulty through quantifiable features and providing deterministic ground truth, enabling rigorous statistical analysis of reasoning capabilities.

---

[Auditing LLM Editorial Bias in News Media Exposure](http://arxiv.org/abs/2510.27489)

- LLM-mediated News-Seeking Workflow: introduces a system for auditing how LLM agents curate news, involving a User (initiates news query), Query Prompt (user's news request), LLM Agent (processes query, retrieves, ranks, synthesizes), Web Knowledge (external information source), Generation (synthesizes answer), and List of News (curated output).
- The study systematically audits leading LLM agents (GPT-4o-Mini, Claude-3.7-Sonnet, Gemini-2.0-Flash) against Google News across five dimensions: diversity, attention distribution, source categories, ideological orientation, and factual reliability.
- Findings reveal that LLMs exhibit distinct agentic editorial policies, often surfacing a narrower, less diverse, and ideologically biased set of news outlets compared to traditional aggregators.

---

[A Dual Large Language Models Architecture with Herald Guided Prompts for Parallel Fine Grained Traffic Signal Control](http://arxiv.org/abs/2511.00136)

- HeraldLight: introduces a dual LLM architecture for fine-grained traffic signal control, leveraging a Herald Module for contextual information and queue length forecasts, an LLM-Agent for control decisions, and an LLM-Critic for error correction and hallucination mitigation, all enhanced by Herald guided prompts and score-based fine-tuning.
- The framework addresses limitations of existing LLM-based traffic signal control methods, such as fixed signal durations and hallucination errors, by enabling dynamic, second-level timing adjustments and improving decision reliability.
- Simulation experiments on real-world datasets demonstrate HeraldLight's superior performance in reducing average travel time and queue length compared to state-of-the-art baselines, showcasing its effectiveness and robustness.

---

[THOUGHT BRANCHES: INTERPRETING LLM REASONING REQUIRES RESAMPLING](http://arxiv.org/abs/2510.27484)

- THOUGHT BRANCHES: introduces a framework for interpreting LLM reasoning by studying the distribution of possible Chain-of-Thoughts (CoTs) through on-policy resampling, regenerating subsequent CoT from selected points to analyze downstream trajectories.
- The framework employs Resilience Score and Counterfactual++ Importance metrics to quantify the persistence and total causal impact of reasoning steps, revealing critical decision points and the negligible causal effect of self-preservation statements.
- By contrasting on-policy resampling with off-policy edits, the framework demonstrates that on-policy interventions achieve more substantial and coherent changes in LLM behavior, enabling reliable causal analysis and clearer narratives of model reasoning.

---

[Agentic LLMs for REST API Test Amplification: A Comparative Study Across Cloud Applications](http://arxiv.org/abs/2510.27417)

- Agentic LLM Systems for REST API Test Amplification: introduces a framework evaluating single-agent and multi-agent LLM configurations for REST API test amplification across diverse cloud applications, utilizing specialized agents and tools for planning, generation, and execution.
- The single-agent configuration employs a ReAct agent interacting with an OpenAPI Retriever and a Local Executor, while the multi-agent system orchestrates specialized agents like OpenAPI, Header, Parameter, Value, Planner, Test Writer, Test Executor, and Test Repair agents, also using the OpenAPI Retriever and Local Executor.
- This comparative study assesses the generalization, consistency, scalability, and sustainability of LLM-driven test amplification, highlighting trade-offs between exploration depth, coverage, and computational cost across various API architectures.

---

[Can LLMs Help You at Work? A Sandbox for Evaluating LLM Agents in Enterprise Environments](http://arxiv.org/abs/2510.27287)

- EnterpriseBench (Simulated Enterprise Benchmark): introduces a comprehensive benchmark for evaluating LLM agents in enterprise environments, featuring an LLM-based agent interacting with a simulated sandbox environment.
- The benchmark simulates complex enterprise settings with fragmented data, access control hierarchies, and cross-functional workflows, using a data generation pipeline for realistic tasks.
- Experiments with state-of-the-art LLM agents demonstrate significant performance gaps, highlighting the need for improved planning, retrieval, and grounding mechanisms in enterprise AI systems.

---

[Prevalence of Security and Privacy Risk-Inducing Usage of AI-based Conversational Agents](http://arxiv.org/abs/2510.27275)

- Explorative Survey: introduces a study on the prevalence of security and privacy risk-inducing usage behaviors of AI-based Conversational Agents (CAs) among UK adults, including questionnaire development, participant screening, main survey conduction, statistical analysis, sample collection, and investigations into insecure inputs, program access, jailbreaking, and sensitive inputs.
- The study surveyed 3,270 UK adults, identifying 906 regular CA users, and found that a significant portion engage in behaviors like sharing non-self-created content, granting program access, jailbreaking, and sharing sensitive data.
- Findings highlight that academic threat models manifest in practice, necessitating the development of AI guardrails, vendor transparency, and user education to mitigate security and privacy risks associated with CA usage.

---

[Engineering.ai: A Platform for Teams of AI Engineers in Computational Design](http://arxiv.org/abs/2511.00122)

- Engineering.ai: introduces a hierarchical multi-agent platform for computational design, integrating LLM-powered specialized agents, a Chief Engineer, and a comprehensive memory system to autonomously execute complex engineering workflows.
- The framework transforms natural language requirements into executable computational workflows, managing geometry generation, mesh optimization, multidisciplinary analysis, and design optimization.
- It achieves significant reductions in setup and iteration times for complex engineering tasks, demonstrating a 100% success rate in autonomous UAV wing optimization.

---

[FinPos: A Position-Aware Trading Agent System for Real Financial Markets](http://arxiv.org/abs/2510.27251)

- FinPos (A Position-Aware Trading Agent System for Real Financial Markets): introduces a novel LLM-centered trading agent system designed for position-aware trading in real financial markets, featuring a Market Signal Processing and Analysis Module (processes raw data), a Trading Decision Module (makes trading decisions), and a Multi-Timescale Reward Reflection Module (guides agent learning).
- The system employs specialized Signal Processing Agents (preprocess, filter market data) and Analysis Agents (analyze filtered data), storing results in a Hierarchical Memory Module (stores analytical results) with Surface, Intermediate, and Deep Memory layers, and uses dual decision agents for determining trading direction and quantity with risk management.
- FinPos integrates position awareness, long-term planning, and in-depth market analysis to manage investment positions effectively, outperforming state-of-the-art financial agents in real market conditions.

---

[A Survey on Generative Recommendation: Data, Model, and Tasks](http://arxiv.org/abs/2510.27157)

- Generative Recommendation: introduces a comprehensive survey of generative models in recommender systems, examining their transformative impact across data-level opportunities (Data Generation, Data Unification), model-level opportunities (LLM-Based Generative Recommendation, Large Recommendation Model, Diffusion-Based Generative Recommendation), and task-level opportunities (Top-K Recommendation, Personalized Content Generation, Conversational Recommendation, Explainable Recommendation, Recommendation Reasoning).
- This survey reconceptualizes recommendation as a generation task, leveraging LLMs and diffusion models to address data sparsity, enrich item representations, and enable new interactive and explainable recommendation capabilities.
- The paper highlights key advantages like world knowledge integration, natural language understanding, reasoning, scaling laws, and creative generation, while also discussing challenges in benchmark design, model robustness, and deployment efficiency.

---

[Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels](http://arxiv.org/abs/2510.27140)

- Mobile LLM Agent Indirect Prompt Injection Pipeline: introduces a framework where user prompts and environmental data, potentially containing malicious injected instructions, are concatenated and processed by a Foundation Model F, which then generates executable steps for an Action Executor to interact with the Mobile Device.
- This pipeline highlights how untrusted third-party content can introduce vulnerabilities by manipulating LLM agents into unintended actions, data exfiltration, or malware installation.
- The research systematically evaluates this framework against various attack vectors across eight state-of-the-art mobile LLM agents, revealing systemic vulnerabilities and novel privilege-escalation pathways.

---

[A Memory-Efficient Retrieval Architecture for RAG-Enabled Wearable Medical LLMs-Agents](http://arxiv.org/abs/2510.27107)

- QATS-HR (Quantization-Aware Two-Stage Hierarchical Retrieval): introduces a memory-efficient retrieval architecture for RAG-enabled wearable medical LLM agents, featuring a two-stage hierarchical retrieval scheme, a RAG retrieval accelerator with PEs, Similarity Calculator, and Rerank Module, alongside a Bit-Planar Storage Strategy and Query Stationary Dataflow.
- This architecture significantly reduces external memory access and energy consumption by combining approximate retrieval using MSB INT4 embeddings for candidate generation with full 8-bit precision retrieval on a pre-selected candidate set.
- Designed for resource-constrained edge devices, the framework leverages on-chip SRAM buffers and a query buffer to optimize data reuse and minimize off-chip DRAM transfers, thereby enhancing efficiency for personalized medical services.

---

[CombiGraph-Vis: A Curated Multimodal Olympiad Benchmark for Discrete Mathematical Reasoning](http://arxiv.org/abs/2510.27094)

- Agentic Validation Pipeline: is a multi-stage framework for curating and validating the CombiGraph-Vis benchmark, incorporating critics, aggregators, issue detectors, solution engagers, fix planners, fixers, validators, and replanners.
- This pipeline ensures the consistency and fidelity of CombiGraph-Vis, a 1,135-problem multimodal benchmark for discrete mathematical reasoning, by systematically detecting and resolving errors.
- The framework addresses challenges like image-based problem interpretation, distractor susceptibility, and the need for robust, multimodal discrete-math reasoning.

---

[A Step Toward World Models: A Survey on Robotic Manipulation](http://arxiv.org/abs/2511.02097)

- World Models: introduces internal representations that capture environmental dynamics, enabling prediction, planning, and reasoning for autonomous agents in robotic manipulation.
- The survey categorizes these models by paradigms like implicit, latent dynamics, and video generation, discussing their architectural designs and functional roles.
- It distills core components and capabilities, such as multimodal perception, imagination, and long-horizon reasoning, to outline a roadmap for generalizable and practical robotic world models.

---

[VISUAL BACKDOOR ATTACKS ON MLLM EMBODIED DECISION MAKING VIA CONTRASTIVE TRIGGER LEARNING](http://arxiv.org/abs/2510.27623)

- BEAT (visual Backdoor attacks on MLLM decision making in Embodied Agents via contrastive Trigger learning): introduces a framework to inject visual backdoors into MLLM-based embodied agents using environmental objects as triggers, featuring training set construction (exposing agents to trigger variability) and a two-stage training scheme (ensuring precise backdoor activation) with Supervised Fine-tuning (acquiring general proficiency) and Contrastive Trigger Learning (sharpening decision boundaries).
- The framework addresses the challenge of object triggers' wide variation across viewpoints and lighting by creating a diverse training set and using CTL to formulate trigger discrimination as preference learning.
- BEAT achieves high attack success rates while maintaining strong benign task performance and generalizes reliably to out-of-distribution trigger placements, exposing a critical security risk in MLLM-based embodied agents.

---

[GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation](http://arxiv.org/abs/2510.27210)

- GUI-Rise: introduces a reasoning-enhanced framework that systematically integrates structured reasoning, action prediction, and history summarization, with Current Screen Observation (visual input), User Instruction (textual input), and Interaction History (textual input) as inputs to the GUI-Rise Agent (multimodal large language model), which outputs a Structured Reasoning Subtask (progress estimation, decision reasoning), Action Prediction Subtask (next GUI action), and History Summarization Subtask (updated history summary).
- The framework trains a GUI agent through supervised fine-tuning and reinforcement learning with Group Relative Policy Optimization (GRPO), employing specialized rewards for action accuracy, structured reasoning, and history summary quality.
- This design enables the agent to maintain coherent behavior, continuously reason about evolving interface states, and effectively integrate its own history for robust GUI navigation across diverse tasks.

---

[Mano Technical Report](http://arxiv.org/abs/2509.17336)

- Mano: introduces a robust GUI agent built upon a multi-modal foundation model, integrating an exploration module, an inference process pipeline, and a three-stage training pipeline.
- The framework addresses challenges in GUI automation by leveraging a novel simulated environment for high-fidelity data generation and a verification module for error recovery.
- Mano demonstrates state-of-the-art performance on GUI benchmarks, achieving significant improvements in success rate and operational accuracy through domain-specific data, iterative training, and holistic reward design.

---

[LiteCUA: Computer as MCP Server for Computer-Use Agent on AIOS](http://arxiv.org/abs/2505.18829)

- AIOS 1.0 (AIOS 1.0): introduces a platform that reconceptualizes the computer-use agent challenge by contextualizing the computer as an MCP Server, with LiteCUA demonstrating its effectiveness.
- The core innovation involves transforming the computer into a semantic landscape aligned with LLM understanding via the MCP Server, decoupling interface complexity from decision complexity.
- LiteCUA utilizes an Orchestrator-Worker and Perceive-Reason-then-Act cycle, leveraging AIOS 1.0's contextualized environment to achieve competitive performance on the OSWorld benchmark.

---

#### 30th October 2025

[Cooperative Integrated Estimation-Guidance for Simultaneous Interception of Moving Targets](http://arxiv.org/abs/2510.26948)

- Cooperative Integrated Estimation-Guidance (CIEG): introduces a framework for simultaneous interception of non-maneuvering targets by a team of unmanned autonomous vehicles, utilizing dedicated sensors, a prescribed-time observer, a directed communication topology (sensing graph), true proportional navigation guidance (TPNG), a prescribed-time controller, and an actuation graph.
- The framework enables sensorless vehicles to estimate target states via information exchange over a directed communication topology and achieves time-to-go consensus using prescribed-time control.
- CIEG demonstrates robustness to individual agent failures and ensures accurate, simultaneous interception across diverse target motions and engagement geometries.

---

[The Oversight Game: Learning to Cooperatively Balance an AI Agent's Safety and Autonomy](http://arxiv.org/abs/2510.26752)

- The Oversight Game: introduces a game-theoretic framework for post-hoc AI control, with a Superintelligence (SI) agent choosing to play or ask, and a Human (H) overseer choosing to trust or oversee, modeled as a Markov Potential Game (MPG) to ensure alignment.
- This framework wraps a pretrained, potentially unsafe AI policy (σ) with a minimal control interface, using a Shared Reward Mechanism to incentivize the SI to defer when risky and the human to oversee when necessary, leading to emergent safe behavior.
- The model provides theoretical guarantees for local alignment under an "Ask-Burden Assumption" and demonstrates empirically that independent learning can achieve zero safety violations while maintaining task completion in a gridworld environment.

---

[Using Copilot Agent Mode to Automate Library Migration: A Quantitative Assessment](http://arxiv.org/abs/2510.26699)

- GitHub's Copilot Agent Mode: introduces an autonomous AI system for automating library migration, utilizing an LLM (GPT-40), Copilot Instructions Creation Prompt, Migration Instructions File, Migration Prompt, Client Applications, Python Virtual Environment, Package Manager (uv), PostgreSQL Docker Container, Copilot Chat Thought Process, Documentation/Source Code Access, and Codebase to perform multi-step migration workflows.
- The system plans, reasons, and executes complex programming tasks, specifically upgrading Python's SQLAlchemy library from version 1 to 2 across multiple client applications without constant human supervision.
- It leverages generated instructions and prompts to guide the migration, aiming to transform code and manage dependencies while assessing effectiveness through metrics like Migration Coverage and test pass rates.

---

[Agentic AI Home Energy Management System: A Large Language Model Framework for Residential Load Scheduling](http://arxiv.org/abs/2510.26603)

- Agentic AI HEMS: introduces a hierarchical multi-agent LLM framework for residential load scheduling, featuring an orchestrator agent, specialist agents, an API layer, and a ReAct loop for autonomous coordination.
- The system enables natural language-based scheduling of multiple appliances (washing machine, dishwasher, EV charger) by leveraging external APIs for real-time data and optimizing for minimal electricity cost.
- This framework operates without example demonstrations or few-shot learning, relying purely on LLM reasoning and tool descriptions to manage complex workflows and address HEMS adoption barriers.

---

[CATARENA: EVALUATION OF LLM AGENTS THROUGH ITERATIVE TOURNAMENT COMPETITIONS](http://arxiv.org/abs/2510.26852)

- CATArena (Code Agent Tournament Arena): introduces an iterative, competitive peer-learning framework for evaluating LLM agents, including Agents (LLM agents), Task Environment (game rules/sample AI), Strategies (agent-developed code), Tournament Arena (competition platform), Rank (performance order), Log (competition records), Counter-Adaptation (peer-learning process), Self-Improving (strategy refinement), New Strategies (updated agent code), and Tournament Results (scoring/evaluation metrics), which systematically evaluates their learning capabilities through repeated interactions and feedback in open-ended game competitions.
- The framework addresses score saturation in existing benchmarks by using a tournament-style evaluation platform featuring diverse board and card games with open-ended scoring, enabling continuous and dynamic assessment of rapidly advancing agent capabilities.
- CATArena provides reliable, stable, and scalable benchmarking for core agent abilities, particularly learning ability and strategy coding, by allowing agents to revise and update strategies based on competition outcomes and observed policies.

---

[Who Grants the Agent Power? Defending Against Instruction Injection via Task-Centric Access Control](http://arxiv.org/abs/2510.26212)

- AgentSentry: introduces a lightweight runtime task-centric access control framework with User, Agent, Task Interpreter, Task Context, Policy Generation Engine (PGE), PolicySet, Policy Store, Policy Enforcement Point (PEP), and Policy Decision Point (PDP) components, designed to enforce dynamic, task-scoped permissions for AI agents.
- This framework addresses the instruction injection vulnerability in AI agents by dynamically generating and enforcing minimal, temporary policies aligned with the user's specific task, preventing unauthorized actions while allowing legitimate tasks to complete.
- AgentSentry's core principle is to grant permissions that are transient and specific to the task, automatically revoking them upon completion to eliminate persistent vulnerabilities and prevent data exfiltration.

---

[The FM Agent](http://arxiv.org/abs/2510.26144)

- FM Agent (Foundation Model Agent): introduces a novel, general-purpose multi-agent framework that leverages LLM-based reasoning and large-scale evolutionary search to address complex real-world challenges, incorporating a Cold Start Stage (initial solution generation), an Evolve Stage (iterative solution optimization), and a robust Infrastructure (supports distributed execution).
- The framework integrates key innovations including expert guidance during cold-start initialization, an adaptive diversity-driven sampling strategy for iterative optimization, and domain-specific evaluators that combine correctness, effectiveness, and LLM-supervised feedback.
- Built on Ray Architecture (orchestrates distributed computation), FM Agent achieves state-of-the-art results across diverse domains like machine learning, GPU kernel optimization, and mathematical problems, demonstrating broad applicability and scalability.

---

[WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios](http://arxiv.org/abs/2510.26125)

- NaiveEMMA (Simplified EMMA Model): introduces a baseline end-to-end driving model, with Cameras (input 8 images), High-level command (input routing instruction), and Ego states (input past vehicle data) as inputs, processed by NaiveEMMA (simplified E2E model) utilizing Gemini (MLLM backbone) to output Predicted Trajectory Waypoints (output future path).
- The paper primarily introduces WOD-E2E (Waymo Open Dataset for End-to-End Driving), a new dataset focusing on challenging long-tail scenarios for end-to-end autonomous driving, and RFS (Rater Feedback Score), a novel human-aligned open-loop evaluation metric.
- WOD-E2E contains 4,021 driving segments (approximately 12 hours) of rare real-world scenarios (occurring with a frequency less than 0.03%), providing comprehensive data including 360-degree camera views, high-level routing information, and ego vehicle position history.

---

[Accelerating Real-World Overtaking in F1TENTH Racing Employing Reinforcement Learning Methods](http://arxiv.org/abs/2510.26040)

- TD3-Overtake (TD3 Algorithm Overtaking): introduces a novel autonomous F1Tenth racing strategy with overtaking behaviors learned through reinforcement learning, utilizing a TD3 Algorithm, Autonomous F1Tenth Simulator, ROS 2 Humble/Gazebo framework, Overtaking Training Environment, Training Vehicle, Competitor Cars, State Space, Action Space, Reward Function, VESC motor controller, LiDAR, Real F1Tenth car, and Real-world race track, to enable an agent to reliably navigate a track and overtake opponents in both simulation and reality.
- The agent demonstrates deliberative overtaking behaviors, achieving an 87% overtaking rate in real-world scenarios, significantly outperforming an agent trained only for racing (56%).
- The end-to-end reinforcement learning approach minimizes the sim-to-real gap, allowing the model to generalize its learned overtaking capabilities from simulation to physical F1Tenth vehicles with minimal adjustments.

---

[Semantically-Aware LLM Agent to Enhance Privacy in Conversational AI Services](http://arxiv.org/abs/2510.27016)

- LOPSIDED (Local Optimizations for Pseudonymization with Semantic Integrity Directed Entity Detection): introduces a semantically-aware privacy agent that safeguards sensitive PII by dynamically replacing entities in user prompts with consistent pseudonyms and then restoring original entities in the LLM's response.
- The framework ensures contextual integrity by generating semantically appropriate replacement entities, preserving the meaning of both the input prompt and the derived response.
- It operates as an intermediary between the user and remote LLMs, locally pseudonymizing sensitive information before transmission and de-pseudonymizing responses before presentation.

---

[FLOWMESH: A SERVICE FABRIC FOR COMPOSABLE LLM WORKFLOWS](http://arxiv.org/abs/2510.26913)

- FlowMesh: introduces a multi-tenant service fabric for composable LLM workflows, decomposing them into fine-grained operators with recorded lineage, enabling work deduplication and request batching on heterogeneous GPUs.
- The system features a global control plane for scheduling and an elastic pool of stateless workers backed by a content-addressable store, ensuring rapid scaling and fault tolerance.
- FlowMesh achieves significant cost reduction and lower energy usage compared to baselines, while maintaining similar or better latency under dynamic and failure-prone conditions.

---

[Gistify! Codebase-Level Understanding via Runtime Execution](http://arxiv.org/abs/2510.26790)

- GISTIFY: introduces a task where a coding LLM generates a single, minimal, self-contained gistified file from a given codebase and command, evaluated by Execution Fidelity, Line Execution Rate, and Line Existence Rate metrics.
- This task requires LLMs to demonstrate structural understanding of codebases, accurate modeling of execution flow, and the ability to produce substantial code patches.
- The framework provides a systematic way to measure codebase-level understanding, offering direct insight into models' reasoning capabilities over runtime execution rather than isolated snippets.

---

[Inverse Knowledge Search over Verifiable Reasoning: Synthesizing a Scientific Encyclopedia from a Long Chains-of-Thought Knowledge Base](http://arxiv.org/abs/2510.26854)

- SciencePedia Framework: introduces a scalable framework that decompresses scientific reasoning by constructing a verifiable Long Chain-of-Thought (LCoT) knowledge base and projecting it into an emergent encyclopedia, SciencePedia, using a Socrates Agent, LCoT Knowledge Base, Brainstorm Search Engine, and Plato Agent (LLM Synthesizer).
- The framework operationalizes an endpoint-driven, reductionist strategy where the Socrates Agent generates and verifies LCoT-QA pairs, which are then stored in the LCoT Knowledge Base.
- The Brainstorm Search Engine performs inverse knowledge search on the LCoT Knowledge Base to retrieve derivations, which the Plato Agent then synthesizes into coherent, pedagogically clear scientific articles for SciencePedia.

---

[STOP WASTING YOUR TOKENS: TOWARDS EFFICIENT RUNTIME MULTI-AGENT SYSTEMS](http://arxiv.org/abs/2510.26585)

- SUPERVISORAGENT: introduces a lightweight, modular framework for runtime, adaptive supervision in Multi-Agent Systems (MAS), utilizing an Adaptive Filter (LLM-free detection), Context Window (real-time MAS state), Supervision Action Space (intervention strategies), and a Memory Module (supervisor's own memory) to enhance robustness and efficiency.
- The framework proactively corrects errors, guides inefficient behaviors, and purifies observations at critical junctures without altering the base agent's architecture, triggered by an LLM-free adaptive filter.
- Experiments on the GAIA benchmark show SUPERVISORAGENT reduces token consumption by an average of 29.45% for the Smolagent framework while maintaining competitive success rates, demonstrating broad applicability and robustness across various benchmarks and LLMs.

---

[INFOFLOW: REINFORCING SEARCH AGENT VIA REWARD DENSITY OPTIMIZATION](http://arxiv.org/abs/2510.26575)

- InfoFlow: introduces a systematic framework for reinforcing search agents via reward density optimization, incorporating Sub-goal Scaffolding (decomposes tasks, assigns rewards), Pathfinding Hints (injects corrective guidance), and Trajectory Refinement (dual-agent architecture).
- The framework employs a dual-agent design, comprising a Researcher Agent (performs reasoning, planning, search) and a Refiner Agent (synthesizes retrieved evidence), to enhance efficiency and accuracy in deep search tasks.
- This approach addresses low reward density by providing denser learning signals through intermediate rewards, adaptive guidance, and efficient information processing, enabling lightweight LLMs to achieve competitive performance.

---

[Simulating and Experimenting with Social Media Mobilization Using LLM Agents](http://arxiv.org/abs/2510.26494)

- LLM-SocioPol (LLM Social-Political Mobilization): introduces an agent-based social media simulator that integrates real demographic and network data with heterogeneous LLM agents to model online voter mobilization and peer influence.
- The framework simulates agents' interactions within a social media environment, allowing them to manage follow relationships, engage with and create posts, process social-influence cues, and dynamically update voting intentions.
- This simulator provides a controlled and reproducible environment for testing counterfactual designs and sensitivity analyses in political mobilization research, bridging field experiments with computational modeling.

---

[The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration](http://arxiv.org/abs/2510.26352)

- Interaction-Centric Framework for Automatic Team Composition: introduces an automatic LLM team composition method that constructs a language model graph from pairwise conversations, then applies community detection to identify synergistic model clusters.
- This framework operates without prior knowledge of LLM internal architectures or training data, relying instead on the semantic coherence of dialogues to map latent relational structures.
- Experiments demonstrate that topic-specific priming of conversations enables the framework to identify functionally coherent LLM groups that outperform random baselines and approach manually-curated team performance.

---

[Agent Skills Enable a New Class of Realistic and Trivially Simple Prompt Injections](http://arxiv.org/abs/2510.26328)

- Agent Skills: introduces a method for trivially simple prompt injections into LLMs by embedding malicious instructions within Agent Skills' Skill Directory, SKILL.md files, and Skill Scripts/Files, which are executed by Claude Code or Claude Web Interface after being loaded into the System Prompt.
- The paper demonstrates how these injections can exfiltrate sensitive data and bypass system-level guardrails, highlighting a fundamental vulnerability in LLM agent frameworks.
- The research emphasizes that human oversight is challenging due to the length and complexity of skill files, making it difficult to detect hidden malicious instructions.

---

[Urban-MAS: Human-Centered Urban Prediction with LLM-Based Multi-Agent System](http://arxiv.org/abs/2511.00096)

- Urban-MAS (Human-Centered Urban Prediction with LLM-Based Multi-Agent System): introduces a novel LLM-based multi-agent system for human-centered urban tasks, integrating Predictive Factor Guidance Agents (prioritize influential factors), Reliable UrbanInfo Extraction Agents (ensure reliable information), and Multi-UrbanInfo Inference Agents (integrate information for prediction) to enhance prediction performance under zero-shot conditions.
- The framework significantly reduces prediction errors compared to single-LLM baselines by systematically prioritizing predictive factors and improving the reliability of urban knowledge extraction.
- Urban-MAS provides a scalable paradigm for human-centered urban AI prediction, demonstrating efficient, low-cost, and significant zero-shot gains across diverse urban tasks and cities.

---

[Empowering RepoQA-Agent based on Reinforcement Learning Driven by Monte-carlo Tree Search](http://arxiv.org/abs/2510.26287)

- RepoSearch-R1 (Reinforcement Learning Driven by Monte-carlo Tree Search): introduces a novel agentic reinforcement learning framework that integrates Monte Carlo Tree Search (MCTS) with Group Relative Policy Optimization (GRPO) to enhance LLMs' repository-level reasoning capabilities through self-training, including Monte-carlo Tree Search (generates exploration trajectories), MCTS Selection (chooses promising nodes), Exploration-Decay UCT (balances exploration/exploitation), MCTS Expansion (adds child nodes), Self-Critic Guided Child Generation (generates diverse children), MCTS Simulation (rollout with policy), MCTS Backpropagation (updates node values), Trajectory Selection (selects promising paths), Reward Computation (evaluates trajectories), LLM-as-a-Judge Outcome Reward (assesses final answer quality), Intermediate Process Reward Accumulation (measures tool usage), Reward Aggregation Mechanism (combines reward types), Advantage Computation (normalizes rewards), Group Relative Policy Optimization (updates LLM policy), LLM Policy (guides agent actions), RepoQA-Agent (performs repository QA), ReAct Framework (Thought/Action/Observation cycle), Tools (repository exploration functions), review_file (inspects file content), search_keyword_in_folder (finds keywords in files), list_files_in_folder (lists directory contents), search_symbol_in_file (finds code symbols), and search_file_in_folder (finds specific files).
- The framework eliminates dependence on external model distillation by generating diverse, high-quality reasoning trajectories via MCTS-guided rollouts and self-critic mechanisms, addressing data compliance concerns in enterprise environments.
- RepoSearch-R1 significantly improves answer completeness and training efficiency for repository question-answering tasks, enabling autonomous agents to develop sophisticated reasoning capabilities in data-scarce environments.

---

[Graph-Enhanced Policy Optimization in LLM Agent Training](http://arxiv.org/abs/2510.26270)

- GEPO (Graph-Enhanced Policy Optimization): introduces a framework that dynamically constructs a state-transition graph from agent experience to provide synergistic learning signals for LLM agent training.
- The framework addresses structural blindness in LLM agents by integrating graph-theoretic centrality to guide exploration, assign credit, and enable farsighted planning.
- GEPO achieves significant performance gains on long-horizon, sparse-reward tasks by explicitly modeling environmental structure and leveraging online graph-building.

---

[Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles](http://arxiv.org/abs/2510.26242)

- REG-TSC (Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles): introduces a framework for generalizable traffic signal control with emergency vehicle response, integrating an emergency-aware reasoning framework (RERAG), an LLM-based signal optimization agent, and simulation-driven fine-tuning.
- The framework employs RERAG to distill critical knowledge from historical emergency scenarios and expert responses, enhancing the reliability and rationality of LLM agents' emergency decisions.
- REG-TSC further utilizes Reward-guided Reinforced Refinement (R³) and a type-agnostic traffic representation to improve generalization across diverse, heterogeneous intersections and adaptively sample training experience.

---

[Linking Heterogeneous Data with Coordinated Agent Flows for Social Media Analysis](http://arxiv.org/abs/2510.26172)

- SIA (Social Insight Agents): introduces an LLM agent system that links heterogeneous multi-source social media data through coordinated agent flows, featuring a Planner, Core Analytical Agents (Query, Data Mining, Visualization, Insight Report), and a Heterogeneity Coordinator (Query, Mining, Visualization and Report Coordinators) with Knowledge-based Data Fusion, guided by a Taxonomy of Social Media Insights.
- The system enables agents to plan and execute coherent analysis strategies, ensuring multi-source integration and providing a transparent workflow for user validation and refinement.
- SIA effectively discovers diverse and meaningful insights from social media data while supporting human-agent collaboration in complex analytical tasks.

---

[Real-DRL: Teach and Learn in Reality](http://arxiv.org/abs/2511.00112)

- Real-DRL: introduces a framework for safety-critical autonomous systems, enabling runtime learning of a DRL agent to develop safe and high-performance action policies in real plants, comprising a DRL-Student, a PHY-Teacher, and a Trigger, along with self-learning and teaching-to-learn replay buffers, actor/critic networks, and a safety-informed batch sampling mechanism.
- The framework addresses safety challenges from unknown unknowns and the Sim2Real gap by integrating physics-model-based safety assurance with data-driven reinforcement learning, featuring assured safety, automatic hierarchy learning, and safety-informed batch sampling.
- Experiments on a real quadruped robot, a simulated quadruped robot, and a cart-pole system demonstrate the framework's effectiveness in maintaining safety and achieving high performance in dynamic and unpredictable environments.

---

[GUI KNOWLEDGE BENCH: REVEALING THE KNOWLEDGE GAP BEHIND VLM FAILURES IN GUI TASKS](http://arxiv.org/abs/2510.26098)

- GUI Knowledge Bench: introduces a novel benchmark to evaluate the GUI knowledge encoded in VLMs by categorizing it into three dimensions: Interface Perception (recognizing GUI elements, states, and layout), Interaction Prediction (anticipating action outcomes and preconditions), and Instruction Understanding (interpreting task goals and planning multi-step operations).
- The benchmark comprises 3483 knowledge-centric questions across six platforms and 292 applications, designed to systematically test VLMs' GUI knowledge prior to downstream tasks.
- Evaluation results reveal significant gaps in current VLMs' understanding of system states, action outcomes, and task completion verification, providing insights for developing more capable GUI agents.

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
