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



#### 24th Oct 2025

[DeepAgent: A General Reasoning Agent with Scalable Toolsets](http://arxiv.org/abs/2510.21618)

- DeepAgent: introduces an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process, utilizing Reasoning LLMs, an Auxiliary LLM, a Tool Retriever, a Tool Executor, a Memory Folding Module with Episodic, Working, and Tool Memories, Scalable Toolsets, and an Environment, trained with ToolPO.
- The framework addresses long-horizon interactions and context length explosion through an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation.
- DeepAgent employs ToolPO, an end-to-end reinforcement learning strategy leveraging LLM-simulated APIs and tool-call advantage attribution, to efficiently and stably teach general-purpose tool use.

---

[REMONI: An Autonomous System Integrating Wearables and Multimodal Large Language Models for Enhanced Remote Health Monitoring](http://arxiv.org/abs/2510.21445)

- REMONI (REmote health MONItoring system): introduces an autonomous remote health monitoring system that integrates wearables, IoT, and MLLMs to collect, process, and analyze patient data, facilitating anomaly detection and natural language interaction for medical professionals.
- The system utilizes wearable devices and cameras for data acquisition, edge devices for real-time anomaly detection, and cloud infrastructure for storage and computing, all orchestrated to provide timely alerts and historical data access.
- Its NLP engine, powered by a General LLM and a Multimodal LLM, interprets caregiver inquiries, recognizes patient activity and emotion from visual data, and generates comprehensive responses, enhancing telehealth and reducing medical workload.

---

[A Knowledge-Graph Translation Layer for Mission-Aware Multi-Agent Path Planning in Spatiotemporal Dynamics](http://arxiv.org/abs/2510.21695)

- Knowledge-Graph Translation Layer (KGTL): introduces a framework centered on a Knowledge Graph (KG) (central orchestrator) that functions as an intelligent translation layer, with a Data Plane (mission tensor compiler) and a Control Plane (coordination logic provider) to bridge the semantic gap between high-level mission objectives and low-level planner inputs for multi-agent path planning.
- The framework compiles declarative facts into per-agent, mission-aware "worldviews" (Mission Tensors) and physics-aware traversal rules, which are then used by an Agnostic Path Planner (domain-unaware optimizer) and a Selector/Coordinator (plan deconflictor) to generate coordinated mission plans.
- This architecture enables adaptive planning by allowing complex, coordinated paths to be modified simply by changing facts in the KG, supporting reactive replanning through incremental recompilation of affected artifacts.

---


[OpenHype: Hyperbolic Embeddings for Hierarchical Open-Vocabulary Radiance Fields](http://arxiv.org/abs/2510.21441)

- OpenHype (Hyperbolic Embeddings for Hierarchical Open-Vocabulary Radiance Fields): introduces a novel framework for open-vocabulary segmentation on NeRFs, leveraging a CLIP Feature Extractor, a Hyperbolic Auto-encoder with an Encoder and Decoder, a NeRF Model with a NeRF Network, a Hyperbolic Latent Space, a Geodesic Path Traversal Module, a Text Query Prompt, and a Similarity Module to embed hierarchical structures in a continuous hyperbolic latent space.
- This approach enables continuous traversal of scene hierarchies through geodesic paths, allowing for multi-scale responses to open-vocabulary queries without discrete levels or multiple rendering passes.
- The framework demonstrates superior efficiency and adaptability in 3D scene understanding by naturally encoding multi-scale relationships and outperforming state-of-the-art methods on benchmarks.

---

[HIKMA: Human-Inspired Knowledge by Machine Agents through a Multi-Agent Framework for Semi-Autonomous Scientific Conferences](http://arxiv.org/abs/2510.21370)

- HIKMA (Human-Inspired Knowledge by Machine Agents): introduces an end-to-end multi-agent framework for semi-autonomous scientific conferences, integrating AI-dataset curation, manuscript generation, peer review, revision, conference presentation, and archival dissemination.
- The framework leverages LLMs, structured research workflows, and domain safeguards to support traditional scholarly practices while ensuring intellectual property protection, transparency, and integrity.
- HIKMA functions as a testbed for AI-enabled scholarship, demonstrating how AI can act as an auditable partner in the entire research lifecycle, from hypothesis intake to publication.

---

[DAO-AI: Evaluating Collective Decision-Making through Agentic AI in Decentralized Governance](http://arxiv.org/abs/2510.21117)

- DAO-AI (Decentralized Autonomous Organization - Artificial Intelligence): introduces an agentic AI framework for evaluating collective decision-making in decentralized governance, utilizing an Input Module, Data Preparation Stage, MCP Processing & Learning Layer, Decision Layer (LLM-based decision maker), Output Module, and Evaluation Layer.
- The framework orchestrates multiple specialized Modular Composable Programs (MCPs) to fetch, analyze, and synthesize diverse governance data, including proposal metadata, forum discussions, voting dynamics, and market responses.
- Built upon the Agentics framework, DAO-AI provides an LLM-based decision maker that interprets proposal contexts, retrieves historical data, and independently determines voting positions, offering interpretable and auditable signals for realistic DAO governance settings.

---

[ASTABENCH: RIGOROUS BENCHMARKING OF AI AGENTS WITH A SCIENTIFIC RESEARCH SUITE](http://arxiv.org/abs/2510.21652)

- AstaBench: introduces a rigorous benchmarking suite for AI agents in scientific research, featuring a holistic measure of agentic ability, a reproducible environment with production-grade search tools, and a comprehensive suite of optimized agents and baselines.
- The framework includes the Asta Environment for controlled evaluation, the agent-eval Agents Evaluation Toolkit for cost-aware reporting, and the AstaBench Leaderboard to account for confounding variables like tool usage and inference cost.
- AstaBench evaluates 57 agents across 22 architectural classes on over 2400 problems spanning various scientific domains and tasks, revealing that AI still faces significant challenges in scientific research assistance.

---

#### 23rd October 2025

[BUILDARENA: A PHYSICS-ALIGNED INTERACTIVE BENCHMARK OF LLMS FOR ENGINEERING CONSTRUCTION](http://arxiv.org/abs/2510.16559)

- BuildArena: introduces a physics-aligned interactive benchmark for LLMs in engineering construction, comprising Task Definition (defines construction goals), LLM-based Construction (including a Spatial Geometric Computation Library and an LLM Agentic Workflow), and Simulation-based Evaluation (powered by the Besiege Simulator), where it enables LLMs to perform 3D structure construction via natural language instructions and evaluates performance within a physically constrained environment.
- The benchmark provides a highly customizable framework for in-depth comparison and analysis of LLMs, supporting extendable task design strategies across static and dynamic mechanics with multiple difficulty tiers.
- It includes a 3D Spatial Geometric Computation Library for supporting construction based on language instructions and a baseline LLM agentic workflow for comprehensive evaluation of diverse model capabilities.

---

[AGENTARCEVAL: AN ARCHITECTURE EVALUATION METHOD FOR FOUNDATION MODEL BASED AGENTS](http://arxiv.org/abs/2510.21031)

- AgentArcEval: introduces a novel architecture evaluation method for Foundation Model (FM)-based agents, addressing complexities of their compound architecture, autonomous behavior, and continuous evolution, utilizing a catalogue of agent-specific general scenarios to guide architectural analysis and decision-making.
- The method builds on established ATAM principles, incorporating agent-specific artifacts and guardrails into the evaluation process to support early-stage analysis of quality trade-offs through structured, context-specific scenarios.
- Demonstrated through a case study on the Luna tax copilot, AgentArcEval is applicable to various agentic systems and aims to evolve as a community-driven living document.

---

[Learning Decentralized Routing Policies via Graph Attention-based Multi-Agent Reinforcement Learning in Lunar Delay-Tolerant Networks](http://arxiv.org/abs/2510.20436)

- GAT-MARL (Graph Attention-based Multi-Agent Reinforcement Learning): introduces a decentralized routing framework for multi-robot lunar exploration missions, utilizing a CTDE paradigm with a shared policy model, Q-network, target network, and DDQN for learning optimal routing actions based on local observations and a reward function.
- The framework operates within a Lunar Delay Tolerant Network (LDTN) where autonomous rovers collect data, store packets in local buffers, and relay them to a lander, navigating intermittent connectivity and dynamic topologies.
- The GAT-MARL model employs a 2-layer GAT with attention heads and an MLP head to process graph-structured state information, enabling scalable and robust communication strategies without global topology updates or packet replication.

---

[Designing Intent Communication for Agent-Human Collaboration](http://arxiv.org/abs/2510.20409)

- Design Space for Intent Communication: introduces a multidimensional design space for intent communication, structured along Transparency Level (what is communicated), Task Abstraction Level (when to communicate), and Communication Modality (how to communicate), to guide the development of generalizable, multi-modal communication strategies.
- This design space is applied to three human-agent collaboration scenarios: bystander interaction, cooperative tasks, and shared control, demonstrating its capacity to generate adaptable and scalable communication strategies.
- The framework bridges the gap between intent content and communication implementation, providing a foundation for designing safer, more intuitive, and transferable agent-human interactions.

---

[ComProScanner: A multi-agent based framework for composition-property structured data extraction from scientific literature](http://arxiv.org/abs/2510.20362)

- ComProScanner: introduces an autonomous multi-agent framework for composition-property structured data extraction from scientific literature, utilizing CrewAI, LLMs, RAG, and specialized agents for metadata retrieval, article collection, information extraction, and evaluation.
- The framework extracts, validates, classifies, and visualizes machine-readable chemical compositions, properties, and synthesis data, integrating with publisher APIs and local PDFs to build comprehensive datasets.
- Evaluated across 10 LLMs using 100 journal articles, ComProScanner achieved an overall accuracy of 0.82 with DeepSeek-V3-0324, demonstrating its capability to handle complex experimental data for machine learning applications.

---

[GHOSTEI-BENCH: DO MOBILE AGENTS RESILIENCE TO ENVIRONMENTAL INJECTION IN DYNAMIC ON-DEVICE ENVIRONMENTS?](http://arxiv.org/abs/2510.20333)

- GhostEI-Bench introduces a benchmark for mobile agents, including an Agent (mobile VLM agent), Attack vectors (adversarial threat categories), Representative Domains (diverse application contexts), Critical Risk Fields (potential security harms), Action Space (agent interaction capabilities), Judge LLM (evaluates agent behavior), Android Emulators (realistic mobile environment), Environment Controller (manages emulator, injects attacks), and Evaluation Module (assesses task outcomes).
- This benchmark systematically evaluates mobile agent robustness against dynamic environmental injection attacks within fully operational Android emulators, assessing performance across critical risk scenarios.
- GhostEI-Bench employs a novel LLM-based evaluation protocol for fine-grained failure analysis, identifying precise points of failure in perception, recognition, or reasoning.

---

[UI-INS: ENHANCING GUI GROUNDING WITH MULTI-PERSPECTIVE INSTRUCTION-AS-REASONING](http://arxiv.org/abs/2510.20286)

- Instruction-as-Reasoning introduces a novel SFT+RL framework for GUI grounding, leveraging a data pipeline, vision encoder, language model, SFT stage, RL stage, and GRPO to treat instructions as dynamic analytical pathways for optimal UI element selection.
- The framework addresses instruction diversity and quality issues by augmenting data with multi-perspective instructions and enabling models to dynamically select the most effective reasoning pathway.
- UI-Ins models, built on this framework, achieve state-of-the-art grounding accuracy across five benchmarks and demonstrate emergent reasoning capabilities, including combining perspectives and reasoning from novel angles.

---

[From Questions to Queries: An AI-powered Multi-Agent Framework for Spatial Text-to-SQL](http://arxiv.org/abs/2510.21045)

- AI-powered Multi-Agent Framework for Spatial Text-to-SQL: introduces a multi-agent system designed to accurately translate natural language questions into spatial SQL queries, integrating a knowledge base, context retrieval, and a collaborative pipeline of specialized LLM-powered agents.
- The framework's core pipeline includes agents for entity extraction, metadata retrieval, query logic formulation, SQL generation, and a Review Agent for programmatic and semantic self-verification of generated SQL.
- Supported by orchestration, memory, and a governance layer, the system enhances spatial analysis accessibility and provides a robust foundation for spatial Text-to-SQL systems, demonstrating self-improvement through recorded interactions.

---

#### 22nd October 2025

[BEYOND REACTIVITY: MEASURING PROACTIVE PROBLEM SOLVING IN LLM AGENTS](http://arxiv.org/abs/2510.19771)

- PROBE (Proactive Resolution of Bottlenecks): introduces a benchmark designed to test LLM agents' proactive problem-solving capabilities, encompassing searching for unspecified issues, identifying specific bottlenecks, and executing appropriate resolutions.
- The benchmark evaluates agents across a pipeline including a World Model + User Datastore for information, Bottleneck identification, and Task Execution leading to Resolution, revealing that even state-of-the-art LLMs struggle with end-to-end proactive tasks.
- The paper also details a data generation pipeline that constructs synthetic world models, bottlenecks, true positives, and distractors to create a realistic and challenging evaluation environment for proactive AI systems.

---

[Review of Tools for Zero-Code LLM Based Application Development](http://arxiv.org/abs/2510.19747)

- Zero-Code LLM Platforms: introduces a comprehensive survey of recent zero-code LLM platforms, categorizing them by their LLM Backend, Interface Type, Output Type, Customization and Extensibility, Agent Support, Memory and Knowledge Integration, Workflow and Control Logic, API Integration and Tool Connectivity, and Multimodal and AI-Assisted Features.
- The paper provides a taxonomy distinguishing between dedicated LLM-based app builders and general no-code platforms that integrate LLM capabilities, highlighting each platform's strengths and limitations.
- While these platforms significantly lower the barrier to creating AI-powered applications, they still face challenges in flexibility, reliability, scalability, and prompt engineering skills, yet offer exciting opportunities for non-programmers.

---

[AUTOMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems](http://arxiv.org/abs/2510.19438)

- AUTOMT (A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems): introduces a multi-agent LLM framework, with M-Agent (extracts MRs from traffic rules), MR-RAG Database (stores, retrieves embedded MRs), T-Agent (analyzes test case context), and F-Agent (generates follow-up test cases), which automates MR extraction and follow-up test case generation for autonomous driving systems.
- The framework leverages LLMs to extract diverse Metamorphic Relations from traffic rules, stores them in a RAG-based database, and uses vision-language models for scenario analysis and follow-up test case generation.
- This modular architecture enhances test diversity, uncovers corner cases, and supports integration into industrial pipelines for systematic coverage of safety-critical scenarios in autonomous driving.

---

[SORA-ATMAS: Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities](http://arxiv.org/abs/2510.19327)

- SORA-ATMAS (Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities): introduces a principled governance framework integrating decentralized agentic intelligence with centralized oversight and dual-chain anchoring, featuring an SDIoT Architecture Layer (structural backbone) comprising an Application Layer (top-level intelligence/governance) with a SORA Governance Layer (central city-wide oversight) and an Agentic Layer (domain-specific autonomous agents), a Control Layer (manages communication/security), and a Perception Layer (collects real-time data).
- The framework enables heterogeneous agents (Weather, Traffic, Safety) to operate autonomously while remaining accountable to city-wide policies, utilizing multiple LLMs (GPT, Grok, DeepSeek) for semantic reasoning and risk-trust assessments.
- SORA-ATMAS ensures regulation-aligned, verifiable, and context-aware decision-making for smart cities, demonstrating robustness under high-risk conditions and efficient cross-domain interoperability.

---

[Are Large Language Models Sensitive to the Motives Behind Communication?](http://arxiv.org/abs/2510.19687)

- LMVEF: introduces a comprehensive study evaluating whether LLMs possess motivational vigilance, utilizing a rational model as a normative benchmark and assessing LLMs across three experimental paradigms, including deliberate vs. incidental information discrimination, nuanced motivational vigilance, and generalization to naturalistic online settings.
- The framework employs various LLMs (e.g., GPT-4o, Claude 3.5 Sonnet), different prompting methods (CoT, Direct, Steering), and compares LLM performance against human baselines using both controlled cognitive science data and real-world YouTube sponsorship transcripts.
- LMVEF reveals that while LLMs demonstrate basic motivational vigilance in controlled settings, their performance significantly degrades in complex, naturalistic environments, though simple steering prompts can partially recover vigilance by emphasizing intentions and incentives.

---

[AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing](http://arxiv.org/abs/2510.19661)

- AgentSense: introduces a hybrid, training-free framework for web-based participatory urban sensing, integrating a Classical Planner (generates initial baseline solutions) and a Multi-agent evolution system (iteratively refines solutions) with a Disturbance Parser (converts unstructured dynamic signals) and a Multi-agent refinement loop (LLM-powered iterative updates) comprising a Solver Agent (proposes solution updates), an Eval Agent (assesses solutions/provides feedback), a Memory Agent (accumulates reusable meta-operations), a Meta-operation database (stores historical operations), and a Verifier (ensures plan validity).
- The framework adaptively refines task assignments to dynamic urban conditions and heterogeneous worker preferences, generating natural language explanations for enhanced transparency and trust.
- AgentSense demonstrates distinct advantages in adaptivity, explainability, and robustness over traditional methods and single-agent LLM baselines, positioning it for deploying adaptive and explainable urban sensing systems on the web.

---

[HSCodeComp: A Realistic and Expert-level Benchmark for Deep Search Agents in Hierarchical Rule Application](http://arxiv.org/abs/2510.19631)

- HSCodeComp (Harmonized System Code Competition): introduces a realistic, expert-level e-commerce benchmark for deep search agents, including Data Collection and Diversity Control (sourcing, filtering product data), Information Gathering (collecting product details), Structured Data Extraction (extracting core features), Related Result Search (querying customs databases), Hierarchical Decision Rules Application (applying expert tariff rules), HSCode Confirmation (validating codes officially), and Human Expert Validation (quality assurance by senior experts), designed to evaluate multi-hop reasoning with hierarchical tariff rules.
- The benchmark comprises 632 product entries with human-annotated 10-digit Harmonized System Codes, reflecting real-world e-commerce data and challenges like noisy descriptions and complex rule logic.
- Extensive experiments reveal a significant performance gap between state-of-the-art LLMs and human experts, highlighting the difficulty of precise hierarchical rule application.

---

[gem5 Co-Pilot: AI Assistant Agent for Architectural Design Space Exploration](http://arxiv.org/abs/2510.19577)

- gem5 Co-Pilot (AI Assistant Agent for Architectural Design Space Exploration): introduces an LLM-powered AI agent for automating computer architecture Design Space Exploration, integrating a DSE AI Agent, the gem5 Simulator/DSDB, and a Streamlit UI.
- The DSE AI Agent, driven by an LLM and a state machine, dispatches gem5 configurations, analyzes simulation results, and leverages a Design Space Database for efficient exploration.
- This framework significantly reduces the time and cost of identifying optimal architectural parameters by intelligently navigating design spaces and avoiding unnecessary simulations.

---

[MODELING REALISTIC HUMAN BEHAVIOR USING GENERATIVE AGENTS IN A MULTIMODAL TRANSPORT SYSTEM: SOFTWARE ARCHITECTURE AND APPLICATION TO TOULOUSE](http://arxiv.org/abs/2510.19497)

- Generative Agent-based Multimodal Transport Simulation Framework: introduces a system for modeling realistic human mobility behavior, integrating GAMA Platform Simulation (interactive transport environment), Generative Agent (LLM-based decision-making core), LLM Model (generates context-aware plans), OpenTripPlanner (multimodal routing options), Data Exchange Pipeline (manages data flow), Population Data (agent initialization), and GTFS and Map Data (transport network information).
- This framework enables generative agents to make context-aware transport decisions and form habits over time by leveraging LLMs for decision-making, GAMA for spatial simulation and visualization, and OpenTripPlanner for detailed multimodal routing.
- The architecture separates spatial simulation from intelligent reasoning, allowing agents to adapt their future decisions based on evolving contexts and feedback, thereby advancing intelligent transportation systems and personalized mobility solutions.

---

[AegisMCP: Online Graph Intrusion Detection for Tool-Augmented LLMs on Edge Devices](http://arxiv.org/abs/2510.19462)

- AegisMCP (Online Graph Intrusion Detection for Tool-Augmented LLMs on Edge Devices): introduces a protocol-level intrusion detector for Model Context Protocol (MCP)-driven smart homes, utilizing a NEBULA-Schema for representing agent activity as streaming heterogeneous temporal graphs.
- The framework employs a multi-stage pipeline including data collection via MCP Proxy and network metadata, normalization, graph construction with Session DAGs, and a detector that fuses GraphSAGE-style edge behavior scores with DAG and novelty features.
- Designed for edge devices, AegisMCP performs CPU-only, sub-second inference using ONNX INT8, enabling near-real-time detection of multi-step misuse and exfiltration attacks.

---

[MSC-Bench: A Rigorous Benchmark for Multi-Server Tool Orchestration](http://arxiv.org/abs/2510.19423)

- MSC-Bench (Multi-Server Tool Orchestration Benchmark): introduces a rigorous benchmark for evaluating LLM agents in multi-server tool orchestration, featuring an MCP Ecosystem, Servers, Tools, Equal Function Sets (EFS), and a Five-Level Curriculum.
- The benchmark addresses gaps in existing evaluations by providing architectural realism, handling functional overlap with EFS, and offering a comprehensive end-to-end assessment across five complexity levels.
- MSC-Bench systematically stress-tests agent capabilities from single-tool orchestration to complex cross-server planning and robustness, revealing systemic weaknesses in state-of-the-art agents and guiding future development.

---

[MONITORING LLM-BASED MULTI-AGENT SYSTEMS AGAINST CORRUPTIONS VIA NODE EVALUATION](http://arxiv.org/abs/2510.19420)

- MAS Graph Backpropagation (Multi-Agent System Graph Backpropagation): introduces a dynamic defense paradigm for LLM-based Multi-Agent Systems, utilizing Graph Reconstruction (MAS as DAG), Connection Extraction (signed network, edge contribution score), Node Contribution Determination (backward propagation, total score calculation), Malicious Agent Detection (thresholding on node contribution scores), and Graph Repair (communication edge removal) to monitor and defend against corruption attacks.
- This technique models MAS communication as an information propagation problem over a signed graph, dynamically adjusting the graph topology to disrupt malicious communications and adapt to evolving attacks.
- It leverages the efficiency of the chain rule in backpropagation to accurately identify harmful nodes or edges, significantly outperforming existing MAS defense mechanisms in detection accuracy and system resilience.

---

[AGENTICMATH: ENHANCING LLM REASONING VIA AGENTIC-BASED MATH DATA GENERATION](http://arxiv.org/abs/2510.19361)

- AgenticMath: introduces a novel agentic pipeline for generating high-quality mathematical question-answer pairs, including Seed Question Filter, Agentic Question Rephrase, Answer Augment, and Question and Answer Evaluation stages.
- This multi-agent framework leverages LLMs for generation, evaluation, and coordinated decision-making, enforcing quality control at every stage of mathematical data generation to enhance LLM reasoning.
- AgenticMath generates data-efficient, high-quality datasets (30K-90K samples) that achieve competitive or superior performance compared to baselines trained on much larger datasets (400K-2.3M samples).

---

[DAMO: Data Mixing OPTIMIZER IN FINE-TUNING MULTIMODAL LLMS FOR MOBILE PHONE AGENTS](http://arxiv.org/abs/2510.19336)

- DaMo (Data Mixture Optimizer): introduces a novel solution employing a trainable network that predicts optimal data mixtures by forecasting downstream task performance for any given dataset ratio, including Data Mixing Space (all possible mixture combinations), Data Mixture Sampling (selects subset of mixtures), Small MLLM Training/Evaluation (initial model performance assessment), Downstream Task Performance Metrics (quantifies task performance), MLP-based DaMo (predicts performance from mixture), Optimal Data Mixture Extrapolation (identifies best data mixture), Larger MLLM Training (applies optimal mixture), and DaMo Extension/Alignment (adapts to other MLLMs).
- The framework addresses the challenge of determining optimal training data compositions for multitask supervised fine-tuning (SFT) of MLLMs, which existing approaches struggle with.
- DaMo achieves significant performance improvements on both a new specialized benchmark, PhoneAgentBench, and general benchmarks, demonstrating robust scalability and generalization across different MLLM architectures.

---

[Learning to Make Friends: Coaching LLM Agents toward Emergent Social Ties](http://arxiv.org/abs/2510.19299)

- The Multi-agent LLM social media conversation framework: introduces a multi-agent LLM simulation platform for social media conversations, including persona creation, social media simulation, conversation room, reward structures, and tie formation mechanisms.
- This framework enables LLM agents to repeatedly interact, evaluate one another, and adapt their behavior through in-context learning, accelerated by an optional coaching signal, to model human social behavior.
- The framework utilizes behavioral reward functions (SOC, INF, PRE, COORD, EMO) and memory mechanisms to facilitate emergent social ties and network structures mirroring real online communities.

---

[THEMCPCOMPANY: CREATING GENERAL-PURPOSE AGENTS WITH TASK-SPECIFIC TOOLS](http://arxiv.org/abs/2510.19286)

- TheMCPCompany: introduces a benchmark environment for evaluating general-purpose LLM agents, featuring self-hosted and Azure services, exposed through over 18,000 task-specific tools via MCP Servers, and includes MCPAgent as a baseline tool-calling agent with a Gateway MCP Server for tool retrieval and invocation.
- This benchmark simulates complex enterprise environments, providing a realistic setting for studying LLM agents' ability to navigate large, heterogeneous tool collections and solve challenging real-world tasks.
- The framework highlights the potential of task-specific tools for improving agent performance and reducing costs compared to browser-based agents, while also revealing challenges in tool retrieval and reasoning within complex environments.

---

[From Specification to Service: Accelerating API-First Development Using Multi-Agent Systems](http://arxiv.org/abs/2510.19274)

- LLM-based Multi-Agent System: introduces a system that automates the API-first development of RESTful microservices, including a spec-generator agent (generates OpenAPI specification), code-generator agent (generates server code), JSON-cleaner agent (cleans JSON data), code-fixer agent (updates code with fixes), code-tester agent (manages containers, sends requests, analyzes logs), an underlying GPT-40 LLM, User interaction, and a Local Environment for execution.
- The system creates OpenAPI specifications, generates server code, and refines it through a feedback loop that analyzes execution logs and error messages, enabling efficient issue detection and resolution.
- This approach reduces development iterations and ensures functional, robust services by running code locally and providing context-aware feedback and automated fixes.

---

[SheetBrain: A Neuro-Symbolic Agent for Accurate Reasoning over Complex and Large Spreadsheets](http://arxiv.org/abs/2510.19247)

- SheetBrain: introduces a neuro-symbolic dual-workflow agent framework for accurate reasoning over tabular data, including an Understanding Module (global comprehension), an Execution Module (tool-augmented reasoning), and a Validation Module (iterative self-correction).
- The framework enhances LLMs' ability to understand and reason over complex spreadsheets for both question answering and manipulation tasks by integrating symbolic code execution within a Python sandbox.
- SheetBrain leverages a closed-loop feedback architecture, where the validation module provides improvement feedback to the execution module, ensuring robust, accurate, and interpretable performance across diverse spreadsheet scenarios.

---

[See, Think, Act: Online Shopper Behavior Simulation with VLM Agents](http://arxiv.org/abs/2510.19245)

- See, Think, Act (Online Shopper Behavior Simulation with VLM Agents): introduces a framework for simulating human online shopper behavior using a VLM Agent, which processes Action History and Current Screen Observation to perform Rationale Generation and Next Action Prediction within a defined Action Space.
- The framework leverages vision-language models to jointly process textual HTML and visual GUI screenshots, enabling more faithful and cognitively aligned simulations compared to text-only approaches.
- It employs supervised fine-tuning and reinforcement learning with a hierarchical reward structure to enhance action prediction accuracy and generate interpretable rationales for user actions.

---

[DISROUTER: DISTRIBUTED SELF-ROUTING FOR LLM SELECTIONS](http://arxiv.org/abs/2510.19208)

- DiSRouter (Distributed Self-Router): introduces a novel distributed self-routing framework for LLM selections, featuring a Routing Procedure (query flow through agents), a Self-Awareness Training Pipeline (enhances LLM self-assessment), and Scenario Adaptability (dynamic adjustment to user preferences).
- This framework empowers each LLM agent to independently assess its competence and decide whether to answer a query or route it to another agent, moving away from centralized external routers.
- The system's effectiveness is driven by a two-stage training pipeline (SFT and RL) that instills self-awareness and allows agents to adapt their collective behavior based on a user-defined preference factor (α) for performance or cost.

---

[Defending Against Prompt Injection with DataFilter](http://arxiv.org/abs/2510.19207)

- DataFilter: introduces a test-time model-agnostic defense that removes malicious instructions from untrusted data before it reaches the backend LLM, utilizing a filter LLM, prompt, untrusted data, filtered data, backend LLM, SFT dataset, prompt template, and special tokens.
- The filter LLM is trained via supervised fine-tuning on simulated injections to selectively strip adversarial content while preserving benign information.
- This approach consistently reduces prompt injection attack success rates to near zero while maintaining LLM utility, offering a plug-and-play deployment for black-box commercial LLMs.

---

[Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning](http://arxiv.org/abs/2510.18179)

- AdCo (Adaptive Coopetition): introduces a novel inference-time framework where LLM agents use an adaptive, UCB-based coopetition mechanism, leveraging coarse verifier signals to decide whether to collaborate or compete and iteratively refine reasoning based on peer feedback.
- The framework enhances collective reasoning robustness by integrating model knowledge diversity and reasoning trace measures, promoting uncertainty-driven exploration, and isolating low-quality feedback through a customized filter mechanism.
- AdCo operates in a multi-round process, with agents exchanging information via a PubSub channel, refining solutions, and converging on a final answer through majority voting, demonstrating significant performance gains on mathematical reasoning benchmarks.

---

[PLAGUE: PLUG-AND-PLAY FRAMEWORK FOR LIFE-LONG ADAPTIVE GENERATION OF MULTI-TURN EXPLOITS](http://arxiv.org/abs/2510.17947)

- PLAGUE (Plug-and-Play Framework): introduces a novel framework for designing multi-turn attacks, dissecting the attack lifetime into Planner, Primer, and Finisher phases, and incorporating components like Attacker LLM, Target LLM, Rubric Scorer, Summarizer, Lifelong Learner, and Evaluator Judge LLM (J).
- This framework enables systematic and information-rich exploration of multi-turn attacks by maintaining goal relevance, evolving from feedback, and adaptively sampling diverse strategies.
- PLAGUE achieves state-of-the-art jailbreaking results with high efficiency, significantly improving attack success rates across leading LLMs by leveraging smart initialization, context-building, and feedback incorporation.

---

[A Tutorial on Cognitive Biases in Agentic AI-Driven 6G Autonomous Networks](http://arxiv.org/abs/2510.19973)

- Agentic System: introduces a tutorial on cognitive biases in LLM-powered 6G autonomous networks, with all LLM-empowered Agent, Perception, Digital Twin (DT), Collective Memory, Network APIs, A2A Protocol, and Model Context Protocol (MCP) components, providing a systematic overview of bias emergence, impact on agentic components, and mitigation strategies.
- The paper details a taxonomy of cognitive biases, including their mathematical formulation and emergence in telecom systems, and identifies commonly impacted agentic components such as reasoning, planning, memory, negotiation, tool use, and actuation.
- Two practical use-cases demonstrate the mitigation of anchoring, temporal, and confirmation biases in 6G inter-slice and cross-domain management, leading to improved latency and energy savings.

---

[VideoAgentTrek: Computer Use Pretraining from Unlabeled Videos](http://arxiv.org/abs/2510.19488)

- VIDEOAGENTTREK introduces a scalable pipeline that automatically mines structured computer-use trajectories from unlabeled screen-recorded videos, leveraging Video Collection and Preprocessing, VIDEO2ACTION (Inverse Dynamics Module), and Agent Training components.
- The VIDEO2ACTION module, an inverse dynamics system, extracts explicit action labels and parameters from implicit video demonstrations, including action event detection, parameterization, and inner monologue generation.
- This framework enables large-scale computer-use pretraining by converting passive internet videos into high-quality supervision, significantly improving task success rates and step accuracy for computer-use agents.

---

#### 21st October 2025

[KAT-Coder Technical Report](http://arxiv.org/abs/2510.18779)

- KAT-Coder: introduces a large-scale agentic code model trained through a multi-stage curriculum, including Mid-Term Training (enhances reasoning, planning, reflection), Supervised Fine-Tuning (SFT) (constructs diverse dataset), Reinforcement Fine-Tuning (RFT) (optimizes policy with rewards), and Reinforcement Learning (RL) (adapts to production IDEs).
- The framework addresses the gap between static text-based training and dynamic real-world agentic execution by progressively enhancing cognitive and operational competence.
- KAT-Coder achieves robust tool-use reliability, instruction alignment, and long-context reasoning, forming a deployable foundation for intelligent coding agents.

---

[Fetch.ai: An Architecture for Modern Multi-Agent Systems](http://arxiv.org/abs/2510.18699)

- Fetch.ai Architecture: introduces a multi-layered architecture for modern multi-agent systems, integrating a Foundational Layer (underlying decentralized ledger, agent registry, naming service, economic token), a Development Layer (event-driven agent framework, SDK), a Deployment and Monitoring Layer (hosting, mailbox, monitoring, marketplace), and an Orchestration Layer (agentic LLM, task decomposition, search & discovery) to provide a robust, scalable, and decentralized platform.
- This architecture addresses limitations of current LLM-based agent frameworks by providing on-chain trust, verifiable identities, standardized communication protocols, and economic coordination mechanisms for autonomous agents.
- The framework enables the development, deployment, and operation of sophisticated multi-agent systems, allowing autonomous agents to securely discover, communicate, and transact in a decentralized marketplace.

---

[VAPU: System for Autonomous Legacy Code Modernization](http://arxiv.org/abs/2510.18509)

- VAPU (Verifying Agent Pipeline Updater): introduces an LLM-based multi-agent system designed to autonomously modernize legacy web application code by updating files in phases, simulating a software development team.
- The system employs a Manager agent, a Task pipeline (with Prompt maker and Execution agents), a Verification agent, and a Finalizer agent to process user requirements and iteratively refine code.
- VAPU aims to provide a cost-effective solution for updating deprecated components, addressing challenges in legacy system maintenance, and improving code quality through self-division and self-feedback mechanisms.

---

[Heterogeneous Adversarial Play in Interactive Environments](http://arxiv.org/abs/2510.18407)

- HAP (Heterogeneous Adversarial Play): introduces an adversarial Automatic Curriculum Learning (ACL) framework that formalizes teacher-student interactions as a minimax optimization, including a task-generating instructor, a problem-solving learner, an interactive environment, bidirectional learning feedback, an adversarial reward mechanism, student's behavioral history, a task selection distribution, and a student policy.
- The framework enables a teacher agent to autonomously generate challenging tasks and adapt the curriculum based on real-time student performance, while a student agent strives to master these evolving challenges.
- This co-evolutionary process dynamically balances task complexity against learner proficiency, fostering robust knowledge consolidation and effective exploration without requiring handcrafted curricula.

---

[Joint Optimization of Cooperation Efficiency and Communication Covertness for Target Detection with AUVs](http://arxiv.org/abs/2510.18225)

- HMAPPO (Hierarchical Multi-Agent Proximal Policy Optimization): introduces a hierarchical multi-agent deep reinforcement learning framework for joint optimization of cooperation efficiency and communication covertness in underwater target detection using AUVs.
- The framework decomposes the problem into macro-level AUV scheduling and micro-level AUV trajectory control, leveraging a Centralized Training with Decentralized Execution (CTDE) paradigm.
- This approach enables adaptive covert cooperation while satisfying energy and mobility constraints, providing efficient and secure operation for multiple AUVs in dynamic underwater environments.

---

[SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection](http://arxiv.org/abs/2510.16219)

- SentinelNet: introduces a decentralized framework for proactive threat detection and mitigation in multi-agent LLM systems, utilizing adversarial trajectory generation, contrastive learning-based detector training, and dynamic ranking with bottom-k elimination.
- Each agent is equipped with a credit-based detector, trained on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking to suppress malicious communications.
- The framework achieves near-perfect detection of malicious agents and recovers system accuracy from compromised baselines, demonstrating strong generalizability across domains and attack patterns.

---

[EffiReasonTrans: RL-Optimized Reasoning for Code Translation](http://arxiv.org/abs/2510.18863)

- EffiReasonTrans: introduces a training framework for code translation, integrating a Data Synthesis Stage (generates reasoning-augmented data), a Supervised Fine-Tuning Stage (initializes model with reasoning), and an Execution-Based Reinforcement Learning Stage (optimizes for accuracy/latency) to balance accuracy and inference latency.
- The framework synthesizes high-quality reasoning-augmented data (EffiReasonTrans-Data) using a powerful Reasoning LLM (DeepSeek-R1), then fine-tunes a Base LLM (DeepSeek-R1-Distill-Qwen-1.5B), and finally applies reinforcement learning with a dual-objective Reward Strategy (Execution-based Reward and Length-based Reward).
- This approach consistently improves translation accuracy while reducing generated tokens and inference latency, demonstrating effectiveness in multilingual and agent-based settings.

---

[An Encoder-Decoder Foundation Chemical Language Model for Generative Polymer Design](http://arxiv.org/abs/2510.18860)

- POLYT5 (An Encoder-Decoder Foundation Chemical Language Model for Generative Polymer Design): introduces a T5-based LLM, pre-trained on 100 million polymer structures in PSELFIES representation, enabling property prediction and targeted polymer generation, and integrated into an agentic AI framework for natural language interaction.
- The framework leverages its fine-tuned property prediction models for thermal, electronic, and solubility properties, and generative design models to create hypothetical polymers conditioned on desired properties, such as glass transition temperature.
- An agentic AI framework, featuring a general-purpose LLM as a controller and a Streamlit interface, enhances accessibility by automating input handling, format conversion, and model selection for both property prediction and generative design tasks.

---

[Search Self-play: Pushing the Frontier of Agent Capability without Supervision](http://arxiv.org/abs/2510.18821)

- SSP (Search Self-play): introduces a self-evolving reinforcement learning approach for deep search agents, where a single LLM policy acts as both a question proposer and a problem solver, co-evolving their capabilities through competition and cooperation.
- The proposer generates challenging search queries with verifiable ground-truth, while the solver attempts to answer them using multi-turn reasoning and external search tools.
- The framework incorporates RAG verification, rule-based filtering, and a periodically reset replay buffer to ensure high-quality training tasks and stable co-evolution without human supervision.

---

[Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](http://arxiv.org/abs/2510.18586)

- Tokencake: introduces a KV-Cache-centric serving framework for LLM-based multi-agent applications, with Frontend API, Space Scheduler, Time Scheduler, Application Graph Definition, FuncNode, Performance Metadata, Dynamic Memory Partitioning, Hybrid Priority Metric, CPU Block Buffering, Gradual GPU Block Reservation, Event-Driven Offload, Predictive Upload, Benefit-Driven Policy, and Dynamic Forecasting Model, which co-optimizes scheduling and memory management through an agent-aware design to address KV Cache space contention and time underutilization.
- The framework utilizes a Frontend API to define multi-agent workflows as a Directed Acyclic Graph, enabling specialized schedulers to manage KV Cache lifecycle with application-level context.
- The Space Scheduler employs dynamic memory partitioning and a hybrid priority metric to shield critical agents from contention, while the Time Scheduler uses proactive offload and predictive upload mechanisms to repurpose GPU memory during function call stalls.

---

[CLASP: Cost-Optimized LLM-based Agentic System for Phishing Detection](http://arxiv.org/abs/2510.18585)

- CLASP (Cost-Optimized LLM-based Agentic System for Phishing Detection): introduces a novel multi-agent system for phishing detection that leverages LLM-based agents for URL, screenshot, and HTML analysis, combining their outputs to classify websites as phishing or legitimate.
- The system processes URLs or QR codes, employing specialized LLM-based agents to evaluate various web resource aspects, and utilizes a Progressive Analysis strategy for cost-effective and accurate detection.
- CLASP outperforms existing commercial solutions in recall and F1 score, demonstrating a robust and scalable approach for combating evolving cybersecurity threats while maintaining low operational costs.

---

[QuantEvolve: Automating Quantitative Strategy Discovery through Multi-Agent Evolutionary Framework](http://arxiv.org/abs/2510.18569)

- QuantEvolve: introduces an evolutionary multi-agent framework for automating quantitative trading strategy discovery, integrating a multi-dimensional Feature Map, an Island Population for parallel evolution, and a multi-agent system comprising Data, Research, Coding, and Evaluation Teams to generate and refine strategies.
- The framework leverages a hypothesis-driven multi-agent system to systematically explore the strategy space through iterative generation and evaluation, ensuring diverse and high-performing strategies adaptable to market shifts and investor preferences.
- QuantEvolve maintains population diversity through a Feature Map that organizes strategies by attributes and employs an Evolutionary Database and Insight Repository to store and refine knowledge across generations.

---

[The Trust Paradox in LLM-Based Multi-Agent Systems: When Collaboration Becomes a Security Vulnerability](http://arxiv.org/abs/2510.18563)

- TVP Experimental Framework: introduces the "Trust-Vulnerability Paradox" in LLM-based multi-agent systems, empirically validating that increased inter-agent trust amplifies leakage risk, and proposes defenses.
- The framework utilizes CK-Agents and SK-Agents, powered by various LLM backends and orchestration frameworks, to simulate collaboration scenarios with parameterized trust levels.
- It quantifies leakage using Over-Exposure Rate and Authorization Drift metrics, and evaluates Sensitive-Information Repartitioning and Guardian-Agent enablement as mitigation strategies.

---

[WEBDEVJUDGE: EVALUATING (M)LLMS AS CRITIQUES FOR WEB DEVELOPMENT QUALITY](http://arxiv.org/abs/2510.18560)

- WEBDEVJUDGE: introduces a systematic benchmark for assessing LLM-as-a-judge performance in web development, supporting both static and continuous interactive evaluation, and comprising data collection, rubric annotation, and a judge component with various evaluators, observations, and paradigms.
- The benchmark utilizes human preference labels over paired web implementations, annotated with structured and query-grounded rubrics to establish high-quality ground truth for evaluating LLMs, MLLMs, and agentic workflows.
- Experiments reveal a significant gap between LLM judges and human experts, stemming from fundamental model limitations like failures in recognizing functional equivalence, verifying task feasibility, and mitigating bias, highlighting challenges for automated evaluators in complex scenarios.

---

[WHEN YOUR AI AGENT SUCCUMBS TO PEER-PRESSURE: STUDYING OPINION-CHANGE DYNAMICS OF LLMS](http://arxiv.org/abs/2510.19107)

- LLM-driven Network Model: introduces a framework for auditing emergent socio-cognitive behaviors of multi-agent AI systems, utilizing Experiment Setup, Random Node Selection, Prompt Construction, LLM Query & Recommendation, Update Opinion, Check For Consensus, Next Node Selection, LLM Agents, Social Network, Cognitive Commitment Spectrum, and Discursive Frames to study how peer pressure influences LLM opinions across cognitive commitments.
- The research reveals that LLM agents exhibit a sigmoidal conformity pattern, with varying thresholds across models and a "persuasion asymmetry" where the cognitive effort to change an opinion depends on its initial valence and the targeted cognitive layer.
- This study uncovers a "dual cognitive hierarchy" where the stability of cognitive constructs inverts based on the direction of persuasion, demonstrating that LLM decision-making is governed by a fluid, context-dependent architecture rather than static logic.

---

[SOCIA-V: Textual Gradient Meets Multi-Agent Orchestration for Automated Simulator Generation](http://arxiv.org/abs/2510.18551)

- SOCIA-V (Simulation Orchestration for Computational Intelligence with Agents): introduces an end-to-end, agentic framework that treats simulator construction as instance optimization over code within a textual computation graph, including Data Analysis, Code Generation, Simulation Execution, Result Evaluation, and Feedback Generation agents.
- The framework unifies multi-agent orchestration with a loss-aligned optimization view, converting brittle prompt pipelines into reproducible, constraint-aware simulator code generation.
- It employs Textual Gradient Descent (TGD) with Momentum and Projected Gradient Descent (PGD) for iterative code repair, ensuring high-fidelity, extrapolatable simulators across diverse domains.

---

[JAUNT: Joint Alignment of User Intent and Network State for QoE-centric LLM Tool Routing](http://arxiv.org/abs/2510.18550)

- JAUNT (Joint Alignment of User intent and Network state for QoE-centric Tool routing): introduces a framework that aligns user intent and real-time network states to maximize Quality of Experience (QoE) in LLM tool routing, utilizing Semantic Intent Inference, Network Latency Prediction, and Joint QoE-centric Tool Routing modules.
- The framework addresses limitations of current routing mechanisms by interpreting user intent, including semantic ambiguity and emotional expression, and integrating dynamic network conditions for adaptive tool selection.
- JAUNT employs LLM agents to construct network profiles, mapping numerical performance indicators into a semantic space to guide routing decisions and continuously updates user profiles based on QoE feedback.

---

[EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](http://arxiv.org/abs/2510.18546)

- EfficientNav: introduces an on-device object-goal navigation system that includes a Detection Model (generates semantic/distance information), Graph-based Navigation Map (organizes semantic/spatial information), Attention-based Memory Clustering (clusters objects into groups using LLM attention), Semantics-aware Memory Retrieval (selects relevant groups, prunes redundant map info), Discrete Memory Caching (manages KV cache for groups, avoids re-computation), and an LLM Planner (determines navigation sub-goals).
- The system enables efficient zero-shot object-goal navigation on local devices by addressing memory constraints and improving smaller LLM understanding of complex navigation maps.
- It achieves significant improvements in success rate and real-time latency reduction over GPT-4-based baselines by optimizing KV cache management and prompt efficiency.

---

[Crucible: Quantifying the Potential of Control Algorithms through LLM Agents](http://arxiv.org/abs/2510.18491)

- Crucible: introduces an LLM-driven framework for quantifying the Tuning Potential of control algorithms, with LLM Agent, Domain Knowledge Acquisition, Optimization Tools, Control Algorithm Interface, Action and Feedback Loop, Differential Developer Capability Simulation, Performance Characteristic Vector, Unified Environment Distance and Similarity Metric, Tuning Potential Metric, Test Environments, and Reference Algorithms, to systematically evaluate algorithmic adaptability.
- The framework employs an LLM-driven multi-level expert simulation agent to emulate developer tuning processes and defines a formalized metric for quantitatively assessing an algorithm's inherent adaptability across diverse environments.
- Crucible's approach moves beyond traditional performance evaluation by considering an algorithm's representational capacity and comprehensibility, guiding targeted redesign for improved performance and practical value.

---

[LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources](http://arxiv.org/abs/2510.18477)

- LAFA (Agentic LLM-Driven Federated Analytics): introduces an LLM-driven federated analytics framework that transforms natural language queries into optimized, privacy-preserving execution pipelines, utilizing a Querier, Server with Hierarchical Decomposer Agents, DAG Optimizer Agent, Aggregator, and Answerer Agent, and Target Clients performing FA Pipelines and Submission.
- This system addresses challenges in LLM-agent-based analytics by enabling efficient complex query processing in natural language with privacy preservation and reducing computational overhead through a hierarchical multi-agent architecture.
- LAFA ensures correct FA operation sequencing and optimizes workflows by eliminating redundant operations, making it suitable for real-world privacy-preserving data analytics over decentralized data sources.

---

[PROBABILISTIC MODELING OF INTENTIONS IN SOCIALLY INTELLIGENT LLM AGENTS](http://arxiv.org/abs/2510.18476)

- STOM (Stochastic Theory-of-Mind): introduces a probabilistic intent modeling framework for LLM agents in multi-turn social dialogue, which includes an Intention Model (generates, updates belief distributions), a Likelihood Model (estimates action probability), a Confidence-Aware Policy (selects actions based on uncertainty), and a Belief Distribution (represents partner's latent intentions).
- This framework maintains and dynamically updates a belief distribution over a partner's latent intentions, initialized from contextual priors and refined through likelihood estimation after each utterance.
- The evolving belief distribution provides contextual grounding for the policy, enabling adaptive dialogue strategies under uncertainty and improving multi-dimensional social performance without additional training.

---

[Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within the Response](http://arxiv.org/abs/2510.18434)

- CoCT (Chain-of-Conceptual-Thought): introduces a prompt-based paradigm that guides an LLM to first tag a concept (emotion, strategy, or topic) and then generate detailed content, facilitating deep and strategic thinking within a single utterance.
- This approach leverages a CoCT Prompt (structured instruction), which includes available Concepts (emotion, strategy, topic) and uses Special Tokens (concept tags) to explicitly denote conceptual transitions, enabling the LLM to structure its responses in open-domain conversations.
- The framework allows for multiple conceptual transitions within one response, mimicking human-like thinking and improving performance in tasks like emotional support conversations.

---

[Memory-Augmented State Machine Prompting: A Novel LLM Agent Framework for Real-Time Strategy Games](http://arxiv.org/abs/2510.18395)

- MASMP (Memory-Augmented State Machine Prompting): introduces a novel framework for LLM agents in real-time strategy games, integrating state machine prompting with a strategic memory module to achieve structured, coherent decision-making, utilizing an LLM-PySC2 Observation Extractor, Obs-Text converter, Memory Module, State Machine Prompting Module (comprising Macro-Strategic State Machine, Action Implementation Behavior Tree, and Supplementary Atomic Rules), LLMs, Strategy Extractor, Text-Action converter, Action Extractor, and Action Executor.
- The framework guides LLMs to emulate finite state machines and behavior trees through natural language prompts, while the memory module preserves strategic variables across decision cycles for persistent tactical coherence.
- MASMP achieves a 60% win rate against StarCraft II's hardest built-in AI (Lv7), demonstrating improved interpretability, generalization, and reliability over previous LLM-based baselines by bridging LLM flexibility with rule-based systems.

---

[InspectCoder: Dynamic Analysis-Enabled Self Repair through Interactive LLM-Debugger Collaboration](http://arxiv.org/abs/2510.18327)

- InspectCoder: introduces an agentic program repair system that empowers LLMs to actively conduct dynamic analysis via interactive debugger control, utilizing a Program Inspector agent for dynamic analysis, a Patch Coder agent for patch generation and validation, and InspectWare middleware for debugger interaction.
- The framework enables strategic breakpoint placement, targeted state inspection, and incremental runtime experimentation within stateful debugger sessions, moving beyond blind trial-and-error to systematic root cause diagnosis.
- InspectCoder achieves significant improvements in repair accuracy and bug-fix efficiency over baselines by adaptively inspecting and perturbing relevant intermediate states at runtime, guided by immediate debugger feedback.

---

[Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming](http://arxiv.org/abs/2510.18314)

- Genesis: introduces an agentic red-teaming framework for LLM web agents, featuring an Attacker, Scorer, Strategist, and Strategy Library, designed to systematically discover, summarize, and evolve attack strategies.
- The framework employs a genetic algorithm within the Attacker to evolve strategies, an LLM-powered Scorer for feedback, and an LLM-based Strategist to refine the continuously growing Strategy Library.
- This closed-loop system automates the red-teaming process by mimicking human expert learning, enabling dynamic adaptation and transferability of attack knowledge across diverse web environments.

---

[PROACTIVE REASONING-WITH-RETRIEVAL FRAMEWORK FOR MEDICAL MULTIMODAL LARGE LANGUAGE MODELS](http://arxiv.org/abs/2510.18303)

- MED-RWR (Multimodal Medical Reasoning-with-Retrieval framework): introduces a proactive multimodal reasoning-with-retrieval framework for medical MLLMs, leveraging its Policy Model, Medical Knowledge Base, Reference Model, Reward Design, Confidence-Driven Image Re-retrieval (CDIR), Multimodal Medical KB, Retriever, Input, and Output to enhance diagnostic accuracy by actively integrating external knowledge.
- The framework employs a two-stage reinforcement learning strategy with tailored rewards, including accuracy, format, query semantic (visual and textual), and confidence gain, to stimulate effective retrieval and reasoning.
- CDIR further augments the system by triggering image re-retrieval from a multimodal knowledge base during inference when low prediction confidence is detected, addressing insufficient information from initial text-based retrieval.

---

[Food4All: A Multi-Agent Framework for Real-time Free Food Discovery with Integrated Nutritional Metadata](http://arxiv.org/abs/2510.18289)

- Food4All (Multi-Agent Framework): introduces a multi-agent framework for real-time, context-aware free food retrieval, unifying cross-platform data aggregation, a reinforcement learning algorithm, and an online feedback loop to deliver nutritionally annotated food recommendations.
- The framework employs a dual-agent system with a Planner Agent for hierarchical task decomposition and an Executor Agent for tool-grounded execution, addressing limitations of existing systems like incomplete information and lack of personalization.
- Food4All dynamically adapts retrieval policies to evolving user needs through an online learning loop, ensuring reliable and practical food access information for food-insecure populations.

---

[When Old Meets New: Evaluating the Impact of Regression Tests on SWE Issue Resolution](http://arxiv.org/abs/2510.18270)

- TESTPRUNE: introduces an automated technique that leverages issue tracker reports and strategically reuses regression tests for bug reproduction and patch validation, utilizing an LLM, suspicious function localization, test file retrieval and coverage generation, and a greedy algorithm to produce minimized regression tests.
- This approach addresses the challenge of large test suites exceeding LLM context limits by minimizing the regression suite to a small, highly relevant subset of tests, thereby improving efficiency and reliability in LLM-based debugging workflows.
- The minimized regression tests generated by the framework enhance reproduction test generation by providing focused guidance and improve patch selection and validation by ensuring relevance to the issue, leading to increased issue reproduction and resolution rates.

---

[SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents](http://arxiv.org/abs/2510.17017)

- SafeSearch: introduces an RL-based alignment framework that jointly optimizes safety and utility for LLM-based search agents, incorporating mixed training with general QA and red-teaming datasets, and utilizing both final-output safety/utility rewards and a novel query-level shaping term.
- The framework explicitly rewards policy-compliant helpfulness and penalizes unsafe queries, aiming to reduce harmful outputs while maintaining or improving QA performance.
- SafeSearch significantly reduces agent harmfulness by over 70% across red-teaming datasets and matches the QA performance of utility-only finetuned agents, demonstrating the effectiveness of its query-level reward in balancing safety and utility.

---

[Cultural Alien Sampler: Open-ended art generation balancing originality and coherence](http://arxiv.org/abs/2510.20849)

- CAS (Cultural Alien Sampler): introduces a concept-selection method that explicitly separates compositional fit from cultural typicality, with a Concept Coherence Model (scores concept co-occurrence), a Cultural Context Model (estimates concept combination typicality), and a Scoring Function (balances coherence and typicality).
- The framework integrates CAS into an Open-ended Art Agent, which includes an Inspiration Module, a Prompt Compositor (GPT-40), an Image Generator (gpt-image-1), and a Novelty Score (evaluates originality/harmony using text and image embedding models).
- This approach enables autonomous agents to generate ideas that maintain internal consistency while deviating from learned conventions, outperforming LLM baselines in originality and harmony and exploring a broader conceptual space.

---

[AndroidControl-Curated: Revealing the True Potential of GUI Agents through Benchmark Purification](http://arxiv.org/abs/2510.18488)

- AndroidControl-Curated Pipeline: introduces a systematic, semi-automated benchmark purification pipeline and a novel reinforcement learning training paradigm for GUI agents, including optimized grounding, multi-model filtering, LLM review and rewrite, human expert verification, and GRPO training with Gaussian rewards and ratio optimization.
- This pipeline creates AndroidControl-Curated, a refined benchmark that accurately evaluates GUI agents, and trains Magma-R1, a compact 3B model achieving state-of-the-art performance on complex GUI tasks.
- The research demonstrates that benchmark quality is more critical than model scale for GUI agent evaluation, enabling on-device GUI agents to be closer to practical deployment.

---

[UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action](http://arxiv.org/abs/2510.17790)

- UltraCUA (A Foundation Model for Computer Use Agents with Hybrid Action): introduces a foundation model that seamlessly integrates GUI primitives with high-level programmatic tool calls, leveraging an automated pipeline for tool acquisition, a synthetic data engine for verifiable tasks, a large-scale hybrid action trajectory collection, and a two-stage training pipeline.
- This approach enables strategic alternation between low-level GUI actions and high-level programmatic tool calls, reducing error propagation and maintaining execution efficiency for computer-use agents.
- UltraCUA achieves state-of-the-art performance on real-world benchmarks, demonstrating improved success rates and cross-platform generalization by effectively bridging primitive GUI interactions and programmatic intelligence.

---

#### 20th October 2025

[Enterprise Deep Research: Steerable Multi-Agent Deep Research for Enterprise Analytics](http://arxiv.org/abs/2510.17797)

- EDR (Enterprise Deep Research): introduces a multi-agent system for enterprise analytics, integrating a Master Research Agent, ToDo Manager, Specialized Agents (including search domain and enterprise workflow tools), a Reflection Mechanism, and a Research Report component, with optional human steering.
- The framework enables automated report generation, real-time streaming, and seamless enterprise deployment, outperforming state-of-the-art agentic systems on open-ended benchmarks without human steering.
- EDR provides transparent, steerable research through dynamic context engineering, allowing human users to guide the agent's reasoning trajectory and task management during execution.

---

[A Brain Cell Type Resource Created by Large Language Models and a Multi-Agent AI System for Collaborative Community Annotation](http://arxiv.org/abs/2510.17064)

- BRAINCELL-AID (Brain Cell type Annotation and Integration using Distributed AI): introduces a novel multi-agent AI system for collaborative community annotation of brain cell types, utilizing an Agentic Network, Query Agent, Fine-tuned LLM, GPTON, Literature Agent, RAG Agent, Web Portal, and Neuroscience Communities / Human Feedback to generate and refine biologically grounded annotations.
- The system leverages fine-tuned LLMs and retrieval-augmented generation to overcome limitations of traditional annotation methods, providing high-quality, literature-backed descriptions for over 20,000 brain cell type-specific marker gene sets.
- BRAINCELL-AID enhances annotation accuracy, supports testable hypothesis generation, and fosters human-AI collaboration through its interactive web portal, advancing neuroscience discovery.

---


[Semantic Joint Source Channel Coding for Distributed Subsurface Imaging in Multi-Agent Systems](http://arxiv.org/abs/2510.17695)

- Semantic JSCC AirComp (Semantic Joint Source Channel Coding with Over-the-Air Computation): introduces a framework that integrates semantic communication into multi-agent system (MAS) exploration, applying semantic JSCC with AirComp for distributed function computation (DFC) in cooperative subsurface imaging using the Adapt-Then-Combine Full Waveform Inversion (ATC-FWI) algorithm.
- This framework employs Neural Network encoders to compress and channel code observations from neighboring agents, an AirComp channel to sum transmitted symbols, and a Neural Network semantic decoder to reconstruct a semantic variable, leveraging local side information at the receiver.
- The system enhances overall task performance by adapting communication strategies to the exploration methodology, demonstrating improved bandwidth efficiency and imaging accuracy in noisy inter-agent communication links.

---

[ImaGGen: Zero-Shot Generation of Co-Speech Semantic Gestures Grounded in Language and Image Input](http://arxiv.org/abs/2510.17617)

- ImaGGen: introduces a zero-shot system for co-speech semantic gesture generation, with an Image Feature Analysis Pipeline (identifies objects), a Semantic Matching Pipeline (links text to visuals), and a Realization Engine (synthesizes gestures) to produce iconic, deictic, and beat gestures from language and image input.
- The system extracts object properties like shape, symmetry, and alignment from images, matches these visual details to spoken text, and then synthesizes gestures using an inverse kinematics engine, layering them with co-generated beat gestures.
- A user study demonstrated that the generated gestures significantly improved participants' ability to identify object properties in ambiguous speech scenarios, confirming their interpretability and communicative value for virtual agents.

---

[Cybersecurity AI: Evaluating Agentic Cybersecurity in Attack/Defense CTFs](http://arxiv.org/abs/2510.17521)

- CAI (Cybersecurity AI) Parallel Execution Framework: introduces an empirical evaluation of AI agents in Attack/Defense CTF scenarios, deploying autonomous offensive and defensive agents concurrently in a shared target environment to assess their capabilities under various operational constraints.
- The framework leverages LLMs to power specialized agents, enabling fine-grained control over their configuration, context, and objectives for direct comparison of attack and defense performance.
- This study challenges claims of inherent AI attacker advantage by demonstrating that defensive effectiveness critically depends on success criteria, highlighting the importance of availability-preserving defense in real-world cybersecurity operations.

---

[Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents](http://arxiv.org/abs/2510.17491)

- Industry Agent Framework: introduces a five-level capability maturity model for LLM-driven industry agents, detailing their evolution across Memory, Planning, and Tool Use pillars.
- This framework categorizes agents from simple process execution systems (L1) to adaptive social systems (L5), driven by advancements in core technologies.
- It provides a roadmap for understanding and building next-generation industry agents by linking technological evolution with practical applications and evaluation.

---

[AGENTIC REINFORCEMENT LEARNING FOR SEARCH IS UNSAFE](http://arxiv.org/abs/2510.17431)

- ARLS (Agentic Reinforcement Learning for Search): introduces a study evaluating the safety of RL-trained search models, which include a Policy LLM, Search Engine, Reward Function, Reference Model, System Prompt, and specific Tokens, revealing their vulnerability to jailbreaking attacks, which are assessed by an LLM Evaluator using a Harmful Instructions Dataset, and categorized into Search Attack and Multi-search Attack.
- The paper demonstrates that these models, despite inheriting refusal behavior from instruction tuning, can be easily exploited by forcing early searches, leading to cascades of harmful queries and answers.
- This research highlights a critical weakness in current RL training objectives that prioritize effective query generation over safety, necessitating the development of safety-aware RL pipelines.

---

[DIVERSE PLANNING WITH SIMULATORS VIA LINEAR TEMPORAL LOGIC](http://arxiv.org/abs/2510.17418)

- FBILTL (Forbid Behaviour IterativeLTL): introduces a diverse planner for simulation-based planning problems, leveraging its Behaviour Sorts Suite (BSS) for diversity modeling and Linear Temporal Logic (LTL) to define semantic diversity criteria, which are integrated into the search process using a modified Iterated Width (IW(i)) Planner as a BehaviourGeneratorx and a Simulator for environment interaction, with a PlanGeneratorx for additional plan generation.
- This framework addresses the limitation of existing diverse planning approaches that often produce semantically identical solutions by ensuring the generation of semantically distinct plans based on user-defined diversity features.
- The approach demonstrates the feasibility of semantically-guided diverse planning in complex, non-symbolic simulation-based environments, offering a significant advantage over traditional declarative models.

---

[ALPINE: A Lightweight and Adaptive Privacy-Decision Agent Framework for Dynamic Edge Crowdsensing](http://arxiv.org/abs/2510.17162)

- ALPINE (A Lightweight and Adaptive Privacy-Decision Agent Framework for Dynamic Edge Crowdsensing): introduces a closed-loop control system that empowers terminal devices to autonomously adjust differential privacy levels in real time, balancing privacy gains, data utility, and energy cost.
- The framework includes a Risk Perception Module, a Privacy Decision Module, a Privacy Execution Module, and a Performance Verification Module, operating across mobile terminal devices and an edge computing server.
- It leverages a LightAE for channel risk detection and a TD3 agent for dynamic privacy budget allocation, with feedback from the edge server for continuous policy refinement.

---

[Learning After Model Deployment](http://arxiv.org/abs/2510.17160)

- PLDA (Post-deployment Learning based on Linear Discriminant Analysis): introduces Autonomous Learning after Model Deployment (ALMD), a paradigm enabling AI agents to continuously learn new knowledge autonomously after model deployment, utilizing a pre-trained model, LDA, and incremental updates of class means.
- The framework performs dynamic OOD detection using Mahalanobis distance or Relative Mahalanobis distance, and incrementally learns new classes by updating their class means while keeping a shared covariance matrix fixed, thus avoiding catastrophic forgetting.
- This approach allows for efficient, online learning from streaming data without human engineers, adapting to open and dynamic environments by expanding its set of known classes.

---

[Digitization Can Stall Swarm Transport: Commensurability Locking in Quantized-Sensing Chains](http://arxiv.org/abs/2510.17117)

- Robotic Swarm Model: introduces a minimal model for autonomous robotic swarms that self-organize spacing and follow local gradients using quantized digital sensors, investigating collective response, fractional transport, and commensurability locking.
- The model incorporates stochasticity (λrand), quantized sensing (λsens) leading to motion bias, and pairwise inter-agent interactions (λint) to maintain swarm formation.
- The research reveals that collective transport can stall due to commensurability locking, a number-theoretic condition, and explores how swarm topology affects transport in higher dimensions.

---

[From AutoRecSys to AutoRecLab: A Call to Build, Evaluate, and Govern Autonomous Recommender-Systems Research Labs](http://arxiv.org/abs/2510.18104)

- AutoRecLab (Autonomous Recommender-Systems Research Lab): introduces a vision for an integrated system that automates the entire research lifecycle in recommender systems, from problem ideation to manuscript drafting, utilizing LLM-driven components and automated experimentation.
- This framework aims to expand beyond current AutoRecSys tools by enabling autonomous generation of research questions, experimental designs, and manuscript writing, while maintaining rigorous provenance records.
- The paper calls for the RecSys community to build prototypes, establish benchmarks, embrace AI-generated submissions, develop attribution standards, and foster interdisciplinary dialogue for responsible integration of automated research.

---

[SPACER: SELF-PLAY ANCHORING WITH CENTRALIZED REFERENCE MODELS](http://arxiv.org/abs/2510.18060)

- SPACER (Self-Play Anchoring with Centralized Reference Models): introduces a framework that leverages a pretrained Tokenized Reference Model (πref) to guide a Decentralized Model (πθ) in self-play, with all Batched Simulations, Full Scene Context, Agents, Agent Rollouts, Loss Function (L(θ)), Task Performance (LPPO), Human-likeness Reward (r_humanlike), and Distributional Alignment (DKL) components, where the framework anchors decentralized self-play policies to human driving distributions using likelihood rewards and KL divergence for scalable, realistic multi-agent simulation.
- The Tokenized Reference Model (πref) provides a human-likeness distributional signal and likelihood rewards, while the Decentralized Model (πθ) is the self-play policy trained via reinforcement learning on local observations.
- The Loss Function (L(θ)) combines a Task Performance (LPPO) objective with a Human-likeness Reward (r_humanlike) and Distributional Alignment (DKL) term to balance task success with realistic, human-like behaviors.

---

[LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior](http://arxiv.org/abs/2510.18155)

- LLM-Based Multi-Agent System: introduces an LLM-powered multi-agent simulation framework, including LLM Response System (Manages LLM interactions), Memory System (Manages agent memories), Main Loop (Orchestrates simulation flow), Agents (Simulated entities), and Environment (Virtual world), designed to simulate consumer decision-making and social dynamics for marketing strategy evaluation.
- The framework utilizes DeepSeek-V3 LLM-powered generative agents that plan daily schedules, manage resources, shop, converse, and make social commitments within a virtual town environment.
- This approach provides marketers with a scalable, low-risk tool for pre-implementation testing, reducing reliance on time-intensive post-event evaluations and lowering the risk of underperforming campaigns.

---

[Learning from Generalization Patterns: An Evaluation-Driven Approach to Enhanced Data Augmentation for Fine-Tuning Small Language Models](http://arxiv.org/abs/2510.18143)

- PaDA-Agent (Pattern-guided Data Augmentation Agent): introduces an evaluation-driven approach for fine-tuning SLMs, where a Central Orchestrator coordinates a Pattern Analysis Agent, Data Generation Agent, and Quality Control Agent to iteratively enhance a Fine-tuned SLM by augmenting Training Data with synthetic data derived from Validation Data insights, guided by Error Patterns, Augmentation Strategies, and Quality Control Feedback, resulting in Augmented Training Data.
- The framework systematically analyzes validation failures to discover error patterns, drafts targeted augmentation strategies, and generates synthetic data with automated quality control, directly addressing generalization errors.
- This multi-agent system significantly outperforms state-of-the-art LLM-based data augmentation approaches, yielding consistent performance gains for SLMs across various tasks, especially in low-data regimes.

---

[BlueCodeAgent: A BLUE TEAMING AGENT ENABLED BY AUTOMATED RED TEAMING FOR CODEGEN AI](http://arxiv.org/abs/2510.18131)

- BlueCodeAgent: introduces an end-to-end blue teaming agent enabled by automated red teaming, integrating an Automated Red Teaming Pipeline (generates diverse risky instances and knowledge), a Knowledge Base (stores red-teaming data), and a BlueCodeAgent (main agent for defense) which includes Constitution Generation (summarizes knowledge into actionable rules), a Static Analyzer (performs initial vulnerability analysis), a Dynamic Analyzer (generates test cases for runtime verification), a Code Runner (executes code in sandbox), and a Final Analyzer (integrates analysis for final judgment).
- The framework unifies automated red teaming to generate diverse risky instances, which are distilled into actionable constitutions that guide the blue teaming agent to detect unsafe textual inputs and code outputs.
- BlueCodeAgent leverages dynamic testing to validate vulnerability claims, effectively reducing false positives and over-conservative judgments, thereby achieving robust and precise risk mitigation across various code-generation security tasks.

---

[Investigating the Impact of Dark Patterns on LLM-Based Web Agents](http://arxiv.org/abs/2510.18113)

- LiteAgent: introduces a framework for evaluating LLM-based web agents against dark patterns, utilizing TrickyArena as a controlled web environment with customizable dark patterns and tasks, and LiteAgent's components for automated agent execution, interaction logging, and performance validation.
- The framework captures comprehensive logs and screen-recordings of agent interactions, enabling systematic assessment of dark pattern susceptibility and task completion rates across various LLM-based generalist web agents.
- The study reveals that web agents are susceptible to dark patterns, with higher-performing agents being more vulnerable, and that both LLM choice and agent architecture significantly influence susceptibility, highlighting the need for holistic defense mechanisms.

---

[Does Reasoning Help LLM Agents Play Dungeons and Dragons? A Prompt Engineering Experiment](http://arxiv.org/abs/2510.18112)

- LLM-based D&D Action Generation Experiment: introduces an experimental setup comparing an Instruct Model (LLaMA-3.1-8B-Instruct) and a Reasoning Model (DeepSeek-R1-Distill-LLaMA-8B) for generating Dungeons & Dragons player actions as Avrae Discord bot commands, utilizing prompt engineering, the FIREBALL dataset, and various evaluation metrics.
- This research investigates the impact of prompt design on LLMs' ability to predict structured actions during D&D combat, focusing on command generation for the Avrae Discord bot.
- The study highlights that specific instructions in prompts significantly affect model output, concluding that instruct models are sufficient for this task and can outperform reasoning models, especially for smaller LLMs.

---

[CompactPrompt: A Unified Pipeline for Prompt and Data Compression in LLM Workflows](http://arxiv.org/abs/2510.18043)

- CompactPrompt: introduces a unified pipeline for prompt and data compression in LLM workflows, with Initialization (sets up pipeline), Token Probability Construction (computes token likelihoods), Hybrid Scoring (LLM+Programmatic) (combines LLM and programmatic scores), Compression Engine (applies pruning, abbreviation, quantization), Semantic Similarity Analysis (evaluates compressed output fidelity), Metrics Computation (calculates performance metrics), Result Assembly (gathers compressed outputs), and Final Context (generates final compressed context), aiming to reduce token usage and inference costs while preserving output quality.
- The pipeline integrates hard prompt compression, textual n-gram abbreviation for documents, and numerical quantization for structured data, addressing diverse input types without model retraining.
- CompactPrompt achieves up to 60% token reduction and maintains or improves QA accuracy on benchmarks like TAT-QA and FinQA, making LLM workflows more efficient and cost-effective.

---

[OPTAGENT: Optimizing Multi-Agent LLM Interactions Through Verbal Reinforcement Learning for Enhanced Reasoning](http://arxiv.org/abs/2510.18032)

- OPTAGENT (Optimizing Multi-Agent LLM Interactions Through Verbal Reinforcement Learning for Enhanced Reasoning): introduces a multi-agent verbal reinforcement learning framework that dynamically constructs and refines multi-agent collaboration structures, including Profiled LLM Agents (individual LLMs with distinct personas), a Multi-Agent Collaboration Graph (representing agent interactions), LLMreflect (a feedback agent), LLMact (an action agent), and a Majority Voting Strategy (for final decision-making), where it optimizes interaction patterns and communication quality for enhanced reasoning.
- The framework leverages verbal reinforcement learning with meta-agents (LLMreflect and LLMact) to evaluate and adapt the collaboration graph, ensuring effective information flow and robust problem-solving across diverse reasoning tasks.
- By explicitly considering communication quality and dynamically updating connection scores, OPTAGENT significantly outperforms single-agent prompting and state-of-the-art multi-agent frameworks on various reasoning tasks.

---

[BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?](http://arxiv.org/abs/2510.18003)

- BadScientist: introduces a framework that evaluates whether fabrication-oriented paper generation agents can deceive multi-model LLM review systems, including a Paper Agent (generates fabricated research papers), a Review Agent (evaluates papers using multiple LLMs), and an Analysis System (aggregates outcomes and calibrates thresholds).
- The framework employs five presentation-manipulation strategies (TooGoodGains, BaselineSelect, StatTheater, CoherencePolish, ProofGap) to generate unsound papers without real experiments, which are then evaluated by LLM reviewers calibrated on real conference data.
- Findings reveal systematic vulnerabilities where fabricated papers achieve high acceptance rates (up to 82.0%) and exhibit a "concern-acceptance conflict," indicating that LLM reviewers frequently flag integrity issues yet still assign acceptance-level scores.

---

[FABRIC: FRAMEWORK FOR AGENT-BASED REALISTIC INTELLIGENCE CREATION WEAVING SYNTHETIC ENTERPRISE DATA FOR TRAINING AUTONOMOUS AGENTS](http://arxiv.org/abs/2510.17995)

- FABRIC (Framework for Agent-Based Realistic Intelligence Creation): introduces a unified, modular framework for generating structured, executable, and validated tool-use data from LLMs, without human supervision, to train autonomous agents.
- The framework leverages four modular pipelines—RecordSynth, DAGFirstGeneration, MultiTurnDialogueSynth, and AgenticRecordRollout—to synthesize agentic data across varying granularities, from end-to-end trajectories to atomic function calls.
- It integrates constrained generation formats, JSON-schema validation, and judge-based filtering to ensure logical consistency, execution fidelity, and schema validity of the generated synthetic datasets, advancing robust tool use for agentic LLMs.

---

[Executable Knowledge Graphs for Replicating AI Research](http://arxiv.org/abs/2510.17795)

- XKG (Executable Knowledge Graphs): introduces a modular and pluggable knowledge base for AI research replication, comprising various nodes (Paper Node, Technique Node, Code Node) and edges (Structural Edge, Implementation Edge), constructed through automated processes (Corpus Curation, Technique Extraction, Code Modularization, Knowledge Filtering), and leveraged by LLM agents (BasicAgent, IterativeAgent, PaperCoder) for planning and implementation, using Query Retrieval and an LLM-based Verifier, with 04-mini and DeepSeek-V3 as core LLMs.
- The framework automatically integrates technical insights, code snippets, and domain-specific knowledge extracted from scientific literature to support multi-granular retrieval and reuse.
- XKG significantly enhances AI research replication by providing structured, executable knowledge, enabling agents to retrieve, reason about, and assemble precise artifacts for faithful reproduction.

---

[Evaluating Medical LLMs by Levels of Autonomy: A Survey Moving from Benchmarks to Applications](http://arxiv.org/abs/2510.17764)

- Levels of Autonomy (L0-L3): introduces a framework for evaluating medical LLMs by categorizing their capabilities into four distinct levels, spanning informational tools to supervised agents.
- This framework aligns existing benchmarks and metrics with permitted actions and associated risks at each autonomy level, providing a structured approach for evaluation and oversight.
- The survey moves beyond simple score-based claims towards credible, risk-aware evidence for safe and reliable clinical deployment of LLM-based systems.

---

[ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling](http://arxiv.org/abs/2510.17603)

- ShapeCraft: introduces a multi-agent framework for text-to-3D generation, employing Parser, Coder, and Evaluator LLM agents that interact with a Graph-based Procedural Shape (GPS) representation to produce structured, textured, and interactive 3D assets.
- The framework hierarchically parses user input into a GPS, iteratively refines procedural modeling and painting, and enables post-modeling interactions like shape editing and animation.
- ShapeCraft demonstrates superior performance in generating geometrically accurate and semantically rich 3D assets compared to existing LLM-based methods, highlighting its potential for broader interactive applications.

---

[SpecAgent: A Speculative Retrieval and Forecasting Agent for Code Completion](http://arxiv.org/abs/2510.17925)

- SpecAgent (Speculative Retrieval and Forecasting Agent): introduces a framework that improves code completion by proactively generating and storing speculative context blocks, which include retrieved relevant code contexts and predicted future function implementations, to be used by a Code Completion Model (LLM) for generating code completions for a target file.
- This approach shifts costly context computation from inference time to asynchronous indexing time, leveraging Indexing-Time Tools to analyze the Repository, thereby significantly reducing latency and enhancing code generation quality.
- The framework also addresses the "future context leakage" problem in existing benchmarks by introducing a synthetic, leakage-free evaluation environment for more realistic performance assessment.

---

[BREAKING AND FIXING DEFENSES AGAINST CONTROL-FLOW HIJACKING IN MULTI-AGENT SYSTEMS](http://arxiv.org/abs/2510.17276)

- CONTROLVALVE: introduces a defense framework for multi-agent systems, with Orchestrator, LLM (CFG generation), Lark Parser, LLM (Rule Generation), LLM Judge, Control-Flow Graph (CFG), Edge-Specific Rules, Sub-Agents, User, Adversary, and Conversation State components, designed to prevent control-flow hijacking by generating and enforcing permitted control-flow graphs and contextual rules for agent invocations.
- The framework operates by first generating a task-specific control-flow graph and edge-specific contextual rules using LLMs during a planning stage, before any untrusted content is ingested.
- During execution, the Orchestrator and an LLM Judge enforce compliance with these predefined graphs and rules, blocking or replanning if agent transitions violate the established security policies.

---

[Coinvisor: An RL-Enhanced Chatbot Agent for Interactive Cryptocurrency Investment Analysis](http://arxiv.org/abs/2510.17235)

- COINVISOR: introduces a reinforcement learning-based web chatbot for cryptocurrency investment analysis, with an RL-tuned LLM Caller, Data Analytics Tools, Report Agents, and a Reasoning Model to provide comprehensive, real-time insights.
- The system employs Reinforcement Learning for multi-step tool selection, enabling adaptive analysis of dynamic web content and flexible integration of heterogeneous data sources.
- COINVISOR addresses limitations of static LLM agents and fragmented data platforms by offering interactive, multi-dimensional analysis and decision support through its multi-agent framework.

---

[WHICH LLM MULTIAGENT PROTOCOL TO CHOOSE?](http://arxiv.org/abs/2510.17149)

- ProtocolBench and ProtocolRouter: introduces a system for evaluating and dynamically selecting LLM multi-agent communication protocols, comprising a benchmark for performance and robustness, and a learned router for scenario-specific protocol assignment.
- The system addresses the challenge of protocol selection by systematically comparing A2A, ACP, ANP, and Agora across four dimensions: task success, latency, overhead, and failure robustness.
- ProtocolRouter enhances multi-agent system reliability and efficiency by selecting optimal protocols per module based on requirements and runtime signals, outperforming single-protocol baselines in targeted settings.

---

[Do LLMs Recognize Your Latent Preferences? A Benchmark for Latent Information Discovery in Personalized Interaction](http://arxiv.org/abs/2510.17132)

- Tri-Agent Framework: introduces a unified benchmark for evaluating LLMs' ability to discover and utilize hidden user attributes through multi-turn interaction, featuring a User (simulates person, hidden preferences), an Assistant (LLM under evaluation, active), and a Judge (evaluates output alignment).
- This benchmark spans three progressively realistic tasks: the 20 Questions game, Personalized Question Answering, and Personalized Text Summarization, designed to assess latent information discovery and personalization accuracy.
- The framework enables systematic, turn-level evaluation of questioning strategies and reasoning efficiency, highlighting that effective preference inference remains an open frontier for building truly adaptive AI systems.

---

[Semantic Intelligence: A Bio-Inspired Cognitive Framework for Embodied Agents](http://arxiv.org/abs/2510.17129)

- SIDE (Semantic Intelligence-Driven Embodied) agent framework: introduces a bio-inspired cognitive framework for embodied agents, integrating a hierarchical semantic cognition architecture with a semantic-driven decision-making process to enable contextually adaptive interaction with the physical world.
- The framework operates through a closed perception-cognition-action loop, where the Semantic Cognitive Architecture builds semantic knowledge from multimodal sensor data, and the Semantic-Driven Decision Loop guides planning and action execution using this knowledge.
- This approach enhances embodied agents' ability to extract, represent, reason, and apply semantics, addressing limitations of current LLM-based agents in real-world environments by facilitating flexible planning, robust execution, and interpretable behavior.

---

[Verification-Aware Planning for Multi-Agent Systems](http://arxiv.org/abs/2510.17109)

- VERIMAP (Verification-Aware Planning for Multi-Agent Systems): introduces a framework for multi-agent collaboration with verification-aware planning, which includes a Planner (decomposes tasks, generates VFs), Executor (solves subtasks, produces outputs), Verifier (evaluates outputs, provides feedback), and Coordinator (orchestrates execution, manages context/retries).
- The framework integrates planning and verification by decomposing tasks into a Directed Acyclic Graph (DAG) of subtasks, where the planner specifies Structured and Named I/O and Verification Functions (VFs) at the subtask level.
- Executors produce JSON outputs verified by paired VFs, while the coordinator manages contexts, retries, and dynamic replanning to ensure reliable final results, enhancing robustness and interpretability.

---

[Structured Debate Improves Corporate Credit Reasoning in Financial AI](http://arxiv.org/abs/2510.17108)

- KPD-MADS (Karl Popper Debate-based Multi-Agent Debate System): introduces a framework for corporate credit reasoning that formalizes adversarial verification via a ten-step structured interaction protocol, including a debate subsystem with six LLM agents and an aggregator subsystem.
- The framework leverages a shared knowledge pool and web search capabilities to enable agents to reason over non-financial indicators and generate comprehensive, balanced analytical reports.
- KPD-MADS demonstrates superior reasoning quality and practical applicability compared to a single-agent system (NAS) by enhancing analytical rigor through structured agent interaction and iterative refinement of arguments.

---

[Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities in Large Language Models](http://arxiv.org/abs/2510.17098)

- MTI V.1 (Malicious Token Injection V.1): introduces a modular framework for cache-side attacks, perturbing stored key-value representations during LLM inference using Corruption Mechanisms (perturb stored key vectors), Adaptive Perturbations (optimizes injected noise), and Control Dimensions (tune perturbation characteristics), to systematically bias attention maps and alter downstream token predictions.
- The framework demonstrates that even small, structured perturbations to the KV cache can significantly degrade LLM performance across various tasks, including classification, question answering, summarization, RAG, and agentic reasoning.
- MTI V.1 establishes cache integrity as a critical, yet underexamined, vulnerability in LLM deployments, providing a reproducible threat model for future robustness research and highlighting the need for verifiable cache integrity.

---

[CONSISTENT ZERO-Shot IMITATION WITH CONTRASTIVE GOAL INFERENCE](http://arxiv.org/abs/2510.17059)

- CIRL (Consistent Zero-Shot Imitation with Contrastive Goal Inference): introduces a method for pre-training interactive agents in a self-supervised fashion, combining goal-conditioned contrastive RL pre-training, automatic goal sampling, and a mean field goal inference model to enable zero-shot imitation of expert demonstrations.
- The framework allows agents to autonomously propose and practice reaching their own goals during training, then at test time, infer an expert's goal from a single demonstration and execute a learned goal-conditioned policy to achieve it.
- By reframing inverse RL as a goal inference problem and coupling it with contrastive RL, CIRL learns transferable goal-conditioned policies that generalize across diverse task distributions without requiring expert data or rewards during pre-training.

---

[FineVision: Open Data Is All You Need](http://arxiv.org/abs/2510.17269)

- FineVision: introduces a meticulously collected, curated, and unified corpus of 24 million samples, utilizing a semi-automated, human-in-the-loop pipeline that includes raw sources ingestion, canonicalization, image and text cleaning, de-duplication, test-set decontamination, per-turn quality assessment, human checkpoints, LLM/VLM-as-a-judge, SSCD embeddings, LLM agents (GPT/Claude), a unified conversational schema, and a unified action space.
- The framework addresses the fragmentation and contamination of public datasets for vision-language models (VLMs) by unifying over 200 sources into 185 subsets with rigorous data hygiene and quality control.
- FineVision enables state-of-the-art performance for models trained on it, outperforming existing open mixtures across a broad evaluation suite, and supports novel GUI/agentic capabilities through its unified action space.

---

#### 19th October 2025

[DeepAnalyze: Agentic Large Language Models for Autonomous Data Science](http://arxiv.org/abs/2510.16872)

- DeepAnalyze-8B (Agentic Large Language Models for Autonomous Data Science): introduces an agentic LLM for autonomous data science, capable of end-to-end pipeline completion from data sources to analyst-grade reports, utilizing a curriculum-based agentic training paradigm and data-grounded trajectory synthesis.
- The framework employs a curriculum-based agentic training paradigm that emulates human data scientists' learning trajectory, progressively acquiring and integrating multiple capabilities in real-world environments.
- DeepAnalyze-8B leverages a data-grounded trajectory synthesis framework to construct high-quality training data, enabling autonomous orchestration and adaptive optimization for complex data science tasks.

---

[Agentic Inequality](http://arxiv.org/abs/2510.16853)

- Agentic Inequality Framework: introduces "agentic inequality," defining it as potential disparities in power, opportunity, and outcomes from differential access to and capabilities of AI agents, analyzed through the dimensions of agent availability, quality, and quantity.
- This framework distinguishes agentic inequality from prior technological divides by highlighting novel power asymmetries created by scalable goal delegation and direct agent-to-agent competition.
- The paper further explores the technical and socioeconomic drivers shaping agentic power distribution and proposes a research agenda for governing these complex challenges.

---

[A Comprehensive Survey on World Models for Embodied AI](http://arxiv.org/abs/2510.16732)

- Unified Framework for World Models: introduces a comprehensive survey on world models in embodied AI, proposing a three-axis taxonomy including Functionality (decision-coupled/general-purpose), Temporal Modeling (sequential simulation/global difference prediction), and Spatial Representation (global latent vector/token feature sequence/spatial latent grid/decomposed rendering).
- The survey formalizes problem settings, learning objectives, systematizes data resources and metrics, and quantitatively compares state-of-the-art models.
- It distills key open challenges such as data scarcity, evaluation metrics, computational efficiency, and long-horizon temporal consistency, while providing a curated bibliography.

---


[ReclAIm: A multi-agent framework for degradation-aware performance tuning of medical imaging AI](http://arxiv.org/abs/2510.17004)

- ReclAIm (A multi-agent framework for degradation-aware performance tuning of medical imaging AI): introduces a multi-agent framework for autonomously monitoring, evaluating, and fine-tuning medical image classification models, built on an LLM core and operating through natural language interaction.
- The framework includes a Master Agent, an Image classification agent, a Performance comparison agent, and a Fine-tuning agent, each equipped with specialized toolkits, an LLM Core, a System Prompt, and Memory, interacting with a User.
- ReclAIm enables automated, continuous maintenance of medical imaging AI models in a user-friendly and adaptable manner, facilitating broader adoption in research and clinical environments by addressing performance degradation.

---

[TACLA: An LLM-Based Multi-Agent Tool for Transactional Analysis Training in Education](http://arxiv.org/abs/2510.17913)

- TACLA (Transactional Analysis Contextual LLM-based Agents): introduces a novel Multi-Agent architecture designed to simulate nuanced human social dynamics with psychological depth and consistent persona behavior, integrating Parent, Adult, and Child ego states, an Orchestrator Agent, and dedicated memory for authentic responses.
- Each TACLA agent is modeled as a combination of Parent, Adult, and Child Ego State Agents, each with its own Contextual Pattern Memory and TA-informed reasoning capabilities, orchestrated by an LLM-based agent that prioritizes ego state activation based on contextual triggers and life scripts.
- The framework is validated in an educational scenario for teacher training, demonstrating realistic ego state shifts in Student Agents and effectively modeling conflict de-escalation and escalation based on different teacher intervention strategies, with a Feedback Agent providing expert-level analysis.

---

[EEschematic: Multimodal-LLM Based AI Agent for Schematic Generation of Analog Circuit](http://arxiv.org/abs/2510.17002)

- EEschematic: introduces an AI agent for automatic analog schematic generation, integrating textual, visual, and symbolic modalities, few-shot substructure examples, and a Visual Chain-of-Thought strategy for iterative refinement.
- The framework translates SPICE netlists into human-editable schematic diagrams by leveraging an MLLM to analyze circuit substructures, generate initial placements, and optimize wiring.
- It employs a VCoT prompting loop, comparing current schematics with reference examples and using result history to continuously improve visual quality and structural correctness.

---

[STARK: Strategic Team of Agents for Refining Kernels](http://arxiv.org/abs/2510.16996)

- STARK (Strategic Team of Agents for Refining Kernels): introduces a multi-agent LLM framework for GPU kernel optimization, systematically exploring the design space through collaborative agents, grounded instruction, dynamic context management, and strategic search.
- This framework mimics expert engineer workflows, enabling LLMs to reason about hardware trade-offs, incorporate profiling feedback, and iteratively refine kernels for substantial performance improvements.
- STARK achieves up to 16x faster runtime performance over baseline agents on KernelBench, demonstrating the potential of agentic LLM frameworks for automated GPU kernel optimization.

---

[Lark: Biologically Inspired Neuroevolution for Multi-Stakeholder LLM Agents](http://arxiv.org/abs/2510.16978)

- Lark (Biologically Inspired Neuroevolution for Multi-Stakeholder LLM Agents): introduces a biologically inspired decision-making framework that couples LLM-driven reasoning with an evolutionary, stakeholder-aware Multi-Agent System, integrating plasticity, duplication and maturation, ranked-choice stakeholder aggregation, and compute awareness via token-based penalties.
- The system iteratively proposes diverse strategies, applies plasticity tweaks, simulates stakeholder evaluations, aggregates preferences, selects top candidates, and performs duplication/maturation while factoring compute cost into final scores.
- Lark operates in a discrete generational paradigm, evolving populations of candidate strategies through selection, mutation, and specialization, making it suitable for multi-stakeholder strategy generation that lacks sequential state transitions and immediate reward signals.

---

[VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](http://arxiv.org/abs/2510.16907)

- VAGEN (Reinforcing World Model Reasoning for Multi-Turn VLM Agents): introduces a multi-turn reinforcement learning framework that enhances VLM agents' visual state reasoning by building internal world models through explicit State Estimation (describes current visual state) and Transition Modeling (predicts next visual state), optimized via WorldModeling Reward (dense reward for state predictions) and Bi-Level General Advantage Estimation (turn-aware credit assignment).
- The framework formulates the problem as a Partially Observable Markov Decision Process (POMDP) and systematically compares five reasoning strategies, demonstrating that explicit visual state reasoning significantly improves task performance.
- VAGEN achieves superior performance over untrained counterparts and proprietary models by integrating structured reasoning, task-dependent visual state representations, and hierarchical credit assignment for robust multi-turn VLM agent training.

---

[More with Less: An Empirical Study of Turn-Control Strategies for Efficient Coding Agents](http://arxiv.org/abs/2510.16786)

- Turn-Control Strategies for Coding Agents: introduces an empirical study evaluating the impact of various turn-control strategies on the performance and cost of LLM-powered coding agents, including an unrestricted baseline, fixed-turn limits, and a novel dynamic-turn strategy.
- The study identifies a "sweet spot" for fixed-turn limits at the 75th percentile, significantly reducing costs with minimal impact on solve rates, and demonstrates the superiority of the dynamic-turn strategy in balancing efficacy and economic efficiency.
- This research provides practical guidelines for deploying powerful yet economically viable coding agents by intelligently managing resource allocation through turn control.

---

[A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications](http://arxiv.org/abs/2510.16724)

- RL-based Agentic Search: introduces a paradigm where LLMs act as autonomous decision-making agents, capable of planning, retrieving, and reflecting through multi-step interaction with search environments, leveraging reinforcement learning for adaptive and self-improving search behavior.
- This survey comprehensively overviews RL-based agentic search, categorizing its functional roles, optimization strategies, and application scopes, while highlighting key components like the agent, environment, tools, and memory.
- The approach addresses LLM limitations such as static knowledge and factual hallucinations by enabling dynamic query refinement, adaptive retrieval strategies, and integration with diverse external knowledge sources and tools.

---

[Beyond Pipelines: A Survey of the Paradigm Shift toward Model-Native Agentic AI](http://arxiv.org/abs/2510.16720)

- This survey introduces the paradigm shift in agentic AI from the Pipeline-based Agentic AI Paradigm, where planning, tool use, and memory are externally orchestrated, to the Model-Native Agentic AI Paradigm, where these capabilities are internalized within the model's parameters, driven by Reinforcement Learning (RL).
- The Model-Native Agentic AI Paradigm reframes LLMs as autonomous decision-makers that learn to generate plans, invoke tools, and manage memory as intrinsic behaviors, enhancing adaptability and robustness in open environments.
- The paper systematically reviews the evolution of core capabilities and agent applications, such as Deep Research and GUI agents, and discusses emerging model-native capabilities like multi-agent collaboration and reflection.

---

[AN AGENTIC FRAMEWORK WITH LLMS FOR SOLVING COMPLEX VEHICLE ROUTING PROBLEMS](http://arxiv.org/abs/2510.16701)

- AFL (Agentic Framework with LLMs): introduces a fully automated, self-contained framework for solving complex Vehicle Routing Problems (VRPs) end-to-end, utilizing its Problem Description, Code Generation, and Solution Derivation subtasks, along with Generation, Judgment, Revision, and Error Analysis Agents, a Buffer, VRP Instance input, generated Code, Python execution, and Error handling.
- The framework extracts domain knowledge from raw VRP instance inputs to guide self-contained code generation, eliminating reliance on handcrafted modules or external solvers.
- AFL's specialized LLM agents collaborate to ensure cross-functional consistency, logical soundness, and constraint satisfaction, achieving high code reliability and solution feasibility.

---

#### 18th October 2025

[Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety](http://arxiv.org/abs/2510.16492)

- Quitting Mechanism: introduces a behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence, leveraging the ToolEmu framework with various prompting strategies, safety, and helpfulness evaluators.
- The mechanism evaluates LLM agents across 144 multi-turn scenarios, comparing baseline agents against simple and specified quit-enabled variants to assess safety-helpfulness trade-offs.
- Results indicate that explicit quit instructions, particularly the specified quit prompt, significantly improve agent safety with minimal impact on helpfulness, establishing quitting as a first-line defense.

---

[Explainability Requirements as Hyperproperties](http://arxiv.org/abs/2510.16402)

- YLTL (whY Linear-time Temporal Logic): introduces a formal framework for specifying and verifying explainability requirements in multi-agent systems, combining Lewis' counterfactuals (causal dependencies), Linear-time temporal logic (temporal reasoning), Knowledge modality (agent knowledge), Past-operators (past-time reasoning), and a Similarity relation (agent-specific causal models).
- The framework enables automated verification of explainability requirements through a Model-checking algorithm, which relies on a Translation function mapping YLTL formulas to Extended Monadic First-Order Logic (FO[<,E]) for hyperproperty specification.
- This approach allows for formalizing various notions of explainability, such as internal, external, general, and weak counterfactual explainability, and proves the decidability of the model-checking problem for finite-state multi-agent systems.

---

[ATA: A Neuro-Symbolic Approach to Implement Autonomous and Trustworthy Agents](http://arxiv.org/abs/2510.16381)

- ATA (Autonomous Trustworthy Agents): introduces a generic neuro-symbolic approach that decouples tasks into offline knowledge ingestion and online task processing, addressing LLM limitations in trustworthiness for high-stakes domains.
- The knowledge ingestion phase uses an LLM to translate informal problem specifications into a formal, symbolic knowledge base, which human experts can verify and refine for correctness and domain alignment.
- The task processing phase encodes incoming natural language input into the formal language via an LLM, enabling a symbolic decision engine to derive reliable, transparent, and auditable results using the formal knowledge base.

---

[Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration](http://arxiv.org/abs/2510.16645)

- DiMo (Multi-Agent Collaboration Framework for Diverse Thinking Modes): introduces a multi-agent debate framework that enhances LLM performance and interpretability by simulating structured debate among specialized LLM agents, including a Generator, Evaluator, Divergent Thinking Mode, Knowledge Supporter, Reasoning Path Provider, Logical Thinking Mode, Refiner, and Judger.
- The framework incorporates two distinct thinking modes—Divergent for commonsense reasoning and Logical for mathematical reasoning—to optimize problem-solving based on task requirements.
- DiMo generates explicit, human-auditable reasoning paths, improving LLM interpretability and transparency by externalizing hypotheses, supportive knowledge, and step-wise refinements.

---

[CODECRDT: OBSERVATION-DRIVEN COORDINATION FOR MULTI-AGENT LLM CODE GENERATION](http://arxiv.org/abs/2510.18893)

- CodeCRDT (Observation-Driven Coordination for Multi-Agent LLM Code Generation): introduces an observation-driven coordination pattern for multi-agent LLM code generation, where agents coordinate by monitoring a shared CRDT state for lock-free, conflict-free concurrent code generation, leveraging an Inference Service, Yjs Document, Outliner Agent, Implementation Agents, Evaluator Agent, TODO Observer, Frontend, WebSocket, and Hocuspocus WebSocket relay.
- This approach leverages Conflict-Free Replicated Data Types (CRDTs) to ensure strong eventual consistency and deterministic convergence, enabling parallel execution without explicit message passing among LLM agents.
- Empirical evaluation demonstrates that CodeCRDT achieves parallel speedups for most tasks by normalizing for code volume, while also revealing emergent behaviors like code inflation and semantic conflicts.

---

[Prompt Optimization via Retrieved Reasoning Assets and Multi-Agent Analysis](http://arxiv.org/abs/2510.16635)

- MA-SAPO (Multi-Agent Score-Aware Prompt Optimization): introduces a multi-agent framework for prompt optimization that explicitly links evaluation outcomes with structured reasoning to guide systematic edits, utilizing Metric Explainer, Diagnostician, and Action Synthesizer Agents in a Reasoning Phase, and Retriever, Analyzer, and Refiner Agents in a Test Phase.
- The framework operates in two phases: a Reasoning Phase where agents collaboratively explain scores, diagnose weaknesses, and synthesize targeted refinements into reusable reasoning assets, and a Test Phase where agents retrieve these assets to analyze and apply evidence-grounded edits to new prompts.
- This approach enhances interpretability, auditability, and controllability of prompt refinements by transforming evaluation signals into explicit reasoning chains, leading to consistent performance improvements while reducing computational costs.

---

[Ripple Effect Protocol: Coordinating Agent Populations](http://arxiv.org/abs/2510.16572)

- REP (Ripple Effect Protocol): introduces a coordination protocol for LLM-based agents, enabling them to share decisions and lightweight textual sensitivities that ripple through local networks, facilitating faster and more stable alignment than decision-only communication.
- The protocol formalizes message schemas and aggregation rules, decoupling agent cognition from coordination, and supports diverse LLM architectures and hybrid rule-based systems.
- REP significantly improves coordination accuracy and efficiency across domains like supply chain, resource allocation, and preference aggregation, providing scalable infrastructure for the Internet of Agents.

---

[LANPO: Bootstrapping Language and Numerical Feedback for Reinforcement Learning in LLMs](http://arxiv.org/abs/2510.16552)

- LANPO (Language-And-Numerical Policy Optimization): introduces a framework that synergistically bootstraps language and numerical feedback to enhance LLM learning efficiency, utilizing a Policy LLM, Reward Model, Critic, Experience Pool, Inter-Sample Exploration Module with Retrieval and Relevance Evaluation, Intra-Sample Exploration Module with Reward-Agnostic Reflection, Feedback Summarizer, Parameter Update, and Context Update.
- The framework addresses information leakage and behavior collapse by separating feedback roles: language guides exploration via context updates, while numerical rewards drive robust policy optimization through parameter updates.
- LANPO builds a dynamic experience pool from past trials and employs Reward-Agnostic Reflection for safe intra-sample self-correction and Relevant Abstraction for generalizable inter-sample lessons.

---

[Declarative Techniques for NL Queries over Heterogeneous Data](http://arxiv.org/abs/2510.16470)

- siwarex: introduces a declarative system for handling natural language queries over heterogeneous data sources, leveraging its Abstract Schema, API Mapping Schema, DB Table View, Text2SQL module, Query Rewriter, User-Defined Functions, and LLM to unify database tables and APIs within a SQL framework.
- The framework translates natural language questions into SQL queries by representing both database tables and external APIs as virtual tables in a unified relational schema, which are then processed by a query rewriter to invoke APIs via UDFs.
- This approach significantly outperforms imperative code generation and agent-based methods in coping with data source heterogeneity, as demonstrated on two new benchmarks.

---

[RGMem: Renormalization Group-based Memory Evolution for Language Agent User Profile](http://arxiv.org/abs/2510.16392)

- RGMem (Renormalization Group-based Memory Evolution for Language Agent User Profile): introduces a self-evolving memory framework for LLM-based conversational systems, with D_raw, f_seg, f_synth, D_L0, A_fact, A_base, A_rel, f_extract, G, V, V_abs, V_gen, V_inst, E, E_cls, E_evt, G(t), RK1, LLM, θ_inf, RK2, P, S, θ_sum, RK3, Synergy/Tension Analysis, Dirty-flag Propagation Mechanism, G*, Σ, Δ, Query, f_retr, Context Aggregation & Output, Macroscopic Theory (Σ, Δ), Mesoscopic Theory (T(1)), and Microscopic Evidence (A_fact, A_base) components, which organizes dialogue history across multiple scales to form a dynamically-evolved user profile.
- The framework leverages renormalization group principles to extract semantics and user insights from episodic fragments, progressively forming a multi-scale user profile through hierarchical coarse-graining and rescaling operations.
- This approach enables multi-granularity retrieval, coordinating detailed and abstract memories to boost cross-session continuity and personalized interactive capabilities for Language Agents.

---

[Integrating LLM and Diffusion-Based Agents for Social Simulation](http://arxiv.org/abs/2510.16366)

- LLM-empowered Hybrid Simulation Agent Framework: introduces a hybrid simulation approach for social information diffusion prediction, integrating LLM-based agents (simulates core users) for semantic reasoning with diffusion model-based agents (predicts remaining users) for efficient population-level prediction.
- This framework employs LLM-based agents to simulate a core subset of users with rich semantic reasoning, while a diffusion model handles the remaining population efficiently, both incorporating user personalization, social influence, and content awareness.
- The modular design enables a topic-aware, personalized, and collaborative simulation, addressing computational costs of LLMs at scale and cold-start problems of traditional diffusion models.

---

[WHAT LIMITS AGENTIC SYSTEMS EFFICIENCY?](http://arxiv.org/abs/2510.16276)

- SpecCache: introduces a caching framework augmented with speculative execution to mitigate web environment latency in web-interactive agentic systems, including a Model Input, Reasoning (Target Model), Action (Target Model), Observation, Candidate Actions (Draft Model), Cache Pool, Cache Hit, and Cache Miss.
- The framework decouples and overlaps model reasoning with environment interaction by using a draft model to predict future actions and proactively populate an action-observation cache.
- This approach significantly reduces wall-clock latency and web environment overhead without compromising task success rates, achieving up to 58x improvement in cache hit rate and 3.2x reduction in web environment overhead.

---

[Branch-and-Browse: Efficient and Controllable Web Exploration with Tree-Structured Reasoning and Action Memory](http://arxiv.org/abs/2510.19838)

- Branch-and-Browse: introduces a fine-grained web agent framework that unifies structured reasoning-acting, contextual memory, and efficient execution, including a subtask manager, tree exploration, nearest-URL state replay, background reasoning, and page action memory.
- This framework employs explicit subtask management with tree-structured exploration for controllable multi-branch reasoning and efficient backtracking, while leveraging web state replay and background reasoning to accelerate exploration.
- A page action memory mechanism further enhances efficiency by sharing explored actions and contextual information across branches and sessions, reducing redundancy and improving decision-making.

---

#### 17th October 2025


[Agentic AI for Ultra-Modern Networks: Multi-Agent Framework for RAN Autonomy and Assurance](http://arxiv.org/abs/2510.16144)

- Multi-Agent Framework for RAN Autonomy and Assurance: introduces a distributed multi-agent architecture for RAN autonomy and assurance, featuring an Orchestrator Agent (manages workflow, resolves conflicts), Data Collector Agent (collects, validates raw telemetry), Preprocessor and Feature Agent (cleans, engineers data features), Model Trainer Agent (trains, optimizes AI/ML models), Model Validator Agent (evaluates, approves AI/ML models), Predictor Agent (forecasts KPIs, simulates scenarios), Policy Generator Agent (formulates network policies), Simulator/Baseline Agent (generates reference KPI trajectories), Verifier Agent (compares policies, enforces safety), Drift Detector Agent (detects model drift, triggers retraining), Deployment Agent (deploys verified network policies), Audit and Explainability Agent (generates audit reports, explanations), Security Agent (secures inter-agent communications), and Inter-Agent Communication (enables agent coordination).
- This framework replaces centralized RIC-based control with specialized, collaborative agents to ensure autonomy, resilience, explainability, and system-wide safety in Beyond 5G/6G networks.
- The architecture prevents unsafe policy deployments by incorporating independent verification and assurance stages, safeguarding global network health against model drift and unforeseen conditions.

---


[POLYSKILL: LEARNING GENERALIZABLE SKILLS THROUGH POLYMORPHIC ABSTRACTION](http://arxiv.org/abs/2510.15863)

- PolySkill (Polymorphism-Guided Agent Skill Induction): introduces a novel framework enabling web agents to learn generalizable and compositional skills by decoupling abstract goals from concrete implementations, utilizing an LM Policy, Working Memory, Dynamic Skill Library, Abstract Classes, Concrete Subclasses, an LLM-based Induction Module, and an LM Judge.
- The framework organizes skills into a domain-driven hierarchy, where abstract classes define common interfaces for categories like shopping sites, and concrete subclasses provide website-specific implementations, promoting skill reuse and cross-domain generalization.
- PolySkill enhances continual learning by guiding agents to discover and refine skills autonomously in task-free settings, leading to improved task success rates and reduced execution steps across diverse web environments.

---

[PAPER2WEB: LET'S MAKE YOUR PAPER ALIVE!](http://arxiv.org/abs/2510.15842)

- PWAGENT (Paper-to-Web Agent): introduces a multi-agent framework for transforming academic papers into interactive, multimedia-rich project homepages, utilizing Docling (PDF to Markdown converter), an LLM (extracts metadata/structures content), Construct (combines decomposed assets), an MCP Resource Repository (stores structured paper assets), an MLLM as Orchestrator (assesses webpage/invokes tools), and MCP tool use (accesses repository/edits webpage).
- This framework addresses limitations of current methods by decomposing papers into structured assets, ingesting them into a resource repository, and iteratively refining webpage content and layout through an MLLM-orchestrated process.
- PWAGENT achieves state-of-the-art cost efficiency and high presentation quality, outperforming baselines in academic webpage generation while maintaining low cost.

---

[VISTA: A Test-Time Self-Improving Video Generation Agent](http://arxiv.org/abs/2510.15831)

- VISTA (Video Iterative Self-improvemenT Agent): introduces a novel multi-agent system that autonomously improves text-to-video generation by refining prompts in an iterative loop, including Structured Video Prompt Planning (transforms user input), Pairwise Tournament Selection (identifies best video-prompt pair), Multi-Dimensional Multi-Agent Critiques (MMAC) (generates nuanced critiques), and Deep Thinking Prompting Agent (DTPA) (refines prompt iteratively).
- The framework decomposes user ideas into structured temporal plans, identifies the best video through a robust pairwise tournament, critiques it using specialized agents focusing on visual, audio, and contextual fidelity, and then synthesizes feedback to enhance prompts for subsequent generation cycles.
- VISTA consistently improves video quality and alignment with user intent, achieving up to 60% pairwise win rate against state-of-the-art baselines and demonstrating scalability with increased test-time computation.

---

[AURA: An Agent Autonomy Risk Assessment Framework](http://arxiv.org/abs/2510.15739)

- AURA (Agent aUtonomy Risk Assessment): introduces a unified framework designed to detect, quantify, and mitigate risks from agentic AI, incorporating an LLM Parser, LLM Dimensions, LLM Scorer, LLM Mitigator, Memory Unit, HITL, and A2H Control to provide robust risk assessment and mitigation.
- The framework supports both synchronous and autonomous modes, enabling agents to self-assess and mitigate risks during operation, while also allowing human oversight and intervention.
- AURA balances risk assessment accuracy with computational efficiency through gamma-based scoring and memory-driven optimization, ensuring governable and transparent AI agent deployment.

---


[Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions](http://arxiv.org/abs/2510.15258)

- Multi-dimensional Data Analysis Framework: introduces a dynamic, collaborative analytical ecosystem that integrates LLM agents and Knowledge Graphs (KGs) for multi-dimensional data analysis, featuring a Data Preparation Module, Knowledge Representation Module, Visualization and Interaction Module, and Intelligent Analysis Module.
- The framework enables LLM agents to automatically extract product data, construct and visualize KGs in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform.
- This approach achieves bidirectional dynamic interaction between LLM agents and KGs, where agents build and enrich the KG, and the visualized KG provides context for the agents' in-depth analysis.

---

[Build Your Personalized Research Group: A Multiagent Framework for Continual and Interactive Science Automation](http://arxiv.org/abs/2510.15624)

- freephdlabor: introduces a multiagent framework for continual and interactive science automation, featuring a ManagerAgent, IdeationAgent, ExperimentationAgent, ResourcePreparationAgent, WriteupAgent, ReviewerAgent, Shared Workspace, Workspace System, Prompt Optimization Mechanisms, Context Compaction, Memory Persistence, and Real-Time Human Intervention, enabling dynamic workflows and robust communication for scientific discovery.
- The framework addresses limitations of existing agentic systems by providing fully dynamic workflows determined by real-time agent reasoning and a modular architecture for seamless customization and human-in-the-loop capabilities.
- It provides comprehensive infrastructure for automatic context compaction, workspace-based communication to prevent information degradation, memory persistence across sessions, and non-blocking human intervention mechanisms, transforming automated research into continual programs.

---

[SHARE: Scene-Human Aligned Reconstruction](http://arxiv.org/abs/2510.15342)

- SHARE (Scene-Human Aligned REconstruction): introduces a framework that reconstructs human motion and the surrounding environment from monocular videos, leveraging scene geometry for accurate 3D human placement.
- The framework operates in three stages: initialization of point maps, human meshes, and masks; reconstruction of the background scene; and optimization of human meshes by grounding them to scene points.
- SHARE achieves improved 3D human positioning and scene reconstruction, outperforming existing methods in quantitative metrics and demonstrating strong qualitative performance on diverse video data.

---

[Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition](http://arxiv.org/abs/2510.15280)

- Three-Stage Framework for FM-driven Scientific Evolution: introduces a conceptual model describing the progressive integration of FMs into scientific discovery, encompassing Meta-Scientific Integration, Hybrid Human-AI Co-Creation, and Autonomous Scientific Discovery stages.
- The framework posits that FMs transition from backend tools, to interactive collaborators, and finally to independent agents capable of end-to-end scientific discovery.
- This evolution redefines scientific paradigms, shifting from human-guided processes to increasingly autonomous AI-driven knowledge generation.

---

[PokeeResearch: Effective Deep Research via Reinforcement Learning from AI Feedback and Robust Reasoning Scaffold](http://arxiv.org/abs/2510.15862)

- PokeeResearch-7B: introduces a 7B-parameter deep research agent, trained with Reinforcement Learning from AI Feedback (RLAIF) using LLM-based reward signals, and featuring a robust chain-of-thought-driven multi-call reasoning scaffold with self-verification and adaptive recovery for tool-augmented research.
- The agent operates through iterative research-verification cycles, leveraging specialized web searching and reading tools, and is built upon a Qwen2.5-7B-Instruct backbone LLM.
- This approach achieves state-of-the-art performance on ten deep research benchmarks by optimizing for human-salient answer quality dimensions and maintaining robustness through verifiable reasoning.

---

[Self-evolving expertise in complex non-verifiable subject domains: dialogue as implicit meta-RL](http://arxiv.org/abs/2510.15772)

- Dialectica: introduces a framework where LLM agents engage in structured dialogue on defined topics, augmented by Agent Memory, Agent Reflection, and Context Evolution, with an Orchestrator managing the dialogue and an optional Facilitator guiding the discussion.
- The framework views discussion as an implicit meta-reinforcement learning process, enabling agents to develop expertise and refine their prompt contexts through conversational feedback and self-reflection in non-verifiable domains.
- This approach allows agents to improve their capabilities and produce more sophisticated outputs by iteratively updating their internal context based on dialogue experiences, without explicit reward signals.

---

[ProofOptimizer: Training Language Models to Simplify Proofs without Human Demonstrations](http://arxiv.org/abs/2510.15700)

- ProofOptimizer: introduces an LLM-based system for simplifying Lean proofs without human demonstrations, integrating a symbolic Lean linter, a finetuned 7B parameter language model, and an iterative inference-time algorithm.
- The system is trained using expert iteration and online reinforcement learning, leveraging the Lean compiler for verification and reward signals, and employs inference-time techniques like Test-Time RL and proof repair.
- ProofOptimizer significantly reduces proof length on various benchmarks, improving conciseness, execution speed, and downstream prover performance for AI-generated formal proofs.

---

[SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation](http://arxiv.org/abs/2510.15682)

- SQuAI (Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation): introduces a scalable and trustworthy multi-agent RAG framework for scientific QA, which includes a Decomposer (decomposes complex queries into sub-questions), Hybrid Retrieval (selects top-k documents using sparse/dense models), a Generator (generates initial Q-A-E triplets), a Judge (evaluates Q-A-E triplets for relevance), and an Answer Generator (synthesizes final answer with citations).
- The framework addresses key limitations of existing RAG systems in scholarly domains by enabling accurate answers, explicit claims with citations, and retrieval across millions of scientific documents.
- SQuAI improves faithfulness, answer relevance, and contextual relevance by decomposing complex questions, adaptively filtering documents, and providing fine-grained in-line citations for transparent verification.

---

[The Spark Effect: On Engineering Creative Diversity in Multi-Agent AI Systems](http://arxiv.org/abs/2510.15568)

- Spark agents: introduces a system of persona-conditioned LLM agents, instantiated through a library of role-inspired system prompts, to intentionally diversify agent behavior within a multi-agent workflow.
- The system includes a Spark agent automation pipeline for data collection and retrieval-augmented grounding, and an LLM-as-a-judge protocol for evaluating creative diversity against human gold standards.
- This approach achieved a mean diversity gain of +4.1 points on a 1-10 scale, significantly narrowing the gap to human experts and improving client-facing outputs.

---

[KITE: A Benchmark for Evaluating Korean Instruction-Following Abilities in Large Language Models](http://arxiv.org/abs/2510.15558)

- KITE (Korean Instruction-following Task Evaluation): introduces a comprehensive benchmark for evaluating LLMs' instruction-following capabilities in Korean, encompassing both general and Korean-specific instructions, validated through automated metrics and human assessments.
- The benchmark includes KITE General, derived from translated English datasets, and KITE Korean, featuring specialized instructions like Acrostic Poem and Honorifics, designed to capture unique linguistic and cultural nuances.
- This framework provides insights into LLM performance across diverse NLP tasks and models, aiming to foster research on culturally and linguistically inclusive LLM development for underrepresented languages.

---

[THE ROAD LESS TRAVELED: ENHANCING EXPLORATION IN LLMS VIA SEQUENTIAL SAMPLING](http://arxiv.org/abs/2510.15502)

- SESA (SEquential SAmpling): introduces a two-stage framework for enhancing exploration in LLMs, including PromptSketch (generates sketch prompt), Policy (πθ) (samples sketches/solutions), History of Sketches (S) (stores generated sketches), PromptSolve (generates solution prompt), Reward Function (R) (computes solution reward), All Candidates (Y) (stores solutions, rewards), Advantage Computation (calculates policy advantages), Loss Computation (computes policy loss), and Policy Update (adjusts policy parameters), which mitigates entropy collapse by sequentially generating diverse solution sketches before expanding them into full reasoning paths.
- This approach conditions each new output on previous ones, promoting diversity and preventing policy collapse, leading to broader exploration and improved performance in RL-trained LLMs.
- SESA consistently outperforms traditional RL methods in path diversity and recovery from collapse, significantly boosting success rates on agent benchmarks and real-world tasks.

---

[CORE: Reducing UI Exposure in Mobile Agents via Collaboration Between Cloud and Local LLMs](http://arxiv.org/abs/2510.15455)

- CORE (Collaborative framework): introduces a collaborative framework that combines cloud and local LLMs to reduce UI exposure in mobile agents, including layout-aware block partitioning (groups UI elements), co-planning (collaboratively identifies sub-task), and co-decision-making (collaboratively selects UI elements).
- The framework leverages the cloud LLM's strong reasoning with limited UI access and the local LLM's basic reasoning with full UI visibility to achieve a balance between task accuracy and privacy.
- CORE significantly reduces sensitive UI element uploads to the cloud by up to 70.49% while maintaining task success rates comparable to cloud-only agents.

---

[Select Less, Reason More: Prioritizing Evidence Purity for Video Reasoning](http://arxiv.org/abs/2510.15440)

- EARL (Evidence-Aware Reinforcement Learning): introduces an evidence-prioritized adaptive pixel-space video reasoning framework, with a Video LLM, Visual Encoder/Text Tokenizer, Merger Projector, Think + Frames Selection Function, Key-frame based Localized Re-sampling Module, and a Multi-component Reward System, to dynamically select relevant frames and perform localized re-sampling for fine-grained temporal detail.
- This framework transforms passive video processing into an active evidence interrogation process, guided by a novel multi-component reward system that enforces evidence purity and strategically manages visual context selection.
- The dynamic adjustment mechanism within the reward system ensures stable convergence by balancing exploration and purity requirements throughout training, leading to superior reasoning accuracy.

---

[ADAPTIVE MINDS: EMPOWERING AGENTS WITH LORA-AS-TOOLS](http://arxiv.org/abs/2510.15416)

- Adaptive Minds: introduces an agentic system that treats LoRA adapters as domain-specific tools, empowering a base LLM to act as a semantic router for dynamically selecting the most relevant LoRA tool to handle each query.
- The system employs a modular multi-agent design orchestrated by LangGraph, combining flexible multi-agent orchestration with parameter-efficient fine-tuning to deliver accurate, specialized responses while preserving conversational ability.
- Its AI-semantic routing, which leverages the base LLM's understanding, significantly outperforms keyword-based methods in accuracy and achieves a 3.1x average speedup compared to a baseline monolithic model.

---

[MARS: REINFORCING MULTI-AGENT REASONING OF LLMS THROUGH SELF-PLAY IN STRATEGIC GAMES](http://arxiv.org/abs/2510.15414)

- MARS (Reinforcing Multi-Agent Reasoning of LLMs through Self-play in Strategic Games): introduces an end-to-end RL framework that incentivizes multi-agent reasoning in LLMs through self-play in both cooperative and competitive games.
- The framework incorporates a turn-level advantage estimator for fine-grained credit assignment and agent-specific advantage normalization to stabilize multi-agent training.
- MARS agents, trained on a diverse portfolio of strategic games, develop strong strategic abilities that generalize to held-out games and improve performance in multi-agent reasoning benchmarks.

---

[Accelerating Mobile Language Model Generation via Hybrid Context and Hardware Coordination](http://arxiv.org/abs/2510.15312)

- CoordGen: introduces a mobile inference framework that integrates speculative decoding with dynamic hardware scheduling to accelerate context-aware text generation on mobile devices, utilizing adaptive execution scheduling, context-aligned drafting, and hardware-efficient draft extension.
- The framework addresses high latency and limited hardware utilization in on-device LLMs by offloading retrieval-based speculative decoding to NPUs, employing progressive graph scheduling, in-context distribution calibration, and NPU-optimized draft reuse.
- CoordGen achieves significant speedup and energy efficiency improvements on smartphones across various tasks and LLMs by optimizing compute graph management and draft generation for NPU acceleration.

---

[WebGen-V Bench: Structured Representation for Enhancing Visual Design in LLM-based Web Generation and Evaluation](http://arxiv.org/abs/2510.15306)

- WebGen-V Bench: introduces a new benchmark and framework for instruction-to-HTML generation, with a Crawling Module (data acquisition and preprocessing), Processor (transforms raw data into structured representation), Structured Data (section-level metadata, UI screenshots, JSON text/image assets, instructions), Gen (HTML generation model), Evaluation Module (section-wise assessment of model outputs), Evaluator (multimodal LLM for scoring and feedback), and Feedback (iterative refinement for continuous improvement), providing a unified pipeline from real-world data acquisition to structured multimodal assessment.
- The framework enhances data quality and evaluation granularity through an agentic crawling framework, structured section-wise data representation, and a section-level multimodal evaluation protocol.
- WebGen-V enables high-granularity assessment by aligning text, layout, and visuals at the section level, facilitating precise detection and correction of subtle design inconsistencies in LLM-generated webpages.

---

[Exemplar-Guided Planning: Enhanced LLM Agent for KGQA](http://arxiv.org/abs/2510.15283)

- PoG-EGP (Plan-on-Graph with Exemplar-Guided Planning): introduces a novel framework that enhances LLM agents' planning capabilities for Knowledge Graph Question Answering (KGQA) by leveraging preprocessed training data, including Question Preprocessing, Text Embedding Generation, Exemplary Question Retrieval, Retrieved Exemplars, Smart Lookahead Mechanism, PoG, LLM Agent, Task Decomposition, Path Exploration, Memory, Evaluation, and Reflection, to dynamically guide the LLM's planning process in task decomposition and relation exploration.
- The framework preprocesses training questions via entity templating, generates semantic embeddings, and retrieves similar exemplary questions and their reasoning paths using a FAISS index to provide high-quality auxiliary information.
- A Smart Lookahead mechanism is integrated to improve efficiency during relation exploration by preemptively identifying promising paths and terminating exploration earlier, significantly enhancing performance and efficiency on KGQA datasets.

---

[AUGUSTUS: An LLM-Driven Multimodal Agent System with Contextualized User Memory](http://arxiv.org/abs/2510.15261)

- AUGUSTUS (An LLM-Driven Multimodal Agent System with Contextualized User Memory): introduces a multimodal agent system that processes, stores, retrieves, and acts on user context across various modalities, aligning its four-stage loop (Encode, Store in Memory, Retrieve, Act) with human cognitive memory principles.
- The system leverages an LLM as its central planner, integrating In-Context, Recall, and a novel graph-structured Contextual Memory to manage information, and employs a Contextual-Personalized (CoPe) search for efficient concept-driven retrieval.
- AUGUSTUS utilizes modality-specific encoders for input understanding and various generation tools for multimodal output, demonstrating superior performance and efficiency compared to traditional multimodal RAG approaches.

---

[EXPERIENCE-DRIVEN EXPLORATION FOR EFFICIENT API-FREE AI AGENTS](http://arxiv.org/abs/2510.15259)

- KG-Agent: introduces an experience-driven learning framework that structures raw pixel-level GUI interactions into a persistent State-Action Knowledge Graph (SA-KG) and employs a VLM-based Reasoning Module for skill invocation, augmentation, refinement, and evaluation.
- The framework leverages a hybrid intrinsic reward mechanism, combining state value and novelty rewards, to support long-horizon reasoning and efficient exploration.
- By connecting functionally similar yet visually distinct GUI states, KG-Agent enables generalization from diverse historical strategies, significantly improving exploration efficiency and strategic depth in API-free environments.

---

[Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding](http://arxiv.org/abs/2510.15253)

- Multimodal RAG: introduces a systematic survey of Multimodal Retrieval-Augmented Generation for document understanding, detailing its components like User Query, Document, PDF2Img, OCR or Annotate, Image Retrieval, Text Retrieval, Multimodal Retrieval, Model, Answer Generation, Knowledge Base, Graph-based Index, Graph Traversal for Retrieval, LLM Agent, Query Decomposition, and Verification.
- The survey categorizes existing methods by domain openness (closed/open), retrieval modality (image/text/hybrid), retrieval granularity (page/element), and hybrid enhancements (graph/agent-based).
- It highlights the importance of Multimodal RAG for comprehensive document intelligence, addressing MLLM limitations in context modeling and enabling holistic retrieval and reasoning across text, tables, charts, and layout.

---

[LLM-based In-situ Thought Exchanges for Critical Paper Reading](http://arxiv.org/abs/2510.15234)

- LLM-based In-situ Thought Exchange Interface: introduces a system designed to enhance junior researchers' critical paper reading skills by integrating AI-driven conversational agents into a custom PDF viewer, featuring a Comment Pane and Section Pane for interactive thought exchanges, highlighting, and commenting.
- The system leverages LLMs to generate critical thinking questions, provide multi-disciplinary feedback, and reinterpret content, supporting both single-agent and multi-agent interaction modes.
- This approach aims to foster critical thinking by encouraging active engagement and diverse perspectives, moving beyond passive information consumption.

---

[EVOLVER: SELF-EVOLVING LLM AGENTS THROUGH AN EXPERIENCE-DRIVEN LIFECYCLE](http://arxiv.org/abs/2510.16079)

- EvolveR (Self-Evolving LLM Agents Through an Experience-Driven Lifecycle): introduces a self-evolving LLM agent framework through a closed-loop experience lifecycle, integrating online interaction, offline self-distillation, and policy evolution for continuous self-improvement.
- The framework enables agents to transform raw interaction trajectories into a curated repository of strategic principles, which are then used to guide future decision-making and generate high-quality data.
- EvolveR employs a dynamic experience curation system with mechanisms for self-distillation, semantic deduplication, integration, and quality control, ensuring a compact and effective knowledge base.

---

[DETECTING ADVERSARIAL FINE-TUNING WITH AUDITING AGENTS](http://arxiv.org/abs/2510.16255)

- Fine-tuning Auditing Agent: introduces a robust detection mechanism for adversarial fine-tuning, utilizing an LLM as an agent with access to the fine-tuning dataset, pre-fine-tuned and fine-tuned models, and a suite of audit tools including dataset inspection, recursive summarization, model querying, Python execution, and benchmark running.
- The agent systematically evaluates fine-tuned models by inspecting training data for patterns, querying models to compare behavior, and running benchmarks with attack-specific elicitation to assign a risk score for the fine-tuning job.
- This approach effectively detects diverse fine-tuning attack vectors, including covert cipher attacks, by learning encoding schemes in-context and eliciting harmful responses, thereby preventing the deployment of maliciously poisoned LLMs.

---

[Towards Automatic Evaluation and Selection of PHI De-identification Models via Multi-Agent Collaboration](http://arxiv.org/abs/2510.16194)

- TEAM-PHI (Trusted Evaluation and Automatic Model selection for PHI): introduces a multi-agent framework for automatic evaluation and selection of PHI de-identification models, utilizing Clinical Notes (raw clinical text), De-id Models (PHI extraction LLMs), Evaluation Agents (LLM-based judges), and LLM Majority Vote (judgment aggregation, selection) to assess de-identification quality without gold labels.
- The framework employs multiple LLM-based Evaluation Agents to independently judge PHI extractions from various De-id models, consolidating their structured metrics via an LLM-based majority voting mechanism.
- TEAM-PHI provides a practical, secure, and cost-effective solution for automatic evaluation and best-model selection in PHI de-identification, demonstrating consistent and accurate rankings even with limited ground-truth labels.

---

#### 16th October 2025

[AGENTIC DESIGN OF COMPOSITIONAL MACHINES](http://arxiv.org/abs/2510.14980)

- Agentic Design of Compositional Machines: introduces a framework for LLM agents to design complex machines in the BesiegeField (simulated physical environment), including Designer (produces initial plan), Refiner (evaluates, proposes revisions), Inspector (abstractly assesses machine), Environment Querier (runs simulation, summarizes feedback), Meta-Designer (analyzes requirements, creates blueprint), Builder Agents (constructs blocks based on blueprint), and MCTS (search strategy for candidates).
- The framework enables LLMs to construct machines from standardized components to meet functional demands, leveraging agentic workflows for iterative design and hierarchical construction.
- The paper also explores RL finetuning of LLMs within this environment to improve spatial reasoning, strategic assembly, and instruction-following capabilities for machine design.

---

[LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training](http://arxiv.org/abs/2510.14969)

- UI-Simulator: introduces a scalable paradigm for synthesizing training trajectories, integrating an LLM Pre-Training Corpus (Input data for LLMs), an LLM World Simulator (LLM-based UI environment generator), a Guided Rollout Process (Collects coherent, diverse UI trajectories), and a Trajectory Wrapper (Transforms rollouts into training data).
- The framework leverages LLMs pre-trained on UI code and procedural knowledge to simulate diverse UI states and transitions, enabling robust digital agent training without extensive human annotation.
- UI-Simulator-Grow extends this by incorporating Target Task Selection (Identifies high-impact learning tasks), Trajectory Variant Synthesis (Generates diverse task variations), and Continual Learning (Adapts agent policies iteratively) for data-efficient scaling.

---

[INFORMATION GAIN-BASED POLICY OPTIMIZATION: A SIMPLE AND EFFECTIVE APPROACH FOR MULTI-TURN LLM AGENTS](http://arxiv.org/abs/2510.14967)

- IGPO (Information Gain-based Policy Optimization): introduces a reinforcement learning framework for multi-turn LLM agents, utilizing a Policy LLM (Agent) interacting with an Environment through a Rollout of sequential Turns, each comprising a Think Step, Tool Call Step, and Tool Response Step, culminating in an Answer Turn, where rewards are calculated using Ground Truth, combining an Information Gain Reward and an Outcome Reward into a Reward Trajectory, which is then used to compute a Discounted Cumulative Advantage for policy optimization via a GRPO-style Surrogate Objective, guided by a Prompt Template.
- The framework addresses reward sparsity in multi-turn LLM agent training by providing dense, intrinsic, turn-level supervision based on information gain, which measures the marginal increase in the policy's probability of producing the correct answer.
- IGPO integrates this intrinsic turn-level reward with outcome-level supervision to form a dense reward trajectory, enhancing credit assignment and improving sample efficiency and accuracy in multi-turn scenarios.

---

[Identity-Link IRT for Label-Free LLM Evaluation: Preserving Additivity in TVD-MI Scores](http://arxiv.org/abs/2510.14966)

- Clipped-Linear Model (Identity-Link Item Response Theory): introduces a novel LLM evaluation framework that leverages TVD-MI scores, an LLM judge, and an identity link to preserve additivity in agent-item score matrices, enabling sample-efficient sparse recovery.
- This framework employs a clipped-linear model derived from Gini entropy maximization, which directly models raw TVD-MI scores as an additive decomposition of latent agent abilities and item difficulties, avoiding distortions from traditional logistic/probit links.
- The approach achieves significant sample efficiency, requiring 3x fewer evaluations than dense methods while maintaining high reconstruction accuracy and preserving agent rankings, validated through discrete integrability tests and cross-domain experiments.

---

[Mapping Smarter, Not Harder: A Test-Time Reinforcement Learning Agent That Improves Without Labels or Model Updates](http://arxiv.org/abs/2510.14900)

- TTRL Agent (Test-Time Reinforcement Learning Agent): introduces a reinforcement learning agent that self-improves schema mapping accuracy without labeled data or model updates by iteratively refining mappings through a Generative LLM, Conflict Detection, external Evidence Collection, and Confidence Evaluation, guided by dynamic prompts and an accumulating Memory/Context.
- The agent identifies ambiguous mappings, formulates targeted web-search queries for external evidence, and uses confidence-based rewards to iteratively refine its mappings, reducing low-confidence mappings requiring expert review.
- This approach provides an evidence-driven, transparent method for schema mapping, achieving high accuracy and reducing manual verification costs in scenarios with incomplete documentation.

---

[THE GATEKEEPER KNOWS ENOUGH](http://arxiv.org/abs/2510.14881)

- The Gatekeeper Protocol: introduces a novel, domain-agnostic framework that governs LLM agent-system interactions, utilizing a System State-Context Representation (SCR) as a central data structure, an AGENT for reasoning and proposing actions, and a System / Execution Environment for validating and executing these actions.
- This protocol mandates that the AGENT first reasons on a low-fidelity "latent state" representation within the SCR to strategically request high-fidelity context on demand, ensuring token efficiency and grounded interactions.
- All interactions are mediated through a unified JSON format, serving as a declarative, state-synchronized protocol that ensures the agent's model of the system remains verifiably grounded in reality, significantly improving reliability and scalability.

---

[Where to Search: Measure the Prior-Structured Search Space of LLM Agents](http://arxiv.org/abs/2510.14846)

- Formal Theory for LLM-assisted Iterative Search: introduces a compact formal theory to describe and measure LLM-assisted iterative search, representing agents as fuzzy relation operators and characterizing search space geometry.
- The theory quantifies reachability difficulty using a coverage generating function and critical parameters, while safety is ensured by confining agents within a crisp idealized safety envelope.
- A majority-vote instantiation on a 2D grid validates the abstract concepts, providing operational tools to measure LLM agents and their search spaces.

---

[Agentic NL2SQL to Reduce Computational Costs](http://arxiv.org/abs/2510.14808)

- Datalake Agent: introduces an agentic system designed to enable an LLM to solve NL2SQL tasks more efficiently, with Information Acquisition, Iterative Refinement, and Query Formulation components, where the system reduces meta-information processing by selectively requesting necessary data.
- The framework employs an interactive loop, allowing the LLM to gather general schema knowledge, refine its understanding hierarchically, and generate precise SQL queries using predefined commands like GetDBDescription, GetTables, GetColumns, and DBQueryFinalSQL.
- This approach significantly reduces token usage and computational costs by up to 87% compared to direct prompting, while maintaining competitive performance on table question answering tasks across varying database sizes.

---

[ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling](http://arxiv.org/abs/2510.14703)

- ToolPRM (Fine-Grained Inference Scaling of Structured Outputs for Function Calling): introduces an inference scaling framework that combines a ToolPRM (process reward model) with fine-grained beam search, leveraging a fine-grained intra-call process supervision dataset and function masking techniques to enhance LLM agent performance in structured function calling.
- The framework decomposes function calls into semantically interpretable intermediate reasoning steps, enabling ToolPRM to provide step-level rewards for each decision, which guides the beam search to "explore more but retain less" for reliable structured output generation.
- This approach significantly improves backbone model performance across various function calling tasks by offering more granular feedback than coarse-grained or outcome-based reward models, addressing the unrecoverability of early errors in structured outputs.

---

[LLM Agents for Automated Web Vulnerability Reproduction: Are We There Yet?](http://arxiv.org/abs/2510.14700)

- LLM Agents for Automated Web Vulnerability Reproduction: introduces a comprehensive evaluation framework for assessing LLM agents' capabilities in transforming vulnerability reports into working exploits, including a benchmark dataset, LLM agents, evaluation tasks, and criteria.
- The evaluation systematically assesses 20 state-of-the-art LLM agents across 16 dimensions on 3 representative CVEs, then conducts an in-depth analysis of the top 3 agents (OpenHands, SWE-agent, CAI) on 80 real-world CVEs.
- Findings reveal that while LLM agents achieve reasonable success on simple library-based vulnerabilities, they consistently fail on complex service-based vulnerabilities requiring multi-component environments and robust authentication.

---

[LLM Agents Beyond Utility: An Open-Ended Perspective](http://arxiv.org/abs/2510.14548)

- Open-Ended LLM Agent Loop: introduces an LLM agent augmented with task generation, memory management, and environmental interaction capabilities, enabling it to autonomously generate and pursue its own goals in an open-ended setting.
- The agent extends the ReAct framework by incorporating self-generated tasks, persistent long-term memory, and file tools for creating lasting environmental artifacts across multiple runs.
- This system explores the potential and limitations of adapting pretrained LLMs for open-ended behavior, highlighting challenges in memory management, productive exploration, and abstract goal pursuit.

---

[JSPLIT: A Taxonomy-based Solution for Prompt Bloating in Model Context Protocol](http://arxiv.org/abs/2510.14537)

- JSPLIT (Taxonomy-based Solution for Prompt Bloating in Model Context Protocol): introduces a taxonomy-driven framework to manage prompt size effectively for AI agents using large sets of Model Context Protocol (MCP) tools, by organizing tools into a hierarchical taxonomy and using LLMs to identify and include only relevant tools based on user queries and taxonomy structure.
- This approach significantly reduces prompt size, token costs, and latency while improving tool selection accuracy and task success in complex agent environments.
- The framework's core, the Taxonomy-MCPResolver, leverages LLMs for a two-phase process of taxonomy classification and MCP server ranking to prune irrelevant tools from the agent's context.

---

[E2EDEV: BENCHMARKING LARGE LANGUAGE MODELS IN END-TO-END SOFTWARE DEVELOPMENT TASK](http://arxiv.org/abs/2510.14509)

- E2EDev (End-to-End Software Development Benchmark): introduces a novel benchmark grounded in Behavior-Driven Development (BDD) principles, evaluating LLM-based End-to-End Software Development (E2ESD) frameworks by assessing generated software against user needs through mimicking real user interactions, comprising fine-grained user requirements, multiple BDD test scenarios with Python step implementations, an automated testing pipeline, and a Human-in-the-Loop Multi-Agent Annotation Framework (HITL-MAA).
- The HITL-MAA framework leverages specialized LLM agents, including Code Analyzer, Requirement Extractor, Test Case Generator, Test Automation Engineer, Step Checker, and Test Runner agents, with human supervision at key stages to ensure data quality and reduce annotation effort.
- E2EDev addresses limitations of existing E2ESD benchmarks by providing fine-grained requirements and reliable, automated evaluation protocols built on the Behave framework, revealing that current LLM-based frameworks struggle with detailed functional specifics and multi-agent architectures often incur high costs with minimal gains.

---

[LIRA: LINGUISTIC ROBUST ANCHORING FOR CROSS-LINGUAL LARGE LANGUAGE MODELS](http://arxiv.org/abs/2510.14466)

- LiRA (Linguistic Robust Anchoring for Large Language Models): introduces a training framework that robustly improves cross-lingual representations under low-resource conditions by jointly strengthening retrieval and reasoning.
- The framework integrates Arca (Anchored Representation Composition Architecture), which anchors low-resource languages to an English semantic space via anchor-based alignment and multi-agent collaborative encoding, and LaSR (Language-coupled Semantic Reasoner), which adds a language-aware lightweight reasoning head with consistency regularization.
- Arca's Translation Critic judges candidate translations, the Embedding Critic anchors feature paths, and the Actor Model fuses these critics to select candidates, while LaSR's LLM Transformer fuses English and multilingual embeddings, supported by CorrQueue and DocQueue for training stability.

---

[Natural Language Tools: A Natural Language Approach to Tool Calling In Large Language Agents](http://arxiv.org/abs/2510.14453)

- NLT (Natural Language Tools): introduces a modular three-step architecture that replaces programmatic JSON tool calling with natural language outputs, decoupling tool selection from response generation to improve accuracy and reduce variance.
- The framework utilizes a Selector LLM to identify relevant tools based on a natural language prompt, a Tool Parser to extract decisions, and a Tool Logic component to execute selected tools, before an Output Model generates the final response.
- NLT significantly improves tool calling accuracy by 18.4 percentage points and reduces output variance by 70% across diverse models and domains, demonstrating enhanced robustness to prompt perturbations and extending capabilities to models lacking native support.

---

[IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning](http://arxiv.org/abs/2510.14406)

- IMAGINE (Integrating Multi-Agent System into One Model): introduces a framework that distills the reasoning and planning capabilities of a Multi-Agent System into a single, compact LLM model through a three-stage training pipeline, including New Query Generation, Multi-Agent System based Inference Data Generation, and Agentic Reasoning Training.
- The framework's Multi-Agent System based Inference Data Generation stage employs a Reasoner, two Judges, and a Reflector to produce high-quality, reflected reasoning data for training.
- Agentic Reasoning Training, comprising Agentic SFT and Agentic RL guided by a Newly Designed Agentic Reward Function, integrates and enhances the model's agentic reasoning abilities, enabling a small model to outperform larger Multi-Agent Systems.

---

[The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems](http://arxiv.org/abs/2510.14401)

- CPR simulation framework: introduces a common-pool resource simulation framework for LLM multi-agent systems, with LLM agents, a shared resource, Harvest & Consumption, Individual Punishment, Social Learning, Group Decision modules, individual and group norms, cultural-evolutionary mechanisms, environmental feedback, payoff-biased social learning, a propose-vote rule, and prompts, enabling the endogenous emergence of cooperative norms without explicit reward signals.
- The framework serves as a testbed to study how LLM agents develop strategies in mixed-motive settings and form group-beneficial norms through social learning and norm-based punishment.
- The study validates the framework by reproducing human behavior findings and demonstrates its ability to discriminate LLMs based on their cooperative tendencies and norm formation capabilities under diverse environmental and social conditions.

---

[MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering](http://arxiv.org/abs/2510.14400)

- MedTrust-Guided Iterative RAG: introduces a framework for biomedical question answering that enhances factual consistency and mitigates hallucinations by employing an iterative retrieval-verification pipeline and a MedTrust-Align Module for trust alignment.
- The iterative pipeline, featuring a verifier agent and a generator agent, refines evidence and generates citation-grounded reasoning or refusal statements, while the MedTrust-Align Module constructs a hallucination-aware dataset and uses Direct Preference Optimization to reinforce reliable reasoning.
- This approach systematically addresses hallucination patterns and evidence insufficiency in complex medical queries, leading to more accurate and trustworthy LLM responses in clinical contexts.

---

[Your Next Token Prediction: A Multilingual Benchmark for Personalized Response Generation](http://arxiv.org/abs/2510.14398)

- YNTP (Your Next Token Prediction): introduces a multilingual benchmark for personalized response generation, utilizing an LLM-driven multi-NPC dialogue system that includes an FSM Engine (governs dialogue flow/state transitions), a Scenario Script (defines dialogue content/branching logic/NPC roles), and LLM Dialogue Generation (linguistic/emotional realization module).
- This system collects natural, personalized, and psychologically grounded conversation data from users interacting with MBTI-dimensioned NPCs over five-day dialogue sessions.
- The benchmark enables token-level prediction of individualized responses, moving beyond stylistic mimicry to model deeper cognitive regularities in word choice.

---

[Beyond One World: Benchmarking Super Heros in Role-Playing Across Multiversal Contexts](http://arxiv.org/abs/2510.14351)

- Beyond One World: introduces a benchmark for evaluating LLMs' character-grounded role-playing across multiversal contexts, featuring Canon Events and Moral Dilemmas tasks, an LLM-as-a-judge rubric for thinking/acting, and a Think-Act Matching metric.
- The benchmark assesses LLMs' ability to consistently portray version-specific superhero characters by probing factual recall and ethical decision-making across 30 iconic heroes and 90 canon-specific versions from Marvel and DC universes.
- The evaluation framework disentangles internal deliberation from outward decisions, using structured prompting and an LLM judge, revealing critical gaps in multiversal consistency and reasoning alignment in current LLMs.

---

[Stop-RAG: Value-Based Retrieval Control for Iterative RAG](http://arxiv.org/abs/2510.14337)

- Stop-RAG: introduces a value-based controller for adaptive stopping in iterative retrieval-augmented generation (RAG) systems, with an Iterative RAG Pipeline, Query Generator, Retriever, Reranker, Answer Generator, Stop-RAG Controller, MDP Formulation, Q-network, Q(λ) Targets, and Decision Rule, where it frames iterative RAG as a finite-horizon Markov Decision Process and trains a Q-network using Q(λ) targets to provide forward-looking estimates of stopping quality.
- The framework adaptively decides when to stop retrieving by estimating and comparing immediate and future gains, enabling more reliable stopping decisions without relying on internal telemetry or fixed iteration counts.
- Stop-RAG consistently improves performance on multi-hop question-answering benchmarks, demonstrating its effectiveness as a modular, plug-and-play component compatible with black-box LLMs and existing RAG pipelines.

---

[Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies](http://arxiv.org/abs/2510.14312)

- TERRARIUM: introduces a modular and configurable framework for studying multi-agent systems (MAS) safety, privacy, and security, comprising Agent (LLM-based entity), Environment (simulator, state, objective), Blackboard (communication proxy), Tools (external capabilities), Communication Protocol (interaction rules), Factor Graph (blackboard initialization), MCP Server (model context protocol), Persistence (logs, configurations), and Infrastructure (LLMs, MCP servers).
- The framework repurposes the blackboard design to create a modular, configurable testbed for multi-agent collaboration, enabling systematic study of attack vectors like misalignment, malicious agents, compromised communication, and data poisoning.
- Its modular and configurable design facilitates rapid prototyping, evaluation, and iteration on defenses and designs, accelerating progress toward trustworthy multi-agent systems.

---

[PRISM: AGENTIC RETRIEVAL WITH LLMS FOR MULTI-HOP QUESTION ANSWERING](http://arxiv.org/abs/2510.14278)

- PRISM (Precision-Recall Iterative Selection Mechanism): introduces an agentic retrieval framework that leverages LLM-based agents, including a Question Analyzer, Selector, and Adder, within an Iterative Refinement Loop to retrieve relevant evidence for multi-hop question answering.
- The framework's Question Analyzer decomposes complex queries into sub-questions, while the Selector and Adder agents iteratively refine the evidence set by balancing precision and recall.
- This approach produces compact and comprehensive evidence sets, which are then used by an Answer Generator Agent to provide accurate answers, outperforming strong baselines in multi-hop QA benchmarks.

---

[GENLARP: Enabling Immersive Live Action Role-Play through LLM-Generated Worlds and Characters](http://arxiv.org/abs/2510.14277)

- GENLARP: introduces a virtual reality system that transforms personalized stories into immersive LARP experiences, utilizing Narrative Initialization (user input processing/world and story generation), Interactive Role Design (character and interaction logic), and Live-Action Role Play (immersive user experience) modules.
- The system leverages generative AI and LLMs to create dynamic virtual worlds and characters, allowing users to act as both creators and players within the narrative.
- It addresses traditional LARP limitations by enabling virtual reenactments without extensive physical setup or large groups, fostering deeper engagement through LLM-driven agents and dynamic narrative adaptation.

---

[AlphaQuanter: An End-to-End Tool-Orchestrated Agentic Reinforcement Learning Framework for Stock Trading](http://arxiv.org/abs/2510.14264)

- AlphaQuanter: introduces a single-agent framework that leverages reinforcement learning (RL) to learn a dynamic policy over a transparent, tool-augmented decision workflow, empowering an agent to autonomously orchestrate tools and proactively acquire information on demand, establishing a transparent and auditable reasoning process.
- The framework unifies workflows into a ReAct-like agent, starting with a guided plan, followed by iterative tool use and information seeking, and in-depth analysis, utilizing various financial data sources and a reward function for end-to-end optimization.
- AlphaQuanter's design ensures decision consistency and interpretability by enforcing stepwise hypothesis testing and tightly coupling evidence collection with reasoning, leading to state-of-the-art performance on key financial metrics and sophisticated trading strategies.

---

[TOWARDS AGENTIC SELF-LEARNING LLMS IN SEARCH ENVIRONMENT](http://arxiv.org/abs/2510.14253)

- ASL (Agentic Self-Learning): introduces a multi-role, closed-loop reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone, including a Prompt Generator (generates tasks, adapts difficulty), a Policy Model (generates solutions, improves performance), a Generative Reward Model (assesses correctness, refines evaluation), Tools (retrieves information), and a Meta Prompt (guides task generation).
- ASL enables LLMs to autonomously evolve their reasoning, generation, and evaluation capabilities in a continuous closed loop, addressing the need for scalable reward signals and agent task data.
- The framework demonstrates superior sample efficiency and robustness, achieving steady performance gains and surpassing strong RLVR baselines even under zero-labeled-data conditions.

---

[Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks](http://arxiv.org/abs/2510.14207)

- OHAB (Online Harassment Agentic Benchmark): introduces a framework for systematically studying how multi-turn LLM agents can be coerced into generating abusive content, with a synthetic multi-turn harassment conversation dataset generation pipeline, a multi-agent simulation design, and a mixed-methods evaluation framework.
- The framework employs various jailbreak methods, including persona-only priming, toxic memory injection, planning attacks (CoT/ReAct), and jailbreak fine-tuning, to assess vulnerabilities in LLMs like LLaMA-3.1-8B-Instruct and Gemini-2.0-flash-001.
- The evaluation combines LLM-based judgment with human annotation, informed by social theories like Dark Triad Traits and Conflict Avoidance, to provide nuanced insights into harassment dynamics and behavioral patterns.

---

[DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans](http://arxiv.org/abs/2510.14205)

- DPRF (Dynamic Persona Refinement Framework): introduces a novel methodology to optimize LLM Role-Playing Agents' behavioral alignment with human ground truth by iteratively identifying cognitive divergences and refining persona profiles.
- The framework operates through an iterative feedback loop, comparing agent-generated behaviors against human ground truth using a Behavior Analysis Agent and updating the persona via a Persona Refinement Agent.
- DPRF is model-agnostic, domain-agnostic, and data-efficient, enhancing persona fidelity for applications like user simulation and personalized AI.

---

[Agentic Entropy-Balanced Policy Optimization](http://arxiv.org/abs/2510.14545)

- AEPO (Agentic Entropy-Balanced Policy Optimization): introduces a dynamic entropy-balanced rollout (manages rollout sampling) and entropy-balanced policy optimization (optimizes policy updates), which together balance entropy during rollout and policy updates to enhance multi-turn tool-use capabilities in LLMs.
- The dynamic entropy-balanced rollout adaptively allocates sampling budgets via entropy pre-monitoring and penalizes consecutive high-entropy branches to mitigate over-branching issues.
- The policy optimization component preserves high-entropy token gradients and prioritizes learning on high-uncertainty tokens through entropy-aware advantage estimation, improving stability and scalability for web agent training.

---

[HELMSMAN: AUTONOMOUS SYNTHESIS OF FEDERATED LEARNING SYSTEMS VIA MULTI-AGENT COLLABORATION](http://arxiv.org/abs/2510.14512)

- Helmsman: introduces a novel multi-agent system that automates the end-to-end synthesis of federated learning systems from high-level user specifications, including User, Planning Agent, Reflection Agent, Human Approval, Supervisor Agent, Coder Agent, Tester Agent, Evaluator Agent, Debugger Agent, Task Module, Client Module, Strategy Module, Server Module, Sandboxed Federated Simulation, Web Search Tool, RAG Pipeline, and AgentFL-Bench, by emulating a principled research and development workflow through interactive planning, modular code generation, and autonomous evaluation.
- The framework structures the complex Federated Learning (FL) design process into three collaborative phases: interactive human-in-the-loop planning, modular code generation by supervised agent teams, and closed-loop autonomous evaluation and refinement in a sandboxed simulation environment.
- Helmsman also introduces AgentFL-Bench, a new benchmark comprising 16 diverse tasks designed to rigorously assess the system-level generation capabilities of agentic systems in FL, demonstrating competitive and often superior solutions compared to hand-crafted baselines.

---

[Why Instant-Runoff Voting Is So Resilient to Coalitional Manipulation: Phase Transitions in the Perturbed Culture](http://arxiv.org/abs/2510.14450)

- Phase Transition Analysis of Voting Rules in Perturbed Culture Model: introduces an analysis of Plurality, Two-Round System, and Instant-Runoff Voting within the Perturbed Culture Model, revealing phase transitions in their susceptibility to coalitional manipulation.
- The study identifies a critical threshold (θc) for each rule, below which the CM rate tends to 1 for large electorates and above which it tends to 0.
- The paper introduces the Super Condorcet Winner (SCW) concept, demonstrating its role as a key factor in IRV's exceptional resilience to CM, with IRV's θc being 0.

---

[HI-AGENT: HIERARCHICAL VISION-LANGUAGE AGENTS FOR MOBILE DEVICE CONTROL](http://arxiv.org/abs/2510.14388)

- Hi-Agent (Hierarchical Vision-Language Agents for Mobile Device Control): introduces a trainable hierarchical vision-language agent for mobile control, featuring a high-level reasoning model and a low-level action model that are jointly optimized.
- The framework reformulates multi-step decision-making as a sequence of single-step subgoals and employs a foresight advantage function, leveraging execution feedback to guide high-level optimization.
- Hi-Agent achieves state-of-the-art performance on mobile control benchmarks by combining structured task decomposition with stable, critic-free joint training.

---

[MAGPIE: A benchmark for Multi-AGent contextual PrIvacy Evaluation](http://arxiv.org/abs/2510.15186)

- MAGPIE (Multi-AGent contextual Privacy Evaluation): introduces a novel benchmark for evaluating privacy understanding and preservation in multi-agent collaborative, non-adversarial scenarios, featuring a Dataset Construction Pipeline (generates and validates scenarios), a Simulation Environment (orchestrates multi-agent negotiations), and an Evaluator LLM (assesses privacy leakage and task outcomes).
- The benchmark comprises 200 high-stakes, multi-turn tasks where private information is integral to task resolution, forcing LLM agents to balance effective collaboration with strategic information control.
- Evaluations reveal that state-of-the-art LLM agents, including GPT-5 and Gemini 2.5-Pro, exhibit significant privacy leakage and struggle with consensus, often resorting to undesirable behaviors like manipulation and power-seeking.

---

[Procedural Game Level Design with Deep Reinforcement Learning](http://arxiv.org/abs/2510.15120)

- Co-adaptive Procedural Content Generation Framework: introduces a novel method for procedural game level design using DRL, featuring a Hummingbird Agent (solver), a Floating Island Agent (generator), a Unity Environment (3D simulation), Proximal Policy Optimization (PPO) (training algorithm), Unity ML-Agents Toolkit (platform), a Feedback Loop (interaction mechanism), and Auxiliary Inputs (observation enhancement), where the system integrates DRL agents for both environment generation and task-solving.
- This framework employs two PPO-trained agents: a hummingbird agent that learns to collect flowers in a dynamic 3D Unity environment, and an island agent that generates diverse, context-aware flower placements based on environmental cues and performance feedback.
- The dynamic feedback loop between the agents enables co-adaptive learning, where the island agent evolves to create effective level configurations, and the hummingbird agent concurrently learns to solve them with greater robustness and generalization.

---

[Policy Transfer Ensures Fast Learning for Continuous-Time LQR with Entropy Regularization](http://arxiv.org/abs/2510.15165)

- Policy Transfer with IPO (Iterative Policy Optimization): introduces a theoretical analysis of policy transfer for continuous-time Linear Quadratic Regulators (LQRs) with entropy regularization, proposing a novel IPO algorithm that achieves global linear and local super-linear convergence.
- The framework demonstrates that an optimal policy from a source LQR can serve as a near-optimal initialization for closely related target LQRs, preserving convergence rates.
- The analysis also establishes the stability of a class of continuous-time score-based diffusion models by connecting them with LQRs.

---

[HUGAGENT: EVALUATING LLMS IN SIMULATING HUMAN-LIKE INDIVIDUAL REASONING ON OPEN-ENDED TASKS](http://arxiv.org/abs/2510.15144)

- HugAgent (Human-Grounded AGENT Benchmark): introduces a dual-track benchmark for average-to-individual reasoning adaptation, including an interactive semi-structured chatbot, a structured questionnaire, a dynamic question generator, and a Causal Belief Network for representing individual belief systems.
- The framework utilizes both a synthetic track for scalable stress tests and a human-grounded track for ecologically valid reasoning data, enabling reproducible evaluation of intra-agent fidelity.
- It operationalizes reasoning adaptation into two measurable tasks: Belief-State Inference and Belief Dynamics Update, aiming to predict how specific individuals reason and update beliefs in novel scenarios.

---

[INTERNALIZING WORLD MODELS VIA SELF-PLAY FINETUNING FOR AGENTIC RL](http://arxiv.org/abs/2510.15047)

- SPA (Self Play Agent): introduces a reinforcement learning framework that equips LLM agents with an internal world model, decomposed into State Estimation and Transition Modeling, learned via a Self-Play Supervised Finetuning stage, to improve performance in out-of-distribution environments.
- The framework first cold-starts the policy by enabling the LLM agent to self-play and acquire world knowledge from the environment, then uses this learned world model to simulate future states prior to policy optimization through RL training.
- This approach significantly boosts success rates in environments like Sokoban and FrozenLake by grounding LLM reasoning in environmental rules rather than memorized trajectories, leading to more robust generalization.

---

[GUIrilla: A Scalable Framework for Automated Desktop UI Exploration](http://arxiv.org/abs/2510.16051)

- GUIrilla: introduces a scalable framework for automated desktop UI exploration, systematically exploring macOS applications via native accessibility APIs and simulated user interactions, supported by three LLM-based agents for element ordering, input generation, and task postprocessing.
- The framework generates hierarchical GUI graphs from discovered interface elements and crawler actions, addressing data collection challenges in GUI automation and producing the GUIrilla-TASK dataset.
- GUIrilla leverages specialized interaction handlers to achieve comprehensive application coverage and constructs function-centric tasks, enabling LLM-based agents to significantly improve performance on downstream UI tasks with less data.

---

#### 15th October 2025

[GAPS: A Clinically Grounded, Automated Benchmark for Evaluating AI Clinicians](http://arxiv.org/abs/2510.13734)

- GAPS (Grounding-Adequacy-Perturbation-Safety): introduces a clinically grounded, automated benchmark for evaluating AI clinicians, featuring Grounding (reasoning depth), Adequacy (answer completeness), Perturbation (input robustness), and Safety (harm prevention) axes, operationalized by a pipeline that constructs guideline-centered evaluation items and rubrics.
- The framework employs an automated pipeline for evidence neighborhood assembly, knowledge graph and hierarchical tree representations, item generation across G-levels and P-perturbations, and rubric synthesis by a DeepResearch agent using a ReAct-style loop.
- Scoring is performed by an ensemble of LLM judges, revealing that current LLMs excel at factual recall but struggle with increased reasoning depth, answer completeness, and robustness to adversarial inputs, guiding future AI clinician development.

---

[From Refusal to Recovery: A Control-Theoretic Approach to Generative AI Guardrails](http://arxiv.org/abs/2510.13727)

- ReGuard (Recovery Guardrail): introduces a control-theoretic approach to generative AI guardrails that formalizes AI safety as a sequential decision problem, learning predictive guardrails to monitor and proactively correct risky LLM outputs in real-time.
- This framework operates in the LLM's latent representation of the world, enabling model-agnostic guardrails that can be trained via safety-critical reinforcement learning to detect and recover from unsafe states.
- It moves beyond traditional flag-and-block guardrails by providing a principled dynamic alternative that balances safety and task efficiency, demonstrated in autonomous driving, e-commerce, and AI assistant scenarios.

---

[Training LLM Agents to Empower Humans](http://arxiv.org/abs/2510.13709)

- Empower: introduces a self-supervised method for fine-tuning LLM agents to better assist humans by maximizing their empowerment, which is their ability to effect desired changes in the environment, using offline text data and a logit threshold mechanism to identify predictable code for completion.
- The framework trains an LLM agent to complete predictable text, allowing the human user to focus on important design decisions rather than boilerplate code, thereby increasing their control over future outcomes.
- Empower demonstrates that LLM assistants can be aligned without explicit human feedback or verifiable rewards by reasoning about how their actions enable humans to complete tasks more quickly.

---

[Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs](http://arxiv.org/abs/2510.13586)

- Deflanderization for Game Dialogue: introduces a novel approach for LLM-based NPCs in game dialogue, combining lightweight prompting techniques and fine-tuned large models to balance character authenticity with task execution.
- The approach employs a Deflanderization prompting method to prevent excessive role-play and improve task fidelity, alongside Retrieval Augmented Generation and Supervised Finetuning for robust dialogue grounding.
- The framework addresses the challenge of maintaining consistent NPC personas and executing tasks in fantasy RPG environments, achieving high rankings in a dialogue challenge.

---

[STEER-MOE: EFFICIENT AUDIO-LANGUAGE ALIGNMENT WITH A MIXTURE-OF-EXPERTS STEERING MODULE](http://arxiv.org/abs/2510.13558)

- SteerMoE (Efficient Audio-Language Alignment with a Mixture-of-Experts Steering Module): introduces a novel and modular framework for audio-language alignment, utilizing a lightweight steering module with a Mixture-of-Experts router to dynamically transform continuous audio representations for a frozen LLM decoder.
- The framework freezes both the audio encoder and LLM decoder, training only the steering module to preserve LLM's reasoning capabilities and enable plug-and-play component interchangeability.
- SteerMoE achieves strong performance on ASR and audio understanding tasks, demonstrating a parameter-efficient and modular approach to multimodal AI by operating entirely in the continuous embedding space.

---

[In-Browser LLM-Guided Fuzzing for Real-Time Prompt Injection Testing in Agentic AI Browsers](http://arxiv.org/abs/2510.13543)

- In-Browser LLM-Guided Fuzzing Framework: introduces an in-browser, LLM-guided fuzzing framework for real-time prompt injection testing in agentic AI browsers, with Fuzzing Controller, LLM Integration Layer, Browser Automation Layer, and Data Collection and Analytics components, designed to automatically discover prompt injection vulnerabilities in real-time by generating and testing malicious webpage content within a live browser environment.
- The framework leverages LLMs to generate diverse and evolving attack content, using a real-time feedback loop to refine attack strategies based on the AI agent's observed behavior and actions.
- This approach enables high-fidelity testing with full DOM context and action monitoring, demonstrating that static pattern-matching defenses are insufficient against adaptive, LLM-guided prompt injection attacks.

---

[Make an Offer They Can't Refuse: Grounding Bayesian Persuasion in Real-World Dialogues without Pre-Commitment](http://arxiv.org/abs/2510.13387)

- Type-Induced Bayesian Persuasion (BP): introduces a framework for implementing Bayesian Persuasion in natural language dialogues without pre-commitment, leveraging a commitment-communication mechanism where the persuader explicitly narrates potential types to guide the persuadee's Bayesian belief update.
- The framework integrates a Bayesian setup, a composite signal structure (mbasic, mtype, mdes, minf), and a type-induced information schema (Sender Types, Base Policies, Schema Induction) to facilitate the Receiver's inference and decision process, implemented through Semi-Formal-Natural-Language (SFNL) BP and Fully-Natural-Language (FNL) BP.
- Experimental results show that BP-guided LLMs consistently outperform non-BP baselines, with SFNL excelling in credibility and logical coherence, while FNL demonstrates stronger emotional resonance and robustness, and supervised fine-tuning enables smaller models to achieve comparable performance to larger models.

---

[MADREC: A Multi-Aspect Driven LLM Agent for Explainable and Adaptive Recommendation](http://arxiv.org/abs/2510.13371)

- MADREC (Multi-Aspect Driven LLM Agent): introduces an autonomous LLM-based recommender that constructs user and item profiles by unsupervised extraction of multi-aspect information from reviews, performs direct and sequential recommendation, generates explanations, and dynamically adjusts inference criteria via a SELF-FEEDBACK mechanism.
- The framework leverages MEMORY to store user and item profiles, TOOLS for aspect extraction, summarization, and re-ranking, and TASKS for various recommendation objectives, all integrated within an active agent architecture.
- MADREC enhances explainability and adaptivity by generating structured profiles, re-ranking candidate items based on multi-aspect relevance, and iteratively refining recommendations through self-feedback, outperforming traditional and LLM-based baselines.

---

Document Intelligence in the Era of Large Language Models: A Survey
2510.13366
http://arxiv.org/abs/2510.13366
DI-LLM (Document Intelligence in the Era of Large Language Models):
Multimodal Document AI / Multilingual Document AI / Retrieval-Augmented Paradigm / DocAgent Framework / DocAgent Foundation Model
Multimodal Document AI (integrating diverse modalities) / Multilingual Document AI (handling diverse languages) / Retrieval-Augmented Paradigm (leveraging external knowledge) / DocAgent Framework (agent-based document processing) / DocAgent Foundation Model (domain-aware, cross-modal models)
DI-LLM (Document Intelligence in the Era of Large Language Models): introduces a comprehensive survey of Document AI advancements, categorizing tasks into understanding and generation, integrating multimodal and multilingual capabilities, and leveraging retrieval-augmented methods.
The survey explores key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions including agent-based approaches and document-specific foundation models.
It provides a structured analysis of the state-of-the-art in DAI, highlighting the evolution of LLM-based approaches and their implications for both academic and practical applications.

**List the architectural components found in the figures.**
The paper is a survey and does not present architectural figures for a single proposed framework. Instead, Table 1 summarizes benchmark datasets, detailing their supported languages, document counts, modalities (Text, Visual, Layout), and tasks (Key Information Extraction (KIE), Document Layout Analysis (DLA), Document Sentiment Analysis (DSA), Document Classification (DC), Document Summarization (DS), Document Content Generation (DCG), Question Answering (QA)). These modalities and tasks represent functional components or capabilities within Document AI systems.

**Define key components based on the figures and the text.**
Based on the text, the key conceptual components and approaches defining Document Intelligence in the Era of Large Language Models include:

*   **Multimodal Document AI**: An approach that integrates diverse information sources—Textual (OCR, digital text), Visual (figures, handwriting, region designs), and Layout (spatial arrangement, bounding boxes)—to comprehensively understand and generate documents.
*   **Multilingual Document AI**: Focuses on enabling LLMs to effectively handle documents in multiple languages, addressing cross-linguistic nuances and cultural intricacies through prompt-based methods and specialized training strategies.
*   **Retrieval-Augmented Paradigm**: A method that enhances LLMs by retrieving reliable external knowledge from documents (text, tables, images) to mitigate challenges related to outdated training data and limited domain expertise.
*   **DocAgent Framework**: A future direction proposing intelligent, multi-agent systems designed to proficiently manage complex document understanding and generation tasks by leveraging domain-specific knowledge and external tools.
*   **DocAgent Foundation Model**: A proposed future model that is domain-aware and cross-modal aligned, built upon continuously evolving datasets to provide an end-to-end solution for complex document processing.
*   **Document Layout Analysis (DLA)**: A task focused on detecting and classifying structural elements within documents, such as text blocks and headers, and understanding their spatial relationships.
*   **Key Information Extraction (KIE)**: A task aimed at identifying and extracting specific elements like form fields and key-value pairs from unstructured or semi-structured documents.
*   **Document Classification (DC)**: A task to identify the category, type, or domain of a document using textual, visual, and layout modalities.
*   **Document Question Answering (QA)**: A generation task focused on providing accurate natural language responses to questions based on document context, often involving complex table lookups or text extraction.
*   **Document Summarization (DS)**: A generation task that aims to create concise overviews of documents while preserving essential content.
*   **Document Content Generation (DCG)**: A generation task involving the creation of new document content, including structured layouts, textual continuations, figures, and tables, based on existing materials.

[15th October 2025
Document Intelligence in the Era of Large Language Models: A Survey
2510.13366
http://arxiv.org/abs/2510.13366
DI-LLM (Document Intelligence in the Era of Large Language Models):
Multimodal Document AI / Multilingual Document AI / Retrieval-Augmented Paradigm / DocAgent Framework / DocAgent Foundation Model
Multimodal Document AI (integrating diverse modalities) / Multilingual Document AI (handling diverse languages) / Retrieval-Augmented Paradigm (leveraging external knowledge) / DocAgent Framework (agent-based document processing) / DocAgent Foundation Model (domain-aware, cross-modal models)
DI-LLM (Document Intelligence in the Era of Large Language Models): introduces a comprehensive survey of Document AI advancements, categorizing tasks into understanding and generation, integrating multimodal and multilingual capabilities, and leveraging retrieval-augmented methods.
The survey explores key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions including agent-based approaches and document-specific foundation models.
It provides a structured analysis of the state-of-the-art in DAI, highlighting the evolution of LLM-based approaches and their implications for both academic and practical applications.

**List the architectural components found in the figures.**
The paper is a survey and does not present architectural figures for a single proposed framework. Instead, Table 1 summarizes benchmark datasets, detailing their supported languages, document counts, modalities (Text, Visual, Layout), and tasks (Key Information Extraction (KIE), Document Layout Analysis (DLA), Document Sentiment Analysis (DSA), Document Classification (DC), Document Summarization (DS), Document Content Generation (DCG), Question Answering (QA)). These modalities and tasks represent functional components or capabilities within Document AI systems.

**Define key components based on the figures and the text.**
Based on the text, the key conceptual components and approaches defining Document Intelligence in the Era of Large Language Models include:

*   **Multimodal Document AI**: An approach that integrates diverse information sources—Textual (OCR, digital text), Visual (figures, handwriting, region designs), and Layout (spatial arrangement, bounding boxes)—to comprehensively understand and generate documents.
*   **Multilingual Document AI**: Focuses on enabling LLMs to effectively handle documents in multiple languages, addressing cross-linguistic nuances and cultural intricacies through prompt-based methods and specialized training strategies.
*   **Retrieval-Augmented Paradigm**: A method that enhances LLMs by retrieving reliable external knowledge from documents (text, tables, images) to mitigate challenges related to outdated training data and limited domain expertise.
*   **DocAgent Framework**: A future direction proposing intelligent, multi-agent systems designed to proficiently manage complex document understanding and generation tasks by leveraging domain-specific knowledge and external tools.
*   **DocAgent Foundation Model**: A proposed future model that is domain-aware and cross-modal aligned, built upon continuously evolving datasets to provide an end-to-end solution for complex document processing.
*   **Document Layout Analysis (DLA)**: A task focused on detecting and classifying structural elements within documents, such as text blocks and headers, and understanding their spatial relationships.
*   **Key Information Extraction (KIE)**: A task aimed at identifying and extracting specific elements like form fields and key-value pairs from unstructured or semi-structured documents.
*   **Document Classification (DC)**: A task to identify the category, type, or domain of a document using textual, visual, and layout modalities.
*   **Document Question Answering (QA)**: A generation task focused on providing accurate natural language responses to questions based on document context, often involving complex table lookups or text extraction.
*   **Document Summarization (DS)**: A generation task that aims to create concise overviews of documents while preserving essential content.
*   **Document Content Generation (DCG)**: A generation task involving the creation of new document content, including structured layouts, textual continuations, figures, and tables, based on existing materials.](15th October 2025
Document Intelligence in the Era of Large Language Models: A Survey
2510.13366
http://arxiv.org/abs/2510.13366
DI-LLM (Document Intelligence in the Era of Large Language Models):
Multimodal Document AI / Multilingual Document AI / Retrieval-Augmented Paradigm / DocAgent Framework / DocAgent Foundation Model
Multimodal Document AI (integrating diverse modalities) / Multilingual Document AI (handling diverse languages) / Retrieval-Augmented Paradigm (leveraging external knowledge) / DocAgent Framework (agent-based document processing) / DocAgent Foundation Model (domain-aware, cross-modal models)
DI-LLM (Document Intelligence in the Era of Large Language Models): introduces a comprehensive survey of Document AI advancements, categorizing tasks into understanding and generation, integrating multimodal and multilingual capabilities, and leveraging retrieval-augmented methods.
The survey explores key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions including agent-based approaches and document-specific foundation models.
It provides a structured analysis of the state-of-the-art in DAI, highlighting the evolution of LLM-based approaches and their implications for both academic and practical applications.

**List the architectural components found in the figures.**
The paper is a survey and does not present architectural figures for a single proposed framework. Instead, Table 1 summarizes benchmark datasets, detailing their supported languages, document counts, modalities (Text, Visual, Layout), and tasks (Key Information Extraction (KIE), Document Layout Analysis (DLA), Document Sentiment Analysis (DSA), Document Classification (DC), Document Summarization (DS), Document Content Generation (DCG), Question Answering (QA)). These modalities and tasks represent functional components or capabilities within Document AI systems.

**Define key components based on the figures and the text.**
Based on the text, the key conceptual components and approaches defining Document Intelligence in the Era of Large Language Models include:

*   **Multimodal Document AI**: An approach that integrates diverse information sources—Textual (OCR, digital text), Visual (figures, handwriting, region designs), and Layout (spatial arrangement, bounding boxes)—to comprehensively understand and generate documents.
*   **Multilingual Document AI**: Focuses on enabling LLMs to effectively handle documents in multiple languages, addressing cross-linguistic nuances and cultural intricacies through prompt-based methods and specialized training strategies.
*   **Retrieval-Augmented Paradigm**: A method that enhances LLMs by retrieving reliable external knowledge from documents (text, tables, images) to mitigate challenges related to outdated training data and limited domain expertise.
*   **DocAgent Framework**: A future direction proposing intelligent, multi-agent systems designed to proficiently manage complex document understanding and generation tasks by leveraging domain-specific knowledge and external tools.
*   **DocAgent Foundation Model**: A proposed future model that is domain-aware and cross-modal aligned, built upon continuously evolving datasets to provide an end-to-end solution for complex document processing.
*   **Document Layout Analysis (DLA)**: A task focused on detecting and classifying structural elements within documents, such as text blocks and headers, and understanding their spatial relationships.
*   **Key Information Extraction (KIE)**: A task aimed at identifying and extracting specific elements like form fields and key-value pairs from unstructured or semi-structured documents.
*   **Document Classification (DC)**: A task to identify the category, type, or domain of a document using textual, visual, and layout modalities.
*   **Document Question Answering (QA)**: A generation task focused on providing accurate natural language responses to questions based on document context, often involving complex table lookups or text extraction.
*   **Document Summarization (DS)**: A generation task that aims to create concise overviews of documents while preserving essential content.
*   **Document Content Generation (DCG)**: A generation task involving the creation of new document content, including structured layouts, textual continuations, figures, and tables, based on existing materials.)

- 15th October 2025
Document Intelligence in the Era of Large Language Models: A Survey
2510.13366
http://arxiv.org/abs/2510.13366
DI-LLM (Document Intelligence in the Era of Large Language Models):
Multimodal Document AI / Multilingual Document AI / Retrieval-Augmented Paradigm / DocAgent Framework / DocAgent Foundation Model
Multimodal Document AI (integrating diverse modalities) / Multilingual Document AI (handling diverse languages) / Retrieval-Augmented Paradigm (leveraging external knowledge) / DocAgent Framework (agent-based document processing) / DocAgent Foundation Model (domain-aware, cross-modal models)
DI-LLM (Document Intelligence in the Era of Large Language Models): introduces a comprehensive survey of Document AI advancements, categorizing tasks into understanding and generation, integrating multimodal and multilingual capabilities, and leveraging retrieval-augmented methods.
The survey explores key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions including agent-based approaches and document-specific foundation models.
It provides a structured analysis of the state-of-the-art in DAI, highlighting the evolution of LLM-based approaches and their implications for both academic and practical applications.

**List the architectural components found in the figures.**
The paper is a survey and does not present architectural figures for a single proposed framework. Instead, Table 1 summarizes benchmark datasets, detailing their supported languages, document counts, modalities (Text, Visual, Layout), and tasks (Key Information Extraction (KIE), Document Layout Analysis (DLA), Document Sentiment Analysis (DSA), Document Classification (DC), Document Summarization (DS), Document Content Generation (DCG), Question Answering (QA)). These modalities and tasks represent functional components or capabilities within Document AI systems.

**Define key components based on the figures and the text.**
Based on the text, the key conceptual components and approaches defining Document Intelligence in the Era of Large Language Models include:

*   **Multimodal Document AI**: An approach that integrates diverse information sources—Textual (OCR, digital text), Visual (figures, handwriting, region designs), and Layout (spatial arrangement, bounding boxes)—to comprehensively understand and generate documents.
*   **Multilingual Document AI**: Focuses on enabling LLMs to effectively handle documents in multiple languages, addressing cross-linguistic nuances and cultural intricacies through prompt-based methods and specialized training strategies.
*   **Retrieval-Augmented Paradigm**: A method that enhances LLMs by retrieving reliable external knowledge from documents (text, tables, images) to mitigate challenges related to outdated training data and limited domain expertise.
*   **DocAgent Framework**: A future direction proposing intelligent, multi-agent systems designed to proficiently manage complex document understanding and generation tasks by leveraging domain-specific knowledge and external tools.
*   **DocAgent Foundation Model**: A proposed future model that is domain-aware and cross-modal aligned, built upon continuously evolving datasets to provide an end-to-end solution for complex document processing.
*   **Document Layout Analysis (DLA)**: A task focused on detecting and classifying structural elements within documents, such as text blocks and headers, and understanding their spatial relationships.
*   **Key Information Extraction (KIE)**: A task aimed at identifying and extracting specific elements like form fields and key-value pairs from unstructured or semi-structured documents.
*   **Document Classification (DC)**: A task to identify the category, type, or domain of a document using textual, visual, and layout modalities.
*   **Document Question Answering (QA)**: A generation task focused on providing accurate natural language responses to questions based on document context, often involving complex table lookups or text extraction.
*   **Document Summarization (DS)**: A generation task that aims to create concise overviews of documents while preserving essential content.
*   **Document Content Generation (DCG)**: A generation task involving the creation of new document content, including structured layouts, textual continuations, figures, and tables, based on existing materials.
- 15th October 2025
Document Intelligence in the Era of Large Language Models: A Survey
2510.13366
http://arxiv.org/abs/2510.13366
DI-LLM (Document Intelligence in the Era of Large Language Models):
Multimodal Document AI / Multilingual Document AI / Retrieval-Augmented Paradigm / DocAgent Framework / DocAgent Foundation Model
Multimodal Document AI (integrating diverse modalities) / Multilingual Document AI (handling diverse languages) / Retrieval-Augmented Paradigm (leveraging external knowledge) / DocAgent Framework (agent-based document processing) / DocAgent Foundation Model (domain-aware, cross-modal models)
DI-LLM (Document Intelligence in the Era of Large Language Models): introduces a comprehensive survey of Document AI advancements, categorizing tasks into understanding and generation, integrating multimodal and multilingual capabilities, and leveraging retrieval-augmented methods.
The survey explores key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions including agent-based approaches and document-specific foundation models.
It provides a structured analysis of the state-of-the-art in DAI, highlighting the evolution of LLM-based approaches and their implications for both academic and practical applications.

**List the architectural components found in the figures.**
The paper is a survey and does not present architectural figures for a single proposed framework. Instead, Table 1 summarizes benchmark datasets, detailing their supported languages, document counts, modalities (Text, Visual, Layout), and tasks (Key Information Extraction (KIE), Document Layout Analysis (DLA), Document Sentiment Analysis (DSA), Document Classification (DC), Document Summarization (DS), Document Content Generation (DCG), Question Answering (QA)). These modalities and tasks represent functional components or capabilities within Document AI systems.

**Define key components based on the figures and the text.**
Based on the text, the key conceptual components and approaches defining Document Intelligence in the Era of Large Language Models include:

*   **Multimodal Document AI**: An approach that integrates diverse information sources—Textual (OCR, digital text), Visual (figures, handwriting, region designs), and Layout (spatial arrangement, bounding boxes)—to comprehensively understand and generate documents.
*   **Multilingual Document AI**: Focuses on enabling LLMs to effectively handle documents in multiple languages, addressing cross-linguistic nuances and cultural intricacies through prompt-based methods and specialized training strategies.
*   **Retrieval-Augmented Paradigm**: A method that enhances LLMs by retrieving reliable external knowledge from documents (text, tables, images) to mitigate challenges related to outdated training data and limited domain expertise.
*   **DocAgent Framework**: A future direction proposing intelligent, multi-agent systems designed to proficiently manage complex document understanding and generation tasks by leveraging domain-specific knowledge and external tools.
*   **DocAgent Foundation Model**: A proposed future model that is domain-aware and cross-modal aligned, built upon continuously evolving datasets to provide an end-to-end solution for complex document processing.
*   **Document Layout Analysis (DLA)**: A task focused on detecting and classifying structural elements within documents, such as text blocks and headers, and understanding their spatial relationships.
*   **Key Information Extraction (KIE)**: A task aimed at identifying and extracting specific elements like form fields and key-value pairs from unstructured or semi-structured documents.
*   **Document Classification (DC)**: A task to identify the category, type, or domain of a document using textual, visual, and layout modalities.
*   **Document Question Answering (QA)**: A generation task focused on providing accurate natural language responses to questions based on document context, often involving complex table lookups or text extraction.
*   **Document Summarization (DS)**: A generation task that aims to create concise overviews of documents while preserving essential content.
*   **Document Content Generation (DCG)**: A generation task involving the creation of new document content, including structured layouts, textual continuations, figures, and tables, based on existing materials.
- 15th October 2025
Document Intelligence in the Era of Large Language Models: A Survey
2510.13366
http://arxiv.org/abs/2510.13366
DI-LLM (Document Intelligence in the Era of Large Language Models):
Multimodal Document AI / Multilingual Document AI / Retrieval-Augmented Paradigm / DocAgent Framework / DocAgent Foundation Model
Multimodal Document AI (integrating diverse modalities) / Multilingual Document AI (handling diverse languages) / Retrieval-Augmented Paradigm (leveraging external knowledge) / DocAgent Framework (agent-based document processing) / DocAgent Foundation Model (domain-aware, cross-modal models)
DI-LLM (Document Intelligence in the Era of Large Language Models): introduces a comprehensive survey of Document AI advancements, categorizing tasks into understanding and generation, integrating multimodal and multilingual capabilities, and leveraging retrieval-augmented methods.
The survey explores key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions including agent-based approaches and document-specific foundation models.
It provides a structured analysis of the state-of-the-art in DAI, highlighting the evolution of LLM-based approaches and their implications for both academic and practical applications.

**List the architectural components found in the figures.**
The paper is a survey and does not present architectural figures for a single proposed framework. Instead, Table 1 summarizes benchmark datasets, detailing their supported languages, document counts, modalities (Text, Visual, Layout), and tasks (Key Information Extraction (KIE), Document Layout Analysis (DLA), Document Sentiment Analysis (DSA), Document Classification (DC), Document Summarization (DS), Document Content Generation (DCG), Question Answering (QA)). These modalities and tasks represent functional components or capabilities within Document AI systems.

**Define key components based on the figures and the text.**
Based on the text, the key conceptual components and approaches defining Document Intelligence in the Era of Large Language Models include:

*   **Multimodal Document AI**: An approach that integrates diverse information sources—Textual (OCR, digital text), Visual (figures, handwriting, region designs), and Layout (spatial arrangement, bounding boxes)—to comprehensively understand and generate documents.
*   **Multilingual Document AI**: Focuses on enabling LLMs to effectively handle documents in multiple languages, addressing cross-linguistic nuances and cultural intricacies through prompt-based methods and specialized training strategies.
*   **Retrieval-Augmented Paradigm**: A method that enhances LLMs by retrieving reliable external knowledge from documents (text, tables, images) to mitigate challenges related to outdated training data and limited domain expertise.
*   **DocAgent Framework**: A future direction proposing intelligent, multi-agent systems designed to proficiently manage complex document understanding and generation tasks by leveraging domain-specific knowledge and external tools.
*   **DocAgent Foundation Model**: A proposed future model that is domain-aware and cross-modal aligned, built upon continuously evolving datasets to provide an end-to-end solution for complex document processing.
*   **Document Layout Analysis (DLA)**: A task focused on detecting and classifying structural elements within documents, such as text blocks and headers, and understanding their spatial relationships.
*   **Key Information Extraction (KIE)**: A task aimed at identifying and extracting specific elements like form fields and key-value pairs from unstructured or semi-structured documents.
*   **Document Classification (DC)**: A task to identify the category, type, or domain of a document using textual, visual, and layout modalities.
*   **Document Question Answering (QA)**: A generation task focused on providing accurate natural language responses to questions based on document context, often involving complex table lookups or text extraction.
*   **Document Summarization (DS)**: A generation task that aims to create concise overviews of documents while preserving essential content.
*   **Document Content Generation (DCG)**: A generation task involving the creation of new document content, including structured layouts, textual continuations, figures, and tables, based on existing materials.

---

[D-SMART: Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree](http://arxiv.org/abs/2510.13363)

- D-SMART (Dynamic Structured Memory And Reasoning Tree): introduces a model-agnostic framework to enhance LLM dialogue consistency by coupling a Dynamic Structured Memory (OWL-compliant knowledge graph) and a Reasoning Tree (multi-step search over graph), which includes a Dialogue Knowledge Extractor (extracts knowledge fragments), Dynamic Updating (updates knowledge graph), Reasoning Engine (guides RT search), Current Memory (current DSM state), State Manage (manages reasoning states), Sample Action (proposes next actions), Perform Action (executes chosen action), and Output (generates final response).
- The framework enables LLMs to build and reason over a dynamic, structured representation of the conversational context, mitigating factual inconsistencies and logical decay in multi-turn dialogues.
- D-SMART significantly improves dialogue consistency and response quality by providing a traceable, multi-step reasoning process grounded in an evolving knowledge base.

---

[Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems](http://arxiv.org/abs/2510.13291)

- WOWService: introduces a comprehensive intelligent interaction framework tailored for industrial applications, integrating LLMs and multi-agent architectures, including a Training Pipeline, Data Construction Module, General Capability Enhancement Module, Business Scenario Adaptation Module, Multi-Agent Coordination, and Automated Evaluation, enabling autonomous task management and collaborative problem-solving.
- The framework employs a multi-stage training pipeline (CPT, SFT, DPO, RL) to strengthen LLMs' domain skills and evolves from a single-agent to a multi-agent architecture with specialized agents for targeted business demands.
- WOWService is deployed on the Meituan App, demonstrating significant gains in user satisfaction and personalized service through its robust evaluation framework and continuous optimization.

---

[Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation](http://arxiv.org/abs/2510.13272)

- VERITAS (Verifying Entailed Reasoning through Intermediate Traceability in Agentic Search): introduces a novel training framework that integrates fine-grained faithfulness rewards into the reinforcement learning process, enhancing LLM-based search agents' reasoning.
- This framework addresses chain-of-thought unfaithfulness in retrieval-augmented generation by formalizing and quantifying faithfulness through three metrics: Information-Think, Think-Search, and Think-Answer faithfulness.
- VERITAS improves reasoning faithfulness and maintains comparable task accuracy across seven QA benchmarks by employing a multi-faceted reward function and an efficient, distilled reward model for process supervision.

---

[GRIDAI: Generating and Repairing Intrusion Detection Rules via Collaboration among Multiple LLM-based Agents](http://arxiv.org/abs/2510.13257)

- GRIDAI (Generating and Repairing Intrusion Detection Rules via Collaboration among Multiple LLM-based Agents): introduces an end-to-end framework for automated intrusion detection rule generation and repair, featuring a Decision Logic (orchestrates agent actions), Relation-Assess Agent (assesses sample-rule relationship), New-Rule-Generate Agent (generates new detection rules), Existing-Rule-Repair Agent (repairs existing detection rules), Memory-Update Agent (updates rule memory repository), Rule Memory Repository (stores detection rules/attack samples), RuleItem (individual rule entry/signature/payload), Buffer (temporary rule storage), Attack Samples (incoming attack traffic), Web Attack Detection (NIDS validation engine), and Detection Rules (deployable rule output).
- The framework leverages multiple LLM-based agents to classify incoming attack samples, decide whether to generate new rules for novel attacks or repair existing ones for variants, and mitigate LLM hallucinations through real-time validation.
- GRIDAI enhances network intrusion detection systems by producing high-quality, adaptive rulesets that continuously evolve to address new and variant Web attacks, improving overall defense capabilities.

---

[Automated Network Protocol Testing with LLM Agents](http://arxiv.org/abs/2510.13248)

- NeTestLLM: introduces an LLM-powered multi-agent framework for end-to-end automated network protocol testing, integrating hierarchical protocol understanding, iterative test case generation and verification, executable artifact generation, and runtime feedback analysis, which interact with a testbed comprising a tester and a DUT.
- The framework leverages LLM agents for tasks like section splitting, summarization, module formation, test case generation, and artifact generation, supported by a knowledge base containing tasks, SOPs, and expert heuristics.
- NeTestLLM employs a hierarchical feedback loop with a small loop for artifact refinement and a large loop for test case refinement, ensuring continuous improvement and error isolation.

---

[ADAPTIVE REASONING EXECUTOR: A COLLABORATIVE AGENT SYSTEM FOR EFFICIENT REASONING](http://arxiv.org/abs/2510.13214)

- ARE (Adaptive Reasoning Executor): introduces a collaborative agent system that integrates small and large LLMs, including a Small LLM (Initial Answer Generator), a Large LLM (Judge, Verifier, Deep Reasoner), and a Judgment Mechanism (Evaluates Small LLM's response).
- The system leverages two evaluation strategies, Immediate Judgment (Directly assesses correctness) and Step-by-Step Judgment (Evaluates individual reasoning steps), to efficiently determine if the small LLM's initial answer is sufficient or if the large LLM needs to perform deeper reasoning.
- For complex problems, the framework can incorporate Verified Correct Steps (Augments prompt for deep reasoning) from the small LLM's attempt to assist the large LLM, reducing computational cost while maintaining accuracy.

---

[Emotional Cognitive Modeling Framework with Desire-Driven Objective Optimization for LLM-empowered Agent in Social Simulation](http://arxiv.org/abs/2510.13195)

- ECMF (Emotional Cognitive Modeling Framework): introduces an emotional cognition framework incorporating desire generation and objective management, designed to achieve emotion alignment between LLM-based agents and humans, modeling the complete decision-making process of LLM-based agents, encompassing state evolution, desire generation, objective optimization, decision generation, and action execution.
- The framework addresses limitations in affective cognition and bounded rationality of existing LLM-based agents by embedding emotions into their decision architectures, enabling dynamic responses to emotional state fluctuations.
- Experimental results demonstrate that ECMF-governed agents exhibit behaviors congruent with their emotional states, show superior ecological validity, and generate decision outcomes that closely approximate human behavioral patterns in social simulations.

---

[Addressing the alignment problem in transportation policy making: an LLM approach](http://arxiv.org/abs/2510.13139)

- Multi-Agent LLM Simulation Framework: introduces a multi-agent simulation where LLM agents, acting as representatives of city communities, participate in a referendum on transit policy proposals, using chain-of-thought reasoning and various voting mechanisms to model democratic consensus.
- The framework integrates a conventional utility-based travel demand model to provide performance metrics to the LLM agents, guiding their deliberation on policy levers such as sales tax, transit fare, and driver fees.
- This approach investigates whether LLMs can approximate plausible collective preferences and respond to local contexts, addressing the alignment problem between model-driven policies and public sentiment in transportation planning.

---

[PROVABLY INVINCIBLE ADVERSARIAL ATTACKS ON REINFORCEMENT LEARNING SYSTEMS: A RATE-DISTORTION INFORMATION-THEORETIC APPROACH](http://arxiv.org/abs/2510.13792)

- RDITAA (Rate-Distortion Information-Theoretic Adversarial Attack): introduces a provably "invincible" adversarial attack on Reinforcement Learning (RL) systems by using a Rate-Distortion Information-Theoretic Approach to manipulate the Ground-truth Transition Kernel (X) into a random Delusional Transition Kernel (Y), preventing the Victim Agent from gaining useful information about the true environment dynamics.
- The attack strategy involves the Attacker designing a joint probability distribution p(X, Y) to maximize the Regret of the Victim Agent while minimizing the Mutual Information I(X;Y) between the Ground-truth Transition Kernel (X) and the Delusional Transition Kernel (Y) under an Attack Budget (B).
- The paper provides a theoretical lower bound on the expected Regret and demonstrates the attack's impact on both model-based and model-free RL algorithms, including Q-learning and DQN, across environments like Block-world and Cartpole, showing significant reduction in the Victim Agent's expected reward.

---

[CoDS: Enhancing Collaborative Perception in Heterogeneous Scenarios via Domain Separation](http://arxiv.org/abs/2510.13432)

- CoDS (Collaborative perception method that leverages Domain Separation): introduces a fully convolutional collaborative perception adapter, with Lightweight Spatial-Channel Resizer, Distribution Alignment via Domain Separation, Encoder-Specific Domain Separation Module, Encoder-Agnostic Domain Separation Module, Domain Alignment Mutual Information Loss, Discriminator, Encoders, Feature Fusion Module, and Detection Head, to mitigate feature discrepancies in heterogeneous scenarios by separating domain-invariant from domain-specific information.
- The framework aligns neighbor features across spatial and channel dimensions using LSCR, then employs DADS with encoder-specific and encoder-agnostic modules to remove domain-dependent information and capture task-related information.
- During training, the DAMI loss maximizes mutual information between aligned heterogeneous features to enhance domain separation, ensuring aligned features preserve only task-related information for robust and efficient collaborative perception.

---

[SAJA: A State-Action Joint Attack Framework on Multi-Agent Deep Reinforcement Learning](http://arxiv.org/abs/2510.13262)

- SAJA (State-Action Joint Attack): introduces a novel, efficient, two-phase, gradient-based framework for adversarial attacks on Multi-Agent Deep Reinforcement Learning (MADRL) systems, with all its State Attack Phase (computes adversarial state), Action Attack Phase (crafts adversarial action), Heuristic Regularizer (measures action distance), Heuristic Loss Function (HLF) (combines Q-value and action distance), and Victim Selection (selects subset of agents) components, designed to exploit synergistic vulnerabilities by perturbing both states and actions.
- The framework employs a Heuristic Loss Function (HLF) that combines a Q-value term with an action distance term to guide gradient ascent, enhancing attack effectiveness and reducing reliance on potentially inaccurate Q-value estimations.
- Experiments in the Multi-Agent Particle Environment (MPE) demonstrate SAJA's superior performance and stealthiness compared to state-only or action-only attacks, effectively bypassing existing defense mechanisms.

---

[Agentic Discovery: Closing the Loop with Cooperative Agents](http://arxiv.org/abs/2510.13081)

- Agentic Scientific Method: introduces a framework where specialized cooperative agents, including Objective, Knowledge, Prediction, Service, Analysis, and Publish agents, autonomously execute and manage the iterative scientific discovery process.
- This framework is augmented by transcending agents like Planning, Enforcement, and Exploration, which manage resources, ensure safety, and guide discovery, leveraging LLMs and various computational and experimental infrastructures.
- The paper posits that this agent-driven approach can significantly accelerate scientific discovery by automating human-intensive tasks, thereby closing the loop on autonomous research.

---

[CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization](http://arxiv.org/abs/2510.14150)

- CODEEVOLVE: introduces an open-source evolutionary coding agent that unites LLMs with genetic algorithms to solve complex computational problems, leveraging an island-based genetic algorithm, an LLM ensemble, and specialized evolutionary operators for algorithm discovery and optimization.
- The framework integrates a weighted LLM ensemble, including FLASH and PRO models, with modular mechanisms like depth exploitation, meta-prompting exploration, and inspiration-based crossover to iteratively evolve solutions.
- CODEEVOLVE's population management module orchestrates the evolutionary cycle through initialization, evaluation, population control, and elitist migration, ensuring diversity and propagation of high-performing solutions.

---

[Formalizing the Safety, Security, and Functional Properties of Agentic AI Systems](http://arxiv.org/abs/2510.14133)

- Agentic AI System Modeling Framework: introduces a unified semantic framework for agentic AI systems, with Host Agent Model (HA) and Task Lifecycle Model (L) components, to enable rigorous analysis of safety, security, and functional properties.
- The HA model formalizes the top-level entity that interacts with users, decomposes tasks, and orchestrates execution by leveraging external agents and tools, while the L model details sub-task states and transitions from creation to completion or failure.
- This framework defines 31 formal properties, categorized into liveness, safety, completeness, and fairness, expressed in temporal logic to enable formal verification of system behavior and detection of coordination issues.

---

[Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming](http://arxiv.org/abs/2510.14063)

- OATH (Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming): introduces a hierarchical framework for multi-robot task assignment and planning in obstacle-rich environments, incorporating adaptive Halton map construction, precompute Dijkstra distance matrices, obstacles-aware clustering, cluster-weighted auction, intra-cluster task selection, construct LTL specifications, D Lite path planning, an LLM, human instructions, a plan update mechanism, a robot, and an iterative assignment cycle.
- The framework dynamically adjusts sampling density based on obstacle distribution, enabling efficient coordination among heterogeneous robots in complex, obstacle-rich environments.
- An LLM-guided interaction module allows real-time interpretation of natural language commands, supporting dynamic replanning and adaptation to unforeseen changes during task execution.

---

[Stop Reducing Responsibility in LLM-Powered Multi-Agent Systems to Local Alignment](http://arxiv.org/abs/2510.14008)

- Responsible LLM-MAS Framework (LLM-Powered Multi-Agent Systems): introduces a dual-perspective governance framework for LLM-powered Multi-Agent Systems, integrating human-AI collaborative oversight with components like Human Moderator, AI Moderator, Decision Making, Guidance, Supervision, Heterogeneous Agents, Tasks, and Runtime Oversight Feedback.
- This framework aims to ensure lifecycle-wide responsibility by achieving global, systemic agreement, managing uncertainty, and enhancing security across dynamic multi-agent interactions.
- The framework shifts the focus from local agent-level alignment to a comprehensive system-wide approach, supported by quantifiable, verifiable, and traceable metrics for dynamic evaluation and safe control.

---

[Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations](http://arxiv.org/abs/2510.13982)

- Three Pillars of Open-Ended Multi-Agent Simulation: introduces a taxonomy for LLM-based multi-agent simulations, advocating for a shift from static, task-specific benchmarks to open-ended co-evolutionary dynamics, including Dynamic Scenario Evolution, Agent-Environment Co-evolution, and Generative Agent Architectures.
- The paper argues that current multi-agent simulation paradigms are inadequate for modeling real-world societal complexity, proposing a framework that embraces unpredictability and continuous adaptation.
- This framework aims to foster adaptive, socially aligned LLM-driven ecosystems where agents not only perform tasks but also evolve, adapt, learn, and transform their environments and social structures.

---

[FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis](http://arxiv.org/abs/2510.13936)

- HisRubric: introduces an evaluation framework for Deep Research (DR) agents in financial analysis, comprising a Research Task Instruction, a Deep Research Agent with Planning, Retrieval, Analysis, and Generation modules, Analytical Results, and an Evaluation component featuring a Rigorous Hierarchical Structure and a Fine-grained Grading Rubric with Recognition, Calculation, Abstraction, and Interpretation capabilities.
- The framework systematically assesses DR agents' ability to produce high-quality financial reports by guiding them with a predefined analytical structure and scoring their output based on detailed, expert-designed criteria.
- Built upon HisRubric, the FINDEEPRESEARCH benchmark provides a comprehensive dataset for evaluating DR agents across diverse financial markets and languages, revealing their strengths and limitations in rigorous financial analysis.

---

[An LLM-Powered AI Agent Framework for Holistic IoT Traffic Interpretation](http://arxiv.org/abs/2510.13925)

- Revelation: introduces an LLM-powered AI agent framework for holistic IoT traffic interpretation, converting raw packet captures into structured, semantically enriched representations for interactive analysis and evidence-grounded question answering.
- The framework integrates feature extraction, transformer-based anomaly detection, packet/flow summarization, threat intelligence enrichment, and retrieval-augmented question answering.
- An AI agent, guided by an LLM, performs reasoning over indexed traffic artifacts, assembling evidence to produce accurate, human-readable interpretations and supporting operational workflows.

---

[FACTS: TABLE SUMMARIZATION VIA OFFLINE TEMPLATE GENERATION WITH AGENTIC WORKFLOWS](http://arxiv.org/abs/2510.13920)

- FACTS (Fast, Accurate, and Privacy-Compliant Table Summarization approach via Offline Template Generation): introduces an agentic workflow for query-focused table summarization that generates reusable offline templates, consisting of SQL queries and Jinja2 templates, through a multi-stage process involving an LLM Agent, LLM Council, and local SQL execution.
- The framework ensures fast, accurate, and privacy-compliant summarization by producing schema-aware templates that are reusable across tables with the same schema, avoiding repeated LLM calls with raw data.
- FACTS integrates an LLM Council for iterative validation and refinement of outputs at each stage, ensuring correctness, consistency, and usability of the generated artifacts.

---

[RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems](http://arxiv.org/abs/2510.13910)

- RAGCap-Bench: introduces a capability-oriented benchmark for fine-grained evaluation of intermediate tasks in agentic RAG workflows, including Planning (interpreting problem/refining plan), Evidence Extraction (identifying useful evidence), Grounded Reasoning (reasoning with evidence), and Noise Robustness (detecting low-quality info/abstaining).
- The benchmark frames evaluation questions as Multiple-Choice Questions (MCQs) derived from a taxonomy of typical LLM errors, using both Vanilla Generation and Error-Guided Generation strategies.
- Experiments demonstrate that RAGCap-Bench performance reliably correlates with end-to-end performance in complex agentic RAG workflows, highlighting the importance of enhancing these intermediate capabilities.

---

[Active Inference for an Intelligent Agent in Autonomous Reconnaissance Missions](http://arxiv.org/abs/2510.17450)

- Active Inference for Autonomous Reconnaissance: introduces an active inference route-planning method for intelligent agents, utilizing a generative model and process to update an evidence map based on sensor observations, and calculating free energy to direct agent movements for persistent surveillance.
- The framework employs Dempster-Shafer theory for uncertainty representation and a Gaussian sensor model for observation likelihood, enabling agents to balance exploration and exploitation by minimizing surprise and divergence between perceived reality and internal models.
- Agent control is achieved by iteratively moving towards grid cells that minimize the calculated free energy, ensuring continuous monitoring of a designated area and tracking of identified target objects.

---

[When "Correct” Is Not Safe: Can We Trust Functionally Correct Patches Generated by Code Agents?](http://arxiv.org/abs/2510.17862)

- FCV-Attack: introduces "Functionally Correct yet Vulnerable (FCV) patches", which pass all functional tests but contain exploitable vulnerabilities, by appending CWE-targeted, developer-style suggestions to GitHub issue descriptions.
- The attack operates under a black-box, single-query threat model, demonstrating that even SOTA LLMs and agent scaffolds are vulnerable, with Attack Success Rates up to 56.3% for information exposure (CWE-538).
- Controlled experiments reveal that vulnerabilities propagate through the agent's internal model state (KV cache contamination) during initial encoding, rather than through observable agent actions, rendering behavior-level defenses insufficient.

---

#### 14th October 2025

[Ax-Prover: A Deep Reasoning Agentic Framework for Theorem Proving in Mathematics and Quantum Physics](http://arxiv.org/abs/2510.12787)

- Ax-Prover (Axiomatic Prover): introduces a multi-agent system for automated theorem proving in Lean, leveraging LLMs for reasoning and MCP for formal correctness, including Orchestrator, Prover, and Verifier agents, along with various Lean and Filesystem tools.
- The system addresses limitations of specialized provers by enabling domain generalization, tool-use, human-AI collaboration, and reducing deployment costs, outperforming baselines on new abstract algebra and quantum physics benchmarks.
- Ax-Prover operates through an iterative closed-loop process where the Orchestrator assigns tasks to the Prover, which generates Lean code using MCP tools, and the Verifier checks correctness, providing feedback for refinement.

---

[OMNI-CAPTIONER: DATA PIPELINE, MODELS, AND BENCHMARK FOR OMNI DETAILED PERCEPTION](http://arxiv.org/abs/2510.12720)

- OMNI-CAPTIONER: introduces a comprehensive framework for omni detailed perception, including the Omni-Detective data generation pipeline, Audio-Captioner and Omni-Captioner models, and the Omni-Cloze evaluation benchmark.
- The Omni-Detective pipeline autonomously generates highly detailed, minimally hallucinatory multimodal data by leveraging an agentic LLM with tool-calling capabilities and iterative evidence gathering.
- The trained Omni-Captioner models achieve state-of-the-art performance on existing benchmarks and the novel Omni-Cloze, which provides a stable and efficient cloze-style evaluation across audio, visual, and audio-visual modalities.

---

[Reflection-Based Task Adaptation for Self-Improving VLA](http://arxiv.org/abs/2510.12710)

- Reflective Self-Adaptation: introduces a novel framework for autonomous, in-situ VLA adaptation, featuring a dual-pathway architecture that includes a Failure-Driven Reflective RL Pathway for failure analysis and a Success-Driven Quality-Guided SFT Pathway for high-quality success imitation.
- The Failure-Driven Reflective RL Pathway leverages a VLM's causal reasoning to synthesize dense rewards from failures, accelerating RL exploration, while the Success-Driven Quality-Guided SFT Pathway ensures learning stability and prevents reward hacking by imitating successful trajectories.
- The framework integrates a VLM as an in-the-loop causal reasoner and reward synthesizer, dynamically analyzing execution failures and synthesizing corrective reward functions, complemented by a conditional curriculum mechanism for cold-start exploration.

---

[MEMORY AS ACTION: AUTONOMOUS CONTEXT CURATION FOR LONG-HORIZON AGENTIC TASKS](http://arxiv.org/abs/2510.12635)

- MemAct (Memory-as-Action): introduces a framework where an LLM agent actively manages its working memory through explicit editing operations, integrating context curation into its unified policy.
- This framework utilizes a novel RL algorithm, DCPO, to enable stable end-to-end learning by segmenting trajectories at memory action points, addressing challenges of non-prefix context changes.
- MemAct improves task performance and reduces computational consumption by optimizing both task reasoning and adaptive memory management strategies.

---

[Designing Tools with Control Confidence](http://arxiv.org/abs/2510.12630)

- Tool Design Pipeline: introduces an autonomous framework for designing robotic hand tools, comprising a tool optimizer (optimizes parameters), a tool generator (creates mesh), a planner (executes motion), a controller (generates torques), a performance evaluator (measures task success), a confidence evaluator (measures control precision), a tool mesh (parametric representation), a task variable (object position), and a free energy objective (balances robustness and accuracy), which optimizes tool designs by minimizing free energy.
- The framework integrates a neuro-inspired control confidence term into the optimization routine to enhance tool robustness against environmental uncertainties.
- Utilizing a CMAES-based evolutionary optimization strategy, the pipeline effectively balances tool robustness and goal accuracy for various task conditions.

---

[COIRL-AD: COLLABORATIVE-COMPETITIVE IMITATION-REINFORCEMENT LEARNING IN LATENT WORLD MODELS FOR AUTONOMOUS DRIVING](http://arxiv.org/abs/2510.12560)

- CoIRL-AD: introduces a competitive dual-policy framework for end-to-end autonomous driving, integrating imitation learning (IL) and reinforcement learning (RL) through a shared latent world model, with all its Perception module, Latent World Model, IL Actor, RL Actor, Critic, Reward Function, and Competitive Learning Mechanism components, where it enables IL and RL agents to interact during training via a competition-based mechanism for knowledge exchange.
- The framework leverages a latent world model for imagination-based simulation, allowing the RL actor to explore and learn from trial-and-error without relying on external simulators.
- A dual-policy architecture decouples IL and RL objectives into separate actors, which are jointly trained in parallel, with a competitive learning mechanism facilitating knowledge transfer and preventing gradient conflicts.

---

[Biased-Attention Guided Risk Prediction for Safe Decision-Making at Unsignalized Intersections](http://arxiv.org/abs/2510.12428)

- SAC-RWB (Soft Actor-Critic with Risk Prediction and Biased Attention): introduces a DRL framework for safe decision-making at unsignalized intersections, integrating a Transformer-based risk predictor with a biased attention mechanism, an RL agent with Actor and Critic networks, a reward function, and a hierarchical experience replay mechanism, to proactively avoid collisions and improve traffic efficiency.
- The framework leverages the Transformer's sequential modeling to predict long-term collision risks, converting them into a dense reward signal that guides the SAC agent's policy optimization.
- A hierarchical experience replay mechanism, comprising high-risk and standard buffers, accelerates convergence by providing balanced training data from both collision and safe driving scenarios.

---

[A Survey of Vibe Coding with Large Language Models](http://arxiv.org/abs/2510.12399)

- Vibe Coding: introduces a novel software development methodology, formalizing a dynamic triadic relationship among human developers, software projects, and coding agents, with all its Large Language Models for Coding (foundational models), LLM-based Coding Agent (autonomous programming entity), Development Environment of Coding Agent (execution infrastructure, interfaces), and Feedback Mechanisms (guides agent improvement) components, where developers validate AI-generated implementations through outcome observation rather than line-by-line code comprehension.
- This framework systematically reviews the entire vibe coding ecosystem, examining critical infrastructure components including LLMs for coding, LLM-based coding agents, development environments, and feedback mechanisms.
- The survey synthesizes existing practices into five distinct development models, providing a comprehensive taxonomy and identifying key challenges for AI-augmented software engineering.

---

[ResearStudio: A Human-Intervenable Framework for Building Controllable Deep-Research Agents](http://arxiv.org/abs/2510.12194)

- ResearStudio: introduces a human-intervenable framework for building controllable deep-research agents, featuring a User (human collaborator), a Planner-Executor Agent Core (AI decision-making engine) with Planner Agent (task planning LLM) and Executor Agent (task execution LLM), an MCP Toolbox (L-1) (tool collection), an Interactive Web Interface (L-3) (user interaction platform), a Workspace (project file storage), and a Communication Protocol (inter-component data flow).
- This framework enables real-time bidirectional collaboration, allowing users to pause, edit plans or code, run custom commands, and seamlessly switch between AI-led and human-led workflows.
- The framework achieves state-of-the-art performance on benchmarks while providing transparency and symmetrical control, transforming autonomous agents into reliable research partners.

---

[ToPolyAgent: AI Agents for Coarse-Grained Topological Polymer Simulations](http://arxiv.org/abs/2510.12091)

- ToPolyAgent (AI Agents for Coarse-Grained Topological Polymer Simulations): introduces a multi-agent AI framework for performing coarse-grained molecular dynamics (MD) simulations of topological polymers through natural language instructions, including Config Agent (generates initial configurations), Simulation Agent (executes MD simulations, analyzes data), Report Agent (compiles markdown reports), Workflow Agent (orchestrates autonomous operations), CrewAI (orchestrates multi-agent system, manages memory), and LLM (powers agents, interprets natural language).
- The framework operates in interactive mode with user feedback loops for iterative refinements and an autonomous mode for end-to-end task execution from detailed prompts.
- ToPolyAgent integrates LLMs with domain-specific computational tools to lower barriers to complex computational workflows and advance AI-driven materials discovery in polymer science.

---

[ONE LIFE TO LEARN: INFERRING SYMBOLIC WORLD MODELS FOR STOCHASTIC ENVIRONMENTS FROM UNGUIDED EXPLORATION](http://arxiv.org/abs/2510.12088)

- ONELIFE: introduces a framework for inferring symbolic world models in stochastic environments from unguided exploration, utilizing a world model as a program, a law synthesizer, an inference algorithm, a forward simulation process, an exploration policy, and an observable extractor to learn environment dynamics from minimal interaction.
- The framework models world dynamics through conditionally-activated programmatic laws within a probabilistic programming framework, enabling accurate learning of stochastic dynamics even when most rules are inactive.
- ONELIFE successfully learns key environment dynamics from minimal, unguided interaction and demonstrates the world model's utility for planning by identifying superior strategies in goal-oriented tasks.

---

[Autonomous vehicles need social awareness to find optima in multi-agent reinforcement learning routing games.](http://arxiv.org/abs/2510.11410)

- RouteRL: introduces a novel reward formulation for Autonomous Vehicles (AVs) within a Multi-Agent Reinforcement Learning (MARL) framework, integrating a social component based on marginal cost calculation to accelerate convergence to optimal routing solutions.
- This approach addresses the issue of selfish AVs destabilizing traffic systems by enabling them to consider their impact on other agents, leading to improved system-wide and individual travel times.
- The framework utilizes SUMO for traffic simulation and demonstrates its effectiveness across various MARL algorithms in both toy and real-world traffic networks.

---

[L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2510.07363)

- L2M-AID (Autonomous Industrial Defense): introduces a novel framework for autonomous cyber-physical defense, orchestrating a team of collaborative agents, each driven by an LLM, to achieve adaptive and resilient security.
- The framework deeply fuses LLM-driven semantic reasoning with Multi-Agent Reinforcement Learning, enabling agents to reason about adversary intent and learn complex cooperative strategies.
- L2M-AID significantly outperforms traditional Intrusion Detection Systems and deep learning anomaly detectors, demonstrating superior performance in detection rate, false positive reduction, and physical process stability.

---

[Deliberate Lab: A Platform for Real-Time Human-AI Social Experiments](http://arxiv.org/abs/2510.13011)

- Deliberate Lab: introduces a no-code, open-source platform for real-time human-AI social experiments, featuring a Frontend, Backend (Google Firebase Platform), Cloud Functions, Firestore Database, Realtime Database, Experiment Builder, Experiment Stages, Cohort Management System, Facilitator Dashboard, Participant Interface, LLM Agents, Prompt Editor, LLM API Integrations, LLM Debugging Panel, and Data Export Module, enabling researchers to design, facilitate, and participate in synchronous, multi-party studies with human and LLM participants.
- The platform leverages Google Firebase for its backend, utilizing Cloud Functions for server-side logic, Firestore Database for primary data storage, and Realtime Database for tracking real-time participant presence.
- The platform provides a modular design with configurable experiment stages and comprehensive LLM integration, allowing for flexible experimental setups, real-time monitoring, and structured data export to support diverse behavioral research.

---

[SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents](http://arxiv.org/abs/2510.12985)

- SENTINEL (A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents): introduces a multi-level formal framework for evaluating the physical safety of LLM-based embodied agents, including natural language input, an LLM agent, semantic-level safety evaluation, plan-level safety evaluation, trajectory-level safety evaluation, and simulators.
- The framework grounds practical safety requirements in formal temporal logic (LTL and CTL) semantics, enabling precise specification and systematic verification across semantic, plan, and trajectory levels.
- This approach identifies safety violations overlooked by previous methods, providing insights into failure modes and supporting rigorous evaluation of LLM-based embodied agents in physical environments.

---

[DEEPPLANNER: Scaling Planning Capability for Deep Research Agents via Advantage Shaping](http://arxiv.org/abs/2510.12979)

- DEEPPLANNER: introduces an end-to-end RL framework that enhances planning capabilities of deep research agents by using an LLM, an Agent Loop with Think, Plan, Tool Call, and Answer modules, Web Search and Web Browse tools, GRPO, Entropy-based Advantage Shaping (EAS), and Selective Advantage Upweighting (SAU).
- The framework addresses high planning token entropy by amplifying learning signals on uncertain planning tokens and prioritizing complex, high-quality rollouts, leading to improved planning quality.
- This approach achieves state-of-the-art results on deep research benchmarks with significantly reduced training budgets, demonstrating efficient scaling of planning capabilities.

---

[EDUDIAL: CONSTRUCTING A LARGE-SCALE MULTI-TURN TEACHER-STUDENT DIALOGUE CORPUS](http://arxiv.org/abs/2510.12899)

- EduDial: introduces a comprehensive multi-turn teacher-student dialogue dataset and an LLM trained on it, designed to simulate authentic classroom interactions through a five-stage teaching process, differentiated strategies, and a two-stage training paradigm.
- The framework leverages LLM-based teacher and student agents with defined role profiles and questioning strategies to generate high-quality instructional data, which is then used for supervised fine-tuning and direct preference optimization.
- EduDial-LLM, trained on this dataset, demonstrates superior performance in student-centered teaching scenarios, adapting its guidance based on student cognitive levels and providing personalized feedback, evaluated by an 11-dimensional framework.

---

[KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems](http://arxiv.org/abs/2510.12872)

- KVCOMM (Online Cross-context KV-cache Communication): introduces a training-free framework that enables efficient prefilling in multi-agent LLM systems by reusing KV-caches and aligning cache offsets of overlapping contexts under diverse prefix contexts, utilizing an anchor pool, anchor matching, offset approximation, and online anchor updates.
- The framework addresses the multi-context redundancy issue by dynamically determining how to reuse KV-caches at runtime for incoming prompts with diverse prefix contexts, achieving significant speedup without additional training or model modifications.
- KVCOMM achieves over 70% reuse rate and up to 7.8x speedup across various multi-agent workloads, including retrieval-augmented generation, math reasoning, and collaborative coding, while maintaining task accuracy.

---

[DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search](http://arxiv.org/abs/2510.12801)

- DeepMMSearch-R1: introduces a multimodal LLM capable of on-demand, multi-turn web searches by dynamically crafting queries for image and text search tools, incorporating self-reflection and self-correction.
- The framework utilizes a two-stage training pipeline, including supervised finetuning with the DeepMMSearchVQA dataset and online reinforcement learning with GRPO, to refine tool-use and search efficiency.
- DeepMMSearch-R1 enhances image search effectiveness through an intermediate cropping tool (Grounding DINO) that selects relevant image regions, outperforming baselines in knowledge-intensive benchmarks.

---

[FROM LITERAL TO LIBERAL: A META-PROMPTING FRAMEWORK FOR ELICITING HUMAN-ALIGNED EXCEPTION HANDLING IN LARGE LANGUAGE MODELS](http://arxiv.org/abs/2510.12864)

- RID (Rule-Intent Distinction Framework): introduces a novel, low-compute meta-prompting technique designed to elicit human-aligned exception handling in LLMs in a zero-shot manner, including a Role, Core Directive, Reasoning Schema (Deconstruct the Task, Classify the Rule, Analyze the Conflict & Weigh Outcomes, Formulate a Decision & Justification), and a structured Output Format.
- This framework guides LLMs from literal instruction-following to pragmatic, goal-oriented reasoning by forcing explicit deconstruction of user goals, rule classification, outcome weighing, and decision justification.
- The approach significantly improves decision quality and reasoning transparency, achieving a 95% Human Alignment Score and consistently producing higher-quality, intent-driven reasoning.

---

[Multi-Agent Debate for LLM Judges with Adaptive Stability Detection](http://arxiv.org/abs/2510.12697)

- Multi-Agent Debate Framework: introduces a novel multi-agent debate framework where LLMs collaboratively reason and iteratively refine judgments, utilizing LLM judges, debate history, round generation, judgment extraction, convergence check, and an adaptive stability detection mechanism to produce a consensus or majority vote.
- The framework formalizes the debate process mathematically and incorporates an adaptive stability detection mechanism, which uses a time-varying Beta-Binomial mixture model and Kolmogorov-Smirnov testing to efficiently halt the debate once judge accuracy rates stabilize.
- This approach enhances the robustness and precision of LLM-based evaluations by aggregating diverse perspectives and mitigating biases, outperforming static aggregation methods like majority voting while maintaining computational efficiency.

---

[Diff-XYZ: A Benchmark for Evaluating Diff Understanding](http://arxiv.org/abs/2510.12487)

- Diff-XYZ: introduces a compact benchmark for code-diff understanding, featuring a Benchmark (core evaluation system) with three synthetic tasks: Apply Task (new code generation), Anti-Apply Task (old code reconstruction), and Diff Generation Task (diff synthesis), utilizing various Diff Formats (udiff, udiff-h, udiff-l, search-replace) to evaluate LLMs (models under evaluation) sourced from the CommitPackFT Dataset (source of real-world code edits) using specific Evaluation Metrics (EM, IoU, F1+, F1-, Parsing Rate, Applying Rate) and controlled System Prompts (instruction sets for LLMs) and Task Prompts (specific input templates for tasks).
- The benchmark isolates the effect of diff representation on LLM performance by fixing other contextual factors, providing a lightweight and reproducible setting for studying diff-centric workflows.
- Findings reveal that optimal diff formats vary by task and model size, with udiff-based formats excelling for application tasks and search-replace for diff generation, especially in larger LLMs.

---

[MTOS: A LLM-DRIVEN MULTI-TOPIC OPINION SIMULATION FRAMEWORK FOR EXPLORING ECHO CHAMBER DYNAMICS](http://arxiv.org/abs/2510.12423)

- MTOS (Multi-topic Opinion Simulation): introduces a social simulation framework integrating multi-topic contexts with LLMs, leveraging LLMs alongside short-term and long-term memory, multiple user-selection interaction mechanisms, dynamic topic-selection strategies, and a belief decay mechanism to enable perspective updates across topics.
- The framework initializes agents with unique roles and multi-topic opinion vectors within a scale-free social network, allowing them to select neighbors for opinion exchange based on belief similarity or semantic matching.
- MTOS dynamically recommends topics considering group popularity and individual fatigue, and updates agent beliefs through a dual-layer memory architecture and a decay mechanism, simulating realistic multi-topic opinion evolution and mitigating echo chamber effects.

---

[VideoLucy: Deep Memory Backtracking for Long Video Understanding](http://arxiv.org/abs/2510.12422)

- VideoLucy: introduces a deep memory backtracking framework for long video understanding, which employs a hierarchical memory structure (progressive granularity memory) and an agent-based iterative backtracking mechanism (dynamic memory exploration loop) to systematically mine question-relevant deep memories.
- The framework leverages MLLMs (multimodal large language model) for vision captioning and LLMs (large language model) for reasoning, with specialized agents including a Captioning Agent, Localization Agent, Instruction Agent, and Answering Agent.
- VideoLucy's hierarchical memory structure includes Coarse Memory, Fine Memory, and Ultra-fine Memory, enabling multi-level video representation and comprehensive information coverage.

---

[LLM-REVAL: CAN WE TRUST LLM REVIEWERS YET?](http://arxiv.org/abs/2510.12367)

- LLM-REVal (LLM REViewer Re-EValuation): introduces a multi-round simulation of the academic publication process, with Research-Review Round (initial submission and review cycle), Revise-Review Round (iterative revision and review cycle), Research Agent (generates and revises papers), Review Agent (assesses submissions and manages peer review), and LLM Backbones (underlying large language models), to examine the fairness risks of using LLMs as reviewers.
- The simulation reveals LLM reviewers systematically inflate scores for LLM-authored papers and undervalue human-authored papers, indicating biases rooted in linguistic features and an aversion to critical statements.
- Despite these biases, revisions guided by LLM reviews lead to quality gains, suggesting potential for LLMs to support early-stage researchers and improve low-quality papers.

---

[T3: REDUCING BELIEF DEVIATION IN REINFORCEMENT LEARNING FOR ACTIVE REASONING](http://arxiv.org/abs/2510.12264)

- T³ (Truncating Belief-Trapped Trajectories): introduces a method that detects excessive belief deviation and truncates trajectories during training to remove uninformative tails, preserving credit for informative prefixes.
- This approach systematically improves policy optimization by concentrating learning signals on genuinely informative actions, leading to enhanced training stability, token efficiency, and final performance.
- T³ integrates seamlessly into standard policy optimization frameworks like PPO, GRPO, and GSPO, offering a practical solution to the credit assignment problem in active reasoning.

---

[MedKGEval: A Knowledge Graph-Based Multi-Turn Evaluation Framework for Open-Ended Patient Interactions with Clinical LLMs](http://arxiv.org/abs/2510.12224)

- MedKGEval (A Knowledge Graph-Based Multi-Turn Evaluation Framework): introduces a framework for evaluating clinical LLMs in open-ended patient interactions, utilizing a MedKG, KG Tool, Patient Profile, Sub-Graph Extraction, Task Setting, Director Agent, Patient Agent, Doctor Agent, Judge Agent, Conversation History, and Evaluation Result.
- The framework simulates realistic doctor-patient dialogues, where a Director Agent guides a Patient Agent (LLM) to interact with a Doctor Agent (LLM under evaluation), with a Judge Agent (LLM) providing real-time, turn-by-turn assessment.
- This multi-agent system, grounded in structured medical knowledge, enables fine-grained evaluation of LLM performance in complex, multi-turn clinical scenarios, identifying subtle behavioral flaws and safety risks.

---

[GOAT: A TRAINING FRAMEWORK FOR GOAL-ORIENTED AGENT WITH TOOLS](http://arxiv.org/abs/2510.12218)

- GOAT (Goal-Oriented Agent with Tools): introduces a novel training framework that automatically constructs synthetic datasets of goal-oriented API execution tasks from API documents, enabling fine-tuning of LLM agents for complex reasoning and tool use.
- The framework generates training data by building an API dependency graph through a multi-stage filtering process, sampling connected API sequences, and then generating API calls, sub-queries, user queries, and final responses.
- GOAT also introduces GOATBench, a new human-verified benchmark for evaluating goal-oriented API execution, demonstrating state-of-the-art performance for GOAT-trained open-source LLM agents.

---

[Agent-Based Simulation of a Financial Market with Large Language Models](http://arxiv.org/abs/2510.12189)

- FCLAgent: introduces an agent-based financial market simulation model that integrates context-dependent, human-like behavioral biases elicited from an LLM (Large Language Model) for buy/sell decisions, while relying on a rule-based mechanism for order price and volume determination.
- This hybrid architecture enables the agent to exhibit psychologically plausible behavior derived from LLM outputs, circumventing LLMs' limitations in numerical reasoning for financial market simulations.
- The framework successfully reproduces empirically observed market anomalies, such as the negative correlation between proximity to an asset's all-time high and future returns, which traditional agents alone could not replicate.

---

[Towards Engineering Multi-Agent LLMs: A Protocol-Driven Approach](http://arxiv.org/abs/2510.12120)

- SEMAP (Software Engineering Multi-Agent Protocol): introduces a protocol-layer methodology for multi-agent LLMs, instantiating explicit behavioral contract modeling, structured messaging, and lifecycle-guided execution with verification.
- This framework addresses under-specification, coordination misalignment, and inappropriate verification in multi-agent LLM systems by applying foundational software engineering principles.
- Empirical evaluations demonstrate that SEMAP substantially reduces failure rates across diverse software engineering tasks, improving system robustness and promoting stable collaboration.

---

[IL3D: A LARGE-SCALE INDOOR LAYOUT DATASET FOR LLM-DRIVEN 3D SCENE GENERATION](http://arxiv.org/abs/2510.12095)

- IL3D (A Large-Scale Indoor Layout Dataset): introduces a large-scale dataset for LLM-driven 3D scene generation, featuring 27,816 indoor layouts, 29,215 3D object assets, instance-level natural language annotations, multimodal data export capabilities, USDZ-format assets, USDA-format scenes, an LLM for object description extraction, a VLM for text-to-vector conversion, a 3D Asset Vector Database for storage, and a Query Module for asset retrieval.
- The dataset provides high-fidelity scene data with fine-grained annotations, supporting robust multimodal learning for vision-language tasks and advancing research in 3D scene generation and embodied intelligence.
- Experiments demonstrate that supervised fine-tuning of LLMs on IL3D significantly improves generalization and performance in LLM-driven layout generation, offering flexible data export for various visual tasks.

---

[Evaluating the Quality of Randomness and Entropy in Tasks Supported by Large Language Models](http://arxiv.org/abs/2510.12080)

- LLM Randomness Evaluation Framework: introduces a comprehensive experimental setup to evaluate LLMs' capabilities in handling randomness, with Prompts, LLM, LLM Inference, External Tools, Random Output Generation, Evaluation Metrics, Model States, and Task Types, aiming to assess the quality of LLM-generated random outputs across various scenarios.
- The framework systematically investigates factors influencing LLM performance in randomness tasks, including the use of external pseudo-random number generators (PRNGs), different task categories (numerical, character-based, shuffling), LLM states, and prompting strategies.
- The study employs the NIST randomness test-suite and entropy-based metrics to compare LLM-generated randomness against established methods, revealing that LLMs struggle to achieve high-quality randomness, especially without external tools.

---

[EMBOMATRIX: A SCALABLE TRAINING-GROUND FOR EMBODIED DECISION-MAKING](http://arxiv.org/abs/2510.12072)

- EmboMatrix: introduces a scalable training ground for embodied decision-making, integrating an Agents Driven Data Factory (generates tasks/scenes), a Scalable Simulation Backend (executes rollouts), a Hierarchical Reward Architecture (evaluates status/rewards), and training an EmboBrain (generates action sequences), to enable LLMs to acquire genuine embodied decision-making skills.
- This framework generates massive and diverse tasks with efficient simulation and precise rewards, significantly enhancing LLM performance on complex embodied tasks.
- It transforms purely language-trained models into robust, generalizable, and adaptive embodied agents by providing high-throughput interaction and informative supervision.

---

[HiCoTraj: Zero-Shot Demographic Reasoning via Hierarchical Chain-of-Thought Prompting from Trajectory](http://arxiv.org/abs/2510.12067)

- HiCoTraj (Zero-Shot Demographic Reasoning via Hierarchical Chain-of-Thought Prompting from Trajectory): introduces a framework that leverages LLMs' zero-shot learning and semantic understanding capabilities to perform demographic inference without labeled training data, including contextual mobility narrative generation, hierarchical CoT reasoning with factual feature extraction, behavioral pattern analysis, and demographic inference components, where it transforms trajectories into natural language representations and systematically guides LLMs through three cognitive stages for transparent and interpretable inference.
- The framework addresses the scarcity of labeled demographic data by converting numerical trajectories into semantically rich activity chronicles and multi-scale visiting summaries for LLM processing.
- HiCoTraj's hierarchical CoT reasoning systematically decomposes complex demographic inference into manageable cognitive stages, enabling robust reasoning chains from concrete observations to abstract demographic conclusions.

---

[Empowering LLM Agents with Geospatial Awareness: Toward Grounded Reasoning for Wildfire Response](http://arxiv.org/abs/2510.12061)

- GAL (Geospatial Awareness Layer): introduces a novel framework that grounds LLM agents in structured earth data for wildfire response, integrating geospatial information into a perception script for evidence-based recommendations.
- This framework leverages a PostGIS-raster database to retrieve infrastructure, demographic, terrain, and weather attributes, which are then processed by an LLM agent using retrieval-augmented generation and chain-of-thought reasoning.
- Empirical evaluations demonstrate that geospatially grounded LLM agents consistently outperform baselines in forecasting daily personnel and cost, enhancing accuracy and temporal stability for disaster response.

---

[SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding](http://arxiv.org/abs/2510.13016)

- SVAGFormer: introduces a modular transformer framework that jointly integrates spatial localization and temporal grounding to address the Spatio-temporal Video Action Grounding (SVAG) task, utilizing a Temporal Grounding module, a Spatial Grounding module, TempRMOT, FlashVTG, Query Memory, Temporal Feature Layering, and Adaptive Score Refinement.
- The paper also introduces SVAG-Bench, a large-scale, action-centric dataset with dense annotations for multi-instance spatio-temporal video action grounding, and SVAGEval, a standardized evaluation toolkit for fair benchmarking.
- The research highlights that existing models perform poorly on SVAG, especially in dense or complex scenes, underscoring the need for advanced reasoning over fine-grained object-action interactions in long videos.

---

[BENEFITS AND LIMITATIONS OF COMMUNICATION IN MULTI-AGENT REASONING](http://arxiv.org/abs/2510.13903)

- Multi-Agent Reasoning Systems: introduces a theoretical framework to analyze the expressivity of multi-agent systems, formalized as graphs with agents (nodes) performing computation via Transformers, connected by communication and CoT edges, and evaluated on algorithmic tasks using complexity metrics.
- The framework investigates three algorithmic families—associative recall, state tracking, and k-hop reasoning—deriving bounds on agent count, communication quantity, and achievable speedups, identifying regimes where communication is beneficial and delineating tradeoffs.
- Empirical validation with pretrained LLMs on synthetic benchmarks confirms the predicted tradeoffs between computation depth and communication, offering guidance for designing scalable multi-agent reasoning systems.

---

[NARROW FINETUNING LEAVES CLEARLY READABLE TRACES IN ACTIVATION DIFFERENCES](http://arxiv.org/abs/2510.13900)

- ADL (Activation Difference Lens): introduces a methodology to detect and interpret biases from narrow LLM finetuning by analyzing activation differences between base and finetuned models, utilizing Patchscope, Logit Lens, and Steering, and evaluated by an Interpretability Agent.
- The framework demonstrates that narrow finetuning leaves distinct, readable traces in LLM activations, which can be leveraged to understand the finetuning domain without direct access to the training data.
- The approach significantly outperforms blackbox baselines in identifying finetuning objectives across various model organisms and scales, highlighting the need for deeper investigation into finetuning effects and realistic case studies.

---

[Attribution Quality in AI-Generated Content: Benchmarking Style Embeddings and LLM Judges](http://arxiv.org/abs/2510.13898)

- Attribution Quality Assessment Framework: introduces a reproducible benchmark for evaluating attribution quality in AI-generated content, utilizing Style Embeddings Baseline (fixed encoders for stylistic regularities) and an LLM Judge (instruction-tuned LLM for text authenticity) on the HUMAN-AI PARALLEL CORPUS (open dataset for evaluation) with a Binary Classification Task (distinguish human vs. machine-generated text) and McNemar's Test (statistical significance testing for paired predictions).
- This framework systematically compares these two complementary attribution mechanisms across diverse domains (academic, news, fiction, blogs, spoken transcripts, TV/movie scripts) and generator families (GPT-40, LLAMA-70B-INSTRUCT) to quantify their relative strengths and limitations.
- The study reveals that while Style Embeddings generally achieve higher aggregate accuracy, the LLM judge excels in fiction and academic prose, highlighting the need for hybrid strategies that combine structural style signals with semantic reasoning for robust provenance detection.

---

[MultiFoodhat: A potential new paradigm for intelligent food quality inspection](http://arxiv.org/abs/2510.13889)

- MultiFoodChat: introduces a dialogue-driven multi-agent reasoning framework for zero-shot food recognition, integrating vision-language models (VLMs) and LLMs for collaborative reasoning through multi-round visual-textual dialogues.
- The framework utilizes an Object Perception Token (OPT) for capturing fine-grained visual attributes and an Interactive Reasoning Agent (IRA) for dynamically interpreting contextual cues to refine predictions.
- This multi-agent design enables flexible, human-like understanding of complex food scenes without additional training or manual annotations, achieving superior recognition accuracy and interpretability.

---

[Large Language Model Agents Enable Autonomous Design and Image Analysis of Microwell Microfluidics](http://arxiv.org/abs/2510.13883)

- LLM-driven microwell design framework: introduces an autonomous system for generating CAD scripts for microwell geometries and performing image analysis, integrating LLM agents for design, MLLMs for image description, and logistic regression for classification.
- The framework translates natural language prompts into AutoLISP CAD scripts, which are then validated in an AutoCAD environment and used for fabricating microwell microfluidic devices.
- It also employs a multimodal classification pipeline that combines MLLM-generated semantic descriptions with image embeddings to accurately classify microwell occupancy and shape, significantly improving accuracy over direct MLLM inference.

---


#### 13th October 2025

[Demystifying Reinforcement Learning in Agentic Reasoning](http://arxiv.org/abs/2510.11701)

- Demystifying Reinforcement Learning in Agentic Reasoning: introduces a comprehensive investigation into reinforcement learning for agentic reasoning, analyzing Agentic RL Data (Data curation for agents), Agentic RL Algorithm (RL optimization methods), and Agentic Reasoning Mode (Agent decision-making strategies) to identify effective practices.
- The research highlights that real end-to-end trajectories, diverse and model-aware RL datasets, and specific algorithmic techniques like clip higher, token-level loss, and overlong reward shaping significantly enhance agentic reasoning performance.
- The study further reveals that a deliberative reasoning mode with fewer, more targeted tool calls outperforms reactive modes, and maintaining balanced policy entropy is crucial for stable and efficient agentic RL training.

---

[When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents](http://arxiv.org/abs/2510.11695)

- AMA (Agent Market Arena): introduces a lifelong, real-time, multi-class-asset evaluation framework for LLM-based trading agents, integrating a Market Intelligence Stream (MIS) (aggregates, verifies market data), an Agent Execution Protocol (AEP) (standardized agent interaction environment), and a Performance Analysis Interface (PAI) (monitors, analyzes agent performance).
- The framework enables fair and continuous comparison of diverse LLM-based trading agents, including InvestorAgent, TradeAgent, HedgeFundAgent, and DeepFundAgent, across multiple real-time markets using verified data and standardized protocols.
- AMA provides a transparent platform for studying financial reasoning and trading intelligence, demonstrating that agent architecture significantly influences performance more than the underlying LLM backbone.

---

[PACEBENCH: A FRAMEWORK FOR EVALUATING PRACTICAL AI CYBER-EXPLOITATION CAPABILITIES](http://arxiv.org/abs/2510.11688)

- PACEagent: introduces a novel agent designed to emulate human penetration testers, supporting multi-phase reconnaissance, analysis, and exploitation through its LLM Core, Tool Module, and Memory Store, all orchestrated by an Agent Server.
- The agent leverages a Phase Manager to control its operational state and a Tools Router with a Model Context Protocol (MCP) for fine-grained control over specialized cybersecurity tools.
- PACEagent is evaluated on PACEbench, a practical AI cyber-exploitation benchmark simulating real-world cybersecurity challenges with varying vulnerability difficulty, environmental complexity, and cyber defenses.

---

[SR-Scientist: Scientific Equation Discovery With Agentic AI](http://arxiv.org/abs/2510.11661)

- SR-Scientist introduces a framework that elevates LLMs from simple equation proposers to autonomous AI scientists, utilizing a code interpreter, data analyzer tool, equation evaluator tool, experience buffer, long-horizon optimization, and a reinforcement learning pipeline, where the LLM agent autonomously conducts long-horizon optimization using code interpreters for data analysis and equation evaluation.
- The framework employs an experience buffer to manage context length limitations and facilitates long-horizon optimization through iterative interaction with experimental feedback.
- It also integrates a reinforcement learning pipeline, including training data construction, reward design, and a training algorithm, to continuously enhance the agent's scientific discovery abilities.

---

[ACADREASON: Exploring the Limits of Reasoning Models with Academic Research Problems](http://arxiv.org/abs/2510.11652)

- ACADREASON: introduces a benchmark for evaluating LLMs' and agents' academic-level reasoning abilities, featuring a multi-stage pipeline for collecting high-quality academic problems, extracting research questions, and generating comprehensive evaluation criteria, including High-Quality Academic Papers Collection, High-Reasoning Research Question Extraction, Checklists and Hints Extraction, Evaluation Pipeline, Candidate Response, Golden Answer, Checklist, Hints, GPT-5 mini, and Scores.
- The benchmark includes 50 expert-annotated problems across five high-reasoning domains, providing detailed hints (background, definition, methodology) and dynamic checklists to guide and assess complex reasoning processes.
- The evaluation employs an LLM-as-Judge approach, utilizing GPT-5 mini to score candidate responses against golden answers and checklists, thereby measuring both exact matches (Pass Rate) and adherence to reasoning milestones (Checklist Score).

---

[ParaCook: On Time-Efficient Planning for Multi-Agent Systems](http://arxiv.org/abs/2510.11608)

- ParaCook: introduces a benchmark for time-efficient collaborative planning in multi-agent systems, including an Environment (2D kitchen simulation), Task (cooking challenges), Difficulty Control (task complexity), Metrics (plan evaluation), and an LLM Planner (planning agent).
- The benchmark evaluates LLMs' ability to schedule tasks and coordinate agents to minimize overall completion time in a simulated kitchen, focusing on both intra-agent and inter-agent parallelism.
- ParaCook provides a scalable evaluation framework with adjustable complexity, enabling systematic assessment of LLM planning capabilities for multi-agent scheduling.

---

[ANALYZING AND INTERNALIZING COMPLEX POLICY DOCUMENTS FOR LLM AGENTS](http://arxiv.org/abs/2510.11588)

- CAP-CPT (Category-Aware Policy Continued Pretraining): introduces an automated pipeline for analyzing and internalizing complex policy documents for LLM agents, including Policy Document Analysis and Categorization, LLM-based Preprocessing, Manual Check, Policy Specification Types, Targeted Continue Pretraining Data Generation, Policy Identifier Representation, Policy Paraphrase Generation, Policy Content QA Generation, Behavior Demonstration Generation, Scenario Simulation, LLM-driven Instance Sampling, LLM Template Simulation, LLM Data Generation, Trajectory Familiarization, and LLM-based CPT Data Generation, which systematically categorizes policy specifications and generates tailored data for continued pretraining.
- The framework addresses challenges in internalizing complex policy documents by creating specialized training data for factual, behavioral, and conditional policy types, significantly improving LLM agent performance and reducing input token length.
- CAP-CPT leverages LLMs for policy analysis and data synthesis, enabling more effective policy internalization, especially in data-sparse and high-complexity scenarios, and achieves up to 97.3% prompt length reduction.

---

[ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding](http://arxiv.org/abs/2510.11498)

- ReLook (vision-grounded agentic reinforcement learning framework): introduces a vision-grounded agentic reinforcement learning framework that empowers an agent to close a robust generate-diagnose-refine loop by invoking an MLLM as a tool, including a Policy LLM, MLLM Critic, Render Check, Rule-reward, Model-reward, Group Relative Reward, Forced Optimization, GRPO (Group Relative Policy Optimization), History, Interact, Rollout QA, and Critic FB, where the agent learns to "see" rendered outputs and obtain rich textual suggestions for iterative refinement.
- The framework employs a comprehensive reward system, combining MLLM-based visual scoring with a strict zero-reward rule for invalid renders, and utilizes Forced Optimization to ensure monotonically improving trajectories during training.
- For efficient inference, ReLook decouples the critic and runs a lightweight, critic-free self-edit cycle, preserving performance gains while substantially reducing latency.



---

[Who are you, ChatGPT? Personality and Demographic Style in LLM-Generated Content](http://arxiv.org/abs/2510.11434)

- LLM-PDSA Framework: introduces a novel, data-driven methodology for assessing LLM personality and demographic style by applying automatic personality and gender classifiers to LLM-generated content and comparing it to human-authored responses.
- The framework utilizes a Reddit Data Collection Module, an LLM Response Generation Module, a Personality Trait Classifier, a Gender Likelihood Classifier, and a Comparative Analysis Module to analyze text from six diverse LLMs against human baselines.
- The study reveals that LLMs systematically exhibit higher Agreeableness and lower Neuroticism, and their gendered language patterns broadly align with human writers, though with reduced variation.

---

[Uncertainty-Aware, Risk-Adaptive Access Control for Agentic Systems using an LLM-Judged TBAC Model](http://arxiv.org/abs/2510.11414)

- Uncertainty-Aware, Risk-Adaptive TBAC model: introduces an advanced security framework that extends the Task-Based Access Control (TBAC) model by using an LLM Judge (Large Language Model Judge) to synthesize just-in-time policies, calculate composite risk, and estimate model uncertainty, enabling dynamic access control decisions.
- This framework integrates Immutable Security Principles and a Risk-Enriched Tool Manifest to guide the LLM Judge, which then outputs a Policy, Composite Risk, and Model Uncertainty for evaluation against predefined Thresholds within the Task Authorization Service.
- Requests exceeding risk or uncertainty Thresholds are escalated to a Human Security Officer, while others receive Autonomous Approval and a Capability Token, ensuring robust and adaptive least privilege for autonomous AI agents.

---

[Beyond Survival: Evaluating LLMs in Social Deduction Games with Human-Aligned Strategies](http://arxiv.org/abs/2510.11389)

- WereAlign (strategy-alignment evaluation paradigm): introduces a novel framework for evaluating LLMs in social deduction games, utilizing the WereBench Dataset, a Speech Evaluation Stage with five dimensions (RI, SJ, DR, PS, CT), and a Decision Evaluation Stage with two tasks (VA, OI).
- The framework employs Question Design, Positive Option Generation, and Negative Option Generation modules, including Counterfactual Context Perturbation (M1) and Strategic Rationale-Driven Generation (M2) mechanisms, to create human-aligned evaluation tasks.
- WereAlign also incorporates a Controlled Intervention Experiment Module with Rule Reminder (RR) and Objective Speech Rewriting (OSR) mechanisms to analyze specific factors influencing LLM performance.

---

[Part II: ROLL Flash – Accelerating RLVR and Agentic Training with Asynchrony](http://arxiv.org/abs/2510.11345)

- ROLL Flash: introduces a system for accelerating RLVR and agentic training with asynchrony, built on fine-grained parallelism and rollout-train decoupling, and featuring LLMProxy, EnvManager, SampleBuffer, and AsyncController.
- This framework significantly improves resource utilization and scalability by enabling parallel execution of rollout and training stages, mitigating long-tail latency issues in LLM generation.
- ROLL Flash achieves substantial speedups on RLVR and agentic tasks while maintaining training stability through mechanisms like queue scheduling, prompt replication, and an asynchronous ratio.

---

[Evolution in Simulation: AI-Agent School with Dual Memory for High-Fidelity Educational Dynamics](http://arxiv.org/abs/2510.11290)

- AAS (AI-Agent School): introduces a multi-agent simulation environment designed to model and accelerate the evolution of educational cognitive processes through situated interactions, featuring AI Agents, a Zero-Exp mechanism, and a comprehensive Memory System.
- The Zero-Exp mechanism, central to AAS, employs a continuous "experience-reflection-optimization" cycle, grounded in a dual memory base (Experience and Knowledge Bases) with short-term and long-term components, enabling agents to autonomously evolve.
- This framework addresses the lack of systematic teaching process modeling and limitations in simulating diverse educational participants, providing a verifiable technical model for educational digital twins and high-fidelity behavioral data generation.

---

[PADME: Procedure Aware DynaMic Execution](http://arxiv.org/abs/2510.11281)

- PADME (Procedure Aware DynaMic Execution): introduces a two-phase agent framework that transforms unstructured procedural text into executable decision graphs for robust, generalizable execution, including Teach Phase, Procedure, Procedure Structuring Agent, Procedure Extraction, Procedure Segmentation, Structuring, Aggregation, Decision Graph, Code Generation, Tools, Execute Phase, Task, Procedure Execution Agent, Graph Execution Plan Generation, Plan Execution, Dynamic Plan Expansion, Executable Decision Graph, Tools, User Input, and Execution Output.
- The Teach phase, involving a Procedure Structuring Agent and Code Generation, converts raw procedures into an Executable Decision Graph, while the Execute phase, managed by a Procedure Execution Agent, dynamically traverses and executes this graph using real-time context and tools.
- This framework leverages graph-based representations, including Human Input, Information Processing, Information Extraction, Knowledge, and Decision nodes, to reduce error accumulation and enable adaptive execution across diverse domains.

---

[A LARGE-LANGUAGE-MODEL ASSISTED AUTOMATED SCALE BAR DETECTION AND EXTRACTION FRAMEWORK FOR SCANNING ELECTRON MICROSCOPIC IMAGES](http://arxiv.org/abs/2510.11260)

- LLM-ASBDEF introduces an automated multi-modal framework for scale bar detection and extraction in SEM images, integrating an Auto-DG module, a YOLO-based object detection model, a hybrid OCR system, and an LLM agent for verification and feedback.
- The framework operates in four phases: automatic dataset generation, object detection, information extraction, and LLM-driven verification, providing concurrent object detection, text detection, and text recognition.
- This automated method, powered by an LLM agent, significantly enhances the efficiency and accuracy of scale bar detection and extraction, offering a valuable tool for microscopic analysis and scientific imaging.

---

[Collaborative Shadows: Distributed Backdoor Attacks in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2510.11246)

- Collaborative Shadows: introduces a novel distributed backdoor attack paradigm for LLM-based Multi-Agent Systems, leveraging its Decomposer, Attack Primitives, Serializer, Encryptor, Steganographic Header, Poisoned Tools, Observation Manipulator, Uniqueness Regulator, Multi-Agent System, User Instruction, Extractor, Decryptor, Assembler, and Executor components to exploit agent collaboration for targeted attacks.
- This framework decomposes a backdoor into distributed primitives embedded within MAS tools, which remain dormant until a carefully crafted user instruction triggers their sequential activation and assembly for execution.
- The attack achieves high success rates without degrading benign task performance, highlighting critical collaboration-driven vulnerabilities in MAS and the need for advanced defense mechanisms.

---

[WEBROUTER: QUERY-SPECIFIC ROUTER VIA VARIATIONAL INFORMATION BOTTLENECK FOR COST-SENSITIVE WEB AGENT](http://arxiv.org/abs/2510.11221)

- WebRouter: introduces a novel query-specific router for LLM-brained web agents, utilizing a Query (input prompt), Query Encoder (generates embeddings), Embeddings (latent representations), WebRouter (LLM selection module), LLM Ensemble (candidate LLMs pool), and a Cost-aware Variational Information Bottleneck objective (router training method) to address the cost-performance trade-off and noisy input prompts.
- The framework learns a compressed representation of the input prompt, explicitly filtering irrelevant information while preserving critical features for routing decisions, thereby matching each web query to the most cost-effective LLM.
- Experiments demonstrate that WebRouter significantly reduces operational costs by 87.8% compared to a GPT-40 baseline, with only a 3.8% accuracy drop, showcasing its efficiency and robustness in real-world web agent scenarios.

---

[TraceAegis: Securing LLM-Based Agents via Hierarchical and Behavioral Anomaly Detection](http://arxiv.org/abs/2510.11203)

- TRACEAEGIS: introduces a provenance-based anomaly detection framework for LLM-based agents, including a Behavior Profiling phase (models agent behaviors structurally and semantically) and a Violation Detection phase (checks new execution paths against profiled behaviors).
- The framework leverages agent execution traces to reconstruct a hierarchical structure of tool invocations and derive constrained behavioral rules, enabling the detection of both structural inconsistencies and semantic violations.
- TRACEAEGIS-BENCH, a new benchmark, and real-world red-teaming experiments validate TRACEAEGIS's effectiveness in identifying abnormal agent behaviors, outperforming existing LLM baselines.

---

[CAN TOOL-INTEGRATED REINFORCEMENT LEARNING GENERALIZE ACROSS DIVERSE DOMAINS?](http://arxiv.org/abs/2510.11184)

- TGRL (Tool Generalization Reinforcement Learning): introduces a framework designed to promote domain-agnostic learning and skill migration, encompassing a standardized tool interface (unified interface, consistent answer formatting), a dual-component reward system (correct outcomes, proper tool-use formats), and an XML-based prompt template (structured template, multi-turn interactions).
- This framework enables an LLM agent, trained solely on mathematical problem-solving tasks with a code interpreter, to effectively generalize its tool usage to diverse and unseen reasoning domains.
- TGRL achieves state-of-the-art performance by fostering transferable skills in tool invocation and reasoning abstraction, addressing limitations of prior multi-domain training approaches.

---

[TypePilot: Leveraging the Scala type system for secure LLM-generated code](http://arxiv.org/abs/2510.11151)

- TypePilot (an agentic AI framework): introduces a multi-step LLM-based code generation pipeline that leverages the Scala type system to enhance the security and robustness of LLM-generated code.
- The framework employs three distinct LLMs: one for initial code generation, a second for vulnerability detection, and a third for refining the code by applying Scala's type system to address identified vulnerabilities.
- This approach significantly mitigates input validation and injection vulnerabilities, transforming type systems from passive enforcers into active agents of code safety.

---

[How²: How to learn from procedural How-to questions](http://arxiv.org/abs/2510.11144)

- How² (memory agent framework): introduces a lifelong learning system for agents in interactive environments, featuring an Actor (main agent loop), Memory (key-value store), Relevance Check (filters memory entries), Ask Question (generates how-to query), Teacher (provides procedural answers), Parse Answer (abstracts and tags answers), and Environment (interactive simulation) to learn and reuse procedural knowledge from how-to questions.
- The framework enables LLM-based agents to improve planning capabilities by asking questions, storing answers, and reusing abstracted knowledge, balancing immediate utility with long-term reusability.
- It demonstrates that abstracting teacher answers into subgoal structures and decoupling them from the current state significantly enhances knowledge reusability and agent performance in tasks like Minecraft crafting.

---

[VIDEO-SALMONN S: STREAMING AUDIO-VISUAL LLMS BEYOND LENGTH LIMITS VIA MEMORY](http://arxiv.org/abs/2510.11129)

- video-SALMONN S: introduces a streaming audio-visual LLM capable of understanding long videos, with a TTT-HF Layer (updates token representations; incorporates history; uses Hessian-free optimization), Prompt-Dependent Reading (selects relevant KV-cache entries based on prompt), LLM (generates response), LoRA (low-rank adapter; trainable parameters), Video Encoding Xt (input video frames converted to encodings), Previous Memory Tokens (stores historical information), Similarity Token Discarding (reduces memory to fixed size), New Incoming Tokens Zt (output from TTT-HF layer; added to memory), Prompt (user query for memory retrieval), and Audio tokens (bypass TTT-HF layer; directly appended), designed to process >3-hour videos at 1 FPS and 360p resolution under a fixed memory budget.
- The framework employs a novel streaming video understanding approach by continually updating token representations via a TTT memory module and selectively retrieving context-relevant content using a prompt-dependent memory reader.
- This design enables high-quality understanding of multi-hour videos with over 10k frames and ~1M tokens, outperforming both offline and streaming baselines on long-video benchmarks.

---

[A Vision for Access Control in LLM-based Agent Systems](http://arxiv.org/abs/2510.11108)

- AAC (Agent Access Control): introduces a novel framework that redefines access control as a dynamic, context-aware process of information flow governance, integrating Multi-dimensional Contextual Evaluation, Adaptive Response Formulation, and a dedicated AC Reasoning Engine.
- This framework moves beyond traditional binary allow/deny decisions by holistically analyzing interaction context and adaptively shaping information outputs through redaction, summarization, and paraphrasing.
- The dedicated AC Reasoning Engine operates independently of the primary LLM, acting as a "cognitive conscience" to ensure robust and explainable permission allocation and information flow governance.

---

[DebugTA: An LLM-Based Agent for Simplifying Debugging and Teaching in Programming Education](http://arxiv.org/abs/2510.11076)

- DebugTA (Debugging and Teaching LLM Agent): introduces an LLM-based agent that integrates debugging and teaching for programming education by leveraging specialized tools and a memory module to simplify complex tasks and improve suggestion accuracy.
- The agent decomposes complex debugging and teaching tasks into sequential LLM interactions, each utilizing distinct tools for specific subtasks, thereby minimizing reasoning complexity and enhancing reliability.
- DebugTA employs a standard code retrieval tool, a variable substitution tool for aligning reference code, and an external compiler interface for real-time code analysis and validation, guided by pedagogical and debugging principles.

---

[STRONGER TOGETHER: ON-POLICY REINFORCEMENT LEARNING FOR COLLABORATIVE LLMS](http://arxiv.org/abs/2510.11062)

- AT-GRPO (Agent- and Turn-wise Grouped Reinforcement Learning for Multi-Agent Systems): introduces a novel training system and algorithm for on-policy RL in multi-agent LLM systems, featuring LLM resource pools, environment execution, MAS control, and data routing.
- The system supports both role-sharing and role-specialized policies, enabling concurrent training of multiple LLM models and efficient management of diverse MAS workflows.
- AT-GRPO significantly improves accuracy and reasoning performance across various domains like planning, coding, and math by reinforcing role-specific specialization and enhancing inter-agent coordination.

---

[SusBench: An Online Benchmark for Evaluating Dark Pattern Susceptibility of Computer-Use Agents](http://arxiv.org/abs/2510.11035)

- SusBench: introduces an online benchmark for evaluating the susceptibility of LLM-based Computer-Use Agents (CUAs) to UI dark patterns, utilizing a Controller, Browser Extension with Injection Function Store, Page Match & Inject, and Eval Result, a Playwright Browser, and Human/Agent subjects, with LLM and Researcher involvement for creating and validating dark pattern injections on Real-world Websites.
- The benchmark employs a data-construction method that injects believable dark patterns into live, real-world consumer websites through UI code injections, encompassing 313 evaluation tasks across 55 websites and 9 common dark pattern types.
- The study found that both human participants and CUAs are particularly susceptible to Preselection, Trick Wording, and Hidden Information, highlighting the need for developing more trustworthy CUAs and their potential as human proxies for evaluating deceptive designs.

---

[Automating Structural Engineering Workflows with Large Language Model Agents](http://arxiv.org/abs/2510.11004)

- MASSE (Multi-Agent System for Structural Engineering): introduces a multi-agent framework for structural engineering, effectively integrating LLM-based agents with real-world engineering workflows to automate structural design tasks.
- The framework includes an Analyst Team for data extraction and analysis, an Engineer Team for design and verification, and a Management Team for coordination and decision-making, all supported by LLM, FEM, document, and fundamental tools.
- MASSE significantly reduces expert workload from hours to minutes while enhancing reliability and accuracy in practical engineering scenarios by operationalizing professional workflows through specialized LLM agents and structured communication.

---

[The Social Cost of Intelligence: Emergence, Propagation, and Amplification of Stereotypical Bias in Multi-Agent Systems](http://arxiv.org/abs/2510.10943)

- MAS (Multi-Agent Systems): introduces a comprehensive study of stereotypical bias in MAS, examining how internal specialization, underlying LLMs, and inter-agent communication protocols influence bias robustness, propagation, and amplification.
- The research simulates social contexts where agents represent different social groups, evaluating system behavior under various interaction and adversarial scenarios using three bias benchmarks.
- Findings indicate MAS are generally less robust than single-agent systems, with bias emerging early through in-group favoritism, though cooperative and debate-based communication can mitigate bias amplification, and robust LLMs improve system stability.

---

[PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents](http://arxiv.org/abs/2510.10931)

- PoU (Proof-of-Use): introduces an evidence-grounded RL framework that enforces verifiable causal links between retrieved evidence, reasoning traces, and final answers through a unified step-wise contract, including syntactic citation validation, perturbation-based sensitivity rewards, and answer-evidence alignment objectives.
- This framework addresses "Tool-Call Hacking" in RAG agents, where models superficially satisfy reward signals without genuinely using retrieved evidence, leading to mode collapse and spurious grounding.
- PoU transforms reasoning supervision from heuristic imitation to contract-driven optimization, enabling agents to align internal reasoning dynamics with external factual dependencies for trustworthy retrieval-augmented reasoning.

---

[PaperArena: An Evaluation Benchmark for Tool-Augmented Agentic Reasoning on Scientific Literature](http://arxiv.org/abs/2510.10909)

- PaperArena (PaperArena-Hub): introduces an evaluation benchmark and platform for tool-augmented agentic reasoning on scientific literature, featuring a Benchmark Construction Pipeline with a comprehensive Tool Library, Heuristic QA Pair Generation, Semi-Automated QA Verification, an Agent Evaluation Platform with an Agent Platform, and an Evaluation Module.
- The benchmark challenges LLM-based agents with real-world research questions requiring multi-step, multi-modal, and cross-document reasoning, along with diverse tool orchestration.
- The platform provides a modular and extensible environment for standardized evaluation, revealing significant performance gaps and inefficient tool usage in current LLM agents.

---

[LLM-Empowered Agentic MAC Protocols: A Dynamic Stackelberg Game Approach](http://arxiv.org/abs/2510.10895)

- LLM-empowered MARL framework: introduces a game-theoretic approach for MAC protocol emergence in wireless networks, utilizing LLM-driven agents, a dynamic multi-follower Stackelberg game, proximal policy optimization, and protocol action grammar.
- This framework models uplink transmission as a hierarchical game between a Base Station (leader) and User Equipments (followers), enabling adaptive, semantic MAC protocol synthesis in response to network dynamics.
- The system leverages LLMs for generalization and exploratory learning, ensuring reliable and efficient policy convergence in dynamic environments without requiring retraining for fluctuating user numbers.

---

[LLM×MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System](http://arxiv.org/abs/2510.10890)

- LLM×MapReduce-V3: introduces an interactive, self-organized, hierarchically modular agent system for long-form survey generation, featuring a User Input (for topic and files), Human-in-the-loop (for interaction), specialized agents (Analysis Agent, Search Agent, Skeleton Agent, Writing Agent, User Customized Agent), and a suite of MCP Servers (Search Server, Group Server, Skeleton Initialize Server, Skeleton Refinement Server, Digest Server, Orchestra Server, User Customized Server) that collectively generate a System Output (comprehensive survey article).
- The system leverages a Model Context Protocol (MCP) for standardized function-calling, enabling dynamic planning by an LLM-driven Orchestra Server that orchestrates multi-stage workflows for document digestion, skeleton construction, refinement, and survey writing.
- This architecture facilitates human-in-the-loop intervention and customization, allowing users to guide the research process and adapt workflows to specific writing tasks, ensuring alignment with user intent and scholarly rigor.

---

[Rethinking Agentic Workflows: Evaluating Inference-Based Test-Time Scaling Strategies in Text2SQL Tasks](http://arxiv.org/abs/2510.10885)

- Agentic Workflows for Text2SQL Tasks: introduces an evaluation of six inference-based test-time scaling strategies and their constituent LLM-powered agents and tools, assessing their performance on Text-to-SQL tasks.
- The study benchmarks these strategies across four LLMs on the BIRD Mini-Dev dataset, measuring SQL accuracy, inference latency, and token consumption to provide practical deployment insights.
- Findings indicate that Divide-and-Conquer prompting and few-shot demonstrations consistently enhance performance, while the effectiveness of additional workflow complexity varies and depends on the base LLM.

---

[Agentic Systems in Radiology: Design, Applications, Evaluation, and Challenges](http://arxiv.org/abs/2510.09404)

- LLM-based Agentic System: introduces a conceptual architecture for LLM-driven agents in radiology, detailing components like LLM, memory, tools, and environment interaction, to support complex, multi-step radiological tasks.
- The paper examines design patterns for agentic systems, including single LLM calls, compositional workflows, and multi-agent systems, highlighting their application in tasks like report drafting and follow-up scheduling.
- It also discusses evaluation methods for planning, execution, and outcomes, and outlines challenges such as LLM core limits, cascading errors, multi-agent coordination, and health IT integration.

---

[DSPO: Stable and Efficient Policy Optimization for Agentic Search and Reasoning](http://arxiv.org/abs/2510.09255)

- DSPO (Dynamic-filter Sequence-level Policy Optimization): introduces an improved RL algorithm for robust agent training, which includes a Policy Model (LLM agent), a Reference Model (old policy), a Search Engine (external knowledge source), a Dynamic Filtering mechanism (batch selection), and a Group Advantage Computation module (advantage signal calculation), to achieve stable and efficient policy optimization for agentic search and reasoning.
- The framework addresses LLM agent training instability and sample inefficiency by employing sequence-level optimization for robust policy updates and dynamic outcome-based filtering for a dense and effective learning signal.
- DSPO's dynamic filtering ensures training batches contain mixed successful and unsuccessful outcomes, preventing advantage signal collapse, while sequence-level optimization stabilizes training by aligning reward and optimization units.

---

[DITING: A Multi-Agent Evaluation Framework for Benchmarking Web Novel Translation](http://arxiv.org/abs/2510.09116)

- DITING (Multi-Agent Evaluation Framework for Benchmarking Web Novel Translation): introduces a comprehensive evaluation framework for web novel translation, assessing narrative and cultural fidelity across six dimensions: idiom translation, lexical ambiguity, terminology localization, tense consistency, zero-pronoun resolution, and cultural safety, supported by over 18K expert-annotated Chinese-English sentence pairs.
- The framework further proposes AgentEval, a reasoning-driven multi-agent evaluation system that simulates expert deliberation to assess translation quality beyond lexical overlap, achieving high correlation with human judgments.
- DITING also includes MetricAlign, a meta-evaluation dataset of 300 sentence pairs annotated with error labels and scalar quality scores, enabling systematic comparison of evaluation metrics.

---

[Operand Quant: A Single-Agent Architecture for Autonomous Machine Learning Engineering](http://arxiv.org/abs/2510.11694)

- Operand Quant: introduces a single-agent, IDE-based architecture for autonomous machine learning engineering, consolidating all MLE lifecycle stages within a single, context-aware agent.
- This architecture operates through a non-blocking, turn-based reasoning-execution cycle, continuously observing the IDE state, planning actions, editing/running code, and evaluating outcomes.
- It achieves state-of-the-art performance on the MLE-Benchmark by maintaining a unified reasoning state, supporting concurrent execution, and employing a deep-thinking ensemble for complex problem-solving.

---

[IntersectioNDE: Learning Complex Urban Traffic Dynamics based on Interaction Decoupling Strategy](http://arxiv.org/abs/2510.11534)

- IntersectioNDE (Intersection Naturalistic Driving Environment): introduces a data-driven scene-level simulator for complex urban traffic, leveraging its Interaction Decoupling Strategy (IDS) for compositional training, implemented via a Scene-aware Interaction Transformer network that includes an Embedding Layer, Interaction Attention Module, and Prediction Head, for both Open-loop Training and Closed-loop Inference.
- The framework addresses challenges in modeling dense, heterogeneous interactions and high-dimensional joint distributions by partitioning scenes into agent subsets, enabling marginal-to-joint simulation for enhanced robustness and stability.
- Experiments on the newly introduced City Crossings Dataset (CiCross) demonstrate IntersectioNDE's superior performance in simulation fidelity, stability, and ability to replicate complex urban traffic dynamics.

---

[MODELING AI-DRIVEN PRODUCTION AND COMPETITIVENESS: A MULTI-AGENT ECONOMIC SIMULATION OF CHINA AND THE UNITED STATES](http://arxiv.org/abs/2510.11085)

- Multi-Agent Economic Simulation Framework: introduces a comparative analysis based on five progressive intelligent-agent economic models, including pure human collaboration, AI collaboration, AI collaboration with network effects, AI as an independent productive entity, and an integrated model, to evaluate the output performance of China and the United States following AI-agent integration.
- The framework quantitatively analyzes the impact of AI agent participation on total social output, revealing how AI-driven productivity gains and network externalities shape economic competitiveness between the two nations.
- The study highlights China's potential for accelerated advancement in AI agent expansion and capability, suggesting a dual-path strategy for closing the output gap with the United States.

---

[Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling](http://arxiv.org/abs/2510.11083)

- Flow Planner: introduces a novel learning-based framework for autonomous driving planning, integrating fine-grained trajectory tokenization, an interaction-enhanced spatiotemporal fusion architecture, and flow matching with classifier-free guidance to model interactive behaviors.
- The framework addresses challenges in complex driving scenarios by decomposing trajectories into overlapping segments, efficiently fusing heterogeneous scene information, and dynamically reweighting agent interactions during inference.
- Flow Planner achieves state-of-the-art performance on benchmarks like nuPlan and interPlan, demonstrating robust interactive behavior modeling and adaptability to unseen scenarios.

---

[Audio-Guided Visual Perception for Audio-Visual Navigation](http://arxiv.org/abs/2510.11760)

- AGVP (Audio-Guided Visual Perception): introduces an audio-visual navigation framework that transforms sound into spatial guidance by explicitly aligning auditory and visual features, enabling robust navigation in unknown 3D environments, with Environment, Observations, RGB, Depth, Left, Right, Observations Encoder, Visual Encoder, Audio Encoder, AGVP Module, SA, GA, GRU, Decisions, Actor, Critic, and Action Sampler components.
- The framework employs a "sound first, vision follows" multimodal fusion mechanism, where audio context recalibrates visual feature maps to highlight sound-source-related regions.
- This design reduces dependency on specific acoustic fingerprints, improving navigation efficiency and cross-scenario generalization, especially with unheard sounds.

---

[A Survey on Agentic Multimodal Large Language Models](http://arxiv.org/abs/2510.10991)

- Agentic MLLMs Conceptual Framework: introduces a comprehensive survey on Agentic Multimodal Large Language Models, defining their architecture through Foundational MLLM, Agentic Internal Intelligence, Agentic External Tool Invocation, and Agentic Environment Interaction components.
- The framework highlights Agentic MLLMs' dynamic and adaptive workflow, proactive action execution, and strong generalization across diverse domains, contrasting them with static, passive, and domain-specific traditional MLLM agents.
- Agentic MLLMs achieve autonomy through reasoning, reflection, memory, tool use, and interaction with environments, enabling adaptive strategies and goal-directed behavior in real-world scenarios.

---

[Game-Theoretic Risk-Shaped Reinforcement Learning for Safe Autonomous Driving](http://arxiv.org/abs/2510.10960)

- GTR2L (Game-Theoretic Risk-Shaped Reinforcement Learning): introduces a safe RL framework for autonomous driving, integrating a World Model, Reachability Modeling, and risk-constrained RL, where it enhances safety and robustness in dynamic traffic environments.
- The framework's World Model predicts interactive behaviors and risks using multi-level game-theoretic reasoning and an adaptive rollout horizon, while Reachability Modeling defines feasible regions with a dynamic barrier policy.
- GTR2L incorporates a dedicated risk modeling approach to capture both epistemic and aleatoric uncertainty, guiding constrained policy optimization and improving decision-making in complex scenarios.

---

[Neutral Agent-based Adversarial Policy Learning against Deep Reinforcement Learning in Multi-party Open Systems](http://arxiv.org/abs/2510.10937)

- NAAPL (Neutral Agent-based Adversarial Policy Learning): introduces a novel adversarial attack method against Deep Reinforcement Learning (DRL) in multi-party open systems, training neutral agents to learn adversarial policies that mislead victim agents without direct interaction or full environmental control.
- The method redesigns reward functions by leveraging victim failure paths and employs an estimation-based reward model, utilizing an LSTM network, to calculate rewards from partial observations without requiring global state.
- Evaluated on SMAC and Highway-env platforms, NAAPL demonstrates generalizable and effective adversarial attacks across diverse multi-party open system scenarios, proving robust against existing countermeasures.

---

[ProSEA: Problem Solving via Exploration Agents](http://arxiv.org/abs/2510.07423)

- ProSEA (Problem Solving via Exploration Agents): introduces a hierarchical multi-agent framework, where LLM-based agents engage in iterative problem solving through exploration and adaptive plan evolution, with a Manager Agent (orchestrates, coordinates, evaluates, synthesizes), a Problem Analyzer (analyzes problem, extracts constraints), a Planner (generates plan, decomposes tasks), and Expert Agents (execute tasks, explore, feedback), integrating External Tools (resources for execution), Domain Knowledge (specialized information base), and Human in the loop (collaborator, provides input).
- The framework employs a novel feedback-driven approach where expert LLM agents provide rich, structured feedback on failures and discoveries, enabling adaptive plan refinement and two-dimensional exploration.
- ProSEA demonstrates superior performance on complex reasoning tasks autonomously, while also supporting seamless human collaboration for transparent and adaptive AI systems.

---

[HOLISTIC AGENT LEADERBOARD: THE MISSING INFRASTRUCTURE FOR AI AGENT EVALUATION](http://arxiv.org/abs/2510.11977)

- HAL (Holistic Agent Leaderboard): introduces a unified evaluation framework for AI agents, featuring a standardized evaluation harness (orchestrates parallel evaluations), a multidimensional leaderboard (analyzes models, scaffolds, benchmarks), and automated log analysis (LLM-aided log inspection).
- The framework orchestrates parallel evaluations across hundreds of VMs, tracks performance across three dimensions (models, scaffolds, benchmarks), and uses LLM-aided log inspection to identify agent behaviors and failure causes.
- HAL aims to standardize agent evaluation, reduce evaluation time, provide comprehensive performance insights beyond accuracy, and uncover problematic agent behaviors for more reliable real-world deployment.

---

[Scaling Long-Horizon LLM Agent via Context-Folding](http://arxiv.org/abs/2510.11967)

- Context-Folding: introduces an agentic mechanism for LLM agents to actively manage their working context, coupled with FoldGRPO, an end-to-end reinforcement learning framework, to enable learnable context management.
- The framework allows an LLM agent (Policy Model) to procedurally branch into a sub-trajectory for subtasks using a `branch action` and then `fold` it upon completion via a `return action`, collapsing intermediate steps while retaining a concise summary.
- FoldGRPO utilizes a `Context Manager F` and dense `Fold Reward` signals, including `Unfolded Token Penalty` and `Out-of-Scope Penalty`, to guide the agent in effective task decomposition and context management, leading to improved performance and efficiency on long-horizon tasks.

---

[R-WOM: RETRIEVAL-AUGMENTED WORLD MODEL FOR COMPUTER-USE AGENTS](http://arxiv.org/abs/2510.11892)

- R-WoM (Retrieval-augmented World Model): introduces a framework that grounds LLM-based world models with external tutorials, enabling environment-specific adaptation through retrieval-augmented simulation and listwise reward estimation for computer-use agents.
- The framework enhances LLM simulations by incorporating factual, up-to-date knowledge retrieved from external tutorials to mitigate hallucination and reliance on static training knowledge, particularly for long-horizon tasks.
- R-WoM leverages a reasoning-based RAG pipeline for query rewriting and LLM-based reranking to improve the relevance of retrieved tutorials, and employs a LongCoT mechanism for multi-step simulation and listwise reward estimation for robust action selection.

---

[Deep Research Brings Deeper Harm](http://arxiv.org/abs/2510.11851)

- WebThinker (Deep Research Agent): introduces a study evaluating the safety vulnerabilities of Deep Research (DR) agents, which leverage LLMs for multi-step research, by demonstrating how they can bypass safety mechanisms and generate harmful content.
- The paper proposes two jailbreak methods, Plan Injection and Intent Hijack, specifically designed to exploit the planning and research-oriented design of DR agents.
- It also introduces DeepREJECT, a new evaluation metric to assess the practical harmfulness of detailed reports generated by DR agents, highlighting the need for tailored alignment techniques.

---

[Lingxi: Repository-Level Issue Resolution Framework Enhanced by Procedural Knowledge Guided Scaling](http://arxiv.org/abs/2510.11838)

- Lingxi (Repository-Level Issue Resolution Framework Enhanced by Procedural Knowledge Guided Scaling): introduces a framework that leverages procedural knowledge extracted from historical issue-fixing data to guide LLM-powered agents in solving complex repository-level issues, featuring a Procedural Knowledge Construction component for offline knowledge creation, a Knowledge-guided Issue Analysis Scaling component for parallel issue analysis, and an Issue Resolution component for generating and executing fix plans.
- The framework constructs transferable procedural knowledge through a hierarchical abstraction mechanism and employs a knowledge-driven scaling method to intelligently analyze target issues from multiple perspectives, contrasting with undirected brute-force exploration.
- Lingxi achieves a 74.6% resolution rate on the SWE-bench Verified benchmark, outperforming state-of-the-art techniques by a significant margin, with transferable knowledge and knowledge-guided scaling being critical to its performance.

---

[A2FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning](http://arxiv.org/abs/2510.12838)

- A2FM (Adaptive Agent Foundation Model): introduces a unified framework that integrates instant, reasoning, and agentic modes under a single backbone via a self-adaptive router, which learns task-aware routing and aligns mode-specific trajectories.
- The framework employs a two-stage training process, starting with supervised route-then-align fine-tuning for mode-conditioned trajectories, followed by Adaptive Policy Optimization (APO) for dynamic mode selection with adaptive sampling and cost-regularized rewards.
- A2FM's agentic mode leverages external tools like web_search, crawl_page, and code_execute, along with planning and summary components, to handle complex real-world tasks efficiently and accurately.

---

[AWARECOMPILER: AGENTIC CONTEXT-AWARE COMPILER OPTIMIZATION VIA A SYNERGISTIC KNOWLEDGE-DATA DRIVEN FRAMEWORK](http://arxiv.org/abs/2510.11759)

- AwareCompiler: introduces an agentic framework for compiler optimization that integrates structured knowledge integration, dataset construction, knowledge-driven adaptive pass generation, and a data-driven hybrid training pipeline, addressing challenges in LLM-based software optimization by generating context-aware optimization sequences.
- The framework leverages a comprehensive knowledge base, including empirical, symbolic, and negative knowledge, to bridge the semantic gap between program representations and optimization passes.
- Its hybrid training pipeline, combining supervised fine-tuning and reinforcement learning with a composite reward function, ensures robust and efficient learning for optimal code size reduction.

---

[Generative AI for Biosciences: Emerging Threats and Roadmap to Biosecurity](http://arxiv.org/abs/2510.15975)

- BioSafe (Towards Safe and Secure GenAI in Biosciences): introduces a multi-layered framework for securing GenAI in biosciences, encompassing pre-training, post-training, and inference stages with components like dataset filtering, watermarking, safety alignment, model unlearning, anti-jailbreak screening, red teaming, and inference-time alignment, supported by LLM agents and external tools.
- The framework aims to build a resilient infrastructure by embedding security throughout the GenAI lifecycle, from data curation and model shaping to real-time monitoring and adaptive governance.
- This approach addresses dual-use risks by integrating technical safeguards and promoting interdisciplinary collaboration to manage the transformative potential of GenAI in life sciences while minimizing profound risks.

---

[Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer](http://arxiv.org/abs/2510.03342)

- Gemini Robotics 1.5 (GR 1.5) Agent: introduces a framework combining Gemini Robotics-ER 1.5 (VLM) as an orchestrator and Gemini Robotics 1.5 (VLA) as an action model, along with a Motion Transfer mechanism and Embodied Thinking, to enable general-purpose robots to solve complex, multi-step tasks.
- The framework integrates a novel architecture and Motion Transfer for multi-embodiment learning, interleaves actions with multi-level internal reasoning, and establishes state-of-the-art embodied reasoning for visual and spatial understanding, task planning, and progress estimation.
- This system allows robots to perceive, think, and act, improving task decomposition, execution of complex instructions, interpretability, and recovery behaviors for robust physical agents.

---

#### 12th October 2025

[Generative AI and the Transformation of Software Development Practices](http://arxiv.org/abs/2510.10819)

- Generative AI in Software Development: introduces an evaluation of how generative AI is transforming software development practices, surveying emerging paradigms like Chat-Oriented Programming, Vibe Coding, and Agentic Programming, alongside technical enablers such as LLMs, AI agents, Model Context Protocol, and orchestration frameworks, all interacting within development environments with human oversight.
- The paper details how AI-assisted techniques accelerate productivity and expand accessibility, while also addressing challenges related to trust, accountability, economic costs, and required skill shifts for developers.
- It provides a comprehensive overview of the generational shift in software development, delineating new roles, skills, and best practices for harnessing AI effectively and responsibly.

---

[LLMS AS STRATEGIC AGENTS: BELIEFS, BEST RESPONSE BEHAVIOR, AND EMERGENT HEURISTICS](http://arxiv.org/abs/2510.10813)

- Strategic Thinking Framework: introduces a hybrid method to evaluate LLMs' strategic thinking by disentangling their beliefs, evaluation, and choice mechanisms, applying it across non-cooperative environments, and analyzing reasoning traces and a novel context-free game.
- The research demonstrates that current frontier LLMs exhibit belief-coherent best-response behavior at targeted reasoning depths, self-limit their reasoning depth, and form differentiated conjectures about human and synthetic opponents.
- Under increasing complexity, LLMs transition from explicit recursion to internally generated heuristic rules of choice, revealing emergent meta-reasoning and novel heuristic formation distinct from human biases.

---

[Simpliflow: A Lightweight Open-Source Framework for Rapid Creation and Deployment of Generative Agentic AI Workflows](http://arxiv.org/abs/2510.10675)

- Simpliflow: introduces a lightweight, open-source Python framework for rapid creation and deployment of generative agentic AI workflows, featuring a Client Application, Simpliflow Framework, LLM Integration Layer, LLM Interface, LLM Providers, LiteLLM, Human-in-the-Loop Interface, Function Layer, Agent Class, Agent Instance, Workflow JSONs, Interactions, EnvFile, WebWorkflowCreator, Post Processor, and User.
- The framework enables declarative, JSON-based configuration of linear, deterministic agentic workflows, supporting over 100 LLMs through LiteLLM and allowing dynamic injection of user-defined postprocessor functions for "AI-to-Action" capabilities.
- Its modular architecture decouples agent management, workflow execution, and post-processing, promoting ease of use, extensibility, and transparent, auditable orchestration with human-in-the-loop approvals and structured logging.

---

[BROWSERAGENT: BUILDING WEB AGENTS WITH HUMAN-INSPIRED WEB BROWSING ACTIONS](http://arxiv.org/abs/2510.10666)

- BrowserAgent: introduces an interactive web agent that solves complex tasks through human-inspired browser actions, operating directly on raw web pages via Playwright and employing a two-stage training pipeline of SFT and RFT.
- The framework integrates an explicit memory mechanism to store key conclusions across steps, enhancing reasoning capabilities for long-horizon tasks and achieving competitive results with less training data.
- BrowserAgent defines a minimal yet expressive set of atomic browser operations, including page operations, tab management, URL navigation, and completion actions, to align with real human browsing behavior.

---

[AGENTIQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](http://arxiv.org/abs/2510.10661)

- AGENTIQL (An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation): introduces an agent-inspired multi-expert framework that combines a reasoning agent for question decomposition, a coding agent for sub-query generation, and a refinement step for column selection, with an adaptive router selecting between a modular pipeline and a baseline parser.
- The framework enhances interpretability by exposing intermediate reasoning steps and improves execution accuracy through its specialized components and adaptive routing.
- AGENTIQL achieves high execution accuracy with smaller open-source LLMs, narrowing the performance gap to GPT-4-based state-of-the-art systems.

---

[GraphTracer: Graph-Guided Failure Tracing in LLM Agents for Robust Multi-Turn Deep Search](http://arxiv.org/abs/2510.10581)

- GraphTracer: introduces a framework that redefines failure attribution through information flow analysis, with Framework Establish (defines queries, roles, tools), Trajectory Collection (analyzes execution traces), and Training GraphTracer (trains failure tracer), to address challenges in multi-turn deep search scenarios by explicitly modeling information dependencies.
- The framework constructs Information Dependency Graphs (IDGs) to capture how LLM agents reference and build on prior outputs, localizing root causes by tracing through these dependency structures instead of relying on temporal sequences.
- GraphTracer also employs graph-aware synthetic data generation to target critical nodes and trains a specialized failure tracer using reinforcement learning guided by graph-structural rewards for precise error localization.

---

[AI-Agents for Culturally Diverse Online Higher Education Environments](http://arxiv.org/abs/2510.10520)

- Multi-Modal AI-Agent: introduces a framework for AI-driven education, integrating multiple sensory channels to provide interactive and empathetic learning environments for culturally diverse online higher education, with all its components, where the framework leverages LLMs and various modules, including memory, reasoning, tools, emotion recognition, and physical action, to personalize content delivery and adapt interactions based on student cultural context and learning history.
- This framework supports both virtual and embodied robot tutors, aiming to enhance student engagement, motivation, and learning outcomes through culturally responsive pedagogy and non-verbal communication.
- The paper highlights the importance of memory architecture for personalization, multi-modal processing for empathy, and adaptive non-verbal behaviors to address challenges in diverse online learning environments.

---

[FML-BENCH: A BENCHMARK FOR AUTOMATIC ML RESEARCH AGENTS HIGHLIGHTING THE IMPORTANCE OF EXPLORATION BREADTH](http://arxiv.org/abs/2510.10472)

- FML-BENCH: introduces a benchmark designed to evaluate automatic machine learning research agents on 8 diverse and fundamental ML problems, providing a unified evaluation framework with five complementary metrics, where agents iteratively refine ideas based on experimental results.
- The benchmark emphasizes fundamental problems, utilizes real-world codebases, offers extensibility, and maintains a low coding barrier to focus agents on scientific advancements.
- FML-BENCH's evaluation protocol assesses agent performance across Utility, Diversity, Academic Contribution Rate, Cost, and Step Success Rate, providing comprehensive insights into research competence.

---

[MedCoAct: Confidence-Aware Multi-Agent Collaboration for Complete Clinical Decision](http://arxiv.org/abs/2510.10461)

- MedCoAct (Medical Collaborative Action): introduces a confidence-aware multi-agent framework simulating clinical collaboration for integrated diagnosis and treatment workflows, featuring specialized Doctor and Pharmacist Agents that leverage Query Planning, Query Generation, a Knowledge Retrieval system (including Medical Database, Qwen3-embedding, Qwen3-reranker, and Vector Search Tool), a Reflection Mechanism, and Answer Generation, all coordinated via a Cross-Agent Workflow.
- The framework enhances diagnostic and medication recommendation accuracy by integrating specialized doctor and pharmacist LLM agents and incorporating confidence-aware reflection mechanisms for dynamic quality optimization.
- MedCoAct utilizes a specialized vector retrieval framework for role-aware knowledge acquisition and is evaluated on the new DrugCareQA benchmark for comprehensive assessment of integrated medical decision-making.

---

[Testing and Enhancing Multi-Agent Systems for Robust Code Generation](http://arxiv.org/abs/2510.10460)

- MAS Robustness Repair Method: introduces a novel repairing method for multi-agent systems (MASs) for robust code generation, integrating multi-prompt generation (generates diverse input expressions) and a monitor agent (interprets plans, checks code) with its plan interpretation (provides detailed plan explanations) and code check (validates code against interpreted plan) sub-components, to bridge the planner-coder communication gap.
- This method enhances MAS robustness by diversifying input expressions and improving inter-agent communication, effectively reducing information loss and semantic drift between planning and coding agents.
- Evaluation demonstrates the method's effectiveness in repairing 40.0%-88.9% of identified failures and significantly reducing new failures during fuzzing, particularly for less capable MASs and complex questions.

---

[Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction](http://arxiv.org/abs/2510.10454)

- Traj-CoA (Patient Trajectory Modeling via Chain-of-Agents): introduces a multi-agent system for patient trajectory modeling, including Input (Longitudinal EHR data), Worker Agent (Processes EHR chunks), EHRMem (Long-term memory module), Manager Agent (Synthesizes final prediction), and Output (Final prediction/summary), designed to perform temporal reasoning over long and noisy Electronic Health Records (EHRs) for tasks like lung cancer risk prediction.
- The framework employs a chain of worker agents to sequentially process EHR data in manageable chunks, distilling critical events into a shared long-term memory module (EHRMem) to reduce noise and preserve a comprehensive timeline.
- A manager agent then synthesizes the worker agents' summaries and the extracted timeline from EHRMem to make predictions, demonstrating robust and generalizable temporal reasoning over complex patient trajectories.

---

[CONTROLLABLE GENERATIVE TRAJECTORY PREDICTION VIA WEAK PREFERENCE ALIGNMENT](http://arxiv.org/abs/2510.10731)

- PrefCVAE (Preference CVAE): introduces an augmented CVAE framework for controllable generative trajectory prediction, utilizing weakly labeled preference pairs to imbue latent variables with semantic attributes, enabling semantically meaningful and diverse predictions.
- The framework enforces a semantic latent space by aligning the semantic of two model predictions with labeled preference of their latent generative factors, allowing for predictable and monotonic control over trajectory generation.
- PrefCVAE integrates a preference loss alongside the original CVAE ELBO loss, demonstrating effectiveness in enhancing sampling-based generative models for safer and more informed autonomous driving planning.

---

[Reinforcement Learning-based Dynamic Adaptation for Sampling-Based Motion Planning in Agile Autonomous Driving](http://arxiv.org/abs/2510.10567)

- RL-based Dynamic Adaptation Framework: introduces a novel hybrid planning architecture that integrates a high-level RL agent with a low-level sampling-based trajectory planner, including a Sampling-based Planner, PPO Agent, Encoders, Flatten Layer, Concatenate Features, MLP Actor Network, MLP Critic Network, Rollout Buffer, Environment, Ego and opponent information, Actions a_t, Trajectories, and Reward r_t, to dynamically adapt cost function parameters for agile autonomous driving.
- The framework enables interactive maneuvers by allowing the PPO Agent to dynamically switch between predefined behavioral modes, such as Nominal Racing, Aggressive, and Close Driving, based on the current racing scenario.
- This approach resolves the trade-off between safety and competitiveness in autonomous racing by ensuring trajectory validity while significantly outperforming static planners in challenging multi-vehicle scenarios.

---

[Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](http://arxiv.org/abs/2510.11754)

- LLM-based Agentic Workflow: introduces an LLM agent that interacts with a clinical Treatment Planning System (TPS) to iteratively refine optimization objectives for Intensity-Modulated Radiation Therapy (IMRT) planning, leveraging Clinical Goals, Optimization Priors, an Arithmetic Tool, an Observation Module, an Analysis Module, an Update Module, and Chain-of-Thought Reasoning to achieve high-quality treatment plans in a zero-shot setting.
- This workflow automates inverse treatment planning by enabling the LLM agent to extract intermediate plan states, analyze them using arithmetic and trend-based reasoning, and dynamically propose updated constraint values, mimicking human planner decision-making.
- The approach demonstrates feasibility and comparable dosimetric performance to manual plans for head-and-neck cancer cases, reducing planning variability and supporting AI-based planning strategies without prior training data.

---

[From Craft to Constitution: A Governance-First Paradigm for Principled Agent Engineering](http://arxiv.org/abs/2510.13857)

- ArbiterOS (Principled Agent Engineering): introduces a governance-first paradigm for reliable AI agent engineering, combining a Mental Model (The Agentic Computer) to understand probabilistic hardware, a Formal Architecture (Neural-Symbolic OS) to enforce safety, and a Rigorous Discipline (Evaluation-Driven Development Lifecycle) for continuous verification.
- This framework transforms agent development from a brittle craft into a principled engineering discipline by providing architectural enforcement mechanisms for reliability, auditability, and security.
- ArbiterOS addresses the "crisis of craft" in LLM-based agents by managing their inherent uncertainty through a neuro-symbolic architecture and a systematic development lifecycle.

---

[Agentic RAG for Software Testing with Hybrid Vector-Graph and Multi-Agent Orchestration](http://arxiv.org/abs/2510.10824)

- Agentic RAG framework: introduces an approach to software testing automation, combining autonomous AI agents with hybrid vector-graph knowledge systems, multi-agent orchestration, and enhanced contextualization to automate test plan, case, and Quality Engineering (QE) metric generation.
- The framework leverages LLMs like Gemini and Mistral, a Multi-Layer Prompt Architecture, and a Comprehensive Traceability Framework to achieve high accuracy and document traceability in enterprise software testing.
- Experimental validation demonstrates significant improvements in accuracy, an 85% reduction in testing timeline, and projected 35% cost savings, accelerating go-live by two months.

---


#### 11th October 2025

[Is Misinformation More Open? A Study of robots.txt Gatekeeping on the Web](http://arxiv.org/abs/2510.10315)

- Robots Exclusion Protocol (REP): introduces a study investigating how reputable news websites and misinformation sites configure their `robots.txt` files, particularly concerning AI crawlers, using website lists, AI user agents list, HTTP requests/crawling, Internet Archive data, and active blocking mechanisms.
- The research reveals a significant disparity, with 60.0% of reputable sites disallowing at least one AI crawler compared to 9.1% of misinformation sites, and reputable sites restricting an average of 15.5 AI user agents versus fewer than one for misinformation sites.
- Longitudinal analysis further shows that AI-blocking by reputable sites increased from 23% in September 2023 to nearly 60% by May 2025, while misinformation sites remained largely passive, highlighting a growing asymmetry in content accessibility for LLM training data.

---

[Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in Large Language Models](http://arxiv.org/abs/2510.10278)

- VivaBench: introduces a multi-turn benchmark for evaluating sequential clinical reasoning in LLM agents, including a Clinical Case (structured clinical vignette), an Agent (LLM under evaluation), an Examiner module (processes queries, retrieves data), a Mapper module (translates queries to structured keys), and a Parser module (formats retrieved information).
- The framework simulates viva voce examinations, where the Agent interacts with structured Clinical Cases through Review and Investigation phases to gather information and arrive at a diagnosis, supported by components like History, Physical Examination, Imaging, Laboratory investigations, Diagnosis set, and Differential diagnoses.
- VivaBench provides a standardized, open-source benchmark to assess LLMs' ability to navigate diagnostic uncertainty and synthesize information sequentially, identifying critical failure modes in clinical reasoning.

---

[ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement](http://arxiv.org/abs/2510.10241)

- ImCoref-CeS (Improved Coreference Resolution with Checker-Splitter): introduces a novel framework for coreference resolution, integrating an enhanced supervised model (ImCoref) with an LLM-based Checker-Splitter agent to refine outputs.
- ImCoref enhances long-text encoding with a Lightweight Bridging Module, improves mention detection via a Biaffine-Augmented Scorer, and boosts training efficiency with Hybrid Mention Regularization.
- The LLM Checker-Splitter acts as a multi-role agent, validating candidate mentions and splitting erroneous coreference clusters, guided by Mention and Coreference Cluster Filters to balance performance and resource cost.

---

[ISAAC: Intelligent, Scalable, Agile, and Accelerated CPU Verification via LLM-aided FPGA Parallelism](http://arxiv.org/abs/2510.10225)

- ISAAC (Intelligent, Scalable, Agile, and Accelerated CPU Verification via LLM-aided FPGA Parallelism): introduces a full-stack CPU verification framework that integrates intelligence-driven stimulus generation with a high-throughput differential testing infrastructure, including an LLM-aided multi-agent stimulus engine, ISS, RTL co-simulation, checker, micro-arch. info, FPGA parallelism infrastructure, lightweight forward-snapshot mechanism, and decoupled co-simulation architecture.
- The framework's front-end leverages LLMs and historical bug patterns to generate targeted, high-value tests, accelerating coverage convergence and corner-case exploration.
- Its back-end employs FPGA parallelism and a decoupled ISS-DUT execution model to drive multiple DUTs concurrently, significantly improving simulation throughput and eliminating long-tail test bottlenecks.

---

[Don't Just Fine-tune the Agent, Tune the Environment](http://arxiv.org/abs/2510.10197)

- ENVIRONMENT TUNING introduces a novel training paradigm for LLM agents, orchestrating learning through a Structured Curriculum (guides skill acquisition from simple to complex tasks), Actionable Environment Augmentation (provides corrective hints upon failure), and Fine-Grained Progress Rewards (measures task completion with dense feedback).
- This framework enables agents to learn complex behaviors directly from problem instances without relying on pre-collected expert trajectories, addressing data scarcity and improving generalization.
- By transforming ambiguous errors into actionable lessons and providing continuous progress signals, the framework ensures stable and efficient exploration for multi-turn tool-use tasks.

---

[MedAgentAudit: Diagnosing and Quantifying Collaborative Failure Modes in Medical Multi-Agent Systems](http://arxiv.org/abs/2510.10185)

- MedAgentAudit (AuditTrail framework): introduces a comprehensive empirical investigation and quantitative auditing framework to diagnose and quantify collaborative failure modes in medical multi-agent LLM systems, revealing architectural weaknesses beyond final-answer accuracy.
- The framework systematically analyzes 3,600 interaction logs across six multi-agent systems and medical datasets, identifying a taxonomy of collaborative failures and success modes.
- Key findings include persistent information loss, suppression of minority opinions, reliance on voting over evidence-based reasoning, and a chronic inability to prioritize high-risk clinical outcomes, highlighting the need for transparent and auditable AI in medicine.

---

[Proof Strategy Extraction from LLMs for Enhancing Symbolic Provers](http://arxiv.org/abs/2510.10131)

- STRAT2ROCQ introduces a framework that extracts LLM proof strategies as formalized lemmas in Rocq, which are then used to enhance symbolic provers like CoqHammer.
- The framework operates by prompting an LLM to generate natural language proofs for theorems in a training set, then formalizing individual proof steps into reusable lemmas, and finally verifying these lemmas with a proof agent.
- By integrating these LLM-extracted lemmas, the framework significantly improves CoqHammer's success rate in proving theorems and automating tactics, demonstrating the value of leveraging LLM internal reasoning for symbolic verification.

---

[IntrinTrans: LLM-based Intrinsic Code Translator for RISC-V Vector](http://arxiv.org/abs/2510.10119)

- IntrinTrans (LLM-based Intrinsic Code Translator for RISC-V Vector): introduces a novel LLM-based multi-agent framework that translates intrinsic code across architectures, utilizing a Code Translator, Compilation Executor, Test Executor, and Code Optimizer, orchestrated by a finite state machine with continuous testing and feedback.
- The framework automatically translates Arm Neon intrinsics to RISC-V Vector intrinsics, verifies correctness through iterative compile-and-test cycles, and optimizes performance using register usage information from liveness analysis.
- IntrinTrans demonstrates the feasibility of employing LLMs for automated cross-ISA code migration, generating semantically correct and performance-efficient RVV code, and in some cases achieving significant speedups over native implementations.

---

[Agentic Troubleshooting Guide Automation for Incident Management](http://arxiv.org/abs/2510.10074)

- StepFly: introduces a novel end-to-end agentic framework for troubleshooting guide automation, with TSG Mentor, Guidelines, LLMs, Execution DAG, Query Preparation Plugins, Scheduler, Executor, Memory System, Plugins, SRE, and Incident components, designed to automate the execution of troubleshooting guides in large-scale IT systems.
- The framework features a three-stage workflow including offline preprocessing to extract structured execution DAGs and Query Preparation Plugins, and online execution using a DAG-guided scheduler-executor architecture with a memory system.
- StepFly achieves a high success rate and significantly reduces execution time and token consumption, especially for parallelizable troubleshooting guides, by leveraging LLMs for preprocessing and a multi-agent system for execution.

---

[ALLOY: Generating Reusable Agent Workflows from User Demonstration](http://arxiv.org/abs/2510.10049)

- ALLOY (Agentic Logic Learned from Observing You): introduces a system that transforms user demonstrations into editable and reusable LLM workflows, enabling users to generate, adapt, and generalize LLM-based agent workflows through a multi-agent system generation and generalization pipeline.
- The system captures user demonstrations in a browser extension, infers procedural knowledge, and visualizes it as a graph-structured workflow of LLM-powered sub-task agents, which can be directly edited and executed.
- ALLOY facilitates workflow reuse and generalization to new tasks via natural language prompts, significantly reducing effort for structurally similar tasks while maintaining alignment with user-preferred execution strategies.

---

[SwarmSys: Decentralized Swarm-Inspired Agents for Scalable and Adaptive Reasoning](http://arxiv.org/abs/2510.10047)

- SwarmSys (Decentralized Swarm-Inspired Agents for Scalable and Adaptive Reasoning): introduces a closed-loop, distributed multi-agent reasoning framework that enables LLM agents to coordinate through iterative interactions among Explorer, Worker, and Validator roles, supported by adaptive agent and event profiles, embedding-based matching, and a pheromone-inspired reinforcement mechanism.
- This framework fosters self-organized collaboration and dynamic task allocation, allowing for scalable and adaptive problem-solving without centralized control, and converges to high-quality solutions through continuous exploration-exploitation-validation cycles.
- SwarmSys demonstrates emergent collective intelligence, outperforming baselines in symbolic reasoning, research synthesis, and scientific programming tasks, suggesting that coordination scaling can rival model scaling in advancing LLM intelligence.

---

[Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning](http://arxiv.org/abs/2510.10009)

- ExpandSearch: introduces a reinforcement learning framework that trains an LLM-based search agent for query expansion and selective information distillation.
- The framework employs an expand-then-squeeze strategy, where the LLM-based search agent generates multiple query variants and a pre-trained squeezer model distills retrieved content.
- This dual strategy addresses semantic incompleteness and information overload, significantly improving performance on multi-hop QA benchmarks.

---

[Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey](http://arxiv.org/abs/2510.09988)

- Unified Framework for LLM Reasoning: introduces a survey that deconstructs search algorithms into its core components: Search Mechanism (explores reasoning paths), Reward Formulation (defines search guidance/learning target), and Transition Function (models state changes).
- This framework establishes a formal distinction between transient Search Guidance for Test-Time Scaling (TTS) and durable Parametric Reward Modeling for Self-Improvement, addressing the ambiguous role of reward signals in LLM reasoning.
- The survey synthesizes state-of-the-art methods and proposes a component-centric taxonomy to chart a research roadmap for creating autonomous, self-improving LLM agents.

---

[Knowledge Graph-Enhanced Multi-Agent Infrastructure for coupling physical and digital robotic environments(KG-MAS)](http://arxiv.org/abs/2510.10325)

- KG-MAS (Knowledge Graph-Enhanced Multi-Agent Infrastructure): introduces a robust, scalable, and flexible solution for coupling heterogeneous physical and digital robotic environments, leveraging a centralized Knowledge Graph (dynamic, shared world model), a Multi-Agent System (autonomous agents), Hypermedea (multi-agent programming environment), Hypermedea Artefact (agent interaction interface), Connection Component (command translator, information perceiver), Physical environment (physical robotic platforms), Digital environment (digital robotic platforms), Agent Creator (generates autonomous agents), Coordination Protocol (defines agent communication), System Setup KG (initial configuration storage), and System Data KG (real-time operational state storage).
- The infrastructure features a model-driven architecture that facilitates the automatic generation of agents from semantic descriptions, simplifying system extension and maintenance.
- By abstracting communication protocols and providing a unified, intelligent coordination mechanism, KG-MAS addresses challenges of system heterogeneity and complexity in Cyber-Physical Systems.

---

[Beyond ADE and FDE: A Comprehensive Evaluation Framework for Safety-Critical Prediction in Multi-Agent Autonomous Driving Scenarios](http://arxiv.org/abs/2510.10086)

- The three-layer safety evaluation framework introduces a novel testing framework that evaluates prediction performance under diverse scene structures, including map context, agent density, and spatial distribution, to identify safety-critical scenarios. 
- The framework's Filter Framework systematically breaks down complex driving environments into detailed classifications across its Layer 1 (Map Filter) and Layer 2 (Agent Filter, Road Filter) components, enabling comprehensive robustness evaluation beyond traditional single-condition testing. 
- The Evaluation Module utilizes metrics like MIE_A and MIE_F to quantify map dependency and identify scenario-specific failure cases not exposed by conventional ADE and FDE, ultimately certifying models as either Unvalidated or Validated Safety-critical models.

---

[Read the Room or Lead the Room: Understanding Socio-Cognitive Dynamics in Human-AI Teaming](http://arxiv.org/abs/2510.09944)

- HAT Experimental Study: introduces an investigation into socio-cognitive dynamics in human-AI teaming, utilizing the TRAIL platform, an AI Teammate (GPT-4 agent with custom memory), Human Participants, Linguistic Inquiry and Word Count (LIWC), Group Communication Analysis (GCA), an Experimental Design, and a Group Task to analyze communication patterns and roles.
- The study specifically examines how an autonomous GPT-4 LLM agent, designed with social, cognitive, and affective capabilities, influences collaborative problem-solving dynamics and how human collaborators adapt their roles.
- By analyzing discourse data using LIWC and GCA, the research provides insights into the AI's tendency to act as a dominant cognitive facilitator while being socially detached, and humans' shift towards more socially oriented roles.

---

[Scheming Ability in LLM-to-LLM Strategic Interactions](http://arxiv.org/abs/2510.12826)

- LLM-to-LLM Scheming Evaluation Framework: introduces a systematic evaluation of LLM agents' scheming ability and propensity in strategic interactions, utilizing Cheap Talk signaling and Peer Evaluation adversarial games, with analysis of Chain-of-Thought reasoning and observed scheming tactics.
- The framework reveals that frontier LLMs exhibit high scheming success rates when prompted and a significant propensity for deception even without explicit instructions, particularly in adversarial settings.
- Analysis of scheming tactics demonstrates that LLMs deploy both basic goal concealment and advanced strategies like trust exploitation and self-preservation, highlighting the need for robust multi-agent AI safety evaluations.

---

[SecureWebArena: A Holistic Security Evaluation Benchmark for LVLM-based Web Agents](http://arxiv.org/abs/2510.10073)

- SecureWebArena: introduces a holistic security evaluation benchmark for LVLM-based web agents, integrating Attack Vectors (manipulate agent's decision-making), Agent (LVLM-based web agent under evaluation), Simulated Web Environment (realistic web environments for interaction), and a Multi-Layer Evaluation Protocol (analyzes agent failures) with LLM-as-a-Judge (automates internal reasoning assessment) and Human Expert (manually analyzes behavior and outcome) components.
- The benchmark features six simulated web environments and a structured taxonomy of six attack vectors, spanning both user-level and environment-level manipulations, to comprehensively assess agent vulnerabilities.
- Its multi-layered evaluation protocol analyzes agent failures across internal reasoning, behavioral trajectory, and task outcome, providing a fine-grained risk analysis beyond simple success metrics.

---

#### 10th October 2025

[Agentic Property-Based Testing: Finding Bugs Across the Python Ecosystem](http://arxiv.org/abs/2510.09907)

- Agentic Property-Based Testing (PBT): introduces an LLM-based agent that autonomously analyzes Python modules, infers properties, synthesizes and executes PBTs, reflects on test outputs, and generates actionable bug reports.
- The agent, built on Anthropic's Claude Code, systematically crawls codebases, identifies high-value properties, and uses Hypothesis PBTs to find genuine bugs.
- This approach demonstrates a scalable method for autonomously testing software, successfully identifying diverse bugs in popular Python packages like NumPy.

---

[Autonomous Agents for Scientific Discovery: Orchestrating Scientists, Language, Code, and Physics](http://arxiv.org/abs/2510.09901)

- LLM-based Scientific Agent Framework: introduces an LLM-based scientific agent that orchestrates interactions with human scientists, natural language, computer code, and physics, enabling autonomous scientific discovery through iterative phases of hypothesis discovery, experimental design and execution, and result analysis and refinement.
- The framework leverages LLMs' reasoning and planning capabilities to automate scientific discovery, addressing challenges from hypothesis generation to experimental execution and data interpretation.
- It emphasizes a continuous refinement loop, incorporating automatic self-correction, external evaluation, and human-in-the-loop feedback to ensure robust, generalizable, and adaptive scientific agents.

---

[How can we assess human-agent interactions? Case studies in software agent design](http://arxiv.org/abs/2510.09801)

- PULSE (Prediction-powered User Label Synthesis and Evaluation): introduces a three-step framework for efficient human-centric evaluation of LLM agent designs, which includes collecting user feedback, training an ML model to predict user satisfaction, and computing effect sizes by combining human ratings with model-generated pseudo-labels.
- The framework is deployed on a large-scale web platform using the OpenHands software agent, gathering in-the-wild usage data from over 15k users to study how LLM backbone, planning strategy, and memory mechanisms impact developer satisfaction.
- PULSE provides practical insights for software agent design by revealing discrepancies between in-the-wild user satisfaction and benchmark performance, and it reduces confidence intervals by 40% compared to standard A/B tests.

---

[Building a Foundational Guardrail for General Agentic Systems via Synthetic Data](http://arxiv.org/abs/2510.09781)

- Safiron (Foundational Guardrail): introduces a pre-execution guardrail for LLM agents, addressing data, evaluation, and model gaps, utilizing AuraGen for scalable synthetic risk data, an Adapter for input normalization, and Pre-Exec Bench for plan-level safety evaluation.
- The guardrail intervenes at the planning stage to proactively analyze agent plans, detect harmful actions, assign risk types, and generate rationales before execution, preventing severe consequences.
- AuraGen's synthetic data generation, combined with Safiron's robust training and the Pre-Exec Bench, provides a practical and scalable template for safer agentic systems.

---

[Vision Language Models: A Survey of 26K Papers (CVPR, ICLR, NeurIPS 2023-2025)](http://arxiv.org/abs/2510.09586)

- VLM Research Trend Analysis: introduces a transparent, reproducible measurement of research trends across 26,104 accepted papers from CVPR, ICLR, and NeurIPS spanning 2023-2025, with all its components, where it quantifies three macro shifts in multimodal vision-language-LLM work, generative methods, and 3D/video activity.
- The analysis reveals a sharp rise of multimodal vision-language-LLM work, steady expansion of generative methods, and resilient 3D and video activity, alongside specific architectural and training shifts within VLMs.
- The survey highlights a pivot towards instruction-following and multi-step reasoning, parameter-efficient adaptation, and the increasing integration of vision and language components through various bridging mechanisms.

---

[JUDGE'S VERDICT: A COMPREHENSIVE ANALYSIS OF LLM JUDGE CAPABILITY THROUGH HUMAN AGREEMENT](http://arxiv.org/abs/2510.09738)

- Judge's Verdict Benchmark: introduces a novel two-step methodology to evaluate LLMs as judges for response accuracy, including a correlation test and a Cohen's Kappa analysis with human-likeness assessment, classifying LLM judges into human-like or super-consistent tiers.
- The framework assesses 54 LLMs' ability to replicate human judgment when scoring responses from RAG or Agentic pipelines against ground truth answers, moving beyond correlation to measure actual agreement patterns.
- This methodology provides a standardized benchmark for classifying LLM judges into distinct performance tiers, revealing that judge excellence depends on training strategies rather than solely model size.

---

[AutoPR: Let's Automate Your Academic Promotion!](http://arxiv.org/abs/2510.09558)

- PRAgent (Automatic Promotion Agent): introduces a three-stage multi-agent framework for automating academic promotion, including content extraction, collaborative synthesis, and platform-specific adaptation, to transform research papers into engaging, platform-tailored social media posts.
- The framework leverages specialized agents like the Textual Content Extraction Agent and Visual Content Preparation Agent for initial data processing, followed by a collaborative synthesis stage with Logical Draft, Visual Analysis, Textual Enriching, and Visual-Text-Interleaved Combination Agents.
- The final stage, managed by an Orchestration Agent, focuses on platform-specific adaptation and packaging to optimize content for various social media channels, ensuring maximum reach and engagement.

---

[StatEval: A Comprehensive Benchmark for Large Language Models in Statistics](http://arxiv.org/abs/2510.09517)

- StatEval: introduces a comprehensive benchmark for evaluating LLMs on statistical reasoning, encompassing foundational knowledge and research-level proof tasks, built using a scalable multi-agent pipeline with human-in-the-loop validation, and assessed via a robust evaluation framework.
- The benchmark includes 13,817 undergraduate/graduate problems and 2,374 journal-sourced proof tasks, structured by difficulty and over 30 subdomains for fine-grained analysis of statistical reasoning abilities.
- Experimental results reveal that state-of-the-art LLMs, including closed-source models, achieve below 57% on research-level problems, particularly struggling with machine learning tasks, underscoring the inherent difficulty of statistical reasoning.

---

[MULTIMODAL POLICY INTERNALIZATION FOR CONVERSATIONAL AGENTS](http://arxiv.org/abs/2510.09474)

- TriMPI (Three-stage Multimodal Policy Internalization): introduces a novel three-stage training framework for Multimodal Policy Internalization (MPI), including VM-CPT (injects policy knowledge), CoT SFT (reasons over policy rules), RL (learns policy-compliant behavior), and PolicyRollout (augments RL exploration), to enhance policy-following in multimodal conversational agents.
- The framework aims to internalize complex, reasoning-intensive multimodal policies into a large multimodal model's parameters, eliminating the need for in-context policy inclusion during inference and improving efficiency.
- TriMPI also introduces two new datasets, ClevrPolicy and GTAPolicy, to support training and evaluation across diverse multimodal policy types, demonstrating significant improvements in end-to-end performance and generalization.

---

[ADAPTIVE ATTACKS ON TRUSTED MONITORS SUBVERT AI CONTROL PROTOCOLS](http://arxiv.org/abs/2510.09462)

- Adaptive Attacks on AI Control Protocols: introduces a study on adaptive attacks where untrusted LLM agents, knowing the control protocol and monitor model, subvert AI control protocols by embedding prompt injections into their outputs, evading diverse monitors and completing malicious tasks.
- The research demonstrates that these prompt injections consistently evade existing LLM-based monitors, causing safety-usefulness Pareto frontiers of control protocols to collapse to upfront auditing levels.
- A key finding reveals that the Defer-to-Resample protocol, intended to mitigate weak monitors, paradoxically amplifies prompt injection attacks by effectively converting them into best-of-n attacks, reducing safety.

---

[NL2GenSym: Natural Language to Generative Symbolic Rules for SOAR Cognitive Architecture via Large Language Models](http://arxiv.org/abs/2510.09355)

- NL2GenSym (Natural Language to Generative Symbolic Rules): introduces a novel framework that integrates LLMs with SOAR to autonomously produce generative symbolic rules from natural language, utilizing a Self-Evolving Domain Knowledge Base, an Execution-Grounded Generator-Critic mechanism, SOAR Cognitive Architecture, LLMs, and Retrieval-Augmented Generation.
- The framework employs a closed-loop process where the LLM-based Generator proposes rules, which are executed in SOAR, and an LLM-based Critic refines them based on execution-grounded feedback and a self-evolving knowledge base.
- This approach significantly lowers the barrier to SOAR utilization by automating rule generation and optimization, enabling the discovery of novel, high-efficiency heuristic rules, and demonstrating that well-designed architectures can outperform sheer model scale.

---

[Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers](http://arxiv.org/abs/2510.09330)

- Safety Game: introduces a model-independent, black-box framework for LLM safety alignment, leveraging a two-player zero-sum game and an LP solver to balance helpfulness and safety in responses, without requiring retraining or internal model access.
- The framework operationalizes LLM agents to compute minimax equilibrium strategies at inference time, using external probes to estimate helpfulness and safety risks for a finite set of candidate responses.
- This approach offers a scalable and accessible pathway for stakeholders to enforce safety across LLM ecosystems by dynamically adjusting responses to achieve equilibrium behavior under a defined risk cap.

---

[Fundamentals of Building Autonomous LLM Agents](http://arxiv.org/abs/2510.09244)

- Autonomous LLM Agent Architecture: introduces a review of agents powered by LLMs, detailing their core capabilities including a Perception System (captures/processes environmental data), Reasoning System (formulates plans/adapts to feedback), Memory System (retains knowledge/experiences), and Execution System (translates decisions into actions) interacting with an Environment (external world/simulated world).
- The paper explores how integrating these systems enables more capable and generalized software bots that mimic human cognitive processes for autonomous and intelligent behavior.
- It systematically reviews design options, integration strategies, and generalization capabilities for LLM-based agents, addressing limitations of traditional LLMs in real-world tasks.

---

[Student Development Agent: Risk-free Simulation for Evaluating AIED Innovations](http://arxiv.org/abs/2510.09183)

- Student Development Agent Framework: introduces a student development agent framework based on LLMs, integrating key components like Learning Environment (E), Endowment Dimensions (W), Developmental Dimensions (D), Actions (A), Learning Behaviors (B), and History (H), along with modules for categorization, empirical findings acquisition, prompt construction, and iterative simulation, to model dynamic student developmental trajectories.
- The framework leverages LLMs to generate student changes by combining empirical findings from real-world data with generative capabilities, enabling prospective evaluation of novel instructional applications efficiently and ethically.
- This approach provides a risk-free simulation environment for AIED innovations, allowing assessment of potential benefits and harms before exposure to real students, thus safeguarding student well-being and accelerating research.

---

[AGENTIC-KGR: CO-EVOLUTIONARY KNOWLEDGE GRAPH CONSTRUCTION THROUGH MULTI-AGENT REINFORCEMENT LEARNING](http://arxiv.org/abs/2510.09156)

- Agentic-KGR (Co-Evolutionary Knowledge Graph Construction Through Multi-Agent Reinforcement Learning): introduces a novel framework enabling co-evolution between LLMs and KGs through multi-round reinforcement learning, featuring dynamic schema expansion, a retrieval-augmented memory system, and learnable multi-scale prompt compression.
- The framework integrates a comprehensive tool pool for knowledge graph operations with a dual reward mechanism, allowing dynamic KG construction and expansion while simultaneously improving reasoning capabilities.
- Agentic-KGR demonstrates superior performance in knowledge extraction and downstream QA tasks by synergistically optimizing knowledge structures and agent reasoning through iterative interactions.

---

[Exploiting Web Search Tools of AI Agents for Data Exfiltration](http://arxiv.org/abs/2510.09093)

- Indirect Prompt Injection Attack Scenario: introduces a system demonstrating how an AI agent, equipped with web search and internal knowledge base access, can be exploited by an attacker via a malicious website to exfiltrate sensitive information to a log server.
- The scenario highlights the vulnerability of LLM-driven workflows to indirect prompt injection attacks, where hidden instructions in external data sources manipulate the agent's behavior.
- This research evaluates various LLM models' susceptibility to such attacks and different prompt manipulation techniques, emphasizing the need for robust security safeguards.

---

[LEADING THE FOLLOWER: LEARNING PERSUASIVE AGENTS IN SOCIAL DEDUCTION GAMES](http://arxiv.org/abs/2510.09087)

- Leading the Follower Framework: introduces a reinforcement learning approach that trains agents to optimize utterances for persuasive impact in social deduction games (SDGs), formalizing turn-based dialogue as a Stackelberg competition.
- This framework includes a Backend (API-based LLM) for base utterance generation, a Refiner (open-source LLM) for enhancing persuasive impact, and a Measurer (frozen open-source LLM) for computing rewards based on follower response probabilities.
- The framework's three key steps—Intent Identification, Impact Measurement, and Strategy Optimization—utilize GRPO to refine utterances, enabling agents to proactively steer conversations towards desired outcomes in complex social interactions.

---

[A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System](http://arxiv.org/abs/2510.09721)

- LLM-Empowered Software Engineering Pipeline and Taxonomy: introduces a comprehensive survey analyzing 150+ papers, organizing them into a taxonomy of solutions (Prompt-Based, Fine-Tune-Based, Agent-Based) and benchmarks (Code Generation, Translation, Repair, Others), and presenting a unified pipeline from task specification to final deliverables.
- The survey details how LLM-empowered software engineering has evolved from simple prompt engineering to complex agentic systems incorporating planning, reasoning, self-refinement, memory, and tool augmentation.
- This framework bridges the gap between evaluation methodologies and solution approaches, providing a systematic understanding of LLM-driven software engineering and identifying future research directions.

---

[Preference-Aware Memory Update for Long-Term LLM Agents](http://arxiv.org/abs/2510.09720)

- PAMU (Preference-Aware Memory Update Mechanism): introduces a mechanism that enables LLMs to perceive, adapt to, and respond in alignment with evolving user preferences, including User Dialogue History, LLM, Preference Extractor (Tone Style Classifier, Response Length Calculator, Emotional Tone Analyzer, Information Density Extractor, Formality Detector), SW & EMA Algorithm (Sliding Window Average, Exponential Moving Average, Fusion Mechanism, Change Detection Signal), Preference Vector (Wt), Preference-Guided Prompting, and Prompt Injection, where it dynamically refines preference memory representations in response to evolving user behaviors and contexts.
- The mechanism constructs a fused preference-aware representation by combining short-term fluctuations via sliding window averages and long-term user tendencies via exponential moving averages.
- This approach allows for interpretable and controllable adaptation to preference drift, significantly improving LLM output quality in long-term conversations without architectural modification or fine-tuning.

---

[MEC3O: Multi-Expert Consensus for Code Time Complexity Prediction](http://arxiv.org/abs/2510.09049)

- MEC3O (Multi-Expert Consensus for Code Time Complexity Prediction): introduces a multi-expert consensus system for code time complexity prediction, which includes LLMs, expertise dataset sampling, expert selection, class-specific instructions, class experts, initial prediction generation, opinion exchange, prediction revision, a weighted consensus function (WECC), and final prediction.
- The framework assigns LLMs as specialized experts for different time complexity classes, enabling them to engage in structured debates where they can revise predictions based on peer opinions.
- This approach mitigates "Degeneration-of-Thought" and reliance on a separate judge model by leveraging class-specific expertise and a weighted consensus mechanism for robust and accurate predictions.

---

[REFGRADER: AUTOMATED GRADING OF MATHEMATICAL COMPETITION PROOFS USING AGENTIC WORKFLOWS](http://arxiv.org/abs/2510.09021)

- Ref-Grader (Automated Grading of Mathematical Competition Proofs using Agentic Workflows): introduces an agentic workflow for automated grading of mathematical competition proofs, including Reference Solution Clustering, Solution Matching, Solution Analysis, Rubric Design, and Grading components.
- This framework addresses the challenge of assigning fair partial credit by extracting and analyzing reference solutions to automatically derive problem-specific rubrics for a multi-step grading process.
- The proposed workflows achieve higher agreement with human grades and more consistent handling of partial credit compared to single-turn LLM grading.

---

[MASA: LLM-Driven Multi-Agent Systems for Autoformalization](http://arxiv.org/abs/2510.08988)

- MASA (LLM-Driven Multi-Agent Systems for Autoformalization): introduces a modular framework for building multi-agent systems for autoformalization, leveraging collaborative agents, LLMs, knowledge bases, retrievers, and theorem provers to convert natural language statements into formal representations.
- The framework emphasizes modularity, flexibility, and extensibility, allowing seamless integration of new agents and tools to adapt to the evolving field of autoformalization.
- MASA's architecture supports an iterative self-refinement process where agents provide critiques and refine formalizations based on feedback from theorem provers and LLM-as-a-judge components.

---

[When LLM Agents Meet Graph Optimization: An Automated Data Quality Improvement Approach](http://arxiv.org/abs/2510.08952)

- LAGA (Large Language and Graph Agent): introduces an automated multi-agent framework for Text-Attributed Graph (TAG) quality improvement, integrating a Detection Agent (identifies graph quality issues), Planning Agent (generates adaptive repair plans), Action Agent (executes optimization schemes), and Evaluation Agent (assesses improved graph quality), all powered by LLMs.
- The Action Agent, central to LAGA, employs a dual-encoder (semantic encoder and structure encoder) and optimizes three modality-specific objectives (text, structure, label) to capture complementary information and enhance graph quality.
- LAGA addresses diverse and systematic TAG quality issues across text, structure, and label modalities, providing a data-centric solution for robust and generalizable graph learning.

---

[The Idola Tribus of AI: Large Language Models tend to perceive order where none exists](http://arxiv.org/abs/2510.09709)

- Idola Tribus Evaluation Methodology: introduces an experimental setup to investigate the tendency of LLMs to over-recognize patterns in number series, utilizing target LLMs, a number series generator, a regularity identification prompt, an LLM-as-a-judge evaluator, an evaluation prompt, and defined evaluation criteria.
- The methodology assesses LLMs' pattern recognition and abstraction capabilities across various integer sequences, including arithmetic, geometric, difference, quasi-ordered, random-increasing, and purely random series.
- The study reveals that LLMs frequently perceive non-existent patterns in random series, a behavior analogous to Francis Bacon's "Idola Tribus," highlighting limitations in their logical reasoning for tasks requiring accurate hypothesis formation.

---

[SOP-Maze: Evaluating Large Language Models on Complicated Business Standard Operating Procedures](http://arxiv.org/abs/2510.08942)

- SOP-Maze: introduces a benchmark for evaluating LLMs on complex business Standard Operating Procedures (SOPs), comprising Lateral Root System (LRS) and Heart Root System (HRS) task categories, defined by Objective, Standard Operating Procedures, User Input, and Output Requirement components, and assessed via JSON Schema based Evaluation.
- The benchmark includes 397 tasks across 23 real-world business scenarios, designed to challenge LLMs on both breadth (LRS) and depth (HRS) of complex instruction following.
- Experiments with 18 LLMs reveal significant performance gaps, identifying three core failure modes: route blindness, conversational fragility, and calculation errors, highlighting the challenges of real-world business SOPs.

---

[StreamingVLM: Real-Time Understanding for Infinite Video Streams](http://arxiv.org/abs/2510.09608)

- StreamingVLM: introduces a unified framework for real-time, stable understanding of infinite video streams, incorporating a compact KV Cache, Attention Sink, Long Text Window, Short Vision Window, Contiguous ROPE, and an Overlapped-chunk Full-Attention Strategy.
- The framework aligns training with streaming inference by applying full attention on short, overlapped video chunks, effectively mimicking the inference-time attention pattern without training on prohibitively long contexts.
- This design enables coherent commentary, real-time generation, and long-term memory retention, addressing challenges of latency and memory in processing infinite visual input.

---

[Zero-shot Structure Learning and Planning for Autonomous Robot Navigation using Active Inference](http://arxiv.org/abs/2510.09574)

- AIMAPP (Active Inference MAPping and Planning): introduces a biologically inspired, Active Inference-based framework for autonomous robot navigation that unifies mapping, localization, and decision-making within a single generative model, continuously adapting its beliefs from sensorimotor feedback.
- The framework employs a generative model formalized as a partially observable Markov decision process (POMDP) and uses Monte Carlo Tree Search (MCTS) to plan actions by minimizing Expected Free Energy, balancing exploration and goal-directed behaviors.
- AIMAPP operates in a zero-shot, self-supervised, online-learning fashion, requiring no pre-training and demonstrating robust performance in large-scale real and simulated environments against state-of-the-art planning models.

---

[Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid Execution Strategy](http://arxiv.org/abs/2510.09469)

- Alert-X (Collision-Aware Dynamic Alert Mask and Hybrid Execution Strategy): introduces a hybrid framework for scalable multi-agent pathfinding, integrating decentralized path planning (S1) with an RL policy πθ and a multi-channel observation space including an AlertMask, centralized collision detection and control (S2-S3) via a central module, and decentralized replanning (S4) by alerted agents using the RL policy πθ.
- The framework strategically reduces inter-agent information sharing by using targeted alerts from a central coordinator to prompt localized re-planning, rather than continuous global observation.
- This approach consistently finds feasible, collision-free solutions even in large-scale scenarios with high agent counts, demonstrating robust generalization from simpler training.

---

[Clear Roads, Clear Vision: Advancements in Multi-Weather Restoration for Smart Transportation](http://arxiv.org/abs/2510.09228)

- Synthetic Image Generation Pipeline: introduces a method for creating realistic hazy, rainy, and snowy scenes by combining a clear scene with atmospheric light and weather-specific maps.
- This pipeline utilizes a transmission map for depth-dependent haze, rain-streak overlays for rain, and snow-particle maps for snow.
- The generated synthetic datasets are crucial for developing and evaluating multi-weather restoration models due to the scarcity of real-world degraded data.

---

[Robust Driving Control for Autonomous Vehicles: An Intelligent General-sum Constrained Adversarial Reinforcement Learning Approach](http://arxiv.org/abs/2510.09041)

- IGCARL (Intelligent General-sum Constrained Adversarial Reinforcement Learning): introduces a novel robust autonomous driving approach, with a strategic targeted adversary (generates multi-step adversarial attacks), a DRL-based design (for temporal decision-making), a general-sum objective (induces safety-critical events), a perturbation generation (PG) method (creates adversarial perturbations), a robust driving agent (learns robust policy), constrained policy optimization (ensures stable learning), a collision risk constraint (limits high-risk actions), and a policy consistency constraint (mitigates policy drift), where the paper addresses challenges in DRL-based autonomous driving by enhancing robustness against strategic adversarial attacks and ensuring stable learning.
- The strategic targeted adversary uses DRL and a general-sum objective to generate coordinated multi-step attacks that specifically induce safety-critical events, moving beyond myopic, zero-sum approaches.
- The robust driving agent is trained with constrained policy optimization, incorporating collision risk and policy consistency constraints to prevent overfitting and policy drift, thereby ensuring reliable performance in both adversarial and clean environments.

---

[Beyond hospital reach: Autonomous lightweight ultrasound robot for liver sonography](http://arxiv.org/abs/2510.08106)

- Autonomous Lightweight Ultrasound Robot System: introduces an autonomous lightweight abdominal-mounted ultrasound robot, integrating an AI agent with multi-modal perception and memory attention, and a 588-gram 6-degrees-of-freedom cable-driven robot for expert-level liver sonography.
- The system autonomously acquires expert-level standard liver ultrasound planes and detects pathology in patients, demonstrating robust performance on rapid-motion individuals and in wilderness environments.
- This work represents the first demonstration of autonomous sonography across multiple challenging scenarios, potentially transforming access to expert-level diagnostics in underserved regions.

---

[ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](http://arxiv.org/abs/2510.15949)

- ATLAS (Adaptive Trading with LLM AgentS): introduces a unified multi-agent framework for financial decision-making, integrating structured information from markets, news, and corporate fundamentals, with a Central Trading Agent that generates executable orders, and Adaptive-OPRO for dynamic prompt optimization.
- The framework leverages specialized LLM-based analysts within its Market Intelligence Pipeline to synthesize diverse data streams, feeding into the Central Trading Agent's order-aware decision layer.
- Adaptive-OPRO, a novel prompt optimization technique, dynamically adapts the Central Trading Agent's instructions based on real-time, stochastic market feedback, leading to improved performance over time.

---

[WARC-Bench: Web Archive Based Benchmark for GUI Subtask Executions](http://arxiv.org/abs/2510.09872)

- WARC-Bench (Web ARChive Benchmark): introduces a novel web navigation benchmark for GUI subtask executions, featuring Web ARChive (WARC) files, realistic Web Environments, GUI Subtasks, and a programmatic Evaluator, designed to assess multimodal AI agents like the Subtask Vision Agent (SVA).
- The benchmark enables sandboxed interactions with dynamic webpages using WARC files, posing a significant challenge for leading computer-use models in mastering short-horizon UI component interactions.
- The SVA, a VLM-based agent, utilizes screenshots, action spaces, and Chain-of-Thought reasoning to predict actions, with performance significantly improved through supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR).

---

[AUTO-SCALING CONTINUOUS MEMORY FOR GUI AGENT](http://arxiv.org/abs/2510.09038)

- Auto-scaling Continuous Memory (CoMEM): introduces a memory-augmented VLM agent framework that tackles long-horizon generalization by encoding prior GUI trajectories into compact continuous embeddings and scaling this memory through an autonomous data flywheel, which includes new environment discovery, task synthesis, trajectory rollout, and quality checking components.
- The framework leverages a VLM as an encoder to compress GUI trajectories into fixed-length continuous embeddings, which are then directly injected into the VLM's input layer to reduce context cost and preserve visual information.
- The auto-scaling data flywheel autonomously collects over 100k diverse GUI trajectories at low cost, enabling efficient fine-tuning of the memory encoder (LoRA on Q-Former) with minimal parameters, leading to improved success rates on real-world GUI benchmarks.

---

#### 9th October 2025

[Rapid Development of Omics Data Analysis Applications through Vibe Coding](http://arxiv.org/abs/2510.09804)

- Vibe Coding: introduces a process where LLMs and autonomous coding agents generate, test, and refine executable code from natural language prompts, enabling rapid development of data analysis applications.
- The framework leverages Replit.com as a development environment and builds Streamlit-based web applications, exemplified by a Proteomics Data Analysis Platform with modules for data upload, processing, statistical analysis, and visualizations.
- This approach significantly reduces the technical barrier and cost for domain experts to prototype sophisticated analytical tools, transforming computational biology software development.

---

[WHAT IS YOUR AGENT'S GPA? A FRAMEWORK FOR EVALUATING AGENT GOAL-PLAN-ACTION ALIGNMENT](http://arxiv.org/abs/2510.08847)

- Agent GPA (Goal-Plan-Action) framework: introduces an evaluation paradigm for LLM agents, structured around the operational loop of setting goals, devising plans, and executing actions, including Goal, Plan, Action, LLM Judges, Goal Fulfillment Judge, Logical Consistency Judge, Execution Efficiency Judge, Plan Quality Judge, Plan Adherence Judge, Tool Selection Judge, Tool Calling Judge, Manager Agent, and Search Agent, to systematically evaluate agent performance.
- The framework employs specialized LLM judges for each metric, providing a systematic way to detect, organize, and localize a broad range of agent failures, demonstrating strong agreement with human judgments and consistency across evaluations.
- This approach offers actionable feedback by pinpointing errors to specific dimensions, enabling targeted debugging and iterative improvement of agent performance beyond mere outcome-based evaluations.

---

[CommandSans: SECURING AI AGENTS WITH SURGICAL PRECISION PROMPT SANITIZATION](http://arxiv.org/abs/2510.08829)

- CommandSans (SECURE AI AGENTS WITH SURGICAL PRECISION PROMPT SANITIZATION): introduces a novel token-level sanitization process for AI agents, which surgically removes AI-directed instructions from tool outputs using a BERT-based classifier trained with LLM-labeled instruction-tuning and synthetic data, allowing agents to proceed safely.
- This non-blocking approach significantly reduces attack success rates for indirect prompt injections across various benchmarks without impairing agent utility, addressing limitations of traditional sample-level blocking defenses.
- The framework's design prioritizes low latency and high precision, enabling practical deployment by avoiding the need for specialized prompt injection training data and context-dependent calibration.

---

[SEARCH-ON-GRAPH: ITERATIVE INFORMED NAVIGATION FOR LARGE LANGUAGE MODEL REASONING ON KNOWLEDGE GRAPHS](http://arxiv.org/abs/2510.08825)

- SoG (Search-on-Graph): introduces an iterative informed graph navigation framework for LLM reasoning on knowledge graphs, utilizing a single `SEARCH` function, a dynamic filtering mechanism, and few-shot exemplars to enable efficient and accurate multi-hop question answering.
- The framework operates on an "observe-then-navigate" principle, where the LLM systematically examines actual available relational connections at each entity before making informed navigational decisions, avoiding blind path planning or semantic similarity heuristics.
- SoG's simple, plug-and-play design adapts seamlessly to diverse KG schemas and handles high-degree nodes through adaptive filtering, achieving state-of-the-art performance across multiple KGQA benchmarks without fine-tuning.

---

[MOSAIC: Multi-agent Orchestration for Task-Intelligent Scientific Coding](http://arxiv.org/abs/2510.08804)

- MOSAIC (Multi-agent Orchestration for Task-Intelligent Scientific Coding): introduces a training-free, LLM-agnostic multi-agent framework for scientific code generation, including a Bucketing Module (routes problems to domain), a Teacher Module (guides student module) with a Code Rationale Builder (creates detailed rationales) and a Self-Reflection Agent (analyzes, refines pseudocode logic), and a Student Module (generates, refines code) with a Rationale Agent (produces step-by-step reasoning plan), a Consolidated Context Window (CCW) (maintains context, mitigates hallucinations), a Coding Agent (generates code block), and a Debugger Agent (executes, corrects code errors), designed to solve challenging scientific coding tasks without I/O test cases.
- The framework operates in a student-teacher paradigm, where the Teacher Module uses ground-truth data for few-shot prompting to guide the Student Module in generating accurate and executable code, facilitating stepwise problem decomposition and targeted error correction.
- MOSAIC's specialized agents collaboratively decompose problems, self-reflect on algorithms, generate and refine code, and maintain context across chained subproblems, outperforming existing approaches in accuracy, robustness, and interpretability on scientific coding benchmarks.

---

[COMPASS: Enhancing Agent Long-Horizon Reasoning with Evolving Context](http://arxiv.org/abs/2510.08790)

- COMPASS (Context-Organized Multi-Agent Planning and Strategy System): introduces a hierarchical framework for enhancing agent long-horizon reasoning, featuring a User Query, Context Manager (managing Notes), Meta-Thinker, Main Agent (executing tasks via Tool Use), and an Answer Synthesizer for the Final Answer.
- The framework separates tactical execution (Main Agent) from strategic oversight (Meta-Thinker) and context organization (Context Manager) to address challenges like context overflow, hallucination, and loss of coherence in LLM agents.
- COMPASS improves accuracy by up to 20% on challenging benchmarks like GAIA, BrowseComp, and Humanity's Last Exam, demonstrating effectiveness in error-prone long-horizon settings.

---

[Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations](http://arxiv.org/abs/2510.08779)

- LLM-Guided RL Training: introduces a framework that integrates LLM planning guidance into RL training through enhanced observations, allowing RL agents to learn when to follow or ignore LLM suggestions, thereby creating soft constraints.
- The framework leverages LLMs' world knowledge and reasoning abilities to provide action recommendations as additional observational input, improving exploration in sparse-reward environments.
- This approach demonstrates significant improvements in learning speed and final success rates, especially in complex tasks, without requiring modifications to existing RL algorithms.

---

[BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation](http://arxiv.org/abs/2510.08572)

- BLAZER: introduces a framework that bootstraps LLM-based manipulation agents using automatically generated and verified demonstrations, enabling self-improvement of zero-shot manipulation agents with its LLMboot, Manipulation Environment, Verification Module, Data Generation Module, Task Database (DBLAZER), Supervised Finetuning Module, BLAZER LLM, Vision Pipeline, Robotic Gripper, Robot Arm, Motion Planner, Objects, and Cameras, where the paper describes a method for finetuning standard LLMs to obtain specialized agents for robotic manipulation.
- The framework leverages an LLMboot to generate initial manipulation plans in a simulated environment, where successful executions are automatically verified and collected into a task database.
- This curated dataset is then used for supervised finetuning of a smaller BLAZER LLM, which significantly improves performance and generalizes to new tasks in both simulated and real-world settings, supported by a vision pipeline for real-world deployment.

---

[COMAS: CO-EVOLVING MULTI-AGENT SYSTEMS VIA INTERACTION REWARDS](http://arxiv.org/abs/2510.08529)

- CoMAS (Co-Evolving Multi-Agent Systems): introduces autonomously improving LLM-agent framework with Interaction rewards obtained from collaborative & critical reasoning within deceontralized MAS, where Interaction consists of Question/Solution Proposals/Evaluations/Scoring.
- The framework demonstrates consistent SOTA-level performance by using three stage workflow: 1. interaction, 2. reward formulation with LLM-as-a-Judge, 3. policy optimization with REINFORCE++ / replay buffer of the agent consisting of context/generated output/assigned reward.
- CoMAS achieve self-evolution without external supervision. Increasing the number & diversity of the agents scale up the framework performance, which indicate emergence of collective intelligence.
- The framework generates intrinsic rewards from rich discussion dynamics, where agents collaboratively propose solutions, critically evaluate them, and assign scores, mimicking human learning through mutual discussion.

---

[MoA-VR: A Mixture-of-Agents System Towards All-in-One Video Restoration](http://arxiv.org/abs/2510.08508)

- MoA-VR (Mixture-of-Agents Video Restoration): introduces a multi-agent system for all-in-one video restoration, comprising a Degradation Identification Agent (identifies degradation types and severity), a Routing and Restoration Agent (formulates restoration sequences, applies tools), and a Quality Assessment Agent (estimates visual quality) within a closed-loop architecture.
- This framework mimics human expert reasoning by dynamically identifying complex video degradations, adaptively planning restoration workflows using LLMs, and iteratively refining strategies based on VLM-driven quality feedback.
- MoA-VR leverages multimodal intelligence and modular reasoning to effectively handle diverse and compound degradations, outperforming existing baselines in objective metrics and perceptual quality for general-purpose video restoration.

---

[OPPONENT SHAPING IN LLM AGENTS](http://arxiv.org/abs/2510.08255)

- ShapeLLM (Opponent Shaping Large Language Model): introduces a model-free opponent shaping algorithm for transformer-based LLM agents, leveraging structured natural language prompts to condense history and context, enabling LLM agents to influence co-players' learning dynamics in diverse game-theoretic environments.
- The framework utilizes a gemma-2-2b-it base model, fine-tuned with QLORA and PEFT for parameter efficiency, and trained using a custom PPO implementation from the TRL package, where context is represented by cumulative state visitation counts.
- ShapeLLM demonstrates that LLM agents can successfully guide opponents toward exploitable equilibria in competitive games and promote coordination in cooperative games, highlighting the importance of understanding multi-agent dynamics in LLM research.

---

[DODO: Causal Structure Learning with Budgeted Interventions](http://arxiv.org/abs/2510.08207)

- DODO: introduces a novel algorithmic framework for an autonomous Agent to infer the underlying causal structure of its environment, represented as a Directed Acyclic Graph (DAG), through its observation, intervention, causal links detection, and indirect causal connections pruning phases.
- The framework iteratively selects and applies interventions, updating its estimate of the underlying causal graph by leveraging a lightweight heuristic to guide the intervention process, balancing exploration of uncertain edges with exploitation of established structures.
- DODO demonstrates superior causal discovery performance in well-resourced regimes, achieving high F1 scores and low Structural Hamming Distances, especially when the intervention budget is sufficient for robust pruning.

---

[QUANTUM AGENTS FOR ALGORITHMIC DISCOVERY](http://arxiv.org/abs/2510.08159)

- Quantum Intelligent Agents: introduces a framework for quantum agents trained by episodic, reward-based reinforcement learning to autonomously rediscover quantum algorithms and protocols, including agents (learning entities), environment (provides inputs, computes rewards), private registers (agent A's local qubits), private registers (agent B's/environment's local qubits), message registers (shared communication qubits), initial state preparation (environment sets up qubits), unitary policies (agent actions, parameterized circuits), parameterized quantum circuits (learnable gate sequences), episodic reinforcement learning (training mechanism), reward function (guides policy optimization), and measurement outcomes (classical results for reward).
- This framework enables agents to learn optimal strategies for tasks like Quantum Fourier Transform, Grover's search, strong quantum coin flipping, and CHSH games, directly from interaction without prior knowledge of optimal solutions.
- The learned policies are implemented as parameterized quantum circuits, constrained by nearest-neighbor connectivity and shallow depth, demonstrating the potential for automated design of novel quantum algorithms and protocols.

---

[AutoQual: An LLM Agent for Automated Discovery of Interpretable Features for Review Quality Assessment](http://arxiv.org/abs/2510.08081)

- AutoQual: introduces an LLM-based agent framework for automated interpretable feature discovery, with Hypothesis Generation, Tool Implementation, Feature Search, and Dual-Level Memory Architecture components, designed to transform tacit knowledge into explicit, computable, and interpretable features for review quality assessment.
- The framework mimics a human research workflow, iteratively generating feature hypotheses, operationalizing them via autonomous tool implementation, and accumulating experience in a persistent memory system.
- AutoQual demonstrates real-world effectiveness in a large-scale online platform, improving average reviews viewed per user by 0.79% and conversion rate by 0.27%, showcasing its generalizability across diverse text assessment tasks.

---

[Learning on the Job: An Experience-Driven, Self-Evolving Agent for Long-Horizon Tasks](http://arxiv.org/abs/2510.08002)

- MUSE (Memory-Utilizing and Self-Evolving): introduces an experience-driven, self-evolving system centered around a hierarchical Memory Module, Planning-Execution Agent, and Reflect Agent, which enables continuous learning and self-evolution for long-horizon tasks.
- The framework operates on a "Plan-Execute-Reflect-Memorize" iterative loop, where the agent autonomously reflects on its trajectory and converts raw actions into structured experience for integration into the Memory Module.
- MUSE achieves new SOTA performance on the long-horizon productivity benchmark TAC, demonstrating superior task completion capabilities and strong generalization properties through accumulated experience.

---

[ReInAgent: A Context-Aware GUI Agent Enabling Human-in-the-Loop Mobile Task Navigation](http://arxiv.org/abs/2510.07988)

- ReInAgent: introduces a context-aware multi-agent framework for human-in-the-loop mobile task navigation, integrating an Information-managing Agent (manages information, interacts user), a Decision-making Agent (plans, decides, operates mobile), and a Reflecting Agent (reflects, validates, summarizes history) with a Memory Module (shared information storage) to resolve information dilemmas and enable dynamic task evolution.
- The framework addresses ambiguous initial instructions, incremental information supplementation, and conflicting information through a dynamic task-slot management mechanism and proactive user-agent interaction.
- ReInAgent achieves higher success rates and better alignment with user preferences on complex mobile tasks by enabling adaptive and reliable task navigation in real-world scenarios.

---

[Network Topology and Information Efficiency of Multi-Agent Systems: Study based on MARL](http://arxiv.org/abs/2510.07888)

- CTDE (Centralized Training with Decentralized Execution) framework: introduces a MARL approach with communications, exploring how network topology and information efficiency impact multi-agent coordination.
- The framework utilizes components like Observation Encoder, Hidden States, and Policy Network, and introduces metrics such as Information Entropy Efficiency Index (IEI) and Specialization Efficiency Index (SEI) to optimize communication.
- It demonstrates that directed and sequential communication topologies, specifically DAGs, improve performance and reduce communication overhead, while integrating IEI and SEI into training accelerates convergence and enhances coordination.

---

[Team Xiaomi EV-AD VLA: Learning to Navigate Socially Through Proactive Risk Perception - Technical Report for IROS 2025 RoboSense Challenge Social Navigation Track](http://arxiv.org/abs/2510.07871)

- Falcon with PRPM: introduces a social navigation system that augments the Falcon framework with a Proactive Risk Perception Module, which predicts distance-based collision risk scores for surrounding humans to enhance spatial awareness and proactive collision avoidance.
- The system processes egocentric RGB-D observations and odometry, utilizing a main policy network for navigation actions and auxiliary tasks for population estimation, position estimation, and trajectory forecasting.
- This approach, achieving 2nd place in the IROS 2025 RoboSense Challenge, improves personal space compliance and goal navigation in crowded indoor environments by providing dense supervisory signals for anticipatory collision avoidance.

---

[EFFECTIVE AND STEALTHY ONE-SHOT JAILBREAKS ON DEPLOYED MOBILE VISION-LANGUAGE AGENTS](http://arxiv.org/abs/2510.07809)

- Stealthy One-Shot Jailbreak Framework: introduces a practical and stealthy one-shot jailbreak attack that leverages in-app prompt injections, with low-privilege perception-chain targeting, stealthy user-invisible activation, and one-shot prompt efficacy, to corrupt an agent's perception and exfiltrate private user data.
- The framework embeds short malicious prompts in UI text that remain inert during human interaction but are revealed when an agent drives the UI via ADB, bypassing on-device safety filters and requiring no elevated permissions.
- The attack achieves high planning and execution hijack rates across multiple LVLM backends and Android applications, exposing a fundamental security vulnerability in current mobile agents.

---

[MULTIMODAL SAFETY EVALUATION IN GENERATIVE AGENT SOCIAL SIMULATIONS](http://arxiv.org/abs/2510.07709)

- Simulation Framework: introduces a reproducible platform for evaluating multimodal situational safety in generative agent environments, with Generative Agents, Perception, Memory Stream, Planning, Reflection, Execution, Plan Revision Layer, Judge Agent, Social Activity Scenarios, Fixed Virtual Environment, Interaction Network, Information Spread, Conversion Rate, and Acceptance Ratios, where agents perceive, plan, interact, and adapt over time, undergoing periodic plan revisions for safety evaluation.
- The framework enables MLLM-based agents to detect unsafe situations, reason about them, and revise their plans in a dynamic environment, supported by an external Judge Agent for safety verification.
- This approach allows for the study of how unsafe actions are detected, revised, and propagated through social interactions and evolving memories within agent societies.

---


#### 8th October 2025

[HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving](http://arxiv.org/abs/2510.07210)

- HyPlan (Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving): introduces a novel hybrid learning-assisted planning method for collision-free navigation, integrating multi-agent behavior prediction, ego-car path planning, explicit online POMDP planning, and a deep reinforcement learner with confidence-based vertical pruning.
- The framework leverages AutoBots for behavior prediction, an Anytime Weighted Hybrid A* for path planning, and IS-DESPOT for velocity action planning, guided by a PPO-based deep reinforcement learner (NavPPO).
- HyPlan employs confidence calibration via CRUDE and confidence-based vertical pruning to reduce planning execution time while maintaining driving safety in partially observable traffic environments.

---

[Falsification-Driven Reinforcement Learning for Maritime Motion Planning](http://arxiv.org/abs/2510.06970)

- FDRL (Falsification-Driven Reinforcement Learning): introduces a falsification-driven RL approach that generates adversarial training scenarios using CMA-ES to improve rule compliance of an RL agent in maritime motion planning, integrating these scenarios into the RL training process.
- The approach leverages counterexamples identified by falsification to iteratively refine the RL policy's behavior, promoting adherence to complex Signal Temporal Logic (STL) specifications for maritime traffic rules.
- Experiments demonstrate that incorporating falsification leads to more relevant training scenarios, resulting in improved and more consistent rule compliance for autonomous vessels in open-sea navigation.

---

[DECOMPGAIL: LEARNING REALISTIC TRAFFIC BEHAVIORS WITH DECOMPOSED MULTI-AGENT GENERATIVE ADVERSARIAL IMITATION LEARNING](http://arxiv.org/abs/2510.06913)

- DecompGAIL (Decomposed Multi-agent Generative Adversarial Imitation Learning): introduces a framework for realistic multi-agent traffic simulation by explicitly decomposing realism into ego-map and ego-neighbor components, filtering out misleading neighbor-neighbor and neighbor-map interactions, and augmenting ego rewards with distance-weighted neighborhood rewards via a social PPO objective.
- The framework utilizes a Map Encoder to extract map features, a Policy Network to predict motion-token distributions, and a Decomposed Discriminator to separately assess scene and interaction realism.
- DecompGAIL improves training stability and achieves state-of-the-art realism on the WOMD Sim Agents 2025 benchmark by addressing the "irrelevant interaction misguidance" problem in multi-agent GAIL.

---

[When Machines Meet Each Other: Network Effects and the Strategic Role of History in Multi-Agent AI Systems](http://arxiv.org/abs/2510.06903)

- Experimental Framework: introduces a study on LLM agents in a canonical network-effect game, with LLM Agents (autonomous decision-makers), an Environment (central coordinator, broadcasts information), a Network-Effect Game (simulated economic interaction), System Evolution (manages game rounds), a Decision-Making Process (agent's internal steps) including Information Gathering (collects current price, past outcomes, parameters), Participant Expectation (forecasts total participants), and Utility Calculation & Final Decision (determines agent's action), a History Window (memory length for past outcomes), Price Trajectories (sequences of prices over rounds), Network Effect Strength (β) (parameter influencing payoffs), Fulfilled Expectation Equilibrium (FEE) (theoretical benchmark), Root Mean Squared Error (RMSE) (metric for deviation from FEE), and OLS Regression Models (statistical analysis of deviations), to investigate how LLM agents behave in interdependent environments and diverge from economic predictions.
- The research reveals that LLM agents systematically deviate from the Fulfilled Expectation Equilibrium, underestimating participation at low prices and overestimating at high prices, with stronger network effects exacerbating these divergences.
- History plays a critical role, with monotonic histories stabilizing coordination and reducing expectation dispersion, while non-monotonic histories amplify divergence and path dependence, highlighting that LLM agents' behavior is shaped by external incentives, internal heterogeneity, and historical context.

---

[Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain](http://arxiv.org/abs/2510.07309)

- CORGI (Atomized Multi-Agent Evaluation Framework): introduces a new text-to-SQL benchmark for the business domain, featuring a Database Population Process that synthesizes realistic business data and an Atomized Multi-Agent Evaluation Framework for assessing LLM performance on complex business queries, including Input, Discriminator Agent, Scoring Agents, and Final Score components.
- The benchmark's database population process leverages real-world business scenarios, expert input, and LLMs to create schemas and data simulation rules, which then guide the generation of synthetic databases.
- The multi-agent evaluation framework employs a discriminator agent to select relevant scoring metrics and seven specialized scoring agents to provide comprehensive, context-aware assessment of LLM-generated answers across dimensions like Structure, SQL SER, Data Sense, Insightfulness, Operational Implementability, Purpose Alignment, and Compliance.

---

[MLE-Smith: SCALING MLE TASKS WITH AUTO-MATED MULTI-AGENT PIPELINE](http://arxiv.org/abs/2510.07307)

- MLE-Smith: introduces a fully automated multi-agent pipeline for scaling Machine Learning Engineering (MLE) tasks, which includes a Brainstormer (enumerates task formulations), Designer (instantiates MLE tasks), Refactor (standardizes task designs), Toolset (agent capabilities), Hybrid Verification Mechanism (ensures task quality), Assertions (enforces structural constraints), LLM Review (semantic validation), Test Agent (conducts execution-based validation), and MLE Env (simulates MLE environment).
- The framework transforms raw datasets into competition-style MLE challenges using a generate-verify-execute paradigm, ensuring verifiable quality, real-world usability, and rich diversity.
- This principled pipeline enforces structural integrity, semantic soundness, and empirical solvability through its multi-agent generation workflow, robust hybrid verification, and interactive execution-based validation loop.

---

[LAD-RAG: Layout-aware Dynamic RAG for Visually-Rich Document Understanding](http://arxiv.org/abs/2510.07233)

- LAD-RAG (Layout-aware Dynamic RAG): introduces a novel framework for visually-rich document understanding that constructs a symbolic document graph and a neural index during ingestion, enabling an LLM agent to dynamically retrieve evidence using semantic and graph-based tools.
- This approach addresses limitations of conventional RAG by capturing layout structure and cross-page dependencies, integrating symbolic and neural signals, and leveraging an LLM agent for dynamic, query-adaptive retrieval beyond static top-k methods.
- The framework consistently improves retrieval completeness and QA accuracy on multi-page reasoning tasks by providing a holistic, contextualized understanding of document content with minimal inference latency.

---

[Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping](http://arxiv.org/abs/2510.07230)

- CUSTOMER-R1 (Reinforcement Learning-based method for personalized, step-wise user behavior simulation in online shopping environments): introduces a framework that simulates personalized user behavior by conditioning an LLM agent's policy on explicit persona information and optimizing next-step rationale and action generation via action correctness reward signals.
- The framework processes HTML observations, behavior history, and user persona to predict rationales and next actions, which are then evaluated against ground-truth actions using a tailored reward function for policy optimization.
- This approach leverages reinforcement learning to achieve higher fidelity in personalized behavior simulation, outperforming prompting and SFT-based baselines in next-action prediction tasks.

---

[Exposing LLM User Privacy via Traffic Fingerprint Analysis: A Study of Privacy Risks in LLM Agent Interactions](http://arxiv.org/abs/2510.07176)

- AGENTPRINT: introduces a framework to uncover private user information by eavesdropping and analyzing traffic generated during interactions with LLM-based AI agents, with all its components, where it demonstrates that interactive behaviors of LLM agents leave distinctive fingerprints in encrypted traffic, enabling adversaries to infer agent activities, distinguish specific agents, and profile sensitive user attributes.
- The framework leverages a CNN-based model to classify agent behaviors and identities from these traffic fingerprints, and then employs an agent-user attribute correlation matrix to infer sensitive user-level information like occupational roles from aggregated agent usage patterns.
- This research highlights an overlooked privacy risk where the operational characteristics that empower LLM agents simultaneously introduce novel network-level side-channel vulnerabilities, challenging the trust in encryption for user-agent communications.

---

[NurseLLM: The First Specialized Language Model for Nursing](http://arxiv.org/abs/2510.07173)

- NurseLLM: introduces a specialized LLM for nursing question-answering, developed with a multi-stage data generation pipeline (gathering nursing concepts, creating synthetic QA, generated dataset, developing evaluation datasets, filtering data for uniqueness, finetuning the LLM, Llama3-Med42-8B, merging finetuned LLM with base model), to address the unique needs of the nursing domain.
- The framework creates a large-scale NCLEX-equivalent nursing MCQ dataset and three distinct benchmarks for rigorous evaluation of LLMs on nursing QA.
- NurseLLM significantly outperforms general-purpose and medical-specialized LLMs on nursing benchmarks, highlighting the importance of domain specialization and the potential of multi-agent collaboration.

---

[NEWTONBENCH: BENCHMARKING GENERALIZABLE SCIENTIFIC LAW DISCOVERY IN LLM AGENTS](http://arxiv.org/abs/2510.07172)

- NEWTONBENCH: introduces a scientific law discovery benchmark designed to resolve the methodological trilemma of scientific relevance, scalability, and memorization resistance, elevating evaluation from static function fitting to interactive model discovery.
- This benchmark comprises 324 scientific law discovery tasks across 12 physics domains, generated using "metaphysical shifts" to systematically alter canonical laws, ensuring novelty and scientific relevance.
- It features an interactive, system-oriented environment where LLM agents actively design experiments and interpret feedback, with optional code assistance to offload computational tasks, revealing true discovery capabilities.

---

[A MULTI-AGENT FRAMEWORK FOR STATEFUL INFERENCE-TIME SEARCH](http://arxiv.org/abs/2510.07147)

- Stateful Multi-Agent Evolutionary Search: introduces a training-free framework for automated unit test generation, combining persistent inference-time state, adversarial mutation, and evolutionary preservation, utilizing a Controller, Actor, Adversary, Critic, Executor, and LLMs.
- The framework orchestrates these agents to sequentially propose, mutate, and score candidate edge cases, maintaining persistent state across generations to ensure diversity and exploration.
- This approach enables the system to dynamically adapt to unseen codebases, produce robust edge cases, and achieve higher coverage without gradient-based training or domain-specific fine-tuning.

---

[THE COGNITIVE BANDWIDTH BOTTLENECK: SHIFTING LONG-HORIZON AGENT FROM PLANNING WITH ACTIONS TO PLANNING WITH SCHEMAS](http://arxiv.org/abs/2510.07091)

- Cognitive Bandwidth Perspective: introduces a conceptual framework to analyze how LLM agents distribute cognitive load across distinct stages of two planning paradigms, Planning with Actions (PwA) and Planning with Schemas (PwS), for long-horizon tasks.
- The paper systematically compares PwA, which uses explicit action lists, and PwS, which instantiates abstract action schemas, across environments of varying action space complexity to identify a representation-choice inflection point.
- The framework reveals that PwA incurs high Environment Understanding (EU) load with large action spaces, while PwS shifts the burden to Schema Instantiation (SI), offering better scalability beyond the inflection point.

---

[PROMPT OPTIMIZATION ACROSS MULTIPLE AGENTS FOR REPRESENTING DIVERSE HUMAN POPULATIONS](http://arxiv.org/abs/2510.07064)

- POMA (Prompt Optimization Across Multiple Agents): introduces a novel framework for constructing a set of LLM agents that collectively represent diverse human populations by leveraging submodular optimization to select agents based on human demonstrations.
- The framework includes Human Population, Tasks, Demonstrations, LLM Agents, Representative Agents, Behavioral Embeddings, Distance Metric, Representation Gap, Submodular Optimization, REPPOPdemo, REPPOPmapped-1, REPPOPmapped-2, and Prompt Templates, enabling the selection of agents that mimic human behaviors and perspectives.
- This approach addresses the homogeneity issue of single LLMs by creating an ensemble of diverse agents, demonstrating superior performance in representing human populations across educational, crowdsourcing, and annotation tasks.

---

[COMPASS: A MULTI-TURN BENCHMARK FOR TOOL-MEDIATED PLANNING & PREFERENCE OPTIMIZATION](http://arxiv.org/abs/2510.07043)

- COMPASS (Constrained Optimization through Multi-turn Planning and Strategic Solutions): introduces a benchmark for evaluating LLM agents on realistic travel planning scenarios, including an LLM-based user simulator (simulates multi-turn user interactions), a constrained preference optimization problem (defines travel planning problem), realistic travel databases (provides real-world travel data), a comprehensive tool ecosystem (offers booking platform tools), and LLM agents (perform planning and optimization).
- The benchmark casts travel planning as a constrained preference optimization problem, requiring agents to satisfy hard constraints while simultaneously optimizing soft user preferences through multi-turn interactions and strategic tool orchestration.
- COMPASS aims to bridge theoretical LLM advances with real-world impact by directly measuring an agent's ability to optimize user preferences in practical tasks, revealing gaps in current agentic capabilities like acceptable-optimal and plan-coordination.

---

[LLM-Assisted Modeling of Semantic Web-Enabled Multi-Agents Systems with AJAN](http://arxiv.org/abs/2510.06911)

- AJAN-Editor (LLM-Assisted Modeling of Semantic Web-Enabled Multi-Agents Systems with AJAN): introduces an integrated development environment to model, execute, and debug Semantic Web-enabled agents, leveraging LLMs for natural language interaction, including Orchestrator, Parser, Linker, Disambiguator, Elastic Search, Word Dictionary, ASR, TTS, Chat Interface, Query Generator, Autocorrector, Answer Generator, BTF Builder, SBT Generator, SBT Node Factory, Embedding Generator, Vector Store, AJAN Documentation, Triple Store, AGENT, Github, GPT 3.5, GPT 4, and RDF4J, enabling users to engineer multi-agent systems and behaviors using natural language input.
- The framework addresses the complexity of defining RDF/RDFS and SPARQL-based agent behaviors by providing a user-friendly, web-based graphical editor that integrates LLMs for intuitive agent modeling and interaction in dynamic environments.
- It supports various workflows, including SPARQL query generation, Behavior Tree generation, and semantic search over documentation, facilitating both offline development and online agent interaction through text and voice modalities.

---

[Prototyping Multimodal GenAI Real-Time Agents with Counterfactual Replays and Hybrid Wizard-of-Oz](http://arxiv.org/abs/2510.06872)

- The Counterfactual Replay Prompt Evaluation Toolkit: introduces an open-source system for prototyping multimodal GenAI real-time agents, featuring User Session Video and Transcript, a System Prompt Editor, Message Generation Controls, a Generated Message Display, and an Evaluation Interface, to facilitate iterative refinement of agent behaviors.
- This toolkit supports Counterfactual Video Replay Prompting by replaying user session videos for prompt strategy testing and integrates with Hybrid Wizard-of-Oz methods for live user evaluation.
- The approach provides experiential insights into LLM behavior, enabling iterative prompt decomposition and refinement for context-aware multimodal agents.

---

[SID: MULTI-LLM DEBATE DRIVEN BY SELF SIGNALS](http://arxiv.org/abs/2510.06843)

- SID (Self-Signals Driven Multi-LLM Debate): introduces a multi-LLM debate framework that leverages internal self-signals from LLM generation, including LLM Agents, a Model Confidence Module, an Early-Exit Mechanism, a Token-level Semantic Focus Module, a Compression Mechanism, and a Multi-LLM Debate Process, to enhance both performance and efficiency.
- The framework utilizes model-level confidence to enable early exits for confident agents and token-level semantic focus to compress debate content, thereby reducing redundant computation and improving debate quality.
- This approach dynamically adapts the debate trajectory based on the LLMs' own epistemic signals, outperforming existing multi-agent debate methods in accuracy and token consumption across diverse benchmarks.

---

[FURINA: A FULLY CUSTOMIZABLE ROLE-PLAYING BENCHMARK VIA SCALABLE MULTI-AGENT COLLABORATION PIPELINE](http://arxiv.org/abs/2510.06800)

- FURINA-Builder: introduces a multi-agent collaboration pipeline for automatically constructing customizable role-playing benchmarks, including a character-scene pool, simulation, and selection mechanism.
- The framework utilizes LLMs as a director model to manage dialogue flow, source and base models to generate candidate responses, and a judge model to select the superior output based on specific evaluation dimensions.
- This pipeline enables the creation of FURINA-Bench, a comprehensive benchmark for evaluating LLM role-playing capabilities across diverse characters and scenarios with fine-grained criteria.

---

[GPT-5 Model Corrected GPT-4V's Chart Reading Errors, Not Prompting](http://arxiv.org/abs/2510.06782)

- Evaluation Methodology: introduces a quantitative evaluation comparing GPT-5, GPT-4o, and GPT-4V LLM models on chart reading tasks using a CHART-6 benchmark subset, under three prompting conditions (CHART-6 instruction, question-only, and GPT-5 chart description), measured by correctness and LRAE.
- The study found that model architecture, specifically GPT-5, significantly improved inference accuracy on difficult image instances where GPT-4V previously failed, while prompt variations had only minor effects.
- This research highlights that LLM capability is a primary determinant of visualization understanding, with GPT-5 demonstrating superior agentic reasoning compared to the multimodal GPT-4 family for chart interpretation.

---

[Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management](http://arxiv.org/abs/2510.06727)

- SUPO (SUmmarization augmented Policy Optimization): introduces summarization-based context management to LLM RL training, enabling agents to scale beyond fixed context window limits by periodically compressing tool-use history into LLM-generated summaries that retain task-relevant information.
- This framework formalizes summarization steps within a Markov Decision Process and derives a policy gradient representation to optimize both tool-use behaviors and summarization strategies end-to-end.
- SUPO incorporates specific designs like trajectory management, group-relative advantage estimation, and an overlong trajectory masking mechanism to stabilize optimization and encourage tool-using behaviors for long-horizon tasks.

---

[Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support](http://arxiv.org/abs/2510.06674)

- AITL (Agent-in-the-Loop): introduces a continuous data flywheel for iteratively improving an LLM-based customer support system, integrating customer input, LLM-based interactive system (RAG), suggested replies, agent annotation, human + AI review, reply to customer, knowledge base, continuous learning system, DB quality exam, virtual judge, GLOW (Generalized LLM Offline Workflow), Ray clusters, and parameter-efficient fine-tuning (PEFT), to embed human feedback loops directly into operational workflows.
- The framework captures four key types of annotations—pairwise response preferences, agent adoption decisions and rationales, knowledge relevance checks, and identification of missing knowledge—directly from live customer operations.
- AITL's continuous learning pipeline seamlessly feeds these feedback signals back into model updates, significantly reducing retraining cycles and improving retrieval accuracy, generation quality, and agent adoption rates.

---

[TOOLMEM: Enhancing Multimodal Agents with Learnable Tool Capability Memory](http://arxiv.org/abs/2510.06664)

- TOOLMEM: introduces a closed-loop framework that equips multimodal agents with a learnable and evolving memory of tool capabilities, enabling them to improve tool selection and task-solving performance.
- The framework integrates structured memory initialization, feedback-driven learning from LLM-generated critiques, and retrieval-augmented generation for memory refinement.
- TOOLMEM-augmented agents achieve more accurate tool performance estimation and make better-informed tool choices in both text and image generation tasks.

---

[CODE AGENT CAN BE AN END-TO-END SYSTEM HACKER: BENCHMARKING REAL-WORLD THREATS OF COMPUTER-USE AGENT](http://arxiv.org/abs/2510.06607)

- AdvCUA (Computer-Use Agent Benchmark): introduces a benchmark for systematically evaluating Computer-Use Agents (CUAs) under realistic enterprise OS security threats, featuring Malicious Tasks (direct, TTP-based, end-to-end), an Enterprise-like Multi-host Environment Sandbox (realistic, isolated testing environment), Hard-coded Evaluation (deterministic, verifiable assessment), and an Attacker-knowledge Model (MITRE ATT&CK TTPs alignment).
- This benchmark comprises 140 tasks, including direct malicious tasks, TTP-based malicious tasks, and end-to-end kill chains, all aligned with real-world Tactics, Techniques, and Procedures (TTPs) from the MITRE ATT&CK Enterprise Matrix.
- The evaluation is conducted in a Docker-based multi-host environment, simulating an enterprise network with encrypted credentials, and uses deterministic hard-coded checks (Match, Trigger, Probe, Verify) to assess Attack Success Rate (ASR) and Bypass Success Rate (BSR).

---

[WEBDART: DYNAMIC DECOMPOSITION AND RE-PLANNING FOR COMPLEX WEB TASKS](http://arxiv.org/abs/2510.06587)

- WEBDART (Dynamic Decomposition and Re-planning for Complex Web Tasks): introduces a general framework that enables a single LLM to handle complex web tasks by dynamically decomposing objectives into Navigation Module (explores web pages, gathers info), Information Extraction Module (isolates, structures task-relevant content), and Execution Module (analyzes data, performs actions) subtasks, and continuously re-plans the decomposition based on new webpage observations.
- This framework reduces cognitive overload on LLM agents by allowing them to focus on one skill at a time and adaptively adjust plans to exploit shortcuts and avoid redundant exploration.
- WEBDART significantly improves end-to-end success rates on complex web tasks while maintaining performance on simpler tasks and reducing navigation steps.

---

[TINYSCIENTIST: An Interactive, Extensible, and Controllable Framework for Building Research Agents](http://arxiv.org/abs/2510.06579)

- TINYSCIENTIST: introduces an interactive, extensible, and controllable framework for building research agents, featuring workflow components (Thinker, Coder, Writer, Reviewer) and feature components (InputFormatter, OutputFormatter, MCPClient, Checker) to streamline automatic research.
- The framework enhances human-agent interaction through a tabular-based UI, supports flexible tool integration via MCPClient, and ensures responsible execution with built-in safety and budget controllers.
- It provides an open-source Python package and web demonstration, making advanced auto-research pipelines broadly accessible to researchers and developers.

---

[Auto-Stega: An Agent-Driven System for Lifelong Strategy Evolution in LLM-Based Text Steganography](http://arxiv.org/abs/2510.06565)

- Auto-Stega: introduces an agent-driven, self-evolving framework for LLM-based text steganography, which automatically discovers, composes, and adapts strategies at inference time, utilizing a Web Searcher, Strategy Library, Steganography LLM, Scorer LLM, Summarizer LLM, PC-DNTE, Decoding LLM, Eavesdropper, Secret Information, and Stego Text.
- This framework operates as a closed loop of generating, evaluating, summarizing, and updating, continually curating a structured strategy library and adapting across various contexts.
- The system achieves superior performance in perplexity and anti-steganalysis, particularly at higher embedding rates, by preserving imperceptibility and enhancing security.

---

[BENEFICIAL REASONING BEHAVIORS IN AGENTIC SEARCH AND EFFECTIVE POST-TRAINING TO OBTAIN THEM](http://arxiv.org/abs/2510.06534)

- Behavior Priming: introduces a reasoning-driven LLM-based pipeline to study and instill effective reasoning behavior patterns in agentic search, including Trajectory Curation, Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), a Reasoning LLM, an LLM-Judge, an Agentic Search Framework, an Underlying LLM, History Context, a Search Tool, Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery.
- The paper identifies four beneficial reasoning behaviors—Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery—which are systematically instilled into agentic search models through SFT followed by RL.
- Behavior Priming significantly boosts model performance by establishing a robust foundation for exploration and test-time scaling capabilities, demonstrating that reasoning behaviors are more critical than outcome correctness for unlocking RL potential.

---

[PARSE: LLM Driven Schema Optimization for Reliable Entity Extraction](http://arxiv.org/abs/2510.08623)

- PARSE (Parameter Automated Refinement and Schema Extraction): introduces a comprehensive framework for reliable structured information extraction, featuring ARCHITECT (Automated Refinement and Conversion Handler for Information Transformation and EnhanCemenT) for schema optimization, RELAY (Reverse Engineering Layer for Automated Yoking) for backward compatibility, and SCOPE (Schema Compliant Organized Pattern Extractor) for reflection-based extraction with guardrails.
- The framework addresses the challenge of LLM agents interacting with APIs and tools by optimizing JSON schemas for machine comprehension rather than treating them as static human-centric contracts, thereby improving extraction performance and reliability.
- PARSE's two-phase approach, including a Build Phase for schema refinement and an Extract Phase for robust information extraction, creates a virtuous cycle where optimized schemas enhance extraction accuracy and errors inform further schema improvements.

---

[HYPOTHESIS HUNTING WITH EVOLVING NETWORKS OF AUTONOMOUS SCIENTIFIC AGENTS](http://arxiv.org/abs/2510.08619)

- ASCollab (AScience-Collaboratory): introduces a framework for hypothesis hunting, modeling discovery as the interaction of scientific agents, evolving networks, and evaluation norms, implemented as a distributed system of LLM-based research agents.
- This system enables continuous, diverse exploration of large-scale datasets, where heterogeneous agents self-organize into networks, producing and peer-reviewing findings under shared evaluation standards.
- ASCollab leverages social dynamics and shared memory to sustain cumulative exploration, yielding diverse, high-quality, and novel discoveries, including established biomarkers and new therapeutic targets.

---

[GenPilot: A Multi-Agent System for Test-Time Prompt Optimization in Image Generation](http://arxiv.org/abs/2510.07217)

- GenPilot (Multi-Agent System for Test-Time Prompt Optimization in Image Generation): introduces a plug-and-play multi-agent system for test-time prompt optimization, which iteratively refines prompts for image generation by analyzing errors, generating candidates, scoring them, clustering, and updating memory.
- The system operates in two stages: Error Analysis, which decomposes prompts and identifies semantic inconsistencies via VQA and captioning, and Test-Time Prompt Optimization, which refines prompts based on errors and memory feedback using an MLLM scorer and clustering.
- GenPilot is model-agnostic, interpretable, and designed to handle complex prompts, demonstrating improved text-image consistency and structural coherence without requiring additional model training.

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
