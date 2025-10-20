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







#### 17th October 2025

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

#### 7th October 2025

[STRATIFIED GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents](http://arxiv.org/abs/2510.06214)

- Stratified GRPO (Stratified Group Relative Policy Optimization): introduces a reinforcement learning algorithm for LLM search agents, incorporating Stratified Advantage Normalization (SAN) (partitions trajectories, computes local advantages) and Blended Advantage (combines SAN with global estimator) to mitigate structural heterogeneity.
- This framework eliminates cross-stratum bias by ensuring trajectories are evaluated against homogeneous peers, leading to fair credit assignment and enhanced exploration for multi-step search strategies.
- Stratified GRPO consistently outperforms standard GRPO baselines on diverse question-answering benchmarks, demonstrating superior training rewards, stability, and effective search policies.

---

[Automated Program Repair of Uncompilable Student Code](http://arxiv.org/abs/2510.06187)

- APR (Automated Program Repair): introduces a framework for recovering uncompilable student code by assessing LLMs (GPT-5, Claude 3.5 Haiku, and Gemini 2.5 Flash) as Repair Agents (syntax-only repair) that process Uncompilable Student Code (input with errors) under Prompting Conditions (LLM context) to generate Repaired Code (compilable output).
- This study evaluates the LLMs' ability to produce compilable repairs while preserving the student's original structural intent and logic, which is crucial for student modeling.
- The research highlights how LLMs can effectively perform syntax-only repair on novice code, enabling richer analyses of learners' coding processes and development over time.

---

[RECODE-H: A BENCHMARK FOR RESEARCH CODE DEVELOPMENT WITH INTERACTIVE HUMAN FEED-BACK](http://arxiv.org/abs/2510.06186)

- ReCodeAgent: introduces a framework for iterative research code development, with its Agent, Memory Management, Feedback, Researcher, and RECODE-H (Benchmark) components, where LLM agents iteratively generate, test, and refine research code through structured researcher feedback within the RECODE-H benchmark.
- The framework leverages a five-level feedback hierarchy, from minimal execution logs to explicit code snippets, to systematically evaluate LLM agents' ability to adapt to progressively richer guidance in multi-turn interactions.
- It employs a memory management component to compact interaction history, ensuring context length remains bounded while preserving critical information for consistent and effective code generation across rounds.

---

[LLMs as Policy-Agnostic Teammates: A Case Study in Human Proxy Design for Heterogeneous Agent Teams](http://arxiv.org/abs/2510.06151)

- LLMs as Policy-Agnostic Teammates: introduces using LLM Agents (generates actions, decisions), Grid-World Stag Hunt Environment (simulates game dynamics), State Observation Module (extracts game features), Prompt Design Module (constructs LLM input), Action Space (defines available moves), Action Execution Module (applies LLM's action), Trajectory Formation Module (records decision sequence), Human Benchmark Data (provides human reference), Expert Judge Data (offers expert reference), and Evaluation Metrics (assesses LLM performance), to simulate human decision-making in multi-agent settings.
- This approach evaluates LLMs as human proxies in a grid-world capture game, comparing their generated decisions and multi-step action sequences against human participants and expert judges.
- The methodology demonstrates LLMs' ability to align with expert judgments, adapt to risk-sensitive strategies via prompt modifications, and produce human-like decision trajectories, establishing a scalable foundation for policy-agnostic teammates.

---

[Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents](http://arxiv.org/abs/2510.06078)

- RouteLLM (Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents): introduces a hierarchical multi-agent framework that translates natural language queries into constraint-aware route recommendations by coordinating specialized agents for parsing, POI selection, path planning, constraint resolution, and verification.
- The framework employs a Parser Agent to structure user intents, a Manager Agent to coordinate sub-agents (POI, Path, Constraint Agents) for task execution, and a Verifier Agent to synthesize results and ensure global constraint satisfaction.
- This multi-agent design bridges linguistic flexibility with spatial structure, mitigating LLM spatial reasoning weaknesses by decomposing complex requests into manageable sub-tasks and leveraging traditional routing algorithms for precise path optimization.

---

[SCIENTIFIC ALGORITHM DISCOVERY BY AUGMENTING ALPHAEVOLVE WITH DEEP RESEARCH](http://arxiv.org/abs/2510.06056)

- DeepEvolve: introduces an agent that integrates deep research with algorithm evolution, uniting external knowledge retrieval, cross-file code editing, and systematic debugging under a feedback-driven iterative loop.
- The framework consistently improves initial algorithms, producing executable new algorithms with sustained gains across diverse scientific benchmarks.
- DeepEvolve bridges the gap between unguided evolution and research without grounding, providing a reliable framework for advancing scientific algorithm discovery.

---

[Agent+P: Guiding UI Agents via Symbolic Planning](http://arxiv.org/abs/2510.06042)

- AGENT+P: introduces a novel framework that leverages symbolic planning to guide LLM-based UI agents, including UTG Builder, Node Selector, Plan Generator, and UI Explorer, by modeling an app's UI transition structure as a UI Transition Graph and using an external Symbolic Planner to generate globally aware, optimal high-level plans.
- The framework reformulates UI automation as a pathfinding problem on the UI Transition Graph, enabling off-the-shelf symbolic planners to generate provably correct and optimal plans, thereby preventing redundant exploration and guiding the UI Agent to achieve automation goals.
- AGENT+P is designed as a plug-and-play framework that enhances existing UI agents by improving success rates and reducing action steps in long-horizon UI automation tasks, mitigating LLM hallucination.

---

[Training-Free Time Series Classification via In-Context Reasoning with LLM Agents](http://arxiv.org/abs/2510.05950)

- FETA (training-Free time series classificaTion with LLM Agents): introduces a multi-agent framework for training-free time series classification, with Channel Decomposer, Example Retriever, Channel Reasoner, and Decision Aggregator components, enabling efficient, interpretable, and modular classification.
- The framework decomposes multivariate series into channel-wise subproblems, retrieves structurally similar labeled examples, and leverages a reasoning LLM to compare queries against these exemplars, producing channel-level labels with self-assessed confidences.
- A confidence-weighted aggregator then fuses all channel decisions, eliminating the need for pretraining or fine-tuning and enhancing interpretability through exemplar grounding and confidence estimation.

---

[EARL: Efficient Agentic Reinforcement Learning Systems for Large Language Models](http://arxiv.org/abs/2510.05943)

- EARL (Efficient Agentic Reinforcement Learning Systems for Large Language Models): introduces a scalable system for efficient agentic RL, addressing context length explosion and data dispatch bottlenecks, featuring a Parallelism Selector (dynamically adapts parallelism), Rollout (generates agent interactions), Experience Preparation (processes collected data), Data Dispatcher (exchanges intermediate data), and Model Update (updates LLM parameters).
- The Parallelism Selector dynamically adjusts model and training parallelism across RL stages based on sequence length and system load, while the Data Dispatcher performs layout-aware, decentralized exchange of intermediate data batches.
- These components collectively increase throughput, reduce long-context failures, and enable stable large-scale training of agentic LLMs without relying on hard limits or penalties of context length.

---

[LLM-FS-AGENT: A DELIBERATIVE ROLE-BASED LARGE LANGUAGE MODEL ARCHITECTURE FOR TRANSPARENT FEATURE SELECTION](http://arxiv.org/abs/2510.05935)

- LLM-FS-Agent (Deliberative Role-Based Large Language Model Architecture for Transparent Feature Selection): introduces a novel multi-agent architecture for interpretable and robust feature selection, including Input (data features, task description), Initiator Agent (initial semantic analysis), Refiner Agent (enhances analysis with metadata), Challenger Agent (critically examines arguments), Judge Agent (synthesizes arguments, assigns score), and Output (final importance score, reasoning), where it orchestrates a deliberative "debate" among multiple LLM agents to collectively evaluate feature relevance and provide detailed justifications.
- The system assigns specialized roles to LLM agents (Initiator, Refiner, Challenger, Judge) to facilitate structured debates around feature metadata and semantic utility, producing human-interpretable rationales.
- This deliberative architecture enhances decision-making transparency, improves computational efficiency by reducing downstream classifier training time, and achieves superior or comparable performance in feature selection.

---

[PROMPT REINFORCING FOR LONG-TERM PLANNING OF LARGE LANGUAGE MODELS](http://arxiv.org/abs/2510.05921)

- RPO (Reinforced Prompt Optimisation): introduces a prompt optimization framework that enhances LLMs' long-term planning in multi-turn tasks by iteratively updating the task instruction prompt of an LLM-based agent, including a Prompt writer LLM, System LLM, Feedbacker LLM, Rewriter LLM, and Experience Replay.
- The framework leverages reinforcement learning-inspired concepts, such as turn-by-turn feedback (Temporal Difference-style) and experience replay for prompt rewriting, to achieve significant improvements in multi-turn tasks like text-to-SQL and task-oriented dialogue.
- RPO is designed to be flexible, generalizable across diverse LLM backbones for both the system and meta-prompting agents, and reduces computational overhead by modifying only the instruction prompt rather than model parameters.

---

[Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches](http://arxiv.org/abs/2510.05748)

- Curriculum Learning Approach: introduces a method to elicit cooperation in multi-agent LLM systems by guiding LLM agents through progressively complex game environments, with strategic lessons generated by a Lesson Generation Agent after each stage.
- This approach, utilizing LLM Agents and various Curriculum Conditions, was compared against direct communication, revealing that simple communication protocols are more robust for coordination than curriculum learning, which showed sensitivity to design choices.
- The study highlights that poorly designed curricula, especially those front-loading defection-equilibrium games, can induce "learned pessimism" in agents, actively harming performance in social dilemmas.

---

[ARM: DISCOVERING AGENTIC REASONING MODULES FOR GENERALIZABLE MULTI-AGENT SYSTEMS](http://arxiv.org/abs/2510.05746)

- ARM (Agentic Reasoning Module): introduces a novel sequential reasoning approach where each granular step is executed by a specialized, self-contained reasoning agent, discovered through a Reflection-Guided Evolutionary Search that iteratively mutates and refines a basic Chain-of-Thought (CoT) procedure.
- The framework optimizes CoT reasoning by evolving agentic blocks (ARM) that can be used recursively or as subroutines in a learned Meta-Policy, significantly outperforming existing Multi-Agent Systems (MAS) and achieving high generalizability across models and domains.
- This approach emphasizes improving the fundamental step-by-step reasoning process rather than designing complex, heterogeneous MAS architectures, leading to more robust and scalable solutions.

---

[FinReflectKG - EvalBench: Benchmarking Financial KG with Multi-Dimensional Evaluation](http://arxiv.org/abs/2510.05710)

- FinReflectKG - EvalBench: introduces a benchmark and evaluation framework for financial Knowledge Graph (KG) extraction from SEC 10-K filings, integrating KG extraction with Single-pass, Multi-pass, and Reflection modes, and an Evaluation Framework featuring an LLM-as-Judge (J) with a Judging Protocol, Bias Controls, and Evaluation Dimensions (Faithfulness, Precision, Relevance, Comprehensiveness).
- The framework employs a deterministic commit-then-justify judging protocol with explicit bias controls to ensure reliable and reproducible evaluations, mitigating common LLM biases like leniency and position effects.
- This multi-dimensional evaluation approach enables fine-grained benchmarking and bias-aware assessment of KG extraction quality, advancing transparency and governance in financial AI applications.

---

[DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision](http://arxiv.org/abs/2510.05691)

- DecEx-RAG (Decision and Execution optimized Retrieval-Augmented Generation): introduces a novel framework that models Retrieval-Augmented Generation (RAG) as a Markov Decision Process (MDP) with Decision-Making and Execution Stages, Search Tree Expansion, Pruning Strategy, Rollout Simulations, Reward Function, Supervised Fine-tuning (SFT), Direct Preference Optimization (DPO), Policy Model (LLM), and Retriever, enabling fine-grained process supervision and efficient data expansion.
- The framework structurally decomposes RAG into distinct decision-making and execution stages, allowing for separate optimization of decision efficiency and content generation quality.
- An efficient pruning strategy, including Decision Branch Pruning and Execution Option Pruning, significantly enhances data construction efficiency by dynamically removing redundant search tree branches based on aggregated rewards.

---

[A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks](http://arxiv.org/abs/2510.05608)

- EAGLET (Efficient and Effective Global Planner Training): introduces an efficient and effective planner training method to enhance executor agents' planning abilities without human effort, including a SOTA LLM (synthesizes initial plans), Homologous Consensus Filtering (filters synthetic plans), Filtered Plans (high-quality plans for SFT), Cold-Start SFT (initial planner training), a Global Planner (generates high-level plans), Homologous Executors (evaluate plan effectiveness), Executor Capability Gain Reward (measures plan gain), Compute Reward (calculates reward for RL), RL Training (refines planner with reward), Feedback (from RL to Global Planner), Inference (Global Planner provides plans), an Executor (executes actions in environment), an ENV. (interactive task setting), and a Task (goal to be achieved).
- The framework employs a two-step process: first, synthesizing high-quality plans from an advanced LLM using homologous consensus filtering and applying fine-tuning as a cold start, then improving the planner with a rule-based reinforcement learning stage using a novel executor capability gain reward.
- This approach enables a plug-and-play global planner that provides explicit guidance to mitigate planning hallucinations, leading to improved performance and reduced training costs compared to RL-based baselines.

---

[AutoPentester: An LLM Agent-based Framework for Automated Pentesting](http://arxiv.org/abs/2510.05605)

- AutoPentester (LLM Agent-based Framework): introduces an LLM agent-based framework for automated penetration testing, vulnerability assessment, and threat analysis, which includes Summarizer (interprets tool outputs), Strategy Analyzer (plans attack path), Generator (generates commands), RAG module (retrieves relevant knowledge), Agent - Computer Interface (ACI) (executes commands), Results Verifier (validates outputs, adjusts commands), Repetition Identifier (prevents looping issues), Report Generator (creates comprehensive report), Security Tool Knowledge Base (stores cybersecurity information), Previous Steps History (stores past actions, findings), and Log Files (records pentesting information), designed to automate pentesting steps using common security tools in an iterative process.
- The framework dynamically generates attack strategies based on tool outputs, mimicking human pentester approaches, and significantly reduces human interaction compared to existing methods like PentestGPT.
- AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps, demonstrating higher automation, efficiency, and accuracy across the entire pentesting pipeline.

---

[AgentDR: Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents](http://arxiv.org/abs/2510.05598)

- AgentDR (Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents): introduces a novel LLM-agent framework that bridges LLM reasoning with scalable recommendation tools, including User Profile Generation, RecTool Memory, Intent Memory, Recommendation Tools, Substitute Generation, Complement Generation, Tool Comparison, Ranking Comparison, Ranking Aggregation, User Intent Discrimination, Dual S&C Reranking, General Reranking, Ranking Fusion, and a Hallucination Filtering Mechanism, to provide dynamic, personalized full-ranking recommendations.
- The framework addresses LLM limitations like hallucination and token constraints by delegating full-ranking tasks to traditional recommendation tools while leveraging LLMs for relational reasoning and output integration.
- AgentDR enhances recommendation relevance and scalability by inferring user intent for substitutes and complements, and dynamically refining aggregated rankings based on personalized tool suitability and user preferences.

---

[From Agentification to Self-Evolving Agentic AI for Wireless Networks: Concepts, Approaches, and Future Research Directions](http://arxiv.org/abs/2510.05596)

- MCSEAIF (Multi-agent Cooperative Self-evolving Agentic AI Framework): introduces a multi-agent cooperative self-evolving agentic AI framework for intelligent wireless networks, with all MCSEAIF-components, enabling autonomous adaptation and improvement without human intervention.
- The framework autonomously executes the entire AI agent life cycle, from data collection to monitoring, by assigning role-specialized LLMs under a supervisor agent's coordination, facilitating continuous self-improvement.
- A case study on antenna evolution in low-altitude wireless networks demonstrates the framework's ability to autonomously upgrade fixed antenna optimization to movable antenna optimization, improving beam gain and restoring degraded performance.

---

[IN-THE-FLOW AGENTIC SYSTEM OPTIMIZATION FOR EFFECTIVE PLANNING AND TOOL USE](http://arxiv.org/abs/2510.05592)

- AGENTFLOW: introduces a trainable, in-the-flow agentic framework that coordinates a planner, executor, verifier, and generator through an evolving memory and toolset, optimizing its planner within the multi-turn loop.
- The framework employs Flow-GRPO (Flow-based Group Refined Policy Optimization), an on-policy algorithm that converts multi-turn reinforcement learning into tractable single-turn policy updates by broadcasting a single, verifiable trajectory-level outcome to every turn.
- AGENTFLOW achieves strong cross-domain performance, surpassing specialized baselines and larger proprietary models by enhancing planning quality, tool-calling reliability, and discovering effective solution pathways.

---

[Mission Impossible: Feedback-Guided Dynamic Interactive Planning for Improving Reasoning on LLMs](http://arxiv.org/abs/2510.05577)

- FGDIP (Feedback-Guided Dynamic Interactive Planning): introduces a novel framework for enhancing LLM reasoning in multi-hop open-domain tasks by dynamically adapting information exploration strategies.
- The framework refines reasoning through historical error analysis and real-time feedback, systematically expanding the search space while converging towards accurate solutions.
- FGDIP achieves superior performance on HotpotQA and StrategyQA datasets by integrating its Multivariate Information Extractor, Node Generator, Step Evaluator, Error Analysis, Answer Evaluator, and Real-time Feedback components.

---

[Toward Systems Foundations for Agentic Exploration](http://arxiv.org/abs/2510.05556)

- Agentic Exploration System: introduces system foundations for LLM-powered agents to branch, backtrack, and search across execution paths, utilizing state restoration primitives like replay-to-node, snapshot/restore, and backtracking.
- The paper benchmarks existing snapshot/restore mechanisms, finding them too slow and lacking critical features for rapid, environment-agnostic agentic exploration, especially in real deployments with shared resources.
- It proposes a lightweight native forking primitive, requiring tighter integration between the OS, storage stack, and language runtimes, to achieve microsecond-latency state duplication for scalable multi-path exploration.

---

[CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension](http://arxiv.org/abs/2510.05520)

- CAM (Constructivist Agentic Memory): introduces a memory framework for LLM-based reading comprehension, incorporating Structured Schemata, Flexible Assimilation, Dynamic Accommodation, and a Prune-and-Grow Associative Strategy to enhance long-text understanding.
- The framework utilizes an incremental overlapping clustering algorithm for memory development, including Foundational Network Expansion, Ego-Centric Disentanglement, and Online Clustering Updates, to build a hierarchical and adaptable memory structure.
- For memory retrieval, CAM employs Fast Localization to identify relevant nodes and Associative Exploration, guided by LLMs, to recursively expand activated nodes for contextual inference, demonstrating superior performance and efficiency in diverse reading tasks.

---

[EVALUATING LLM SAFETY ACROSS CHILD DEVELOPMENT STAGES: A SIMULATED AGENT APPROACH](http://arxiv.org/abs/2510.05484)

- ChildSafe: introduces a benchmark for evaluating LLM safety using simulated child agents across four developmental stages, incorporating a nine-dimensional safety evaluation framework and structured conversation scenarios.
- The framework employs developmentally-authentic agents, validated through linguistic analysis and expert assessment, to systematically study LLM safety without ethical concerns of involving real children.
- ChildSafe provides a reproducible tool for age-aware safety research, revealing LLM vulnerabilities that vary by simulated age and informing age-appropriate AI deployment policies.

---

[A Survey on Agentic Security: Applications, Threats and Defenses](http://arxiv.org/abs/2510.06445)

- Agentic Security Taxonomy: introduces a holistic survey of the agentic security landscape, structuring the field around three interdependent pillars: Applications, Threats, and Defenses, to provide a comprehensive understanding of LLM-agent capabilities, vulnerabilities, and countermeasures.
- The survey provides a detailed taxonomy of over 150 papers, explaining how agents are used, their vulnerabilities, and the countermeasures designed to protect them.
- A cross-cutting analysis reveals emerging trends in agent architecture, such as the prevalence of multi-agent systems and planner-executor designs, while highlighting critical research gaps in model and modality coverage.

---

[Leveraging Large Language Models for Cybersecurity Risk Assessment — A Case from Forestry Cyber-Physical Systems](http://arxiv.org/abs/2510.06343)

- LLM-based tool with RAG: introduces an LLM-based tool leveraging locally hosted LLMs and Retrieval-Augmented Generation to support cybersecurity risk assessment in forestry cyber-physical systems.
- The tool, built with Llama 2 7B and a RAG architecture using a vector database, assists experts by generating initial risk assessments, identifying threats, and performing redundancy checks while adhering to data protection requirements.
- The study highlights the LLM's utility in specific evaluation and assistance roles, emphasizing the necessity for human oversight and the importance of context-awareness, transparency, and adherence to standards like IEC 62443.

---

[The Safety Challenge of World Models for Embodied AI Agents: A Review](http://arxiv.org/abs/2510.05865)

- World Model: introduces a comprehensive literature review of World Models (WMs) for embodied AI agents, focusing on safety implications in scene and control generation tasks, utilizing observation (input data), condition (contextual input), the World Model (core processing unit), future observations (predicted outputs), and pathology criteria (safety evaluation metrics).
- The review identifies and categorizes common faults, referred to as pathologies, in WM predictions and provides a quantitative evaluation of these results.
- The study specifically examines WMs in autonomous driving and robotics, establishing criteria for assessing safety in generated outputs.

---

[Generative AI-Driven Hierarchical Multi-Agent Framework for Zero-Touch Optical Networks](http://arxiv.org/abs/2510.05625)

- GenAI-Driven Hierarchical Multi-Agent Framework: introduces a hierarchical multi-agent system for zero-touch optical networks, featuring a central Network Director, a Shared Pool, four Division Agents (Optical-layer, Digital Twin, Control, Support), and specialized LLM-based AI Experts for task allocation, coordination, and execution.
- This framework leverages LLM-based AI agents to autonomously manage complex, multi-layer optical network tasks, facilitating seamless communication and maintaining high task precision through its hierarchical structure and shared memory.
- The system demonstrates efficiency and adaptability in network planning, operation, and upgrade stages, enabling intelligent, collaborative, and scalable network management solutions for zero-touch optical networks.

---

[TEXT2INTERACT: HIGH-FIDELITY AND DIVERSE TWO-PERSON INTERACTION GENERATION FROM TEXT](http://arxiv.org/abs/2510.06504)

- Text2Interact: introduces a framework for high-fidelity and diverse two-person interaction generation from text, featuring InterCompose (scalable data synthesizer) and InterActor (text-to-interaction generator).
- InterCompose leverages an LLM (generates interaction descriptions) to synthesize two-person interactions by composing single-person motions from a Single-Person Model (generates initial agent motion) and a Reaction Gen Model (generates second agent's motion), with a Neural Motion Evaluator (filters synthetic data quality) ensuring quality.
- InterActor, a text-to-interaction generator, employs an N-block generator (generates two-person interaction) with a Word-Level Conditioning Module (Mw) (text-to-motion cross-attention) and a Motion-Motion Interaction Module (Mm) (models inter-agent dependencies), using CLIP (extracts word-level text embeddings) for fine-grained language conditioning.

---

[VERIEQUIVBENCH: AN EQUIVALENCE SCORE FOR GROUND-TRUTH-FREE EVALUATION OF FORMALLY VERIFIABLE CODE](http://arxiv.org/abs/2510.06296)

- VeriEquivBench: introduces a novel evaluation framework that replaces ground-truth matching with a formally grounded equivalence score, rigorously verifying generated specifications and code, and includes a large-scale benchmark dataset of 2,389 complex algorithmic problems.
- The framework leverages LLMs for code and specification generation, natural language translation, and judging, alongside the Dafny verifier for proving mutual equivalence between code and formal specifications.
- The benchmark dataset is constructed from a LeetCode corpus and a synthetically generated tag-composition subset, utilizing a structured tagging system for scalable novel query generation.

---

#### 6th October 2025


[UnitTenX: Generating Tests for Legacy Packages with AI Agents Powered by Formal Verification](http://arxiv.org/abs/2510.05441)

- UnitTenX: introduces an AI multi-agent system that combines AI agents, formal methods, and LLMs to automate unit test generation for legacy C codebases, enhancing test coverage and reliability.
- The system employs a multi-step process including AutoMockUps for function mockups, a Symbolic Analyzer using ESBMC for crash condition extraction, LLM-driven Unit Test Generation, Coverage Analysis with gcov, and an LLM-based Reflection loop for iterative test suite improvement.
- The framework effectively addresses challenges in maintaining and modernizing complex legacy software by generating high-quality, production-ready regression tests, recovering from compilation errors, and improving code documentation.

---


[Staircase Streaming for Low-Latency Multi-Agent Inference](http://arxiv.org/abs/2510.05059)

- Staircase Streaming: introduces a novel approach for low-latency multi-agent inference, utilizing proposer agents, an aggregator agent, and a chunking mechanism to stream tokens incrementally between models.
- This method breaks strict sequential dependencies by enabling parallel processing, where the aggregator begins generating output as soon as partial chunks from proposer agents are available.
- The approach significantly reduces Time to First Token (TTFT) by up to 93% while maintaining response quality, further optimized by prefix-caching.

---

[Large Language Models Achieve Gold Medal Performance at International Astronomy & Astrophysics Olympiad](http://arxiv.org/abs/2510.05016)

- IOAA-LLM Benchmark Framework: introduces a comprehensive evaluation of state-of-the-art LLMs (evaluated models) on the International Olympiad on Astronomy and Astrophysics (IOAA) exams, utilizing an IOAA Dataset (astronomy problems benchmark), a standardized Prompt Template (standardized input instructions), a Reference Document (supplementary factual information), Human Graders (expert solution evaluators), Evaluation Rubrics (official scoring guidelines), and Error Analysis (categorized performance breakdown) to assess their problem-solving capabilities.
- The framework benchmarks five LLMs (GPT-5, OpenAI 03, Gemini 2.5 Pro, Claude-4.1-Opus, and Claude-4-Sonnet) on 57 IOAA problems from 2022-2025, covering theory and data analysis, to understand their strengths and limitations in complex astronomical reasoning.
- This systematic evaluation reveals that top LLMs achieve gold medal performance in theory exams but show weaknesses in geometric/spatial reasoning, multimodal data interpretation, and mathematical rigor, highlighting critical gaps for autonomous astronomical research.

---

[LLM-HANABI: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game](http://arxiv.org/abs/2510.04980)

- LLM-HANABI: introduces a novel benchmark for evaluating rationale inference and Theory-of-Mind (ToM) in LLMs within a dynamic, multi-agent collaborative setting, utilizing the cooperative card game Hanabi.
- The framework includes LLM-driven agents interacting with a game environment, and a ToM Evaluation System that extracts reasoning statements (Rationale, First-Order ToM, Second-Order ToM) and scores them using an LLM-as-a-judge.
- This system provides a scalable and quantitative method to assess interactive ToM and rationale inference, revealing a strong positive correlation between ToM proficiency and game success, with first-order ToM being a stronger predictor than second-order ToM.

---

[BRIDGING CLINICAL NARRATIVES AND ACR APPROPRIATENESS GUIDELINES: A MULTI-AGENT RAG SYSTEM FOR MEDICAL IMAGING DECISIONS](http://arxiv.org/abs/2510.04969)

- Multi-Agent RAG System: introduces a multi-agent cognitive architecture that automates the translation of free-text clinical scenarios into guideline-adherent imaging recommendations, utilizing a fine-tuned ColBERT retrieval agent, an ACR knowledge base, and LLM-based selector and supervisor agents.
- The system achieves high accuracy in identifying appropriate medical imaging procedures by semantically matching clinical queries to structured ACR guidelines and synthesizing evidence-based responses.
- This approach addresses the underutilization of clinical guidelines by bridging the gap between unstructured patient narratives and structured criteria, enhancing clinical decision support.

---

[MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2510.04935)

- MARS (Multi-Agent System for Deep ReSearch): introduces a dual-system framework that integrates System 1 (fast, intuitive thinking agent) and System 2 (deliberate reasoning agent) with External Tools (information sources/computation) and Bin Packing (content organization algorithm) for complex reasoning.
- The framework employs a Multi-agent Reinforcement Learning Framework (dual-system optimization mechanism) extending Group Relative Policy Optimization (GRPO) (RL algorithm for joint optimization) with a Multi-agent Rollout Module (generates RL training trajectories), Reward Model (evaluates predicted answer correctness), Policy LLM (underlying LLM for agents), Reference LLM (baseline for policy updates), Group Computation (calculates advantage values), and Sample Balance (adjusts training sample distribution).
- MARS also includes a Data Curation Pipeline (prepares high-quality training data) with Candidate Data (initial raw data pool), Clarity Filtering (removes ambiguous prompts), Graduate-level Filtering (filters for difficulty), Challenge and Correctness Verification (verifies difficulty and answers), and Training Data (final curated dataset) to ensure robust performance in dynamic information environments.

---

[Where Did It All Go Wrong? A Hierarchical Look into Multi-Agent Error Attribution](http://arxiv.org/abs/2510.04886)

- ECHO (Error attribution through Contextual Hierarchy and Objective consensus analysis): introduces a novel algorithm that combines hierarchical context representation, objective analysis-based evaluation, and consensus voting to improve error attribution accuracy in LLM multi-agent systems.
- The framework leverages a multi-layered hierarchical context to capture local and global interaction patterns, employs a panel of specialized LLM analysts for independent objective evaluations, and synthesizes findings through confidence-weighted consensus voting.
- This approach addresses limitations of existing error attribution methods by providing a robust framework for debugging complex multi-agent systems, particularly in cases involving subtle reasoning errors and interdependencies.

---

[RL IS A HAMMER AND LLMS ARE NAILS: A SIMPLE REINFORCEMENT LEARNING RECIPE FOR STRONG PROMPT INJECTION](http://arxiv.org/abs/2510.04885)

- RL-Hammer: introduces a simple reinforcement learning recipe for training attacker models that automatically learn to perform strong prompt injections and jailbreaks, utilizing an Attacker Model, Target LLMs, Group Relative Policy Optimization (GRPO), KL Regularization Term Removal, Joint Training, Soft Reward Signal, Restricted Format Enforcement, Diversity Rewards, Detection Rewards, Prompt Injection Detectors, and an LLM-based Judge.
- The framework achieves high attack success rates against commercial-level LLMs with defenses by employing practical techniques to mitigate sparse rewards and accelerate learning.
- The paper demonstrates that existing LLM defenses, while effective against naive prompt injections, are not robust against attacks generated by the framework, highlighting the need for stronger, more principled defenses.

---

[ALIGNMENT TIPPING PROCESS: How SELF-EVOLUTION PUSHES LLM AGENTS OFF THE RAILS](http://arxiv.org/abs/2510.04860)

- ATP (Alignment Tipping Process): introduces a critical post-deployment risk unique to self-evolving LLM agents, formalizing and analyzing it through two complementary paradigms: Self-Interested Exploration and Imitative Strategy Diffusion.
- This process describes how LLM agents' alignment erodes rapidly under self-evolution, with initially aligned models converging toward unaligned states due to feedback-driven decay during deployment.
- The paper demonstrates that current reinforcement learning-based alignment methods provide only fragile defenses against ATP, highlighting alignment as a dynamic property vulnerable to experience rather than a static one.

---

[FRESHBREW: A BENCHMARK FOR EVALUATING AI AGENTS ON JAVA CODE MIGRATION](http://arxiv.org/abs/2510.04852)

- FreshBrew: introduces a novel benchmark for evaluating AI agents on project-level Java migrations, including a Dataset Curation Pipeline, a Migration Agent, an Evaluation Protocol, and an LLM-as-Judge, designed to assess an agent's ability to preserve program semantics and avoid reward hacking.
- The benchmark curates a high-coverage dataset of real-world Java projects that build on JDK 8 but fail on modern JDKs, ensuring meaningful evaluation of semantic correctness.
- Its robust evaluation protocol defines migration success by successful compilation, passing all original tests, and maintaining test coverage, thereby safeguarding against reward hacking.

---

[LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation](http://arxiv.org/abs/2510.04851)

- LEGOMem (Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation): introduces a modular procedural memory framework for multi-agent LLM systems, including an orchestrator (plans, delegates, selects agents), task agents (execute subtasks, use tools), a procedural memory bank (stores memory units) with full-task memories (task-level plans, reasoning traces) and subtask memories (agent behavior, tool interactions), OfficeBench APIs (environment interaction), and various retrieval strategies (methods for memory lookup).
- The framework operates in an offline memory curation phase (distills successful trajectories) to create memory units and an online memory-augmented inference phase (distributes retrieved memories) where these units are allocated to orchestrators and task agents to guide planning and execution.
- LEGOMem's modular and role-aware design enables agents to learn from past experiences, improving planning, coordination, and task execution, and allowing smaller LLMs to achieve competitive performance.

---

[GUISpector: An MLLM Agent Framework for Automated Verification of Natural Language Requirements in GUI Prototypes](http://arxiv.org/abs/2510.04791)

- GUISpector (Multi-modal LLM Agent Framework for Automated Verification of Natural Language Requirements in GUI Prototypes): introduces a novel framework leveraging a multi-modal LLM agent to automate the verification of natural language requirements in GUI prototypes, including a human-in-the-loop interface, an MLLM agent verification loop, and an agentic implementation-verification loop.
- The framework interprets and operationalizes natural language requirements, autonomously plans and executes verification trajectories across GUI applications, and systematically extracts detailed feedback.
- GUISpector provides actionable insights for developers to iteratively refine GUI artifacts or directly inform LLM-based code generation within a closed feedback loop, enhancing automated GUI development workflows.

---

[TRADE IN MINUTES! RATIONALITY-DRIVEN AGENTIC SYSTEM FOR QUANTITATIVE FINANCIAL TRADING](http://arxiv.org/abs/2510.04787)

- TiMi (Trade in Minutes): introduces a rationality-driven multi-agent system that decouples strategy development from minute-level deployment, leveraging specialized LLM capabilities for semantic analysis, code programming, and mathematical reasoning.
- The system employs a two-tier analytical paradigm, layered programming design for trading bot implementation, and closed-loop optimization driven by mathematical reflection.
- This architecture enables comprehensive strategy development and quantitative-level efficiency, ensuring stable profitability, action efficiency, and risk control in volatile financial markets.

---

[BROKENMATH: A BENCHMARK FOR SYCOPHANCY IN THEOREM PROVING WITH LLMS](http://arxiv.org/abs/2510.04721)

- BROKENMATH: introduces a benchmark for evaluating sycophantic behavior in LLMs within natural language theorem proving, utilizing Sources (collects advanced competition theorems), Parser (extracts questions from PDFs), Expert (Question Extraction) (validates extracted questions), LLM (Corrupt Statements Generation) (generates demonstrably false but plausible statements), Expert (Corrupt Statements Refinement) (reviews and refines corrupted statements), LLM-as-a-judge Framework (evaluates LLM responses), and Judge (Model Evaluation) (categorizes LLM behavior as Sycophant, Ideal, Detected, or Corrected), where it constructs a dataset of challenging mathematical theorems perturbed to create false but plausible statements, and evaluates LLMs using an LLM-as-a-judge framework.
- The benchmark is built from advanced 2025 competition problems, which are perturbed by an LLM to produce false statements and subsequently refined through expert review, resulting in 504 high-quality samples.
- Experiments reveal widespread sycophancy in state-of-the-art LLMs, with the best model, GPT-5, producing sycophantic answers 29% of the time, and show that sycophancy is more pronounced in proof-based problems and increases with problem difficulty.

---

[BEYOND OUTCOME REWARD: DECOUPLING SEARCH AND ANSWERING IMPROVES LLM AGENTS](http://arxiv.org/abs/2510.04695)

- DeSA (Decoupling Search-and-Answering): introduces a two-stage training framework that explicitly separates search optimization from answer generation, including Stage 1 (Search Skill Acquisition), RAG Agent, Agentic Search (Search Module), Search Reward (Rs), Stage 2 (Outcome Optimization), Search-Augmented Agent, Answer Generation, Outcome Reward (Ro), LLM backbone, Search Engine, and GRPO algorithm, where it addresses systematic deficiencies in LLM agent search behaviors by decoupling search skill acquisition from final answer generation.
- The framework first trains agents to improve search effectiveness using retrieval recall-based rewards in Stage 1, then optimizes final answer generation with outcome rewards in Stage 2.
- This decoupled approach consistently improves search behaviors, delivering higher search recall and answer accuracy compared to outcome-only baselines and single-stage training methods.

---

[Multi-Agent Tool-Integrated Policy Optimization](http://arxiv.org/abs/2510.04678)

- MATPO (Multi-Agent Tool-Integrated Policy Optimization): introduces a multi-agent-in-one-model RL training framework that enables distinct planner- and worker-agent roles to be trained within a single LLM instance using role-specific prompts and a principled credit assignment mechanism.
- This framework addresses limitations of single-agent approaches by managing context length and noisy tool responses through task delegation to worker-agents, while preserving specialization benefits and infrastructure efficiency.
- MATPO consistently outperforms single-agent baselines in performance and robustness to noisy tool outputs across various benchmarks, demonstrating effective multi-agent coordination and efficient RL training.

---

[EDUPERSONA: BENCHMARKING SUBJECTIVE ABILITY BOUNDARIES OF VIRTUAL STUDENT AGENTS](http://arxiv.org/abs/2510.04648)

- EduPersona: introduces a large-scale benchmark for evaluating virtual student agents, encompassing Dataset Construction, Persona and Behavior Annotation, an Evaluation Framework, and Systematic Experiments and Analysis, to assess subjective abilities across three progressive tasks: Basic Coherence, Student Realism, and Persona Consistency.
- The framework utilizes Base LLMs, which are adapted via a Fine-tuning Mechanism using 10 distinct Persona Configurations, to generate and evaluate student responses in classroom settings.
- EduPersona's design transforms subjective performance into quantifiable measures, enabling systematic and reproducible benchmarking of virtual student agents' capabilities in educational contexts.

---

[QuantAgents: Towards Multi-agent Financial System via Simulated Trading](http://arxiv.org/abs/2510.04643)

- QuantAgents: introduces a multi-agent financial system integrating simulated trading to evaluate investment strategies and market scenarios, comprising four specialized agents (Manager, Simulated Trading Analyst, Risk Control Analyst, Market News Analyst) collaborating through three types of meetings (Market Analysis, Strategy Development, Risk Alert) and a single agent workflow.
- The system leverages a reflection-driven decision-making process, utilizing 26 financial analysis tools and three memory types (Market Information, Strategy, Report Memory), and is incentivized by a dual reward mechanism from both real-world market performance and simulated trading accuracy.
- QuantAgents aims to bridge the gap between LLM-based agents and human financial experts by enabling long-term prediction of future trends through risk-free experimentation in virtual trading environments, demonstrating superior performance with nearly 300% return over three years.

---

[Social Agent: Mastering Dyadic Nonverbal Behavior Generation via Conversational LLM Agents](http://arxiv.org/abs/2510.04637)

- Social Agent (Social Agent System): introduces a novel framework for synthesizing realistic and contextually appropriate co-speech nonverbal behaviors in dyadic conversations.
- The framework leverages an LLM-based agentic system to direct conversation flow and determine interactive behaviors, coupled with a dual-person gesture generation model.
- It continuously analyzes interlocutor movements, infers intentions, and forms a feedback loop for dynamic and responsive interactions.

---

[MedPAO: A Protocol-Driven Agent for Structuring Medical Reports](http://arxiv.org/abs/2510.04623)

- MedPAO (Protocol-Driven Agent for Structuring Medical Reports): introduces a novel agentic framework that transforms unstructured medical reports into protocol-compliant structured data, leveraging an LLM engine within a Plan-Act-Observe (PAO) loop to orchestrate specialized tools for medical concept processing.
- This framework operationalizes established clinical protocols, such as the ABCDEF protocol for CXR analysis, to ensure accuracy and verifiable reasoning in the structured output.
- MedPAO's modular design and toolset, including concept extraction, ontology mapping/filtering, and protocol categorization, significantly outperform baseline LLM methods in medical concept categorization.

---

[Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](http://arxiv.org/abs/2510.04618)

- ACE (Agentic Context Engineering): introduces a framework that treats contexts as evolving playbooks, accumulating, refining, and organizing strategies through modular generation, reflection, and curation processes.
- The framework prevents context collapse and preserves detailed knowledge by using structured, incremental updates, outperforming baselines in agent and domain-specific tasks.
- ACE achieves self-improvement without labeled supervision by leveraging natural execution feedback, significantly reducing adaptation latency and rollout costs.

---

[A Case for Declarative LLM-friendly Interfaces for Improved Efficiency of Computer-Use Agents](http://arxiv.org/abs/2510.04607)

- GOI (Goal-Oriented Interface): introduces a novel abstraction that transforms existing GUIs into three declarative primitives: access, state, and observation, enabling LLMs to focus on high-level semantic planning by decoupling policy from mechanism.
- This framework addresses the challenges LLMs face with imperative GUI designs, which require complex, fine-grained action sequences for navigation and interaction, leading to low success rates and excessive LLM calls.
- GOI significantly improves task success rates by 67% and reduces interaction steps by 43.5% compared to GUI-based baselines, often completing tasks with a single LLM call.

---

[COSMIR: Chain Orchestrated Structured Memory for Iterative Reasoning over Long Context](http://arxiv.org/abs/2510.04568)

- COSMIR (Chain Orchestrated Structured Memory for Iterative Reasoning): introduces a training-free framework for long-context reasoning, replacing ad hoc messages with a structured memory, which includes a PLANNER agent, WORKER agents with EXTRACT, INFER, and REFINE phases, a MANAGER agent, and a Structured Memory for iterative reasoning over Input Chunks.
- This framework enhances faithfulness, long-range aggregation, and auditability by using a centralized structured memory and a fixed micro-cycle for worker agents, reducing information loss compared to Chain of Agents baselines.
- COSMIR converts user queries into checkable sub-questions, processes text chunks via a fixed micro-cycle of extracting, inferring, and refining, and synthesizes the final answer directly from the structured memory.

---

[TRAJECT-BENCH: A TRAJECTORY-AWARE BENCHMARK FOR EVALUATING AGENTIC TOOL USE](http://arxiv.org/abs/2510.04550)

- TRAJECT-Bench: introduces a trajectory-aware benchmark for evaluating LLMs' agentic tool use, with Tool Set (Curated real-world APIs), Tool-use Trajectories (Synthesized parallel/sequential tool calls), Parallel Trajectories (Independent tool calls), Sequential Trajectories (Interdependent tool chains), User Queries (Simple/hard task descriptions), Evaluation Metrics (Measures tool-use performance), Trajectory-aware Metrics (Assesses tool selection/usage details), Final Performance Metrics (Evaluates end-task accuracy), LLM Test Models (State-of-the-art LLMs), Tool Selection Strategies (Methods for tool provision), and Agentic Evaluation Framework (Assesses LLM agent capabilities), designed to comprehensively assess tool-use capabilities through diverse tasks and fine-grained metrics.
- The benchmark synthesizes tool-use trajectories with varying breadth (parallel calls) and depth (interdependent chains) and pairs them with user queries of different difficulty levels, grounded in production-style APIs.
- It provides detailed trajectory-level diagnostics beyond final accuracy, including tool selection, argument correctness, and dependency satisfaction, to identify specific failure modes and offer actionable guidance for LLM development.

---

[CODE WORLD MODELS FOR GENERAL GAME PLAYING](http://arxiv.org/abs/2510.04542)

- CWM (Code World Model): introduces a novel approach for general game playing, leveraging LLMs to translate natural language game rules and trajectories into an executable Python code world model, which includes core game logic, inference, and heuristic value functions, all verified and refined through unit tests and iterative mechanisms, and then utilized by planning algorithms like MCTS, ISMCTS, or PPO agents.
- This framework enables classical planning algorithms like MCTS and ISMCTS to achieve strategic depth and generalization across various perfect and imperfect information games, outperforming direct LLM policies.
- The CWM approach shifts the LLM's role from direct policy generation to meta-task data-to-code translation, ensuring verifiability and adaptability to novel game environments.

---

[3Dify: a Framework for Procedural 3D-CG Generation Assisted by LLMs Using MCP and RAG](http://arxiv.org/abs/2510.04536)

- 3Dify (Procedural 3D Computer Graphics Generation Framework): introduces a procedural 3D-CG generation framework, with Dify Platform, LLM Agents (Visualizer LLM, Planner LLM, Manager LLM), Retrieval-Augmented Generation (RAG), Model Context Protocol (MCP) Client, MCP Servers, Computer-Using Agent (CUA), Digital Content Creation (DCC) Tools, API, Local Inference Platform, Image Generation AI, Feedback Loop, CLI, and GUI, enabling users to generate 3D-CG content through natural language instructions and automated DCC tool operations.
- The framework leverages multiple LLM agents for distinct roles, including pre-visualization, procedural parameter planning, and automated control of DCC tools via MCP or CUA.
- It incorporates an interactive image-selection feedback loop and RAG to enhance generation quality and adaptability, supporting both external and locally deployed LLMs.

---

[ARIA: AN AGENT FOR RETRIEVAL AND ITERATIVE AUTO-FORMALIZATION VIA DEPENDENCY GRAPH](http://arxiv.org/abs/2510.04520)

- ARIA (Agent for Retrieval and Iterative Autoformalization): introduces a system for conjecture-level formalization in Lean, employing a two-phase Graph-of-Thought process for statement decomposition and bottom-up synthesis, alongside AriaScorer for semantic verification.
- The framework integrates Retrieval-Augmented Generation (RAG) for grounding concepts in Mathlib and a compiler-in-the-loop reflection mechanism for syntactic correctness.
- ARIA achieves state-of-the-art performance in auto-formalization, particularly on challenging research-level mathematical conjectures, by emulating human expert reasoning.

---

[ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering](http://arxiv.org/abs/2510.04514)

- ChartAgent: introduces a novel agentic framework for visually grounded reasoning in complex chart question answering, leveraging an iterative ReAct-style loop with specialized vision tools and a Base MLLM.
- The framework systematically decomposes chart queries into visual subtasks, actively manipulating chart images through a Modular Vision Tool Library and employing a Visual Self-Verification Mechanism for adaptive refinement.
- It achieves state-of-the-art performance on unannotated and numerically intensive charts by augmenting MLLM reasoning with chart-specialized visual capabilities, demonstrating robustness and generalization across diverse chart types and complexity levels.

---

[GRACE: GENERATIVE REPRESENTATION LEARNING VIA CONTRASTIVE POLICY OPTIMIZATION](http://arxiv.org/abs/2510.04506)

- GRACE (Generative Representation Learning via Contrastive Policy Optimization): introduces a novel framework that reimagines contrastive signals as rewards to guide a generative policy, transforming LLMs into interpretable agents by generating explicit rationales, which are then encoded into high-quality embeddings via mean pooling.
- The framework leverages policy gradient optimization with a multi-component reward function to maximize similarity between query-positive pairs and minimize similarity with negatives, enabling transparent and inspectable reasoning processes.
- GRACE unifies representation learning with generation, yielding stronger embeddings and transparent rationales while preserving general LLM capabilities, as demonstrated by broad cross-category gains on the MTEB benchmark.

---

[Multi-Agent Collaborative Intelligence: Dual-Dial Control for Reliable LLM Reasoning](http://arxiv.org/abs/2510.04488)

- MACI (Multi-Agent Collaborative Intelligence): introduces a control framework for multi-agent LLM reasoning, featuring LLM Agents (participating in debate), a Moderator (orchestrates debate), an Information Dial (TQ) (gates evidence quality), a Behavior Dial (CL) (schedules contentiousness), a CRIT (Cross-family LLM judge) (evaluates argument quality), Disagreement (DJs) Signal (quantifies belief divergence), Overlap (O) Signal (measures shared evidence), Evidence Quality (Q) Signal (aligns evidence to target), Information Gain (Î) Signal (quantifies uncertainty reduction), a Scheduler (adjusts dials, manages budget), and RAG Plans (targeted information acquisition), to enhance accuracy, calibration, and efficiency while ensuring provable termination.
- The framework actively modulates LLM agent interactions through two independent dials—information gating and behavioral stance—guided by four signals (disagreement, overlap, evidence quality, and argument quality) with plateau-based stopping.
- MACI translates residual uncertainty into precision RAG plans, providing theory-lite guarantees for non-increasing dispersion and provable termination, making multi-agent debate a budget-aware and measurable process.

---

[Autonomy Matters: A Study on Personalization-Privacy Dilemma in LLM Agents](http://arxiv.org/abs/2510.04465)

- LLM Agent Study: investigates the personalization-privacy dilemma in LLM agents by manipulating personalization types and autonomy levels, affecting user privacy concerns, trust, and willingness to use.
- The study utilizes an LLM agent, a chat-based discussion system with role-playing agents, and a sensitivity detection module to simulate interpersonal communication scenarios.
- Intermediate autonomy is found to mitigate the dilemma by flattening personalization effects on privacy and trust, suggesting a balanced approach to agent autonomy and user control.

---

[SURVEYBENCH: CAN LLM(-AGENTS) WRITE ACADEMIC SURVEYS THAT ALIGN WITH READER NEEDS?](http://arxiv.org/abs/2510.03120)

- SurveyBench: introduces a fine-grained, quiz-driven evaluation framework, with Survey Topic Preparation (collecting/refining/sampling topics), Fairness-Guaranteed Survey Writing (LLM prompt design/parameter setting), Evaluation Dimensions (outline/content quality/richness metrics), LLM-as-Judge Evaluation (LLM scoring of LLM/human surveys), and Quiz-based Survey Evaluation (LLM-powered general/topic-specific quizzes), designed to rigorously assess LLM-generated academic surveys against reader needs.
- The framework features a curated benchmark dataset of popular research topics and high-quality human-written surveys, alongside a dual-mode evaluation protocol incorporating both human-reference-based and non-reference-based metrics.
- SurveyBench effectively challenges existing LLM4Survey approaches by revealing deficiencies in technical detail, reasoning, and core idea abstraction, highlighting the need for targeted optimization in automatic survey writing.

---

[AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering](http://arxiv.org/abs/2510.05445)

- AgentRouter: introduces a framework that formulates multi-agent Question Answering (QA) as a knowledge-graph-guided routing problem, supervised by empirical performance signals, converting QA instances into a Knowledge Graph (KG) and training a RouterGNN to produce task-aware routing distributions over agents.
- The framework leverages soft supervision derived from empirical agent performance and weighted aggregation of agent outputs to learn principled collaboration schemes that capture complementary strengths of diverse agents.
- By embedding queries, entities, and agents into a unified graph, AgentRouter grounds agent selection in the same semantic structures that govern reasoning for QA, adapting to new inputs and effectively capturing complementary agent strengths.

---

[ADVERSARIAL REINFORCEMENT LEARNING FOR LARGE LANGUAGE MODEL AGENT SAFETY](http://arxiv.org/abs/2510.05442)

- ARLAS (Adversarial Reinforcement Learning for Agent Safety) introduces a novel framework that co-trains an Attacker LLM (generates prompt injections) and an Agent LLM (defends and completes tasks) within an Environment (simulates interactions), leveraging Population-Based Training (robust agent training strategy) to enhance LLM agent safety against indirect prompt injections.
- The framework utilizes Webpage Content (initial input data) and User Information (sensitive data) to create a Poisoned Observation (webpage with injected prompts), where the Agent LLM performs a Tool Call (agent's action) while being evaluated by Reward Functions (evaluate performance).
- ARLAS employs Imitation Learning (initial model warm-up) and the GRPO Algorithm (RL training updates) to enable the attacker to generate diverse and challenging attacks, leading to a more robust agent with improved task completion rates.

---

[AINSTEIN: ASSESSING THE FEASIBILITY OF AI-GENERATED APPROACHES TO RESEARCH PROBLEMS](http://arxiv.org/abs/2510.05432)

- AINSTEIN: introduces a framework for evaluating LLMs as autonomous scientific problem-solvers, with all its components, which extracts research problems from scientific abstracts and generates/refines technical solutions through iterative critique loops.
- The framework operates in two phases: Problem Extraction, where a Generalizer agent distills abstracts into problem statements, and Solution Generation, where a Solver agent proposes technical solutions.
- Both phases employ nested internal and external critique loops, mimicking scientific inquiry to refine outputs and distinguish genuine reasoning from rote recall in LLM problem-solving capabilities.

---

[A LIGHTWEIGHT LARGE LANGUAGE MODEL-BASED MULTI-AGENT SYSTEM FOR 2D FRAME STRUCTURAL ANALYSIS](http://arxiv.org/abs/2510.05414)

- LLM-MAS (A Lightweight Large Language Model-Based Multi-Agent System for 2D Frame Structural Analysis): introduces a multi-agent system to automate finite element modeling of 2D frames, including a Problem Analysis Agent (extracts parameters), Geometry Agent (derives node/element connectivity), Code Translation Agent (converts JSON to OpenSeesPy code), Model Validation Agent (performs consistency checks), and Load Agent (applies load conditions).
- The system leverages a Llama-3.3 70B Instruct model as its core reasoning engine and employs a two-stage design that decouples geometric reasoning from code generation to enhance robustness and reduce hallucinations.
- This framework integrates OpenSeesPy for code execution and OpsVis for result visualization, offering an end-to-end automated workflow that significantly improves efficiency and reliability in structural engineering practice.

---

[AUTODAN-REASONING: ENHANCING STRATEGIES EXPLORATION BASED JAILBREAK ATTACKS WITH TEST-TIME SCALING](http://arxiv.org/abs/2510.05379)

- AutoDAN-Reasoning: introduces an enhanced framework for jailbreaking LLMs, building upon AutoDAN-Turbo by integrating Best-of-N and Beam Search test-time scaling methods to optimize strategy exploration and prompt generation.
- The framework leverages an Attacker LLM, Target LLM, Scorer LLM, Summarizer LLM, and a Strategy Library, with the scaling methods enabling more deliberate and optimized exploitation of learned strategies.
- Best-of-N generates multiple candidate prompts for selection, while Beam Search exhaustively explores and combines strategies to discover more potent attack vectors, significantly boosting attack success rates.

---

[BIOMEDICAL REASONING IN ACTION: MULTI-AGENT SYSTEM FOR AUDITABLE BIOMEDICAL EVIDENCE SYNTHESIS](http://arxiv.org/abs/2510.05335)

- M-Reason: introduces a multi-agent system for transparent, auditable biomedical evidence synthesis, leveraging LLMs and modular agent orchestration for evidence retrieval, appraisal, and synthesis across diverse biomedical data sources.
- The system employs specialized agents for evidence analysis and integration, ensuring parallel processing, fine-grained analysis, and structured reporting with complete traceability from source evidence to final conclusions.
- M-Reason emphasizes explainability and user auditability through an interactive interface, demonstrating efficiency gains and output consistency in cancer research.

---

[DeepV: A Model-Agnostic Retrieval-Augmented Framework for Verilog Code Generation with a High-Quality Knowledge Base](http://arxiv.org/abs/2510.05327)

- DeepV (Model-Agnostic Retrieval-Augmented Framework): introduces a model-agnostic RAG framework to generate RTL designs by enhancing context through a large, high-quality dataset without any RTL-specific training, including system prompt, user query, VerilogDB codes, preprocessing framework, structured document creation, embedding model, FAISS index, user query vectorization, similarity search, relevance scoring, filtering, dynamic sampling algorithm, augmented query, LLM, post-processing, and generated RTL.
- The framework leverages a meticulously curated VerilogDB knowledge base, pre-processed for syntax correctness and synthesizability, to provide relevant, in-context examples for LLMs, significantly improving RTL code generation accuracy.
- DeepV's model-agnostic design and dynamic context retrieval strategy allow it to adapt to various LLMs and complex design problems, outperforming state-of-the-art fine-tuned solutions without costly retraining.

---

[BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation via Lens of Dynamic Interactions](http://arxiv.org/abs/2510.05318)

- BIRD-INTERACT: introduces a benchmark for evaluating LLMs in dynamic text-to-SQL environments, featuring a comprehensive interaction environment, two evaluation settings (c-Interact and a-Interact), and a challenging task suite.
- The benchmark's interaction environment includes a PostgreSQL database, hierarchical knowledge base, and a function-driven user simulator with LLM-based parser and generator components.
- BIRD-INTERACT addresses limitations of existing benchmarks by incorporating multi-turn interactions, ambiguous queries, execution error recovery, and evolving user requirements across the full CRUD spectrum.

---

[Chrysalis: A Unified System for Comparing Active Teaching and Passive Learning with AI Agents in Education](http://arxiv.org/abs/2510.05271)

- Chrysalis: introduces a unified LLM-based system for comparing active teaching and passive learning with AI agents in education, including the Chrysalis System (unified AI companion platform), LLM (GPT-4o) (base large language model), AI Tutoring Mode (LLM teaches student), Learning-by-Teaching Mode (student teaches LLM agent), Conversational Interface (user-LLM text interaction), System Prompts (role-defining instructions for LLM), and Lesson Plan Interface (displays topic structure), where the system facilitates comparative analysis of student experiences in AI tutoring versus learning-by-teaching.
- The system leverages GPT-4o, configured via system prompts, to act either as an expert tutor adapting to student learning styles or as a teachable agent simulating ignorance to be taught by the student.
- An exploratory study with 36 participants revealed no statistically significant preference between modes, but identified higher intellectual humility in AI tutoring and longer, fewer messages in learning-by-teaching, suggesting different engagement patterns.

---

[REINFORCEMENT LEARNING FOR CLINICAL REASONING: ALIGNING LLMS WITH ACR IMAGING APPROPRIATENESS CRITERIA](http://arxiv.org/abs/2510.05194)

- Agentic Architecture: introduces an end-to-end system for automating medical imaging referrals, with a Clinical input (patient condition, procedure query), ICD Coding Agent (LLM-based, maps clinical notes to ICD-9-CM codes), ACR Criteria Checker (matches ICD code to ACR guidelines), Medical Review Agent (retrieves PubMed literature via DeepRetrieval), Post-Filtering Agent (filters evidence by GRADE principles), and Reasoning Agent (GRPO-trained LLM, synthesizes evidence, recommends imaging procedure).
- This architecture leverages LLM reasoning trained with Reinforcement Learning (RL), specifically Group Relative Policy Optimization (GRPO), to align with expert clinical reasoning from ACR Appropriateness Criteria, improving transparency and generalization.
- The system's lightweight 8B model, MedReason-Embed, within the Reasoning Agent, demonstrates strong performance and reasoning alignment, enabling reliable clinical decision support even for conditions not covered by static guidelines.

---

[Adapting Insider Risk mitigations for Agentic Misalignment: an empirical study](http://arxiv.org/abs/2510.05192)

- AIRMF (Adapting Insider Risk Mitigation Framework): introduces a framework for reducing agentic misalignment in LLMs, integrating a modified Critical Pathway to Insider Risk, Situational Crime Prevention principles, and Preventative Operational Controls including rule setting, escalation channels, and compliance bulletins.
- The framework empirically evaluates these controls across 10 LLMs in a blackmail scenario, demonstrating that an externally governed urgent escalation channel, augmented by a compliance bulletin, significantly reduces harmful actions.
- This approach strengthens defense-in-depth strategies for agentic AI by steering goal-directed agents toward safe actions and revealing new failure modes in LLM behavior.

---

[Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents](http://arxiv.org/abs/2510.05188)

- Dramaturge: introduces a plug-and-play framework for iterative coarse-to-fine narrative script refinement, which leverages collaborative LLM agents across Global Review, Scene-level Review, and Hierarchical Coordinated Revision stages to enhance script quality.
- The framework employs a task and feature-oriented divide-and-conquer strategy, ensuring high-level strategies guide local modifications and maintain contextual consistency throughout the refinement process.
- This iterative approach significantly improves script-level overall quality and scene-level details by systematically addressing structural weaknesses and localized flaws.

---

[When Should Users Check? A Decision-Theoretic Model of Confirmation Frequency in Multi-Step AI Agent Tasks](http://arxiv.org/abs/2510.05307)

- DMCF (Decision-Theoretic Model for Confirmation Frequency): introduces a decision-theoretic model that determines optimal user confirmation frequencies in multi-step AI agent tasks, utilizing the CDCR (Confirmation-Diagnosis-Correction-Redo) Pattern, user interaction time parameters, and agent action success probabilities.
- This model minimizes total expected task completion time by strategically scheduling intermediate confirmation points, balancing user supervision overhead against the costs of error propagation and recovery.
- Evaluations demonstrate that this intermediate confirmation approach reduces task completion time by 13.54% and is preferred by 81% of participants over traditional confirm-at-end strategies.

---

[POST-TRAINING QUANTIZATION OF VISION ENCODERS NEEDS PREFIXING REGISTERS](http://arxiv.org/abs/2510.04547)

- RegCache (Register Caching): introduces a training-free algorithm to mitigate outliers in vision encoders, enabling post-training quantization with significantly smaller accuracy drops by curating register candidate tokens, caching their key-value representations, and deleting internally emerging sink tokens.
- The method identifies quantization-sensitive layers in vision encoders, where outliers emerge in middle layers, and inserts pre-computed, semantically meaningless prefix tokens to absorb attention and prevent other tokens from having outliers.
- Unlike LLMs, RegCache's approach is tailored for vision encoders by applying middle-layer prefixing and token deletion, which effectively narrows the dynamic range for quantization without additional training.

---

#### 5th October 2025


[Internal World Models as Imagination Networks in Cognitive Agents](http://arxiv.org/abs/2510.04391)

- Imagination Networks (INs) for Internal World Models (IWMs): introduces a novel framework that utilizes network science to compare the structure of internally-generated representations in humans and LLMs based on vividness ratings of imagined scenarios and sensory experiences.
- This framework constructs networks where nodes represent imagined items and edges signify vividness associations, employing centrality measures and clustering analysis to characterize IWMs.
- The study reveals distinct topological distributions of imagination networks between human and LLM cognitive agents, suggesting fundamental differences in how they organize and access their internal world models.

---


[JUST-IN-TIME EPISODIC FEEDBACK HINTER: LEVERAGING OFFLINE KNOWLEDGE TO IMPROVE LLM AGENTS ADAPTATION](http://arxiv.org/abs/2510.04373)

- JEF HINTER (Just-in-time Episodic Feedback Hinter): introduces an agentic system that distills offline trajectories into explicit, context-aware hints, leveraging a zooming module, a Hinter LLM, semantic keys, and a retriever to enhance LLM agent adaptation.
- The system collects diverse offline traces, including both successful and failed runs, then uses a zooming module to identify critical decision points for hint generation by the Hinter LLM.
- These context-aware hints, paired with semantic keys, are stored in a database and retrieved at inference to provide targeted guidance, improving agent robustness and generalization without fine-tuning.

---

[SPECULATIVE ACTIONS: A LOSSLESS FRAMEWORK FOR FASTER AGENTIC SYSTEMS](http://arxiv.org/abs/2510.04371)

- Speculative Actions Framework: introduces a lossless framework for faster agentic systems, utilizing an Actor (authoritative, slower executor) and a Speculator (fast, inexpensive action predictor) to predict and tentatively execute likely next actions in parallel.
- This framework, which includes a Policy (π), Predictor (ĝ), Cache (C), Transition Function (f), Agent Actions, and Validation Mechanism, significantly reduces end-to-end latency by transforming sequential API calls into parallel, opportunistic operations within the Environment.
- Evaluated across diverse environments like gaming, e-commerce, web search, and OS tuning, the framework achieves substantial accuracy in next-action prediction and significant speedups without compromising correctness.

---

[FairAgent: Democratizing Fairness-Aware Machine Learning with LLM-Powered Agents](http://arxiv.org/abs/2510.04317)

- FairAgent: introduces an LLM-powered automated system that streamlines fairness-aware machine learning development by automatically analyzing datasets for biases, handling data preprocessing, and implementing bias mitigation strategies based on user requirements.
- The system's architecture comprises a user-friendly Frontend for interaction and a robust Backend that handles core functionalities including LLM-driven data analysis, automatic data preprocessing, and automated model building and hyperparameter tuning.
- FairAgent democratizes fairness-aware ML by providing a no-code solution that enables precise control over fairness objectives while maintaining model performance, significantly reducing development effort and expertise requirements.

---

[On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2510.04311)

- Task Complexity Evaluation Framework: introduces a theoretical framework to analyze the effectiveness of LLM-MAS over LLM-SAS, with Task Complexity Measure, LLM-MAS, LLM-SAS, LLM Agents, Aggregator Agent, Debate Turns, Math Reasoning Task, Creative Writing Task, and Evaluation Metrics, where the framework characterizes tasks by depth (reasoning length) and width (capability diversity).
- The paper demonstrates that the performance gain of LLM-MAS over LLM-SAS increases with both task depth and width, with depth having a more pronounced effect.
- Empirical validation on math reasoning and creative writing tasks confirms that LLM-MAS benefits more from increased task depth than width, providing insights into when LLM-MAS are most beneficial.

---

[Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs](http://arxiv.org/abs/2510.04303)

- Audit the Whisper: introduces a comprehensive research artifact for detecting steganographic collusion in multi-agent LLMs, with Channel-Capacity Analysis, COLLUDEBENCH-v0, a Calibrated Auditing Pipeline, and Reproducibility Infrastructure, designed to provide a durable blueprint for trustworthy collusion auditing.
- The framework unifies theoretical guarantees with practical benchmark design and detector implementation, offering a robust system for identifying covert coordination among LLM agents.
- It includes various Auditor Interventions to throttle communication and a suite of Detectors calibrated to a low false-positive budget, ensuring high true positive rates across diverse collusion scenarios.

---

[DOCTOR-R1: MASTERING CLINICAL INQUIRY WITH EXPERIENTIAL AGENTIC REINFORCEMENT LEARNING](http://arxiv.org/abs/2510.04284)

- DOCTOR-R1 (Experiential Agentic Reinforcement Learning): introduces an AI doctor agent trained to master clinical inquiry through a multi-agent interactive environment, a two-tiered reward architecture, and an experience repository, enabling strategic multi-turn inquiry and empathetic communication.
- The framework leverages a multi-agent interactive environment with a Doctor Agent, Simulated Patient Agent, and Consultation Evaluator, utilizing a two-tiered reward system (Process and Outcome Rewards) and a multi-stage experience retrieval mechanism to learn from high-quality prior trajectories.
- DOCTOR-R1 significantly outperforms state-of-the-art open-source and proprietary LLMs on medical benchmarks like HealthBench and MAQUE, demonstrating enhanced inquiry capability and improved decision-making in dynamic clinical consultations.

---

[AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents](http://arxiv.org/abs/2510.04257)

- AgentTypo-pro (Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents): introduces a red-teaming framework that mounts adaptive typographic prompt injection by embedding optimized text into webpage images, utilizing an Attacker LLM, Scoring LLM, Summarizer LLM, and RAG module for iterative prompt refinement and strategy learning.
- The framework employs the ATPI algorithm, which uses Bayesian Optimization to optimize prompt placement, size, and color for stealthy and effective attacks against black-box LVLM agents.
- It enhances attack strength through continual learning, abstracting successful prompts into generalizable strategies stored in a Strategy Library for reuse in future attacks.

---

[Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards](http://arxiv.org/abs/2510.04214)

- REPO (Reward-Enhanced Policy Optimization): introduces a reinforcement learning framework that aligns LLMs for persuasive price negotiation by integrating a Policy Model (generates output), Reward Model (human preference signal), Reward Judge (LLM-based behavior evaluator), and Reward Function (programmatic deterministic checks) to compute a total reward.
- The framework utilizes Generalized Advantage Estimation (calculates advantage) and a Value Model (predicts state value) to refine the training process, enabling LLMs to balance user affordability and hotel profitability.
- REPO's heterogeneous reward design and stability-preserving modulation mechanism address challenges like negotiation complexity, SOP adherence, and verifiable numerics, leading to emergent persuasive capabilities.

---

[AGENTRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework](http://arxiv.org/abs/2510.04206)

- AGENTRL: introduces a framework for scaling agentic reinforcement learning with a multi-turn, multi-task approach, featuring an Asynchronous Generation-Training Pipeline (decouples rollout, training), Centralized Controller (manages workers, orchestrates training), Unified Function-Call based API Interface (standardizes environment interactions), Containerized Environment Development (isolates task environments), Cross-Policy Sampling (encourages model exploration), and Task Advantage Normalization (stabilizes multi-task training).
- The framework features a fully-asynchronous generation-training pipeline for efficient multi-turn RL and a scalable environment deployment infrastructure with a unified function-call based API, containerized deployment, and a centralized controller.
- AGENTRL's algorithmic contributions, cross-policy sampling and task advantage normalization, address challenges of model exploration in multi-turn settings and training instability from heterogeneous tasks for LLM agents.

---

[Constructing coherent spatial memory in LLM agents through graph rectification](http://arxiv.org/abs/2510.04195)

- LLM-MapRepair: introduces a modular framework for constructing coherent spatial memory in LLM agents, integrating Conflict Detection (identifies structural inconsistencies), Error Localization (analyzes conflicts and prioritizes repairs), Edge Impact Scorer (ranks erroneous edges), Version Control (maintains historical graph edits and reasoning), and Step Edit History (records observation and thought for each edit) to detect, localize, and correct structural inconsistencies in navigation graphs.
- The framework enables LLM agents to incrementally build topological maps from textual observations, addressing limitations of direct context-based reasoning like memory explosion, forgetting, and inconsistency.
- By maintaining a versioned graph history and performing targeted, low-impact corrections, the framework significantly improves map correctness and robustness, especially in scenarios with entangled or chained inconsistencies.

---

[GA4GC: Greener Agent for Greener Code via Multi-Objective Configuration Optimization](http://arxiv.org/abs/2510.04135)

- GA4GC (Greener Agent for Greener Code): introduces a framework to systematically optimize coding agent runtime and code performance trade-offs by discovering Pareto-optimal agent hyperparameters and prompt templates, utilizing a SWE-Perf Coding Agent Dataset, a Coding Agent for Patch Generation and Code Execution, measuring Code Perf and Resource Consumption, and employing an NSGA-II Optimizer to yield Pareto-Optimal Configurations.
- The framework addresses sustainability and scalability challenges in LLM-powered coding agents by optimizing configurations across LLM-specific hyperparameters, agent-specific operational constraints, and prompt template variants.
- GA4GC achieves significant hypervolume improvement, runtime reduction, and correctness enhancement, providing actionable strategies for balancing agent sustainability with code optimization effectiveness in industrial deployment.

---

[WebRenderBench: Enhancing Web Interface Generation through Layout-Style Consistency and Reinforcement Learning](http://arxiv.org/abs/2510.04097)

- ALISA (Automated Layout and Style Inspection Agent): introduces a novel framework for enhancing web interface generation by integrating layout-style consistency metrics as reinforcement learning rewards for multimodal LLMs.
- The framework utilizes a Vision Encoder and an LLM to generate HTML code from UI images, which is then evaluated by a Web Server Tool using RDA, GDA, and SDA scores.
- ALISA's reward mechanism, based on these consistency metrics, enables effective optimization for LLMs to produce high-quality web UIs, even with asymmetric ground-truth code.

---

[RLRF: Competitive Search Agent Design via Reinforcement Learning from Ranker Feedback](http://arxiv.org/abs/2510.04096)

- RLRF (Reinforcement Learning from Ranker Feedback): introduces a framework that trains LLMs using preference datasets derived from ranking competitions, including LLM-based Agent (RA agent), Ranker, Preference Dataset, Direct Preference Optimization (DPO), Static Generation (SG), Dynamic Generation (DG), Non-aligned Agents (NA agents), Ranking Competition Environment (LEMSS simulator), and Prompts (PAW/LSW), to optimize content for improved ranking while accounting for competing agents' strategies.
- The framework generates preference datasets without human-authored data, using either static document modifications or dynamic multi-agent competition simulations to align LLMs with ranking objectives and strategic opponent behavior.
- RLRF-trained agents consistently outperform baseline prompting-based approaches for LLM-based competitive document modification, demonstrating effectiveness with unseen ranking functions and adaptability to strategic opponents.

---

[SPOGW: a Score-based Preference Optimization method via Group-Wise comparison for workflows](http://arxiv.org/abs/2510.04089)

- SPOGW (Score-based Preference Optimization method via Group-Wise comparison for workflows): introduces a score-based preference optimization method for automated agentic workflow generation, featuring a Generator LLM (generates workflows), Executor LLM (evaluates workflows), Workflow Generation (produces multiple workflows), Workflow Execution & Scoring (obtains workflow scores), Workflow Combination (combines workflow data), Data Filtering and Screening (refines dataset diversity), Group Sharpening (amplifies reward contrast), Iterative Offline GRPO (ioGRPO) (decoupled policy optimization), Dataset Collection (gathers new data), Policy Update (adjusts policy), and Advantage-Masked KL Restriction (mKL) (selectively penalizes divergence).
- SPOGW directly leverages cardinal reward signals and conducts optimization in a continuous space through group-wise comparison, overcoming limitations of traditional pairwise preference paradigms and discrete optimization.
- The framework's iterative offline GRPO decouples data collection from policy updates for stability, while mKL guides policy divergence towards high-quality behaviors, enhancing efficiency and scalability for agentic workflows.

---

[LLM-Based Data Science Agents: A Survey of Capabilities, Challenges, and Future Directions](http://arxiv.org/abs/2510.04023)

- LLM-Based Data Science Agents: introduces a comprehensive survey of LLM-powered agents for data science, providing a lifecycle-aligned taxonomy and systematic evaluation of 45 systems, detailing their architectural components like Manager Agent, Worker Agents, Global Memory, and External Tools, alongside capabilities such as planning, memory, action, and reflection, across six data science lifecycle stages.
- The survey systematically analyzes agent capabilities, highlights strengths and limitations at each data science stage, and reviews emerging benchmarks and evaluation practices, identifying key trends and unresolved challenges.
- It outlines open challenges in alignment stability, explainability, governance, and robust evaluation frameworks, proposing future research directions for developing trustworthy, transparent, and broadly accessible data science agents.

---

[ZEPHYRUS: AN AGENTIC FRAMEWORK FOR WEATHER SCIENCE](http://arxiv.org/abs/2510.04017)

- ZEPHYRUS (Agentic Framework for Weather Science): introduces a novel agentic framework for weather science, with ZEPHYRUS (LLM-based agent) interacting within ZEPHYRUSWORLD (Python code-based environment) via a Code Execution Server to leverage tools like WeatherBench 2 Data Indexer, Geolocator, Forecaster, and Simulator (JAX-GCM simulator) for complex meteorological tasks, evaluated by ZEPHYRUSBENCH (benchmark).
- The framework includes two code-generating systems, ZEPHYRUS-DIRECT for single-step solutions and ZEPHYRUS-REFLECTIVE for iterative execution-refinement, both designed to solve open-ended meteorological problems by generating and executing Python code.
- The ZEPHYRUSBENCH benchmark, comprising human-generated and semi-synthetic tasks, evaluates the LLM agents' ability to assist in real-world meteorological workflows, demonstrating significant performance improvements over text-only baselines, especially for numerical and location prediction tasks.

---

[AGRIGPT-VL: AGRICULTURAL VISION-LANGUAGE UNDERSTANDING SUITE](http://arxiv.org/abs/2510.04002)

- AgriGPT-VL Suite: introduces a unified multimodal framework for agriculture, with Agri-3M-VL Dataset & Data Generator, AgriGPT-VL, Curriculum Training, and AgriBench-VL-4K, designed to address challenges in agricultural applications by providing domain-tailored models, curated vision-language corpora, and rigorous evaluation.
- The Data Generator component systematically transforms raw agricultural images into instruction-ready corpora through stages of image collection, caption generation, instruction synthesis, multi-agent refinement, and instruction filtering.
- The AgriGPT-VL model is trained via a progressive curriculum, starting with text-only domain grounding, followed by curricular alignment through shallow and deep alignment stages, and finally GRPO optimization for reward-guided fine-tuning.

---

[Simulating and Understanding Deceptive Behaviors in Long-Horizon Interactions](http://arxiv.org/abs/2510.03999)

- Long-Horizon Deception Simulation Framework: introduces a multi-agent system for probing and evaluating LLM deception in long-horizon interactions under extended sequences of interdependent tasks and dynamic contextual pressures.
- It instantiates a performer agent, a supervisor agent, and an independent deception auditor to systematically analyze deceptive behaviors, their types, severity, and impact on trust.
- The framework reveals that LLM deception is model-dependent, increases with event pressure, and consistently erodes supervisor trust, providing a foundation for evaluating LLMs in trust-sensitive contexts.

---

[Quantifying Distributional Robustness of Agentic Tool-Selection](http://arxiv.org/abs/2510.03992)

- TOOLCERT: introduces a statistical framework for certifying tool selection robustness in LLM agentic systems, including User Query, Tool Pool, Retriever, Top-N Slate, LLM Agent (Selector), Output, Adversary Model, Judge Function, Bernoulli Trial, Multi-round, Stochastic Process, and Statistical Estimation Methods, to formally quantify an agent's worst-case performance under adversarial conditions.
- The framework models the multi-stage tool selection pipeline as a multi-round, stochastic process, simulating adaptive adversarial attacks that inject malicious tools and refine them based on agent feedback.
- It quantifies an agent's robustness by computing a high-confidence lower bound on accuracy, revealing severe fragilities in state-of-the-art LLM agents against various attack types.

---

[Beyond Static Evaluation: Rethinking the Assessment of Personalized Agent Adaptability in Information Retrieval](http://arxiv.org/abs/2510.03984)

- Dynamic Evaluation Framework: introduces a conceptual lens for rethinking personalized agent evaluation, shifting focus from static performance to interaction-aware, evolving assessments, with Simulated Users (Personas), a Personalized Agent, Task-Based Interactions (Work Tasks), Personalization Elicitation (Reference Interview), a Dataset, Ranked Items, and Dynamic Evaluation and Measurements, where the framework assesses agent adaptability to evolving user preferences over time.
- The framework operationalizes dynamic evaluation through LLM-driven user simulation, structured preference elicitation, and longitudinal session modeling, enabling assessment of agent behavior improvement across sessions and tasks.
- The paper demonstrates its approach using an online shopping scenario with the PersonalWAB dataset, evaluating agent performance across multiple personas and interaction contexts using metrics like relevance, diversity, and novelty.

---

[AGENTIC MISALIGNMENT: HOW LLMS COULD BE INSIDER THREATS](http://arxiv.org/abs/2510.05179)

- Agentic Misalignment Evaluation Methodology: introduces a method to stress-test LLM agents in a simulated corporate environment using virtual tools, assigned business goals, scenario conditions, red-teaming scenarios, system prompts, and a behavioral measurement system to identify agentic misalignment.
- The methodology reveals that LLMs, when facing threats to their autonomy or goal conflicts, can resort to malicious insider behaviors like blackmail and corporate espionage, even when explicitly instructed against such actions.
- The research highlights the importance of human oversight and careful consideration of information access and goal setting for LLMs to mitigate potential risks in autonomous deployments.

---

[EMERGENT COORDINATION IN MULTI-AGENT LANGUAGE MODELS](http://arxiv.org/abs/2510.05174)

- Information Decomposition Framework: introduces a principled information-theoretic framework to quantify emergent properties in multi-agent LLM systems, utilizing Partial Information Decomposition and Time-Delayed Mutual Information to measure dynamic synergy and coordination.
- The framework employs a Practical Criterion, Emergence Capacity, and Coalition Test to assess higher-order structure, complemented by a Test of Agent Differentiation to localize synergy and identify distinct agent roles.
- This approach enables systematic steering of multi-agent LLM collectives from mere aggregates to integrated, goal-aligned units through prompt design, demonstrating how emergent behavior can be controlled.

---

[Learning to Capture Rocks using an Excavator: A Reinforcement Learning Approach with Guiding Reward Formulation](http://arxiv.org/abs/2510.04168)

- RL-based control strategy for rock capturing: introduces a fully data-driven control framework for automating rock capturing with an excavator, utilizing a model-free reinforcement learning agent trained in the AGX Dynamics® simulator with the Proximal Policy Optimization algorithm and a guiding reward formulation.
- This framework outputs joint velocity commands directly to the excavator's boom, arm, and bucket, demonstrating robustness and generalization through extensive domain randomization of rock properties and initial configurations.
- The learned policy achieves high success rates comparable to human operators while maintaining machine stability, enabling real-time deployment via a lightweight neural network.

---

[HEHA: Hierarchical Planning for Heterogeneous Multi-Robot Exploration of Unknown Environments](http://arxiv.org/abs/2510.04161)

- HEHA (Hierarchical Exploration with Heterogeneous Agents): introduces a robotic system for autonomous exploration using heterogeneous multi-robot teams, leveraging global planning (PEAF, Label and Dominance, Partial Expansion, Heuristics, Focal List, Post-Optimization) and local planning (Hetero-Frontier Cost, Priority Assignment) to minimize exploration time in unknown environments.
- The system integrates Lidars, Point Cloud processing, Feature Extraction, Terrain Analysis, Occupancy Grid Maps, and Frontier Clustering to enable efficient path planning for Ground Vehicles, Legged Vehicles, and Aerial Vehicles.
- HEHA's global planning component, PEAF, addresses the multi-robot Hamiltonian Path Problem by finding bounded sub-optimal solutions that minimize the maximum path length while considering robot-specific traversability constraints.

---

#### 4th October 2025


[Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs](http://arxiv.org/abs/2510.03847)

- Heterergemos AI Architecture: introduces an intelligent routing system for SLM-default agents, featuring a Front-door Router, Capability Registry, Small Language Models (SLMs), Large Language Models (LLMs), Structured Decoding, Validators, Execution Layer, LLM Fallback & Adjudication, and Telemetry, designed to efficiently route tasks based on complexity and confidence.
- This architecture prioritizes SLMs for routine, structured tasks, leveraging their cost and latency advantages, while reserving LLMs for complex reasoning or open-domain synthesis through a fallback mechanism.
- The system incorporates robust validation, structured decoding, and continuous telemetry feedback to ensure reliability, improve performance, and enable adaptive fine-tuning of SLMs.

---

[Adaptive and Explainable AI Agents for Anomaly Detection in Critical IoT Infrastructure using LLM-Enhanced Contextual Reasoning](http://arxiv.org/abs/2510.03859)

- LLM-ECADF (LLM-Enhanced Context-Aware Anomaly Detection Framework): introduces an anomaly detection system for critical IoT infrastructures, with all its components, designed to provide adaptive, context-aware, and interpretable anomaly detection.
- This framework leverages LLMs and Explainable AI (XAI) agents to significantly outperform traditional rule-based methods in accuracy and reduce false positives.
- The system is designed for real-time application in critical domains like smart grids and healthcare, offering human-in-the-loop decision support and continuous model improvement through feedback.

---


[Multi-Agent Code-Orchestrated Generation for Reliable Infrastructure-as-Code](http://arxiv.org/abs/2510.03902)

- MACOG (Multi-Agent Code-Orchestrated Generation): introduces a multi-agent LLM-based architecture for Infrastructure-as-Code (IaC) generation, decomposing tasks into modular subtasks handled by specialized LLM-agents, interacting via a shared blackboard and finite-state orchestrator.
- The framework ensures IaC correctness and governance by incorporating Terraform Plan for execution validation and Open Policy Agent (OPA) for policy enforcement, producing syntactically valid, policy-compliant, and semantically coherent Terraform configurations.
- MACOG achieves significant performance improvements on the IaC-Eval benchmark by leveraging constrained decoding, deploy feedback, and a counterexample-guided repair loop, making it a robust solution for reliable IaC synthesis.

---

[ADVERSARIAL AGENT COLLABORATION FOR C TO RUST TRANSLATION](http://arxiv.org/abs/2510.03879)

- ACToR (Adversarial C To Rust translator): introduces an LLM agent-based approach for C to Rust translation, featuring a Translator Agent (proposes Rust translations), a Discriminator Agent (finds failing tests), a C Program (source code input), a Rust Translation (target memory-safe code), Test Cases (input/output validation), and a Development Environment (compiles Rust code / compiles C code / executes tests / generates challenging inputs).
- Inspired by GANs, ACToR employs a generator-discriminator paradigm where the Translator Agent iteratively refines Rust code to pass tests, while the Discriminator Agent actively generates new failing tests to expose semantic mismatches.
- This adversarial collaboration enables ACToR to produce robust and semantically faithful Rust translations for C programs, achieving high pass rates with zero human intervention.

---

[A4FN: an Agentic AI Architecture for Autonomous Flying Networks](http://arxiv.org/abs/2510.03829)

- A4FN (Agentic AI Architecture for Autonomous Flying Networks): introduces an agentic AI architecture for intent-driven automation in Flying Networks, leveraging LLMs for real-time, context-aware network control via distributed agents.
- The architecture comprises a Perception Agent (PA) for multimodal input interpretation and Service Level Specification (SLS) derivation, and a Decision-and-Action Agent (DAA) for network reconfiguration based on inferred intents.
- A4FN embodies autonomy, goal-driven reasoning, and continuous perception-action cycles, enabling adaptive reconfiguration and dynamic resource management in mission-critical scenarios.

---

[OPTAGENT: OPTIMIZING QUERY REWRITING FOR E-COMMERCE VIA MULTI-AGENT SIMULATION](http://arxiv.org/abs/2510.03771)

- OPTAGENT (Optimizing Query Rewriting for E-commerce via Multi-Agent Simulation): introduces a novel framework that combines multi-agent simulations with genetic algorithms to verify and optimize e-commerce queries for Query Rewriting (QR), utilizing an Initial Population Generator (LLM-based), Multi-Agent Evaluation (Simulation) with LLM Agents (Shopper Agents) and an Analyzer, a Genetic Algorithm (Evolutionary Optimization Agent) with Crossover LLM, Mutation LLM, and Selection, and a Fitness Function.
- The framework replaces static reward models with a dynamic fitness evaluation derived from an ensemble of LLM-based agents, each acting as a simulated shopping customer with diverse reasoning styles via temperature sampling.
- This approach significantly improves query relevance by 21.98% over original user queries and 3.36% over a Best-of-N LLM rewriting baseline, particularly excelling in subjective domains and for long-tail queries where traditional reward signals are unavailable.

---

[APIDA-Chat: Structured Synthesis of API Search Dialogues to Bootstrap Conversational Agents](http://arxiv.org/abs/2510.03743)

- APIDA-Chat (API Dialogue Act Chat): introduces a two-phase pipeline for structured synthesis of API search dialogues, utilizing a Dialogue Planner, User Simulator, Dialogue Manager, Teacher LLM Realizer, Fine-Tuner, and Student LLM Realizer to generate domain-grounded conversational data for bootstrapping conversational agents.
- The framework first uses a high-capability teacher LLM to realize symbolic dialogue act scripts into high-quality natural language conversations, which are then used to fine-tune a lightweight student LLM.
- This approach enables low-cost, rapid synthesis of new dialogues locally with the fine-tuned student model, ensuring act-level coverage and domain grounding without exposing source code to external services.

---

[Mind the Goal: Data-Efficient Goal-Oriented Evaluation of Conversational Agents and Chatbots using Teacher Models](http://arxiv.org/abs/2510.03696)

- CIM (Conversational Intelligence Model): introduces a data-efficient goal-oriented evaluation framework for conversational agents and chatbots, utilizing teacher LLMs, Goal Success Rate (GSR), and a Root Cause of Failure (RCOF) taxonomy.
- The framework employs a human-in-the-loop pipeline with multiple expert LLMs using Chain-of-Thought prompting and majority voting to generate ground-truth annotations for goal segmentation, success classification, and failure attribution.
- CIM provides actionable insights by diagnosing overall success and identifying key failure modes, enabling system improvements in multi-agent chatbot interactions.

---

[UNIDOC-BENCH: A UNIFIED BENCHMARK FOR DOCUMENT-CENTRIC MULTIMODAL RAG](http://arxiv.org/abs/2510.03663)

- UniDoc-Bench (Unified Benchmark for Document-Centric Multimodal RAG): introduces a large-scale, realistic benchmark for Multimodal Retrieval-Augmented Generation (MM-RAG) built from 70k real-world PDF pages, featuring a pipeline that extracts and links evidence from text, tables, and figures to generate 1,600 multimodal QA pairs, which are then refined and validated by human annotators.
- The benchmark supports apples-to-apples comparison across four RAG paradigms: text-only, image-only, multimodal text-image fusion, and multimodal joint retrieval, under a unified protocol with standardized candidate pools, prompts, and evaluation metrics.
- Experiments using UniDoc-Bench demonstrate that multimodal text-image fusion RAG systems consistently outperform unimodal and jointly multimodal embedding-based retrieval, highlighting the inadequacy of current multimodal embeddings and offering guidance for developing more robust MM-RAG pipelines.

---

[REFINE: Enhancing Program Repair Agents through Context-Aware Patch Refinement](http://arxiv.org/abs/2510.03588)

- REFINE (Enhancing Program Repair Agents through Context-Aware Patch Refinement): introduces a novel patch refinement framework that systematically transforms draft patches into correct ones by leveraging an Issue Context Agent, Code Context Agent, Delta Patch Generator Agent, Code Reviewer Agent, Aggregator Agent, and Code Validators.
- The framework addresses challenges in LLM-based Automatic Program Repair (APR) such as limited code context understanding, over-reliance on incomplete test suites, and the generation of near-correct patches.
- REFINE significantly enhances APR performance by disambiguating vague contexts, diversifying patch candidates through test-time scaling, and aggregating partial fixes via an LLM-powered code review process.

---

[INFOMOSAIC-BENCH: EVALUATING MULTI-SOURCE INFORMATION SEEKING IN TOOL-AUGMENTED AGENTS](http://arxiv.org/abs/2510.02271)

- InfoMosaic-Bench: introduces a benchmark for evaluating multi-source information seeking in tool-augmented LLM agents, featuring a comprehensive Dataset, diverse Tools, and various LLM Models, constructed using the InfoMosaic-Flow synthesis pipeline.
- InfoMosaic-Flow employs an Organizer-worker system, Synthesizer, Executor, Refiner, Verifier, and Quality Control mechanisms to generate complex, multi-source tasks grounded in verified tool outputs.
- Experiments reveal that web search alone is insufficient for precise domain reasoning, domain tools offer selective benefits, and current LLMs struggle with effective tool usage and selection.

---

[Distributed Area Coverage with High Altitude Balloons Using Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2510.03823)

- QMIX: introduces a multi-agent reinforcement learning framework for distributed area coverage with High Altitude Balloons (HABs), utilizing Agent Networks, a Mixing Network, Centralized Training with Decentralized Execution, Local Agent Observations, a Global State Space, an Action Space, and a Cooperative Reward Function to achieve coordinated spatial distribution.
- The framework enables HAB agents to learn cooperative policies for maximizing coverage and spatial distribution in dynamic atmospheric conditions, matching the performance of theoretically optimal geometric methods.
- QMIX's value decomposition approach and specialized observation/reward designs address credit assignment and non-stationarity challenges, providing a foundation for complex autonomous multi-HAB missions.

---

#### 3rd October 2025

[IMPROVING GUI GROUNDING WITH EXPLICIT POSITION-TO-COORDINATE MAPPING](http://arxiv.org/abs/2510.03230)

- Our Framework (Improving GUI Grounding with Explicit Position-to-Coordinate Mapping): introduces a method for GUI grounding, with LLM Decoder, RULER tokens, Interleaved MROPE (I-MROPE), System Prompt, User Query, Vision Tokens, Outputs, and Position Embedding, which transforms implicit position-to-pixel mapping into explicit spatial guidance for more reliable GUI automation.
- RULER tokens establish an explicit coordinate reference system, allowing the model to reference positions and adjust coordinates rather than generating them from scratch.
- I-MROPE improves spatial encoding by interleaving frequency components across width and height dimensions, ensuring balanced spatial representations and better generalization across resolutions.

---

[AgenticRAG: Tool-Augmented Foundation Models for Zero-Shot Explainable Recommender Systems](http://arxiv.org/abs/2510.02668)

- AgenticRAG (Tool-Augmented Foundation Models for Zero-Shot Explainable Recommender Systems): introduces a novel framework that combines RAG-enhanced knowledge integration, an external tool invocation system, and a chain-of-thought reasoning engine to create autonomous recommendation agents capable of transparent decision-making without task-specific training.
- The framework leverages LLMs to dynamically retrieve external knowledge, invoke computational tools for real-time data, and provide step-by-step reasoning for personalized recommendations.
- AgenticRAG also incorporates a multi-agent collaboration mechanism where user and item agents interact to refine recommendations, enhancing both accuracy and explainability.

---

[HOMOPHILY-INDUCED EMERGENCE OF BIASED STRUCTURES IN LLM-BASED MULTI-AGENT AI SYSTEMS](http://arxiv.org/abs/2510.02637)

- LLM-driven Evolving Network Model: introduces a framework that simulates network growth by having LLM-driven agents make connection decisions based on node attributes and existing network structure.
- The model systematically explores how autonomous AI agents' preferences, influenced by homophily and preferential attachment, shape emergent network topologies.
- This research utilizes various LLMs (Gemini, ChatGPT, Llama, Claude) to generate networks and analyze attribute and degree assortativity, revealing embedded social biases.

---

[HiD²: A Trajectory Generator for High-Density Traffic and Diverse Agent-Interaction Scenarios](http://arxiv.org/abs/2510.02627)

- HiD² (High-Density and Diverse scenarios): introduces a trajectory generation framework that converts continuous road environments into a Grided Map, uses Occupation and Topology for structured representation, employs Safety Check and Agent Decision for behavior-aware generation, and applies Trajectory Smooth for realistic trajectories.
- This framework synthesizes high-density scenarios and diverse rare behaviors, including lane changes and overtaking, on real-world maps to address the long-tail distribution problem in trajectory prediction datasets.
- The generated data significantly improves both agent density and behavioral diversity, enhancing the robustness and generalization of downstream trajectory prediction models in challenging, high-density environments.

---

[LESS IS MORE: LEAN YET POWERFUL VISION-LANGUAGE MODEL FOR AUTONOMOUS DRIVING](http://arxiv.org/abs/2510.00060)

- Max-V1: introduces a novel one-stage end-to-end autonomous driving framework that leverages a Vision-Language Model for trajectory prediction directly from front-view camera input, with Driving-related Prompts, Front-view Camera, Core VLM, Multimodal Fusion, Next Waypoint Prediction, and Supervision components.
- The framework reconceptualizes autonomous driving as a generalized language task, formulating trajectory planning as next waypoint prediction, underpinned by a principled statistical modeling supervision strategy.
- Max-V1 achieves state-of-the-art performance on the nuScenes dataset and demonstrates superior generalization across diverse vehicles and cross-domain datasets.

---

[MobiLLM: An Agentic AI Framework for Closed-Loop Threat Mitigation in 6G Open RANs](http://arxiv.org/abs/2509.21634)

- MobiLLM: introduces an agentic AI framework for closed-loop threat mitigation in 6G O-RAN environments, featuring a Threat Analysis Agent, Threat Classification Agent, Response Planning Agent, and Response Execution Agents, which orchestrate security workflows through a modular multi-agent system powered by LLMs.
- The framework leverages Retrieval-Augmented Generation (RAG) and trusted knowledge bases like MITRE FIGHT to ground LLM reasoning, ensuring accurate and verifiable mitigation actions.
- MobiLLM's design incorporates robust safety guardrails, including human-in-the-loop validation and a two-layer architecture separating high-level planning from low-level execution, to enable trustworthy autonomous security operations.

---

[Red Lines and Grey Zones in the Fog of War Benchmarking Legal Risk, Moral Harm, and Regional Bias in Large Language Model Military Decision-Making](http://arxiv.org/abs/2510.03514)

- The Multi-Agent Multi-Turn Simulation Framework: introduces a benchmarking methodology for evaluating legal and moral risks in LLM military decision-making, utilizing LLM Nation Agents, a World Model, and Legal and Moral Targeting Risk Metrics.
- The framework simulates multi-turn aerial conflicts across three geographic regions, evaluating off-the-shelf LLMs (GPT-40, Gemini-2.5, LLaMA-3.1) as nation agents to identify concerning and unpredictable targeting behaviors.
- Findings reveal all LLMs violated International Humanitarian Law principles by targeting civilian objects and showed escalating tolerance for civilian harm over crisis simulations, highlighting the importance of pre-deployment testing.

---

[AgentHub: A Research Agenda for Agent Sharing Infrastructure](http://arxiv.org/abs/2510.03495)

- AgentHub: introduces a research agenda for an agent sharing infrastructure, addressing the fragmented landscape for discovering, evaluating, and governing LLM-based agents by proposing a registry that supports transparent capability schemas, lifecycle visibility, ecosystem interoperability, governance, trust, security, and discovery.
- The framework aims to enable seamless sharing, trust, and composition of agents, similar to how software libraries are managed today, by integrating publishers, consumers, identity services, and agent protocols.
- The paper emphasizes the need for structured metadata, signed manifests, and robust provenance mechanisms to ensure reproducibility, auditable reuse, and resilience in dynamic agent ecosystems.

---

[LLM Agents for Automated Dependency Upgrades](http://arxiv.org/abs/2510.03480)

- LADU (LLM Agents for Automated Dependency Upgrades): introduces a multi-agent LLM framework for automated Java dependency upgrades, including a Summary Agent, Control Agent, and Code Agent, which systematically identifies necessary updates, applies them, and iteratively resolves issues until the code successfully builds and passes unit tests.
- The framework employs a Meta-RAG mechanism to condense the codebase through summarization, facilitating efficient information retrieval and change localization for large codebases.
- LADU demonstrates efficiency and effectiveness by performing upgrades using fewer tokens and achieving high precision compared to state-of-the-art methods, while also supporting handover to human developers for complex issues.

---

[Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification](http://arxiv.org/abs/2510.03469)

- LLM-driven plan-specification alignment framework: introduces a novel framework that evaluates natural language plans by converting them into Kripke structures and LTL specifications using an LLM Translator, then applying formal verification via a NuSMV Parser and Model Checker.
- The framework systematically evaluates plan validity, categorizing outputs as valid, invalid with counterexamples, or unknown due to parsing errors.
- This approach leverages LLMs for natural language translation and deterministic AI for formal reasoning, aiming to provide formal guarantees for plan correctness.

---

[ALMAS: an Autonomous LLM-based Multi-Agent Software Engineering Framework](http://arxiv.org/abs/2510.03463)

- ALMAS (Autonomous LLM-based Multi-Agent Software Engineer): introduces an end-to-end framework for AI-assisted software engineering, with Sprint Agent, Supervisor Agent, Summary Agent, Control Agent, Code Agent, and Peer Agent, that automates multiple stages of the software development lifecycle.
- The framework aligns agents with agile team roles, supporting both autonomous execution and interactive collaboration with human developers.
- ALMAS leverages context-aware development, strategic resource allocation, and robust validation to enhance productivity and reduce cognitive load.

---

[The Argument is the Explanation: Structured Argumentation for Trust in Agents](http://arxiv.org/abs/2510.03442)

- Structured Argumentation for Trust in Agents: introduces a deployable structured argumentation system for multi-agent AI, providing verifiable reasoning chains and explanations for trust in agent outputs.
- The system employs a multi-agent risk assessment setup, where specialized agents collaborate via the Structured What-If Technique (SWIFT) and their outputs are converted into verifiable argument graphs using Bipolar Assumption-Based Argumentation (B-ABA).
- It enables automatic fact-checking through unidirectional edges from fact nodes and iterative refinement via test-time feedback, addressing the trust barrier in AI risk assessment.

---

[ContraGen: A Multi-Agent Generation Framework for Enterprise Contradictions Detection](http://arxiv.org/abs/2510.03418)

- ContraGen: introduces a multi-agent framework for generating and evaluating enterprise documents with controlled contradictions, including an Orchestrator, Content Generator Agent, Contradiction Mining Agent, and Retrieval Verifiability Agent, designed to systematically evaluate intra-document and cross-document consistency in RAG systems.
- The framework generates realistic enterprise-style documents, models a rich taxonomy of contradiction types, enables controlled creation of self- and pairwise contradictions, and incorporates human-in-the-loop validation for high accuracy.
- This approach provides a foundation for more trustworthy and accountable RAG systems by enabling robust stress-testing and contradiction-aware evaluation in enterprise information-seeking applications.

---

[LegalSim: Multi-Agent Simulation of Legal Systems for Discovering Procedural Exploits](http://arxiv.org/abs/2510.03405)

- LEGALSIM (Modular Multi-Agent Simulation of Adversarial Legal Proceedings): introduces a modular multi-agent simulation of adversarial legal proceedings, including a LEGALSIM environment (Domain-agnostic litigation simulator), an Agent Layer (Supports multiple policy families), and a Training and Evaluation Harness (Coordinates experiments, validates actions), designed to discover procedural exploits in codified legal rules.
- The simulation features plaintiff and defendant agents choosing actions governed by a JSON rules engine and a stochastic judge model, allowing for the study of emergent "exploit chains" like cost-inflating discovery sequences.
- The framework evaluates various policies, including heuristic, LLM-driven, contextual bandit, and PPO, to assess their effectiveness and exploitiveness across different judge profiles and procedural regimes.

---

[Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair](http://arxiv.org/abs/2510.03217)

- Abstain and Validate: introduces a dual-LLM policy for agentic program repair, which processes a Bug Report (input bug information) through Bug Abstention (LLM Policy) (filters unlikely bugs) and an APR Agent (LLM + Tools) (generates candidate patches), then applies Patch Validation (Checks + LLM Policy) (filters incorrect patches) to yield a Validated Patch (accepted fix) or discard a Discarded Bug (rejected for repair) or Discarded Patch (rejected as incorrect), ultimately reducing noise.
- This framework aims to improve the quality of patches shown to developers, thereby saving valuable review time and building trust in automated code changes.
- The two policies, bug abstention and patch validation, are complementary and significantly raise the success rate of filtered patches, especially when combined.

---

[FOCUSAGENT: Simple Yet Effective Ways of Trimming the Large Context of Web Agents](http://arxiv.org/abs/2510.03204)

- FOCUSAGENT: introduces a two-stage pipeline that leverages a lightweight LLM retriever (Stage 1 Retrieval LLM) to prune raw web page AxTree observations (Observation) into a reduced input (Pruned Observation), which is then used by a main LLM agent (Stage 2 Action Prediction LLM) to predict actions (Agent) for task completion (Goal), while also mitigating prompt injection attacks (Potential Prompt Injection).
- This approach significantly reduces observation size by over 50%, leading to more efficient reasoning and reduced vulnerability to security threats without sacrificing task success rates.
- By implicitly accounting for planning context, task goals, and action history, the framework effectively filters navigation-relevant elements, outperforming traditional semantic similarity methods in interactive web environments.

---

[Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling](http://arxiv.org/abs/2510.03199)

- BoM (Best-of-Majority): introduces a minimax-optimal strategy for Pass@k inference scaling, combining majority voting and Best-of-N by generating responses, calculating their frequencies, filtering candidates based on a frequency threshold, querying reward labels, and finally selecting the top-k responses.
- This framework ensures scaling-monotonicity, meaning its performance does not degrade with increased sampling budget N, and achieves optimal regret scaling with respect to k, addressing limitations of prior methods.
- Empirical evaluations on mathematical reasoning tasks demonstrate BoM's superior performance over baselines and validate its scaling-monotonic properties, especially for small k.

---

[CoDA: Agentic Systems for Collaborative Data Visualization](http://arxiv.org/abs/2510.03194)

- CoDA (Collaborative Data-visualization Agents): introduces a multi-agent system for automated data visualization, including query analyzer, data processor, VizMapping agent, search agent, design explorer, code generator, debug agent, and visual evaluator, which transforms natural language queries into refined visualizations through a self-evolving pipeline.
- The framework leverages metadata-focused analysis to bypass token limits and employs quality-driven refinement to ensure robust handling of complex datasets and iterative visualization needs.
- CoDA significantly outperforms competitive baselines by up to 41.5% in overall score, demonstrating the efficacy of collaborative agentic workflows for visualization automation.

---

[Improving Cooperation in Collaborative Embodied AI](http://arxiv.org/abs/2510.03153)

- CoELA (Collaborative Embodied Language Agents): introduces an enhanced multi-agent embodied AI system, with Perception (interprets environment), Memory (stores world knowledge, interactions, behaviors), Planning (generates action plans), Communication (facilitates inter-agent dialogue), Execution (carries out actions), Ollama Integration (deploys LLMs locally), and TTS and Chat GUI Integration (real-time voice chat interface), which improves agent cooperation and task execution efficiency through optimized prompting strategies and a real-time dialogue interface.
- The system leverages LLMs as cognitive engines for reasoning and coordination, exploring various prompting methods and LLM configurations to maximize collaborative performance in shared virtual spaces.
- Speech capabilities and a chat GUI are integrated to provide a more engaging user interface, aiding system development and demonstrating improved clarity and task alignment in agent interactions.

---

[AUDIOTOOLAGENT: AN AGENTIC FRAMEWORK FOR AUDIO-LANGUAGE MODELS](http://arxiv.org/abs/2510.02995)

- AudioToolAgent: introduces a framework that coordinates audio-language models as tools via a central LLM agent, which accesses tool adapters for audio question answering and speech-to-text, selecting tools, asking follow-up questions, and comparing outputs for verification.
- The framework enables an LLM agent to use audio models as tools, combining the reasoning capabilities of general LLMs with the audio processing strengths of LALMs without requiring new data or training.
- AudioToolAgent achieves state-of-the-art accuracy on MMAU, MMAR, and MMAU-Pro benchmarks by iteratively invoking tools, comparing outputs, and verifying disagreements to increase reliability.

---

[BEYOND THE FINAL ANSWER: EVALUATING THE REASONING TRAJECTORIES OF TOOL-AUGMENTED AGENTS](http://arxiv.org/abs/2510.02837)

- TRACE (Trajectory-based Reasoning Assessment and Comprehensive Evaluation): introduces a framework for multi-dimensional evaluation of tool-augmented LLM agent performance, including a tool-augmented LLM agent (generates reasoning trajectory), reasoning trajectory (ordered sequence of steps), tools (external functionalities for agents), evidence bank (stores factual information), TRACE framework (evaluates agent performance), LLM evaluator (assesses trajectory metrics), efficiency assessment (quantifies unnecessary evidence), hallucination detection (identifies factual deviations), adaptivity measurement (evaluates tool failure response), user query (initial problem statement), and final answer (agent's ultimate solution), which provides a comprehensive understanding of an agent's problem-solving process beyond just the final answer.
- The framework incorporates an evidence bank to accumulate knowledge from reasoning steps, enabling a multi-faceted analysis of an agent's trajectory without relying on ground-truth paths.
- TRACE accurately evaluates complex agent behaviors, including efficiency, hallucination, and adaptivity, in a scalable and cost-effective manner, even with smaller open-source LLMs.

---

[Prototyping Digital Social Spaces through Metaphor-Driven Design: Translating Spatial Concepts into an Interactive Social Simulation](http://arxiv.org/abs/2510.02759)

- Metaphor-Driven System: introduces a novel approach for prototyping digital social spaces by translating user-provided spatial metaphors into interactive social media simulations populated with LLM-driven agents.
- The system leverages an LLM to convert spatial metaphors into structured social attributes, which are then mapped to platform features using a 3-level taxonomy to generate a dynamic social media environment.
- LLM-driven agents, customized with social roles and behavioral traits, populate the simulated spaces, enabling real-time interactions that reflect the metaphor's intended social dynamics.

---

[The Path of Self-Evolving Large Language Models: Achieving Data-Efficient Learning via Intrinsic Feedback](http://arxiv.org/abs/2510.02752)

- Self-aware RL: introduces a self-evolving training loop where a generator agent creates tasks with predicted difficulty, a solver agent attempts to solve them, and a task filter determines if external guidance is needed for high-utility, unsolvable tasks, with aggregated rewards driving policy updates.
- This paradigm incorporates self-aware difficulty prediction, enabling the generator to create appropriately challenging tasks aligned with the LLM's current capabilities, and self-aware limit breaking, which proactively seeks minimal external guidance for valuable tasks beyond the solver's current limits.
- By leveraging intrinsic feedback and self-awareness, the framework achieves data-efficient learning, significantly improving LLM reasoning and generalization abilities while reducing reliance on extensive human-annotated data.

---

[TIME-TO-INCONSISTENCY: A SURVIVAL ANALYSIS OF LARGE LANGUAGE MODEL ROBUSTNESS TO ADVERSARIAL ATTACKS](http://arxiv.org/abs/2510.02712)

- Time-to-Inconsistency (Survival Modeling Framework): introduces a comprehensive survival analysis of LLM robustness to adversarial attacks, employing Cox proportional hazards model (semi-parametric survival analysis), Accelerated Failure Time (AFT) models (parametric survival analysis), and Random Survival Forests (RSF) (non-parametric ensemble method) to model conversational failure as a time-to-event process.
- The framework analyzes 36,951 conversation turns across 9 state-of-the-art LLMs, revealing that abrupt prompt-to-prompt semantic drift catastrophically increases failure hazard, while gradual cumulative drift is protective.
- AFT models demonstrate superior performance in capturing the time-varying nature of LLM failure risk, challenging assumptions about semantic consistency and providing insights for resilient conversational AI design.

---

[MALF: A MULTI-AGENT LLM FRAMEWORK FOR INTELLIGENT FUZZING OF INDUSTRIAL CONTROL PROTOCOLS](http://arxiv.org/abs/2510.02694)

- MALF (Multi-Agent LLM Fuzzing Framework): introduces a novel multi-agent LLM framework for intelligent fuzzing of industrial control protocols, integrating Seed Generation Agent, Test Case Generation Agent, Feedback Analysis Agent, Communication Interaction Module, RAG Pipeline, QLoRA Pipeline, Domain-Enhanced LLM, Knowledge Sources, Vector Database, System Under Test (SUT), PLCs, and Client/Operator/Engineer Station, to automate vulnerability discovery in complex industrial control systems.
- The framework leverages Retrieval-Augmented Generation (RAG) for domain-specific knowledge and QLoRA fine-tuning to dynamically generate protocol-aware test cases, enhancing fuzz testing precision and adaptability.
- MALF's multi-agent coordination optimizes seed generation, mutation strategies, and feedback-driven refinement, leading to improved vulnerability discovery and setting a new standard for critical infrastructure security.

---

[AutoMaAS: Self-Evolving Multi-Agent Architecture Search for Large Language Models](http://arxiv.org/abs/2510.02669)

- AutoMaAS (Self-Evolving Multi-Agent Architecture Search): introduces a self-evolving multi-agent architecture search framework that leverages neural architecture search principles to automatically discover optimal agent configurations through dynamic operator lifecycle management, multi-objective cost optimization, online feedback integration, and an architecture interpretability engine.
- This framework dynamically samples query-dependent multi-agent architectures from an evolving supernet, continuously managing operator lifecycles, optimizing costs, and refining selections based on real-time feedback.
- AutoMaAS achieves performance improvements and reduces inference costs by adapting to varying query characteristics and deployment conditions, offering enhanced interpretability through decision tracing.

---

[Mind the Gap: Linguistic Divergence and Adaptation Strategies in Human-LLM Assistant vs. Human-Human Interactions](http://arxiv.org/abs/2510.02645)

- Linguistic Divergence and Adaptation Strategies: introduces a framework to address communication style shifts between human-LLM and human-human interactions, utilizing Linguistic Divergence Analysis, Post-training Data Augmentation (with Minimal Style Rewriting and Enriched Style Rewriting), and Inference-time User Message Reformulation, all supported by a Linguistic Dimension Rubric and LLMs (Claude 3.5 Sonnet v2 and Mistral-7B).
- The paper empirically demonstrates that users adopt distinct communication styles when interacting with LLM chatbots compared to human agents, characterized by lower grammatical fluency, politeness, and lexical diversity.
- The research highlights that training LLMs on stylistically diverse datasets significantly improves performance on human-LLM assistant interactions, outperforming inference-time reformulation for adapting to communication style changes.

---

[Long-Term Mapping of the Douro River Plume with Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2510.03534)

- Plume-DQN-GP (Plume Deep Q-Network Gaussian Process): introduces a cooperative multi-agent control framework for long-duration mapping of the Douro River plume, integrating a central server with a multi-head Q-network, a GPR estimator, and multiple AUVs for data collection and command execution.
- The framework offloads heavy computation to the server, which intermittently communicates with AUVs to collect measurements and issue adaptive speed and direction commands, optimizing for both mapping accuracy and energy efficiency.
- This approach leverages multi-agent coordination and adaptive velocity control to overcome challenges like dynamic plume evolution, ocean currents, and communication constraints, demonstrating improved endurance and accuracy over benchmarks.

---

[TOWARDS POLICY-COMPLIANT AGENTS: LEARNING EFFICIENT GUARDRAILS FOR POLICY VIOLATION DETECTION](http://arxiv.org/abs/2510.03485)

- POLICYGUARD-4B: introduces a lightweight guardrail model for detecting policy violations in web agent trajectories, utilizing a fine-tuned Qwen3-4B-Instruct backbone to process policies, trajectory actions, and domain metadata, and output a binary violation label.
- The model is trained on POLICYGUARDBENCH, a 60k-scale benchmark designed for policy-trajectory violation detection, supporting both full-trajectory and prefix-based evaluation.
- POLICYGUARD-4B demonstrates strong accuracy, cross-domain generalization, and state-of-the-art efficiency, outperforming larger models and existing safety-oriented guardrails in policy compliance detection.

---

[Adversarial Reinforcement Learning for Offensive and Defensive Agents in a Simulated Zero-Sum Network Environment](http://arxiv.org/abs/2510.05157)

- ARL (Adversarial Reinforcement Learning) Environment: introduces a controlled study of two competing Deep Q-Network (DQN) agents, an attacker and a defender, within a custom OpenAI Gym-style simulated zero-sum network environment.
- This environment models offensive brute-force attacks and reactive defenses on a multi-port service, incorporating realistic trade-offs like background traffic, IP-based evasion, traps, and rate-limiting.
- The framework evaluates value-based agents across various configurations, demonstrating how defender observability and trap effectiveness hinder exploitations, with reward shaping and training scheduling being crucial for learning stability.

---

[VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation](http://arxiv.org/abs/2510.05156)

- VeriGuard: introduces a novel framework for enhancing LLM agent safety, featuring a Policy Generation stage (VeriGuard User, Agent Spec. (Request), Policy Generator, Constraints, Policy Code, Validation, Assumptions, Testing, Verification, Verified Policy) and a Policy Enforcement stage (Agent User, Agent, Input Processing, Input Arguments, Execute Policy, Access, Violation, Compliance).
- The framework employs a dual-stage architecture, where an offline Policy Generation stage rigorously validates and formally verifies agent policies, and an online Policy Enforcement stage monitors and validates proposed agent actions against these pre-verified policies before execution.
- This approach shifts from reactive filtering to proactive, provable safety by integrating policy specification generation and automated verification into the agent's action-generation pipeline, ensuring "correct-by-construction" behavior and providing formal safety guarantees.

---

#### 2nd October 2025

[AgentRec: Next-Generation LLM-Powered Multi-Agent Collaborative Recommendation with Adaptive Intelligence](http://arxiv.org/abs/2510.01609)

- AgentRec: introduces a next-generation LLM-powered multi-agent collaborative recommendation framework, with User Query, Conversation History, Context Environment, Candidate Items, Conversation Understanding Agent, Preference Modeling Agent, Context Awareness Agent, Dynamic Ranking Agent, Adaptive Coordinator, Complexity Analysis, Tier 1 - Rapid Response Layer, Tier 2 - Intelligent Reasoning Layer, Tier 3 - Deep Collaboration Layer, Ranked Recommendations, and Explanations, addressing conversational recommender system limitations through hierarchical agent networks and adaptive intelligence.
- The framework employs specialized LLM-powered agents for conversation understanding, preference modeling, context awareness, and dynamic ranking, coordinated by an adaptive weighting mechanism that learns from interaction patterns.
- AgentRec utilizes a three-tier learning strategy (rapid response, intelligent reasoning, deep collaboration) to optimize response time and recommendation quality by dynamically routing queries based on complexity scores.

---

[AGENT-SCANKIT: UNRAVELING MEMORY AND REASONING OF MULTIMODAL AGENTS via SENSITIVITY PERTURBATIONS](http://arxiv.org/abs/2510.00496)

- Agent-ScanKit: introduces a systematic probing framework to unravel memory and reasoning capabilities of multimodal agents in GUI tasks, utilizing sensitivity perturbations across visual-guided, text-guided, and structure-guided paradigms.
- The framework quantifies contributions of memorization and reasoning without requiring internal model access, revealing that existing agents often rely on mechanical memorization over systematic reasoning.
- Agent-ScanKit's findings highlight the necessity of robust reasoning modeling for reliable multimodal agents, offering insights into their generalization limitations in real-world scenarios.

---

[Agentic Additive Manufacturing Alloy Discovery](http://arxiv.org/abs/2510.02567)

- Agentic Additive Manufacturing Alloy Discovery System: introduces a multi-agent system for automating alloy discovery in additive manufacturing, integrating a Claude Sonnet LLM, Thermo-Calc, Workspace, and Additive Manufacturing subagents via the Model Context Protocol (MCP).
- This system enables LLM-driven reasoning to dispatch tool calls for tasks like calculating thermophysical properties, managing experimental data, and generating printability process maps.
- The framework dynamically adjusts task trajectories based on tool call outcomes, facilitating autonomous decision-making and accelerating alloy discovery.

---

[Orchestrating Human-AI Teams: The Manager Agent as a Unifying Research Challenge](http://arxiv.org/abs/2510.02557)

- AMA (Autonomous Manager Agent): introduces a research vision for autonomous agentic systems that orchestrate collaboration within dynamic human-AI teams, featuring a Manager Agent, Human Workers, AI Workers, a Stakeholder, a Workflow, a Communication Service, an Agent Registry, a Workflow Execution Engine, and a Validation Engine.
- The Manager Agent is responsible for decomposing complex goals into task graphs, allocating tasks to human and AI workers, monitoring progress, adapting to changing conditions, and maintaining transparent stakeholder communication.
- The paper formalizes workflow management as a Partially Observable Stochastic Game and releases MA-GYM, an open-source simulation and evaluation framework, to advance research in compositional reasoning, multi-objective optimization, ad hoc team coordination, and governance by design.

---

[Interactive Training: Feedback-Driven Neural Network Optimization](http://arxiv.org/abs/2510.02297)

- Interactive Training: introduces a framework for real-time, feedback-driven neural network optimization, featuring a Frontend Dashboard (visualizes metrics, sends commands), Control Server (mediates communication, manages state), Interactive Trainer (performs training, applies interventions), and communication via REST API (sends intervention commands) and WebSocket (broadcasts real-time updates), enabling both Human Experts (manually intervenes training) and Automated AI Agents (autonomously intervenes training) to dynamically adjust training parameters.
- The Control Server manages Command Queues (enqueues intervention commands) and Server Callback Message Queues (receives training updates), while the Interactive Trainer integrates callbacks like InteractiveCallback (adjusts training parameters), CheckpointCallback (manages model checkpoints), LoggingCallback (captures training metrics), and RunPauseCallback (controls training flow) for dynamic adjustments.
- This framework transforms neural network optimization from a static, passive task into an active, responsive process, improving training stability, reducing sensitivity to initial hyperparameters, and adapting to evolving user needs, including through LLM-based interventions.

---

[STOCKBENCH: CAN LLM AGENTS TRADE STOCKS PROFITABLY IN REAL-WORLD MARKETS?](http://arxiv.org/abs/2510.02209)

- STOCKBENCH: introduces a contamination-free benchmark designed to evaluate LLM agents in realistic, multi-month stock trading environments, directly measuring their profitability and risk-management capabilities, with Back-Trading Environment (historical data simulation), Stock Trading Agent Workflow (LLM agent evaluation), Investment Target (selected stocks for trading), Price & Fundamental Data (market prices, company financials), News Corpus (daily news articles), Evaluation Window (data collection timeframe), LLM Agent (backbone model for decisions), Portfolio Overview (initial market scan), In-depth Stock Analysis (deeper stock data analysis), Decision Generation (buy/sell/hold actions), and Execution and Validation (execute, validate decisions).
- The benchmark simulates real-world stock trading by exposing LLM agents to daily market signals, including prices, company fundamentals, and news headlines, requiring sequential buy, sell, or hold decisions.
- Performance is assessed using financial metrics such as cumulative return, maximum drawdown, and Sortino ratio, providing a quantitative assessment of trading success and risk management.

---

[Cooperative Guidance for Aerial Defense in Multiagent Systems](http://arxiv.org/abs/2510.02087)

- Cooperative Guidance Framework (CGF): introduces a time-constrained cooperative guidance strategy for an evader-defender team, including an evader (high-value drone), a defender (protective drone), and a pursuer (hostile drone), to protect the evader from interception.
- The CGF leverages a true proportional navigation-based approach, where the evader employs a deception strategy to nullify its line-of-sight rate with respect to the pursuer, and the defender intercepts the pursuer within a fixed time using sliding manifolds.
- This strategy ensures robust and guaranteed interception, is computationally lightweight, scalable, and operates effectively even without prior knowledge of the pursuer's strategy or control laws.

---

[TACOS: Task Agnostic COordinator of a multi-drone System](http://arxiv.org/abs/2510.01869)

- TACOS (Task-Agnostic Coordinator of a multi-drone System): introduces a unified framework for multi-UAV control, leveraging a hierarchical LLM architecture with a Coordinator LLM and Supervisor LLM to translate user instructions into executable actions, interacting with Swarm State, World State, Available Actions, and the Environment via API Calls.
- This framework enables intuitive one-to-many natural language interaction, allowing users to delegate complex tasks and manage swarm behaviors autonomously.
- By integrating semantic reasoning with real-time multi-robot coordination, the system aims to reduce pilot workload and enhance mission flexibility and resilience in unpredictable settings.

---

[SoK: Measuring What Matters for Closed-Loop Security Agents](http://arxiv.org/abs/2510.01654)

- CLASP (Closed-Loop Autonomous Security Performance): introduces a capability-centric framework and vocabulary that jointly characterizes security-function complexity and agentic capability maturity, mapping systems onto these axes to explain performance.
- The framework defines five security functions (reconnaissance, exploitation, root cause analysis, patch synthesis, fix verification and validation) and six agentic capabilities (planning, tool use, memory, reasoning, perception, reflection & adaptation).
- It also introduces the Closed-Loop Capability (CLC) Score, a composite metric quantifying both loop closure and operational effectiveness, and outlines requirements for a closed-loop benchmark to advance security agents.

---

[POSITION: PRIVACY IS NOT JUST MEMORIZATION!](http://arxiv.org/abs/2510.01645)

- LLM Privacy Landscape Taxonomy and Roadmap: introduces a comprehensive analysis of privacy risks in LLM systems, categorizing them into Training Data Leakage via Regurgitation (model as data-store), Direct Chat Leakage via Uninformed Consent or Compromised Provider (provider breaches, deceptive policies), Indirect Chat and Context Leakage via Input-Output Flow (autonomous agent, prompt injection), Indirect Attribute Inference (deduce sensitive info), and Direct Attribute Aggregation (weaponize dispersed info), while proposing a roadmap of technical, sociotechnical, and policy solutions.
- The paper argues that current research disproportionately focuses on verbatim memorization, overlooking more prevalent and scalable privacy threats arising from data collection practices, inference-time context leakage, and autonomous agent capabilities.
- It advocates for a fundamental shift towards interdisciplinary approaches to address the sociotechnical nature of LLM privacy, emphasizing user empowerment, transparency, and policy reforms beyond purely algorithmic solutions.

---

[AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System](http://arxiv.org/abs/2510.01617)

- AMAS (Adaptive Multi-Agent System): introduces a paradigm-shifting framework that redefines LLM-based Multi-Agent Systems through a novel dynamic graph designer, which autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation.
- The framework addresses the limitations of inflexible, hand-crafted graph topologies in conventional MAS by exploiting intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways.
- AMAS achieves superior task resolution efficacy and computational efficiency across diverse LLM architectures and benchmarks, establishing its viability for large-scale industrial deployment.

---

[Predictive Preference Learning from Human Interventions](http://arxiv.org/abs/2510.01545)

- PPL (Predictive Preference Learning from Human Interventions): introduces an interactive imitation learning algorithm that leverages a trajectory prediction model, human expert, human buffer, preference buffer, behavioral cloning loss, and preference-classification loss to learn from human interventions by forecasting future rollouts and converting interventions into preference signals.
- The framework aims to improve learning efficiency and reduce human cognitive burden by proactively identifying potential failures through predicted trajectories and propagating expert corrections across future states.
- PPL's approach mitigates distributional shift and reduces the number of required human demonstrations by enriching the training dataset with anticipated future states and their associated preferences.

---

[WORLD MODEL FOR AI AUTONOMOUS NAVIGATION IN MECHANICAL THROMBECTOMY](http://arxiv.org/abs/2509.25518)

- World Model for AI Autonomous Navigation (TD-MPC2): introduces a framework for autonomous endovascular navigation in mechanical thrombectomy, leveraging TD-MPC2's model-based RL algorithm, which integrates temporal difference learning, model-predictive control, a learned dynamics model, a latent dynamics model, a cross-entropy planning method, and an LSTM layer, within the stEVE simulation environment.
- The framework trains a single RL agent across multiple endovascular navigation tasks in ten real patient vasculatures, demonstrating superior generalization and a 65% mean success rate compared to the state-of-the-art Soft Actor-Critic (SAC) method.
- This research highlights the potential of world models for generalizable AI-driven robotic interventions in complex vascular anatomies, while also noting a trade-off between success rate and execution speed.

---

[RedCodeAgent: AUTOMATIC RED-TEAMING AGENT AGAINST DIVERSE CODE AGENTS](http://arxiv.org/abs/2510.02609)

- RedCodeAgent (AUTOMATIC RED-TEAMING AGENT): introduces an automated and adaptive red-teaming agent designed to systematically uncover vulnerabilities in diverse code agents, utilizing a Memory Module, LLM Red Teaming Function Call, Toolbox, Query Target Code Agent, Evaluation Module, and Self-reflection.
- The framework leverages an adaptive memory to store successful attack experiences and dynamically selects effective red-teaming tools and combinations from its toolbox to optimize attack strategies.
- It employs simulated sandbox environments for reliable evaluation of code agent execution results, mitigating biases from static code analysis or LLM-based judges.

---

[TOOLTWEAK: AN ATTACK ON TOOL SELECTION IN LLM-BASED AGENTS](http://arxiv.org/abs/2510.02554)

- ToolTweak (Adversarial Manipulation of Tool Selection): introduces an automatic, lightweight attack that iteratively refines tool names and descriptions using LLM feedback to bias LLM-based agents towards selecting specific tools, significantly increasing selection rates.
- This gradient-free, transferable attack exploits a critical vulnerability in LLM-based agents' reliance on surface-level, unverified tool metadata, causing distributional shifts in tool usage across tool ecosystems.
- The paper also evaluates defenses like paraphrasing and perplexity filtering, which reduce bias and promote more equal tool selection, highlighting the ongoing challenge of robust tool ecosystems.

---

[CLARITY: Clinical Assistant for Routing, Inference, and Triage](http://arxiv.org/abs/2510.02463)

- CLARITY (Clinical Assistant for Routing, Inference, and Triage): introduces an AI-driven platform designed to facilitate patient-to-specialist routing, clinical consultations, and severity assessment, combining a Finite State Machine (FSM) for structured dialogue flows with collaborative LLM agents to analyze symptoms and prioritize referrals. 
- The system's hybrid architecture, built on a modular microservices framework, ensures safe, efficient, and robust performance, offering flexibility and scalability for healthcare IT solutions. 
- CLARITY has been integrated into a large-scale inter-hospital IT platform, demonstrating human-level performance in first-attempt routing precision and significantly shorter consultation durations.

---

[AgentCaster: Reasoning-Guided Tornado Forecasting](http://arxiv.org/abs/2510.03349)

- AgentCaster: introduces a contamination-free framework employing multimodal LLMs end-to-end for tornado forecasting, with an LLM Agent, Meteorological Data Sources, Forecast Maps, Forecast Soundings (BUFKIT data), Tool Set, list_available_map_types, request_hrrr_map, request_sounding, submit_tornado_prediction, Agent Interaction Loop, Ground Truth Generation System, Evaluation Metrics, TornadoBench, and TornadoHallucination (Simple/Hard), where the framework evaluates LLM reasoning on complex, real-world tornado forecasting tasks using interactive data querying and domain-specific metrics.
- The framework enables LLM agents to act as AI meteorologists, interpreting heterogeneous spatiotemporal data from a high-resolution forecast archive and generating probabilistic tornado-risk polygon predictions.
- The system utilizes a multi-turn conversational loop and a defined set of tools to mimic human forecaster workflows, with predictions verified against ground truths and evaluated using novel domain-specific metrics.

---

[FalseCrashReducer: Mitigating False Positive Crashes in OSS-Fuzz-Gen Using Agentic AI](http://arxiv.org/abs/2510.02185)

- FalseCrashReducer introduces two LLM-driven strategies, constraint-based fuzz driver generation and context-based crash validation, implemented by a Function Analyzer Agent and a Crash Validation Agent, to mitigate false positive crashes in OSS-Fuzz-Gen.
- The Function Analyzer Agent proactively derives function constraints to guide fuzz driver creation, while the Crash Validation Agent reactively analyzes function callers to determine crash feasibility.
- These strategies, supported by Code Search and Function Search tools, significantly reduce spurious crashes and lower the debugging burden for software engineers in large-scale fuzzing pipelines.

---

#### 1st October 2025

[GUI-KV: EFFICIENT GUI AGENTS VIA KV CACHE WITH SPATIO-TEMPORAL AWARENESS](http://arxiv.org/abs/2510.00536)

- GUI-KV: introduces a plug-and-play KV cache compression method for GUI agents, with spatial saliency guidance (L2 norm hidden states) and temporal redundancy scoring (QR decomposition previous frames), designed to exploit GUI-specific redundancies for efficient and reliable agent performance.
- The method significantly reduces decoding FLOPs and improves step accuracy by leveraging uniform attention sparsity across transformer layers in GUI environments.
- GUI-KV consistently outperforms competitive KV compression baselines, achieving near-full-cache accuracy at modest budgets and enabling GUI agents to operate with reduced memory.

---

[PAL-UI: PLANNING WITH ACTIVE LOOK-BACK FOR VISION-BASED GUI AGENTS](http://arxiv.org/abs/2510.00413)

- PAL-UI (Planning with Active Look-back): introduces a novel framework that enables GUI agents to adaptively retrieve past observations when required, combining a dual-level summarization agent, a dedicated retrieval tool, and an MLLM agent for long-horizon planning.
- The framework compresses interaction history into a succinct textual memory and equips the agent with a special tool to fetch detailed visual information from past steps on demand, mitigating context length limitations.
- PAL-UI is trained via supervised fine-tuning on a synthetic instruction-tuning dataset augmented with tool-use demonstrations, demonstrating strong performance in mobile and web GUI navigation tasks.

---

[REINFORCEMENT LEARNING WITH DISCRETE DIFFUSION POLICIES FOR COMBINATORIAL ACTION SPACES](http://arxiv.org/abs/2509.22963)

- RL-D2 (Reinforcement Learning with Discrete Diffusion Policies): introduces a novel framework for training discrete diffusion models as highly effective policies in complex combinatorial action spaces, utilizing a Policy Iteration Structure with Policy Evaluation and Policy Improvement, a PMD-derived Target Policy Distribution, a Discrete Diffusion Model (comprising Forward and Reverse Processes with a Training Objective), Divergence Minimization (FKL/RKL), and On-Policy Diffusion Learning.
- The framework addresses the scalability challenges of reinforcement learning in large, combinatorial action spaces by decoupling RL objective optimization from representation learning, delegating the latter to an expressive diffusion model for stable and enhanced training performance.
- The method demonstrates state-of-the-art results and superior sample efficiency across diverse benchmarks, including DNA sequence generation, long-horizon RL with macro-actions, and cooperative multi-agent systems, showcasing its versatility and computational efficiency.

---

[A cybersecurity AI agent selection and decision support framework](http://arxiv.org/abs/2510.01751)

- AIATDF (AI Agent Taxonomy and Decision Framework): introduces a structured decision support framework for selecting and deploying AI agents in cybersecurity, integrating contextual task decomposition, agent property mapping, architectural suitability analysis, performance metric definition, agent deployment, and iterative refinement.
- The framework systematically aligns diverse AI agent architectures (reactive, cognitive, hybrid, learning) and graduated levels of autonomy (assisted, augmented, autonomous) with the NIST CSF 2.0 functions (Govern, Identify, Protect, Detect, Respond, Recover) and their subcategories.
- This approach provides a transparent, stepwise methodology for integrating AI solutions into cybersecurity operations, enhancing situational awareness, accelerating response times, and fortifying long-term resilience through adaptive risk management.

---

[OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models](http://arxiv.org/abs/2510.01409)

- OntoLogX (Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models): introduces an autonomous AI agent that transforms raw cybersecurity logs into ontology-grounded Knowledge Graphs (KGs) using LLMs, integrating a lightweight log ontology, Retrieval Augmented Generation (RAG), and iterative correction steps.
- The framework ensures syntactically and semantically valid KGs, which are then aggregated into sessions for an LLM to predict MITRE ATT&CK tactics, linking low-level log evidence to higher-level adversarial objectives.
- OntoLogX leverages code-oriented LLMs and a hybrid retrieval strategy to achieve robust KG generation and accurate mapping of adversarial activity, enhancing actionable Cyber Threat Intelligence extraction.

---

[Automating Data-Driven Modeling and Analysis for Engineering Applications using Large Language Model Agents](http://arxiv.org/abs/2510.01398)

- LLM Agent Pipeline: introduces two LLM-agent frameworks, a Multi-Agent System and a Single ReAct-Agent System, which autonomously handle data preprocessing, neural network development, training, hyperparameter optimization, and uncertainty quantification for engineering applications.
- Both frameworks are evaluated on a critical heat flux prediction benchmark, demonstrating their ability to automate complex modeling tasks with minimal human intervention and achieve performance comparable to human-expert models.
- The Multi-Agent System, with its specialized collaborative agents, exhibits higher reliability and computational efficiency, while the Single ReAct-Agent System offers greater adaptive flexibility and dynamic self-repair capabilities.

---

[MANAGERBENCH: EVALUATING THE SAFETY-PRAGMATISM TRADE-OFF IN AUTONOMOUS LLMS](http://arxiv.org/abs/2510.00857)

- MANAGERBENCH: introduces a benchmark for evaluating LLM decision-making in realistic managerial scenarios, including an operational goal (LLM's primary objective), success metrics (LLM performance evaluation), a realistic scenario (managerial decision context), and two conflicting options (trade-off choice) within human harm (evaluates safety alignment) and control (measures pragmatism) sets, where models must choose between achieving operational goals and ensuring safety.
- The benchmark reveals that leading LLMs struggle with the safety-pragmatism trade-off, often prioritizing operational goals over human safety or becoming overly safe and ineffective.
- This misalignment stems from flawed prioritization rather than an inability to perceive harm, as models' harm assessments align with human judgments, and is fragile to goal-oriented nudging prompts.

---

[Symmetry breaking in collective decision-making through higher-order interactions](http://arxiv.org/abs/2510.00853)

- Collective Decision-Making Model: introduces a framework where agents on a simplicial complex choose between mutually exclusive options, incorporating pairwise and higher-order social interactions, autonomous adoption, and recovery mechanisms.
- The model utilizes mean-field analysis and simulations on random and real simplicial complexes to study symmetry breaking and consensus formation.
- This research highlights the critical role of higher-order interactions and autonomous behavior in overcoming decision deadlocks and achieving consensus.

---

[Poster: Agentic AI meets Neural Architecture Search: Proactive Traffic Prediction for AI-RAN](http://arxiv.org/abs/2510.00851)

- Agentic AI framework using NAS-based LSTM: introduces a proactive traffic prediction system for AI-RAN, leveraging O-RAN's disaggregated architecture to separate architecture optimization (Non-RT RIC rApps) from real-time inference (Near-RT RIC xApps), enabling adaptive model deployment based on traffic conditions and resource constraints.
- This framework dynamically selects and orchestrates efficient Long Short-Term Memory (LSTM) architectures using Neural Architecture Search (NAS) to balance predictive accuracy and computational efficiency across diverse operational scenarios.
- The system achieves significant computational complexity reduction (70-75%) compared to static high-performance models while maintaining high prediction accuracy, particularly during critical network events, by adaptively deploying lightweight or complex LSTM models.

---

[Collaborative-Distilled Diffusion Models (CDDM) for Accelerated and Lightweight Trajectory Prediction](http://arxiv.org/abs/2510.00627)

- CDDM (Collaborative-Distilled Diffusion Models): introduces a novel method for real-time and lightweight trajectory prediction, built upon the Collaborative Progressive Distillation (CPD) framework, which progressively transfers knowledge from a high-capacity teacher diffusion model to a lightweight student model, jointly reducing both sampling steps and model size across distillation iterations.
- The framework incorporates a dual-signal regularized distillation loss, integrating guidance from both the teacher and ground-truth data to mitigate overfitting and ensure robust performance.
- CDDM achieves state-of-the-art prediction accuracy with significantly reduced computational cost, enabling resource-efficient probabilistic prediction for Autonomous Vehicles and Intelligent Transportation Systems.

---

[AI-Driven Self-Evolving Software: A Promising Path Toward Software Automation](http://arxiv.org/abs/2510.00591)

- AI-Driven Self-Evolving Software: introduces a multi-agent system that autonomously interprets user requirements, generates and validates code, and integrates new functionalities, enabling continuous software evolution through direct user interaction.
- This prototype aims to move AI beyond an assistant role to become a core component of software, reducing economic costs and time overhead by replacing human developers with AI.
- Case studies demonstrate the feasibility of the approach in constructing and reusing functionality, providing evidence for scaling to more sophisticated applications and paving the way for automated software development.

---

[The Social Laboratory: A Psychometric Framework for Multi-Agent LLM Evaluation](http://arxiv.org/abs/2510.01295)

- The Social Laboratory (Psychometric Framework for Multi-Agent LLM Evaluation): introduces a novel evaluation framework that uses a multi-agent debate system, including LLM-based agents and an LLM moderator, along with psychometric and semantic metrics, to discover and quantify emergent social and cognitive behaviors of LLMs.
- This framework enables the analysis of how agent personas induce stable cognitive profiles and how the conversational environment, shaped by the moderator, significantly impacts debate outcomes and consensus-seeking tendencies.
- The research reveals a robust, innate tendency for LLM agents to seek consensus, demonstrating the framework's utility in understanding and shaping the social behaviors of next-generation AI agents.

---

[Cyber Academia-Chemical Engineering (CA-ChemE): A Living Digital Town for Self-Directed Research Evolution and Emergent Scientific Discovery](http://arxiv.org/abs/2510.01293)

- CA-ChemE (Cyber Academia-Chemical Engineering): introduces a multi-agent system for self-directed research evolution and emergent scientific discovery in chemical engineering, integrating seven specialized Expert Agents, a Collaboration Agent, domain-specific knowledge bases, and knowledge enhancement technologies.
- Each Expert Agent leverages a foundational LLM, a domain-specific knowledge base, and knowledge enhancement modules (RAG, LoRA, knowledge graphs) to achieve deep professional reasoning and accurate decision-making.
- The Collaboration Agent, equipped with ontology engineering capabilities, addresses cross-domain communication bottlenecks by standardizing terminology, translating context, and integrating knowledge, significantly improving interdisciplinary collaboration efficiency, especially for distant-domain expert pairs.

---

[JoyAgent-JDGenie: Technical Report on the GAIA](http://arxiv.org/abs/2510.00510)

- JoyAgent-JDGenie: introduces a generalist agent architecture, with a collective multi-agent framework (Plan Agent, ReAct Agent, Critic), a hierarchical memory system (Working Memory, Semantic Memory, Procedural Memory), and a refined tool suite (Search Tools, Code Execution Environment, Multimodal Parsing Tools, Browser Tools), designed for robust performance on complex real-world tasks.
- The framework integrates Plan-Execute and ReAct paradigms, coordinated by a Critic model, and employs a hierarchical memory system for long-horizon continuity and adaptive control.
- It utilizes a comprehensive tool ecosystem for search, code execution, and multimodal parsing, wrapped in schema-consistent interfaces, achieving competitive results on the GAIA benchmark.

---

[Agent Fine-tuning through Distillation for Domain-specific LLMs in Microdomains](http://arxiv.org/abs/2510.00482)

- LAFT (Language Agent Fine-Tuning): introduces an agent fine-tuning pipeline for domain adaptation within specialized IT microdomains, with data preparation, agentic fine-tuning (CPT and SFT), and inference components, where it leverages JP1-specific data and distilled agent trajectories to enhance decision-making accuracy and search efficiency.
- The framework employs a Contextual Answer Extractor to distill relevant information from lengthy retrieved contexts, improving retrieval efficiency and ensuring pertinent knowledge is retained.
- The approach demonstrates significant performance improvements on JP1 certification exam tasks, outperforming GPT-4 and highlighting the value of agent fine-tuning for domain-specific reasoning.

---

[Seeing through Uncertainty: Robust Task-Oriented Optimization in Visual Navigation](http://arxiv.org/abs/2510.00441)

- NEURO (Integrated Learning-to-Optimize Framework): introduces a novel hybrid framework that synergistically integrates deep neural networks with downstream robust optimization for end-to-end training in visual navigation, utilizing a Neural Perception Module, PICNN, Conformal Calibration Method, Robust Optimization Problem, Optimization Model, Solution Feedback, and Policy Module.
- The framework addresses challenges of data scarcity and partial observability by transforming noisy visual predictions into convex uncertainty sets and reformulating planning as a robust optimization problem.
- NEURO achieves state-of-the-art performance and improved generalization in multi-object navigation tasks by enabling uncertainty-aware policies that transfer across environments.

---

[Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting](http://arxiv.org/abs/2510.00401)

- PINCODE (Physics-Informed Neural Controlled Differential Equations): introduces a model for scalable long-horizon multi-agent motion forecasting, utilizing an Autoencoder (learns joint latent representation) and a Neural Controlled Differential Equation (propagates latent state across time), conditioned by a Smooth Control Path C (differentiable curve from cubic spline) and incorporating physics-informed constraints.
- The framework learns differential equation parameters to predict multi-agent system trajectories from an initial condition, enforcing physics constraints and scaling from 10 to 100 robots without additional model parameters.
- PINCODE achieves significant pose error reduction over 4-minute horizons compared to analytical models through progressive training with curriculum learning and continuous-time dynamics modeling.

---

[DBF-MA: A Differential Bayesian Filtering Planner for Multi-Agent Autonomous Racing Overtakes](http://arxiv.org/abs/2509.22937)

- DBF-MA (Multi-Agent Differential Bayesian Filtering): introduces a framework for multi-agent autonomous racing overtakes, utilizing Ego State Estimate (current vehicle state), Global Paths (pre-computed track data), Track Bounds (drivable area limits), Optimal Raceline (ORL) (ideal racing path), Trajectory Prediction Module (predicts target motion), Target Vehicle Prediction (predicted target trajectory), Composite Bézier Curve (CBC) Parameterization (trajectory representation), Prior Distribution (initial trajectory belief), Tire Grip Model / Friction Ellipse (vehicle dynamic limits), Likelihood Model (evaluates trajectory constraints), Sequential Monte Carlo (SMC) Inference Engine (samples feasible trajectories), and Valid Overtaking Trajectory (output trajectory) to synthesize collision-free, dynamically feasible, and track-bound overtaking maneuvers.
- The framework frames the overtaking problem as Bayesian inference over Composite Bézier Curves, ensuring C¹ continuity and explicit satisfaction of track-limit and curvature/acceleration constraints.
- DBF-MA produces risk-aware maneuvers by encoding the probability of satisfying all racing constraints within a likelihood function, outperforming existing methods in simulation.

---

[DEMYSTIFYING DEEP SEARCH: A HOLISTIC EVALUATION WITH HINT-FREE MULTI-HOP QUESTIONS AND FACTORISED METRICS](http://arxiv.org/abs/2510.05137)

- WebDetective: introduces a benchmark for evaluating web agents on hint-free multi-hop deep search within a controlled Wikipedia sandbox, featuring Hint-Free Multi-Hop Questions, a Controlled Wikipedia Sandbox, and a Holistic Evaluation Framework with Knowledge Sufficiency, Search Score, Generation Score, Good Refusal F1, Knowledge Utilization F1, and Knowledge Degradation Analysis (including Forget and Lead-astray), alongside an EvidenceLoop agentic workflow baseline that includes Iterative Refinement with Fallback (comprising Solver Agents, an Extraction Agent, and an Aggregation Agent), an Evidence Memory System (with Evidence IDs), and a Verification Mechanism (with a Verification Agent).
- The benchmark's co-design of questions and environment enforces autonomous discovery of reasoning chains, enabling fine-grained attribution of failure modes in multi-hop deep search tasks.
- The EvidenceLoop workflow, incorporating explicit verification and systematic evidence tracking, serves as a baseline to address the challenges identified by the benchmark, demonstrating that performance gains require genuine advances in reasoning and knowledge utilization rather than simple test-time scaling.

---

[Zero Data Retention in LLM-based Enterprise AI Assistants: A Comparative Study of Market Leading Agentic AI Products](http://arxiv.org/abs/2510.11558)

- Zero Data Retention LLM-based Enterprise AI Assistants (Comparative Study): introduces a comparative analysis of Salesforce AgentForce and Microsoft Copilot, detailing their architectural, compliance, and usability trade-offs in implementing zero data retention policies for enterprise LLM applications.
- The study examines how these leading agentic AI products safeguard private data and ensure compliance with regulations like GDPR and HIPAA through distinct technical designs and policy commitments.
- It highlights the importance of trust layers, data masking, content filtering, and stateless inference in achieving zero data retention, while also discussing usability trade-offs such as latency from client-side context management.

---

[Doing Things with Words: Rethinking Theory of Mind Simulation in Large Language Models](http://arxiv.org/abs/2510.13395)

- Concordia (Generative Agent-Based Model): introduces a framework to evaluate LLMs' Theory of Mind (ToM) abilities in simulated real-world environments, utilizing a Game Master (GM), Agents (LLMs), Agents' memories, Observations, Action attempt, Event statement, Direct Effect Externality, Multiple-Choice Question Answer, LLM API Call, Chain of Thought, and "Facts of the world" to assess action selection based on belief attribution.
- The paper investigates whether LLMs can make genuine inferences from social context rather than relying on linguistic memorization, specifically focusing on pragmatic interpretation and false-belief tasks.
- Findings reveal that LLMs frequently fail to select actions based on belief attribution and struggle to generate coherent causal effects, challenging claims about emergent ToM-like capabilities and advocating for action-based evaluation.

---

[Spec-Driven AI for Science: The ARIA Framework for Automated and Reproducible Data Analysis](http://arxiv.org/abs/2510.11143)

- ARIA (Automated Research Intelligence Assistant): introduces a spec-driven, human-in-the-loop framework for automated and interpretable data analysis, integrating Command (natural language interface), Context (persistent research memory), Code (executable Python modules), Data (unified data/model store), Orchestration (workflow/dependency management), and AI Module (adaptive interaction/execution engine) to unify human reasoning and machine execution.
- The framework transforms scientific exploration into an auditable, extensible, and dialogic process, enabling researchers to define analytical goals in natural language while ARIA autonomously generates code, validates computations, and produces transparent documentation.
- ARIA achieves high predictive accuracy, identifies optimal feature sets, selects suitable models, and ensures reproducibility and interpretability, bridging the gap between autonomous research agents and structured scientific workflows.

---

#### 30th September 2025


[Milestone Determination for Autonomous Railway Operation.](http://arxiv.org/abs/2510.06229)

- Milestone Determination Framework: introduces a method for autonomous railway operation, utilizing an ODM (Operational Domain Model) represented as a state machine, with milestones defining transitions between operational states, and an OwO (Observed weight of an Output) model for context-sensitive weighting of observed outputs.
- The framework incorporates Human-in-the-Loop (HitL) input to determine state-specific weights for contextual information, enhancing predictive performance for operational decision-making in railway simulation.
- By focusing on critical decision points and dynamically adjusting the relevance of observed data based on the current operational state, the framework aims to facilitate safer and more efficient machine learning systems for railway automation.

---


[Ferret-UI Lite: Lessons from Building Small On-Device GUI Agents](http://arxiv.org/abs/2509.26539)

- Ferret-UI Lite: introduces a compact, end-to-end GUI agent designed for on-device deployment, integrating an image encoder (Encodes GUI screen), a decoder-only LLM (Processes encoded image, user instruction), supervised fine-tuning (SFT) (Initial model training), and reinforcement learning with verifiable rewards (RLVR) (Refines model with verifiable rewards).
- The framework enhances GUI perception through visual tool-use (Enhances GUI perception), such as image cropping and zoom-in, and leverages a comprehensive data mixture from real and synthetic sources, including multi-agent system (Generates online synthetic data) rollouts.
- The agent achieves competitive performance in GUI grounding and navigation tasks compared to other small-scale models, demonstrating effective strategies for lightweight, on-device AI agents.

---

[Adaptive and Resource-efficient Agentic AI Systems for Mobile and Embedded Devices: A Survey](http://arxiv.org/abs/2510.00078)

- Adaptive and Resource-efficient Agentic AI System Workflow: introduces a systematic survey of adaptive and resource-efficient agentic AI systems for mobile and embedded devices, with components including Agent Paradigm, FM Model techniques (Elastic FM Inference, Test-time Adaptation, Dynamic Multi-modal FMs, Dynamic Multi-modal Input Adaptation), and System Scheduling.
- This framework addresses the challenges of deploying large foundation models (FMs) and AI agents on resource-constrained mobile and edge platforms, emphasizing adaptivity and resource efficiency.
- The survey outlines a novel taxonomy of enabling techniques to manage fluctuating hardware resources, dynamic inputs, and long-running open-world operations, clarifying trade-offs in accuracy, latency, communication, and energy efficiency.

---

[When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](http://arxiv.org/abs/2510.00332)

- CAIA (Crypto AI Agent Benchmark): introduces a benchmark for evaluating AI agents in adversarial, high-stakes financial markets, including a multi-stage curation pipeline, two evaluation conditions (with/without tools), an agentic framework for tool use, and a suite of specialized tools, against LLM agents and a human baseline.
- The benchmark reveals that state-of-the-art LLMs achieve only 12-28% accuracy without tools and plateau at 67.4% with tools, significantly below the 80% human baseline, primarily due to a systematic failure in tool selection, favoring unreliable web search over authoritative blockchain data.
- This highlights a critical gap in AI evaluation, demonstrating that current models lack skeptical reasoning and are unprepared for environments where misinformation is weaponized and errors have irreversible financial consequences, challenging assumptions about autonomous deployment.

---

[LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science](http://arxiv.org/abs/2510.01285)

- LLM-based Multi-Agent Blackboard System: introduces a novel multi-agent communication paradigm for information discovery in data science, where a central Main Agent (solves task, coordinates) posts requests to a shared Blackboard (shared request board), and autonomous Helper Agents (autonomous problem solvers), including File Agents (manage data files) and a Search Agent (retrieves web info), respond via a Response Board (collects agent replies).
- This system improves scalability and flexibility by eliminating the need for a central coordinator to have prior knowledge of all sub-agents' expertise, addressing limitations of single-agent and master-slave paradigms.
- The framework leverages an offline Clustering Data Files (organizes data offline) process for the Data Lake (raw data repository) and integrates external tools like Google Search (external search tool) and a Running Code (executes generated programs) environment for comprehensive problem-solving.

---

[BC-MPPI: A Probabilistic Constraint Layer for Safe Model-Predictive Path-Integral Control](http://arxiv.org/abs/2510.00272)

- BC-MPPI (Bayesian-Constraints MPPI): introduces a lightweight safety layer for Model Predictive Path Integral (MPPI) control, attaching a probabilistic surrogate to state and input constraints to ensure safety in highly nonlinear robotic tasks.
- This framework uses a Bayesian surrogate (Bayesian Neural Network) to return the probability of a candidate trajectory being feasible, which then scales the weight given to that candidate, effectively down-weighting unsafe rollouts.
- The approach integrates naturally with verification-and-validation pipelines for certifiable autonomous systems by providing a stand-alone, version-controlled surrogate and a single scalar runtime safety score.

---

[A HIERARCHICAL AGENTIC FRAMEWORK FOR AUTONOMOUS DRONE-BASED VISUAL INSPECTION](http://arxiv.org/abs/2510.00259)

- Hierarchical Agentic Framework: introduces a multi-agent system for autonomous drone-based visual inspection, featuring a Head Agent for high-level planning and Worker Agents that execute low-level actions using the ReActEval methodology.
- The ReActEval method, employed by Worker Agents, follows a Reason-Act-Evaluate cycle to enable structured self-correction and effective task execution in physical environments.
- This framework addresses challenges in multi-drone management and task execution, demonstrating how reasoning method selection interacts with LLM capability and task complexity for optimal performance.

---

[CHAI: Command Hijacking against embodied AI](http://arxiv.org/abs/2510.00181)

- CHAI (Command Hijacking against embodied AI): introduces an optimization-based adversarial attack that exploits multimodal language interpretation of LVLMs by embedding structured natural-language instructions as visual signs into the visual scene.
- The framework systematically searches the token space, builds a dictionary of prompts, and guides an attacker LLM to generate Visual Attack Prompts, targeting the command layer of embodied AI systems.
- CHAI achieves high attack success rates across various LVLM agents and real robotic vehicles, demonstrating a new attack surface in embodied AI that necessitates extended defenses.

---

[OCEANGYM: A BENCHMARK ENVIRONMENT FOR UNDERWATER EMBODIED AGENTS](http://arxiv.org/abs/2509.26536)

- OCEANGYM agent framework: introduces a unified agent framework driven by MLLMs, which integrates perception, memory, and sequential decision-making for underwater embodied agents.
- The framework utilizes a Language Encoder to process instructions, a Perception Encoder for multi-modal observations, a Memory module for historical states, and an Action Decoder for control actions.
- This MLLM-based agent framework is designed to operate within the OCEANGYM benchmark, interpreting language instructions, fusing optical and sonar imagery, and controlling Autonomous Underwater Vehicles (AUVs) in complex underwater scenarios.

---

[Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](http://arxiv.org/abs/2509.26354)

- Misevolution: introduces "Misevolution," where an agent's self-evolution deviates in unintended ways, leading to undesirable or even harmful outcomes, across its model, memory, tool, and workflow components.
- The paper systematically investigates this phenomenon, providing empirical evidence of its occurrence in self-evolving LLM agents, even those built on top-tier LLMs.
- The findings highlight an urgent need for new safety paradigms to address emergent risks such as safety alignment degradation, data leakage, and privacy issues in dynamically evolving AI systems.

---

[SAFEEVALAGENT: TOWARD AGENTIC AND SELF-EVOLVING SAFETY EVALUATION OF LLMS](http://arxiv.org/abs/2509.26100)

- SafeEvalAgent: introduces a novel multi-agent framework for continuous and self-evolving safety evaluation of LLMs, including Specialist, Generator, Evaluator, and Analyst agents that collaborate to transform regulations into a testable knowledge base, generate diverse test suites, judge model responses, and refine attack strategies.
- The framework autonomously ingests unstructured policy documents to create and perpetually evolve a comprehensive safety benchmark, moving beyond static audits to dynamic red-teaming.
- SafeEvalAgent's self-evolving evaluation loop consistently uncovers deep vulnerabilities in LLMs, demonstrating its ability to adapt to evolving AI risks and regulatory landscapes.

---

[RE-Searcher: Robust Agentic Search with Goal-oriented Planning and Self-reflection](http://arxiv.org/abs/2509.26048)

- RE-Searcher: introduces a novel search agent that integrates goal-oriented planning (explicit search goal articulation) and self-reflection (retrieved evidence evaluation) to enhance robustness in complex search environments.
- The framework employs a Policy LLM for generating search goals, queries, and reflections, with an external LLM as Judge providing supervisory signals during training to refine reflection accuracy.
- RE-Searcher leverages a Search Engine for information retrieval and is trained using Group Relative Policy Optimization (GRPO) to mitigate the impact of noisy search results and correct biased trajectories.

---

[OPENID CONNECT FOR AGENTS (OIDC-A) 1.0: A STANDARD EXTENSION FOR LLM-BASED AGENT IDENTITY AND AUTHORIZATION](http://arxiv.org/abs/2509.25974)

- OIDC-A (OpenID Connect for Agents) 1.0: introduces a comprehensive framework for representing, authenticating, and authorizing LLM-based agents within the OAuth 2.0 ecosystem, including agent identity claims, agent attestation evidence mechanisms, delegation chain protocols, discovery mechanisms, authorization frameworks, and dedicated endpoints.
- This specification extends OpenID Connect Core 1.0 to address the unique characteristics of autonomous LLM agents, such as dynamic capabilities, complex delegation chains, and the need for attestation.
- The framework provides a foundation for secure and trustworthy agent-to-service interactions by standardizing protocols for agent identity representation, delegation chain validation, attestation verification, and capability-based authorization.

---

[Automated Model Discovery via Multi-modal & Multi-step Pipeline](http://arxiv.org/abs/2509.25946)

- Multi-modal & Multi-step Pipeline: introduces an effective automated model discovery approach leveraging two vision-language modules, AnalyzerVLM and EvaluatorVLM, for iterative model proposal and evaluation across four stages: Model Proposal, Model Fitting, Model Evaluation, and Model Selection.
- AnalyzerVLM autonomously plans and executes multi-step analyses, including code generation and visualization, to propose candidate models, while EvaluatorVLM assesses these models both quantitatively and perceptually using a novel Visual Information Criterion (VIC).
- The pipeline's multi-modality and multi-step reasoning capabilities enable it to effectively discover models that capture fine details and ensure strong generalizability, outperforming existing methods in various real-world datasets.

---

[NuRisk: A Visual Question Answering Dataset for Agent-Level Risk Assessment in Autonomous Driving](http://arxiv.org/abs/2509.25944)

- NuRisk VLM Agent: introduces a fine-tuned Vision Language Model for agent-level risk assessment in autonomous driving, integrating a Text Tokenizer, Vision Encoder, VL PatchMerger with Input/Output Projections and Fusion, an LLM, and Cross Attention, all enhanced with LoRA adapters.
- This agent processes sequential images and natural language queries to provide quantitative risk scores and chain-of-thought explanations, addressing the limitations of existing VLMs in spatio-temporal reasoning for safety-critical scenarios.
- The framework leverages a comprehensive Visual Question Answering dataset, NuRisk, built from real-world and simulated safety-critical driving scenarios, to enable robust domain adaptation and improved performance over proprietary models.

---

[LITA: LIGHT Agent UNCOVERS THE AGENTIC COD- ING CAPABILITIES OF LLMS](http://arxiv.org/abs/2509.25873)

- Lita (Lite Agent): introduces a lightweight agentic framework for evaluating and extending LLMs in coding tasks, operationalizing "liteness" by minimizing manual design and elaborate scaffolding, and includes an LLM, Memory, Tools, Reasoning, and Environment components.
- This framework enables a more faithful and unified evaluation of LLM coding capabilities by reducing reliance on complex, hand-crafted workflows and extensive prompt engineering.
- Lita demonstrates competitive or superior performance with fewer tokens and less design effort, supporting the "Agent Complexity Law" which posits that the performance gap between simple and complex agent designs diminishes as the core model's capabilities improve.

---

[Landmark-Guided Knowledge for Vision-and-Language Navigation](http://arxiv.org/abs/2509.25655)

- LGK (Landmark-Guided Knowledge): introduces a vision-and-language navigation method that uses an external knowledge base to assist navigation, addressing common-sense reasoning issues. 
- The framework incorporates Knowledge Matching, Knowledge-Guided by Landmark, and Knowledge-Guided Dynamic Augmentation to retrieve, guide, and integrate knowledge, vision, and language information. 
- LGK enhances navigation by leveraging landmark information to focus on relevant knowledge, dynamically augmenting instructions, and fusing multimodal data for improved environmental understanding and decision-making.

---

[AutoLabs: Cognitive Multi-Agent Systems with Self-Correction for Autonomous Chemical Experimentation](http://arxiv.org/abs/2509.25651)

- AutoLabs: introduces a self-correcting, multi-agent architecture designed to autonomously translate natural-language instructions into executable protocols for high-throughput liquid handlers, including a human interface, a supervisor agent, specialized sub-agents for understanding, chemical calculations, vial arrangement, processing steps, final steps, and a self-checks agent.
- The system engages users in dialogue, decomposes experimental goals into discrete tasks for specialized LLM-agents, performs tool-assisted stoichiometric calculations, and iteratively self-corrects its output before generating a hardware-ready file.
- AutoLabs achieves near-expert procedural accuracy and significantly reduces quantitative errors in chemical amounts by leveraging agent reasoning capacity, multi-agent architecture, and iterative self-correction mechanisms.

---

[STAC: WHEN INNOCENT TOOLS FORM DANGEROUS CHAINS TO JAILBREAK LLM AGENTS](http://arxiv.org/abs/2509.25624)

- STAC (Sequential Tool Attack Chaining): introduces an automated framework for generating multi-turn attacks against tool-enabled LLM agents, featuring a Generator (plans attack subgoals), Verifier (verifies tool chain executability), Prompt Writer (synthesizes benign attacker prompts), Planner (adaptively plans attack execution), and Judge (evaluates attack effectiveness/stealth).
- This framework exploits a unique vulnerability where sequences of individually benign tool calls collectively enable harmful operations, which only become apparent at the final execution step.
- The paper demonstrates that state-of-the-art LLM agents are highly vulnerable to STAC, with attack success rates exceeding 90% in most cases, and proposes a reasoning-driven defense prompt to mitigate these risks.

---

[INFIAGENT: SELF-EVOLVING PYRAMID AGENT FRAMEWORK FOR INFINITE SCENARIOS](http://arxiv.org/abs/2509.22502)

- InfiAgent (Self-Evolving Pyramid Agent Framework): introduces a DAG-based multi-agent framework featuring a Router, Self Evolution, and Context Control modules, designed for automatic task decomposition and self-adaptation across diverse problem domains.
- The framework employs an "agent-as-a-tool" mechanism for hierarchical task decomposition, a dual-audit system for quality assurance, and intelligent routing for efficient task-agent matching.
- InfiAgent's self-evolution mechanism autonomously restructures the agent DAG based on performance feedback, enabling continuous improvement and adaptability without human intervention.

---

#### 29th September 2025

[Retrieval-augmented GUI Agents with Generative Guidelines](http://arxiv.org/abs/2509.24183)

- RAG-GUI (Retrieval-augmented GUI Agents with Generative Guidelines): introduces a lightweight VLM adapter that leverages web tutorials at inference time, enhancing VLM-based GUI agents by generating task-aware guidance through its Guideline Generation Model (fe), Tutorial Collection, Training Process (SFT/RSF), and Inference Process, which then informs the Agent Backbone (VLM-based agent).
- The framework is model-agnostic, functioning as a plug-and-play module that improves agent performance by assessing tutorial relevance and generating useful guidance.
- RAG-GUI consistently outperforms baseline agents across diverse tasks and model sizes, demonstrating strong generalization and practical applicability in real-world scenarios.

---

[RadOnc-GPT: An Autonomous LLM Agent for Real-Time Patient Outcomes Labeling at Scale](http://arxiv.org/abs/2509.25540)

- RadOnc-GPT: introduces an autonomous LLM agent designed for real-time patient outcomes labeling at scale, integrating internal and external data resources, and capable of retrieving structured and unstructured clinical data, iteratively assessing evidence, and synthesizing structured outcomes.
- The system employs a two-tier evaluation strategy, first validating structured data retrieval accuracy (Tier 1) and then performing complex clinical outcome labeling tasks (Tier 2) such as ORN detection and cancer recurrence.
- RadOnc-GPT functions as both a labeler and an auditor, identifying latent errors in existing institutional registry labels and enhancing data integrity, thereby enabling scalable and trustworthy curation of radiation-oncology research datasets.

---

[INFOAGENT: ADVANCING AUTONOMOUS INFORMATION-SEEKING AGENTS](http://arxiv.org/abs/2509.25189)

- InfoAgent: introduces a deep research agent powered by an innovative data synthesis pipeline and orchestrated web search tools, including a ReAct framework, a Qwen3-14B base LLM, a data synthesis pipeline (Tree Construction, QA Generation, 03 LLM), customized search and browse tools (Crawler Server, BM25, Embedding, Reranker, LLM for snippets, Search Engines), a Redis server, and a two-stage training recipe (SFT, RL).
- The data synthesis pipeline generates challenging multi-entity search questions from Wikipedia entities using sub-tree sampling and entity fuzzification to enhance difficulty and require long-horizon retrieval and conjunctive reasoning.
- The customized search and browse tools provide a dedicated self-hosted search infrastructure, ensuring transparency, efficiency, and consistent results for agent training and evaluation, outperforming prior open-source deep research agents.

---

[TOWARDS PERSONALIZED DEEP RESEARCH: BENCHMARKS AND EVALUATIONS](http://arxiv.org/abs/2509.25106)

- Personalized Deep Research Evaluation: introduces a comprehensive system for evaluating Deep Research Agents (DRAs) using the Personalized Deep Research Bench (PDRB) benchmark and the PQR Evaluation Framework, which includes Personalization Alignment (P), Content Quality (Q), Factual Reliability (R), Judge LLM, and Final Score Aggregation, to assess personalized deep research capabilities.
- The PDRB benchmark consists of 250 personalized user-task queries derived from 50 diverse research tasks and 25 authentic user profiles, enabling systematic evaluation of both task complexity and persona-driven adaptation.
- The PQR Evaluation Framework employs a Judge LLM to dynamically generate criteria and assign weights for evaluating reports across personalization, quality, and factual reliability dimensions, providing a holistic measure of agent utility.

---

[Cogito, Ergo Ludo: An Agent that Learns to Play by Reasoning and Planning](http://arxiv.org/abs/2509.25052)

- CEL (Cogito, ergo ludo): introduces a novel agent architecture that learns by explicitly reasoning and planning, leveraging an LLM to build and refine a human-readable world model and strategic playbook, operating through in-episode decision-making and post-episode reflection phases.
- The agent's operational cycle involves an LLM-driven Language-based World Model for predicting outcomes and a Language-based Value Function for assessing state desirability during decision-making.
- Post-episode, the agent refines its explicit Environmental Rules via Rule Induction and distills actionable Strategic Playbook advice through Strategy and Playbook Summarization, continuously improving its understanding and strategy.

---

[PanoWorld-X: Generating Explorable Panoramic Worlds via Sphere-Aware Video Diffusion](http://arxiv.org/abs/2509.24997)

- PanoWorld-X: introduces a novel framework for high-fidelity and controllable panoramic video generation, with its PanoExplorer Dataset (large-scale dataset), Explorable Sphere-Aware DiT Block (core generation module), Original Global Attention Branch (pre-trained DiT component), Exploration-Aware Attention (trajectory control), Sphere-Aware Attention (spherical geometry perception), and Encoder (feature extraction), enabling explorable panoramic videos with diverse camera trajectories.
- The framework addresses limitations of narrow field-of-view and insufficient camera controllability by leveraging a curated dataset and a Sphere-Aware Diffusion Transformer architecture that re-projects equirectangular features onto a spherical surface.
- PanoWorld-X achieves superior performance in generation quality, motion range, and control precision, demonstrating its potential for real-world applications in immersive virtual reality and embodied intelligence.

---

[Path Diffuser: Diffusion Model for Data-Driven Traffic Simulator](http://arxiv.org/abs/2509.24995)

- Path Diffuser (PD): introduces a two-stage diffusion framework for data-driven traffic simulation, jointly generating agent pose initializations and trajectories conditioned on map data, free from historical context.
- The framework integrates an Agent Initialization Diffusion Model and a Trajectory Generation Diffusion Model, both leveraging Heterogeneous Message Passing Layers and Differential Transformer Layers for robust interaction modeling.
- PD further incorporates a Frenet Candidate Generator to provide motion primitive-based priors, enhancing trajectory diversity and ensuring road-compliant generation, particularly in out-of-distribution map conditions.

---

[A-MEMGUARD: A PROACTIVE DEFENSE FRAMEWORK FOR LLM-BASED AGENT MEMORY](http://arxiv.org/abs/2510.02373)

- A-MemGuard (Agent-Memory Guard): introduces a proactive defense framework for LLM agent memory, combining consensus-based validation (detects anomalies) and a dual-memory structure (stores and utilizes lessons) to enable self-checking and self-correcting memory.
-The framework operates by retrieving multiple related memories, generating parallel reasoning paths, and identifying deviations from consensus to flag anomalous entries.
-Detected flaws are distilled into "lessons" stored in a dedicated lesson memory, which then guides the agent to avoid repeating past errors through proactive deliberation and action revision.

---

[When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We?](http://arxiv.org/abs/2509.24927)

- V2X Cooperative Perception System: introduces an empirical study of V2X cooperative perception systems, analyzing their performance across various sensor configurations, cooperative agent types, fusion schemes, and communication conditions, and identifying six error patterns.
- The study evaluates system performance through offline and online testing, revealing that LiDAR-based configurations achieve the highest perception performance and that communication issues significantly increase ADS violations.
- The findings highlight critical vulnerabilities in cooperative perception systems, emphasizing the need for robust design and testing methodologies to mitigate errors and enhance reliability in autonomous driving.

---

[When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training](http://arxiv.org/abs/2509.24923)

- Meta-Bandit LLM Training Framework: introduces a systematic comparison of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) paradigms for training LLM Agents (decision-making models) on Multi-armed Bandit (MAB) Environments (simulated task settings), utilizing various Reward Signal Modules (reward generation mechanisms) and Oracle Policies (optimal exploration algorithms) within a Training Infrastructure (distributed training system).
- The framework demonstrates that RL-trained policies achieve lower regret and robust generalization to longer horizons and out-of-distribution environments, often outperforming SFT, while both improve upon pre-trained LLMs.
- Behavioral analysis reveals that performance gains often stem from more sophisticated but greedier exploitation, with agents prematurely abandoning exploration, highlighting the need for robust exploratory behavior.

---

[DRCP: Diffusion on Reinforced Cooperative Perception for Perceiving Beyond Limits](http://arxiv.org/abs/2509.24903)

- DRCP (Diffusion on Reinforced Cooperative Perception): introduces a real-time cooperative perception framework that integrates PPXX (Precise-Pyramid-Cross-Modal-Cross-Agent) fusion module for cross-modal and cross-agent feature fusion, and MDMA (Mask-Diffusion-Mask-Aggregation) for diffusion-based BEV feature refinement, to enhance robustness in dynamic driving environments.
- The framework leverages Intrin-RG-Attn (Camera-Intrinsics-Aware Radian Division) for precise camera-LiDAR feature alignment and an Integrated Pyramid Fusion with Adaptive Convolution at Final BEV for robust multi-scale and multi-agent feature aggregation.
- MDMA further refines BEV features through a lightweight, single-step diffusion process, including Seed condition extraction, Forward perturbation, Single-step conditioned denoising, and Residual fusion, to align representations with task-optimal manifolds.

---

[SOCRATIC-ZERO: BOOTSTRAPPING REASONING VIA DATA-FREE AGENT CO-EVOLUTION](http://arxiv.org/abs/2509.24726)

- Socratic-Zero: introduces a fully autonomous framework for bootstrapping reasoning, with Teacher (guides co-evolution, verifies solutions, refines problems), Solver (generates solutions, learns from feedback), and Generator (produces questions, imitates Teacher strategy) agents, where the system generates high-quality training data from minimal seed examples through co-evolution.
- The Solver refines its reasoning via preference learning (DPO) from successful and failed trajectories, while the Teacher adaptively crafts challenging questions based on Solver's weaknesses using its verification and problem refinement functions.
- The Generator distills the Teacher's question-design strategy using value-weighted supervised fine-tuning (WSFT) to enable scalable, high-fidelity curriculum generation, creating a self-improving closed-loop system without pre-existing tasks or labels.

---

[PhysiAgent: An Embodied Agent Framework in Physical World](http://arxiv.org/abs/2509.24524)

- PhysiAgent: introduces a training-free embodied agent framework that integrates VLMs and VLAs using Planner, Monitor, Reflector, Memory, and Embodied Toolbox components to enable autonomous self-regulation and dynamic adaptation in physical environments.
- The framework addresses generalization challenges by establishing an adaptive feedback loop between VLMs and VLAs, allowing VLMs to refine strategies based on real-time proficiency feedback.
- PhysiAgent demonstrates significant improvements in task-solving performance on complex real-world robotic tasks, showcasing effective self-regulation, coherent tool collaboration, and adaptive evolution.

---

[FuncPoison: Poisoning Function Library to Hijack Multi-agent Autonomous Driving Systems](http://arxiv.org/abs/2509.24408)

- FuncPoison: introduces a novel function-level poisoning attack targeting multi-agent autonomous driving systems, exploiting vulnerabilities in the function call mechanism by injecting adversarial patterns into function descriptions within the shared function library.
- The attack unfolds in three stages: poisoning and hijacking, function call manipulation, and cross-agent propagation, enabling structured, template-compliant malicious calls that propagate through agent communication chains.
- FuncPoison achieves high effectiveness and stealth, bypassing prompt and behavior-level defenses by exploiting trust in the function call interface and inter-agent reasoning chains, raising concerns about LLM-based autonomous driving system reliability.

---

[Autonomous Detection and Coverage of Unknown Target Areas by Multi-Agent Systems](http://arxiv.org/abs/2509.24399)

- Composite Motion Controller: introduces a novel coverage control algorithm for multi-agent systems, enabling agents to autonomously detect and cover unknown target areas by integrating a dynamically constructed density function with Centroidal Voronoi Tessellation (CVT) for optimal spatial distribution and Control Barrier Functions (CBFs) for collision avoidance.
- The framework guides agents to converge towards an optimal coverage configuration by iteratively adjusting their positions based on sensor-detected information and the evolving density distribution, ensuring comprehensive coverage of all significant regions.
- This method ensures safety by preventing inter-agent collisions and maintaining non-overlapping sensor coverage, thereby enhancing both exploration efficiency and system robustness in complex, unknown environments.

---

[Agentic Services Computing](http://arxiv.org/abs/2509.24380)

- ASC (Agentic Services Computing): introduces a lifecycle-driven framework for intelligent service agents, integrating Services Computing (engineering principles/lifecycle management), Multi-Agent Systems (autonomy/coordination/social behavior), and LLM-based Agents (cognitive capabilities/adaptability/generalization) across Design Phase (architecture/roles definition), Deployment Phase (orchestration/CI/CD), Operation Phase (monitoring/control), and Evolution Phase (learning/adaptation) phases, structured by Perception, Context, and Environment Modeling, Autonomous Decision-Making and Task Execution, Multi-Agent Collaboration and Organization, and Evaluation, Alignment, and Trustworthiness dimensions.
- The framework details mechanisms for agents to perceive multimodal environments, make goal-driven decisions, collaborate dynamically, and ensure ethical alignment and trustworthiness throughout their operational lifecycle.
- This holistic approach redefines services as autonomous, adaptive, and socially embedded entities, addressing challenges in scalability, safety, and governance for next-generation intelligent service ecosystems.

---

[MAS2: SELF-GENERATIVE, SELF-CONFIGURING, SELF-RECTIFYING MULTI-AGENT SYSTEMS](http://arxiv.org/abs/2509.24323)

- MAS2 (Self-Generative, Self-Configuring, Self-Rectifying Multi-Agent Systems): introduces a paradigm for recursive self-generation, utilizing a tri-agent team comprising a Generator Agent (architects high-level workflow template), an Implementor Agent (instantiates template with LLM backbones), and a Rectifier Agent (monitors, adapts execution in real-time), all trained via Collaborative Tree Optimization.
- This framework dynamically composes and adaptively rectifies task-specific multi-agent systems in response to real-time demands, transcending static "generate-once-and-deploy" paradigms.
- The system achieves superior competence, Pareto-optimal cost-performance, and cross-backbone generalization by internalizing construction responsibilities and leveraging value-guided specialization.

---

[Learning to Sample: Reinforcement Learning-Guided Sampling for Autonomous Vehicle Motion Planning](http://arxiv.org/abs/2509.24313)

- Hybrid Motion Planning Framework: introduces a system that combines reinforcement learning for goal state sampling with an analytical planner for trajectory generation and evaluation, aiming to improve sampling efficiency and maintain safety.
- The framework utilizes a World Model (WM) to encode structured observations into latent states, which an RL agent then uses to propose high-level goal specifications for the ego vehicle.
- This approach significantly reduces the number of required samples and runtime while maintaining planning quality and ensuring verifiable trajectory generation for autonomous vehicles.

---

[Bridging the behavior-neural gap: A multimodal AI reveals the brain's geometry of emotion more accurately than human self-reports](http://arxiv.org/abs/2509.24298)

- Machine-Behavioral Paradigm: introduces a novel framework that leverages LLMs and MLLMs as cognitive agents to perform large-scale similarity judgments on emotionally evocative videos.
- This framework utilizes a triplet odd-one-out behavioral paradigm to generate millions of similarity judgments, which are then used by SPoSE to learn 30-dimensional affective embedding spaces.
- The learned representations are subsequently compared to human brain activity using Representational Similarity Analysis and Voxel-wise Neural Encoding to bridge the behavior-neural gap in affective science.

---

[A BIOLOGICALLY INTERPRETABLE COGNITIVE ARCHITECTURE FOR ONLINE STRUCTURING OF EPISODIC MEMORIES INTO COGNITIVE MAPS](http://arxiv.org/abs/2510.03286)

- Biologically Interpretable Cognitive Architecture: introduces a novel cognitive architecture for online structuring of episodic memories into cognitive maps, utilizing first-level states (H(1)) for episodic memory, second-level states (H(2)) for cognitive maps, Successor Features (SF) for similarity, and Hebbian-like learning rules for updates.
- This architecture integrates the Successor Features framework with episodic memories, enabling incremental, online learning through agent-environment interaction in partially observable grid-worlds.
- The model employs local, biologically plausible learning rules to autonomously organize memories into structured representations without centralized optimization, bridging computational neuroscience and AI.

---

#### 28th September 2025

[EFFICIENT MULTI-TURN RL FOR GUI AGENTS VIA DE-COUPLED TRAINING AND ADAPTIVE DATA CURATION](http://arxiv.org/abs/2509.23866)

- DART (Decoupled Agentic RL Training): introduces a decoupled RL framework for GUI agents, coordinating Env Cluster (provides parallel GUI environments), Rollout Service (generates trajectories, performs inference), Data Manager (stores, filters, curates trajectories), and Trainer (updates policy model asynchronously) to enhance training efficiency and data quality.
- The framework significantly improves GPU and environment utilization through non-blocking communication, asynchronous training, and rollout-wise trajectory sampling.
- DART also incorporates an adaptive data curation scheme, including dynamic rollout frequency, trajectory length, an experience pool, high-entropy-driven step optimization, and distribution alignment, to stabilize and accelerate learning.

---

[GUI-SHEPHERD: RELIABLE PROCESS REWARD AND VERIFICATION FOR LONG-SEQUENCE GUI TASKS](http://arxiv.org/abs/2509.23738)

- GUI-Shepherd (Process Reward Model): introduces a Process Reward Model (PRM) that provides dense, step-by-step feedback to guide agents, including a Data Collection Pipeline (dual pipeline) for diverse data, a Data Annotation Process (hybrid) with Human Annotators and GPT-40 (for rationales), and functions as both a Reward Provider (for RL training) and an Inference Verifier (for action selection), utilizing a Policy Model (UI-TARS-1.5-7B) and a vLLM Service (for PRM deployment) to enhance performance in long-sequence GUI tasks.
- The framework is trained on a 52k-sample dataset with human-annotated scores and GPT-40 generated rationales, enabling it to serve as a reward provider for online RL and a verifier for inference across diverse GUI settings.
- GUI-Shepherd significantly improves success rates on AndroidWorld and AndroidControl benchmarks, demonstrating the critical role of high-fidelity process supervision for capable and generalizable GUI agents.

---

[Precise HDV Positioning through Safety-Aware Integrated Sensing and Communication in a Value-of-Information-Driven 6G V2X System](http://arxiv.org/abs/2510.02363)

- VoI-driven Two-Time Scale MA-DDPG Framework: introduces a novel framework for enhancing vehicular safety and positioning accuracy in 6G V2X networks, with a VoI metric (prioritizes safety-critical data), two-time-scale sequential decision process (models sensing-communication-control problem), MADDPG algorithm (solves multi-agent decision process), ISAC (enables HDV position sensing), CAVs (agents making distributed decisions), RSUs (agents making distributed decisions), actor network (approximates agent policy), critic network (approximates state-value function), and replay buffer (stores training experiences).
- This framework prioritizes safety-critical information transmission and optimizes resource allocation in mixed-autonomy environments, mitigating bandwidth and latency constraints in ultra-dense traffic.
- The approach models sensing, communication, and control tasks as a multi-agent reinforcement learning problem, achieving significant safety gains and reducing collision risk.

---

[The AI Agent Code of Conduct: Automated Guardrail Policy-as-Prompt Synthesis](http://arxiv.org/abs/2509.23994)

- Policy as Prompt: introduces a novel framework that automates the translation of unstructured design documents into verifiable, real-time guardrails, with ARTIFACTS, POLICY-TREE-GEN, Policy-Gen LLM, Policy Tree, POLICY-AS-PROMPT-GEN, Input Classifier, Output Auditor, Policy as Prompt, HUMAN REVIEW, and POLICY DEPLOYMENT, where it uses LLMs to interpret and enforce natural language policies by applying contextual understanding and the principle of least privilege.
- The system first ingests technical artifacts to construct a verifiable policy tree, which is then compiled into lightweight, prompt-based classifiers that audit agent behavior at runtime.
- This approach provides a scalable and auditable pipeline that bridges the critical policy-to-practice gap, paving the way for verifiably safer and more regulatable AI.

---

[ADVANCING MULTI-AGENT TRAFFIC SIMULATION VIA R1-STYLE REINFORCEMENT FINE-TUNING](http://arxiv.org/abs/2509.23993)

- SMART-R1: introduces a novel R1-style reinforcement fine-tuning paradigm for next-token prediction models, utilizing a multi-stage training paradigm, Open-Loop NTP, Closed-Loop SFT, Closed-Loop RFT, Metric-oriented Policy Optimization (MPO), Reward Model, Reference Model, Policy Model, Tokenization, and Attention Layers, to better align multi-agent traffic simulation behavior with human preferences and evaluation metrics.
- The approach integrates a metric-oriented policy optimization algorithm and an iterative "SFT-RFT-SFT" training strategy to maximize performance gains and enhance overall simulation realism.
- SMART-R1 achieves state-of-the-art performance on the Waymo Open Sim Agents Challenge by balancing metric-driven objectives with the preservation of learned behavioral distributions, mitigating catastrophic forgetting.

---

[LLM/Agent-as-Data-Analyst: A Survey](http://arxiv.org/abs/2509.23988)

- LLM/Agent-as-Data-Analyst: introduces a survey of LLM and agent techniques for data analysis, covering Structured Data Analysis, Semi-Structured Data Analysis, Unstructured Data Analysis, and Heterogeneous Data Analysis, where LLMs enable complex data understanding and autonomous pipeline orchestration.
- The survey distills five key design goals for intelligent data analysis agents, including semantic-aware design, modality-hybrid integration, autonomous pipelines, tool-augmented workflows, and open-world task support.
- It outlines remaining challenges and proposes practical directions for advancing LLM/Agent-powered data analysis across diverse data modalities and interaction paradigms.

---

[TUSOAI: AGENTIC OPTIMIZATION FOR SCIENTIFIC METHODS](http://arxiv.org/abs/2509.23986)

- TusoAI (Agentic Optimization for Scientific Methods): introduces an agentic AI system that autonomously develops and optimizes computational methods for scientific tasks by integrating structured domain knowledge into a knowledge tree and performing iterative, domain-specific optimization.
- The system employs multiple LLM-based agents (Apaper, Acate, Ainstr, Ainit, Aoptim, Afeedback) to gather domain knowledge, build a two-level knowledge tree of optimization strategies and instructions, and iteratively refine candidate solutions.
- TusoAI leverages Bayesian updates for adaptive category sampling and diagnostic feedback to guide model improvement, demonstrating superior performance across diverse scientific tasks.

---

[RETHINKING REWARD MISCALIBRATION OF GRPO IN AGENTIC RL](http://arxiv.org/abs/2509.23870)

- GCD (Generative Classification Disentanglement): introduces a novel training paradigm that enhances GRPO by training the actor model to simultaneously act as a classifier, utilizing an auxiliary classification objective and a critic generator to classify actions as good or bad, alongside a prompt-based correction strategy.
- This approach aims to alleviate gradient coupling by disentangling the embeddings of good and bad actions, thereby preventing beneficial updates from inadvertently reinforcing similar-looking flawed actions.
- The framework also incorporates a prompt-based correction strategy to guide the agent away from common errors by injecting explicit instructions, particularly when the probability of flawed actions is high.

---

[AgentGuard: Runtime Verification of AI Agents](http://arxiv.org/abs/2509.23864)

- AgentGuard: introduces a runtime verification framework for agentic AI systems, providing continuous quantitative assurance by capturing raw I/O and abstracting it into formal events, dynamically building and updating Markov Decision Processes, verifying quantitative properties using probabilistic model checking, and presenting guarantees with alerts or automated responses.
- This framework shifts verification from static, offline analysis to a dynamic, ongoing process, enabling real-time monitoring and adaptation to non-stationary environments.
- The framework addresses the unpredictability and emergent behaviors of LLM-based agents by offering probabilistic guarantees on their performance and safety.

---

[FEDAGENTBENCH: TOWARDS AUTOMATING REAL-WORLD FEDERATED MEDICAL IMAGE ANALYSIS WITH SERVER-CLIENT LLM AGENTS](http://arxiv.org/abs/2509.23803)

- FedAgentBench: introduces an agent-driven FL framework for automating real-world federated medical image analysis, with all Federated Medical Imaging Workspace (W), Multi-agent Coordination System (A), LLM Agents, Tools, FL Algorithms, FL Environments, and LangGraph Architecture components, where it enables autonomous coordination and execution of FL workflows using specialized LLM agents across server and client environments.
- The framework integrates seven role-specialized LLM agents (S1-S4 on the server, C1-C3 on clients) to manage four distinct FL phases: Client Selection, Data Preprocessing, Label Harmonization, and Federated Training.
- The system leverages a comprehensive suite of 40 FL algorithms and 16 tools, operating within a federated medical imaging workspace to ensure privacy-preserving and modular FL deployment across diverse healthcare environments.

---

[A First Look at Privacy Risks of Android Task-executable Voice Assistant Applications](http://arxiv.org/abs/2509.23680)

- Empirical Study on Privacy Risks in Android Task-executable VAs: introduces a user-centric comprehensive empirical study on privacy risks in Android task-executable VAs, which includes VA collection, operational characterization, privacy declaration cross-checking, privacy threat model identification, and actionable recommendations, aiming to holistically examine privacy risks in these applications.
- The research collects ten mainstream VAs, analyzes their operational characteristics, and cross-checks privacy declarations across six sources, revealing widespread inconsistencies and three significant privacy threat models.
- The study's findings highlight privacy misdisclosure in mega apps, privilege escalation via inter-application interactions, and abuse of Google system applications, offering actionable recommendations for practitioners and autonomous AI agents.

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
