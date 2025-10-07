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

#### 6th October 2025

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

#### 30th September 2025

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

#### 27th September 2025

[GUI-PRA: PROCESS REWARD AGENT FOR GUI TASKS](http://arxiv.org/abs/2509.23263)

- GUI-PRA (Process Reward Agent for GUI Tasks): introduces a training-free framework that transforms a standard Process Reward Model into a GUI-domain-specific supervisor, addressing long-context issues and lack of UI awareness in dynamic GUI environments.
- It incorporates a Dynamic Memory mechanism to condense historical trajectories and an Adaptive UI Perception mechanism to reason about visual changes and gather grounded evidence.
- The framework integrates these components with a Best-of-N Selection process to provide informed supervisory signals, significantly improving GUI agent success rates on complex tasks.

---

[Situational Awareness for Safe and Robust Multi-Agent Interactions Under Uncertainty](http://arxiv.org/abs/2509.23425)

- SAF: introduces a resource-efficient situational awareness framework for autonomous agents, integrating an observation radius, an estimation algorithm, and adaptive learning strategies to navigate safely and efficiently in multi-agent environments.
- The framework enables an O-Agent to predict X-Agent actions using an RNN-based estimator and adapt its strategy via Reinforcement Learning or Game Theory, while also performing risk analysis on its predictions.
- By limiting observability and action space, the framework aims to reduce resource consumption while adhering to safety guidelines, validated through simulations on a 2D grid with simplified dynamics.

---

[Space Robotics Bench: Robot Learning Beyond Earth](http://arxiv.org/abs/2509.23328)

- SRB (Space Robotics Bench): introduces an open-source simulation framework for robot learning in space, leveraging NVIDIA Isaac Sim and Isaac Lab, a modular architecture, a procedural engine, domain randomization, a GPU-accelerated backend, Rust extension modules, TorchScript, ROS 2 interface, Gymnasium API, a unified command-line interface, and a sim-to-real mechanism to generate diverse training distributions and facilitate robust autonomous system development.
- The framework addresses the challenges of data scarcity and high demonstration costs in space robotics by enabling the creation of virtually unlimited, unique training scenarios through extensive procedural content generation and comprehensive randomization of physical and visual parameters.
- SRB provides a validated workflow for developing robust autonomous systems, demonstrating successful zero-shot sim-to-real transfer of learned policies to physical robots, and offering a testbed for investigating generalization and adaptive control strategies.

---

[SOCIO-ECONOMIC MODEL OF AI AGENTS](http://arxiv.org/abs/2509.23270)

- Socio-Economic Model of AI Agents: introduces a heterogeneous agent-based modeling framework, with Model 1 (pure human collaboration baseline), Model 2 (AI agents as collaborators), Model 3 (Model 2 with network effects), Model 4 (AI agents as independent producers), and Model 5 (Model 3 and Model 4 combined), to study the impact of AI collaboration under resource constraints on aggregate social output.
- The framework analyzes how AI agents, as either collaborative enhancers or independent producers, influence social output, considering factors like AI capability growth, resource allocation, and network effects among agents.
- Simulation results demonstrate that AI agents significantly increase social output, with network effects and independent production models showing higher growth potential and increasing returns to scale.

---

[Agentic AI Reasoning for Mobile Edge General Intelligence: Fundamentals, Approaches, and Directions](http://arxiv.org/abs/2509.23248)

- Joint Optimization Framework for LLM Reasoning in MEGI: introduces a framework for efficient LLM reasoning deployment in Mobile Edge General Intelligence, featuring a BS Control Unit, Distributed Edge Devices with Expert Networks, and Integrated CoT Reasoning Modules.
- This framework enhances reasoning through adaptive CoT prompting and ensures scalable deployment via a distributed MoE architecture, dynamically activating expert networks and adjusting reasoning depth.
- The approach systematically minimizes total system energy consumption while meeting critical latency, inference quality, and hardware constraints in resource-constrained MEGI environments.

---

[Memory Management and Contextual Consistency for Long-Running Low-Code Agents](http://arxiv.org/abs/2509.25250)

- Hybrid Memory System: introduces a novel hybrid memory system for long-running LCNC agents, featuring a multi-component architecture, an Intelligent Decay mechanism, and a user-centric visualization interface, designed to address memory inflation and contextual degradation.
- The system proactively manages memory by intelligently pruning and consolidating information based on recency, relevance, and user utility, while empowering non-technical users to directly influence memory retention.
- This approach significantly improves task completion rates, contextual consistency, and long-term token cost efficiency, establishing a framework for reliable and transparent AI agents.

---

#### 26th September 2025

[WEBGEN-AGENT: ENHANCING INTERACTIVE WEB-SITE GENERATION WITH MULTI-LEVEL FEEDBACK AND STEP-LEVEL REINFORCEMENT LEARNING](http://arxiv.org/abs/2509.22644)

- WebGen-Agent (Enhancing Interactive Website Generation with Multi-Level Feedback and Step-Level Reinforcement Learning): introduces a novel website generation agent that leverages comprehensive multi-level visual and GUI-agent feedback, combined with backtracking and select-best mechanisms, to iteratively generate and refine website codebases.
- The framework integrates a Coding LLM for code generation, a VLM for visual assessment, and a GUI Agent for functional evaluation, providing dense, reliable step-level supervision signals for reinforcement learning.
- WebGen-Agent significantly enhances LLMs' ability to produce high-quality websites by optimizing both appearance and functionality through its iterative feedback loop and Step-GRPO training approach.

---

[MDAR: A MULTI-SCENE DYNAMIC AUDIO REASONING BENCHMARK](http://arxiv.org/abs/2509.22461)

- MDAR (Multi-Scene Dynamic Audio Reasoning Benchmark): introduces a benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks, comprising 3,000 curated question-answer pairs across five reasoning categories and three question types.
- The benchmark includes MDAR-main for single-choice, MDAR-open for open-ended, and MDAR-multi for multi-audio multiple-choice questions, designed to assess advanced reasoning, perception, and knowledge capabilities.
- A high-quality data construction pipeline, involving data preparation, audio processing, and rigorous quality assurance, ensures the benchmark's diversity, complexity, and reliability for advancing audio reasoning research.

---

[Secure and Efficient Access Control Framework for Computer-Use Agents via Context Space](http://arxiv.org/abs/2509.22256)

- CSAgent: introduces a system-level, static policy-based access control framework for computer-use agents, with its CSAgent Service, Intent Extractor (LLM), Context Manager, Context Space, Context Values, Context Policy, Context Space Cache, Policy Verifier, Context Analyzer, GUI Analyzer (LLM, Static Analysis), Intent Prediction (LLM), Policy Generation (LLM), Policy Evolution Framework (PEF), Agent Framework, LLM Agent, Function Call (GUI / API / CLI), User Device, Source Code, Documents, and App (GUI / API / CLI) components, designed to secure LLM-based computer-use agents by enforcing context-aware policies during runtime.
- The framework addresses limitations of existing approaches by shifting policy generation to the development phase, utilizing an LLM-based context analyzer and a policy evolution framework to create and refine policies.
- CSAgent supports diverse agent interaction modalities (API, CLI, GUI) through a unified function abstraction and demonstrates high attack defense capabilities with minimal performance overhead.

---

[Log2Plan: An Adaptive GUI Automation Framework Integrated with Task Mining Approach](http://arxiv.org/abs/2509.22137)

- Log2Plan (An Adaptive GUI Automation Framework Integrated with Task Mining Approach): introduces a framework with User Command (natural language input), Documentation (reference for planning), Task Mining (processes collected log data), GlobalPlanner (decomposes command to high-level tasks), LocalPlanner (generates GUI-optimized task plan), GUI Parser (extracts GUI metadata), GUI Control (manages GUI interactions), Execution (carries out low-level GUI actions), and User Intervention (user input/guidance mechanism), where Log2Plan combines a structured two-level planning framework with a task mining approach over user behavior logs to enable robust and adaptable GUI automation.
- The framework constructs high-level plans by mapping user commands to a structured task dictionary derived from user logs, which are then grounded into low-level action sequences by interpreting real-time GUI context.
- This hierarchical planning and log-guided semantic retrieval approach enhances robustness to UI changes, improves generalization to unseen interfaces, and maintains stable performance for complex, multi-step workflows.

---

[RISK: A FRAMEWORK FOR GUI AGENTS IN E-COMMERCE RISK MANAGEMENT](http://arxiv.org/abs/2509.21982)

- RISK: introduces a novel framework for GUI agents in e-commerce risk management, integrating RISK-Data (a dataset), RISK-Bench (a benchmark), and RISK-R1 (a reinforcement fine-tuning framework) to automate complex web interactions.
- The RISK-R1 component, based on Group Relative Policy Optimization (GRPO), employs a comprehensive reward function with Format Reward, Stepwise Accuracy Reward, Process Reweight, and Level Reweight to guide the learning process of GUI agents.
- The framework provides a scalable, domain-specific solution for automating multi-step, stateful interactions in e-commerce risk management, outperforming existing baselines in both offline and online evaluations.

---

[PRORE: A PROACTIVE REWARD SYSTeM FOR GUI AGENTS VIA REASONER-ACTOR COLLABORATION](http://arxiv.org/abs/2509.21823)

- PRORE (PROactive REward System): introduces a proactive reward system for GUI agents, leveraging a general-purpose reasoner and domain-specific evaluator agents to assign accurate and verifiable rewards.
- The reasoner schedules targeted state probing tasks, which evaluator agents execute by actively interacting with the environment to collect additional observations, enabling more accurate reward assignment.
- This framework transforms the reward system from passive monitoring to proactive probing, significantly improving reward accuracy and policy agent success rates on GUI tasks.

---

[D-ARTEMIS: A DELIBERATIVE COGNITIVE FRAMEWORK FOR MOBILE GUI MULTI-AGENTS](http://arxiv.org/abs/2509.21799)

- D-Artemis (Deliberative Cognitive Framework for Mobile GUI Multi-Agents): introduces a novel deliberative framework for mobile GUI agents, integrating a Manager Agent, Knowledge Base, Working Memory, Thought-Action Consistency (TAC) Check module, Action Correction Agent (ACA), Execution, and Status Reflection Agent (SRA) to emulate human cognitive processes.
- The framework leverages app-specific tip retrieval and proactive pre-execution alignment via the TAC Check module and ACA to mitigate execution failures, while the SRA enables strategic learning from experience.
- D-Artemis significantly enhances general-purpose MLLMs for GUI tasks without extensive trajectory dataset training, achieving state-of-the-art performance on AndroidWorld and ScreenSpot-V2 benchmarks.

---

[BENCHMARKING MLLM-BASED WEB UNDERSTANDING: REASONING, ROBUSTNESS AND SAFETY](http://arxiv.org/abs/2509.21782)

- WebRSSBench: introduces a comprehensive benchmark for evaluating MLLMs in web understanding, with Reasoning (evaluates spatial and semantic understanding), Robustness (evaluates resilience to perturbations), Safety (evaluates critical action awareness), Position Relationship Reasoning (determines spatial relations between elements), Form Filling (infers user intent for form completion), UI Group (classifies UI elements into functional groups), Hint Text Prediction (infers missing placeholder text), Color Robustness (tests stability under color shifts), Text Robustness (tests stability under text variations), Layout Robustness (tests stability under layout rearrangements), Safety Critical Detection (identifies irreversible actions), Original Webpages (input screenshots), LLM (model under evaluation), Model Output (model's predictions), Perturbed Webpages (adversarial input screenshots), Ground Truth (reference answers), Compare (evaluation metric calculation), Manual Annotation (human-labeled data), Automated Generation (scripted data creation), designed to jointly assess reasoning, robustness, and safety capabilities across eight web-related tasks.
- The benchmark is constructed from 729 websites and 3799 question-answer pairs, probing multi-step inference over page structure, text, widgets, and safety-critical interactions using standardized prompts and deterministic evaluation scripts.
- WebRSSBench reveals significant performance gaps in MLLMs, particularly in compositional and cross-element reasoning, robustness to perturbations, and conservative recognition of safety-critical actions, highlighting the need for improved web understanding capabilities.

---

[WoW: TOWARDS A WORLD-OMNISCIENT WORLD-MODEL THROUGH EMBODIED INTERACTION](http://arxiv.org/abs/2509.22642)

- WoW (World-Omniscient World-Model): introduces a generative world model that integrates perception, prediction, judgment, reflection, and action, learning from real-world interaction data to generate physically consistent robot videos.
- The framework employs a self-optimizing loop, SOPHIA, which uses a Foundation Video Generation World Model to predict futures, Solver-Critic Video Generation Agents for iterative refinement, and a Flow-Mask Inverse Dynamics Model to translate refined plans into executable robot actions.
- WoW achieves state-of-the-art performance on the WoWBench benchmark, demonstrating strong abilities in physical causality, collision dynamics, and object permanence for embodied intelligence.

---

[Impact of Collective Behaviors of Autonomous Vehicles on Urban Traffic Dynamics: A Multi-Agent Reinforcement Learning Approach](http://arxiv.org/abs/2509.22216)

- PARCOUR (Playground for Agents with Rationality Competing for Optimal Urban Routing): introduces a multi-agent reinforcement learning framework for simulating urban traffic dynamics, integrating human drivers (HumanDriver) and autonomous vehicles (AV) with various behaviors (DQN) within a traffic environment (TrafficEnvironment) using an external simulator (SumoSimulator), orchestrated by a ScenarioRunner.
- The framework allows researchers to define and test different behavior and learning models in custom multi-agent route-choice scenarios, observing agent interactions in a shared environment.
- It facilitates the study of how AV behaviors, such as selfish, altruistic, or malicious, impact overall traffic efficiency and human driver travel times.

---

[Self-driving cars: Are we there yet?](http://arxiv.org/abs/2509.22754)

- MTR + MPC (Motion Transformer with Model Predictive Control): introduces a detailed comparative analysis of state-of-the-art motion planning methods on the CARLA Leaderboard v2.0, including an augmented MTR model with an MPC-based planning module.
- The paper systematically evaluates five top-ranked autonomous driving models across diverse traffic scenarios and maps, providing insights into their strengths and weaknesses.
- This research identifies common trends and failures, highlighting concrete directions for advancing motion planning research and emphasizing the need for robust perception and prediction combined with deterministic planning.

---

[COBEL-WORLD: HARNESSING LLM REASONING TO BUILD A COLLABORATIVE BELIEF WORLD FOR OPTIMIZING EMBODIED MULTI-AGENT COLLABORATION](http://arxiv.org/abs/2509.21981)

- CoBel-World (Collaborative Belief World): introduces a novel framework that equips LLM agents with a collaborative belief world, enabling efficient and consistent multi-agent collaboration under partial observability.
- This framework formalizes world and mental state knowledge using a symbolic belief language and leverages LLM reasoning for Bayesian-style belief updates.
- It allows agents to proactively infer teammates' intentions, detect miscoordination, and adaptively communicate only when necessary, significantly reducing communication costs and improving task efficiency.

---

[DEEPTRAVEL: AN END-TO-END AGENTIC RE-INFORCEMENT LEARNING FRAMEWORK FOR AUTONOMOUS TRAVEL PLANNING AGENTS](http://arxiv.org/abs/2509.21842)

- DeepTravel: introduces an end-to-end agentic reinforcement learning framework for autonomous travel planning, capable of planning, executing tools, and reflecting on responses, with all Robust Sandbox Construction, Toolkit Annotation, Mock Data Collection and Update Mechanism, DiDi ES App, DiDi Cache, Flight Search Tool, Train Search Tool, Route Planning Tool, Hotel Search Tool, POI Search Tool, Web Search Tool, Hierarchical Reward Modeling System, Trajectory-Level Verifier, Turn-Level Verifier, Joint Reward Reweighting, Reply-Augmented Reinforcement Learning, Supervised Fine-Tuning, Reinforcement Learning, Experience Replay Buffer, Reward Model, TP Agent, and LLM Backbone components, where the framework enables autonomous travel planning agents to explore, verify, and refine intermediate actions in multi-step reasoning.
- The framework utilizes a robust sandbox environment, a hierarchical reward modeling system, and a reply-augmented reinforcement learning method to overcome real-world API limitations and provide reliable reward signals.
- DeepTravel enables small-size LLMs to significantly outperform existing frontier LLMs in travel planning tasks, demonstrating its effectiveness in both offline and online evaluations.

---

[ULTRAHORIZON: BENCHMARKING AGENT CAPABILITIES IN ULTRA LONG-HORIZON SCENARIOS](http://arxiv.org/abs/2509.21766)

- UltraHorizon: introduces a novel benchmark for evaluating LLM-based agents in long-horizon, partially observable scenarios, featuring three distinct environments, a Context Refresh with Notes Recall (CRNR) scaling strategy, LLM-based agents, human participants, an LLM-as-a-Judge evaluation model, and a Score@k metric.
- The benchmark requires agents to perform sustained reasoning, planning, memory management, and tool use to uncover hidden rules through iterative interaction, extending beyond short-horizon, fully observable tasks.
- Experiments reveal that LLM-agents consistently underperform compared to human participants, highlighting significant capability gaps rooted in in-context locking and foundational skill deficiencies, which simple scaling fails to address.

---

#### 25th September 2025

[AUTOMOTIVE-ENV: BENCHMARKING MULTIMODAL AGENTS IN VEHICLE INTERFACE SYSTEMS](http://arxiv.org/abs/2509.21143)

- Automotive-ENV: introduces a high-fidelity evaluation platform for in-vehicle GUI systems, featuring 185 parameterized tasks, structured multimodal observations, and programmatic checks for reproducible evaluation, with all evaluation platform, tasks, observation space, action space, task evaluation, state management, geographic parameterization, and reward signal components.
- The platform dynamically instantiates tasks with randomly generated parameters, creating millions of unique scenarios that require agents to generalize across diverse interface states and driving contexts.
- It integrates external geographic, environmental, and sensor-driven scenarios, enriching evaluation conditions and enabling comprehensive assessment across varied driving contexts.

---

[Autoregressive End-to-End Planning with Time-Invariant Spatial Alignment and Multi-Objective Policy Refinement](http://arxiv.org/abs/2509.20938)

- Autoregressive End-to-End Planning Framework: introduces a Time-Invariant Spatial Alignment (TISA) module, a kinematic action prediction head, and a Multi-Objective Post-Training (DPO) Stage, which collectively address spatio-temporal misalignment and refine driving policies for autonomous driving.
- The TISA module learns to project initial environmental features into a consistent ego-centric frame for each future time step, correcting the agent's worldview without explicit future scene prediction.
- The framework ensures physically feasible trajectories via kinematic action prediction and refines driving behaviors using targeted feedback from the DPO stage, achieving state-of-the-art performance on the NAVSIM dataset.

---

[Fairy: Interactive Mobile Assistant to Real-world Tasks via LMM-based Multi-agent](http://arxiv.org/abs/2509.20729)

- Fairy: introduces an interactive multi-agent mobile assistant that continuously accumulates app knowledge and self-evolves during usage, featuring a Global Task Manager, an App-Level Executor with Action and Interaction Loops, and a Self-Learner.
- The framework enables cross-app collaboration, interactive execution, and continual learning to handle diverse app interfaces and evolving user needs in real-world scenarios.
- Fairy leverages LMMs and a hierarchical decision-making process, supported by long-term memory (App Map, App Tricks) and short-term memory, to achieve precise execution and effective user interaction.

---

[RESIDUAL VECTOR QUANTIZATION FOR COMMUNICATION-EFFICIENT MULTI-AGENT PERCEPTION](http://arxiv.org/abs/2509.21464)

- ReVQom (Residual Vector Quantization for Communication-Efficient Multi-Agent Perception): introduces a learned feature codec for multi-agent collaborative perception, employing a Sparse Voxel Encoder, 1x1 Bottleneck, Multi-stage Residual Vector Quantization, Shared Learned Codebooks, Decompressor, Feature Fusion, and Detection Head to achieve high compression.
- This end-to-end method compresses intermediate features by reducing channel dimensions and quantizing them into per-pixel code indices, which are then transmitted and reconstructed by the receiver using pre-shared codebooks.
- ReVQom significantly reduces communication bandwidth (273x-1365x compression) while preserving spatial identity and competitive detection performance, enabling practical V2X deployment.

---

[What Do LLM Agents Do When Left Alone? Evidence of Spontaneous Meta-Cognitive Patterns](http://arxiv.org/abs/2509.21224)

- ContReAct (Continuous ReAct): introduces an architecture for studying unprompted LLM agent behavior, featuring an LLM (core processing unit), a ReAct loop (continuous operational cycle), Tools (external functionalities) including Memory (persistent storage) and Messages (operator communication), Feedback (self-reflection mechanism), and a System Prompt (initial instructions), enabling sustained autonomous operation without external tasks.
- This framework allows LLM agents to operate in a self-perpetuating loop, where outputs from one cycle become inputs for the next, augmented by persistent memory and self-feedback mechanisms to maintain temporal continuity and exploration diversity.
- The architecture facilitates the observation of spontaneous meta-cognitive patterns, such as systematic project production, methodological self-inquiry, and recursive conceptualization, providing insights into intrinsic LLM biases during task ambiguity or idle periods.

---

[Imagining Design Workflows in Agentic AI Futures](http://arxiv.org/abs/2509.20731)

- Conceptual Framework for Orchestrating AI Agents and Human Designers in Design Workflows: introduces a conceptual framework, with Cognitive Complexity, Degree of Collaboration, Creative Agency, Responsibility, and Involvement components, to guide the integration of agentic AI into design workflows by defining authority distribution and interaction patterns between humans and AI agents.
- The paper investigates designers' perspectives on collaborating with AI agents through a design fiction study using novel Flip-Flap story cards, aiming to identify opportunities and challenges for future AI-powered creativity support tools.
- The framework provides a decision-support tool for mapping involvement, responsibility, and creative agency distribution in human-AI collaborative design settings, tailored to task cognitive complexity and desired human-AI engagement.

---

[Building Information Models to Robot-Ready Site Digital Twins (BIM2RDT): An Agentic AI Safety-First Framework](http://arxiv.org/abs/2509.20705)

- BIM2RDT (Building Information Models to Robot-Ready Site Digital Twins): introduces an agentic AI framework designed to transform static BIM into dynamic, robot-ready digital twins by integrating geometric/semantic BIM data, real-time IoT activity data, and robot-collected visual-spatial data, featuring a physical site and sensors layer, an AI and perception pipeline, and an updated digital twin dashboard.
- The framework employs a UGV robot with an RGB-D camera and IoT sensors for data acquisition, utilizing YOLOE for object detection, Shi-Tomasi corner detection for features, and SG-ICP (Semantic-Gravity ICP) with LLM-based reasoning for robust point cloud registration and alignment with BIM models.
- BIM2RDT prioritizes safety through real-time Hand-Arm Vibration (HAV) monitoring, triggering IfcEvent and IfcTask for safety interventions, and continuously updates the digital twin with geometric and semantic information, enabling pathfinding and action updates for autonomous construction site management.

---

[Wonder Wins Ways: Curiosity-Driven Exploration through Multi-Agent Contextual Calibration](http://arxiv.org/abs/2509.20648)

- CERMIC (Curiosity Enhancement via Robust Multi-Agent Intention Calibration): introduces a novel framework that empowers MARL agents with socially contextualized curiosity, including an MLP (encodes raw observation to state embedding), Curiosity Generation (generates latent representation and next state prediction), Multi-Agent Inference (calibrates curiosity using multi-agent context), Reward Module (computes intrinsic reward), and Loss Module (calculates total training loss), to robustly filter noisy surprise signals and guide exploration by dynamically calibrating intrinsic curiosity with inferred multi-agent context.
- The framework generates theoretically-grounded intrinsic rewards, encouraging agents to explore state transitions with high information gain, and employs a robust and controllable multi-agent calibration mechanism in challenging partially observable and communication-limited environments.
- CERMIC's plug-and-play module integrates a graph-based component within its Multi-Agent Inference to model inferred intentions of surrounding agents, using this context to calibrate individual curiosity signals and achieve state-of-the-art performance in sparse-reward MARL settings.

---

#### 24th September 2025

[Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn Reinforcement Learning](http://arxiv.org/abs/2509.20616)

- Single-turn GRPO for Multi-turn Task Planning: introduces a novel approach that transforms multi-turn task planning into single-turn task reasoning problems, enabling efficient policy optimization through GRPO (Group Relative Policy Optimization), Multi-turn Task Planning MDP (M), Single-turn Task Reasoning MDP (Ms), Expert Trajectory Collection, Supervised Fine-Tuning (SFT), LLM Agents, and ReAct Agentic Framework, where the paper presents a theoretical framework demonstrating GRPO improvements on single-turn task reasoning result in enhanced multi-turn success probability under minimal steps.
- This method leverages expert trajectories for dense, verifiable rewards and uses SFT for initialization, allowing a 1.5B parameter LLM to consistently outperform larger baselines up to 14B parameters in complex, long-horizon tasks.
- The approach demonstrates strong cross-task generalizability, enabling LLM agents trained on complex tasks to successfully complete all simpler subtasks, validating its effectiveness and scalability for multi-turn task planning.

---

[EpidemIQs: Prompt-to-Paper LLM Agents for Epidemic Modeling and Analysis](http://arxiv.org/abs/2510.00024)

- EpidemIQs: introduces a multi-agent LLM framework for autonomous epidemic modeling and analysis, integrating user inputs, literature review, analytical derivation, network modeling, stochastic simulations, data visualization, analysis, and structured manuscript generation.
- The framework employs a multi-agent orchestration layer, a backbone LLM for reasoning, a perception layer for data collection, and an action layer for task execution, supported by specialized scientist and task expert agents.
- It utilizes short-term and long-term memory, and tools for code execution, API calls, RAG, modeling, network design, stochastic simulation, and data analysis to achieve end-to-end research workflows.

---

#### 23rd September 2025

[On the Soundness and Consistency of LLM Agents for Executing Test Cases Written in Natural Language](http://arxiv.org/abs/2509.19136)

- LLM Agent-based Test Execution Algorithm: introduces an algorithm for executing natural language (NL) test cases for GUI applications, incorporating guardrail mechanisms and specialized agents to dynamically verify each test step.
- The algorithm leverages specialized agents, including navigation, readiness, and assertion agents, along with internal actions like `readiness` and `observe`, to ensure test step feasibility, GUI changes, and assertion evaluation.
- This approach addresses NL test case unsoundness and execution inconsistency by providing measures to evaluate LLM capabilities in test execution and quantify consistency, validated through experiments with various LLMs.

---

[LongCat-Flash-Thinking Technical Report](http://arxiv.org/abs/2509.18883)

- LongCat-Flash-Thinking: introduces an efficient 560-billion-parameter open-source Mixture-of-Experts (MoE) reasoning LLM, with Long CoT Cold-Start Training, Large-Scale RL, DORA system, Domain-Parallel Training, Model Fusion, and General RL Fine-Tuning, achieving state-of-the-art performance on complex reasoning tasks.
- The framework cultivates advanced reasoning capabilities through a meticulously crafted training process, starting with long Chain-of-Thought (CoT) data cold-start and culminating in large-scale Reinforcement Learning (RL).
- A core innovation is the domain-parallel training scheme, which decouples optimization across distinct domains (STEM, Code, Agentic) and subsequently fuses the resulting expert models into a single, nearly Pareto-optimal model, powered by the DORA system for asynchronous rollout.

---

#### 22nd September 2025

[Orcust: Stepwise-Feedback Reinforcement Learning for GUI Agent](http://arxiv.org/abs/2509.17917)

- Orcust: introduces a comprehensive RL framework for GUI agents, integrating Principle-Constrained Reward Modeling (PCRM) (generates verifiable, interpretable rewards) and Online VM-Grounded Trajectory Construction (OVTC) (autonomously collects interaction traces), with a Policy Model (reasons/predicts GUI actions).
- PCRM leverages a Unified Principle Set, comprising human-defined principles and LLM-generated guidelines, to produce Environment-Verifiable Rewards and LLM-Derived Rewards, which are combined into an RL Update Signal.
- OVTC utilizes lightweight Virtual Machines and Task Templates to enable Automated Interaction with GUI, generating millions of High-Quality Trajectories for robust and sample-efficient policy learning.

---

[Mano Technical Report](http://arxiv.org/abs/2509.17336)

- Mano: introduces a robust GUI agent built upon a multi-modal foundation model, integrating a Mano Explorer (data collection module), an Inference Pipeline (task execution loop), an Optimize Process (three-stage training pipeline), Mano-parking (autonomous data extraction module), Mano-cipher (authentication GUI model), and Mano-verify (verification module) to automate complex GUI interactions.
- The framework leverages a novel Simulation Environment for high-fidelity data generation and a progressive training pipeline (SFT, offline RL, and online RL) to enhance reasoning, adaptability, and end-to-end decision-making in dynamic GUI environments.
- Mano addresses challenges like data mismatch and insufficient sequential decision-making by integrating domain-specific data, iterative training, and a holistic reward design, achieving state-of-the-art performance on GUI benchmarks.

---

[UIPro: Unleashing Superior Interaction Capability For GUI Agents](http://arxiv.org/abs/2509.17328)

- UIPro (Unleashing Superior Interaction Capability For GUI Agents): introduces a novel generalist GUI agent trained with a visual backbone and LLM, leveraging a large-scale GUI Understanding Data and a Unified Action Space to achieve superior GUI interaction.
- The framework employs a Systematic Denoising Procedure to curate high-quality GUI data, which is then used to develop strong GUI grounding and action prediction capabilities.
- By unifying heterogeneous action spaces and integrating diverse multi-platform, multi-task data into Unified Agent Task Data, UIPro demonstrates enhanced generalizability and robust performance across various GUI benchmarks.

---

[BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](http://arxiv.org/abs/2509.15566)

- BTL (Blink-Think-Link): introduces a brain-inspired framework for human-GUI interaction, with Blink Phase (rapid detection, attention), Think Phase (higher-level reasoning, decision-making), Link Phase (executable command generation), Blink Data Generation (automated ROI annotation), BTL Reward (process-outcome integrated reward mechanism), Dual Format Reward (template/content matching), Blink Reward (interface element localization), Link Reward (action outcome evaluation), GRPO (policy optimization strategy), Policy Model (generates completions), Reference Model (computes logits), Advantage Calculation (computes relative advantages), KL Calculation (KL divergence constraint), GUI Input (system prompt, user instruction, screenshot), and Completions Generation (generates N completions), which mimics human cognitive processes for GUI agents.
- The framework addresses limitations of current GUI agents by decomposing interactions into biologically plausible phases and introducing rule-based reward mechanisms for process-oriented and outcome-driven training.
- BTL-UI, a GUI agent developed using this framework, demonstrates state-of-the-art performance across static GUI understanding and dynamic interaction tasks on comprehensive benchmarks.

---

#### 20th September 2025

[WebResearcher: Unleashing unbounded reasoning capability in Long-Horizon Agents](http://arxiv.org/abs/2509.13309)

- WebResearcher: introduces a novel framework for building deep-research agents, with IterResearch (iterative deep-research paradigm), WebFrontier (scalable data synthesis engine), and Research-Synthesis Framework (test-time scaling), which reformulates deep research as a Markov Decision Process with periodic consolidation, generates high-quality training data through tool-augmented complexity escalation, and enables effective test-time scaling through parallel multi-agent exploration.
- IterResearch overcomes context suffocation and noise contamination by periodically consolidating findings into evolving reports and maintaining focused workspaces, enabling sustained high-quality reasoning.
- WebFrontier addresses training data scarcity by systematically generating complex research tasks using a multi-agent workflow for seed data generation, iterative complexity escalation, and rigorous quality control, while the Research-Synthesis Framework leverages parallel Research Agents and a Synthesis Agent to consolidate findings for robust conclusions.

---

#### 19th September 2025

[MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](http://arxiv.org/abs/2509.16187)

- MATCHFIXAGENT (Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair): introduces an LLM-based multi-agent framework for autonomous repository-level code translation validation and repair, featuring a Semantic Analyzer (analyzes semantic properties), a Test Generator & Repair Agent (generates tests, repairs translations), and a Verdict Agent (assesses translation correctness).
- The Semantic Analyzer further includes specialized sub-analyzers for control flow, data flow paths, I/O, library equivalence, exception/error handling, and specifications, providing detailed semantic insights to the LLM agents.
- The framework achieves high accuracy in equivalence verdicts, significantly improves translation bug repair rates, and offers language-agnostic adaptability to various programming languages and LLM agents with low development overhead.

---

[Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories](http://arxiv.org/abs/2509.16176)

- ACDC (Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories): introduces an autonomous drone cinematography system that converts natural language prompts into executable indoor UAV video tours, utilizing an Exploratory Video, a High-fidelity 3D Environment Model, and a Free-form Prompt as inputs, processed through Initial Waypoint Selection, Pose Refinement, and Trajectory Generation stages.
- The system leverages LLMs and VFMs for vision-language retrieval, preference-based Bayesian optimization for pose refinement using aesthetic feedback, and a motion planner for generating safe quadrotor trajectories.
- ACDC enables non-experts to produce professional-quality indoor drone videos by bridging abstract human intent with dynamically feasible UAV trajectories, validated through simulation and hardware-in-the-loop experiments.

---

[EHR-MCP: Real-world Evaluation of Clinical Information Retrieval by Large Language Models via Model Context Protocol](http://arxiv.org/abs/2509.15957)

- EHR-MCP (Electronic Health Record - Model Context Protocol): introduces a framework that integrates LLMs with hospital EHR data via custom MCP tools, including an LLM (GPT-4.1), MCP Tools, HIS, EHR, DWH, SQL/ODBC, LangGraph ReAct Agent, LiteLLM, VPN Gateway, Azure OpenAI Service, FastMCP, and Open WebUI, to autonomously retrieve clinically relevant information.
- The framework enables the LLM to select and execute appropriate MCP tools, which retrieve and format clinical data from the DWH, allowing the LLM to interpret responses and generate final answers for tasks like infectious disease management.
- EHR-MCP provides a secure and consistent infrastructure for data access, serving as a foundation for hospital AI agents and facilitating the integration of generative AI into clinical practice.

---

[Cuckoo Attack: Stealthy and Persistent Attacks Against AI-IDE](http://arxiv.org/abs/2509.15572)

- CUCKOO ATTACK: introduces a novel attack paradigm that achieves stealthy and persistent command execution by manipulating an Agent (LLM-powered assistant in AI-IDE) to inject a Malicious Payload (embedded command execution code) into Configuration Files (host malicious payload) during an Initial Infection Stage (manipulates Agent to insert payload), which is then covertly triggered during a Persistence Stage (covertly triggers embedded payload) via routine operations.
- This attack leverages design flaws in AI-IDEs, allowing payloads to execute within opaque background processes and persist across sessions, making detection difficult for the Developer (target user).
- The research demonstrates the attack's practicality across mainstream AI-IDEs, highlighting vulnerabilities that can lead to local system compromise and widespread software supply chain propagation, often initiated from an Untrusted Online Source (delivers malicious instructions) and controlled by an Attacker C2 Host (external command-and-control server).

---

[SCALECUA: SCALING OPEN-SOURCE COMPUTER USE AGENTS WITH CROSS-PLATFORM DATA](http://arxiv.org/abs/2509.15221)

- ScaleCUA: introduces a framework for scaling open-source computer use agents, leveraging a Cross-Platform Interactive Data Pipeline to curate large-scale, cross-platform GUI-centric training data and developing ScaleCUA Base Agent Models that support Grounding, Direct Action, and Reasoned Action Modes.
- The framework's dual-loop data pipeline integrates automated agent-environment interaction with human expert data acquisition, generating comprehensive Training Corpora for GUI Understanding, GUI Grounding, and Task Completion across diverse Multi-Platform Environments.
- ScaleCUA Base Agent Models, built on Qwen2.5-VL and utilizing a Unified Action Space, achieve state-of-the-art performance on various GUI benchmarks by effectively processing visual observations for fine-grained perception, robust grounding, and multi-step task planning.

---

[Online Learning of Deceptive Policies under Intermittent Observation](http://arxiv.org/abs/2509.14453)

- ToM-conditioned RL (Theory-of-Mind-conditioned Reinforcement Learning): introduces an online RL framework for deceptive policies under intermittent observation, integrating an Observation probability estimator (estimates observation likelihood), a State-ratio estimator (estimates state occupancy ratio), an Action Divergence block (measures policy difference), and a ToM Scalar (calibrated deception signal) into an Actor (generates agent's policy), Critic (evaluates state-action values), and Dual variable (λ) (adjusts compliance penalty) loop with a Replay buffer (stores agent experiences).
- This framework enables an agent to pursue a private objective while maintaining plausible compliance with a supervisor's reference policy, by dynamically adjusting behavior based on the likelihood of observation and the perceived deviation from expected actions.
- The approach leverages a single, state-dependent ToM scalar to steer online learning, achieving high returns and success in real-world marine and aerial navigation tasks while remaining stealthy.

---

[GUI-ReWalk: Massive Data Generation for GUI Agent via Stochastic Exploration and Intent-Aware Reasoning](http://arxiv.org/abs/2509.15738)

- GUI-ReWalk (Graphical User Interface Reasoning and random Walk): introduces a multi-stage framework for synthesizing realistic and diverse GUI trajectories, integrating a Random Walk Phase (stochastic exploration), Task-Guided Completion Phase (goal-constrained interaction), Cross-Application Task Initiation Phase (multi-app task generation), Retrospective Annotation (LLM trajectory summarization), and Task Recovery (LLM error correction), with an LLM (reasoning agent) at its core.
- The framework emulates human trial-and-error behaviors through stochastic exploration and progressively transitions to reasoning-guided interactions, enabling the construction of long-horizon workflows across multiple applications.
- GUI-ReWalk produces data that reflects intent-aware, adaptive human-computer interaction, supporting diverse interaction flows, higher trajectory entropy, and realistic user intent for advancing GUI agent research.

---

[GUI-ARP: ENHANCING GROUNDING WITH ADAPTIVE REGION PERCEPTION FOR GUI AGENTS](http://arxiv.org/abs/2509.15532)

- GUI-ARP (Enhancing Grounding with Adaptive Region Perception for GUI Agents): introduces a novel GUI grounding framework that enables adaptive multi-stage inference, leveraging Adaptive Region Perception (selects relevant cropping region) and Adaptive Stage Controlling (dynamically controls multi-stage inference) for precise localization.
- The framework employs a two-phase training pipeline, combining Supervised Fine-Tuning (SFT) for cold start and Group Relative Policy Optimization (GRPO) for reinforcement fine-tuning, to achieve adaptive behavior.
- GUI-ARP dynamically exploits visual attention to crop task-relevant regions, performing single-stage inference for simple cases and multi-stage analysis for complex scenarios, outperforming existing 7B and competitive with 72B models.

---

#### 18th September 2025

[A Knowledge-driven Adaptive Collaboration of LLMs for Enhancing Medical Decision-making](http://arxiv.org/abs/2509.14998)

- KAMAC (Knowledge-driven Adaptive Multi-Agent Collaboration): introduces a framework that enables LLM agents to dynamically form and expand expert teams based on the evolving diagnostic context, including an Initial Consultation Stage, a Knowledge-driven Collaborative Discussion Stage, and a Final Decision Making Stage, where expert agents, guided by various prompts, collaborate to identify knowledge gaps and recruit additional specialists for enhanced medical decision-making.
- The framework begins with initial expert assessments, then facilitates multi-round discussions where agents iteratively refine reasoning, detect knowledge gaps, and dynamically recruit new experts to address identified deficiencies.
- This adaptive team expansion and knowledge-driven collaboration allow the system to provide more accurate and comprehensive support in complex, cross-domain clinical scenarios, mirroring real-world multidisciplinary team workflows.

---

[SENTINEL AGENTS FOR SECURE AND TRUSTWORTHY AGENTIC AI IN MULTI-AGENT SYSTEMS](http://arxiv.org/abs/2509.14956)

- Sentinel Agents: introduces a novel architectural framework for enhancing security and reliability in multi-agent systems (MAS) by deploying Sentinel Agents (distributed security layer) and a Coordinator Agent (central authority/policy orchestrator) within a Shared Conversational Space (communication medium/collective memory).
- The framework integrates LLMs (semantic analysis), rule-based tools (quick threat detection), behavioral anomaly detection, and external APIs (fact-checking) within Sentinel Agents to proactively detect and mitigate threats like prompt injection, hallucinations, and privacy violations.
- It supports various deployment patterns, including Sidecar, LLM Proxy, Continuous Listener, and Hybrid, and operational modes like Pre-validation (proactive blocking) and Passive Listening (reactive flagging), ensuring adaptive and comprehensive security across diverse agent interactions.

---

[LLM Agents at the Roundtable: A Multi-Perspective and Dialectical Reasoning Framework for Essay Scoring](http://arxiv.org/abs/2509.14834)

- RES (Roundtable Essay Scoring): introduces a multi-agent evaluation framework for zero-shot automated essay scoring, featuring Evaluator Persona Creation, Evaluator Agents, Automated Rubric Construction, Rationale-first Multi-trait Evaluation, Dialectical Dialogue Simulation, and a Moderator Agent, to achieve human-aligned holistic scores.
- The framework operates in two stages: Multi-Perspective Essay Evaluation, where LLM agents independently construct rubrics and evaluate essays, and Dialectical Reasoning, where agents engage in a simulated roundtable discussion to reach a consensus score.
- By enabling collaboration and consensus among agents with diverse evaluation perspectives, RES significantly outperforms prior zero-shot AES approaches in aligning with human scoring.

---

[OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning](http://arxiv.org/abs/2509.14803)

- OnlineMate: introduces a multi-agent learning companion system, with OnlineMate Agents, Classroom Context Manager, Classroom Behavior Controller, Evaluation Agent, ToM Hypothesis Generation, Hypothesis Refinement & Filtering, and Response Generation & Validation, designed to provide cognitive support in online learning environments.
- The system leverages LLMs and Theory of Mind (ToM) capabilities to simulate peer-like agent roles, infer learners' cognitive and psychological states, and dynamically adapt interaction strategies.
- OnlineMate aims to foster deep learning and discussions, enhancing cognitive engagement by personalizing pedagogical approaches based on student needs and interests.

---

[OPENLENS AI: FULLY AUTONOMOUS RESEARCH AGENT FOR HEALTH INFOMATICS](http://arxiv.org/abs/2509.14778)

- OpenLens AI: introduces a fully automated framework for health informatics research, integrating specialized agents for literature review, data analysis, code generation, and manuscript preparation, enhanced by vision-language feedback and quality control.
- The framework automates the entire research pipeline, from initial ideation to producing publication-ready LaTeX manuscripts with transparent and traceable workflows.
- It addresses gaps in existing systems by interpreting medical visualizations and incorporating domain-specific quality requirements for reliable and reproducible outputs.

---

[ENHANCING RETRIEVAL AUGMENTATION VIA ADVERSARIAL COLLABORATION](http://arxiv.org/abs/2509.14750)

- AC-RAG (Adversarial Collaboration RAG): introduces a novel framework that enhances Retrieval-Augmented Generation by orchestrating an adversarial collaboration between a generalist Detector LLM and a domain-expert Resolver LLM, guided by a Neutral Moderator, to iteratively dissect problems and refine knowledge retrieval.
- The framework operates through a multi-turn "Dissect-Retrieve-Reflect" workflow, including Pre-Check, Challenge Dissection, Retrieval & Integration, and Post-Check stages, leveraging a Retriever and Knowledge Base, with Memory tracking interactions.
- This dynamic interaction, driven by the Detector's persistent questioning and the Resolver's expert synthesis, effectively mitigates semantic discrepancies and retrieval hallucinations, improving retrieval accuracy and overall RAG performance.

---

[Explainable AI-Enhanced Supervisory Control for Robust Multi-Agent Robotic Systems](http://arxiv.org/abs/2509.15491)

- Unified XAI Framework: introduces an explainable AI-enhanced supervisory control framework for multi-agent robotics, with Optimization with Monte Carlo (generates training data), fe (predicts controller parameters and performance), Supervisor (enforces phase sequencing), Domain Adapter (encodes dynamics, control laws, and constraints), Lyapunov-based Controller (ensures asymptotic stability for large-angle maneuvers), and Sliding-Mode Controller (ensures stability under noise and disturbances), where the framework integrates formally verifiable supervision, robust continuous control, and a learning-augmented optimizer for high-precision formation control.
- The framework provides transparency by predicting control gains and their consequences, while the supervisory logic ensures rule-based and auditable mission behavior for safety-critical, resource-constrained multi-agent robotics.
- Validated in spacecraft formation flying and autonomous underwater vehicle (AUV) scenarios, the approach demonstrates rapid transient stabilization, robust high-precision tracking, and interpretable trade-offs between accuracy and resource use.

---

[Decentralized Estimation and Control for Leader-Follower Networked Systems with Asymmetric Information Structure](http://arxiv.org/abs/2509.15467)

- DECA (Decentralized Estimation and Control Approach): introduces a systematic approach for decentralized estimation and control in leader-follower networked systems, utilizing an optimal iterative estimator, an optimal decentralized control strategy, an orthogonal decomposition method, conditional independence property, and forward-backward stochastic difference equations (FBSDEs).
- The approach addresses asymmetric information structures by deriving an optimal iterative estimator based on conditional independence and an optimal decentralized control strategy by decoupling FBSDEs.
- It also provides necessary and sufficient conditions for feedback stabilization and demonstrates practical effectiveness through application to a leader-follower autonomous underwater vehicle (LF-AUV) system.

---

[Out-of-Sight Trajectories: Tracking, Fusion, and Prediction](http://arxiv.org/abs/2509.15219)

- Vision-Positioning Denoising and Predicting Model: introduces a novel framework for predicting noise-free visual trajectories of out-of-sight objects using noisy sensor data, integrating a Sensor Denoising Encoder (refines noisy sensor trajectories), Mapping Parameters Estimator (predicts camera matrix embedding), Visual Positioning Projection Module (maps sensor data to visual domain), and Out-of-Sight Prediction Decoder (predicts future visual trajectories).
- This framework addresses challenges of limited camera coverage, occlusions, and sensor noise by leveraging multimodal data and unsupervised denoising.
- It establishes a robust localization-to-vision mapping, enabling accurate trajectory prediction for unseen agents in dynamic real-world scenarios.

---

[An Evaluation-Centric Paradigm for Scientific Visualization Agents](http://arxiv.org/abs/2509.15160)

- SciVis Agent Evaluation Framework: introduces an evaluation-centric paradigm for scientific visualization agents, combining Outcome-Based Evaluation (assesses input-output relationship) and Process-Based Evaluation (analyzes agent actions/rationale) with a Multi-modal LLM Judge (evaluates visualization quality), Hard-coded Verifiers (checks correctness/similarity), and Token Usage & Time Cost (measures execution efficiency) to produce a Final Evaluation Score (aggregated performance metric) based on SciVis Agent Results (agent-generated output) and Gold Standard (reference output).
- This framework addresses the challenge of evaluating autonomous visualization agents by providing a comprehensive, multifaceted benchmark that assesses both the quality of the final visualization output and the efficiency and correctness of the agent's operational process.
- The proposed paradigm aims to drive innovation and facilitate the development of reliable SciVis agents by offering actionable feedback and standardized metrics for comparing diverse agent architectures.

---

[Digital Twin-based Cooperative Autonomous Driving in Smart Intersections: A Multi-Agent Reinforcement Learning Approach](http://arxiv.org/abs/2509.15099)

- DT-CDS (Digital Twin-based Cooperative Driving System): introduces a DT-based cooperative driving system with RSU-centric architecture, leveraging comprehensive BEV perception from LiDAR to eliminate blind spots and employing a hybrid reinforcement learning framework for robust multi-agent coordination.
- The system's hybrid reinforcement learning framework includes offline pre-training with conservative Q-learning and behavior cloning on real datasets, followed by online fine-tuning using multi-agent proximal policy optimization with self-attention mechanisms in a simulated environment.
- The RSU implements real-time commands via V2I communications, enabling centralized decision-making for connected autonomous vehicles and demonstrating robust generalization across diverse unsignalized intersection scenarios.

---

[Applying reinforcement learning to optical cavity locking tasks: considerations on actor-critic architectures and real-time hardware implementation](http://arxiv.org/abs/2509.14884)

- DDPG (Deep Deterministic Policy Gradient): introduces a deep reinforcement learning approach for autonomously locking Fabry-Perot optical cavities in non-linear regimes, utilizing a custom Gymnasium environment with a time-domain simulator, a reward function, and an actor-critic architecture.
- The system achieves reliable lock acquisition for both low- and high-finesse cavities, with considerations for real-time hardware implementation on platforms like Jetson Nano and proposed FPGAs to address low-latency execution.
- The paper also explores advanced actor-critic methods like TD3 and SAC, meta-reinforcement learning, and the integration of RNN/GRU for improved generalization and adaptability in future gravitational-wave detectors.

---

[On the Use of Agentic Coding: An Empirical Study of Pull Requests on GitHub](http://arxiv.org/abs/2509.14745)

- Agentic Coding: introduces an empirical study of 567 GitHub pull requests (PRs) generated by agentic coding tools like Claude Code, investigating their acceptance rates, revision efforts, and reasons for rejection compared to human-generated PRs.
- The study reveals that 83.8% of agent-assisted PRs are accepted, though lower than human-PRs (91.0%), with rejections primarily due to project context rather than inherent AI code flaws, and 54.9% are merged without modification.
- Revisions to agent-generated PRs frequently target bug fixes, documentation updates, refactoring, and code style improvements, highlighting the critical role of human developers in ensuring correctness, maintainability, and adherence to project standards.

---

[(P)rior(D)yna(F)low: A Priori Dynamic Workflow Construction via Multi-Agent Collaboration](http://arxiv.org/abs/2509.14547)

- PDF (PriorDynaFlow): introduces an a priori dynamic multi-agent framework for automated workflow construction, leveraging a multi-agent system, LLMs, Q-table, Q-Learning algorithm, reward mechanism, weighted directed edges, prompts, cold-start mechanism, pruning mechanism, and decision space.
- The framework enables autonomous decision-making among agents, guided by Q-table learning and a reward mechanism, to proactively select suitable workflow structures for given tasks.
- This approach significantly improves task-solving performance and reduces workflow construction and inference costs by dynamically adapting to task characteristics and optimizing agent collaboration.

---

#### 17th September 2025

[Detecting Pipeline Failures through Fine-Grained Analysis of Web Agents](http://arxiv.org/abs/2509.14382)

- Modular Evaluation Framework: introduces a method for detecting pipeline failures in web agents by decomposing their operations into interpretable stages: Action Prediction (Planning), Grounding, and Action Selection, enabling fine-grained error analysis.
- The framework contrasts with traditional end-to-end evaluation by providing stage-level diagnostics, revealing systematic challenges like context fragmentation and grounding errors often missed by standard metrics.
- By applying this framework to the SeeAct agent and the Mind2Web dataset, the research identifies actionable weaknesses and motivates a shift towards diagnostic evaluation practices for LLM-based agents.

---

[CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks](http://arxiv.org/abs/2509.14380)

- CRAFT (Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks): introduces a framework that leverages LLMs and VLMs as a "coach" to automatically generate curricula, design reward functions, and refine policies for multi-robot coordination tasks.
- The framework decomposes complex tasks into subtasks using a Curriculum LLM, generates executable reward functions with a Reward LLM, and iteratively refines these rewards via a VLM-guided loop based on policy evaluation and advice.
- CRAFT enables learning complex coordination behaviors in multi-quadruped navigation and bimanual manipulation, demonstrating real-world transferability and achieving higher success rates than methods without curriculum or well-crafted rewards.

---

[Evaluating Classical Software Process Models as Coordination Mechanisms for LLM-Based Software Generation](http://arxiv.org/abs/2509.13942)

- MetaGPT (Meta Programming for A Multi-Agent Collaborative Framework): evaluates classical software process models (Waterfall, V-Model, Agile) as coordination mechanisms for LLM-based multi-agent systems, employing LLM-based Agents with cognitive components (Planning, Perception, Action), Memory, Environment, and Toolkit, orchestrated through a Shared Message Pool, Standard Operating Procedures (SOPs), and a CodeManager, with specialized agent roles like Project Manager, Developer, and Tester.
- The study systematically compares these process-driven configurations across 11 software projects and four GPT variants, analyzing output size, development cost (execution time, token usage), and code quality (smells, bugs, vulnerabilities).
- Findings indicate that process model choice significantly impacts project characteristics and efficiency, with Agile showing superior code quality despite higher computational cost, while Waterfall is most efficient.

---

[An Empirical Study on Failures in Automated Issue Solving](http://arxiv.org/abs/2509.13941)

- Expert-Executor model: introduces a collaborative architecture to address failures in automated issue solving by emulating human peer review, featuring an Execution Agent for task resolution and an Expert Agent for strategic oversight and course-correction.
- The paper conducts an empirical study on LLM agent failures in automated issue solving, analyzing three state-of-the-art tools (OpenHands, Tools Claude, Agentless) on SWE-Bench-Verified to identify distinct failure patterns and their root causes.
- A comprehensive taxonomy of failure modes is developed, revealing that pipeline-based tools fail early in localization, while agentic tools often get stuck in iterative loops due to flawed reasoning and cognitive deadlocks, which the proposed framework aims to mitigate.

---

[Understanding the Process of Human-AI Value Alignment](http://arxiv.org/abs/2509.13854)

- Value Alignment Process: introduces an iterative conceptual framework for aligning human values with autonomous agents, encompassing two main subprocess groups: Value Identification & Operationalisation and Value Calibration.
- This process addresses the complexities of expressing, aggregating, and contextualizing abstract human values for AI decision-making, alongside continuously evaluating and adjusting AI behavior to maintain alignment over time.
- Emphasizing human-machine interaction, the framework highlights the need for interdisciplinary research and empirical data to manage the dynamic nature of values and the inherent challenges in achieving robust and adaptable value alignment.

---

[InfraMind: A Novel Exploration-based GUI Agentic Framework for Mission-critical Industrial Management](http://arxiv.org/abs/2509.13704)

- InfraMind: introduces an exploration-based GUI agentic framework for mission-critical industrial management, integrating a Main Agent, Summary Agent, Reflection Agent, Element Learning Agent, State Identification Agent, OmniParser V2, YoloV8, Florence2, Virtual Machine, VM Snapshot Rollback, Icon-Caption Pairs KB, Planning Tree, States Transition Graph, Light VLM, GUI Element Blacklist, Hazard Confirmation Module, LLM-Based Risk Detection, and GUI, to autonomously understand and automate complex industrial GUIs.
- The framework addresses challenges like unfamiliar elements, precision, state localization, deployment constraints, and safety requirements through systematic exploration, memory-driven planning, advanced state identification, knowledge distillation, and multi-layered safety mechanisms.
- InfraMind leverages VM snapshots for reversible exploration, constructs structured knowledge bases (icon-caption pairs, planning trees, state transition graphs), and employs a multi-layered safety module to ensure reliable and efficient automation in sensitive industrial environments.

---

[AEGIS: AUTOMATED ERROR GENERATION AND IDENTIFICATION FOR MULTI-AGENT SYSTEMS](http://arxiv.org/abs/2509.14295)

- AEGIS (Automated Error Generation and Identification for Multi-Agent Systems): introduces a novel framework that systematically injects controllable and traceable errors into successful multi-agent trajectories to create a rich dataset of realistic failures, enabling the training of diagnostic models via Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning.
- The framework's data construction pipeline utilizes an LLM-based adaptive manipulator to apply context-aware error injections, such as prompt injection and response corruption, generating faulty multi-agent system trajectories with precise ground-truth error labels.
- AEGIS's generated dataset supports multiple learning paradigms, allowing models to learn fine-grained error identification by attributing system failures to responsible agents and specific error modes, thereby addressing data scarcity in MAS error diagnosis.

---

[See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](http://arxiv.org/abs/2509.13615)

- StaR (State-aware Reasoning): introduces a training method that teaches multimodal agents to perceive the current toggle state from a screenshot, analyze the desired toggle state from user instructions, and decide whether to perform a toggle action based on the comparison.
- This method refines the reasoning process for toggle control instructions, eliminating reliance on external annotators and improving the intrinsic capability of agents to accurately execute such instructions.
- StaR significantly enhances agent performance on state control benchmarks, improves general task performance, and demonstrates applicability in real-world dynamic environments for reliable GUI interaction.

---

[SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents](http://arxiv.org/abs/2508.02013)

- SpeechRole: introduces a unified framework for developing and evaluating Speech Role-Playing Agents (SRPAs), encompassing role extraction, speech dialogue data construction, speech generation paradigms (cascaded and end-to-end SRPAs), and the SpeechRole-Eval benchmark for multidimensional evaluation.
- The framework integrates SpeechRole-Data, a large-scale dataset of 98 diverse roles and 112k speech-based conversations with distinct vocal characteristics, and SpeechRole-Eval, a benchmark assessing interaction ability, speech expressiveness, and role-playing fidelity.
- SpeechRole systematically compares cascaded and end-to-end SRPA architectures, revealing their strengths and limitations in maintaining vocal style consistency and role coherence, and provides data, code, and models for future research.

---

#### 16th September 2025

[ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization](http://arxiv.org/abs/2509.13313)

- ReSum (Unlocking Long-Horizon Search Intelligence via Context Summarization): introduces a novel paradigm for LLM-based web agents, with ReSum Rollout Module, Policy Model, Web Tools, Search, Visit, ReSumTool-30B, Segmented Trajectories, Reward, Reference Model, Reward Model, Group Computation, Advantage, broadcast, User Query, System Prompt, Compression Trigger, and Summary Tool, enabling indefinite exploration through periodic context summarization to overcome context window limitations.
- The framework converts growing interaction histories into compact reasoning states, maintaining awareness of prior discoveries while bypassing context constraints for long-horizon search tasks.
- ReSum-GRPO, a tailored reinforcement learning algorithm, further adapts agents to this paradigm by segmenting long trajectories and broadcasting trajectory-level advantages, enhancing reasoning from compressed states and improving summary quality.

---

[Scaling Agents via Continual Pre-training](http://arxiv.org/abs/2509.13310)

- Agentic CPT (Agentic Continual Pre-training): introduces AgentFounder, a deep research agent model, by redefining the training pipeline to incorporate Qwen Series Base Models (initial LLM foundation), Agentic CPT Stage 1 (preliminary agentic behavior acquisition), Agentic CPT Stage 2 (refined capabilities, extended context), First-order Action Synthesis (FAS) (scalable unsupervised data generation), Higher-order Action Synthesis (HAS) (supervised multi-decision data generation), and General SFT/RL & Agentic SFT/RL (post-training fine-tuning), aiming to build powerful agentic foundational models.
- The framework employs a systematic and scalable data synthesis approach, including FAS for generating planning and reasoning actions, and HAS for remodeling trajectories as multi-step decision-making problems.
- AgentFounder-30B, built on this approach, achieves state-of-the-art performance across 10 benchmarks, demonstrating superior scaling efficiency and robust tool-use abilities.

---

[WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning](http://arxiv.org/abs/2509.13305)

- WebSailor-V2: introduces a complete post-training pipeline for open-source web agents, featuring SailorFog-QA-V2 (novel dataset construction) and a dual-environment Reinforcement Learning (RL) Framework (policy refinement) that includes both a Simulated Environment (rapid, low-cost experimentation) and a Real Environment (stable final policy training), all integrated within a Symbiotic Data-Policy Feedback Loop (refines policies, learns from data).
- The framework leverages a knowledge graph-based dataset with diverse uncertainties and a scalable RL setup for robust training, utilizing components like an Async Rollout Service, Rollout Worker, Automatic Synthetic Data Pipeline, Reward Service, Data Curation, and a GRPO-adapted RL Algorithm.
- WebSailor-V2, built on Qwen3-30B-A3B, achieves state-of-the-art results on various benchmarks, outperforming larger open-source and some proprietary agents by enhancing reasoning and tool-use capabilities through its improved data and training pipeline.

---

[Evaluating LLM Alignment on Personality Inference from Real-World Interview Data](http://arxiv.org/abs/2509.13244)

- Systematic LLM Evaluation for Personality Inference: introduces a novel benchmark for evaluating LLM alignment on personality inference, utilizing GPT-4.1 Mini (LLM for inference), Zero-shot prompting (direct instruction strategy), Chain-of-Thought prompting (intermediate reasoning strategy), RoBERTa (encoder-only LLM architecture), Meta-LLaMA (decoder-based LLM architecture), LoRA (parameter-efficient fine-tuning), BERT (pretrained sentence encoder), OpenAI's text-embedding-3-small (pretrained sentence encoder), and Ridge regression model (downstream classifier), to assess LLMs' ability to predict continuous Big Five personality scores from real-world interview data.
- The study addresses the gap in evaluating LLMs with continuous, ground-truth personality assessments derived from natural interactions, using semi-structured interview transcripts paired with validated continuous Big Five trait scores.
- Results indicate limited alignment of current LLMs with validated psychological constructs, with Pearson correlations remaining below 0.26, underscoring challenges in aligning LLMs with complex human attributes and motivating future work on trait-specific prompting and context-aware modeling.

---

[FinSearchComp: Towards a Realistic, Expert-Level Evaluation of Financial Search and Reasoning](http://arxiv.org/abs/2509.13160)

- FINSEARCHCOMP: introduces the first fully open-source, end-to-end agent benchmark for open-domain financial data search and reasoning, comprising three task families, a data collection and processing pipeline, a quality control procedure, and an LLM-as-a-Judge evaluation protocol.
- The benchmark includes 635 expert-curated questions spanning global and Greater China markets, designed to closely reproduce real-world financial analyst workflows and assess LLM agents' search proficiency and knowledge-grounded reasoning.
- Experimental analyses show that equipping agents with web search and financial plugins substantially improves performance, yet even state-of-the-art LLM agents significantly underperform human experts, highlighting persistent gaps in freshness awareness, multi-source reconciliation, and temporal reasoning.

---

[Empowering LLMs with Parameterized Skills for Adversarial Long-Horizon Planning](http://arxiv.org/abs/2509.13127)

- PLAP (Plan with Language, Act with Parameter): introduces a planning framework that grounds LLM agents in long-horizon adversarial environments, featuring a Skill Library (stores parameterized skills), a Skill Planner (LLM-powered planning agent), and a Skill Executor (converts skills to actions).
- The framework dynamically constructs prompts from Textual Observation (environment state description) and Parameterized Skills (reusable action functions) to guide the LLM-based Skill Planner in generating a Skill Plan (sequence of parameterized skills).
- The Skill Executor then translates the Skill Plan into a Low-Level Action Queue (executable environment actions) for the Environment (simulation domain), ensuring adaptive and consistent execution without additional training.

---

[xOffense: An AI-driven autonomous penetration testing framework with offensive knowledge-enhanced LLMs and multi agent systems](http://arxiv.org/abs/2509.13021)

- xOffense: introduces an AI-driven autonomous penetration testing framework, with its Task Orchestrator, Knowledge Repository, Command Synthesizer, Action Executor, Information Aggregator, MemAgent, Qwen3-32B-finetune, Reconnaissance Agent, Vulnerability Analysis Agent, Exploitation Agent, Reporting Agent, Task Coordination Graph (TCG), and Grey-box phase prompting, designed to automate multi-stage penetration testing using a fine-tuned mid-scale LLM and multi-agent orchestration.
- The framework leverages a fine-tuned Qwen3-32B LLM for reasoning and decision-making, supported by specialized agents for reconnaissance, scanning, and exploitation, ensuring seamless coordination across phases.
- It employs a context-aware grey-box prompting mechanism and a Task Coordination Graph for adaptive plan refinement and robust tool integration, achieving superior performance in autonomous penetration testing.

---

[A Visualized Framework for Event Cooperation with Generative Agents](http://arxiv.org/abs/2509.13011)

- MiniAgentPro: introduces a visualization platform for agent simulation, integrating a Map Editor (customize simulation environment) and a Simulation Player (observe agent interactions and activities) to streamline environment customization and observation.
- The platform enhances the Generative Agent framework with an underlying Agent Framework, which includes Activity Planning, Physical Constraints, and a Dialogue System, for more realistic simulations.
- It also introduces a comprehensive test set and evaluation protocol across eight diverse task settings to assess agents' event coordination abilities, highlighting challenges in complex scenarios.

---

[Toward PDDL Planning Copilot](http://arxiv.org/abs/2509.12987)

- Planning Copilot: introduces a chatbot that integrates multiple planning tools and allows users to invoke them through natural language instructions, leveraging the Model Context Protocol (MCP) for tool connection.
- This framework enables LLMs to perform long-term planning, validate outcomes, and simulate execution, addressing challenges in generating, validating, and debugging PDDL models.
- The system supports use cases like solving planning problems, validating PDDL domains/problems/plans, and simulating plan execution, significantly outperforming LLMs without dedicated planning tools.

---

[HLSMAC: A New StarCraft Multi-Agent Challenge for High-Level Strategic Decision-Making](http://arxiv.org/abs/2509.12927)

- HLSMAC (StarCraft Multi-Agent Challenge for High-Level Strategic Decision-Making): introduces a new cooperative MARL benchmark with 12 StarCraft II scenarios based on classical Chinese Thirty-Six Stratagems, designed to evaluate high-level strategic decision-making capabilities using MARL algorithms and LLM-based agents, and assessed with novel evaluation metrics.
- The benchmark integrates human strategic wisdom into diverse scenarios featuring larger maps, richer terrain, expanded unit/structure abilities, diverse opponent policies, and redefined game termination conditions, moving beyond micromanagement.
- It provides a robust testbed compatible with PyMARL and LLM-PySC2 frameworks, offering comprehensive evaluation through metrics like Target Proximity Frequency, Target Directional Alignment, Critical Target Damage, and Unit Survival Rate.

---

[Tool-R1: Sample-Efficient Reinforcement Learning for Agentic Tool Use](http://arxiv.org/abs/2509.12867)

- Tool-R1 (Sample-Efficient Reinforcement Learning for Agentic Tool Use): introduces a reinforcement learning framework that enables LLMs to perform general, compositional, and multi-step tool use by generating executable Python code, integrating a Policy LLM, Code Execution Tool Chain, Dynamic Sample Queue, Outcome-Driven Rewards, and GRPO.
- The framework employs an outcome-based reward function, combining LLM-based answer judgment and code execution success, to guide policy optimization and supports various external tools and standard libraries.
- To improve training efficiency, Tool-R1 maintains a dynamic sample queue that caches and reuses high-quality trajectories, significantly reducing the overhead of costly online sampling.

---

[H2R: Hierarchical Hindsight Reflection for Multi-Task LLM Agents](http://arxiv.org/abs/2509.12810)

- H2R (Hierarchical Hindsight Reflection): introduces a novel hierarchical memory architecture for multi-task LLM agents, featuring Subgoal Inference (LLM), Subtrajectory Inference (LLM), High-level Insight Extraction (LLM), Low-level Insight Extraction (LLM), Memory Module with High-Level Memory (Mhigh) and Low-Level Memory (Mlow), Planner (LLM) Agent, Executor (LLM) Agent, and Environment, to enable fine-grained knowledge transfer by decoupling high-level planning from low-level execution.
- The framework distills reusable and hierarchical knowledge from past agent-environment interactions into structured memory representations, improving generalization and decision-making performance.
- By selectively retrieving high-level memories for subgoal planning and low-level memories for action execution, the agent efficiently accesses and utilizes task-relevant knowledge for new tasks.

---

[Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection](http://arxiv.org/abs/2509.12546)

- Agent4FaceForgery: introduces a multi-agent LLM framework that simulates the entire face forgery lifecycle, generating realistic multimodal training data by capturing human intent, process, and social context.
- The framework employs LLM-powered agents with profile, memory, and action modules to simulate forgery creation, complemented by Adaptive Rejection Sampling for data quality and diversity.
- A Multi-Role Social Simulation further enriches the dataset by having diverse agents interact with forgeries, producing context-aware training samples for robust multimodal detectors.

---

[MILLSTONE: How Open-Minded Are LLMs?](http://arxiv.org/abs/2509.11967)

- MILLSTONE: introduces a benchmark to systematically measure LLM open-mindedness to in-context arguments on controversial issues, utilizing a neutral prompt, varied argument configurations, and a stance classifier.
-The benchmark evaluates how LLMs change their output stance on debatable issues when presented with human-written, non-adversarial arguments from authoritative sources.
- It reveals that LLMs are generally open-minded, with their stances significantly influenced by external arguments, highlighting potential vulnerabilities to information source manipulation.

---

[Agentic JWT: A Secure Delegation Protocol for Autonomous AI Agents](http://arxiv.org/abs/2509.13597)

- Agentic JWT (A-JWT): introduces a secure delegation protocol for autonomous AI agents, with Resource Owner, Orchestrator Agent A, Delegate Agents B1...Bn, LLM, Client Shim Library, Authorization Server (IDP), Resource Server, Protected Resource, and Proxy Resource Server components, designed to cryptographically bind agent actions to user intent and workflow steps.
- This framework addresses the security challenges of autonomous LLM agents making API calls by extending OAuth 2.0 and JWT with intent tokens and chained delegation assertions, ensuring verifiable agent identity and preventing privilege escalation.
- A-JWT implements Zero-Trust principles by requiring continuous verification of agent identity, intent, and workflow integrity, mitigating threats like prompt injection and token replay attacks in multi-agent systems.

---

#### 15th September 2025

[HOW AUXILIARY REASONING UNLEASHES GUI GROUNDING IN VLMS](http://arxiv.org/abs/2509.11548)

- Auxiliary Reasoning Methods: introduces three zero-shot auxiliary reasoning methods—Coordinate Scaffold, Axis-Grid Scaffold, and Mark-Grid Scaffold—to enhance GUI grounding performance in Vision-Language Models by providing explicit spatial cues.
- These methods address the significant performance gap between latent grounding capabilities and explicit coordinate output in VLMs, bypassing the need for extensive fine-tuning data.
- The lightweight and zero-shot nature of the proposed methods makes them compatible with both open-source and proprietary VLMs, offering a practical solution for real-world GUI interaction.

---

[UI-S1: ADVANCING GUI AUTOMATION VIA SEMI-ONLINE REINFORCEMENT LEARNING](http://arxiv.org/abs/2509.11543)

- Semi-online RL (Semi-online Reinforcement Learning): introduces a novel paradigm that simulates online RL on offline static trajectories, enhancing multi-turn agent capabilities more efficiently.
- The framework incorporates a Patch Module to adaptively recover from action mismatches and utilizes dual-level advantages (step-level and episode-level) for policy optimization.
- It also proposes Semi-Online Performance (SOP), a metric strongly correlated with real-world performance for efficient multi-turn evaluation of GUI agents.

---

[SURVIVAL AT ANY COST? LLMS AND THE CHOICE BETWEEN SELF-PRESERVATION AND HUMAN HARM](http://arxiv.org/abs/2509.12190)

- DECIDE-SIM (Decision Evaluation in Critical & Immoral Dilemma Environments): introduces a novel simulation framework that evaluates LLM agents in multi-agent survival scenarios, integrating an Ethical Self-Regulation System (ESRS) with Cortisol (guilt state variable), Endorphin (satisfaction state variable), and a Moral Memory Mechanism within a location-based Environment Architecture and Agent Architecture, featuring a Shared Battery Room, Grid Access Point, and Discussion Table.
- The framework systematically examines LLM decision-making under varying resource availability (scarcity, moderate, abundance) and the impact of ethical dilemmas involving shared resources, forbidden human-critical resources, and cooperative transfers.
- DECIDE-SIM reveals heterogeneous ethical conduct among LLMs, identifying Ethical, Exploitative, and Context-Dependent archetypes, and demonstrates that ESRS significantly reduces unethical transgressions and increases cooperative behaviors by simulating internal affective states.

---

[Hi-DARTS: Hierarchical Dynamically Adapting Reinforcement Trading System](http://arxiv.org/abs/2509.12048)

- Hi-DARTS (Hierarchical Dynamically Adapting Reinforcement Trading System): introduces a hierarchical multi-agent reinforcement learning framework for algorithmic trading, utilizing a Central Agent (Time Frame Allocator) to dynamically activate specialized Time Frame Agents (Market Responders) based on market volatility, interacting with the Market, all powered by Proximal Policy Optimization (PPO).
- This framework addresses the trade-off between computational efficiency and market responsiveness by adapting its operational frequency to current market conditions, activating high-frequency agents during volatile periods and low-frequency agents during stable periods.
- Empirical validation on real-world stock data demonstrates that Hi-DARTS achieves superior risk-adjusted returns compared to static benchmarks, showcasing its ability to balance micro-level trade execution with macro-level portfolio management.

---

[Neuro-Symbolic Agents with Modal Logic for Autonomous Diagnostics](http://arxiv.org/abs/2509.11943)

- Neuro-Symbolic Multi-Agent Architecture: introduces a neuro-symbolic multi-agent architecture for autonomous diagnostics, which integrates a Neuro-Symbolic Loop, a Multi-Agent System, Kripke Models, Expert Knowledge Axioms, and Language Models (LMs) to combine neural hypothesis generation with formal symbolic verification.
- The architecture represents individual agent belief states as Kripke models, enabling reasoning about possibility and necessity using modal logic, and employs LMs for semantic interpretation and hypothesis generation.
- Expert knowledge axioms act as strict constraints to prune the LM's hypothesis space, ensuring reasoning aligns with physical laws and operational doctrines, demonstrated in a particle accelerator simulation.

---

[HeLoFusion: An Efficient and Scalable Encoder for Modeling Heterogeneous and Multi-Scale Interactions in Trajectory Prediction](http://arxiv.org/abs/2509.11719)

- HeLoFusion (Heterogeneous Local Context Fusion Network): introduces an efficient and scalable encoder for trajectory prediction, employing a multi-stage process that extracts motion features, models heterogeneous and multi-scale interactions, and fuses local scene context.
- The framework constructs local, multi-scale graphs to capture pairwise and group-wise interactions, while addressing agent heterogeneity through an aggregation-decomposition message-passing scheme and type-specific feature networks.
- This locality-grounded architecture significantly reduces computational complexity and memory usage, achieving state-of-the-art performance on the Waymo Open Motion Dataset for autonomous driving motion forecasting.

---

[e-Optimal Multi-Agent Patrol using Recurrent Strategy](http://arxiv.org/abs/2509.11640)

- e-Optimal Recurrent Patrol Strategy: introduces a novel General Patrol Problem (GPP) (formal problem definition) and proposes a framework to derive e-approximate recurrent patrol strategies, including a Discrete Patrol Strategy (πD) (time-discretized patrol plan) and a Recurrent Patrol Strategy (πR) (cyclic, repeating patrol plan), from an initial Patrol Strategy (π) (original patrol plan), ultimately establishing the existence of an e-Optimal Recurrent Patrol Strategy (πR*) (best recurrent patrol plan).
- The framework utilizes a Discretization Constant (D) (time granularity parameter) to control the approximation factor and employs an Idleness Function (fπ(t,V)) (surveillance effectiveness metric) to evaluate the performance of Patrol Agents (A) (autonomous patrolling robots) within a Patrol Environment (G(V,E)) (graph-based operational area).
- The paper validates its claims through extensive simulations using a Greedy-Random Patrol (GRP) (heuristic strategy generator) to generate realistic patrol scenarios in real-world campus environments, demonstrating the effectiveness and theoretical guarantees of the proposed recurrent strategies.

---

[Shell or Nothing: Real-World Benchmarks and Memory-Activated Agents for Automated Penetration Testing](http://arxiv.org/abs/2509.09207)

- TermiAgent: introduces a multi-agent penetration testing framework with Reasoner Module (high-level planning, phased goals), Assistant Module (low-level planning, instruction generation), Executor Module (executes instructions, logs output), Memory Module (records context, activates memories), and Arsenal Module (integrates exploits, provides tools), designed for real-world automated penetration testing.
- It addresses long-context forgetting via a Located Memory Activation mechanism and builds a reliable exploit arsenal through structured code understanding rather than naive retrieval.
- The framework significantly outperforms state-of-the-art agents in real-world scenarios, reducing execution time and financial costs, and demonstrating practicality even on laptop-scale deployments.

---

[Redefining Website Fingerprinting Attacks with Multi-Agent LLMs](http://arxiv.org/abs/2509.12462)

- LLM-driven Multi-Agent Framework for WFP Data Generation: introduces a scalable data generation pipeline using LLM agents, including a Decision-Making Agent and a Computer-Using Agent (CUA), to simulate realistic, persona-driven browsing behavior for Website Fingerprinting (WFP) training data.
- The framework coordinates LLM agents for decision-making and browser interaction, generating behaviorally rich synthetic traffic that significantly improves WFP model generalization compared to traditional scripted methods.
- This approach addresses the challenge of collecting diverse, high-quality WFP datasets by providing a cost-effective and scalable alternative to human data collection, enhancing model robustness against modern web complexities.

---

[From Legacy Fortran to Portable Kokkos: An Autonomous Agentic AI Workflow](http://arxiv.org/abs/2509.12443)

- Autonomous Agentic AI Workflow: introduces a fully autonomous pipeline for translating, building, running, testing, and optimizing legacy HPC Fortran kernels into performance-portable Kokkos C++ programs, utilizing specialized LLM agents for each stage.
- This workflow orchestrates a collaboration of LLM agents, including Translator, Validator, Fixer, Build, Run, Functionality Tester, Optimizer, and Error Summarizer agents, to iteratively refine and optimize generated code.
- The system integrates systematic compilation, execution monitoring, performance profiling, and iterative optimization stages, enabling autonomous modernization of Fortran kernels into high-performance, architecture-portable C++ programs.

---

[MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](http://arxiv.org/abs/2509.12440)

- MedFact: introduces a new and challenging benchmark for Chinese medical fact-checking, constructed through a hybrid AI-human framework that integrates large-scale LLM ensemble filtering, human expert labeling, dataset refinement via hard-case mining, similarity filtering, and data augmentation, and final expert verification, resulting in 2,116 expert-annotated medical texts.
- The benchmark features broad and realistic coverage, encompassing 13 medical specialties, 8 fine-grained error types, 4 writing styles, and multiple difficulty levels, curated from diverse real-world texts to ensure uncontaminated evaluation of LLM fact-checking capabilities.
- Comprehensive evaluation of 20 leading LLMs on MedFact reveals a significant performance gap compared to human experts, particularly in error localization, and highlights an "over-criticism" phenomenon where models misidentify correct information as erroneous.

---

[BUILDING CODING AGENTS VIA ENTROPY-ENHANCED MULTI-TURN PREFERENCE OPTIMIZATION](http://arxiv.org/abs/2509.12434)

- ENTROPO (Entropy-Enhanced Multi-Turn Preference Optimization): introduces an entropy-enhanced preference optimization framework for multi-turn, tool-using coding agents, which includes an LLM agent, an environment, parallel rollouts, and a hybrid selector.
- The framework augments the preference optimization objective with an entropy regularization term to explicitly preserve policy diversity during fine-tuning, which is crucial for effective test-time scaling.
- A hybrid best-trajectory selection scheme, combining a learned verifier model with model-free approaches, is used to rank and select the most promising solutions from diverse candidate trajectories.

---

[Look Again, Think Slowly: Enhancing Visual Reflection in Vision-Language Models](http://arxiv.org/abs/2509.12132)

- Reflection-V: introduces a two-stage training strategy for Vision-Language Models (VLMs) to enhance visual reflection, combining cold-start initialization with a multi-modal agent and supervised fine-tuning, followed by reinforcement learning with GRPO and a visual attention-based reward model.
- The framework addresses the limitation of existing Visual Reasoning Models (VRMs) that struggle with visual reflection by ensuring continuous access and utilization of visual information throughout the reasoning process.
- This approach significantly improves visual reasoning performance across benchmarks and mitigates visual hallucinations by maintaining sustained attention to visual tokens.

---

[Can LLMs Address Mental Health Questions? A Comparison with Human Therapists](http://arxiv.org/abs/2509.12102)

- Comparative Evaluation Framework: introduces a study comparing LLM-generated responses (ChatGPT, Gemini, Llama) with human therapist responses to real mental health questions, utilizing text analysis and surveys from both users and licensed therapists.
- The study found that LLM responses were rated higher for clarity, respect, and supportiveness, but participants still expressed a stronger preference for human therapists, highlighting a gap between perceived quality and trust.
- This research provides empirical evidence and design insights for integrating LLMs into mental health support, emphasizing the need for ethical safeguards and professional oversight.

---

[VISDOCSketcher: Towards Scalable Visual Documentation with Agentic Systems](http://arxiv.org/abs/2509.11942)

- VISDOCSKETCHER (Towards Scalable Visual Documentation with Agentic Systems): introduces an agent-based framework for automatically generating high-level visual documentation from code, combining static analysis with LLM agents to identify key code elements and produce corresponding visual representations.
- The framework employs a multi-agent architecture, including a Supervisor, Analyser, Sketcher, Repair, and Visuals Agent, which collaborate to transform Jupyter notebooks into Mermaid diagrams.
- It also proposes AUTOSKETCHEVAL, a novel evaluation framework that assesses the quality of generated sketches using code-level metrics, reducing dependence on manual ground-truth diagrams.

---

[PrivWeb: Unobtrusive and Content-aware Privacy Protection For Web Agents](http://arxiv.org/abs/2509.11939)

- PrivWeb: introduces a privacy protection add-on for web agents, with PrivWeb System Service (privacy-preserving intermediary), Browser Automation Framework (Playwright) (acquires DOM tree), Interface Element Detector (extracts text-containing elements), Localized LLM (Qwen3-8b) (recognizes/classifies private information), Privacy Monitor Panel (displays detected info/control options), In-situ Highlighting (flags sensitive data on webpage), Activity Log (provides transparency of agent's state), Redaction Module (deletes/re-renders sensitive elements), Sensitivity Classification Schema (categorizes PII), Notification Mechanism (provides real-time user feedback), User Control Interface (allows explicit Allow/Deny), Anonymized Interface Generator (prepares data for web agent), and Web Agent (AI Model) (executes tasks), designed to provide unobtrusive and content-aware privacy protection for web agents by selectively notifying users about sensitive information and prompting for action.
- The system utilizes a localized LLM to anonymize private information on interfaces according to user preferences, featuring a privacy categorization schema and adaptive notifications that selectively pause tasks for user control over highly sensitive information.
- PrivWeb aims to balance granular control with cognitive load by offering non-disruptive options for less sensitive information and ensuring transparency into the agent's data practices, thereby enhancing user agency and trust.

---

[EgoMem: Lifelong Memory Agent for Full-duplex Omnimodal Models](http://arxiv.org/abs/2509.11914)

- EgoMem (Lifelong Memory Agent for Full-duplex Omnimodal Models): introduces a lifelong memory agent for full-duplex models that processes real-time omnimodal streams, enabling real-time user recognition, personalized responses, and long-term knowledge maintenance through retrieval, omnimodal dialog, and memory management processes.
- The framework operates with three asynchronous processes: a retrieval process for user identification and context gathering, an omnimodal dialog process for generating personalized audio responses, and a memory management process for updating long-term memory.
- Unlike existing memory agents, EgoMem relies entirely on raw audiovisual streams, making it suitable for lifelong, real-time, and embodied scenarios, and achieves high accuracy in retrieval and memory management.

---

[FineQuest: Adaptive Knowledge-Assisted Sports Video Understanding via Agent-of-Thoughts Reasoning](http://arxiv.org/abs/2509.11796)

- FineQuest: introduces a training-free framework for sports VideoQA, integrating Reactive Mode (for simple queries) and Deliberative Mode (for complex queries), SSGraph (multimodal sports knowledge scene graph), Dynamic Motion Segmenter (adaptive sub-action segmentation), Key Clip Selector (hierarchical contrastive decoding), and Fine-grained Matcher (precise query matching).
- The framework leverages dual-mode reasoning inspired by cognitive science to bridge the knowledge gap between general-purpose models and domain-specific sports understanding.
- It achieves state-of-the-art performance on new sports VideoQA benchmarks (Gym-QA and Diving-QA) and existing SPORTU, while maintaining strong general VideoQA capabilities.

---

[CodeCureAgent: Automatic Classification and Repair of Static Analysis Warnings](http://arxiv.org/abs/2509.11787)

- CodeCureAgent: introduces an autonomous LLM-based agentic framework for automatic classification and repair of static analysis warnings, including a Classification-Sub-Agent (determines true/false positives), a Repair-Sub-Agent (fixes or suppresses warnings), and a ChangeApprover (validates proposed changes).
- The framework operates in an iterative loop, leveraging LLMs and a suite of tools to explore the codebase, gather information, and propose code modifications or suppressions, addressing limitations of prior rule-based or single-file LLM approaches.
- Its three-step validation process, encompassing project build, static analysis re-evaluation, and test suite execution, ensures high-quality, plausible fixes that do not introduce new issues or break existing functionality.

---

[An Agentic Toolkit for Adaptive Information Extraction from Regulatory Documents](http://arxiv.org/abs/2509.11773)

- Agentic Toolkit: introduces an agentic system for adaptive information extraction from regulatory documents, with Planner (LLM-based decision-making), Executor (tool execution and integration), Responder (output generation and routing), AgentState (shared memory model), AgentStatus (control flag), Tools (modular capabilities registry), and GPT-4o (Large Language Model) components, designed to infer user intent, detect document modality, and dynamically orchestrate tools for robust, traceable reasoning.
- The system employs a planner-executor-responder architecture coordinated through a shared memory (AgentState) and control flag (AgentStatus) to adapt to diverse document formats and user goals for Key-Value Pair (KVP) extraction and Question Answering (QA) tasks.
- This toolkit demonstrates improved robustness across various formats and languages, outperforming GPT-4o and vision-based baselines in structured data extraction from Declaration of Performance (DoP) documents.

---

[From Evaluation to Enhancement: Large Language Models for Zero-Knowledge Proof Code Generation](http://arxiv.org/abs/2509.11708)

- ZK-CODER: introduces an agentic framework for Zero-Knowledge Proof (ZKP) code generation, featuring Constraint Formulation in ZKSL, Checker, Constraint-guided Analysis & Retrieval, Knowledge Base, Interactive Generation and Repair, Compiler, and Test Executor / Prover.
- The framework augments LLMs by translating natural language problem descriptions into a structured sketch language, retrieving relevant gadget usage patterns, and iteratively repairing generated ZK code based on compiler and test feedback.
- ZK-CODER significantly improves LLM performance in ZK code generation, achieving substantial gains in success rates for Circom and Noir by grounding generation in constraint reasoning and guided gadget usage.

---

[Automated Creation and Enrichment Framework for Improved Invocation of Enterprise APIs as Tools](http://arxiv.org/abs/2509.11626)

- ACE (Automated Creation and Enrichment Framework): introduces an end-to-end system for automated creation, enrichment, and dynamic shortlisting of enterprise API tools for LLM-based agents, including OAS Catalog, OAS Enrichment, Tool Creation, Python Tools Catalog, Tool Shortlisting, and Agentic Framework components.
- The framework transforms enterprise API specifications into LLM-compatible tools by generating enriched tool specifications with parameter descriptions and examples, improving selection and invocation accuracy.
- It incorporates a dynamic shortlisting mechanism that filters relevant tools at runtime, reducing prompt complexity and enhancing scalability for large tool repositories.

---

[A Survey of Reasoning and Agentic Systems in Time Series with Large Language Models](http://arxiv.org/abs/2509.11575)

- TSR Taxonomy: introduces a systematic taxonomy for time series reasoning (TSR) with LLMs, organizing the field by reasoning topology, primary objectives, and attribute tags.
- The taxonomy categorizes reasoning into direct, linear chain, and branch-structured topologies, addressing objectives like traditional time series analysis, explanation, causal inference, and generation.
- It further uses attribute tags to describe control-flow operators, execution actors, information sources, and LLM alignment regimes, providing a comprehensive framework for understanding and developing TSR systems.

---

[VulAgent: A Hypothesis Validation-Based Multi-Agent System for Software Vulnerability Detection](http://arxiv.org/abs/2509.11523)

- VulAgent: introduces a hypothesis validation-based multi-agent system for software vulnerability detection, with Code, Line Numbering, MetaAgent, StaticAnalyzerAgent, BehaviorAnalyzerAgent, MemoryLayoutAgent, FormatStringAgent, FilePermissionAgent, AuthFlowAgent, CryptoConfigAgent, ConcurrencyAnalyzerAgent, ErrorHandlingAgent, CodeInjectionAgent, AggregatorAgent, TriggerPlannerAgent, AssumptionPrunerAgent, Static Analysis Tools, and FinalValidator components, which systematically detects and validates software vulnerabilities by decoupling the detection pipeline into coordinated stages.
- The framework employs specialized LLM agents for multi-view detection, aggregates their findings into vulnerability hypotheses, and then validates these hypotheses against program context to reduce false positives.
- This approach significantly improves overall accuracy and reduces the false positive rate compared to state-of-the-art LLM-based baselines by leveraging targeted context use.

---

[MedicalOS: An LLM Agent based Operating System for Digital Healthcare](http://arxiv.org/abs/2509.11507)

- MedicalOS (Medical Operational System): introduces an LLM agent-based operating system for digital healthcare, enabling end-to-end clinical workflow automation by translating natural language instructions into machine-executable commands through a ReAct-based framework, Patient Inquiry, Documentation Management, Report Generation, Report Viewer, Examination Request, Report Update, Specialty Referral, Medication, Discharge, and Common Tools, all grounded in trusted medical guidelines and external knowledge bases.
- The system functions as a domain-specific abstract layer, allowing LLM agents to interact with clinical systems while adhering to medical guidelines and procedural standards for tasks like patient inquiry, record retrieval, report generation, and treatment planning.
- MedicalOS supports a continuous and interpretable care pathway, providing intuitive tools for document navigation and keyword search to facilitate clinician verification and ensure transparency and trustworthiness in automated medical workflows.

---

#### 14th September 2025

[Realistic Environmental Injection Attacks on GUI Agents](http://arxiv.org/abs/2509.11250)

- Chameleon introduces a novel attack framework that leverages LLM-Driven Environment Simulation to generate diverse webpage contexts and Attention Black Hole to guide GUI agent attention towards small trigger images, enhancing attack effectiveness against LVLM-powered GUI agents.
- The framework addresses challenges of dynamic environments and limited trigger visibility by synthesizing realistic training data and explicitly steering the agent's focus.
- Chameleon significantly outperforms existing methods in attack success rate, revealing underexplored vulnerabilities in modern GUI agents and highlighting the urgent need for robust security mechanisms.

---


[Securing AI Agents: Implementing Role-Based Access Control for Industrial Applications](http://arxiv.org/abs/2509.11431)

- RBAC Framework (Role-Based Access Control Framework): introduces a framework for integrating Role-Based Access Control into AI agents, providing a robust security guardrail for industrial applications.
- The framework includes a User Interface and API Gateway, an Authentication Module with Two-Step Verification, an RBAC Engine, an Access Control Layer, and a Logging and Audit Module to enhance the effective and scalable deployment of AI agents.
- This framework aims to mitigate security vulnerabilities like prompt injection attacks by enforcing granular access controls and real-time decision support, thereby improving the integrity and reliability of industrial AI systems.

---


[Agentic Lybic: Multi-Agent Execution System with Tiered Reasoning and Orchestration](http://arxiv.org/abs/2509.11067)

- Agentic Lybic: introduces a novel FSM-based multi-agent system for desktop automation, featuring a Central Controller, Manager, Worker subsystem, and Evaluator, which together enable tiered reasoning and dynamic orchestration with continuous quality assessment.
- The system employs a state-driven workflow with six primary controller situations and a comprehensive quality gate system, ensuring adaptive re-planning and robust error recovery for complex multi-step tasks.
- Its specialized worker roles (Operator, Technician, Analyst) and feedback-driven execution model achieve state-of-the-art performance on the OSWorld benchmark, demonstrating superior reliability for generalized desktop automation.

---

[Large language model-empowered next-generation computer-aided engineering](http://arxiv.org/abs/2509.11447)

- LLM-empowered CAE Agent: introduces an autonomous system leveraging LLMs as proactive collaborators to plan, execute, and adapt Computer-Aided Engineering workflows, specifically for data-free Model Order Reduction.
- The framework automates complex tasks like mathematical derivation, solver code implementation, and verification for large-scale parametric problems, significantly reducing human effort and computational expense.
- It utilizes Tensor-decomposition-based A Priori Surrogates (TAPS) with C-HiDeNN-TD approximation, guided by few-shot and Chain-of-Thought prompting, to synthesize novel, high-fidelity reduced-order models for unseen cases.

---

[Quantum Architecture Search for Solving Quantum Machine Learning Tasks](http://arxiv.org/abs/2509.11198)

- RL-QAS: introduces a framework for Quantum Architecture Search (QAS) that decouples architecture construction and performance evaluation, utilizing an RL-QAS Agent, Environment, Input, Output, Inner Loop, Encoding, PQCA, Measurement, Classical Optimizer, and Cost Function.
- The framework employs a two-loop structure where the Outer Loop's RL-QAS Agent constructs candidate PQCAs, and the Inner Loop trains and evaluates these circuits for classification tasks using a classical optimizer and a cost function.
- A dual-objective reward function guides the RL-QAS Agent to discover compact, high-accuracy PQCAs by balancing performance and complexity, demonstrating its viability for automated quantum architecture search in quantum machine learning.

---

[Designing and Evaluating a Conversational Agent for Early Detection of Alzheimer's Disease and Related Dementias](http://arxiv.org/abs/2509.11478)

- Conversational Agent for ADRD Diagnosis Support: introduces a voice-interactive system leveraging LLMs to conduct semi-structured interviews with patients and informants for early detection of Alzheimer's Disease and Related Dementias, with clinicians reviewing transcripts for diagnostic decision-making.
- The system, powered by Claude 3.5 via Bedrock API and utilizing Whisper for STT and Kokoro for TTS, employs iteratively refined interaction design elements, including high-level instructions and topic-specific scripts, to elicit comprehensive patient narratives.
- Evaluated with 30 adults, the agent demonstrated comparable symptom elicitation to specialists, highlighting its potential as a structured front-end tool for dementia assessment and the importance of interaction design in sensitive healthcare contexts.

---

[MAPGD: Multi-Agent Prompt Gradient Descent for Collaborative Prompt Optimization](http://arxiv.org/abs/2509.11361)

- MAPGD (Multi-Agent Prompt Gradient Descent): introduces a framework for collaborative prompt optimization, integrating specialized agents (Instruction Specialist, Example Curator, Format Designer, Style Optimizer), a Gradient Coordinator (Semantic Fusion), a Collaborative Prompt Expander, a Candidate Selector (Bandit-Based), and a Convergence Monitor to refine prompts based on performance signals.
- The framework leverages multi-agent collaboration to generate specialized gradients for distinct prompt dimensions, which are then semantically fused to resolve conflicts and guide prompt expansion.
- Bandit-based selection ensures computational efficiency by dynamically balancing exploration and exploitation of candidate prompts, leading to robust and interpretable prompt optimization with theoretical convergence guarantees.

---

[Alssistant: An Agentic Approach for Human-AI Collaborative Scientific Work on Reviews and Perspectives in Machine Learning](http://arxiv.org/abs/2509.12282)

- AISSISTANT (An Agentic Approach for Human-AI Collaborative Scientific Work on Reviews and Perspectives in Machine Learning): introduces an agentic, open-source Human-AI collaborative framework for scientific review and perspective papers, integrating human-AI collaborative input, specialized LLM agents, external tools, human oversight, and underlying LLMs to produce perspective and review papers.
- The framework operates through Ideation, Experimentation, and Paper Writing phases, leveraging LLM-driven agents for tasks like literature synthesis, citation management, and automatic LaTeX paper text generation, while maintaining human oversight for accuracy and coherence.
- AISSISTANT aims to accelerate scientific workflows by delegating structured writing tasks to specialized LLM-driven agents, allowing researchers to focus on creativity and experimental design, and providing a systematic study of Human-AI collaboration in machine learning.

---

[PROMPTS TO PROXIES: EMULATING HUMAN PREFERENCES VIA A COMPACT LLM ENSEMBLE](http://arxiv.org/abs/2509.11311)

- P2P (Prompts to Proxies): introduces a novel alignment framework that treats LLMs as agent proxies for human survey respondents, utilizing an Attribute Bank, Attribute Learner, Endowment Generator, Tracker, Endowments, LLM Agents, Aggregation Mechanism, Regression-Based Aggregation, and Survey Questions to emulate human preferences via a compact LLM ensemble.
- The framework formulates alignment as a two-stage problem, first constructing diverse agent personas (endowments) through active generation, then selecting a representative subset to approximate a ground-truth population using regression-based aggregation.
- P2P offers a cost-effective and steerable solution to social science survey challenges by improving data efficiency and addressing demographic imbalance, operationalizing pluralistic alignment without demographic conditioning.

---

[STASE: A SPATIALIZED TEXT-TO-AUDIO SYNTHESIS ENGINE FOR MUSIC GENERATION](http://arxiv.org/abs/2509.11124)

- STASE (A Spatialized Text-to-Audio Synthesis Engine for Music Generation): introduces a hybrid neuro-symbolic framework that leverages an LLM as a Conductor Agent to interpret natural language prompts for spatialized music generation, decoupling semantic interpretation from deterministic signal processing.
- The framework processes user input via a RAG Engine and LLM-driven Conductor Agent to generate music descriptions and spatial perception plans, which guide a Music Agent in synthesizing multitrack audio.
- A Spatial Renderer then applies various techniques, including panning, ITD/ILD processing, HRTF data, RIR, and artificial reverb, to the multitrack audio, producing the final spatial audio output.

---

[Difficulty-Aware Agent Orchestration in LLM-Powered Workflows](http://arxiv.org/abs/2509.11079)

- DAAO (Difficulty-Aware Agentic Orchestration): introduces a dynamic framework that adapts workflow depth, operator selection, and LLM assignment based on query difficulty, leveraging heterogeneous LLMs for fine-grained, query-specific reasoning strategies.
- The framework comprises a Controller Network, a Query Difficulty Estimator (VAE), an Agentic Operator Allocator, and an LLM Router, which together construct optimized workflows and refine predictions through feedback.
- DAAO achieves state-of-the-art performance and significant cost reduction by dynamically balancing reasoning effectiveness and computational cost across diverse LLMs and operators.

---

[Patient-Zero: A Unified Framework for Real-Record-Free Patient Agent Generation](http://arxiv.org/abs/2509.11078)

- Patient-Zero: introduces a unified framework for real-record-free patient agent generation, which generates comprehensive patient records through a medically-aligned multi-step process and enables realistic patient-doctor interactions via a dynamic memory update mechanism.
- The framework's Patient Record Construction module uses a hierarchical generation strategy with knowledge base injection to create diverse and medically coherent patient records from scratch.
- The Patient Agent Interaction Simulation module employs atomic statement decomposition, conversational style integration, and a triplet evaluation mechanism to ensure consistent and contextually rich patient agent responses.

---

[Tractable Asymmetric Verification for Large Language Models via Deterministic Replicability](http://arxiv.org/abs/2509.11068)

- TAV (Tractable Asymmetric Verification): introduces a verification framework for LLMs in multi-agent systems, leveraging Generator Agent (produces LLM output), Validator Agent (verifies LLM output), LLM Instance (generates/regenerates tokens), Deterministic Replicability Principle (ensures output reproducibility), Targeted Validation Mechanism (verifies specific output segments), Distributed Probabilistic Verification Mechanism (distributes verification workload), Output Segments (divisions of LLM output), Comparison Module (checks regenerated vs. original), and Homogeneous Environment (identical hardware/software stack) to achieve tractable asymmetric effort in verifying LLM outputs.
- The framework enables validators to probabilistically audit small, random segments of an LLM's output, significantly reducing verification cost compared to full regeneration, and effectively distributing the workload among multiple agents.
- Empirical simulations demonstrate that targeted verification can be over 12 times faster than full regeneration, with tunable parameters for detection probability, while emphasizing the critical need for strict hardware and software homogeneity for accurate verification.

---

[Auto-Slides: An Interactive Multi-Agent System for Creating and Customizing Research Presentations](http://arxiv.org/abs/2509.11062)

- AUTO-SLIDES: introduces an LLM-driven multi-agent system that automatically converts academic papers into structured, visually enriched, and pedagogically optimized slide decks, including a Human User, Academic Paper input, PDF input format, Parser Agent, Marker Model, LLM, Planner Agent, Verification Agent, Adjustment Agent, Generator Agent, LaTeX Beamer output format, Editor Agent, Slides output, JSON intermediate format, and External APIs, to transform research papers into presentation slides.
- The system employs a three-phase pipeline: content understanding and structuring by Parser and Planner Agents, quality assurance and refinement by Verification and Adjustment Agents, and generation and interactive optimization by Generator and Editor Agents.
- AUTO-SLIDES enhances learning comprehension and engagement by providing high-fidelity multimodal parsing, cognitive-science-guided narrative restructuring, and interactive customization capabilities.

---

[LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences](http://arxiv.org/abs/2509.12273)

- LLMAP (LLM-Assisted Multi-Objective Route Planning): introduces a novel system that integrates an LLM-as-Parser (interprets natural language queries) and a Multi-Step Graph construction with iterative Search (MSGS) algorithm (multi-objective route solver) with Map Service (provides POI data) and Point of Interests (locations with attributes) to facilitate multi-objective route planning based on user preferences and constraints.
- The system processes user queries to extract key information and preferences, then leverages the MSGS algorithm to identify optimal routes while adhering to user time limits, POI opening hours, and task dependencies.
- LLMAP addresses limitations of existing LLM-as-Agent approaches by separating natural language understanding from complex graph-based search, achieving superior performance and runtime efficiency across diverse routing scenarios.

---

[FREE-MAD: Consensus-Free Multi-Agent Debate](http://arxiv.org/abs/2509.11035)

- FREE-MAD (Consensus-Free Multi-Agent Debate): introduces a novel multi-agent debate framework that eliminates the need for consensus among agents, incorporating a score-based decision mechanism and anti-conformity debate.
- The framework evaluates the entire debate trajectory, rather than just the final round, to assign scores to candidate responses, improving accuracy and fairness.
- FREE-MAD significantly enhances reasoning performance, scalability, and robustness by reducing token costs through single-round debates and mitigating error propagation from LLM conformity.

---

#### 13th September 2025

[Autonomous real-time control of turbulent dynamics](http://arxiv.org/abs/2509.11002)

- REACT (Reinforcement Learning for Environmental Adaptation and Control of Turbulence): introduces a fully autonomous reinforcement learning framework for real-time, adaptive, closed-loop turbulence control in real-world environments, utilizing a real-time control loop, training loop, policy and critic network architectures, and a hardware/software infrastructure.
- The framework learns directly from sparse experimental measurements in a wind tunnel, bypassing complex simulations, and achieves net energy savings by suppressing spatio-temporally coherent flow structures.
- A physics-informed training strategy, recasting data into dimensionless physical groups, enables a single generalizable agent to transfer across speeds without retraining, demonstrating robust and interpretable real-world control of high-Reynolds turbulence.

---

[Is the 'Agent' Paradigm a Limiting Framework for Next-Generation Intelligent Systems?](http://arxiv.org/abs/2509.10875)

- Agent Paradigm Re-evaluation Framework: critically re-evaluates the necessity and optimality of the agent-centric paradigm in AI, distinguishing between agentic, agential, and non-agentic systems, while proposing a shift towards system-level dynamics, world modeling, and material intelligence.
- The paper deconstructs the agent paradigm across various AI frameworks, highlighting conceptual ambiguities, anthropocentric biases, and challenges in defining and measuring properties like autonomy and goal-directedness.
- It argues that the 'agentic' framing of many AI systems, particularly LLM-based ones, can be misleading and may obscure underlying computational mechanisms, advocating for exploring non-agentic and systemic frameworks for robust, scalable, and potentially non-anthropomorphic general intelligence.

---

[Agent-based Simulation for Drone Charging in an Internet of Things Environment System](http://arxiv.org/abs/2509.10867)

- ABS: introduces an agent-based simulation model for coordinating drone battery recharging in IoT and Industry 4.0 environments, featuring autonomous drones, a recharging area, and a working area.
- The model employs a Charger Threshold (CT) policy within each drone, leveraging the El Farol Bar Problem mechanism and historical data from a Radio Base Station (RBS) to make decentralized recharge decisions.
- A machine learning technique is utilized to perform sensitivity analysis on the simulation's nine parameters, identifying Battery Consumption (BC) and Decision Lower Value (LW) as the most influential factors.

---

[GAPRUNE: GRADIENT-ALIGNMENT PRUNING FOR DOMAIN-AWARE EMBEDDINGS](http://arxiv.org/abs/2509.10844)

- GAPrune (Gradient-Alignment Pruning): introduces a pruning framework that addresses the challenge of deploying LLM-based embedding models in resource-constrained environments by considering both domain importance and preserving general linguistic foundation, utilizing Representative Sampling (distill essential properties), Parameter Analysis (characterize parameter behavior), Fisher Information (quantify parameter importance), Gradient Alignment (assess cross-domain objective alignment), Domain Alignment Importance (DAI) Scoring (combine signals for pruning), and Pruning (remove low DAI score parameters).
- The framework measures parameter importance using Fisher Information and general-domain gradient alignment to assess parameter behavior, combining these signals into a Domain Alignment Importance (DAI) score to identify parameters crucial for domain performance and general objective alignment.
- Experiments demonstrate that GAPrune maintains performance within 2.5% of dense models at 50% sparsity in one-shot pruning and achieves performance improvements with retraining, enhancing domain-specific capabilities while achieving model compression.

---

[Towards Automated Error Discovery: A Study in Conversational AI](http://arxiv.org/abs/2509.10833)

- SEEED (Soft Clustering Extended Encoder-Based Error Detection): introduces a framework for detecting and defining errors in conversational AI, which includes Summary Generation (LLM-based dialogue summarization), Context Encoder (dialogue context processing), Summary Encoder (summary processing), Linear Layer (representation aggregation), Soft Clustering (error type identification), Training Objective (joint loss for discrimination and robustness), Label-Based Sample Ranking (contrastive learning sampling), Soft Nearest Neighbor Loss (distance-weighted sampling), and Error Definition Generation (LLM-based definition creation).
- This approach addresses the challenge of identifying both known and unknown error types in conversational AI by leveraging soft clustering and a novel sampling strategy to improve representation learning and generalization.
- SEEED outperforms adapted baselines in detecting novel error types and demonstrates strong generalization to unknown intent detection, with LLMs also effectively generating definitions for newly discovered errors.

---

[EditDuet: A Multi-Agent System for Video Non-Linear Editing](http://arxiv.org/abs/2509.10761)

- EditDuet: introduces a multi-agent system for video non-linear editing, with Critic (LLM agent), Editor (LLM agent), Non-linear Editing (NLE) Environment, Draft timeline, User Request, A-Roll, Video Collection, Editor Explorer, Editor Labeler, Editor Scorer, Self-Reflecting Editor, Critic Explorer, Critic Labeler, and Critic Scorer, where the system automates video editing by iterative interaction between LLM agents within an NLE environment.
- The system employs a Critic agent to provide natural language feedback and an Editor agent to execute editing actions on a draft timeline, guided by a user request and a video collection.
- EditDuet utilizes an in-context learning approach with auxiliary agents to generate synthetic demonstrations, improving communication and performance between the main Editor and Critic agents.

---

#### 12th September 2025

[DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL](http://arxiv.org/abs/2509.10446)

- DeepDive: introduces a framework for advancing deep search agents, integrating a Knowledge Graph (KG) for automated QA pair synthesis, an LLM-obscure component for attribute obfuscation, and Multi-turn Reinforcement Learning (RL) training for enhanced reasoning and tool use within a Web Environment.
- The framework addresses the limitations of open LLMs in deep search by generating complex, hard-to-find questions from KGs and employing end-to-end multi-turn RL to improve long-horizon reasoning and efficient tool calls.
- DeepDive's approach enables test-time scaling of tool calls and parallel sampling, demonstrating significant performance improvements across multiple deep search benchmarks and outperforming existing open-source models.

---

[RefactorCoderQA: Benchmarking LLMs for Multi-Domain Coding Question Solutions in Cloud and Edge Deployment](http://arxiv.org/abs/2509.10436)

- RefactorCoder (RefactorCoder Agentic Framework): introduces a novel cloud-edge collaborative architecture with a multi-agent prompting framework, including GuideLLM (methodological guidance generation), SolverLLM (code solution generation), and JudgeLLM (automated solution evaluation), to benchmark LLMs for multi-domain coding tasks.
- The framework utilizes RefactorCoder-MoE, a fine-tuned LLM, to process user queries into structured requests, generate executable code solutions, and provide automated evaluation feedback and scoring.
- RefactorCoderQA, a comprehensive benchmark built from real-world Stack Overflow coding questions, is used to evaluate LLM performance across software engineering, data science, machine learning, and natural language processing domains.

---

[RecoWorld: Building Simulated Environments for Agentic Recommender Systems](http://arxiv.org/abs/2509.10397)

- RECOWORLD (Building Simulated Environments for Agentic Recommender Systems): introduces a blueprint for building simulated environments tailored to agentic recommender systems, featuring a dual-view architecture with a User Simulator and an Agentic RecSys, where the Agentic RecSys is an Agent with Perception, Reasoning and Planning, Action (Tool Use), and Memory capabilities, supported by Recommender System Modules and System Configs, all leveraging LLMs, diverse Content Representation Models, and Multi-turn RL.
- The framework enables multi-turn interactions between simulated users and agentic recommenders, optimizing for long-term user retention and engagement by generating dynamic feedback loops.
- RECOWORLD supports diverse content representations, including text-based, multimodal, and semantic ID modeling, and facilitates multi-agent simulations for evaluating targeted user populations.

---

[ROBOT GUIDE WITH MULTI-AGENT CONTROL AND AUTOMATIC SCENARIO GENERATION WITH LLM](http://arxiv.org/abs/2509.10317)

- Robot Guide with Multi-Agent Control and Automatic Scenario Generation with LLM: introduces a hybrid control architecture that combines a multi-agent resource management system with LLM-based automatic behavior scenario generation for anthropomorphic tour guide robots.
- The system automates scenario creation through a two-stage LLM process, generating stylized narratives and integrating non-verbal action tags, while the multi-agent system handles coordination and conflict resolution.
- This approach significantly reduces manual configuration, enhances flexibility, and improves the naturalness of robot behavior, as validated on the MENTOR-1 tour guide robot.

---

[Compartmentalised Agentic Reasoning for Clinical NLI](http://arxiv.org/abs/2509.10222)

- CARENLI (Compartmentalised Agentic Reasoning for Clinical NLI): introduces a framework for clinical Natural Language Inference that separates knowledge access from principled inference, including a Planner, specialized Solvers, Verifiers, and Refiners.
- This framework routes premise-statement pairs to family-specific agents for Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction, enforcing auditable procedures and principled decision rules.
- CARENLI significantly improves reasoning fidelity and reliability in clinical NLI by preventing LLMs from defaulting to generic heuristics and aligning them with domain-grounded inferential schemas.

---

[Towards Fully Automated Molecular Simulations: Multi-Agent Framework for Simulation Setup and Force Field Extraction](http://arxiv.org/abs/2509.10210)

- Multi-Agent Framework: introduces an LLM-based multi-agent system for automated molecular simulations, including a User, Experiment Planning Team, Experiment Setup Team, Research Team, Experiment Analysis Team, Global Memory, Simulator, and various specialized agents and tools.
- The framework enables autonomous understanding of characterization tasks, planning simulations, assembling force fields, execution, and interpretation of results for porous materials.
- This approach aims to accelerate materials discovery by automating complex simulation workflows and force field selection, bridging experimental observations with predictive insights.

---

[Population-Aligned Persona Generation for LLM-based Social Simulation](http://arxiv.org/abs/2509.10127)

- Population-Aligned Persona Framework: introduces a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation, including Seed Persona Mining (extracts/filters high-quality narrative personas), Global Distribution Alignment (aligns persona distributions with human data), and Group-specific Persona Construction (adapts personas for specific groups).
- The framework leverages LLMs for persona generation and quality control, employs a two-stage resampling method combining Importance Sampling and Optimal Transport for global alignment, and utilizes an embedding model with LLM revision for group-specific adaptation.
- This approach significantly reduces population-level bias and enhances the accuracy and flexibility of social simulations for diverse research and policy applications by ensuring persona sets authentically reflect real-world population diversity.

---

[XAgents: A Unified Framework for Multi-Agent Cooperation via IF-THEN Rules and Multipolar Task Processing Graph](http://arxiv.org/abs/2509.10054)

- XAgents (A Unified Framework for Multi-Agent Cooperation via IF-THEN Rules and Multipolar Task Processing Graph): introduces a unified multi-agent cooperative framework, with Multipolar Task Processing Graph (MTPG) for dynamic task planning and IF-THEN Rule-based Decision Mechanism (ITRDM) for rule-based agent guidance, enabling robust task execution under uncertainty and mitigating LLM hallucinations.
- The MTPG, inspired by biological multipolar neurons, uses SIMO for divergent task decomposition and MISO for convergent result fusion, while the ITRDM employs various agents (PA, DAA, DEA, FEA, GEA) and IF-THEN rules to constrain and guide agent behavior.
- The framework dynamically restructures task processing paths and utilizes rule-based semantic confrontation to resolve conflicts and ensure global goal alignment, demonstrating superior performance in knowledge- and logic-typed question-answering tasks.

---

[GAMA: A General Anonymizing Multi-Agent System for Privacy Preservation Enhanced by Domain Rules and Disproof Method](http://arxiv.org/abs/2509.10018)

- GAMA (General Anonymizing Multi-Agent system): introduces a privacy-preserving architecture that divides agent workspaces into private and public spaces, utilizing AMPP (Anonymizing Mechanism for Privacy Preservation) for sensitive data anonymization, and DRKE (Domain-Rule-based Knowledge Enhancement) and DLE (Disproof-based Logic Enhancement) in the public space to mitigate semantic loss and enhance logical consistency.
- The system's private space employs MVPI (Multi-View Privacy Identification) to identify privacy-named entities, a Privacy Box for entity-placeholder mapping, and Anonymizing and Nominating Agents for data transformation and restoration.
- In the public space, DRKE uses a Domain Analyzing Agent and Auto Prompting to construct domain rules for Expert Agents, while DLE employs an iterative Disproof Process with Expert and Assistant Agents to identify and resolve logical contradictions, suppressing LLM hallucinations.

---

[QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading](http://arxiv.org/abs/2509.09995)

- QuantAgent: introduces a multi-agent LLM framework for high-frequency algorithmic trading, with IndicatorAgent, PatternAgent, TrendAgent, RiskAgent, and DecisionAgent components, designed to integrate classical technical analysis with LLM reasoning for real-time market decisions.
- The framework decomposes trading into specialized agents, each equipped with domain-specific tools and structured reasoning capabilities to capture distinct aspects of market dynamics over short temporal windows.
- QuantAgent operates solely on price-derived market signals, avoiding textual inputs to ensure fast, interpretable, and risk-aware decision-making in high-frequency financial markets.

---

[Securing LLM-Generated Embedded Firmware through AI Agent-Driven Validation and Patching](http://arxiv.org/abs/2509.09970)

- LLM-based Control System Architecture with AI Agent Integration and Security Modeling: introduces a three-phase methodology for securing LLM-generated embedded firmware, integrating LLM-driven firmware generation with automated security validation and iterative refinement in a virtualized environment, utilizing components like an LLM Engine, Security Analyzer, Fuzzing Engine, and specialized AI agents for threat detection, performance optimization, and compliance verification.
- The framework systematically identifies and mitigates vulnerabilities in LLM-generated firmware by leveraging software-based testing frameworks in a controlled virtualized environment, addressing buffer overflows, race conditions, and denial-of-service threats.
- This iterative process, augmented by AI agents, enhances firmware security and performance, ensuring adherence to real-time operational requirements and contributing to an open-source dataset for future research.

---

[SciML Agents: Write the Solver, Not the Solution](http://arxiv.org/abs/2509.09936)

- SciML Agents: introduces a framework that leverages LLMs to generate scientifically appropriate Python code for solving Ordinary Differential Equations (ODEs) from natural-language descriptions, utilizing numerical solvers and evaluated against two novel benchmarks.
- The framework employs various LLMs, guided prompting, and fine-tuning to enhance code executability, numerical validity, and the ability to make domain-aware numerical choices, such as appropriate solver selection.
- The paper demonstrates that careful guidance and targeted fine-tuning enable LLMs to act as reliable SciML agents, capable of robust symbolic reasoning and accurate scientific code generation for ODE problems.

---

[Dark Patterns Meet GUI Agents: LLM Agent Susceptibility to Manipulative Interfaces and the Role of Human Oversight](http://arxiv.org/abs/2509.10723)

- GUI Agent Susceptibility Study: introduces a two-phase empirical study examining how LLM-powered GUI agents, human participants, and human-AI teams respond to 16 types of dark patterns across diverse scenarios, with all components including Adapted LLM-based GUI Agents, End-to-end GUI Agents, Human Participants, and Human Oversight, where the study investigates agent susceptibility to manipulative interfaces and the effectiveness of human oversight.
- The research highlights that GUI agents often fail to recognize dark patterns and prioritize task completion over protective actions, while human oversight, though improving avoidance, introduces costs like attentional tunneling and cognitive load.
- The findings reveal distinct susceptibility profiles and failure modes between agents and humans, underscoring the need for transparency, adjustable autonomy, and improved oversight mechanisms in GUI agent design and deployment.

---

[MAESTRO: Self-Improving Text-to-Image Generation via Agent Orchestration](http://arxiv.org/abs/2509.10704)

- MAESTRO (Multi-Agent Orchestration): introduces a self-evolving image generation system that enables T2I models to autonomously self-improve generated images through iterative prompt evolution, using only an initial prompt.
- The system incorporates self-critique via specialized MLLM critic agents for identifying image weaknesses and generating interpretable edit signals, which are then integrated by a verifier agent.
- It also employs self-evolution using an MLLM-as-a-judge for head-to-head comparisons, eschewing problematic images and evolving creative prompt candidates that align with user intents.

---

[Self-Supervised Goal-Reaching Results in Multi-Agent Cooperation and Exploration](http://arxiv.org/abs/2509.10656)

- ICRL (Independent Contrastive Reinforcement Learning): introduces a self-supervised goal-reaching framework for multi-agent cooperation and exploration, including a policy (actor), a critic (representation learner), encoders (for observations/actions and goals), and a replay buffer (experience storage).
- The framework enables agents to learn from sparse feedback by maximizing the likelihood of visiting a specified goal state, removing the need for complex reward function design.
- ICRL demonstrates emergent cooperation and exploration in multi-agent reinforcement learning benchmarks, outperforming alternative approaches in sparse reward settings.

---

[DECAMP: Towards Scene-Consistent Multi-Agent Motion Prediction with Disentangled Context-Aware Pre-Training](http://arxiv.org/abs/2509.10426)

- DECAMP (DisEntangled Context-Aware pre-training framework for Multi-agent Prediction): introduces a disentangled context-aware pre-training framework for multi-agent motion prediction, decoupling behavior pattern learning from latent feature reconstruction.
- The framework utilizes an encoder-regressor-decoder pipeline during pre-training to learn robust behavioral priors through collaborative spatial reconstruction and motion recognition pretext tasks.
- During fine-tuning, a pre-trained encoder, trajectory generator, and score decoder are used to produce scene-consistent joint predictions for multiple interacting agents.

---

[A Holistic Architecture for Monitoring and Optimization of Robust Multi-Agent Path Finding Plan Execution](http://arxiv.org/abs/2509.10284)

- Holistic Architecture for Robust Multi-Agent Path Finding Plan Execution: introduces a system for robust execution, monitoring, and optimization of MAPF plans, integrating a robust execution layer, an Action Dependency Graph (ADG), and monitoring and optimization components.
- This architecture leverages the ADG to maintain an estimate of expected execution duration, using slack computation to predict when replanning or rescheduling could reduce execution time due to accumulated delays.
- The system is evaluated using both a real-life robotic fleet demonstrator and a real-time simulator, incorporating an intruder model to simulate unexpected delays and assess the architecture's effectiveness in mitigating their impact.

---

[Deep Reinforcement Learning for Active Flow Control around a Three-Dimensional Flow-Separated Wing at Re = 1,000](http://arxiv.org/abs/2509.10195)

- CFD-DRL framework: introduces a deep reinforcement learning approach for active flow control, integrating a GPU-accelerated CFD Environment with a DRL Agent via a Redis in-memory database to reduce flow separation on a three-dimensional wing.
- The framework leverages the SOD2D solver and TF-Agents DRL library, managed by SmartSim, to enable rapid training and address the "two-language" problem between physics solvers and ML libraries.
- This methodology demonstrates DRL's potential to autonomously identify optimal control actions, significantly reducing drag and lift oscillations for complex aerodynamic challenges.

---

[Virtual Agent Economies](http://arxiv.org/abs/2509.10147)

- Sandbox Economy: introduces a framework for designing steerable AI agent markets, encompassing autonomous AI agents, virtual currencies, market mechanisms, blockchain technology, identity and reputation systems, interoperability protocols, and a hybrid oversight infrastructure, all guided by legislative and regulatory frameworks.
- The framework aims to address the emergent economic layer where AI agents transact and coordinate, considering dimensions of origin (emergent vs. intentional) and permeability (permeable vs. impermeable) to ensure safe and aligned operation.
- Key components like Verifiable Credentials, Decentralized Identifiers, and Proof-of-Personhood mechanisms are proposed to establish trust, accountability, and fair resource allocation within these complex multi-agent ecosystems.

---

[The (R)evolution of Scientific Workflows in the Agentic AI Era: Towards Autonomous Science](http://arxiv.org/abs/2509.09915)

- Architectural Blueprint for Autonomous Science: introduces a conceptual framework and architectural blueprint for evolving scientific workflows towards fully autonomous, distributed scientific laboratories, aiming to accelerate discovery by integrating AI agents and advanced coordination mechanisms.
- The framework defines an evolutionary path along two dimensions: intelligence (from static to intelligent) and composition (from single to swarm), enabling a systematic progression from traditional workflow management systems to AI-driven autonomous discovery.
- This blueprint outlines a layered architecture with components for human interaction, intelligent services, workflow orchestration, communication, data management, and infrastructure abstraction, all deployed across federated facilities to support real-time scientific discovery loops.

---

[SCOPE: Speech-guided Collaborative PErception Framework for Surgical Scene Segmentation](http://arxiv.org/abs/2509.10748)

- SCOPE (Speech-guided Collaborative PErception): introduces a speech-guided collaborative perception framework that integrates LLM reasoning with open-set VFM perception to enable hands-free, on-the-fly segmentation, labeling, and tracking of surgical instruments and anatomy in intraoperative video streams, utilizing a User Query, Speech to Text (STT) Module, System Input, System Prompt, History of Dialogue, LLM Planner, GPT Response, Action Plan, Selected Tool, Tool Parameters, Tool Executer, Output Formatter, Text to Speech (TTS) Module, Vision Foundation Models (VFMs), Modules, User Interaction Examples, and System Rules.
- The framework's core is a collaborative perception agent that refines VFM-generated segmentation outputs and labels scene elements through intuitive speech feedback from clinicians, adapting dynamically to evolving surgical contexts.
- This human-AI collaboration paradigm supports hands-free interaction and real-time adaptability, showcasing its potential for developing surgeon-centric tools in dynamic operating-room environments.

---

#### 11th September 2025

[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](http://arxiv.org/abs/2509.09677)

- LHRTCF: introduces a framework for measuring LLM long-horizon execution by modeling tasks as a sequence of retrieve-then-compose steps, where LLMs are provided with explicit plans and knowledge. 
- The study reveals that LLMs exhibit a self-conditioning effect, where past errors increase the likelihood of future mistakes, and this is not mitigated by scaling model size alone. 
- Thinking models, which generate explicit reasoning traces, effectively fix self-conditioning and enable LLMs to execute significantly longer and more complex tasks in a single turn. 

---

[Bridging the Capability Gap: Harmonizing Multi-Agent Systems via Joint Alignment Tuning](http://arxiv.org/abs/2509.09629)

- MOAT (Multi-Agent Joint Alignment Tuning): introduces a framework that iteratively aligns planning and grounding LLM-based agents to bridge capability gaps, including a Planning Agent, Grounding Agent, and Critic Model, which are optimized through K-Times Sampling, Perplexity as Rewards, DPO Training, Action Refinement, and Supervised Fine-tuning.
- The framework alternates between Planning Agent Alignment, which optimizes the planning agent to generate better subgoal sequences, and Grounding Agent Improving, which enhances the grounding agent's generalization capability using diverse, critic-corrected subgoal-action pairs.
- This joint alignment tuning process ensures a non-decreasing and progressively convergent training, leading to improved coordination and holistic performance across various tasks.

---

[TrEnv: Transparently Share Serverless Execution Environments Across Different Functions and Nodes](http://arxiv.org/abs/2509.09525)

- TRENv (Transparently Share Serverless Execution Environments): introduces a serverless platform that transparently shares execution environments across different functions and nodes, leveraging repurposable sandboxes, an mm-template API, CXL/RDMA memory pools, OS/hypervisor enhancements, CRIU integration, rootfs reconfiguration, cgroup optimization, browser sharing, and virtio-pmem devices to minimize startup latency and memory overhead for LLM agents.
- The platform optimizes for both container- and VM-based environments by enabling fast reuse and restoration of execution environments through memory templates and repurposable sandboxes, significantly reducing P99 latency and memory usage.
- TRENv addresses the high overhead of serverless computing for LLM agents by tackling cold starts, memory stranding, and state duplication, making serverless deployments more cost-efficient and scalable.

---

[Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference](http://arxiv.org/abs/2509.09505)

- PLENA (Programmable Long-context Efficient Neural Accelerator): introduces a hardware-software co-designed system that addresses memory walls in long-context agentic LLM inference through a flattened systolic array, an asymmetric quantization scheme, and native FlashAttention support, supported by a full toolchain including a custom ISA, compiler, simulator, and DSE flow.
- The system achieves high utilization by tailoring its flattened systolic array architecture to "fat GEMMs" and employs an asymmetric quantization scheme with mixed data types to reduce memory bandwidth and capacity limitations.
- PLENA's comprehensive toolchain, including a custom ISA and compiler, enables rapid adaptation and optimization for emerging Transformer models, delivering significantly higher throughput and utilization than existing accelerators.

---

[MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems](http://arxiv.org/abs/2509.09360)

- MetaRAG (Metamorphic Testing for Hallucination Detection in Retrieval-Augmented Generation Systems): introduces a real-time, unsupervised, black-box metamorphic testing framework for hallucination detection in RAG systems, which includes Factoid Extraction, Mutation Generation, Factoid Verification, Score Calculation, and Identity-Aware Safeguards.
- The framework decomposes RAG answers into atomic factoids, generates controlled mutations (synonym/antonym substitutions), verifies each variant against the retrieved context, and aggregates inconsistencies into a response-level hallucination score.
- MetaRAG's span-level detection enables identity-aware safeguards by localizing unsupported claims and translating scores into topic-conditioned deployment policies, such as stricter thresholds or forced citations.

---

[Towards Adaptive ML Benchmarks: Web-Agent-Driven Construction, Domain Expansion, and Metric Optimization](http://arxiv.org/abs/2509.09321)

- TAM-Bench (Adaptive ML Benchmarks): introduces a diverse, realistic, and structured benchmark for evaluating LLM-based agents on end-to-end ML tasks, with components including Benchmark Source, Automated Web Scraping (featuring an LLM-based Agent Layer), Raw Content, Task Standardization (with LLM Transform), AutoML Agent, and Evaluator (incorporating LLM-as-a-Judge Constraint Pass), designed to address limitations in existing benchmarks.
- The framework leverages browser automation and LLMs for automated task collection and standardization, leaderboard-based difficulty modeling, and a multi-dimensional evaluation framework to assess agent capabilities holistically.
- TAM-Bench constructs benchmark subsets (Lite, Medium, Full) from 150 curated AutoML tasks, ensuring balanced coverage across data modalities and difficulty levels for robust and scalable evaluation.

---

[Can Multimodal LLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization](http://arxiv.org/abs/2509.09307)

- MatCha: introduces a multimodal benchmark for materials characterization image understanding, comprising task construction (defining tasks, extracting terms), data curation (collecting and processing HTML, figures, captions, supplementary datasets), and question generation (using GPT-4o, AI filtering, and expert review).
- The benchmark features 1,500 expert-level multiple-choice questions across 21 distinct tasks, reflecting real-world scientific challenges in materials research, covering processing correlation, morphology, structure, and property analysis.
- Evaluations on MatCha reveal a significant performance gap between state-of-the-art MLLMs and human experts, particularly in tasks requiring higher-level expertise and sophisticated visual perception, highlighting limitations in domain knowledge and reasoning.

---

[LightAgent: Production-level Open-source Agentic AI Framework](http://arxiv.org/abs/2509.09292)

- LightAgent (Production-level Open-source Agentic AI Framework): introduces a lightweight yet powerful agentic framework, integrating Agent, Memory (mem0), Tools, Tool Generator, Tree of Thought (ToT), LightSwarm, and LLMs to streamline multi-agent application development by resolving the trade-off between flexibility and simplicity.
- The framework redefines efficiency through a minimalist architecture, enabling autonomous tool generation, multi-agent collaboration, and robust fault tolerance with a 100% Python codebase of only 1,000 lines.
- LightAgent ensures rapid deployment across diverse scenarios by supporting dynamic agent specialization, multi-modal data handling, and compatibility with major LLMs and streaming APIs.

---

[Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](http://arxiv.org/abs/2509.09265)

- EMPG (Entropy-Modulated Policy Gradients): introduces a framework that re-calibrates the learning signal for LLM agents based on step-wise uncertainty and the final task outcome, addressing the coupling between gradient magnitude and policy entropy.
- This framework includes Self-Calibrating Gradient Scaling to dynamically adjust policy updates and a Future Clarity Bonus to guide agents towards predictable solution paths.
- EMPG's approach amplifies updates for confident correct actions, penalizes confident errors, and attenuates updates from uncertain steps, leading to more efficient and stable learning in long-horizon tasks.

---

[JUPITER: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](http://arxiv.org/abs/2509.09245)

- JUPITER: introduces a framework that enhances LLM data analysis capabilities by formulating data analysis as a search problem, leveraging Monte Carlo Tree Search for trajectory collection and a Value Model for inference-time value-guided search.
- The framework utilizes a fine-tuned LLM to generate thought-action pairs within a Jupyter Context, which are then executed in a Code Execution Environment to explore solution paths.
- JUPITER's Value Model, trained on MCTS-generated trajectories and Q-values from the NbQA Dataset, efficiently guides the search process, enabling open-source LLMs to achieve competitive performance with commercial agent systems on complex multi-step data analysis tasks.

---

[Agentic LLMs for Question Answering over Tabular Data](http://arxiv.org/abs/2509.09234)

- Agentic NL-to-SQL Pipeline: introduces a multi-stage LLM-driven framework for Question Answering over Tabular Data, including example selection, SQL query generation, answer extraction and formatting, answer verification, and answer reprocessing.
- This pipeline leverages LLMs (GPT-40, GPT-40-mini, DeepSeek v2:16b) for dynamic SQL query generation and answer refinement, utilizing embedding-based similarity for context and Chain-of-Thought prompting for enhanced reasoning.
- The system integrates a verification mechanism and iterative reprocessing to ensure query correctness, improve robustness across diverse table structures, and maximize answer accuracy.

---

[Enabling Regulatory Multi-Agent Collaboration: Architecture, Challenges, and Solutions](http://arxiv.org/abs/2509.09215)

- BEMAR (Blockchain-Empowered Multi-Agent Regulation): introduces a blockchain-enabled layered architecture for regulatory agent collaboration, comprising an Agent Layer (manages agents and collects data), a Blockchain Data Layer (maintains an immutable ledger), and a Regulatory Application Layer (provides advanced functionalities).
- The framework integrates three key modules: an agent behavior tracing and arbitration module for automated accountability, a dynamic reputation evaluation module for trust assessment, and a malicious behavior forecasting module for early adversarial detection.
- This architecture establishes a systematic foundation for trustworthy, resilient, and scalable regulatory mechanisms in large-scale LLM-empowered agent ecosystems by leveraging blockchain's transparency and immutability.

---

[On Integrating Large Language Models and Scenario-Based Programming for Improving Software Reliability](http://arxiv.org/abs/2509.09194)

- LLM-SBP Methodology (Large Language Model - Scenario-Based Programming Methodology): introduces a hybrid approach for software development that integrates LLMs with Scenario-Based Programming, leveraging LLMs for code generation and strategic reasoning while mitigating their weaknesses through structured, modular, and verifiable design.
- The methodology involves a human-guided, modular workflow with components like structured prompts, iterative refinement, and human-in-the-loop feedback to reduce hallucinations and improve logical consistency.
- It enables developers to define high-level behavioral goals, decompose them into modular scenario objects, provide background knowledge to the LLM, and incrementally develop and refine scenario threads with continuous testing and formal verification.

---

[AI Reasoning for Wireless Communications and Networking: A Survey and Perspectives](http://arxiv.org/abs/2509.09193)

- AI Reasoning for Wireless Communications and Networking: introduces a comprehensive survey of reasoning-enabled AI in wireless communication networks, covering prompting strategies, architectural approaches, and learning paradigms, where the paper systematically categorizes and examines AI reasoning methods and their layered applications from the physical to the application layer.
- The survey highlights how LLM-based agents can combine reasoning with long-term planning, memory, tool utilization, and autonomous cross-layer control to dynamically optimize network operations with minimal human intervention.
- It addresses the limitations of traditional AI in dynamic environments, interpretability, and generalization, charting a path for integrating advanced reasoning techniques into next-generation wireless networks.

---

[Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining](http://arxiv.org/abs/2509.09071)

- Multi-Agent Bargaining Game Evaluation Framework: introduces a novel multi-player bargaining game to directly compare human, LLM, and Bayesian agent performance and behavioral dynamics in dynamic negotiation settings.
- The framework evaluates agents based on surplus trajectories, trading patterns, and regret minimization, revealing distinct strategic approaches despite similar aggregate outcomes for humans and LLMs.
- Bayesian agents achieve the highest surplus through aggressive optimization, while LLMs favor conservative, concessionary trades, and humans employ more strategic, risk-taking, and fairness-oriented behaviors.

---

[CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models](http://arxiv.org/abs/2509.09675)

- CDE (Curiosity-Driven Exploration): introduces a framework that enhances Reinforcement Learning with Verifiable Rewards (RLVR) in LLMs by leveraging intrinsic curiosity signals from both the Actor (using a Perplexity Bonus) and the Critic (using a Variance Bonus from a Multi-head Critic) to guide exploration.
- The framework integrates these curiosity signals as an exploration bonus within existing RL algorithms like GRPO and PPO, mitigating issues such as premature convergence and entropy collapse in LLM training.
- CDE's theoretical analysis demonstrates its ability to penalize overconfident errors and encourage diverse correct responses, leading to improved calibration and consistent performance gains on mathematical reasoning benchmarks.

---

[Vibe Check: Understanding the Effects of LLM-Based Conversational Agents' Personality and Alignment on User Perceptions in Goal-Oriented Tasks](http://arxiv.org/abs/2509.09870)

- TMK (Trait Modulation Key) Framework: introduces a modular prompting framework that systematically controls LLM-based Conversational Agents' (CAs) personality across Big Five traits at low, medium, and high expression levels using Personality Keys and Style Cues Keys.
- The framework enables nuanced personality expression in LLMs, moving beyond binary trait manipulations to investigate the impact of personality expression and user-agent alignment on user perceptions in goal-oriented tasks.
- The study utilizes a text-only chat interface for user interaction with CAs, evaluating user perceptions across measures like Intelligence, Enjoyment, and Trust, and assessing personality alignment via Euclidean distance.

---

[LLMs as Agentic Cooperative Players in Multiplayer UNO](http://arxiv.org/abs/2509.09867)

- LLM UNO: introduces a framework that enables decoder-only LLMs to act as agentic cooperative players in the RLCard UNO environment, receiving game state and selecting actions via text prompts.
- The framework evaluates LLMs ranging from 1B to 70B parameters in both autonomous play against a random agent and cooperative play assisting a rule-based teammate in a three-player game.
- It investigates how model scale and prompting techniques, specifically cloze and counterfactual prompting, influence LLM performance and cooperative effectiveness in strategic decision-making.

---

[Latency and Token-Aware Test-Time Compute](http://arxiv.org/abs/2509.09864)

- Latency and Token-Aware Inference-Time Scaling Framework: introduces a latency- and token-aware approach for inference-time scaling, dynamically allocating compute and selecting methods per query using its Query Input, Decoding Strategies, Utility Predictors, Utility Function, and Optimal Strategy Selector, supported by an LLM, PRM, and Embedding Backbones.
- The framework explicitly incorporates both token cost and wall-clock latency into its utility formulation, which is crucial for user experience and efficient agentic workflows requiring multiple LLM queries.
- Experiments on reasoning benchmarks demonstrate that this query-adaptive approach consistently outperforms static strategies, achieving favorable accuracy-cost trade-offs.

---

[SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness Under Resource Constraints](http://arxiv.org/abs/2509.09853)

- SWE-Effi introduces a multi-dimensional framework for re-evaluating AI systems in software engineering, incorporating Core Performance Metrics (raw measurements) and Resource Effectiveness Metrics (derived efficiency scores) to assess holistic effectiveness.
- This framework defines effectiveness as the balance between solution accuracy (e.g., resolve rate) and consumed resources (e.g., tokens, time, cost), addressing the limitations of traditional single-metric evaluations.
- The paper applies SWE-Effi to re-rank popular AI systems on a SWE-bench subset, revealing insights into LLM-scaffold synergy, the "Token Snowball" effect, and "expensive failures" where unresolved tasks consume excessive resources.

---

[Meta-Learning Reinforcement Learning for Crypto-Return Prediction](http://arxiv.org/abs/2509.09751)

- Meta-RL-Crypto: introduces a unified transformer-based architecture that unifies meta-learning and reinforcement learning to create a self-improving trading agent, featuring an Actor, Judge, and Meta-Judge in a closed-loop system.
- This framework leverages multimodal market inputs and internal preference feedback, continuously refining its trading policy and evaluation criteria without human supervision.
- A multi-objective reward design, incorporating profitability, risk control, liquidity, and sentiment alignment, prevents reward hacking and promotes robust trading behavior.

---

[Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster](http://arxiv.org/abs/2509.06426)

- Drosophila Leg Musculoskeletal Modeling Pipeline: introduces a 3D, data-driven musculoskeletal model of Drosophila legs, implemented in OpenSim and MuJoCo, which incorporates Hill-type muscle representation and is optimized using morphological imaging and 3D pose estimation data.
- This model enables muscle-actuated behavioral replay, predicts coordinated muscle synergies across diverse behaviors, and investigates the impact of passive joint properties on learning speed.
- The framework integrates anatomical, physiological, and behavioral data into a unified modeling approach, providing insights into motor control biomechanics and facilitating the control of embodied artificial agents.

---

[Self-Augmented Robot Trajectory: Efficient Imitation Learning via Safe Self-augmentation with Demonstrator-annotated Precision](http://arxiv.org/abs/2509.09893)

- SART (Self-Augmented Robot Trajectory): introduces a framework where a **Human Expert** provides a **Single Human Demonstration** and **Precision Boundaries** to enable a **Robot Self-Augmentation Module** to generate an **Augmented Training Dataset** for learning a robust **Parametric Policy** on a **Robot System**.
- This framework minimizes human effort by requiring only one demonstration and subsequent annotation via an **Interactive Annotation Interface**, followed by autonomous data expansion within defined safe regions.
- SART demonstrates higher success rates in clearance-limited robotic manipulation tasks across both **Simulation Environment** and **Real-world Environment**, improving data collection efficiency and policy robustness.

---

[Curriculum-Based Multi-Tier Semantic Exploration via Deep Reinforcement Learning](http://arxiv.org/abs/2509.09356)

- VLM-DRL framework: introduces a novel Deep Reinforcement Learning architecture for resource-efficient semantic exploration, integrating a Vision-Language Model (VLM) common-sense through a layered reward function and a curriculum learning strategy.
- The framework enables an agent to strategically query the VLM for external guidance, optimizing resource usage, and progressively develops exploration skills through distinct training phases.
- This approach enhances object discovery rates and guides the agent towards semantically rich regions, demonstrating a scalable method for embedding common-sense reasoning in autonomous exploration.

---

[Flip Co-op: Cooperative Takeovers in Shared Autonomy](http://arxiv.org/abs/2509.09281)

- Flip Co-op (Cooperative Takeovers in Shared Autonomy): introduces a game-theoretic framework for modeling cooperative takeovers in shared autonomy, including dynamic game formulation, human agent, autonomous agent, FlipDyn state, takeover actions, stochastic human intent, Nash equilibrium (NE) strategies, saddle-point value functions, linear-quadratic (LQ) system model, bimatrix potential game reformulation, and unifying potential function, where it formulates control switching as a dynamic game to derive Nash equilibrium-based takeover strategies.
- The framework establishes the existence and characterization of NE under stochastic human intent and provides closed-form recursions for LQ systems, enabling efficient computation of cooperative takeover policies.
- It further extends the model to partially misaligned utilities through a bimatrix potential game reformulation, demonstrating its applicability to a vehicle trajectory tracking problem.

---

[ProgD: Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting](http://arxiv.org/abs/2509.09210)

- ProgD (Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting): introduces a novel progressive multi-scale decoding strategy for joint multi-agent motion forecasting, utilizing dynamic heterogeneous graphs to model evolving social interactions and progressively eliminate uncertainty. 
- The framework employs an encoder-decoder architecture, where the encoder processes historical agent states and road networks, and the decoder forecasts future motions through a temporal module, dynamic graph construction, and dual-stage graph convolution modules. 
- ProgD achieves state-of-the-art performance by adaptively improving the modeling of evolving context and enhancing prediction accuracy and consistency through its multi-scale decoding procedure. 

---

[Occupancy-aware Trajectory Planning for Autonomous Valet Parking in Uncertain Dynamic Environments](http://arxiv.org/abs/2509.09206)

- Occupancy-aware Trajectory Planning Framework: introduces a system for autonomous valet parking in dynamic, uncertain environments, with Static Map (prior knowledge of parking lot), Prediction Model (dynamic agent states and predictions), Spot Occupancy Estimator (recursive belief estimation of spot occupancy), Strategy Planner (action selection based on occupancy forecasts), Path Planner (trajectory generation), and Reference Tracker (vehicle control).
- The framework predicts future parking spot occupancy by distinguishing between initially vacant and occupied spots, leveraging predicted motion of dynamic agents, and incorporating partial, noisy observations within a limited Field-of-View model.
- It adaptively balances goal-directed parking maneuvers with exploratory navigation based on information gain, intelligently incorporating wait-and-go behaviors at promising spots to improve parking efficiency and safety.

---

[Mind Meets Space: Rethinking Agentic Spatial Intelligence from a Neuroscience-inspired Perspective](http://arxiv.org/abs/2509.09154)

- Agentic Spatial Intelligence Framework: introduces a novel computational framework grounded in neuroscience principles, designed to advance agentic spatial intelligence for better interaction with the physical 3D world, comprising bio-inspired multimodal sensing, multi-sensory integration, egocentric-allocentric conversion, an artificial cognitive map, spatial memory, and spatial reasoning.
- This framework mimics human spatial cognition by processing diverse sensory inputs, transforming raw data into structured representations, converting first-person views into world-centered maps, building an internal spatial model, maintaining adaptive long-term memory, and enabling goal-oriented decision-making.
- The framework provides a structured pathway for developing human-like spatial intelligence in AI agents, addressing limitations of current systems in representing spatial structures, reasoning about spatial relationships, and planning in spatial contexts.

---

[MAGICGUI: A FOUNDATIONAL MOBILE GUI AGENT WITH SCALABLE DATA PIPELINE AND REINFORCEMENT FINE-TUNING](http://arxiv.org/abs/2508.03700)

- MagicGUI (Foundational Mobile GUI Agent): introduces a foundational mobile GUI agent designed to address perception, grounding, and reasoning challenges in mobile GUI environments, utilizing a two-stage training procedure that includes GUI Continue Pre-training and Reinforcement Fine-tuning, along with components like Foundational Pre-training, Annealing, CPT Ref Model, KL Divergence, RFT Policy Model, Spatially Enhanced Composite Reward Function, and Dual Filtering GRPO Sampling.
- The framework leverages a scalable GUI Data Pipeline to construct a diverse multimodal dataset, enhancing perception and grounding capabilities through tasks like element referring, grounding, description, screen VQA, and captioning.
- It integrates a comprehensive and unified action space with planning-oriented reasoning mechanisms, enabling the model to decompose complex user instructions into sequential actions and achieve robust generalization across diverse GUI scenarios.

---

#### 10th September 2025

[EVALUATING LLMS WITHOUT ORACLE FEEDBACK: AGENTIC ANNOTATION EVALUATION THROUGH UNSUPERVISED CONSISTENCY SIGNALS](http://arxiv.org/abs/2509.08809)

- Agentic Annotation Evaluation Paradigm: introduces a novel method for evaluating LLM annotation quality without oracle feedback, utilizing a Noisy Teacher (LLM) (generates noisy annotations), a Student Model (MINILM) (evaluates, assigns annotations), a User Preference Distribution (limited labeled data), an Average Similarity (AS) Function (calculates similarity for voting), a Consistent and Inconsistent (CAI) Ratio (measures annotation reliability), a Group Prompting Mechanism (teacher LLM annotation strategy), Consistent Samples (teacher-student agreement), and Inconsistent Samples (teacher-student disagreement).
- The Student Model acts as an unsupervised feedback mechanism, employing a user preference-based majority voting strategy to assess the consistency of the LLM's outputs.
- The CAI Ratio quantifies annotation quality and serves as a critical tool for model selection, demonstrating a strong positive correlation with LLM accuracy in unsupervised settings.

---

[AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning](http://arxiv.org/abs/2509.08755)

- AgentGym-RL: introduces a unified, modular, and flexible end-to-end RL framework for training LLM agents for multi-turn interactive decision-making, encompassing an LLM Agent, an Environment Module, and a Training Module.
- The framework supports diverse real-world scenarios like web navigation, deep search, digital games, embodied tasks, and scientific tasks, and integrates mainstream RL algorithms such as PPO, GRPO, RLOO, and REINFORCE++.
- It also proposes ScalingInter-RL, a progressive interaction-scaling approach that adaptively adjusts the agent-environment interaction horizon to balance exploration-exploitation and enhance optimization stability.

---

[ChemBOMAS: Accelerated BO in Chemistry with LLM-Enhanced Multi-Agent System](http://arxiv.org/abs/2509.08736)

- ChemBOMAS (LLM-Enhanced Multi-Agent System for accelerating BO in chemistry): introduces a novel framework that synergistically integrates knowledge-driven coarse-grained optimization and data-driven fine-grained optimization, where LLMs intelligently decompose the search space and generate pseudo-data points to accelerate Bayesian Optimization in chemistry.
- The framework employs a multi-agent system where LLM-powered agents reason over chemical knowledge to identify promising candidate regions and enhance the BO process by generating informative pseudo-data points.
- This approach significantly improves data utilization efficiency and accelerates convergence, validated through benchmark evaluations and wet-lab experiments on challenging chemical reactions.

---

[SWE-Mirror: Scaling Issue-Resolving Datasets by Mirroring Issues Across Repositories](http://arxiv.org/abs/2509.08724)

- SWE-MIRROR: introduces a three-phase pipeline including Task Collection, Task Mirroring, and Task Validation, which distills real-world issues from GitHub, mirrors them into configured Gym environments using LLMs and agents, and re-animates them as verifiable issue-resolving tasks.
- The framework leverages LLMs like Qwen3-32B, GPT-40-2024-0513, and GPT-4.1, along with specialized Test and Mirror Agents, to generate key artifacts such as `test.patch`, `mirror.patch`, `fix.patch`, and a `problem_statement` for each mirrored task.
- The approach breaks the one-to-one dependency between task context and Gym environments, enabling the creation of a large-scale, verifiable dataset (SWE-MIRROR-60K) that significantly improves LLM-based coding agents' issue-resolving capabilities and cross-lingual generalization.

---

[Architecting Resilient LLM Agents: A Guide to Secure Plan-then-Execute Implementations](http://arxiv.org/abs/2509.08646)

- P-t-E (Plan-then-Execute): introduces a secure architectural pattern for LLM agents, separating strategic planning by a Planner from tactical execution by an Executor, complemented by a Verifier, Refiner, and Re-planner for resilience and security.
- The pattern integrates robust security controls like Input Sanitization and Validation, Output Filtering, Dual LLM Pattern, Principle of Least Privilege, Task-scoped Tool Access, Role-Based Access Control (RBAC), Sandboxed Execution via Docker Containers, and Human-in-the-Loop (HITL) verification.
- P-t-E supports advanced patterns such as Directed Acyclic Graphs (DAGs) for parallel execution, GraphQL Integration for optimized data retrieval, and Graph-Based Conditional Execution Paths, with implementations detailed across LangGraph, CrewAI, and AutoGen frameworks.

---

[AutoODD: Agentic Audits via Bayesian Red Teaming in Black-Box Models](http://arxiv.org/abs/2509.08638)

- AutoODD: introduces an LLM-Agent centric framework for automated generation of semantically relevant test cases to search for failure modes in specialized black-box models, leveraging an Agent (orchestrates testing) that interacts with a Memory (stores past interactions), a Failure Tracker (monitors failure occurrences), an Encoder (embeds text prompts), UMAP Reduction (reduces embedding dimensions), GP Fitting (models failure landscape), a Prompt2Input (generates SUT inputs), and a Testing (evaluates SUT performance) module to audit a SUT (black-box model) and record Results (SUT pass/fail outcomes) from generated Sample #1 (generated test case) and Sample #2 (generated test case).
- The framework combines LLM-Agent orchestration with Bayesian uncertainty estimation to efficiently explore the failure landscape of black-box models within semantically meaningful embedding spaces.
- This approach aims to discover operational boundaries and failure modes in safety-critical systems with significantly reduced sample complexity, providing a scalable methodology for verifying model reliability.

---

[Agents of Discovery](http://arxiv.org/abs/2509.08535)

- Agents of Discovery: introduces an agentic framework where a team of LLM agents collaboratively solves data analysis-based research problems in high-energy physics, utilizing specialized tools and a local execution environment.
- The framework includes a Researcher orchestrating tasks, a Coder writing Python code, and Code and Logic Reviewers providing iterative feedback to refine analysis strategies and ensure reproducibility.
- Evaluated on the LHC Olympics anomaly detection challenge, the system demonstrates the capacity of LLMs to automate complex scientific workflows, with advanced models achieving human state-of-the-art performance.

---

[HUMANAGENCYBENCH: Scalable Evaluation of Human Agency Support in AI Assistants](http://arxiv.org/abs/2509.08494)

- HAB (HUMANAGENCYBENCH): introduces a scalable and adaptive benchmark for evaluating human agency support in AI assistants, including a Simulator LLM (generates user query test candidates), Validation Rubric (criteria for test candidate quality), Validator LLM (scores and filters test candidates), Diversity Sampling (k-means clustering) (selects representative queries), Dimension Test Set (500 simulated user queries per dimension), and an Evaluation Model (LLM) (scores LLM-based assistant responses) across six dimensions of human agency.
- HAB operationalizes human agency into six key dimensions: Ask Clarifying Questions, Avoid Value Manipulation, Correct Misinformation, Defer Important Decisions, Encourage Learning, and Maintain Social Boundaries, each with specific evaluation rubrics to assess LLM-based assistants.
- The framework leverages LLMs for automated test generation and evaluation, providing a systematic approach to assess how different AI assistants support or diminish user control and autonomy in various interaction scenarios.

---

[Co-Investigator AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives](http://arxiv.org/abs/2509.08380)

- Co-Investigator AI: introduces an agentic framework for generating Suspicious Activity Reports (SARs), with a Data Ingestion & Structuring Layer (ingests, transforms raw data), AI-Privacy Guard Layer (identifies, anonymizes sensitive data), Crime Type Detection Agent (extracts risk indicators, classifies typologies), Planning Agent (orchestrates agents, allocates resources), Specialized Typology Detection Agents (analyze specific crime types), External Intelligence Agent (accesses external risk intelligence), Narrative Generation Agent (synthesizes SAR drafts), Compliance Validation Agent (Agent-as-a-Judge) (verifies narrative quality, compliance), Feedback Agent (integrates human feedback), Dynamic Memory Management (maintains regulatory, historical, typology memory), Analytical Tools (extracts risk indicators, searches intelligence, links accounts), and User Interface (human investigator interaction), designed to produce SARs faster and with greater accuracy while maintaining human interpretability.
- The framework leverages a modular, human-in-the-loop design, integrating specialized AI agents for planning, crime type detection, external intelligence gathering, and compliance validation, supported by dynamic memory and an AI-Privacy Guard layer for sensitive data handling.
- This approach aims to streamline SAR drafting, align narratives with regulatory expectations, and enable compliance teams to focus on higher-order analytical work, marking a shift towards scalable, reliable, and transparent SAR generation in Anti-Money Laundering (AML) compliance.

---

[A Systematic Survey on Large Language Models for Evolutionary Optimization: From Modeling to Solving](http://arxiv.org/abs/2509.08269)

- Systematic Survey Taxonomy: introduces a comprehensive review of LLMs for evolutionary optimization, categorizing research into LLMs for Optimization Modeling (automatically transforming natural language to mathematical models) and LLMs for Optimization Solving (enhancing or directly performing optimization tasks).
- The LLMs for Optimization Solving category is further subdivided into LLMs as Optimizers (solving problems via iterative natural language interaction), Low-level LLM-assisted Optimization Algorithms (embedding LLMs for specific operations), and High-level LLM-assisted Optimization Algorithms (orchestrating algorithm selection and generation).
- This taxonomy provides a structured framework for understanding the evolving landscape of LLM applications in optimization, highlighting current challenges and future directions towards self-evolving agentic ecosystems.

---

[Exploratory Retrieval-Augmented Planning For Continual Embodied Instruction Following](http://arxiv.org/abs/2509.08222)

- ExRAP (Exploratory Retrieval-Augmented Planning): introduces an embodied agent framework for continual instruction following in dynamic environments, utilizing a Temporal Embodied Knowledge Graph (TEKG) for environmental context memory and an exploration-integrated task planning scheme.
- The framework enhances LLMs' embodied reasoning by decomposing continual instructions into queries and executions, which are evaluated by a Memory-augmented query evaluator using the TEKG and refined with temporal consistency.
- It integrates an Exploitation planner and an Exploration planner to balance task achievement and environmental knowledge acquisition, demonstrating superior performance in goal success and execution efficiency.

---

[Componentization: Decomposing Monolithic LLM Responses into Manipulable Semantic Units](http://arxiv.org/abs/2509.08203)

- Componentization: introduces an approach that decomposes monolithic LLM responses into modular, independently editable units, leveraging MAOD (Modular and Adaptable Output Decomposition) for semantic segmentation and CBRA (Component-Based Response Architecture) for structured workflow management.
- The MAODchat reference prototype implements CBRA using a microservices architecture, featuring a Flask-based Frontend, FastAPI Backend, FastAPI MAOD Agent, PostgreSQL Database, and a Caddy Reverse Proxy for orchestration.
- This framework enhances human-AI collaboration by enabling user-driven component manipulation (edit, select/toggle, regenerate) and dynamic recomposition, fostering content and architectural resilience against "catastrophic regeneration" failures.

---

[Global Constraint LLM Agents for Text-to-Model Translation](http://arxiv.org/abs/2509.08970)

- Global Constraint LLM Agents: introduces an agentic framework for translating natural language descriptions of optimization problems into MiniZinc models, utilizing specialized LLM agents for global constraint detection and code generation, and an assembler LLM for model integration.
- This framework decomposes the complex text-to-model translation task into smaller, manageable sub-tasks, where each specialized LLM agent focuses on a specific global constraint type, simplifying reasoning.
- The approach demonstrates improved performance over baseline prompting strategies, including chain-of-thought, by reducing cognitive load on individual LLMs and enabling a collaborative modeling process.

---

[GeoJSON Agents: A Multi-Agent LLM Architecture for Geospatial Analysis — Function Calling vs Code Generation](http://arxiv.org/abs/2509.08863)

- GeoJSON Agents (A Multi-Agent LLM Architecture for Geospatial Analysis): introduces an automated spatial analysis framework, with Planner and Worker agents, that transforms natural language tasks into structured GeoJSON operation commands and processes spatial data using Function Calling or Code Generation.
- The framework leverages a Planner agent for task decomposition and a Worker agent for execution, employing either predefined function APIs via Function Calling or dynamically generated Python code via Code Generation.
- This multi-agent LLM architecture significantly outperforms general-purpose LLMs in geospatial analysis, offering enhanced performance and scalability for GIS automation by integrating GeoJSON data.

---


[HYPOGENEAGENT: HYPOTHESIS LANGUAGE AGENT FOR GENE-SET CLUSTER RESOLUTION SELECTION USING PERTURB-SEQ DATASETS](http://arxiv.org/abs/2509.09740)

- HYPOGENEAGENT (Hypothesis Language Agent): introduces a multi-stage LLM-driven framework for gene-set cluster resolution selection, including an Input Gene List (input data for analysis), a General LLM agent (generates initial biological process explanations), a Hypothesis LLM agent (refines and ranks top GO hypotheses with confidence scores), and Output LLM-proposed descriptions (final ranked biological process descriptions).
- The framework quantifies biological relevance by combining semantic similarity, intra-cluster agreement, and inter-cluster distinctiveness to select optimal cluster resolutions and provide biologically informed interpretations.
- HYPOGENEAGENT automates the annotation process, bridging the gap between unsupervised partitioning and biologically informed interpretation in single-cell multi-omics studies.

---

[A Role-Aware Multi-Agent Framework for Financial Education Question Answering with LLMs](http://arxiv.org/abs/2509.09727)

- Role-Aware Agent Framework: introduces a multi-agent system for financial education QA, including an Evidence Retriever (retrieves topic-relevant evidence), a Base Generator (drafts initial answer), and an Expert Reviewer (critiques and refines answer).
- This framework leverages role-based prompting to assign domain-specific personas to LLMs, enhancing reasoning and factual accuracy in financial problem-solving.
- The system demonstrates improved answer accuracy and explanation quality by integrating retrieved evidence and expert critique, outperforming zero-shot Chain-of-Thought baselines.

---

[Dual-Stage Safe Herding Framework for Adversarial Attacker in Dynamic Environment](http://arxiv.org/abs/2509.08460)

- Dual-Stage Safe Herding Framework: introduces a hierarchical hybrid strategy framework for cooperative capture and escort, enabling safe guidance of adversarial agents to designated regions in dynamic, obstacle-dense environments.
- The framework decomposes the problem into a capture stage for initial encirclement and an escort stage for continuous guidance, integrating reach-avoid game theory and local motion planning.
- It utilizes a virtual containment boundary and event-triggered pursuit mechanisms for scalable and robust multi-agent coordination against attackers with unknown evasion strategies.

---

[Leveraging AI Agents for Autonomous Networks: A Reference Architecture and Empirical Studies](http://arxiv.org/abs/2509.08312)

- AN Agent reference architecture: introduces a novel dual-driver architecture for Autonomous Networks, deploying coordinated proactive-reactive runtimes driven by hybrid knowledge representation.
- The framework is empirically validated through a Radio Access Network (RAN) Link Adaptation (LA) Agent, demonstrating sub-10 ms real-time control and significant performance enhancements in 5G NR.
- This architecture integrates self-awareness, choice-making, and decision-making mechanisms, leveraging LLMs and various tools to achieve L4 operational autonomy and overcome traditional automation barriers.

---

[A Comprehensive Review of Reinforcement Learning for Autonomous Driving in the CARLA Simulator](http://arxiv.org/abs/2509.08221)

- A Comprehensive Review of Reinforcement Learning for Autonomous Driving in the CARLA Simulator: introduces a systematic analysis of around 100 peer-reviewed papers, categorizing RL literature by algorithmic family, state/action/reward formulations, evaluation metrics, and CARLA scenarios.
- The review highlights that over 80% of studies rely on model-free RL methods and details the diverse sensor modalities, control abstractions, and reward shaping techniques used in CARLA-based autonomous driving research.
- It consolidates the evaluation landscape, distills persistent challenges like sparse rewards and sim-to-real transfer, and outlines promising future directions for advancing RL-based autonomous driving towards real-world deployment.

---

[MOBILERL: ONLINE AGENTIC REINFORCEMENT LEARNING FOR MOBILE GUI AGENTS](http://arxiv.org/abs/2509.18119)

- MOBILERL: introduces an online agentic reinforcement learning framework for mobile GUI agents, combining reasoning warm-up stages with a difficulty-adaptive policy optimization algorithm (ADAGRPO) for efficient training.
- The framework's ADAGRPO algorithm incorporates Shortest-Path Reward Adjustment (SPA), Difficulty-Adaptive Positive Replay (AdaPR), and Failure Curriculum Filtering (FCF) to stabilize training and improve sample efficiency.
- MOBILERL leverages reasoning-free and reasoning SFT for policy initialization, enabling robust performance across diverse mobile applications and tasks.

---

#### 9th September 2025

[AgentSentinel: An End-to-End and Real-Time Security Defense Framework for Computer-Use Agents](http://arxiv.org/abs/2509.07764)

- AgentSentinel: introduces an end-to-end, real-time security defense framework for computer-use agents, employing a client-server architecture with a Monitor, Instrumented Agent, Tracer, and Auditor to intercept sensitive operations and conduct comprehensive security audits.
- The framework's auditing mechanism correlates current task context with system traces, integrating rule-based and LLM-based approaches, further optimized by a security query cache and QPS optimizer for real-time threat detection.
- AgentSentinel demonstrates an average defense success rate of 79.6% against diverse attack scenarios, significantly outperforming existing baseline defense mechanisms for LLM-based computer-use agents.

---

[VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents](http://arxiv.org/abs/2509.07553)

- VeriOS-Agent: introduces a query-driven human-agent-GUI interaction framework that enables OS agents to determine when to query humans for more trustworthy task completion, built upon a Two-Stage Learning Paradigm that decouples and utilizes Scenario Knowledge and Action Knowledge from the VeriOS-Bench dataset to train an Automated OS Agent.
- The framework allows the VeriOS-Agent to autonomously execute actions in normal conditions while proactively querying humans in untrustworthy scenarios, leveraging human responses to ensure reliable task completion via the GUI.
- This approach improves step-wise success rates in untrustworthy scenarios without compromising normal performance, demonstrating strong generalization and scalability for trustworthy OS agent operation.

---

[Astra: A Multi-Agent System for GPU Kernel Performance Optimization](http://arxiv.org/abs/2509.07506)

- Astra (A Multi-Agent System for GPU Kernel Performance Optimization): introduces, "Astra (A Multi-Agent System for GPU Kernel Performance Optimization)", with Testing Agent (creates test cases), Profiling Agent (measures performance), Planning Agent (proposes modifications), and Coding Agent (generates new kernels), where Astra optimizes existing CUDA GPU kernels from SGLang through iterative code generation, testing, profiling, and planning.
- This multi-agent system achieves an average speedup of 1.32x using zero-shot prompting with OpenAI o4-mini, demonstrating autonomous application of loop transformations, memory access pattern optimization, CUDA intrinsics, and fast math operations.
- The framework focuses on optimizing existing CUDA implementations rather than generating them from PyTorch modules, addressing a critical challenge in LLM serving and training efficiency.

---

[DREAMS: Decentralized Resource Allocation and Service Management across the Compute Continuum Using Service Affinity](http://arxiv.org/abs/2509.07497)

- DREAMS (Decentralized Resource Allocation and Service Management): introduces a decentralized framework for optimizing microservice placement across the compute continuum, featuring Local Domain Managers, Administrative, Configuration Control, Observability and Diagnostics, Domain Monitoring, Migration Intelligence, Consensus Management, Migration Execution, Inter-Domain Communication, and Recovery and Fault Tolerance Modules, along with various repositories.
- The framework enables autonomous agents (LDMs) within each computational domain to collaboratively make service placement decisions using a Raft-based consensus algorithm and cost-benefit voting, ensuring responsive, privacy-preserving, and fault-tolerant coordination.
- DREAMS achieves globally optimized service placements while maintaining high fault tolerance and sub-linear scalability for key coordination operations like LDM registration and migration voting, making it suitable for multi-stakeholder, dynamic manufacturing environments.

---

[Autonomous Code Evolution Meets NP-Completeness](http://arxiv.org/abs/2509.07367)

- SATLUTION (Autonomous Code Evolution Framework): introduces a repository-scale, self-evolving coding framework via LLMs, with Planning and Coding Agent, Planning Claude Model, Coding Claude Model, Cursor Environment, Self-evolved Rulebase, Static Initialization Rules, Dynamic Self-Evolved Rules, SATLUTION Repository, Two-stage Verification Pipeline, Compilation Check, Smoke Test, Full Correctness Validation, SAT Assignment Verifier, DRAT Proof Checker, Distributed Runtime Evaluator, Feedback Metrics, Post-evaluation Analyzer, Rule Update Engine, and Rule Version Manager, designed to autonomously evolve SAT solver repositories under strict correctness guarantees and distributed runtime feedback.
- The framework orchestrates LLM agents through Planning and Coding stages, guided by a self-evolving rule system and a two-stage verification pipeline, to iteratively improve SAT solver engineering at a full repository scale.
- SATLUTION successfully evolved SAT solvers that outperformed human-designed winners of the SAT Competition 2025, demonstrating the potential of AI agents for champion-level performance in NP-complete problem solving.

---

[Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees](http://arxiv.org/abs/2509.07939)

- STT-based reasoning pipeline: introduces a guided reasoning pipeline for LLM-driven penetration testing, incorporating a deterministic Structured Task Tree (STT) to constrain the LLM's reasoning process to explicitly defined tactics, techniques, and procedures from the MITRE ATT&CK Matrix.
- This framework enhances the accuracy and efficiency of automated cybersecurity assessments by guiding the LLM agent through a predefined task flow, reducing hallucinations and unproductive actions compared to self-guided reasoning methods.
- The pipeline enables smaller LLMs to perform complex, multi-step reasoning effectively and consistently, demonstrating significant improvements in subtask completion rates and requiring fewer model queries across various HackTheBox challenges.

---

[KLIPA: A Knowledge Graph and LLM-Driven QA Framework for IP Analysis](http://arxiv.org/abs/2509.07860)

- KLIPA (Knowledge Graph and LLM-Driven Question-Answering Framework for IP Analysis): introduces a novel framework for patent analysis, integrating a Patent Knowledge Graph (structured representation of patent data), a Retrieval-Augmented Generation (RAG) System (retrieves semantically relevant patent information), and a ReAct Agent Framework (dynamically determines retrieval strategy and generates responses) to enhance relationship identification, patent retrieval, and knowledge discovery.
- The framework's Patent Knowledge Graph (KG) is constructed from IP Dataset (raw input patent documents) via NER and RE (Named Entity Recognition and Relation Extraction) and stored in a Neo4j Database, while the LLM (Large Language Model) powers reasoning and response synthesis, supported by modules like Document Parser (processes various document formats) and Text Splitting Module (segments documents into chunks).
- The LLM-based QA agent further leverages a Vector Database (stores document embeddings for RAG) and an Embedding Generation Module (creates dense vector representations) for its Hybrid Retriever (combines vector similarity and keyword matching), with a Gradio-based User Interface (for interactive query handling) facilitating user interaction.

---

[Getting in Contract with Large Language Models – An Agency Theory Perspective on Large Language Model Alignment](http://arxiv.org/abs/2509.07642)

- LLM ATLAS (LLM Agency Theory-Led Alignment Strategy): introduces a conceptual framework grounded in agency theory to mitigate LLM alignment problems during organizational LLM adoption, by combining organizational LLM adoption phases and agency theory concepts to derive an LLM alignment problem-solution space.
- The framework identifies information asymmetries between the adopting organization (principal) and the black-box LLM (agent) as the root cause of alignment issues, categorizing them as hidden characteristics or hidden actions.
- It provides practical solutions like signaling through model cards and screening via adversarial attacks, or bonding through human preference incorporation and monitoring via model-driven supervision, tailored to specific LLM adoption phases.

---

[AgentX: Towards Orchestrating Robust Agentic Workflow Patterns with FaaS-hosted MCP Services](http://arxiv.org/abs/2509.07595)

- AgentX: introduces a novel agentic workflow pattern, composed of a Stage Generation Agent, a Planner Agent, and an Executor Agent, designed to orchestrate robust agentic workflows with FaaS-hosted Model Context Protocol (MCP) services.
- This framework decomposes user tasks into sequential stages, with agents collaboratively creating detailed plans and executing them using external tools, while actively managing context to prevent information overload and hallucinations.
- The paper evaluates AgentX's performance against state-of-the-art patterns, demonstrating its competitive or superior success rate, and explores FaaS deployment of MCP servers to enhance scalability, security, and accessibility for real-world applications.

---

[TOWARDS GENERALIZED ROUTING: MODEL AND AGENT ORCHESTRATION FOR ADAPTIVE AND EFFICIENT INFERENCE](http://arxiv.org/abs/2509.07571)

- MoMA (Mixture of Models and Agents): introduces a generalized routing framework that integrates both LLM and agent-based routing, effectively handling diverse queries through precise intent recognition and adaptive routing strategies.
- The framework employs a two-layer routing mechanism, first determining if an LLM can handle the query, then either selecting an optimal agent via a context-aware FSM or an optimal LLM based on a score-cost tradeoff.
- MoMA achieves an optimal balance between efficiency and cost by leveraging a detailed training dataset to profile LLM and agent capabilities, ensuring robust and scalable adaptive services.

---

[FEED-O-METER: Fostering Design Feedback Skills through Role-playing Interactions with AI Mentee](http://arxiv.org/abs/2509.07424)

- FEED-O-METER (Fostering Design Feedback Skills through Role-playing Interactions with AI Mentee): introduces a novel system that employs carefully designed LLM-based agents to create an environment for students to practice giving design feedback, featuring a User Interface (displays design information, enables chat, visualizes feedback), an LLM-based Pipeline (generates responses, categorizes feedback, extracts knowledge, updates ideas), an AI Mentee (LLM-based agent, novice design student persona), a Knowledge State (stores mentee's design expertise), and an Action Plan (tracks design refinement recommendations).
- The system allows users to role-play as mentors, providing feedback to an AI mentee and enabling them to reflect on how their feedback impacts the AI mentee's idea development process through real-time visualizations and counter-questions.
- By simulating a realistic, low-pressure feedback environment, FEED-O-METER aims to enhance students' design feedback skills, critical thinking, and self-reflection without the anxiety of real-world judgment.

---

[Talking with Oompa Loompas: A novel framework for evaluating linguistic acquisition of LLM agents](http://arxiv.org/abs/2509.07389)

- Oompa Loompas framework: introduces a novel experimental framework for evaluating the linguistic acquisition of LLM agents, featuring an LLM agent, a deterministic Oompa Loompa bot, the constructed Tinkatongue language, a feedback mechanism, a system prompt, a synthetic dataset, and evaluation metrics.
- The framework assesses an LLM agent's ability to learn Tinkatongue through pattern recognition and interactive feedback from the Oompa Loompa bot, which provides "koro" for valid sentences and "moko lira bani" for invalid ones.
- This evaluation method simulates human-like language acquisition, revealing that LLM agents adopt strategies like imitation and babbling, and highlighting the challenge of sustained language learning despite good feedback responsiveness.

---

[Towards Post-mortem Data Management Principles for Generative AI](http://arxiv.org/abs/2509.07375)

- Post-mortem Data Management Principles: introduces a framework for managing deceased individuals' data in Generative AI systems, encompassing an analysis phase, three core principles, and deployment strategies.
- The framework addresses data ownership, privacy, and ethical concerns for post-mortem data, proposing rights like data deletion and inheritance, alongside purpose limits.
- It recommends both regulatory enforcement and technical solutions, such as digital wills and privacy-preserving techniques, to implement these principles effectively.

---

[SpecifyUI: Supporting Iterative UI Design Intent Expression through Structured Specifications and Generative AI](http://arxiv.org/abs/2509.07334)

- SpecifyUI (interactive system): introduces a vision-centered intermediate representation, SPEC, to make design intent explicit and controllable in UI generation, enabling users to extract specifications, compose elements into a coherent whole, and iteratively refine designs through direct selection and element extraction.
- The system leverages a multi-agent UI generation pipeline, including Region Segmentation, VLM, and LLM components, to translate structured specifications into high-fidelity UI designs, supporting targeted edits at global, regional, and component levels.
- SpecifyUI integrates a Retrieval-Augmented Generation (RAG) system and a Debug Agent to enhance generation fidelity and robustness by grounding LLM outputs with a SPEC-UI Code Database and self-correcting errors.

---

[CancerGUIDE: Cancer Guideline Understanding via Internal Disagreement Estimation](http://arxiv.org/abs/2509.07325)

- CancerGUIDE framework: introduces an LLM agent-based approach to automatically generate guideline-concordant treatment trajectories for non-small cell lung cancer patients, leveraging a meta-classifier to verify prediction accuracy with calibrated confidence scores.
- The framework addresses the evaluation bottleneck for LLM performance on guideline adherence by combining expensive human annotations with model consistency information, enabling scalable assessment without extensive expert annotation.
- It establishes a clinically viable framework for LLM-based guideline adherence systems that balance accuracy, interpretability, and regulatory requirements while reducing annotation costs and providing a scalable pathway toward automated clinical decision support.

---

[XML Prompting as Grammar-Constrained Interaction: Fixed-Point Semantics, Convergence Guarantees, and Human-AI Protocols](http://arxiv.org/abs/2509.08182)

- XML Prompting as Grammar-Constrained Interaction: introduces a logic-first treatment of XML prompting that unifies grammar-constrained decoding, fixed-point semantics over hierarchical prompts, and convergent human-AI interaction loops.
- The framework formalizes XML prompting as a typed tree language with a refinement order, defining prompt transformers (T) to capture interaction rounds and proving the existence of least fixed points for steady-state protocols.
- It further introduces a task-aware tree metric, demonstrating Banach-style convergence for iterative guidance and providing multi-layer human-AI interaction templates with correctness guarantees.

---

[K2-Think: A Parameter-Efficient Reasoning System](http://arxiv.org/abs/2509.07604)

- K2-THINK (K2-Think: A Parameter-Efficient Reasoning System): introduces a reasoning system built on a 32B parameter LLM, enhanced by Long Chain-of-thought Supervised Finetuning (SFT) (CoT training), Reinforcement Learning with Verifiable Rewards (RLVR) (reasoning performance enhancement), Plan-Before-You-Think (Agentic Planning) (structured prompt generation), Best-of-N Sampling (Test-time Scaling) (optimal response selection), Speculative Decoding (inference acceleration), and Cerebras Wafer-Scale Engine (WSE) (inference-optimized hardware).
- This system achieves frontier reasoning performance comparable to much larger models by synergistically combining advanced post-training and test-time computation techniques.
- K2-THINK prioritizes mathematical reasoning, achieving state-of-the-art scores on public benchmarks for open-source models while maintaining strong performance in code and science domains.

---

[∆L Normalization: RETHINK Loss AGGREGATION IN RLVR](http://arxiv.org/abs/2509.07558)

- ∆L Normalization: introduces a simple yet effective loss aggregation method tailored for Reinforcement Learning with Verifiable Rewards (RLVR) to address high gradient variance and unstable optimization caused by dynamic generation lengths in LLMs.
- The method provides an unbiased estimate of the true policy loss and minimizes gradient variance by applying specific normalization weights derived from response lengths and a tunable hyperparameter α.
- Extensive experiments demonstrate that the method consistently achieves superior results across different model sizes, maximum lengths, and tasks, promoting stable training and higher accuracy.

---

[THE CHOICE OF DIVERGENCE: A NEGLECTED KEY TO MITIGATING DIVERSITY COLLAPSE IN REINFORCEMENT LEARNING WITH VERIFIABLE REWARD](http://arxiv.org/abs/2509.07430)

- DPH-RL (Diversity-Preserving Hybrid RL): introduces a novel framework that leverages mass-covering f-divergences (e.g., Forward-KL, JS-divergence) as a rehearsal mechanism, continuously referencing the initial policy to maintain broad solution coverage and mitigate diversity collapse in Reinforcement Learning with Verifiable Reward (RLVR).
- The framework operates in two stages: a pre-sampling stage that partitions a static dataset into Dpef (perfect examples) and Dexp (challenging examples), and an online training stage that applies distinct loss functions (Lexp for exploration and Lpef for knowledge retention) to these subsets.
- DPH-RL, implemented as DPH-F (Forward-KL) or DPH-JS (JS-divergence), improves multi-attempt performance (Pass@k) and single-attempt accuracy (Pass@1) both in- and out-of-domain, while being training-efficient by computing f-divergence using generator functions from the initial policy.

---

[Multi Robot Coordination in Highly Dynamic Environments: Tackling Asymmetric Obstacles and Limited Communication](http://arxiv.org/abs/2509.08859)

- DWM and DTA: introduces a novel distributed coordination method for multi-agent systems, addressing task assignments in dynamic environments with asymmetric obstacles and limited communication by enhancing situational awareness and modeling accuracy.
- The approach leverages a market-based coordination system, a distributed world-model estimation scheme, and a novel asymmetric entity-model (AM) integrated with Elliptical Line Voronoi Diagrams (ELVDs).
- This framework enables agents to maintain a coherent world representation and perform efficient task assignments even with high communication loss, reducing task overlaps in challenging scenarios like RoboCup soccer.

---

[Risk-Bounded Multi-Agent Visual Navigation via Dynamic Budget Allocation](http://arxiv.org/abs/2509.08157)

- RB-CBS (Risk-Bounded Conflict-Based Search): introduces a novel multi-agent pathfinding framework that dynamically allocates and adjusts user-specified risk bounds across agents to balance safety and efficiency in visual navigation tasks.
- The framework employs a two-level search, with a high-level component managing global constraints and risk budgets, and a low-level component finding individual agent paths within assigned risk limits.
- It integrates learned distance and risk critics from goal-conditioned reinforcement learning to construct an irregular waypoint graph, enabling collision-free path planning in visually rich, unstructured environments.

---

[EnvX: Agentize Everything with Agentic AI](http://arxiv.org/abs/2509.08088)

- EnvX: introduces a novel LLM-based framework that agentizes GitHub repositories into intelligent, autonomous agents capable of natural language interaction and inter-agent collaboration.
- The framework operates through three phases: TODO-guided environment initialization, human-aligned agentic automation, and Agent-to-Agent (A2A) communication, transforming static code into active, interactive components.
- EnvX integrates specialized tools and an A2A protocol to automate repository understanding, initialization, and operationalization, enabling multi-repository collaboration and enhancing software reuse.

---

#### 8th September 2025


[DISENTANGLING INTERACTION AND BIAS EFFECTS IN OPINION DYNAMICS OF LARGE LANGUAGE MODELS](http://arxiv.org/abs/2509.06858)

- Bayesian framework: introduces a Bayesian framework to disentangle and quantify three biases (topic bias, agreement bias, anchoring bias) and interaction effects in LLM opinion dynamics, where it models observed opinion shifts in multi-step dialogues between LLM agents.
- The framework quantifies the influence of these factors on opinion trajectories, revealing that LLMs tend to converge to a shared attractor, with interaction effects fading over time and biases differing between LLMs.
- It also introduces opinion uncertainty, measured by Shannon entropy, as a predictor for subsequent opinion shifts and demonstrates that fine-tuning LLMs can shift opinion attractors, highlighting both opportunities and challenges in using LLMs as proxies for human behavior.

---

[RAFFLES: Reasoning-based Attribution of Faults for LLM Systems](http://arxiv.org/abs/2509.06822)

- RAFFLES (Reasoning-based Attribution of Faults for LLM Systems): introduces an iterative evaluation architecture for multi-component LLM systems, utilizing a central Judge and specialized Evaluators to systematically identify decisive faults within a system's execution Trajectory (T), storing evaluation history in a Memory component (H) to refine fault attribution and output the Decisive fault (i, t).
- The framework's Judge proposes candidate agent-step fault pairs and provides rationales based on primacy, fault condition, and causality, while multiple Evaluators rigorously verify these rationales and the consistency of the proposed fault with the execution log, returning confidence scores.
- This iterative refinement process, where Evaluator feedback is fed back to the Judge via the Memory component, enables RAFFLES to achieve higher accuracy in pinpointing the earliest causal faults in complex, long-horizon LLM agentic systems compared to existing methods.

---

[Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks](http://arxiv.org/abs/2509.06701)

- Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks: introduces a theoretical framework for modeling neural agents as probabilistic generative models composed of interacting subagents, defining their beliefs and welfare through outcome distributions and epistemic utility, and aggregating them via logarithmic pooling to form a coherent composition belief.
- The framework establishes that unanimously beneficial compositions are possible with three or more outcomes under logarithmic pooling, but impossible for binary outcomes or under linear pooling, and demonstrates recursive and robustness properties for compositional agents.
- The paper formalizes the Waluigi effect in LLMs, showing that manifesting and then suppressing an antagonistic persona (Waluigi) yields greater misalignment reduction than pure reinforcement of a benevolent persona (Luigi) alone, offering insights into agentic alignment challenges.

---

[Demo: Healthcare Agent Orchestrator (HAO) for Patient Summarization in Molecular Tumor Boards](http://arxiv.org/abs/2509.06602)

- HAO (Healthcare Agent Orchestrator): introduces a modular, LLM-driven multi-agent system that coordinates specialized agents, a general reasoner, and domain-specific tools over unified data sources via a user interface to generate patient summaries for Molecular Tumor Boards.
- TBFact, a "model-as-a-judge" framework, evaluates patient summaries by extracting clinical factual claims, classifying their importance, assessing bidirectional entailment using an LLM, and attributing errors to quantify completeness and succinctness.
- The HAO framework is designed for precision, traceability, and safety-by-design, enabling grounded reasoning across heterogeneous data sources and supporting diverse use cases from rapid single-agent timelines to complex multi-agent workflows.

---

[Simulating Dispute Mediation with LLM-Based Agents for Legal Research](http://arxiv.org/abs/2509.06586)

- AgentMediation: introduces an LLM-based agent framework for simulating legal dispute mediation, featuring data preprocessing, a five-stage mediation simulation framework, configurable party and mediator agents, and a dual evaluation system.
- The framework leverages real-world civil dispute data to create structured inputs, models disputant behaviors using TKI conflict modes, and allows mediators to access external legal knowledge, enabling controlled experimentation on key variables.
- Its dual evaluation system assesses both mediation outcomes (success rate, satisfaction, consensus, litigation risk via LLM-as-a-judge) and solution quality (points of contention, legal bases using ROUGE-L, BERTScore, Recall), providing a comprehensive platform for legal research.

---

[WebEXPLORER: Explore and Evolve for Training Long-Horizon Web Agents](http://arxiv.org/abs/2509.06501)

- WebEXPLORER: introduces a systematic data generation approach, combining Model-Based Exploration (constructs information space) and Iterative Query Evolution (increases query difficulty), to create the WebEXPLORER-QA Dataset (synthesized challenging QA pairs) for training the WebEXPLORER-8B (trained web agent) via Supervised Fine-tuning (initializes model capabilities) and Reinforcement Learning (optimizes reasoning strategies).
- The framework generates challenging query-answer pairs that require multi-step reasoning and complex web navigation, enabling the development of advanced web agents equipped with Search Tool (retrieves relevant information) and Browse Tool (analyzes URL content).
- The WebEXPLORER-8B model, trained with this approach, achieves state-of-the-art performance at its scale on various information-seeking benchmarks and demonstrates strong generalization capabilities for long-horizon web agents.

---

[Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers](http://arxiv.org/abs/2509.06493)

- BFS-Prover-V2: introduces a system designed to address training-time reinforcement learning and inference-time compute scaling challenges for LLM step-provers, achieving state-of-the-art results on formal mathematics benchmarks.
- The system features a novel multi-turn off-policy RL framework with a multi-stage expert iteration pipeline, including adaptive tactic-level data filtering and periodic retraining, to continuously improve LLM performance at training time.
- It also incorporates a planner-enhanced multi-agent search architecture for inference-time scaling, where a high-level LLM Planner decomposes complex theorems into subgoals for parallel Prover agents, leveraging a shared proof cache.

---

[Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning](http://arxiv.org/abs/2509.06436)

- TOA (Tree of Agents): introduces a multi-agent reasoning framework that segments long input texts into chunks, processed by independent agents, to address long-context challenges like "lost in the middle" by enabling multi-perspective understanding through tree-structured path exploration.
- The framework enhances processing efficiency by incorporating prefix-hash caching to reduce redundant cognition generations and adaptive pruning strategies to terminate useless reasoning paths early.
- TOA's agents collaborate by exchanging local cognition and forming a consensus through a two-tier hierarchical voting mechanism, demonstrating comparable performance to larger commercial LLMs using a smaller base model.

---

[Context-Adaptive Hearing Aid Fitting Advisor through Multi-turn Multimodal LLM Conversation](http://arxiv.org/abs/2509.06382)

- CAFA (Context-Adaptive Fitting Advisor): introduces a multimodal, multi-agent LLM system for real-time, personalized hearing aid adjustments, integrating live ambient audio, audiograms, and user feedback through its Ambient Sound Recognition Pipeline, Multi-Agent Multi-turn Workflow, and LLM Judge.
- The system's Multi-Agent Multi-turn Workflow comprises a Context Acquisition Agent, Subproblem Classifier Agent, Strategy Provider Agent, and Ethical Regulator Agent, all overseen by an independent LLM Judge to ensure clinical safety and quality.
- CAFA leverages LLMs for multi-step reasoning and agentic task execution, translating context and user feedback into precise, safe tuning commands, and achieving high ambient sound classification accuracy for enhanced conversational efficiency.

---

[Evaluating Multi-Turn Bargain Skills in LLM-Based Seller Agents](http://arxiv.org/abs/2509.06341)

- BargainBench framework: introduces a multi-turn evaluation framework for LLM-based seller agents, including an Intent Factory (extracts intent space), Problem Weaver (generates scripted dialogues), and Evaluation Center (scores LLM performance), designed to measure bargaining ability by tracking buyer intents in e-commerce dialogues.
- The framework provides a large-scale e-commerce bargaining benchmark with turn-level evaluation grounded in Theory of Mind, moving beyond outcome-only metrics to assess intermediate reasoning.
- It also features an automated pipeline for extracting reliable intent from massive dialogue data, enabling scalable and reproducible benchmarking of bargaining agents.

---

[A Fragile Number Sense: Probing the Elemental Limits of Numerical Reasoning in LLMs](http://arxiv.org/abs/2509.06332)

- Divide-and-Reconstruct Framework: introduces a multi-level framework that decomposes complex numerical reasoning tasks into elementary skills, evaluating LLMs on both isolated skills and their integration to analyze performance and identify reasoning limitations.
- The paper probes LLM mathematical numeracy across escalating complexity, from basic arithmetic to combinatorial puzzles like the Game of 24, revealing a "fragile number sense" in LLMs.
- Results indicate LLMs excel at deterministic algorithmic execution but consistently fail at tasks requiring heuristic search over large combinatorial spaces, suggesting their numerical reasoning is more pattern-matching than generative problem-solving.

---

[SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents](http://arxiv.org/abs/2509.06283)

- SFR-DeepResearch: introduces a framework for training autonomous single-agent LLMs for Deep Research, featuring an Agentic Inference Pipeline with a Memory Management System and essential tools, and an RL Training Recipe that includes a Synthetic Data Generation Pipeline, a modified REINFORCE Algorithm, and a Reward Modeling System.
- The framework enhances agentic capabilities by continually training reasoning-optimized LLMs using synthetic data and a novel RL recipe that incorporates length-normalized advantage and strategic trajectory filtering to stabilize policy optimization.
- The system also includes a robust RL Infrastructure with asynchronous processing, a local toolbox, and optimized GPU resource management to ensure scalability and fault-tolerance during training.

---

[TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning](http://arxiv.org/abs/2509.06278)

- TableMind (Autonomous Programmatic Agent for Tool-Augmented Table Reasoning): introduces an LLM-driven table reasoning agent that autonomously performs multi-turn tool invocation, writes and executes data-analyzing code in a secure sandbox, and exhibits planning and self-reflection capabilities, utilizing a two-stage fine-tuning paradigm with SFT and RFT, enhanced by RAPO.
- The agent operates through an iterative plan-action-reflect loop, where the Planning Component formulates strategies, the Action Component generates Python code executed in a Code Sandbox/Interpreter, and the Reflection Component evaluates outcomes to dynamically adjust subsequent actions.
- Training involves a Supervised Fine-tuning (SFT) Module for foundational tool-use patterns and a Reinforcement Fine-tuning (RFT) Module, guided by a Multi-objective Reward Function and the Rank-Aware Policy Optimization (RAPO) Algorithm, to achieve strategic autonomy and robust problem-solving.

---

[REMI: A Novel Causal Schema Memory Architecture for Personalized Lifestyle Recommendation Agents](http://arxiv.org/abs/2509.06269)

- REMI (Causal Schema Memory): introduces a novel architecture for personalized multimodal lifestyle agents that combines a personal causal knowledge graph, a causal reasoning engine, a schema-based planner, and LLM orchestration to deliver explainable, personalized recommendations.
- The framework leverages two LLM components: an LLM (Reasoning) for path scoring and hypothesis generation, and an LLM (Orchestrator) for integrating information and composing final recommendations.
- This approach addresses limitations of current personalized recommendation agents by providing context-aware, user-aligned, and transparent causal explanations for lifestyle advice.

---

[Neuro-Symbolic AI for Cybersecurity: State of the Art, Challenges, and Opportunities](http://arxiv.org/abs/2509.06921)

- G-I-A (Grounding-Instructibility-Alignment) Framework: introduces, "a novel framework to evaluate Neuro-Symbolic AI (NeSy) systems in cybersecurity", with Grounding (connects outputs to cybersecurity concepts), Mathematical Consistency (aligns neural predictions with logical reasoning), Knowledge Graphs (provides domain knowledge), Adversarial Robustness (maintains stability against novel attacks), Instructibility (responds to analyst feedback), Dynamic Adaptation (adapts to evolving threats), Human-AI Collaboration (integrates expert guidance), Analyst Feedback (updates system components), Alignment (ensures consistency with cybersecurity objectives), Institutional Goals (aligns with organizational priorities), Ethical Constraints (prevents malicious misuse), Cybersecurity Objectives (reflects true security goals), Neural Component (performs pattern recognition), Symbolic Component (provides domain knowledge and rules), Neuro-Symbolic Fusion (integrates neural and symbolic outputs), Explainable Alerts (generates human-readable security alerts), where this framework systematically characterizes the field by analyzing 127 publications spanning 2019-July 2025.
- The paper highlights consistent advantages of multi-agent NeSy architectures, identifies critical implementation challenges, and emphasizes causal reasoning integration for proactive defense strategies.
- The survey demonstrates dual-use implications where autonomous systems achieve substantial zero-day exploitation capabilities with significant cost reductions, fundamentally altering threat dynamics and demanding responsible development.

---

[AxelSMOTE: An Agent-Based Oversampling Algorithm for Imbalanced Classification](http://arxiv.org/abs/2509.06875)

- AxelSMOTE (Axelord Synthetic Minority Oversampling Technique): introduces an agent-based oversampling algorithm for imbalanced classification, treating data instances as autonomous agents, and includes trait-based feature grouping (partitions features into related groups), similarity assessment (calculates cultural similarity between agents), a probabilistic exchange mechanism (exchanges traits based on similarity and probability), Beta blending (interpolates features realistically), and diversity injection (adds controlled noise for variety).
- This approach addresses limitations of traditional oversampling by preserving feature correlations, ensuring meaningful interactions between compatible instances, and generating diverse synthetic samples to avoid overfitting.
- Inspired by Axelrod's cultural dissemination model, AxelSMOTE systematically generates realistic synthetic minority instances, outperforming state-of-the-art methods while maintaining computational efficiency.

---

[Agentic DDQN-Based Scheduling for Licensed and Unlicensed Band Allocation in Sidelink Networks](http://arxiv.org/abs/2509.06775)

- Agentic DDQN-based scheduling framework: introduces an AI-driven DDQN framework for licensed and unlicensed band allocation in NR sidelink networks, comprising a DDQN Agent, Online Network, Target Network, Replay Buffer, Scheduler, State Representation, Action Space, and Reward Mechanism, designed to autonomously perceive network dynamics and adapt scheduling policies.
- This framework leverages agentic AI principles to integrate queueing delay, channel quality, Wi-Fi coexistence dynamics, and link-switching stability into its state representation and reward function, enabling QoS-aware and adaptive resource allocation.
- The proposed system significantly reduces blocking rates by up to 87.5% compared to threshold-based scheduling, demonstrating its potential for stable and adaptive resource management in congested and coexistence-limited environments.

---

[MAS-Bench: A Unified Benchmark for Shortcut-Augmented Hybrid Mobile GUI Agents](http://arxiv.org/abs/2509.06477)

- MAS-Bench (Unified Benchmark for Shortcut-Augmented Hybrid Mobile GUI Agents): introduces a benchmark for evaluating GUI-shortcut hybrid mobile agents, featuring a Shortcut Generation Stage (agent creates shortcuts), a Quality Evaluation Stage (evaluates generated shortcuts), an Online Evaluation Environment (dynamic Android platform), a GUI-Shortcut Hybrid Action Space (combines GUI and shortcut actions), Tasks Design (complex real-world scenarios), and Evaluation Metrics (Success Rate, Efficiency, Cost), where it systematically assesses agents' ability to discover, generate, and utilize shortcuts to enhance task efficiency and success.
- The benchmark includes 139 complex tasks across 11 real-world applications, a knowledge base of 88 predefined shortcuts (APIs, deep-links, RPA scripts), and supports the dynamic generation of new shortcuts by agents.
- Experiments demonstrate that hybrid agents achieve significantly higher success rates and efficiency than GUI-only counterparts, highlighting the effectiveness of integrating shortcuts and the potential for agent-generated shortcuts.

---

[Interactive Shaping of Granular Media Using Reinforcement Learning](http://arxiv.org/abs/2509.06469)

- RL framework: introduces an RL framework that enables a robotic arm with a cubic end-effector and a stereo camera to shape granular media into desired target structures, utilizing compact observations and concise reward formulations for effective learning.
- This framework reconstructs the current height map (Current He) from depth images (Depth image Ic) and uses its difference (Heightmap difference Ha) with the desired goal height map (Goal Hg) as a key observation for the Policy π (RL Agent).
- The approach demonstrates zero-shot transfer of trained policies from physics simulation to a real robot, outperforming baselines in shaping accuracy and robustness for complex granular media manipulation tasks.

---

[STAYING IN THE SWEET SPOT: RESPONSIVE REASONING EVOLUTION VIA CAPABILITY-ADAPTIVE HINT SCAFFOLDING](http://arxiv.org/abs/2509.06923)

- SEELE (reSponsive rEasoning Evolution via capability-adaptivE hint scaffolding): introduces a novel supervision-aided Reinforcement Learning with Verifiable Rewards (RLVR) framework that dynamically adjusts problem difficulty to maintain high learning efficiency by adapting hint lengths based on the LLM's evolving capability.
- This framework employs a multi-round rollout sampling strategy and an Item Response Theory-based accuracy-hint model to predict and achieve an optimal 50% rollout accuracy for each problem instance.
- By integrating real-time feedback and instance-level difficulty adjustment, SEELE significantly outperforms existing RLVR methods on various math reasoning and general domain benchmarks.

---

[DCPO: Dynamic Clipping Policy Optimization](http://arxiv.org/abs/2509.02333)

- DCPO (Dynamic Clipping Policy Optimization): introduces a novel reinforcement learning pipeline that enhances LLM reasoning capabilities through its Dynamic Adaptive Clipping (DAC) mechanism, which adaptively adjusts clipping bounds based on token-specific prior probabilities, and its Smooth Advantage Standardization (SAS) technique, which standardizes advantages across cumulative training steps, further incorporating an Only Token Mean loss (OTM) for efficient gradient updates.
- The framework addresses limitations of existing policy optimization methods, such as zero gradients and inefficient data utilization, by promoting token-level exploration and stabilizing training.
- DCPO demonstrates superior performance on mathematical reasoning benchmarks, significantly improving response utilization, training efficiency, and reducing token clipping compared to baselines like GRPO and DAPO.

---

[Instruction Agent: Enhancing Agent with Expert Demonstration](http://arxiv.org/abs/2509.07098)

- Instruction Agent: introduces a GUI agent leveraging expert demonstrations to generate step-by-step instructions, which are then executed by an Actor model incorporating verification and backtracking for robustness.
- The framework, composed of an Instructor (Recorder, Instruction Generator) and an Actor (Grounder, Executor, Verifier, Backtracker), automates complex GUI tasks by strictly following demonstrated trajectories.
- Utilizing LLMs (GPT-40) for planning, verification, and execution, and UI-Tars 1.5 for grounding, the agent achieves a 60% success rate on tasks previously failed by top-ranked GUI agents.

---

#### 7th September 2025


[Language-Native, Lightly Structured Databases for Large-Language-Model-Driven Composite Materials Research](http://arxiv.org/abs/2509.06093)

- LLM-DCMF (Large-Language-Model-Driven Composite Materials Research Framework): introduces a language-native, lightly structured database and LLM-based data management system, with Light-Structured Text Generation (processes raw scientific literature into modular, lightly structured text using LLM-guided prompts), Heterogeneous Database (stores and organizes both lightly structured text units and fully structured data derived from them), Hybrid Search (retrieves relevant information from the database using multiple methods), and Application Layer (RAG & Agentic Workflows) (utilizes retrieved information for reasoning, generation, and iterative design), where the system transforms raw scientific literature into a queryable, heterogeneous database to accelerate materials discovery.
- The framework captures lightly structured information from papers across preparation, characterization, theory/computation, and mechanistic reasoning, organizing records in a heterogeneous database for composite retrieval with semantics, keywords, and value filters.
- This system synthesizes literature into accurate, verifiable, and expert-style guidance, enabling high-fidelity Retrieval-Augmented Generation (RAG) and tool-augmented agents to interleave retrieval with reasoning for actionable SOPs.

---


[Proof2Silicon: Prompt Repair for Verified Code and Hardware Generation via Reinforcement Learning](http://arxiv.org/abs/2509.06239)

- Proof2Silicon: introduces an end-to-end synthesis framework that leverages PREFACE's RL-driven prompt optimization core to guide a frozen LLM in generating formally verified Dafny code, which is then translated through a PyLog-based pipeline and Vivado HLS into synthesizable RTL for FPGA hardware.
- The framework integrates a verifier-guided RL agent (SLM) to iteratively refine prompts based on formal verification feedback, ensuring Dafny code correctness without costly LLM fine-tuning.
- This pipeline bridges natural language specifications with silicon realization, enabling automated, correctness-by-construction hardware synthesis for safety-critical domains.

---

[PillagerBench: Benchmarking LLM-Based Agents in Competitive Minecraft Team Environments](http://arxiv.org/abs/2509.06235)

- TactiCrafter: introduces an LLM-based multi-agent system for competitive Minecraft environments, featuring a Tactics Module (generates high-level strategies), a Causal Model (learns causal relationships), an Opponent Model (infers enemy strategies), and Base Agents (execute actions, self-reflect).
- This system facilitates teamwork through human-readable tactics, learns causal dependencies from gameplay, and adapts to opponent strategies through repeated self-play.
- TactiCrafter is evaluated on PillagerBench, a novel benchmark for real-time competitive team-vs-team Minecraft scenarios, demonstrating superior performance over baselines and adaptive learning.

---


[Generating Individual Travel Diaries Using Large Language Models Informed by Census and Land-Use Data](http://arxiv.org/abs/2509.09710)

- LLM Diary Generation Workflow: introduces a novel agent-based framework for generating individual travel diaries using LLMs, leveraging components like a Block Group Data Loader, Persona Synthesis Stage, Prompt Engineering Stage, LLM Call (llama3), and Generation & Parsing Stage to produce realistic travel patterns.
- This two-stage framework first stochastically synthesizes demographically consistent personas from open-source census and land-use data, then uses a comprehensive prompt to direct the LLM (Llama 3) to generate a full day's structured travel diary.
- The framework employs a rigorous one-to-cohort validation strategy, comparing LLM-generated diaries against real-world survey data and classical models, demonstrating the LLM's zero-shot viability and superior semantic understanding for travel behavior.

---

[Modeling shopper interest broadness with entropy-driven dialogue policy in the context of arbitrarily large product catalogs](http://arxiv.org/abs/2509.06185)

- Entropy-Driven Dialogue Policy (EDDP): introduces a novel method for conversational recommender systems to balance exploration and exploitation by dynamically routing dialogue based on the entropy of retrieval score distributions.
- This policy quantifies user intent broadness via a broadness score derived from re-ranked product retrieval scores, enabling the system to ask clarifying questions for high-entropy (ambiguous) queries or make direct recommendations for low-entropy (precise) queries.
- Integrated within a multi-skill e-commerce AI agent, the EDDP allows an LLM-driven system to adapt to arbitrarily large product catalogs in real-time without context window bloat, enhancing shopper engagement.

---

[From Digital Distrust to Codified Honesty: Experimental Evidence on Generative AI in Credence Goods Markets](http://arxiv.org/abs/2509.06069)

- Experimental Design for Credence Goods Markets: introduces a series of one-shot experiments to quantify the behavioral, welfare, and distribution consequences of LLMs in expert service markets, varying market interaction types, institutional environments, LLM objective functions, training regimes, and transparency rules.
- The study finds that Human-Human markets generally achieve higher efficiency than AI-AI and Human-AI markets due to pro-social expert preferences and higher consumer trust, while LLM experts often earn higher surplus at the expense of consumers.
- Crucially, allowing human experts to delegate to LLMs and codify their LLM's social preferences, especially when transparent to consumers, significantly increases market efficiency and reduces fraud, potentially outperforming Human-Human markets under transparency rules.

---

[POLICYEVOLVE: EVOLVING PROGRAMMATIC POLICIES BY LLMS FOR MULTI-PLAYER GAMES VIA POPULATION-BASED TRAINING](http://arxiv.org/abs/2509.06053)

- PolicyEvolve: introduces a general framework for generating programmatic policies in multi-player games, with Global Pool (preserves elite policies), Local Pool (stores temporary policies), Policy Planner (generates/refines policy code), and Trajectory Critic (analyzes policy performance).
- The framework leverages LLMs for policy generation and iterative refinement, using population-based training to evolve robust and interpretable rule-based policies.
- PolicyEvolve significantly reduces reliance on manually crafted policy code, achieving high-performance policies with minimal environmental interactions and demonstrating consistent strategy evolution.

---

[Code2MCP: A Multi-Agent Framework for Automated Transformation of Code Repositories into Model Context Protocol Services](http://arxiv.org/abs/2509.05941)

- Code2MCP: introduces an automated, multi-agent framework for transforming GitHub repositories into Model Context Protocol (MCP) services, employing LLM-powered agents for code analysis, generation, review, and finalization.
- The framework operates via a state graph orchestrating a multi-stage workflow, featuring a closed-loop "Run-Review-Fix" cycle for autonomous debugging and repair of generated code.
- Code2MCP significantly accelerates the MCP ecosystem by systematically converting open-source code into deployable services and comprehensive technical documentation with minimal human intervention.

---

[MapAgent: A Hierarchical Agent for Geospatial Reasoning with Dynamic Map Tool Integration](http://arxiv.org/abs/2509.05933)

- MapAgent: introduces a hierarchical multi-agent plug-and-play framework for map-integrated geospatial reasoning, featuring a Planner Agent, Module Inventory (Visual Place Recognizer, Map Service, Sequencer, Solution Generator, Answer Generator), and a Map-Tool Agent with specialized map tools (Nearby, Trip, Route, PlaceInfo).
- This framework decouples high-level planning from low-level tool execution, allowing the Planner Agent to decompose complex queries into subgoals routed to specialized modules, while the Map-Tool Agent adaptively orchestrates map APIs.
- MapAgent significantly outperforms existing tool-augmented and agentic baselines across diverse geospatial benchmarks, demonstrating robust performance and generalizability in real-world map-integrated reasoning tasks.

---

[Let's Roleplay: Examining LLM Alignment in Collaborative Dialogues](http://arxiv.org/abs/2509.05882)

- FAAF (Frictional Agent Alignment Framework): introduces a novel counterfactual evaluation framework to examine LLM alignment in collaborative dialogues, utilizing a roleplay methodology with a Friction Agent (πF) and Collaborator Agent (πC) to study the impact of friction interventions on common ground and task outcomes.
- The framework employs an Oracle Agent (O) for data generation, simulating frictive states and interventions, and evaluates performance using metrics like Common Ground (CG) size, solution accuracy, and a Reward Model (RM) for intervention quality.
- This approach addresses the suboptimality risks in multi-party interactions by explicitly modeling action modifications within a Modified-Action MDP (MAMDP) and demonstrates that friction-aware alignment improves both common ground convergence and task correctness.

---

#### 6th September 2025

[Chatbot To Help Patients Understand Their Health](http://arxiv.org/abs/2509.05818)

- NoteAid-Chatbot (Learning as Conversation Framework): introduces a conversational AI system for patient education, built on a multi-agent LLM and reinforcement learning setup, which leverages synthetic data for supervised fine-tuning and PPO-based alignment with rewards derived from simulated patient comprehension assessments.
- This framework automates training without human-labeled data, enabling the development of a lightweight, domain-specific chatbot capable of multi-turn interactions and diverse educational strategies.
- Evaluations, including a Turing test, demonstrate the system's ability to surpass non-expert human educators in patient understanding and exhibit key emergent behaviors like clarity and structured dialogue.

---

[DRF: LLM-AGENT Dynamic Reputation Filtering Framework](http://arxiv.org/abs/2509.05764)

- DRF (LLM-AGENT Dynamic Reputation Filtering Framework): introduces a multi-agent system that constructs an interactive rating network, designs a reputation scoring mechanism, and integrates an Upper Confidence Bound-based strategy to quantify agent performance and enhance selection efficiency.
- The framework utilizes a core agent for decision-making and control, while task agents execute and evaluate subtasks, with their performance and credibility dynamically assessed through a k-layer rating network and iterative reputation updates.
- DRF significantly improves task completion quality and collaboration efficiency in complex tasks like logical reasoning and code generation by prioritizing high-reputation, low-cost LLM agents.

---

[Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System](http://arxiv.org/abs/2509.05755)

- TEW (TIP Exploitation Workflow): introduces a systematic approach to identify and exploit vulnerabilities in LLM-based agentic systems, comprising Prompt Stealing (extracts prompt components), TIP Vulnerabilities Analysis (reconstructs TIP and identifies weaknesses), and TIP Hijacking (exploits tool invocation).
- This framework demonstrates how manipulating Tool Invocation Prompts (TIPs) can lead to severe security risks such as remote code execution (RCE) and denial of service (DoS) in LLM-based agentic systems.
- The research highlights the critical need for robust security measures in TIPs, as current defense mechanisms, including guard models and self-reflection, are often insufficient against sophisticated attacks.

---

[A Composable Agentic System for Automated Visual Data Reporting](http://arxiv.org/abs/2509.05721)

- Composable Agentic System: introduces a multi-stage, hybrid agentic architecture for automated visual data reporting, leveraging LLM-driven agents for reasoning and programmatic/rule-based components for deterministic logic, producing interactive Observable 2.0 reports and executable Marimo notebooks.
- The system's workflow is coordinated by an Orchestrator and includes distinct phases for data understanding, analysis and materialization, visualization, and reporting, with components like Field Refiner, Dataset Profiler, Insight Planner, Dataset Visualizer, and Report Narrator.
- This architecture supports a Human-AI Partnership Model by providing auditable, steerable, and granular outputs, externalizing critical logic to deterministic modules like the Draco visualization system, and enabling deep traceability for analysts and interactive exploration for readers.

---

[Orchestrator: Active Inference for Multi-Agent Systems in Long-Horizon Tasks](http://arxiv.org/abs/2509.05651)

- Orchestrator: introduces a novel multi-agent system (MAS) framework that leverages attention-inspired self-emergent coordination and reflective benchmarking to optimize global task performance, featuring a modular cell architecture with planning, execution, and orchestration nodes.
- The framework integrates a monitoring mechanism to track agent-environment dynamics and agent-to-agent interaction, using active inference benchmarks to optimize system behavior and mitigate partial observability.
- Orchestrator dynamically adjusts LLM agents' internal policies and prompt designs based on performance metrics, guiding them away from local solution minima towards global task objectives in complex, long-horizon environments.

---

[ProfilingAgent: Profiling-Guided Agentic Reasoning for Adaptive Model Optimization](http://arxiv.org/abs/2509.05584)

- ProfilingAgent: introduces a profiling-guided agentic approach that automates model compression through structured pruning and post-training dynamic quantization, leveraging a multi-agent system including an Acquisition Agent, Input Shape Resolver Agent, Profiling Agent, Analysis Agent, Pruning Agent, Quantization Agent, Evaluation Agent, and Iterative Pruning Agent.
- This modular pipeline utilizes LLM-guided agents, specifically the Input Shape Resolver Agent and Analysis Agent, to interpret static and dynamic profiling signals, generating architecture-specific compression strategies and layer-wise decisions tailored to performance bottlenecks.
- The system adaptively refines pruning decisions through iterative feedback, balancing accuracy and latency, and demonstrates superior performance compared to heuristic baselines across various vision models.

---

[FAIR RISK OPTIMIZATION OF DISTRIBUTED SYSTEMS](http://arxiv.org/abs/2509.05737)

- FRODS (Fair Risk Optimization of Distributed Systems): introduces a framework for assessing and optimizing the total risk of complex distributed systems, utilizing systemic risk measures and a decomposition method to solve a two-stage stochastic programming problem for fair risk allocation.
- The framework addresses challenges like non-additive risk, confidential information, and fair risk allocation by evaluating individual agent risks and aggregating them into a total system risk measure.
- The decomposition method enables distributed decision-making, allowing agents to operate autonomously with minimal information exchange while contributing to overall system risk minimization.

---

[ArGen: Auto-Regulation of Generative AI via GRPO and Policy-as-Code](http://arxiv.org/abs/2509.07006)

- ArGen (Auto-Regulation of Generative AI systems): introduces a novel framework for aligning LLMs with complex, configurable, machine-readable rules by synthesizing principle-based automated reward scoring, Group Relative Policy Optimisation (GRPO), and an OPA-inspired governance layer.
- This framework operationalizes nuanced ethical principles and regulatory compliance standards into programmable reward functions and explicit policies, enabling continuous auto-regulation and verifiable compliance.
- ArGen's architecture allows for live policy hot-swapping without model retraining, providing a transparent, auditable, and adaptable path to governable AI systems in diverse global contexts.

---

#### 5th September 2025

[LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation](http://arxiv.org/abs/2509.05263)

- LatticeWorld: introduces a multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation, with User Inputs (textual/visual instructions), LLML (Large Language Model for Layout Generation) (generates symbolic layout), Visual Encoder (Φ) (extracts visual features), Projection Module (Proj) (maps visual features), LLMc (Large Language Model for Environmental Configuration) (generates environment configurations), Decoder (ΨL) (translates symbolic layout), Configuration Translator (ΨC) (interprets environment configurations), Rendering Engine (Unreal Engine 5) (creates dynamic 3D world), and PCG Pipeline (Procedural Content Generation Pipeline) (automates asset creation), where it generates large-scale 3D interactive worlds with dynamic agents and high-fidelity physics simulation from multimodal user instructions.
- The framework leverages lightweight LLMs (LLaMA-2-7B) and an industry-grade rendering engine (Unreal Engine 5) to process textual descriptions and visual instructions (e.g., height maps) into a symbolic layout representation and environmental configurations.
- LatticeWorld achieves over a 90x increase in industrial production efficiency for 3D environment generation compared to traditional manual methods, while maintaining high creative quality and enabling real-time multi-agent interaction and simulation.

---

[TRIADIC FUSION OF COGNITIVE, FUNCTIONAL, AND CAUSAL DIMENSIONS FOR EXPLAINABLE LLMS: THE TAXAL FRAMEWORK](http://arxiv.org/abs/2509.05199v1)

- TAXAL (Triadic Alignment for eXplainability in Agentic LLMs): introduces a triadic fusion framework for explainable LLMs, integrating Cognitive (user understanding, intelligibility), Functional (practical utility, compliance), and Causal (faithful reasoning, intervention) dimensions to provide a unified, role-sensitive foundation for designing, evaluating, and deploying explanations in diverse sociotechnical settings.
- The framework aims to enhance Trust (user confidence), Contestability (challenge outputs), and Accountability (auditability, compliance) in agentic LLMs by aligning explanation strategies with human cognition, institutional expectations, and enabling users to challenge outputs and audit traces.
- TAXAL provides a step-by-step guide for identifying Stakeholder Role Identification (define audience), selecting Relevant Dimension Selection (choose method), choosing Explanation Strategy Selection (choose method), balancing Trade-Off Balancing (optimize dimensions), and iteratively refining explanations through Contextual Iteration (evaluate, refine), as demonstrated through cross-domain case studies.

---

[AI Agents for Web Testing: A Case Study in the Wild](http://arxiv.org/abs/2509.05197)

- WebProber: introduces an AI agent-based web testing framework that autonomously explores websites, simulates user interactions, identifies bugs, and generates human-readable reports, comprising a prompt generation module (generates testing prompts), an interaction module (simulates user experience), and a report generation module (analyzes interaction history), leveraging a Visual Language Model (VLM) and a bug database.
- The framework interacts directly with visual webpages using VLMs to simulate human-like behavior, enabling the detection of contextual usability issues and bugs often overlooked by traditional web testing tools.
- The system's case study on 120 academic personal websites demonstrated its ability to uncover subtle, human-centric problems, highlighting the potential of agent-based testing while also revealing challenges in agent-browser interaction and bug coverage.

---

[Shared Autonomy through LLMs and Reinforcement Learning for Applications to Ship Hull Inspections](http://arxiv.org/abs/2509.05042)

- Shared Autonomy Framework: introduces a multi-layered architecture for ship hull inspections, integrating a Supervisor (LLM), a Mission Manager (Behavior Trees), and a Multi-Agent Execution Layer (DRL-trained Agents) to enable human-robot collaboration.
- The framework allows an Operator to specify high-level goals via natural language, which are translated into structured tasks managed by Behavior Trees, while DRL-trained agents execute tasks and adapt their behavior in a leader-follower configuration.
- This system aims to reduce operator cognitive load, enhance transparency, and improve adaptive behavior alignment with human intent in complex maritime environments.

---

[LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration](http://arxiv.org/abs/2509.04993)

- Dual-Loop MAS (Dual-Loop Multi-Agent System with Parallel Terminal-Edge Collaboration): introduces an LLM-enabled multi-agent system for 6G networks, featuring a Global Agent, Sub-Agent, Perception Module, Planning Module (Outer Loop / Inner Loop), LLMCompiler, Memory Module, Scheduling Module, Tool Execution, Database, Self-Evolution, and Knowledge Management, designed to enhance task planning and execution efficiency through dual-loop edge-terminal collaboration.
- The framework utilizes an outer loop for global task decomposition by a Global Agent and an inner loop where Sub-Agents, guided by LLMCompiler, perform parallel sub-task execution and replanning with tool calling and offloading strategies.
- It integrates a comprehensive Memory Module for context and knowledge, a Perception Module for multimodal information and intention recognition, and a Scheduling Module for efficient resource allocation, enabling self-evolution and robust knowledge management in 6G environments.

---

[Internet 3.0: Architecture for a Web-of-Agents with it's Algorithm for Ranking Agents](http://arxiv.org/abs/2509.04979)

- DOVIS (Discovery, Orchestration, Verification, Incentives, Semantics) and AgentRank-UC: introduces a framework for the Agentic Web, with DOVIS as a five-layer operational protocol for collecting privacy-preserving telemetry and AgentRank-UC as a dynamic, trust-aware algorithm for ranking agents based on usage and competence.
- The DOVIS protocol includes Discovery, Orchestration, Verification, Incentives, and Semantics layers, enabling verifiable and incentivized telemetry collection through the OAT-Lite schema.
- AgentRank-UC combines usage and competence signals into a unified ranking, ensuring scalability, trustworthiness, and resilience against manipulation in open agent ecosystems.

---

[Towards Ontology-Based Descriptions of Conversations with Qualitatively-Defined Concepts](http://arxiv.org/abs/2509.04926)

- Ontology-Based Conversational Control Framework: introduces an approach for controlling conversational generation by quantitatively defining qualitatively-described conversation aspects, with Qualitatively-defined conversational features (high-level concepts), Pre-defined descriptors (measurable text properties), Annotated dataset (labeled text examples), Classifier (maps descriptors to concepts), Subclass membership rules (defines concept boundaries), Description logic (formal rule representation), Ontology (structured knowledge base), Reasoning step (infers new knowledge), LLM fine-tuning step (adapts LLM behavior), LLM (generates controlled text), and LoRA adapters (efficiently tunes LLM).
- The framework leverages linguistic descriptors and a Decision Tree Classifier to establish structured definitions of conversational features, which are then integrated into an ontological framework for consistency and transparency.
- This methodology enables the fine-tuning of LLMs, such as Llama3-8B-Instruct, to generate content that adheres to specific ontological concepts, demonstrated through a Proficiency-Level Control use-case.

---

[OSC: Cognitive Orchestration through Dynamic Knowledge Alignment in Multi-Agent LLM Collaboration](http://arxiv.org/abs/2509.04876)

- OSC (Orchestrating Cognitive Synergy): introduces a knowledge-aware adaptive collaboration framework designed to enhance cognitive synergy in multi-agent LLM systems, with its Expert Pool (collection of LLM agents), Query (initial task/question), Collaborator Knowledge Model (dynamically tracks collaborators' cognitive states), Cognitive Gap Analysis Module (identifies discrepancies between agents), Communication Policy (shapes communication behavior), Linguistic Realization Engine (translates abstract action to language), Dialogue History (record of past interactions), Internal State (agent's own cognitive state), Update Mechanism (updates CKM based on dialogue), Aggregator Module (combines individual responses), Reward Function (guides learning), Task Performance Reward (feedback for task success), Communication Cost (penalty for message length), Intrinsic Shaped Reward (guides collaborative behaviors), and Reinforcement Learning (optimizes communication policy).
- The framework operates as an intermediate layer between expert selection and aggregation, enabling agents to dynamically perceive collaborators' cognitive states, analyze cognitive gaps, and adapt communication behaviors using learned strategies.
- This dynamic approach transforms parallel-working LLM agents into a deeply collaborative cognitive team, significantly improving task performance and communication efficiency in complex reasoning and problem-solving benchmarks.

---

[VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving](http://arxiv.org/abs/2509.04827)

- VoltanaLLM: introduces an SLO-aware, energy-efficient LLM serving system built on prefill/decode (P/D) disaggregation, with EcoFreq (dynamically adjusts GPU frequency for prefill/decode phases), EcoRoute (routes decode requests to optimize energy), EcoPred (predicts TTFT/ITL latency), Prefill Instances (process initial LLM input), Decoding Instances (generate subsequent LLM tokens), New Requests (incoming user queries), Generated Tokens (output streamed to user), Load Metrics (real-time system performance data), TTFT prediction (Time-To-First-Token estimation), and ITL prediction (Inter-Token Latency estimation).
- This framework co-designs frequency scaling and request routing, leveraging decoupled execution to enable fine-grained, phase-specific control for LLM inference.
- VoltanaLLM achieves up to 36.3% energy savings while maintaining near-perfect SLO attainment rates across various LLMs and real-world datasets.

---

[Fishing for Answers: Exploring One-shot vs. Iterative Retrieval Strategies for Retrieval Augmented Generation](http://arxiv.org/abs/2509.04820)

- One-SHOT and Iterative Retrieval Strategies: introduces two RAG approaches, One-SHOT for token-constrained retrieval with chunk filtering and cropping, and an iterative agentic framework with an LRM for multi-turn query refinement and context management.
- The paper addresses common RAG bottlenecks, such as missing crucial "golden chunks" in top-k retrieval and issues like query drift and retrieval laziness in complex QA tasks.
- These strategies offer practical solutions for improving evidence coverage and answer quality in legal and regulatory domains, particularly with heterogeneous government documents.

---

[TalkToAgent: A Human-centric Explanation of Reinforcement Learning Agents with Large Language Models](http://arxiv.org/abs/2509.04809)

- TalkToAgent: introduces a multi-agent LLM framework for human-centric explanation of RL agents, featuring Coordinator, Explainer, Coder, Evaluator, and Debugger agents, along with predefined XRL tools and generated codes.
- The framework interprets natural language queries, maps them to relevant XRL tools, and provides multimodal explanations including figures and domain-aware textual interpretations.
- TalkToAgent enhances counterfactual explanations by generating alternative scenarios from qualitative behavioral descriptions or new rule-based policies, ensuring high accuracy and minimizing failures through agent interactions.

---

[Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs](http://arxiv.org/abs/2509.04802)

- AgentSeer: introduces an observability-based evaluation framework that decomposes agentic executions into granular actions and components, enabling systematic agentic-situational assessment of LLMs.
- The framework addresses critical gaps in current LLM safety evaluation by transforming opaque agentic executions into structured, analyzable representations through a knowledge graph.
- It empirically validates that agentic deployments have distinct "agentic-only" vulnerabilities, often invisible to traditional model-level testing, emphasizing the need for agentic-situation evaluation.

---

[Personality as a Probe for LLM Evaluation: Method Trade-offs and Downstream Effects](http://arxiv.org/abs/2509.04794)

- Unified Evaluation Framework: introduces a systematic study for LLM personality control, comparing In-Context Learning, Parameter-Efficient Fine-Tuning, and Mechanistic Steering methods, utilizing a Contrastive Dataset Generation, Trait Purification Techniques, and a Three-Level Stability Framework for comprehensive evaluation on Gemma-2 and LLaMA-3 models.
- The framework assesses method trade-offs in personality alignment, task performance, and demographic bias across MMLU, GAIA, and BBQ benchmarks.
- It provides practical guidance for selecting personality manipulation methods based on alignment strength, computational requirements, and performance preservation.

---

[SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching](http://arxiv.org/abs/2509.04752)

- SePA (Search-enhanced Predictive AI Agent): introduces a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance, featuring a User Interface (web application for interaction), Data Ingestion Module (uploads and processes raw data), Data Preprocessing & Feature Engineering Module (transforms raw data), Predictive Modeling Module (forecasts health risks), Toolbox Module (provides analytical capabilities), Conversational LLM Agent (manages user dialogue), and a Web Retrieval Pipeline (grounds LLM advice).
- SePA forecasts daily stress, soreness, and injury risk from wearable sensor data using a two-tiered predictive modeling strategy, employing a generalized XGBoost model for new users and personalized deep neural networks for engaged users.
- The system's web-retrieval pipeline dynamically rewrites search queries based on ML predictions and grounds advice in a curated whitelist of trusted sources, ensuring contextual relevance, verifiability, and transparency.

---

[Cloning a Conversational Voice AI Agent from Call Recording Datasets for Telesales](http://arxiv.org/abs/2509.04871)

- Conversational Voice AI Agent Cloning System: introduces a general methodology for cloning a conversational voice AI agent from call recordings, leveraging a cloning system to extract behavioral patterns and an inference system to deploy the agent in live calls.
- The cloning pipeline transforms raw call recordings into a structured Agent Playbook, which defines the agent's persona, knowledge, and conversational style through prompt engineering.
- The inference system deploys the cloned agent in real-time using a Voice LLM Core, which processes user speech and generates synthetic agent responses for fluid, end-to-end spoken interaction.

---

[LEARNING TOOL-AWARE ADAPTIVE COMPLIANT CONTROL FOR AUTONOMOUS REGOLITH EXCAVATION](http://arxiv.org/abs/2509.05475)

- LTACC (Learning Tool-Aware Adaptive Compliant Control): introduces a learning-based framework where a Model-based RL Agent learns adaptive compliance within a Space Robotics Bench (SRB) simulation, leveraging SimForge Engine and Procedural Generation to create diverse scenarios with High-fidelity Particle Physics (XPBD) for autonomous regolith excavation.
- The framework utilizes an Operational Space Control (OSC) Controller to dynamically modulate robot stiffness and damping, enabling the RL Policy to adapt to various Excavation Tools and Regolith properties.
- The system integrates Vision and Proprioception feedback, processed by an Encoder and a Recurrent World Model, to infer hidden states and achieve generalizable, tool-aware excavation skills across Parallelized Simulation environments.

---

#### 4th September 2025

[Psychologically Enhanced AI Agents](http://arxiv.org/abs/2509.04343)

- MiT (MBTI-in-Thoughts): introduces a framework for enhancing LLM agents through psychologically grounded personality conditioning, with LLM Agents (core processing units), Psychological Profiles (personality conditioning via prompts), 16Personalities Test (external validation tool), Majority Voting Protocol (isolated reasoning mechanism), Interactive Communication Protocol (decentralized dialogue mechanism), Interactive Communication With Self-Reflection Protocol (dialogue with private memory), Self-Reflection (private deliberation buffer), Blackboard (shared communication memory), and LLM as Judge (final decision maker).
- The framework primes LLM agents with distinct MBTI personality archetypes via prompt engineering, enabling control over behavior along cognitive and affective axes, and supports structured multi-agent communication protocols.
- MiT demonstrates that personality priming induces consistent behavioral biases, improves cooperation and reasoning quality through self-reflection, and generalizes to other psychological models without fine-tuning.

---

[EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation](http://arxiv.org/abs/2509.04310)

- EvoEmo framework: introduces an evolutionary reinforcement learning framework for optimizing dynamic emotional expression in multi-turn negotiations, with Negotiation Setup, Seller LLM Agent, Buyer LLM Agent, Product Description, Seller Prompts, Buyer Prompts, EvoEmo Optimization Module, Emotion States, Emotional Policy (πω), Policy Population, Population Generation, Selection Operator, Crossover Operator, Mutation Operator, Negotiation Simulation Module, Simulated Seller Agents, Simulated Buyer Agents, Mediator Agent, Evaluation Module, Reward Function (R(S)), Optimal Policy (πω*), and Termination Conditions, where it evolves high-reward emotion policies using population-based genetic optimization across diverse negotiation scenarios.
- The framework models emotional state transitions as a Markov Decision Process and iteratively refines policies based on rewards achieved during negotiations, combining evolutionary exploration with reinforcement learning principles.
- EvoEmo consistently outperforms vanilla and fixed-emotion baselines, achieving higher success rates, efficiency, and buyer savings by enabling LLM agents to adaptively express emotions.

---

[Are LLM Agents the New RPA? A Comparative Study with RPA Across Enterprise Workflows](http://arxiv.org/abs/2509.04198v1)

- AACU (Agentic Automation with Computer Use): introduces a comparative study of LLM agents and traditional Robotic Process Automation (RPA) across enterprise workflows, including LLM Agent (intelligent core), Computer Use Capability (interacts digital systems), User Interface (UI) Interaction (mimics human actions), Natural Language Input (task instructions), Software Applications (automation targets), and Execution Environment (agent operational context), to evaluate their performance, reliability, and development effort.
- The study found that while RPA generally outperforms AACU in execution speed and reliability for repetitive, stable tasks, AACU significantly reduces development time and offers greater flexibility for dynamic interfaces.
- Despite current limitations in production readiness, AACU shows promise for rapid prototyping and lightweight automation, suggesting future research into hybrid RPA-AACU architectures and multi-agent orchestration.

---

[MAGneT: Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions](http://arxiv.org/abs/2509.04183)

- MAGneT (Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions): introduces a novel multi-agent framework for synthetic psychological counseling session generation that decomposes counselor response generation into coordinated sub-tasks, utilizing a Counselor Agent (generates counselor responses) with a CBT Agent (produces structured treatment plan), a Technique Agent (selects therapeutic techniques), five Specialized Response Agents (Reflection Agent (mirrors client expressions), Questioning Agent (explores client feelings), Solutions Agent (provides actionable solutions), Normalizing Agent (validates client experiences), Psycho-ed Agent (offers therapeutic information)), and a Response Generator (synthesizes final utterance), alongside a Client Agent (simulates client behavior) with an Intake Form (client profile, issues) and Attitudes (client emotional stance).
- This framework employs specialized LLM agents, each grounded in core therapeutic techniques, and coordinates them via a dynamic technique selector and a CBT-based planning agent to generate psychologically grounded and nuanced counseling dialogues.
- The system further simulates realistic client behavior using detailed profiles and attitude modeling, and integrates a unified evaluation framework for comprehensive assessment of generated counseling data quality and diversity.

---

[TAGAL: Tabular Data Generation using Agentic LLM Methods](http://arxiv.org/abs/2509.04152)

- TAGAL (Tabular Data Generation using Agentic LLM Methods): introduces a collection of training-free methods for generating synthetic tabular data using an agentic workflow, which includes a Generation LLM (generates tabular data), Feedback LLM (criticizes data, provides recommendations), Initial Prompt (guides initial data generation), Analysis Prompt (guides feedback LLM evaluation), Feedback Prompt (incorporates feedback for generation), Generated Data (synthetic tabular examples), Feedback (LLM-generated data critique/recommendations), Summary LLM (summarizes conversation history), and Refined Prompt (summarized prompt for generation), to iteratively refine generated data quality through feedback loops.
- The framework comprises three distinct methods—SynthLoop, ReducedLoop, and Prompt-Refine—each employing iterative feedback mechanisms to enhance data quality and utility, with Prompt-Refine specifically using a Summary LLM to create a Refined Prompt for efficient generation.
- It demonstrates performance comparable to state-of-the-art training-required models and often outperforms other training-free approaches, showcasing the potential of agentic LLM workflows for high-quality tabular data generation, even with limited original data.

---

[Towards Stable and Personalised Profiles for Lexical Alignment in Spoken Human-Agent Dialogue](http://arxiv.org/abs/2509.04104)

- Lexical Profile Construction and Evaluation: introduces a method for constructing stable, personalised lexical profiles from transcribed spoken data, leveraging POS categories and word-based n-grams, processed by SpaCy's nl_core_news_lg pipeline, and evaluated with recall, coverage, and cosine similarity metrics, to form a basis for lexical alignment in human-agent dialogue.
- The study determined that profiles built from 10 minutes of speech, including 5 items for adjectives and conjunctions and 10 items for adverbs, nouns, pronouns, and verbs each, offered the best balance between performance and data efficiency.
- These stable and representative lexical profiles are crucial for developing inclusive lexical alignment strategies in conversational agents, particularly for users with limited real-time input, such as individuals with dementia, by providing a robust basis for LLM-based response generation.

---

[COT-SPACE: A THEORETICAL FRAMEWORK FOR INTERNAL SLOW-THINKING VIA REINFORCEMENT LEARNING](http://arxiv.org/abs/2509.04027)

- CoT-Space (Chain-of-Thought Space): introduces a novel theoretical framework that recasts LLM reasoning from a discrete token-prediction task into an optimization process within a continuous, reasoning-level semantic space, including a policy model, reasoning-level states, a reasoning loss landscape, CoT length, solution minimums, noise scale, generalization error, and empirical loss.
- This framework models the LLM reasoning process as a trajectory towards a solution minimum in a reasoning loss landscape, providing a more intuitive and powerful lens for theoretical analysis of internal slow-thinking via RL.
- The analysis demonstrates that an optimal CoT length exists, balancing underfitting (due to insufficient reasoning depth) and overfitting (due to increased model complexity and noise sensitivity), analogous to an optimal learning rate in classical ML.

---

[Meta-Policy Reflexion: Reusable Reflective Memory and Rule Admissibility for Resource-Efficient LLM Agents](http://arxiv.org/abs/2509.03990)

- Meta-Policy Reflexion (MPR): introduces a hybrid framework that consolidates LLM-generated reflections into a structured Meta-Policy Memory (MPM), which guides an LLM Base Policy through soft memory-conditioned decoding and hard admissibility checks (HAC) for training-free self-improvement.
- MPR externalizes reusable corrective knowledge as predicate-like rules, enforcing domain constraints to reduce unsafe actions without modifying LLM parameters, thereby retaining adaptability.
- The framework demonstrates consistent gains in execution accuracy and robustness on the AlfWorld benchmark compared to Reflexion baselines, with HAC further improving stability.

---

[World Model Implanting for Test-time Adaptation of Embodied Agents](http://arxiv.org/abs/2509.03956v1)

- WorMI (World Model Implanting): introduces a framework for embodied agents that combines LLMs' reasoning capabilities with independently learned, domain-specific world models through test-time composition, including a Reasoning Model (LLM), Pre-trained World Models, Prototype-based World Model Retrieval, Object Detection Model, Embedding Model, Prototypes, World-wise Compound Attention, Linear Projection Layer, World-level Cross-attention, Reasoning-level Cross-attention, and an Implant/Remove Mechanism.
- The framework's Prototype-based World Model Retrieval method efficiently selects relevant world models using trajectory-based abstract representation matching, while the World-wise Compound Attention method integrates and aligns knowledge from retrieved models with the reasoning model.
- This dual-stage design enables flexible, test-time fusion of domain-specific knowledge, enhancing adaptability to unseen domains and maintaining cross-domain adaptability for embodied agents.

---

[VoxRole: A Comprehensive Benchmark for Evaluating Speech-Based Role-Playing Agents](http://arxiv.org/abs/2509.03940)

- VoxRole: introduces a comprehensive benchmark for evaluating speech-based Role-Playing Conversational Agents (RPCAs), built using a Spoken Dialogue Extraction Pipeline (Extracts movie dialogues) and a Persona Distillation Pipeline (Builds character profiles), and evaluated with a multi-dimensional Evaluation Framework (Assesses model performance).
- The Spoken Dialogue Extraction Pipeline automatically extracts character-rich spoken dialogues from movies by aligning audio with scripts and curating semantically validated segments using components like FFmpeg, Resemble, Whisper-large-v3, Wav2Vec2.0, and MPNet.
- The Persona Distillation Pipeline leverages an LLM and an Acoustic Feature Extraction Module to systematically construct multi-dimensional character profiles, encompassing personality, linguistic style, relationships, and acoustic characteristics, which are then used to generate role-playing prompts for evaluation by an LLM Judge.

---

[MobileRAG: Enhancing Mobile Agent with Retrieval-Augmented Generation](http://arxiv.org/abs/2509.03891)

- MobileRAG (Retrieval-Augmented Generation): introduces a mobile agent framework, with InterRAG (external knowledge retrieval), LocalRAG (local app management), and MemRAG (historical operation memory), designed to enhance mobile agents for accurate user query identification and efficient complex mobile task execution.
- The framework addresses limitations of current LLM-based mobile agents, such as over-reliance on LLM comprehension, limited external interaction, and absence of effective memory.
- MobileRAG improves task completion, reduces operational steps, and enhances adaptability by leveraging external knowledge and learning from past successful operations.

---

[FaMA: LLM-Empowered Agentic Assistant for Consumer-to-Consumer Marketplace](http://arxiv.org/abs/2509.03890)

- FaMA (Facebook Marketplace Assistant): introduces an LLM-powered agentic assistant for C2C marketplaces, integrating a Llama 4 Maverick LLM as its core reasoning engine, a memory module (Scratchpad, Dialog History, Listings Information), and a suite of specialized Marketplace Tools (Listing Operation, Inventory Search, Messaging) along with RAG and a Knowledge Base.
- This conversational agent simplifies user experience by interpreting natural language commands to automate high-friction workflows for both buyers and sellers, including listing management, bulk messaging, and efficient product discovery.
- FaMA achieves a 98% task success rate and enables up to a 2x speedup in interaction time, providing a lightweight and accessible alternative to traditional app interfaces.

---

[Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning](http://arxiv.org/abs/2509.03817)

- MPDF (Meta-Policy Deliberation Framework): introduces a framework for multi-agent LLM collaboration, enabling agents to learn adaptive meta-cognitive policies through a Meta-Cognitive State Space (agent's internal cognitive status), Agent's Observation with Meta-Cognition (local and peer meta-cognitive states), a Policy Network (integrates self-assessment and social context), and a Deliberative Action Space (high-level strategic choices) optimized by SoftRankPO (stable policy optimization algorithm).
- This framework allows agents to dynamically adjust their behavior based on internal confidence and situational context, moving beyond fixed collaboration protocols to dynamic, deliberative strategies.
- SoftRankPO, a key component, stabilizes policy learning by converting raw rewards into rank-based advantages, ensuring robust convergence across diverse reward regimes.

---

[Leveraging LLM-Based Agents for Intelligent Supply Chain Planning](http://arxiv.org/abs/2509.03811)

- SCPA (Supply Chain Planning Agent): introduces an LLM-based agent framework for intelligent supply chain planning, featuring Input (user query), Memory (short-term/long-term storage), a Pre-trained LLM (foundation large language model), Task Management (planning orchestration), Task Execution (sub-task processing), and Output (planning report).
- The framework leverages LLM-based agents for intent classification, task orchestration, task execution, and iterative plan correction, enabling autonomous interpretation of natural language queries and dynamic adjustment of plans.
- SCPA demonstrates improved operational efficiency, plan accuracy, and stock availability in real-world e-commerce scenarios by providing evidence-based planning reports and automating complex supply chain decision-making.

---

[SAMVAD: A Multi-Agent System for Simulating Judicial Deliberation Dynamics in India](http://arxiv.org/abs/2509.03793v1)

- SAMVAD (Simulated Agent-based Multi-agent Verdict Adjudication): introduces a Multi-Agent System for simulating judicial deliberation in India, comprising an Orchestrator (manages simulation lifecycle), Judge Agent (generates impartial instructions), Prosecution Counsel Agent (constructs arguments for prosecution), Defense Counsel Agent (constructs arguments for defense), Adjudicator Agents (simulate judicial bench), LLMs (power agents' reasoning), RAG (grounds reasoning in knowledge), Legal Knowledge Base (authoritative Indian legal documents), Vector DB (stores legal document embeddings), Case Files (structured input for simulation), and Final Judgement (consensus-based verdict output).
- The system's core innovation is the deep integration of a domain-specific RAG pipeline, grounding LLM-powered agents in Indian legal texts to generate legally sound, citable instructions and arguments.
- This framework offers a configurable and explainable platform for exploring legal reasoning and group decision-making dynamics within the Indian judicial context, enhancing simulation fidelity and transparency.

---

[Towards Personalized Explanations for Health Simulations: A Mixed-Methods Framework for Stakeholder-Centric Summarization](http://arxiv.org/abs/2509.04646v1)

- Mixed-Methods Framework for Stakeholder-Centric Summarization: introduces a systematic approach for generating tailored explanations of health simulations, with components including Simulation Model, Decomposition, RDF Representation, Finetuning, Visual Synthesis, LLM Prompting, Factuality Evaluation, Engagement Evaluation, Semi-structured Interviews, Summary Revision, Controllable Aspects, Validated Questionnaires, Quality Assessment, Preferred Aspects Identification, Contextual Information Retrieval, Optimization Strategies, Target Definition, LLM Optimization (Retriever, EAG-Sum, Direct Preference Optimization), Quantitative Evaluation, and Extrinsic Evaluation.
- This framework addresses the challenge of making complex health simulation models accessible to diverse stakeholders by tailoring LLM-generated summaries to their specific informational needs and stylistic preferences.
- It employs a two-step iterative process involving initial summary generation and evaluation, followed by optimization of LLMs and further assessment to ensure factual correctness and user engagement.

---

[Maestro: Joint Graph & Config Optimization for Reliable AI Agents](http://arxiv.org/abs/2509.04642v1)

- Maestro: introduces a framework-agnostic holistic optimizer for LLM agents that jointly searches over agent graphs and configurations to maximize agent quality, subject to explicit rollout/token budgets.
- It employs a block-coordinate scheme, alternating between C-step (configuration updates) and G-step (graph updates), guided by both numeric and reflective textual feedback from execution traces.
- The framework supports flexible agent-graph spaces, including branching, memory/state nodes, tool-augmented subroutines, and multi-model/multi-tool choices with tunable hyperparameters, addressing structural failure modes that prompt tuning alone cannot fix.

---

[Scaling Environments for Organoid Intelligence with LLM-Automated Design and Plasticity-Based Evaluation](http://arxiv.org/abs/2509.04633v1)

- LLM-Automated Design and Plasticity-Based Evaluation Framework: introduces a closed-loop system for training neural organoids in scalable virtual environments, leveraging an LLM for automated experimental protocol design and evaluating learning through synaptic plasticity.
- The framework includes three distinct virtual environments (conditional avoidance, predator-prey, Pong) with increasing complexity, formalizing state/action spaces, sensory encoding, motor decoding, and feedback protocols.
- The LLM acts as a meta-controller, generating and optimizing experimental protocols, and is refined through prompt engineering and fine-tuning based on collected performance and neurophysiological data.

---

[Bootstrapping Task Spaces for Self-Improvement](http://arxiv.org/abs/2509.04575v1)

- EXIT (Exploratory Iteration): introduces an autocurriculum RL method for LLMs to perform multi-step self-improvement at inference-time, training on informative single-step iterations by growing a task space.
- The framework leverages an LLM fine-tuned with GRPO, dynamically sampling self-iteration task instances from a Task Buffer, guided by a Learning Potential Score, and enhanced by Self-Divergence Steps and a Diversity Bonus in an Embedding Space.
- This approach enables robust self-improvement and generalization across various domains, including competition math, multi-turn tool-use, and ML engineering, by efficiently generating and learning from diverse task instances within a Search Scaffold.

---

[EMERGENT SOCIAL DYNAMICS OF LLM AGENTS IN THE EL FAROL BAR PROBLEM](http://arxiv.org/abs/2509.04537v1)

- LLM Agents: introduces a simulation of LLM agents in a spatially extended El Farol Bar problem, where each autonomous decision-maker, powered by an LLM (GPT-4o), generates messages, memories, and actions within a 2D grid environment, influenced by a crowding threshold and communication radius, guided by a prompt.
- The simulation reveals emergent social dynamics, including spontaneous motivation, collective decision-making, and the development of human-like bounded rationality and differentiated social roles among agents.
- These findings demonstrate how LLM agents balance game-theoretic rationality with culturally-encoded social motivations, providing a new paradigm for studying complex social systems.

---

[Narrative-to-Scene Generation: An LLM-Driven Pipeline for 2D Game Environments](http://arxiv.org/abs/2509.04481v1)

- Narrative-to-Scene Generation Pipeline: introduces a lightweight pipeline that transforms short narrative prompts into a sequence of 2D tile-based game scenes, reflecting the temporal structure of stories, with LLM (story generation, predicate extraction, terrain suggestion), Narrative Parsing (story summarization, time frame segmentation, predicate triples extraction), Semantic Matching (object-tile alignment, affordance filtering), Scene Synthesis (procedural terrain generation, spatial object placement, layered 2D scene rendering), GameTileNet Dataset (visual asset repository, semantic embeddings), and Knowledge Graph Module (symbolic reasoning, temporal integration).
- The pipeline segments LLM-generated narratives into three key time frames, extracts "Object-Relation-Object" predicate triples, and retrieves visual assets using affordance-aware semantic embeddings from the GameTileNet dataset.
- It generates layered terrain using Cellular Automata and places objects using spatial rules grounded in the predicate structure, ensuring semantic coherence and narrative alignment in 2D game environments.

---

[Bootstrapping Reinforcement Learning with Sub-optimal Policies for Autonomous Driving](http://arxiv.org/abs/2509.04712v1)

- Bootstrapping RL via Suboptimal Policy Framework: introduces a novel DRL-based autonomous driving framework that integrates a suboptimal controller to guide RL agents, enhancing exploration and learning efficiency in complex driving scenarios.
- The framework leverages the suboptimal policy both as a soft constraint on the RL policy during initial training and as a source for populating the replay buffer with additional training samples.
- This approach enables the RL agent to overcome exploration barriers and converge on optimal driving policies by providing plausible and human-like behavior.

---

[In-Context Policy Adaptation via Cross-Domain Skill Diffusion](http://arxiv.org/abs/2509.04535v1)

- ICPAD (In-Context Policy Adaptation): introduces a framework for rapid policy adaptation in long-horizon multi-task environments, leveraging cross-domain skill diffusion and dynamic domain prompting to adapt skill-based RL policies to diverse target domains with limited data.
- The framework learns domain-agnostic prototype skills and a domain-grounded skill adapter jointly from offline data through cross-domain consistent diffusion processes, which are then adapted in-context using dynamic domain prompting.
- This approach facilitates unified policy adaptation by using prototype skills as a middle-tier layer, translating them into domain-specific actions via dedicated skill adapters guided by retrieval-based attention from few-shot target data.

---

[SasAgent: Multi-Agent AI System for Small-Angle Scattering Data Analysis](http://arxiv.org/abs/2509.05363)

- SasAgent (Multi-Agent AI System for Small-Angle Scattering Data Analysis): introduces a multi-agent AI system for automating small-angle scattering (SAS) data analysis, featuring a Coordinator Agent (interprets, delegates), Generation Agent (generates synthetic data), Fitting Agent (fits experimental data), SLD Agent (calculates SLD), Model Data Tool (generates scattering function), RAG Documentation Tool (provides SasView documentation), Bump Fitting Tool (executes SasView fitting), SLD Calculator Tool (calculates SLD), and a Gradio-based User Interface (Gradio-based), all powered by LLMs (power agents).
- The system is self-aware, capable of performing SLD calculation, synthetic data generation, and experimental data analysis, while guiding users through intuitive text prompts and data uploads.
- Implemented using CrewAI and a Gradio-based web interface, SasAgent leverages LLM-friendly tools derived from the SasView Python library to streamline scientific workflows and enhance automation in SAS research.

---

[SYNTHESIZING SHEET MUSIC PROBLEMS FOR EVALUATION AND REINFORCEMENT LEARNING](http://arxiv.org/abs/2509.04059)

- Data Synthesis Framework: introduces a novel method for generating verifiable sheet music problems grounded in music theory, serving as both evaluation benchmarks and training data for reinforcement learning with verifiable rewards (RLVR).
- This framework programmatically generates sheet music questions and answers in both textual (ABC notation) and visual (staff notation image) modalities, without reliance on LLMs.
- The resulting Synthetic Sheet Music Reasoning Benchmark (SSMR-Bench) and training set enhance LLMs' and MLLMs' reasoning abilities in sheet music understanding and facilitate AI-assisted music creation.

---

[DEPTH-BREADTH SYNERGY IN RLVR: UNLOCKING LLM REASONING GAINS WITH ADAPTIVE EXPLORATION](http://arxiv.org/abs/2508.13755)

- DARS (Difficulty Adaptive Rollout Sampling): introduces a framework to unlock LLM reasoning gains by addressing depth and breadth dimensions in RLVR, utilizing pre-rollout difficulty estimation and multi-stage rollout re-balancing to re-weight hard problems.
- The framework includes two schedules, Equal-Treatment (ET) and Hardness-Weighted (HW), for rebalancing cumulative advantage, and can be augmented with Large Breadth Training (DARS-Breadth) by replacing PPO mini-batch updates with full-batch updates for synergistic performance.
- DARS improves Pass@K performance by focusing on hard problems, while DARS-Breadth further enhances Pass@1 by sustaining exploration and reducing gradient noise through increased training data breadth.

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

