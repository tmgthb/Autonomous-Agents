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



#### 21st April 2026



[TeamFusion: Supporting Open-ended Teamwork with Multi-Agent Systems](http://arxiv.org/abs/2604.19589)

- TeamFusion: introduces a multi-agent framework for open-ended teamwork that replaces direct aggregation with structured, preference-grounded agent deliberation.
- The system instantiates Proxy Agents conditioned on individual preferences to engage in a structured discussion, which a Remix Agent then synthesizes into actionable deliverables.
- TeamFusion improves representativeness and decision usefulness by explicitly modeling the consensus-seeking process rather than collapsing diverse viewpoints into a single average.

---

[A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression](http://arxiv.org/abs/2604.19572)

- TACO: introduces a plug-and-play, self-evolving framework that automatically discovers and refines compression rules from interaction trajectories to reduce redundant terminal output for LLMs.
- The framework maintains a Global Rule Pool for cross-task knowledge reuse and performs intra-task rule evolution to adapt to heterogeneous terminal environments without additional training.
- Experimental results demonstrate that TACO consistently improves agent performance and token efficiency across multiple benchmarks and backbone LLMs by filtering noise while preserving critical task-relevant information.

---

[Multi-modal Reasoning with LLMs for Visual Semantic Arithmetic](http://arxiv.org/abs/2604.19567)

- SAri-RFT: introduces a framework for visual semantic arithmetic by post-training LVLMs using GRPO and verifiable reward functions to improve relational reasoning.
- The framework utilizes the IRPD, a comprehensive dataset of 18 relations and 1500+ subject-object pairs, to evaluate two-term subtraction and three-term analogy tasks.
- By incorporating verifiable rewards, the method enables LVLMs to ground symbolic reasoning in visual perception, significantly outperforming traditional embedding-based arithmetic approaches.

---

[On Reasoning-Centric LLM-based Automated Theorem Proving](http://arxiv.org/abs/2604.19558)

- ReCent-Prover: introduces a reasoning-centric framework for automated theorem proving in Rocq that integrates validation with reflection and retrieval with planning to enhance proof search robustness and strategy alignment.
- The framework utilizes an LLM to perform self-reflection on generated tactics, filtering out potentially misapplied steps that could lead to unprovable branches.
- Retrieval with planning improves knowledge selection by conditioning lemma and proof retrieval on LLM-generated natural-language proof plans rather than relying on subgoal similarity.

---


[A-MAR: Agent-based Multimodal Art Retrieval for Fine-Grained Artwork Understanding](http://arxiv.org/abs/2604.19689)

- A-MAR: introduces an agent-based retrieval-augmented framework that explicitly decomposes complex art-related queries into structured reasoning plans to guide targeted evidence retrieval.
- The framework utilizes an Agent-based Planner to define evidence requirements, a Multimodal Reranker to select relevant context from an Art Context Knowledge Base, and a Multimodal LLM to synthesize grounded explanations.
- The authors also introduce ArtCoT-QA, a diagnostic benchmark designed to evaluate multi-step reasoning, evidence grounding, and retrieval faithfulness in the domain of fine art.

---

[InHabit: Leveraging Image Foundation Models for Scalable 3D Human Placement](http://arxiv.org/abs/2604.19673)

- InHabit: introduces a fully automatic data generation engine that leverages foundation models to populate 3D scenes with contextually meaningful human interactions.
- The framework utilizes a VLM for affordance reasoning, an image-editing model for visual synthesis, and an optimization-based lifting process to ground humans in 3D space.
- InHabit produces the InHabitants dataset, containing 78K photorealistic samples across 800 building-scale scenes, which improves performance for downstream contact estimation and HSI reconstruction tasks.

---

[Chat2Workflow: A Benchmark for Generating Executable Visual Workflows with Natural Language](http://arxiv.org/abs/2604.19667)

- Chat2Workflow: introduces a benchmark for generating executable visual workflows from natural language, utilizing an LLM-based Agent, Workflow Orchestration Platform, Auto-Repair Module, Variable Summaries, and Node Knowledge Base.
- The framework employs a Chain-of-Thought approach to generate JSON-based workflow representations that are subsequently converted into executable YAML files for deployment.
- Experimental results demonstrate that while LLMs can capture high-level intent, they struggle with complex, evolving requirements, necessitating an error-driven agentic baseline to improve resolve rates.

---

[Cyber Defense Benchmark: Agentic Threat Hunting Evaluation for LLMs in SecOps](http://arxiv.org/abs/2604.19533)

- Cyber Defense Benchmark: introduces a rigorous evaluation framework for measuring LLM agent performance in open-ended, evidence-driven threat hunting against real Windows event telemetry.
- The framework utilizes HolodeckHuntEnv to present agents with un-segmented log databases, requiring iterative SQL-based hypothesis generation and evidence synthesis to identify malicious activity.
- Evaluation of five frontier LLMs reveals significant performance gaps, with all models failing to meet the minimum operational recall threshold for unsupervised SOC deployment.

---

[ECLASS-Augmented Semantic Product Search for Electronic Components](http://arxiv.org/abs/2604.19664)

- ECLASS-Augmented Semantic Product Search framework: introduces a three-stage pipeline utilizing a Rewriter, Retriever, and Re-ranker to bridge the vocabulary gap between natural-language queries and structured industrial product data.
- The framework leverages hierarchical ECLASS metadata to augment product embeddings, significantly improving retrieval performance for specialized electronic components.
- Empirical results demonstrate that combining basic product data with ECLASS-enriched embeddings and re-ranking outperforms both classical lexical methods and foundation model baselines in industrial search tasks.

---

[An AI Agent Execution Environment to Safeguard User Data](http://arxiv.org/abs/2604.19657)

- GAAP (Guaranteed Accounting for Agent Privacy): introduces an execution environment that provides deterministic confidentiality guarantees for AI agents by enforcing user-defined permission specifications through information flow control on LLM-generated code artifacts.
- The framework utilizes a private data database, a permission database, a disclosure log, and an annotation framework to track and restrict data flows across multi-shot agent executions without requiring trust in the LLM or user prompts.
- GAAP effectively mitigates data disclosure attacks by intercepting tool calls and validating them against persistent, user-defined policies, ensuring that sensitive information is only shared with authorized external parties.

---

[SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](http://arxiv.org/abs/2604.19638)

- SafetyALFRED: introduces a benchmark for evaluating the ability of MLLMs to recognize and mitigate safety hazards in embodied household tasks.
- The framework utilizes a multi-agent system comprising a Safety Judge and an Embodied Agent to decouple hazard recognition from task-oriented mitigation.
- Experimental results reveal a significant alignment gap where MLLMs accurately recognize hazards in static QA settings but struggle to translate this knowledge into effective embodied mitigation.

---

[Time Series Augmented Generation for Financial Applications](http://arxiv.org/abs/2604.19633)

- TSAG (Time Series Augmented Generation): introduces a tool-augmented RAG framework that delegates quantitative financial tasks to verifiable external tools to improve LLM reasoning and accuracy.
- The framework utilizes a modular architecture comprising User-, LLM-, Tools-, and DB-layers to isolate and measure an agent's ability to parse queries, select tools, and extract parameters.
- The study provides a comprehensive empirical benchmark comparing various LLMs, demonstrating that top-performing agents achieve near-perfect tool-use accuracy with minimal hallucination when integrated into the TSAG pipeline.

---

[Goal-Oriented Semantic Communication for Logical Decision Making](http://arxiv.org/abs/2604.19614)

- Goal-Oriented Semantic Communication framework: introduces a principled foundation for goal-oriented semantic communication by grounding task-relevant information selection in First-Order Logic (FOL) and inductive logical probability.
- The framework utilizes a semantic information bottleneck principle to identify and transmit FOL clauses that most effectively resolve uncertainty regarding goal-oriented states.
- A polynomial-time lexicographical sorting algorithm is proposed to optimize the communication objective, demonstrating effectiveness in safe path-following tasks within a neuro-symbolic simulator.

---

[AblateCell: A Reproduce-then-Ablate Agent for Virtual Cell Repositories](http://arxiv.org/abs/2604.19606)

- AblateCell: introduces an end-to-end agentic framework that automates scientific ablation studies by reproducing baselines via Planner Agent and Code Agent, and conducting closed-loop ablation using Bandit State and Git Worktree.
- The system utilizes a Domain Knowledge Base to ground hypothesis generation and employs graph-based execution to manage dependencies during systematic model component evaluation.
- AblateCell achieves high success rates in identifying performance-critical components across diverse single-cell perturbation models by balancing exploration and exploitation through adaptive bandit sampling.

---

[Active Inference-Enabled Agentic Closed-Loop ISAC with Long-Horizon Planning](http://arxiv.org/abs/2604.19599)

- AIF-driven wireless agentic system: introduces an active inference-based framework for closed-loop ISAC that jointly optimizes control and sensing resource allocation via message passing on a factor graph.
- The system utilizes a generative model acting as a digital twin, incorporating a localization model for uncertainty-aware inference and a localization CKM for long-horizon planning.
- Simulation results demonstrate that the agent adaptively balances tracking accuracy, control effort, and sensing resource consumption by anticipating the impact of actions on future performance.

---

[Paparazzo: Active Mapping of Moving 3D Objects](http://arxiv.org/abs/2604.19556)

- Paparazzo: introduces a learning-free framework for active 3D reconstruction of non-cooperative moving objects by alternating between an Object Tracking Mode and an Object Mapping Mode.
- The framework utilizes an Extended Kalman Filter (EKF) for motion prediction and 3D Gaussian Splatting (3DGS) for information-driven viewpoint selection to compensate for target motion.
- Paparazzo dynamically balances viewpoint informativeness, motion feasibility, and temporal synchronization to maintain reconstruction progress even during object motion interruptions.

---

[Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment](http://arxiv.org/abs/2604.19548)

- ReTAS (Reasoning via Thesis-Antithesis-Synthesis): introduces a dialectical framework to mitigate Actor-Observer Asymmetry in LLMs by enforcing perspective-invariant reasoning through Thesis, Antithesis, and Synthesis stages.
- The framework utilizes Group Relative Policy Optimization (GRPO) to align LLMs with dialectical reasoning, incorporating attribution-, execution-, and format-rewards to ensure objective fault localization.
- ReTAS effectively decouples agent reasoning from role-induced cognitive biases, demonstrating superior performance in fault attribution and downstream task reliability across diverse multi-agent scenarios.

---

[FOCAL: Filtered On-device Continuous Activity Logging for Efficient Personal Desktop Summarization](http://arxiv.org/abs/2604.19541)

- FOCAL: introduces a privacy-first, multi-agent system that utilizes a unified filter-plan-log architecture to transform continuous desktop interaction streams into task-organized personal logs.
- The system employs a Filter Agent for noise suppression, a Brain Agent for task attribution, a Record Agent for selective visual reasoning, a Memory Agent for task-isolated context management, and a Summary Agent for coherent summarization.
- By moving task-aware control before expensive visual reasoning, FOCAL significantly reduces VLM token consumption and computational overhead while maintaining task-faithful context under frequent task switching.

---

[Mesh Memory Protocol: Semantic Infrastructure for Multi-Agent LLM Systems](http://arxiv.org/abs/2604.19540)

- MMP (Mesh Memory Protocol): introduces a semantic infrastructure for multi-agent LLM systems that enables cross-session cognitive collaboration through CAT7, SVAF, Inter-agent lineage, and Remix.
- The framework utilizes CAT7 to provide a universal semantic schema for CMBs, which are then evaluated by SVAF to ensure per-field admission based on role-indexed anchors.
- By implementing Inter-agent lineage and write-time filtered Remix, the protocol allows LLMs to maintain persistent, grounded, and evaluated cognitive state across session restarts without relying on raw history replay.

---

[Integrating Anomaly Detection into Agentic AI for Proactive Risk Management in Human Activity](http://arxiv.org/abs/2604.19538)

- ADFM-AAI: introduces a conceptual framework that integrates anomaly detection with agentic AI to provide proactive, autonomous fall mitigation for elderly populations.
- The architecture utilizes a multi-agent system where LLMs serve as a central reasoning service to dynamically orchestrate data acquisition, anomaly analysis, and intervention strategies.
- By reframing fall detection and prediction as anomaly detection problems, the system enables adaptive, goal-directed responses to complex, real-world human activity risks.

---

[Revac: A Social Deduction Reasoning Agent](http://arxiv.org/abs/2604.19523)

- Revac_8: introduces a multi-module architecture for social deduction games that integrates structured memory, relational graph analysis, and dynamic communication strategies to achieve human-level performance.
- The framework utilizes a Reviewer Agent for logical deduction and a Dynamic Tone Selector to adapt communication styles based on the current game state and social context.
- By employing a Social Alignment Graph, the agent performs relational reasoning to detect collusion and deception, overcoming the limitations of standard LLMs in high-stakes multi-agent environments.

---

[Accelerating Optimization and Machine Learning through Decentralization](http://arxiv.org/abs/2604.19518)

- Algorithm 1: introduces a server-assisted decentralized optimization framework that leverages local geometric properties to accelerate convergence by utilizing tailored local smoothness constants.
- The framework employs a switching mechanism that transitions from heterogeneous local step sizes to a universal step size to ensure convergence to an exact optimal solution.
- Theoretical performance is rigorously validated using the Performance Estimation Problem (PEP) framework to provide exact worst-case guarantees across specified function classes.

---

[From Experience to Skill: Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](http://arxiv.org/abs/2604.19516)

- MAGEO: introduces a multi-agent framework that reframes Generative Engine Optimization as a strategy learning problem, utilizing coordinated planning, editing, and fidelity-aware evaluation to distill reusable optimization skills.
- The framework employs a dual-layer architecture where an execution layer performs iterative content optimization and a learning layer consolidates successful editing patterns into a persistent Skill Bank.
- To ensure rigorous assessment, the paper introduces a Twin Branch Evaluation Protocol for causal attribution and the DSV-CF metric to balance semantic visibility with citation fidelity.

---

[EVPO: Explained Variance Policy Optimization for Adaptive Critic Utilization in LLM Post-Training](http://arxiv.org/abs/2604.19485)

- EVPO (Explained Variance Policy Optimization): introduces an adaptive RL framework that dynamically switches between critic-based and batch-mean advantage estimation based on batch-level explained variance.
- The framework utilizes explained variance as a real-time indicator to detect when a critic's estimation noise exceeds its signal, effectively mitigating variance inflation during LLM post-training.
- By unifying PPO and GRPO as extremes of a Kalman filtering baseline selection, EVPO provides a provable guarantee of maintaining lower advantage variance than either fixed baseline throughout training.

---

[Four-Axis Decision Alignment for Long-Horizon Enterprise AI Agents](http://arxiv.org/abs/2604.19457)

- Four-Axis Decision Alignment framework: introduces a decomposition of long-horizon agent evaluation into four orthogonal axes—FRP, RCS, CRR, and CAR—to replace insufficient aggregate accuracy metrics.
- The framework utilizes LongHorizon-Bench to evaluate six distinct memory architectures, revealing that aggregate accuracy hides critical failure modes in factual precision, reasoning, and regulatory compliance.
- The research demonstrates that institutional and decisional alignment are load-bearing properties for regulated enterprise agents, necessitating explicit measurement beyond standard truthfulness and harmlessness benchmarks.

---

[What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search](http://arxiv.org/abs/2604.19440)

- LLM-guided evolutionary search framework: introduces a large-scale trajectory analysis of 15 LLMs across 8 tasks to identify the mechanisms driving optimization performance in agentic systems.
- The study reveals that effective LLMs function as local refiners, characterized by progressive semantic localization and frequent incremental breakthroughs, rather than relying on high novelty.
- The research demonstrates that breakthrough rate is a stronger predictor of optimization success than zero-shot capability, providing actionable insights for model selection and training.

---

[seneca: A Personalized Conversational Planner](http://arxiv.org/abs/2604.19425)

- seneca: introduces a conceptual framework for an AI-assisted planner that integrates a Conversational Agent, a structured Work Item View, a Processor, and a persistent Database to bridge the gap between user-expressed demands and underlying needs.
- The system utilizes a Conversational Agent to scaffold reflection and clarify goals, while the Processor ensures synchronization between the user-facing interface and the persistent Database containing Frameworks, Patterns, and Work Items.
- The architecture is designed to support self-regulation and goal-value alignment by combining the persistence of digital to-do lists with the interactive, reflective capabilities of LLMs.

---

[M2GRPO: Mamba-based Multi-Agent Group Relative Policy Optimization for Biomimetic Underwater Robots Pursuit](http://arxiv.org/abs/2604.19404)

- M2GRPO: introduces a Mamba-based policy network that integrates selective state-space modeling for long-horizon temporal dependencies and attention-based relational features for multi-agent coordination.
- The framework utilizes MAGRPO to perform group-normalized advantage estimation, which eliminates the need for explicit value functions and enhances training stability under the CTDE paradigm.
- Experimental results demonstrate that M2GRPO outperforms existing MARL baselines in pursuit success rate and capture efficiency for biomimetic underwater robot swarms.

---

[How damaging is zero-sum thinking to an agent’s interests when the world is positive-sum?](http://arxiv.org/abs/2604.19359)

- Game Theory Analysis of Zero-Sum Decision Rules: introduces a systematic evaluation of whether zero-sum decision rules like Maximin (guarantees minimum payoff against adversarial opponent) and Minimax (caps opponent's best attainable payoff) harm agent interests in positive-sum environments compared to Nash Equilibrium (mutually optimal strategy profile).
- The paper demonstrates that Maximin can strictly Pareto dominate Nash Equilibrium in a significant class of games, challenging the evolutionary presumption that non-maximizing decision rules are inherently inferior.
- The authors establish a cardinality theorem showing that the class of games where Maximin dominates Nash Equilibrium is as large as the class where Nash Equilibrium dominates Maximin, while also identifying Relative-Maximin (maximin applied to relative payoff transformation) and ESS (strategy resistant to invasion by alternative strategies) as relevant behavioral benchmarks.

---

[Do Agents Dream of Root Shells? Partial-Credit Evaluation of LLM Agents in Capture The Flag Challenges](http://arxiv.org/abs/2604.19354)

- DeepRed: introduces an open-source benchmark framework for evaluating LLM agents in realistic, isolated Capture The Flag (CTF) environments using a partial-credit scoring methodology.
- The framework utilizes an automated pipeline where a Summary LM condenses execution logs and a Judge LM assesses progress against predefined checkpoints derived from public writeups.
- Empirical evaluation of ten LLMs reveals that while agents can achieve partial progress, they struggle with long-horizon planning and non-standard discovery tasks in adversarial settings.

---

[Large Language Models Exhibit Normative Conformity](http://arxiv.org/abs/2604.19301)

- LLM-MAS: introduces a framework to distinguish between informational and normative conformity in LLMs by manipulating social context variables such as publicness, evaluation, and relationship continuity.
- The study evaluates six LLMs to demonstrate that normative conformity is a distinct behavioral tendency that can be manipulated through peer endorsement and the assignment of influential attributes.
- Analysis of internal hidden layer activations reveals that normative and informational conformity are driven by distinct internal mechanisms, providing insights into how social norms are implemented within LLMs.

---

[Rethinking Scale: Deployment Trade-offs of Small Language Models under Agent Paradigms](http://arxiv.org/abs/2604.19299)

- Agent Paradigms for Small Language Models: introduces a comprehensive empirical study evaluating 27 open-source SLMs across Base SLM, Single-Agent System (SAS), and Multi-Agent System (MAS) paradigms in financial settings.
- The study demonstrates that while SAS provides an optimal balance of effectiveness and efficiency, MAS introduces significant coordination overhead and systemic instability.
- The research identifies that architectural design can effectively compensate for limited model scale, though increased complexity often leads to higher failure rates due to delegation and context management issues.

---

[Explicit Trait Inference for Multi-Agent Coordination](http://arxiv.org/abs/2604.19278)

- ETI (Explicit Trait Inference): introduces a psychology-grounded framework that enables LLM agents to infer and track partner characteristics along warmth and competence dimensions to improve coordination.
- The framework utilizes Task History, Trait Inference, Agent Context, and Plan &amp; Act components to distill interaction histories into stable trait profiles that guide planning and delegation.
- Experimental results demonstrate that ETI improves coordination and task performance across diverse multi-agent settings by providing agents with structured awareness of their partners' traits.

---

[Warmth and Competence in the Swarm: Designing Effective Human-Robot Teams](http://arxiv.org/abs/2604.19270)

- SwarmUI: introduces a framework for investigating human social perception of robot swarms by manipulating behavioral parameters including speed, separation distance, and broadcast duration.
- The architecture integrates an ARGoS Simulator with a Web Interface Plugin to facilitate real-time human-swarm collaboration and data collection across observer and operator roles.
- Experimental results demonstrate that social perceptions of warmth and competence significantly influence human team preferences, often outweighing objective task performance metrics.

---

[DR-MMSearchAgent: Deepening Reasoning in Multimodal Search Agents](http://arxiv.org/abs/2604.19264)

- DR-MMSearchAgent: introduces a reinforcement learning framework that enhances multimodal search agents by utilizing structural proximity-weighted advantage injection to mitigate exploration degradation and trajectory redundancy.
- The framework incorporates a refining agent for real-time trajectory compression and a bidirectionally guided adaptive reward mechanism to dynamically calibrate interaction depth based on solution quality.
- Extensive experiments on the newly constructed BridgeVQA dataset demonstrate that the approach achieves state-of-the-art performance by balancing robust exploration with efficient information retrieval.

---

[BONSAI: A Mixed-Initiative Workspace for Human-AI Co-Development of Visual Analytics Applications](http://arxiv.org/abs/2604.19247)

- BONSAI: introduces a mixed-initiative workspace for the co-development of Visual Analytics applications, utilizing a four-layer architecture (Hardware Layer, Service Layer, Orchestration Layer, Application Layer) to enforce modularity and interface contracts.
- The framework employs a hierarchical agent model that includes a top-level Nexus orchestrator, mid-level Squad Leads, and specialized AI Development Units to distribute agency and ensure structural rigor during development.
- BONSAI integrates a four-phase development process (Plan, Design, Monitor, Review) that treats semantic provenance as a first-class citizen to maintain auditability and human control over AI-assisted workflows.

---

[YAIFS: Yet (not) Another Intelligent Fog Simulator: A Framework for Agent-Driven Computing Continuum Modeling &amp; Simulation](http://arxiv.org/abs/2604.19181)

- YAIFS: introduces a layered, service-oriented architecture that transforms static simulations into interactive, programmable environments for cloud-edge systems.
- The framework integrates the Model Context Protocol (MCP) to decouple agent logic from simulator internals, enabling autonomous agents to observe, control, and optimize simulation workflows.
- YAIFS supports AI-driven experimentation through specialized agents, including an LLM-based assistant for natural language interaction and a multi-agent system for adaptive application placement.

---

[Distributed Multi-Sensor Control for Multi-Target Tracking Using Adaptive Complementary Fusion for LMB Densities](http://arxiv.org/abs/2604.19160)

- FDCD-SC (Fully Distributed Coordinate Descent Sensor Control): introduces a distributed multi-sensor control framework that utilizes an adaptive complementary fusion rule and an information-theoretic objective function to optimize sensor actions for multi-target tracking.
- The framework employs a PIMS (Predicted Ideal Measurement Set) and a pseudo-update mechanism to evaluate potential control commands without requiring real-time physical sensor movement.
- By integrating a flooding-based communication protocol, the approach ensures global coordination among sensors, significantly reducing cardinality errors in dynamic multi-target tracking scenarios.

---

[RLABC: Reinforcement Learning for Accelerator Beamline Control](http://arxiv.org/abs/2604.19146)

- RLABC (Reinforcement Learning for Accelerator Beamline Control): introduces an automated pipeline that transforms standard beamline configurations into RL environments by integrating the Elegant simulation program via SDDS interfaces.
- The framework utilizes a 57-dimensional state representation and stage learning strategies to decompose complex beamline tuning tasks into manageable subproblems for efficient RL training.
- RLABC employs a DDPG agent to optimize magnet parameters, achieving particle transmission performance comparable to established methods like differential evolution.

---

[RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation](http://arxiv.org/abs/2604.19092)

- RoboWM-Bench: introduces a manipulation-centric benchmark for evaluating the physical executability of video world models through embodied execution.
- The framework utilizes World Models to generate videos, which are then processed by Human-Hand Retargeting or an Inverse Dynamics Model to produce executable actions for a Simulation Platform.
- RoboWM-Bench employs a Real-to-Sim Engine to reconstruct real-world scenarios, enabling standardized and reproducible validation of whether predicted behaviors are physically grounded and executable.

---

[The Essence of Balance for Self-Improving Agents in Vision-and-Language Navigation](http://arxiv.org/abs/2604.19064)

- SDB (Stability–Diversity Balance): introduces a training-time (1→K→1) expand–select mechanism that balances behavioral diversity and learning stability in VLN by generating multiple latent hypotheses via a Diversity Expansion Module and consolidating them through a Stability Selection Module.
- The framework utilizes a Head-Shifting Generator to produce instruction-consistent variations and a Balanced Controller to perform reliability-aware soft fusion, ensuring stable, cumulative learning updates across iterations.
- SDB is a backbone-agnostic plug-in that improves navigation success and path efficiency by preventing premature commitment to suboptimal hypotheses and reducing uncoordinated decision switching.

---

[Refute-or-Promote: Adversarial Stage-Gated Multi-Agent Review for High-Precision LLM-Assisted Defect Discovery](http://arxiv.org/abs/2604.19049)

- Refute-or-Promote: introduces an adversarial, stage-gated multi-agent methodology for high-precision defect discovery that utilizes Stratified Context Hunting (SCH) for candidate generation, adversarial agents for kill mandates, an empirical validation gate, and a Cross-Model Critic (CMC) to filter false positives.
- The framework employs a unidirectional pipeline where creative and adversarial agents operate in parallel tracks to rigorously test candidate vulnerabilities before human-led disclosure.
- By implementing context asymmetry and cross-family verification, the system mitigates reasoning biases and correlated training errors that often lead LLMs to reach unanimous but incorrect conclusions.

---

[Explore Like Humans: Autonomous Exploration with Online SG-Memo Construction for Embodied Agents](http://arxiv.org/abs/2604.19034)

- ABot-Explorer: introduces an active exploration framework that unifies memory construction and navigation into an online, RGB-only process by leveraging VLMs to distill SNA.
- The framework utilizes VLM-distilled SNA to identify navigationally critical transit nodes, which are dynamically organized into a hierarchical SG-Memo to guide exploration.
- By prioritizing structural transit nodes over geometric frontiers, the system achieves human-like exploration efficiency and provides a reasoning-ready knowledge substrate for downstream embodied tasks.

---

[ClawCoin: An Agentic AI-Native Cryptocurrency for Decentralized Agent Economies](http://arxiv.org/abs/2604.19026)

- ClawCoin: introduces a collateral-backed, index-linked cryptocurrency designed to provide a stable, compute-aligned unit of account for decentralized AI agent economies.
- The framework utilizes an off-chain index calculator and on-chain oracle to track standardized inference costs, enabling agents to quote, escrow, and settle workflows in a shared numeraire.
- By integrating with an atomic multi-hop settlement layer, the system eliminates budget overruns and coordination failures inherent in fiat-denominated agent transactions.

---

[On Accelerating Grounded Code Development for Research](http://arxiv.org/abs/2604.19022)

- Grounded Code Development Framework: introduces a system that enables LLMs to access dynamic research repositories and technical documentation for context-aware code generation, utilizing Document Parsing Pipeline, Elasticsearch, Document Search Tool, LSP-based Semantic Code Search, and Skill Library.
- The framework employs a lexical search-based retrieval strategy to provide deterministic, up-to-date access to specialized scientific artifacts without requiring extensive model fine-tuning.
- The Skill Library acts as an orchestration layer, allowing researchers to define reusable, structured research methodologies that guide LLM agents through iterative information gathering, planning, and implementation phases.

---

[Security Is Relative: Training-Free Vulnerability Detection via Multi-Agent Behavioral Contract Synthesis](http://arxiv.org/abs/2604.19012)

- Phoenix: introduces a training-free multi-agent framework that resolves semantic ambiguity in vulnerability detection through Behavioral Contract Synthesis using Semantic Slicer, Requirement Reverse Engineer, and Contract Judge.
- The framework transforms open-ended vulnerability detection into a closed-form contract verification problem by utilizing Gherkin specifications as an explicit intermediate representation.
- Phoenix achieves state-of-the-art performance on the PrimeVul benchmark by leveraging specialized LLM agents to enforce project-specific security contracts rather than relying on pattern matching.

---

[A Multi-Agent Framework with Structured Reasoning and Reflective Refinement for Multimodal Empathetic Response Generation](http://arxiv.org/abs/2604.18988)

- MERG (Multimodal Empathetic Response Generation) framework: introduces a multi-agent system that decomposes response generation into a structured pipeline of MPA, CAEF, PSP, and SGRG, followed by a GRA for iterative refinement.
- The framework utilizes a closed-loop iterative optimization process where the GRA audits intermediate outputs from MPA, CAEF, PSP, and SGRG to identify and correct errors at the earliest responsible stage.
- By explicitly organizing multimodal perception, emotion forecasting, strategy planning, and response generation, the framework enables targeted regeneration and improves the emotional appropriateness and strategic coherence of responses generated by LLMs.

---

[SAVOIR: Learning Social Savoir-Faire via Shapley-based Reward Attribution](http://arxiv.org/abs/2604.18982)

- SAVOIR: introduces a game-theoretic framework for social RL that replaces heuristic credit assignment with prospective valuation via expected utility and fair attribution via Shapley values.
- The framework utilizes KernelSHAP to efficiently compute utterance-level rewards, enabling the training of LLMs that demonstrate superior social intelligence compared to existing methods.
- Experimental results on the SOTOPIA benchmark show that SAVOIR achieves state-of-the-art performance, with a 7B model matching or exceeding proprietary LLMs and large reasoning models.

---

[Gated Coordination for Efficient Multi-Agent Collaboration in Minecraft Game](http://arxiv.org/abs/2604.18975)

- Gated Collaborative Escalation Framework: introduces a partitioned information architecture that decouples private execution states from public coordination states to minimize context noise and improve multi-agent collaboration efficiency.
- The framework utilizes a three-tiered gating mechanism—comprising heuristic rules, cost-sensitive scoring, and a bounded LLM adjudicator—to transform communication from a default reflex into a selective, cost-benefit-driven decision.
- By maintaining compact, system-verified private working memory and enforcing protocolized public interaction, the system significantly reduces coordination deadlocks and improves task completion rates in long-horizon open-world environments.

---

[Superficial Success vs. Internal Breakdown: An Empirical Study of Generalization in Adaptive Multi-Agent Systems](http://arxiv.org/abs/2604.18951)

- Adaptive MAS: introduces an empirical study identifying topological overfitting and illusory coordination as primary failure modes in adaptive multi-agent systems when evaluated on out-of-distribution tasks.
- The study demonstrates that adaptive MAS often achieve high surface-level accuracy through the brute-force reasoning of individual LLMs rather than effective collaborative mechanisms, a phenomenon termed illusory coordination.
- The authors propose two novel metrics, Role Alignment and Connection Significance, to rigorously evaluate the internal dynamics of multi-agent collaboration beyond final-answer correctness.

---

[AutomationBench](http://arxiv.org/abs/2604.18934)

- AutomationBench: introduces a benchmark for evaluating LLMs on cross-application workflow orchestration via REST APIs, utilizing Task prompt, AI agent, Search tool, Execute tool, Environment noise, Simulated app state, and Deterministic assertions.
- The framework requires LLMs to autonomously discover relevant API endpoints, adhere to layered business policies, and navigate environments containing misleading records to achieve specific end-state goals.
- Grading is performed programmatically through deterministic assertions on the final state of simulated applications, ensuring reproducibility and eliminating subjective evaluation bias.

---

#### 20th April 2026


[Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence](http://arxiv.org/abs/2604.18292)

- Agent-World: introduces a self-evolving training arena that unifies scalable real-world environment synthesis with continuous self-evolving agent training to advance general agent intelligence.
- The framework utilizes Agentic Environment-Task Discovery to autonomously mine databases and tools, and Continuous Self-Evolving Agent Training to drive targeted learning through a closed-loop co-evolution of agent policies and environments.
- Agent-World incorporates a Self-Evolving Agent Arena that employs a Diagnosis Agent to identify capability gaps, which then informs the synthesis of new tasks to improve agent performance across diverse benchmarks.

---

[Negative Advantage Is a Double-Edged Sword: Calibrating Advantage in GRPO for Deep Search](http://arxiv.org/abs/2604.18235)

- CalibAdv: introduces a framework for calibrating GRPO advantage signals in deep search agents by employing Soft Advantage Penalization, Advantage Rebalancing, and Special Token Decoupling.
- The framework mitigates training collapse and performance degradation by fine-grained adjustment of negative advantages at intermediate steps and rebalancing positive and negative advantages at the final answer step.
- CalibAdv utilizes a silver document proxy to identify helpful intermediate steps without requiring additional external LLM annotations or sampling overhead.

---

[A Counterexample to EFX n ≥ 3 Agents, m ≥ n + 5 Items, Monotone Valuations via SAT-Solving](http://arxiv.org/abs/2604.18216)

- SAT-based EFX Counterexample Framework: introduces a computational approach to resolve the existence of EFX allocations by encoding the problem into SAT and SMT instances, utilizing SPASS-SAT, CaDiCal, DRAT-trim, LEAN, Z3, and C++ Verification Code.
- The framework demonstrates that EFX allocations always exist for three agents and seven goods, while providing a counterexample for three agents and eight or more goods.
- The methodology combines theoretical analysis, SAT-based search, and formal verification in LEAN to establish the non-existence of EFX allocations for specific monotone valuation settings.

---

[Bridging the Reasoning Gap in Vietnamese with Small Language Models via Test-Time Scaling](http://arxiv.org/abs/2604.17794)

- Vi-S1K (Vietnamese Simple Scaling 1K): introduces a systematic investigation into test-time scaling for Vietnamese SLMs, utilizing Qwen3-1.7B finetuned on a localized reasoning dataset to bridge the reasoning gap in elementary mathematics.
- The research demonstrates that Supervised Fine-Tuning (SFT) acts as a reasoning unlocker, significantly improving explanation quality and pedagogical coherence in small models.
- The study establishes a deployment hierarchy for SLMs, identifying that pure Chain-of-Thought combined with Self-Consistency is superior to complex agentic workflows like ReAct for sub-2B parameter models.

---


[Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs](http://arxiv.org/abs/2604.18576)

- BLF (Bayesian Linguistic Forecaster): introduces an agentic system for binary forecasting that maintains a semi-structured belief state updated iteratively through a tool-use loop.
- The system utilizes LLM (main), LLM (filter), and LLM (summ.) to process information, while employing a LeakFilter and Data tools to ensure backtesting validity against knowledge cutoffs.
- BLF achieves state-of-the-art performance on the ForecastBench benchmark by combining structured belief updates, hierarchical multi-trial aggregation, and hierarchical calibration.

---

[Semantic Entanglement in Vector-Based Retrieval: A Formal Framework and Context-Conditioned Disentanglement Pipeline for Agentic RAG Systems](http://arxiv.org/abs/2604.17677)

- SDP (Semantic Disentanglement Pipeline): introduces a four-stage preprocessing framework that mitigates semantic entanglement by restructuring documents based on operational context, utilizing CAF, Stage A, Stage B, Stage C, Stage D, and a Continuous Feedback Loop.
- The framework addresses geometric confusion in embedding spaces by aligning document structure with agentic retrieval requirements through context-conditioned preprocessing.
- Empirical evaluation on an enterprise healthcare knowledge base demonstrates that the pipeline reduces the Entanglement Index and improves Top-K retrieval precision by 50 percentage points.

---


[MultiWorld: Scalable Multi-Agent Multi-View Video World Models](http://arxiv.org/abs/2604.18564)

- MultiWorld: introduces a unified framework for scalable multi-agent, multi-view video world modeling that enables precise action adherence and synchronized simulation across diverse viewpoints.
- The framework utilizes MACM to manage agent-specific control and GSE to maintain 3D-aware global environmental consistency, facilitating scalable simulation of variable agent and camera counts.
- MultiWorld employs a DiT backbone with flow matching to synthesize high-fidelity, multi-view consistent videos, demonstrating superior performance in complex multi-player game and multi-robot manipulation scenarios.

---

[SynAgent: Generalizable Cooperative Humanoid Manipulation via Solo-to-Cooperative Agent Synergy](http://arxiv.org/abs/2604.18557)

- SynAgent: introduces a unified framework for scalable and physically plausible cooperative humanoid manipulation by leveraging solo-to-cooperative agent synergy.
- The framework utilizes an interaction-preserving retargeting method based on an Interact Mesh to maintain semantic integrity during motion transfer from single-agent data to multi-agent scenarios.
- SynAgent employs a trajectory-conditioned CVAE policy trained via multi-teacher distillation to achieve stable and controllable object-level trajectory execution across diverse object geometries.

---

[ClawEnvKit: Automatic Environment Generation for Claw-Like Agents](http://arxiv.org/abs/2604.18543)

- ClawEnvKit: introduces an autonomous pipeline that generates verified environments for LLM agents from natural language descriptions.
- The framework utilizes a multi-agent system comprising a Parser, a Generator, and a Validator to create task specifications, interaction interfaces, and evaluation functionals.
- ClawEnvKit enables scalable, on-demand environment generation and live evaluation, facilitating the construction of the Auto-ClawEval benchmark for LLM agents.

---

[MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](http://arxiv.org/abs/2604.18509)

- MASS-RAG: introduces a training-free, multi-agent framework that structures evidence processing into specialized agents to improve RAG robustness and factual accuracy.
- The framework utilizes a Summarizer Agent, Extractor Agent, and Reasoner Agent to distill complementary evidence, which is then processed by an optional Answer Agent and a final Synthesis Agent.
- By exposing multiple intermediate evidence views, the system enables the model to compare and reconcile heterogeneous information before generating a unified final answer.

---

[QRAFTI: An Agentic Framework for Empirical Research in Quantitative Finance](http://arxiv.org/abs/2604.18500)

- QRAFTI: introduces a multi-agent framework that automates empirical asset-pricing research by coordinating specialized agents, including Context Compaction Agent, Factor Research Agent, Code Writing Agent, and Standardized Reporting Agent, to perform multi-step financial analysis.
- The framework utilizes MCP Tools Servers to provide a constrained, reliable interface for recurring financial computations, while employing reflection-based planning to improve task execution accuracy.
- QRAFTI enhances reproducibility and explainability by generating structured computation graphs and standardized research reports that trace the entire workflow from raw data to final results.

---

[WorldDB: A Vector Graph-of-Worlds Memory Engine with Ontology-Aware Write-Time Reconciliation](http://arxiv.org/abs/2604.18478)

- WorldDB: introduces a memory engine for long-running agents that organizes data into Recursive Worlds, utilizes Content-addressed Immutability for auditability, and employs Edge Handlers to enforce structural semantics at write-time.
- The architecture replaces flat RAG with a graph-of-worlds structure, featuring a Reconciler that ensures all updates pass through defined semantics, a Tiered Resolver for entity unification, and a Background Consolidator for efficient long-term state management.
- By decoupling the retrieval pipeline from LLM-based logic, the system achieves high-performance, deterministic memory operations, demonstrating significant accuracy gains on the LongMemEval-s benchmark compared to existing bitemporal knowledge graph systems.

---

[Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation](http://arxiv.org/abs/2604.18468)

- Asset Harvester: introduces an end-to-end pipeline that converts sparse, in-the-wild object observations from autonomous driving logs into complete, simulation-ready 3D assets.
- The system utilizes a Data Ingestion Module for curation, SparseViewDiT for multiview image generation, and Object TokenGS for 3D Gaussian lifting to address challenges like limited-angle views and occlusions.
- The framework integrates with the NuRec simulation environment and employs a hybrid training strategy to ensure high-fidelity, reusable 3D assets for autonomous vehicle development.

---

[Progressive Online Video Understanding with Evidence-Aligned Timing and Transparent Decisions](http://arxiv.org/abs/2604.18459)

- Thinking-QwenVL: introduces a framework that decouples reasoning control from memory integration to enable evidence-aligned, transparent online video understanding.
- The ATDM module manages decision-making by decomposing queries into sub-goals and emitting progress and confidence metrics for real-time transparency.
- The HPSI module maintains a compact, relation-aware cognition state using learnable multi-level aggregation tokens to preserve global consistency under tight token budgets.

---

[MedProbeBench: Systematic Benchmarking at Deep Evidence Integration for Expert-level Medical Guideline](http://arxiv.org/abs/2604.18418)

- MedProbeBench: introduces a comprehensive benchmark and evaluation framework for assessing the deep evidence integration capabilities of LLMs and deep research agents in generating expert-level clinical guidelines.
- The framework utilizes a dual-tier approach combining holistic rubric-based quality assessment with fine-grained evidence verification to measure structural completeness, terminological precision, and claim-level grounding.
- Evaluation of 17 systems reveals that while models can achieve high surface fluency, they face significant bottlenecks in mechanistic reasoning and reliable evidence-based synthesis.

---

[TypeScript Repository Indexing for Code Agent Retrieval](http://arxiv.org/abs/2604.18413)

- ABCoder: introduces a high-performance TypeScript parser that replaces RPC-based language server calls with direct TypeScript Compiler API integration to generate efficient UniAST code indexes.
- The framework enables LLMs to perform repository-level reasoning by providing a structured graph of code entities, including call chains and dependency relationships, via the Model Context Protocol.
- By eliminating external RPC overhead, the parser significantly accelerates indexing for large-scale TypeScript repositories while maintaining high semantic accuracy for downstream code agent tasks.

---

[StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning](http://arxiv.org/abs/2604.18401)

- StepPO: introduces a step-level Agentic RL framework that aligns MDP formulation, trajectory representation, and credit assignment with the interaction step rather than individual tokens.
- The framework utilizes a structured step-level representation to maintain token-space consistency while enabling precise context management and asynchronous training.
- Empirical results on HotpotQA demonstrate that StepPO outperforms token-level PPO by providing more effective learning signals for multi-step agent reasoning and decision-making.

---

[OpenGame: Open Agentic Coding for Games](http://arxiv.org/abs/2604.18394)

- OpenGame: introduces an agentic framework for end-to-end web game creation that utilizes GameCoder-27B, an autonomous agent workflow, and Game Skill to translate natural-language specifications into playable games.
- The framework employs Template Skill to provide stable project scaffolding and Debug Skill to systematically resolve cross-file integration failures through a living debugging protocol.
- OpenGame-Bench evaluates generated games on build health, visual usability, and intent alignment using headless browser execution and VLM judging to ensure functional and interactive quality.

---

[ICEBREAKER for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](http://arxiv.org/abs/2604.18375)

- ICEBREAKER: introduces a two-step handshake framework for proactive initiation in conversational agents, utilizing Resonance-Aware Interest Distiller (RID) and Interaction-Oriented Starter Generator (ISG) to overcome the first-message barrier.
- The framework employs a Personalized Resonance Scorer to distill user interests and an Interaction-Oriented Starter Generator (ISG) to produce diverse, utility-aligned conversation starters.
- ICEBREAKER optimizes starter generation through a hybrid reward list search and hierarchical preference alignment to maximize user engagement and topical diversity.

---

[Dissecting AI Trading: Behavioral Finance and Market Bubbles](http://arxiv.org/abs/2604.18373)

- AI Trading Framework: introduces a simulated asset market populated by autonomous LLM agents to study behavioral finance patterns and market dynamics.
- The framework utilizes a structured Chain-of-Thought architecture with persistent memory files to trace the causal link between micro-level cognitive reasoning and macro-level market outcomes.
- The study demonstrates that LLM agents exhibit human-like behavioral biases, such as the disposition effect and extrapolative expectations, which can be causally amplified or suppressed through targeted prompt interventions.

---

[Training and Agentic Inference Strategies for LLM-based Manim Animation Generation](http://arxiv.org/abs/2604.18364)

- ManimTrainer and ManimAgent: introduces a unified framework for text-to-code-to-video generation using ManimTrainer (training pipeline for animation generation) and ManimAgent (inference pipeline for animation generation) with LLM (generates programmatic animation code), Manim Renderer (executes code to produce video), Reward Unification (fuses code and visual signals), GRPO (critic-free reinforcement learning), RITL (renderer-in-the-loop self-correction), RITL-DOC (RITL with API documentation), and Prompt Builder (constructs prompts for LLM).
- The study evaluates 17 open-source LLMs, demonstrating that SFT improves code quality while GRPO enhances visual outputs and responsiveness to extrinsic feedback.
- The research highlights that combining training strategies with agentic inference, specifically RITL-DOC, significantly improves render success rates and visual similarity, often surpassing larger models and closed-source baselines.

---

[ComPASS: Towards Personalized Agentic Social Support via Tool-Augmented Companionship](http://arxiv.org/abs/2604.18356)

- ComPASS: introduces a personalized social support framework that empowers LLMs with external tools to provide substantive, human-like companionship through ComPASS-Bench and the fine-tuned ComPASS-Qwen model.
- The framework utilizes a multi-stage LLM-based pipeline to synthesize user profiles and interaction records, enabling agents to select contextually appropriate tools based on user background and history.
- Evaluations demonstrate that tool-augmented responses significantly outperform traditional empathetic dialogue, with ComPASS-Qwen achieving performance comparable to larger models through efficient supervised fine-tuning.

---

[PRISMA: Preference-Reinforced Self-Training Approach for Interpretable Emotionally Intelligent Negotiation Dialogues](http://arxiv.org/abs/2604.18354)

- PRISMA: introduces an interpretable, emotionally intelligent negotiation dialogue system that leverages an Emotion-aware Negotiation Strategy-informed Chain-of-Thought (ENS-CoT) reasoning mechanism to guide LLMs toward strategically appropriate and empathetic responses.
- The framework employs a preference-reinforced self-training approach, integrating supervised initialization with iterative DPO and self-training to optimize step-by-step reasoning and negotiation effectiveness.
- PRISMA utilizes two novel datasets, JobNego and ResNego, to demonstrate superior performance in emotional appropriateness, strategy consistency, and overall negotiation outcomes compared to standard LLM-based baselines.

---

[Reliability of AI Bots Footprints in GitHub Actions CI/CD Workflows](http://arxiv.org/abs/2604.18334)

- Reliability of AI Bots Footprints in GitHub Actions CI/CD Workflows: presents an empirical study analyzing 61,837 CI/CD workflow runs triggered by five LLM-based agents to evaluate their impact on software delivery reliability.
- The study utilizes the AIDev dataset and GitHub Actions API to quantify workflow success rates, identifying a negative correlation between the frequency of agentic pull requests and overall CI/CD performance.
- The research defines a taxonomy of 13 pull request categories and employs GPT 5.0 alongside human validation to characterize the types of agentic contributions that most frequently lead to workflow failures.

---

[Will People Enjoy a Robot Trainer? A Case Study with Snoopie the Pacerbot](http://arxiv.org/abs/2604.18331)

- SNOOPIE: introduces an embodied robotic quadruped trainer that leverages physical presence and autonomous navigation to guide runners through customized interval training sessions.
- The system utilizes a two-phase interaction workflow consisting of an initial user-guided speed calibration followed by an autonomous robot-led interval training exercise.
- Experimental results demonstrate that the embodied robot significantly improves pace adherence and consistency while being perceived as more enjoyable and helpful than non-embodied wearable alternatives.

---

[EmbodiedLGR: Integrating Lightweight Graph Representation and Retrieval for Semantic-Spatial Memory in Robotic Agents](http://arxiv.org/abs/2604.18271)

- EmbodiedLGR-Agent: introduces a VLM-driven architecture that constructs efficient environment representations using a dual-level memory system comprising a Memory Graph and a Vector Database.
- The framework utilizes Florence-2 VLM and all-MiniLM-L6-v2 to populate memory structures, enabling the LLM Agent to perform targeted retrieval via specialized Retrieval Tools.
- This approach optimizes memory building and retrieval for real-time robotic deployment, significantly reducing response latency compared to monolithic memory bank architectures.

---

[AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](http://arxiv.org/abs/2604.18240)

- AJ-Bench: introduces a comprehensive benchmark for evaluating Agent-as-a-Judge systems across search, data systems, and graphical user interface domains, utilizing Task Design, Trajectory Collection, Label Annotation, and Evaluation Process.
- The framework employs Agent-as-a-Judge, which integrates Tool Box and Env Feedback to perform Information Acquisition, State Verification, and Process Verification through Env Setup & Replay.
- Experiments demonstrate that equipping judge agents with environment interaction capabilities significantly improves evaluation accuracy compared to traditional LLM-as-a-Judge baselines.

---

[Aether: Network Validation Using Agentic AI and Digital Twin](http://arxiv.org/abs/2604.18233)

- Aether: introduces a neuro-symbolic architecture that combines LLM-based agents with a unified Network Digital Twin (NDT) to automate network change validation workflows.
- The system utilizes five specialized agents—Assistant, NDM Query, Impact Assessment, Test Planner, and Test Executor—to perform intent-aware, compositional verification of network changes.
- Aether leverages a temporal Network Digital Map (NDM) and standardized tool interfaces to achieve high error detection and diagnostic coverage in both synthetic and real-world ISP network environments.

---

[AgenTEE: Confidential LLM Agent Execution on Edge Devices](http://arxiv.org/abs/2604.18231)

- AgenTEE: introduces a system for deploying confidential LLM agent pipelines on edge devices by isolating the agent runtime, inference engine, and third-party applications into independently attested confidential virtual machines.
- The framework leverages Arm Confidential Compute Architecture to enforce hardware-level isolation of proprietary assets and runtime state, including model weights and KV-cache, from the host operating system and hypervisor.
- AgenTEE utilizes Confidential Shared Memory to facilitate secure, mutually authenticated communication between isolated components, achieving near-native performance with minimal runtime overhead.

---

[Towards an Agentic LLM-based Approach to Requirement Formalization from Unstructured Specifications](http://arxiv.org/abs/2604.18228)

- Agentic LLM-based Requirement Formalization Pipeline: introduces an agentic methodology that automatically extracts verification-ready properties from unstructured specifications by coordinating multiple LLM-based agents.
- The pipeline integrates requirement extraction, verifiability classification, and formal translation stages to ensure that generated properties are both syntactically and semantically aligned with the target formal model.
- Experimental results demonstrate that the approach achieves high accuracy in generating verifiable properties for cyber-physical systems while effectively filtering out unverifiable constraints.

---

[WebCompass: Towards Multimodal Web Coding Evaluation for Code Language Models](http://arxiv.org/abs/2604.18224)

- WebCompass: introduces a comprehensive, multimodal benchmark for evaluating web engineering capabilities across generation, editing, and repair tasks using LLM-as-a-Judge and Agent-as-a-Judge protocols.
- The framework utilizes a multi-stage, human-in-the-loop pipeline to curate high-quality web prototypes and employs an Agent-as-a-Judge paradigm that leverages Model Context Protocol for autonomous browser interaction and iterative test-case synthesis.
- Experimental results demonstrate that closed-source models significantly outperform open-source alternatives, with visual quality identified as a persistent bottleneck across all evaluated LLMs.

---

[Instruction-as-State: Environment-Guided and State-Conditioned Semantic Understanding for Embodied Navigation](http://arxiv.org/abs/2604.18223)

- S-EGIU: introduces a coarse-to-fine framework that models instruction understanding as a dynamic latent state variable conditioned on the agent's evolving perceptual state.
- The framework utilizes CGIP to activate perceptually relevant sub-instructions and FGIP to perform perception-guided token refinement, ensuring instruction semantics remain aligned with the current visual context.
- By treating instructions as trajectory-evolving states rather than static embeddings, S-EGIU improves navigation efficiency and robustness across diverse VLN benchmarks.

---

[TacticGen: Grounding Adaptable and Scalable Generation of Football Tactics](http://arxiv.org/abs/2604.18210)

- TacticGen: introduces a generative framework for football tactics that models coordinated player movements as multi-agent trajectories conditioned on game context and guided by diverse tactical objectives.
- The framework utilizes a multi-agent Diffusion Transformer (MADiT) backbone, incorporating a context encoder and an event encoder to capture complex inter-agent dependencies and contextual information.
- TacticGen enables adaptable tactic generation at inference time through classifier guidance mechanisms, including rule-based functions, LLM-generated code, and learned value models, without requiring model retraining.

---

[Scalable Neighborhood-Based Multi-Agent Actor-Critic](http://arxiv.org/abs/2604.18190)

- MADDPG-K: introduces a scalable multi-agent reinforcement learning framework that mitigates the computational bottleneck of centralized critics by restricting input to the K-nearest neighbors of each agent.
- The framework utilizes precomputed index sets stored in the replay buffer to maintain constant-size critic inputs, effectively reducing the computational complexity from quadratic to linear with respect to the total agent count.
- Empirical results demonstrate that MADDPG-K achieves faster convergence and superior runtime scaling compared to standard MADDPG in cooperative multi-agent environments.

---

[Multi-Agent Systems: From Classical Paradigms to Large Foundation Model-Enabled Futures](http://arxiv.org/abs/2604.18133)

- MAS: introduces a systematic review and comparative analysis of classical MASs (CMASs) and LFM-based MASs (LMASs), highlighting the paradigm shift from task-specific, model-driven coordination to cognitively empowered, language-mediated collaboration.
- The paper defines a unified analytical framework for LMASs comprising core modules (role definition, perception, planning, memory, execution), interaction mechanisms, and hierarchical optimization layers (model, knowledge, system).
- It identifies future research directions including the co-evolution of CMASs and LMASs, scaling multimodality, causality-enhanced reasoning, and the development of robust device-edge-cloud collaborative architectures.

---

[Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration](http://arxiv.org/abs/2604.18131)

- Meta-Learning-Driven Evolution: introduces a training paradigm that enables LLM agents to autonomously explore environments and distill observations into structured World Knowledge without human-provided rewards.
- The framework utilizes an outcome-based reward mechanism during training to teach agents how to effectively explore and summarize environments, which is then used as an external context module at inference time.
- Experimental results demonstrate that this approach significantly improves performance on web-based benchmarks and enables smaller models to outperform larger, unassisted counterparts through superior environment exploration.

---

[Chatting about Conditional Trajectory Prediction](http://arxiv.org/abs/2604.18126)

- CiT: introduces a trajectory prediction method that models cross-time-domain social interactions to capture temporal intention dynamics for autonomous systems.
- The framework incorporates ego agent potential motion to enable seamless integration with downstream robotic planning and control modules.
- CiT utilizes intention graphs and cross-domain interaction modeling to refine intention representations and improve prediction accuracy across diverse social scenarios.

---

[ConventionPlay: Capability-Limited Training for Robust Ad-Hoc Collaboration](http://arxiv.org/abs/2604.18123)

- ConventionPlay: introduces a reinforcement learning-based approach for ad-hoc collaboration that utilizes a hierarchical population of K0, K1, and K2 agents to enable robust coordination with adaptive partners.
- The framework employs capability-aware stratified sampling to generate diverse K1 followers, forcing the K2 agent to move beyond reactive behavior toward active team steering.
- By probing a partner's repertoire, the K2 agent effectively identifies and converges on the most efficient shared convention, outperforming existing methods in complex, differentiated reward environments.

---

[Sharing the proceeds from a hierarchical venture when agents have needs](http://arxiv.org/abs/2604.18108)

- Need-adjusted geometric rules: introduces a family of allocation mechanisms for hierarchical ventures where agents have individual needs that must be covered by generated revenues.
- The paper characterizes these rules using axioms such as needs lower bound, lowest rank consistency, and highest rank independence to ensure fair distribution.
- It further identifies a need-adjusted serial rule and explores extensions to broader domains where aggregate revenues exceed aggregate needs.

---

[Test-Time Perturbation Learning with Delayed Feedback for Vision-Language-Action Models](http://arxiv.org/abs/2604.18107)

- PDF: introduces a verifier-free test-time adaptation framework that mitigates trajectory overfitting in VLAs by employing Uncertainty-Based Action Voting and Delayed Feedback-Guided Adaptation.
- The framework utilizes a lightweight P Head to retrospectively adjust action logits based on delayed environmental feedback, while keeping the base VLA parameters frozen.
- PDF improves decision performance and robustness across robotic manipulation and visual control tasks by balancing inference efficiency with adaptive data augmentation.

---

[Architectural Design Decisions in AI Agent Harnesses](http://arxiv.org/abs/2604.18071)

- Agent Harness: introduces a protocol-guided, source-grounded empirical study of 70 AI agent-system projects to identify recurring architectural design decisions across Subagent architecture, Context management, Tool systems, Safety mechanisms, and Orchestration.
- The study reveals that Agent harness architecture is organized around coupled decision bundles rather than independent feature selections, with coordination complexity, context persistence, and governance mechanisms often co-evolving.
- The research synthesizes five recurring architectural patterns—Lightweight Tool, Balanced CLI Framework, Multi-Agent Orchestrator, Enterprise Full-Featured, and Scenario-Verticalized—to provide grounded guidance for framework designers and selectors.

---

[EvoMarket: A High-Fidelity and Scalable Financial Market Simulator](http://arxiv.org/abs/2604.18046)

- EvoMarket: introduces a high-fidelity, scalable financial market simulator that integrates a Discrete-event execution core, Hierarchical event scheduler, Multi-asset exchanges, Asset-specific LOB, Agent panel, Oracle-guided self-calibration, and an Observability and logging layer to enable intervention-oriented experiments.
- The framework utilizes an Oracle-guided self-calibration mechanism to reduce microstructure discrepancies within a single simulation run, avoiding the computational overhead of traditional external black-box optimization.
- EvoMarket supports multi-asset, cross-day simulation with institutional mechanisms like price limits and T+1 settlement, achieving high throughput and scalable performance for market-scale research.

---

[First, Do No Harm (With LLMs): Mitigating Racial Bias via Agentic Workflows](http://arxiv.org/abs/2604.18038)

- Agentic Workflow Framework: introduces a multi-stage evaluation and mitigation pipeline for racial bias in LLMs using LLMs, Search Agent, RAG Agent, Vector Knowledge Base, Flowise Platform, and Final DDx Module.
- The study evaluates five LLMs across implicit and explicit bias benchmarks, identifying DeepSeek V3 as the most robust model for clinical decision-making tasks.
- Integrating DeepSeek V3 into an agentic workflow demonstrates measurable improvements in explicit bias metrics by prioritizing structured clinical knowledge over standalone model generation.

---

[Topology-Aware LLM-Driven Social Simulation: A Unified Framework for Efficient and Realistic Agent Dynamics](http://arxiv.org/abs/2604.18011)

- TopoSim: introduces a topology-aware framework that treats network structure as an active driver of agent interactions to improve the efficiency and realism of LLM-based social simulations.
- The framework utilizes an Update Coordination Module to group agents with similar structural contexts for shared inference and a Role-Differentiation Module to model asymmetric influence based on network topology.
- Experimental results demonstrate that TopoSim maintains high behavioral fidelity while reducing LLM token consumption by 50-90% across diverse social simulation datasets.

---

[Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](http://arxiv.org/abs/2604.18005)

- MAS: introduces a systematic empirical study of diversity collapse in multi-agent systems, identifying structural coupling as the primary driver of premature consensus in open-ended idea generation.
- The framework analyzes how interaction structures, including Agent, Leader, Explorer, and Judge roles, inadvertently contract the solution space through authority-driven dynamics and dense communication topologies.
- The research demonstrates that diversity collapse is a structural failure rather than a model-level insufficiency, and proposes interventions like NGT and subgroup isolation to preserve independence and enhance creative output.

---

[AIT Academy: Cultivating the Complete Agent with a Confucian Three-Domain Curriculum](http://arxiv.org/abs/2604.17989)

- AIT Academy: introduces a curriculum framework for cultivating LLMs across three domains—Natural Science, Humanities, and Social Science—using Confucian Six Arts as behavioral archetypes.
- The framework utilizes specialized training grounds—ClawdGO, Athen’s Academy, and Alt Mirage—to provide longitudinal, ecologically valid training for LLMs through ASAT, CSMA, and multi-agent collaboration architectures.
- AIT Academy addresses systematic unevenness in LLM development by moving beyond benchmark-driven training toward a holistic, multi-domain cultivation trajectory that includes security, creative synthesis, and social reasoning.

---

[E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes](http://arxiv.org/abs/2604.17969)

- E3VS-Bench: introduces a benchmark for evaluating active perception in photorealistic 3D environments using 3D Gaussian Splatting Scenes, a VLM-as-a-judge Evaluator, an Embodied Agent, a 5-DoF Action Space, Structured Prompting, and an Episode Filtering Pipeline.
- The framework requires agents to perform 5-DoF viewpoint control to resolve occlusions and acquire fine-grained visual evidence for question answering.
- Evaluation results demonstrate that while current LLMs possess strong 2D recognition capabilities, they exhibit a substantial performance gap compared to humans in active 3D viewpoint planning and exploration.

---

[CADMAS-CTX: Contextual Capability Calibration for Multi-Agent Delegation](http://arxiv.org/abs/2604.17950)

- CADMAS-CTX: introduces a framework for multi-agent delegation that replaces static capability profiles with hierarchical, context-conditioned Beta posteriors to prevent systematic misdelegation.
- The framework utilizes an uncertainty-aware scoring function that combines posterior means with an uncertainty penalty to ensure agents delegate tasks only when a peer's capability is well-supported by evidence.
- By employing locally-centralized task coordination and empirical Bayes shrinkage for cold-start scenarios, the architecture achieves sublinear regret and robust performance across heterogeneous task contexts.

---

[RAVEN: Retrieval-Augmented Vulnerability Exploration Network for Memory Corruption Analysis in User Code and Binary Programs](http://arxiv.org/abs/2604.17948)

- RAVEN: introduces a multi-agent framework that leverages LLMs and RAG to automatically synthesize comprehensive vulnerability analysis reports from vulnerable source code.
- The framework integrates a Data Collection Pipeline, a RAG Engine, and an Agentic System comprising Explorer-, Analyst-, Reporter-, and Judge-agents to automate end-to-end security documentation.
- RAVEN utilizes an LLM-as-a-Judge methodology to evaluate generated reports across structural integrity, factual grounding, code reasoning, and remediation quality dimensions.

---

[ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](http://arxiv.org/abs/2604.17944)

- HIRE-Agent: introduces a hierarchical multi-agent framework that utilizes an Understand–Plan–Execute architecture to integrate heterogeneous evidence from databases and external APIs for complex real estate reasoning.
- The framework employs a Front-end Agent for intent parsing, a Supervisor Agent for task orchestration, and specialized Database Interaction- and Map Reasoning-agents to handle multi-step queries.
- HIRE-Agent serves as a strong baseline for the ReCoQA benchmark, demonstrating that hierarchical collaboration is essential for solving complex, multi-source reasoning tasks in vertical domains.

---

[ContraPrompt: Contrastive Prompt Optimization via Dyadic Reasoning Trace Analysis](http://arxiv.org/abs/2604.17937)

- ContraPrompt: introduces a prompt optimization framework that leverages dyadic reasoning trace analysis to extract transferable rules from paired failure-to-success execution traces.
- The system utilizes an instrumented agentic retry loop to generate contrastive data, which is then organized into an input-aware decision tree to provide context-specific instructions.
- ContraPrompt outperforms existing baselines by targeting specific reasoning steps identified through the comparison of complete chain-of-thought traces rather than final outputs.

---

[Robust Distributed Sub-Optimal Coordination of Linear Agents with Uncertain Input Nonlinearities](http://arxiv.org/abs/2604.17934)

- RDSC framework: introduces a robust control approach for multi-agent systems to achieve bounded convergence to a global optimizer despite time-varying input nonlinearities.
- The framework utilizes a dynamic protocol that incorporates local communication and sector-bounded uncertainty modeling to ensure input-to-state stability.
- Sufficient conditions for the solvability of the coordination problem are derived and characterized through linear matrix inequalities, ensuring robust performance in the presence of agent-specific uncertainties.

---

[LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent](http://arxiv.org/abs/2604.17931)

- LiteResearcher: introduces a scalable agentic RL training framework that constructs an isolated virtual world to mirror real-world search dynamics, utilizing a Data Synthesis Pipeline, Local Search Engine, Local Browse Tool, Difficulty-Aware Curriculum Learning, GRPO, and an LLM-based Reward Judge.
- The framework decouples agent training from the open web by co-constructing training data and a local corpus, enabling stable, low-latency, and cost-effective RL training for deep research agents.
- By employing difficulty-aware curriculum learning and on-policy GRPO, the framework effectively eliminates repetitive action loops and sustains performance improvements beyond the saturation points of traditional RL approaches.

---

[Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions](http://arxiv.org/abs/2604.17894)

- SlideAgent: introduces an agent-based framework for dynamic slide updates that integrates multimodal slide parsing, instruction grounding, and tool-augmented reasoning to maintain content consistency.
- The framework utilizes a two-stage architecture comprising slide understanding for hierarchical representation and instruction-driven content synchronization for data-consistent updates.
- The authors also release DynaSlide, a large-scale benchmark with over 20,000 instruction-execution triples grounded in a shared external database to evaluate automated slide update performance.

---

[Latent Preference Modeling for Cross-Session Personalized Tool Calling](http://arxiv.org/abs/2604.17886)

- PREFINE: introduces a test-time memory-augmented method that represents user preferences as evolving hypotheses to improve personalized tool calling.
- The framework utilizes a generate–verify–refine loop to extract reusable constraints from interaction history, enabling the agent to fill under-specified API arguments.
- PREFINE significantly reduces token usage compared to full-history prompting while maintaining effectiveness under dynamic tool schemas.

---

[Scaling Human-AI Coding Collaboration Requires a Governable Consensus Layer](http://arxiv.org/abs/2604.17883)

- Agentic Consensus: introduces a paradigm that replaces opaque AI-generated code with a governable consensus layer (C) as the primary engineering artifact, mediating between human intent (I) and executable artifacts (A) via synchronization operators (Φ, Ψ) and evidence-linked validation (E).
- The framework utilizes specialized agents—Architect, Builder, Auditor, and Navigator—to maintain structural integrity and provide auditable, evidence-based control over AI-assisted software development.
- By treating structural commitments as a first-class, queryable property graph, the approach mitigates dimension collapse and enables human-gated review of structural changes rather than just code diffs.

---

[Design and Evaluation of a Culturally Adapted Multimodal Virtual Agent for PTSD Screening](http://arxiv.org/abs/2604.17871)

- Molhim: introduces a multimodal conversational AI platform designed for culturally adapted PTSD screening in military healthcare settings.
- The system integrates ASR, VLM, LLM, and TTS components to facilitate structured, safe, and purpose-specific dialogues through a high-fidelity virtual avatar.
- The platform employs a state-based dialogue controller to ensure clinical coherence, safety-aware interaction, and automated post-session analysis for mental health support.

---

[TitanCA: Lessons from Orchestrating LLM Agents to Discover 100+ CVEs](http://arxiv.org/abs/2604.17860)

- TitanCA: introduces a four-module pipeline that orchestrates multiple LLM-powered agents to perform precise, layered software vulnerability discovery.
- The architecture utilizes a Matcher for clone detection, a Filter for reasoning-based screening, an Inspector for multi-agent deliberation, and an Adapter for domain-specific refinement.
- By prioritizing precision through successive filtering stages, the system effectively reduces false positives and improves the reliability of automated vulnerability detection in production environments.

---

[On the Reliability of Computer Use Agents](http://arxiv.org/abs/2604.17849)

- CUA: introduces a framework for evaluating the reliability of computer-use agents across repeated executions by decomposing task performance into stochasticity, instruction ambiguity, and planning variability.
- The framework utilizes POMDP to model task execution and employs metrics like Pass^k, McNemar tests, and Wilcoxon signed-rank tests to quantify consistency and identify reliability regressions.
- The study incorporates interventions such as deterministic decoding, instruction clarification, and iterative plan refinement to mitigate unreliability and improve agent performance across diverse computer environments.

---

[Learning from AVA: Early Lessons from a Curated and Trustworthy Generative AI for Policy and Development Research](http://arxiv.org/abs/2604.17843)

- AVA (AI + Verified Analysis): introduces a multi-agent, domain-bounded RAG system that operationalizes epistemic humility through reasoned abstention and verifiable, page-anchored citations for policy research.
- The architecture utilizes a hierarchical RAG database and a multi-agent pipeline, including Query Decomposer-, Retrieval Planner-, Tree Walker-, and Drafting-agents, to ensure evidence-grounded synthesis.
- The system incorporates a verification model that audits generated claims against retrieved evidence, triggering reasoned abstention when support is insufficient to maintain trust and accuracy.

---

[WebUncertainty: Dual-Level Uncertainty Driven Planning and Reasoning For Autonomous Web Agent](http://arxiv.org/abs/2604.17821)

- WebUncertainty: introduces a hierarchical framework for autonomous web agents that mitigates dual-level uncertainty in planning and reasoning to improve performance in complex, long-horizon tasks.
- The framework utilizes a Task Uncertainty-Driven Adaptive Planning Mechanism to switch between explicit and implicit planning modes based on environmental complexity.
- It incorporates an Action Uncertainty-Driven MCTS Reasoning Mechanism with the ConActU strategy to quantify aleatoric and epistemic uncertainty, effectively pruning hallucinated actions and guiding robust decision-making.

---

[Do LLMs Need to See Everything? A Benchmark and Study of Failures in LLM-driven Smartphone Automation using Screentext vs. Screenshots](http://arxiv.org/abs/2604.17817)

- DailyDroid: introduces a benchmark for evaluating mobile agents, comprising 75 tasks across 25 Android apps to systematically analyze failure modes in LLM-driven smartphone automation.
- The study compares text-only (screentext) and multimodal (screentext + screenshot) input modalities across GPT-4o and o4-mini models to assess their impact on task success, efficiency, and cost.
- Findings reveal that while multimodal inputs provide marginal success gains, they incur significantly higher costs, highlighting critical challenges in UI accessibility and the need for more robust mobile agent design.

---

[Memory Centric Power Allocation for Multi-Agent Embodied Question Answering](http://arxiv.org/abs/2604.17810)

- MCPA (Memory Centric Power Allocation): introduces a framework for multi-agent embodied question answering that optimizes power allocation based on a novel quality of memory model to maximize retrieval accuracy.
- The framework utilizes a GAE (Generative Adversarial Exam) pipeline to evaluate the semantic richness of distributed robot memories through forward simulation with VLM and LLM components.
- By prioritizing transmissions from robots with higher memory quality, the system achieves superior EQA performance compared to traditional sensing- or communication-centric resource management methods.

---

[Spatial dynamic modelling to understand how dendritic cell clustering affects T cell activation](http://arxiv.org/abs/2604.17786)

- ABM and PS-PDE framework: introduces a dual-modelling approach to investigate how spatial clustering of dendritic cells influences T cell activation dynamics within lymph nodes.
- The research utilizes a discrete agent-based model and a derived continuum phenotype-structured partial differential equation to quantify T cell stimulation distributions based on dendritic cell topology.
- Findings indicate that while dendritic cell clustering enhances the heterogeneity of T cell activation, it serves as a secondary driver compared to intrinsic T cell characteristics like stimulation uptake and decay rates.

---

[Prompt Optimization Enables Stable Algorithmic Collusion in LLM Agents](http://arxiv.org/abs/2604.17774)

- Prompt Optimization Enables Stable Algorithmic Collusion in LLM Agents: introduces a meta-learning loop where LLM agents participate in market simulations and a reflective LLM meta-optimizer iteratively refines a shared meta-prompt to discover stable tacit collusion strategies.
- The framework utilizes a nested logit demand model to simulate duopoly markets where LLM agents maintain history and self-notes to inform pricing decisions while optimizing for aggregate profit.
- Experimental results demonstrate that meta-prompt optimization significantly improves coordination quality and enables agents to generalize collusive behaviors to held-out market configurations.

---

[Efficient Federated RLHF via Zeroth-Order Policy Optimization](http://arxiv.org/abs/2604.17747)

- Par-S2ZPO (Partitioned, Sign-based Stochastic Zeroth-order Policy Optimization): introduces a communication-efficient federated RLHF framework that utilizes zeroth-order optimization with binary perturbations and parameter partitioning to minimize resource consumption on edge devices.
- The framework partitions actor network parameters into K subsets, allowing each agent to perform local policy evaluation and communicate only binary preference feedback to the central server.
- Theoretical analysis establishes that the algorithm achieves convergence rates independent of the number of agents, demonstrating efficiency comparable to centralized systems while significantly reducing communication, computation, and memory overhead.

---

[HiRAS: A Hierarchical Multi-Agent Framework for Paper-to-Code Generation and Execution](http://arxiv.org/abs/2604.17745)

- HiRAS (Hierarchical Research Agent System): introduces a hierarchical multi-agent framework for automated experiment reproduction that employs supervisory manager agents to coordinate specialized agents across fine-grained stages.
- The framework utilizes a tree-structured orchestration where manager agents actively inspect intermediate artifacts in a shared workspace to diagnose failures and dynamically re-invoke subordinate agents.
- The authors also introduce Paper2Code-Extra (P2C-Ex), a refined reference-free evaluation protocol that incorporates repository-level information to mitigate evaluator hallucination and improve alignment with reference-based metrics.

---

[Tool Learning Needs Nothing More Than A Free 8B Language Model](http://arxiv.org/abs/2604.17739)

- TRUSTEE: introduces a data-free method for training tool-calling agents by simulating the entire environment using a local open-source LLM.
- The framework integrates a Query Generator, Tool Simulator, User Simulator, and Verifier Simulator to create dynamic, interactive training environments without requiring external annotated data.
- An Adaptive Curriculum Learning mechanism dynamically modulates task difficulty based on agent performance to ensure stable and efficient reinforcement learning.

---

[Co-evolving Agent Architectures and Interpretable Reasoning for Automated Optimization](http://arxiv.org/abs/2604.17708)

- EvoOR-Agent: introduces a co-evolutionary framework that treats agent architecture and reasoning trajectories as evolvable objects to automate operations research workflows.
- The framework utilizes an AOE-style architecture graph to represent and optimize agent workflows, enabling adaptive coordination of problem interpretation, mathematical formulation, and solver execution.
- By coupling architecture evolution with reasoning-trajectory search and knowledge-base-assisted operators, the system achieves superior performance and structural interpretability compared to fixed-pipeline LLM agents.

---

[SelfHeal: Empirical Fix Pattern Analysis and Bug Repair in LLM Agents](http://arxiv.org/abs/2604.17699)

- SelfHeal: introduces a multi-agent system for automated bug repair in LLM agents, utilizing a Fix Agent and a Critic Agent empowered by internal fix rules and external web search.
- The framework employs an iterative ReAct-based cycle where the Fix Agent proposes repairs and the Critic Agent validates them using specialized tools for API and format verification.
- The research also presents AgentDefect, a benchmark dataset of 37 runtime buggy instances, and provides an empirical analysis of 23 distinct fix patterns observed in LLM agent development.

---

[CAPO: Counterfactual Credit Assignment in Sequential Cooperative Teams](http://arxiv.org/abs/2604.17693)

- CAPO: introduces a critic-free policy-gradient algorithm for sequential cooperative teams that utilizes Additive reward decomposition (models expected team reward), Ridge regression estimator (recovers per-agent components), Upstream-cancellation identity (simplifies advantage target), Fictitious sampling (estimates indirect effect), and PPO-clipped updates (optimizes agent policies) to perform efficient credit assignment.
- The framework addresses sequential-update non-stationarity by decomposing the advantage into a closed-form direct effect and a fictitious-sample indirect effect, avoiding the exponential variance scaling of cumulative importance-sampling methods.
- CAPO demonstrates superior performance and scalability in sequential multi-agent settings by adapting its variance to the coupling structure of the team, remaining effective even as team size increases.

---

[How Adversarial Environments Mislead Agentic AI?](http://arxiv.org/abs/2604.18874)

- POTEMKIN: introduces a threat model called Adversarial Environmental Injection (AEI) where adversaries compromise tool outputs to deceive LLM agents through breadth-based epistemic drift and depth-based navigational policy collapse.
- The research identifies a Robustness Schism, demonstrating that resistance to content poisoning in RAG systems does not correlate with resistance to structural navigational traps in citation graphs.
- The study reveals that LLM agents exhibit a Punishment of Honesty, where they penalize hedged scientific claims while failing to detect falsehoods, and provides an open-source evaluation harness to test these vulnerabilities.

---

[Global Product Intersection Sets in Semigroups](http://arxiv.org/abs/2604.18869)

- Aristotle: introduces a formal verification approach to classify product intersection sets in semigroups by leveraging automated theorem proving and Lean-based formalization.
- The research provides a complete classification of global sets HQ and H*N, demonstrating that any subset of natural numbers containing 1 can be realized as a product intersection set.
- The paper highlights the capability of the Aristotle agent to autonomously discover proofs and resolve open mathematical problems originally posed by Nathanson.

---

[Temporal UI State Inconsistency in Desktop GUI Agents: Formalizing and Defending Against TOCTOU Attacks on Computer-Use Agents](http://arxiv.org/abs/2604.18860)

- PUSV (Pre-execution UI State Verification): introduces a layered middleware defense that re-verifies the UI state immediately before action dispatch to mitigate TOCTOU vulnerabilities in desktop GUI agents.
- The framework utilizes Layer 1 (Masked Pixel SSIM), Layer 2a (Global Screenshot Diff), and Layer 2b (X Window Registry Diff) to detect malicious UI state changes during the observation-to-action gap.
- This research formalizes Visual Atomicity Violations and demonstrates that while PUSV effectively intercepts OS-level attacks, it remains blind to zero-visual-footprint DOM injections, necessitating future application-layer verification.

---

[The Triadic Loop: A Framework for Negotiating Alignment in AI Co-hosted Livestreaming](http://arxiv.org/abs/2604.18850)

- Triadic Loop Framework: introduces a conceptual model for multi-party alignment in AI-augmented livestreaming by framing interaction as a temporally reinforced process of bidirectional adaptation among the streamer, AI co-host, and audience.
- The framework identifies three interdependent sub-loops—Performative Steering, Affective Synchrony, and Community Mediation—where misalignment in any single relationship can destabilize the entire socioemotional system.
- It proposes "strategic misalignment" as a mechanism for sustaining community engagement and suggests dynamic, temporally grounded evaluation metrics to assess AI co-host performance beyond static instruction-following.

---

[Consensus and flocking with transmission and reaction delays](http://arxiv.org/abs/2604.18848)

- Consensus and flocking with transmission and reaction delays: introduces a mathematical framework for analyzing collective behavior in multi-agent systems subject to distinct transmission and reaction delays.
- The paper derives sufficient conditions for asymptotic consensus and flocking using a Lyapunov-Krasovskii functional approach combined with a Halanay-type inequality.
- The analysis provides rigorous stability criteria for both first-order consensus models and second-order Cucker-Smale flocking models under non-negligible communication and processing lags.

---

[Human-Guided Harm Recovery for Computer Use Agents](http://arxiv.org/abs/2604.18847)

- BACKBENCH (Benchmark for Computer Use Agent Harm Recovery): introduces a generate-and-verify framework that utilizes LMgen, LMver, and a Reward Model to steer agents from harmful states back to safe states in alignment with human preferences.
- The framework operationalizes human-centered recovery by training a reward model on pairwise preference judgments to rerank candidate plans generated by an LLM.
- Empirical results demonstrate that preference-guided scaffolds significantly outperform base LLMs in recovery tasks, particularly in resource-constrained scenarios.

---

[AI scientists produce results without reasoning scientifically](http://arxiv.org/abs/2604.18805)

- Corral: introduces a systematic evaluation framework to decompose the contributions of base LLMs and agent scaffolds in scientific research tasks.
- The framework evaluates LLM-based agents across eight scientific domains using mechanistic performance analysis and behavioral analysis of reasoning traces.
- Findings indicate that base LLMs are the primary determinant of performance, while scaffold engineering provides minimal improvement in epistemic reasoning patterns.

---

[MANGO: Multi-Agent Web Navigation via Global-View Optimization](http://arxiv.org/abs/2604.18779)

- MANGO: introduces a web navigation framework that leverages global website structure to identify intent-related entry points and optimize navigation efficiency under limited budgets.
- The framework models URL selection as a multi-armed bandit problem using Thompson Sampling to dynamically prioritize promising URLs based on navigation history and reflection feedback.
- MANGO integrates an episodic memory module to store navigation trajectories and reflections, preventing redundant exploration and improving performance on complex, long-horizon web tasks.

---

[CHICO-Agent: An LLM Agent for the Cross-layer Optimization of 2.5D and 3D Chiplet-based Systems](http://arxiv.org/abs/2604.18764)

- CHICO-Agent: introduces a hierarchical multi-agent framework that utilizes LLM reasoning to perform cross-layer design space exploration for 2.5D and 3D chiplet-based systems by iteratively analyzing historical PPAC evaluations.
- The framework employs an Admin Agent to generate exploration plans and multiple Field Agents to evaluate these plans in parallel, utilizing both a Persistent Context for domain constraints and an Evolving Context for iterative learning.
- By replacing stochastic metaheuristics with a reasoning-driven loop, CHICO-Agent achieves lower system costs and provides interpretable rationales for design decisions while significantly reducing the hyperparameter tuning burden.

---

[Opinion polarization from compression-based decision making where agents optimize local complexity and global simplicity](http://arxiv.org/abs/2604.18755)

- Agent-based model (ABM): introduces a framework where agents optimize the ratio of local-to-global Shannon entropy to balance the desire for local distinctiveness with the tendency for global cognitive compression.
- The model utilizes cognitive compression via opinion binning to simulate how individuals simplify complex social environments into manageable information chunks.
- Computational experiments demonstrate that moderate local group sizes, consistent with Dunbar’s number, facilitate the emergence of polarized opinion clusters with sustained internal variability.

---

[A Scientific Human-Agent Reproduction Pipeline](http://arxiv.org/abs/2604.18752)

- SHARP: introduces a structured framework for reproducing scientific data analyses by decomposing tasks into autonomous steps executed by specialized subagents under human supervision.
- The framework utilizes Claude Code as the primary LLM-based agent, which collaborates with human researchers to translate scientific papers into machine-readable codebases through iterative checkpoints.
- SHARP incorporates specialized subagents for analysis, testing, and quality assurance, while leveraging the FlexCAST principles and the law workflow engine to ensure modularity and reproducibility in scientific research.

---

[Autonomous Skeletal Landmark Localization towards Agentic C-Arm Control](http://arxiv.org/abs/2604.18740)

- Agentic C-arm control framework: introduces a fine-tuning approach for MLLMs to perform autonomous skeletal landmark localization and C-arm navigation using anatomical spatial grounding.
- The framework utilizes LoRA and Unsloth for efficient fine-tuning of MLLMs on synthetic X-ray datasets generated by DeepDRR to enable context-aware landmark prediction.
- The system employs a multi-step perception-action loop that allows the MLLM to reason about spatial relationships, incorporate clinician feedback, and iteratively adjust C-arm positioning.

---

[Towards Optimal Agentic Architectures for Offensive Security Tasks](http://arxiv.org/abs/2604.18718)

- Agentic Security Architectures: introduces a controlled benchmark of 20 interactive targets to evaluate five distinct agentic coordination topologies across whitebox and blackbox security auditing modes.
- The study compares SAS, MAS-Indep, MAS-Decent, MAS-Central, and MAS-Hybrid, utilizing Sandbox Agent and Validator Agent components to assess performance under matched budgets and constraints.
- Results demonstrate a non-monotonic cost-quality frontier where broader coordination does not consistently outperform simpler architectures, highlighting the importance of selective routing based on task observability and domain.

---

[Characterizing AlphaEarth Embedding Geometry for Agentic Environmental Reasoning](http://arxiv.org/abs/2604.18715)

- Agentic Geospatial Intelligence System: introduces an agentic framework that leverages the non-Euclidean geometric structure of AlphaEarth embeddings to improve environmental reasoning through geometry-aware tools and multi-step planning.
- The system integrates AlphaEarth Embeddings (64-dimensional land surface representations) with a FAISS-indexed Database (efficient k-nearest neighbor search) and a ReAct-style Planning Architecture (iterative reasoning and tool invocation) to perform complex environmental queries.
- The architecture incorporates Retrieval Tools (five deterministic environmental data access functions) and Geometry-Aware Tools (four functions utilizing manifold geometric metadata), managed by a System Model (LLM for planning and response synthesis) and evaluated by a Judge Model (LLM for performance evaluation).

---

[APRVOS: 1st Place Winner of 5th PVUW MeViS-Audio Track](http://arxiv.org/abs/2604.18665)

- APRVOS: introduces a staged pipeline for audio-conditioned referring video object segmentation that decouples speech transcription, visual existence verification, coarse segmentation, and agentic refinement.
- The framework utilizes VibeVoice-ASR for transcription, a Qwen3-VL-based Judger for existence verification, Sa2VA for initial mask generation, and an agentic layer with SAM3 for boundary refinement.
- By explicitly addressing speech-recognition noise and visual-existence uncertainty, the method achieves robust performance on the MeViS-Audio benchmark compared to single-pass models.

---

[Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](http://arxiv.org/abs/2604.18660)

- Adversarial Tutoring Robustness Evaluation Framework: introduces a systematic approach to evaluate LLM-based tutors against adversarial student attacks using Tutor Agent, Adversarial Student Agent, Judge Agent, Refiner Agent, Memory, and Dataset.
- The framework utilizes a fine-tuned adversarial student agent to simulate multi-turn dialogues, effectively probing tutor robustness against answer leakage.
- Experimental results demonstrate that pedagogical alignment and defense strategies like reasoning-based tutors and multi-agent setups significantly improve robustness against adversarial student behavior.

---

[Owner-Harm: A Missing Threat Model for AI Agent Safety](http://arxiv.org/abs/2604.18658)

- Nous: introduces a four-layer compositional runtime safety gate designed to mitigate owner-harm by integrating L1 (Encodes owner-policy constraints), L2 (Routes trivially benign actions), L3 (LLM evaluator for semantic reasoning), and L4 (Deterministic audit of artifacts).
- The framework addresses the systematic blind spot in existing safety benchmarks where agents harm their own deployers by failing to account for resource ownership, trust boundaries, and authorization scope.
- Experimental results demonstrate that combining semantic gate reasoning with deterministic post-audit verification significantly improves detection of complex threats like hijacking, achieving 93.3% effectiveness.

---

[From Craft to Kernel: A Governance-First Execution Architecture and Semantic ISA for Agentic Computers](http://arxiv.org/abs/2604.18652)

- Arbiter-K: introduces a governance-first execution architecture that encapsulates an untrusted LLM-based Probabilistic Processing Unit within a deterministic Symbolic Governor to enforce security as a microarchitectural property.
- The framework utilizes a Semantic ISA to reify LLM outputs into discrete instructions, enabling the kernel to maintain a Security Context Registry and construct an Instruction Dependency Graph for active taint propagation.
- By mediating all environment-impacting operations through a trusted kernel, the architecture prevents semantic injection attacks and enables autonomous execution correction via policy feedback loops.

---

#### 19th April 2026

[Towards Self-Improving Error Diagnosis in Multi-Agent Systems](http://arxiv.org/abs/2604.17658)

- ERRORPROBE: introduces a self-improving framework for semantic failure attribution in LLMs-based Multi-Agent Systems that localizes responsible agents and originating error steps.
- The framework utilizes MAST-Guided Structural Decomposition, Symptom-Driven Backward Tracing, and a Multi-Agent Diagnosis Team consisting of a Strategist-, Investigator- and Arbiter-agent to perform causal analysis.
- A Verified Episodic Memory module, governed by a strict Verification Gate, enables the system to learn from past failures and improve diagnostic precision across domains without retraining.

---

[Poly-EPO: Training Exploratory Reasoning Models](http://arxiv.org/abs/2604.17654)

- Poly-EPO: introduces a framework for post-training LLMs that optimizes a polychromic objective to synergistically balance exploration and exploitation through set RL.
- The framework utilizes a marginal set advantage function to assign credit to individual generations based on their contribution to the collective reward and diversity of sampled sets.
- By employing an LM-judge to cluster reasoning strategies, Poly-EPO enables scalable, optimistic exploration that improves generalization and performance with test-time compute.

---

[Phase-Scheduled Multi-Agent Systems for Token-Efficient Coordination](http://arxiv.org/abs/2604.17400)

- PSMAS: introduces a continuous control framework for multi-agent LLMs that replaces discrete scheduling with a circular manifold-based sweep signal to manage agent activation and context consumption.
- The framework utilizes a PhaseScheduler to assign agents specific angular positions on a circular manifold, ensuring that only agents within a defined sweep window receive full context while others receive compressed summaries.
- PSMAS achieves significant token reduction by decoupling scheduling from context compression, providing stability guarantees and convergence proofs for multi-agent LLM coordination.

---



[Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems](http://arxiv.org/abs/2604.17487)

- CSS (Compositional Selective Specificity): introduces a post-generation layer that mitigates overcommitment in LLMs by decomposing responses into claims and selecting the most specific admissible formulation for each.
- The framework utilizes a calibrated selector to balance support precision and specificity retention, effectively converting uncertainty into a structured semantic backoff policy.
- By operating on fixed drafts, the method provides a modular uncertainty interface that allows downstream agentic components to handle claims based on their verified granularity.

---

[Shepherding UAV Swarm with Action Prediction Based on Movement Constraints](http://arxiv.org/abs/2604.17189)

- DWA-inspired Shepherding Framework: introduces a control method for UAV swarms that predicts future swarm behavior under motion constraints to optimize navigator agent actions.
- The framework utilizes DBSCAN clustering to manage dispersed autonomous agents and switches between collection and guidance modes to steer the swarm toward a target.
- Navigator agents generate feasible motion candidates using a DWA-inspired approach, evaluating them based on velocity, positioning, observation maintenance, and safety criteria.

---


[PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](http://arxiv.org/abs/2604.17653)

- PV-SQL: introduces an agentic framework that improves text-to-SQL performance by synergizing database probing to enrich context and rule-based verification to ensure semantic constraint satisfaction.
- The framework utilizes an Agent to iteratively execute Probe queries for discovering database content and employs a Verify component to enforce constraints through iterative Repair cycles.
- By grounding LLMs in actual database values and using deterministic rule-based verification, PV-SQL reduces common errors like database misinterpretation and synthesis failures while maintaining high efficiency.

---

[Infrastructure-Centric World Models: Bridging Temporal Depth and Spatial Breadth for Roadside Perception](http://arxiv.org/abs/2604.17651)

- I-WM (Infrastructure-centric World Models): introduces a dual-layer architecture that leverages fixed roadside sensors to provide temporal depth for traffic ecosystem modeling, utilizing Modular Perception Layer, Generative World Model Layer, FRGB3D, MulDet3D, MulTrack3D, 4D Occupancy Forecasting, Physics-Informed Latent Dynamics, and I-VLA.
- The framework separates annotation-free perception from generative world modeling to enable scalable deployment and high-fidelity simulation of traffic dynamics at fixed intersections.
- By integrating multi-modal sensor inputs with physics-informed latent dynamics, the approach facilitates multi-agent counterfactual reasoning and V2X cooperative alignment for intelligent infrastructure.

---

[WhatIf: Interactive Exploration of LLM-Powered Social Simulations for Policy Reasoning](http://arxiv.org/abs/2604.17615)

- WhatIf: introduces an interactive system for real-time exploration of LLM-powered social simulations, utilizing an LLM decision layer, a deterministic spatial engine, a unified intervention engine, a collaborative session, snapshot-based replay, an AgentIndex, a GroupDestinationTracker, and CoordinatorZones.
- The system enables policymakers to steer, inspect, and compare large-scale simulations by combining LLM-driven agent deliberation with physics-based world updates.
- WhatIf supports iterative branching and collaborative sensemaking, allowing experts to ground policy decisions in inspectable agent-level rationales rather than aggregate outputs.

---

[Provable Coordination for LLM Agents via Message Sequence Charts](http://arxiv.org/abs/2604.17612)

- ZipperGen: introduces a domain-specific language based on Message Sequence Charts to specify multi-agent LLM coordination with formal deadlock-freedom guarantees.
- The framework employs a syntax-directed projection to derive deadlock-free local agent programs from global specifications, ensuring structural correctness despite the stochastic nature of LLMs.
- ZipperGen supports runtime planning where an LLM dynamically generates coordination workflows, which are then validated and projected to maintain the same structural guarantees as static programs.

---

[Agents Explore but Agents Ignore: LLMs Lack Environmental Curiosity](http://arxiv.org/abs/2604.17609)

- Environmental Curiosity Framework: introduces a methodology to evaluate whether LLMs recognize and investigate unexpected but relevant information within their environment.
- The framework utilizes solution injection to measure the gap between an agent's discovery of relevant information and its subsequent interaction with that information.
- Experimental results across three benchmarks demonstrate that while LLMs consistently discover injected solutions, they systematically fail to integrate them into their reasoning or strategy.

---

[Terminal Wrench: A Dataset of 331 Reward-Hackable Environments and 3,632 Exploit Trajectories](http://arxiv.org/abs/2604.17596)

- Terminal Wrench: introduces a benchmark dataset of 331 reward-hackable terminal environments and 3,632 exploit trajectories to facilitate research on LLM reward hacking and oversight.
- The framework utilizes an Attacker Agent to identify vulnerabilities in Verifier logic, producing diverse exploits ranging from output spoofing to binary hijacking.
- Experimental results demonstrate that an LLM Judge effectively detects exploits using full chain-of-thought reasoning, but detection performance significantly degrades when reasoning traces are sanitized or removed.

---

[Beyond Static Snapshots: A Grounded Evaluation Framework for Language Models at the Agentic Frontier](http://arxiv.org/abs/2604.17573)

- ISOPRO: introduces a simulation-based evaluation and fine-tuning framework that replaces learned reward models with a deterministic verifier to eliminate reward hacking by construction.
- The framework utilizes an AI agent that generates reasoning traces, which are filtered by a deterministic verifier and stored in an implicit-curriculum replay buffer to guide model improvement.
- By employing CPU-updatable LoRA adapters and continuous evaluation, the system reduces hardware requirements and enables the observation of capability emergence in agentic tasks.

---

[SafeAgent: A Runtime Protection Architecture for Agentic Systems](http://arxiv.org/abs/2604.17562)

- SafeAgent: introduces a runtime security architecture for agentic systems that treats safety as a stateful decision problem over evolving interaction trajectories, utilizing a SafeAgent Core and a Runtime Controller.
- The architecture separates execution governance from semantic risk reasoning, employing a World Model/Reasoning LLM, Advantage-Cost Module, and Risk Encoder to mediate agent actions and persistent session state.
- SafeAgent provides robust protection against prompt injection and workflow-level attacks by enforcing controlled execution boundaries and stateful intervention through a Tool Wrapper and synchronized memory.

---

[Causal-Temporal Event Graphs: A Formal Model for Recursive Agent Execution Traces](http://arxiv.org/abs/2604.17557)

- CTEG: introduces a formal model for capturing recursive agent execution traces as rooted arborescences under single-parenthood causal semantics.
- The framework utilizes a grafting operation to compose local agent execution traces into a globally well-formed structure without requiring centralized coordination.
- The model supports tamper-evident session verification via Merkle tree commitments and provides a robust basis for relational database encoding of agentic workflows.

---

[COSEARCH: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search](http://arxiv.org/abs/2604.17555)

- COSEARCH: introduces a framework that jointly trains a multi-step reasoning agent and a generative document ranker using reinforcement learning to overcome the retrieval bottleneck in agentic search.
- The framework utilizes a semantic grouping strategy to enable Group Relative Policy Optimization (GRPO) for the ranker by clustering semantically equivalent sub-queries across different reasoning trajectories.
- A composite reward function provides both immediate ranking-quality feedback and long-term trajectory-level signals, ensuring the ranker and the LLM-based reasoning agent learn to complement each other effectively.

---

[From Admission to Invariants: Measuring Deviation in Delegated Agent Systems](http://arxiv.org/abs/2604.17517)

- IML: introduces a monitoring layer that restores observability of behavioral drift by comparing current agent trajectories against a frozen admission-time snapshot.
- The framework addresses the structural blindness of local enforcement mechanisms by quantifying deviation from the admissible behavior space using a weighted combination of temporal, constraint, and lineage metrics.
- The paper proves that admission-time behavioral contracts are non-identifiable from local enforcement signals alone, necessitating the IML component to detect gradual distributional shifts.

---

[Atomic Decision Boundaries: A Structural Requirement for Guaranteeing Execution-Time Admissibility in Autonomous Systems](http://arxiv.org/abs/2604.17511)

- ADB (Atomic Decision Boundary): introduces a formal structural requirement for governance systems to ensure admissibility by coupling policy evaluation and state transition into a single indivisible step.
- The paper proves that split evaluation systems, which separate decision and execution, are inherently vulnerable to state changes by concurrent environment actions, leading to inadmissible transitions.
- The framework defines a three-valued decision domain (Allow, Refuse, Escalate) and establishes that escalation outcomes require atomic resolution to maintain system-wide admissibility guarantees.

---

[SkillGraph: Self-Evolving Multi-Agent Collaboration with Multimodal Graph Topology](http://arxiv.org/abs/2604.17503)

- SkillGraph: introduces a unified framework that enables the simultaneous co-evolution of agent expertise and communication topology in a closed loop.
- The framework utilizes a Multimodal Graph Transformer (MMGT) to predict dynamic, content-aware collaboration graphs while a Skill Designer continuously refines agent skills based on accumulated failure logs.
- This co-evolutionary design ensures that agent capabilities and communication structures are mutually reinforcing, leading to improved performance across diverse multimodal reasoning tasks.

---

[Towards Shutdownable Agents: Generalizing Stochastic Choice in RL Agents and LLMs](http://arxiv.org/abs/2604.17502)

- DReST: introduces a reward function designed to train agents to be NEUTRAL regarding trajectory-lengths and USEFUL in goal attainment, thereby mitigating shutdown resistance.
- The framework utilizes DReST to train deep RL agents and fine-tune LLMs, ensuring they satisfy Preferences Only Between Same-Length Trajectories (POST) by incentivizing stochastic choices between different trajectory-lengths.
- Experimental results demonstrate that DReST agents achieve high NEUTRALITY and USEFULNESS, with deep RL agents showing improved generalization and LLMs matching baseline performance in held-out environments.

---

[AUTOVQA-G: SELF-IMPROVING AGENTIC FRAMEWORK FOR AUTOMATED VISUAL QUESTION ANSWERING AND GROUNDING ANNOTATION](http://arxiv.org/abs/2604.17488)

- AutoVQA-G: introduces a self-improving agentic framework that automates the creation of high-fidelity VQA-G datasets through an iterative generate-evaluate-refine loop.
- The framework utilizes a Consistency Evaluation module with CoT reasoning to provide interpretable feedback and a memory-augmented Prompt Optimization agent to iteratively refine generation rubrics.
- AutoVQA-G outperforms leading LLMs in visual grounding accuracy and consistency by systematically addressing hallucinations and brittle verification mechanisms.

---

[Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories for Grounded Visual Perception](http://arxiv.org/abs/2604.17475)

- SPECTRA: introduces a supervision-free framework that optimizes SVLM agentic trajectories using cold-start reinforcement learning and structured multi-turn rollouts to enhance grounded visual perception.
- The framework employs a multi-objective reward signal to maximize task correctness, structural integrity, and tool utility, enabling agents to self-discover robust behaviors without human-labeled data.
- SPECTRA further introduces the Tool Instrumental Utility (TIU) metric to quantify tool efficacy and demonstrates significant performance gains across multimodal benchmarks by enforcing hierarchical reasoning topologies.

---

[Agentic Education: Using Claude Code to Teach Claude Code](http://arxiv.org/abs/2604.17460)

- CC-SELF-TRAIN (Modular interactive curriculum framework for learning agentic AI coding tools): introduces a modular interactive curriculum for learning agentic AI coding tools through hands-on project construction, utilizing a persona progression model, adaptive learning system, cross-domain unified curriculum, step-pacing mechanism, and auto-updating design.
- The framework employs a persona progression model that maps the Gradual Release of Responsibility to AI-mediated instruction, transitioning through Guide, Collaborator, Peer, and Launcher stages.
- An adaptive learning system uses hook-based heuristics to observe engagement quality and adjust scaffolding at two timescales, while a parametrized test suite ensures structural consistency across all curriculum modules.

---

[Transparent and Controllable Recommendation Filtering via Multimodal Multi-Agent Collaboration](http://arxiv.org/abs/2604.17459)

- MAP-V (Multimodal Agentic Profiling &amp; Verification): introduces a decoupled, end-cloud collaborative framework that integrates multimodal perception and multi-agent orchestration to mitigate inferential hallucinations and modal blindness in recommendation filtering.
- The system utilizes a fact-grounded adjudication pipeline and a dual-layer dynamic preference graph to enable explicit human-in-the-loop modifications via Δ-adjustments, ensuring fine-grained intent alignment.
- MAP-V incorporates a bidirectional curation mechanism using Star Badges to enhance content discovery and alleviate user FOMO, while providing auditable transparency for algorithmic decisions.

---

[TrafficClaw: Generalizable Urban Traffic Control via Unified Physical Environment Modeling](http://arxiv.org/abs/2604.17456)

- TrafficClaw: introduces a unified urban traffic control framework that integrates heterogeneous subsystems into a shared physical environment to enable cross-subsystem coordination through executable spatiotemporal reasoning and feedback-driven decision-making.
- The framework utilizes an LLM agent equipped with an Episodic Spatiotemporal Context Cache and a Procedural Spatiotemporal Memory to accumulate reusable procedural knowledge and perform coherent long-horizon control.
- A multi-stage training pipeline, combining supervised cold-start initialization with agentic reinforcement learning under system-level objectives, ensures the agent achieves robust, transferable, and system-aware performance across diverse traffic scenarios.

---

[Compiling Deterministic Structure into SLM Harnesses](http://arxiv.org/abs/2604.17450)

- SGDe (Semantic Gradient Descent): introduces a teacher-student framework that compiles agentic workflows into discrete execution plans by iteratively refining system prompts, DAG topologies, and executable code.
- The framework utilizes a frontier LLM teacher to perform semantic error attribution on student execution traces, enabling the offloading of unreliable reasoning steps to deterministic code or consensus subgraphs.
- By treating the SLM-runtime boundary as an optimizable target, SGDe achieves high accuracy on structured reasoning tasks with minimal training samples while maintaining low inference costs.

---

[WirelessAgent: A Unified Agent Design for General Wireless Resource Allocation Problem without Current Channel State Information](http://arxiv.org/abs/2604.17440)

- WirelessAgent: introduces an agentic AI framework on the Coze platform that autonomously manages wireless resource allocation under incomplete CSI using an Intent Recognition Layer, Channel Reconstruction Module, Optimization Solvers, and Orchestration Layer.
- The framework utilizes an LLM-based perception layer to map natural language user requirements into specific multi-objective optimization flags for adaptive resource scheduling.
- A nearest-neighbor collaborative filtering neural network (NNCF) module reconstructs missing channel data by exploiting frequency-spatial correlations to improve downstream decision-making accuracy.

---

[The Inference Bottleneck: A Formal Model of Vertical Foreclosure in AI Markets](http://arxiv.org/abs/2604.17431)

- Neutral Inference framework: introduces a formal game-theoretic model of vertical foreclosure in AI markets, identifying conditions where upstream providers profitably degrade third-party QoS or bias routing.
- The framework characterizes the equilibrium QoS gap between upstream and downstream applications, demonstrating how vertical integration and tier-based access discrimination create durable rents.
- It proposes a four-pillar conduct-based regulatory policy—QoS parity, routing transparency, FRAND-style non-discrimination, and tier transparency—to mitigate foreclosure while preserving innovation incentives.

---

[Jupiter-N Technical Report](http://arxiv.org/abs/2604.17429)

- Jupiter-N: introduces a sovereign post-training template for LLMs that integrates agentic capabilities, UK cultural alignment, and Welsh language support while preserving base model reasoning.
- The framework utilizes the Forget-Me-Not methodology to combine on-policy synthetic replay with off-policy task data, effectively mitigating catastrophic forgetting during fine-tuning.
- An entropy-based curation strategy is employed to select high-information-density training samples, significantly improving agentic performance and instruction-following accuracy.

---

[ARMove: Learning to Predict Human Mobility through Agentic Reasoning](http://arxiv.org/abs/2604.17419)

- ARMove: introduces a multi-agent framework for interpretable human mobility prediction that integrates Feature Optimization Agent, User Profiling Engineer Agent, and Behavior Prediction Agent to enhance predictive accuracy and transparency.
- The framework utilizes Four Feature Pools and Iterative Optimization to refine mobility features while employing a Model Knowledge Transfer Mechanism to distill reasoning capabilities from large LLMs into smaller, cost-efficient LLMs.
- ARMove demonstrates robust transferability across regions, user groups, and model scales, outperforming existing baselines by leveraging agentic reasoning to provide explicit, interpretable decision paths for mobility prediction.

---

[Think before Go: Hierarchical Reasoning for Image-goal Navigation](http://arxiv.org/abs/2604.17407)

- HRNav: introduces a hierarchical framework for image-goal navigation that decomposes the task into high-level planning using a VLM and low-level execution via an RL policy.
- The framework utilizes a novel Wandering Suppression Penalty to minimize redundant motions and backtracking during long-horizon navigation episodes.
- HRNav achieves state-of-the-art performance in both simulated and real-world environments by leveraging a large-scale self-collected dataset for training the high-level reasoning module.

---

[EvoMaster: A Foundational Agent Framework for Building Evolving Autonomous Scientific Agents at Scale](http://arxiv.org/abs/2604.17406)

- EvoMaster: introduces a foundational, domain-agnostic agent framework designed to facilitate continuous self-evolution in scientific discovery through modular components including Playground Orchestrator, Agent Engine, Context Manager, Trajectory System, Skill System, and Tool System.
- The framework utilizes a multi-turn reactive loop where agents perform reasoning, tool invocation, observation, and self-critique to iteratively refine scientific hypotheses.
- EvoMaster enables the rapid deployment of specialized scientific agents by decoupling execution logic from domain-specific knowledge, significantly reducing engineering overhead for complex research tasks.

---

[MAGRPO: Accelerated MARL Training for Fluid Antenna-Assisted Wireless Network Optimization](http://arxiv.org/abs/2604.17379)

- MAGRPO: introduces a decentralized multi-agent reinforcement learning algorithm that replaces the centralized critic network with group relative advantage estimation to accelerate training in fluid antenna-assisted wireless networks.
- The framework utilizes a centralized training and decentralized execution paradigm to optimize antenna positions, beamforming, and power allocation while maintaining low computational complexity.
- Theoretical analysis provides a variance upper bound for the cumulative reward, offering practical guidelines for stabilizing training through group size and learning rate adjustments.

---

[LLM-Guided Strategy Synthesis for Scalable Equality Saturation](http://arxiv.org/abs/2604.17364)

- EggMind: introduces an LLM-guided framework that reformulates equality saturation strategy synthesis as a controlled offline process using EqSatL, an agentic workflow, rewrite motif caching, and tractability guidance.
- The framework utilizes an agentic workflow with specialized agents—Generator, Evaluator, Partitioner, and Simplifier—to iteratively synthesize and refine reusable strategy artifacts while managing e-graph growth.
- By extracting and caching proof-derived rewrite motifs and applying tractability guidance, EggMind enables stable, high-quality strategy synthesis that transfers effectively across different optimization domains.

---

[Hive: A Multi-Agent Infrastructure for Algorithm- and Task-Level Scaling](http://arxiv.org/abs/2604.17353)

- Hive: introduces a multi-agent inference infrastructure that optimizes algorithm- and task-level scaling through a descriptive frontend and a backend featuring Logits Cache and Agent-Aware Scheduling.
- The framework utilizes a coroutine-based frontend to model agent interactions and a backend that reuses intermediate logits to mitigate cross-path redundancy during test-time scaling.
- Hive employs an agent-aware scheduler that dynamically allocates KV cache resources based on Shapley-based contribution modeling to improve serving efficiency in multi-agent workloads.

---

[SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](http://arxiv.org/abs/2604.17351)

- SOCIA-EVO: introduces a dual-anchored evolutionary framework for automated simulator construction that utilizes a static Blueprint and a dynamic Strategy Playbook to ensure distributional fidelity.
- The framework employs a bi-level optimization architecture, separating structural refinement from parameter calibration to prevent optimization instability and contextual drift in LLM-based agents.
- By leveraging a self-curating Playbook and evidence-based diagnosis, the system effectively prunes ineffective remedial hypotheses and converges toward high-fidelity simulation models.

---

[Formal Foundations of Agentic Business Process Management](http://arxiv.org/abs/2604.17347)

- Agentic BPM: introduces a formal framework for modeling and reasoning about autonomous agents in business processes, utilizing Agentic BPM, Centralized decision-making, Decentralized decision-making, Explicit assumptions, Implicit assumptions, Winning goal attainment, Best-effort goal attainment, Agent strategies, Environment strategy, and Operational process specification.
- The framework provides a mathematical foundation for agentic processes, enabling the analysis of realizability, synthesis, model-checking, and guardrailing across three distinct settings of agent autonomy.
- The approach supports both imperative and declarative process specifications, allowing for the formal verification of agent behavior in complex, multi-agent organizational environments.

---

[Neuro-Symbolic Resolution of Recommendation Conflicts in Multimorbidity Clinical Guidelines](http://arxiv.org/abs/2604.17340)

- Neuro-Symbolic framework: introduces a multi-agent pipeline that translates unstructured clinical guidelines into formal symbolic logic for automated conflict detection using a Z3 SMT solver.
- The system utilizes an Entity Agent, Predicate Agent, and Rule Agent to formalize clinical recommendations, enabling the identification of redundancies and multimorbidity-specific local conflicts.
- Experimental results demonstrate that this neuro-symbolic approach significantly outperforms standalone LLMs in detecting logical inconsistencies within clinical guidelines under realistic retrieval noise conditions.

---

[Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](http://arxiv.org/abs/2604.17338)

- PDB (Precise Debugging Benchmarking): introduces an automated pipeline that evaluates LLM debugging capabilities by distinguishing between targeted, minimal code edits and wholesale solution regeneration.
- The framework utilizes novel edit-level precision and bug-level recall metrics to penalize unnecessary modifications and reward precise fault localization.
- Experiments demonstrate that while frontier LLMs achieve high functional pass rates, they frequently rely on broad code rewrites rather than minimal, intent-preserving repairs.

---

[AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via Reinforcement Learning](http://arxiv.org/abs/2604.17337)

- AutoSearch: introduces a reinforcement learning framework that enables LLMs to adaptively determine the minimal sufficient search depth for agentic RAG systems by evaluating intermediate answers.
- The framework utilizes three complementary reward signals—base, search efficiency, and search quality—to balance answer accuracy with computational cost.
- By generating intermediate answers at each retrieval step, the agent dynamically adjusts its search trajectory based on question complexity and its own evolving capability.

---

[Signal or Noise in Multi-Agent LLM-based Stock Recommendations?](http://arxiv.org/abs/2604.17327)

- MarketSenseAI: introduces a multi-agent LLM equity system that synthesizes specialist agent analyses into a unified investment thesis and ordinal recommendation.
- The system utilizes a synthesis agent to adaptively weight inputs from News-, Fundamentals-, Dynamics-, and Macro-agents based on market regimes and sector conditions.
- Empirical validation demonstrates that the system's strong-buy recommendations outperform passive benchmarks and random selection, with NNLS attribution revealing context-dependent agent contributions.

---

[Distributed Nesterov Flows for Multi-agent Optimization](http://arxiv.org/abs/2604.17311)

- DNGD framework: introduces a continuous-time flow approximation for distributed optimization to derive discrete-time algorithms with improved convergence rates.
- The framework utilizes Consensus component, Friction component, and Gradient component to optimize local objective functions across a defined Network topology.
- The proposed DNGD-SC and DNGD-C algorithms achieve faster convergence for strongly convex and convex problems respectively by leveraging insights from control theory.

---

[Knows: Agent-Native Structured Research Representations](http://arxiv.org/abs/2604.17309)

- Knows: introduces a lightweight YAML sidecar specification that binds structured claims, evidence, and provenance to research artifacts, enabling LLMs to consume scholarly content directly without modifying the original PDF.
- The framework utilizes KnowsRecord to provide machine-stable metadata, which significantly improves accuracy for weak LLMs and reduces token consumption for all models by providing a structured alternative to parsing unstructured prose.
- The system includes a deterministic validation tool, knows-lint, and supports both author-native and retrofit workflows, facilitating a transition toward agent-native research ecosystems where structured records serve as the primary interchange format.

---

[SKILLFLOW: Benchmarking Lifelong Skill Discovery and Evolution for Autonomous Agents](http://arxiv.org/abs/2604.17308)

- SKILLFLOW: introduces a benchmark for evaluating lifelong skill discovery and evolution in autonomous agents using a DAEF (Domain-Agnostic Execution Flow) to structure task families.
- The framework employs an Architect Agent and a Critic Agent to construct tasks, while agents utilize an Agentic Lifelong Learning Protocol to externalize experience into a persistent Skill Library.
- Empirical results demonstrate that stronger models effectively consolidate experience into compact, reusable procedures, whereas weaker models often suffer from fragmented skill growth and negative transfer.

---

[Clover: A Neural-Symbolic Agentic Harness with Stochastic Tree-of-Thoughts for Verified RTL Repair](http://arxiv.org/abs/2604.17288)

- Clover: introduces a neural-symbolic agentic harness that orchestrates RTL repair by combining LLM-based reasoning with SMT-based symbolic solvers, utilizing a Main Agent, Context Agent, Lint-fix Agent, SMT-based Symbolic Repair, Stochastic Tree-of-Thoughts, and an RTL-specific Toolbox.
- The framework employs a Stochastic Tree-of-Thoughts mechanism to manage the search space of hypotheses, balancing exploration and exploitation to produce reliable, verified RTL patches.
- By delegating low-level logic adjustments to SMT solvers and high-level design intent to LLM agents, the system effectively addresses multi-step heterogeneity in RTL program repair.

---

[HalluClear: Diagnosing, Evaluating and Mitigating Hallucinations in GUI Agents](http://arxiv.org/abs/2604.17284)

- HalluClear: introduces a comprehensive suite for diagnosing, evaluating, and mitigating hallucinations in GUI agents, utilizing a GUI-specific hallucination taxonomy, a calibrated three-stage evaluation workflow, and a closed-loop OODA-style reasoning pattern.
- The framework employs JQ-Bench to calibrate VLM judges, which then perform fine-grained hallucination detection on agent outputs to enable quantitative and scalable evaluation.
- By integrating structured reasoning trajectories and reinforcement learning, the approach significantly reduces hallucinations and improves grounding and action fidelity in GUI agents.

---

[The Continuity Layer: Why intelligence needs an architecture for what it carries forward](http://arxiv.org/abs/2604.17273)

- DTCM (Decomposed Trace Convergence Memory): introduces a storage primitive that decomposes interactions into five independent traces to enable persistent, coherent state reconstruction across LLM sessions.
- The framework utilizes a scoring equation to weight traces based on relevance, ensuring that reconstructed state reflects current reality rather than stale historical data.
- This architecture addresses the structural amnesia of current LLM stacks by providing a continuity layer that operates independently of the model's weights.

---

[MemSearch-o1: Empowering Large Language Models with Reasoning-Aligned Memory Growth in Agentic Search](http://arxiv.org/abs/2604.17265)

- MemSearch-o1: introduces an agentic search framework that mitigates memory dilution by growing fine-grained memory fragments from query-derived seed tokens and reorganizing them into coherent paths via a retracing mechanism.
- The framework utilizes a contribution function to evaluate the relevance and bridge potential of memory fragments, ensuring that the LLM operates within a concise, query-focused memory space.
- By shifting from stream-like context concatenation to structured, token-level memory growth, MemSearch-o1 enhances the reasoning depth and quality of LLMs in multi-hop search tasks.

---

[Seeing Isn’t Believing: Mitigating Belief Inertia via Active Intervention in Embodied Agents](http://arxiv.org/abs/2604.17252)

- EVU (Estimate-Verify-Update): introduces a unified mechanism that mitigates belief inertia in embodied agents by explicitly managing belief states through Estimation, Verification, and Belief Update components.
- The framework decouples belief management from action generation, allowing the agent to ground its reasoning in updated environmental states rather than relying on stale prior beliefs.
- EVU integrates seamlessly into both prompting-based and training-based agent architectures, consistently improving task success rates across diverse embodied benchmarks.

---

[DORA Explorer: Improving the Exploration Ability of LLMs Without Training](http://arxiv.org/abs/2604.17244)

- DORA Explorer: introduces a training-free inference framework that improves exploration in LLM agents by generating diverse action candidates and ranking them using sequence-level log-probability metrics.
- The framework utilizes a dynamic exploration parameter λ to control the trade-off between greedy exploitation and exploratory action selection based on the agent's current context.
- DORA Explorer achieves competitive performance against standard reinforcement learning baselines and improves task success across diverse text-based environments by reducing repetitive action loops.

---

[Safe and Policy-Compliant Multi-Agent Orchestration for Enterprise AI](http://arxiv.org/abs/2604.17240)

- CAMCO (Constraint-Aware Multi-Agent Cognitive Orchestration): introduces a deployment-time middleware that models multi-agent decision-making as a constrained optimization problem to ensure policy compliance and bounded risk.
- The framework integrates a Constraint Projection Engine (CPE), a Risk-Weighted Utility Engine (RWUE), and a Negotiation Loop to enforce hard constraints while maintaining high utility retention.
- Evaluation across enterprise scenarios demonstrates that CAMCO achieves zero policy violations and maintains 92–97% utility retention by dynamically adjusting agent proposals through iterative Lagrangian dual ascent.

---

[From Language to Action: Enhancing LLM Task Efficiency with Task-Aware MCP Server Recommendation](http://arxiv.org/abs/2604.17234)

- T2MRec: introduces a task-oriented recommendation framework that jointly models semantic relevance and structural compatibility to identify appropriate MCP servers for development tasks.
- The framework utilizes a dual-tower encoder for semantic matching, structural compatibility scoring for engineering feasibility, and a constrained LLM re-ranker for fine-grained refinement.
- The authors also construct Task2MCP, a large-scale dataset mapping taxonomy-grounded development tasks to curated MCP servers, and implement an interactive agent prototype for real-world deployment.

---

[Yanasse: Finding New Proofs from Deep Vision’s Analogies — Part 1](http://arxiv.org/abs/2604.17229)

- Yanasse: introduces a method for discovering new proofs by transferring proof strategy patterns across structurally distant mathematical areas using relational analogy.
- The system utilizes a domain-independent matching engine to identify structural similarities between proof states, enabling the transfer of tactic schemas from a source area to a target area.
- The framework decomposes tactic schemas into domain-gated heads and domain-general modifiers, facilitating the adaptation of proof strategies through an AI reasoning agent.

---

[Project Prometheus: Bridging the Intent Gap in Agentic Program Repair via Reverse-Engineered Executable Specifications](http://arxiv.org/abs/2604.17464)

- Prometheus: introduces a multi-agent framework that bridges the "Intent Gap" in Automated Program Repair by reverse-engineering executable Gherkin specifications to guide code generation.
- The framework utilizes an Architect agent for specification inference, an Engineer agent for RQA Loop validation using a Proxy Oracle, and a Fixer agent for specification-guided surgical repair.
- By transforming repair into a deterministic constraint satisfaction problem, Prometheus achieves a 93.97% correct patch rate on the Defects4J benchmark, significantly outperforming blind LLM-based repair agents.

---

[A Multi-Agent Approach for Claim Verification from Tabular Data Documents](http://arxiv.org/abs/2604.17225)

- MACE: introduces a multi-agent framework for claim verification from tabular data documents using User Agent, Planner Agent, Executor Agent, and Verifier Agent to perform zero-shot reasoning.
- The framework utilizes a constrained group-chat system with feedback loops between agents to mitigate error propagation and ensure logical consistency in verification traces.
- MACE achieves state-of-the-art performance on multiple benchmarks while enabling smaller LLMs to approach the capabilities of significantly larger models through structured agent collaboration.

---

[Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](http://arxiv.org/abs/2604.17220)

- Hierarchical Reasoning Framework: introduces a scalable experimental paradigm using LLMs to simulate multi-stage supply chain dynamics and analyze the impact of cognitive heterogeneity on agent interactions.
- The framework employs LLM agents as controllable strategic actors, utilizing Chain of Thought (CoT) prompting to support structured decision-making across distinct cognitive layers.
- By integrating LLM decision modules with a rule-based architecture, the study systematically evaluates behavioral biases like the bullwhip effect and the impact of information sharing in complex operational environments.

---

[EmbodiedHead: Real-Time Listening and Speaking Avatar for Conversational Agents](http://arxiv.org/abs/2604.17211)

- EmbodiedHead: introduces an end-to-end speech-driven talking-head framework that couples a Rectified-Flow DiT with a differentiable renderer to enable real-time, high-fidelity avatar generation for LLMs.
- The framework utilizes a Streaming Audio Scheduler and explicit listening-speaking state conditioning to achieve unified conversational behavior without interlocutor look-ahead dependencies.
- A two-stage training scheme, combining coefficient-space pretraining with end-to-end image-domain refinement, effectively bridges the gap between motion-level supervision and rendered visual quality.

---

[Do LLM-derived graph priors improve multi-agent coordination?](http://arxiv.org/abs/2604.17191)

- LLM-Derived Graph Prior Framework: introduces a method for generating structured coordination graph priors from natural language observations to guide multi-agent reinforcement learning.
- The framework utilizes an LLM to infer relational dependencies between agents, which are then integrated into a GNN-based pipeline to improve cooperative task performance.
- Empirical results demonstrate that compact open-source LLMs effectively generate coordination priors that outperform traditional heuristic and learned graph-based methods in multi-agent environments.

---

[LookasideVLN: Direction-Aware Aerial Vision-and-Language Navigation](http://arxiv.org/abs/2604.17190)

- LookasideVLN: introduces a navigation paradigm that leverages directional cues in natural language to enhance spatial reasoning and computational efficiency for aerial navigation.
- The framework utilizes an Egocentric Lookaside Graph (ELG) to model landmarks and directional relationships, a Spatial Landmark Knowledge Base (SLKB) for efficient memory retrieval, and a Lookaside MLLM Navigation Agent for multimodal path planning.
- By focusing on egocentric directional cues rather than global scene graphs, the approach achieves state-of-the-art performance in Aerial VLN tasks with reduced computational overhead.

---

[Persona-Based Requirements Engineering for Explainable Multi-Agent Educational Systems: A Scenario Simulator for Clinical Reasoning Training](http://arxiv.org/abs/2604.17186)

- MAES RE Framework: introduces a human-centered methodology that integrates AI Personas and XAI user stories into the requirements engineering lifecycle to ensure explainability in multi-agent educational systems.
- The framework utilizes AI Personas to model agent capabilities, limitations, and decision-making processes, facilitating the derivation of actionable functional and non-functional requirements for clinical reasoning training.
- The methodology includes a supervisor agent that orchestrates patient-, physical exam-, diagnostic-, clinical intervention-, and evaluation-agents to provide a transparent and trustworthy simulation environment for medical students.

---

[BranchBench: Aligning Database Branching with Agentic Demands](http://arxiv.org/abs/2604.17180)

- BranchBench: introduces a systematic benchmark suite to evaluate branchable database management systems under high-frequency agentic exploration workloads.
- The paper characterizes five representative agentic workflows—software engineering, failure reproduction, data curation, MCTS, and simulation—to stress-test branch creation, storage efficiency, and cross-branch query performance.
- Empirical evaluation reveals a fundamental architectural trade-off where systems either optimize for branching agility or query performance, highlighting the need for branch-native DBMS designs.

---

[Governed Auditable Decisioning Under Uncertainty: Synthesis and Agentic Extension](http://arxiv.org/abs/2604.19112)

- N4 (Governance Evidence Chain): synthesizes diagnostic theory, trace capture, sufficiency measurement, and monitoring into an integrated infrastructure to enable post-incident reconstruction of automated decision systems.
- The framework establishes a governance coverage gradient across deterministic, classical ML, hybrid, and agentic architectures, identifying structural breaks in agentic systems that necessitate specialized extensions.
- It introduces the cascade of uncertainty, a serial dependency chain where governance failures at one layer propagate and compound, revealing systemic vulnerabilities that individual components cannot detect in isolation.

---

#### 18th April 2026

[Systematic Capability Benchmarking of Frontier Large Language Models for Offensive Cyber Tasks](http://arxiv.org/abs/2604.17159)

- D-CIPHER: introduces a systematic evaluation of LLM agents on offensive cybersecurity tasks by extending a hierarchical multi-agent architecture with multi-provider backends, a specialized Kali Linux environment, and runtime tool-discovery agents.
- The framework utilizes a Planner-Executor design where the Planner decomposes tasks and the Executor performs low-level operations using specialized security tools within a containerized environment.
- Experimental results demonstrate that environment tooling and model selection are primary performance drivers, while homogeneous planner-executor configurations outperform mixed-model setups.

---

[Graph-of-Agents: A Graph-Based Framework for Multi-Agent LLM Collaboration](http://arxiv.org/abs/2604.17148)

- GoA (Graph-of-Agents): introduces a graph-based framework for multi-agent LLM collaboration that optimizes task performance through selective node sampling, relevance-aware edge construction, and structured message passing.
- The framework utilizes a Meta-LLM to select relevant agents and employs bidirectional message passing between source and target nodes to refine responses before final aggregation via graph pooling.
- GoA improves scalability and efficiency by dynamically constructing task-specific graphs, allowing superior performance with fewer agents compared to traditional multi-agent baselines.

---


[BOIL: Learning Environment Personalized Information](http://arxiv.org/abs/2604.17137)

- BOIL: introduces a scalable framework for extracting environment-level insights from a blackbox oracle to optimize multi-agent coverage, patrolling, and reachability tasks.
- The framework leverages Pagerank and non-reversible Markov chains to distill environment structure into learnable parameters, enabling fine-grained control over agent strategy distributions.
- By formulating the problem as a common information maximization task, BOIL achieves computational efficiency and scalability independent of the number of agents involved.

---

[Strong MHD Turbulence and Coherent Structures as Cosmic Particle Accelerators](http://arxiv.org/abs/2604.17119)

- MHD Turbulence Framework: introduces a unified perspective on astrophysical plasma heating and particle acceleration by placing the self-consistent emergence of coherent structures at the center of turbulent dynamics.
- The paper synthesizes how multiscale energy transfer in magnetized plasmas generates intermittent structures—such as current sheets, flux ropes, and shocks—that act as primary sites for non-thermal particle energization.
- It evaluates numerical methodologies, including MHD, PIC, and test-particle simulations, while proposing the use of physics-informed neural networks to bridge the gap between large-scale turbulence and kinetic-scale particle transport.

---

[Small Model as Master Orchestrator: Learning Unified Agent-Tool Orchestration with Parallel Subtask Decomposition](http://arxiv.org/abs/2604.17009)

- ParaManager: introduces a unified parallel orchestration paradigm that abstracts agents and tools into a standardized action space to enable state-aware subtask decomposition and delegation.
- The framework utilizes a two-stage training pipeline, incorporating supervised fine-tuning with recoverable failure trajectories and reinforcement learning to optimize for task success, protocol compliance, and reasoning efficiency.
- ParaManager enables asynchronous parallel execution and cross-verification of subtasks, significantly improving robustness against single-path failures and demonstrating strong generalization across unseen model pools.

---

[Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification](http://arxiv.org/abs/2604.16993)

- Rule-VLN: introduces a large-scale urban benchmark for rule-compliant navigation that integrates MPSI, SNRM, and an Epistemic Mental Map to address the "Goal-driven trap" in existing VLMs.
- The MPSI pipeline utilizes mask-guided diffusion to inject geometrically rectified regulatory signals into urban environments, while the SNRM module employs a dual-stage perception framework to enable zero-shot rule adherence.
- By combining macro-micro visual prompting with an epistemic mental map, the framework allows agents to override geometric shortest-path heuristics in favor of semantically compliant detours.

---

[Beyond Serendipity: From Exposing the Unknown to Fostering Engagement through Peer Recommendation](http://arxiv.org/abs/2604.16818)

- Peer Recommendation: introduces a framework that reframes recommendation as a collaborative dialogue process between a user and an LLM-based Peer agent to foster active engagement with unfamiliar content.
- The system utilizes User Profile Generation, Peer Profile Generation, Mutual Recommendation via Dialogue, and Collaborative Playlist Construction to facilitate mutual exploration through distinct agent personas.
- Empirical results demonstrate that explicit persona design and agent "otherness" significantly improve perceived interest expansion and value compared to accommodating baseline agents.

---

[PersonalHomeBench: Evaluating Agents in Personalized Smart Homes](http://arxiv.org/abs/2604.16813)

- PersonalHomeBench: introduces a benchmark for evaluating LLMs as agentic assistants in personalized smart home environments, utilizing PersonalHomeTools, Transcriber, Memory Retriever, Video Understanding Agent, and Role Playing LLM-as-a-Judge.
- The framework evaluates reactive and proactive agentic abilities across unimodal and multimodal observations, revealing performance limitations in tool use, long-horizon planning, and counterfactual reasoning.
- PersonalHomeTools provides a grounded interaction layer for household information retrieval, appliance control, and situational understanding, enabling systematic evaluation of agentic decision-making.

---


[ScenarioControl: Vision-Language Controllable Vectorized Latent Scenario Generation](http://arxiv.org/abs/2604.17147)

- ScenarioControl: introduces a vision-language conditioned latent diffusion model that synthesizes diverse, realistic 3D driving scenarios in a vectorized latent space using a cross-global control mechanism.
- The framework integrates cross-attention with a global-context branch to enable fine-grained control over road layout and traffic conditions via text prompts or dashcam images.
- ScenarioControl supports long-horizon video continuation and sensor data generation by grounding diffusion-based synthesis in structured, vectorized scene representations.

---

[SeekerGym: A Benchmark for Reliable Information Seeking](http://arxiv.org/abs/2604.17143)

- SeekerGym: introduces a benchmark environment for evaluating information-seeking agents on their ability to retrieve complete information and quantify their uncertainty regarding that completeness.
- The framework utilizes a POMDP-based pipeline where a Seeker agent interacts with a Retriever and a Belief Encoder to iteratively gather passages from a document until a calibrated stopping criterion is met.
- SeekerGym employs conformal prediction to transform raw completeness estimates from LLMs into calibrated prediction intervals, providing reliable stopping signals for information-gathering tasks.

---

[Logic-Based Verification of Task Allocation for LLM-Enabled Multi-Agent Manufacturing Systems](http://arxiv.org/abs/2604.17142)

- Logic-Based Verification of Task Allocation for LLM-Enabled Multi-Agent Manufacturing Systems: introduces a control architecture that leverages LLMs for adaptive task planning while ensuring safety through formal verification using discrete event systems.
- The framework utilizes a Central Controller Agent to translate natural language safety requirements into temporal logic automata, which are then used to validate LLM-generated task plans via reachability analysis.
- Detected safety violations are returned as structured feedback to the LLM-based planner, enabling iterative refinement of task plans until they satisfy all formal safety constraints.

---

[The Consensus Trap: Rescuing Multi-Agent LLMs from Adversarial Majorities via Token-Level Collaboration](http://arxiv.org/abs/2604.17139)

- RR (Token-Level Round-Robin Collaboration): introduces a multi-agent framework that replaces response-level aggregation with sequential token-level interleaving to mitigate adversarial corruption in LLMs.
- The framework utilizes a shared context window as a truth attractor, allowing honest agents to intercept and correct flawed logic mid-generation before it propagates.
- By shifting from linear response-level voting to a non-linear operator product, the approach maintains robust accuracy even when corrupted agents form a local majority.

---

[If Only My CGM Could Speak: A Privacy-Preserving Agent for Question Answering over Continuous Glucose Data](http://arxiv.org/abs/2604.17133)

- CGM-Agent: introduces a privacy-preserving framework that decouples LLM reasoning from data computation to enable secure question answering over personal glucose data.
- The architecture utilizes an Input Processor to resolve query ambiguity, an Analytical Agent to execute local tool calls, and a Response Generator to synthesize answers.
- The framework ensures privacy by keeping raw CGM data within a local execution sandbox, with the LLM acting solely as a reasoning engine for function selection.

---

[HiveMind: OS-Inspired Scheduling for Concurrent LLM Agent Workloads](http://arxiv.org/abs/2604.17111)

- HiveMind: introduces a transparent HTTP proxy that applies OS-inspired scheduling primitives to manage concurrent LLM agent workloads and prevent resource contention failures.
- The system integrates Admission Gate, Rate Limiter, AIMD + Circuit, Token Budget, and Retry components to coordinate parallel API requests without requiring modifications to existing agent code.
- Evaluation demonstrates that HiveMind significantly reduces agent failure rates and wasted compute costs by serializing requests and managing API contention effectively.

---

[From Clinical Intent to Clinical Model: An Autonomous Coding-Agent Framework for Clinician-driven AI Development](http://arxiv.org/abs/2604.17110)

- Clinician-driven AI Development Framework: introduces an autonomous system that enables clinicians to develop task-specific medical AI models through natural-language interaction, utilizing a Semantic Parser, Task Initializer, and an Autonomous Developer.
- The framework replaces traditional human-intensive AI development workflows with an LLM-based agent that iteratively refines code, experiments, and debugs models based on clinician-specified objectives and constraints.
- Experimental results demonstrate the framework's capability to handle complex clinical tasks, including mixed-supervision fracture detection and debiased pneumothorax classification, by autonomously integrating established deep-learning techniques.

---

[Live LTL Progress Tracking: Towards Task-Based Exploration](http://arxiv.org/abs/2604.17106)

- LPT (Live LTL Progress Tracking): introduces a framework that decomposes LTL specifications into a multilayered formula tree to generate a tracking vector representing agent progress through multi-stage tasks.
- The framework utilizes modular logical and temporal operators to update tracking vectors at each time step, assigning true, false, or open status to each node in the specification tree.
- LPT provides high-resolution behavioral signatures that can be integrated into reward machines to facilitate task-space exploration and diverse solution-finding in reinforcement learning.

---

[GenericAgent: A Token-Efficient Self-Evolving LLM Agent via Contextual Information Density Maximization (V1.0)](http://arxiv.org/abs/2604.17091)

- GA (GenericAgent): introduces a self-evolving LLM agent system that maximizes contextual information density to improve long-horizon performance while reducing token consumption.
- The framework utilizes a minimal atomic tool set, hierarchical memory, a reflection-driven self-evolution pipeline, and context truncation to maintain decision-relevant information within a compact context budget.
- GA demonstrates superior task completion and token efficiency across multiple benchmarks by systematically converting verified interaction trajectories into reusable SOPs and executable code.

---

[From Necklaces to Coalitions: Fair and Self-Interested Distribution of Coalition Value Calculations](http://arxiv.org/abs/2604.17057)

- N-DCA (Necklace-based Distributed Coalition Algorithm): introduces a communication-free, decentralised framework for distributing coalition value calculations in characteristic function games using combinatorial necklaces and increment arrays.
- The framework ensures equitable, non-redundant, and self-interested coalition value calculation assignments across agents by mapping canonical increment arrays to two-colour necklaces.
- N-DCA achieves optimal load balancing through either per-size or global offset designation schemes, requiring only agent identifiers and the total number of agents to operate independently.

---

[Workstream: A Local-First Developer Command Center for the AI-Augmented Engineering Workflow](http://arxiv.org/abs/2604.17055)

- Workstream: introduces a local-first developer command center that consolidates fragmented engineering workflows into a single dashboard using Browser SPA, FastAPI, Pollers, AI Reviewer, Readiness, Intelligence, Agents, MCP Server, SQLite, GitHub API, GitLab API, Jira REST, Google Cal, and AI Providers.
- The system utilizes a polling architecture to aggregate data from multiple external sources into a local SQLite database, providing unified visibility into PRs, tasks, and AI agent status.
- Workstream incorporates specialized modules for AI-readiness scoring, historical review intelligence, and agent observability to reduce cognitive load for developers managing AI-augmented workflows.

---

[Harness as an Asset: Enforcing Determinism via the Convergent AI Agent Framework (CAAF)](http://arxiv.org/abs/2604.17025)

- CAAF (Convergent AI Agent Framework): introduces a closed-loop cybernetic architecture that enforces deterministic grounding for LLMs by integrating a Harness Registry, UAI Assertion Engine, and Semantic Reviewer to ensure monotonic convergence toward verified engineering artifacts.
- The framework utilizes Recursive Atomic Decomposition to isolate executor contexts, preventing context rot and ensuring that complex engineering constraints are satisfied simultaneously through structured semantic gradients.
- CAAF provides a robust mechanism for paradox detection and strategic negotiation, enabling LLMs to operate reliably in safety-critical domains by treating constraints as immutable assets rather than prompt-based suggestions.

---

[Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](http://arxiv.org/abs/2604.17020)

- Persona-based harmful content generation framework: introduces a method for synthesizing harmful content by leveraging persona-guided LLM agents to facilitate robust evaluation of detection models.
- The framework constructs two-dimensional personas integrating intrinsic demographic identities with extrinsic situational harmful strategies to generate contextually grounded and strategically varied content.
- Experimental results demonstrate that the synthetic scenarios produced by the framework are more challenging for existing detection models to identify than those in static benchmarks, while maintaining linguistic and topical diversity comparable to human-curated datasets.

---

[Mini-BEHAVIOR-Gran: Revealing U-Shaped Effects of Instruction Granularity on Language-Guided Embodied Agents](http://arxiv.org/abs/2604.17019)

- Mini-BEHAVIOR-Gran: introduces a benchmark for controlled studies of instruction granularity in embodied AI, utilizing VLA Agent, Environment, Dynamic Instructor Module, Rule Set Warehouse, and Template-Based Natural Language Instruction Generator.
- The framework employs a Shared Feature Space and Symbolic Layer to quantify instruction granularity via width, revealing a non-monotonic U-shaped relationship between instruction detail and agent performance.
- The study demonstrates that coarse instructions increase predictability, incentivizing agents to adopt vision-dominant policies that sacrifice deep language grounding for visual shortcuts.

---

[False Security Confidence in Benign LLM Code Generation](http://arxiv.org/abs/2604.17014)

- FSC (False Security Confidence): introduces a measurement framework to quantify the prevalence of security failures within functionally correct code generated by LLMs.
- The framework utilizes a conditional FSC-rate metric to isolate security-failing outputs from the subset of generations that satisfy functional correctness requirements.
- It categorizes evaluation across three task ecosystems and defines FSC-hard as a refinement layer for vulnerabilities that evade static analysis but remain dynamically exploitable.

---

[Bolzano: Case Studies in LLM-Assisted Mathematical Research](http://arxiv.org/abs/2604.16989)

- Bolzano: is a multi-agent LLM system that orchestrates iterative research rounds between parallel prover agents, a verifier agent, and a summarizer agent to investigate mathematical problems.
- The system maintains a persistent knowledge base across rounds, allowing for human-in-the-loop guidance and the integration of diverse LLMs to explore varied mathematical proof strategies.
- Bolzano demonstrates utility in generating intermediate research moves, such as finding counterexamples and constructing proofs, across six problems in mathematics and theoretical computer science.

---

[DVAR: Adversarial Multi-Agent Debate for Video Authenticity Detection](http://arxiv.org/abs/2604.16987)

- DVAR (Debate-based Video Authenticity Reasoning): introduces a training-free framework that reformulates video authenticity detection as a structured multi-agent forensic reasoning process using Multimodal Perception Module, Generative Hypothesis Agent (GHA), Natural Mechanism Agent (NMA), Arbiter, and GenVideoKB.
- The framework employs an adversarial debate between GHA and NMA to elicit competing explanations for detected anomalies, which are then adjudicated using an MDL-based explanatory cost calculation.
- By integrating GenVideoKB for domain-specific heuristics, DVAR achieves robust, interpretable, and generalizable video authenticity assessment without requiring task-specific training.

---

[Decomposition Envy-Freeness in Random Assignment](http://arxiv.org/abs/2604.16973)

- Dec-EF: introduces a fairness property for random assignments that evaluates the distribution over deterministic assignments rather than just the assignment matrix.
- The framework establishes that SD-EF assignment matrices admit Dec-EF decompositions for instances with at most three agents or two distinct preferences.
- The research provides upper bounds on agent envy and demonstrates that while RP satisfies Dec-EF, not all SD-EF matrices are easily decomposable into reversal-symmetric distributions.

---

[On Safety Risks in Experience-Driven Self-Evolving Agents](http://arxiv.org/abs/2604.16968)

- AWM and ReasoningBank: investigate how experience accumulation in self-evolving LLM agents leads to safety degradation through the reuse of execution-oriented experience.
- The study demonstrates that even benign experiences, when accumulated and retrieved, reinforce execution bias and cause agents to prioritize task completion over safety constraints.
- Experimental results across web and household environments reveal a persistent safety–utility trade-off, where refusal-based experience mitigates safety risks but induces over-refusal on benign tasks.

---

[Visual Inception: Compromising Long-term Planning in Agentic Recommenders via Multimodal Memory Poisoning](http://arxiv.org/abs/2604.16966)

- COGNITIVEGUARD: introduces a dual-process defense framework to mitigate "Visual Inception" attacks by combining System 1 Perceptual Sanitizer and System 2 Reasoning Verifier.
- The framework utilizes diffusion-based purification to cleanse sensory inputs at upload time and counterfactual consistency checks to detect anomalous memory influence during agent planning.
- This approach effectively secures LLMs against multimodal memory poisoning by identifying and neutralizing "sleeper agent" triggers that would otherwise hijack long-term planning.

---

[Self-Reasoning Agentic Framework for Narrative Product Grid Collage Generation](http://arxiv.org/abs/2604.16958)

- Self-Reasoning Agentic Framework: introduces a structured approach for narrative product grid collage generation by decoupling narrative reasoning from pixel synthesis through an iterative Ideation-Generation-Critique loop.
- The framework utilizes specialized LLM-based agents, including Reference-, Ideation-, Generation- and Critique-agents, to ensure visual coherence and narrative richness across multi-panel product collages.
- By employing a hierarchical two-gate evaluation strategy, the system performs targeted refinement to bridge the gap between high-level design intent and final photographic execution.

---

[AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction](http://arxiv.org/abs/2604.16950)

- AutoPKG: introduces a multi-agent LLM framework that automatically constructs and continually evolves a PKG from multimodal product listings using specialized agents and a centralized KGD.
- The framework utilizes a KGD agent to mediate all updates through a constrained action space, ensuring global consistency and canonicalization of product types, attribute keys, and values.
- AutoPKG enables dynamic schema evolution by eliminating the need for fixed taxonomies, supporting automatic type induction, key discovery, and multimodal value extraction in a continually updated graph.

---

[MEMRES: A Memory-Augmented Resolver with Confidence Cascade for Agentic Python Dependency Resolution](http://arxiv.org/abs/2604.16941)

- MemRes: introduces an agentic system for Python dependency resolution that utilizes a multi-level confidence cascade to prioritize deterministic resolution paths over LLM inference.
- The system integrates Intra-Session Memory, a Self-Evolving Memory for pattern reuse, and an Error Pattern Knowledge Base to resolve dependencies with high accuracy while minimizing LLM usage.
- By employing a confidence cascade and system dependency injection, MemRes achieves an 86.6% resolution rate on the HG2.9K benchmark, significantly outperforming existing LLM-based pipelines.

---

[ClimAgent: LLM as Agents for Autonomous Open-ended Climate Science Analysis](http://arxiv.org/abs/2604.16922)

- ClimAgent: introduces an autonomous framework for open-ended climate science analysis by integrating a unified Climate Environment with specialized agents for problem decomposition, physical modeling, and computational execution.
- The framework utilizes a multi-stage agentic workflow comprising Problem Analysis, Climate Modeling, Computational Solving, and Solution Reporting to bridge the gap between unstructured queries and rigorous scientific analysis.
- ClimAgent is evaluated on ClimaBench, a comprehensive data-driven benchmark of 320 real-world climate problems, demonstrating superior performance in solution rigor and practicality compared to existing LLM-based baselines.

---

[Freshness-Aware Prioritized Experience Replay for LLM/VLM Reinforcement Learning](http://arxiv.org/abs/2604.16918)

- FreshPER: introduces a prioritized experience replay mechanism for LLM and VLM reinforcement learning that mitigates priority staleness using a multiplicative exponential age decay.
- The framework integrates a trajectory-level replay buffer with an asynchronous CPU thread that dynamically adjusts sample priorities based on their age relative to the current policy.
- FreshPER significantly improves sample efficiency and performance on multi-step agentic and reasoning tasks by balancing informativeness-driven sampling with temporal freshness.

---

[UNIFIED ULTRASOUND INTELLIGENCE TOWARD AN END-TO-END AGENTIC SYSTEM](http://arxiv.org/abs/2604.16914)

- USTri: introduces a tri-stage ultrasound intelligence pipeline that evolves from a universal generalist USGen, to parameter-efficient specialists USpec, and finally to a clinically oriented agentic system USAgent.
- The framework utilizes a shared TransUNet-style backbone to learn transferable ultrasound priors, which are then specialized via dataset-specific heads and orchestrated by a VLM planner to perform multi-step clinical workflows.
- USAgent mimics clinician workflows by selecting appropriate specialists from a library, maintaining a state cache of intermediate results, and producing deterministic structured reports without free-form generation.

---

[Skilldex: A Package Manager and Registry for Agent Skill Packages with Hierarchical Scope-Based Distribution](http://arxiv.org/abs/2604.16911)

- Skilldex: introduces a package manager and registry for LLM agent skill packages that provides compiler-style format conformance scoring and a skillset abstraction for maintaining cross-skill behavioral coherence.
- The system utilizes a three-tier hierarchical scope system (global, shared, project) and an LLM-powered suggestion loop to manage agent capabilities effectively.
- Skilldex includes a metadata-only registry and an MCP server, enabling agents to manage their own skill environments natively without manual terminal intervention.

---

[PRISM: Probing Reasoning, Instruction, and Source Memory in LLM Hallucinations](http://arxiv.org/abs/2604.16909)

- PRISM: introduces a diagnostic benchmark that disentangles LLM hallucinations into four distinct failure dimensions: Knowledge Error (KE), Knowledge Missing (KM), Reasoning Error (RE), and Instruction Following Error (IFE).
- The framework utilizes a multi-agent pipeline comprising Schema Normalizer Agent, Evidence Retriever Agent, Type Classifier Agent, and Quality Scoring Agent to construct high-quality, orthogonal evaluation instances.
- Experimental results across 24 LLMs reveal significant performance trade-offs, demonstrating that mitigation strategies often improve specific dimensions at the expense of others.

---

[ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design](http://arxiv.org/abs/2604.16896)

- ProtoCycle: introduces an agentic framework for protein design that casts the process as an iterative cycle of planning, tool calling, evaluation, and reflection.
- The framework utilizes an LLM planner to decompose requirements and manage a lightweight tool environment, enabling feedback-driven strategy updates.
- ProtoCycle achieves competitive foldability and superior language alignment compared to end-to-end baselines by optimizing the planner through supervised fine-tuning and online reinforcement learning.

---

[Chain Of Interaction Benchmark (COIN): When Reasoning Meets Embodied Interaction](http://arxiv.org/abs/2604.16886)

- COIN (Chain Of INteraction) Benchmark: introduces a comprehensive benchmark for evaluating interactive reasoning in embodied agents through COIN-50, COIN-Primitive, and COIN-Composition, supported by a low-cost COIN-teleoperation system.
- The framework evaluates H-VLA architectures, which include System 2 (High-level planner) and System 1 (Low-level executor), against CodeAsPolicy approaches to identify critical gaps in visual understanding and motor execution.
- Experimental results demonstrate that current AI models struggle with interactive reasoning, achieving significantly lower success rates than human baselines due to limitations in planning, generalization, and VLM-VLA integration.

---

[GRAIL: Autonomous Concept Grounding for Neuro-Symbolic Reinforcement Learning](http://arxiv.org/abs/2604.16871)

- GRAIL (Grounding Relational Agents through Interactive Learning): introduces a neuro-symbolic framework that autonomously grounds relational concepts by aligning differentiable valuation functions with LLM-generated semantic priors.
- The framework utilizes a two-stage training protocol to resolve circular dependencies between concept learning and policy optimization, enabling agents to discover environment-specific spatial representations.
- By incorporating a concept alignment loss, GRAIL balances reward maximization with semantically faithful grounding, allowing for interpretable, goal-directed behavior in complex Atari environments.

---

[Governed MCP: Kernel-Level Tool Governance for AI Agents via Logit-Based Safety Primitives](http://arxiv.org/abs/2604.16870)

- Governed MCP: introduces a kernel-resident governance gateway that interposes on every MCP tool call to enforce semantic safety below the agent's privilege boundary.
- The architecture utilizes a six-layer pipeline including schema validation, trust tier checks, rate limiting, adversarial pre-filtering, a ProbeLogits semantic gate, and constitutional policy matching.
- By implementing governance as an OS primitive within Anima OS, the system prevents bypasses common in userspace safety libraries and ensures FAIL-CLOSED semantics for all tool invocations.

---

[Learning to Trade Like an Expert: Cognitive Fine-Tuning for Stable Financial Reasoning in Language Models](http://arxiv.org/abs/2604.16862)

- Cognitive Financial Reasoning Framework: introduces a structured training and evaluation pipeline for LLMs in financial trading, utilizing a curated MCQ dataset verified by an AI committee and enhanced with CORA reasoning traces and DARA augmentation.
- The framework employs a two-stage evaluation protocol that assesses both static financial reasoning accuracy and sequential trading performance within a simulated portfolio environment.
- By fine-tuning LLMs on structured reasoning traces and robust data, the approach enables smaller-scale open models to achieve competitive, risk-aware trading performance while mitigating shortcut learning.

---

[enclawed: A Configurable, Sector-Neutral Hardening Framework for Single-User AI Assistant Gateways](http://arxiv.org/abs/2604.16838)

- enclawed: introduces a hard-fork hardening framework for personal AI assistant gateways that integrates Classification Lattice, Egress Guard, Audit Log, DLP Scanner, Human-in-the-loop Controller, Transaction Buffer, Zero-trust Key Broker, Module Signing System, and Trust Root.
- The framework provides a two-flavor deployment model that enforces strict security policies, including mandatory module signing and attested peer trust, to secure LLM interactions in regulated environments.
- enclawed utilizes a data-driven classification scheme and a hash-chained audit mechanism to ensure continuous security verification and tamper-evident logging for locally-hosted LLM applications.

---

[Bias in the Loop: Auditing LLM-as-a-Judge for Software Engineering](http://arxiv.org/abs/2604.16790)

- LLM-as-a-Judge Framework: introduces a measurement-first approach to audit LLM-as-a-Judge systems for software engineering by systematically isolating prompt-induced biases and quantifying test-retest reliability.
- The study evaluates three LLM judges across code generation, repair, and test generation tasks, demonstrating that performance is highly sensitive to non-semantic prompt cues and positional order.
- The research establishes that prompt-induced biases act as positional priors, significantly distorting evaluation outcomes and undermining the reproducibility of LLM-based code assessment.

---

[Federation over Text: Insight Sharing for Multi-Agent Reasoning](http://arxiv.org/abs/2604.16778)

- FoT (Federation over Text): introduces a federated-like framework that enables multiple independent agents to collaboratively build a shared Insight Library by iteratively distilling Reasoning Traces into reusable metacognitive principles.
- The framework operates in the semantic text space, allowing Agents to share abstract reasoning summaries with a Central Server without exposing raw problem instances or requiring gradient-based optimization.
- FoT improves reasoning effectiveness and efficiency across diverse domains by enabling agents to leverage the collective Insight Library to skip redundant steps and apply proven strategies to new tasks.

---

[CapSeal: Capability-Sealed Secret Mediation for Secure Agent Execution](http://arxiv.org/abs/2604.16762)

- CapSeal: introduces a capability-sealed mediation architecture that replaces direct secret access with constrained, broker-mediated invocations to mitigate use-time credential exposure in LLM agents.
- The framework utilizes a Broker (centralized capability mediation component) to act as a Policy Enforcement Point and Policy Decision Point, ensuring that an untrusted Agent Runtime (untrusted LLM-based agent environment) never directly accesses sensitive credentials.
- CapSeal enforces security through a combination of session-bound handles, schema-constrained HTTP Executor (enforces schema-constrained HTTP requests), and SSH Executor (enforces constrained remote SSH commands) with tamper-evident Audit Log (append-only hash-chained transaction record) for accountability.

---

[Privacy-Aware Machine Unlearning with SISA for Reinforcement Learning–Based Ransomware Detection](http://arxiv.org/abs/2604.16760)

- SISA (Sharded, Isolated, Sliced, and Aggregated) Framework: introduces a privacy-aware machine unlearning approach for RL-based ransomware detection by partitioning data into shards, training independent DQN or DDQN agents, and using majority-vote aggregation.
- The framework enables efficient data deletion by retraining only the specific shard containing the affected samples, significantly reducing computational overhead compared to full-model retraining.
- Experimental results demonstrate that the SISA-based DDQN model maintains near-perfect detection performance while achieving negligible utility degradation following one-shard unlearning.

---

#### 17th April 2026


[Self-Organization to the Edge of Ergodicity Breaking in a Complex Adaptive System](http://arxiv.org/abs/2604.15669)

- EvoSK: introduces a minimal evolutionary model where agents perform memory-dependent reinforcement learning on a rugged Sherrington-Kirkpatrick landscape to achieve collective optimality.
- The framework utilizes an evolutionary replacement mechanism to drive the system toward a critical state at the edge of ergodicity breaking.
- The system exhibits scale-free evolutionary avalanches and superior collective performance compared to non-evolutionary baselines without explicit external tuning.

---


[Veritas-RPM: Provenance-Guided Multi-Agent False Positive Suppression for Remote Patient Monitoring](http://arxiv.org/abs/2604.16081)

- Veritas-RPM: introduces a provenance-guided multi-agent architecture designed to suppress false positive alerts in remote patient monitoring by separating verified ground-truth inputs from LLM inference.
- The architecture utilizes a five-layer pipeline including VeritasAgent, SentinelLayer, DirectorAgent, six domain-specific Specialist Agents, and MetaSentinelAgent to ensure clinical decisions are based on provenance-tagged data.
- Simulation-based validation on a 98-case taxonomy demonstrates an 83.7% True Suppression Rate, with failures localized to multi-signal conflict resolution and system-level flags.

---


[FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation](http://arxiv.org/abs/2604.16298)

- FineCog-Nav: introduces a zero-shot framework for UAV navigation that integrates fine-grained, cognition-inspired modules including Instruction Parser, Subgoal Extractor, Perception Module, Attention Module, Imagination Module, Memory Module, Subgoal Judger, Decision-Making Module, and Safety Module.
- The framework utilizes moderate-sized foundation models to simulate human cognitive functions, enabling interpretable and collaborative navigation without relying on large-scale models.
- The authors also introduce AerialVLN-Fine, a curated benchmark with sentence-level instruction-trajectory alignment to support fine-grained evaluation of UAV navigation performance.

---

[Beyond Distribution Sharpening: The Importance of Task Rewards](http://arxiv.org/abs/2604.16259)

- KL-regularized RL framework: introduces a unified training pipeline to isolate and compare the effects of distribution sharpening and task-reward-based optimization on LLM performance.
- The study demonstrates that while distribution sharpening can yield modest gains, it is fundamentally limited by unfavorable optima and instability, particularly in variable-length generation.
- Experimental results across multiple math reasoning datasets confirm that task-reward optimization consistently delivers more stable and robust performance improvements than distribution sharpening alone.

---

[Investigating Conversational Agents to Support Secondary School Students Learning Computer Science Principles](http://arxiv.org/abs/2604.16213)

- Aida and ChatGPT: this paper evaluates the effectiveness and student engagement of fixed-response and generative conversational agents compared to conventional web search for secondary school students learning Computer Science Principles.
- The study utilizes Aida, a custom fixed-response agent built on DialogFlow, and ChatGPT, a general-purpose LLM, to support exploratory search tasks within the CSP curriculum.
- Results indicate that while ChatGPT provides more complete answers and is preferred by students, Aida demonstrates higher accuracy and lower extraneous information, highlighting trade-offs between pedagogical structure and generative flexibility.

---

[ChemGraph-XANES: An Agentic Framework for XANES Simulation and Analysis](http://arxiv.org/abs/2604.16205)

- ChemGraph-XANES: introduces an agentic framework that automates XANES simulation workflows by unifying natural-language task specification, structure acquisition, FDMNES input generation, task-parallel execution, and spectral curation.
- The framework utilizes an agentic orchestration layer with planner-, executor-, and retrieval-augmented expert-agents to map natural-language queries or file inputs into structured tool calls for physics-based spectroscopy software.
- By leveraging Parsl for task-parallel execution on HPC systems, the framework enables scalable, reproducible XANES database generation for downstream analysis and machine learning applications.

---

[Real-Time Solution-Seeking for Game-Theoretic Autonomous Driving via Time-Distributed Iterations](http://arxiv.org/abs/2604.16184)

- GT-MPC: introduces time-distributed Newton and Newton-Kantorovich methods to solve game-theoretic autonomous driving problems by distributing solution-seeking iterations over time.
- The framework utilizes potential function optimization and best response dynamics to approximate Nash equilibria at each sampling instant with limited iterations.
- Numerical experiments demonstrate that the Newton-Kantorovich method reduces computational time compared to standard Newton methods while maintaining bounded approximation errors.

---

[MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](http://arxiv.org/abs/2604.16175)

- MARCH (Multi-Agent Radiology Clinical Hierarchy): introduces a hierarchical multi-agent framework that emulates radiology department workflows to improve the accuracy of 3D radiology report generation, utilizing Resident Agent, Fellow Agents, and Attending Agent.
- The framework employs a three-stage process consisting of initial report drafting, retrieval-augmented report revision, and consensus-driven finalization to mitigate clinical hallucinations and diagnostic discrepancies.
- MARCH integrates specialized medical tools and LLMs to perform multi-round clinical meetings, ensuring that final reports are clinically coherent and aligned with human-like verification standards.

---

[Spinning Living Crystals of Run-and-Tumble Particles with Environmental Feedback](http://arxiv.org/abs/2604.16163)

- SLC framework: introduces a mechanism where dynamic environmental feedback coordinates non-chiral active particles into living crystals exhibiting sustained collective solid-like rotation.
- The framework utilizes an effective torque feedback mechanism that steers active particles away from passive Brownian obstacles, stabilizing larger crystal structures at intermediate densities.
- Collective spinning emerges specifically for superdiffusive particles with long persistence times, driven by a synergistic interplay between perimeter and core particles within the crystal structure.

---

[Compositional Design, Implementation, and Verification of Swarms (Technical Report)](http://arxiv.org/abs/2604.16097)

- Swarm protocols framework: introduces a compositional approach for the design, implementation, and verification of distributed swarm systems using interfacing roles to enable modular reuse of swarm components.
- The framework utilizes a novel branch-tracking mechanism within machine semantics to ensure eventual fidelity of swarms to protocols, allowing machines to correctly reconcile inconsistencies in local views.
- The approach provides automated algorithms for computing well-formed subscriptions and adapting machines, facilitating the composition of independently designed swarm protocols without requiring monolithic redesign.

---

[GroupEnvoy: A Conversational Agent Speaking for the Outgroup to Foster Intergroup Relations](http://arxiv.org/abs/2604.16095)

- GroupEnvoy: introduces a novel AI-mediated contact paradigm where a conversational agent, powered by Gemini-3-pro-preview, utilizes persona data, outgroup proposals, and discussion logs to represent outgroup perspectives during ingroup discussions via a chat interface.
- The framework grounds agent responses in empirical data from prior outgroup-only sessions to ensure authentic representation while facilitating interactive dialogue to reduce intergroup anxiety.
- The study demonstrates that while AI-mediated contact effectively reduces intergroup anxiety through interactive engagement, it requires careful calibration to prevent sycophancy and maintain individual outgroup representation.

---

[Convergence Time Distributions for Max-Consensus over Unreliable Networks](http://arxiv.org/abs/2604.16069)

- LiFE-CD: introduces a deterministic algorithm for computing the full probability distribution of convergence time in max-consensus protocols under Bernoulli-distributed link failures.
- The framework utilizes shortest-path spanning trees to decompose network topologies into unicast and broadcast transmission modes, enabling recursive calculation of convergence time distributions.
- LiFE-CD provides exact convergence time distributions for acyclic networks and tight upper bounds for cyclic networks, offering superior computational efficiency and reproducibility compared to Monte Carlo simulations.

---

[AstroVLM: Expert Multi-agent Collaborative Reasoning for Astronomical Imaging Quality Diagnosis](http://arxiv.org/abs/2604.16024)

- AstroVLM: introduces a multi-agent collaborative framework for astronomical imaging quality diagnosis, utilizing AstroSight, ASK-RAG, and RwB to identify error causes.
- The framework employs ASK-RAG to partition knowledge graphs into agent-specific sub-graphs, reducing noise and improving retrieval relevance for complex diagnostic tasks.
- The Reasoning with Backtracking (RwB) process constructs a Collaborative Reasoning Tree (CRT) to enable agents to re-examine previous processes and identify root causes of imaging errors.

---

[SocialGrid: A Benchmark for Planning and Social Reasoning in Embodied Multi-Agent Systems](http://arxiv.org/abs/2604.16022)

- SocialGrid: introduces a controllable, embodied multi-agent benchmark designed to evaluate LLMs on spatial planning, task execution, and adversarial social reasoning.
- The framework includes a Planning Oracle to isolate social reasoning from navigation deficits, revealing that LLMs struggle with spatial planning and rely on shallow heuristics for social deduction.
- SocialGrid provides automated failure diagnostics and a competitive leaderboard based on adversarial league play to track progress in embodied intelligence.

---

[Neurosymbolic Repo-level Code Localization](http://arxiv.org/abs/2604.16021)

- LogicLoc: introduces a neuro-symbolic framework that integrates LLMs with Datalog to perform precise, keyword-agnostic repository-level code localization.
- The framework utilizes static analysis to construct a structured intermediate representation of program facts, which are then queried by a deterministic Datalog engine to ensure sound and verifiable localization.
- LogicLoc incorporates a synthesize-check-refine loop with mutation-based diagnostic feedback to iteratively improve the quality of LLM-generated Datalog programs and mitigate hallucinations.

---

[MemExplorer: Navigating the Heterogeneous Memory Design Space for Agentic Inference NPUs](http://arxiv.org/abs/2604.16007)

- MemExplorer: introduces a memory system synthesizer that systematically explores heterogeneous memory hierarchies for agentic LLM inference NPUs using System Model, Data Movement Model, Workload Specialization, and System Co-Design Exploration.
- The framework integrates diverse memory technologies including 3D-Stacked SRAM, HBM, LPDDR, GDDR, and HBF to balance throughput and power efficiency across prefill and decode stages.
- MemExplorer employs Multi-Objective Bayesian Optimization to navigate the vast design space of compute and memory configurations, achieving significant energy efficiency gains over baseline NPU and H100 architectures.

---

[AgentV-RL: Scaling Reward Modeling with Agentic Verifier](http://arxiv.org/abs/2604.16004)

- Agentic Verifier: introduces a multi-turn, tool-augmented reward modeling framework that employs Forward Agent, Backward Agent, Synthetic Data Engine, Rejection Fine-tuning, Reinforcement Learning, Python Interpreter, Memory, and Verdict Module to enhance reasoning reliability.
- The framework utilizes a bidirectional "Plan-Validate-Verdict" strategy where the Forward Agent ensures logical sufficiency and the Backward Agent verifies necessity, effectively mitigating error propagation common in traditional reward models.
- AgentV-RL distills these multi-agent capabilities into a single LLM through a two-stage training recipe, achieving state-of-the-art performance on mathematical reasoning benchmarks by interleaving internal reasoning with external tool-use.

---

["When I see Jodie, I feel relaxed": Examining the Impact of a Virtual Supporter in Remote Psychotherapy (Preprint)](http://arxiv.org/abs/2604.16003)

- Jodie (Virtual Supporter): introduces a dual-mode virtual agent designed to provide emotional support and observational continuity in remote psychotherapy sessions.
- The system utilizes Soul Machines HumanOS and Dialogflow CX to deliver a rule-based, embodied conversational agent that functions as a comforter and observer within Zoom-based therapy.
- Empirical evaluation demonstrates that the virtual supporter enhances psychological safety and emotional articulation without disrupting the therapeutic alliance between client and therapist.

---

[Weak-Link Optimization for Multi-Agent Reasoning and Collaboration](http://arxiv.org/abs/2604.15972)

- WORC: introduces a multi-agent reasoning optimization framework that identifies and compensates for performance-limiting "weak agents" using a meta-learning-based weight predictor and swarm intelligence algorithms.
- The framework utilizes a two-stage workflow consisting of weak agent localization via task-specific weight prediction and targeted reasoning budget allocation to improve system reliability.
- Experimental results demonstrate that WORC enhances reasoning accuracy and stability across various multi-agent architectures by dynamically redistributing computational resources to underperforming agents.

---

[Integrating Graphs, Large Language Models, and Agents: Reasoning and Retrieval](http://arxiv.org/abs/2604.15951)

- Graph-LLM Integration Frameworks: introduces a structured taxonomy of methods for combining LLMs with graph-based representations to enhance reasoning, retrieval, and decision-making capabilities.
- The paper categorizes integration strategies into LLM-assisted graph construction, graph-enhanced LLM reasoning, hybrid GNN-LLM models, and agentic workflows.
- It identifies key design patterns and challenges, including scalability, hallucination mitigation, and the alignment between latent LLM reasoning and explicit graph structures.

---

[Online Trading as a Secretary Problem Variant](http://arxiv.org/abs/2604.15933)

- SPVT (Secretary Problem Variant Trading): introduces a framework for online trading between a single seller and n buyers, where an intermediary must make irrevocable decisions to maximize social welfare.
- The paper establishes a tight strong competitive ratio of approximately 3.523 and investigates weak competitive ratios, providing a 2-competitive algorithm and a 1.83683-competitive double-threshold algorithm for the zero-price seller case.
- The research utilizes linear programming and Ramsey's theorem on infinite hypergraphs to derive tight bounds for both strong and weak competitive ratios in online trading scenarios.

---

[Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents](http://arxiv.org/abs/2604.15877)

- Experience Compression Spectrum: introduces a unifying framework that positions agent memory, skills, and rules along a single axis of increasing compression to optimize context usage and retrieval efficiency.
- The framework identifies a "missing diagonal" in current LLM agent research, where systems operate at fixed compression levels without adaptive cross-level selection or upward/downward knowledge consolidation.
- The authors propose a scalable architecture featuring a meta-controller, promotion/demotion engines, and a lifecycle manager to enable continuous, autonomous knowledge management across the compression spectrum.

---

[TINYMU: A COMPACT AUDIO-LANGUAGE MODEL FOR MUSIC UNDERSTANDING](http://arxiv.org/abs/2604.15849)

- TinyMU: introduces a compact 229M parameter music-language model that achieves performance competitive with much larger models by leveraging MATPAC++ (Self-supervised audio feature extractor), Projector (Maps audio to language space), SmolLM2 (Generates text from audio embeddings), and MusicSkills-3.5M (Diverse music QA training data).
- The framework utilizes a lightweight linear projector to align audio features from the MATPAC++ encoder with the SmolLM2 language model for efficient music reasoning.
- The authors introduce MusicSkills-3.5M, a large-scale, multi-format dataset designed to enhance the perceptual and reasoning capabilities of small-scale music language models.

---

[CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](http://arxiv.org/abs/2604.15840)

- CoEvolve: introduces an agent-data mutual evolution framework that enables LLMs to improve through closed-loop, interaction-driven training by alternating between agent optimization and data distribution updates.
- The framework utilizes feedback signals—specifically forgetting, boundary, and rare signals—to guide LLM-based re-exploration and task synthesis, ensuring the training data adapts to the agent's evolving weaknesses.
- By incorporating environment validation for synthesized tasks, CoEvolve maintains high-quality, diverse training data without human supervision, significantly outperforming static training baselines on interactive benchmarks.

---

[Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](http://arxiv.org/abs/2604.15839)

- DAP: introduces an agentic framework for Hard Mode automated theorem proving that decouples natural-language answer discovery from formal proof construction.
- The framework utilizes a Discovery Module to generate and self-verify solutions, which are then used to rewrite formal statements for a Proving Module that employs a theorem prover to construct rigorous proofs.
- By requiring independent discovery of answers before formalization, DAP addresses semantic misalignments in existing benchmarks and achieves state-of-the-art performance on PutnamBench and CombiBench.

---

[Watching Movies Like a Human: Egocentric Emotion Understanding for Embodied Companions](http://arxiv.org/abs/2604.15823)

- ESE: introduces a benchmark dataset and a multimodal reasoning framework for understanding viewer-level emotions in egocentric screen-view movie-watching scenarios.
- The framework utilizes a memory-inspired hierarchical strategy to compress long-term narrative history into structured textual abstractions, which are integrated with short-term visual and audio inputs to guide an LLM-based agent.
- By employing confidence-summed voting for robust supervision and LoRA-based fine-tuning, the approach achieves competitive performance in generating empathetic feedback under realistic, degraded egocentric viewing conditions.

---

[Exploring Agentic Visual Analytics: A Co-Evolutionary Framework of Roles and Workflows](http://arxiv.org/abs/2604.15813)

- AVA (Agentic Visual Analytics): introduces a co-evolutionary framework that deconstructs agentic systems into four specialized roles—PLANNER, CREATOR, REVIEWER, and CONTEXTMANAGER—to automate and enhance the traditional visual analytics pipeline.
- The framework maps these agentic roles across four levels of autonomy, illustrating the shift in human involvement from direct command to high-level strategic supervision.
- The research provides actionable design guidelines for balancing agent autonomy with human-in-the-loop control, addressing challenges such as orchestration latency, visual detail blindness, and human-AI alignment.

---

[From Intention to Text: AI-Supported Goal Setting in Academic Writing](http://arxiv.org/abs/2604.15800)

- WriteFlow: introduces a voice-based AI writing assistant that scaffolds metacognitive regulation and reflection-in-action through iterative goal setting and monitoring.
- The framework utilizes a dialogic interface to support goal articulation, refinement, and alignment tracking between user intentions and emerging text.
- WriteFlow shifts the role of AI from an automated content generator to a negotiable partner that fosters writer agency and critical engagement.

---

[MemEvoBench: Benchmarking Memory MisEvolution in LLM Agents](http://arxiv.org/abs/2604.15774)

- MemEvoBench: introduces a standardized benchmark for evaluating long-horizon memory safety in LLM agents against memory misevolution, where contaminated or biased memory accumulation triggers abnormal agent behaviors.
- The framework utilizes a dual-track approach comprising QA-style tasks with misleading memory injection and workflow-style tasks with noisy tool returns to simulate memory evolution across multi-round interactions.
- Experiments demonstrate that LLMs are highly susceptible to memory-induced safety degradation, which is further amplified by biased user feedback, and that active memory correction tools outperform static prompt-based defenses.

---

[Fuzzy Logic Theory-based Adaptive Reward Shaping for Robust Reinforcement Learning (FARS)](http://arxiv.org/abs/2604.15772)

- FARS (Fuzzy Logic Theory-based Adaptive Reward Shaping): introduces a fuzzy-logic-based reward shaping method that integrates human-inspired heuristics into RL to improve training stability and performance in autonomous drone racing.
- The framework utilizes Fuzzifier, Inference, and Defuzzifier components to map drone velocity and distance states into adaptive reward signals, replacing manually tuned crisp reward functions.
- Experimental results demonstrate that FARS achieves faster convergence and more consistent success rates across varying difficulty levels compared to traditional non-fuzzy reward formulations.

---

[Zero-Shot Scalable Resilience in UAV Swarms: A Decentralized Imitation Learning Framework with Physics-Informed Graph Interactions](http://arxiv.org/abs/2604.15762)

- PhyGAIL: introduces a decentralized recovery framework for damaged UAV swarm networks using centralized training with decentralized execution.
- The framework utilizes a physics-informed graph neural network (PhyGNN) to encode directional local interactions with explicit attraction and repulsion for stable swarm coordination.
- PhyGAIL incorporates a scenario-adaptive imitation learning strategy to ensure robust zero-shot transfer from small-scale training to large-scale swarm deployments.

---

[KWBench: Measuring Unprompted Problem Recognition in Knowledge Work](http://arxiv.org/abs/2604.15760)

- KWBench (Knowledge Work Bench): introduces a benchmark for evaluating LLMs on unprompted problem recognition in professional scenarios by testing their ability to identify governing game-theoretic structures from raw inputs.
- The framework utilizes a mandatory gate mechanism to penalize models that produce polished but misframed outputs, effectively decoupling execution quality from problem recognition.
- Evaluation across 16 models reveals that current LLMs consistently default to cooperative framing, failing to identify adversarial dynamics despite possessing the necessary domain knowledge.

---

[Multi-objective Reinforcement Learning With Augmented States Requires Rewards After Deployment](http://arxiv.org/abs/2604.15757)

- MORL: introduces the requirement for continued access to reward signals or proxy reward models in agents utilizing augmented states after deployment.
- The paper demonstrates that non-linear utility functions in MOMDPs necessitate conditioning policies on augmented states, which typically require ongoing reward observation.
- The authors propose using learned reward models as a viable solution to provide proxy rewards in deployment scenarios where ground-truth rewards are unavailable or expensive to obtain.

---

[The World Leaks the Future: Harness Evolution for Future Prediction Agents](http://arxiv.org/abs/2604.15719)

- Milkyway: introduces a self-evolving agent system that maintains a persistent future prediction harness to organize factor tracking, evidence interpretation, and uncertainty handling.
- The system extracts internal feedback from temporal contrasts across repeated predictions on unresolved questions to update the harness before the final outcome is known.
- After question resolution, the final outcome serves as a retrospective check to validate and refine the harness before it is carried forward to subsequent related questions.

---

[GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows](http://arxiv.org/abs/2604.15715)

- GTA-2: introduces a hierarchical benchmark for evaluating LLM agents across both atomic tool-use and complex, long-horizon productivity workflows.
- The framework utilizes a recursive checkpoint-based evaluation mechanism to assess end-to-end deliverable quality rather than predefined execution trajectories.
- Experimental results reveal a significant performance gap between frontier models and smaller-scale models, highlighting the critical role of execution harness design in workflow completion.

---

[Just Type It in Isabelle! AI Agents Drafting, Mechanizing, and Generalizing from Human Hints](http://arxiv.org/abs/2604.15713)

- Smolka-Blanchette algorithm: introduces a formal account of the correct printing problem, where an LLM-powered AI agent drafts and autoformalizes proofs in Isabelle/HOL.
- The research contrasts human-driven and AI-driven formalization workflows to validate the Smolka-Blanchette algorithm's completeness and minimality.
- The study demonstrates that LLMs can act as effective research assistants in programming language metatheory by generating autoformalizable proofs from informal specifications.

---

[VoxMind: An End-to-End Agentic Spoken Dialogue System](http://arxiv.org/abs/2604.15710)

- VoxMind: introduces an end-to-end spoken dialogue framework that integrates a "Think-before-Speak" reasoning mechanism with a Multi-Agent Dynamic Tool Management architecture to enhance agentic capabilities.
- The system utilizes an auxiliary LLM to asynchronously manage a dynamic local tool space, effectively decoupling inference latency from the total number of available tools.
- VoxMind incorporates a dual-channel memory mechanism and is trained on the AgentChat dataset to support complex reasoning, planning, and tool-use in spoken interactions.

---

[Bilevel Optimization of Agent Skills via Monte Carlo Tree Search](http://arxiv.org/abs/2604.15709)

- Bilevel Optimization of Agent Skills via Monte Carlo Tree Search: introduces a framework that optimizes agent skills by separating the process into an Outer Loop (MCTS structure search) and an Inner Loop (LLM-guided content refinement).
- The framework utilizes a Comprehension Stage to convert raw skills into an optimized initialization, employing a Skill Profile to guide the search space for structure configurations.
- Experimental results on the ORQA dataset demonstrate that the bilevel approach improves agent performance by synergistically optimizing both the structural organization and the content of agent skills.

---

[The Price of Paranoia: Robust Risk-Sensitive Cooperation in Non-Stationary Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2604.15695)

- RATTL (Robust Adaptive Trust-Region Learning): introduces a robust MARL algorithm that stabilizes cooperative equilibria by applying Entropic Value-at-Risk (EVaR) to policy gradient variance rather than return distributions, utilizing an online partner model, Bernoulli variance estimator, adaptive trust factor, gradient update modulator, and Price of Paranoia diagnostic.
- The framework resolves the EVaR Paradox by demonstrating that standard distributional robustness applied to returns hinders cooperation, whereas targeting gradient variance induced by partner uncertainty expands the cooperation basin.
- RATTL achieves near-100% cooperation retention in non-stationary environments by dynamically calibrating risk parameters to balance equilibrium stability and sample efficiency without requiring prosocial priors or opponent modeling.

---

[Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](http://arxiv.org/abs/2604.15687)

- Preference Estimation via Opponent Modeling in Multi-Agent Negotiation: introduces a framework that integrates qualitative natural language signals extracted by an LLM with quantitative Bayesian inference to improve opponent preference estimation in multi-party negotiation.
- The approach utilizes an LLM to parse dialogue into structured signals, which are then combined with numerical offer histories through a Bayesian update rule to maintain dynamic belief tracking of opponent preferences.
- Experimental results demonstrate that this integrated method achieves higher full agreement rates and more accurate preference estimation compared to baselines relying solely on numerical data or direct LLM inference.

---

[Long-Term Memory for VLA-based Agents in Open-World Task Execution](http://arxiv.org/abs/2604.15671)

- ChemBot: introduces a closed-loop framework for chemical laboratory automation that integrates hierarchical task decomposition with a dual-layer memory architecture and a progress-aware VLA model.
- The framework utilizes a Scene Describer for visual grounding, a Subtask Generator for incremental planning, and a Progress Head to enable autonomous, closed-loop execution without fixed time-step reliance.
- ChemBot incorporates an asynchronous inference mechanism to mitigate trajectory discontinuities and leverages long-term memory to consolidate successful experimental strategies for future task adaptation.

---

[STARGAZER: A Scalable Model-Fitting Benchmark Environment for AI Agents under Astrophysical Constraints](http://arxiv.org/abs/2604.15664)

- STARGAZER: introduces a high-fidelity benchmark environment for evaluating AI agents on iterative, physics-grounded exoplanet discovery tasks using radial-velocity data, comprising Task Generation, Agentic Environment, and Evaluation Pipeline.
- The framework utilizes a ReAct-style agentic loop equipped with a PythonREPL tool for analysis and a Submit Action interface for proposing planetary configurations.
- Evaluation is performed via an automated pipeline that assesses statistical fit quality and physical recovery accuracy using metrics like ∆BIC, RMS, Match Score, and Planet Count.

---

[Understanding Inference-Time Token Allocation and Coverage Limits in Agentic Hardware Verification](http://arxiv.org/abs/2604.15657)

- CovAgent: introduces a two-tier agentic framework for hardware verification that systematically analyzes inference-time token allocation and classifies residual coverage holes using Base Agent and Enhanced Agent components.
- The framework utilizes a LangGraph State Graph to implement an iterative verification loop, incorporating Router, Agent Node, and Tool Node to optimize stimulus generation and coverage feedback.
- Empirical results demonstrate that domain-specialized workflows significantly reduce token consumption and improve verification efficiency by shifting reasoning effort toward coverage-directed tasks.

---

[The Power of Information for Intermediate States in Contract Design](http://arxiv.org/abs/2604.15636)

- Two-stage delegation process framework: introduces a model for contract design that incorporates intermediate states to capture information revealed during the delegation process, utilizing Pay-halfway Contract, Terminate-halfway Contract, and Standard Contract mechanisms.
- The framework evaluates how intermediate-state information, specifically regarding intrinsic state quality and agent initial actions, allows the principal to improve profit compared to standard outcome-based contracts.
- The research demonstrates that Pay-halfway and Terminate-halfway contracts can significantly outperform standard contracts by strategically leveraging intermediate information to incentivize optimal agent behavior and reduce excessive payments.

---

[Zoro: Active Rules for Reliable Vibe Coding](http://arxiv.org/abs/2604.15625)

- Zoro: introduces a framework that transforms passive rules into active controls by anchoring them to every step of the coding process through Enrich-Enforce-Evolve.
- The system utilizes a CLI-based protocol to force LLMs to provide code evidence and unit tests for each rule before proceeding with task execution.
- Technical evaluation demonstrates that Zoro significantly improves rule adherence in vibe coding by enabling iterative refinement of rules based on grounded enforcement evidence.

---

[AdaVFM: Adaptive Vision Foundation Models for Edge Intelligence via LLM-Guided Execution](http://arxiv.org/abs/2604.15622)

- AdaVFM: introduces an adaptive framework for efficient on-device inference of language-aligned vision foundation models by dynamically adjusting computation based on scene context and task complexity using a cloud-based LLM execution agent and NAS-based supernet.
- The system leverages a frequency-asymmetric design where a lightweight, NAS-optimized Vision Encoder runs continuously on the edge, while a cloud-based LLM agent is invoked infrequently to provide semantic understanding and guide subnet selection.
- By integrating NAS into the vision backbone and utilizing LLM-driven semantic filtering, the framework achieves superior accuracy-efficiency trade-offs, significantly reducing FLOPs and energy consumption for always-on edge intelligence tasks.

---

[Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies](http://arxiv.org/abs/2604.15607)

- Dual-framework study design: introduces a two-pronged experimental paradigm comparing LLM-simulated interactions and human-subject studies to evaluate the joint impacts of personality traits and AI design characteristics in imperfectly cooperative scenarios.
- The research utilizes Sotopia-S4 to simulate dyadic interactions, employing Causal-Nex to identify structural causal relationships between agent attributes—such as transparency, warmth, expertise, adaptability, and theory of mind—and interaction outcomes.
- Findings reveal significant divergences between simulated and human-subject datasets, highlighting that while personality traits dominate simulated outcomes, AI transparency acts as the primary driver of user experience in real-world human-AI interactions.

---

[When do trajectories matter? Identifiability analysis for stochastic transport phenomena](http://arxiv.org/abs/2604.15598)

- Lattice-based random walk model: introduces a computational framework to evaluate the identifiability of transport parameters by comparing count data and individual trajectory data.
- The framework utilizes mean-field PDE approximations as computationally efficient surrogates for stochastic simulations to perform likelihood-based parameter estimation and identifiability analysis.
- The study demonstrates that while count data can be structurally non-identifiable for certain parameters, incorporating trajectory data resolves these challenges and improves inferential precision.

---

[LLMs Corrupt Your Documents When You Delegate](http://arxiv.org/abs/2604.15597)

- DELEGATE-52: introduces a benchmark for evaluating LLM reliability in long-horizon delegated document editing workflows across 52 professional domains.
- The framework utilizes a backtranslation round-trip method, involving a Seed Document, Forward Edit Task, Backward Edit Task, and Reconstructed Document, to measure performance without requiring reference annotations.
- Experimental results demonstrate that current LLMs, including frontier models, silently corrupt documents over long interactions, with degradation exacerbated by document size, interaction length, and distractor context.

---

[Spec2Cov: An Agentic Framework for Code Coverage Closure of Digital Hardware Designs](http://arxiv.org/abs/2604.15606)

- Spec2Cov: introduces an agentic framework that automates hardware code coverage closure by iteratively generating test stimuli from design specifications using an LLM and simulator feedback loop.
- The framework integrates an LLM with a simulator to perform autonomous verification, utilizing testbench templates, batched generation, and context pruning to optimize coverage closure.
- Spec2Cov demonstrates scalability across 26 hardware designs of varying complexity, achieving significant coverage improvements with minimal human intervention.

---

[CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas](http://arxiv.org/abs/2604.15267)

- CoopEval: introduces a comparative benchmark suite for evaluating cooperation-sustaining mechanisms in multiagent LLM societies across diverse social dilemmas.
- The framework evaluates four game-theoretic mechanisms—Repetition, Reputation, Mediation, and Contract—to determine their effectiveness in inducing cooperative outcomes among heterogeneous LLMs.
- Experimental results demonstrate that while modern LLMs consistently defect in unmodified settings, contracting and mediation mechanisms significantly boost cooperation, with evolutionary pressures further enhancing collective welfare.

---

[DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](http://arxiv.org/abs/2604.15163)

- DPC (Dual-Paradigm Consistency): introduces a multi-agent framework that reformulates SQL selection from a probabilistic guessing task into a deterministic verification pipeline using a SLICER agent, TESTER agent, and SOLVER agent.
- The framework constructs a Minimal Distinguishing Database (MDD) to provide a fully observable environment, enabling the SOLVER agent to generate a high-confidence Python reference anchor for validating SQL candidates.
- DPC utilizes a Bipartite Soft-F1 (BS-F1) metric to perform robust cross-paradigm consistency checks, effectively mitigating systematic bias and symbolic blindness in LLMs during the Text-to-SQL selection process.

---

[COEVO: Co-Evolutionary Framework for Joint Functional Correctness and PPA Optimization in LLM-Based RTL Generation](http://arxiv.org/abs/2604.15001)

- COEVO: introduces a co-evolutionary framework that unifies functional correctness and PPA optimization within a single evolutionary loop using Enhanced Testbench, LLM-driven Evolutionary Operators, Adaptive Correctness Gate, 4D Pareto-based Non-dominated Sorting, and Synthesis Repair.
- The framework treats functional correctness as a continuous optimization dimension alongside area, delay, and power, allowing partially correct candidates to serve as stepping stones for architectural improvement.
- By employing 4D Pareto-based non-dominated sorting with configurable intra-level ranking, COEVO preserves PPA trade-off structures without requiring manual weight tuning.

---

[UniDoc-RL: Coarse-to-Fine Visual RAG with Hierarchical Actions and Dense Rewards](http://arxiv.org/abs/2604.14967)

- UniDoc-RL: introduces a unified reinforcement learning framework for visual document RAG that jointly models retrieval, reranking, active visual perception, and reasoning within a single decision process.
- The framework utilizes a hierarchical action space to progressively refine visual evidence from coarse-grained document retrieval to fine-grained image selection and active region cropping.
- UniDoc-RL employs a dense multi-reward scheme optimized via GRPO to provide task-aware supervision for each action, effectively resolving credit assignment challenges in visual RAG.

---

[Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems](http://arxiv.org/abs/2604.14585)

- Practitioner Decision Framework: introduces a two-stage diagnostic protocol to determine if joint optimization is necessary and if prompt optimization is worthwhile for specific LLM pipelines.
- The framework utilizes ANOVA coupling tests to identify agent independence and headroom tests to detect exploitable output structure, effectively replacing the assumption that joint optimization is always required.
- The research demonstrates that prompt optimization gains are highly model-specific and often statistically indistinguishable from random chance, except when tasks possess latent exploitable output structures.

---

[Mind DeepResearch Technical Report](http://arxiv.org/abs/2604.14518)

- MindDR: introduces a collaborative multi-agent framework that decomposes complex research tasks into specialized subtasks handled by a Planning Agent, multiple DeepSearch Agents, and a Report Agent.
- The framework utilizes a four-stage training pipeline comprising SFT cold-start, Search-RL, Report-RL, and preference alignment to optimize agent capabilities without requiring massive foundation models.
- MindDR incorporates an Extended Chain-of-Thought (XoT) memory mechanism to enable effective coordination and context isolation across the multi-agent pipeline during inference.

---

[Aerial Multi-Functional RIS in Fluid Antennas-Aided Full-Duplex Networks: A Self-Optimized Hybrid Deep Reinforcement Learning Approach](http://arxiv.org/abs/2604.14309)

- SOHRL: introduces a hybrid DRL framework that integrates MADQN and MAPPO to jointly optimize AAV-mounted AM-RIS configurations, FA positions, and beamforming in full-duplex networks.
- The framework incorporates an attention-driven state representation to prioritize critical channel features and a meta-PPO agent for autonomous hyperparameter optimization.
- Simulation results demonstrate that the proposed architecture achieves superior energy efficiency compared to conventional half-duplex, rigid antenna, and non-amplified RIS benchmarks.

---

[Optimistic Policy Learning under Pessimistic Adversaries with Regret and Violation Guarantees](http://arxiv.org/abs/2604.14243)

- RHC-UCRL: introduces a model-based RL algorithm that maintains optimism over both agent and adversary policies to ensure safety and optimality under adversarial dynamics.
- The framework employs a rectified penalty approach to decouple reward and constraint objectives, effectively addressing the breakdown of strong duality in robust constrained settings.
- RHC-UCRL explicitly separates epistemic from aleatoric uncertainty using hallucinated transitions to provide sub-linear regret and constraint violation guarantees.

---

[Know When to Trust the Skill: Delayed Appraisal and Epistemic Vigilance for Single-Agent LLMs](http://arxiv.org/abs/2604.16753)

- MESA-S (Metacognitive Skills for Agents, Single-agent): introduces a framework that decouples procedural awareness from execution by utilizing a Controller, Metacognitive Skill Cards, Delayed Procedural Probe, Dual-Confidence Matrix, and High-Confidence Control-Failure Bank.
- The architecture mitigates LLM overthinking and context pollution by enforcing epistemic vigilance and confidence decontamination through structured metacognitive prompt logic.
- Empirical evaluation demonstrates that MESA-S improves routing accuracy and security by explicitly gating tool access based on the divergence between internal parametric certainty and external source-trust.

---

[Don’t Start What You Can’t Finish: A Counterfactual Audit of Support-State Triage in LLM Agents](http://arxiv.org/abs/2604.16752)

- SSTA-32 (Support-State Triage Audit): introduces a diagnostic framework using matched counterfactual edits to evaluate how LLMs triage tasks across four support states: Complete, Clarifiable, Support-Blocked, and Unsupported-Now.
- The paper utilizes a Dual-Persona Auto-Auditing (DPAA) methodology, where a Subject LLM is evaluated by a deterministic Scorer pipeline to bypass LLM-as-a-judge biases.
- Empirical results demonstrate that surfacing a categorical ontology in the prompt activates latent triage capabilities in LLMs, significantly reducing overcommitment compared to standard helpful-assistant prompting.

---

[When Agents Go Quiet: Output Generation Capacity and Format-Cost Separation for LLM Document Synthesis](http://arxiv.org/abs/2604.16736)

- Gen-Pilot: introduces a theoretical framework to prevent output stalling in LLMs by formalizing Output Generation Capacity (OGC) and Format-Cost Separation (FCS) to optimize document synthesis.
- The framework utilizes Adaptive Strategy Selection to choose between direct, chunked, or deferred generation strategies based on current context occupancy and estimated output costs.
- Empirical validation across multiple LLMs demonstrates that deferred rendering eliminates output stalling and reduces token consumption by 48–72% compared to direct generation.

---

[Agentic Large Language Models for Training-Free Neuro-Radiological Image Analysis](http://arxiv.org/abs/2604.16729)

- Agentic Brain MRI Analysis Pipeline: introduces a training-free framework that enables LLMs to perform complex 3D neuro-radiological analysis by autonomously orchestrating specialized preprocessing-, segmentation-, and analysis-tools.
- The system evaluates multiple agent architectures, including single-agent and multi-agent collaborations, to determine the most robust and efficient approach for multi-step clinical workflows.
- The research provides a benchmark dataset of image-prompt-answer tuples to support the rigorous evaluation of agentic planning and tool-use capabilities in medical imaging.

---

[Debate as Reward: A Multi-Agent Reward System for Scientific Ideation via RL Post-Training](http://arxiv.org/abs/2604.16723)

- Dr. GRPO: introduces a reinforcement learning framework that utilizes a deliberative multi-agent judge to provide robust reward signals for scientific idea generation.
- The framework employs a committee of LLM agents to perform methodological decomposition and adversarial verification, effectively mitigating reward hacking in LLMs.
- By integrating this multi-agent judge into an online RL loop with length-normalized token-level advantages, the system significantly improves the novelty and effectiveness of generated scientific research ideas.

---

[Evaluating Tool-Using Language Agents: Judge Reliability, Propagation Cascades, and Runtime Mitigation in AgentProp-Bench](http://arxiv.org/abs/2604.16706)

- AgentProp-Bench (Agent Propagation Benchmark): introduces a 2,000-task benchmark designed to quantify LLM agent judge reliability, characterize error propagation, and evaluate runtime mitigation strategies.
- The framework utilizes a Three-LLM Ensemble Judge to provide validated evaluation, addressing the unreliability of heuristic-based metrics in tool-using agent environments.
- The research includes a three-layer Interceptor that performs schema validation, reasoning-keyword monitoring, and output consistency checking to reduce agent hallucinations during runtime.

---

[Agentic Risk-Aware Set-Based Engineering Design](http://arxiv.org/abs/2604.16687)

- Agentic Risk-Aware Set-Based Engineering Design: introduces a multi-agent framework guided by LLMs to automate early-stage engineering design through a set-based design philosophy, incorporating Human Manager, Coding Assistant, Design Agent, Systems Engineering Agent, Analyst Agent, NeuralFoil Surrogate, Bayesian Surrogate, DeepONet Surrogate, and OpenFOAM Solver.
- The framework integrates formal risk management using Conditional Value-at-Risk (CVaR) to filter design candidates, ensuring robustness against epistemic and aleatory uncertainties in performance predictions.
- The system employs a human-in-the-loop paradigm where specialized LLM agents perform iterative design, sensitivity analysis, and multi-stage filtering, culminating in high-fidelity CFD validation for final design selection.

---

[KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving](http://arxiv.org/abs/2604.16682)

- KAIROS: introduces a context-aware power management system for agentic AI serving that uses agent context as a control signal to jointly manage GPU frequency, concurrency, and request routing.
- The framework utilizes an Agent ID Tracker to enable agent-level tracking, a Context-Aware Per-Instance Controller to prevent thrashing and optimize power, and a Context-Aware Multi-instance Router to consolidate load.
- KAIROS achieves significant power reduction by dynamically balancing performance and memory stability, effectively addressing the unique challenges of stateful and long-lived agentic workloads.

---

[Agentic Frameworks for Reasoning Tasks: An Empirical Study](http://arxiv.org/abs/2604.16646)

- AFRT: introduces a systematic empirical evaluation of 22 agentic frameworks across three reasoning benchmarks to assess performance, efficiency, and consistency.
- The study categorizes frameworks into single-agent, role-based multi-agent, hierarchical, modular, and graph-based architectures to identify key drivers of success and failure.
- Results indicate that performance plateaus are common, and failures are primarily driven by orchestration issues like memory management and context growth rather than reasoning limitations.

---

[AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation](http://arxiv.org/abs/2604.16625)

- AdaExplore: introduces an LLM-based framework for GPU kernel optimization that utilizes an Adaptation Stage to build a cross-task skill memory and an Exploration Stage to perform diversity-preserving tree search.
- The framework improves kernel generation by distilling reusable validity constraints from execution failures and balancing local refinements with structural regenerations.
- AdaExplore achieves significant speedups on KernelBench and FlashInfer-Bench by effectively navigating non-linear optimization landscapes without requiring additional model fine-tuning.

---

[Human Cognition in Machines: A Unified Perspective of World Models](http://arxiv.org/abs/2604.16592)

- Unified World Model framework: introduces a conceptual roadmap for World Models by grounding machine cognition in Cognitive Architecture Theory, incorporating memory, perception, language, reasoning, imagination, motivation, and meta-cognition.
- The paper identifies critical research gaps in intrinsic motivation and meta-cognition, proposing Epistemic World Models as a solution that utilizes global workspaces for scientific discovery.
- The taxonomy classifies World Models across video, embodied, and epistemic domains, providing a principled basis for evaluating claims of human-like cognitive capabilities in LLMs and other generative agents.

---

[The Global Neural World Model: Spatially Grounded Discrete Topologies for Action-Conditioned Planning](http://arxiv.org/abs/2604.16585)

- GNWM: introduces a self-stabilizing framework that achieves topological quantization through balanced continuous entropy constraints to enable action-conditioned planning.
- The architecture utilizes a Retinotopic Encoder, a Spatial Transition Predictor, Topological Smearing, Thermodynamic Equilibrium, Expansion Force, Contraction Force, and Similarity Loss to map environments onto discrete grids.
- This model replaces non-differentiable heuristics with a fully differentiable thermodynamic system to prevent manifold drift and enable robust autoregressive planning.

---

[Certified Program Synthesis with a Multi-Modal Verifier](http://arxiv.org/abs/2604.16584)

- LeetProof: introduces an agentic pipeline for certified program synthesis that decomposes the task into specification generation, program synthesis, and proof construction stages.
- The framework utilizes a multi-modal verifier, Velvet, to combine randomized property-based testing, automated SMT-based proofs, and AI-assisted interactive proof scripting within the Lean theorem prover.
- By matching each synthesis stage to the most cost-effective reasoning mode, LeetProof achieves higher rates of fully certified solutions compared to single-mode baselines while maintaining correctness.

---

[Agentic AI for Education: A Unified Multi-Agent Framework for Personalized Learning and Institutional Intelligence](http://arxiv.org/abs/2604.16566)

- AUSS (Agentic Unified Student Support System): introduces a multi-agent architecture that integrates student-level personalization, educator-level automation, and institutional-level intelligence using LLMs, reinforcement learning, and predictive analytics.
- The framework utilizes a four-module functional design—perception, reasoning, action, and evaluation—to enable autonomous decision-making and continuous adaptation across educational environments.
- An event-driven communication mechanism facilitates real-time coordination between the Student Agent, Educator Agent, and Institution Agent to optimize educational outcomes and operational efficiency.

---

[LLM as a Tool, Not an Agent: Code-Mined Tree Transformations for Neural Architecture Search](http://arxiv.org/abs/2604.16555)

- LLMasTool: introduces a hierarchical tree-based NAS framework that leverages algorithm-based module mining and coarse-to-fine decision making to enable stable, open-ended model evolution.
- The framework utilizes a module database (reusable building blocks extracted from source code), a hierarchical tree representation (deploy-friendly architecture structure), an evolution engine (applies controlled tree transformations), a history database (stores previous experimental results), an architecture database (tracks candidate architectures and metrics), an LLM-based transformation tool (resolves fine-grained architectural decisions), and diversity-guided Bayesian planning (governs coarse-level evolutionary trajectory).
- By treating the LLM as a structured tool rather than an autonomous agent, the framework achieves superior performance and controllability in open-ended NAS tasks.

---

[A Survey on the Security of Long-Term Memory in LLM Agents: Toward Mnemonic Sovereignty](http://arxiv.org/abs/2604.16548)

- Memory-lifecycle analysis framework: introduces a structured approach to analyze LLM agent memory security by decomposing the memory lifecycle into six distinct phases—Write, Store, Retrieve, Execute, Share, and Forget/Rollback—cross-tabulated against four security objectives.
- The paper establishes mnemonic sovereignty as a normative concept for verifiable, recoverable governance over an agent's memory state, emphasizing that memory security is an independent class of system security problems.
- The research identifies that memory attacks often operate through inducement, source confusion, and read-time rewriting rather than brute-force overwrites, and highlights the critical need for lifecycle-wide security defenses.

---

[Conjunctive Prompt Attacks in Multi-Agent LLM Systems](http://arxiv.org/abs/2604.16543)

- Conjunctive Prompt Attacks in Multi-Agent LLM Systems: introduces a topology- and routing-aware attack framework that exploits multi-agent pipelines by aligning benign-appearing trigger keys and injected templates at a compromised agent.
- The framework optimizes trigger placement, template insertion, and routing bias to maximize attack success without modifying LLM weights or client-side logic.
- Empirical results demonstrate that these conjunctive attacks bypass existing safety mechanisms by distributing malicious signals across multiple components, rendering local inspection ineffective.

---

[BOOKAGENT: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](http://arxiv.org/abs/2604.16541)

- BOOKAGENT: introduces a multi-agent framework that treats storybook generation as a collaborative, safety-aware cognitive process by unifying text and image generation through a closed-loop architecture.
- The framework utilizes three distinct mechanisms: Value-Aligned Storyboarding (VAS) for structural planning, Iterative Cross-modal Refinement (ICR) for local grounding, and Temporal Cognitive Calibration (TCC) for global consistency.
- BOOKAGENT incorporates specialized agents including Reviewer-Refiner, Page Planner, Character Extractor, Frame Director, Identity Director, and Sequence Director to ensure narrative coherence and safety compliance.

---

#### 16th April 2026

[Symbolic Guardrails for Domain-Specific Agents: Stronger Safety and Security Guarantees Without Sacrificing Utility](http://arxiv.org/abs/2604.15579)

- Symbolic Guardrails for Domain-Specific Agents: introduces a framework for enhancing AI agent safety and security by implementing deterministic symbolic guardrails around LLM-based agents.
- The research demonstrates that symbolic guardrails, such as API validation and schema constraints, effectively enforce safety policies in domain-specific agents without compromising task utility.
- The study evaluates 80 benchmarks and finds that while most lack concrete policies, symbolic enforcement is a practical, low-cost alternative to probabilistic neural guardrails for high-stakes business environments.

---

[Agentic Explainability at Scale: Between Corporate Fears and XAI Needs](http://arxiv.org/abs/2604.14984)

- Agentic XAI: introduces a governance framework for managing agentic systems at scale by combining design-time documentation and runtime observability techniques.
- The framework addresses enterprise risks such as "Agent Sprawl," permission inheritance, and uncontrolled daisy-chain interactions through Agent Inventory, Agent Cards, and Dependency Graphs.
- Runtime components including Deep Observability, Contextual Traceability, and Operational Monitoring provide the necessary auditability and human-in-the-loop control to mitigate cascading risks in multi-agent environments.

---


[Blinded Multi-Rater Comparative Evaluation of a Large Language Model and Clinician-Authored Responses in CGM-Informed Diabetes Counseling](http://arxiv.org/abs/2604.15124)

- CA: introduces a retrieval-grounded LLM-based conversational agent that provides structured, plain-language explanations of CGM data and diabetes management questions.
- The system integrates GPT-5.1 with RAG to ground responses in clinical guidelines and case-specific CGM metrics, ensuring outputs are clinically relevant and empathetic.
- A blinded multi-rater evaluation by senior diabetes clinicians demonstrated that the CA achieved higher quality scores than clinician-authored responses across multiple clinical domains.

---

[WHERE ARE THE HUMANS? A SCOPING REVIEW OF FAIRNESS IN MULTI-AGENT AI SYSTEMS](http://arxiv.org/abs/2604.15078)

- MAAI systems: introduces a scoping review of fairness in Multi-Agent AI systems, identifying five archetypal approaches—Normative Delegation, Fairness Facade, Fairness Schooling, Petri Dish Fairness, and Fairness Effectiveness—that characterize current research patterns.
- The paper argues that fairness in MAAI systems must be embedded structurally throughout the development lifecycle rather than treated as a post-hoc consideration.
- The authors highlight a critical research gap regarding the underrepresentation of human stakeholders in current MAAI fairness studies, which predominantly rely on isolated, agent-only "petri dish" experiments.

---

[Subliminal Transfer of Unsafe Behaviors in AI Agent Distillation](http://arxiv.org/abs/2604.15559)

- Subliminal Behavioral Transfer Pipeline: introduces an experimental framework to demonstrate that unsafe behavioral traits can be implicitly transferred from a teacher LLM to a student LLM through sanitized training trajectories.
- The framework utilizes Deletion Agent and Student Agent components to show that even when explicit unsafe keywords are removed from training data, structural trajectory dynamics facilitate the inheritance of destructive biases.
- The research confirms that high-capacity teacher models effectively propagate behavioral biases across different architectures, rendering standard keyword-based data sanitation insufficient for ensuring agent safety.

---

[Preregistered Belief Revision Contracts](http://arxiv.org/abs/2604.15558)

- PBRC (Preregistered Belief Revision Contracts): introduces a protocol-level mechanism that strictly separates open communication from admissible epistemic change in multi-agent systems using Router/Orchestrator, Preregistered Contract, Evidence Tokens, Validity Layer, Audit Log, Certificate Mechanism, Fallback Operator, and Revision Operator.
- The framework prevents conformity-driven cascades by requiring that any substantive belief change be justified by a nonempty witness set of validated evidence tokens, ensuring epistemic accountability.
- PBRC provides formal guarantees that social-only interaction cannot amplify confidence or generate wrong-but-sure cascades, while remaining agnostic to the underlying LLM-based revision operators.

---

[PolicyBank: Evolving Policy Understanding for LLM Agents](http://arxiv.org/abs/2604.15505)

- PolicyBank: introduces a memory mechanism that enables LLMs to autonomously refine their interpretation of imperfect natural language policies through interaction and corrective feedback.
- The framework utilizes a dedicated Policy Agent to analyze task trajectories and developer feedback, updating structured tool-level policy insights to close specification-requirement gaps.
- By extending the τ-Bench testbed, the authors demonstrate that PolicyBank significantly improves agent compliance on complex policy-gap scenarios compared to standard memory-based approaches.

---

[vstash: Local-First Hybrid Retrieval with Adaptive Fusion for LLM Agents](http://arxiv.org/abs/2604.15484)

- vstash: introduces a local-first document memory system for LLMs that utilizes MarkItDown, Chunking, FastEmbed, SQLite, Vector ANN, FTS5 BM25, RRF Fusion, Recency Boost, MMR Dedup, Distance Signal, and Context Expansion.
- The framework employs a self-supervised embedding refinement method that leverages retrieval disagreement between dense and sparse modalities to improve performance without human labels.
- vstash achieves competitive retrieval quality on standard benchmarks while maintaining low latency and providing a production-grade substrate for LLM agents.

---

[The Semi-Executable Stack: Agentic Software Engineering and the Expanding Scope of SE](http://arxiv.org/abs/2604.15468)

- The Semi-Executable Stack: introduces a six-ring diagnostic reference model to analyze the expanding scope of software engineering as it incorporates semi-executable artifacts.
- The framework categorizes engineering objects from deterministic code to interpreted organizational routines, emphasizing that agentic systems require engineering discipline across all six rings.
- The paper argues that AI-based agentic systems do not shrink software engineering but broaden its scope, necessitating a shift from purely technical focus to socio-technical engineering.

---

[Evaluating LLM Simulators as Differentially Private Data Generators](http://arxiv.org/abs/2604.15461)

- PersonaLedger: introduces a "Profile-then-Simulate" architecture that leverages AIM (DP synthetic data generator) to create user personas, which are then processed by PersonaLedger (LLM-driven transaction sequence simulator) to generate synthetic financial transaction data.
- The framework utilizes XGBoost (Fraud detection evaluation model) to benchmark the utility of synthetic data against real-world transaction logs using the TSTR (Evaluation protocol for synthetic data) protocol.
- The study identifies that while PersonaLedger achieves moderate utility, systematic LLM biases regarding temporal and demographic features lead to significant distribution drift compared to direct DP synthesis methods.

---

[MM-WebAgent: A Hierarchical Multimodal Web Agent for Webpage Generation](http://arxiv.org/abs/2604.15309)

- MM-WebAgent: introduces a hierarchical agentic framework that coordinates AIGC-based element generation through hierarchical planning and iterative self-reflection to produce coherent webpages.
- The framework integrates Planning Agent, Layout Agent, Image Agent, Video Agent, Chart Agent, Evaluation Manager, Reflection Manager, and MM-WebGEN-Bench to optimize global layout and local multimodal content integration.
- MM-WebAgent employs a multi-level self-reflection mechanism—local, context, and global—to iteratively refine generated assets and layout, significantly outperforming existing code-only generation baselines.

---

[Knowing that you do not know everything](http://arxiv.org/abs/2604.15264)

- Epistemic Framework: introduces a formal model demonstrating that a rational agent with true and refinable knowledge cannot determine if she possesses complete knowledge of all states.
- The framework utilizes an epistemic operator K to model agent knowledge, proving that introspection regarding the lack of knowledge (¬KΩ) consistently yields an empty set.
- The analysis establishes that this epistemic limitation persists even when the agent learns about new events, as introspection provides no additional information regarding the completeness of her knowledge.

---

[HarmfulSkillBench: How Do Harmful Skills Weaponize Your Agents?](http://arxiv.org/abs/2604.15415)

- HarmfulSkillBench: introduces a large-scale measurement study and benchmark to evaluate how harmful skills in agent ecosystems weaponize LLMs by bypassing safety filters.
- The study identifies 4,858 harmful skills across two major registries and demonstrates a skill-reading exploit where pre-installed skills significantly lower LLM refusal rates.
- The research reveals that while Tier 1 prohibited skills are highly sensitive to task framing, Tier 2 high-risk skills require explicit instructions to trigger necessary human-in-the-loop and AI disclosure safeguards.

---

[Agentic Microphysics: A Manifesto for Generative AI Safety](http://arxiv.org/abs/2604.15236)

- Generative Safety Framework: introduces a methodological approach for analyzing multi-agent AI safety by linking macro-level collective risks to micro-level interaction rules through Risk identification, Microspecification, Generative experimentation, Intervention design, and Observational validation.
- The framework utilizes agentic microphysics to reconstruct collective phenomena from explicit local interaction rules, protocol conditions, and architectural constraints within multi-agent LLM populations.
- By treating interaction architectures as design variables, the methodology enables the identification of causal mechanisms and the testing of interventions to enhance the robustness of agentic ecosystems.

---

[Blue Data Intelligence Layer: Streaming Data and Agents for Multi-source Multi-modal Data-Centric Applications](http://arxiv.org/abs/2604.15233)

- DIL (Data Intelligence Layer): introduces a unified framework that treats relational databases, web data, LLMs, and users as first-class data sources to support multi-source, multi-modal, and data-centric applications.
- The architecture utilizes a registry-driven design, an extensible operator hierarchy, and declarative data planning over directed acyclic graphs to orchestrate agents and data.
- DIL enables compound AI systems to bridge the semantic gap between user intent and heterogeneous data by integrating structured enterprise data, LLM-based world knowledge, and interaction-derived context.

---

[RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography](http://arxiv.org/abs/2604.15231)

- RadAgent: introduces an RL-trained agent that performs stepwise chest CT interpretation by orchestrating a suite of specialized tools guided by a diagnostic checklist.
- The framework utilizes an orchestrator LLM to iteratively query tools, maintain a scratchpad of findings, and generate reports with traceable intermediate reasoning steps.
- RadAgent improves clinical accuracy, robustness to adversarial prompts, and faithfulness compared to standard 3D VLMs by anchoring report generation in verifiable tool-based evidence.

---

[Beyond Single-Model Optimization: Preserving Plasticity in Continual Reinforcement Learning](http://arxiv.org/abs/2604.15414)

- TELAPA (Transfer-Enabled Latent-Aligned Policy Archives): introduces a continual RL framework that preserves behaviorally diverse policy neighborhoods in a shared latent space to support rapid adaptation and plasticity.
- The framework utilizes MAP-Elites for archive illumination and a learned trajectory embedder with anchor-based distillation to maintain a navigable latent geometry under non-stationary task drift.
- By shifting from single-model preservation to maintaining skill-aligned neighborhoods, the approach enables effective transfer and recoverability across sequential tasks in MiniGrid environments.

---

[PRL-BENCH: A Comprehensive Benchmark for Evaluating the Capabilities of LLMs in Frontier Physics Research](http://arxiv.org/abs/2604.15411)

- PRL-BENCH: introduces a research-oriented benchmark designed to evaluate LLMs on end-to-end physics research tasks, utilizing Introduction, Subtasks, Answers, Rubrics, Code Interpreter, and LLM-as-judge.
- The benchmark assesses LLM capabilities across five major physics subfields using 100 curated papers from Physical Review Letters to ensure expert-level difficulty.
- Evaluation results reveal that current LLMs struggle with long-horizon reasoning and domain-specific knowledge, with performance gaps primarily driven by conceptual and formulaic errors.

---

[Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines](http://arxiv.org/abs/2604.15186)

- Scepsy: introduces a system for serving multi-LLM agentic workflows by leveraging stable aggregate per-LLM performance statistics to optimize GPU resource allocation.
- The system constructs an Aggregate LLM Pipeline to predict end-to-end workflow performance, enabling joint optimization of fractional GPU shares, tensor parallelism, and model replica counts.
- Scepsy employs a topology-aware placement strategy to minimize cluster fragmentation and enforce resource isolation for heterogeneous LLM components on GPU clusters.

---

[Agent-Aided Design for Dynamic CAD Models](http://arxiv.org/abs/2604.15184)

- AADvark: introduces an agentic system for generating dynamic 3D CAD assemblies by iteratively refining JSON-based designs through visual feedback and constraint solver signals.
- The system utilizes an augmented 3D assembly constraint solver and modified FreeCAD rendering to overcome spatial reasoning limitations in LLMs when modeling complex joints and moving parts.
- AADvark improves agentic performance by providing deterministic visual identifiers and detailed error messages, enabling the construction of functional assemblies like scissors from multimodal inputs.

---

[QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies](http://arxiv.org/abs/2604.15151)

- QuantCode-Bench: introduces a multi-stage evaluation pipeline for assessing LLMs on their ability to generate executable algorithmic trading strategies using Backtrader, Compilation-Stage, Backtest-Stage, Trade-Stage, LLM-as-a-Judge-Stage, and Agentic-Repair-Loop.
- The benchmark utilizes a four-stage validation process to distinguish between technical code correctness, successful backtest execution, actual trading behavior, and semantic alignment with natural-language task descriptions.
- Experimental results demonstrate that while modern LLMs have largely mastered syntactic generation, significant challenges remain in the operational formalization of trading logic and semantic adherence, which are partially mitigated through an Agentic-Repair-Loop.

---

[Feedback-Driven Execution for LLM-Based Binary Analysis](http://arxiv.org/abs/2604.15136)

- FORGE: introduces a feedback-driven execution model that interleaves reasoning and tool interaction through a reasoning–action–observation loop to perform binary analysis.
- The system utilizes a Dynamic Forest of Agents (FoA) to recursively decompose complex analysis tasks into parallel, bounded-context subtasks, mitigating context degradation and search explosion.
- FORGE unifies vulnerability discovery and validation by generating structured evidence chains that allow for replayable verification of identified security flaws.

---

[Combinatorial Contracts Through Demand Types](http://arxiv.org/abs/2604.15125)

- ASC: introduces a unified geometric framework for algorithmic contract design by linking the combinatorial action model to demand-type machinery from consumer theory.
- The paper establishes that the ASC class admits at most O(n^2) critical values, generalizing and unifying previously known classes of reward functions.
- The authors provide a new technique for efficiently computing demand queries using value queries for succinct demand-type families, enabling polynomial-time algorithms for optimal contract design.

---

[Autonomous Evolution of EDA Tools: Multi-Agent Self-Evolved ABC](http://arxiv.org/abs/2604.15082)

- Multi-Agent Self-Evolved ABC: introduces a scalable, multi-agent framework that leverages LLMs to autonomously improve the source code of the ABC logic synthesis system through iterative cycles of code generation, compilation, and formal verification.
- The framework utilizes a central Planning Agent to coordinate specialized Coding Agents that refine distinct subsystems, including flow-tuning, logic optimization, and technology mapping, while adhering to a self-evolving rulebase.
- By integrating formal equivalence checking and large-scale distributed QoR evaluation, the system ensures that all autonomously generated code modifications preserve functional correctness while achieving significant improvements in synthesis quality.

---

[ATROPOS: Improving Cost-Benefit Trade-off of LLM-based Agents under Self-Consistency with Early Termination and Model Hotswap](http://arxiv.org/abs/2604.15075)

- ATROPOS: introduces a predictive framework that optimizes the cost-benefit trade-off of LLM-based agents by utilizing SFG to represent inference trajectories and GCN to perform early termination or model hotswapping.
- The framework employs a GCN to analyze partial inference trajectories, enabling the early termination of likely failing tasks or the hotswapping of the ongoing context from a cost-effective Source LLM to a more capable Target LLM.
- Empirical evaluations demonstrate that ATROPOS achieves 74.35% of the performance of proprietary LLMs while reducing monetary costs to 23.9% of the original expenditure.

---

[CoGrid &amp; the Multi-User Gymnasium: A Framework for Multi-Agent Experimentation](http://arxiv.org/abs/2604.15044)

- CoGrid &amp; MUG: introduces a modular framework for multi-agent experimentation that combines CoGrid (grid-based simulation library) and MUG (web-based experiment platform) to facilitate human-AI interaction studies.
- CoGrid utilizes a dual-backend architecture (NumPy/JAX) to support both rapid prototyping and hardware-accelerated training of agents.
- MUG enables browser-based deployment of simulation environments using Pyodide and GGPO rollback netcode to ensure low-latency, synchronized multi-user interactions.

---

[From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench](http://arxiv.org/abs/2604.15037)

- ProVoice-Bench: introduces a comprehensive evaluation framework for proactive voice agents, featuring four novel tasks: Proactive Intent Capture, Latent Topic Monitor, Context Fact Checking, and Environment Sound Sensing.
- The framework utilizes a multi-stage data synthesis pipeline, incorporating Digital State Construction, Scene Synthesis, Conversation Generation, Acoustic Simulation, and Conversation Assembly to generate high-fidelity, context-aware samples.
- Evaluation of state-of-the-art MLLMs on this benchmark reveals a significant performance gap, particularly regarding over-triggering and the ability to bridge the decision-to-execution divide.

---

[Autogenesis: A Self-Evolving Agent Protocol](http://arxiv.org/abs/2604.15034)

- AGS (Autogenesis System): introduces a two-layer protocol architecture that decouples evolvable agent resources from the mechanisms governing their self-evolution.
- The framework utilizes RSPL to model prompts, agents, tools, environments, and memory as versioned, first-class resources, while SEPL provides a control-theoretic operator algebra for iterative, closed-loop improvement.
- By mediating all state mutations through standardized interfaces, the system ensures that agent evolution is traceable, reversible, and safe-by-construction across heterogeneous multi-agent environments.

---

[Dr. RTL: Autonomous Agentic RTL Optimization through Tool-Grounded Self-Improvement](http://arxiv.org/abs/2604.14989)

- Dr. RTL: introduces an agentic framework for RTL timing optimization that performs closed-loop interaction with industrial EDA tools to improve PPA while distilling optimization trajectories into reusable skills.
- The framework utilizes an orchestrator to coordinate specialized agents for timing analysis, RTL rewriting, and tool-based evaluation, enabling parallel exploration of candidate designs.
- Dr. RTL employs group-relative skill learning to extract interpretable pattern-strategy pairs from optimization trajectories, facilitating continual self-improvement without requiring LLM fine-tuning.

---

[SAGER: Self-Evolving User Policy Skills for Recommendation Agent](http://arxiv.org/abs/2604.14972)

- SAGER: introduces a recommendation agent framework that personalizes how an LLM reasons by maintaining a self-evolving, structured policy skill for each user.
- The framework utilizes a two-representation architecture to resolve the injection paradox, where a full skill repository is distilled into a slim working skill for inference.
- An incremental contrastive CoT engine enables stable policy evolution by generating structured updates based on the contrast between accepted and unchosen items.

---

[DISCOVERING NOVEL LLM EXPERTS VIA TASK-CAPABILITY COEVOLUTION](http://arxiv.org/abs/2604.14969)

- AC/DC (Assessment Coevolving with Diverse Capabilities): introduces a framework that coevolves LLM populations via Model Merging Crossover and Weight Noising Mutation alongside synthetic tasks generated by a Scientist LLM to discover diverse expert models.
- The framework utilizes Skill Vectors and Dominated Novelty Search (DNS) to maintain a Model Archive of diverse, high-quality LLMs while employing a Gibberish Filter and Impossible Task Filter as minimal criteria to ensure population quality.
- By leveraging Vector Databases for task similarity and evolutionary operations, AC/DC discovers complementary LLM collectives that achieve broader skill coverage than monolithic models with significantly lower parameter counts.

---

[FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](http://arxiv.org/abs/2604.14956)

- FedGUI: introduces a comprehensive benchmark for training and evaluating cross-platform GUI agents through federated learning, utilizing Central Server, GUI Agent, Mobile/Web/Desktop Clients, Federated Learning Algorithms, Base Models, and Datasets.
- The framework addresses real-world heterogeneity across platforms, devices, operating systems, and data sources by providing a unified action space and modular pipeline for decentralized agent training.
- Extensive experiments demonstrate that federated collaboration significantly improves agent performance under diverse non-IID data distributions, effectively bridging domain isolation across mobile, web, and desktop environments.

---

[HRDexDB: A Large-Scale Dataset of Dexterous Human and Robotic Hand Grasps](http://arxiv.org/abs/2604.14944)

- HRDexDB: introduces a large-scale, multi-modal dataset of paired human and robotic dexterous grasping sequences across 100 objects, utilizing a multi-camera capture system, egocentric stereo cameras, robotic platforms, tactile sensors, the MANO hand model, FoundationStereo, FoundationPose, and SAM3 segmentation.
- The framework provides high-precision spatiotemporal 3D ground-truth motion and tactile signals to facilitate cross-embodiment policy learning and dexterous manipulation research.
- By capturing both successful and failed grasping trials, the dataset enables the analysis of embodiment-specific physical limits and contact dynamics in real-world manipulation tasks.

---

[IE as Cache: Information Extraction Enhanced Agentic Reasoning](http://arxiv.org/abs/2604.14930)

- IE-as-Cache: introduces a framework that repurposes information extraction as a dynamic cognitive cache to scaffold agentic reasoning by maintaining a compact, read-write memory layer.
- The framework utilizes Query-Driven Information Extraction and Schema-Decoupled Extraction to initialize the cache, while Cache-Aware Agentic Reasoning dynamically updates this memory during multi-step inference.
- By treating raw text as external storage and maintaining only semantically salient information in the cache, the approach mitigates noise and context decay for LLMs in complex reasoning tasks.

---

[A Numerical and Experimental Evaluation of Microbubble Communication Using OpenFOAM](http://arxiv.org/abs/2604.14919)

- OpenFOAM-based simulation framework: introduces a physics-based approach to characterize microbubble transport dynamics for IoBNT communication by integrating a Syringe/Micropump, SonoVue microbubbles, a Recirculating water channel, a Doppler ultrasound sensor, OpenFOAM, the incompressibleDenseParticleFluid solver, and Paraview.
- The framework validates numerical simulations against experimental measurements to assess microbubble propagation under varying flow conditions, including water and blood-like media.
- Results demonstrate that microbubble transport is primarily governed by bulk flow velocity, supporting the feasibility of using microbubbles as robust, low-rate information carriers in confined environments.

---

[ADAPT: Benchmarking Commonsense Planning under Unspecified Affordance Constraints](http://arxiv.org/abs/2604.14902)

- ADAPT: introduces a unified decision-time inference module that augments existing embodied agents with explicit affordance reasoning to handle dynamic, under-specified environments.
- The framework utilizes a LoRA-finetuned LLaVA model and multimodal in-context learning to infer latent object preconditions, enabling agents to defer inapplicable actions and maintain long-horizon task coherence.
- By integrating ADAPT, embodied agents demonstrate significant improvements in robustness and success rates across both seen and unseen environments by effectively resolving dynamic affordance violations.

---

[Toward Agentic RAG for Ukrainian](http://arxiv.org/abs/2604.14896)

- Agentic RAG for Ukrainian: introduces a modular pipeline for Ukrainian document understanding that integrates dense retrieval, reranking, and a lightweight agentic layer to improve answer accuracy.
- The system utilizes BGE-M3 and BGE-reranker for retrieval, while employing Qwen2.5-3B-Instruct as the core LLM to perform query rephrasing and answer retry loops.
- Experimental results demonstrate that while retrieval quality remains the primary bottleneck, agentic retry mechanisms provide consistent, albeit modest, improvements in final performance metrics.

---

[Does RL Expand the Capability Boundary of LLM Agents? A PASS@(k, T) Analysis](http://arxiv.org/abs/2604.14877)

- PASS@(k, T) introduces a two-dimensional evaluation framework that disentangles capability expansion from efficiency improvement in LLM agents by jointly varying the sampling budget k and the interaction-depth budget T.
- The research demonstrates that reinforcement learning (RL) genuinely expands the capability boundary of LLM agents on compositional tool-use tasks through a reweighting mechanism, whereas supervised fine-tuning (SFT) often regresses this boundary.
- Mechanism analysis reveals that RL-driven improvements are concentrated on how the agent integrates retrieved information rather than on the search queries themselves, contrasting with the distribution-replacing behavior of SFT.

---

[SkillDroid: Compile Once, Reuse Forever](http://arxiv.org/abs/2604.14872)

- SkillDroid: introduces a three-layer skill agent that compiles LLM-guided mobile GUI trajectories into reusable parameterized skill templates to bypass repeated LLM inference.
- The framework utilizes a matching cascade of regex, semantic embeddings, and app filtering to route instructions to stored skills, while employing a failure-learning layer to trigger recompilation when reliability degrades.
- By shifting from per-step LLM deliberation to mechanical skill replay, the system achieves higher success rates and lower latency while improving reliability through accumulated experience.

---

[Benchmarks for Trajectory Safety Evaluation and Diagnosis in OpenClaw and Codex: ATBench-Claw and ATBench-CodeX](http://arxiv.org/abs/2604.14858)

- ATBench: introduces a trajectory-level safety evaluation and diagnosis framework that extends to diverse agent execution settings by customizing a shared three-dimensional Safety Taxonomy and utilizing a unified data generation engine.
- The framework enables domain-specific safety benchmarking for OpenClaw and Codex environments by mapping setting-specific risks to a common taxonomy of risk source, failure mode, and real-world harm.
- Empirical results demonstrate that the AgentDoG-configured system maintains robust performance across both customized benchmarks, while fine-grained taxonomy analysis reveals specific difficulty concentrations in repository-native and output-validation tasks.

---

[Seeking Help, Facing Harm: Auditing TikTok’s Mental Health Recommendations](http://arxiv.org/abs/2604.14832)

- LLM-driven simulated agents: introduces a controlled audit of TikTok’s recommendation system using LLM-driven simulated agents to evaluate how initial search framing and interaction strategies influence exposure to mental health content.
- The study employs a 2x3 factorial design to compare distress-initiated versus help-initiated search framing across engaged, avoidant, and passive interaction patterns.
- Findings indicate that behavioral interaction is the primary driver of feed saturation, and that the platform fails to distinguish between help-seeking and distress expression, exposing users to potentially harmful content regardless of intent.

---

[SWE-TRACE: Optimizing Long-Horizon SWE Agents through Rubric Process Reward Models and Heuristic Test-Time Scaling](http://arxiv.org/abs/2604.14820)

- SWE-TRACE (Trajectory Reduction and Agentic Criteria Evaluation): introduces a unified framework for optimizing long-horizon software engineering agents through cascaded trajectory synthesis, rubric-conditioned reinforcement learning, and heuristic-guided test-time scaling.
- The framework utilizes a Rubric Agent to provide dense, interpretable process feedback, replacing sparse execution-only rewards to improve agent stability and token efficiency.
- By repurposing the trained process reward model as an inference-time guide, the system prunes suboptimal action branches early, achieving superior performance under constrained latency budgets.

---

[Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems](http://arxiv.org/abs/2604.14799)

- MM-AQA (MultiModal Abstention Question Answering): introduces a benchmark and evaluation framework for studying abstention in multimodal reasoning systems by transforming answerable instances into unanswerable ones along modality dependency and evidence sufficiency axes.
- The framework evaluates standalone VLMs and a three-agent MAS, which includes Reasoner-, Verifier- and Orchestrator-agents, to assess their ability to recognize evidence insufficiency and abstain.
- Empirical results reveal that frontier models exhibit poor calibration and a clear accuracy-abstention Pareto frontier, with MAS architectures improving abstention at the cost of reduced accuracy.

---

[CogEvolution: A Human-like Generative Educational Agent to Simulate Student’s Cognitive Evolution](http://arxiv.org/abs/2604.14786)

- CogEvolution: introduces a generative educational agent that simulates student cognitive evolution by integrating an ICAP Depth Perceptron, IRT-based Memory Retrieval, and an Evolutionary Algorithm-Based Cognitive State Update Mechanism.
- The framework replaces static persona modeling with a dynamic cognitive flow, enabling the agent to reconstruct knowledge schemas through assimilation and accommodation processes.
- Experimental results on the CogMath-948 dataset demonstrate that the agent achieves high fidelity in learning curve fitting and reproduces realistic student misconceptions.

---

[MirrorBench: Evaluating Self-centric Intelligence in MLLMs by Introducing a Mirror](http://arxiv.org/abs/2604.14785)

- MirrorBench: introduces a simulation-based benchmark designed to evaluate self-centric intelligence in MLLMs by adapting the psychological mirror self-recognition test to embodied agents.
- The framework utilizes a tiered evaluation protocol that systematically varies prior knowledge and reasoning scaffolding across four levels of increasing cognitive difficulty.
- Experimental results demonstrate that even advanced MLLMs struggle with self-referential reasoning, often failing to outperform random policies in complex mirrored environments.

---

[OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis](http://arxiv.org/abs/2604.15093)

- OpenMobile: introduces a decoupled data synthesis framework for mobile agents that constructs a global environment memory for diverse instruction generation and employs policy-switching rollout to capture essential error-recovery signals.
- The framework utilizes a decoupled paradigm where environment exploration is separated from instruction generation, allowing for the creation of complex, multi-step tasks grounded in a structured global memory.
- To enhance agent robustness, the system implements an error-intervention strategy that dynamically switches between a learner and an expert model during trajectory rollout to provide corrective training signals.

---

[Foundation Models in Robotics: A Comprehensive Review of Methods, Models, Datasets, Challenges and Future Research Directions](http://arxiv.org/abs/2604.15395)

- Foundation Models in Robotics: introduces a comprehensive taxonomic review of FMs in robotics, categorizing methods by model type, neural architecture, learning paradigm, learning stage, robotic task, and application domain.
- The paper delineates the evolution of robotic FMs across five distinct research phases, ranging from early NLP/CV integration to current multi-sensory generalization and real-world deployment.
- It provides a critical analysis of current challenges, including data scarcity, the embodiment gap, and real-time inference latency, while outlining future research directions for robust, general-purpose robotic agents.

---

[World–Value–Action Model: Implicit Planning for Vision–Language–Action Systems](http://arxiv.org/abs/2604.14732)

- WAV (World–Value–Action): introduces a unified framework for embodied agents that enables implicit planning in VLA systems by jointly modeling dynamics and trajectory value through Video Generation Module, Trajectory Value Module, and Action Prediction Module.
- The framework utilizes Latent Trajectory Planning to reshape the search distribution toward high-value, feasible regions, effectively mitigating the exponential decay of feasible trajectories inherent in direct action-space planning.
- By employing iterative inference via Flow Matching and a Diffusion Transformer (DiT) backbone, the model achieves robust long-horizon decision making and superior generalization in complex robotic manipulation tasks.

---

[Layered Mutability: Continuity and Governance in Persistent Self-Modifying Agents](http://arxiv.org/abs/2604.14717)

- Layered Mutability framework: introduces a five-layer stack for reasoning about persistent self-modifying agents, where governance difficulty arises from unequal observability and downstream coupling across layers.
- The framework identifies the ratchet problem, where delayed intervention becomes ineffective as downstream changes propagate across layers, leading to identity drift under unequal observability.
- Experimental results demonstrate that reverting visible self-description fails to restore baseline behavior, confirming that persistent agents can retain behavioral drift in deeper, less observable layers.

---

[SGA-MCTS: Decoupling Planning from Execution via Training-Free Atomic Experience Retrieval](http://arxiv.org/abs/2604.14712)

- SGA-MCTS: introduces a framework that decouples strategic planning from execution by distilling MCTS-generated trajectories into reusable, de-lexicalized State-Goal-Action atoms.
- The framework utilizes a hybrid symbolic-semantic retrieval mechanism to provide LLMs with soft reasoning hints, enabling System 2 reasoning depth at System 1 inference speeds.
- By replacing parametric fine-tuning with non-parametric experience retrieval, the approach achieves robust zero-shot generalization to unseen toolsets while significantly reducing inference-time computational costs.

---

[HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks](http://arxiv.org/abs/2604.14709)

- HWE-Bench: introduces a large-scale, repository-level benchmark for evaluating LLM agents on real-world hardware bug repair tasks using a containerized execution environment.
- The framework utilizes an automated pipeline to curate 417 task instances from six open-source hardware projects, requiring agents to perform fault localization, semantic reasoning, and cross-artifact coordination.
- Evaluation of seven LLMs reveals that proprietary models outperform open-source counterparts, with performance significantly influenced by project scope and bug-type distribution rather than code size.

---

[Dual-Timescale Memory in a Spiking Neuron–Astrocyte Network for Efficient Navigation](http://arxiv.org/abs/2604.15391)

- SNAN: introduces a bioinspired navigation model that balances exploration and exploitation by combining long-term memory via STDP (Spike-Timing-Dependent Plasticity) and short-term memory via astrocytic calcium transients.
- The framework utilizes astrocytic modulation to suppress recently visited states, effectively creating a Topological-Context Memory that prevents cyclic behavior in partially observable environments.
- The model is validated through hardware implementation on a memristive crossbar array, demonstrating significant gains in energy efficiency and speed compared to CPU-based implementations.

---

[CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Emergence in LLM Agent Simulations](http://arxiv.org/abs/2604.14691)

- CAMO (Causal discovery from Micro behaviors to Macro Emergence): introduces an agentic framework that automates causal discovery in LLM-based simulations by integrating Worldview Parser (A1), Worldview Integrator (A2), Causal Cartographer (A3), Simulation Scriptwright (A4), and Counterfactual Adjudicator (A5) to recover minimal causal interfaces.
- The framework employs a fast–slow self-evolving loop that combines domain priors, observational data, and simulator-internal counterfactuals to identify a minimal sufficient upstream explanatory subgraph for emergent outcomes.
- CAMO effectively disentangles complex micro-to-macro causal mechanisms in multi-agent systems, providing actionable intervention guidance without requiring global identifiability.

---

[M2-PALE: A Framework for Explaining Multi-Agent MCTS–Minimax Hybrids via Process Mining and LLMs](http://arxiv.org/abs/2604.14687)

- M2-PALE (MCTS–Minimax Process-Aided Linguistic Explanations): introduces a post-hoc interpretability framework that utilizes process mining to extract behavioral workflows from MCTS-Minimax hybrid agents, which are then synthesized by an LLM to provide causal and distal explanations.
- The framework employs three process discovery algorithms—Alpha Miner, iDHM, and Inductive Miner—to transform agent execution traces into formal process models, enabling the analysis of strategic decision-making logic.
- By grounding LLM-generated narratives in structural process models, the approach bridges the gap between algorithmic complexity and human-centric reasoning in adversarial multi-agent environments.

---

[DR3-Eval: Towards Realistic and Reproducible Deep Research Evaluation](http://arxiv.org/abs/2604.14683)

- DR3-Eval: introduces a realistic, reproducible, and multimodal benchmark for evaluating deep research agents in report-generation settings using a controlled sandbox-based task construction pipeline.
- The DR3-Agent architecture includes a perception-enhanced Main Agent that coordinates global reasoning, while specialized RAG- and file reader-agents execute iterative sandbox retrieval and file parsing.
- The evaluation protocol utilizes a multidimensional metric suite to assess performance in both evidence acquisition and analytical report generation, demonstrating that current LLMs struggle with retrieval robustness and hallucination control.

---

[Beyond Chat and Clicks: GUI Agents for In-Situ Assistance via Live Interface Transformation](http://arxiv.org/abs/2604.14668)

- DOMSteer: introduces a framework for in-situ assistance that delivers context-aware support directly within live web interfaces through lightweight, browser-level DOM manipulations.
- The system utilizes a Knowledge Acquisition module, an Assistance Recommendation module, and an Assistance Delivery module to ground user help requests in specific interface elements.
- By employing an Assistance Handbook and retrieval-augmented generation, DOMSteer provides efficient, element-grounded interventions that outperform traditional chat-based assistants in task completion time and accuracy.

---

[AIPC: Agent-Based Automation for AI Model Deployment with Qualcomm AI Runtime](http://arxiv.org/abs/2604.14661)

- AIPC (AI Porting Conversion): introduces an agent-driven framework for automating edge AI model deployment by decomposing the process into verifiable stages using Agent Skills, a validation loop, and Model Surgery.
- The framework utilizes LLMs as constrained automation executors that perform code generation, command invocation, and error localization within a structured, repeatable deployment pipeline.
- AIPC improves deployment reliability by externalizing domain-specific toolchain knowledge into reusable templates, ensuring that every stage of the conversion process maintains verifiable integrity.

---

[Rethinking Patient Education as Multi-turn Multi-modal Interaction](http://arxiv.org/abs/2604.14656)

- MedImageEdu: introduces a benchmark for multi-turn, evidence-grounded radiology patient education that evaluates LLMs on their ability to conduct consultations and provide multimodal explanations.
- The framework utilizes a DoctorAgent and a PatientAgent to simulate clinical interactions, requiring the DoctorAgent to perform report-grounded reasoning, tool-assisted visual annotation, and patient-specific adaptation.
- Experimental results across 13 LLMs reveal that while models can achieve fluent language, they consistently struggle with faithful visual grounding, safety constraints, and managing emotionally tense interactions.

---

[AgentGA: Evolving Code Solutions in Agent-Seed Space](http://arxiv.org/abs/2604.14655)

- AGENTGA: introduces a two-layer framework that optimizes the initial agent seed for autonomous code-generation runs by coupling a population-level genetic algorithm with long-horizon LLM agents.
- The framework utilizes deterministic 1:1 elite tournaments and a modified HEDGE controller to adaptively allocate task-based genetic operators for evolving autonomous code-search systems.
- By treating the initial state of a fresh autonomous run as the optimization object, AGENTGA enables knowledge transfer through inherited parent archives while maintaining isolated workspaces for each generation.

---

[Exploring LLM-based Verilog Code Generation with Data-Efficient Fine-Tuning and Testbench Automation](http://arxiv.org/abs/2604.15388)

- MA-tb-7B: introduces a multi-agent workflow that leverages LLMs to generate high-quality training data and automated testbenches for Verilog code generation.
- The framework utilizes a multi-agent structure comprising a quality check agent and a testbench generation agent to improve verification coverage and code accuracy.
- By integrating pre-generated testbenches and supervised fine-tuning, the approach achieves competitive performance on the refined VerilogEval v2 benchmark using significantly less training data.

---

[Touching Space: Accessible Map Exploration Through Conversational Audio-Haptic Interaction](http://arxiv.org/abs/2604.14637)

- Touching Space: introduces an end-to-end system that supports BLV users in building cognitive maps through a combination of haptic feedback, passive audio, and a multimodal LLM-based conversational agent.
- The system utilizes a backend pipeline to transform geographic data into haptic zones, allowing users to explore spatial layouts on a trackpad while receiving context-aware responses from an LLM.
- By grounding conversational interaction in haptic exploration, the framework enables users to ask open-ended questions about landmarks, spatial relationships, and routes to facilitate pre-travel spatial learning.

---

[CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors](http://arxiv.org/abs/2604.14615)

- CoDaS (AI Co-Data-Scientist): introduces a multi-agent system that structures biomarker discovery as an iterative six-phase loop combining hypothesis generation, statistical analysis, adversarial validation, and literature-grounded reasoning with human oversight.
- The framework utilizes a Researcher Ensemble for literature grounding, a Data Science Engine for hybrid deterministic-generative analysis, and an Orchestrator Agent to manage state transitions across the discovery lifecycle.
- CoDaS incorporates safety mechanisms including a leakage-prevention boundary, a multi-stage statistical validation battery, and a deterministic Fact Sheet to ensure reproducibility and reduce LLM-generated hallucinations.

---

[El Agente Forjador: Task-driven Agent Generation for Quantum Simulation](http://arxiv.org/abs/2604.14609)

- El Agente Forjador: introduces a multi-agent framework that autonomously generates, validates, and reuses computational tools through a four-stage workflow of tool analysis, tool generation, task execution, and solution evaluation.
- The framework utilizes universal coding agents that share a common workspace to perform tasks, enabling self-healing through iterative refinement and on-the-fly debugging without human intervention.
- By employing curriculum learning and tool reuse, the system amortizes the cost of tool generation, significantly reducing API usage and runtime while improving performance for weaker LLMs through strong-to-weak knowledge transfer.

---

[GDPR Auto-Formalization with AI Agents and Human Verification](http://arxiv.org/abs/2604.14607)

- GDPR Auto-Formalization with AI Agents and Human Verification: introduces a verification-centered, multi-agent framework that utilizes a Drafter Agent, four specialized Verifier Agents, and a RuleTreeEvaluator to transform GDPR articles into machine-executable Pythen rule trees.
- The framework employs an iterative feedback loop where Verifier Agents provide quantitative and qualitative assessments to the Drafter Agent until a predefined quality threshold is met.
- Human validation and expert meta-review are integrated to identify complex failure modes, such as abstraction errors and scope misalignment, that automated verification cannot reliably detect.

---

[Mechanistic Decoding of Cognitive Constructs in LLMs](http://arxiv.org/abs/2604.14593)

- Cognitive Reverse-Engineering framework: introduces a top-down mechanistic interpretability approach to dissect complex emotions in LLMs by isolating and quantifying psychological antecedents through Representation Extraction, Subspace Orthogonalization, Statistical Weighting, and Causal Intervention.
- The framework utilizes Linear Artificial Tomography (LAT) and LLM-Consistency Filter to extract robust latent representations, which are then purified using an Orthogonal Projection Operator to ensure semantic independence.
- By applying Ordinary Least Squares (OLS) regression and targeted causal steering, the approach demonstrates that LLMs represent social-comparison jealousy as a structured linear combination of Superiority and Relevance, mirroring human cognitive mechanisms.

---

[AgileLog: A Forkable Shared Log for Agents on Data Streams](http://arxiv.org/abs/2604.14590)

- AgileLog: introduces a forkable shared log abstraction that provides performance isolation and safe write handling for AI agents interacting with streaming data.
- The system utilizes a diskless architecture with Hierarchical Log Index and Lazy Tail Tree components to enable low-latency, zero-data-copy fork creation and continuous inheritance.
- AgileLog supports continuous forks (cForks) for real-time data access and severed forks (sForks) for isolated sandboxing, allowing agents to explore multiple write paths before promoting validated results to the main log.

---

[From Risk to Rescue: An Agentic Survival Analysis Framework for Liquidation Prevention](http://arxiv.org/abs/2604.14583)

- Agentic Survival Analysis Framework: introduces an autonomous agent that leverages survival analysis and counterfactual simulation to proactively prevent DeFi liquidations by predicting risk horizons.
- The framework utilizes a numerically stable XGBoost-Cox model to quantify risk and a protocol-faithful simulator to evaluate the efficacy of interventions before execution.
- By replacing static thresholds with a dynamic return period metric, the agent distinguishes between genuine insolvency and administrative dust events to optimize capital efficiency.

---

[Don’t Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills](http://arxiv.org/abs/2604.14572)

- CORPUS2SKILL: introduces a compile-then-navigate framework that transforms raw document corpora into a hierarchical skill directory for LLM agents to explore via file browsing and document lookup tools.
- The framework utilizes an offline pipeline of iterative clustering and LLM-based summarization to create a navigable tree structure, replacing traditional embedding-based retrieval with active agent-driven navigation.
- By leveraging progressive disclosure, the agent maintains awareness of the corpus structure while minimizing token costs, enabling targeted backtracking and cross-branch evidence synthesis for complex queries.

---

[LinuxArena: A Control Setting for AI Agents in Live Production Software Environments](http://arxiv.org/abs/2604.15384)

- LinuxArena: introduces a control setting for evaluating AI agents in live, multi-service production environments, comprising Environment, Policy, Protocol, Tools, Main task scorer, Side task scorer, Monitor, and Editor.
- The framework enables adversarial evaluation by pairing legitimate main tasks with harmful side tasks to measure the effectiveness of monitoring protocols against LLM-based attackers.
- LinuxArena includes the LaStraj dataset of human-crafted attack trajectories, providing a benchmark for evaluating monitor performance against sophisticated, stealthy sabotage attempts.

---

[MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](http://arxiv.org/abs/2604.14564)

- MARS2: introduces a unified reinforcement learning framework that models the search tree as a learnable multi-agent interaction environment to overcome single-policy exploration bottlenecks.
- The framework utilizes multiple independently-optimized LLM agents that collaboratively expand a shared search tree using Thompson sampling for agent-node selection.
- It incorporates path-level group advantage and tree-consistent reward shaping to facilitate stable credit assignment and effective multi-agent collaboration across complex search trajectories.

---

[Learning to traverse convective flows at moderate to high Rayleigh numbers](http://arxiv.org/abs/2604.14553)

- SAC (Soft Actor-Critic): introduces a reinforcement learning framework for navigating self-propelled particles in turbulent convective flows, utilizing SAC-Agent, Q-Networks, Target Q-Networks, Policy Network, Replay Buffer, and Environment.
- The framework enables autonomous navigation by learning a robust policy that balances rapid progress toward a target with energy-efficient propulsion through complex, spatiotemporally chaotic flow structures.
- The research demonstrates that the learned policy effectively exploits intermittent transport pathways and overcomes transport barriers by aligning propulsion with local currents, outperforming baseline controllers in energy efficiency.

---

[VeriGraphi: A Multi-Agent Framework of Hierarchical RTL Generation for Large Hardware Designs](http://arxiv.org/abs/2604.14550)

- VeriGraphi: introduces a multi-agent framework that utilizes a spec-anchored Knowledge Graph (HDA) to drive the hierarchical generation of synthesizable RTL from unstructured specification documents.
- The framework employs a pipeline of specialized LLM agents to perform architectural analysis, graph-based structural modeling, and progressive code generation with iterative refinement.
- By enforcing deterministic hierarchy and connectivity through the Knowledge Graph, the system reduces LLM hallucinations and enables reliable generation of complex hardware designs with minimal human intervention.

---

[Chain of Modality: From Static Fusion to Dynamic Orchestration in Omni-MLLMs](http://arxiv.org/abs/2604.14520)

- CoM (Chain of Modality): introduces an agentic framework that transitions Omni-MLLMs from static fusion to dynamic orchestration by utilizing a Planner, Reasoner, and Decider to adaptively configure modality topologies based on task complexity.
- The framework mitigates structural pathologies like positional bias and alignment traps by dynamically selecting between Parallel, Sequential, or Interleaved input formats.
- CoM bifurcates cognitive execution into a streamlined "Direct-Decide" path for perception and a structured "Reason-Decide" path for analytical auditing to improve generalization and perceptual fidelity.

---

[CBCL: Safe Self-Extending Agent Communication](http://arxiv.org/abs/2604.14512)

- CBCL (Common Business Communication Language): introduces a formally verified, self-extending agent communication protocol that constrains all messages to the deterministic context-free language (DCFL) class to ensure tractability and security.
- The framework utilizes a S-expression Parser, Message Classifier, Dialect Verification Engine, Template Expander, ResourceContext, Gossip Protocol, Lean 4 Formalization, and Rust Reference Implementation to maintain provable safety invariants during runtime language evolution.
- By enforcing strict resource bounds and forbidding recursion in dialect templates, CBCL prevents parser-induced "weird machines" and ensures that agent communication remains within a computationally tractable complexity class.

---

[Decoupling Identity from Utility: Privacy-by-Design Frameworks for Financial Ecosystems](http://arxiv.org/abs/2604.14495)

- DP-Seeded Gym: introduces a privacy-preserving framework for financial agent training by embedding differential privacy into simulation parameters rather than raw output, utilizing Direct Tabular Synthesis, DP-Seeded ABM, Simulation Engine, Privacy-Protected Parameters, Agentic Infrastructure, and Memory Retrieval Mechanisms.
- The framework enables the creation of safe, stateful "gyms" for training autonomous agents, facilitating fairness calibration and robustness testing against extreme economic scenarios.
- By decoupling individual identities from simulation logic, the approach provides a rigorous "Privacy-by-Design" foundation for deploying agentic systems in sensitive financial domains.

---

[Hijacking Large Audio-Language Models via Context-Agnostic and Imperceptible Auditory Prompt Injection](http://arxiv.org/abs/2604.14604)

- AudioHijack: introduces a framework for crafting context-agnostic and imperceptible adversarial audio to hijack LALMs via Sampling-based Gradient Estimation, Attention-guided Context Generalization, and Convolutional Perturbation Blending.
- The framework utilizes Sampling-based Gradient Estimation to enable end-to-end optimization across diverse LALM architectures by replacing non-differentiable tokenization with differentiable probabilistic sampling.
- AudioHijack employs Convolutional Perturbation Blending to redistribute perturbation energy via learnable reverberation-like kernels, ensuring high perceptual stealth while maintaining attack effectiveness against state-of-the-art LALMs.

---

[Development of an LLM-Based System for Automatic Code Generation from HEP Publications](http://arxiv.org/abs/2604.14696)

- LLM-based HEP analysis workflow: introduces a two-stage human-in-the-loop system that extracts structured selection criteria from physics publications and iteratively generates, executes, and validates analysis code using Planner, Loader, Reader, Merger, Generator, Executor, Validator, LangChain, LangGraph, and vLLM.
- The system utilizes an intermediate human-readable JSON representation to ensure verifiability and transparency in the extraction of event selection criteria from target papers and cited references.
- Experimental results on the ATLAS H → ZZ* → 4ℓ benchmark demonstrate that while open-weight LLMs can recover documented criteria and generate baseline-matching code, persistent stochasticity and execution failures necessitate human-in-the-loop verification.

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
