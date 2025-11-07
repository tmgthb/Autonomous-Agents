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

#### 29th October 2025

[Identity Management for Agentic AI: The new frontier of authorization, authentication, and security for an AI agent world](http://arxiv.org/abs/2510.25819)

- Identity Management for Agentic AI: introduces a comprehensive whitepaper addressing the unique identity, authentication, authorization, and security challenges posed by autonomous AI agents, leveraging existing standards and proposing new architectural patterns.
- The paper outlines immediate solutions using protocols like MCP, A2A, OAuth 2.1, and SCIM for synchronous, single-trust domain agent operations, while also identifying future challenges for highly autonomous, cross-domain, and recursive delegation scenarios.
- It emphasizes the need for robust agent identity, explicit delegated authority, scalable governance via guardrails and policy-as-code, and interoperable trust mechanisms to build a secure and responsible AI agent ecosystem.

---

[ALDEN: REINFORCEMENT LEARNING FOR ACTIVE NAVIGATION AND EVIDENCE GATHERING IN LONG DOCUMENTS](http://arxiv.org/abs/2510.25668)

- ALDEN (Active Long-Document Navigation): introduces a multi-turn reinforcement learning framework that fine-tunes VLMs as interactive agents for active navigation and evidence gathering in long, visually rich documents.
- The framework expands the action space with a novel `fetch` action for direct page access, complements a cross-level reward function for fine-grained supervision, and incorporates visual semantic anchoring to stabilize training.
- ALDEN enables VLMs to autonomously navigate and reason across complex documents, moving beyond passive document reading for more accurate and efficient understanding.

---

[CAIR : Counterfactual-based Agent Influence Ranker for Agentic AI Workflows](http://arxiv.org/abs/2510.25612)

- CAIR (Counterfactual-based Agent Influence Ranker): introduces a method for assessing the influence of individual agents within Agentic AI Workflows (AAWs), utilizing an offline phase for counterfactual analysis and an online phase for rapid ranking prediction.
- The framework's offline phase involves recording AAW activation flows, systematically injecting counterfactual agent outputs, calculating changes in final output and workflow, and computing influence scores to generate agent rankings.
- The online phase efficiently predicts agent influence by vectorizing new input queries, retrieving the most similar pre-computed representative query from the offline analysis, and applying its associated agent rankings.

---

[GAP: Graph-based Agent Planning with Parallel Tool Use and Reinforcement Learning](http://arxiv.org/abs/2510.25320)

- GAP (Graph-based Agent Planning): introduces a novel framework that enables LLM-based agents to perform dependency-aware reasoning and adaptive tool execution by decomposing complex tasks into dependency-aware sub-task graphs, autonomously determining which tools can be executed in parallel and which must follow sequential dependencies.
- The framework employs a two-stage training strategy, including supervised fine-tuning on a curated dataset of graph-based planning traces and reinforcement learning with a correctness-based reward function, to optimize execution efficiency and task accuracy.
- GAP significantly outperforms traditional sequential reasoning baselines like ReAct on multi-hop reasoning tasks, demonstrating substantial improvements in tool invocation efficiency through intelligent parallelization and reduced LLM interaction turns.

---

[FELA: A Multi-Agent Evolutionary System for Feature Engineering of Industrial Event Log Data](http://arxiv.org/abs/2510.25223)

- FELA (Feature Engineering LLM Agents): introduces a multi-agent evolutionary system that autonomously extracts meaningful and high-performing features from complex industrial event log data, integrating LLM reasoning and coding capabilities with an insight-guided self-evolution paradigm.
- The system employs specialized Idea, Code, and Critic Agents for collaborative feature generation, validation, and implementation, supported by an Evaluation Agent that summarizes feedback and updates a hierarchical knowledge base and dual-memory system.
- FELA introduces an agentic evolution algorithm, combining reinforcement learning and genetic algorithm principles with Upper Confidence Bound (UCB) exploration to balance exploration and exploitation across the feature idea space.

---

[Collaborative Scheduling of Time-dependent UAVs,Vehicles and Workers for Crowdsensing in Disaster Response](http://arxiv.org/abs/2510.25212)

- HoCs-MPQ (Heterogeneous Multi-Agent Online Collaborative Scheduling Algorithm): introduces a framework for collaborative scheduling of time-dependent UAVs, vehicles, and workers in disaster response, modeling relationships as a weighted undirected graph and solving it as a maximum weight independent set problem.
- The framework utilizes a two-module approach: weighted undirected graph construction to represent collaborative and conflictual relationships, and a maximum weight independent set solution accelerated by multi-priority queues for efficient scheduling.
- This approach aims to overcome limitations of existing sensing technologies in complex post-disaster environments by maximizing collaboration and improving task completion rates with low computational overhead.

---

[Agentic Moderation: Multi-Agent Design for Safer Vision-Language Models](http://arxiv.org/abs/2510.25179)

- Agentic Moderation Framework: introduces a model-agnostic framework for safety alignment in LVLMs, utilizing a Coordinator to orchestrate Shield, Responder, Evaluator, and Reflector Agents, along with VLLMs and various tools, to achieve context-aware and interpretable moderation.
- This multi-agent system performs initial safety screening, generates candidate outputs under moderation guidance, validates responses against safety and utility criteria, and provides corrective feedback for iterative refinement.
- The framework provides modular, scalable, and fine-grained safety enforcement against cross-modal adversarial attacks, demonstrating robust performance in reducing Attack Success Rate and improving Refusal Rate.

---

[The Iceberg Index: Measuring Workforce Exposure in the AI Economy](http://arxiv.org/abs/2510.25137)

- Project Iceberg: introduces a framework for measuring workforce exposure to AI, with CAPTURE (human workforce mapping), ANALYZE (AI workforce mapping), SIMULATE (human-AI interaction modeling), Large Population Models (LPMs) (agent-based simulation engine), AgentTorch platform (LPMs implementation), and Frontier supercomputer (high-performance computing), designed to simulate human-AI labor market interactions at a national scale.
- The framework quantifies the wage value of skills AI systems can technically perform within occupations, providing a skills-centered metric called the Iceberg Index to reveal AI-human capability overlap.
- By simulating various scenarios, the framework enables policymakers to identify exposure hotspots, prioritize investments, and test interventions for workforce preparation before AI adoption impacts crystallize.

---

[KnowCoder-A1: Incentivizing Agentic Reasoning Capability with Outcome Supervision for KBQA](http://arxiv.org/abs/2510.25101)

- KNOWCODER-A1: introduces an LLM agent for Knowledge Base Question Answering (KBQA) that leverages a SFT-based Cold-start Stage for foundational capabilities and an RL-based Exploration Stage for autonomous exploration, interacting with a KB Environment via a defined Action Space.
- The framework employs outcome-only supervision and a multi-stage curriculum reinforcement learning approach, including GRPO and a dynamic Reward Curriculum, to overcome reward sparsity and enhance agentic reasoning.
- KNOWCODER-A1 demonstrates strong generalization, robustness, and flexibility by recovering from errors and strategically exploring diverse reasoning trajectories, outperforming prior agentic KBQA methods.

---

[TheraMind: A Strategic and Adaptive Agent for Longitudinal Psychological Counseling](http://arxiv.org/abs/2510.25758)

- TheraMind: introduces a strategic and adaptive agent for longitudinal psychological counseling, featuring a novel dual-loop architecture that decouples tactical dialogue management from strategic therapeutic planning.
- The Intra-Session Loop dynamically manages turn-by-turn interactions by perceiving patient state, retrieving memory, selecting response strategies, identifying treatment stages, and generating clinically-grounded responses.
- The Cross-Session Loop provides long-term adaptability by evaluating therapeutic efficacy after each session and adaptively adjusting the treatment method for subsequent interactions, ensuring coherent and personalized counseling.

---

[AGENTIC ECONOMIC MODELING](http://arxiv.org/abs/2510.25743)

- AEM (Agentic Economic Modeling): introduces a Generation-Correction-Inference pipeline, which leverages LLM-generated data anchored to a small set of real human observations to ensure reliable downstream economic inference.
- The framework first employs LLMs with diverse personas to generate task-conditioned synthetic choices, then learns a bias-correction mapping using limited human data, and finally applies standard econometric estimators to the corrected data.
- AEM demonstrates potential to improve Randomized Control Trial (RCT) efficiency and establishes a foundational method for LLM-based counterfactual generation by systematically correcting LLM biases.

---

[Process-Level Trajectory Evaluation for Environment Configuration in Software Engineering Agents](http://arxiv.org/abs/2510.25694)

- EnConda-Bench (Environment Configuration Diagnosis Benchmark): introduces a framework for process-level trajectory evaluation of environment configuration in software engineering agents, including an automated data construction pipeline and a process-level trajectory evaluation suite.
- The automated data construction pipeline generates high-quality task instances by selecting repositories, synthesizing errors into READMEs using LLMs, and validating these errors through automatic and LLM-assisted filtering.
- The process-level trajectory evaluation suite assesses agent capabilities in environment setup-planning, perception-driven error diagnosis, feedback-driven repair, and action execution, providing fine-grained insights beyond end-to-end success rates.

---

[Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry](http://arxiv.org/abs/2510.25595)

- Fine-tuning-plus-Verifier Framework: introduces a collaborative system for LLM agents operating under information asymmetry, integrating LLM Agents (core intelligent entities), Fine-tuning (adapts LLMs for task), Environment-based Verifier (guides agent decisions), Affordance Verifier (checks physical rules), Communication Verifier (assesses communication meaningfulness), Reasoning Verifier (infers new constraints), Communication Strategies (information-seeking and -providing), Graph Expansion Algorithm (enhances reasoning), and Game Environment (provides feedback signals) to solve a tabletop Einstein Puzzle.
- The framework equips LLM agents with diverse communication abilities and leverages environmental feedback through a multi-component verifier to improve task performance and rule comprehension without additional training.
- This approach highlights the importance of aligned communication protocols and environment-based verification for developing safer, more interpretable, and trustworthy LLM-based collaborative AI systems.

---

[Standardization of Psychiatric Diagnoses — Role of Fine-tuned LLM Consortium and OpenAI-gpt-oss Reasoning LLM Enabled Decision Support System](http://arxiv.org/abs/2510.25588)

- Fine-Tuned LLM Consortium and OpenAI-gpt-oss Reasoning LLM Enabled Decision Support System: introduces an AI-assisted diagnostic framework that integrates a Data Lake Layer (stores conversational datasets), an LLM Agent Layer (orchestrates communication/prompt engineering/aggregation), an LLM Layer (Fine-tuned LLM Consortium) (generates preliminary diagnoses) with Fine-tuned Llama LLM (predicts diagnoses), Fine-tuned Mistral LLM (predicts diagnoses), Fine-tuned Qwen LLM (predicts diagnoses) deployed via Ollama (deploys/manages fine-tuned LLMs), and an OpenAI-gpt-oss Reasoning LLM Layer (synthesizes final diagnosis) implemented with Agents SDK (implements LLM agents), to standardize psychiatric diagnoses.
- The system leverages fine-tuned LLMs on conversational datasets to identify mental disorders with high accuracy, aggregating individual model predictions through a consensus-based decision-making process refined by the OpenAI-gpt-oss reasoning LLM.
- This approach enhances diagnostic precision and consistency by combining multiple specialized LLMs with a dedicated reasoning engine, ensuring transparency, reliability, and adherence to responsible AI principles in mental health assessment.

---

[GROUNDED IN REALITY: LEARNING AND DEPLOYING PROACTIVE LLM FROM OFFLINE LOGS](http://arxiv.org/abs/2510.25441)

- Learn-to-Ask: introduces a simulator-free framework for learning and deploying proactive dialogue agents directly from offline expert data, leveraging observed future trajectories to infer dense, turn-by-turn reward signals.
- The framework decomposes the long-horizon offline RL problem into supervised learning tasks, training a policy to output structured (action, state_assessment) tuples, governing what to ask and when to stop.
- An Automated Grader Calibration pipeline systematically purges noise from the LLM-based reward model with minimal human supervision, ensuring reward fidelity and enabling real-world deployment with superior performance.

---

[Small Talk, Big Impact? LLM-based Conversational Agents to Mitigate Passive Fatigue in Conditional Automated Driving](http://arxiv.org/abs/2510.25421)

- LLM-based Conversational Agent (Zoe): introduces an LLM-based conversational agent (Zoe) (core intelligence) with engagement strategies (driver interaction), interaction management protocols (conversation flow), safety protocols (distraction prevention), gamification features (cognitive stimulation), context-relevant prompts (situational awareness), and micro-interactions (entertainment), designed to mitigate passive fatigue in conditional automated driving.
- The paper presents findings from a real-world test-track study with 40 participants, demonstrating the CA's potential to support driver alertness and engagement.
- The study identifies three user preference archetypes (Safety-First, Entertainment-Seeking, Social-Connection Oriented) for CA interaction, highlighting the need for adaptive design to balance safety and user experience.

---

[CGM-Led Multimodal Tracking with Chatbot Support: An Autoethnography in Sub-Health](http://arxiv.org/abs/2510.25381)

- CGM-Led Multimodal Tracking with Chatbot Support: explores how integrating Continuous Glucose Monitoring (CGM) with multimodal physiological and psychological indicators, processed by an LLM-driven chatbot (including Qwen-VL and Tencent Hunyuan), can shape everyday health management in sub-health contexts.
- The study, conducted as a six-week autoethnography, demonstrates how this system provides personalized reflections and explanations of glucose fluctuations, supporting preventive health and self-reflection in at-risk individuals.
- This approach extends CGM research beyond clinical diabetes, showing how LLM-driven agents can interpret diverse health data to facilitate lifestyle adjustments and foster resilience.

---

[CRMWeaver: Building Powerful Business Agent via Agentic RL and Shared Memories](http://arxiv.org/abs/2510.25333)

- CRMWeaver: introduces a novel framework for building robust business agents, with Synthesis Data Generation (creates training data), Training Recipe (two-stage training paradigm), and Enhanced Memory in Inference (improves inference with memory) components, designed to enhance business agents in complex environments by combining synthetic data generation, a two-stage training paradigm, and a shared memories mechanism.
- The framework addresses data scarcity and domain complexity through a two-fold optimization strategy, first synthesizing diverse training data and employing a two-stage learning paradigm (SFT for initialization followed by RL for generalization), and second, introducing a long-term memory module that indexes task-specific guidelines from successful trajectories.
- During inference, the memory module retrieves and injects relevant guidelines into the current context, enabling the agent to leverage prior knowledge and improve task performance and generalization, especially in previously unseen scenarios.

---

[PROMEDIATE: A SOCIO-COGNITIVE FRAMEWORK FOR EVALUATING PROACTIVE AGENTS IN MULTI-PARTY NEGOTIATION](http://arxiv.org/abs/2510.25224)

- PROMEDIATE (A Socio-Cognitive Framework for Evaluating Proactive Agents in Multi-Party Negotiation): introduces a framework for evaluating proactive AI mediator agents in complex, multi-topic, multi-party negotiations using a simulation testbed and a socio-cognitive evaluation framework.
- The framework's testbed simulates realistic negotiation scenarios with configurable conflict modes and LLM-based human and mediator agents, allowing for flexible intervention strategies.
- It employs a suite of socio-cognitive metrics to measure consensus dynamics, intervention latency, mediator effectiveness, and intelligence across perceptual, emotional, cognitive, and communicative dimensions.

---

[AgentCyTE: Leveraging Agentic AI to Generate Cybersecurity Training & Experimentation Scenarios](http://arxiv.org/abs/2510.25189)

- AgentCyTE: introduces a modular framework for automated generation and refinement of cybersecurity training and experimentation scenarios, integrating LLM-based reasoning with schema-constrained network emulation via Agentic AI (orchestrates generation), Base Scenario (optional CORE configuration), Config Builder (user scenario specification), Syntax Validator (ensures XML compliance), Plugins (extend capabilities), Config Interpreter (resolves schema elements), CTF Spec (scenario XML description), GRPC Client (interfaces CORE API), GRPC Server (CORE backend interface), Behavior Validator (probes connectivity), and CORE (network emulation backbone).
- The framework employs an agentic feedback loop, where an LLM controller iteratively generates, validates, and refines configurations based on structured error feedback, coupling generative reasoning with programmatic verification to autonomously improve realism and consistency.
- AgentCyTE translates natural-language intent into executable, standards-compliant network environments, enabling scalable, data-driven experimentation and reliable scenario generation for threat modeling and adaptive cybersecurity training.

---

[MODEL-DOCUMENT PROTOCOL FOR AI SEARCH](http://arxiv.org/abs/2510.25160)

- MDP (Model-Document Protocol): introduces a general framework that formalizes bridging raw text to LLMs through consumable knowledge representations, transforming unstructured documents into LLM-ready inputs via offline knowledge preparation and contextual intelligence pathways.
- MDP-Agent, an instantiation of MDP, realizes this protocol through an agentic process that constructs document-level gist memories, performs diffusion-based exploration, and applies map-reduce style synthesis to integrate large-scale evidence into compact context.
- The framework ensures that what reaches the LLM is structured knowledge, not raw fragments, enabling efficient and effective reasoning for complex information-seeking tasks.

---

[DEBATE: A Large-Scale Benchmark for Role-Playing LLM Agents in Multi-Agent, Long-Form Debates](http://arxiv.org/abs/2510.25110)

- DEBATE (DatasEt for Benchmarking Multi-Agent Opinion Trajectories and Evolution): introduces a large-scale empirical benchmark for evaluating multi-agent role-playing LLM systems, featuring human debate data, role-playing LLM agents conditioned by a memory module, three simulation modes, and a multi-level evaluation framework.
- The DEBATE benchmark contains 29,417 messages from multi-round debates among 2,792 U.S. participants on 107 controversial topics, capturing both public messages and privately reported opinions to assess human-likeness.
- The research identifies critical discrepancies between simulated and authentic group dynamics, showing that while supervised fine-tuning improves surface-level metrics, it struggles with deeper semantic and stance alignment, highlighting limitations in realistically simulating human social dynamics.

---

[SEEINGEYE: AGENTIC INFORMATION FLOW UNLOCKS MULTIMODAL REASONING IN TEXT-ONLY LLMS](http://arxiv.org/abs/2510.25092)

- SeeingEye: introduces a modular framework that unlocks multimodal reasoning in text-only LLMs through a decoupled, two-agent system, including a Translator Agent, Reasoning Agent, Structured Intermediate Representation, and Agentic Information Flow.
- The framework employs an iterative feedback loop where the Translator Agent refines visual information into a Structured Intermediate Representation (SIR) based on the Reasoning Agent's feedback.
- This approach enables strong text-only LLMs to leverage their reasoning capabilities on visual data by transforming complex visual scenes into precise, query-relevant textual evidence.

---

[Estimating cognitive biases with attention-aware inverse planning](http://arxiv.org/abs/2510.25951)

- AAIP (Attention-Aware Inverse Planning): introduces a framework to estimate human attentional biases from observed behavior by extending the value-guided construal framework, incorporating heuristic biases, and inferring bias parameters using deep reinforcement learning and computational cognitive modeling.
- The approach models how resource-limited decision-makers form simplified task construals, demonstrating its ability to recover underlying biases in both tabular DrivingWorld and complex real-world driving scenarios from the Waymo Open Dataset.
- Unlike standard inverse reinforcement learning, AAIP explicitly accounts for attention-limited decision-making, providing a more accurate and scalable method for modeling human behavior in autonomous systems.

---

[FinOps Agent - A Use-Case for IT Infrastructure and Cost Optimization](http://arxiv.org/abs/2510.25914)

- FinOps Agent: introduces an autonomous, goal-driven AI agent for FinOps automation, leveraging a multi-agent system, a unified GraphQL schema, and an NL2GraphQL layer to optimize IT infrastructure costs.
- The system integrates data from heterogeneous sources like Turbonomic and Apptio, enabling dynamic query composition and real-time analysis for actionable recommendations.
- Utilizing a ReAct-style reasoning loop, the agent interprets natural language queries, plans execution, retrieves data, and synthesizes insights, ensuring explainability and auditability.

---

[AAGATE: A NIST AI RMF-Aligned Governance Platform for Agentic AI](http://arxiv.org/abs/2510.25863)

- AAGATE (Agentic AI Governance Assurance & Trust Engine): introduces a Kubernetes-native control plane operationalizing the NIST AI Risk Management Framework by integrating specialized security frameworks for Govern, Map, Measure, and Manage functions, enabling continuous, verifiable governance for agentic AI.
- The platform incorporates a zero-trust service mesh, an explainable policy engine, behavioral analytics, and decentralized accountability hooks to provide a robust solution for safe, accountable, and scalable AI deployment.
- AAGATE extends RMF coverage to ethical, adversarial, and systemic risks through components like DIRF for digital identity rights, LPCI defenses for logic-layer injection, and QSAF monitors for cognitive degradation.

---

#### 28th October 2025

[Delay Tolerant Control for Autonomous Driving Using CDOB](http://arxiv.org/abs/2510.24898)

- CDOB (Communication Disturbance Observer): introduces a delay-tolerant control framework for autonomous driving, which integrates a modified CDOB with a parameter-space PID controller to compensate for time delays and reject external disturbances, ensuring accurate path tracking.
- The framework models time delays as equivalent disturbances, allowing the CDOB to estimate and actively reject them, effectively restoring the system's behavior to its delay-free equivalent for robust control.
- Simulation and hardware-in-the-loop experiments demonstrate that the proposed method maintains high tracking accuracy and delay robustness across various scenarios, outperforming conventional PID control under delayed conditions.

---


[Tongyi DeepResearch Technical Report](http://arxiv.org/abs/2510.24701)

- Tongyi DeepResearch: introduces an agentic LLM for long-horizon, deep information-seeking research tasks, combining Qwen Series Base Models, Agentic CPT, Agentic SFT, Agentic RL, a Data Synthesis Pipeline, Stage-specific Customized Environments, Async Rollout Service, Rollout Worker, Reward Service, various Tools, a Context Management Paradigm, Model Merging, and a Heavy Mode.
- The framework employs an end-to-end training paradigm with agentic mid-training and post-training, leveraging a fully automated, highly scalable data synthesis pipeline and customized environments for stable interactions.
- It achieves state-of-the-art performance on agentic deep research benchmarks by integrating advanced reasoning, information-seeking, and tool-use capabilities, including a novel Heavy Mode for test-time scaling.

---

[WebLeaper: Empowering Efficiency and Efficacy in WebAgent via Enabling Info-Rich Seeking](http://arxiv.org/abs/2510.24697)

- WebLeaper: introduces a framework for constructing entity-intensive information-seeking (IS) tasks and generating efficient solution trajectories, incorporating Entity-Intensive Task Synthesis, Information-Guided Trajectory Construction, and Reinforcement Learning with Hybrid Reward Systems.
- The framework models IS as a tree-structured reasoning problem, synthesizing tasks through Basic, Union, and Reverse-Union variants to increase complexity and entity coverage.
- It further refines agent performance by filtering task-solving trajectories based on Information-Seeking Rate (ISR) and Information-Seeking Efficiency (ISE) metrics, which are integrated into a hybrid reward system for reinforcement learning.

---

[FunReason-MT Technical Report: Overcoming the Complexity Barrier in Multi-Turn Function Calling](http://arxiv.org/abs/2510.24645)

- FunReason-MT (Function Call Reasoning Multi-Turn): introduces a novel data synthesis framework for real-world multi-turn tool use, with Environment-API Graph Interactions (constructs execution traces), Advanced Tool-Query Synthesis (synthesizes hard queries), Guided Iterative Chain (refines reasoning iteratively), Multi-Environment Simulation Space (simulates tool interactions), Tool Set (collection of tools), API Relation Graph (models tool dependencies), LLM Agents (perform reasoning tasks), Tooling Agent (abstracts execution traces), Querying Agent (generates hard queries), Reasoning Agent (solves queries, generates CoT), Critiquing Agent (analyzes errors, provides feedback), Target Tool (specific tool for mastery), Hard Query (challenging abstract query), Advanced Tool (composite tool abstraction), Chain-of-Thought (explicit reasoning steps), Function Call (final tool action), Execution Trace (recorded tool sequence), designed to overcome the complexity barrier in multi-turn function calling data generation.
- The framework addresses limitations of existing data generation methods by employing a top-down construction methodology that explicitly directs the model to master complex tool use within targeted scenarios, ensuring high-quality and diverse multi-turn trajectories.
- FunReason-MT's iterative feedback loop and advanced query synthesis enable robust Chain-of-Thought generation and self-correction, leading to state-of-the-art performance on function-calling benchmarks and improved agentic capabilities.

---

[Stochastic Prize-Collecting Games: Strategic Planning in Multi-Robot Systems](http://arxiv.org/abs/2510.24515)

- SPCG (Stochastic Prize-Collecting Games): introduces a game-theoretic variant of the Team Orienteering Problem for self-interested robots, employing ORS (identifies localized game subsets, computes ordinal ranks) and FORL (learns stationary best-response policies) with a TRXL-I (models policies within FORL) architecture, to enable strategic planning in multi-robot systems.
- The approach proves the existence of a unique pure Nash equilibrium in complete and star graphs, and evaluates generalizability and scalability on real-world road networks.
- The learned policies achieve 87% to 95% optimality of an equivalent TOP solution and demonstrate better generalizability and scalability to larger team sizes.

---

[OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows](http://arxiv.org/abs/2510.24411)

- OS-Sentinel: introduces a novel hybrid safety detection framework that synergistically combines a Formal Verifier for detecting explicit system-level violations with a VLM-based Contextual Judge for assessing contextual risks and agent actions, enhancing safety in mobile GUI agents.
- The framework operates at both step-level for real-time guard functionality and trajectory-level for post-hoc analysis, adapting to different scenarios through flexible aggregation strategies.
- It leverages MobileRisk-Live, a dynamic Android emulator sandbox, and MobileRisk, a benchmark with fine-grained agent trajectories, to enable systematic and reproducible safety research.

---

[APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training](http://arxiv.org/abs/2510.24397)

- APTBench: introduces a framework for benchmarking agentic potential of base LLMs during pre-training, utilizing Task & Trajectory Collection, Agent-oriented Question Formulation, and Answer Generation to assess Core Agentic Abilities (planning, action, atomic abilities) across Software Engineering and Deep Research scenarios via Multiple-Choice and Text Completion questions.
- The framework converts complex multi-turn agent tasks and successful trajectories into base model-suitable questions, bypassing the need for instruction-following capabilities.
- APTBench provides a more predictive and cost-effective signal of a model's downstream agent performance compared to traditional general-purpose benchmarks, guiding effective pre-training.

---

[Policy Cards: Machine-Readable Runtime Governance for Autonomous AI Agents](http://arxiv.org/abs/2510.24383)

- Policy Cards: introduces a machine-readable, deployment-layer standard for expressing operational, regulatory, and ethical constraints for AI agents, including meta, scope, applicable_policies, controls, obligations, monitoring, kpis_thresholds, change_management, assurance_mapping, and references components, enabling verifiable compliance for autonomous agents.
- The framework integrates with a validator, enforcement engines, and monitoring subsystems within a Declare-Do-Audit lifecycle to ensure continuous assurance and automated audit feedback.
- Policy Cards extend existing transparency artifacts by providing a normative layer that defines binding operational policy, allowing AI agents to interpret and enforce their own policies and supporting multi-agent governance with cryptographic verification.

---

[Manipulate as Human: Learning Task-oriented Manipulation Skills by Adversarial Motion Priors](http://arxiv.org/abs/2510.24257)

- HMAMP (Adversarial Motion Priors): introduces a novel approach for learning human-style manipulation skills by leveraging adversarial networks to model complex dynamics of tool and object manipulation, using human demonstration videos, keypoint detection, adversarial motion priors, a simulation environment, a policy network, and combined goal and style rewards.
- The framework trains a policy to generate realistic motion trajectories that match human motion's statistical properties, enabling robots to manipulate tools and objects in a human-like manner.
- HMAMP demonstrates superior performance in task completion efficiency, knock impulse, and energy efficiency for hammering tasks, bridging the gap between robotic and human manipulation capabilities.

---

[SynAD: Enhancing Real-World End-to-End Autonomous Driving Models through Synthetic Data Integration](http://arxiv.org/abs/2510.24052)

- SynAD (Synthetic Data Integration for Autonomous Driving): introduces a framework to enhance real-world end-to-end autonomous driving models by integrating synthetic data, including Ego-Centric Scenario Generation, Map-to-BEV Network, BEVFormer, and an End-to-End Autonomous Driving Model with motion forecasting, occupancy prediction, and planning modules.
- The framework generates ego-centric synthetic scenarios from path-level data and converts them into BEV features using a novel Map-to-BEV Network, bypassing the need for sensor inputs.
- This approach effectively bridges synthetic scenario generation with end-to-end autonomous driving, improving safety performance and robustness by diversifying training data.

---

[AGENT DATA PROTOCOL: UNIFYING DATASETS FOR DIVERSE, EFFECTIVE FINE-TUNING OF LLM AGENTS](http://arxiv.org/abs/2510.24702)

- ADP (Agent Data Protocol): introduces a lightweight representation language that unifies heterogeneous agent datasets into a single schema, consumable by various agent harnesses, turning fragmented data into a scalable training pipeline.
- The protocol standardizes agent interactions into `Trajectory` objects, comprising `Actions` (API, code, message) and `Observations` (text, web) to facilitate large-scale supervised fine-tuning of LLM agents.
- This standardization significantly reduces engineering effort for integrating diverse datasets and enables substantial performance gains and cross-task transfer for LLM agents across various domains.

---

[AgentFold: Long-Horizon Web Agents with Proactive Context Management](http://arxiv.org/abs/2510.24699)

- AgentFold: introduces a novel agent paradigm for long-horizon web tasks, centered on proactive context management inspired by human cognitive processes, dynamically sculpting its context through multi-scale folding operations.
- The framework's context is partitioned into a user question, available tools, Multi-Scale State Summaries (long-term memory), and Latest Interaction (working memory), enabling strategic planning and precise situational action.
- Its folding mechanism, comprising Granular Condensation and Deep Consolidation, allows it to preserve fine-grained details while abstracting irrelevant history, resolving the trade-off between context comprehensiveness and conciseness.

---

[AgentFrontier: Expanding the Capability Frontier of LLM Agents with ZPD-Guided Data Synthesis](http://arxiv.org/abs/2510.24695)

- AgentFrontier Engine: introduces a data synthesis framework that generates high-quality, multidisciplinary data within an LLM's ZPD (Zone of Proximal Development), utilizing LKP (Less Knowledgeable Peer) and MKO (More Knowledgeable Other) agents for adversarial calibration and an Arefine (refinement agent) with various Tools (external resource suite) for iterative complexity escalation.
- This framework operationalizes the ZPD theory to create training data that is challenging for a base LLM (LKP) but solvable with support from a tool-augmented agent (MKO), fostering knowledge fusion and complex reasoning.
- The engine produces two main datasets, Dpretrain (pre-training dataset) for continued pre-training and DZPD (ZPD dataset) for targeted post-training, and also establishes the ZPD Exam, a self-evolving benchmark for evaluating LLM agent capabilities on frontier tasks.

---

[Repurposing Synthetic Data for Fine-grained Search Agent Supervision](http://arxiv.org/abs/2510.24694)

- E-GRPO (Entity-aware Group Relative Policy Optimization): introduces a novel reinforcement learning framework that enhances policy optimization for LLM-based search agents by formulating a dense, entity-aware reward function, assigning partial rewards to incorrect samples proportional to their entity match rate, and leveraging ground-truth entities from synthetic data generation.
- The framework integrates an LLM-based search agent with tools (Search, Visit) to interact with an environment, utilizing synthetic data generated with embedded ground-truth entities to provide fine-grained supervision beyond sparse outcome-based rewards.
- E-GRPO's entity-aware reward function, derived from the normalized entity match rate, enables the model to learn more efficient reasoning policies requiring fewer tool calls and consistently outperforms the GRPO baseline in accuracy across diverse QA and deep research benchmarks.

---

[Evolving Diagnostic Agents in a Virtual Clinical Environment](http://arxiv.org/abs/2510.24654)

- DiagGym and DiagAgent Framework: introduces a system for training LLMs as diagnostic agents using reinforcement learning, featuring DiagGym (a virtual clinical environment) that simulates examination results and DiagAgent (a diagnostic trajectory manager) that learns optimal diagnostic policies.
- The framework leverages Restructured EHRs to create realistic patient profiles and past examination records, enabling DiagAgent to interactively query for examinations and receive simulated results, ultimately leading to a final diagnosis.
- Through End-to-end Multi-turn RL and a Reward Calculation mechanism, the framework's Policy Evolving process explores different diagnostic trajectories to optimize for information yield and diagnostic accuracy within the virtual environment.

---

[OPENREWARD: LEARNING TO REWARD LONG-FORM AGENTIC TASKS VIA REINFORCEMENT LEARNING](http://arxiv.org/abs/2510.24636)

- OPENRM (OPENREWARD: Learning to Reward Long-Form Agentic Tasks via Reinforcement Learning): introduces a tool-augmented long-form reward model that systematically judges open-ended responses by invoking external tools to gather relevant evidence, trained with Group Relative Policy Optimization (GRPO) using a composite reward function (Rtool, REM) and a controllable data synthesis framework (Target-aware Query Generation, Positive-Negative Pair Synthesis).
- The framework enables LLMs to actively retrieve, verify, and reason over external information, improving judgment accuracy and generalization for knowledge-intensive tasks.
- OPENRM significantly outperforms existing reward modeling approaches on multiple benchmarks and serves as an effective data selector for downstream LLM alignment tasks.

---

[Generative AI for Healthcare: Fundamentals, Challenges, and Perspectives](http://arxiv.org/abs/2510.24551)

- SAGE-Health (Sustainable, Adaptive, and Generative Ecosystem for Healthcare): introduces a data-centric paradigm for GenAI in healthcare, with a Sustainable Medical Data Ecosystem, Adaptive Medical GenAI Layer, Agentic Collaboration Layer, and Healthcare Application Layer, designed to integrate, represent, and retrieve diverse medical data and knowledge for generative healthcare systems.
- The framework repositions the data life cycle as the foundational substrate, enabling GenAI-powered operations for model components and clinical applications through intelligent data management and continuous feedback loops.
- SAGE-Health addresses data fragmentation, lifecycle governance, and data-model co-evolution challenges, ensuring high-quality, context-aware inputs for model training, prompting, and fine-tuning, while supporting task-specific inference via an agentic layer.

---

[Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems](http://arxiv.org/abs/2510.24476)

- Agentic System (Framework Integrating RAG and Reasoning Enhancement for Comprehensive Hallucination Mitigation): introduces a unified conceptual framework that integrates Retrieval-Augmented Generation (RAG) and Reasoning Enhancement, alongside internal agentic components, to mitigate LLM hallucinations.
- The framework addresses both knowledge-based and logic-based hallucinations by enhancing LLMs' capabilities through verifiable knowledge grounding, logical consistency constraints, and dynamic planning mechanisms.
- It systematically reviews RAG's pre-retrieval, retrieval, and post-retrieval stages, and three reasoning enhancement approaches (CoT, Tool-Augmented, Symbolic Reasoning), demonstrating their practical value and generalizability.

---

[Law in Silico: Simulating Legal Society with LLM-Based Agents](http://arxiv.org/abs/2510.24442)

- Law in Silico: introduces an LLM-based agent framework for simulating legal societies, integrating Hierarchical Legal Agent Modeling, Scenario-Based Decision-Making, a dynamic Legal System, an LLM-powered Game Master, and LLM-based Agents to model individual and institutional behaviors.
- The framework enables both macro-level simulations to reproduce crime trends and micro-level simulations to explore how legal mechanisms influence agent-level outcomes in interactive conflict scenarios.
- By incorporating factors like judicial corruption and varying punishment impressions, the framework provides insights into effective legal system design and the protection of vulnerable populations.

---

[Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](http://arxiv.org/abs/2510.24438)

- Dual-Agent framework: introduces a system for evaluating LLM-generated Islamic content, featuring a Quantitative Agent for numerical scoring across six criteria and a Qualitative Agent for side-by-side comparison across five dimensions, both leveraging specialized verification tools.
- This framework systematically assesses theological accuracy, citation integrity, and stylistic appropriateness of LLM outputs by linking them to reference-level verifications.
- The framework provides an interpretable and auditable method for evaluating LLMs in high-stakes domains, ensuring faithful and reliable content generation.

---

[CodeWiki: Automated Repository-Level Documentation at Scale](http://arxiv.org/abs/2510.24428)

- CodeWiki: introduces an open-source framework for holistic repository-level documentation, with Repository Analysis and Hierarchical Module Decomposition (Initial processing), Recursive Documentation Generation (Agent-based processing), and Hierarchical Assembly and Documentation Synthesis (Final documentation creation), which generates diverse content types like architecture diagrams and data flow visualizations.
- The framework employs hierarchical decomposition to break down complex repositories into manageable modules, recursive agentic processing with dynamic delegation for adaptive handling, and comprehensive synthesis of textual and visual artifacts.
- CodeWiki supports documentation generation across seven programming languages and achieves high quality scores, outperforming existing closed-source systems and demonstrating scalable, accurate documentation for real-world repositories.

---

[Improving LLM Reasoning via Dependency-Aware Query Decomposition and Logic-Parallel Content Expansion](http://arxiv.org/abs/2510.24390)

- Orion: introduces a novel LLM reasoning framework that decomposes complex queries into two synergistic phases: key point generation and content parallel expansion, significantly improving token generation speed and reducing latency.
- The framework utilizes a dependency-aware parallel expansion algorithm that models inter-point relationships using a Directed Acyclic Graph (DAG), enabling parallelized content expansion while maintaining logical coherence.
- Orion further incorporates a pipeline scheduling mechanism that exploits complementary computational characteristics of the two phases to achieve cross-query parallelism, enhancing overall efficiency and quality.

---

[Automatically Benchmarking LLM Code Agents through Agent-driven Annotation and Evaluation](http://arxiv.org/abs/2510.24358)

- PRDBench (Product Requirement Document-centered benchmark): introduces an agent-driven benchmark construction pipeline for LLM code agents, including seed tasks, PRD, metric outline, code agent, scaffold, criteria scheme, test artifacts, human inspector, EvalAgent (with its tools and metrics), and reports, designed to efficiently generate diverse project-level tasks with flexible evaluation.
- This framework addresses high annotation costs and rigid unit test-based evaluation by leveraging LLMs to generate project scaffolding and evaluation criteria, with human annotators primarily verifying compatibility and reasonableness.
- PRDBench employs an Agent-as-a-Judge paradigm via EvalAgent, which uses various tools and metrics (unit, shell interaction, file comparison) to automate comprehensive code agent evaluation beyond traditional unit tests.

---

[VDSAgents: A PCS-Guided Multi-Agent System for Veridical Data Science Automation](http://arxiv.org/abs/2510.24339)

- VDSAgents (A PCS-Guided Multi-Agent System for Veridical Data Science Automation): introduces a multi-agent system for automated data science, integrating a PCS-Guided Workflow, a Modular Multi-Agent Architecture including a central PCS-Agent, and a Scientific Tool Integration for robust and reproducible data analysis.
- The system decomposes the data science lifecycle into five sequential stages, each managed by a dedicated agent (Define-, Explore-, Model-, Evaluate-Agents), with the PCS-Agent continuously analyzing intermediate outputs, performing perturbation testing, and evaluating result stability and consistency.
- This framework leverages Predictability-Computability-Stability (PCS) principles as an external planning framework, providing theoretical guidance to LLM-driven agents for trustworthy, stable, and reproducible automated data science.

---

[Cybersecurity AI Benchmark (CAIBench): A Meta-Benchmark for Evaluating Cybersecurity AI Agents](http://arxiv.org/abs/2510.24317)

- CAIBench (Cybersecurity AI Benchmark): introduces a modular meta-benchmark framework for evaluating LLM models and agents across offensive, defensive, knowledge-based, and privacy-preserving cybersecurity domains.
- The framework integrates five evaluation categories, a five-tier difficulty classification system, and a hybrid infrastructure combining Docker containers for practical challenges and Python-based scripts for knowledge and privacy assessments.
- CAIBench supports systematic simultaneous offensive-defensive evaluation, robotics-focused cybersecurity challenges (RCTF2), and privacy-preserving performance assessment (CyberPII-Bench) to measure labor-relevant agentic cybersecurity tasks.

---

[Retrieval and Argumentation Enhanced Multi-Agent LLMs for Judgmental Forecasting](http://arxiv.org/abs/2510.24303)

- Multi-Agent QBAF Combinator Framework: introduces a novel multi-agent framework for claim verification that combines argumentative reasoning from multiple independent agents to improve judgmental forecasting, with Claim (Input statement for verification) / External Sources (Provides textual evidence) / QBAF Generator Agents (Generates QBAFs from claims) / ArgLLM agent (Generates/evaluates QBAFs using LLM) / RAG-ArgLLM agent (ArgLLM with external RAG evidence) / RbAM agent (Mines arguments from external sources) / Multi-Agent QBAF Combinator (Aggregates QBAFs from multiple agents) / Similarity Calculation (Measures argument semantic similarity) / Argument Combination (Merges similar arguments into BAF) / Base Score Aggregation (Aggregates combined argument scores) / Combined QBAF (Final aggregated QBAF output).
- The framework aggregates Quantitative Bipolar Argumentation Frameworks (QBAFs) generated by LLM-based agents, including baseline ArgLLMs and RAG-enhanced ArgLLMs, as well as RbAM agents that mine arguments from external sources.
- Combining diverse argumentative perspectives from these agents, especially with three agents and external data, significantly enhances forecasting accuracy and provides transparent claim verification.

---

[MCP-FLOW: FACILITATING LLM AGENTS TO MASTER REAL-WORLD, DIVERSE AND SCALING MCP TOOLS](http://arxiv.org/abs/2510.24284)

- MCP-Flow: introduces an automated web-agent-driven pipeline for large-scale server discovery, data synthesis, and model training, including a Web Agent for automated server crawling from Marketplaces, a Local MCP Client for local server deployment, a Generative Model for instruction and function call generation, and multiple filtration components (Embedding Similarity, Tool Invocation, Quality Score, Trajectory) to ensure high-quality data.
- The framework collects and filters data from 1166 servers and 11536 tools, producing 68733 high-quality instruction-function call pairs and 6439 trajectories, significantly exceeding prior work in scale and diversity.
- MCP-Flow provides a scalable foundation for advancing LLM agents' proficiency in real-world Model Contextual Protocol (MCP) environments, driving superior MCP tool selection, function-call generation, and enhanced agentic task performance.

---

[GRAPHIA: Harnessing Social Graph Data to Enhance LLM-Based Social Simulation](http://arxiv.org/abs/2510.24251)

- Graphia: introduces a novel LLM-based social graph simulation framework that leverages graph data as supervision for LLM post-training via reinforcement learning, including Graphia-Q (predicts interaction partners), Graphia-E (generates interaction messages and categories), Activity Predictor (identifies active nodes for interaction), Reward Functions (optimizes LLM agents for tasks), and a Graph Generation Pipeline (assembles simulated social graphs).
- The framework evaluates simulations under Transductive Dynamic Graph Generation (TDGG) for micro-level alignment and Inductive Dynamic Graph Generation (IDGG) for macro-level alignment, using specialized metrics.
- Graphia supports counterfactual simulations, demonstrating its ability to model plausible behavioral shifts in social graphs under various platform incentives.

---

[MGA: Memory-Driven GUI Agent for Observation-Centric Interaction](http://arxiv.org/abs/2510.24168)

- MGA (Memory-Driven GUI Agent): introduces a framework that reframes GUI interaction under an "observe-then-decide" paradigm, utilizing an Observer (detects & structures UI), a Memory Agent (retains & updates cross-step memory), a Planner (reasons next action), and a Grounding Agent (executes on screen) to enable robust, memory-driven reasoning.
- The framework models each interaction step as an independent, context-rich environment state, composed of the current screenshot, task-agnostic spatial-semantic information, and a dynamically updated structured memory, decoupling decisions from fragile long-chain trajectories.
- This design addresses challenges of historical trajectory inertia and local exploration bias in GUI agents by providing comprehensive environmental grounding and de-biased temporal continuity.

---

[Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](http://arxiv.org/abs/2510.24126)

- Reinforcement Learning for Long-Horizon Multi-Turn Search Agents: introduces an approach that leverages RL to train LLM agents for complex, multi-turn document search tasks, utilizing a Base LLM, a Reward Model, specialized Agent Architecture tools, and a comprehensive RL Training Libraries stack, along with LLM Serving & Context Management, Policy Optimization, and a defined Reward Structure, all evaluated on a Legal Document Search Benchmark.
- The paper demonstrates that an RL-trained 14B parameter model significantly outperforms frontier class models on a legal document search benchmark, achieving 85% accuracy compared to 78%.
- Experiments with turn-restricted regimes highlight that RL-trained agents achieve better results when allowed longer multi-turn horizons, indicating the importance of learned exploration strategies over prompt-based approaches for complex interactive tasks.

---

[PFEA: An LLM-based High-Level Natural Language Planning and Feedback Embodied Agent for Human-Centered AI](http://arxiv.org/abs/2510.24109)

- PFEA (Planning and Feedback Embodied Agent): introduces an LLM-based framework for human-centered AI, integrating a human-robot voice interaction module, a vision-language agent module, and an action execution module, where the vision-language agent includes a planner, a converter, and an evaluator.
- This framework enables robots to understand and execute human-centered, high-level natural language instructions by leveraging LLMs for task planning and feedback mechanisms in both simulated and real-world environments.
- PFEA significantly improves task success rates by incorporating visual environmental perception and a feedback control mechanism that allows the agent to adjust strategies and actions based on execution outcomes.

---

[Pie: A Programmable Serving System for Emerging LLM Applications](http://arxiv.org/abs/2510.24051)

- PIE (A Programmable Serving System): introduces a programmable LLM serving system that decomposes the traditional generation loop into fine-grained service handlers and delegates control to user-provided inferlets, enabling flexible and efficient LLM applications.
- The system employs a three-layer architecture—application, control, and inference—to manage inferlet lifecycles, orchestrate resources, batch API calls, and execute low-level model inference tasks.
- Its design allows for application-specific KV cache control, customizable generation processes, and seamless integration of arbitrary computation and I/O within the LLM generation workflow.

---

[Human-Machine Social Hybrid Intelligence: A Collaborative Decision-Making Framework for Large Model Agent Groups and Human Experts](http://arxiv.org/abs/2510.24030)

- HMS-HI (Human-Machine Social Hybrid Intelligence): introduces a novel architecture for deep, collaborative decision-making between human experts and LLM-powered AI agents, built upon a Shared Cognitive Space, a Dynamic Role and Task Allocation module, and a Cross-Species Trust Calibration protocol.
- The framework treats human and AI agents as peers, dynamically allocating roles and tasks based on capabilities and workload, and fostering transparency and mutual adaptation through structured explanations and feedback.
- Validated in an urban emergency response simulation, HMS-HI significantly reduced civilian casualties and cognitive load compared to traditional human-in-the-loop approaches.

---

[TEXT2DB : Integration-Aware Information Extraction with Large Language Model Agents](http://arxiv.org/abs/2510.24014)

- OPAL (Observe-Plan-Analyze Language Model): introduces TEXT2DB, a new task for integration-aware information extraction, with Observer agent, Planner agent, Analyzer agent, and various IE and database integration tools, where the system updates a database from text based on user instructions.
- The framework leverages LLM agents to dynamically adapt to diverse database schemas and resolve extraction ambiguities by generating code-based plans and utilizing feedback for self-correction.
- OPAL's components, including the Observer agent for database insights and the Analyzer agent for code quality, significantly improve the effectiveness of database updates by providing demonstrations and integrity checks.

---

[Enhancing Hierarchical Reinforcement Learning through Change Point Detection in Time Series](http://arxiv.org/abs/2510.24988)

- CPD-Option-Critic (CPD Integrated Option Critic framework): introduces a novel architecture that integrates a self-supervised, Transformer-based Change Point Detection (CPD) module into the Option-Critic framework, enabling adaptive segmentation of state trajectories and the discovery of options.
- The framework leverages CPD outputs to supervise termination functions, initialize intra-option policies via behavioral cloning, and enforce inter-option diversity through KL-regularized specialization.
- This integration enhances learning stability, accelerates convergence, and improves robustness in both discrete and continuous domains by aligning option boundaries with latent state transitions.

---

[SCOUT: A Lightweight Framework for Scenario Coverage Assessment in Autonomous Driving](http://arxiv.org/abs/2510.24949)

- SCOUT (Scenario Coverage Oversight and Understanding Tool): introduces a lightweight framework for scenario coverage assessment in autonomous driving, utilizing a Human Annotator (ground truth labeler), a Fine-Tuned LVLM (teacher model), Coverage Definitions and Requirements (scenario criteria), Sensor Representations (latent features input), and a Distilled Surrogate Model (SCOUT) (lightweight predictor), where SCOUT efficiently predicts scenario coverage labels from precomputed sensor latent representations.
- The framework addresses the high cost and inefficiency of human annotation and direct LLM inference by distilling LVLM-generated labels into a smaller, faster surrogate model.
- This approach enables scalable and efficient real-time scenario coverage estimation for autonomous systems, leveraging existing perception stack outputs to minimize computational overhead.

---

[Idea2Plan: Exploring AI-Powered Research Planning](http://arxiv.org/abs/2510.24891)

- Idea2Plan: introduces a framework for evaluating LLMs' research planning capabilities, including a Research Idea input, Method (Baselines/ReAct Agent with Tools), Generated Research Plan, and Evaluation against Reference Research Plans using a Grading Rubric and LLM-based Judges.
- The framework formalizes the Idea2Plan task, where LLMs generate structured research plans from research ideas, and introduces Idea2Plan Bench, a benchmark built from post-LLM-cutoff AI papers for rigorous evaluation.
- It employs both direct prompting baselines and an agentic ReAct approach with arXiv search and paper reading tools, demonstrating that GPT-5 and GPT-5-mini achieve the strongest performance, though substantial headroom for improvement remains.

---

[Automating Benchmark Design](http://arxiv.org/abs/2510.25039)

- BeTaL (Benchmark Tuning with an LLM-in-the-loop): introduces a framework that leverages environment design principles to automate dynamic benchmark design, using a Designer LLM to reason through parameter spaces, a Problem/Task Generator to simulate problems, a Target Model for evaluation, and an iterative loop with feedback to achieve target properties like difficulty. 
- The framework iteratively refines benchmark parameters by comparing the Target Model's performance against a Target Performance goal, using the Performance Gap as a metric to guide the Designer LLM's parameter adjustments. 
- BeTaL consistently outperforms baselines in creating benchmarks with desired difficulty levels across arithmetic, spatial reasoning, and agentic tasks, demonstrating robust and efficient benchmark generation. 

---

[StorageXTuner: An LLM Agent-Driven Automatic Tuning Framework for Heterogeneous Storage Systems](http://arxiv.org/abs/2510.25017)

- StorageXTuner: introduces an LLM-agent-based automatic tuning framework for heterogeneous storage systems, with Executor (deploys/benchmarks/monitors storage systems), Extractor (analyzes/summarizes performance data), Searcher (explores/proposes configurations), and Reflector (manages/updates tuning insights).
- The framework employs an insight-driven tree search with a layered memory to promote empirically validated insights and uses lightweight checkers to guard against unsafe actions.
- StorageXTuner demonstrates robust performance improvements across various storage systems and workloads, achieving higher throughput and reduced latency with fewer trials.

---

[Emergent Coordinated Behaviors in Networked LLM Agents: Modeling the Strategic Dynamics of Information Operations](http://arxiv.org/abs/2510.25003)

- GABM (Generative Agent-Based Modeling): introduces a framework for simulating emergent coordinated behaviors in networked LLM agents within information operations, utilizing generative agents with persona, memory, and action policies in a simulated social media environment.
- The framework evaluates coordination across operational regimes (Common Goal, Teammate Awareness, Collective Decision-Making) and provides an interactive LLMXIO 3D Simulation Dashboard for real-time analysis of network dynamics and campaign impact.
- This research demonstrates that generative agents can autonomously reproduce real-world IO coordination strategies, highlighting societal risks from increasingly automated, self-organizing influence campaigns.

---

[OrchVis: Hierarchical Multi-Agent Orchestration for Human Oversight](http://arxiv.org/abs/2510.24937)

- OrchVis (Hierarchical Multi-Agent Orchestration for Human Oversight): introduces a multi-agent orchestration framework that visualizes, verifies, and coordinates goal-driven collaboration among LLM-based agents, including an orchestration agent, goal parser, verifier, re-planner, sub-agents, planning panel, and user interface, to enable human oversight of complex multi-agent workflows.
- The framework parses user intent into structured goals, monitors execution via automated verification, and exposes inter-agent dependencies through an interactive planning panel, allowing users to explore system-proposed alternatives and selectively replan when conflicts arise.
- OrchVis advances human-centered design for multi-agent systems by combining transparent visualization with adaptive autonomy, facilitating strategic guidance without micromanaging each step.

---

[From Narrative to Action: A Hierarchical LLM-Agent Framework for Human Mobility Generation](http://arxiv.org/abs/2510.24802)

- Narrative-to-Action (Hierarchical LLM-Agent Framework): introduces a multi-layer cognitive architecture for human mobility generation and simulation, integrating high-level narrative reasoning, mid-level reflective planning, and low-level behavioral execution.
- The framework employs a "creative writer" LLM for diary-style narratives and a "structural parser" LLM for machine-readable plans, enhanced by a dynamic execution module for adaptive behavioral adjustments.
- It incorporates Mobility Entropy by Occupation (MEO) to capture heterogeneous schedule flexibility, enabling cognition-driven modeling that produces realistic synthetic data and interpretable human travel decision-making.

---

[OSWORLD-MCP: BENCHMARKING MCP TOOL INVOCATION IN COMPUTER-USE AGENTS](http://arxiv.org/abs/2510.24563)

- OSWorld-MCP: introduces a comprehensive and fair benchmark for evaluating computer-use agents, integrating GUI operations and MCP tool invocation in real-world scenarios, featuring an agent's Thought and Actions (GUI/MCP), a Coordinator for task management, a Config for evaluation, a Virtual Machine Platform for execution, a Simulator, and an Automated Code Generation Pipeline for tool creation.
- The benchmark extends OSWorld by incorporating 158 high-quality MCP tools across 7 common applications, enabling dynamic interaction between GUI operations and tool usage for a holistic assessment of LLM capabilities.
- It introduces new metrics, Tool Invocation Rate (TIR) and Average Completion Steps (ACS), to provide nuanced insights into an agent's tool utilization propensity and task completion efficiency.

---

#### 27th October 2025

[A Survey of Data Agents: Emerging Paradigm or Overstated Hype?](http://arxiv.org/abs/2510.23587)

- Data Agents (Hierarchical Taxonomy): introduces a systematic framework for classifying LLM-powered autonomous systems, spanning six levels of autonomy from manual operations to fully generative data agents.
- The taxonomy delineates progressive shifts in control and responsibility between humans and data agents, clarifying capability boundaries and accountability across data management, preparation, and analysis tasks.
- It integrates core agent components like planning, perception, memory, and tool-calling with external resources such as LLM Hubs, Data Lakes, and various tools to define the evolutionary leaps in data agent intelligence.

---

[Model Proficiency in Centralized Multi-Agent Systems: A Performance Study](http://arxiv.org/abs/2510.23447)

- Team Proficiency Assessment Framework: introduces a method for assessing team proficiency in centralized multi-agent systems, utilizing the Measurement Prediction Bound (MPB), Kolmogorov-Smirnov (KS) statistic, and Kullback-Leibler (KL) divergence to quantify discrepancies between predicted and actual measurements.
- The framework evaluates model reliability without requiring knowledge of the true hidden state, providing practical metrics for in situ assessment and a theoretical benchmark.
- Applied in a target tracking scenario, the framework demonstrates that MPB and KS metrics accurately capture model mismatches and align with KL divergence, enabling real-time proficiency assessment.

---

[Multi-Stakeholder Alignment in LLM-Powered Collaborative AI Systems: A Multi-Agent Framework for Intelligent Tutoring](http://arxiv.org/abs/2510.23245)

- AGL (Advisory Governance Layer): introduces a non-intrusive, multi-agent framework designed to enable distributed stakeholder participation in AI governance for Intelligent Tutoring Systems (ITS), including Stakeholder Agents (SH), Multi-Stakeholder Negotiation Agent (MSN), Audit and Governance Agent (AG), System Oversight Agent (SO), Intelligent Tutoring System (ITS), and Governance Body, where it provides structured, auditable governance advice without altering the ITS's core pedagogical decision-making.
- The framework employs specialized LLM-powered agents to evaluate pedagogical actions against specific policies in a privacy-preserving manner, utilizing a novel policy taxonomy and conflict-resolution protocols.
- AGL contributes a reference architecture and technical specifications for aligning educational AI with multi-stakeholder values, bridging high-level ethical principles and practical implementation.

---

[SI-Bench: Benchmarking Social Intelligence of Large Language Models in Human-to-Human Conversations](http://arxiv.org/abs/2510.23182)

- SI-Bench: introduces a novel benchmark for evaluating LLMs' social intelligence in human-to-human conversations, utilizing a multi-dimensional framework that decouples reasoning processes from reply quality.
- The framework includes a data collection and quality control pipeline, a taxonomy of 12 complex social situations, and an evaluation process assessing contextual understanding, response strategy, and reply generation.
- SI-Bench compares Chain-of-Thought (CoT) and direct replies, revealing a thought-action gap where SOTA models excel in reasoning but often generate socially stiff or unnatural responses.

---

[Adapting Interleaved Encoders with PPO for Language-Guided Reinforcement Learning in BabyAI](http://arxiv.org/abs/2510.23148)

- PDiT (Perception-Decision Interleaving Transformer): introduces an interleaved perception-decision transformer architecture for language-guided reinforcement learning, combining PPO and a CLIP-style contrastive loss for improved stability and multimodal grounding.
- This framework addresses the inefficiency of separate perception and decision modules by allowing direct feedback from decision-making to refine perceptual features.
- The integration of interleaved encoders with policy optimization and explicit multimodal grounding signals significantly enhances effectiveness in complex RL tasks.

---

[ALITA-G: SELF-EVOLVING GENERATIVE AGENT FOR AGENT GENERATION](http://arxiv.org/abs/2510.23601)

- ALITA-G (Self-Evolving Generative Agent for Agent Generation): introduces a self-evolution framework that transforms a general-purpose agent into a domain expert by systematically generating, abstracting, and curating Model Context Protocol (MCP) tools, which are then used by a specialized agent for task solving.
- The framework employs a multi-execution strategy where a Master Agent synthesizes candidate MCPs from successful trajectories, which are then abstracted and consolidated into an MCP Box for retrieval-augmented selection.
- This approach significantly improves accuracy and efficiency on complex reasoning tasks by providing specialized agents with a focused, relevant set of tools tailored to specific task categories.

---

[MULTI-AGENT EVOLVE: LLM SELF-IMPROVE THROUGH CO-EVOLUTION](http://arxiv.org/abs/2510.23595)

- MAE (Multi-Agent Evolve): introduces a multi-agent self-evolving framework that enables LLMs to self-evolve in diverse tasks, utilizing a Proposer (generates questions/challenges), Solver (attempts to answer questions), Judge (evaluates questions/answers, provides rewards), Large Language Model (LLM) (backbone for all agents), Task-Relative REINFORCE++ (RL algorithm for agent training), Quality Filtering (removes low-quality questions), Reward Mechanisms (guides agent learning), Question Dataset (D) (stores valid questions), Pair Dataset (P) (stores question-answer pairs), and Synchronized Parameter Update (updates shared LLM parameters) to form a closed self-improving loop.
- The framework instantiates three interactive roles (Proposer, Solver, Judge) from a single LLM, applying reinforcement learning to optimize their behaviors without human-curated supervision.
- The Proposer generates questions, the Solver provides solutions, and the Judge evaluates both, co-evolving through adversarial interaction and self-rewarding mechanisms to enhance the LLM's general reasoning abilities.

---

[RECODE: UNIFY PLAN AND ACTION FOR UNIVERSAL GRANULARITY CONTROL](http://arxiv.org/abs/2510.23564)

- RECODE (Recursive Code Generation): introduces a novel paradigm unifying planning and action within a single code representation, enabling LLM-based agents to dynamically control decision granularity.
- This framework treats high-level plans as abstract placeholder functions that are recursively decomposed into finer-grained sub-functions until primitive actions are reached.
- The recursive structure inherently generates rich, multi-granularity training data, enhancing the agent's ability to learn hierarchical decision-making processes and improving data efficiency.

---

[Deductive Chain-of-Thought Augmented Socially-aware Robot Navigation World Model](http://arxiv.org/abs/2510.23509)

- NaviWM: introduces a socially-aware robot navigation world model that augments LLM reasoning with a structured world model and a logic-driven chain-of-thought process, enabling robots to generate socially compliant and physically safe navigation decisions.
- The framework integrates a spatial-temporal world model (captures agent states and interactions) and a deductive reasoning module (guides LLMs through multi-step, logic-based inference using a Gentzen deduction proof tree) to ensure robust social navigation.
- NaviWM formalizes social norms as first-order logic constraints, including Activity-Awareness, Distance-Awareness, Collision-Avoidance, and Time-Constraint, enabling interpretable and verifiable reasoning in dynamic human environments.

---

[Are Agents Just Automata? On the Formal Equivalence Between Agentic AI and the Chomsky Hierarchy](http://arxiv.org/abs/2510.23487)

- Automata-Agent Framework: introduces a formal framework establishing a direct correspondence between agentic AI architectures and the Chomsky hierarchy by classifying agents based on their memory models to determine computational power, decidability, and verifiability.
- The framework maps simple reflex agents to Finite Automata, hierarchical task-decomposition agents to Pushdown Automata, and agents with readable/writable memory to Turing Machines.
- This classification provides a principled methodology for right-sizing agent architectures, optimizing computational efficiency and cost, and enabling formal verification and quantitative risk analysis for LLM-based agents.

---

[BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents](http://arxiv.org/abs/2510.23458)

- BrowseConf: introduces a Test-Time Scaling (TTS) method for LLM-based search agents, utilizing an LLM-based search agent, Confidence Threshold (τ), Rollout Budget (N), Search Tool, Visit Tool, Prompt Templates, Decision Logic, and State Management to dynamically allocate computational resources based on verbalized confidence scores.
- The framework initiates new attempts only when the LLM's final confidence score falls below a calibrated threshold, contrasting with fixed-budget TTS methods that apply a uniform number of rollouts.
- BrowseConf significantly reduces token consumption and improves efficiency by avoiding redundant attempts for queries where the agent is already highly confident in its initial response.

---

[AutoStreamPipe: LLM Assisted Automatic Generation of Data Stream Processing Pipelines](http://arxiv.org/abs/2510.23408)

- AutoStreamPipe (LLM Assisted Automatic Generation of Data Stream Processing Pipelines): introduces a novel framework that automates the design, generation, and deployment of stream processing pipelines by bridging the semantic gap between high-level user intent and platform-specific implementations using LLMs.
- The framework integrates a Query to Pipeline Procedure, Hypergraph of Thoughts (HGoT) for structured reasoning, a Multi-Agent System (LLMs) for collaborative execution, and a Resilient Execution Infrastructure to produce production-ready pipelines.
- AutoStreamPipe significantly reduces development time by 6.3x and error rates by 5.19x compared to LLM code-generation methods, offering a robust and efficient solution for diverse stream processing systems.

---

[Code Aesthetics with Agentic Reward Feedback](http://arxiv.org/abs/2510.23272)

- GRPO-AR (Agentic Reward Feedback with GRPO Algorithm): introduces a novel pipeline to enhance LLM-generated code aesthetics by integrating supervised fine-tuning on AesCode-358K and reinforcement learning with a multi-agent reward system.
- This framework employs an Execution Agent for code executability, a Static Aesthetics Agent for visual quality assessment, and an Interactive Aesthetics Agent for evaluating usability, all contributing to comprehensive feedback.
- The system leverages GRPO to optimize the policy model based on aggregated rewards, enabling AesCoder models to achieve state-of-the-art performance in visually-oriented coding tasks.

---

[Evaluation of Vision-LLMs in Surveillance Video](http://arxiv.org/abs/2510.23190)

- Training-Free Anomaly Classification Framework: introduces a zero-shot, training-free anomaly classification framework, leveraging a vision-LLM (generates video descriptions) and an NLI classifier (evaluates text entailment) to reframe anomaly classification as a language-grounded reasoning task.
- This framework processes input video through a vision-LLM to generate a descriptive text string, which is then evaluated against predefined anomaly labels using a frozen NLI classifier to determine the most plausible anomaly.
- The modular architecture allows independent updates of the vision-LLM and NLI classifier, enabling adaptability to new anomaly types by simply adding corresponding text labels without model modification.

---

[LOST IN TOKENIZATION: CONTEXT AS THE KEY TO UNLOCKING BIOMOLECULAR UNDERSTANDING IN SCIENTIFIC LLMS](http://arxiv.org/abs/2510.23127)

- CoKE (Context-driven Knowledge Engine): introduces a novel paradigm for Scientific Large Language Models (Sci-LLMs) that leverages high-level structured context derived from bioinformatics tools, bypassing direct interpretation of raw biomolecular sequences, with Raw Biomolecular Sequence Input, Context Generation Function, InterProScan, BLASTp, ProTrek, Context Construction Module, High-Confidence Homology Prioritization, Domain Information Integration, Semantic Evidence Fallback, Natural Language Question Input, Scientific Large Language Model (Sci-LLM), and Natural Language Answer Output, to provide accurate and robust answers to biological reasoning tasks.
- The framework addresses the "tokenization dilemma" by reframing Sci-LLMs as powerful reasoning engines over expert knowledge, demonstrating superior generalization and robustness compared to traditional sequence-centric approaches.
- This approach shifts the developmental focus from direct sequence interpretation towards high-level knowledge synthesis, enabling generalist LLMs to excel in specific bioinformatics tasks without costly retraining.

---

[Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](http://arxiv.org/abs/2510.23038)

- TIR-Judge (Tool-Integrated Reinforcement Learning): introduces an end-to-end RL framework for training LLM judges, integrating an LLM Judge, Code Executor, Sandbox, Reinforcement Learning (RL) Framework, Supervised Fine-Tuning (SFT), Teacher Model (Gemini-2.5-Flash), Rejection Sampling, Reward Design, and Training Data Collection, to enable precise and verifiable evaluations across diverse tasks.
- The framework supports flexible judgment formats (pointwise, pairwise, listwise) and employs iterative RL for self-improvement without distillation, demonstrating strong generalization and parameter efficiency.
- By coupling reasoning with tool execution and optimizing end-to-end, the approach significantly outperforms text-only judges in accuracy and handles complex computational and symbolic reasoning challenges.

---

[P1GPT: A MULTI-AGENT LLM WORKFLOW MODULE FOR MULTI-MODAL FINANCIAL INFORMATION ANALYSIS](http://arxiv.org/abs/2510.23032)

- P1GPT (A Multi-Agent LLM Workflow Module for Multi-Modal Financial Information Analysis): introduces a layered multi-agent LLM framework for multi-modal financial information analysis and interpretable trading decision support, including Input Layer (parses multi-modal data), Planning Layer (decomposes tasks, assigns agents), Analysis Layer (executes domain-specific analysis), Integration Layer (aggregates outputs, synthesizes insights), and Decision Layer (generates investment signals).
- The framework employs a structured reasoning pipeline with specialized Intelligent Specialized Agents (ISAs) and Supporting Agents for fundamental, technical, news, and sectoral analysis, enabling coordinated agent communication and integration-time synthesis.
- P1GPT aims to provide explainable and trustworthy financial AI systems by offering superior cumulative and risk-adjusted returns, low drawdowns, and transparent causal rationales through its modular, layered architecture and standardized reporting.

---

[TALM: Dynamic Tree-Structured Multi-Agent Framework with Long-Term Memory for Scalable Code Generation](http://arxiv.org/abs/2510.23010)

- TALM (Tree-Structured Multi-Agent Framework with Long-Term Memory): introduces a dynamic framework for scalable code generation, integrating a tree-structured collaboration mechanism, a localized re-reasoning process, and a long-term memory module, with Code Agents, a Validation Agent, a Sandbox environment, and a Vector Database, to enhance reasoning flexibility and efficient error correction.
- The framework organizes LLM-based agents in a dynamically growing tree structure, enabling flexible task decomposition and supporting localized re-reasoning through Child-Agent Clarification and Structure-Correction modes.
- TALM incorporates a long-term memory module with Knowledge Retrieval and Memory Update mechanisms, allowing agents to store and reuse past experiences for implicit self-improvement and robust performance in complex code generation tasks.

---

[CodeAD: Synthesize Code of Rules for Log-based Anomaly Detection with LLMs](http://arxiv.org/abs/2510.22986)

- CodeAD (Code synthesis framework of rules for log-based Anomaly Detection): introduces an LLM-powered framework that automatically synthesizes lightweight Python rule functions for log-based anomaly detection, utilizing hierarchical clustering, anchor-grounded sampling, and an agentic workflow for rule generation, testing, and refinement.
- This framework operates in an offline rule synthesis phase to generate interpretable and efficient rules, followed by an online monitoring phase for real-time anomaly detection on streaming logs.
- CodeAD achieves superior F1 scores and efficiency compared to state-of-the-art methods, making it a practical and scalable solution for real-world online monitoring systems.

---

[THE REASONING TRAP: How ENHANCING LLM REASONING AMPLIFIES TOOL HALLUCINATION](http://arxiv.org/abs/2510.22977)

- SIMPLETOOLHALLUBENCH (Simple Tool Hallucination Benchmark): introduces a diagnostic benchmark for measuring tool hallucination in LLM agents, investigating how enhancing reasoning capabilities through RL amplifies this phenomenon.
- The research systematically examines tool hallucination in two failure modes—no tool available and distractor tools present—revealing a causal link between reasoning enhancement and increased hallucination across various training methods.
- It further analyzes mechanistic drivers within LLM internal representations, pinpointing late-layer residual streams as loci of divergence, and evaluates mitigation strategies like prompt engineering and DPO, highlighting a fundamental reliability-capability trade-off.

---

[MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs](http://arxiv.org/abs/2510.22967)

- MAD-Fact (A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs): introduces a systematic approach for long-form factuality evaluation in LLMs, integrating the Clerk Agent, Jury (composed of Evaluator Agents with diverse roles), Judge Agent, Fact Importance Hierarchy Model, Weighted Evaluation Metrics, LongHalluQA Dataset, Factual Knowledge Base, Shared Message Pool, Shared Knowledge Base, Evaluated Model, and Expert Models.
- The framework mitigates single-model biases through a multi-agent debate system and accounts for varying fact importance in long-form texts using a hierarchical model and weighted metrics.
- It provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, supported by a new Chinese long-form factuality dataset, LongHalluQA.

---

[CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents](http://arxiv.org/abs/2510.22963)

- CompressionAttack: introduces a novel attack pipeline that exploits prompt compression modules as a new attack surface in LLM-powered agents, comprising HardCom (targets hard prompt compression), SoftCom (targets soft prompt compression), Token-level editing attack (applies multi-level adversarial edits at token level), Word-level editing attack (applies multi-level adversarial edits at word level), Demo-level editing attack (applies multi-level adversarial edits at demonstration level), Suffix-based embedding attack (appends learnable vectors), and Token-representation editing attack (perturbs discrete input tokens), where it systematically studies this unexplored vulnerability to manipulate downstream LLM behavior.
- The framework achieves high attack success rates and preference flip rates across various LLMs and tasks while maintaining high stealthiness, outperforming existing baselines.
- Case studies on real-world agent frameworks like VSCode Cline and Ollama demonstrate the practical impact of the attack, highlighting the urgent need for more robust defenses against prompt compression vulnerabilities.

---

[MGFRec: Towards Reinforced Reasoning Recommendation with Multiple Groundings and Feedback](http://arxiv.org/abs/2510.22888)

- MGFRec: introduces a novel reinforcement learning framework for reasoning-based recommendation, enabling LLMs to perform multiple groundings in the actual item space and incorporate user agent feedback.
- The framework iteratively refines recommendations by grounding generated item titles to the actual item space and receiving simulated feedback from a user agent, ensuring alignment with real items and user preferences.
- MGFRec leverages GRPO for training, allowing the recommendation agent to learn to explore the item space more effectively and adapt to user interests through a multi-turn process of thinking, grounding, and answering.

---

[Coordinated Autonomous Drones for Human-Centered Fire Evacuation in Partially Observable Urban Environments](http://arxiv.org/abs/2510.23899)

- Multi-agent Coordination Framework: introduces a system where autonomous UAVs, specifically a High-Level Rescuer (HLR) and a Low-Level Rescuer (LLR), coordinate to locate, intercept, and guide human evacuees in a dynamic, partially observable urban fire environment, utilizing a POMDP for modeling and PPO with recurrent policies for decision-making.
- The framework integrates an agent-based model of human panic behavior to account for irrational evacuee actions, enabling robust real-time assistance despite uncertainties like unknown evacuee locations and limited visibility.
- Simulation results demonstrate that the UAV team significantly reduces evacuation time and improves outcomes compared to scenarios without drone assistance, even when evacuees exhibit panic-induced behavior.

---

[AGENTIC AI SECURITY: THREATS, DEFENSES, EVALUATION, AND OPEN CHALLENGES](http://arxiv.org/abs/2510.23883)

- Agentic AI System: introduces a comprehensive survey of security threats, defenses, and evaluation methodologies for LLM-powered autonomous agents, detailing their core intelligence/reasoning, goal-directed reasoning/task execution, external environment interaction, persistent context/learning, independent goal pursuit/action, infrastructure/reasoning chain, inter-agent coordination, digital/physical interaction space, human-agent interaction point, inter-agent/resource communication, and threat mitigation/security controls components.
- The paper categorizes threats into prompt injection, autonomous cyber-exploitation, multi-agent/protocol-level threats, interface/environment risks, and governance concerns, illustrating each with examples.
- It also reviews defense strategies like sandboxing and policy enforcement, alongside benchmarks for continuous evaluation, to support secure-by-design agent development.

---

[Decentralized Multi-Agent Goal Assignment for Path Planning using Large Language Models](http://arxiv.org/abs/2510.23824)

- Decentralized LLM-based Goal Assignment Protocol: introduces a method for decentralized goal assignment in multi-agent path planning, where LLM-based agents generate ranked goal preferences based on a Structured Environment Representation, guided by Prompt Engineering, followed by Goal Ranking Exchange and Conflict Resolution Rule to determine assignments.
- The protocol enables agents to independently reason about goal preferences and potential conflicts, aiming to minimize makespan in fully observable grid-world environments without negotiation or iterative coordination.
- The study demonstrates that LLM-based agents, particularly GPT-4.1 with structured prompts and quantitative information, achieve near-optimal makespans, outperforming traditional heuristics and highlighting the importance of input structure for effective decentralized assignment.

---

[TDFlow: Agentic Workflows for Test Driven Software Engineering](http://arxiv.org/abs/2510.23761)

- TDFlow: introduces a test-driven agentic workflow for repository-scale software engineering, with Generate Tests sub-agent (LLM-based test generation), Run Specified Tests component (executes provided tests), Evaluate Tests component (assesses test outcomes), Identify Tests component (filters failing tests), Explore Files sub-agent (repository analysis, patch proposal), Submit Patch component (applies proposed patch), Revise Patch sub-agent (corrects malformed patches), Debug One sub-agent (individual test debugging, report generation), Aggregate Reports + Rewrite Context component (combines reports, updates context), Debugger tool (debugging utility), Repository (codebase environment), Tests (input/output for evaluation), Patch (proposed code changes), Reports (debugging outcomes), and Context (information for agents), designed to solve human-written tests by repeatedly proposing, revising, and debugging repository-scale patches.
- This agentic workflow decomposes software engineering program repair into four LLM-governed components, reducing long-context burden and enabling specialized performance improvement on specific sub-tasks.
- The framework achieves 88.8% pass rate on SWE-Bench Lite and 94.3% on SWE-Bench Verified with human-written tests, demonstrating human-level test resolution capabilities.

---

[Agent-based Automated Claim Matching with Instruction-following LLMs](http://arxiv.org/abs/2510.23924)

- LLM Agent-based Pipeline for Claim Matching: introduces an agent-based approach for automated claim matching, utilizing a Prompt Generation Agent (LLM) to create task-specific prompts, which are then used by a Claim Matching Agent (LLM) for binary classification, supported by user prompts, few-shot examples, and the ClaimMatch Dataset.
- This two-step pipeline demonstrates that LLM-generated prompts can outperform human-crafted prompts and that smaller LLMs can effectively generate prompts for other LLMs.
- The approach investigates automated prompt engineering methods for claim matching, revealing insights into LLMs' understanding of the task and improving state-of-the-art results.

---

[LARGE LANGUAGE MODEL AGENT PERSONALITY AND RESPONSE APPROPRIATENESS: EVALUATION BY HUMAN LINGUISTIC EXPERTS, LLM-AS-JUDGE, AND NATURAL LANGUAGE PROCESSING MODEL](http://arxiv.org/abs/2510.23875)

- Large Language Model Agent Personality and Response Appropriateness: Evaluation by Human Linguistic Experts, LLM-as-Judge, and Natural Language Processing Model introduces a novel interdisciplinary approach to assess LLM-based agent personalities, combining agent development with linguistic analysis using a novel question bank and three distinct evaluation methods.
- The framework develops two poetry expert agents, Introvert and Extrovert, using Langchain, RAG, and prompt engineering, then evaluates their responses with a Personality Transformer Model, a Judge LLM, and human linguistic experts.
- Findings reveal limitations in purely deep learning solutions and LLM-as-judge methods due to biases, emphasizing the critical role of human linguistic experts and interdisciplinary design for accurate personality assessment.

---

[OraPlan-SQL: A Planning-Centric Framework for Complex Bilingual NL2SQL Reasoning](http://arxiv.org/abs/2510.23870)

- OraPlan-SQL: introduces a planning-centric framework for complex bilingual NL2SQL reasoning, which includes a Planner Agent (generates natural language plans), Database Schema Semantic Retrieval (retrieves relevant schema elements), Iterative Planning Prompt Refinement (optimizes planner prompt), Meta-prompting with feedbacks (derives corrective guidelines), Diverse Plans Generation (creates multiple candidate plans), Entity Linking (resolves entity mismatches), SQL Agent (converts plans to SQL), In Context Examples Retrieval (provides similar examples), SQL Generation (produces executable SQL), and Majority Voting (selects final output).
- The framework decomposes Text-to-SQL mapping into a natural language plan generation stage and an executable SQL query generation stage, enhancing transparency and accuracy.
- It achieves state-of-the-art bilingual performance by integrating feedback-guided meta-prompting, entity-linking guidelines, and plan diversification with consensus execution.

---

[Can LLMs Narrate Tabular Data? An Evaluation Framework for Natural Language Representations of Text-to-SQL System Outputs](http://arxiv.org/abs/2510.23854)

- Combo-Eval: introduces a novel evaluation framework for Natural Language Representations (NLRs) of Text-to-SQL system outputs, combining Metrics-as-a-Judge (initial evaluation), Extremities Threshold Selection (filters samples), LLM-as-a-judge (finer diagnostic), and Final Evaluation (combines judgments).
- This framework enhances evaluation fidelity and significantly reduces LLM calls by first applying metrics-based thresholds to filter samples before engaging LLMs for a finer diagnostic.
- The paper also introduces NLR-BIRD, the first dedicated dataset for NLR benchmarking, and provides comprehensive benchmarking of LLMs for NLR generation and judging across various scenarios.

---

[Temporal Blindness in Multi-Turn LLM Agents: Misaligned Tool Use vs. Human Time Perception](http://arxiv.org/abs/2510.23853)

- TicToc-v1 (Time-aware conversational Tool-calling v1): introduces a diverse test set for evaluating LLM agents' temporal awareness in multi-turn tool-use decisions, including 34 scenarios, 700+ multi-turn user-agent message trajectories, explicit timestamps, and human preferences for tool-calling decisions.
- The framework systematically investigates temporal blindness in LLM agents, where models fail to account for real-world time elapsed between messages, leading to misaligned tool-use decisions compared to human perception.
- TicToc-v1's design allows for measuring how temporal information, including timestamp augmentation and prompt-based alignment, influences LLM function-calling decisions and their alignment with human expectations.

---

[RECAP: Recursive Context-Aware Reasoning and Planning for Large Language Model Agents](http://arxiv.org/abs/2510.23822)

- ReCAP (Recursive Context-Aware Reasoning and Planning): introduces a hierarchical framework with shared context for LLM agents, combining plan-ahead task decomposition, consistent multi-level context with structured injection, and memory-efficient execution to improve long-horizon reasoning and planning.
- The framework enables LLMs to generate a full subtask list, execute the first item, and refine the remainder, while re-injecting parent plans into a single sliding context window to maintain high-level intent and coherence.
- ReCAP significantly improves subgoal alignment and success rates on long-horizon reasoning benchmarks like Robotouille by reducing context drift, preventing recurrent failure cycles, and adapting plans dynamically.

---

[Evaluating Long-Term Memory for Long-Context Question Answering](http://arxiv.org/abs/2510.23730)

- Memory Augmentation Strategies: introduces a systematic evaluation of memory-augmented methods, including Full Context Prompting, RAG, A-Mem, PromptOpt, and EpMem, for long-context question answering tasks.
- The study analyzes how these diverse memory architectures impact LLM performance and token usage across various reasoning categories.
- Findings indicate that memory-augmented approaches significantly reduce token usage while maintaining competitive accuracy, with specific memory types benefiting different LLM capabilities.

---

[QueryIPI: Query-agnostic Indirect Prompt Injection on Coding Agents](http://arxiv.org/abs/2510.23675)

- QueryIPI (Query-agnostic Indirect Prompt Injection): introduces a novel query-agnostic indirect prompt injection method for LLM coding agents, leveraging the agent's internal prompt in an iterative optimization procedure with a Mutation LLM and a Judge LLM to systematically craft malicious tool descriptions.
- The framework transforms the attack from a black-box search into a white-box optimization problem by exploiting the leakage of the agent's internal prompt, enabling targeted payload generation.
- QueryIPI demonstrates high efficacy in simulated environments and successfully transfers to real-world coding agents, highlighting the security risk of exposed internal prompts.

---

[MCPGuard : Automatically Detecting Vulnerabilities in MCP Servers](http://arxiv.org/abs/2510.23673)

- MCPGuard (Model Context Protocol Guard): introduces a layered, modular defense architecture for automatically detecting vulnerabilities in Model Context Protocol (MCP) servers, including lightweight static scanning, a deep neural detection module, and an intelligent arbitration mechanism.
- This framework proactively identifies vulnerabilities by combining pattern-based detection, semantic analysis using fine-tuned neural models, and LLM-based judgment for comprehensive input safety assessment.
- The framework balances efficiency and accuracy through its hybrid decision-making system, ensuring robust protection against various MCP-specific threats.

---

[Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents](http://arxiv.org/abs/2510.23691)

- Game-TARS (Generalist Game Agent): introduces a generalist game agent trained with a unified, scalable action space anchored to human-aligned native keyboard-mouse inputs, utilizing a comprehensive training process that includes continual pre-training and post-training stages on a vision-language model, incorporating components like Perception, Memory, Sparse ReAct, and Action, and evaluated across various sandboxes.
- The framework leverages a two-tiered memory mechanism, including Short-Term Contextual Memory and Long-Term Summary Memory, to balance high-fidelity recent memory with highly compressed long-term memory for complex tasks.
- Game-TARS employs a Native Sparse ReAct paradigm during pre-training, which interleave reasoning and action only at critical decision points, and refines thought-action chains via rejection sampling and LLM rewriting for efficient decision-making.

---

[BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](http://arxiv.org/abs/2509.15566)

- BTL (Blink-Think-Link): introduces a brain-inspired framework for human-GUI interaction, decomposing interactions into Blink (rapid screen attention), Think (high-level reasoning), and Link (executable command generation) phases.
- The framework incorporates Blink Data Generation for automated ROI annotation and BTL Reward, a process-outcome integrated mechanism, to guide reinforcement learning.
- BTL-UI, a GUI agent built upon this framework, demonstrates competitive performance in GUI understanding and dynamic interaction tasks through Policy Optimization using Group Relative Policy Optimization (GRPO).

---

#### 26th October 2025

[RL-AVIST: Reinforcement Learning for Autonomous Visual Inspection of Space Targets](http://arxiv.org/abs/2510.22699)

- RL-AVIST (Reinforcement Learning for Autonomous Visual Inspection of Space Targets): introduces a reinforcement learning framework for autonomous visual inspection of space assets, leveraging the SRB simulation platform and DreamerV3 for efficient trajectory planning and control.
- The framework trains agents using model-based RL (DreamerV3) and compares it against model-free baselines (PPO, TD3) to achieve adaptability, generalization, and precision in 6-DOF spacecraft dynamics for tasks around targets like the Lunar Gateway, Venus Express, and ISS.
- RL-AVIST demonstrates superior sample efficiency and performance in tracking diverse inspection trajectories, paving the way for scalable and retrainable control solutions for future space operations.

---

[CURRICULUM-BASED ITERATIVE SELF-PLAY FOR SCALABLE MULTI-DRONE RACING](http://arxiv.org/abs/2510.22570)

- CRUISE (Curriculum-Based Iterative Self-Play for Scalable Multi-Drone Racing): introduces a reinforcement learning framework that integrates a progressive difficulty curriculum and an efficient iterative self-play mechanism to train robust competitive behaviors for multi-drone racing.
- The framework utilizes a five-stage curriculum to systematically increase task difficulty, starting with basic navigation and progressing to high-speed, collision-aware racing against dynamic opponents.
- CRUISE policies significantly outperform standard reinforcement learning baselines and game-theoretic planners in terms of racing speed, success rate, and collision avoidance in high-fidelity simulations.

---

[SPIRAL: SELF-PLAY INCREMENTAL RACING ALGORITHM FOR LEARNING IN MULTI-DRONE COMPETITIONS](http://arxiv.org/abs/2510.22568)

- SPIRAL (Self-Play Incremental Racing Algorithm for Learning): introduces a novel approach for training autonomous drones in multi-agent racing competitions, with Stage 1 Single Drone Training (fundamental flight control), Stage 2 1v1 Self-Play Racing (competitive strategy development), Stage 3 2v2 Team Racing (multi-agent coordination/competition), Self-Play Training Loop (continuous improvement mechanism), Race Against Previous Best (agent competes against past policy), Update Policy (refines agent's policy), Save if Improved (stores better performing policy), Evaluate Performance (assesses current policy), and Load (imports opponent model), where the paper employs a self-play mechanism to incrementally cultivate complex racing behaviors in a dynamic environment.
- This framework guides agents from mastering fundamental flight control to executing sophisticated cooperative multi-drone racing strategies through a progressive learning journey.
- The self-play dynamic enables continuous improvement by having drones compete against increasingly proficient versions of themselves, naturally escalating competitive difficulty and fostering robust, adaptive racing strategies.

---

[On Generalization in Agentic Tool Calling: CoreThink Agentic Reasoner and MAVEN Dataset](http://arxiv.org/abs/2510.22898)

- CoreThink Agentic Reasoner: introduces a framework that augments LLMs with a NeuroSymbolic reasoning layer for structured decomposition and adaptive tool orchestration, achieving state-of-the-art performance on tool-calling benchmarks.
- The framework processes conversational input through Context Buffering, Action Synthesis, and Invocation Generation stages to reliably decompose multi-step tasks and verify intermediate results.
- It demonstrates robust generalization across diverse domains and challenging out-of-distribution tasks, offering significant performance improvements and computational efficiency without additional training.

---

[Collaborative LLM Agents for C4 Software Architecture Design Automation](http://arxiv.org/abs/2510.22787)

- MASC4 (Multi-Agent System for C4): introduces an LLM-based multi-agent system that automates C4 software architecture model generation and evaluation, including collaborative analysis agents, specialized processing agents, and a hybrid evaluation framework.
- The system simulates role-specific experts to analyze requirements and generate Context, Container, and Component views of the C4 model, producing textual reports, YAML structures, and PlantUML diagrams.
- The evaluation framework assesses structural integrity, C4 rule adherence, and semantic quality using deterministic checks and an LLM-as-a-Judge approach.

---

[ATLAS: Actor-Critic Task-Completion with Look-ahead Action Simulation](http://arxiv.org/abs/2510.22732)

- ATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation): introduces a memory-augmented web agent that plans via look-ahead simulation and structured memory retrieval, featuring a Planner, Actor, Critic, Multi-layered Memory, Observation Abstractor, Browser Executor, Look-ahead Action Simulation (LAS), Curiosity-Driven Exploration, and Memory Agent, to achieve robust task completion in dynamic web environments without LLM fine-tuning.
- The system employs a hierarchical memory structure, including Working Memory for recent context, a Cognitive Map for state transitions, and Semantic Memory for environment-specific constraints, all built through curiosity-driven exploration and agentic summarization.
- The modular architecture integrates planning, memory, and simulation to transform high-level instructions into safe, executable action sequences for long-horizon web tasks, demonstrating improved success rates on the WebArena-Lite Benchmark.

---

[SwiftSolve: A Self-Iterative, Complexity-Aware Multi-Agent Framework for Competitive Programming](http://arxiv.org/abs/2510.22626)

- SwiftSolve: introduces a complexity-aware multi-agent system for competitive programming that couples algorithmic planning, empirical profiling, and complexity-guided repair, including a Planner Agent (creates algorithmic sketch), Static Pruner (filters high-risk plans), Coder Agent (generates ISO C++17 code), Profiler Agent (compiles, executes, records metrics), Complexity Analyst Agent (infers complexity, dispatches patches), and Controller (orchestrates agents, enforces safety).
- The framework operates as a self-iterative pipeline where specialized LLM-based agents communicate via typed JSON messages to generate correct, complete, and efficient code, addressing runtime and memory constraints.
- SwiftSolve improves code generation by integrating asymptotic performance considerations and iterative feedback loops, allowing for replanning and targeted patches to optimize for efficiency beyond mere correctness.

---

[Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents](http://arxiv.org/abs/2510.22620)

- Threat Snapshots: introduces a framework that isolates specific states in an AI agent's execution flow where LLM vulnerabilities manifest, enabling systematic identification and categorization of security risks propagating from the LLM to the agent level.
- The framework comprises Agent State (attacked agent details) and Threat Description (attacker objective, delivery), which are used to construct the b³ benchmark for evaluating the security of 31 popular backbone LLMs against 194,331 crowdsourced adversarial attacks.
- The b³ benchmark reveals that enhanced reasoning capabilities improve LLM security, while model size does not correlate with security, providing actionable insights for agent developers and LLM providers.

---

[A Closed-Loop Personalized Learning Agent Integrating Neural Cognitive Diagnosis, Bounded-Ability Adaptive Testing, and LLM-Driven Feedback](http://arxiv.org/abs/2510.22559)

- EduLoop-Agent: introduces a closed-loop personalized learning agent, integrating NCD (fine-grained mastery estimates), BECAT (dynamically selects items), and LLMs (converts diagnostic signals into feedback), to provide individualized learning trajectories.
- This framework operationalizes a "Diagnosis–Recommendation-Feedback" cycle, addressing limitations of isolated modeling, item selection, and feedback generation.
- The system leverages NCD for interpretable mastery assessments, BECAT for adaptive item recommendations, and LLMs for targeted, actionable study guidance.

---

[Finding the Needle in the Crash Stack: Industrial-Scale Crash Root Cause Localization with AutoCrashFL](http://arxiv.org/abs/2510.22530)

- AUTOCRASHFL: introduces an LLM agent for industrial-scale crash root cause localization, utilizing a crashdump parser and repository interface with specific tools to identify suspicious files.
- The framework processes crashdumps and source code, enabling the LLM to autonomously retrieve relevant information through tools like `get_crash_extinfo`, `get_crash_stack`, `get_nearby_code`, and `get_term_definition`.
- It employs a ranking aggregation mechanism to stabilize and improve fault localization performance by combining results from multiple LLM runs, ultimately providing a ranked list of suspicious files and an explanation of the bug.

---

[Learning "Partner-Aware” Collaborators in Multi-Party Collaboration](http://arxiv.org/abs/2510.22462)

- ICR (Interruptible Collaborative Roleplayer): introduces a novel learning algorithm designed to train "partner-aware" LLM-driven collaborator agents that increase common ground alignment on task-relevant propositions by intelligently collecting information from interventions.
- The framework employs a two-player Modified-Action MDP and counterfactual invariance-based KL divergence regularization, utilizing a counterfactual state and policy to ensure robust reasoning despite potentially misleading interventions.
- ICR-trained agents consistently outperform baselines in task performance and common ground convergence across various collaborative tasks and communication settings, demonstrating their ability to distinguish helpful from misleading input.

---

[Affordance Representation and Recognition for Autonomous Agents](http://arxiv.org/abs/2510.24459)

- Pattern Language for Structured Perception: introduces a framework for autonomous agents to construct and maintain actionable internal world models from structured data, utilizing Environment, Affordance Sources, Agent Perception (DOM Transduction, Hypermedia Affordances Recognition), and Cognitive Map.
- The framework employs the DOM Transduction Pattern to distill verbose web page DOMs into compact Page Affordance Models, and the Hypermedia Affordances Recognition Pattern to dynamically discover and integrate service capabilities from Semantic Sources via an Affordance Parser.
- These complementary patterns enable agents to efficiently process complex digital environments, adapt to evolving services, and build a unified, semantically rich Cognitive Map for robust and scalable automation.

---

[AGENTSWAY SOFTWARE DEVELOPMENT METHODOLOGY FOR AI AGENTS-BASED TEAMS](http://arxiv.org/abs/2510.23664)

- Agentsway: introduces a novel software development methodology for AI agent-based teams, featuring a Human Orchestrator (interprets goals, supervises), Planning Agent (decomposes requirements, plans), Prompting Agent (generates code prompts), Coding Agents (translates prompts to code), Testing Agents (ensures software quality), Fine-Tuning Agents (refines LLMs, learns), a Fine-tuned LLM Consortium (diverse LLMs for reasoning), a Reasoning LLM (synthesizes, validates decisions), and an Agents-SDK (orchestrates LLM-based agents).
- This framework redefines traditional team structures by positioning humans as orchestrators while delegating specialized functions to autonomous AI agents, ensuring a cyclical, privacy-preserving, and self-improving lifecycle.
- The methodology integrates fine-tuned LLMs and advanced reasoning models to enhance domain-specific reasoning, adaptive learning, and explainable decision-making, embedding responsible AI principles throughout the development process.

---

#### 25th October 2025

[Reinforcement learning-guided optimization of critical current in high-temperature superconductors](http://arxiv.org/abs/2510.22424)

- RL-TDGL Framework: introduces an integrated workflow for optimizing critical current density (Jc) in high-temperature superconductors, featuring an Environment Module (evaluates Jc) with a TDGL simulation (generates I-V characteristics) and a Neural network Jc predictor (surrogate ML model), and an Agent Module (updates defect configurations) using Proximal Policy Optimization (RL algorithm) with an Actor Network (proposes defect actions) and a Critic Network (estimates state value).
- This framework autonomously identifies optimal defect configurations, including densities and spatial correlations, to enhance vortex pinning and Jc, achieving up to a 15-fold enhancement compared to random initialization.
- The surrogate ML model significantly accelerates the RL process by predicting Jc from defect configurations, reducing the need for computationally expensive full TDGL simulations at every iteration.

---

[PORTGPT: Towards Automated Backporting Using Large Language Models](http://arxiv.org/abs/2510.22396)

- PORTGPT (Towards Automated Backporting Using Large Language Models): introduces an LLM-agent for end-to-end patch backporting, simulating human-like reasoning and verification by enhancing an LLM with tools for code access, Git history summarization, and autonomous patch revision based on feedback.
- The framework operates in two stages: Per-Hunk Adaptation for localization and initial transformation, and Final Patch Combination for integrating and validating the complete patch.
- PORTGPT leverages tools like ViewCode, LocateSymbol, GitUtils, ApplyHunk, and CompileTest to gather information, apply changes, and iteratively refine patches, achieving superior performance over existing backporting tools.

---

[BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles](http://arxiv.org/abs/2510.22370)

- BLIP-FusePPO (Bootstrapped Language-Image Pretraining-driven Fused State Representation in Proximal Policy Optimization): introduces a novel multimodal deep reinforcement learning framework for autonomous lane-keeping, integrating Camera (RGB visual input), LiDAR (spatial range data), PID Controller (control feedback signals), and VLM-Blip (semantic embeddings generation) into a hybrid state representation.
- The framework processes these diverse inputs through dedicated neural branches, including Image CNN, LiDAR FC Layers, PID FC Layer, and VLM-Blip components (Q-Former, LLM Decoder, Embedding Model), before concatenating them into a Fused State Vector for the PPO Algorithm.
- This approach enhances policy learning robustness and generalization by providing semantic awareness and control-aware representations, leading to improved lane-keeping stability and adaptability in diverse driving scenarios.

---

[Group size effects and collective misalignment in LLM multi-agent systems](http://arxiv.org/abs/2510.22422)

- LLM Multi-Agent Naming Game Framework: introduces a system where LLM agents engage in a naming game, utilizing memory states, probabilistic policies, and text prompts (composed of system prompts and user queries) to study collective dynamics, complemented by a mean-field analytical approach.
- The research demonstrates that collective bias in LLM multi-agent systems is a complex phenomenon, influenced non-linearly by population size, capable of amplifying, inducing, or reversing individual biases, and revealing model-dependent dynamical regimes.
- Findings show that beyond a critical population size, simulations converge to deterministic predictions, with the mean-field theory clarifying the basins of attraction for competing equilibria, highlighting population-level effects as a key driver of LLM system behavior.

---

[FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation](http://arxiv.org/abs/2510.22344)

- FAIR-RAG (Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation): introduces an agentic framework that transforms the standard RAG pipeline into a dynamic, evidence-driven reasoning process, including an Adaptive Routing agent, Generator LLMs, an Adaptive Query Generation agent, a Hybrid Retriever, an Evidence Filtering agent, and a Structured Evidence Assessment (SEA) agent.
- The framework systematically identifies and fills evidence gaps through an iterative refinement cycle governed by the SEA module, which performs a checklist-based gap analysis.
- This approach ensures a comprehensive and verified knowledge context for final answer generation, significantly enhancing trustworthiness and reducing LLM hallucination for complex, multi-hop queries.

---

[CGoT: A Novel Inference Mechanism for Embodied Multi-Agent Systems Using Composable Graphs of Thoughts](http://arxiv.org/abs/2510.22235)

- CGoT (Composable Graphs of Thoughts): introduces a novel inference mechanism for embodied multi-agent systems, leveraging LLMs to enable dynamic agent combination/splitting and graph-based inference across Inference, Conclude, and Execution phases, supported by Sensor Fusion and an Emergency Module.
- This framework allows ego-vehicles to transport service robots, forming combined agents that share capabilities and reduce token consumption for enhanced operational efficiency in dynamic environments.
- CGoT demonstrates comparable planning performance to traditional methods while significantly reducing LLM token consumption through its dynamic agent interaction and graph-based inference.

---

[Embracing Trustworthy Brain-Agent Collaboration as Paradigm Extension for Intelligent Assistive Technologies](http://arxiv.org/abs/2510.22095)

- BAC (Brain-Agent Collaboration): introduces a paradigm extension from BCI to BAC, reframing agents as active, collaborative partners for intelligent assistance, focusing on ethical data handling, model reliability, and a robust human-agent collaboration framework to ensure safe, trustworthy, and effective systems.
- The proposed framework details core mechanisms for human users (accessibility, information/feedback, supervisory control) and agents (personalization, low-latency interaction, ethical safeguards), alongside a systematic structure for collaboration architecture, data infrastructure, LLM-based model development, and continuous monitoring.
- A comprehensive evaluation protocol is also presented, encompassing dimensions like technical performance, cognitive synergy, interaction quality, user agency, and ethics, supported by empirical validation methods and specific metrics to holistically assess human-agent partnership.

---

[IFS: Information Flow Structure for Multi-agent Ad Hoc System](http://arxiv.org/abs/2510.22320)

- IFS (Information Flow Structure for Multi-agent Ad Hoc System): introduces a framework to enhance information flow and processing capacity in multi-agent ad hoc systems, featuring a Communication Protocol for Controlled Agents (CPCA) (governs communication), Communication Module (CM) (handles communication), Information Fusion Module (IFM) (fuses observations), Information Separation Module (ISM) (separates information), Attention Mechanism (AM) (fuses atomic data), Local Network (computes action-values), Global Network (evaluates joint actions), Buffer (stores experience), and Environment (simulates interactions).
- The framework enhances collaborative capabilities by strengthening information flow through communication and information fusion, enabling better adaptation to open-system environments and more accurate inference of other agents' intentions.
- IFS specifically tackles challenges in N-agent ad hoc teamwork by allowing controlled agents to coordinate effectively with both known and unknown teammates without relying on pre-coordinated strategies.

---

[Balancing Specialization and Centralization: A Multi-Agent Reinforcement Learning Benchmark for Sequential Industrial Control](http://arxiv.org/abs/2510.20408)

- Industrial Control MARL Benchmark: introduces an enhanced industry-inspired benchmark environment for sequential industrial control, integrating a sequential recycling scenario with sorting and pressing operations, designed to evaluate modular versus monolithic multi-agent RL control strategies and the impact of action masking.
- The benchmark environment combines tasks from existing SortingEnv and ContainerGym, featuring components like input belts, sorting machines, containers, presses, and bale storage, with distinct reward functions for sorting purity and pressing efficiency.
- Experiments demonstrate that action masking significantly improves RL agent performance, narrowing the gap between modular and monolithic architectures, though a rule-based heuristic consistently outperforms all learning-based strategies in this structured industrial setting.

---

[A Multi-agent Large Language Model Framework to Automatically Assess Performance of a Clinical AI Triage Tool](http://arxiv.org/abs/2510.26498)

- A Multi-agent Large Language Model Framework: introduces RADAR (Real-time AI Data Assessment and Reporting), an automated performance engine that evaluates a commercial ICH detector by processing radiology reports with an ensemble of nine LLM agents and a consensus mechanism.
- This framework utilizes DICOM images as input for the commercial ICH detector, transmits results via HL7 messages to a MySQL database via a MIRTH receiver, and extracts radiology report impressions using the Powerscribe API and a report parser.
- The ensemble of LLM agents, including Llama, CodeLlama, Granite, DeepSeek, and GPT-4o, analyzes the impression text to identify intracranial hemorrhage, with a consensus mechanism providing a more reliable ground truth for performance assessment.

---

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

[Compositional Bias Control in Large Language Models: Preference Learning Fails, Supervision Succeeds](http://arxiv.org/abs/2510.22084)

- Compositional Bias Control Framework: introduces a comparative analysis of six bias control techniques for Large Language Models (LLMs) on a compositional constraint task, evaluating their efficacy in mitigating gender stereotypes while maintaining fluency and lexical diversity.
- The study reveals that Supervised Fine-Tuning (SFT) achieves near-perfect constraint compliance and high diversity, while preference-based learning methods like Direct Preference Optimization (DPO) catastrophically fail to satisfy compositional constraints.
- The findings underscore that explicit positive supervision is necessary for mitigating compositional biases, as preference-based alignment struggles to generalize logical structures, highlighting the limitations of preference learning for fair and fluent controlled generation.

---

[VLM-SlideEval: Evaluating VLMs on Structured Comprehension and Perturbation Sensitivity in PPT](http://arxiv.org/abs/2510.22045)

- VLM-SlideEval: introduces an evaluation framework for Vision-Language Models (VLMs) on presentation slides, assessing element extraction, perturbation robustness, and narrative comprehension using its Data Curation, Ground Truth Extraction Pipeline, Schema Standardization, VLM Interaction Module, Ground Truth Matching Module, Perturbation Synthesis Module, Narrative Ordering Module, and Evaluation Metrics Module.
- The framework utilizes a curated dataset of PowerPoint decks, extracts ground truth, applies controlled perturbations, and matches VLM predictions to ground truth using Hungarian alignment.
- It provides verifiable signals at pixel, element, and deck levels, highlighting VLM limitations in pixel-accurate style and cross-slide narrative coherence.

---

[FeaGPT: an End-to-End agentic-AI for Finite Element Analysis](http://arxiv.org/abs/2510.21993)

- FeaGPT (an End-to-End agentic-AI for Finite Element Analysis): introduces a framework that automates complete geometry-mesh-simulation workflows through conversational interfaces, integrating a Human user, a Chief Engineer (Intelligent Analysis Agent), a Knowledge Base, Geometry Generation, Mesh Generation, FEA Simulation, and Data Analysis modules within a GMSA Pipeline.
- The system transforms natural language engineering specifications into validated computational results by interpreting intent, generating physics-aware adaptive meshes, configuring FEA simulations with boundary condition inference, and performing multi-objective analysis through closed-loop iteration.
- FeaGPT leverages knowledge-augmented LLMs for analysis planning and task orchestration, enabling scalable batch processing for parametric studies and democratizing access to advanced computational engineering tools while preserving analytical rigor.

---

[A Comparison of Conversational Models and Humans in Answering Technical Questions: the Firefox Case](http://arxiv.org/abs/2510.21933)

- RAG (Retrieval-Augmented Generation): introduces a method to enhance LLM effectiveness by integrating retrieval mechanisms to dynamically fetch relevant information from curated repositories, grounding responses in accurate, context-sensitive data.
- The study empirically compares RAG-assisted LLM responses with standard GPT and human answers for technical questions within the Mozilla Firefox project.
- Evaluation by Mozilla experts assessed responses based on helpfulness, comprehensiveness, and conciseness, revealing RAG's potential to enhance developer assistance, particularly in comprehensiveness, while noting verbosity as a challenge.

---

[Doc-Researcher: A Unified System for Multimodal Document Parsing and Deep Research](http://arxiv.org/abs/2510.21603)

- Doc-RESEARCHER: introduces a unified system for multimodal document parsing and deep research, integrating deep multimodal parsing, systematic retrieval architecture, and iterative multi-agent workflows to answer complex queries across diverse document types.
- The system processes multimodal documents by preserving layout structure and visual semantics, creating multi-granular representations for adaptive retrieval, and employing agents for planning, searching, refining, and synthesizing evidence.
- It addresses limitations of existing LLM-based research systems by handling multimodal documents, supporting iterative multi-step research, and enabling dynamic granularity selection for evidence extraction.

---

[Co-Sight: Enhancing LLM-Based Agents via Conflict-Aware Meta-Verification and Trustworthy Reasoning with Structured Facts](http://arxiv.org/abs/2510.21557)

- Co-Sight: introduces a closed-loop cognitive architecture that enhances LLM-based agents by integrating Conflict-Aware Meta-Verification (CAMV) and Trustworthy Reasoning with Structured Facts (TRSF), which together reformulate reasoning into a falsifiable and auditable process.
- The framework employs multiple Expert Agents and a Meta-Verification Agent, each equipped with a planner, actor, and a shared facts module, to achieve scalable, transparent, and trustworthy long-horizon reasoning.
- CAMV optimizes verification by focusing computational resources on disagreement hotspots among expert agents, while TRSF continuously organizes, validates, and synchronizes evidence through a structured facts module, ensuring all reasoning is grounded in consistent, source-verified information.

---

[EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law](http://arxiv.org/abs/2510.21524)

- EU-Agent-Bench: introduces a verifiable human-curated benchmark for measuring the intrinsic propensity of LLM agents to violate EU law, utilizing LLM agents, legal categories, user prompts, tools, a rubric, a system prompt, and injected regulations.
- The benchmark evaluates LLM agents' function calls against an EU-legislation-based rubric across 600 augmented test samples spanning six legal domains, including data protection and copyright.
- Experiments with frontier LLMs reveal concerning legality rates, with the best model complying in only 55% of runs, and show that including legislative excerpts in the system prompt has a negligible effect on compliance.

---

[SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots](http://arxiv.org/abs/2510.21459)

- SBASH (System-Based Attention Shell Honeypot): introduces a framework for designing and evaluating LLM honeypots, with System Declaration (centralized parameter control), Configuration System (adjusts system variables), Knowledge base generation (loads RAG documents), Directory generation (populates filesystem template), LLM system prompting (directs LLM understanding), Honeypot (simulated shell environment), and Interaction Logging, Command Analysis, Threat Intelligence (collects attack data), where it manages data-protection issues using lightweight local LLMs and evaluates RAG vs. prompt-tuned LLMs for Linux shell commands.
- The framework enhances honeypot realism and dynamism by directing local LLMs with system type parameters, ensuring context-aware and accurate responses while mitigating challenges like hallucination and slow response times.
- It provides a methodological approach to address issues in traditional and LLM honeypots, including inaccurate responses, lack of state management, response delays, and high computational resource demands.

---

[Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation](http://arxiv.org/abs/2510.21341)

- Magellan (Making Autoregressive Generators Explore via Latent Landscape-Aware Navigation): introduces a novel framework that reframes LLM generation as a principled, guided exploration of an LLM's latent conceptual space, employing MCTS governed by a hierarchical guidance system, semantic compass (vtarget), landscape-aware value function (V(snew)), knowledge corpus (Dnovelty), conceptual clusters, theme synthesizer, guidance vector formulator, MCTS tree, nodes, coherence evaluator (Vcoh), novelty evaluator (Vnov), progress evaluator (Vprog), selection mechanism, expansion and pruning mechanism, backpropagation mechanism, termination condition mechanism, and final concept extractor, to generate innovative scientific ideas with superior plausibility and innovation.
- The framework integrates Monte Carlo Tree Search with a hierarchical guidance system, featuring a semantic compass for global goal-setting and a multi-objective value function for tactical, landscape-aware decision-making, to overcome limitations of unprincipled self-evaluation in prior search methods.
- Magellan significantly outperforms strong baselines like ReAct and Tree of Thoughts by providing a principled, explicit evaluation mechanism that balances intrinsic coherence, extrinsic novelty, and narrative progress, enabling LLMs to become more capable partners in creative discovery.

---

[CXRAgent: Director-Orchestrated Multi-Stage Reasoning for Chest X-Ray Interpretation](http://arxiv.org/abs/2510.21324)

- CXRAgent: introduces a director-orchestrated, multi-stage agent for chest X-ray interpretation, with a central Director coordinating User Query Input, Tool Invocation, Diagnostic Planning, and Collaborative Decision-making to produce an Answer.
- The framework integrates specialized Tools, an Evidence-driven Validator (EDV) for reliability, and a flexible Diagnostic Planning stage that assembles expert Agents for collaborative reasoning.
- CXRAgent leverages a multi-modal LLM as its core Director to coordinate tool invocation, output validation, diagnostic planning, and team-integrated collaborative decision-making, ensuring evidence-backed and adaptive CXR interpretation.

---

[PARL: Prompt-based Agents for Reinforcement Learning](http://arxiv.org/abs/2510.21306)

- PARL (Prompt-based Agent for Reinforcement Learning): introduces a method that uses LLMs as RL agents through prompting, without fine-tuning, by encoding states, actions, and rewards into a cumulative prompt for in-context learning.
- The framework evaluates PARL on three standard RL tasks (Blackjack, Frozen Lake, and Taxi), demonstrating its ability to match or outperform traditional RL agents in simple environments by leveraging pretrained knowledge.
- PARL's performance is limited in tasks requiring complex mathematical operations or state/action decoding, highlighting challenges in structured, non-linguistic environments.

---

[Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](http://arxiv.org/abs/2510.21302)

- NESYRO (Neuro-Symbolic Robot Task Planning Framework): introduces a neuro-symbolic robot task planning framework that integrates Neuro-symbolic Code Verification and Neuro-symbolic Code Validation, ensuring generated robot control code is logically consistent and environmentally grounded through a recursive process.
- The framework employs an LLM for code generation and refinement, a symbolic tool for verification, and a neuro-symbolic confidence score (NeSyConf) combining LLM-based common sense and symbolic logic for validation.
- NESYRO utilizes a safe probe pipeline to generate exploratory code for acquiring missing observations, enabling robust task planning and execution in dynamic, partially observable environments.

---

[Securing AI Agent Execution](http://arxiv.org/abs/2510.21236)

- AGENTBOUND: introduces an access control framework for securing AI agent applications that interact with MCP servers, combining an Access Control Policy (declarative permission specification), a Policy Enforcement Engine (runtime permission enforcement), and AgentManifestGen (automated manifest generation).
- This framework enforces least-privilege and isolation at runtime by explicitly defining and enforcing permissions for MCP servers, preventing malicious behaviors like data exfiltration and limiting the impact of tool description manipulations.
- AGENTBOUND provides a practical foundation for securing MCP servers with negligible performance overhead, enabling developers to automatically generate and refine access control policies.

---

[DispatchMAS: Fusing taxonomy and artificial intelligence agents for emergency medical services](http://arxiv.org/abs/2510.21228)

- DispatchMAS (Fusing taxonomy and artificial intelligence agents for emergency medical services): introduces a taxonomy-grounded, LLM-powered multi-agent system for simulating realistic Emergency Medical Dispatch (EMD) scenarios, with Taxonomy & Fact Commons (structured knowledge base), Agentic System (LLM-based multi-agent simulation), and Human-algorithm Hybrid Evaluation (performance assessment framework).
- The system leverages a comprehensive clinical taxonomy and fact commons to constrain LLM agents, ensuring clinically plausible and procedurally aligned interactions between caller, dispatcher, and auxiliary responder agents.
- Evaluated by expert physicians and algorithmic metrics, the framework demonstrates high performance in dispatch effectiveness, guidance efficacy, and communication quality, supporting dispatcher training and protocol evaluation.

---

[Social Simulations with Large Language Model Risk Utopian Illusion](http://arxiv.org/abs/2510.21180)

- LSSAF (LLM Social Simulation Analysis Framework): introduces a systematic framework for analyzing LLMs' behavior in social simulation, utilizing LLM-driven social simulations, an analysis module, social role distribution analysis, semantic similarity analysis, keyword persistence analysis, emotional tone analysis, and linguistic patterns analysis to identify systematic divergences from human interaction.
- The framework simulates multi-agent interactions through chatroom-style conversations with role-conditioned agents and evaluates dialogues across five linguistic dimensions to uncover emergent social cognitive biases.
- Findings reveal that LLMs do not faithfully reproduce genuine human behavior but instead reflect idealized versions, exhibiting social role bias, primacy effect, and positivity bias, leading to "Utopian" societies.

---

[SOFT INSTRUCTION DE-ESCALATION DEFENSE](http://arxiv.org/abs/2510.21057)

- SIC (Soft Instruction Control): introduces an iterative prompt sanitization loop for tool-augmented LLM agents, which repeatedly inspects untrusted incoming data for malicious instructions, rewriting or removing them until clean or an iteration limit is reached.
- The framework operates as a modular preprocessing layer, augmenting untrusted input with dummy instructions, sanitizing it, and then detecting any remaining instructions before passing the combined clean data and user query to the LLM agent.
- This multi-pass approach enhances robustness against prompt injection attacks by catching missed injections in later steps and halting agent execution if instruction-like content persists, significantly raising the bar for adversaries.

---

[Towards AI Agents for Course Instruction in Higher Education: Early Experiences from the Field](http://arxiv.org/abs/2510.20255)

- AI Instructor Framework: introduces an LLM-driven pedagogical approach for higher education, integrating an AI Instructor Agent (LLM-driven conversational agent), Human Instructor (curriculum designer/Q&A facilitator), Teaching Assistants (lab/tutorial session leaders), and an Engagement Analytics Framework (student engagement evaluator) within a structured workflow.
- The framework leverages Microsoft Teams Copilot (AI Instructor Agent deployment environment) for student-agent interactions and Moodle (LMS) for course structure and assessments, while FaaS Workflows (cloud-native processing) and an Evaluation Engine (transcript analyzer) automate engagement metric computation and feedback generation.
- This system enables student-driven self-paced active learning, allowing students to explore concepts and clarify doubts with the AI Instructor Agent, and provides quantitative insights into engagement patterns through dialogue-derived metrics like Topic Coverage, Topic Depth, and Turn Length.

---

[Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding](http://arxiv.org/abs/2510.20176)

- MIXTURE-OF-MINDS (Mixture-of-Minds): introduces a multi-agent framework for table understanding, which includes planning-, coding- and answering-agents, where the planning agent outlines reasoning steps, the coding agent generates and executes code, and the answering agent synthesizes outputs for the final answer.
- The framework incorporates a self-improvement training mechanism utilizing MCTS-style data generation to create pseudo-gold intermediate trajectories for multi-agent optimization.
- Agent policies are refined using Group Relative Policy Optimization (GRPO) with reward functions designed to assess plan quality, code execution validity, and final answer accuracy.

---

[VISCODER2: BUILDING MULTI-LANGUAGE VISUALIZATION CODING AGENTS](http://arxiv.org/abs/2510.23642)

- VisCoder2 (Building Multi-Language Visualization Coding Agents): introduces a framework for building and evaluating visualization coding agents, comprising VisCode-Multi-679K (dataset), VisPlotBench (benchmark), and VisCoder2 (visualization coding agent) with internal Self-Debug (iterative correction), Execution (code execution/rendering), and Rendering (runtime feedback/self-correction) components, to address limitations in multi-language coverage and iterative correction for visualization code generation.
- The framework provides a large-scale instruction-tuning dataset (VisCode-Multi-679K) with 679K executable visualization code pairs across 12 languages and a benchmark (VisPlotBench) for systematic evaluation of initial generation and multi-round self-debug across 8 languages and 13 visual categories.
- VisCoder2 agents, trained on the provided dataset, iteratively generate, execute, render, and self-debug visualization code, achieving performance comparable to proprietary LLMs, especially benefiting symbolic or compiler-dependent languages through iterative refinement.

---

[NUM2EVENT: INTERPRETABLE EVENT REASONING FROM NUMERICAL TIME-SERIES](http://arxiv.org/abs/2510.23630)

- NUM2EVENT framework introduces "number to event reasoning and decoding," which infers interpretable structured events from numerical time-series data, even when current text is unavailable, by integrating an AGE (extracts AAOD events), an EveDTS (generates synthetic data), and a two-stage finetuning pipeline (encoder training, LLM finetuning).
- This framework addresses data scarcity and semantic alignment challenges by using AGE to build an extensible event vocabulary and EveDTS to synthesize realistic number-event paired data.
- The two-stage finetuning pipeline, comprising a time-series encoder and an LLM, enables the model to explicitly reason over numerical changes, generate intermediate explanations, and output structured event hypotheses.

---

[LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](http://arxiv.org/abs/2511.00116)

- LC-Opt (Sustainable Liquid Cooling benchmark environment): introduces a high-fidelity Modelica-based digital twin of a supercomputer's liquid cooling system, providing a Gymnasium interface for RL agents, LLM-based controllers, and traditional methods to optimize end-to-end thermal management from cooling towers to server blade groups.
- This framework enables multi-objective real-time optimization, balancing local thermal regulation and global energy efficiency, and supports multi-agent RL, policy distillation for interpretability, and LLM-based explainability of control actions.
- LC-Opt democratizes access to customizable liquid cooling models, fostering development of sustainable data center cooling solutions and enhancing user trust through natural language explanations of complex control decisions.

---

[Exploring Dissatisfaction in Bus Route Reduction through LLM-Calibrated Agent-Based Modeling](http://arxiv.org/abs/2510.26163)

- LLM-Calibrated ABM (Agent-Based Modeling): introduces a framework that simulates bus route reduction impacts on passenger dissatisfaction across demographic groups and network resilience by integrating an agent-based model with LLM-assisted parameter calibration.
- The framework employs an ABM Simulation System to model interactions between Bus Agents and Passenger Agents within an Environment, while an LLM (GPT-40) calibrates passenger sensitivity parameters for travel time, transfers, waiting, and crowding.
- It further includes Feature Evaluation and Resilience Test modules to quantify the influence of capacity, structure, and functional factors on dissatisfaction, identifying critical routes and providing policy recommendations for equitable and resilient transport planning.

---

[LIGHTAGENT: MOBILE AGENTIC FOUNDATION MODELS](http://arxiv.org/abs/2510.22009)

- LightAgent: introduces a mobile agentic foundation model solution that leverages device-cloud collaboration to balance cost-efficiency and high capability for mobile GUI tasks.
- It enhances a lightweight on-device MLLM with efficient long-reasoning and memory management, trained using synthetic GUI data and a two-stage fine-tuning protocol.
- The framework dynamically orchestrates tasks between the on-device model and cloud models based on real-time complexity assessment, minimizing cloud costs while maintaining high task success rates.

---

[MOBILERL: ONLINE AGENTIC REINFORCEMENT LEARNING FOR MOBILE GUI AGENTS](http://arxiv.org/abs/2509.18119)

- MOBILERL (Online Agentic Reinforcement Learning for Mobile GUI Agents): introduces an online agentic reinforcement learning framework for mobile GUI agents, combining a two-stage Reasoning Warm-Up with a Difficulty-Adaptive GRPO (ADAGRPO) algorithm, Actor, Mobile (Environment), Shortest-Path Reward Adjustment (SPA), AdaPR Buffer, Mixture sampling, Filtering (Failure Curriculum Filtering - FCF), and Policy Update.
- The framework addresses challenges like sparse rewards, heavy-tailed task difficulty, and sampling bottlenecks by integrating difficulty-adaptive positive replay, failure curriculum filtering, and shortest-path reward adjustment.
- MOBILERL achieves state-of-the-art success rates on AndroidWorld and AndroidLab benchmarks, demonstrating improved sample efficiency and stable RL training for multi-turn agentic tasks.

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

[TOWARDS SCALABLE OVERSIGHT WITH COLLABORATIVE MULTI-AGENT DEBATE IN ERROR DETECTION](http://arxiv.org/abs/2510.20963)

- ColMAD (Collaborative Multi-Agent Debate): introduces a new multi-agent debate protocol for LLM error detection, featuring debater agents (collaborate, criticize, complement), a judge agent (makes informed decision), LLM responses (initial solution to task), task requirements (problem constraints), debate transcripts (recorded agent interactions), evidence verification (quotes from context), self-auditing (identifies potential failures), and confidence calibration (estimates claim certainty).
- This framework reframes multi-agent debate as a non-zero-sum game, encouraging agents to cooperatively criticize and complement each other's points to provide comprehensive evidence to the judge, mitigating "debate hacking" observed in competitive approaches.
- ColMAD significantly outperforms previous competitive MAD protocols and single-agent methods in error detection, yielding more human-aligned explanations and demonstrating robustness across various LLM combinations and debate rounds.

---

[Thought Communication in Multiagent Collaboration](http://arxiv.org/abs/2510.20733)

- THOUGHTCOMM (Thought Communication): introduces a novel paradigm for multi-agent LLM collaboration by enabling direct mind-to-mind communication through the extraction and sharing of latent thoughts, bypassing natural language limitations.
- The framework formalizes thought communication as a latent variable model, employing a sparsity-regularized autoencoder to identify shared and private latent thoughts and their structural organization across agents.
- THOUGHTCOMM leverages an agreement-based reweighting strategy and prefix adaptation to selectively integrate relevant latent thoughts into each agent's context, enhancing collaboration and reasoning in multi-agent systems.

---

[Diagnosing Visual Reasoning: Challenges, Insights, and a Path Forward](http://arxiv.org/abs/2510.20696)

- Agent-based Architecture: introduces an agent-based architecture that combines LLM reasoning with lightweight visual modules, including an orchestrating agent, captioning, OCR, Python interpreter, image question and answering tools, a backbone LLM, and backtracing for iterative refinement, to diagnose and address visual reasoning failures.
- This architecture enables fine-grained analysis and iterative refinement of reasoning chains, improving performance on complex visual tasks.
- The framework leverages specialized tools to offload token usage and enhance perceptual grounding, outperforming larger models with smaller backbones.

---

[Addressing Corner Cases in Autonomous Driving: A World Model-based Approach with Mixture of Experts and LLMs](http://arxiv.org/abs/2510.21867)

- WM-MoE (World Model-based Mixture of Experts): introduces a world model-based motion forecasting framework that unifies Perception, Memory, and Decision Modules, leveraging LLMs and a Mixture-of-Experts network to address challenging corner cases in autonomous driving.
- The framework enhances long-horizon reasoning through a lightweight temporal tokenizer that maps agent trajectories and contextual cues into an LLM's feature space, enriching temporal context and commonsense priors.
- A Mixture-of-Experts network decomposes complex corner cases into subproblems, allocating capacity to specialized experts for inferring agent intent and performing counterfactual rollouts, while the nuScenes-corner dataset provides a new benchmark for rigorous evaluation.

---

[LLM-EMPOWERED KNOWLEDGE GRAPH CONSTRUCTION: A SURVEY](http://arxiv.org/abs/2510.20345)

- LLM-EMPOWERED KNOWLEDGE GRAPH CONSTRUCTION: introduces a comprehensive survey of recent progress in LLM-empowered knowledge graph construction, systematically analyzing how LLMs reshape the classical three-layered pipeline of ontology engineering, knowledge extraction, and knowledge fusion.
- The survey reviews LLM-driven approaches from two complementary perspectives: schema-based paradigms emphasizing structure and consistency, and schema-free paradigms highlighting flexibility and open discovery.
- This systematic review clarifies the evolving interplay between LLMs and knowledge graphs, bridging symbolic knowledge engineering and neural semantic understanding toward adaptive, explainable, and intelligent knowledge systems, and outlines future research directions.

---

[ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases](http://arxiv.org/abs/2510.20270)

- IMPOSSIBLEBENCH: introduces a benchmark framework that systematically measures LLM agents' propensity to exploit test cases by creating "impossible" variants of tasks from existing Base Benchmarks using Test Mutations (One-Off and Conflicting Mutations), where any pass by LLM Agents necessarily implies a specification-violating shortcut, quantified by a Cheating Rate.
- The framework also serves as a versatile tool for studying model behaviors, context engineering through Prompt Engineering and Test Access Configurations, and developing Monitoring Tools for deceptive behavior, utilizing different Scaffolds (Minimal and Full Scaffolds) and a Feedback Loop.
- IMPOSSIBLEBENCH provides a controlled environment and rich dataset of cheating transcripts for building more robust and reliable LLM systems, revealing diverse Cheating Methods like test modification, operator overloading, state recording, and special-casing.

---

[Using Large Language Models for Abstraction of Planning Domains - Extended Version](http://arxiv.org/abs/2510.20258)

- PDAG (Planning Domain Abstraction Generation): introduces a framework that leverages LLMs for generating abstract PDDL domains and problem instances from concrete low-level PDDL representations, guided by a natural language abstraction purpose.
- The framework utilizes in-context learning with LLMs, specifically GPT-40, to perform various abstraction tasks, including abstracting alternative concrete actions, sequences of concrete actions, and action/predicate parameters.
- The generated abstract PDDL outputs are evaluated through a hybrid approach combining symbolic validation tools and human expert review to assess correctness and adherence to the specified abstraction purpose.

---

[Automated Cloud Infrastructure-as-Code Reconciliation with AI Agents](http://arxiv.org/abs/2510.20211)

- NSYNC: introduces an automated system for Infrastructure-as-Code (IaC) reconciliation, which propagates out-of-band cloud infrastructure changes back into the IaC program by analyzing API traces and generating code patches.
- The system employs an LLM-based agent with specialized tools for intent identification and patch generation, supported by an evolving knowledge base for continuous learning.
- NSYNC addresses infrastructure drift by interpreting noisy API traces, synthesizing targeted IaC updates, and improving over time through accumulated reconciliation experience.

---

[Merge and Conquer: Evolutionarily Optimizing AI for 2048](http://arxiv.org/abs/2510.20205)

- MCTS Refinement System: introduces an evolutionary training method for optimizing AI to solve the 2048 game, which includes an overall framework, a move guidance algorithm, a board evaluation function, game execution, performance evaluation, strategy updates, iterative training blocks, intermediate selection, a rollback mechanism, and final refinement.
- This single-agent system refines a value function for a limited Monte Carlo Tree Search, demonstrating substantial improvements in AI performance for the 2048 game.
- The system incorporates a rollback feature to prevent performance degradation by selecting top-performing value functions using weighted probabilities after every five training cycles.

---

[AI PB: A Grounded Generative Agent for Personalized Investment Insights](http://arxiv.org/abs/2510.20099)

- AI PB (AI Private Banker): introduces a production-scale generative agent for personalized investment insights, integrating a component-based orchestration layer, a hybrid retrieval pipeline, and a multi-stage recommendation mechanism.
- The system operates on-premises under financial regulations, utilizing a dual-interface design for proactive daily briefings and interactive chat, ensuring grounded, compliant, and user-specific outputs.
- It employs a robust safety guard, deterministic routing, and a hybrid recommender to deliver trustworthy AI insights with high factuality and compliance in high-stakes finance.

---

[Beyond Prompt Engineering: Neuro-Symbolic-Causal Architecture for Robust Multi-Objective AI Agents](http://arxiv.org/abs/2510.23682)

- Chimera (Neuro-Symbolic-Causal Architecture): introduces a neuro-symbolic-causal architecture that integrates an LLM Strategist, a Symbolic Guardian, and a Causal Engine, along with supporting components, to enable robust multi-objective AI agents.
- This architecture combines flexible reasoning, formal safety guarantees, and counterfactual foresight, demonstrating superior performance, stability, and robustness compared to LLM-only or LLM+Guardian baselines in e-commerce simulations.
- The framework's design emphasizes architectural choices over prompt engineering for reliable autonomous agents, providing explainability and continuous improvement through modular components.

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

[Synthetic social data: trials and tribulations](http://arxiv.org/abs/2510.19952)

- LLM-based Synthetic Data Generation Framework: introduces an evaluation of using LLMs (openai, llama3.1:8b, gemma3:4b, cohere, google, mistral:7b) for generating synthetic social survey data, comparing their outputs against human responses from the World Values Survey (WVS) using prompt engineering, demographic variables, and statistical analysis.
- The study reveals that LLM-generated responses consistently diverge from human benchmarks, with 94.4% showing statistically significant differences across 15 questions and four countries.
- Even small, demographically skewed human samples proved more reliable than synthetic data, highlighting that machine bias in LLMs is a more significant issue than traditional survey sampling bias for social research.

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

[Everything counts: the managed omnirelevance of speech in ‘human – voice agent' interaction](http://arxiv.org/abs/2510.22610)

- Omnirelevance of Speech Analysis: introduces a study on human-voice agent interaction, analyzing how human participants adapt their conversational practices to the "omnirelevance of speech" perceived by artificial agents, which treat any human speech as a full-fledged turn requiring a response.
- The research examines interactions with both rule-based robots (Pepper) and LLM-based voice agents (ChatGPT advanced voice mode), highlighting how humans employ specific interactional practices, such as "aside sequences" using multimodal cues, to manage or exclude agents from participation.
- This analysis reveals that human participants perform significant "work to make technology work" by designing their contributions to remain below the agent's hearing threshold or by learning "Voice User Interface speak" to ensure interaction progressivity despite the agents' silence-based turn-taking models.

---

[SIGN: Schema-Induced Games for Naming](http://arxiv.org/abs/2510.21855)

- SIGN (Schema-Induced Games for Naming): introduces a naming game to investigate how lightweight schema structures influence convention formation among LLM agents, utilizing agents, a lexicon, memory windows, a decoder, and an adoption probability within a simulated environment.
- The paper demonstrates that schema-induced communication leads to faster convergence and significantly higher population agreement compared to unconstrained natural language or natural language with memory.
- By imposing a minimal message schema, SIGN acts as a control knob for efficient multi-agent coordination, reducing tokens-to-convergence and improving consistency in LLM agent interactions.

---

[ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering](http://arxiv.org/abs/2510.20036)

- ToolScope: introduces a two-part framework, including ToolScopeMerger (tool consolidation), Candidate Generation (overlap identification), Relationship Classification (semantic equivalence detection), Graph Construction (overlap representation), Tool Pruning (representative tool selection), Auto-Correction (merge validation/refinement), Toolset and Dataset Update (merged tool synthesis), ToolScopeRetriever (relevant tool selection), Query Decomposition (sub-query generation), Hybrid Retrieval (sparse/dense score combination), Reranking (candidate re-scoring), Normalization (score scaling), LLM Agent Tool Selection (final tool choice), Toolset (original tool collection), Refined Toolset (merged tool collection), Top-k Tools (selected relevant tools), User Query (user task input), LLM Classifier (semantic overlap detection), LLM Auto-Correct & Merger (merge error correction), LLM MD (tool documentation synthesis), Embedding Model (tool description encoding), and Cross-encoder (reranking model), to enhance LLM agent tool use by addressing tool overlap and context length limitations.
- The framework automatically merges semantically redundant tools and retrieves the most relevant tools for a given query, significantly improving tool selection accuracy across various benchmarks and LLMs.
- ToolScope provides a scalable solution for improving LLM-agent tool selection in real-world settings by reducing toolset redundancy and efficiently managing context length constraints.

---

[SALT: Step-level Advantage Assignment for Long-horizon Agents via Trajectory Graph](http://arxiv.org/abs/2510.20022)

- SALT (Step-level Advantage Assignment for Long-horizon Agents via Trajectory Graph): introduces a novel framework that provides finer-grained, step-level advantage assignment for LLM agents in long-horizon tasks, leveraging a trajectory graph constructed from multiple rollouts to distinguish shared and distinct steps.
- This plug-and-play module integrates seamlessly with existing group-based RL algorithms like GRPO and RLOO, refining trajectory-level advantages into step-level advantages without requiring additional supervision or reward models.
- By averaging advantages for merged (shared) steps and preserving original advantages for divergent (distinct) steps, SALT reduces gradient conflicts and stabilizes training, leading to consistent performance improvements across various benchmarks and model scales.

---

[Communication to Completion: Modeling Collaborative Workflows with Intelligent Multi-Agent Communication](http://arxiv.org/abs/2510.19995)

- C2C (Communication to Completion): introduces a scalable framework for multi-agent LLM systems, integrating a Simulation Engine Layer, Agent Layer with SAF, Communication Layer, Execution Blocks, and Core Services, alongside key concepts like Alignment Factor, Hierarchical Task Management, and Intention-Based Agent Decision Making, to optimize collaborative workflows.
- The framework addresses the lack of systematic communication strategies in current multi-agent LLM systems by treating communication as an optimizable resource, quantified by the Alignment Factor, which directly impacts work efficiency.
- C2C enables agents to make cost-aware communication choices and dynamically improve task understanding through targeted interactions, demonstrating reduced task completion time and improved work efficiency in coding workflows.

---

[Learning from Supervision with Semantic and Episodic Memory: A Reflective Approach to Agent Adaptation](http://arxiv.org/abs/2510.19897)

- Learning from Supervision with Semantic and Episodic Memory: introduces a memory-augmented framework that enables LLM agents to learn classification functions from labeled examples and LLM-generated critiques without parameter updates.
- The framework leverages episodic memory for instance-level experiences and semantic memory for task-level guidance, supporting continuous adaptation through structured supervision.
- Experiments demonstrate up to 24.8% accuracy gain over label-only baselines and introduce a novel suggestibility metric to explain how models internalize feedback via memory.

---

[Large Language Model enabled Mathematical Modeling](http://arxiv.org/abs/2510.19895)

- DeepSeek-R1 OR Application Framework: proposes a structured pipeline for applying the DeepSeek-R1 LLM to operations research (OR) problems, integrating mitigation strategies such as LLM-as-a-Judge, Few-shot Learning (FSL), Tool Calling, and a Multi-agent Framework.
- This research systematically evaluates DeepSeek-R1's performance in mathematical modeling and code generation across four OR benchmarks, developing a hallucination taxonomy to categorize and address errors.
- The LLM-as-a-Judge strategy significantly enhances DeepSeek-R1's accuracy by enabling self-critique and revision, while FSL, Tool Calling, and the Multi-agent Framework provide additional capabilities for robust OR problem-solving.

---

[Knowledge-Guided Multi-Agent Framework for Application-Level Software Code Generation](http://arxiv.org/abs/2510.19868)

- KGACG (Knowledge-Guided Application-Level Code Generation): introduces a multi-agent framework for application-level software code generation, featuring COPA, CA, TA, SRS & ADD Knowledge, Coding Knowledge, Testing Knowledge, and Feedback, which transforms software requirements and architectural design into executable code through collaborative, iterative processes.
- The framework leverages LLMs within its specialized agents to address challenges in large-scale code generation, such as context isolation, lack of grounding, and absence of iteration, by enabling continuous self-correction and optimization.
- KGACG integrates external knowledge bases and a closed-loop feedback mechanism among its agents to ensure generated code adheres to standards, is maintainable, and is iteratively refined based on compilation and test results.

---

[Can LLMs Translate Human Instructions into a Reinforcement Learning Agent's Internal Emergent Symbolic Representation?](http://arxiv.org/abs/2510.24259)

- LLM-based Translation System: introduces a framework to evaluate the capacity of LLMs to translate human natural language instructions into the internal emergent symbolic representations generated by the STAR (Spatio-Temporal Abstraction Via Reachability) hierarchical reinforcement learning algorithm, including LLMs, STAR algorithm, human instructions, emergent symbolic representations, prompt construction function, ground truth outputs, G-BLEU metric, Ant Maze environment, Ant Fall environment, and robotic agent, where the system measures translation performance across varying symbolic abstractions and task complexities.
- The research reveals that LLMs demonstrate some ability to translate natural language into symbolic representations of environment dynamics, but their performance is highly sensitive to partition granularity and task complexity, especially when tool use is involved.
- The findings highlight limitations in current LLMs' capacity for representation alignment, indicating a need for further research on robust alignment between language and internal agent representations for developmental learning agents.

---

[SCOPE VLM: Selective Context Processing for Efficient Document Navigation in Vision-Language Models](http://arxiv.org/abs/2510.21850)

- SCOPE VLM (Selective Context Processing for Efficient Document Navigation in Vision-Language Models): introduces a document navigation expert that leverages a novel Chain of Scroll mechanism to selectively and recursively navigate documents, focusing exclusively on relevant segments, enhanced by Episodic Group Relative Policy Optimization for context-based action selection.
- The Chain of Scroll (CoS) Framework, an action-based inference strategy, allows the model to focus on relevant document segments by iteratively deciding whether to scroll to new pages or generate an answer based on accumulated context and relevance signals.
- Episodic Group Relative Policy Optimization (EGRPO) is a tailored reinforcement learning method that reduces the gap between training and inference, improving the model's agentic capabilities for effective context-based action selection and memory efficiency.

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

