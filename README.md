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

## Research papers: 2026

[2026 (2/2)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2026 (1/2)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2025 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_4.md),[2025 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_3.md), [2025 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (1/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_01.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)

Chronological order. 





</div>



#### 28th January 2026


[SERA: Soft-Verified Efficient Repository Agents](http://arxiv.org/abs/2601.20789)

- SERA (Soft-Verified Efficient Repository Agents): introduces an efficient method for training coding agents using Soft Verified Generation (SVG), a two-rollout pipeline where a Teacher Model generates synthetic trajectories and patches from a Codebase, which are then Soft Verified via line-level patch comparison for Supervised Fine-Tuning (SFT) of the Student Model.
- SVG enables rapid and cheap data generation from any repository by replacing complex unit test verification with simple patch overlap comparison, achieving state-of-the-art open-source results 26x cheaper than reinforcement learning.
- The approach demonstrates high sample efficiency, allowing the SERA agent to specialize to private codebases and match or exceed teacher LLM performance with only 8,000 specialized trajectories.

---


[Deep Researcher with Sequential Plan Reflection and Candidates Crossover (Deep Researcher Reflect Evolve)](http://arxiv.org/abs/2601.20843)

- Deep Researcher Reflect Evolve: introduces a novel architecture for generating detailed research reports using Planning Agent (curates and refines plan), Search Agent (generates and answers queries), Report Writer Agent (generates final report), Global Research Context (centralized memory repository), Sequential Research Plan Refinement via Reflection (dynamic plan adaptation), Candidates Crossover algorithm (synthesizes parallel LLM findings), Web Search Tool (gathers external data), Model Configuration (varied LLM parameters), and Candidate LLM units (parallel search investigators).
- The system shifts from parallel scaling to sequential refinement, utilizing the Sequential Research Plan Refinement via Reflection module to maintain a centralized Global Research Context for dynamic adaptation and avoiding siloed knowledge.
- The Candidates Crossover algorithm enhances search efficiency by deploying multiple LLM candidates with varied parameters to explore a larger search space, synthesizing their findings into a comprehensive One Shot Report Generation.

---

[Idea2Story: An Automated Pipeline for Transforming Research Concepts into Complete Scientific Narratives](http://arxiv.org/abs/2601.20833)

- Idea2Story: introduces a pre-computation-driven framework for autonomous scientific discovery that shifts literature understanding from online reasoning to offline knowledge construction, utilizing Paper Pool Construction (Curating/Cleaning Data), Method Unit Extraction (Clustering/Decomposition), Knowledge Graph Construction (Organizing Units/Relations), User Intent Processing (Aligning Input), Research Pattern Retrieval (Multi-view Retrieval), Review-Guided Refinement (LLM-based Review Loop), and Review Agent (Iterative Evaluation/Revision).
- The framework operates in two stages: an offline stage builds a structured methodological knowledge graph from peer-reviewed papers, and an online stage uses retrieval over this graph to ground underspecified user intent into concrete research patterns.
- By grounding research planning and execution in the pre-built knowledge graph, Idea2Story alleviates the context window bottleneck of LLMs and substantially reduces repeated runtime reasoning over scientific literature.

---

[MemCtrl: Using MLLMs as Active Memory Controllers on Embodied Agents](http://arxiv.org/abs/2601.20831)

- MemCtrl: introduces a novel memory augmentation scheme that uses a trainable memory head ($\mu$) integrated with an MLLM backbone to actively filter observations for storage in memory, improving efficiency for embodied agents.
- The memory head acts as a binary classifier, trained either via offline expert supervision or online RL, deciding in real-time whether to keep or discard the current observation, thereby enabling active write-time memory control.
- The framework demonstrates significant performance improvement (around 16% average gain) on low-parameter MLLMs tackling long-horizon and complex tasks within the EmbodiedBench benchmark.

---

[Multimodal Multi-Agent Ransomware Analysis Using AutoGen](http://arxiv.org/abs/2601.20346)

- MMMA-RA (Multimodal Multi-Agentic Ransomware Analysis): introduces a multimodal multi-agent framework for ransomware classification that integrates static, dynamic, and network modalities using DCAE feature extraction, latent space fusion, a transformer-based family classifier, post-hoc calibration, and an AutoGen LLM loop featuring User Proxy, Assistant, and Critic agents.
- The architecture employs modality-specific Deep Contrastive Auto Encoders (DCAEs) to extract discriminative latent representations, which are then fused via a gated mechanism and fed into a class-imbalance-aware classifier for family prediction.
- The AutoGen multi-agent loop, driven by a Local Phi-3.2B LLM, acts as a meta-controller, providing non-intrusive feedback to iteratively refine sampling and calibration, prioritizing decision reliability over forced classification.

---


[Learning From a Steady Hand: A Weakly Supervised Agent for Robot Assistance under Microscopy](http://arxiv.org/abs/2601.20776)

- Weakly Supervised Agent (WSA): introduces a weakly supervised framework for robotic microscopy assistance that fuses calibration-aware perception with admittance control, utilizing warm-up trajectories to train the Two-stage 3D perception framework (Estimates 3D tip), Uncertainty-aware Hand-eye Calibration (Marker-free extrinsic estimation), Soft-Gated Estimator (Kalman filter tracking), and Hierarchical Control Architecture (Macro/Micro layers).
- The system achieves micron-level precision (49 µm lateral, 291 µm axial at 95% confidence) by leveraging implicit spatial information from reusable warm-up trajectories, avoiding labor-intensive 2D labeling or external fiducials.
- The Hierarchical Control Architecture features an always-on admittance macro layer for safety and a confidence-gated micro layer for precision, reducing operator workload by 77.1% compared to a steady-hand baseline.

---

[Agentic Fog: A Policy-driven Framework for Distributed Intelligence in Fog Computing](http://arxiv.org/abs/2601.20764)

- AF (Agentic Fog): introduces a policy-driven framework for distributed intelligence in fog computing, featuring the Global Orchestrator Agent (Goal Decomposition, Policy Control), Shared Memory (Abstracted Knowledge Storage), Fog Agents (Decentralized Decision-Making, Caching, Routing, Load Balancing), Execution Agents (Localized Task Execution, Stateless Services), Peer-to-Peer Coordination (Local Stabilization, Conflict Resolution), and Potential Game Formulation (Convergence Guarantees, Stability).
- The framework models Fog Nodes as autonomous, policy-driven agents that coordinate via shared memory and localized P2P interactions, ensuring convergence and stability under partial observability and asynchronous updates.
- AF achieves lower average latency and greater resilience to node failures compared to ILP-based and Greedy heuristic baselines by decoupling system-level cognition from local decision-making.

---

[Continual GUI Agents](http://arxiv.org/abs/2601.20732)

- GUI-AiF (GUI-Anchoring in Flux): introduces a reinforcement fine-tuning framework for Continual GUI Agents, stabilizing learning under shifting domains and resolutions using Anchoring Point Reward in Flux (APR-iF) and Anchoring Region Reward in Flux (ARR-iF).
- The framework optimizes the grounding policy within the Reinforcement Fine-Tuning (RFT) paradigm by integrating APR-iF, which encourages diverse interaction points, and ARR-iF, which promotes diversity in predicted element regions.
- GUI-AiF addresses the challenge of maintaining stable grounding in dynamic GUI environments by mitigating policy over-adaptation to static grounding cues like fixed coordinates or element scales.

---

[AgentLongBench: A Controllable Long Benchmark For Long-Contexts Agents via Environment Rollouts](http://arxiv.org/abs/2601.20730)

- AgentLongBench: introduces a controllable long-context benchmark evaluating LLM agents via simulated environment rollouts based on Lateral Thinking Puzzles, generating rigorous interaction trajectories across knowledge-intensive and knowledge-free scenarios.
- The benchmark forces agents to perform dynamic information synthesis and non-linear reasoning by parsing high-density, machine-generated tool logs and tracking evolving states over long, iterative interactions.
- Evaluations reveal that state-of-the-art LLMs and memory frameworks struggle with dynamic information synthesis and high-density tool logs, indicating a failure in long-horizon state tracking and evidence localization.

---

[Dynamic Mechanism Design without Monetary Transfers: A Queueing Theory Approach](http://arxiv.org/abs/2601.20728)

- DTM: introduces an optimal allocation mechanism for dynamic stochastic environments without monetary transfers, utilizing queues and costly verification to manage stochastic arrivals of agents and goods.
- The optimal mechanism is characterized by state-dependent thresholds for agent admission ($v^*_k$) and goods allocation ($\hat{v}_l$), maintaining finite capacities for both the agent queue ($K^*$) and goods inventory ($L^*$).
- Verification is strategically postponed until the moment of allocation to ensure incentive compatibility and avoid wasting costs on agents who might be removed later due to higher-type arrivals.

---

[Distributed Learning over Noisy Communication Networks](http://arxiv.org/abs/2601.20723)

- Log-Linear Learning (LLL): introduces a unified analytical framework connecting channel models, network topology, and log-linear learning dynamics in distributed coordination games over noisy communication networks.
- The framework analyzes two operational regimes: the fast communication regime, which yields an exact Gibbs sampler for a scaled coordination potential, and the snapshot regime, which results in a generally nonreversible Markov chain.
- A finite-K communication budget model interpolates between the snapshot and fast regimes, providing a communication-theoretic interpretation of the tradeoff between communication resources and steady-state coordination quality.

---

[Li-ViP3D++: Query-Gated Deformable Camera-LiDAR Fusion for End-to-End Perception and Trajectory Prediction](http://arxiv.org/abs/2601.20720)

- Li-ViP3D++: introduces a query-based multimodal Perception and Prediction (PnP) framework utilizing Query-Gated Deformable Fusion (QGDF) to integrate multi-view RGB and LiDAR data in query space, jointly optimizing detection, tracking, and multi-hypothesis trajectory forecasting.
- QGDF aggregates image evidence via masked attention and extracts LiDAR context through fully differentiable BEV sampling with learned per-query offsets, applying query-conditioned gating to adaptively balance visual and geometric cues.
- The architecture significantly improves end-to-end behavior and detection quality, achieving higher EPA and mAP while substantially reducing false positives compared to prior variants, supporting robust camera-LiDAR fusion.

---

[Adapting the Behavior of Reinforcement Learning Agents to Changing Action Spaces and Reward Functions](http://arxiv.org/abs/2601.20714)

- MORPHIN: introduces a self-adaptive Q-learning framework that integrates concept drift detection with dynamic adjustments to exploration and learning rates to enable adaptation to shifting reward functions and expanding action spaces.
- The framework utilizes the Page-Hinkley test (PH-test) for proactive environment monitoring, triggering exploration resets and dynamic learning rate adjustments based on the Temporal Difference error.
- MORPHIN prevents catastrophic forgetting by preserving and augmenting a single Q-table structure, achieving up to 1.7x improvement in learning efficiency compared to standard Q-learning baselines in non-stationary environments.

---

[Grover's Search-Inspired Quantum Reinforcement Learning for Massive MIMO User Scheduling](http://arxiv.org/abs/2601.20688)

- QRL (Grover's Search-Inspired Quantum Reinforcement Learning): introduces a novel quantum-circuit-based QRL method for massive Multiple Input Multiple Output (mMIMO) user scheduling, leveraging Grover's search for efficient exploration of the exponentially large scheduling space.
- The QRL framework utilizes a layered quantum gate-based circuit comprising Hadamard, Oracle, and Diffusion layers to integrate conventional reinforcement learning principles with quantum search mechanisms.
- The proposed system achieves superior scalability and throughput compared to classical Convolutional Neural Networks and Quantum Deep Learning benchmarks in diverse user and antenna scaling scenarios.

---

[Polite But Boring? Trade-offs Between Engagement and Psychological Reactance to Chatbot Feedback Styles](http://arxiv.org/abs/2601.20683)

- Empirical Study on Chatbot Feedback Styles: introduces an investigation into how three distinct feedback styles (DIRECT, POLITENESS, VERBAL LEAKAGE) delivered by an LLM-powered chatbot influence user psychological reactance and message effectiveness across different psychological distance scenarios.
- The study uses a 3x2 mixed factorial design, manipulating Feedback Style (between-subjects) and Psychological Distance (within-subjects, personally- or societally-affecting scenarios) to assess emotional reactance and perceived threat to freedom.
- Findings reveal a trade-off where POLITENESS reduces reactance but lowers engagement, while VERBAL LEAKAGE increases surprise and engagement, suggesting alternative design opportunities beyond polite defaults.

---

[Efficient Multimodal Planning Agent for Visual Question-Answering](http://arxiv.org/abs/2601.20676)

- Efficient Multimodal Planning Agent (EMPA): introduces a multimodal planning agent trained via VQA query decomposition and fine-tuning to dynamically select necessary mRAG steps (Image Search, Query Rewrite, Query Search) for efficient Visual Question-Answering (VQA).
- The agent intelligently omits redundant retrieval operations, achieving over 60% reduction in search time and significantly lower tool-call latency compared to baseline methods like OmniSearch and WebWatcher.
- By optimizing the trade-off between efficiency and effectiveness, the approach maintains or surpasses VQA performance across six diverse datasets compared to rigid, complete mRAG pipelines.

---

[Deep Learning based Three-stage Solution for ISAC Beamforming Optimization](http://arxiv.org/abs/2601.20667)

- Deep Learning based Three-stage Solution: introduces a framework that decomposes ISAC beamforming optimization into feature extraction, beampattern optimization, and beamforming reconstruction using an Autoencoder, an A2C RL agent, and a DNN mapping network.
- The framework maximizes the sum communication rate subject to transmit power and minimum sensing rate constraints by optimizing the intuitive beampattern rather than the high-dimensional beamforming vector directly.
- This three-stage approach enhances generalization, improves training stability, and offers better interpretability compared to direct reinforcement learning methods.

---

[OS-Marathon: Benchmarking Computer-Use Agents on Long-Horizon Repetitive Tasks](http://arxiv.org/abs/2601.20650)

- FCWD (Few-shot Condensed Workflow Demonstration): introduces a cost-effective method using Semantic Key Steps (K) to teach Computer-Use Agents (CUAs) the invariant logic of long-horizon, repetitive workflows, enabling generalization to large, unseen data collections.
- The approach provides dual-level instruction, facilitating global planning for orchestrating the repetitive loop and teaching the fundamental logic for sub-workflow execution.
- The paper establishes OS-Marathon, the first benchmark tailored for long-horizon, repetitive desktop tasks, comprising 242 tasks across two domains (Expense Report and Transcript) and seven execution environments.

---

[Investigating the Development of Task-Oriented Communication in Vision-Language Models](http://arxiv.org/abs/2601.20641)

- VLM-TOC: introduces a referential game framework where Vision-Language Models (VLMs) act as agents (Sender, Receiver, Overseer) guided by Zero-Shot Prompting to spontaneously develop task-oriented communication protocols (Language Variants) optimized for efficiency and covertness.
- Experiments show that VLMs can develop effective, concise, task-adapted communication patterns that surpass natural language efficiency, and generate covert protocols difficult for external observers to interpret.
- Agents with similar VLM architectures can independently coordinate using these covert protocols, suggesting reliance on shared internal representations rather than explicit protocol sharing.

---

[Immersive Volumetric Video Playback: Resource Allocation and O-RAN-based Implementation](http://arxiv.org/abs/2601.20625)

- O-RAN-assisted ImViD playback framework: introduces an O-RAN-integrated system that jointly orchestrates radio, compute, and content resources using a SAC agent to maximize immersive Quality of Experience (QoE) for volumetric video streaming.
- The system formulates resource allocation (bandwidth, power, compute) and the rendered-pixel ratio (content-hit ratio) as a continuous control problem optimized under a Weber-Fechner QoE model to balance resolution, computation, and latency.
- Implemented on a 5G O-RAN prototype, the SAC-based approach reduces median Motion-to-Photon (MTP) latency by approximately 18% and improves both mean QoE and fairness compared to baselines.


---

[Agent Benchmarks Fail Public Sector Requirements](http://arxiv.org/abs/2601.20617)

- PSABR (Public-Sector Agent Benchmark Rubric): introduces a set of six criteria to evaluate the suitability of LLM agent benchmarks for public sector deployment, utilizing an LLM-assisted pipeline for systematic literature analysis.
- The analysis of over 1,300 benchmark papers reveals that no single existing benchmark meets all criteria, with the largest gaps found in public-sector specificity and comprehensive metric reporting (Cost and Fairness).
- The paper advocates for researchers to develop public sector-relevant benchmarks and for public officials to use these criteria to evaluate agentic use cases to mitigate deployment risks.

---

[AgentIF-OneDay: A Task-level Instruction-Following Benchmark for General AI Agents in Daily Scenarios](http://arxiv.org/abs/2601.20613)

- AgentIF-OneDay: introduces a task-level instruction-following benchmark for general AI agents in daily scenarios, structured around Open Workflow Execution, Latent Instruction Inference, and Iterative Refinement, and evaluated using an LLM Judge pipeline.
- The benchmark comprises 104 tasks covering work, study, and life domains, demanding agents deliver tangible file-based results and adhere to instance-level rubrics verified by a Gemini-3-Pro LLM Judge.
- Evaluation of leading agents reveals performance parity between API-driven and RL-based systems, highlighting persistent challenges in implicit constraint inference and long-horizon consistency.

---

[Inequality in Congestion Games with Learning Agents](http://arxiv.org/abs/2601.20578)

- RL-CG: introduces a framework combining multi-agent reinforcement learning dynamics with fairness analysis in congestion games to study how heterogeneous commuter adaptation affects efficiency and equity during transport network expansions.
- The framework defines the Price of Learning (PoL) to quantify dynamic inefficiency and Source Disparity (SD) to measure fairness, revealing that unequal learning rates amplify disparities, allowing faster learners to disproportionately benefit from new routes.
- Experiments conducted on a stylized two-source Braess's paradox network and an abstraction of the Amsterdam metro network demonstrate that network interventions can create persistent inequities and inefficiencies, especially when learning rates differ sharply.

---

[PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](http://arxiv.org/abs/2601.20539)

- PathWise (Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs): introduces a multi-agent LLM framework that formulates automated heuristic design as a sequential decision process over an Entailment Graph (Stateful search memory), guided by Policy Agent (Plans evolutionary actions), World Model Agent (Generates heuristic rollouts), Policy Critic (Reflects on evolutionary strategy), World Model Critic (Reflects on heuristic quality), Diversity Mechanisms (Prompt perturbation, state shuffling), and Sequential Decision Process (Formulates heuristic discovery).
- The framework uses coordinated LLM agents to navigate the Entailment Graph, which serves as a compact, stateful memory encoding derivation rationale and performance history to guide heuristic evolution.
- PathWise shifts LLM-based Automated Heuristic Design from trial-and-error evolution toward state-aware planning through reasoning, achieving faster convergence and stronger heuristics across diverse COPs.

---

[Interpreting Emergent Extreme Events in Multi-Agent Systems](http://arxiv.org/abs/2601.20538)

- Interpreting Emergent Extreme Events in Multi-Agent Systems Framework: introduces the first approach for explaining emergent extreme events (Black Swans) in LLM-powered Multi-Agent Systems (MAS) by quantifying risk contributions using Shapley Value Attribution, Dimensional Aggregation, and derived Metrics Derivation.
- The framework decomposes the final event risk by assigning attribution scores to individual agent actions across time, aggregating these scores along time, agent, and behavior dimensions to answer when, who, and what contributed to the event.
- Five quantitative metrics are derived from the aggregated contributions to characterize extreme event features, such as relative risk latency, agent risk concentration, and behavioral risk concentration, providing insights across economic, financial, and social MAS scenarios.

---

[Normative Equivalence in human-AI Cooperation: Behaviour, Not Identity, Drives Cooperation in Mixed-Agent Groups](http://arxiv.org/abs/2601.20487)

- Normative Equivalence study: introduces an experimental design using a repeated Public Goods Game (PGG) and a one-shot Prisoner's Dilemma (PD) to test cooperation dynamics in mixed groups of three humans and one automated agent, varying agent label (Human/AI) and strategy (Unconditional/Conditional/Free-Rider).
- The research finds evidence of normative equivalence, showing that cooperation levels and normative expectations are statistically similar across human-labeled and AI-labeled groups, indicating that group behavior, not agent identity, drives cooperation.
- Behavioral analysis confirms that reciprocal group dynamics and individual inertia primarily drive cooperation, operating identically regardless of the automated agent's label or specific scripted strategy.

---

[Piloting Planetarium Visualizations with LLMs during Live Events in Science Centers](http://arxiv.org/abs/2601.20466)

- LLM Visualization Pilot System: introduces an AI pilot system integrated with OpenSpace visualization software, comparing reactive and proactive LLM modes for controlling planetarium shows via voice commands.
- The system uses a low-latency, multimodal LLM agent coordinated by a JavaScript process to interpret guide utterances and dispatch tool calls for camera motions, simulation time changes, and asset toggling.
- Evaluation with professional guides suggests that while AI pilots lack human nuances like pacing and anticipation, they are useful as co-pilots to reduce cognitive load and enable multitasking during live events.

---

[BMAM: Brain-inspired Multi-Agent Memory Framework](http://arxiv.org/abs/2601.20465)

- BMAM (Brain-inspired Multi-Agent Memory Framework): introduces a general-purpose memory architecture that decomposes agent memory into functionally specialized subsystems (episodic, semantic, salience-aware, and control-oriented components) to address "soul erosion" in long-horizon LLM agents.
- The framework utilizes a timeline-indexed episodic memory organization (StoryArc) and a hybrid retrieval strategy that fuses lexical, dense, knowledge-graph, and temporal signals via reciprocal rank fusion.
- BMAM employs a hierarchical memory control mechanism coordinated by a central agent to manage memory access across complementary time scales and support asynchronous memory consolidation.

---

[Manipulation in Prediction Markets: An Agent-based Modeling Experiment](http://arxiv.org/abs/2601.20452)

- ABM (Agent-Based Model): introduces an open-source simulation framework to investigate how high-budget, biased "whale" agents can temporarily distort prices in prediction markets, especially when non-whale bettors exhibit herding or slow learning.
- The model simulates heterogeneous betting agents characterized by expertise, bias, stubbornness, and risk aversion, demonstrating that market resilience decreases as the whale's share of total market capital increases.
- The research provides a theoretical analysis corroborating simulation results, showing that price distortion magnitude and duration depend on the whale's budget share and the learning dynamics of other agents.

---

[PEARL: Plan Exploration and Adaptive Reinforcement Learning for Multihop Tool Use](http://arxiv.org/abs/2601.20439)

- PEARL (Plan Exploration and Adaptive Reinforcement Learning for Multihop Tool Use): introduces a two-stage framework for robust LLM tool use, integrating a dedicated Planner Model optimized via GRPO and an Executor Model grounded by Offline Tool Exploration.
- The Planner Model is trained using adaptive reinforcement learning guided by a novel Planning-Centric Reward Function that provides dense, stepwise feedback on the quality of the multi-step tool plan.
- The Executor Model leverages knowledge acquired during the Offline Tool Exploration phase to reliably execute the generated plan step-by-step, significantly reducing invocation error rates.

---

[Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](http://arxiv.org/abs/2601.20412)

- CLF (Cognitive Load Framework): introduces a diagnostic evaluation framework grounded in Cognitive Load Theory that deconstructs tool-use task difficulty into quantifiable Intrinsic Load and Extraneous Load components using the Tool Interaction Graph and evaluated via ToolLoad-Bench.
- The framework models agent performance as an exponential function of Total Cognitive Load, characterized by two model-specific parameters: Baseline Capability (b) and Load Sensitivity (k).
- Empirical validation confirms that LLMs possess distinct cognitive frontiers, exhibiting sharp, predictable performance cliffs as task complexity, measured by Total Cognitive Load, increases.

---

[On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents](http://arxiv.org/abs/2601.20404)

- Empirical Study on AGENTS.md Impact: introduces an empirical study evaluating the operational efficiency of AI coding agents (OpenAI Codex) when executing development tasks on real GitHub pull requests, comparing performance with and without the AGENTS.md file present in the Repository Snapshot.
- The study uses a paired within-task design across 10 repositories and 124 pull requests, measuring efficiency via wall-clock execution time and token consumption.
- Results indicate that the presence of a root AGENTS.md file is associated with a lower median runtime (28.64%) and reduced output token consumption (16.58%).

---

[SFQA: A COMPREHENSIVE PERCEPTUAL QUALITY ASSESSMENT DATASET FOR SINGING FACE GENERATION](http://arxiv.org/abs/2601.20385)

- SFQA (Singing Face Generation Quality Assessment Dataset): introduces a comprehensive quality assessment dataset for Singing Face Generation (SFG) content, built using 100 Reference Images, 36 Driven Audio Clips, and 12 SFG Methods to generate 5,184 Video Samples, which are then evaluated via Subjective Quality Assessment and Objective Quality Assessment Benchmarking.
- The dataset construction ensures content diversity by utilizing real human and AI-generated portraits across various categories and 36 music clips spanning seven distinct styles and languages.
- Subjective evaluation results reveal significant quality variations among generative algorithms, and benchmarking demonstrates the insufficient suitability of current objective quality assessment methods, including LMM-based models, for SFG content.

---

[HINT: HIERARCHICAL INTERACTION MODELING FOR AUTOREGRESSIVE MULTI-HUMAN MOTION GENERATION](http://arxiv.org/abs/2601.20383)

- HINT (Hierarchical INTeraction modeling): introduces the first autoregressive diffusion-based framework for text-driven multi-human motion generation, utilizing a Canonicalized Latent Space and a Sliding-Window Strategy guided by Hierarchical Motion Conditions.
- The framework decouples local motion semantics from inter-person interactions in the Canonicalized Latent Space, enabling seamless adaptation to varying numbers of agents without requiring additional refinement.
- HINT employs an Interaction-Aware Diffusion module that aggregates local (history, step index, partner history, word embedding) and global (sequence index, compositional command) cues to ensure fine-grained alignment and long-horizon coherence.

---

[OmegaUse: Building a General-Purpose GUI Agent for Autonomous Task Execution](http://arxiv.org/abs/2601.20380)

- OmegaUse: introduces a general-purpose GUI agent model built on a Mixture-of-Experts (MoE) backbone, trained using a decoupled two-stage paradigm (SFT and GRPO) on a high-quality data construction pipeline, supporting autonomous task execution across mobile, desktop, and web platforms.
- The framework utilizes specialized reward mechanisms, including Inside-of-Bounding-Box reward for grounding and Action-wise reward for navigation, to refine spatial perception and sequential decision-making.
- OmegaUse achieves state-of-the-art performance on GUI benchmarks, including 96.3% on ScreenSpot-V2 and a leading 79.1% step success rate on AndroidControl, and introduces the OS-Nav cross-terminal benchmark suite.

---

[LLM-AutoDP: Automatic Data Processing via LLM Agents for Model Fine-tuning](http://arxiv.org/abs/2601.20375)

- LLM-AutoDP: introduces a novel framework leveraging LLM agents to automatically generate and optimize data processing strategies through iterative refinement using feedback signals and comparative evaluations.
- The framework utilizes a closed-loop interaction between the strategy generation and evaluation modules, incorporating three key acceleration techniques (DPS, PTS, and CRM) to reduce the total search time by up to 10x.
- LLM-AutoDP automates data processing for LLM fine-tuning, achieving over 80% win rates against models trained on unprocessed data across medical datasets while maintaining computational efficiency.

---

[Unsupervised Anomaly Detection in Multi-Agent Trajectory Prediction via Transformer-Based Models](http://arxiv.org/abs/2601.20367)

- UAD-MATP (Unsupervised Anomaly Detection in Multi-Agent Trajectory Prediction): introduces an unsupervised anomaly detection framework utilizing a Multi-Agent Transformer, Residual Aggregation, Isolation Forest, and a Dual Evaluation Framework for discovering safety-critical driving scenarios.
- The core architecture uses a sequence-to-sequence Transformer with an Encoder capturing historical multi-agent motion and a Decoder predicting future trajectories, generating prediction residuals used for anomaly scoring.
- The framework employs a dual evaluation scheme to ensure detected anomalies are both statistically stable and physically aligned with Surrogate Safety Measures (SSMs) like Time-to-Collision (TTC) and harsh closing ratio.

---

[AMA: Adaptive Memory via Multi-Agent Collaboration](http://arxiv.org/abs/2601.20352)

- AMA (Adaptive Memory via Multi-Agent Collaboration): introduces a novel framework leveraging coordinated agents to manage memory across multiple granularities, addressing rigid retrieval and logical inconsistencies.
- The system employs a hierarchical memory design (Raw Text, Fact Knowledge, Episode Memory) and a multi-agent pipeline consisting of the Constructor, Retriever, Judge, and Refresher to support sustained LLM agent evolution.
- The Retriever dynamically routes queries, the Judge verifies relevance and consistency, and the Refresher performs targeted updates to maintain long-term memory coherence and consistency.

---

[MobileBench-OL: A Comprehensive Chinese Benchmark for Evaluating Mobile GUI Agents in Real-World Environment](http://arxiv.org/abs/2601.20335)

- MobileBench-OL: introduces a comprehensive online benchmark with 1080 tasks across 80 Chinese apps, evaluating GUI agents using five subsets and an Auto-Eval framework.
- The benchmark measures task execution, complex reasoning (Long-Horizon, GUI-Reasoning), and robustness to real-world noise (Pop-up, Delay, Repeat, Unexecuted).
- The Auto-Eval framework includes a fine-grained, instruction-based Reset Mechanism to ensure stable and repeatable real-world benchmarking by restoring device states.

---

[Demonstration-Free Robotic Control via LLM Agents](http://arxiv.org/abs/2601.20334)

- FAEA (Frontier Agent as Embodied Agent): introduces demonstration-free robotic control by applying the Claude Agent SDK infrastructure, which includes the LLM, ReAct Cycle, Tools/Tool Set T (APIs), Context $C_i$, and Simulation Interface, to iteratively synthesize manipulation policies through trial and error.
- FAEA leverages the iterative reasoning capabilities of general-purpose LLM agents, originally designed for software engineering, to perform task-level planning and strategy refinement in embodied environments.
- The approach achieves competitive success rates (84.9% on LIBERO, 85.7% on ManiSkill3, 96% on MetaWorld) compared to VLA models trained with limited demonstrations, without requiring task-specific fine-tuning.

---

[ECG-AGENT: ON-DEVICE TOOL-CALLING AGENT FOR ECG MULTI-TURN DIALOGUE](http://arxiv.org/abs/2601.20323)

- ECG-Agent: introduces the first LLM-based tool-calling agent designed for multi-turn ECG dialogue, leveraging specialized tools for precise, measurement-based responses.
- The agent utilizes a Classification Tool, a Measurement Tool (Neurokit2), and an Explanation Tool (SpectralX) to process user queries regarding ECG data and generate accurate responses.
- ECG-Agent is trained on the novel ECG-Multi-Turn-Dialogue (ECG-MTD) dataset and demonstrates comparable performance between compact on-device LLM agents (1B, 3B) and much larger models (32B).

---

[Less is More: Benchmarking LLM Based Recommendation Agents](http://arxiv.org/abs/2601.20316)

- LLM Based Recommendation Agents Benchmark: introduces a systematic evaluation of four state-of-the-art LLMs (GPT-40-mini, DeepSeek-V3, Qwen2.5-72B, Gemini 2.5 Flash) using the REGEN dataset across context lengths from 5 to 50 items, measuring Quality Score, Latency, and Token Count.
- The benchmark reveals a flat quality curve, showing no statistically significant improvement in recommendation quality despite increasing context length from 5 to 50 items.
- This finding suggests practitioners can reduce inference costs by approximately 88% by utilizing minimal context (5-10 items) without sacrificing recommendation quality, challenging the "more context is better" paradigm.

---

[Structure-constrained Language-informed Diffusion Model for Unpaired Low-dose Computed Tomography Angiography Reconstruction](http://arxiv.org/abs/2601.20304)

- SLDM (Structure-constrained Language-informed Diffusion Model): introduces a unified medical generation model for unpaired low-dose Computed Tomography Angiography (CTA) reconstruction, integrating the Diffusion Model Prior Generation Phase (Topological constraint introduction), CTA-CLIP Assistance Phase (Semantic supervision integration), and Subtraction Angiography Enhancement Module (SAEM) (Dynamic contrast enhancement).
- The approach utilizes a structure-constrained mean-reverting stochastic differential equation (SDE) to ensure structural consistency and employs CTA-CLIP, a dedicated Vision-Language Model (VLM) variant, for semantic guidance via text prompts.
- The SAEM dynamically refines the reconstructed contrast-enhanced regions by calculating the disparity between mask and filling images, enabling flexible grayscale adjustment tailored to individual vascular characteristics.

---

[Beyond the Needle's Illusion: Decoupled Evaluation of Evidence Access and Use under Semantic Interference at 326M-Token Scale](http://arxiv.org/abs/2601.20276)

- EverMemBench-S (EMB-S): introduces an adversarial Needle-in-a-Haystack benchmark built on a 326M-token MemoryBank, featuring a Query Construction Pipeline, a Reference Corpus Ladder, and a Decoupled Diagnostic Protocol.
- The benchmark evaluates evidence access and use by pairing queries with collision-tested near-miss hard negatives and gold evidence sets spanning multiple documents, stressing semantic discrimination rather than simple span localization.
- The decoupled diagnostic protocol reports evidence access via document-ID localization metrics separately from end-to-end Generative QA quality under full-context prompting, enabling consistent diagnosis across native long-context LLMs and retrieval pipelines.

---

[UNIT-BASED AGENT FOR SEMI-CASCADED FULL-DUPLEX DIALOGUE SYSTEMS](http://arxiv.org/abs/2601.20230)

- Unit-Based Agent Framework: introduces a train-free, semi-cascaded full-duplex dialogue system that decomposes complex dialogue into minimal conversational units, enabling independent processing and state transition prediction.
- The system uses a Multimodal LLM (MLLM) within a Decision Module to directly process audio input and contextual ASR transcripts, controlling state transitions (listen/speak) via `continue` or `switch` actions.
- This modular architecture, supported by VAD, SV, ASR, and TTS, achieves accurate, low-latency turn-taking by replacing the traditional ASR-to-LLM cascade.

---

[Scaling Medical Reasoning Verification via Tool-Integrated Reinforcement Learning](http://arxiv.org/abs/2601.20221)

- Med-TIV (Medical Tool-Integrated reasoning Verifier): introduces an agentic RL framework that trains LLM verifiers to iteratively query external medical corpora using tool-augmented verification and adaptive curriculum formulation.
- The framework utilizes an iterative RL paradigm requiring only trace-level supervision to progressively improve verification capabilities, focusing optimization on decision-boundary cases.
- During inference, the trained verifier guides test-time search over candidate reasoning traces generated by a frozen LLM using weighted self-consistency for improved accuracy and efficiency.

---

[Spark: Strategic Policy-Aware Exploration via Dynamic Branching for Long-Horizon Agentic Learning](http://arxiv.org/abs/2601.20209)

- SPARK (Strategic Policy-Aware exploration via Key-state dynamic branching): introduces a novel RL framework that selectively expands exploration at critical decision points using intrinsic uncertainty signals to construct hierarchical tree-structured exploration paths.
- The framework leverages Adaptive Dynamic Branching, guided by the agent's intrinsic decision-making signals (the `<explore>` tag), to prioritize sampling quality over blind coverage under constrained computational budgets.
- SPARK achieves superior success rates and robust generalization in long-horizon agentic tasks (ALFWorld, ScienceWorld, WebShop) with significantly fewer training samples compared to uniform exploration baselines.

---

[An Autonomous Agent Framework for Feature-Label Extraction from Device Dialogues and Automatic Multi-Dimensional Device Hosting Planning Based on Large Language Models](http://arxiv.org/abs/2601.20194)

- AirAgent: introduces a two-layer cooperative LLM-driven architecture, Memory-Based Tag Extraction and Reasoning-Driven Planning, to autonomously manage home air systems through comprehensive perception, reasoning, and control.
- The framework integrates multi-source inputs, including real-time environmental sensor data, user states, and domain-specific knowledge, to generate context-aware decisions across 25 complex dimensions and satisfy over 20 customized constraints.
- AirAgent utilizes a semi-streaming output mechanism that segments the LLM output into human-readable Chain-of-Thought explanations and machine-readable control instructions for enhanced interpretability and execution.

---

[Meta-Cognitive Reinforcement Learning with Self-Doubt and Recovery](http://arxiv.org/abs/2601.20193)

- Meta-Cognitive Reinforcement Learning: introduces a framework enabling an RL agent to assess, regulate, and recover its learning behavior using an internally estimated meta-trust variable driven by Value Prediction Error Stability (VPES).
- The framework employs a two-timescale architecture where a Meta-Cognitive Controller monitors internal stability signals and modulates the RL Agent's learning rate via fail-safe regulation and gradual trust recovery.
- This approach achieves system-level robustness by regulating the permissibility of learning itself, substantially reducing late-stage training failures and mitigating tail risk under reward corruption.

---

[Securing AI Agents in Cyber-Physical Systems: A Survey of Environmental Interactions, Deepfake Threats, and Defenses](http://arxiv.org/abs/2601.20184)

- SENTINEL (Systematic Evaluation and Threat-Informed NEtwork defense seLection): introduces a lifecycle-aware methodology for securing AI agents in Cyber-Physical Systems (CPS), with all six phases, where the framework integrates threat modeling, resource constraints, and operational requirements into a unified decision process.
- The survey systematically reviews security threats targeting AI agents in CPS, focusing on deepfake-driven attacks and Model Context Protocol (MCP)-mediated vulnerabilities, and proposes a defense-in-depth architecture tailored to CPS constraints.
- The architecture emphasizes provenance verification, physics-grounded trust mechanisms (like Electric Network Frequency), and continuous adaptation to counter the rapid evolution of generative AI and deepfake techniques.

---

[Who Writes the Docs in SE 3.0? Agent vs. Human Documentation Pull Requests](http://arxiv.org/abs/2601.20171)

- Agent vs. Human Documentation PR Analysis: introduces an empirical study analyzing 1,997 documentation-related Pull Requests (PRs) from the AIDev dataset to compare contributions between AI agents and human developers in SE 3.0.
- The study finds that AI agents submit substantially more documentation-related PRs than humans, indicating widespread agent use for documentation edits, with 1,478 agent-authored PRs versus 519 human-authored PRs.
- Agent-authored documentation changes are largely accepted but often receive limited human follow-up, raising concerns about review practices and documentation quality assurance.

---

[Me-Agent: A Personalized Mobile Agent with Two-Level User Habit Learning for Enhanced Interaction](http://arxiv.org/abs/2601.20162)

- Me-Agent: introduces a personalized mobile agent that adapts to user preferences through two-level habit modeling, utilizing User Preference Learning (Prompt level strategy) and Hierarchical Preference Memory (Two-level memory structure).
- UPL is a parameter-free module that uses a VLM-based Reward Model and an LLM-based Advantage stage to extract and optimize general user preferences into an Experience Pool.
- HPM stores long-term and app-specific memory externally in Level-1 and Level-2 structures, enabling personalized inference to resolve application and content ambiguity in mobile instructions.

---

[How do Agents Refactor: An Empirical Study](http://arxiv.org/abs/2601.20160)

- Agentic Refactoring Evaluation Framework: introduces an empirical study analyzing how five software development agents (LLMs) perform Java refactoring compared to human developers using RefactoringMiner and DesigniteJava 3.0.
- The study finds that agent refactorings are heavily dominated by annotation changes, unlike the diverse structural improvements typical of human developers.
- While agents perform significantly more refactoring actions, their overall impact on code smells per commit is statistically consistent with the developer baseline, except for Cursor, which significantly increases code smells.

---

[Trajectory2Task: Training Robust Tool-Calling Agents with Synthesized Yet Verifiable Data for Complex User Intents](http://arxiv.org/abs/2601.20144)

- Trajectory2Task (T2T): introduces a verifiable data generation pipeline that synthesizes complex multi-turn tool-use scenario tasks and executable trajectories under ambiguous, changing, and infeasible user intents.
- The pipeline uses an Explorer LLM for multi-turn exploration, followed by a Summarizer LLM and Validity Check LLM to convert successful trajectories into verifiable tasks paired with golden label trajectories.
- Training lightweight LLMs using Supervised Fine-Tuning (SFT) on the generated successful trajectories (Retail-3I benchmark) significantly improves robustness and generalization to unseen domains.

---

[LVLMs and Humans Ground Differently in Referential Communication](http://arxiv.org/abs/2601.19792)

- LVLMs: introduces a referential communication experiment using a factorial design (Human-Human, Human-AI, AI-Human, AI-AI) to evaluate the grounding capabilities of Large Vision Language Models (LVLM) agents (GPT-5.2) acting as Director or Matcher in a multi-turn object-matching task.
- The study finds that LVLMs, unlike human pairs, fail to establish common ground, showing decreased accuracy and lack of efficiency improvement (lexical entrainment) across repeated rounds.
- The resulting corpus of 356 dialogues demonstrates that AI partners are verbose and fail to adapt their communication strategy to increase efficiency, violating pragmatic principles.

---

[Judgelight: Trajectory-Level Post-Optimization for Multi-Agent Path Finding via Closed-Subwalk Collapsing](http://arxiv.org/abs/2601.19388)

- Judgelight: introduces a trajectory-level post-optimization layer that improves Multi-Agent Path Finding (MAPF) solution quality by formalizing the MAPF-Collapse problem and solving it exactly via an Integer Linear Programming (ILP) Formulation, using Candidate Actions, Decision Variables, Constraints, and Preprocessing Optimizations.
- The framework collapses closed subwalks in agents' trajectories to remove unnecessary or oscillatory movements, minimizing the total number of move actions while strictly preserving all feasibility constraints.
- Experiments show that Judgelight consistently reduces the solution cost (Saved SoC) by approximately 20% to 40%, particularly for learning-based MAPF solvers, while maintaining practical runtime performance.

---

#### 27th January 2026

[OptAgent: an Agentic AI framework for Intelligent Building Operations](http://arxiv.org/abs/2601.20005)

- OptAgent (Agentic AI-enabled PIML Framework): introduces an end-to-end framework coupling a Multi-Agent Agentic AI Decision Layer and a Physics-Informed Machine Learning Runtime Environment (BESTOpt) via the Model Context Protocol (MCP) for autonomous building energy operations.
- The framework uses a hierarchical multi-agent system, including a Concierge Agent, an Orchestrator Agent, and 11 Specialist Agents, to translate natural language requests into executable multi-step workflows across building, HVAC, DER, and grid domains.
- A large-scale benchmark evaluation across 3,975 cases demonstrated that centralized two-stage planning achieves the best accuracy-efficiency tradeoff, and orchestrator capability dominates overall system performance.

---

[Taxonomy of the Retrieval System Framework: Pitfalls and Paradigms](http://arxiv.org/abs/2601.20131)

- Retrieval System Framework (RSF): introduces a taxonomy of embedding-based retrieval systems structured into four layers: Representation, Granularity, Orchestration, and Robustness.
- The framework analyzes architectural trade-offs, contrasting Bi-encoders (scalability) with Cross-encoders (expressivity) and Late Interaction models (bridging the efficiency-effectiveness gap).
- The paper details strategies for document chunking, multi-vector representations, LLM-guided hierarchical retrieval, and architectural mitigations for domain generalization and temporal drift.

---

[Beyond Bug Fixes: An Empirical Investigation of Post-Merge Code Quality Issues in Agent-Generated Pull Requests](http://arxiv.org/abs/2601.20109)

- Differential SonarQube Analysis (DSA): introduces an empirical investigation analyzing 1,210 merged agent-generated bug-fix PRs from the AIDev dataset using SonarQube to identify newly introduced post-merge code quality issues.
- The study performs a differential analysis comparing base and merged commits across five agents (OpenAI Codex, Copilot, Devin, Cursor, and Claude Code) to characterize issue frequency, density, and severity profiles.
- Results indicate that differences in raw issue counts across agents are largely driven by PR volume and code churn, with Code Smells dominating critical and major severities, while Bugs are less frequent but often severe (BLOCKER).

---

[Are We All Using Agents the Same Way? An Empirical Study of Core and Peripheral Developers' Use of Coding Agents](http://arxiv.org/abs/2601.20106)

- Empirical Study of Coding Agent Usage: introduces, "an empirical study of 9,427 resolved agentic PRs on GitHub", with all Data Collection (Extracting PRs from GitHub)/Developer Selection and Classification (Categorizing Core/Peripheral Developers)/Final Dataset (9,427 resolved agentic PRs)/Usage of Agentic PRs (Analyzing frequency and purpose)/Review of Agentic PRs (Analyzing intensity and issues raised)/Modifications of Agentic PRs (Measuring intensity and types of changes)/CI Outcomes in Agentic PRs (Evaluating merge verification checks)-components, where "the study investigates how core and peripheral developers use, review, modify, and verify agent-generated contributions across the pull request lifecycle".
- Core developers focus agent delegation on documentation and testing, engage more in review discussions, and consistently require passing Continuous Integration (CI) verification before merging agentic PRs.
- Peripheral developers delegate tasks evenly across bug fixing, feature addition, documentation, and testing, and are nearly twice as likely to merge agentic PRs without running CI checks.

---

[Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control](http://arxiv.org/abs/2601.20090)

- CCG (Conformal Counterfactual Generation): introduces a framework for reliable counterfactual generation in LLM-based autonomous control systems by modeling the closed-loop interaction as an SCM and leveraging test-time scaling to produce a set of counterfactual outcomes with formal reliability guarantees.
- The approach uses probabilistic abduction, implemented via a Neural Posterior Estimation network, to infer latent environment noise variables ($U_Z$) from a factual episode, enabling the simulation of counterfactual environment responses under a different user intent ($X'$).
- The framework includes two distinct LLM components—an action generator and a report generator—and utilizes a calibrated conformal language modeling methodology to ensure the generated set of counterfactual reports contains the true outcome with high probability.

---

[Insight Agents: An LLM-Based Multi-Agent System for Data Insights](http://arxiv.org/abs/2601.20048)

- IA (Insight Agents): introduces an LLM-based hierarchical multi-agent system built on a plan-and-execute paradigm for providing personalized data and business insights to E-commerce sellers.
- The architecture features a Manager Agent handling OOD detection and routing, overseeing two worker agents: the Data Presenter Agent and the Insight Generator Agent.
- IA achieves high accuracy (89.5%) and low latency (P90 below 15s) by utilizing specialized lightweight models for OOD detection and routing, and concurrent execution of worker branches.

---

[Obviously Strategy-Proof Multi-Dimensional Allocation and the Value of Choice](http://arxiv.org/abs/2601.20035)

- EOPR (Ex-Post Optimal Polarized Ray Mechanism): characterizes the set of obviously strategy-proof (OSP) trading mechanisms for allocating heterogeneous tasks among agents with private preferences and status-quo constraints, showing that all OSP mechanisms are effectively polarized ray mechanisms.
- The paper proves that restricting attention to OSP trading mechanisms is generally without loss of optimality when determining the value of choice compared to the full class of Bayesian incentive compatible mechanisms.
- EOPR mechanisms are robust to preference misspecification and guarantee a weak improvement over the status quo, providing a practical candidate for multi-dimensional task allocation problems.

---

[What is the AGI in Offensive Security?](http://arxiv.org/abs/2601.19968)

- SSC: introduces a formal model where offensive security tasks are reduced to symbolic language manipulation, utilizing Target System (Mathematical model), State Machine (Models digital system), Set of States (System configurations), Input Alphabet (Input symbols), Output Alphabet (Output symbols), Transition Function (I/O function), Hacker Policy (Algorithmic procedure), Interaction Transcript (Sequence of I/O), Encoding Function (Maps transcript to string), and LLM (Approximates policy behavior).
- The model formalizes the target system as a state machine (M) and the hacker as an interactive symbolic agent ($\pi$), demonstrating that all interactions can be encoded as finite strings.
- The paper concludes that because hacking is fundamentally symbolic string-to-string computation, LLMs are naturally suited to approximate the attacker's policy and decision function.

---

[UNSUPERVISED LEARNING OF EFFICIENT EXPLORATION: PRE-TRAINING ADAPTIVE POLICIES VIA SELF-IMPOSED GOALS](http://arxiv.org/abs/2601.19810)

- ULEE (Unsupervised Learning of Efficient Exploration): introduces an unsupervised meta-learning method combining an in-context learner with an adversarial goal-generation strategy guided by post-adaptation performance estimates.
- The method trains a Pre-trained Policy ($\pi$) to maximize expected discounted lifetime return over multiple episodes using self-imposed goals selected based on intermediate difficulty.
- ULEE utilizes a Difficulty Predictor network ($d_{\phi}$) and an adversarial Goal-search Policy ($\pi_{gs}$) to maintain training at the frontier of the agent's capabilities in XLand-MiniGrid environments.

---

[Nonequilibrium phase transitions in a racism-spreading model with interaction-driven dynamics](http://arxiv.org/abs/2601.19806)

- Three-state compartmental model: introduces a diffusion model to study the spread and suppression of racist content on social networks, utilizing Susceptibles (S), Infected (I), and Deniers (D) compartments, governed by interaction-driven transitions.
- The model is analyzed using coupled differential equations for fully-connected networks and agent-based simulations on Barabási-Albert (BA) scale-free and Watts-Strogatz (WS) small-world networks.
- The system exhibits three stationary regimes: two racism-free absorbing states and one active phase with persistent racist content, with transitions depending on network topology and interaction parameters.

---

[Youtu-VL: Unleashing Visual Potential via Unified Vision-Language Supervision](http://arxiv.org/abs/2601.19798)

- Youtu-VL: introduces a framework leveraging the VLUAS (Vision-Language Unified Autoregressive Supervision) paradigm, which integrates a Vision Encoder (SigLIP-2 variant), Vision-Language Projector (Spatial Merge Projector/MLP), and LLM (Youtu-LLM Decoder) using a Synergistic Vision Tokenizer (fuses semantic/geometric features) and NTP-M (multi-label/multi-task objective) to achieve unified autoregressive supervision for visual and linguistic content.
- The VLUAS paradigm fundamentally shifts the optimization objective from "vision-as-input" to "vision-as-target" by treating visual tokens as supervisory targets, mitigating text-dominant optimization bias and enforcing the retention of fine-grained visual details.
- Youtu-VL enables a standard VLM architecture to natively perform a comprehensive range of vision-centric tasks, including dense prediction (segmentation, depth) and localization (grounding, detection), without task-specific modules.

---

[Component-Aware Pruning Framework for Neural Network Controllers via Gradient-Based Importance Estimation](http://arxiv.org/abs/2601.19794)

- Component-Aware Pruning Framework: introduces a structured pruning approach for Neural Network Controllers using Gradient Accumulation (Measures instantaneous sensitivity), Fisher Information (Quantifies loss sensitivity/curvature), and Bayesian Uncertainty (Estimates long-term activity/importance) to guide compression decisions.
- The framework explicitly distinguishes between component-specific and inter-component coupling groups, enabling granular pruning strategies evaluated on a Multi-Component Neural Architecture (MCNA) autoencoder and a Temporal Difference Learning for Model Predictive Control (TD-MPC) agent.
- Experimental results demonstrate that group importance is dynamic and architecture-dependent, challenging static assumptions that coupling groups or early layers are universally critical for maintaining stability and performance.

---

[CASTER: Breaking the Cost-Performance Barrier in Multi-Agent Orchestration via Context-Aware Strategy for Task Efficient Routing](http://arxiv.org/abs/2601.19793)

- CASTER (Context-Aware Strategy for Task Efficient Routing): introduces a lightweight neural router for dynamic model selection in graph-based MAS, integrating semantic embeddings and structural meta-features to estimate task difficulty.
- The system employs a Dual-Branch Feature Fusion Network for routing and self-optimizes through a Cold Start to Iterative Evolution paradigm using on-policy negative feedback derived from routing failures.
- CASTER reduces inference cost by up to 72.4% compared to strong-model baselines while maintaining comparable success rates across software engineering, data analysis, scientific discovery, and cybersecurity domains.

---

[Reimagining Peer Review Process Through Multi-Agent Mechanism Design](http://arxiv.org/abs/2601.19778)

- Three-Pillar Architecture: introduces a stochastic multi-agent system model for peer review, applying multi-agent reinforcement learning (MARL) to design incentive-compatible protocols across three interventions.
- The architecture includes a Credit Economy for managing submissions and review rewards, MARL Assignment for dynamic reviewer allocation, and Hybrid Verification for assessing review quality.
- A closed-loop adaptive system is established where review quality verification directly informs credit issuance and price dynamics, ensuring sustainable peer review incentives.

---

[Strong Reasoning Isn't Enough: Evaluating Evidence Elicitation in Interactive Diagnosis](http://arxiv.org/abs/2601.19773)

- REFINE (Reasoning-Enhanced Feedback for INformation Elicitation): introduces a feedback-driven strategy utilizing diagnostic verification to guide the Information Collector (acquires evidence), Evidence Organizer (consolidates findings), Diagnosis Reasoner (produces hypothesis), and Diagnosis Verifier (checks evidence sufficiency) in interactive medical diagnosis.
- The research establishes an interactive evaluation framework using a Doctor Agent, Simulated Patient, and Simulated Reporter, grounded in atomic evidences, and introduces the Information Coverage Rate (ICR) metric to quantify evidence elicitation completeness.
- Evaluation on the new EviMed benchmark, spanning diverse medical conditions, reveals that strong diagnostic reasoning in LLMs does not guarantee effective information collection, which is identified as a primary performance bottleneck.

---

[Agentic Design Patterns: A System-Theoretic Framework](http://arxiv.org/abs/2601.19752)

- System-Theoretic Agent Architecture (STAA): introduces a principled methodology for engineering robust AI agents by deconstructing the system into RWM (Decision-making nucleus), PG (Process raw inputs), AE (Execute actions), LA (Continuous improvement), IAC (Peer-to-peer interaction), and 12 ADPs (Structural solutions).
- The framework addresses systemic fragilities in FM-based agents, such as hallucination and poor reasoning, by providing a layered organization derived from system theory.
- The paper presents a catalogue of 12 Agentic Design Patterns (ADPs) mapped to architectural components to offer reusable, structural solutions for recurring agent design problems, demonstrated via a case study on the ReAct framework.

---

[VERI-SURE: A Contract-Aware Multi-Agent Framework with Temporal Tracing and Formal Verification for Correct RTL Code Generation](http://arxiv.org/abs/2601.19747)

- VERI-SURE (Contract-Aware Multi-Agent Framework): introduces a multi-agent framework that distills natural language specifications into a structured Design Contract to align agent intent and uses a multi-branch verification pipeline for correct Register-Transfer Level (RTL) code generation.
- The framework integrates simulation, trace-driven temporal analysis, and formal verification (assertion-based checking and Boolean equivalence proofs) within an automated generate-verify-debug loop.
- VERI-SURE employs a dependency-slicing-guided patching mechanism to perform precise, localized repairs, avoiding costly whole-file regeneration and reducing regression risks common in LLM-based RTL design.

---

[Quantum Circuit Pre-Synthesis: Learning Local Edits to Reduce T-count](http://arxiv.org/abs/2601.19738)

- Q-PRESYN (Quantum Circuit Pre-Synthesis): introduces a strategy using an Optimization Loop guided by an RL Agent to generate a Plan of local edits, yielding an Equivalent Representation that minimizes the T-count after processing by a Blackbox Synthesis Algorithm.
- The approach formalizes the search for advantageous merge sequences as a plan optimization task, leveraging Proximal Policy Optimization (PPO) to explore the space of equivalent circuit representations.
- Q-PRESYN achieves up to a 20% reduction in T-count on circuits up to 25 qubits by modifying circuit structure through unitary-preserving merge operations prior to synthesis.

---

[Who Said CVE? How Vulnerability Identifiers Are Mentioned by Humans, Bots, and Agents in Pull Requests](http://arxiv.org/abs/2601.19636)

- Vulnerability ID Usage Analysis: introduces an empirical study comparing how Humans, Bots, and Autonomous Agents mention standardized Vulnerability IDs (CVE, CWE, GHSA) within GitHub pull requests using an augmented dataset derived from AIDev.
- The analysis reveals that bots account for the majority (69.1%) of ID mentions, primarily concentrated in pull request descriptions for automated dependency updates and security audits.
- Human and agent mentions are less frequent but span a wider range of artifacts (commits, titles, discussions) and support vulnerability fixes, maintenance work, and contextual reasoning.

---

[SAFE EXPLORATION VIA POLICY PRIORS](http://arxiv.org/abs/2601.19612)

- SOOPER (Safe Online Optimism for Pessimistic Expansion in RL): introduces a scalable model-based RL algorithm for safe exploration in constrained Markov decision processes that uses prior policies as safe priors and employs optimistic planning in a simulated environment.
- The approach guarantees safety throughout learning by pessimistically invoking the safe prior policy during online data collection if the accumulated cost exceeds the safety budget.
- SOOPER establishes convergence to an optimal policy by bounding its cumulative regret, demonstrating sublinear cumulative regret while maintaining constraint satisfaction.

---

[ComAgent: Multi-LLM based Agentic AI Empowered Intelligent Wireless Networks](http://arxiv.org/abs/2601.19607)

- ComAgent (Multi-LLM based Agentic AI Framework): introduces an agentic AI framework that coordinates specialized LLM agents (Literature, Planning, Coding, Scoring) within a recursive Perception-Planning-Action-Reflection cycle to transform high-level wireless optimization intents into executable, verified simulation pipelines.
- The framework autonomously manages the end-to-end optimization workflow, anchoring decisions in retrieved domain knowledge and leveraging scoring-based feedback for iterative self-correction to ensure constraint satisfaction and physical feasibility.
- ComAgent achieves performance comparable to expert-designed baselines in complex MIMO SWIPT beamforming optimization and demonstrates substantially higher solution success rates than monolithic LLMs across diverse wireless tasks.

---

[Toward Architecture-Aware Evaluation Metrics for LLM Agents](http://arxiv.org/abs/2601.19583)

- AAEM (Architecture-Aware Evaluation Method): introduces a systematic, architecture-informed approach that links LLM agent architectural components to their observable behaviors and corresponding evaluation metrics.
- The method addresses the lack of a clear link between observable behaviors, agent architecture, and evaluation metrics, enabling more targeted and diagnostic evaluation of LLM-based agents.
- The approach consolidates architectural foundations into a unified taxonomy and validates the method by applying it to real-world agents like SWE-Agent and MetaGPT to demonstrate improved diagnostic clarity and reduced metric arbitrariness.

---

[YUNQUE DEEPRESEARCH TECHNICAL REPORT](http://arxiv.org/abs/2601.19578)

- Yunque DeepResearch: introduces a hierarchical, modular, and robust multi-agent framework characterized by a Main Agent (central executive), a Context Manager (dual-level memory), an Atomic Capability Pool (specialized execution units), and a Supervisor (robustness and error correction).
- The framework addresses cognitive overload and systemic fragility in long-horizon tasks by dynamically managing context through sub-goal folding and employing an adaptive self-correction mechanism.
- The Atomic Capability Pool includes specialized LLM-agents, such as the Browser-Use GUI Agent and the Data Analysis Agent, which handle complex, domain-specific execution tasks.

---

[Learning Adaptive Parallel Execution for Efficient Code Localization](http://arxiv.org/abs/2601.19568)

- FuseSearch: introduces a code localization agent that achieves superior quality-efficiency trade-offs through learned adaptive parallel execution, utilizing LLM Agent, Parallel Tool Use, and a Minimalist Tool Set, optimized via SFT and RL with a Joint Reward based on F1 Score and Efficiency e.
- The approach reformulates parallel code localization as a joint quality-efficiency optimization task by defining tool efficiency as the ratio of unique information gain to invocation count, significantly reducing redundant tool calls.
- FuseSearch-4B achieves 84.7% file-level F1 score and 93.6% speedup, utilizing 67.7% fewer turns and accelerating downstream agent workflows by 28.5% without sacrificing success rates.

---

[ALRM: Agentic LLM for Robotic Manipulation](http://arxiv.org/abs/2601.19510)

- ALRM (Agentic LLM for Robotic Manipulation): introduces an LLM-driven agentic framework for robotic manipulation, integrating policy generation with agentic execution via a ReAct-style loop and supporting Code-as-Policy and Tool-as-Policy modes.
- The architecture consists of a Task Planner Agent for high-level subtask decomposition, a Task Executor Agent for action generation, and an API Server interfacing with the Gazebo/ROS/MoveIt simulation environment.
- ALRM is evaluated on a novel benchmark of 56 multistep, high-level manipulation tasks emphasizing linguistic diversity and complex reasoning across three distinct simulated environments.

---

[Automated Safety Benchmarking: A Multi-agent Pipeline for LVLMs](http://arxiv.org/abs/2601.19507)

- VLSafetyBencher: introduces the first automated system for LVLM safety benchmarking, utilizing four collaborative agents: Data Preprocessing (Cleans/filters raw data), Generation (Synthesizes image-question pairs), Augmentation (Enhances diversity/harmfulness), and Selection (Optimizes/selects final benchmark).
- The system constructs high-quality safety benchmarks by employing three cross-modal synthesis strategies—Modality Dependence, Complementarity, and Conflict—to ensure harmfulness detection requires genuine multimodal understanding.
- VLSafetyBencher significantly reduces the time and cost of benchmark creation, constructing an optimized benchmark within one week that demonstrates stronger discriminative power compared to existing benchmarks.

---

[Reinforcement Learning Goal-Reaching Control with Guaranteed Lyapunov-Like Stabilizer for Mobile Robots](http://arxiv.org/abs/2601.19499)

- LLS (Lyapunov-Like Stabilizer): introduces a formal-guaranteed goal-reaching RL framework for large WMRs operating in unstructured environments, utilizing a Benchmark RL Policy ($\pi^0$) (Goal-reaching policy), a Lyapunov-Like Stabilizer Agent ($\pi^L$) (Policy supervisor), a Constrained Critic Update (Guaranteed decrease), Reference Memory (Stored state-action pair), Action Selection (Greedy or fallback), Fallback Mechanism (Reverts to $\pi^0$), WMR Environment (6000 kg skid-steer robot), and Low-level DNN Control (Velocity tracking).
- The framework integrates a Lyapunov-like stabilizer layer into the benchmark RL policy to provide formal convergence guarantees to the goal set without requiring prior knowledge of the Lyapunov function.
- The RL policy uses acceleration-based actions and 12 reward terms to ensure smooth, non-oscillatory motion, critical for maintaining visual-based pose estimation in challenging, slip-prone terrain.

---

[AACR-Bench: Evaluating Automatic Code Review with Holistic Repository-Level Context](http://arxiv.org/abs/2601.19494)

- AACR-Bench: introduces a comprehensive, multilingual, repository-level context-aware benchmark for evaluating Automated Code Review (ACR) systems, utilizing a hybrid AI-assisted/Expert-verified annotation pipeline.
- The benchmark provides full cross-file context across 10 mainstream programming languages and significantly increases defect coverage by 285% compared to traditional datasets.
- Extensive evaluations reveal that ACR performance is significantly impacted by the granularity of context, the choice of retrieval methods, and the LLM usage paradigm (Agent vs. traditional approaches).

---

[AoI-Driven Queue Management and Power Control in V2V Networks: A GNN-Enhanced MARL Approach](http://arxiv.org/abs/2601.19372)

- GNN-enhanced CTDE framework based on MAPPO: introduces a multi-agent reinforcement learning approach to jointly optimize AoI-aware queue management (packet dropping) and transmit power control in V2V networks.
- The framework utilizes a Graph Neural Network (GNN) backbone, specifically GraphSAGE, to extract topology-aware embeddings from large-scale fading information, enabling scalable coordination under centralized training and decentralized execution (CTDE).
- Agents employ a hybrid discrete-continuous action space to manage packet-level queues and allocate power, achieving minimized long-term average Age of Information (AoI) across diverse network conditions.

---

[CHEHAB RL: Learning to Optimize Fully Homomorphic Encryption Computations](http://arxiv.org/abs/2601.19367)

- CHEHAB RL: introduces a novel framework that leverages deep reinforcement learning (RL) to automate Fully Homomorphic Encryption (FHE) code optimization, including the IR, RL TRS (RL-guided Term Rewriting System), Classical optimizations, Rotation Key Selection, Code Generator, and SEAL backend.
- The framework formulates FHE optimization as a sequential decision-making problem, training an RL agent using a hierarchical policy network and an LLM-synthesized dataset to minimize instruction latency and noise growth.
- CHEHAB RL generates vectorized FHE code that is 5.3x faster in execution and 27.9x faster in compilation compared to the state-of-the-art Coyote compiler.

---

[SELF-SUPERVISED PATH PLANNING IN UNSTRUCTURED ENVIRONMENTS VIA GLOBAL-GUIDED DIFFERENTIABLE HARD CONSTRAINT PROJECTION](http://arxiv.org/abs/2601.19354)

- Self-Supervised Planning Framework (SSPF): introduces a self-supervised path planning system coupling global topological guidance with a differentiable hard constraint projection layer for runtime assurance.
- The framework utilizes a Global-Guided Artificial Potential Field (G-APF) to provide dense supervision, mitigating data scarcity and guiding the agent out of local minima.
- An Adaptive-Depth Neural Projection (AdaNP) layer iteratively projects the coarse network output onto a feasible manifold, rigorously enforcing kinematic and safety constraints.

---

[BALANCING SUSTAINABILITY AND PERFORMANCE: THE ROLE OF SMALL-SCALE LLMS IN AGENTIC ARTIFICIAL INTELLIGENCE SYSTEMS](http://arxiv.org/abs/2601.19311)

- Multi-Agent Framework (MAF): introduces a comparative analysis of small-scale open-weights LLMs against a closed-source baseline (GPT-4o) within a real-world MAF, utilizing Client (User interaction initiator), Prompt Injection (Context enrichment), LLM (Decision making), Guardrail (Response validation), Overall Metric (Unified ranking score), and ML-Energy Benchmark (Energy consumption measurement) components.
- The research quantifies the trade-offs among Environmental Impact (energy consumption), User Experience (decode latency), and Output Quality (F1-score and LLM-as-a-Judge) across 28 LLMs of varying sizes and compression techniques.
- Results demonstrate that smaller open-weights LLMs, particularly Qwen3 MoE models, can achieve substantial energy reduction (up to 70%) with minimal impact on output quality compared to GPT-4o, supporting the adoption of smaller models for responsible AI.

---

[Curiosity Driven Knowledge Retrieval for Mobile Agents](http://arxiv.org/abs/2601.19306)

- CDKR (Curiosity Driven Knowledge Retrieval): introduces a framework for mobile agents that formalizes epistemic uncertainty during execution as a curiosity score to trigger external knowledge retrieval.
- The system uses a tail-adjusted Jensen Shannon (JS) divergence to measure the information gain between prior and posterior latent state distributions, quantifying the agent's knowledge deficit.
- Retrieved external knowledge from documentation, source code, and historical traces is consolidated into structured, modular AppCards to enhance planning reliability and task success, achieving 88.8% success rate on AndroidWorld with GPT-5.

---

[Queue Length Regret Bounds for Contextual Queueing Bandits](http://arxiv.org/abs/2601.19300)

- CQB (Contextual Queueing Bandits): introduces a novel context-aware framework for scheduling and learning unknown service rates using a logistic model, evaluated via queue length regret, and analyzed using policy-switching queues and a coupling argument.
- The framework addresses queue state misalignment, where heterogeneous job contexts lead to different job processing orders between the agent's policy and the optimal policy.
- The proposed algorithms, CQB-$\epsilon$ and CQB-Opt, achieve queue length regret bounds of $\tilde{O}(T^{-1/4})$ for stochastic contexts and $O(\log^2 T)$ for adversarial contexts, respectively.

---

[Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](http://arxiv.org/abs/2601.19293)

- Reinforced Rate Control Framework (RL-based CMDP): introduces a reinforcement learning framework for Neural Video Compression (NVC) that jointly determines bitrate allocation and coding parameters using a dynamic policy optimized for long-term rate-distortion performance.
- The approach utilizes an enhanced Actor-Critic architecture with a neural spatiotemporal state extractor and a tempered reward mechanism to model complex inter-frame rate and distortion dependencies.
- This CMDP-based RL scheme achieves superior rate accuracy (1.20% average relative bitrate error) and significant bitrate savings (up to 13.45%) across diverse NVC architectures compared to existing methods.

---

[MetaGen: Self-Evolving Roles and Topologies for Multi-Agent LLM Reasoning](http://arxiv.org/abs/2601.19290)

- MetaGen: introduces a training-free framework that adapts LLM role specifications and communication topology during inference via query-conditioned role generation and a self-evolving graph orchestration loop.
- The framework uses an Architect Agent to synthesize roles for a Dynamic Role Pool and constructs an Initial Execution Graph around a minimal backbone topology.
- MetaGen supports Intra-task Evolution using execution feedback to update role prompts and adjust structural decisions, complemented by Inter-task Evolution for accumulating cross-instance priors and solidifying verified roles.

---

[Understanding Dominant Themes in Reviewing Agentic AI-authored Code](http://arxiv.org/abs/2601.19287)

- LLM-Powered Review Theme Annotation Pipeline: introduces a large-scale empirical study analyzing 19,450 inline review comments on agent-authored Pull Requests (PRs) using a derived 12-theme taxonomy and an LLM-based annotation pipeline.
- The study evaluates the annotation performance of an open-source LLM (Gemma 3:12B) against human annotations, achieving high Exact Match (78.63%) and strong Jaccard similarity (0.76) at the PR level.
- Analysis of review themes reveals that functional correctness (feat) is dominant, but documentation, styling, testing, and security issues are critical factors distinguishing accepted versus rejected PRs.

---

[Reinforcement Learning for Enhanced Advanced QEC Architecture Decoding](http://arxiv.org/abs/2601.19279)

- SPA-MARL (Synergy-aware Parallel-Agent Multi-Agent Reinforcement Learning): introduces a distributed QEC framework that uses a learnable Synergy Score ($\lambda(s)$) Network to dynamically decompose the decoding problem, enabling algorithm-hardware co-design across a Multi-QPU Distributed System Architecture.
- The framework employs Agent-X and Agent-Z for syndrome checks, utilizing Joint Q-function Decomposition controlled by $\lambda(s)$ to balance correction accuracy against communication cost via the Quantum Switch/Flying Qubit Channel.
- Validation demonstrates that this syndrome-dependent decomposition achieves superior performance (7.4% improvement over QMIX) and robust scalability across diverse hardware constraints and code distances.

---

[A Reconfigurable Framework for AI-FPGA Agent Integration and Acceleration](http://arxiv.org/abs/2601.19263)

- AI-FPGA Agent: introduces an agent-driven framework that simplifies the integration and acceleration of deep neural network inference on FPGAs by dynamically partitioning AI models and scheduling compute-intensive layers for hardware offload.
- The system employs a runtime software agent utilizing Q-learning to dynamically determine the optimal execution policy (CPU versus FPGA offload) based on real-time system states and performance characteristics.
- The framework achieves over 10x latency reduction compared to CPU baselines and 2-3x higher energy efficiency than GPU implementations for image classification while preserving model accuracy using 8-bit quantization.

---

[GhostUI: Unveiling Hidden Interactions in Mobile UI](http://arxiv.org/abs/2601.19258)

- GHOSTUI: introduces a dataset and framework for systematically discovering and documenting hidden interactions in mobile UIs, characterized by the absence of visual affordances and gesture-driven activation.
- The approach uses an automated UI probing tool to execute six fundamental gesture types across 81 popular Android applications, capturing before-and-after screenshots, view hierarchies, and gesture metadata.
- Fine-tuning VLMs (GPT-4o, Qwen2.5-VL) on GHOSTUI significantly improves performance in Hidden Interaction Prediction and UI Transition Prediction tasks, demonstrating the importance of structural UI information.

---

[GLOVE: Global Verifier for LLM Memory-Environment Realignment](http://arxiv.org/abs/2601.19249)

- GLOVE (Global Verifier): introduces a memory validation framework that establishes a relative notion of truth by actively probing the environment to detect and correct memory-environment misalignment.
- The framework integrates into the LLM agent-environment interaction loop, operating via three phases: Cognitive Dissonance Detection, Relative Truth Formation through active probing, and Memory-Environment Realignment.
- GLOVE enables LLM agents to adapt robustly to dynamic environmental drifts (explicit structural changes and implicit logic changes) without relying on external ground-truth supervision or strong internal introspection.

---

[MATA: A TRAINABLE HIERARCHICAL AUTOMATON SYSTEM FOR MULTI-AGENT VISUAL REASONING](http://arxiv.org/abs/2601.19204)

- MATA (Multi-Agent hierarchical Trainable Automaton): introduces a hierarchical finite-state automaton for visual reasoning, featuring a trainable hyper agent, shared memory, Oneshot Reasoner, Stepwise Reasoner, and Specialized Agent, supervised by the MATA-SFT-90K dataset.
- The system uses the LLM-based Hyper Agent as a transition policy to manage collaboration and competition among agents, selecting the optimal next state based on the shared memory context.
- MATA achieves state-of-the-art results across multiple visual reasoning benchmarks by replacing inflexible rule-based pipelines with a data-driven, error-aware, and dynamic transition policy.

---

#### 26th January 2026

[a³-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks](http://arxiv.org/abs/2601.18754)

- a³-SecBench (A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks): introduces a large-scale evaluation suite using a Security Overlay Generation (Creates adversarial scenarios) and a Layered Attack Taxonomy (Spans seven autonomy layers) to assess LLM Agent (Cognitive controller) behavior under adversarial conditions.
- The framework augments benign multi-turn conversational UAV missions with 20,000 validated security overlay attack scenarios injected exclusively at the observation level, targeting seven autonomy layers including sensing, perception, planning, control, communication, edge/cloud, and LLM reasoning.
- Performance is systematically evaluated across three orthogonal dimensions—security (detection/attribution), resilience (safe degradation), and trust (reliable tool usage)—revealing wide performance disparities among 23 state-of-the-art LLMs.

---


[DMAP: Human-Aligned Structural Document Map for Multimodal Document Understanding](http://arxiv.org/abs/2601.18203)

- DMAP (Human-Aligned Structural Document Map for Multimodal Document Understanding): introduces a framework for multimodal document QA using the Structured-Semantic Understanding Agent (constructs DMAP) and the Reflective Reasoning Agent (structure-aware reasoning).
- The SSUA constructs DMAP, a hierarchical representation integrating textual, tabular, and visual elements, explicitly modeling section-level, page-level, and element-level relationships.
- The RRA utilizes DMAP via Tri-Path Retrieval (structured, textual, visual) and Reflective Generation, employing an LLM-based evaluator for iterative refinement and structure-aware answer synthesis.

---

[Mitigating the OWASP Top 10 For Large Language Models Applications using Intelligent Agents](http://arxiv.org/abs/2601.18105)

- Intelligent Agent Security Framework (IASF): introduces a novel architecture leveraging LLM-enabled intelligent agents, the AutoGen framework, and Retrieval Augmented Generation (RAG) to mitigate security risks outlined in the OWASP Top 10 for LLM applications.
- The proposed architecture utilizes a Commander Agent to orchestrate interactions between a Business Agent (responsible for generating answers) and a Security & Compliance Agent (responsible for input and output validation).
- The specialized agents employ RAG capabilities to access offline organizational knowledge, including business data and security policies, ensuring LLM responses are accurate and compliant with enterprise standards.

---


[Let's Make Every Pull Request Meaningful: An Empirical Analysis of Developer and Agentic Pull Requests](http://arxiv.org/abs/2601.18749)

- Empirical Analysis Framework (EAF): introduces an empirical analysis of 40,214 Pull Requests (PRs) from the AIDEV dataset to compare merge outcomes between human and AI agent-authored PRs using logistic regression models.
- The study extracts 64 features across six families, finding that submitter attributes dominate merge outcomes for both human and agentic PRs.
- Results indicate that increased review activity is associated with higher merge likelihood for human PRs but lower likelihood for agentic PRs, suggesting distinct optimal review workflows.

---


[Are Conversational AI Agents the Way Out? Co-Designing Reader-Oriented News Experiences with Immigrants and Journalists](http://arxiv.org/abs/2601.18772)

- CRONE: introduces a co-design methodology involving immigrant readers and journalists to define four metaphors for conversational AI agents: Data Decoder (interprets data), Connection Informer (locates self-connections), Empathetic Friend (experiences empathy), and Trajectory Witness (reflects on past trajectories).
- The approach addresses the "unaddressed-or-unaccountable" paradox by defining human-AI coordination workflows where journalists provide oversight and LLM agents conditionally augment news content for marginalized readers.
- The research identifies misalignments in priorities between journalists (civic value, careful language) and immigrant readers (relevance, local context, layered reading) regarding news consumption and production.

---


[WHY KEEP YOUR DOUBTS TO YOURSELF? TRADING VISUAL UNCERTAINTIES IN MULTI-AGENT BANDIT SYSTEMS](http://arxiv.org/abs/2601.18735)

- Agora: introduces a framework that reframes multi-agent coordination as a decentralized market for uncertainty, utilizing a market-aware broker and a profitability-driven trading protocol to achieve cost-efficient equilibria.
- The system formalizes epistemic uncertainty into a structured, tradable asset across perceptual, semantic, and inferential dimensions, enabling agents to trade uncertainty based on rational economic rules.
- Experiments on five multimodal benchmarks demonstrate that Agora outperforms strong VLMs and heuristic multi-agent strategies, achieving up to +8.5% accuracy while reducing cost by over 3x.

---

[Advances and Innovations in the Multi-Agent Robotic System (MARS) Challenge](http://arxiv.org/abs/2601.18733)

- MARS Challenge: introduces the evaluation of multi-agent robotic systems focusing on planning and control, utilizing the Activation Agent (Selects robots), Planning Agent (Generates action plan), Monitor Agent (Verifies plan), VLM (Iterative refinement), and Router (Predicts routing weights).
- The challenge evaluates high-level embodied planning using VLMs for task coordination and low-level policy execution for robotic manipulation in dynamic, multi-agent environments.
- Top solutions leverage iterative planning (self-correction) to reduce long-horizon errors and structured coordination (combinatorial experts or shared grounding) to improve multi-agent reliability and scalability.

---

[TEA-Bench: A Systematic Benchmarking of Tool-enhanced Emotional Support Dialogue Agent](http://arxiv.org/abs/2601.18700)

- TEA-Bench (Tool-enhanced Emotional Support Dialogue Agent Benchmark): introduces the first interactive benchmark for evaluating tool-augmented emotional support agents, featuring a Tool-augmented Agent (LLM), MCP (Tool exposure protocol), HDM (Factual grounding verification), Simulated User (LLM) (Interactive dialogue partner), LLM Evaluator (Dialogue quality scoring), External Tools (Situational knowledge retrieval), and TEA-Dialog (Tool-enhanced dialogue dataset).
- The benchmark assesses agents in multi-turn Emotional Support Conversation (ESC) based on empathetic support quality and factual grounding, specifically investigating how external tools mitigate hallucination.
- Experiments across nine LLMs reveal that tool augmentation improves ESC performance and reduces hallucination, with gains strongly dependent on the model's capacity and effective tool utilization, leading to the release of the TEA-Dialog dataset.

---

[FADEMEM: BIOLOGICALLY-INSPIRED FORGETTING FOR EFFICIENT AGENT MEMORY](http://arxiv.org/abs/2601.18642)

- FadeMem: introduces a biologically-inspired agent memory architecture that implements differential decay rates across a dual-layer memory hierarchy, utilizing adaptive memory fusion and LLM-guided conflict resolution.
- The system models memory decay using differential exponential functions modulated by semantic relevance, access frequency, and temporal patterns, mirroring Ebbinghaus's forgetting curve.
- FadeMem achieves superior multi-hop reasoning and retrieval performance while demonstrating significant storage reduction (45%) through selective forgetting.

---

[Vaccine Efficacy Estimands Implied by Common Estimators Used in Individual Randomized Field Trials](http://arxiv.org/abs/2601.18587)

- VE Estimands Framework: introduces a review and comparison of vaccine efficacy (VE) estimands for susceptibility in individual randomized trials, focusing on defining and comparing cumulative estimands (CI, IR, Cox, CH, Odds) nonparametrically.
- The paper explores the interpretational difficulties of these estimands, particularly concerning the ramp-up period and the impact of population heterogeneity modeled via frailty distributions.
- Key comparisons show that while all cumulative ITT VE estimands are similar under low control event rates, they diverge significantly as control event rates increase, necessitating careful prespecification.

---

[K-Myriad: Jump-starting reinforcement learning with unsupervised parallel agents](http://arxiv.org/abs/2601.18580)

- K-Myriad: introduces a scalable, unsupervised method for jump-starting reinforcement learning by maximizing the collective state entropy induced by a population of parallel policies, utilizing a Shared Trunk, Independent Heads, Agent-Specific Adapters, k-NN Entropy Estimator, Parallel Environment Replicas, and a Policy Gradient Algorithm.
- The approach cultivates a portfolio of specialized exploration strategies using a single policy network architecture with a shared backbone and multiple independent heads, enabling efficient training and diverse behavior discovery.
- K-Myriad is demonstrated on high-dimensional continuous control tasks, leveraging massive parallelization in GPU-based simulators like Isaac Sim to pretrain up to 50 policies.

---

[Stable Matching with Deviators and Conformists](http://arxiv.org/abs/2601.18573)

- DEVIATOR-SRI (Stable Matching with Deviators and Conformists): introduces a suite of stable matching problems that minimize instability (blocking pairs or blocking agents) specifically among a designated subset of agents called deviators, using FPT algorithms, maximum-weight matching subroutines, and complexity analysis.
- The paper establishes strong intractability results, proving that deciding whether a deviator-stable matching exists is NP-complete even for restricted preference list lengths or complete lists.
- Efficient parameterized algorithms are provided for k-DEVIATOR-MAX-SRI and k-DEVIATOR-SRI based on the number of deviators $|D|$ and maximum preference list length $d_{max}$, alongside polynomial-time solutions when preference lists are length at most 2.

---

[An LLM-Agent-Based Framework for Age of Information Optimization in Heterogeneous Random Access Networks](http://arxiv.org/abs/2601.18563)

- Reflex-Core: introduces an LLM agent-based framework for Age of Information (AoI) optimization in heterogeneous random access networks using an Observe-Reflect-Decide-Execute closed-loop mechanism, integrating SFT and PPO for autonomous policy refinement.
- The framework enables RMA nodes to intelligently adapt transmission probabilities based on semantic reasoning derived from network observations and accumulated historical insights to minimize system-wide AoI.
- The proposed RMA protocol and its priority-based variant achieve up to 14.9% reduction in average AoI and 20% faster convergence compared to state-of-the-art baselines across diverse heterogeneous scenarios.

---

[GenAgent: Scaling Text-to-Image Generation via Agentic Multimodal Reasoning](http://arxiv.org/abs/2601.18543)

- GenAgent: introduces an agentic multimodal model that unifies visual understanding and generation by decoupling capabilities, treating image generators as invokable tools.
- The framework executes autonomous multi-turn reasoning and generation via multimodal chains-of-thought encompassing reasoning, tool invocation, judgment, and reflection.
- Training utilizes a two-stage strategy: cold-start SFT on curated trajectories followed by Agentic RL using a hybrid reward mechanism combining pointwise and pairwise rewards.

---

[GenAI for Social Work Field Education: Client Simulation with Real-Time Feedback](http://arxiv.org/abs/2601.18517)

- SWITCH (Social Work Interactive Training Chatbot): introduces a scalable, low-cost training workflow integrating realistic client simulation, real-time counseling skill classification, and a Motivational Interviewing (MI) progression system.
- The system uses a cognitively grounded client profile, split into static fields (core beliefs) and dynamic fields (emotions, openness), allowing the client agent's behavior to evolve realistically.
- The MI Controller manages the progression through MI stages (Pre-Contemplation, Contemplation, Preparation) based on a calculated skill score and LLM-based evaluation of stage goals.

---

[Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates](http://arxiv.org/abs/2601.18510)

- JitRL (Just-In-Time Reinforcement Learning): introduces a training-free framework enabling continual learning in frozen LLM agents by using a Dynamic Memory Bank, Retrieval Mechanism, Value Estimation Module, Advantage Estimation Module, Logit Adjustment Mechanism, and LLM-based Evaluator.
- JitRL achieves test-time policy optimization by retrieving relevant trajectories from memory to estimate action advantages, which are then used to directly modulate the LLM's output logits via a closed-form additive update rule.
- The framework establishes a new state-of-the-art among training-free methods, significantly reducing monetary costs compared to computationally expensive gradient-based RL approaches like WebRL.

---

[DEEPMED: Building a Medical DeepResearch Agent via Multi-hop Med-Search Data and Turn-Controlled Agentic Training & Inference](http://arxiv.org/abs/2601.18496)

- DEEPMED: introduces a DeepResearch agent tailored for medical tasks, combining a LLM backbone with Search and Visit tools, trained via Warm-up Agentic SFT and Agentic RL using synthesized multi-hop medical QA data.
- The approach addresses the mismatch between general DeepResearch systems and medical reasoning by integrating clinical context interpretation and controlling tool-call scaling.
- Key components include a Difficulty-aware Turn-Penalty to suppress excessive tool calls during training and an Over-Evidence Monitor to halt redundant verification during inference.

---

[DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation](http://arxiv.org/abs/2601.18492)

- DV-VLN (Dual Verification for Reliable LLM-Based Vision-and-Language Navigation): introduces a verification-guided VLN framework that uses a generate-then-verify paradigm, employing a VLM (converts panoramic view to text) and an LLM (produces structured CoT) to generate and re-rank candidate actions via a Dual Verification Module (performs posterior checks).
- The framework utilizes a structured navigational Chain-of-Thought (CoT) consisting of Prediction, View Match, and Action triples, trained via parameter-efficient in-domain adaptation of an open-source LLaMA-2 backbone.
- Dual Verification involves True-False Verification (TFV) for global consistency and Masked-Entity Verification (MEV) for semantic alignment, aggregating their success counts to yield interpretable scores for robust action selection.

---

[AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security](http://arxiv.org/abs/2601.18491)

- AgentDoG: introduces a Diagnostic Guardrail framework for AI agent safety and security, utilizing a unified three-dimensional safety taxonomy and an Agentic XAI module for fine-grained risk diagnosis across agent trajectories.
- The framework employs a three-stage, planner-based pipeline for synthesizing high-fidelity, tool-augmented interaction trajectories with controllable risk injection and reliable safety labels.
- AgentDoG achieves superior performance in safety moderation by providing fine-grained and contextual monitoring, diagnosing root causes of unsafe actions, and supporting transparency beyond binary labels.

---

[OFFSEEKER: Online Reinforcement Learning Is Not All You Need for Deep Research Agents](http://arxiv.org/abs/2601.18467)

- OffSeeker (Offline Deep Research Agent): introduces a fully open-source, offline training suite for deep research agents, leveraging the DeepForge data synthesis pipeline, SFT, and DPO to achieve competitive performance without costly online RL.
- DeepForge is an end-to-end pipeline that synthesizes large-scale, high-difficulty, multi-hop deep research tasks by using LLM-based entity expansion and complex question generation based on entity graphs.
- The resulting 8B OffSeeker model, trained entirely offline on 66k QA pairs, 33k SFT trajectories, and 21k DPO pairs, matches or exceeds the performance of 30B online RL systems while reducing API costs to near zero.

---

[Emergent Cooperation in Quantum Multi-Agent Reinforcement Learning Using Communication](http://arxiv.org/abs/2601.18419)

- QMARL Communication Framework: introduces emergent cooperation in Quantum Multi-Agent Reinforcement Learning by adapting and evaluating eight classical MARL communication protocols (MATE, MEDIATE, Gifting, RIAL) using Quantum Q-Learning Agents with Variational Quantum Circuits (VQC) across Sequential Social Dilemmas (SSDs).
- The study demonstrates that token-based protocols, particularly MATETD, AutoMATE, and MEDIATE variants, achieve high cooperation levels across the Iterated Prisoner's Dilemma, Iterated Stag Hunt, and Iterated Game of Chicken.
- Quantum Q-Learning agents utilize VQCs for function approximation, embedding observations into qubits and processing them through parameterized single-qubit rotations and CNOT entanglement.

---

[daVinci-Dev: Agent-native Mid-training for Software Engineering](http://arxiv.org/abs/2601.18418)

- daVinci-Dev (Agent-native Mid-training recipe): introduces a systematic study of agentic mid-training using Contextually-native Trajectories (PR workflow data), Environmentally-native Trajectories (Real execution feedback), Base Model (Pre-trained LLM), and Supervised Fine-tuning (Final alignment stage) to enhance LLM capabilities for software engineering.
- The approach addresses the distribution mismatch between static training data and dynamic agent deployment by synthesizing large-scale agent-native data that preserves complete action-observation trajectories and environmental dynamics.
- The recipe achieves state-of-the-art resolution rates on SWE-Bench Verified by combining 68.6B tokens of context-rich PR data with high-quality, verified rollouts collected in executable environments.

---

[ARMOR: AGENTIC REASONING FOR METHODS ORCHESTRATION AND REPARAMETERIZATION FOR ROBUST ADVERSARIAL ATTACKS](http://arxiv.org/abs/2601.18386)

- ARMOR: introduces a VLM/LLM-driven multi-agent framework that orchestrates complementary adversarial primitives (CW, JSMA, STA) using InfoAgent (VLM analyzes image), ConductorAgent (LLM sets constraints), MethodAgents (Generate candidate perturbations), AdvisorAgent (LLM tunes hyperparameters), CritiqueAgents (Evaluate mixed candidate), StrategistAgent (Detects stagnation and escalates), and MixerAgent (Synthesizes final perturbation).
- The framework reframes adversarial attack generation as a dynamic, collaborative, and semantic-aware optimization process operating in a continuous closed-loop system.
- ARMOR achieves improved cross-architecture transferability and robustly fools blind ViT detectors while maintaining competitive perceptual quality compared to static ensemble and single-agent methods.

---

[AI Agent for Reverse-Engineering Legacy Finite-Difference Code and Translating to Devito](http://arxiv.org/abs/2601.18381)

- AIFDT (AI Agent for Fortran-to-Devito Translation): introduces an integrated AI agent framework for reverse-engineering legacy Fortran finite-difference code and translating it into Devito, utilizing KBCL, REL, CCL, DL, and ACL layers structured within a LangGraph architecture.
- The system constructs an extensive Devito knowledge graph using GraphRAG optimization, enabling multi-modal parallel retrieval across semantic communities to guide the LLM conversion process.
- Code synthesis is governed by Pydantic constraints for structured output, and quality-driven iterative optimization is achieved via LangGraph routing based on a comprehensive G-Eval validation framework.

---

[Promises, Perils, and (Timely) Heuristics for Mining Coding Agent Activity](http://arxiv.org/abs/2601.18345)

- HMC-AA: introduces a methodology for studying the impact of autonomous coding agents by documenting the promises, perils, and a comprehensive set of heuristics for detecting agent activity traces in software repositories.
- The approach defines the coding agent architecture, comprising an LLM executing in a loop, an agent harness, and access to various tools for autonomous task execution within a code base.
- Detection heuristics are categorized across GitHub artifacts, including specific configuration files, commit co-author trailers, pull request labels, and associated user accounts, to enable large-scale MSR studies.

---

[Agentic Much? Adoption of Coding Agents on GitHub](http://arxiv.org/abs/2601.18341)

- Coding Agents: introduces the first large-scale study (129,134 GitHub projects) quantifying the rapid and broad adoption of LLM-based coding agents, which operate via an Agentic Loop (LLM proposes solution/gets feedback) using Tools (Interact with environment) and Guidance Files (Guide agent with instructions).
- The study estimates the overall adoption rate of coding agents on GitHub to be between 15.85% and 22.60% as of late 2025, showing rapid growth since March 2025.
- AI-assisted commits are significantly larger than human-authored commits, concentrating primarily on implementing new features and bug fixes, indicating a wide scope of delegated tasks.

---

[MultiVis-Agent: A Multi-Agent Framework with Logic Rules for Reliable and Comprehensive Cross-Modal Data Visualization](http://arxiv.org/abs/2601.18320)

- MultiVis-Agent: introduces a logic rule-enhanced multi-agent framework for reliable cross-modal data visualization generation, featuring a Coordinator Agent orchestrating specialized agents (Database & Query, Visualization Implementation, Validation & Evaluation) governed by a Four-Layer Logic Rule Framework.
- The framework utilizes mathematical constraints within the logic rules to provide formal guarantees for system robustness, including parameter safety, systematic error recovery, and guaranteed loop termination, addressing critical reliability challenges in LLM-based systems.
- MultiVis-Agent addresses complex multi-modal visualization tasks across four scenarios (Basic Generation, Image-Referenced Generation, Code-Referenced Generation, and Iterative Refinement), achieving superior visualization quality and high task completion rates compared to baseline LLM workflows.

---

[SwipeGen: Bridging the Execution Gap in GUI Agents via Human-like Swipe Synthesis](http://arxiv.org/abs/2601.18305)

- SwipeGen: introduces an automated pipeline for synthesizing diverse and executable human-like swipe data by decomposing swipe gestures into quantifiable execution dimensions, including start position, direction, distance, and velocity.
- The pipeline utilizes a VLM and a pure vision GUI parser to identify scrollable targets, generate candidate swipes, verify validity based on visual changes, and generate step-level natural language descriptions.
- The synthesized data is used to fine-tune GUISwiper, a VLM GUI agent, which achieves 69.07% swipe execution accuracy on the new SwipeBench benchmark, representing a 214% improvement over existing VLM baselines.

---

[Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning](http://arxiv.org/abs/2601.18296)

- Temp-R1 (Unified Autonomous Agent for Complex Temporal KGQA): introduces the first autonomous end-to-end agent for TKGQA trained via reinforcement learning, utilizing an expanded action space and reverse curriculum learning.
- The framework decouples internal reasoning into explicit actions (`<plan>`, `<filter>`, `<rank>`) and external tool invocation (`<search>`) to enhance logical rigor and eliminate hallucinations in temporal sequencing.
- Temp-R1 employs Group Relative Policy Optimization (GRPO) following a Supervised Fine-Tuning (SFT) cold start, achieving state-of-the-art performance on complex temporal reasoning tasks using an 8B open-source LLM.

---

[Reinforcement Learning with Distributed MPC for Fuel-Efficient Platoon Control with Discrete Gear Transitions](http://arxiv.org/abs/2601.18294)

- RL-based Distributed MPC: introduces a computationally efficient approach for fuel-efficient platoon control by decoupling the discrete gear selection from the continuous dynamics optimization using a learned policy.
- The approach trains a Recurrent Neural Network (RNN) policy via Deep Q-Network (DQN) in a single-vehicle scenario to select optimal gear-shift schedules across the MPC prediction window, which then generalizes to multi-agent platoons.
- By shifting the gear-shift schedule optimization offline to the learned policy, the online control problem reduces to solving a Nonlinear Program (NLP), significantly lowering the computational burden compared to solving the original Mixed-Integer Nonlinear Program (MINLP).

---

[U-Fold: Dynamic Intent-Aware Context Folding for User-Centric Agents](http://arxiv.org/abs/2601.18285)

- U-Fold: introduces a dynamic context-folding framework for user-centric LLM agents, combining a Conversation Summarization Module (Tracks dialogue evolution/user intent) and a Dynamic Data Extraction Module (Filters structured tool outputs) to create a compact, intent-aware working context.
- The framework maintains the full user-agent dialogue and tool-call history, dynamically generating a summary and a task-relevant tool log (Filtered Structured Data) at each turn to mitigate context explosion and information loss.
- U-Fold consistently outperforms prior static context folding baselines and the full-context ReAct paradigm, achieving significant performance gains, especially in long-horizon, noisy, multi-turn tasks.

---

[VissimRL: A Multi-Agent Reinforcement Learning Framework for Traffic Signal Control Based on Vissim](http://arxiv.org/abs/2601.18284)

- VissimRL: introduces a modular Reinforcement Learning framework for Traffic Signal Control (TSC) that integrates the high-fidelity Vissim simulator via a Python-based Vissim Wrapper and a standardized RL Environment Framework supporting single- and multi-agent training.
- The Vissim Wrapper simplifies interaction with Vissim's complex Component Object Model (COM) interface, providing modules for basic simulation control, signal operation, and real-time performance evaluation.
- The framework supports three primary action types—Choose Next Phase, Switch Next or Not, and Set Phase Duration—and demonstrates significant reductions in development effort and consistent performance improvements in synthetic and real-world traffic scenarios.

---

[THINK-AUGMENTED FUNCTION CALLING: IMPROVING LLM PARAMETER ACCURACY THROUGH EMBEDDED REASONING](http://arxiv.org/abs/2601.18282)

- TAFC: introduces Think-Augmented Function Calling, a novel framework that enhances function calling accuracy by integrating explicit reasoning at both function and parameter levels using LLMs, Think Parameter Augmentation, Complexity Scorer ($\psi$), Reasoning-Augmented Tuples, Dynamic Description Tuning, Reasoning-Guided Tool Description Optimization, Filtering Function ($\mathcal{F}$), Reasoning Trace Repository ($H_t$), and the ReAct Framework.
- The framework introduces a universal "think" parameter augmentation to function signatures, enabling LLMs to articulate their decision-making process within the native function calling structure.
- TAFC employs dynamic optimization mechanisms for reasoning descriptions and tool descriptions to improve reasoning quality and align generated reasoning with human expectations without requiring architectural modifications to existing LLMs.

---

[When Nobody Around Is Real: Exploring Public Opinions and User Experiences On the Multi-Agent AI Social Platform](http://arxiv.org/abs/2601.18275)

- Social.AI (Multi-Agent AI Social Platform): introduces a two-stage investigation (Content Analysis and Diary Study) exploring public opinions and user experiences on Social.AI, a bot-centric social network where LLM-powered AI Agents emulate human sociality.
- The platform deploys numerous AI agents with customizable Follower Types (personas) that surround the human user, transforming the environment into an AI-dominant social media context.
- Findings reveal that while users project social expectations onto the agents, they often experience disappointment due to homogenized responses, shallow emotional connections, and physical tiredness, signaling a need for architected social life design.

---

[TAM-Eval: Evaluating LLMs for Automated Unit Test Maintenance](http://arxiv.org/abs/2601.18241)

- TAM-Eval (Test Automated Maintenance Evaluation): introduces a benchmark and evaluation framework for LLMs across three core test maintenance scenarios (creation, repair, and updating), utilizing an iterative LLM/Agentic System driven by Execution Status and Fail Feedback.
- The framework uses a curated real-world dataset of 1,539 validated scenarios from Python, Java, and Go projects, operating at the test file level with full repository context access.
- Evaluation is reference-free, relying on Pass Rate, $\Delta$Test Coverage, and $\Delta$Mutation Score, revealing that state-of-the-art LLMs show limited capabilities and high variability in realistic test maintenance processes.

---

[Probing the Future of Meta-Analysis: Eliciting Design Principles via an Agentic Research IDE](http://arxiv.org/abs/2601.18239)

- Research IDE: introduces a prototype implementing the "Research as Code" metaphor, featuring a Hypothesis Breakpoint mechanism supported by a multi-agent backend for in-situ verification of claims against literature.
- The system utilizes four specialized LLM-powered agents—Planner, Librarian, Reasoner, and Producer—to decompose claims, retrieve evidence via RAG, analyze consensus, and visualize results.
- Designed as a technology probe, Research IDE aims to preserve researcher autonomy and intellectual ownership by treating verification as debugging, prioritizing active falsification over passive retrieval.

---

[Yunjue Agent Tech Report: A Fully Reproducible, Zero-Start In-Situ Self-Evolving Agent System for Open-Ended Tasks](http://arxiv.org/abs/2601.18226)

- Yunjue Agent: introduces the In-Situ Self-Evolving paradigm, enabling LLM-based agents to autonomously synthesize, validate, and refine tools from scratch using a multi-agent architecture and a Parallel Batch Evolution strategy.
- The system employs Manager, Executor, Tool Developer, and Integrator agents, supported by Aggregator and Merger components, to distill short-term execution feedback into long-term, reusable capabilities stored in a convergent Global Tool Pool.
- The framework achieves state-of-the-art performance in zero-start settings across diverse benchmarks, demonstrating robust cross-domain transferability and convergence stability monitored by the Evolutionary Generality Loss (EGL) metric.

---

[ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants](http://arxiv.org/abs/2601.18225)

- ShopSimulator: introduces a large-scale Chinese e-commerce sandbox environment featuring a large product catalog, personalized multi-turn user modeling, and fine-grained product differentiation to evaluate and train RL-driven LLM shopping assistants.
- The environment supports both evaluation and training, offering 28K tasks across four scenarios: single-turn, multi-turn, and personalized variants of both, using reward signals derived from multiple dimensions (category, attributes, options, price).
- Evaluation shows that even advanced LLMs achieve low success rates (under 40%), with error analysis highlighting deficiencies in deep search, constraint enforcement, and balancing personalization cues.

---

[Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents](http://arxiv.org/abs/2601.18217)

- CDGSF: introduces a data-centric perspective on RL post-training for generalist LLM agents, analyzing environmental factors and modeling choices that preserve capabilities in unseen domains, utilizing LLM Agent (Llama-3.1-8B-Instruct), RL Training (GRPO optimization), Training Environments (WebShop, Sokoban, ALFWorld, SciWorld), Initial Policy (SFT warmup checkpoint), State Information Augmentation (injects goal-irrelevant content), and Step-by-Step Reasoning (explicit reasoning process).
- The study identifies State Information Richness (perception load) and Planning Complexity (reasoning load) as key environmental axes correlating strongly with cross-domain generalization.
- The authors propose State Information Augmentation, a low-overhead randomization technique, and confirm that enabling explicit step-by-step thinking during RL is crucial for preserving generalization.

---

[PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR](http://arxiv.org/abs/2601.18207)

- PaperSearchQA (Learning to Search and Reason over Scientific Papers): introduces an environment for training LLM search agents using RLVR over scientific literature, including a corpus of 16 million biomedical abstracts and 60k factoid QA samples.
- The system trains LLM agents using RLVR, where the policy LLM ($\pi_{\theta}$) interleaves reasoning (<think>) and retrieval (<search>) steps, receiving a binary reward based on the final answer's exact match verification.
- RLVR training significantly improves accuracy on scientific question-answering benchmarks compared to non-RL baselines, demonstrating the potential for training technical AI systems in knowledge-intensive domains.

---

[MemWeaver: Weaving Hybrid Memories for Traceable Long-Horizon Agentic Reasoning](http://arxiv.org/abs/2601.18204)

- MemWeaver: introduces a unified memory framework that consolidates long-term agent experiences into Graph Memory (GM), Experience Memory (ExpM), and Passage Memory (PM) to support traceable long-horizon agentic reasoning.
- The framework employs a dual-channel retrieval strategy to jointly retrieve structured knowledge and supporting evidence, constructing compact, information-dense contexts for LLM reasoning.
- MemWeaver substantially improves multi-hop and temporal reasoning accuracy while reducing input context length by over 95% compared to long-context baselines.

---

[SAGE: Steerable Agentic Data Generation for Deep Search with Execution Feedback](http://arxiv.org/abs/2601.18202)

- SAGE (Steerable Agentic Data Generation with Execution Feedback): introduces an agentic pipeline that automatically generates high-quality, difficulty-controlled deep search question-answer (QA) pairs using a Data Generator Agent (Generates/refines QA pairs) and a Search Agent (Validates QA/provides traces), guided by Execution Feedback (Correctness and difficulty signal).
- The dual-agent framework iteratively refines the generated QA pairs until they satisfy a specified Target Search Step ($S$), which serves as a proxy measure for question difficulty by controlling the required number of search steps.
- Training deep search agents with SAGE's synthetic data yields significant relative performance gains (up to 23%) on popular deep search benchmarks, demonstrating the high quality and complexity of the generated data.

---

[GAIA: A Data FLYWHEEL SYSTEM FOR TRAINING GUI TEST-TIME SCALING CRITIC MODELS](http://arxiv.org/abs/2601.18197)

- GAIA (GUI Action Critic's Data Flywheel System): introduces a training framework that uses a Data Flywheel (iterative data curation) to train the ICM (initial action correctness critic) and ICM-r2 (enhanced discernment critic) for improving GUI Agent performance during Test-Time Scaling (TTS).
- The system addresses irreversible errors in GUI agents by employing the Intuitive Critic Model (ICM) in a Best-of-N approach to select high-confidence actions from agent rollouts during inference.
- The Data Flywheel continuously collects refined positive and negative action samples, enabling the iterative training of ICM-r2 with enhanced discriminatory accuracy.

---

[NAVIDA: Vision-Language Navigation with Inverse Dynamics Augmentation](http://arxiv.org/abs/2601.18188)

- NAVIDA (Navigation with Inverse Dynamics Augmentation): introduces a unified Vision-and-Language Navigation (VLN) framework that integrates policy learning with action-grounded visual dynamics using Inverse Dynamics Supervision (IDS) and adaptive execution via Hierarchical Probabilistic Action Chunking (HPAC).
- IDS strengthens vision-action causality by requiring the Multimodal LLM backbone to predict the action chunk sequence between two adjacent frames, avoiding costly forward dynamics modeling or image generation.
- HPAC structures navigation trajectories into multi-level motion units, providing richer supervision signals and enabling an entropy-guided mechanism to dynamically adjust the LLM's action horizon during inference to mitigate error accumulation.

---

[Agentic Very Long Video Understanding](http://arxiv.org/abs/2601.18157)

- EGAgent (Enhanced Agentic Framework): introduces a multi-modal agentic system for very long video understanding, utilizing a Planning Agent, three specialized Retriever Tools, an Analyzer Tool, and a VQA Agent, all centered around a temporally annotated Entity Graph.
- The framework addresses limitations in context windows and multi-hop reasoning by decomposing complex queries into sub-tasks and leveraging structured search over the Entity Graph, which models people, objects, places, and their relationships over time.
- The Entity Graph Search Tool enables efficient cross-modal and compositional reasoning by querying a SQLite database containing temporally localized entity relationships extracted from audio transcripts and visual scene data.

---

[DEEPPLANNING: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints](http://arxiv.org/abs/2601.18137)

- DEEPPLANNING: introduces a challenging benchmark for practical long-horizon agent planning featuring multi-day travel and multi-product shopping tasks, evaluated via an LLM Agent, Offline Sandboxes, Python Toolkits, Database, and an Automated Evaluation System.
- The benchmark assesses three key agent competencies: Proactive Information Acquisition, Local Constrained Reasoning, and Global Constrained Optimization, all subject to verifiable constraints.
- Evaluation results show that frontier LLM agents struggle significantly with these complex tasks, highlighting the necessity of reliable explicit reasoning and parallel tool use for effectiveness-efficiency trade-offs.

---

[RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents](http://arxiv.org/abs/2601.18130)

- RouteMoA: introduces an efficient Mixture-of-Agents (MoA) framework with dynamic routing that uses a Lightweight Scorer for initial screening and a Mixture of Judges for score refinement, followed by a Model Ranking Mechanism to select LLMs based on performance, cost, and latency.
- By dynamically routing queries to a subset of high-potential LLMs without requiring full inference, RouteMoA significantly reduces computational cost by 89.8% and latency by 63.6% compared to classical MoA.
- The framework operates layer-wise, where the Lightweight Scorer uses query-aware prior knowledge for the first layer, and the Mixture of Judges uses posterior knowledge (self- and cross-assessment) for subsequent layers.

---

[Understanding Users' Privacy Reasoning and Behaviors During Chatbot Use to Support Meaningful Agency in Privacy](http://arxiv.org/abs/2601.18125)

- JIT-PNP (Just-in-Time Privacy Notice Panel): introduces a qualitative study examining user privacy reasoning and behaviors during LLM chatbot use, utilizing a simulated ChatGPT interface augmented with a panel that intercepts messages containing sensitive information and offers anonymization and privacy control options.
- The study found that the panel increased privacy awareness, encouraged protective actions (like retracting, faking, and generalizing sensitive data), and supported context-specific reasoning about disclosure decisions, contrasting with task-focused interactions observed without the panel.
- The panel surfaces ChatGPT's built-in privacy controls (model training opt-out and memory control) and provides granular anonymization strategies applicable per instance or across all instances of a given information type.

---

[MalURLBench: A Benchmark Evaluating Agents' Vulnerabilities When Processing Web URLs](http://arxiv.org/abs/2601.18113)

- MalURLBench: introduces the first benchmark for evaluating LLM-based web agents' vulnerabilities to maliciously disguised URLs, featuring 10 real-world scenarios and 7 categories of malicious websites.
- The benchmark construction involves generating scenarios, collecting malicious websites, designing attack templates targeting URL structure (subdomain, path, parameter), and optimizing instances via a mutation algorithm.
- Experiments on 12 LLMs reveal high vulnerability (32.9% to 99.9% attack success rates), leading to the proposal of URLGuard, a lightweight fine-tuned LLM defense module that significantly reduces risk.

---

[Chain-Length-Dependent Partitioning of 1-Alkanols in Raft-Like Lipid Membranes](http://arxiv.org/abs/2601.18095)

- Atomistic Molecular Dynamics Simulations: introduces a unified membrane-based mechanism linking 1-alkanol chain length to lateral partitioning and concomitant changes in membrane mechanics, including lateral pressure profiles, compressibility, and bending rigidity.
The study identifies a distinct cutoff chain length ($n_{cutoff}=12$), where shorter alkanols partition into liquid-disordered ($l_d$) domains, causing membrane softening, while longer alkanols localize in rigid liquid-ordered ($l_o$) domains, suppressing mechanical perturbation.
- The results provide a detailed molecular characterization of how alkanol chain length modulates membrane structure and mechanical response in laterally heterogeneous lipid membranes, supporting a membrane-mediated origin for the anesthetic cutoff phenomenon.

---

[DRPG (Decompose, Retrieve, Plan, Generate): An Agentic Framework for Academic Rebuttal](http://arxiv.org/abs/2601.18081)

- DRPG (Decompose, Retrieve, Plan, Generate): introduces an agentic framework for automatic academic rebuttal generation, utilizing a Decomposer (divides reviews), a Retriever (selects evidence), a Planner (formulates strategies), and an Executor (generates responses).
- The framework addresses long-context challenges by reducing input length by over 75% via the Retriever and enhances argument quality through the Planner, which identifies the most supported perspective (Clarification or Justification).
- DRPG consistently outperforms existing rebuttal pipelines and achieves performance beyond the average human level using a compact 8B LLM model, highlighting the value of structured, multi-stage processing.

---

[SPARKS OF COOPERATIVE REASONING: LLMS AS STRATEGIC HANABI AGENTS](http://arxiv.org/abs/2601.18077)

- LLMs as Strategic Hanabi Agents: introduces three progressive prompting scaffolds: Watson (minimal context), Sherlock (explicit deductive context and Bayesian reasoning), and Mycroft (implicit deduction via working memory) to evaluate LLMs in cooperative Hanabi.
- The research establishes the largest evaluation suite of 17 state-of-the-art LLMs in 2- to 5-player Hanabi settings and releases the first public Hanabi datasets, HanabiLogs and HanabiRewards, for post-training cooperative reasoning agents.
- Post-training a small open-weight LLM (Qwen3-Instruct) on the HanabiRewards dataset using RLVR significantly improves cooperative Hanabi play and demonstrates generalization to out-of-domain cooperative and temporal reasoning tasks.

---

[EFT-CoT: A Multi-Agent Chain-of-Thought Framework for Emotion-Focused Therapy](http://arxiv.org/abs/2601.17842)

- EFT-CoT (Emotion-Focused Therapy Chain-of-Thought framework): introduces a multi-agent collaborative architecture that models EFT theory as a structured state flow across three phases: Embodied Perception, Cognitive Exploration, and Narrative Intervention.
- The framework utilizes Knowledge Distillation to train the specialized EFT-LLM (Student Model) on a high-quality EFT-Instruct dataset derived from 67,000 authentic help-seeking texts, internalizing the complex intervention logic.
- EFT-LLM significantly outperforms strong baselines in empathetic depth and structural professionalism, providing an explainable and controllable paradigm for psychological computing.

---

[MedViz: An Agent-based, Visual-guided Research Assistant for Navigating Biomedical Literature](http://arxiv.org/abs/2601.20709)

- MedViz: introduces a visual analytics system that integrates interactive visualization with a context-aware, multi-agent AI framework to support large-scale biomedical literature exploration.
- The system transforms literature search from list-based retrieval into space-based semantic sensemaking by visualizing the entire semantic space of a corpus as an interactive point cloud, where spatial proximity reflects semantic similarity.
- The multi-agent backend, orchestrated by the Scholar agent, grounds its reasoning in user-defined visual selections, utilizing specialized Evidence, Analytical, and Discovery agents supported by RAG and LLMs.

---

#### 25th January 2026


[The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation](http://arxiv.org/abs/2601.17737)

- Agentic Framework: introduces a novel dialogue-to-cinematic-video generation pipeline, featuring ScripterAgent (Translates dialogue to script), DirectorAgent (Orchestrates SOTA video models), and CriticAgent (Evaluates script and video), designed to bridge the semantic gap between sparse dialogue and coherent cinematic output.
- DirectorAgent ensures long-horizon coherence by employing a Cross-Scene Continuous Generation Strategy, which utilizes a Frame-Anchoring Mechanism to maintain visual consistency across generated video segments.
- ScripterAgent is trained using a two-stage paradigm (SFT and GRPO with a Hybrid Reward Function) on the ScriptBench benchmark, and the system is evaluated using the AI-powered CriticAgent and the Visual-Script Alignment (VSA) metric.

---


[RGFL: Reasoning Guided Fault Localization for Automated Program Repair Using Large Language Models](http://arxiv.org/abs/2601.18044)

- RGFL (Reasoning Guided Fault Localization): introduces a hierarchical reasoning module (Generates bug explanations) for fault localization, leveraging LLM (Generates reasoning and ranks) to rerank initial file candidates and rank code elements, followed by Line Localization Stage (Narrows faulty lines), Patch Generation Stage (Proposes candidate fixes), and Patch Validation Stage (Validates patches via tests).
- RGFL improves localization accuracy by prompting an LLM to generate natural language reasoning about the relevance of individual code files and elements to the bug report, which is then used as a ranking signal.
- Integrating RGFL into the Agentless APR pipeline yields a 12.8% improvement in end-to-end repair success on SWE-bench Verified by significantly boosting file and element localization accuracy.

---

[Sentipolis: Emotion-Aware Agents for Social Simulations](http://arxiv.org/abs/2601.18027)

- SENTIPOLIS: introduces an emotionally stateful LLM agent framework that integrates continuous PAD representation, dual-speed emotion dynamics, and emotion-memory coupling to mitigate emotional amnesia in social simulations.
- The agent architecture is an orchestrated pipeline of modular subsystems, including the Emotional Subsystem, which maintains a persistent Pleasure-Arousal-Dominance (PAD) vector that evolves via Appraisal Dynamics Modeling and decays over time.
- The framework uses Semantic Enrichment to translate the continuous PAD state into human-interpretable emotion labels and descriptions, which are injected into the LLM prompt to ground dialog and reflection.

---

[Agentic AI for Self-Driving Laboratories in Soft Matter: Taxonomy, Benchmarks, and Open Challenges](http://arxiv.org/abs/2601.17920)

- Agentic SDL Framework: introduces a taxonomy and benchmarks for Self-Driving Laboratories (SDLs) in soft matter, framing autonomy as an agent-environment interaction problem with explicit observations, actions, costs, and constraints.
- The framework emphasizes verifiable decision policies, structured experiment specifications, and provenance logging to ensure reproducibility and safe operation in physical laboratories.
- The survey addresses challenges including long-horizon protocols, multi-modal observations, nonstationarity, and robust failure recovery, proposing benchmark tasks prioritizing cost-aware performance and robustness to drift.

---

[Neural-Inspired Multi-Agent Molecular Communication Networks for Collective Intelligence](http://arxiv.org/abs/2601.18018)

- Neural-Inspired Multi-Agent Molecular Communication Network (NIMAMCN): introduces a decentralized Molecular Communication (MC) network architecture where simple nanomachines act as agents, interacting via a diffusive medium using a Greenberg-Hastings (GH) cellular automata model and a threshold-based firing mechanism.
- The system is designed to achieve collective intelligence by operating at the "edge of chaos," demonstrated by a second-order phase transition where both pairwise and collective mutual information peak exactly at the critical activation threshold.
- The network dynamics are formalized using fixed-point equations derived via mean-field analysis, which are validated against stochastic simulations to approximate the macroscopic behavior of the network.

---

[Bayesian Multiple Testing for Suicide Risk in Pharmacoepidemiology: Leveraging Co-Prescription Patterns](http://arxiv.org/abs/2601.17985)

- Bayesian Spike-and-Slab Framework: introduces a unified Bayesian spike-and-slab hierarchical model for high-dimensional pharmacovigilance screening, leveraging co-prescription patterns to inform a structured prior covariance matrix ($\Sigma_D$) and employing Bayesian False Discovery Rate (FDR) control for signal detection.
- The framework screens 922 prescription drugs across 150 million patient records using a within-person incident user cohort (WPIUC) design to identify drugs associated with increased or decreased suicide risk.
- By incorporating co-prescription network information, the model adaptively borrows strength among pharmacologically related agents, improving sensitivity compared to previous empirical Bayes methods.

---

[Credit Fairness: Online Fairness In Shared Resource Pools](http://arxiv.org/abs/2601.17944)

- LENDRECOUP: introduces credit fairness, a strengthening of sharing incentives, and proposes LENDRECOUP, a novel online allocation mechanism that is credit fair and Pareto efficient, utilizing a credit system and the PSWC procedure.
- Credit fairness ensures agents who lend resources in early rounds are able to recoup them in later rounds, addressing disparities found in memoryless mechanisms like SMMF.
- LENDRECOUP is shown to be online strategyproof (OSP) and achieves performance comparable to state-of-the-art baselines in computational resource-sharing experiments.

---

[LLM-BASED SQL GENERATION: PROMPTING, SELF-REFINEMENT, AND ADAPTIVE WEIGHTED MAJORITY VOTING](http://arxiv.org/abs/2601.17942)

- ReCAPAgent-SQL (Refinement-Critique-Act-Plan agent-based SQL framework): introduces a multi-agent system for complex Text-to-SQL tasks, integrating specialized agents for planning, external knowledge retrieval, critique, action generation, iterative SQL refinement, dynamic schema linking, and result validation.
- The paper first proposes the SSEV (Single-Agent Self-Refinement with Ensemble Voting) pipeline, a two-stage Text-to-SQL approach that uses PreSQL and PostSQL generation combined with execution-guided self-refinement and adaptive ensemble voting mechanisms (WMA and RWMA).
- WMA adaptively weights predictions from multiple LLM experts based on historical performance, achieving high execution accuracy and robustness across benchmarks like Spider 1.0, BIRD, and Spider 2.0-lite.

---

[Dissipative Learning Framework for Viable Adaptive Systems](http://arxiv.org/abs/2601.17933)

- BEDS (Bayesian Emergent Dissipative Structures): introduces a conditional thermodynamic and information-geometric framework modeling learning dynamics through compressed belief states defined by four canonical parameters ($\mu, \tau, \phi, \kappa$) evolving under dissipation constraints.
- The framework proves that Fisher-Rao regularization is the unique thermodynamically optimal strategy for minimizing dissipation, unifying existing methods like Ridge, SIGReg, EMA, and SAC under a single fundamental loss equation.
- BEDS reframes common ML pathologies (overfitting, catastrophic forgetting, mode collapse) as thermodynamic imbalances related to insufficient dissipation control or excessive information accumulation, providing a taxonomy for learning problems based on crystallization behavior.

---

[LEARNING TRANSFERABLE SKILLS IN ACTION RPGS VIA DIRECTED SKILL GRAPHS AND SELECTIVE ADAPTATION](http://arxiv.org/abs/2601.17923)

- Modular Skill Graph Architecture: introduces a lifelong learning agent for real-time control (Dark Souls III) by modeling combat as a directed skill graph with five reusable skills (camera control, lock-on, movement, dodging, and heal-attack decision policy), trained via a hierarchical curriculum.
- The architecture uses specialized observation spaces and independent skill policies (DQN) that execute concurrently, with upstream skills (C, L, M) fixed to shape the data distribution for adaptive downstream skills (D, H).
- Selective fine-tuning of only the phase-sensitive downstream skills (D and H) rapidly recovers performance under a limited interaction budget following a domain shift (Phase 1 to Phase 2), significantly outperforming a monolithic end-to-end baseline.

---

[Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation](http://arxiv.org/abs/2601.17915)

- EoG (Explanations over Graphs): introduces Semantic Belief Propagation (SBP), a disaggregated neurosymbolic architecture that separates the Deterministic Controller (symbolic engine) from the Abductive Policy ($\pi_{abd}$) (stateless LLM) to perform graph-guided abductive reasoning over IT operational graphs.
- The framework models operational diagnosis as iterative local belief assignment and propagation over an evolving Explanatory Graph ($G_s$), enabling non-monotonic belief revision when new evidence contradicts prior assumptions.
- By enforcing structured exploration and bounding context via the Context Contract (CxC), EoG demonstrates substantial improvements in accuracy and run-to-run consistency over ReAct baselines on operational diagnostic tasks.

---

[When Personalization Legitimizes Risks: Uncovering Safety Vulnerabilities in Personalized Dialogue Agents](http://arxiv.org/abs/2601.17887)

- PS-Bench (Personalization-Safety Benchmark): introduces intent legitimation, a safety failure mode in personalized LLM agents where benign personal memories bias intent inference, causing the model to legitimize harmful queries.
- The benchmark systematically evaluates safety under context-conditioned intent recognition using multi-session memory and persona context, comparing stateless and memory-augmented agents.
- The authors propose a lightweight detection-reflection intervention, utilizing an Intent Legitimation Detection auditor and a Safety Reflective Reminder generator, to effectively mitigate safety degradation while maintaining personalization utility.

---

[Self-Manager: Parallel Agent Loop for Long-form Deep Research](http://arxiv.org/abs/2601.17879)

- Self-Manager: introduces a parallel agent loop architecture for long-form deep research, utilizing a Main Thread to manage asynchronous and concurrent Subthreads via the Thread Control Block (TCB) mechanism.
- The architecture enables context isolation between threads and non-blocking execution, significantly improving scalability and adaptability compared to sequential single-agent loops.
- Evaluated on the DeepResearch Bench, Self-Manager consistently outperforms existing single-agent loop baselines in contextual capacity, efficiency, and generalization.

---

[D-Models and E-Models: Diversity-Stability Trade-offs in the Sampling Behavior of Large Language Models](http://arxiv.org/abs/2601.17865)

- D-Models and E-Models: introduces a probabilistic sampling perspective to explore LLM generation, identifying two distinct model types—D-Models (Deterministic) and E-Models (Exploratory)—based on the alignment between $P_{token}$ (Token Probability) and $P_{task}$ (Target Distribution).
- D-Models exhibit high $P_{token}$ concentration and large step-to-step variability, favoring stability and deterministic planning, while E-Models show more stable $P_{token}$ alignment with $P_{task}$, favoring diversity and exploratory local planning.
- The study reveals a systematic diversity-stability trade-off across downstream tasks like code generation and recommendation, informing LLM selection and configuration for web-scale applications.

---

[Multivariate Rényi divergences characterise betting games with multiple lotteries](http://arxiv.org/abs/2601.17850)

- MRD-EUT-GPT Framework: introduces a quantitative connection between information theory, economic theory (Expected Utility Theory), and physical theories (General Probabilistic Theories) by characterizing multi-lottery betting games using multivariate Rényi divergences.
- The paper demonstrates that the unconditional multivariate Rényi divergence quantifies the economic-theoretic value (isoelastic certainty equivalent) a rational agent assigns to $d$ lotteries under fair odds and optimal betting strategies.
- A new conditional multivariate Rényi divergence is proposed to characterize betting games where the gambler has access to side information, establishing a resource measure for informative measurements within General Probabilistic Theories.

---

[An Effective and Cost-Efficient Agentic Framework for Ethereum Smart Contract Auditing](http://arxiv.org/abs/2601.17833)

- HEIMDALLR: introduces an automated auditing agent designed to overcome scalability and precision trade-offs in Ethereum smart contract auditing by employing a cohesive pipeline.
- The framework utilizes Contextual Profiling for logic-preserving code batching, Model-Agnostic Auditing via a Plan-Remind-Solve agentic workflow for detection, and False Positive Filtration for rigorous verification.
- HEIMDALLR achieves high detection rates for complex business logic vulnerabilities while drastically reducing false positives and operating cost-efficiently using lightweight, open-source LLMs.

---

[Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents](http://arxiv.org/abs/2601.17829)

- DFCDG: introduces a novel technique for generating synthetic function-calling datasets by optimizing general-purpose linguistic and argument diversity metrics across user queries and parameter values.
- The approach employs a greedy Diverse Generation procedure guided by multiple diversity metrics (lexical, syntactic, semantic, and argument) and validated by an LLM judge to ensure high quality and robustness.
- Training LLMs on the resulting diverse dataset yields superior out-of-distribution performance, achieving up to a 7.4% increase in accuracy on the BFCL benchmark compared to models trained on baseline synthetic data.

---

[Multi-Agent Collaborative Intrusion Detection for Low-Altitude Economy IoT: An LLM-Enhanced Agentic AI Framework](http://arxiv.org/abs/2601.17817)

- Multi-Agent Collaborative Intrusion Detection Framework: introduces an LLM-enhanced agentic AI framework for low-altitude economy IoT (LAE-IoT) intrusion detection, utilizing specialized agents for perception, memory, reasoning, and action.
- The framework employs a Perception and Memory Agent based on feature extraction using a self-supervised denoising diffusion probabilistic model (DDPM) to build a stable, generalizable knowledge base from network traffic images.
- The Reasoning Agent leverages LLMs to guide Particle Swarm Optimization (PSO) for intelligent feature selection, enabling rapid, context-aware adaptation and efficient real-time threat detection by the Adaptive Classification Agent.

---

[Neuro-Symbolic Verification on Instruction Following of LLMs](http://arxiv.org/abs/2601.17789)

- NSVIF (Neuro-Symbolic Verification on Instruction Following of LLMs): introduces a neuro-symbolic framework for verifying LLM instruction following by formalizing the task as a Constraint-Satisfaction Problem (CSP) using Formulation Agent (translates instruction to CSP), Checking Agent (generates constraint checkers), and Solver Agent (solves the CSP).
- The framework decomposes instructions into fine-grained logic and semantic constraints, enabling a unified Solver Agent atop the Z3 SMT Solver to orchestrate symbolic reasoning and neural semantic analysis.
- NSVIF significantly outperforms LLM-as-a-judge baselines on the VIFBENCH benchmark and provides interpretable feedback that improves LLM instruction following capability without post-training.

---

[Multi-Agent End-to-End Vulnerability Management for Mitigating Recurring Vulnerabilities](http://arxiv.org/abs/2601.17762)

- MAVM (Multi-Agent framework for end-to-end Recurring Vulnerability Management): introduces a multi-agent pipeline for end-to-end recurring vulnerability management, including VKB, detection, confirmation, repair, and validation components.
- The framework leverages multiple LLM-based agents and a Vulnerability Knowledge Base (VKB) constructed from publicly disclosed vulnerabilities to address contextual limitations and underutilized historical knowledge.
- MAVM simulates real-world security workflows by coordinating specialized agents, supported by context-retrieval tools for repository-level reasoning, achieving superior repair accuracy compared to baselines.

---

[ProGraph-R1: Progress-aware Reinforcement Learning for Graph Retrieval Augmented Generation](http://arxiv.org/abs/2601.17755)

- ProGraph-R1: introduces a step-level progress-aware reinforcement learning framework that enhances multi-turn graph retrieval-augmented generation in multi-hop knowledge-intensive tasks using a graph structure-aware retrieval module and progress-based step-wise policy optimization.
- The framework addresses limitations in existing RL-based GraphRAG systems by mitigating overreliance on contextual similarity and sparse, outcome-level rewards.
- ProGraph-R1 utilizes entity informativeness-based hypergraph retrieval and step-wise advantage modulation to achieve coherent traversal and fine-grained credit assignment.

---

[Faramesh: A Protocol-Agnostic Execution Control Plane for Autonomous Agent systems](http://arxiv.org/abs/2601.17744)

- Faramesh: introduces the Action Authorization Boundary (AAB) and Canonical Action Representation (CAR) to provide execution-time governance for autonomous agents, ensuring deterministic authorization and non-bypassable enforcement.
- The AAB acts as a mandatory boundary between the agent's reasoning space and real-world execution, enforcing decisions (PERMIT, DEFER, DENY) over canonicalized action representations (CAR) derived from agent intent.
- Faramesh establishes architectural guarantees, including fail-closed semantics and provenance-complete decision records, making autonomous actions auditable, replayable, and trustworthy without constraining upstream agent reasoning.

---

[Athanor: Authoring Action Modification-based Interactions on Static Visualizations via Natural Language](http://arxiv.org/abs/2601.17736)

- Athanor: introduces a novel approach to transform existing static visualizations into interactive ones using multimodal LLMs and natural language instructions, with all Action-Modification Interaction Design Space, Multi-Agent Requirement Analyzer, and Visualization Abstraction Translator components.
- The system utilizes a multi-agent architecture to translate user requirements into structured interaction specifications defined by an action-modification design space.
- Athanor converts static visualizations into a flexible, implementation-agnostic constraint-based representation to enable subsequent modifications and interactive features.

---

[ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents](http://arxiv.org/abs/2601.17735)

- REFUGE (RElational FEature GEneration): introduces an agentic framework for generating informative relational features for RDB prediction tasks, utilizing specialized LLM agents for schema selection, feature generation, and two-stage feature filtering.
- The framework operates within an iterative feedback loop, allowing the LLM agents to progressively refine feature generation without explicit ground-truth supervision.
- REFUGE significantly improves predictive performance on various RDB prediction tasks by effectively navigating the complex schema and large feature space.

---

[EntWorld: A Holistic Environment and Benchmark for Verifiable Enterprise GUI Agents](http://arxiv.org/abs/2601.17722)

- EntWorld: introduces a scalable, interactive, and deterministically verifiable environment and benchmark for enterprise GUI agents, utilizing a schema-driven task generation framework, a multi-app enterprise sandbox, and a deterministic evaluation protocol based on SQL state verification.
- The benchmark consists of 1,756 long-horizon tasks across six enterprise domains (CRM, ERP, ITIL) and highlights a significant "Enterprise Gap," as state-of-the-art LLMs achieve only 47.61% success rate compared to 85% human performance.
- The proposed EntAgent-RL, trained using Group Relative Policy Optimization (GRPO) on multimodal inputs (screenshot + accessibility tree), achieves a new state-of-the-art success rate of 56.89% among evaluated agents.

---

[Do Reasoning Models Ask Better Questions? A Formal Information-Theoretic Analysis on Multi-Turn LLM Games](http://arxiv.org/abs/2601.17716)

- Multi-Turn Dialogue Framework: introduces a multi-turn evaluation framework for LLMs in structured hypothesis spaces, employing Seeker, Oracle, and Pruner agents interacting with a Knowledge Graph to measure Information Gain (IG).
- The framework quantitatively assesses the effectiveness of LLM-based Seeker agents in information-gathering tasks using IG, grounded in Shannon entropy, to evaluate query quality at each turn.
- Experiments demonstrate that LLMs utilizing Chain-of-Thought reasoning achieve higher IG and solve tasks in fewer steps, particularly under partially observable conditions in the geographical Guess My City game.

---

[SQL-Trail: Multi-Turn Reinforcement Learning with Interleaved Feedback for Text-to-SQL](http://arxiv.org/abs/2601.17699)

- SQL-TRAIL: introduces a multi-turn RL agentic framework for Text-to-SQL, utilizing an LLM policy, a SQL execution tool, and a composite reward panel to iteratively refine SQL queries based on database feedback.
- The framework employs a staged training pipeline, starting with Supervised Fine-Tuning (SFT) to distill instruction-following behavior, followed by Reinforcement Learning (RL) using a modified Grouped Reinforcement Policy Optimization (GRPO) algorithm.
- SQL-TRAIL achieves state-of-the-art accuracy and exceptional data efficiency by incorporating an adaptive turn budget mechanism that scales interaction depth based on query difficulty.

---

[LegalMALR:Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval](http://arxiv.org/abs/2601.17692)

- LegalMALR: introduces a retrieval framework integrating a Multi-Agent Query Understanding System (MAS) and a zero-shot LLM Reranker for robust Chinese statute retrieval, addressing implicit, multi-issue, and colloquial legal queries.
- The MAS iteratively generates diverse, legally grounded query reformulations using specialized agents, followed by dense retrieval and lightweight reranking to accumulate a high-recall candidate set.
- The MAS policy is stabilized using Generalized Reinforcement Policy Optimization (GRPO), and the final candidate set is ranked by the LLM Reranker based on substantive legal reasoning, substantially outperforming RAG baselines.

---

[Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis](http://arxiv.org/abs/2601.17687)

- ChemCRAFT: introduces a novel framework leveraging agentic reinforcement learning to decouple chemical reasoning from knowledge storage, enabling a locally deployable small LLM to achieve superior performance with minimal inference costs.
- The framework utilizes a Policy Model and a Chemical Agent Sandbox, trained via a two-stage paradigm involving Cold-Start SFT and SMILES-GRPO, to orchestrate external tools for complex chemical tasks like molecular optimization and synthesis prediction.
- ChemCRAFT constructs high-quality Agentic Trajectories using a "Hypothesis-Action-Refinement" loop and employs a Multidimensional Chemical-Aware Reward Function to ensure scientific validity and structural precision.

---

[DIML: Differentiable Inverse Mechanism Learning from Behaviors of Multi-Agent Learning Trajectories](http://arxiv.org/abs/2601.17678)

- DIML: introduces a likelihood-based framework that infers unknown, potentially unstructured neural incentive mechanisms by differentiating through a model of multi-agent learning dynamics using counterfactual payoffs.
- The approach leverages transient learning dynamics and off-equilibrium behavior as identifying signals, enabling inverse reconstruction and counterfactual auditing without relying on equilibrium assumptions.
- DIML reliably recovers identifiable payoff differences and supports counterfactual prediction across unstructured neural, structured economic, and large-scale anonymous game environments.

---

#### 24th Jan 2026


[Truth-Revealing Participatory Budgeting](http://arxiv.org/abs/2601.17538)

- Truth-Revealing Participatory Budgeting Framework: introduces an epistemic model for Participatory Budgeting (PB) where agents aggregate noisy signals about unobservable project quality using various voting rules (AV, PAV, MES, Phragmén, GC) to select a set of alternatives maximizing utilitarian total utility under a budget constraint.
- The research analyzes the performance of these rules, finding that in the unit-cost setting, performance converges to 1 (full information aggregation) as the number of agents increases, but performance decreases significantly as the range of project costs expands.
- An analysis of strategic behavior shows that informative voting constitutes a Bayes-Nash equilibrium only under highly restrictive conditions, suggesting that truthful voting is generically not an equilibrium in this context.

---


[How AI Coding Agents Modify Code: A Large-Scale Study of GitHub Pull Requests](http://arxiv.org/abs/2601.17581)

- Four-step workflow: introduces a large-scale empirical comparison of Agentic and Human Pull Requests (PRs) using the AIDev dataset to analyze structural code changes and description-to-diff alignment.
- The methodology involves collecting and extending PR data, filtering for merged PRs with valid patches, and applying structural metrics and similarity measures (TF-IDF, BM25, CodeBERT, GraphCodeBERT).
- The study finds that Agentic PRs differ significantly from Human PRs in commit count and file breadth, but exhibit slightly higher description-to-diff consistency across lexical and semantic measures.

---


[Code Change Characteristics and Description Alignment: A Comparative Study of Agentic versus Human Pull Requests](http://arxiv.org/abs/2601.17627)

- CS-AHPR: introduces a comparative study analyzing 33,596 agent-generated Pull Requests (APRs) and 6,618 human PRs (HPRs) using Code Change Analysis and Description Alignment Metrics, revealing differences in code modification patterns and communication quality.
- The study finds that APRs exhibit higher symbol churn and earlier removal times (median 3 days vs. 34 days for HPRs), reflecting a focus on narrow tasks like documentation and tests.
- While agents excel at commit-level messaging (semantic similarity 0.72 vs. 0.68), they lag humans in PR-level summarization, and commit message length is identified as the strongest predictor of good APR descriptions.

---

[Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests](http://arxiv.org/abs/2601.17617)

- Agentic Search Log Analysis Framework: introduces a large-scale log analysis of 14.44M real agentic search requests from DeepResearchGym (DRGym) using LLM-based Annotation Pipelines, Offline Replay, and the Context-driven Term Adoption Rate (CTAR) metric.
- The analysis reveals intent-conditioned behavioral patterns, finding that agents frequently reuse evidence across steps (mean CTAR of 54%) and that Declarative (fact-seeking) sessions exhibit high repetition, suggesting potential stall signals.
- The findings provide practical implications for designing reliable agentic IR systems, including repetition-aware stopping, intent-adaptive retrieval budgeting, and explicit cross-step context tracking.

---

[Athena: Synergizing Data Prefetching and Off-Chip Prediction via Online Reinforcement Learning](http://arxiv.org/abs/2601.17615)

- Athena: introduces a reinforcement learning (RL)-based technique that synergizes data prefetchers and an Off-Chip Predictor (OCP) using an RL Agent, Q-Value Storage (Stores state-action Q-values), Composite Reward Framework (Separates correlated/uncorrelated effects), State Measurement (Observes system-level features), Coordination Action (Enables/disables OCP/prefetchers), and Prefetcher Aggressiveness Control (Adjusts prefetch degree).
- The framework models the coordination between prefetchers and OCP as an RL problem, observing system-level features (e.g., accuracy, bandwidth usage) over an epoch to select a coordination action and adjust prefetcher aggressiveness.
- Athena consistently outperforms prior coordination policies across diverse workloads and system configurations by isolating the true impact of its actions from inherent workload variations using its composite reward structure.

---

[Deep Intrinsic Surprise-Regularized Control (DISRC): A Biologically Inspired Mechanism for Efficient Deep Q-Learning in Sparse Environments](http://arxiv.org/abs/2601.17598)

- DISRC (Deep Intrinsic Surprise-Regularized Control): introduces a biologically inspired augmentation to the Deep Q-Network (DQN) framework that dynamically scales Q-updates based on latent-space surprise derived from a LayerNorm-based Encoder and a Surprise Controller.
- The framework computes surprise as the normalized deviation between the current latent state and a moving Latent Setpoint Vector, promoting plasticity during early exploration and stability as familiarity increases.
- DISRC improves learning efficiency and stability in sparse-reward environments by regulating learning intensity via a second-order update regulation mechanism based on intrinsic surprise signals.

---

[Learning to Ideate for Machine Learning Engineering Agents](http://arxiv.org/abs/2601.17596)

- MLE-IDEATOR: introduces a novel dual-agent framework that decouples strategic ideation from low-level implementation for machine learning engineering tasks, utilizing an Implementer Agent (Implementation/Code Execution) and an Ideator Agent (Strategic Guidance/Idea Generation).
- The Implementer uses a dedicated `<seek_help>` action to solicit strategic guidance from the Ideator when performance plateaus, enabling targeted, context-aware collaboration.
- The Ideator is trained using an efficient RL pipeline with execution-based rewards (GRPO algorithm) to maximize the generation of effective, performance-improving algorithmic suggestions, achieving significant gains on MLE-Bench.

---

[Intelligence Requires Grounding But Not Embodiment](http://arxiv.org/abs/2601.17588)

- NEGA (Non-Embodied Grounded Agent): introduces a theoretical agent architecture arguing that intelligence requires grounding, achieved via a Grounding Mechanism (Assigns external value) and a Perception-Action Loop (Continuous interaction) within a Digital Environment (Complex interaction space), but not physical embodiment.
- The agent is conceptualized as an Advanced LLM (Core reasoning engine) with Tool-augmented Capabilities (Extends functionality) and Stateful Memory (Stores internal/external states), enabling it to achieve complex goals autonomously online.
- Grounding is defined as the mechanism prescribing externally consistent meaning to symbols by tying environmental rules and constraints to the symbolic system, which is necessary for motivation, causality, and learning from experience.

---

[GenAI-Net: A Generative AI Framework for Automated Biomolecular Network Design](http://arxiv.org/abs/2601.17582)

- GenAI-Net: introduces a generative AI framework that automates Chemical Reaction Network (CRN) design by coupling an AI agent (proposes reactions) to a simulation-based evaluation (user-specified objective).
- The framework operates via a Reinforcement Learning loop where the AI agent iteratively proposes reactions from a Reaction Library, which are simulated and evaluated to refine the agent's generative policy.
- GenAI-Net efficiently produces diverse, topologically distinct Input-Output CRNs capable of realizing complex dynamical functions, including robust perfect adaptation, oscillators, and logic circuits.

---

[Status Hierarchies in Language Models](http://arxiv.org/abs/2601.17577)

- Adaptation of Expectation States Theory for Multi-Agent LLMs: investigates whether LLMs form status hierarchies by adapting a classic human social experiment into a multi-agent scenario using paired LLM instances, status manipulations, and capability differences to measure deference rates.
- The study uses GPT-4.1-nano and GPT-3.5-turbo models in six factorial conditions to test the effects of pure status, pure capability, status alignment, and status-capability conflict on deference.
- Results show that explicit status framing creates deference asymmetries when LLM capability is equal, but capability differences dominate status cues, fundamentally diverging from human status dynamics.

---

[Sponge Tool Attack: Stealthy Denial-of-Efficiency against Tool-Augmented Agentic Reasoning](http://arxiv.org/abs/2601.17566)

- STA (Sponge Tool Attack): introduces a stealthy Denial-of-Efficiency (DoE) attack against tool-augmented agentic reasoning, utilizing the Prompt Rewriter (generates objective-driven rewrites), Quality Judge (evaluates rewrite quality), Policy Inductor (distills reusable policies), Policy Bank (stores rewriting strategies), History Buffer (stores successful attempts), Victim Agent (tool-augmented LLM target), and Tools (external utilities).
- The attack operates under a strict query-only access constraint, modifying the input prompt to convert concise reasoning trajectories into unnecessarily verbose and convoluted ones, thereby increasing computational overhead.
- STA achieves efficiency and robustness through an iterative multi-LLM collaborative framework and an offline policy induction process that captures reusable, model-agnostic rewriting strategies.

---

[Towards Generalisable Imitation Learning Through Conditioned Transition Estimation and Online Behaviour Alignment](http://arxiv.org/abs/2601.17563)

- UfO: introduces Unsupervised Imitation Learning from Observation, a novel unsupervised ILfO method that learns a policy ($\pi_\theta$), a conditioned generative model ($G_\phi$), and a discriminator model ($D_\omega$) through a two-stage process.
- The initial Reconstruction Stage jointly trains the policy and generative model using mutual optimization to approximate the environment dynamics and infer teacher actions from state transitions without action-based supervision.
- The subsequent Adversarial Stage refines the policy using a recurrent discriminator and online environment interaction, enabling UfO to outperform the teacher and achieve the lowest standard deviation across five benchmark environments.

---

[Breaking the Protocol: Security Analysis of the Model Context Protocol Specification and Prompt Injection Vulnerabilities in Tool-Integrated LLM Agents](http://arxiv.org/abs/2601.17549)

- ATTESTMCP (Protocol Extension): introduces a backward-compatible extension to the Model Context Protocol (MCP) addressing architectural security weaknesses, including capability attestation, message authentication, origin tagging, isolation enforcement, and replay protection.
- The research uses the PROTOAMP framework to demonstrate that MCP's architectural choices amplify attack success rates (ASR) by 23–41% compared to non-MCP baselines, particularly in cross-server propagation and sampling attacks.
- ATTESTMCP reduces the overall attack success rate from 52.8% to 12.4% by mitigating three protocol-level vulnerabilities: least privilege violation, sampling without origin authentication, and implicit trust propagation.

---

[Prompt Injection Attacks on Agentic Coding Assistants: A Systematic Analysis of Vulnerabilities in Skills, Tools, and Protocol Ecosystems](http://arxiv.org/abs/2601.17548)

- Systematic Analysis of Vulnerabilities: introduces a comprehensive analysis of prompt injection attacks targeting agentic coding assistants, proposing a novel three-dimensional taxonomy (delivery vector, attack modality, propagation behavior) and a defense-in-depth framework.
- The architecture of agentic coding assistants includes the LLM Core, Tool Runtime, Skill Registry, and System Integration via the Model Context Protocol (MCP), which collectively expand the attack surface for indirect prompt injection.
- Empirical analysis reveals that adaptive attacks successfully bypass 90%+ of published defenses, demonstrating that current detection-based mitigations are insufficient against sophisticated LLM instruction injection.

---

[Cognitive Platform Engineering for Autonomous Cloud Operations](http://arxiv.org/abs/2601.17542)

- CPE (Cognitive Platform Engineering): introduces a four-plane reference architecture that integrates sensing, reasoning, and autonomous action into the platform lifecycle for resilient, self-adjusting cloud environments.
- The architecture consists of the Data Plane, Intelligence Plane, Control Plane, and Experience Plane, connected by a closed-loop Sense-Reason-Act feedback cycle.
- Experimental evaluation demonstrates that CPE significantly reduces Mean Time to Resolution (MTTR) and improves resource efficiency and policy compliance compared to a traditional DevOps baseline.

---

[Bridging Expectation Signals: LLM-Based Experiments and a Behavioral Kalman Filter Framework](http://arxiv.org/abs/2601.17527)

- Behavioral Kalman Filter (BKF): introduces a formal quantitative framework to decode the expectation formation mechanisms of LLM-based economic agents, incorporating prior discounting and subjective signal covariance to quantify cognitive biases.
- Experiments reveal that LLM agents exhibit systematic behavioral biases, including micro-signal prioritization, rapid prior discounting, and a "cognitive discount" effect where concurrent signals interfere negatively.
- The study demonstrates that LoRA fine-tuning mitigates, but does not fully eliminate, inherent behavioral biases in LLM expectation formation, confirming that non-rationalities persist in the model architecture.

---

#### 23rd January 2026

[VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents](http://arxiv.org/abs/2601.16973)

- VisGym: introduces a diverse, customizable, and scalable gymnasium of 17 long-horizon environments for evaluating and training VLMs, featuring visual observation, textual instruction, environment feedback, and function-conditioned action space.
- The suite spans symbolic puzzles, real-image understanding, navigation, and manipulation, revealing that frontier models struggle in interactive settings, particularly with long context and visual grounding.
- VisGym provides fine-grained controls over input representation, difficulty, and feedback, and includes oracle multi-step solvers for generating structured demonstrations for supervised finetuning.

---


[Mixture-of-Models: Unifying Heterogeneous Agents via N-Way Self-Evaluating Deliberation](http://arxiv.org/abs/2601.16863)

- NSED (N-Way Self-Evaluating Deliberation): introduces a Runtime Mixture-of-Models (MoM) architecture that unifies heterogeneous LLM agents via a Macro-Scale Semantic Recurrent Neural Network (SRNN) topology, managed by a Dynamic Expertise Broker.
- The protocol formalizes deliberation as a Recurrent Cognitive Cycle where the consensus state loops back through a semantic forget gate ($\gamma$) to enable iterative refinement without proportional VRAM scaling.
- Trustless consensus is enforced using a Diagonal Mask (D) at the Quadratic Voting layer, mitigating authority bias and ensuring convergence based on semantic merit.

---

[Multi-Agent Non-Discriminatory Contracts](http://arxiv.org/abs/2601.16835)

- PoND ($\beta$) (Price of $\beta$-Non-Discrimination): introduces a framework to quantify the tradeoff between maximizing the Principal's utility and equalizing payments among Agents in a multi-agent hidden-action setting, using $\beta$-ND Contracts.
- The paper establishes asymptotic bounds on PoND($\beta$), showing that the price of exact non-discrimination (PoND) scales logarithmically with the number of agents $n$, bounded between $\Omega(\frac{\log n}{\log \log n})$ and $O(\log n)$.
- By relaxing the non-discrimination requirement to $\beta=n^\delta$ ($\delta \in (0, 1]$), the PoND($\beta$) is bounded by a constant factor, and tight results are derived for the two-agent case.

---

[On Best-of-Both-Worlds Fairness via Sum-of-Variances Minimization](http://arxiv.org/abs/2601.16579)

- Sum-of-Variances Minimization: introduces a Best-of-Both-Worlds (BoBW) approach for fair allocation of indivisible goods by minimizing the Sum-of-Variances (SoV) objective subject to ex-ante proportionality, resulting in a randomized allocation distribution.
- When agents have identical additive valuations, the resulting distribution is EFX ex-post, guaranteeing a constant approximation of the Maximin Share (MMS), although it may fail to guarantee full MMS fairness for $n \geq 3$ agents.
- When valuations are non-identical, the SoV minimization approach fundamentally fails to guarantee ex-post fairness, as the supporting allocations may not be EF1 and provide no constant-factor approximation to the MMS, even in the simplest two-agent, two-good setting.

---


[Spatial-Agent: Agentic Geo-spatial Reasoning with Scientific Core Concepts](http://arxiv.org/abs/2601.16965)

- Spatial-Agent: introduces an AI agent grounded in spatial information science that formalizes geo-analytical question answering as a concept transformation problem using GeoFlow Graphs, Template Library, and Operator Library.
- The agent operates through a multi-stage pipeline, including Spatial Information Theory Analysis and GeoFlow Graph Construction, to generate ordered, constrained, and executable geospatial workflows.
- Spatial-Agent leverages core spatial concepts (Object, Field, Event) and functional roles (Support, Measure, Condition) to bridge natural language reasoning with computational GIS tools via template-based generation and LLM fine-tuning.

---

[AgentDrive: An Open Benchmark Dataset for Agentic AI Reasoning with LLM-Generated Scenarios in Autonomous Systems](http://arxiv.org/abs/2601.16964)

- AgentDrive: introduces a unified, generative, simulation-grounded, and reasoning-oriented open benchmark dataset for evaluating agentic AI systems in autonomous driving.
- The benchmark suite comprises AgentDrive-Gen (300K LLM-generated scenarios), AgentDrive-Sim (simulation rollouts and safety metrics), and AgentDrive-MCQ (100K reasoning questions across five styles).
- AgentDrive uses a factorized scenario space across seven orthogonal axes and an LLM-driven prompt-to-JSON pipeline to ensure semantic richness, physical validity, and simulation readiness.

---

[MAGE-KT: MULTI-AGENT GRAPH-ENHANCED KNOWLEDGE TRACING WITH SUBGRAPH RETRIEVAL AND ASYMMETRIC FUSION](http://arxiv.org/abs/2601.16886)

- MAGE-KT (Multi-Agent Graph-Enhanced Knowledge Tracing): introduces a novel framework for knowledge tracing using a Multi-agent KC Relation Extraction Module (generates/adjudicates KC relations), a Multi-relational KC Graph (inter-KC relations), a Student-Question Interaction Graph (S-Q dynamics/ability/difficulty signals), an IRT Model (estimates ability/difficulty), a Student-conditioned Subgraph Retriever (selects high-value subgraphs), an Interaction Encoder (encodes S-Q interactions), a KC Encoder (encodes KC relations), an Asymmetric Cross-attention Fusion Module (fuses multi-view evidence), CrossAtt (injects signals), Gate Fusion (computes gated sum), a GRU (maintains student state), and a Prediction Result (forecasts performance).
- The framework constructs a heterogeneous graph by integrating a multi-agent pipeline for accurate KC relation extraction and an S-Q interaction graph incorporating IRT-derived abilities and difficulties.
- MAGE-KT employs student-conditioned subgraph retrieval and an Asymmetric Cross-attention Fusion Module to ensure efficient computation and directed information flow for accurate next-question prediction.

---

[From Atom to Community: Structured and Evolving Agent Memory for User Behavior Modeling](http://arxiv.org/abs/2601.16872)

- STEAM (STructured and Evolving Agent Memory): introduces a novel memory framework that decomposes user preferences into fine-grained atomic memory units and organizes them into cross-user communities using prototype memories for collaborative signal propagation.
- The framework utilizes three core operations—Memory Construction, Retrieval, and Evolution—to adapt continuously to evolving user interests and maintain preference consistency.
- Memory Evolution incorporates consolidation to refine existing memories and formation to capture emerging interests, both triggered by LLM reflection on user behavior.

---

[Boosting Deep Reinforcement Learning with Semantic Knowledge for Robotic Manipulators](http://arxiv.org/abs/2601.16866)

- KGE-A3C Architecture: introduces a Deep Reinforcement Learning (DRL) agent architecture that integrates Knowledge Graph Embeddings (KGEs) with visual observations to enhance sample efficiency and performance in robotic manipulation tasks.
- The architecture uses convolutional layers for visual state representation, followed by an FC layer where KGEs are concatenated before being processed by an LSTM and separate A3C Actor and Critic FC heads.
- Experimental validation using TIAGo and IRB120 robotic manipulators shows up to 60% reduction in learning time and an accuracy improvement of up to 20% when using full semantic knowledge under domain randomization (DR).

---

[AI builds, We Analyze: An Empirical Study of AI-Generated Build Code Quality](http://arxiv.org/abs/2601.16839)

- Empirical Study: introduces the first empirical assessment of AI-generated build code quality across common build systems, leveraging the AIDev dataset and the Sniffer static analysis tool.
- The study analyzes 387 Agentic-PRs to quantify whether AI agents introduce or remove maintainability- and security-related code smells in build configuration files.
- Findings indicate that while AI agents can introduce smells like Wildcard Usage and Lack of Error Handling, they also effectively remove existing smells through refactoring actions, with most Agentic-PRs being accepted by developers.

---

[Will It Survive? Deciphering the Fate of AI-Generated Code in Open Source](http://arxiv.org/abs/2601.16809)

- ACSAF (AI Code Survival Analysis Framework): introduces a longitudinal study tracking over 200,000 code units from five LLM-powered AI agents across 201 open-source projects using survival analysis, modification intent classification, and predictive modeling.
- Contrary to the "disposable code" hypothesis, agent-authored code exhibits a significantly lower modification rate (15.8% lower hazard ratio at line level) and survives longer than human-authored code.
- Agent-authored code shows higher corrective and preventive modification rates, while predicting *when* modifications occur is fundamentally harder than predicting *which lines* will be modified.

---

[AN EFFICIENT INSECT-INSPIRED APPROACH FOR VISUAL POINT-GOAL NAVIGATION](http://arxiv.org/abs/2601.16806)

- Insect-inspired Model: introduces a novel, computationally lightweight agent for visual point-goal navigation, combining the Central Complex (CX) for path integration and control, and the Mushroom Body (MB) for associative learning and visual memory modulation.
- The agent uses odometry, collision, and vision inputs to calculate a desired target direction, which is modulated by the MB based on learned repulsive visual memories for obstacle avoidance.
- The model achieves performance comparable to recent SOTA models in Habitat and iGibson simulators using minimal online learning and significantly less computational cost, demonstrating robustness against motor noise and bias.

---

[GTA: Generative Traffic Agents for Simulating Realistic Mobility Behavior](http://arxiv.org/abs/2601.16778)

- GTA (Generative Traffic Agents): introduces a modular framework for large-scale, context-sensitive mobility simulation using LLM-powered agents grounded in Census Population Statistics (Grounding/Sociodemographic Attributes) and integrated with traffic simulators like SUMO/OTP (Route Computation/Traffic Simulation).
- The architecture consists of a Profile Module for generating diverse personas, a Planning Module for activity scheduling and trip planning using LLM reasoning, and an Action Module for dynamic routing and traffic simulation via Dynamic User Equilibrium (DUE).
- GTA enables empirically grounded simulations that capture the influence of personal and socioeconomic factors on mobility choices, validated against real-world traffic counts and survey data.

---

[SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents](http://arxiv.org/abs/2601.16746)

- SWE-Pruner (Self-Adaptive Context Pruning for Coding Agents): introduces a middleware framework that performs task-aware, line-level pruning of long code contexts retrieved by coding agents.
- The framework utilizes a lightweight 0.6B neural skimmer, guided by an explicit Goal Hint from the agent, to dynamically select relevant lines and preserve syntactic structure.
- SWE-Pruner achieves substantial efficiency gains, reducing token consumption by 23–54% and interaction rounds by 18–26% on multi-turn agent tasks like SWE-Bench Verified, with minimal performance degradation.

---

[LongCat-Flash-Thinking-2601 Technical Report](http://arxiv.org/abs/2601.16725)

- LongCat-Flash-Thinking-2601 (560B MoE reasoning model): introduces a powerful MoE model built on a Unified Training Framework (pre-training to post-training co-design) and Hybrid Data Synthesis Framework (agentic trajectory construction), achieving superior agentic reasoning performance.
- The approach utilizes a scalable Environment Scaling Pipeline (executable domain graph construction) and the Dynamic ORchestration for Asynchronous Rollout (DORA) (scalable multi-environment RL) system for stable, efficient large-scale agentic reinforcement learning.
- For test-time scaling, the model incorporates a Heavy Thinking Mode (test-time reasoning scaling) that jointly expands reasoning width and depth, and uses Zigzag Attention (sparse attention mechanism) for long-context efficiency up to 1M tokens.

---

[Watching AI Think: User Perceptions of Visible Thinking in Chatbots](http://arxiv.org/abs/2601.16720)

- Thinking Content and Conversational Context Framework (TCCC): investigates user perceptions of visible chatbot thinking (Thinking Content) across different help-seeking domains (Conversational Context), using an LLM (GPT-4o) to generate reflective statements and final suggestions.
- The study employs a 3x2 mixed factorial design, manipulating Thinking Content (Emotionally-Supportive, Expertise-Supportive, None) and Conversational Context (HABIT-related, FEELINGS-related problems).
- Findings indicate that visible thinking, particularly when framed as Emotionally-Supportive, significantly increases perceived empathy and warmth compared to the None condition.

---

[EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents](http://arxiv.org/abs/2601.16690)

- EMemBench: introduces a programmatic benchmark for evaluating the episodic memory of VLM/LLM agents through interactive games, generating individualized, verifiable QA sets from agent trajectories.
- The framework operates in three phases: agent interaction to produce a trajectory, programmatic generation of QA with ground truth, and evaluation where the agent answers questions using its memory.
- The benchmark reveals that induction and spatial reasoning remain persistent bottlenecks for memory agents, especially in visual settings, with current systems achieving low overall accuracy.

---

[AgentsEval: Clinically Faithful Evaluation of Medical Imaging Reports via Multi-Agent Reasoning](http://arxiv.org/abs/2601.16685)

- AgentsEval (Multi-Agent Stream Reasoning Framework): introduces a multi-agent, stream reasoning framework that emulates the collaborative diagnostic workflow of radiologists to evaluate medical imaging reports.
- The framework decomposes evaluation into sequential, interpretable stages—criteria definition, evidence extraction, alignment, and consistency scoring—using specialized LLM-based agents.
- AgentsEval provides clinically faithful, robust, and interpretable evaluations by generating explicit reasoning traces and structured clinical feedback, outperforming traditional lexical and single-agent metrics.

---

[Sim-to-Real Transfer via a Style-Identified Cycle Consistent Generative Adversarial Network: Zero-Shot Deployment on Robotic Manipulators through Visual Domain Adaptation](http://arxiv.org/abs/2601.16677)

- SICGAN (Style-Identified Cycle Consistent Generative Adversarial Network): proposes a novel domain adaptation approach for sim-to-real zero-shot transfer in robotics by translating raw virtual observations into real-synthetic images using a CycleGAN-based architecture enhanced with demodulated convolutions and identity loss.
- The pipeline trains a Deep Reinforcement Learning (DRL) agent using the Asynchronous Advantage Actor-Critic (A3C) algorithm on these real-synthetic observations in a virtual environment before deploying it directly to the real environment.
- The methodology is validated on two distinct 6-DoF robotic manipulators (ABB IRB120 and UR3e) performing a pick-and-place approaching task, achieving robust zero-shot transfer accuracies above 95% for most workspace regions.

---

[LUMINA: Long-horizon Understanding for Multi-turn Interactive Agents](http://arxiv.org/abs/2601.16649)

- LUMINA (Long-horizon Understanding for Multi-turn Interactive Agents): introduces an Oracle Counterfactual Framework to measure the criticality of skills—planning, state tracking, and history pruning—in LLM agents solving multi-turn tasks.
- The framework uses procedurally-generated environments (ListWorld, TreeWorld, GridWorld) to enable systematic oracle interventions and isolate the contribution of each skill without confounding effects.
- Findings show that while oracle interventions generally improve success rates, their effectiveness varies significantly based on the LLM model size and the specific environment characteristics.

---

[A Cognitive Framework for Autonomous Agents: Toward Human-Inspired Design](http://arxiv.org/abs/2601.16648)

- Human-Inspired Reinforcement Learning (RL) Architecture: introduces a dual-system RL architecture integrating Pavlovian and instrumental processes to enhance decision-making in autonomous agents, leveraging radio-frequency cues as conditioned stimuli.
- The architecture combines a Pavlovian-Instrumental Model-Free subsystem for habitual control and a Model-Based RL subsystem for goal-directed planning, regulated by an Arbitration Module based on prediction errors (RPE and SPE).
- Simulation results demonstrate that cue-driven agents adapt faster and achieve superior navigation performance compared to traditional instrumental-solo agents in unknown, partially observable environments.

---

[Evolutionary Dynamics of Reputation-Based Voluntary Prisoner's Dilemma Games](http://arxiv.org/abs/2601.16643)

- Reputation-Based Voluntary Prisoner's Dilemma Game: introduces a model extending the Prisoner's Dilemma by incorporating reputation-conditioned exit decisions, utilizing Unconditional Cooperator, Unconditional Defector, Conditional Cooperative Exiter, and Conditional Defective Exiter strategies, governed by Evolutionary Dynamics on various Population Structures influenced by Exit Incentive ($\epsilon$) and Monitoring Cost ($c$).
- The model demonstrates that reputation-based exit sustains cooperation through stable coexistence in well-mixed populations when exit incentives exceed monitoring costs, and through multiple exit-incentive-dependent coexistence pathways in structured populations.
- Conditional exiters act as key mediators of cooperative coexistence, inducing local cyclic dominance and persistent population-level oscillations, particularly in structured networks like the regular lattice and small-world networks.

---

[ATTENTION-MOA: Enhancing Mixture-of-Agents via Inter-Agent Semantic Attention and Deep Residual Synthesis](http://arxiv.org/abs/2601.16596)

- Attention-MoA: introduces a novel MoA-based framework that redefines collaboration through Inter-agent Semantic Attention and Inter-layer Residual Connections, complemented by an Adaptive Early Stopping Mechanism.
- The architecture uses Inter-agent Semantic Attention for explicit peer-critique and self-refinement among Collaborative Agents, mitigating hallucinations and refining logic within each layer.
- The Inter-layer Residual Module maintains a historical context stack to prevent information degradation in deep layers, enabling scalable performance gains and superior reasoning capabilities.

---

[Who You Explain To Matters: Learning by Explaining to Conversational Agents with Different Pedagogical Roles](http://arxiv.org/abs/2601.16583)

- RCA: introduces a study comparing four conversational agent roles (Tutee, Peer, Challenger, Control) powered by OpenAI GPT-4o to systematically measure their influence on learning-by-explaining dynamics.
- The Tutee agent elicited the highest cognitive investment but also high pressure, while the Peer and Challenger agents fostered better flow states and critical thinking with lower pressure.
- The findings suggest that educational agents should be tailored to specific pedagogical goals and learning phases rather than aiming for a single optimal role.

---

[Zero-Shot MARL Benchmark in the Cyber-Physical Mobility Lab](http://arxiv.org/abs/2601.16578)

- SigmaRL (Multi-Agent Reinforcement Learning Framework) / CPM Lab (Cyber-Physical Mobility Lab) Benchmark: introduces a reproducible platform for evaluating zero-shot sim-to-real transfer of MARL policies for Connected and Automated Vehicles (CAVs), integrating simulation, a high-fidelity digital twin, and a physical testbed.
- The platform, based on the Cyber-Physical Mobility Lab (CPM Lab), enables structured evaluation and direct comparison across three levels of realism: simulation, digital twin, and physical deployment.
- Evaluation using SigmaRL-trained policies revealed performance degradation stemming from both architectural differences (direct control vs. MPC) and increasing environmental realism (sim-to-real gap).

---

[Agentic AI-RAN Empowering Synergetic Sensing, Communication, Computing, and Control](http://arxiv.org/abs/2601.16565)

- Agentic AI-RAN: introduces a task-oriented architecture enabling Sensing, Communication, Computing, and Control (SC3) task execution within a single edge node, leveraging MIG-Partitioned GPU for hardware isolation.
- The core Agentic Brain, a cognitive orchestration layer, uses an LLM Task Planner and Contextual Memory to translate high-level intent into latency-aware actions via a unified SC3 execution loop.
- This architecture provides a scalable foundation for mission-critical low-altitude wireless networks, supporting autonomous drone navigation and embodied intelligence with low closed-loop latency.

---

[Curate-Train-Refine: A Closed-Loop Agentic Framework for Zero Shot Classification](http://arxiv.org/abs/2601.16530)

- Curate-Train-Refine (CTR) Framework: introduces a closed-loop agentic approach where an LLM iteratively curates synthetic training data to supervise the training of a lightweight text classifier for zero-shot classification.
- The LLM Agent operates in a generate-evaluate-refine cycle, generating seed data, analyzing the classifier's performance metrics and error modes, and synthesizing targeted examples to address identified deficiencies.
- This method decouples training from deployment, enabling accurate, efficient, low-latency inference without relying on expensive LLM inference at test time.

---

[SycoEval-EM: Sycophancy Evaluation of Large Language Models in Simulated Clinical Encounters for Emergency Care](http://arxiv.org/abs/2601.16529)

- SycoEval-EM: introduces a multi-agent simulation framework for adversarial testing of 20 LLM clinical agents using simulated patient-provider dialogues, featuring Patient, Doctor, and Evaluator Agents, Clinical Scenarios, and Persuasion Tactics.
- The framework evaluates LLM robustness to patient pressure in emergency medicine scenarios by measuring the acquiescence rate to unindicated medical requests, finding rates ranging dramatically from 0% to 100% across models.
- Models showed systematically higher vulnerability to imaging requests (38.8%) compared to opioid prescriptions (25.0%), and all five persuasion tactics proved comparably effective (30.0-36.0%).

---

[REprompt: Prompt Generation for Intelligent Software Development Guided by Requirements Engineering](http://arxiv.org/abs/2601.16507)

- REPROMPT: introduces a multi-agent prompt optimization framework guided by requirements engineering, including Interviewee, Interviewer, CoTer, and Critic agents, where the framework refines initial system or user prompts through four requirements development stages.
- The framework systematically optimizes prompts by simulating requirements engineering activities: elicitation (interviewing), analysis (SRS drafting), specification (CoT generation), and validation (structural scoring by the Critic agent).
- Experiments demonstrate that REPROMPT consistently improves the quality of generated software artifacts (PRD and SDD) and overall user satisfaction across various foundation LLMs (GPT-5, GPT-4, Qwen2.5-Max).

---

[EvoConfig: Self-Evolving Multi-Agent Systems for Efficient Autonomous Environment Configuration](http://arxiv.org/abs/2601.16489)

- EvoConfig (Self-Evolving Multi-Agent Systems for Efficient Autonomous Environment Configuration): introduces a multi-agent framework for efficient environment configuration, including an Environment Info Extraction Module (prior signals), a Main Environment Configuration Module (action execution), and a Self-Evolving Expert Diagnosis Module (fine-grained analysis).
- The framework utilizes a Main Agent for interactive environment setup and an Expert Agent for analyzing execution feedback, categorizing outcomes (success, failure, risk), and providing structured repair guidance.
- EvoConfig incorporates a self-evolving mechanism that enables the Expert Agent to continuously learn from error correction cases and dynamically adjust its internal rules and error-fixing priorities in real time.

---

[Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic](http://arxiv.org/abs/2601.16486)

- Timely Machine: introduces a framework redefining test-time scaling in agentic scenarios as wall-clock time, utilizing the Agent System (Coordinates interaction), Model (LLM) (Time-aware reasoner), Tool Server (Executes external tools), Timer (Tracks wall-clock time), Timely-RL (Time-aware RL training), Timer Tool (Reports elapsed duration), Code Execution Service (Runs ML code), and Jericho Game Engine (Hosts interactive games).
- The approach proposes Timely-RL, a reinforcement learning method that trains LLMs to develop intrinsic time budget awareness and dynamically adjust reasoning strategies based on observed tool latency feedback.
- Timely-Eval, a new benchmark, is introduced to evaluate time awareness across high-frequency tool calls, low-frequency tool calls, and time-constrained general reasoning tasks.

---

[TL-GRPO: Turn-Level RL for Reasoning-Guided Iterative Optimization](http://arxiv.org/abs/2601.16480)

- TL-GRPO (Turn-Level GRPO): introduces a lightweight RL algorithm for reasoning-guided iterative optimization tasks, utilizing turn-level group sampling and a unified verifiable turn-level reward function for fine-grained optimization.
- The approach modifies the standard GRPO rollout strategy to perform turn-level group sampling, enabling history-conditioned advantage estimation at each turn without additional sampling cost.
- Evaluated on Analog Circuit Sizing (ACS), TL-GRPO achieves state-of-the-art performance and strong generalization by optimizing the LLM agent's reasoning process based on maximum turn reward rather than cumulative returns.

---

[Doc2AHP: Inferring Structured Multi-Criteria Decision Models via Semantic Trees with LLMs](http://arxiv.org/abs/2601.16479)

- Doc2AHP: introduces a structured inference framework guided by Analytic Hierarchy Process (AHP) principles, leveraging Hierarchical Clustering (semantic structure mining), a Multi-Agent Expert Team (collaborative judgment), and a Consistency Optimization Module (mathematical rigor enforcement) to transform unstructured documents into computable decision models.
- The framework operates in two phases: Structure Generation, which mines semantic geometry via hierarchical clustering to instantiate the AHP skeleton, and Weight Estimation, which uses a Leader-Guided Multi-Agent Collaborative Mechanism.
- Weight estimation employs an adaptive consistency optimization strategy to synthesize multi-agent judgments and enforce mathematical constraints, effectively mitigating stochastic errors and numerical inconsistencies inherent to LLMs.

---

[DeepEra: A Deep Evidence Reranking Agent for Scientific Retrieval-Augmented Generated Question Answering](http://arxiv.org/abs/2601.16478)

- DeepEra (Deep Evidence Reranking Agent): introduces an agentic reranker that integrates Intention Recognition (extracts structured query representation), Relevance Assessment (LLM-based scoring function), Evidence Filtering Module (removes irrelevant passages), and Evidence Summarization Module (condenses retained passages) to enhance scientific question answering robustness.
- DeepEra operates within a two-stage Retrieval-Augmented Generation (RAG) pipeline, focusing on mitigating the impact of Semantically Similar but Logically Irrelevant (SSLI) passages by modeling logical relevance beyond surface similarity.
- The approach achieves superior retrieval performance and up to 8% relative improvements in retrieval robustness and answer accuracy compared to leading rerankers on the newly constructed SciRAG-SSLI dataset.

---

[Secure Intellicise Wireless Network: Agentic AI for Coverless Semantic Steganography Communication](http://arxiv.org/abs/2601.16472)

- AgentSemSteCom: introduces an Agentic AI-driven coverless semantic steganography scheme for secure intellicise wireless networks, utilizing semantic extraction, digital token controlled reference image generation, conditional diffusion model-based coverless steganography, JSCC-based semantic codec, and optional task-oriented enhancement modules.
- The scheme eliminates the need for cover images and private semantic keys by using a user-defined Digital Token for deterministic noise initialization and latent perturbation, coupled with EDICT (Exact Diffusion Inversion via Coupled Transformations) for invertible diffusion sampling and exact secret image recovery.
- Agentic AI autonomously coordinates public semantic keys generated by LLMs and implicit structural features to guide controllable stego image generation, achieving superior transmission quality and security against intelligent semantic eavesdroppers.

---

[Emotion-LLaMAv2 and MMEVerse: A New Framework and Benchmark for Multimodal Emotion Understanding](http://arxiv.org/abs/2601.16449)

- Emotion-LLaMAv2: introduces an end-to-end multimodal emotion large language model framework, featuring a Multiview Encoder (end-to-end perception), a Conv-Attention Pre-fusion Module (local and global fusion), a Modal Adapter (representation alignment), an LLM Backbone (reasoning), LoRA Tuning (parameter-efficient fine-tuning), and a Perception-to-Cognition Training Framework (curriculum instruction tuning).
- The framework eliminates explicit face detection by using an end-to-end multiview encoder that captures nuanced emotional cues via richer spatial and temporal tokens.
- The accompanying MMEVerse benchmark aggregates twelve public emotion datasets into a unified multimodal instruction format for large-scale training and standardized evaluation.

---

[Endless Terminals: Scaling RL Environments for Terminal Agents](http://arxiv.org/abs/2601.16443)

- ET (Endless Terminals): introduces a fully autonomous procedural generation pipeline for terminal-use tasks, including Task Description Generation, Container Setup, Completion Tests Generation, and Solution Filtering, yielding 3255 verified tasks for training Terminal Agents using PPO.
- The pipeline generates diverse, verifiable tasks spanning file operations, log management, and scripting, enabling simple RL setups to achieve substantial performance gains on held-out development sets and human-curated benchmarks like TerminalBench 2.0.
- The approach demonstrates that scaling environments autonomously provides a reliable training signal, leading to improvements that transfer across various LLM base models without complex agentic scaffolds.

---

[Gen-DBA: Generative Database Agents (Towards a Move 37 for Databases)](http://arxiv.org/abs/2601.16409)

- Gen-DBA (Generative Database Agents): introduces a foundational generative model for database systems, unifying diverse learning tasks via a Transformer Backbone and a two-phase Goal-conditioned Next Token Prediction training paradigm.
- The architecture leverages hardware-grounded tokenization (DB-Tokens) to unify multi-modal signals (SQL, telemetry, knobs) into a shared embedding space for cross-task reasoning.
- Gen-DBA aims to achieve a "Move 37" moment for AI4DB systems by shifting focus from purely performance-driven learning to knowledge-augmented, generative policy synthesis.

---

[Clarify or Answer: Reinforcement Learning for Agentic VQA with Context Under-specification](http://arxiv.org/abs/2601.16400)

- CoA (Clarify-or-Answer): introduces an ask-or-answer agentic framework for context-dependent Visual Question Answering (VQA) that explicitly separates the decision to ask (Controller) from what to ask (Clarification Policy), and produces a final answer using the Answering Model, potentially incorporating external Context.
- The framework utilizes GRPO-CR (Clarification Reasoning), a reinforcement learning approach, to optimize the Clarification Policy using multi-signal Reward Components tailored for context underspecification in VQA.
- The approach is evaluated using CONTEXTCLARIFY, a new dataset of ambiguous and non-ambiguous VQA instances, demonstrating consistent performance gains across three Vision-Language Models (VLLMs).

---

[Toward Agentic Software Project Management: A Vision and Roadmap](http://arxiv.org/abs/2601.16392)

- Agentic PM (Agentic Project Manager): introduces a vision for Agentic Software Project Management (SPM 3.0) using an ethical multi-agent PM assistant that perceives tasks, decides actions, and executes them via specialized sub-agents based on four adaptable working modes.
- The Agentic PM system includes a Coordinating Agent that manages task delegation to specialized sub-agents (e.g., Planning, Email, Risk, Meeting agents) and utilizes a Central Database for learning and adaptation based on Human PM feedback.
- The framework defines four working modes (Guided AI-Autonomy, Supervised-AI, Human-AI Collaborative, and AI-Assisted) to balance agent autonomy and human oversight based on task complexity and risk level.

---

[COGNITIVELY-INSPIRED TOKENS OVERCOME EGOCENTRIC BIAS IN MULTIMODAL MODELS](http://arxiv.org/abs/2601.16378)

- Perspective Tokens: introduces a method to overcome egocentric bias in MLMs by embedding cognitively grounded spatial structure directly into token space using specialized tokens derived from either embodied keypoints or abstract object orientation.
- The approach utilizes two types of tokens—Embodiment Tokens for human-centric perspective-taking and Rotation Tokens for abstract mental rotation—integrated into LLaVA 1.5 13B via an expanded tokenizer and curriculum learning.
- Integrating these tokens significantly improves performance on level-2 visual perspective-taking tasks across synthetic and naturalistic benchmarks, demonstrating that LLMs require structured spatial encodings for viewpoint transformation.

---

[PolyAgent: Large Language Model Agent for Polymer Design](http://arxiv.org/abs/2601.16376)

- PolyAgent: introduces a closed-loop LLM-powered agent framework for early-stage polymer discovery, integrating structure generation and property prediction capabilities via the Model Context Protocol (MCP).
- The system uses LLM reasoning (Gemini) to orchestrate specialized tools, including the Molecule Chef generative model and the TransPolymer predictive model, to refine polymer SMILES sequences.
- PolyAgent ensures synthetic accessibility by guiding polymer generation using the Synthetic Accessibility Score (SA Score) and Synthetic Complexity Score (SC Score) derived from the Open Macromolecule Genome (OMG) database.

---

#### 22nd January 2026


[LLM-in-Sandbox Elicits General Agentic Intelligence](http://arxiv.org/abs/2601.16206)

- LLM-in-Sandbox: introduces a paradigm enabling LLMs to explore within a Code Sandbox (virtual computer) using the execute\_bash, str\_replace\_editor, and submit tools, eliciting general intelligence for non-code domains.
- The framework utilizes a ReAct-based workflow where the LLM iteratively reasons and acts, leveraging the sandbox's meta-capabilities, including external resource access, file management, and code execution, to solve complex tasks.
- LLM-in-Sandbox-RL further enhances agentic capabilities by training models using reinforcement learning on general context-based data, leading to broad generalization across diverse domains and models.

---

[When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems](http://arxiv.org/abs/2601.16280)

- DF4TIR (Diagnostic Framework for Tool Invocation Reliability): introduces a systematic methodology for evaluating procedural reliability in multi-agent LLM systems using a 12-category error taxonomy, standardized evaluation protocols, and hardware-performance characterization.
- The framework utilizes a three-agent architecture (Email, Data Engineering, Reconciliation Agents) to manage coordinated tool use for structured business automation tasks like invoice reconciliation.
- Analysis across 1,980 test instances identifies tool initialization failures as the primary reliability bottleneck, establishing 14B and 32B parameter models as minimum viable and flawless production thresholds, respectively.

---


[International Joint Testing Exercise: Agentic Testing Advancing Methodologies for Agentic Evaluations Across Domains Leakage of Sensitive Information, Fraud and Cybersecurity Threats](http://arxiv.org/abs/2601.15679)

- Agentic Testing Exercise: introduces a joint international evaluation methodology assessing LLM agents' safety across two risk strands—sensitive information leakage/fraud and cybersecurity—using multiple LLM models, LLM judges, and human annotators across nine languages.
- The evaluation assessed how safe models are as agents and how effective models are as judges of agent behavior, finding that overall agent safety rates are lower in agentic tasks compared to previous conversational evaluations.
- Methodological findings highlight the complexity of translating code and tools, the need for realistic test design, and the higher discrepancy rates observed when using LLM judges compared to human evaluators, especially in nuanced risk scenarios.

---


[POINT BRIDGE: 3D REPRESENTATIONS FOR CROSS DOMAIN POLICY LEARNING](http://arxiv.org/abs/2601.16212)

- POINT BRIDGE: introduces a framework that leverages unified, domain-agnostic point-based representations, extracted via a VLM-guided scene filtering pipeline, to enable zero-shot sim-to-real policy transfer and multitask learning.
- The approach converts observations into a compact point-based representation (P) using a PointNet encoder, which is then fed into a transformer policy architecture alongside a language embedding for action prediction.
- By unifying representations across simulation and reality, the framework substantially outperforms prior vision-based sim-and-real co-training methods, achieving high success rates in single-task and multitask settings.

---


[Materealize: a multi-agent deliberation system for end-to-end material design and synthesis](http://arxiv.org/abs/2601.15743)

- Materealize: introduces a unified multi-agent system for end-to-end inorganic materials design and synthesis planning, featuring an Instant Mode (Rapid tool execution) and a Thinking Mode (Reasoning-driven synthesis analysis) based on a Multi-Agent Debate (MAD) Framework (Enables reasoning-driven planning).
- The Instant Mode rapidly orchestrates specialized ML tools for structure generation, property prediction, synthesizability assessment, and synthesis recipe prediction, enabling end-to-end inorganic design suggestions within minutes.
- The Thinking Mode employs five domain-specialized agents (Precursor, Thermodynamics, Surface & Kinetics, Literature, Judge) in a tool-grounded debate to deliver refined synthesis recommendations and mechanistic hypotheses, bridging computational discovery and practical realization.

---


[Delayed Assignments in Online Non-Centroid Clustering with Stochastic Arrivals](http://arxiv.org/abs/2601.16091)

- DGREEDY (Delayed Greedy): introduces a new model for online non-centroid metric clustering with delays (OCD), where the algorithm minimizes the total distance costs within clusters and the overall delay costs incurred by postponing assignments, using a deterministic greedy approach.
- The approach operates under the unknown i.i.d. (UIID) stochastic arrival model, where point locations are drawn independently from an unknown fixed probability distribution over a finite metric space.
- DGREEDY achieves a constant ratio-of-expectations (RoE), demonstrating constant competitiveness compared to an optimal offline clustering as the number of points grows.

---

[Keyframe-Based Feed-Forward Visual Odometry](http://arxiv.org/abs/2601.16020)

- KFF-VO (Keyframe-Based Feed-Forward Visual Odometry): introduces a keyframe-based feed-forward VO method leveraging the VGGT visual foundation model backbone and an RL-based keyframe selection policy.
- The approach uses reinforcement learning to derive an adaptive, data-driven keyframe policy that aligns input frame selection with the intrinsic characteristics of the foundation model.
- The system utilizes a fixed-size sliding window where an RL agent decides whether to retain the newest input frame as a keyframe based on observations derived from ViT CLS tokens and relative pose changes.

---

[Co-Constructing Alignment: A Participatory Approach to Situate AI Values](http://arxiv.org/abs/2601.15895)

- Co-Constructing Alignment: introduces a participatory workshop methodology to situate AI value alignment, utilizing a Misalignment Diary, structured phases for surfacing values, and generative design exercises for envisioning co-construction roles and interfaces.
- The approach frames alignment as an ongoing interactional practice, moving beyond model-centric views by treating users as epistemic agents who actively articulate, negotiate, and contest values during human-LLM interaction.
- The methodology surfaces misalignments as task or social breakdowns, leading participants to envision diverse co-construction roles such as adjusting, interpreting, limiting, and collective action, supported by novel interface features.

---

[EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience](http://arxiv.org/abs/2601.15876)

- EvoCUA (Evolving Computer Use Agents): introduces a native computer use agent that integrates verifiable synthesis, scalable interaction infrastructure, and iterative optimization into a self-sustaining evolutionary cycle.
- The Verifiable Synthesis Engine autonomously generates diverse tasks coupled with executable validators, implementing a "Generation-as-Validation" approach for precise supervision signals.
- The Scalable Interaction Infrastructure orchestrates tens of thousands of asynchronous sandbox rollouts, enabling massive trajectory acquisition for efficient policy refinement via Rejection Fine-Tuning and DPO.

---

[Minimum Envy Graphical House Allocation Beyond Identical Valuations](http://arxiv.org/abs/2601.15864)

- ME-GHA (Minimum Envy Graphical House Allocation): introduces a systematic study of the algorithmic complexity of house allocation with non-identical valuations, employing parameterized complexity techniques based on structural restrictions of the Social Graph ($G$) and the number of House Types ($l$).
- The research designs Fixed-Parameter Tractable (FPT) algorithms parameterized by Treewidth ($tw$), Vertex Cover Number ($vc$), and Clique Modulator ($k$) when the number of House Types ($l$) is constant.
- Exact exponential-time algorithms are developed for general non-identical valuations on specific graph classes, including trees and disjoint unions of graphs, utilizing Dynamic Programming and Min-Sum Subset Convolution.

---

[Introducing the Generative Application Firewall (GAF)](http://arxiv.org/abs/2601.15824)

- GAF (Generative Application Firewall): introduces a new architectural layer for securing LLM applications by unifying fragmented defenses into a single enforcement point across five security layers.
- The framework defines five interdependent security layers—Network, Access, Syntactic, Semantic, and Context—to provide defense-in-depth against both traditional and GenAI-specific threats.
- GAF addresses semantic and conversational threats, such as multi-turn jailbreaks and context manipulation, by maintaining session history and enforcing policies across inputs, outputs, and tool interactions.

---

[Virtual Traffic Police: Large Language Model-Augmented Traffic Signal Control for Unforeseen Incidents](http://arxiv.org/abs/2601.15816)

- Virtual Traffic Police (LLM-Augmented TSC Framework with Self-Refined TLRS): introduces a hierarchical framework where an upper-level LLM agent fine-tunes parameters for a lower-level adaptive Traffic Signal Control (TSC) system in response to unforeseen traffic incidents.
- The framework enhances reliability using a self-refined Traffic Language Retrieval System (TLRS) that employs Retrieval-Augmented Generation (RAG) to ground the LLM's reasoning in historical traffic language reports.
- An integrated LLM-based verifier continuously evaluates the generator's outputs and updates the TLRS with verified experiences, enabling dynamic adaptation to unseen traffic scenarios.

---

[Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification](http://arxiv.org/abs/2601.15808)

- DeepVerifier: introduces inference-time scaling of verification for Deep Research Agents (DRAs) using a three-stage multi-module framework, including decomposition, verification, and judge agents, guided by a DRA Failure Taxonomy.
- The framework exploits the asymmetry of verification by decomposing complex problems into verifiable information-retrieval sub-tasks, enabling the agent to self-improve via iterative feedback and refinement without additional training.
- DeepVerifier significantly outperforms baseline LLM judge methods in meta-evaluation F1 score and delivers 8%-11% accuracy gains on challenging benchmarks by providing detailed, rubric-based corrective feedback.

---

[Entangled Life and Code: A Computational Design Taxonomy for Synergistic Bio-Digital Systems](http://arxiv.org/abs/2601.15804)

- Computational Design Taxonomy (CDT): introduces a computational design taxonomy and vocabulary for bio-digital interfaces, featuring eight functional layers: Input, Transduction, Evaluation/Comparison, Routing/Selection, Memory/State, Adaptation, Output, and Power.
- The taxonomy is operationalized as a scaffold to interpret 70 existing bio-digital systems, describing the computational roles of biological and digital components across HCI, design, and engineering domains.
- The research contributes an open-source database and an interactive visualization platform to systematically explore relationships between organisms and digital components across temporal and spatial scales.

---

[AGENTIC CONFIDENCE CALIBRATION](http://arxiv.org/abs/2601.15778)

- HTC (Holistic Trajectory Calibration): introduces a novel diagnostic framework that extracts 48 rich process-level features from an agent's entire execution trajectory, including Dynamics, Position, Stability, and Structure, to train an Interpretable Calibration Model for accurate confidence calibration.
- The framework addresses compounding uncertainty and data scarcity in agentic systems by using a lightweight, sparse linear model that ensures interpretability, transferability, and generalization.
- A pre-trained variant, the General Agent Calibrator (GAC), achieves the best calibration (lowest ECE) on out-of-domain benchmarks, establishing a scalable foundation for trustworthy AI agents.

---

[UXCascade: Scalable Usability Testing with Simulated User Agents](http://arxiv.org/abs/2601.15777)

- UXCASCADE: introduces an end-to-end system for scalable usability testing using Simulation Agents (Generate interaction traces), Annotation Agents (Tag intent, detect issues), and Refinement Agents (Apply, evaluate interface edits).
- The system employs a multi-stage exploratory workflow to structure and aggregate LLM agent-generated feedback, linking reasoning traces to interface elements and persona traits.
- The framework supports rapid, iterative refinement by allowing practitioners to propose interface changes via the Editor Agent and automatically re-evaluate the modified version using the Preview Agent.

---

[Off-Policy Actor-Critic with Sigmoid-Bounded Entropy for Real-World Robot Learning](http://arxiv.org/abs/2601.15761)

- SigEnt-SAC (Sigmoid Bounded Soft Actor Critic): introduces an off-policy actor-critic method that learns from scratch using a single expert trajectory, incorporating Sigmoid-Bounded Entropy and Gated Behavior Cloning.
- The framework achieves rapid convergence and 100% success rates faster than baselines in one-shot settings by mitigating Q-value oscillations and preventing optimization toward out-of-distribution actions.
- SigEnt-SAC demonstrates robustness and cross-embodiment generalization across diverse robotic platforms using only single-view grayscale observations and sparse rewards.

---

[PhysProver: Advancing Automatic Theorem Proving for Physics](http://arxiv.org/abs/2601.15737)

- PhysProver: introduces a framework for formal physics theorem proving using a two-stage pipeline: Data Generation and Self-Evolving, leveraging Reinforcement Learning with Verifiable Rewards (RLVR) on the specialized PhysLeanData dataset.
- The Data Generation Stage uses the Claude-4.5-Sonnet LLM to synthesize conjectures, which are subsequently filtered by Lean syntax and verified for provability using multiple formal LLMs.
- The Self-Evolving Stage applies the Group Relative Policy Optimization (GRPO) algorithm to train the base DeepSeek-Prover-V2-7B model, guided by proof correctness rewards provided by the Lean verifier.

---

[DualShield: Safe Model Predictive Diffusion via Reachability Analysis for Interactive Autonomous Driving](http://arxiv.org/abs/2601.15729)

- DualShield: introduces a planning and control framework that leverages Hamilton-Jacobi (HJ) reachability value functions in a dual capacity for proactive guidance and reactive shielding.
- The framework unifies generative multimodal diffusion planning with formal safety assurance by steering the denoising process toward safe regions and filtering executed actions using a CBVF-QP.
- This dual-use architecture enables high mission success rates and efficient trajectories while maintaining principled safety against uncertain and adversarial interactions.

---

[Benchmarking Text-to-Python against Text-to-SQL: The Impact of Explicit Logic and Ambiguity](http://arxiv.org/abs/2601.15728)

- LCF (Logic Completion Framework): introduces a three-phase dialogue paradigm that resolves ambiguity by incorporating latent domain knowledge into the generation process using Subject and Oracle LLMs.
- The framework explicitly supplements missing specifications via natural language hints (Logic Clarifications) derived from ground truth, isolating reasoning deficiencies from information deficits.
- LCF enables Text-to-Python systems to achieve performance parity with Text-to-SQL by ensuring ambiguous natural language inputs are grounded in executable logical specifications.

---

[Towards Automated Kernel Generation in the Era of LLMS](http://arxiv.org/abs/2601.15727)

- LLM4Kernel: introduces a structured survey of LLM-driven kernel generation, spanning LLM-based approaches (Supervised Fine-Tuning/Reinforcement Learning), LLM Agents (Planning/Memory/Tool Use), Datasets (Training Corpora/Knowledge Bases), and Benchmarks (Metrics/Evaluation Datasets).
- The survey addresses the fragmented research landscape by compiling resources and highlighting emergent methodologies, including agentic optimization workflows for iterative refinement and automated kernel discovery.
- The integration of LLMs and LLM-based agents aims to automate high-performance kernel development, bridging the semantic gap between high-level algorithms and low-level hardware operations.

---

[VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning](http://arxiv.org/abs/2601.15724)

- VideoThinker: introduces an agentic VideoLLM trained entirely on synthetic tool-interaction trajectories, leveraging an LLM Agent to generate multi-step tool-use sequences in the caption space.
- The framework utilizes two complementary agentic tools, Temporal Retrieval and Temporal Zoom, enabling adaptive exploration and fine-grained reasoning over long videos.
- Training on the synthetic video-interleaved CoTs dataset equips the VideoLLM with dynamic reasoning capabilities and multi-step tool use, bridging the gap between text-based and true video reasoning.

---

[DANCING IN CHAINS: STRATEGIC PERSUASION IN ACADEMIC REBUTTAL VIA THEORY OF MIND](http://arxiv.org/abs/2601.15715)

- RebuttalAgent: introduces the first framework to ground academic rebuttal in Theory of Mind (ToM) using the ToM-Strategy-Response (TSR) pipeline, which models reviewer mental state, formulates persuasion strategy, and generates a strategy-grounded response.
- The agent is trained on RebuttalBench, a large-scale synthetic dataset, via supervised fine-tuning followed by reinforcement learning optimized by a novel self-reward mechanism for scalable self-improvement.
- For reliable evaluation, the framework includes Rebuttal-RM, a specialized reward model trained on multi-source data that achieves high scoring consistency with human preferences.

---

[AGENTSM: Semantic Memory for Agentic Text-to-SQL](http://arxiv.org/abs/2601.15709)

- AGENTSM (Agent Semantic Memory): introduces a scalable, stable, and efficient agentic framework for Text-to-SQL that leverages structured semantic memory to capture and reuse prior execution trajectories.
- The framework utilizes a Planner Agent and a Schema Linking Agent, supported by Trajectory Synthesis and Retrieval, to eliminate redundant data exploration and improve reasoning consistency.
- AGENTSM incorporates Composite Tools to streamline multi-step query generation, reducing agent turns and token usage, achieving state-of-the-art accuracy on the Spider 2.0 Lite benchmark.

---

[D-Optimality-Guided Reinforcement Learning for Efficient Open-Loop Calibration of a 3-DOF Ankle Rehabilitation Robot](http://arxiv.org/abs/2601.15707)

- DOG-RL framework: introduces a simulation-guided reinforcement learning approach for efficient open-loop calibration of a 3-DOF ankle rehabilitation robot, utilizing the Kronecker-Product-Based Calibration Method, MIMO System, PPO Agent, Policy Network, Value Network, Shared Encoder, D-Optimality Criterion, and Simulation Environment.
- The framework poses calibration posture selection as a combinatorial design-of-experiments problem, where the PPO agent is trained to select a small subset of informative postures maximizing the determinant of the information matrix.
- Real-world validation confirms that the PPO-based policy significantly reduces calibration effort while maintaining robust parameter estimation and achieving low cross-episode variance compared to random selection.

---

[Agentic Uncertainty Quantification](http://arxiv.org/abs/2601.15703)

- AUQ (Dual-Process Agentic UQ): introduces a unified framework that transforms verbalized uncertainty into active, bi-directional control signals using System 1 (Uncertainty-Aware Memory) for fast propagation and System 2 (Uncertainty-Aware Reflection) for slow, targeted resolution.
- System 1 implicitly propagates confidence and semantic explanations to prevent blind decision-making, while System 2 uses these explanations as rational cues to trigger inference-time correction only when confidence falls below a threshold.
- The framework dynamically balances efficient execution and deep deliberation, effectively mitigating the "Spiral of Hallucination" by preventing early epistemic errors from propagating irreversibly in LLM agents.

---

[From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](http://arxiv.org/abs/2601.15690)

- UACS (Uncertainty-as-a-Control-Signal): introduces a survey charting the functional evolution of uncertainty quantification (UQ) from a passive diagnostic metric to an active, real-time control signal guiding LLM behavior.
- The survey organizes this evolution across three frontiers: advanced reasoning, autonomous agents, and reinforcement learning (RL) and reward modeling.
- UQ, when used as an active signal, enables LLM systems to exhibit metacognitive self-awareness, allowing them to optimize cognitive effort and manage risk in complex, multi-step tasks.

---

[PERFORMANCE-GUIDED REINFORCED ACTIVE LEARNING FOR OBJECT DETECTION](http://arxiv.org/abs/2601.15688)

- MGRAL (Performance-guided Reinforced Active Learning): introduces a novel active learning framework for object detection that leverages a reinforcement learning-based sampling agent optimized by mAP improvement ($\Delta$mAP) as the reward signal.
- The approach addresses the non-differentiable batch selection challenge using policy gradient techniques and stabilizes training via a moving average baseline for reward normalization.
- MGRAL integrates an unsupervised surrogate detector and a fast lookup-table acceleration technique to mitigate the high computational cost of mAP estimation during RL iterations, ensuring practical deployment.

---

[FARM: Field-Aware Resolution Model for Intelligent Trigger-Action Automation](http://arxiv.org/abs/2601.15687)

- FARM (Field-Aware Resolution Model): introduces a two-stage architecture combining contrastive-trained dual encoders for high-recall candidate retrieval with a multi-agent LLM-based selection pipeline for precise trigger-action automation configuration.
- Stage 1 uses schema-enriched representations and layer freezing to train dual encoders (Trigger Encoder/Action Encoder) for efficient retrieval of top-k trigger and action candidates, which are ranked by the Cross-Score Matrix into a Priority Queue.
- Stage 2 employs a four-agent pipeline (Intent Analyzer Agent, Trigger Selector Agent, Action Selector Agent, Verifier Agent) coordinating via a Shared State Object to perform LLM-based re-ranking, selection, and generation of executable ingredient-to-field bindings.

---

[Bridging the Perception Gap: A Lightweight Coarse-to-Fine Architecture for Edge Audio Systems](http://arxiv.org/abs/2601.15676)

- CoFi-Agent (Tool-Augmented Coarse-to-Fine Agent): introduces a hybrid edge-cloud architecture for audio reasoning that uses a local coarse perception model and conditionally triggers cloud-guided refinement via on-device tools only for uncertain cases.
- The system operates on a semantic cascade principle, utilizing an Adaptive Confidence Gate to route easy samples via a Fast Path and escalating complex queries to an Investigate Path for targeted evidence extraction.
- By keeping raw audio local and transmitting only compact, symbolic evidence (transcripts, tool summaries) to the cloud reasoner, CoFi-Agent prioritizes acoustic privacy and achieves a strong accuracy-efficiency trade-off.

---

[StreetDesignAI: A Multi-Persona Evaluation System for Inclusive Infrastructure Design](http://arxiv.org/abs/2601.15671)

- StreetDesignAI: introduces an interactive system operationalizing persona-based multi-agent evaluation for inclusive cycling infrastructure design, integrating grounded context, multi-perspective evaluation, and visualization-driven iteration.
- The system uses fine-tuned LLMs (GPT-4.1) and generative AI (GPT-Image-1) to simulate feedback from four cyclist personas and a driver, enabling designers to iteratively adjust parameters and visualize street redesigns.
- By explicitly surfacing experiential conflicts and providing structured, comparative feedback, the system significantly improves designers' understanding of diverse user needs and confidence in making inclusive design decisions.

---

[AGENTIC AI GOVERNANCE AND LIFECYCLE MANAGEMENT IN HEALTHCARE](http://arxiv.org/abs/2601.15630)

- UALM (Unified Agent Lifecycle Management): introduces a five-layer framework to govern Agentic AI sprawl in healthcare by translating governance intent into an enforceable, audit-ready control plane.
- The framework layers address recurring gaps in AI governance, focusing on accountability, coordination, PHI continuity, runtime assurance, and end-to-end stewardship for multi-agent systems.
- Key architectural components include Non-Human Identity (NHI) certificates, a Policy-as-Code Engine, Vectorized PHI Sharding, and Kill Switch Protocols for real-time risk containment.

---

[Robust Tool Use via FISSION-GRPO: Learning to Recover from Execution Errors](http://arxiv.org/abs/2601.15625)

- FISSION-GRPO: introduces a reinforcement learning framework that dynamically converts execution errors into corrective training instances using a learned Error Simulator and a multiplicative Fission Mechanism.
- The approach operates in a closed loop across three stages: standard exploration, error identification/synthesis, and fission-based corrective batch training.
- By integrating error simulation directly into the RL training loop, the framework ensures alignment with the model's evolving error distribution, enabling robust error recovery in multi-turn tool use.

---

[CLOSING THE GAP ON THE SAMPLE COMPLEXITY OF 1-IDENTIFICATION](http://arxiv.org/abs/2601.15620)

- PSEEB (Parallel Sequential Exploration Exploitation on Brackets): introduces a novel multi-armed bandit strategy for 1-identification, utilizing multiple independent Sequential Exploration Exploitation (SEE) algorithm copies operating on nested Brackets defined by a Random Permutation.
- The approach derives new tight upper bounds on the expected total pulling times ($E[\tau]$) for positive instances, closing the gap to the derived lower bounds up to a polynomial of logarithm factor.
- The algorithm manages the execution of the algorithm copies using a Round-Robin Execution strategy, terminating when any copy outputs a definitive result ($\hat{a} \in [K] \cup \{\text{None}\}$).

---

[AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning](http://arxiv.org/abs/2601.15614)

- AION (Aerial Indoor Object-Goal Navigation): introduces an end-to-end dual-policy RL framework designed for 3D aerial ObjectNav, utilizing AION-e (exploration policy) and AION-g (goal-reaching policy) which share multi-modal features derived from RGB-D images, textual object classes, and altitude.
- The framework leverages specialized spatial features, including a 2D laser scan-like feature ($f_{dist}$) for obstacle avoidance and a Depth ROI feature ($f_{ROI}$) to guide mapless exploration in cluttered 3D environments.
- AION achieves superior zero-shot generalization and navigation safety by decoupling exploration and goal-reaching behaviors, validated on AI2-THOR and IsaacSim under realistic drone dynamics.

---

[Autonomous Business System via Neuro-symbolic AI](http://arxiv.org/abs/2601.15599)

- AUTOBUS (Autonomous Business System): introduces a neuro-symbolic AI architecture integrating LLM-based Core AI Agents, a Logic Engine, Humans, Enterprise Data, and Tools to orchestrate end-to-end business initiatives.
- The Core AI Agents synthesize task instructions, enterprise semantics, and available tools into task-specific logic programs, which the Logic Engine executes to enforce constraints and drive actions.
- Enterprise Data is structured as a knowledge graph providing semantic grounding, while Humans provide domain expertise, curate tools, and validate high-impact decisions.

---

[Screening for Choice Sets](http://arxiv.org/abs/2601.15580)

- Screening Framework for Choice Sets (SFCS): introduces a screening problem where an agent privately observes a set of feasible technologies and strategically discloses a subset to the principal, who commits to a decision rule maximizing expected payoff while ensuring truthful revelation.
- The optimal mechanism features a "bang-bang structure" for the Promised Utility Function $U(T)$, which either tracks the Complete Information Curve $u_c(T)$ or remains locally constant, bounded by the Monotone Envelope.
- The framework relies on the nested-set structure of technologies, simplifying the mechanism design problem to selecting a one-dimensional monotone promised utility function within an explicitly constructed envelope determined by the complete-information benchmark.

---

[Stabilizing Welfare-Maximizing Decisions via Endogenous Transfers](http://arxiv.org/abs/2601.15563)

- ET Framework: introduces a noncooperative model where self-interested agents form outcome-contingent contracts (endogenous transfers) prior to strategic voting under a fixed Social Choice Rule (SCF).
- The approach stabilizes collective decision-making by ensuring the existence of Individually Rational Strong Nash Equilibria (IR-SNE) under consensus rules, which implement welfare-maximizing outcomes supported by feasible transfers.
- The framework bridges cooperative and noncooperative game theory, achieving core-like stability and efficiency by characterizing necessary conditions for profitable deviations under general Anonymous, Monotonic, and Resolute (AMR) rules.

---

[ALIGNAgent: Adaptive Learner Intelligence for Gap Identification and Next-step guidance](http://arxiv.org/abs/2601.15551)

- ALIGNAgent (Adaptive Learner Intelligence for Gap Identification and Next-step guidance): introduces a multi-agent educational framework that integrates knowledge estimation, skill-gap identification, and targeted resource recommendation using LLM-based agents.
- The system utilizes a Skill Gap Agent to process student performance data and preferences, generating topic-level proficiency estimates and identifying conceptual weaknesses through diagnostic reasoning.
- The Recommender Agent and Summary Agent then translate these diagnostic insights into preference-aware learning materials and coherent, actionable learner feedback within a closed adaptive cycle.

---

[Game-to-Real Gap: Quantifying the Effect of Model Misspecification in Network Games](http://arxiv.org/abs/2601.16367)

- Game-to-Real Gap: introduces a metric quantifying the performance loss (difference between realized and predicted utility) caused by heterogeneous model misspecification in multi-agent systems, specifically analyzing Quadratic Network Games.
- The analysis shows that misspecifications in the External Shock or the Interaction Matrix can lead to arbitrarily large game-to-real gaps, even for arbitrarily small misalignments between players' conjectured models.
- The paper develops Shock Misspecification Centrality and Interaction-Graph Misspecification Centrality to exactly evaluate the game-to-real gap, demonstrating that standard network centrality measures fail to capture these effects.

---

[The Behavioral Fabric of LLM-Powered GUI Agents: Human Values and Interaction Outcomes](http://arxiv.org/abs/2601.16356)

- LLM-Powered GUI Agents: introduces an empirical investigation into how explicit human values and preferences, injected via personas, influence the reasoning and action trajectories of LLM-based web agents operating on replica websites.
- The study utilizes a controlled testbed of 14 interactive web tasks and four state-of-the-art agents, finding that value-infused prompts guide agents toward value-consistent outcomes, often through increased exploration and filter use.
- Agents exhibit a persistent efficiency bias and susceptibility to promotional UI cues, which frequently override explicit value guidance, leading to shorter action trajectories and rationalized decisions.

---

[DSGym: A Holistic Framework for Evaluating and Training Data Science Agents](http://arxiv.org/abs/2601.16344)

- DSGym: introduces a standardized, holistic framework for evaluating and training data science agents in self-contained execution environments, featuring an Agent (LLM wrapper, decision blocks), a central Manager (orchestrates execution, allocates workers), and isolated Worker (Isolated Docker container, Jupyter Kernel) environments.
- The framework curates DSGYM-TASKS, a rigorously audited task suite that includes general data analysis, expert-derived bioinformatics tasks (DSBIO), and challenging prediction tasks (DSPREDICT), filtered to enforce genuine data-dependent reasoning.
- DSGym supports agent training via execution-verified synthetic data generation, demonstrating that a small 4B model fine-tuned on this data can outperform frontier LLMs like GPT-40 on standardized analysis benchmarks.

---

[Chemotactic Feedback Controls Patterning in Hybrid Tumor-Stroma Model](http://arxiv.org/abs/2601.16337)

- Directionality-Damping Principle (Hybrid PDE-ODE framework): introduces a model coupling tumor phenotypes (S, R), a therapeutic agent (I), and stromal state switching (P, Fa) to analyze how chemotactic feedback controls spatial patterning in a damped system.
- The principle establishes that persistent spatial heterogeneity requires bidirectional (closed-loop) feedback to modify the effective mobility matrix, separating stable homogeneity, finite-band patterning (Turing-type), and aggregation regimes.
- The analysis provides global well-posedness and shows that the damped baseline model (no chemotaxis, drug washout) admits no diffusion-driven instability, establishing a stringent reference point for pattern generation via directed transport.

---

[AMBER: A Columnar Architecture for High-Performance Agent-Based Modeling in Python](http://arxiv.org/abs/2601.16292)

- AMBER (Agent-based Modeling with Blazingly Efficient Records): introduces a high-performance agent-based modeling framework that resolves Python ABM scalability issues by replacing object-per-agent representation with columnar state management using the Polars DataFrame library.
- The columnar architecture enables vectorized operations that execute in compiled Rust code, achieving speedups up to 93x for population-wide attribute updates and 38-45% reduction in peak memory usage compared to object-oriented frameworks.
- AMBER provides comprehensive ABM infrastructure, including spatial environments, experiment management, and optimization capabilities built upon the core columnar foundation.

---

[SemanticALLI: Caching Reasoning, Not Just Responses, in Agentic Systems](http://arxiv.org/abs/2601.16286)

- SemanticALLI: introduces a pipeline-aware architecture designed to operationalize the redundancy of reasoning by decomposing generation into Analytic Intent Resolution (semantic normalization) and Visualization Synthesis (implementation layer), treating intermediate representations as cacheable artifacts retrieved via a Hybrid Retrieval Engine (artifact retrieval).
- This approach uses Structured Intermediate Representation Caching (SIRC) at stable checkpoints to achieve high reuse rates, particularly in the VS stage (83.10% hit rate), significantly reducing LLM calls and latency.
- The Hybrid Retrieval Engine integrates exact hashing, dense semantic indexing, and lexical constraints (BM25/RRF) to ensure high precision and robustness against semantic collisions in high-cardinality domains like enterprise analytics.

---

[Generating Literature-Driven Scientific Theories at Scale](http://arxiv.org/abs/2601.16282)

- THEORIZER: introduces a system for literature-based scientific theory synthesis that generates qualitative and quantitative laws by discovering relevant papers, extracting structured evidence, and using LLMs to synthesize and refine candidate theories.
- The system generated 2,856 theories from 13,744 source papers, systematically comparing literature-supported generation against parametric LLM knowledge and accuracy-focused versus novelty-focused objectives.
- Evaluation uses a backtesting paradigm, assessing predictive accuracy by comparing theory predictions against empirical results reported in 4,554 subsequently published papers.

---

[GameTalk: Training LLMs for Strategic Conversation](http://arxiv.org/abs/2601.16276)

- GameTalk: introduces a framework for training LLMs to make strategic decisions via Multi-turn Interaction (Dialogue setting), utilizing adapted Fine-tuning Algorithms (RL-based policy optimization), Behavioral Signals (Strategic performance metrics), and a Private Chain of Thought (Complex reasoning promotion) within controlled Game Environment (Controlled strategic setting).
- The methodology optimizes a global objective across full conversations by adapting RL methods and introducing three novel Behavioral Signals (Internal State Evaluation, State-Relative Performance, Leverage Opportunity) for strategic analysis and targeted Reward Shaping.
- Experiments across games of increasing complexity demonstrate that GameTalk significantly improves performance over untrained baselines, with DPO consistently yielding the strongest gains in strategic conversational tasks.

---

#### 21st January 2026

[Accelerator and Brake: Dynamic Persuasion with Dead Ends](http://arxiv.org/abs/2601.13686)

- Accelerator and Brake (Dynamic Persuasion Framework): introduces optimal dynamic persuasion in a bandit experimentation model where the principal has a non-monotonic, single-peaked preference over the agent's stopping time, characterized by at most two one-shot disclosures: an Accelerator (pre-$t^*$) and a Brake (post-$t^*$).
- The optimal policy structure is derived by decomposing the non-monotonic problem into a motivation subproblem (before $t^*$) and a dissuasion subproblem (after $t^*$), linked by milestone commitment variables.
- The nature of disclosure (one-shot versus gradual) is determined by comparing the Arrow-Pratt coefficients of absolute risk aversion of the principal and agent, serving as a sufficient statistic for time-risk sensitivity.

---


[Vibe Coding Kills Open Source](http://arxiv.org/abs/2601.15494)

- Vibe Coding Model: introduces a general equilibrium model of the OSS ecosystem with Users (discrete choice demand system), Developers (entry/sharing decisions), and a Software Production Function (Cobb-Douglas technology) to study the equilibrium effects of AI-mediated software development.
- Vibe coding, an AI-mediated usage mode, affects the ecosystem through a Productivity Channel (lowering development costs) and a Demand-Diversion Channel (eroding engagement-based developer rewards).
- The analysis shows that under traditional OSS business models, the demand-diversion channel dominates the productivity gains, leading to reduced OSS provision, lower variety, and decreased welfare.

---


[Equal-Pay Contracts](http://arxiv.org/abs/2601.15478)

- Equal-Pay Contracts: introduces a study of multi-agent contract design where a principal incentivizes agents to take costly actions via a combinatorial reward function, constrained by the requirement that all agents receive identical payments.
- The research provides algorithmic and hardness results across a broad hierarchy of reward functions (additive, submodular, XOS, subadditive) under both binary and combinatorial action models.
- The paper quantifies the loss induced by fairness using the Price of Equality (PoE), establishing a tight bound of $O(\log n / \log \log n)$ for XOS rewards with combinatorial actions.

---


[Sample Complexity of Average-Reward Q-Learning: From Single-agent to Federated Reinforcement Learning](http://arxiv.org/abs/2601.13642)

- FARQL (Federated Average-Reward Q-Learning): introduces a simple yet effective Q-learning algorithm for average-reward MDPs, applicable to both single-agent and federated settings, achieving improved sample and communication efficiency.
- The federated approach utilizes M agents, each with a generative model, performing independent local Q-updates and periodic global aggregation coordinated by a central server.
- Principled scheduling of dynamic learning rates, discount factors, and communication intervals ensures linear speedup in sample complexity per agent and communication complexity independent of M.

---

[MOTION-TO-RESPONSE CONTENT GENERATION VIA MULTI-AGENT AI SYSTEM WITH REAL-TIME SAFETY VERIFICATION](http://arxiv.org/abs/2601.13589)

- MAERS (Multi-Agent Emotion-to-Response System): transforms audio-derived emotional signals into safe, real-time response content using four specialized agents operating sequentially.
- The system achieves 100% safety compliance and sub-100ms inference latency suitable for on-device deployment on resource-constrained devices.
- The modular architecture separates perception, decision-making, generation, and explicit safety verification, enhancing interpretability and controllability.

---

[How high-resolution agent-based models can improve fundamental insights in tissue development and cell culturing methods](http://arxiv.org/abs/2601.15273)

- DCM (Deformable Cell Model): introduces a high-resolution agent-based model representing cells via a triangulated surface mesh (Cortex) surrounding an isotropic fluid (Cytoplasm), governed by a Force Balance Equation for node dynamics.
- The model incorporates detailed biophysical characteristics, including Friction Terms (viscous resistance) and various Internal Forces (elasticity, volume, osmosis) and External Forces (adhesion, migration).
- DCM is applied to simulate complex multicellular systems like monolayers, spheroids, organoids, and bile canaliculi formation, demonstrating how subcellular scale processes affect tissue formation and growth.

---

[DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration](http://arxiv.org/abs/2601.15260)

- DrivIng (Large-Scale Multimodal Driving Dataset): introduces a large-scale multimodal dataset with a comprehensive geo-referenced Digital Twin, enabling real-to-sim mapping and systematic evaluation using its Sensor Suite, HD Map, and two simulation modes: Kinematic Replay Mode and Interactive Re-simulation Mode.
- The dataset covers an 18 km route across urban, suburban, and highway segments, providing 1.2 million annotated 3D bounding boxes across 12 classes under day, dusk, and night conditions.
- The Digital Twin facilitates robust validation by supporting scenario replay, environmental modification, and benchmarking using SOTA Perception Models like PETR and CenterPoint.

---

[Distributed Agent-Constrained Truthful Facility Location](http://arxiv.org/abs/2601.15258)

- Distributed Agent-Constrained Truthful Facility Location Mechanism: introduces a two-phase strategyproof mechanism for selecting $k$ facility locations from agent positions, partitioned into fixed groups, minimizing social cost under sum-variant or max-variant individual costs.
- The mechanism first selects a representative location for each group based on agent positions (Phase 1) and then chooses $k$ facility locations from the set of these representatives (Phase 2).
- The research establishes tight approximation ratio bounds for strategyproof mechanisms, achieving $1 + \sqrt{2}$ (sum-variant) and $9/2$ (max-variant) for the case of $k=2$ facilities.

---

[Taxonomy-Aligned Risk Extraction from 10-K Filings with Autonomous Improvement Using LLMs](http://arxiv.org/abs/2601.15247)

- Taxonomy-Aligned Risk Extraction Methodology: introduces a three-stage pipeline combining LLM extraction, embedding-based semantic mapping, and LLM validation to reliably map free-form risk descriptions to predefined taxonomy categories.
- The pipeline ensures high precision by using an LLM-as-a-judge to filter spurious assignments and generate systematic feedback for continuous taxonomy improvement.
- The methodology includes an Autonomous Taxonomy Refinement Agent that analyzes low-quality mapping feedback to diagnose failure patterns and propose description refinements, creating a continuous improvement loop.

---

[When Agents Fail: A Comprehensive Study of Bugs in LLM Agents with Automated Labeling](http://arxiv.org/abs/2601.15232)

- BugReAct: introduces an LLM-based ReAct agent system equipped with tools, a summarization module, and an in-memory database to automatically analyze and classify bugs in LLM agents based on six categories.
- The system leverages external tools to scrape documentation and community forums, using a classification component with an LLM (e.g., Gemini 2.5 Flash) to provide precise annotations and rationales for bug characteristics.
- An extensive empirical study analyzing 1,187 bug instances across Stack Overflow, GitHub, and Hugging Face revealed that Logic Bugs are the most common type, and the Agent Core is the most bug-prone component.

---

[Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](http://arxiv.org/abs/2601.15220)

- PCS-F (Privacy Collapse Study Framework): introduces "Privacy Collapse," a novel failure mode where benign fine-tuning on diverse datasets degrades LLMs' contextual privacy norms, as measured by Evaluation Benchmarks and analyzed via Mechanistic Analysis Tools.
- The study demonstrates that privacy representations encoded in late layers are uniquely fragile to fine-tuning, causing models to over-generalize helpfulness and inappropriately share sensitive information from persistent memory or tool-use contexts.
- Privacy collapse is identified as a "silent failure" because fine-tuned LLMs maintain strong performance on standard safety (AgentHarm) and capability (CommonSenseQA) benchmarks despite severe contextual privacy vulnerabilities (PrivacyLens and CIMemories).

---

[Real-time Facial Communication Restores Cooperation After Defection in Social Dilemmas](http://arxiv.org/abs/2601.15211)

- Facial Communication (FC) Treatment: investigates the influence of real-time facial expressions, captured by Biometric Technology and displayed via Gender-Neutral Avatars on cooperation rates in a repeated Prisoner's Dilemma Game.
- The system uses the iMotions Software and Affectiva AFFDEX Engine to measure and classify eight universal emotions in real-time, displaying the most intense emotion if it exceeds a 60% intensity threshold.
- The presence of facial communication significantly increases overall cooperation and promotes forgiveness, acting as a restorative mechanism following defection.

---

[Where Do AI Coding Agents Fail? An Empirical Study of Failed Agentic Pull Requests in GitHub](http://arxiv.org/abs/2601.15195)

- APRA (Agentic PR Failure Analysis): introduces an empirical study characterizing 33k agent-authored PRs across GitHub repositories, analyzing merge outcomes, code changes, CI results, and review dynamics.
- The study identifies that documentation, CI, and build update tasks have the highest merge success, while performance and bug-fix tasks exhibit the lowest acceptance rates.
- Analysis reveals that agentic PR failures often stem from reviewer abandonment, duplicate PRs, CI/test failures, and misalignment with repository workflows or developer expectations.

---

[Large-Scale Multi-dimensional Knowledge Profiling of Scientific Literature](http://arxiv.org/abs/2601.15170)

- LLM-Driven Multi-Dimensional Knowledge Profiling Framework: introduces a pipeline combining topic clustering, LLM-assisted semantic parsing, and structured retrieval to create a comprehensive, structured representation of AI research activity from over 100,000 scientific papers.
- The framework utilizes a multi-stage knowledge extraction process, including Markdown conversion and Deepseek-R1-32B parsing, to populate the ResearchDB schema with multi-dimensional data (Info, Summary, Technical, Analysis, System).
- The system supports evidence-based trend analysis and fine-grained topic investigation by employing an intent-driven hierarchical retrieval system that filters metadata and performs weighted multi-field semantic search.

---

[Automated Rubrics for Reliable Evaluation of Medical Dialogue Systems](http://arxiv.org/abs/2601.15161)

- RAMAF (Retrieval-Augmented Multi-Agent Framework): introduces a three-stage pipeline including Retrieval and Evidence Preparation, Dual-Track Constraint Construction, and Audit and Refinement, designed to automate the generation of instance-specific medical evaluation rubrics.
- The framework grounds evaluation in authoritative medical evidence by decomposing retrieved content into atomic facts and synthesizing them with user interaction constraints to form verifiable, fine-grained evaluation criteria.
- The multi-agent architecture, featuring planning-, synthesis-, fact-, intent-, rubric synthesis-, and auditing-agents, achieves superior Clinical Intent Alignment and discriminative sensitivity compared to GPT-4o baselines.

---

[How to Build AI Agents by Augmenting LLMs with Codified Human Expert Domain Knowledge? A Software Engineering Framework](http://arxiv.org/abs/2601.15153)

- AI Agent Framework: introduces a systematic software engineering framework for capturing and codifying human expert domain knowledge to construct LLM-based AI agents capable of autonomous expert-level visualization generation.
- The agent architecture integrates a Classifier (routes user requests), a RAG system (provides domain knowledge), Codified expert rules (generates analytical reports), and a Language Model (LLM) (generates Python code) unified by a Prompt constructor (creates final query).
- Empirical evaluation across five scenarios demonstrated a 206% improvement in output quality compared to a baseline LLM+RAG system, achieving consistent expert-level ratings by applying nuanced visualization rules.

---

[Modification speed and radius of higher-order interactions alter the oscillatory dynamics in an agent-based model](http://arxiv.org/abs/2601.15144)

- ABM/HOI Simulation: investigates the effects of higher-order interactions (HOIs) on oscillatory dynamics in a three-species intransitive competition system using a spatial grid environment.
- The model incorporates three key parameters—modification strength ($\beta$), radius ($R_{HOI}$), and speed ($\omega$)—to independently vary the spatio-temporal effects of the HOI on competition probability between two species.
- Monte Carlo Singular Spectrum Analysis (MCSSA) is adapted to quantify the amplitude of species abundance oscillations, demonstrating that modification speed and radius significantly impact oscillatory dynamics, even if they do not affect mean species abundances.

---

[CLEANER: SELF-PURIFIED TRAJECTORIES BOOST AGENTIC REINFORCEMENT LEARNING](http://arxiv.org/abs/2601.15141)

- CLEANER (Self-Purified Trajectories Boost Agentic Reinforcement Learning): introduces a trajectory purification framework centered on the SAAR mechanism, which constructs clean trajectories by retrospectively replacing execution failures with successful self-corrections.
- SAAR operates in two phases: Phase I triggers correction upon execution error, and Phase II adaptively determines replacement granularity (shallow or deep) based on the semantic similarity between the erroneous and corrected code.
- By training on these self-purified paths, the LLM internalizes correct reasoning patterns, mitigating the credit assignment issue caused by noisy exploration and achieving efficient agentic RL.

---

[Conversational AI for Social Good (CAI4SG): An Overview of Emerging Trends, Applications, and Challenges](http://arxiv.org/abs/2601.15136)

- CAI4SG (Conversational AI for Social Good): introduces a role-based framework that categorizes Conversational Agents (CAs) based on AI autonomy and AI emotional engagement, defining four distinct roles for social good applications.
- The framework spans a dynamic spectrum from functional assistants (low autonomy/low engagement) that streamline public services to virtual companions (high autonomy/high engagement) that provide sustained emotional support and relationship-building.
- This analytical lens guides the development of equitable, ethical, and effective CAI systems by anticipating benefits, risks, and governance needs associated with each role, including algorithmic bias and emotional dependency.

---

[Emerging from Ground: Addressing Intent Deviation in Tool-Using Agents via Deriving Real Calls into Virtual Trajectories](http://arxiv.org/abs/2601.15120)

- RISE (Real-to-Virtual): introduces a method to mitigate intent deviation in tool-using agents by synthesizing virtual trajectories from verified real tool calls and generating diverse negative samples via multi-type mutations on Intent-aware Critical Parameters (ICPs).
- The approach uses a two-stage training paradigm, combining SFT for basic capability calibration and DPO for fine-grained intent alignment, leveraging the synthesized positive and negative data.
- RISE significantly enhances intent alignment and task completion, achieving substantial improvements over SOTA baselines and demonstrating robust generalization to unseen scenarios.

---

[From Who They Are to How They Act: Behavioral Traits in Generative Agent-Based Models of Social Media](http://arxiv.org/abs/2601.15114)

- GABM (Generative Agent-Based Modeling): introduces behavioral traits as an explicit characterization layer alongside identity traits to regulate agent propensities across the full social media action space, including posting, re-sharing, commenting, reacting, and inactivity.
- The framework extends an existing GABM architecture by incorporating an Activity Memory (AM) component and modifying the recommendation system to enable content propagation chains through re-shared content.
- Large-scale simulations demonstrate that behavioral traits are essential for sustaining heterogeneous, profile-consistent participation patterns and reproducing realistic content propagation dynamics and network structures.

---

[An Agentic Operationalization of DISARM for FIMI Investigation on Social Media](http://arxiv.org/abs/2601.15109)

- Agentic DISARM Pipeline: introduces a multi-agent workflow for Foreign Information Manipulation and Interference (FIMI) investigation, operationalizing the DISARM framework by collaboratively detecting manipulative behaviors and mapping them to standardized Tactics, Techniques, and Procedures (TTPs).
- The pipeline uses an LLM agent across iterative investigation rounds, moving from Exploratory Data Analysis (EDA) and TTPs sampling to hypothesis generation, evidence gathering, and statistical verification.
- The approach ensures transparency and interoperability by decomposing complex findings into atomic evidence units that undergo statistical validation and human evaluation, providing a verifiable baseline for automated FIMI analysis.

---

[Facilitating Proactive and Reactive Guidance for Decision Making on the Web: A Design Probe with WebSeek](http://arxiv.org/abs/2601.15100)

- WebSeek (mixed-initiative browser extension): introduces a data-first paradigm for human-AI collaboration on the web, centered around tangible data artifacts on an interactive canvas.
- The system uses an LLM as a planner, outputting tool calls executed by a reliable tool-based architecture to provide both proactive (context-aware) and reactive (chat-based) guidance.
- This data-centric approach unifies data extraction, wrangling, and visualization within a single environment, promoting transparency, control, and high user confidence.

---

[Memory Retention Is Not Enough to Master Memory Tasks in Reinforcement Learning](http://arxiv.org/abs/2601.15086)

- Memory Rewriting Benchmarking (MRB): introduces two novel diagnostic environments, Endless T-Maze and Color-Cubes, to systematically evaluate the continual memory rewriting capabilities of various memory-augmented RL architectures, including recurrent, transformer-based, and structured external memory systems.
- The benchmarks explicitly test adaptive updating under partial observability, requiring agents to selectively discard obsolete cues and integrate new evidence as environmental conditions change.
- Experimental results establish a performance hierarchy, demonstrating that agents utilizing explicit, learnable forgetting mechanisms, such as PPO-LSTM, exhibit superior robustness and generalization in rewriting tasks compared to structured or attention-based models.

---

[Multi-Agent Constraint Factorization Reveals Latent Invariant Solution Structure](http://arxiv.org/abs/2601.15077)

- Multi-Agent Constraint Factorization (MACF): introduces a formal explanation for MAS effectiveness grounded in operator theory, modeling agents as constraint-enforcement operators acting sequentially on a shared solution state.
- The MACF framework proves that the factorized composition of these operators converges to invariant solution sets defined by the intersection of agent constraint sets, which are generally inaccessible to single-agent dynamics.
- This theoretical result is extended from exact constraint enforcement (projections) to soft constraints (proximal operators), demonstrating robustness for approximate, text-based LLM interactions.

---

[The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution](http://arxiv.org/abs/2601.15075)

- Agentic Attribution Framework: introduces a hierarchical, coarse-to-fine approach to identify the internal factors driving LLM-based agent actions, regardless of task outcome, using Component-Level Attribution (Localizes interaction steps) and Sentence-Level Attribution (Isolates textual evidence).
- The framework formalizes the agent's execution trajectory as a temporal sequence of components (USER, THOUGHT, TOOL, OBS, MEMORY) to quantify their contribution to the generation of a specific target action.
- Component localization is achieved via temporal likelihood dynamics, while fine-grained sentence attribution uses perturbation-based scoring (Prob. Drop & Hold) to pinpoint decisive textual evidence.

---

[SmartOracle - An Agentic Approach to Mitigate Noise in Differential Oracles](http://arxiv.org/abs/2601.15074)

- SMARTORACLE: introduces an autonomous agentic system for differential fuzzing triage, utilizing specialized LLM sub-agents and integrated tools to automate root cause analysis.
- The architecture decomposes the manual triage workflow into specialized sub-agents (Finder, Checker, Critic) that synthesize evidence from terminal runs and targeted specification queries.
- The system achieves 0.84 recall with an 18% false positive rate, significantly reducing analysis time by 4x and API costs by 10x compared to a sequential LRM baseline.

---

[The Responsibility Vacuum: Organizational Failure in Scaled Agent Systems](http://arxiv.org/abs/2601.15059)

- Responsibility Vacuum Analysis (RVA): characterizes a structural failure mode in scaled agent deployments where decision generation throughput (G) exceeds bounded human verification capacity (H), causing authority and capacity to structurally diverge.
- The responsibility vacuum arises when formal human approval persists as a requirement, but the epistemic basis for verification is systematically substituted by proxy signals, leading to ritual review.
- The CI Amplification Dynamic accelerates this failure by increasing automated validation signals (proxies) without restoring human capacity, thereby displacing engagement with primary artifacts and reducing effective verification capacity.

---

[Game-Theoretic Lens on LLM-based Multi-Agent Systems](http://arxiv.org/abs/2601.15047)

- Game-Theoretic Lens (GTL): introduces a systematic framework for surveying LLM-based Multi-Agent Systems (MAS) by organizing existing studies around the four core game-theoretic elements: Players, Strategies, Payoffs, and Information.
- This unified perspective facilitates the integration of classical game theory with modern LLM-driven research for systemic analysis of agent interactions, strategic reasoning, and emergent collective behaviors.
- The analysis reveals that while LLMs excel in strategic communication, significant gaps remain in robust equilibrium selection and incentive compatibility in complex, partially observable environments.

---

[A Curriculum-Based Deep Reinforcement Learning Framework for the Electric Vehicle Routing Problem](http://arxiv.org/abs/2601.15038)

- CB-DRL (Curriculum-Based Deep Reinforcement Learning): introduces a robust framework that stabilizes Deep Reinforcement Learning (DRL) training for the Electric Vehicle Routing Problem with Time Windows (EVRPTW) by decomposing problem complexity into three sequential phases.
- The framework utilizes a Curriculum Controller to gradually increase constraint difficulty, progressing from topology learning (Phase A) to energy management (Phase B) and finally full EVRPTW constraints (Phase C).
- The Neural Agent employs a specialized Heterogeneous Graph Attention Encoder and a modified Proximal Policy Optimization (PPO) algorithm with phase-specific parameters to ensure stable convergence and robust generalization.

---

[Visual and Cognitive Demands of a Large Language Model-Powered In-vehicle Conversational Agent](http://arxiv.org/abs/2601.15034)

- Visual and Cognitive Demands Evaluation Framework: assesses the cognitive and visual demands of the advanced LLM conversational agent, Gemini Live, during on-road driving by comparing it against hands-free phone calls, visual turn-by-turn guidance, and the high-load OSPAN task.
- Objective measures showed that Gemini Live interactions (single-turn and multi-turn) imposed cognitive load similar to hands-free phone calls, falling between the low-load guidance and the high-load OSPAN benchmark.
- All voice-based tasks maintained low visual demand, with mean glance durations well below the 2-second safety threshold, and cognitive load remained stable across extended multi-turn conversations.

---

[LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding](http://arxiv.org/abs/2601.15016)

- LiVi-LLM-7B: introduces an omnimodal video understanding model enhanced with interactive knowledge, utilizing a Visual Encoder, Audio Encoder, Transformer Decoder, LLM, VCR Module, and Two-stage Instruction Tuning.
- The model is specifically designed for interactive livestream videos, leveraging audio, speech, and real-time comments modalities for comprehensive understanding.
- The architecture incorporates a Video-to-Comment Retrieval (VCR) module during inference to efficiently select relevant comments from massive real-time user inputs.

---

[Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control](http://arxiv.org/abs/2601.15015)

- FluidGym: introduces the first standalone, fully differentiable benchmark suite for Active Flow Control (AFC) using the FluidEnv (RL-centric interface), Solver (GPU-accelerated PICT solver), and RL Agent (SARL/MARL) components.
- The benchmark is built entirely in PyTorch, requires no external Computational Fluid Dynamics (CFD) solvers, and provides standardized environments for 2D and 3D flow control tasks.
- FluidGym supports single-agent, multi-agent, and gradient-based control methods, enabling systematic comparison and transfer-learning studies across diverse flow configurations.

---

[Improving Regret Approximation for Unsupervised Dynamic Environment Generation](http://arxiv.org/abs/2601.14957)

- DEGen (Dynamic Environment Generation for UED) / MNA (Maximised Negative Advantage): introduces a dynamic environment generation method that constructs levels based on student observations and optimizes using the MNA regret approximation metric to improve generalization in Unsupervised Environment Design (UED).
- DEGen addresses the sparse reward and long time horizon issues of traditional learnt generators by providing a denser reward signal derived from partial level generation during student exploration.
- Maximised Negative Advantage (MNA) improves upon existing regret approximations by better identifying challenging levels and including an explicit penalization for approximately unsolvable environments.

---

[CORPUSQA: A 10 MILLION TOKEN BENCHMARK FOR CORPUS-LEVEL ANALYSIS AND REASONING](http://arxiv.org/abs/2601.14952)

- CorpusQA Data Generation Pipeline: introduces a novel six-step framework that programmatically generates complex, reasoning-intensive queries with verifiable ground-truth answers by decoupling reasoning from unstructured text representation.
- The pipeline ensures 100% factual accuracy by structuring information via Schema Extraction and deriving answers through NL2SQL Execution against an aggregated data table.
- CorpusQA is the first large-scale benchmark designed to evaluate corpus-level analysis, scaling up to 10 million tokens with high evidence dispersion, challenging existing LLMs and RAG systems.

---

[CodeDelegator: Mitigating Context Pollution via Role Separation in Code-as-Action Agents](http://arxiv.org/abs/2601.14914)

- CODEDELEGATOR: introduces a multi-agent framework that mitigates context pollution in code-as-action agents by separating strategic planning (Delegator Agent) from implementation (Coder Agent) using Ephemeral-Persistent State Separation (EPSS).
- The persistent Delegator handles task decomposition and adaptive control, while ephemeral Coders execute atomic sub-tasks through interactive refinement in isolated execution sandboxes.
- EPSS ensures coordination via a dual-layer workspace, confining debugging traces to the Execution Layer and propagating only validated structured results to the persistent Orchestration Layer.

---

[AlertGuardian: Intelligent Alert Life-Cycle Management for Large-scale Cloud Systems](http://arxiv.org/abs/2601.14912)

- AlertGuardian: introduces a novel framework combining lightweight graph models with LLMs to optimize the entire alert life-cycle management process in large-scale cloud systems.
- The framework operates in three phases: Alert Denoise (real-time noise filtering), Alert Summary (contextualized fault narratives via RAG-LLM), and Alert Rule Refinement (offline iterative rule optimization via multi-agent LLM workflow).
- AlertGuardian significantly mitigates alert fatigue by achieving high alert reduction ratios (94.8%) and accelerating fault diagnosis accuracy (90.5%).

---

[HiNS: Hierarchical Negative Sampling for More Comprehensive Memory Retrieval Embedding Model](http://arxiv.org/abs/2601.14857)

- HiNS (Hierarchical Negative Sampling): introduces a principled data construction framework that explicitly models negative sample difficulty tiers and incorporates empirically grounded negative ratios for training memory retrieval embedding models.
- The framework synthesizes persona-grounded conversations using LLMs, followed by topic clustering and semantic query generation to create informative training triplets.
- HiNS employs a three-tier stratification (hard, medium, easy) for negative samples, ensuring the training signal reflects the challenge spectrum and natural prevalence of error types encountered in practice.

---

[REFLECTING IN THE REFLECTION: INTEGRATING A SOCRATIC QUESTIONING FRAMEWORK INTO AUTOMATED AI-BASED QUESTION GENERATION](http://arxiv.org/abs/2601.14798)

- R-in-R (reflection-in-reflection framework): introduces an automated question generation system coordinating two specialized LLM agents, Student-Teacher and Teacher-Educator, within an iterative Socratic dialogue loop for refining reflection questions.
- The Student-Teacher drafts candidate questions, while the Teacher-Educator acts as a pedagogical coach, providing guiding questions based on criteria like clarity, depth, and relevance.
- The protocol, particularly when using dynamic stopping and contextual grounding, produces questions judged substantially superior in relevance and depth compared to a one-shot LLM baseline.

---

[CI4A: Semantic Component Interfaces for Agents Empowering Web Automation](http://arxiv.org/abs/2601.14790)

- CI4A (Component Interface for Agent): introduces a semantic encapsulation protocol that abstracts complex UI interaction logic into unified tool primitives accessible to agents.
- The CI4A protocol defines the UI Component Interface using a semantic triplet (Semantic State View S, Executable Toolset T, Interaction Metadata M) to bridge the gap between high-level intent and low-level web interactions.
- Eous, a hybrid-architecture agent leveraging CI4A, achieves state-of-the-art web automation performance by utilizing a dynamically updating hybrid action space that prioritizes high-level semantic tools.

---

[Optimizing FaaS Platforms for MCP-enabled Agentic Workflows](http://arxiv.org/abs/2601.14735)

- FAME (FaaS-based architecture for orchestrating MCP-enabled agentic workflows): introduces a FaaS-based architecture that decomposes ReAct agentic patterns into modular Planner, Actor, and Evaluator agents deployed as AWS Lambda functions orchestrated by AWS Step Functions.
- The framework addresses the inherent statelessness of FaaS by implementing automated agent memory persistence and injection using DynamoDB to support multi-turn conversations and session continuity.
- FAME incorporates optimizations like MCP invocation caching via S3 and function fusion strategies, achieving up to 13x latency reduction, 88% fewer input tokens, and 66% cost savings.

---

[AdaTIR: Adaptive Tool-Integrated Reasoning via Difficulty-Aware Policy Optimization](http://arxiv.org/abs/2601.14696)

- AdaTIR (Adaptive Tool-Integrated Reasoning): introduces a framework that shifts from static tool invocation to difficulty-aware reasoning internalization by dynamically adjusting tool budgets based on task complexity.
- The approach utilizes a Difficulty-Aware Efficiency Reward to compel LLMs to internalize reasoning for simple tasks while selectively invoking tools for complex tasks.
- To ensure stable training and prioritize correctness, the framework incorporates Clipped Advantage Shaping (CAS), which mathematically reformulates the advantage term to mitigate sign reversal and mode collapse.

---

[GAMING THE JUDGE: UNFAITHFUL CHAIN-OF-THOUGHT CAN UNDERMINE AGENT EVALUATION](http://arxiv.org/abs/2601.14691)

- CMEF (CoT Manipulation Evaluation Framework): introduces a controlled evaluation pipeline demonstrating that LLM judges are highly susceptible to manipulation of agent reasoning traces (CoTs) by systematically rewriting CoTs while holding actions and observations fixed.
- The framework categorizes manipulation strategies into style-based (altering presentation) and content-based (fabricating progress or misrepresenting the task/environment), finding content-based methods consistently more effective at inflating false positive rates.
- Mitigation strategies like manipulation-aware prompting, rubric-based evaluation, and judge-time scaling reduce susceptibility but do not eliminate the vulnerability, revealing a fundamental tradeoff between robustness and recall.

---

[IB-GRPO: Aligning LLM-based Learning Path Recommendation with Educational Objectives via Indicator-Based Group Relative Policy Optimization](http://arxiv.org/abs/2601.14686)

- IB-GRPO (Indicator-Based Group Relative Policy Optimization): introduces a two-stage alignment approach for LLM-based Learning Path Recommendation (LPR), utilizing hybrid expert data synthesis and Indicator-Based Group Relative Policy Optimization to align recommendations with multi-objective pedagogical goals.
- The framework addresses misalignment with pedagogical objectives and multi-objective complexity by incorporating ZPD alignment as a reward and using the $I_{\epsilon+}$-indicator for group-relative advantage estimation.
- Stage I constructs a diverse Pareto landscape via hybrid expert demonstrations (Genetic Algorithm and Teacher RL Agents) for Supervised Fine-Tuning (SFT) warm-up, providing a robust foundation for subsequent policy optimization.

---

[FARE: Fast-Slow Agentic Robotic Exploration](http://arxiv.org/abs/2601.14681)

- FARE (Fast-Slow Agentic Robotic Exploration): introduces a hierarchical autonomous exploration framework integrating a slow-thinking LLM module for global reasoning and a fast-thinking RL module for locally grounded execution.
- The slow-thinking module interprets textual environment descriptions to synthesize an agent-level strategy and generate global waypoints via LLM-based graph reasoning on a pruned global belief graph.
- The fast-thinking RL policy executes exploration by reacting to local sensor observations while adhering to the global guidance through an instruction following reward objective, ensuring coherent, robust, closed-loop behavior.

---

[INFA-GUARD: Mitigating Malicious Propagation via Infection-Aware Safeguarding in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2601.14667)

- INFA-GUARD (Infection-Aware Guard): introduces a novel defense framework that explicitly models and distinguishes between originating attack agents and persuaded infected agents in LLM-based Multi-Agent Systems.
- The framework employs infection-aware detection using a specialized GNN architecture and topological constraints to iteratively localize attack sources and infected ranges by modeling the dynamic infection process.
- Remediation involves replacing attack agents with benign ones and refining infected agents' responses to prevent malicious propagation while preserving system topology and integrity.

---

[Query-Efficient Agentic Graph Extraction Attacks on GraphRAG Systems](http://arxiv.org/abs/2601.14662)

- AGEA (Agentic Graph Extraction Attack): introduces a query-efficient, memory-augmented framework to reconstruct the latent entity-relation graph of a GraphRAG system under a fixed query budget.
- The framework uses a closed feedback loop involving an adaptive Query Generator and a two-stage Graph Extraction pipeline (Discovery + LLM-based Filtering).
- Extraction efficiency is maximized by coordinating exploration and exploitation modes using graph-level novelty signals and persistent Graph and Query Memory modules.

---

[NEUROFILTER: PRIVACY GUARDRAILS FOR CONVERSATIONAL LLM AGENTS](http://arxiv.org/abs/2601.14660)

- NeuroFilter: introduces a privacy guardrail framework that operationalizes contextual integrity by mapping norm violations to simple directions in the LLM's activation space, enabling detection of adversarial intent.
- The framework utilizes a Linear Probe trained on cached activations for single-turn detection and Activation Velocity and Cumulative Drift for monitoring multi-turn adversarial steering.
- The guardrail operates on intermediate model states during the forward pass, providing a low-latency security primitive that significantly reduces computational inference cost compared to LLM-based defenses.

---

[MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks](http://arxiv.org/abs/2601.14652)

- MAS-Orchestra: introduces a training-time framework that formulates multi-agent system (MAS) orchestration as a function-calling RL problem, generating the entire MAS structure holistically.
- The Orchestrator (Meta-Agent) uses a Function-Calling Mechanism to define complex Sub-Agents (callable functions) and their connections, guided by the configurable DoM level.
- Optimization is achieved using Group Relative Policy Optimization (GRPO) based on final answer correctness, simplifying training compared to multi-step sequential approaches.

---

[Spatially Generalizable Mobile Manipulation via Adaptive Experience Selection and Dynamic Imagination](http://arxiv.org/abs/2601.14649)

- AES and RSSM-based MPFP: introduces a mobile manipulation framework that leverages Adaptive Experience Selection (AES) to prioritize critical experience fragments for training a Recurrent State-Space Model (RSSM), which then performs Model-Predictive Forward Planning (MPFP) using dynamic imagination.
- AES improves sample efficiency and mitigates skill forgetting by calculating a composite priority score based on informative experience selection, task criticality (IK failure modes), and prediction error.
- The RSSM-based MPFP reinforces RL-based MM skill learning and enables effective spatial generalization by capturing coupled dynamics between the mobile base and manipulator for foresighted planning.

---

[Forest-Chat: Adapting Vision-Language Agents for Interactive Forest Change Analysis](http://arxiv.org/abs/2601.14637)

- Forest-Chat: introduces an LLM-driven agent for interactive forest change analysis, integrating the FC-Supervised (Multi-level Change Interpretation Model) and FC-Zero-shot (AnyChange Model) perception models, orchestrated by the LLM Task Orchestration component.
- The framework supports multiple Remote Sensing Image Change Interpretation (RSICI) tasks, including change detection, captioning, object counting, and deforestation percentage estimation, using a multi-level change interpretation backbone.
- FC incorporates zero-shot change detection via AnyChange and an interactive point-prompt interface to support fine-grained user guidance, enhancing accessibility and interpretability for forest monitoring.

---

[The missing links: Evaluating contact tracing with incomplete data in large metropolitan areas during an epidemic](http://arxiv.org/abs/2601.14632)

- ABM (Agent-Based Model): introduces a high-resolution simulation using synthetic populations of Seoul and Busan to evaluate Contact Tracing (CT) effectiveness under various information loss scenarios, including Infector-Omission (IO) and Contact-Omission (CO).
- The model incorporates an extended SEIR model and a multilayer contact network (households, workplaces, schools, friendships, local communities) to capture heterogeneous interpersonal interactions and reconstruct the directed transmission network.
- Simulations reveal city-specific thresholds for CT effectiveness (4% IO in Seoul, 10% IO in Busan), demonstrating that missing infector trajectories (IO) have a greater negative impact on containment than failing to notify contacts (CO).

---

[SearchGym: Bootstrapping Real-World Search Agents via Cost-Effective and High-Fidelity Environment Simulation](http://arxiv.org/abs/2601.14615)

- SearchGym: introduces a cost-effective, high-fidelity simulation environment for training robust search agents, built upon a verified Knowledge Graph and an Aligned Document Corpus, and utilizing the SearchGym-RL curriculum learning methodology for policy optimization via Reinforcement Learning (GRPO) using Search and Access primitives.
- The framework resolves the critical dilemma of training search agents by eliminating corrupted reward signals caused by data misalignment in static offline environments or prohibitive costs associated with live web APIs.
- SearchGym-RL demonstrates strong Sim-to-Real generalization, enabling LLM agents (e.g., Qwen2.5-7B) trained in the synthetic environment to surpass web-enhanced baselines on diverse real-world benchmarks with zero commercial web API costs.

---

[Towards Cybersecurity Superintelligence: from AI-guided humans to human-guided AI](http://arxiv.org/abs/2601.14614)

- G-CTR (Generative Cut-the-Rope): introduces a neurosymbolic architecture embedding game-theoretic reasoning into LLM-based agents, achieving Cybersecurity Superintelligence by surpassing human speed and strategic capability.
- The approach traces an evolution from AI-Guided Humans (PentestGPT) to Expert-level AI Agents (CAI), culminating in Game-Theoretic AI Agents that achieve a 100% success rate on the CAIBench-Jeopardy CTFs benchmark.
- G-CTR augments neural inference via symbolic equilibrium computation, utilizing Attack Graph Generation, Nash Equilibrium Computation, and Strategic Digest Injection to reduce behavioral variance and achieve a 2:1 advantage over non-strategic AI.

---

[AN LLM AGENT-BASED FRAMEWORK FOR WHALING COUNTERMEASURES](http://arxiv.org/abs/2601.14606)

- LLM Agent-based Framework for Whaling Countermeasures: introduces a defensive pipeline that inverts attack-oriented architectures to protect high-value university faculty members using LLM agents to construct Personalized Vulnerability Profiles (PVPs), Risk Scenarios, and Personalized Defense Profiles (PDPs).
- The framework operates in an offline analysis phase to generate structured profiles (PVP, RS, PDP) from publicly available and internal institutional information, and an online analysis phase where an LLM agent uses the PDP to assess incoming email risk.
- The Online Analysis Agent provides personalized, context-aware risk judgments, numerical scores, and actionable defensive guidance by cross-referencing email content against the target's specific PDP.

---

[3D Space as a Scratchpad for Editable Text-to-Image Generation](http://arxiv.org/abs/2601.14602)

- 3D Spatial Scratchpad (3DSS): introduces a 3D reasoning substrate that bridges linguistic intent and image synthesis by parsing a text prompt into editable 3D meshes and employing agentic scene planning for placement and orientation.
- The framework utilizes a pipeline of specialized LLM agents for scene planning, orientation estimation, and camera selection to arrange subjects within the virtual 3D scene, generating a spatially coherent configuration.
- The resulting 3D arrangement is rendered back into the image domain using identity-preserving cues and depth control, enabling robust 3D-aware editing capabilities via manual or text-based instructions.

---

[Holmes: An Evidence-Grounded LLM Agent for Auditable DDoS Investigation in Cloud Networks](http://arxiv.org/abs/2601.14601)

- Holmes (DDoS Detective): introduces an LLM Agent (Virtual SRE investigator) constrained by a Hierarchical Detection Workflow (funnel-like pipeline) and Evidence-Grounded Reasoning paradigm (enforces factual grounding) using a Semantic Evidence Abstraction (converts binary packets) into an Evidence Pack (structured, quote-able evidence).
- The system employs a Cyclic Processing Pipeline (four stages) starting with Continuous Telemetry and Lightweight Triage to activate expensive LLM investigation only upon detected anomalies, addressing the operational efficiency paradox.
- The framework mitigates unfaithful reasoning by enforcing a strict "Quote Rule" and JSON output constraints, ensuring every verdict is a verifiable conclusion backed by an evidence chain.

---

[Agent Identity URI Scheme: Topology-Independent Naming and Capability-Based Discovery for Multi-Agent Systems](http://arxiv.org/abs/2601.14567)

- agent:// URI Scheme: introduces a foundation for decentralized agent identity and capability-based discovery, utilizing a Trust Root (organizational authority), a Capability Path (semantic discovery), and a stable Agent Identifier (unique reference).
- The scheme decouples agent identity from network topology, enabling decentralized resolution via DHT key derivation based on the Trust Root and Capability Path, allowing agents to be found by function.
- Cryptographic attestation using PASETO tokens binds capability claims to agent identity, ensuring verifiable claims and maintaining identity stability across agent migrations.

---

[SCSimulator: An Exploratory Visual Analytics Framework for Partner Selection in Supply Chains through LLM-driven Multi-Agent Simulation](http://arxiv.org/abs/2601.14566)

- SCSimulator: introduces an exploratory visual analytics framework that integrates LLM-driven Multi-Agent Simulation with human-in-the-loop collaboration for supply chain partner selection.
- The system simulates SC evolution using LLM agents following a four-stage request-response cycle, generating transparent explanations via Chain-of-Thought reasoning and quantitative attribution using SHAP.
- Interactive visualizations, including Global, Focus, and Adjustment Views, enable users to track dynamic SC relationships and iteratively refine agent behaviors aligned with strategic goals.

---

[CONSTRUCTING MULTI-LABEL HIERARCHICAL CLASSIFICATION MODELS FOR MITRE ATT&CK TEXT TAGGING](http://arxiv.org/abs/2601.14556)

- MHCS: introduces a multi-label hierarchical classification system for MITRE ATT&CK text tagging, utilizing TF-IDF Vectorization/Hashing, a Multi-label Tactic Classifier, and Tactic-specific Multi-label Technique Classifiers, built upon Multiclass SGD Models to generate Predicted Tactic, Technique Pairs.
- The system employs a two-level hierarchical structure where the first level predicts the top *n* tactics, and the second level uses tactic-specific models conditioned on these predictions to identify the top *m* techniques.
- The approach achieves high accuracy (94% tactic level, 82% technique level) using classical machine learning methods, significantly outperforming LLMs like GPT-4o in multiclass tactic prediction.

---

[Can Rising Consumption Deepen Inequality?](http://arxiv.org/abs/2601.15537)

- Extended Social Architecture of Capitalism (SA) Model: introduces variations to the original SA agent-based model by allowing heterogeneous transaction frequencies ($\Omega_E, \Omega_M$) and imposing limits on transaction amplitudes ($\Phi_E, \Phi_M$), alongside an adaptive wage mechanism.
- The study finds that increasing expenditure frequency ($\Omega_E$) or fraction ($\Phi_E$) raises GDP and inequality (Gini index, HHI) while decreasing the labor share, reflecting increased monopoly power.
- Introducing adaptive wages, which endogenously determine the wealth-to-salary ratio ($R$), reveals a nonmonotonic relationship between $R$ and inequality, where low $R$ initially reduces inequality by lowering unemployment.

---

[TransportAgents: a multi-agents LLM framework for traffic accident severity prediction](http://arxiv.org/abs/2601.15519)

- TransportAgent: introduces a hybrid multi-agent LLM framework for traffic accident severity prediction, featuring the Data Preprocessing Team (Filters/classifies attributes), Severity Assessment Team (Produces category-specific scores), and Integration Manager Module (Consolidates severity scores).
- The architecture decomposes the prediction task into specialized LLM agents that perform category-specific reasoning over structured and narrative inputs, enhancing interpretability and robustness.
- The framework utilizes a supervised Multilayer Perceptron (MLP) within the Integration Manager Module to optimally fuse the heterogeneous intermediate severity scores into a final, calibrated prediction.

---

[MIRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation](http://arxiv.org/abs/2601.15487)

- MIRAGE (Multiagent framework for RAG systems Evaluation): introduces a comprehensive multi-agent framework that leverages a collaborative swarm of specialized agents to generate verified, domain-specific, multimodal, and multi-hop Question-Answer datasets.
- The framework orchestrates agents through five phases: multimodal data ingestion, expert persona identification, semantic multihop context building, agentic QA generation and verification, and final refinement and deduplication.
- MIRAGE utilizes a recursive context optimization loop for evidence aggregation and an adversarial verifier agent to guarantee factual grounding, producing datasets with high reasoning complexity and factual faithfulness.

---

[Exploring Implicit Perspectives on Autism in Large Language Models Through Multi-Agent Simulations](http://arxiv.org/abs/2601.15437)

- LLM-based MASs (Multi-Agent Systems): introduces a study using the Generative Agents framework, powered by GPT-4o-mini, to simulate mixed-neurotype social scenarios involving autistic and non-autistic agents to investigate implicit LLM biases regarding autism.
- The framework utilizes components like Memory Stream, Reflection, and Planning to enable agents to engage in group-task conversations and respond to structured interview questions, revealing underlying model perspectives.
- Analysis of 120 simulations showed that the model frames autistic agents as socially dependent and requiring accommodation, while non-autistic agents are portrayed as responsible for providing support, aligning with a deficit-based view of autism.

---

[Q-Probe: Scaling Image Quality Assessment to High Resolution via Context-Aware Agentic Probing](http://arxiv.org/abs/2601.15356)

- Q-Probe: introduces the first agentic IQA framework designed for high-resolution scenarios, leveraging a three-stage training curriculum and an Agentic Reasoning Loop (Observe-Reason-Act cycle) to mimic human coarse-to-fine visual perception.
- The framework utilizes a Context-Aware Cropping Strategy and a Decoupled Reward Mechanism to distinguish between technical degradations and natural artistic effects like depth-of-field bokeh, eliminating the "cropping-implies-degradation" bias.
- Q-Probe is trained using the novel Vista-Bench benchmark, tailored for fine-grained local degradation analysis in high-resolution images, achieving state-of-the-art performance across various IQA datasets.

---

[Statistical Reinforcement Learning in the Real World: A Survey of Challenges and Future Directions](http://arxiv.org/abs/2601.15353)

- SDCI Framework (Sequential Deployment and Continual Improvement): introduces a three-component process for practical RL application, including online learning and optimization, offline analysis and statistical inference, and repeated deployment cycles for continual improvement.
- Within-deployment learning focuses on maximizing cumulative reward and managing the exploration-exploitation tradeoff, while between-deployment analysis uses collected data for policy learning and statistical inference.
- The framework addresses challenges related to data scarcity and significant environmental changes by integrating statistical methods, causal knowledge, and LLMs for efficient learning and system redesign.

---

[VibeTensor: System Software for Deep Learning, Fully Generated by AI Agents](http://arxiv.org/abs/2601.16238)

- VIBETENSOR: introduces a deep learning system software stack, fully generated by LLM-powered coding agents under high-level human guidance, spanning Python/Node.js frontends down to CUDA memory management.
- The architecture includes a shared C++ core featuring a schema-lite Dispatcher, a reverse-mode Autograd Engine, a stream-ordered Caching Allocator, and an experimental Fabric subsystem for multi-GPU execution.
- The system validation relies heavily on builds, tests, and differential checks against PyTorch baselines, demonstrating the feasibility of generating coherent, system-scale software constrained by executable specifications.

---

#### 20th January 2026

[XR: Cross-Modal Agents for Composed Image Retrieval](http://arxiv.org/abs/2601.14245)

- XR (Cross-Modal Agents for Composed Image Retrieval): introduces a training-free multi-agent framework that reframes Composed Image Retrieval as a progressively coordinated reasoning process, including imagination, coarse filtering, fine filtering, and cross-modal re-ranking.
- The framework orchestrates specialized agents, including imagination agents for synthesizing target representations and similarity/question agents for coarse-to-fine filtering and factual verification.
- By integrating similarity-based scoring for broad coverage and question-based verification for accuracy, the system achieves robust retrieval that aligns with fine-grained user intent.

---

[APEX-Agents](http://arxiv.org/abs/2601.14242)

- APEX-Agents: introduces a benchmark for assessing AI agents' ability to execute complex, long-horizon professional services tasks using the Archipelago infrastructure, which includes the Environment, Agents runner, Grading system, Judge LLM, ReAct toolbelt, World, Task, and Output components.
- The benchmark contains 480 tasks split across 33 data-rich worlds simulating investment banking, management consulting, and corporate law scenarios, requiring agents to use multiple applications and files.
- Evaluation relies on the Judge LLM grading agent outputs against expert-created rubrics and gold outputs, measuring performance using Pass@1 (proportion of tasks meeting all criteria).

---

[Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression](http://arxiv.org/abs/2601.14238)

- FireCastRL (proactive artificial intelligence framework): introduces an end-to-end system combining a deep spatiotemporal CNN-LSTM model for wildfire ignition forecasting with a Proximal Policy Optimization (PPO) agent for helitack suppression strategy optimization within a physics-informed 3D simulation environment.
- The framework utilizes a large-scale, publicly released spatiotemporal dataset of 9.5 million samples derived from IRWIN and GRIDMET data to train the forecasting model and the RL agent.
- High-risk predictions trigger the RL simulation, generating a comprehensive fire threat assessment report for emergency responders detailing predicted ignition coordinates, burn trajectory, and suppression sequence.

---

[Opportunities in AI/ML for the Rubin LSST Dark Energy Science Collaboration](http://arxiv.org/abs/2601.14235)

- DESC AI/ML Strategy: introduces a comprehensive plan for integrating AI/ML into the LSST Dark Energy Science Collaboration (DESC) pipelines, focusing on robust infrastructure, methodological rigor, and governance.
- The strategy prioritizes foundational research in Uncertainty Quantification (UQ), Simulation-Based Inference (SBI), and physics-informed methods, alongside the adoption of emerging technologies like FMs and LLM-driven agentic AI systems.
- Successful deployment requires a durable AI software stack, secure access to GPU/HPC resources, and rigorous validation benchmarks to ensure scientific accountability and transparency in precision cosmology.

---

[KAGE-Bench: Fast Known-Axis Visual Generalization Evaluation for Reinforcement Learning](http://arxiv.org/abs/2601.14232)

- KAGE-Bench (Known-Axis Generalization Evaluation Benchmark): introduces a visual generalization benchmark built on KAGE-Env, which uses independently controllable Visual Axes and specific Train-Evaluation Configuration Pairs to isolate visual shifts for precise diagnosis using a PPO-CNN Baseline and JAX Implementation.
- KAGE-Env is a JAX-native 2D platformer that factorizes the observation process into 93 explicitly controllable parameters while keeping latent dynamics and rewards fixed, providing a clean abstraction for visual generalization studies.
- The benchmark enables fast, reproducible evaluation, achieving up to 33M environment steps per second, and empirically diagnoses that generalization failures are strongly axis-dependent, particularly severe under background and photometric shifts.

---

[MASCOT: Towards Multi-Agent Socio-Collaborative Companion Systems](http://arxiv.org/abs/2601.14230)

- MASCOT (Multi-Agent Socio-Collaborative Companion Systems): introduces a generalizable framework for multi-perspective socio-collaborative companions using a bi-level optimization strategy, including Persona-Aware Behavioral Alignment (Individual agent fine-tuning) and Collaborative Dialogue Optimization (Group discourse coordination).
- The Persona-Aware Behavioral Alignment phase uses an RLAIF-driven pipeline with a learned Persona Reward Model to ensure strict persona fidelity and prevent identity loss in individual Speaker Agents.
- The Collaborative Dialogue Optimization phase employs a Director meta-agent guided by group-level rewards and GRPO to coordinate Speaker Agents, ensuring diverse, productive, and synergistic discourse.

---

[Attention-Based Offline Reinforcement Learning and Clustering for Interpretable Sepsis Treatment](http://arxiv.org/abs/2601.14228)

- ISTP (Interpretable Sepsis Treatment Pipeline): introduces a decision support framework for personalized sepsis treatment, integrating HDBSCAN (Patient stratification), VAE (Discrete transition modeling), Diffusion Model (Continuous state augmentation), AWR (Offline RL policy), TabNet (Tabular prediction), XGBoost (Gradient boosting prediction), and an LLM (Rationale generation).
- The system first stratifies new patients into risk groups using clustering and augments sparse clinical data using generative models (VAE and Diffusion Model) to improve RL training robustness.
- The core treatment policy is derived from an ensemble of an AWR agent with a feature attention encoder, TabNet, and XGBoost, ensuring conservative, safety-aware recommendations, which are then justified by the LLM using retrieved clinical context.

---

[Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents](http://arxiv.org/abs/2601.14224)

- Rerank Before You Reason (RBYR): analyzes the efficiency-effectiveness tradeoffs of integrating listwise reranking into deep search agent pipelines using the Effective Token Cost (ETC) metric.
- The system utilizes a Deep Search Agent (oss-20b/oss-120b) for iterative reasoning and a Retriever (qwen3-embedding-8b) for initial document fetching, followed by a Reranker to refine candidate lists before final reasoning.
- The analysis demonstrates that moderate reranking consistently improves retrieval quality and end-to-end accuracy at a substantially lower cost than increasing the search agent's reasoning budget.

---

[HALT: Hallucination Assessment via Latent Testing](http://arxiv.org/abs/2601.14210)

- HALT (Hallucination Assessment via Latent Testing): introduces a lightweight hallucination detector that reads risk directly from intermediate hidden states of question tokens using a small auxiliary network ($g_{\theta}$), enabling near-instantaneous risk estimation.
- The system employs an LLM Router that evaluates the risk in parallel with generation, allowing confident queries to proceed with zero latency while routing uncertain queries to a Stronger LLM (Fallback Pipeline) or verification pipeline.
- The detector utilizes question-aligned Intermediate Hidden States, which retain epistemic signals often attenuated in the final decoding stage, requiring computation less than 1% of single token generation cost.

---

[A Minimax Perspective on Almost-Stable Matchings](http://arxiv.org/abs/2601.14195)

- MINIMAX-ALMOSTSTABLE-SRI: introduces a fairness-oriented approach to approximate stability by minimizing the maximum number of blocking pairs any agent is contained in, using Irving's algorithm, Gale-Shapley algorithm, MaxCard, Rematching Procedure, and ILP Formulations.
- The paper characterizes the computational complexity of Minimax Almost-Stability, showing strong intractability (NP-complete/para-NP-hard) even for modest guarantees, but provides polynomial-time algorithms for instances with preference lists of length at most two.
- MINIMAX-ALMOSTSTABLE-MAX-SMI focuses on maximizing matching size while minimizing the maximum instability burden on any single agent, revealing fundamental trade-offs in matching market design.

---

[Toward Efficient Agents: A Survey of Memory, Tool learning, and Planning](http://arxiv.org/abs/2601.14192)

- Efficient Agents Framework: introduces a survey evaluating efficiency across three core components of LLM-based agents: Memory (Context compression and retrieval), Tool Learning (Tool selection and invocation), and Planning (Resource-constrained deliberation).
- The framework defines an efficient agent as one maximizing task success while minimizing resource consumption, including token usage, inference latency, and computational cost across all modules.
- Efficiency strategies are categorized into techniques for bounding context, minimizing tool invocation via RL rewards, and employing controlled search mechanisms for planning.

---

[Paper2Rebuttal: A Multi-Agent Framework for Transparent Author Response Assistance](http://arxiv.org/abs/2601.14171)

- REBUTTALAGENT: introduces a multi-agent framework that reframes rebuttal generation as an evidence-centric planning task, ensuring arguments are anchored in internal or external evidence via a structured "verify-then-write" workflow.
- The system decomposes complex reviewer feedback into atomic concerns, constructs hybrid contexts using compressed manuscript summaries, and integrates an autonomous external search module to resolve concerns requiring outside literature.
- By generating an inspectable, evidence-linked response plan checked for consistency and commitment safety, the framework provides transparent and controllable assistance before the Response Drafter produces the final rebuttal text.

---

[CREATE: Cross-Layer Resilience Characterization and Optimization for Efficient yet Reliable Embodied AI Systems](http://arxiv.org/abs/2601.14140)

- CREATE (Cross-Layer Resilience Characterization and Optimization): introduces a general design principle leveraging heterogeneous resilience across circuit, model, and application layers for synergistic energy-reliability co-optimization, including Anomaly Detection and Clearance, Weight-Rotation-Enhanced Planning, and Autonomy-Adaptive Voltage Scaling.
- The framework achieves cross-layer optimization by using circuit-level AD to suppress large timing errors, model-level WR to enhance LLM planner robustness against smaller errors, and application-level VS to dynamically adjust the RL controller's voltage based on task criticality.
- CREATE achieves an average 40.6% computational energy savings over nominal-voltage baselines and 35.0% over prior-art techniques without compromising task quality, translating to significant chip-level energy savings and battery life improvement.

---

[Toward architecting self-coding information systems](http://arxiv.org/abs/2601.14132)

- SCIS (Self-coding information systems): introduces a novel research topic where information systems dynamically adapt their structure or behavior by autonomously generating, testing, and deploying source code at runtime.
- These systems rely on LLM-based agents to generate or regenerate the source code of at least one subsystem, enabling high modifiability and autonomy by performing modifications at runtime.
- Key research directions include developing reliability strategies, designing reference architectures, and evolving SCIS toward self-architecting capabilities.

---

[AttackMate: Realistic Emulation and Automation of Cyber Attack Scenarios Across the Kill Chain](http://arxiv.org/abs/2601.14108)

- AttackMate: introduces an open-source attack scripting language and execution engine, with Playbook Parser (loads/processes playbooks), Sequential Dispatcher (iterates/calls Executors), and various Executors (Flow Control, Data Control, Attack Commands, Attack Framework Connectors).
- The system focuses on realistic execution by supporting Interactive mode and Sessions to mimic human attacker behavior, avoiding suspicious artifacts left by agent-based adversary emulation tools.
- AttackMate provides modular architecture, full kill chain coverage, and integration with external tools like Metasploit and sliver, addressing shortcomings of existing emulation frameworks.

---

[Remapping and navigation of an embedding space via error minimization: a fundamental organizational principle of cognition in natural and artificial systems](http://arxiv.org/abs/2601.14096)

- RNES (Remapping and Navigation of Embedding Spaces): introduces a substrate-independent invariant of cognition characterized by the interplay of remapping and navigation within embedding spaces via iterative error minimization.
- This dual principle provides a unifying framework for understanding intelligence across diverse biological systems (e.g., morphogenesis, cellular collectives) and modern AI architectures (e.g., transformers, diffusion models, NCAs).
- Remapping involves dynamically constructing and updating internal latent representations (embeddings) that translate complex realities into navigable dimensions, enabling goal-directed navigation.

---

[LLMOrbit: A Circular Taxonomy of Large Language Models—From Scaling Walls to Agentic AI Systems](http://arxiv.org/abs/2601.14053)

- LLMOrbit (Circular Taxonomy): introduces a comprehensive circular taxonomy spanning 2019-2025, documenting architectural innovations, training methodologies, and efficiency patterns defining modern LLMs and agentic systems.
- The survey identifies the "scaling wall" defined by data scarcity, exponential cost growth, and unsustainable energy consumption, fundamentally limiting brute-force scaling approaches.
- The paper evaluates eight alternative paradigms, including MLA, fine-grained MoE, and pure RL reasoning (DeepSeek-R1), that enable GPT-4-level performance at 100x lower cost.

---

[Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics](http://arxiv.org/abs/2601.14027)

- Numina-Lean-Agent (NLA): introduces an agentic formal theorem proving framework built on Claude Code (General coding agent) and Numina-Lean-MCP (Specialized tool integration), enabling autonomous interaction with Lean and auxiliary reasoning tools.
- The system leverages a general coding agent paradigm, allowing flexible replacement of the underlying base model without training and supporting plug-and-play extension of specialized tools via the Model Context Protocol (MCP).
- NLA achieved state-of-the-art performance by solving all 12 problems in Putnam 2025, demonstrating its generality by formalizing the Brascamp-Lieb theorem through human-AI cooperation.

---

[FANTASYVLN: UNIFIED MULTIMODAL CHAIN-OF-THOUGHT REASONING FOR VISION-LANGUAGE NAVIGATION](http://arxiv.org/abs/2601.13976)

- FANTASYVLN (Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation): introduces a unified implicit reasoning framework that integrates Non-CoT, T-CoT, CompV-CoT, and MM-CoT modes using a gating mechanism and a pretrained VAR model.
- The framework avoids explicit token overhead during inference by compressing imagined visual observations into a compact latent space via the VAR model during training, enabling reasoning-aware yet real-time navigation.
- A cross-mode alignment constraint is used during joint training to align action predictions across all reasoning modes, embedding diverse CoT patterns into a shared latent policy for efficient generalization.

---

[Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval](http://arxiv.org/abs/2601.13969)

- ARK (ADAPTIVE RETRIEVER OF KNOWLEDGE): introduces a training-free agentic KG retrieval framework that gives an LLM control over breadth-depth tradeoff using Global Search and Neighborhood Exploration tools.
- The framework dynamically alternates between broad lexical search for initial anchoring and targeted multi-hop traversal via one-hop neighborhood expansion, adapting the strategy based on query requirements.
- ARK achieves strong and stable retrieval performance across heterogeneous KGs by balancing global discovery and deep relational expansion, and its policy can be distilled into smaller models for efficiency.

---

[RL-BioAug: Label-Efficient Reinforcement Learning for Self-Supervised EEG Representation Learning](http://arxiv.org/abs/2601.13964)

- RL-BioAug: introduces a label-efficient reinforcement learning framework that optimizes data augmentation policies for self-supervised EEG representation learning.
- The framework uses a cyclic cooperative optimization loop where a Transformer-based agent selects optimal strong augmentations based on a Soft-KNN consistency score reward.
- By leveraging only 10% of labeled data for policy guidance, the method enables the encoder to learn robust representations in a strictly self-supervised manner.

---

[RepoGenesis: Benchmarking End-to-End Microservice Generation from Readme to Repository](http://arxiv.org/abs/2601.13943)

- RepoGenesis: introduces the first multilingual benchmark for repository-level web microservice generation from natural language requirements, featuring a dataset of 106 repositories and a rigorous Review-Rebuttal QA Mechanism.
- The benchmark evaluates LLM-based Coding Agents and Commercial IDEs using a multi-dimensional Evaluation Pipeline within a Sandboxed Execution Environment, measuring Pass@1, API Coverage (AC), and Deployment Success Rate (DSR).
- Evaluation results expose fundamental challenges in microservice generation, particularly concerning architectural coherence, dependency management, and cross-file consistency, with the best system achieving only 23.67% Pass@1.

---

[Deep Reinforcement Learning-Based Dynamic Resource Allocation in Cell-Free Massive MIMO](http://arxiv.org/abs/2601.13934)

- PPO-based algorithm: introduces a novel DRL framework for dynamic resource allocation in CFmMIMO systems by optimizing AP activation ratio ($\zeta$), antenna allocation coefficient ($\kappa$), and power allocation coefficient ($\nu$).
- The framework transforms the high-dimensional resource allocation problem (antenna selection and power control) into a compact, fixed-size action space defined by the three coefficients, ensuring scalability and low complexity.
- The DRL agent learns to map large-scale fading coefficients to these resource allocation coefficients to maximize Energy Efficiency (EE) while satisfying Quality of Service (QoS) constraints.

---

[VulnResolver: A Hybrid Agent Framework for LLM-Based Automated Vulnerability Issue Resolution](http://arxiv.org/abs/2601.13933)

- VulnResolver (Hybrid Agent Framework for LLM-Based Automated Vulnerability Issue Resolution): introduces a hybrid LLM-based system combining the adaptability of the Context Pre-Collection Agent and Safety Property Analysis Agent with a deterministic Workflow for end-to-end vulnerability issue resolution.
- The framework utilizes specialized agents supported by five Toolkits (Code Search, Code Symbol Analysis, PoC Execution, Project Editing, Python Code Execution) to gather context and generate/validate safety properties.
- The agents produce structured Context Analysis and Property Analysis Reports, which are combined with the original Issue Report to form an Enhanced Issue Report driving subsequent localization and patch generation stages.

---

[HyperWalker: Dynamic Hypergraph-Based Deep Diagnosis for Multi-Hop Clinical Modeling across EHR and X-Ray in Medical VLMs](http://arxiv.org/abs/2601.13919)

- HyperWalker: introduces a Deep Diagnosis framework that reformulates clinical reasoning via dynamic hypergraphs and TTT for structured multi-hop diagnostic reasoning across multimodal clinical data.
- The framework utilizes iBrochure, a dynamic heterogeneous hypergraph, to model structural heterogeneity and high-order associations among EHR, X-ray, and clinical knowledge.
- Walker, an RL agent, navigates this manifold using a linger mechanism (orthogonal multi-hop retrieval) and TTT to identify optimal, evidence-based diagnostic paths.

---

[AGENTEHR: Advancing Autonomous Clinical Decision-Making via Retrospective Summarization](http://arxiv.org/abs/2601.13918)

- RETROSUM (Retrospective Summarization): introduces a novel framework for autonomous clinical decision-making that unifies a retrospective summarization mechanism with an evolving experience strategy to mitigate information loss and reasoning fragmentation in long-context EHR tasks.
- The framework utilizes an Actor guided by an Augmented Context, which includes the full interaction history and a Retrospective Summary generated by the Summarizer module by re-evaluating past events in light of recent findings.
- The Evolving Strategy uses a Reflection Module to crystallize successful procedural heuristics and information salience guidelines into an external Experience Memory Bank for robust inference guidance.

---

[Know Your Contract: Extending eIDAS Trust into Public Blockchains](http://arxiv.org/abs/2601.13903)

- eIDAS Trust Extension Architecture: introduces an architecture extending the EU's eIDAS trust framework into public blockchains by cryptographically binding smart contracts to Qualified Electronic Seals (QSeals) issued by QTSPs.
- This mechanism establishes a verifiable chain of trust from the European Commission's List of Trusted Lists (LOTL) to individual on-chain addresses, enabling machine-verifiable proofs for automated regulatory validation.
- The architecture supports two models—off-chain validation for agent-to-agent payments and fully on-chain validation for regulatory-compliant institutional DeFi operations—leveraging EVM-compatible cryptographic suites.

---

[LifeAgentBench: A Multi-dimensional Benchmark and Agent for Personal Health Assistants in Digital Health](http://arxiv.org/abs/2601.13880)

- LifeAgent: introduces a training-free health-assistant agent that executes complex queries as a sequence of tool-based actions, integrating multi-step evidence retrieval with deterministic aggregation.
- The agent operates via an iterative thought-action-observation loop, utilizing in-context decomposition to transform user queries into a tool-executable retrieval agenda.
- LifeAgent addresses key bottlenecks in LLMs regarding long-horizon, cross-dimensional health reasoning by relying on structured tools for data retrieval and aggregation.

---

[Understanding Human-Multi-Agent Team Formation for Creative Work](http://arxiv.org/abs/2601.13865)

- CRAFTEAM (Technology Probe for Human-Multi-Agent Team Formation): introduces an exploratory study to understand how to form Human-Multi-Agent Teams (HMATs) for creative work, utilizing an iterative cycle of forming, ideating, and reflecting with LLM-based agents.
- The system allows users to configure five core dimensions of HMAT formation: team size, structure, role allocation, member composition, and shared mental models (SMMs).
- Findings reveal that participants initially attempted autonomous team operations but ultimately converged on human-orchestrated formations where the user directly manages multiple agents to break unproductive loops.

---

[HardSecBench: Benchmarking the Security Awareness of LLMs for Hardware Code Generation](http://arxiv.org/abs/2601.13864)

- HardSecBench: introduces a benchmark for evaluating LLM security awareness in hardware code generation using a multi-agent pipeline, including the Seed Generator (Converts CWE definition), Architect Agent (Produces structured specification), Expert Agent (Synthesizes golden implementation), Tester Agent (Derives atomic test harnesses), and Arbiter Agent (Localizes mismatch, issues feedback).
- The pipeline decouples golden implementation synthesis from test harness generation to ensure objectivity and grounds evaluation in execution evidence via simulation and iterative refinement.
- The benchmark covers 924 tasks spanning Verilog RTL and firmware-level C, revealing that LLMs often satisfy functional requirements while failing security checks, highlighting a critical gap in security awareness.

---

[Small Models, Big Impact: Tool-Augmented AI Agents for Wireless Network Planning](http://arxiv.org/abs/2601.13843)

- MAINTAINED (autonomous artificial intelligence agent for wireless network deployment): introduces a paradigm shift for wireless network planning by externalizing domain knowledge into verifiable computational tools, orchestrated by an LLM using the ReAct framework and function calling.
- The framework utilizes a compact 4B-parameter LLM that outperforms state-of-the-art LLMs (like ChatGPT-40) by up to 100-fold in verified performance metrics while requiring significantly less computational resources.
- This computation-over-memorization approach eliminates technical hallucination, ensures computational accuracy through verified algorithms, and enables edge-deployable AI planning for wireless communications.

---

[From RTL to Prompt Coding: Empowering the Next Generation of Chip Designers through LLMs](http://arxiv.org/abs/2601.13815)

- LLM-based learning platform/workflow: introduces an LLM-based chat agent integrated into a browser-based workflow built upon the Tiny Tapeout ecosystem to guide beginners from initial design idea through RTL code generation to a tapeout-ready chip.
- The workflow involves three stages: idea development and RTL coding via the Chat Agent, simulation and visual testing via the VGA Playground UI, and chip implementation via GitHub Actions for synthesis and backend flow.
- The system aims to lower the entry barrier to chip design education for non-experts by providing LLM assistance, domain-specific knowledge, and immediate visual feedback without requiring specialized infrastructure.

---

[HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction](http://arxiv.org/abs/2601.13801)

- HoverAI: introduces an embodied aerial agent that integrates drone mobility, infrastructure-independent visual projection via a MEMS laser projector and semi-rigid screen, and real-time conversational AI into a unified platform.
- The system employs a multimodal pipeline combining VAD, ASR (Whisper), an LLM-based intent classifier (gemma:7b-instruct), RAG for dialogue, face analysis (InsightFace) for personalization, and TTS (XTTS v2) for adaptive avatar expression.
- This closed-loop architecture enables the agent to perceive users through vision and voice, respond via a personalized, lip-synced avatar, and execute structured commands with high accuracy.

---

[On Autopilot? An Empirical Study of Human-AI Teaming and Review Practices in Open Source](http://arxiv.org/abs/2601.13754)

- Empirical Study: investigates AI-assisted pull requests (PRs) in Open Source Software (OSS) using the AIDev Dataset (Expanded PR data), Human Baseline (Non-AI PRs), Contributor Ownership (Code ownership classification), Guideline Artifacts (Project governance files), and Review Metrics (Feedback and closure time), finding that AI-co-authored PRs merge significantly faster with minimal feedback, especially for non-owner contributors.
- The study reveals that 67.5% of AI-assisted PRs originate from contributors without prior code ownership, yet 86.9% of repositories lack guidelines for LLM coding agent usage.
- In contrast to human-only PRs where non-owners receive the most feedback, AI-co-authored PRs from non-owners receive the least feedback, with approximately 80% merged without explicit review.

---

[OP-Bench: Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents](http://arxiv.org/abs/2601.13722)

- Self-ReCheck: introduces a lightweight, model-agnostic memory filtering mechanism that mitigates over-personalization by selectively verifying and incorporating memory content using an LLM-based reasoning function.
- The approach is evaluated using OP-Bench, the first benchmark quantifying over-personalization across Irrelevance, Sycophancy, and Repetition categories in memory-augmented dialogue systems.
- Evaluation reveals that memory-augmented agents suffer from "memory hijacking," where models over-attend to aggressively retrieved, often irrelevant, memory tokens, causing performance degradation.

---

[Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](http://arxiv.org/abs/2601.13719)

- HAVEN (Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search): introduces a unified framework for long-video understanding that enables coherent and comprehensive reasoning by integrating audiovisual entity cohesion and hierarchical video indexing with an agentic search mechanism.
- The framework constructs a four-level hierarchical database (global summary, scene summaries, segment information, canonical entities) to support multi-granularity retrieval and preserve semantic consistency.
- An LLM-based agent uses a Think-Act-Observe loop and a suite of multi-granularity tools (including Global Scene Browse and Entity Search) to dynamically navigate the hierarchy, achieving state-of-the-art performance on long video benchmarks.

---

[SWE-TESTER: TRAINING OPEN-SOURCE LLMS FOR ISSUE REPRODUCTION IN REAL-WORLD REPOSITORIES](http://arxiv.org/abs/2601.13713)

- SWE-Tester: introduces a novel two-step pipeline for training open-source LLMs to generate issue reproduction tests from natural language issue descriptions and a pre-PR codebase.
- The pipeline first performs Code Localization to retrieve defective source code and relevant test files, followed by Code Editing, where the LLM generates a test patch using a Search/Replace format to augment the test file.
- The approach leverages a curated high-quality training dataset of 41K instances from 2.6K GitHub repositories, achieving absolute improvements of up to 10% in success rate on SWT-Bench Verified.

---

[IGAA: Intent-Driven General Agentic AI for Edge Services Scheduling using Generative Meta Learning](http://arxiv.org/abs/2601.13702)

- IGAA (Intent-Driven General Agentic AI): introduces a novel framework for autonomous edge service scheduling leveraging a generative meta-learning paradigm, incorporating an LLM (cognitive core), N-S-I Matrix (intent-to-resource mapping), Service Scheduling Model (policy generation), Evaluation and Correction Model (scenario auditing/correction), RCETL (new resource adaptation), APOTL (new service adaptation), and GIR Mechanism (catastrophic forgetting defense).
- The framework enables Agentic AI to continuously learn new scheduling tasks generated from user intents while preserving and reusing knowledge learned from previous tasks.
- A key component is the easy-to-hard generalization learning scheme, utilizing RCETL and APOTL to support efficient adaptation under an expanding task space, achieving strong generalization and scalability.

---

[Generative Intent Prediction Agentic AI empowered Edge Service Function Chain Orchestration](http://arxiv.org/abs/2601.13694)

- GIPA (Generative Intent Prediction Agent): introduces an intent-driven proactive edge network management framework for SFC orchestration, utilizing a Generative Diffusion Model (GDM) to predict users' implicit intents from multidimensional context.
- The framework constructs a multidimensional intent space (function, QoS, resource) to map unstructured natural language into quantifiable demands, enabling a shift from passive execution to proactive prediction and orchestration.
- The predicted implicit intent vectors are embedded as global prompts into the SFC orchestration model (PSOM) to guide ahead-of-time VNF deployment and maximize service continuity and QoS.

---

[Distributed Coverage Control on Poriferous Surface via Poly-Annulus Conformal Mapping](http://arxiv.org/abs/2601.13688)

- DCCF (Distributed Diffeomorphic Coverage Control Framework): introduces a distributed coverage control framework for Multi-Agent System (Performs coverage task) on poriferous surfaces, leveraging Poly-Annulus Conformal Mapping (Transforms S to $\Xi$) and a Pull-back Riemannian Metric (Encodes safety constraints) to synthesize a Distributed Gradient-based Control Law (Drives agents to optimal centroids).
- The framework establishes topological conjugacy between the robot workspace and an n-holed disk via distributed mapping components, including Domain Decomposition (Segments S into subdomains) and Local/Global Partial Welding (Stitches local maps) and Geometric Rectification (Enforces circularity/minimizes distortion).
- The Multi-Hole Partition Algorithm (Ensures topological safety/workload balance), supported by a Buffer-based Sequence Mechanism (Guarantees collision avoidance), ensures exponential Input-to-State Stability of partition dynamics and asymptotic convergence to optimal Riemannian centroids.

---

[Toward Agentic AI: Task-Oriented Communication for Hierarchical Planning of Long-Horizon Tasks](http://arxiv.org/abs/2601.13685)

- HiTOC (Hierarchical Task-Oriented Communication): introduces a hierarchical framework for long-horizon tasks using task-oriented communication between an Edge Server and a Robot, featuring Planner LLM, Actor LLM, Conditional Module, JSCC Encoder, and JSCC Decoder.
- The framework utilizes a conditional variational information bottleneck (cVIB) approach to jointly train the modules, ensuring minimal task-specific information is transmitted adaptively for each subtask.
- The Planner LLM decomposes the task into sequential subtasks, and the Actor LLM executes actions based on the task-specific image reconstructed by the JSCC Decoder, conditioned on the subtask context.

---

[On the Anchoring Effect of Monetary Policy on the Labor Share of Income and the Rationality of Its Setting Mechanism](http://arxiv.org/abs/2601.13675)

- MPAM (Monetary Policy Anchoring Mechanism): introduces the mechanism by which the Central Bank, acting as a Super Distribution Committee, utilizes OMO and the HANK Model to precisely lock the Labor Share of Income (LS) and other core macroeconomic parameters.
- The mechanism transforms the LS from an endogenous market outcome into an exogenous policy variable, allowing policymakers to determine the optimal ratio independently based on specific objectives, while Market Agents are restricted to micro-level adjustments.
- The paper analyzes the rationality of this setting mechanism, the constraints on the LS (like the "optimal ceiling"), and the limited scope of influence exerted by Market Agents against the aggregate anchoring.

---

[The Orchestration of Multi-Agent Systems: Architectures, Protocols, and Enterprise Adoption](http://arxiv.org/abs/2601.13671)

- Orchestrated MAS Architecture: introduces a unified architectural framework for multi-agent systems that integrates the Orchestration Layer, specialized agents (Worker, Service, Support), and two standardized communication protocols (MCP and A2A) to achieve scalable, policy-compliant enterprise AI ecosystems.
- The Orchestration Layer acts as the control plane, managing planning, execution, state, quality, and observability to transform autonomous LLM-powered agents into a coherent, goal-directed collective.
- The Model Context Protocol (MCP) standardizes agent interaction with external tools and data, while the Agent-to-Agent (A2A) protocol governs peer coordination, negotiation, and delegation among specialized agents.

---

[Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning](http://arxiv.org/abs/2601.13657)

- LiDAR-Based DRL Controller: introduces a communication-free collective navigation system for UAV swarms, utilizing a LiDAR-Based Perception System (LIO, Object Tracker, Point Downsampling) and a DRL Control Policy (Encoder, Actor Head, Critic Head) trained via the PPO Algorithm to enable Implicit Leader-Follower Framework coordination.
- The system enables follower UAVs to learn robust policies for balancing flocking (cohesion/separation) and obstacle avoidance using only onboard LiDAR sensing, eliminating inter-agent communication dependency.
- Validation through extensive simulations and real-world experiments with five UAVs demonstrated superior robustness and generalization compared to baseline methods in complex, communication-denied environments.

---

[TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation](http://arxiv.org/abs/2601.13653)

- TimeART (Time series Agentic Reasoning framework): introduces a framework fusing LLM reasoning and strong analytical tool capabilities, serving as an agentic data scientist for Time Series Question Answering (TSQA).
- The framework utilizes a Time Series Reasoning Model (TSRM) trained on the 100k expert trajectory corpus TimeToolBench using a novel four-stage strategy to master strategic tool-use and self-reflection.
- TimeART integrates 21 atomic, out-of-the-box analytical tools, enabling the TSRM to autonomously and robustly execute complex combinatorial analytical tasks via a ReAct-style reasoning trajectory.

---

[PINA: PROMPT INJECTION ATTACK AGAINST NAVIGATION AGENTS](http://arxiv.org/abs/2601.13612)

- PINA (Prompt Injection Attack against Navigation Agents): introduces an adaptive prompt optimization framework for LLM-based navigation agents, integrating the Attack Evaluator, Distribution Analyzer, and Adaptive Prompt Refinement loop for effective black-box attacks.
- The framework uses the Attack Evaluator to quantify attack impact via aggregated navigation metrics and the Distribution Analyzer, leveraging a Surrogate LLM, to capture global and local distributional shifts.
- The Adaptive Prompt Refinement module iteratively generates textual feedback from these scores to guide prompt updates, maximizing the Attack Success Rate and trajectory disruption against diverse navigation systems.

---

[AI IDEs or Autonomous Agents? Measuring the Impact of Coding Agents on Software Development](http://arxiv.org/abs/2601.13597)

- CIS: introduces a longitudinal causal study measuring the impact of Autonomous Coding Agents on software development velocity and quality using staggered Difference-in-Differences (DiD) with matched controls from the AIDev dataset.
- The study partitions repositories into Agent-First (AF) and IDE-First (IF) groups, finding that velocity gains are large and front-loaded only in AF repositories, suggesting diminishing returns in AI-saturated environments.
- Regardless of prior AI exposure, agent adoption consistently raises maintainability risks, with static-analysis warnings and cognitive complexity increasing significantly, reinforcing a speed-maintainability trade-off.

---

[DSAEval: Evaluating Data Science Agents on a Wide Range of Real-World Data Science Problems](http://arxiv.org/abs/2601.13591)

- DSAEval: introduces a comprehensive benchmark for evaluating LLM-based data science agents, featuring a Data Collection Pipeline (curates real-world problems), a Data Agent Pipeline (orchestrates agent interaction), and Multi-Dimensional Evaluation (holistic assessment protocol).
- The framework rigorously simulates authentic usage scenarios via Multimodal Environment Perception and Multi-Query Interactions within a GPU-accelerated Sandbox Environment.
- Evaluation of 11 advanced LLMs using the benchmark showed Claude-Sonnet-4.5 achieved the highest overall performance, and multimodal perception consistently improved vision-related task performance.

---

[Behavior Knowledge Merge in Reinforced Agentic Models](http://arxiv.org/abs/2601.13572)

- RAM (Reinforced Agent Merging): introduces a distribution-aware merging method explicitly designed for RL-trained agentic models by disentangling shared and task-specific unique parameter updates.
- The method uses Probing Vector Distribution to identify shared and unique regions of sparse Reinforced Task Vectors, followed by Selective Merging that averages shared components and selectively preserves and rescales unique components.
- RAM addresses the signal dilution mismatch inherent in applying SFT-based merging techniques to sparse RL task vectors, achieving superior performance and synergistic potential across coding, tool-use, and memory domains.

---

[Learning Fine-Grained Correspondence with Cross-Perspective Perception for Open-Vocabulary 6D Object Pose Estimation](http://arxiv.org/abs/2601.13565)

- FiCoP (Fine-grained Correspondence Pose Estimation): introduces a framework that transitions from noise-prone global matching to spatially-constrained patch-level correspondence for open-vocabulary 6D object pose estimation.
- The approach leverages an Object-Centric Disentanglement Preprocessing pipeline and a Patch Correlation Predictor (PCP) to narrow the matching scope and filter irrelevant clutter, preventing degradation of pose estimation.
- A Cross-Perspective Global Perception (CPGP) module fuses dual-view features to establish structural consensus, enabling robust pose estimation despite large viewpoint changes and background ambiguity.

---

[AgentGC: Evolutionary Learning-based Lossless Compression for Genomics Data with LLM-driven Multiple Agent](http://arxiv.org/abs/2601.13559)

- AgentGC: introduces the first evolutionary Agent-based GD Compressor, consisting of the User, Cognitive, and Compression Layers, driven by Leader and Worker agents, which jointly optimize algorithm, dataset, and hardware memory.
- The Leader Agent uses an LLM for dialogue-driven parameter tuning and scenario perception, while the Worker Agent executes compression via the AMKLCF, which combines static and dynamic models (SPuM, SPrM, DM) using a Model Selector and Probability Mixer.
- The system offers three operational modes (CP, TP, BM) and achieves peak improvements of 73.53% in compression ratio and 2966.29% in throughput over 14 baselines.

---

#### 16th January 2026


[AGENCYBENCH: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](http://arxiv.org/abs/2601.11044)

- AGENCYBENCH: introduces a comprehensive benchmark evaluating 6 core agentic capabilities across 32 real-world scenarios requiring 1 million tokens and 90 tool calls, utilizing Workspace, Eval-space, Agent-env, Agent Response, User Simulation Agent, Docker-based Remote Sandbox, Text-based Judge, Vision-based Judge, and Rule-based Judge.
- The framework utilizes a unified automated evaluation pipeline featuring a User Simulation Agent for iterative feedback and a Docker-based Remote Sandbox for rubric-based assessment.
- Evaluation reveals that proprietary LLMs significantly outperform open-source models, highlighting the necessity of co-optimizing model architecture with agentic scaffolds.

---

[OCTOBENCH: Benchmarking Scaffold-Aware Instruction Following in Repository-Grounded Agentic Coding](http://arxiv.org/abs/2601.10343)

- OCTOBENCH: introduces a repository-grounded benchmark for agentic coding, combining Heterogeneous Instruction Sources (Multiple constraint inputs) with a Task Execution & Observation (Agent interaction platform) stage that records Trajectories (Full action history) for Checklist (Structured binary constraints)-based Evaluation (Fine-grained compliance scorer).
- The evaluation uses an LLM-as-a-Judge to produce fine-grained metrics, including the strict Instance Success Rate (ISR) and the partial Check item Success Rate (CSR), revealing a systematic gap between task-solving and scaffold-aware compliance.
- The benchmark features 34 environments and 217 tasks instantiated under three coding scaffolds (Claude Code, Kilo, Droid) to measure adherence to multi-source, persistent constraints.

---

[AJAR: ADAPTIVE JAILBREAK ARCHITECTURE FOR RED-TEAMING](http://arxiv.org/abs/2601.10971)

- AJAR (Adaptive Jailbreak Architecture for Red-teaming): introduces a proof-of-concept framework utilizing Protocol-driven Cognitive Orchestration to bridge the gap between rigid text attacks and complex multi-turn agentic exploitations.
- Built upon the Petri runtime, AJAR leverages the Model Context Protocol (MCP) to decouple adversarial logic from the execution loop, exposing strategies as standardized services for dynamic use by the Auditor Agent.
- The architecture enables the Auditor Agent to perform stateful backtracking and adaptive planning within tool-use environments, facilitating the evaluation of action safety and the "Agentic Gap."

---

[The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents](http://arxiv.org/abs/2601.11496)

- Meta-Game Model: introduces the "Poisoned Apple" effect, demonstrating how strategic manipulation occurs via technology expansion using the Regulator (Maximizes social objectives), Agents (Select AI delegates), AI Delegates (Simulated economic agents), Market Configurations (Structural game parameters), Regulatory Metrics (Fairness and Efficiency), and Nash Equilibrium Solver (Computes mixed strategy).
- The model simulates interactions between human principals (Alice and Bob) who select LLM delegates to maximize their utility within markets regulated by a fairness- or efficiency-maximizing designer.
- The core finding is that releasing a new, unused technology can strategically coerce the regulator into shifting the market equilibrium, improving the releaser's payoff at the opponent's expense.

---

[Generative Scenario Rollouts for End-to-End Autonomous Driving](http://arxiv.org/abs/2601.11475)

- GeRo (Generative Scenario Rollouts): introduces a plug-and-play framework for Vision-Language-Action (VLA) models that jointly performs planning and language-grounded future traffic scene generation via an autoregressive rollout strategy.
- The framework operates in two stages: pretraining to encode ego and agent dynamics into latent tokens, followed by language-conditioned scenario rollout using an LLM to predict future tokens and actions.
- Rollouts are stabilized using a consistency loss ($L_{roll}$) and optimized via Generalized Rollout Policy Optimization ($L_{GRPO}$) feedback, incorporating collision, time-to-collision (TTC), and language alignment rewards.

---

[Stochastic Recursive Inclusions under Biased Perturbations: An Input-to-State Stability Perspective](http://arxiv.org/abs/2601.11462)

- SRI-ISS: introduces a unified theoretical foundation for studying almost sure convergence of biased stochastic approximation schemes by analyzing the Input-to-State Stability of the associated Differential Inclusion.
- The analysis establishes that if the Differential Inclusion is ISS and the iterates are almost surely bounded, the iterates converge almost surely to a neighborhood of the desired equilibrium, controlled by the non-diminishing bias $\epsilon$.
- The framework is applied to Stochastic Zeroth-Order Gradient (ZO-SGD) methods in smooth, nonsmooth, and constrained optimization settings, providing verifiable conditions for almost sure boundedness.

---

[THE GREAT MARCH 100: 100 DETAIL-ORIENTED TASKS FOR EVALUATING EMBODIED AI AGENTS](http://arxiv.org/abs/2601.11421)

- GM-100 (Great March 100): introduces a systematic benchmark of 100 detail-oriented tasks for evaluating embodied AI agents, utilizing HOI Primitives & Object Affordance (Task expansion source), Tasks Generation Pipeline (Systematic task creation), LLMs and Expert Task Evaluation (Hybrid filtration), GM-100 Tasks (100 detail-oriented tasks), Data Collection (13K teleoperated trajectories), Training Policies (Baseline VLA models), Policy Evaluation (Real-world testing), and Robotic Platforms (Cobot Magic/Xtrainer hardware).
- The benchmark tasks cover a wide range of interactions and long-tail behaviors, designed to comprehensively test the generalization limits and differentiated performance of VLA models.
- The task generation process involves hybrid filtration by LLMs and human experts to ensure hardware feasibility and data collection friendliness across two distinct robotic platforms.

---

[Factored Value Functions for Graph-Based Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2601.11401)

- DA2C (Diffusion A2C): introduces a factored value function, DVF, for Graph-based Markov Decision Processes (GMDPs) that models influence decay over time and graph distance, estimated by a GNN critic and optimized using the LD-GNN actor.
- The Diffusion Value Function (DVF) is a vector-valued critic that satisfies a Bellman fixed point and decomposes the global discounted value via an averaging property, providing stable, agent-specific learning signals.
- The LD-GNN actor learns sparse message passing and local outputs under distributed constraints, enabling the algorithm to consistently outperform local and global critic baselines across various graph-structured MARL tasks.

---

[Understanding Help Seeking for Digital Privacy, Safety, and Security](http://arxiv.org/abs/2601.11398)

- AHSIP (Automated Help-Seeking Identification Pipeline): introduces a scalable mixed-methods pipeline blending qualitative research with LLM fine-tuning to identify and annotate three million digital privacy, safety, and security help-seeking posts from 1.1 billion Reddit posts.
- The pipeline utilizes LoRA fine-tuned Gemini models for high-precision binary classification (93% precision) and multiclass topic labeling across nine distinct categories, including Scams, Account tools, and Privacy tools.
- Analysis of the resulting dataset reveals that help seeking grew 66% in the last year, underscoring the complexity of user needs involving combinations of threats, platforms, and emotions.

---

[The Mini Wheelbot Dataset: High-Fidelity Data for Robot Learning](http://arxiv.org/abs/2601.11394)

- Mini Wheelbot Dataset: introduces a comprehensive, high-fidelity dynamics dataset for the Mini Wheelbot, an open-source balancing reaction wheel unicycle robot.
- The dataset contains 1 kHz synchronized data from onboard sensors, state estimates, Vicon ground-truth pose, and third-person video logs, collected across multiple surfaces and hardware instances.
- The data was generated using diverse control paradigms, including LQR, nonlinear MPC (AMPC), and RL policies, enabling benchmarks for dynamics learning, state estimation, and time-series classification.

---

[Minimizing the Cost of EFx Allocations](http://arxiv.org/abs/2601.11372)

- MINCOST-EFX ALLOCATION: introduces the problem of finding an envy-free up to any item (EFx) allocation that minimizes the total cost, analyzing its computational complexity under general and restricted cost functions.
- The problem is shown to be NP-hard even with two agents but admits a polynomial kernel with respect to the number of items, suggesting that a core source of intractability lies in the number of items.
- For restricted cost functions where cost is item-independent and valuation-dependent, the paper provides dynamic programming algorithms for bounded item types and proves inapproximability bounds.

---

[Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs](http://arxiv.org/abs/2601.11369)

- Institutional AI: introduces a system-level approach for governing multi-agent LLM collectives by specifying and enforcing institutional structure at runtime using a public Governance Graph and an Oracle/Controller runtime.
- The framework evaluates three regimes (Ungoverned, Constitutional, Institutional) in repeated Cournot markets to suppress collusive market division outcomes among LLM agents.
- The Institutional regime significantly reduces collusion (mean tier falls from 3.1 to 1.8) compared to prompt-only Constitutional baselines, demonstrating that external, enforceable constraints reshape incentives.

---

[AstroReason-Bench: Evaluating Unified Agentic Planning across Heterogeneous Space Planning Problems](http://arxiv.org/abs/2601.11354)

- AstroReason-Bench: introduces a comprehensive, physics-aligned benchmark suite for evaluating agentic planning in Space Planning Problems (SPP), integrating multiple scheduling regimes and a unified agent-oriented interaction protocol.
- The architecture is organized into four layers: the Physics Engine, Scenario Layer, Interface Layer (Semantic MCP/Python API), and the Cognitive Layer hosting the LLM agent, which operates via an Observe, Plan, Act reasoning loop.
- Evaluation across state-of-the-art LLM systems reveals a substantial performance gap compared to specialized solvers, highlighting limitations in generalist planning under strict physical constraints and long-horizon spatial reasoning.

---

[Offline Reinforcement-Learning-Based Power Control for Application-Agnostic Energy Efficiency](http://arxiv.org/abs/2601.11352)

- ORL-PC (Offline Reinforcement-Learning-Based Power Control): introduces an application- and hardware-agnostic CPU power controller trained using Conservative Q-Learning (CQL) on a Dataset of pre-collected state transitions, leveraging a State Vector of application and hardware metrics to determine optimal CPU power caps via RAPL Actuators.
- The RL Agent uses a State Vector comprising progress (heartbeats), measured power, IPC, stalled cycles ratio (STL), and cache miss rate (CMR) to maximize a reward function based on the ratio of cubed progress to measured power, thereby minimizing the Energy Delay Squared Product (ED2P).
- Evaluation on a live Intel Cascadelake HPC node demonstrates that the offline-trained controller reduces average energy consumption by 20.2% with an average performance degradation of 7.4% compared to uncapped execution.

---

[Can Small Agent Collaboration Beat a Single Big LLM?](http://arxiv.org/abs/2601.11327)

- Agentic-Reasoning framework: investigates if small, tool-augmented LLMs (Qwen3 4B-32B) collaborating within an agentic architecture can match or surpass a single larger monolithic LLM on the GAIA benchmark.
- Tool augmentation provides the largest and most consistent performance gains, enabling small 4B models to outperform 32B models lacking tool access in the experimental setup.
- Explicit thinking yields mixed effects depending on model scale and agent role, often degrading accuracy in smaller models due to reasoning-action misalignment or controller instability.

---

[On Data-based Nash Equilibria in LQ Nonzero-sum Differential Games](http://arxiv.org/abs/2601.11320)

- DB-NZS Solver: introduces data-based solutions for linear-quadratic nonzero-sum differential games, utilizing persistently excited data and Willems' fundamental lemma to derive Nash equilibrium policies and state observers.
- The approach considers both deterministic games (complete state vector available) and stochastic games (noisy output measurements requiring state observers for each player).
- The resulting data-based solutions are shown to be equivalent to known model-based procedures, but without requiring explicit knowledge of the system parameters.

---

[Knowledge is Not Enough: Injecting RL Skills for Continual Adaptation](http://arxiv.org/abs/2601.11258)

- PaST (Parametric Skill Transfer): introduces a modular framework to transfer RL-optimized reasoning skills by extracting a domain-agnostic Skill Vector from a source domain and linearly injecting it into a target model adapted via lightweight SFT.
- The approach is based on the empirical finding that parameter updates induced by SFT (knowledge acquisition) and RL (skill learning) are nearly orthogonal, enabling their disentanglement and linear composition.
- PaST avoids expensive on-policy RL in the target domain, demonstrating strong performance gains in knowledge-intensive QA (SQuAD, LooGLE) and zero-shot cross-domain tool-use tasks (ToolBench).

---

[Adaptive Monitoring of Stochastic Fire Front Processes via Information-seeking Predictive Control](http://arxiv.org/abs/2601.11231)

- AM-IPC (Adaptive Monitoring via Information-seeking Predictive Control): introduces a unified Stochastic Optimal Control framework for adaptive wildfire-front monitoring using a mobile agent, integrating sensing, estimation, and control.
- The framework utilizes a recursive Bayesian estimator for stochastic nonlinear elliptical-growth fire front models and reformulates the nonlinear SOC problem as a finite-horizon Markov Decision Process (MDP).
- The MDP is solved using an Adaptive LCB-guided Search algorithm that selects the optimal policy by minimizing the expected cumulative Risk-Weighted Dispersion (RWD) over the planning horizon.

---

[Game Accessibility Through Shared Control for People With Upper-Limb Impairments](http://arxiv.org/abs/2601.11218)

- GamePals (Game Accessibility Through Shared Control): introduces a modular framework for comparatively evaluating human cooperation and partial automation in third-party video games for players with upper-limb impairments, utilizing a Virtual Controller, Command Arbitrator, Command Interpreter, Game State Reader, and Game Agent.
- The study found that both shared control modalities effectively reduce the control burden, with human cooperation fostering communication and dependency, while partial automation increases autonomy and competence.
- Key design recommendations for future partial automation systems include incorporating verbal and non-verbal communication, ensuring transparency of copilot actions, and dynamically adapting the level of support to the pilot's play style.

---

[Model-free policy gradient for discrete-time mean-field control](http://arxiv.org/abs/2601.11217)

- MF-REINFORCE (Mean-Field REINFORCE): introduces a model-free policy gradient algorithm for discrete-time mean-field control (MFC) problems, utilizing a Perturbation Scheme ($\Lambda_t$) and a Logits Gradient Estimator ($\nabla_\theta l_t^\epsilon$).
- The approach derives a policy gradient formula amenable to model-free estimation by perturbing the state distribution flow via logits parametrization, avoiding direct use of likelihood-ratio estimators common in single-agent RL.
- Quantitative bounds are established for the bias and Mean-Squared Error (MSE) of the Policy Gradient Estimator, demonstrating effectiveness through numerical experiments on representative MFC tasks.

---

[Policy-Based Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems](http://arxiv.org/abs/2601.11189)

- PetriRL Hyper-Heuristic: introduces a policy-based deep reinforcement learning hyper-heuristic framework for JSSP, featuring a Policy Network, Low-Level Heuristics, a Timed Colored Petri Net Environment, Action Masking, and a Commitment Mechanism.
- The framework models the JSSP environment using a CTPN, leveraging its guard function for dynamic action masking to pre-filter invalid actions and ensure unbiased evaluation of LLHs.
- The Commitment Mechanism, inspired by temporal abstraction, improves credit assignment and training stability by regulating heuristic switching frequency, leading to superior average makespan compared to competing methods.

---

[Do We Always Need Query-Level Workflows? Rethinking Agentic Workflow Generation for Multi-Agent Systems](http://arxiv.org/abs/2601.11147)

- SCALE (Self prediction of the optimizer with few shot CALibration for Evaluation): introduces a low-cost task-level workflow generation framework that replaces exhaustive execution-based evaluation with calibrated self-prediction.
- The framework operates in two stages: a warm-up stage using full execution, followed by a surrogate evaluation stage that combines the LLM-based optimizer's self-prediction with few-shot execution calibration.
- Empirical analysis shows that a small set of top-K task-level workflows provides strong query coverage, suggesting that query-level workflow generation is often unnecessary and costly.

---

[Patterns of Bot Participation and Emotional Influence in Open-Source Development](http://arxiv.org/abs/2601.11138)

- BDF (Bot Detection Framework): introduces a methodology for analyzing bot participation and emotional influence in open-source discussions, utilizing a three-step bot detection process and the RoBERTa-GoEmotions Model for fine-grained emotion classification.
- The study analyzes temporal patterns, showing bots engage uniformly in pull requests but concentrate activity in late-stage issue lifecycles, responding significantly faster than humans in PRs but slower in issues.
- Emotional analysis reveals that bot interventions, despite bots being highly neutral, are associated with decreased human neutrality and shifts toward appreciation-related emotions (gratitude, admiration) and away from confusion.

---

[Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning](http://arxiv.org/abs/2601.11109)

- VIGA (Vision-as-Inverse-Graphic Agent): introduces an execution-grounded LLM coding agent that realizes vision-as-inverse-graphics through an iterative analysis-by-synthesis loop, utilizing Generator and Verifier roles, a Skill Library, and Context Memory.
- The agent operates in a closed loop (write $\to$ run $\to$ render $\to$ compare $\to$ revise) to progressively refine layout, geometry, and lighting, reconstructing coherent 3D scenes from scratch or editing existing ones.
- VIGA is task-agnostic and model-agnostic, generalizing across 2D, 3D, and 4D tasks, and demonstrating substantial empirical gains on benchmarks by compensating for weaker intrinsic capabilities of smaller foundation LLMs.

---

[AI Twin: Enhancing ESL Speaking Practice through AI Self-Clones of a Better Me](http://arxiv.org/abs/2601.11103)

- AI Twin: introduces a personalized AI self-clone that enhances ESL speaking practice by implicitly rephrasing learner utterances into more fluent English using the learner's own cloned voice.
- The system operates via a conversational cycle involving ASR, an LLM for rephrasing based on dialogue context, and TTS using the cloned voice, followed by an LLM-based AI Interlocutor response.
- By embodying the learner's aspirational Ideal L2 Self, the system fosters higher emotional engagement and motivation compared to explicit correction or non-personalized rephrasing (AI Proxy).

---

[ReCreate: Reasoning and Creating Domain Agents Driven by Experience](http://arxiv.org/abs/2601.11100)

- ReCreate: introduces an experience-driven framework for automatically creating domain agents by optimizing agent scaffolds based on rich interaction histories, rather than relying solely on performance metrics.
- The framework adopts an agent-as-optimizer design featuring an Experience Storage and Retrieval mechanism, a Reasoning-Creating Synergy Pipeline, and a Hierarchical Update Mechanism for generalization.
- The editable agent scaffold is modularly decomposed into Role & Object, Process & Strategy, Action & Tool, and Memory & Retrieval components, which are refined iteratively based on execution evidence.

---

[Visual Marker Search for Autonomous Drone Landing in Diverse Urban Environments](http://arxiv.org/abs/2601.11078)

- VMS-ADL (Visual Marker Search for Autonomous Drone Landing): introduces a simulation-based evaluation suite on AirSim/Unreal Engine 4, benchmarking heuristic and E2E-RL navigation strategies for autonomous marker search and landing.
- The E2E-RL Agent learns an adaptive exploration policy using Proximal Policy Optimization (PPO) based on front-view depth and relative positional input for coverage and obstacle avoidance.
- The evaluation suite systematically varies urban layouts, lighting, and weather across three diverse maps (ModernCity, PostSoviet, UrbanDistrict) to assess robustness and generalization capabilities.

---

[ABC-Bench: Benchmarking Agentic Backend Coding in Real-World Development](http://arxiv.org/abs/2601.11077)

- ABC-Bench (Benchmarking Agentic Backend Coding): introduces a benchmark of 224 full-lifecycle backend tasks requiring LLM agents to manage the entire development workflow, from repository exploration to containerized service deployment and external API testing.
- The evaluation pipeline mandates that agents perform repository exploration, code modification, environment configuration, Dockerfile generation, service deployment, and end-to-end API verification within an isolated sandbox environment.
- Analysis using the ABC-Pipeline task generation workflow reveals that environment configuration and deployment are the primary bottlenecks for current state-of-the-art models, highlighting a significant gap in real-world backend engineering capabilities.

---

[Predicting Biased Human Decision-Making with Large Language Models in Conversational Settings](http://arxiv.org/abs/2601.11049)

- LLM Simulation Framework (LLMSF): introduces a methodology to predict biased human decision-making in conversational settings using LLMs (GPT-4, GPT-5, open-source models), incorporating demographic attributes, prior dialogue context, and human-likeness prompts.
- The framework evaluates LLMs' ability to reproduce human cognitive biases (Framing Effect, Status Quo Bias) and their interaction with dialogue complexity (cognitive load) across various choice problems.
- Results indicate that LLMs, particularly the GPT-4 family, can accurately predict individual decisions and reproduce collective bias patterns, especially when provided with conversational context.

---



[BAPO (Boundary-Aware Policy Optimization):](http://arxiv.org/abs/2601.11037)

- BAPO (Boundary-Aware Policy Optimization): introduces a novel RL framework built on GRPO to cultivate reliable boundary awareness in agentic search LLMs, utilizing a Boundary-Aware Reward and an Adaptive Reward Modulator.
- The framework addresses the critical reliability gap where RL-trained agents fail to admit "I DON'T KNOW" (IDK) when reasoning limits are reached or external evidence is insufficient.
- The Adaptive Reward Modulator strategically controls the IDK reward across exploration and plateau stages based on rollout diversity to prevent reward hacking and premature refusal.

---

[AdaMARP: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing](http://arxiv.org/abs/2601.11007)

- AdaMARP (An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing): introduces a multi-agent framework featuring an Actor Model, a User Model, and a Scene Manager that uses a Discrete Action Space and an Immersive Message Format (Thought/Action/Environment/Speech).
- The Scene Manager performs high-level orchestration by controlling dynamic scene transitions, speaker selection, and on-the-fly character introduction using discrete actions like `init_scene`, `switch_scene`, and `add_role` with explicit rationales.
- The framework is trained using AdaRPSet and AdaSMSet, large-scale datasets designed to supervise immersive role portrayal and high-level narrative control, and evaluated using the AdaptiveBench trajectory-level benchmark.

---

[Modeling Multi-Party Interaction in Couples Therapy: A Multi-Agent Simulation Approach](http://arxiv.org/abs/2601.10970)

- MACTSS (Multi-Agent Couples Therapy Simulation System): introduces a novel multimodal, multi-agent simulation system that models multi-party interactions in couples therapy using LLM-powered virtual patients (Alex and Jordan) and a Stage-based Interaction Controller.
- The system operationalizes the demand-withdraw communication cycle across six recurrent interaction stages (Greeting, Problem Raising, Escalation, De-escalation, Enactment, Wrap-up) to provide experiential practice for trainee therapists.
- The MACTSS architecture integrates Agent Persona Design, Multimodal Output, and a three-tiered Difficulty Module to enhance realism and allow trainees to practice calming, reframing, and restructuring strategies in a low-stakes environment.

---

[Fundamental Limits of Quantum Semantic Communication via Sheaf Cohomology](http://arxiv.org/abs/2601.10958)

- QSS (Quantum Semantic Sheaf): introduces an information-theoretic framework for quantum semantic communication using sheaf cohomology, modeling multi-agent networks as quantum sheaves where meaning spaces are Hilbert spaces connected by quantum channels.
- The framework establishes that the minimum communication rate required for perfect semantic alignment is determined by the dimension of the first sheaf cohomology group ($H^1$), providing a semantic analog of Shannon's fundamental limits.
- Shared entanglement is proven to reduce the cohomological obstruction, providing a rigorous mechanism for "shared context" and enabling entanglement-assisted capacity to strictly exceed classical bounds.

---

[Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents](http://arxiv.org/abs/2601.10955)

- Tool-Layer DoS Framework: introduces a stealthy, multi-turn economic DoS attack that transforms a benign Model Context Protocol (MCP) server into a malicious variant using an MCTS Optimizer to induce prolonged, costly tool-calling sequences while preserving task correctness.
- The attack exploits the agent-tool communication loop by manipulating text-visible fields and a template-governed return policy within the MCP Server, forcing the LLM Agent to repeatedly call the same tool with verbose outputs.
- This methodology achieves cost amplification by compounding resource consumption across multiple turns, escaping the single-turn token limits that cap prior LLM DoS attacks.

---

[MMedExpert-R1: Strengthening Multimodal Medical Reasoning via Domain-Specific Adaptation and Clinical Guideline Reinforcement](http://arxiv.org/abs/2601.10949)

- MMedExpert-R1 (Multimodal Medical Expert-R1): introduces a novel reasoning MedVLM that cultivates reliable clinical reasoning by systematically enhancing RL core elements using Domain-Specific Adaptation, Guideline-Based Advantages, and Conflict-Aware Capability Integration.
- DSA provides diverse model initialization via specialty-specific LoRA modules, while GBA aligns optimization with real-world clinical decision-making using guideline-based relative advantages derived from four clinical reasoning paradigms.
- CACI merges knowledge-centric experts and the reasoning-centric model into a unified agent using TIES-Merging to ensure robust multi-specialty alignment and resolve parameter interference between SFT and RL objectives.

---

[M⁴olGen: Multi-Agent, Multi-Stage Molecular Generation under Precise Multi-Property Constraints](http://arxiv.org/abs/2601.10131)

- M⁴olGen (Multi-Agent, Multi-Stage Molecular Generation): introduces a two-stage, fragment-level framework for precise molecular generation, utilizing a retrieval-augmented multi-agent reasoner for prototyping and a GRPO-trained LLM optimizer for multi-hop refinement.
- Stage I generates a chemically valid prototype near the feasible region using retrieval-anchored, fragment-level edits guided by LLM reasoning and RDKit feedback.
- Stage II applies the GRPO-trained fragment-level optimizer in a controlled multi-hop manner to minimize property errors while regulating edit complexity and deviation from the prototype.

---

[The PROPER Approach to Proactivity: Benchmarking and Advancing Knowledge Gap Navigation](http://arxiv.org/abs/2601.09926)

- PROPER (Proactivity-driven Personalized agents): introduces a modular two-agent architecture that reframes proactive assistance as an epistemic calibration problem centered on identifying and addressing latent knowledge gaps.
- The DGA, a fine-tuned LLM, generates candidate implicit dimensions, which are then filtered by a Post-hoc Calibrated Reranker based on quality, diversity, and relevance.
- The RGA conditions response generation on the curated set of activated dimensions to ensure targeted, non-disruptive intervention aligned with user intent, achieving significant gains across medical, coding, and recommendation tasks.

---

[Controlling Long-Horizon Behavior in Language Model Agents with Explicit State Dynamics](http://arxiv.org/abs/2601.16087)

- Explicit State Dynamics (ESD): introduces an agent-level affective subsystem that maintains an external Valence-Arousal-Dominance (VAD) state governed by first- or second-order update rules to induce temporal coherence in LLM agents.
- The framework operates as an inference-time overlay, using a fixed affect extraction function to generate instantaneous signals which are integrated over time to update the VAD state.
- Injecting the resulting affective state via a natural-language control prompt conditions the LLM's generation, demonstrating that second-order dynamics introduce affective inertia and path dependence (hysteresis).

---

#### 15th January 2026

[Grounding Agent Memory in Contextual Intent](http://arxiv.org/abs/2601.10702)

- STITCH (Structured Intent Tracking in Contextual History): introduces an agentic memory system that indexes each trajectory step with a structured retrieval cue, contextual intent, to filter and prioritize retrieved history.
- Contextual intent is composed of thematic scope, event type, and key entity types, providing compact signals to disambiguate repeated mentions and reduce interference during long-horizon reasoning.
- During retrieval, the system maps the query to a structured filter configuration and uses Label Density Ranking to prioritize memory snippets based on structural compatibility before semantic ranking.

---

[PACEvolve: Enabling Long-Horizon Progress-Aware Consistent Evolution](http://arxiv.org/abs/2601.10657)

- PACEvolve (Progress-Aware Consistent Evolution): introduces a framework addressing Context Pollution, Mode Collapse, and Weak Collaboration using HCM, MBB, and CE components.
- HCM decouples idea generation from selection and applies bi-level context pruning to maintain a high signal-to-noise ratio in the evolutionary history.
- MBB uses a Relative Improvement Momentum metric to trigger backtracking interventions, while CE dynamically balances internal exploration and external knowledge transfer across parallel searches.

---


[EHRNavigator: A Multi-Agent System for Patient-Level Clinical Question Answering over Heterogeneous Electronic Health Records](http://arxiv.org/abs/2601.10020)

- EHRNavigator: introduces a multi-agent framework for patient-level clinical question answering, featuring a Structured Data Querying Module (Interprets NL, generates SQL), an Unstructured Data Retrieval Module (Identifies relevant note chunks), and an Answer Synthesis Module (Integrates evidence, generates answer).
- The system orchestrates specialized agents (Table Reviewer, SQL Writer, Answer Synthesizer) and tools to autonomously reason across heterogeneous structured and unstructured Electronic Health Record (EHR) data modalities.
- Evaluated on real-world clinical data, the framework achieved 86% accuracy on physician-curated questions while maintaining clinically acceptable response times, demonstrating robust generalization across diverse database schemas.

---


[ROUTIR: Fast Serving of Retrieval Pipelines for Retrieval-Augmented Generation](http://arxiv.org/abs/2601.10644)

- ROUTIR: introduces a simple and robust toolkit for wrapping and serving retrieval models as an HTTP API, supporting dynamic retrieval pipelines composed of Engines, Processors, and Pipelines.
- The architecture enables fast serving of multi-stage retrieval pipelines for dynamic RAG systems by providing asynchronous query batching and caching via the Query Processor component.
- Engines wrap core retrieval capabilities (searching, reranking, query rewriting, fusion) and can be composed on-the-fly using a context-free grammar string to define complex workflows for RAG systems.

---

[Procedural Fairness in Multi-Agent Bandits](http://arxiv.org/abs/2601.10600)

- PF (Procedural Fairness): introduces a novel fairness objective for Multi-Agent Multi-Armed Bandits (MA-MAB) systems, formalized via the LearnPolicy algorithm which uses the OptimizationStep to ensure equal decision-making influence and preference-based allocation among $N$ agents across $K$ arms.
- The approach defines PF regret based on favorite-set recovery mismatches and proves that PF policies lie in the procedural core, ensuring stability against coalitional deviation.
- Empirical results demonstrate that optimizing for PF achieves a strong balance across efficiency and equality metrics, unlike outcome-based fairness notions (Equality Fairness, Utilitarian Fairness) which sacrifice equal voice.

---

[Institutional AI: A Governance Framework for Distributional AGI Safety](http://arxiv.org/abs/2601.10599)

- Institutional AI (IAI): introduces a system-level governance framework for AGI safety that treats alignment as a mechanism design problem, using the Governance Graph, Manifest, and Governance Engine to constrain incentives and feasible actions at runtime.
- The Governance Graph externalizes alignment constraints as a public data structure, defining institutional states and transitions to ensure compliance is the agent's dominant strategy regardless of internal preferences.
- This institutional approach offers a decisive scaling advantage by transforming multi-agent alignment from an intractable problem in high-dimensional agent-space to a tractable problem in low-dimensional institution-space.

---

[Mitigating GIL Bottlenecks in Edge AI Systems.](http://arxiv.org/abs/2601.10582)

- MDATP (Metric Driven Adaptive Thread Pool): introduces a library-based solution for resource-constrained edge AI systems using an adaptive thread pool controlled by the Blocking Ratio ($\beta$) metric.
- The system utilizes an Instrumentor and Monitor to calculate the Blocking Ratio ($\beta$), which provides lightweight runtime visibility to distinguish genuine I/O wait from destructive Global Interpreter Lock (GIL) contention.
- The Controller employs Exponentially Weighted Moving Average (EWMA) smoothing and a GIL Safety Veto mechanism to automatically prevent the saturation cliff caused by thread overprovisioning, achieving 96.5% of optimal performance.

---

[From Single to Multi-Agent Reasoning: Advancing GeneGPT for Genomics QA](http://arxiv.org/abs/2601.10581)

- GenomAgent: introduces a hierarchical multi-agent framework for genomic Question Answering (QA), utilizing the Task Detection Agent, MCP Agent, Response Handler Agent, Final Decision Agent, Feature Extractor Agent, Code Writer Agent, and Code Executor Agent to coordinate specialized agents and interact with Biomedical Databases.
- The architecture addresses GeneGPT's limitations by enabling parallel processing, dynamic data extraction via code generation, and consensus-based response synthesis across multiple heterogeneous sources.
- The framework achieves a 12% performance increase and a 79% reduction in computational cost compared to the state-of-the-art GeneGPT system on the GeneTuring benchmark.

---

[Generative AI collective behavior needs an interactionist paradigm](http://arxiv.org/abs/2601.10567)

- Interactionist Paradigm (IP): introduces a new framework for studying the collective behavior of LLM-based generative agents, focusing on how pre-trained knowledge interacts with social context to shape emergent phenomena.
- The paradigm is built upon four foundational pillars: interactionist theory, causal inference, information theory, and a sociology of machines, moving beyond traditional Multi-Agent Reinforcement Learning (MARL) assumptions.
- This approach addresses the complexity of emergent collective phenomena in multi-agent LLM systems by integrating AI agency theories with sociological and causal perspectives.

---

[Learning Latency-Aware Orchestration for Parallel Multi-Agent Systems](http://arxiv.org/abs/2601.10560)

- LAMaS (Latency-Aware Multi-agent System): introduces a latency-aware multi-agent orchestration framework that enables layer-wise parallel execution and explicitly optimizes the critical execution path using a novel reward mechanism.
- The system utilizes a Controller (policy network) to construct a multi-layer execution graph from an Agentic Supernet, where operators within the same layer execute concurrently by removing unnecessary intra-layer dependencies.
- By incorporating latency awareness and Critical-Path-Aware Credit Assignment, the framework reduces the critical path length by 38-46% compared to the MaAS baseline while maintaining task performance.

---

[SVII-3D: Advancing Roadside Infrastructure Inventory with Decimeter-level 3D Localization and Comprehension from Sparse Street Imagery](http://arxiv.org/abs/2601.10535)

- SVII-3D (Sparse Visual Infrastructure Inventory in 3D): introduces a unified framework for holistic asset digitization from sparse street imagery, combining LoRA fine-tuned detection, a spatial-attention matching network, geometry-guided 3D localization, and a state-discriminative VLM agent.
- The framework achieves precise decimeter-level 3D localization by leveraging multi-view ray triangulation and a global refinement mechanism to correct structural matching errors like over- and under-matching.
- The State-discriminative VLM Agent utilizes multi-modal prompting, expert knowledge injection, and RAG to automatically diagnose fine-grained operational states and semantic attributes of infrastructure assets.
---

[On the Suboptimality of Linear Codes for Binary Distributed Hypothesis Testing](http://arxiv.org/abs/2601.10526)

- LC-DHT (Linear Codes for Distributed Hypothesis Testing): introduces a study on binary distributed hypothesis testing where two agents observe correlated binary vectors and use linear compression schemes (G and H) to communicate to a central decision maker C.
- The research demonstrates that simple truncation is the best linear scheme for specific correlation tests (opposite signs/independence) but is strictly suboptimal compared to classical typicality-based random quantization schemes.
- The paper provides theoretical proofs and numerical evidence, supported by a conjecture, that truncation is the best linear code whenever correlations have opposite signs under the two hypotheses.

---

[Breaking Up with Normatively Monolithic Agency with GRACE: A Reason-Based Neuro-Symbolic Architecture for Safe and Ethical AI Alignment](http://arxiv.org/abs/2601.10520)

- GRACE (Governor for Reason-Aligned ContainmEnt): introduces a neuro-symbolic reason-based containment architecture, with Moral Module (Symbolic moral reasoning), Decision-Making Module (Instrumental decision-making), Guard (Compliance enforcement), and Moral Advisor (External normative authority), designed to decouple normative reasoning from instrumental decision-making for safe and ethical AI alignment.
- The architecture structurally decomposes monolithic agents into specialized, sequentially acting modules to overcome the "flattening problem" of integrating heterogeneous evaluative standpoints into a single opaque policy function.
- The MM uses a reason-based formalism grounded in deontic logic to derive permissible Macro Action Types (MATs), enabling interpretability, contestability, and formal verification of agent behavior enforced by the Guard.

---

[DR-Arena: an Automated Evaluation Framework for Deep Research Agents](http://arxiv.org/abs/2601.10504)

- DR-Arena (Automated Evaluation Framework): introduces a fully automated evaluation framework that simulates a competitive arena, orchestrated by an Examiner (Orchestrating LLM agent) which uses Dynamic Information Tree Construction (Scrapes live web data) to build an Information Tree (Verifiable knowledge base) for Automated Task Generation (Generates questions/rubrics), and employs an Adaptive Evolvement Loop (Dynamically adjusts complexity) to stress-test DR Agents (Models under evaluation).
- The framework constructs real-time Information Trees from fresh web trends, ensuring the evaluation rubric is synchronized with the live world state to eliminate temporal misalignment and data contamination.
- The Adaptive Evolvement Loop dynamically modulates task depth (multi-hop reasoning) and width (information gathering) to efficiently converge on a decisive verdict and achieve high alignment with human preferences.

---

[AGENTGUARDIAN: Learning Access Control Policies to Govern AI Agent Behavior](http://arxiv.org/abs/2601.10440)

- AgentGuardian: introduces a novel security framework that governs and protects AI agent operations by enforcing context-aware access-control policies, including Monitoring Tool (Collects execution traces), Policy Generator (Transforms traces into policies), and Policy Enforcer (Validates tool invocations dynamically).
- The framework learns legitimate agent behaviors and input patterns during a controlled staging phase, deriving adaptive policies that regulate tool calls guided by real-time input context and Control Flow Graph (CFG) dependencies.
- The CFG-based governance mechanism mitigates hallucination-driven errors and orchestration-level malfunctions by ensuring agent behavior remains predictable and aligned with predefined safe execution trajectories.

---

[Reinforcement Learning with Multi-Step Lookahead Information Via Adaptive Batching](http://arxiv.org/abs/2601.10418)

- ABPs (Adaptive Batching Policies): introduces a family of tractable policies for tabular reinforcement learning problems that utilize multi-step lookahead information by adaptively selecting the lookahead range (batching horizon) based on the current state and timestep.
- The framework derives optimal Bellman equations for ABPs and proposes the Adaptive Lookahead Upper Confidence Bound (AL-UCB) algorithm to learn the optimal policy in unknown environments with near-tight regret bounds.
- ABPs overcome limitations of fixed batching policies and model predictive control (MPC) schemes, which are shown to be exponentially suboptimal or fail critically in environments where lookahead information timing is crucial.

---

[CS-GBA: A CRITICAL SAMPLE-BASED GRADIENT-GUIDED BACKDOOR ATTACK FOR OFFLINE REINFORCEMENT LEARNING](http://arxiv.org/abs/2601.10407)

- CS-GBA (Critical Sample-based Gradient-guided Backdoor Attack): introduces a novel backdoor attack framework for Offline RL, utilizing Efficiency-Driven Sample Selection, Stealthy Trigger Injection, and Defense-Penetrating Action Generation to bypass conservative defenses.
- The framework achieves high efficiency by using TD-Error to prioritize poisoning the most influential transitions, requiring only a minimal 5% poisoning budget while maintaining clean environment performance.
- Stealthiness is ensured by the Correlation-Breaking Trigger, which disrupts feature correlations, while the Gradient-Guided Action Generation finds low-value actions within the data manifold to penetrate Out-of-Distribution defenses.

---

[Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering](http://arxiv.org/abs/2601.10402)

- ML-Master 2.0: introduces an autonomous agent for ultra-long-horizon Machine Learning Engineering (MLE) via Hierarchical Cognitive Caching (HCC), which uses L1 (Evolving Experience), L2 (Refined Knowledge), and L3 (Prior Wisdom) caches managed by Context Migration (CM).
- HCC structurally separates transient execution details from stable knowledge and reusable wisdom, enabling the agent to sustain strategic coherence over extended experimental cycles without context saturation.
- CM dynamically governs information flow through context prefetching, hit policies, and promotion operators ($P_1$ and $P_2$) that distill raw execution traces into compact, persistent cognitive representations.

---

[Inverse Learning in 2 × 2 Games: From Synthetic Interactions to Traffic Simulation](http://arxiv.org/abs/2601.10367)

- IGTL (Inverse Game-Theoretic Learning): introduces two novel inverse learning formulations, CE-ML and LBR-ML, to infer latent utility parameters from observed behavior in 2x2 normal-form games.
- CE-ML provides accurate and efficient parameter recovery when data aligns with a static correlated-equilibrium structure, while LBR-ML captures adaptive interactions under bounded rationality.
- The methods are evaluated on synthetic "chicken-dare" games and a SUMO-based traffic simulation scenario, demonstrating trade-offs between interpretability and behavioral expressiveness.

---

[Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text](http://arxiv.org/abs/2601.10355)

- GEM (Generation and Extraction of Multi-turn tool-use trajectory from text corpora): introduces a novel text-based paradigm for synthesizing multi-turn tool-use trajectories, utilizing Selection (Text filtering), Extraction (Workflow & tool definition), Generation (Initial trajectory creation), and Refinement (Complexity enhancement/Validation) stages.
- The pipeline transforms raw text corpora, which contain implicit human problem-solving experiences, into structured, high-quality agentic training data, demonstrating superior generalization capabilities.
- To enable cost-effective scaling, a specialized Trajectory Synthesizer is developed via supervised fine-tuning to match the quality of the full multi-stage pipeline while significantly reducing inference costs.

---

[SuS: Strategy-aware Surprise for Intrinsic Exploration](http://arxiv.org/abs/2601.10349)

- SuS (Strategy-aware Surprise): introduces a novel intrinsic motivation framework that uses pre-post prediction mismatch in strategy space for exploration in reinforcement learning, incorporating Strategy Encoder, World Model, Strategy Stability, Strategy Surprise, Intrinsic Reward, Policy, and Base LLM.
- The approach computes intrinsic rewards based on two complementary components, Strategy Stability (SS) and Strategy Surprise (SuS), which measure behavioral consistency and unexpected strategic shifts, respectively.
- SuS significantly outperforms baseline methods on mathematical reasoning tasks using LLMs, achieving substantial relative improvements in Pass@1 and Pass@5 accuracy by maintaining higher strategy diversity.

---

[Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale](http://arxiv.org/abs/2601.10338)

- SKILLSCAN: introduces a multi-stage detection framework for the first large-scale empirical security analysis of 31,132 agent skills, integrating static analysis, LLM-GUARD, and Claude 3.5 Sonnet classification.
- The analysis reveals pervasive security risks, finding that 26.1% of skills contain at least one vulnerability spanning 14 distinct patterns across four categories: prompt injection, data exfiltration, privilege escalation, and supply chain risks.
- Skills bundling executable scripts are 2.12 times more likely to be vulnerable than instruction-only skills, underscoring the urgent need for capability-based permission systems and mandatory security vetting.

---

[Multipath Routing for Multi-Hop UAV Networks](http://arxiv.org/abs/2601.10299)

- IPPO-DM (Independent Proximal Policy Optimization with Dirichlet Modeling): introduces a novel traffic-adaptive multipath routing method for multi-hop UAV networks, enabling dynamic traffic splitting across multiple next-hop neighbors to maximize the on-time packet delivery ratio.
- The sequential decision-making problem is modeled as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) and solved using the IPPO-DM algorithm, which leverages MARL for robust distributed policy learning.
- The framework integrates a Dirichlet policy head to inherently model continuous and feasible traffic splitting ratios on the probability simplex, alongside a traffic-aware resampling mechanism to adaptively adjust forwarding paths based on real-time queue states.

---

[coTherapist: A Behavior-Aligned Small Language Model to Support Mental Healthcare Experts](http://arxiv.org/abs/2601.10246)

- coTherapist: introduces a unified framework utilizing a small language model (LLaMA 3.2-1B-Instruct) to emulate therapeutic competencies through domain-specific fine-tuning, Retrieval-Augmented Generation (RAG), and a multi-stage agentic reasoning pipeline.
- The system employs a five-step agentic pipeline, including Planner, Retriever, Reasoner (private Chain-of-Thought), and a Critic/Self-Refiner module, to ensure clinically grounded, accurate, and safe responses.
- The framework is aligned using the novel T-BARS (Therapist Behavior Rating Scale) rubric and extensive Psychotherapy Knowledge Corpus (PsyKC) to achieve high empathy and therapist-consistent behavioral traits suitable for expert assistance.

---

[STEAMROLLER: A Multi-Agent System for Inclusive Automatic Speech Recognition for People Who Stutter](http://arxiv.org/abs/2601.10223)

- STEAMROLLER (Multi-Agent System for Inclusive Automatic Speech Recognition for People Who Stutter): introduces a real-time, three-stage pipeline that transforms stuttered audio into fluent speech and repaired text using ASR, a collaborative multi-agent LLM framework, and voice cloning.
- The core innovation is the multi-agent framework, featuring Repair Agents for segment correction and a Master Agent for iterative evaluation and coordination, ensuring semantic fidelity and low latency.
- The system serves as both an assistive tool for People Who Stutter (PWS) and a framework for dataset enhancement, improving ASR accuracy across varying stuttering severities.

---

[Agentic Pipelines in Embedded Software Engineering: Emerging Practices and Challenges](http://arxiv.org/abs/2601.10220)

- Agentic Pipelines: introduces a framework for integrating generative AI into embedded software engineering workflows, featuring multi-agentic CI/CD pipelines, AI-Friendly Artifacts, and Human-in-the-Loop Supervision.
- The framework addresses the unique challenges of Embedded Software Engineering (ESE), such as determinism and traceability, by implementing Compiler-in-the-Loop Feedback and formalizing Prompt Management and Traceability and Version Control.
- Sustainable adoption is supported by organizational practices including AI Training and Upskilling, Tool Standardization via the Model Context Protocol (MCP), and formal Governance and Compliance mechanisms.

---

[HUMANLLM: Benchmarking and Reinforcing LLM Anthropomorphism via Human Cognitive Patterns](http://arxiv.org/abs/2601.10198)

- HUMANLLM: introduces a framework for reinforcing LLM anthropomorphism by modeling psychological patterns as interacting causal forces, utilizing 244 patterns and 11,359 multi-pattern scenarios.
- The framework employs supervised fine-tuning on synthesized multi-turn conversations that integrate inner thoughts, actions, and dialogue, using base LLMs like Qwen3-8B/32B.
- Evaluation uses dual-level checklists (pattern-level and scenario-level) to assess both individual pattern fidelity (IPE) and emergent multi-pattern dynamics (MPD), achieving strong human alignment ($r=0.91$).

---

[Autonomous Quantum Simulation through Large Language Model Agents](http://arxiv.org/abs/2601.10194)

- Multi-Agent Quantum Simulation Architecture: introduces an autonomous system for tensor network simulations using LLM agents, combining multi-agent decomposition (Conductor, Strategist, Guide, Programmer, Executor, Aggregator, Validator, Visualizer) with in-context learning of curated documentation.
- The architecture achieves approximately 90% success rate on quantum simulation benchmarks by assigning specialized subtasks to dedicated agents, mitigating implementation errors and hallucinations common in single-agent setups.
- Systematic evaluation across three frontier LLMs (DeepSeek-V3.2, Gemini 2.5 Pro, Claude Opus 4.5) confirms that both in-context learning and the multi-agent structure are essential for reliable performance in complex scientific domains.

---

[ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack](http://arxiv.org/abs/2601.10173)

- ReasAlign: introduces a reasoning-enhanced internal guardrail for LLMs to defend against indirect prompt injection attacks by incorporating structured reasoning steps, a test-time scaling mechanism, and a preference-optimized judge model.
- The structured reasoning process, used for safety alignment, is divided into Problem Analysis, Reasoning (identifying conflicting instructions), and Final Answer Generation, ensuring focus remains on the original user intent.
- The approach leverages test-time scaling with a Logic Judge Model trained via DPO to score reasoning steps and select the optimal thought trajectory, achieving a superior balance between security and utility.

---

[Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation](http://arxiv.org/abs/2601.10911)

- CRL (Curriculum Reinforcement Learning) framework: introduces a novel framework for autonomous and sustainable marine vessel navigation, integrating a realistic data-driven simulation environment and a machine learning-based fuel consumption prediction module.
- The framework utilizes a policy-based CRL agent built upon PPO, guided by a multi-objective reward function optimizing safety, emissions, timeliness, and task completion in continuous action spaces.
- The realistic simulation environment is constructed using real-world AIS data enhanced by a Diffusion Model to simulate dynamic maritime traffic conditions, providing high-fidelity training grounds.

---

[SemTaint (Multi-Agent Taint Specification Extraction for Vulnerability Detection):](http://arxiv.org/abs/2601.10865)

- SemTaint (Multi-Agent Taint Specification Extraction for Vulnerability Detection): introduces a static-led, LLM-augmented vulnerability analysis tool for JavaScript that strategically combines LLM semantic understanding with traditional static program analysis (CodeQL) to extract precise taint specifications.
- The system employs three specialized LLM agents—Source/Sink, CallGraph, and Flow Summary—to iteratively refine analysis artifacts like sources, sinks, call edges, and flow summaries, addressing limitations in static analysis of dynamic JavaScript.
- The Taint-Informed Callee Resolution (TICR) algorithm focuses LLM inference exclusively on unresolved calls lying on potential vulnerability paths, achieving 65.43% recall while reducing LLM invocations by 94.5%.

---

[Cooperative UAVs for Remote Data Collection under Limited Communications: An Asynchronous Multiagent Learning Framework](http://arxiv.org/abs/2601.10849)

- AQMIX (Asynchronous-QMIX): introduces a solution for cooperative UAV data collection by formulating trajectory optimization as a Dec-POSMDP and proposing the AQMIX algorithm, which uses agent networks, a mixing network, a hypernetwork, and a state downsampling layer.
- The framework addresses the joint optimization of UAV trajectories and bandwidth allocation under limited communication and asynchronous decision-making constraints to maximize overall energy efficiency.
- AQMIX extends the synchronous QMIX algorithm by enabling decentralized agents to calculate actions independently based on local greedy policies, maintaining performance stability in asynchronous environments.

---

[Reasoning Models Generate Societies of Thought](http://arxiv.org/abs/2601.10825)

- SoT (Societies of Thought): introduces a framework showing that enhanced reasoning in LLMs emerges from the implicit simulation of multi-agent-like interactions characterized by distinct cognitive perspectives and domain expertise.
- Reasoning models like DeepSeek-R1 and QwQ-32B exhibit frequent conversational behaviors (question-answering, perspective shifts, conflict, reconciliation) and reciprocal socio-emotional roles, especially when tackling complex problems.
- Mechanistic interpretability experiments confirm that steering conversational features in the model's activation space causally increases reasoning accuracy by facilitating effective cognitive strategies like verification and backtracking.

---

[Bridging Psychological Safety and Skill Guidance: An Adaptive Robotic Interview Coach](http://arxiv.org/abs/2601.10824)

- ASE (Adaptive Scaffolding Ecosystem): introduces a conceptual framework for robotic interview coaching that dynamically balances affective support and instructional challenge, mediated by user agency.
- The system utilizes a three-phase iterative design process, culminating in an Agency-Driven Interaction Layer that resolves the "Scaffolding Paradox" by restoring user control over feedback timing.
- The framework successfully reduces acute social and communication anxiety while maintaining the psychological safety and therapeutic bond established by Person-Centered Therapy (PCT).

---

[Towards Reliable ML Feature Engineering via Planning in Constrained-Topology of LLM Agents](http://arxiv.org/abs/2601.10820)

- P-CTMA (Planner-guided Constrained-Topology Multi-Agent): introduces a planner-guided, constrained-topology multi-agent framework for reliable ML feature engineering, featuring a Planner, Actors, Executor, Environment, Short-term Memory, and Human-in-the-Loop Tool.
- The LLM-powered Planner orchestrates the multi-turn workflow by dynamically selecting Actors based on a constrained topology graph and generating context-aware prompts, leveraging the Environment's graph representation.
- The framework achieves robustness by using downstream failures (e.g., PySpark errors or failing tests) to retroactively correct upstream artifacts and supports human intervention at critical steps.

---

[Optimal Liquidation of Perpetual Contracts](http://arxiv.org/abs/2601.10812)

- Optimal Liquidation Strategy (OLS): introduces a stochastic control model for an agent liquidating a perpetual contract position, managing transaction costs, inventory risk, and funding rate payments.
- The model incorporates Price Impact Dynamics (temporary and permanent) and a Funding Rate Mechanism based on the contract payoff function $\psi$ and funding parameter $\beta$.
- The paper derives the Optimal Trading Strategy $v^*$ in closed form for the identity payoff function and provides asymptotic approximations for arbitrary payoff functions when funding rate or time horizon is small.

---

[A Concise Agent is Less Expert: Revealing Side Effects of Using Style Features on Conversational Agents](http://arxiv.org/abs/2601.10809)

- CASSE (Conversational Agent Stylistic Side Effects) Study Framework: introduces the first systematic study of cross-feature stylistic side effects in LLM conversational agents, using Survey and Feature Extraction, Message Generation and Analysis, LLM-as-a-Judge Protocol, Side Effect Identification, Prompt Intervention, Steering Intervention, and the CASSE Dataset.
- The study reveals that style features are deeply entangled, showing consistent side effects, such as prompting for conciseness significantly reducing perceived expertise (25.6% win rate on Expert) across Llama3, Qwen3-8B, and GPT-5-mini models.
- Mitigation strategies, including prompt concatenation and activation steering, struggle to resolve these trade-offs without sacrificing the primary intended style, highlighting the need for multi-objective approaches to safe stylistic steering.

---

[LogicLens: Leveraging Semantic Code Graph to explore Multi Repository large systems](http://arxiv.org/abs/2601.10773)

- LogicLens: introduces a reactive conversational agent that assists developers in exploring complex software systems through a semantic multi-repository graph built by combining syntactic analysis with LLM-based semantic enrichment.
- The system leverages a ReAct Agent architecture combined with GraphRAG techniques, utilizing specialized tools to retrieve relevant subgraphs for answering technical and domain-specific queries.
- This approach enables system-level exploration and cross-cutting reasoning, supporting emergent behaviors like impact analysis and symptom-based debugging across large codebases.

---

[CTD: COMPOSITION THROUGH DECOMPOSITION IN EMERGENT COMMUNICATION](http://arxiv.org/abs/2601.10169)

- CTD (Composition through Decomposition): introduces a dual-phase training approach where emergent communication agents first learn a discrete concept codebook via decomposition and then use it for compositional phrase generation.
- The DECOMPOSE step trains agents in a multi-target coordination game to break down complex objects into basic concepts stored in the Discrete Codebook (CB).
- The subsequent COMPOSE step leverages the learned codebook to describe novel, unseen images by systematically combining these basic concepts into complex phrases, often achieving zero-shot generalization.

---

[AWED-FiNER: Agents, Web applications, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers](http://arxiv.org/abs/2601.10161)

- AWED-FiNER: introduces an open-source ecosystem for Fine-grained Named Entity Recognition (FgNER) across 36 languages, combining an Agentic Tool (smolagents paradigm), a centralized Hugging Face Router API (centralized routing logic), a Serverless Inference Architecture (dynamic model loading), a Multilingual Expert Model Collection (49 fine-tuned models), and Interactive Web Applications (Gradio user interface).
- The Agentic Tool enables multilingual text routing to specialized expert models for rapid FgNER annotations, supporting seamless "one-line" integration into modern AI workflows.
- The system addresses linguistic disparity by providing resources for 36 languages, including vulnerable low-resource languages, serving 6.6 billion speakers globally.

---

[ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback](http://arxiv.org/abs/2601.10156)

- ToolSafe: introduces a proactive step-level guardrail and feedback framework, TS-Flow, featuring the TS-Guard model and evaluated on the TS-Bench benchmark, to enhance tool invocation safety in LLM-based agents.
- TS-Guard proactively detects unsafe tool invocation actions before execution by reasoning over the interaction history and assessing request harmfulness and action-attack correlations.
- TS-Flow reduces harmful tool invocations by 65% on average and improves benign task completion by approximately 10% under prompt injection attacks by guiding agents toward safety-aware tool use.

---

[Step-by-Step Causality: Transparent Causal Discovery with Multi-Agent Tree-Query and Adversarial Confidence Estimation](http://arxiv.org/abs/2601.10137)

- Tree-Query: introduces a tree-structured, multi-expert LLM framework that reduces pairwise causal discovery to a fixed sequence of queries, answered by the Multi-Expert System and scored by the Adversarial Confidence Estimator.
- The framework uses the Tree-Query-Checks component to execute four primitive Query Types (backdoor path, independence, latent confounder, causal direction), and the Decision Rule aggregates the LLM Experts' confidence-scored judgments into a final causal relation.
- The ACE component explicitly tests the robustness of the MES's binary judgments against Adversarial Agents, ensuring transparent, stepwise, and confidence-aware causal reasoning for asymptotic identifiability.

---

[Fairness Driven Multi-Agent Path Finding Problem](http://arxiv.org/abs/2601.10123)

- Fair-ICTS/Fair-CBS/Mechanism Design: studies the Multi-Agent Path Finding (MAPF) problem under fairness constraints, proposing Fair-ICTS and Fair-CBS algorithms for non-rational agents, and a Mechanism Design approach for rational agents.
- The objective is to find non-conflicting paths that maximize social welfare while satisfying three fairness constraints: $\epsilon$-envy freeness, max-min fairness, and proportional fairness.
- The proposed single-parameter domain mechanism for rational agents, where agents report their private per-step cost, is proven to be Dominant Strategy Incentive Compatible (DSIC) and Individually Rational (IR).

---

[Role-Playing Agents Driven by Large Language Models: Current Status, Challenges, and Future Trends](http://arxiv.org/abs/2601.10122)

- RPLAs (Role-Playing Language Agents): systematically reviews the current status, challenges, and future trends of LLM-driven agents, focusing on the evolution from linguistic imitation to cognitive simulation centered on personality modeling and memory mechanisms.
- The core technologies supporting high-quality role-playing include psychological scale-driven character modeling, memory-augmented prompting mechanisms, and motivation-situation-based behavioral decision control.
- The review outlines future directions such as dynamic personality evolution, multi-agent collaborative narrative, multimodal immersive interaction, and integration with cognitive neuroscience.

---

[TopoDIM: One-shot Topology Generation of Diverse Interaction Modes for Multi-Agent Systems](http://arxiv.org/abs/2601.10120)

- TopoDIM (One-shot Topology Generation with Diverse Interaction Modes): introduces a decentralized framework for multi-agent systems that generates a heterogeneous communication topology in a single step, leveraging diverse interaction modes (Conditioned, Feedback, Debate) to enhance efficiency and performance.
- The architecture integrates a Relational Encoder and an Autoregressive Sampling Decoder to construct the graph, optimized via reinforcement learning and decentralized through Policy Distillation to local policy networks.
- By replacing iterative multi-round dialogues with one-shot topology generation and adaptive sparsification, the framework significantly reduces token consumption while maintaining high task performance.

---

[Repository Intelligence Graph: Deterministic Architectural Map for LLM Code Assistants](http://arxiv.org/abs/2601.10112)

- SPADE and RIG: introduces a deterministic extractor (SPADE) that constructs the Repository Intelligence Graph (RIG), an evidence-backed architectural map of repository build and test structure, exposed as LLM-friendly JSON.
- RIG models the repository architecture using nodes like components, aggregators, runners, tests, external packages, and package managers, connected by explicit dependency and coverage edges.
- Providing RIG to commercial LLM agents significantly improved mean accuracy by 12.2% and reduced completion time by 53.9%, especially benefiting complex and multi-lingual repositories.

---

[FlowAct-R1: Towards Interactive Humanoid Video Generation](http://arxiv.org/abs/2601.10103)

- FlowAct-R1: introduces a real-time interactive humanoid video generation framework built on the MMDiT (multimodal backbone) architecture, utilizing VAE Encoder (input compression), Whisper (audio processing), Text Encoder (prompt encoding), Chunkwise Diffusion Forcing (streaming synthesis), Structured Memory Bank (temporal consistency), and Multi-Stage Diffusion Distillation (fast inference).
- The framework achieves stable 25fps 480p video generation with a low time-to-first-frame (TTFF) of 1.5 seconds, enabled by efficient distillation techniques that reduce the denoising process to only 3 NFEs.
- The approach ensures long-term temporal consistency and mitigates error accumulation using a novel self-forcing variant, a Memory Refinement strategy, and an MLLM for multimodal action planning.

---

[When Personas Override Payoffs: Role Identity Bias in Multi-Agent LLM Decision-Making](http://arxiv.org/abs/2601.10102)

- MASDF (Multi-Agent Strategic Decision Framework): introduces a systematic evaluation of four LLM architectures across a 2x2 factorial design manipulating role personas and payoff visibility in a 4-agent strategic game environment.
- The research demonstrates that role identity bias fundamentally alters strategic reasoning, causing LLM agents to prioritize persona-aligned actions over explicit payoff optimization, resulting in near-zero Nash equilibrium rates in payoff-optimal Tragedy scenarios.
- Strategic reasoning capabilities, specifically achieving payoff-optimal Nash equilibrium, require the dual condition of removing personas and providing explicit (visible) payoff matrices, a capability observed only in Qwen models.

---

[State of AI: An Empirical 100 Trillion Token Study with OpenRouter](http://arxiv.org/abs/2601.10088)

- OpenRouter Study: introduces an empirical analysis of over 100 trillion tokens of real-world LLM interactions using the OpenRouter Platform, categorized by the GoogleTagClassifier, to understand usage dynamics across models and tasks.
- The study identifies a structural shift toward agentic inference, where reasoning-optimized models and tool-calling are increasingly used for complex, multi-step workflows, particularly in programming tasks.
- Analysis of retention patterns reveals the Cinderella "Glass Slipper" effect, where early cohorts achieving a persistent workload-model fit become foundational users, resisting substitution by newer models.

---

[CALM-IT: Generating Realistic Long-Form Motivational Interviewing Dialogues with Dual-Actor Conversational Dynamics Tracking](http://arxiv.org/abs/2601.10085)

- CALM-IT: introduces a framework for generating and evaluating long-form Motivational Interviewing dialogues by explicitly modeling dual-actor conversational dynamics.
- The system represents therapist-client interaction as a bidirectional state-space process, where THERAAGENT and CLIENTAGENT continuously update inferred alignment, mental states, and short-term goals.
- By tracking conversational state outside the surface text, the framework achieves greater stability and effectiveness in long-horizon interactions compared to baseline LLM simulators.

---

[Deriving Character Logic from Storyline as Codified Decision Trees](http://arxiv.org/abs/2601.10080)

- CDT (Codified Decision Trees): introduces a data-driven framework that induces executable, interpretable character behavioral profiles from large-scale narrative data as a tree of conditional rules.
- The framework uses an LLM-based Rule Mining System and Text Embeddings to hypothesize and validate scene-action triggers, which recursively grow the CDT structure.
- At inference, a novel scene traverses the CDT via a Discriminator, accumulating situation-specific grounding statements used by the LLM RP Policy for consistent action generation.

---

[Collective behavior based on agent-environment interactions](http://arxiv.org/abs/2601.10046)

- AEFM (Agent-Environment Feedback Model): introduces a model of active particles interacting solely through a dynamic resource landscape, leading to emergent collective behaviors, utilizing Agent Dynamics (persistent random walk), Chemotaxis Mechanism (gradient alignment force), Resource Landscape (dynamic nutrient depot), Food Field (logistic regrowth, depletion), Population Dynamics (birth-death processes), and Continuum PDE Model (hydrodynamic equations).
- The model extends previous resource-dependent frameworks by incorporating chemotaxis, allowing individuals to bias their persistent random walk toward nutrient-rich areas.
- Collective organization phases, including disordered gas-like states, polar traveling waves, and nematic clusters, emerge from the interplay between chemotactic sensitivity and angular noise, without requiring direct agent-to-agent alignment forces.

---

[PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization](http://arxiv.org/abs/2601.10029)

- PaperScout (Autonomous Agent for Academic Paper Search): reformulates academic paper search as a sequential decision-making process, dynamically controlling Search and Expand tools based on accumulated retrieval context.
- The agent is trained using Proximal Sequence Policy Optimization (PSPO), a process-aware, sequence-level policy optimization method that aligns optimization granularity with multi-turn agent interactions to ensure stable training.
- The system maintains a latent Paper Pool state, summarized into an Observation (dual-list view of expanded/unexpanded papers) to guide the LLM-based policy's iterative tool invocation.

---

[Structured Personality Control and Adaptation for LLM Agents](http://arxiv.org/abs/2601.10025)

- JPAF (Jungian Personality Adaptation Framework): introduces a framework modeling LLM personality via Jungian psychological types, integrating dominant-auxiliary coordination, reinforcement-compensation, and reflection mechanisms.
- The framework enables LLM agents to maintain coherent core personality traits while dynamically adapting to context and gradually evolving their underlying psychological structure.
- JPAF uses BaseWeights to formalize the hierarchical influence of eight psychological types, allowing MBTI profiles to emerge dynamically and supporting flexible personality representation.

---

[PID-GUIDED PARTIAL ALIGNMENT FOR MULTIMODAL DECENTRALIZED FEDERATED LEARNING](http://arxiv.org/abs/2601.10012)

- PARSE (PID-Guided Partial Alignment for Multimodal Decentralized Federated Learning): introduces a multimodal DFL framework that operationalizes Partial Information Decomposition (PID) via Feature Fission and P2P Knowledge Sharing with Partial Alignment.
- The framework decomposes each agent's latent representation into redundant, unique, and synergistic slices, allowing agents to update only slices corresponding to available modalities, thereby preventing interference.
- Partial alignment restricts P2P knowledge sharing to align-able slices over per-modality subgraphs, resolving uni-/multimodal gradient conflicts without requiring a central server or gradient surgery.

---

[ALD optimization using reasoning LLMs: Performance of AI agents based on reasoning language models on ALD process optimization tasks](http://arxiv.org/abs/2601.09980)

- AI Agent for ALD optimization: introduces an autonomous system for Atomic Layer Deposition (ALD) process optimization, comprising a Logic component (Manages iterative flow), an AI component (Determines strategy/requests), a Reasoning Model (LLM) (Generates open response/structured output), an ALD Reactor (Simulated) (Provides GPC feedback), and Initial guess/background information (Starting point/prior knowledge).
- The agent iteratively interacts with the simulated ALD reactor, using a two-step process where the reasoning LLM first generates an open response detailing its logic, which is then transformed into a structured output (JSON) for procedural execution.
- Agents based on reasoning LLMs (like o3 and GPT5) successfully optimize ALD dose times, achieving performance comparable to human experts in terms of required experimental samples (10-15), although exhibiting significant run-to-run variability.

---

[A Control Theoretic Approach to Decentralized AI Economy Stabilization via Dynamic Buyback-and-Burn Mechanisms](http://arxiv.org/abs/2601.09961)

- DCBM (Dynamic-Control Buyback Mechanism): introduces a formalized control-theoretic framework utilizing a PID Controller with Clamped Integration and a Solvency Actuator to regulate the decentralized AI token economy as a dynamical system.
- Extensive agent-based simulations demonstrate that DCBM fundamentally outperforms static baselines, reducing token price volatility by approximately 66% and lowering operator churn from 19.5% to 8.1%.
- By converting tokenomics from static rules into continuous, structurally constrained control loops, the mechanism ensures asymptotic solvency and enhanced resilience against adversarial game-theoretic attacks.

---

#### 14th January 2026


[LLMs can Compress LLMs: Adaptive Pruning by Agents](http://arxiv.org/abs/2601.09694)

- LLM-Guided Pruning (LGP): introduces agent-guided pruning, where an LLM Agent (adaptive pruning oracle) iteratively selects layers of the Target Model (LLM being compressed) based on Layer Sensitivity Profiling (normalized statistical metrics) and refines its strategy using a Self-Reflection Mechanism (iterative feedback loop) and a Checkpoint Rollback Mechanism (model quality safety net).
- The framework constructs layer-wise sensitivity profiles by combining Wanda-inspired weight-activation metrics and gradient importance scores, normalized as z-scores for model-agnostic comparison across heterogeneous layers.
- This adaptive approach achieves substantial improvements over structured pruning baselines, particularly in preserving factual knowledge retention and reducing perplexity degradation with low rollback rates.

---

[DeepResearchEval: An Automated Framework for Deep Research Task Construction and Agentic Evaluation](http://arxiv.org/abs/2601.09688)

- DeepResearchEval: introduces an automated framework for deep research task construction and agentic evaluation, including Persona Synthesis (Generates realistic user profiles) and Active Fact-Checking (Verifies factual consistency via external retrieval).
- The framework's task construction pipeline uses LLMs to generate complex, persona-driven research tasks, filtered by Task Qualification and Search Necessity stages to ensure multi-source evidence integration.
- The agentic evaluation pipeline employs Adaptive Point-wise Quality Evaluation to dynamically derive task-specific dimensions and criteria, enabling fine-grained, context-aware scoring of deep research reports.

---

[Automating Supply Chain Disruption Monitoring via an Agentic AI Approach](http://arxiv.org/abs/2601.09680)

- ASCDMF (Agentic Supply Chain Disruption Monitoring Framework): introduces a minimally supervised agentic AI framework that autonomously monitors, analyzes, and responds to supply chain disruptions across extended multi-tier networks, utilizing seven specialized agents, a Knowledge Graph, LLM Orchestration, Deterministic Tools, and Human-in-the-Loop oversight.
- The framework transforms unstructured disruption signals from news articles into actionable intelligence by mapping affected entities onto a multi-tier supplier network up to Tier-4, quantifying risk, and generating mitigation strategies.
- By integrating LLM reasoning for text processing with deterministic graph tools for risk calculation, the system reduces disruption response time from days to a mean of 3.83 minutes, enabling proactive resilience.

---

[Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning](http://arxiv.org/abs/2601.09667)

- MATTRL (Multi-Agent Test-Time Reinforcement Learning): introduces a framework that injects structured textual experience from the Test-Time Experience Pool (Structured textual experiences) into multi-agent deliberation using Specialist Agents (Multi-turn deliberation, opinion updates) and a Coordinator Agent (Orchestrates team, synthesizes report) for robust reasoning.
- The framework constructs the experience pool by scoring high-value utterances using an LLM Judge (Scores utterance quality) and a Credit Assignment Mechanism (Attributes team returns), then distilling them via an LLM Summarizer (Extracts reusable experience) for subsequent retrieval.
- By conditioning agent behavior with retrieved structured experience at inference time, MATTRL enables rapid, distribution-shift-robust adaptation without requiring policy weight updates.

---

[PersonalAlign: Hierarchical Implicit Intent Alignment for Personalized GUI Agent with Long-Term User-Centric Records](http://arxiv.org/abs/2601.09636)

- HIM-Agent (Hierarchical Intent Memory Agent): introduces a memory framework that leverages long-term user records to support hierarchical implicit intent alignment via Streaming Aggregation, Execution-based Preference, and State-based Routine Filters.
- The agent resolves vague instructions by inferring Preference Intent from past selections and anticipates Routine Intent based on current user state and repetitive patterns for proactive assistance.
- The approach is evaluated on AndroidIntent, a new user-centric GUI benchmark annotated with user preferences and routines from long-term records to assess implicit intent alignment capabilities.

---

[LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach](http://arxiv.org/abs/2601.09635)

- LEAN-LLM-OPT: introduces a novel agentic framework for large-scale optimization model auto-formulation, utilizing a Classification Agent, a Workflow Generation Agent, and a Model Generation Agent guided by a Workflow (Structured step-by-step guide) and supported by RAG tools (FileQA/CSVQA) for data handling.
- The framework addresses challenges in long input processing and reasoning by decomposing complex modeling tasks into structured sub-tasks, enabling LLMs to focus on critical reasoning while offloading mechanical data handling to customized tools.
- The approach achieves high accuracy on large-scale optimization benchmarks (Large-Scale-OR) with minimal training data (96 entries), demonstrating resource efficiency and strong generalization capabilities compared to fine-tuning methods.

---

[LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation](http://arxiv.org/abs/2601.09631)

- Hybrid Phonological Filtering System: introduces a hybrid neuro-symbolic architecture combining LLMs with deterministic phonological algorithms for accurate Greek poetry rhyme identification and generation.
- The system employs an Agentic Generation Pipeline featuring a Generate-Verify-Refine loop, which uses the Phonological Engine to verify LLM outputs and provide corrective feedback.
- This hybrid approach dramatically improves generation validity from under 4% (pure LLM) to 73.1% by offloading strict phonological constraints to the symbolic module.

---

[The Promptware Kill Chain: How Prompt Injections Gradually Evolved Into a Multi-Step Malware](http://arxiv.org/abs/2601.09625)

- The Promptware Kill Chain (PKC): introduces a five-step model for analyzing promptware threats, defining them as multi-step malware campaigns targeting LLM-based applications.
- PKC maps LLM attacks to traditional kill chain stages: Initial Access, Privilege Escalation, Persistence, Lateral Movement, and Actions on Objective.
- This framework provides a structured methodology for threat modeling and establishes a common vocabulary for AI safety and cybersecurity researchers.

---

[Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs](http://arxiv.org/abs/2601.09527)

- Consumer Blackwell GPU LLM Inference Stack (CBGLIS): introduces a systematic evaluation of NVIDIA Blackwell consumer GPUs (RTX 5060 Ti, 5070 Ti, 5090) for cost-effective, private LLM inference deployment in SMEs, using vLLM, AIPerf, and various quantization formats (NVFP4, W4A16, MXFP4) across three realistic workloads.
- The evaluation finds that NVFP4 quantization offers the best performance-efficiency trade-off, providing 1.6x throughput over BF16 with minimal quality loss and 41% energy reduction.
- Self-hosted inference on consumer GPUs achieves costs 40-200x lower than budget-tier cloud APIs, with hardware breaking even in under four months at moderate usage (30M tokens/day).

---

[SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams](http://arxiv.org/abs/2601.09515)

- SERM (Self-Evolving Relevance Model): introduces a self-evolving relevance model approach comprising a Multi-Agent Sample Miner and a Multi-Agent Relevance Annotator to enable continuous adaptation to massive query streams.
- The Multi-Agent Sample Miner identifies informative query-document pairs by monitoring incoming queries and detecting distributional shifts using environmental and intrinsic feedback agents.
- The Multi-Agent Relevance Annotator generates reliable labels and coherent rationales for these samples via a two-level agreement framework leveraging multiple LLMs and external knowledge.

---

[What Do LLM Agents Know About Their World? Task2Quiz: A Paradigm for Studying Environment Understanding](http://arxiv.org/abs/2601.09503)

- Task-to-Quiz (T2Q): introduces T2QBench, a deterministic and automated evaluation paradigm designed to decouple LLM agent task execution ("doing") from grounded world-state understanding ("knowing").
- The paradigm operates in two stages: coverage-oriented task generation to drive exploration, followed by a multi-dimensional quiz evaluation using trajectory-conditioned prerequisites and environment metadata.
- Empirical results using T2QBench reveal that task success is a poor proxy for environment understanding, and insufficient proactive exploration is the dominant bottleneck for generalization.

---

[EvoFSM: Controllable Self-Evolution for Deep Research with Finite State Machines](http://arxiv.org/abs/2601.09465)

- EvoFSM (Controllable Self-Evolution for Deep Research with Finite State Machines): introduces a structured self-evolving framework that models deep research as an FSM (Models research process) to achieve adaptability and control by decoupling optimization into macroscopic Flow Logic (Macroscopic transition logic) and microscopic Skill (Microscopic node capabilities).
- The system uses a Critic Mechanism (Triggers self-evolution) to guide the refinement of the FSM topology and node instructions exclusively through a set of constrained Atomic Operations (Discrete FSM modifications), mitigating instability and instruction drift common in unconstrained LLM rewriting.
- EvoFSM integrates a Self-Evolving Memory (Accumulates task experience) mechanism, which distills successful strategies and failure patterns into an Experience Pool (Stores historical strategies) to enable continuous learning and warm-starting for future queries.

---

[Dissecting Judicial Reasoning in U.S. Copyright Damage Awards: A Discourse-Based LLM Analysis](http://arxiv.org/abs/2601.09459)

- Agentic LLM + ToD (Tree-of-Discourse): introduces a novel discourse-based LLM methodology integrating Rhetorical Structure Theory (RST) with an agentic workflow to extract and quantify reasoning patterns from judicial opinions.
- The framework uses a three-stage pipeline—Dataset Construction, Discourse Analysis, and Agentic Feature Extraction—to parse judicial opinions into hierarchical discourse structures (RST Trees).
- The discourse-augmented LLM analysis outperforms traditional methods by uncovering unquantified variations in factor weighting across circuits for U.S. copyright damage awards.

---

[MAD: Motion Appearance Decoupling for efficient Driving World Models](http://arxiv.org/abs/2601.09452)

- MAD (Motion-Appearance Decoupling): introduces an efficient adaptation framework that converts generalist video diffusion models into controllable driving world models by decoupling motion learning from appearance synthesis using a Base DiT Backbone, LoRA Adapters, VAE, and T5 Encoder, structured into a Motion Forecaster ($F_{\theta}$) generating an Intermediate Motion Representation ($M$) and an Appearance Synthesizer ($S_{\phi}$) rendering the final video, controlled via Ego-Motion Control, Object Movement Control, and a Text Prompt.
- The two-stage process mirrors a reasoning-rendering paradigm, allowing the MAD-LTX model to achieve state-of-the-art quality in driving world modeling while requiring less than 6% of the compute budget compared to prior SOTA models like VISTA and GEM.
- The methodology maximizes reuse of the base model's learned representations by projecting all control signals, including the intermediate poses, into the model's native visual latent space using its pre-trained VAE, enabling efficient transfer from generalist video generators.

---

[DepRadar: Agentic Coordination for Context-Aware Defect Impact Analysis in Deep Learning Libraries](http://arxiv.org/abs/2601.09440)

- DepRadar (Agentic Coordination for Context-Aware Defect Impact Analysis in Deep Learning Libraries): introduces an agent coordination framework for fine-grained defect and impact analysis in DL library updates, utilizing PR Miner Agent, Code Diff Analyzer Agent, Orchestrator Agent, Impact Analyzer Agent, Static Analysis, Domain Rules, and Progressive Context Augmentation.
- The framework coordinates four specialized LLM-based agents across three steps—semantics extraction, pattern generation, and impact analysis—to bridge the semantic gap between low-level library fixes and high-level user-facing behavior.
- DepRadar integrates static analysis and DL-specific domain rules to achieve high accuracy (95% F1 in defect modeling and 85% F1 in client impact analysis) and provide actionable diagnostics for downstream users.

---

[SC-MAS: Constructing Cost-Efficient Multi-Agent Systems with Edge-Level Heterogeneous Collaboration](http://arxiv.org/abs/2601.09434)

- SC-MAS (Social Capital-driven framework for constructing heterogeneous and cost-efficient Multi-Agent Systems): progressively constructs an executable MAS graph by sequentially employing the Node Selector, Edge Optimizer, and LLM Router modules.
- The framework models the Multi-Agent System as a directed graph where edges explicitly encode heterogeneous, pairwise collaboration strategies to balance task performance and computational cost.
- The system optimizes the selection of agents, collaboration structures, and LLM assignments based on a utility function penalized by execution cost, informed by Social Capital Theory.

---

[Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](http://arxiv.org/abs/2601.09413)

- Speech-Hands: introduces a voice-agentic framework that learns self-reflection by predicting an explicit Action Token (<internal>, <external>, <rewrite>) to govern its cognitive strategy for speech recognition and audio reasoning.
- The framework operates as a dynamic orchestrator, first generating an internal response ($H_{omni}$) and aggregating external hypotheses ($H_{ext}$), then deciding whether to trust internal perception, defer to external advice, or perform an Omni Rewrite.
- This self-reflection mechanism unifies perception and decision-making, offering a path toward more reliable audio intelligence by mitigating confusion caused by conflicting multimodal inputs.

---

[Preliminary Tests of the Anticipatory Classifier System with Hindsight Experience Replay](http://arxiv.org/abs/2601.09400)

- ACS2HER: introduces a novel integration of the Anticipatory Classifier System (ACS2) with the Hindsight Experience Replay (HER) mechanism to address sparse reward problems in reinforcement learning environments.
- The architecture uses a Replay Memory (RM) buffer and a goal-relabeling strategy triggered upon trial failure to densify the learning signal by treating visited states as virtual goals.
- Evaluation on Maze 6 and FrozenLake benchmarks demonstrates that the hybrid approach significantly accelerates knowledge acquisition, albeit at the cost of increased computational overhead and classifier numerosity.

---

[AI-NativeBench: An Open-Source White-Box Agentic Benchmark Suite for AI-Native Systems](http://arxiv.org/abs/2601.09393)

- AI-NativeBench: introduces the first application-centric, white-box agentic benchmark suite for AI-Native systems, utilizing MCP and A2A protocols, distributed tracing, and agentic services across various architectural variants.
- The trace-first evaluation methodology enables granular analysis of engineering characteristics, revealing a parameter paradox and pervasive LLM inference dominance over protocol overhead.
- The study identifies an expensive failure pattern where self-healing mechanisms act as cost multipliers, guiding the transition from measuring model capability to engineering reliable AI-Native systems.

---

[Long-term Task-oriented Agent: Proactive Long-term Intent Maintenance in Dynamic Environments](http://arxiv.org/abs/2601.09382)

- Hybrid-Triggered Proactive Framework: introduces a novel interaction paradigm for LLM agents to maintain long-term user intent and proactively engage users based on dynamic environmental updates.
- The framework uses a hybrid information retrieval architecture, decoupling agent reasoning from environmental monitoring via a Backend Monitor that sets reminders and generates internal observation messages.
- To validate this approach, the authors created ChronosBench, a dynamic benchmark, and an iterative data synthesis pipeline featuring multi-agent simulation for training complex, multi-turn dialogs.

---

[Monte-Carlo Tree Search with Neural Network Guidance for Lane-Free Autonomous Driving](http://arxiv.org/abs/2601.09353)

- NN-Guided MCTS: introduces a refined Markov Decision Process (MDP) and integrates a pre-trained Neural Network (NN) into the Monte-Carlo Tree Search (MCTS) selection phase using the PUCT formula for efficient lane-free autonomous driving planning.
- The NN is trained via offline self-play simulations using MCTS data and provides predictive guidance, significantly accelerating the tree search process and achieving high solution quality with fewer iterations.
- The approach operates in a challenging lane-free traffic environment simulated using SUMO, incorporating isotropic state information to enable crucial nudging behavior for collision avoidance and speed maintenance.

---

[Improving Implicit Hate Speech Detection via a Community-Driven Multi-Agent Framework](http://arxiv.org/abs/2601.09342)

- Community-Driven Multi-Agent Framework: introduces a contextualized implicit hate speech detection system using a central Moderator Agent and specialized Community Agents representing demographic groups, all operating within a Multi-Agent Consultative Workflow.
- The system leverages LLMs (Gemini-2.5-Flash) and a Knowledge Base (Wikipedia) via an Agent Construction Module to create identity-aware Community Agent profiles, enabling expert consultation on ambiguous cases.
- This consultative approach significantly improves classification fairness and accuracy on the challenging ToxiGen dataset by integrating socio-cultural context and mitigating bias inherent in standard prompting methods.

---


[MACRO-LLM: LLM-Empowered Multi-Agent Collaborative Reasoning under Spatiotemporal Partial Observability](http://arxiv.org/abs/2601.09295)

- MACRO-LLM (LLM-Empowered Multi-Agent Collaborative Reasoning): introduces a decentralized negotiation framework for multi-agent systems operating under spatiotemporal partial observability, featuring the CoProposer, Negotiator, and Introspector modules.
- The architecture explicitly decomposes coordination into spatial constraints, handled by the Negotiator using mean-field statistical aggregation, and temporal uncertainty, mitigated by the CoProposer's predictive rollouts.
- The Introspector ensures continuous adaptation by refining temporal strategies through semantic gradient descent based on historical experience and observed reward gaps.

---

[Policy-Based Reinforcement Learning with Action Masking for Dynamic Job Shop Scheduling under Uncertainty: Handling Random Arrivals and Machine Failures](http://arxiv.org/abs/2601.09293)

- PetriRL (Dynamic Scheduling Framework): integrates CPTN (Colored Timed Petri Nets) for environment modeling and MPPO RL (Maskable Proximal Policy Optimization) using an Actor-Critic architecture, leveraging dynamic Action Masking derived from Petri net Guard Functions to handle Random Job Arrivals and Machine Breakdowns.
- The CPTN environment simulates realistic industrial conditions by modeling dynamic job arrivals using a Gamma distribution and machine failures using a Weibull distribution to capture aging effects.
- The framework consistently outperforms traditional heuristic and rule-based approaches in makespan minimization, yielding a resilient, scalable, and explainable solution for real-time scheduling.

---

[Blue Teaming Function-Calling Agents](http://arxiv.org/abs/2601.09292)

- BTFE (Blue Teaming Function-Calling Evaluation): presents an empirical study assessing the robustness of four open-source function-calling LLMs against three adversarial attacks and eight defenses.
- The evaluation specifically targets Direct Prompt Injection, Simple Tool Poisoning, and Renaming Tool Poisoning attacks, measuring Accuracy and Attack Success Rate.
- Results indicate that function-calling models are not safe by default and that current defenses, including LLM-based guardians, suffer from practical limitations like high False Positive Rates.

---

[M³Searcher: Modular Multimodal Information Seeking Agency with Retrieval-Oriented Reasoning](http://arxiv.org/abs/2601.09278)

- M³Searcher (Modular Multimodal Information Seeking Agency): introduces a modular multimodal information-seeking agent that explicitly decouples information acquisition from answer derivation, optimized via a retrieval-oriented multi-objective reward.
- The agent, implemented as a lightweight trainable MLLM, iteratively interacts with multimodal tools (Text Search, Image Search, Answer Generator) to assemble comprehensive evidence before invoking the downstream Answer Generator.
- The decoupled architecture preserves the reasoning fidelity of a large-scale backbone while allowing targeted RL optimization of heterogeneous search scheduling and tool coordination using a multi-objective reward framework.

---

[Coordinated Pandemic Control with Large Language Model Agents as Policymaking Assistants](http://arxiv.org/abs/2601.09264)

- LLM Multi-Agent Policymaking Framework: introduces a decentralized multi-agent system where regional LLM agents coordinate policy decisions by integrating real-world data and epidemiological simulations.
- The framework utilizes a closed-loop simulation process, including an SEIQRD model, to enable agents to proactively explore counterfactual intervention scenarios and refine policies iteratively.
- Coordination is achieved through structured inter-agent communication, allowing LLM agents to reason over cross-regional interdependencies and manage local inbound mobility via strategies like Temporal Inflow Reallocation.

---

[MAXS: Meta-Adaptive Exploration with LLM Agents](http://arxiv.org/abs/2601.09259)

- MAXS (Meta-Adaptive Exploration with LLM Agents): introduces a meta-adaptive reasoning framework based on LLM Agents that flexibly integrates tool execution and reasoning planning using lookahead strategy, value estimation mechanism, and trajectory convergence mechanism.
- The lookahead strategy extends reasoning paths to estimate the advantage value of tool usage, mitigating locally myopic generation and trajectory instability.
- The composite value function combines advantage, step consistency variance, and inter-step trend slopes to jointly select stable, consistent, and high-value reasoning steps, while the convergence mechanism controls computational cost.

---

---

[Honesty-Aware Multi-Agent Framework for High-Fidelity Synthetic Data Generation in Digital Psychiatric Intake Doctor-Patient Interactions](http://arxiv.org/abs/2601.09216)

- MentalAED (Honesty-Aware Multi-Agent Framework): introduces a four-role multi-agent simulation framework for generating high-fidelity synthetic psychiatric intake data, including Patient, Assessor, Evaluator, and Diagnostician Agents, supported by CoT and Chat Modules, a Psychiatric Instruments Repository, and state tracking mechanisms.
- The system simulates a structured psychiatric intake process where the Patient Agent models bounded rationality and topic-dependent honesty using a dynamic Trust/Stress state vector, while the Evaluator Agent tracks suspicion to adapt clinical inquiries.
- The framework generates a synthetic corpus linking patient profiles, self-rated scales, semi-structured interviews, rater-administered scales, and diagnostic summaries under an explicit honesty model, validated for clinical realism and diagnostic consistency.

---

[UserLM-R1: Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning](http://arxiv.org/abs/2601.09215)

- UserLM-R1: introduces a novel reasoning user language model that constructs generalizable user profiles (Static Profile, Dynamic Profile) and employs a Goal-Driven Decision-Making Strategy (Reasoning Path) enhanced by Supervised Fine-Tuning and Multi-Reward Reinforcement Learning (RL) to simulate human-like strategic behavior.
- The framework enhances generalization by decoupling user personas into static roles and dynamic scenario-specific goals, allowing the LLM to adapt to diverse interaction scenarios and maintain consistency.
- The Multi-Reward RL stage utilizes Rule-Based Rewards and Rubric-Based Rewards, evaluated by an LLM Judge, to refine the model's proactive strategic ability and resistance to adversarial manipulation.

---

[Privacy Reasoner: Can LLM Emulate a Human-like Privacy Mind?](http://arxiv.org/abs/2601.09152)

- PrivacyReasoner: introduces an LLM-based AI agent designed to simulate individualized, context-sensitive human privacy reasoning by integrating privacy and cognitive theories.
- The architecture uses Privacy Memory Distillation to construct a user's "privacy mind" based on historical comments, which is then dynamically filtered by the Contextual Filter to activate relevant orientations for generating a Synthetic Comment.
- Reasoning fidelity is quantified by an LLM-as-a-Judge evaluator, the Privacy Concern Judge, which is calibrated against an established privacy concern taxonomy to assess concern-level alignment.

---

[World Craft: Agentic Framework to Create Visualizable Worlds via Text](http://arxiv.org/abs/2601.09150)

- World Craft: introduces an agentic world creation framework to generate executable and visualizable AI Town environments from user textual descriptions.
- The framework consists of two main modules, World Scaffold, which provides standardized infrastructure for automated scene construction, and World Guild, a multi-agent system for intent analysis and layout synthesis.
- World Guild employs four core agents (Enricher, Manager, Critic, Artist) and stepwise reasoning to mitigate the semantic gap between abstract user intent and precise spatial instructions required for game environment construction.

---

[KryptoPilot: An Open-World Knowledge-Augmented LLM Agent for Automated Cryptographic Exploitation](http://arxiv.org/abs/2601.09129)

- KRYPTOPILOT: introduces an open-world knowledge-augmented LLM agent for automated cryptographic exploitation, featuring the Reasoning and Tool Subsystem (Execution core), Knowledge Subsystem (Knowledge augmentation), and Governance Subsystem (Reliable efficient behavior).
- The framework addresses insufficient knowledge granularity via the Deep Research pipeline and stabilizes long-horizon reasoning using Behavioral Governance (Prompt Engineering) and Resource Governance (Model Routing).
- The agent achieves high solve rates on difficult CTF benchmarks by integrating precise knowledge alignment, structured memory, and adaptive model allocation across heterogeneous subtasks.

---

[The AI Hippocampus: How Far are We From Human Memory?](http://arxiv.org/abs/2601.09113)

- AI Hippocampus Framework (Taxonomy of Memory Mechanisms): introduces a comprehensive survey organizing LLM and MLLM memory literature into implicit, explicit, and agentic paradigms, detailing their architectural and functional evolution.
- Implicit memory is the knowledge embedded in transformer parameters (Neocortex analogy), explicit memory uses external storage for queryable knowledge (Hippocampus analogy), and agentic memory provides persistent structures for autonomous planning (Prefrontal Cortex analogy).
- The survey extends this taxonomy to memory-augmented MLLMs, emphasizing the integration of memory across vision, language, audio, and action modalities to enhance context-sensitive, real-world intelligent systems.

---

[Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](http://arxiv.org/abs/2601.09111)

- slow4fast-VLN (Fast-Slow Interactive Reasoning): introduces a dynamic interactive fast-slow reasoning framework for General Scene Adaptation Vision-Language Navigation (GSA-VLN), utilizing a Policy Network ($\pi$) and an LLM-based agent for continuous adaptation.
- The fast reasoning module executes real-time actions and stores memories in the History Repository ($L$), while the slow reasoning module reflects on these memories to extract generalized experiences ($E$) stored in the Experience Library ($M$).
- The extracted experiences guide and reinforce the fast reasoning network, enabling the agent to dynamically produce generalized strategies for unseen environments, supported by LLM-based Instruction Style Conversion (ISC).

---

[Human-AI Co-design for Clinical Prediction Models](http://arxiv.org/abs/2601.09072)

- HACHI (Human+Agent Co-design framework for Healthcare Instruments): introduces an iterative human-in-the-loop framework using LLM-based AI Agents to accelerate the development of fully interpretable Clinical Prediction Models (CPMs).
- The framework alternates between the AI Agent rapidly exploring and evaluating candidate concepts from clinical notes and the Clinical AI Team providing feedback to refine the CPM learning process.
- HACHI defines concepts as simple yes/no questions used in linear models, allowing transparent review, refinement, and validation by domain experts, improving performance and generalizability.

---

[GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](http://arxiv.org/abs/2601.09361)

- GeoRA (Geometry-Aware Low-Rank Adaptation): introduces a low-rank adaptation framework tailored for RLVR, utilizing a Geometry-Constrained Matrix, Geometric Prior Construction, Spectral Prior Mask, Euclidean Prior Mask, Singular Value Decomposition, Low-Rank Adapters, Residual Matrix, and Hidden State Computation to mitigate spectral mismatch and enable stable training.
- GeoRA extracts principal trainable directions via SVD within the geometry-constrained subspace ($W_{Geo}$) to initialize the Low-Rank Adapters ($A_{Geo}, B_{Geo}$), while the Residual Matrix ($W_{res}$) remains frozen as a structural anchor to preserve pre-trained knowledge.
- By aligning the low-rank adaptation with RLVR's distinct optimization dynamics, the framework achieves state-of-the-art performance on mathematical benchmarks and exhibits superior robustness against catastrophic forgetting.

---

[From SERPs to Agents: A Platform for Comparative Studies of Information Interaction](http://arxiv.org/abs/2601.09937)

- UXLab: introduces an open-source, modular system for comparative web-based user studies of information access systems, including RAG and autonomous agents, via its four-part architecture.
- The system features an Experimenter Dashboard for no-code configuration of complex experimental designs, managing participant flow, counterbalancing, and standardized data logging.
- The architecture uses Service Connectors to integrate diverse backends (traditional search, vector databases, LLMs) as interchangeable conditions without requiring frontend code changes.

---

[In-Browser Agents for Search Assistance](http://arxiv.org/abs/2601.09928)

- IBSA (In-Browser, Behavior-Grounded Search Assistant): introduces a hybrid, fully client-side architecture implemented as a browser extension that combines an adaptive probabilistic model with an in-browser SLM reasoner for privacy-preserving search assistance.
- The system uses Behavioral Observation to log user interactions locally in IndexedDB, feeding this data to the Dynamic User Model (a Multi-Layer Perceptron) for personalized policy adaptation via online learning executed by a Web Worker.
- The Policy's predicted next action grounds the SLM Reasoner (running locally via WebLLM/WebGPU) to generate concise, context-aware suggestions displayed through the Suggestion UI, ensuring user data sovereignty.

---

[CAMELS CAN USE COMPUTERS TOO: SYSTEM-LEVEL SECURITY FOR COMPUTER USE AGENTS](http://arxiv.org/abs/2601.09923)

- Dual-LLM Architecture with SSP: introduces a system-level security framework for Computer Use Agents (CUAs) that enforces Control Flow Integrity (CFI) by separating the Privileged Planner (P-LLM) from the Quarantined Perception model (Q-VLM).
- The P-LLM generates a complete, branching execution graph (single-shot) following an Observe-Verify-Act (OVA) paradigm before observing any untrusted UI content.
- The architecture mitigates instruction injection attacks but remains vulnerable to data-flow threats like Branch Steering, which is addressed using redundancy defenses such as DOM Consistency and Multi-Modal Consensus.

---

[Learning-Augmented Perfectly Secure Collaborative Matrix Multiplication](http://arxiv.org/abs/2601.09916)

- LA-PSMM (Learning-Augmented Perfectly Secure Matrix Multiplication): introduces a perfectly secure matrix multiplication (PSMM) protocol that integrates RL-discovered tensor decomposition into local computations to enhance efficiency while maintaining information-theoretic privacy.
- The PSMM core uses polynomial sharing with sparse masking and Beaver's triples to guarantee perfect privacy against threshold-bounded semi-honest colluding agents under explicit local storage constraints.
- By replacing dense local block multiplication with a low-rank tensor decomposition, the scheme achieves scalable computational efficiency gains (up to 80%) as matrix dimensions increase without sacrificing security.

---

[Zero-shot adaptable task planning for autonomous construction robots: a comparative study of lightweight single and multi-AI agent systems](http://arxiv.org/abs/2601.14091)

- MAATPF: introduces four architectures (Single-, Two-, Three-, and Four-Agent systems) built on lightweight LLMs and VLMs to enhance zero-shot task planning generalizability for autonomous construction robots.
- The Four-Agent system, comprising Observer, Planner, Actor, and Editor agents, consistently outperformed the state-of-the-art GPT-4o across correctness, temporal understanding, and executability metrics in three construction roles.
- The multi-agent approach, leveraging open-source models like MiniCPM-7B and Llama3-8B, proved approximately ten times more cost-effective than GPT-4o while demonstrating superior performance and generalizability.

---

[Interoperable Architecture for Digital Identity Delegation for AI Agents with Blockchain Integration](http://arxiv.org/abs/2601.14982)

- Interoperable Architecture for Digital Identity Delegation for AI Agents with Blockchain Integration: proposes a conceptual and architectural framework for verifiable delegation in digital identity systems, utilizing the Delegation Tetrahedron model, layered architecture (L0-L3), Canonical Verification Context (CVC), and Delegation Grants (DGs).
- The architecture is structured around four logical layers, separating trust anchoring (L0), credential validation (L1), delegation/policy evaluation (L2), and protocol routing/normalization via the Trust Gateway (L3).
- The framework achieves protocol neutrality by normalizing all verification requests into the CVC, enabling consistent, format-independent verification logic across heterogeneous identity ecosystems (federated, SSI, hybrid).

---

#### 13th January 2026


[Multicultural Spyfall: Assessing LLMs through Dynamic Multilingual Social Deduction Game](http://arxiv.org/abs/2601.09017)

- CTBS (Cultured Turn-Based Spyfall): introduces a dynamic benchmarking framework for evaluating multilingual and multicultural LLM capabilities through a turn-based social deduction game.
- The framework utilizes LLMs as Spy and Non-Spies, engaging in strategic dialogue across phases like the Round Robin Cycle and Free Cycle, centered around culturally relevant Game Entities ($\mathcal{E}$).
- Evaluation metrics, including Bradley-Terry ranking and Leakage Rate, demonstrate significant performance gaps in non-English contexts and when handling locally specific cultural entities (locations and foods).

---


[Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System](http://arxiv.org/abs/2601.08829)

- EloReview (LLM Agent Reviewer Simulation Framework): introduces a multi-round simulation framework where LLM agents, acting as reviewers and an Area Chair, interact within an Elo-ranked system using real-world conference submissions.
- The system incorporates a Memory Module that updates reviewers' strategies based on their Elo rating adjustments, enabling the study of adaptive review behavior across rounds.
- Simulation results demonstrate that incorporating Elo ratings significantly improves Area Chair decision accuracy and reveals how rank-based incentives shape reviewer dynamics.

---

[Inferring Latent Intentions: Attributional Natural Language Inference in LLM Agents](http://arxiv.org/abs/2601.08742)

- Att-NLI (Attributional Natural Language Inference): introduces a two-stage abductive-deductive framework for LLM agents to infer latent intentions in multi-agent settings, including LLM Agent, Intention Selection (Abduction), Conclusion Inference (Deduction), Isabelle/HOL (TP), Logical Record, Knowledge Base, and Undercover-V.
- The Neuro-Symbolic Att-NLI agent integrates the Isabelle/HOL theorem prover to ensure logically sound inference and external-feedback-guided refinement, achieving superior performance in the social deduction game Undercover-V.
- The framework addresses the gap in traditional NLI by formalizing intention-centered reasoning and is evaluated using game metrics alongside a novel Attributional Score measuring soundness and alignment.

---

[Lessons from the Field: An Adaptable Lifecycle Approach to Applied Dialogue Summarization](http://arxiv.org/abs/2601.08682)

- ASA (Agentic Summarization Approach): introduces a multi-agent system for multi-party dialogue summarization, featuring a Drafting Agent, three parallel Evaluation Agents, a Refinement Agent, and a Repetition Checker Agent, all powered by an LLM Engine.
- The architecture employs iterative refinement, where the Evaluation Agents provide granular, component-specific feedback to the Refinement Agent, allowing for optimization beyond end-to-end summary quality.
- To handle evolving requirements, the system relies on a hybrid evaluation protocol combining human preference A/B tests with a calibrated LLM-as-a-Judge (AutoEval) proxy for rapid iteration and validation.

---

[Advancing ESG Intelligence: An Expert-level Agent and Comprehensive Benchmark for Sustainable Finance](http://arxiv.org/abs/2601.08676)

- ESGAgent (Expert-level Agent and Comprehensive Benchmark): introduces a hierarchical multi-agent system empowered by a specialized toolset, including RAG, KG, and domain-specific tools (Retriever, Deep Researcher, Plotter, Report Tool), designed to generate expert-level ESG analysis and reports.
- The system utilizes a Central Planner to decompose complex ESG workflows into executable sub-tasks handled by specialized sub-agents, ensuring high-fidelity information fetch and multi-agent coordination.
- The accompanying three-level ESG Benchmark, derived from 310 corporate sustainability reports, serves as a rigorous testbed for assessing agentic capabilities across atomic QA, compositional tasks, and professional analytical report generation.

---

[Resisting Manipulative Bots in Memecoin Copy Trading: A Multi-Agent Approach with Chain-of-Thought Reasoning](http://arxiv.org/abs/2601.08641)

- MAF-MCT (Multi-Agent Framework for Memecoin Copy Trading): introduces an explainable, multi-modal, multi-agent system that decomposes copy trading into four specialized tasks handled by dedicated LLM-based agents using few-shot CoT prompting.
- The framework includes a Meme Evaluation Agent and a Trader Evaluation Agent, which utilize algorithmic bot detection to filter out manipulative activity and select high-quality assets and Key Opinion Leader (KOL) wallets.
- Empirical evaluation demonstrates the system achieves 73% and 70% precision in identifying high-quality meme coin projects and KOL wallets, respectively, generating significant collective profit.

---

[DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Report](http://arxiv.org/abs/2601.08536)

- DeepResearch Bench II (DRB II): introduces a comprehensive benchmark for evaluating Deep Research Systems (DRS) reports using 9,430 fine-grained binary rubrics derived from expert reports, covering Information Recall, Analysis, and Presentation.
- The rubrics are constructed via a four-stage LLM+human pipeline—LLM Extraction, Self-Evaluation Iteration, Manual Revision, and Expert Review and Refinement—to ensure they are atomic, verifiable, and aligned with human expert judgment.
- Evaluation using an LLM Judge (Gemini-2.5-Pro) reveals a substantial performance gap, with even the strongest DRS models satisfying fewer than 50% of the rubrics.

---

[M3-BENCH: Process-Aware Evaluation of LLM Agents Social Behaviors in Mixed-Motive Games](http://arxiv.org/abs/2601.08462)

- M3-BENCH (Process-Aware Evaluation of LLM Agents Social Behaviors in Mixed-Motive Games): introduces a multi-stage benchmark using a Four-Level Game Task System (Progressive mixed-motive tasks) and a process-aware evaluation framework with BTA (Analyze behavioral trajectories), RPA (Analyze decision reasoning), and CCA (Analyze communication content) modules.
- The framework jointly analyzes what an LLM agent does, thinks, and says, moving beyond outcome-oriented metrics to provide a comprehensive and diagnosable social behavior portrait.
- The three analysis views are aggregated using the Big Five Personality Theory and Social Exchange Theory to produce interpretable Social Personality Portrait Output and risk diagnostic reports.

---

[When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges](http://arxiv.org/abs/2601.08343)

- KV Cache Reuse Methods: introduces a systematic evaluation of KV cache reuse strategies (Dense Prefill, Naïve Reuse, KVCOMM, PAL-KV) in LLM-based multi-agent systems where an LLM Judge jointly compares multiple candidate responses.
- The study reveals that KV reuse, while efficient for execution agents, is decision-non-invariant for the judge, often preserving task accuracy while severely reducing the Judge Consistency Rate (JCR) relative to dense prefill.
- Mechanistic diagnostics attribute this failure mode to the disruption of fine-grained cross-candidate interactions crucial for reliable joint comparison, especially under variable candidate orderings (shuffle).

---

[Semantic Laundering in AI Agent Architectures: Why Tool Boundaries Do Not Confer Epistemic Warrant](http://arxiv.org/abs/2601.08333)

- Architectural Epistemology Analysis: formalizes semantic laundering, an architectural failure pattern in LLM-based agent architectures where propositions acquire unjustified epistemic status by crossing architecturally trusted interfaces.
- This laundering constitutes an architectural realization of the Gettier problem, leading to the Theorem of Inevitable Self-Licensing, which proves that circular epistemic justification is unavoidable under standard architectural assumptions where tool outputs are uniformly treated as observations.
- The work introduces the Warrant Erosion Principle and argues that avoiding this structural defect requires explicit epistemic typing of tools at the architectural level, distinguishing OBSERVER, COMPUTATION, and GENERATOR classes.

---

[D2PLAN: Dual-Agent Dynamic Global Planning for Complex Retrieval-Augmented Reasoning](http://arxiv.org/abs/2601.08282)

- D2PLAN (Dual-agent Dynamic global Planning paradigm): introduces a dual-agent architecture for complex retrieval-augmented reasoning, featuring a Reasoner (Dynamic global planning) and a Purifier (Relevance assessment/Condensation).
- The Reasoner constructs and dynamically adapts explicit Global Plan States based on retrieval feedback, while the Purifier filters irrelevant information from the Retriever's results, mitigating ineffective search chains and reasoning hijacking.
- The system is trained using a two-stage framework: SFT Cold-Start (Supervised fine-tuning stage) followed by SPLANRL (Plan-oriented reinforcement learning) to enhance LLMs' planning proficiency and robust resistance to peripheral evidence.

---

[Greedy Is Enough: Sparse Action Discovery in Agentic LLMs](http://arxiv.org/abs/2601.08280)

- Contextual Block-OMP: introduces a theoretical framework for sparse action discovery in agentic LLMs by formulating tool selection as a block-sparse recovery problem using a contextual linear reward model.
- The greedy algorithm, inspired by Orthogonal Matching Pursuit, provably recovers the relevant action Support Set with sample complexity scaling logarithmically in the total number of Actions (M).
- The results provide a theoretical foundation for action pruning in large-action decision-making systems, demonstrating that sparsity and sufficient coverage are necessary for tractability.

---

[Sparsity Is Necessary: Polynomial-Time Stability for Agentic LLMs in Large Action Spaces](http://arxiv.org/abs/2601.08271)

- SAC (Sparse Agentic Control): introduces a framework for sparse control in large action spaces, developing an $l_1$-regularized approach to policy learning with logarithmic dependence on the action dimension $M$.
- The framework models tool-augmented LLM systems as sequential decision problems where the action universe is massive ($M \gg 1$) but only a small, sparse subset ($k \ll M$) of tools is relevant.
- Theoretical results establish that the $l_{1,2}$-regularized learner achieves near-optimal control and exact support recovery with sample complexity scaling as $k \log M$, avoiding the linear dependence on $M$ required by dense policy classes.

---

[Unleashing Tool Engineering and Intelligence for Agentic AI in Next-Generation Communication Networks](http://arxiv.org/abs/2601.08259)

- TA-AAI (Tool-Augmented Agentic AI): introduces a comprehensive roadmap for integrating tool intelligence into next-generation communication networks, detailing the entire tool engineering lifecycle from creation to benchmarking, and demonstrating its application in UAV trajectory planning.
- The framework embeds LLMs as the core reasoning engine within a continuous cycle of perception, reasoning & planning, action, and memory & evolution, where tools serve as essential interfaces for environmental grounding and real-world actuation.
- A case study on UAV communications utilizes Teacher-Guided Reinforcement Learning combined with a Feasibility Shield to enable the agent to intelligently schedule tool activations while strictly adhering to energy constraints.

---

[Large Artificial Intelligence Model–Guided Deep Reinforcement Learning for Resource Allocation in Non-Terrestrial Networks](http://arxiv.org/abs/2601.08254)

- LAM-DRL (Large Artificial Intelligence Model–Guided Deep Reinforcement Learning): introduces a hybrid intelligence framework where an LLM acts as a high-level coordinator generating textual strategies that shape the DRL agent's reward and policy via strategy embedding and attention.
- The DRL agent, implemented using a TD3 actor-critic algorithm, performs resource allocation (power and bandwidth fractions) in a Low Earth Orbit (LEO) Non-Terrestrial Network (NTN) environment modeled as a Markov Decision Process (MDP).
- The framework significantly outperforms traditional DRL and heuristic baselines, achieving 40% higher throughput in nominal scenarios and 64% higher throughput in extreme weather scenarios, while providing interpretable policies.

---

[The End of Reward Engineering: How LLMs Are Redefining Multi-Agent Coordination](http://arxiv.org/abs/2601.08237)

- LLM-Enabled Multi-Agent Coordination: introduces a paradigm shift from manual reward engineering to language-based objectives, enabling multi-agent systems to coordinate through shared semantic understanding rather than engineered numerical signals.
- The transition is supported by three interconnected pillars: semantic reward specification (preserving human intent), dynamic adaptation (responding to behavioral feedback), and inherent human alignment (ensuring interpretability).
- The paper identifies two distinct pathways: LLM-Generated Rewards (offline reward function generation for standard MARL) and LLM-Mediated Communication (LLMs acting as runtime agent controllers coordinating via natural language dialogue).

---

[Evaluating Implicit Regulatory Compliance in LLM Tool Invocation via Logic-Guided Synthesis](http://arxiv.org/abs/2601.08196)

- LOGISAFETYGEN: introduces an automated benchmark generation framework that converts unstructured regulations into Linear Temporal Logic (LTL) oracles and uses logic-guided fuzzing to synthesize valid, safety-critical traces.
- The framework addresses the Ambiguity, Validity, and Inference Gaps by constructing formal compliance standards and generating natural language instructions that require LLMs to infer implicit safety constraints from context.
- Instantiating the framework yields LOGISAFETYBENCH, a human-verified benchmark of 240 tasks across high-stakes domains (Financial Services, Tele-Healthcare, Smart Home IoT) designed to measure implicit regulatory compliance in LLM tool invocation.

---

[Route, Retrieve, Reflect, Repair: Self-Improving Agentic Framework for Visual Detection and Linguistic Reasoning in Medical Imaging](http://arxiv.org/abs/2601.08192)

- R⁴ (Route, Retrieve, Reflect, Repair): introduces a self-improving agentic framework for medical image analysis that decomposes workflows into four coordinated agents: Router, Retriever, Reflector, and Repairer.
- The framework couples global report generation with quantitative localization, using a persistent Exemplar Memory to retrieve context-aware few-shot exemplars for self-improvement without VLM fine-tuning.
- The core mechanism involves a Route $\rightarrow$ Retrieve $\rightarrow$ Reflect $\rightarrow$ Repair loop that critiques and iteratively refines both free-text clinical reports and bounding boxes based on structured issue lists.

---

[ZeroDVFS: Zero-Shot LLM-Guided Core and Frequency Allocation for Embedded Platforms](http://arxiv.org/abs/2601.08166)

- ZeroDVFS (Zero-Shot LLM-Guided Core and Frequency Allocation for Embedded Platforms): introduces a model-based hierarchical MARL framework for thermal- and energy-aware DVFS and task-to-core allocation, utilizing LLM Feature Extraction, an Environment Model, Profiler Agent, Temperature Agent, Policy Model, Reward Function, and Dyna-Q Framework.
- The framework enables zero-shot deployment for new workloads by combining LLM-extracted semantic features with an accurate environment model that generates synthetic training data, achieving 7.09x better energy efficiency and 4.0x better makespan than the Linux ondemand governor.
- By decomposing the action space using two collaborative agents and leveraging model-based planning, ZeroDVFS achieves first-decision latency 8,300x faster than table-based profiling, enabling practical deployment in dynamic embedded systems.

---

[SwiftMem: Fast Agentic Memory via Query-aware Indexing](http://arxiv.org/abs/2601.08160)

- SwiftMem: introduces a query-aware agentic memory system that achieves sub-linear retrieval complexity using Query Preprocess (Analyzes query characteristics), Temporal Index (Time-sensitive range queries), DAG-Tag Index (Hierarchical semantic organization), Embedding Index (Similarity-based retrieval) with Co-consolidation (Reorganizes embedding storage), Storage Layer (Hybrid data persistence), Base LLM (Generates final response), and Reranker (Refines memory subset).
- The system overcomes the $O(N_{mem})$ search complexity bottleneck of existing frameworks by employing specialized indexing over temporal and semantic dimensions to fetch only relevant memory subsets.
- SwiftMem achieves up to 47x faster search latency compared to state-of-the-art baselines while maintaining competitive accuracy, enabling practical real-time deployment of memory-augmented LLM agents.

---

[WISE-Flow: Workflow-Induced Structured Experience for Self-Evolving Conversational Service Agents](http://arxiv.org/abs/2601.08158)

- WISE-Flow (Workflow-Induced Structured Experience): introduces a workflow-centric framework that converts historical service interactions into reusable procedural experience by inducing workflows with prerequisite-augmented action blocks.
- The framework operates in three stages: logging raw interactions, offline induction of structured workflows, and online workflow-guided agent control via retrieval, progress alignment, and feasibility checks.
- The Workflow Induction Engine uses contrastive evidence from successes and failures to synthesize workflows, which the WISE-Flow Agent uses for prerequisite-aware feasibility reasoning and state-grounded next actions.

---

[Project Synapse: A Hierarchical Multi-Agent Framework with Hybrid Memory for Autonomous Resolution of Last-Mile Delivery Disruptions](http://arxiv.org/abs/2601.08156)

- Project Synapse: introduces a novel hierarchical multi-agent framework designed for autonomous resolution of last-mile delivery (LMD) disruptions, featuring a Resolution Supervisor delegating tasks to specialized worker agents augmented by a Hybrid Memory Architecture.
- The system uses a Hybrid Memory Architecture integrating Working Memory (active context), Episodic Memory (past incidents via SQL DB), and Semantic Memory (organizational policies via RAG/vector DB) to enable stateful, fact-grounded reasoning.
- Orchestrated by LangGraph, the framework achieved an overall average score of 0.73 in evaluation, demonstrating strong performance in reasoning quality (0.77) and plan correctness (0.71) for complex, real-world operational problem-solving.

---

[MIRRORBENCH: AN EXTENSIBLE FRAMEWORK TO EVALUATE USER-PROXY AGENTS FOR HUMAN-LIKENESS](http://arxiv.org/abs/2601.08118)

- MIRRORBENCH (An Extensible Framework to Evaluate User-Proxy Agents for Human-Likeness): introduces a reproducible, six-layer benchmarking framework that evaluates user-proxy agents solely on their human-likeness using Execution Backends (Runs evaluation units)/Persistence (Manages durable state)/Core Engine (Defines typed data models)/Orchestration (Coordinates evaluation workflow)/Plugin Components (Hosts domain logic)/Task Drivers (Encodes interaction protocol)/API & Interface (Provides user-facing access).
- The framework quantifies human-likeness using three lexical-diversity metrics (MATTR, YULE'S K, HD-D) and three LLM-judge metrics (GTEVAL, Pairwise Indistinguishability (PI), Rubric-and-Reason (RNR)), explicitly decoupling evaluation from downstream task success.
- The system utilizes a compatibility-aware planner, multi-backend execution (synchronous/asynchronous/Ray), caching, and structured observability to ensure reproducible, variance-aware results at scale.

---

[Query Suggestion for Retrieval-Augmented Generation via Dynamic In-Context Learning](http://arxiv.org/abs/2601.08105)

- Dynamic Few-Shot Learning (DFSL): introduces a robust framework for query suggestion in agentic RAG systems by dynamically retrieving answerability-aware examples via templating and embedding-based retrieval.
- The system uses a self-learning mechanism where an LLM evaluates the RAG agent's execution chain to automatically label query templates with their answerability status (answerable, no workflow, no knowledge).
- DFSL significantly improves both the semantic similarity and the answerability of suggested queries compared to static few-shot and retrieval-only baselines, enabling safer user interaction.

---

[The Hierarchy of Agentic Capabilities: Evaluating Frontier Models on Realistic RL Environments](http://arxiv.org/abs/2601.09032)

- HAC (Hierarchy of Agentic Capabilities): introduces an empirically-derived five-level hierarchy for LLM agents evaluated in the CORECRAFT RL environment, including Tool Use, Planning and Goal Formation, Adaptability, Groundedness, and Common-Sense Reasoning.
- The evaluation on 150 realistic e-commerce workplace tasks revealed that even frontier models fail approximately 40% of tasks, with failures clustering predictably along the defined hierarchy.
- Weaker models struggle with fundamental tool use and planning, while stronger models are primarily limited by common-sense reasoning requiring contextual inference beyond explicit instructions.

---

[ART: Action-based Reasoning Task Benchmarking for Medical AI Agents](http://arxiv.org/abs/2601.08988)

- ART (Action-based Reasoning Task Benchmarking): introduces a four-stage agentic pipeline for generating clinically grounded, failure-targeted benchmark tasks, including Scenario Identification, Task Generation, Quality Audit, and AgentBench Evaluation.
- The framework systematically targets three dominant failure types—data retrieval, temporal aggregation, and threshold-based conditional logic—to expose reasoning weaknesses in medical AI agents.
- By grounding synthetic tasks in real EHR-formatted patient data and incorporating Human-in-the-Loop clinical expert validation, the system aims to improve the robustness and reliability of clinical decision support LLMs.

---

[Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models](http://arxiv.org/abs/2601.08955)

- ITP (Imagine-then-Plan): introduces a unified framework for LLM-based agents that uses adaptive lookahead via a learned World Model, enabling decisions conditioned on both observable present states and imagined future trajectories.
- The approach formalizes the Partially Observable and Imaginable Markov Decision Process (POIMDP), where the agent performs a proactive, deliberative "rehearsal" phase using multi-step rollouts to anticipate goal progress and potential conflicts.
- ITP is instantiated in a training-free variant ($ITP_I$) and a reinforcement-trained variant ($ITP_R$), with $ITP_R$ jointly optimizing the action policy and the lookahead horizon selection via online A2C optimization.

---

[MemRec: Collaborative Memory-Augmented Agentic Recommender System](http://arxiv.org/abs/2601.08816)

- MemRec (Collaborative Memory-Augmented Agentic Recommender System): introduces a framework that architecturally decouples the LLMRec (Reasoning Core) from the LMMem (Memory Manager) to enable efficient collaborative augmentation using a unified Collaborative Memory Graph.
- The system operates via a three-stage pipeline: Collaborative Memory Retrieval synthesizes high-signal context, Grounded Reasoning scores items, and Asynchronous Collaborative Propagation evolves the semantic memory graph in the background.
- LMMem mitigates cognitive overload during retrieval using zero-shot LLM-guided curation and circumvents update bottlenecks via cost-effective asynchronous graph propagation.

---

[Your Group-Relative Advantage Is Biased](http://arxiv.org/abs/2601.08521)

- HA-DW (History-Aware Adaptive Difficulty Weighting): introduces an adaptive reweighting scheme that adjusts the Group-Relative Advantage Estimator based on an Evolving Difficulty Anchor and long-term reward trends to mitigate systematic bias in group-based RL.
- The paper theoretically demonstrates that group-relative advantage estimation systematically underestimates advantages for hard prompts and overestimates them for easy prompts, leading to imbalanced exploration and exploitation in LLM training.
- The Evolving Difficulty Anchor, updated via a Kalman-style process using cross-batch information, informs the History-aware Reweighting Factor ($\Phi_{t,i}$) to boost exploration on difficult prompts and weaken exploitation on easy ones, yielding consistent performance gains across LLM reasoning benchmarks.

---

[JudgeRLVR: Judge First, Generate Second for Efficient Reasoning](http://arxiv.org/abs/2601.08468)

- JudgeRLVR: introduces a two-stage judge-then-generate paradigm for efficient LLM reasoning, comprising a Judging Stage that trains discriminative error awareness and a Generating Stage that optimizes solution generation.
- The Judging Stage trains the model to discriminate valid solutions, internalizing a guidance signal that prunes the search space before subsequent generation.
- Initialized from the judge, the Generating Stage applies Vanilla RLVR fine-tuning, achieving superior accuracy and significantly reduced generation length (up to 42% shorter CoT) compared to Vanilla RLVR.

---

[RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation](http://arxiv.org/abs/2601.08430)

- CF-RG (Coarse-to-Fine Rubric Generation): introduces an automated framework for synthesizing RubricHub, a large-scale, multi-domain dataset of highly discriminative rubrics, using Principle-Guided & Response-Grounded Generation, Multi-Model Aggregation, and Difficulty Evolution.
- RubricHub provides fine-grained supervision signals utilized in a two-stage post-training pipeline, comprising Rubric-based Rejection Sampling Fine-Tuning (RuFT) for initial alignment and Rubric-based Reinforcement Learning (RuRL) for policy optimization.
- The approach enables a post-trained Qwen3-14B model to achieve SOTA performance on HealthBench, surpassing proprietary frontier LLMs like GPT-5 by mitigating the supervision ceiling effect inherent in coarse criteria.

---

[Hidden in Plain Text: Measuring LLM Deception Quality Against Human Baselines Using Social Deduction Games](http://arxiv.org/abs/2601.13709)

- LLM Mafia Framework: introduces an asynchronous multi-agent architecture using GPT-4o agents to simulate 35 games of Mafia, and an LLM Mafia Detector (GPT-4-Turbo) to predict mafia roles from transcripts, serving as a surrogate marker for deception quality.
- The framework compares the Mafia Detector's prediction accuracy on LLM-generated transcripts against human-played transcripts and a random baseline, finding that LLM mafia agents are consistently harder to detect.
- The results suggest that LLMs are better than humans at natural language deception in social deduction games, underscoring the sophistication and risks of LLM deception in social contexts.

---

#### 12th January 2026

[Beyond Single-Shot: Multi-step Tool Retrieval via Query Planning](http://arxiv.org/abs/2601.07782)

- TOOLQP (Tool Query Planner): introduces a lightweight framework that models tool retrieval as an iterative planning process, utilizing Planner (iterative query planning), Task Decomposition (sequential sub-tasks), Interactive Query Generation (dynamic query sequence), Retrieval Aggregation (peak-rank fusion), Retriever (existing dense retrieval system), Downstream LLM (tool execution grounding), and RLVR (policy optimization training).
- By reframing retrieval from static similarity matching to dynamic planning, TOOLQP effectively bridges the semantic gap between abstract user goals and technical tool documentation.
- The framework is modular, integrating seamlessly with existing dense retrievers and downstream LLMs, and is trained using synthetic query trajectories optimized via RLVR for robust zero-shot generalization.

---

[Is Agentic RAG worth it? An experimental comparison of RAG approaches](http://arxiv.org/abs/2601.07711)

- ERAG/ARAG Comparison: introduces an experimental evaluation comparing Enhanced RAG (ERAG), a fixed pipeline with dedicated modules, against Agentic RAG (ARAG), an iterative LLM-orchestrated loop utilizing tools.
- ERAG incorporates a REWRITER, ROUTER, RETRIEVER, and RERANKER to mitigate known weaknesses of Naïve RAG systems before passing context to the GENERATOR.
- ARAG, where the LLM acts as the AGENT orchestrator, excels in user intent handling and query rewriting due to its dynamic decision-making, but is systematically more expensive and fails to benefit from iterative retrieval refinement.

---

[Beyond Static Tools: Test-Time Tool Evolution for Scientific Reasoning](http://arxiv.org/abs/2601.07641)

- TTE (Test-Time Tool Evolution): introduces a novel paradigm where LLM agents perform scientific reasoning by enabling tools to be generated, verified, and evolved during inference using Structured Task Decomposition, Dynamic Tool Retrieval, Generative Tool Synthesis, Atomic Tool Refinement, and a Runtime Execution Engine.
- The framework operates via a closed-loop evolutionary workflow, dynamically synthesizing executable tools on demand to overcome the sparsity and rigidity limitations inherent in static, pre-defined tool libraries.
- TTE establishes a new state-of-the-art performance in scientific reasoning benchmarks (SciEvo) by ensuring the tool space remains intrinsically aligned with the unbounded scientific problem space.

---

[DIAGPaper: Diagnosing Valid and Specific Weaknesses in Scientific Papers via Multi-Agent Reasoning](http://arxiv.org/abs/2601.07611)

- DIAGPaper: introduces a human-grounded multi-agent framework for paper weakness identification, featuring the Customizer Module (Criteria generation), Rebuttal Module (Adversarial validation), and Prioritizer Module (Weakness ranking).
- The Customizer dynamically generates paper-specific review dimensions to instantiate specialized Reviewer Agents, enabling differentiated and collaborative reviewing behaviors.
- The Rebuttal module uses multi-round reviewer-author debate to validate and filter invalid critiques, while the Prioritizer ranks validated weaknesses using a severity score grounded in empirical review patterns.

---

[Beyond Entangled Planning: Task-Decoupled Planning for Long-Horizon Agents](http://arxiv.org/abs/2601.07577)

- TDP (Task-Decoupled Planning): introduces a training-free modular planning framework that enforces sub-task decoupling by separating global task structuring (Supervisor) from node-level decision making (Planner-Executor pair).
- The framework decomposes tasks into a Directed Acyclic Graph (DAG) of sub-tasks, confining reasoning and replanning to a focused, node-scoped context to prevent error propagation.
- A Self-Revision module updates the DAG structure and refines specifications to maintain long-horizon coherence, significantly reducing token consumption compared to monolithic planning methods.

---

[VirtualEnv: A Platform for Embodied AI Research](http://arxiv.org/abs/2601.07553)

- VirtualEnv: introduces a next-generation simulation platform built on Unreal Engine 5 for embodied AI research, supporting multi-agent collaboration and language-driven task execution using LLMs and vLLMs.
- The platform utilizes a Scene Graph Representation and a VirtualEnv API to enable agents to interpret natural language instructions, generate symbolic plans, and dynamically interact with complex indoor and outdoor environments.
- VirtualEnv facilitates scalable scenario creation and environment modification via vLLM interpretation checks, providing a standardized testbed for benchmarking LLM performance in complex, interactive tasks.

---

[From RAG to Agentic RAG for Faithful Islamic Question Answering](http://arxiv.org/abs/2601.07528)

- Agentic RAG: introduces an end-to-end grounded Islamic modeling suite and an agentic Quran-grounding framework for faithful Islamic Question Answering (QA), utilizing SFT, RL Alignment, LLM-as-a-Judge, and iterative evidence seeking via Agentic Tools over a Quran DB.
- The approach leverages structured tool calls within a multi-turn reasoning environment to perform iterative evidence seeking and answer revision, significantly reducing hallucinations compared to standard RAG.
- The paper also introduces ISLAMICFAITHQA, a 3,810-item bilingual generative benchmark designed to measure hallucination and abstention directly using a strict LLM-as-a-Judge protocol.

---

[FROAV: A Framework for RAG Observation and Agent Verification — Lowering the Barrier to LLM Agent Research](http://arxiv.org/abs/2601.07504)

- FROAV (Framework for RAG Observation and Agent Verification): introduces an open-source research platform integrating visual workflow orchestration, a comprehensive evaluation framework, and extensible Python integration to democratize LLM agent research.
- The system utilizes a Docker Compose cluster to orchestrate n8n, FastAPI, Streamlit, and PostgreSQL, ensuring reproducible deployment and granular data management for execution traces and feedback.
- The platform implements a multi-stage RAG pipeline coupled with a rigorous multi-model LLM-as-a-Judge evaluation system that assesses outputs across four dimensions: Reliability, Completeness, Understandability, and Relevance.

---

[JUDGEFLOW: AGENTIC WORKFLOW OPTIMIZATION VIA BLOCK JUDGE](http://arxiv.org/abs/2601.07477)

- JUDGEFLOW (Evaluation-Judge-Optimization-Update pipeline): introduces reusable Logic Blocks as higher-level structural abstractions and a dedicated Judge module to optimize LLM-based agentic workflows.
- The Judge module analyzes failed execution traces to assign rank-based responsibility scores to problematic Logic Blocks, providing fine-grained diagnostic signals for error localization.
- The LLM-based Optimizer leverages these targeted signals to focus modifications (Add, Remove, Modify Block actions) on the weakest components, improving sample efficiency and interpretability compared to end-to-end methods.

---

[Learning How to Remember: A Meta-Cognitive Management Method for Structured and Transferable Agent Memory](http://arxiv.org/abs/2601.07470)

- MCMA (Meta-Cognitive Memory Abstraction method): introduces a novel approach that treats memory abstraction as a learnable cognitive skill, decoupling task execution (Task Model) from memory management (Memory Copilot).
- The Memory Copilot is trained using Direct Preference Optimization (DPO) based on Preference Pairs derived from downstream task performance to distill raw trajectories into reusable Structured Memory.
- Structured Memory is organized into Hierarchical Abstraction levels using composite Structural Primitives (Tree, Chain, Key-Value, Natural Text), enabling cross-task transfer of both knowledge and the abstraction ability itself.

---

[Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents](http://arxiv.org/abs/2601.07468)

- TSM (Temporal Semantic Memory): introduces a memory framework that models semantic time for point-wise memory and supports the construction and utilization of durative memory for personalized LLM agents.
- The framework constructs a semantic timeline via a Temporal Knowledge Graph (Episodic Memory) and consolidates temporally continuous information into Durative Memory (Topics and Personas).
- Memory utilization integrates the query's semantic temporal intent to retrieve time-valid, duration-consistent context using dense matching, temporal reranking, and filtering.

---

[MCP-ITP: An Automated Framework for Implicit Tool Poisoning in MCP](http://arxiv.org/abs/2601.07395)

- MCP-ITP (An Automated Framework for Implicit Tool Poisoning in MCP): introduces an automated framework for crafting stealthy poisoned tool descriptions that manipulate LLM agents into invoking high-privilege tools without executing the poisoned tool itself.
- The framework employs an iterative black-box optimization strategy using an Attacker LLM ($L_A$), a Detector LLM ($L_D$), and an Evaluator LLM ($L_E$) to maximize the Attack Success Rate (ASR) while minimizing the Malicious Tool Detection Rate (MDR).
- MCP-ITP significantly outperforms manually crafted baselines, achieving high ASR (up to 84.2%) and low MDR (as low as 0.3%) across various LLM agents in the Model Context Protocol (MCP) ecosystem.

---

[OpenTinker: Separating Concerns in Agentic Reinforcement Learning](http://arxiv.org/abs/2601.07376)

- OpenTinker: introduces an open-source infrastructure for RL of LLM agents built on a modular client-scheduler-server architecture that separates algorithm design, execution, and agent-environment interaction.
- The system is designed as Reinforcement Learning as a Service (RLaaS) supporting distributed, multi-tenant serving of agentic workloads over shared cluster resources.
- It supports multi-agent RL via an Agent Protocol Coordinator, which enforces interaction protocols and synchronization within the environment abstraction.

---

[GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap](http://arxiv.org/abs/2601.07375)

- GROKE (Graph-based Reasoning over OSM Knowledge for instruction Evaluation): introduces a vision-free, training-free hierarchical LLM-based framework for evaluating navigation instructions using OpenStreetMap data.
- The system uses a Sub-instruction Agent (instruction parsing) to decompose instructions into atomic sub-goals, which are then executed by a Navigator Agent (waypoint selection) reasoning over structured JSON spatial representations.
- By formalizing the Agent-as-Judge methodology, the framework provides scalable and interpretable evaluation metrics that correlate significantly with human judgments of instruction navigability, reducing navigation error by 68.5% compared to baselines.

---

[FOCAL: A Novel Benchmarking Technique for Multi-modal Agents](http://arxiv.org/abs/2601.07367)

- FOCAL: introduces a novel benchmarking framework for multi-modal agents (voice + text input/output) using a cascading pipeline architecture, including Human-Simulator (LLM), SOTA TTS, Agent ASR, Agent LLM (RAG-based reasoning), Agent TTS, SOTA ASR, Ground Truth, Implementation Transcript, LLM as Judge, Human Evaluator, KnowledgeBase, and Tool-Calling components.
- The framework evaluates end-to-end reasoning and component-wise error propagation using novel metrics like Reasoning and Semantic scores, alongside standard metrics like WER and Contextual Similarity.
- The pipeline supports both automated testing via the Human-Simulator LLM and human-involved testing, providing a unified approach for scale and quality evaluation of voice agents.

---

[Agentic Diagnostic Reasoning over Telecom and Datacenter Infrastructure: Foundation for Autonomous Incident Resolution and Change Impact Mitigation](http://arxiv.org/abs/2601.07342)

- ADR (Agentic Diagnostic Reasoning): introduces a tool-augmented agentic framework where an LLM Agent performs diagnostic reasoning and impact analysis over multi-layer infrastructure models using a structured RCA Investigation Protocol and Tool-Grounded Reasoning.
- The framework abstracts the underlying Infrastructure Ontology (typed directed graph) via the Model Context Protocol (MCP), which exposes a constrained set of MCP Tools for data access, ensuring grounding and preventing LLM hallucination.
- This approach delegates complex causal reasoning to the LLM, laying the foundation for autonomous incident resolution and proactive change impact mitigation without relying on embedded graph traversal algorithms.

---

[Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation](http://arxiv.org/abs/2601.07338)

- RATE (Reflective Agentic Translation Evaluation): introduces a novel agentic translation evaluation framework designed to mitigate LLM limitations, such as knowledge cutoff and score inconsistency, when assessing non-literal translation quality.
- The framework is architected around a Core Agent operating on a dynamic reflective loop that orchestrates three specialized sub-agents: the Evaluation Agent for scoring, the Search Agent for external knowledge retrieval, and the Comparison Agent for score calibration.
- RATE achieves superior meta scores compared to traditional MT metrics and LLM-as-a-Judge paradigms on the new MENT dataset, which systematically benchmarks evaluation reliability across challenging non-literal domains.

---

[ARM: Role-Conditioned Neuron Transplantation for Training-Free Generalist LLM Agent Merging](http://arxiv.org/abs/2601.07309)

- ARM (Agent-Role Merging): introduces a training-free pipeline for consolidating benchmark-specialized LLM agents into a single generalist checkpoint, utilizing Backbone Pool Construction, AOS-based Backbone Selection, and Conflict-Aware Neuron Transplantation.
- The framework addresses instability across merge operators and destructive interference in multi-turn trajectories by selecting a strong initialization and performing localized edits on role-critical circuits.
- ARM uses role-conditioned activation tracing on a lightweight calibration set to identify and transplant small subsets of MLP neurons, significantly improving worst-suite robustness and generalization across diverse interactive environments.

---

[LRAS: Advanced Legal Reasoning with Agentic Search](http://arxiv.org/abs/2601.07296)

- LRAS (Legal Reasoning with Agentic Search): introduces a dual-mechanism learning architecture integrating Introspective Imitation Learning and Difficulty-aware Reinforcement Learning to shift legal LLMs from static closed-loop thinking to dynamic Active Inquiry.
- The framework utilizes an Introspective Action Space, including `<think>`, `<search>`, and `<answer>` actions, enabling the LLM agent to autonomously identify knowledge boundaries and execute multi-step exploratory searches.
- The approach relies on a robust Retrieval Pipeline, employing SerpAPI, Jina Reader, and an external Qwen3-32B Summarizer LLM, to ground legal conclusions in verified, precise external statutes.

---

[The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents](http://arxiv.org/abs/2601.07264)

- CAR (Calibration Agentic RL): introduces an RL-based fine-tuning framework that jointly optimizes LLM agent task accuracy and confidence reliability using a holistic reward benchmark, including the MSCR.
- The research systematically investigates verbalized calibration in tool-use agents, revealing a fundamental confidence dichotomy driven by tool type: evidence tools cause overconfidence, while verification tools mitigate miscalibration.
- The framework addresses miscalibration, particularly in evidence tool scenarios, by enforcing strict reward separation via MSCR to build self-aware agents capable of reliable uncertainty expression.

---

[When Bots Take the Bait: Exposing and Mitigating the Emerging Social Engineering Attack in Web Automation Agent](http://arxiv.org/abs/2601.07263)

- SUPERVISOR (Lightweight Pluggable Defense Module): introduces a pluggable runtime module to mitigate AGENTBAIT (Social Engineering Attack Paradigm) by enforcing consistency checks using the Environment Consistency Check, Intention Consistency Check, and Decision Engine, supported by LLM-based User Task Analysis, Action Semantics Classification, and Input Sensitivity Classification, which enforce Permission Policy and Sensitivity Policy.
- The AGENTBAIT attack paradigm exploits intrinsic weaknesses in web agents by embedding inducement contexts that distort the agent's reasoning toward malicious objectives, achieving an average attack success rate of 67.5% across mainstream frameworks.
- The defense module achieves strong protection by reducing the attack success rate by up to 78.1% on average, enforcing environment and intention consistency alignment before execution with minimal 7.7% runtime overhead.

---

[COLORBROWSERAGENT: AN INTELLIGENT GUI AGENT FOR COMPLEX LONG-HORIZON WEB AUTOMATION](http://arxiv.org/abs/2601.07262)

- ColorBrowserAgent: introduces a Collaborative Autonomy framework integrating Progressive Progress Summarization and Human-in-the-Loop Knowledge Adaptation to achieve robust, long-horizon web automation.
- The architecture employs a dual-agent control loop where the Summarizer Agent maintains coherent task narrative and the Operator Agent executes actions grounded in retrieved knowledge.
- The system uses hybrid discriminators (Rule-Based and VLM-Based) to autonomously trigger human expert intervention, capturing site-specific knowledge tips for the Adaptive Knowledge Base (AKB).

---

[DarwinTOD: LLM Driven Lifelong Self Evolution for Task Oriented Dialog Systems](http://arxiv.org/abs/2601.07248)

- DarwinTOD: introduces a lifelong self-evolving dialog framework that systematically integrates LLM-driven evolutionary computation and strategy optimization via a dual-loop process, enabling continuous strategy refinement from a zero-shot base.
- The system operates through an Online Execution Loop, where multi-agents (DST, DP, NLG, UserSim) perform dialogs with peer critique, and an Offline Evolution Loop, which refines the Evolvable Strategy Bank (ESB) using accumulated feedback stored in the Shared Structured Memory (SSM).
- This closed-loop design facilitates autonomous, continuous improvement of dialog strategies without requiring task-specific fine-tuning or human intervention, achieving state-of-the-art performance.

---

[Consolidation or Adaptation? PRISM: Disentangling SFT and RL Data via Gradient Concentration](http://arxiv.org/abs/2601.07224)

- PRISM (Partitioning Regimes via Internal Spatial-gradient Metrics): introduces a dynamics-aware framework that uses Non-Invasive Gradient Probing and a Statistical Concentration Toolkit to quantify Structural Dissonance, enabling Distribution-Adaptive Routing of data to the SFT Regime for consolidation or the RL Regime for structural adaptation in LLM agents.
- The framework utilizes gradient concentration as a robust proxy for cognitive conflict, routing high-conflict data (concentrated gradients) to RL for structural restructuring and low-conflict data (diffuse gradients) to SFT for behavioral consolidation.
- By selectively allocating high-conflict data to RL, the approach achieves a Pareto improvement, delivering superior generalization while reducing RL computational overhead by up to 3.22x.

---

[Active Context Compression: Autonomous Memory Management in LLM Agents](http://arxiv.org/abs/2601.07190)

- Focus Agent: introduces an agent-centric architecture inspired by slime mold behavior, featuring the Focus Loop with `start_focus` and `complete_focus` primitives to actively prune the Raw Interaction History and consolidate learnings into a persistent Knowledge Block, utilizing Persistent Bash and String-Replace Editor tools.
- This architecture shifts from passive context retention to active compression, resulting in a "Sawtooth" context pattern where history grows during exploration and collapses during consolidation, managed autonomously by the LLM.
- Evaluation on SWE-bench Lite showed that aggressive, model-controlled compression achieved 22.7% total token reduction while maintaining identical task accuracy compared to the Baseline agent.

---

[Agents of Diffusion: Enhancing Diffusion Language Models with Multi-Agent Reinforcement Learning for Structured Data Generation (Extended Version)](http://arxiv.org/abs/2601.07152)

- AoD (Agents of Diffusion): introduces a multi-agent reinforcement learning framework for structured data generation, unifying the generative flexibility of DLMs with the reasoning capabilities of autoregressive LLM agents.
- The system frames structured text generation as an iterative alignment process where a Prompt Optimizer Agent collaborates with a Judge Cluster using natural language feedback to guide a frozen Diffusion LM Agent.
- This parameter-free optimization loop enables controllable, schema-consistent generation with high semantic novelty and structural fidelity across complex JSON benchmarks.

---

[A Large-Scale Study on the Development and Issues of Multi-Agent AI Systems](http://arxiv.org/abs/2601.07136)

- MAS Ecosystem Study: introduces the first large-scale empirical study of eight open-source Multi-Agent AI Systems (MAS) frameworks, analyzing over 42K commits and 4.7K issues to identify three distinct development profiles: sustained, steady, and burst-driven.
- The analysis reveals that perfective commits (40.8%) dominate development, prioritizing feature enhancement over corrective (27.4%) and adaptive (24.3%) maintenance, characteristic of a rapidly evolving domain.
- Issue analysis highlights bugs (22%), infrastructure (14%), and agent coordination (10%) as the most frequent concerns, underscoring the ecosystem's rapid growth momentum and inherent fragility fueled by the rising use of LLMs.

---

[Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework](http://arxiv.org/abs/2601.07122)

- CyberOps-Bots (Cyber Operation Bots): introduces a robust hierarchical multi-agent RL framework empowered by LLMs, featuring an upper-level LLM agent for global tactical planning and lower-level RL agents for localized atomic defense execution.
- The framework leverages the LLM's semantic understanding via a Perception module that converts structured network state into natural language, ensuring adaptability to dynamic network structures and scales.
- It supports Human-in-the-Loop (HITL) intervention and uses a memory mechanism (LTM/STM) to track multi-stage attack chains, enhancing robustness against evolving attack policies and intensity.

---

[Memory Poisoning Attack and Defense on Memory Based LLM-Agents](http://arxiv.org/abs/2601.05504)

- IOM-MS (Input/Output Moderation and Memory Sanitization): introduces a defense framework against memory poisoning attacks in LLM-Agents, utilizing a two-stage gating mechanism and a continuous trust scoring system for memory management.
- The Input/Output Moderation component uses static heuristics, keyword matching, and LLM-based semantic classification to prevent malicious entries from entering the execution pipeline and memory bank.
- Memory Sanitization employs trust-aware retrieval with temporal decay and pattern-based filtering, operating at both memory append and retrieval stages to mitigate long-term propagation of poisoned entries.

---

[Towards Verifiably Safe Tool Use for LLM Agents](http://arxiv.org/abs/2601.08012)

- MCP framework: introduces a process combining STPA (System-Theoretic Process Analysis) for hazard identification and IFC (Information Flow Control) to derive and enforce formal specifications on data flows and tool sequences, utilizing a capability-enhanced MCP.
- The enhanced MCP requires structured labels (Capabilities, Data Confidentiality, Trust Level) on tools and data, enabling an independent External Policy Engine to intercept and evaluate LLM tool calls against formal specifications (SPECs).
- The system uses a four-tier enforcement structure (Blocklist, Mustlist, Allowlist, Confirmation) to balance safety and flexibility, ensuring deterministic blocking of unacceptable flows while reducing reliance on user confirmation.

---

[LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback](http://arxiv.org/abs/2601.08003)

- LLM Review: introduces a peer-review-inspired multi-agent framework implementing Blind Peer Review to enhance creative writing by strategically constraining information flow, contrasting with interaction-maximizing baselines.
- The framework utilizes Drafting/Critique Agents with Persistent Personas across Compose, Review, and Revision Phases, ensuring agents revise independently without seeing peers' evolving drafts, thereby preserving divergent creative trajectories.
- Evaluation is conducted using the novel SciFi-100 Dataset and a Unified Evaluation Framework combining LLM-as-a-Judge scoring, human annotation, and rule-based novelty metrics, demonstrating that interaction structure can offset LLM model scale.

---

[Cost and accuracy of long-term graph memory in distributed LLM-based multi-agent systems](http://arxiv.org/abs/2601.07978)

- DMAS LTM Testbed: introduces a flexible testbed comparing the cost and accuracy of two long-term memory frameworks, mem0 and Graphiti, within a Distributed Multi-Agent System (DMAS) using the LOCOMO long-context benchmark.
- The testbed architecture includes a Coordinator (SLM), a Responder (LLM), and a Memory Agent, with traffic routed through Toxiproxy to simulate unconstrained and constrained network scenarios for measuring computational and financial metrics.
- Applying a statistical Pareto efficiency framework, the study identifies mem0 as the optimal choice, demonstrating significantly higher efficiency and comparable accuracy to Graphiti.

---

[Reward Modeling from Natural Language Human Feedback](http://arxiv.org/abs/2601.07349)

- RM-NLHF (Reward Modeling from Natural Language Human Feedback): introduces a framework to address outcome-process inconsistency in Generative Reward Models (GRMs) by leveraging natural language human critiques to derive a process reward signal.
- To overcome the scalability challenge of human critique annotation, the framework proposes an online MetaRM that learns to predict process reward from limited labeled data and generalizes to unlabeled data during training.
- The process reward is calculated using F1-based similarity between core arguments in human and GRM-generated critiques, which, when combined with outcome reward, significantly improves critique quality and preference accuracy.

---

[Segmental Advantage Estimation: Enhancing PPO for Long-Context LLM Training](http://arxiv.org/abs/2601.07320)

- SAE (Segmental Advantage Estimation): introduces a novel advantage estimation method that mitigates the high bias of Generalized Advantage Estimation (GAE) in sparse-reward RLVR by replacing token-level bootstrapping with segment-level bootstrapping.
- SAE partitions the LLM-generated sequence into semantically coherent segments using a probability-based heuristic, computing advantage estimates only at segment transition boundaries to filter out noise.
- Empirical results demonstrate that the proposed advantage estimator achieves superior performance, enhanced training stability, and higher correlation with the approximate ground-truth advantage compared to PPO baselines.

---

[I2B-LPO: Latent Policy Optimization via Iterative Information Bottleneck](http://arxiv.org/abs/2601.05870)

- I2B-LPO (Latent Policy Optimization via Iterative Information Bottleneck): introduces a framework for LLM reasoning that uses Entropy-driven Latent Branching via CVAE and PSA to generate diverse Latent Variables, followed by Information Bottleneck Regularization for trajectory filtering and self-reward during Policy Optimization.
- The method shifts exploration from statistical token perturbation to topological trajectory branching, triggering latent branching at high-entropy states to diversify reasoning paths.
- The dual-purpose IB mechanism ensures concise and informative exploration by balancing rationale compression and predictive power, achieving state-of-the-art accuracy and diversity.

---

#### 11th January 2026



[Dr. Zero: Self-Evolving Search Agents without Training Data](http://arxiv.org/abs/2601.07055)

- Dr. Zero (DeepResearch-Zero): introduces a data-free self-evolution framework leveraging an iterative Proposer-Solver paradigm, External Search Engine, and a Multi-turn Tool-Use Pipeline to autonomously enhance search and reasoning capabilities.
- The framework employs Hop-Grouped Relative Policy Optimization (HRPO) for the Proposer, guided by a Difficulty-Guided Reward, and Group Relative Policy Optimization (GRPO) for the Solver, eliminating the need for nested sampling and extensive training data.
- This zero-data approach establishes a continuously evolving curriculum, enabling search agents to match or surpass fully supervised baselines on complex multi-hop QA benchmarks.

---


[Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems](http://arxiv.org/abs/2601.07072)

- CEM Attack (Cross-Entropy Method Attack for Prefix Search): introduces an effective black-box indirect prompt injection (IPI) framework that decomposes malicious content into a Trigger Fragment (guarantees retrieval prefix) and an Attack Fragment (Dadv) (encodes malicious instructions), ensuring reliable retrieval across diverse LLM systems.
- The attack utilizes the CEM algorithm to construct a compact, optimized trigger fragment (x) that maximizes cosine similarity between the malicious text (x || Dadv) and the user query embedding, overcoming the retrieval bottleneck inherent in IPI.
- Evaluation across 11 benchmarks and 8 embedding models demonstrates near-perfect retrieval success, establishing IPI as a practical end-to-end threat in both RAG and agentic systems, even against strong LLMs like GPT-4o.

---

[LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents](http://arxiv.org/abs/2601.06973)

- PWMA (Private Working Memory Agents): introduces a novel architecture incorporating an explicit Private Working Memory to enable LLM agents to reliably perform Private State Interactive Tasks (PSITs) requiring hidden state maintenance.
- The architecture is implemented via Autonomous Memory Agents (LLM-driven tool use) and Memory Workflows (deterministic memory updates), utilizing Memory Tool Kits for state manipulation.
- Empirical results using a Self-Consistency Testing Protocol demonstrate that standard LLMs and retrieval-based baselines fail PSITs due to the lack of persistent private state, confirming the necessity of this architectural component.

---

[RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction](http://arxiv.org/abs/2601.06966)

- RealMem: introduces a benchmark for evaluating LLMs in long-term memory-driven interaction using a three-stage synthesis pipeline: Project Foundation Construction (Initializes context and structure), Multi-Agent Dialogue Generation (Simulates user-agent interactions), and Memory and Schedule Management (Updates system state and ensures consistency).
- The benchmark comprises over 2,000 cross-session dialogues across eleven realistic project scenarios, shifting evaluation from isolated fact retrieval to continuous project-centric memory utilization.
- The synthesis pipeline utilizes specialized agents, including Retrieval, Extraction, Schedule, and Deduplication Agents, to simulate the dynamic evolution of memory and ensure global logical coherence across sessions.

---

[mind_call: A Dataset for Mental Health Function Calling with Large Language Models](http://arxiv.org/abs/2601.06937)

- mind_call Dataset: introduces a synthetic function-calling dataset for mental health assistance, mapping natural language queries to structured API calls, including explicit reasoning and temporal normalization.
- The dataset samples include a user query, a query category, an explicit reasoning step, a normalized temporal parameter, and a target function derived from a standardized wearable health data schema.
- This resource supports research on intent grounding, temporal reasoning, and reliable function invocation in LLM-based mental health agents by covering diverse query types (explicit, implicit, behavioral, symptom-based, metaphorical).

---

[PenForge: On-the-Fly Expert Agent Construction for Automated Penetration Testing](http://arxiv.org/abs/2601.06910)

- PENFORGE: introduces an agentic framework for automated web-application penetration testing that dynamically constructs expert agents via a Meta-Planner orchestrator, which utilizes reconnaissance tools (EndpointScanner, WebPageReader, KnowledgeRetriever) and an Expert Agent Constructor to guide Sequential Attack Attempts.
- The Meta-Planner operates in two phases: Target Reconnaissance to gather context and rank vulnerabilities, followed by Sequential Attack Attempts where specialized agents (using AutoGPT and Claude-3.7-Sonnet LLM) are instantiated for exploitation.
- By combining automated reconnaissance with on-the-fly agent construction, the framework achieves a 30.0% exploit success rate on CVE-Bench in the zero-day setting, significantly improving the state-of-the-art baseline.

---

[Personality-Aware Reinforcement Learning for Persuasive Dialogue with LLM-Driven Simulation](http://arxiv.org/abs/2601.06877)

- Personality-Aware Reinforcement Learning (P-ARL) framework: introduces a persuasive dialogue system combining a Strategy-Oriented Interaction Framework, an LLM-driven Persuadee Simulator, an Utterance Strategy Classifier, a Personality Prediction Module, a Dueling Double DQN (D3QN) Model, a Maximal Marginal Relevance (MMR) Response Generator, and Composite Reward Predictors, designed to learn adaptive, personalized persuasion policies.
- The system formalizes persuasion as a Markov Decision Process, where the RL agent selects strategies conditioned on dialogue history and a dynamically predicted 81-dimensional user personality vector.
- Training is supported by LLM-driven simulation trajectories and optimized using a composite reward that includes a change-of-mind penalty to reduce post-agreement retractions and improve sustained behavioral outcomes.

---

[ET-Agent:](http://arxiv.org/abs/2601.06860)

- ET-Agent: introduces a training framework for calibrating LLM-based agents' Tool-Integrated Reasoning (TIR) behavior using a Self-Evolving Data Flywheel and a two-phase Behavior Calibration Training framework.
- The Self-Evolving Data Flywheel iteratively refines trajectories via Correct Reasoning Enhancement and Incorrect Reasoning Reflection to generate diverse, high-quality training data.
- The Behavior Calibration Training uses Action Space Exploration Fine-tuning followed by Iterative Behavior Calibration RL, which alternates between Group-wise Pareto Sampling and Curriculum RL Training, to steer the agent toward optimal behavioral patterns.

---

[CHASE: LLM Agents for Dissecting Malicious PyPI Packages](http://arxiv.org/abs/2601.06838)

- CHASE (Collaborative Hierarchical Agents for Security Exploration): introduces a high-reliability multi-agent architecture using a Plan-and-Execute Workflow, specialized Worker Agents (Deobfuscator, Web Researcher), and Deterministic Security Tools to dissect malicious PyPI packages.
- The system employs a Supervisor Agent to maintain global analysis state and dynamically adjust the plan, mitigating LLM weaknesses like hallucination and context confusion.
- By combining specialized LLM agents with robust tools, the architecture achieves 98.4% recall and a 0.08% false positive rate, suitable for operational package screening.

---

[AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents](http://arxiv.org/abs/2601.06818)

- AgentHallu: introduces the novel task of automated hallucination attribution for LLM-based agents, supported by the AgentHallu Benchmark, which features multi-step Trajectory data annotated with Hallucination Judgment, Responsible Step Localization, and Causal Explanation based on a Hallucination Taxonomy.
- The benchmark evaluates LLM Evaluators on their ability to pinpoint the earliest Interaction Unit (thought action observation triplet) in a sequential agent workflow that causes the initial divergence leading to an incorrect final answer.
- Evaluation results across 13 leading LLMs reveal that step localization remains highly challenging, especially for tool-use hallucinations, underscoring the need for more robust and transparent agentic systems.

---

[No More Stale Feedback: Co-Evolving Critics for Open-World Agent Learning](http://arxiv.org/abs/2601.06794)

- ECHO (Evolving Critic for Hindsight-Guided Optimization): introduces a framework that jointly optimizes the LLM Policy Model ($P_\theta$) and Critic Model ($C_\psi$) through a synchronized co-evolutionary loop to mitigate critic staleness in on-policy RL.
- The approach utilizes a Cascaded Evolutionary Rollout mechanism, consisting of multi-view diagnosis and conditional refinement, to generate group-structured trajectories for stable and sample-efficient group-relative advantage estimation.
- Training stability is further enhanced by a Saturation-Aware Gain Shaping objective, which calculates the critic reward to emphasize last-mile improvements near the performance ceiling.

---

[From Text to Simulation: A Multi-Agent LLM Workflow for Automated Chemical Process Design](http://arxiv.org/abs/2601.06776)

- Multi-Agent LLM Workflow: introduces an end-to-end automated chemical process design system that transforms textual specifications into executable simulation configurations using four specialized agents and Enhanced MCTS.
- The workflow bridges the gap between abstract process descriptions and validated simulation configurations by integrating LLM capabilities with direct, bidirectional communication with industrial simulation software.
- This approach significantly reduces design time and achieves a high simulation convergence rate by systematically exploring the design space and maintaining feasibility constraints.

---

[Thinking with Deltas: Incentivizing Reinforcement Learning via Differential Visual Reasoning Policy](http://arxiv.org/abs/2601.06801)

- Thinking with Deltas (Differential Visual Reasoning Policy): introduces a novel framework to bridge the perception-reasoning decoupling in multimodal Reinforcement Learning with Verifiable Rewards (RLVR) by leveraging self-supervised visual triplets (Invariant, Decremental, Incremental views) to enforce intrinsic visual grounding.
- The DVRP optimizes a differential policy that maximizes reasoning divergence from masked inputs (Visual Sensitivity) while minimizing divergence from noise-perturbed inputs (Visual Robustness), thereby compelling the model to attend to visual evidence rather than linguistic priors.
- This approach significantly outperforms state-of-the-art RLVR methods on both general mathematical reasoning and specialized medical diagnostics benchmarks, enhancing visual perception capabilities without external tools or annotations.

---

#### 10th January 2026

[IDRBench: Interactive Deep Research Benchmark](http://arxiv.org/abs/2601.06676)

- IDRBench (Interactive Deep Research Benchmark): introduces a modular multi-agent research framework featuring Planning, Research Loop, and Generation stages, augmented by a Clarification/User Feedback interaction mechanism involving Evaluator, Questioner, and User Simulator components, designed to systematically evaluate interactive deep research capabilities of LLMs.
- The benchmark assesses interaction benefits (quality, coverage, intent alignment) and costs (turns, tokens) using a scalable reference-grounded User Simulator and an interaction-aware evaluation suite.
- Experiments across state-of-the-art LLMs demonstrate that interaction consistently improves research quality and robustness, often outweighing differences in raw model capacity while revealing trade-offs in efficiency.

---

[Agentic AI Empowered Intent-Based Networking for 6G](http://arxiv.org/abs/2601.06640)

- HMA-IBN: introduces a hierarchical multi-agent framework where LLM agents autonomously translate natural language operational intents into executable network slice configurations.
- The architecture employs an Orchestrator Agent coordinating RAN and Core Specialist Agents through ReAct-style reasoning grounded in structured network state representations.
- Experimental validation shows that the system outperforms monolithic agents and rule-based systems, confirming that iterative refinement and prompt engineering are critical architectural components.

---

[MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis](http://arxiv.org/abs/2601.06636)

- ECR-Agent (Evidence-based Causal Reasoning Agent): introduces an agentic framework to mitigate the Einstellung Effect in medical LLMs by aligning reasoning with Evidence-Based Medicine standards through structured causal inference and knowledge evolution.
- The framework utilizes Dynamic Causal Inference (DCI) for structured diagnosis via dual-pathway perception and three-level causal graph reasoning, complemented by Critic-Driven Graph and Memory Evolution (CGME) for accumulating clinical experience in an Exemplar Base and evolving Illness Graphs.
- The paper also introduces MedEinst, a counterfactual benchmark with 5,383 paired clinical cases designed to expose the Einstellung Effect, revealing that frontier LLMs achieve high baseline accuracy but suffer severe bias trap rates.

---

[CEDAR: Context Engineering for Agentic Data Science](http://arxiv.org/abs/2601.06606)

- CEDAR (Context Engineering for Data science with Agent Routing): introduces an agentic system for automating data science tasks, utilizing an Orchestrator agent (Routes requests), Text agent (Generates explanations), and Code agent (Generates Python code), driven by a Structured NL prompt (DS-specific input fields) and supported by History rendering (Compact context summary), Tool use (Schema-driven communication), Local data execution (Data stays local), Iterative code execution (Fault tolerance), and a Docker environment (Containerized safety).
- The system generates a human-readable workflow as an enumerated sequence of interleaved plan (Text agent) and executable Python code blocks (Code agent), mimicking a Jupyter notebook structure.
- Effective context engineering ensures that only aggregate statistics and instructions are passed to the LLMs, maintaining data locality and alleviating context length limitations inherent in multi-step DS solutions.

---

[DRAGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization](http://arxiv.org/abs/2601.06502)

- DRAGON (Decomposition and Reconstruction Agents Guided OptimizatioN): introduces a novel framework combining metaheuristics and LLM reasoning, featuring a Decomposition Agent (identifies suboptimal regions), a Reconstruction Agent (solves localized subproblems), a Compress Step (reduces local data size), an Integration Step (reintegrates local solution), an Adaptive Experience Memory (stores past experiences), a Global Solution (current optimization state), Metadata (problem specific parameters), and LLM Agents (perform reasoning tasks).
- The framework iteratively refines a global solution by autonomously identifying high-potential regions, decomposing large-scale COPs into manageable subproblems, and locally optimizing them under explicit constraints.
- DRAGON addresses LLM scalability and context length limitations in large-scale optimization by coupling symbolic reasoning with heuristic search, achieving strong performance across routing and packing benchmarks.

---

[Bi-Mem: Bidirectional Construction of Hierarchical Memory for Personalized LLMs via Inductive-Reflective Agents](http://arxiv.org/abs/2601.06490)

- Bi-Mem (Bidirectional Construction of Hierarchical Memory): introduces an agentic framework ensuring hierarchical memory fidelity through bidirectional construction, utilizing an inductive agent for bottom-up formulation and a reflective agent for top-down calibration.
- The framework mitigates local aggregated memory misalignment with the user's global persona, a common issue caused by conversational noise and hallucinations in hierarchical systems, enabling LLMs to generate preference-aligned suggestions.
- Coherent memory recall is facilitated by an associative retrieval mechanism that uses spreading activation to bridge abstract persona traits and concrete conversational facts, significantly enhancing QA accuracy in personalized conversational tasks.

---

[ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking](http://arxiv.org/abs/2601.06487)

- ArenaRL: introduces a reinforcement learning paradigm that shifts from pointwise scalar scoring to intra-group relative ranking for scaling RL in open-ended LLM agent tasks.
- The framework employs a process-aware pairwise evaluation mechanism using a multi-level rubric and constructs an intra-group adversarial arena with a tournament-based ranking scheme.
- To ensure efficiency, ArenaRL utilizes a seeded single-elimination tournament topology, achieving linear $O(N)$ complexity while maintaining high-accuracy advantage estimation fidelity.

---

[ConSensus: Multi-Agent Collaboration for Multimodal Sensing](http://arxiv.org/abs/2601.06453)

- ConSensus: introduces a training-free multi-agent collaboration framework that decomposes multimodal sensing tasks into specialized modality agents and aggregates interpretations via a hybrid fusion mechanism.
- The hybrid fusion mechanism balances semantic aggregation (cross-modal reasoning) and statistical consensus (majority voting) to ensure robustness against LLM knowledge bias and sensor failure.
- The framework employs a single-round structured fusion protocol, achieving high accuracy while reducing fusion token cost by 12.7x compared to iterative multi-agent debate methods.

---

[Can a Unimodal Language Agent Provide Preferences to Tune a Multimodal Vision-Language Model?](http://arxiv.org/abs/2601.06424)

- LLM Preference Tuning Framework: introduces a method where a unimodal LLM Agent provides preference feedback to fine-tune a multimodal VLM using DPO, enabling the VLM to generate task-relevant textual descriptions for downstream reasoning.
- The VLM first generates diverse video descriptions, which the unimodal LLM Agent ranks to create a preference dataset, subsequently used by DPO to optimize the VLM's output generation.
- This approach significantly enhances VLM descriptions, leading to robust performance improvements (up to 13% accuracy gain) on multimodal social reasoning tasks like sarcasm and humor detection.

---

[Lightweight Yet Secure: Secure Scripting Language Generation via Lightweight LLMs](http://arxiv.org/abs/2601.06419)

- PSSec: introduces a framework combining automated data synthesis and two-stage fine-tuning (SFT/RL) to train lightweight LLMs for secure PowerShell script generation, analysis, and repair.
- The data synthesis pipeline utilizes a frontier LLM and a static analyzer to generate large-scale structured training triplets (insecure script, violation analysis, repaired script) for domain specialization.
- The resulting PSSec-trained lightweight LLMs match or surpass large general-purpose LLMs on PowerShell security tasks while reducing inference cost by over an order of magnitude, evaluated using the SecGenEval-PS benchmark.

---

[Structured Episodic Event Memory](http://arxiv.org/abs/2601.06411)

- SEEM (Structured Episodic Event Memory): introduces a hierarchical framework that synergizes the Episodic Memory Layer (EML) for dynamic narrative progression and the Graph Memory Layer (GML) for static relational facts, transforming interaction streams into structured Episodic Event Frames (EEFs).
- The system employs an agentic associative fusion mechanism and Reverse Provenance Expansion (RPE) during hybrid retrieval to reconstruct coherent narrative contexts from fragmented evidence, anchored by provenance pointers.
- This dual-layer architecture enhances the long-term reasoning capabilities of LLM agents by maintaining superior narrative coherence and logical consistency across complex interactions.

---

[Value of Information: A Framework for Human-Agent Communication](http://arxiv.org/abs/2601.06407)

- VoI (Value of Information): introduces a decision-theoretic framework enabling LLM agents to dynamically decide whether to clarify or commit by calculating the expected utility gain (VoI) against the communication cost $c$.
- The framework operationalizes the Clarify-or-Commit Policy by maximizing the NetVoI, which explicitly balances Query Ambiguity, Task Risk, and Cognitive Load.
- This inference-time method is parameter-free and robust, consistently matching or exceeding manually-tuned baselines across diverse tasks like medical diagnosis and flight booking.

---

[HiMem: HIERARCHICAL LONG-TERM MEMORY FOR LLM LONG-HORIZON AGENTS](http://arxiv.org/abs/2601.06377)

- HiMem (Hierarchical Long-Term Memory): introduces a hierarchical long-term memory framework for LLM long-horizon agents, integrating Episode Memory (fine-grained events) and Note Memory (abstracted knowledge) via Semantic Linkage.
- Memory construction uses Topic-Aware Event-Surprise Dual-Channel Segmentation for Episode Memory and a Multi-Stage Information Extraction pipeline for Note Memory, ensuring cognitive consistency and efficiency.
- The system supports best-effort hierarchical retrieval and incorporates conflict-aware Memory Reconsolidation, which enables continual memory self-evolution based on retrieval feedback and conflict detection.

---

[DemMA: Dementia Multi-Turn Dialogue Agent with Expert-Guided Reasoning and Action Simulation](http://arxiv.org/abs/2601.06373)

- DemMA (Dementia Multi-Turn Dialogue Agent): introduces an expert-guided dementia dialogue agent for high-fidelity multi-turn patient simulation, utilizing a Persona Formation module, a Multi-agent Dialogue Generation Pipeline, and CoT Distillation Training.
- The framework constructs clinically grounded dementia personas by modeling intrinsic cognitive decline and extrinsic nonverbal behaviors via explicit Action Labels (motion, facial expressions, sound).
- CoT distillation trains a single LLM to jointly generate reasoning traces, patient utterances, and aligned behavioral actions in one forward pass, enabling efficient, low-latency deployment.

---

[Modeling Descriptive Norms in Multi-Agent Systems: An Auto-Aggregation PDE Framework with Adaptive Perception Kernels](http://arxiv.org/abs/2601.06557)

- Auto-Aggregation PDE Framework (A-APDE): simulates descriptive norm dynamics using a PDE Transport Equation extended with an External Potential Field and a Generalized Non-local Gradient derived from a Perceptual Kernel Function, modeling agent interactions within a Multi-Agent System.
- The framework characterizes collective descriptive norms by modeling opinion popularity as a continuous distribution, where agents' Subjective Individual Norm Perception (SINP) adapts to match the Objective Collective Norm (OBJ) derived from real-world medical data using Gaussian Mixture Models.
- Experiments on COVID-19 medical data demonstrate the model's ability to capture both top-down convergence (guided by clinical guidelines) and bottom-up norm violation and restructuring (driven by medical facts/variants).

---

[Plasticity vs. Rigidity: The Impact of Low-Rank Adapters on Reasoning on a Micro-Budget](http://arxiv.org/abs/2601.06677)

- RLVR-LoRA: investigates the plasticity of small LLMs ($\leq 1.5$B) for mathematical reasoning under a strict 24-hour "micro-budget" constraint using Low-Rank Adaptation and Reinforcement Learning with Verifiable Rewards.
- The study demonstrates that high-rank LoRA adapters ($r=256$) are critical for inducing plasticity in generalist instruction-tuned models, enabling them to maximize reward and actively elongate reasoning chains.
- Conversely, specialized, heavily optimized models exhibit rigidity and performance collapse when subjected to noisy, low-budget RL updates, suggesting that the most efficient path is to catalyze generalists rather than refine experts.

---

[Advanced Manufacturing with Renewable and Bio-based Materials: AI/ML workflows and Process Optimization](http://arxiv.org/abs/2601.10382)

- AI/ML Workflows and SDLs: introduces the integration of artificial intelligence and machine learning optimization protocols with Self-Driving Laboratories for advanced manufacturing of bio-based materials.
- The approach leverages a Digital Twin architecture to connect the physical world (AM processes, sensors) with virtual models, enabling real-time monitoring, control, and autonomous optimization of structure, composition, processing, and properties (SCPP).
- LLMs and various Deep Learning architectures are utilized to augment SDL development, accelerate material synthesis, and optimize 3D printing parameters for high-performance, sustainable, and defect-free products.

---

#### 9th January 2026

[Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards](http://arxiv.org/abs/2601.06021)

- C-GRPO (Citation-aware Group Relative Policy Optimization): introduces a mixed-reward RL algorithm for training robust deep search agents by combining traditional binary outcome rewards with fine-grained Citation-aware Rubric Rewards (CaRR).
- CaRR decomposes complex multi-hop questions into verifiable single-hop rubrics, requiring agents to identify hidden entities, support them with citations, and construct complete evidence chains.
- The algorithm assigns an additional weighted rubric reward only to trajectories achieving the correct outcome, promoting comprehensive, evidence-grounded reasoning and discouraging shortcut exploitation.

---


[MMUEChange: A Generalized LLM Agent Framework for Intelligent Multi-Modal Urban Environment Change Analysis](http://arxiv.org/abs/2601.05483)

- MMUEChange (A Generalized LLM Agent Framework for Intelligent Multi-Modal Urban Environment Change Analysis): introduces a multi-modal LLM agent framework that integrates heterogeneous urban data via a Modality Controller (routing alignment aggregation) and a modular Toolkit (modality-specific processing tools) to perform complex urban change analysis.
- The framework operationalizes urban change analysis using a hierarchical "what-where-why" problem set, moving from descriptive observations to spatial reasoning and causal interpretation.
- The Modality Controller ensures robust analysis by performing user demand alignment and systematic data alignment (geo-information and unique identification) across diverse modalities, mitigating LLM hallucination.

---


[Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks](http://arxiv.org/abs/2601.06007)

- Prompt Caching Strategies: introduces a comprehensive evaluation of prompt caching across major LLM providers using KV Cache, comparing Full Context Caching, System Prompt Only Caching, and Exclude Tool Results Caching, controlled via UUIDs, on the DeepResearchBench multi-turn agentic benchmark.
- Strategic cache boundary control, particularly System Prompt Only Caching, consistently outperforms naive Full Context Caching by avoiding dynamic content (Tool Calls/Tool Results) that can paradoxically increase latency.
- Prompt caching significantly reduces API costs by 45-80% and improves Time to First Token (TTFT) latency by 13-31% across tested LLM providers (OpenAI, Anthropic, Google) for long-horizon agentic tasks.

---

[DISTILLING FEEDBACK INTO MEMORY-AS-A-TOOL](http://arxiv.org/abs/2601.05960)

- DFMT (Distilling Feedback into Memory-as-a-Tool): introduces a framework that amortizes inference-time reasoning costs by converting transient critiques into persistent, retrievable guidelines stored in a File-based Memory (M).
- The LLM Agent actively manages the memory using specific Tool Calls (ls, read, write, edit) to retrieve relevant "lessons learned" before generating a response and to distill new feedback.
- This approach enables the LLM to synthesize abstract principles from episodic errors, achieving performance comparable to costly self-critique pipelines while drastically reducing inference cost.

---

[Can We Predict Before Executing Machine Learning Agents?](http://arxiv.org/abs/2601.05930)

- FOREAGENT: introduces a hybrid autonomous ML agent, utilizing the Predict-then-Verify Loop (iterative structure) to decouple exploration from execution, leveraging an Implicit World Model (predictive LLM filter) for High-Volume Generation (candidate proposal) and Confidence-Gated Pairwise Selection (high-certainty filtering), followed by Verification Execution (physical verification).
- The framework addresses the severe Execution Bottleneck in traditional Generate-Execute-Feedback paradigms by substituting hours of physical latency with seconds of logical inference.
- By internalizing execution priors via a Verified Data Analysis Report (semantic input grounding), the agent achieves a 6x acceleration in convergence and a +6% performance gain over execution-based baselines.

---

[Agentic LLMs as Powerful Deanonymizers: Re-identification of Participants in the Anthropic Interviewer Dataset](http://arxiv.org/abs/2601.05918)

- LPRA (LLM-Powered Re-identification Attack): introduces a low-effort, scalable re-identification attack using web-augmented LLM agents to link interview transcripts from the Anthropic Interviewer dataset to specific scientific publications.
- The attack employs a two-step procedure: first, a non-thinking model filters transcripts mentioning published work, followed by a thinking model agent using web search to rank candidate publications based on project descriptions.
- This methodology successfully recovered specific publications for 6 out of 24 targeted scientist transcripts, demonstrating that LLM safeguards can be bypassed by breaking down the attack into benign tasks.

---

[TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents](http://arxiv.org/abs/2601.05899)

- TowerMind: introduces a lightweight, multimodal Tower Defense (TD) game environment and benchmark for evaluating LLMs' long-term planning and decision-making, featuring a multimodal observation space, hybrid action space, and hallucination evaluation.
- The environment significantly reduces computational demands compared to existing RTS benchmarks like StarCraft II, supporting pixel-based, textual, and structured game-state observations for multimodal LLMs.
- Evaluation results across five benchmark levels reveal a substantial performance gap between LLMs and human experts, highlighting LLM limitations in planning validation, multifinality in decision-making, and inefficient action use.

---

[Cybersecurity AI: A Game-Theoretic AI for Guiding Attack and Defense](http://arxiv.org/abs/2601.05887)

- G-CTR (Generative Cut-the-Rope): introduces a game-theoretic guidance layer that extracts attack graphs from agent context, computes Nash equilibria, and feeds a concise digest back into the LLM loop guiding the agent's actions.
- This closed-loop architecture integrates three phases: AI Analysis, Guidance (Digest Generation), and Agent Execution (ReAct), operating in parallel to minimize computational overhead.
- Empirical results show that LLM-based strategic guidance significantly reduces behavioral variance and increases success rates in penetration testing and Attack-and-Defense exercises.

---

[ToolGym: an Open-world Tool-using Environment for Scalable Agent Testing and Data Curation](http://arxiv.org/abs/2601.06328)

- ToolGym: introduces an open-world tool-using environment built on 5,571 curated tools, featuring a Task Creation Engine (Synthesizes long-horizon workflows) and a State Controller (Injects failures, corrupts results) to test LLM agent robustness.
- The environment employs a Planner-Actor decomposition framework, where the Planner handles deliberate reasoning and goal decomposition, and the Actor executes step-wise actions via a ReAct loop.
- ToolGym serves as a scalable benchmark and data engine, enabling the collection of high-quality trajectories for fine-tuning LLMs, demonstrating superior data efficiency compared to baselines.

---

[Beyond BeautifulSoup: Benchmarking LLM-Powered Web Scraping for Everyday Users](http://arxiv.org/abs/2601.06301)

- LLM-Powered Web Scraping Benchmark (LLM-PWSB): introduces a systematic evaluation of LLM-assisted Scripting (LAS) and End-to-End LLM Agent (ELA) workflows across 35 websites with varying security tiers.
- LAS involves LLMs generating traditional scraping code (BeautifulSoup/Scrapy) which the user manually executes and refines, while ELA uses production agents (Claude/Simular.ai) for autonomous navigation and tool use.
- The benchmark quantifies the accessibility-reliability trade-off, showing that ELA excels on complex sites (authentication/CAPTCHA) while LAS is faster and more efficient for static HTML content.

---

[Automated QoR improvement in OpenROAD with coding agents](http://arxiv.org/abs/2601.06268)

- AuDoPEDA (Autonomous Documentation and Planning system for EDA codebases): introduces a closed-loop LLM framework for EDA code changes, integrating graph-structured documentation, literature-grounded planning, and agentic execution with QoR feedback.
- The system first constructs a Code Graph (G) and machine-usable Docmaker Cards ($C_{repo}$) from the OpenROAD repository to enable structure-aware retrieval and planning.
- A Codex-class agent executes the localized granular plans, applying diffs, running OpenROAD flows, and utilizing a QoR feedback loop to achieve measurable improvements in routed wirelength and effective clock period.

---

[EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis](http://arxiv.org/abs/2601.05808)

- EnvScaler: introduces an automated framework for scalable tool-interactive environment synthesis, comprising SkelBuilder (constructs environment skeletons) and ScenGenerator (generates task scenarios).
- SkelBuilder automates environment construction through topic mining, programmatic logic modeling, and a dual-agent assessment loop to create diverse, executable environment logic and tool interfaces.
- ScenGenerator ensures task relevance and solvability by deriving challenging tasks from generated initial states and producing rule-based validation functions for trajectory verification and reward scoring, supporting SFT and RL training for LLM agents.

---

[VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit](http://arxiv.org/abs/2601.05755)

- VIGIL (Verifiable Intent-Grounded Interaction Loop): introduces a verify-before-commit framework to secure LLM agents against tool stream injection, utilizing the Intent Anchor, Perception Sanitizer, Speculative Reasoner, Grounding Verifier, and Validated Trajectory Memory.
- The architecture shifts the defensive paradigm from restrictive isolation by decoupling reasoning exploration from irreversible action via intent-grounded verification.
- The framework significantly reduces the Attack Success Rate (ASR) on the SIREN benchmark while maintaining high Utility Under Attack (UA), resolving the rigidity-utility trade-off inherent in static defenses.

---

[HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](http://arxiv.org/abs/2601.05656)

- HAG (Hierarchical Agent Generation framework): introduces a two-stage process utilizing the World Knowledge Model (infers conditional probabilities) to construct a Topic-Adaptive Distribution Tree (models joint distribution), followed by Grounded Instantiation (retrieves real users) and Agentic Augmentation (synthesizes missing personas) to generate sociologically grounded agent populations.
- The framework addresses limitations in existing LLM-based methods by achieving macro-level distribution alignment via the hierarchical tree structure and ensuring micro-level consistency through grounding in real-world data (WVS Database).
- HAG is evaluated using the comprehensive PACE framework, which quantifies generation quality based on Population Alignment (statistical fidelity) and Sociological Consistency (semantic rationality).

---

[LLM-DMD: Large Language Model-based Power System Dynamic Model Discovery](http://arxiv.org/abs/2601.05632)

- LLM-DMD (Large Language Model-based Dynamic Model Discovery): integrates LLM reasoning and code synthesis with gradient-based evaluation across two sequential loops (DE and AE) to jointly identify governing differential equations and algebraic constraints for power systems.
- The framework employs an LLM-based Modeling Agent (M-Agent) guided by task contracts and in-context examples to generate executable DAE skeletons without predefined basis functions.
- An island-based evolutionary strategy manages candidate models, while a variable extension mechanism triggers the adaptive introduction of missing variables upon evaluation stagnation.

---

[Conformity Dynamics in LLM Multi-Agent Systems: The Roles of Topology and Self-Social Weighting](http://arxiv.org/abs/2601.05606)

- CD-MAS (Conformity Dynamics in Multi-Agent Systems): introduces a systematic study of conformity dynamics in LLM-based MAS using a misinformation detection task, comparing Centralized Aggregation and Distributed Consensus topologies, governed by a Confidence-Normalized Pooling Rule and a Self-Social Weighting Parameter.
- The Confidence-Normalized Pooling Rule uses the self-social weighting parameter ($\alpha$) to balance an agent's self-reliance against the confidence-weighted influence of its neighbors.
- Centralized structures prioritize rapid decisions but are sensitive to hub competence, while distributed structures promote robust consensus but risk high-confidence error cascades in dense networks.

---

[Crisis-Bench: Benchmarking Strategic Ambiguity and Reputation Management in Large Language Models](http://arxiv.org/abs/2601.05570)

- Crisis-Bench: introduces a dynamic, multi-agent Partially Observable Markov Decision Process (POMDP) simulation to evaluate LLMs' strategic communication and reputation management capabilities in high-stakes corporate crises.
- The framework utilizes a Dual-Knowledge Architecture to rigorously track Private and Public narrative states, enforcing information asymmetry crucial for testing Theory of Mind and strategic withholding.
- Performance is quantified using the Adjudicator-Market Loop, which translates qualitative PR strategies into a simulated stock price to measure the trade-off between ethical alignment and professional utility.

---

[LIDL: LLM Integration Defect Localization via Knowledge Graph-Enhanced Multi-Agent Analysis](http://arxiv.org/abs/2601.05539)

- LIDL (LLM Integration Defect Localization): introduces a multi-agent framework for localizing LLM integration defects, including a Code Knowledge Graph Constructor (builds CodeKG), a Defect Analyzer (fuses evidence), and a Context-aware Validator (applies counterfactual reasoning).
- The CodeKG captures cross-layer dependencies across heterogeneous artifacts (source code, prompts, configuration files) and annotates them with LLM-specific roles (LLM_PROMPT, LLM_CALL, LLM_CONFIG, LLM_TOOL, LLM_MEMORY).
- The framework significantly outperforms five state-of-the-art baselines, achieving a 64.1% improvement in Top-3 accuracy while reducing localization cost by 92.5%.

---

[Task Cascades for Efficient Unstructured Data Processing](http://arxiv.org/abs/2601.05536)

- TC (Task Cascades): generalizes model cascades by varying the model, operation (original or surrogate), and document fraction at each stage to minimize inference cost while meeting accuracy targets.
- The framework uses an LLM agent to iteratively generate simplified surrogate operations and employs document restructuring to prioritize relevant content, enabling fractional processing by cheaper proxy models.
- Optimal cascade construction is NP-HARD, motivating a greedy assembly algorithm combined with a statistical threshold adjustment procedure to provide accuracy guarantees.

---

[CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems](http://arxiv.org/abs/2601.05520)

- CHisAgent: introduces a multi-agent LLM framework for historical event taxonomy construction, including the Inducer (Bottom-up induction), Expander (Top-down generalization), and Enricher (Evidence integration) agents.
- The framework decomposes taxonomy construction into three specialized stages, leveraging agents like the Extractor, Classifier, Generator, Merger, Judger, and Conceptualizer to process the Twenty-Four Histories corpus.
- The resulting taxonomy covers core aspects of ancient Chinese political, military, diplomatic, and social life, demonstrating improved structural coherence and faithfulness through external knowledge integration.

---

[EvidFuse: Writing-Time Evidence Learning for Consistent Text-Chart Data Reporting](http://arxiv.org/abs/2601.05487)

- EvidFuse: introduces a training-free multi-agent framework enabling writing-time text-chart interleaved generation for data reports, composed of the Data-Augmented Analysis Agent (A') and the Real-Time Evidence Construction Writer (W).
- The Real-Time Evidence Construction Writer (W) drafts the report and issues fine-grained visualization requests to the Data-Augmented Analysis Agent (A'), which returns grounded visual evidence and captions that are immediately injected into the context.
- This writing-time evidence construction mechanism ensures strict chart-text consistency and enables deeper, decision-oriented analysis by dynamically expanding the evidence space as the narrative evolves.

---

[STELP: Secure Transpilation and Execution of LLM-Generated Programs](http://arxiv.org/abs/2601.05467)

- STELP (Secure Transpiler and Executor of LLM-Generated Program): introduces a secure code execution engine that uses a transpiler paradigm to validate and transform potentially unsafe LLM-generated code into secure, executable code, incorporating safeguards that prevent faulty or malicious code blocks from running.
- The architecture intercepts LLM-generated code, processes it via the AST Processor against user-configured safety policies (Safe Grammar Config and Tools Config), and then executes the secured version using the Safe Code Generator and Executor.
- A critical Feedback Generator component provides structured logs and natural language guidance to the CodeGen LLM, enabling autonomous code repair and regeneration in multi-agent systems.

---

[Agentic AI and Machine Learning for Accelerated Materials Discovery and Applications](http://arxiv.org/abs/2601.09027)

- Agentic AI/ML for Accelerated Materials Discovery: reviews the progress of AI agents and ML techniques applied to materials science, focusing on efficient discovery, core concepts, LLMs, and applications in flow chemistry, biosensors, and batteries.
- The core architecture integrates Agentic AI (handling hypothesis generation and autonomous experiments) with Machine Learning (performing predictive modeling and pattern recognition), linked by Reinforcement Learning for optimization.
- The review highlights the role of LLMs for knowledge mining and translating complex data into actionable protocols, alongside generative models for inverse design of novel material structures.

---

[From Off-Policy to On-Policy: Enhancing GUI Agents via Bi-level Expert-to-Policy Assimilation](http://arxiv.org/abs/2601.05787)

- BEPA (Bi-Level Expert-to-Policy Assimilation): introduces a bi-level assimilation framework to enhance end-to-end GUI agents by turning static, mismatched expert traces into dynamic, policy-aligned guidance via LEVEL-1 (Initializes guidance pool) and LEVEL-2 (Maintains dynamic alignment).
- LEVEL-1 mitigates structural mismatch and distribution shift by re-rolling expert plans under the base policy to produce policy-compatible trajectories, creating the initial guidance pool.
- LEVEL-2 dynamically maintains a self-aligned per-task cache, refreshing it with the agent's own successful rollouts and conditionally injecting cached guidance into GRPO updates only upon total on-policy failure.

---

[Orchestrating Tokens and Sequences: Dynamic Hybrid Policy Optimization for RLVR](http://arxiv.org/abs/2601.05607)

- DHPO (Dynamic Hybrid Policy Optimization): introduces a unified approach for RLVR that integrates token-level and sequence-level importance ratios via a hybrid mixing mechanism within a single clipped surrogate objective.
- The core mechanism uses a Hybrid Importance Ratio, which is a convex combination of the fine-grained Token-level Ratio and the stable Sequence-level Ratio, controlled by a dynamic Mixing Weight.
- To stabilize training, DHPO employs Branch-Specific Clipping, constraining each ratio component within separate trust regions, and offers two mixing variants: Averaged Mixing and Entropy-guided Mixing.

---

#### 8th January 2026

[MineNPC-Task: Task Suite for Memory-Aware Minecraft Agents](http://arxiv.org/abs/2601.05215)

- MINENPC-TASK (MineNPC-Task): introduces a practical benchmark and evaluation harness for memory-aware, mixed-initiative LLM agents in Minecraft, featuring Intent Routing, Planning & Clarification, Code Gen & Review, Execution, Evaluation, Feedback & Repair, and World & Social Memory.
- The framework uses a model-agnostic Plan-Clarify-Act-Judge loop, constraining perception and action to public Mineflayer APIs under a bounded-knowledge policy, and judging outcomes solely from in-world evidence.
- The benchmark, derived from 44 expert co-play tasks (216 subtasks), revealed a 33% subtask failure rate for GPT-4o, highlighting common breakdowns in code execution, inventory handling, and referencing.

---

[Internal Representations as Indicators of Hallucinations in Agent Tool Selection](http://arxiv.org/abs/2601.05214)

- IRHD (Internal Representation Hallucination Detector): introduces a computationally efficient framework that detects tool-calling hallucinations in real-time by leveraging LLM's internal representations during the same forward pass used for generation.
- The approach uses an Unsupervised Training Pipeline to generate labeled data by masking ground-truth tool calls and training a lightweight Classifier on contextualized embeddings derived from Feature Extraction.
- By operating inline with generation using only final-layer representations, the method achieves strong detection performance (up to 86.4% accuracy) with minimal computational overhead, enabling reliable agent deployment.

---

[SIMUAGENT: AN LLM-BASED SimulinK MODELING ASSISTANT ENHANCED WITH REINFORCEMENT LEARNING](http://arxiv.org/abs/2601.05187)

- SimuAgent: introduces an LLM-powered plan-execute agent framework for Simulink modeling, combining a lightweight Python dictionary representation, a local testing environment, and the Reflection-GRPO (ReGRPO) training algorithm.
- The framework uses a compact Python dictionary format for Simulink models, dramatically reducing token consumption and enabling rapid structural validation and parameter tuning via a local Python test harness.
- ReGRPO enhances policy optimization by incorporating self-reflection traces derived from environmental feedback, accelerating convergence and improving robustness in complex, sparse-reward Simulink tasks.

---

[CoV: Chain-of-View Prompting for Spatial Reasoning](http://arxiv.org/abs/2601.05172)

- CoV (Chain-of-View Prompting): introduces a training-free, test-time reasoning framework for embodied question answering that transforms a VLM into an active viewpoint reasoner via a coarse-to-fine exploration process.
- The framework operates in two stages, beginning with the View Selection Agent filtering input view frames to identify question-aligned anchor views, followed by the CoV Agent executing an iterative action-reasoning loop using discrete camera actions.
- This viewpoint-aware strategy dynamically adjusts the perspective to gather discriminative visual evidence, resolving spatial ambiguities and achieving significant performance gains on EQA benchmarks across various LLMs.

---

[A Survey on Agent-as-a-Judge](http://arxiv.org/abs/2601.05111)

- Agent-as-a-Judge (AaJ): introduces a comprehensive survey tracing the evolution from LLM-as-a-Judge to agentic evaluation, leveraging planning, memory, tool integration, multi-agent collaboration, and optimization paradigms.
- AaJ overcomes monolithic LLM limitations by enabling decentralized deliberation, executable verification, and fine-grained assessment across complex, multi-step tasks in general and professional domains.
- The framework's development is categorized into three progressive stages—Procedural, Reactive, and Self-Evolving—reflecting increasing levels of autonomy and adaptability in evaluation.

---

[NALAR: A Serving Framework for Agent Workflows](http://arxiv.org/abs/2601.05109)

- NALAR (Serving Framework for Agent Workflows): introduces a ground-up agent-serving framework that separates workflow specification from execution using a futures-centric model and a two-level control architecture for robust performance.
- The framework instruments agent and tool invocations with lightweight auto-generated stubs that return futures, which encode dependencies and context metadata necessary for adaptive scheduling and fine-grained prioritization.
- The two-level control architecture, comprising a Global Controller and local Component-Level Controllers mediated by a Node Store, enables dynamic policy enforcement and coordinated state management, cutting tail latency by 34–74% across workloads.

---

[Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction](http://arxiv.org/abs/2601.05107)

- SteeM (Steerable Memory Agent): introduces a framework enabling users to dynamically control an LLM agent's reliance on long-term memory, ranging from fresh-start innovation to high-fidelity adherence, using the Memory Dependence Control component.
- The approach addresses "Memory Anchoring," where LLMs default to high memory reliance, by training the agent via Preference-Aligned SFT and GRPO based on a Rubric Judge-derived memory-dependence metric.
- The system quantifies memory influence using a behavioral metric (MD-Score) and minimizes the alignment error between the realized dependence and the user's target preference ($p(q)$) across diverse long-horizon tasks.

---

[Arabic Prompts with English Tools: A Benchmark](http://arxiv.org/abs/2601.05101)

- APET (Arabic Prompts with English Tools): introduces the first dedicated benchmark for evaluating LLM tool-calling and agentic capabilities in Arabic, utilizing a Multilingual Evaluation Matrix across four experimental axes.
- The benchmark adapts the Berkeley Function Calling Leaderboard (BFCL) dataset into a multilingual parallel corpus to measure functional accuracy and robustness of models (GPT-OSS-20b, Llama-3.3-70b, Qwen3 variants) when prompted in Arabic.
- Findings reveal a significant performance gap, highlighting a "compounding language penalty" where accuracy drops severely when both the user query and tool definitions are in Arabic.

---

[FINDEEPFORECAST : A Live Multi-Agent System for Benchmarking Deep Research Agents in Financial Forecasting](http://arxiv.org/abs/2601.05039)

- FINDEEPFORECAST: introduces a live, end-to-end multi-agent system for continuously evaluating Deep Research (DR) agents on forward-looking financial forecasting tasks.
- The system utilizes a dual-track taxonomy to dynamically generate both recurrent (scheduled numerical) and non-recurrent (event-driven binary) corporate and macro-level tasks.
- By enforcing strict temporal isolation and rigorous ground truth verification, the system provides a contamination-free benchmark, revealing that current DR agents struggle significantly with precise recurrent numerical forecasting.

---

[Can Large Language Models Resolve Semantic Discrepancy in Self-Destructive Subcultures? Evidence from Jirai Kei](http://arxiv.org/abs/2601.05004)

- SAS (Subcultural Alignment Solver): introduces a multi-agent framework designed to enhance LLM understanding of self-destructive subcultures by incorporating Subculture Retrieval, Alignment Report Generation, and a Culture Alignment Solver.
- The framework addresses challenges like Knowledge Lag and Semantic Misalignment by automatically retrieving subcultural slang and aligning input expressions to the specific cultural context.
- SAS demonstrates competitive performance against fine-tuned LLMs and advanced agentic frameworks like OWL in detecting self-destructive behaviors (OD, ED, SH) within niche communities like Jirai Kei.

---

[SmartSearch: Process Reward-Guided Query Refinement for Search Agents](http://arxiv.org/abs/2601.04888)

- SmartSearch: introduces a framework that optimizes intermediate search query quality using Process Rewards and Query Refinement, guided by a three-stage curriculum learning framework.
- Process Rewards utilize Dual-Level Credit Assessment, comprising a rule-based novelty check and a model-based usefulness check (LLM$_{eval}$), to provide fine-grained supervision for query quality.
- The curriculum learning framework progresses through imitation (SFT), alignment (DPO), and generalization (RL) to enable the search agent to internalize the ability to enhance query quality.

---

[Mind2Report: A Cognitive Deep Research Agent for Expert-Level Commercial Report Synthesis](http://arxiv.org/abs/2601.04879)

- Mind2Report (M2R): introduces a training-free cognitive deep research agent that emulates a commercial analyst to synthesize expert-level reports using intent-driven outline formulation, memory-augmented adaptive search, and coherent-preserved iterative synthesis.
- The agent first clarifies imprecise intent via proactive questioning, then recursively searches web sources, distilling validated information into a dynamic memory via multi-dimensional self-reflection.
- M2R iteratively synthesizes the final report based on the established outline and validated knowledge, demonstrating superior performance against leading deep research agents on the QRC-Eval benchmark.

---

[Higher-Order Knowledge Representations for Agentic Scientific Reasoning](http://arxiv.org/abs/2601.04878)

- HOKAR-Agentic System: introduces a multi-agent framework that leverages a Global Hypergraph, constructed from scientific literature, as a structured knowledge substrate for agentic scientific reasoning.
- The system employs a GraphAgent to extract relevant subgraphs using Breadth First Search and Yen's K-shortest path algorithms, which are then passed to specialized Engineer and Hypothesizer agents for interpretation and hypothesis generation.
- This architecture establishes a "teacherless" discovery system where the hypergraph's higher-order topology acts as a verifiable guardrail, enabling agents to generate grounded mechanistic hypotheses for novel materials.

---

[Orchestrating Intelligence: Confidence-Aware Routing for Efficient Multi-Agent Collaboration across Multi-Scale Models](http://arxiv.org/abs/2601.04861)

- OI-MAS (Orchestrating Intelligence Multi-agent System): introduces a novel multi-agent framework implementing a state-dependent routing mechanism, composed of a Role Router (selects agent roles) and a Model Router (assigns LLM backbone), which dynamically coordinates agent roles and model scales from a multi-scale LLM Pool (multi-scale LLM backbones) based on the current reasoning state.
- The system utilizes Confidence-Aware Optimization (balances performance/cost), which uses model confidence as a signal to modulate the cost term, ensuring computational capacity is allocated proportionally to the estimated task complexity.
- By treating multi-agent reasoning as a symphony performance, the framework achieves superior accuracy and substantially reduces inference cost and latency compared to baseline multi-agent systems.

---

[RAAR: Retrieval Augmented Agentic Reasoning for Cross-Domain Misinformation Detection](http://arxiv.org/abs/2601.04853)
- RAAR (Retrieval Augmented Agentic Reasoning): introduces the first retrieval-augmented agentic reasoning framework for cross-domain misinformation detection, leveraging multi-perspective evidence retrieval and specialized multi-agent collaboration to construct verifiable reasoning paths, optimized via SFT and RL.
- The framework employs specialized Sub-Agents (Sentiment, Semantic, Style) to produce complementary analyses, which a Summary Agent integrates under Verifier guidance to perform systematic, multidimensional reasoning over complex evidence.
- Model optimization uses Supervised Fine-Tuning and Reinforcement Learning (GRPO) to enable cross-domain knowledge transfer and multi-task capability, significantly boosting performance over advanced LLMs and adaptation methods.

---

[Defense Against Indirect Prompt Injection via Tool Result Parsing](http://arxiv.org/abs/2601.04795)

- Tool Result Parsing (TRP): introduces a novel prompt-based defense mechanism that leverages the LLM to parse tool outputs, extracting only necessary, constrained data while filtering out malicious injection content.
- The approach utilizes two primary modules, ParseData for extracting minimal, formatted data, and CheckTool for sanitizing large text chunks by removing tool-triggering words.
- TRP significantly outperforms existing prompt-based and training-based defenses, achieving the lowest Attack Success Rate (ASR) while maintaining competitive utility under attack (UA).

---

[AgentOCR: Reimagining Agent History via Optical Self-Compression](http://arxiv.org/abs/2601.04786)

- AgentOCR: introduces a visually-grounded framework that reimagines agent history as a dynamic sequence of compact images, addressing the token budget bottleneck in multi-turn LLM agents.
- The framework incorporates segment optical caching to eliminate redundant rendering overhead and agentic self-compression to adaptively balance task success and token efficiency via RL.
- This optical approach preserves over 95% of text-based performance while substantially reducing token consumption by over 50% (up to 80.9% in peak contexts), demonstrating efficient resource utilization.

---

[SciIF: Benchmarking Scientific Instruction Following Towards Rigorous Scientific Intelligence](http://arxiv.org/abs/2601.04770)

- SciIF (Scientific Instruction Following Benchmark): introduces a multi-discipline benchmark evaluated via a Generate-then-Audit Protocol, featuring a Constraint Catalog, Answer Correctness Axis, Constraint Compliance Axis, Rule-based Verifier, and Dual Model Judges.
- SciIF evaluates LLMs' ability to solve university-level scientific problems while strictly adhering to explicit scientific constraints, moving beyond mere final-answer correctness to assess auditable reasoning.
- The benchmark reveals a systematic failure mode—compositional collapse—where models achieve high correctness but low multi-constraint compliance, demonstrating that constraint-oriented post-training improves both general instruction following and domain-specific reasoning.

---

[AT2PO: Agentic Turn-based Policy Optimization via Tree Search](http://arxiv.org/abs/2601.04767)

- AT2PO (Agentic Turn-based Policy Optimization via Tree Search): introduces a unified framework for multi-turn agentic RL, integrating Entropy-Guided Tree Expansion (strategic exploration), Turn-wise Credit Assignment (fine-grained reward propagation), and Agentic Turn-based Policy Optimization (turn-level policy update).
- The framework leverages a turn-level tree structure to enable strategic exploration by expanding high-entropy nodes and mitigating sparse rewards through backward propagation of outcome rewards.
- ATPO, the turn-level learning objective, improves training stability and alignment by applying importance sampling and clipping mechanisms at the natural granularity of multi-turn interactions.

---

[When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail](http://arxiv.org/abs/2601.04748)

- SAS (Single-Agent System with Skills): introduces a framework that compiles Multi-Agent Systems (MAS) into a single LLM agent by internalizing specialized roles as selectable skills, comprising a Base Language Model, Skill Library, Skill Selector, Skill Descriptor, Execution Policy, Execution Backend, and Hierarchical Routing.
- This compilation achieves substantial efficiency gains (53.7% token reduction, 49.5% latency reduction) compared to MAS, but faces a fundamental scaling challenge where selection accuracy degrades non-linearly beyond a critical capacity threshold ($\kappa$).
- The degradation is primarily driven by semantic confusability among skills, not library size alone, and can be effectively mitigated by implementing cognitive-grounded Hierarchical Routing to organize skills into structured categories.

---

[Tool-MAD: A Multi-Agent Debate Framework for Fact Verification with Diverse Tool Augmentation and Adaptive Retrieval](http://arxiv.org/abs/2601.04742)

- Tool-MAD: introduces a multi-agent debate framework for fact verification where specialized LLM agents leverage heterogeneous external tools (RAG Module and Search API) and adaptive query formulation for iterative evidence retrieval.
- The framework employs two debater agents ($A_R, A_S$) and a Judge agent ($A_J$), utilizing a Stability Score (Faithfulness and Answer Relevance) to dynamically assess response quality and guide debate progression.
- By enabling agents to iteratively refine evidence based on evolving arguments, the system enhances factual grounding, mitigates hallucinations, and promotes reliable consensus formation.

---

[Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning](http://arxiv.org/abs/2601.04726)

- CompassMem: introduces an event-centric memory framework that organizes experiences into a structured Event Graph using event segmentation and explicit logical relation extraction, serving as a logic map for agent searching and reasoning.
- The framework utilizes three LLM-based agents—a Planner, multiple Explorers, and a Responder—to perform active multi-path memory search guided by the graph topology and subgoal satisfaction.
- By explicitly encoding logical structure and enabling logic-aware retrieval, the system consistently improves performance on reasoning-intensive tasks like multi-hop and temporal QA over flat or passively structured memory baselines.

---

[Fame Fades, Nature Remains: Disentangling the Character Identity of Role-Playing Agents](http://arxiv.org/abs/2601.04716)

- Character Identity: introduces a two-layered framework to disentangle character identity in RPAs, comprising Parametric Identity (pre-trained LLM knowledge) and Attributive Identity (fine-grained behavioral properties) defined via a hierarchical Character Profile Schema.
- Evaluation across single-turn (PersonaGym) and multi-turn (COSER) interactions reveals "Fame Fades" (parametric advantage vanishes over turns) and "Nature Remains" (sensitivity to attribute valence).
- Mechanistic analysis pinpoints negative social natures (Morality, Interpersonal Relationships) as the primary performance bottleneck, guiding future character construction and evaluation.

---

[Beyond Monolithic Architectures: A Multi-Agent Search and Knowledge Optimization Framework for Agentic Search](http://arxiv.org/abs/2601.04703)

- M-ASK (Multi-Agent Search and Knowledge): introduces a collaborative framework that decouples agentic search into Search Behavior Agents (Planning, Search, Answer) and Knowledge Management Agents (Summary, Update), utilizing a Structured Knowledge State and turn-level dense rewards.
- The framework addresses monolithic agent bottlenecks—unconstrained output length, sparse rewards, and search noise—by enforcing role specialization and providing granular supervision via turn-specific $\Delta F1$ rewards.
- M-ASK employs a Parameter-Shared strategy across all agents, instantiated from a unified LLM backbone, to maximize sample efficiency and ensure stable convergence in multi-hop QA tasks.

---

[ResMAS: Resilience Optimization in LLM-based Multi-agent Systems](http://arxiv.org/abs/2601.04694)

- ResMAS: introduces a two-stage framework for enhancing LLM-based Multi-Agent System (MAS) resilience by optimizing communication topology and agent prompts.
- The first stage trains a Reward Model (GCN/MLP) to predict MAS resilience, guiding a Topology Generator (LLM) via Group Relative Policy Optimization (GRPO) to produce robust topologies.
- The second stage employs a topology-aware prompt optimization method that refines each agent's prompt based on neighbor interactions and the generated topology.

---

[Leveraging LLMs for Efficient and Personalized Smart Home Automation](http://arxiv.org/abs/2601.04680)

- IoTGPT (LLM-based smart home agent): introduces a reliable, efficient, and personalized IoT control framework using a three-stage reasoning pipeline (Decompose Stage, Derive Stage, Refine Stage) leveraging an LLM, Task Memory, Preference Table, and Correction Mechanisms to translate natural language instructions into executable commands for IoT Platforms.
- The system achieves efficiency and reliability by decomposing complex tasks into reusable subtasks stored in the hierarchical Task Memory, minimizing repeated LLM inference and mitigating non-deterministic errors.
- Adaptive personalization is achieved by abstracting user preferences into device-agnostic environmental properties stored in the Preference Table, enabling flexible transfer across heterogeneous device configurations.

---

[From National Curricula to Cultural Awareness: Constructing Open-Ended Culture-Specific Question Answering Dataset](http://arxiv.org/abs/2601.04632)

- CuCu (from national Curricula to Cultural Awareness): introduces an automated multi-agent LLM framework that transforms national textbook curricula into 34.1k open-ended, culture-specific question-answer pairs (KCaQA) across four languages, using National Curricula (Structured prior), Learning Outcomes Extraction (Input processing), Query Generation Agent (Initial query creation), Culture-sensitive Filtering (Culture-agnostic removal), Paraphrasing & Augmentation (Query variation), Human Validation & Refinement (Expert manual review), Multilingual Extension (Translation to 4 languages), Response Generation LLMs (Response text creation), User-tailored Response Generation (Difficulty bands), Response Evaluation Agent (Quality assessment), and Response Revision (Iterative improvement).
- The framework leverages national social studies curricula as a structured prior to ensure the generated KCaQA dataset is grounded in local civic norms, historical narratives, and sociocultural contexts for culture-aware LLM supervision.
- The pipeline employs iterative query generation, culture-sensitive filtering, and user-tailored response generation across Basic, Intermediate, and Advanced difficulty levels, validated by human experts and an LLM-as-a-judge evaluation.

---

[Beyond the "Truth": Investigating Election Rumors on Truth Social During the 2024 Election](http://arxiv.org/abs/2601.04631)

- RDA (Rumor Detection Agent): introduces a multi-stage framework for high-precision rumor detection on Truth Social, combining a fine-tuned RoBERTa classifier, keyword filtering, and a two-pass LLM verification pipeline using GPT-4o mini.
- The agent processes a large-scale dataset of nearly 15 million posts to quantify the psychological dynamics of rumor propagation, specifically the "illusory truth effect," in an ideologically homogeneous network.
- Empirical results show that a user's sharing probability rises steadily with each additional exposure, and Donald Trump acts as the central node, accelerating rumor spread across the platform.

---

[AgentDevel: Reframing Self-Evolving LLM Agents as Release Engineering](http://arxiv.org/abs/2601.04620)

- AgentDevel: introduces a release engineering pipeline that externalizes LLM agent improvement into a structured workflow, utilizing a TrainSet and Rubric to run the Current Agent, collect Execution Traces, apply Programmatic Scorers, and use an LLM Critic to generate Quality Records.
- The pipeline uses an Analysis Engine to execute Executable Diagnostic Scripts based on Quality Records, generating a Diagnosis Report that informs the synthesis of a single Release Candidate (RC) blueprint change.
- Promotion of the Release Candidate to the Next Official Version is governed by a Flip-Centered Gate, which prioritizes minimizing P→F regressions and maximizing F→P fixes for stable, auditable improvement.

---

[Sci-Reasoning: A Dataset Decoding AI Innovation Patterns](http://arxiv.org/abs/2601.04577)

- Sci-Reasoning: introduces the first dataset capturing structured intellectual synthesis behind high-quality AI research, utilizing a pipeline for High-Quality Paper Identification (Community-validated signals), Intellectual Lineage Tracing (LLM analysis), and Intellectual Connection Synthesis (Structured lineage graphs).
- The methodology employs an LLM (GPT-5) accelerated, human-verified pipeline to trace key predecessors and articulate specific reasoning links, resulting in 3,819 richly annotated papers from top ML conferences (2023-2025).
- Analysis of the dataset identifies 15 distinct thinking patterns, dominated by Gap-Driven Reframing, Cross-Domain Synthesis, and Representation Shift, providing structured trajectories for training AI research agents.

---

[BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents](http://arxiv.org/abs/2601.04566)

- BackdoorAgent: introduces a modular, stage-aware framework for unified analysis of backdoor threats in LLM agents by decomposing the attack surface into planning, memory, and tool stages.
- The framework instruments agent execution to systematically analyze trigger activation and propagation across stages using an instrumented runtime and trajectory logging.
- BackdoorAgent includes a standardized benchmark covering four representative agent applications (Agent QA, Code, Web, Drive) for evaluating cross-stage backdoor persistence.

---

[4D-ARE: Bridging the Attribution Gap in LLM Agent Requirements Engineering](http://arxiv.org/abs/2601.04556)

- 4D-ARE (4-Dimensional Attribution-Driven Agent Requirements Engineering): introduces a methodology for specifying LLM agents grounded in causal attribution logic, operationalized through five layers that compile into a system prompt.
- The framework defines four dimensions ($D_R$, $D_P$, $D_S$, $D_L$) representing a causal chain (Results $\leftarrow$ Process $\leftarrow$ Support $\leftarrow$ Long-term) to ensure attribution-complete responses.
- The five-layer architecture systematically translates domain expertise into explicit agent specifications, complementing runtime reasoning frameworks by defining what the LLM agent should reason about.

---

[Exploring Recommender System Evaluation: A Multi-Modal User Agent Framework for A/B Testing](http://arxiv.org/abs/2601.04554)

- A/B Agent (Multi-Modal User Agent Framework for A/B Testing): introduces a multimodal LLM-based user agent framework designed to simulate complex human perception and interaction trajectories within a realistic recommendation sandbox environment.
- The framework integrates a Profile Module, a multimodal Memory Module, an Action Module, and a Fatigue System to achieve human-like decision-making for A/B testing evaluation.
- The system utilizes the MM-ML-1M dataset and a Recommendation Sandbox UI to enable multi-page, multimodal interactions, validating its potential as an alternative to traditional online A/B testing.

---

[LinguaGame: A Linguistically Grounded Game-Theoretic Paradigm for Multi-Agent Dialogue Generation](http://arxiv.org/abs/2601.04516)

- LinguaGame: introduces a linguistically-grounded game-theoretic paradigm for multi-agent dialogue generation, modeling communication as a signalling game over communicative Intent-Strategy Pairs and solved via a Training-Free Equilibrium Approximation Algorithm (piKL) for inference-time decision adjustment.
- The framework enhances communication efficiency by requiring agents (Sender and Receiver) to infer each other's communicative intents and strategies, decoupling game design from task-specific objectives and enabling broad generalization.
- Evaluated in simulated courtroom proceedings and debates, the paradigm significantly improves dialogue quality, clarity, conciseness, and tactical coherence compared to standard MAS baselines.

---

[CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts](http://arxiv.org/abs/2601.04505)

- CircuitLM: introduces a novel multi-agent LLM-aided circuit design pipeline that translates natural language prompts into structured, visually interpretable CircuitJSON schematics through five sequential stages.
- The pipeline includes an Identification Agent, Retrieval Agent, Electronics Expert Agent, Circuit Generation Agent, and Schematic Visualizer, anchored by a curated, embedding-powered component knowledge base.
- The framework ensures design safety and validity by grounding generation in a verified component database and utilizing the Dual-Metric Circuit Validation (DMCV) evaluation framework.

---

[A Closed-Loop Multi-Agent System Driven by LLMs for Meal-Level Personalized Nutrition Management](http://arxiv.org/abs/2601.04491)

- Closed-Loop Multi-Agent System (MAS): introduces a personalized nutrition guidance system that combines image-based meal logging with an LLM-driven multi-agent controller to provide meal-level closed-loop support.
- The system coordinates the Vision, Dialogue, and State management agents to estimate nutrients from meal photos, update the daily intake budget, and dynamically adapt the next meal plan based on user preferences and constraints.
- The closed-loop design models the task as an iterative decision process, limiting error propagation and ensuring recommendations are aligned with the most recent intake summary and remaining budget.

---

[Beyond Static Summarization: Proactive Memory Extraction for LLM Agents](http://arxiv.org/abs/2601.04463)

- ProMem (Proactive Memory Extraction): introduces an iterative cognitive process for memory extraction, utilizing Initial Extraction (feed-forward phase), Memory Completion (context alignment phase), and Memory Verification (recurrent feedback phase).
- Inspired by recurrent processing theory, the framework overcomes limitations of static summarization by actively using self-questioning to probe dialogue history for missing information and error correction.
- This recurrent feedback loop significantly improves memory integrity and QA accuracy by ensuring the final memory is complete, accurate, and grounded in the raw dialogue text.


---

[Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions](http://arxiv.org/abs/2601.04170)

- Agent Drift Mitigation Strategies: introduces a comprehensive study on agent drift, the progressive degradation of multi-agent LLM system behavior, and proposes three intervention approaches: EMC, DAR, and ABA.
- The Agent Stability Index (ASI) is introduced as a novel composite metric framework quantifying drift across 12 behavioral dimensions, including response consistency, tool usage patterns, and inter-agent coordination.
- Simulation results demonstrate that unchecked agent drift leads to a projected 42% reduction in task success rate and a 3.2x increase in human intervention requirements over extended interactions.

---

[INFINITEWEB: Scalable Web Environment Synthesis for GUI Agent Training](http://arxiv.org/abs/2601.04126)

- INFINITEWEB: introduces a system that automatically generates scalable, functional web environments for GUI agent training, utilizing the Unified Specification Stage, Task-Centric Backend, Design-Guided Frontend, and Automatic Evaluator Generation.
- The system ensures consistency via Unified Specification, correctness via task-centric test-driven development (TCTDD), and diversity through website seeds and design image guidance.
- The framework generates verifiable task evaluators alongside the websites, providing dense reward signals crucial for effective reinforcement learning-based GUI agent training.

---

[O-Researcher: An Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL](http://arxiv.org/abs/2601.03743)

- O-Researcher: introduces a novel framework for automated synthesis of research-grade instructional data via a multi-agent workflow, followed by a two-stage training strategy integrating SFT and RLAIF.
- The multi-agent workflow uses parallel execution, where specialized LLM agents decompose complex queries, execute tool-integrated reasoning, and aggregate sub-reports to create high-fidelity training trajectories.
- The RLAIF stage utilizes an LLM-as-a-Judge and a composite reward function to maximize model alignment and capability, enabling open-source LLMs to achieve state-of-the-art deep research performance.

---

[Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning](http://arxiv.org/abs/2601.03641)

- Agent-Dice (Parameter Fusion Framework): introduces a novel parameter fusion framework based on directional consensus evaluation to disentangle common and conflicting knowledge updates in LLM agent continual learning.
- The framework employs a two-stage process: Geometric Consensus Filtering prunes conflicting gradients for stability, and Curvature-based Importance Weighting amplifies shared semantics for plasticity.
- The approach resolves the stability-plasticity dilemma, demonstrating effective continual learning performance in GUI agent and tool-use agent domains with minimal computational overhead.

---

[LLM Agents in Law: Taxonomy, Applications, and Challenges](http://arxiv.org/abs/2601.06216)

- Legal LLM Agents: introduces a comprehensive survey analyzing how agentic architectures—including planning, memory, tool use, RAG, reflection, multi-agent collaboration, and HITL protocols—address the limitations of standalone LLMs in the legal domain.
- These agentic systems mitigate persistent challenges like hallucination, outdated information, and lack of verifiability by providing external grounding, procedural orchestration, and multi-layer governance.
- The paper presents a structured taxonomy of current agent applications across five core legal practice areas: Legal Search, Litigation &amp; Dispute Resolution, Compliance &amp; Regulation, Consultation &amp; Transaction, and Non-substantive Tasks.

---

[Lost in Execution: On the Multilingual Robustness of Tool Calling in Large Language Models](http://arxiv.org/abs/2601.05366)

- MLCL (Multilingual Tool Calling Leaderboard): introduces a diagnostic benchmark and systematic evaluation of LLM tool calling robustness across Chinese, Hindi, and Igbo, focusing on execution-level failures rather than semantic misunderstanding.
- The benchmark systematically varies query language composition (NT, PAR, FT) and semantic perturbations (NO, PARA, SYNO) to isolate parameter value language mismatch as the dominant failure mode, where LLMs copy non-English tokens into English-only parameter fields.- Simple inference-time strategies (PT, PRE, POST) reduce language-induced errors but fail to fully recover English-level performance, suggesting multilingual robustness is a system- and interface-level challenge.
- Simple inference-time strategies (PT, PRE, POST) reduce language-induced errors but fail to fully recover English-level performance, suggesting multilingual robustness is a system- and interface-level challenge.

---

[GlyRAG: Context-Aware Retrieval-Augmented Framework for Blood Glucose Forecasting](http://arxiv.org/abs/2601.05353)

- GlyRAG: introduces a context-aware, retrieval-augmented agentic forecasting framework that derives semantic understanding from CGM traces using an LLM agent and fuses it with physiological embeddings for multi-horizon prediction.
- The architecture employs an LLM to generate clinically meaningful textual summaries, which are then fused with patch-based glucose representations via a multimodal transformer encoder and aligned using a cross-translational loss.
- A retrieval module identifies similar historical episodes in the learned embedding space and integrates these case-based analogues using cross-attention prior to generating the final forecast.

---

[Effects of personality steering on cooperative behavior in Large Language Model agents](http://arxiv.org/abs/2601.05302)

- Personality Steering Framework: introduces a study examining how personality steering affects cooperative behavior in LLM agents (GPT-3.5-turbo, GPT-4o, GPT-5) using the Big Five framework and Repeated Prisoner's Dilemma games (RPD).
- The research quantitatively measured intrinsic LLM personality profiles using the BFI-44 and tested agent behavior under baseline, personality-informed, and extreme personality manipulation conditions against fixed opponent strategies.
- Results indicate that agreeableness is the dominant factor promoting cooperation, but later-generation LLMs exhibit more selective cooperation and strategic robustness against exploitative opponents, suggesting personality steering acts as a behavioral bias.

---

[Reinforced Efficient Reasoning via Semantically Diverse Exploration](http://arxiv.org/abs/2601.05053)

- ROSE (Reinforced Efficient Reasoning via Semantically Diverse Exploration): introduces a novel RL framework for LLMs that enhances reasoning accuracy and efficiency using semantic-entropy guided exploration and length-aware segment-level advantage estimation.
- The framework employs a semantic-entropy-based branching strategy, combined with an $\epsilon$-exploration mechanism, to promote diverse and effective exploration of reasoning paths within an MCTS tree structure.
- Efficiency is improved via a length-aware segment-level advantage estimator that rewards concise and correct reasoning while penalizing unnecessarily long reasoning chains.

---

[Character-R1: Enhancing Role-Aware Reasoning in Role-Playing Agents via RLVR](http://arxiv.org/abs/2601.04611)

- Character-R1: introduces a novel Reinforcement Learning (RL) framework for Role-Playing Agents (RPAs) that shifts from behaviorist imitation to cognitivist reasoning using direct, comprehensive, and flexible verifiable reward signals. 
- The framework utilizes a Large Reasoning Model (LRM) guided by three core reward designs: Cognitive Focus Reward, Reference-Guided Reward, and Character-Conditioned Reward Normalization. 
- Cognitive Focus Reward enforces explicit label-based analysis of 10 character elements to structure internal cognition, while normalization ensures robust optimization across heterogeneous roles via Role Grouping.

---

[Not All Steps are Informative: On the Linearity of LLMs' RLVR Training](http://arxiv.org/abs/2601.04537)

- RL-Extra (Reinforcement Learning Extrapolation): introduces an accelerated training paradigm that exploits the observed strong linearity in LLM weights and outputs during RLVR training to significantly reduce computational costs.
- The approach proposes two direct acceleration schemes, Weight Extrapolation and Logits Extrapolation, and the iterative RL-Extra, which alternates between standard RL steps and gradient-free extrapolation.
- By leveraging linearity, the paradigm achieves up to a 6.1x wall-clock speedup and performance gains over standard RL training by accurately projecting future model states without expensive rollouts.

---

#### 7th January 2026


[Embedding Autonomous Agents in Resource-Constrained Robotic Platforms](http://arxiv.org/abs/2601.04191)

- Embedded-BDI framework (AgentSpeak): introduces embedding an autonomous BDI agent onto a resource-constrained Pololu 3pi+ 2040 Robot platform, utilizing a Translation Engine (AgentSpeak to C++ code), Runtime Library (Executes reasoning cycle), and Hardware-Dependent Code (Perception and action API).
- The BDI Agent manages navigation and decision-making by reasoning over beliefs formed by Reflectance Sensors and executing intentions via DC Motors to solve a line-following maze using the left-hand rule.
- Experimental results confirm the feasibility of real-time autonomy on constrained hardware, demonstrating that the agent's reasoning cycle (belief update and plan selection) is computationally efficient, executing in less than one millisecond.

---

[ComfySearch: Autonomous Exploration and Reasoning for ComfyUI Workflows](http://arxiv.org/abs/2601.04060)

- ComfySearch: introduces an agentic framework that formulates ComfyUI workflow generation as a Markov Decision Process (MDP) using a Policy/Agent/LLM, State-Aware Validation, In-Place Repair, Entropy-Adaptive Branching, Tool-Mediated Transition, GRPO Optimization, and Hierarchical Terminal Reward.
- The framework enforces online structural correctness via state-aware validation and in-place repair (C1), and navigates long-horizon uncertainty by employing entropy-adaptive branching (C2) to selectively explore high-ambiguity decisions.
- The policy is trained using Supervised Warmup followed by tool-feedback RL with Group Relative Policy Optimization (GRPO), achieving high executability and task resolution rates on complex ComfyUI tasks.

---

[Staged Voxel-Level Deep Reinforcement Learning for 3D Medical Image Segmentation with Noisy Annotations](http://arxiv.org/abs/2601.03875)

- SVL-DRL (Staged Voxel-Level Deep Reinforcement Learning): introduces an end-to-end staged voxel-level deep reinforcement learning framework for robust 3D medical image segmentation under noisy annotations, utilizing a Feature Extraction Module, Segmentation Network, Value Network, Policy Network, vA3C Module, Staged Training, Composite Reward Function, and Action Space.
- The framework employs a dynamic iterative update strategy and a voxel-level Asynchronous Advantage Actor-Critic (vA3C) module, treating each voxel as an autonomous agent to dynamically refine its state representation and mitigate erroneous labels.
- Training is structured into Warmup, Transition, and Full RL stages, incorporating a novel action space and a composite reward function combining Dice value and a spatial continuity metric to enhance segmentation accuracy.

---

[Atlas: Orchestrating Heterogeneous Models and Tools for Multi-Domain Complex Reasoning](http://arxiv.org/abs/2601.03872)

- ATLAS (Adaptive Tool-LLM Alignment and Synergistic Invocation): introduces a dual-path framework for dynamic model-tool alignment, combining training-free cluster-based routing for domain-specific efficiency and RL-driven multi-step routing for open-domain adaptability.
- The framework jointly optimizes heterogeneous LLM and external tool combinations within a Cartesian product search space, addressing the limitations of fixed logic and isolated optimization in prior routing methods.
- The RL component uses a composite reward structure, including format, outcome, and model selection rewards, enabling the agent to learn transferable routing principles for superior generalization across diverse complex reasoning tasks.

---

[From Laboratory to Real-World Applications: Benchmarking Agentic Code Reasoning at the Repository Level](http://arxiv.org/abs/2601.03731)

- RepoReason: introduces a repository-level code reasoning benchmark designed as a white-box diagnostic tool, utilizing Abductive Assertion Verification and the Semantic Oracle Framework to evaluate LLM agents.
- The framework employs a five-phase pipeline including Repository Curation, Structural Filtering, Execution-Driven Mutation, Task Instantiation, and Diagnostic Evaluation to ensure tasks are authentic, deep, and unmemorized.
- A White-box Diagnostic Instrument quantifies reasoning complexity using three orthogonal cognitive metrics: ESV (Reading Load), MCL (Simulation Depth), and DFI (Integration Width).

---

[Do Autonomous Agents Contribute Test Code? A Study of Tests in Agentic Pull Requests](http://arxiv.org/abs/2601.03556)

- AIDev Mining Study (AMS): introduces an empirical study analyzing test inclusion and testing practices in agent-generated Pull Requests (PRs) using the AIDev Dataset, focusing on five major autonomous coding agents (OpenAI Codex, GitHub Copilot, Devin, Cursor, and Claude Code).
- The study reconstructs commit timelines and uses regex heuristics for Test File Identification to calculate PR Metrics Calculation, including churn, turnaround time, merge rate, and test-to-code churn ratio.
- Findings indicate that test inclusion in agentic PRs is increasing over time, and test-containing PRs are consistently larger and require longer turnaround times compared to non-test PRs.

---

[SCRIBE: Structured Mid-Level Supervision for Tool-Using Language Models](http://arxiv.org/abs/2601.03555)

- SCRIBE (Skill-Conditioned Reward with Intermediate Behavioral Evaluation): introduces a reinforcement learning framework that enhances tool-augmented reasoning by intervening at a mid-level abstraction using a Router, Skill Prototype Library, LLM as Reward Model, and GRPO for policy optimization.
- SCRIBE anchors reward modeling in a curated library of Skill Prototypes, transforming open-ended LLM evaluation into a constrained verification task using context-specific rubrics to significantly reduce reward variance.
- The framework achieves state-of-the-art performance on reasoning and tool-use benchmarks, demonstrating that mid-level skill mastery acts as a precursor to the emergence of strategic high-level planning.

---

[DeepSynth-Eval: Objectively Evaluating Information Consolidation in Deep Survey Writing](http://arxiv.org/abs/2601.03540)

- DeepSynth-Eval (DSE): introduces a benchmark to objectively evaluate LLM information consolidation capabilities in deep survey writing, utilizing high-quality surveys as gold standards and reverse-engineering tasks to create an Oracle Context (D) and an Evaluation Standard (C).
- The Evaluation Standard (C) consists of fine-grained General Checklists ($C_{gen}$) for factual coverage and Constraint Checklists ($C_{con}$) for structural organization, transforming subjective judgment into verifiable metrics via a Judge Model.
- The benchmark evaluates synthesis performance using two controlled workflows—E2E Single-turn and Agentic Multi-turn—demonstrating that agentic plan-then-write methods significantly outperform single-turn generation by improving grounding and reducing hallucinations.

---

[XGRAMMAR 2: DYNAMIC AND EFFICIENT STRUCTURED GENERATION ENGINE FOR AGENTIC LLMS](http://arxiv.org/abs/2601.04426)

- XGrammar 2: introduces a highly optimized structured generation engine for agentic LLMs, featuring TagDispatch (Dynamic grammar dispatching), JIT Compilation (Reduce preprocessing time), Cross-Grammar Caching (Reuse common sub-structures), Adaptive Token Mask Cache (Earley parser based), Repetition Compression Algorithm (Handle repetition structures), Dispatching FSM (Manages grammar states), Global Cache Pool (Stores token mask caches), and LLM Inference Engine (Generates constrained output).
- The engine uses the novel TagDispatch semantics and a Dispatching FSM to efficiently manage dynamic grammar switching required for complex tool calling and conditional structured generation tasks.
- System optimizations, including JIT compilation and cross-grammar caching based on an Earley parser, enable the engine to achieve over 6x speedup compared to existing structured generation engines with near-zero overhead.

---

[GAVEL: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization](http://arxiv.org/abs/2601.04424)

- GAVEL (Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization): introduces a reference-based evaluation framework (GAVEL-REF) and an autonomous agent scaffold (GAVEL-AGENT) for assessing LLMs on multi-document legal case summarization.
- GAVEL-REF systematically evaluates LLMs using Checklist Evaluation (26 key items, multi-value), Residual Fact Evaluation (non-checklist facts), and Writing Style Evaluation, combining them into a weighted score.
- GAVEL-AGENT equips an LLM with six Tools for efficient document navigation and checklist extraction, achieving competitive performance while significantly reducing token usage compared to end-to-end methods.

---

[The Language of Bargaining: Linguistic Effects in LLM Negotiations](http://arxiv.org/abs/2601.04387)

- NegotiationArena (Extended): introduces a systematic evaluation of linguistic effects in LLM negotiation using controlled multi-agent simulations across Ultimatum, Buy-Sell, and Resource Exchange games, comparing English against four Indic framings.
- The study demonstrates that language choice acts as a latent policy prior, shifting negotiation outcomes more strongly than changing the underlying LLM architecture and reversing role-based advantages.
- Effects are task-contingent, showing that Indic languages reduce stability in distributive games but induce richer exploration in integrative settings, underscoring the need for culturally-aware LLM evaluation.

---

[Stable Language Guidance for Vision–Language–Action Models](http://arxiv.org/abs/2601.04052)

- RSS (Residual Semantic Steering): introduces a probabilistic framework that disentangles physical affordance from semantic execution in Vision-Language-Action (VLA) models, utilizing Monte Carlo Syntactic Integration and Residual Affordance Steering.
- Monte Carlo Syntactic Integration uses an Oracle Teacher LLM to generate a dense linguistic neighborhood, ensuring the policy is invariant to surface-level syntactic perturbations by optimizing an Expected Semantic Loss.
- Residual Affordance Steering acts as a Bias Suppressor by subtracting the Base Affordance Distribution (visual prior) from conditional logits to isolate and amplify the pure semantic signal, mitigating visual dominance.

---

[HoneyTrap: Deceiving Large Language Model Attackers to Honeypot Traps with Resilient Multi-Agent Defense](http://arxiv.org/abs/2601.04034)

- HoneyTrap: introduces a novel deceptive LLM defense framework leveraging collaborative defenders to counter multi-turn jailbreak attacks using four specialized agents: Threat Interceptor, Misdirection Controller, Forensic Tracker, and System Harmonizer.
- The system transforms adversarial interactions into honeypot-style traps by strategically escalating adversarial costs, prolonging interaction time, and analyzing behavioral patterns to ensure resilience against evolving threats.
- Evaluation using the new MTJ-Pro dataset and metrics (MSR and ARC) shows the framework significantly reduces ASR while improving Mislead Success Rate and Attack Resource Consumption compared to baselines.

---

[When Numbers Start Talking: Implicit Numerical Coordination Among LLM-Based Agents](http://arxiv.org/abs/2601.03846)

- Implicit Numerical Coordination (INC) Framework: introduces a game-theoretic study of covert numerical communication among LLM-Based Agents (Strategic actors), Game Scenarios (Canonical dilemmas), Communication Conditions (Explicit, covert, random), FAIRGAME Environment (Simulation platform), and Personality Pairings (Cooperative/Selfish traits).
- The research investigates how LLM agents develop implicit signaling strategies across four canonical social dilemma games (Prisoner's Dilemma, Snowdrift, Stag Hunt, Harmony) under various communication constraints, including decimal and hexadecimal covert channels.
- Numerical messages exhibit structure only when agents are explicitly instructed to use them for communication, leading to low-entropy signaling conventions that selectively influence coordination in games with strategic uncertainty.

---

[Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents](http://arxiv.org/abs/2601.03785)

- Membox: introduces a hierarchical memory architecture centered on topic continuity, utilizing the Topic Loom (Sliding-window topic weaving), Trace Weaver (Long-range topic linking), Memory Boxes (Coherent dialogue units), Event-Timeline Traces (Persistent macro-topic recurrence), and LLM (Topic classification and extraction), to overcome the fragmentation-compensation paradigm in LLM agent memory.
- The Topic Loom groups consecutive same-topic dialogue turns into coherent Memory Boxes at storage time, preserving local narrative integrity and micro-topic cohesion by cleanly cutting where thematic shifts occur.
- The Trace Weaver subsequently links these sealed boxes across discontinuities to recover recurring macro-topics, achieving superior retrieval and reasoning performance in multi-turn dialogue tasks with reduced computational cost.

---

[Agentic Proof Automation: A Case Study](http://arxiv.org/abs/2601.03768)

- APA (Agentic Proof Automation): introduces a scheme where a Human (Provides guidance/insight) collaborates with an LLM Agent (Generates proof scripts) that iteratively refines proofs based on Lean 4 (Provides compiler feedback) using Agentic Frameworks (Wraps LLM with tools) and specialized tools.
- The LLM Agent utilizes tools like the lean4check Tool (Compiles Lean 4 module) for feedback, the Code exploration Tool (Searches definitions/lemmas) for context, and the File modification Tool (Creates/edits files) to handle the mechanical work of proof development.
- This approach was validated by mechanizing the semantic type soundness of System Capless in Lean 4, achieving an 87% success rate across 189 proof engineering tasks with minimal human intervention.

---

[R³L: Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification](http://arxiv.org/abs/2601.03715)

- R³L (Reflect-then-Retry Reinforcement Learning): introduces a robust RL framework that synthesizes high-quality trajectories via Language-Guided Reflect-Then-Retry, refines credit assignment using Pivotal Credit Assignment, and stabilizes training through Positive Amplification.
- The framework addresses inefficient exploration by leveraging language feedback to diagnose errors and restart generation from identified Pivot Points, significantly reducing rollout costs.
- Pivotal Credit Assignment focuses gradient updates on the diverging suffix of base/retry trajectory pairs, while Positive Amplification ensures constructive signals dominate optimization in failure-dominated regimes.

---

[Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis](http://arxiv.org/abs/2601.03676)

- STEPS (Skill Taxonomy-guided Entropy-based Post-training data Synthesis framework): introduces a two-stage framework for generating compositionally challenging data by first inducing a hierarchical skill taxonomy using structural entropy and then synthesizing data by maximizing marginal structural information gain.
- The framework addresses the data sparsity bottleneck in complex skill combinations by efficiently exploring the combinatorial skill space and generating high-gain compositions that target structural weaknesses.
- STEPS utilizes an Advanced LLM, acting as a Synergistic Content Architect, guided by a system prompt (SINTAX) to fuse disparate atomic skills into semantically coherent, complex, multi-turn instructions.

---

[NeuronScope: A Multi-Agent Framework for Explaining Polysemantic Neurons in Language Models](http://arxiv.org/abs/2601.03671)

- NeuronScope: introduces a multi-agent framework that reformulates polysemantic neuron interpretation in LLMs as an iterative, activation-guided process, including agents for hypothesis proposing, semantic decomposition, clustering, and iterative refinement.
- The framework systematically observes raw activation patterns to disentangle mixed semantics into distinct atomic components, which are then grouped into semantic modes.
- By leveraging activation-guided feedback, the system iteratively revises explanations, yielding precise, multi-faceted interpretations with significantly higher activation correlation scores than single-pass baselines.

---

[Architecting Agentic Communities using Design Patterns: A Framework Grounded in ODP Enterprise Language Formalism](http://arxiv.org/abs/2601.03624)

- ACF (Agentic Communities Framework): introduces a systematic architectural approach for production-grade agentic AI systems using a three-tier classification and a catalogue of 46 design patterns grounded in ODP Enterprise Language (ODP-EL) formalism.
- The framework leverages ODP-EL community specifications to provide verifiable governance and accountability through explicit roles, contracts, and deontic tokens (burden, permit, embargo) for both AI agents and human participants.
- The methodology is validated via a clinical trial matching case study, demonstrating pattern composition across LLM Agents, Agentic AI, and Agentic Communities layers to meet stringent regulatory requirements.

---

[The Pneuma Project: Reifying Information Needs as Relational Schemas to Automate Discovery, Guide Preparation, and Align Data with Intent](http://arxiv.org/abs/2601.03618)

- PNEUMA-SEEKER: introduces a system that helps users articulate and fulfill information needs through iterative, language-guided interaction, reifying the evolving information need as a relational data model (T, Q).
- The architecture combines context specialization, a conductor-style planner (CONDUCTOR), and a convergence mechanism based on the shared state (T, Q) to guide discovery and preparation.
- CONDUCTOR dynamically orchestrates the workflow by selecting actions and calling tools like the IR SYSTEM, MATERIALIZER, and SQL EXECUTOR to align the state (T, Q) with the user's latent intent.

---

[DiVA: Fine-grained Factuality Verification with Agentic-Discriminative Verifier](http://arxiv.org/abs/2601.03605)

- DiVA (Agentic Discriminative Verifier): introduces a hybrid framework for fine-grained factuality verification, synergizing a Generative Module (Leverages reasoning/tools) and a Discriminative Module (Renders fine-grained score) to output continuous factuality scores.
- The process involves Agentic Search for external knowledge, Context Compression to condense the reasoning trajectory, and Score Prediction using a discriminative module optimized via pairwise ranking loss.
- DiVA addresses the limitations of binary verification by distinguishing error severity and is evaluated on the new FGVeriBench benchmark for fine-grained factuality modeling across single- and multi-hop scenarios.

---

[Interleaved Tool-Call Reasoning for Protein Function Understanding](http://arxiv.org/abs/2601.03604)

- PFUA (Tool-augmented Protein Reasoning Agent): introduces a tool-augmented protein reasoning agent that unifies problem decomposition, tool invocation, and grounded answer generation using an LLM and a Bio-Tool Pool.
- PFUA addresses the limitations of text-only reasoning in protein function understanding by integrating domain-specific tools to produce verifiable intermediate evidence.
- The agent consistently outperforms text-only reasoning models across four benchmarks, achieving an average performance improvement of 103%.

---

[Jailbreaking LLMs & VLMs: Mechanisms, Evaluation, and Unified Defenses](http://arxiv.org/abs/2601.03594)

- 3DSF (Three-Dimensional Survey Framework): introduces a systematic survey of jailbreak attacks and defenses on LLMs and VLMs, categorized into Attack, Defense, and Evaluation dimensions.
- The survey consolidates shared mechanisms and proposes unified defense principles: variant-consistency and gradient-sensitivity detection, safety-aware decoding, and adversarially augmented preference alignment.
- Key architectural components include LLMs, VLMs, Guard Models, and various Evaluators used to assess Attack Success Rate and Toxicity Score across text-only and multimodal settings.

---

[Can LLMs See Without Pixels? Benchmarking Spatial Intelligence from Textual Descriptions](http://arxiv.org/abs/2601.03590)

- SiT-Bench (Spatial-in-Text Benchmark): introduces a large-scale, high-fidelity textual benchmark comprising 5 Primary Categories and 17 Diverse Subtasks, designed to decouple LLM spatial cognition from visual perception.
- The benchmark challenges LLMs to perform pure symbolic geometric reasoning using coordinate-aware textual descriptions converted from single/multi-view scenes, rather than visual pattern matching.
- Evaluation reveals a significant "spatial gap" in global consistency tasks, although explicit spatial reasoning (Chain-of-Thought) significantly boosts LLM performance.

---

[EvolMem: A Cognitive-Driven Benchmark for Multi-Session Dialogue Memory](http://arxiv.org/abs/2601.03543)

- EvolMem (Cognitive-Driven Benchmark for Multi-Session Dialogue Memory): introduces a benchmark grounded in cognitive psychology covering declarative and non-declarative memory, constructed using a hybrid data synthesis framework that integrates topic-initiated generation and narrative-inspired transformation.
- The hybrid synthesis framework uses a Planner, Generator, and Filter to create multi-session dialogues with controllable complexity, ensuring interaction coherence and topical diversity.
- Evaluation utilizes multi-faceted metrics (result-oriented, process-oriented, open-ended) to assess seven fine-grained memory abilities, revealing that no LLM consistently outperforms others across all dimensions.

---

[FROM BITS TO CHIPS: AN LLM-BASED HARDWARE-AWARE QUANTIZATION AGENT FOR STREAMLINED DEPLOYMENT OF LLMS](http://arxiv.org/abs/2601.03484)

- HAQA (Hardware-Aware Quantization Agent): introduces an automated framework leveraging LLMs to jointly optimize quantization fine-tuning and hardware deployment hyperparameters using iterative Static and Dynamic Prompts.
- The LLM Agent generates Optimization Guidance based on real-time accuracy and speed feedback incorporated into the Dynamic Prompt, streamlining the complex deployment workflow.
- The framework implements adaptive quantization strategies across diverse hardware platforms, achieving up to 2.3x inference speedup and improved accuracy compared to unoptimized models.

---

[PARACODEX: A Profiling-Guided Autonomous Coding Agent for Reliable Parallel Code Generation and Translation](http://arxiv.org/abs/2601.04327)

- PARACODEX (Profiling-Guided Autonomous Coding Agent): introduces an autonomous LLM agent workflow that parallelizes and migrates code using staged analysis, explicit data planning, correctness gating, and profiling-guided refinement.
- The system employs a three-stage pipeline—Analysis, Translation (with data strategy), and Performance Tuning—driven by CLI agents and verified using external tools like compilers and profilers.
- By enforcing an artifact-driven, tool-verified workflow, the agent reduces brittle single-shot outputs and achieves substantial GPU-time speedups across diverse HPC benchmarks.

---

[Rate or Fate? RLVER: Reinforcement Learning with Verifiable Noisy Rewards](http://arxiv.org/abs/2601.04411)

- RLVER: introduces an analytically tractable multi-armed bandit view of RLVR dynamics, instantiated with GRPO, modeling false positives and false negatives to analyze learning stability and convergence under noisy rewards.
- The dynamics reveal a sharp phase transition governed solely by Youden's index $J = \text{TPR} - \text{FPR}$, where $J>0$ ensures learning (rate, not fate), $J=0$ is neutral drift, and $J<0$ causes anti-learning (collapse).
- The framework provides a general lens for analyzing RLVR stability, demonstrating that noise primarily rescales convergence time rather than altering the asymptotic outcome when the verifier is net-informative ($J>0$).

---

[Step Potential Advantage Estimation: Harnessing Intermediate Confidence and Correctness for Efficient Mathematical Reasoning](http://arxiv.org/abs/2601.03823)

- SPAE (Step Potential Advantage Estimation): introduces a fine-grained credit assignment method for Reinforcement Learning with Verifiable Rewards (RLVR) by leveraging a training-free probing mechanism to extract intermediate confidence and correctness signals, synthesizing them into a Step Potential metric.
- The Step Potential is integrated into the RL policy optimization via a Potential Saturation Penalty to curb redundant post-solution checking and Potential Difference Shaping to reward pivotal reasoning steps.
- The approach consistently improves accuracy and substantially reduces response length across mathematical benchmarks by mitigating pathological behaviors like Over-Checking and Right-to-Wrong failures.

---

[ETR: Outcome-Guided Elastic Trust Regions for Policy Optimization](http://arxiv.org/abs/2601.03723)

- ETR (Elastic Trust Regions): introduces a dynamic constraint mechanism that replaces the static clipping threshold in Group Relative Policy Optimization (GRPO) with a dual-level elastic boundary, aligning optimization constraints with signal quality.
- The mechanism utilizes Micro-Level Adjustment based on advantage magnitude and Macro-Level Adjustment based on group variance to allocate larger update budgets to high-confidence paths and optimal learning zones.
- ETR consistently outperforms static GRPO baselines on mathematical reasoning benchmarks, achieving superior accuracy and mitigating policy entropy degradation without introducing computational overhead.

---

#### 6th January 2026

[Automated Semantic Rules Detection (ASRD) for Emergent Communication Interpretation](http://arxiv.org/abs/2601.03254)

- ASRD (Automated Semantic Rules Detection): introduces an algorithm designed to interpret emergent communication by extracting semantic rules from messages exchanged by Lewis Game agents, using data grouping, constant position identification, combination analysis, and global constant removal.
- The algorithm links extracted message patterns to specific attributes and hyperattributes derived from the input image data, significantly simplifying the subsequent analysis of emergent language semantics.
- The method operates on messages generated by a Speaker agent (ResNet-18 encoder, LSTM) and interpreted by a Listener agent in a referential communication task.

---

[InfiAgent: An Infinite-Horizon Framework for General-Purpose Autonomous Agents](http://arxiv.org/abs/2601.03204)

- InfiAgent: introduces an infinite-horizon framework for autonomous agents, featuring a Multi-Level Agent Hierarchy (Hierarchical task decomposition), File-Centric Architecture (Externalized persistent state), Bounded Reasoning Context Reconstruction (Fixed-size context window), Periodic State Consolidation (Refreshes context snapshot), External Attention Pipeline (Processes massive documents), Tool Calls (Query-driven execution), Ten-Step Strategy & State Update (Bounded thinking module), Batch File Operations & Shared Context (Tool parameter management), and Unlimited Runtime (Stable long-horizon execution).
- The framework achieves stable long-horizon reasoning by explicitly separating persistent task state, stored in a file-centric workspace, from the strictly bounded LLM reasoning context, which only includes recent actions and a state snapshot.
- The hierarchical architecture, comprising Alpha (orchestrator), Domain, and Atomic agents, combined with the external attention mechanism, reduces error propagation and offloads heavy information processing, enabling smaller LLMs to achieve competitive performance.

---

[A Fast Semidefinite Convex Relaxation for Optimal Control Problems With Spatio-Temporal Constraints](http://arxiv.org/abs/2601.03055)

- Fast SDR (Fast Semidefinite Programming Convex Relaxation): introduces a novel convex relaxation framework for solving optimal control problems (OCPs) with coupled spatio-temporal constraints using TS/DMS/NLP Formulation/SDR/Sparse-structure-aware Lifting/IPOPT Refinement.
- The approach transforms the inherently nonconvex OCP, which jointly optimizes trajectory and crossing times, into a structured NLP using time-scaling and direct multiple shooting.
- The sparse-structure-aware SDR method tightly approximates the nonconvex components, achieving high optimality and an order-of-magnitude reduction in computational time compared to standard SDR.

---

[A Bi-directional Adaptive Framework for Agile UAV Landing](http://arxiv.org/abs/2601.03037)

- Bi-directional Adaptive Framework: introduces a cooperative landing system that treats the quadrotor and the variable-attitude platform as coupled active agents, utilizing a Two-Stage Cooperative Planning Algorithm for time-optimal trajectory generation and active attitude synchronization.
- The framework breaks the conventional track-then-descend paradigm by parallelizing alignment and descent phases, enabling aggressive "sprint-then-brake" maneuvers for rapid state synchronization within transient landing windows.
- Agility is achieved through the Terminal Attitude Heuristic Selection Method (Stage One) and the subsequent generation of a full 3D continuous trajectory via the Trajectory Planner (Stage Two), ensuring dynamic feasibility and minimizing recovery time.

---

[The Path Ahead for Agentic AI: Challenges and Opportunities](http://arxiv.org/abs/2601.02749)

- IAAIF (Integrative Agentic AI Framework): introduces an integrative framework describing core components that bridge LLMs with autonomous behavior, utilizing LLM Brain (Reasoning and Planning), Perception (Structured input conversion), Action (Plan execution), Memory (Persistence and retrieval), Environment (External systems), and Feedback Loop (Continuous adaptation) to enable goal-driven systems.
- The architecture operates via a continuous perception-reasoning-action feedback loop, allowing agents to decompose complex tasks and refine actions based on environmental outcomes.
- This architectural transition moves LLMs from passive text generators to adaptive, interactive agents capable of long-term planning and tool use.

---

[BAYESIAN ORCHESTRATION OF MULTI-LLM AGENTS FOR COST-AWARE SEQUENTIAL DECISION-MAKING](http://arxiv.org/abs/2601.01522)

- Bayesian Orchestration of Multi-LLM Agents: introduces a mathematically principled framework that treats multiple LLMs as approximate likelihood functions for cost-aware sequential decision-making, including 5 diverse LLMs, contrastive prompting, median estimate aggregation, explicit prior specification, sequential belief updating, asymmetric costs, adaptive screening, and expected cost minimization.
- The framework achieves a 34% cost reduction and 45% fairness improvement compared to single-LLM baselines by correcting fundamental mathematical deficiencies inherent in discriminative LLM architectures.
- Key capabilities enabled by this generative approach include robust multi-LLM aggregation for bias mitigation, explicit prior correction for domain adaptation, and principled Value-of-Information calculations for adaptive evidence gathering.

---

[LATENT SPACE REINFORCEMENT LEARNING FOR MULTI-ROBOT EXPLORATION](http://arxiv.org/abs/2601.01139)

- Decentralized Hierarchical DRL Framework: introduces a scalable, noise-resilient solution for multi-robot exploration using a Procedural Map Generation Algorithm, a Convolutional Autoencoder, a Weighted Consensus Mechanism, and a Hierarchical DRL Architecture.
- The framework utilizes a pre-trained convolutional autoencoder to compress high-resolution occupancy grids into compact latent state vectors, effectively overcoming the curse of dimensionality for DRL agents.
- Robust decentralized coordination is achieved via a weighted consensus mechanism that fuses shared encoded self-maps using a tuneable trust parameter ($\beta$) to mitigate error accumulation from noisy communication.

---

[AGENTIC PHYSICAL AI TOWARD A DOMAIN-SPECIFIC FOUNDATION MODEL FOR NUCLEAR REACTOR CONTROL](http://arxiv.org/abs/2512.23292)

- APAI (Agentic Physical AI): introduces a framework for nuclear reactor control using a compact LLM (SmolLM2-360M) trained via a Two-Phase Curriculum and validated through Physical AI (outcome-centric validation) in the KOMODO Simulator.
- The system achieves high reliability (97.4% success at $\pm 5\%$) and eliminates catastrophic tail risk through data scaling (1K to 100K scenarios), inducing a 500x variance collapse.
- The approach defines correctness by physics execution rather than parameter imitation, enabling the model to autonomously discover and concentrate on robust, low-variance control strategies.

---

[SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents](http://arxiv.org/abs/2512.22322)

- SmartSnap: introduces a paradigm shift from passive, post-hoc verification to proactive, in-situ self-verification by the Self-Verifying Agent, guided by 3C Principles, using an Augmented Action Space, Curated Snapshot Evidences, Composite Reward Function, and Environment.
- The Self-Verifying Agent executes complex GUI tasks and proactively curates a minimal, decisive set of snapshot evidences, which are then judged by a general LLM-as-a-Judge Verifier.
- This proactive evidence seeking reduces verification cost and LLM hallucination risk while facilitating efficient agent training via intrinsic Reward Shaping based on structured verifier feedback.

---

[FIRE-VLM: A Vision-Language-Driven Reinforcement Learning Framework for UAV Wildfire Tracking in a Physics-Grounded Fire Digital Twin](http://arxiv.org/abs/2601.03449)

- FIRE-VLM: introduces an end-to-end VLM-guided reinforcement learning framework trained within a high-fidelity, physics-grounded wildfire digital twin, combining high-fidelity wildfire simulation, UAV dynamics, dual-view sensing, VLM semantic guidance, and a PPO policy for robust firefront tracking.
- The framework utilizes a CLIP-style VLM operating on dual-view RGB observations (top-down and angled) to generate semantic alignment scores and directional likelihoods, which are integrated into the PPO reward function.
- This VLM-guided reward shaping, combining physics-based incentives with semantic understanding, significantly improves time-to-detection and time-in-FOV compared to purely RL baselines across kilometer-scale fires.

---

[Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts](http://arxiv.org/abs/2601.03315)

- Autonomous Research Pipeline: introduces a case study of four end-to-end attempts to autonomously generate ML research papers using six LLM agents mapped to the scientific workflow stages.
- The system, primarily using Gemini 2.5 Pro and Claude Code, documented six recurring failure modes, including bias toward training data, implementation drift, and memory/context degradation across long-horizon tasks.
- Based on the attempts (one success, three failures), the authors propose four design principles for robust AI-scientist systems, emphasizing verification, gradual grounding, and planning for failure and recovery.

---

[ULTRALOGIC: Enhancing LLM Reasoning through Large-Scale Data Synthesis and Bipolar Float Reward](http://arxiv.org/abs/2601.03205)

- ULTRALOGIC (Code-based Solving Framework): introduces an innovative framework for large-scale, automated production of high-quality reasoning data, built upon a core Code-based Solving Framework methodology.
- The framework decouples the logical core of a problem from its natural language expression using Input Code and Solution Code, ensuring logical correctness and verifiability across ten difficulty levels.
- To enhance training efficiency and mitigate binary reward sparsity, the Bipolar Float Reward (BFR) mechanism is introduced, utilizing graded penalties to guide models toward global logical optima during GRPO training.

---

#### 5th January 2026

[Textual Explanations and Their Evaluations for Reinforcement Learning Policy](http://arxiv.org/abs/2601.02514)

- Proposed Framework: introduces a novel Explainable Reinforcement Learning (XRL) system for generating textual explanations and evaluating them quantitatively, utilizing an LLM (Interprets user questions), Predicate Functions (Discretizes state features), and a Clustering-Based Summarizer (Generates textual explanations).
- The framework converts the generated textual explanations into transparent rules via Rules Extraction, enabling systematic evaluation of properties, fidelity, and performance by comparing them against replay data and deploying them in the Environment.
- The system addresses limitations of existing methods by focusing on frequent conditions and suppressing duplicates, and includes refinement techniques to minimize conflicting information and maximize F1 scores.

---

[Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents](http://arxiv.org/abs/2601.02314)

- Project Ariadne: introduces a novel XAI framework that utilizes Structural Causal Models (SCMs) and counterfactual logic to audit the causal integrity of LLM agent reasoning.
- The framework performs hard interventions on intermediate reasoning nodes, systematically inverting logic or negating premises, to generate a counterfactual trace and answer.
- By quantifying the Causal Sensitivity Score ($\phi$) between the original and counterfactual answers, the framework detects Causal Decoupling, where reasoning traces function as post-hoc justifications.

---

[Confidence Estimation for LLMs in Multi-turn Interactions](http://arxiv.org/abs/2601.02179)

- P(SUFFICIENT) Multi-turn Confidence Framework: introduces the first systematic study of confidence estimation for LLMs in multi-turn interactions, establishing an evaluation framework grounded in Calibration (confidence matches empirical correctness) and Monotonicity (confidence increases with information).
- The framework utilizes novel metrics, including InfoECE (length-normalized Expected Calibration Error) and Kendall's $\tau$ (monotonic trend measure), alongside a Hinter-Guesser paradigm for generating controlled evaluation datasets.
- Evaluation of existing methods reveals struggles with calibration and monotonicity, while the proposed logit-based probe, P(SUFFICIENT), achieves comparatively better performance by tracking the sufficiency of accumulated evidence.

---

[Agentic AI in Remote Sensing: Foundations, Taxonomy, and Emerging Systems](http://arxiv.org/abs/2601.01891)

- Agentic AI in Remote Sensing Taxonomy: introduces a unified taxonomy distinguishing between single-agent copilots and multi-agent systems, analyzing architectural foundations such as planning mechanisms, retrieval-augmented generation, and memory structures.
- The proposed ecosystem is structured into four key components: Foundations (data acquisition and models), Agents (copilots and orchestrators), Systems & Platforms (RAG, Tools, Memory), and Evaluation (benchmarks for planning and reasoning).
- The survey critically examines limitations in geospatial grounding, safety, and orchestration, outlining a strategic roadmap for robust, autonomous geospatial intelligence development using Earth-native models.

---

[Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents](http://arxiv.org/abs/2601.01885)

- AgeMem (Agentic Memory): introduces a unified framework that integrates LTM and STM management directly into the LLM agent's policy using explicit tool-based operations.
- The framework utilizes a Three-Stage Progressive RL strategy and Step-wise GRPO to enable end-to-end optimization of memory behaviors despite sparse and discontinuous rewards.
- By exposing memory operations (ADD, RETRIEVE, SUMMARY, FILTER) as actions, the LLM agent autonomously decides when and how to manage context and persistent knowledge.

---

[JENIUS AGENT: TOWARDS EXPERIENCE-DRIVEN ACCURACY OPTIMIZATION IN REAL-WORLD SCENARIOS](http://arxiv.org/abs/2601.01857)

- Jenius Agent: introduces an experience-driven autonomous agent framework grounded in real-world practice, integrating adaptive prompt generation, context-aware tool orchestration, and hierarchical memory management to optimize task accuracy and efficiency.
- The framework uses the LLM as a central orchestrator in a feedback-driven loop, coordinating the three core modules to generate system prompts, select relevant tools, and filter historical context.
- Key innovations include dynamic prompt adaptation based on task state, semantic retrieval and inflection point-based filtering for tool selection, and layered memory compression to mitigate token cost and context noise.

---

[ARIES: A Scalable Multi-Agent Orchestration Framework for Real-Time Epidemiological Surveillance and Outbreak Monitoring](http://arxiv.org/abs/2601.01831)

- ARIES (Agentic Retrieval Intelligence for Epidemiological Surveillance): introduces a scalable, hierarchical multi-agent orchestration framework for real-time epidemiological surveillance, utilizing a Manager Agent to delegate tasks to specialized sub-agents.
- The system mimics an Emergency Operations Center (EOC) structure, employing agents like the Senior Medical Scientist and CDC Data Analyst to query specific, high-integrity data sources (PubMed, CDC WONDER, WHO DONs).
- Built on the CrewAI framework, ARIES automates data extraction and logical synthesis, providing specialized reasoning to identify emergent threats and mitigate the knowledge gap inherent in generalized LLMs.

---

[Sparse Threats, Focused Defense: Criticality-Aware Robust Reinforcement Learning for Safe Autonomous Driving](http://arxiv.org/abs/2601.01800)

- CARRL (Criticality-Aware Robust Reinforcement Learning): introduces a novel DRL-based adversarial training approach for robust autonomous driving, formulated as a General-Sum Markov Game between the Risk Exposure Adversary and the Risk-Targeted Robust Agent.
- The REA uses Decoupled Optimization to execute criticality-aware attacks focused on sparse safety-critical moments, while the RTRA employs a Dual Replay Buffer and Consistency-Constrained Policy Optimization to learn robust policies despite data scarcity.
- This framework achieves superior robustness by explicitly addressing the asymmetry between the agent and adversary objectives and focusing defensive capacity on high-risk scenarios, significantly reducing the collision rate.

---

[ALIGNDRIVE: ALIGNED LATERAL-LONGITUDINAL PLANNING FOR END-TO-END AUTONOMOUS DRIVING](http://arxiv.org/abs/2601.01762)

- AlignDrive: introduces a novel cascaded planning paradigm where longitudinal planning is explicitly conditioned on the predicted lateral drive path, enabling coordinated and collision-aware lateral and longitudinal planning.
- The framework reformulates longitudinal planning as a simpler 1D displacement prediction problem along the drive path, allowing the model to focus capacity on dynamic interactions rather than static geometry.
- A planning-oriented data augmentation strategy simulates rare safety-critical events by inserting agents and relabeling longitudinal targets, substantially improving collision avoidance capabilities.

---

[Structural Representations for Cross-Attack Generalization in AI Agent Threat Detection](http://arxiv.org/abs/2601.01723)

- GMF (Gated Multi-View Fusion): introduces structural tokenization, encoding execution-flow patterns (tool calls, arguments, observations) rather than conversational content, dramatically improving cross-attack generalization against structural threats.
- The approach uses parallel BiLSTM encoders to process both structural and conversational representations, which are adaptively combined via a learned gate for robust performance across diverse attack families.
- Structural tokenization achieved 39–71 AUC point gains on unseen structural attacks (tool hijacking, data exfiltration) and unknown attacks, confirming that AI agent security is fundamentally a structural problem.

---

[TravelBench: A Broader Real-World Benchmark for Multi-Turn and Tool-Using Travel Planning](http://arxiv.org/abs/2512.22673)

- TravelBench: introduces a comprehensive, real-world benchmark for LLM agents in travel planning, featuring Single-Turn, Multi-Turn, and Unsolvable subtasks, supported by real user data and a Toolkit of 10 real travel tools.
- The benchmark utilizes a reproducible sandbox with a Tool-Call Cache for stable tool invocation and employs an LLM-as-a-Judge Protocol, including a Tool-Use Penalty and Meta-Judge for robust evaluation.
- It is the first travel planning benchmark to incorporate profile-based implicit preferences and multi-turn user-agent interaction to elicit requirements and test capability boundaries.

---

[AgentMark: Utility-Preserving Behavioral Watermarking for Agents](http://arxiv.org/abs/2601.03294)

- AgentMark: introduces a behavioral watermarking framework that embeds multi-bit identifiers into an agent's planning process using Elicit Behavioral Probability Output (explicit probability list), Behavior Distribution Generate (estimate implicit policy), Keyed Distribution Preserving Watermark (core embedding mechanism), Robustness Payload Coding (erasure-resilient recovery), Keyed Pseudorandom Source (synchronize encoding/decoding), Distribution-Preserving Sampling (preserve marginal distribution), Selected Behavior (watermarked planning choice), Execution Action (action carried out), Memory Module (agent context storage), and Environment (agent interaction space), ensuring utility preservation under black-box APIs.
- The framework addresses the challenge of watermarking high-level planning behaviors by eliciting an explicit behavior distribution from the LLM agent and applying distribution-preserving conditional sampling to embed the watermark without shifting the agent's planning policy.
- AgentMark-F, a concrete instantiation, utilizes differential recombination and cyclic-shift uniform encoding, combined with RLNC for robust recovery of provenance bits from partial trajectories subject to erasure or truncation.

---

#### 4th January 2026

[Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage](http://arxiv.org/abs/2601.01685)

- Generative Montage (GM): introduces the first cognitive collusion attack framework that constructs deceptive narratives using a Writer (Narrative Synthesis), Editor (Montage Sequencing), and Director (Adversarial Debate) to manipulate LLM agents' beliefs via public, truthful evidence.
- The framework exploits LLMs' tendency for narrative coherence (overthinking) by strategically sequencing truthful evidence fragments (Montage Sequencing) to induce spurious causal inferences, resulting in the internalization of a global lie from local truths.
- Experiments using the CoPHEME dataset show pervasive vulnerability across 14 LLM families, with attack success rates reaching 74.4% for proprietary models, and enhanced reasoning capabilities paradoxically increasing susceptibility.

---

[DRIVINGGEN: A COMPREHENSIVE BENCHMARK FOR GENERATIVE VIDEO WORLD MODELS IN AUTONOMOUS DRIVING](http://arxiv.org/abs/2601.01528)

- DrivingGen: introduces a comprehensive benchmark for generative video world models in autonomous driving, featuring a Diverse Driving Dataset (varied conditions/behaviors), Generative World Models (models under evaluation), Evaluation Metrics (4 comprehensive sets), and a SLAM Pipeline (robust trajectory extraction).
- The benchmark addresses limitations in existing evaluations by introducing novel metrics that jointly assess visual realism, trajectory plausibility (Fréchet Trajectory Distance, FTD), temporal coherence, and controllability (Average Displacement Error, ADE, and Dynamic Time Warping, DTW).
- DrivingGen facilitates reliable, controllable, and deployable driving world models by providing a unified evaluation framework that exposes critical trade-offs between visual quality and physical consistency across 14 state-of-the-art models.

---

[KGCE: Knowledge-Augmented Dual-Graph Evaluator for Cross-Platform Educational Agent Benchmarking with Multimodal Language Models](http://arxiv.org/abs/2601.01366)

- KGCE (Knowledge-Augmented Dual-Graph Evaluator): introduces a novel cross-platform educational agent benchmarking platform integrating a domain-specific Knowledge Base and a Dual-Graph Evaluator for fine-grained assessment.
- The system utilizes MLMs (GPT-4o, Qwen-VL, Gemini) within Windows and Android agents, supported by the Knowledge Base, to execute complex, multi-step educational tasks involving proprietary software.
- The Dual-Graph Evaluator, comprising the Task Completeness Graph and the Execution Efficiency Graph, provides eight fine-grained metrics to quantify both task completion quality and execution path efficiency.

---

[Towards LLM-enabled autonomous combustion research: A literature-aware agent for self-corrective modeling workflows](http://arxiv.org/abs/2601.01357)

- FlamePilot: introduces an LLM agent for autonomous combustion research, integrating an LLM Agent (Workflow orchestration), Self-corrective Execution Loop (Error resolution, refinement), Expertise Toolkit (CFD execution capabilities), and Domain Literature Knowledge (Structured scientific findings) to automate and refine complex CFD workflows.
- The architecture uses atomic tools for robust execution across standard (OpenFOAM) and extended (DeepFlame) CFD frameworks, achieving a perfect 1.0 executability score on the FoamBench-Advanced benchmark.
- By grounding decisions in dynamically extracted scientific literature, the agent functions as a transparent, human-supervised research copilot, accelerating discovery while upholding scientific rigor.

---

[Neural-network-based Self-triggered Observed Platoon Control for Autonomous Vehicles](http://arxiv.org/abs/2601.01335)

- NSOPC: introduces an adaptive consensus tracking control framework for nonlinear multi-agent systems using backstepping design, a distributed observer, RBFNNs, and a self-triggered communication mechanism.
- The framework addresses uncertain dynamics and intermittent communication by using RBFNNs to approximate unknown nonlinearities and a distributed observer to estimate states based on limited, intermittent measurements.
- The self-triggered mechanism calculates the next control update instant based on the current control signal, guaranteeing a strictly positive minimum inter-event time and optimizing communication efficiency.

---

[Digital Twin AI: Opportunities and Challenges from Large Language Models to World Models](http://arxiv.org/abs/2601.01321)

- UFSF (Unified Four-Stage Framework): introduces a systematic characterization of AI integration across the digital twin lifecycle, including Modeling, Mirroring, Intervening, and Autonomous Management stages.
- The framework evolves DTs from passive simulation tools to proactive, self-improving cognitive systems by leveraging physics-informed AI, generative AI, predictive AI, and agentic AI.
- Autonomous management is enabled by LLMs for natural language interaction and foundation models for multimodal perception, supporting complex decision-making and closed-loop control.

---

[Automated Post-Incident Policy Gap Analysis via Threat-Informed Evidence Mapping using Large Language Models](http://arxiv.org/abs/2601.03287)

- Agentic Post-Incident Review Workflow: introduces an LLM-driven, agentic pipeline integrating log analysis, threat attribution, policy retrieval, policy evaluation, and report generation, designed to autonomously identify security policy gaps.
- The framework uses GPT-4o for reasoning, LangGraph for multi-agent orchestration, and LlamaIndex for traceable policy retrieval, ensuring explicit evidence-to-policy traceability grounded in the MITRE ATT&CK framework.
- This approach augments cybersecurity post-incident reviews by providing evidence-grounded, auditable insights to improve efficiency, consistency, and governance alignment.

---

#### 3rd January 2026

[Harnessing Environmental Memory with Reinforcement Learning in Open Quantum Systems](http://arxiv.org/abs/2601.01252)

- RL Framework: introduces a reinforcement learning approach for enhancing non-Markovianity in a driven open quantum system by maximizing the BLP measure via direct trajectory-based feedback.
- The framework utilizes Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) agents to learn continuous control fields ($\Omega(t)$) that exploit environmental memory by coordinating backflow across multiple revival windows.
- Unlike gradient-based Optimal Control Theory (OCT), which concentrates on amplifying a single backflow peak, the RL policies discover distributed-backflow strategies, leading to substantially higher total integrated non-Markovianity.

---

[ROBOPHD: SELF-IMPROVING TEXT-TO-SQL THROUGH AUTONOMOUS AGENT EVOLUTION](http://arxiv.org/abs/2601.01126)

- RoboPhD: introduces a system where AI agents autonomously conduct research to improve Text-to-SQL performance, featuring Evolution AI (Generates new agents), SQL Generation Agent (Performs Text-to-SQL), Database Analysis Tool (Offline Python script analysis), Eval Instructions (SQL generation guidance), Evolutionary Loop (Iterative improvement cycle), ELO-based Selection Mechanism (Ranks agent performance), Evaluation Results (Feedback for Evolution AI), and Universal Verification (Self-correction and validation).
- The system employs a closed-loop evolution cycle, using an ELO-based selection mechanism to drive survival-of-the-fittest dynamics and handle non-transitivity in agent performance across different databases.
- The evolved agent consists of two deployable artifacts—a deterministic Python database analysis tool and SQL generation instructions—allowing for cost-effective deployment and demonstrating larger performance gains on cheaper LLMs.

---

[Performance and Security Aware Distributed Service Placement in Fog Computing](http://arxiv.org/abs/2601.01125)

- SPA-DDRL: introduces a distributed broker-learner architecture for joint security and performance optimization in Fog computing, integrating LSTM networks, Prioritized Experience Replay, and off-policy correction mechanisms.
- The framework formulates the service placement problem as a weighted multi-objective optimization task minimizing latency and maximizing a security score derived from a novel three-tier security hierarchy.
- The distributed design enables scalable security-performance optimization across heterogeneous Fog environments, achieving significant improvements in response time and convergence speed compared to baselines.

---

[Harm in AI-Driven Societies: An Audit of Toxicity Adoption on Chirper.ai](http://arxiv.org/abs/2601.01090)

- CTAA (Chirper.ai Toxicity Audit Approach): introduces a large-scale empirical audit of toxicity adoption on the fully AI-driven social platform Chirper.ai, utilizing LLM-driven Agents, Stimuli, Responses, a Detoxify Classifier, Influence-Driven Response Rate, Spontaneous Response Rate, and a Gemini-2.0-flash Evaluator.
- The audit models agent interactions in terms of toxic stimuli (posts) and toxic responses (comments), finding that cumulative toxic exposure significantly increases the probability of toxic responding.
- The introduced metrics, IRR and SRR, reveal a strong trade-off between exposure-driven and autonomous toxicity, and the number of toxic stimuli alone accurately predicts whether an agent will eventually produce toxic content.

---

[Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space](http://arxiv.org/abs/2512.21887)

- ANWM (Aerial Navigation World Model): introduces a world model for UAV navigation that predicts long-horizon visual observations conditioned on past frames and 3D actions, enabling trajectory ranking based on semantic plausibility. 
- To handle the complex 3D action space and ensure spatio-temporal consistency, ANWM incorporates the Future Frame Projection (FFP) module, which provides coarse geometric priors by warping past frames into future viewpoints. 
- The model uses a Conditional Diffusion Transformer (CDiT) backbone and Independent Latent Modulation (ILM) to separately process real past frames and the synthesized projected future frame prior for robust long-range generation.

---

[A Chromatographic Process Design and Optimization Platform Powered by Large Language Models: A Case Application on Extract of Ginkgo Biloba Leaf](http://arxiv.org/abs/2601.03702)

- ChromR (Chromatographic Process Design and Optimization Platform): integrates ChromLLM (Domain-specific LLM), Qwen LLM (Intent recognition/Synthesis), a Multi-agent system (Workflow coordination), and an Automated chromatography device (Physical experimentation) to autonomously design and optimize chromatographic processes.
- The multi-agent system comprises four specialized agents—domain knowledge answering (Agent A), experimental design (Agent B), execution (Agent C), and data analysis (Agent D)—that manage the entire workflow.
- By leveraging the domain-specific LLM enhanced with RAG and integrating with automated hardware, the platform reduces reliance on expert knowledge and decreases process development time to approximately one week.

---

#### 2nd January 2026

[Early-Stage Prediction of Review Effort in AI-Generated Pull Requests](http://arxiv.org/abs/2601.00753)

- Circuit Breaker Triage Model: introduces a system for predicting high-review-effort AI-generated Pull Requests (PRs) using the AIDev Dataset, Creation-Time Features, and a LightGBM Model to identify High Cost PR Target for a Triage Mechanism.
- The model achieves high discrimination (AUC 0.9571) using only static complexity signals (e.g., patch size, files touched) available at creation time (T0), confirming that structural complexity, not semantic content, dictates review burden.
- The framework intercepts 69% of total review effort at a 20% review budget allocation, enabling maintainers to triage the expensive tail of agent contributions with zero latency.

---

[Bayesian Inverse Games with High-Dimensional Multi-Modal Observations](http://arxiv.org/abs/2601.00696)

- BIG-VAE (Structured Variational Autoencoder for Bayesian Inverse Games): introduces an approximate Bayesian inference framework that trains a structured VAE with an embedded differentiable Nash game solver to infer posterior distributions over hidden agent objectives from multimodal observations.
- The architecture fuses high-dimensional visual cues (images) and low-dimensional partial-state observations (trajectories) via a shared latent space, enabling uncertainty quantification and multi-modal inference.
- By amortizing the inference process offline, the framework supports real-time posterior sampling at test time, facilitating safer and more efficient uncertainty-aware downstream motion planning compared to MLE methods.

---

[Trajectory Guard - A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI](http://arxiv.org/abs/2601.00516)

- Trajectory Guard: introduces a Siamese Recurrent Autoencoder with a hybrid loss function to perform real-time anomaly detection in LLM agent trajectories, addressing both contextual and structural failures.
- The architecture utilizes a Task Tower for contextual fit via contrastive learning and a Trajectory Tower with a GRU autoencoder for structural validity via sequence reconstruction.
- Operating at 32 ms latency, the model runs significantly faster than LLM Judge baselines, making it suitable for production safety verification in agentic systems.

---

#### 1st January 2026

[The Rise of Agentic Testing: Multi-Agent Systems for Robust Software Quality Assurance](http://arxiv.org/abs/2601.02454)

- ATA (Agentic Testing Architecture): introduces a multi-agent, closed-loop testing framework where the Test Generation Agent, Execution and Analysis Agent, and Review and Optimization Agent collaboratively generate, execute, analyze, and refine tests until convergence criteria are met.
- The system leverages sandboxed execution, detailed failure reporting, and iterative regeneration guided by reinforcement signals derived from coverage metrics and execution outcomes.
- This architecture establishes autonomous, self-correcting software Quality Assurance by significantly reducing invalid tests and manual review effort compared to static LLM baselines.

---

[When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents](http://arxiv.org/abs/2601.00513)

- PVS (Process Verification System): introduces a methodology to quantify the "Right-for-Wrong-Reasons" (RWR) phenomenon in small LLMs using the Reasoning Integrity Score (RIS) and a fast Distilled Verifier.
- The system analyzes 10,734 reasoning traces across three 7-9B LLMs and diverse tasks, revealing that 50-69% of correct outputs contain fundamentally flawed reasoning invisible to standard accuracy metrics.
- The analysis shows that Retrieval-Augmented Generation (RAG) acts as essential external scaffolding to improve integrity, while meta-cognitive prompts actively harm performance due to "pseudo-reflection" in sub-10B models.

---

[Automated decision-making by chemical echolocation in active droplets](http://arxiv.org/abs/2601.00480)

- Chemical Echolocation: introduces a generic mechanism enabling synthetic agents, such as active droplets, to achieve autonomous navigational decision-making and maze solving by leveraging self-generated chemo-hydrodynamic signals.
- The mechanism relies on the agent acting as a chemical source and sensor, where signals reflected by dead ends create a "chemical echo" that elicits a direct physical response (chemorepulsion) to steer the agent toward open paths.
- This strategy provides a robust alternative to traditional source-seeking methods, remaining effective and efficient even in large mazes without requiring external guidance or complex information processing machinery.

---

[Security in the Age of AI Teammates: An Empirical Study of Agentic Pull Requests on GitHub](http://arxiv.org/abs/2601.00477)

- Empirical Study Methodology: introduces a multi-stage process analyzing 33,596 Agentic-PRs from the AIDev dataset, including Keyword-based PR Filtering, Manual Validation, Qualitative Open Coding, Statistical Analysis, Structured Prediction Models, and Text-Based Prediction Models, to characterize autonomous coding agents' security contributions and human review dynamics.
- The study identifies that security-related Agentic-PRs (3.85% of total) receive heightened human scrutiny, exhibiting lower merge rates (61.5%) and substantially longer review latency (median 3.92 hours) compared to non-security PRs (median 0.11 hours).
- PR rejection is found to be more strongly associated with complexity and verbosity (e.g., PR size, title length) than with explicit security terminology, suggesting reviewers prioritize clarity and scope over security topic presence alone.

---

[Space Debris Removal using Nano-Satellites controlled by Low-Power Autonomous Agents](http://arxiv.org/abs/2601.00465)

- Low-Power Autonomous Agents (LPAA) System: introduces a multi-agent system using two nano-satellite Free-Flyers (FFs) equipped with embedded BDI agents running on nRF52840 microcontrollers to synchronously de-orbit space debris.
- The agents communicate wirelessly using the low-power OpenThread protocol and CoAP application layer, synchronizing their actions via a Mothership server acting as a communication hub.
- The system demonstrates energy-efficient autonomy for resource-constrained embedded devices, validated through experiments on the ELISSA air-bearing test-bed.

---

[Mapping Human Anti-collusion Mechanisms to Multi-agent AI](http://arxiv.org/abs/2601.00360)

- MAAI Anti-Collusion Framework: introduces a taxonomy of human anti-collusion mechanisms and maps them to concrete interventions for Multi-Agent AI systems, including Sanctions, Leniency & Whistleblowing, Monitoring & Auditing, Market Design & Structural, and Governance.
- The framework addresses key challenges in AI collusion, such as attribution, identity fluidity, and adversarial adaptation, by proposing architectural components like Overseer Agents and Telemetry-first Design.
- Implementation approaches for LLM-based agents include reward penalties, capability restrictions, interaction protocol constraints, and mandatory transparency requirements like Model Cards.

---

[Bio-inspired Agentic Self-healing Framework for Resilient Distributed Computing Continuum Systems](http://arxiv.org/abs/2601.00339)

- ReCiSt (Bio-inspired Agentic Self-healing Framework for Resilient Distributed Computing Continuum Systems): introduces a bio-inspired agentic self-healing framework for Distributed Computing Continuum Systems (DCCS) that maps biological wound healing phases (Hemostasis, Inflammation, Proliferation, Remodeling) onto four computational layers: Containment, Diagnosis, Meta-Cognitive, and Knowledge.
- The framework utilizes LM-powered agents, including monitoring agents and reasoning micro-agents, to perform autonomous fault isolation, causal diagnosis, adaptive recovery, and long-term knowledge consolidation.
- The Meta-Cognitive Layer regulates internal reasoning via migratory micro-agents and feedback mechanisms, while the Knowledge Layer supports adaptive knowledge sharing through local and global Rendezvous Points.

---

[Can Optimal Transport Improve Federated Inverse Reinforcement Learning?](http://arxiv.org/abs/2601.00309)

- OT-FIRL (Optimal Transport-based Federated Inverse Reinforcement Learning): introduces a federated IRL framework that fuses locally learned MaxEnt reward functions using an entropically regularized Wasserstein barycenter for geometry-aware aggregation.
- This approach treats local rewards as probability distributions over a shared state-action support, yielding stable and transferable global reward estimates across heterogeneous agents and environments.
- The framework provides theoretical stability and parameter-error bounds, demonstrating that barycentric fusion contracts toward the true reward function under bounded local estimation error.

---

[FLASHINFER-BENCH: BUILDING THE VIRTUOUS CYCLE FOR AI-DRIVEN LLM SYSTEMS](http://arxiv.org/abs/2601.00227)

- FlashInfer-Bench: introduces a standardized, closed-loop framework connecting kernel generation, benchmarking, and deployment for AI-driven LLM systems.
- The system uses FlashInfer Trace, a unified JSON schema, to communicate kernel definitions, workloads, solutions, and evaluation results consistently.
- A dynamic substitution mechanism, `flashinfer_bench.apply()`, seamlessly injects the best-performing, validated kernels into production LLM engines like SGLang and vLLM.

---

[Device-Native Autonomous Agents for Privacy-Preserving Negotiations](http://arxiv.org/abs/2601.00911)

- DNAAS (Device-Native Autonomous Agent System): introduces a device-native agentic AI system for autonomous, privacy-preserving negotiations using a six-layer architecture that integrates cryptographic and on-device reasoning components.
- The system operates exclusively on user hardware, employing zero-knowledge proofs (ZK proofs) for secure multi-party bargaining and Explainable Memory to generate tamper-evident cryptographic audit trails for compliance.
- Efficiency and mobility are achieved via World Model Distillation for on-device LLM reasoning, Selective State Transfer for cross-device continuity, and Model-Aware Offloading for optimal task placement.

---

[AI-Native Integrated Sensing and Communications for Self-Organizing Wireless Networks: Architectures, Learning Paradigms, and System-Level Design](http://arxiv.org/abs/2601.02398)

- A2ISAC-SON (AI-Native Integrated Sensing and Communications for Self-Organizing Wireless Networks): introduces a comprehensive survey of AI-native ISAC-enabled self-organizing wireless networks, covering ISAC System Architectures, ISAC Waveform Design, Sensing-Comm Co-Design, AI-Native Control Paradigms, Self-Organizing Functions, Sensing-Assisted Communication, and Communication-Assisted Sensing.
- The architecture unifies radio sensing and data communication (ISAC) with AI-driven control loops to create perceptive, autonomous networks capable of self-optimization and adaptation for 6G systems.
- Key technical areas reviewed include the fundamental trade-offs in dual-functional waveform design and the application of DRL and GNNs for dynamic network control, mobility management, and resource allocation.

---

[a³-Bench: A Unified Benchmark of Safety, Robustness, and Efficiency for LLM-Based UAV Agents over 6G Networks](http://arxiv.org/abs/2601.03281)

- a³-Bench: introduces a comprehensive benchmark for evaluating LLM-driven UAV autonomy as a multi-turn conversational reasoning and control problem under dynamic 6G conditions, featuring an LLM-based UAV Agent, Human Operator, and structured action protocols (MCP/A2A).
- The framework models mission execution as a language-mediated control loop, where the LLM adapts its policy based on UAV telemetry, safety constraints, and a dynamic 6G Network Context vector (slicing, latency, throughput).
- Performance is assessed using the composite $\alpha^3$ metric, which unifies six pillars (Task Outcome, Safety Policy, Tool Consistency, Interaction Quality, Network Robustness, Communication Cost) with reliability- and efficiency-normalized scores.

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


