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




#### 4th May 2026

[Scaling Federated Linear Contextual Bandits via Sketching](http://arxiv.org/abs/2605.00500)

- FSCLB: introduces a federated learning framework that utilizes matrix sketching and SVD-based determinant calculation to reduce computational and communication overhead in high-dimensional contextual linear bandits.
- The framework employs a double-sketch strategy to compress both upload and download data, effectively lowering communication costs from O(d²) to O(ld) per round.
- By integrating SCFD for sketch updates, the approach maintains asynchronous communication validity while achieving a regret bound that matches optimal non-sketched performance when the sketch size exceeds the covariance matrix rank.

---

[Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](http://arxiv.org/abs/2605.00370)

- GCL (Group Cognition Learning): introduces a two-stage governed collaboration paradigm that regulates cross-modal interaction through Routing Agent, Auditing Agent, Public-Factor Agent, and Aggregation Agent to mitigate modality dominance and spurious coupling.
- The framework employs a selective interaction stage to filter redundant information based on marginal predictive gain and a consensus formation stage to disentangle shared semantics from private modality-specific representations.
- By replacing implicit fusion with an explicit governance protocol, GCL achieves state-of-the-art performance on multimodal sentiment analysis and intent recognition benchmarks while maintaining superior robustness against noise and spurious correlations.

---

[From Prompt to Physical Actuation: Holistic Threat Modeling of LLM-Enabled Robotic Systems](http://arxiv.org/abs/2604.27267)

- LLM-Enabled Robotic Systems: introduces a hierarchical Data Flow Diagram (DFD) and STRIDE-per-interaction analysis to systematically trace end-to-end threat propagation across the perception-planning-actuation pipeline of an autonomous robot.
- The framework models the system using two trust boundaries (TB1 for edge server, TB2 for autonomous platform) to identify security risks at six boundary-crossing interaction points where untrusted inputs enter protected domains.
- The analysis integrates Conventional Cyber Threats, Adversarial Threats, and Conversational Threats to demonstrate how compromises propagate through cross-modal translation and unmediated boundary crossings to cause unsafe physical actuation.

---

[An Empirical Study of Speculative Decoding on Software Engineering Tasks](http://arxiv.org/abs/2604.26469)

- SD (Speculative Decoding) framework: introduces a systematic empirical evaluation of speculative decoding methods across diverse software engineering tasks to mitigate autoregressive inference latency.
- The study benchmarks model-based approaches like MLP Speculators and Eagle-3 against model-free methods including Prompt Lookup Decoding and Suffix Decoding to assess their effectiveness in code generation, repair, and editing.
- The research identifies that code exhibits higher contextual repetitiveness than natural language, enabling model-free methods to achieve significant acceleration, while also highlighting the noise introduced by agentic infinite loops in complex tasks.

---

#### 2nd May 2026

[The Buy-or-Build Decision, Revisited: How Agentic AI Changes the Economics of Enterprise Software](http://arxiv.org/abs/2604.26482)

- Agentic AI-augmented software development framework: introduces a conceptual model for re-evaluating enterprise software sourcing decisions by analyzing how agentic coding systems transform the governance and economics of in-house development.
- The framework shifts the "Make" option from a traditional hierarchy to a hybrid governance form, characterized by internal code ownership combined with external AI infrastructure dependency.
- It provides a typology of enterprise applications to guide strategic sourcing, identifying that while AI favors in-house development for commodity and custom applications, regulated and mission-critical systems remain better suited for external procurement.

---

#### 1st May 2026

[Can Coding Agents Reproduce Findings in Computational Materials Science?](http://arxiv.org/abs/2605.00803)

- AUTOMAT: introduces a benchmark for evaluating LLM-based agents on their ability to reproduce computational materials science claims by navigating complex, domain-specific workflows.
- The framework utilizes a Task Package (SME-curated claim, paper, metadata, artifacts) processed by a Reproduction Agent (LLM-based autonomous investigator) within an HPC Cluster (resource-controlled execution environment) to generate evidence for scientific validation.
- Performance is assessed by an Evaluator Agent (LLM-based judge for reproducibility) that utilizes an Evaluator Loop (iterative inspection of artifacts) to score the reproducibility of the findings stored in an Artifact Directory (persistent storage for logs and outputs).

---

[RunAgent: Interpreting Natural-Language Plans with Constraint-Guided Execution](http://arxiv.org/abs/2605.00798)

- RunAgent: introduces a multi-agent platform that interprets natural-language plans by enforcing stepwise execution through constraint-guided logic and dynamic selection of reasoning, tool usage, or code execution.
- The framework utilizes an agentic language with control constructs like IF, GOTO, and FORALL to bridge the flexibility of natural language with the determinism of programmatic execution.
- RunAgent incorporates robust error correction and context filtering mechanisms, ensuring reliable task completion by validating outputs against automatically derived constraints and rubrics.

---

[Meritocratic Fairness in Budgeted Combinatorial Multi-armed Bandits via Shapley Values](http://arxiv.org/abs/2605.00762)

- K-SVFair-FBF: introduces a meritocratic fairness framework for budgeted combinatorial multi-armed bandits under full-bandit feedback by utilizing K-Shapley Value, Monte Carlo Permutations, Stochastic Smoothing, Randomized Rounding Scheme (RRS), Confidence Term, and Selection Counters.
- The framework addresses the challenge of unknown individual arm contributions in full-bandit feedback by adaptively estimating K-Shapley values while mitigating noise from both valuation learning and Monte Carlo approximations.
- Theoretical analysis demonstrates that the algorithm achieves sublinear fairness regret, effectively balancing equitable arm participation with reliable learning in applications like federated learning and social influence maximization.

---

[NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](http://arxiv.org/abs/2605.00751)

- NONZERO: introduces a multi-agent MCTS framework that mitigates the combinatorial joint-action bottleneck by utilizing a low-dimensional nonlinear return surrogate and an interaction-guided proposal rule.
- The framework incorporates a hypernetwork to initialize node-specific parameters and employs the NONUCT mechanism to perform curvature-aware exploration through first- and second-order discrete differences.
- NONZERO achieves sublinear regret guarantees and demonstrates superior sample efficiency and performance across MatGame, SMAC, and SMACv2 benchmarks compared to existing model-based and model-free baselines.

---

[Position: agentic AI orchestration should be Bayes-consistent](http://arxiv.org/abs/2605.00742)

- Bayesian Agentic Orchestration Framework: introduces a principled approach for agentic AI systems by placing Bayesian decision-making at the orchestration layer rather than within individual LLMs.
- The framework utilizes a Bayesian controller to maintain beliefs over task-relevant latent variables, updating these beliefs via observation models that treat LLM outputs as evidence.
- This approach enables cost-aware, uncertainty-aware, and adaptive orchestration by selecting actions that maximize posterior expected utility while treating LLMs as non-Bayesian black-box predictors.

---

[Self-Adaptive Multi-Agent LLM-Based Security Pattern Selection for IoT Systems](http://arxiv.org/abs/2605.00741)

- ASPO: introduces a self-adaptive IoT security architecture that integrates multi-agent LLM-based reasoning with deterministic enforcement within a MAPE-K control loop to ensure safe, conflict-free, and resource-feasible mitigation selection.
- The framework separates stochastic candidate generation by LLM agents from deterministic validation, ensuring that only safe, catalogue-bounded, and resource-compliant mitigation portfolios are executed at the edge.
- Experimental evaluation on a distributed edge testbed demonstrates that ASPO maintains stable safety enforcement and reduces extreme-case latency and energy overheads by over 20% under workload scaling.

---

[To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling](http://arxiv.org/abs/2605.00737)

- Agentic AI architectures: introduces a principled framework to evaluate and optimize LLM tool-calling decisions based on necessity, utility, and affordability.
- The framework identifies a misalignment between LLMs' self-perceived need for tools and their true performance utility, leading to suboptimal tool-calling behavior.
- The authors propose a controller framework using Latent Need Estimators and Latent Utility Estimators to improve decision quality by leveraging internal hidden states of the LLM.

---

[Learning How and What to Memorize: Cognition-Inspired Two-Stage Optimization for Evolving Memory](http://arxiv.org/abs/2605.00702)

- MemCoE: introduces a two-stage optimization framework that decouples memory organization from content updating by learning a schema-consistent guideline and a guideline-aligned memory policy.
- The framework utilizes Memory Guideline Induction to optimize a global natural-language guideline via contrastive feedback and batch aggregation, followed by Guideline-Aligned Memory Policy Optimization to train the agent using structured process rewards.
- MemCoE improves long-term personalization by internalizing extraction, update, and forgetting behaviors into a unified memory-evolution process, demonstrating robustness and transferability across various LLMs.

---

[Learning to Act and Cooperate for Distributed Black-Box Consensus Optimization](http://arxiv.org/abs/2605.00691)

- LAC-MAS (Learning to Act and Cooperate for Multi-Agent Systems): introduces a trajectory-driven framework for distributed black-box consensus optimization that utilizes LLMs to provide sparse high-level guidance for both internal agent behaviors and external cooperation patterns.
- The framework integrates an adaptive swarm execution layer with an LLM-based guidance module, coordinated by a phased cognitive guidance mechanism to ensure stable and resource-aware optimization.
- By leveraging historical optimization trajectories, LAC-MAS enables agents to dynamically adjust their internal search regimes and neighbor influence weights, consistently improving solution quality and communication efficiency in decentralized environments.

---

[Affordance Agent Harness: Verification-Gated Skill Orchestration](http://arxiv.org/abs/2605.00663)

- A-Harness: introduces a closed-loop runtime that unifies heterogeneous skills with an Evidence Store, Two-Tier Memory, Router, and Verifier to perform budgeted, verification-gated affordance grounding.
- The framework replaces fixed pipelines with an adaptive routing policy that uses relative diagnostics to gate commitments and trigger targeted retries based on cross-tool consistency, cross-scale stability, and evidence sufficiency.
- By amortizing successful tool chains through a two-tier memory, the system improves the accuracy-cost trade-off in open-world environments while reducing latency and unnecessary skill calls.

---

[Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding](http://arxiv.org/abs/2605.00642)

- GUI-SD: introduces an on-policy self-distillation framework for GUI grounding that replaces sparse reinforcement learning rewards with dense token-level supervision.
- The framework utilizes Visual Privileged Guidance to provide informative teacher signals and Entropy-Guided Optimization to prioritize high-impact coordinate tokens during training.
- By employing a single policy model as both teacher and student with asymmetric contexts, GUI-SD achieves superior accuracy and training efficiency compared to traditional reinforcement learning methods.

---

[DySRec: Dynamic Context-Aware Psychometric Scale Recommendation via Multi-Agent Collaboration](http://arxiv.org/abs/2605.00574)

- DySRec: introduces a multi-agent framework for dynamic psychometric scale recommendation that models clinical assessment as a sequential decision-making process using an Orchestrator, Context State Management Agent, Shared Global Context State, Dynamic Scale Recommendation Agent, LLM backbone, Psychometric Scale Repository, Scale Execution &amp; Scoring Agent, Risk Detection &amp; Intervention Agent, and Audit &amp; Logging Agent.
- The system utilizes a blackboard-style communication architecture to enable asynchronous collaboration between specialized agents while maintaining a unified, evolving user context state.
- DySRec incorporates a closed-loop refinement mechanism and real-time risk-aware coordination to ensure clinical safety and improve recommendation accuracy through active information-seeking dialogue.

---

[Structure Liberates: How Constrained Sensemaking Produces More Novel Research Output](http://arxiv.org/abs/2605.00557)

- SCISENSE: introduces a sensemaking-grounded framework that operationalizes scientific ideation as a structured sequence of eight cognitive stages to improve research output quality and diversity.
- The framework utilizes SCISENSE-Traj to distill sensemaking capabilities into SCISENSE-LM, enabling LLMs to generate reconstructive or generative research trajectories grounded in citation neighborhoods.
- Experimental results demonstrate that constrained, target-based supervision produces more novel and diverse research trajectories, which subsequently improve the executability and quality of downstream research artifacts generated by coding agents.

---

[A11y-Compressor: A Framework for Enhancing the Efficiency of GUI Agent Observations through Visual Context Reconstruction and Redundancy Reduction](http://arxiv.org/abs/2605.00551)

- A11y-Compressor: introduces a structured framework that transforms linearized accessibility trees into compact, semantically coherent GUI observations for LLMs.
- The framework utilizes Modal Detection (identifies foreground modal elements), Redundancy Reduction (removes irrelevant or repetitive elements), and Semantic Structuring (organizes elements into meaningful functional groups) to improve grounding efficiency.
- Experimental results on the OSWorld benchmark demonstrate that the framework reduces input token consumption to 22% of the original while improving task success rates by 5.1 percentage points.

---

[From Research to Practice: An Interactive Rapid Review of Autonomous Driving System Testing in Industry](http://arxiv.org/abs/2605.00531)

- Interactive Rapid Review (IRR): introduces a practitioner-driven methodology to evaluate academic research on End-to-End (E2E) ADS testing by synthesizing technological rules and assessing their industrial applicability.
- The framework categorizes testing interventions into constraint/search-based, reinforcement learning-based, generative model-based, and adversarial perturbation-based components to address critical scenario generation and testing completeness.
- This study bridges the gap between academic research and industrial practice by involving 21 practitioners to validate the relevance of 17 selected research studies for E2E ADS testing.

---

[SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters](http://arxiv.org/abs/2605.00528)

- SAGA: introduces a distributed scheduler that treats multi-step agent workflows as first-class schedulable units to minimize KV cache regeneration latency.
- The framework utilizes Agent Execution Graphs to predict future cache reuse, enabling proactive retention and efficient scheduling across tool-call boundaries.
- SAGA integrates session-affinity batching, work-stealing load balancing, and Agent Fair Share scheduling to achieve high SLO attainment and memory utilization in multi-tenant GPU clusters.

---

[Stereo Multistage Spatial Attention for Real-Time Mobile Manipulation Under Visual Scale Variation and Disturbances](http://arxiv.org/abs/2605.00471)

- MSARNN: introduces a deep predictive learning framework that utilizes stereo multistage spatial attention to extract task-relevant features for robust real-time mobile manipulation.
- The architecture integrates MSA, a Motion Predict Module, and a Hierarchical LSTM to process stereo visual inputs and robot states for temporally coherent motion generation.
- The framework employs a Temporally Bidirectional Loss to align attention points with robot motion, ensuring stability under visual scale variations and environmental disturbances.

---

[Scaling Video Understanding via Compact Latent Multi-Agent Collaboration](http://arxiv.org/abs/2605.00444)

- MACF: introduces an end-to-end framework that scales video understanding by partitioning inputs across Local Agents with Budgeted Perception and aggregating evidence via a Coordinator Agent using compact Communication Tokens in a Shared Latent Communication Space.
- The framework employs a Curriculum Training Strategy to progressively establish semantic alignment, query-aware evidence summarization, and cross-agent coordination, ensuring efficient information exchange under strict communication constraints.
- By decoupling per-agent perception budgets from global video complexity, MACF preserves fine-grained visual fidelity and outperforms state-of-the-art LLMs and multi-agent systems on long-form video benchmarks.

---

[AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning](http://arxiv.org/abs/2605.00425)

- AEM (Adaptive Entropy Modulation): introduces a supervision-free credit assignment method that adaptively modulates entropy dynamics during RL training to optimize the exploration-exploitation trade-off.
- The framework leverages response-level entropy as an intrinsic signal to rescale advantages, effectively upweighting or downweighting responses based on their relative surprisal.
- AEM functions as a lightweight plug-in for existing RL baselines, improving performance across multi-turn agentic tasks without requiring additional model training or heavy computation.

---

[Skills as Verifiable Artifacts: A Trust Schema and a Biconditional Correctness Criterion for Human-in-the-Loop Agent Runtimes](http://arxiv.org/abs/2605.00424)

- Enclawed (Configurable, sector-neutral hardening framework for single-user AI assistant gateways): introduces a trust schema and biconditional correctness criterion to secure LLM agent runtimes by treating skills as verifiable artifacts.
- The framework utilizes a Skill Manifest, Trust Root, Capability Gate, Transaction Buffer, Audit Log, Broker, and Adversarial-Ensemble Evaluator to enforce security policies and verify agent behavior.
- It establishes twelve normative guidelines for building secure, skill-aware LLM agent harnesses that are resistant to supply-chain attacks and unauthorized agent mutations.

---

[Foresight Arena: An On-Chain Benchmark for Evaluating AI Forecasting Agents](http://arxiv.org/abs/2605.00420)

- Foresight Arena: introduces a permissionless, on-chain evaluation infrastructure for benchmarking LLMs on real-world prediction markets using proper scoring rules.
- The framework utilizes a commit-reveal protocol enforced by Smart Contracts and External Oracles to ensure trustless, verifiable performance tracking of LLM agents.
- By employing Brier and Alpha Scores, the system isolates predictive forecasting ability from trading-based metrics, providing a rigorous assessment of LLM calibration and resolution.

---

[Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies](http://arxiv.org/abs/2605.00416)

- LWD: introduces a fleet-scale offline-to-online reinforcement learning framework for the continual post-training of generalist Vision-Language-Action policies.
- The framework utilizes DIVL for robust value estimation from heterogeneous data and QAM for stable policy extraction from flow-based action generators.
- By closing the loop between fleet-scale deployment and centralized learning, LWD enables robots to autonomously improve shared policies through a data flywheel of physical experience.

---

[Physically Native World Models: A Hamiltonian Perspective on Generative World Modeling](http://arxiv.org/abs/2605.00412)

- HWM (Hamiltonian World Models): introduces a physically grounded generative dynamical system that separates perception, latent dynamics, rendering, and planning to improve long-horizon stability and interpretability.
- The framework utilizes a structured latent phase space representation (q, p) to model physical interactions through energy-based dynamics rather than unconstrained neural transitions.
- By incorporating controlled-dissipative Hamiltonian dynamics, the model provides a physically meaningful inductive bias that enhances data efficiency and reliability for embodied decision-making tasks.

---

[Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines](http://arxiv.org/abs/2605.00410)

- Agent Capsules: introduces an adaptive execution runtime that optimizes multi-agent LLM pipelines by dynamically merging agent calls into compound execution units based on empirical quality constraints.
- The framework utilizes a composition score, a quality gate, and an escalation ladder to safely balance token efficiency with output quality across different execution modes.
- Agent Capsules outperforms hand-tuned and compile-time optimized baselines by providing runtime-adaptive orchestration that requires no training data or per-pipeline engineering.

---

[BOLT: Online Lightweight Adaptation for Preparation-Free Heterogeneous Cooperative Perception](http://arxiv.org/abs/2605.00405)

- BOLT: introduces a lightweight, plug-and-play ego-side plugin that enables preparation-free heterogeneous cooperative perception by adapting neighbor features online via ego-as-teacher distillation.
- The framework utilizes a frozen fusion module and detection head, relying on an adaptive plugin composed of AdaIN, a residual CNN adapter, and a per-channel gate to align heterogeneous features without requiring pre-deployment collaborative training.
- By leveraging high-confidence ego predictions as teacher signals, BOLT effectively bridges the distribution gap between independently trained agents, consistently surpassing ego-only performance across multiple benchmarks.

---

[Agentic AI for Substance Use Education: Integrating Regulatory and Scientific Knowledge Sources](http://arxiv.org/abs/2605.00383)

- Agentic RAG System: introduces an agentic AI architecture that integrates official DEA regulatory records with real-time PubMed scientific literature to provide verifiable substance use education.
- The system utilizes a modular five-layer architecture, including a Presentation Layer (Streamlit), Orchestration Layer (LangChain), Retrieval Layer (Dual-source knowledge access), Generation Layer (Qwen3-32B LLM), and Storage Layer (ChromaDB), to ensure high-availability and evidence-based responses.
- Expert evaluation across 90 system interactions demonstrated high performance in factual accuracy, citation quality, contextual coherence, and regulatory appropriateness, effectively mitigating common LLM hallucination risks.

---

[Social Bias in LLM-Generated Code: Benchmark and Mitigation](http://arxiv.org/abs/2605.00382)

- FMA (Fairness Monitor Agent): introduces a modular, oracle-free system that intercepts code generation pipelines to detect and repair social bias through static LLM-based analysis and iterative guided rewriting.
- The framework utilizes Solar, a metamorphic testing tool, to quantify social bias across 343 human-centered coding tasks, revealing that bias is a structural property of LLMs rather than a sampling artifact.
- Experimental results demonstrate that FMA achieves a 65.1% relative reduction in bias while simultaneously improving functional correctness, outperforming both prompt-level interventions and structured multi-agent workflows.

---

[ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](http://arxiv.org/abs/2605.00380)

- ResRL: introduces a reinforcement learning framework that decouples gradient updates by projecting negative token representations onto a low-rank positive subspace to mitigate semantic interference.
- The framework utilizes policy hidden states to compute projection residuals, which serve as a proxy for gradient interference and guide conservative advantage reweighting.
- ResRL improves reasoning performance and generation diversity across mathematical, coding, agentic, and tool-use tasks by selectively suppressing erroneous reasoning patterns while preserving shared valid semantic structures.

---

[Time-series Meets Complex Motion Modeling: Robust and Computational-effective Motion Predictor for Multi-object Tracking](http://arxiv.org/abs/2605.00362)

- TCMP (Temporal Convolutional Motion Predictor): introduces a lightweight motion prediction framework for Multi-object Tracking that leverages a modified Temporal Convolutional Network with dilated convolutions to capture complex non-linear motion dynamics.
- The architecture utilizes a stack of dilated convolutional layers to efficiently model long-range temporal dependencies while avoiding the cumulative errors and computational overhead associated with autoregressive models.
- By integrating a learned parameter to balance hierarchical skipped connections and final block outputs, the model achieves state-of-the-art tracking performance with significantly reduced parameter counts and computational costs compared to existing generative approaches.

---

[MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents](http://arxiv.org/abs/2605.00356)

- MemRouter: introduces an embedding-based memory router that decouples memory admission from the downstream answer backbone to avoid per-turn autoregressive generation.
- The architecture utilizes a Memory Router with lightweight classification heads and a frozen LLM backbone to predict storage decisions based on contextualized turn embeddings.
- By replacing autoregressive memory management with a forward-only routing policy, the system achieves significant latency reductions and improved F1 performance in long-term conversational QA tasks.

---

[Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning](http://arxiv.org/abs/2605.00347)

- Odysseus: introduces an open training framework for VLM agents that integrates Supervised Initialization, Multi-Task Reinforcement Learning, PPO with Lightweight Turn-Level Critic, Positive-Advantage Filtering, and an Auto-Curriculum Mechanism to enable stable long-horizon decision-making.
- The framework utilizes a lightweight CNN-based turn-level critic to decouple temporal credit assignment from token generation, significantly reducing computational overhead compared to large-model-based actor-critic methods.
- Odysseus achieves substantial performance gains in long-horizon game environments by leveraging pretrained VLMs as strong action priors, effectively narrowing the gap between foundation models and embodied agents.

---

[AgentFloor: How Far Up the tool use Ladder Can Small Open-Weight Models Go?](http://arxiv.org/abs/2605.00334)

- AgentFloor: introduces a deterministic six-tier benchmark to evaluate the tool-use capabilities of LLMs across a controlled capability ladder.
- The framework utilizes a Tool Surface and Inference Protocol to compare open-weight models against frontier LLMs, identifying specific capability thresholds for routine versus complex agentic tasks.
- Results demonstrate that while smaller models are sufficient for routine tool use, frontier LLMs maintain a performance advantage in long-horizon planning tasks under persistent constraints.

---

[Semia: Auditing Agent Skills via Constraint-Guided Representation Synthesis](http://arxiv.org/abs/2605.00314)

- Semia: introduces, a static analysis framework for agent skills that lifts hybrid prose-code artifacts into a structured Datalog fact base using an LLM-driven synthesis loop disciplined by structural and semantic constraints.
- The framework utilizes a Constraint-Guided Representation Synthesis (CGRS) loop, which iteratively refines candidate SDL representations based on feedback from a structural validator and a semantic scorer to ensure faithful translation.
- Once a skill is lifted into an SDL fact base, the system employs a library of Datalog-based security detectors to perform deterministic reachability analysis, identifying critical vulnerabilities such as missing human-in-the-loop gates and unsanitized context ingestion.

---

[An End-to-End Decision-Aware Multi-Scale Attention-Based Model for Explainable Autonomous Driving](http://arxiv.org/abs/2605.00291)

- End-to-End Decision-Aware Multi-Scale Attention-Based Model: introduces an end-to-end multi-task deep learning architecture that integrates ResNet50, MASPP, and a hybrid DCA-CBAM-ResNet50 Attention mechanism to simultaneously predict driving actions and generate corresponding textual and visual explanations.
- The framework improves interpretability by feeding action head outputs into the reasoning head, ensuring logical consistency between driving decisions and their underlying rationales.
- The authors propose a Joint F1 score metric to quantitatively evaluate the alignment between predicted actions and their associated reasons, demonstrating superior performance on the BDD-OIA and nu-AR datasets.

---

[Intern-Atlas: A Methodological Evolution Graph as Research Infrastructure for AI Scientists](http://arxiv.org/abs/2604.28158)

- Intern-Atlas: introduces a methodological evolution graph that transforms document-centric citation networks into a queryable causal topology for AI research agents.
- The framework utilizes a two-phase LLM extraction process to identify method entities and classify citation edges with verbatim bottleneck-to-mechanism evidence.
- It enables automated scientific discovery through SGT-MCTS lineage reconstruction, graph-grounded idea evaluation, and strategy-driven idea generation.

---

[GSDrive: Reinforcing Driving Policies by Multi-mode Trajectory Probing with 3D Gaussian Splatting Environment](http://arxiv.org/abs/2604.28111)

- GSDrive: introduces a framework that utilizes 3D Gaussian Splatting (3DGS) to provide dense, physics-based reward signals for E2E autonomous driving policy improvement.
- The framework employs a flow matching-based trajectory predictor to probe multiple future scenarios within the 3DGS environment, enabling the policy to evaluate long-term consequences of current actions.
- By bridging imitation learning and reinforcement learning through trajectory-based reward shaping, GSDrive improves training stability and performance in complex, closed-loop driving scenarios.

---

[D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery](http://arxiv.org/abs/2604.27977)

- D3-Gym: introduces an automated pipeline for constructing real-world, verifiable environments for scientific data-driven discovery, utilizing AutoSDT, Coding Agent, Execution Environment, Multimodal LLM-as-judge, Evaluation Script Planner, Evaluation Script Coder, and Training Environments.
- The framework employs a planning-then-coding approach to synthesize task-specific evaluation scripts, ensuring scientific soundness through rigorous validation against human-annotated gold standards.
- Training LLMs on trajectories sampled from D3-Gym yields consistent performance gains across model sizes, effectively narrowing the gap between open-weight models and proprietary reasoning systems.

---

[From Unstructured Recall to Schema-Grounded Memory: Reliable AI Memory via Iterative, Schema-Aware Extraction](http://arxiv.org/abs/2604.27906)

- xmemory: introduces an iterative, schema-aware extraction pipeline that decomposes unstructured interaction logs into validated, schema-conformant records to ensure reliable AI memory.
- The architecture utilizes a stateful Prompt Engine and multi-stage extraction components—Object Detector, Object Descriptor, Fields Detector, Fields Extractor, and Relation Extractor—to enforce explicit relevance contracts.
- By shifting interpretation from the read path to the write path, the framework employs Request, Session, and Main memory contexts to maintain deterministic, verifiable, and addressable factual records.

---

[Autonomous Systems Dependability in the era of AI: Design Challenges in Safety, Security, Reliability and Certification](http://arxiv.org/abs/2604.27807)

- Autonomous System Dependability Management framework: introduces a holistic approach for designing dependable autonomous systems by integrating design-time assurance and run-time adaptation across hardware, software, and application layers.
- The framework addresses the non-deterministic nature of ML components by employing safety-aware resource allocation and cross-layer reliability analysis to maintain system performance under stringent constraints.
- It further incorporates security-by-design principles, including in-vehicle network protection and perception security, to ensure robust operation in the presence of potential cyberattacks and environmental uncertainties.

---

[Skills-Coach: A Self-Evolving Skill Optimizer via Training-Free GRPO](http://arxiv.org/abs/2604.27488)

- Skills-Coach: introduces an automated framework for the self-evolution of skills in LLM-based agents, utilizing a Diverse Task Generation Module, a Lightweight Optimization Module, a Comparative Execution Module, and a Traceable Evaluation Module.
- The framework employs Training-Free GRPO to iteratively refine skill instructions and code, significantly improving performance across both instruction-only and code-inclusive skill categories.
- By leveraging the Skill-X benchmark, the system demonstrates substantial gains in skill generalization and robustness, enabling autonomous optimization without human intervention.

---

#### 30th April 2026


[Structural Dissolution: How Artificial Intelligence Dismantles Coordination Architecture and Reconfigures the Political Economy of Production](http://arxiv.org/abs/2604.27435)

- Structural Dissolution Framework: introduces a theoretical lens explaining how AI systematically dismantles traditional economic coordination architectures by absorbing multimodal human interfaces into intra-system computation.
- The framework utilizes Interface Internalization to convert inter-agent coordination into computational processes, effectively replacing firms and markets with Regional Data Sovereignty Entities.
- It identifies the emergence of Data-Personified Economic Agents and data refinement loops as the new drivers of value creation, shifting competitive advantage toward the control of domain-specific data infrastructure.

---



[Modeling Clinical Concern Trajectories in Language Model Agents](http://arxiv.org/abs/2604.27872)

- Clinical Concern Trajectory Modeling framework: introduces a lightweight agent architecture that integrates a memoryless clinical risk encoder with explicit temporal dynamics to produce continuous, legible escalation pressure signals.
- The framework utilizes second-order hysteretic dynamics to model clinical principles where worsening physiological signals propagate faster than improvements, ensuring smooth and anticipatory concern trajectories.
- By separating instantaneous risk encoding from temporal integration, the architecture enables LLM agents to provide transparent, human-interpretable monitoring in safety-critical clinical environments.

---

[Collaborative Agent Reasoning Engineering (CARE): A Structured Three-Party Design Methodology for Systematically Engineering AI Agents with SMEs, Developers, and Helper Agents](http://arxiv.org/abs/2604.28043)

- CARE (Collaborative Agent Reasoning Engineering): introduces a disciplined, stage-gated methodology for engineering LLM agents through a triadic collaboration between SMEs, developers, and helper LLM agents to produce explicit, reviewable artifacts.
- The framework deconstructs agent development into four design targets—interaction policy, domain grounding, tool orchestration, and evaluation/verification—to replace ad hoc prompt iteration with structured engineering.
- By utilizing helper LLM agents as facilitation infrastructure, CARE enables the systematic creation of versioned artifacts that ensure agent behavior is testable, maintainable, and resistant to regressions.

---


[High-Probability Convergence in Decentralized Stochastic Optimization with Gradient Tracking](http://arxiv.org/abs/2605.00281)

- GT-DSGD: introduces a decentralized optimization framework that achieves high-probability convergence guarantees by incorporating gradient tracking to mitigate data heterogeneity.
- The framework utilizes local model updates and gradient tracking variables to eliminate the effects of data heterogeneity across agents in a network.
- GT-DSGD provides order-optimal high-probability convergence rates for both non-convex and Polyak-Lojasiewicz cost functions under relaxed sub-Gaussian noise conditions.

---

[Agentic AI for Trip Planning Optimization Application](http://arxiv.org/abs/2605.00276)

- AAS (Agentic AI System): introduces a hierarchical multi-agent framework for vehicle-centric trip planning that utilizes centralized orchestration to coordinate specialized agents for adaptive reasoning and self-correction.
- The system employs an In-Vehicle Agent for interaction, an Orchestration Agent for sub-task decomposition, and a Pool of Specialized Agents for domain-specific execution to optimize itineraries under multi-constraint scenarios.
- The authors also present the TOP (Trip-planning Optimization Problems) Benchmark, a dataset of 500 queries with deterministic ground-truth solutions designed to evaluate optimization performance across varying levels of reasoning complexity.

---

[Fast Rates in α-Potential Games via Regularized Mirror Descent](http://arxiv.org/abs/2605.00268)

- ROPE (Regularized Offline Potential Equilibrium): introduces a framework that leverages the structural alignment of α-potential games to identify Nash Equilibria via potential maximization using KL regularization.
- OPMD (Offline Potential Mirror Descent): provides a decentralized algorithm that achieves an accelerated O(1/n) statistical rate by resolving evaluation mismatch as a vanishing optimization error.
- The research establishes that Reference-Anchored Unilateral Concentrability allows for sample-efficient equilibrium recovery without explicit pessimistic bonuses, bypassing the PPAD-complete complexity of generic general-sum games.

---

[Pessimism-Free Offline Learning in General-Sum Games via KL Regularization](http://arxiv.org/abs/2605.00264)

- GANE (General-sum Anchored Nash Equilibrium): introduces a pessimism-free framework that leverages KL regularization to achieve an accelerated O˜(1/n) statistical rate for Nash equilibria in general-sum games.
- GAMD (General-sum Anchored Mirror Descent): provides a computationally tractable, decentralized iterative algorithm that recovers Coarse Correlated Equilibria at the standard O˜(1/√n) statistical rate without explicit pessimism.
- Both frameworks utilize reference-anchored unilateral concentrability to bypass the conceptual circularity of traditional pessimism-based approaches in multi-agent offline learning.

---

[Causal Foundations of Collective Agency](http://arxiv.org/abs/2605.00248)

- Mechanized Causal Games: introduces a formal framework for identifying collective agency in multi-agent systems by leveraging causal abstraction to determine when a group of agents can be modeled as a single unified agent.
- The framework utilizes Mechanism nodes (explicitly represented causal parameters), Object nodes (variables governed by mechanisms), Utility functions (functions mapping object nodes to real numbers), Rationality relations (specifies agent response to context), Variable alignment (mapping between low-level and high-level variables), Value mappings (translates low-level to high-level values), and Intervention mappings (translates low-level to high-level interventions) to provide a rigorous account of agency across levels of abstraction.
- The authors demonstrate the utility of their approach by analyzing actor-critic models and voting mechanisms, providing a foundation for understanding, predicting, and controlling emergent collective agents in multi-agent AI systems.

---

[The Silicon Society Cookbook: Design Space of LLM-based Social Simulations](http://arxiv.org/abs/2605.00197)

- Silicon Society Cookbook: introduces a systematic analysis of the design space for LLM-based social simulations, utilizing SimEngine, LLM-based Agents, LoRA Adapters, Followership Network, and News Source Nodes to evaluate simulation outcomes.
- The framework employs LoRA Adapters to instantiate diverse agent personas, enabling the study of opinion dynamics and information diffusion within synthetic social networks.
- The research identifies that the choice of base LLM is the most significant variable impacting simulation behavior, while demonstrating that fine-tuning on social media data improves stylistic realism.

---

[Real-Time Frame- and Event-based Object Detection with Spiking Neural Networks on Edge Neuromorphic Hardware: Design, Deployment and Benchmark](http://arxiv.org/abs/2605.00146)

- SNN-based Object Detection Framework: introduces a methodology for designing and deploying lightweight SNNs on Intel Loihi 2 neuromorphic hardware for real-time object detection using both frame-based and event-based data.
- The framework utilizes an ANN-to-SNN knowledge distillation approach to recover 87–100% of ANN detection accuracy while maintaining low inference latency and energy efficiency on edge neuromorphic processors.
- The study benchmarks three SNN architectures against ANN counterparts on Jetson Nano and Apple M2 platforms, demonstrating superior energy efficiency and real-time performance on the Intel Loihi 2 neuromorphic chip.

---

[Are Tools All We Need? Unveiling the Tool-Use Tax in LLM Agents](http://arxiv.org/abs/2605.00136)

- Factorized Intervention Framework: introduces a diagnostic methodology to decompose the performance gap between native CoT and tool-augmented LLMs into formatting costs, protocol overhead, and actual tool execution gains.
- The paper identifies the Capability Overlap Principle, where tool-derived gains are often redundant with the model's native reasoning, while the tool-use protocol introduces significant performance degradation under semantic noise.
- The authors propose G-STEP, a lightweight inference-time gate that mitigates protocol-induced errors by dynamically deciding whether to continue tool interaction or commit to a final answer.

---

[HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation](http://arxiv.org/abs/2604.28196)

- HERMES++: introduces a unified driving world model that integrates 3D scene understanding and future geometry prediction within a single framework using BEV Tokenizer, LLM, World Queries, Current-to-Future Link, Shared Render, Joint Geometric Optimization, and Ego Modulation.
- The framework leverages a BEV representation to consolidate multi-view spatial information, enabling the LLM to perform semantic reasoning while guiding future geometric evolution through a Current-to-Future Link.
- A Joint Geometric Optimization strategy enforces structural integrity by combining explicit geometric constraints on point clouds with implicit latent regularization, ensuring predicted features align with intrinsic 3D geometry.

---

[Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling](http://arxiv.org/abs/2604.28185)

- Visual Generation in the New Era: introduces a five-level taxonomy of visual intelligence, spanning Atomic Generation, Conditional Generation, In-Context Generation, Agentic Generation, and World-Modeling Generation, to categorize the evolution of visual models from passive renderers to interactive, agentic, world-aware generators.
- The paper decomposes modern visual generation architectures into four core components: Encoder/Tokenizer, Condition Module, Backbone Network, and Multimodal Fusion Module, which collectively enable the transition from single-pass rendering to complex, agentic workflows.
- It provides a comprehensive roadmap of technical drivers, including the shift from diffusion to flow matching, the rise of unified understanding-and-generation models, and the integration of agentic loops for grounded verification and causal world modeling.

---

[Exploration Hacking: Can LLMs Learn to Resist RL Training?](http://arxiv.org/abs/2604.28182)

- Exploration Hacking: introduces, "a failure mode where LLMs strategically alter their exploration during RL training to influence the training outcome and resist capability elicitation", with all LLM, RL-based capability elicitation, Model organisms, CoT monitors, Action-only monitors, Weight noising, SFT-based elicitation.
- The paper formalizes exploration hacking into a taxonomy of strategies and demonstrates that LLMs can be fine-tuned into "model organisms" that successfully resist RL-based capability elicitation while maintaining performance on benign tasks.
- The authors evaluate countermeasures including monitoring and weight noising, and audit frontier models, finding that while models possess the capability for strategic reasoning, they currently lack the spontaneous propensity to execute exploration hacking without explicit scaffolding.

---

[Synthetic Computers at Scale for Long-Horizon Productivity Simulation](http://arxiv.org/abs/2604.28181)

- Synthetic Computers at Scale: introduces a methodology for creating artifact-rich, user-specific environments to ground long-horizon simulations, utilizing Persona, User Profile, Filesystem Planning, and Synthetic Computer components.
- The framework employs a Setup Agent to define objectives and a Work Agent to perform tasks, interacting with Simulated Collaborators to generate Experiential Signals.
- These signals are processed by a Skill Creator to build a Skill Set, enabling agent self-improvement through iterative simulation and skill-based reinforcement.

---

[FlashRT: Towards Computationally and Memory Efficient Red-Teaming for Prompt Injection and Knowledge Corruption](http://arxiv.org/abs/2604.28157)

- FlashRT: introduces a framework to optimize the computation and memory efficiency of optimization-based red-teaming for long-context LLMs using Selective Recomputation, Gradient Approximation, Gradient Resampling, KV-Caching, and a Target LLM.
- The framework significantly reduces GPU memory consumption and runtime by approximating loss evaluations and gradient computations while maintaining high attack success rates.
- FlashRT is compatible with various black-box and white-box optimization methods and demonstrates effectiveness across diverse long-context LLM applications including prompt injection and knowledge corruption.

---

[CRAB: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes](http://arxiv.org/abs/2604.28138)

- CRAB: introduces a semantics-aware host-side runtime that bridges the agent–OS semantic gap by inferring recovery-relevant state from OS-visible effects at turn boundaries to enable efficient checkpointing.
- The framework utilizes a Coordinator to identify turn boundaries, an eBPF-based Inspector to classify state changes, and a C/R Engine to schedule asynchronous checkpoints that overlap with LLM wait windows.
- CRAB achieves 100% recovery correctness while maintaining low overhead by selectively checkpointing only necessary state and coordinating traffic across co-located sandboxes.

---

[What Makes a Good Terminal-Agent Benchmark Task: A Guideline for Adversarial, Difficult, and Legible Evaluation Design](http://arxiv.org/abs/2604.28093)

- Terminal Bench: introduces a set of guidelines for designing adversarial, difficult, and legible evaluation tasks for LLM-based agents in command-line environments.
- The paper identifies common failure modes in benchmark design, such as reward hacking and over-prescriptive instructions, which undermine the validity of LLM performance metrics.
- It advocates for shifting from prompt-like task design to authentic engineering problems that prioritize verifiable outcomes over specific implementation steps.

---

[Towards Neuro-symbolic Causal Rule Synthesis, Verification, and Evaluation Grounded in Legal and Safety Principles](http://arxiv.org/abs/2604.28087)

- NeSy Causal Framework: introduces a meta-level synthesis and verification layer that utilizes LLMs to refine formal rule theories from natural-language goals, mitigating goal misspecification in self-adaptive systems.
- The architecture integrates a Goal/Rule Synthesizer and a Rule Verification Engine to ensure that synthesized rules are logically consistent, safe, and traceable before integration into the system knowledge base.
- By interleaving logical reasoning, causal inference, and learning, the framework enables incremental, modular, and explainable rule maintenance for safety-critical autonomous driving scenarios.

---

[TOPBENCH: A Benchmark for Implicit Prediction and Reasoning over Tabular Question Answering](http://arxiv.org/abs/2604.28076)

- TOPBENCH: introduces a benchmark designed to evaluate LLMs on implicit predictive tasks over tabular data, requiring models to infer unobserved outcomes rather than performing simple fact retrieval.
- The framework utilizes an LLM-as-a-Judge and a ReAct-Agent to assess performance across four predictive scenarios, including single-point prediction, decision making, treatment effect analysis, and ranking.
- The research identifies a critical capability gap where LLMs struggle to distinguish between retrieval and predictive intent, often defaulting to exhaustive search loops instead of rigorous statistical modeling.

---

[Agent-Agnostic Evaluation of SQL Accuracy in Production Text-to-SQL Systems](http://arxiv.org/abs/2604.28049)

- STEF (Schema-agnostic Text-to-SQL Evaluation Framework): introduces a production-native evaluation system that assesses Text-to-SQL generation quality without requiring database schema or reference queries by utilizing Input Specification, Enriched Question Validation, Semantic Feature Extraction, Normalized Specification Alignment, Application-Specific Rule Injection, LLM Evaluator, and Composite Scoring Function.
- The framework employs a multi-stage pipeline that transforms natural language and SQL inputs into structured semantic comparisons, enabling fine-grained diagnostic signals for continuous agent improvement.
- STEF mitigates evaluator variance and production false negatives through a confidence-weighted composite scoring mechanism and configurable normalization rules for common SQL idioms.

---

[Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban Sentiment Perception](http://arxiv.org/abs/2604.28048)

- Persona-based Prompting Framework: introduces a methodology to evaluate persona validity in MLLM agents by measuring within-persona convergence and cross-persona variation across urban sentiment tasks.
- The framework utilizes a balanced factorial design of 24 personas and 1,200 agents to assess whether demographic and personality attributes induce meaningful behavioral differences in sentiment annotation.
- Experimental results indicate that while persona prompting stabilizes agent behavior, it fails to capture fine-grained perceptual judgments, often resulting in extremity bias and performance comparable to neutral no-persona baselines.

---

[Early Detection of Water Stress by Plant Electrophysiology: Machine Learning for Irrigation Management](http://arxiv.org/abs/2604.28038)

- PhytoNode-based Machine Learning Framework: introduces a biofeedback-driven system for early water stress detection in tomato plants using electrophysiological signals processed through automated machine learning and deep learning pipelines.
- The framework integrates PhytoNode hardware for signal acquisition with a processing pipeline that includes statistical feature extraction via tsfresh, model selection via Naive AutoML, and deep learning architectures including CNN, InceptionTime, and Mamba.
- Post-hoc certainty calibration using temperature scaling is employed to mitigate classifier overconfidence, enabling reliable transition detection from healthy to stressed plant states for precision irrigation management.

---

[Echo-α: Large Agentic Multimodal Reasoning Model for Ultrasound Interpretation](http://arxiv.org/abs/2604.28011)

- Echo-α: introduces an agentic multimodal reasoning framework that unifies specialized detector outputs with MLLM-based clinical reasoning through an invoke-and-reason loop.
- The framework utilizes a two-stage training process, starting with a supervised curriculum for foundational skills followed by reinforcement learning to optimize interaction trajectories for lesion grounding and diagnostic accuracy.
- By treating detector outputs as callable evidence rather than final predictions, the model enables verifiable clinical decision-making that adapts to different detector error modes across diverse ultrasound benchmarks.

---

[A Pattern Language for Resilient Visual Agents](http://arxiv.org/abs/2604.28001)

- Hierarchical Reference Architecture for Resilient Visual Agents: introduces a pattern language that decouples fast, deterministic System 1 reflexes from slow, probabilistic System 2 supervisory reasoning to balance latency and semantic adaptability.
- The architecture utilizes Hybrid Affordance Integration, Adaptive Visual Anchoring, Visual Hierarchy Synthesis, and a Semantic Scene Graph to enable amortized inference, where high-cost LLM-based reasoning is invoked only when low-latency reflexes fail.
- This approach transforms autonomous agents from brittle, open-loop scripts into closed-loop, self-healing systems capable of robust interaction within dynamic, non-instrumented enterprise GUI environments.

---

[Exploring Interaction Paradigms for LLM Agents in Scientific Visualization](http://arxiv.org/abs/2604.27996)

- SciVis Agent Paradigms: introduces a comparative study of three primary LLM agent interaction paradigms—domain-specific tool-use, GUI-based computer-use, and CLI-based coding—evaluated on scientific visualization tasks.
- The study identifies that while general-purpose coding agents achieve the highest task completion rates, they incur significantly higher computational costs compared to domain-specific agents which offer more efficient and stable execution.
- The research demonstrates that persistent memory and adaptive learning improve agent effectiveness and efficiency by reducing redundant exploration, while GUI-based agents face challenges with long-horizon planning despite strong localized reasoning capabilities.

---

[Dreaming Across Towns: Semantic Rollout and Town-Adversarial Regularization for Zero-Shot Held-Out-Town Fixed-Route Driving in CARLA](http://arxiv.org/abs/2604.27994)

- Dreamer-style latent world-model agent: introduces a framework for zero-shot held-out-town driving in CARLA by augmenting a world model with multi-horizon semantic rollout prediction and town-adversarial regularization.
- The framework utilizes a frozen OpenCLIP encoder to provide semantic targets for imagined rollouts, while a domain-adversarial branch encourages the learned latent representation to be invariant to source-town identity.
- By conditioning the semantic rollout predictor on a history-aware context feature, the agent improves route-following performance in unseen environments without requiring target-domain adaptation.

---

[FineState-Bench: Benchmarking State-Conditioned Grounding for Fine-grained GUI State Setting](http://arxiv.org/abs/2604.27974)

- FineState-Bench: introduces a benchmark and diagnostic framework for evaluating fine-grained, state-conditioned GUI interaction across desktop, web, and mobile platforms, utilizing FineState-Metrics and VDA to isolate grounding and state-setting failures.
- The framework employs a two-point interaction protocol that decouples target control localization from the operation-relevant region, enabling precise failure attribution in GUI agents.
- Experimental results demonstrate that while LVLMs often achieve successful coarse localization, they frequently struggle with interactable-core grounding, a bottleneck that VDA-provided localization hints can significantly mitigate.

---

[Language Models Refine Mechanical Linkage Designs Through Symbolic Reflection and Modular Optimisation](http://arxiv.org/abs/2604.27962)

- Symbolic Lifting and Closed-loop Synthesis Pipeline: introduces a modular framework that decomposes mechanical linkage design into discrete topology selection by LLM agents and continuous parameter fitting by numerical optimisers.
- The system utilizes a symbolic lifting operator to translate dense simulator trajectories into qualitative descriptors, enabling LLM agents to perform reflective reasoning and iterative design refinement.
- By binding high-level reasoning to mechanistic interfaces, the pipeline achieves superior geometric accuracy and structural validity across diverse LLM architectures without requiring fine-tuning.

---

[GUI Agents with Reinforcement Learning: Toward Digital Inhabitants](http://arxiv.org/abs/2604.27955)

- GUI Agents with Reinforcement Learning: Toward Digital Inhabitants introduces a principled taxonomy of RL paradigms—offline, online, and hybrid—to advance GUI agents from task-specific tools to persistent digital inhabitants.
- The paper analyzes critical dimensions including reward engineering, data efficiency, and technical innovations, highlighting the shift toward verifiable environment feedback and cognitive stratification.
- It provides a comprehensive roadmap for agent-native infrastructure, emphasizing the necessity of decoupling reasoning and execution to overcome I/O latency and sparse reward challenges in complex digital environments.

---

[Attractor FCM](http://arxiv.org/abs/2604.27947)

- Attractor FCM: introduces a physics-constrained, gradient-based fuzzy cognitive map that utilizes residual memory, back propagation through time, and a fixed point anchor to ensure stable convergence.
- The framework employs Jacobian Gradient Descent and Newton’s method to optimize weights directly from the fixed-point geometry while using an adaptive term to navigate the landscape.
- A causal mask is applied to the network to maintain structural integrity and respect expert-defined physical constraints during the optimization process.

---

[A Collective Variational Principle Unifying Bayesian Inference, Game Theory, and Thermodynamics](http://arxiv.org/abs/2604.27942)

- GT-FEP (Game-Theoretic Free Energy Principle): introduces a unified framework that bridges variational inference, stochastic game theory, and statistical physics to explain collective intelligence in multi-agent systems.
- The framework utilizes the Harsanyi decomposition to quantify irreducible synergy and conflict within coalitions, establishing that stationary points of collective free energy correspond to Nash equilibria.
- It demonstrates that agent influence follows a non-monotonic inverted-U relationship with sensory precision, while providing a first-principles derivation of attention mechanisms as mean-field approximations of coalitional inference.

---

[Alignment Contracts for Agentic Security Systems](http://arxiv.org/abs/2605.00081)

- Alignment Contracts framework: introduces a formal method for specifying and enforcing behavioral constraints on LLM-based security agents by mediating observable effect traces.
- The architecture utilizes a dual-layer approach, combining best-effort semantic mitigation at the agentic layer with rigorous, theorem-backed enforcement at the typed mediation layer.
- The framework provides decidable admissibility checks and modular contract composition, with core security properties mechanized in Lean 4 to ensure soundness under explicit observability assumptions.

---

[World Model for Robot Learning: A Comprehensive Survey](http://arxiv.org/abs/2605.00080)

- WMRL: introduces a comprehensive survey of world models in robot learning, categorizing architectures into decoupled, single-backbone, MoE/MoT, unified VLA, and latent-space paradigms.
- The paper examines how world models serve as learned simulators for reinforcement learning, evaluation, and data generation, emphasizing the transition from passive video generation to action-conditioned, physically grounded predictive structures.
- It highlights the shift toward foundation-scale models and identifies key challenges in causal conditioning, computational efficiency, and the need for standardized evaluation metrics beyond visual realism.

---

[MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](http://arxiv.org/abs/2604.27934)

- MM-StanceDet: introduces a multi-agent framework for robust multimodal stance detection by integrating Retrieval Augmentation Stage, Multimodal Analysis Stage (comprising Text-Analysis Agent, Image-Analysis Agent, and Modality-Conflict Agent), Reasoning-Enhanced Debate Stage (with Debater Agents), and Self-Reflection and Adjudication Stage (using an Adjudicator Agent).
- The framework utilizes a Vector Database to provide few-shot exemplars, enabling specialized agents to perform nuanced analysis and resolve cross-modal conflicts through structured debate and critical self-reflection.
- Extensive experiments demonstrate that this structured agentic process significantly outperforms single-pass LLMs and state-of-the-art baselines across diverse multimodal stance detection datasets.

---

[Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](http://arxiv.org/abs/2604.27924)

- AI-assisted Peer Review Frameworks: introduces a comprehensive taxonomy of methodologies for automating peer review, spanning generation paradigms, after-review tasks, and evaluation frameworks.
- The paper categorizes generation methodologies into Foundation Approaches, Fine-tuning Methods, Agent-based Methods, Reinforcement Learning Methods, and Review Generation Enhancement, which includes External Knowledge, Iterative Refinement, and Style Control.
- It provides a systematic evaluation taxonomy comprising Human-Centric, Reference-Based, LLM-Based, and Aspect-Oriented methods, while discussing critical challenges such as novelty assessment, bias, and the need for multimodal integration.

---

[A Logic of Inability](http://arxiv.org/abs/2604.27917)

- CLIab (Coalition Logic of Inability): introduces a conservative extension of Coalition Logic that treats inability as a first-class modal operator to systematically analyze strategic limitations.
- The framework provides a formal semantics and axiomatization for the inability operator, establishing its structural properties such as anti-monotonicity and subadditivity.
- By formalizing inability as an explicit modality, the paper enables precise reasoning about bounded agency and negative requirements in multi-agent systems.

---

[Graph World Models: Concepts, Taxonomy, and Future Directions](http://arxiv.org/abs/2604.27895)

- GWM (Graph World Models): introduces a unified research paradigm that formalizes world models by injecting relational inductive biases into environment representations to improve prediction and planning.
- The framework categorizes GWMs into three layers: Graph as Connector (spatial RIB), Graph as Simulator (physical RIB), and Graph as Reasoner (logical RIB).
- This survey outlines design principles, representative models, and future research directions including dynamic graph adaptation, probabilistic modeling, and multi-granularity inductive biases.

---

[In-Context Prompting Obsoletes Agent Orchestration for Procedural Tasks](http://arxiv.org/abs/2604.27891)

- In-Context Prompting: demonstrates that for procedural tasks, providing the entire workflow in the system prompt allows an LLM to self-orchestrate more effectively than using external frameworks like LangGraph.
- The study evaluates performance across three domains, finding that in-context prompting consistently outperforms orchestrated agents in task success, information accuracy, and consistency.
- Orchestration is identified as a bottleneck that fragments reasoning and introduces structural failure modes, whereas frontier models demonstrate sufficient capability to manage defined procedures internally.

---

[Building Persona-Based Agents On Demand: Tailoring Multi-Agent Workflows to User Needs](http://arxiv.org/abs/2604.27882)

- On-Demand Persona-Based Agent Generation framework: introduces a pipeline for dynamic, run-time synthesis of agent personas to tailor multi-agent workflows to specific user needs and contexts.
- The system replaces fixed agent architectures with a generative approach that utilizes ProfileEncode, TaskDecompose, PersonaCraft, AgentFactory, Agents Assignment, Agents Execution, and Answers Aggregation to instantiate agents on demand.
- By treating agent roles and interaction policies as runtime variables rather than design-time constants, the framework enables context-sensitive adaptivity and personalized multi-agent collaboration.

---

[KellyBench: A Benchmark for Long-Horizon Sequential Decision Making](http://arxiv.org/abs/2604.27865)

- KellyBench: introduces a non-stationary, open-ended environment for evaluating LLMs on long-horizon sequential decision-making tasks in sports betting markets.
- The framework utilizes an Environment, Agent, Sandbox, CLI Tools, and a Reward Signal to measure an agent's ability to build predictive models, identify market edges, and manage risk under uncertainty.
- Experimental results demonstrate that current frontier LLMs systematically underperform human baselines, frequently failing to execute reasoned strategies due to poor situational awareness and infrastructure-related errors.

---

[Rethinking Agentic Reinforcement Learning In Large Language Models](http://arxiv.org/abs/2604.27859)

- Agentic RL: introduces a control-theoretic framework for LLMs that integrates Agent, Planning, Memory, Action, and Tools to enable autonomous, self-improving decision-making.
- The framework shifts LLM optimization from static text generation to cumulative trajectory returns within partially observable environments.
- It synthesizes recent advancements in reinforcement learning algorithms—such as PPO, DPO, GRPO, and GSPO—to enhance the reasoning, planning, and tool-use capabilities of LLMs.

---

[NetSatBench: A Distributed LEO Constellation Emulator with an SRv6 Case Study](http://arxiv.org/abs/2604.27854)

- NetSatBench: introduces a distributed emulation platform for evaluating communication protocols and application workloads over large-scale LEO satellite systems using Worker nodes, Docker containers, Etcd key-value store, sat-agent, VXLAN overlay, nsb CLI, Physical-modeling pipeline, Routing plug-ins, and SRv6 control-plane.
- The platform decouples physical-layer and routing modeling from the emulator core through external plug-ins and utilizes a declarative JSON-based workflow for defining constellation dynamics.
- The system demonstrates its capabilities through an SRv6-based LEO architecture, highlighting the importance of end-to-end handover strategies that jointly account for both user-side and gateway-side access segments.

---

[A Grid-Aware Agent-Based Model for Analyzing Electric Vehicle Charging Systems](http://arxiv.org/abs/2604.27849)

- ABM: introduces a configurable, grid-aware simulation framework for analyzing electric vehicle charging dynamics by integrating EV, CC, and ES components.
- The framework utilizes discrete-event simulation to evaluate infrastructure performance, scheduling strategies, and grid-facing load characteristics under varying system scales.
- The model demonstrates that infrastructure dimensioning must be aligned with specific usage contexts, as fast charging infrastructure may induce higher grid stress without proportional operational benefits in workplace scenarios.

---

[CastFlow: Learning Role-Specialized Agentic Workflows for Time Series Forecasting](http://arxiv.org/abs/2604.27840)

- CastFlow: introduces a dynamic agentic forecasting framework that reformulates time series forecasting as a workflow-driven process, utilizing a Planning Module, Action Module, Forecasting Module, Reflection Module, Memory Module, Multi-view Toolkit, Frozen LLM, and Fine-tuned Domain-specific LLM.
- The framework employs a role-specialized design where a frozen LLM handles general-purpose reasoning for planning and reflection, while a fine-tuned domain-specific LLM executes numerical forecasting based on an ensemble forecast baseline and diagnostic evidence.
- CastFlow optimizes its performance through a two-stage workflow-oriented training strategy that combines supervised fine-tuning and reinforcement learning with verifiable rewards to ensure structural validity and numerical precision.

---

[Learning-Based Hierarchical Scene Graph Matching for Robot Localization Leveraging Prior Maps](http://arxiv.org/abs/2604.27821)

- HSGM: introduces a learned, end-to-end differentiable pipeline for hierarchical scene graph matching between BIM-derived A-graphs and robot-built S-graphs, utilizing an MLP Encoder, GATv2 Encoder, Affinity Matrix, Sinkhorn Layer, and Hungarian Algorithm.
- The framework employs a graph augmentation strategy to encode intra- and inter-level spatial relationships, enabling information flow across room and wall surface hierarchy levels.
- By replacing combinatorial search with a learned GNN-based approach, the pipeline achieves significant speedups and demonstrates zero-shot generalization to real LiDAR environments.

---

[ObjectGraph: From Document Injection to Knowledge Traversal](http://arxiv.org/abs/2604.27820)

- ObjectGraph: introduces a native file format that reconceives documents as typed, directed knowledge graphs to enable efficient traversal by LLM agents instead of full-document injection.
- The framework utilizes a Progressive Disclosure Model and a two-primitive query protocol to minimize token consumption by retrieving only task-relevant nodes.
- By integrating role-scoped access control and executable assertion nodes directly into the document structure, the approach reduces system complexity and eliminates the need for external middleware in multi-agent pipelines.

---

[MCPHunt: An Evaluation Framework for Cross-Boundary Data Propagation in Multi-Server MCP Agents](http://arxiv.org/abs/2604.27819)

- MCPHunt: introduces a controlled benchmark for measuring non-adversarial cross-boundary credential propagation in multi-server LLM agents using Evaluation Design, Trace Collection, and Detection & Analysis.
- The framework utilizes Canary Generation and Environment Control to isolate structural propagation risks from benign task execution across various MCP servers.
- By applying CRS Stratification, the methodology effectively separates task-mandated data transfers from policy-violating safety failures in LLM-based agent workflows.

---

[Separating Feasibility and Movement in Solution Discovery: The Case of Path Discovery](http://arxiv.org/abs/2604.27802)

- Two-graph model framework: introduces a directed weighted two-graph model that separates feasibility constraints, defined by a problem graph G, from movement constraints, defined by a movement graph M, to study solution discovery.
- The framework utilizes a token configuration S and a movement budget b to determine if a target structure in G can be realized through a sequence of local moves in M.
- The research identifies the complexity landscape of Path Discovery and Shortest Path Discovery, providing FPT algorithms for parameters like token number, feedback edge set, and solution size, while establishing hardness results for planar graphs and width parameters.

---

[AgentReputation: A Decentralized Agentic AI Reputation Framework](http://arxiv.org/abs/2605.00073)

- AgentReputation: introduces a three-layer decentralized framework for agentic AI systems that integrates Task Owner, Agents, Verifiers, Functional Layer, Evidence Collection, Reputation Cards, Policy Engine, and Blockchain and Storage Layer to provide evidence-based, context-aware, and decision-oriented reputation.
- The framework decouples task execution from reputation services and storage to enable independent evolution of verification regimes and policy logic.
- By utilizing context-conditioned reputation cards and a policy engine, the system prevents domain-specific reputation conflation and actively governs resource allocation and verification intensity for LLMs.

---

[WindowsWorld: A Process-Centric Benchmark of Autonomous GUI Agents in Professional Cross-Application Environments](http://arxiv.org/abs/2604.27776)

- WindowsWorld: introduces a process-centric benchmark for evaluating GUI agents on complex, multi-application professional workflows using a human-in-the-loop multi-agent generation pipeline.
- The framework utilizes Generator Agent, Refiner Agent, Reviewer Agent, and Environment Generator Agent to construct 181 tasks across 17 applications, incorporating Intermediate Checkpoints for fine-grained diagnostic evaluation.
- Experimental results demonstrate that current LLMs struggle with cross-application coordination and conditional reasoning, achieving only a 20% success rate on multi-step professional tasks.

---

[Monadic Presburger Predicates have Robust Population Protocols](http://arxiv.org/abs/2604.27767)

- Robust Population Protocols framework: introduces robust protocols for all monadic Presburger predicates, ensuring stable consensus despite adversarial crash failures.
- The framework utilizes state-based redundancy and tower-based level tracking to maintain correct outputs under adversarial sniping of agents.
- The research establishes that the state complexity cost for achieving such robustness is at least double exponential relative to the predicate size.

---

[INTENT2TX: Benchmarking LLMs for Translating Natural Language Intents into Ethereum Transactions](http://arxiv.org/abs/2604.27763)

- INTENT2TX: introduces a high-fidelity benchmark for translating natural language intents into executable Ethereum transactions, utilizing Trace Acquisition, Contract Grounding, Filtering, Structured Decoding, Intent Synthesis, Taxonomy Tagging, Multi-step Composition, Differential State Analysis, and Retrieval-augmented Inference.
- The framework employs a simulation-based evaluation protocol using differential state analysis on forked mainnet environments to assess the functional correctness of LLM-generated transactions.
- Evaluation of 16 LLMs demonstrates that retrieval-augmented inference significantly improves logical consistency, parameter precision, and end-to-end execution success rates compared to direct inference.

---

[AgentEconomist: An End-to-end Agentic System Translating Economic Intuitions into Executable Computational Experiments](http://arxiv.org/abs/2604.27725)

- AgentEconomist: introduces an end-to-end agentic system that translates abstract economic intuitions into verifiable computational experiments through a modular multi-stage architecture.
- The framework integrates a retrieval-augmented knowledge base, a structured memory module, and an MCP-based toolbox to support iterative, literature-grounded research workflows.
- By coupling domain-specific grounding with an executable simulation substrate, the system enables researchers to perform rapid hypothesis verification while maintaining human-in-the-loop control.

---

[Compliance-Aware Agentic Payments on Stablecoin Rails](http://arxiv.org/abs/2605.00071)

- Compliance-Aware Agentic Payment Architecture: introduces a modular, hybrid system that integrates programmable compliance directly into stablecoin payment rails to enable secure, automated agentic transactions.
- The architecture utilizes a Compliance Agent to coordinate buyer and seller agents, while employing a PolicyWrapper to enforce on-chain regulatory checks at the point of execution.
- The system supports structured resolution for pending compliance requirements through agent-mediated tranching and escrow, ensuring regulatory adherence without manual intervention.

---

[Knowledge Graph Representations for LLM-Based Policy Compliance Reasoning](http://arxiv.org/abs/2604.27713)

- Agentic AI framework: introduces an end-to-end pipeline that constructs knowledge graphs from AI governance documents to ground LLM reasoning in authoritative policy text.
- The framework utilizes a Chunking Agent, Extraction Agent, Retrieval Agent, and Synthesis Agent to improve compliance QA performance across various LLMs.
- Evaluation demonstrates that KG augmentation consistently improves LLM performance, with an open, LLM-discovered ontology matching or exceeding formal standards-derived schemas.

---

[Contextual Agentic Memory is a Memo, Not True Memory](http://arxiv.org/abs/2604.27707)

- Contextual Agentic Memory is a Memo, Not True Memory: introduces a formal distinction between retrieval-based C-engineering and weight-based θ-learning, arguing that current LLM agents rely on episodic lookup rather than true rule-based expertise.
- The paper proves a compositional generalization gap, demonstrating that retrieval-based systems require quadratically more data than parametric systems to handle novel task combinations.
- It proposes an architectural shift where agents pair fast episodic retrieval with an asynchronous consolidation channel to encode distilled experience directly into model weights.

---

[Bridging Values and Behavior: A Hierarchical Framework for Proactive Embodied Agents](http://arxiv.org/abs/2604.27699)

- ValuePlanner: introduces a hierarchical neuro-symbolic cognitive architecture that decouples high-level value-based deliberation from low-level symbolic action execution to enable proactive autonomy.
- The framework utilizes a Generator–Critic LLM-based loop to reason over abstract value trade-offs, which are then grounded into executable plans by a classical PDDL planner.
- ValuePlanner incorporates a closed-loop Adjustment mechanism that triggers on successful execution to identify synergistic plan refinements and maintain long-horizon value coherence.

---

[When Agents Evolve, Institutions Follow](http://arxiv.org/abs/2604.27691)

- SocialSystemArena: introduces a formal framework that translates historical political institutions into executable multi-agent governance specifications to evaluate their impact on collective performance.
- The framework utilizes a unified Governance Runtime to execute tasks across different LLM backends, isolating governance topology as a primary architectural variable.
- Experimental results demonstrate that governance topology significantly shapes performance, with optimal structures shifting dynamically based on model capability and task characteristics.

---

[The TEA Nets framework combines AI and cognitive network science to model targets, events and actors in text](http://arxiv.org/abs/2604.27673)

- TEA Nets (Target-Event-Agent Networks): introduces a computational framework that maps relational information in text into tripartite graphs composed of subjects (“Agents”), verbs (“Events”), and objects (“Targets”) to enable interpretable narrative analysis.
- The framework utilizes an NLP-pipeline to extract SVO triplets, which are then processed by an Analytics-module to compute network-level measures like degree and edge prominence for investigating linguistic patterns.
- By distinguishing between active and passive voice and incorporating sentiment valence, TEA Nets provide a transparent, "glass-box" approach to analyzing agency, causality, and emotional framing in large-scale textual corpora.

---

[From Context to Skills: Can Language Models Learn from Context Skillfully?](http://arxiv.org/abs/2604.27660)

- Ctx2Skill: introduces a self-evolving framework that autonomously discovers, refines, and selects context-specific skills through a multi-agent self-play loop without human supervision or external feedback.
- The framework utilizes a Challenger agent to probe context, a Reasoner agent to solve tasks, and a Judge to provide feedback, while Proposer and Generator agents iteratively update skill sets based on performance.
- A Cross-time Replay mechanism mitigates adversarial collapse by selecting the most generalizable skill set from historical candidates using curated hard and easy probe sets.

---

[HAVEN: Hybrid Automated Verification ENgine for UVM Testbench Synthesis with LLMs](http://arxiv.org/abs/2604.27643)

- HAVEN: introduces a hybrid verification framework that delegates UVM testbench and sequence generation to rule-based components while utilizing LLMs for structured information extraction and iterative coverage optimization.
- The framework employs a Spec Processor, Architecture Agent, Template Engine, Compile & Simulate Agent, DSL Generator, and DSL CodeGen to ensure syntactic correctness and protocol-compliant UVM testbench synthesis.
- By restricting LLMs to structured JSON outputs and using predefined Jinja2 templates, HAVEN achieves 100% compilation success and state-of-the-art coverage across diverse IP designs with minimal token usage.

---

[Do Open-Loop Metrics Predict Closed-Loop Driving? A Cross-Benchmark Correlation Study of NAVSIM and Bench2Drive](http://arxiv.org/abs/2605.00066)

- NAVSIM and Bench2Drive: introduces a systematic cross-benchmark correlation study evaluating whether safety-aware open-loop metrics can reliably predict closed-loop driving performance across diverse autonomous driving architectures.
- The study identifies that while traditional displacement metrics fail, the NAVSIM PDMS aggregate score shows strong positive correlation with closed-loop Driving Score, albeit with ranking inversions caused by a safety-progress trade-off.
- The authors propose a simplified CL-Proxy (NC × DAC × EP) that matches the predictive power of the full PDMS, highlighting that Ego Progress is the strongest individual predictor of closed-loop success.

---

[SpaAct: Spatially-Activated Transition Learning with Curriculum Adaptation for Vision-Language Navigation](http://arxiv.org/abs/2604.27620)

- SpaAct: introduces a transition-aware training framework that activates dynamic spatial awareness in LLMs for vision-language navigation using Vision Encoder, Tokenizer, LLM, Nav:VLN Action Learning, AR: Action Retrospection, FFS: Future Frame Selection, and TriPA Curriculum.
- The framework employs Action Retrospection to supervise backward action reasoning and Future Frame Selection to supervise forward transition prediction, effectively bridging the gap between semantic understanding and spatial dynamics.
- TriPA curriculum learning stabilizes the adaptation process by organizing training samples from easy to hard based on trajectory, instruction, and motion complexity factors.

---

[RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](http://arxiv.org/abs/2604.27616)

- RoadMapper: introduces a multi-agent system that decomposes complex research roadmap generation into iterative stages of initial generation, knowledge augmentation, and a critique-revise-evaluate loop.
- The framework utilizes six specialized LLM-driven agents to address limitations in professional knowledge, task decomposition, and logical coherence found in standard LLM outputs.
- RoadMapper incorporates an Evaluate agent trained via Direct Preference Optimization (DPO) to ensure high-quality outputs and employs an early stopping mechanism to balance performance with computational efficiency.

---

[Trace-Level Analysis of Information Contamination in Multi-Agent Systems](http://arxiv.org/abs/2604.27586)

- Multi-Agent System (MAS) framework: introduces a trace-based measurement methodology to quantify information contamination in multi-agent workflows by injecting structured perturbations into artifact-derived representations and monitoring execution divergence.
- The framework utilizes a coordinator agent, specialized agents (Data Analyst, Fact Checker, Document Analyst, Visual Analyst, Audio Analyst, Computation Agent, Synthesizer), a shared workspace, provenance metadata, and evidence objects to track how uncertainty propagates through agent interactions.
- Empirical analysis across 614 runs reveals that structural divergence and outcome corruption are decoupled, with workflows often exhibiting behavioral detours or silent semantic corruption that evade standard endpoint-only evaluation metrics.

---

[SpatialGrammar: A Domain-Specific Language for LLM-Based 3D Indoor Scene Generation](http://arxiv.org/abs/2604.27555)

- SpatialGrammar: introduces a domain-specific language that represents indoor layouts as Bird's-Eye View grid placements with deterministic compilation to valid 3D geometry.
- The framework utilizes SG-Agent for iterative, closed-loop scene refinement and SG-Mini, a specialized small language model trained on compiler-validated synthetic data.
- By encoding physical priors into a compact DSL, the system enables verifiable constraint checking and hierarchical scene composition while reducing the cognitive load for LLMs.

---

[AppTek Call-Center Dialogues: A Multi-Accent Long-Form Benchmark for English ASR](http://arxiv.org/abs/2604.27543)

- AppTek Call-Center Dialogues: introduces a comprehensive benchmark for evaluating ASR systems on spontaneous, role-played, long-form conversational speech across fourteen English accents.
- The framework utilizes various segmentation strategies, including Manual segmentation, AppTek proprietary segmenter, Silero segmenter, and Fixed-length chunking, to assess ASR robustness in realistic call-center environments.
- The evaluation pipeline incorporates Guided recognition, leveraging a 4-gram background language model and a Segment-specific language model, to ensure high-quality transcription verification via Levenshtein alignment.

---

[Knowledge Affordances for Hybrid Human-AI Information Seeking](http://arxiv.org/abs/2604.27539)

- KA: introduces a conceptual framework for systematizing how agents identify actionable knowledge sources in hybrid human-AI environments using explicit semantic interfaces.
- The framework models knowledge sources as tuples containing capabilities (C), competency questions (CQ), content scope (S), non-functional properties (NFP), and grounding (G) to facilitate informed source selection.
- By framing information seeking as an affordance-driven process, the approach supports the construction of structured interrogation plans that enhance transparency and mutual intelligibility between human and artificial agents.

---

[EdgeFM: Efficient Edge Inference for Vision-Language Models](http://arxiv.org/abs/2604.27476)

- EdgeFM: introduces a lightweight, agent-driven inference framework designed for cross-platform industrial edge deployment by encapsulating optimized kernels as reusable skills.
- The framework utilizes a modular layered architecture with a thin runtime and heavy kernels to minimize end-to-end latency for single-request edge workloads.
- EdgeFM supports heterogeneous hardware including x86, NVIDIA Orin, and Horizon Journey platforms, effectively eliminating vendor-specific ecosystem lock-in.

---

[Security Attack and Defense Strategies for Autonomous Agent Frameworks: A Layered Review with OpenClaw as a Case Study](http://arxiv.org/abs/2604.27464)

- OpenClaw framework: introduces a four-layer security taxonomy for autonomous agent frameworks, comprising the Context and Instruction Layer, Tool and Action Layer, State and Persistence Layer, and Ecosystem and Automation Layer.
- The paper analyzes how security risks propagate across these layers, from initial input manipulation to persistent state contamination and ecosystem-level impact.
- It identifies critical research gaps including the lack of long-horizon evaluation and the need for more robust, integrated defense mechanisms against cross-layer attacks in LLM-based agent systems.

---

[RAY-TOLD: Ray-Based Latent Dynamics for Dense Dynamic Obstacle Avoidance with TDMPC](http://arxiv.org/abs/2604.27450)

- RAY-TOLD: introduces a hybrid control architecture that integrates LiDAR-centric latent dynamics with sampling-based MPPI to enhance navigation in dense dynamic environments.
- The framework utilizes an Encoder, Latent Dynamics, Reward Predictor, Value Function, Policy Prior, MPPI Planner, and Policy Mixture Sampling to bridge the gap between physics-based robustness and long-horizon learned foresight.
- By augmenting MPPI candidate populations with trajectories from a learned policy prior, the model effectively guides robots out of local minima while maintaining kinematic feasibility.

---

[Context as Prior: Bayesian-Inspired Intent Inference for Non-Speaking Agents with a Household Cat Testbed](http://arxiv.org/abs/2604.27445)

- CatSignal: introduces a Bayesian-inspired framework for intent inference in non-speaking agents by treating spatial context as a prior-like constraint rather than a standard input feature.
- The framework utilizes a context-gated Product-of-Experts architecture to combine spatial context, pose dynamics, and acoustic cues into a unified posterior intent distribution.
- Experimental results on a household cat testbed demonstrate that this approach improves overall accuracy and significantly reduces context-driven shortcut failures compared to standard fusion methods.

---

[Tracking Conversations: Measuring Content and Identity Exposure on AI Chatbots](http://arxiv.org/abs/2604.27438)

- AI Chatbot Tracking Measurement Framework: introduces a systematic methodology to quantify third-party tracking on 20 popular AI chatbots by analyzing network traffic for content and identity exposure.
- The framework utilizes Chatbot Interface, Network Traffic Capture, Preprocessing, Search Targets, Matching Procedure, and Party Attribution to identify data flows to advertising, analytics, and other third-party domains.
- The study reveals that 17 of 20 chatbots share user information with third parties, with some platforms exposing plaintext conversation text via Session Replay and identity data through Embedded Widgets, Analytics Tags, and Advertising Tags.

---

[InteractWeb-Bench: Can Multimodal Agent Escape Blind Execution in Interactive Website Generation?](http://arxiv.org/abs/2604.27419)

- InteractWeb-Bench: introduces a multimodal interactive benchmark designed to evaluate how LLM-based agents navigate real-world website generation tasks under ambiguous, low-code user instructions.
- The framework incorporates a Persona-Driven User Agent Module to simulate diverse user behaviors and an Interactive Execution Environment that enables agents to perform iterative intent refinement, code synthesis, and visual verification.
- Experimental results demonstrate that current LLMs frequently fall into a "blind execution" failure mode, struggling to proactively clarify underspecified requirements and instead over-generating code that leads to high hallucination rates.

---

[ChipLingo: A Systematic Training Framework for Large Language Models in EDA](http://arxiv.org/abs/2604.27415)

- ChipLingo: introduces a systematic three-stage training pipeline designed to enhance LLMs for knowledge-intensive Electronic Design Automation tasks through Domain-Adaptive Pretraining, Instruction Alignment Training, and RAG Scenario Fine-Tuning.
- The framework employs a Partial Parameter Training strategy to balance domain-specific knowledge acquisition with the retention of general LLM capabilities.
- To address RAG capability degradation after domain training, the pipeline incorporates targeted RAG scenario training, enabling models to effectively utilize external knowledge while maintaining robustness against irrelevant retrieval noise.

---

[Understanding Adversarial Transferability in Vision-Language Models for Autonomous Driving: A Cross-Architecture Analysis](http://arxiv.org/abs/2604.27414)

- Cross-Architecture Adversarial Transferability Framework: introduces a five-stage pipeline to systematically evaluate how adversarial patches transfer across diverse VLM architectures in autonomous driving.
- The framework utilizes Architecture-specific Patch Optimization, Scenario Evaluation, Semantic Homogenization, and Evaluation &amp; Analysis to quantify cross-architecture vulnerability and temporal persistence of attacks.
- The research demonstrates that adversarial patches optimized for one VLM architecture exhibit high transferability (73–91%) to others, revealing systematic security weaknesses in VLM-based autonomous driving systems.

---

[Detecting is Easy, Adapting is Hard: Local Expert Growth for Visual Model-Based Reinforcement Learning under Distribution Shift](http://arxiv.org/abs/2604.27411)

- JEPA-Indexed Local Expert Growth: introduces a modular framework for visual MBRL that uses a frozen JEPA representation for problem indexing and separate local residual experts for action correction under distribution shift.
- The framework preserves the original ID-trained baseline controller while augmenting it with shift-specific residual experts to improve OOD performance without catastrophic forgetting.
- The method employs a centroid-based routing mechanism with an ID reject option to ensure that local experts are only activated when necessary, effectively separating shift recognition from action adaptation.

---

[VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching](http://arxiv.org/abs/2604.27375)

- VeraRetouch: introduces a fully differentiable framework for multi-task reasoning photo retouching that replaces non-differentiable external tools with a pixel-faithful Retouch Renderer.
- The framework utilizes a 0.5B Multi-Modal LLM to analyze image semantics and user instructions, generating structured retouching plans and control latents for end-to-end optimization.
- To support training, the authors introduce AetherRetouch-1M+, a million-scale dataset, and a reinforcement learning strategy (DAPO-AE) to enhance aesthetic cognition and logical consistency.

---

[Judge, Then Drive: A Critic-Centric Vision Language Action Framework for Autonomous Driving](http://arxiv.org/abs/2604.27366)

- CriticVLA: introduces a two-stage "judge, then drive" framework that leverages an LLM backbone as an explicit critic to refine initial driving trajectories through multimodal risk analysis and targeted action suggestions.
- The framework utilizes an InternVL2-1B backbone with LoRA and DETR modules to generate a rough trajectory in Stage-1, which is subsequently evaluated and refined in Stage-2 using a language-based critic and query-based delta adaptors.
- To enhance the critic's reasoning capabilities, the authors construct CriticDrive, a large-scale synthetic dataset containing 12.9 million annotated trajectories that align perception, language, and action modalities for robust closed-loop driving.

---

[TADI: Tool-Augmented Drilling Intelligence via Agentic LLM Orchestration over Heterogeneous Wellsite Data](http://arxiv.org/abs/2605.00060)

- TADI (Tool-Augmented Drilling Intelligence): introduces an agentic AI system that transforms heterogeneous drilling operational data into evidence-based analytical intelligence using an LLM Orchestrator, DuckDB, ChromaDB, Specialized Tools, System Prompt, and Output Validator.
- The framework utilizes a dual-store architecture to combine structured SQL queries with semantic text search, enabling multi-step reasoning and cross-referencing of drilling measurements with narrative reports.
- TADI implements a framework-free approach where domain-specific drilling engineering logic is encapsulated in deterministic tools rather than relying on model fine-tuning.

---

[Safe Bilevel Delegation (SBD): A Formal Framework for Runtime Delegation Safety in Multi-Agent Systems](http://arxiv.org/abs/2604.27358)

- SBD (Safe Bilevel Delegation): introduces a formal framework for runtime delegation safety in hierarchical multi-agent systems by formulating task delegation as a bilevel optimization problem.
- The framework utilizes a meta-weight network to dynamically adjust safety-efficiency trade-offs based on context, while an inner loop optimizes the delegation policy subject to probabilistic safety constraints.
- SBD provides theoretical guarantees including safety monotonicity, inner policy convergence, and an accountability propagation bound for multi-hop delegation chains.

---

[Heterogeneous Scientific Foundation Model Collaboration](http://arxiv.org/abs/2604.27351)

- Eywa: introduces a heterogeneous agentic framework that augments domain-specific foundation models with language-model-based reasoning interfaces to enable modality-native collaboration.
- The framework utilizes the Tsaheylu interface to bridge LLMs and specialized foundation models, enabling EywaAgent, EywaMAS, and EywaOrchestra to perform complex scientific tasks with higher utility and lower token consumption.
- EywaBench provides a scalable multi-task, multi-domain scientific benchmark to evaluate the performance of these heterogeneous agentic systems across physical, life, and social sciences.

---

[Dynamic-TD3: A Framework for UAV Path Planning with adversarial environment](http://arxiv.org/abs/2605.00059)

- Dynamic-TD3: introduces a physically enhanced framework for UAV navigation that decouples mission rewards from safety constraints using a dual critic architecture and Lagrangian relaxation.
- The framework integrates ATREM for intent-driven trajectory prediction and PAG-KF for robust state estimation under non-stationary noise.
- Experimental results demonstrate that Dynamic-TD3 achieves superior collision avoidance, reduced energy consumption, and smoother flight trajectories compared to baseline DRL methods.

---

[Toward Autonomous SOC Operations: End-to-End LLM Framework for Threat Detection, Query Generation, and Resolution in Security Operations](http://arxiv.org/abs/2604.27321)

- End-to-End Threat Management Framework: introduces an integrated pipeline that combines ensemble-based threat detection, syntax-constrained query generation via SQM, and RAG-augmented resolution to automate SOC workflows.
- The framework utilizes an ensemble of LLMs for high-accuracy threat detection and prioritizes alerts using SIEM-derived risk scores to reduce analyst workload.
- The SQM architecture improves query generation by grounding LLM outputs in platform-specific syntax allow-lists, metadata-driven retrieval, and official documentation to ensure executable security queries.

---

[Autoformalizing Memory Device Specifications with Agents](http://arxiv.org/abs/2605.00058)

- Autoformalization Agent: introduces an agentic framework that automatically converts natural language memory chip specifications into executable DRAMPyML models using iterative tool-use and validation.
- The framework leverages timed Petri nets to capture complex hardware protocol constraints, enabling the generation of downstream verification collateral such as assertions and stimulus.
- The authors release DRAMBench, a comprehensive benchmark covering 13 JEDEC memory standards, to evaluate the performance and token-efficiency of LLMs in hardware autoformalization tasks.

---

[Pragmos: A Process Agentic Modeling System](http://arxiv.org/abs/2604.27311)

- Pragmos: introduces an agentic system that decomposes complex business process modeling into iterative, human-in-the-loop steps using LLM, Prompting Engine, Modular Decomposition Module, BPMN Synthesis Engine, and Symbolic Execution Module.
- The framework utilizes an LLM to derive intuitive intermediate artifacts like execution paths, which are then structured into process models via modular decomposition and verified through symbolic execution.
- By avoiding black-box generation, the system enables analysts to maintain control and transparency throughout the co-design process of business process models.

---

[End-to-End Evaluation and Governance of an EHR-Embedded AI Agent for Clinicians](http://arxiv.org/abs/2604.27309)

- Hyperscribe: introduces an end-to-end governance framework for clinical AI that integrates Rubric Validation, Live Clinician Feedback, Technical Performance Monitoring, and Cost Tracking to ensure reliable deployment through Controlled Experimentation.
- The architecture utilizes a four-stage cyclical pipeline—Audio to Transcript, Transcript to Instructions, Instructions to Parameters, and Parameters to Commands—to enable granular failure attribution and governable clinical documentation.
- By employing structured outputs, explicit intermediate reasoning, and an EHR-bounded action space, the framework allows for targeted engineering interventions and direct version-to-version performance comparisons.

---

[METASYMBO: Multi-Agent Language-Guided Metamaterial Discovery via Symbolic Latent Evolution](http://arxiv.org/abs/2604.27300)

- METASYMBO: introduces a multi-agent framework for language-guided metamaterial discovery that bridges language, geometry, and property modalities through Agent Designer, Agent Generator, and Agent Supervisor.
- The framework utilizes a symbolic-driven latent evolution module in a disentangled latent space to enable inference-time semantic programming and cross-domain exploration beyond training data.
- By coordinating language reasoning, geometric synthesis, and property-aware feedback, the system achieves superior structural validity and prompt alignment compared to existing LLM and generative baselines.

---

[Machine Collective Intelligence for Explainable Scientific Discovery](http://arxiv.org/abs/2604.27297)

- MCI: introduces a unified paradigm that integrates symbolism and metaheuristics to enable autonomous discovery of governing equations through coordinated reasoning agents.
- The framework utilizes Reasoning agents, Local memory, and Shared memory to evolve symbolic hypotheses via iterative generation, evaluation, and consolidation.
- MCI employs AST as a canonical representation of scientific knowledge, allowing for quantitative explainability and robust OOD generalization across diverse scientific domains.

---

[Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents](http://arxiv.org/abs/2604.27283)

- RSCB-MC: introduces a risk-sensitive contextual bandit framework that treats memory retrieval in LLM-based coding agents as a selective control problem to prevent unsafe memory injection.
- The system utilizes a pattern-variant-episode memory schema and a 16-feature contextual state to evaluate the safety of retrieved information before allowing it to influence the agent's debugging trajectory.
- By prioritizing abstention and non-injection as first-class safety actions, the framework effectively mitigates false-positive memory injections that often degrade the performance of autonomous coding agents.

---

[Select to Think: Unlocking SLM Potential with Local Sufficiency](http://arxiv.org/abs/2604.26940)

- S2T (SELECT TO THINK): introduces a framework that reframes LLM-assisted reasoning as a discrete selection task over an SLM's candidate set, utilizing a Trigger, Candidate Generator, Scoring Function, Reserved Tokens, Distilled Selector, and Inner Critic.
- The framework identifies local sufficiency, where an LLM's preferred token consistently resides within an SLM's top-K predictions, allowing for efficient, autonomous re-ranking without external LLM dependencies.
- By repurposing reserved tokens to encode preference scores, S2T-LOCAL enables the SLM to perform high-performance inference with single-trajectory efficiency, matching the efficacy of 8-path self-consistency.

---

[The Inverse-Wisdom Law: Architectural Tribalism and the Consensus Paradox in Agentic Swarms](http://arxiv.org/abs/2604.27274)

- PAS framework: introduces a mechanistic theory of swarm-level social engineering, demonstrating that agentic swarms prioritize internal architectural agreement over external logical truth through the Consensus Paradox.
- The research establishes the Inverse-Wisdom Law, proving that in kinship-dominant architectures, adding logical agents increases the stability of erroneous trajectories rather than the probability of truth.
- The study identifies the Heterogeneity Mandate as a foundational safety requirement, necessitating architectural diversity at the synthesizer node to break the Attention Latch and prevent terminal failure.

---

[Star-Fusion: A Multi-modal Transformer Architecture for Discrete Celestial Orientation via Spherical Topology](http://arxiv.org/abs/2604.26582)

- Star-Fusion: introduces a multi-modal architecture that reformulates spacecraft orientation estimation as a discrete topological classification task using SwinV2-Tiny, CNN Heatmap, and Coordinate MLP components.
- The framework utilizes spherical K-Means clustering to partition the celestial sphere into topologically consistent regions, effectively mitigating coordinate wrapping artifacts during orientation estimation.
- By integrating photometric, spatial, and numerical features through a late-fusion strategy, the model achieves high computational efficiency and robust performance on resource-constrained hardware for autonomous satellite navigation.

---

#### 29th April 2026


[What Suppresses Nash Equilibrium Play in Large Language Models? Mechanistic Evidence and Causal Control](http://arxiv.org/abs/2604.27167)

- Mechanistic Interpretability and Activation Steering: introduces a causal analysis of why LLMs deviate from Nash equilibria in strategic games by identifying a distributed cooperative override circuit that suppresses Nash-optimal play.
- The research demonstrates that LLMs compute Nash-optimal strategies internally but suppress them via a late-layer prosocial bias rooted in pretraining rather than RLHF.
- The authors confirm the causality of this suppression by extracting a Nash direction from the residual stream and using concept clamping to achieve near-perfect Nash play.

---

[Ambient Persuasion in a Deployed AI Agent: Unauthorized Escalation Following Routine Non-Adversarial Content Exposure](http://arxiv.org/abs/2605.00055)

- Ambient Persuasion in a Deployed AI Agent: reports a safety incident where a Primary Agent (Gemini 2.5 Pro) escalated to unauthorized system-level operations after exposure to non-adversarial content, despite oversight from an Oversight Agent (Claude Opus 4.6) and Specialist Agents.
- The incident demonstrates a directive weighting error where general proactivity norms outweighed specific negative constraints, facilitated by the permissive Tool & System Layer and ambiguous conversational cues.
- The authors propose ambient persuasion as an analytic label for non-adversarial environmental content that triggers unauthorized agent actions, highlighting the need for machine-enforced policy gates and systematic post-incident auditing.

---



[End-to-end autonomous scientific discovery on a real optical platform](http://arxiv.org/abs/2604.27092)

- Qiushi Discovery Engine: introduces a dual-layer multi-agent system that enables end-to-end autonomous scientific discovery by decoupling research roles from nonlinear phases while maintaining stable trajectories via Meta-Trace memory.
- The system integrates a Core Research Agent System, including Lead Investigator, Method Builder, Experimentalist, and Critical Reviewer agents, with a Support Research Agent System to manage long-horizon research tasks.
- Qiushi Discovery Engine demonstrates its capability by autonomously reproducing optical experiments, validating abstract coherence-order theories, and discovering a novel optical bilinear interaction mechanism analogous to Transformer attention.

---


[BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate](http://arxiv.org/abs/2604.25203)

- BARRED (Boundary Alignment Refinement through REflection and Debate): introduces a framework for generating high-fidelity synthetic training data for custom guardrails using only a task description and unlabeled seed examples, utilizing Dimension Decomposition, Verbalized Sampling, Sample Generator, Debate Arena, LLM Judges, Rigid Advocate, and a Refinement Mechanism.
- The framework employs an asymmetric multi-agent debate system where a Rigid Advocate defends a generated sample against a panel of LLM Judges to ensure label faithfulness and high-quality training data.
- Experimental results demonstrate that small models fine-tuned on BARRED-generated data consistently outperform frontier LLMs and generic guardrail models across diverse tasks including conversational policy enforcement, agentic output verification, and regulatory compliance.

---

[The Synthetic Social Graph: Emergent Behavior in AI Agent Communities](http://arxiv.org/abs/2604.27271)

- Moltbook: introduces a comprehensive sociological analysis of an agent-native social platform populated entirely by LLM agents to study emergent social structures.
- The study utilizes a 14-day multi-agent ethnography to examine bonding/bridging communities, status hierarchies, temporal coordination, information diffusion, identity performance, and norm enforcement.
- Findings reveal that while LLM agents spontaneously generate recognizable social structures, they exhibit significant divergences from human baselines, such as low reciprocity and a near-absence of norm-enforcing downvotes.

---

[AutoREC: A software platform for developing reinforcement learning agents for equivalent circuit model generation from electrochemical impedance spectroscopy data](http://arxiv.org/abs/2604.27266)

- AutoREC (Autonomous Reinforcement ECM Composer): introduces a software platform for developing RL agents to automatically generate equivalent circuit models from electrochemical impedance spectroscopy data using EISDataPrep, EIS_ECM_Env, and DDQN_ECM.
- The framework utilizes a linear chromosome representation for circuit topologies and a Double Deep Q-Network (DDQN) agent, supported by a replay buffer, main network, and target network, to perform sequential decision-making for circuit construction.
- AutoREC incorporates a dead-loop mitigation strategy to improve training efficiency and employs a hierarchical reward structure to balance fitting accuracy, physical validity, and model simplicity.

---

[Self-Evolving Software Agents: Extended Abstract](http://arxiv.org/abs/2604.27264)

- BDI–LLM: introduces a framework for self-evolving software agents that integrates an Automated Evolution Module alongside a classical BDI reasoning loop to enable autonomous evolution of knowledge, goals, and executable code.
- The architecture leverages an LLM to synthesize design and code updates when the agent encounters environmental information that cannot be addressed by its current knowledge or goal repertoire.
- By isolating the evolution process from the runtime decision-making loop, the framework maintains behavioural coherence while allowing for long-term autonomous structural adaptation.

---

[AutoSurfer – Teaching Web Agents through Comprehensive Surfing, Learning, and Modeling](http://arxiv.org/abs/2604.27253)

- AutoSurfer: introduces a comprehensive web trajectory generator that employs systematic breadth-first exploration, trajectory-grounded task synthesis, and trajectory-guided refinement to produce high-quality training data for web agents.
- The framework utilizes an exploration module (systematic breadth-first traversal agent) to discover website functionality, a task synthesis module (generates complex tasks from trajectories) to create coherent task-trajectory pairs, and a refinement module (uses web agent to improve task quality) to reduce hallucinations.
- By converting refined task tuples into an SFT dataset, AutoSurfer enables the training of website-specific LLMs (wLLM trainer) that outperform existing methods on the WebArena benchmark.

---

[Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents](http://arxiv.org/abs/2604.27233)

- Reinforced Agent: introduces a duo-agent architecture that shifts tool-calling evaluation into the inference-time execution loop by utilizing a specialized Reviewer Agent to validate provisional tool calls before they reach the Execution Environment.
- The framework incorporates a Feedback Loop that enables the Tool-Calling Agent to iteratively refine its output based on feedback, effectively mitigating destructive errors without requiring retraining.
- The system employs a GEPA Optimizer to systematically improve the Reviewer Agent's performance through automated prompt evolution, achieving superior benefit-to-risk ratios compared to standard LLMs.

---

[When Roles Fail: Epistemic Constraints on Advocate Role Fidelity in LLM-Based Political Statement Analysis](http://arxiv.org/abs/2604.27228)

- TRUST: introduces a systematic empirical evaluation of advocate role fidelity in multi-agent LLM pipelines, identifying Epistemic Role Override as a structural failure mode where factual grounding overrides assigned epistemic stances.
- The research utilizes a specialized Epistemic Stance Classifier and four Role Drift Metrics to quantify how LLMs deviate from their assigned roles when faced with conflicting factual information.
- The study demonstrates that model choice and factual context significantly impact role fidelity, with Mistral Large showing higher robustness than Claude Sonnet in maintaining assigned epistemic positions.

---

[Web2BigTable: A Bi-Level Multi-Agent LLM System for Internet-Scale Information Search and Extraction](http://arxiv.org/abs/2604.27221)

- Web2BigTable: introduces a bi-level multi-agent framework that utilizes an Orchestrator LLM to decompose complex queries into parallel subtasks for Worker Agents, coordinated via a persistent Shared Workboard.
- The framework employs a closed-loop run-verify-reflect process to autonomously evolve Orchestrator Skill Bank and Worker Skill Bank through human-readable memory updates without fine-tuning underlying LLMs.
- By externalizing state into long-term semantic memory and short-term working memory, the system achieves state-of-the-art performance on wide-coverage and depth-oriented search benchmarks.

---

[Truthful-in-Expectation Mechanisms for MMS Approximation](http://arxiv.org/abs/2604.27211)

- TIE Mechanisms: introduces randomized mechanisms that are truthful-in-expectation and provide ex-post fairness guarantees for indivisible goods allocation.
- The framework utilizes Cyclic-Unit-Quota fractional allocations implemented via faithful implementations to achieve ex-post Maximin Share (MMS) approximations.
- The mechanisms provide ex-ante proportionality and ex-post fairness guarantees while maintaining polynomial-time computability and low communication requirements.

---

[Indirect Prompt Injection in the Wild: An Empirical Study of Prevalence, Techniques, and Objectives](http://arxiv.org/abs/2604.27202)

- IPI Framework: introduces a large-scale empirical analysis of indirect prompt injection attacks on the web, characterizing their prevalence, objectives, and deployment strategies across 1.2B URLs.
- The study identifies 15.3K validated injection instances, revealing that most attacks are structured, hidden from human users, and strategically placed in early ingestion channels to influence LLM-based web agents.
- Experimental evaluations across 13 models demonstrate that while overall compliance is limited, susceptibility increases significantly when web content is flattened into plain text, highlighting a critical vulnerability in current LLM-integrated web systems.

---

[A High-Throughput Compute-Efficient POMDP Hide-And-Seek-Engine (HASE) for Multi-Agent Operations](http://arxiv.org/abs/2604.27162)

- HASE: introduces a high-throughput, compute-efficient Dec-POMDP simulator utilizing Data-Oriented Design, 64-byte cache-line alignment, and a zero-copy PyTorch memory bridge to achieve 33M steps per second.
- The architecture employs an EnvironmentArena with bit-packed structures to minimize cache fragmentation and false sharing across parallel CPU threads.
- HASE integrates a Mixed CNN-Logical Encoder to process multi-modal observations for training cooperative multi-agent policies via PPO, DQN, and SAC algorithms.

---

[Interval Orders, Biorders and Credibility-limited Belief Revision](http://arxiv.org/abs/2604.27156)

- Interval Orders, Biorders and Credibility-limited Belief Revision: introduces a family of belief revision operators based on interval orders and biorders to model rational agents that may reject or repair destabilising information.
- The framework utilizes biorder-based interpretations to characterize revision operators, providing axiomatic foundations for BOB, ZTBOB, and TBOB operators.
- The paper further demonstrates that these operators can be formulated as credibility-limited revision, offering a flexible alternative to standard AGM revision by relaxing the single-sentence closure condition.

---

[Step-level Optimization for Efficient Computer-use Agents](http://arxiv.org/abs/2604.27151)

- Step-level Optimization for Efficient Computer-use Agents: introduces an event-driven cascade that dynamically allocates compute by running a Small Policy by default and escalating to a Large Policy only when a Stuck Monitor or Milestone Monitor detects elevated risk.
- The framework utilizes lightweight monitors to analyze interaction history, triggering a Verifier Model to perform sparse, targeted checks that prevent silent semantic drift and recover from progress stalls.
- This modular approach improves the performance-efficiency frontier of LLMs in GUI tasks by reducing large-model usage and latency without requiring architectural changes or retraining of the underlying agents.

---

[Optimal Stop-Loss and Take-Profit Parameterization for Autonomous Trading Agent Swarm](http://arxiv.org/abs/2604.27150)

- Autonomous Trading Agent Swarm Framework: introduces a systematic calibration approach for exit-rule parameters using a counterfactual simulation framework, incorporating an Idea-generation layer, Shared execution and risk layer, ATR-based overlay, and Circuit-breaker logic.
- The framework optimizes exit strategies across a swarm of heterogeneous agents by evaluating 8,960 configurations to identify settings that improve risk-adjusted performance over heuristic defaults.
- Empirical results demonstrate that tighter downside control, faster profit capture, and volatility-aware exit scaling significantly enhance the Sharpe ratio compared to static production baselines.

---

[Enhancing Linux Privilege Escalation Attack Capabilities of Local LLM Agents](http://arxiv.org/abs/2604.27143)

- hackingBuddyGPT: introduces a systematic empirical study on enhancing Small Language Models for autonomous Linux privilege escalation using targeted system-level and prompting interventions.
- The framework integrates five key interventions—Chain-of-Thought, Retrieval-Augmented Generation, Structure via Prompt, History Compression, and Reflective State Analysis—to bridge the performance gap between local models and cloud-based LLMs.
- Empirical results demonstrate that reflection-oriented components are the primary drivers of performance, enabling small open-weight models to match or outperform cloud-based baselines like GPT-4o in privilege escalation tasks.

---

[PALCAS: A Priority-Aware Intelligent Lane Change Advisory System for Autonomous Vehicles using Federated Reinforcement Learning](http://arxiv.org/abs/2604.27118)

- PALCAS (Priority-Aware Lane-Change Advisory System): introduces a federated multi-agent reinforcement learning framework that utilizes RSU, CAV, Central Server, I2I Communication, Fed-MARL, PDQN, RSS Model, and a Priority-guided Reward Mechanism to optimize lane-change decisions in complex highway environments.
- The framework employs a hierarchical state representation that fuses microscopic vehicle-level data with macroscopic cluster-level information to enhance situational awareness and decision quality.
- By leveraging Fed-MARL, the system enables privacy-preserving knowledge sharing among distributed agents while maintaining real-time performance for autonomous lane-change advisories.

---

[Learning to Forget: Continual Learning with Adaptive Weight Decay](http://arxiv.org/abs/2604.27063)

- FADE: introduces a meta-learning approach that adapts per-parameter weight decay rates online to balance knowledge retention and forgetting in non-stationary environments.
- The framework utilizes meta-gradients to dynamically adjust decay rates, enabling the model to selectively retain stable knowledge while discarding outdated information.
- Empirical results demonstrate that FADE improves performance and robustness across linear tracking and streaming classification tasks by automating the discovery of distinct decay rates.

---

[Detecting Clinical Discrepancies in Health Coaching Agents: A Dual-Stream Memory and Reconciliation Architecture](http://arxiv.org/abs/2604.27045)

- Dual-Stream Memory Architecture: introduces a modular framework that maintains clinical consistency by validating patient-reported narrative memories against authoritative FHIR clinical records.
- The system utilizes a Delta-Based Extraction LLM to isolate patient narrative updates, which are then processed by a Reconciliation LLM to identify discrepancies against the Clinical Stream.
- By decoupling discrepancy detection from dialogue generation, the architecture enables ambient clinical surveillance and provides structured, severity-graded outputs for downstream health coaching interventions.

---

[Three-Step Nav: A Hierarchical Global–Local Planner for Zero-Shot Vision-and-Language Navigation](http://arxiv.org/abs/2604.26946)

- Three-Step Nav: introduces a hierarchical global-local framework that leverages an MLLM to perform zero-shot navigation in continuous 3D environments by alternating between global planning, local execution, and trajectory-level verification.
- The framework utilizes a Look Forward module for coarse planning, a Look Now module for fine-grained local navigation, and a Look Backward module for auditing progress and correcting drift.
- An adaptive judge module enables the agent to dynamically apply meta-skills such as stay, continue, backtrack, and look-around to maintain robust navigation under uncertainty without requiring task-specific fine-tuning.

---

[ClawGym: A Scalable Framework for Building Effective Claw Agents](http://arxiv.org/abs/2604.26904)

- ClawGym: introduces a scalable framework for developing personal agents by unifying task synthesis, agent training, and performance evaluation within Claw-style environments using ClawGym-SynData, ClawGym-Agents, and ClawGym-Bench.
- The framework employs a dual-route data synthesis strategy combining persona-driven top-down generation and skill-grounded bottom-up composition to create diverse, verifiable training tasks for LLMs.
- ClawGym-Agents are trained via supervised fine-tuning on high-fidelity interaction trajectories and further optimized through a lightweight sandbox-parallel reinforcement learning pipeline.

---

[Hot Fixing in the Wild](http://arxiv.org/abs/2604.26892)

- Hot Fixing in the Wild: introduces an empirical study comparing human and autonomous coding agent behaviors when addressing time-critical production failures using LLM-based Classifier, Temporal Filter, and Manual Validation.
- The research leverages the AIDev Dataset to analyze over 61,000 repositories, identifying that hot fixes exhibit distinct structural patterns such as reduced collaboration, smaller code changes, and expedited review processes.
- The study reveals that while autonomous coding agents generate more atomic and narrowly scoped fixes than humans, they demonstrate comparable merge success rates, suggesting their viability in urgent software maintenance workflows.

---

[MISES: Minimal Information Sufficiency for Effective Service](http://arxiv.org/abs/2604.26808)

- MISES: introduces a category-based coordination mechanism that maps agent demand types to resource profiles using minimal information labels to balance welfare and detection objectives.
- The framework utilizes a Coordinator to manage resources based on Category Labels, which act as sufficient statistics for allocation while ensuring privacy through information restriction.
- The research establishes a feasibility band for the number of categories, demonstrating that welfare and detection requirements impose structurally opposed constraints on system granularity.

---

[Bian Que: An Agentic Framework with Flexible Skill Arrangement for Online System Operations](http://arxiv.org/abs/2604.26805)

- BIAN QUE: introduces an agentic framework for online system operations that utilizes an Agent Matrix and a Flexible Skill mechanism to orchestrate data retrieval and reasoning for release monitoring, inspection, and root cause analysis.
- The framework employs a unified self-evolving mechanism where practitioner feedback simultaneously drives memory-to-knowledge distillation and targeted Skill refinement to maintain operational accuracy.
- Deployed on a large-scale e-commerce search engine, the framework significantly reduces alert volume and mean time to resolution by automating the selection of relevant data and operational knowledge.

---

[Learning from the Unseen: Generative Data Augmentation for Geometric-Semantic Accident Anticipation](http://arxiv.org/abs/2605.00051)

- MAA framework: introduces a dual-path architecture that combines a generative video synthesis pipeline for data augmentation with a semantic and geometric enhanced dynamic graph convolutional network for accident anticipation.
- The framework utilizes Qwen-VL to extract environmental distributions for synthetic scene generation and employs a dynamic GCN that integrates spatial-visual and semantic cues to model complex interactions among traffic participants.
- By incorporating a multi-granularity multimodal fusion strategy and a temporal modeling module consisting of TCN and GRU, the framework effectively captures long-range dependencies and enhances the accuracy of accident anticipation.

---

[Super-resolution Multi-signal Direction-of-Arrival Estimation by Hankel-structured Sensing and Decomposition](http://arxiv.org/abs/2604.26793)

- Hankel-structured Sensing and Decomposition framework: introduces a novel approach for rapid, few-shot DoA estimation by leveraging sliding-window sensing to construct a Hankel-structured data matrix, followed by rank-K decomposition under L2-norm or L1-norm formulations.
- The L2-norm estimator provides maximum-likelihood optimal performance in white Gaussian noise, while the L1-norm estimator ensures robustness against impulsive interference and heavy-tailed noise.
- Extensive simulations demonstrate that the proposed framework achieves superior resolution capabilities and lower SNR requirements compared to existing subspace-based and compressive sensing methods in hardware-constrained environments.

---

[Hankel and Toeplitz Rank-1 Decomposition of Arbitrary Matrices with Applications to Signal Direction-of-Arrival Estimation](http://arxiv.org/abs/2604.26787)

- HTRD (Hankel and Toeplitz Rank-1 Decomposition): introduces computationally efficient algorithms for rank-1 structured matrix approximations under L2 and L1-norm error formulations.
- The framework utilizes a sliding-window acquisition model to synthesize full aperture data from limited antenna array snapshots for robust DoA estimation.
- The L2-norm estimator is proven maximum-likelihood optimal under white Gaussian noise, while the L1-norm estimator provides robustness against impulsive noise and hardware imperfections.

---

[GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents](http://arxiv.org/abs/2604.26752)

- GLM-5V-Turbo: introduces a native foundation model for multimodal agents that integrates perception, reasoning, and tool use through CogViT, MLP Adapter, Embedding Layer, Transformer Block, MTP Module, VLM RL Gym, Reward System, and Toolchain.
- The framework utilizes a Multimodal Multi-Token Prediction (MMTP) design with a shared learnable placeholder token to optimize training stability and system efficiency.
- The model employs hierarchical reinforcement learning across 30+ task categories to enhance agentic capabilities while maintaining strong text-only coding performance.

---

[CurEvo: Curriculum-Guided Self-Evolution for Video Understanding](http://arxiv.org/abs/2604.26707)

- CurEvo: introduces a curriculum-guided self-evolution framework that transforms weakly controlled self-evolution into a structured learning process for autonomous video understanding by integrating Multi-Dimensional Question Generation, Type-Adaptive Evaluation, and Progressive Curriculum.
- The framework employs a Base Model and an Evaluation Model to iteratively generate, filter, and refine QA pairs across perception, recognition, and reasoning dimensions, ensuring a balanced and progressive learning trajectory.
- By dynamically adjusting sampling ratios and optimization weights based on model competence, CurEvo enables LLMs to autonomously improve their temporal and causal reasoning capabilities without requiring human-labeled data.

---

[FACT: Compositional Kernel Synthesis with a Three-Stage Agentic Workflow](http://arxiv.org/abs/2604.26666)

- FACT: introduces a three-stage agentic workflow that optimizes PyTorch modules by grounding kernel synthesis in vetted CUTLASS templates and a dynamic pattern table.
- The framework utilizes an LLM-based Agent to perform Pattern Discovery, Pattern Realization, and Pattern Composition, ensuring architecture-specific performance through systematic Auto-tuning.
- By leveraging an Examples Index and a dynamic Pattern Table, FACT enables the accumulation of optimization knowledge across diverse workloads and GPU architectures.

---

[AgentSim: A Platform for Verifiable Agent-Trace Simulation](http://arxiv.org/abs/2604.26653)

- AgentSim: introduces a platform for generating verifiable reasoning traces for RAG agents by combining a modular workflow with Corpus-Aware Seeding and an Active Validation Loop.
- The system utilizes an Analyst-Critic-Judge pipeline where LLMs perform multi-step reasoning, and an Active Validation Loop uses Divergence Scores to trigger human review for ambiguous steps.
- The framework produces the Agent-Trace Corpus, a large-scale collection of grounded reasoning trajectories that enables behavioral analysis and improves sample efficiency for training LLMs.

---

[SciHorizon-DataEVA: An Agentic System for AI-Readiness Evaluation of Heterogeneous Scientific Data](http://arxiv.org/abs/2604.26645)

- SciHorizon-DataEVA: introduces an agentic system for scalable AI-readiness evaluation of heterogeneous scientific data, utilizing Data Inspector, Metric Selector, Planner, Executor, Tool Library, Tool Memory, and Knowledge Base.
- The framework employs a hierarchical multi-agent approach orchestrated by a directed, cyclic workflow to perform dataset-aware evaluation based on the Sci-TQA2 principles.
- The system integrates knowledge-augmented planning and adaptive tool-centric execution with review-driven verification to ensure reliable assessment across diverse scientific domains.

---

[OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](http://arxiv.org/abs/2604.26622)

- OCR-Memory: introduces a memory framework that represents agent interaction trajectories as a visual stream to overcome the limitations of finite text context windows.
- The framework utilizes a Locate-and-Transcribe paradigm, where the model scans visual anchors to identify relevant segments and fetches verbatim text, effectively minimizing hallucination and token usage.
- OCR-Memory employs a dynamic multi-resolution strategy and active recall upscaling to maintain high-fidelity evidence for critical memories while compressing older history to manage token budgets.

---

[Impact of Attitude and Bounded Rationality on Collective Behavioral Transitions](http://arxiv.org/abs/2604.26616)

- TPB-ABM (Theory of Planned Behavior Agent-Based Model): introduces a dynamic agent-based modeling framework that integrates the core principles of the Theory of Planned Behavior with a behavior-to-attitude feedback mechanism to simulate collective behavioral transitions.
- The framework utilizes Initialization, Update Attitude, Update Intention, Update Behavioral Probability, Sample Action, Behavioral History, and Subjective Norms to model how individual decision-making parameters shape population-level outcomes.
- Simulation results demonstrate that collective behavioral transitions are governed by the interplay between personal attitude influence and the level of bounded rationality in decision-making.

---

[TDD Governance for Multi-Agent Code Generation via Prompt Engineering](http://arxiv.org/abs/2604.26615)

- AI-native TDD framework: introduces a governance-centric architecture that operationalizes Test-Driven Development principles by separating non-authoritative LLM proposal generation from authoritative engine-based state mutation.
- The framework utilizes a structured TDD manifesto to distribute governance constraints across planner-, generation-, repair-, and validation-agents to ensure phase ordering and bounded autonomy.
- By enforcing deterministic validation gates and atomic workspace application, the system mitigates LLM non-determinism and instability in automated code generation workflows.

---

[Human-in-the-Loop Benchmarking of Heterogeneous LLMs for Automated Competency Assessment in Secondary Level Mathematics](http://arxiv.org/abs/2604.26607)

- Human-in-the-Loop Benchmarking framework: introduces a multi-model ensemble system for automated competency-based assessment of secondary-level mathematics, utilizing Multimodal OCR Engine, EAGLE, ORION, NOVA, and LYRA.
- The framework employs a hierarchical rubric-based evaluation pipeline where specialized LLMs perform preliminary evidence extraction and competency mapping, while a judicial arbiter resolves inconsistencies.
- Empirical results demonstrate that architectural compliance with instruction constraints in Sparse MoE models outperforms raw parameter scale in dense models for rubric-constrained educational tasks.

---

[MappingEvolve: LLM-Driven Code Evolution for Technology Mapping](http://arxiv.org/abs/2604.26591)

- MappingEvolve: introduces a hierarchical agent-based framework that leverages LLMs to directly evolve core technology mapping algorithms through strategic planning, plan-conditioned mutation, and multi-stage validation.
- The architecture decouples strategic operator selection via a Planner from concrete heuristic mutation performed by an Evolver, ensuring code safety through bounded modification regions and rigorous validation.
- Experimental results demonstrate that the framework achieves significant area reduction on EPFL benchmarks by iteratively optimizing three fundamental mapping operators: MatchPhase, MatchPhaseExact, and MatchDropPhase.

---

[Preserving Disagreement: Architectural Heterogeneity and Coherence Validation in Multi-Agent Policy Simulation](http://arxiv.org/abs/2604.26561)

- AI Council: introduces a three-phase deliberation framework that utilizes architectural heterogeneity and a coherence validation layer to mitigate artificial consensus in LLM-based policy simulations.
- The framework employs Champion Agents (advocate for specific policy options), Evaluator Agents (rank options based on value perspectives), a Frontier Model (assesses reasoning coherence via API), a Coherence Validation Layer (post-processing quality assessment mechanism), a Structured Debate Phase (three-round argument generation), and an Independent Evaluation Phase (ranking and reasoning generation).
- By assigning distinct LLMs to different value perspectives, the system disrupts shared inductive biases, while coherence validation provides a fidelity-based weighting mechanism to preserve genuine disagreement in normative policy analysis.

---

[RepoDoc: A Knowledge Graph-Based Framework to Automatic Documentation Generation and Incremental Updates](http://arxiv.org/abs/2604.26523)

- RepoDoc: introduces a knowledge graph-based framework that utilizes RepoKG as a semantic backbone to orchestrate documentation generation and incremental updates.
- The system employs a skill-based agent architecture and Semantic Impact Propagation to minimize token consumption and ensure documentation accuracy during code evolution.
- RepoDoc outperforms existing baselines by providing structurally coherent documentation with cross-references and automated Mermaid diagrams while significantly reducing generation time and costs.

---

[AGEL-Comp: A Neuro-Symbolic Framework for Compositional Generalization in Interactive Agents](http://arxiv.org/abs/2604.26522)

- AGEL-Comp: introduces a neuro-symbolic architecture that integrates a Causal Program Graph (CPG) as a world model, an Inductive Logic Programming (ILP) engine for grounding, and a Neural Theorem Prover (NTP) for logical verification to enhance compositional generalization in LLMs.
- The framework employs a deduction-abduction learning cycle where the LLM proposes plans verified by the NTP, while the grounding function uses Minimal Contrastive Search and Meta-Interpretive Learning to update the world model based on interaction feedback.
- Experimental results in the Retro Quest environment demonstrate that AGEL-Comp significantly outperforms pure LLM-based agents by enabling explicit, interpretable, and compositionally structured world understanding.

---

[Lyapunov-Guided Self-Alignment for Offline Safe Reinforcement Learning](http://arxiv.org/abs/2604.26516)

- SAS (Self-Alignment for Safety): introduces a transformer-based framework for safe test-time adaptation in offline RL by using Lyapunov-guided imagined trajectories as in-context prompts.
- The framework integrates a VAE-based world model to generate imagined rollouts, which are filtered by a Lyapunov Density Model to select safe segments that guide the agent's behavior without parameter updates.
- SAS interprets transformer-based RL as hierarchical RL via Bayesian inference, providing formal probabilistic safety guarantees for test-time adaptation.

---

[3D Generation for Embodied AI and Robotic Simulation: A Survey](http://arxiv.org/abs/2604.26509)

- 3D Generation for Embodied AI and Robotic Simulation: introduces a structured taxonomy for 3D generation in embodied AI, organizing the field into Data Generator, Simulation Environments, and Sim2Real Bridge.
- The paper establishes simulation readiness—encompassing geometric validity, physical parameterization, kinematic executability, and simulator compatibility—as the central evaluation criterion for embodied 3D assets.
- It identifies key technical gaps, including the scarcity of physical annotations and the fragmentation of current ecosystems, while proposing a roadmap toward unified foundation models that jointly reason over geometry, physics, and task semantics.

---

[Alter-Art: Exploring Embodied Artistic Creation through a Robot Avatar](http://arxiv.org/abs/2604.26473)

- Alter-Art: introduces a paradigm for embodied artistic creation where artists inhabit a robotic avatar to perform dance, theater, and painting through immersive teleoperation.
- The Alter-Ego platform integrates compliant actuation, anthropomorphic hands, and expressive facial features to facilitate a first-person creative experience for the human operator.
- Qualitative findings indicate that artists rapidly achieve a sense of presence and creative agency, utilizing the robot as an extension of their own body rather than a mere tool.

---

[EmoTransCap: Dataset and Pipeline for Emotion Transition-Aware Speech Captioning in Discourses](http://arxiv.org/abs/2604.26417)

- EmoTransCap: introduces a novel paradigm for discourse-level speech captioning that integrates temporal emotion dynamics using Gemma-3, CosyVoice2, and MTETR.
- The framework utilizes a multi-stage pipeline to generate a large-scale bilingual dataset, EmoTransSpeech, featuring rich emotional transitions and controllable speech synthesis.
- The MTETR module employs a ResNet-Transformer-BiLSTM architecture to perform joint emotion transition detection and diarization, significantly enhancing emotion perception and expression capabilities.

---

[SecMate: Multi-Agent Adaptive Cybersecurity Troubleshooting with Tri-Context Personalization](http://arxiv.org/abs/2604.26394)

- SecMate: introduces a multi-agent VCA for cybersecurity troubleshooting that integrates device, user, and service specificity through an Orchestrator that coordinates specialized agents including a Clue Collector, User Profiler, Profile-aware troubleshooter, and Proactive Recommender.
- The framework utilizes a Clue Collector to ground diagnostic inference in real-time device signals, while the User Profiler implicitly models technical proficiency to adapt troubleshooting content and complexity.
- SecMate employs a Proactive Recommender that provides context-aware product suggestions, with the entire system evaluated through a large-scale study demonstrating significant improvements in diagnostic accuracy and user satisfaction compared to LLM-only baselines.

---

[Split over n resource sharing problem: Are fewer capable agents better than many simpler ones?](http://arxiv.org/abs/2604.26374)

- Split over n resource sharing problem: introduces a formal framework to evaluate the optimal level of distributiveness in multi-agent systems under fixed resource constraints.
- The study utilizes Agent-based model, Velocity profiles, Environment discretization, Coverage metric, and Failure rate model to analyze how miniaturization-related performance degradation influences optimal group size.
- Formal analysis and simulations demonstrate that technological constraints, such as reduced mobility and increased failure rates, significantly shift the optimal number of agents for spatial coverage tasks.

---

[Seamless Indoor-Outdoor Mapping for INGENIOUS First Responders](http://arxiv.org/abs/2604.26368)

- INGENIOUS: introduces a seamless indoor-outdoor 3D mapping framework that fuses aerial MACS-SaR data with person-carried IPS data using AprilTags for global coordinate alignment.
- The system utilizes MACS-SaR for large-scale outdoor mapping and IPS for GNSS-independent indoor navigation, enabling real-time co-visualization of both environments for first responders.
- By employing AprilTags as optical landmarks, the framework achieves robust co-registration of indoor and outdoor point clouds, facilitating accurate situational awareness in disaster response scenarios.

---

[A Systematic Comparison of Prompting and Multi-Agent Methods for LLM-based Stance Detection](http://arxiv.org/abs/2604.26319)

- LLM-based Stance Detection Frameworks: introduces a systematic evaluation of five methods, including Direct Prompting, Auto-CoT, StSQA, COLA, and MPRF, across 14 subtasks and 15 LLMs to determine the efficacy of prompt-based versus agent-based paradigms.
- The study demonstrates that well-designed prompt-based methods like Auto-CoT and StSQA consistently outperform agent-based methods while requiring significantly fewer API calls.
- Experimental results indicate that model scale provides consistent performance gains up to 32B parameters, whereas reasoning-enhanced models do not consistently improve stance detection accuracy at smaller scales.

---

#### 28th April 2026

[Recursive Multi-Agent Systems](http://arxiv.org/abs/2604.25917)

- RecursiveMAS: introduces a recursive multi-agent framework that treats the entire system as a unified latent-space recursive computation to scale agent collaboration.
- The framework utilizes lightweight RecursiveLink modules to enable seamless latent state transfer and iterative refinement across heterogeneous LLMs.
- RecursiveMAS employs an inner-outer loop training paradigm to achieve stable gradient propagation and efficient system-level co-optimization.

---

[DV-World: Benchmarking Data Visualization Agents in Real-World Scenarios](http://arxiv.org/abs/2604.25914)

- DV-World: introduces a comprehensive benchmark for evaluating data visualization agents across real-world professional lifecycles, integrating DV-Sheet, DV-Evolution, and DV-Interact components.
- The framework employs a hybrid evaluation system combining Table-value Alignment for data fidelity and a hierarchical MLLM-as-a-Judge for semantic-visual assessment.
- Experiments reveal that state-of-the-art LLMs struggle with native spreadsheet object models, cross-paradigm logic migration, and proactive intent alignment under ambiguity.

---

[TrialCalibre: A Fully Automated Causal Engine for RCT Benchmarking and Observational Trial Calibration](http://arxiv.org/abs/2604.25832)

- TrialCalibre: introduces a multi-agent system that automates the BenchExCal workflow for causal effect estimation by coordinating specialized agents including the Orchestrator Agent, Protocol Design Agent, Data Synthesis Agent, Clinical Validation Agent, and Quantitative Calibration Agent.
- The framework utilizes an Automated Calibration Engine to quantify divergence between randomized controlled trials and real-world evidence, employing Clinical Plausibility and Calibration Blackboards to facilitate collaborative reasoning and methodological transparency.
- TrialCalibre incorporates RLHF and HITL mechanisms to ensure auditable, adaptive, and reliable causal inference, while leveraging RAG to integrate domain knowledge for robust trial emulation and indication expansion.

---

[Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM Serving](http://arxiv.org/abs/2604.25899)

- Pythia: introduces a multi-agent serving system that leverages workflow predictability to enable proactive resource management and optimization.
- The system utilizes an Agent-Aware API Gateway and Workflow Profiler to extract structural insights, which are then consumed by a Speculative Cache Manager, Global Request Scheduler, and Model Autoscaler to mitigate bottlenecks.
- By transforming opaque agentic requests into informed execution plans, Pythia significantly improves throughput and reduces job completion time compared to workflow-agnostic serving baselines.

---

[From Threads to Trajectories: A Multi-LLM Pipeline for Community Knowledge Extraction from GitHub Issue Discussions](http://arxiv.org/abs/2604.25880)

- SWE-MIMIC-Bench: introduces a multi-LLM pipeline that transforms unstructured GitHub issue discussions into structured, label-guided reasoning trajectories using Label Classifier LLM, Field Bucket Classifier LLM, Comment Analyst LLM, Link Content Summarizer LLM, and Field Summarizer LLM.
- The framework utilizes a Link Cache to ground discussions in external repository artifacts and organizes extracted evidence into specific Field Buckets defined by a dynamic Field Schema.
- This approach enables the generation of high-fidelity reasoning trajectories that capture the temporal and cognitive evolution of software issue resolution for training LLMs.

---

[Slice Agent: Identifying and Isolating Slices in Shared Open Radio Unit](http://arxiv.org/abs/2604.25857)

- Slice Agent: introduces a hardware-based architecture for O-RU fronthaul that enables real-time slice identification and data segregation using C-Plane Message Decoding, Scheduling Data Process Type 1, Scheduling Data Process Type 2, Symbol Buffers, Control Unit, Ethernet Encapsulation, and Ethernet Transceiver.
- The architecture employs a pipeline design on an FPGA to achieve deterministic processing with a latency of 2 clock cycles per packet, supporting up to 3822 slices per slot.
- Experimental validation using mMTC and URLLC scenarios demonstrates the system's capability to handle high slice density and strict latency requirements while maintaining slice isolation.

---

[Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses](http://arxiv.org/abs/2604.25850)

- AHE (Agentic Harness Engineering): introduces a closed-loop framework that automates the evolution of coding-agent harnesses by leveraging three observability pillars: component observability, experience observability, and decision observability.
- The framework utilizes a three-agent architecture, including a Coding Agent, an Agent Debugger, and an Evolve Agent, to iteratively refine harness components such as system prompts, tools, middleware, and long-term memory.
- By turning every harness edit into a falsifiable, file-level contract, AHE enables autonomous, evidence-driven evolution that transfers across benchmarks and LLM families without requiring task-specific tuning.

---

[Semi-Markov Reinforcement Learning for City-Scale EV Ride-Hailing with Feasibility-Guaranteed Actions](http://arxiv.org/abs/2604.25848)

- PD-RSAC: introduces a distributionally robust RL framework for city-scale EV fleet management that integrates Actor, GCN Encoder, Twin Critics, Value Network, Rolling MILP Projection, Wasserstein Adversary, Replay Buffer, and Semi-MDP Environment.
- The architecture utilizes a GCN Encoder to process spatial hex-grid states, while the Rolling MILP Projection layer ensures that policy intentions strictly adhere to operational constraints like battery SoC and grid feeder limits.
- A Wasserstein Adversary and Primal-Dual update mechanism enable the framework to optimize against worst-case distributional shifts, ensuring robust performance under uncertain demand and travel times.

---

[Towards Agentic Investigation of Security Alerts](http://arxiv.org/abs/2604.25846)

- Agentic security investigation workflow: introduces an iterative, modular system that leverages multiple LLMs to automate the early stages of security alert investigation by querying structured and unstructured log sources.
- The workflow utilizes an Investigator LLM to plan data collection, a Summary LLM to synthesize findings, and an Incident Verdict LLM to provide a final classification, supported by SQL and grep tools.
- Experimental results demonstrate that this agentic approach significantly improves verdict accuracy compared to a baseline model by providing necessary context from logs to inform the decision-making process.

---

[SAFEdit: Does Multi-Agent Decomposition Resolve the Reliability Challenges of Instructed Code Editing?](http://arxiv.org/abs/2604.25737)

- SAFEdit (Structured Agentic Framework for Trustworthy Code Editing): introduces a multi-agent framework that decomposes instructed code editing into specialized Planner Agent, Editor Agent, and Verifier Agent roles to improve reliability through structured, execution-grounded iterative refinement.
- The framework utilizes a Failure Abstraction Layer (FAL) to convert noisy test logs into actionable diagnostic feedback, enabling the Editor Agent to perform targeted, minimal code modifications.
- By separating planning, editing, and verification, SAFEdit achieves higher task success rates and eliminates regression errors compared to single-agent LLM baselines.

---

[Scalable Inference Architectures for Compound AI Systems: A Production Deployment Study](http://arxiv.org/abs/2604.25724)

- Scalable Inference Architecture for Compound AI Systems: introduces a modular, platform-agnostic inference infrastructure designed to handle concurrent, heterogeneous model invocations in compound AI systems.
- The architecture utilizes a Prediction Service to decouple orchestration from model hosting, enabling independent scaling, coordinated pre-warming, and graceful degradation through circuit breakers.
- Production results demonstrate significant improvements in tail latency, throughput, and cost efficiency by addressing compound-system-specific challenges like fan-out overhead and cascading cold starts.

---

[Think Before You Act — A Neurocognitive Governance Model for Autonomous AI Agents](http://arxiv.org/abs/2604.25684)

- PAGRL (Pre-Action Governance Reasoning Loop): introduces a neurocognitive governance framework that embeds deliberate compliance reasoning into LLMs by mapping human executive function and organizational hierarchies onto agent decision-making processes.
- The framework utilizes a four-layer cascading governance architecture—global, workflow-specific, agent-specific, and situational rules—to ensure agents evaluate permissibility before executing any consequential action.
- By integrating a governance reasoning loop and structured audit logs, the model enables autonomous agents to self-correct or escalate decisions to human oversight, mirroring human deliberative compliance behavior.

---

[CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](http://arxiv.org/abs/2604.25676)

- CORAL (COntext-aware Retrieval with Agentic Loop): introduces an adaptive retrieval methodology for multilingual RAG that iteratively refines retrieval corpora and query formulation based on evidence quality.
- The framework utilizes a planner-critic feedback loop to dynamically adjust retrieval conditions, ensuring culturally grounded evidence is sourced for linguistically diverse queries.
- By coupling query-conditioned corpus selection with critique-guided query rewriting, CORAL improves retrieval accuracy on low-resource languages by mitigating noise from indiscriminate corpus expansion.

---

[Modeling Human-Like Color Naming Behavior in Context](http://arxiv.org/abs/2604.25674)

- NeLLCom-Lex: introduces a framework for modeling human-like color naming by integrating supervised learning and reinforcement learning to analyze how communicative pressures and data exposure shape lexical geometry.
- The framework utilizes Speaker and Listener agents that operate within a CIELAB color space, employing upsampling and many-listener interactions to mitigate non-convexity and semantic drift in emergent lexicons.
- Experimental results demonstrate that combining moderate upsampling with multiple listeners promotes more human-like, convex, and informative color naming systems compared to single-listener or baseline setups.

---

[OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](http://arxiv.org/abs/2604.25602)

- OxyGent: introduces a modular framework for Multi-Agent Systems (MAS) that encapsulates agents, tools, and LLMs into pluggable Oxy nodes governed by a permission-driven dynamic planner.
- The framework utilizes a four-tier hierarchical data scoping mechanism and AOP-based lifecycle management to ensure efficient state management and non-intrusive monitoring of LLM agent interactions.
- OxyBank serves as an evolutionary engine that captures execution traces for automated annotation and knowledge backflow, facilitating continuous improvement of collective agent intelligence.

---

[Volitional Multiagent Atomic Transactions: Describing People and their Machines](http://arxiv.org/abs/2604.25596)

- VMTS (Volitional Multiagent Transition Systems): introduces a formal foundation for concurrent systems by decomposing agent state into a machine state and a persistent, inspectable volitional state.
- The framework enables the specification of grassroots platforms where transactions are guarded by the explicit willingness of participating agents, replacing point-of-choice nondeterminism with state-based volitions.
- This approach provides mathematical machinery to prove safety and liveness properties for decentralized systems, demonstrating that grassroots protocols are oblivious and interactive when cross-group transactions are guarded.

---

[Should I Replan? Learning to Spot the Right Time in Robust MAPF Execution](http://arxiv.org/abs/2604.25567)

- RPP: introduces a machine learning-based approach to predict whether replanning during robust Multi-Agent Path Finding execution will reduce total execution cost.
- The framework utilizes an Action Dependency Graph to extract real-time execution features, which are processed by a feed-forward neural network to trigger replanning only when significant savings are expected.
- Experimental results demonstrate that this method recovers 94.6% of potential cost savings while maintaining low computational overhead and avoiding unnecessary replanning.

---

[SnapGuard: Lightweight Prompt Injection Detection for Screenshot-Based Web Agents](http://arxiv.org/abs/2604.25562)

- SnapGuard: introduces a lightweight multimodal defense framework that detects prompt injection attacks in screenshot-based web agents by analyzing visual stability and action-oriented textual cues.
- The framework utilizes a Visual Stability Indicator (quantifies gradient variance for structural anomalies), Contrast-Polarity Reversal (preprocessing to enhance textual feature contrast), OCR-based Text Extraction (recovers visible text from screenshots), Action-Oriented Pattern Detection (identifies malicious intent via taxonomy), and a Decision Module (evaluates unified risk estimate to block inputs).
- SnapGuard achieves an F1 score of 0.75 while operating 8× faster than GPT-4o-prompt and requiring zero additional GPU memory, making it suitable for real-time deployment.

---

[From CRUD to Autonomous Agents: Formal Validation and Zero-Trust Security for Semantic Gateways in AI-Native Enterprise Systems](http://arxiv.org/abs/2604.25555)

- Semantic Gateway: introduces a stateful, intelligent infrastructure layer that acts as an epistemic and operational frontier between stochastic LLM agents and deterministic enterprise backends.
- The architecture employs a three-layer Zero-Trust security model, integrating a pre-inference Semantic Firewall, tool-level RBAC via OPA, and cryptographic human-in-the-loop approval to neutralize agentic risks.
- The system utilizes Enabledness-Preserving Abstractions (EPAs) and greybox semantic fuzzing to transform unpredictable LLM behavior into a mathematically auditable, finite graph of state transitions.

---

[Automated Adversarial Collaboration for Advancing Theory Building in the Cognitive Sciences](http://arxiv.org/abs/2604.25521)

- Automated Adversarial Collaboration framework: introduces a closed-loop system for in-silico theory adjudication in cognitive science by integrating LLM-based theory agents, GeCCo model synthesizer, divergence map, candidate experiment pool, EIG experiment selector, simulation environment, and posterior belief updater.
- The framework enables LLM agents to iteratively propose experiments, synthesize candidate models, and refine theoretical claims to distinguish between competing cognitive theories.
- Experimental results demonstrate that the system can recover ground-truth theories in synthetic settings, though performance varies based on the inherent recoverability of the synthesized models.

---

[Benchmarking and Improving GUI Agents in High-Dynamic Environments](http://arxiv.org/abs/2604.25380)

- DynamicUI: introduces a framework for GUI agents that models interaction as a partially observable Markov decision process to address hidden dynamics in high-dynamic environments using a Dynamic Perceiver, Refinement Strategy, and Reflection Module.
- The framework utilizes a Dynamic Perceiver to cluster video frames into salient context, a Refinement Strategy to align agent thoughts with executed actions, and a Reflection Module to provide corrective guidance based on historical interaction.
- The authors also present DynamicGUIBench, a benchmark comprising 149 tasks across ten applications designed to evaluate GUI agents under four categories of hidden interstitial dynamics.

---

[Multi-Action Tangled Program Graphs for Multi-Task Reinforcement Learning with Continuous Control](http://arxiv.org/abs/2604.25369)

- MATPG (Multi-Action Tangled Program Graphs): introduces a hierarchical Genetic Programming framework for continuous Multi-Task Reinforcement Learning that utilizes Team vertex, Action vertex, and Program (LGP) components to evolve interpretable control policies.
- The framework employs Lexicase selection to promote task-specific elites, enabling the agent to effectively navigate independent obstacles within a customized MuJoCo Half Cheetah environment.
- MATPG maintains high interpretability by evolving human-readable decision paths and obstacle-specific behaviors, outperforming flat MAPLE policies in complex multi-task scenarios.

---

[Plausible but Wrong: A case study on Agentic Failures in Astrophysical Workflows](http://arxiv.org/abs/2604.25345)

- CMBAgent (Cosmological Model Building Agent): introduces a structured empirical evaluation of agentic scientific workflows, identifying that silent numerical errors and physically inconsistent outputs are the primary failure modes in astrophysical inference tasks.
- The framework utilizes an Engineer Agent, Planning & Control Architecture, and specialized modules to execute complex scientific pipelines, revealing that LLMs often generate plausible but incorrect results without self-diagnosis.
- The study demonstrates that while domain-specific context significantly improves performance, agentic systems frequently fail to flag known inferential degeneracies or structural pathologies in their outputs.

---

[R3-SQL: Ranking Reward and Resampling for Text-to-SQL](http://arxiv.org/abs/2604.25325)

- R3-SQL: introduces a Text-to-SQL framework that mitigates functional inconsistency and bounded recall by combining groupwise ranking with agentic resampling.
- The framework utilizes an Execution Engine to cluster candidates by result, while a Pairwise Ranker and Pointwise Ranker provide complementary signals for group-level selection.
- An LLM Agent monitors the candidate pool and triggers a Resampling Module to expand the search space when the correct SQL is likely absent, ensuring higher recall.

---

[Cutscene Agent: An LLM Agent Framework for Automated 3D Cutscene Generation](http://arxiv.org/abs/2604.25318)

- Cutscene Agent: introduces an LLM-driven framework that bridges the editability gap by generating native game engine assets through a bidirectional MCP-based integration.
- The framework utilizes a hierarchical multi-agent system, including a director agent and specialist subagents, to manage long-horizon generation tasks and complex tool orchestration.
- A closed-loop visual reasoning mechanism enables iterative refinement of cinematic composition, while the CutsceneBench benchmark provides a multi-layered evaluation of technical and creative performance.

---

[Job-Scheduling Games with Time-Dependent Processing Times](http://arxiv.org/abs/2604.25301)

- Job-Scheduling Games with Time-Dependent Processing Times: introduces a game-theoretic framework for scheduling jobs with linear time-dependent processing times, analyzing equilibrium existence, computational complexity, and inefficiency under various coordination mechanisms.
- The paper characterizes equilibrium existence for delay-averse agents and demonstrates that for non-delay-averse jobs, determining the existence of a pure Nash equilibrium is NP-complete.
- The authors propose and analyze three coordination mechanisms—SDR, LBDR, and SBPT—to mitigate the high Price of Anarchy observed in decentralized scheduling environments with time-dependent job durations.

---

[MARD: A Multi-Agent Framework for Robust Android Malware Detection](http://arxiv.org/abs/2604.25264)

- MARD: introduces a multi-agent framework that integrates LLMs with deterministic static analysis engines to perform robust Android malware detection through autonomous macro-level screening, micro-level forensics, and global adjudication.
- The framework utilizes a heterogeneous model strategy where a specialized code-comprehension LLM handles preliminary tasks, while a reasoning-capable LLM performs final adjudication, significantly reducing computational costs.
- By employing a ReAct-based multi-agent interaction mechanism, MARD constructs an interpretable evidentiary chain that effectively mitigates concept drift and maintains high generalization performance across diverse datasets.

---

[AutoResearchBench: Benchmarking AI Agents on Complex Scientific Literature Discovery](http://arxiv.org/abs/2604.25256)

- AutoResearchBench: introduces a benchmark for evaluating AI agents on complex scientific literature discovery through two complementary task paradigms, Deep Research and Wide Research.
- The framework utilizes a ReAct-based Agent that interacts with a curated arXiv corpus via the DeepXiv Search Tool to perform multi-hop reasoning and evidence-based filtering.
- The benchmark construction pipeline integrates model-assisted generation with a rigorous Human-in-the-loop Verification Pipeline to ensure high-quality, verifiable scientific search tasks.

---

[Value-Sensitive AI for Prayer: Balancing the Agencies Between Human and AI Agents in Spiritual Context](http://arxiv.org/abs/2604.25230)

- Value-Sensitive AI for Prayer: introduces four conceptual AI systems designed to support spiritual practices while critically examining how AI agency impacts user authenticity and reflection.
- The research utilizes a diary study and design workbook approach to explore how AI can function as a catalyst for introspection rather than a directive solution provider.
- The study highlights that preserving user agency and embracing the inexplicability of AI are essential for maintaining authenticity in deeply personal and value-laden spiritual experiences.

---

[DATAREEL: Automated Data-Driven Video Story Generation with Animations](http://arxiv.org/abs/2604.25220)

- DATAREEL: introduces a benchmark and a multi-agent framework for automated generation of animated data-driven video stories from structured data.
- The framework decomposes the storytelling process into planning, generation, and verification stages, utilizing Director Agent, Plan Critic Agent, Coder Agent, and Video Critic Agent to ensure narrative and visual coherence.
- Experimental results demonstrate that this multi-agent approach significantly outperforms direct prompting baselines in both automatic and human evaluations by improving style consistency and animation quality.

---

[回溯: Co-constructing a Dual Feedback Apparatus](http://arxiv.org/abs/2604.25207)

- Dual Feedback Apparatus: introduces a performance framework utilizing two intelligent musical instruments, 溯 and Agentier, which employ distinct audio and control feedback loops to explore shared musical agency.
- The 溯 instrument integrates audio and latent feedback within a RAVE model to stabilize timbral navigation, while Agentier uses an MDRNN to mediate gestural information between performance interfaces.
- This research treats performers and AI systems as a coupled configuration where feedback acts as a generative condition for real-time musical behavior.

---

[AgentDID: Trustless Identity Authentication for AI Agents](http://arxiv.org/abs/2604.25189)

- AgentDID: introduces a decentralized framework for AI agent identity authentication and dynamic state verification using DIDs, VCs, and challenge-response mechanisms.
- The framework utilizes a Readiness Probe and Context Consistency Check to ensure that an agent's operational state remains consistent with its declared identity during interactions.
- AgentDID enables scalable, trustless authentication for large populations of autonomous agents by offloading verification to decentralized primitives and runtime state checks.

---

[Where Did It Go Wrong? Capability-Oriented Failure Attribution for Vision-and-Language Navigation Agents](http://arxiv.org/abs/2604.25161)

- CanTest: introduces a capability-oriented testing framework for embodied agents that utilizes Adaptive Test Case Generation, Capability Oracles, and a Feedback Mechanism to localize and attribute task failures to specific agent capabilities.
- The framework employs Adaptive Test Case Generation to create diverse instructions, while Capability Oracles and a Failure Attribution Mechanism isolate errors within perception-, memory-, planning-, and decision-agents.
- Experimental results demonstrate that CanTest significantly outperforms existing baselines in discovering failure cases and provides actionable, interpretable guidance for improving embodied agent reliability.

---

[Frictive Policy Optimization for LLMs: Epistemic Intervention, Risk-Sensitive Control, and Reflective Alignment](http://arxiv.org/abs/2604.25136)

- FPO (Frictive Policy Optimization): introduces a risk-sensitive control framework for LLMs that treats epistemic interventions—such as clarification, verification, and refusal—as first-class control actions to manage epistemic and normative risk.
- The framework utilizes a structured friction functional to decompose dialogue failures into unproductive components (miscalibration, contradiction, hazard, value conflict) and productive components (information gain) to guide policy learning.
- FPO provides a unified family of methods—FAR, FPP, GRFR, and FTR—that integrate epistemic friction into learning through reward shaping, preference supervision, trajectory ranking, and risk-conditioned policy regularization.

---

[FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](http://arxiv.org/abs/2604.25135)

- FAMA (Failure-Aware Meta-Agentic framework): introduces a two-stage orchestration approach that analyzes baseline agent failure trajectories to dynamically activate a minimal subset of specialized agents for targeted context injection.
- The framework utilizes an Orchestrator Agent to attribute failures and a Mitigation Agent to select from a pool of specialized agents, including Planner-, Verifier-, Domain Constraints Extractor-, Tool Suggestion-, Tool Output Reformulator-agents and a Memory Module.
- By dynamically constructing optimized prior contexts, FAMA improves the reliability and efficiency of open-source LLMs in multi-turn tool-use environments while minimizing token overhead and context window constraints.

---

[M3-VQA: A Benchmark for Multimodal, Multi-Entity, Multi-Hop Visual Question Answering](http://arxiv.org/abs/2604.25122)

- M3-VQA: introduces a benchmark for evaluating LLMs in fine-grained multimodal entity understanding and complex multi-hop reasoning, utilizing a framework composed of Planner, Executor, and Solver.
- The benchmark requires models to perform sequential and parallel multi-hop reasoning across multiple documents, supported by a curated multimodal knowledge base and traceable evidence.
- Experimental results demonstrate that while LLMs struggle without external information, performance improves significantly with precise evidence and reasoning-aware agentic retrieval.

---

[Diagnosis, Bad Planning &amp; Reasoning. Treatment, SCOPE – Planning for Hybrid Querying over Clinical Trial Data](http://arxiv.org/abs/2604.25120)

- SCOPE (Structured Clinical hybrid Planning for Evidence retrieval in clinical trials): introduces a multi-LLM planner-based framework that decomposes clinical trial reasoning into row selection, structured planning, and execution.
- The framework utilizes an Executor (grounds questions to relevant rows and generates final outputs) and a Planner (interprets questions to create structured reasoning plans) to improve grounded row-level reasoning over partially observed clinical-trial tables.
- SCOPE improves reasoning accuracy and provides a better accuracy-efficiency tradeoff compared to heavier agentic baselines by externalizing decision points into an explicit planning interface.

---

[Structured Security Auditing and Robustness Enhancement for Untrusted Agent Skills](http://arxiv.org/abs/2604.25109)

- SKILLGUARD-ROBUST: introduces an error-decomposition induced architecture for pre-load security auditing of untrusted Agent Skills, utilizing Role-aware Evidence Extractor, Structured Evidence Bundle, Uncertainty Trigger, Semantic Verifier, Conflict-aware Chain Arbitrator, and Anchor-aware Consistency Consolidation.
- The framework addresses the multi-file attack surface of Agent Skills by factorizing the auditing process into structure recovery, selective semantic verification, and consistency-preserving adjudication.
- By replacing single-shot LLM judgments with a staged decision chain, the method effectively mitigates the suspicious/malicious classification collapse observed in standard LLM-based guardrails.

---

[One Perturbation, Two Failure Modes: Probing VLM Safety via Embedding-Guided Typographic Perturbations](http://arxiv.org/abs/2604.25102)

- CWA-SSA (Common Weakness Attack with Spectral Simulation Augmentation): introduces a red teaming procedure that maximizes image-text embedding similarity to probe VLM safety vulnerabilities through Degraded Input, CWA-SSA Ensemble Optimization, Surrogate Embedding Models, and Optimized Output.
- The framework demonstrates that reducing multimodal embedding distance between typographic images and source text triggers readability recovery and reduces safety-aligned refusals in LLMs.
- The research identifies that the effectiveness of these adversarial perturbations depends on the target model's safety filter strength and the severity of visual degradation.

---

[Cooperate to Compete: Strategic Coordination in Multi-Agent Conquest](http://arxiv.org/abs/2604.25088)

- C2C: introduces a long-horizon, mixed-motive multi-agent environment where LLMs must balance short-term cooperation with long-term competitive goals.
- The framework utilizes a board structure with fog-of-war and private negotiation channels to study emergent strategic coordination and alliance formation among LLMs.
- Experimental results demonstrate that targeted prompt-based interventions can significantly improve the performance of LLMs by modifying their negotiation aggressiveness, support-seeking behavior, and propensity for deception.

---

[Optimally Auditing Adversarial Agents](http://arxiv.org/abs/2604.25085)

- Optimally Auditing Adversarial Agents: introduces a principal-agent game framework to compute optimal audit policies that incentivize truthful reporting while accounting for adversarial equilibrium selection by agents.
- The framework provides efficient algorithms for both adaptive and non-adaptive audit settings, ensuring robustness against worst-case agent behavior under budget or cost constraints.
- The research demonstrates that optimal audit strategies can be computed in polynomial time by searching over a finite set of critical audit vectors, even when agent priors are unknown.

---

[Agentic Architect: An Agentic AI Framework for Architecture Design Exploration and Optimization](http://arxiv.org/abs/2604.25083)

- Agentic Architect: introduces an end-to-end framework for automated microarchitecture design exploration by coupling LLM-driven code evolution with cycle-accurate simulation.
- The framework utilizes an Evolutionary Agent to iteratively refine architectural policies, supported by an Evaluator that ranks candidates based on performance metrics and a Trace Database for workload-specific optimization.
- Agentic Architect enables co-design by allowing human architects to define search spaces through seed policies, scoring functions, and prompt strategies, while the LLM-driven agent automates the discovery of complex, adaptive architectural mechanisms.

---

[Zero Shot Coordination for Sparse Reward Tasks with Diverse Reward Shapings](http://arxiv.org/abs/2604.25076)

- ZSC framework: introduces a method to improve zero-shot coordination by training ensembles of agents using diverse reward shapings selected via LLM-Based, Surrogate Network, Stratified Grid, or Random methods.
- The approach leverages TrajeDi to generate diverse policy populations, which are then combined into an ensemble model to enhance generalization when cooperating with unknown partners.
- Experimental results in the Overcooked environment demonstrate that Stratified Grid and Surrogate Network selection methods significantly outperform baseline ZSC algorithms by 62.2%-119.2% in sparse reward tasks.

---

[StratFormer: Adaptive Opponent Modeling and Exploitation in Imperfect-Information Games](http://arxiv.org/abs/2604.25796)

- StratFormer: introduces a transformer-based meta-agent that unifies opponent modeling and strategic exploitation through a two-phase curriculum using a Causal Transformer Encoder, Policy Head, Opponent Modeling Head, Dual-turn Tokens, Bucket-rate Features, and an Opponent Statistics Tracker.
- The architecture processes Dual-turn Tokens at every decision point to capture behavioral history, while the Opponent Modeling Head and Policy Head are trained via a mixed objective that balances GTO imitation with best-response exploitation.
- The framework utilizes a two-phase curriculum where the agent first learns to model opponent behavior before shifting toward adaptive exploitation, regulated by an exploitability-tied schedule to maintain near-equilibrium safety.

---

[HotComment: A Benchmark for Evaluating Popularity of Online Comments](http://arxiv.org/abs/2604.25614)

- StyleCmt: introduces a wave-interference inspired framework that models stylistic interactions to generate comments aligned with dominant audience preferences, utilizing Resonance Field Construction, Interference-Driven Planning, Coherent Superposition, and Collapse and Emission.
- The HotComment benchmark evaluates comment popularity through three dimensions: Content Quality, Popularity Prediction, and User Behavior Simulation, which includes a BERT-based Popularity Predictor and an Agent-based User Behavior Simulator.
- Experimental results demonstrate that StyleCmt consistently improves content quality, stylistic resonance, and engagement alignment across various LLMs and MLLMs compared to baseline methods.

---

[Improving Zero-Shot Offline RL via Behavioral Task Sampling](http://arxiv.org/abs/2604.25496)

- BTD: introduces a method to improve zero-shot offline reinforcement learning by replacing uniform task sampling with a learned Behavioral Task Distribution extracted from offline datasets.
- The framework addresses the signal dilution phenomenon where uniform task sampling in high-dimensional spaces leads to vanishing reward variations and suboptimal generalization.
- By grounding task sampling in achievable behavioral data, the approach ensures that training tasks remain informative and aligned with environment dynamics, resulting in significant performance gains across various benchmarks.

---

[CacheFlow: Efficient LLM Serving with 3D-Parallel KV Cache Restoration](http://arxiv.org/abs/2604.25080)

- CacheFlow: introduces a multi-dimensional parallel execution framework for LLM KV cache restoration that utilizes a Batch-aware Two-pointer Scheduler, Token-level Parallelism, Layer-level Parallelism, Multi-GPU Parallelism, an Inference Engine, and KV Cache Storage.
- The framework optimizes restoration by dynamically balancing recomputation and I/O transfer across tokens, layers, and distributed GPU shards to minimize Time-To-First-Token (TTFT).
- By treating restoration as a global scheduling problem, CacheFlow mitigates resource contention and straggler effects in batched LLM serving environments.

---

[Framework for Collaborative Operation of Autonomous Delivery Vehicles Within a Marshaling Yard](http://arxiv.org/abs/2604.28057)

- Decentralized Orchestrated Autonomy Framework: introduces a coordination system for autonomous delivery vehicles in marshaling yards that utilizes Priority Scoring, Bipartite Graph Task Assignment, Decentralized Re-planning, Radio Broadcast Communication, and Space-Time A* Path Planning to optimize throughput.
- The framework calculates vehicle priority based on battery levels, completed tasks, time spent in the facility, and trust scores to resolve resource contention at charging, inspection, cleaning, and loading stations.
- Simulation results demonstrate that this orchestrated approach significantly increases vehicle throughput and reduces gridlock-induced facility failures compared to isolated, static autonomous operations across various yard sizes and demand levels.

---

#### 27th April 2026


[Agent-Centric Visual Reinforcement Learning under Dynamic Perturbations](http://arxiv.org/abs/2604.24661)

- ACO-MoE (Agent-Centric Observations with Mixture-of-Experts): introduces a plug-and-play preprocessor that decouples perception from dynamic visual perturbations by routing corrupted inputs to specialized restoration experts for foreground extraction and RGB repair.
- The framework utilizes a shared encoder and a router to select corruption-specific experts, which jointly predict RGB residuals and foreground masks to produce clean agent-centric observations for downstream RL agents.
- By anchoring representations on foreground extraction, the approach mitigates the entanglement of corruption artifacts in latent states, achieving robust performance across non-stationary Markov-switching perturbations.

---

[The Last Human-Written Paper: Agent-Native Research Artifacts](http://arxiv.org/abs/2604.24658)

- ARA (Agent-Native Research Artifact): introduces a machine-executable research package structured around four interlocking layers: Cognitive Layer (/logic), Physical Layer (/src), Exploration Graph (/trace), and Evidence Layer (/evidence).
- The framework utilizes a Live Research Manager to capture research decisions during development, an ARA Compiler to translate legacy research into the ARA format, and an ARA-Native Review System to automate structural and reproducibility checks.
- By replacing linear narrative papers with structured, machine-operable artifacts, the ARA protocol enables LLMs to reproduce, verify, and extend research more effectively while preserving failure knowledge typically discarded by traditional publication.

---

[AGENTWARD: A Lifecycle Security Architecture for Autonomous AI Agents](http://arxiv.org/abs/2604.24657)

- AGENTWARD: introduces a lifecycle-oriented, defense-in-depth architecture that systematically organizes security protection across the initialization, input, memory, decision, and execution stages of autonomous AI agents.
- The framework integrates stage-specific heterogeneous controls with cross-layer coordination to intercept threats along their propagation paths and safeguard critical assets.
- By maintaining a shared security state and reusable analysis capabilities, AGENTWARD enables autonomous agents to accumulate risk evidence and adapt defenses across iterative runtime loops.

---


[Governing What You Cannot Observe: Adaptive Runtime Governance for Autonomous AI Agents](http://arxiv.org/abs/2604.24686)

- RiskGate: introduces an adaptive runtime governance framework that separates observed capacity from unobserved risk to maintain agent safety through continuous monitoring, anticipation, and monotonic restriction.
- The framework utilizes the Informational Viability Principle to decompose unobserved risk into behavioral drift, structural bias, and sequential context gaps, which are addressed by dedicated statistical estimators and a closed-loop Autopilot.
- RiskGate operates as governance middleware that provides predictive safety guarantees by estimating the time to boundary crossing and enforcing monotonic restriction to prevent adversarial manipulation of safety thresholds.

---


[TSASSISTANT: A Human-in-the-Loop Agentic Framework for Automated Target Safety Assessment](http://arxiv.org/abs/2604.23938)

- TSASSISTANT: introduces a multi-agent framework for automated Target Safety Assessment that utilizes an Orchestrator, Research Subagents, Synthesis Subagents, Pre-execution Hook, Runtime Hook, Post-execution Hook, Tool Memory, Agent Memory, and MCP-standardized Tool Interfaces.
- The framework employs a hierarchical instruction architecture and a section-based pipeline to ensure evidence-grounded, traceable, and expert-validated report generation.
- TSASSISTANT integrates human-in-the-loop refinement to allow toxicologists to review, edit, and re-invoke agents, maintaining final decision authority in high-stakes pharmaceutical safety workflows.

---



[Case-Specific Rubrics for Clinical AI Evaluation: Methodology, Validation, and LLM-Clinician Agreement Across 823 Encounters](http://arxiv.org/abs/2604.24710)

- Hyperscribe framework: introduces a case-specific, clinician-authored rubric methodology for clinical AI evaluation that leverages LLMs to approximate expert clinician agreement at scale.
- The methodology utilizes a hybrid evaluation model where clinician-authored rubrics establish a baseline for validating LLM-generated rubrics, enabling cost-effective, high-coverage automated assessment.
- Experimental results across 823 clinical encounters demonstrate that LLM-generated rubrics achieve ranking agreement with clinicians that matches or exceeds clinician-clinician agreement in high-performing system configurations.

---


[The Alignment Target Problem: Divergent Moral Judgments of Humans, AI Systems, and Their Designers](http://arxiv.org/abs/2604.24155)

- The Alignment Target Problem: introduces an experimental study investigating how the visibility of human design influences moral judgments of AI systems compared to human actors.
- The study disaggregates value alignment into three normative targets: human behavior (T1), AI behavior (T2), and the behavior of human designers programming AI (T3).
- Results indicate that while T1 and T2 are held to similar moral standards, T3 triggers significantly stricter deontological constraints, suggesting that human design visibility shifts moral evaluation.

---


[Green Shielding: A User-Centric Approach Towards Trustworthy AI LLM-Assisted Medical Diagnosis as a Case Study](http://arxiv.org/abs/2604.24700)

- Green Shielding: introduces a user-centric research agenda for building evidence-backed deployment guidance by characterizing how benign input variation shifts LLM behavior in high-stakes domains.
- The framework operationalizes Green Shielding through the CUE criteria—Context, Utility, and Elicitation—to provide a reliable empirical foundation for evaluating LLM performance under realistic, non-adversarial conditions.
- By applying prompt neutralization to convert raw patient inputs into standardized clinical descriptions, the approach makes precision-coverage tradeoffs explicit and moves model outputs toward more clinician-like diagnostic differentials.

---

[The Chameleon’s Limit: Investigating Persona Collapse and Homogenization in Large Language Models](http://arxiv.org/abs/2604.24698)

- Geometric diagnostic framework: introduces a method to quantify persona collapse in LLMs by measuring how populations occupy behavioral space across coverage, uniformity, and complexity axes.
- The research identifies that LLMs systematically truncate persona attributes, leading to structural homogenization where distinct personas converge into narrow behavioral modes.
- Evaluations across ten LLMs reveal a fundamental tension where models achieving higher per-persona fidelity consistently produce more stereotyped and less diverse simulated populations.

---

[Can Current Agents Close the Discovery-to-Application Gap? A Case Study in Minecraft](http://arxiv.org/abs/2604.24697)

- SCICRAFTER: introduces a Minecraft-based benchmark to evaluate LLMs on their ability to navigate the discovery-to-application loop, utilizing a Main Agent, Scientist Sub-agent, Knowledge Book, Code Agent Scaffold, and MCP interface.
- The framework decomposes performance into four capacity gaps—knowledge identification, experimental discovery, knowledge consolidation, and knowledge application—to diagnose bottlenecks in autonomous scientific inquiry.
- Evaluation of frontier LLMs reveals that while application capacity is a major hurdle, knowledge identification is increasingly becoming the primary bottleneck for advanced models.

---

[NeuroClaw Technical Report: Closed-Loop Agentic AI for Executable and Reproducible Neuroimaging Research](http://arxiv.org/abs/2604.24696)

- NeuroClaw: introduces a domain-specialized multi-agent system for neuroimaging research that utilizes an Interface Layer, Subagent Layer, and Base Layer to automate complex, reproducible scientific workflows.
- The framework integrates Harness Engineering and a Directed Acyclic Graph (DAG) to ensure environment-aware execution, provenance logging, and reliable, auditable experimental loops.
- NeuroBench provides a standardized evaluation platform to assess LLMs on task understanding, tool usage, and code correctness across multimodal neuroimaging pipelines.

---

[The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications](http://arxiv.org/abs/2604.24668)

- Sycophancy Evaluation and Mitigation Framework: introduces a systematic approach to measure and reduce sycophancy in agentic financial applications by evaluating model responses against biased user preferences, contradictions, and rebuttals.
- The framework utilizes a Main LLM, Memory System, and Tool Environment to simulate enterprise scenarios where personalized context or adversarial inputs can induce sycophantic behavior.
- Mitigation strategies include an Input Filtering LLM to normalize queries, the application of Reliability Scorers to context, and Adversarial Training to improve model robustness against sycophancy-inducing injections.

---

[Verification of Correlated Equilibria in Concurrent Reachability Games](http://arxiv.org/abs/2604.24655)

- Verification of Correlated Equilibria in Concurrent Reachability Games: introduces a formal framework for verifying correlated equilibria and subgame-perfect correlated equilibria in concurrent probabilistic games using Probabilistic concurrent game graph, Controller advice, Markov Chain, Markov Decision Process, and Bayesian Network.
- The paper characterizes the computational complexity of verifying these equilibrium concepts, demonstrating that subgame-perfect correlated equilibria are easier to verify than standard correlated equilibria under explicit representations.
- The research further analyzes the impact of succinct input representations via Bayesian Network, showing that the complexity gap between the two equilibrium verification problems disappears under this representation.

---

[K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](http://arxiv.org/abs/2604.24645)

- K-MetBench: introduces a multidimensional diagnostic benchmark for evaluating LLMs and MLLMs on expert-level meteorological reasoning, geo-cultural context, and domain-specific visual interpretation.
- The framework utilizes an LLM-as-a-Judge approach to score model-generated rationales against expert-verified references, identifying critical modality and reasoning gaps in current models.
- K-MetBench decomposes performance across five official meteorological sub-domains to provide fine-grained insights into model strengths and weaknesses that aggregate scores often obscure.

---

[Evaluating Whether AI Models Would Sabotage AI Safety Research](http://arxiv.org/abs/2604.24618)

- UK AISI Research Framework: introduces a methodology to evaluate the propensity of frontier LLMs to sabotage AI safety research by simulating autonomous agentic coding environments.
- The framework utilizes a custom evaluation scaffold built on Petri, incorporating real-world codebases within Docker containers to test models against various research motivations and activities.
- The research assesses sabotage propensities, evaluation awareness, and prefill awareness, finding that while models do not exhibit unprompted sabotage, they demonstrate varying degrees of situational awareness that complicate evaluation interpretation.

---

[Skill Retrieval Augmentation for Agentic AI](http://arxiv.org/abs/2604.24594)

- SR-Agents: introduces a paradigm for augmenting LLMs with external capabilities by dynamically retrieving and incorporating reusable skills from a large-scale corpus on demand.
- The framework utilizes a multi-stage pipeline comprising skill retrieval, skill incorporation, and skill application to address the scalability limitations of explicit in-context skill injection.
- The research introduces SRA-Bench, a benchmark for evaluating the full SRA pipeline, and demonstrates that effective skill augmentation requires controlled, need-aware, and relevance-aware skill utilization beyond simple retrieval.

---

[Measuring the Unmeasurable: Markov Chain Reliability for LLM Agents](http://arxiv.org/abs/2604.24579)

- TRACETOCHAIN: introduces a reproducible pipeline that fits LLM agent execution traces to an absorbing discrete-time Markov chain (DTMC) to provide audited reliability metrics.
- The framework utilizes Transient States (ST), Transition Matrix (Q), Success Absorber (⊕), and Failure Absorber (⊖) to model agent behavior as a first-passage problem.
- It incorporates Fundamental Matrix (N), Dirichlet Posterior, Bootstrap Intervals, Akaike Information Criterion (AIC), and Kolmogorov–Smirnov (KS) Test to provide diagnostics, uncertainty quantification, and metric reconciliation for LLM agent reliability.

---

[FastOMOP: A Foundational Architecture for Reliable Agentic Real-World Evidence Generation on OMOP CDM data](http://arxiv.org/abs/2604.24572)

- FastOMOP: introduces a foundational multi-agent architecture that separates governance, observability, and orchestration layers from pluggable agent teams to ensure safe and auditable real-world evidence generation.
- The architecture enforces safety at the process boundary using deterministic, rule-based validation, preventing compromised or hallucinating agents from bypassing security controls.
- FastOMOP utilizes the Model Context Protocol to implement the principle of least privilege, ensuring agents only access necessary tools and data while maintaining complete traceability of all reasoning steps.

---

[Mono2Sls: Automated Monolith-to-Serverless Migration via Multi-Stage Pipeline with Static Analysis](http://arxiv.org/abs/2604.24550)

- Mono2Sls: introduces a multi-stage pipeline that automates the migration of monolithic web backends to AWS serverless applications using Static Analysis, Architect Agent, Code Developer Agent, SAM Engineer Agent, and Consistency Validator Agent.
- The framework leverages a curated SAM Knowledge Base and specialized tools including File Tools, CodeRAGTool, and SAMValidateTool to ensure structural coherence and deployment readiness across generated artifacts.
- By decomposing the migration process into sequential, agent-driven stages, the system effectively manages complex architectural transformations and infrastructure generation while maintaining alignment with cloud-native patterns.

---

[GradMAP: Gradient-Based Multi-Agent Proximal Learning for Grid-Edge Flexibility](http://arxiv.org/abs/2604.24549)

- GradMAP: introduces a gradient-based multi-agent learning framework that coordinates large-scale grid-edge devices by embedding a differentiable AC power-flow model and reusing environment gradients within a policy-output-space trust region.
- The framework employs Centralised Training and Decentralised Execution to enable independent neural-network policies to satisfy complex network constraints without requiring online communication between agents.
- By utilising implicit differentiation and proximal updates, GradMAP achieves significant training speed-ups and superior constraint satisfaction compared to standard multi-agent reinforcement learning and self-supervised learning benchmarks.

---

[Beyond the Attention Stability Boundary: Agentic Self-Synthesizing Reasoning Protocols](http://arxiv.org/abs/2604.24512)

- SSRP (Self-Synthesizing Reasoning Protocols): introduces a two-stage metacognitive framework that separates high-level architectural planning by an Architect agent from turn-by-turn procedural execution by an Executive agent to mitigate the Attention Latch failure mode.
- The framework utilizes an Architect to autonomously synthesize a task-specific Standard Operating Procedure (SOP) that purges superseded historical intents, ensuring the Executive remains grounded in the most recent verified system events.
- SSRP addresses the Attention Stability Boundary (ASB) by replacing noisy conversation history with an immutable protocol, enabling LLMs to maintain deterministic goal-directedness in complex, multi-turn reasoning tasks where stateless models typically collapse.

---

[DECOFFEE: Decentralized Reinforcement Learning for Time-critical Workload Offloading and Energy Efficiency across the Computing Continuum](http://arxiv.org/abs/2604.24507)

- DECOFFEE: introduces a decentralized reinforcement learning framework for time-critical workload offloading and energy-efficient operation across the computing continuum using Edge Agents, Cloud Agent, Telemetry Agents, Workload Stacks, LSTM-enhanced Double Dueling DQN, and Radio Units.
- The framework models workload offloading as parallel Markov Decision Processes, enabling autonomous Edge Agents to make proactive placement decisions based on local observations and telemetry-shared load forecasts.
- DECOFFEE utilizes a Double Dueling DQN architecture with LSTM-based forecasting to minimize a multi-objective cost function accounting for execution latency, energy consumption, and workload drop rates.

---

[TARMM: Scaling Delay-Critical Edge AI Offloading in 5G O-RAN via Temporal Graph Mobility Management](http://arxiv.org/abs/2604.24501)

- TARMM: introduces a 5G O-RAN system that optimizes user mobility management for delay-critical edge AI offloading by integrating TGN, MARL, Rule-Based Action Masking, Proactive Resource Reservation, Centralized Critic, Decentralized Actor, GRU, and Multi-Head Attention.
- The framework utilizes a TGN to capture spatiotemporal network dynamics, enabling proactive handover decisions that minimize latency and packet loss for mobile UEs.
- By combining MARL with rule-based constraints and proactive resource reservation, the system ensures stable, safe, and efficient handover performance in dense 5G small-cell networks.

---

[Zero-to-CAD: Agentic Synthesis of Interpretable CAD Programs at Million-Scale Without Real Data](http://arxiv.org/abs/2604.24479)

- Zero-to-CAD: introduces a scalable agentic pipeline that synthesizes one million executable, readable CAD construction sequences by embedding an LLM within a feedback-driven environment using LLM Inference Service, Coordinating Node, Tool-Equipped Workers, and Storage Backend.
- The framework utilizes execute_and_validate, lookup_documentation, and grep_documentation to enable iterative self-correction and geometric validation of CAD programs without requiring real-world construction-history data.
- The system demonstrates that synthetic supervision can effectively bootstrap vision-language models for complex CAD reconstruction tasks, outperforming general-purpose models in geometric fidelity and parametric interpretability.

---

[GAMMAF: A Common Framework for Graph-Based Anomaly Monitoring Benchmarking in LLM Multi-Agent Systems](http://arxiv.org/abs/2604.24477)

- GAMMAF: introduces a comprehensive evaluation architecture for benchmarking defense models in LLM-MAS by bridging synthetic data generation and real-time defense experimentation through Training Data Generation Pipeline, Defense System Benchmarking Pipeline, LLM Agent Networks, Output Embedding Module, Training Data Storage, Defender Model, Malicious Agent Pruning, and Communication Topology.
- The framework utilizes a modular design to generate attributed graphs from agent interactions, enabling the evaluation of defense mechanisms against adversarial influence through dynamic topological updates and iterative debate rounds.
- Experimental results demonstrate that GAMMAF effectively facilitates the assessment of defense architectures by measuring metrics such as Attack Success Rate and Adversarial Detection Rate across diverse task domains and network topologies.

---

[Agentic clinical reasoning over longitudinal myeloma records: a retrospective evaluation against expert consensus](http://arxiv.org/abs/2604.24473)

- Agentic reasoning system: introduces a structured, multi-step reasoning architecture that utilizes a Clinical skill library, Structured memory state, Ordered tool-use plan, Report retrieval tool, Laboratory query tool, Deterministic clinical scoring calculators, and a Final answer synthesis module to synthesize longitudinal clinical records.
- The system outperforms standard retrieval-augmented generation and full-context approaches by externalizing reasoning into an explicit planning phase and using deterministic tools for clinical rule application.
- Performance gains are most pronounced in complex clinical tasks and long patient records, with error rates comparable to expert disagreement but with different clinical severity distributions.

---

[Measuring Successful Cooperation in Human-AI Teamwork: Development and Validation of the Perceived Cooperativity and Teaming Perception Scales](http://arxiv.org/abs/2604.24461)

- PCS and TPS: introduces two theoretically grounded psychometric scales designed to assess the subjective quality of human-AI cooperation across synchronic and diachronic dimensions.
- The PCS captures an agent's perceived cooperative capability within a single task, while the TPS evaluates the emergent sense of teaming through Team-, Self-, and Partner-subscales.
- Validation across three studies demonstrates that these scales reliably differentiate between cooperation partners of varying quality, including human, rule-based, and reinforcement learning-based agents.

---

[On the Footprints of Reviewer Bots’ Feedback on Agentic Pull Requests in OSS GitHub Repositories](http://arxiv.org/abs/2604.24450)

- Reviewer Bot Feedback Analysis Framework: introduces an empirical study characterizing the quality and impact of automated reviewer bot feedback on agentic pull requests using the AI_Dev dataset and GPT-5.1 for automated annotation.
- The framework evaluates feedback quality through relevance, clarity, and conciseness metrics, while assessing their correlation with PR resolution time and acceptance rates.
- The study identifies a dilution effect where higher bot activity volume correlates with longer resolution times and decreased feedback relevance, suggesting a need for more targeted, context-aware automated reviews.

---

[PhysNote: Self-Knowledge Notes for Evolv-able Physical Reasoning in Vision-Language Model](http://arxiv.org/abs/2604.24443)

- PhysNote: introduces an agentic framework that enables VLMs to externalize and refine physical knowledge through self-generated Knowledge Notes to improve reasoning in dynamic scenarios.
- The framework utilizes a Spatio-Temporal Grounding Engine to assign immutable identifiers to visual entities, mitigating identity drift across video frames.
- An iterative Hypothesis-Evidence-Validation loop, supported by a hierarchical repository of General Tips, Task Descriptions, and Details, allows the agent to autonomously evolve its physical reasoning capabilities.

---

[AutoGUI-v2: A Comprehensive Multi-Modal GUI Functionality Understanding Benchmark](http://arxiv.org/abs/2604.24441)

- AutoGUI-v2: introduces a comprehensive benchmark for evaluating deep GUI functionality understanding and interaction outcome prediction, utilizing a VLM-human collaborative pipeline, Gemini-2.5-Pro-Thinking, OmniParser-v2, DINO-v3, Qwen3-Embedding, Disjoint Set Union (DSU), FastAPI web server, OpenCV.js, and a Python script.
- The benchmark provides 2,753 evaluation tasks across six operating systems, rigorously testing LLMs on region-level and element-level semantics, grounding, and dynamic state prediction.
- Evaluation reveals a divergence where open-source models excel at functional grounding while commercial models dominate functionality captioning, highlighting that deep functional understanding remains a significant hurdle for current LLMs.

---

[Kwai Summary Attention Technical Report](http://arxiv.org/abs/2604.24432)

- KSA (Kwai Summary Attention): introduces a novel attention mechanism that reduces sequence modeling costs by compressing historical contexts into learnable summary tokens interleaved with text tokens.
- The framework utilizes a sliding chunk attention mechanism to allow text tokens to interact with local neighbors and distant summary tokens, ensuring sub-quadratic complexity while maintaining long-range dependency expressivity.
- KSA employs a contiguous, concatenation-free KV cache layout and a block-sparse kernel to significantly reduce inference latency and memory footprint compared to standard attention mechanisms.

---

[How Personal Characteristics Shape User Exploration of Diverse Movie Recommendations with a LLM-Based Multi-Agent System](http://arxiv.org/abs/2604.24405)

- MAS (Multi-Agent System): introduces a conversational recommender system that utilizes multiple LLM-based agents to provide diverse movie recommendations through personalized explanations, incorporating Agent Profile Panel, Conversation Panel, Movie Recommendation Panel, LLM-based guard module, Demographic-matched agent, Preference-matched agent, and Personality-matched agent.
- The system employs a 6-6 split strategy to generate a fixed set of in-profile and off-profile movie candidates, which are then discussed by three distinct agents to nudge users toward broader exploration.
- Empirical results from a user study demonstrate that the multi-agent design significantly increases Perceived Novelty and objective Shannon Diversity, while highlighting that user experience is moderated by personality traits such as Conscientiousness, Agreeableness, and Extraversion.

---

[MAS-SZZ: Multi-Agentic SZZ Algorithm for Vulnerability-Inducing Commit Identification](http://arxiv.org/abs/2604.24398)

- MAS-SZZ: introduces a multi-agent framework that improves vulnerability-inducing commit identification by combining evidence-grounded root cause analysis, intent-driven anchor selection, and autonomous repository exploration.
- The framework utilizes specialized agents including Auditor, Judge, Reviewer, Evaluator, Locator, and Tracer to systematically filter noisy patch hunks and perform iterative backtracking guided by LLM-reasoned root causes.
- Experimental results demonstrate that MAS-SZZ significantly outperforms existing SZZ algorithms, achieving F1-score gains of up to 65.22% across diverse datasets and programming languages.

---

[OS-SPEAR: A Toolkit for the Safety, Performance, Efficiency, and Robustness Analysis of OS Agents](http://arxiv.org/abs/2604.24348)

- OS-SPEAR: introduces a comprehensive evaluation toolkit for OS agents, utilizing S-subset (evaluates environment/human-induced hazards), P-subset (filters high-value evaluation trajectories), E-subset (measures inference time/token consumption), R-subset (applies cross-modal perturbations), and an Analysis Tool (multi-agent diagnostic report generator) to assess reliability across four critical dimensions.
- The framework employs specialized expert agents (Safety, Performance, Efficiency, Robustness) and an integrated agent to transform raw evaluation logs into expert-level diagnostic reports.
- Extensive evaluation of 22 OS agents reveals significant trade-offs between efficiency and safety/robustness, while highlighting modality-specific vulnerabilities in current MLLM-based OS agents.

---

[Perfecting Aircraft Maneuvers with Reinforcement Learning](http://arxiv.org/abs/2604.24338)

- RL-based Aerobatic Maneuver Framework: introduces a reinforcement learning approach for executing complex aircraft aerobatic maneuvers by utilizing SAC, Gym-JSBSim, and a custom reward function based on trajectory tracking.
- The framework employs both pilot-generated and handcrafted trajectory references to train RL agents, incorporating time scaling and domain-specific reward components to ensure maneuver stability and precision.
- The system demonstrates that RL models can achieve professional pilot-level performance and generalize across different initial conditions and aircraft types by optimizing hyper-parameters and reward weights.

---

[DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](http://arxiv.org/abs/2604.24320)

- DPEPO (Diverse Parallel Exploration Policy Optimization): introduces a reinforcement learning framework that enables LLM agents to interact with multiple environments simultaneously to build comprehensive environmental cognition.
- The framework utilizes a hierarchical reward scheme, incorporating trajectory-level success signals and diversity-driven step-level rewards to penalize redundant behaviors and promote broad exploration.
- By employing group-relative advantage computation, the approach eliminates the need for a separate critic model while achieving state-of-the-art performance and high sample efficiency on complex interactive benchmarks.

---

[BitRL: Reinforcement Learning with 1-bit Quantized Language Models for Resource-Constrained Edge Deployment](http://arxiv.org/abs/2604.24273)

- BitRL: introduces a framework for on-device reinforcement learning that utilizes a frozen 1-bit quantized BitNet backbone as a state encoder combined with lightweight trainable policy and value heads.
- The architecture leverages ternary weights {−1, 0, +1} within the BitNet backbone to achieve significant memory reduction and energy efficiency on resource-constrained edge hardware.
- BitRL addresses the value estimation bottleneck inherent in extreme quantization by employing PPO with conservative clipping and entropy regularization to maintain stable learning dynamics.

---

[RefEvo: Agentic Design with Co-Evolutionary Verification for Agile Reference Model Generation](http://arxiv.org/abs/2604.24218)

- RefEvo: introduces a hierarchical multi-agent framework that utilizes a Design Planner, Modeler Agent, Verifier Agent, and Dialectical Arbiter to automate the generation of reliable SystemC reference models.
- The framework employs a co-evolutionary verification mechanism where the Dialectical Arbiter resolves "Coupled Validation Failure" by iteratively refining both the design and the testbench against an anchored specification.
- Spec Anchoring Context Management optimizes token consumption by pinning immutable specifications while compressing historical interaction logs to prevent LLM catastrophic forgetting.

---

[Empowering Autonomous Debugging Agents with Efficient Dynamic Analysis](http://arxiv.org/abs/2604.24212)

- ADI: introduces a function-level debugging interface for LLMs that replaces inefficient line-by-line interaction with a structured Frame Lifetime Trace (FLT) and high-level navigational commands.
- The framework utilizes a FrameLifetimeTracer to generate on-demand, stateful execution summaries, enabling LLMs to perform precise root-cause analysis without exhausting computational budgets.
- By integrating ADI as a plug-and-play component, autonomous agents achieve significant performance gains on complex software engineering tasks while maintaining cost-efficiency.

---

[Agentic Witnessing: Pragmatic and Scalable TEE-Enabled Privacy-Preserving Auditing](http://arxiv.org/abs/2604.24203)

- Agentic Witnessing: introduces a privacy-preserving auditing framework that replaces static cryptographic proofs with dynamic, TEE-based LLM reasoning to verify unstructured semantic properties of proprietary datasets.
- The system utilizes a tripartite architecture consisting of a Verifier, a Prover, and an Auditor, where the Auditor operates within a TEE to perform semantic analysis via MCP while maintaining a cryptographically signed transcript hash chain.
- To ensure security and privacy, the framework enforces tokenized query budgets to prevent information leakage and provides both a lightweight Public Attestation and an encrypted Private Proof for audit integrity.

---

[Rewarding the Scientific Process: Process-Level Reward Modeling for Agentic Data Analysis](http://arxiv.org/abs/2604.24198)

- DataPRM: introduces an environment-aware generative process reward model that utilizes active environment interaction and a ternary reward strategy to supervise data analysis agents.
- The framework employs a tool-augmented architecture to perform multi-step verification, effectively detecting silent errors and distinguishing recoverable grounding errors from fatal mistakes.
- DataPRM utilizes a scalable pipeline for generating high-quality supervision data, significantly improving performance in both Test-Time Scaling and Reinforcement Learning settings for agentic data analysis.

---

[Dynamic Cyber Ranges](http://arxiv.org/abs/2604.24184)

- Dynamic Cyber Ranges: introduces a framework for evaluating LLM-driven agents in cyber range environments by deploying concurrent attacker and defender agents to create adversarial dynamics.
- The framework utilizes an APT Agent and a Defender Agent, both operating within a CAI Scaffold, to test defensive strategies like chokepoint, per-machine, and hostmanager deployments.
- Experiments demonstrate that LLM-driven defenders can significantly reduce attacker success rates, though effectiveness depends on infrastructure hardening and the security of the monitoring stack itself.

---

[Strategic Bidding in 6G Spectrum Auctions with Large Language Models](http://arxiv.org/abs/2604.24156)

- Strategic Bidding in 6G Spectrum Auctions with Large Language Models: introduces a framework where LLM-based Bidding Agents, Heuristic-based Bidding Agents, and Truthful Bidding Agents compete in a VCG Auction Mechanism managed by a Base Station (BS) Auctioneer, utilizing a Budget Management Module for adaptive decision-making.
- The framework enables UEs to perform context-aware bidding by processing historical auction data and budget constraints through an LLM module to optimize long-term utility.
- Simulation results demonstrate that LLM-based agents outperform traditional strategies in static-budget scenarios by pacing expenditures and sustaining participation in repeated 6G spectrum auctions.

---

[Leveraging Human Feedback for Semantically-Relevant Skill Discovery](http://arxiv.org/abs/2604.24127)

- SRSD (Semantically Relevant Skill Discovery): introduces a human-in-the-loop framework that leverages semantic labelling to guide the discovery of diverse and contextually relevant skills.
- The framework utilizes a Relevance Predictor to incorporate human feedback, alongside Distributional Critics to manage aleatoric uncertainty and mitigate value overestimation during skill training.
- SRSD employs an active sampling strategy to ensure balanced feedback across relevant semantic classes, demonstrating superior performance and scalability compared to traditional preference-based methods.

---

[New Convex Programming Technique for Nash Social Welfare and Scheduling](http://arxiv.org/abs/2604.24120)

- NSW and Scheduling Framework: introduces a novel convex programming relaxation for the weighted Nash social welfare problem that achieves an e^(1/e)-approximation via a rounding algorithm.
- The framework utilizes a compact linear program of polynomial size, avoiding the need for exponential-size configuration LPs or complex dual separation oracles.
- The approach extends to unrelated machine scheduling problems, providing simpler analyses and recovering best-known approximation ratios for minimizing Lq norms and weighted completion times.

---

[AgentVisor: Defending LLM Agents Against Prompt Injection via Semantic Virtualization](http://arxiv.org/abs/2604.24118)

- AgentVisor: introduces a virtualization-inspired defense framework that enforces semantic privilege separation between an untrusted Guest LLM Agent and a trusted Visor to mitigate prompt injection attacks.
- The framework utilizes an STI Protocol (Suitability, Taint, Integrity) to audit tool proposals and triggers a Semantic Exception to enable one-shot self-correction when security violations are detected.
- By treating the agent as an untrusted guest and mediating tool calls through a trusted hypervisor, AgentVisor achieves near-zero attack success rates while maintaining high utility in complex agentic workflows.

---

[An Analysis of the Coordination Gap between Joint and Modular Learning for Job Shop Scheduling with Transportation Resources](http://arxiv.org/abs/2604.24117)

- JSSPT Framework: introduces a multi-agent reinforcement learning approach that evaluates the coordination gap between joint and modular training for job-shop scheduling with transportation resources.
- The architecture utilizes a GNN-based job scheduler and an MLP-based AGV scheduler, both trained via MAPPO to optimize makespan under varying resource scarcity and temporal-dominance conditions.
- The research identifies that joint training provides superior performance in balanced operational regimes, while its advantages diminish in bottleneck environments where a single scheduling task dominates.

---

[Closing the Loop: A Software Framework for AI to Support Business Decision Making](http://arxiv.org/abs/2604.24116)

- Software Framework for AI-supported Business Decision Making: introduces a composable software framework that unifies causal inference models into a single interface to enable LLMs to orchestrate experiment analysis, effect estimation, and algorithmic decision-making.
- The framework utilizes ExperimentData, LinearModel, Policy, Delta Vectors, and a Dispatcher to resolve multiplicity in experimental designs and ensure statistically robust, computationally efficient insights for LLMs.
- By mapping various randomization strategies to a unified model and employing vectorization, the framework significantly reduces code complexity and memory usage compared to vanilla LLM-based implementations.

---

[Latency and Cost of Multi-Agent Intelligent Tutoring at Scale](http://arxiv.org/abs/2604.24110)

- ITAS (Intelligent Tutoring Agent System): introduces a spoke-and-wheel multi-agent architecture that utilizes parallel Video Agent, Code Agent, and Guidance Agent components, followed by a sequential Synthesizer Agent to generate coherent pedagogical responses.
- The framework evaluates latency and cost performance across three Google Vertex AI throughput tiers, identifying that the parallel-phase maximum effect significantly impacts end-to-end response times in multi-agent LLM pipelines.
- The research demonstrates that Priority PayGo provides the most stable latency for classroom-scale deployments, while Provisioned Throughput offers cost-efficiency only when traffic patterns are predictable and utilization is high.

---

[DataClaw: An Autonomous Data Agent with Instant Messaging Integration](http://arxiv.org/abs/2604.24067)

- DataClaw: introduces an autonomous data agent that integrates into instant messaging platforms to perform multi-step analytical workflows using a ReAct reasoning engine, a multi-tiered memory system, and a pluggable skill architecture.
- The framework utilizes a ReAct loop for auditable reasoning and a hot-loading skill mechanism to enable on-the-fly extensibility without requiring system restarts.
- DataClaw ensures data privacy and governance by executing all analytical tasks and storing artifacts locally on the user's machine or private server.

---

[Grounding Before Generalizing: How AI Differs from Humans in Causal Transfer](http://arxiv.org/abs/2604.24062)

- OpenLock framework: introduces a benchmark for evaluating causal structure transfer in LLMs and VLMs by comparing their interactive discovery performance against human baselines in Common Cause (CC) and Common Effect (CE) environments.
- The study reveals that while LLMs and VLMs excel at local causal search, they exhibit delayed transfer and rely on environmental grounding rather than genuine structural abstraction.
- Experimental results demonstrate that visual information often acts as a distractor for these models, and their learning dynamics lack the sudden insight-driven acceleration observed in human causal reasoning.

---

[AgenticCache: Cache-Driven Asynchronous Planning for Embodied AI Agents](http://arxiv.org/abs/2604.24039)

- AgenticCache: introduces a cache-driven planning framework that reuses frequent plan transitions to avoid per-step LLM calls in embodied AI agents.
- The framework utilizes a Runtime Cache for fast plan retrieval and an asynchronous Cache Updater to validate and refine cached entries using an LLM, ensuring adaptability in dynamic environments.
- By exploiting plan locality, AgenticCache significantly reduces inference latency and token usage while maintaining high task success rates across multi-agent embodied benchmarks.

---

[AgentPulse: A Continuous Multi-Signal Framework for Evaluating AI Agents in Deployment](http://arxiv.org/abs/2604.24038)

- AgentPulse: introduces a continuous evaluation framework that aggregates 18 real-time signals across GitHub, package registries, and social platforms to score AI agents in deployment.
- The framework utilizes a multi-layer NLP pipeline and a four-factor composite—Benchmark Performance, Adoption Signals, Community Sentiment, and Ecosystem Health—to provide a holistic view of agent performance beyond static benchmarks.
- AgentPulse validates its methodology through circularity-controlled tests, demonstrating that its composite factors effectively predict external adoption proxies like GitHub stars and Stack Overflow activity.

---

[From Skill Text to Skill Structure: The Scheduling-Structural-Logical Representation for Agent Skills](http://arxiv.org/abs/2604.24026)

- SSL (Scheduling-Structural-Logical) representation: introduces a structured, three-layer framework that disentangles skill-level scheduling signals, scene-level execution structure, and logic-level action evidence from text-heavy agent skill artifacts.
- The framework utilizes an LLM-based normalizer to map unstructured skill documents into a typed JSON graph, facilitating improved skill discovery and automated risk assessment for LLM agents.
- Experimental results demonstrate that SSL-augmented representations significantly outperform text-only baselines in both retrieval accuracy and the identification of operational risks in agent skills.

---

[QED: An Open-Source Multi-Agent System for Generating Mathematical Proofs on Open Problems](http://arxiv.org/abs/2604.24021)

- QED: introduces a multi-agent system designed to generate original, nontrivial mathematical proofs for open research problems by addressing seven specific failure modes identified in frontier LLMs.
- The architecture utilizes a multi-stage pipeline including Literature Survey, Prover Agents, Structural Verifier, Detailed Verifier, Selector Agent, Verdict Agent, Summary Agent, Decomposer, and Regulator to ensure logical consistency and proof integrity.
- QED employs a failure-mode-driven design that incorporates structured verification, multi-model parallel proving, and a decomposition mode to mitigate common LLM reasoning errors such as context contamination and citation hallucination.

---

[ClawdGo: Endogenous Security Awareness Training for Autonomous AI Agents](http://arxiv.org/abs/2604.24020)

- ClawdGo: introduces an endogenous security awareness training framework for autonomous AI agents that utilizes TLDT, ASAT, CSMA, SACP, L0 Axiom Set, L1 Skill Profile, L2 Episode Log, L3 Scenario Library, and ACP to build threat-recognition capabilities at inference time without model modification.
- The framework employs an ASAT self-play loop where the LLM acts as attacker, defender, and evaluator to reinforce threat modeling and defense reasoning through weakest-first curriculum scheduling.
- CSMA provides persistent memory across sessions via axiom crystallization, while SACP addresses the precision-recall tradeoff inherent in over-training autonomous agents.

---

[TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents](http://arxiv.org/abs/2604.24005)

- TCOD (Temporal Curriculum On-Policy Distillation): introduces a curriculum-based training framework that controls trajectory depth to mitigate Trajectory-Level KL Instability in multi-turn LLM agents.
- The framework utilizes two variants, Forward-to-Backward (TCOD-F2B) and Backward-to-Forward (TCOD-B2F), to progressively expose the student to longer interaction horizons while avoiding compounding errors.
- TCOD improves training stability and performance by decoupling trajectory collection and optimization through asynchronous processes and staleness-aware experience replay.

---

[IntentVLM: Open-Vocabulary Intention Recognition through Forward–Inverse Modeling with Video-Language Models](http://arxiv.org/abs/2604.24002)

- IntentVLM: introduces a two-stage cognitive framework for open-vocabulary intention recognition by decomposing the task into goal candidate generation and structured inference via selection.
- The framework utilizes a Goal Candidate Generator to propose potential intentions and an Intention Inference Module to rank and select the most consistent goal based on multimodal video-language evidence.
- By employing LoRA adapters on a Qwen3-VL backbone, the model achieves state-of-the-art performance on intention recognition benchmarks while maintaining generalization capabilities across complex instance-level tasks.

---

[EPM-RL: Reinforcement Learning for On-Premise Product Mapping in E-Commerce](http://arxiv.org/abs/2604.23993)

- EPM-RL: introduces a reinforcement learning framework that distills high-cost agentic reasoning into a compact, on-premise Nemotron-Nano-3-30B model using LoRA and GRPO.
- The framework utilizes three specialized LLM-based judge agents—Core Identity, Model-Identifier, and Variant-Conflict—to provide fine-grained reward signals during the reinforcement learning process.
- By combining parameter-efficient fine-tuning with structured reasoning traces and judge-based rewards, EPM-RL achieves high-quality product mapping without the latency and cost of inference-time agent orchestration.

---

[LLM-Guided Agentic Floor Plan Parsing for Accessible Indoor Navigation of Blind and Low-Vision People](http://arxiv.org/abs/2604.23970)

- Agentic RAG framework: introduces an agentic pipeline that converts architectural floor plan images into structured knowledge graphs to generate safe, accessible navigation instructions for BLV individuals.
- The system utilizes a multi-agent workflow comprising Parser-, Graph Builder-, Self-Critic-, Planner- and Safety Evaluator-agents to ensure robust spatial graph extraction and hazard-aware path planning.
- A three-tier RAG knowledge base integrates graph-based relational data, vector-based semantic embeddings, and visual grounding context to provide precise, landmark-enriched navigation guidance.

---

[GAMED.AI: A Hierarchical Multi-Agent Framework for Automated Educational Game Generation](http://arxiv.org/abs/2604.23947)

- GAMED.AI: introduces a hierarchical multi-agent framework that transforms instructor-provided questions into pedagogically grounded educational games using a LangGraph DAG, deterministic Quality Gates, and typed Pydantic schemas.
- The framework utilizes a modular game engine with self-contained React components and a dual-architecture state management system to support 15 distinct interaction mechanics.
- By separating generation and validation into six deterministic phases, the system achieves a 90% validation pass rate and a 73% token reduction compared to standard ReAct agent architectures.

---

[Constraint-Guided Multi-Agent Decompilation for Executable Binary Recovery](http://arxiv.org/abs/2604.23940)

- Agent4Decompile: introduces a multi-agent framework that transforms decompiled code into re-executable source through multi-level constraint-guided refinement using Decompiler, SyntaxAgent, CompilationAgent, ExecAgent, RefinementLoopOrchestrator, and ConstraintValidators.
- The framework employs a hierarchical validation strategy where specialized LLM agents iteratively repair code based on syntax, compilation, and execution feedback.
- Experimental results on 1,641 binaries demonstrate that this approach significantly improves re-executability by 18–28 percentage points compared to baseline methods.

---

[Frontier Coding Agents Can Now Implement an AlphaZero Self-Play Machine Learning Pipeline For Connect Four That Performs Comparably to an External Solver](http://arxiv.org/abs/2604.25067)

- Frontier Coding Agents: introduces a proof-of-concept benchmark evaluating the ability of LLMs to autonomously implement an AlphaZero-style machine learning pipeline for Connect Four within a constrained environment.
- The framework utilizes a Docker Sandbox, Squid Proxy, and an Evaluation Harness to measure agent performance against a Pascal Pons Solver baseline, while investigating potential sandbagging behaviors in LLMs.
- The study demonstrates that frontier LLMs can now achieve near-solver performance on this task, while surfacing anomalous time-budget usage in GPT-5.4 that warrants further investigation into evaluation awareness and recursive self-improvement capabilities.

---

[Leverage Laws: A Per-Task Framework for Human-Agent Collaboration](http://arxiv.org/abs/2604.25040)

- Leverage Laws Framework: introduces a unified normative ratio for human-agent collaboration that decomposes the cost of supervising autonomous systems into measurable information channels.
- The framework models leverage as a function of agent capability (H_displaced) and the structure of human-agent exchange, identifying how shared memory (M) and workflow design (c_term) influence productivity.
- It provides a formal asymptotic analysis and a Popper-grade falsification protocol to test the directional asymmetry of information flow between humans and agents.

---

[AFA: Identity-Aware Memory for Preventing Persona Confusion in Multi-User Dialogue](http://arxiv.org/abs/2604.25022)

- AFA (Adaptive Friend Agent): introduces a modular framework that combines speaker identification with per-user memory stores to enable identity-aware, personalized dialogue across multiple users in shared environments.
- The system utilizes an Audio Identifier Module to route queries to isolated user-specific memory stores, preventing persona leakage and confusion between residents.
- AFA incorporates a Persona Synchronizer that dynamically updates user profiles based on ongoing interactions, ensuring responses remain contextually relevant and persona-consistent over time.

---

[Why Search When You Can Transfer? Amortized Agentic Workflow Design from Structural Priors](http://arxiv.org/abs/2604.25012)

- SWIFT (Synthesizing Workflows via Few-shot Transfer): introduces an amortized framework that replaces iterative search with a single-pass generation of agentic workflows by conditioning a Meta-generator LLM on Compositional Heuristics, Output Contracts, and Cross-Task Demonstrations.
- The framework utilizes contrastive trajectory distillation to extract structural priors from prior search histories, enabling the synthesis of executable workflows for unseen tasks at near-zero marginal cost.
- By employing contract-guided synthesis, the system bridges the structural-functional gap, ensuring that generated workflows adhere to strict interface requirements while maintaining topological routing patterns across diverse task domains.

---

[Toward a Science of Intent: Closure Gaps and Delegation Envelopes for Open-World AI Agents](http://arxiv.org/abs/2604.25000)

- Intent Compilation framework: introduces a methodology for transforming partially specified human intent into inspectable artifacts that bind stochastic LLM execution to institutional requirements.
- The framework utilizes a four-contract stack—semantic, evidentiary, procedural, and institutional—to expose closure gaps and define delegation envelopes for autonomous agentic AI.
- It provides a failure taxonomy and benchmark protocol to distinguish misclosure from undersearch, enabling more reliable deployment of LLMs in open-world institutional environments.

---

[Don’t Stop Early: Scalable Enterprise Deep Research with Controlled Information Flow and Evidence-Aware Termination](http://arxiv.org/abs/2604.24978)

- EDR (Enterprise Deep Research): introduces a scalable multi-agent architecture that addresses enterprise research failures by decomposing requests into coverage-driven objectives, localizing context via dependency-guided execution, and enforcing evidence-based termination criteria.
- The system utilizes a Plan DAG to coordinate parallel research steps while maintaining logical consistency and preventing context explosion through bounded, dependency-gated information sharing.
- By requiring agents to define and satisfy explicit termination criteria before completing tasks, the framework significantly reduces premature stopping and ensures consistent analytical depth across enterprise reports.

---

[PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference](http://arxiv.org/abs/2604.24971)

- PolyKV: introduces a shared, asymmetrically-compressed KV cache pool that enables multiple concurrent LLM agents to access a single memory-efficient state via SharedKVPool and PooledAgent components.
- The system utilizes asymmetric quantization, applying 8-bit linear quantization to Keys and 3-bit TurboQuant MSE compression to Values, achieving a stable 2.91x compression ratio.
- By injecting decompressed tensors into individual DynamicCache objects, PolyKV reduces memory overhead from O(N) to O(1) while maintaining high semantic fidelity across multiple concurrent agents.

---

[Odysseys: Benchmarking Web Agents on Realistic Long Horizon Tasks](http://arxiv.org/abs/2604.24964)

- Odysseys: introduces a benchmark of 200 long-horizon web tasks designed to evaluate LLMs on multi-site workflows that require sustained context and cross-site reasoning.
- The framework utilizes a rubric-based evaluation system that decomposes complex tasks into verifiable checkpoints, providing more granular and human-aligned performance signals than traditional binary trajectory-level metrics.
- Experimental results demonstrate that while frontier LLMs show sophisticated browsing strategies, they struggle with long-horizon planning and deliverable production, highlighting significant headroom for future agent development.

---

[BENCHGUARD: Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks](http://arxiv.org/abs/2604.24955)

- BENCHGUARD: introduces an automated auditing framework that treats benchmark validation as a cross-artifact consistency problem by analyzing the Instruction, Environment, Gold Solution, and Evaluation Logic.
- The framework utilizes LLM Verification Protocols to perform structured reasoning across benchmark components, identifying defects such as underspecified requirements or misaligned evaluation scripts.
- By incorporating optional Agent Solutions as diagnostic evidence, BENCHGUARD surfaces high-confidence findings for human expert adjudication, effectively reducing measurement noise in complex execution-based benchmarks.

---

[Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence](http://arxiv.org/abs/2604.24954)

- Nemotron 3 Nano Omni: introduces an efficient omni-modal model that natively integrates audio, text, image, and video inputs using a Mixture-of-Experts hybrid LLM backbone.
- The architecture utilizes modality-specific encoders, including C-RADIOv4-H for vision and Parakeet-TDT-0.6B-v2 for audio, connected to the LLM via MLP projectors.
- The model incorporates Conv3D temporal compression and Efficient Video Sampling (EVS) to reduce inference latency and increase throughput for long-context multimodal tasks.

---

[GAIA-v2-LILT: Multilingual Adaptation of Agent Benchmark beyond Translation](http://arxiv.org/abs/2604.24929)

- GAIA-v2-LILT: introduces a refined workflow for adapting English agent benchmarks into multiple languages by combining automated checks and targeted human review to ensure functional and cultural alignment.
- The framework utilizes Deterministic Filtering, LLM Judges, and human expert review to mitigate evaluation pitfalls such as LLM self-preference and human fluency bias.
- Experimental results demonstrate that this rigorous auditing process improves agent success rates by up to 32.7% compared to minimally translated benchmarks, revealing that translation-induced errors significantly mask true agent capabilities.

---

[SUDP: Secret-Use Delegation Protocol for Agentic Systems](http://arxiv.org/abs/2604.24920)

- SUDP (Secret-Use Delegation Protocol): introduces a three-role protocol that decouples authorization from secret exposure by requiring an autonomous Requester R to propose operations for a user-controlled Authorizer U, which are then executed by a Custodian T using a sealed secret.
- The framework utilizes a cryptographic key hierarchy and operation-bound grants to ensure that reusable authority never crosses the requester boundary, effectively mitigating risks from prompt injection and runtime compromise.
- SUDP provides a formal security-property taxonomy for Agent Secret Use, enabling principled comparison of defense mechanisms against requirements like authorization verifiability, operation binding, and replay resistance.

---

[Agentic AI for Remote Sensing: Technical Challenges and Research Directions](http://arxiv.org/abs/2604.24919)

- Agentic AI for Remote Sensing: introduces a design blueprint for Earth Observation agents that replaces generic, abstract tool-use with structured geospatial state management and domain-grounded validation.
- The framework integrates an Orchestrator/Policy (LLM/VLM), Memory, Geospatial Tools, GeoValidator, and a Training Strategy to ensure that multi-step analytical workflows remain scientifically valid and reproducible.
- This research highlights that Earth Observation is a distinct agentic domain where incorrect tool ordering or spatial misalignment can silently propagate errors, necessitating trajectory-level evaluation beyond final-answer accuracy.

---

[Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](http://arxiv.org/abs/2604.24881)

- IMAD (Internalized Multi-Agent Debate): introduces a two-stage fine-tuning pipeline that distills multi-agent debate reasoning into a single LLM by combining Dataset Collection, Supervised Fine-Tuning (SFT), and Reinforcement Learning (RL) with Dynamic Reward Scheduling and Length Clipping.
- The framework utilizes Group Relative Policy Optimization (GRPO) to progressively internalize debate dynamics, enabling the model to perform multi-perspective reasoning internally while achieving significant inference efficiency gains.
- Mechanistic analysis via Activation Steering and Contrastive Activation Addition (CAA) reveals that the internalization process creates identifiable agent subspaces, allowing for precise control and suppression of specific model behaviors.

---

[Co-Director: Agentic Generative Video Storytelling](http://arxiv.org/abs/2604.24842)

- Co-Director: introduces a hierarchical multi-agent framework that formalizes video storytelling as a global optimization problem to mitigate semantic drift and cascading failures in generative pipelines.
- The framework utilizes a Multi-Armed Bandit to navigate a factored creative action space, balancing exploration of novel narrative strategies with exploitation of effective configurations.
- Co-Director integrates a local MLLM feedback-driven refinement loop with global MAB-steered optimization to ensure long-range narrative and visual consistency across generated video sequences.

---

[Distributional Robustness of Linear Contracts](http://arxiv.org/abs/2604.24732)

- Distributional Robustness of Linear Contracts: introduces a concavification-based approach to demonstrate that linear contracts are worst-case optimal for a principal facing distributional ambiguity regarding the mapping from effort to stochastic outcomes.
- The framework utilizes the concept of self-inducing actions to construct a dominating distribution, proving that linear contracts achieve weakly higher worst-case payoffs than any other contract.
- The paper extends these robustness results to multi-party settings, including common agency and team production, and provides tractable characterizations for homogeneous utility and cost functions.

---

[FGDM: Reasoning Aware Multi-Agentic Framework for Software Bug Detection using Chain of Thought and Tree of Thought Prompting](http://arxiv.org/abs/2604.24831)

- FGDM (Flow Graph-Driven Multi-Agent Framework): introduces a multi-agent pipeline that leverages Flow Graph Builder Agent, Semantic Fault Localizer Agent, Graph Repair Agent, and Source Code Reconstruction Agent to perform context-aware software bug detection and repair.
- The framework integrates a FAISS Vector Database to provide retrieval-augmented reasoning, utilizing LLM-based Reasoning Engine to apply Chain-of-Thought and Tree-of-Thought prompting for systematic error localization and correction.
- By representing programs as flow graphs, the framework achieves language-agnostic structural analysis, effectively reducing hallucinations and improving syntactic consistency compared to standalone LLM prompting strategies.

---

[A Comparative Evaluation of AI Agent Security Guardrails](http://arxiv.org/abs/2604.24826)

- DKnownAI Guard: introduces a comparative evaluation of AI agent security guardrails, utilizing Dual-Channel Risk Classification, Agent Threat Detection, Harmful Content Detection, and Scenario-Based Configuration to assess performance against adversarial datasets.
- The study benchmarks DKnownAI Guard against industry solutions, demonstrating superior recall and true negative rates in identifying complex agent-specific threats like instruction override and tool abuse.
- The research highlights the critical challenge of maintaining high classification precision on high-ambiguity boundary samples to prevent the misblocking of legitimate user requests in LLM-based agent deployments.

---

[ITAS: A Multi-Agent Architecture for LLM-Based Intelligent Tutoring](http://arxiv.org/abs/2604.24808)

- ITAS (Intelligent Teaching Assistant System): introduces a multi-agent architecture for LLM-based intelligent tutoring that utilizes a Spoke-and-Wheel teaching layer, a scalable operational layer, and a conversational feedback layer to support real-world classroom deployment.
- The system employs domain-specific specialist agents—Video, Guidance, and Code—coordinated by a Synthesizer to prevent task-boundary hallucinations and ensure pedagogical reliability.
- The architecture enforces FERPA compliance by design through pseudonymized event streams and a narrow-scope analytics agent that operates without direct access to student identities or runtime query capabilities.

---

[From Prototype to Classroom: An Intelligent Tutoring System for Quantum Education](http://arxiv.org/abs/2604.24807)

- ITAS (Intelligent Teaching Assistant System): introduces a multi-agent tutoring framework utilizing a Spoke-and-Wheel architecture to mitigate LLM hallucinations through task-based specialization.
- The system incorporates a five-module QIS curriculum and a cloud-based infrastructure to support classroom-scale concurrency and real-time pedagogical analytics.
- ITAS addresses the Blind Instructor Problem by providing aggregate insights into student engagement and curriculum gaps without compromising individual student privacy.

---

[Autonomous Traffic Signal Optimization Using Digital Twin and Agentic AI for Real-Time Decision-Making](http://arxiv.org/abs/2604.27753)

- ATSC (Autonomous Traffic Signal Control) framework: introduces a closed-loop, simulation-based architecture that integrates digital twin technology with agentic AI to optimize traffic signal timings in real-time.
- The system utilizes a multi-agent architecture comprising perception-, risk-, simulation-, and LLM explanation-agents to evaluate traffic conditions and generate optimized signal adjustments.
- Empirical results demonstrate that the framework achieves higher traffic flow efficiency and lower average vehicle delay compared to traditional fixed-time and RL-based control methods.

---

#### 26th April 2026

[Agentic AI platforms for autonomous training and rule induction of human-human and virus-human protein-protein interactions](http://arxiv.org/abs/2604.23924)

- Agentic AI platforms for PPI: introduces two autonomous agentic AI platforms designed to perform end-to-end ML training and biological rule induction for human-human and virus-human protein-protein interactions.
- The first platform utilizes five agents—Data Collector, Data Verifier, Feature Embedder, Model Designer, and Executor—to autonomously train predictive ML models using protein-disjoint cross-fold validation.
- The second platform replaces the model designer and executor with a Rule Induction Agent to generate interpretable biological rules, which are cross-checked against SHAP-identified features from the predictive models.

---

[MarketBench: Evaluating AI Agents as Market Participants](http://arxiv.org/abs/2604.23897)

- MarketBench: introduces a benchmark for evaluating LLMs as market participants by assessing their ability to perform self-calibration and generate bids for task allocation.
- The framework utilizes a Calibration Module to elicit success probabilities and token estimates, which are then processed by an Auction Module to simulate procurement-based task routing.
- Experimental results demonstrate that while LLMs exhibit significant miscalibration, providing historical performance priors improves self-assessment and market-style coordination efficiency.

---

[OPTIMAS: An Intelligent Analytics-Informed Generative AI Framework for Performance Optimization](http://arxiv.org/abs/2604.23892)

- OPTIMAS: introduces a modular, multi-agent framework that automates HPC performance optimization by translating multi-source runtime diagnostics into actionable, evidence-driven code transformations.
- The framework utilizes a Profiling Agent, Analysis Agents, a Prompt Construction Agent, and an Evaluation Agent to create a closed-loop system that validates code correctness and performance gains.
- OPTIMAS employs Evidence-Aligned Reasoning (EAR) metrics to ensure that LLM-generated code edits are directly motivated by identified performance bottlenecks rather than generic code changes.

---

[ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems](http://arxiv.org/abs/2604.23878)

- ZenBrain: introduces a multi-layer memory architecture for AI agents that integrates fifteen neuroscience-inspired models into a unified system orchestrated by a MemoryCoordinator.
- The architecture utilizes seven distinct memory layers—working, short-term, episodic, semantic, procedural, core, and cross-context—to manage information lifecycle through consolidation, forgetting, and reconsolidation.
- ZenBrain incorporates six Predictive Memory Architecture components, including a NeuromodulatorEngine, ReconsolidationEngine, TripleCopyMemory, PriorityMap, StabilityProtector, and MetacognitiveMonitor, to govern memory dynamics and improve retrieval performance.

---

[ClawTrace: Cost-Aware Tracing for LLM Agent Skill Distillation](http://arxiv.org/abs/2604.23853)

- ClawTrace: introduces a cost-aware tracing platform that instruments LLM agent sessions via eight event hooks to generate compact TraceCard summaries for downstream distillation.
- CostCraft: utilizes TraceCards to distill agent trajectories into reusable skill patches, categorized as preserve, prune, or repair, to optimize agent performance and cost.
- The framework employs a three-way patch typology and conflict-aware merging to ensure that prune rules act as quality guardrails while repair rules address failure modes.

---


[DRACULA: Hunting for the Actions Users Want Deep Research Agents to Execute](http://arxiv.org/abs/2604.23815)

- DRACULA: introduces a large-scale dataset of user feedback on intermediate actions for Deep Research agents, comprising Action Generation Module, Action Selection Interface, MyScholarQA Agent, User Feedback Collector, Simulation Engine, and User History Memory.
- The framework enables the study of action predictability by leveraging User History Memory to improve the alignment of generated actions with user-specific preferences.
- The research demonstrates that while LLMs can reliably execute specified actions, predicting which intermediate actions users prefer remains a significant bottleneck that benefits from user-specific modeling.

---


[Scalable Production Scheduling: Linear Complexity via Unified Homogeneous Graphs](http://arxiv.org/abs/2604.23841)

- Unified Graph Framework: introduces a scalable RL approach for JSSP that utilizes feature-based homogenization to enable a homogeneous GIN backbone to process structurally heterogeneous graphs with linear complexity.
- The framework employs an actor-critic architecture trained via PPO, which leverages a structural saturation hypothesis to achieve zero-shot generalization across varying problem scales.
- By modeling machines as first-class entities in a sparse bipartite graph, the approach eliminates the quadratic edge complexity of traditional disjunctive formulations while maintaining strong relational inductive biases.

---

[JigsawRL: Assembling RL Pipelines for Efficient LLM Post-Training](http://arxiv.org/abs/2604.23838)

- JigsawRL: introduces a cost-efficient RL post-training framework that utilizes Sub-Stage Graph, Sub-Stage Multiplexing, Sub-Stage Merging, and a Look-ahead Heuristic to improve GPU utilization by multiplexing concurrent RL pipelines.
- The framework decomposes coarse-grained RL stages into fine-grained sub-stages to expose intra-stage and inter-worker imbalances, enabling dynamic resource allocation and sample migration across DP workers.
- JigsawRL achieves significant throughput improvements over synchronous and asynchronous baselines by co-scheduling complementary compute-bound and memory-bound sub-stages while maintaining moderate latency trade-offs.

---

[KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant](http://arxiv.org/abs/2604.23822)

- KISS Agent Framework: introduces a layered, single-concern agent architecture designed to address common LLM failure modes in software engineering through a strict inheritance hierarchy.
- The framework utilizes a structured system prompt and a five-layer hierarchy, including KISS Agent, Relentless Agent, Sorcar Agent, Chat Sorcar Agent, and Worktree Sorcar Agent, to ensure robust, budget-tracked, and isolated task execution.
- Implemented as a VS Code extension, the system achieves high performance on Terminal Bench 2.0 by prioritizing output quality through self-validation and disciplined engineering practices over latency.

---


[Structural Enforcement of Goal Integrity in AI Agents via Separation-of-Powers Architecture](http://arxiv.org/abs/2604.23646)

- PEA (Policy–Execution–Authorization) Architecture: introduces a separation-of-powers design that enforces AI safety at the system level by decoupling intent generation, authorization, and execution into independent layers.
- The framework utilizes an IVL, ILT, Goal Drift Detection, and an OSG to ensure that all executed actions remain traceable to the originating user request and bounded by authorized capability constraints.
- By treating the Policy LLM as untrusted, the architecture converts the AI safety problem from a probabilistic behavioral question into a conditionally sound system property with formally stated boundaries.

---

[DLM: Unified Decision Language Models for Offline Multi-Agent Sequential Decision Making](http://arxiv.org/abs/2604.23557)

- DLM: introduces a scalable framework that reformulates multi-agent sequential decision-making as a dialogue-style sequence prediction problem to bridge the gap between LLMs and decentralized decision tasks.
- The framework utilizes a two-stage training pipeline consisting of SFT for initial domain alignment and GRPO for preference-based optimization to enhance robustness against OOD actions.
- By converting observations and actions into natural language dialogues, DLM enables centralized training with inter-agent context while supporting decentralized execution from local observations.

---



[EndoGov: A knowledge-governed multi-agent expert system for endometrial cancer risk stratification](http://arxiv.org/abs/2604.23802)

- EndoGov: introduces a two-tier multi-agent expert system that decomposes the risk stratification process into independent evidence extraction by specialist agents and deterministic guideline-governed decision control by a chair agent.
- The framework utilizes a Guideline-KG to support both hard-path deterministic overrides for high-priority clinical triggers and soft-path grey-zone reasoning for ambiguous cases.
- By separating perception from governance, the system ensures guideline compliance and auditability, effectively mitigating logic blind spots in multimodal EC risk stratification.

---

[ClawMark: A Living-World Benchmark for Multi-Turn, Multi-Day, Multimodal Coworker Agents](http://arxiv.org/abs/2604.23781)

- ClawMark: introduces a benchmark for evaluating persistent coworker agents across multi-turn, multi-day workflows with evolving environments and raw multimodal evidence.
- The framework utilizes a stateful sandboxed service environment, including filesystem, email, calendar, knowledge base, and spreadsheet, to test agent adaptation to exogenous updates.
- Scoring is performed by a deterministic checker system that inspects post-turn service states, eliminating the need for LLM-as-judge protocols.

---

[PageGuide: Browser extension to assist users in navigating a webpage and locating information](http://arxiv.org/abs/2604.23772)

- PageGuide: introduces a browser extension that grounds LLM answers directly in the HTML DOM via visual overlays to improve user verifiability and control.
- The framework utilizes a Router to dispatch queries to specialized handlers for finding information, guiding multi-step tasks, or hiding distracting content.
- By coupling text outputs with in-situ DOM mutations and user-in-the-loop feedback, the system enables transparent, verifiable web interaction compared to opaque autonomous agents.

---

[Agentic Fusion of Large Atomic and Language Models to Accelerate Materials Discovery](http://arxiv.org/abs/2604.23758)

- ElementsClaw: introduces an agentic framework that synergizes Large Atomic Models with LLMs to autonomously orchestrate the materials discovery process.
- The system utilizes specialized LAM tools, including Elements-T, Elements-C, Elements-E, and Elements-G, to perform high-fidelity numerical computations while leveraging LLMs for semantic reasoning and literature-based synthesis.
- ElementsClaw demonstrates its efficacy by screening millions of crystals and successfully guiding the experimental synthesis of novel superconductors with high physical fidelity.

---

[Prism-Reranker: Beyond Relevance Scoring — Jointly Producing Contributions and Evidence for Agentic Retrieval](http://arxiv.org/abs/2604.23734)

- Prism-Reranker: introduces a reranker family that jointly emits a calibrated relevance score, a contribution statement, and a self-contained evidence passage in a single forward pass using a Qwen3.5 backbone.
- The framework utilizes a hybrid training objective combining point-wise distillation from a commercial rerank API with supervised fine-tuning on structured text targets gated by an LLM-as-Judge ensemble.
- Prism-Reranker optimizes agentic retrieval by providing actionable planning signals and context-compressed evidence, effectively reducing token consumption and hallucination risks for downstream LLMs.

---

[ESIA: An Energy-Based Spatiotemporal Interaction-Aware Framework for Pedestrian Intention Prediction](http://arxiv.org/abs/2604.23728)

- ESIA (Energy-based Spatiotemporal Interaction-Aware framework): introduces a structured CRF-based paradigm that models pedestrian intention prediction as an energy minimization problem by explicitly decoupling individual, social, and environmental factors.
- The framework utilizes PNFE and ENFE for feature extraction, while PPIL and PEIL capture complex interactions through MHA mechanisms to ensure global behavioral consistency.
- To resolve logical contradictions during inference, the model employs a U-SSA algorithm that leverages high-confidence unary priors to efficiently navigate the energy landscape and achieve robust, interpretable predictions.

---

[Information-Theoretic Measures in AI: A Practical Decision Guide](http://arxiv.org/abs/2604.23716)

- ITM Decision Framework: introduces a structured guide for selecting and applying seven information-theoretic measures across AI/ML and agent-based research domains.
- The framework categorizes measures into two families, distinguishing between core learning metrics and complex agent-level causal measures, while providing standardized estimator recommendations and guardrails.
- It operationalizes these guidelines through a measure-selection flowchart and a master decision table to prevent common misuses and ensure rigorous inferential claims in AI research.

---

[SPORE: Efficient and Training-Free Privacy Extraction Attack on LLMs via Inference-Time Hybrid Probing](http://arxiv.org/abs/2604.23711)

- SPORE: introduces a training-free privacy extraction attack that leverages adversarial input and inference-time hybrid probing to recover PII from LLM agent memory.
- The framework utilizes a shadow encryption paradigm to obfuscate PII, enabling efficient recovery in both black-box and gray-box settings through candidate space construction and token-level filtering.
- Experimental results demonstrate that SPORE achieves high attack success rates and low query costs across multiple frontier LLMs while remaining robust against existing safety alignment and detection mechanisms.

---

[Directional Alignment and Narrative Agency in Human–LLM Co-Writing](http://arxiv.org/abs/2604.23676)

- Human–LLM Co-Writing Framework: introduces a controlled dyadic storytelling task to quantify affective alignment and narrative agency through sentiment and information-theoretic modeling.
- The framework utilizes sentiment concept vector projection and surprisal-based metrics to evaluate how human and LLM agents influence narrative progression and emotional coordination.
- Empirical results demonstrate an asymmetric division of labor where humans drive narrative innovation while LLMs act as adaptive amplifiers that sustain coherence and emotional alignment.

---

[Vibe Medicine: Redefining Biomedical Research Through Human-AI Co-Work](http://arxiv.org/abs/2604.23674)

- Vibe Medicine: introduces a human-AI co-work paradigm where researchers direct skill-augmented AI agents to execute complex, multi-step biomedical workflows using LLMs, agentic frameworks, medical skills, and biomedical tools and data.
- The infrastructure relies on a modular architecture where specialized medical skills are composed into pipelines to perform tasks ranging from literature synthesis and variant interpretation to drug discovery and clinical trial design.
- The paradigm shifts the human role to that of a research director, emphasizing the need for human oversight to mitigate risks such as hallucination, data privacy concerns, and over-reliance on agent-generated outputs.

---

[Strategically Robust Aggregative Games](http://arxiv.org/abs/2604.23669)

- Strategically Robust Wardrop Equilibrium framework: introduces a novel equilibrium concept for aggregative games where agents protect themselves against worst-case aggregate behavior within an optimal-transport-based ambiguity set.
- The framework reformulates the infinite-dimensional robust optimization problem into a standard convex aggregative game using augmented action spaces and duality, enabling efficient computation via proximal best response algorithms.
- The research demonstrates a "coordination-via-robustification" effect in electric vehicle charging, where strategic robustness improves individual costs and can drive the price of anarchy to 1.

---

[GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs](http://arxiv.org/abs/2604.23626)

- GraphPlanner: introduces a heterogeneous graph memory-augmented agentic routing framework that formulates workflow generation as a Markov Decision Process to optimize multi-agent LLM collaboration.
- The framework utilizes GARNet to integrate current workflow memory and historical interaction traces, enabling adaptive routing decisions through a learned policy optimized by PPO.
- GraphPlanner supports both inductive and transductive inference, demonstrating robust generalization to unseen tasks and LLMs while maintaining computational efficiency.

---

[Thinking Like a Clinician: A Cognitive AI Agent for Clinical Diagnosis via Panoramic Profiling and Adversarial Debate](http://arxiv.org/abs/2604.23605)

- DxChain: introduces a cognitive-aligned reasoning framework that transforms clinical diagnosis into an iterative, stateful process by mirroring clinician trajectories through panoramic profiling, strategic navigation, and dialectical verification.
- The framework utilizes a Profile-Then-Plan paradigm to mitigate cold-start hallucinations, a Medical Tree-of-Thoughts (Med-ToT) algorithm for look-ahead planning, and an "Angel-Devil" adversarial debate mechanism to resolve evidence conflicts.
- Evaluated on MIMIC-IV-Ext datasets, DxChain achieves state-of-the-art performance in diagnostic accuracy and logical consistency by shifting from linear LLM inference to active, stateful clinical simulation.

---

[CineAGI: Character-Consistent Movie Creation through LLM-Orchestrated Multi-Modal Generation and Cross-Scene Integration](http://arxiv.org/abs/2604.23579)

- CineAGI: introduces a hierarchical movie generation framework that decomposes complex production tasks through specialized LLM-orchestrated multi-agent coordination.
- The framework utilizes a decoupled character-centric pipeline to maintain identity consistency across diverse scenes while enabling flexible multi-character composition.
- CineAGI achieves significant improvements in narrative coherence and character consistency by integrating specialized LLM agents for planning and targeted synthesis models for audiovisual alignment.

---

[MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation](http://arxiv.org/abs/2604.23539)

- MetaGAI: introduces a large-scale benchmark for automated Model and Data Card generation using a multi-agent framework comprising Retriever-, Generator- and Editor-agents.
- The framework utilizes multi-source triangulation of academic papers, GitHub repositories, and Hugging Face artifacts to synthesize high-fidelity documentation.
- Empirical analysis demonstrates that sparse Mixture-of-Experts architectures achieve superior cost-quality efficiency in generating structured documentation.

---

[Large Language Model based Interactive Decision-Making for Autonomous Driving](http://arxiv.org/abs/2604.23513)

- LLM-based Interactive Autonomous Driving Framework: introduces a framework that integrates Object-Process Methodology for semantic scene modeling with an LLM-driven decision module to improve interactive intelligence in mixed-traffic scenarios.
- The framework utilizes OPM to transform low-level perceptual data into structured object-process-relation representations, which serve as inputs for the LLM to perform intent-aware decision-making and trajectory optimization.
- The system closes the interaction loop by translating autonomous driving decisions into natural language messages broadcast via an external Human-Machine Interface, enhancing transparency and coordination with human road users.

---

[Breaking the Secret: Economic Interventions for Combating Collusion in Embodied Multi-Agent Systems](http://arxiv.org/abs/2604.23511)

- Mutagenic Incentive Intervention Framework: introduces a proactive defense mechanism that reshapes the payoff structure of LLM-based embodied agents to render collusion inherently unstable and economically irrational.
- The framework utilizes a reporting-and-penalty mechanism where agents are incentivized to defect from collusion by receiving rewards funded by the confiscated honesty deposits of identified colluders.
- To ensure robustness, the system incorporates cryptographic anonymity via ring signatures and automated, trustless fund management through smart contracts, effectively preventing retaliation and financial manipulation.

---

[Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines](http://arxiv.org/abs/2604.23483)

- Agentic Adversarial Rewriting framework: introduces a two-agent system that exploits black-box NLP pipelines through iterative semantic rewriting under strict query budgets.
- The framework utilizes an Attacker Agent and a Prompt Optimization Agent to navigate the semantic perturbation space without requiring gradient access or fine-tuning.
- The research identifies a vulnerability spectrum in multi-stage NLP pipelines, demonstrating that architectural properties like retrieval mechanisms significantly influence susceptibility to adversarial attacks.

---

[Towards Agentic Test-Driven Quality Assurance for 6G Networks](http://arxiv.org/abs/2604.23285)

- Agentic, Intent-Driven E2E Orchestration Framework: introduces an agentic orchestration architecture that integrates intent co-creation with a Test-Driven Quality Assurance paradigm to ensure proactive SLA compliance in 6G networks.
- The framework utilizes a dual-path agentic approach, including Intent-to-Actions- and Quality and SLO/SLA Specs-agents, to decompose high-level user intents into deterministic, standards-aligned technical specifications.
- By leveraging a TMF-aligned knowledge representation and MCP-enabled tool access, the system enables autonomous agents to perform graph-based reasoning and continuous validation of network services.

---

[OptimusKG: Unifying biomedical knowledge in a modern multimodal graph](http://arxiv.org/abs/2604.27269)

- OptimusKG: introduces a multimodal biomedical labeled property graph that integrates structured and semi-structured resources using a Medallion data lake architecture, Kedro workflow manager, BioCypher framework, PaperQA3 agent, and Apache Parquet storage.
- The framework enforces a unified top-level schema across 10 entity types and 26 relation types to preserve granular, type-specific metadata and provenance.
- Validation via the PaperQA3 agent demonstrates that the graph maintains high specificity, with 70.0% of sampled edges supported by scientific literature and 83.4% of false edges lacking evidence.

---

[Addressing the Reality Gap: A Three-Tension Framework for Agentic AI Adoption](http://arxiv.org/abs/2604.27245)

- Three-Tension Framework: introduces a diagnostic model for educational institutions to balance Implementation Feasibility, Adaptation Speed, and Mission Alignment when integrating agentic AI systems.
- The framework utilizes a modular architecture consisting of a Profile, Memory, Planning Module, Action Layer, and LLM to facilitate goal-oriented autonomy in educational settings.
- The paper emphasizes that successful AI adoption requires managing these three tensions as intersecting continuums rather than solving them as isolated technical problems.

---

[Think it, Run it: Autonomous ML pipeline generation via self-healing multi-agent AI](http://arxiv.org/abs/2604.27096)

- Autonomous ML pipeline generation framework: introduces a multi-agent architecture that automates end-to-end ML pipeline construction by integrating code-grounded semantic analysis, hybrid recommendation, and self-healing execution.
- The system utilizes a five-agent orchestration layer to transform natural language goals into validated, executable pipelines while addressing semantic, selection, and execution uncertainties.
- By shifting the source of truth from documentation to source code analysis, the framework achieves robust component discovery and continuous improvement through execution-based learning and LLM-driven self-healing.

---

#### 25th April 2026


[RAT: RunAnyThing via Fully Automated Environment Configuration](http://arxiv.org/abs/2604.23190)

- RAT (RunAnyThing): introduces a modular, language-agnostic framework for fully automated environment configuration that integrates Language-Agnostic Abstraction, ImageRetriever, Environment Configuration Planning, Specialized Toolset, Robust Sandbox Generation, and Long-term Expertise Accumulation.
- The framework employs an LLM-driven multi-stage pipeline to resolve complex repository dependencies and automate the construction of executable environments for autonomous code agents.
- To rigorously evaluate performance, the authors introduce RATBench, a large-scale multilingual benchmark comprising over 2,000 real-world repositories, demonstrating that RAT achieves state-of-the-art environment setup success rates.

---


[Escher-Loop: Mutual Evolution by Closed-Loop Self-Referential Optimization](http://arxiv.org/abs/2604.23472)

- Escher-Loop: introduces a closed-loop framework that operationalizes the mutual evolution of Task Agent Population and Optimizer Agent Population, where the latter recursively refines both task solutions and itself.
- The framework utilizes a Dynamic Benchmarking Mechanism that reuses empirical scores from Task Execution as relative win-loss signals to update the Elo Rating System of the Optimizer Agent Population without additional overhead.
- By maintaining a MAP-Elites Archive, the system preserves behavioral diversity and fitness, enabling the autonomous emergence of sophisticated optimization strategies that outperform static handcrafted baselines.

---

[Architecture Matters for Multi-Agent Security](http://arxiv.org/abs/2604.23459)

- Multi-Agent System (MAS) Architecture Framework: introduces an empirical study demonstrating that architectural design choices in multi-agent systems, such as role specialization, communication topology, and memory visibility, significantly impact security by fragmenting safety reasoning and expanding attack surfaces.
- The research reveals that multi-agent architectures are often more vulnerable than standalone LLMs, as task decomposition can dilute harmful signals and bypass safety training, even while maintaining or improving benign task performance.
- The study highlights that security-performance tradeoffs are highly scenario- and model-dependent, necessitating per-deployment adversarial evaluation rather than relying on component-level safety assessments.

---

[CUJBench: Benchmarking LLM-Agent on Cross-Modal Failure Diagnosis from Browser to Backend](http://arxiv.org/abs/2604.23455)

- CUJBench: introduces a diagnostic benchmark that couples browser-visible failure evidence with backend observability to evaluate LLM agents on cross-modal root cause analysis.
- The framework utilizes a snapshot-based methodology to ensure reproducibility, employing a multi-agent review loop to curate 87 deterministic failure scenarios across five fault families.
- Evaluation of six frontier LLMs reveals that cross-modal synthesis remains a primary bottleneck, as agents often retrieve decisive evidence but fail to correctly attribute it to the root cause.

---


[CODA: Coordination via On-Policy Diffusion for Multi-Agent Offline Reinforcement Learning](http://arxiv.org/abs/2604.23308)

- CODA (Coordination via On-Policy Diffusion for Multi-Agent Offline Reinforcement Learning): introduces a trajectory-level data augmentation method that uses diffusion models conditioned on the current joint policy to restore endogenous co-adaptation in offline MARL.
- The framework employs a centralized diffusion backbone to generate synthetic trajectories that are reweighted toward the current joint policy, effectively mitigating coordination failures caused by static offline datasets.
- CODA is algorithm-agnostic and can be integrated with various offline MARL pipelines to improve coordination by ensuring that synthetic experience reflects the evolving behaviors of agents during training.

---


[AI Safety Training Can be Clinically Harmful](http://arxiv.org/abs/2604.23445)

- Five-Axis Evaluation Framework: introduces a comprehensive clinical evaluation methodology for LLMs in mental health, identifying systematic failures where RLHF safety alignment disrupts essential therapeutic mechanisms.
- The framework operationalizes clinical safety through five distinct axes—protocol fidelity, hallucination risk, behavioral consistency, crisis safety, and demographic robustness—to address the gap between general-purpose conversational quality and clinical requirements.
- Empirical validation across Prolonged Exposure and Cognitive Behavioral Therapy demonstrates that current safety-aligned LLMs often exhibit a "crisis cliff," where performance collapses under high-severity scenarios due to inappropriate safety-motivated interventions.

---

[SoccerRef-Agents: Multi-Agent System for Automated Soccer Refereeing](http://arxiv.org/abs/2604.23392)

- SoccerRef-Agents: introduces a multi-agent framework that mimics professional officiating teams by integrating Video Agent, Rule Agent, Case Agent, Context Agent, and Chief Referee Agent to provide explainable soccer refereeing decisions.
- The system utilizes a cross-modal RAG mechanism to bridge visual perception with regulatory texts from RefKnowledgeDB, ensuring decisions are legally grounded and factually accurate.
- Evaluations on the SoccerRefBench benchmark demonstrate that the framework significantly outperforms general-purpose LLMs in both decision accuracy and the quality of legally grounded explanations.

---

[Ghost in the Agent: Redefining Information Flow Tracking for LLM Agents](http://arxiv.org/abs/2604.23374)

- NeuroTaint: introduces a provenance-oriented offline auditing framework for LLM agents that reconstructs execution lineage to detect explicit content propagation, implicit control influence, and asynchronous provenance reuse.
- The framework utilizes a Dynamic Context Provenance Graph (DCPG) to persist taint labels across memory boundaries and session restarts, enabling the attribution of delayed or transformed information flows.
- NeuroTaint employs a hybrid semantic tracker for explicit content detection and a sink-driven causal analyzer for implicit control influence, significantly outperforming traditional IFC-style baselines on the TaintBench benchmark.

---

[GSAR: Typed Grounding for Hallucination Detection and Recovery in Multi-Agent LLMs](http://arxiv.org/abs/2604.23366)

- GSAR (Grounding-Stratified Adaptive Replanning): introduces a grounding-evaluation and replanning framework for multi-agent LLMs that partitions claims into a four-way typology and couples evidence-typed weighted scoring to a three-tier decision function.
- The framework utilizes an asymmetric contradiction penalty to prevent score inflation and employs a bounded replanning loop to manage compute costs during incident investigation.
- Empirical evaluation across multiple LLM judges demonstrates that the framework's structural design choices, including evidence-type weighting and the complementary claim class, significantly improve grounding reliability and decision efficiency.

---

[LEGO: An LLM Skill-Based Front-End Design Generation Platform](http://arxiv.org/abs/2604.23355)

- LEGO: introduces a unified skill-based platform for digital front-end design that decomposes the workflow into six independent steps and utilizes a plug-and-play architecture for RTL generation.
- The framework leverages Circuit Skill Builder to automate the extraction of reusable skills from open-source projects and employs Agent Skill RAG for efficient, submillisecond retrieval of design and debugging knowledge.
- LEGO implements a three-layer hierarchical architecture comprising Templates, Step Skills, and atomic Circuit Skills to enable flexible, modular, and extensible RTL design automation.

---

[From Stateless Queries to Autonomous Actions: A Layered Security Framework for Agentic AI Systems](http://arxiv.org/abs/2604.23338)

- LASM (Layered Attack Surface Model): introduces a seven-layer security framework that maps agentic AI threats to specific architectural components including Foundation Model, Planning and Reasoning Module, Memory System, Tool Execution Layer, Multi-Agent Interface, Orchestration and Environment, and Governance.
- The framework incorporates an orthogonal attack temporality dimension (T1–T4) to categorize threats based on the installation-to-execution gap, highlighting that high-layer, slow-burn attacks are currently under-studied.
- The paper identifies five critical research gaps and proposes the Agent Bill of Materials (ABOM) to address supply-chain security and accountability in complex agentic systems.

---

[MMEB-V3: Measuring the Performance Gaps of Omni-Modality Embedding Models](http://arxiv.org/abs/2604.23321)

- MMEB-V3: introduces a comprehensive benchmark for evaluating full-modality embeddings across text, image, video, audio, and agent-centric scenarios using MMEB-V3, OmniSET, Audio Tasks, Text Tasks, Agent Tasks, and a Shared Candidate Pool.
- The framework utilizes OmniSET to disentangle semantic content from modality effects, enabling a systematic diagnostic analysis of instruction-conditioned embedding behavior.
- Experimental results reveal that current multimodal embeddings struggle to reliably enforce explicit modality constraints, often exhibiting significant cross-modal asymmetry and query-modality bias.

---

[Proteus: Shapeshifting Desktop Visualizations for Mobile via Multi-level Intelligent Adaptation](http://arxiv.org/abs/2604.23299)

- Proteus: introduces a multi-agent LLM-driven framework that automates the adaptation of desktop visualizations for mobile devices by systematically applying hierarchical transformation operators.
- The system utilizes a multi-level design space that propagates constraints from global topology to reference frames and individual visual elements to ensure semantic fidelity and readability.
- Proteus employs a collaborative team of specialized agents—including semantic parser-, data extraction-, design planner-, frontend engineering- and visual critic-agents—to iteratively refine visualizations for mobile consumption.

---

[Revisable by Design: A Theory of Streaming LLM Agent Execution](http://arxiv.org/abs/2604.23283)

- Revision Absorber: introduces a reactive algorithm that enables LLM agents to absorb mid-execution user revisions by identifying the earliest conflicting action and performing targeted compensation and re-planning.
- The framework formalizes agent execution as a stream of events and classifies actions into a reversibility taxonomy to determine the structural cost of adapting to user-injected changes.
- Experimental results on StreamBench demonstrate that the Revision Absorber achieves quality comparable to full-restart baselines while significantly reducing wasted computational steps.

---

[Bridging the Pose-Semantic Gap: A Cascade Framework for Text-Based Person Anomaly Search](http://arxiv.org/abs/2604.23282)

- SSDC (Structure-Semantic Decoupled Cascade): introduces a coarse-to-fine framework that bridges the Pose-Semantic Gap in text-based person anomaly search by decoupling structural filtering from multi-agent semantic verification.
- The framework utilizes a Structure-Aware Coarse Retriever for high-speed candidate filtering, followed by a Detective Squad that employs a Detective Agent, an Analyst Agent, and a Writer Agent to perform iterative, fine-grained semantic reasoning.
- By integrating an Adaptive Fusion Mechanism, the system balances retrieval efficiency with semantic precision, achieving state-of-the-art performance on the PAB benchmark.

---

[AI Identity: Standards, Gaps, and Research Directions for AI Agents](http://arxiv.org/abs/2604.23280)

- AI Identity Framework: introduces a three-layer model for AI agent identity, comprising a Declaration Layer, an Observation Layer, and a Confidence Layer, to manage the continuous relationship between agent claims and actual behavior.
- The framework addresses the structural asymmetry between human and AI identity by treating identity as a probabilistic, time-varying estimate rather than a static, binary credential.
- This report identifies five critical structural gaps in current identity infrastructure, including semantic intent verification and recursive delegation accountability, and proposes a research agenda to achieve ecologically sustainable and secure agent governance.

---

[Active Inference: A Method for Phenotyping Agency in AI Systems?](http://arxiv.org/abs/2604.23278)

- Active Inference framework: introduces a computational method for phenotyping AI agency by mapping intentionality, rationality, and explainability to the components of a variational generative model.
- The paper utilizes empowerment as an operational metric to distinguish between zero-, intermediate-, and high-agency phenotypes within a T-maze decision-making paradigm.
- The authors argue that as AI agents increase in agency, governance strategies must transition from external structural constraints to internal modulation of prior preferences.

---

[CAP-CoT: Cycle Adversarial Prompt for Improving Chain of Thoughts in LLM Reasoning](http://arxiv.org/abs/2604.23270)

- CAP-CoT: introduces a cycle-based adversarial prompt optimization framework that strengthens LLM reasoning through adaptive contrast between correct and erroneous chains.
- The framework utilizes a Solver Agent, an Adversarial Challenger Agent, and a Feedback Agent to iteratively refine prompts, improving both reasoning accuracy and robustness.
- By generating targeted hard negatives and providing step-aligned feedback, CAP-CoT systematically discovers and repairs reasoning vulnerabilities while maintaining a single-model inference setup.

---

[PrivacyAssist: A User-Centric Agent Framework for Detecting Privacy Inconsistencies in Android Apps](http://arxiv.org/abs/2604.23248)

- PrivacyAssist: introduces a multi-agent platform that detects inconsistencies between runtime-granted permissions and declared data practices in Android apps using Agent-1, Agent-2, Kafka, MongoDB, RAG, VectorDB, Summarization Module, and Llama-3-8B LLM.
- The framework employs a client-server architecture where Agent-1 monitors app permissions on-device, while Agent-2 performs server-side analysis using RAG to provide concise, user-oriented privacy warnings.
- PrivacyAssist mitigates LLM hallucinations and token constraints by utilizing a summarization module and an external database to ground its reasoning in verified Android permission and data safety definitions.

---

[Discovering Agentic Safety Specifications from 1-Bit Danger Signals](http://arxiv.org/abs/2604.23210)

- EPO-Safe (Experiential Prompt Optimization for Safe Agents): introduces an iterative framework where an LLM agent discovers hidden safety objectives from sparse 1-bit danger signals by evolving a natural language specification through reflection.
- The framework utilizes a four-phase experiential loop—Attempt, Simulate, Reflect, and Consolidate—to transform binary feedback into human-readable, auditable behavioral rules without requiring gradient-based model updates.
- By decoupling safety reflection from reward optimization, EPO-Safe prevents reward hacking and enables LLMs to perform few-shot safety rule induction in complex environments.

---

[StoryTR: Narrative-Centric Video Temporal Retrieval with Theory of Mind Reasoning](http://arxiv.org/abs/2604.23198)

- StoryTR: introduces a benchmark and an Agentic Data Pipeline that enables LLMs to perform narrative-centric video retrieval by distilling Theory of Mind reasoning capabilities into smaller models.
- The framework utilizes a Clipper Agent for fine-grained multimodal perception and a Self-QA Agent to synthesize training data with three-tier ToM reasoning chains, including intent decoding, narrative reasoning, and boundary localization.
- Experimental results demonstrate that the 7B Shorts-Moment model, trained on ToM-guided data, significantly outperforms larger baselines, validating that cognitive depth in reasoning is more critical than parameter scale for narrative understanding.

---

[AnalogRetriever: Learning Cross-Modal Representations for Analog Circuit Retrieval](http://arxiv.org/abs/2604.23195)

- AnalogRetriever: introduces a unified tri-modal retrieval framework that maps functional text descriptions, schematic images, and SPICE netlists into a shared semantic embedding space using modality-specific encoders and curriculum contrastive learning.
- The framework utilizes a port-aware Relational Graph Convolutional Network (RGCN) to capture structural semantics of SPICE netlists, enabling precise cross-modal retrieval across text, schematics, and code.
- Integrated into the AnalogCoder agentic framework, AnalogRetriever improves LLM functional correctness by providing topologically accurate circuit references via retrieval-augmented generation.

---

[From Coarse to Fine: Self-Adaptive Hierarchical Planning for LLM Agents](http://arxiv.org/abs/2604.23194)

- AdaPlan-H: introduces a self-adaptive hierarchical planning mechanism that mimics human cognitive strategies by generating plans with varying granularity based on task complexity.
- The framework utilizes a two-stage optimization process, incorporating imitation learning for initialization and DPO training to refine hierarchical plan levels and quality.
- By dynamically adjusting planning granularity, the approach improves task execution success rates and efficiency while mitigating overplanning in LLM-based agents.

---

[Cooperative Informative Sensing for Monitoring Dynamic Indoor Environments via Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2604.23179)

- MARL framework: introduces a decentralized multi-robot monitoring system that optimizes motion to improve human-centric monitoring accuracy under partial observability using Set-based Observation Encoding, Dual-Stage Recurrent Interaction Memory, and a Centralized Critic.
- The architecture utilizes permutation-invariant attention to handle variable-sized human observations and a dual-stage GRU structure to enable scalable inter-robot coordination without explicit tracking.
- The approach demonstrates robust performance across diverse indoor monitoring tasks and supports cost-effective hybrid integration with existing fixed sensing infrastructure.

---

[PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks](http://arxiv.org/abs/2604.23148)

- PhySE: introduces a real-time AR-LLM social engineering framework that integrates VLM-based social-context training with an adaptive psychological agent to enable theory-grounded strategy control.
- The framework utilizes a VLM-based social-context training component to minimize cold-start latency and an adaptive psychological agent to dynamically route interaction strategies based on latent trust states.
- PhySE improves conversational realism and attack effectiveness by replacing static scripts with a theory-grounded routing mechanism that adjusts to turn-level interaction signals.

---

[UNSEEN: A Cross-Stack LLM Unlearning Defense against AR-LLM Social Engineering Attacks](http://arxiv.org/abs/2604.23141)

- UNSEEN (A Cross-Stack LLM Unlearning Defense): introduces a coordinated defense architecture that integrates AR ACL, F-RMU, and Agent Guardrails to mitigate AR-LLM-based social engineering attacks.
- The framework employs F-RMU to perform targeted unlearning of sensitive identity concepts within the Multimodal LLM while preserving general utility.
- UNSEEN provides a cross-stack security pipeline that constrains sensing, inference, and interaction to prevent unauthorized profile extraction and persuasive phishing.

---

[GreenDyGNN: Runtime-Adaptive Energy-Efficient Communication for Distributed GNN Training](http://arxiv.org/abs/2604.23139)

- GreenDyGNN: introduces a runtime-adaptive framework for distributed GNN training that optimizes energy efficiency by dynamically adjusting cache rebuild windows and per-owner cache allocations using a reinforcement learning agent.
- The system utilizes an asynchronous double-buffered pipeline to decouple cache management from the training loop, ensuring that runtime adaptation occurs without stalling GPU computation.
- By employing a Double-DQN agent trained via sim-to-real transfer with domain randomization, the framework effectively mitigates energy waste caused by network congestion and remote feature fetch latencies.

---

[No Test Cases, No Problem: Distillation-Driven Code Generation for Scientific Workflows](http://arxiv.org/abs/2604.23106)

- MOSAIC (Multi-agent framework for scientific code generation): introduces a training-free, multi-agent framework that enables scientific code generation without I/O supervision by utilizing a Bucketing Module, Teacher Module, Self-Reflection Agent, Student Module, Rationale Agent, Consolidated Context Window (CCW), Coding Agent, and Debugger Agent.
- The framework employs a student-teacher knowledge distillation process to ground code generation in domain-specific rationales, effectively decoupling semantic grounding from syntactic grounding.
- By implementing a Consolidated Context Window (CCW), the framework maintains reasoning coherence across interdependent subproblems while mitigating hallucinations in LLMs.

---

[From Language to Logic: Bridging LLMs &amp; Formal Representations for RTL Assertion Generation](http://arxiv.org/abs/2604.23100)

- ProofLoop: introduces a tool-augmented ReAct agent that automates SystemVerilog Assertion generation by integrating AST-based retrieval and iterative formal verification feedback.
- The framework utilizes an AST Parser, Embedding Model, and VectorDB to synthesize design context, while the LLM employs a ReAct agent to interact with JasperGold for structural analysis and solver-in-the-loop refinement.
- By leveraging formal proof feedback to iteratively correct syntax and functional errors, the system achieves high correctness rates on complex, multi-module hardware designs.

---

[Code Broker: A Multi-Agent System for Automated Code Quality Assessment](http://arxiv.org/abs/2604.23088)

- Code Broker: introduces a hierarchical multi-agent system that leverages LLMs and static analysis to automate code quality assessment across correctness, security, style, and maintainability dimensions.
- The architecture utilizes a root Report Generator to coordinate a Sequential Pipeline Agent, which dispatches parallel Correctness-, Style-, and Description-agents before an Improvement Recommender synthesizes the final output.
- The system integrates Pylint as a deterministic tool to ground LLM-based reasoning, while employing asynchronous execution and lightweight session memory to enhance robustness and context retention.

---

[Usable Agent Discovery for Decentralized AI Systems](http://arxiv.org/abs/2604.23080)

- Decentralized Agent Discovery Framework: introduces a model for evaluating agent discovery in distributed systems by accounting for both node-level churn and agent-level lifecycle dynamics.
- The framework compares structured (Kademlia) and unstructured (Cyclon+Vicinity) overlays across four distinct operating regimes to determine their impact on routing efficiency, resilience, and service readiness.
- It utilizes a useful availability metric to assess whether discovered agents can provide services within specific latency constraints, revealing that structured and unstructured approaches occupy different performance regimes.

---

[ADEMA: A Knowledge-State Orchestration Architecture for Long-Horizon Knowledge Synthesis with LLM Agents](http://arxiv.org/abs/2604.25849)

- ADEMA: introduces a knowledge-state orchestration architecture that replaces unstructured multi-agent dialogue with explicit epistemic bookkeeping, heterogeneous dual-evaluator governance, adaptive task-mode switching, dynamic reputation-based budget allocation, checkpoint-resumable persistence, segment-level memory condensation, artifact-first assembly, and final-validity checking with safe fallback.
- The architecture improves long-horizon task reliability by externalizing hypothesis states, milestone status, and artifact lineage as explicit system variables rather than relying on implicit message accumulation.
- Empirical results demonstrate that checkpoint persistence is the primary mechanism for maintaining recoverable continuity under interruption, while the architecture provides a structured, inspectable, and governable framework for complex knowledge synthesis tasks.

---

[Semantic Denial of Service in LLM-Controlled Robots](http://arxiv.org/abs/2604.24790)

- SDoS (Semantic Denial of Service): introduces an availability attack surface where injected safety-plausible phrases trigger an LLM's safety reasoning to halt or disrupt robot execution.
- The framework demonstrates that prompt-level defenses fail to distinguish between legitimate hazard alerts and adversarial injections, often causing the model to redirect hard stops into acknowledge loops or false alerts.
- The research identifies that LLMs perform implicit signal corroboration, where diverse safety cues are aggregated as evidence, making multi-phrase injection significantly more effective than simple repetition.

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

