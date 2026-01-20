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

[2026 (1/1)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2025 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_4.md),[2025 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_3.md), [2025 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (1/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_01.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)

Chronological order. 





</div>








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

---

