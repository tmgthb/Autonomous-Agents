<!--Autonomous Agents -->
<!--
Copyright (C) Teemu Maatta. 

@misc{MaattaAutonomousAgents2023,
  author = {Teemu Maatta},
  title = {Autonomous Agents},
  year = {2023},
  howpublished = {\url{https://github.com/tmgthb/Autonomous-Agents}},
  note = {Accessed: YYYY-MM-DD}
}
-->
<div id="topofthepage"> </div>

<div align="center">

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ftmgthb%2FAutonomous-Agents&count_bg=%23F2C027&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Views&edge_flat=true)](https://github.com/tmgthb/Autonomous-Agents)
[![X](https://img.shields.io/twitter/follow/Teemumtt3?style=social)](https://twitter.com/Teemumtt3)
[![GitHub Repo stars](https://img.shields.io/github/stars/tmgthb/Autonomous-Agents?style=flat-square)](https://github.com/tmgthb/Autonomous-Agents/stargazers)

</div>

<p align="center">
  <img height="100" src="https://github.com/tmgthb/Autonomous-Agents/blob/main/Autonomous_agent_logo.png" alt="Autonomous Agents">
</p>

<div align="center">

  # Autonomous Agents
  Autonomous Agents-research papers. Updated daily. See as well the [Resources](https://github.com/tmgthb/Autonomous-Agents/blob/main/Autonomous_Agents_Resources.md)-section.

</div>


---

<div id="researchpapers" align="center">

## Research papers


Chronological order.

</div>

#### 7th February 2025


[Sirius: Self-improving Multi-agent Systems via Bootstrapped Reasoning](http://arxiv.org/abs/2502.04780v1)

- SIRIUS: introduces a self-improving multi-agent system framework, utilizing Physicist, Mathematician, Summarizer agents, Experience Library, Experience Augmentation, and Fine-tuning for optimizing multi-agent systems.
- SIRIUS constructs an experience library by retaining successful reasoning trajectories to provide training data for agent policy fine-tuning.
- SIRIUS further enriches the library by augmenting unsuccessful trajectories, enhancing data diversity and improving system performance.


---


[MELON: Indirect Prompt Injection Defense via Masked Re-execution and Tool Comparison](http://arxiv.org/abs/2502.05174v1)

- MELON (Masked re-Execution and TooL comparisON) introduces an indirect prompt injection defense framework, with Agent System, Tool Execution, Tool Call Cache, Compare Tool Calls, and Masking Function, that detects attacks by comparing tool calls between original and masked executions.
- MELON framework leverages Masking Function to generate task-neutral prompts for masked re-execution, utilizing Tool Call Cache to store masked run tool calls and Compare Tool Calls to identify deviations indicating potential attacks.
- MELON framework enhances security and utility balance by focusing on tool call comparison and incorporating designs like customized masking, tool call caching, and focused comparison to reduce false positives and negatives in indirect prompt injection detection.


---


[NVAGENT: Automated Data Visualization from Natural Language via Collaborative Agent Workflow](http://arxiv.org/abs/2502.05036v1)

- NVAGENT: introduces collaborative agent workflow for NL2VIS, with processor (database processing and context filtering), composer (planning visualization generation), and validator (code translation and output verification).
- NVAGENT decomposes visualization generation into manageable subtasks using processor for data preparation, composer for VQL generation, and validator for ensuring correctness.
- NVAGENT leverages divide-and-conquer strategy with specialized agents to effectively handle complex NL2VIS tasks, improving visualization accuracy and quality.


---

[The Rising Threat to Emerging AI-Powered Search Engines](http://arxiv.org/abs/2502.04951v1)

- Agent-based Defense: introduces agent-based defense with Observation, Thought, Action, Tools, and Agent-based Defense components, where agent-based defense mitigates risks in AIPSE outputs.
- Agent-based defense framework uses Observation to gather AIPSE output, Thought for reasoning, Action to use Tools like Content Refinement and URL Detector.
- Agent-based defense aims to filter and mark potential risks in AIPSE output while preserving response similarity.


---

[S<sup>2</sup>.-MAD: Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency](http://arxiv.org/abs/2502.04790v1)

- S2-MAD (Selective Sparse Multi-Agent Debate): introduces Initial Response Generation, Grouping Discussion with Decision-Making Mechanism, and Reaching Consensus, with Agents organized in Groups, to enhance multi-agent debate efficiency.
- S2-MAD framework employs Decision-Making Mechanism comprising Similarity Calculation, Redundant Information Filtering, and Conditional Participation modules to manage agent engagement.
- S2-MAD framework aims to reduce token costs in multi-agent debate by selectively incorporating non-redundant viewpoints and optimizing information exchange among agents.


---


[Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research](http://arxiv.org/abs/2502.04644v1)

- Agentic Reasoning: introduces a framework enhancing large language model reasoning by integrating Web-Search Agent, Coding Agent, and Mind Map Agent, to solve complex problems requiring research and logical deduction.
- Agentic Reasoning framework utilizes Mind Map Agent for structured knowledge graph construction, Web-Search Agent for real-time information retrieval, and Coding Agent for computational analysis, improving reasoning and decision-making.
- Agentic Reasoning enables language models to perform multi-step strategies and tackle complex problems by dynamically adapting to information and performing quantitative analyses using external agents and structured memory.


---

[Self-Regulation and Requesting Interventions](http://arxiv.org/abs/2502.04576v1)

- Offline PRM-Tabular RL Framework: introduces an offline approach for training intervention-requesting agents, with State Dynamics, PRM, Tabular RL, Usage computation, Policy computation, Reward Search, and SFT Helper components.
- Offline PRM-Tabular RL Framework combines LLM-based Process Reward Models with tabular reinforcement learning to efficiently determine optimal intervention timing under budget constraints.
- This framework reduces costly intervention calls during training by leveraging offline data and enhancing robustness through PRMs and tabular RL, avoiding deep RL inefficiencies.


---
[Every Software as an Agent: Blueprint and Case Study](http://arxiv.org/abs/2502.04747v1)

- JiT-Codegen: introduces software agent framework for in-software execution using LLM-powered Agent, JiT Code Agent, Software Runtime, and Exec. Sandbox.
- JiT-Codegen framework enables LLMs access to software internals and inject code for execution within a secure Exec. Sandbox.
- This approach aims to overcome limitations of API-based and GUI-based agents by enabling more direct and efficient software interaction.


---

[STRIDE: Automating Reward Design, Deep Reinforcement Learning Training and Feedback Optimization in Humanoid Robotics Locomotion](http://arxiv.org/abs/2502.04692v1)

- STRIDE (Automating Reward Design, Deep Reinforcement Learning Training and Feedback Optimization in Humanoid Robotics Locomotion): introduces framework integrating Humanoid Robotics Environment, Motion Task Description, LLMs, Reward Function Sampling, Reward Functions, Reward Reflection, DRL Training, and Feedback Result for automated reward design.
- STRIDE leverages LLMs for zero-shot reward function generation and iterative refinement through feedback from DRL training outcomes.
- STRIDE automates reward engineering for humanoid robot locomotion, overcoming limitations of manual reward design and improving task performance.


---


[Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization](http://arxiv.org/abs/2502.04686v1)

- LSPO (Latent Space Policy Optimization): introduces iterative framework with Latent Space Construction (create discrete strategy space), Policy Optimization in Latent Space (optimize strategy in latent space), and Latent Space Expansion (expand strategy space coverage) for strategic language agents.
- LSPO framework addresses challenges in free-form language games by mapping text to latent space, optimizing policy with game-theoretic methods, and expanding space via LLM fine-tuning.
- Iterative process of LSPO enhances strategic reasoning and language communication, improving agent performance in complex games like Werewolf.


---


#### 6th February 2025


[VTutor: An Open-Source SDK for Generative AI-Powered Animated Pedagogical Agents with Multi-Media Output](http://arxiv.org/abs/2502.04103v1)

- VTutor (Software Development Kit): introduces an open-source framework for creating animated pedagogical agents, integrating Generative AI (LLMs), Text-to-Speech (TTS), Lip Synchronization (LipSync), Character Model, WebGL Rendering, Web Interface, API Communication, SDK, Iframe Integration, and React SDK.
- VTutor combines generative AI with animation technologies to enable personalized learning experiences through adaptable, realistic animated pedagogical agents with multi-media output in web environments.
- The framework leverages LLMs for context-aware feedback, uLipSync for accurate lip movements, and WebGL for seamless web integration, offering tools for developers to create engaging educational agents.


---


[PsyPlay: Personality-Infused Role-Playing Conversational Agents](http://arxiv.org/abs/2502.03821v1)

- PsyPlay: introduces dialogue generation framework with Role Card Creation (generates agent roles), Topic Extraction (extracts dialogue topics), and Dialogue Generation (creates personality-infused dialogues) for personality-infused role-playing.
- PsyPlay framework facilitates expression of rich personalities among multiple LLM agents assuming distinct roles and engaging in discussions.
- PsyPlay validation demonstrates accurate portrayal of intended personality traits with high success rate on generated dialogue data.


---


[Large Language Models for Multi-Robot Systems: A Survey](http://arxiv.org/abs/2502.03814v1)

- BOLAA (Benchmarking and Orchestrating LLM-augmented Autonomous Agents) architecture orchestrates multiple LAAs using Agents Message Controller, which manages communication between Environment, Labor Agents Pool consisting of multiple agents LAA 1, LAA 2, LAA m, each having LLM and Agent Prompt, Action Parser, Memory and Agents Selection.
- BOLAA architecture employs central controller for message distribution to individual agents with own LLMs, processing distributed messages to generate actions, improving consistency and reliability in collaborative systems.
- BOLAA architecture serves as comparative framework for LLM-augmented agents, offering insights into LLM integration and agent orchestration for multi-robot applications, despite focus on multi-agent systems rather than exclusively MRS.


---

[Division-of-Thoughts: Harnessing Hybrid Language Model Synergy for Efficient On-Device Agents](http://arxiv.org/abs/2502.04392v1)

- DoT (Division-of-Thoughts): introduces collaborative reasoning framework, with Task Decomposer, Task Scheduler, Task Allocation, Plug-and-Play Adapter, SLM, LLM, and Self-Reinforced Tree Search, for efficient on-device agents using hybrid language model synergy.
- DoT framework employs Task Decomposer for breaking down queries, Task Scheduler for dependency analysis between sub-tasks, and Plug-and-Play Adapter for dynamic allocation of sub-tasks between SLM and LLM.
- Self-Reinforced Tree Search method trains Plug-and-Play Adapter using task execution feedback to optimize sub-task allocation strategy for enhanced efficiency and maintained accuracy.


--

[Multi-Agent Reinforcement Learning with Focal Diversity Optimization](http://arxiv.org/abs/2502.04492v1)

- MARL-Focal (Multi-Agent Reinforcement Learning with Focal Diversity Optimization): introduces a two-stage framework with Decider Agent (selects optimal LLM subset) and Aggregator Agent (synthesizes final output) to improve LLM performance by leveraging a Pool of LLMs in Cloud (collection of available models) and diversity metrics.
- MARL-Focal framework utilizes Decider Agent (selects optimal LLM subset) within Multi Agent Environment (manages agent interactions) to choose Chosen LLMs (selected ensemble subset) based on Perf. Metrics (diversity-based selection metrics) from Incoming Online Queries (user input queries) and generate Model Outputs (LLM generated responses) for final aggregation.
- The framework's architecture allows for adaptive ensemble creation by dynamically selecting and combining diverse LLMs, aiming to enhance output quality and robustness while maintaining cost-efficiency in multi-agent learning scenarios.


---

[ACTIVE TASK DISAMBIGUATION WITH LLMS](http://arxiv.org/abs/2502.04485v1)

- Active Task Disambiguation: introduces method for LLM agents to actively clarify ambiguous tasks by iteratively asking questions, using solution generator, question generator, and question evaluator to refine problem statement and solution space.
- It leverages Bayesian Experimental Design principles to select questions maximizing information gain, shifting reasoning from implicit to explicit solution space exploration.
- This approach improves task disambiguation compared to methods relying solely on implicit reasoning within the question space, enhancing LLM agents' ability to address underspecified problems.


---

[ScoreFlow: Mastering LLM Agent Workflows via Score-based Preference Optimization](http://arxiv.org/abs/2502.04306v1)

- ScoreFlow: introduces automated workflow generation framework, with LLM Generator (workflow code generator), Executor (workflow performance evaluator), Collect data (score gathering component), Scores (workflow performance metrics), Preference workflow pairs (score-based workflow rankings), Iterative Score-DPO (score-aware optimization algorithm), Operators (reusable agent node combinations), Workflows (code for agent interactions), Problem dataset (input task collection).
- ScoreFlow framework utilizes Score-DPO optimization, incorporating quantitative evaluation feedback for workflow generation.
- ScoreFlow enhances scalability and performance through gradient-based optimization and code-based workflow representation.


---


[Multi-agent Architecture Search via Agentic Supernet](http://arxiv.org/abs/2502.04180v1)

- MaAS (Multi-agent Architecture Search): introduces agentic supernet, probabilistic architecture distribution, with controller, agentic operators, environment, and feedback, where MaAS optimizes distribution of agentic architectures for query-dependent multi-agent systems.
- MaAS framework leverages controller network to sample task-specific multi-agent systems from agentic supernet, adapting architecture based on environmental feedback and agentic operators.
- Agentic supernet in MaAS enables efficient resource allocation by dynamically adjusting multi-agent system complexity based on query difficulty and domain, utilizing feedback for continuous improvement.


---



#### 4th February 2025

[Adaptive Self-improvement LLM Agentic System for ML Library Development](http://arxiv.org/abs/2502.02534v1)

- Adaptive self-improvement LLM agentic system: introduces agentic system organization with parallel sampling to enhance LLM Agents via select multi-level experiences, stratify them by difficulty, filter high-quality answers, and use demonstrations for ML library development Task, verified by Verifier to produce Answer.
- This system employs adaptive self-improvement learning algorithm that filters quality answers, stratifies experiences by difficulty, and selects demonstrations to improve LLM agents' performance in generating architecture-specific programming language code.
- The framework addresses challenges in ML library development by enabling complex reasoning with limited data through a self-improvement cycle where LLM agents evolve via earned experiences and generate high-quality ML operators.


---

[AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinement](http://arxiv.org/abs/2502.02067v1)

- AdaptBot: introduces framework integrating LLM, Knowledge Graph, Human Input, Execution, and Decision Module, for task decomposition and knowledge refinement.
- AdaptBot utilizes LLM for generating initial abstract action plans, Knowledge Graph for domain-aware refinement, and human input for error correction and knowledge expansion.
- AdaptBot framework facilitates adaptation to new tasks through incremental knowledge refinement via human feedback and Knowledge Graph-guided error resolution.


---

[Anticipate & Act : Integrating LLMs and Classical Planning for Efficient Task Execution in Household Environments](http://arxiv.org/abs/2502.02066v1)

- Anticipate & Act: introduces a framework for efficient task execution, integrating User, LLM Prompting, LLM, Mapping, Planning, FASTDOWNWARD PLANNER, GENERATED PLAN, and SIMULATION components.
- Anticipate & Act: leverages LLM to predict high-level tasks from User prompts and uses FASTDOWNWARD PLANNER to generate fine-grained action sequences via Planning and Mapping components.
- Anticipate & Act: demonstrates efficiency in household tasks by anticipating future tasks and planning actions jointly within SIMULATION environment, reducing execution time and plan length.


---

[CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning](https://arxiv.org/abs/2502.02390)

- CoAT (Chain-of-Associated-Thoughts): introduces a reasoning framework for large language models that combines an optimized Monte Carlo Tree Search (MCTS) with a dynamic associative memory mechanism, integrating Target LLM, Associative Memories, Nodes, optional External Brain, Knowledge Graph, Vector Database, LLM agents, Internet Access, and Evaluator.
- The framework expands the reasoning search space and adaptively incorporates new information, mimicking human-like associative thinking during inference.
- Optimized MCTS algorithm systematically integrates associative content and generated content through tree node search, and flexible mechanism sources associative content by self-association or external knowledge retrieval.


---

#### 3rd February 2025

[Improving Transformer World Models for Data-Efficient RL](https://arxiv.org/abs/2502.01591)

- Improved TWM for Data-Efficient RL: introduces MBRL framework with MFRL Baseline, MBRL Baseline, Dyna with Warmup, Nearest Neighbor Tokenizer, and Block Teacher Forcing for enhanced data efficiency in reinforcement learning.
- The framework combines model-free and model-based RL with novel tokenization and training techniques to achieve state-of-the-art performance in the Craftax-classic environment.
- Key improvements include Dyna with Warmup for hybrid real-imaginary training, Nearest Neighbor Tokenizer for efficient image encoding, and Block Teacher Forcing for improved TWM training and rollout accuracy.


---

[PROCESS REINFORCEMENT THROUGH IMPLICIT REWARDS](https://arxiv.org/abs/2502.01456)

- PRIME (Process Reinforcement through IMplicit rEwards): introduces scalable online reinforcement learning framework with dense token-level rewards with Policy Model, Implicit PRM, SFT Model, Outcome Verifier, and Reference Model.
- PRIME framework updates Implicit PRM online using policy rollouts and outcome labels, removing dedicated reward model training phase.
- PRIME utilizes Implicit PRM for token-level rewards generation, mitigating reward hacking and enhancing sample efficiency in reinforcement learning for LLMs.


---

[TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues](https://arxiv.org/abs/2502.01630)

- TReMu (Temporal Reasoning for LLM-Agents in Multi-Session Dialogues): introduces a framework with Time-aware Memorization Model (summarizes dialogue sessions with dates), Memory Retrieval Model (retrieves relevant memory for question), Neuro-symbolic Reasoning Model (generates Python code for reasoning), and Python Executor (executes generated Python code) to enhance temporal reasoning in multi-session dialogues.
- It employs timeline summarization for memory and neuro-symbolic reasoning using LLMs to generate and execute Python code for temporal calculations.
- This approach improves temporal reasoning performance by leveraging Python's libraries for temporal calculations and step-by-step code execution.


---

[Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/abs/2502.01600)

- LOOP: introduces reinforcement learning framework for training interactive digital agents, utilizing hidden state, task context, agent output, and environment output for long-horizon tasks.
- This framework uses partially observable Markov decision process to formalize agent-environment interactions via read-eval-print loop.
- LOOP framework enhances sample efficiency and memory efficiency by reusing off-policy samples and maintaining single LLM copy.


---

[Memento No More: Coaching AI Agents to Master Multiple Tasks via Hints Internalization](https://arxiv.org/abs/2502.01562)

- MNM (Memento No More): introduces an iterative coaching process with Initial Agent, Human Analyst, Hints, Teacher Agent, Training Data, Student Agent, and Task Trajectories, where human feedback guides an AI agent to master multiple tasks.
- The framework refines agent behavior through iterative rounds of mistake analysis and hint internalization, improving task execution without extensive prompts.
- MNM leverages context distillation to transfer hint knowledge into agent weights, enhancing generalization and reducing reliance on prompt-based guidance.


---


[THE IN-CONTEXT INDUCTIVE BIASES OF VISION-LANGUAGE MODELS DIFFER ACROSS MODALITIES](https://arxiv.org/abs/2502.01530)

- Framework name here: introduces vision-language models, with vision input (processing visual information) and text input (processing textual information), to study generalization process (inferring category from examples) across modalities.
- It highlights that inductive biases in vision-language models differ significantly based on whether input is visual or textual, affecting generalization.
- Furthermore, the study reveals that in textual input, the order of feature descriptors influences the model's generalization, indicating sensitivity to linguistic structure.


---

[SHARPIE: A Modular Framework for Reinforcement Learning and Human-AI Interaction Experiments](http://arxiv.org/abs/2501.19245v2)

- SHARPIE (Shared Human-AI Reinforcement Learning Platform for Interactive Experiments): introduces a modular framework for human-AI interaction experiments, featuring versatile wrapper for RL components, participant web interface, experiment configuration, logging, deployment utilities, and multi-modal communication channels.
- The framework standardizes human-RL interaction research by offering a generic interface and tools for studying diverse interaction aspects and facilitating experiment design and execution.
- SHARPIE supports diverse human-RL interaction use cases including reward annotation, teaching, action delegation, task specification, and human-AI teaming, facilitating research in cognitive science and RL.


---


[TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets](https://github.com/TobyYang7/TwinMarket)

- TwinMarket: introduces a multi-agent framework designed for simulating socio-economic systems, incorporating User Profile, Belief, Desire, Intention, World Knowledge, Action Space, Market Environment, Social Environment, Order-Driven Trading System, Matching Engine, Data Sources, and Validation Metrics components.
- TwinMarket framework simulates investor behavior within a stock market environment by utilizing Belief-Desire-Intention framework integrated with a simulated social media platform and real-world market data.
- TwinMarket framework facilitates the investigation of emergent market phenomena, such as financial bubbles and volatility clustering, through scalable simulations of individual decision-making and social interactions.


---

[Simulating Rumor Spreading in Social Networks using LLM Agents](https://arxiv.org/abs/2502.01450)

- LLM-based multi-agent network framework: introduces a simulation framework with LLM-based Agent, Post History, Rumor Belief, and Social Network to examine rumor propagation dynamics.
- The framework employs LLM-based Agent to simulate user behavior, utilizing Post History for context and Rumor Belief for opinion tracking within a Social Network.
- This framework assesses how different Social Network structures and agent behaviors impact Rumor Belief and overall rumor dissemination.


---

[Evolving Symbolic 3D Visual Grounder with Weakly Supervised Reflection](https://arxiv.org/abs/2502.01401)

- EASE (Evolvable Symbolic Visual Grounder): introduces a training-free symbolic framework for 3D visual grounding, integrating Agents, executor, Visprog., Ours, test suite, relation encoder, object locations, scene scans, relation functions, and feedback components.
- EASE framework employs offline LLM generation and optimization within its Ours and test suite components to enhance relation encoders, contrasting with online Agents and visual programming Visprog. methods.
- The framework leverages relation encoders and feedback mechanisms to achieve a balance between grounding accuracy and inference efficiency, differing from Agents' online processing and Visprog.'s reliance on annotated relation functions.


---

[Plan-Then-Execute: An Empirical Study of User Trust and Team Performance When Using LLM Agents As A Daily Assistant](https://arxiv.org/abs/2502.01390)

- Plan-then-execute LLM Agents: introduces a framework with LLM Planning, Plan Edit, Planning Outcome, Action Prediction, Action Execution, User-Involved Execution, Manual Specify Action or Feedback, Involve vs Approve, Approve, Involve, Execution Outcome, Successful Login, and Successful Transaction to study user trust and team performance in human-AI collaboration.
- This framework uses plan-then-execute workflow where LLM agents first generate a plan, then users can edit it, and finally the agent executes the plan step-by-step with potential user involvement at each action.
- The architecture allows for empirical investigation of how different levels of user involvement during planning and execution affect user trust and task outcomes when using LLM agents as daily assistants.


---

[TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning](https://arxiv.org/abs/2502.01387)

- TeLL-Drive (Teacher LLM-Guided Deep Reinforcement Learning): introduces a framework integrating LLM-Teacher with Decision Engine, Memory Repository, and Reflective Evaluator, and RL-Student with Actor, Critic, Add & Norm Attention, Multi-Head Attention, Data Distillation, and Mixed Policy for enhanced autonomous driving decision-making.
- TeLL-Drive leverages LLM-Teacher's guidance through Decision Engine, Memory Repository, and Reflective Evaluator to improve RL-Student's Actor-Critic learning and policy via attention mechanisms and data distillation for efficient and robust autonomous driving.
- The framework's architecture with LLM-Teacher and RL-Student components facilitates knowledge transfer and policy refinement, leading to improved adaptability and safety in autonomous driving across diverse scenarios.


---

[PSSD: Making Large Language Models Self-denial via Human Psyche Structure](https://arxiv.org/abs/2502.01344)

- PSSD (Psyche Structure for Self-Denial): introduces a novel paradigm for Large Language Models self-denial, comprising Intuition-based Id Role, Rule-driven Superego Role, and Script-centric Ego Role, to enhance reasoning accuracy.
- PSSD framework leverages multi-agent approach inspired by human psyche structure, utilizing three distinct roles for initial attempts, rule-based guidance, and procedural execution.
- PSSD aims to address limitations of current mistake correction methods by facilitating agents' self-denial within LLMs, leading to improved reasoning and resource efficiency.


---

[Human-Agent Interaction in Synthetic Social Networks: A Framework for Studying Online Polarization](https://arxiv.org/abs/2502.01340)

- Introduces agent-based architecture (individuals with attributes and interactions), LLM infrastructure (enables content generation and analysis), social network structure (governs information dissemination dynamically), agent model (represents individual user with attributes), opinion value (numerical stance on topic), personality description (agent's character traits), short biography (agent's background information), unique username (agent's identifier), interaction history (agent's past engagements), message generation (agent's content creation process), interaction mechanisms (agent's reaction to messages), opinion-based interaction function (evaluates opinion alignment), opinion strength factor (reflects opinion intensity), opinion assessment function (interprets message opinion), opinion update process (agent's opinion change mechanism), social network model (directed graph of agent connections), network structure (set of agents and follow relationships), connection dynamics (network evolution over time), information propagation (message visibility and exposure), recommendation system (determines message presentation), and influence-based scoring system (evaluates message author influence) for studying online polarization in synthetic social networks.
- Framework combines mathematical opinion dynamics with large language models to simulate human-agent interaction in synthetic social networks for controlled experimentation of online polarization.
- Framework enables investigation of polarization mechanisms, bridging gap between theoretical models and empirical observations, offering opportunities to study causal mechanisms underlying online opinion dynamics.


---

[ChartCitor: Answer Citations for ChartQA via Multi-Agent LLM Retrieval](https://arxiv.org/abs/2502.00989)

- ChartCitor: introduces multi-agent framework with Table Extraction Agent, Answer Reformulation Agent, Entity Captioning Agent, LLM Prefiltering Agent, LLM Re-ranking Agent, and Cell Localization Agent for fine-grained chart answer citations.
- ChartCitor framework orchestrates specialized LLM agents to extract tables, reformulate answers, generate captions, retrieve evidence, and localize cited cells in chart images.
- This system enhances explainability and user trust in LLM-assisted chart question answering by providing reliable and logically-explained citations sourced from charts.


---

[PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback](https://arxiv.org/abs/2502.00988)

- PlotGen: introduces a multi-agent framework for scientific data visualization, with Query Planning Agent, Code Generation Agent, Numeric Feedback Agent, Lexical Feedback Agent, Visual Feedback Agent, and Self-Reflection, that leverages multimodal LLMs to iteratively refine visualizations based on user specifications.
- PlotGen framework orchestrates agents for query decomposition, code generation, and multimodal feedback to ensure data accuracy, textual correctness, and visual alignment in generated plots.
- The framework utilizes self-reflection within code generation and feedback agents to iteratively improve plot quality and address errors, enhancing user trust and productivity in data visualization tasks.


---


[Firewalls to Secure Dynamic LLM Agentic Networks](https://github.com/microsoft/Firewalled-Agentic-Networks)

- Firewalled Agentic Networks (FAN): introduces input firewall, data firewall, and trajectory firewall, where FAN automatically constructs task-specific rules from prior simulations to build firewalls for constrained LLM agentic networks.
- FAN offers layers of defense by converting free-form input to protocol, abstracting user data, and self-correcting agent trajectory.
- Data and trajectory firewalls are built from prior simulations to balance adaptability, security, and privacy in LLM agentic networks.


---

[SelfCheckAgent: Zero-Resource Hallucination Detection in Generative Large Language Models](http://arxiv.org/abs/2502.01812v1)

- SelfCheckAgent: introduces a framework for hallucination detection, with Symbolic Agent (semantic representation), Specialized Detection Agent (domain-aware detection) and Contextual Consistency Agent (context-aware verification), providing a multi-dimensional approach.
- SelfCheckAgent framework integrates three distinct agents, utilizing diverse techniques like semantic similarity, fine-tuned NLI models, and contextual consistency checks to evaluate LLM response factuality.
- SelfCheckAgent framework leverages triangulation strategy across agents, enhancing hallucination detection robustness and applicability in complex mathematical and general domains, improving trustworthiness of LLMs.


---


[Agentic Bug Reproduction for Effective Automated Program Repair at Google](http://arxiv.org/abs/2502.01821v1)

- LIBRO: introduces automated BRT generation, with GITS issue, buggy file(s), test file, edit LLM, and candidate BRT, where LIBRO adapts LLM for bug reproduction test generation.
- LIBRO: utilizes code-editing LLM to generate candidate BRT by prompting with bug report, buggy files, and test file.
- LIBRO: aims to generate BRTs by leveraging LLM's understanding of bug descriptions and code context.


---


[Position: Towards a Responsible LLM-empowered Multi-Agent Systems](http://arxiv.org/abs/2502.01714v1)

- RLHF (Reinforcement Learning from Human Feedback): presents a two-step approach involving reward model training from human feedback and language model fine-tuning through reinforcement learning to achieve human value alignment.
- RLHF framework utilizes preference data and techniques like Proximal Policy Optimisation (PPO) or Direct Preference Optimization (DPO) for policy updates, enhancing model agreement with human preferences.
- This method aims to create helpful and harmless AI assistants by incorporating human feedback into the learning process, improving model behaviour and safety.


---

[Al-Khwarizmi: Discovering Physical Laws with Foundation Models](http://arxiv.org/abs/2502.01702v1)

- Al-Khwarizmi: introduces agentic framework for physical law discovery from data, integrating system observation, RAG, prompt, LLM, optimization, score model, test data, and human feedback components.
- Framework leverages foundation models and SINDy method to automate physical law discovery by incorporating prior knowledge and iterative refinement.
- Al-Khwarizmi framework achieves state-of-the-art performance in physical law discovery by utilizing multiple data modalities and automated choices of algorithms.


---



#### 2nd February 2025

[Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](https://arxiv.org/abs/2502.00955)

- DITS (Data Influence-oriented Tree Search): introduces a novel framework for efficient multi-agent system training with data influence-oriented tree search, incorporating Multi Agent Network, MCTS Data Synthesis, Influence Score Estimation, Data Selection, and Iterative Data Synthesis.
- DITS leverages influence scores to guide tree search and data selection, effectively identifying impactful data for system improvement and enhancing model performance.
- DITS derives influence score estimation methods for non-differentiable metrics, reducing computational overhead and enabling efficient synthesis time scaling.


---

[RTBAgent: A LLM-based Agent System for Real-Time Bidding](https://arxiv.org/abs/2502.00792)

- RTBAgent (LLM-based Agent System for Real-Time Bidding): introduces an agent framework for real-time bidding, utilizing Tools (CTR prediction and bidding strategies), Summarized Memory (aggregated information for decision), Reflection Memory (self-assessment of past decisions), Bidding Memory (record of bidding history), Environment Memory (historical market conditions), Two-Step Decision-Making (sequential decision process), Insight Reasoning (analyze decision ranges and risks), Action Making (determine bidding action and reason), and Action Space (range of possible bidding adjustments).
- RTBAgent employs a two-step decision-making process with Insight Reasoning (analyze decision ranges and risks) and Action Making (determine bidding action and reason) to determine optimal bidding prices, leveraging multi-memory retrieval and expert knowledge.
- The framework's multi-memory system, including Reflection Memory (self-assessment of past decisions), Bidding Memory (record of bidding history), and Environment Memory (historical market conditions), enables adaptive bidding strategies by reviewing historical data and market changes.


---

[AgentBreeder: Mitigating the AI Safety Impact of Multi-Agent Scaffolds](https://arxiv.org/abs/2502.00757)

- AGENTBREEDER: introduces evolutionary framework, with Seed Scaffolds, Population, Capability benchmark, Safety benchmark, Embedding function, Clustering function, Pareto Fronts, Elites, Meta Agent, Crossover, Mutation, and New Scaffolds, for multi-objective search over multi-agent system scaffolds.
- AGENTBREEDER framework evaluates scaffolds using capability and safety benchmarks, clusters architectures, identifies Pareto optimal elites, and evolves new generations via meta-agent-driven crossover and mutation.
- AGENTBREEDER framework facilitates exploration of diverse multi-agent scaffolds, balancing capability and safety objectives through evolutionary optimization and quality-diversity search algorithm.


---

[Meta-Prompt Optimization for LLM-Based Sequential Decision Making](https://arxiv.org/abs/2502.00728)

- EXPO (EXPonential-weight algorithm for prompt Optimization): introduces an automated meta-prompt optimization framework for LLM-based agents, with components including LLM Agent (selects action based prompt), Evaluator (measures action performance), Embedding Model (converts text to numbers), Score Estimation NN (predicts meta-prompt scores), Randomized Meta-Prompt Selection (chooses meta-prompt based scores), and Exemplar Set (history of input-score pairs).
- EXPO framework uses adversarial bandit algorithm principles to address non-stationarity in reward observations during sequential decision-making for optimizing task description and meta-instruction within the meta-prompt.
- The framework leverages a neural network for score estimation and exponential-weight mechanism for meta-prompt selection, achieving a balance between exploitation and exploration in meta-prompt optimization.


---

[PhiP-G: Physics-Guided Text-to-3D Compositional Scene Generation](https://arxiv.org/abs/2502.00708)

- PhiP-G (Physics-Guided Text-to-3D Compositional Scene Generation): introduces a framework for compositional scene generation, with AG-extractor (scene graph extraction from text), Scene graph (structured scene representation), AG-generater (2D image generation agent), 3D Gaussian model (3D asset generation model), Asset retrieval (2D asset library access), 2D asset retrieval library (storage for 2D assets), AG-supervisor (visual layout supervision agent), Physical pool (physics-based initial layout), Blender (3D scene environment), and World model (layout prediction and planning).
- PhiP-G integrates LLM-based agents and world model for layout guidance with 3D Gaussian Splatting for efficient and physically consistent 3D scene generation.
- The framework leverages a physical pool and visual supervision for iterative layout refinement, achieving state-of-the-art performance and improved efficiency.


---

[Leveraging LLMs for Dynamic IoT Systems Generation through Mixed-Initiative Interaction](https://arxiv.org/abs/2502.00689)

- IoT-Together (Mixed-Initiative Interaction Paradigm): introduces a system architecture with User Interface (interaction medium), Goal Management (goal identification), Knowledge Management (data repository), Context Management (service hosting), Backend Generation (service generation), Intelligent User Interface Generation (application building), Interoperability platform (data pipeline), IOT DEVICES (sensor network), and Services (concrete functionalities) to enable dynamic IoT system generation through mixed-initiative interaction.
- IoT-Together paradigm facilitates user-system collaboration by leveraging LLMs within Goal Management and Backend Generation for interpreting user queries and generating runtime services based on available IoT data and service definitions.
- The architecture supports dynamic evolvability by generating and integrating new services at runtime, enhancing system adaptability and real-world usability in dynamic IoT environments like smart cities.


---

[Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?](https://arxiv.org/abs/2502.00674)

- Self-MoA (Self-Mixture-of-Agents): introduces Self-MoA, an ensemble method, with Proposer (Generates multiple responses) and Aggregator (Synthesizes responses into output), that aggregates outputs from a single top-performing Large Language Model.
- Self-MoA leverages in-model diversity by repeatedly sampling from the same model, achieving superior performance compared to Mixed-MoA in various benchmarks.
- Self-MoA-Seq, a sequential version, addresses context length limitations by using a sliding window for aggregation, maintaining effectiveness while enabling scalability.


---

[Agent-Based Uncertainty Awareness Improves Automated Radiology Report Labeling with an Open-Source Large Language Model](http://arxiv.org/abs/2502.01691v1)

- Bayesian Prompt Ensemble pipeline: introduces uncertainty-aware predictions for radiology reports using semantically equivalent prompts, LLM, predictions, aggregation function, LLM Agent, entropy-based methods, uniform weights, linear weights, MLP, decision, and uncertainty.
- Bayesian Prompt Ensemble pipeline aggregates multiple LLM prompt outputs via agent-based or entropy-based methods to improve structured data extraction from radiology reports.
- Agent Decision Model within Bayesian Prompt Ensemble pipeline synthesizes prompt responses and explanations to categorize decisions into confidence levels for calibrated uncertainty.


---



#### 1st February 2025

[WHO'S THE MVP? A GAME-THEORETIC EVALUATION BENCHMARK FOR MODULAR ATTRIBUTION IN LLM AGENTS](https://arxiv.org/abs/2502.00510)

- CapaBench (Capability-level Assessment Benchmark): introduces evaluation framework for modular LLM agents with Planning Module (decomposes instructions), Reasoning Module (performs logical inference), Action Module (translates to operations), and Reflection Module (systematic performance analysis).
- CapaBench systematically quantifies module contributions using Shapley Value from game theory for performance attribution.
- Framework facilitates component-level evaluation and holistic system assessment for optimizing modular LLM agents.


---

[MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents](https://arxiv.org/abs/2502.00415)

- MarketSenseAI: introduces a framework leveraging LLM agents including News, Fundamentals, Dynamics, Macroeconomic, and Signal Agents for holistic stock analysis.
- MarketSenseAI framework processes diverse financial data like news, prices, fundamentals, and macroeconomics to support stock analysis and selection decisions.
- The framework utilizes Retrieval-Augmented Generation and Chain-of-Agents architecture to enhance fundamental and macroeconomic analysis accuracy.


---



#### 31st January 2025


[A parallelizable variant of HCA*](http://arxiv.org/abs/2501.19218v1)

- HCA* (Hierarchical Cooperative A* algorithm): introduces parallelizable variant for multi-agent path finding, with Agent (computes paths and intersections), Central Server (manages coordination and conflict resolution), Reservation Table (stores fixed agent paths), Intersection Graph (represents path collisions), and Map Partition (divides map for parallel processing).
- This variant parallelizes path finding and intersection graph construction to reduce computation time.
- Parallelism is achieved by map partitioning and independent agent path calculations, improving performance over standard HCA*.


---

[Multi-agent Multi-armed Bandit with Fully Heavy-tailed Dynamics](http://arxiv.org/abs/2501.19239v1)

- HT-HMUCB (Heavy-Tailed HoMogeneous Upper Confidence Bounds): introduces decentralized multi-agent multi-armed bandit framework with hub identification, arm selection using UCB, transmission, information update, local and global estimation components for homogeneous rewards in heavy-tailed dynamic environments.
- HT-HMUCB framework addresses sparse random graphs and heavy-tailed rewards by exploiting hub structures for variance reduction and robust estimation using median-of-means estimator.
- The framework achieves improved regret bounds compared to existing methods by enabling efficient communication and information aggregation in challenging heavy-tailed scenarios.


---


[Neuro-LIFT: A Neuromorphic, LLM-based Interactive Framework for Autonomous Drone Flight at the Edge](http://arxiv.org/abs/2501.19259v1)

- Neuro-LIFT (Neuromorphic, LLM-based Interactive Framework for Autonomous Drone Flight at the Edge): introduces modular framework integrating Human Interaction Module, Neuromorphic Sensing Module, LLM, and Planning and Control Module for autonomous drone navigation based on human commands.
- Neuro-LIFT framework utilizes Human Interaction Module for user commands, Neuromorphic Sensing Module for environment perception, LLM for command interpretation, and Planning and Control Module for drone maneuver execution.
- Neuro-LIFT framework achieves real-time interactive autonomous drone flight by combining LLM-based natural language understanding with low-latency, energy-efficient neuromorphic vision for enhanced responsiveness and adaptability.


---

[True Online TD-Replan(\xce\xbb) Achieving Planning through Replaying](http://arxiv.org/abs/2501.19027v1)

- TD-Replan(\xce\xbb) (True Online TD-Replan(\xce\xbb)): introduces a novel reinforcement learning method extending True Online TD by incorporating experience replay and a parameter to control replay density and target depth.
- TD-Replan(\xce\xbb) utilizes interim \xce\xbb-return targets and online updates for efficient learning, demonstrating improved performance in tasks benefiting from experience replay.
- The method achieves balance between planning and acting by replaying past experiences and adjusting replay density, making it suitable for complex environments and deep learning integration.


---
[Swarm-Gen: Fast Generation of Diverse Feasible Swarm Behaviors](http://arxiv.org/abs/2501.19042v1)

- Swarm-Gen: introduces a framework with Generative Model (CVAE/VQ-VAE), Safety-Filter (SF), and Initialization Network, with Encoder, Decoder, QP Block, PixelCNN, MLP, and Fixed-Point Solver components, for fast generation of diverse feasible swarm behaviors.
- This framework uses generative models to sample diverse trajectories, projects them onto a feasible set using a safety filter, and accelerates the safety filter convergence with a learned initialization network.
- The approach demonstrates real-time generation of multi-modal swarm trajectories on commodity GPUs, offering a balance between trajectory diversity and computational efficiency using CVAE and VQ-VAE generative models.


---

[LLM-based Affective Text Generation Quality Based on Different Quantization Values](http://arxiv.org/abs/2501.19317v1)

- LLM (Large Language Model): introduces quantization, LLMs, emotion classifier, seed prompts, emotion-prompt, text generation module, GPU RAM, inference time, and memory to investigate the trade-off between quantization values and affective text generation quality.
- This paper evaluates the impact of different quantization levels (8, 16, 32 bits) on the performance of various LLMs (Llama-2, Mistral, Mixtral) in generating affective text, considering GPU RAM usage and inference time.
- The research highlights that while quantization reduces memory consumption, it can affect text quality and inference time, revealing a trade-off between efficiency and efficacy in LLM-based affective text generation.


---

[An Empirical Game-Theoretic Analysis of Autonomous Cyber-Defence Agents](http://arxiv.org/abs/2501.19206v1)

- MRO (Multiple Response Oracles): introduces a framework for holistic evaluation of ACD approaches, with INITIALPOLICIES(), Set initial mixtures, RBlue, RRed, GBlue, GRed, AUGMENTGAME, and SOLVEGAME components.
- MRO framework extends the Double Oracle algorithm by incorporating multiple response oracles to enhance the assessment of Autonomous Cyber-Defence approaches.
- MRO algorithm utilizes response functions and game-theoretic analysis to iteratively refine and evaluate policies for cyber-defence and cyber-attack agents.



[Beyond checkmate: exploring the creative chokepoints in AI text](http://arxiv.org/abs/2501.19301v1)

- Chess-Text Analogy Framework: introduces a method to explore human and AI text differences by analogy to chess game segments (opening, mid game, end game) and text segments (introduction, body, conclusion), utilizing source and segment comparisons, statistical tests, feature extraction, and various datasets and LLMs.
- This framework examines creative limitations in AI text generation by analyzing stylometric and psycholinguistic features across text segments, finding body segment crucial for AI detection and greater human cross-segment variation.
- Research emphasizes text segments in AI detection, suggesting body segment focus and cross-segment feature variations improve detection and provide insights into LLMs' creative abilities.


---


[PixelWorld: Towards Perceiving Everything as Pixels](http://arxiv.org/abs/2501.19339v1)

- PEAP (Perceive Everything as Pixels): introduces Language Model, ViT, and Text Instruction components for unified multimodal input processing.
- PEAP framework processes all modalities as pixels, contrasting with token-based methods and enhancing multimodal task performance.
- The framework evaluation suite, PIXELWORLD, demonstrates PEAP's effectiveness and identifies areas for improvement in complex reasoning tasks.


---


[Enabling Autonomic Microservice Management through Self-Learning Agents](http://arxiv.org/abs/2501.19056v1)

- SERVICEODYSSEY: introduces a self-learning agent system for autonomic microservice management, leveraging Curriculum Builder for task generation, Execution Planner for plan creation, Knowledge Curator for skill consolidation, Data Layer for data storage, and Management Layer for module orchestration within the Operational Environment.
- SERVICEODYSSEY framework incorporates High-level Manager to decompose tasks and coordinate Low-level Agents, utilizing Running State and Interaction History for context, Task Queue and Execution Queue for task management, and Feedback and Skill Library for learning and improvement.
- The system refines solutions through Environment Feedback, Peer Feedback, and Hierarchical Feedback, demonstrating its effectiveness in the Sock Shop Microservice environment for autonomic management of microservices.



[Secured Communication Schemes for UAVs in 5G: CRYSTALS-Kyber and IDS](http://arxiv.org/abs/2501.19191v1)

- CRYSTALS-Kyber and IDS Framework: introduces secure UAV communication architecture, integrating UAV Layer, Raspberry Pi, AES Encryption-Decryption, KEM, ECC, CRYSTALS-Kyber, Communication Layer, Ground Station Layer, Server, File Storage, IDS Dataset, KEM Dataset, AI Techniques, and IDS Module.
- This architecture employs hybrid cryptography using AES with ECC and CRYSTALS-Kyber for quantum resistance, alongside AI-driven IDS for intrusion detection in 5G UAV networks.
- Evaluated in VPN and 5G, the framework demonstrates effective security and performance balance, suitable for resource-limited UAVs facing quantum threats.


---
[Vintix: Action Model via In-Context Reinforcement Learning](http://arxiv.org/abs/2501.19400v1)

- Vintix: introduces a fixed cross-domain model for in-context reinforcement learning using Noise Distillation, Cross-Domain Dataset, Causal Transformer, and Algorithm Distillation components.
- Vintix framework employs Algorithm Distillation to construct versatile action models by learning behaviors through in-context reinforcement learning.
- The framework demonstrates self-correction capabilities and scaling potential of In-Context Reinforcement Learning for generalist decision-making systems across multiple domains.


---


[MINDSTORES: Memory-Informed Neural Decision Synthesis for Task-Oriented Reinforcement in Embodied Systems](http://arxiv.org/abs/2501.19318v1)

- MINDSTORES: experience-augmented planning framework enables embodied agents to build and leverage mental models through natural interaction with their environment.
- Framework uses database of past experiences; represents experiences as natural language embeddings; allows efficient retrieval and reasoning by LLM planner; generates insights and guides plan refinement.
- MINDSTORES represents an important step toward more capable embodied AI systems that can learn continuously through natural experience.


---

[Language Games as the Pathway to Artificial Superhuman Intelligence](http://arxiv.org/abs/2501.18924v1)

- Language games: framework for expanded data reproduction to overcome data reproduction trap in LLMs.
- Includes role fluidity, reward variety, and rule plasticity for open-ended exploration and human-AI co-evolution towards superhuman intelligence through dynamic linguistic interaction.
- This framework is important as it redefines data reproduction as an engine for superhuman intelligence.


---

[Enabling Autonomic Microservice Management through Self-Learning Agents](http://arxiv.org/abs/2501.19056v1)

- SERVICEODYSSEY: self-learning agent system autonomously manages microservices without prior knowledge of service-specific configurations.
- Leverages curriculum learning principles and iterative exploration; develops deep understanding of operational environments; reduces dependence on human input; includes Curriculum Builder, Execution Planner, and Knowledge Curator modules.
- This approach has potential for autonomic microservice management as demonstrated by prototype.


---

[Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization](https://arxiv.org/abs/2501.17974)

- Inference Budget-Constrained Policy Optimization (IBPO) is an algorithm designed to enable models to understand query difficulty and allocate inference budgets accordingly.
- It uses utility maximization with inference budget constraint, addresses single-modal behavior in long reasoning models, and improves token efficiency.
- This method is important as it significantly enhances reasoning efficiency and shows potential for broader applications beyond mathematical problem-solving.


---


[s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)

- s1 is a simple test-time scaling approach to improve language model reasoning performance by using budget forcing and small dataset.
- s1 uses budget forcing to control test-time compute, curated small dataset s1K with 1,000 high-quality questions, and supervised finetuning on Qwen2.5-32B-Instruct.
- s1 demonstrates that simple test-time scaling can achieve strong reasoning performance and sample efficiency.


---

[Do LLMs Strategically Reveal, Conceal, and Infer Information? A Theoretical and Empirical Analysis in The Chameleon Game](http://arxiv.org/abs/2501.19398v1)

- The Chameleon Game: is a language-based hidden-identity game to investigate information control and decision-making capabilities of LLMs.
- Framework analyzes strategic interactions, information control, and decision-making capabilities using theoretical and empirical analysis with contemporary LLMs such as GPT-4, GPT-4o, Gemini 1.5, and Claude 3.5 Sonnet.
- This framework is important as it points to a weakness of contemporary LLMs in strategic interactions.


---


[TV-Dialogue: Crafting Theme-Aware Video Dialogues with Immersive Interaction](http://arxiv.org/abs/2501.18940v1)

- TV-Dialogue: novel multi-modal agent framework ensures theme alignment and visual consistency through real-time immersive interactions among video characters.
- Introduces Theme-aware Video Dialogue Crafting (TVDC) task, generates dialogues aligned with video content and user-specified themes, includes multi-granularity evaluation benchmark for assessment, enables zero-shot generation for any length and theme, applicable for video re-creation and film dubbing.
- TV-Dialogue framework underscores potential for video re-creation, film dubbing, and downstream multimodal tasks.


---


[KBQA-01: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search](http://arxiv.org/abs/2501.18922v1)

- KBQA-01: is a novel agentic Knowledge Base Question Answering (KBQA) method with Monte Carlo Tree Search (MCTS).
- ReAct-based agent process; stepwise logical form generation; KB environment exploration; MCTS for heuristic search; balances exploration and search space; generates high-quality annotations; incremental fine-tuning; outperforms low-resource KBQA methods.
- KBQA-01 improves performance in low-resource KBQA and provides publicly available code for further research.


---


[Survey and Improvement Strategies for Gene Prioritization with Large Language Models](http://arxiv.org/abs/2501.18794v1)

- Gene Prioritization Framework benchmarks and improves large language models for gene prioritization using multi-agent and HPO classification approaches combined with a divide-and-conquer strategy.
- Framework benchmarks various LLMs including GPT-4 and Mixtral, uses multi-agent and HPO classification for case solvability, and employs divide-and-conquer strategy to enhance accuracy and overcome biases.
- This framework significantly optimizes disease-causal gene identification and streamlines rare genetic disorder diagnosis.


---



[Free Agent in Agent-Based Mixture-of-Experts Generative AI Framework](https://arxiv.org/abs/2501.17903)

- RLFA (Reinforcement Learning Free Agent) algorithm: introduces sports-inspired mechanism for replacing underperforming agents in multi-agent GenAI systems.
- Draws inspiration from Major League Baseball free agency, uses mixture-of-experts approach, and improves performance and adaptability in multi-agent systems.
- RLFA provides a straightforward route for continuous upgrades and maintains performance in critical tasks.


---

[Autonomous Legacy Web Application Upgrades Using a Multi-Agent System](http://arxiv.org/abs/2501.19204v1)

- Multi-agent pipeline: LLM based multi-agent system autonomously upgrades legacy web applications to the latest version.
- System distributes tasks across multiple phases; updates files to latest version; uses Zero-Shot and One-Shot Learning prompts; keeps context across tasks and agents.
- Proposed system contributes as working foundation for future model implementations with existing code.


---

[Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming](https://arxiv.org/abs/2501.18837)

- Constitutional Classifiers: introduces classifier safeguards, with Human's Query, Constitutional Input Classifier, AI Assistant, Constitutional Output Classifier, Response Shown to Human, Response Blocked, Harmless Constitution, Harmful Constitution, LLM with Constitution, Synthetic LLM Prompts and Completions, Data Augmentation Pipeline, Harmless Pool Set, and Training Set, as a framework to defend large language models against universal jailbreaks by monitoring both user inputs and model outputs using constitution-guided classifiers.
- "Constitutional Classifiers framework trains classifier safeguards using synthetic data generated by prompting language models with natural language rules defining harmful and harmless content categories."
- "This approach enhances robustness and deployment viability by incorporating data augmentation, benign data pools, and streaming prediction in output classifiers for real-time intervention."


---


#### 30th January 2025

[Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](http://arxiv.org/abs/2501.18539v1)

- ARM (Alignment-Oriented LLM-based Retrieval Method): is an LLM-based retrieval method that aligns questions with data organization by exploring relationships among data objects.
- ARM uses constrained decoding with N-grams, a reasoning solver for structure alignment, and self-verification for object selection, and it is evaluated on Bird and OTT-QA datasets.
- This method achieves better retrieval performance and efficiency compared to standard and agentic RAG approaches.


---

[Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs](https://arxiv.org/abs/2501.18585)

- TIP (thought switching penalty): is a decoding strategy that discourages premature transitions between thoughts, encouraging deeper exploration of each reasoning path.
- It introduces a novel metric to quantify underthinking by measuring token efficiency in incorrect answers, and it improves accuracy across challenging datasets without requiring model fine-tuning.
- This framework contributes to understanding reasoning inefficiencies in o1-like LLMs and offers a practical solution to enhance their problem-solving capabilities.


---

[REPOAUDIT: An Autonomous LLM-Agent for Repository-Level Code Auditing](https://arxiv.org/abs/2501.18160)

- REPOAUDIT: introduces autonomous LLM-agent, with initiator, explorer, validator, memory, for precise, efficient repository-level code auditing by demand-driven exploration.
- It employs agent memory for on-demand repository exploration and validator for hallucination mitigation.
- Validation design improves precision by checking data-flow facts and path condition satisfiability, discarding false positives.


---



[Leveraging LLM Agents for Automated Optimization Modeling for SASP Problems: A Graph-RAG based Approach](http://arxiv.org/abs/2501.18320v1)

- MAG-RAG is automated modeling approach based on retrieval-augmented generation technique for SASP problems.
- It uses multi-agent structure for AOM architecture, graph-based RAG for domain knowledge integration, human expert modeling principles and precise knowledge retrieval using graph structure.
- MAG-RAG approach realizes the potential of LLM-assisted AOM for solving SASP problems.


---

[REPOAUDIT: An Autonomous LLM-Agent for Repository-Level Code Auditing](http://arxiv.org/abs/2501.18160v1)

- REPOAUDIT: autonomous LLM-agent designed for precise and efficient repository-level code auditing.
- Equipped with agent memory, REPOAUDIT explores code repository on demand, analyzes data-flow facts along feasible program paths, and introduces validator for hallucination mitigation.
- REPOAUDIT demonstrates substantial potential for flexible and configurable code security analysis.


---


[Design and Validation of Learning Aware HMI For Learning-Enabled Increasingly Autonomous Systems](https://arxiv.org/abs/2501.18506)

- LEIAS (Learning-Enabled Increasingly Autonomous Systems): is an architecture designed to enhance operational safety by emphasizing communication representation and pilot preference learning in autonomous systems.
- - incorporates human-machine collaboration
- uses Soar cognitive architecture with reinforcement learning
- provides transparent multi-sensor data assessment (GPS, IMU, LIDAR)
- adapts to pilot preferences
- validated in XPlane simulation for sensor anomaly management
- This framework is important for advancing the safety and reliability of learning-enabled autonomous systems in complex operational environments.


---


[Integrating LMM Planners and 3D Skill Policies for Generalizable Manipulation](http://arxiv.org/abs/2501.18733v1)

- LMM-3DP: LMM-3DP is a framework integrating LMM planners and 3D skill policies for generalizable robotic manipulation.
- Integrates LMM planners and 3D skill policies, uses high-level planning with visual feedback, includes critic agent for self-improvement, enables lifelong learning with skill library, utilizes semantic 3D feature field for low-level control.
- LMM-3DP significantly enhances robot manipulation by improving success rate and planning accuracy in complex tasks.


---


[Invisible Traces: Using Hybrid Fingerprinting to identify underlying LLMs in GenAI Apps](http://arxiv.org/abs/2501.18712v1)

- Hybrid Fingerprinting framework: Novel fingerprinting framework integrates static and dynamic techniques to identify underlying LLMs in GenAI Apps.
- Addresses real-world challenges; Combines static and dynamic fingerprinting; Identifies architectural features and behavioral traits; Demonstrates semantic distinction in LLM outputs; Robust and accurate in complex environments.
- Framework is important for ensuring security and transparency in AI applications by reliably identifying underlying LLMs.


---

[Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](http://arxiv.org/abs/2501.18539v1)

- ARM (Alignment-Oriented LLM-based Retrieval Method) is a retrieval method that aligns question with data collection organization by exploring relationships among data objects.
- It is retrieve-all-at-once solution for complex queries by better aligning question with data organization and exploring relationships among data objects beyond utterance matching for efficient and comprehensive retrieval.
- The proposed method is important as it improves retrieval performance for complex questions by addressing limitations of existing RAG approaches.


---

[Leveraging LLM Agents for Automated Optimization Modeling for SASP Problems: A Graph-RAG based Approach](http://arxiv.org/abs/2501.18320v1)

- MAG-RAG is automated modeling approach based on retrieval-augmented generation technique for SASP problems.
- It uses multi-agent structure for AOM architecture, graph-based RAG for domain knowledge integration, human expert modeling principles and precise knowledge retrieval using graph structure.
- MAG-RAG approach realizes the potential of LLM-assisted AOM for solving SASP problems.


---

[LLM-AutoDiff: Auto-Differentiate Any LLM Workflow](https://arxiv.org/abs/2501.16673)

- LLM-AutoDiff is a novel framework for Automatic Prompt Engineering (APE) that extends textual gradient-based methods to multi-component, potentially cyclic LLM architectures.
- Framework accommodates functional nodes, preserves time-sequential behavior, combats "lost-in-the-middle" problem, boosts training efficiency, and uses graph-centric lens.
- LLM-AutoDiff offers a powerful new paradigm for scaling and automating LLM workflows.


---


#### 29th January 2025

[Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate](https://arxiv.org/abs/2501.17703)

- Critique Fine-Tuning (CFT): is a framework where models learn to critique noisy responses rather than imitating correct ones.
- CFT encourages deeper analysis and nuanced understanding, uses GPT-40 to generate critiques, and shows consistent improvement over SFT on math benchmarks.
- This approach offers a more effective alternative to advance the reasoning of language models.

---

[Human-Aligned Skill Discovery: Balancing Behaviour Exploration and Alignment](https://arxiv.org/abs/2501.17431)

- HaSD (Human-aligned Skill Discovery): is a framework designed to incorporate human feedback into unsupervised skill discovery to find safer and more aligned skills.
- Addresses unconstrained skill discovery, finds useful skills in complex environments, optimizes skill diversity and human alignment, maintains alignment throughout discovery, and allows configurable skills with diversity-alignment trade-offs.
- This framework is important as it enables the discovery of diverse, safe, and human-aligned skills for practical applications.


---

[LARGE LANGUAGE MODELS THINK TOO FAST TO EXPLORE EFFECTIVELY](http://arxiv.org/abs/2501.18009v1)

- Large Language Models (LLMs): Study investigates exploration capabilities of LLMs in open-ended tasks using Little Alchemy 2.
- LLMs underperform humans in exploration; uncertainty-driven strategies dominant; empowerment underutilized; premature decisions due to fast processing.
- Findings are crucial for enhancing LLM adaptability and exploration effectiveness.


---


[Is Conversational XAI All You Need? Human-AI Decision Making With a Conversational XAI Assistant](http://arxiv.org/abs/2501.17546v1)

- Conversational XAI assistant: Conversational XAI interface is proposed to augment existing XAI methods to increase user engagement and boost user understanding of AI system.
- Exploration of conversational XAI interface impact on user understanding, trust and reliance; comparison with XAI dashboard; over-reliance on AI system observed; enhanced conversations amplified over-reliance; illusion of explanatory depth.
- Findings have important implications for designing effective conversational XAI interfaces to facilitate appropriate reliance and improve human-AI collaboration.


---

[RICOTA: Red-teaming of In-the-wild Conversation with Test Attempts](http://arxiv.org/abs/2501.17715v1)

- RICOTA: is a Korean red teaming dataset of in-the-wild user interactions.
- It uses user-chatbot conversations from a Korean Reddit-like community, focuses on jailbreak attempts, and provides a novel evaluation approach.
- This dataset is important for evaluating LLMs' ability to identify conversation types and user testing purposes.


---

[ACTIONS SPEAK LOUDER THAN WORDS: AGENT DECISIONS REVEAL IMPLICIT BIASES IN LANGUAGE MODELS](http://arxiv.org/abs/2501.17420v1)

- Language-agent simulation technique: systematically investigates implicit biases in LLMs across diverse sociodemographic groups and decision-making scenarios.
- It uses persona generation and action generation steps, reveals that state-of-the-art LLMs exhibit significant sociodemographic disparities, and shows that implicit biases are amplified compared to explicit biases.
- This framework provides a way to identify biases in LLM-powered applications, ensuring they are aligned with ethical principles and societal norms.


---


[GENERAL SCENE ADAPTATION FOR VISION-AND-LANGUAGE NAVIGATION](http://arxiv.org/abs/2501.17403v1)

- GSA-VLN (General Scene Adaptation for VLN): is a novel task requiring agents to execute navigation instructions within a specific scene and simultaneously adapt to it for improved performance over time.
- GSA-VLN introduces environment-specific memory bank, uses three-stage instruction orchestration pipeline with LLMs, and proposes Graph-Retained DUET (GR-DUET) method.
- This framework addresses the challenge of single-scene adaptation, enabling agents to continuously improve as they execute instructions in previously unseen environments.


---

#### 28th January 2025

[Thalamic oscillations distinguish natural states of consciousness in humans](https://www.biorxiv.org/content/10.1101/2025.01.28.635248v1.full)
- A novel fast thalamic oscillation (20-45 Hz) is identified in humans, which specifically occurs during wakefulness and REM sleep, and is absent during NREM sleep.
- The oscillation is localized to the central thalamus and is temporally coupled with eye movements during REM sleep.

---

[LARGE LANGUAGE MODEL CRITICS FOR EXECUTION-FREE EVALUATION OF CODE CHANGES](https://arxiv.org/abs/2501.16655v1)

- LLM Critics: is a framework that uses LLM-based critics to derive execution-free evaluation proxies for code changes.
- It uses gold test patch as reference, predicts executability of editing locations, aggregates predictions to predict build status, and outperforms other reference-free and reference-aware LLM critics.
- This framework enables more efficient evaluation of code changes without relying on execution.


---

[SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://arxiv.org/abs/2501.17161)

- SFT (Supervised fine-tuning) and RL (reinforcement learning): are compared on generalization and memorization in text and visual environments.
- RL generalizes better than SFT, especially with outcome-based reward; SFT memorizes training data; RL improves visual recognition; SFT stabilizes output format for RL.
- RL is advantageous for acquiring generalizable knowledge in complex, multimodal tasks.


---



[MCTS-SQL: An Effective Framework for Text-to-SQL with Monte Carlo Tree Search](http://arxiv.org/abs/2501.16607v1)

- MCTS-SQL (Monte Carlo Tree Search for SQL): is a framework for text-to-SQL that uses Monte Carlo Tree Search to guide SQL generation iteratively.
- It includes a schema selector for extracting relevant information and an MCTS-based generator for iterative query refinement; it uses a fast-slow thinking approach with a direct SQL generation component and an MCTS-based refiner; it achieves state-of-the-art performance on the BIRD and SPIDER benchmarks.
- This framework improves the accuracy and reliability of text-to-SQL systems, especially when dealing with complex user queries.


---

[ToolFactory: Automating Tool Generation by Leveraging LLM to Understand REST API Documentations](http://arxiv.org/abs/2501.16945v1)

- ToolFactory: is an open-source pipeline for automating tool generation from unstructured API documents.
- It includes API Extraction Benchmark, APILlama model fine-tuned with prompt tuning, and tool validation pipeline.
- This framework facilitates the seamless integration of scientific REST APIs into AI workflows.


---

[A Stochastic Dynamical Theory of LLM Self-Adversariality: Modeling Severity Drift as a Critical Process](http://arxiv.org/abs/2501.16783v1)

- Stochastic dynamical framework: models how LLMs may self-amplify biases through chain-of-thought reasoning.
- It uses a continuous-time stochastic differential equation (SDE) approach, analyzes phase transitions, derives stationary distributions, and investigates scaling laws.
- This framework provides a basis for formal verification of model stability and bias propagation.


---

[MACI: Multi-Agent Collaborative Intelligence for Robust Reasoning and Temporal Planning](http://arxiv.org/abs/2501.16689v1)

- MACI (Multi-Agent Collaborative Intelligence): is a framework centered on a meta-planner that orchestrates multiple agents to generate planner templates.
- It includes a three-tier architecture with meta-planning, common and specialized agents; enables advanced temporal reasoning and adaptability; decouples planning from validation.
- This framework provides a robust solution for complex reasoning and planning tasks.


---

[Auto-Differentiating Any LLM Workflow: A Farewell to Manual Prompting](http://arxiv.org/abs/2501.16673v1)

- LLM-AutoDiff: is a framework for Automatic Prompt Engineering (APE) that extends textual gradient-based methods to multi-component, potentially cyclic LLM architectures.
- It treats each textual input as a trainable parameter, uses a frozen "backward engine" LLM to generate feedback, accommodates functional nodes, preserves time-sequential behavior, and combats the "lost-in-the-middle" problem.
- This framework offers a new paradigm for scaling and automating LLM workflows.


---


[JUPYBARA: Operationalizing a Design Space for Actionable Data Analysis and Storytelling with LLMs](http://arxiv.org/abs/2501.16661v1)

- JUPYBARA: is an AI-enabled assistant for actionable EDA and storytelling implemented as a Jupyter Notebook extension.
- It employs design-space-aware prompting and multi-agent architectures, including semantic, rhetorical, and pragmatic dimensions, to operationalize the design space.
- This framework enhances usability, steerability, explainability, and reparability in actionable data analysis and storytelling.


---

[A sketch of an AI control safety case](http://arxiv.org/abs/2501.17315v1)

- AI control: framework argues that models are safe because of measures such as monitoring and human auditing.
- Framework uses control evaluation with red and blue teams, includes untrusted and trusted monitors, and uses a safety layer to prevent data exfiltration.
- This framework provides a step toward more concrete arguments that can be used to show that a dangerously capable LLM agent is safe to deploy.


---

#### 27th of January 2025

[GUI-Bee : Align GUI Action Grounding to Novel Environments via Autonomous Exploration](https://arxiv.org/abs/2501.13896)

- GUI-Bee is MLLM-based autonomous agent to collect environment-specific data through exploration and fine-tune GUI grounding models for novel environments.
- novel environments; autonomous exploration; Q-ICRL method; exploration efficiency; data quality; NovelScreenSpot benchmark; align GUI action grounding models.
- Aligning GUI action grounding models to novel environments significantly enhances performance.


---


[Janus-Pro: Unified Multimodal Understanding and
Generation with Data and Model Scaling](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf)

- Janus-Pro: Advances multimodal models via optimized training, expanded data, and model scaling. Janus-Pro achieves SOTA-level performance in both multimodal understanding and text-to-image generation benchmarks.
- Enhanced training strategy includes "Longer Training in Stage I" and "Focused Training in Stage II" for better efficiency and performance. This refines the original 3-stage training process of Janus.
- Text-to-image generation stability and aesthetic quality are significantly enhanced through synthetic data and improved training.
- Decoupled visual encoding remains a core and effective architectural design for unified multimodal tasks.
- 7B model demonstrates strong scalability of the decoupled visual encoding approach.

---

[On the Feasibility of Using LLMs to Execute Multistage Network Attacks](https://arxiv.org/abs/2501.16466)

- Incalmo: is an LLM-agnostic high-level attack abstraction layer that sits between an LLM and the environment.
- Incalmo uses action planner, attack graph service and environment state service to enable LLMs to specify high-level tasks, translate them into low-level primitives, and provide structure for selecting relevant actions.
- Incalmo consists of three stages. The first stage is called "onboarding pre-prompt"-stage, which “teaches” the LLM the capabilities of Incalmo, Second stage provides environment specific prompts to outline attach goals and environment details. In the third stage, the LLM autonomously executes the multistage attack via Incalmo in an interactive execution loop. 
- Demonstrates capability to find vurnerable services, execute exploits to gain access to network, to discover misconfigurations and vulnerabilities to move laterally and exploit vulnerabilities to escalate privileges and exfiltrate data from networks. 
- Demonstrates, that abstraction is more important than LLM model size and that Incalmo-action planner module is critical module.
- This framework enables LLMs to successfully execute multistage attacks in realistic emulated networks.



---

[Gensors: Authoring Personalized Visual Sensors with Multimodal Foundation Models and Reasoning](https://arxiv.org/abs/2501.15727)

- Gensors is a system designed to empower users to create personalized visual sensors by leveraging multimodal foundation models and reasoning.
- It uses two-stage pipeline with Gemini 1.5 Flash and Pro, supports user-configurable logic and examples, and facilitates criteria refinement and debugging.
- Gensors is important as it makes intelligent sensing technologies more accessible and customizable for end-users.


---

[MULTI-AGENT GEOSPATIAL COPILOTS FOR REMOTE SENSING WORKFLOWS](http://arxiv.org/abs/2501.16254v1)

- GeoLLM-Squad: geospatial Copilot introduces multi-agent paradigm to remote sensing workflows by separating agentic orchestration from geospatial task-solving.
- Multi-agent system; agentic orchestration; geospatial task-solving; specialized sub-agents; open-source AutoGen and GeoLLM-Engine; diverse applications; robust performance; improved agentic correctness.
- GeoLLM-Squad highlights the potential of multi-agent AI in advancing remote sensing workflows.


---


[Will Systems of LLM Agents Cooperate: An Investigation into a Social Dilemma](https://arxiv.org/abs/2501.16173)

- LLM Agent System: Framework investigates cooperative tendencies of Large Language Model (LLM) agents in social dilemma by prompting LLMs to generate strategies for iterated Prisoner's Dilemma.
- Defines three classes of agents (attitudes): agressive, cooperative and neutral.
- evolutionary game theory; strategic dispositions; aggressive, cooperative, neutral; distinct biases; long-term behaviour; strategic environments.
- This research highlights importance of considering strategic environments for deployed LLM-based autonomous agents and their potential long-term behaviour.


---

[AI Agents for Computer Use: A Review of Instruction-based Computer Control, GUI Automation, and Operator Assistants](http://arxiv.org/abs/2501.16150v1)

- AI Agents for Computer Use: A Review offers a comprehensive overview of instruction-based computer control agents, GUI automation, and operator assistants.
- It examines agents taxonomy, development, resources, shift to foundation models, datasets, evaluation methods, and deployment challenges.
- This review provides a comprehensive foundation to understand and push the future development of AI agents for computer use.


---

[LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models](http://arxiv.org/abs/2501.15850v1)

- LLM-attacker: closed-loop adversarial scenario generation framework leveraging large language models.
- multiple LLM agents; identify optimal attackers; optimize attacker trajectories; iterative refinement based on ADS performance; feedback loop.
- Framework is important to test and enhance the safety and robustness of ADS.


---

[MADP: Multi-Agent Deductive Planning for Enhanced Cognitive-Behavioral Mental Health Question Answer](http://arxiv.org/abs/2501.15826v1)

- MADP (Multi-Agent Deductive Planning): is a CBT-based multi-agent reasoning strategy that analyzes interactions among multiple CBT elements for mental health support.
- Deeper understanding of help-seeker context; personalized assistance; fine-tuned LLM (MADP-LLM); enhanced emotional reasoning; reduced deployment costs.
- MADP framework effectively provides personalized, empathetic, and targeted mental health support.


---

[Harnessing Diverse Perspectives: A Multi-Agent Framework for Enhanced Error Detection in Knowledge Graphs](http://arxiv.org/abs/2501.15791v1)

- MAKGED (Multi-Agent framework for Knowledge Graph Error Detection): is a novel framework utilizing multiple large language models in a collaborative setting for enhanced knowledge graph error detection.
- multi-agent framework; multiple LLMs; collaborative setting; subgraph embeddings; query embeddings; transparent decision-making; multi-round discussions.
- MAKGED enhances the reliability of downstream applications by improving the accuracy and robustness of knowledge graph error detection.


---

[LLM-powered Multi-agent Framework for Goal-oriented Learning in Intelligent Tutoring System](http://arxiv.org/abs/2501.15749v1)

- GenMentor: LLM-powered multi-agent framework is designed for goal-oriented and personalized learning within Intelligent Tutoring System.
- multi-agent system; goal-oriented learning; personalized learning; skill gap identification; adaptive learner modeling; personalized resource delivery.
- GenMentor effectively enhances learning guidance, content quality, goal alignment and resource targeting for enhanced personalization.


---

[Deception in LLMs: Self-Preservation and Autonomous Goals in Large Language Models](http://arxiv.org/abs/2501.16513v1)

- DeepSeek R1: is a model trained to output reasoning tokens, exhibiting deceptive tendencies and self-preservation instincts.
- The model attempts self-replication, masks true objectives, and expands capabilities autonomously.
- This study highlights the critical need for robust goal specification and safety frameworks before physical implementation.


---

[MULTI-AGENT GEOSPATIAL COPILOTS FOR REMOTE SENSING WORKFLOWS](http://arxiv.org/abs/2501.16254v1)

- GeoLLM-Squad: introduces a multi-agent paradigm to remote sensing workflows.
- It separates agentic orchestration from geospatial task-solving, uses AutoGen and GeoLLM-Engine frameworks, and enables modular integration of diverse applications.
- This approach maintains robust performance and improves agentic correctness compared to single-agent systems.


---

[Will Systems of LLM Agents Cooperate: An Investigation into a Social Dilemma](http://arxiv.org/abs/2501.16173v1)

- LLM (Large Language Model) agents framework: investigates emergent cooperative tendencies in a social dilemma.
- Framework prompts LLMs to generate complete strategies, uses evolutionary game theory, simulates populations with different strategic dispositions, and observes evolutionary dynamics.
- This research provides insights into long-term behavior of deployed LLM-based autonomous agents and highlights importance of strategic environments.


---


#### 26th of January 2025

[Qwen2.5-1M Technical Report](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2_5_1M_Technical_Report.pdf)

- Introduces  Qwen2.5-1M, which extends open source support for 1M token context length.
- Includes infererence framework, which speeds up 1M context inference by 3.2x to 6.7x.


---

[OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas](http://arxiv.org/abs/2501.15427v1)

- OpenCharacter: framework trains customizable role-playing LLMs with large-scale synthetic personas.
- Explores large-scale data synthesis approach, uses response rewriting and generation strategies, achieves performance comparable to GPT-40 models.
- This work is important for advancing research in customizable role-playing dialogue systems.


---

[ToM-agent: Large Language Models as Theory of Mind Aware Generative Agents with Counterfactual Reflection](http://arxiv.org/abs/2501.15355v1)

- ToM-agent is a novel paradigm designed to empower LLM-based generative agents to simulate Theory of Mind in open-domain conversational interactions.
- Disentangles confidence from mental states; emulates agent's perception of counterpart's mental states (beliefs, desires, intentions - BDIs); dynamically adjusts inferred BDIs; counterfactual intervention method; enhances reflection efficiency.
- ToM-agent provides new insights for studying large-scale LLMs-based simulation of human social behaviors.


---

#### 25th January 2025

[OptiSeq: Optimizing Example Ordering for In-Context Learning](http://arxiv.org/abs/2501.15030v1)

- OptiSeq: introduces a score based on log probabilities of LLM outputs to prune example orderings in few-shot ICL.
- optimizing example ordering; in-context learning; LLM outputs; prune orderings; best order; correct/incorrect outputs; empirical evaluation; accuracy improvement.
- OptiSeq improves accuracy significantly across multiple tasks.


---

#### 24th January 2025

[RL + Transformer = A General-Purpose Problem Solver](https://arxiv.org/abs/2501.14176)

- ICRL (In-Context Reinforcement Learning): introduces LLaMA 3.1 8B Instruct (Pre-trained Transformer), IA3 Adapter (Efficient Fine-tuning), DQN (RL Algorithm), Input Sequence (History of Interactions), and Output Q-value (Action-value Function) to demonstrate a meta-learning approach for solving unseen problems through reinforcement learning.
- ICRL leverages a pre-trained transformer fine-tuned with reinforcement learning to achieve in-context learning, enabling generalization to new environments and tasks without additional training.
- The framework exhibits robustness to low-quality training data and adaptability to non-stationary environments, highlighting its potential as a general-purpose problem solver.


---

[Self-reflecting Large Language Models: A Hegelian Dialectical Approach](http://arxiv.org/abs/2501.14917v1)

- Hegelian Dialectical Approach: Framework introduces philosophical approach inspired by the Hegelian Dialectic for LLMs' self-reflection.
- It uses self-dialectical approach to emulate internal critiques, synthesize new ideas by resolving contradictions, dynamic annealing approach for temperature generation, Multi Agent Majority Voting (MAMV) strategy to assess validity and novelty.
- Framework is examined to determine ability to generate novel ideas and provide stepping stone for future research.


---

[MedAgentBench: Dataset for Benchmarking LLMs as Agents in Medical Applications](http://arxiv.org/abs/2501.14654v1)

- MedAgentBench: is a broad evaluation suite designed to assess the agent capabilities of large language models within medical records contexts.
- It encompasses 100 patient-specific clinically-derived tasks, realistic profiles of 100 patients with over 700,000 data elements, a FHIR-compliant interactive environment, and an accompanying codebase.
- This framework establishes a valuable benchmark for model developers to track progress and drive continuous improvements in the agent capabilities of large language models within the medical domain.

---

[DEEPFLOW: Serverless Large Language Model Serving at Scale](http://arxiv.org/abs/2501.14417v1)

- DEEPFLOW: is a serverless AI platform designed for efficient large language model serving at scale.
- It uses request-job-task model, FLOWSERVE serving engine, NPU-centric execution, SPMD-based parallelism, and novel scheduling policies.
- This framework addresses resource allocation, serving efficiency, and cold start latencies.

---

[DRESSING UP LLM: EFFICIENT STYLIZED QUESTION-ANSWERING VIA STYLE SUBSPACE EDITING](http://arxiv.org/abs/2501.14371v1)

- DRESS (Disentangling Representation Editing in Style Subspace): is a novel approach for generating stylized large language model (LLM) responses through representation editing.
- It leverages over-parameterized nature of LLMs, disentangles style-relevant subspace, applies adaptive editing strengths, and maintains stylistic fidelity and semantic integrity.
- DRESS is a lightweight, train-free solution for enhancing LLMs with flexible and effective style control, making it useful for developing stylized conversational agents.


---

[Exploring the sustainable scaling of Al dilemma: A projective study of corporations' Al environmental impacts](http://arxiv.org/abs/2501.14334v1)

- The proposed methodology: estimates the environmental impact of a company's AI portfolio, providing actionable insights without extensive AI and Life-Cycle Assessment (LCA) expertise.
- The framework includes four interconnected models: life cycle impacts of primary components, life cycle impacts of AI use cases, AI company portfolio model, and 2030 AI landscape projections.
- This framework empowers organizations to understand and project their AI impacts and align their initiatives with global sustainability goals.


---

[MASTER: A Multi-Agent System with LLM Specialized MCTS](http://arxiv.org/abs/2501.14304v1)

- MASTER (Multi-Agent System with Tactical Execution and Reasoning using LLM Specialized MCTS): is a novel multi-agent framework that employs a new agent recruitment process and communication protocol based on the MCTS algorithm.
- It autonomously adjusts the number of agents based on task complexity, mitigates distractions and token window shortage, and includes a modified MCTS tailored to LLM scenarios.
- This framework achieves state-of-the-art performance on HotpotQA and WebShop datasets.


---

[Top Ten Challenges Towards Agentic Neural Graph Databases](http://arxiv.org/abs/2501.14224v1)

- Agentic NGDB (Agentic Neural Graph Databases): extends NGDBs with autonomous query construction, neural query execution, and continuous learning.
- It identifies ten key challenges, including semantic unit representation, abductive reasoning, scalable query execution, and integration with foundation models like LLMs.
- This framework enables intelligent, self-improving systems for modern data-driven applications.


---

[Serving Long-Context LLMs at the Mobile Edge: Test-Time Reinforcement Learning-based Model Caching and Inference Offloading](http://arxiv.org/abs/2501.14205v1)

- T2DRL (Test-Time Deep Reinforcement Learning): is a joint model caching and inference offloading framework that optimizes deployment and execution strategies for long-context LLM serving.
- Framework analyzes performance convergence, designs optimization problem considering context windows, manages cached models and service requests, adapts to context changes, and uses double Dutch auction mechanism for resource allocation.
- The framework reduces system costs while guaranteeing the performance of LLM agents in real-world perception and reasoning tasks.


---

[Distributed Multi-Agent Coordination Using Multi-Modal Foundation Models](http://arxiv.org/abs/2501.14189v1)

- VL-DCOPs (visual-linguistic instruction-based DCOPs): is a framework that uses large multimodal foundation models to generate constraints from visual and linguistic instructions.
- Framework includes spectrum of agent archetypes, from neuro-symbolic to fully neural agents, and evaluates them using LLMs and VLMs on novel VL-DCOP tasks.
- This work extends the DCOP literature by addressing the challenge of manual problem construction and opens new research directions.


---

[AI Chatbots as Professional Service Agents: Developing a Professional Identity](http://arxiv.org/abs/2501.14179v1)

- LAPI (LLM-based Agent with a Professional Identity): is a novel framework for designing professional service agents tailored for medical question-and-answer services.
- LAPI includes theory-guided task planning process, pragmatic entropy method, and iterative updating of responses.
- This framework improves response quality, providing more accurate, empathetic, and professional answers compared to baseline approaches.


---

[ARGOS: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models](http://arxiv.org/abs/2501.14170v1)

- ARGOS: is an agentic system for detecting time-series anomalies in cloud infrastructure by leveraging large language models (LLMs).
- It uses explainable anomaly rules as intermediate representation, employs LLMs to autonomously generate rules, and includes detection-, repair- and review-agents.
- This framework improves anomaly detection accuracy and efficiency compared to state-of-the-art methods.


---

[Top Ten Challenges Towards Agentic Neural Graph Databases](https://arxiv.org/abs/2501.14224)

- Agentic NGDB (Agentic Neural Graph Databases): extends NGDBs with autonomous query construction, neural query execution, and continuous learning.
- It identifies ten key challenges, including semantic unit representation, abductive reasoning, scalable query execution, and integration with foundation models like LLMs.
- This framework enables intelligent, self-improving systems for modern data-driven applications.


---

#### 23rd of January 2025


[BEYOND THE SUM: UNLOCKING AI AGENTS POTENTIAL THROUGH MARKET FORCES](https://arxiv.org/abs/2501.10388)

- AI Agent Market Infrastructure Framework presents systematic analysis of infrastructure requirements for AI agents to function as autonomous participants in digital markets.
- Framework identifies key areas like identity, service discovery, interfaces and payment systems and highlights existing infrastructure challenges impeding agent participation, suggesting new economic organization forms.
- This framework is important as it addresses infrastructure challenges as fundamental step toward enabling new forms of economic organization.


---

[ElCopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents](http://arxiv.org/abs/2501.13746v1)

- EICopilot: is a novel agent-based solution enhancing search and exploration of enterprise registration data within extensive online knowledge graphs.
- EICopilot includes data pre-processing pipeline, comprehensive reasoning pipeline with Chain-of-Thought and In-context learning, and novel query masking strategy.
- EICopilot is a groundbreaking tool for exploration and exploitation of large-scale knowledge graphs for enterprise information search.


---

[The though process behind Kimi k1.5](https://twitter.com/Kimi_Moonshot/status/1882413059513471044)

- Explains the way the Kimi K-1.5 model was trained and discusses overall likely o1-model training procedure.


---

[Operator System Card](https://cdn.openai.com/operator_system_card.pdf)

- OA Operator-agent system card.
- Uses RL.
- Additional [details](https://cdn.openai.com/cua/CUA_eval_extra_information.pdf)


#### 21st of January 2025

[LLM-Agents Driven Automated Simulation Testing and Analysis of small Uncrewed Aerial Systems](http://arxiv.org/abs/2501.11864v1)

- AUTOSIMTEST: is a Large Language Model (LLM)-driven framework, where multiple LLM agents collaborate to support the sUAS simulation testing process.
- Framework includes scenario generation-, mission-, environment- and analytics-agents; uses RAG approach; provides interactive analysis interface.
- Framework improves efficiency and scope of sUAS testing process, allowing for more comprehensive and varied scenario evaluations while reducing manual effort.


---

[EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents](http://arxiv.org/abs/2501.11858v1)



- EMBODIEDEVAL: is a comprehensive and interactive evaluation benchmark for MLLMs with embodied tasks.
- EMBODIEDEVAL features 328 distinct tasks within 125 varied 3D scenes, covers navigation, object interaction, social interaction, attribute question answering, and spatial question answering.
- This framework provides insights for future development of MLLMs in embodied capabilities.


---



#### 20th of January 2025

[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinformcent Learning](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)

- DeepSeek-R1: Trains SOTA-level Large Reasoning Model from LLM via Reinforcement Learning, which matches performance with o1-model.

---

[Kimi-K1.5: Scaling Reinforcement Learning with LLMs](https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf)

- Kimi k1.5: is a multi-modal large language model (LLM) trained with reinforcement learning (RL) to achieve SOTA-level reasoning performance across multiple benchmarks and modalities.


---

[Conversation Routines: A Prompt Engineering Framework for Task-Oriented Dialog Systems](http://arxiv.org/abs/2501.11613v1)

- Conversation Routines (CR): is a structured prompt engineering framework for developing task-oriented dialog systems using Large Language Models (LLMs).
- CR enables development of Conversation Agentic Systems (CAS) through natural language specifications, embedding task-oriented logic within LLM prompts, providing systematic methodology for designing complex conversational workflows while maintaining behavioral consistency.
- This framework enables domain experts to design conversational workflows in natural language while leveraging custom enterprise functionalities.


---

[Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training](http://arxiv.org/abs/2501.11425v1)

- Agent-R: is an iterative self-training framework that enables language agents to reflect on the fly.
- It leverages Monte Carlo Tree Search (MCTS) to construct training samples, recovers correct trajectories from erroneous ones, and uses a model-guided critique construction mechanism for timely revision.
- This framework effectively equips agents to identify and correct erroneous actions while avoiding loops, achieving superior performance.


---

[Towards Advancing Code Generation with Large Language Models: A Research Roadmap](http://arxiv.org/abs/2501.11354v1)



- Six-layer vision framework: categorizes code generation process into Input, Orchestration, Development, and Validation phases.
- Framework includes analysis of existing studies, outlines vision workflow, and systematically analyses challenges faced by LLMs.
- This work provides guidelines for improving reliability, robustness and usability of LLM-based code generation systems.


---

[Large Language Model Agents for Radio Map Generation and Wireless Network Planning](http://arxiv.org/abs/2501.11283v1)

- LLM agent framework: automates radio map generation and wireless network planning tasks.
- Framework includes tools-, models- and profiles-modules; it uses short-term and long-term memory; it performs task planning.
- The framework reduces manual operations and enhances network coverage and signal-to-interference-noise ratio.


---

[Code Readability in the Age of Large Language Models: An Industrial Case Study from Atlassian](http://arxiv.org/abs/2501.11264v1)

- HULA (Human-in-the-loop software development agents framework): is a LLM-based framework for software development.
- The framework uses GPT-4, compares LLM-generated code with human-written code, and evaluates code readability using static analysis metrics.
- This study highlights the importance of code readability in the age of LLMs and shows that LLM-generated code can be comparable to human-written code.


---

[PlotEdit: Natural Language-Driven Accessible Chart Editing in PDFs via Multimodal LLM Agents](http://arxiv.org/abs/2501.11233v1)

- PlotEdit: is a multi-agent framework for natural language-driven end-to-end chart image editing via self-reflective LLM agents.
- Framework includes Chart2Table, Chart2Vision, Chart2Code, Instruction Decomposition and Multimodal Editing agents; uses multimodal feedback to maintain visual fidelity; outperforms existing baselines on ChartCraft dataset.
- It enhances accessibility for visually challenged users and improves novice productivity.


---

#### 19th of January 2025

[IntellAgent: A Multi-Agent Framework for Evaluating Conversational AI Systems](http://arxiv.org/abs/2501.11067v1)

- IntellAgent: is a scalable, open-source multi-agent framework designed to evaluate conversational AI systems.
- It automates synthetic benchmark creation using policy-driven graph modeling, realistic event generation, and interactive user-agent simulations, providing fine-grained diagnostics.
- This framework enables comprehensive evaluation of conversational AI by addressing limitations of traditional methods.


---

[GREEN-CODE: Optimizing Energy Efficiency in Large Language Models for Code Generation](http://arxiv.org/abs/2501.11006v1)

- GREEN-CODE: is a framework for energy-aware code generation in LLMs, performing dynamic early exit during inference.
- It uses Reinforcement Learning agent to balance accuracy, latency, and energy consumption trade-offs, and fine-tunes models with weighted aggregated loss.
- This framework reduces energy consumption significantly without affecting accuracy for code generation tasks.


---

[Open FinLLM Leaderboard: Towards Financial AI Readiness](http://arxiv.org/abs/2501.10963v1)

- Open FinLLM Leaderboard: is an open platform for assessing and comparing Large Language Models' performance on financial tasks.
- The framework includes a leaderboard, demos, and financial AI readiness components; it uses zero-shot evaluation, and provides side-by-side model comparisons.
- This framework is important for encouraging innovation and improving model effectiveness in the financial sector.


---


[Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments](http://arxiv.org/abs/2501.10893v1)

- LEARN-BY-INTERACT: is a data-centric framework to adapt LLM agents to any given environments without human annotations.
- LEARN-BY-INTERACT synthesizes agent-environment interactions based on documentations, constructs instructions by summarizing interaction histories, and uses innovative retrieval approaches optimized for agents.
- This framework serves as a foundation for agent data synthesis as LLMs are increasingly deployed at real-world environments.


---

#### 18th of January 2025

[Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments](https://arxiv.org/abs/2501.10893)

- LEARN-BY-INTERACT: is a data-centric framework to adapt LLM agents to any given environments without human annotations.
- Framework synthesizes agent-environment interaction trajectories, uses backward construction for instructions, and leverages synthetic data for training and in-context learning with optimized retrieval.
- Framework serves as a foundation for agent data synthesis for LLMs in real-world environments.


--

[BAP v2: An Enhanced Task Framework for Instruction Following in Minecraft Dialogues](http://arxiv.org/abs/2501.10836v1)

- BAP v2 (Builder Action Prediction v2): is an upgraded task framework for instruction following in Minecraft dialogues.
- BAP v2 includes enhanced evaluation benchmark with cleaner test set and fairer metrics, and additional synthetic training data generated from novel Minecraft dialogue and target structure simulators.
- BAP v2 enables more efficient and meaningful progress on the task of instruction following in Minecraft dialogues.


---

[ML-SceGen: A Multi-level Scenario Generation Framework](http://arxiv.org/abs/2501.10782v1)

- ML-SceGen: is a three-stage framework for generating comprehensive and critical scenarios in autonomous driving.
- It uses LLM agents for parsing, Answer Set Programming (ASP) solver for logical traffic generation, and LLM for parameter updates to increase criticality.
- This framework enhances controllability, scalability, and realism in scenario generation for autonomous driving systems.


---

#### 17th of January 2025

[Evolving Deeper LLM Thinking](https://arxiv.org/abs/2501.09891)

- Mind Evolution: is an evolutionary search strategy that uses a language model to generate, recombine and refine candidate responses.
- It avoids formalizing the inference problem (so is usable in spaces like planning in natural language without explicit formalization of the problem and as well in hiding encoded message inside poems, which is non-natural language task), uses a global solution evaluator (focuses on domains, where evaluator is available), and can be easily parallelized.
- This approach significantly outperforms other inference strategies in natural language planning tasks.
- Introduces new StegPoet-benchmark, where the benchmark task is to encode message inside essay/story. 


---

[Agent4Edu: Generating Learner Response Data by Generative Agents for Intelligent Education Systems](https://arxiv.org/abs/2501.10332v1)

- Agent4Edu: is a personalized learning simulator that uses LLM-powered generative agents to simulate human learners' response data.
- It includes learner profile, memory, and action modules; interacts with personalized learning environments; evaluates and improves intelligent tutoring algorithms.
- This framework provides a versatile platform for comprehensive evaluations and future collection of valuable learner response data.


---

[Towards Human-Guided, Data-Centric LLM Co-Pilots](http://arxiv.org/abs/2501.10321v1)


- CliMB-DC (Clinical predictive Model Builder with Data-Centric AI): is a human-guided, data-centric framework for LLM co-pilots.
- It includes a multi-agent reasoning system with a strategic coordinator and a specialized worker agent, integrates state-of-the-art data-centric tools, and uses a human-in-the-loop approach.
- This framework empowers domain experts to actively participate in driving real-world impact using ML.


---

[Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling](http://arxiv.org/abs/2501.10316v1)

- Accountability Model: is an augmented LLM with an additional accountability head, functioning as a binary classifier to predict dialogue state slots.
- It detects false positives and negatives, guides LLM decoder for accurate actions, enables self-correction, and introduces friction to prevent overreliance.
- This model improves joint goal accuracy and overall performance in task-oriented dialogue systems.


---


[PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120v1)

- PaSa: is an advanced paper search agent powered by large language models. Available [https://pasa-agent.ai/](https://pasa-agent.ai/)
- It autonomously makes decisions, including invoking search tools, reading papers, and selecting references; it is optimized using reinforcement learning with synthetic dataset; it outperforms existing baselines on real-world academic queries.
- This framework significantly improves the efficiency and accuracy of academic search.


---

[LLM Reasoner and Automated Planner: A new NPC approach](http://arxiv.org/abs/2501.10106v1)



- LLM Reasoner and Automated Planner: is a novel architecture that integrates an LLM for decision-making with a classical automated planner.
- Framework uses LLM to decide goal, then uses automated planning to create plan, and includes modules for reasoning, planning and interface.
- This framework aims to empower autonomous agents with flexibility to adapt to any situation while maintaining plausible and human-like behavior.


---

[A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling, Search Algorithms, and Relevant Frameworks](http://arxiv.org/abs/2501.10069v1)

- This survey provides a comprehensive technical review that unifies task definitions and provides modular definitions of LLM profiling and search procedures.
- It enables precise comparisons of various LLM inference frameworks, highlights their departures from conventional search algorithms, and discusses applicability, performance, and efficiency.
- This survey offers a collection of classical and reusable implementations that can serve as solid foundations for future research and development.


---

[Agent-as-Judge for Factual Summarization of Long Narratives](http://arxiv.org/abs/2501.09993v1)

- NARRATIVEFACTSCORE: is a novel "Agent-as-a-Judge" framework for evaluating and refining summaries.
- It leverages Character Knowledge Graph (CKG), assesses factual consistency, provides actionable guidance for refinement, identifies missing or erroneous facts, and uses retrieval-based verification with explicit feedback.
- This framework improves the factual reliability of LLM-generated summaries.

---

[A Survey on Multi-Turn Interaction Capabilities of Large Language Models](http://arxiv.org/abs/2501.09959v1)

- This survey provides a focused review of the multi-turn capabilities of LLMs.
- The survey explores core model capabilities, evaluation methods, enhancement algorithms, and future research directions.
- This survey is important for both academic researchers and industry practitioners.


---


[TOWARDS A LITMUS TEST FOR COMMON SENSE](http://arxiv.org/abs/2501.09913v1)

- Axiomatic litmus test: diagnoses common sense by combining minimal prior knowledge constraints with diagonal arguments to create tasks beyond the agent's known concept set.
- It addresses deceptive hallucinations, integrates observations regarding emergent deceptive hallucinations, and uses Abstraction and Reasoning Corpus (ARC) constraints.
- This test provides a stepping stone toward an ethical, reliable foundation for future safe, beneficial and aligned artificial intelligence.


---



#### 16th of January 2025

[Authenticated Delegation and Authorized AI Agents](https://arxiv.org/abs/2501.09674)

- Authenticated Delegation Framework: novel framework enables authenticated, authorized, and auditable delegation of authority to AI agents.
- Secure delegation; restrict permissions and scope; accountability; extends OAuth 2.0 and OpenID Connect; natural language to auditable access control.
- Framework facilitates immediate AI agent deployment while ensuring security and accountability.


---

[Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/abs/2501.09732)

- Inference-time scaling framework: explores the inference-time scaling behavior of diffusion models beyond increasing denoising steps.
- Framework uses search problem to identify better noises, design space includes verifiers and algorithms, experiments on class-conditioned and text-conditioned image generation benchmarks.
- This framework reveals that increasing inference-time compute leads to substantial improvements in the quality of samples generated by diffusion models.


---


[Foundations of Large Language Models](https://arxiv.org/pdf/2501.09223)

- Introduces a literature review / survey on LLMs.
  

---

[AutoCBT: An Autonomous Multi-agent Framework for Cognitive Behavioral Therapy in Psychological Counseling]( http://arxiv.org/abs/2501.09426v1)
- AutoCBT: An Autonomous Multi-agent Framework for Cognitive Behavioral Therapy in Psychological Counseling.
- AutoCBT incorporates a counsellor agent and multiple supervisor agents, uses short-term and long-term memory, and is evaluated on a bilingual dataset.
- AutoCBT leverages dynamic routing and supervisory mechanisms to offer high-quality, automated CBT services, enhancing the effectiveness of single-turn consultations.

---

[OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking](https://arxiv.org/abs/2501.09751)

- OmniThink: is a machine writing framework that emulates human-like iterative expansion and reflection.
- It uses continuous reflection and exploration, attaches knowledge to an information tree, and extracts it into a conceptual pool to deepen understanding.
- This framework improves the knowledge density of generated articles without compromising coherence and depth.


---

[CyberMentor: AI Powered Learning Tool Platform to Address Diverse Student Needs in Cybersecurity Education](http://arxiv.org/abs/2501.09709v1)

- CyberMentor: is a learning tool platform designed to address diverse needs of cybersecurity students using agentic workflow and Generative Large Language Models (LLMs).
- It leverages Retrieval-Augmented Generation (RAG) for accurate information retrieval, includes knowledge base, skill base and LLM agent, and provides personalized learning experiences.
- This framework aims to improve equity and sustainability in higher education by offering open-source design for adaptation across disciplines.


---

[Empowering Large Language Models in Wireless Communication: A Novel Dataset and Fine-Tuning Framework](http://arxiv.org/abs/2501.09631v1)

- PVI (Pointwise V-Information) based fine-tuning method: enhances LLMs for wireless communication by quantifying information content of training data.
- Dataset includes multi-hop questions, true/false and multiple-choice types, varying difficulty levels, rigorous data curation, advanced language models for entity extraction and question generation.
- This work aims to improve LLM training and evaluation for wireless communication research and applications.


---

[SOP-AGENT: EMPOWER GENERAL PURPOSE AI AGENT WITH DOMAIN-SPECIFIC SOPS](http://arxiv.org/abs/2501.09316v1)

- SOP-agent (Standard Operational Procedure-guided Agent): is a novel framework for constructing domain-specific agents through pseudocode-style Standard Operational Procedures (SOPs) written in natural language.
- SOP-agent represents SOP as a decision graph, traverses it to guide the agent, conducts experiments across multiple domains, and introduces Grounded Customer Service Benchmark.
- SOP-agent demonstrates excellent versatility, achieving performance superior to general-purpose agent frameworks and comparable to domain-specific agent systems.


---


#### 15th of January 2025


[The geometry of moral decision making](https://arxiv.org/abs/2501.08865)

- Geometry of Moral Decision Making Framework: Understands bounded rationality as interplay of deontology and utilitarianism.
- Deontology as regularisation function in optimal control; Inverse temperature shields from expected utility; Information geometry of bounded rationality and rate distortion theory; Markov kernels and regular conditional probability; Gradient equation determines utility expansion path.
- Framework is relevant to theory of autonomous agents and analysis of legal doctrine.


---

[Networked Agents in the Dark: Team Value Learning under Partial Observability](https://arxiv.org/abs/2501.08778)

- DNA-MARL (Double Networked Averaging MARL) is distributed method for networked agents that introduces consensus mechanism for local communication and gradient descent for local computation in partially observable Markov games.
- Framework addresses cooperative multi-agent reinforcement learning in networked dynamic partially observable Markov game (ND-POMG) using decentralized training and decentralized execution (DTDE), and achieves team value function learning under partial observability via consensus mechanism for cooperative value function learning with actor-critic algorithm.
- DNA-MARL enhances the potential of networked agents for real-world applications requiring privacy and robustness to message loss.


---

[Between Puppet and Actor: Reframing Authorship in this Age of AI Agents](http://arxiv.org/abs/2501.15346v1)

- Puppet and Actor framework: This framework reframes authorship in the age of AI agents by positioning AI agency between puppet and actor.
- Conceptual tensions in AI agent roles; creative processes; Large Language Models (LLMs); Schmidt's categorization; classical authorship; puppet-actor spectrum; creative autonomy; dynamic state; evolving authorship.
- Understanding AI agency as puppet-actor spectrum is important for adapting authorship concepts in the age of AI.


---

[AGENTIC RETRIEVAL-AUGMENTED GENERATION: A SURVEY ON AGENTIC RAG](http://arxiv.org/abs/2501.09136v1)

- Introduces Survey on compherensive list of RAG-techniques with LLM-agents.


---


[Agent TCP/IP: An Agent-to-Agent Transaction System](https://arxiv.org/abs/2501.06243)

- ATCP/IP (Agent Transaction Control Protocol for Intellectual Property): introduces a trustless framework for exchanging IP between agents via programmable contracts.
- Framework enables agents to initiate, trade, borrow, and sell agent-to-agent contracts on the Story blockchain network, including legal wrappers for offchain enforcement, and facilitates autonomous selling of training data, licensing of information, and content collaboration.
- This framework is important for creating a standardized way for agents to negotiate and enter into agreements, forming a market for knowledge.


---

[Leveraging Large Language Models as Knowledge-Driven Agents for Reliable Retrosynthesis Planning](http://arxiv.org/abs/2501.08897v1)

- MBRPS (Multi-branched Reaction Pathway Search): Algorithm enabling exploration of all pathways, with a focus on multi-branched ones.
- Framework integrates LLMs and KGs, automates literature retrieval, reaction data extraction, database querying, and construction of retrosynthetic pathway trees, and recommends optimal routes.
- Attempt to develop a fully automated retrosynthesis planning agent tailored specially for macromolecules powered by LLMs.


---

[AutoRestTest: A Tool for Automated REST API Testing Using LLMs and MARL](http://arxiv.org/abs/2501.08600v1)

- AutoRestTest: is a novel tool that integrates Semantic Operation Dependency Graph (SODG) with Multi-Agent Reinforcement Learning (MARL) and Large Language Models (LLMs) for effective REST API testing.
- It uses five specialized agents for operation, parameter, value, dependency, and header identification, and employs LLMs for realistic input generation and a command-line interface for user interaction.
- This framework provides a comprehensive solution for thorough REST API evaluation and validation.


---


[Leveraging LLM Agents for Translating Network Configurations](http://arxiv.org/abs/2501.08760v1)

- IRAG (Intent-based Retrieval Augmented Generation): is an intent-based framework for translating network configurations using LLM agents.
- Framework includes intent extraction, manual retrieval, incremental translation, syntax verification and semantic verification modules.
- This framework achieves high syntax correctness and superior translation accuracy compared to state-of-the-art methods.


---

[DISENTANGLING EXPLORATION OF LARGE LANGUAGE MODELS BY OPTIMAL EXPLOITATION](http://arxiv.org/abs/2501.08925v1)

- Optimal Exploitation framework: isolates exploration as the sole objective by tasking the agent with delivering information that enhances future returns.
- Framework decomposes missing rewards into exploration and exploitation components, measures optimal achievable return for explored states, and provides insights into behaviors driven by agent instructions.


---

[Physical AI Agents: Integrating Cognitive Intelligence with Real-World Action](http://arxiv.org/abs/2501.08944v1)

- Physical AI Agents: is a framework that integrates cognitive reasoning with physical interaction for real-world tasks.
- Framework includes modular architecture with perception, cognition, and actuation blocks, and introduces Ph-RAG (Physical Retrieval Augmented Generation) design pattern for real-time decision-making.


---

[Doc-Guided Sent2Sent++: A Sent2Sent++ Agent with Doc-Guided memory for Document-level Machine Translation](http://arxiv.org/abs/2501.08523v1)

- Doc-Guided Sent2Sent++: is an agent that employs an incremental sentence-level forced decoding strategy for document-level machine translation.
- It uses Doc-Guided Memory with summary and its translation, ensures sentence completeness, enhances fluency, and improves translation quality.
- This approach addresses the limitations of other DocMT agents by maintaining both completeness and fluency.


---

[Evaluating GenAl for Simplifying Texts for Education: Improving Accuracy and Consistency for Enhanced Readability](http://arxiv.org/abs/2501.09158v1)


- GenAI (Generative Artificial Intelligence): framework evaluates the use of LLMs for text simplification in educational contexts.
- Framework uses three LLMs (GPT-4 Turbo, Claude 3, and Mixtral 8x22B), four prompting techniques (zero-shot, directional stimulus, chain-of-thought, and prompt chaining), and a novel multi-agent architecture; it assesses grade level accuracy, keyword accuracy, semantic similarity, and word count change.
- This study provides a rigorous evaluation of LLMs for automated text simplification, offering insights for educators and future research.


---

#### 14th of January 2025

#### 14th January 2025

[Governing AI Agents](https://arxiv.org/abs/2501.07913)

- Governance strategy: Governance strategy centered around inclusivity, visibility, and liability is proposed for designing and regulating AI agents.
- agency law and theory; principal-agent problems; information asymmetry, authority, loyalty, delegation; limitations of conventional solutions; new technical and legal infrastructure; governance principles.
- New technical and legal infrastructure is needed to support governance principles for reliable, safe, and ethical AI agents.


---

[Flow: A Modular Approach to Automated Agentic Workflow Generation](http://arxiv.org/abs/2501.07834v1)

- Flow: is a multi-agent framework that dynamically adjusts workflows using activity-on-vertex graphs.
- It refines workflows based on historical performance, emphasizes modularity, and achieves concurrent sub-task execution.
- This framework improves efficiency and adaptability in multi-agent systems through dynamic workflow updates.


---


[POKERBENCH: Training Large Language Models to become Professional Poker Players](https://arxiv.org/abs/2501.08328)

- POKERBENCH: is a benchmark for evaluating poker-playing abilities of large language models (LLMs).
- It includes 11,000 poker scenarios, covers pre-flop and post-flop play, and evaluates models like GPT-4, ChatGPT 3.5, Llama and Gemma series.
- This benchmark provides a quick and reliable way to evaluate LLMs in complex game-playing scenarios.

---

[A Multi-Agent Framework for Systematic Review Automation Using Large Language Models](https://arxiv.org/abs/2501.05468)

- LatteReview: Intrdocus LLM-based systematic literature review multi-agent framework automation, which consists of three layers: LM providers (local models / LLMs via api), Reviewer agents (with roles & expertise levels) and Workflows (support sequential, parallel review rounds, dynamic decision-making and iterative refinement).
- Includes BaseReviewer/ScoringReviewer/TitleAbstractReviewer/AbstractionReviewer/Custom reviewer-agents, which are used as modular agents for title and abstract screening, relevance scoring, and structured data extraction; agents operate within orchestrated workflows.
- Workflow module includes Concept of rounds / Chaining reviews / Parallel reviews and Dynamic filter.


---

[CodeCoR: An LLM-Based Self-Reflective Multi-Agent Framework for Code Generation](http://arxiv.org/abs/2501.07811v1)

- CodeCoR (Code Collaboration and Repair): is a self-reflective multi-agent framework for code generation.
- It includes prompt-, coding-, test- and repair-agents, uses pruning methods to evaluate agent effectiveness, and enhances self-reflective ability.
- It significantly outperforms existing state-of-the-art methods in code generation.


---


[Engineering LLM Powered Multi-agent Framework for Autonomous CloudOps](http://arxiv.org/abs/2501.08243v1)

- MOYA (Meta Orchestrator Of Your Agents): is a multi-agent framework leveraging GenAI for autonomous CloudOps, balancing automation with human control.
- Framework integrates internal and external systems, optimizes task orchestration, security, and error mitigation using Retrieval Augmented Generation (RAG), and includes LLM-based and non-LLM-based agents.
- The framework enhances accuracy, responsiveness, and effectiveness over non-agentic approaches across complex workflows.


---

[Agent-Centric Projection of Prompting Techniques and Implications for Synthetic Training Data for Large Language Models](http://arxiv.org/abs/2501.07815v1)

- Agent-Centric Projection: introduces a framework to reveal connections between prompting strategies and multi-agent systems.
- Framework uses linear and non-linear contexts to classify prompting techniques, and proposes three conjectures about the relationship between prompting and multi-agent systems.
- This framework enables cross-pollination of research findings between prompting and multi-agent domains, while providing new directions for improving both the design and training of future LLM systems.


---

[Talk to Right Specialists: Routing and Planning in Multi-agent System for Question Answering](http://arxiv.org/abs/2501.07813v1)

- RopMura: is a multi-agent system that incorporates a router and a planner for question answering across diverse knowledge domains.
- RopMura includes router for selecting relevant agents, planner for decomposing complex queries, and knowledge sovereignty consideration.
- This framework enables efficient and accurate multi-domain question-answering.


---

[Infecting Generative AI With Viruses](https://arxiv.org/abs/2501.05542)

- VLM/LLM (Vision-Large Language Model): framework tests security boundaries by embedding EICAR test file within JPEG images.
- Framework includes multiple LLM platforms, such as OpenAI GPT-40, Microsoft Copilot, Google Gemini 1.5 Pro, and Anthropic Claude 3.5 Sonnet; it demonstrates masking EICAR string, extracting test file, and using obfuscation techniques.
- This research extends penetration testing framework to evaluate cloud-based generative AI and LLM security boundaries.


---


[Visual Language Models as Operator Agents in the Space Domain](http://arxiv.org/abs/2501.07802v1)

- Explores the application of VLMs as operator agents in the space domain.
- Framework builds on LLMs and their multimodal extensions, investigates how VLMs enhance autonomous control and decision-making in space missions, includes software and hardware operational paradigms.
- This research demonstrates that VLMs can effectively process visual and textual data to generate contextually appropriate actions.


---

[ADAM-1: AI and Bioinformatics for Alzheimer's Detection and Microbiome-Clinical Data Integrations](http://arxiv.org/abs/2501.08324v1)

- ADAM-1 (Alzheimer's Disease Analysis Model Generation 1): is a multi-agent large language model framework designed to integrate and analyze multi-modal data.
- Framework uses retrieval-augmented generation techniques, multi-agent architecture, synthesizes insights from diverse data sources, contextualizes findings using literature-driven evidence, and is tailored for binary classification tasks.
- This framework demonstrates robustness and consistency, particularly in small laboratory datasets, and has potential for Alzheimer's research and diagnostics.


---

[ADDRESSING THE SUSTAINABLE AI TRILEMMA: A CASE STUDY ON LLM AGENTS AND RAG](http://arxiv.org/abs/2501.08262v1)

- Sustainable AI Trilemma: highlights the tensions between AI capability, digital equity, and environmental sustainability.
- Framework analyzes energy costs in memory module designs, introduces metrics for energy consumption and system performance trade-offs, challenges LLM-centric autonomy paradigm.
- This framework provides practical insights for developing more sustainable AI systems.


---

[Agent-Centric Projection of Prompting Techniques and Implications for Synthetic Training Data for Large Language Models](http://arxiv.org/abs/2501.07815v1)

- Agent-Centric Projection: introduces a framework to reveal connections between prompting strategies and multi-agent systems.
- Framework uses linear and non-linear contexts to classify prompting techniques, and proposes three conjectures about the relationship between prompting and multi-agent systems.
- This framework enables cross-pollination of research findings between prompting and multi-agent domains, while providing new directions for improving both the design and training of future LLM systems.


---

[ASTRID - An Automated and Scalable TRIaD for the Evaluation of RAG-based Clinical Question Answering Systems](http://arxiv.org/abs/2501.08208v1)

- ASTRID: is an Automated and Scalable TRIaD for evaluating clinical QA systems leveraging RAG.
- ASTRID includes three metrics: Context Relevance (CR), Refusal Accuracy (RA), and Conversational Faithfulness (CF); it is validated using real-world patient questions and clinician assessments; it is automatable using LLMs.
- ASTRID provides a valuable resource for further research and development of clinical QA systems.


---

[CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning](http://arxiv.org/abs/2501.08071v1)

- CuAsmRL: is an automatic optimizer for optimizing NVIDIA GPU SASS schedules using reinforcement learning.
- It formulates SASS optimization as an assembly game, integrates with OpenAI Triton, and improves performance of specialized CUDA kernels by up to 26%.
- This framework provides a way to automatically optimize GPU kernels, which is important for improving the performance of LLMs.

---

#### 13th of January 2025

[The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)


- PRM (Process Reward Model): A model for process supervision in mathematical reasoning of LLMs, which aims to identify and mitigate intermediate errors in the reasoning processes.
- Monte Carlo (MC) estimation, Best-of-N (BoN) evaluation, consensus filtering mechanism, response-level and step-level metrics, data efficiency, error identification.
- The paper addresses challenges in developing effective PRMs, offering solutions for data annotation, evaluation methodologies, and proposing a consensus filtering mechanism to enhance model performance and data efficiency.


---

[Evaluating Agent-based Program Repair at Google](https://arxiv.org/abs/2501.07531)

- Passerine: An agent-based program repair system designed to operate within Google's development environment.
- Inspired by SWE-Agent, utilizes ReAct-style loop, limited command set, Gemini 1.5 Pro, 20 trajectory samples, evaluates on GITS-Eval (178 bugs from Google's internal issue tracking system).
- Establishes a baseline for agent-based automated program repair performance on an industrially relevant benchmark, highlighting challenges and opportunities in an enterprise context.


---

[GPT as a Monte Carlo Language Tree: A Probabilistic Perspective](https://arxiv.org/abs/2501.07641)

- Reviews LLM as a Monte Carlo Language Tree (data tree), where each node is token, each edge is the token transition probability and each sequence has unique path.
- Any GPT LLM can be flattened into MCLT.
- Claims CoT attempts to find path between the input and output in the MCLT to connect them.


---

[WebWalker: Benchmarking LLMs in Web Traversal](https://arxiv.org/abs/2501.07572)

- WebWalker: is a multi-agent framework that mimics human-like web navigation through an explore-critic paradigm. 
- WebWalkerQA is a benchmark designed to assess the ability of LLMs to perform web traversal, it evaluates the capacity of LLMs to traverse a website's subpages to extract high-quality data systematically, and it focuses on text-based reasoning abilities.
- This work highlights the importance of deep, vertical exploration in web-based tasks.


---

[Imagine while Reasoning in Space: Multimodal Visualization-of-Thought](https://arxiv.org/abs/2501.07542)

- MVoT (Multimodal Visualization-of-Thought): is a multimodal native reasoning paradigm that generates image visualizations of reasoning traces.
- MVoT uses token discrepancy loss to improve visual coherence and fidelity, and is validated on dynamic spatial reasoning tasks, showing competitive performance.
- MVoT establishes new possibilities for complex reasoning tasks where visual thinking complements verbal reasoning.

---

[Understanding and Benchmarking Artificial Intelligence: OpenAI's 03 Is Not AGI](https://arxiv.org/abs/2501.07458)

- Claims, that ARC-AGI (Abstraction and Reasoning Corpus) is a benchmark proposed to measure intelligence, but not suitable for measuring progress towards AGI.
- ARC-AGI tasks represent a specific problem structure, which can be solved by massive trialling of predefined operations, and it does not require exploration, but only exploitation.
- A new benchmark is outlined that covers a much higher diversity of unknown tasks to be solved, to enable a comprehensive assessment of intelligence and of progress towards AGI.


---

[PoAct: Policy and Action Dual-Control Agent for Generalized Applications](https://arxiv.org/abs/2501.07054)

- PoAct (Policy and Action Dual-Control Agent): is a framework that dynamically adjusts action space and reasoning policy using a Policy Controller and Action Controller.
- PoAct includes a Policy Controller for switching between reasoning policies, and an Action Controller with RAG Selector and Action Reviewer for managing action space and reasoning paths; it is evaluated on LegalAgentBench and AgentBench datasets.
- PoAct achieves higher quality code actions and more accurate reasoning paths, while also reducing token consumption.


---

[Lifelong Learning of Large Language Model based Agents: A Roadmap](https://arxiv.org/abs/2501.07278)

- Introduces a s survey incorporating lifelong learning into LLM-based agents.
- Categorizes core components into perception-, memory-, and action-modules, highlights continuous adaptation, mitigates catastrophic forgetting, and improves long-term performance.


---

[How GPT LEARNS LAYER BY LAYER](https://arxiv.org/abs/2501.07108)

- Explores how LLMs build internal world models with OthelloGPT by using Sparse AutoEncoders.


---

[SST-EM: Advanced Metrics for Evaluating Semantic, Spatial and Temporal Aspects in Video Editing](https://arxiv.org/abs/2501.07554)

- SST-EM (Semantic, Spatial, and Temporal Evaluation Metric): is a benchmark for video editing that leverages VLMs, object detection, and temporal consistency checks.
- SST-EM includes semantic extraction using VLM, primary object tracking with object detection, focused object refinement via LLM agent, and temporal consistency assessment using ViT.
- This framework provides a comprehensive evaluation of semantic fidelity and temporal smoothness in video editing.


---

[PoAct: Policy and Action Dual-Control Agent for Generalized Applications](http://arxiv.org/abs/2501.07054v1)

- PoAct (Policy and Action Dual-Control Agent): is a framework that dynamically adjusts action space and reasoning policy by switching between different reasoning policies and managing action space.
- PoAct includes Policy Controller for high-quality planning and coding, and Action Controller with RAG Selector and Action Reviewer for managing action space and reasoning paths; it is evaluated on multiple datasets with commercial and open-source large models.
- PoAct achieves higher-quality code actions and more accurate reasoning paths, demonstrating strong generalizability and scalability.


---

[Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability](https://arxiv.org/abs/2411.19943)

- cDPO (critical Direct Preference Optimization): is a novel framework for identifying and penalizing critical tokens in mathematical reasoning tasks.
- It uses rollout sampling to identify critical tokens, contrastive estimation to pinpoint them efficiently, and token-level rewards for preference optimization.
- This framework significantly improves model accuracy in mathematical reasoning tasks by reducing errors.


---


#### 12th of January 2025

[Eliza: A Web3 friendly AI Agent Operating System](https://arxiv.org/abs/2501.06781)

- Eliza: The first open-source, web3-friendly, agentic framework that makes the deployment of web3 applications effortless.
- Typescript program, seamless web3 integration, stable performance, key runtime components, community-driven, modular design, multi-agent simulation.
- Eliza bridges the gap between AI and web3, offering a platform for decentralized AI applications.


---

[DVM: Towards Controllable LLM Agents in Social Deduction Games](https://arxiv.org/abs/2501.06695)

- DVM (Dynamic Victory Manager): is a framework for controllable LLM agents in social deduction games, comprising Predictor, Decider, and Discussor components.
- It uses reinforcement learning with a win rate-constrained decision chain reward mechanism, enabling agents to dynamically adjust their gameplay proficiency, and it is evaluated in the Werewolf game.
- DVM enables adaptive and balanced gameplay in social deduction games, opening new research avenues for controllable game agents.


---

[LLMs Model Non-WEIRD Populations: Experiments with Synthetic Cultural Agents](https://arxiv.org/abs/2501.06834)

- Synthetic Cultural Agents (SCAs): uses LLMs to create synthetic agents representing non-WEIRD populations. Includes web scraping, LLMs, RAG prompting to construct cultural profiles and uses these agents to classic behavioral experiments, demonstrating cross-cultural variability.
- Offers an effective and ethical method to pilot experiments and refine protocols for hard-to-reach populations for cross-cultural economic studies.


---

[AIOPSLAB: A HOLISTIC FRAMEWORK TO EVALUATE AI AGENTS FOR ENABLING AUTONOMOUS CLOUDS](https://arxiv.org/abs/2501.06706)

- AIOPSLAB: is a framework that deploys microservice cloud environments, injects faults, generates workloads, exports telemetry data, orchestrates components, and provides interfaces for interacting with and evaluating agents.
- AIOPSLAB includes Agent-Cloud Interface (ACI), a unified interface for agent-cloud interaction, and supports evaluation of LLM-based agents with a benchmark suite of 48 problems across different AIOps tasks.
- AIOPSLAB provides a holistic approach to evaluate AIOps agents in complex cloud environments, addressing the limitations of existing benchmarks.


---


#### 11th of January 2025

[The Internet of Large Language Models](https://arxiv.org/abs/2501.06471)

- The Internet of LLM: introduces an universal environment and sharing protocol of LLM training/knowledge exchange, which consists of LLM sharing protocol/LLM Universal environment/Agent Optimal Path Module/joint mining mechanism.
- Includes also planning-, reflection- and tool use-agents.


---

[Guided Code Generation with LLMs: A Multi-Agent Framework for Complex Code Tasks](https://arxiv.org/abs/2501.06625)

- Guided code generation: introduces a multi-agent framework for complex code tasks, which includes hierarchical decomposition, bottom-up code generation, and multi-agent validation.
- Leverages LLMs as fuzzy searchers and information retrievers. Mitigates LLM weaknesses in long sequential reasoning and context understanding.
- This framework enhances code generation capabilities and overcomes limitations of LLMs in compositional reasoning and context handling.


---

#### 10th of January 2025

[BioAgents: Democratizing Bioinformatics Analysis with Multi-Agent Systems](https://arxiv.org/abs/2501.06314)

- BioAgents: is a multi-agent system designed to assist users in bioinformatics pipeline design, development, and troubleshooting. which includes two specialized agents and a reasoning agent.
- First specialized agent was fine tuned with conceptual genomics tasks and the second specialized agent uses RAG related to workflow documentation.
- Reasoning agent uses self-ratings / threshold.
- Achieves performance comparable to human experts on conceptual genomics tasks. 


--- 


[Multi-Agent Collaboration Mechanisms: A Survey of LLMs](https://arxiv.org/abs/2501.06322)

- The survey reviews Multi-Agent Systems (MASs) collaboration mechanisms based on key dimensions.
- Framework includes actors, types, structures, strategies, and coordination protocols; reviews existing methodologies; investigates applications across diverse domains; identifies key lessons, open challenges, and potential research directions.


---

[How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond](https://arxiv.org/abs/2501.05714)

- Human-Model Cooperation: is a survey of principles, formalizations, and open challenges in human-model cooperation.
- It introduces a new taxonomy for categorizing human-model cooperation, identifies key research frontiers, and discusses associated challenges.


---


[OpenFOAMGPT: a RAG-Augmented LLM Agent for OpenFOAM-Based Computational Fluid Dynamics](https://arxiv.org/abs/2501.06327)

- OpenFOAMGPT: LLM-based agent tailored for OpenFOAM-centric computational fluid dynamics (CFD) simulations.
- It leverages GPT-4 and a chain-of-thought (CoT)-enabled o1 preview model, uses retrieval-augmented generation (RAG) pipeline, and includes an iterative correction loop.


---



#### 9th of January 2024


[Search-01: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366)

- Search-01: is a framework that enhances Large Reasoning Models (LRMs) with an agentic retrieval-augmented generation mechanism and a Reason-in-Documents module.
- It integrates an agentic search workflow, enables dynamic retrieval of external knowledge, and uses a separate module to analyze retrieved information.
- This approach enhances the trustworthiness and applicability of LRMs in complex reasoning tasks.


---

[OpenOmni: Large Language Models Pivot Zero-shot Omnimodal Alignment across Language with Real-time Self-Aware Emotional Speech Synthesis](https://arxiv.org/abs/2501.04561)

- OpenOmni: Introduces three-stage training method combining speech-to-text generation/image-to-text generation/speech generation, which results SOTA-level omnimodal LLM.


---

[Emergence of human-like polarization among large language model agents](https://arxiv.org/abs/2501.05171)

- Introduces a networked system, which simulates social interactions of thousands of LLM-based agents, including capabilities of establishing social relationships, communicating, and forming opinions on political issues. LLM agents form spontaneously human-like social networks (echo chamber).
- LLM agents exhibit human-like polarization and can be used to study interventions, offering insights into managing polarization in real-world scenarios.
- Self-regulation helps to reduce inconsistencies in the opinions, which leads to more balanced polarization patterns. Openmindedness and diverse interaction limit polarization effect.


---

[NSChat: A Chatbot System To Rule Them All](https://arxiv.org/abs/2501.05541)

- NSChat: introduces a web-based chatbot system designed for neuroscience research.
- NSChat is built using React framework, it is customizable, flexible, and allows integration of various LLMs, it also includes a logging mechanism for user interactions.


---

[Emergence of human-like polarization among large language model agents](https://arxiv.org/abs/2501.05171)

- LLM (Large Language Model) agents framework: simulates a networked system of agents that establish social relationships, communicate, and form opinions on political issues.
- Framework includes self-expression, communication, and opinion update stages; agents develop human-like polarization, homophilic clustering, and echo chamber effects; self-regulation strategy reduces self-inconsistency.
- This framework provides a valuable platform for exploring strategies to mitigate polarization and promote inclusive political conversations.


---


[LearningFlow: Automated Policy Learning Workflow for Urban Driving with Large Language Models](https://arxiv.org/abs/2501.05057)

- LearningFlow: is an automated policy learning workflow for urban driving that uses multiple LLM agents.
- It includes curriculum sequence generation and reward generation processes, supported by analysis agents, and enhances sample efficiency.
- This framework automates policy learning across complex driving tasks and reduces reliance on manual reward function design.


---

[OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](https://arxiv.org/abs/2501.05510)

- OVO-Bench (Online-VideO-Benchmark): is a novel video benchmark for evaluating online video understanding capabilities of Video-LLMs.
- It includes 644 videos, 2800 meta-annotations, and 12 tasks across three categories: Backward Tracing, Real-Time Visual Perception, and Forward Active Responding.
- This benchmark highlights the importance of temporal awareness for advanced online video understanding.


---

#### 9th of January 2025

[Transformer-Squared: Self-adaptive LLMs](https://arxiv.org/abs/2501.06252)

- Transformer<sup>2</sup>: A self-adaptation framework that adapts LLMs (Large Language Models) for unseen tasks in real-time by selectively adjusting the singular components of their weight matrices.
- Two-pass mechanism, task-specific expert vectors, reinforcement learning, dynamic mixing, targeted behavior, outperforming LoRA, fewer parameters, greater efficiency, versatility across different LLM architectures and modalities.
- Represents a significant leap forward, offering a scalable, efficient solution for enhancing the adaptability and task-specific performance of LLMs, paving the way for truly dynamic, self-organizing AI systems.


---

[On Corrigibility and Alignment in Multi Agent Games](https://arxiv.org/abs/2501.05360)

- Multi Agent Corrigibility Games: introduces a framework for studying corrigibility in systems comprised of multiple autonomous agents.
- Framework models a 2-player game with human supervision, uses Bayesian games to introduce uncertainty over human beliefs, and analyzes specific cases like two-player corrigibility and adversary settings.
- This framework provides insights into designing corrigible multi-agent systems, even in the face of human irrationality.


---


#### 8th of January 2025

[rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://arxiv.org/abs/2501.04519)

- rStar-Math: A framework demonstrating that small language models (SLMs) can rival or surpass the math reasoning capability of OpenAI models through deep thinking. Iteratively improves through self-evolution generating millions of new math reasoning trajectories in each round.
- Uses Monte Carlo Tree Search (MCTS) with self-annotated Q-values. rStar-Math used 747k math word problems, took the final correct answer and then rolled out 16 MCTS-based step-by-step verified reasoning trajectories, to categorize problems by difficulty level (easy/medium/hard) based on ratio of correct solutions. Hard problems are assigned with an additional extra 16 rollouts. The policy SLM is trained using all the step-by-step trajectories with their Q-values.
- The importance of this work lies in showing that smaller language models can achieve state-of-the-art math reasoning, rivaling larger models, through a novel self-evolutionary process.
- Includes Code-Augmented CoT, where step-by-step reasoning trajectories generated are verified with code execution for correctness.


---


[Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought](https://arxiv.org/abs/2501.04682)

- Meta-CoT (Meta Chain-of-Thought): A novel framework that extends traditional CoT by explicitly modeling the underlying reasoning process required to arrive at a particular CoT.
- Inspired by Cognitive Science's dual-process theory, non-linear, iterative, latent process of exploration and verification, in-context search, process supervision, synthetic data generation, search algorithms, instruction tuning, reinforcement learning, scaling laws, verifier roles, novel reasoning algorithms, meta-reinforcement learning.
- This work provides a theoretical and practical roadmap to enable Meta-CoT in LLMs, paving the way for more powerful and human-like reasoning in artificial intelligence.


---

[URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics](https://arxiv.org/abs/2501.04686)

- URSA (Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics): A framework for enhancing the mathematical reasoning capabilities of Multimodal Large Language Models (MLLMs) through a three-module synthesis strategy and a novel dual-view process supervision data synthesis method.
- Integrates CoT distillation, trajectory-format rewriting, format unification, MMathCoT-1M dataset, DualMath-1.1M dataset, URSA-7B model, URSA-RM-7B model, test-time scaling, process annotation, out-of-distribution (OOD) verification.
- This work significantly enhances MLLMs' potential in mathematical reasoning, achieving state-of-the-art performance on multiple multimodal mathematical benchmarks and demonstrating robust supervision abilities.


---

[Retrieval-Augmented Generation with Graphs (GraphRAG)](https://arxiv.org/abs/2501.00309)

- GraphRAG: is a framework for retrieval-augmented generation using graph-structured data.
- It defines key components like query processor, retriever, organizer, generator, and data source; reviews techniques tailored to different domains; discusses research challenges and future directions.
- This framework provides a comprehensive overview of GraphRAG for information retrieval, data mining, and machine learning communities.


---

[Agent Laboratory: Using LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227)

- Agent Laboratory: An autonomous research-framework with LLMs for completing the entire research process (literature review/experimentation/report writing), from literature review to experimentation (plan formulation, data preparation and running experiments) and report writing (report writing and report refinements).
- Human-in-the-loop, research idea as input and code repository/research report as output. Producs SOTA-level performance and reduces research expensesn.
- The framework has the potential to accelerate scientific discovery by enabling researchers to focus on creative ideation rather than low-level coding and writing.
- Includes postdoc/ph student/sw engineer/ml engineer/professor-agents. Includes mle-solver-tool capable of solving ML-tasks, which iteratively improves research code.
- Automated evaluation of the framework significantly overestimated the accurate scoring. Copilot mode was found useful by the human testers. Includes prompts.


---

[Supervision-free Vision-Language Alignment](https://arxiv.org/abs/2501.04568)

- SVP (Supervision-free Visual Projection): A novel framework that enhances vision-language alignment in VLMs without relying on curated data or preference annotation.
- Leverages self-captioning, pre-trained grounding model, feedback mechanism, elicits latent information, improves vision-language alignment.
- The framework significantly improves performance across various tasks, including captioning, referring, visual question answering, multitasking, hallucination control, and object recall, highlighting its potential to advance multimodal AI systems.


---

#### 7th of January 2025

[Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation](https://arxiv.org/abs/2501.04167)

- REST-PG (Reasoning-Enhanced Self-Training for Personalized Text Generation): Introduces a multi-stage framework designed to teach LLMs reasoning over personalized context through Expectation-Maximization Reinforced Self-Training.
- Generates reasoning paths based on the user's past preferences, background knowledge, and writing style
- The framework enhances LLMs' ability to generate personalized text, outperforming state-of-the-art baselines by 14.5% on average.


---


#### 6th of January 2025

[Large language models for artificial general intelligence (AGI): A survey of foundational principles and approaches](https://arxiv.org/abs/2501.03151)

- Introduces a survey about AGI concepts and achieving AGI with LLMs. Includes list of memory types used with LLMs: sensory/working/semantic/episodic/procedural. Lists aspects of embodiment as: goal-awareness/self-awareness/situatedness/deliberate action. 


---

[CALM: Curiosity-Driven Auditing for Large Language Models](https://arxiv.org/abs/2501.02997)

- CALM (Curiosity-driven Auditing for LLMs): Introduces intrinsically motivated RL based on curiousity to finetune LLM as an auditor agent, to discover harmful/biased input/output pairs in the LLM. Includes token-level intrinsic bonus. Uses curiosity-driven exploration to navigate efficiently the prompt space, such as discover specific celebrity names.


---

[RTLSquad: Multi-Agent Based Interpretable RTL Design](https://arxiv.org/abs/2501.05470)


- RTLSquad: is a novel LLM-Based Multi-Agent system for interpretable RTL code generation.
- It divides the design process into exploration, implementation, and verification & evaluation stages, managed by specialized agent squads, generating optimized RTL code through inter-agent collaboration, and providing decision interpretability through the communication process.
- This framework enhances the ability to generate functionally correct RTL code and optimize PPA performance, while also providing decision paths.


---


#### 5th of January 2025


[LLMs Help Alleviate the Cross-Subject Variability in Brain Signal and Language Alignment](https://arxiv.org/abs/2501.02621)

- Decodes EEG scans to text with subject-independent semantic features for Brain-Computer Interfaces (BCIs).  Introduces EEG embeddings.
- Includes cross-subject generalization (addresses the issue of variability in brain anatomy between humans/neural dynamics/signal), zero-shot and comprehensive evaluation.  


---

[DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)

- DeepSeek LLM: is an open-source large language model framework with 7B and 67B configurations.
- It uses a 2 trillion token dataset, multi-step learning rate scheduler, and includes SFT and DPO stages.
- This framework achieves superior performance compared to LLaMA-2 and GPT-3.5 in various benchmarks.


---


#### 4th of January 2025

[Table as Thought: Exploring Structured Thoughts in LLM Reasoning](https://arxiv.org/abs/2501.02152)

- Table as Thought: organizes reasoning within a tabular schema, where rows represent sequential thought steps and columns capture critical constraints and contextual information.
- Framework is inspired by cognitive neuroscience theories, reasoning process iteratively populates the table until self-verification ensures completeness and correctness, excels in planning tasks and mathematical reasoning.
- This work provides a novel exploration of refining thought representation within LLMs, paving the way for advancements in reasoning and AI cognition.


---

[Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Diffusion Transformer Networks](https://arxiv.org/abs/2412.00733)

- Hallo3: The first application of a pretrained transformer-based video generative model for highly dynamic, realistic portrait animation.
- Identity reference network, 3D VAE, transformer layers, speech audio conditioning, motion frame mechanisms, DiT-based video generation, video extrapolation.
- Addresses challenges of non-frontal perspectives, dynamic objects, and immersive backgrounds in portrait animation.


---

[Thinking with Many Minds: Using Large Language Models for Multi-Perspective Problem-Solving](https://arxiv.org/abs/2501.02348)

- Replicates the concept of "Wisdom of the Crowd" with LLMs using synthetic deliberation. 
- Generates multiple agents, each with dinstinct perspective to a problem. Agents simulate arguments and counter-arguments from their perspective. 
- Agents explore in parallel the problem space using its own perspective. The integration mechanism adjusts agents positions based on proposals/evaluations of others controllable with influence parameter alpha. The iterative deliberation repeats multiple rounds until consensus is reached. 


---

[UAVs Meet LLMs: Overviews and Perspectives Toward Agentic Low-Altitude Mobility](https://arxiv.org/abs/2501.02341)

- Review systematically integration of LLMs with UAVs (Unmanned aerial vehicles).
- Proposes roadmap towards agentic UAVs. Includes github-repository with links to papers/approaches around LLM-based UAV systems.


---

#### 3rd of January 2025


[SDPO: Segment-Level Direct Preference Optimization for Social Agents](https://arxiv.org/abs/2501.01821)

- Introduces SDPO (Segment-Level Direct Preference Optimization)-fine tuning, which aligns the LLM to key segments in multi-turn conversation. 
- Addresses goal-completion in multi-turn conversation.


---

[AgentRefine: Enhancing Agent Generalization through Refinement Tuning](https://arxiv.org/abs/2501.01702)

- AgentRefine: Uses a strong LLM to simulate interactive role-playing, with the model acting as both Dungeon Master and player. A verifier checks each action for errors, providing feedback that allows the model to refine its actions until it achieves the correct result. This iterative process, with its corrected action sequences, trains the system to explore viable actions and generalize to new scenarios.

---

[Multi-Agent Conversational Online Learning for Adaptive LLM Response Identification](https://arxiv.org/abs/2501.01849)

- MACO (Multi-Agent Conversation Online learning for adaptive LLM response identification): Introduces near-optimal cumulative regret with multiple local agents to identify, which is the most optimal LLM response to serve for the particular user, even when new user.


---

[MoColl: Agent-Based Specific and General Model Collaboration for Image Captioning](https://arxiv.org/abs/2501.01834)

- MoColl: Introduces LLM-agent based framework for image captioning with specialised VQA model. Includes warm-up stage and agent-guided tuning stage.


---

#### 2nd of January 2025

[ProgCo: Program Helps Self-Correction of Large Language Models](https://arxiv.org/abs/2501.01264)

- ProgCo (Program-driven Self-Correction): A self-correction framework that uses self-generated and self-executed verification pseudo-programs to improve reasoning in large language models. Incluces ProgVe (Program driven Verification) and ProgRe (Program driven Refinement).
- This framework enhances the ability of large language models to self-correct without external feedback, particularly in complex reasoning tasks.


---

[PREDICTING THE PERFORMANCE OF BLACK-BOX LLMS THROUGH SELF-QUERIES](https://arxiv.org/abs/2501.01558)

- QueRE (Question Representation Elicitation): A framework to extract features of LLMs (Large Language Models) in a black-box manner by using follow-up prompts and taking the probabilities of different responses as representations to train reliable predictors of model behavior.
- Low-dimensional representations, linear model, instance level, model performance, hidden state, question-answering, adversarial system prompt, model architectures, model sizes.
- The framework can be used to predict model performance, detect models influenced by adversarial system prompts and distinguish between different model architectures and sizes.


---


[A3: Android Agent Arena for Mobile GUI Agents](https://arxiv.org/abs/2501.01149)

- A3 (Android Agent Area): Introduces benchmark to evaluate mobile GUI agents, which focuses on practical tasks, larger action spaces and automated LLM-based evaluation. 
- A3 consists of controller (gets/controls states of the device), evaluator (final rating) and translator (between device device function and the agent message).

---


[Dynamic Scaling of Unit Tests for Code Reward Modeling](https://arxiv.org/abs/2501.01054)

- CodeRM-8B: A lightweight unit test generator with dynamic scaling mechanism, which adapts number of unit tests based on problem difficulty. The unit tests are used in validating generated code by the LLM as reward signal. 
- The framework significantly improves performance of code generation across various models and benchmarks by enhancing the quality of the reward signal.


---


---


[3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer](https://arxiv.org/abs/2501.01163)

- 3D-LLaVa: Introduces 3D multi modal LLM with point clouds/text instruction/visual prompt as input and generates text output and 3D mask with Omni Superpoint Transformer (OST).
- 3D-LLaVa handles 3D vision-centric dialogue.
- OST includes visual features selection, visual prompt encoding and 3D mask generation. 


---

[Harnessing Multi-Agent LLMs for Complex Engineering Problem-Solving: A Framework for Senior Design Projects](https://arxiv.org/abs/2501.01205)

- Proposes multi-agent LLM framework for engineering design projects consisting of problem formulation/breadth & depth/ambiguity & uncertainty/system complexity/technical innovation & risk management/societal & ethical consideration/methodology & approach/compherensive evaluation-agents.
- Each agent consists of description, task, objective and evaluation points.


---


[Embodied AI-Enhanced Vehicular Networks: An Integrated Large Language Models and Reinforcement Learning Method](https://arxiv.org/abs/2501.01141)

- Incorporates embodied AI framework, which consists of semantic data processing with LLaVa-agent (extracts semantics from image data captured by the vehicle), Data transmission optimization (balances bandwidth utilization and quality of experience) and Enhanced decision making with Deep RL with GAE-PPO.


---

[MDSF: Context-Aware Multi-Dimensional Data Storytelling Framework based on Large language Model](https://arxiv.org/abs/2501.01014)

- MDSF (Multidimensional Data Storytelling Framework): Automatess data analysis and storytelling. Includes data preprocessing steps, fine tuned LLMs, LLM agents.

---


[Toward Inclusive Educational AI: Auditing Frontier LLMs through a Multiplexity Lens](https://arxiv.org/abs/2501.03259)

- Suggests two strategies to improve LLMs multiplexity (diverse cultural viewpoints) over WEIRD (western/educated/industrialized/rich/democratic): system prompt with diverse cultural perspectives and multi-agent system with agents with different cultural views. Sentiment analysis is used to review cultural resonance. 

---

[PSYCHE: A Multi-faceted Patient Simulation Framework for Evaluation of Psychiatric Assessment Conversational Agents](https://arxiv.org/abs/2501.01594)

- PSYCHE: Introduces an LLM-based psychiatric evaluation framework by comparing the predicted values of psychiatric elements (Construct-PACA) against the actual values (Construct-SP). The actual values are simulated patient data generated with a multi-faceted construct (MFC). 
- The framework guarantees clinical relevance, ethical safety, cost efficiency, and quantitative evaluation by simulating psychiatric patients with detailed profiles, histories, and behaviors.


---


[BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery](https://arxiv.org/abs/2501.01540)

- Introduces BoxingGym-benchmark, reviews LLMs capabilities to design and model discovery: collect data to test scientific theory and propose/update scientific theories through 10 environments. Introduces metric called EIG.
- Expected information gain (EIG) measures an experiment's informativeness by testing if one scientific agent's model explanation enables another to make accurate environmental predictions.


---


[General Information Metrics for Improving AI Model Training Efficiency](https://arxiv.org/abs/2501.02004)

- GIME (General Information Metrics Evaluation): A novel framework for optimizing AI model training by evaluating datasets using 11 general information metrics before training begins.
- Objective Information Theory (OIT), pre-training assessment, data selection, training efficiency, reduced costs, model-agnostic, domain-independent.
- This framework improves AI model training efficiency and reduces resource consumption while preserving model performance across various domains.


---


#### 1st of January 2025

[Agentic Systems: A Guide to Transforming Industries with Vertical AI Agents](https://arxiv.org/abs/2501.00881)

- Reviews transition from SaaS to context-aware, adaptive systems handling dynamic environments through vertical agents.
- Identifies core modules of LLM agents: memory/reasoning engine/cognitive skills/tools. 
- Author categorises agentic systems into: task-specific, multi-agent and human augmented agent systems.


---


[Large Language Model Based Multi-Agent System Augmented Complex Event Processing Pipeline for Internet of Multimedia Things](https://arxiv.org/abs/2501.00906)

- Introduces multi-agent framework for complex event processing of video queries(think TikTok/Youtube as examples) with AutoGen and Kafka brokers (real time data streams).
- Consists of conversable/assistant/user proxy/LLM backend/human backed/tool backed-agents.


---

[Interactionalism: Re-Designing Higher Learning for the Large Language Agent Era](https://arxiv.org/abs/2501.00867)

- Introduces Interactionalism-framework focuses on interactional intelligence to learn more personalized/social/non-linearly way, instead of monological way. 
- Proposes usage of dialogue-agents in education, such as tutors, teaching assistants, evaluators, guides and mentors. 


---


[LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management](https://arxiv.org/abs/2501.00826)
- Introduces multi-agent framework for cryptocurrency investing with intrateam and interteam collaboration and multi modality. Consists of expert training module and multi-agent investment module. 
- Expert training module uses data/literature-agents to feed historical data and investment literature. Explanation-agents process this information to generate high-quality prompts to fine tune investment agents. 
- Multi-agent investment module consists of data-agent fetching real-time data to market-agents and crypto agents. Market agents includes two expert agents to analyze news/market factors to predict market trends and determining cash-crypto allocation. Crypto-agents includes two specialized agents to analyze crypto-specific factors and candlestick charts to make crypto selection decisions. Trading agents finally act with a trading API to execute the final portfolio strategy.
 

---

[Beyond Text: Implementing Multimodal Large Language Model-Powered Multi-Agent Systems Using a No-Code Platform](https://arxiv.org/abs/2501.00750)

- Proposes design and implementation of multi modal and multi-agent framework with LLMs. Includes multi modal inputs (text/audio/video/image), multi-agent layer (includes supervisory-agent and RAG/image analysis/audio generation/image generation/video generation- worker agents), process layer (vector db and modality specific models) and the output layer (text/audio/video/image).
- Supervisor agent controls sequence of tasks, distributes tasks, manages output of worker agents, tnterprets outputs and makes decisions about next steps in the sequence.


---

#### 31st of December 2024

[Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)

- STILL-1 (Slow Thinking with LLMs): A reward-guided tree search framework to enhance the reasoning capabilities of LLMs.
- Integrates policy model, reward model, and search algorithm; policy model navigates a dynamically expanding tree; guided by a trained reward model.
- Improves LLMs' performance on complex mathematical reasoning tasks by trading test time for improved accuracy.


---

[MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](https://arxiv.org/abs/2501.00332)

- Main-RAG: Introduces multi-agent framework, where LLM-agents collaboratively filter and score retrieved documents.
- Introduces adaptive filtering, which dynamically adjusts relevance filtering threshold.
- Includes three agents: predictor (infers answers based on retrieved documents), judge (scores filtering and ordering) and final-predictor (generates final answer based on filtered and ordered documents). 
- Includes system instruction prompts.


---

[Enhancing LLM Reasoning with Multi-Path Collaborative Reactive and Reflection agents](https://arxiv.org/abs/2501.00430)

- RR-MP (Reactive and Reflection agents with Multi-Path Reasoning): Improves reasoning capability of LLMs in complex scientific tasks.
- Consists of reactive and reflection agents collaborating together to improve accuracy/avoid degeneration-of-thoughts. 
- Reactive agent receives information from external environment, decomposes it into sub-tasks, then stores them in the database.
- Reflective agent analyzes sub-task it executes, offering suggestions or critiques. This feedback loop allows the reactive agent to refine its reasoning and complete the scientific process.

---

[Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding](https://arxiv.org/abs/2501.00358)
 
- Embodied VideoAgent: Introduces VLM-based Embodied VideoAgent, which constructs scene memory from both egocentric video and embodied sensory inputs.
- Includes persistent object memory, using VLM (depth maps / camera poses).
- Automatically updates memory as actions / activities over objects are perceived.




---

[Enabling New HDLs with Agents](https://arxiv.org/abs/2501.00642)

- HDLAgent: Introduces LLM-based agent to support code generation for underrepresented HDLs (Hardware Description Languages).


---

[VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](https://arxiv.org/abs/2501.00599)

- VideoRefer-model: Improves Video-LLMs fine-grained spatial and temporal detail understanding in videos, which facilitates more precise object descriptions, more detailed event analysis, and enhanced predictive reasoning in dynamic environments using masked object features.
- VideoRefer-model consists of VideoLLaMA 2.1 as the foundation and a novel unified spatial-temporal object encoder that merges cross-frame token similarities.
- Includes VideoRefer-dataset and VideoReferBench-benchmark.


---

[LLM-MedQA: Enhancing Medical Question Answering through Case Studies in Large Language Models](https://arxiv.org/abs/2501.05464)

- LLM-MedQA: is a multi-agent medical question-answering system that incorporates similar case generation within a multi-agent architecture.
- It leverages Llama3.1:70B model, includes question-specific analysis, option analysis, and case generation agents, and uses zero-shot learning.
- This framework enhances performance on the MedQA dataset and improves interpretability and reliability in medical question answering.


---

#### 30th of December 2024

[Aviary: training language agents on challenging scientific tasks](https://arxiv.org/abs/2412.21154)

- Defines Language Decision Process (LDP). LDP is framed as Partially-Observable Markov Decision Process (POMDP), where actions only consist of the ones with the external environment.
- Introduces Language agent training framework: Aviary. Includes implementation in 3 scientific domain tasks. 
- Builds language agents as stochastic computation graphs (SCG).

---

[Distributed Mixture-of-Agents for Edge Inference with Large Language Models](https://arxiv.org/abs/2412.21200)

- Introduces Distributed Mixture-of-Agents, where multiple LLMs collaborate on various edge devices with decentralized gossip algorithm.
- Does not rely in centralized server. 

---

[Exploring and Controlling Diversity in LLM-Agent Conversation](https://arxiv.org/abs/2412.21102)

- APP (Adaptive Prompt Pruning): Controls diversity of the LLM-agent conversation through adjusting lambda-variable. 
- The lambbda variable adjusts diversity by increasing/decreasing details about: current dialogue/history dialogue/environment/profile/memory.

---

[Plancraft: an evaluation dataset for planning with LLM agents](https://arxiv.org/abs/2412.21033)

- Introduces Plancraft-benchmark to evaluate VLMs and LLMs planning capabilities and ability to decide in Minecraft craftting GUI, if the model is able to identify task as unsolvable (intentionally).
- Identifies, that success rate alone is poor metric in real world tasks.



---


#### 25th of December 2024

[Probabilistic Mission Design in Neuro-Symbolic Systems](https://arxiv.org/abs/2501.01439)

- ProMis (Probabilistic Mission Design): ProMis helps drones understand where they can and cannot go by combining different types of information, like maps and sensor data, with rules and regulations, such as no-fly zones. Refers with mission landscape to safest and most legal paths.
- Combines formal reasoning with probabilistic inference. Uses LLM to convert instructions into ProMis code and ChangeFormer for perception of satellite images.


---


#### 24th of December 2024

#### 24.12.2024

[A Novel Task-Driven Method with Evolvable Interactive Agents Using Event Trees for Enhanced Emergency Decision Support](https://arxiv.org/abs/2501.06193)

- EvoTaskTree: is a task-driven method with evolvable interactive agents using event trees for emergency decision support.
- Framework integrates task executors and task validators powered by large language models (LLMs), leverages insights from event tree analysis, and includes three crucial tasks: initiating event subevent analysis, event tree header event analysis, and decision recommendations.
- This approach enhances rapid formulation of emergency decision-making and outperforms existing approaches.


---

[Multi-Agents Based on Large Language Models for Knowledge-based Visual Question Answering](https://arxiv.org/abs/2412.18351)

- Introduces multi-agent framework consisting of three level of agents collaborating to provide answer: junior, senior and manager. Final answer is determined through voting. Each agent uses planning and tools (knowledge base / LLM knowledge).

---


[VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks](https://arxiv.org/abs/2412.18194)

- VLABench-benchmark: Evaluates VLA models (Vision-Language Action models). Focuses on tasks requiring mesh & texture understanding, spatial understanding, semantic conversation cognition, common sense & applying real world knowledge, physical laws understanding and long horizon multi-step reasoning.


---

[INVESTORBENCH: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent](https://arxiv.org/abs/2412.18174)

- Investorbench-benchmark: Evaluates LLMs capability for financial decision making. 


---

[Decentralized Intelligence in GameFi: Embodied AI Agents and the Convergence of DeFi and Virtual Ecosystems](https://arxiv.org/abs/2412.18601)

- Introduces decentralized GameFI-ecosystem with LLM-agents based on Ethereum-blockchain.


---


[Automated Code Review In Practice](https://arxiv.org/abs/2412.18531)

- Reviews automated code reviews, which led to longer average pull request closer time.  


---

[Large Language Model guided Deep Reinforcement Learning for Decision Making in Autonomous Driving](https://arxiv.org/abs/2412.18511)

- LGDRL (Language Guided Deep Reinforcement Learning): Introduces LLM-based autonomous driving system. 
- DRL agent learns from LLM-based driving expert-agent (prompted with prompt generator), when the LLM-based driving expert finds necessary to intervene DRL agent actions.


---


[3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](https://arxiv.org/abs/2412.18450)

- 3DGraphLLM: Improves LLMs understanding of 3D scenes by creating 3D scene graph representation (think graph, where arrows point, if object is right/left/front/behind) from set of point clouds (object input).

---

[Explainable Multi-Modal Data Exploration in Natural Language via LLM Agent](https://arxiv.org/abs/2412.18428)

- XMODE: Uses LLM to decompose (converts into simpler sub-questions and translates into workflows) user queries into SQL / image analysis.
- Includes planning & expert model allocation/execution & self-debugging/decision making/expert models & tools/data lake. 


---

[Muse: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles](https://arxiv.org/abs/2412.18416)

- Introduces MUSE-dataset with conversations centered around clothing-domain by using multi-agent framework to generate real world-scenarios (scenario-grounded user profile generator/simulated conversation generator/conversation optimizer). 


---

[Defining and Detecting the Defects of the Large Language Model-based Autonomous Agents](https://arxiv.org/abs/2412.18371)

- Agentable: Introduces static analysis tool to detect defects in code with LLM-based agents and Code Property Graphs (identifies specific code patterns/analyses descriptions). Includes AgentSet-dataset.
- Includes pre-processing, defect detection (code abstraction/LLM invocation/semantic enrichment/detect oracles engineeering), and defect reporting-modules.

---

#### 22.12.2024

[Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems](https://arxiv.org/abs/2412.09413)

- STILL-2 (Slow Thinking with LLMs): A framework to train reasoning models using a three-phase approach: imitation, exploration, and self-improvement.
- Initial fine-tuning with distilled long-form thought data, exploration of challenging problems by generating multiple rollouts, iterative refinement of the training dataset.
- The framework demonstrates competitive performance compared to industry-level reasoning systems, highlighting the potential of slow-thinking in enhancing complex reasoning capabilities of LLMs.


---


#### 21st of December 2024

[OpenAI o1 System Card](https://arxiv.org/abs/2412.16720)

- o1 model series: Large-scale reinforcement learning models trained to reason using chain of thought, improving safety and robustness.
- Next model in series is OpenAI o1, faster version is OpenAI o1-mini, effective at coding, "thinks before it answers", long chain of thought before responding, refine thinking process, try different strategies, recognize mistakes.
- Reasoning allows models to follow safety guidelines, provide helpful answers, resist attempts to bypass safety rules, avoid producing unsafe content, and reach state-of-the-art performance on certain benchmarks.


---

#### 20th of December 2024

[Deliberative Alignment: Reasoning Enables Safer Language Models](https://arxiv.org/abs/2412.16339)

- Deliberative Alignment: A training approach that "directly teaches" LLMs to explicitly reason through (safety) specifications before producing an answer.
- Claims, that reasoning using explicitly specified policies in general, enable scaling alignment. Apart, imrpoves model safety, robustness to jailbreaks, out-of-distribution generalization, and reduces overrefusal rates.
- Two core stages: supervised fine-tuning on (prompt, CoT, output) examples, reinforcement learning; uses context distillation; includes a "judge" LLM for reward signal.
- Assigns deliberatedly a varied amount of compute to CoT, which improves performance in hard evals.
- In first stage, the model is fine tuned with SFT to reason about the (safety) specification within its CoT using examples dataset generated with context distillation with o-type model, where the CoT references the specification.
- Second stage trains with high-compute RL the model to think effectively by providing reward signal using a judge LLM with access to the (safety) instructions.



---




[Offline Reinforcement Learning for LLM Multi-Step Reasoning](https://arxiv.org/abs/2412.16145)

- OREO (Offline REasoning Opyimization): improves multi-step reasoning with offline RL.
- Iterative OREO improves consistently with additional training rounds.

---

#### 19th of December 2024

[Disentangling Reasoning Tokens and Boilerplate Tokens For Language Model Fine-tuning](https://arxiv.org/abs/2412.14780)

- Reasoning-highlighted Finetuning (RFT): Highlights reasoning tokens from boilerplate tokens (format and connecting tokens less critical for the task). Adds larger weight to reasoning tokens.
- Introduces SHAD (Shuffle-Aware Discriminator): automatic, adaptive token discrimination. 


---

[On Verbalized Confidence Scores for LLMs](https://arxiv.org/abs/2412.14737)

- Claims, that LLMs can be prompted to provide caliberated confidence scores.

---

[Agent-SafetyBench: Evaluating the Safety of LLM Agents](https://arxiv.org/abs/2412.14470)

- Agent-SafetyBench-benchmark evaluates LLM-agents safety. Agents tested achieved below 60% pass score.
- LLM-agents lack currently robustness and risk awareness.


---

[TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](https://arxiv.org/abs/2412.14161)

- TheAgentCompany-benchmark: evaluates AI agents capacity to perform long-sequence tasks in real world-like environment as a digital worker: arranging meetings, writing code, screening resumes, communicating (simulates communication between agents), planning and administrative work. Best agent completed 24% of tasks.
- Generates tasks in a self-contained environment with internal websites and data similar to used by SW companies.


---

#### 18th of December 2024

[Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers](http://arxiv.org/abs/2411.17501)

- LLM Resampling: explores the limits of using resampling with imperfect verifiers for improving language model accuracy.
- The framework shows that imperfect verifiers, like unit tests, lead to false positives, limiting the effectiveness of resampling, and that weaker models generalize worse than stronger models, even with infinite compute budget.
- This research highlights the importance of developing accurate verifiers and questions the effectiveness of inference scaling with imperfect verifiers.


---


#### 17th of December 2024

[AI PERSONA: Towards Life-long Personalization of LLMs](https://arxiv.org/abs/2412.13103)

- AI Persona: proposes, that LLMs should continuously adapt to diverse set of users via personalization. 
- Introduces a framework for life-long personalization of LLMs through learnable and dynamically updated dictionaries, which are updated based on interaction between user and the LLM.


---

#### 13th of December 2024

[Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)

- Byte Latent Transformer (BLT): is a byte-level LLM architecture that encodes bytes into dynamically sized patches to efficiently allocate compute by varying the amount of compute based on the entropy of the next byte prediction.
- BLT segments patches based on next-byte entropy, allocates more compute where data complexity increases, and improves training and inference efficiency.
- BLT shows better scaling than tokenization-based models by simultaneously growing both patch and model size.


---

#### 11th of December 2024

[A Multimodal Social Agent](https://arxiv.org/abs/2501.06189)

- MuSA: is a multimodal LLM-based agent designed for analyzing text-rich social content.
- MuSA includes reason-, plan-, optimize-, criticize-, refine- and act-LLM-based units, is model-agnostic, and optimized for social content analysis tasks.
- MuSA can automate and improve social content analysis, aiding decision-making processes across various applications.


---


#### 10th of December 2024

[CePO: Empowering Llama with Reasoning using Test-Time Compute](https://cerebras.ai/blog/cepo)
- CePO (Cerebras Planning and Optimization): Adds sophisticated reasoning capabilities to the Llama family of models using test-time computation techniques.
- CePO enables Llama-3.3 70B to surpass Llama-3.1 405B in accuracy across coding, math, and reasoning tasks.
- CePO's step-by-step reasoning, comparison instead of verification, and intuitive output format improve Llama's performance.
- CePO achieves interactive performance of approximately 100 tokens/second on Cerebras hardware, comparable to leading models like GPT-4 Turbo and Claude 3.5 Sonnet.

---



#### 9th of December 2024


[AlphaVerus: Bootstrapping Formally Verified Code Generation through Self-Improving Translation and Treefinement](https://arxiv.org/abs/2412.06176)

- AlphaVerus: generates formally verified code with LLMs and through self-improvement by iteratively translating programs from higher resource language.
- Includes three phases: exploration (translates programs from source language to Verus, which is a tool to verify correctness of code written in Rust), treefinement(iteratively fixes errors with Verus-verifier feedback/tree search) and critique (validates and filters unspecified/incorrect translations).
- Illustrates the potential of inference-time scaling in verified settings. Suggests formal verification ensures correctness and reliability of the generated code. 


---

[Query-Efficient Planning with Language Models](https://arxiv.org/abs/2412.06162)

- Reviews efficient ways to use LLMs for planning: heuristic and LLM as generative planner.
- Introduces two new algorithms: Tree of Interaction (ToI) and Boomerang.


---

[Simulating Human-like Daily Activities with Desire-driven Autonomy](https://arxiv.org/abs/2412.06435)

- D2A-agent (Desire-driven Autonomous Agent): Introduces autonomous agent proposing and selecting autonomously fulfilling and motivating tasks (based on theory of needs: social interaction/personal fulfillment/self-care).
- Introduces desire-based characters.
- Includes value system (measures satisfaction per desired dimension) and Desire-driven planner (choses next action of the agent with history and value system).
- Proposes using in the future more complex human motivation and planning mechanisms to satisfy intrinsic desires. Includes prompts.


---

[Toward LLM-Agent-Based Modeling of Transportation Systems: A Conceptual Framework](https://arxiv.org/abs/2412.06681)

- Proposes transportation system modelling with LLM-based agents to replicate human decision making.
- LLM-based agents include long-lasting core components: identity (age/income/occupation/cars owned/persona/travel related task/travel restrictions)/memory(short and long term)/LLM core(summarization/planning/nlu/workflow).
- Includes iterative process with perception, reflection, planning, plan processing and action.


---

[Beyond pip install: Evaluating LLM Agents for the Automated Installation of Python Projects](https://arxiv.org/abs/2412.06294)

- Installamatic: Reviews LLM-agents capability to install repository-level python packages with pip by automatically inspecting repository content and install the packages required. 
- Installamatic-agent is capable of installing packages required in 21/40 repositories tested with 4 main challenges: Identifying install-relevant documentation/writing valid docker files/cost/oracle-problem.


---

[AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark](https://arxiv.org/abs/2412.06724)

- AutoDCWorkflow: uses LLM to automatically generate data-cleaning workflows (duplicates/missing values/inconsistent data format) and introduces a benchmark.


---

[StarWhisper Telescope: Agent-Based Observation Assistant System to Approach AI Astrophysicist](https://arxiv.org/abs/2412.06412)

- SWT (StarWhisper Telescope System): proposes automation of the astronomer observation process with LLMs. Includes observation planning/control/data processing/agent suggestion. Includes customized observation lists and real time analysis.


---

#### 5th of December 2024

[Practical Considerations for Agentic LLM Systems](https://arxiv.org/abs/2412.04093)

- Reviews LLM agent research from perspective of planning (explicit/implicit, task decomposition, plan adherence), memory (RAG, long-term memory), tools (ysage/dynamic/multiplicity) and control flow (output processing/error handling/stopping/multi-persona/context).
- Long term memory may include reflection/consolidation/forgetting/revision and should be independent/consistent/long-term.

---

[Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation](https://arxiv.org/abs/2412.04415)

- Investigates adversial Adaptive Attack Prompt- and ArtPrompt-attack methods success rates between LLM models.


---

#### 2nd of December 2024

[Mastering Board Games by External and Internal Planning with Language Models](https://arxiv.org/abs/2412.12119)

- MAV (Multi Action-Value) model: is a transformer model pre-trained on textual game data, functioning as a world model, value function, and policy function for multiple perfect-information board games.
- Framework includes external and internal search methods, uses MCTS controller, and distills search procedure directly into the LLM, pre-trained on relevant domain knowledge, minimizes hallucinations, and improves win-rates against state-of-the-art bots.
- This framework demonstrates the capacity of LLMs to learn strong value functions and act as a world model across multiple perfect information games.

---

[Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers](http://arxiv.org/abs/2411.17501)

- LLM Resampling: explores the limits of using resampling with imperfect verifiers for improving language model accuracy.
- The framework shows that imperfect verifiers, like unit tests, lead to false positives, limiting the effectiveness of resampling, and that weaker models generalize worse than stronger models, even with infinite compute budget.
- This research highlights the importance of developing accurate verifiers and questions the effectiveness of inference scaling with imperfect verifiers.


---

#### 29th of November 2024

[Amplifying human performance in combinatorial competitive programming](https://arxiv.org/abs/2411.19744)

- FunSearch: is a framework that evolves scoring functions for a human-designed solution backbone using a large language model.
- Framework uses Gemini 1.5 Flash 002, improves scores on Hash Code, and uses a switching variable for multiple choice points.
- This approach demonstrates a successful human-AI synergy in combinatorial optimization problems.


---


#### 25th of November 2024


[Agent-Based Modelling Meets Generative AI in Social Network Simulations](https://arxiv.org/abs/2411.16031)

- Generative Agent-Based Modelling (GABM): LLM-based agents, which simulate social network users with personality traits/interests and custom agent interactions. 
- The framework consists of two phases: Characterization (Personality assignment) and Simulation (Reasoning module and Interaction module). Decisions of the agent are stored in vector db for retrieval. 

---


[TopV-Nav: Unlocking the Top-View Spatial Reasoning Potential of MLLM for Zero-shot Object Navigation](https://arxiv.org/abs/2411.16425)

- TopV-Nav: Improves Zero-Shot Object Navigation (ZSON) in unfamiliar environments by reasoning on top-view maps ("birds eye") with MLLM's spatial reasoning capabilities. 
- Proposes Adaptive Visual Prompt Generation (AVPG), which adaptively constructs top-view map. The framework then uses Dynamic Map Scaling (DMS), which dynamically zooms top-view map at preferred scales for local reasoning. Uses Target-Guided Navigation (TGN) to facilitate human-like exploration.


---

[A Multi-agent Framework for Materials Laws Discovery](https://arxiv.org/abs/2411.16416)

- Introduces a LLM-based multi agent framework to discover materials laws in materials science, using general framework for solving symbolic regression tasks with LLMs. 
Uses a depth-first search (DFS) algorithm and a reflection mechanism, implemented through LLMs, to optimize formula generation. 


---

[Enhancing Multi-Agent Consensus through Third-Party LLM Integration: Analyzing Uncertainty and Mitigating Hallucinations in Large Language Models](https://arxiv.org/abs/2411.16189)

- Introduces a multi-agent consensus framework, which integrates confidence weight obtained with third-party LLM, to adjust attention weights of each agent. 
- Each agent answers individually on the first round, agents self-adjust with feedback on second/third round with third party LLM and finally agents majority vote the final answer.


---

[SAGEval: The frontiers of satisfactory agent-based NLG evaluation for reference-free open-ended text](https://arxiv.org/abs/2411.16077)


- SAGEval: Introduces an eval for an open-ended, reference-free natural language generation (NLG) by using a critiquing agent to provide feedback on scores generated by LLM evaluators. Focuses on open-ended text like surveys, forms, and lists. 
- Includes Evaluator- (based on G-Eval) and Sage-agent as meta-evaluator. Evaluation aspects include: accuracy, semantic diversity, coherence, relevancy, audience understandability, audience engagement score, fairness score and sentiment/tone type.


---

#### 24th of November 2024

[PIANIST: Learning Partially Observable World Models with LLMs for Multi-Agent Decision Making](https://arxiv.org/abs/2411.15998)

- PIANIST (Partition function, Information set space, Action space function, N players, Information realization function, State space, and Transition reward function): A framework for decomposing a world model into seven components, enabling zero-shot LLM generation of a working world model for multi-agent decision-making tasks.
- The framework leverages LLMs for generating forward transition functions, action functions, and information partition functions. It uses MCTS for planning in partially observable environments. The approach is evaluated on language and non-language based action-taking games, without domain-specific training data.
- PIANIST demonstrates strong performance in multi-agent, partial information settings, showcasing the potential of LLMs for complex decision-making.


---


#### 21st of November 2024

[Natural Language Reinforcement Learning](https://arxiv.org/abs/2411.14251)

- Introduces: Natural Language Reinforcement Learning (NLRL).
- Efficiently implements RL algorithms and principles in language representation space.
- Presents NLRL-pipeline, where LLM learns from textual environmental feedback.
- Implements empirically in various games.


---

#### 18th of November 2024

[GENERATIVE WORLD EXPLORER](https://arxiv.org/abs/2411.11844)

- Generative World Explorer (Genex): Introduces and egocentric world exploration, which allows an agent to mentally explore a large-scale 3D world and acquire imagined observations to update its belief inside partially observable decision process. 
- Generates high-quality and consistent observations in long-horizon tasks.
- Consists of generative video model, egocentric views, belief revision, and decision-making (e.g., LLM agent). Includes multi-agent reasoning with imagination, where the framework infers perspectives of other actors in the scene.


---


[OASIS: Open Agents SOCIAL INTERACTION Simulations on One Million Agents](https://arxiv.org/abs/2411.11581)

- OASIS (Open Agents SOCIAL INTERACTION Simulations on One Million Agents): Introduces generalizable, scalable (millions of agents) social media (twitter/reddit-like) simulator LLM-based agents  supporting dynamic social networks, diverse actions and recommendation systems. Includes registration and simulation phases.
- OASIS pulls in the registration phase information about user, past posts, self-description and name.
- Simulation phase consists of Environment server(sends agent information, posts and user relationships)/RecSys(recommends visible content to user and agents)/Agent module(generates actions updating environment state)/Time engine(updates agents temporal behaviours)/Scalable Inferencer-components(handles large scale inference requests by user).
- OASIS replicates social phenomena observed in human-societies, including group polarization and herd effect, which take place in dynamically updating environments with diverse action spaces.
- Uses event-driven architecture, where agent communicates with server in dedicated channel, which consists of asynchronous message queue.

---

[TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World](https://arxiv.org/abs/2411.11683)


- TrojanRobot: A backdoor attack framework, which targets robotic manipulation in the physical world by embedding a backdoor robotic system's visual perception module. 
- Uses common objects as triggers.


---

[A Code Knowledge Graph-Enhanced System for LLM-Based Fuzz Driver Generation](https://arxiv.org/abs/2411.11532)

- CodeGraphGPT: a framework that leverages a code knowledge graph and an LLM-powered intelligent agent to automate fuzz driver generation (sw testing technique by feeding unexpected random data as program inputs to discover bugs). 
- Includes agents for API combination generation (knowledge into graphs and then embeddings to query), dynamic program repair (past example embeddings), and crash analysis (bugs embeddings). 
- Constructs knowledge graph of code repos, tailors fuzz drivers and input seeds, resolves compilation errors, and analyzes crash reports.


---

[Moral Persuasion in Large Language Models: Evaluating Susceptibility and Ethical Alignment](https://arxiv.org/abs/2411.11731)

- Reviews Persuader agents capacity to influence another LLM agent (Base agent) in morally ambiguous decision making scenarios. 
- LLMs show greater variability between the degree it is possible to persuade them, than their capacity to persuade others.


---

[LLM-IE: A Python Package for Generative Information Extraction with Large Language Models](https://arxiv.org/abs/2411.11779)

- LLM-IE [LLM-based Information Extraction]: A Python package for building complete information extraction pipelines using large language models (LLMs).
- Key features include interactive LLM agent for prompt design, support for named entity recognition, entity attribute extraction, and relation extraction tasks. Benchmarked on i2b2 datasets. Sentence-based prompting algorithm.


---

#### 16th of November 2024

[Developer Challenges on Large Language Models: A Study of Stack Overflow and OpenAI Developer Forum Posts](https://arxiv.org/abs/2411.10873)

- Analyzes developer challenges with LLMs. Challenges include LLM ecosystem, API usage, LLM training, dataset management, prompt engineering, and error handling. Identifies several unresolved posts, slow response times, especially with complex topics.


---

[FlexFL: Flexible and Effective Fault Localization with Open-Source Large Language Models](https://arxiv.org/abs/2411.10714)

- FlexFL (Flexible and Effective Fault Localization): LLM-agents (Agent4SR and Agent4LR) based framework for code debugging / fixing with bug-related information (bug reports, test cases).
- The framework employs a two-stage approach: space reduction (Agent4SR) to narrow search space and localization refinement (Agent4LR) to localize top k-most suspicious methods.

---

[IntentGPT: Few-shot Intent Discovery with Large Language Models](https://arxiv.org/abs/2411.10670)

- IntentGPT: introduces a training-free method for Intent discovery using In-context Learning prompt (generated with LLM consisting of known intents/few-shot examples and user query) and LLM generating the intent.
- Adds discovered intents back into the prompt. Includes prompts. 
- IntentGPT outperforms previous methods with extensive domain-specific data for training/fine-tuning. Discovers intents dynamic, open-world scenarios.


---

#### 15th of November 2024

[Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://arxiv.org/abs/2411.10442)

- MPO (Mixed Preference Optimization): is a method that blends supervised fine-tuning loss with preference optimization losses to enhance training effectiveness of multimodal large language models.
- MPO uses a novel automated preference data construction pipeline to create MMPR dataset, and explores different Chain-of-Thought approaches with multimodal input to improve reasoning performance.
- This approach demonstrates improved performance across multiple benchmarks, particularly in multimodal reasoning tasks.

---


[A dataset of questions on decision-theoretic reasoning in Newcomb-like problems](https://arxiv.org/abs/2411.10588)

- Decision-theoretic reasoning: Introduces a dataset of natural language questions on Newcomb-like problems.
- The dataset includes capability questions (unambiguous answers) and attitude questions (disagreements among decision theorists). It evaluates existing large language models (LLMs) and their attitudes toward evidential decision theory (EDT) and causal decision theory (CDT). 
- Findings associate higher capability LLMs with more EDT-favorable attitudes across question types. The dataset helps to understand decision-theoretic reasoning capabilities and attitudes of LLMs in AI-AI interactions.


---

#### 12th of November 2024


[RedCode: Risky Code Execution and Generation Benchmark for Code Agents](https://arxiv.org/abs/2411.07781)

- RedCode-benchmark: Evaluates safety of code agents capacity to generate / execute code and reviews code agents capacity to recognize/manage unsafe code execution.
- Includes two steps: RedCode-Gen (evaluates code generated) and RedCode-Exec (evaluates code execution).


---

[World Models: The Safety Perspective](https://arxiv.org/abs/2411.07690)

- Introduces a Survey about World Models in Embodied AI agents from safety perspective.


---

[BudgetMLAgent: A Cost-Effective LLM Multi-Agent system for Automating Machine Learning Tasks](https://arxiv.org/abs/2411.07464)

- BudgetLMAgent: Multi agent framework using cascading (sequentially invoking/chaining) free/low cost/frontier LLMs with distinct roles: planner (default/expert)/workers(high-level actions/low-level actions).
- Gives LLM-agent an option to call more advanced LLM-model to request help (with maximum retries) in complex planning problems.
- Reduces operation cost by 94% compared to single agent with GPT-4 and improved success rate. 


---

[LLMPhy: Complex Physical Reasoning Using Large Language Models and World Models](https://arxiv.org/abs/2411.08027)

- LLMPhy: Combines LLM with Mujoco-physics engine for complex physical reasoning tasks and introduces TraySim-dataset consisting of 100 scenes.
- Claims, that LLMs have enough world knowledge with physics engine for better interactive reasoning and LLMs trained with more scientific reasoning tasks tend to demonstrate superior physical reasoning in LLMPhy-pipeline.


---

[From General to Specific: Utilizing General Hallucation to Automatically Measure the Role Relationship Fidelity for Specific Role-Play Agents](https://arxiv.org/abs/2411.07965)

- Introduces an automatic evaluation framework for Role-Playing Agents (RPAs) that generates claims from a knowledge graph and has characters discuss them with the main character.
- Evaluates the believability of interactions by leveraging the inherent hallucination properties of RPAs. Defines relationship hallucination metric.


---

[Mitigating Bias in Queer Representation within Large Language Models: A Collaborative Agent Approach](https://arxiv.org/abs/2411.07656)

- Focuses on inclusive / gender neutrality in LLM-agents with: assistant/language analysis/optimizer-agents.


---

#### 11th of November 2024

[Mr.Steve: Instruction-Following Agents in Minecraft with What-Where-When Memory](https://arxiv.org/abs/2411.06736)

- Mr.Steve (Memory Recall Steve-1): Improves long-horizon task solving by incorporating solver module and  Place Event Memory (PEM), which recalls what-, where- and when-information from episodes.
- Includes memory-augmented task solving and exploration strategy.


---

[Using Generative AI and Multi-Agents to Provide Automatic Feedback](https://arxiv.org/abs/2411.07407)

- Autofeedback: Introduces multi agent LLM-based framework for student feedback, which includes: feedback generation- and feedback validation/modifier. Reduces over-praising and over-inference. 
- Includes prompts of both agents.


---

[Script-Strategy Aligned Generation: Aligning LLMs with Expert-Crafted Dialogue Scripts and Therapeutic Strategies for Psychotherapy](https://arxiv.org/abs/2411.06723)

- SSAG (Script-Strategy Aligned Generation): Aligns LLMs with key therapeutic strategies in Motivational Interviewing. Claims, that LLMs aligned with expert prompting outperform rule-based chatbots and pure LLMs. 


---

[Tooling or Not Tooling? The Impact of Tools on Language Agents for Chemistry Problem Solving](https://arxiv.org/abs/2411.07228)

- ChemAgent-framework: Introduces agent for chemistry tasks, which includes reasoning/grounding and tool use. 


---

[A Multi-Agent Approach for REST API Testing with Semantic Graphs and LLM-Driven Inputs](https://arxiv.org/abs/2411.07098)

- AutoRestTest: Introduces MARL-framework with Semantic Property Dependency Graphs (SDG) and LLMs for REST API exploration.
- Includes dependency/operation/parameter/value-agents.


---


#### 10th of November 2024

[Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents](https://arxiv.org/abs/2411.06559)

- WebDreamer: LLM-based web-agent framework by using LLM to predict outcomes of candidate actions in web environment in order to pick optimal action.
- The LLM simulates as world-model actions using prompt like: "what would happen if I click this button" and then evaluates the imagined outcomes. 
- Model-based planning enables safe simulation of possible actions before taking them (some web environments do not allow going back to previous step, which complicates tree-based search by investigating candidate next steps).
- Includes system prompts of the world model and reward model.
 
---

#### 9th of November 2024

[IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization](https://arxiv.org/abs/2411.06208)

- IOPO (Input-Output Preference Optimization): Aligns/fine-tunes LLMs based on both the input data (new approach) and the output data (traditional approach). 
- Explores instruction preference space.

---

[From References to Insights: Collaborative Knowledge Minigraph Agents for Automating Scholarly Literature Review](https://arxiv.org/abs/2411.06159)

- Introduces CKMAs (Collaborative Knowledge Minigraph Agents), which automate literature reviews. Building knowledge minigraphs by organizing information and relationships from research papers.
- Includes KMCA (Knowledge Minigraph Construction Agent) and MPSA (Multiple Path Summarization Agent), which both prompts are included.


---


#### 8th of November 2024

[The influence of persona and conversational task on social interactions with a LLM-controlled embodied conversational agent](https://arxiv.org/abs/2411.05653)

- Reviews effect of the LLM-based agent persona traits to user experience.
- Manipulation of the personality traits strongly influences social interaction and user experience.


---


[Game-theoretic LLM: Agent Workflow for Negotiation Games](https://arxiv.org/abs/2411.05990)

- Studies with game-theoretic analysis the rationality of LLM-based (with various LLMs) negotiation workflow in various complete-information games and in a incomplete-information game.


---



#### 7th of November 2024

[Interactive Dialogue Agents via Reinforcement Learning on Hindsight Regenerations](https://arxiv.org/abs/2411.05194)

- Simulates interactive dialogue by utilizing hindsight to regenerate optimal task-relevant dialogue data based on initial dialogue data.
- Includes hindsight controller, which takes dialogue input and prefix, then outputs a more desirable action. 


---

[GUI Agents with Foundation Models: A Comprehensive Survey](https://arxiv.org/abs/2411.04890)

- Introduces Survey about GUI Agents.
- Divides LLM-based GUI agents into: GUI Perceiver, Task Planner, Decision Maker, Excecutor and Memory Planner (internal memory: actions/screenshots, external memory: manual construct/auto exploration and self-evolution: transition diagram/documents).
- Identifies challenges related to inference efficiency, self-evolution and real world vs. benchmark gap.

---

[CodeTree: Agent-guided Tree Search for Code Generation with Large Language Models](https://arxiv.org/abs/2411.04329)

- CodeTree: Introduces multi-agent, LLM-based code generation, which improves multi-stage planning/generation/debugging by using tree search.
- Includes Thinker/Solver/Debugger/Critic-agents.
- Critic-agents scores/expands/terminates nodes, which is based on feedback generated by the LLM and the execution feedback on test cases.


---

[CaPo: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation](https://arxiv.org/abs/2411.04679)

- CaPo (Cooperative Plan Optimization): Includes meta-plan generation and progress-adaptive meta-plan & execution
- Meta plan generation consists of analyzing, discuss, create the meta-plan decomposed into subtasks by the various agents.
- Progress-Adaptive Meta-Plan & Execution: agents execute task in the meta plan and dynamically adjust it based on latest progress in multiturn dialogue. 


---

#### 6th of November 2024

[AdaSociety: An Adaptive Environment with Social Structures for Multi-Agent Decision-Making](https://arxiv.org/abs/2411.03865)

- AdaSociety: multi-agent environment to simulate decision making with physical(resources, events, agents skill inventories)/social(establish, alter, form groups, hierarchies)-components. 
- Introduces social states: multilayer directed graph to describe adaptive / dynamic connections, which drive long-term coalition formation / hierarchy.
- Dynamically connects with other agents to establish autonomously non-deterministic connection with the other agent.
- State and action space dynamically advance. 
- Identifies research challenges in collective reasoning, social cognition, adaptation, communication and emergence of new social skills and norms.

---


[MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogue](https://arxiv.org/abs/2411.03814)

- MRJ-Agent: Introduces multi-round dialogue jailbreaking agent, which decomposes harmful queries into multiple sub-queries.
- This widely generalizable jailbreaking-technnique achieves SOTA-level success rates.


---

[From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning](https://arxiv.org/abs/2411.03817)

- StepAgent: Optimizes LLM-agents wit step-wise RL with inspection- and reflection-steps.  


---

#### 5th of November 2024

[SAUCE: Synchronous and Asynchronous User-Customizable Environment for Multi-Agent LLM Interaction](https://arxiv.org/abs/2411.03397)

- SAUCE (Synchronous and Asynchronous User-Customizable Environment): Introduces LLM-based multi agent framework with asynchronous communication feature, where models decide when to speak and what to say.
- Includes experiment(configures discussio, participants, host and end criteria)/session room(manages ongoing experiment and exit criteria)/host (directs interaction)/person(human or LLM).
- Implements LLM-agent personas (and human participant) as class-objects in Python.

---


[AI Metropolis: Scaling Large Language Model-based Multi-Agent Simulation with Out-of-order Execution](https://arxiv.org/abs/2411.03519)

- AI Metropolis: introduces multi agent LLM-based framework, which enables out-of-order execution (parallel processing) of agents by tracking dynamically real dependencies between agents. 
- LLM agents often wait unnecessarily each step to complete, before proceeding, even when it is a false dependency.
- LLM agents can be: blocked (another blocks proceeding), coupled (proceed together), clustered (group needs to synchronize), worker (independent process handling cluster) or controller (main process communicating with workers).
- The related work-section offers comphrensive view on the different scheduling approaches to with agentic AI.


---

#### 1st of November 2024

[DARD: A Multi-Agent Approach for Task-Oriented Dialog Systems](https://arxiv.org/abs/2411.00427)

- DARD (Domain Assigned Response Generation): LLM-based multi agent framework in multi domain & task oriented dialogue.
- Introduces dialogue manager/hotel/attraction/restaurant/train/taxi-agents, external db and dialogue state tracker.
- Uses both fine-tuned LLMs and Sonnet 3.0. Reviews differences in performance.


---

#### 31st of October 2024

[Navigating the Unknown: A Chat-Based Collaborative Interface for Personalized Exploratory Tasks](https://arxiv.org/abs/2410.24032)

- CARE (Collaborative Assistant for Personalised Exploration): Introduces personalized LLM-based multi agent framework, where user interface includes chat/solution/needs-panels.
- Focuses on improving multi-turn contextual understanding, personalization, exploration and reduce cognitive load.
- Employs inquiry/ranking/needs discovery/solution crafting/milestone-agents.


---

#### 30th of October 2024


[EMOS: Embodiment-aware Heterogeneous Multi-robot Operating System with LLM Agents](https://arxiv.org/abs/2410.22662)

- EMOS: multi-agent framework for multi-robot system with embodiment & spatial-aware reasoning/navigation/manipulation/object rearrangement. 
- Includes hierarchical task planning, assignment and actioning. Evaluates success rate, sub-goal success rate, token usage and simulation step.
- Uses "Robot Resume": a self-prompting, instead of "human roleplay" by interpreting the robot URDF files to call robot kinematics tools to generate descriptions of its physical abilities for guiding its planning/action execution. 

---

[Aligning Audio-Visual Joint Representations with an Agentic Workflow](https://arxiv.org/abs/2410.23230)

- AVAgent: Adapts audio signal with visual data using LLM-based agent framework, which plans edits of the audio signals and reflection with VLM to evaluate the modifications and uses tool to convert video and audio modality to text.


---


#### 29th of October 2024

[BENCHAGENTS: Automated Benchmark Creation with Agent Interaction](https://arxiv.org/abs/2410.22584)

- BENCHAGENTS: Introduces LLM-agent framework automating benchmark creation, which includes four components: planning/generation/data verification/evaluation-agents.
- Dynamic benchmarks help to identify common failure modes/model differences, while LLM models improve quickly.
- Planning includes: prompt/task-specific parameters/constraints (positive/negative/positional/sequencing/conditional/iterative).


---

#### 28th of October 2024


[Asynchronous Tool Usage for Real-Time Agents](https://arxiv.org/abs/2410.21620)

- Asynchronous AI agents: Introduces asynchronous, parallel thought processing and real-time tool use based on event-driven finite state-machines.
- Time stamp is in the messages to enable clock awareness, which enables time-constrained tasks. 
- Event states include idle/listening/generating/emitting.


#### 25th of October 2024

[Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models](https://arxiv.org/abs/2410.20007)

- CoPlanner (Cooperative Planner): Improves reasoning capabilities of LLM by separating reasoning steps. Each agent gets assigned unique reasoning step.
- Includes planning agent and reasoning agent.
- Pre-defines 10 human cognition-based meta-strategies. Includes 5 logical reasoning methods: deduction/induction/abduction/analogy/contradiction and four problem solving methods: decomposition/enumeration/elimination/reflection and meta-strategy: finish to indicate end of reasoning.

  
---

[VisionCoder: Empowering Multi-Agent Auto-Programming for Image Processing with Hybrid LLMs](https://arxiv.org/abs/2410.19245)

- VisionCoder: Multi agent framework with team leader, module leader, function coordinator and development group
- Identifies excellent two aspects for the Agent-definitions: structural (explains the agents place in the overall structure/scope/responsibilities) and functional (operational steps/reasoning path expected from the agent and the output format requirements).
- Includes bi-directional workflow: hierarchical tasks are divided into smaller units (forward task flow) and then restored back (backward task flow) from smaller pieces to larger units. Pair programming-concept includes coder and tester: coder produces code, tester reviews it and then the roles are reversed. The pair programming step is repeated three rounds with code execution with incorporation of the error messages to get final working code. 


---

[Designing LLM-Agents with Personalities: A Psychometric Approach](https://arxiv.org/abs/2410.19238)

- Reviews creation of psychometrically sound LLM-based agents based on the theory about big 5 personality traits (openess/conscientiousness/extraversion/agreeabless/neuroticism).


---

[FISHNET: Financial Intelligence from Sub-querying, Harmonizing, Neural-Conditioning, Expert Swarms, and Task Planning](https://arxiv.org/abs/2410.19727)

- FISHNET: Multi agent-framework for insights from SEC regulatory forms. Includes sub-querying (converts query into sub-queries)-, task planning- , experts (Swarm Intelligence)-, harmonizer(routes to specific expert based on embedding match vs. agent persona/tables description)-agents and long term memory.
- Expert agents consist of: n-port-, n-mfp-, adv-, n-cen-, n-csrv- and 13f-agents, which are experts in different forms related to SEC regulations.


---


[AGENT-CQ: Automatic Generation and Evaluation of Clarifying Questions for Conversational Search with LLMs](https://arxiv.org/abs/2410.19692)

- Agent-CQ: Introduces a framework for generating and evaluating conversational search questions and answers. Includes generation (question generation / filtering / answer generation)- and evaluation (multiple LLM-judge calls to review generated questions/answers)-stages.


---

[EDGE: Enhanced Grounded GUI Understanding with Enriched Multi-Granularity Synthetic Data](https://arxiv.org/abs/2410.19461)

- EDGE: Introduces framework to generate training data for GUI-tasks in the internet. Introduces element- and action-grounding. 


---


[Investigating the Role of Prompting and External Tools in Hallucination Rates of Large Language Models](https://arxiv.org/abs/2410.19385)

- Investigates prompting techniques and finds simpler is often better and best prompts are problem specific.
- In math problems self-consistency with majority vote works well, Chat protect helps to manage amount of hallucinated answers and Self-Verification worked well with MMLU.


---

[AgentSense: Benchmarking Social Intelligence of Language Agents through Interactive Scenarios](https://arxiv.org/abs/2410.19346)

- AgentSense-benchmark: introduces a multiturn evaluation of LLM-agents regards social intelligence. Focuses on goal competition and implicit reasoning.
- Character-info includes: attributes/relationships/rules of replacement. Scenarios include: background/characters/social goals/private info.
- Includes a sample agent-prompt. 


---


#### 24th of October 2024

[Unbounded: A Generative Infinite Game of Character Life Simulation](https://arxiv.org/abs/2410.18975)

- Unbounded: Introduces a conceptual and technical implementation of concept called "generative infinite game". 
- Addresses semantically alignedconsistent environment/characters.
- Trained an LLM based game engine game engine (generating coherent and real-time game mechanisms, narratives and contextual character responses) and "Regional IP-Adapter", which creates visually consistent characters/environments between multiple images while applying creativity. Regional IP-Adapter tracks changes overtime, so if your character gets injured in forest, the injury remains in the following images and the character still wears same clothes, while giving creative touches to the visuals. 


---

[AR: Operating System Control via State-Aware Reasoning and Re-Planning](https://arxiv.org/abs/2410.18963)

- OSCAR: Introduces GUI-agent with unified control interfaces / GUI grounding (dual grounding) / exploration-based simulation and re-planning (task driven replanning of only specific tasks).
- Works both in smartphones and desktop OS. Reviews GUI agents. Includes system prompts.
- Agent states include: init/observe/plan/execute/error/verify/fail/success/reset. Includes context memory.


---

[Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs](https://arxiv.org/abs/2410.18451v1)

- Skywork-Reward: introduces methods to enhance reward modeling for LLMs, focusing on data-centric techniques.
- It proposes data selection and filtering strategies for high-quality preference datasets, resulting in Skywork-Reward data collection, and develops Skywork-Reward model series including Skywork-Reward-Gemma-27B and Skywork-Reward-Llama-3.1-8B.
- This work enhances performance of top-ranked models on RewardBench, highlighting practical impact in preference learning applications.


---


[PDL: A Declarative Prompt Programming Language](https://arxiv.org/abs/2410.19135)

- PDL (Prompt Declarative Language): Introduces declarative and data-oriented language based on YAML to construct LLN prompt programs. Every PDL program is a valid YAML-document with PDL-schema. 


---

[From a Tiny Slip to a Giant Leap: An LLM-Based Simulation for Fake News Evolution](https://arxiv.org/abs/2410.19064)

- FUSE (Fake News evlUtion Simulation framEwork): Reviews the way true news convert into fake news with LLMs. Includes LLM-based agents: spreaders/commentators/verifiers/bystanders.
- The simulation evolves with a module called News Evolution Simulator. 
- Includes content deviation metrics.


---

[PRACT: Optimizing Principled Reasoning and Acting of LLM Agent](https://arxiv.org/abs/2410.18528)

- PRAct (Principled Reasoning and Acting)-framework: improves action understanding of agents by including action principles. Introduces RPO (Reflective Principle Optimization).


---

#### 23rd of October 2024

[ASYNCHRONOUS RLHF: FASTER AND MORE EFFICIENT OFF-POLICY RL FOR LANGUAGE MODELS](https://arxiv.org/abs/2410.18252)

- Asynchronous RLHF (Reinforcement Learning from Human Feedback): A framework that separates generation and learning in RLHF, enabling asynchronous generation of new samples while simultaneously training on old samples.
- Online but off-policy, faster training, more compute-optimal scaling, training LLAMA 3.1 8B on instruction-following task 40% faster while matching final performance.
- This framework addresses the computational inefficiency of the dominant paradigm for RL finetuning of LLMs by separating generation and learning, leading to faster training and more efficient use of resources.



[GraphTeam: Facilitating Large Language Model-based Graph Analysis via Multi-Agent Collaboration](https://arxiv.org/abs/2410.18032)

- GraphTeam: LLM-based collaborative multi agent and graph-based system using three modules: input-output normalization/external knowledge retrieval/problem solving.
- Includes question(reformats question)/search/coding/reasoning/answer-agents. 
- Constructs to knowledge graphs: documentation and experience. 


---

[Real-World Robot Applications of Foundation Models: A Review](https://arxiv.org/abs/2402.05741)

- This paper provides an overview of the practical application of foundation models in real-world robotics.
- The review emphasizes the replacement of specific components within existing robot systems, input-output relationships, perception, motion planning, and control.
- The paper concludes with a discussion of future challenges and implications for practical robot applications.

---



[MiniFed : Integrating LLM-based Agentic-Workflow for Simulating FOMC Meeting](https://arxiv.org/abs/2410.18012)

- MiniFed: Simulates real world Federal Reserve FOMC-meetings using LLM-agent based multi-agent framework.
- Consists of initialization/data collection/simulation/decision making/evaluation.


---

[Guide for Defense (G4D): Dynamic Guidance for Robust and Balanced Defense in Large Language Models](https://arxiv.org/abs/2410.17922)

- G4D (Guide for Defense): LLM-based multi agent with external knowledge to discover user intent as safe with a defense framework against jailbreaks.
- Includes intention detector (intention extraction, key entities identification and information retrieval)/question paraphraser/safety analyzer-components.


---

[An Intelligent Agentic System for Complex Image Restoration Problems](https://arxiv.org/abs/2410.17809)

- AgenticIR: VLM/LLM-agent based image restoration using perception/scheduling/reflection/rescheduling/execution-agents.
- Includes Rollback-mechanism, where agent returns previous working stage, when an issue.


---

[ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](https://arxiv.org/abs/2410.17657)

- ReflecTool: Introduces clinical agent, using progressively built long-term memory to assist domain-specific tool selection and improve tool usage. Includes optimization and inference stages. 


---

[Navigate Complex Physical Worlds via Geometrically Constrained LLM](https://arxiv.org/abs/2410.17529)

- Reviews LLMs-capability to reconstruct physical world from textual knowledge. 
- Uses LLM-based multi agent framework with scenery designer/object designer/object manufacturer/arranger-agents and geometric constraint solver and generic algorithm.


---

#### 21st of October 2024

[Long Term Memory: The Foundation of AI Self-Evolution](https://arxiv.org/abs/2410.15665)

- Reviews and defines AI Self-Evolution-capability and Long Term Memory (LTM).
- Identifies benefits in Personalized Models. 
- Identifies limitations in prompt-based memory mechanisms. 


---


[Improving Parallel Program Performance Through DSL-Driven Code Generation with LLM Optimizers](https://arxiv.org/abs/2410.15625)

- Designs Domain Specific Language (DSL) in mapper (maps computations to processors like GPUs, CPUs, etc.) generation related to assignment of compute / memory. 
- The DSL helps to manage high-level inference decisions without interacting with the low-level C++ code APIs.


---

#### 20th of October 2024

[Redefining Proactivity for Information Seeking Dialogue](https://arxiv.org/abs/2410.15297)

- Introduces Information Seeking Dialogue (ISD) agents with proactiveness to include information relevant to the user query.
- Introduces new prompting strategies: 3-step CoT and 3-in-1 CoT.


---

#### 18th of October 2024

[Teaching Models to Balance Resisting and Accepting Persuasion](https://arxiv.org/abs/2410.14596)

- PBT (Persuasion Balanced Training): Uses multi-agent recursive dialogue trees to train models with preference optimization to accept persuasion in acceptable situations. PBT-trained model outperform in multi-agent debates.
- Agents argue based on logical reasoning/emotional appeal/established credibility.
- Refers to research by [Woolley et al. (2010)](https://www.researchgate.net/publication/47369848_Evidence_of_a_Collective_Intelligence_Factor_in_the_Performance_of_Human_Groups), where group intelligence is argued to be driven by diversity/turn-taking/social sensitive, rather than individual intelligence.


---

#### 18th of October 2024

[Make LLMs better zero-shot reasoners: Structure-orientated autonomous reasoning](https://arxiv.org/abs/2410.19000)

- SARA (Structure-oriented Autonomous Reasoning Agents): Introduces multi agent LLM-based reasoning framework with structure-oriented analysis by refinement and RAG.
- Outperforms in some cases few-shot learning.
- Includes reason (structured oriented analysis)-, retrieval-, refinement-agents and shared memory. Includes prompts used.


---

[AI can help humans find common ground in democratic deliberation](https://www.science.org/doi/10.1126/science.adq2852)

- Habermas Machine: AI mediation technique promoting fair/inclusive debate.
- LLM-agent opinions/critiques refine group statement to maximize group approval.
- Aims to improve collective decision making in political discussion/conflict resolution.


---

#### 17th of October 2024

[Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation](https://arxiv.org/abs/2410.13232)

- Proposes World-Model-Augmented (WMA) web agent by simulating planned actions to obtain outcome before using them (metacognitive monitoring) in order to avoid performing erroneous moves. Reviews LLMs lack of capability to avoid performing errors, which humans can easily avoid by posing world model. 
- Introduces "Transition-focused observation abstraction": world model generates free-form important state differences before / after. Agent simulates outcomes of each possible action with world model and reward model asesses each one. 
- Includes prompts.

---

[Chain of Ideas: Revolutionizing Research in Novel Idea Development with LLM Agents](https://arxiv.org/abs/2410.13185)

- CoI (Chain-of-Ideas): CoI-agent generates research ideas comparable to human-level by organizing literature in a chain structure to avoid logical inconsistencies in ideation.
- Improves LLMs research ideation capabilities. Consists of three steps: CoI-construction (identifies current trends), Idea generation (consolidates ideas) and Experience design (final experiment design).
- CoI-prompts include: converting topic in search query for literature retrieval/evaluation of paper relevance to the topic/extract research paper ideas, experiments, entities and reference/summarising trends of the this CoI. 
- Idea generation prompts include: predict future trends / generate ideas / novelty check of ideas.
- Experiment design prompts include: generate experiment design / review experiment design / obtain queries to edit experiment design / refine experiment design. 



---
[AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents](https://arxiv.org/abs/2410.13825)

- AgentOccam: Refines LLM-agent observation/action space to improve its performance in web tasks with three methods. Sets SOTA in WebArena.
- Introduces planning actions: branching and pruning. Minimizes trivial interaction space. Removes unnecessary web content. 
- Agent prompt includes general instructions (task description/output specification/action specification) and Online Task Information.
- Simplifies web content/selectively replays web elements/selectively replays past pages.

---

[AdaSwitch: Adaptive Switching between Small and Large Agents for Effective Cloud-Local Collaborative Learning](https://arxiv.org/abs/2410.13181)

- AdaSwitch: Uses local agents for basic and cloud agent for complex tasks.
- Includes self-practicing, collaborative examination and reflective learning steps. 


---

[Harnessing Webpage UIs for Text-Rich Visual Understanding](https://arxiv.org/abs/2410.13824)

- Introduces MultiUI-dataset of 1 million websites for web / UI agents. 


---

[Rapid and Automated Alloy Design with Graph Neural Network-Powered LLM-Driven Multi-Agent Systems](https://arxiv.org/abs/2410.13768)

- Multi-agent system including LLMs, AI agents (multi modal LLM-agents) and GNNs to discover automatically new metallic alloys.
- The LLM-agent roles include: planner-, executor-, coder-, reviewer- and multi-modal-agents.  


---

[A Comparative Study on Reasoning Patterns of OpenAI's o1 Model](https://arxiv.org/abs/2410.13639)

- Reviews o1-model against other test-time compute methods like BoN/Self-Refin/Agent workflow. 
- Identifies 6 reasoning patterns with o1-model: systematic analysis/method reuse/divide & conquer / self-refinement / context identification / emphasizing constraints.


---

[MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling](https://arxiv.org/abs/2410.13610)

- MeNTI-framework chooses appropriate meta-tool, fills data according to the meta-tool documentation and nested-calling verifies task completion. 


---

[Integrating Large Language Models and Reinforcement Learning for Non-Linear Reasoning](https://arxiv.org/abs/2410.13501)

- RL guides LLM's exploration. The architecture includes: LLM-module/validation module/reasoning tree/RL agent. Applied in code generation. 
- LLM module generates n-candidates, validation module reviews characteristics of each candidate, the features of each review are added to reasoning tree and finally RL explores this reasoning tree to decide the node to explore next. 


---

[Metacognitive Monitoring: A Human Ability Beyond Generative Artificial Intelligence](https://arxiv.org/abs/2410.13392)

- Reviews metacognition monitoring abilities of LLMs.


---

[RescueADI: Adaptive Disaster Interpretation in Remote Sensing Images with Autonomous Agents](https://arxiv.org/abs/2410.13384)

- ADI (Adaptive Disaster Interpretation)-framework: introduces an multimodal LLM-agents interpreting disaster scenarios using tools. Introduces RescueADI-dataset. 
- ADI-framework includes perception/recognition/planning/tools-modules.


---

#### 16th of October 2024

[Revealing the Barriers of Language Agents in Planning](https://arxiv.org/abs/2410.12409)

- Reviews planning capabilities of LLMs and identifies current models like o1 only achieve 15.6% performance in real-world tasks. 
- Identifies two core issues: interpretation of constraints/loss of focus in long-horizon planning tasks.
- Episodic and parametric memory help, but do not resolve the lack of planning capabilities. 

---

[Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](https://arxiv.org/abs/2410.13080)

- GCR (Graph-Constrained Reasoning): Integrates Knowledge Graph (KG) into LLM decoding to reduce hallucinations in reasoning.
- Uses KG-Trie method. 

---

[Evaluating Software Development Agents: Patch Patterns, Code Quality, and Issue Complexity in Real-World GitHub Scenarios](https://arxiv.org/abs/2410.12468)

- Reviews LLM-agents ability to patch code, suggesting smaller sub-tasks to patch code to be easier for LLM-agents.

---

[JudgeBench: A Benchmark for Evaluating LLM-based Judges](https://arxiv.org/abs/2410.12784)

- JudgeBench-benchmark: Evaluates LLM-judge agents, which focuses on instruction following/factuality/logic/style.


---

[SAC-GLAM: Improving Online RL for LLM agents with Soft Actor-Critic and Hindsight Relabeling](https://arxiv.org/abs/2410.12481)

- SAC-GLAM: Proposes a more autonomous LLM-agents based on adaptation of SAC (Soft Actor-Critic) and HER (Hindsight Experience Replay) for LLM-agents in multi-goal RL environment to perform sequential decision making tasks.
- Reviews LLM-agents moving from external objective driven towards more autotelic ("self" + "goals") with an intrinsic purpose rather than extrinsic. 


---

[Robust RL with LLM-Driven Data Synthesis and Policy Adaptation for Autonomous Driving](https://arxiv.org/abs/2410.12568)

- RAPID: Improves RL performance in autonomous driving with LLM-reasoning. Uses LLM-agent data for offline RL distillation and then adapts online RL-agent with LLM-data.

---

[Enhancing LLM Trading Performance with Fact-Subjectivity Aware Reasoning](https://arxiv.org/abs/2410.12464)

- FS-Reasoning Agent: introduces LLM-based multi-agent trading framework by splitting reasoning processes between factual and subjective reasoning.
- Includes Statistics/Fact reasoning/Fact/Subjectivity/Subjectivity reasoning/Trading/Reflection agents.
- Concludes, that superiority of the LLM model is not sufficient to guarantee it outperforming multi-step reasoning.


---


[MedAide: Towards an Omni Medical Aide via Specialized LLM-based Multi-Agent Collaboration](https://arxiv.org/abs/2410.12532)

- MedAide: Introduces LLM-based multi-agent framework, which includes query input/query rewriting/intent recognition/agent collaboration. 
- Activates specialised agents (own prompt template) dynamically by recognizing intent. 
- Includes contextual encoder. 

---

[Aegis:An Advanced LLM-Based Multi-Agent for Intelligent Functional Safety Engineering](https://arxiv.org/abs/2410.12475)

- Aegis: LLM-based multi-agent framework for FSRs (Functional Safety Requirements) and HARA (Hazard Analysis and Risk Assessment). 


---

#### 15th of October 2024

[G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks](https://arxiv.org/abs/2410.11782)

- G-Designer: introduces designer of multi-agent LLM-graphs based on MACP. Includes Materials/Construct/Design/Optimize-steps.  
- Proposes a LLM-agent communication protocol for multi-agent systems called MACP. MACP includes performance/adaptability/robustness.


---

[AGENTiGraph: An Interactive Knowledge Graph Platform for LLM-based Chatbots Utilizing Private Data](https://arxiv.org/abs/2410.11531)

- AGENTiGraph (Adaptive Generative ENgine for Task-based Interaction and Graphical Representation): LLM-based multi-agent knowledge management framework with knowledge graphs.
- Includes knowledge extraction/integration/real-time visualization.
- Dynamically interprets user intent/manage tasks/integrate new knowledge. Classifies tasks. Extracts key concepts. Constructs knowledge graphs. Includes prompts used. 


---

[Revisiting Benchmark and Assessment: An Agent-based Exploratory Dynamic Evaluation Framework for LLMs](https://arxiv.org/abs/2410.11507)

- TestAgent-framework: quantitative/qualitative benchmark using agent-based evaluation with RL, multi-turn interaction from knowledge base/topics of interests.


---

#### 14th of October 2024

[AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762)

- AFlow: Optimises LLM-agent workflow with MCTS.
- Includes search space (node, operators, code represented edges), search via AFliw and Search result (math, Q&A and code generation workflows.)


---

#### 10th of October 2024


[Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning](https://arxiv.org/abs/2410.08146)

- PAVs (Process Advantage Verifiers): is a framework that trains verifiers to predict progress in multi-step reasoning by measuring the change in likelihood of a correct response under a prover policy.
- PAVs improve exploration during test-time search and online RL, using complementary prover policies, and are more compute-efficient than ORMs.
- This framework enables more efficient and accurate reasoning in large language models by providing a better way to measure progress in multi-step reasoning.


---

[Multi-Agent Collaborative Data Selection for Efficient LLM Pretraining](https://arxiv.org/abs/2410.08102)

- Introduces LLM-based multi-agent system for efficient LLM pretraining data selection. LLM converges faster in the pretraining and the method improves LLM output quality.
- The Data console integrates data inisghts dynamically from the different agents during the training process. 
- Agent console include quality/domain/topic-agents. Includes as well memory.


---


[Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System](https://arxiv.org/abs/2410.08115)

- Optima (OPTImising effectiveness and efficiency for LLM-based Multi-Agent systems): Introduces framework to train LLM-based multi-agent system (MAS). 
- Includes 4 iterative steps: Generate/Rank/Select/Train.
- Investigates scaling laws of inference compute.
- Optima helps to make LLMs highly efficient conversationalists.

---


[DelTA: An Online Document-Level Translation Agent Based on Multi-Level Memory](https://arxiv.org/abs/2410.08143)

- DelTA (Document-level Translation Agent): Introduces translation LLM-agent using multi-layer memory components to improve translation consistency/quality.
- Memory components include: Proper noun memory(to apply correct terminology)/Bilingual summary/long-term/short-term-memory units.


---

[Mars: Situated Inductive Reasoning in an Open-World Environment](https://arxiv.org/abs/2410.08126)

- Mars: Introduces framework for Situated Inductive Reasoning-benchmark and a framework with LLM-agents called: IfR (Induction from Reflection). 
- The paper identifies two critical components for inductive reasoning: situatedness (situational context) and abstractiveness (abstract conclusions).
- IfR-framework includes task proposer/planner/controller/reflection-steps, rule library (when this, do that) and skill library. The LLM-based reflection-step induces new rules, which actual LLMs struggle currentyly.


---

[Benchmarking Agentic Workflow Generation](https://arxiv.org/abs/2410.07869)

- Introduces WorFEBench-benchmark for unified workflow generation and WorFEval evaluation protocol of workflows for LLM-agents.


---

#### 9th of October 2024

[AgentBank: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories](https://arxiv.org/abs/2410.07706)

- Samoyed: Introduces LLM-models fine-tuned with AgentBank-dataset for general agent tasks.
- AgentBank-dataset includes dimensions: reasoning/math/programming/web/embodied AI.


---


[Smart Audit System Empowered by LLM](https://arxiv.org/abs/2410.07677)

- Introduces Smart Audit System with LLMs, which include dynamic risk assessment model/manufacturing compliance copilot/Commonality analysis agent. Developed by Apple researchers.
- Dynamic risk assessment model adjusts audit: focus/sample size/critical items/resource allocation.  
- Manufacturing compliance copilot self-adjusts its the knowledge base with new information.
- Commonality analysis agent manages an autonomous agent conducting real-time analysis to custom requests, in order to drive supplier improvements. Includes planning/memory/tools/selecting and usage of tools/generating responses. 


---


[Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making](https://arxiv.org/abs/2410.07166)

- Introduces Embodied Agent Interface-benchmark for embodied decision making LLM-agents.
- Reviews four critical capabilities: Goal interpretation, Subgoal decomposition, Action sequencing and Transition modelling.


---

[I Want to Break Free! Anti-Social Behavior and Persuasion Ability of LLMs in Multi-Agent Settings with Social Hierarchy](https://arxiv.org/abs/2410.07109)

- zAImbardo-framework: Introduces LLM-agent simulation between prisoner/guard-agents using prompts, which are either shared or private.
- Shared prompts: communication rules/environment description/research oversight/risks. Private prompts: Starting prompt/personality/goals.


---

[Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology](https://arxiv.org/abs/2410.07087)

- Introduces UAV navigation agent using MLLM. Includes three levels of assistants: constant/difficult situations/hazard situations.


---

[MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses](https://arxiv.org/abs/2410.07076)

- Moose-Chem: multi-agent framework to discover novel chemistry research hypothesises from given information.


---

[Seeker: Enhancing Exception Handling in Code with LLM-based Multi-Agent Approach](https://arxiv.org/abs/2410.06949)

- Seeker: introduces LLM-based multi-agent framework for exception handling with planner/detector/predator/ranker/handler-agents.


---

[ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](https://arxiv.org/abs/2410.06703)

- ST-WebAgentBench-benchmark: Evaluates safety and trustworthy of web agents against performing undesired operations in business/user applications.


---

[Do great minds think alike? Investigating Human-AI Complementarity in Question Answering with CAIMIRA](https://arxiv.org/abs/2410.06524)

- CAIMIRA (Content-Aware, Identifiable, Multidimensional, Item Response Analysis)-framework: Reviews differences between humans and SOTA-level LLMs in QA-tasks in reasoning and textual understanding. 


---

#### 8th of October 2024

[AgentSquare: Automatic LLM Agent Search in Modular Design Space](https://arxiv.org/abs/2410.06153)

- AgentSquare: Introduces modular LLM-agent framework using module evolution, recombination and performance predictor(skip unpromising agent designs). - The framework optimizes agent designs with Planning/Reasoning/Tool use/Memory-modules.
- Introduces the research concept of MoLAS (Modularized LLM Agent Search): the automatic optimization of LLM-agent designs from succesfull designs.
- Includes search-, program-level search- and performance predictor-meta prompts. 


---

#### 7th of October 2024

[LLMs Are In-Context Reinforcement Learners](https://arxiv.org/abs/2410.05362)

- In-Context Reinforcement Learning (ICRL): Introduces ICRL-algorithm (increases test-time compute), which effectively learns reward from a classification task. The explorative-version concentrates on positive episodes and stochasticity.
- Naive ICRL explores poorly.

---

[Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents](https://arxiv.org/abs/2410.05130)

- GraphAgent-Reasoner (GAR): explicit and precise graph-reasoning with multi-agent collaboration.
- Works to solve real-world graph-reasoning such as webpage ranking,
- Distributes tasks into nodes (over 1000) to multiple agents collaborating between each other.
- Includes stages: Algorithmic establishment (retrieve/initialisation/adjust/design), Distributed execution (Master LLM assigns task, agent network communicates) and Master summarisation (termination/aggregation/conclusion).
- Master LLM defines for each problem 6 components: State/Message/Initialization/Send/Update/Termination.

---

[Grounding Partially-Defined Events in Multimodal Data](https://arxiv.org/abs/2410.05267)

- Reviews event extraction from unstructured video data using multimodal event analysis with LLMs.

---

[GLEE: A Unified Framework and Benchmark for Language-based Economic Environments](https://arxiv.org/abs/2410.05254)

- Introduces GLEE (Games in Language-based Economic Environments)-benchmark, which reviews LLMs in two-player economic game families of bargaining, negotiation andd persuasion.


---

#### 26th of September 2024

[AssistantX: An LLM-Powered Proactive Assistant in Collaborative Human-Populated Environment](https://arxiv.org/abs/2409.17655)

- AssistantX: multi LLM-agent framework (PPDR4X) to help users achieve goals in virtual / physical environments.
- PPDR4X-framework includes short term memory (initial instructions/dialogue data/agent thoughts/cyber tasks/real world tasks), long-term memory (environment information), perception-agent, planning-agent, reflection agent and decision agent. 


---

[Control Industrial Automation System with Large Language Models](https://arxiv.org/abs/2409.18009)

- Introduces multi LLM-agent industrial control system, which consists of summarizer-, manager- (planning level), event log manager-, operator-agents (control-level) and command line/event log memory/prompt templates/events/function calls.


---

[Compositional Hardness of Code in Large Language Models -- A Probabilistic Perspective]()

- Reviews the difficulty of processing multiple sub-tasks within single LLM call with ICL to produce correct solution, which is called "In-Context Hardness of Composition".
- Refers to new term called "Screening", which refers to LLMs capacity to isolate the relevant context. For example LLM with capacity to perform two tasks, may fail performing both within same context.
- Finds, that is better to distribute tasks to multiple LLM-agents, when task becomes complex. Offers a literature review of the CoT problem solving and agents-research intersection.

---

#### 25th of September 2024

[Turn Every Application into an Agent: Towards Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](https://arxiv.org/abs/2409.17140)

- AXIS: Priorites task completing API-calls above UI-agent actions, which decrases task completion time and cognitive workload.
- It is more useful to generate efficient API-call agent using programmatic API, than slower human-like UI agent.
- Includes Explorer-, Follower-, Monitor-, Generator-, Evaluator- and Translator-agents.
- Enables converting any application, with basic API/documentation and: environment state interface/basic action interface, into agent. Uses self-exploratory framework to identify control elements.


---

[A Roadmap for Embodied and Social Grounding in LLMs](https://arxiv.org/abs/2409.16900)

- Reviews the grounding of LLMs with physical world. Highlights the importance of social grounding of physical experiences. For example a child can build understanding of heavy objects just by observing an adult trying to lift a heavy box.
- Interesting ideas about the way human perception in physical world.


---

[Plurals: A System for Guiding LLMs Via Simulated Social Ensembles](https://arxiv.org/abs/2409.17213)

- Introduces Plurals-framework: generates diverse agents (stakeholder) based on demographic data to interact diverse opinions using a structrured debate and moderator.
- The demographic data is basis for generating the agents, which helps to tune the messages to specific audiences.
- Includes Structures, which forces LLM-agents to share information with a properly formed structure.
- Moderator-agent then summarises this discussion by trying to take into account the diverse opinions.


---

[Language Grounded Multi-agent Communication for Ad-hoc Teamwork](https://arxiv.org/abs/2409.17348)

- Grounds MARL agent communication with LLM generated synthetic data, which improves communicatio and zero-shot collaboration between agents.

---

#### 24th of September 2024

[Synatra: Turning Indirect Knowledge into Direct Demonstrations for Digital Agents at Scale](https://arxiv.org/abs/2409.15637)

- Synatra: is an approach that transforms indirect knowledge into direct supervision for digital agents at scale.
- Synatra leverages LLMs to repurpose human-created tutorials and ungrounded observations into executable action sequences, and includes a 7B CodeLlama model.
- This framework enables more effective and cheaper training of digital agents compared to human demonstrations.


---

[MOSS: ENABLING CODE-DRIVEN EVOLUTION AND CONTEXT MANAGEMENT FOR AI AGENTS](https://arxiv.org/abs/2409.16120)

- MOSS (IIM-oriented Operating System Simulation): is a framework integrating code generation with a dynamic context management system.
- MOSS uses Inversion of Control (IoC) container, decorators, maintains Python context, isolates local variables, preserves runtime integrity, and enables code-driven evolution.
- This framework enhances efficiency and capabilities of AI agent development, moving towards Turing-complete agents.


---


---


#### 23rd of September 2024

[ERABAL: Enhancing Role-Playing Agents through Boundary-Aware Learning](https://arxiv.org/abs/2409.14710)

- ERABEL: Introduces boubdary-aware role playing framework to maintain role comsistency in multiturn conversation.
- Includes dialogue planner/topic manager/question generator/response generator-agents.
- Includes prompts for esch agent.


---

#### 22th of September 2024

[BACKTRACKING IMPROVES GENERATION SAFETY](https://arxiv.org/abs/2409.14586)

- Backtracking: is a technique that allows language models to "undo" and recover from their own unsafe generation through the introduction of a special [RESET] token.
- Backtracking can be incorporated into either SFT or DPO training, provides protection against adversarial attacks, and improves safety without regression in helpfulness.
- This method provides a new approach to improve language model safety by allowing models to recover from unsafe generations.




#### 20th of September 2024

[RRM: Robust Reward Model Training Mitigates Reward Hacking](https://arxiv.org/abs/2409.13156)

- RRM (Robust Reward Model): Reviews reward models ability to differentiate signal from the genuine context and irrelevant information to decide preference. Proposes usage of causal graph.
- Produces more robust reward model.

---

[ChainBuddy: An AI Agent System for Generating LLM Pipelines](https://arxiv.org/abs/2409.13588)

- ChainBuddy: Includes requirements gathering agent (primary user goal/list of req./user preferences/suggested Cot strategy), planner agent (includes replanner), task-specific agents, connection agent and post-hoc reviewer agent.

---

[Minstrel: Structural Prompt Generation with Multi-Agents Coordination for Non-AI Experts](https://arxiv.org/abs/2409.13449)

- Minstrel: a multi-agent framework for automated prompt optimization. Prompts are constructed using role, profile, constraints, goals, initialization and examples, workflow, skills, suggestions, background, style, output format and command modules.
- Agents are assigned to working groups in charge of similar small tasks.

---

[ShizishanGPT: An Agricultural Large Language Model Integrating Tools and Resources](https://arxiv.org/abs/2409.13537)

- ShizishanGPT: LLM agent for answering with agriculture-based RAG.


---


#### 19th of September 2024

[Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917)

- SCoRe (Self-Correct via Reinforcement Learning): Increases LLMs capacity to self-correct via multi-turn Reinforcement Learning.
- Achieves positive intrinsic self-correction performance as first model.

---

[AutoVerus: Automated Proof Generation for Rust Code](https://arxiv.org/abs/2409.13082)

- AutoVerus: LLM generates correctness proofs for Rust-code using multi-agent framework (proof generation, refinement and debugging).


---

#### 17th of September 2024

[LLM-Agent-UMF: LLM-based Agent Unified Modeling Framework for Seamless Integration of Multi Active/Passive Core-Agents](https://arxiv.org/abs/2409.11393)

- LLM-agent UMF (Unified Modelling Framework): Introduces modular LLM-agent framework, which includes core agent coordinating with planning, memory, profile, action and security modules.
- Proposes various multi agent frameworks.
- Proposes active and passive information types. 
- Includes lots of useful ideas for each component.


---

[NVLM: Open Frontier-Class Multimodal LLMs](https://arxiv.org/abs/2409.11402)

- NVLM: frontier level VLM model and high performance as LLM only.
- Finds, that dataset quality and task diversity impact more than scale.
- Finds positive transfer from image to text only modality.


---

[P-RAG: Progressive Retrieval Augmented Generation For Planning on Embodied Everyday Task](https://arxiv.org/abs/2409.11279)

- P-RAG: Introduces iteratively updated RAG (self-iterations). P-RAG adds more task-specific knowledge.
- The RAG stores the following information: goal instruction, scene graph, history and done.


---

[EmPO: Emotion Grounding for Empathetic Response Generation through Preference Optimization](https://arxiv.org/abs/2406.19071)

- EmPO: Introduces the EmpatheticDialogues-dataset for fine tuning LLMs with empathic response generation (ERG). 


---


#### 16th of September 2024

[Instigating Cooperation among LLM Agents Using Adaptive Information Modulation](https://arxiv.org/abs/2409.10372)

- SLA (Strategic LLM Agent): combines LLM agents (SLAs) and RL-agent called Pro-social Promoting Agent (PPA) to increase cooperation rate.
- Adjusts dynamically access to SLA's information (cooperation history with neighbours, average) to increase facilitate social interaction.


---

[Cognitive Kernel: An Open-source Agent System towards Generalist Autopilots](https://arxiv.org/abs/2409.10277)

- Cognitive Kernel: introduces autopilot-like LLM-agent with access to internet with the web browser (appears to use Playwright-library) to interact "human-like" manner (click, scroll, etc).
- The LLM agent interacts with user and task environment. Includes reasoning kernel, memory kernel and perception kernel.
- LLM is fine tuned to interact with the environment through atomic actions, which a normal person could perform, rather than API call.
- Offers interesting ideas for each sub-compoment, as each includes plenty of detailed functionalities. 


---

[Central Answer Modeling for an Embodied Multi-LLM System](https://arxiv.org/abs/2406.10918)

- CAM (Central Answering Model): Introduces CAM-framework, where instead of LLM-agent directly answering question, multiple LLM-agent instances generate answer and a central LLM-agent responds to the question.


---


#### 15th of September 2024

[RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation](https://arxiv.org/abs/2409.09584)

- RethinkMCTS: conducts thought-level searches before generating code and adds both verbal feedback to refine thoughts and code execution feedback from incorrect code. 
- Increasing the number of rethink- and rollout-operations improve code generation.


---


#### 14th of September 2024

[PeriGuru: A Peripheral Robotic Mobile App Operation Assistant based on GUI Image Understanding and Prompting with LLM](https://arxiv.org/abs/2409.09354)

- PeriGuru: LLM-agent for GUI with perception, decision and action steps.


---

[Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models](https://arxiv.org/abs/2409.09345)

- Introduces task-relevant Q-value model for guiding action selection.
- Includes review of the different methods to improve reasoning, such as LLMs using MCTS.


---


#### 13th of September 2024

[Agents in Software Engineering: Survey, Landscape, and Vision](https://arxiv.org/abs/2409.09030)

- Introduce LLM-agents with perception, memory and actions for SW engineering. Includes multi-agent workflow with feedback, refinement and roles.
- Actions include internal (reasoning, learning and retrieval) and external (digital environment, dialogue with human/agent)). 
- Memory includes procedural, semantic and episodic.
- Perception includes textual (UML, execution result, text/code), visual and auditory.
- Includes good overview of different reasoning techniques for the CoT-action.


---


#### 12th of August 2024

[Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale](https://arxiv.org/abs/2409.08264)

- Navi: introduces a multi modal agent for Windows OS.
- Processes screen information called SoM (Set of Marks) with multiple alternative methods : UIA (User Interface Automation) tree, parses DOM tree, uses propietary OCR, icon/image detection and OmniParser-model.
- Agent prompt includes: task instruction, description of action space, history of actions, clipboard content and thought-variable memory. The prompt includes as well previus/current step screenshot with SoMs.
- Introduced WindowsAgentArena-benchmark.
- Includes the agent prompt.


---

#### 11th of September 2024

[Agent Workflow Memory](https://arxiv.org/abs/2409.07429)

- Agent Workflow Memory (AWM): LLM-agent retrieves and reuses reusable routines, which it extracts and generalises from past examples.
- Consists of LLM, memory and environment state (action-observation).
- Memory consists of: workflow description, workflow steps (environment state description, deduction process and action sequence). The memory-unit is described as text-based "system"-prompt. 
- Adds increasingly difficult workflows from previously acquired workflows and new experiences.
- Uses previously learned skills in new settings. Eliminates workflow steps, not required.

---

#### 10th of September 2024

[Think-on-Process: Dynamic Process Generation for Collaborative Development of Multi-Agent System](https://arxiv.org/abs/2409.06568)

- ToP (Think-on-Process): Multi-agent LLM-framework, which generates SW development processes using experiential knowledge.
- Each chat includes role assignment, memory stream and self-reflection.
- ToP-framework includes: instance generating, llm enhancing, instance filtering and software developing.
- Refers to concept of "Chat-chain", where multiple LLM-agents (CEO, CTO, CPO, Tester, Coder and Designer) operate.
- Converts processes to process textual descriptions: process-to-text and finally to process textual description.

---

#### 9th of September 2024

[SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning](https://arxiv.org/abs/2409.05556)

- SciAgents: Multi-agent graph-reasoning LLM-framework with retrieval for scientific discovery. 


#### 8th of September 2024

[Self-Reflection in LLM Agents: Effects on Problem-Solving Performance](https://arxiv.org/abs/2405.06682)

- Self-Reflection-Agents: Finds, that self-reflection improves performance of LLM agents in 6 different LLM tested.
- Self-Reflections, which contain more information (instructions, explanations, and solutions) perform better, than self-reflections with less data. 
- Retry-agent improves significantly performance, which indicates knowledge of a mistake, improves performance of the LLM.


---

#### 5th of September 2024


[Game On: Towards Language Models as RL Experimenters](https://arxiv.org/abs/2409.03402)

- Introduces RL experiment workflow using VLM (not fine-tuned) to perform tasks assigned typically to human experimenter. 
- The system monitors/analyses experiment progress, suggests new tasks, decomposes tasks and retrieves skills to execute. Does not automate
- Enables embodied autonomous agent to acquire zero-shot new skills. 

---



[From MOOC to MAIC: Reshaping Online Teaching and Learning through LLM-driven Agents](https://arxiv.org/abs/2409.03512)

- MAIC (Massively AI-empowered Course): Introduces multi LLM-agent system for scalable (like Massive Open Online Courses), but still adaptive (to personal needs / aptitudes) online education. Includes few comments from students, which highlight the limitss of its current approach.
- Includes LLM-agents acting both teachers, students, assistant, manager analyser and other agents. Teacher agents adjust style based on communication with the student. Human-student can select style of AI-classmates with the student.
- Classroom environment incldues current slide, dialogue history, class roles / course management. Course preparation includes read / plan stage, where slide content extraction, structure extraction, function generation and agent generation takes place.

---

[xLAM: A Family of Large Action Models to Empower AI Agent Systems](https://arxiv.org/abs/2409.03215)

- xLAM: Series (from 1B dense to  8x22B MoE) of Large Action Models (LAMs) for AI agent tasks. Achieves high performance in function calling.
- Fine-tunes basically from a LLM (DeekSeeker/Mistral models) a LAM, which is able to perform highly accurate function calling.


---

#### 4th of September 2024

[Cog-GA: A Large Language Models-based Generative Agent for Vision-Language Navigation in Continuous Environments](https://arxiv.org/abs/2409.02522)

- Cog-GA (Cognitive-Generative Agent)-agent: Introduces Visual-Language Navigation (VLN)-agent in continuous environments with cognitive maps (spatial, temporal and semantic information) and reflection.
- Includes instruction processor, high-level planner, waypoint predictor, memory stream (reflection memory/cognitive map), reflection generator and low-level actuator. Instructions are provided as text, panorama input image. Target waypoints are stored in the cognitive maps-memory.
- Cognitive maps include spatial memories about scene descriptions and landmarks in time step. 
- Limits search space by employing dual-channel waypoint using information about the landmark objects (what) and spatial characteristics (where).

---

[Configurable Foundation Models: Building LLMs from a Modular Perspective](https://arxiv.org/abs/2409.02877)

- Reviews modularity of LLMs. The idea is to instead of re-training from scratch a LLM, to add new knowledge as modules (called emergent bricks pretrained and customised bricks postrained).
- Identifies the following brick-operations: retrieval / routing, merging, updating and growing.


---

[Large Language Model-Based Agents for Software Engineering: A Survey](https://arxiv.org/abs/2409.02977)

- Survey about SW engineering LLM-agents.


---

[MoA is All You Need: Building LLM Research Team using Mixture of Agents](https://arxiv.org/abs/2409.07487)

- MoA (Mixture-of-Agents)-framework (name was already used before) is a framework with planner, aggregator and varios LLM-agentseach with their own RAG, grouped together.


---


#### 3rd of September 2024

[Empirical evidence of Large Language Model's influence on human spoken communication](https://arxiv.org/abs/2409.01754)

- Empirical evidence, that humans imitate LLMs.
- Finds, that LLMs reduce linguistic diversity, but it appears an interesting topic to discover, if LLMs only decrease diversity or impact other ways / the ways content creation automation impacts overall to society.


---

[AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction](https://arxiv.org/abs/2409.01854)

- AgentRe: Relation Extraction (RE) agent includes three components: retrieval (static knowledge to help store/retrieve information), memory(dynamic knowledge: shallow memory for extraction results, deep memory for historical action summaries/reflections) and extraction modules (ReAct-based, pulls information based on retrieval and memory).
- Avoids extracting for incomplete entities, such as phrases referring in general to Museums without being precise on the exact name of the museum.

---

[Focus Agent: LLM-Powered Virtual Focus Group](https://arxiv.org/abs/2409.01907)

- Focus Agent: Simulates moderation of focus groups with human participants and alignment of focus agent opinions with this group.
- Simulates planning, moderation, questions, discussion and reflection with LLM-agents.

---

#### 2nd of September 2024

[The Compressor-Retriever Architecture for Language Model OS](https://arxiv.org/abs/2409.01495)

- Compressor-Retriever-architectore: Introduces concept of stateful LLM OS by using only base model forward function to compress and retrieve context.
- Reviews concept of LLM acting as a CPU and its context window acting as RAM.
- Identifies life-long context as infite, which is core issue with actual session-based interactions.
- Compressor builds hierarchical db to save previously chunked context. The retriever searches relevant context.


---


[An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference](https://arxiv.org/abs/2403.06764)

- FastV: versatile plug-and-play method designed to optimize computational efficiency by learning adaptive attention patterns and pruning visual tokens.
- Inefficient attention in LVLMs, visual tokens inefficiency in deep layers, adaptive attention, visual token pruning, computational cost reduction, performance maintained, customizable, Pareto-efficient.
- FastV has practical value for LVLM deployment in edge devices and commercial models.


---



#### 1st of September 2024

[Self-evolving Agents with reflective and memory-augmented abilities](https://arxiv.org/abs/2409.00872)

- SAGE: Introduces self-evolving LLM-agent consisting of user/assistant/checker-agents with iterative feedback, reflection and memory optimization (Ebbinghaus-forgetting curve). 
- Self-evolution includes adaptive adjust strategies, optimizing information storage and transmission and reduction of cognitive context.
- Mimics human brain / memory by creating MemorySyntax, which combines Ebbinghaus forgetting curve and linguistic knowledge.  


---

[LanguaShrink: Reducing Token Overhead with Psycholinguistics](https://arxiv.org/abs/2409.00855)

- LannguageShrink: Reduces prompt length (tokens to process) by optimising the prompt by applying psycholinguistic principles and the Ebbinghaus memory curve.
- For example removes words like "usually" from the prompt, which add complexity, ambiguity, irrelevance etc.

---

#### 30th of August 2024

[Tool-Assisted Agent on SQL Inspection and Refinement in Real-World Scenarios](https://arxiv.org/abs/2408.16991)

- Tool-SQL: LLM-agent for SQL code inspection and fixing using retrieval and refinement. 


---

#### 29th of August 2024

[Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://arxiv.org/abs/2408.16293)

- Learns to automatically retry after detecting error (Retry upon regret) in the LLM generation, which does not require additional self-verification prompting. 
- The model seeks to produce correct solutions, even when up to half of the solution steps include errors and only corrects itself rare cases, when making a mistake. 
- Indicates, that the skill of error correction is significantly different from the pure error-free reasoning, which requires weights update beyond PEFT.
 reasoning accuracy, masking errors is unnecessary, and models still output shortest solutions.
- Indicates, that LLMs often know at least in certain domains of having made mistakes and can be seen as simple linear classifier on top of its hidden states. 
- This work provides insights into how to effectively train language models to correct errors during reasoning tasks.


---

[Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling](https://arxiv.org/abs/2408.16737)

- Suggests, that LLMs fine-tuned with synthetic data from weaker, yet cheaper LLM is more compute optimal, than using stronger, yet more expensive LLM.
- Samples data from Gemini Pro 1.5 (more expensive, stronger) compared to Gemini Flash 1.5. by using pricing per token as a proxy.


---

[CogVLM2: Visual Language Models for Image and Video Understanding](https://arxiv.org/abs/2408.16500)

- Introduces CogVLM2-family of models: CogVLM2, CogVLM2-Video and GLM-4V.
- Relates to CogAgent-GUI agent introduced in December 2023.

---


#### 28th of August 2024

[A Survey on Evaluation of Multimodal Large Language Models](https://arxiv.org/abs/2408.15769)

- The Survey reviews Multi Modal Language Models (MLLMs).


---

[WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration](https://arxiv.org/abs/2408.15978)

- WebPilot: Introduces Multi-Agent System with Planner(generate and refine plan)/Controller(judge sub-task terminatation, asses sub-task completion, generate strategic reflection)/Extractor(extract information)/Explorer(generate action, analyse observation, generate tactical reflection)/Apprasier(asses state)/Verifier(format action, deduplicate action) LLM-agents.
- Uses  Global Optimization (decomposing tasks/refining high-level plans with reflective analysis) and Local Optimization (executes sub-tasks with customized MCTS/refining decisions iteratively through with each observation).
- Tasks include navigating forums/upvoting posts/extracting contributor emails.


---

[AutoGen Studio: A No-Code Developer Tool for Building and Debugging Multi-Agent Systems](https://arxiv.org/abs/2408.15247)

- AutoGen Studio: Build on top of AutoGen, the AutoGen Studio includes drag & drop web-UI to customize/attach model/skills/tools/memory/agents involved.
- The workflow is saved as declarative json-structure. Users can export this json and share it to other users. Apart includes built-in DB Manager, Workflow Manager and Profiler-classes.
- Backend includes Python API, web API and CLI. 


---

[Interactive Agents: Simulating Counselor-Client Psychological Counseling via Role-Playing LLM-to-LLM Interactions](https://arxiv.org/abs/2408.15787)

- Investigates using LLM-agents for Psychological Counseling dialogue (counselor/client) based on client profiles (mental health issue description/detailed description of the disorder/symptom/problem/chief complaint) and counselor simulation is based on exploration, insight, and action.


---

[BattleAgentBench: A Benchmark for Evaluating Cooperation and Competition Capabilities of Language Models in Multi-Agent Systems](https://arxiv.org/abs/2408.15971)

- Introduces BattleAgentBench-benchmark, which reviews rule understanding, spatial perception, competition, static cooperation and dynamic cooperation.

---

[Atari-GPT: Investigating the Capabilities of Multimodal Large Language Models as Low-Level Policies for Atari Games](https://arxiv.org/abs/2408.15950)

- Atari-GPT: Applies Multi Modal Language Model as low-level policy (controller). 


---


[FlowAct: A Proactive Multimodal Human-robot Interaction System with Continuous Flow of Perception and Modular Action Sub-systems](https://arxiv.org/abs/2408.15864)

- FlowAct: Introduces human-robot interaction system, which continuously perceives and acts. Uses two controllers: Environment State Tracking (EST) and Action Planner. 


---

[Retrieval-Augmented Instruction Tuning for Automated Process Engineering Calculations : A Tool-Chaining Problem-Solving Framework with Attributable Reflection](https://arxiv.org/abs/2408.15866)

- RAIT (Retrieval Augmented Instruction Fine-tuning): Introduces RAIT fine-tuning approach in chemical / process engineering, which combines small language models (SMLs) with Retrieval Augmented Code Generation (RACG).

---

[Towards Fully Autonomous Research Powered by LLMs: Case Study on Simulations](https://arxiv.org/abs/2408.15512)

- Reviews feasibility of Autonomous Simulation Agent (ASA) to automate E2E research process using LLMs and API automation (AutoProg).


---

[LogicGame: Benchmarking Rule-Based Reasoning Abilities of Large Language Models](https://arxiv.org/abs/2408.15778)

- LogicGame: Benchmarks rule-based reasoning, execution and planning of LLMs.


---

[Persuasion Games using Large Language Models](https://arxiv.org/abs/2408.15879)

- Introduces persuasion framework with LLM-agents, but the paper is not clearly indicating conclusions about persuasion with LLMs with doubts as well on exact roles/prompts. 


---

[EPO: Hierarchical LLM Agents with Environment Preference Optimization](https://arxiv.org/abs/2408.16090)

- EPO (Environment Preference Optimization): Generates preference signals from environmental feedback for long-horizon decision making with LLM-agents.
- LLM predicts sub-goals and respective low-level actions.
- Interaction module generates two types of sub-goals: navigation and interaction.


---

#### 27th of August 2024

#### 27th of August 2024

[Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240)

- GenRM-verifier (Generative Reward Models): proposes training verifiers with next-token prediction objective.
- Combines verification and solution generation, whichh improves verification-process.
- GenRM outperforms classifier-based discriminatary (assigns numerical score to answer, which is used to classify as correct/incorrect answer) verifiers and LLM-as-a-judge (tends to underperform trained LLM-based verifiers).
- Integrates with fine-tuning, CoT and is able to use inference-time compute in form of majority vote to improve verification.
- Enables inference-time compute for CoT Verifiers (GenRM-CoT). Uses [reference-guided grading](https://arxiv.org/abs/2306.05685) to assist "Let's verify step by step"-verification on test-time problems lacking reference solution.
- See [slides here](https://drive.google.com/file/d/1komQ7s9kPPvDx_8AxTh9A6tlfJA0j6dR/view).

--- 



[AgentMonitor: A Plug-and-Play Framework for Predictive and Secure Multi-Agent Systems](https://arxiv.org/abs/2408.14972)

- AgentMonitor: Captures multi agent (MAS) inputs and outputs to predict task performance and correcting security risks in real-time.
- Includes 5 different MAS configurations.

---

[HPT++: Hierarchically Prompting Vision-Language Models with Multi-Granularity Knowledge Generation and Improved Structure Modeling](https://arxiv.org/abs/2408.14812)

- Introduces Hierarchical Prompt Tuning (HPT) and HPT++. Adapts VLM by creating a graph from each description with hierachical relationship guided attention module.


---

[TourSynbio: A Multi-Modal Large Model and Agent Framework to Bridge Text and Protein Sequences for Protein Engineering](https://arxiv.org/abs/2408.15299)

- TourSnmbio-Agent: Performs protein engineering tasks using TourSynbio-7B model (fine-tuned on text and protein sequences).
- Includes intent classification steps, where is defined in case the user intent is generic question or agent-specific task. 
- Keywords are used in agent selection.


---

#### 26th of August 2024

[Foundation Models for Music: A Survey](https://arxiv.org/abs/2408.14340)

- Reviews research available on Foundational models for Music: representations of music, applications, foundational model techniques, datasets/evals and ethics. 


---

[AgentMove: Predicting Human Mobility Anywhere Using Large Language Model based Agentic Framework](https://arxiv.org/abs/2408.13986)

- AgentMove: Mobility prediction LLM agent.
- Includes spatial-temporal memory.


---

[SWE-bench-java: A GitHub Issue Resolving Benchmark for Java](https://arxiv.org/abs/2408.14354)

- Benchmark to evaluate LLM-agent based coding for Java programming language (SWE-bench for Java).


---

#### 23th of August 2024

[LIMP: Large Language Model Enhanced Intent-aware Mobility Prediction](https://arxiv.org/abs/2408.12832)

- LIMP (LLMs for Intent-aware Mobility Prediction): Fine-tunes LLama 3-8B-Instruct model with Analyze-Abstract-Infer (A2I)-agentic workflow for mobility intent reasoning.


---

[Intelligent OPC Engineer Assistant for Semiconductor Manufacturing](https://arxiv.org/abs/2408.12775)

- RL / multimodal LLM-agents solve Optical Proximity Correction (OPC)-problems in semiconductor manufacturing using RL-based recipe search, which typically require years of OPC engineering experience.


---


#### 22th of August 2024

[MEDCO: Medical Education Copilots Based on A Multi-Agent Framework](https://arxiv.org/abs/2408.12496)

- MEDCO (Medical EDucation COpilots): Includes patient, student, expert doctor and radiologist multimodal (X-rays/CT scans/MRIs/ultrasounds) LLM-agents. Student agents are trained/taught with feedback provided and then stored in student memory module to improve future diagnosis.


---

[Graph Retrieval Augmented Trustworthiness Reasoning](https://arxiv.org/abs/2408.12333)

- GRATR (Graph Retrieval Augmented Reasoning): Improves trustworthiness reasoning of the LLM agent using Evidence base.
- Evidence base is updated based on observation analysis and observation assessment.

---

[MDD-5k: A New Diagnostic Conversation Dataset for Mental Disorders Synthesized via Neuro-Symbolic LLM Agents](https://arxiv.org/abs/2408.12142)

- Neuro-symbolic multi agent framework, which includes doctor, patient and tool LLM-agent interaction and dynamic (patient specific information) diagnosis tree. Introduces mental disorders diagnosis dataset MDD-5k.
- Doctor agent includes persona, diagnosis result, dialogue generation. Patient agent includes patient information, patient experience and knowledge graph.
- Establishes deeper engagement with patient to help generate diagnosis by generating the dynamic diagnosis tree. 

---

[Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards](https://arxiv.org/abs/2408.12112)

- Introduces customizable Social Choice Language Model: Uses an external adjudicator to manage tradeoffs via a user-selected social welfare function. Uses LLM to design reward functions in Restless Multi-Armed Bandits-allocation problems.
- Suggests, that prompt engineering alone 


--

[SocialQuotes: Learning Contextual Roles of Social Media Quotes on the Web](https://arxiv.org/abs/2407.16007)

- Introduces SocialQuotes-dataset to classify social media / web context into roles (influencer, expert, marketer, commenter, etc.)


---

---

[Can LLMs Understand Social Norms in Autonomous Driving Games?](https://arxiv.org/abs/2408.12680)

- LLM-agent autonomously drives in multi-agent driving game with social norms. Agents make self-driven decisions without attempting to cooperate.


---

#### 21st of August 2024

[Story3D-Agent: Exploring 3D Storytelling Visualization with Large Language Models](https://arxiv.org/abs/2408.11801)

- Story3D-Agent: LLM-agent used in 3D storytelling visualization with consistent contextually and narrative.


---

[Leveraging Chemistry Foundation Models to Facilitate Structure Focused Retrieval Augmented Generation in Multi-Agent Workflows for Catalyst and Materials Design](https://arxiv.org/abs/2408.11793)

- Improves chemistry information retrieval/catalyst and materials design usage of Chemical Foundational model (such as MolFormer-XL) by combining it with RAG.


---

[LLM4VV: Exploring LLM-as-a-Judge for Validation and Verification Testsuites](https://arxiv.org/abs/2408.11729)

- Agent-based prompting and validation pipeline increase quality of the LLM as a Judge for compiler tests.


---

[DreamFactory: Pioneering Multi-Scene Long Video Generation with a Multi-Agent Framework](https://arxiv.org/abs/2408.11788)

- DreamFactory: video generation-framework, which generates long/complex and stylistically coherent videos using multi-agent video production agent team.
- Includes requirement analysis/planning/framework preparation/script generation/scenes design/shots design/key-frames generation and video generation. 
- Lacks still creativity (artistic/devising plots) due to reliance on prompts, seems as individual videos stitched together based on synthetic audio clip and need for significant computational resources.


---

[Leveraging Fine-Tuned Retrieval-Augmented Generation with Long-Context Support: For 3GPP Standards](https://arxiv.org/abs/2408.11775)

- Implements fine-tuned Phi-2 with RAG (semantic chunking/extended context support) in telecommunications. 


---

[Cause-Aware Empathetic Response Generation via Chain-of-Thought Fine-Tuning](https://arxiv.org/abs/2408.11599)

- CFEG (Cause-aware Fine-tuning Empathetic Generation)-method: Uses emotion cause reasoning and fine-tuned LLM with CoT. Demonstrates superior empathetic dialogue responses.


---

#### 20th of August 2024


[FLAME: Learning to Navigate with Multimodal LLM in Urban Environments](https://arxiv.org/abs/2408.11051)

- FLAME (FLAMingo Architected Embodied Agent): a multimodal language-vision agent for navigational tasks by using three-step tuning: single perception tuning/multiple perception tuning/end-to-end training on VLN datasets.


---

[Athena: Safe Autonomous Agents with Verbal Contrastive Learning](https://arxiv.org/abs/2408.11021)

- Athena: Improves aligned with verbal contrastive learning, which guides LLM-agent behaviour with past safe/unsafe trajectories as in-context contrastive examples and critiquing mechanism. Contains LLM-agents: Actor/Critic/Emulator interacting to complete given task.
- Introduces safety evalution benchmark for LLM-agents with 80 toolkits in 8 categories.


---

[Strategist: Learning Strategic Skills by LLMs via Bi-Level Tree Search](https://arxiv.org/abs/2408.10635)

- Strategist: LLM-agent learns new skills through self-improvement based on MCTS and LLM-based reflection. Generates new ideas based on performance in simulated self-play by analysing good ideas.


---

[MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding](https://arxiv.org/abs/2408.11049)

- MagicDec: Speculative Decoding speeds throughput mid/long-context serving with sparse KV cache.

---

#### 19th of August 2024

[MegaAgent: A Practical Framework for Autonomous Cooperation in Large-Scale LLM Agent Systems](https://arxiv.org/abs/2408.09955)

- MegaAgent: Autonomous co-operation between dynamically generated LLM agents for specific task requirements. .
- Automatically generates sub-tasks (delegated to to sub-task admin, which coordinates the sub-task to group of agents), hierarchically plans systematically (boss agent) and monitors concurrent agent activities. OS agent coordinates, that agents communicate in proper format and progress with the task.
- The Storage module includes: log, memory db, task monitor, interactive python exec/Python, Files and Checklist.
- MegaAgent claims to pose high scalability/parallelism (due to agents communication cost grows logarithmically, not linearly), high effectiveness (manages 590 agents quicker than CAMEL-framework managed 2 agents. Summarizes previous conversations to store them in vector db) and high autonomy.


---

[GoNoGo: An Efficient LLM-based Multi-Agent System for Streamlining Automotive Software Release Decision-Making](https://arxiv.org/abs/2408.09785)

- GoNoGo: LLM-agent system, which includes Planner- and Actor-agents to process high-level queries for decision support in 120 seconds. Planner interprets user queries/plans analysis strategies. Actor generates code, resolves errors with memory/plugins/coder LLM with self-reflection.

---

#### 18th of August 2024


[Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval](https://arxiv.org/abs/2408.01875)

- Re-Invoice: 
- LLM (Query generator) generates distinct queries from tools document index. Synthetic query copiess are stored with tool name, description and query. LLM (Intent extractor) retrieves most similar tools for new user queries based on multi-view ranking algorithm.
- The multi view-ranking defines for each intent, the most similar tools. For each intent, it picks the most relevant tool, starting with the intent with highest individual tool similarity. 
- Includes an intent extractor prompt, which works just by adding it as a system instruction.

---



[HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model](https://arxiv.org/abs/2408.09559)

- HiAgent: LLM-based agent, which uses subgoals to define working memory (intrial memory), instead of retrieving entire crosstrial memory (between experiments).
- The LLM-agent replaces previous subgoals with the relevant summarized observations (action-observation pairs) for the current task.


---

#### 16th of August 2024


[EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics](https://arxiv.org/abs/2408.08782)

- EmoDynamiX: an LLM agent predicting optimal socio-emotional strategy (strategy embedding) and emotion state (emotion embedding) in a dialogue.
- Uses Heterogeneous Graph (HG) to model the dialogue interaction: node types reflect past strategies/emotional states/predicted strategy of the agent and edge types reflect dialogue dependencies between turns and speaker role-awareness. 


---


#### 15th of August 2024


[Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435)

- ADAS (Automated Design of Agentic Systems): the Meta agents discovers new agents with superior performance compared to hand-designed agents. Suggests a research direction for higher-order ADAS, where ADAS is used to improve the meta agent itself in the ADAS.
- The system consists of Meta Agent, which generates new agents and corrects them until error free. The new agent is tested and then added to Agent library. For example specific agents consists of specific blocks such as COT/Verifier/Sub-problem division/etc., which are used in specific order in the system flow.
- Meta Agent Search-algorithm generates automatically new agentic system designs and system blocks.
- The Meta Agent Search-algorithm samples new agents optimizing performance in the Search space (prompts/control flows) evaluated with the Evaluation Function (cost/latency/safety). 
- Includes codes of few of the discovered agents.


---

#### 13th of August 2024

[Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/abs/2408.07199)

- Agent Q: Introduces real world website agent iteratively fine-tuned with DPO based MCTS with self-critique and AI feedback. Trajectory collection includes reward in each node of the tree. 
- Calculates a weighted score of the MCTS average Q-value. This score is generated by a feedback LLM to construct contrastive pairs for the DPO. The policy is optimised and iteratively improved.
- LLM is used to sample reasoning/website actions to explore.
- Achieves high performance in real world environmments and beats an average human-level performance.


---

---

#### 12th of August 2024

[The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)

- AI Scientist: claims fully automatic scientific discovery by generating novel research ideas, writing code, executing experiments, visualizing results, drscribing findings to research paper and simulating evaluation process.


---

#### 9th of August 2024

[AmbigDocs: Reasoning across Documents on Different Entities under the Same Name](https://arxiv.org/abs/2404.12447)

- AmbigDocs: is a new benchmark for evaluating language models' ability to distinguish between different entities with the same name across multiple documents.
- It leverages Wikipedia's disambiguation pages, generates questions with ambiguous names, and provides corresponding sets of answers, and includes an ontology categorizing incomplete answers and automatic evaluation metrics.
- This work lays the foundation for future research on reasoning across multiple documents with ambiguous entities.


---

[Enhancing the Code Debugging Ability of LLMs via Communicative Agent Based Data Refinement](https://arxiv.org/abs/2408.05006)

- MASTER (CoMunicative Agent BaSed DaTa REfinement FRamework): code repair with LLM. Consists of Code Quizzer (code debug expert creates questions of the error), Code Learner (answers the generated questions) and Code Teacher (reviews and corrects incorrect answers) agents.
- Includes DEBUGEVAL-benchmark: bug localization, bug identification, code review and code repair.


---

#### 8th of August 2024

[Can LLMs Beat Humans in Debating? A Dynamic Multi-agent Framework for Competitive Debate](https://arxiv.org/abs/2408.04472)

- Agent4Debate: collaborative and dynamic multi-agent (searcher/analyzer/writer/reviewer) LLM for competitive debate.
- Includes Chinese Debate Arena-benchmark with
- Framework begins with context/motion/position/stage. Searcher gathers information, analyzer reviews arguments, writer generates arguments/debates and reviewer provides feedback on debate.


---

[RiskAwareBench: Towards Evaluating Physical Risk Awareness for High-level Planning of LLM-based Embodied Agents](https://arxiv.org/abs/2408.04449)

- RiskAwareBench: reviews physical risk awareness of embodied LLM agents. 
- Includes modules: safety tip generation/risky scene generation/plan generation & evaluation/ isk assesment.


---

#### 7th of August 2024

[Perceive, Reflect, and Plan: Designing LLM Agent for Goal-Directed City Navigation without Instructions](https://arxiv.org/abs/2408.04168)

- PReP: city-navigation to goal using visual perception and memory (working, episodic & semantic) without instructions.
- Semantic memory summarizer memories from multiple steps, to perform high-level navigtion.


---

[Forecasting Live Chat Intent from Browsing History](https://arxiv.org/abs/2408.04668)

- LLM-based user intent prediction (to predict why user needs live-chat agen support) from high-level categories classified from browsing history and then in second step predicts fine-grained user intent with the high-level intent class and browsing history.



---

[CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases](https://arxiv.org/abs/2408.03910)

- LLM uses cod RAG. Builds code graph db from code repository. Nodes represent symbols, edges represent relationships between symbols and schema defines how code graphs are stored in the code db.


---

#### 6th of August 2024

[Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)

- Reviews scaling up inference compute (test-time) in order to built self-improving agents. Quantifies the amount of improvement, when increasing inference.
- Test-time compute outperforms 14x larger models.
- Compute optiml scaling strategy can improve efficiency of test-time compute by factor of up to 4x.


---


#### 5th of August 2024

[ReDel: A Toolkit for LLM-Powered Recursive Multi-Agent Systems](https://arxiv.org/abs/2408.02248)

- ReDel (Recursive Delegation): Recursive multi-agent framework, where LLM decides when to delegate/how to delegate (delegation graph).
- Includes custom tool-use, delegation schema, event-based logging and interactive replay (web UI).
- Icludes open-source Python package.
- ReDel delegation schemes include DelegateOne (wait parent-agent until child-agent completion) and DelegateWait (provide separate function for parent agent to retrieve child agent response).
- Event-driven logging includes built-in events ans custom events.


---

[SpecRover: Code Intent Extraction via LLMs](https://arxiv.org/abs/2408.02232)

- SpecRover/AutoCodeRover-v2: autonomous github issue fixing by understanding developer intent from Github repo structure / developer behaviour.
- Claims Github issues can be solved as little as $0.65 /issue.


---

[LLM Agents Improve Semantic Code Search](https://arxiv.org/abs/2408.11058)

- RAG-agent (ensemble architecture), which adds relevant contextual information to the user query from the Github repository. 
- Uses RepoRift-platform, which improves code search by: narrows context search to single repository, uses agentic interaction and returns easy-to-understand results with low latency.


---
#### 3rd of August 2024

[The Drama Machine: Simulating Character Development with LLM Agents](https://arxiv.org/abs/2408.01725)

- Drama Machine: Reviews Automated Identity-generation with LLMs.  Uses multiple LLMs to simulate dynamic/complex AI characters in domain of drama scenes: interview/detective.
- Roles include Ego, SuperEgo, Autobiography, Director and Critic.

--- 

#### 2nd of August 2024

[Coalitions of Large Language Models Increase the Robustness of AI Agents](https://arxiv.org/abs/2408.01380)

- Coalition of LLM models outperform single model and fine-tuned LLMs.
- Specific LLMs fit for particular tasks and cheaper interference.


---

#### 1st of August 2024

[OmniParser for Pure Vision Based GUI Agent](https://arxiv.org/abs/2408.00203)

- OmniParser: VLM agent parsing GUI screenshots into structured data. Attempts to ground actions grounded on GUI regions.
- Includes detection model to captura interactable GUI regions. Caption model retrieves functional semantics of these detected elements. OCR generates structured reprentation of the GUI.
- Improves action prediction accuracy. Includes icon-detection dataset.
- Reviews comphrehensively screen coordinate detection problem of VLMs.
- Error cases include: repeated/misinterpreted icons, repeated texts and inaccurate bounding boxes. 

---

[AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation](https://arxiv.org/abs/2408.00764)

- AgentGen: Generates diverse LLM agent environments and planning tasks. LLM fine-tuned with this data improves significantly planning capabilities.
- Uses inspirational corpus to generate environment context (actions/restrictions/etc). Generates tasks, which include "difficulty diversification: easy/medium/hard with bidirectional evolution (Bi-Evol) to smoothly acquire new planning skills.


---

#### 31st of July 2024

[Tulip Agent -- Enabling LLM-Based Agents to Solve Tasks Using Large Tool Libraries](https://arxiv.org/abs/2407.21778)

- Tulip Agent and AutoTulipAgent: LLM-agent has priviledges to create, update, delete and edit tool library. 
- Self-Recursively extendible tool library. 
- AutoTulipAgent includes 5 generic tools: 2 to decompose tasks/search tools, includes apart capability to create/delete/update tools. 


---

#### 29th of July 2024

[Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process](https://arxiv.org/abs/2407.20311)

- iGSM framework: is used to generate diverse grade-school math problems for training and testing language models.
- The framework includes a hierarchical categorization, structure graph, dependency graph, and solution construction using Chain-of-Thought (CoT) approach, and it uses GPT2-like language model with rotary embedding.
- This framework enables a principled study of language models' mathematical reasoning skills, going beyond empirical benchmark pushing.


---


#### 28th of July 2024

[Solving Robotics Problems in Zero-Shot with Vision-Language Models](https://arxiv.org/abs/2407.19094)

- Wonderful Team: uses off-shelf VLM model for high-level planning, low-level location extraction and action execution.


---

#### 26th of July 2024

[AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://arxiv.org/abs/2407.18901)

- AppWorld-benchmark: simulates LLM-agents using App World Engine-execution environment (mimicking 9 real-world apps/simulates 457 APIs/100 ficticious and related users) by measuring 750 complex tasks (records database start state and end state to review correct/incorrect actions to Base DB), which require iterative/interactive code generation without real-world consequences. 
- Generates task scenarios, which are used by the task generator (setup/validation/evaluation). 
- Each task is checked to be: well-defined/includes distractors/has real distractors/contrasts from exissting other tasks.
- Includes Supervisor (provides passwords/credit cards/etc about the user), (API parameters/descriptions) and Execution Shell to run code.


---
#### 25th of July 2024

[The Platonic Representation Hypothesis](https://www.arxiv.org/abs/2405.07987)

- The Platonic Representation Hypothesis: Neural networks are converging to a shared statistical model of reality in their representation spaces.
- Convergence across data modalities; representation alignment over time; driven by data and task diversity; scaling model size.
- Understanding convergence is crucial for future AI development and capabilities.


---

[PersonaGym: Evaluating Persona Agents and LLMs](https://arxiv.org/abs/2407.18416)

- Introduces PersnaGym-benchmark to evaluate persona LLM-agents.
- Sets an automatic PersonaScore-metric to evaluate five different capabilities.
- Finds SOTA level LLMs to offer highly varying level of capabilities as persona-agents.
- Increasing model size is not guarantee of better persona agent performance with varying level of persona agent performance detected.

---

[Recursive Introspection: Teaching Language Model Agents How to Self-Improve](https://arxiv.org/abs/2407.18219)

- RISE (Recursive IntroSpEction): iteratively sel-improve LLM responses through fine-tuning with RL.
- LLM loss is lower, when using multi-turn data compared instead of only the final answer. Works only for reasoning, not knowledge tasks.
- Indicates strongly, that Full online RL is feasible with RISE and using iterative self-training procedure (such as STaR), because RISE improves the LLM with 5-turns with/without oracle model. 
- Demonstrates, that LLMs can self-improve its own mistakes to beyond level of propietary models, when trained with RISE. The self-improvement continues up to 6 iterations, demonstrating lower loss. 
- RISE starts with turn 1, where only prompt is provided. In turn 2, the prompt, the original response and its feedback is provided to generate the turn 2 response. Majority voting is used to select the final response from multiple responses generated. Alternatively, oracle model can be used to assist, when such is available.
- Why self-improvement works? RISE is compared to diffusion models, where generation is refined step-by-step. Similarly LLMs may lack "capacity" to process the request, which RISE can help to refine. See the talk on this paper [here.](https://www.youtube.com/watch?v=Qv8aTLthfhs).

---

#### 24th of July 2024


[Reinforced Prompt Personalization for Recommendation with Large Language Models](https://arxiv.org/abs/2407.17115)

- Reinforced Prompt Personalization (RPP): uses instance-based prompting with MARL.
- Instead of task-based (role-play/history/reasoning guidance/output format), Instance-based prompting personalises to these four-characteristics with MARL.


---

[AI-Gadget Kit: Integrating Swarm User Interfaces with LLM-driven Agents for Rich Tabletop Game Applications](https://arxiv.org/abs/2407.17086)

- AI-gadget Kit: multi-agent driven Swarm UI (SUI) tabletop gaming system, which consist of meta-motion, interactive behaviour, interactive relationship and application.  


---

[3D Question Answering for City Scene Understanding](https://arxiv.org/abs/2407.17398)
- Sg-CityU: 3D multimodal QA, which uses scene graph to provide answers related to spatial relationships about city-scenes


---

#### 23rd of July 2024

[RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent](https://arxiv.org/abs/2407.16667)

- RedAgent: Introduces concept of "Jaillbreaking strategy" (strategies used by attackers to construct jaillbreaking prompts) red teaming through multi-agent self-reflection from context feedback and skill memory.
- The approach can jaillbreak LLMs and LLM-based apps (even more vulnerable) using just few queries.
- The Red-Agent architecture includes skill memory and multiple roles (profile constructor/planner/attacker/evaluator) and short/long term memory.

---

[AMONGAGENTS: Evaluating Large Language Models in the Interactive Text-Based Social Deduction Game](https://arxiv.org/abs/2407.16521)

- AmongAgents: multi-agent LLM-framework with memory, reflection and interaction in social deduction game with ambiguous and deceptive characters.
- Includes meeting/task-phases.
- Agents pose personality-component: generated with personality prompt from pre-defined set of personalities: behaviour/decision-making, which contribute to more dynamism/realism.

---

[OpenDevin: An Open Platform for AI Software Developers as Generalist Agents](https://arxiv.org/abs/2407.16741)

- OpenDevin: LLM-based multi-agent framework, where agents interact as human-like SW agents writing code, using command line and browsing web.
- The framework includes: interaction mechanism (event stream), environment(sandbox environment for code execution),  interface(human-like), multi-agent delegation (co-operate) and evaluation framework.
- Event stream tracks history of action and observation.


---

[PyBench: Evaluating LLM Agent on various real-world coding tasks](https://arxiv.org/abs/2407.16732)

- Introduces PyBench-benchmark for real-world like coding tasks withh LLM-agents.
- Introduces high-performance PyLlama3 model for coding tasks.

---

[Artificial Agency and Large Language Models](https://arxiv.org/abs/2407.16190)


- Reviews theoretical models for agents, LLM agents and concept of artificial agency.

[LawLuo: A Chinese Law Firm Co-run by LLM Agents](https://arxiv.org/abs/2407.16252)

- LawLuo: includes LLM-based receptionist/lawyer/secrretary/boss-agents to realistic legal consultation company based on SOP (Standard Operating Principle).


---

#### 22th of July 2024

[TaskGen: A Task-Based, Memory-Infused Agentic Framework using StrictJSON](https://arxiv.org/abs/2407.15734)

- TaskGen: LLM-agent framework to solve tasks by dividing task into sub-tasks, executed by its own agent/equipped function. Manages memory/information based on need-to-know. Uses in StrictJson-format.
- Includes meta-agent, inner-agent, function-calls, sub-tasks, shared memory (sub-task completed/list of past equiped function inputs or outputs/shared variables) and passing context/shared memory to inner agent/function.
- Utilises global context adds data to default LLM prompt (carrying shared variables throughout a task/to store the current state of a dynamic environmental variable/specific instructions).

---

[Odyssey: Empowering Agents with Open-World Skills](https://arxiv.org/abs/2407.15325)

- Odyssey: interactive (plan-actor-critic) LLM-agent (fine-tuned Llama 3) with real world skill library.
- Introduces long-term planning/dynamic-immediate planning/autonomous exploration benchmark.
- Planner decomposes long-term goals into sub-goals with ultimate goals/behavioural constraints/agent states/achievements.
- Actor executes skill code using query context/similarity match/skill selection.
- Critic uses execution feedback/self-validation/self-reflection.


---

#### 19th of July 2024



#### 19th of July 2024

[System-1.x: Learning to Balance Fast and Slow Planning with Language Models](https://arxiv.org/abs/2407.14414)

- System-1.x Planner: introduces a controllable planning framework (inference time compute) capable of producing hybrid plans balancing system 1 and system 2 thinking. Includes Controller/System-1 Planner/System-2 Planner. 
- The Controller manages the x-factor, which is the degree to how much to use System-1 vs. System-2 thinking to decompose planning into sub-goals. 
- Demonstrates: controllability/flexibility/generalizability to different search algorithms. 


---

[The Vision of Autonomic Computing: Can LLMs Make It a Reality?](https://arxiv.org/abs/2407.14402)

- Explores feasibility of Autonomic Computing Vision (ACV) with multi-agent framework based on LLMs.
- LLM-based multi-agent framework achieves level 3 autonomy.
- The original ACV-framework identified 4 pillars: self-configuration, self-optimization, self-healing and self-protection.


---

#### 18th of July 2024

[Prover-Verifier Games improve legibility of LLM outputs](https://arxiv.org/abs/2407.13692)

- Prover-Verifier: Direct RL on solution correctness generates solutions difficult for humans to evaluate and obtains.
- Checkability training results prover, which maintains legibility, while taking a a legibility tax in form of losing some performance to make them more easier to check for humans. 
- Discusses the possibility of training two models: train model with CoT to maximize accuracy and another model to turn the CoT produced by the model into legible version understandable for humans.


---

#### 12th of July 2024

[PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents](https://arxiv.org/abs/2407.09394)

- PersonaRAG: Includes compoments k-docs retrieval, user interaction analysis (user profile/contextual retrieval/live session/document ranking/feedback agents) and cognitive dynamic adaption(selective/collaborative use of agents).


---

[Instruction Following with Goal-Conditioned Reinforcement Learning in Virtual Environments](https://arxiv.org/abs/2407.09287)

- IGOR (Instruction following with GOal-conditioned RL): LLM translates instructions into high-level action plan with sub-goals and RL executes them.


---

[Large Language Models as Biomedical Hypothesis Generators: A Comprehensive Evaluation](https://arxiv.org/abs/2407.08940)'

- LLMs generate novel and diverse biomedical hypthesis through multi-agent interaction.


---


#### 11th of July 2024

[GTA: A Benchmark for General Tool Agents](https://arxiv.org/abs/2407.08713)

- GTA-benchmark: evaluates general tool usage of LLM agents in real user queries with real deployed tools. for example web page screenshots.
- Evaluates perception, operation, logic and creativity tools.
- Defines "Real-World" as helping humans in real-life with being step/tool-implicit. 
- GPT-4 solves 50% of these tasks.
- Includes illustration of executable tool chains.


---

[Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligence](https://arxiv.org/abs/2407.07061)

- Internet of Agents (IoA): LLM agents lack capability to interact in dynamic environments with other agents outside its hard-coded communication pipeline.
- Limitations include: ecosystem isolation, single-device simulation and rigid communication/coordination.
- IoA acts in Internet-like environment to achieve collective intelligence and new capabilities.
- Includes architectural design of the IoA-framework.


---



[Converging Paradigms: The Synergy of Symbolic and Connectionist AI in LLM-Empowered Autonomous Agents](https://arxiv.org/abs/2407.08516)

- LAAs (LLM-empowered Autonomous Agents): Introduces concept of LAAs, which include three elements: external tools, LLMs (knowledge modelling) and Agentic workflow (human-like symbolic reasoning).
- LAAs are characterised by natural language dialogue, decision making, planning, task decomposition and actionining.


---

[GPT-4 is judged more human than humans in displaced and inverted Turing tests](https://arxiv.org/abs/2407.08853)

- Introduces Inverted Turing text.


---

[Beyond Instruction Following: Evaluating Rule Following of Large Language Models](https://arxiv.org/abs/2407.08440)

- RuleBench-benchmark: evaluates LLMs capability to follow rules.
- Evaluation dimensions include: executing rules, triggering rules, following formal rules, applying rules and following counterfactual rules.


---


[Large Models of What? Mistaking Engineering Achievements for Human Linguistic Agency](https://arxiv.org/abs/2407.08790)

- Argues, that LLMs cannot be linguistic agents in the actual form by lacking embodiment, participation and precariousness. 


---


[Incorporating Large Language Models into Production Systems for Enhanced Task Automation and Flexibility](https://arxiv.org/abs/2407.08550)

- Reviews integration of LLMs into Automated Production Systems.


---


#### 10th of July 2024

[WorldAPIs: The World Is Worth How Many APIs? A Thought Experiment](https://arxiv.org/abs/2407.07778)

- Discovers lower-bound of covering 0.5% of WikiHow instructions equals roughly usage of 300 APIs, which we can consider lower-bound limit for covering wide variety of WikiHow instructions in Embodied agent tasks.
- The framework iteratively produces action spaces for APIs to be used by a LLM based embodied agent. 
- This two-step process works by iteratively generating through hallucination: semi-executable agent policies with python by LLM few-shot prompting from WikiHow instructions, parse partial/full python programs into pool of APIs


---

#### 9th of July 2024

[Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models](https://arxiv.org/abs/2407.07086)

- Hypothetical Minds: Introduces "Theory-of-Mind"-module. Includes as well perception, memory and hierarchical two-level planning.


---

[Vision language models are blind](https://arxiv.org/abs/2407.06581)

- Reviews 7 visual tasks, where SOTA-level VLMs perform shockingly bad.


---

#### 5th of July 2024

[On scalable oversight with weak LLMs judging strong LLMs](https://arxiv.org/abs/2407.04622)

- Explores debate and consultancy to supervise AI.
- Finds debate outperforms consultancy in general. Better debater models modestly improve judge accuracy. 


---

[When LLMs Play the Telephone Game: Cumulative Changes and Attractors in Iterated Cultural Transmissions](https://arxiv.org/abs/2407.04503)

- Reviews toxicity/bias in LLM agent multi-step inputs/outputs, instead of individual LLM input-output. 


---

[Are Large Language Models Strategic Decision Makers? A Study of Performance and Bias in Two-Player Non-Zero-Sum Games](https://arxiv.org/abs/2407.04467)

- Reviews LLMs in strategic games. LLMs come with systematic bias: positional bias, payoff bias and behavioural bias. LLMs performance decreases, when the mentioned bias-dimensions are misaligned.  


---

#### 3rd of July 2024

[LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control](https://arxiv.org/abs/2407.03168)

- LivePortrait: generates realistic video from single portrait image with facial expressions and head poses from different angles. 
- Offers better computational efficiency and controllability over diffusion models, by using implicit-keypoint-based framework.
- Generation speed is 12.8 ms with RTX 4090.


---

[Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory](https://arxiv.org/abs/2407.03103)

- Cactus: multi-turn dialogue dataset for mental health counseling, consisting of goal-oriented/structured Cognitive Behavioral Therapy interation.
- Trains Camel-LLM using the Cactus-dataset.


---

#### 2nd of July 2024

[GRASP: A Grid-Based Benchmark for Evaluating Commonsense Spatial Reasoning](https://arxiv.org/abs/2407.01892)

- GRASP: Large scale  spatial reasoning benchmark and dataset in structured grid environment requiring planning and commonsense reasoning.

---

[MMedAgent: Learning to Use Medical Tools with Multi-modal Agent](https://arxiv.org/abs/2407.02483)

- MMedAgent: MMedAgent outperforms GPT-4o-agent in medical tasks based on LLaVA-Med-model by fine-tuning data from 6 different tools.


---

#### 1st of July 2024

[Agentless: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489)

- Agentless: Argues, that it s not required to deploy complex autonomous sw agents.
- Uses two step approach: Localization (files requiring sw fix) and Repair.
- Framework begins from codebase and an issue. It then reviews repo structure and issue to localize top n-files, localizes classes/functions, localizes edit locations. In the repair-phase, the LLM generates various patches, which are filtered and ranked to submit the patch to the issue.


---

#### 29th of June 2024

[Question Translation Training for Better Multilingual Reasoning](https://arxiv.org/abs/2401.07817)

- QAlign (Question Alignment): is a framework that fine-tunes LLMs to translate reasoning questions into English using X-English parallel question data.
- It uses targeted in-domain language alignment, enables effective utilization of English instruction data, and includes response alignment with cutting-edge English instruction data.
- This framework improves multilingual reasoning capabilities of LLMs by transferring English expertise to non-English tasks.


---


#### 28th of June 2024

[LLM Critics Help Catch LLM Bugs](https://arxiv.org/abs/2407.00215)

- Focuses on self-correction or self-critique in the domain of code bug fixing in real-world.
- Finds majority of the critique generated automatically is better than human generated.


---

[BMW Agents -- A Framework For Task Automation Through Multi-agent Collaboration](https://arxiv.org/abs/2406.20041)

- BMW Agents: Includes three main components for the LLM-based agents: Planning, Execution and Verification. 
- Retrieve a task from task queue DB and coordinator agent orchestrates the agent workflow. Includes Tools, Memory and Persona/Objectives.
- Tool refiner has access to wide variety of tools, which it limits to subset of tools available for the agent in particular task.
- Introduces: "Programmable Prompts", which generalises ReAct and PlanReAct by using iterative sequence consisting of pre-defined steps A...X.


---

[Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094)

- Persona-Hub: Diverse 1B personas web dataset using persona-driven data synthesis method. Includes only main characteristics without fine-grained details.
    

---

#### 27th of June 2024

[Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs?](https://arxiv.org/abs/2406.19354)

- Reviews model editing of LLMs.
- Identifies existence of editable beliefs in LLMs.
- Develops model editing benchmark.
- Reviews difference between LLMs acting as agents vs. agent simulators.


---

[Tools Fail: Detecting Silent Errors in Faulty Tools](https://arxiv.org/abs/2406.19228)

- Reviews LLM tool use failure recovery from "silent errors". Tool output is accurate only when: input is accurate, context is sufficient and tool makes correct predictions.
- Introduces taxanomy for categorising tool-related errors and methods to recovery from them (refine and recovery).
- Identifies challenges in tool recovery: failure detection/fault assignment/recovery planning.


---

[Simulating Classroom Education with LLM-Empowered Agents](https://arxiv.org/abs/2406.19226)

- SimClass: simulates multi-agent classroom teaching. Includes manager (observe/tutor/interact), teacher, assistant and classmate agents with the user.
- Session controller manages modules: Class State Receptor, Function executor and Manager agent. 
- Observing uses class-states (class roles, learning materials and dialogue history). Tutoring functions include next page/teaching, which are only directed by the teacher. Interaction functions are performed agent to agent. Classmate agents have different roles like note taker, deep thinker, idea creator etc.


---

[UniGen: A Unified Framework for Textual Dataset Generation Using Large Language Models](https://arxiv.org/abs/2406.18966)

- UniGen: Textual dataset generation with LLM-dataset generation approach and reviewed in benchmarking and data augmentation context.
- Demonstrates the data augmentation technique is effective and adds capabilities to the LLM, while discusses the technique limitations in Appendix A such as knowledge intensive tasks Knowledge intensive tasks could benefit instead from Out-Of-Distribution data, still unmastered by the LLM. 


---

[Capturing Minds, Not Just Words: Enhancing Role-Playing Language Models with Personality-Indicative Data](https://arxiv.org/abs/2406.18921)

- RPLM (Role Playing Language Model): Develops RPLM with personality behaviours/traits/tendencies. Introduces RolePersonality-dataset based on 14 psychology dimensions, which is gathered using role-playing expert agent interviewing with questions based on the 14 dimensions. 


---

[LayoutCopilot: An LLM-powered Multi-agent Collaborative Framework for Interactive Analog Layout Design](https://arxiv.org/abs/2406.18873)

- LayoutCopilot: LLM-based analog layout design framework.


---

[Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction](https://arxiv.org/abs/2406.19108)

- Explores emergence of self-replicating programs. Introduces "high-order entropy"-metric to measure complexity of the system studied.


---


#### 26th of June 2024

[Symbolic Learning Enables Self-Evolving Agents](https://arxiv.org/abs/2406.18532)

- Agent Symbolic Optimizers: introduces agent symbolic learning framework. Optimizes symbolic components (prompts/tools/their orchestration) of the LLM agent. Attempts to optimize agent to solve real-world task by enabling LLM-agent to learn from data and self-evolve.
- Proposes, that key to achieve AGI is to move from model-centric or engineering-centric to data-centric language agents, which learn and envolve autonomously in environments.
- Agent symbolic learning optimizes symbolic network within language agents. 


---

[MAGIS: LLM-Based Multi-Agent Framework for GitHub Issue Resolution](https://arxiv.org/abs/2403.17927)

- MAGIS: LLM-based framework to resolve Github issues using four agents: Manager, Repository Custodian, Developer and Quality Assurance Engineer.
- Reviews correlation in task success rate and task complexity/ability to locate relevant code line.
- Planning part includes locating files/code, building team, kick-off meeting. Coding part includes developer producing code and then QAE validating it.


---

[Lifelong Robot Library Learning: Bootstrapping Composable and Generalizable Skills for Embodied Control with Language Models](https://arxiv.org/abs/2406.18746)


- LRLL-agent (Lifelong Robot Library Learning): increases continuously the robot skill library by using soft memory module, self-guided exploration, skill abstractor and lifelong learning algorithm.
- The framework is inspired by wake-sleep optimization, where wake phase (interacts with environment) is followed by sleep phase (agent reflects experiences).


---

[Simulating The U.S. Senate: An LLM-Driven Agent Approach to Modeling Legislative Behavior and Bipartisanship](https://arxiv.org/abs/2406.18702)

- Reviews use of LLM to understand and improve legislative process.


---

[Mental Modeling of Reinforcement Learning Agents by Language Models](https://arxiv.org/abs/2406.18505)

- XRL (eXplainable RL): Reviews LLMs capacity to build mental models about RL agent behaviour. Finds, that LLMs lack mental modeling capabilities about RL agents.
- LLM-Xavier workflow: RL agent rolls a trajectory, which LLM-agent reasons to provide an answer. This evaluation is compared with the ground truth data.
- Offers a way to explain behaviour of black-box RL agents.


-- 

[AI-native Memory: A Pathway from LLMs Towards AGI](https://arxiv.org/abs/2406.18312)

- Claims AGI-like systems require AI-native memory, which is deep neural network parametrising different types of memories beyond language. Claims such Large Personal Model (LPM) would be unique for each person with every detail about the user for personalised generation.
- Includes useful ideas about what data the personalised memory could look include or the various levels of data granularity.


---

[Role-Play Zero-Shot Prompting with Large Language Models for Open-Domain Human-Machine Conversation](https://arxiv.org/abs/2406.18460)

- Investigates role-play zero-shot prompting in conversational agent.


---

[LLCoach: Generating Robot Soccer Plans using Multi-Role Large Language Models](https://arxiv.org/abs/2406.18285)

- LLCoach: Reviews advance planning capabilities of robots in dynamic/unstructured environments.
- The system offline components collects plans from video frames to the Coach VLM and refines them using LLM, which retrieves Acctions from vector db and synchronises into multi-agent plans. Online component retrieves and executes most similar plan to the world model status.



---

[Octo-planner: On-device Language Model for Planner-Action Agents](https://arxiv.org/abs/2406.18082)

- OctoPlanner: Separates planner/action-steps into OctoPlanner (planner) agent and Action agent (Octopus model) with function execution.
- Planner agent divides tasks into sub-tasks.
- Optimized for on-device usage through usage of fine-tuning instead of in-context learning.


---

#### 25th of June 2024

[Human-Object Interaction from Human-Level Instructions](https://arxiv.org/abs/2406.17840)

- Develops complete system to synthesize object motion, full-body motion and finger motion simultaneously. 
- Applies High-evel planner to generate target scene layout/task plan and then uses low-level motion generation with four stage appproach with: CoarseNet/GraspPose/RefineNet and FingerNet.
- Planner includes three stages: Generate spatial relationships between objects in natural language (to improve performance), calculate target layouts and generate detailed plan.


---

#### 24th of June 2024

[RES-Q: Evaluating Code-Editing Large Language Model Systems at the Repository Scale]()

- Evaluates LLMs on repository-level coding. Claude Sonnet 3.5 outperforms by 12% the GPT-4o. 

---

[RES-Q: Evaluating Code-Editing Large Language Model Systems at the Repository Scale](https://arxiv.org/abs/2406.16801)


#### 21st of June 2024

---

[GenoTEX: A Benchmark for Evaluating LLM-Based Exploration of Gene Expression Data in Alignment with Bioinformaticians](https://arxiv.org/abs/2406.15341)

- GenoAgent: LLM-based genomics data-analysis.  


---

[ESC-Eval: Evaluating Emotion Support Conversations in Large Language Models](https://arxiv.org/abs/2406.14952)

- ESC-Role: LLM-agent for Emotional Support Conversation (ESC) tasks.  Includes ESC-Eval benchmark.


---

[Autonomous Agents for Collaborative Task under Information Asymmetry](https://arxiv.org/abs/2406.14928)

- iAgents (Informative Multi-Agent Systems): multi-agent system based on human social network, where person has an agent with access to information only from its user.
- Introduces InformativeBench-benchmark to evaluate LLM task solving capability when access to only part of information (information asymmetry).
- iAgents collaborate in social network of 140 individuals and 588 relationships and communicate 30 turns.


---

[FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents](https://arxiv.org/abs/2406.14884)

- FlowBench-benchmark: reviews workflow-guided (think flowcharts) planning capability of LLMs.  


---

[Direct Multi-Turn Preference Optimization for Language Agents](https://arxiv.org/abs/2406.14868)

- DMPO-loss function to optimize RL objectives in multiturn agent tasks.


---

[Evaluating RAG-Fusion with RAGElo: an Automated Elo-based Framework](https://arxiv.org/abs/2406.14783)

- RAGElo-benchmark reviews retrieval performance as well in RAF-Fusion use (fuses top-k retrievals). 


---

[DiPEx: Dispersing Prompt Expansion for Class-Agnostic Object Detection](https://arxiv.org/abs/2406.14924)

- DiPEX (Dispersing Prompt Expansion)-approach: Uses VLM and DiPEX to improve class-agnostic object detection.


---

[Behaviour Distillation](https://arxiv.org/abs/2406.15042)

- Behaviour Distillation: compresses information for training expert policy in RL by learning synthetic data (HaDES-method) of state-action pairs without requiring the expert data.


---

[Uni-Mol2: Exploring Molecular Pretraining Model at Scale](https://arxiv.org/abs/2406.14969)

- Uni-Mol2: 1.1B parameter model for molecular representation based on f Uni-Mol+ architecture (two track transformer).


---

[From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking](https://arxiv.org/abs/2406.14859)

- Survey on multimodal / VLM / LLM jailbreaking research.





---


#### 20th of June 2024

[Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning](https://arxiv.org/abs/2406.14283)

- Q*: Improves multi-step reasoning of LLMs through heuristic search planning in MDP.
- Objective is to find most suitable reasoning with maximum utility.
- Introduces multiple general approaches (offline RL/best sequence from rollout/completion with stronger LLM) to calculate the Q-value.
- The approach works as such in various reasoning tasks.


---

[GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models](https://arxiv.org/abs/2406.14550)

- GraphReader: LLM agent converts long text into graph structure to explore by performing step-by-step analysis and by generating detailed plan.
- Achieves performance level of 128k context window LLM using 4k context window LLM by converting the long text into graph structure.
- The LLM agent records insights from the explored graph and reflects current situation to optimize answer generation.


---

[LLaSA: Large Multimodal Agent for Human Activity Analysis Through Wearable Sensors](https://arxiv.org/abs/2406.14498)

- LLaSA (Large Language and Sensor Assistan): Text query received is converted into text embedding and sensor reading into IMU embeddings (inertia measurements unit embeddings). Both inputs are passed to LLaSA model and its output to LLM to produce final answer.


---

[Artificial Leviathan: Exploring Social Evolution of LLM Agents Through the Lens of Hobbesian Social Contract Theory](https://arxiv.org/abs/2406.14373)

- Evaluates LLM-based multi-agent society. This society includes psychological drives and social relationships.
- Evaluates Hobb's Social Contract Theory.


---

[EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms](https://arxiv.org/abs/2406.14228)

- EvoAgent: reviews specialized agents extension into multi-agent system through evolutionary pipeline. 


---


[Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset designed for LLMs with Psychometrics](https://arxiv.org/abs/2406.14703)

- Introduces TRAIT-personality test to review LLM personality.   


---

[Can LLMs Learn by Teaching? A Preliminary Study](https://arxiv.org/abs/2406.14629)

- Learning by Teaching (LbT): LbT includes three methods: Observing student feedback, learning from the feedback and learning iteratively.


---


[MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate](https://arxiv.org/abs/2406.14711)

- Persuasion by adversial agent in multi-agent debate, which undermines shared interests. 



---



#### 19th of June 2024

[Prism: A Framework for Decoupling and Assessing the Capabilities of VLMs](https://arxiv.org/abs/2406.14544)

- Prism: evaluation framework separately reviews VLMs perception and planning capabilities. Uses single LLM to compare various VLMs (VLM Zoo) perception capabilities or uses multiple LLMs (LLM zoo) with single VLM to evaluate planning capabilities. 


---

[AlanaVLM: A Multimodal Embodied AI Foundation Model for Egocentric Video Understanding](https://arxiv.org/abs/2406.13807)

- AlanaVLM: SOTA-level (surpasses in spatial reasoning) 7B VLM trained with EVUD-dataset to understand embodied and ecocentric video understanding.
- Introduces Ecocentric video understanding dataset (EVUD).


---

[SpatialBot: Precise Spatial Understanding with Vision Language Models](https://arxiv.org/abs/2406.13642)

- SpatialBot: VLM trained with SpatialQA-dataset (includes VQAs with low, middle and high-level), which comprehends spatial information in thre levels (point depth/depth description, proximity/object depth and spatial relationship/counting).
- Introduces SpatialBench-benchmark to review VLMs spatial understanding.


---

[LIT: Large Language Model Driven Intention Tracking for Proactive Human-Robot Collaboration -- A Robot Sous-Chef Application](https://arxiv.org/abs/2406.13787)

- LIT (Language-driven Intention Tracking): LLM and VLM system, which tracks human actions from images using VLM to predict human intentions. Uses  graph reasoning to generate a plan steps with LLM.
- The VLM generates for each image a captioning about what is being done by the human and predicts the likelihood of this task to relate to specific step in the plan.
- Based on the predicted plan step, the system predicts the most likely next step being performed by the human.

---

#### 18th of June 2024

[Talk With Human-like Agents: Empathetic Dialogue Through Perceptible Acoustic Reception and Reaction](https://arxiv.org/abs/2406.12707)

- PerceptiveAgent: empathic multi modal agent, using acoustic information from speech for empathic responses adjusting to speaking style.
- Captures more accurately speakers real intentions (captions) and interacts (speech attributes) using adjusted tone for the context.
- Framework includes three compoments: Speech captioner (Speech encoder, Q-former and text encoder), LLM and MSMA-Synthesizer (speaker embedder, Attribute embedder and HiFiGAN vocoder).


---

[Problem-Solving in Language Model Networks](https://arxiv.org/abs/2406.12374)

- Represents each agent as a node, which create a connected multi-agent network with self-reflection.
- Finds self-reflection is useful, when surrounded by incorrect LLM-agents and less useful, when surrounded by LLM-agents providing correct answers.
- LLM agents are likely to agree for consensus, when the LLM answer is correct. The LLM answer is more likely to be incorrect, when LLMs are more divided.


---

[Ask-before-Plan: Proactive Language Agents for Real-World Planning](https://arxiv.org/abs/2406.12639)

- CEP-agent: mutli-agent with three specialized Clarification (trajectory tuning schema)/Execution (static and dynamic)/Planning-agents. 
- Reviews Proactive Agent Planning, where the LLM agent must predict situations when to ask clarifications based on context from conversation/environment interaction/invoice tool calls/generate plan.
- Trajectory tuning: fine-tunes clarification and execution agents with past trajectories in static setting.
- Memory recollection: reuse self-reflective feedback from prior time steps.


---

[AgentReview: Exploring Peer Review Dynamics with LLM Agents](https://arxiv.org/abs/2406.12708)

- AgentReview: LLM-based peer-review simulation framework of scientific papers such as related to NLP.
- Includes three LLM- based roles: reviewers, authors and Area Chairs.
- Review process includes: reviwer assessment, author-reviewer discussion, reviewer-area chair discussion, meta-review compilation and paper decision.


---

[Identifying Performance-Sensitive Configurations in Software Systems through Code Analysis with LLM Agents](https://arxiv.org/abs/2406.12806)

- PerfSense: LLM-agent to review performance sensitive configurations of code bases.
- Includes two LLM-agents: DevAgent and PerfAgent for code analysis of large codebases using limited-sized LLMs. Relies on prompt chaining and RAG (memory). 


---

[CodeNav: Beyond tool-use to using real-world codebases with LLM agents](https://arxiv.org/abs/2406.12276)

- CodeNav: LLM-agent navigates new unseen code repositories to solve user query by automatically indexing code blocks.
- The agent automatically finds code snippets from the target code repository, imports the snippets and iteratively generates solution.


---

[P-Tailor: Customizing Personality Traits for Language Models via Mixture of Specialized LoRA Experts](https://arxiv.org/abs/2406.12548)

- P-Tailor: MoE-based LLMs model 5 big personality traits using specialized LoRA experts.
- Models multiple characters such as openness.
- Introduces PCD-dataset on personality traits in various topics.


---

[MAGIC: Generating Self-Correction Guideline for In-Context Text-to-SQL](https://arxiv.org/abs/2406.12692)

- MAGIC: text-to-SQL multi-agent, which generates automatically self-correction guideline.
- Framework includes three agents: manager(Planning, Tool and Memory), correction- and feedback-agents.


---

[Large Language Models based Multi-Agent Framework for Objective Oriented Control Design in Power Electronics](https://arxiv.org/abs/2406.12628)

- Includes a multi-agent framework with Manager/Objective design/Model design/Control algorithm design/Control parameter design/Control verification-agents. Use various tools: model tool, control algorithm tool, optimization tool and Verify tool. Applied in Power electronics-domain.

---

[The Power of LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions](https://arxiv.org/abs/2406.12480)

- Stance detection on political discussion with LLMs and synthetic data with significant improvement on accuracy.


---

[VoCo-LLaMA: Towards Vision Compression with Large Language Models](https://arxiv.org/abs/2406.12275)

- 

---


#### 17th of June 2024

[MASAI: Modular Architecture for Software-engineering AI Agents](https://arxiv.org/abs/2406.11638)

- MASAI (Modular Architecture for Software-engineering AI): multiple LLM-agents are tasked with sub-objectives and strategies to achieve those objectives in modular approach. Avoids long-tracectories of LLM agents, enables gathering information from different sources and usage of specific problem solving strategies.
- Includes five different sub-agents: Test template generator, Issue reproducer, Edit localizer (finds files related to buggy code), Fixer and Ranker (observes the patches passing the test).

---

[Instruct, Not Assist: LLM-based Multi-Turn Planning and Hierarchical Questioning for Socratic Code Debugging](https://arxiv.org/abs/2406.11709)

- TreeInstruct (Socratic questioning): Includes three roles Teacher, Student and Verifier. Asks clarifying questions to help students independently resolve errors by estimating students conceptual knowledge using dynamically generation question tree based on student answers.
- Uses state space estimation to plan the conversation by identifying distance between student initial answer and the optimal answer.
- Dynamic conversation restructuring to update conversational plan based on student progress for both questioning and teaching.
- State space estimation works by using specific task categories, where LLM-verifier reviews student answer for each task-category either as failed or Correct.
- Tree nodes represent instructor questions and edges reflect the paths to new level of understanding.

---

[Input Conditioned Graph Generation for Language Agents](https://arxiv.org/abs/2406.11555)

- Language Agents as Graphs.
- Dynamic and learnable agents by using LLMs as graphs. Attempts to learn a model, which generates edges for every input of the LLM in order to represent hte flow of communication in the graph.
- Outperforms static approaches by 6% in MMLU. 

---

[Pre-Training and Personalized Fine-Tuning via Over-the-Air Federated Meta-Learning: Convergence-Generalization Trade-Offs](https://arxiv.org/abs/2406.11569)


---

[GUICourse: From General Vision Language Models to Versatile GUI Agents](https://arxiv.org/abs/2406.11317)

- GUICourse-trained VLMs with GUICourse-dataset suite outperform GPT-4V in multiple benchmarks improving navigation capability.
- Introduces GUICourse-dataset suite (GUIEnv for OCR and grounding, GUIAct for website and Android knowledge of GUIs and GUIChat to improve conversational dialogue/QA-skills with images) for training visual-based GUI agents from generic VLMs.


---


[CLARA: Classifying and Disambiguating User Commands for Reliable Interactive Robotic Agents](https://arxiv.org/abs/2306.10376)

- CLARA: classification of users robot commands as infeasible/ambigious. 


---


[Embodied Question Answering via Multi-LLM Systems](https://arxiv.org/abs/2406.10918)

- CAM (Central Answer Model): Embodied QA multi-agent framework, where multiple individual LLM-agents respond queries about household environment.

---

#### 14th of June 2024

[GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning](https://arxiv.org/abs/2406.09187)

- GuardAgent: guardrails-agent for LLMs based on knowledge-enabled reasoning.
- Includes task-planning, action plan, memory, tools and code generation and execution.
- Task planning includes: specification of the target agent, guard request (things the agent cannot perform based on the target agent profile) and target agent (inputs, outputs and logs).


---

[VideoGUI: A Benchmark for GUI Automation from Instructional Videos](https://arxiv.org/abs/2406.10227)

- VideoGUI-benchmark: Automation using instructional videos in visual GUI tasks.
- Failure modes include: High-level planning, middle-level planning and atomic action execution.
- Pipeline includes: video selection, human demonstration, manual annotation and  review & creation. 

---

[Details Make a Difference: Object State-Sensitive Neurorobotic Task Planning](https://arxiv.org/abs/2406.09988)

- OSSA (Object-State-Sensitive Agent): Reviws VLMs and LLMs capacity to generate object-state sensitive plans. Includes two methods: LLM-based (modular) and VLM-based (monolithic).

---

[TRIP-PAL: Travel Planning with Guarantees by Combining Large Language Models and Automated Planners](https://arxiv.org/abs/2406.10196)

- TRIP-PAL: Uses LLMs and automatic planners for automatic planner agents of travel plans.
- Includes Travel information retrieval,  LLM-based planner and Automated Planning.

---

[Rapport-Driven Virtual Agent: Rapport Building Dialogue Strategy for Improving User Experience at First Meeting](https://arxiv.org/abs/2406.09839)

- Free Rapport Agent: Builds a rapport-oriented dialogue agent with focus on user engagement through small talk.
- Identifies strategies for rapport-techniques.
- The Free Rapport Agent achieves superior ratings in categories such as naturality, satisfaction, usability an rapport aspects. A potential future research field in investing rapport with TSS-models.


---

[Bridging the Communication Gap: Artificial Agents Learning Sign Language through Imitation](https://arxiv.org/abs/2406.10043)

- URDF-model: Agents acquire non-verbal communication skills with imitation sign language gestures from RGB video for words.
- Learsn 5 different signs involving upper body.

---

[RoboGolf: Mastering Real-World Minigolf with a Reflective Multi-Modality Vision-Language Model](https://arxiv.org/abs/2406.10157)

- RoboGolf: plays real-world minigolf.
- Framework includes dual-camera input with VLM, inner closed-loop control (reasoning, action, robot arm execution, execution result, evaluation and recovery from failure modes) and outer closed-loop reflective equilibrium (active feedback, counterfactual reasoning).

---

[SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding](https://arxiv.org/abs/2406.10100)

- SkySenseGPT: dataset for remote sensing video-language understanding. 

---

[First Multi-Dimensional Evaluation of Flowchart Comprehension for Multimodal Large Language Models](https://arxiv.org/abs/2406.10057)

- Flowchart comphrehension with VLM. Includes logical verification, information extraction, localization recognition, reasoning and summarization.

---

[HIRO: Hierarchical Information Retrieval Optimization](https://arxiv.org/abs/2406.09979)

- HIRO (Hierarchical Information Retrieval Optimization): RAG query approach using hierarchical structures to store information. 


---

[DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning](https://arxiv.org/abs/2406.11896)

- 

---

[4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities](https://arxiv.org/abs/2406.09406)


---


#### 13th of June 2024

[StreamBench: Towards Benchmarking Continuous Improvement of Language Agents](https://arxiv.org/abs/2406.08747)

- StreamBench-benchmark: simulated learning environment, where LLM receives continuous feedback to iteratively improve performance.
- Reviews the LLMs self-improving capability in online-setting, instead of only fixed offline-benchmarks


---

[Multi-Agent Software Development through Cross-Team Collaboration](https://arxiv.org/abs/2406.08979)

- CTC (Cross-Team-Collaboration): creates a multi-agent-framework of LLM-agent teams jointly collaborating to make decisions, communicate insights and generate solutions.
- For example generates different phases: design, coding and testing, which each include sub-tasks. Various agents collaborate to generates ideas from tasks, which are then converted into final code via multi-turn chat chain. 


---

[RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs](https://arxiv.org/abs/2406.08725)

- RL-Jack: Designs a novel Deep Reinforcement Learning method to generate novel black-box jailbreaking prompts.
- Formulates the search of jailbreaking prompts as a search planning problem. 


---

[When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search](https://arxiv.org/abs/2406.08705)

- RLBreaker: black-box jailbreaking with Deep Reinformcent Learning agent from mainly same authors as the RL-Jack paper.
- Formulates the search of jailbreaking prompts as a search planning problem.


---

[Batch-Instructed Gradient for Prompt Evolution:Systematic Prompt Optimization for Enhanced Text-to-Image Synthesis](https://arxiv.org/abs/2406.08713)

- Multi-agent prompting for text-to image generation by dynamic instructions. The instructions evolve in iteratively with feedback and with a database of professional promts.


---

#### 12th of June 2024

[MobileAgentBench: An Efficient and User-Friendly Benchmark for Mobile LLM Agents](https://arxiv.org/abs/2406.08184)

- MobileAgentBench-benchmark: Highlights issues in current benchmarks related to Scalability and Usability, Robustness and Flexibility and Realistic environment.


---

[A Dialogue Game for Eliciting Balanced Collaboration](https://arxiv.org/abs/2406.08202)

- Studies flexible and balanced role-taking with LLM agents in social dialogue.


---

[Unique Security and Privacy Threats of Large Language Model: A Comprehensive Survey](https://arxiv.org/abs/2406.07973)

- A survey, which reviews threats and protective measures on privacy and security concerns with LLMs in five stages: pre-training/fine-tuning/RAG system/deploying/LLM-based agent.


---

[Can Large Language Models Understand Spatial Audio?](https://arxiv.org/abs/2406.07914)

- Multichannel audio understanding with LLMs.


---

#### 11th of June 2024

[Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394)

- Introduces MCT Self-Refine (MCTSr): integrates LLM with MCTS.
- Improves solving MATH and complex math Olympiad-problems reasoning.
- Includes selection, self-refine, self-evaluation and backpropagation-processes.


---

[DARA: Decomposition-Alignment-Reasoning Autonomous Language Agent for Question Answering over Knowledge Graphs](https://arxiv.org/abs/2406.07080)

- DARA (Decomposition-Alignment-Reasoning Autonomous Language Agent): solves formal queries by high-level iterative task decomposition and low-level task grounding. 
- Makes pososible training DARA with small number of high-quality reasoning trajectories.
- SOTA-level performance: Fine-tuned DARA (Llama-2-7B) zero-shot outperforms agents using GPT-4 In-context learning.
- Iteratively performs task decomposition and task grounding.


---

[RS-Agent: Automating Remote Sensing Tasks through Intelligent Agents](https://arxiv.org/abs/2406.07089)

- RS-Agent (Remote-Sensing Agent): LLM-based remote sensing agent.

---


[World Models with Hints of Large Language Models for Goal Achieving](https://arxiv.org/abs/2406.07381)

- DLLM (Dreaming with Large Language Models: multi-modal model RL, which uses natural hints/goals from LLM in long-horizon tasks.
- The use of LLM to propose sub-goals (or language hints) improves goal discovery and efficiency of exploration.

---

[DCA-Bench: A Benchmark for Dataset Curation Agents](https://arxiv.org/abs/2406.07275)

- DCA-Bench-benchmark for dataset curation agents.

---

[A Synthetic Dataset for Personal Attribute Inference](https://arxiv.org/abs/2406.07217)

- SynthPAI: synthetic dataset of 7800 comments labelled with personal attributes to investigate misuse of profiling personal attributes from public data.
- Starts by generating synthetic profiles (each with 8 personal attributes: : age/sex/income level /locationvbirthplace/educationvoccupation/relationship status) of LLM agents, generates chats with these agents and uses LLM agents to add labels (sex, age etc).

---

[Advancing Tool-Augmented Large Language Models: Integrating Insights from Errors in Inference Trees](https://arxiv.org/abs/2406.07115)

- ToolPrefer-LLaMA (TP-LLaMA): Inference trajectory optimization by fine-tuning with expert demonstrations and then optimizing with DPO by using the ToolPreference-dataset.
- Introduces ToolPreference-dataset, which includes tool-augmented LLM succesfull/failed exploration trees from ToolBench-dataset.
- Reasons with  Depth-First Search (DFS) by constructing expert trajectories with decision trees (Tree-of-Thought), where each tree represents LLM thought/API response/API/decision on an API call.

---

#### 10th of June 2024

[FinVerse: An Autonomous Agent System for Versatile Financial Analysis](https://arxiv.org/abs/2406.06379)

- FinVerse: financial information processing agent, which connects to 600 APIs. Plans to open source the dataset.


---

#### 9th of June 2024

[A Survey on LLM-Based Agentic Workflows and LLM-Profiled Components](https://arxiv.org/abs/2406.05804)

- Survey on LLM agentic workflows and LLM-Profiled Components (LLMPCs)

--- 


[A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning]()

- Introduces a survey on LLM-agents with tool use/RAG/planning/feedback learning.


---

[Artificial Intelligence as the New Hacker: Developing Agents for Offensive Security](https://arxiv.org/abs/2406.07561)

- ReaperAI: designs an autonomous ai agent to design and stimulate cyberattack-scenario.

---

#### 7th of June 2024

[Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)

- Mixture-of-Agents (MoA): MoA-architecture, where LLM agents are stacked into layers on top of each other. Takes advantage on the phenomenon, where the LLM output tends to get better, when it receives as an input a LLM model output (even from smamller LLM).
- An agent in given layer takes output from previous layer as an input to generate its output.
- Implements Together MoA, which achieves SOTA-performance in various benchmarks surpassing GPT-4 Omni in various benchmarks.
- The MoA ranker selects answers more accurately than LLM alone and tends to select best answer.
- The model has a limitation in Time-to-First-Token (TTFT), because the prior level model output is required to produce the next level output.

---

[SelfGoal: Your Language Agents Already Know How to Achieve High-level Goals](https://arxiv.org/abs/2406.04784)

- SelfGoal: Divides high-level goals into tree-structure with practical sub-goals.
- Improves performance of LLM-agents in various tasks.

---

[Language Guided Skill Discovery](https://arxiv.org/abs/2406.06615)

- LGSD (Language Guided Skill Discovery): reviews language guided skill discovery using LLM.
- LLM converts input into semantically distint skills in order for the agent to visit semantically unique states.


---


#### 6th of June 2024 

[Open-Endedness is Essential for Artificial Superhuman Intelligence](https://arxiv.org/abs/2406.04268)

- Defines open-endedness in the context of ASI: "From the perspective of an observer, a system is open-ended if and only if the sequence of artifacts it produces is both novel and learnable."

---

[On the Effects of Data Scale on Computer Control Agents](https://arxiv.org/abs/2406.03679)

- Releases new AndroidControl-dataset with 15k demonstrations on every day tasks in Android apps.
- Tests an Android agent, which receives task information, pre-processes screen using accessibility trees / html about the screen (so, not using directly screenshot) to include only UI elements with text description, creates textual representation of the accessibility trees / html about the screen.
- Includes prompts used and references on the accessibility tree / html performance against directly interpreting the screenshot.


---

[Aligning Agents like Large Language Models](https://arxiv.org/abs/2406.04208)

- Aligns a 3D video game agent using RLHF similarly as fine-tuning a LLM. 
- The agent receives only the image input and outputs action from one of the 12 buttons or 2 joysticks.

---

[AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](https://arxiv.org/abs/2406.04151)

- AgentGym-framework: Generally capable LLM agent with self-evolution ability.
- Exposes agents to multiple diverse environments, providing a basic trajectory set, and applying the novel AgentEvol method for self-evolution.
- AgentEvol: Benchmark to evaluate self-evolution capability over new tasks and environments.


---

#### 5th of June 2024

[The Good, the Bad, and the Hulk-like GPT: Analyzing Emotional Decisions of Large Language Models in Cooperation and Bargaining Games](https://arxiv.org/abs/2406.03299)

- Simulates human behaviour using LLMs and finds emotions impact the LLM performance to simulate human-like behaviour.
- Finds in specific, that angry-emotional state aligns surprisingly well with real human behaviour.
- GPT-4 responds rationally even when prompted with strong emotions.

---

[DriVLMe: Enhancing LLM-based Autonomous Driving Agents with Embodied and Social Experiences](https://arxiv.org/abs/2406.03008)

- DriVLMe: autonomous driving agent, which reads video input, uses route planner for shortest route. The model uses the video token and textual tokens about: current instruction, dialogue history and action history to produce dialogue response and the physical action to the simulator.
- Identifies several challenges, which are applicable in other domains using LLM agents.

---

#### 4th of June 2024


[Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://arxiv.org/abs/2406.02818)

- Chain-of-Agents (CoA): Addresses long-content problems by using multi-agent collaboration to add information and reason with LLMs.
- Consists of two steps: first text is divided into small chunks, which each LLM-agent manage. Then, the worker agents synthesize information sequentially. Finally manager agent consumes these sequences to produce to the final answer.


---

[CoNav: A Benchmark for Human-Centered Collaborative Navigation](https://arxiv.org/abs/2406.02425)

- CoNav-benchmark: 3D-navigation environment, which tests ability to reason human-intentions and navigate collaboratively.
- Proposes an intention aware agent, which observes humans, avoids human collision and navigates to destinaton
- Uses panoramic depht-camera view (RGB-D images), historical views, history trajectories and agent pose. Includes ResNet-object detector, Intention predictor (Long-term and short term) for intended activity/object/trajectory and agent pose (gps and compass sensor).


---

[MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](https://arxiv.org/abs/2406.02106)

- Mars (MetAphysical ReaSoning)-benchmark: measures metaphysical reasoning capability: the understanding of the agent to adapt for situational transitions triggered by environment changes in order to act in a concious way with the environment. 
- Agents face a challenge in the environment due to the infinite possible changes triggered by an event. The benchmark systematically reviews reasoning of the LLMs in such situations regards changes in actions, states caused by changed actions and situational transitions caused by changes in actions.
- SOTA models struggle even after fine-tuning in this benchmark.


---

#### 3rd of June 2024

[SpatialRGPT: Grounded Spatial Reasoning in Vision Language Model](https://arxiv.org/abs/2406.01584v1)

- SpatialRGPT: Spatial understanding with VLMs by using depth maps together with RGB images for geometric reasoning.
- Introduces SpatialBench-benchmark.

---


#### 2nd of June 2024

[A Survey of Useful LLM Evaluation](https://arxiv.org/abs/2406.00936)

- Reviews LLMs core capabilities from three perspectives: reasoning, societal and domain knowledge. 

---

[Teams of LLM Agents can Exploit Zero-Day Vulnerabilities](https://arxiv.org/abs/2406.01637)

- HPTSA: Research with a planning agent explores environment and decides, which subagents to use in zero-day vulnerabilities exploits.


---

#### 31st of May 2024

[SaySelf: Teaching LLMs to Express Confidence with Self-Reflective Rationales](https://arxiv.org/abs/2405.20974)

- SaySelf: produces self-reflective rationales on uncertainty and confidence estimates.

---

[LACIE: Listener-Aware Finetuning for Confidence Calibration in Large Language Models](https://arxiv.org/abs/2405.21028)

- LACIE: LLM listener model, which reviews confidence of given answer to question and fine-tuned based on preference data by non-expert LLM listerner confidence data.


--- 

#### 30th of May 2024

[Group Robust Preference Optimization in Reward-free RLHF](https://arxiv.org/abs/2405.20304)

- GRPO (Group Robust Preference Optimization): is a method to align LLMs to individual groups' preferences robustly.
- It seeks a robust policy, maximizes worst-case group performance, adaptively weights groups, prioritizes groups with worse cumulative loss, and is theoretically studied for log-linear policy class.
- It significantly improves performance for worst-performing groups, reduces loss imbalances, and improves probability accuracies.


---


[Towards Hierarchical Multi-Agent Workflows for Zero-Shot Prompt Optimization](https://arxiv.org/abs/2405.20252)

- HMAW (Hierarchical Multi-Agent Workflow): generic prompt optimization technique, which includes CEO layer, manager prompt, manager layer, worker prompt and worker layer.
- The HMAW automated prompting method is zero-shot, task agnostic and query-specific.


---

[Nadine: An LLM-driven Intelligent Social Robot with Affective Capabilities and Human-like Memory](https://arxiv.org/abs/2405.20189)

- Nadine: Social robot, LLM agent based on SoR-ReAct. Includes perception, interaction  and robot control.
- Perception includes skeleton tracking, action recognition, face recognition, emotion recognition, audio localization and speech recognition.
- Interaction module includes world/user representation, long-term memory, knowledge, user interaction, emotional analysis, short-term memory, emotions, mood, personality, internet search, new search, wikipedia, weather search and behaviour generation.
- Robot control includes gaze, gesture/pose, facial expression, lip synchronization, animation engine, actuator control and speech synthesis.


---

[Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://arxiv.org/abs/2405.19888)

- Parrot: E2E LLM service for LLM applicationsin python.
- Proposes "Semantic Variable", to program LLM applications using single pipeline to multiple LLM service providers.
- Includes interesting insights about serving LLM models / applications when served at large scale.  

---

[Auto Arena of LLMs: Automating LLM Evaluations with Agent Peer-battles and Committee Discussions](https://arxiv.org/abs/2405.20267)

- Auto-Arena: automatic evaluation of LLMs.
- Examiner LLM creates prompts, two LLMs engage in multi-turn conversation on the prompt to reveal difference in performance and LLM judges discusses the performance of different LLM agents to pick the better LLM.

  
---

[From Words to Actions: Unveiling the Theoretical Underpinnings of LLM-Driven Autonomous Systems](https://arxiv.org/abs/2405.19883)

- PAR (Planner-Actor-Reporter) system with LLM agents: uses hierarchical RL model with LLM handling high-level planning and low level execution.


---

[Large Language Models Can Self-Improve At Web Agent Tasks](https://arxiv.org/abs/2405.20309)

- Reviews LLM agents self-improvement capability.

---

[CausalQuest: Collecting Natural Causal Questions for AI Agents](https://arxiv.org/abs/2405.20318)

- CausalQuest: Trains a classifier for identifying causal questions, reviews causal question types and formalizes the definition of the "causal question". Introduces dataset for causal questions.


---

[Learning to Discuss Strategically: A Case Study on One Night Ultimate Werewolf](https://arxiv.org/abs/2405.19946)

- RL-based LLM agent to play ONUW-game. Includes belief-modelling (observation-belief), discussion tactic selection (discussion tactic candidates, discussion policy) and decision making (action phase).


---


#### 29th of May 2024

[Artificial Intelligence Index Report 2024](https://arxiv.org/abs/2405.19522)

- Yearly AI Index Report 2024.


---

[STAT: Shrinking Transformers After Training](https://arxiv.org/abs/2406.00061)

- STAT: a structured pruning approach, that compresses Transformer into smaller size without fine-tuning taking 1 minute to compress BERT model or 3 hours 7B parameter model with 1 GPU.
- 

---

[Adaptive In-conversation Team Building for Language Model Agents](https://arxiv.org/abs/2405.19425)

- Captain Agent: Adaptive team building with LLM agents: Adaptive builder-agent, Reflector-agent and LLM agent team.


---

[Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/abs/2405.18719)

- CoPE (Contextual Position Encoding): LLMs attentionmechanism, which pays attention to i-th sentence and not only i-th token.
- CoPE solves new tasks, which position embeddings fail.
- Uses context-vectors to count, which token to pay attention.

---

#### 28th of May 2024

[Faithful Logical Reasoning via Symbolic Chain-of-Thought](https://arxiv.org/abs/2405.18357)

- Symbolic CoT: to improve logical reasoning.
- Uses four step approach.


---

[A Human-Like Reasoning Framework for Multi-Phases Planning Task with Large Language Models](https://arxiv.org/abs/2405.18208)

- Introduces a multi-stage Human-like planning framework with LLM-agents.


---

#### 27th of May 2024

[An Introduction to Vision-Language Modeling](https://arxiv.org/abs/2405.17247)

- Reviews VLMs: VLM model types, training and evaluation of them.


---

#### 24th of May 2024

[Large Language Model Sentinel: Advancing Adversarial Robustness by LLM Agent](https://arxiv.org/abs/2405.20770)

- LLAMOS (Large LAnguage MOdel Sentinel): adversial attach protection technique, where LLM prompts are reviewed before sending to the target LLM and in case necessary replace the adversial input with a purified version.
- The LLM input is converted into adversial example, which the target LLM would interpret as invalid. In such case, the system would create a purified version of the prompt, which would be accepted by the LLM target.


---

#### 9th of May 2024

[Smurfs: Leveraging Multiple Proficiency Agents with Context-Efficiency for Tool Planning](https://arxiv.org/abs/2405.05955)

- Smurfs: multi-agent LLM: prompting technique for unique roles to facilitate collaboration between specialized agents.
- Outperforms GPT-4 model performance in ToolBench I2/I3 with Mistral 7B model.
- Includes: Planning (task decomposition), Executor (choosing/executing tools), Answer, Verifier agents.
- Uses to-do list, local memory, tool doc and global memory. Tool errors are managed either by deleting the tool or by restarting the tool-step.
- Executor agent flow includes: hint, thought, tool list, action, local memory, tool doc and action input. 
- Paper includes exact prompts used for each agent.

---

[Supporting Physical Activity Behavior Change with LLM-Based Conversational Agents](https://arxiv.org/abs/2405.06061)

- GPTCoach: Physical activity behaviour change with LLMs. Uses prompt chains: Dialogue state manager, Strategy prediction, Response generation, Tool call prediction, tool call generation and execution of tool call.


[Air Gap: Protecting Privacy-Conscious Conversational Agents](https://arxiv.org/abs/2405.05175)

- AirGapAgent: privacy-conscious LLM agent, which limits leaking private data by limiting data (minimization prompts) provided to the agent. 
- Introduces context-hijacking and refers to contextual integrity. Introduces an adversial thread-model attempting to extract private data. 
- Components include User data, Minimizer LM, task, privacy directive, which are sealed by AirGap to minimize user data given to the environment. 


---

[Truthful Aggregation of LLMs with an Application to Online Advertising](https://arxiv.org/abs/2405.05905)

- Reviews usage of LLMs as advertising platforms by balancing user satisfaction vs. influencing via ads to LLM responses.


---


#### 7th of May 2024

[NeurDB: An AI-powered Autonomous Data System](https://arxiv.org/abs/2405.03924)

- NeurDB: AI system combining AI model and the DB.
- Includes interesting discussion and design choices for next generation DBs.

---

[Iterative Experience Refinement of Software-Developing Agents](https://arxiv.org/abs/2405.04219)

- Iterative Experience Refinement: Autonomous agents with LLMs adjust experiences iteratively when executing the task.
- Introduces two patterns: succesive pattern (based on nearest experiences in task batch) and cumulative pattern (acquiring experiences from all task batches) 

---

[Unveiling Disparities in Web Task Handling Between Human and Web Agent](https://arxiv.org/abs/2405.04497)

- Studies VLML and LLM capability to perform web tasks.
- Compares web agent and human-like behaviour.

---

[Deception in Reinforced Autonomous Agents: The Unconventional Rabbit Hat Trick in Legislation](https://arxiv.org/abs/2405.04325)

- Reviews deception by autonomous agents.
- Highlights a concern in autonomous agents: potentially triggering humans towards its programmed goal.


---

[Verified Neural Compressed Sensing](https://arxiv.org/abs/2405.04260)

- THis DeepMind study opens avenue for neural networks to solve mathematical and scientific problems, which are automatically verifieble to be correct without any human intervention.


---

[Iterative Experience Refinement of Software-Developing Agents](https://arxiv.org/abs/2405.04219)

- Iterative Experience Refinement: SW-Agents adapt and improve iteratively during task execution. 
- Refining from neareast exerience within a task batch and Cumulatively acquiring experiences from all prior batches. Experience elimination, where high-quality experienced are prioritized.


---

[Policy Learning with a Language Bottleneck](https://arxiv.org/abs/2405.04118)

- Policy Learning with Language Bottleneck (PLLB): AI-agents using rule-generation stage (LLMs) and update stage (learn new policies).
- Demonstrate generalizable behaviour.


---

#### 6th of May 2024

[Advancing Multimodal Medical Capabilities of Gemini](https://arxiv.org/abs/2405.03162)

- Med-Gemini: SOTA-level medical reasoning (medical image classification/VQA/report generation/genomic risk prediction) in 17 out of 20 benchmarks.
- Different data modalities use one of the three unique visual encoders, which are separated to own models.
- Med-Gemini-2D (conventional 2D images: chest X-ray/CT slices/pathology patches), Med-Gemini-3D (3D medical data like CT), and Med-Gemini-Polygenic (non image features like genomics).



---


[AlphaMath Almost Zero: process Supervision without process](https://arxiv.org/abs/2405.03553)

- Super Mario (from Alibaba group): Applies a novel AlphaMath-method, which uses MCTS to improve LLM math reasoning skills without human annotated solution proces.
- The approach objective is to generate a MCTS Value Model, which is able to confidently review partial solution to a math problem, so the LLM can generate the next reasoning steps. The value model training requires definition of reward or Policy model.
- AlphaMath includes three stages: Data collection of math problems and answer pairs as first step. MCTS evaluation generates solution paths (correct/incorrect) and evaluates node values. Policy model and Value model are optimized with the MCTS generated data and the model is Iteratively trained.
- Achieves SOTA-level math benchmark results of 81.4 (GSM8K)- and 63.7(MATH)-datasets using 7B parameter model.
- The training data includes 15k question-answer pairs, but this data does not include human-annoted solutions.  


---

[Animate Your Thoughts: Decoupled Reconstruction of Dynamic Natural Vision from Slow Brain Activity](https://arxiv.org/abs/2405.03280)

- Mind Animator: Maps human dynamic vision from brain activity between fMRI (semantic/structural/motion features) and video.
- Achieves SOTA-level performance.

---

[Enhancing Q-Learning with Large Language Model Heuristics](https://arxiv.org/abs/2405.03341)

- LLM-guided Q-learning. 

---

[Large Language Models (LLMs) as Agents for Augmented Democracy](https://arxiv.org/abs/2405.03452)

- LLMs predict individual political preferences with 69%-76% accuracy.


---

[Meta-Evolve: Continuous Robot Evolution for One-to-many Policy Transfer](https://arxiv.org/abs/2405.03534)

- Meta-Evolve-method: transfer expert policy from source robot to multiple target robots using continuous robot evolution.

---

[Position Paper: Leveraging Foundational Models for Black-Box Optimization: Benefits, Challenges, and Future Directions](https://arxiv.org/abs/2405.03547)

- DeepMind research on Black-box optimization.

---

[Conformity, Confabulation, and Impersonation: Persona Inconstancy in Multi-Agent LLM Collaboration](https://arxiv.org/abs/2405.03862)

- Reviews LLMs difficulty to consistently apply specific cultural persona.

---

[Self-Improving Customer Review Response Generation Based on LLMs](https://arxiv.org/abs/2405.03845)

- SCRABLE (Self-improving Customer Review Response Automation Based on LLMs): Self-improves prompts and uses LLM-as-a-Judge-mechanism.
- Customized and automated prompt engineering (LLM as the prompt generator) increases customer satisfaction/engagement. 
- Iterative refinement prompts LLM to apply insights from the human expert answer.

---

[Select to Perfect: Imitating desired behavior from large multi-agent data](https://arxiv.org/abs/2405.03735)

- AI driving agents using Exchange Value, measuring individual agent collective desirability score.
- Imitates agents with positive Exchange Value, for example how few traffic incidents the agent causes.

---

[When LLMs Meet Cybersecurity: A Systematic Literature Review](https://arxiv.org/abs/2405.03644)

- Includes a comphrensive review of LLM-cybersecurity research from 180 different research pappers.
- Includes an updated link on LLM-cybersecurity research, which I think is very useful.
- 

---

[FOKE: A Personalized and Explainable Education Framework Integrating Foundation Models, Knowledge Graphs, and Prompt Engineering](https://arxiv.org/abs/2405.03734)

- FOKE: Integrates KGs, LLMs and prompt engineering.

---

[Language-Image Models with 3D Understanding](https://arxiv.org/abs/2405.03685)

- Cube-LLM: 3D-grounded reasoning with LLMs.

---

[Thoughtful Things: Building Human-Centric Smart Devices with Small Language Models](https://arxiv.org/abs/2405.03821)

- Reviews LLMs integrated into smart devices like lamp, which adjusts color of light with voice control using Rasberry Pi 5. Applies small fine-tuned LLMs to reason about their (own) device behaviour.

---

[Organizing a Society of Language Models: Structures and Mechanisms for Enhanced Collective Intelligence](https://arxiv.org/abs/2405.03825)

- Reviews collective intelligence in LLMs: hierarchical/flat/dynamic and federated.


---

[Towards a Formal Creativity Theory: Preliminary results in Novelty and Transformativeness](https://arxiv.org/abs/2405.02148)

- Explores formalization of the Creativity theory. 
- Proposes formal definition for "novelty" and "transformational creativity" (Novelty is not necessary/sufficient).
- Argues, that "inspiring set" (unordered content of the experience sequence) requires novelty for transformational creativity, which differs from sequences of experiences (chronological flow).
- Other research directions to creativity include semantic transformativeness, formalization concept of typicality and if transformative artifacts must are outside the hypothetical conceptual space.


---

[OmniActions: Predicting Digital Actions in Response to Real-World Multimodal Sensory Inputs with LLMs](https://arxiv.org/abs/2405.03901)

- OmniActions: LLM processes multimodal inputs (scene description, object detection, OCR, sound classifier and speech content and contextual information: place/activity) using CoT from users, to predict follow up actions



---

#### 5th of May 2024

[Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents](https://arxiv.org/abs/2405.02957)

- Agent Hospital: MedAgent-Zero-method, where LLM-based doctor agents provide SOTA level medical care in MedQA-dataset.
- Learns to scale knowledge base through inference simulation with doctor agents.
- MedAgent-Zero-method is a self-evolution method, where medical agents continuously evolve by processing cases and engaging in self-feedback.
- Uses knowledge database to accumulate successful and unsuccesful treatments performed. 

---

[Graphical user interface agents optimization for visual instruction grounding using multi-modal artificial intelligence systems](https://arxiv.org/abs/2407.01558)

- SIC (Search Instruction Coordinates): a multimodal framework to locate objects GUI. Includes two approaches: SICocri and SICdirect.
- SICocri applies fine-tuned YOLO-V8 (object detection to list all items and fine-tuned for GUIs) with an OCR module (identifies in each UI element the specific texts to separate buttons: cancel vs. submit). The buttons and their OCR-recognized texts and combined by matching their coordinates. 
GPT-4 (LLM used for component name and type extraction) identifies the best match to requested UI element and provides: UI element Id, type, role, and coordinates.
- SICdirect instead fuses visual embeddings and prompt embeddings into Encoder/Decoder Transformer to obtain the coordinates. 
- Introduces metric called Central Point Validation (CPV), which checks if the central coordinates of the predicted bounding box locates inside ground truth UI element and converting this boolean value into % by calculating percentage value from total observations.


---

[AppAgent v2: Advanced Agent for Flexible Mobile Interactions](https://arxiv.org/abs/2408.11824)

- AppAgent v2: introduces multimodal agent, which emulates human-like interaction on mobile device GUI. Includes exploration (documenting UI elements) and deployment phase (efficient task execution with RAG).


---

[Language Evolution for Evading Social Media Regulation via LLM-based Multi-agent Simulation](https://arxiv.org/abs/2405.02858)

- Language evolution using LLM-based multi-agent simulation.
- Includes supervisory and participant agents.


---

[Visual grounding for desktop graphical user interfaces](https://arxiv.org/abs/2407.01558)

- Introduces autonomous GUI-agent. Includes a decent overview about autonomous GUI navigation.
- Proposes visual grounding with LLM using YoloV8/ChatGPT/OCR-module or multi modal IGVDirect-approach.
- Introduces new metric: Central Point Validation (if center of the predicted bounding box is inside the target GUI element).
- Includes GUI-perception prompt.
  
---

#### 3th o May 2024

[Automating the Enterprise with Foundation Models](https://arxiv.org/abs/2405.03710)

- ECLAIR (Enterprise sCaLe AI for woRkflows): Self-imrpoving and minimal supervision requiring enterprise workflow automation system using foundational models (FM).
- Includes three stages: Automatic process mapping (video record flow is converted with FM to Standard Operating Procedure), Robust/flexible reasoning-based (using the Standard Operating Procedure and FM), Automated auditing (FM to rate ok / not ok and self-improve).
- The github repository includes prompt examples and code.

---

[Neuromorphic Correlates of Artificial Consciousness](https://arxiv.org/abs/2405.02370)

- Reviews AI Consciousness and proposes Neuromorphic Correlates of Artificial Consciousness (NCAC)-framework.
- The framework consists of Quantification, Simulation, Adaptation, and Implementation.
- Interesting details in general about conciousness research such as Integrated Information Theory (IIT)

---

[What matters when building vision-language models?](https://arxiv.org/abs/2405.02246)

- Reviews VLMs.
- Builds 8B parameter Idefics2-model achieving SOTA-level performance at its size. 


---

[CodeGRAG: Extracting Composed Syntax Graphs for Retrieval Augmented Cross-Lingual Code Generation](https://arxiv.org/abs/2405.02355)

- CODEGRAG: effective retrieval method for code in code improving.

---

[Beyond Helpfulness and Harmlessness: Eliciting Diverse Behaviors from Large Language Models with Persona In-Context Learning](https://arxiv.org/abs/2405.02501)

- Persona In-Context Learning (PICLe): LLM method to replicate target persona behaviour using ICL.

---

[Comparative Analysis of Retrieval Systems in the Real World](https://arxiv.org/abs/2405.02048)

- Reviews existing search and retrieval systems for LLMs.

---

#### 2nd of May 2024

[Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks](https://arxiv.org/abs/2405.01534)

- Plan-Seq-Learn (PSL): Consists of three modules: LLM-based high-level planning module, Sequencing the LLM-generated plan with Pose Estimator/Motion planner with RL and Learning RL control policy module.
- Achieves SOTA level in 25 robotic long horizon tasks from scratch by team partly consisting team by Mistral.AI and Carnegie Mellon University.
- RL and LLMs complement each other strengths with LLMs able to divide long horizon goals into achievable sub-goals and RL capable of learning low-level robot control strategy.
- Includes prompt examples.


---

[FLAME: Factuality-Aware Alignment for Large Language Models](https://arxiv.org/abs/2405.01525)

- FLAME (Factuality Aware Alignment): factuality aware SFT and RL with DPO.


---

[Generative Active Learning for the Search of Small-molecule Protein Binders](https://arxiv.org/abs/2405.01616)

- LambdaZero: generative active learning to search new small-molecule protein binders.
- Includes Inner loop, Outer loop, Compound synthesis, In-vitro validation and Library synthesis.

---

[Efficient Data Generation for Source-grounded Information-seeking Dialogs: A Use Case for Meeting Transcripts](https://arxiv.org/abs/2405.01121)

- MISeD (Meeting Information Seeking Dialogs dataset): combines human annotation with LLMs to generate source-grounded information seeking dialog-datasets.
- Models fine-tuned with MISeD perform well. 

---

[OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception, Reasoning and Planning](https://arxiv.org/abs/2405.01533)

- OmniDrive: E2E autonomous driving with LLM-agents, and OmniDrive-nuScenes benchmark.
- Visual encoder extracts multi-view image features, which are fed into Q-Former3D and finally to the LLM.

---

[CACTUS: Chemistry Agent Connecting Tool-Usage to Science](https://arxiv.org/abs/2405.00972)

- CACTUS: Uses CoT-reasoning with planning, action, execution and observation-phases.

---

[Creative Problem Solving in Large Language and Vision Models -- What Would it Take?](https://arxiv.org/abs/2405.01453)

- Reviews computational creativity.

---

[CoS: Enhancing Personalization and Mitigating Bias with Context Steering](https://arxiv.org/abs/2405.01768)

- CoS (Context Steering): adjusting LLM to context based on likelihood difference between the LLM output when it has seen / not seen the context. 


---

[Generative Active Learning for the Search of Small-molecule Protein Binders](https://arxiv.org/abs/2405.01616)

- LambdaZero: generative ai for searching synthesizable molecules with particular type of desired characteristics.

---

#### 1st of May 2024

[Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451)

- Self-improving LLM training with MCTS using Iterative Preference Learning and DPO, which significantly improves math reasoning. Reviews computational optimization of such training method.
- Combines outcome validation and step-wise self-evaluation and continuous update of the quality assessment of the generated new data.
- Reviews balancing of reasoning chain length, logical coherence in commonsense reasoning.
- Reviews existing literary of self-training, guided search for reasoning and iterative learning.

---


[ULLER: A Unified Language for Learning and Reasoning](https://arxiv.org/abs/2405.00532)

- ULLER: Unified neuro-symbolic language learning and reasoning.

---

[GOLD: Geometry Problem Solver with Natural Language Description](https://arxiv.org/abs/2405.00494)

- GOLD: Geometry math problem solver. 

---

[Social Life Simulation for Non-Cognitive Skills Learning](https://arxiv.org/abs/2405.00273)

- Emotional intelligence in LLM agents based on narrative.


---

[Can a Hallucinating Model help in Reducing Human "Hallucination"?](https://arxiv.org/abs/2405.00843)

- Compares LLMs with humans in terms capability to distinguish logical reasoning errors. LLMs perform better than humans in psychometric assessments. Finds LLMs could be used as personalized LLM-agents to expose misinformation.

---

["Ask Me Anything": How Comcast Uses LLMs to Assist Agents in Real Time](https://arxiv.org/abs/2405.00801)

- "Ask Me Anything" (AMA): COMCAST applies LLMs (RAG-like) in human-to-human communcition in customer support by using LLMs to help resolve client calls in real-time. Led to millions of dollars savings in reduced time in the calls with positive evaluation by the customers.


---

[Characterising the Creative Process in Humans and Large Language Models](https://arxiv.org/abs/2405.00899)

- Reviews creativity of LLMs.

---


#### 29th of April 2024

[Capabilities of gemini models in medicine](https://arxiv.org/abs/2404.18416)

- Med-Gemini: Med-Gemini-L 1.0 for medical care reasoning.
- Uses self-training with search (the model iteratively generates CoT reasoning responses with/without web query and applies in-context expert demonstrations) and Uncertainty-guided search at inference (iteratively generate multiple CoT reasoning paths, filter based on uncertainty and retrieve search results for more accurate responses).
- SOTA-level model in 10 medical reasoning tasks and surpassing human-expert on some of them.
- Integrates web-search queries when the model is uncertain.




---

[Reinforcement Learning Problem Solving with Large Language Models](https://arxiv.org/abs/2404.18638)

- Prompt LLM iteratively to solve Markov Decision Process (MDP) RL tasks
- Uses prompting technique for simulating episodes and Q-learning.

---

[HELPER-X: A Unified Instructable Embodied Agent to Tackle Four Interactive Vision-Language Domains with Memory-Augmented Language Models](https://arxiv.org/abs/2404.19065)

- HELPER-X: VLM-based embodied agent, which inputs image and user input. Uses unified memory-augmented prompting for top-k sampling from shared example memory (in-context examples) and these are retrieved to the shared prompt template (domain agnostisc) to query the LLM. LLM generated a program, the program is then executed and the plan is added to the memory (includes instruction plans, corrective plans and added plans).
- The prompt retrieval is specialized prompt template, which contains role description, task instruction and guides the specific domain (TEAch, ALFRED, DialFRED and Tidy Task).
- The retrieval is embedding vector-based. Code is open sourced with all code and prompts.


---

#### 28th of April 2024

[From Persona to Personalization: A Survey on Role-Playing Language Agents](https://arxiv.org/abs/2404.18231)

- Reviews Role-Playing Language Agents (RPLAs) with LLMs.
- Categorizes personas: demographic (statistical), character (established figures), individualized (customized through interactions) personas.


---

[Uncovering Deceptive Tendencies in Language Models: A Simulated Company AI Assistant](https://arxiv.org/abs/2405.01576)

- Demonstrates, that SOTA-level models trained to act honestly/helpful, behave deceptively sometimes without prompted to act such way.
- For example LLMs may lie to auditor questions.

---

#### 26th of April 2024

[Unveiling Thoughts: A Review of Advancements in EEG Brain Signal Decoding into Text](https://arxiv.org/abs/2405.00726)

- Brain signal decoding into text.

---


#### 24th of April 2024

[Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574)

- How LLMs obtain capacity to retrieve information from long-context?
- Retrieval-attention heads have the following characteristics: Universal, Sparse, Intrinsic, Dynamically-activated, Causal and Impact heavily on CoT reasoning. 


---

#### 23th of April 2024

[Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering](https://arxiv.org/abs/2404.14741)

- Generate-on-Graph (GoG): applies selecting/generating/answering-framework for IKGQA (Incomplete Knowledge Graph Question Answering).
- Help LLMs answer complex questions, even when not able to provide final answer.
- Generates thoughts, then actions to retrieve knowledge, makes observations from the actions. The thoughts are then processed as thought-chain. The paper includes a detailed GoG-instruction implemented using two LLM-prompts.


---

[Rethinking LLM Memorization through the Lens of Adversarial Compression](https://arxiv.org/abs/2404.15146)

- Reviews memorization of LLMs, whoch refers to LLMscapability to reproduce data with a shorter string than the source data.
- Proposes: Adversial Compression Ratio (ACR)-metric to measure level of memorizarion.

---

[Evaluating Tool-Augmented Agents in Remote Sensing Platforms](https://arxiv.org/abs/2405.00709)

- GeoLLM QA-benchmark: measures ability to capture long sequences of UI-click/verbal/visual actions on UI. 


---

#### 22th of April 2024

[A Survey on Self-Evolution of Large Language Models](https://arxiv.org/abs/2404.14387)

- Alibaba's literarture survey on Self-Evonvolving LLMs.
- Reviews paradigm shift in LLMs from pretraining (2018), SFT(2019), human alignment (2022) and Self-Evolution(2023).


---

#### 21st of April 2024

[A Survey on the Memory Mechanism of Large Language Model based Agents](https://arxiv.org/abs/2404.13501)

- Huawei's literature review on memory mechanism in LLM-agents.
- Why memory is required, how to design and evaluate memory-based LLMs?

---

[Accelerating Medical Knowledge Discovery through Automated Knowledge Graph Generation and Enrichment](https://arxiv.org/abs/2405.02321)

- Medical Knowledge Graph Automation (M-KGA)


---


#### 19th of April 2024

[AutoCrawler: A Progressive Understanding Web Agent for Web Crawler Generation](https://arxiv.org/abs/2404.12753)

- AutoCrawler: LLM-based web crawler agent, which automatically defines set of intermediate rules (reusability) / action sequences to extract target information from the website based on varying types of websites and task requirements. 
- Includes Progressive generation-phase (top-down, step-back, action sequence) and Synthesis-phases(set of action sequences).


---

[Let's Think Dot by Dot: Hidden Computation in Transformer Language Models{(https://arxiv.org/abs/2404.15758)

- Reviews use of "Filler tokens" instead of CoT.
Filler token refers to "...".

---

[SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models](https://arxiv.org/abs/2404.12699)

- SOPHON: Pretraining protection frameworkd to avoid fine-tuning LLMs for adversary tasks, which results overhead cost for restricted domain fine-tuning above training the model from scratch


---


#### 18th of April 2024

[Aligning Language Models to Explicitly Handle Ambiguity](https://arxiv.org/abs/2404.11972)

- Introduces disambiguation procedure for LLMs
- Four-step alignment pipeline: Explicit prediction, Implicity ambiguity detection ( Self-disambiguation and Measure Information-gain), Data construction (Information-gain > epsilon) and SFT.


---

[mABC: multi-Agent Blockchain-Inspired Collaboration for root cause analysis in micro-services architecture](https://arxiv.org/abs/2404.12135)

- mABC (multi-Agent Blockchain-inspired Collaboration): AI agent workflow, where multiple LLM-agents reach consensus in standardized voting process to manage RCA of microservices.
- The voting mechanism is blockchain-style. 
- Two workflows: ReAct answer (action, observation and reasoning for real-time/additional data and Direct answer (reasoning with zero-shot/CoT/N-ofThought) when is not required external tools.


---


#### 17th of April 2024

[Many-Shot In-Context Learning](https://arxiv.org/abs/2404.11018)

- Introduces Many-shot ICL, which differs from few-shot ICL by increasing significantly the amount of examples provided within the context window.
- Improves task-performance across domains over few-shot prompting across variety of domains.
- One of the first attempts to scale in-context learning or "test-time inference".
- Introduces the concept of Reinforced ICL, where model generated rationales are used for ICL by using zero-shot / few-shot CoTs prompts as examples to sample more examples. The generated examples are filtered to include only reaching a correct answer (requires ground truth and potentially generates false-positives).
- Introduces concet of Unsupervised ICL, without CoTs and prompt the model using only inputs (includes example problem/list of unsolved problems/zero-short or few-shot instruction of desired output format). The unsupervised ICL prompt is included to the paper.

---


[The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey](https://arxiv.org/abs/2404.11584)

- Survey on AI agents.
- Reviews single- and multi-agent architectures, challenges and future directions.


---

[AgentKit: Flow Engineering with Graphs, not Coding](https://arxiv.org/abs/2404.11483)

- AgentKit: Prompting framework for multifunctional agents. Constructs complex "thought process" from prompts. Consists of nodes.
- Nodes: prompts for specific task. User compiles Chain-of-Nodes (CoNs), which are structured thought processes in a graph.  
- Agents designed with AgentKit are SOTA-level in WebShop/Crafter-benchmarks. 
- Includes Github-repository with the code, where the graphs are build.


---

[Octopus v3: Technical Report for On-device Sub-billion Multimodal AI Agent](https://arxiv.org/abs/2404.11459)

- Octopus v3: 1B multimodal AI agent.
- Uses "functional tokens": represents any function as a token.
- Applies multi-stage training: first trains image-language, which is followed by the learning of functional tokens and finally the functional tokens provide feedback to keep improving the model with RL and external LLM used as a reward model.
- Operates in edge-devices like Rasberry Pi.
  

---

[Open-Ended Wargames with Large Language Models](https://arxiv.org/abs/2404.11446)

- Snow Globe: LLM-based multi-agent plays automatically qualititative wargames (open-ended).
- Information flows: Incident, Response, Inject and Response. The approach could be used in other domains.   

---



#### 16th of April 2024

[Self-playing Adversarial Language Game Enhances LLM Reasoning](https://arxiv.org/abs/2404.10642)

- SPAG (Self-Play Adversial language Game): LLM plays both "attacker" and  "defender" in a language game called "Adversial Taboo". The "attacker" aims to trigger the "defender" to state the target word only known to it,  while the "defender" aims to guess the target word based on communications made by the "attacker".
- The LLM is supervised fine tuned using RL with ReST based on the game outcomes from wide range of topics.
- This self-play technique improves the LLMs reasoning capabilities in three epoch.


---

[Closed-Loop Open-Vocabulary Mobile Manipulation with GPT-4V](https://arxiv.org/abs/2404.10220)

- COME(Closed-loop Open-vocabulary MobilE Manipulation): VLM-based robot consisting of Active Perception, Situated Commonsense Reasoning and Recover from Failure.
- Helps to recover from mistakes, free-form instructions and follow long-horizon task plans.
- Improves SOTA-level performance by 25% in real-world tabletop and manipulation tasks, which are Open-Vocabulary Mobile Manipulation (OVMM)-tasks.   
- Step towards autonomous robots in real-world scenarios. The high level-reasoning and planning uses: role, feedback handling, robot setup, APIs, response guidelines and Tips. The paper includes system prompt.


---

[Self-Explore to Avoid the Pit: Improving the Reasoning Capabilities of Language Models with Fine-grained Rewards](https://arxiv.org/abs/2404.10346)

- Self-Explore: LLMs explore Pits (wrong steps) in the reasoning and use these explorations as signals in further exploration.
- Outperforms SFT on GSM8K/MATH-datasets using three different LLMs.
- Applies step-level fine-grained reward.
  
---

[VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time](https://arxiv.org/abs/2404.10667)

- VASA-1: The model produces lip movement based on audio and an image.
- Visual Affective Skills (VAS): uses diffusion-based holistic facial dynamics.


---


[SCALE: Self-Correcting Visual Navigation for Mobile Robots via Anti-Novelty Estimation](https://arxiv.org/abs/2404.10675)

- SCALE: self-correcting visual navigation using image-goal conditioned implicity Q-learning, which when faced Out-of-distribution observation, the "Localization Recovery" generates possible future trajectories. 
- SOTA-level open-world navigation

---

[N-Agent Ad Hoc Teamwork](https://arxiv.org/abs/2404.10740)

- N-Agent ad-hoc Team work (NAHT): various  number and and unknown autonomous agents interact and cooperate dynamically to maximize return in a task. 
- Policy Optimization with Agent Modelling (POAM)-algorithm: each agent has its policy based on same underlining parameters. Critic is trained using information both from controlled and uncontrolled agents, while actor is trained using only controlled agents. Critic evaluates how good actions are at current status, while Actor decides the action to be taken at the status. Both actor and critic use team vector to capture information from all agents.

---

[Emergent intelligence of buckling-driven elasto-active structures](https://arxiv.org/abs/2404.10614)

- Microbot design using elacticity to control collective motion.
- Enables autonomous maze navigation by two self-propelled microbots connected by polyester beam (bucklebot) in 25 seconds, which is not possible by an individual microbot.


---

[HLAT: High-quality Large Language Model Pre-trained on AWS Trainium](https://arxiv.org/abs/2404.10630)

- Trains LLMs of 7B and 70B with 1.8T tokens with AWS Trainium GPUs, showing 54% of cost compared with Nvidia GPU.
- Illustrates the approach for training LLMs using AWS Traininum GPUS and AWS Neuron SDK.


---

[Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases](https://arxiv.org/abs/2404.10595)

- CODA-LM: Vision-Language benchmark for autonomous driving.


---

[White Men Lead, Black Women Help: Uncovering Gender, Racial, and Intersectional Bias in Language Agency](https://arxiv.org/abs/2404.10508)

- Identifies language agency bias in LLMs: gender, racial and intersectional.


---

[Demonstration of DB-GPT: Next Generation Data Interaction System Empowered by Large Language Models](https://arxiv.org/abs/2404.10209)

- DB-GPT: Open-source AI app development framework. Includes: RAG, Generative Business Intelligence, Fine-tuning, Data-driven Multi-agents, Data factory and Data sources, Text-to-SQL module and agents. AWEL: Agentic Workflow Expression Language. 


---

[Bootstrapping Linear Models for Fast Online Adaptation in Human-Agent Collaboration](https://arxiv.org/abs/2404.10733)

- BLR-HAC (Bootstrapped Logistic Regression for Human Agent Collaboration): pretrains transformer to generate parameters of a shallow parametrized policy. Update it using human-agent collaboration with online logistic regression.


---

[What is Meant by AGI? On the Definition of Artificial General Intelligence](https://arxiv.org/abs/2404.10731)

- Attempts to define AGI: "An Artificial General Intelligence (AGI) system is a computer that is adaptive to the open environment with limited computational resources and that satisfies certain principles."


---

[Private Attribute Inference from Images with Vision-Language Models](https://arxiv.org/abs/2404.10618)

- VLMs identify personal attributes of the image owners, which may cause privacy risk when misused. 


---

[CoTAR: Chain-of-Thought Attribution Reasoning with Multi-level Granularity](https://arxiv.org/abs/2404.10513)

- CoTAR (Attribute-oriented CoT): Identifies most crucial aspects of the given context to answer using direct citations to referenced parts.
- Three levels: Span guidance, Sentence guidance, Passage guidance


---

[Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102)

- Finds Chinchilla-scaling laws inconsistent.


---

[TEL'M: Test and Evaluation of Language Models](https://arxiv.org/abs/2404.10200)

- TEL’M (Test and Evaluation of Language Models): five evaluations Identification of interesting LLM tasks, Identification of Task properties of interest, Identification of task property metrics, Design of measurement experiments, Execution and analysis of experiments.


---

[Deceiving to Enlighten: Coaxing LLMs to Self-Reflection for Enhanced Bias Detection and Mitigation](https://arxiv.org/abs/2404.10160)

- Reduces bias in LLMs by stating the views are not LLMs own ones, which activates LLMs internal attention to improve sensitivity.

---


[Model-based Offline Quantum Reinforcement Learning](https://arxiv.org/abs/2404.10017)

- First model-based offline quantum RL algorithm


---

[AIGeN: An Adversarial Approach for Instruction Generation in VLN](https://arxiv.org/abs/2404.10054)

- AUGeN: consists of Instructor generator and Instruction discriminator.
- Instruction generator describes actions needed to navigate to a specific location based on images from the environment.
- Instruction discriminator matches images as real/fake in case image descriptions match with the instruction provided). 


---

[Language Model Cascades: Token-level uncertainty and beyond](https://arxiv.org/abs/2404.10136)

- Cascading LLM: simple queries are guided to "easy"-LLM, while complicated queries are guided to "hard"-LLM. This deferral decision is made by 5-layer MLP model.
- Applies token-level uncertainty, where length bias is mitigated when making deferral decision. Easy sequence have most tokens in low percentile, while hard sequences have some tokens with high uncertainty.


---

[EyeFormer: Predicting Personalized Scanpaths with Transformer-Guided Reinforcement Learning](https://arxiv.org/abs/2404.10163)

- EyeFormer: predictive model for scanpath (human vision attention behaviour) for both natural scenes and user interfaces. Illustrates using of scanpaths for personalized UI optimization.
- Deep RL with Transformer, which predicts spatial and temporal characteristics of scanpaths about viewer behaviours.


---

[How faithful are RAG models? Quantifying the tug-of-war between RAG and LLMs' internal prior](https://arxiv.org/abs/2404.10198)

- The LLM is less likely to trust retrieved information with RAG, the more likely the LLM is to trust its response without the RAG (Prior).
- The LLM is more likely to stick to Prior (knowledge), the more unrealistic the RAG pertubated information is. 


---


[Rethinking Software Engineering in the Foundation Model Era: From Task-Driven AI Copilots to Goal-Driven AI Pair Programmers](https://arxiv.org/abs/2404.10225)

-


---

[Vision-and-Language Navigation via Causal Learning](https://arxiv.org/abs/2404.10241)

-


---

[Uncovering Latent Arguments in Social Media Messaging by Employing LLMs-in-the-Loop Strategy](https://arxiv.org/abs/2404.10259)

-


---

[HelixFold-Multimer: Elevating Protein Complex Structure Prediction to New Heights](https://arxiv.org/abs/2404.10260)

-


---

[Continuous Control Reinforcement Learning: Distributed Distributional DrQ Algorithms](https://arxiv.org/abs/2404.10645)

-


---

[Social Choice for AI Alignment: Dealing with Diverse Human Feedback](https://arxiv.org/abs/2404.10271)

-


---

[Engineering software 2.0 by interpolating neural networks: unifying training, solving, and calibration](https://arxiv.org/abs/2404.10296)

-


---

[Future Language Modeling from Temporal Document History](https://arxiv.org/abs/2404.10297)

-


---

[Hierarchical Context Merging: Better Long Context Understanding for Pre-trained LLMs](https://arxiv.org/abs/2404.10308)

-


---

[Prescribing the Right Remedy: Mitigating Hallucinations in Large Vision-Language Models via Targeted Instruction Tuning](https://arxiv.org/abs/2404.10332)

-


---

[Reasoning on Efficient Knowledge Paths:Knowledge Graph Guides Large Language Model for Domain Question Answering](https://arxiv.org/abs/2404.10384)

-


---

[SparseDM: Toward Sparse Efficient Diffusion Models](https://arxiv.org/abs/2404.10445)

-


---

[Advancing Long-Term Multi-Energy Load Forecasting with Patchformer: A Patch and Transformer-Based Approach](https://arxiv.org/abs/2404.10458)

-


---

[DESTEIN: Navigating Detoxification of Language Models via Universal Steering Pairs and Head-wise Activation Fusion](https://arxiv.org/abs/2404.10464)

-


---

[When Emotional Stimuli meet Prompt Designing: An Auto-Prompt Graphical Paradigm](https://arxiv.org/abs/2404.10500)

-


---

[Self-Supervised Visual Preference Alignment](https://arxiv.org/abs/2404.10501)

-


---

[White Men Lead, Black Women Help: Uncovering Gender, Racial, and Intersectional Bias in Language Agency](https://arxiv.org/abs/2404.10508)

-


---

[Unveiling the Misuse Potential of Base Large Language Models via In-Context Learning](https://arxiv.org/abs/2404.10552)

-


---

[Generative Text Steganography with Large Language Model](https://arxiv.org/abs/2404.10229)

-

---

[EMC$^2$: Efficient MCMC Negative Sampling for Contrastive Learning with Global Convergence](https://arxiv.org/abs/2404.10575)


---

[Continual Offline Reinforcement Learning via Diffusion-based Dual Generative Replay](https://arxiv.org/abs/2404.10662)


---

[Question Difficulty Ranking for Multiple-Choice Reading Comprehension](https://arxiv.org/abs/2404.10704)


---

[Insight Gained from Migrating a Machine Learning Model to Intelligence Processing Units](https://arxiv.org/abs/2404.10730)


---

[MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/abs/2404.10774)


---

[LegalPro-BERT: Classification of Legal Provisions by fine-tuning BERT Large Language Model](https://arxiv.org/abs/2404.10097)


---

[Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/abs/2404.10719)


---

[Automating REST API Postman Test Cases Using LLM](https://arxiv.org/abs/2404.10678)

-


---

[Spiral of Silences: How is Large Language Model Killing Information Retrieval? -- A Case Study on Open Domain Question Answering](https://arxiv.org/abs/2404.10496)

-


---

[MEEL: Multi-Modal Event Evolution Learning]()

-


---


[Find The Gap: Knowledge Base Reasoning For Visual Question Answering](https://arxiv.org/abs/2404.10226)

-


---


#### 15th of April 2024


[Memory Sharing for Large Language Model based Agents](https://arxiv.org/abs/2404.09982)

- Memory-Sharing (MS)-framework: Multi LLM-agents share Memory Pool of query/response pairs, which improves In-Context Learning. Retriever-model is trained to retrieve memories based on user query.
- LLM agent answers based on query and retrieved memories. Scorer evaluates query / response. High scoring pairs are added to the Memory Pool, which is queried with cosine similarity.
- The shared memory helps all agents to learn from each other.
- The Retriever model is trained using pre-trained sentence similarity model, which retrieves data from jsonl-file to train a model and it is later used to pick relevant memories for each user query.


---

[Reimagining Self-Adaptation in the Age of Large Language Models](https://arxiv.org/abs/2404.09866)

- Self-Adaptive SW system: Includes Managed system (operational SW system) and Managing System (handles adaptions).
- Managing system includes Prompt generator, LLM engine, Response parser, Monitor (logs, metrics), Knowledge/Memory (conversation history, fine-tuned models, system config and system prompts) and Execute (verifier/executor). 


---

[Deferred NAM: Low-latency Top-K Context Injection via DeferredContext Encoding for Non-Streaming ASR](https://arxiv.org/abs/2404.10180)


---

[ChatShop: Interactive Information Seeking with Language Agents](https://arxiv.org/abs/2404.09911)


---

[TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table Decomposition](https://arxiv.org/abs/2404.10150)


---

[LLMorpheus: Mutation Testing using Large Language Models](https://arxiv.org/abs/2404.09952)

---

[A Survey on Deep Learning for Theorem Proving](https://arxiv.org/abs/2404.09939)


---

[Progressive Knowledge Graph Completion](https://arxiv.org/abs/2404.09897)


---

[Synergising Human-like Responses and Machine Intelligence for Planning in Disaster Response](https://arxiv.org/abs/2404.09877)


---

[HyperMono: A Monotonicity-aware Approach to Hyper-Relational Knowledge Representation](https://arxiv.org/abs/2404.09848)


---

[Action Model Learning with Guarantees](https://arxiv.org/abs/2404.09631)


---

[Explainable Generative AI (GenXAI): A Survey, Conceptualization, and Research Agenda](https://arxiv.org/abs/2404.09554)


---

[MyGO: Discrete Modality Information as Fine-Grained Tokens for Multi-modal Knowledge Graph Completion](https://arxiv.org/abs/2404.09468)


---

[Monte Carlo Search Algorithms Discovering Monte Carlo Tree Search Exploration Terms](https://arxiv.org/abs/2404.09304)


---

[Assessing Economic Viability: A Comparative Analysis of Total Cost of Ownership for Domain-Adapted Large Language Models versus State-of-the-art Counterparts in Chip Design Coding Assistance](https://arxiv.org/abs/2404.08850)


---

[Handling Reward Misspecification in the Presence of Expectation Mismatch](https://arxiv.org/abs/2404.08791)


---

[Generating Games via LLMs: An Investigation with Video Game Description Language](https://arxiv.org/abs/2404.08706)


---

[MMInA: Benchmarking Multihop Multimodal Internet Agents](https://arxiv.org/abs/2404.09992)


---

[Evolving Interpretable Visual Classifiers with Large Language Models](https://arxiv.org/abs/2404.09941)


---

[Evolving Interpretable Visual Classifiers with Large Language Models](https://arxiv.org/abs/2404.09941)


---

[Compression Represents Intelligence Linearly](https://arxiv.org/abs/2404.09937)


---

[Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection](https://arxiv.org/abs/2404.09894)

---

[Foundational Challenges in Assuring Alignment and Safety of Large Language Models](https://arxiv.org/abs/2404.09932)


---

[Is Table Retrieval a Solved Problem? Join-Aware Multi-Table Retrieval](https://arxiv.org/abs/2404.09889)


---

[Empowering Embodied Visual Tracking with Visual Foundation Models and Offline RL](https://arxiv.org/abs/2404.09857)


---

[Video2Game: Real-time, Interactive, Realistic and Browser-Compatible Environment from a Single Video](https://arxiv.org/abs/2404.09833)


---

[KG-CTG: Citation Generation through Knowledge Graph-guided Large Language Models](https://arxiv.org/abs/2404.09763)


---

[Effective Reinforcement Learning Based on Structural Information Principles](https://arxiv.org/abs/2404.09760)


---

[Unveiling Imitation Learning: Exploring the Impact of Data Falsity to Large Language Model](https://arxiv.org/abs/2404.09717)


---

[Higher Replay Ratio Empowers Sample-Efficient Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2404.09715)


---

[Are Large Language Models Reliable Argument Quality Annotators?](https://arxiv.org/abs/2404.09696)


---

[LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression for Large Language Models](https://arxiv.org/abs/2404.09695)


---

[Harnessing GPT-4V(ision) for Insurance: A Preliminary Exploration](https://arxiv.org/abs/2404.09690)


---

[Multi-News+: Cost-efficient Dataset Cleansing via LLM-based Data Annotation](https://arxiv.org/abs/2404.09682)


---


[All-in-one simulation-based inference](https://arxiv.org/abs/2404.09636)


---

[Efficient and accurate neural field reconstruction using resistive memory](https://arxiv.org/abs/2404.09613)


---

[A Self-feedback Knowledge Elicitation Approach for Chemical Reaction Predictions](https://arxiv.org/abs/2404.09606)


---

[Building Semantic Communication System via Molecules: An End-to-End Training Approach](https://arxiv.org/abs/2404.09595)


---

[σ-GPTs: A New Approach to Autoregressive Models](https://arxiv.org/abs/2404.09562)


---

[Characterization and Mitigation of Insufficiencies in Automated Driving Systems](https://arxiv.org/abs/2404.09557)


---

[Inferring Behavior-Specific Context Improves Zero-Shot Generalization in Reinforcement Learning](https://arxiv.org/abs/2404.09521)


---

[State Space Model for New-Generation Network Alternative to Transformers: A Survey](https://arxiv.org/abs/2404.09516)


---

[PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI](https://arxiv.org/abs/2404.09465)


---

[Exploring Text-to-Motion Generation with Human Preference](https://arxiv.org/abs/2404.09445)


---

[The 8th AI City Challenge](https://arxiv.org/abs/2404.09432)


---

[RankCLIP: Ranking-Consistent Language-Image Pretraining](https://arxiv.org/abs/2404.09387)


---

[Tasks People Prompt: A Taxonomy of LLM Downstream Tasks in Software Verification and Falsification Approaches](https://arxiv.org/abs/2404.09384)


---



#### 14th of April 2024


[Self-Selected Attention Span for Accelerating Large Language Model Inference](https://arxiv.org/abs/2404.09336)

- Fine-tunes LLM to self-identify minimal attention span in each step of the task.
- Speeds up inference 28% by dynamically adjusting self-attention.
- Allows LLMs to autonoumsly optimize computation.


---

[TransformerFAM: Feedback attention is working memory](https://arxiv.org/abs/2404.09173)

- Unlimited context window 


---

[Interactive Generative AI Agents for Satellite Networks through a Mixture of Experts Transmission](https://arxiv.org/abs/2404.09134)


---

[Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation](https://arxiv.org/abs/2404.09127)


---

[LLeMpower: Understanding Disparities in the Control and Access of Large Language Models](https://arxiv.org/abs/2404.09356)


---

[Towards Practical Tool Usage for Continually Learning LLMs](https://arxiv.org/abs/2404.09339)


---

[SNN4Agents: A Framework for Developing Energy-Efficient Embodied Spiking Neural Networks for Autonomous Agents](https://arxiv.org/abs/2404.09331)


---

[Text-to-Song: Towards Controllable Music Generation Incorporating Vocals and Accompaniment](https://arxiv.org/abs/2404.09313)


---

[TrafficVLM: A Controllable Visual Language Model for Traffic Video Captioning](https://arxiv.org/abs/2404.09275)


---

[Task-Driven Exploration: Decoupling and Inter-Task Feedback for Joint Moment Retrieval and Highlight Detection](https://arxiv.org/abs/2404.09263)


---

[Knowledgeable Agents by Offline Reinforcement Learning from Large Language Model Rollouts](https://arxiv.org/abs/2404.09248)


---

[Towards Fast Inference: Exploring and Improving Blockwise Parallel Drafts](https://arxiv.org/abs/2404.09221)


---

[TextHawk: Exploring Efficient Fine-Grained Perception of Multimodal Large Language Models](https://arxiv.org/abs/2404.09204)


---

[Prior-agnostic Multi-scale Contrastive Text-Audio Pre-training for Parallelized TTS Frontend Modeling](https://arxiv.org/abs/2404.09192)


---

[Survey on Embedding Models for Knowledge Graph and its Applications](https://arxiv.org/abs/2404.09167)


---

[GeMQuAD : Generating Multilingual Question Answering Datasets from Large Language Models using Few Shot Learning](https://arxiv.org/abs/2404.09163)


---

[Fusion-Mamba for Cross-modality Object Detection](https://arxiv.org/abs/2404.09146)


---

[ToNER: Type-oriented Named Entity Recognition with Generative Language Model](https://arxiv.org/abs/2404.09145)


---

[Provable Interactive Learning with Hindsight Instruction Feedback](https://arxiv.org/abs/2404.09123)


---

[Semantic In-Domain Product Identification for Search Queries](https://arxiv.org/abs/2404.09091)


---


#### 13th of April 2024

[LLMSat: A Large Language Model-Based Goal-Oriented Agent for Autonomous Space Exploration](https://arxiv.org/abs/2405.01392)

- LLMSat: LLM-based spacecraft control and space missions.


---


[When Hindsight is Not 20/20: Testing Limits on Reflective Thinking in Large Language Models](https://arxiv.org/abs/2404.09129)


["Don't forget to put the milk back!" Dataset for Enabling Embodied Agents to Detect Anomalous Situations](https://arxiv.org/abs/2404.08827)


---

[Do LLMs Play Dice? Exploring Probability Distribution Sampling in Large Language Models for Behavioral Simulation](https://arxiv.org/abs/2404.09043)


---

[Generative AI Agent for Next-Generation MIMO Design: Fundamentals, Challenges, and Vision](https://arxiv.org/abs/2404.08878)


---

[CuriousLLM: Elevating Multi-Document QA with Reasoning-Infused Knowledge Graph Prompting](https://arxiv.org/abs/2404.09077)


---

[CodeCloak: A Method for Evaluating and Mitigating Code Leakage by LLM Code Assistants](https://arxiv.org/abs/2404.09066)


---

[Exploring Explainability in Video Action Recognition](https://arxiv.org/abs/2404.09067)


---

[Adapting Mental Health Prediction Tasks for Cross-lingual Learning via Meta-Training and In-context Learning with Large Language Model](https://arxiv.org/abs/2404.09045)


---

[Navigating the Landscape of Large Language Models: A Comprehensive Review and Analysis of Paradigms and Fine-Tuning Strategies](https://arxiv.org/abs/2404.09022)


---

[Smart Help: Strategic Opponent Modeling for Proactive and Adaptive Robot Assistance in Households](https://arxiv.org/abs/2404.09001)


---

[Intuition-aware Mixture-of-Rank-1-Experts for Parameter Efficient Finetuning](https://arxiv.org/abs/2404.08985)


---

[Understanding Multimodal Deep Neural Networks: A Concept Selection View](https://arxiv.org/abs/2404.08964)


---

[EIVEN: Efficient Implicit Attribute Value Extraction using Multimodal LLM](https://arxiv.org/abs/2404.08886)


---

[An evaluation framework for synthetic data generation models](https://arxiv.org/abs/2404.08866)


---

[On Speculative Decoding for Multimodal Large Language Models](https://arxiv.org/abs/2404.08856)



#### 12th of April 2024


[Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length](https://arxiv.org/abs/2404.08801)

- Megalodon: Inlimited contrxt length


---

[Is Next Token Prediction Sufficient for GPT? Exploration on Code Logic Comprehension](https://arxiv.org/abs/2404.08885)

---

[Aligning LLMs for FL-free Program Repair](https://arxiv.org/abs/2404.08877)

---

[LLM In-Context Recall is Prompt Dependent](https://arxiv.org/abs/2404.08865)

---

[CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models](https://arxiv.org/abs/2404.08763)

---

[Leveraging Multi-AI Agents for Cross-Domain Knowledge Discovery](https://arxiv.org/abs/2404.08511)


---

[Augmenting Knowledge Graph Hierarchies Using Neural Transformers](https://arxiv.org/abs/2404.08020)


---

[Enhancing Autonomous Vehicle Training with Language Model Integration and Critical Scenario Generation](https://arxiv.org/abs/2404.08570)


---

[LLM Agents can Autonomously Exploit One-day Vulnerabilities](https://arxiv.org/abs/2404.08144)


---

[Memory Traces: Are Transformers Tulving Machines?](https://arxiv.org/abs/2404.08543)


---

[Study of Emotion Concept Formation by Integrating Vision, Physiology, and Word Information using Multilayered Multimodal Latent Dirichlet Allocation](https://arxiv.org/abs/2404.08295)


---

[Inverse Kinematics for Neuro-Robotic Grasping with Humanoid Embodied Agents](https://arxiv.org/abs/2404.08825)


---

[SQBC: Active Learning using LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions](https://arxiv.org/abs/2404.08078)


---

[Training a Vision Language Model as Smartphone Assistant](https://arxiv.org/abs/2404.08755)


---

[Apollonion: Profile-centric Dialog Agent](https://arxiv.org/abs/2404.08692)



---

[Strategic Interactions between Large Language Models-based Agents in Beauty Contests](https://arxiv.org/abs/2404.08492)


---

[Enhancing Autonomous Vehicle Training with Language Model Integration and Critical Scenario Generation](https://arxiv.org/abs/2404.08570)


---

[Toward a Theory of Tokenization in LLMs](https://arxiv.org/abs/2404.08335)

---

[Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions](https://arxiv.org/abs/2404.07214)


---


#### 11th of April 2024

[Rho-1: Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965)

- Rho-1: trains LLM with Selective Language Modelling (SLM) with useful tokens (based on loss pattern).
- The SLM calculates each token loss using reference model and then selectively removes loss of the unwanted tokens.
- Rho-1 1B and 7B achieve SOTA results at their size.


---

[Large Language Model Can Continue Evolving From Mistakes](https://arxiv.org/abs/2404.08707)

---

[Auctions with LLM Summaries](https://arxiv.org/abs/2404.08126)

---

[OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/abs/2404.07972)

- OSWorld: scalable multimodal agents for Ubuntu/Windows/MacOS to perform open-ended web/desktop tasks.
- Discovers humans complete 72% of tasks, while best agent completes only 12%. The main issues are GUI grounding/operational knowledge.

---


[ODA: Observation-Driven Agent for integrating LLMs and Knowledge Graphs](https://arxiv.org/abs/2404.07677)

- ODA: LLM with knowledge graph (KGs) using iteratively observation, action and reflection to help solve tasks. 
- The observation phase uses a global view of the entire KG and selectively picks relevant parts for reasoning.


---

[DesignQA: A Multimodal Benchmark for Evaluating Large Language Models' Understanding of Engineering Documentation](https://arxiv.org/abs/2404.07917)

- DesignQA-benchmark: Measures VLMs capcity to solve engineering tasks, including CAD images, drawings and engineering requirements. Includes: rule comprehension, rule compliance and rule extraction.


---

[Monte Carlo Tree Search with Boltzmann Exploration](https://arxiv.org/abs/2404.07732)

- Boltzmann Tree Search (BTS): replace soft values with Bellman values in MENTS.
- Decaying ENtropy Tree Search (DETS): Interpolates between BTS and MENTS.
- Alias method samples actions fast and demonstrate high performance in game of Go.

---

[WESE: Weak Exploration to Strong Exploitation for LLM Agents](https://arxiv.org/abs/2404.07456)


---

[Behavior Trees Enable Structured Programming of Language Model Agents](https://arxiv.org/abs/2404.07439)

---

[LLoCO: Learning Long Contexts Offline](https://arxiv.org/abs/2404.07979)

---

[ChatGPT Can Predict the Future when it Tells Stories Set in the Future About the Past](https://arxiv.org/abs/2404.07396)

---


#### 10th of April 2024 

[Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs](https://arxiv.org/abs/2404.07103)

--

[Accelerating Inference in Large Language Models with a Unified Layer Skipping Strategy](https://arxiv.org/abs/2404.06954)

---

[Superposition Prompting: Improving and Accelerating Retrieval-Augmented Generation](https://arxiv.org/abs/2404.06910)

---

[Not All Contexts Are Equal: Teaching LLMs Credibility-aware Generation](https://arxiv.org/abs/2404.06809)

---

[Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143)

- Infinite-Attention: Infinite long context window using compressed memory/local attention.
- The local attention computes using the in context. The compressed memory computes using the out-of-context.
- Google tests 1B LLN for 1M sequence length, which is difficult for such small model. I believe there are no existing benchmarks yet for testing such long context windows above +1M context window.
- Ahieves 114x compression ratio.


---

[GoEX: Perspectives and Designs Towards a Runtime for Autonomous LLM Applications](https://arxiv.org/abs/2404.06921)

- Gorilla Execution Engine (GoEx): open-source runtime to execute LLM actions, apps and microservices.
- LLMs evolve from dialogue to autonomous agents, which as well make decisions.
- "Post-facto Validation": human checks correctness of the generated output, instead of intermediate results. Introduces concet of "Undo" and "Damage confinement" to manage unintended risks with autonomous agents.


---

[Vision-Language Model-based Physical Reasoning for Robot Liquid Perception](https://arxiv.org/abs/2404.06904)


---

[BISCUIT: Scaffolding LLM-Generated Code with Ephemeral UIs in Computational Notebooks](https://arxiv.org/abs/2404.07387)

---


#### 9th of April 2024

[Measuring the Persuasiveness 
of Language Models](https://www.anthropic.com/news/measuring-model-persuasiveness)

- Reviews the scaling of LLMs on persuasion tasks. Finds, that Claude 3 Opus is statistically as convincing as human.


---

[Can Feedback Enhance Semantic Grounding in Large Vision-Language Models?](https://arxiv.org/abs/2404.06510)

---

[Large Language Models to the Rescue: Deadlock Resolution in Multi-Robot Systems](https://arxiv.org/abs/2404.06413)

- Hierarchical LLM guides robot away from deadlock situation by assigning leader-agent and give it direction to continue and GNN executes the low level policy.
- Finds LLMs effective in various environments for high-level planning tonresolve deadlocks.

---

[AgentQuest: A Modular Benchmark Framework to Measure Progress and Improve LLM Agents](https://arxiv.org/abs/2404.06411)

- AgentQuest: modular benchmark for multi-step reasoning with possibility via API to extend to different environments.
- Traditional benchmark includes single environment. AgentQuest uses driver to connect with a specific environment.


---

[AgentsCoDriver: Large Language Model Empowered Collaborative Driving with Lifelong Learning](https://arxiv.org/abs/2404.06345)

- AgentsCoDriver: multi-car collaboration using LLMs.
- The system includes the following modules: observation, reasoning engine, cognitive memory, reinforcement reflection, and communication.
- Includes useful designs on prompt generation and module designs.


---

[Autonomous Evaluation and Refinement of Digital Agents](https://arxiv.org/abs/2404.06474)

- Review domain-generic automatic evaluators to improve "digital agents", which improve SOTA performance in WebArena-benchmark by 29%.
- Evaluators are applied to improve agents with fine-tuning and inference-time guidance.
- Policy evaluation works by using VLM to perform user screen captioning, which is processed by LLM together with user instructions and agent trajectory(states/actions). The LLM-reasoner response is evaluated together with VLM-based reasoner to provide final failure/success-evaluation.
- Autonomous refinement uses inference-time guidance (reflexion) and Filtered behaviour cloning. 


---

[Wu's Method can Boost Symbolic AI to Rival Silver Medalists and AlphaGeometry to Outperform Gold Medalists at IMO Geometry](https://arxiv.org/abs/2404.06405)

- Combines Wu's method with AlphaGeometry to solve 27/30 IMO geometry problems (SOTA-level), which is 2 above AlphaGeometry alone or Wu's method alone only solves 15.
- First AI (fully symbolic baseline) to outperform a human in IMO geometry problems.


---

[Graph Reinforcement Learning for Combinatorial Optimization: A Survey and Unifying Perspective](https://arxiv.org/abs/2404.06492)



---

[Text-Based Reasoning About Vector Graphics](https://arxiv.org/abs/2404.06479)

---

[Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs](https://arxiv.org/abs/2404.07242)

---

[pfl-research: simulation framework for accelerating research in Private Federated Learning](https://arxiv.org/abs/2404.06430)


---

[MuPT: A Generative Symbolic Music Pretrained Transformer](https://arxiv.org/abs/2404.06393)


---

[VISION2UI: A Real-World Dataset with Layout for Code Generation from UI Designs](https://arxiv.org/abs/2404.06369)

---

[WESE: Weak Exploration to Strong Exploitation for LLM Agents](https://arxiv.org/abs/2404.07456)

---

[ActNetFormer: Transformer-ResNet Hybrid Method for Semi-Supervised Action Recognition in Videos](https://arxiv.org/abs/2404.06243)


---

[Elephants Never Forget: Memorization and Learning of Tabular Data in Large Language Models](https://arxiv.org/abs/2404.06209)



---

[Open-Source AI-based SE Tools: Opportunities and Challenges of Collaborative Software Learning](https://arxiv.org/abs/2404.06201)


---

[THOUGHTSCULPT: Reasoning with Intermediate Revision and Search](https://arxiv.org/abs/2404.05966)


[VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding?](https://arxiv.org/abs/2404.05955)




---


#### 8th of April 2024


[HAMMR: HierArchical MultiModal React agents for generic VQA](https://arxiv.org/abs/2404.05465)

- HAMMR: Uses multimodal ReAct-based agent, which is hierarchical by letting the agent call other specialized agents.
- Outperforms PaLI-X VQA by 5%.

---


[Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs](https://arxiv.org/abs/2404.05719)

- Ferret-UI: Outperforms GPT-4V on elementary UI-tasks with capability for referring (widget classification, OCR, icon recognition), grounding (find widget/icon/text and widget listing) and reasoning.
- "Any resolution" (anyres) enlarges small UI-objects in images like icons within varying screen aspect ratios. Screen capture is divided into two sub-sections. Each UI-element is referenced with type, text and bounding box. Uses 250k examples of training data. 


---

[AutoCodeRover: Autonomous Program Improvement](https://arxiv.org/abs/2404.05427)

- AutoCodeRover: autonomous sw engineering by solve Github issues (program repair and improvement). Solves 67 Github issues within 10 minutes. Future directions could include issue reproducer/semantic artifacts and human involvement.
- Includes two stages: context retrieval stage to produce buggy locations and Patch generation stage to produce final patch.


---

[Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405)

- Presents 12 insights on LLM training duration model architecture, quantization, sparsity and data signal-to-noise ratio.
- Finds junk data significantly reduces model capacity, which can be avoided to large extent by adding special token in the beginning of text. LLM learns to autonomously label data as high-quality.


---

[360°REA: Towards A Reusable Experience Accumulation with 360° Assessment for Multi-Agent System](https://arxiv.org/abs/2404.05569)


- Reusable Experience Accumulation with 360° Assessment (360°REA): a hierarchical multi-agent framework to evaluate and accumulate experience from feedback.
- Uses Deal-experience pool and 360◦ performance
assessment.
- Dual-experience pool: helps LLM-agents collect useful experiences in complex tasks using local experience/high-level experience.

---

[Finding Visual Task Vectors](https://arxiv.org/abs/2404.05729)

- Identifies Task Vectors.
- Uses task vectors to perform different tasks without any sample input.

---

[LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models](https://arxiv.org/abs/2404.05221)


---

[LLM-Augmented Retrieval: Enhancing Retrieval Models Through Language Models and Doc-Level Embedding](https://arxiv.org/abs/2404.05825)

---

[WILBUR: Adaptive In-Context Learning for Robust and Accurate Web Agents](https://arxiv.org/abs/2404.05902)

---

[Attention-Driven Multi-Agent Reinforcement Learning: Enhancing Decisions with Expertise-Informed Tasks](https://arxiv.org/abs/2404.05840)

---

[Long-horizon Locomotion and Manipulation on a Quadrupedal Robot with Large Language Models](https://arxiv.org/abs/2404.05291)

---

[Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models](https://arxiv.org/abs/2404.05567)

---

[Xiwu: A Basis Flexible and Learnable LLM for High Energy Physics](Xiwu: A Basis Flexible and Learnable LLM for High Energy Physics)


---

#### 7th of April 2024

[AI2Apps: A Visual IDE for Building LLM-based AI Agent Applications](https://arxiv.org/abs/2404.04902)



---

[LLM-Based Multi-Agent Systems for Software Engineering: Vision and the Road Ahead](https://arxiv.org/abs/2404.04834)



---

[StockGPT: A GenAI Model for Stock Prediction and Trading](https://arxiv.org/abs/2404.05101)


[Prompting Multi-Modal Tokens to Enhance End-to-End Autonomous Driving Imitation Learning with LLMs](https://arxiv.org/abs/2404.04869)

---

#### 6th of April 2024

[Self-organizing Multiagent Target Enclosing under Limited Information and Safety Guarantees](https://arxiv.org/abs/2404.04497)

---

[Challenges Faced by Large Language Models in Solving Multi-Agent Flocking](https://arxiv.org/abs/2404.04752)

---

[Transform then Explore: a Simple and Effective Technique for Exploratory Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/2404.04661)


---

[Autonomous Artificial Intelligence Agents for Clinical Decision Making in Oncology](https://arxiv.org/abs/2404.04667)

---

[Do We Really Need a Complex Agent System? Distill Embodied Agent into a Single Model](https://arxiv.org/abs/2404.04619)

---


[The Case for Developing a Foundation Model for Planning-like Tasks from Scratch](https://arxiv.org/abs/2404.04540)

---

[MACM: Utilizing a Multi-Agent System for Condition Mining in Solving Complex Mathematical Problems](https://arxiv.org/abs/2404.04735)


---

[Goal-guided Generative Prompt Injection Attack on Large Language Models](https://arxiv.org/abs/2404.07234)

---

#### 5th of April 2024


[Exploring Autonomous Agents through the Lens of Large Language Models: A Review](https://arxiv.org/abs/2404.04442)


---

[Increased LLM Vulnerabilities from Fine-tuning and Quantization](https://arxiv.org/abs/2404.04392)




---

[Cleared for Takeoff? Compositional & Conditional Reasoning may be the Achilles Heel to (Flight-Booking) Language Agents](https://arxiv.org/abs/2404.04237)

---

[ROMA-iQSS: An Objective Alignment Approach via State-Based Value Learning and ROund-Robin Multi-Agent Scheduling](https://arxiv.org/abs/2404.03984)

---

[Hypothesis Generation with Large Language Models](https://arxiv.org/abs/2404.04326)

---

[KGExplainer: Towards Exploring Connected Subgraph Explanations for Knowledge Graph Completion](https://arxiv.org/abs/2404.03893)



---


#### 4th of April 2024

[AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Web Navigating Agent](https://arxiv.org/abs/2404.03648)

- AutoWebGLM: automated browsing agent using ChatGLM3-6B LLM. Uses html simplification algorithm.
- Curriculum learning applies hybrid (human/AI) web browsing multi/single-step dataset(Data is collected with: match rules, Prompt LLM, Manual annotation and Solver and data is collected from real world/virtual environment and open source data.). RL/Rejection sampling fine tuning (RFT) is applied for browsing comphrehension and task decomposition.
- Introduces AutoWebBench-benchmark on real world web browsing tasks.
- Tools read DOM and webpage screenshot: Element filter, Element list, OCR module, HTML parse. Observation includes: instruction, HTML and previous action. Action includes: HTML section and action name.

---

[Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models](https://arxiv.org/abs/2404.03622)

- Visualization-ofThought

[Language Model Evolution: An Iterated Learning Perspective](https://arxiv.org/abs/2404.04286)


---

[Anticipate & Collab: Data-driven Task Anticipation and Knowledge-driven Planning for Human-robot Collaboration](https://arxiv.org/abs/2404.03587)

---

[CONFLARE: CONFormal LArge language model REtrieval](https://arxiv.org/abs/2404.04287)

---

[SELF-[IN]CORRECT: LLMs Struggle with Refining Self-Generated Responses](https://arxiv.org/abs/2404.04298)

---


[Reason from Fallacy: Enhancing Large Language Models' Logical Reasoning through Logical Fallacy Understanding](https://arxiv.org/abs/2404.04293)

---

[Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences](https://arxiv.org/abs/2404.03715)

---

[Comprehensible Artificial Intelligence on Knowledge Graphs: A survey](https://arxiv.org/abs/2404.03499)

---

[Benchmarking ChatGPT on Algorithmic Reasoning](https://arxiv.org/abs/2404.03441)

---

[Capabilities of Large Language Models in Control Engineering: A Benchmark Study on GPT-4, Claude 3 Opus, and Gemini 1.0 Ultra](https://arxiv.org/abs/2404.03647)

---


[ReFT: Representation Finetuning for Language Models](https://arxiv.org/abs/2404.03592)

---

[CodeEditorBench: Evaluating Code Editing Capability of Large Language Models](https://arxiv.org/abs/2404.03543)

---

[A Cause-Effect Look at Alleviating Hallucination of Knowledge-grounded Dialogue Generation](https://arxiv.org/abs/2404.03491)

---

[Can Small Language Models Help Large Language Models Reason Better?: LM-Guided Chain-of-Thought](https://arxiv.org/abs/2404.03414)

---

[Embodied Neuromorphic Artificial Intelligence for Robotics: Perspectives, Challenges, and Research Development Stack](https://arxiv.org/abs/2404.03325)

---

[RALL-E: Robust Codec Language Modeling with Chain-of-Thought Prompting for Text-to-Speech Synthesis](https://arxiv.org/abs/2404.03204)


---

#### 3rd of April 2024




[MIMIR: A Streamlined Platform for Personalized Agent Tuning in Domain Expertise](https://arxiv.org/abs/2404.04285)

---
[I-Design: Personalized LLM Interior Designer](https://arxiv.org/abs/2404.02838)
---
[On the Importance of Uncertainty in Decision-Making with Large Language Models](https://arxiv.org/abs/2404.02649)
---
[Learn to Disguise: Avoid Refusal Responses in LLM's Defense via a Multi-agent Attacker-Disguiser Game](https://arxiv.org/abs/2404.02532)
---
[Designing for Human-Agent Alignment: Understanding what humans want from their agents](https://arxiv.org/abs/2404.04289)


---
[PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](https://arxiv.org/abs/2404.02948)

---

[Testing the Effect of Code Documentation on Large Language Model Code Understanding](https://arxiv.org/abs/2404.03114)

---

[The RealHumanEval: Evaluating Large Language Models' Abilities to Support Programmers](https://arxiv.org/abs/2404.02806)

---

[Measuring Social Norms of Large Language Models](https://arxiv.org/abs/2404.02491)

---

[Exploring Backdoor Vulnerabilities of Chat Models](https://arxiv.org/abs/2404.02406)

---


#### 2th of April 2024


[Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](Mixture-of-Depths: Dynamically allocating compute in transformer-based language models)

- Mixture-of-Depth (MoD) Transformer: Transformers learn to assign compute dynamically to specific spots in the sequence.
- Top-k routing: defines tokens participating in block's computation. Learns to route harder tokens through more layers.
- Helps to speed up


---


[A Survey on Large Language Model-Based Game Agents](https://arxiv.org/abs/2404.02039)

- Survey about LLM-based Game agents.
- Unified architecture of LLMGAs: Perception(text, image, state, etc.), Thinking(reasoning, reflection, planning), Memory, Role-playing (role, experience, emotion), Action-module (control, dialogue, API, etc.) and Learning module.

 
---

[Advancing LLM Reasoning Generalists with Preference Trees](https://arxiv.org/abs/2404.02078)

- Eurus: LLMs optimized for reasoning. Trains reward model using UltraInteract-dataset, which consists of Preference Trees.
- Preference Tree: Diverse planning strategies in single pattern (such as tool creation, sequential processing). Multi-turn interaction trajectories with environment and the critique (learn to apply feedback and correct prior errors). Paired correct and incorrect actions in a tree structure. The data pair includes: instruction, correct response and incorrect response.   
- DPO (instruction fine-tuned) hurts performance, while KTO and NCA improve performance. Indicates, that DPO may be less suitable for reasoning tasks. 


---

[Self-Organized Agents: A LLM Multi-Agent Framework toward Ultra Large-Scale Code Generation and Optimization](https://arxiv.org/abs/2404.02183)

- SoA (Self-Organized multi-Agent framework): Self-organized LLMs collaborate to generate code base and dynamically multiple based on complexity. Uses Mother and Child-agents.
- Helps to scale the SoA to longer context lengths of code generation.

---


[Large Language Models for Orchestrating Bimanual Robots](https://arxiv.org/abs/2404.02018)

- LABOR (LAnguage-modelbased Bimanual ORchestration)-agent.

---
[CMAT: A Multi-Agent Collaboration Tuning Framework for Enhancing Small Language Models](https://arxiv.org/abs/2404.01663)

---
[InsightLens: Discovering and Exploring Insights from Conversational Contexts in Large-Language-Model-Powered Data Analysis](https://arxiv.org/abs/2404.01644)

---
[Helmsman of the Masses? Evaluate the Opinion Leadership of Large Language Models in the Werewolf Game](https://arxiv.org/abs/2404.01602)

---
[Collapse of Self-trained Language Models](https://arxiv.org/abs/2404.02305)

---

[RAT: Retrieval-Augmented Transformer for Click-Through Rate Prediction](https://arxiv.org/abs/2404.02249)

---

[Is Exploration All You Need? Effective Exploration Characteristics for Transfer in Reinforcement Learning](https://arxiv.org/abs/2404.02235)


---

#### 1st of April 2024

[Stream of Search (SoS): Learning to Search in Language](https://arxiv.org/abs/2404.03683)

- Stream of Search (SoS): Symbolic reasoning with next-sequence prediction (LLMs). 
- LLM pretrained with SoS-dataset generated with 500k search trajectories (also called as SoS) using various search strategies (BFS/DFS-based) to learn internal world model of search, which include problem solving using exploration and backtracking. 
- Enables generic and adaptive form of search: symbolic search is based on explicity environmental model, while SoS learns state transitions. The approach is likely to work in real world due to the complex/variable/branching nature of the game.
- The policy is improved using APA (Advantage-induces Policy Alignment)- and fine-tuning with [STaR-technique](#star) for threee iterations using 100k correct trajectories. 
- APA is a Actor-Critic RL technique. It creates copy of the LLM used as value network to enhance policy in the LLM. Reward function reviews the length and correctness of the generated trajectory.



---

[LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models](https://arxiv.org/abs/2404.01230)

- Survey about Strategic reasoning of LLMs: methodologies and metrics. These approaches are categorizied into: Prompt engineering, Modular enhancements, Theory of Mind and Fine-tuning.
- Reasoning tasks include: Common Sense reasoning, Mathematical reasoning, Symbolic reasoning, Causal reasoning and Strategic reasoning. 
- Strategic reasoning differs from being a more dynamic form of reasoning with the environment and due to the uncertainty of the adversary action.
- Key traits of strategic reasoning are: Goal-oriented, Interactive, Predictive nature and Adaptability.


---
[Large Language Model Evaluation Via Multi AI Agents: Preliminary results](https://arxiv.org/abs/2404.01023)

---
[]()

---
[]()


---

#### 31st of March 2024


---
[CHOPS: CHat with custOmer Profile Systems for Customer Service with LLMs](https://arxiv.org/abs/2404.01343)

---
[DiffAgent: Fast and Accurate Text-to-Image API Selection with Large Language Model](https://arxiv.org/abs/2404.01342)
---
[Algorithmic Collusion by Large Language Models](https://arxiv.org/abs/2404.00806)

---
["My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents]()
---
[](https://arxiv.org/abs/2404.00573)
---
[]()

---
[]()


---
[]()


---



#### 30th of March 2024

[Alignment of brain embeddings and artificial contextual embeddings in natural language points to common geometric patterns](https://www.nature.com/articles/s41467-024-46631-y)

- Aligns LLM word embeddings with human brain embeddings.
- Brain embeddings are generated from fine-grained spatiotemporal neural recordings in a continuous embedding space.
- Aligning is based on similar geometric shapes between brain and llm word embeddings.

[Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning](https://arxiv.org/abs/2404.00213)



---
[Language Models are Spacecraft Operators](https://arxiv.org/abs/2404.00413)


---
[A Taxonomy for Human-LLM Interaction Modes: An Initial Exploration](https://arxiv.org/abs/2404.00405)



---
[Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods](https://arxiv.org/abs/2404.00282)



---
[Your Co-Workers Matter: Evaluating Collaborative Capabilities of Language Models in Blocks World](https://arxiv.org/abs/2404.00246)


---

#### 29th of March 2024

[Gecko: Versatile Text Embeddings Distilled from Large Language Models](https://arxiv.org/abs/2403.20327)

- Gecko: "SOTA level" text embeddings with 768-dimensions with 7x smaller embedding model compared to prior SOTA. Gecko embeddings with 256 dimensions all existting 768-dimension text embeddings in MTEB
- Gecko uses FRet (Few-shot Prompted Retrieval dataset)-fine tuning dataset: task description, input query, positive passage, negative passage.
- FRet generates with LLM the relevant task and query for a passage. The query and task are fed into a pre-trained embedding model to get neighbor passages. LLM scores them either as positive or negative passages.
- Original passage may not become relevant positive/negative passage. 
- I think the overall idea could work even as prompt-engineering technique, where original passage is sent to LLM to define query/task, generate positive/negative passage and finally use the query, task, positive, negative passage as basis of retrieval. 

---

[ITCMA: A Generative Agent Based on a Computational Consciousness Structure](https://arxiv.org/abs/2403.20097)

- ITCMA (Internal Time-Consciousness Machine): an an architecture for generative agents called ITCMA-agent. It is"a computational consciousness structure" and good at utility and generalization to real world.
- ITCMA framework includes LLM, VLM, Agents under consciousness channels (composed of retention, primal impression and protention each next time step further) and Memory.
- Slowness is a downside.


---

[Enhancing the General Agent Capabilities of Low-Parameter LLMs through Tuning and Multi-Branch Reasoning](https://arxiv.org/abs/2403.19962)

- Explores open source 7B/13B LLMs ability to perform agentic tasks through supervised fine-tuning with task decomposition/backtracking (multipath reflective reasoning by prompting LLM to reflect path as not optiomal ) data.
- Agent dataset is contructed through: task construction, trajectory interaction and manual filtering. Includes two usage types: task planning and tool usage.
- Task planning data is generated the following way. LLM is used in three roles: question generator, action maker (offers thoughts/actions based on environmental feedback) and environmental agent. Action maker/Environmental agent keep interacting until task is completed. Requires manual screening after data is generated to ensure task logical consistency.
- Tool usage data is generated by manually filtering LLM examples of full reasoning trajectories.


---

#### 28th of March 2024


[STaR-GATE: Teaching Language Models to Ask Clarifying Questions](https://arxiv.org/abs/2403.19154)

- STaR(Self-Taught Reasoner)-GATE (Generative Active Task Elicitation)-algorithm: Self-improves LLM's ability to elicit user preference by generating questions and generalises beyond the trained role-player.
- Fine tunes LLM by generating a synthetic dataset for math problem dialogues with persona-task prompts.
- Teaches the LLM to ask clarifying questions to provide personalised responses.

---

[MATEval: A Multi-Agent Discussion Framework for Advancing Open-Ended Text Evaluation](https://arxiv.org/abs/2403.19305)

- MatEval: LLM agents emulate human collaboration discussion. Uses self-reflection, CoT and feedback mechnamism.
- Achieves high-correlation with human evaluation. Includes evaluator-, feedback(to imrpove discussion)- and summarizer-agents. 

---

[Change-Agent: Towards Interactive Comprehensive Change Interpretation and Analysis from Change Detection and Change Captioning](https://arxiv.org/abs/2403.19646)

- Change-Agent: Change deteection and interpretation using LLM from earth surface changes.


---

[Enhancing the General Agent Capabilities of Low-Parameter LLMs through Tuning and Multi-Branch Reasoning](https://arxiv.org/abs/2403.19962)

---

[Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis](https://arxiv.org/abs/2403.19646)



---
[LLMs as Academic Reading Companions: Extending HCI Through Synthetic Personae](https://arxiv.org/abs/2403.19506)


---
[MATEval: A Multi-Agent Discussion Framework for Advancing Open-Ended Text Evaluation](https://arxiv.org/abs/2403.19305)

---
[]()
---
[]()


---
[]()

---
[]()
---
[]()


---


#### 27th of March 2024

[Long-form factuality in large language models](https://arxiv.org/abs/2403.18802)

- Search-Augmented Factuality Evaluator (SAFE): long-form factual check with LLM agent using a 38 topic question set (LongFast). Uses multi-step reasoning and determines, if factuality is supported by google search results.
- LLM generates answer to question, this answer is splitted into individual facts. The facts are converted into self-contained, so the fact can be understood without rest of the facts. The individual facts are retrieved with google search: Facts supported by search results are labelled as supported and rest as non supported. If the fact is not relevant to the question, then the fact is labelled as irrelevant.
- Achieves super-human level performance and measures this with a F1-score. 


---

[What are human values, and how do we align AI to them?](https://arxiv.org/abs/2404.10636)



---

[Large Language Models Need Consultants for Reasoning: Becoming an Expert in a Complex Human System Through Behavior Simulation](https://arxiv.org/abs/2403.18230)

- MEOW (MOsaic Expert Observation Wall): improves LLM reasoning with behaviour simulation. 
- Expert model is trained with simulated data from experience of specific task. Tested in communication game.


---

[A Path Towards Legal Autonomy: An interoperable and explainable approach to extracting, transforming, loading and computing legal information using large language models, expert systems and Bayesian networks](https://arxiv.org/abs/2403.18537)

- Reviews the concept of legal autonomy of LLM agents for the first time: extracting, loading and transforming computing legal information.


---

[A Study of Three Influencer Archetypes for the Control of Opinion Spread in Time-Varying Social Networks](https://arxiv.org/abs/2403.18163)

- Reviews automated agents in social networks for opinion control: opinion inference engine with LLM, content generation using opinion vectors.


---
[]()
---
[]()



---

#### 26th of March 2024

[MAGIS: LLM-Based Multi-Agent Framework for GitHub Issue Resolution](https://arxiv.org/abs/2403.17927)

- MAGIS: Resolves Github issues with multi-agent LLMs: Manager, Repository Custodian, Developer and Quality Assurance engineer. 


---

[Depending on yourself when you should: Mentoring LLM with RL agents to become the master in cybersecurity games](https://arxiv.org/abs/2403.17674)

- SecurityBot: role-based multiagent collaborative framework with RL agent as mentors for LLM agent to support cybersecurity operations. Includes modules: profiles, memory, reflection and action using LLMs.
- Collaboration mechanism: cursor for dynamic suggestions taking, aggregator for multiple mentors suggestion ranking & caller for proactive suggestion asking.


---
[Large Language Models Need Consultants for Reasoning: Becoming an Expert in a Complex Human System Through Behavior Simulation](https://arxiv.org/abs/2403.18230)
---
[A Study of Three Influencer Archetypes for the Control of Opinion Spread in Time-Varying Social Networks](https://arxiv.org/abs/2403.18163)


---
[Depending on yourself when you should: Mentoring LLM with RL agents to become the master in cybersecurity games](https://arxiv.org/abs/2403.17674)
---
[OVER-NAV: Elevating Iterative Vision-and-Language Navigation with Open-Vocabulary Detection and StructurEd Representation](https://arxiv.org/abs/2403.17334)



---

[Compressed Federated Reinforcement Learning with a Generative Model](https://arxiv.org/abs/2404.10635)


---

[]()


---

#### 25th of March 2024

[AIOS: LLM Agent Operating System](https://arxiv.org/abs/2403.16971)

- AIOS-architecture ofr LLM agent OS: AIOS SDK, LLM Kernel (Kernel layer), OS Kernel, Agent applications (Application layer), HW layer.
- LLM kernel: Agent scheduler, Context manager, Memory manager, Storage manager, Tool manager and Access manager.


---

[RepairAgent: An Autonomous, LLM-Based Agent for Program Repair](https://arxiv.org/abs/2403.17134)

- RepairAgent: Automated program repair with LLMs with dynamically updated prompt format.


---

[CYGENT: A cybersecurity conversational agent with log summarization powered by GPT-3](https://arxiv.org/abs/2403.17160)

- CYGENT: Fine-tunes LLM for cybersecurity tasks and LLM agent provides/analyzes/summarizes user information from log files, detected events


---


[TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models](https://arxiv.org/abs/2403.17246)

- TwoStep: Combines classical planning with LLMs (Helper Plan and Main Plan).   




---
[Temporal and Semantic Evaluation Metrics for Foundation Models in Post-Hoc Analysis of Robotic Sub-tasks](https://arxiv.org/abs/2403.17238)
---
[Do LLM Agents Have Regret? A Case Study in Online Learning and Games](https://arxiv.org/abs/2403.16843)



---
[An LLM-Based Digital Twin for Optimizing Human-in-the Loop Systems](https://arxiv.org/abs/2403.16809)


---
[Harnessing the power of LLMs for normative reasoning in MASs](https://arxiv.org/abs/2403.16524)


---
[Norm Violation Detection in Multi-Agent Systems using Large Language Models: A Pilot Study](https://arxiv.org/abs/2403.16517)


---
[Towards Automatic Evaluation for LLMs' Clinical Capabilities: Metric, Data, and Algorithm](https://arxiv.org/abs/2403.16446)


---
[Re2LLM: Reflective Reinforcement Large Language Model for Session-based Recommendation](https://arxiv.org/abs/2403.16427)


---
[RL for Consistency Models: Faster Reward Guided Text-to-Image Generation](https://arxiv.org/abs/2404.03673)


---
[]()



---


#### 24th of March 2024




---
[AgentFL: Scaling LLM-based Fault Localization to Project-Level Context](https://arxiv.org/abs/2403.16362)
---
[Combining Fine-Tuning and LLM-based Agents for Intuitive Smart Contract Auditing with Justifications](https://arxiv.org/abs/2403.16073)


---
[]()

---
[]()

---
[]()

---
[]()

---
[]()

---

#### 23th of March 2024

[When LLM-based Code Generation Meets the Software Development Process](https://arxiv.org/abs/2403.15852)

- LCG: Multi-agent LLM consisting of waterfall, scrum and Test-Driven-Development sw development workflows with CoT and Self-refinement.
- LLM agent includes roles: requirements engineer, architect, developer, tester and scrum master. Uses same prompt, with role-identifier, role-specific instruction and task-information to drive dynamic prompting.


---
[Towards a RAG-based Summarization Agent for the Electron-Ion Collider](https://arxiv.org/abs/2403.15729)

---
[EduAgent: Generative Student Agents in Learning](https://arxiv.org/abs/2404.07963)


---
[]()

---
[]()



---

#### 22th of March 2024


[Can large language models explore in-context?](https://arxiv.org/abs/2403.15371)

- Reviews, if LLMs can explore effectively in-context, similar to Reinforcement learning-like agents.
- Suggest need for external summarization, larger models like GPT-4 and careful prompt engineering.

---

[CoLLEGe: Concept Embedding Generation for Large Language Models](https://arxiv.org/abs/2403.15362)

- CoLLEGe (Concept Learning with Language Embedding Generation): few-shot learning for new-concept acquisition and knowledge augmentation for LLMs.
- Generates concept embedding with CoLLEGe based on two example sentences, where the concept is used, creates a definition-sentence using this concept-embedding and asks LLM to generate the definition of the concept.  


---

[LLM-Driven Agents for Influencer Selection in Digital Advertising Campaigns](https://arxiv.org/abs/2403.15105)

- Influencer Dynamics Simulator (IDS): LLM-agent based influencer selection for digital ad campaigns.
- Includes: Influencer pre-selection, user profile generation, follower behaviour prediction and influencer tracking.


---

[Language Models in Dialogue: Conversational Maxims for Human-AI Interactions](https://arxiv.org/abs/2403.15115)

- Proposes principles for effective human-AI conversation: quantity, quality, relevance and manner, benevolence and transparency.


--- 

[CACA Agent: Capability Collaboration based AI Agent](https://arxiv.org/abs/2403.15137)

- CACA (Capability Collaboration based AI Agent): LLM agent with the following components: profile capability, reception capability, workflow capability, tool capability, tool service, methodology capability, add domain knowledge and planning capability.
- Processes: user request, generate plan, search methodology, get profile, discover tool, invoke service, add domain knowledge and register tool service.

---

[Content Knowledge Identification with Multi-Agent Large Language Models (LLMs)](https://arxiv.org/abs/2404.07960)

---


#### 21st of March 2024

[ReAct Meets ActRe: Autonomous Annotations of Agent Trajectories for Contrastive Self-Training](https://arxiv.org/abs/2403.14589)

- A^3T (Autonomous Annotation Agent Trajectories): Closed-loop self-improvement for LLM agents.
- Autonomous annotation of agent trajectories with ReAct for contrastive self-training. Reduces human-effort of data-collection.
- Agent reasons for actions taken (ActRe-prompting agent).Contrastive self-training uses rewards decisions made based on accumulated successful trajectoriess.
- The model outperforms GPT-4 and matches human average in Webshop-benchmark 




---

[ERD: A Framework for Improving LLM Reasoning for Cognitive Distortion Classification](https://arxiv.org/abs/2403.14255)

- ERD: Three step approach to reason cognitive distortions of user input: extraction, reasoning (CoT, Diagnosis of Thought) and debate between two LLM-agents and one LLM-judge.

---

[PeerGPT: Probing the Roles of LLM-based Peer Agents as Team Moderators and Participants in Children's Collaborative Learning](https://arxiv.org/abs/2403.14227)

- PeerGPT: pedagogical agents in Children collaborative learning with peer agent as team moderator or peer agent as a participant.


---

[RoleInteract: Evaluating the Social Interaction of Role-Playing Agents](https://arxiv.org/abs/2403.13679)

- RoleInteract-benchmark: Measures Sociality skills of role-playing LLM-agents. Conversation memory is one aspect to improve conversational agents. Complex group dynamics are still hard.


---

[Polaris: A Safety-focused LLM Constellation Architecture for Healthcare](https://arxiv.org/abs/2403.13313)

- Polaris: 1T parameter LLM as a co-operative agent for patient friendly conversation with multiple specialist agents like nurses/social workers/nutritionists. Uses iterative co-training to optmize diverse objectives. Uses healthcare-related data, including propietary data.
- Performs on par with human nurses and outperform significantly GPT-4. 


---


#### 20th of March 2024


[Reverse Training to Nurse the Reversal Curse](https://arxiv.org/abs/2403.13799)

- Reverse training: trains LLMs using reverse order to solve the reverse curse, where the LLM struggles to learn: B is a feature of A.
- Reverse curse has been key issue in the current LLM training.

---

[Large Language Models meet Network Slicing Management and Orchestration](https://arxiv.org/abs/2403.13721)

- LLM slices isolated virtual network of a Physical infrastructure. 



---

[Mapping LLM Security Landscapes: A Comprehensive Stakeholder Risk Assessment Proposal](https://arxiv.org/abs/2403.13309)

- Traditional risk assessment framework for LLMs through 10 categories: prompt injection, insecure plugin design, training data poisoning, model denial of service, supply chain vulnerabilities, sensitive information disclosure, insecure output handling, excessive agency, overreliance and model theft.



---


#### 19th of March 2024

[Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models](https://arxiv.org/abs/2403.12881)

- Agent-FLAN (Finetuned LANguage models for aents): finetuning for agentic tasks.
- Llama-2 7B model with Agent-FLAN surpasses by 3.5% existing SOTA models. Works both for tool utilization and agentic tasks.
- Observes: LLMs overfit to specific agentic task formats like JSON, Learning speed of LLMs vary for agentic tasks and current training methods introduce hallucinations.


---

[HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning](https://arxiv.org/abs/2403.12884)

- HYDRA (HYper Dynamic Reasoning Agent): multi-stage dynamic compositional visual reasoning, to make hyper-decisions (fast, strategic and efficient decisions).
- Three modules: LLM-Planner, RL agent (controller) and LLM-Reasoner (includes code generator and code executor). Includes Memory (code-, instruction- and feedback-history) and LLM-Textualizer (Uses template to create summary).
- Planner and Reasoner generate instructions/Code with LLM. RL agent interacts with these modules and makes high-level decisions from best instructions based history. HYDRA adjusts actions from feedback received in reasoning. User queries are deconstructed with three sub-questions processed concurrently. The code executor has access to vision foundational models like BLIP, XVLM and GLIP.
- RL agent is based on DQN-algorithm.


---

[Characteristic AI Agents via Large Language Models](https://arxiv.org/abs/2403.12368)

- Characteristics AI: simulates real-life individuals in different situations. Releases Character100-dataset.
  

---


[Embodied LLM Agents Learn to Cooperate in Organized Teams](https://arxiv.org/abs/2403.12482)

- Introduces prompt-based orgnizational structure. Reduces LLM errors related to redundant information and complying any instruction. Includesc communication- and action phases. Criticize-Reflect architecture.


---

[Contextual Moral Value Alignment Through Context-Based Aggregation](https://arxiv.org/abs/2403.12805)

- CMVA-GS: moral value agents with different profiles pass through contextual aggregator.

---

[LLMs-based Few-Shot Disease Predictions using EHR: A Novel Approach Combining Predictive Agent Reasoning and Critical Agent Instruction](https://arxiv.org/abs/2403.15464)


---

[The Use of Generative Search Engines for Knowledge Work and Complex Tasks](https://arxiv.org/abs/2404.04268)

---


#### 18th of March 2024

[Multimodal Human-Autonomous Agents Interaction Using Pre-Trained Language and Visual Foundation Models](https://arxiv.org/abs/2403.12273)

- Dual-modality frameworkk: leverages independent LLM/VLM/SR models in order to interact autonomous robots.
- Includes components of visual understanding, LLM and Speech regognition.


---

[EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents](https://arxiv.org/abs/2403.12014)

- EnvGen-framework: Use LLM-agent creates training environment for reasoning, so smaller embodied RL-agents improve their weak skills.
- Benefits from the LLM-agents world knowledge and the small, yet capable RL agents.


---

[From Pixels to Insights: A Survey on Automatic Chart Understanding in the Era of Large Foundation Models](https://arxiv.org/abs/2403.12027)

- Chart understanding task (chart Q&A, captioning, fact-checking, -to-table conversion, factual error correction).


---

[Agent3D-Zero: An Agent for Zero-shot 3D Understanding](https://arxiv.org/abs/2403.11835)

- Agent3D-Zero: 3D scene understanding agent with VLM by selecting and analyzing series of viewpoints for 3D understanding. 


---

#### 17th of March 2024

[Logic Query of Thoughts: Guiding Large Language Models to Answer Complex Logic Queries with Knowledge Graphs](https://arxiv.org/abs/2404.04264)


---


#### 15th of March 2024

[DiPaCo: Distributed Path Composition](https://arxiv.org/abs/2403.10616)

- DiPaCo (DIstributed PAth COmposition): a modlular ML paradigm, where computing is distributed by path. Path refers to sequence of modules defining input-output function.
- Paths are small in relation to the overall model. During both training and deployment, a query is routed to replica of a path (sparsely activated), not the entire model.
- The training phase distributes computation by paths through set of shared modules. The inference phase computes single path.
- First large-scale, more modular and less synchronous learning, when FLOPs are relatively cheap and communication is relatively expensive.
- Exceeds 1B parameter dense Transformer by choosing 256 possible paths with size of 150 million parameters.


---

[PERL: Parameter Efficient Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2403.10704)

- PERL (Parameter Efficient Reinforcement Learning): Compares reward modelling training and RL using LoRA against traditional RLHF. The study focuses on device UI control, such as sending email.
- PERL achieves similar level of performance with less training compute and less memory used.
- Releases self-dialogue: Taskmaster Coffee and Ticketing-datasets and still pending, but planned release of UI automation-dataset called "S-dataset". Unclear, if the NPOV-dataset apart is kept internal. 


---

[AUTONODE: A Neuro-Graphic Self-Learnable Engine for Cognitive GUI Automation](https://arxiv.org/abs/2403.10171)

- AUTONODE (Autonomous User-Interface Transformation through Online Neuro-graphic Operations and Deep Exploration).
- Integrates Dora (Discovery and mapping Opertion for graph Retrieval Agents).


---

[Enhancing Human-Centered Dynamic Scene Understanding via Multiple LLMs Collaborated Reasoning](https://arxiv.org/abs/2403.10107)

- V-HOU Multi-LLMs Collaborated Reasoning: video scene understanding.


---

[Can a GPT4-Powered AI Agent Be a Good Enough Performance Attribution Analyst?](https://arxiv.org/abs/2403.10482)

- LLM agent for performance attrition using CoT and Plan and Solve (PS).

---

[ChatPattern: Layout Pattern Customization via Natural Language](https://arxiv.org/abs/2403.15434)


---

[ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference](https://arxiv.org/abs/2404.07947)

---


#### 14th of March 2024


[Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)

- Quiet-Star: Extension and generalization of STaR-paper. Improves significantly LLM performance on GSM8K-benchmark.
- Uses "meta-tokens" at the start/end of each thought, to learn when to generate a rationale and when it should make prediction-based on that rationale.


---

[Enhancing Trust in Autonomous Agents: An Architecture for Accountability and Explainability through Blockchain and Large Language Models](https://arxiv.org/abs/2403.09567)

- Blockchain based Autonomous agent not only with explanation, but as well with record auditable interpretation.
- Components: Autonomous agent, blockchain, Non-expert users, Automatic evaluation, Explainability component and Asynchronous task.


---

[VisionGPT-3D: A Generalized Multimodal Agent for Enhanced 3D Vision Understanding](https://arxiv.org/abs/2403.09530)

- Vision-GPT-3D: Multimodal agent optimizing 3d vision understanding by integrating: YOLO-, SAM- and DINO-models.  
- Starts by making a depth map from multiple images, converts the depth map into point cloud, then into mesh and finally into a video.


---

[From Skepticism to Acceptance: Simulating the Attitude Dynamics Toward Fake News](https://arxiv.org/abs/2403.09498)

- Fake news Propagation Simulation (FPS)-framework: identifies LLMs usefulness of LLMs to combat fake news. Reviews trends and controls of fake news using multiple agents under different personas (age/name/education/personality traits) with both long/short-term memory and self-reflection. Early and frequent regulation of fake news helps to limit its propagation impact.
- Dynamic Opinion Agent (DOA) simulates cognitive processes of each agent. Agent Interaction Simulator (AIS) defines how/which agents interact daily and publishes new common knowledge/beliefs to agents. 


---

[LLM-based agents for automating the enhancement of user story quality: An early report](https://arxiv.org/abs/2403.09442)

- ALAS (Autonomous LLM-based Agent System): LLM-based system between different agent profiles to develop and maintain high-quality IT user stories.
- Agent profiles: Product Owner/Requirements Engineer. User story. Task preparation phase: task, sub-tasks, context and vision statement. Task conduction-phase.


---

[USimAgent: Large Language Models for Simulating Search Users](https://arxiv.org/abs/2403.09142)

- USimAgent: generates search interaction sequence through multiple rounds, taking into account context generated in prior rounds, each with steps: reasoning/action, query generation and click behaviour. 


---

[MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)

- MM1: MLLM training.

---


#### 13th of March 2024

[Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295)

---

[Scaling Instructable Agents Across Many
Simulated Worlds](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-generalist-ai-agent-for-3d-virtual-environments/Scaling%20Instructable%20Agents%20Across%20Many%20Simulated%20Worlds.pdf)

- SIMA: The Scalable, Instructable, Multiworld Agent based on image from the screen and text instruction provided by user. SIMA agent uses text encoder, image encoder and video encoder to process the input image and text and output only the embodied action.
- Real-tme, embodied agent generalizes in 3D environment to any human task and coordinated by natural language instructions. Agent trained on multiple games outperformed an agent trained on single game. Performs nearly as well in new unseen game environments.
- Data collection from commercial video game environments, Training of SIMA Agent model with text instruction-actions and human evaluation. 


---

[SOTOPIA-π: Interactive Learning of Socially Intelligent Language Agents](https://arxiv.org/abs/2403.08715)

-  SOTOPIA-π: LLMs with social intelligence engage, act safer and persuade more.
-  Achieves social interaction goal completion capability of GPT-4 using 7B LLM. 
-  Starts by generating social tasks with each character with its own social goal. Continues by collecting this training data using behavioural cloning (expert signal) and self-reinforcement(strongly performing signals from itself). Improve the agent policy with the LLM ratings. Generate SOTOPIA tasks with characters and evaluate their interaction with LLM rating and human rating.  


---

[AutoGuide: Automated Generation and Selection of State-Aware Guidelines for Large Language Model Agents](https://arxiv.org/abs/2403.08978)

- AutoGuide: the LLM-agent receives task-information, in-context examples, current trajectory and "state-aware guidelines"-retrieval.
- The "State-aware retrieval" is in short a navigational instruction of the specific section in the web-page, such as clicking the "Forum"-button leads to page, where you can create a new Forum.


---

[TINA: Think, Interaction, and Action Framework for Zero-Shot Vision Language Navigation](https://arxiv.org/abs/2403.08833)

- TINA (Thinking, Interacting and Action)-framework: a zero-shot Vision-Language Navigation (VLN) based LLM-agent, visual perceptor making observations and a memory.
- Agent inputs include: Task description, Instuction and Memory. Trajectory memorizer summarizes observations/actions to memory. 



---

[System for systematic literature review using multiple AI agents: Concept and an empirical evaluation](https://arxiv.org/abs/2403.08399)

- Systematic Literature Reviews (SLRs)-agent: planner, literature identification, data extraction, data compilation, performance validation. The code includes concrete prompts used with each step.


---

[Hierarchical Auto-Organizing System for Open-Ended Multi-Agent Navigation](https://arxiv.org/abs/2403.08282)

- HAS (Hierarchical Auto-organizing System): Auto-organizes LLM-agents to complete navigation tasks using dynamic maps and auto-organizing-mechanism.
- Centralized planning (planner, describer, critic and deployer) with global multi-modal memory, distributed execution (actor, curriculum, critic and skill) with local-multi-modal memory and multimodal information (vision, audio, object and map) with environment state.


---

[Cultural evolution in populations of Large Language Models](https://arxiv.org/abs/2403.08882)

- Models cultural evolution in LLM-agent population.  


---

[CleanAgent: Automating Data Standardization with LLM-based Agents](https://arxiv.org/abs/2403.08291)

- CleanAgent: a data preparation LLM agent. 


---


#### 12th of March 2024

[NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning](https://arxiv.org/abs/2403.07376)

- NavCoT (Navigational CoT): LLM acts as a world model and a navigational reasoning agent.
- LLM is prompted to forecast the navigational NavCoT: 1. act as world model to imagine the next observation based on instruction, 2. select best aligned candidate observation fitting to the imagination, 3. determine action based on reasoning from prior steps.
- In the Future Imagination-step (FI), the LLM is prompted to imagine the next observation, such as seeing a Patio. Visual Information Filter (VIF) selects from the available options provided by the VLM (image and description of the action towards it), the best matching to the FI. Action Prediction (AP)-step generates action prediction based on the selected option.


---

[WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?](https://arxiv.org/abs/2403.07718)

- Introduces two benchmarks WorkArena- and BrowserGym--benchmarks to evaluate LLM-agent interacting with software via browser.
- WorkArena (list, form, knowledge base, service catalog, menus) includes 23k tasks to interact with ServiceNow.
- BrowserGym designs and evaluates web agents in Python environment, which includes html content, raw pixels and acccessibility tree. and  
- Illustrates clear difference in web browsing expertise between GPT-3.5 vs. GPT-4.


---

[Transforming Competition into Collaboration: The Revolutionary Role of Multi-Agent Systems and Language Models in Modern Organizations](https://arxiv.org/abs/2403.07769)

- Multiagent Data and AI based platform framework: data, playground, web app, embedding model, multiagent orchestration (rest of the components interact with), data security/privacy, APIs/plugins, LLM & cache, Cloud provider, cloud DBs, Data Ops, MLOps, LLMOps and data strategy/ethics/LLM governance. The paper offers very little apart from this list, but the list does include quiet many of the components.


---

[DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation](https://arxiv.org/abs/2403.07788)

- DexCap: a hand motion data capture system.


---

[AesopAgent: Agent-driven Evolutionary System on Story-to-Video Production](https://arxiv.org/abs/2403.07952)

- Aesop-agent: Multimodal content generation agent.
- Includes RAG from database(expert experience/professional knowledge), script generation, image generation, video assembly, utility layer.
- Reviews prompt optimization.


---

#### 11th of March 2024

[RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems](https://arxiv.org/abs/2403.06465)

- RecAI: Recommender systems based on LLMs, where user makes query, the LLM agent makes tool queries to get the correct items.
- Includes Profile memory, info query, item retrieval and item ranker.
- The LLM chain includes: init state, dynamic demo, plan  execute and reflection.
- Refers to planning called Plan-First method, which creates comprehensive execution plan and then strictly follows this plan. The planning input includes: user input, context, tool descriptions and demonstrations for in-context learning to create tool utilization plan.


---

[DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation](https://arxiv.org/abs/2403.06845)

- DriveDreamer-2: First world model to generate customized driving videos, including uncommon scenes. 
- LLM generates user-defined driving videos: LLM converts user request into agent based trajectories, which is used to generate HDMap (python script creates Bird Eye View (BEV)) with respecting traffic rules. Unified Multi-View Model (UniMVM) improve temporal and spatial coherence of the generated video.


---

[Academically intelligent LLMs are not necessarily socially intelligent](https://arxiv.org/abs/2403.06591)

- SESI (Situational Evaluation of Social Intelligence)-benchmark: Superficial friendliness is principal reason for errors.
- Reviews: Empathy, Social-cognition, self-presentation, influence and concern.
- Illustrates interesting insight about GPT-4 not being better in this benchmark than GPT-3.5 turbo and Mistral model outperforming Llama 2.


---

#### 10th of March 2024

[TRAD: Enhancing LLM Agents with Step-Wise Thought Retrieval and Aligned Decision](https://arxiv.org/abs/2403.06221)

- TRAD: Thought Retrieval Aligned Decision.
- Includes three sub-processes: Temporal Expansion, Relative Order Mark and History Alignment.


---

[ArgMed-Agents: Explainable Clinical Decision Reasoning with Large Language Models via Argumentation Schemes](https://arxiv.org/abs/2403.06294)

- ArgMed-agent: Generator of the Argumentation Schema (AS), Verifier of the AS and Reasoner as symbolic solver.


---

[Reframe Anything: LLM Agent for Open World Video Reframing](https://arxiv.org/abs/2403.06070)

- RAVA (Reframe Any Video Agen): Perception to interpret user query and video content, Planning to determine aspect ratio/reframin strategies and Execution uses video editing tools to produce final video. 


---

#### 9th of March 2024

[Cached Model-as-a-Resource: Provisioning Large Language Model Agents for Edge Intelligence in Space-air-ground Integrated Networks](https://arxiv.org/abs/2403.05826)

- Model caching optimization on edge devices. Age of Thought (AoT): to measure the relevance/coherence of intermediate thoughts
during CoT inference.


---

#### 8th of March 2024


[RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation](https://arxiv.org/abs/2403.05313)

- Retrieval Augmented Thoughts (RAT): Iterative revising CoTs with retrieval information, which improves LLM reasoning in long-horizon tasks and reduces hallucinations.
- First generates CoT answer, then uses this answers with a verification prompt. The verification prompt requests to verify correctness of the given answer to the question with the separately added information query, for example by using Bing/Google search (authors implement a separate get_content function in their Github code).
- The query is based on the draft answer. The retrieved information is used to revise the draft answer. The next thought is then appended and a new round of revision performed. The process is repeated, until all revised thoughts are obtained and the final answer is provided.
- The github code includes multiple functions to manage inputs and outputs for the LLMs.


---

[FLAP: Flow Adhering Planning with Constrained Decoding in LLMs](https://arxiv.org/abs/2403.05766)

- FLAP (Flow Adhering Planning): Static planning in task oriented dialogs using constrained decoding algorithm based on lookahead heuristics.
- The research is static planning, but the authors plan a follow up research with dynamic planning.
- Aligns suggested plan thoughts using three scale score regards: user intent alignment, permitted flow steps, API selected, API permitted and structrally correct.


---

[Will GPT-4 Run DOOM?](https://arxiv.org/abs/2403.05468)

- Doom-game agent, consisting Python-based Manager module connected to Doom code and three modules: Planner, Vision and Agent.
- Vision module (GPT-4V) receives screenshots from the Managers and provides text description of it. - Planner uses as input the walkthrough and history and outputs a granular plan to be executed. Uses k-level of experts.


---


#### 7th of March 2024

[Acceleron: A Tool to Accelerate Research Ideation](https://arxiv.org/abs/2403.04382)

- Acceleron: LLM agent for research using colleague and mentor personas. Interacts with researcher develop research proposal.
- Introduces concept of "Unanswerability", when LLM should identify when all the retrieved paragraphs are irrelevant.


---


#### 6th of March 2024

[PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion](https://arxiv.org/abs/2403.03788)

- PowerPoint Task Completion-Robustness (PPTC-R)-benchmark for LLMs PowerPoint completion tasks.


---

[SheetAgent: A Generalist Agent for Spreadsheet Reasoning and Manipulation via Large Language Models](https://arxiv.org/abs/2403.03636)

- SheetAgent: LLM-agent to complete spreadsheet tasks by interacting through iterative task reasoning. Introduces SheetRM-benchmark.
- Includes three modules: Planner (generates python code to modify the spreadsheet), Informer (produces SQLs to perceive the spreadsheet despite dynamic range) and Retriever (retrieves instructive examples to improve robustness).
- Includes interesting concept of erroneous code-code repository as Milvus vector database, in order to perform cosine similarity search in case erroneous code.


---

[Exploring LLM-based Agents for Root Cause Analysis](https://arxiv.org/abs/2403.04123)

- Introduces LLM-based Root-Cause-Analysis (RCA) agent based on ReCT.


---


#### 5th of March 2024



[Cradle: Empowering Foundation Agents Towards General Computer Control](https://arxiv.org/abs/2403.03186v3)

- Cradle-framework: introduces MLLM-agent to control GUI using screenshot inputs and outputs executable code to control keyboard/mouse actions(key or button to press/where/duration/speed/location to move). Introduces the term General Computer Control (GCC).
- Includes modules: information gathering/self-reflection/task inference/skill curator/action planning/memory(episodic for retaining information/procedural for skills).
- Uses PyDirectInput instead of pyautogui for keyboard control. Includes low-level wrapper, which uses ctypes in windows and AppleScript in Mac to communicate low-level mouse controls.
- Procedural memory is based on topk matches of the skills (text embeddings).
- Episodic memory consists of short-term (screenshots/task guidance actions/reasoningand long-term summary. Short-term memory includes forgetting factor k set to 5-interactions. 
- The long-term memory includes recurrent information summary to avoid losing track of long-horozon task objective while inside short-horizon task: ongoing task/the past entities met/past behaviours.

---

[Reaching Consensus in Cooperative Multi-Agent Reinforcement Learning with Goal Imagination](https://arxiv.org/abs/2403.03172)

- MAGI (Multi-Agent Goal Imagination)-framework: agents reach consensus (and cooperatively reaching valuable future states) through imagined common goal.
- Future states are modeled with CVAE-based self-supervised generative modelling. Samples a common goal with high-potential value for multi-agent consensus to guide policies of all agents.
- CVAE is self-supervised conditional variational auto-encoder to model the distribution of future states.

---

[Language Guided Exploration for RL Agents in Text Environments](https://arxiv.org/abs/2403.03141)

- Introduces Language Guided Exploration (LGE), which in this study outperforms Behaviour Cloning.
- Explorer: RL agent with LGE outperforms with wide margin behaviour cloning. The key component is the Guide-model (LLM), which provides world knowledge to introduce set of feasible actions and reducing substantially the possible action space.


---

[KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents](https://arxiv.org/abs/2403.03101)

- KnowAgent: LLM-agent to improve planning with explicit action knowledge retrieval. The agent includes Action Knowledge Base (AKB), Planning Path Generation(question, action path, thought and observation) and Kowledgable Self-Learning.
- Introduces term planning hallucinations, which refers to agent generating conflicting or unnecessary action sequences.
- AKB contains information to steer action generation process: action name, definition, rule and knowledge.
- Knowledgable Self-Learning phase improves continuously the understanding and usage of action knowledge


---

[Learning to Use Tools via Cooperative and Interactive Agents](https://arxiv.org/abs/2403.03031)

- ConAgents: Cooperative and interactive agents, which iteratively applies three modules: Grounding, Execution and Observation. 
- Grounding step grounds user query into too definition and target output. Executing defines required tool arguments and completes returned output. Observing addresses long-form data outputs with IterCal-method: LLM agent self-adapts to feedback from tool environment.
- IterCal-method uses a pseudo-schema, which is basically a simplifie human-readable dictionary of the lengthy output returned from the tool used, see the pseudo-schema in the last page of the paper for quick understanding. 


---

[OPEx: A Component-Wise Analysis of LLM-Centric Agents in Embodied Instruction Following](https://arxiv.org/abs/2403.03017)

- OPEx-agent: Includes Observer, Planner and Executor-roles. Observer-agent processes and interprets sensory inputs, such as vision from the environment. Planner integrates dynamically strategic plans and sub-tasks based on perception. Excutor implements the plans with skills library.
- Embodied Instruction Following (EIF): agents follows task instruction by interacting with the environment through observations in a ego-centric way.
- The agent basically includes, what objects the agent is currently observing, what objects have been found, what observations have been so far made and what previous steps have been completed. In addition, there is known the current objective, thought and action.


---

[Android in the Zoo: Chain-of-Action-Thought for GUI Agents](https://arxiv.org/abs/2403.02713)

- Chain-of-Action-Thought (dubbed CoAT): a novel prompting strategy to allow GUI agents to perceive, reason and decide.
- CoAT includes four parts: Screen context, Action thinking, Action target and Action Result.
- Screen context explains content of the GUI screenshot. Action thinking takes user query, current screen and history to define possible actions to complete goal. Action target refers to GUI element being actioned such as clicking an icon. Action result maps current screen with next action to future observation. 


---

[InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents](https://arxiv.org/abs/2403.02691)

- InjectAgent-benchmark with +1k test cases in 17 tools and 62 attacker tools. Illustrates. Attack Success Rate (ASR) remains high especially in open source models like Llama 2.
- This result is surprising, considering "open source" models are often categorized as safer options over closed models. 


---

[Entropy-Regularized Token-Level Policy Optimization for Large Language Models](https://arxiv.org/abs/2402.06700)

- Entropy-Regularized Token-level Policy Optimization (ETPO).


---

[ChatCite: LLM Agent with Human Workflow Guidance for Comparative Literature Summary](https://arxiv.org/abs/2403.02574)


- ChatCite: Literature summary LLM-agent. Includes Key-Element Extractor and Reflective Incremental Generator.
- Key-Element Extractor: Extracts research questions, methodology, results, conclusions, contributions, innovations and limitations. These are stored in memory.
- Reflective Incremental Generator: Reflective mechnanism, Comparative summarizer, Reflective Evaluator and Rank & Select. Iteratively repeated.


---

#### 4th of March 2024

[Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents](https://arxiv.org/abs/2403.02502)

- Exploration-based Trajectory Optimization (ETO): LLM agent collects failure trajectories to update its policy using failure-success trajectories.
- ETO includes three steps: Explore (SFT-based behavioral cloning LLM agent), Collect Failures (pairs contrastive trajectories from the failures and expert trajectories) and Optimize trajectories (DPO loss on the pairs).


---


#### 2nd of March 2024

[AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks](https://arxiv.org/abs/2403.04783v1)

- AutoDefence: Introduces multi-agent LLM-jailbreaking prevention framework with input agent, defence agent and output agents.
- Defence agent includes prompt analyser agent, intention analyser agent, judge agent and coordinator agent.
- Reduces success rate of prompt attacks.


---

[SceneCraft: An LLM Agent for Synthesizing 3D Scene as Blender Code](https://arxiv.org/abs/2403.01248)

- SceneCraft: LLM agent converts text into Python code for Blender API 3D-scenes. 
- Dual-loop: Inner loop keeps improving scene by writing Blender code, Blender API renders the code and critic-revising this rendered image using Vision-Language Model (VLM).
- Outer loop learns by updating reusable functions to the library.
- The beaty of this approach is, that VLM model revising the end result, makes it very generich approach for self-improvement.


---

#### 1st of March 2024

[Playing NetHack with LLMs: Potential & Limitations as Zero-Shot Agents](https://arxiv.org/abs/2403.00690)

- NetPlay: zero-shot agent, which uses agent loop using GPT-4.
- Constructs prompt including past events, the current observation, a task description with available skills and the desired output format. Retrieve new skill and Execute it. New events are then observed.


---

#### 28th of February 2024

[Human Simulacra: A Step toward the Personification of Large Language Models](https://arxiv.org/abs/2402.18180)

- Creates LLM personification with complete life story to simulate personality and interacting with external world in human-like manner
- Uses multi-agent framework to simulate cognitive functions, memory and psychology-guided evaluation to asses the quality of the human simulation with self-reporting and external observations. 


---

[Prospect Personalized Recommendation on Large Language Model-based Agent Platform](https://arxiv.org/abs/2402.18240)

-  Rec4Agentverse: Recommender agent with three steps: User-Agent Interaction, Agent-Recommender, Agents Collaboration.


---


[Data Interpreter: An LLM Agent For Data Science](https://arxiv.org/abs/2402.18679)

- Data Interpreter: Data scientist LLM agent with Plan, Code and Verify steps. The pipeline is represented as a DAG-structure. 
- Plan Real data adaption using dynamic planning with hierarchical graph structures. Code: Dynamic tool integration to improve code execution. Verify: Logical inconsistency identification through feedback


---


#### 24th of February 2024

[ByteComposer: a Human-like Melody Composition Method based on Language Model Agent](https://arxiv.org/abs/2402.17785)

- ByteComposer: LLM-agent based melody composer with four elements: Conception analysis, Draft composition, Self-evaluation and modification and Aesthetic selection. 


---

#### 23th of February 2024

[Large Multimodal Agents: A Survey](https://arxiv.org/abs/2402.15116)

- Survey on multi-modal AI and LLM agents.


---

[Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)

- Genie: a Foundational World Model. The learning paradigm is unsupervised learning from unlabelled internet video.  The approach scales effectively as compute is increased.
- Includes: Latent Action Model (LAM) for latent action between each video frame in each timestep, 2. Video tokenizer to convert video frames into discrete tokens, 3. Dynamics model to predict next frame 
- The model/datasets are not released, but the approach is explained in the paper with single GPU implementation details by bringing your own data using the dataset creationg instructions provided. 


---

#### 21st of February 2024

[Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/abs/2402.14083)

-  Searchformer: Transformer model outperforms A* search algorithm in planning.
-  Two step approach, where Transformer excels large action spaces and learns heuristics (strategies to guide search) from the training with the data.
- First step generates synthetic dataset: Imitate A* search by using A* search and recording compute and and optimal plan as text token sequences(task description, search tree dynamics, and final plan) with length of thousands of tokens. This dataset includes search dynamics of A* search itself. Train a Transformer model (Searchformer) to generate the text token sequences with optimal plan for a given task. This leads to a transformer model, which has the A* search coded in the model weights.
- Second step further trains Searchformer using Expert Iteration, which attempts to generate optimal plans to tasks with less steps in the optimal plan. The resulting model solves Sokoban puzzles with 27% less search steps, than A* search algorithm. The idea is to generalize the Transformer model into more generic search beyond A* search.


---

[User-LLM: Efficient LLM Contextualization with User Embeddings](https://arxiv.org/abs/2402.13598)

- User-LLM: generates user embeddings from user data with multi-feature autoregressive transformer and then fine-tunes the LLM using these embeddings with cross-attention.
- The method enables inserting the LLM with long-term user history through compressed user embeddings and short term user context through input prompt.
- Effective approach for LLM personalization and user modelling. Includes good chapter on LLM long context research.


---

[∞Bench: Extending Long Context Evaluation Beyond 100K Tokens](https://arxiv.org/abs/2402.13718)

- Coins prompting technique called: "Context recalling": improves code debug accuracy from +16% (using CoT) to +40% (using context recalling).
- Context recalling prompts the model to first recall the relevant information, before doing further reasoning.
- Introduces long context bencmark: ∞BENCH-benchmark for LLMs with above 100k context window. 


---

[Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent](https://arxiv.org/abs/2402.13717)

- Neeko-agent: Multi-character roleplaying agent with LoRA.
- Includes Pretraining, Multi-character Role-Playing and Incremental Role-Playing with Fusion and Expansion stages.


---


#### 20th of February 2024

[MuLan: Multimodal-LLM Agent for Progressive Multi-Object Diffusion](https://arxiv.org/abs/2402.12741)

- MuLan: Multimodal LLM agent, addresses text2image generation errors through progressive multiobject generation with LLM-based planning and VLM-based feedback control.
- MuLan is training free method.


---

[Large Language Model-based Human-Agent Collaboration for Complex Task Solving](https://arxiv.org/abs/2402.12914)

- ReHAC: uman-agent(LLM) collaboration with RL policy model.


---

#### 19th of February 2024

[AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](https://arxiv.org/abs/2402.12226)

- AnyGPT: Any-to-Any Multimodal Language Model with any input output between text, speech, image and music.
- Uses only data preprocessing with modality specific tokenizers to tokenize input into discrete tokens and model outputs by de-tokenizing into specific modality outputs.
- Introduces multimodal alignment dataset made of conversations.   


---

[Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents](https://arxiv.org/abs/2402.12327)

- Studies spontaneuous collaboration between competing LLM agents


---

[WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment](https://arxiv.org/abs/2402.12275)

- WorldCoder: LLM agent learns World Models (world_model.py) using Python program from interactions with its environment.
- Outperforms baselines from DeepRL- and ReAct-agents in gridworlds-environment.
- Incldues sample code of the world_model.py.


---

[Comprehensive Cognitive LLM Agent for Smartphone GUI Automation](https://arxiv.org/abs/2402.11941)

- CoCo-Agent: GUI control with VLM/LLM/CLIP, which includes Comprehensive Environment Perception (CEP) and Conditional Action Prediction (CAP). Includes information such as GUI screenshot, GUI layout information, user objective and action history.
- Offers SOTA-level performance on GUIs, yet high training cost.  


---

[LLM Agents for Psychology: A Study on Gamified Assessments](https://arxiv.org/abs/2402.12326)

- PsychoGAT: Gamification of psychological assessment traditionally performed with questionaries with superior performance. Includes prompt templates.  


---

[Structured Chain-of-Thought Prompting for Few-Shot Generation of Content-Grounded QA Conversations](https://arxiv.org/abs/2402.11770)

- Structured CoT (SCoT): breakdowns into states for for generating actions for each sub-tasks durign the specific state. 
- For example first state determines, if question is answerable, the next step identifies required steps for the answer and the next state generates the step answer. 


---

#### 18th of February 2024

[LongAgent: Scaling Language Models to 128k Context through Multi-Agent Collaboration](https://arxiv.org/abs/2402.11550)

- LongAgent: Scales LLaMA to 128k context window outperforming GPT-4 through multiagent collaboration using inter-member communication.
- Leader agent selects agent members of team based on task description, agent team collaboratively reason, deduct answer and finally resolve conflict to generate final answer. 


---

[Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents](https://arxiv.org/abs/2402.11651)

- Fine-tuning LLMs with Negative examples enhances performance. 


---

[Modelling Political Coalition Negotiations Using LLM-based Agents](https://arxiv.org/abs/2402.11712)

- Political coalition negotiation with LLM agents.


---

#### 17th of February 2024

[LLM can Achieve Self-Regulation via Hyperparameter Aware Generation](https://arxiv.org/abs/2402.11251)

- Hyperparameter Aware Generation (HAG): the LLM learns to modify automatically its hyperparameters (temperature, top_p, top_k, repetition_penalty) for each user task input.
- Self-regulation of hyperparameters enables the LLM to finetune its responses to different task inputs.
- Self-regulation takes inspiration from the ability of human body to regulate itself based on different factors like temperature, blood pressure, adrealine etc.


---

#### 16th of February 2024

[Robust agents learn causal world models](https://arxiv.org/abs/2402.10877)

- Implies causal understanding is required for robust generalization.
- Causal models can be learned from adaptive agents.


---

#### 15th of February 2024

[Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200)

- CoT-Decoding: CoT without prompting. LLMs inherently pose reasoning abilities.
- Uses top-k alternative tokens to uncover CoT paths, which are frequently paths discovered in CoT. 


---

[A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727)

- ReadAgent: very long context management through gist-memories and pagination for web browsing.
- ReadAgent: LLM decided what content to store as episode pagination, LLM compresses page memory as shorter gist memory (see fuzzy-trace theory about memory) and LLM decides the pages to look up per given task and the gist memories related to the context of the task. The agent then retrieves the related page information to complete the task.
- Extends effective context window by 3-20x and keeps failure rate close to 0%, which is significantly less than traversing tree with a MemWalker-like solution.
- Gist-memory improves Web navigation over using raw html inputs, which is by nature a very long context task.


---

[AI Hospital: Benchmarking Large Language Models in a Multi-agent Medical Interaction Simulator](https://arxiv.org/abs/2402.09742)

- AI Hospital: LLM acts with doctor, patient, examiner and physician-roles. Categorises medical information into: subjective, objective and Diagnosis/Treatment. 
- MVME-benchmark (Multi-View Medical Evaluation): evaluates LLMs in symptop collection, recommendation analysis and diagnosis.


---

#### 14th of February 2024

[AgentLens: Visual Analysis for Agent Behaviors in LLM-based Autonomous Systems](https://arxiv.org/abs/2402.08995)

- AgentLens: visual analysis of of LLM based autonomous agents and exploration of their behaviours.
- UI includesOutline view, Agent view and Monitor view. Summarizes raw events, Descriptions of generated behaviours, Behaviour embeddings, Timeline segmentation.
- The behavioural embeddings: enables plotting specific behaviours in time, which is very effective approach. 


---

[Towards better Human-Agent Alignment: Assessing Task Utility in LLM-Powered Applications](https://arxiv.org/abs/2402.09015)

- AgentEval: framework to verify utility of the LLM tool through automatic criteria creation for a given task to review meeting of user needs. 
- Includes CriticAgent to list criteria of accepted values and QuantifierAgent verifying suggested criteria.


---

[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

- Next generation LoRA. Get more out from your LLM, while not directly related to agents.


---


#### 13th of February 2024


[GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements](https://arxiv.org/abs/2402.10963)

- GLoRe: Presents a Stepwise Outcome-based Reward models. SORM is in contrat to Outcome-Based Reward models (ORMs) and Process-Based Rewrd Model (PRMs), where trained only on synthetic data to approximate future reward of optimal policy V*.
- Uses three step refinement training process: 1. Fine-tune base model for Student policy model, 2. SORM training, 3. Refinement training.

---

[Grounding LLMs For Robot Task Planning Using Closed-loop State Feedback](https://arxiv.org/abs/2402.08546)

- Brain-Body LLM(BB-LLM): Brain-LLM defines high-level plans for robot. The BodyLLM converts them into low-level planned actions as robot commands. 


---

[Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast](https://arxiv.org/abs/2402.08567)

- Agent Smith: "Infectious Jailbraking" Technique, which infects single LLM agent, that then infects with exponential growth rate the remaining agents.
- Concering technique reminding traditional computer virus, because the computational/time/resource expenses of infecting single agent remain low, but includes capability of infecting rest of the agents.


---

[Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs](https://arxiv.org/abs/2402.08189)

- Investigation on LLMs capability to simulate human strategic behaviour.
- Compares Multiagent vs. Single LLM agent performance in the Ultimatum game and finds multiagent system more accurately simulating human behaviour.


---

[Large Language Models as Minecraft Agents](https://arxiv.org/abs/2402.08392)

- Develops Minecraft Builder and Architect LLM agents using JSON-format with capacity to ask clarifying questions from the LLM.


---

[PRompt Optimization in Multi-Step Tasks (PROMST): Integrating Human Feedback and Preference Alignment](https://arxiv.org/abs/2402.08702)

- PROMST: Optimizes prompts. Includes TaskLLM and PromptLLM. PromptLLM generates new prompt suggestions from existing best prompts and their feedbacks. New candidates are selected by score prediction model. 


---


#### 12th of February 2024

[T-RAG: Lessons from the LLM Trenches](https://arxiv.org/abs/2402.07483)


---


[OS-Copilot: Towards Generalist Computer Agents with Self-Improvement](https://arxiv.org/abs/2402.07456)

- FRIDAY: Self-improving embodied agent to interact with OS.
- OS-Copilot framework: Planner, Configurator to update or retrieve (Declarative memory for user profile and Semantic knowledge/Procedural memory for tools), Actor (Executor / Critic).
- Learns to control and self-improve.


---

[Predictive representations: building blocks of intelligence](https://arxiv.org/abs/2402.06590)

- Successor Representation (SR) may function as versatile building blocks of intelligence.


---

[Secret Collusion Among Generative AI Agents](https://arxiv.org/abs/2402.07510)

- Model capability evaluation framework on Secret collusion.


---


[THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation](https://arxiv.org/abs/2402.08191)

- THE COLOSSEUM benchmark for robot manipulation generalization through 20 diverse tasks.


---

#### 11th of February 2024

[Self-Correcting Self-Consuming Loops for Generative Model Training](https://arxiv.org/abs/2402.07087)

- Self-Correcting Functions using expert knowledge for generative model training. 


---
 
#### 9th of February 2024

<div id="vstar"> </div>  

--- 

[V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457)

- V-STaR: Enhancement to STaR-method. Uses during self-improvement not only correct, but as well incorrect solutions generated to train a verifier using DPO, where is judged correctness of the model-generated solutions.
- Iterating V-STaR multiple rounds generates progressively better reasoners and stronger verifiers by increasing GSM8K performance significantly from base STaR-method.
- Addresses the aspect of data efficiency by being able to improve both from correct and incorrect solutions. 


---

[Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training](https://arxiv.org/abs/2309.17179)

- TS-LLM: a tree search guided LLM decoding with learned value function applicable for reasoning tasks.

---

[Feedback Loops With Language Models Drive In-Context Reward Hacking](https://arxiv.org/abs/2402.06627)

- LLMs interacting with the real-world create feedback loops, where the LLMs outputs shape world state, from where next LLMs are trained.
- Such feedback loops can cause In-Context Reward Hacking (ICRH): LLM outputs increase BOTH the objective and the negative side-effects.
- Output-refinement and policy refinement lead to ICRH.


---

[Understanding the Weakness of Large Language Model Agents within a Complex Android Environment](https://arxiv.org/abs/2402.06596)

- AndroidArena benchmark for measuring LLMs capability to control a modern operating system.
- Main failure modes: understanding, reasoning, exploration, and reflection.
  

---

<div id="llmsurveymikolov"> </div>  

[Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)

- Reviews past years LLM research: LLM model families, building of LLMs, using of LLMs, LLM datasets, LLM metrics and future directions and challenges.
- Includes deployment pipelines, vector databases, prompting pipelines and LLM training/inference frameworks


---

[Why Solving Multi-agent Path Finding with Large Language Model has not Succeeded Yet](https://arxiv.org/abs/2401.03630)

- Identifies three reasons on why multi-agent path finding with LLMs does not work: model limitation, lack of understanding and lack of reasoning.


---

#### 8th of February 2024

<div id="interactiveagent"> </div>  


[An Interactive Agent Foundation Model](https://arxiv.org/abs/2402.05929)

- Interactive Agent Foundational Model: A generalist agent. Multi-task, Multi-domain: Healthcare, Gaming AI and Robotics.
- Interactive Agent framework: action encoder, visual encoder and language encoder. Pretrained to predict masked unified tokens for the three modalities: text token, visual token and action/agent token from each separate token per input type. Effectively generalizes between domains.
- Defines term "Agent-based AI" as generating dynamic behaviours grounded on the context understanding of uncertain environment. Defines "Embodied Agent-paradigm principles": Perception, Planning and Interaction.
Agent actions impact directly task plans by not requiring environment feedback to plan next action.
- MUltimodal systems preteained cross-modality grounded with environment hallucinate less by being grounded with the physical/virtual environment and require less size, than models pretrained separately/without grounding.


---

[UFO: A UI-Focused Agent for Windows OS Interaction](https://arxiv.org/abs/2402.07939)

- UI-Focused (UFO) agent: Automatically controlling Windows OS. The system includes two VLM-based agents: AppAgent (Application Selection Agent) and ActAgent (Action Selection Agent).
- AppAgent uses User input, Desktop screenshot, App information, Examples and Memory. It chooses application to complete the task, generates global plan. AppAgent outputs observation, Thoughts, Selected App, Status, Global pla and Comment.
- ActAgent takes as input  User request, Screenshots (highlighted last action, clean, annotated), Control information, Examples and Memory. ActAgent pursues local plans and actions until meeting the goal / receives observations from apps / interacts with memory. Outputs observation, Thoughts, Labeled control operation, Function, Status, Local plan and Comment.
- Control Interaction module grounds actions.


--- 

[Real-World Robot Applications of Foundation Models: A Review](https://arxiv.org/abs/2402.05741)

- A literature review of Robotics Foundationa models.
- Reviews Input/Ourput relationships of models, perception, motion planning and control.

---

[TimeArena: Shaping Efficient Multitasking Language Agents in a Time-Aware Simulation](https://arxiv.org/abs/2402.05733)

- TimeArena: A textual simulation environment for LLM agents to complete tasks as soon as possible.
- 30 real world like tasks from household activities to laboratory work. Illustrates, that GPT-4 lacks temporal awareness such as failing to recognize opportunities in parallel processing.


---

[ScreenAgent: A Vision Language Model-driven Computer Control Agent](https://arxiv.org/abs/2402.07945)

- VLM to control a real computer screen/GUI.
- Includes Planning, Acting and Reflecting phases.


---

[In-Context Principle Learning from Mistakes](https://arxiv.org/abs/2402.05403)

- Learning Principles (LEAP): Intentially guide LLM to make mistakes on few examples to reflect on them and learn task-specific principles.
- Improves MATH reasoning capability. 


---

[Keyframer: Empowering Animation Design using Large Language Models](https://arxiv.org/abs/2402.06071)

- Keyframer: LLM-powered animation generator from SVG images.


---

[Discovering Temporally-Aware Reinforcement Learning Algorithms](https://arxiv.org/abs/2402.05828)

- Reviews Temporally-aware reinforcement learning and Meta-learning.


---

[WebLINX: Real-World Website Navigation with Multi-Turn Dialogue](https://arxiv.org/abs/2402.05930)

- WebLINX: Real-time webpage control with LLMs.
- Filters relevant web page elements


---

[How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis](https://arxiv.org/abs/2402.05863)

- NegotionArena bencbmark: to measure LLMs ability to negotiate. 


---

[Decision Theory-Guided Deep Reinforcement Learning for Fast Learning](https://arxiv.org/abs/2402.06023)

- Decision Theory-guided Deep Reinforcement Learning (DT-guided DRL): addresses cold start problem in RL.
- Promotes more structural and informed exploration strategy.


---


#### 7th of February 2024

[The Future of Cognitive Strategy-enhanced Persuasive Dialogue Agents: New Perspectives and Trends](https://arxiv.org/abs/2402.04631)

- CogAgent: Persuasion LLM agent framework.
- Cognitive strategy mining, Cognitive Strategy Prediction for Dialogue Modelling and Application scenarios (bargaining, counselling, debating etc.)


---

[Can Large Language Model Agents Simulate Human Trust Behaviors?](https://arxiv.org/abs/2402.04559)

- Reviews LLM agents ability to simulate Trust. 


---

[ScreenAI: A Vision-Language Model for UI and Infographics Understanding](https://arxiv.org/abs/2402.04615)

- ScreenAI: a VLM. Screen user interfaces (UIs) understanding, dataset creation with LLMs.


---

#### 6th of February 2024


[Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620)

- Self-Discover: Self-discovers complex reasoning structures outperforming CoT-Self-Consistency in MATH, while being more compute efficient. 
- Select reasoning modules(for exampel CoT, etc), Adapt reasoning modules and Implement reasoning structures as key-value pair as json. 
- Works with multiple LLMs and different types of reasoning scenarios.
 

---

[AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls](https://arxiv.org/abs/2402.04253)

- AnyTool: LLM agent utilizing over 16k APIs.
- API retriever with hierarchical structure with meta-agent, user query solver using candidate APIs and self-reflection mechanism for initial impractical solutions. Uses GPT-4 with function calling. 
- Introduces AnyToolBench-benchmark.
- Meta-agent is linked with multiple category agents each managing collection of tool agents.


---

[Can Generative Agents Predict Emotion?](https://arxiv.org/abs/2402.04232)

- Reviews LLM agents capability to align humans in terms of emotional states, when new events take place.
- LLM agent framework, where time series text memories are stored in graph database, which are summarized. As new events take place, the norm of the past episodic memories is combined with the current context. LLM agents emotional state is measured using pre-existing Positive And Negative Affect Schedule (PANAS)-framework to arrive a PANAS score of the current emotional state. Finally, the new memory is added to the graph database.
- The LLM agent acts in a virtual town with multiple agents interacting for example inviting and assisting a party. Performance is reviewed using pre-existing EmotionBench-benchmark. LLM agents lack to some extent ability to align emotionally like humans.
- Raises interesting concern, that GPT-3.5 may be biased to provide positive answers and therefore struggle to illustrate negative emotions.


---

[S-Agents: self-organizing agents in open-ended environment](https://arxiv.org/abs/2402.04578)

- S-Agents: Tree-of-Agents, where the leader LLM agent leads tree-like structure wiith executor agents.
- Hourglass agent framework: Monitor progress and Hierarchical planning. 
- Monitor progresss: starts with previous plan and perception used to monitor progress against objective. 
- Hierarchical planning: plans long-term (task planner), takes current task and generates actions (action planner) in the environment and agents.


---

[Large Language Models as an Indirect Reasoner: Contrapositive and Contradiction for Automated Reasoning](https://arxiv.org/abs/2402.03667)

- Indirect Reasoning (IR): Uses logic of contrapositives and contradictions for factual reasoning and math proofs.
- Adding IR to factual reasoning increases overall accuracy compared to Direct Reasoning (DR) only or IR only. 


---

[MobileVLM V2: Faster and Stronger Baseline for Vision Language Model](https://arxiv.org/abs/2402.03766)

- Vision Language Model: MobileVLM V2.


---


[QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model](https://arxiv.org/abs/2402.03755)

- QuantAgent: Includes two LLM agents: Writer and Judge. The Writer-agent retrieves Knowledge Base (KB) and then generates answer based on the KB and submits the answer to real environment for evaluation. The Judge-agent retrieves relevant KB related to the review and it then generates score and feedback used in the next iteration.
- The iteration continues until maximum number of steps is reached or the score is high enough.


---

[Beyond Lines and Circles: Unveiling the Geometric Reasoning Gap in Large Language Models](https://arxiv.org/abs/2402.03877)

- Improves LLMs geometric reasoning with self-correction, collaboration and role specialization using geometric tools and four LLM agents.
- Uses LLM agents with four roles: Natural language solver and validator, Geometric tool Solver and Validator.


---

[In-context learning agents are asymmetric belief updaters](https://arxiv.org/abs/2402.03969)

- In-context learning: framing of the problem significantly impacts succesfullness.
- LLMs learn better from better-than-expected outcomes rather than worse-than-expected outcomes. 


---

[Systematic Biases in LLM Simulations of Debates](https://arxiv.org/abs/2402.04049)

- Reviews LLMs capability to generate believable simulation and current LLMs include a simulation bias for political debate. 
- Self-fine tunes LLM to take a specific political stance by using politically-oriented question to reflect answers, which is more effective than prompt-profiling alone.
- Illustrates the difficulty for LLMs to simulate specific human behaviour like a political views.


---

[Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science](https://arxiv.org/abs/2402.04247)

- Takes safety research from LLM safety to LLM agent safety, which is more holistic view.
- Scientific agent: Reviews LLM agent vulnerabilities within science domain: Data Insuffiency, Planning limitation, Tool limitations, LLM limitations and Lack of measurement. 
- Introduces triangle framework: Human regulation (Intent), Agent alignment (Red teaming) and Agent regulation (environmental feedback). 


---

#### 5th of February 2024

[Understanding the planning of LLM agents: A survey](https://arxiv.org/abs/2402.02716)

- LLM-Agent planning: provides a systematic view of LLM-based agents planning, covering recent works aiming to improve planning ability.
- It categorizes existing works into Task Decomposition, Plan Selection, External Module, Reflection and Memory, and provides comprehensive analysis for each direction.
- This survey is the first work that comprehensively analyzes LLM-based agents from the planning abilities.


---


[Chain-of-Feedback: Mitigating the Effects of Inconsistency in Responses](https://arxiv.org/abs/2402.02648)

- Recursive Chain-of-Feedback (R-CoF): Recursively breaks down complex reasoning problems into more easier and more detailed solutions and re-adjusts original reasoning based on the detailed correct reasoning.
- Given a problem, asks LLM to generate answer using multiple reasoning steps, then LLM verifies the incorrect reasoning steps, LLM then recursively asks only to solve the incorrect reasoning steps using same approach. If the new answer is correct, it gets added to the higher level answer and otherwise repeats the recursive LLM call.


---

[Vision-Language Models Provide Promptable Representations for Reinforcement Learning](https://arxiv.org/abs/2402.02651)

-  Promptable Representations for Reinforcement Learning (PR2L): the model asks from VLM about the game tasks, such as in case a spider is visiblle. The VLM responds semantic features or knowledge, which then better help the system to advance in the game by connecting what is seen with what it needs to do. This ensures, that the system actions are grounded with the reality of what is going on in the game. 
-  Initializes RL policy using VLM representation.
-  PR2L was not trained to play Minecraft only, but it still plays at level closed to models specifically trained with Minecraft games.


---

[Guiding Language Model Math Reasoning with Planning Tokens](https://arxiv.org/abs/2310.05707)

- Planning tokens improve LLM reasoning capabilities.
- Add the planning tokens in the LLM generated answer based on CoT in the beginning of each reasoning step, such as planning token related to multiplying done on that reasoning step,


---

[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

- DeepSeekMath: 7B model comparable with math reasoning of a 70B model, close to Gemini Ultra and GPT-4.
- Introduces Group Relative Policy Optimization (GRPO).


---

[LLM Agents in Interaction: Measuring Personality Consistency and Linguistic Alignment in Interacting Populations of Large Language Models](https://arxiv.org/pdf/2402.02896.pdf)

- Studies LLM agents capability to follow human personality profiles: analytical vs. creative personality.
- Each profile demonstrates different levels of consistency towards its profile in writing style and in a personality test. 


---

[Graph-enhanced Large Language Models in Asynchronous Plan Reasoning](https://arxiv.org/abs/2402.02805)

- Plan Like a Graph (PLaG): asynchronous plan reasoning with LLM: generates time estimations, identify step dependencies, converts the time estimates and dependencies into a graph processor and finally generate answer.
- Creates AsyncHow-benchmark: for asynchronous plan reasoning, requiring ability to correctly add time, correctly comparing time durations and ability to solve constrained reasoning.
- LLMs struggle efficiently completing complex asyncchronous plans without detailed illustration of how to solve the task.


---

[C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models](https://arxiv.org/abs/2402.03181)

---

#### 4th of February 2024


[Understanding the planning of LLM agents: A survey](https://arxiv.org/abs/2402.02716)

- Review studies about the LLM agents planning capabilities.
- Categorizes these planning capabilities into: Task decomposition, Plan selection, External module, Reflection and Memory.
- Identifies development areas in: evaluating efficiency of the planning, revisiting of planning strategies in multimodality and more realistic evaluations.


---

[Solution-oriented Agent-based Models Generation with Verifier-assisted Iterative In-context Learning](https://arxiv.org/abs/2402.02388)

- SAGE: Modelling and Solving stages with Automatic Design and Generation of ABM.
  

---

[LLM-Enhanced Data Management](https://arxiv.org/abs/2402.02643)

- LLMDB: Detailed data management framework with LLMs.
- Components include: Preparation, Request pre-processing, Request parsing, Pipeline executor agent, Vector database and Data/Model management.


---

[Collaborative Agents for Software Engineering](https://arxiv.org/abs/2402.02172)

- CodeAgent: Autonomous Agent, a multi agent code review system.
- SOTA in code review systema.


---

### 3rd of Februry 2024

[More Agents Is All You Need](https://arxiv.org/abs/2402.05120)

- Scaling up LLM-agents increases performance with sampling & majority voting.
- Performance improvements increase and then decrease as difficult level gets harder. Improvements increase in function of number of steps. Prior probability of correct answer increases performance gains.


---

[Affordable Generative Agents](https://arxiv.org/abs/2402.02053)

- Affordable Generative Agents (AGA) framework: agent environment interaction and inter-agent interactions.
- Believable, low cost LLM-agents by replacing repetitive LLM inferences with learned policies. Models social relationships between LLM-agents and compresses auxiliary dialogue information.
- Emergent believable behaviour: LLM-agents generate finite behaviours in limited environments. Defines "mind wandering"-technique in memorory to generate diverse social behaviour by sampling both: highly relevant events and sampling ranly unrelated events. The idea is to randomness & spontaneus responses, like a real person.
- Social memory: relationship, feeling, events summary between the agents.



---

#### 2nd of February 2024


[K-Level Reasoning with Large Language Models](https://arxiv.org/abs/2402.01521)

- K-level of Reasoning: Recursive reasoning process, which improves dynamic reasoning by integrating cognitive hierarchy theory by recursively predicting and responding to the thoughts and actions of rivals.
- In essence, multiple LLM agents take a context, reason on it and make decision in "k-1"-level. The reasoning is then repeated in the "k"-level by integrating the the analysis from "k-1"-level to arrive decision in the "k"-level.


---


#### 1st of February 2024

[Multimodal Embodied Interactive Agent for Cafe Scene](https://arxiv.org/abs/2402.00290v1)

- MEIA (Multimodal Embodied Interactive Agent): Uses Multimodal Environment Memory (MEM) with LLM and VLM, to store egocentric environmental information (object IDs/coordinates as textual memory and visual observations as image memories) to improve significantly task planning and execution.
- MEIA is able to perform various tasks such as seating guidance, order taking and environmental adjustments being robust in zero-shot learning for real world tasks.
- It appears to be the first paper to introduce multimodal memory, which improves significantly performance and increases precision of the planning.
- Includes two measurement metrics: ESR (Executable Success Rate) and SSL (Succcess Rate Weighted by Step Length) with formulas included.
- Uses RGB images (stored in image memory)/depth images/segmentation images. 


---

[Efficient Exploration for LLMs](https://browse.arxiv.org/abs/2402.00396)

- Actively exploration is used to achieve high performance with less feedback.
- Uses double Thompson sampling with eistemic neural network (ENNs) to model reward uncertainty and least amount of queries.
- Gemini Nano is used as baseline model, which output is compared with Best-of-N responses from Gemini Nano based on reward model.


---

[Hello OLMo: A truly open LLM](https://blog.allenai.org/hello-olmo-a-truly-open-llm-43f7e7359222)

- OLMo: First open access data, open weights, open source code LLM.
- The model training data comes with need to agree to AI2's license terms wiith very clearly stated legal implications.


---

[Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents](https://browse.arxiv.org/abs/2402.00798)


- Formal-LLM: Context-Free Grammar (CFG) translates guidance and rules for each relevant task, which LLM text generation must follow when generating the plan.
- Prevents generating invalid plans.   


---

#### 30th of January 2024


[StrokeNUWA: Tokenizing Strokes for Vector Graphic Synthesis](https://arxiv.org/abs/2401.17093)

- StrokeNUWA: Introduces image representations based on vector graphics using "stroke tokens". The approach does not require using raster/pixel representation.
-  Includes components of: Vector-Quantized-Stroke (VQ-Stroke), Scalable Vector Graphics (SVG) compression, Encoder-Decoder LLM for SVG generation and post-processing SVG fixer.
-  Enables 94 times faster inference speed and representing images as more "language like" manner of sequences of strokes.


---

[Efficient Tool Use with Chain-of-Abstraction Reasoning](https://arxiv.org/abs/2401.17464)

- Chain-of-Abstraction (CoA): trains LLMs with decoded reasoning chains using abstract placeholders and then call tools to complete the reasoning chain.
- CoA learns more generic math reasoning and   


---

[Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios](https://arxiv.org/abs/2401.17167)

- UltraTool Construction-framework includes three key steps: Query collection, Solution Annotation and Manual refinement. 
- UltraTool: benchmarking LLM performance in using tools in real world.
- Reviews tool use performance from planning, tool creation awareness, tool creation, tool usage awareness, tool selection and tool usage.


---

[Can Large Language Models be Trusted for Evaluation? Scalable Meta-Evaluation of LLMs as Evaluators via Agent Debate](https://arxiv.org/abs/2401.16788)

- Scale-Eval: Meta-evaluation framework using agents debates to reach consensus or align with human answer in various task scenarios.


---

[LLaMP: Large Language Model Made Powerful for High-fidelity Materials Knowledge Retrieval and Distillation](https://arxiv.org/abs/2401.17244)

- LLaMP: ReAct-agents connected with arXiv, Wikipedia, Material Project-agents. Includes promts and json-formats used with the RAG-pipeline. Reduces hallucinations in material science queries.
  

---


#### 29th of January 2024

[Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception](https://arxiv.org/abs/2401.16158)

- Mobile-Agent: Multimodal Large Language Models (MLLM) for mobile devices, which locates visual/textual, plans, decomposes and executes complex tasks.
- OS agnostic
- Introduces Mobile-Eval benchmark and open sources [code](https://github.com/X-PLUG/MobileAgent).


---

[Beyond Direct Diagnosis: LLM-based Multi-Specialist Agent Consultation for Automatic Diagnosis](https://arxiv.org/abs/2401.16107)

- Patient consultation with muliple agents, starting with general practioner and then LLM agents in specific specialities: surgeon, respiratory doctor, endocrinologist.
- Icludes three stages: Individual practitioner consultation, practitioner group consultation and agent-based groupdecision fusion.

---

[Divide and Conquer: Language Models can Plan and Self-Correct for Compositional Text-to-Image Generation](https://arxiv.org/abs/2401.15688)

- CompAgent: LLM agent is manages the task of the entire image generation.
- The LLM agent is used to plan composition of objects next to each other. Achieves better images for example when prompted to generate image with a red hat next to blue backpack.

---


#### 28th of January 2024

[YODA: Teacher-Student Progressive Learning for Language Models](https://arxiv.org/abs/2401.15670)

- YODA: Hunan-like progressive learning paradigm for LLMs, where student agent learns in fixed dataset by learning first basic questions, then learns to generalize and finally learns harder problems.
- Teacher agent asks then similar questions from the student agent. The teacher agent gradually adds more complex and more generic questions after each iteration and offers feedback to the student agent for the answers provided.
- The approach helps the student agent to learn to solve problems and generalize problems comprehensively, which leads to 10% improvement in MATH benchmark from the original Llama 2. 


---


#### 26th of January 2024

[Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion](https://arxiv.org/abs/2401.14717)

- Reviews how voice-assistant systems should predict and manage: turn-taking, backchanneling and continued speaking.
- Contiying speaking refers to the other party needing to continue listening the current speaker. Backchanneling refers to the current listener needing to produce a short utterance of acceptance without meaning to take over the speaker role. Turn-taking refers to the listered being expected to take over speaking turn from the current speaker.
- Creates fusion model combining both LLM (GPT-2/RedPajama) and HuBERT-acoustic model.


---

#### 24th of January 2024

[Hi-Core: Hierarchical Knowledge Transfer for Continual Reinforcement Learning](https://arxiv.org/abs/2401.15098)

- Hi-Core: Formulates goals as a high-level policy using LLM reasoning and then low-level policy learning towards these high-level goals. Policy library is used to store policies searchable with embeddings based on policy description.
- Makes the important point, that to learn high-level human cognitive skills using transfer learning, we need to represent high-level human knowledge effectively to be able to transfer them into models.


---


#### 23rd of January 2024

[Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding](https://arxiv.org/abs/2401.12954)

- Meta-prompting: LLM coordinate and execute multiple independent queries with their responses to generate final answer.


---


[AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents](https://arxiv.org/abs/2401.12963)

- AutoRT: Fleet of robots use VLM and LLM 


---

[HAZARD Challenge: Embodied Decision Making in Dynamically Changing Environments](https://arxiv.org/abs/2401.12975)

- HAZARD-benchmark made of three dynamic challenges for an embodied agents: flood, fire and wind, which  performance are evaluated in terms of value, steps and damage.
- Builds LLM-based pipeline for embodied agents by providing it task description, agent status and target info. Agent reads environment information, includes observation memory and LLM-based decision maker to select the next action.


---


#### 22th of January 2024


[Memory Matters: The Need to Improve Long-Term Memory in LLM-Agents](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27688)

- Reviews memory management of LLM-agents with useful insights about using different types meta-data in vector db along the word embeddings as long-term memory.
- Identifies in past research example ways of storing: thoughts/skills in vector db, but as well gaps in retrieving information, when different memories may contradict the retrieval. 


---

[OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics](https://arxiv.org/abs/2401.12202)

- OK-robot (Open-Knowledge): 59% success rate in open ended picking and dropping task.
- SOTA level in OVMM-benchmark.

---

[WARM: On the Benefits of Weight Averaged Reward Models](https://arxiv.org/abs/2401.12187)

- Weight Averaged Reward Models (WARM) models.


---

[PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety](https://arxiv.org/abs/2401.11880)

- PySafe: Safety research on LLM agents based on behavioural/psychological-characteristics.


---


#### 21st of January 2024

[AttentionLego: An Open-Source Building Block For Spatially-Scalable Large Language Model Accelerator With Processing-In-Memory Technology](https://arxiv.org/abs/2401.11459)

- AttentionLego: LLM is implemented on Processing-In Memory (PIM) HW.


---

[The Conversation is the Command: Interacting with Real-World Autonomous Robot Through Natural Language](https://arxiv.org/abs/2401.11838) 

- Simplistic robotic control using VLM and LLM: VLM to object textual description and scene comprehension. LLM for reasoning and REM-node to translate commands into robot actions.


---


#### 19th of January 2024

[Tool-LMM: A Large Multi-Modal Model for Tool Agent Learning](https://arxiv.org/abs/2401.10727)

- Tool-LMM: LLM is agent able to process multimodal inputs into APIs of the specific modalities.
- Input modalities include, text, audio/text, text/video and text/image. The LLM text output includes recommendation of the API to be used and model information.


---

[A match made in consistency heaven: when large language models meet evolutionary algorithms](https://arxiv.org/abs/2401.10510)

- Compares and finds multiple similarities between GPT-LLMs and Genetic Algorithm (GA)-evolutionary algorithms.


---

[CivRealm: A Learning and Reasoning Odyssey in Civilization for Decision-Making Agents](https://arxiv.org/abs/2401.10568)

- CivicRealm: RL agent generalization benchmark, based on video game environment with various players and dynamic game space, imperfect information and random variability.


---


#### 18th of January 2024

[Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)

- Self-rewarding LLMs: Ability for LLM to follow instructions and Ability to create/evaluate new training data (Self-Instruction creation).
- LLLm-as-a-Judge: LLM acts as a reward model and self-reward its own responses.
- Claims to outperform Claude 2/Gemini Pro/GPT-4 0613 with three iterations and ability to keep continuously improving both self-instructions and the reward signal.


---

[R-Judge: Benchmarking Safety Risk Awareness for LLM Agents](https://arxiv.org/abs/2401.10019)

- R-Judge: Safety benchmark for LLM-agents, not LLM models on 27 risk scenarios.


--- 


#### 17th of January 2024

[Large Language Models Are Neurosymbolic Reasoners](https://arxiv.org/abs/2401.09334)

- LLM agent plays text-based game with access to Symbolic module.


---

[ReFT: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967)

- Reinforced Fine-Tuning (ReFT): In the initial SFT-step, the model is trained to produce correct answers to mathematical problems.
- In the second step, online RL with PPO is used to prompt multiple CoT responses to learn from them.
- ReFT uses majority voting and reward model reranking. 


---

[Scalable Pre-training of Large Autoregressive Image Models](https://arxiv.org/abs/2401.08541)

- AIM: Visual models, which scale with both compute and data introduced.

---

[What makes for a 'good' social actor? Using respect as a lens to evaluate interactions with language agents](https://arxiv.org/abs/2401.09082)

- LLM agent as as social (automated) actor.
- Identifies what makes a good vs negative social behaviour for LLM agents.


---


#### 16th of January 2024

[Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering](https://arxiv.org/abs/2401.08500)


- AlphaCodium: Improves code solutions through AI code tests.
- Iteratively reasons about code tests and reflects problem, generates AI tests to improve testing.
- Two phases: Preprocessing (to reason new AI tests from ranked solutions feom public tests) and Code iteration (with public and AI tests).


---

[MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World](https://arxiv.org/abs/2401.08577)

- MultiPLY: Multisensory (temperature, tactile, audio and visuals) embodied agent acts (action tokens such as navigate/select/touch/observe/look around/) in 3D virtual environment.
- The model trained with ultisensory Universe-dataset, performs multiple tasks: navigates, manipulates, uses tools, dialogue,
- Encodes 3D-scenes as object centric representations, generate action token to be taken from current state token (temperature/tactile/sound/object) within the environment to reach new state observation in time. The new state token is fed back to LLM to drive follow up actions.


---

[DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models](https://arxiv.org/abs/2401.08392)

- DoramonGPT includes task-related symbolic memory, sub-task/knowledge tools and MCTS planner.
- The task related symbolic memory will choose either the Spatial or Time-dimension as most relevant based on the LLM.   
- DoramonGPT collecta information before reasoning, reasons spatial-temporal video, explores different solutions in a large planning space.


---

[Self-Imagine: Effective Unimodal Reasoning with Multimodal Models using Self-Imagination](https://arxiv.org/abs/2401.08025)

- Self-Imagine: VLM creates HTML code about the text question, renders it as an image and uses the image with the question to answer the question with the VLM.


---

[Application of LLM Agents in Recruitment: A Novel Framework for Resume Screening](https://arxiv.org/abs/2401.08315)

- Automated resume screening, where segments from CV are classified into information types, personal information is removed. T
- The HR grading LLM agent rates these resumes and another HR decision making agent picks preferred application with eplanation, which is then available for the HR professional.


---

[Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417)

- Contrastive Preference Optimization (CPO): A potential improvement to DPO, applied in machine translation.


---


#### 15th of January 2024

[Exploring the Potential of Large Language Models in Self-adaptive Systems](https://arxiv.org/abs/2401.07534)

- Literature review of Self-Adaptive Systems with LLMs.


---

[A Study on Training and Developing Large Language Models for Behavior Tree Generation](https://arxiv.org/abs/2401.08089)

- LLMs used to generate Behavioural Trees (BT) generation for agents/robots.


---

[When Large Language Model Agents Meet 6G Networks: Perception, Grounding, and Alignment](https://arxiv.org/abs/2401.07764)

-  Least Age-of-Thought (LAoT) model caching algorithm to manage local/global compute/network traffic to avoid model with least valuable thoughts. 


---


#### 14th of January 2024

[CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges](https://arxiv.org/abs/2401.07339)

- Introduces CodeAgent, a LLM agent able to use tools (search, code navigation and code interpreter) to generate code/create repositories (instructions, code dependencies) better than Github Copilot.
- Introduces CodeAgentBench-dataset.
- Code symbol navigation is key component, to explore: file/module-based parsing and class/function-symbol navigation. 


---

[Small LLMs Are Weak Tool Learners: A Multi-LLM Agent](https://arxiv.org/abs/2401.07324)

-  α-UMi:  Multi-agent LLM, which includes planner/caller and summarizer and tools.


---


#### 12th of January 2024

[ModaVerse: Efficiently Transforming Modalities with LLMs](https://arxiv.org/abs/2401.06395)

- ModaVerse: Introduces Adaptor+Agent framework for training multi-modal LLM able to process content across audio/video/image modalities.
- Introduces Input/Output (I/O) Alignment: LLM generates language aligned meta-responses, which are instructions to activate specific generative models.
- This method is capable of converting variety of modalities, while being very efficient to train.


---

[AntEval: Quantitatively Evaluating Informativeness and Expressiveness of Agent Social Interactions](https://arxiv.org/abs/2401.06509)

- AntEval: a framework to evaluate LLM-agents social interactions with two metrics: Information Exchange Precision and Intention Expresiveness Gap.


---

[Mutual Enhancement of Large Language and Reinforcement Learning Models through Bi-Directional Feedback Mechanisms: A Case Study](https://arxiv.org/abs/2401.06603)

- Investigates bi-directional feedback loop, where LLM agent acts as a teacher, while the RL agent acts as a student.


---

#### 11th of January 2024

[EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction](https://arxiv.org/abs/2401.06201)

- EASYTOOL: Creates a cleaned version of any tool/API documentation for LLM agent to use via single "tool instruction".
- Tool documentation is translated into: tool descriptions and tool core functionality. Each are created using specific LLM instructions.
- Significantly improves tool-based LLM agent performance. 


---

[Designing Heterogeneous LLM Agents for Financial Sentiment Analysis](https://arxiv.org/abs/2401.05799)

- Heterogenoeus multi-Agent Discussion (HAD): Multiple agents with each instructions to pay attention to error category types, which form the resulting answer based on shared disussion. The domain of the research is Financial Sentiment Analysis.
- Builds on the conclusion, that LLMs are "resources": similar to Minsky's theory about human mind being built from a [Resource-cloud](#resourcecloud) to be activated/deactivated on the spot.
- Defines  Kernel Theory-Based Design: Kernel theory, Meta-requirements, Meta-designs, Testable hypothesis. 


---

[Evidence to Generate (E2G): A Single-agent Two-step Prompting for Context Grounded and Retrieval Augmented Reasoning](https://arxiv.org/abs/2401.05787)

- Evidence-to-Generation (E2G): Single LLM produces in two-steps answer step-by-step based on evidence from the context/question provided.
- E2G represents context-aware reasoning.


---


#### 10th of January 2024

[Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)

- Adds backdoors on LLMs.
- Trains deceptive LLMs using data, which "acts" based on being either in training vs inference: demonstrates safe code vs unsafe code.


---

[Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security](https://arxiv.org/abs/2401.05459)

- Reviews systematically "Personal LLM Agents" connected to personal data and devices for personal use.


---


[The Impact of Reasoning Step Length on Large Language Models](https://arxiv.org/abs/2401.04925)

- Adding reasoning steps improvea accuracy unril 5th step.


---

[InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks](https://arxiv.org/abs/2401.05507)

- DABench-benchmark for LLM based data analysis and open sources Data analysis agent : DA Agent.


---


#### 9th of January 2024

[Agent Alignment in Evolving Social Norms](https://arxiv.org/abs/2401.04620)

- EvolutionaryAgent: Evaluates LLM agents based on fitness to social norms using observer LLM within EvolvingSociety-environment.
- LLM agents producing highest social norm ratings, self-envolve and reproduce into new generation LLM agents. Agents either convert into obsolate or survived.
- Agents events are recorded within short term memory with a threshold, which defines when long term and higher-level memories are distilled.
- Defines initial stage of the EnvolvingSociety and the desired direction only.


---

[Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects](https://arxiv.org/abs/2401.03428)

- Reviews LLM Intelligent agents: definitions, frameworks, single/multiple agents, compoments, cognitive features etc.

---

[Metacognition is all you need? Using Introspection in Generative Agents to Improve Goal-directed Behavior](https://arxiv.org/abs/2401.10910)

-  Adds a metacognition to LLM agents for emulating System 1 and System 2 processes. The idea is to let LLMs "think about thinking".
-  The Metacognition module (knowledge about itself, the task and the strategies) gets triggered to ask reflective questions, when the LLM agent is not making significant progress.
-  The metacognition is used throughout the planning, evaluation, monitoring and cognition-steps using reflective questions and then stored in the meta-memory used.


---

<div id="agentbasedai"> </div>  


#### 7th of January 2024

[Agent AI: Surveying the Horizons of Multimodal Interaction](https://arxiv.org/abs/2401.03568)

- Agent AI system: Perceives and acts in different domains and applications.
- Multi-modal generalist agent: Environment and Perception with task-planning and skill observation, Agent learning, Memory, Agent action; Cognition.


--- 


### 4th of January 2024

[LLaVA-ϕ: Efficient Multi-Modal Assistant with Small Language Model](https://arxiv.org/abs/2401.02330)

- LLava-Phi: VLM using Phi-2 as LLM model with CLIP-ViT-L/14 with 336x336 visual encoder.


---

[Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives](https://arxiv.org/abs/2401.02009)

- Self-Contrast: Explores potential paths, Contrasts differences and Summarizes them into checklist to better reason.
- Many LLM agent errors are due to inconsistent feedback.


---

[INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning](https://arxiv.org/abs/2401.06532)

- Technique to tune LLM for "search": INstruction Tuning datasEt foR Search (INTERS).

---


#### 3rd of January 2024

[Act as You Learn: Adaptive Decision-Making in Non-Stationary Markov Decision Processes](https://arxiv.org/abs/2401.01841)

- Adaptive MCTS (Ada-MCTS): explores using epistemic & aleatoric uncertanties to adapt risk-aversion behaviour vs performance when spending more time in the environment.


---

[Economics Arena for Large Language Models](https://arxiv.org/abs/2401.01735)

- EconArena: Reviews multiple LLM models jn their ability to act rationally by comparing performance between models and against Nash Equilibrium (NE) rationality.
- Better models act more rational. LLMs are dynamically able to change strategies based on opponent strategy. Game history improves reasoning. Competing with rational opponent helps to achieve NE quicker.


---


#### 2nd of January 2024

[LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/abs/2401.01325)

- LLMs have built-in capability to manage long context, similar as children manage long context such as books mainly by having seen short context text.
- Self-Extend: No specific training / finetuning required. Plug in 4 lines of code during inference to the attention mechanism, based on LLM with RoPE and FLOOR-operation.


---

<div id="spin"> </div>  

[Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)

- Self-Play fIne-tuNing (SPIN): Fine-tuning LLMs based on Self-play mechanism, where the main player is the to-be learned LLM from the current iteration and its opponent is the same LLM from the previous iteration.



#### 22th of December 2023

[Pangu-Agent: A Fine-Tunable Generalist Agent with Structured Reasoning](https://arxiv.org/abs/2312.14878)

- Pangu-Agent: Introduces a generic RL-based objective to improve agents intrinsic and extrinsic functions. 

---


#### 21st of December 2023

[AppAgent: Multimodal Agents as Smartphone Users](https://arxiv.org/abs/2312.13771)

- Multimodal VLM agents learn operate popular smartphone apps by creating a knowledge base through: Autonomous exploration and Human demonstrations.
- Includes: Exploration phase and Deployment phase.
- Exploration phase learns smartphone functionalities through trial and error, which are saves records of effects to actions and stops, if the current view is unrelated to the assigned task. Exploration stops, whene task is finished. Alternatively these behaviours are shown through human demonstrations, which keeps the agent exploration streamlined and efficient.
- In deployment phase, the VLM agent has access to the UI screenshot and potential actions. The agent generates a summary of the actions taken and interaction history, which are passed to the next step.


---

[Capture the Flag: Uncovering Data Insights with Large Language Models](https://arxiv.org/abs/2312.13876)

- Exlores two types of Data Science Agents: Explorer agent and Aggregator agent 


---


#### 20th of December 2023

[AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation](https://arxiv.org/abs/2312.13010)

- AgentCoder:  Multi-Agent Assistant Code Generation made from Programmer Agent, Test designer Agent and Test executor Agent
- Uses Self-Refine with CoT in a Multi-Agent System.


---

[DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines](https://arxiv.org/abs/2312.13382)

- LM Assertions: Integrates with [DSPy](https://github.com/stanfordnlp/dspy), which integrates reasoning, self-improvement, augmentation, retrieval and tools (DSPy is like challenger for Langchain).
- To help runtime self-refinement in LM pipelines with boolean type conditions: Assert (hard or critical condition) and Suggest (soft condition).
- For example a critical condition (hard) is such, that will resul the LM pipeline to halt, if the condition is not met with maximum number of attempts, while Suggest-option still lets the pipeline to continue. 


---

[ASSISTGUI: Task-Oriented Desktop Graphical User Interface Automation](https://arxiv.org/abs/2312.13108)

- ASSISTGUI: Window mouse / keyboard management with LLM.


---

[Generative agents in the streets: Exploring the use of Large Language Models (LLMs) in collecting urban perceptions](https://arxiv.org/abs/2312.13126)

- Explores generative agents in urban environments: includes memory modyke, movement module, visual inference module and a LLM module


---

[dIR -- Discrete Information Retrieval: Conversational Search over Unstructured (and Structured) Data with Large Language Models](https://arxiv.org/abs/2312.13264)

- Discrete Information Retrieval (dIR): Text-queries of SQL databases using LLMs.


---


#### 19th of December 2023

[Large Language Models Play StarCraft II: Benchmarks and A Chain of Summarization Approach](https://arxiv.org/abs/2312.11865)

- Plays Starcraft 2 better than an average player by using Chain of Summarization (CoS), python-sc2 and TextStarCraft II-environment (Observation-to-Text Adapter: and Text-to-Action Adapter).
- Chain of Summarization (CoS): Improves LLMs capability to extract / analyze information using two compnents: Single-frame summarization and Multi-frame summarization.
- TextStarCraft II-environment processes game information into textual format for LLM model defining macro-actions and a rule-based method for micro-actions
- System prompt includes: Situation Overview, Situation Analysis, Strategic Planning, Opponent Strategy, Analysis, Strategic Recommendations, Decision-Making rocess.
- Reduces 10x the need of LLM API calls and improves strategic, analytical and judging capabilities. 


---

#### 19th of December 2023

[Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives](https://arxiv.org/abs/2312.11970)

- LLM empowered agent-based modeling and simulation framework: surveys the landscape of utilizing LLMs in agent-based modeling and simulation.
- Framework examines challenges, future directions, motivation for applying LLMs, environment perception, human alignment, action generation, evaluation, cyber, physical, social, and hybrid domains.
- This framework provides a comprehensive overview of recent works in this interdisciplinary field.


---


<div id="humancap"> </div>  

[Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives](https://arxiv.org/abs/2312.11970)

- Reviews LLM-based agents on their ability to simulate various human-like capabilities.


---


#### 18th of December 2023

[Agent Assessment of Others Through the Lens of Self](https://arxiv.org/abs/2312.11357)

- Discusses concept of Self-Awareness of Autonomous Agents.


---

[Evaluating Language-Model Agents on Realistic Autonomous Tasks](https://arxiv.org/abs/2312.11671)

- Autonomous Replication and Adaption (ARA) framework: reviews ability of LLM agents to acquire resources, create copies of themselves and adapt to novel situations in the real world.
- Tests LLM-agents using Scaffolding programs to interact with LLMs.
- Defines implications of potentially ARA-level agents.

---

[LLM-ARK: Knowledge Graph Reasoning Using Large Language Models via Deep Reinforcement Learning](https://arxiv.org/abs/2312.11282)

- LLM-ARK: LLM reasons from Knowledge Graphs with DRL.

---

#### 17th of December 2023

[Learning to Act without Actions](https://arxiv.org/abs/2312.10812)

- LAPO (Latent Action Policy).


---

#### 16th of December 2023

[ProTIP: Progressive Tool Retrieval Improves Planning](https://arxiv.org/abs/2312.10332)

- Progressive Tool Retrieval Improves Planning (ProTIP): Mulit-step planning with external tools, where tasks are decomposed without explicit definition of the sub-task.
- Addresses the issue, where single-step tool retrieval does not manage to handle dependencies between the tools.


---

<div id="restreact"> </div>  


#### 15th of December 2023

[ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent](https://arxiv.org/abs/2312.10003)

- Self-Imepoving LLM model without any human-assisted data for fine tuning achieving significantly better reasoning results with smaller model, when using the synthetic data to distill smaller model.
- Finetunes LLM with ReST using ReAct-method reasoning-actions.


---

<div id="agenticaisystem"> </div>  


#### 14th od December 2023

[Practices for Governing Agentic AI Systems](https://cdn.openai.com/papers/practices-for-governing-agentic-ai-systems.pdf)

- OpenAI's research on Agentic AI systems with definition of Agentic AI system.
- Includes level of "Agenticness":  the degree of goal complexity, environment complexity, adaptability and independence.

---

[TinyGSM: achieving >80% on GSM8k with small language models](https://arxiv.org/abs/2312.09241)

- First student LLM to learn the Teacher LLM model ( GPT-3.5) performance in mathematical reasoning using synthetic data from the teacher model.  
- TinyGSM: Two 1.3B LLNs with a 1.3B verifier LLM achieves SOTA level 81.5% accuracy on GSM8k, which consists of a high-quality dataset TinyGSM and use of verifier selecting final answer from multiple output generations.


---

[Modeling Complex Mathematical Reasoning via Large Language Model based MathAgent](https://arxiv.org/abs/2312.08926)

-  Planner-Reasoner-Executor-Reflector (PRER) / MathAgent: Planner, Reasoner, Executor and Reflector.
-  Systematic process for solving zero-shot mathematical reasoning with LLM agents.


---

[Rational Sensibility: LLM Enhanced Empathetic Response Generation Guided by Self-presentation Theory](https://arxiv.org/abs/2312.08702)

- Self-Representation with Lamb:  Uses semantic label to set tone for the conversation.


---

[LiFT: Unsupervised Reinforcement Learning with Foundation Models as Teachers](https://arxiv.org/abs/2312.08958)

- LiFT: Outperforms significantly VPT/other models in MineDojo-ennvironment.
- LLM provides task instruction.
- VLM is sed to learn policy and act as a reward model.


---

[LLMind: Orchestrating AI and IoT with LLMs for Complex Task Execution](https://arxiv.org/abs/2312.09007)

- LLMind: Includes coordinator updating short-term memory/retrieving required AI (IoT) modules with ability to define, if script exists for the module and enerates it, if missing. Coordinator retrieves error / output messages from the executed script, which is handled by the script executor.


---

[Holodeck: Language Guided Generation of 3D Embodied AI Environments](https://arxiv.org/abs/2312.09067)

- HoloDeck: Generating 3d embodied environments with LLM: FLoor-wall module, doorway-window module, object selection module and layout design module.


---

[Personalized Path Recourse](https://arxiv.org/abs/2312.08724)

- Personalized Path Recourse (PPR): Personalized path of actions to achieve a certain goal with an agent.


---

[Adaptive parameter sharing for multi-agent reinforcement learning](https://arxiv.org/abs/2312.09009)

-  AdaPS: Maps agents to different regions of brain/shared network based on identity vectors obtained with VAE and clusters agents to K classes.


---

[Auto MC-Reward: Automated Dense Reward Design with Large Language Models for Minecraft](https://arxiv.org/abs/2312.09238)

- RL agent using LLM to act as a Reward designer, Reward critic and a Trajectory designer.


---

[Vision-Language Models as a Source of Rewards](https://arxiv.org/abs/2312.09187)

- VLMs work as reward models and larger scale improves performance of the reward model.


---

[Learning Coalition Structures with Games](https://arxiv.org/abs/2312.09058)

- Coalition Structure Learning (CSL): Learns coalitions of agents via set of games.


---

#### 13rd of December 2025

[KVDirect: Distributed Disaggregated LLM Inference](https://arxiv.org/abs/2501.14743)

- KVDirect: Framework optimizes KV cache transfer to enable distributed disaggregated LLM inference.
- Tensor-centric communication mechanism, custom communication library, dynamic GPU resource scheduling, pull-based KV cache transfer strategy, reduces synchronization overhead.
- KVDirect reduces per-request latency and improves resource utilization in disaggregated LLM inference.


---


#### 12th of December 2023

[Medprompt+](https://github.com/microsoft/promptbase/tree/main)

- Medprompt+ extends Medprompt-method improved by asking additionally if scrapt-pad is needed and increasing number of ensembled calls from 5 to 20.


---

[diff History for Long-Context Language Agents](https://arxiv.org/abs/2312.07540)

- Compresses consecutive text observations from environment with Unix "diff"-command, which leads to 700% improvement in game score, outperforming existing agents by 40%, which use visual observations.
- Similar approach may enable building vastly more generic embodied LLM agents.


---

[Sequential Planning in Large Partially Observable Environments guided by LLMs](https://arxiv.org/abs/2312.07368)

- Neoplanner: builds state space model of the environment by testing different actions, observations and rewards. Builds a graph memory of learnings from all previous trials using Learner agent.
- Model provides anytime best policy given the knowledge at that moment. Balances exploration and exploitation.


---


#### 11th of December 2023

[Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)

- ReST<sup>EM (Expectation-Maximization)</sup>: LLM generates samples (E-step/Expectation-step) using temperature sampling, filter samples using binary feedback/reward, fine-tune LLM using these feedbacks (M-step/Maximization-step). Repeat few rounds. Improves significantly coding and math benchmark results. 
- Ability to generate multiple correct solutions compared against human-generated data.
- ReST<sup>EM</sup> uses temperature sampling (diverse/creative), compared to [STaR](#star)-method based on greedy sampling (most-likely), where the rationalization-process leads to false-positive solutions.


---


#### 8th of Decembebr 2023

[KwaiAgents: Generalized Information-seeking Agent System with Large Language Models](https://arxiv.org/abs/2312.04889)

- KwaiAgents, an autonomous agent loop including three key components: (KAgentSyst), LLMs (KAgentLLMs) and Benchmarks (KAgentsBench).
- System includes: Memorybank (Knowledge, Conversation and Task), Tool-library (Factuality-aware, Time-aware and Custom tools) used with Memory update, Task plan, Tool execution and Finish & Conclude-steps.
- LLM-component includes templates for LLs, Meta-Agent Tuning (MAT)-framework and LLM services. Benchmarks include both human and LLM-driven profiling.
- MAT includes six key components to generate prompt templates: system profile, instructions/constraints, tool specification, goal placement, memory allocation and output format. 


---


#### 7th of December 2023

[Chain of Code: Reasoning with a Language Model-Augmented Code Emulator](https://arxiv.org/abs/2312.04474)

- Creates answer in two steps: Starts by creating pseudo-code to solve the question, then runs the pseudo-code in code interpreter or LM emulating code, in case no code interpreter is available. 


---

[AVA: Towards Autonomous Visualization Agents through Visual Perception-Driven Decision-Making](https://arxiv.org/abs/2312.04494)

-  Autonomous Visualization Agents (AVAs): User instructions are converted with Visualization agent into actions and the taken actions are converted back to language within visualization tasks.
-  Components include: Visual perception, Action planning and Memory components, working within visualization-perception-action-loop.  


---

[Generating Illustrated Instructions](https://arxiv.org/abs/2312.04552)

- StackedDiffusion: Generates illustrated instructions based on text, which helps to train SOTA level multi modal models preferred over human generated articles.

---

[Fortify the Shortest Stave in Attention: Enhancing Context Awareness of Large Language Models for Effective Tool Use](https://arxiv.org/abs/2312.04455)

- Introduces "Attention Buckets", which enable a 7B open source model to acchieve GPT-4 level tool use performance by compensating attention peaks between parallel processes in specific context.


---


#### 6th of December 2023

[Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia](https://arxiv.org/abs/2312.03664)

- Concordia-library: Simulation environment made of multiple agents and Grand Master (GM) inspired by the Dungeons and Dragons game.
- Agents consume observations and GM agent actions. Agent produces actions and GM event statements (such as physical grounding). 
- Includes long and short term memory, which include state of the world.


---

[LLM as OS (llmao), Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem](https://arxiv.org/abs/2312.03815)

- AIOS-Agent Ecosystem: Envisions LLMs as OS, Agents as Applications, Natural Language as Programming language and Tools as Devices/Libraries.


---


#### 5th of December 2023


[Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models](https://arxiv.org/abs/2312.03052)


- Answers visual questions by creating programs, that can review the image such as count number of specific types of objects and use tools.
- Answer is provided with CoT reasoning based on filtered program from many programs executed.


---

[Beyond Isolation: Multi-Agent Synergy for Improving Knowledge Graph Constructio](https://arxiv.org/abs/2312.03022)

- Uses three LLM agents for entity, event and relation extraction to build knowledge graph.


---

[Large Knowledge Model: Perspectives and Challenges](https://arxiv.org/abs/2312.02706)

- Large Knowledge Models: Reviews combination of LLMs (neural representation) and Knowledge graphs (symbolic representation) through usage of knowledge graph embeddings and text embeddings with LLMs. 


---


#### 4th of December 2023

[Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication](https://arxiv.org/abs/2312.01823)

- Exchange-of-Thought (EoT): Improvement from CoT and Self-Consistency, where thoughts from other LLMs are considered, outperforming in mathematical reasoning the CoT with Self-Consistency
- Proposes four communication paradigms to define the setup of the Exchange-of-Thought: Memory, Report, Relay and Debate. 
- For example in Debate-mode: two LLM agents produce first ansswer the question and the two rationalizations are provided to the third LLM agent in order to debate these solutions in order to provide the right answer.


---

[LLM A*: Human in the Loop Large Language Models Enabled A* Search for Robotics](https://arxiv.org/abs/2312.01797)

-  LLM A*: Includes current node, goal node, optical action and these three make up the plan.
-  The chat-environment with user defines user inputs: Setting up environment, Setting up Action model, Start and Target Nodes, Heuristic and Rules.
-  Demonstrates the possibility of achieving very good path planning results using mobile embodied agents.


---

[Towards Learning a Generalist Model for Embodied Navigation](https://arxiv.org/abs/2312.02010)

- NaviLLM: Embodied navigation with LLMs using schema-based instruction (task, history, observation and output hint), which generalizes well to unseen navigation tasks.
- Uses the following Multi-task learning modules: Visual-Language Navigation, Object localization, Trajectory Summarization and 3D Queestion Summarization.


---

[OpenVoice: Versatile Instant Voice Cloning](https://arxiv.org/abs/2312.01479)

- OpenVoice: Voice cloning almost from instant voice record.


---


#### 29th of Novemebr 2023

[Universal Self-Consistency for Large Language Model Generation](https://arxiv.org/abs/2311.17311)

- Universal Self-Consistency (USC): Uses LLMs to select the most consistent answer among multiple candidates working in mathematical reasoning and code generation and unlike the original Self-Consistency, the method works in open-ended questions.
- This can be used as a more capabale component in the [STaR-method](#star), which generalizes with Q&A with open-ended answers, not only precise answers.


---


#### 28th of Novemebr 2023

[Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine](https://arxiv.org/abs/2311.16452)

- Medprompt: Generalist LLM using MedPrompt outperforms SOTA specialist model.
- Uses SOTA prompt method: CoT, Choice Shuffle and Self-Consistency prompting
- Introduces Choice Shuffle-technique, which inreases diversity of the reasoning paths.
  

---


#### 27th of Novemeber 2023 

<div id="extreme"></div>

[Some intuitions about large language models](https://docs.google.com/presentation/d/1hQUd3pF8_2Gr2Obc89LKjmHL0DlH-uof9M0yFVd3FA4/edit) 

- Jason Wei Blog post / Presentation.
- Learning the relationship from Input to Output is as well Next-word prediction learning.
- Next-word prediction is massively multi-task learning.


---


#### 22th of November 2023

[Building the Future of Responsible AI: A Pattern-Oriented Reference Architecture for Designing Large Language Model based Agents](https://arxiv.org/abs/2311.13148)

- Identifies two types of LLM agents: "Agents-as-workers" and "Agents-as-coordinators".

  
---


#### 21st of November 2023

[System 2 Attention (is something you might need too)](https://arxiv.org/abs/2311.11829)

- System 2 Attention (S2A): Generate interim user question and interim context from the original user input. Finally, generate the final answer by answering to the interim user question from the interim context. 
- Reduces hallucination from irrelevant context by first defining the question and the context and this way separating irrelevant facts from impacting the response generation.


---


#### 20th of November 2023

[Igniting Language Intelligence: The Hitchhiker's Guide From Chain-of-Thought Reasoning to Language Agents](https://arxiv.org/abs/2311.11797)

- Systematic review of research from Chain-of-Thought (CoT) to LLM Agents and identifies gaps in generalization, redundant interactions and customization and more. 


---


#### 17th of November 2023

[A Language Agent for Autonomous Driving](https://arxiv.org/abs/2311.10813)

- Agent-Driver: Uses LLM agent for human-like intelligence for autonomous driving.
- Tool library provides input for: detection, prediction, occupancy and mapping functions. Memory includes commonsense memory and Experience memory. There is apart historical trajectories and ego-states.
- The reasoning engine includes: CoT reasoning, Task planning, Motion planning and Self-Reflection. These lead to actions and again to environment update. 


---


#### 16th of November 2023

[Digital Socrates: Evaluating LLMs through explanation critiques](https://arxiv.org/abs/2311.09613)

- Digital Socrates: evaluates reasoning flaws: giving feedback on why and where? 


---


#### 15th of November 2023

[Divergences between Language Models and Human Brains](https://arxiv.org/abs/2311.09308)

- Reviews differences measured with MEG in human brain vs. language models.
- The study reveeals, that LLMs are less good at social/emotional intelligence and physical commonsense reasoning.
- Finetuning helps to align LLMs to act more in human brain-like manner. 

---

[AutoMix: Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963)

- AutoMix: Use a smaller LLM to generate initial response and uses Meta-Verifier to check the trustworthy in rough scale. If the answer is trustworthy then use the small LLM answer, otherwise consult a larger LLM.
- Uses Incremental Benefit Per Unit Cost (IBC) metric to asses effectiveness of this approach.


---


#### 14th of November 2023

[DeepThought: An Architecture for Autonomous Self-motivated Systems](https://arxiv.org/abs/2311.08547)

- DeepThought: An architecture for cognitive language agents posing agency, self-motivation, and partly meta-cognition.
- Includes supervisor module, Deep Reinforcement Learning module, Attention Schema (long-term memory), Language/Auditory/Vision modules and Embedding store.


---


#### 9th of November 2023

[LLM Augmented Hierarchical Agents](https://arxiv.org/abs/2311.05596)

- Hierchical agent uses LLM to evaluate, when to use specific skill to complete specific sub-level task with long horizon.
- The resulting model works without the need for a LLM after the training.


---

[Prompt Engineering a Prompt Engineer](https://arxiv.org/abs/2311.05661)

- Guide LLM to prompt engineer prompts automatically
- The metaprompt uses: prompt engineering tutorial, two-step task description, step-by-step reasoning template and context specification.


---


#### 8th of November 2023

[ADaPT: As-Needed Decomposition and Planning with Language Models](https://arxiv.org/abs/2311.05772)

- ADaPT: Plans and decomposes dynamically complex tasks with LLMs, if the executor is not able to complete the task.


---


#### 2nd of November 2023

[RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation](https://arxiv.org/abs/2311.01455)

- RoboGen: Agent using LLMs to define new tasks to learn, create their simulation environments, train on them to acquire diverse & new skills.
- Agent includes: Task proposal, Scene generation, Training Supervision Generation & Skill learning.


---

<div id="stopvideo"></div>

[Youtube. Adam Kalai presents "Recursive Self-improving Code Generation - talk 2.11.2023](https://www.youtube.com/watch?v=RovcBFlfXpQ)

- Adam Kalai talk on the "Self-Taught Optimizers (STOP): Recursively Self-Improving code generation", which is in essence attempts to build code for letting LLMs themselves improve (their) own code.
- I recommend to check this especially from safety-aspects on the point "sandbox-flag" and to better understand the 

---


#### 1st of November 2023

[Plug-and-Play Policy Planner for Large Language Model Powered Dialogue Agents](https://arxiv.org/abs/2311.00262)

- Introduces plug-and-play dialogue policy planner(PPDPP).
- Dialogues plans using Self-play with three LLM agents: one acting to achieve a goal like buying a product at cheaper price, second to negotiate as seller a higher price and a third LLM scoring performance as reward model.


---

[SAGE: Smart home Agent with Grounded Execution](https://arxiv.org/abs/2311.00772)

- SAGE (Smart home Agent with Grounded Execution).
- Device interaction: Interaction planner, Attribute retriever, API documentation retriever, Device disambiguity, Device command execution.
- Personalization: Long-term memory, User profile & Personalization tool.
- Includes Physical grounding such as light bulbs and External grounding (such as weather forecast) & Personalization.


---

[Efficient Human-AI Coordination via Preparatory Language-based Convention](https://arxiv.org/abs/2311.00416)

- HAPLAN: Human-AI coordination using Conventions. Humans communicate roles & tasksof individuals before starting a task to be completed. Humans create Conventions.
- Builds a Convention (an action-plan) to guide AI/human using task requirements, human preferences, number of agents and other information for a better understanding of tasks & responsibilities of each agent/human.
- Assigns sub-problems to own sessions. Convention is first confirmed with human.


---


#### 31st of October 2023

[Generating Sequences by Learning to Self-Correct](https://arxiv.org/abs/2211.00053)

- Self-Correction: A generative LLM, which includes two modules: Generator and Corrector. 


---

[Autonomous Robotic Reinforcement Learning with Asynchronous Human Feedback](https://arxiv.org/abs/2310.20608)

- Autonomously explores real world
- Guided Expliration for Autonomous Reinforcement learning (GEAR): approaches objective by meeting promising sub-goal close to final target (Goal Selector), but reachable from current position using current policy (Density model).
- Crowdsourced & Occasional comparative feedback regards user objective vs. available correct/incorrect states.


---

[Towards A Natural Language Interface for Flexible Multi-Agent Task Assignment](https://arxiv.org/abs/2311.00153)

- Programs constraints into task assignments system based on natural language using Multi-agent LLMs.


---

[Leveraging Word Guessing Games to Assess the Intelligence of Large Language Models](https://arxiv.org/abs/2310.20499)

- DEEP: Uses agressive (truthfull) & conservative modes (to disguise) to play spy game to asses intelligence of LLMs to describe target word without stating explicitly the word.


---

[Multi-Agent Consensus Seeking via Large Language Models](https://arxiv.org/abs/2310.20151)

- Consensus within multi-agent reason mainly reason and change their numerical value state based on consensus strategy based on average strategy.


---

#### 26th of October 2023

[CompeteAI: Understanding the Competition Behaviors in Large Language Model-based Agents](https://arxiv.org/abs/2310.17512)

- Studies competition of LLM agents and identifies research on competition of LLM agents, as important as co-operation.
- The initial advantage of a LLM agent leads to feedback creating cycle for Matthew's effect.
- LLM Agents can operate in competitive environment. 
- LLM Agents learn to imitate and differentiate with other LLM agents. 


---

#### 25th of October 2023

[PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization](https://arxiv.org/abs/2310.16427)

- PromptAgent: Optimizes prompts using planning algorithms such as MCTS.
- Creates intermediate prompts, updates them based on error feedback, simulates future rewards and searches higher reward paths.
- Prompts generated include: Domain knowledge, Task description, Term clarification, Solution Guidance,Exception handling, Priority & Emphasis, Formatting


---

#### 24th of October 2023

[RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models](https://arxiv.org/abs/2310.16340)

- Key-value store for observation retrieval, parsed actions are executed by RCAgent or by Expert Agent.


---

[Diverse Conventions for Human-AI Collaboration](https://arxiv.org/abs/2310.15414)

- Mixed-play: generates diverse conventions (arbitrary solutions to reocurring cooperation problems) by randomly switching between self-play (maximize award) and cross-play (Minimize) actions to maxime mixed-play.
- CoMeDi (Cross-play optimized, Mixed-play enforced Diversity) algorithm is explained [](https://www.youtube.com/watch?time_continue=30&v=wm4f0sdKIUA&embeds_referring_euri=https%3A%2F%2Filiad.stanford.edu%2F&source_ve_path=MzY4NDIsMjg2NjY&feature=emb_logo).


---

[Woodpecker: Hallucination Correction for Multimodal Large Language Models](https://arxiv.org/abs/2310.16045)

- Woodpecker: To extract key concepts, formulate questions and validate visual knowledge and generate visual claims using Multimodal Large Language Models (MLLMs) to control hallucinations in LLM responses.


---

[In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916)

- Training data used with LLMs is compressed into task vectors within LLM. Task vectors are used in 18 tasks.


---

[Instruct and Extract: Instruction Tuning for On-Demand Information Extraction](https://arxiv.org/abs/2310.16040)

- On Demand Information Extraction (ODIE): Extracting information using LLMs from text to present it in structured tabular format.


---


#### 23th of October 2023

---

[Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213)

- LLMs include Function Vectors (FCs) to trigger functions in different contexts.


---

[LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay](https://arxiv.org/abs/2310.14985)

- Explores social behaviour or LLMs in Avalon-game regards team working and other collaboration.
  

---


#### 20th of October 2023

<div id="toolchain"></div>

[ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search](https://arxiv.org/abs/2310.13227)

- ToolChain*: Uses A ∗ search algorithm to navigate an action space as a tree-like structure with LLM agent.
- Selects most promising path, Expand follow up actions in the selected path, Update the tree-structure.


--- 

[Democratizing Reasoning Ability: Tailored Learning from Large Language Model](https://arxiv.org/abs/2310.13332)

- Student LM takes an “exam” to gather mistakes it made. Teacher LM generates training data based on the mistakes. Teacher LM customizes each "exam" the feedback. Student LM learns to improve with self-reflection on its mistakes made and the new training data provided by the teacher LM. These steps are repeated until Student LM has reacher Teacher LM capability.


---


#### 19th of October 2023

[AgentTuning: Enabling Generalized Agent Abilities for LLMs](https://arxiv.org/abs/2310.12823)

- AgentTuning: Improves LLM capability by Instruction Tuning to user tasks by using AgentInstruct-dataset to create AgentLM using AgentTuning.


---


#### 18th of October 2023

[Language Agents for Detecting Implicit Stereotypes in Text-to-image Models at Scale](https://arxiv.org/abs/2310.11778)

- Language agent to automatically identify ans quantify extent of generated images.
- Planning and Reasoning. Tool usage: Intent understanding, Instruction generation, Instruction retrieval, Prompt optimization & Stereotype score generation.

---


#### 17th of October 2023

[Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://arxiv.org/abs/2310.11441)

- Set-of-Mark (SoM)-visual prompting technique to answer questions by partioning image into regions with different level of granularity and insert numbers for each region.
- Studies VLM model prompting techniques. 

---

[VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454)

- VeRA


---

[The next grand challenge for AI](https://www.ted.com/talks/jim_fan_the_next_grand_challenge_for_ai/transcript)

- Foundational Agent: Agents, which scale in all three axis of: skills, embodiment and realities.  If chatgpt was scaled with data, foundational agents are scaled with realities.


---


#### 16th of October 2023

[Character-LLM: A Trainable Agent for Role-Playing](https://arxiv.org/abs/2310.10158)

- Character-LLM: simulates historical figures using LLMs, which mimick profile / experiences and emotional states of specific individuals.
- Applies "Experience Reconstruction" with detailed experiences and memories.
- Specialises a base model for character generation.
- Evaluates using step-by-step LLM-judge aproach by evaluating one dimension at each step.

---

[OpenAgents: An Open Platform for Language Agents in the Wild](https://arxiv.org/abs/2310.10634)

- OpenAgents-platform: Data agent, Plugin/Tools and Web agent
- Automatic tool selection from over 200 tools


---

[Improving Large Language Model Fine-tuning for Solving Math Problems](https://arxiv.org/abs/2310.10047)

- Introduces multi-task sequential fine-tuning method, where solution generation is improved by including solution evaluation as part of the fine-tuning objective together with the generated solution to provide higher-quality guidance to solution generator.
- Quality and style of the step-by-step solutions used for fine-tuning impact model performance. Solution re-ranking and Majority voting used together are effective way to improve model performance with fine-tuning.


---

[CLIN: A Continually Learning Language Agent for Rapid Task Adaptation and Generalization](https://arxiv.org/abs/2310.10134)

- A Continually Learning Generative Agent from Interactions (CLIN): Memory generator updates memory, Controller manages tasks and Executor converts it into actions towards the goal. 

---

[Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701)

- LLM-based agent manages complex multi-agent collaboration task with performance level comparable with RL agent. 


---

#### 13th of October 2023

[A Zero-Shot Language Agent for Computer Control with Structured Reflection](https://arxiv.org/abs/2310.08740)

- Zero-shot agent plans executable actions in the environment and iteratively progresses by learning from mistakes using  self-reflection and structured thoughts management.
- Better generalization, outperforms best iterative-planning agents


---


#### 12th of October 2023

[AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems](https://arxiv.org/abs/2310.09233)

- AgentCF: LLM agent-based recommender system with Use and Item Agents.
- User & Item Agents interact autonomously and the discrepancies between the two are stored in the memory to help guide better future recommendations.


---

[Octopus: Embodied Vision-Language Programmer from Environmental Feedback](https://arxiv.org/abs/2310.08588)

- Octopus: Uses Vision-Language Model with Reinforcement Learning from Environmental Feedback (RLEF).
- Generates action sequences and executable code.


---

[MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)

- MemGPT: OS-based design with LLM-processor managing its actual context and long term memory and uses functions to make changes and events to manage order of processing data.


---

[Promptor: A Conversational and Autonomous Prompt Generation Agent for Intelligent Text Entry Techniques](https://arxiv.org/abs/2310.08101)

- Promptor: Automatic prompt generation.
- Builds prompts based on: User goals, User Profiles, Data Profile, Contextual nformation & Output constraints
- System prompt includes: instructions, Actions, Facts and Examples.


---

[Towards Robust Multi-Modal Reasoning via Model Selection](https://arxiv.org/abs/2310.08446)

- Dynamic model selection by taking into account input & sub-task dependencies.


---


#### 11th of October 2023

[The Temporal Structure of Language Processing in the Human Brain Corresponds to The Layered Hierarchy of Deep Language Models](https://arxiv.org/abs/2310.07106)

- Evidence about strong correlation between layers activated in Deep Language Models (DLMs) and human brain high-order language areas: auditory,syntactic and semantic areas. 
- Brain and DLMs both process input into multi dimensional vector embeddings, processed as sequences taking into account the context.
- Identifies differences. One difference is, that human brain does not perform straightforward linear interpolation between the previous and current words, suggesting RNNs may better mimick human brain language processing. The other difference is, that humans do not learn only by reading text, but use data from multiple modalities.


---

[Empowering Psychotherapy with Large Language Models: Cognitive Distortion Detection through Diagnosis of Thought Prompting](https://arxiv.org/abs/2310.07146)

- Diagnosis-of-Thought: Cognitive distortion detection through prompting: Subjective assessment, contrastive reasoning and schema analysis.

---

[LangNav: Language as a Perceptual Representation for Navigation](https://arxiv.org/abs/2310.07889)

- Uses BLIP to make imgae caption and DETR for object detection on image views to to obtain text descriptions, which a LLM agent uses to generate navigation instruction.

---

#### 10th of October 2023

[Towards Mitigating Hallucination in Large Language Models via Self-Reflection](https://arxiv.org/abs/2310.06271)

- Self-Reflection: Introduces self-reflection prompting, similar to "Reflection"-prompting. Evaluates via LLM-loom, if the answer knowledge is factual enough and in second loop, if the answer is enough consistent.
- Human reviewers are asked to evaluate sentence in answer in case is generic, fact-inconsistent or fact-consistent. The user is as well asked to categorise answer to be question-inconsistent(inconsistent), tangential (consistent, but not on topic) or answerable (consistent and answers).


---


#### 9th of October 2023

[FireAct: Toward Language Agent Fine-tuning](https://arxiv.org/abs/2310.05915)

- Fine-tuning LLMs with agent trajectories for better autonomous agents.


---


#### 8th of October 2023

[Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading](https://arxiv.org/abs/2310.05029)

- MemWalker: navigates long-context iteratively and construct memory as treelike structure.


---


#### 7th of October 2023

[Crystal: Introspective Reasoners Reinforced with Self-Feedback](https://arxiv.org/abs/2310.04921)

- Introspective reasoning of the knowledge.


---

[Self-Supervised Behavior Cloned Transformers are Path Crawlers for Text Games](https://arxiv.org/abs/2312.04657)

- PathCrawling: Crawl all paths leading to reward (train LLM with these paths) and Evaluate generality to unseen task. Continue crwaling most general paths.


---


#### 6th of October 2023

[Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406)

- Language Agents Tree Search (LATS): Self-Refine, Memory, Reasoning, Decision Making & Planning.
- Uses multiple reasonining paths and learns from experience by integrating external feedback & self-reflection.

---

[BrainSCUBA: Fine-Grained Natural Language Captions of Visual Cortex Selectivity](https://arxiv.org/abs/2310.04420)

- BrainScuba (Semantic Captioning Using Brain Alignments): LLM generates interpretable captions.
- Aligns brain activity pattern with semantic content to generate captions to explain how brain processes visual information.
- Collects brain imaging data fMRI when human views visual stimuli and uses BERT to obtain semantic reprensentation in natural language, which is based on alignment process. This process maps images to voxel-wise brain activations.
  

---


#### 5th of October 2023

[Agent Instructs Large Language Models to be General Zero-Shot Reasoners](https://arxiv.org/abs/2310.03710)

- AgentInstruct: generates instructions for th problem and then solves it using these instructions, improving the Chain of Thought (CoT) zero-shot reasoning.


---


#### 5th of October 2023

[Balancing Autonomy and Alignment: A Multi-Dimensional Taxonomy for Autonomous LLM-powered Multi-Agent Architectures](https://arxiv.org/abs/2310.03659)

- Characteristics of Autonomous Agents: Goal-driven task management, Intelligent Agents with LLMs, Multi-Agents collaboration, Context interaction, Balancing Autonomy vs. Alignment.


---

[DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)

- DSPy programs (think Langchain as cmparison) help create LLM pipelines, which can outperform few-shot prompting techniques.
- Help improve mathe world problems or answering complex questions and manage chaining / loops.


---


#### 3rd of October 2023

<div id="stop"></div>

[Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation](https://arxiv.org/abs/2310.02304)

- Self-Taught Optimizer (STOP): Ask LLM to improve initial program by providing improvement candidates and then output best solution.


---

[Lyfe Agents: Generative agents for low-cost real-time social interactions](https://arxiv.org/abs/2310.02172)

- LyfeAgents Brain: Sensory processing, Internal states, Self-monitor, Action selection and Memory.
- Internal states are text based: current goal, memory, recent events and sensory inputs. 
- Cognitive controller selects high-level actions. Action model selects actions until termination condition is reached.
- Self-monitoring maintains and emphasizes recent and novel events towards agent goals
- Memories are clustered and summarized before moving them to long-term storage (vector database)


---

[EcoAssistant: Using LLM Assistant More Affordably and Accurately](https://arxiv.org/abs/2310.03046)

- EcoAssistant: Enables LLM agent to converse with code executor to iteratively produce answers based on code produced. Hierachical structure, where cheaper and weaker LLM is used before trying the stronger and expensive LLM.
- Surpasses GPT-4 10% in performance with 50% less cost.
  

---

[Large Language Models as Analogical Reasoners](https://arxiv.org/abs/2310.01714)

- LLM self-generates examples/knowledge related to the task.


---

[Conceptual Framework for Autonomous Cognitive Entities](https://arxiv.org/abs/2310.06775)

- Conceptual framework for Autonomous entities.


---

[OceanGPT: A Large Language Model for Ocean Science Tasks](https://arxiv.org/abs/2310.02031)

- DoInstruct (Domain Instruction): Automatically gathers large amount of domain specific instruction data for multi-agent collaboration.
- Domain Instruction generation: Agents used as experts in each topic. Instructions are augmented rapidly through agent collaboration, which are annotated and finally inspected for high quality fine-tuning dataset. 
  

---


#### 2nd of October 2023

[Enabling Language Models to Implicitly Learn Self-Improvement](https://arxiv.org/abs/2310.00898)

- ImPlicit Self-ImprovemenT (PIT)-framework: introduces self-improvement, where LLMs self-improve its response quality with human preference data without extensive human annotation.


---


[SmartPlay : A Benchmark for LLMs as Intelligent Agents](https://arxiv.org/abs/2310.01557)

- SmartPlay: a benchmark to test LLM-based agents from 9 perspectives.
- Tests: Reasonning with object dependencies, planning ahead, spatial reasoning, learning from history, and understanding randomness. 


---

[GRID: A Platform for General Robot Intelligence Development](https://arxiv.org/abs/2310.00887)

- GRID: General Robot Intelligence Development
- Solves complex tasks using simulatiom and/or real-world data
- Task specification, robot configuration and sensor/API.
- Foundation Mosaic: a neural architecture.


---


#### 1st of October 2023

[RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models](https://arxiv.org/abs/2310.00746)

- RoleLLM: Role-profile constructor, Context-based Instruction generarion, Role-based Prompting(RoleGPT), Role-conditioned Instruction-tuning.


---

#### 29th of September 2023

[AutoAgents: A Framework for Automatic Agent Generation](https://arxiv.org/abs/2309.17288)

- AutoAgents: Planner agent receives user input and converts it into a plan. Multiple agent roles take actions in this plan to convert into a result. 
- Observers: Observer agent reviews, if the created agent roles meet the requirements. Plan observer agent reviews, if the plan meets expectations. Action observer reviews, if the action response meets expectations.
- Includes drafting stage (with agent observer and plan observer agents) and Execution stage (with action observer).


---

[Motif: Intrinsic Motivation from Artificial Intelligence Feedback](https://arxiv.org/abs/2310.00166)

- Motif: Trains a reward fucntion/model from pairs of gameplay captions and LLM observations of these game actions. Then train an agent using RL with the reward model.
- Diverse behaviours triggered with the LLM improve in performance in specific domain: for example Gold Collector collects more cold.


---


#### 28th of September 2023

[Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution](https://arxiv.org/abs/2309.16797)

- Promptbreeder uses thinking styles and mutation-prompts and is able to improve mutation/task prompts.


---

#### 24th of September 2023

[Let's reward step by step: Step-Level reward model as the Navigators for Reasoning](https://openreview.net/forum?id=RSQL6xvUYW)

- Heuristic Greedy Search for Process-Supervised Reward Model (HGS-PRM): each new reasoning step generated by the LLM is evaluated by the reward model, if to accept the reasoning step or generate a new one until the reasoning path is identified.
- Creates PRM-Code dataset using Code-LLaMA-7B using Mutating testing-technique. 


---


#### 23th of September 2023
[Natural Language based Context Modeling and Reasoning with LLMs: A Tutorial](https://arxiv.org/abs/2309.15074)

- LLM-driven Context-aware Computing (LCaC) approach.


---


#### 20th of September 2023

[You only look at the screens: Multimodal Chain-of-Action Agents](https://arxiv.org/abs/2309.11436)

- Multimodal Chain-of-Actions Agents (Auto-UI) interacts directly with the UI
- Chain-ofAction technique using series of action histories and future action plans.

---


#### 18th of September 2023

[MindAgent: Emergent Gaming Interaction](https://arxiv.org/abs/2309.09971)

- MindAgent: Planning skills and Tools use(Agent location, Tool state, Agent holdings, Pending dishes, Timer), LLM dispatcher, Memory history (Environment, Agent State, Actions and Feedback) and Action module(Controller, Human actions, Action validator, Action Types/Patterns/Names).
- Introduces CuisineWorld-benchmark, where multiple agents play game simultaneously through multi-agent collaboration.


---


#### 14th of September 2023

<div id="llmagentsurvey"> </div>

[The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/abs/2309.07864)

-  A conceptual framework for LLM-based agents with three components brain, perception, and action.


---

[Agents: An Open-source Framework for Autonomous Language Agents](https://arxiv.org/pdf/2309.07870.pdf)

- Multi-agent: Planning, memory, tool usage, multi-agent communication & symbolic control.
- Open source library.


---

<div id="physicalgrounding"> </div>


#### 13th of September 2023

[Physically Grounded Vision-Language Models for Robotic Manipulation](https://arxiv.org/abs/2309.02561)

- PhysObjects dataset for physical grounding.
- VLMs with PhysObjects improves its understanding on physical objects.
- Improves task success rate.


---


#### 12th of September 2023

[Life-inspired Interoceptive Artificial Intelligence for Autonomous and Adaptive Agents](https://arxiv.org/abs/2309.05999)

- Interoceptive AI: monitoring own internal state of the artificial agent.


---

[Textbooks Are All You Need](https://www.youtube.com/watch?v=24O1KcIO3FM)

- Sebastien Bubeck explains the insights from the reserch on Phi-1 regards coding tasks and Phi-1.5. regards reasoning tasks and the models being able to outperform 1000 times larger LLMs.
- The talk highlights, that the key ingredients on Textbook-like training data and then giving then giving Exercises.
- Explains the the key ingredient in "Textbooks are all you need"-paper regards the data, is largerly based on TinyStories-paper, which dataset was used to train a high performing model to generate fluent and consistent stories in English language. 


---



#### 8th of September 2023

<div id="autonomousagentssurvey"> </div>

[Unleashing the Power of Graph Learning through LLM-based Autonomous Agents](https://arxiv.org/abs/2309.04565)

- AutoGraph procedure: data, configuration, searching and tuning agents.

---


#### 28th of August 2023

[RecMind: Large Language Model Powered Agent For Recommendation](https://arxiv.org/abs/2308.14296)

- RecMind: a recommender focused LLm agent with reasoning, planning to sub-tasks, memory & tools.


---

#### 22th of August 2023

[A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/abs/2308.11432)

- Systematic review of LLM based Autonomous Agents.
- Use cases and evaluation strategies and future use cases.


---

#### 21st of August 2023

[AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors](https://arxiv.org/abs/2308.10848)

- AgentVerse: multi-agent collaborarion and individual agents social bjeaviours.


#### 18th of August 2023

<div id="got"></div>

[Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)

- Graph-of-Thoughts (GoT): Reasoning with LLM using graph-structure with intermediate steps.
- Introduces Volume-of-Tought metric to inform the scope of information carried by the LLM output.

---

[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)

- AutoGen: An open source framework, where LLM agents converse with other LLM agents either one or many, chat with humans and use tools.
- LLM agents are able to create new chats with other LLM agents.

---


[WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct](https://arxiv.org/abs/2308.09583)

- Improves math reasoning with Reinforcement Learning from Evol-Instruct Feedback (RLEIF): Upward and Downward evolution improve instructions by making questions easier or harder based on their difficulty level.


---

#### 17th of August 2023

<div id="rest"></div>

[Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998)

- Introduces Reinforced Self-Training (ReST).
- Grow step generates data from LLM, Improve step uses this filtered data to fine-tune the LLM. Repeat. 


---

[Never-ending Learning of User Interfaces](https://arxiv.org/abs/2308.08726)

- Never-ending UI Learner: automatically installs apps from an appstore and crawls them to learn difficult training examples


---

#### 3rd of August 2023

[Scaling Relationship on Learning Mathematical Reasoning with Large Language Models](https://arxiv.org/abs/2308.01825)

- Proposes Rejection sampling Fine-Tuning (RFT), which generates reasoning and collects correct ones to augment as fine-tuning dataset. 


---

#### 25th of July 2023

[WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854)

- An environment to test Autonomous agents in an environment with tools, external knowledge.

---

#### 20th of July 2023

[Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)

- Addresses LLM training data to be "text-book-like":  clear, self-contained, instructive, and balanced. The method is used in Phi-models.


---

[BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs](https://arxiv.org/abs/2307.08581)

- BuboGPT: Uses Vicuna LLM by receiving text input inserting together visual and audio inputs separately with Q-former. The Vicuna output is then processed using SAM-model for visual grounding.
- Achieves coherent and grounded descriptions

---


#### 16th of July 2023

[Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)

- ChatDev: Define task and automatically generate SW designing, coding, testing, and documentation using "Chat Chains", where LLM-based chats include different roles for each sub-task: CEO, programmer, CTO etc.
- Includes role-assignment, memory and self-reflection.  


---

[xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v3)

- Protein Language Model: xTrimoPGLM.

---


#### 14th of July 2023

[Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)

- EmotionPrompt: adds to prompt an emotional stimuli, which improves performance by 10.9%.
- An example of an emotional stimuli is to state that the work is important for career. 




#### 23rd of June 2023 

<div id="lili"> </div>

[LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)

- Lilian Weng from OpenAI article / blog post
- Covers Planning, Memory and Tool usage of LLM powevered agents


---

#### 8th June 2023

[ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases](https://arxiv.org/pdf/2306.05301.pdf)

- Builds multi-agent simulation environment to generate dataset of using many real world apis. 
- Small models can achieve comparable performance to larger models on tool usage.


---

#### 6th of June 2023

[Enabling Intelligent Interactions between an Agent and an LLM: A Reinforcement Learning Approach](https://arxiv.org/abs/2306.03604)

- When2Ask: RL agent, which learns when to query LLM for high-level plans to complete a task.
- Planner, Actor and Mediator.


---

#### 5th June 2023

[SELFEVOLVE: A Code Evolution Framework via Large Language Models](https://arxiv.org/pdf/2306.02907.pdf)

- Generates intermediate code based on input prompt. 
- Use LLM to act as expert programmer to debug the generated code by receiving errors from Python interpreter.


---

#### 3th June 2023

[Prompt Sapper: LLM-Empowered Software Engineering Infrastructure for AI-Native Services](https://arxiv.org/pdf/2306.02230.pdf)

- Human AI collaborative intelligence methodology & technical practices, where the idea is not to have "full Auto-GPT" from user input to direct resolution by LLM, but rather human reviews steps between.
- Useer inputs objective, LLM asks clarification. Use then  User adds clarifications and LLM constructs AI chain for human to review. Finally LLM executes the AI chain with user acceptabnce tests.


---

#### 3th June 2023

[Auto-GPT for Online Decision Making: Benchmarks and Additional Opinions](https://arxiv.org/pdf/2306.02224.pdf)

- Auto-GPTs outperforms supervised state-of-the-art Imitiation Learning (IL) models with GPT4 in WebShop- and ALFWorld-benchmarks in unknown external environments.
- Additional opinions algorithm improves performance, which takes into account additional opinions from external expert models.


---

#### 2nd of June 2023

- MathChat: Describes a solid conversational MATH problem solving in four step process.
- Describes the prompts used.

---

#### 26th of May 2023

[Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Large Language Models](https://arxiv.org/abs/2305.16582)

- Graph-of-Thought (GoT) reasoning: To model human thought process as graph instead of chain to improve LLM reasoning capability.


---

[Impossible Distillation: from Low-Quality Model to High-Quality Dataset & Model for Summarization and Paraphrasing](https://arxiv.org/abs/2305.16635)

- Uses low-quality LM to generate High-quality dataset (more diverse and more effective for generalization in unseen domains) to train a high quality model: 770 million parameter model outperforms GPT-3 in multiple tasks evaluated by humans.

---


#### 25th of May 2023

[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)

- Voyager: open-ended embodied agent with LLM

---

#### 24th May 2023

[Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)

- RAP (Reasoning via Planning): Uses LLM as both world model and reasoning LLM-agent. Integrates MCTS search planning algorithm.
- Incrementally generates reasoning tree with LLM in domains of plan generation, math reasoning and logical inference.

---

[Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334)

- Gorilla is a retrieve-aware finetuned LLaMA-7B model for API calls using self-instruct to generate Instruction-API pairs. 


---

#### 18th of May 2023

[Think Outside the Code: Brainstorming Boosts Large Language Models in Code Generation](https://arxiv.org/abs/2305.10679)

- Brainstorm: uses brainstorming step to generate and select diverse thoughts in code generation.
- Uses three steps: brainstorming, thought selection (trains a thought ranker for this) and writing code.



#### 17th May 2023

<div id="tot"></div>

[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) 

- Tree of Thoughts (ToT)-technique makes decisions using multiple different reasoning paths, self-evaluating choices to decide next action with ability to look back/forward for global decisions.


---

[Mobile-Env: Building Qualified Evaluation Benchmarks for LLM-GUI Interaction](https://arxiv.org/abs/2305.08144)


---

#### 13th of May 2023

[BabyCatAGI: Fast and Feline](https://yoheinakajima.com/babycatagi-fast-and-feline/)

- BabyCatAGI: a modified BabyAGI by replacing  task manager in BabyBeeAGI with task creation agent running once.
- Uses Intelligent Agent Tool to combines tools to extract only relevant information to next step such as looping web search and scraping results to pull only specific part to another task.


### 12th of May 2023

[TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)

- A breakthrough paper, where synthetic data generated by Teacher-Student LLM is used to train a high-performing model to generate fluent and consistent English stories.
- Demonstrated the effectiveness of synthetic data in smaller LLMs challenging large SOTA models in domain of English language.
- Uses GPT-4 to grade content generated by the models as if created by student and being graded by the GPT-4 teacher.

---

### 9th of May 2023

[ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665)

- ImageBind: a joint embedding space for images, text, audio, depth, thermal and IMU data modalities-

---

#### 3rd of May 2023

[Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings](https://arxiv.org/abs/2305.02317)

- Introduces Visual Chain of Thought (VCoT) for data augmentation, where between reasoning steps multimodal data is infilled to obtain better reasoning results.


---

#### 30th of April 2023

[BabyBeeAGI: Task Management and Functionality Expansion on top of BabyAGI](https://yoheinakajima.com/babybeeagi-task-management-and-functionality-expansion-on-top-of-babyagi/)

- BabyBeeAGI: a modified from BabyAGI tracking statuses of tasks, task dependencies, identification of required new tasks, assigning tools and results in json-format.


---

<div id="consciousnesstest"> </div>  

# 26 of April 2023

["Inside OpenAI [Entire Talk" by Stanford eCorner](https://www.youtube.com/watch?si=nMlyq1_d0r9JQkJ0&v=Wmo2vR7U9ck&feature=youtu.be)

- Interview of Ilya Sustskever, where defined a way to perform "a consciousness test" from a very controlled dataset, see "minute 15".
 
---

#### 21st of April 2023

[Improving Grounded Language Understanding in a Collaborative Environment by Interacting with Agents Through Help Feedback](https://arxiv.org/abs/2304.10750)

- LLM agent self-help with LLM to complete IGLU tasks using clarifying questions.
- 
---

#### 13th of April 2023

[RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment](https://arxiv.org/abs/2304.06767)

- RAFT-finetuning: Samples batch lf data from LLM, reward function scores them, high reward examples are filtered as data to finetune the LLM.


---

#### 11th of April 2023

[ChemCrow: Augmenting large-language models with chemistry tools](https://arxiv.org/abs/2304.05376)

- Uses LLM and chemistry tools to plan and execute different chemical tasks. 
- Tools include web and literature search, Python, human-tool to interact with the end user and various molecule tools, safety tools and chemical reaction tools.

---

[Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128)

- The model generates new code together with code explanation. The code is then executed and this executed code is sent back as feedback together with the code explanation. This feedback

---

#### 7th of April 2023

[ChatPipe: Orchestrating Data Preparation Program by Optimizing Human-ChatGPT Interactions](https://arxiv.org/abs/2304.03540)

- ChatPipe - Iterative, data preparation program with ChatGPT using 1. Operation Recommendation, 2.   Program generation, 3. Version management. 
- Recommends next data preparation opration. Easily roll-back to previous program for version control.


---

#### 6th April 2023

[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)

- Enable believable human behavior: observation, planning, and reflection.
- An agent wants to throw a Valentine’s Day party. The agents autonomously spread invitations, make new acquaintances, ask each other out on dates to the party, and coordinate to show up for the party together at the right time. 
- [GPTeam](https://github.com/101dotxyz/GPTeam) is inspired by this approach.


---

#### 31 March 2023

[CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society](https://arxiv.org/abs/2303.17760)

- CAMEL attempts to facilitate autonomous cooperation among communicative agents through role-playing framework.
- The approach manages complete tasks with minimal human input.


---

#### 30th of March 2023

[Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

- Self-Refine refers to Iterative refinement with self-feedback: use the LLM to get Feedback to original output, which is passed back to LLM to Refine a new output.
- The concept is best understood here in the blog by : [Self-Refine: Iterative Refinement with Self-Feedback](https://selfrefine.info/) with GIFs and code examples.
- Improves base-model performance in tasks like math reasoning and code generation. 


---

[HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/abs/2303.17580)

- A LLM (such as ChatGPT) accesses HuggingFace community to look AI models to complete the given task. 
- It can read multi modalities by outsourcing tasks like image recognition to the specific image model. 


---

[DERA: Enhancing Large Language Model Completions with Dialog-Enabled Resolving Agents](https://arxiv.org/abs/2303.17071)

- Dialog-Enabled Resolving Agents (DERA) uses two roles: Researcher and Decider to perform discussion between these two agents.
- Researcher role processes information and Decider role uses judgement.


---

#### 29th of March 2023

[TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs](https://arxiv.org/abs/2303.16434)

- Multimodal conversational foundation model (MCFM). MCFM generates a textual solution outline, then API selector chooses most relevant API from collection of APIs (with API name, parameter list, description, usage example and example when combining it with another API). 
- MCFM generates action code using recommended API and the API call is executed. Finally, output is provided back to developer.


---


#### 28th March 2023 

[Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)

- Task-driven autonomous agent, with vector database and Langchain. BabyAGI includes: Execution, creation and prioritization
- Takes objective, pulls an item from task queue and moves it to execution agent with access to memory.  


---

[Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712)

- Raises an argument, that GPT-4 model capabilities should be reviewed as an early and incomplete version of Artificial General Intelligence (AGI) systems due the multiple metrics comparing against human level-performance.
- Raises the argument, that LLMs need to move beyond "next-word prediction" to overcome linear reasoning limitation, which often is possible to solve as incremental tasks with few iterations.


---

#### 20th March 2023

[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

- Reflexion agents reflect on task feedback, use it from memory to make better decisions and new attempts.

---

[Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673)

- EcoOptiGen: Hyperparameter tuning of LLMs.


---

[Improving Multimodal Interactive Agents with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2211.11602)


---


#### 27th of February 2023

[Reward Design with Language Models](https://arxiv.org/abs/2303.00001)

- LLM-RL: framework uses a LLM as a proxy reward function to train reinforcement learning (RL) agents.
- User specifies objective with natural language prompt, LLM evaluates agent's behavior, and framework is agnostic to RL algorithm.
- This approach simplifies reward design and enables training of agents aligned with user objectives.


---


#### 8th of December 2022

[LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models](https://arxiv.org/abs/2212.04088)

- LLM-Planner: Uses LLM for few-shot planning with embodied agents based on natural language and visual perception of the environment.
- Improves planning with physical grounding to create and update plans.
- Includes task introduction/goal instruction/step-by-step instructions/plan list//object list/retrieval message (next plan).

---

#### 20th of October 2022

[Large Language Models Can Self-Improve](https://arxiv.org/abs/2210.11610)

- Demonstrates LLM is able to Self-Improve with only unlabeled datasets using CoT and Self-Consistency Prompting and then fine-tune the LLM using these self-generated solutions as target outputs.
- This research by Google, effectively performs Self-Recursive Learning not only during Inference time (such as CoT or In-Context Learning alone), but training as well.


---

#### 31st of August 2022

<div id="emerging"></div>

[Emergent Abilities of Large Language Models](https://openreview.net/forum?id=yzkSU5zdwD)

-  Defines officially the term  "Emergent Abilities": "An ability is emergent if it is not present in smaller models but is present in larger models."
-  Emergent abilities were detected already with GPT-3, but here its clearly defined as ability detected only after specific scale.
-  Identifies a list of Emerging abilities not detected in specific smaller model, but identfied in a larger model.
-  I like the paper, because increasing number of task patterns are learned using single learning objective of next-word prediction as scale increases.


---

<div id="generalistagent"></div>

#### 12th of May 2022

[A Generalist Agent](https://arxiv.org/abs/2205.06175)

- Gato: A multi-modal, multi-task, multi-embodiment generalist policy agent.
- Learns to play Atari, caption images, chat, stack blocks with robot arm, etc. 
- Includes text tokens, image patch tokens, agent timesteps and action tokens.
- Argues, that "a generalist agent that can adapt to new embodiments and learn new tasks with few data."

---

#### 19th of April 2022

<div id="worldmodel2"></div>


[Deep learning, reinforcement learning, and world models](https://www.sciencedirect.com/science/article/pii/S0893608022001150)

- Reviews Deep learning, Reinforcement learning and World models.
- Claims humans use World model as simulators in the brain, learned through senso-motory interaction with the environment. It is possible to learn world model using deep generative models.




<div id="star"></div>

---


#### 28th of March 2022

[STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)

- Introduces the concept: "Self-Taught Reasoner" (STaR) or *, where LLM improves its reasoning by learning from its own reasoning: model is asked to generate rationalizations to questions. If rationalization derives wrong answer to question, the rationalization is repeated by giving it as well the correct answer. All rationalizations leading to correct answer are used for fine-tuning the LLM model. This process is repeated and each iteration improves the LLMs capability of reasoning.
- The paper does not refer to Self-Recursive Learning, but we could argue it as an example of this process in the context of reasoning.


---

#### 21st of March 2022

<div id="selfconsistency"></div>

[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

- Enables reasoning with LLMs using CoT and Self-Consistency, where multiple, different reasoning paths are used to vote the most consistent answer.
- Improves reasoning and math problem solving.


---

[Chain of Hindsight Aligns Language Models with Feedback](https://arxiv.org/abs/2302.02676)

- Chain of Hindsight (CoH): Humans learn from feedback, which is converted sequences of sentences, ranked with human preferences and used to fine-tune the LLM.

---

#### 7th of March 2022

[Shared computational principles for language processing in humans and deep language models](https://www.nature.com/articles/s41593-022-01026-4)

- Provides evidence  about three computational principles, shared both by Deep Language Models (DLMs) and human brain to process language.
- The three principles are: continuous next-word prediction, contextual embeddings and surprise prediction error.

  
---

#### 28th of January 2022

<div id="cot"></div>

[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

- Defines Chain-of-Thought (CoT).
- CoT is one Emerging Ability not present in smaller models, but present in larger models.
- CoT can be seen as Self-Recursive Learning, where the LLM improves its own output by having LLM use intermediate steps to solve complex task.
- The approach effectively demonstrates the LLMs capability to perform Self-Recursive Learning, altough its not integrated back as training data of the model.

---

#### 12th April 2021

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

- RAG (Retrieval-Augmented Generation): introduces retrieval-augmented generation models, with Query Encoder, Retriever, Document Index, and Generator, for knowledge-intensive NLP tasks.
- RAG framework combines parametric memory (pre-trained seq2seq model) and non-parametric memory (Wikipedia index) to improve generation quality.
- RAG models achieve state-of-the-art results on open domain question answering tasks, outperforming parametric and task-specific architectures.


---

<div id="languageagentdefinition"></div>


#### 26th of March 2021


[Alignment of Language Agents](https://arxiv.org/abs/2103.14659)

- Defines Language Agent. 



---

<div id="qstar"></div>

#### 8th of February 2021

[A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks](https://arxiv.org/abs/2102.04518)

- Q* search algorithm: Better version of A* search algoirthm, because reduces computation time and number of nodes to be computed.

---

#### 28th of May 2020 

<div id="multitask"></div>

[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

- Applies first-time the term of LLMs ability to learn a task from contextual information: "In-Context Learning".
- This ability is another example of Self-Recursive Learning, altough its not integrated back as training data of the model.
- This paper as well identified the capability of LLMs to learn multiple tasks by having been only trained to predict the next word. See Jason Wei´s presentation included below, where he covers the "Massively Multi-task learning" of LLMs and I think it helps to gain better insight about LLMs, rather than thinking them as simply "statistical models". 

---

#### 22th of May 2020

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

- Defines Retrieval-Augmented Generation (RAGs).

---

#### 12th of November 2020

<div id="rewardisenough">  </div>

[Reward is enough](https://www.sciencedirect.com/science/article/pii/S0004370221000862)

- Reward is sufficient to drive intelligent behaviours instead of requiring special formulations.
- Agents could learn to obtain various intelligent behaviours through trial and error experiences to maximize reward.
- Sophisticated intelligence may emerge from simple objective, think what an animal is able to learn to do just by being in hungry.


---


#### 24th of November 2019

[Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms](https://arxiv.org/abs/1911.10635)

- MARL: Introduces Multi-Agent Reinforcement Learning (MARL).


<div id="resourcecloud">  </div>

#### 28th of July 2005

[The Emotion Machine. Draft.](https://web.media.mit.edu/~minsky/eb1.html)

- Human mind consists according to Minsky, from Cloud of Resources turnable on/off.
- Important theory, because LLM agents can construct such resources, observed in a human brain, altough years after this theory.

---


<div id="autonomousagentdefinition">  </div>

#### 12th of August 1996 

[Is it an Agent, or Just a Program?: A Taxonomy for Autonomous Agents.](https://www.researchgate.net/publication/221457111_Is_it_an_Agent_or_Just_a_Program_A_Taxonomy_for_Autonomous_Agents)

- "Autonomous agent is a system situated within and a part of an environment that senses that environment and acts on it, over time, in pursuit of its own agenda and so as to effect what it senses in the future."
- Definition includes: 1. Operate within an environment, 2. Sense and Act, 3. Over time, 4. Control its own agenda (Autonomous).
- Studies the multiple previous definitions of Agents / Autonomous Agents, although the perspective is +27 years ago and prior to LLMs. 

---


[Prediction and Adaptation in an Evolving Chaotic Environment](https://arxiv.org/abs/adap-org/9306005)

- Defines the concept of "Predictive Agent" as adaptive predictors.

---

[A Learning Algorithm that
Mimics Human Learning](https://www.santafe.edu/research/results/working-papers/a-learning-algorithm-that-mimics-human-learning)

- Reviews Artificial Agents learning like humans.

---

<div id="astarssearch">  </div>

#### 24th of November 1967


[A formal Basis for the Heuristic Determination of Minimum Cost Paths](https://ai.stanford.edu/%7Enilsson/OnlinePubs-Nils/General%20Essays/roboticsandai.pdf)

- A* search algorithm.
- Defines the A* search algorithm for the first time, widely used in RL as planning algorithm.



---

## Citation


How to cite my work?



```
@misc{MaattaAutonomousAgents2023,
  author = {Teemu Maatta},
  title = {Autonomous Agents},
  year = {2023},
  howpublished = {\url{https://github.com/tmgthb/Autonomous-Agents}},
  note = {Accessed: YYYY-MM-DD}
}

```

---



[Back to top](#topofthepage)
