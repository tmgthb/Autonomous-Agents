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

[![Hits](https://hits.sh/github.com/tmgthb/Autonomous-Agents.svg?view=today-total&label=Views&color=007ec6)](https://hits.sh/github.com/tmgthb/Autonomous-Agents/)
[![X](https://img.shields.io/twitter/follow/Teemumtt3?style=social)](https://twitter.com/Teemumtt3)
[![GitHub Repo stars](https://img.shields.io/github/stars/tmgthb/Autonomous-Agents?style=flat-square)](https://github.com/tmgthb/Autonomous-Agents/stargazers)

</div>

<p align="center">
  <img height="100" src="https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_agent_logo.png" alt="Autonomous Agents">
</p>

<div align="center">

  # Autonomous Agents
  Autonomous Agents-research papers. Updated daily. [Resources-section](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Resources.md)-section. 

</div>


---

<div id="researchpapers" align="center">

## Research papers: 2024

[2025 (1/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2025 (2/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (3/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025.md), [2024](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)

Chronological order. 





</div>

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

[A Novel Task-Driven Method with Evolvable Interactive Agents Using Event Trees for Enhanced Emergency Decision Support](https://arxiv.org/abs/2501.06193)

- EvoTaskTree: is a task-driven method with evolvable interactive agents using event trees for emergency decision support.
- Framework integrates task executors and task validators powered by LLMs, leverages insights from event tree analysis, and includes three crucial tasks: initiating event subevent analysis, event tree header event analysis, and decision recommendations.
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

#### 22th of December 2024

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

[Autonomous chemical research with large language models](https://www.nature.com/articles/s41586-023-06792-0)

- Coscientist: Introduces a autonomous chemical research system for autonomously designing, planning, and performing complex scientific experiments 
- Uses modular approach consisting of: Google, Planner (LLM), Python, retrieval of documentation and execution of experiments.
- Capabilities include planning chemical syntheses, optimizing reactions, and controlling liquid-handling robots.


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

- LLM-IE [LLM-based Information Extraction]: A Python package for building complete information extraction pipelines using LLMs.
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

- MPO (Mixed Preference Optimization): is a method that blends supervised fine-tuning loss with preference optimization losses to enhance training effectiveness of multimodal LLMs.
- MPO uses a novel automated preference data construction pipeline to create MMPR dataset, and explores different Chain-of-Thought approaches with multimodal input to improve reasoning performance.
- This approach demonstrates improved performance across multiple benchmarks, particularly in multimodal reasoning tasks.

---


[A dataset of questions on decision-theoretic reasoning in Newcomb-like problems](https://arxiv.org/abs/2411.10588)

- Decision-theoretic reasoning: Introduces a dataset of natural language questions on Newcomb-like problems.
- The dataset includes capability questions (unambiguous answers) and attitude questions (disagreements among decision theorists). It evaluates existing LLMs and their attitudes toward evidential decision theory (EDT) and causal decision theory (CDT). 
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

#### 4th November 2024

[EMMA: End-to-End Multimodal Model for Autonomous Driving](https://arxiv.org/abs/2410.23262)

- EMMA (End-to-End Multimodal Model for Autonomous Driving): introduces EMMA, built on Gemini, which maps raw camera data and text inputs to driving outputs including planning trajectories, perception objects, and road graph elements, leveraging Chain-of-Thought Reasoning and Generalist Capability.
- The framework recasts autonomous driving tasks as visual question answering problems, processing inputs and generating outputs in a unified language space using task-specific prompts.
- EMMA demonstrates strong performance across motion planning, 3D object detection, and road graph estimation, functioning as a generalist model capable of jointly handling multiple driving tasks.


---

#### 1st of November 2024

[DARD: A Multi-Agent Approach for Task-Oriented Dialog Systems](https://arxiv.org/abs/2411.00427)

- DARD (Domain Assigned Response Generation): LLM-based multi agent framework in multi domain & task oriented dialogue.
- Introduces dialogue manager/hotel/attraction/restaurant/train/taxi-agents, external db and dialogue state tracker.
- Uses both fine-tuned LLMs and Sonnet 3.0. Reviews differences in performance.


---

#### 31st of October 2024

[Empowering biomedical discovery with AI agents](https://www.sciencedirect.com/science/article/pii/S0092867424010705)

- Introduces AI agents for biomedical discovery, consisting of Robotic-, Database-, Reasoning-, Hypothesis-, Brainstorming-, Search Engine-, Analysis- and Experimental Planning-agents.
- Performs tasks including hypothesis generation, workflow planning, and self-assessment, integrating LLMs and machine learning tools.
- Potetial use cases include virtual cell simulation, programmable phenotype control, cellular circuit design, and therapy development.

---


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

[ALOHA Unleashed: A Simple Recipe for Robot Dexterity](https://arxiv.org/abs/2410.13126)

- ALOHA Unleashed: introduces a transformer encoder-decoder architecture with diffusion loss for dexterous bimanual manipulation tasks.
- The framework uses CNNs for image embedding, Transformer Encoder for observation encoding, Transformer Decoder for action denoising, Proprioception MLP, and Diffusion Timestep.
- This approach combines large-scale data collection with diffusion policy to achieve improved performance in challenging manipulation tasks.


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
- This framework enables more efficient and accurate reasoning in LLMs by providing a better way to measure progress in multi-step reasoning.


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

[Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079v3)

- Mooncake: introduces a KVCache-centric disaggregated architecture for LLM serving, with a KVCache-centric Conductor (Global scheduler), Cache-aware Prefill Scheduler (Manages prefill tasks), KVCache Balance Scheduler (Balances KVCache distribution), Load-balance Decoding Scheduler (Manages decoding tasks), Prefill Instance (Executes prefill stage), Decoding Instance (Executes decoding stage), Messenger (Transfers KVCache blocks), Distributed KVCache Pool (Offloads KVCache to CPU/DRAM/SSD), and Paged KVCache (KVCache in GPU VRAM), designed to maximize throughput and meet latency SLOs.
- The architecture separates prefill and decoding clusters, leveraging underutilized CPU, DRAM, and SSD resources for a disaggregated KVCache.
- Mooncake's scheduling incorporates a prediction-based early rejection policy to mitigate overload scenarios and improve resource utilization.

---


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

- GenoTEX: introduces a benchmark for automated analysis of gene expression data, providing expert-curated code and results for gene-trait association problems, encompassing dataset selection, preprocessing, and statistical analysis.
- GenoAgent: presents a team of LLM-based agents with specialized roles (Project Manager, Data Engineer, Statistician, Code Reviewer, Domain Expert) that adopt a multi-step programming workflow with flexible self-correction to collaboratively analyze genomic datasets.
- The benchmark features 1,384 gene-trait association problems, 911 datasets with 152K+ samples, and 238K lines of expert-curated code, demonstrating the potential and challenges of LLM-based methods in scientific discovery.

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

- DLLM (Dreaming with LLMs: multi-modal model RL, which uses natural hints/goals from LLM in long-horizon tasks.
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

#### 27th May 2024

[BIOLOGICAL NEURONS COMPETE WITH DEEP REINFORCEMENT LEARNING IN SAMPLE EFFICIENCY IN A SIMULATED GAMEWORLD](https://arxiv.org/abs/2405.16946)

- DishBrain / Deep Reinforcement Learning (DQN, A2C, PPO) / Active Inference Agent: introduces a comparison of learning efficiency between in vitro biological neural networks using the DishBrain system and state-of-the-art deep reinforcement learning algorithms (DQN, A2C, PPO) and an Active Inference agent in a simulated Pong game, utilizing components like Cultured Biological Neurons, HD-MEA, various input designs, Neural Networks, and a POMDP-based Generative Model.
- The DishBrain system integrates biological neural networks with in silico computation via a high-density multi-electrode array in a real-time closed-loop feedback system.
- Deep RL algorithms (DQN, A2C, PPO) were tested with different input information densities (Image, Paddle+Ball Position, Ball Position), while the Active Inference agent explored the impact of memory horizons on sample efficiency.


---


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

#### 3rd of May 2024

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

[Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)

- CoALA (Cognitive Architectures for Language Agents): introduces a conceptual framework for designing and organizing LLM-based agents, featuring modular memory components, a structured action space, and a generalized decision-making process.
- The framework positions the LLM as the core computational unit, interacting with internal memories (working, episodic, semantic, procedural) and external environments through grounding actions.
- CoALA's decision cycle involves planning (proposal, evaluation, selection) and execution, enabling agents to reason, retrieve, learn, and interact with the world.

---

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


