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


<p align="center">
  <img height="250" src="https://github.com/tmgthb/Autonomous-Agents/blob/main/Autonomous_agents_repository_logo_type.png" alt="Autonomous Agents">
</p>



<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/tmgthb/Autonomous-Agents?style=social)](https://github.com/tmgthb/Autonomous-Agents/stargazers) 
[![Twitter Follow](https://img.shields.io/twitter/follow/Teemumtt3?style=social)](https://twitter.com/Teemumtt3)

</div>  

<div id="topofthepage"> </div>

# Autonomous Agents


Autonomous Agents (LLMs). Updated daily

- [Research papers.](#papers)
- [What is an Autonomous Agent?](#what)
- [Why Autonomous Agents work?](#why)








---



<div id="papers"> </div>  


## Research

Chronological order. 



#### 9th of April 2024

[Measuring the Persuasiveness 
of Language Models](https://www.anthropic.com/news/measuring-model-persuasiveness)

- Reviews the scaling of LLMs on persuasion tasks. Finds, that Claude 3 Opus is statistically as convincing as human.


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

---

[Automatic Authorities: Power and AI](https://arxiv.org/abs/2404.05990)



---

[Autonomous Evaluation and Refinement of Digital Agents](https://arxiv.org/abs/2404.06474)


---

[Wu's Method can Boost Symbolic AI to Rival Silver Medalists and AlphaGeometry to Outperform Gold Medalists at IMO Geometry](https://arxiv.org/abs/2404.06405)


---

[Graph Reinforcement Learning for Combinatorial Optimization: A Survey and Unifying Perspective](https://arxiv.org/abs/2404.06492)



---

[Text-Based Reasoning About Vector Graphics](https://arxiv.org/abs/2404.06479)



---

[pfl-research: simulation framework for accelerating research in Private Federated Learning](https://arxiv.org/abs/2404.06430)


---

[MuPT: A Generative Symbolic Music Pretrained Transformer](https://arxiv.org/abs/2404.06393)


---

[VISION2UI: A Real-World Dataset with Layout for Code Generation from UI Designs](https://arxiv.org/abs/2404.06369)


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
[]()
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
[]()


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

[https://arxiv.org/abs/2403.11835](Agent3D-Zero: An Agent for Zero-shot 3D Understanding)

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

[https://arxiv.org/abs/2401.08500](Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering)

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


#### 5th of December 2022

[https://arxiv.org/abs/2312.03052](https://arxiv.org/abs/2312.03052)

- Answers visual questions by creating programs, that can review the image such as count number of specific types of objects and use tools.
- Answer is provided with CoT reasoning based on filtered program from many programs executed.


---

[Beyond Isolation: Multi-Agent Synergy for Improving Knowledge Graph Constructio](https://arxiv.org/abs/2312.03022)

- Uses three LLM agents for entity, event and relation extraction to build knowledge graph.


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

[AutoMix: Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963)

- AutoMix: Use a smaller LLM to generate initial response and uses Meta-Verifier to check the trustworthy in rough scale. If the answer is trustworthy then use the small LLM answer, otherwise consult a larger LLM.
- Uses Incremental Benefit Per Unit Cost (IBC) metric to asses effectiveness of this approach.


---


#### 14th of November 2023

[DeepThought: An Architecture for Autonomous Self-motivated Systems](https://arxiv.org/abs/2311.08547)

- DeepThought: An architecture for cognitive language agents posing agency, self-motivation, and partly meta-cognition.
- Includes supervisor module, Deep Reinforcement Learning module, Attention Schema (long-term memory), Language/Auditory/Vision modules and Embedding store.


----


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

[The next grand challenge for AI](https://www.ted.com/talks/jim_fan_the_next_grand_challenge_for_ai/transcript)

- Foundational Agent: Agents, which scale in all three axis of: skills, embodiment and realities.  If chatgpt was scaled with data, foundational agents are scaled with realities.


---


#### 16th of October 2023

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

[SmartPlay : A Benchmark for LLMs as Intelligent Agents](https://arxiv.org/abs/2310.01557)

- SmartPlay: a benchmark to test LLM-based agents from 9 perspectives.
- Tests: Reasoning with object dependencies, planning ahead, spatial reasoning, learning from history, and understanding randomness. 


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


#### 29th of Setember 2023

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

[The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864.pdf)

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

[https://arxiv.org/abs/2308.10848](https://arxiv.org/abs/2308.10848)

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


#### 16th of July 2023

[Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)

- ChatDev: Define task and automatically generate SW designing, coding, testing, and documentation using "Chat Chains", where LLM-based chats include different roles for each sub-task: CEO, programmer, CTO etc.
- Includes role-assignment, memory and self-reflection.  

---

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

[Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334)

- Gorilla is a retrieve-aware finetuned LLaMA-7B model for API calls using self-instruct to generate Instruction-API pairs. 


---

#### 17th May 2023

<div id="tot"></div>

[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) 

- Tree of Thoughts (ToT)-technique makes decisions using multiple different reasoning paths, self-evaluating choices to decide next action with ability to look back/forward for global decisions.


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

#### 11th April 2023

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
- The concept is best understood here in the blog by : [https://selfrefine.info/](https://selfrefine.info/) with GIFs and code examples.
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

<div id="astarssearch">  </div>

#### 24th of November 1967


[A formal Basis for the Heuristic Determination of Minimum Cost Paths](https://ai.stanford.edu/%7Enilsson/OnlinePubs-Nils/General%20Essays/roboticsandai.pdf)

- A* search algorithm.
- Defines the A* search algorithm for the first time, widely used in RL as planning algorithm.




<div id="what">  

</div>



----






## What are Autonomous Agents?




#### Autonomous Agent

Autonomous Agents was [defined](#autonomousagentdefinition)  by Franklin & Graesser in 1996 as: "a system situated within and **a part of an environment** that **senses** that environment and **acts** on it, over **time**, in pursuit of its own **agenda** and so as to effect what it senses in the future." 


Good:
- Technological approach agnostic.
- Non-controversial definition: leaves aside Consiousness & definition of AGI.


Negative:
- Lacks aspects about generalization: tasks/objectives/embodiments.
- Vague about human communication and cognitition.


---


####  Generalist Agent 

[Generalist Agent was defined by Reed et al. in 2022](#generalistagent): "**Generalist Agents**, that can adapt to new embodiments and **learn new tasks with few data**." through "...**a multi-modal, multi-task, multi-embodiment** generalist policy."

Positive:
- Generalization of tasks/embodiments.
- Generalization to novel situations
- Multi-modality, especially language/perception/embodiment
- Aspect of Multi-modality (Perception / Language / Embodiment)
- Data efficiency

Negative aspects:
- Lack of other key observations by Franklin & Graesser.
- Vague about cognitive skills: reasoning and planning.



---

#### Reinforcement Learning Agents



[Reinfoceement Learning Agent](http://www.incompleteideas.net/papers/barto-sutton-97.pdf) was defined by Sutton & and Barto (1997): 

"**The reinforcement-learning agent** and its **environment** interact over a sequence of discrete time steps. The specification of their interface defines a particular problme: The actiosn are the choices made by the agent; the situations provide tha agent's basis for making the choices; and **the rewards** are the basis for evaluating these chocices. Everything inside **the agent** is completely known and controllable by the agent; everything outside is incompletely controllable but may or may not be completely known. **A policy** is a stochastic rule by which the agent selects **actions** as a function of situations. Roughly, the agent's objective is to learn a policy that maximizes the amount of reward it receives over the log run"


<p align="center">

  
  <img width="335" alt="image" src="https://github.com/tmgthb/Autonomous-Agents/assets/46755670/6711e82c-c8ea-4be4-8701-1014e0389f00">

  
</p>




Positive:
- Standard definition of the Reinforcement Learning (RL) system. Very similar with An Autonomous Agent-definition by Franklin & Graesser (1996).
- RL systems are provenly versatile and used for: Optimization, Learns from experience, **Generalization**, Delayed Consequences and Exploration [Stanford cs234 lecture slide 19](https://web.stanford.edu/class/cs234/slides/lecture1.pdf).
- Most recent LLM-models use RL during fine-tuning phase


  
Negative:
- RL approaches around language/communication require still more investigation.


---


#### Language Agents / LLM Agents

[Kenton et al. (2021)](#languageagentdefinition) define the concept of Language Agent: " machine learning systems whose actions are restricted to give natural language text-output only, rather than controlling physical actuators which directly influence the world." 

Positive:
- First paper definining LLM-based Agents
- Language-based agents are exceptionally good way of controlling agents towards human perception, plans and objectives.

Negative:
- Text-only
- The definition does not consider RL Agent / Autonomous Agent-aspects, such as environment, embodiment etc.
- LLM-agent poor describes the currently wide variety of components: memory/VLM/reasoning-modules etc. 


---


#### Embodied Agents

Embodied agent-term was used by Brook (1991) in the ["The Role of Learning in Autonomous Robots"(1991)](https://people.csail.mit.edu/brooks/papers/colt.pdf) and Brooks (1991) defined Embodiment in the AI within the  ["Intelligence without reason"](https://people.csail.mit.edu/brooks/papers/AIM-1293.pdf) and in the book: ["New approaches to Intelligence"](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9e1ef9e0a9de1d1c5e36d1a4c735da2fa313c563):

"Embodiment: The **robots have bodies** and experience the world directly--their actions are part of a dynamic with the world, and the  **actions have immediate feedback on the robots' own sensations. **". 

Brooks revits prior literature of Embodiment in the [Building Brains for Bodies](https://dspace.mit.edu/bitstream/handle/1721.1/5948/AIM-1439.pdf?sequence=2&isAllowed=y). Steel and Brooks (1995) define concept of Embodied AI and Embodied Agents within Autonomous agents in the book: ["The Artificial Life Route to Artificial Intelligence Building Embodied, Situated Agent"](https://www.routledge.com/The-Artificial-Life-Route-to-Artificial-Intelligence-Building-Embodied/Steels-Brooks/p/book/9781138545854). 


Positive:
- Embodiment validates capacity to manage real world.
- Physical grounding provides meaning to (symbolic) information processed.

Negative:
- Unclarity regads agents in virtual embodiment in virtual reality.
- The definition does not consider Cognition/Language aspects.


---



#### AI-Agents (Agentic AI)


[Shavit et al. (2023)](#agentaidefinition) define AI Agent: "we will generally conceptualize **agentic AI systems** as operating in **pursuit of goals defined by humans** and in **environments determined by humans** (and often in **cooperation with human** “teammates”), rather than fully-autonomous systems that set their own goals."

Positive:
- Highlights concrete aspects of "agentiness": goal complexity, environment complexity, adaptability and independent execution.
- Includes cooperation with human-in-the-loop
- Identifies there is no binary-distinction between LLM (GPT-4) and Agentic AI system.

Negative:
- Definition itself is porrly framed to reflect the paper's "Agentiness"-aspects such as ability to generalize across variety of tasks.
- Definition does not highlight any human congitive capabilities like search planning, perception etc.
- The level of independence and automatization are controversial from user experience perspective.

Alternative definition uses:


- [Agent AI](#agentbasedai) term is defined: "...as a class of interactive systems that can perceive visual stimuli, language inputs, and other environmentally grounded data, and can produce meaningful embodied actions."


---

####  Autonomous General Agent 


I define next a new term **Autonomous Generalist Agents" (AGA):



**Autonomous Generalist Agent (AGA) perceives, reasons, plans communicates and interacts over time as part of an environments to complete novel objectives.** 


Positive:
- Perceive multimodal information 
- Reason
- Plan own agenda
- Communicate with language
- Interact bi-directionally with the environment
- Internal clock
- Generalize novel tasks
  
Negative:
- Does not include consciousness.



[Back to top](#topofthepage)


<div id="why"> </div>


---



## Why Autonomous Agents work? 



### About predicting next sequence

- LLMs are trained to predict the next word/token. We empirically know this results [Multi-task learning](#multitask). Single training objective results a [Massively Multi-task learning](#extreme). 
- Next sequence prediction is [generic](#extreme) learning process: any "<input, output>"-sequence relationship learning is a "next-word prediction task".
- Next sequence prediction algorithm is generic algorithm.

    - Information is typically sequential: language is sequence of words, DNA is sequence of nucleotides, computer programs are sequences of instructions.
    - Media: Videos are sequence of images, Music is sequence of notes, image is sequence of pixels and speech is sequence of phonemes.
    - Actions: Dance is sequence of movements, day is sequence of events, time is sequence of time steps.
    - Concepts about the world: Causality is sequential (cause-effect). Time is sequential(before-after). Life is sequential(parent-child).


Overall, the next sequence prediction is one of the most generic single learning objectives in a system, which attempts to learn a model about itself or about the world.


### Demystifying Emerging Abilities

- [Emerming Abilities](#emerging) refers to ability present in a larger LLM and not in a smaller one. There are +137 known Emerging abilities(increasing).
- Emerging abilities include Emerging Prompting Strategies such as: [CoT](#cot), which was not present in GPT-2 and emerged in GPT-3 model.

Overall, emerging abilities have increased so far contiuously as compute is scaled up and more data introduced. 



---

### About the context window

- Context word [derives](https://www.etymonline.com/word/context) from latin "contextus" (a joining together). To be precise, the word contextere" (to interweave): "com" (together) and "texere" (to weave).
- The word is not sum of words "con" (with) and "text". For example, saying "another one, please" can be said without specifying explicitly in the preceding text the concept of the "another one". For example the context differs, if we are listening a song vs. in a restaurant. The context does not need to be explicitly written.
- LLM context window size has gradually increased from the 2k context window (GPT-3), to 4k (GPT-3.5), 8k / 32k (GPT-4), 128k (GPT-4.5) for [OpenAI models](https://platform.openai.com/docs/models/), 2M (Claude 3) and 1M (Gemini Pro 1.5) with near [perfect accuracy](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) and the existing empirical research has pushed the textual context window limit above 2M tokens: [LongRoPE](https://arxiv.org/abs/2402.13753) and [MemWalker-interactive agent](https://arxiv.org/abs/2310.05029).  The textual context window in LLMs is already beyond human-level capacity text. 
- Terry Winograd wrote in 2001 a paper called  ["Architectures for Contex"](https://hci.stanford.edu/winograd/papers/context/context.pdf), where he reviews context in language, human-computer dialogue, context vs. setting and virtual and physical context. Winograd argues, that communication is based on common ground between speaker/hearer during the interpretation, which guided not only by physical environment, but as well non-physical shared context, such a common goal. As the context window in LLMs increase, the focus will inevitable turn towards managing context window perceived from other modalities such as vision, sound, robots,  etc.
- Lenat (1998) authored ["The Dimensions of Context Space"](https://web.media.mit.edu/~lieber/Teaching/Common-Sense-Course/Dimensions-Context-Space.pdf) offers "a must-read" analysisw on the various dimensions and aspects of the context. For exampple Lenat proposes to think context being a region in some n-dimensional space.
- Context is a region in n-dimensional embedding space. Text is only one of the dimensions.

Overall, context is n-dimensional space, including text-dimension already in LLMs above human-level, yet lacking in other dimensions at the moment, such as vision, sounds and embodiment. 


---


### Self-Recursive LLMs
- LLMs can Self-Improve its own reasoning outputs using techniques such as [CoT](#cot), [Self-Consistency](#selfconsistency) and [In-Context Learning](#multitask) during Inference.
- LLMs can Self-Improve its model weights with: [STaR](#star), where the LLM itself is fine-tuned using correct CoT reasoning.
- [V-STaR](#vstar) improves the STaR-method by making it data efficient: by learning not only from correct, but as well incorrect solutions generated.
- LMs [Recursively Self-Improving (RSI)](#stop) code with [STOP]#stop). Adam Kalai explains insights from this technique in this [lecture about STOP](#stopvideo).
- [LLM Self-Improves its LLM](#restreact) by finetuning with its own synthetic data without human evaluation to imrove mathematical reasoning.
- LLM fine-tuning may be based on [Self-Play](#spin), where the LLM is fine-tuned based on it playing against itself from previous iteration.

---

### Search planning
- Tree-structures enable searching large reasoning trees for a solution to a complex problem
- [Tree-Of-Thought](#tot) and (ToT or [Graph-of-Thought](#got) are extensions of the CoT-technique with function call. [ToolChain*](#toolchain) is first known an efficient tree search-based planning algorithm for LLMs. ToolChain* offers significantly lower running time compare to MCTS/ToT-DFS/ToT-BFS and significantly better success rate up to 30 steps forward. In fact, it improves significantly reasoning capabilities of LLMs, offering SOTA reasoning with GSM8K.
- Advanced reasoning chains are often open-ended problems between question and answer, in a massive reasoning tree. The ability to search large trees effectively, makes often possible to use algorithms such as A*, MCTS etc to search this space to come up a short, smart path between the problem to solution by using advanced prompting techniques.


---

### Synthetic data enables Small Student Models to outperform their Teachers
- The trend of LLMs using [TinyStories](#tinystories) or [Textbook-like datasets with Exercises](#textbookvideo) is known to significantly improve performance of the LLMs. [TinyGSM](#tinygsm) achieved 81.5% accuracy in GSM8K, outperforming significantly larger LLMs. Synthetic data offers in these examples possibility to distill smaller, yet high performing Student LLMs from the Teacher LLM with similar performance level. Secondly, LLMs can be used to generate diverse, yet cheaply available synthetic data to improve reasoning capabilities.
- Autonomous Agents help generate long-range planning and action data withing real-world, which is motivated by enabling finetuning VLMs or LLMs with this data.


---

### Agent-based AI
- [Interactive Agent Foundational Model](#interactiveagent) uses action tokens to enhance grounding with cross-reality data.


---

### Agents are Resources 
- As per defined by Minsky in 2005, human mind can be seen as a [Resource-cloud](#resourcecloud).
- LLM agents prompting enables resource-rich behaviour from LLMs.

---

### World Models


[Internal model of the world](https://web.media.mit.edu/~minsky/papers/steps.html)

- Minsky defined in 24th of Octoboer 1960 in Steps Towards Artificial Intellicence under chapter: "Models of Oneself":

 "If a creature can answer a question about a hypothetical experiment, without actually performing that experiment, then the answer must have been obtained from some submachine inside the creature. The output of that submachine (representing a correct answer) as well as the input (representing the question) must be coded descriptions of the corresponding external events or event classes. Seen through this pair of encoding and decoding channels, the internal submachine acts like the environment, and so it has the character of a "model." The inductive inference problem may then be regarded as the problem of constructing such a model. 
 
To the extent that the creature's actions affect the environment, :**this internal model of the world will need to include some representation of the creature itself:**. If one asks the creature "why did you decide to do such and such" (or if it asks this of itself), any answer must come from the internal model."

- Minsky writes as well about world models 1968 in the "Matter, Mind and Models", which I recommend to read as a whole, but I add two sentences:

"We use the term "model" in the following sense: To an observer B, an object A* is a model of an object A to the extent that B can use A* to answer questions that interest him about A."

"A man's **model of the world** has a distinctly bipartite structure: One part is concerned with matters of mechanical, geometrical, physical character, while the other is associated with things like goals, meanings, social matters, and the like. This division of W* carries through the representations of many things in W*, especially to M itself."


----


### About Intelligence

#### Artificial General Intelligence (AGI):

- Sparks of AGI in GPT-4: [Artificial General Intelligence](https://arxiv.org/abs/2303.12712) and [Levels of AGI](https://arxiv.org/abs/2311.02462)
- GPT-4 performs [high compared to human-level performance on multiple benchmarks despite incomplete AGI](#sparks), not only on few.
- LLMs can overcome [incremental tasks and Discontinuous tasks](#sparks) by using memory already widely integrated by developers or by using LLM agents-methodologies.

  
---

#### Artificial Super Intelligence (ASI):

ASI concept seems vague, because current AI systems are not generally more capable across all tasks. 

- [AlphaZero](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf) demonstrated superhuman performance in multiple game domains by self-play without domain related human-assistance by using MCTS search algorithm.  


---

#### Consciousness

I add in this section definitions, experiments and thoughts of researchers on the controversial subject of consciousness.

[Consciousness: Here, There but Not Everywhere](https://arxiv.org/abs/1405.7089)

- Integrated Information Theory: Theoretical framework understanding consciousness with mathematical model for a systems consciousness, reviews the subjective-experience, makes testable predictions through experiments and not only limited to human brain-like consciousness.

[Perspetives on Consciousness by Chalmers](https://consc.net/papers/c-and-c.html)

- Useful perspectives on this controversial topic.

[Ilya Sutskever defined a practicalconsciousness test](#consciousnesstest) for LLMs.

- AI Consciousness test.


---

### Free energy principle

- Friston (2010) claims in the [The free energy principle and cognitive agents](https://www.uab.edu/medicine/cinl/images/KFriston_FreeEnergy_BrainTheory.pdf), that biological systems, like human brains, reduce free energy by acting on the world and optimizing their internal states related to perception and action.
- In essence, LLMs take large body of text by deploying compute, which results local order in form of LLM model with various capabilities, but as side result increases entropy through the applied training compute

  
---

## Related work

- Overviews: [A Survey on Large Language Model based Autonomous Agents](#autonomousagentssurvey), [LLM Powered Autonomous Agents](#lili) and [The Rise and Potential of Large Language Model Based Agents: A Survey](#llmagentsurvey) and [Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives](#humancap) and simply about [LLMs](#llmsurveymikolov)
- Cognitive functions / human mind mental resources (#resourcecloud) (planning/execution/verification/etc)
- Memory(short/long/sensorial/embedding),
- Roles (teacher/student/etc),
- Tools (other models/vector DBs/APIs/etc),
- Reasoning paths (vanilla/CoT/ToT/GoT/etc),
- Environments (code interpreter/browser/api/RL environment/real world),
- Embodiments (LLM call/virtual enviroment/robotics/real world) and
- Autonomity (manual/interactive/fully autonomous).


---

#### About intelligent behaviour

- Yann Lecun (2024) in Lex Fridman [podcast](https://www.youtube.com/watch?v=5t1vTLU7s40) states four characters of intelligence behaviour:

  - Capacity to undertand the physical world,
  - The ability to remember and retrieve things,
  - Persistent memory,
  - The ability to reason and plan.


[Back to top](#topofthepage)

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
