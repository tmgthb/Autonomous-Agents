<div align="center"> 
  
[![GitHub Repo stars](https://img.shields.io/github/stars/tmgthb/Autonomous-Agents?style=social)](https://github.com/tmgthb/Autonomous-Agents/stargazers) 
[![Twitter Follow](https://img.shields.io/twitter/follow/Teemumtt3?style=social)](https://twitter.com/Teemumtt3)

</div>  


# Autonomous-Agents

Autonomous Agents (LLMs). Updated daily
- [Research papers.](#papers)
- [What is an Autonomous Agent?](#what)
- [Why Autonomous Agents work?](#why)


---


<div id="papers"> </div>  


#### 16th of January 2024


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

[Application of LLM Agents in Recruitment: A Novel Framework for Resume Screening](https://arxiv.org/abs/2401.08315)

- Automated resume screening, where segments from CV are classified into information types, personal information is removed. T
- The HR grading LLM agent rates these resumes and another HR decision making agent picks preferred application with eplanation, which is then available for the HR professional.

---

[A Study on Training and Developing Large Language Models for Behavior Tree Generation](https://arxiv.org/abs/2401.08089)

- LLMs used to generate Behavioural Trees (BT) generation for agents/robots.

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

#### 9th of January 2024

[Agent Alignment in Evolving Social Norms](https://arxiv.org/abs/2401.04620)

- EvolutionaryAgent: Evaluates LLM agents based on fitness to social norms using observer LLM within EvolvingSociety-environment.
- LLM agents producing highest social norm ratings, self-envolve and reproduce into new generation LLM agents. Agents either convert into obsolate or survived.
- Agents events are recorded within short term memory with a threshold, which defines when long term and higher-level memories are distilled.
- Defines initial stage of the EnvolvingSociety and the desired direction only.


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

#### 18th of Deceomber 2023

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


#### 14th od December 2023

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

#### 7th if October 2023

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

[Large Language Models as Analogical Reasoners](https://arxiv.org/abs/2310.01714)

- LLM self-generates examples/knowledge related to the task.


---

[Conceptual Framework for Autonomous Cognitive Entities](https://arxiv.org/abs/2310.06775)

- Conceptual framework for Autonomous entities.


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

#### 23th of September 2023
[Natural Language based Context Modeling and Reasoning with LLMs: A Tutorial](https://arxiv.org/abs/2309.15074)

- LLM-driven Context-aware Computing (LCaC) approach.


---

#### 20th of September 2023

[You only look at the screens: Multimodal Chain-of-Action Agents](https://arxiv.org/abs/2309.11436)

- Multimodal Chain-of-Actions Agents (Auto-UI) interacts directly with the UI
- Chain-ofAction technique using series of action histories and future action plans.


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

#### 28th of August

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

[WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct](https://arxiv.org/abs/2308.09583)

- Improves math reasoning with Reinforcement Learning from Evol-Instruct Feedback (RLEIF): Upward and Downward evolution improve instructions by making questions easier or harder based on their difficulty level.


---

#### 17th of August 2023

<div id="rest"></div>

[Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998)

- Introduces Reinforced Self-Training (ReST).
- Grow step generates data from LLM, Improve step uses this filtered data to fine-tune the LLM. Repeat. 


---

####

[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)

- AutoGen: An open source framework, where LLM agents converse with other LLM agents either one or many, chat with humans and use tools.
- LLM agents are able to create new chats with other LLM agents.

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

#### 26th of May 2023

[Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Large Language Models](https://arxiv.org/abs/2305.16582)

- Graph-of-Thought (GoT) reasoning: To model human thought process as graph instead of chain to improve LLM reasoning capability.


---

[Impossible Distillation: from Low-Quality Model to High-Quality Dataset & Model for Summarization and Paraphrasing](https://arxiv.org/abs/2305.16635)

- Uses low-quality LM to generate High-quality dataset (more diverse and more effective for generalization in unseen domains) to train a high quality model: 770 million parameter model outperforms GPT-3 in multiple tasks evaluated by humans.

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

#### 28th of January 2022

<div id="cot"></div>

[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

- Defines Chain-of-Thought (CoT).
- CoT is one Emerging Ability not present in smaller models, but present in larger models.
- CoT can be seen as Self-Recursive Learning, where the LLM improves its own output by having LLM use intermediate steps to solve complex task.
- The approach effectively demonstrates the LLMs capability to perform Self-Recursive Learning, altough its not integrated back as training data of the model.


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
  
----

## What are Autonomous Agents?


<div id="what">  </div>

## Definition

Autonomous Agents were [defined 1996 by Franklin & Graesser](#autonomousagentdefinition) as ..."a system situated within and a part of an environment that senses that environment and acts on it, over time, in pursuit of its own agenda and so as to effect what it senses in the future."

In 2023, several research studies have mapped the concepts of Autonomous Agents:

- [A Survey on Large Language Model based Autonomous Agents](#autonomousagentssurvey), [LLM Powered Autonomous Agents](#lili) and [The Rise and Potential of Large Language Model Based Agents: A Survey](#llmagentsurvey) and [Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives](#humancap)

These chains may use different types of:

- Cognitive functions / human mind mental resources (#resourcecloud) (planning/execution/verification/etc)
- Memory(short/long/sensorial/embedding),
- Roles (teacher/student/etc),
- Tools (other models/vector DBs/APIs/etc),
- Reasoning paths (vanilla/CoT/ToT/GoT/etc),
- Environments (code interpreter/browser/api/RL environment/real world),
- Embodiments (LLM call/virtual enviroment/robotics/real world) and
- Autonomity (manual/interactive/fully autonomous).


Consciousness research:
- There is a conceptiually defined [consciousness test](#consciousnesstest).



----


## Why Autonomous Agents work? 

<div id="why"> </div>

### About predicting next-word:
- LLMs are trained to predict the next-word. The single training objective is known to result [Multi-task learning](#multitask). However, it is better understood as a [Massively Multi-task learning](#extreme).
- Next-word prediction is very [generic](#extreme) learning process, where any "<input, output>"-sequence relationship learning is a "next-word prediction task".
- LLMs can overcome [incremental tasks and Discontinuous tasks](#sparks) by using memory already widely integrated by developers or by using LLM agents-methodologies.  

### Demystifying Emerging Abilities
- [Emerming Abilities](#emerging) refers to ability present in a larger LLM and not in a smaller one. There are +137 (and increasing known Emerging Abilities.
- Emerging abilities include Emerging Prompting Strategies such as: [CoT](#cot), which was not present in GPT-2 and emerged in GPT-3 model.
- GPT-4 performs [high compared to human-level performance on multiple benchrmarks despite incomplete AGI](#sparks), not only on few.
- AGI definitions are not commonly agreed. GPT-4 lacks for example "agency" and "intrinsic motivation", which LLM agents are mitigate with RAGs, memories and tools.

### Self-Recursive LLMs
- LLMs can Self-Improve its own reasoning outputs using techniques such as [CoT](#cot), [Self-Consistency](#selfconsistency) and [In-Context Learning](#multitask) during Inference.
- LLMs can Self-Improve its model weights with: [STaR](#star), where the LLM itself is fine-tuned using correct CoT reasoning. 
- LMs [Recursively Self-Improving (RSI)](#stop) code with [STOP]#stop). Adam Kalai explains insights from this technique in this [lecture about STOP](#stopvideo).
- [LLM Self-Improves its LLM](#restreact) by finetuning with its own synthetic data without human evaluation to imrove mathematical reasoning.

### Tree-structures enable searching large reasoning trees for a solution to a complex problem
- [Tree-Of-Thought](#tot) and (ToT or [Graph-of-Thought](#got) are extensions of the CoT-technique with function call. [ToolChain*](#toolchain) is first known an efficient tree search-based planning algorithm for LLMs. ToolChain* offers significantly lower running time compare to MCTS/ToT-DFS/ToT-BFS and significantly better success rate up to 30 steps forward. In fact, it improves significantly reasoning capabilities of LLMs, offering SOTA reasoning with GSM8K.
- Advanced reasoning chains are often open-ended problems between question and answer, in a massive reasoning tree. The ability to search large trees effectively, makes often possible to use algorithms such as A*, MCTS etc to search this space to come up a short, smart path between the problem to solution by using advanced prompting techniques.

### Synthetic data enables Small Student Models to outperform their Teachers
- The trend of LLMs using [TinyStories](#tinystories) or [Textbook-like datasets with Exercises](#textbookvideo) is known to significantly improve performance of the LLMs. [TinyGSM](#tinygsm) achieved 81.5% accuracy in GSM8K, outperforming significantly larger LLMs. Synthetic data offers in these examples possibility to distill smaller, yet high performing Student LLMs from the Teacher LLM with similar performance level. Secondly, LLMs can be used to generate diverse, yet cheaply available synthetic data to improve reasoning capabilities.
- Autonomous Agents help generate long-range planning and action data withing real-world, which is motivated by enabling finetuning VLMs or LLMs with this data.

### Reinforcement Learning Agents
- RL agents can use LLMs and VLMs as Reward models or to generate reward functions.
- LLMs can generate code to for a robot, car or embodied agent to interact within an environment. 

### LLM agents
- Research papers have assigened succesfully wide range of roles, wide range of tasks, wide range of environments and wide range of cognitive functions to LLMs.
- LLM agents research produces research on potential risks and behaviours to be expected in real world, when highly capable LLMs are released in the future. 

### Autonomous Agents
- The recent improvements of voice-models, has made it possible voice-controlled UIs with simple integratio with VLMS, LLMs and other APIs.
  
### Agents are Resources 
- As per defined by Minsky in 2005, human mind can be seen as a [Resource-cloud](#resourcecloud).
- LLM agents prompting enables resource-rich behaviour from LLMs.

----


## Citation



If you found our work valuable, please cite:


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
