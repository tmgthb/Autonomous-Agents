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

## Research papers: 2023

[2025 (1/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2025 (2/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (3/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025.md), [2024](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)

Chronological order. 





</div>


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


#### 14th of December 2023

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


#### 8th of December 2023

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


#### 29th of November 2023

[Universal Self-Consistency for Large Language Model Generation](https://arxiv.org/abs/2311.17311)

- Universal Self-Consistency (USC): Uses LLMs to select the most consistent answer among multiple candidates working in mathematical reasoning and code generation and unlike the original Self-Consistency, the method works in open-ended questions.
- This can be used as a more capabale component in the [STaR-method](#star), which generalizes with Q&A with open-ended answers, not only precise answers.


---


#### 28th of November 2023

[Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine](https://arxiv.org/abs/2311.16452)

- Medprompt: Generalist LLM using MedPrompt outperforms SOTA specialist model.
- Uses SOTA prompt method: CoT, Choice Shuffle and Self-Consistency prompting
- Introduces Choice Shuffle-technique, which inreases diversity of the reasoning paths.
  

---


#### 27th of November 2023 

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

[Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243)

- TorToise (TorToise an expressive, multi-voice text-to-speech system): introduces text-to-speech synthesis framework utilizing autoregressive transformer and diffusion decoder with conditioning inputs and CLVP re-ranking for improved speech quality.
- This framework comprises autoregressive transformer for speech token prediction, diffusion decoder for converting tokens to MEL spectrograms, and vocoder for waveform generation from spectrograms.
- TorToise incorporates conditioning MEL from reference audio and CLVP discriminator to enhance speech synthesis expressiveness and enable speaker cloning capabilities.


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


