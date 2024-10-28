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
  Autonomous Agents Resources. Updated daily. See as well the [Research papers](https://github.com/tmgthb/Autonomous-Agents/blob/main/Autonomous_Agents)-section. 

</div>

- [Introduction to Autonomous Agents](#introduction)
  - [Definitions](#definitions)
  - [Evaluation frameworks and Benchmarks](#benchmarks)
  - [Related work](#relatedwork)
- [Autonomous Agent Systems](#systems)
  - [Perception](#perception)
  - [Reasoning](#reasoning)
  - [Planning](#planning)
  - [Memory](#memory)
  - [Tools](#tools)
  - [Self-Recursive Learning](#selflearning)
  - [Embodiment](#embodiment)
  - [Role play](#roles)
  - [Emotional Intelligence](#emotions)
- [Why Autonomous agents work?](#why)
  - [Next sequence prediction](#nextsequenceprediction)
  - [Demystifying "Emerging abilities"](#demystifyingemergingabilities)
  - [Free energy principle](#freeenergyprinciple)
  - [Interpretability](#interpretability)
  - [Synthetic data](#syntheticdata)
- [Future work](#futureresearch)
  - [Real World Environments](#realworldenvironments)
  - [Simulation](#simulation)
  - [Consciousness](#consciousness)
  - [Brain research](#brainresearch)




<div id="introduction">  

</div>


---

<div align="center">

## Introduction to Autonomous Agents

There exists already over +1k research arXiv-research [papers](https://arxiv.org/search/?searchtype=all&query=%22Autonomous+Agents%22&abstracts=show&size=50&order=-announced_date_first) and +700 Github [repositories](https://github.com/search?q=%22Autonomous%20Agent%22&type=repositories) with the term: "Autonomous Agent" .

</div>

- [Definitions](#definitions)
- [Related work](#relatedwork)
- [Benchmarks](#benchmarks)

---

<div id="definitions">  

</div>



### Definitions

- [Autonomous Agent](#autonomousagent_definition)
- [Artificial General Intelligence (AGi)](#agi_definition)
- [Generalist Agent](#generalistagent_definition)
- [Reinforcement Learning Agent](#rlagent_definition)
- [LLM Agent](#llmagent_definition)
- [Embodied Agent](#embodiedagent_definition)
- [AI Agent](#aiagent_defintion)
- [Autonomous Agent (my definition)](#aga_definition)

---

<div id="autonomousagent_definition">  

</div>



#### Autonomous Agent


Autonomous Agents was [defined](https://github.com/tmgthb/Autonomous-Agents#autonomousagentdefinition)  by Franklin & Graesser in 1996 as: "a system situated within and **a part of an environment** that **senses** that environment and **acts** on it, over **time**, in pursuit of its own **agenda** and so as to effect what it senses in the future." 


Good:
- Technological approach agnostic.
- Non-controversial definition: leaves aside Consiousness & definition of AGI.


Negative:
- Lacks aspects about generalization: tasks/objectives/embodiments.
- Vague about human communication and cognitition.

There are alternatives for this term, such as [Mae(1993):](https://www.cs.uml.edu/~holly/91.549/readings/maes94modeling.pdf)

"Autonomous Agents are systems that inhabit dynamic, unpredictable environment in which they try to satisfy a set of time-dependent goals or motivations."


---

<div id="agi_definition">  

</div>

####  Artificial General Intelligence (AGI)

Artificial General Intelligence (AGI) was used first time by Avrum [Gubrud in 1997](https://web.archive.org/web/20180126125209/https://foresight.org/Conferences/MNT05/Papers/Gubrud/index.html) and defined "By advanced artificial general intelligence, I mean AI systems that rival or surpass the human brain in complexity and speed, that can acquire, manipulate and reason with general knowledge, and that are usable in essentially any phase of industrial or military operations where a human intelligence would otherwise be needed. Such systems may be modeled on the human brain, but they do not necessarily have to be, and they do not have to be "conscious" or possess any other competence that is not strictly relevant to their application. What matters is that such systems can be used to replace human brains in tasks ranging from organizing and running a mine or a factory to piloting an airplane, analyzing intelligence data or planning a battle."

However, the term Artificial General Intelligence (AGI) is currently known throught the terminology defined by Shane [Shane Legg at 2001](https://www.ted.com/talks/shane_legg_and_chris_anderson_the_transformative_potential_of_agi_and_when_it_might_arrive?subtitle=en&geo=es) to Goertzel, who later we went to publish a collection of articules called "Artificial General Intelligence - [Goertzel & Pennachin (2007)](http://repo.darmajaya.ac.id/5336/2/Springer%20-%20Artificial%20General%20Intelligence%20%28%20PDFDrive%20%29.pdf). This original definition refers:

"Applying these ideas to AI, we come to the conclusion that, to roughly emulate the nature of human general intelligence, an artificial general intelligence system should have:
 - the ability to solve general problems in a non-domain-restricted way, in the same sense that a human can;
 - most probably, the ability to solve problems in particular domains and particular contexts with particular efficiency;
 - the ability to use its more generalized and more specialized intelligence capabilities together, in a unified way;
 - the ability to learn from its environment, other intelligent systems, and teachers;
 - the ability to become better at solving novel types of problems as it gains
 experience with them."

[Shane Legg](https://www.ted.com/talks/shane_legg_and_chris_anderson_the_transformative_potential_of_agi_and_when_it_might_arrive?subtitle=en&geo=es) clarified his original definition (see TED talk: 4 min 15 sec) was just systems able to play Go-game, AGI systems were able to do "...many, many other things.", while his current definition is "AGI is a system that can do all cognitive tasks, that people can do, possibly more, but at least the cognitive task, that people can typically do."

AGI is referred in addition with various types of definitions. Perhaps the best paper to check is by [Morris et al (2023)](https://arxiv.org/abs/2311.02462), which not only reviews the different groups (Turing test, Strong AI / AI with consciousness, analogy to human brain, human level cognitive tasks, ability to learn tasks, economically valuable work/OpenAI, flexible and general, capable to earn money and generally performing) of AGI definers, but as well operationalises these groupings into different levels of AGI and defines 6 principles for AGI.


---



<div id="generalistagent_definition">  

</div>


####  Generalist Agent 


[Generalist Agent was defined by Reed et al. in 2022](https://github.com/tmgthb/Autonomous-Agents#generalistagent): "**Generalist Agents**, that can adapt to new embodiments and **learn new tasks with few data**." through "...**a multi-modal, multi-task, multi-embodiment** generalist policy."

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

<div id="rlagent_definition">  

</div>




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



<div id="llmagent_definition">  

</div>




#### LLM Agents / Language Agents


[Kenton et al. (2021)](#languageagentdefinition) define the concept of Language Agent: " machine learning systems whose actions are restricted to give natural language text-output only, rather than controlling physical actuators which directly influence the world." 

Positive:
- First paper definining LLM-based Agents
- Language-based agents are exceptionally good way of controlling agents towards human perception, plans and objectives.

Negative:
- Text-only
- The definition does not consider RL Agent / Autonomous Agent-aspects, such as environment, embodiment etc.
- LLM-agent poor describes the currently wide variety of components: memory/VLM/reasoning-modules etc. 


---


<div id="embodiedagent_definition">  

</div>



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


<div id="aiagent_defintion">  

</div>


#### AI-Agents (Agentic AI)


[Shavit et al. (2023)](https://github.com/tmgthb/Autonomous-Agents#agentaidefinition) define AI Agent: "we will generally conceptualize **agentic AI systems** as operating in **pursuit of goals defined by humans** and in **environments determined by humans** (and often in **cooperation with human** “teammates”), rather than fully-autonomous systems that set their own goals."

Positive:
- Highlights concrete aspects of "agentiness": goal complexity, environment complexity, adaptability and independent execution.
- Includes cooperation with human-in-the-loop
- Identifies there is no binary-distinction between LLM (GPT-4) and Agentic AI system.

Negative:
- Definition itself is porrly framed to reflect the paper's "Agentiness"-aspects such as ability to generalize across variety of tasks.
- Definition does not highlight any human congitive capabilities like search planning, perception etc.
- The level of independence and automatization are controversial from user experience perspective.

Alternative definition uses:


- [Agent AI](https://github.com/tmgthb/Autonomous-Agents#agentbasedai) term is defined: "...as a class of interactive systems that can perceive visual stimuli, language inputs, and other environmentally grounded data, and can produce meaningful embodied actions."


---



<div id="aga_definition">  

</div>




####  Autonomous Agent (my definition) 

TAll the above definitions include gaps, which I have noted along them. 

Therefore, I found it necessary to add my own definition, which I call simply: **Autonomous Agent" (AA):

**Autonomous Agent (AA) perceives, reasons, plans and interacts using language, memories, emotions and tools as part of an environments made of infinite actors, actions, modalities and events to complete novel objectives over time.** 


Positive:
- Perceive multimodal information 
- Reason
- Plan own agenda
- Communicate with language
- Emotional aware
- Includes memory
- Uses tools
- Interact bi-directionally with the environment
- Internal clock
- Generalize novel tasks
  
Negative:
- Do agent find useful human-like consciousness? How it would work?


---


<div id="relatedwork">  

</div>


---

<div align="center">  

### Related work
Includes list of literature reviews by other authors for quick reference.
</div>



- [A Survey on Large Language Model based Autonomous Agents](https://github.com/tmgthb/Autonomous-Agents#autonomousagentssurvey),
- [LLM Powered Autonomous Agents](https://github.com/tmgthb/Autonomous-Agents#lili),
- [The Rise and Potential of Large Language Model Based Agents: A Survey](https://github.com/tmgthb/Autonomous-Agents#llmagentsurvey),
- [Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives](https://github.com/tmgthb/Autonomous-Agents#humancap),
- [LLMs](https://github.com/tmgthb/Autonomous-Agents#llmsurveymikolov),
- [Unleashing the Power of Graph Learning through LLM-based Autonomous Agents](https://arxiv.org/abs/2309.04565)
- [Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives](https://arxiv.org/abs/2312.11970)
- [Agent AI: Surveying the Horizons of Multimodal Interaction](https://arxiv.org/abs/2401.03568)
- [Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security](https://arxiv.org/abs/2401.05459)
- [Understanding the planning of LLM agents: A survey](https://arxiv.org/abs/2402.02716)
- [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)
- [Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods](https://arxiv.org/abs/2404.00282)
- [LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models](https://arxiv.org/abs/2404.01230)
- [A Survey on Large Language Model-Based Game Agents](https://arxiv.org/abs/2404.02039)
- [Comprehensible Artificial Intelligence on Knowledge Graphs: A survey](https://arxiv.org/abs/2404.03499)
- [Graph Reinforcement Learning for Combinatorial Optimization: A Survey and Unifying Perspective](https://arxiv.org/abs/2404.06492)
- [System for systematic literature review using multiple AI agents: Concept and an empirical evaluation](https://arxiv.org/abs/2403.08399)
- [Exploring Autonomous Agents through the Lens of Large Language Models: A Review](https://arxiv.org/abs/2404.04442)
- [System for systematic literature review using multiple AI agents: Concept and an empirical evaluation](https://arxiv.org/abs/2403.08399)
- [Real-World Robot Applications of Foundation Models: A Review](https://arxiv.org/abs/2402.05741)
- [Can Large Language Model Agents Simulate Human Trust Behaviors?](https://arxiv.org/abs/2402.04559)
- [Can Generative Agents Predict Emotion?](https://arxiv.org/abs/2402.04232)
- [Large Multimodal Agents: A Survey](https://arxiv.org/abs/2402.15116)
- [Intelligent agents: theory and practice](https://www.cs.ox.ac.uk/people/michael.wooldridge/pubs/ker95.pdf)

---


<div id="benchmarks">  

</div>


<div align="center">

### Evaluation frameworks and Benchmarks
The benchmarks section includes few well known generic evaluation frameworks around AI research on Intelligence and set of Autonomous Agent component-level benchmarks. 
</div>

- [Intelligent behaviour](#intelligentbehaviour)
- [Artificial General Intelligence](#agi)
- [Artificial Super Intelligence](#asi)
  

---

<div id="intelligentbehaviour"> </div>

#### Intelligent behaviour


- Yann Lecun (2024) in Lex Fridman [podcast](https://www.youtube.com/watch?v=5t1vTLU7s40) states four characters of intelligence behaviour:

  - Capacity to undertand the physical world,
  - The ability to remember and retrieve things,
  - Persistent memory,
  - The ability to reason and plan.
 
---

<div id="agi"> </div>

#### Artificial General Intelligence (AGI):


- Sparks of AGI in GPT-4: [Artificial General Intelligence](https://arxiv.org/abs/2303.12712) and [Levels of AGI](https://arxiv.org/abs/2311.02462)
- GPT-4 performs [high compared to human-level performance on multiple benchmarks despite incomplete AGI](#sparks), not only on few.
- LLMs can overcome [incremental tasks and Discontinuous tasks](#sparks) by using memory already widely integrated by developers or by using LLM agents-methodologies.

  
---

<div id="asi"> </div>

#### Artificial Super Intelligence (ASI):


ASI concept seems vague, because current AI systems are not generally more capable across all tasks. 

- [AlphaZero](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf) demonstrated superhuman performance in multiple game domains by self-play without domain related human-assistance by using MCTS search algorithm.  




---
---

<div id="systems">  </div>


<div align="center">

## Autonomous Agent Systems
</div>

- [Perception](#perception)
- [Reasoning](#reasoning)
- [Planning](#planning)
- [Memory & Context window](#memory)
- [Tools](#tools)
- [Self-Recursive Learning](#selflearning)
- [Embodiment](#embodiment)
- [Role play](#roles)
- [Emotional Intelligence](#emotions)

 

---

<div id="perception"></div>

### Perception

F. Rosenblatt was an early investigator of Perception through the (Perceptron)[https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf]-paper from 1958.

Modern AI systems refer perception through VLMs.


---

<div id="reasoning"></div>

### Reasoning

Reasoning is (defined)[https://dictionary.cambridge.org/dictionary/english/reasoning] by Cambridge dictionary: "the process of thinking about something in order to make a decision".

An autonomous agent is characterized by its ability to make decisions autonomously in order to pursue its goals. Therefore, the reasoning is a fundamental characteristics of the autonomous agent. 

Humans reason in multiple ways. For example mathematical reasoning cannot be only solved using only perception/memory/planning. 

[Peng et al. 2024](https://github.com/tmgthb/Autonomous-Agents#reasoning_study) categorize reasoning into:
- Logical reasoning ((Gemini Ultra)[https://arxiv.org/abs/2312.11805] achieves 80.8% in ChartQA)
  - Inductive
  - Deductive
  - Abductive
- Mathematical reasoning ((Claude 3 Opus)[https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf]: 95% in GSM8K, 60.1% in MATH)
- Commonsense reasoning ((Gemini Ultra)[https://arxiv.org/abs/2312.11805]/(GPT-4)[https://rowanzellers.com/hellaswag/]/(Claude 3 Opus)[https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf]: 95.3%/95.4% in HellaSwag)
- Multi-hop reasoning ((Claude 3)[https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf] 96.4% in ARC)
- Structured-data reasoning (See research such as (Chain-of-Table by Wang et al 2024)[https://arxiv.org/abs/2401.04398v2])

The overall reasoning capability is currently roughly 86.6% (MMLU-benchmark) with (Claude 3 Opus)[https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf] published in March 2024. 


Full human-level reasoning requires more progress/better reliability/better datasets/better benchmarks in multiple dimensions of reasoning such as spatial, tempoeral, emotional, meta-cognition and probabilistic. 

---



---

<div id="planning">  

</div>

### Planning

Planning is defined by (Cambridge)[https://dictionary.cambridge.org/dictionary/english/planning] dictionary as: "the act of deciding how to do something".

According to (Peng et al. (2024))[https://arxiv.org/abs/2406.00936], planning enables agents to autonomously identify and execute actions towards goals.

"Let's think step by step"
- Technique better known as Chain-of-Thought (CoT).
- 
- Tree-structures enable searching large reasoning trees for a solution to a complex problem
- [Tree-Of-Thought](https://github.com/tmgthb/Autonomous-Agents#tot) and (ToT or [Graph-of-Thought](https://github.com/tmgthb/Autonomous-Agents#got) are extensions of the CoT-technique with function call.
- [ToolChain*](#toolchain) is first known an efficient tree search-based planning algorithm for LLMs. ToolChain* offers significantly lower running time compare to MCTS/ToT-DFS/ToT-BFS and significantly better success rate up to 30 steps forward. In fact, it improves significantly reasoning capabilities of LLMs, offering SOTA reasoning with GSM8K.
- Advanced reasoning chains are often open-ended problems between question and answer, in a massive reasoning tree. The ability to search large trees effectively, makes often possible to use algorithms such as A*, MCTS etc to search this space to come up a short, smart path between the problem to solution by using advanced prompting techniques.


Why planning works so well in so many domains?

According to [Noam Brown (2024)](https://www.youtube.com/watch?v=eaAonE58sLU), scaling up "test-time compute" for search planning has been key ingredient in the past AI-breakthroughts (Chess, Go, Poker & No-Press Diplomacy). Cicero-model employed test-time compute in its planning module by predicting actions of all players/predicting what other plays would think Cicero would take/deciding output action and intent for the dialogue model to generate communication back to other players. This additional planning compute made the model especially effective in No-Press Diplomacy game. 

- Considering the AI-models play these games above human-level, I see it logical to apply these methods in LLMs.
- Brown says, that it is easier for humans to verify ("Let's verify step by step") correctness of reasoning chain in specific domains (math/programming/puzzles, while not true in image recognition/information retrieval), than generating the reasoning solution, which means LLMs are better verifiers than generators of the correct reasoning chains.
- Brown calls this as the "Generator-Verifier-gap".
- Brown argues, that if in a given domain, there is a generator-verifier-gap, and we have a good verifier, then it is possible to scale up compute of solution generation and then verify.
- Brown continues, that the "Let's verify step by step"-paper introduces process reward mdel, which instead of conditioning the verifier by the final state, it conditions with every correct step in the process towards the final goal.
- Brown notes, that large companies will prefer scaling up "training/development costs", while maintaining low "inference costs".

Zhang et al. (2024) show, that [GenRM-CoT](https://arxiv.org/abs/2408.15240) outperforms discriminatory verifiers, scaling in inference-time compute, model capacity and dataset size.

Valmeekam et al. (2022) introduced [PlanBench](https://arxiv.org/abs/2206.10498) reviews LLMs planning capabilities using classic planning domains.
- Gives few-shot examples for the planner to learn the planner for the given domain together with instructions about the planning environment. This forces the LLM to think through the planning examples (to evaluate actual reasoning capacity), rather than rather than capability to pattern match training data.
- 8 challenges: basic plan, complex plan for unexpected changes etc. 
- GPT-4 struggled with complex planning scenarios managing only 34% of planning scenarios. LLMs memorize specific words, which change leads to lack of planning capability. LLMs recognize well similar planning scenarios to understand user intent. LLMs struggle to adapt sudden changes in the dynamic environment. Research is required to improve explainability of the LLM planning.

Valmeekam et al. (2023) reviewed further planning capabilities, which [critizes](https://arxiv.org/abs/2305.15771) further LLMs planning capability to:
- model-based reasoning,
- rules & constraints,
- grounding to reality and
- to get feedback.

Suggests to improve planning capabilities using: neurosymbolic AI, train AI models specific for planning (reward working plans). Two groups using LLMs for planning resulted no statistically significant difference between the grou using LLM and group not using it.

Potential use cases for LLMs in planning:
- LLMs can process vast amount of information in complex problem into different planning styles.
- Adapting with dynamic planning by adjusting quickly for example navigation route in case of an accident.
- LLMs can sub-divide tasks into smaller pieces.
- Personal productivity to schedule and prioritize tasks.

Valmeekam et al. (2024) finds o1 is slightly better in travel planning, but difficult to plan in advance is more costly, than traditional LLMs. Suggests using LRM-Modulo approach, which uses external verifier to offer better guarantee. How to ensure control and transparency of the LM model based planning systems? 

Valmeekam et al. (2024) reviewed [self-critiquing its plans](https://arxiv.org/abs/2310.08118) by testing LLM vs. LLM with verifier specifically in planning:
- LLM with verifier was only slightly better, because the errors made by the verifier.
- LLM had fundamental problems to evaluate the models responses.
- One possibility is that LLMs are still poor in understanding cause-and-effect and lack of collobrative reasoning.

Kunde et al. 2024 (review)[https://arxiv.org/abs/2311.00226] finds, that transformers robustly adapt to new tasks through few-shot in-context learning without explicit model optimization. 

Kambhampati et al. (2024) [finds](https://arxiv.org/abs/2402.01817) only 12% of plans are operatable and struggle in self-verification.
- Investigates classical planning problems such as travel planning tasks.
- Suggests "LLM-Modulo"-framework: bringing team-work into LLM planning using collaborative approach with team of experts.
- Offers large boost in LLM planning capabilities.
- Includes problem specification, prompt generator, plan backboard, reformatter, formal critics, commonsense constraint critic, hard constraint critic and meta controller.

Steechly et al. (2024) finds, that LLM is unable to learn the correct algorithm from the demonstrations rather than its ability to execute that algorithm. 
- Suggests that few-shot examples are not guaranteed to improve the generic procedural reasoning of the LLMs in novel instances.
- Suggests, that CoT prompts are likely to only work consistently in sufficiently narrow problem class.
- Suggests, that more important to evaluate the chain-of-thought-process, rather than the final result.



---

<div id="memory">  

</div>

### Memory & Context window

Memory (refers)[https://dictionary.cambridge.org/dictionary/english/memory] to abilitty to remember according to Cambridge dictionary.

Autonomous agents require memory for multiple reasons: to retrieve information, to learn from past, tracking progress, to make decisions, to use context and to communicate.   

According to (Zhang et al. (2024))[https://arxiv.org/abs/2404.13501v1], the Memory in LLM-based agents can be divided into:
- Inside-trial information
- Cross-trial information
- External Knowledge.

(Zhang et al. (2024))[https://arxiv.org/abs/2404.13501v1] find three types of memory operations: memory reading, memory writing and memory management. 

(Li et al. (2024))[https://arxiv.org/abs/2211.05110] divide memory research into three parts:
- World knowledge in LLMs
- Knowledge update in LLMs
- Contextual and Parametric knowledge


- Context word [derives](https://www.etymonline.com/word/context) from latin "contextus" (a joining together). To be precise, the word contextere" (to interweave): "com" (together) and "texere" (to weave).
- The word is not sum of words "con" (with) and "text". For example, saying "another one, please" can be said without specifying explicitly in the preceding text the concept of the "another one". For example the context differs, if we are listening a song vs. in a restaurant. The context does not need to be explicitly written.
- LLM context window size has gradually increased from the 2k context window (GPT-3), to 4k (GPT-3.5), 8k / 32k (GPT-4), 128k (GPT-4.5) for [OpenAI models](https://platform.openai.com/docs/models/), 2M (Claude 3) and 1M (Gemini Pro 1.5) with near [perfect accuracy](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) and the existing empirical research has pushed the textual context window limit above 2M tokens: [LongRoPE](https://arxiv.org/abs/2402.13753) and [MemWalker-interactive agent](https://arxiv.org/abs/2310.05029).  The textual context window in LLMs is already beyond human-level capacity text. 
- Terry Winograd wrote in 2001 a paper called  ["Architectures for Contex"](https://hci.stanford.edu/winograd/papers/context/context.pdf), where he reviews context in language, human-computer dialogue, context vs. setting and virtual and physical context. Winograd argues, that communication is based on common ground between speaker/hearer during the interpretation, which guided not only by physical environment, but as well non-physical shared context, such a common goal. As the context window in LLMs increase, the focus will inevitable turn towards managing context window perceived from other modalities such as vision, sound, robots,  etc.
- Lenat (1998) authored ["The Dimensions of Context Space"](https://web.media.mit.edu/~lieber/Teaching/Common-Sense-Course/Dimensions-Context-Space.pdf) offers "a must-read" analysisw on the various dimensions and aspects of the context. For exampple Lenat proposes to think context being a region in some n-dimensional space.
- Context is a region in n-dimensional embedding space. Text is only one of the dimensions.

Overall, context is n-dimensional space, including text-dimension already in LLMs above human-level, yet lacking in other dimensions at the moment, such as vision, sounds and embodiment. 

Latest research suggest attention can be extended to infite context window in LLMs.


---


<div id="embodiment">  

</div>

### Embodiment
Real-world physical interaction requires Autonomous Agents capable of making emebodied actions and decisions. 

- [Interactive Agent Foundational Model](https://github.com/tmgthb/Autonomous-Agents#interactiveagent) uses action tokens to enhance grounding with cross-reality data.

---

<div id="tools">  

</div>

### Tool use


---



---

<div id="emotions">  

</div>

### Emotional Intelligence



---

<div id="selflearning">  

</div>

### Self-Recursive Improvement


- ["LMs can Self-Improve"](https://arxiv.org/abs/2403.19154) its own reasoning outputs using techniques such as [CoT](https://github.com/tmgthb/Autonomous-Agents#cot), [Self-Consistency](https://github.com/tmgthb/Autonomous-Agents#selfconsistency) and [In-Context Learning](https://github.com/tmgthb/Autonomous-Agents#multitask) during Inference.
- LLMs can Self-Improve its model weights with: [STaR](https://github.com/tmgthb/Autonomous-Agents#star), where the LLM itself is fine-tuned using correct CoT reasoning.
- [V-STaR](https://github.com/tmgthb/Autonomous-Agents#vstar) improves the STaR-method by making it data efficient: by learning not only from correct, but as well incorrect solutions generated.
- LMs [Recursively Self-Improving (RSI)](https://github.com/tmgthb/Autonomous-Agents#stop) code with [STOP]#stop). Adam Kalai explains insights from this technique in this [lecture about STOP](https://github.com/tmgthb/Autonomous-Agents#stopvideo).
- [LLM Self-Improves its LLM](https://github.com/tmgthb/Autonomous-Agents#restreact) by finetuning with its own synthetic data without human evaluation to imrove mathematical reasoning.
- LLM fine-tuning may be based on [Self-Play](https://github.com/tmgthb/Autonomous-Agents#spin), where the LLM is fine-tuned based on it playing against itself from previous iteration.




---

<div id="why">  

</div>




<div align="center">

## Why Autonomous Agents work? 

</div>

- [Next sequence prediction](#nextsequenceprediction)
- [Demystifying "Emerging abilities"](#demystifyingemergingabilities)
- [Free energy principle](#freeenergyprinciple)
- [Interpretability](#interpretability)
- [Synthetic data](#syntheticdata)


<div id="nextsequenceprediction">  

</div>

---


### Next sequence prediction

LLMs are trained to predict the next word/token, which leads to: [Multi-task learning](https://github.com/tmgthb/Autonomous-Agents#multitask):
- Backed up by empirical evidence.

The single training objective: "predict next token" results a [Massively Multi-task learning](https://github.com/tmgthb/Autonomous-Agents#extreme).
- "Massively Multi-task learning" results massive amount of new skills learned from a single objective. 

Next sequence prediction algorithm is generic algorithm.
- Next sequence prediction is [generic learning process](https://github.com/tmgthb/Autonomous-Agents#extreme).
- Any "<input, output>"-sequence relationship, can be learned as "next-sequence prediction task".

Information is typically sequential: 
- language is sequence of words,
- DNA is sequence of nucleotides,
- computer programs are sequences of instructions.
- Media: Videos are sequence of images, Music is sequence of notes, image is sequence of pixels and speech is sequence of phonemes.
- Actions: Dance is sequence of movements, day is sequence of events, time is sequence of time steps.
- Concepts about the world: Causality is sequential (cause-effect). Time is sequential(before-after). Life is sequential(parent-child).

Cross-modality Transformers:
- The universal nature of next-sequence prediction is empirically visible in different Transformer models: ViT for Images, Whisper for Audio, SORA for video.

Next sequence prediction is perhaps the most generic single learning objective known to produce intelligent behaviour. 

    I call this surprising, yet unexpected phenomenon as the "Paradox of Lexical Labyrinth".

    Paradox of Lexical Labyrinth:

    The paradoxical phenomenon whereby seemingly simple mechnanism of a next sequence prediction, such as predicting the next word in a     language, gives rise to advanced cognitive skills like profound reasoning capabilities. The labyrinth refers to the vast & complex      landscape of language, characterized by its infinite potential for compressing meaning, expressions and intelligence.

    Teemu Maatta, 07.06.2024

---



<div id="demystifyingemergingabilities">  

</div>


---


### Demystifying Emerging Abilities

[Emerming Abilities](https://github.com/tmgthb/Autonomous-Agents#emerging) refers to ability present in a larger LLM and not in a smaller one.
- The initial definition refers to situation, where emerging abilities have increased so far contiuously as compute is scaled up and more data introduced.

There are +137 known Emerging abilities(increasing).
- Emerging abilities include Emerging Prompting Strategies such as: [CoT](https://github.com/tmgthb/Autonomous-Agents#cot), which was not present in GPT-2 and emerged in GPT-3 model.

Research has [proven](https://arxiv.org/abs/2403.15796) the existing of Emergent abilities from perspective of pre-training loss, even with continuous metrics.

Overall, Emergent abilities are proven to on language models from the perspective of pre-training loss, instead of model/data size.

Emergent abilities suggest that AI models self-organize internal structures to perform tasks to reduce pre-training loss, even without being explicitly programmed for those specific capabilities.


---


<div id="freeenergyprinciple">  

</div>


### Free energy principle

Friston (2010) claims in the [The free energy principle and cognitive agents](https://www.uab.edu/medicine/cinl/images/KFriston_FreeEnergy_BrainTheory.pdf), that biological systems, like human brains, reduce free energy by acting on the world and optimizing their internal states related to perception and action.
- The basic idea is, that biological agents minimize free energy.

Just like human brain mnimizes free energy, the LLMs minimize the prediction error:
- If we give a LLM the training objective of minimizing loss for "next-sequence prediction" and lot of energy/compute and data, then it will self-organize its weights into optimal local order.
- This compression enables LLMs to learn emerging skills beyond merely memorizing the training data.


---

<div id="interpretability">  </div>

### Interpretability

The ability [to extract and directly interpret LLM features](https://arxiv.org/abs/2406.04093) helps to build Autonomous agents, which understand and interact effectively with human language. 

We know as well, that [CLIP-model neurons](https://distill.pub/2021/multimodal-neurons/) can be matched with biological human brain neurons.

Overall, we are now able to both match human and AI model neurons, but as well interpret LLM model features. 


---

<div id="syntheticdata">  </div>

### Synthetic data

Synthetic data is not useful to train even larger AI models, but more importantly to use efficiently scarce-domain data.

The trend of LLMs using [TinyStories](https://github.com/tmgthb/Autonomous-Agents#tinystories) or [Textbook-like datasets with Exercises](https://github.com/tmgthb/Autonomous-Agents#textbookvideo) is known to significantly improve performance of the LLMs. [TinyGSM](https://github.com/tmgthb/Autonomous-Agents#tinygsm) achieved 81.5% accuracy in GSM8K, outperforming significantly larger LLMs. Synthetic data offers in these examples possibility to distill smaller, yet high performing Student LLMs from the Teacher LLM with similar performance level. Secondly, LLMs can be used to generate diverse, yet cheaply available synthetic data to improve reasoning capabilities.
- Autonomous Agents help generate long-range planning and action data withing real-world, which is motivated by enabling finetuning VLMs or LLMs with this data.

---



---


<div id="futureresearch">  </div>


<div align="center">

## Future research
I add into this section areas for future research, which are required for Autonomous Agents mimicking human-like behaviour.
</div>

- [Consciousness](#consciousness)
- [Real World Environments](#realworldenvironments)

---

<div id="consciousness"> </div>

### Consciousness

There is no single generally agreed definition of Consciousness and I will not try to define it here. 

[Integrated Information Theory (IIT)](https://arxiv.org/abs/1405.7089) and its latest [version 4.0](https://arxiv.org/abs/2212.14787) are one of the key theories existing. This theory includes "Phi", which measures amount of integrated information to quantify level of consciousness of the system. The IIT includes 5 key characteristics:
- Intrinsic
- Composition
- Information
- Integration
- Exclusion

The IIT allows making predictions, which can be tested through experiments and it is not limited to human brain-like consciousness.

[Ilya Sutskever defined, perhaps the first, test-scenario to test, if AI models has consciousness:](https://github.com/tmgthb/Autonomous-Agents#consciousnesstest) for LLMs.

Literature reviews on consciousness:
- [Mathematical Approaches in the Scientific Study of Consciousness](https://jkleiner.de/uploads/preprints/Mathematical%20Approaches%20in%20the%20Scientific%20Study%20of%20Consciousness%20(Preprint,%20Johannes%20Kleiner).pdf)
- [Survey of Consciousness Theory from Computational Perspective](https://arxiv.org/abs/2309.10063)
- [Consciousness in Artificial Intelligence: Insights from the Science of Consciousness](https://arxiv.org/abs/2308.08708)


---


### Brain research


[Movie reconstruction from mouse visual cortex activity](https://www.biorxiv.org/content/10.1101/2024.06.19.599691v1)

- Reconstructs ground-truth video using images from mouse brain.


---

[Brain representation in conscious and unconscious vision](https://www.biorxiv.org/content/10.1101/2024.05.27.596053v1)

- Discovers fronto-parietal cortex is involved in representing unconscious content.


---

- [Real World Environments](#realworldenvironments)


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

