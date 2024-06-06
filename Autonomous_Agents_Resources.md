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
  Autonomous Agents Resources. Updated daily. See as well the [Research papers](https://github.com/tmgthb/Autonomous-Agents/blob/main/Autonomous_Agents.md)-section. 

</div>

- [Introduction to Autonomous Agents](#introduction)
  - [Definition](#definitions)
  - [Benchmarks](#benchmarks)
  - [Related work](#relatedwork)
- [Autonomous Agent Systems](#systems)
  - [Reasoning](#reasoning)
  - [Perception](#perception)
  - [Planning](#planning)
  - [Memory](#memory)
  - [Emotional Intelligence](#emotions)
- [Why Autonomous agents work?](#why)
- [Future work](#emergingfrontiers)




<div id="introduction">  

</div>


---

<div align="center">

## Introduction to Autonomous Agents

There exists already over +1k research arXiv-research [papers](https://arxiv.org/search/?searchtype=all&query=%22Autonomous+Agents%22&abstracts=show&size=50&order=-announced_date_first) and +700 Github [repositories](https://github.com/search?q=%22Autonomous%20Agent%22&type=repositories) with the term: "Autonomous Agent" .

</div>

- [Definition](#definitions)
- [Benchmarks](#benchmarks)
- [Related work](#relatedwork)



<div id="definitions">  

</div>



### Definitions

- [Autonomous Agent](#autonomousagent_definition)
- [Generalist Agent](#generalistagent_definition)
- [Reinforcement Learning Agent](#rlagent_definition)
- [LLM Agent](#llmagent_definition)
- [Embodied Agent](#embodiedagent_definition)
- [AI Agent](#aiagent_defintion)
- [Autonomous Generalist Agent](#aga_definition)


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




####  Autonomous General Agent 


I define next a new term **Autonomous Generalist Agents" (AGA):



**Autonomous Generalist Agent (AGA) perceives, reasons, hierarchically plans and interacts using language, memories, emotions and tools over time as part of an environments to complete novel objectives.** 


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


<div id="literaturereviews">  

</div>


---


### Literature reviews


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



<div id="capabilities">  

</div>


---


### Capabilities


- Cognitive functions / human mind mental resources (#resourcecloud) (planning/execution/verification/etc)
- Memory(short/long/sensorial/embedding),
- Roles (teacher/student/etc),
- Tools (other models/vector DBs/APIs/etc),
- Reasoning paths (vanilla/CoT/ToT/GoT/etc),
- Environments (code interpreter/browser/api/RL environment/real world),
- Embodiments (LLM call/virtual enviroment/robotics/real world) and
- Autonomity (manual/interactive/fully autonomous).


---


<div id="systems">  

</div>


---

## Autonomous Agents Systems

- [Perception](#perception)
- [Planning](#planning)
- [Memory](#memory)
- [Emotional Intelligence](#emotions)
- [Reasoning](#reasoning)


---

<div id="perception">  

</div>

### Perception

F. Rosenblatt was an early investigator of Perception through the (Perceptron)[https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf]-paper from 1958.

Modern AI systems refer perception through VLMs.


---

<div id="planning">  

</div>

### Planning

Planning is defined by (Cambridge)[https://dictionary.cambridge.org/dictionary/english/planning] dictionary as: "the act of deciding how to do something".

According to (Peng et al. (2024))[https://arxiv.org/abs/2406.00936], planning enables agents to autonomously identify and execute actions towards goals.



---

<div id="memory">  

</div>

### Memory

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


---

<div id="emotions">  

</div>

### Emotional Intelligence


---

<div id="reasoning">  

</div>

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

<div id="why">  

</div>




## Why Autonomous Agents work? 


- [About next sequence prediction](#nextsequenceprediction)
- [Demystifying "Emerging abilities"](#demystifyingemergingabilities)
- [World models](#worldmodels)
- [Resource mind-view](#resourcemind)
- [Free energy principle](#freeenergyprinciple)
- [Real World Environments](#realworldenvironments)


<div id="nextsequenceprediction">  

</div>

---


### About predicting next sequence


- LLMs are trained to predict the next word/token. We empirically know this results [Multi-task learning](https://github.com/tmgthb/Autonomous-Agents#multitask). Single training objective results a [Massively Multi-task learning](https://github.com/tmgthb/Autonomous-Agents#extreme). 
- Next sequence prediction is [generic](https://github.com/tmgthb/Autonomous-Agents#extreme) learning process: any "<input, output>"-sequence relationship learning is a "next-word prediction task".
- Next sequence prediction algorithm is generic algorithm.

    - Information is typically sequential: language is sequence of words, DNA is sequence of nucleotides, computer programs are sequences of instructions.
    - Media: Videos are sequence of images, Music is sequence of notes, image is sequence of pixels and speech is sequence of phonemes.
    - Actions: Dance is sequence of movements, day is sequence of events, time is sequence of time steps.
    - Concepts about the world: Causality is sequential (cause-effect). Time is sequential(before-after). Life is sequential(parent-child).


Overall, the next sequence prediction is one of the most generic single learning objectives in a system, which attempts to learn a model about itself or about the world.

We should still not forget the progress made during the last two years in LLMs based on "simplistic" next-sequence prediction objective. Computational reasoning has not only advanced in the last two years, but it is multiple dimensions close to human-level reasoning. 

I like to refer this  surprising phenomenon as the "Paradox of Lexical Labyrinth".

Paradox of Lexical Labyrinth:

The paradoxical phenomenon whereby seemingly simple mechnanism of a next sequence prediction, such as predicting the next word in a language, gives rise to advanced cognitive skills like profound reasoning capabilities. The labyrinth refers to the vast & complex  landscape of language, characterized by its infinite potential for meaning and expression.

---



<div id="demystifyingemergingabilities">  

</div>


---


### Demystifying Emerging Abilities


- [Emerming Abilities](https://github.com/tmgthb/Autonomous-Agents#emerging) refers to ability present in a larger LLM and not in a smaller one. There are +137 known Emerging abilities(increasing).
- Emerging abilities include Emerging Prompting Strategies such as: [CoT](https://github.com/tmgthb/Autonomous-Agents#cot), which was not present in GPT-2 and emerged in GPT-3 model.

Overall, emerging abilities have increased so far contiuously as compute is scaled up and more data introduced. 


<div id="worldmodels">  

</div>


### World Models


[Internal model of the world](https://web.media.mit.edu/~minsky/papers/steps.html)

- Minsky defined in 24th of Octoboer 1960 in Steps Towards Artificial Intellicence under chapter: "Models of Oneself":

 "If a creature can answer a question about a hypothetical experiment, without actually performing that experiment, then the answer must have been obtained from some submachine inside the creature. The output of that submachine (representing a correct answer) as well as the input (representing the question) must be coded descriptions of the corresponding external events or event classes. Seen through this pair of encoding and decoding channels, the internal submachine acts like the environment, and so it has the character of a "model." The inductive inference problem may then be regarded as the problem of constructing such a model. 
 
To the extent that the creature's actions affect the environment, :**this internal model of the world will need to include some representation of the creature itself:**. If one asks the creature "why did you decide to do such and such" (or if it asks this of itself), any answer must come from the internal model."

- Minsky writes as well about world models 1968 in the "Matter, Mind and Models", which I recommend to read as a whole, but I add two sentences:

"We use the term "model" in the following sense: To an observer B, an object A* is a model of an object A to the extent that B can use A* to answer questions that interest him about A."

"A man's **model of the world** has a distinctly bipartite structure: One part is concerned with matters of mechanical, geometrical, physical character, while the other is associated with things like goals, meanings, social matters, and the like. This division of W* carries through the representations of many things in W*, especially to M itself."



---


<div id="resourcemind">  

</div>


### Agents are Resources 


- As per defined by Minsky in 2005, human mind can be seen as a [Resource-cloud](https://github.com/tmgthb/Autonomous-Agents#resourcecloud).
- LLM agents prompting enables resource-rich behaviour from LLMs.


---


<div id="freeenergyprinciple">  

</div>


### Free energy principle


- Friston (2010) claims in the [The free energy principle and cognitive agents](https://www.uab.edu/medicine/cinl/images/KFriston_FreeEnergy_BrainTheory.pdf), that biological systems, like human brains, reduce free energy by acting on the world and optimizing their internal states related to perception and action.
- In essence, LLMs take large body of text by deploying compute, which results local order in form of LLM model with various capabilities, but as side result increases entropy through the applied training compute










---





<div id="emergingfrontiers">  </div>


## Emerging Frontiers 


- [Infite context window](#contextwindow)
- [Self-Learning / Self-Recursive Improvement](#selflearning)
- [Search planning](#searchplanning)
- [Synthetic data](#syntheticdata)
- [Perception](#perception)
- [Physical grounding](#physicalgrounding)
- [Measuring Intelligence, Conciousness and Intelligent Behaviour](#measuringintelligence)


<div id="contextwindow">  </div>


### Infite context window


- Context word [derives](https://www.etymonline.com/word/context) from latin "contextus" (a joining together). To be precise, the word contextere" (to interweave): "com" (together) and "texere" (to weave).
- The word is not sum of words "con" (with) and "text". For example, saying "another one, please" can be said without specifying explicitly in the preceding text the concept of the "another one". For example the context differs, if we are listening a song vs. in a restaurant. The context does not need to be explicitly written.
- LLM context window size has gradually increased from the 2k context window (GPT-3), to 4k (GPT-3.5), 8k / 32k (GPT-4), 128k (GPT-4.5) for [OpenAI models](https://platform.openai.com/docs/models/), 2M (Claude 3) and 1M (Gemini Pro 1.5) with near [perfect accuracy](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) and the existing empirical research has pushed the textual context window limit above 2M tokens: [LongRoPE](https://arxiv.org/abs/2402.13753) and [MemWalker-interactive agent](https://arxiv.org/abs/2310.05029).  The textual context window in LLMs is already beyond human-level capacity text. 
- Terry Winograd wrote in 2001 a paper called  ["Architectures for Contex"](https://hci.stanford.edu/winograd/papers/context/context.pdf), where he reviews context in language, human-computer dialogue, context vs. setting and virtual and physical context. Winograd argues, that communication is based on common ground between speaker/hearer during the interpretation, which guided not only by physical environment, but as well non-physical shared context, such a common goal. As the context window in LLMs increase, the focus will inevitable turn towards managing context window perceived from other modalities such as vision, sound, robots,  etc.
- Lenat (1998) authored ["The Dimensions of Context Space"](https://web.media.mit.edu/~lieber/Teaching/Common-Sense-Course/Dimensions-Context-Space.pdf) offers "a must-read" analysisw on the various dimensions and aspects of the context. For exampple Lenat proposes to think context being a region in some n-dimensional space.
- Context is a region in n-dimensional embedding space. Text is only one of the dimensions.

Overall, context is n-dimensional space, including text-dimension already in LLMs above human-level, yet lacking in other dimensions at the moment, such as vision, sounds and embodiment. 

Latest research suggest attention can be extended to infite context window in LLMs.


---


### Self-Learning / Self-Recursive Improvement


- LLMs can Self-Improve its own reasoning outputs using techniques such as [CoT](https://github.com/tmgthb/Autonomous-Agents#cot), [Self-Consistency](https://github.com/tmgthb/Autonomous-Agents#selfconsistency) and [In-Context Learning](https://github.com/tmgthb/Autonomous-Agents#multitask) during Inference.
- LLMs can Self-Improve its model weights with: [STaR](https://github.com/tmgthb/Autonomous-Agents#star), where the LLM itself is fine-tuned using correct CoT reasoning.
- [V-STaR](https://github.com/tmgthb/Autonomous-Agents#vstar) improves the STaR-method by making it data efficient: by learning not only from correct, but as well incorrect solutions generated.
- LMs [Recursively Self-Improving (RSI)](https://github.com/tmgthb/Autonomous-Agents#stop) code with [STOP]#stop). Adam Kalai explains insights from this technique in this [lecture about STOP](https://github.com/tmgthb/Autonomous-Agents#stopvideo).
- [LLM Self-Improves its LLM](https://github.com/tmgthb/Autonomous-Agents#restreact) by finetuning with its own synthetic data without human evaluation to imrove mathematical reasoning.
- LLM fine-tuning may be based on [Self-Play](https://github.com/tmgthb/Autonomous-Agents#spin), where the LLM is fine-tuned based on it playing against itself from previous iteration.


---


<div id="searchplanning">  

</div>


### Search Planning


- Tree-structures enable searching large reasoning trees for a solution to a complex problem
- [Tree-Of-Thought](https://github.com/tmgthb/Autonomous-Agents#tot) and (ToT or [Graph-of-Thought](https://github.com/tmgthb/Autonomous-Agents#got) are extensions of the CoT-technique with function call. [ToolChain*](#toolchain) is first known an efficient tree search-based planning algorithm for LLMs. ToolChain* offers significantly lower running time compare to MCTS/ToT-DFS/ToT-BFS and significantly better success rate up to 30 steps forward. In fact, it improves significantly reasoning capabilities of LLMs, offering SOTA reasoning with GSM8K.
- Advanced reasoning chains are often open-ended problems between question and answer, in a massive reasoning tree. The ability to search large trees effectively, makes often possible to use algorithms such as A*, MCTS etc to search this space to come up a short, smart path between the problem to solution by using advanced prompting techniques.


---


### Synthetic data


- The trend of LLMs using [TinyStories](https://github.com/tmgthb/Autonomous-Agents#tinystories) or [Textbook-like datasets with Exercises](https://github.com/tmgthb/Autonomous-Agents#textbookvideo) is known to significantly improve performance of the LLMs. [TinyGSM](https://github.com/tmgthb/Autonomous-Agents#tinygsm) achieved 81.5% accuracy in GSM8K, outperforming significantly larger LLMs. Synthetic data offers in these examples possibility to distill smaller, yet high performing Student LLMs from the Teacher LLM with similar performance level. Secondly, LLMs can be used to generate diverse, yet cheaply available synthetic data to improve reasoning capabilities.
- Autonomous Agents help generate long-range planning and action data withing real-world, which is motivated by enabling finetuning VLMs or LLMs with this data.


---


### Physical grounding in real world


- [Interactive Agent Foundational Model](https://github.com/tmgthb/Autonomous-Agents#interactiveagent) uses action tokens to enhance grounding with cross-reality data.


---


### Measuring Intelligence, Conciousness and Intelligent Behaviour


- [Measuring Human Intelligence](https://github.com/tmgthb/Autonomous-Agents#human_intelligence)
- [Measuring Artificial General Intelligence](https://github.com/tmgthb/Autonomous-Agents#agi_intelligence)
- [Measuring Artificial Super Intelligence](https://github.com/tmgthb/Autonomous-Agents#asi_intelligence)
- [Measuring Consciousness](https://github.com/tmgthb/Autonomous-Agents#conciousness)


---


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

In the article The Stream of Consciousness by William James (1892) defines [four characterstics of consciousness](https://webspace.ship.edu/cgboer/jamesselection.html)

- Consciousness is persnal
- Consciousness is in constant change
- Personal consciousness is continuous
- Humans pay pays attention to parts of the consciousness, while excluding others parts

[Consciousness: Here, There but Not Everywhere](https://arxiv.org/abs/1405.7089)

- Integrated Information Theory: Theoretical framework understanding consciousness with mathematical model for a systems consciousness, reviews the subjective-experience, makes testable predictions through experiments and not only limited to human brain-like consciousness.

[Perspetives on Consciousness by Chalmers](https://consc.net/papers/c-and-c.html)

- Useful perspectives on this controversial topic.

[Ilya Sutskever defined a practicalconsciousness test](https://github.com/tmgthb/Autonomous-Agents#consciousnesstest) for LLMs.

- AI Consciousness test.



---


#### Intelligent behaviour


- Yann Lecun (2024) in Lex Fridman [podcast](https://www.youtube.com/watch?v=5t1vTLU7s40) states four characters of intelligence behaviour:

  - Capacity to undertand the physical world,
  - The ability to remember and retrieve things,
  - Persistent memory,
  - The ability to reason and plan.



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




