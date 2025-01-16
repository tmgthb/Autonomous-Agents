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
  Autonomous Agents Resources. Updated daily. See as well the [Research papers](https://github.com/tmgthb/Autonomous-Agents)-section. 

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
  - [Embodiment & Robotics](#embodiment)
  - [Role play](#roles)
  - [Emotional Intelligence](#emotions)
  - [Communication protocols](#protocols)
  - [Operating Systems](#os)
  - [Brain interface](#brain)
  - [Consciousness](#consciousness)
- [Why Autonomous agents work?](#why)
  - [Next sequence prediction](#nextsequenceprediction)
  - [Scaling planning](#scaling_planning)
  - [Demystifying "Emerging abilities"](#demystifyingemergingabilities)
  - [Free energy principle](#freeenergyprinciple)
  - [Interpretability](#interpretability)
  - [Synthetic data](#syntheticdata)




<div id="introduction">  

</div>


---

<div align="center">

## Introduction to Autonomous Agents

+1.3k arXiv research [papers](https://arxiv.org/search/?searchtype=all&query=%22Autonomous+Agents%22&abstracts=show&size=50&order=-announced_date_first) and +1k Github [repositories](https://github.com/search?q=%22Autonomous%20Agent%22&type=repositories) exist with term "Autonomous agents".


</div>

- [Definitions](#definitions)
- [Related work](#relatedwork)
- [Benchmarks](#benchmarks)

---

<div id="definitions">  

</div>



### Definitions

- [Agent](#agent_definition)
- [Autonomous Agent](#autonomousagent_definition)
- [Artificial General Intelligence (AGI)](#agi_definition)
- [SuperIntelligence](#superintelligence_definition)
- [Generalist Agent](#generalistagent_definition)
- [Reinforcement Learning Agent](#rlagent_definition)
- [LLM Agent](#llmagent_definition)
- [Embodied Agent](#embodiedagent_definition)
- [AI Agent](#aiagent_defintion)

- [Autonomous Agent (my definition)](#aga_definition)


---




<div id="agent_definition">  
</div>


#### Agent


The term "agent" originates from the Latin verb *agere*, meaning "to drive, lead, or do"<sup>[1](https://en.wiktionary.org/wiki/agent)</sup> . Its present participle, *agens*, provides the root for "agent," signifying "doing" or "acting" <sup>[2](https://www.dictionary.com/browse/agent)</sup>. This etymology emphasizes the capacity to effect change, underpinning the word's varied meanings <sup>[3](https://www.etymonline.com/word/agent),[4](https://www.merriam-webster.com/dictionary/agent)</sup>.

The Latin root *agere* has also produced related terms like "actor." While both share a common ancestor, they have evolved distinct connotations. "Actor" is often associated with performing arts, while "agent" encompasses broader roles, including those with continuous action or agency<sup>[5](https://www.reddit.com/r/etymology/comments/2ysz48/actor_and_agent/)</sup>.

This chapter will explore various agentic roles, building upon the foundational concept of agency as the capacity to act and effect change. 


---

<div id="autonomousagent_definition">  

</div>



#### Autonomous Agent


Autonomous Agents was [defined](https://github.com/tmgthb/Autonomous-Agents#autonomousagentdefinition)  by Franklin & Graesser in 1996 as: "a system situated within and **a part of an environment** that **senses** that environment and **acts** on it, over **time**, in pursuit of its own **agenda** and so as to effect what it senses in the future." 


Good:
- Agnostic regards underlining tech.
- Excludes controversial aspects: consciousness, AGI, "free will" etc. 



Negative:
- No view regards the degree of generalization / adaption / embodiment / self-construction / communication / cognition.


[Mae (1993)](https://www.cs.uml.edu/~holly/91.549/readings/maes94modeling.pdf) wrote even earlier, yet less cited definition:

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

Good:
- Categorization levels, widely used term

Negative
- Vague: lacks clarity
- Lacks agency, self-construction, etc. 



---

<div id="agi_definition">  

</div>

####  SuperInteligence

Nick Bostrom (2014) defined  SuperIntelligence: 

"An intellect that is much smarter than the best human brains in practically every field, including scientific creativity, general wisdom, and social skills."

Good:
- Categorization levels, widely used term

Negative
- Vague: lacks clarity
- Lacks agency, self-construction, etc. 


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

All the above definitions include gaps, which I have noted along them. 

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
- Lacks aspect of self-constructon and self-replication.

19.12.2024

Based on recent thoughts, I decided to update my prior definition to address the prior gaps. 

Autonomous agents (AA) is defined:

**Autonomous Agent (AA) perceives, reasons, plans, and interacts using language, memories, emotions, and tools within environments of infinite actors, actions, modalities, and events to complete novel objectives over time, driven by survival and replication, and capable of self-construction guided by an adaptable core.**

---


<div id="relatedwork">  

</div>


---

<div id="benchmarks">  

</div>


<div align="center">

### Evaluation frameworks and Benchmarks

</div>

Autonomous agents operate in "Real-World Environments (RWEs)"<sup>[1](https://tmmtt.medium.com/real-world-environments-1995aa68805b),[2](https://arxiv.org/pdf/1904.12901)</sup>.

Therefore, to benchmark Autonomous agents, we should evaluate them in RWEs. RWEs are currently hard problems for agents with unique events. Thus, AI researchers typically prefer to benchmark Autonomous agents rather with reproducible benchmarks. 

For example, Anthropic's LLMs appear to be ahead of the other models in tasks like "pixel counting" and "coding". 

An average developer could spend days of development work to compare performance of different LLMs in a GUI-benchmark, which would not generalize beyond the GUIs beyond its dataset. Thus these results could become quickly invalid as the OS/website/app-design changes. 

Rather, developer could just pick a random GUI, test the LLM-agent in it, and quickly iterate prompting-technique, which improves performance across various LLMs and the learning tends to be transferable towards new tasks.

We can alternatively review AI capabilities from high-level:
- Levels of AGI[1](https://arxiv.org/abs/2311.02462), [2](https://arxiv.org/abs/2303.12712)

We must remember, that above human-level intelligence is not a theoretical concept, but current reality:
- in game-agents like [AlphaZero](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf), which demonstrated superhuman performance in multiple game domains by self-play without domain related human-assistance by using MCTS search algorithm. 



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



Planning refers to<sup>[1](https://dictionary.cambridge.org/dictionary/english/planning)</sup>: "the act of deciding how to do something". and 

In the domain of AI, planning is defined: "devising a plan of action to achieve one’s goals"<sup>[1](https://people.engr.tamu.edu/guni/csce421/files/AI_Russell_Norvig.pdf)</sup>.

or

**"Planning is the reasoning side of acting. It is an abstract, explicit deliberation process that chooses and organizes actions by anticipating their expected outcomes. This deliberation aims at achieving as best as possible some prestated objectives. Automated planning is an area of Artificial Intelligence (AI) that studies this
deliberation process computationally."<sup>[1](https://api.pageplace.de/preview/DT0400.9780080490519_A25022382/preview-9780080490519_A25022382.pdf)</sup>**


Minsky defined 1960 "Planning" as one of five "hard problems of heuristic programming", to achieve AI<sup>[1](http://web.media.mit.edu/~minsky/papers/steps.html)</sup>. STRIPS-system (1971)<sup>[1](https://apps.dtic.mil/sti/tr/pdf/ADA637291.pdf),[2](https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/strips.pdf)</sup> is probably the earliest AI system with multiple other early AI systems with planning  developed in the 1970-1990s<sup>[1](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/833/751)</sup>.

Reinforcement Learning has widely used Planning to advance SOTA-level performance: SoRB<sup>[1](https://arxiv.org/abs/1906.05253)</sup>, Plan2Explore<sup>[2](https://arxiv.org/pdf/2005.05960)</sup>, AlphaZero<sup>[3a](https://arxiv.org/pdf/2106.04615),[3b](https://arxiv.org/pdf/2308.09175)</sup>, DeepNash<sup>[4](https://arxiv.org/pdf/2206.15378)</sup>, Cicero<sup>[5a](https://noambrown.github.io/papers/22-Science-Diplomacy-TR.pdf),[5b](https://arxiv.org/pdf/2210.05492)</sup>.

ChatGPT popularized the concept of RLHF<sup>[1](https://arxiv.org/abs/1706.03741), [2](https://arxiv.org/pdf/2203.02155)</sup> and various variations including RLAIF with LLMs<sup>[1](https://arxiv.org/pdf/2212.08073)</sup>, which are offline RL. Offline RL uses static data collected from previous interactions/simulations. Thus, it suffers data distribution shift during deployment.

Thus, LLM-researchers have moved focused to Online RL with LLMs. Online RL adjusts its policy based on immediate feedback from the environment. For example context and user intent may change rapidly. 

Online RL are zero-shot planners<sup>[1](https://arxiv.org/pdf/2201.07207)</sup> and few-shot planners<sup>[2](https://arxiv.org/pdf/2212.04088)</sup> with ability to generate plans<sup>[3](https://arxiv.org/pdf/2209.11302)</sup>, closed-loop feedback, long-horizon plans<sup>[4](https://arxiv.org/abs/2207.05608)</sup>, value-functions<sup>[5](https://arxiv.org/pdf/2111.03189)</sup>,iterative replanning[6](https://arxiv.org/pdf/2307.06135)</sup>, interactive planning, [7](https://arxiv.org/pdf/2302.01560)</sup>, self-refine plans[8](https://arxiv.org/pdf/2305.16653)</sup>, self-verification[8](https://arxiv.org/pdf/2308.00436)</sup>. 

Agents should excel in complex planning and reasoning, predicting outcomes, and executing strategies for long-term goals.<sup>[1](https://arxiv.org/pdf/2312.11970)</sup>

LLM-based planning approaches include:<sup>[1](https://arxiv.org/pdf/2402.02716)</sup>
- Task decomposition (CoT, ReAct)
- Multi-plan selection (ToT, CoT-SC)
- External planner aided (LLM + PDDL)
- Reflection and Refinement (Reflection, Self-Refine, CRITIC)
- Memory-aided planning (REMEMBER)



---

<div id="memory">  

</div>


### Memory & Context

Memory is [defined](https://dictionary.cambridge.org/dictionary/english/memory) as "the ability to remember information, experiences, and people." 

"Minsky (1985) [proposed]((http://web.media.mit.edu/~minsky/papers/AlienIntelligence.html)) that the human ability to categorize experiences into recognizable objects is fundamental to learning. He argued that human memory is organized around discrete objects rather than working like a hologram, as holographic memory would only be useful when encountering exact replicas of past experiences. This object-based categorization enables us to generalize from experiences and accumulate knowledge, even when situations vary."

Memory is vital for humans and AI in order to retrieve relevant ["context](https://dictionary.cambridge.org/dictionary/essential-british-english/context): about "...all the facts/opinions/etc., which relate to a particular thing/event."

Context-term is not formed from words "con" and "text". Context [derives](https://www.etymonline.com/word/context) actually from latin word "contextus", which refers to "joining together": "com" = together and "texere" = to weave. We tend to think the text input as the LLM context. However, LLM-based agents apply context from multiple modalities and not always explicitly written / said aloud. 

Terry Winograd argued in 2001 ["Architectures for Contex"](https://hci.stanford.edu/winograd/papers/context/context.pdf), that communication is based on common ground between speaker/hearer during the interpretation. This is guided not only by physical environment, but as well non-physical shared context, such a common goal.

["The Dimensions of Context Space"](https://web.media.mit.edu/~lieber/Teaching/Common-Sense-Course/Dimensions-Context-Space.pdf) by Lenat (1998)  offers "a must-read" analysis on the various dimensions and aspects of the context. According to Lenat, Context is a region in n-dimensional embedding space, where text is only one of the dimensions.

LLM context input length has rapidly increased from the 2k context of GPT-3 to actual 1M token productio systems. We will likely see in near future production systems with [infinite context](https://arxiv.org/abs/2404.07143), which may additionally use [LLM fine tuning](https://arxiv.org/abs/2402.13753) or [tree-agents](https://arxiv.org/abs/2310.05029). Interestingly, LLMs are already at "Superintelligence"-level in terms of their capacity to support vastly more textual context than any human.

The ability to support larger context windows is quickly making possible usage of new modalities such as vision, sound, actions, etc.

Traditionally, LLMs are considered "stateless", without retention of the context used in the previous request. ["In-Context Learning" (ICL)](https://arxiv.org/abs/2005.14165) LLMs ability "learn" to process and understand the context provided in the input without explicit parameter updates. Agentic systems today use ICL together with external memory such as vector/graph/sql-databases or simply as text/json/xml-files. We often refer these techniques as Retrieval-Augmented-Generation (RAG), which aims to enhance LLM context with up-to-date/personalized/factual/domain-specific-information. Evidence exist, that LLM are able to [track its own internal state-changes.](https://arxiv.org/abs/2407.11421) Never models like Gemini 2.0 are surprisingly good at such calculations, which go way beyond just pattern matching of the training data. The ability of LLMs to track states is promising for reasoning-tasks. Extra-large input-context windows enable in models like Gemini, to process even large memory structures. KV-caching reuses LLM prompts/tokens/internal states to [significantly reduce latency.](https://arxiv.org/abs/2312.05516) However, alternative KV-caching<sup>[1](https://arxiv.org/abs/2403.11805),[2](https://arxiv.org/pdf/2404.13501v1)</sup>  techniques improve directly the memory management of the LLMs. 


Fine tuning methods have been effectively used in improving LLM performance with extra large context windows and memorizing domain specific knowledge. 

Titan-models were recently introduced as models capable to [memorize at test time](https://arxiv.org/abs/2501.00663). 

Memory<sup>[3](https://arxiv.org/abs/2407.01178)</sup>-architecture suggests infinite context is possible with human-like memory architectures, which support memory consolidation, conscious reasoning and sparse memory.

LLM-based agents apply various types of memory approaches:

- Long term memory<sup>[1](https://arxiv.org/abs/2410.15665v1)</sup>
- Episodic memory<sup>[1](https://arxiv.org/abs/2403.11901),[2](https://arxiv.org/abs/2405.14992),[3](https://arxiv.org/abs/2407.04363),[4](https://arxiv.org/abs/2407.09450),[5](https://arxiv.org/abs/2408.07465), [6](https://arxiv.org/abs/2410.08133),[7](https://arxiv.org/abs/2411.06736),[8](https://arxiv.org/abs/2411.12977)</sup>
- Semantic memory<sup>[1](https://arxiv.org/abs/2405.13009),[2](https://arxiv.org/abs/2411.04999)</sup>
- Procedural memory<sup>[1](https://arxiv.org/abs/2409.01344)</sup>
- Graph memory<sup>[1](https://arxiv.org/abs/2408.15903)</sup>
- Working memory<sup>[1](https://arxiv.org/abs/2312.17259),[2](https://arxiv.org/abs/2305.16338),[3](https://arxiv.org/abs/2306.08129),[4](https://arxiv.org/abs/2402.10548)</sup>
- Dynamic memory<sup>[1](https://arxiv.org/abs/2312.08402)</sup>
- Shared memory / Collective memory<sup>[1](https://arxiv.org/abs/2404.09982)</sup>
- Persistent Experience Memory<sup>[1](https://arxiv.org/abs/2306.07929)</sup>
- Explicit memory<sup>[1](https://arxiv.org/abs/2407.01178)</sup>
- Parametric memory<sup>[1](https://arxiv.org/pdf/2404.13501v1)</sup>



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
<div id="brain">  

</div>


### Brain research


[Movie reconstruction from mouse visual cortex activity](https://www.biorxiv.org/content/10.1101/2024.06.19.599691v1)

- Reconstructs ground-truth video using images from mouse brain.


[Brain representation in conscious and unconscious vision](https://www.biorxiv.org/content/10.1101/2024.05.27.596053v1)

- Discovers fronto-parietal cortex is involved in representing unconscious content.


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
