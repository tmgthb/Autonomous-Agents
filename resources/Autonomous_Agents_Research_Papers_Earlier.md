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

## Research papers: 2022 and earlier

[2025 (1/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2025 (2/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (3/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025.md), [2024](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)

Chronological order. 





</div>


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

#### 12th October 2022

[Interactive Language: Talking to Robots in Real Time](https://arxiv.org/abs/2210.06407)

- Interactive Language: introduces a framework for real-time language-instructable robots, with Teleoperated Data Collection, Hindsight Language Relabeling, Language Conditioned Behavioral Cloning (LCBC), Robot Policy, Real-time Language Guidance, ResNet CNN, CLIP Text Encoder, Vision-Language Transformer, Temporal Transformer, and Policy MLP.
- Interactive Language framework uses behavioral cloning on large language-annotated dataset for training real-time language-guided robot policy.
- This framework facilitates interactive robot control for complex manipulation tasks and demonstrates high success rate on diverse language commands.


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

[A Learning Algorithm that Mimics Human Learning](https://www.santafe.edu/research/results/working-papers/a-learning-algorithm-that-mimics-human-learning)

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


