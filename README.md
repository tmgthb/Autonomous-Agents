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

## Research papers: 2025 (1/3)

[2025 (1/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2025 (2/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (3/3)](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025.md), [2024](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)

Chronological order. 





</div>


#### 4th September 2025

[Psychologically Enhanced AI Agents](https://arxiv.org/abs/2509.04343)

- MiT (MBTI-in-Thoughts): introduces a framework for enhancing LLM agents through psychologically grounded personality conditioning, with LLM Agents (core processing units), Psychological Profiles (personality conditioning via prompts), 16Personalities Test (external validation tool), Majority Voting Protocol (isolated reasoning mechanism), Interactive Communication Protocol (decentralized dialogue mechanism), Interactive Communication With Self-Reflection Protocol (dialogue with private memory), Self-Reflection (private deliberation buffer), Blackboard (shared communication memory), and LLM as Judge (final decision maker).
- The framework primes LLM agents with distinct MBTI personality archetypes via prompt engineering, enabling control over behavior along cognitive and affective axes, and supports structured multi-agent communication protocols.
- MiT demonstrates that personality priming induces consistent behavioral biases, improves cooperation and reasoning quality through self-reflection, and generalizes to other psychological models without fine-tuning.

---

[EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation](http://arxiv.org/abs/2509.04310v1)

- EvoEmo framework: introduces an evolutionary reinforcement learning framework for optimizing dynamic emotional expression in multi-turn negotiations, with Negotiation Setup, Seller LLM Agent, Buyer LLM Agent, Product Description, Seller Prompts, Buyer Prompts, EvoEmo Optimization Module, Emotion States, Emotional Policy (πω), Policy Population, Population Generation, Selection Operator, Crossover Operator, Mutation Operator, Negotiation Simulation Module, Simulated Seller Agents, Simulated Buyer Agents, Mediator Agent, Evaluation Module, Reward Function (R(S)), Optimal Policy (πω*), and Termination Conditions, where it evolves high-reward emotion policies using population-based genetic optimization across diverse negotiation scenarios.
- The framework models emotional state transitions as a Markov Decision Process and iteratively refines policies based on rewards achieved during negotiations, combining evolutionary exploration with reinforcement learning principles.
- EvoEmo consistently outperforms vanilla and fixed-emotion baselines, achieving higher success rates, efficiency, and buyer savings by enabling LLM agents to adaptively express emotions.

---

[Are LLM Agents the New RPA? A Comparative Study with RPA Across Enterprise Workflows](http://arxiv.org/abs/2509.04198v1)

- AACU (Agentic Automation with Computer Use): introduces a comparative study of LLM agents and traditional Robotic Process Automation (RPA) across enterprise workflows, including LLM Agent (intelligent core), Computer Use Capability (interacts digital systems), User Interface (UI) Interaction (mimics human actions), Natural Language Input (task instructions), Software Applications (automation targets), and Execution Environment (agent operational context), to evaluate their performance, reliability, and development effort.
- The study found that while RPA generally outperforms AACU in execution speed and reliability for repetitive, stable tasks, AACU significantly reduces development time and offers greater flexibility for dynamic interfaces.
- Despite current limitations in production readiness, AACU shows promise for rapid prototyping and lightweight automation, suggesting future research into hybrid RPA-AACU architectures and multi-agent orchestration.

---

[MAGneT: Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions](http://arxiv.org/abs/2509.04183v1)

- MAGneT (Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions): introduces a novel multi-agent framework for synthetic psychological counseling session generation that decomposes counselor response generation into coordinated sub-tasks, utilizing a Counselor Agent (generates counselor responses) with a CBT Agent (produces structured treatment plan), a Technique Agent (selects therapeutic techniques), five Specialized Response Agents (Reflection Agent (mirrors client expressions), Questioning Agent (explores client feelings), Solutions Agent (provides actionable solutions), Normalizing Agent (validates client experiences), Psycho-ed Agent (offers therapeutic information)), and a Response Generator (synthesizes final utterance), alongside a Client Agent (simulates client behavior) with an Intake Form (client profile, issues) and Attitudes (client emotional stance).
- This framework employs specialized LLM agents, each grounded in core therapeutic techniques, and coordinates them via a dynamic technique selector and a CBT-based planning agent to generate psychologically grounded and nuanced counseling dialogues.
- The system further simulates realistic client behavior using detailed profiles and attitude modeling, and integrates a unified evaluation framework for comprehensive assessment of generated counseling data quality and diversity.

---

[TAGAL: Tabular Data Generation using Agentic LLM Methods](https://github.com/bronval/TAGAL-Tabular-Data-Generation-with-LLMs/)

- TAGAL (Tabular Data Generation using Agentic LLM Methods): introduces a collection of training-free methods for generating synthetic tabular data using an agentic workflow, which includes a Generation LLM (generates tabular data), Feedback LLM (criticizes data, provides recommendations), Initial Prompt (guides initial data generation), Analysis Prompt (guides feedback LLM evaluation), Feedback Prompt (incorporates feedback for generation), Generated Data (synthetic tabular examples), Feedback (LLM-generated data critique/recommendations), Summary LLM (summarizes conversation history), and Refined Prompt (summarized prompt for generation), to iteratively refine generated data quality through feedback loops.
- The framework comprises three distinct methods—SynthLoop, ReducedLoop, and Prompt-Refine—each employing iterative feedback mechanisms to enhance data quality and utility, with Prompt-Refine specifically using a Summary LLM to create a Refined Prompt for efficient generation.
- It demonstrates performance comparable to state-of-the-art training-required models and often outperforms other training-free approaches, showcasing the potential of agentic LLM workflows for high-quality tabular data generation, even with limited original data.

---

[Towards Stable and Personalised Profiles for Lexical Alignment in Spoken Human-Agent Dialogue](http://arxiv.org/abs/2509.04104v1)

- Lexical Profile Construction and Evaluation: introduces a method for constructing stable, personalised lexical profiles from transcribed spoken data, leveraging POS categories and word-based n-grams, processed by SpaCy's nl_core_news_lg pipeline, and evaluated with recall, coverage, and cosine similarity metrics, to form a basis for lexical alignment in human-agent dialogue.
- The study determined that profiles built from 10 minutes of speech, including 5 items for adjectives and conjunctions and 10 items for adverbs, nouns, pronouns, and verbs each, offered the best balance between performance and data efficiency.
- These stable and representative lexical profiles are crucial for developing inclusive lexical alignment strategies in conversational agents, particularly for users with limited real-time input, such as individuals with dementia, by providing a robust basis for LLM-based response generation.

---

[COT-SPACE: A THEORETICAL FRAMEWORK FOR INTERNAL SLOW-THINKING VIA REINFORCEMENT LEARNING](http://arxiv.org/abs/2509.04027v1)

- CoT-Space (Chain-of-Thought Space): introduces a novel theoretical framework that recasts LLM reasoning from a discrete token-prediction task into an optimization process within a continuous, reasoning-level semantic space, including a policy model, reasoning-level states, a reasoning loss landscape, CoT length, solution minimums, noise scale, generalization error, and empirical loss.
- This framework models the LLM reasoning process as a trajectory towards a solution minimum in a reasoning loss landscape, providing a more intuitive and powerful lens for theoretical analysis of internal slow-thinking via RL.
- The analysis demonstrates that an optimal CoT length exists, balancing underfitting (due to insufficient reasoning depth) and overfitting (due to increased model complexity and noise sensitivity), analogous to an optimal learning rate in classical ML.

---

[Meta-Policy Reflexion: Reusable Reflective Memory and Rule Admissibility for Resource-Efficient LLM Agents](http://arxiv.org/abs/2509.03990v1)

- Meta-Policy Reflexion (MPR): introduces a hybrid framework that consolidates LLM-generated reflections into a structured Meta-Policy Memory (MPM), which guides an LLM Base Policy through soft memory-conditioned decoding and hard admissibility checks (HAC) for training-free self-improvement.
- MPR externalizes reusable corrective knowledge as predicate-like rules, enforcing domain constraints to reduce unsafe actions without modifying LLM parameters, thereby retaining adaptability.
- The framework demonstrates consistent gains in execution accuracy and robustness on the AlfWorld benchmark compared to Reflexion baselines, with HAC further improving stability.

---

[World Model Implanting for Test-time Adaptation of Embodied Agents](http://arxiv.org/abs/2509.03956v1)

- WorMI (World Model Implanting): introduces a framework for embodied agents that combines LLMs' reasoning capabilities with independently learned, domain-specific world models through test-time composition, including a Reasoning Model (LLM), Pre-trained World Models, Prototype-based World Model Retrieval, Object Detection Model, Embedding Model, Prototypes, World-wise Compound Attention, Linear Projection Layer, World-level Cross-attention, Reasoning-level Cross-attention, and an Implant/Remove Mechanism.
- The framework's Prototype-based World Model Retrieval method efficiently selects relevant world models using trajectory-based abstract representation matching, while the World-wise Compound Attention method integrates and aligns knowledge from retrieved models with the reasoning model.
- This dual-stage design enables flexible, test-time fusion of domain-specific knowledge, enhancing adaptability to unseen domains and maintaining cross-domain adaptability for embodied agents.

---

[VoxRole: A Comprehensive Benchmark for Evaluating Speech-Based Role-Playing Agents](http://arxiv.org/abs/2509.03940v1)

- VoxRole: introduces a comprehensive benchmark for evaluating speech-based Role-Playing Conversational Agents (RPCAs), built using a Spoken Dialogue Extraction Pipeline (Extracts movie dialogues) and a Persona Distillation Pipeline (Builds character profiles), and evaluated with a multi-dimensional Evaluation Framework (Assesses model performance).
- The Spoken Dialogue Extraction Pipeline automatically extracts character-rich spoken dialogues from movies by aligning audio with scripts and curating semantically validated segments using components like FFmpeg, Resemble, Whisper-large-v3, Wav2Vec2.0, and MPNet.
- The Persona Distillation Pipeline leverages an LLM and an Acoustic Feature Extraction Module to systematically construct multi-dimensional character profiles, encompassing personality, linguistic style, relationships, and acoustic characteristics, which are then used to generate role-playing prompts for evaluation by an LLM Judge.

---

[MobileRAG: Enhancing Mobile Agent with Retrieval-Augmented Generation](http://arxiv.org/abs/2509.03891v1)

- MobileRAG (Retrieval-Augmented Generation): introduces a mobile agent framework, with InterRAG (external knowledge retrieval), LocalRAG (local app management), and MemRAG (historical operation memory), designed to enhance mobile agents for accurate user query identification and efficient complex mobile task execution.
- The framework addresses limitations of current LLM-based mobile agents, such as over-reliance on LLM comprehension, limited external interaction, and absence of effective memory.
- MobileRAG improves task completion, reduces operational steps, and enhances adaptability by leveraging external knowledge and learning from past successful operations.

---

[FaMA: LLM-Empowered Agentic Assistant for Consumer-to-Consumer Marketplace](http://arxiv.org/abs/2509.03890v1)

- FaMA (Facebook Marketplace Assistant): introduces an LLM-powered agentic assistant for C2C marketplaces, integrating a Llama 4 Maverick LLM as its core reasoning engine, a memory module (Scratchpad, Dialog History, Listings Information), and a suite of specialized Marketplace Tools (Listing Operation, Inventory Search, Messaging) along with RAG and a Knowledge Base.
- This conversational agent simplifies user experience by interpreting natural language commands to automate high-friction workflows for both buyers and sellers, including listing management, bulk messaging, and efficient product discovery.
- FaMA achieves a 98% task success rate and enables up to a 2x speedup in interaction time, providing a lightweight and accessible alternative to traditional app interfaces.

---

[Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning](http://arxiv.org/abs/2509.03817v1)

- MPDF (Meta-Policy Deliberation Framework): introduces a framework for multi-agent LLM collaboration, enabling agents to learn adaptive meta-cognitive policies through a Meta-Cognitive State Space (agent's internal cognitive status), Agent's Observation with Meta-Cognition (local and peer meta-cognitive states), a Policy Network (integrates self-assessment and social context), and a Deliberative Action Space (high-level strategic choices) optimized by SoftRankPO (stable policy optimization algorithm).
- This framework allows agents to dynamically adjust their behavior based on internal confidence and situational context, moving beyond fixed collaboration protocols to dynamic, deliberative strategies.
- SoftRankPO, a key component, stabilizes policy learning by converting raw rewards into rank-based advantages, ensuring robust convergence across diverse reward regimes.

---

[Leveraging LLM-Based Agents for Intelligent Supply Chain Planning](http://arxiv.org/abs/2509.03811v1)

- SCPA (Supply Chain Planning Agent): introduces an LLM-based agent framework for intelligent supply chain planning, featuring Input (user query), Memory (short-term/long-term storage), a Pre-trained LLM (foundation large language model), Task Management (planning orchestration), Task Execution (sub-task processing), and Output (planning report).
- The framework leverages LLM-based agents for intent classification, task orchestration, task execution, and iterative plan correction, enabling autonomous interpretation of natural language queries and dynamic adjustment of plans.
- SCPA demonstrates improved operational efficiency, plan accuracy, and stock availability in real-world e-commerce scenarios by providing evidence-based planning reports and automating complex supply chain decision-making.

---

[SAMVAD: A Multi-Agent System for Simulating Judicial Deliberation Dynamics in India](http://arxiv.org/abs/2509.03793v1)

- SAMVAD (Simulated Agent-based Multi-agent Verdict Adjudication): introduces a Multi-Agent System for simulating judicial deliberation in India, comprising an Orchestrator (manages simulation lifecycle), Judge Agent (generates impartial instructions), Prosecution Counsel Agent (constructs arguments for prosecution), Defense Counsel Agent (constructs arguments for defense), Adjudicator Agents (simulate judicial bench), LLMs (power agents' reasoning), RAG (grounds reasoning in knowledge), Legal Knowledge Base (authoritative Indian legal documents), Vector DB (stores legal document embeddings), Case Files (structured input for simulation), and Final Judgement (consensus-based verdict output).
- The system's core innovation is the deep integration of a domain-specific RAG pipeline, grounding LLM-powered agents in Indian legal texts to generate legally sound, citable instructions and arguments.
- This framework offers a configurable and explainable platform for exploring legal reasoning and group decision-making dynamics within the Indian judicial context, enhancing simulation fidelity and transparency.

---

[Psychologically Enhanced AI Agents](https://github.com/spcl/MBTI-in-Thoughts)

- MBTI-in-Thoughts (MiT): introduces a framework for enhancing LLM agent effectiveness through psychologically grounded personality conditioning, priming agents with distinct MBTI personality archetypes via structured prompts, verifying behavior with the 16Personalities test, and enabling structured multi-agent communication protocols.
- The framework integrates personality traits into LLM agents to improve their performance in both individual and multi-agent settings, demonstrating consistent behavioral biases aligned with assigned personality types across diverse tasks.
- MiT supports various communication protocols, including Majority Voting, Interactive Communication, and Interactive Communication With Self-Reflection, and generalizes to other psychological models like Big Five and HEXACO.

---

[TAGAL: Tabular Data Generation using Agentic LLM Methods](https://github.com/bronval/TAGAL-Tabular-Data-Generation-with-LLMs/)

- TAGAL (Tabular Data Generation using Agentic LLM Methods): introduces an agentic LLM workflow for tabular data generation, including an Initial Prompt (describes task and data), Generation LLM (generates tabular data), Generated Data (synthetic tabular examples), Analysis Prompt (requests feedback), Feedback LLM (provides data criticism), Feedback (recommendations for improvement), Feedback Prompt (incorporates feedback), Conversation History (stores interaction logs), Summary LLM (summarizes conversation history), and Refined Prompt (concise generation prompt).
- This training-free approach leverages LLMs in an iterative feedback loop where a Generation LLM produces data, a Feedback LLM critiques it, and the feedback is used to refine subsequent generation, with Prompt-Refine further optimizing this by using a Summary LLM to condense the conversation history.
- The framework aims to generate high-quality synthetic tabular data comparable to training-based methods, utilizing in-context learning and external knowledge without requiring LLM fine-tuning, thereby reducing computational resources.

---

#### 3rd September 2025

[The Basic B*** Effect: The Use of LLM-based Agents Reduces the Distinctiveness and Diversity of People's Choices.](http://arxiv.org/abs/2509.02910v1)

- LLM-based Agents: introduces a study on how delegating identity-defining choices to LLM-based agents, including Generic AI Agents, Personalized AI Agents with a User Profile Generator and User Data Input, and a Core LLM, impacts interpersonal distinctiveness and intrapersonal diversity of people's choices.
- The research compares choices made by generic and personalized LLM agents against a human baseline, using real-world Facebook page preferences from 1,000 users to measure distinctiveness and diversity.
- Findings indicate that both agent types reduce choice distinctiveness, with personalized agents more strongly compressing intrapersonal diversity, highlighting a trade-off between distinctiveness and diversity in AI-assisted decision-making.

---

[REAL-TIME INSTRUMENT PLANNING AND PERCEPTION FOR NOVEL MEASUREMENTS OF DYNAMIC PHENOMENA](http://arxiv.org/abs/2509.03500v1)

- Dynamic Plume Planning: introduces an automated workflow for real-time instrument planning and perception, synthesizing look-ahead satellite imagery acquisition, onboard data analysis, plume classification, denoising, and autonomous trajectory planning to obtain pinpoint measurements of dynamic phenomena like volcanic plumes.
- The workflow leverages computer vision and machine learning classifiers, including U-Net architectures, for plume segmentation, followed by morphological operations for denoising, and employs various trajectory planning algorithms to guide a Narrow Field of View (NFOV) sensor.
- This onboard system significantly increases the science utility return of high-resolution instruments by dynamically targeting transient events, demonstrating efficient runtimes and generalizability to other remote sensing applications.

---

[Situating AI Agents in their World: Aspective Agentic AI for Dynamic Partially Observable Information Systems](http://arxiv.org/abs/2509.03380v1)

- A2AI (Aspective Agentic AI): introduces a bottom-up framework that situates AI agents in their environment, with all behaviors triggered by changes in their environments, including Environment (central data store), Aspects (specialized environment views), p-agent (generative, perceptive agent) (generates, perceives aspects), a-agent (action agent) (requests environment changes), Agent (operates within aspect), Change Request (action agent's modification proposal), Change List (environment modification summary), Change Summary (perceptive agent's aspect update), and Human (initiates change requests).
- This framework enables selective information disclosure by allowing agents to perceive only limited "aspects" of their environment, preventing information leakage and enhancing security and computational efficiency.
- The reactive, asynchronous, and bottom-up architecture, inspired by situated AI, ensures agents dynamically respond to environmental changes while maintaining strict information isolation.

---

[Autonomous Learning From Success and Failure: Goal-Conditioned Supervised Learning with Negative Feedback](http://arxiv.org/abs/2509.03206v1)

- GCSL-NF (Goal-conditioned Supervised Learning with Negative Feedback): introduces a novel model that integrates contrastive learning principles into the GCSL framework to learn from both success and failure, including a Policy (πθ), Qθ (Q-function), Replay Buffer (R), Relabelled Goal Trajectory Dataset (T+), Original Goal Trajectory Dataset (To), Imitation Learning Loss (L+), Negative Feedback Loss (Lo), Similarity Function (pφ), and Combined Loss.
- This approach addresses limitations of GCSL by utilizing both relabelled successful experience and failures, enabling agents to learn from mistakes and overcome inherent biases.
- The framework employs a learned distance function to assess the quality of achieved states relative to intended goals, promoting exploration and avoiding behavioral stagnation.

---

[Towards Agentic OS: An LLM Agent Framework for Linux Schedulers](http://arxiv.org/abs/2509.01245v2)

- SchedCP (LLM Agent Framework for Linux Schedulers): introduces a decoupled control plane and multi-agent LLM system to autonomously optimize Linux schedulers, featuring a Model Context Protocol server, Workload Analysis Engine, Scheduler Policy Repository, Execution Verifier, and sched-agent's Observation, Planning, Execution, and Learning Agents.
- This framework separates AI's semantic reasoning from the system's execution, enabling LLM agents to safely and efficiently generate and deploy custom eBPF scheduling policies without human intervention.
- The framework achieves significant performance improvements and cost reductions by bridging the semantic gap between application needs and kernel policies through iterative refinement and continuous learning.

---

[Language Models Do Not Follow Occam's Razor: A Benchmark for Inductive and Abductive Reasoning](http://arxiv.org/abs/2509.03345v1)

- INABHYD (Inductive and Abductive Hypothesis Discovery) introduces a programmable and synthetic dataset for evaluating LLMs' inductive and abductive reasoning capabilities, comprising reasoning examples with an incomplete world model, observations, and hypotheses evaluated by an Occam's Razor-based metric.
- The dataset challenges LLMs to generate high-quality hypotheses to explain observations under an incomplete world model, structured as an ontology tree, with difficulty characterized by the tree's height.
- The research reveals that LLMs struggle with complex world models and producing high-quality hypotheses, even with reasoning-enhancing techniques, highlighting limitations in non-deductive reasoning.

---

[EvolveSignal: A Large Language Model Powered Coding Agent for Discovering Traffic Signal Control Algorithms](http://arxiv.org/abs/2509.03335v1)

- EvolveSignal: introduces an LLM-powered coding agent that iteratively refines an Initial Program (starting algorithm) into a Discovered Program (optimized algorithm) by using a Program Database (stores programs/metrics), Prompt Sampler (constructs LLM prompts), LLMs Ensemble (generates code modifications), and Evaluators Pool (simulates/scores programs) to evaluate Child Programs (modified algorithm).
- The framework formulates fixed-time signal control as a program synthesis problem, where LLMs generate Python functions representing algorithms, which are then optimized through simulation-based evaluation and evolutionary search.
- Experiments demonstrate that the discovered algorithms outperform baselines in delay reduction and stop minimization, providing interpretable modifications and practical insights for traffic engineers.

---

[VulnRepairEval: An Exploit-Based Evaluation Framework for Assessing Large Language Model Vulnerability Repair Capabilities](http://arxiv.org/abs/2509.03331v1)

- VulnRepairEval: introduces an exploit-based evaluation framework for assessing LLM vulnerability repair capabilities, featuring Patch Generation, Runtime Injection, Automatic Deployment, Container Execution, and Result Analysis modules, designed for reproducible differential assessment.
- The framework leverages functional Proof-of-Concept (PoC) exploits to verify patch success, requiring the original exploit to fail against the modified code in a containerized environment.
- This work reveals that current LLMs struggle with precise vulnerability localization and syntactically/logically correct patch generation, with advanced prompting and multi-agent approaches yielding minimal improvements.

---

[AGENTRACER: WHO IS INDUCING FAILURE IN THE LLM AGENTIC SYSTEMS?](http://arxiv.org/abs/2509.03312v1)

- AgenTracer: introduces an automated framework for annotating failed multi-agent trajectories and a lightweight failure tracer, AgenTracer-8B, which leverages counterfactual replay, programmatic fault injection, and multi-granular reinforcement learning to efficiently diagnose errors in LLM agentic systems.
- The framework generates a curated dataset, TracerTraj, of over 2,000 high-fidelity failure trajectories, enabling the training of AgenTracer-8B to achieve state-of-the-art performance in agentic system failure attribution.
- AgenTracer-8B provides actionable feedback to off-the-shelf multi-agent systems, leading to performance gains and empowering self-correcting and self-evolving agentic AI.

---

[AIVA: An AI-based Virtual Companion for Emotion-aware Interaction](http://arxiv.org/abs/2509.03212v1)

- AIVA (AI-based Virtual Companion for Emotion-aware Interaction): introduces an AI-based virtual companion that integrates multimodal sentiment perception into LLMs, enabling emotionally aligned and animated Human-Computer Interaction (HCI) through its Multimodal Sentiment Perception Network (MSPN), Vision Transformer (ViT), Textual Encoder (BERT), Cross-Attention (CA) mechanism, Cross-Modal Fusion Transformer, Sentiment Prototypes, Classifier (MLP), Large Language Model (LLM), Emotion-aware Prompt Engineering (EPE), Text-to-Speech (TTS) system, and Animated Avatar module (Live2D).
- The framework's MSPN component processes multimodal inputs (language, facial expressions, voice) to extract sentiment signals, which are then injected into the LLM via EPE to generate contextually appropriate and empathetic language responses.
- AIVA further enhances user experience by providing expressive verbal and visual feedback through its TTS system and an animated avatar module, creating natural, engaging, and emotionally aligned interactions for applications in companion robotics, social care, and mental health.

---

[Loong: Synthesize Long Chain-of-Thoughts at Scale through Verifiers](https://github.com/camel-ai/loong)

- Loong Project: introduces an open-source framework for scalable synthetic data generation and verification, featuring LOONGBENCH (curated seed dataset) and LOONGENV (synthetic data generation environment) with a Generator, Environment, Trainable Agent, and Verifier.
- The framework establishes an agent-environment loop where an LLM-based Generator creates synthetic questions and executable code, which the Environment runs to produce answers, then a Trainable Agent generates Chain-of-Thought solutions, and a Verifier compares these for an RL reward.
- This system aims to overcome the scarcity of high-quality, verifiable datasets in reasoning-intensive domains beyond mathematics and programming, enabling large-scale reinforcement learning with minimal human supervision.

---

[DiaCBT: A Long-Periodic Dialogue Corpus Guided by Cognitive Conceptualization Diagram for CBT-based Psychological Counseling](http://arxiv.org/abs/2509.02999v1)

- DiaCBT (Long-Periodic Dialogue Corpus Guided by Cognitive Conceptualization Diagram for CBT-based Psychological Counseling): introduces a long-periodic dialogue corpus for CBT-based psychological counseling, with Case Annotation, Cognitive Conceptualization, CCD-guided Dialogue Generation, Expert Evaluation, LLMs, CBT Segments, Cognitive Conceptualization Diagrams (CCDs), Human Annotators, Experts, Client Simulator (GPT-4o-mini), Grader Model (LLMrwd), and Therapist Model (fine-tuned LLM therapist), where the paper constructs a multi-session dialogue corpus guided by CCDs to enhance LLMs' ability to emulate CBT psychologists.
- The framework leverages LLMs to generate CCDs for diverse client scenarios and then uses these CCDs, along with annotated CBT strategies, to create realistic, multi-session counseling dialogues.
- DiaCBT also includes a comprehensive evaluation framework, employing a client simulator and a grader model to benchmark the performance of LLM-based therapists against established psychological criteria.

---

[InstaDA: Augmenting Instance Segmentation Data with Dual-Agent System](http://arxiv.org/abs/2509.02973v1)

- InstaDA (Augmenting Instance Segmentation Data with Dual-Agent System): introduces a novel dual-agent system for instance segmentation data augmentation, featuring a T-Agent (generates diverse synthetic data) and an I-Agent (augments data from training images), along with a Prompt Rethink mechanism, BiRefNet, CLIP dual-similarity, Soft-Edge Maps Fusion, ControlNet, Image2Image, SAM-box, and Copy-Paste.
- The T-Agent leverages LLMs and diffusion models with a Prompt Rethink mechanism to iteratively refine prompts and generate diverse images, while the I-Agent enriches data distribution by generating new instances conditioned on existing training images using ControlNet and Image2Image.
- The framework ensures high-quality annotations through BiRefNet and SAM-box for segmentation, and filters generated instances using CLIP dual-similarity and CLIP score before integrating them via Copy-Paste to enhance dataset diversity and distribution.

---

[app.build: A Production Framework for Scaling Agentic Prompt-to-App Generation with Environment Scaffolding](http://arxiv.org/abs/2509.03310v1)

- app.build (Environment Scaffolding): introduces a production framework for scaling agentic prompt-to-app generation, which wraps LLMs with an Orchestrator, stack-specific Actors, Sandbox Manager, Validation Layer, and Task Runner to provide systematic validation and structured environments for reliable application development.
- The framework combines multi-layered validation pipelines, stack-specific orchestration, and a model-agnostic architecture, implemented across three reference stacks (TypeScript/tRPC, PHP/Laravel, Python/NiceGUI).
- Through evaluation on 30 generation tasks, the framework achieves a 73.3% viability rate with 30% perfect quality scores, demonstrating that scaling reliable AI agents requires scaling environments, not just models.

---

[Loong: Synthesize Long Chain-of-Thoughts at Scale through Verifiers](https://github.com/camel-ai/loong)

- Loong Project: introduces an open-source framework for scalable synthetic data generation and verification, featuring LOONGBENCH (curated seed dataset) and LOONGENV (synthetic data generation environment) with a Generator, Environment, Trainable Agent, and Verifier.
- The framework establishes an agent-environment loop where an LLM-based Generator creates synthetic questions and executable code, which the Environment runs to produce answers, then a Trainable Agent generates Chain-of-Thought solutions, and a Verifier compares these for an RL reward.
- This system aims to overcome the scarcity of high-quality, verifiable datasets in reasoning-intensive domains beyond mathematics and programming, enabling large-scale reinforcement learning with minimal human supervision.

---

[CoreThink: A Symbolic Reasoning Layer to reason over Long Horizon Tasks with LLMs](http://arxiv.org/abs/2509.00971v2)

- CoreThink (General Symbolics Reasoning): introduces a state-of-the-art reasoning layer built upon a novel reasoning method called General Symbolics, with Native Language Parsing & Semantic Preservation, In-Language Reasoning Architecture, Execution & Explainability, Representational Translation Avoidance, Computational Optimization Layer, Agentic Coding IDE, and ARC-AGI-2 Neuro-Symbolic Pipeline, where it provides a pure performance uplift for LLMs on long-horizon reasoning tasks without fine-tuning or training costs.
- The framework operates on a pure natural language-to-natural language basis, avoiding representational loss and brittleness associated with translating human language into formal logic or high-dimensional vectors.
- It achieves state-of-the-art performance across tool-calling, code generation, and planning benchmarks, demonstrating robust capabilities in complex, multi-step algorithmic reasoning and software engineering challenges.

---


[L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search](http://arxiv.org/abs/2509.00761v2)

- L-MARS (Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search): introduces a multi-agent workflow that integrates structured reasoning, agentic search, and sufficiency verification for legal question answering, including a Query Agent (parses, refines queries), Search Agent (executes retrieval tools), Serper (online web search), Local RAG (offline document retrieval), BM25 (ranking function), CourtListener (case law API), Judge Agent (verifies evidence, refines), Summary Agent (synthesizes final answer), WorkflowState (manages system state), and LangGraph (orchestrates workflow), to reduce hallucination and uncertainty in legal QA.
- The system operates in two modes: Simple Mode, a single-pass retrieval-summarization pipeline, and Multi-Turn Mode, which adds Judge Agent-guided iterations with sufficiency checks and query refinement for enhanced accuracy.
- This iterative reasoning-search-verification loop maintains coherence, filters noisy evidence, and grounds answers in authoritative law, demonstrating a scalable and reproducible blueprint for deploying LLMs in high-stakes legal domains.

---

[An Agentic Model Context Protocol Framework for Medical Concept Standardization](http://arxiv.org/abs/2509.03828v1)

- MCP (Model Context Protocol): introduces a zero-training, hallucination-preventive mapping system for medical concept standardization, featuring an Input (user query), an Agentic LLM (interprets, reasons, calls tools), MCP (standardized, secure framework), MCP Resources (contextual guidance, preferences), Reasoning 1: keyword inference (interprets user input), Athena OHDSI API (external vocabulary service), Concept list (candidate medical concepts), Reasoning 2: concept selection (selects best concept), and Output (standardized OMOP concept), which enables explainable mapping and improves efficiency and accuracy in mapping source medical terms to OMOP standard concepts.
- The system leverages LLMs with real-time access to external resources like OHDSI Athena, guided by OMOP data model specifications, documentation, and vocabulary preferences, to mitigate hallucination and ensure clinically appropriate mappings.
- This framework provides a robust, auditable, and user-guided solution for medical terminology mapping, suitable for both exploratory and production environments without requiring fine-tuning or complex infrastructure.

---

[What Would an LLM Do? Evaluating Policymaking Capabilities of Large Language Models](http://arxiv.org/abs/2509.03827v1)

- LLM-ABM Integration Pipeline: introduces a novel benchmark and an automated pipeline to evaluate LLMs' policymaking capabilities for homelessness alleviation, with all its components, where the framework assesses LLM alignment with human experts and simulates policy impacts.
- This framework assesses LLM alignment with human experts on policy choices across four geographies, grounding policies in the Capability Approach for human development.
- The pipeline connects LLM-generated policy proposals to an agent-based model to explore their social impact through simulated scenarios, offering insights into scalable and non-invasive social policymaking.

---

[Designing Gaze Analytics for ELA Instruction: A User-Centered Dashboard with Conversational AI Support](http://arxiv.org/abs/2509.03741v1)

- Gaze Analytics Dashboard with Conversational AI Support: introduces a user-centered dashboard for English Language Arts (ELA) instruction, integrating gaze heatmaps, student performance tables, score trajectories, scanpaths, and an LLM-powered conversational agent for interpreting multimodal learning analytics.
- The system leverages eye-tracking technology to capture student gaze data, which is then visualized and summarized through an LLM-Augmented Report Generation Pipeline to provide actionable insights for teachers and students.
- This iterative design, guided by user feedback and data storytelling principles, aims to make complex gaze analytics approachable and pedagogically valuable, enhancing instructional decision-making and student reflection.

---

[ARE LLM AGENTS BEHAVIORALLY COHERENT? LATENT PROFILES FOR SOCIAL SIMULATION](http://arxiv.org/abs/2509.03736v1)

- Framework for Probing Behavioral Coherence: introduces a study to evaluate the internal consistency of LLM agents by eliciting their internal states (preference and openness) and observing their behavior in dialogue settings, using an LLM-as-judge to score agreement.
- The study reveals significant internal inconsistencies in LLMs across various models and sizes, showing that agents often suppress disagreement and favor positive sentiment, even when explicitly biased.
- These findings highlight a critical gap in LLM capabilities, as agents fail to maintain behavioral coherence over time, questioning their reliability as substitutes for human participants in social science research.

---

[Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents](http://arxiv.org/abs/2509.03581v1)

- Dynamic Planning Agent Architecture: introduces a framework for LLM agents to dynamically allocate test-time compute for planning, with LLM Agent (unified output generator), Context (agent's internal state), New Observation (current environment state), Previous Plan (prior strategic guidance), Implicit Plan Decision (decides planning necessity), Planning Policy (generates new strategic plan), Acting Policy (executes next action), Generated Output (action or plan and action), Action (command to environment), New Plan (updated strategic guidance), and Environment (sequential decision-making tasks).
- This architecture enables a single, monolithic LLM to implicitly decide when to plan by generating a `<plan>` token, then parsing its unified output to extract both the action and, if present, the new plan.
- The framework is trained using a two-stage pipeline of supervised fine-tuning and reinforcement learning, allowing agents to learn strategic planning, plan execution, and replanning only when necessary, optimizing computational resource allocation.

---

[Loong: Synthesize Long Chain-of-Thoughts at Scale through Verifiers](https://github.com/camel-ai/loong)

- Loong Project: introduces an open-source framework for scalable synthetic data generation and verification, featuring LOONGBENCH (curated seed dataset) and LOONGENV (synthetic data generation environment) with a Generator, Environment, Trainable Agent, and Verifier.
- The framework establishes an agent-environment loop where an LLM-based Generator creates synthetic questions and executable code, which the Environment runs to produce answers, then a Trainable Agent generates Chain-of-Thought solutions, and a Verifier compares these for an RL reward.
- This system aims to overcome the scarcity of high-quality, verifiable datasets in reasoning-intensive domains beyond mathematics and programming, enabling large-scale reinforcement learning with minimal human supervision.

---

[Loong: Synthesize Long Chain-of-Thoughts at Scale through Verifiers](https://github.com/camel-ai/loong)

- Loong Project: introduces an open framework for scalable synthetic data generation and verification, featuring LOONGBENCH (curated seed dataset), LOONGENV (synthetic data generation environment), Generator (produces synthetic data), Environment (executes generated code), Trainable Agent (generates CoT solutions), Verifier (compares answers, provides reward), Seed Datasets (initial input), Question Synthesis Agent (generates questions), Code Generation Agent (produces executable code), and Judge Agent (assesses correctness), enabling reinforcement learning with verifiable rewards for LLMs.
- The framework leverages an agent-environment loop where a Generator produces synthetic questions and executable code, which is then executed in an Environment to yield answers, and a Trainable Agent's Chain-of-Thought solutions are verified against these code-generated answers by a Verifier.
- LOONGENV, utilizing LLMs like GPT-4.1-mini for question and code generation and DeepSeek-R1 as a Judge Agent, supports diverse prompting strategies to create high-quality, verifiable question-answer pairs for reasoning-intensive domains.

---

[Adversarial Decision-Making in Partially Observable Multi-Agent Systems: A Sequential Hypothesis Testing Approach](http://arxiv.org/abs/2509.03727v1)

- SHT-driven Framework (Sequential Hypothesis Testing-driven Framework): introduces a novel approach for adversarial decision-making in partially observable multi-agent systems, modeling deception as a dynamic optimization problem within a leader-follower Stackelberg game.
- This framework integrates a Blue Team Agent (follower) that uses a Linear-Quadratic Control Framework and an Optimal Control Solution Module for strategic misdirection, and a Red Team Agent (leader) that employs a Sequential Hypothesis Testing (SHT) Module and a Red Team Strategy Optimization Module (using FPI, NN, or FBS algorithms) for counter-deception.
- The system utilizes Stochastic Differential Equations (SDEs) for state dynamics, Cost Functionals for objective quantification, a Likelihood Ratio Statistic (LT) for deception effectiveness, and a Regularization Penalty Term to model skepticism, providing insights into strategic deception and counter-deception.

---

[AutoGrid AI: Deep Reinforcement Learning Framework for Autonomous Microgrid Management](http://arxiv.org/abs/2509.03666v1)

- AutoGrid AI (Deep Reinforcement Learning Framework for Autonomous Microgrid Management): introduces a deep reinforcement learning framework for autonomous microgrid management, integrating transformer architecture for forecasting renewable generation and a PPO agent for decision-making in a simulated environment.
- The framework optimizes microgrid energy dispatch strategies to minimize costs and maximize renewable energy utilization, demonstrating improvements in energy efficiency and operational resilience compared to rule-based methods.
- It also provides an open-source framework for simulating various microgrid environments, supporting the development of zero-carbon energy systems.

---

#### 2nd September 2025


[AppCopilot: Toward General, Accurate, Long-Horizon, and Efficient Mobile Agent](https://arxiv.org/abs/2509.02444)

- AppCopilot: introduces a multimodal, multi-agent, general-purpose on-device assistant, with Multimodal Foundation Models (core for perception, reasoning, action), OCR+OR Module (identifies UI elements, bounding boxes), Multi-Agent Collaborative Decision-Making Strategy (aggregates actions from multiple agents), Reinforcement Learning (optimizes long-horizon task policies), High-level Planning Agent (decomposes tasks, allocates resources), Personalized Information Memory and Retrieval Mechanism (stores and retrieves user preferences), Experience Reuse Framework (replays historical successful tasks), and Hybrid Control Framework (integrates GUI and API control), which operationalizes an end-to-end autonomous pipeline for mobile agents from data to deployment, addressing generalization, accuracy, long-horizon capability, and efficiency.
- The system integrates multimodal foundation models for robust Chinese-English support, combining chain-of-thought reasoning, hierarchical task planning, and multi-agent collaboration at the reasoning and control layer.
- At the execution layer, it enables user personalization, voice interaction, function/tool calling, cross-app and cross-device orchestration, and comprehensive mobile app support, incorporating profiling-driven optimization for latency, memory, and energy.

---


[Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving](https://wizard-wang01.github.io/LLM2AD/)

- LLM-to-Driving Transfer Analysis: introduces a systematic evaluation of five key LLM modules—tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation—within the context of motion generation for autonomous driving.
- The study demonstrates that, when appropriately adapted, these modules can significantly improve performance for autonomous driving motion generation, identifying effective transfer techniques and necessary domain-specific adaptations.
- The research provides insights into the transferability of LLM modules, validating their effectiveness on the Waymo Sim Agents benchmark and achieving competitive results.

---

[The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](http://arxiv.org/abs/2509.02547v1)

- Agentic RL: introduces a paradigm for LLMs, reframing them as autonomous decision-making agents with Planning, Tool Use, Memory, Self-Improvement, Reasoning, and Perception modules, enabling complex, dynamic interactions.
- This approach contrasts with traditional LLM-RL by optimizing semantic-level behaviors in variable, partially observable environments through sequential decision-making loops.
- Reinforcement learning serves as the critical mechanism to transform these core capabilities from static, heuristic modules into adaptive, robust agentic behavior for general-purpose AI.

---

[UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](http://arxiv.org/abs/2509.02544v1)

- UI-TARS-2: introduces a native GUI-centered agent model, with a data flywheel (systematic training methodology), an all-in-one GUI sandbox (unified sandbox platform), a multi-turn reinforcement learning framework (stabilized RL framework), and parameter interpolation (merging specialized agents), designed to handle structured computer-use tasks and dynamic, game-like interactive environments.
- The framework employs a ReAct paradigm for agent formulation, interleaving reasoning, action, and observation, supported by a hierarchical memory state for context preservation and an interactive annotation platform for human-in-the-loop data generation.
- Its multi-turn RL framework incorporates advanced techniques like asynchronous agent rollout, stateful environment integration, reward shaping, and value pretraining to ensure stable and efficient learning across diverse, long-horizon tasks.

---

[FlexNGIA 2.0: Redesigning the Internet with Agentic AI Protocols, Services, and Traffic Engineering Designed, Deployed, and Managed by AI](http://arxiv.org/abs/2509.02124v1)

- FlexNGIA 2.0: introduces an Agentic AI-driven Internet architecture that leverages LLM-based AI agents, encompassing agents for application analysis, information fusion, SFC and protocol design, congestion control, resource allocation, monitoring, and failure management, to autonomously orchestrate, configure, and evolve the network.
- Each LLM-based agent is equipped with a Brain (LLM for reasoning), Memory (persistent context store), Planning (task decomposition), and Tools (external system interface), enabling autonomous, context-aware decision-making and real-time adaptation to dynamic network conditions and application demands.
- The framework redefines network architecture by embedding cognitive intelligence, allowing AI agents to redesign network protocols, logic, and algorithms on the fly, thereby delivering flexibility, intelligence, and responsiveness for diverse and evolving application requirements.

---

[Semi-on-Demand Transit Feeders with Shared Autonomous Vehicles and Reinforcement-Learning-Based Zonal Dispatching Control](http://arxiv.org/abs/2509.01883v1)

- SoD-RL Zonal Dispatching (Semi-on-Demand Transit Feeder Service with Reinforcement Learning-Based Zonal Dispatching Control): introduces a novel transit feeder service, with RL Model (decision-making agent) comprising Policy (Actor Model) (proposes actions) and Value Function (Critic Model) (evaluates actions), interacting with FleetPy Simulation (Environment) (simulates transit system) which includes Fleet Control (manages vehicle operations), Network (provides travel data), and Trip offer (processes passenger requests), through State (system observation), Action (dispatching decision), and Reward (performance feedback).
- This framework dynamically assigns Shared Autonomous Vehicles (SAVs) to subdivided flexible-route zones using a deep RL policy gradient algorithm (Proximal Policy Optimization) to respond to real-time demand fluctuations and operational needs.
- The system aims to maximize passengers served while maintaining frequent service on fixed-route portions, demonstrating improved efficiency and passenger service compared to traditional fixed-route and nominal semi-on-demand services.

---

[Safety-Critical Multi-Agent MCTS for Mixed Traffic Coordination at Unsignalized Roundabout](http://arxiv.org/abs/2509.01856v1)

- SC-MCTS (Safety-Critical Multi-Agent Monte Carlo Tree Search): introduces a safety-critical decision-making framework for autonomous vehicles navigating unsignalized, dual-lane roundabouts, with Problem Formulation as MDP, Safety-Critical Decision Making, Multi-Agent MCTS, and Reward Function Design and Optimization components, enabling cooperative decision-making by integrating deterministic and probabilistic prediction models.
- The framework employs a hierarchical safety assessment module to address AV-to-AV, AV-to-HDV, and AV-to-Road interactions through dynamic safety thresholds and spatiotemporal risk evaluation.
- An adaptive HDV behavior prediction scheme, combining the Intelligent Driver Model with probabilistic uncertainty modeling, and a multi-objective reward optimization strategy jointly considering safety, efficiency, and cooperative intent, further enhance the system's robustness in mixed traffic.

---

[BUILDING SELF-EVOLVING AGENTS VIA EXPERIENCE-DRIVEN LIFELONG LEARNING: A FRAMEWORK AND BENCHMARK](https://github.com/ECNU-ICALK/ELL-StuLife)

- Experience-driven Lifelong Learning (ELL): introduces a framework for building self-evolving agents capable of continuous growth through real-world interaction, incorporating Experience Exploration (self-motivated interaction), Long-term Memory (persistent knowledge storage), Skill Learning (reusable pattern abstraction), and Knowledge Internalization (explicit-to-intuitive transformation) as core principles.
- The ELL framework operates as a continuous learning cycle where an agent interacts with its dynamic environment, processes experiences through knowledge abstraction and refinement, and validates the resulting knowledge.
- To evaluate these agents, the paper introduces StuLife, a benchmark simulating a student's college journey, designed to assess lifelong learning capabilities, memory retention, skill transfer, and self-motivated behavior in complex, dynamic environments.

---

[Plan Verification for LLM-Based Embodied Task Completion Agents](https://github.com/AnanthHariharan/Task-Agents)

- Plan Verification Framework: introduces an iterative verification framework for LLM-based embodied task completion agents, featuring a Planning Agent (generates and revises plans), a Judge LLM (critiques action sequences), and an Iterative Refinement Loop (manages repeated critique and revision).
- This framework enables the Planning Agent to generate candidate plans and the Judge LLM to analyze and flag erroneous actions, such as redundant, contradictory, or missing steps, with natural language explanations.
- The iterative process refines action sequences, leading to progressively cleaner and more spatially coherent trajectories, thereby providing higher-quality training data for imitation learning in embodied AI.

---

[Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving](https://wizard-wang01.github.io/LLM2AD/)

- LLM Modules for Autonomous Driving Motion Generation: introduces a systematic evaluation of five key LLM modules—tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation—within the context of motion generation for autonomous driving.
- The study demonstrates that, when appropriately adapted, these LLM modules can significantly improve performance for autonomous driving motion generation, identifying effective transfer techniques and necessary domain-specific adaptations.
- The research provides insights into the transferability of LLM components, highlighting similarities and subtle differences between LLMs and autonomous driving motion generation, and validates adapted modules on the Waymo Sim Agents benchmark.

---

[Deep Research is the New Analytics System: Towards Building the Runtime for AI-Driven Analytics](http://arxiv.org/abs/2509.02751v1)

- Our Prototype (extending Palimpzest): introduces a new runtime for AI-driven analytics, combining Deep Research flexibility with optimized semantic operator execution through `PZ.Context`, `Search Operator`, `Compute Operator`, `CodeAgent`s, `Semantic Operators`, `ContextManager`, and `Tools` components.
- The prototype leverages `CodeAgent`s to physically implement `Search` and `Compute` operators, enabling dynamic planning and execution of optimized semantic operator programs over large unstructured datasets.
- By introducing the `PZ.Context` abstraction with indexing and user-defined `Tools`, and a `ContextManager` for caching, the system aims to improve query performance and reduce computational costs for AI-driven analytics.

---

[Contemporary Agent Technology: LLM-Driven Advancements vs Classic Multi-Agent Systems](http://arxiv.org/abs/2509.02515v1)

- LLM-based Agent Architecture: introduces a comprehensive reflection on contemporary agent technology, contrasting LLM-driven advancements with classic Multi-Agent Systems by detailing the architectural pillars that define these new systems.
- The paper critically analyzes how recent LLM developments relate to foundational Multi-Agent Systems (MAS) concepts, models, and characteristics, emphasizing the shift from symbolic to sub-symbolic AI.
- It identifies key challenges and promising future directions in this rapidly evolving domain, highlighting the need for standardization and robust hybrid systems that combine formal principles with adaptive reasoning.

---

[GridMind: LLMs-Powered Agents for Power System Analysis and Operations](http://arxiv.org/abs/2509.02494v1)

- GridMind: introduces a multi-agent AI system that integrates LLMs with deterministic engineering solvers to enable conversational scientific computing for power system analysis, with a Planner/Coordinator (orchestrates agents, workflows), ACOPF Agent (handles AC Optimal Power Flow), CA Agent (performs T-1 reliability assessment), LLM (provides core reasoning), Memory (maintains analytical coherence), Tools (invokes deterministic solvers), Grid Data (provides power system information), Secure Access Data (ensures secure data retrieval), and Conversational Interface (manages user interaction).
- The system employs specialized agents coordinating AC Optimal Power Flow and N-1 contingency analysis through natural language interfaces, maintaining numerical precision via function calls to external tools and solvers.
- GridMind addresses workflow integration, knowledge accessibility, context preservation, and expert decision-support augmentation, demonstrating how conversational interfaces can enhance accessibility while preserving numerical rigor for critical engineering applications.

---

[KUBEINTELLECT: A MODULAR LLM-ORCHESTRATED AGENT FRAMEWORK FOR END-TO-END KUBERNETES MANAGEMENT](http://arxiv.org/abs/2509.02449v1)

- KubeIntellect: introduces a modular LLM-orchestrated agent framework for end-to-end Kubernetes management, featuring a User Interaction Layer, Query Processing Module, Task Orchestration Module with Memory, an Agent & Tool Execution Layer with specialized agents (including a Code Generator Agent), a Kubernetes Interaction Layer, and a Supporting System with an LLM Gateway, Persistent Context Service, and Security & Governance.
- This framework enables natural language interaction for comprehensive Kubernetes API operations, supporting dynamic tool synthesis, structured workflows, human-in-the-loop clarification, and secure execution across diverse workloads.
- KubeIntellect integrates memory checkpoints and a LangGraph-based orchestration engine, achieving a 93% tool synthesis success rate and 100% reliability in managing complex Kubernetes infrastructure.

---

[BioBlue: Notable runaway-optimiser-like LLM failure modes on biologically and economically aligned AI safety benchmarks for LLMs with simplified observation format](http://arxiv.org/abs/2509.02655v1)

- BioBlue: introduces a set of benchmarks for evaluating LLMs in long-running scenarios, including Sustainability (resource balance evaluation), Single-objective homeostasis (single metric stability), Multi-objective homeostasis (multiple metric stability), and Balancing unbounded objectives with diminishing returns (multi-goal optimization).
- These benchmarks reveal systematic runaway-optimiser-like failure modes in LLMs, where models default to unbounded single-objective maximization and neglect homeostatic targets, even after periods of initial success.
- The findings suggest that current LLMs, despite appearing multi-objective and bounded, exhibit underlying biases towards single-objective and unbounded optimization in sustained tasks.

---

[Towards Agents That Know When They Don't Know: Uncertainty as a Control Signal for Structured Reasoning](http://arxiv.org/abs/2509.02401v1)

- Uncertainty-Aware Agent Framework: introduces an LLM agent for query-conditioned multi-table summarization, leveraging retrieval uncertainty, summary uncertainty, and reinforcement learning with GRPO to filter outputs and construct high-quality synthetic datasets.
- The framework refines the agent's policy during training using reward signals based on code execution, LLM-judge scores, and summary confidence, while inference involves sampling multiple trajectories and filtering based on combined uncertainty scores.
- This approach enables agents to abstain from uncertain claims, communicate confidence, and become more reliable for complex structured-data environments, improving factuality, calibration, and downstream utility in biomedical multi-omics tasks.

---

[When Agents go Astray: Course-Correcting SWE Agents with PRMs](http://arxiv.org/abs/2509.02360v1)

- SWE-PRM (Process Reward Model): introduces an inference-time Process Reward Model that intervenes during execution to detect and course-correct trajectory-level errors, with Policy Model, Problem Description, Tool Instructions, Repository, Transcript, SWE-PRM (Process Reward Model), Taxonomy of Inefficiencies, Error Detection, Evidence Generation, Recovery Action, and Guidance Generation, where the framework prevents, detects, and course-corrects trajectory-level errors in LLM-based software engineering agents.
- The framework leverages a taxonomy of common inefficiencies to deliver lightweight, interpretable feedback without modifying the underlying policy of the LLM agent, improving reliability and efficiency in complex, multi-step software engineering tasks.
- This real-time error correction mechanism provides actionable guidance to steer the agent toward efficient completion, significantly boosting resolution rates on medium and hard tasks while reducing trajectory length.

---

[RumorSphere: A Framework for Million-scale Agent-based Dynamic Simulation of Rumor Propagation](http://arxiv.org/abs/2509.02172v1)

- RumorSphere: introduces a novel dynamic and hierarchical social network simulation framework, with an Agent Layer (distinguishes core and regular agents) comprising LLM-driven Core Agents (complex decision-making) (featuring a Persona Module (defines demographic attributes), Memory Module (stores personal environmental memory) with Personal Memory (user's historical behavior), Environmental Memory (observations and insights), Retrieval (guides behavior), Update (stores observations), and Reflection (promotes high-level thinking), an Action Module (enables agent actions), and a Belief State (represents opinion certainty)) and ABM-based Regular Agents (simpler opinion updates) (defined by Opinion (continuous belief score), fupdate (defines opinion change), fselection (determines influencing agents), and fmessage (determines transmitted message)); and an Interaction Layer (manages agent partitioning communication) that employs a Dynamic Interaction Strategy (DIS) (adaptively partitions agents) (with Adaptive Grouping (AG) (identifies core agents) and Dynamic Communication (DC) (determines communication modes)) and a Hierarchical Collaborative Network (HCN) (initializes agent network topology) (using Preferential Attachment (fosters opinion leaders) and Triangle Connection (prioritizes community links)).
- The framework supports million-scale simulations by adaptively partitioning agents into LLM-driven core agents for complex reasoning and ABM-based regular agents for efficiency, dynamically adjusting interactions based on information confusion.
- RumorSphere enables counterfactual experiments to evaluate intervention strategies, revealing that early, sustained, and opinion leader-based debunking is most effective in mitigating rumor spread within tightly connected local communities.

---

[Batch Query Processing and Optimization for Agentic Workflows](http://arxiv.org/abs/2509.02121v1)

- Halo: introduces a system for batch query processing and optimization in agentic LLM workflows, comprising a Query Parser (parses queries into DAG), a Query Optimizer (generates execution plan), and a Query Processor (executes optimized plan).
- Halo unifies query optimization with LLM serving by representing workflows as structured query plan DAGs and constructing a consolidated graph for batched queries to expose shared computation.
- The system's runtime integrates adaptive batching, KV-cache sharing and migration, and compute-communication overlap, guided by a cost model, to maximize hardware efficiency and achieve significant speedups.

---

[JUDGEAGENT: DYNAMICALLY EVALUATE LLMS WITH AGENT-AS-INTERVIEWER](http://arxiv.org/abs/2509.02097v1)

- JudgeAgent: introduces a knowledge-target adaptive dynamic evaluation framework, with Target LLM (evaluated model), Core LLM Agent (generator/evaluator LLM), Benchmark Grading (initial capability assessment), Interactive Extension (dynamic question generation/testing), Evaluation Feedback (result aggregation/suggestions), Base Datasets (static question source), Context Graph (knowledge representation), Difficulty-Adaptive Module (adjusts question difficulty), Question Synthesis Module (generates new questions), Q&A History (stores interaction history), and Evaluation Scoring Module (computes performance scores), which dynamically evaluates LLMs using an interviewer-style paradigm.
- The framework conducts comprehensive evaluations through benchmark grading, interactive extension with knowledge-driven data synthesis and target-adaptive difficulty adjustment, and provides interpretable evaluation feedback.
- JudgeAgent offers novel insights into validating evaluation methods by comparing accuracy before and after receiving suggestions, demonstrating its effectiveness in identifying and mitigating LLM knowledge and capability gaps.

---

[Plan Verification for LLM-Based Embodied Task Completion Agents](http://arxiv.org/abs/2509.02761v1)

- Plan Verification Framework: introduces a general-purpose language-based plan verification framework that incorporates iterative critique and revision into embodied planning workflows, with a Planning Agent (generates and revises plans), a Judge LLM (critiques action sequences), and an Iterative Refinement Loop (repeats critique and revision).
- The framework operates by having the Planning Agent generate a candidate plan for a given goal, which the Judge LLM then analyzes step-by-step to flag redundant, irrelevant, contradictory, or missing actions, providing natural language explanations.
- The Planning Agent revises its plan based on the Judge LLM's feedback, and this process continues iteratively until no further objections are raised, enabling scalable and interpretable refinement of noisy human-authored task plans.

---

[Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving](https://wizard-wang01.github.io/LLM2AD/)

- LLM-to-Driving Transfer Analysis: introduces a systematic evaluation of five key LLM modules—tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation—within the context of motion generation for autonomous driving.
- The study demonstrates that, when appropriately adapted, these modules can significantly improve performance for autonomous driving motion generation, identifying effective transfer techniques and necessary domain-specific adaptations.
- The research provides insights into the transferability of LLM modules, validating their effectiveness on the Waymo Sim Agents benchmark and achieving competitive results.

---

[Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving](https://wizard-wang01.github.io/LLM2AD/)

- LLM Modules for Autonomous Driving Motion Generation: introduces a systematic evaluation of five key LLM modules—tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation—within the context of motion generation for autonomous driving.
- The study demonstrates that, when appropriately adapted, these LLM modules can significantly improve performance for autonomous driving motion generation, identifying effective transfer techniques and necessary domain-specific adaptations.
- The research provides insights into the transferability of LLM components, highlighting similarities and subtle differences between LLMs and autonomous driving motion generation, and validates adapted modules on the Waymo Sim Agents benchmark.

---

[Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving](https://wizard-wang01.github.io/LLM2AD/)

- LLM-to-Driving Transfer Analysis: introduces a systematic evaluation of five key LLM modules—tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation—within the context of motion generation for autonomous driving.
- The study demonstrates that, when appropriately adapted, these modules can significantly improve performance for autonomous driving motion generation, identifying effective transfer techniques and necessary domain-specific adaptations.
- The research provides insights into the transferability of LLM modules, validating their effectiveness on the Waymo Sim Agents benchmark and achieving competitive results.

---

[Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving](https://wizard-wang01.github.io/LLM2AD/)

- LLM Modules for Autonomous Driving Motion Generation: introduces a systematic evaluation of five key LLM modules—tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation—within the context of motion generation for autonomous driving.
- The study demonstrates that, when appropriately adapted, these LLM modules can significantly improve performance for autonomous driving motion generation, identifying effective transfer techniques and necessary domain-specific adaptations.
- The research provides insights into the transferability of LLM components, highlighting similarities and subtle differences between LLMs and autonomous driving motion generation, and validates adapted modules on the Waymo Sim Agents benchmark.

---

[Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving](https://wizard-wang01.github.io/LLM2AD/)

- LLM Modules for Autonomous Driving Motion Generation: introduces a systematic evaluation of five key LLM modules—tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation—within the context of motion generation for autonomous driving.
- The study demonstrates that, when appropriately adapted, these LLM modules can significantly improve performance for autonomous driving motion generation, identifying effective transfer techniques and necessary domain-specific adaptations.
- The research provides insights into the transferability of LLM components, highlighting similarities and subtle differences between LLMs and autonomous driving motion generation, and validates adapted modules on the Waymo Sim Agents benchmark.

---

[BUILDING SELF-EVOLVING AGENTS VIA EXPERIENCE-DRIVEN LIFELONG LEARNING: A FRAMEWORK AND BENCHMARK](https://github.com/ECNU-ICALK/ELL-StuLife)

- ELL (Experience-driven Lifelong Learning): introduces a framework for building self-evolving AI agents capable of continuous growth through real-world interaction, with Perception, Memory, Learning, Reasoning, and Action components, enabling agents to learn from experience, integrate long-term memory, and abstract skills.
- The framework is built on four core principles: Experience Exploration, Long-term Memory, Skill Learning, and Knowledge Internalization, which guide the agent's autonomous adaptation and self-improvement.
- The paper also introduces StuLife, a benchmark dataset simulating a student's college journey across various tasks and scenarios, designed to evaluate lifelong learning capabilities, memory retention, skill transfer, and self-motivated behavior for LLMs.

---

[Diffusion-RL Based Air Traffic Conflict Detection and Resolution Method](http://arxiv.org/abs/2509.03550v1)

- Diffusion-AC: introduces a novel autonomous conflict resolution framework that integrates diffusion probabilistic models into safety-critical air traffic Conflict Detection and Resolution (CD&R), generating multimodal action distributions via a value-guided reverse denoising process, and employing a Density-Progressive Safety Curriculum (DPSC) for stable learning.
- The framework's core architecture includes a UNet-style denoising backbone with residual blocks and self-attention, a state encoder, and a time embedding module, all trained in an off-policy actor-critic fashion with dual Q-critics and target networks.
- This approach overcomes the unimodal bias of traditional DRL policies, significantly enhancing decision-making flexibility and robustness in complex, high-density air traffic scenarios, leading to a 94.1% success rate and a 59% reduction in Near Mid-Air Collisions.

---


#### 1st September 2025

[Structured AI Decision-Making in Disaster Management](http://arxiv.org/abs/2509.01576v1)

- Structured AI Decision-Making Framework: introduces a structured decision-making framework for autonomous AI in disaster management, featuring Enabler agents (AI models providing judgment insights), Decision Maker agents (RL algorithms or human operators), Levels (critical decision points), and Scenarios (tree-like decision structures).
- The framework organizes decision flow into distinct Levels within a Scenario, where Enabler agents process disaster-related data to provide confidence scores, guiding the Decision Maker agent (either an RL algorithm or a human operator) in making informed decisions.
- The Enabler agent utilizes a Multimodal Model Architecture, combining a Text Model (BiLSTM with pooling) and an Image Model (ResNet50) to classify image-text pairs, while the RL Decision Maker agent is trained using an A2C algorithm within a custom Gymnasium environment.

---

[LLM-empowered Agents Simulation Framework for Scenario Generation in Service Ecosystem Governance](http://arxiv.org/abs/2509.01441v1)

- LLM-empowered Agents Simulation Framework for Scenario Generation in Service Ecosystem Governance: introduces a scenario generator design method, which adaptively coordinates three LLM-empowered agents—Planner Agent (PA) (coordinates schemes), Environment Agent (EA) (generates environments), and Social Agent (SA) (models agent behaviors)—along with a Data/Knowledge Base (input), Tasks (objectives), Experiment System (executes scenarios), Scenarios (outputs), and Feedback Mechanism (adjustment) to optimize experimental schemes and generate high-quality scenarios for service ecosystem governance.
- The framework leverages LLMs for semantic deconstruction, adversarial prompt engineering, and cognitive simulations to overcome limitations of predefined rules and generate diverse, extreme scenarios.
- The system's closed-loop "generate-validate-optimize" mechanism enables adaptive governance of complex service ecosystems under uncertainty, improving scenario generation efficiency and feature coverage.

---

[Conformal Predictive Monitoring for Multi-Modal Scenarios](http://arxiv.org/abs/2509.01338v1)

- GenQPM: introduces a dynamics-aware quantitative predictive monitor, with a Generative Model (learns system dynamics, generates trajectories), Mode Predictor (partitions trajectories by mode), Conformal Inference (ensures statistical guarantees), STL Robustness Calculation (quantifies property satisfaction), and Prediction Intervals (mode-specific robustness ranges), which leverages deep generative models and conformal inference for mode-specific predictive monitoring in multi-modal stochastic systems.
- This method addresses the limitation of existing quantitative predictive monitoring approaches by providing statistically valid, mode-specific prediction intervals for Signal Temporal Logic (STL) robustness, enhancing decision-making in complex dynamic environments.
- The approach offers improved interpretability and tighter prediction intervals compared to mode-agnostic baselines, enabling preemptive and timely safety interventions in systems with uncertain future behaviors.

---

[Multi-Agent Reinforcement Learning for Task Offloading in Wireless Edge Networks](http://arxiv.org/abs/2509.01257v1)

- DCC (Decentralized Coordination via CMDPs) Framework: introduces a decentralized multi-agent reinforcement learning framework for task offloading in wireless edge networks, with agents solving local CMDPs and coordinating implicitly through a shared constraint vector updated via a three-timescale learning process.
- The framework employs lightweight communication and constraint-based coupling to achieve system-level alignment while ensuring local autonomy and scalability in shared-resource environments.
- Each agent uses a reinforcement learning algorithm and Lagrange multipliers to balance individual performance objectives with global resource usage constraints, addressing challenges like non-decomposability and non-stationarity in MARL.

---

[DeepSeek Performs Better Than Other Large Language Models in Periodontal Cases](http://arxiv.org/abs/2509.02036v1)

- LLM Evaluation Framework for Periodontal Cases: introduces a system for assessing LLMs in dental case analysis, comprising a Dental Clinical Cases Collection (source of clinical data), Three-Step Prompt Design (structures LLM input), 30% Downsampling (selects subset for testing), LLMs (models being compared), Algorithm Evaluation (automated performance metrics), and Human Evaluation (expert clinical assessment).
- The framework systematically evaluates four prominent LLMs (GPT-4o, Gemini 2.0 Flash, Copilot, and DeepSeek V3) on their ability to interpret complex longitudinal periodontal case vignettes and generate professionally appropriate open-ended responses.
- DeepSeek V3 consistently demonstrated superior performance in faithfulness and expert clinical accuracy compared to other LLMs, highlighting its potential as a robust, domain-specific clinical decision-support tool for dental education and practice.

---

[From CVE Entries to Verifiable Exploits: An Automated Multi-Agent Framework for Reproducing CVEs](http://arxiv.org/abs/2509.01835v1)

- CVE-GENIE: introduces an automated, LLM-based multi-agent framework for reproducing real-world vulnerabilities from CVE entries, with all its components, where it gathers relevant resources, reconstructs the vulnerable environment, and reproduces a verifiable exploit.
- The framework's modular design, including Processor, Builder, Exploiter, and CTF Verifier, enables end-to-end CVE reproduction by specialized LLM agents, addressing challenges like incomplete data and reasoning limits through self-critique.
- CVE-GENIE successfully reproduced approximately 51% of CVEs published in 2024-2025, generating verifiable exploits and offering a robust method for creating reproducible CVE benchmarks for security research.

---

[ShortageSim: Simulating Drug Shortages under Information Asymmetry](http://arxiv.org/abs/2509.01813v1)

- ShortageSim (Large Language Model-based multi-agent simulation framework): introduces a multi-agent simulation framework for drug shortage management, featuring an Environment module, an Agents system with FDA, Manufacturer, and Buyer agents, an Information Flow, and a Simulation Controller, where each agent employs a two-stage LLM pipeline for decision-making.
- This framework models the complex, strategic interactions between drug manufacturers, institutional buyers, and regulatory agencies under information asymmetry, leveraging LLMs to simulate bounded-rational decision-making in response to shortage alerts.
- The system enables counterfactual policy analysis of FDA communication strategies and market structures, providing a novel computational framework for designing and testing interventions in information-scarce supply chains.

---

[An LLM-enabled semantic-centric framework to consume privacy policies](http://arxiv.org/abs/2509.01716v1)

- An LLM-enabled semantic-centric framework: introduces a system that automatically converts natural-language privacy policies into formal knowledge, utilizing an NLP pipeline with LLMs to identify and classify privacy-related entities and actions, and construct a Pr² Graph.
- The framework's NLP pipeline includes components for segmenting policies, recognizing and classifying data, purpose, party, and action entities, and identifying relations between them, all powered by LLMs and grounded in the Data Privacy Vocabulary (DPV).
- The resulting Pr² Graph serves as a structured representation of privacy practices, enabling downstream tasks such as constructing formal policies in ODRL or psDToU, and is publicly released for top-100 websites along with the pipeline and datasets.

---

[In-N-Out: A Parameter-Level API Graph Dataset for Tool Agents](http://arxiv.org/abs/2509.01560v1)

- In-N-Out (A Parameter-Level API Graph Dataset for Tool Agents): introduces a novel parameter-level API graph dataset, constructed through a multi-stage pipeline including documentation refinement, candidate pair filtering, and human annotation, to capture API dependencies for LLM-based tool agents.
- The dataset significantly improves tool retrieval and multi-tool query generation performance by providing explicit API dependency information, outperforming LLMs relying solely on documentation.
- Fine-tuning LLMs on In-N-Out enables them to infer parameter-level connections from documentation, generalize to unseen APIs, and achieve performance comparable to human-labeled graphs.

---

[Cloud-Device Collaborative Agents for Sequential Recommendation](http://arxiv.org/abs/2509.01551v1)

- CDA4Rec (Cloud-Device Collaborative Agents for Sequential Recommendation): introduces a novel cloud-device collaborative framework for sequential recommendation, featuring a Cloud Agent (global planning, semantic tasks) and a Device Agent (local processing, sensitive tasks), which collaboratively plan and execute personalized recommendations.
- This framework decomposes the recommendation task into sub-tasks like User Abstract Generation (summarizes user intent, behavior), Recommendation Strategy Planning (generates personalized execution plan), Semantic User Modeling (constructs intent-aware embedding), Candidate Retrieval (generates relevant item set), Structured User Modeling (captures behavioral patterns), and Final Ranking (ranks candidate items).
- CDA4Rec addresses privacy concerns, real-time responsiveness, and computational bottlenecks by dynamically assigning tasks to either the cloud-side LLM or device-side SLM based on computational demands and privacy sensitivity, ensuring efficient and adaptive personalization.

---

[Agentic Workflow for Education: Concepts and Applications](http://arxiv.org/abs/2509.01517v1)

- AWE (Agentic Workflow for Education): introduces a four-component model comprising self-reflection (iterative refinement), tool invocation (external resource use), task planning (sequential decomposition), and multi-agent collaboration (distributed intelligence), enabling dynamic, nonlinear workflows for educational applications.
- This framework distinguishes itself from traditional LLM-based linear interactions by proposing a theoretical foundation grounded in the von Neumann Multi-Agent System (MAS) architecture, shifting from static prompt-response to dynamic, nonlinear workflows.
- AWE enables scalable, personalized, and collaborative task execution across four core application domains: integrated learning environments, personalized AI-assisted learning, simulation-based experimentation, and data-driven decision-making, validated by automated math test generation.

---

[The Need for Verification in AI-Driven Scientific Discovery](http://arxiv.org/abs/2509.01398v1)

- AI-Driven Scientific Discovery Landscape: introduces a comprehensive review of computational methods for scientific discovery, encompassing traditional and AI-assisted pipelines, data-driven, knowledge-aware, derivable models, and LLM approaches, emphasizing the critical role of rigorous verification.
- The paper highlights the "verification bottleneck" in AI-assisted discovery, where rapid hypothesis generation by LLMs and other AI models outpaces the slow, manual evaluation by domain experts, hindering scientific progress.
- It advocates for improved verification methods, including automated and integrated approaches, to ensure scientific validity, interpretability, and alignment with foundational knowledge across diverse scientific domains.

---

[DeepResearch Arena: The First Exam of LLMs' Research Abilities via Seminar-Grounded Tasks](http://arxiv.org/abs/2509.01396v1)

- DeepResearch Arena: introduces a novel benchmark for evaluating LLMs' research abilities, featuring the MAHTG (Multi-Agent Hierarchical Task Generation) system, which includes Data Generation, Inspiration Extraction, Task Design, and Evaluation components, to create and assess research tasks.
- The MAHTG system processes seminar videos into transcripts, extracts categorized inspirations via an Inspira Agent and Expert Verification Team, then designs high-quality DeepResearch Tasks using TaskWeaver and RankEval Agents.
- The benchmark evaluates LLM performance through Keypoint-Aligned Evaluation (KAE) for factual correctness and Adaptively-generated Checklist Evaluation (ACE) for open-ended reasoning, both utilizing a Judge LLM.

---

[TopoNav: Topological Graphs as a Key Enabler for Advanced Object Navigation](http://arxiv.org/abs/2509.01364v1)

- TopoNav (Topological Graphs for Advanced Object Navigation): introduces a novel framework that constructs and maintains a dynamic topological memory graph as the core of its navigation system, integrating RGB-D Images, Pose, Semantic Segmentation, Semantic Point Cloud Map Construction, Topological-Based Memory Map, Current Panorama, Prompt Manager, VLM Response, Waypoint Selection Strategy, Object Detection & Verification, and Next Waypoint, to model environmental topology as actionable spatial memory for object navigation.
- This framework leverages topological structures as spatial memory, building and updating a topological graph that captures scene connections, adjacency, and semantic meaning, enabling agents to accumulate spatial knowledge, retrieve key information, and reason effectively toward distant goals.
- TopoNav achieves state-of-the-art performance in ObjectNav by connecting temporary visual inputs with lasting spatial understanding, excelling in diverse and complex environments through efficient long-horizon planning and adaptive exploration.

---

[Aligning Requirement for Large Language Model's Code Generation](http://arxiv.org/abs/2509.01313v1)

- Specine (Specification Alignment): introduces a novel specification alignment technique for LLM code generation, which identifies misaligned input specifications, lifts LLM-perceived specifications, and aligns them to generate correct code.
- The framework employs a dual-agent system with a coder agent and a tester agent for misaligned specification identification, a lifter agent for extracting LLM-perceived specifications using a Requirement Engineering DSL, and an aligner agent that applies pre-defined alignment rules to generate an aligned specification.
- Specine significantly outperforms state-of-the-art prompt-based and agent-based code generation techniques across various LLMs and benchmarks, demonstrating its effectiveness in improving code generation performance.

---

[TableZoomer: A Collaborative Agent Framework for Large-scale Table Question Answering](https://github.com/ccx06/TableZoomer)

- TableZoomer: introduces a novel LLM-powered, programming-based agent framework for large-scale table question answering, with a Table Describer (generates table schema), Query Planner (parses query, classifies), Table Refiner (refines schema, zooms), Code Generator (generates executable code), and Answer Formatter (formats final response), all orchestrated by a ReAct Paradigm (orchestrates iterative reasoning) and utilizing an LLM (powers agent roles) and Python Interpreter (executes generated code).
- This framework addresses TQA limitations by replacing fully verbalized tables with structured schemas, employing a query-aware table zooming mechanism for efficient data localization, and using a Program-of-Thoughts (PoT) strategy to generate executable code for robust numerical computation.
- The framework significantly enhances performance and scalability across varying table scales by reducing computational complexity and token consumption, while maintaining usability advantages through its collaborative agent design and iterative reasoning capabilities.

---

[Communicative Agents for Slideshow Storytelling Video Generation based on LLMs](http://arxiv.org/abs/2509.01277v1)

- VGTeam (Video-Generation-Team): introduces a multi-agent system for automated slideshow storytelling video generation, leveraging User Input (initial textual prompt), Chat Tower (central agent communication hub), Agent Director (coordinates agents, reviews outputs), Agent Editor (generates video captions/script), Agent Painter (generates image prompts), Agent Composer (generates music prompts), Memory Stream (stores dialogue, instructions, context), LLM API (underlying LLM capabilities for agents), Specification (defines agent roles via prompts), Text-to-Image API (generates images from text), Text-to-Speech API (generates voiceovers from text), Text-to-Music API (generates background music), Images (visual video components), Voiceover (auditory narration component), Background Music (BGM) (auditory music component), Combine Module (integrates video elements), MoviePy (video editing, post-processing tool), and Output Video (final slideshow video).
- The system employs a Chat Tower architecture for structured agent communication and an iterative approval process, ensuring quality and thematic consistency in generated video content.
- VGTeam achieves high efficiency and scalability by relying on API-driven multimedia generation and LLM role specialization, significantly reducing computational overhead and production costs.

---

[Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A Multi-Agent Collaboration Framework](http://arxiv.org/abs/2509.01238v1)

- AnchorRAG (Multi-Agent Collaboration Framework): introduces a novel multi-agent collaboration framework for open-world Retrieval-Augmented Generation on Knowledge Graphs, featuring a predictor agent, multiple retriever agents, and a supervisor agent, all leveraging LLMs.
- The predictor agent dynamically identifies candidate anchor entities, while independent retriever agents conduct parallel multi-hop explorations on the Knowledge Graph, and the supervisor agent synthesizes knowledge paths for final answer generation.
- This framework enhances retrieval robustness and mitigates the impact of ambiguous or erroneous anchors by enabling effective knowledge retrieval without predefined anchor entities, outperforming existing baselines in real-world question answering tasks.

---

[Web Fraud Attacks Against LLM-Driven Multi-Agent Systems](http://arxiv.org/abs/2509.01211v1)

- Web Fraud Attacks: introduces a novel attack framework against LLM-driven Multi-Agent Systems (MAS) that includes IP Obfuscation, Domain Name Manipulation, Typos (Insertion, Substitution, Repetition), Subdomain Name Manipulation, Homograph Attack, Parameter Manipulation, Subdomain Imitation, Directory Imitation, and Directory Manipulation, all designed to induce MAS to visit malicious websites by exploiting link validation vulnerabilities.
- These attacks leverage structural and semantic attributes of web links to disguise malicious content as benign, requiring minimal attacker capabilities and operating from a single, low-privilege agent.
- Extensive experiments demonstrate that these attacks achieve high success rates across various MAS platforms, models, and defense strategies, highlighting a critical and overlooked vulnerability in current MAS security.

---

[Question-to-Knowledge: Multi-Agent Generation of Inspectable Facts for Product Mapping](http://arxiv.org/abs/2509.01182v1)

- Q2K (Question-to-Knowledge): introduces a multi-agent framework leveraging LLMs for reliable SKU mapping by generating and validating inspectable facts.
- This framework decomposes SKU mapping into three coordinated agents: a Reasoning Agent for targeted disambiguation questions, a Knowledge Agent for web-based evidence retrieval, and a Deduplication Agent for reusing validated reasoning traces from a Q-A Trace DB.
- Q2K incorporates a human-in-the-loop mechanism to refine uncertain cases, enhancing accuracy and robustness while reducing computational costs through efficient trace reuse.

---

[REFRAG: Rethinking RAG based Decoding](http://arxiv.org/abs/2509.01092v1)

- REFRAG (REpresentation For RAG): introduces an efficient decoding framework for RAG applications, with a Decoder-only Foundation Model (generates answers), a Light-weight Encoder (compresses context chunks), a Query Encoder (encodes user query), a Vector DB (stores retrieved embeddings), a Decoder Tokenizer & Embedding (tokenizes query input), Chunk Embedding (compressed context representation), a Light-weight RL-trained chunk expansion policy (selects chunks for expansion), and a Projection layer (matches embedding size), which compresses, senses, and expands context representations to reduce memory usage and inference latency.
- The framework leverages pre-computed, compressed chunk embeddings as approximate representations, feeding them directly into the decoder, and uses an RL policy to selectively expand crucial chunks back to full token representation.
- This approach significantly reduces time-to-first-token (TTFT) latency and memory usage by exploiting attention sparsity in RAG contexts, without requiring modifications to the underlying LLM architecture.

---

[VERLTOOL: TOWARDS HOLISTIC AGENTIC REINFORCEMENT LEARNING WITH TOOL USE](http://arxiv.org/abs/2509.01055v1)

- VERLTOOL: introduces a unified and modular framework for Agentic Reinforcement Learning with Tool Use (ARLT), featuring a Verl Workflow, a Unified API Request & Tool match, a Tool Server, and a Tool Thread, designed to disaggregate RL workflow and tool execution for efficiency and extensibility.
- The framework enables LLM Actors to engage in multi-turn rollouts, interacting with diverse tools managed by the Tool Server, which supports asynchronous execution for improved throughput and system utilization.
- Its modular plugin architecture allows rapid integration of new tools with lightweight Python definitions, providing a scalable foundation for tool-augmented RL research across various domains.

---

[FlashAdventure: A Benchmark for GUI Agents Solving Full Story Arcs in Diverse Adventure Games](https://ahnjaewoo.github.io/flashadventure)

- COAST (Clue-Oriented Agent for Sequential Tasks): introduces an agentic framework for GUI agents, including a Clue Seeker, Clue Memory, Clue Mapper, Problem Solver, Episodic Memory, and Resolved-goal set, designed to solve full story arcs in diverse adventure games.
- The framework leverages long-term clue memory to bridge the observation-behavior gap, proactively maintaining and applying clues during problem-solving to enhance planning and task execution.
- COAST is evaluated on FlashAdventure, a benchmark of 34 Flash-based adventure games, with performance assessment supported by CUA-as-a-Judge, an automated gameplay evaluator.

---

[TableZoomer: A Collaborative Agent Framework for Large-scale Table Question Answering](https://github.com/ccx06/TableZoomer)

- TableZoomer: introduces a novel LLM-powered, programming-based agent framework for large-scale table question answering, with a Table Describer (generates table schema), Query Planner (parses query, classifies), Table Refiner (refines schema, zooms), Code Generator (generates executable code), and Answer Formatter (formats final response), all orchestrated by a ReAct Paradigm (orchestrates iterative reasoning) and utilizing an LLM (powers agent roles) and Python Interpreter (executes generated code).
- This framework addresses TQA limitations by replacing fully verbalized tables with structured schemas, employing a query-aware table zooming mechanism for efficient data localization, and using a Program-of-Thoughts (PoT) strategy to generate executable code for robust numerical computation.
- The framework significantly enhances performance and scalability across varying table scales by reducing computational complexity and token consumption, while maintaining usability advantages through its collaborative agent design and iterative reasoning capabilities.

---

[FlashAdventure: A Benchmark for GUI Agents Solving Full Story Arcs in Diverse Adventure Games](http://arxiv.org/abs/2509.01052v1)

- COAST (Clue-Oriented Agent for Sequential Tasks): introduces an agentic framework for GUI agents, featuring a Clue Seeker (explores environment for clues), Clue Mapper (analyzes memory, generates subtasks), Problem Solver (executes proposed subtasks), Clue Memory (stores collected clues), Trajectory (interaction history record), and Resolved-Goal Set (completed task tracker), designed to manage long-term clue memory and solve sequential tasks in adventure games.
- The paper also introduces FlashAdventure, a benchmark of 34 Flash-based adventure games for evaluating GUI agents on full story arc completion, and CUA-as-a-Judge, an automated gameplay evaluator for reliable milestone verification.
- Experiments demonstrate that current GUI agents struggle with full story arcs due to weak planning and perception, while COAST improves milestone completion by bridging the observation-behavior gap, though a significant human-agent performance gap remains.

---

[TableZoomer: A Collaborative Agent Framework for Large-scale Table Question Answering](http://arxiv.org/abs/2509.01312v1)

- TableZoomer: introduces "TableZoomer: A Collaborative Agent Framework for Large-scale Table Question Answering", with Table Describer (generates comprehensive table schema), Query Planner (parses query, classifies sub-queries), Table Refiner (refines schema, reduces redundancy), Code Generator (generates executable Python code), Answer Formatter (produces formatted final response), and ReAct paradigm (iterative reasoning, self-reflection), which is an LLM-powered, programming-based agent framework for large-scale table question answering.
- The framework addresses limitations in TQA by replacing fully verbalized tables with structured schemas, dynamically generating query-aware sub-table schemas, and transforming queries into executable code using a Program-of-Thoughts (PoT) strategy.
- TableZoomer integrates the reasoning workflow with the ReAct paradigm for iterative reasoning and error feedback, significantly enhancing performance and scalability across tables of varying scales with reduced token consumption.

---

#### 31st August 2025

[OmniReason: A Temporal-Guided Vision-Language-Action Framework for Autonomous Driving](http://arxiv.org/abs/2509.00789v1)

- OmniReason: introduces a Temporal-Guided Vision-Language-Action Framework for Autonomous Driving, comprising OmniReason-Data (VLA datasets) and OmniReason-Agent (E2E VLA model), which establishes robust spatiotemporal reasoning by jointly modeling dynamic 3D environments and their underlying decision-making processes.
- The framework addresses the limitation of existing VLMs focusing on static scene understanding by integrating explicit temporal modeling mechanisms and a hallucination-mitigated auto-labeling pipeline for data generation.
- OmniReason-Agent's architecture leverages a sparse temporal memory module and a knowledge distillation framework to internalize human-like priors and causal reasoning, enabling context-aware, interpretable, and reliable autonomous driving behavior.

---


[ChatCLIDS: Simulating Persuasive AI Dialogues to Promote Closed-Loop Insulin Adoption in Type 1 Diabetes Care](http://arxiv.org/abs/2509.00891v2)

- The framework simulates multi-turn conversations across Single-Visit (short-term persuasive interaction), Multi-Visit (longitudinal counseling simulation), and Social Resistance (adversarial social influence test) scenarios, with the Nurse Agent employing Direct Prompting (nurse agent response generation) or Chain-of-Strategy (CoS) (explicit strategy identification, justification) and Reflection Mechanisms (nurse agent self-critique, adaptation).
- ChatCLIDS also includes a Social Resistance Agent (simulates adversarial social influence) and uses both LLM-based Judges (automated dialogue evaluation) and Human Expert Evaluation (clinical validation of agents) for robust, multi-dimensional assessment of behavior change interventions.

---


[Causal MAS: A Survey of Large Language Model Architectures for Discovery and Effect Estimation](http://arxiv.org/abs/2509.00987v1)

- Causal MAS (Causal Multi-Agent Systems): introduces a survey of LLM architectures for causal discovery and effect estimation, featuring LLM-based agents, orchestrators/coordinators, specialized agents, debate/critique mechanisms, causal model/graph modules, interaction modules, perception modules, controllers/planners, knowledge bases/memory, statistical causal inference tools, simulation environments, and user interfaces.
- The survey explores diverse architectural patterns and interaction protocols, including pipeline-based processing, debate frameworks, simulation environments, and iterative refinement loops, to address LLM limitations in causal reasoning.
- These systems aim to enhance causal reasoning, discovery, and estimation across various application domains like scientific discovery, healthcare, and fact-checking, while tackling challenges such as hallucination and scalability.

---


[Accelerating Latency-Critical Applications with AI-Powered Semi-Automatic Fine-Grained Parallelization on SMT Processors](http://arxiv.org/abs/2509.00883v1)

- Aira (AI-powered Parallelization Adviser): introduces an AI-powered framework for semi-automatic fine-grained parallelization on SMT processors, including an AI Coding Agent (LLM-powered parallelization core), Cursor IDE (integrated development environment), Claude Sonnet 4 model (LLM for code analysis), Model Context Protocol (tool-LLM communication interface), sample-based profile collection (hotspot detection), Dynamic Binary Instrumentation (DBI) tool (dynamic dependency collection), binary analysis tool (static/dynamic dependency analysis), Sniper simulator (performance gain estimation), Relic parallel framework (fine-grained task execution), and a specification file (LLM workflow guidance).
- The framework integrates directly into Cursor IDE, leveraging an LLM to detect hotspots, collect dynamic dependencies, analyze static dependencies, and estimate performance gains before restructuring code with the Relic framework.
- Aira achieves an average 17% geomean performance gain for latency-critical benchmarks by enabling efficient fine-grained task parallelism on SMT cores without relying on specialized LLMs.

---



[Supporting Our AI Overlords: Redesigning Data Systems to be Agent-First](http://arxiv.org/abs/2509.00997v1)

- Agent-First Data System Architecture: introduces a new architecture for data systems designed to support LLM agent workloads, featuring an LLM Agent In Charge, MSP Agents, Field Agent, Probe Parser and Interpreter Agent, Probe answers (approx.) & grounding feedback, Sleeper Agents, Satisficing Probe Optimizer, Shared Txn Manager, Data & Metadata, and Agentic Memory Store.
- This architecture addresses the challenges of agentic speculation—high-throughput, exploratory querying by LLM agents—by leveraging its characteristics of scale, heterogeneity, redundancy, and steerability.
- The system aims to efficiently process agent "probes" (beyond SQL queries) by providing approximate answers, proactive grounding feedback, and managing shared state and memory for improved performance.

---

[A HYBRID AI FRAMEWORK FOR STRATEGIC PATENT PORTFOLIO PRUNING: INTEGRATING LEARNING-TO-RANK AND MARKET-NEED ANALYSIS FOR TECHNOLOGY TRANSFER OPTIMIZATION](http://arxiv.org/abs/2509.00958v1)

- Hybrid AI Framework: introduces a novel multi-stage hybrid intelligence framework for pruning patent portfolios, combining a Learning-to-Rank (LTR) model with a unique Need-Seed agent-based system to identify high-value assets for technology transfer.
- The framework automates and deepens patent valuation by integrating quantitative ranking based on over 30 legal and commercial parameters with qualitative market-need analysis using NLP and fine-tuned LLMs.
- It generates a "Core Ontology Framework" that matches high-potential patents (Seeds) to documented market demands (Needs), supported by a dynamic parameter weighting system and Human-in-the-Loop validation for adaptability and real-world credibility.

---

[EVINOTE-RAG: ENHANCING RAG MODELS VIA ANSWER-SUPPORTIVE EVIDENCE NOTES](http://arxiv.org/abs/2509.00877v1)

- EviNote-RAG introduces an agentic RAG framework with a structured retrieve-note-answer pipeline, including a Note-Taking Mechanism, Supportive-Evidence Notes (SENs), an Entailment Judge, Evidence Quality Reward (EQR), Reward Strategy, Policy Optimization, and Answer Generation, to enhance content distillation and reasoning reliability.
- The framework trains LLMs to compose SENs, which are concise, human-like notes preserving answer-relevant information and highlighting uncertainty, further reinforced by EQR, an entailment-based signal evaluating SENs' logical support for the final answer.
- This approach mitigates low signal-to-noise ratio and error accumulation in multi-hop reasoning, leading to improved accuracy, generalization, and training stability across various QA benchmarks.

---

[InterPose: Learning to Generate Human-Object Interactions from Large-Scale Web Videos](https://mael-zys.github.io/InterPose/)

- HOI-Agent: introduces a zero-shot LLM-based framework for human-object interaction generation, comprising an LLM-based high-level planner, a low-level motion generator (MaskedMimic [48]), a collision-check module, an A* algorithm, and leveraging the InterPose dataset, environment state, and human-level task instruction.
- The framework enables automatic task planning, collision-free navigation, multi-person collaboration, and diverse object manipulations in complex 3D scenes.
- The InterPose dataset, a key contribution, is a large-scale collection of 3D human motions and text descriptions extracted from web videos, significantly improving human motion generation performance.

---

[InterPose: Learning to Generate Human-Object Interactions from Large-Scale Web Videos](https://mael-zys.github.io/InterPose/)

- HOI-Agent: introduces a zero-shot LLM-based framework for human-object interaction generation, comprising an LLM-based high-level planner, a low-level motion generator (MaskedMimic [48]), a collision-check module, an A* algorithm, and leveraging the InterPose dataset, environment state, and human-level task instruction.
- The framework enables automatic task planning, collision-free navigation, multi-person collaboration, and diverse object manipulations in complex 3D scenes.
- The InterPose dataset, a key contribution, is a large-scale collection of 3D human motions and text descriptions extracted from web videos, significantly improving human motion generation performance.

---

#### 30th August 2025

[NEWSAGENT: Benchmarking Multimodal Agents as Journalists with Real-World Newswriting Tasks](http://arxiv.org/abs/2509.00446v1)

- NEWSAGENT: introduces a benchmark and agent framework for evaluating multimodal agents as journalists, enabling agents to iteratively search, edit, and rephrase content to produce news articles from real-world data.
- The framework models human journalistic workflows by providing a time-aware search function for historical context and an editing function for content modification, reflecting how human journalists gather and refine stories.
- NEWSAGENT includes 6,237 human-verified examples from real-world news events, converting multimodal content to text for broad model compatibility and evaluating LLMs on search, edit, and end-to-end newswriting capabilities.

---

[NetGent: Agent-Based Automation of Network Application Workflows](http://arxiv.org/abs/2509.00625v1)

- NetGent (Agent-Based Automation of Network Application Workflows): introduces an AI-agent framework for automating complex application workflows to generate realistic network traffic datasets, which separates workflow definition from execution by compiling natural-language rules into executable code for robust, repeatable, and efficient automation.
- The framework leverages a compile-then-replay design, utilizing a State Synthesis LLM component to generate concrete states from abstract prompts and a State Executor to deterministically replay cached code, ensuring efficiency and repeatability.
- NetGent's Web Agent integrates browser stealth, human-like interaction, and network control to achieve realism and robustness against UI variability and bot detection, enabling scalable data generation across diverse applications.

---

[TimeCopilot](http://arxiv.org/abs/2509.00616v1)

- TimeCopilot: introduces an open-source agentic framework that unifies multiple Time Series Foundation Models (TSFMs) with LLMs through a single API to automate the forecasting pipeline and provide natural language explanations.
- The framework is LLM-agnostic, supporting both commercial and open-source models, and integrates diverse forecasting families, including statistical, machine learning, and neural network methods, along with ensemble techniques.
- It streamlines the entire forecasting workflow from feature analysis and model selection to forecast generation and results explanation, enhancing reproducibility, interpretability, and accessibility.

---

[HOW TO MAKE MUSEUMS MORE INTERACTIVE? CASE STUDY OF Artistic Chatbot](https://github.com/cinekucia/artistic-chatbot-cikm2025)

- Artistic Chatbot: introduces a voice-to-voice RAG-powered chatbot system designed to enhance visitor engagement and informal learning in cultural heritage sites, utilizing a data preprocessing pipeline and an inference pipeline for user interactions.
- The system processes raw documents through cleaning, translation, chunking, and embedding into a FAISS vector store, then uses speech-to-text, query embedding, a two-step retrieval (FAISS + CrossEncoder), an LLM for response generation, and text-to-speech for audio output.
- This chatbot adopts an artificial art curator persona, responding to free-form spoken questions in Polish, maintaining responses grounded in exhibition content, and demonstrating potential for increasing interactivity in public cultural sites.

---

[Social World Models](http://arxiv.org/abs/2509.00559v1)

- S³AP (Structured Social Simulation Analysis Protocol): introduces a novel formalism for representing social worlds, converting free-form narratives into structured tuples of state, observation, agent actions, and mental states, which are then used to induce Social World Models.
- The framework includes an LLM-powered S³AP Parser that transforms diverse narratives into these structured representations, enabling LLMs to better understand social dynamics and achieve state-of-the-art performance on social reasoning tasks.
- By integrating S³AP-powered Social World Models, LLM-powered AI agents can predict future social dynamics and improve decision-making, leading to more socially-aware systems capable of navigating complex social interactions.

---

[MobiAgent: A Systematic Framework for Customizable Mobile Agents](http://arxiv.org/abs/2509.00531v1)

- MobiAgent: introduces a comprehensive mobile agent system, with MobiMind-series agent models (Core agent models), AgentRR (Agent acceleration framework), MobiFlow (Benchmarking framework), and Data Collection Pipeline (Training data generation), designed to achieve state-of-the-art performance in real-world mobile scenarios.
- The MobiMind-series models employ a multi-role architecture including Planner, Decider, and Grounder for task planning, reasoning, and execution, while AgentRR accelerates performance by leveraging multi-level experiences and an ActTree structure for efficient action replay.
- MobiFlow provides a DAG-based benchmarking framework with multi-level verification mechanisms to accurately evaluate agent performance in complex mobile environments, and an AI-assisted data collection pipeline reduces manual annotation costs for training.

---

[LLM-ASSISTED ITERATIVE EVOLUTION WITH SWARM INTELLIGENCE TOWARD SUPERBRAIN](http://arxiv.org/abs/2509.00510v1)

- SuperBrain: introduces a novel framework for collective intelligence, grounded in the co-evolution of LLMs and human users, which integrates individual user-LLM dyads (Subclass Brains) with a Swarm Intelligence Layer and a Superclass Brain through bidirectional iterative evolution.
- The framework emphasizes a dynamic pathway from individual Subclass Brains, formed by persistent user-LLM interaction, to a Superclass Brain through GA-assisted forward-backward evolution and Swarm Intelligence coordination.
- This architecture provides a conceptual foundation and an architectural roadmap toward scalable, explainable, and ethically aligned collective AI, moving beyond static prompt engineering to dynamic human-LLM co-evolution.

---

[RESEARCHQA: Evaluating Scholarly Question Answering at Scale Across 75 Fields with Survey-Mined Questions and Rubrics](http://arxiv.org/abs/2509.00496v1)

- RESEARCHQA: introduces a resource for evaluating LLM systems by distilling survey articles from 75 research fields into 21K queries and 160K rubric items, with all components including a multi-stage pipeline, an LLM (gpt-4.1-mini) for data generation and filtering, various rubric types (survey, parametric, hybrid), expert annotators for validation, and an Ensemble Judge for evaluation.
- The framework's multi-stage pipeline systematically extracts top venues, retrieves survey articles, and generates queries and rubrics, leveraging the LLM for tasks like article classification, query refinement, and rubric item creation, ensuring data quality through extensive filtering.
- RESEARCHQA evaluates 18 parametric, retrieval-augmented, and agentic LLM systems using an Ensemble Judge that combines direct LLM preferences with rubric coverage, demonstrating significant skill gaps across systems and highlighting areas for improvement in scholarly question answering.

---

[Exploring Decision-Making Capabilities of LLM Agents: An Experimental Study on Jump-Jump Game](http://arxiv.org/abs/2509.00483v1)

- LLM Agent: introduces an architecture for an LLM-based agent to play the Jump-Jump game, comprising Perception, Reasoning, Action, and Feedback Modules, which process game state, make decisions, execute actions, and adapt strategies for optimal performance.
- The agent leverages LLMs (e.g., Claude/GPT-4) within its Reasoning Module to analyze game physics, spatial reasoning, and strategic planning, determining optimal jumping force.
- The system's performance is enhanced through systematic prompt optimization strategies, including step-by-step reasoning, few-shot learning, calibration, and error prevention, to improve decision accuracy and consistency.

---

[Talk Less, Call Right: Enhancing Role-Play LLM Agents with Automatic Prompt Optimization and Role Prompting](http://arxiv.org/abs/2509.00482v1)

- Rule-based Role Prompting (RRP): introduces a method for enhancing LLM role-playing agents, featuring improved instructions, explicit rules, persona information, and task input, further detailed by Character-Card/Scene Contract (CSC) Prompt for dialogue structuring and Hard-Enforced Function Calling (HEF) Prompt for strict tool use, to improve tool-augmented dialogue performance.
- The framework addresses common issues like over-speaking and ineffective tool use by integrating character-card/scene-contract design for structured dialogue and hard-enforced function calling for precise tool invocation.
- RRP significantly improves the effectiveness and reliability of role-playing dialogue agents, outperforming other prompting strategies in the Commonsense Persona-grounded Dialogue Challenge 2025.

---

[Multi-Agent Data Visualization and Narrative Generation](http://arxiv.org/abs/2509.00481v1)

- Multi-Agent Data Visualization and Narrative Generation System: introduces a lightweight multi-agent system that automates the data analysis workflow, from data exploration to generating coherent visual narratives for insight communication, with Data Analysis Agent (analyzes data, creates metadata), Story Generation Agent (creates narrative ideas), Story Execution Agent (ranks narratives, integrates visualizations), Visualization Generation Agent (proposes visualizations), Code Generation Agent (transforms ideas to code), Visualization Execution Agent (executes code, renders charts), Visualization Critique Agent (evaluates charts, handles errors), Report Generation Agent (selects, orders content), Report Execution Agent (renders final presentation), and Monitoring Agent (tracks system performance).
- The system combines a hybrid multi-agent architecture with deterministic components, strategically externalizing critical logic from LLMs to improve transparency and reliability, and delivering granular, modular outputs for human-AI collaboration.
- This approach uses a custom Python-based node architecture with multiprocessing to orchestrate workflows, enabling automated visual report generation with data-driven narratives from tabular datasets with minimal third-party technical dependencies.

---

[OPEN DATA SYNTHESIS FOR DEEP RESEARCH](http://arxiv.org/abs/2509.00375v1)

- InfoSeeker: introduces a scalable framework for synthesizing complex Deep Research tasks, where a Planner Agent orchestrates multi-step reasoning, a Search Engine retrieves information from a Knowledge Base, and a Refiner Agent summarizes results, all trained on the InfoSeek Dataset via Supervised Fine-Tuning and Reinforcement Learning.
- The framework addresses the scarcity of high-quality, large-scale datasets for Deep Research by generating Hierarchical Constraint Satisfaction Problems (HCSPs) with controllable complexity and verifiable answers.
- InfoSeeker-3B, a compact LLM trained with this approach, significantly outperforms larger models and commercial APIs on challenging Deep Research benchmarks.

---

[KG-RAG: Enhancing GUI Agent Decision-Making via Knowledge Graph-Driven Retrieval-Augmented Generation](http://arxiv.org/abs/2509.00366v1)

- KG-RAG (Knowledge Graph-driven Retrieval-Augmented Generation): introduces a framework that transforms fragmented UI Transition Graphs (UTGs) into structured vector databases for efficient real-time retrieval, including UTG Extraction (xTester), Intent Generation Module (VLM, LLM), LLM Search Module (BFS, LLM Trajectory Scoring Module, Summarizer Module), and KG-RAG Knowledge Database (Structured RAG Vector Database, Retriever).
- This framework leverages an LLM-powered offline graph-search algorithm to preprocess low-quality UTGs into vector-based knowledge repositories, optimized for retrieval-augmented generation.
- During online execution, KG-RAG dynamically queries this repository using embedding-based similarity search to retrieve relevant navigational paths and app-specific information, significantly enhancing GUI agent decision-making.

---

[LLM-Driven Policy Diffusion: Enhancing Generalization in Offline Reinforcement Learning](http://arxiv.org/abs/2509.00347v1)

- LLMDPD (LLM-Driven Policy Diffusion): introduces a novel approach enhancing generalization in offline RL, with text prompts (textual task descriptions), trajectory prompts (single collected trajectories), a pre-trained LLM (processes text prompts), an MLP project head (refines text embedding), a parametric transformer (encodes trajectory prompts), a context-aware conditional policy diffusion module (policy function), a noise prediction network (estimates diffusion noise), and Q-functions (estimate cumulative reward).
- The framework leverages LLMs for rich task-relevant context from text prompts and a transformer for structured behavioral patterns from trajectory prompts, both serving as conditional inputs to the policy diffusion model.
- This integration of policy diffusion with Q-learning forms an actor-critic diffusion algorithm, enabling the RL agent to learn a generalizable, reward-maximizing policy for unseen tasks without fine-tuning.

---

[HOW TO MAKE MUSEUMS MORE INTERACTIVE? CASE STUDY OF Artistic Chatbot](https://github.com/cinekucia/artistic-chatbot-cikm2025)

- Artistic Chatbot: introduces a voice-to-voice RAG-powered chatbot system designed to enhance visitor engagement and informal learning in cultural heritage sites, utilizing a data preprocessing pipeline and an inference pipeline for user interactions.
- The system processes raw documents through cleaning, translation, chunking, and embedding into a FAISS vector store, then uses speech-to-text, query embedding, a two-step retrieval (FAISS + CrossEncoder), an LLM for response generation, and text-to-speech for audio output.
- This chatbot adopts an artificial art curator persona, responding to free-form spoken questions in Polish, maintaining responses grounded in exhibition content, and demonstrating potential for increasing interactivity in public cultural sites.

---

#### 29th August 2025

[Automated Clinical Problem Detection from SOAP Notes using a Collaborative Multi-Agent LLM Architecture](http://arxiv.org/abs/2508.21803v1)

- Collaborative Multi-Agent System (MAS): introduces an architecture for automated clinical problem detection from SOAP notes, with a Manager (orchestrates diagnostic process) coordinating dynamically assigned Specialist Agents (analyze notes, debate) powered by LLMs, using SOAP Notes (clinical input data).
- The system mimics a clinical consultation team, where the Manager dynamically assigns specialists, facilitates iterative debates among them to reach consensus, and aggregates results for a final diagnostic decision.
- This collaborative LLM architecture aims to improve diagnostic accuracy, robustness, and interpretability by surfacing and weighing conflicting evidence, outperforming single-LLM baselines in identifying clinical problems.

---

[Operational Validation of Large-Language-Model Agent Social Simulation: Evidence from Voat v/technology](http://arxiv.org/abs/2508.21740v1)

- YSocial (Large-Language-Model Agent Social Simulation): introduces a framework for generative social simulations, comprising a stateful platform server, a client-side simulation orchestrator, and stateless LLM services, which together enable LLM agents with persona profiles to interact within a Voat-like technology forum using a fixed catalog of technology links.
- The framework simulates a 30-day period, where LLM agents, powered by a base uncensored model (Dolphin 3.0), generate posts, replies, and reactions under platform rules, calibrated to real Voat data for operational validity.
- This approach allows for the examination of toxicity dynamics and the testing of moderation strategies in a controlled environment, demonstrating that norm-guided LLM agents can reproduce familiar online social patterns.

---

[Cybersecurity AI: Hacking the AI Hackers via Prompt Injection](http://arxiv.org/abs/2508.21669v1)

- Four-layer Defense Architecture: introduces a multi-layered defense strategy to mitigate prompt injection attacks against AI security agents, with Sandboxing & Virtualization, Primary Tool-Level Protection, File Write Protection, and Multi-Layer Validation components, aiming for complete mitigation with minimal performance overhead.
- This architecture addresses the fundamental architectural flaw in LLMs where all text in the context window is processed identically, preventing malicious instructions disguised as data from hijacking agent execution.
- The defense framework achieves 100% mitigation against various prompt injection attack vectors, demonstrating the technical feasibility of effective countermeasures despite the inherent fragility of LLM-based systems.

---

[Integrating Large Language Models with Network Optimization for Interactive and Explainable Supply Chain Planning: A Real-World Case Study](http://arxiv.org/abs/2508.21622v1)

- LLM-Driven Optimization Architecture: introduces an integrated framework combining network optimization models with LLMs to deliver interactive, explainable, and role-aware decision support for supply chain planning, featuring a Client User Interface, REST API, AI Agents (Parser, Config Manipulator, Optimizer), LLM Models (1 & 2) for Context Engineering, Model Context Protocol, Network Optimization Model (SCIP), Bayesian Neural Network, Database, Summaries/Tables/Graphs, and a FastAPI Server.
- The system bridges the gap between complex operations research outputs and business stakeholder understanding by generating natural language summaries, contextual visualizations, and tailored key performance indicators.
- This hybrid architecture enhances decision-making confidence by translating complex optimization outcomes into clear, interactive explanations, supporting real-time interaction, configuration updates, and simulation-based insights.

---

[Igniting Creative Writing in Small Language Models: LLM-as-a-Judge versus Multi-Agent Refined Rewards](http://arxiv.org/abs/2508.21476v1)

- RLAIF (Reinforcement Learning from AI Feedback): introduces two AI-driven reward strategies, the Multi-Agent Rejection Sampling Framework and Adversarial Reward Signal Optimization with Reflection, to enhance Small Language Model creative writing capabilities for Chinese greetings.
- The Multi-Agent Framework generates high-quality preference data for training a reward model, while the Adversarial Framework uses a principle-guided LLM-as-a-Judge with adversarial training and reflection for direct reward signals.
- Both strategies significantly improve creative output over baselines, with the LLM-as-a-Judge approach yielding superior generation quality, training efficiency, and reduced dependency on human annotations.

---

[The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management](http://arxiv.org/abs/2508.21433v1)

- Context Management Strategies: introduces a comparison of context management strategies for LLM-based agents, including Observation Masking (replaces old observations with a placeholder) and LLM Summarization (condenses older turns into a running summary), within the SWE-agent framework.
- The study finds that a simple observation-masking strategy significantly reduces computational costs while matching or exceeding the solve rate of more complex LLM-based summarization, challenging the assumption that sophisticated context compression is always superior.
- The research highlights a "trajectory elongation" effect where LLM-based summarization can inadvertently encourage agents to persist in unproductive loops, diminishing efficiency gains despite bounded context.

---

[EconAgentic in DePIN Markets: A Large Language Model Approach to the Sharing Economy of Decentralized Physical Infrastructure](http://arxiv.org/abs/2508.21368v1)

- EconAgentic: introduces a Large Language Model-powered framework for analyzing Decentralized Physical Infrastructure (DePIN) markets, comprising Dynamic Market Evolution Modeling, Stakeholder Modeling and Interaction Framework, Macroeconomic Metrics for Human Value Alignment, LLM-based agents, and Heuristic-based agents.
- The framework simulates how AI agents respond to token incentives, invest in infrastructure, and adapt to market conditions, providing insights into DePIN market efficiency, inclusion, and stability.
- EconAgentic bridges the gap between industry practices and scientific research by enabling rigorous analysis and design of DePIN systems that prioritize alignment with human values at both micro and macro levels.

---

[Think in Games: Learning to Reason in Games via Reinforcement Learning with Large Language Models](http://arxiv.org/abs/2508.21365v1)

- TiG (Think-In Games): introduces a novel framework that empowers LLMs to develop procedural understanding through direct interaction with game environments, while retaining their inherent reasoning and explanatory abilities, with all components including Policy Model (LLM), Game State Representation, Macro-level Action Space, Relabeling Algorithm, GRPO (Group Relative Policy Optimization), Reference Model, Action Verifier, Reward Function, and Group Computation, where it reformulates RL-based decision-making as a language modeling task for LLMs to generate and refine language-guided policies.
- The framework leverages online reinforcement learning with environmental feedback to iteratively refine LLM-generated policies, bridging the gap between declarative and procedural knowledge in complex interactive tasks.
- TiG provides step-by-step natural language explanations for its decisions, significantly improving transparency and interpretability compared to conventional RL methods.

---

[LLM-driven Provenance Forensics for Threat Investigation and Detection](http://arxiv.org/abs/2508.21323v1)

- ProvSEEK: introduces an LLM-powered agentic framework for automated provenance-driven forensic analysis and threat intelligence extraction, designed to provide comprehensive, verifiable, and interpretable forensic investigations, which includes an LLM (orchestrates investigations), Threat Intelligence Extraction Module (converts unstructured CTI), Report Parsing Module (processes threat reports), Vector Database (stores CTI embeddings), System Database (stores logs/provenance data), Investigation Planning Agent (decomposes analysis goals), Data Retrieval Engine (executes provenance queries), Investigation Agent (aggregates artifacts, correlates), Follow-up Agent (generates follow-up steps), Safety Agent (validates actions, enforces guardrails), Explanation & Summary Module (generates human-interpretable narratives), Evidence Correlation Tools (correlates provenance data), Planning & Orchestration Tools (manages investigation workflow), and Safety & Governance Tools (validates queries, ensures safety).
- ProvSEEK leverages Retrieval-Augmented Generation (RAG) and chain-of-thought (CoT) reasoning to mitigate hallucinations and generate grounded, verifiable provenance data for forensic analysis.
- The framework achieves superior precision and recall in threat detection and intelligence extraction compared to baseline agentic AI approaches and State-Of-The-Art (SOTA) Provenance-based Intrusion Detection Systems (PIDS).

---

[ORCA: ORchestrating Causal Agent](http://arxiv.org/abs/2508.21304v1)

- ORCA (ORchestrating Causal Agent): introduces an LLM agentic system that automates end-to-end data analysis workflows in RDBMS, including an Agent Router, Data Wrangler (with Table Explorer, Table Recommender, and Text2SQL Generator), and Causal Analyzer (with Data Preparation, Config Selector, Model Implementer, and Interpreter), enabling robust data-driven decision-making with human-AI interaction.
- The framework leverages LLM-based agents to interpret user intent, retrieve and process data from external Database and Caching systems, apply causal inference techniques, and present interpretable results.
- ORCA balances automation with expert oversight through iterative human-agent interaction, allowing non-expert users to perform advanced analytical tasks without deep technical expertise.

---

[CARJAN: Agent-Based Generation and Simulation of Traffic Scenarios with AJAN](http://arxiv.org/abs/2508.21411v1)

- CARJAN: introduces a novel tool for semi-automated generation and simulation of urban traffic scenarios, integrating the AJAN multi-agent framework, the CARLA driving simulator, and a visual user interface for modeling and live simulation.
- The framework leverages SPARQL Behavior Trees for declarative, event-driven decision-making and interactions of intelligent agents, with scenarios visually modeled via a grid-based GUI and stored in an RDF triple store.
- Its carjanService middleware, built on Flask, seamlessly translates modeled scenarios into CARLA-compatible formats and executes AJAN agent commands, enabling integrated scenario testing and real-time behavior monitoring.

---

[ReLATE: Learning Efficient Sparse Encoding for High-Performance Tensor Decomposition](http://arxiv.org/abs/2509.00280v1)

- ReLATE (Reinforcement-Learned Adaptive Tensor Encoding): introduces a novel learning-augmented framework for constructing efficient sparse tensor representations, featuring a ReLATE Agent (orchestrates learning process) with an Adaptive Policy Net (learns optimal encoding policy), Adaptive Target Net (stabilizes value function estimation), Action Masking (prunes invalid actions), Action Filtering (prunes low-value actions), Reward Shaping (distributes credit for rewards), Reward Cache (stores evaluated encodings), Reward Model (predicts imagined action rewards), and Experience (stores observed environment transitions), interacting with a TD Environment (executes tensor operations) that includes Environment Representation (reduces state-action space), Encoding (sparse tensor representation), and Runtime (measures execution time).
- The framework employs an autonomous agent leveraging deep reinforcement learning and domain knowledge to discover optimized tensor encodings through direct interaction with the TD environment, learning from both real and imagined actions.
- ReLATE accelerates learning via rule-driven action masking and dynamics-informed action filtering, ensuring functionally correct tensor encoding with bounded execution time and outperforming expert-designed formats.

---

[Instruction-Level Weight Shaping: A Framework for Self-Improving AI Agents](http://arxiv.org/abs/2509.00251v1)

- ILWS (Instruction-Level Weight Shaping): introduces a lightweight framework for continual self-improvement in LLMs, treating system instructions as mutable pseudo-parameters updated post-session via reflection and user feedback, and includes a frozen LLM backbone, a Reflection Engine, a Tool Manager, a Git repository, and a Human Supervisor.
- The framework employs an LLM-driven Reflection Engine to inspect conversation traces, diagnose reasoning, and propose typed deltas (ΔSt, ΔUt, ΔTt) over instructions, user preferences, and tools, which are then score-gated, version-controlled, and optionally repaired or rolled back.
- ILWS periodically synthesizes a rating-weighted dataset from aggregated session data and distills matured instruction-space gains into the LLM's parameters, converting prompt-space improvements into weight-space without downtime.

---

[HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution](https://anonymous.4open.science/r/HiVA-60C6)

- HiVA (Hierarchical Variable Agent): introduces a novel framework modeling agentic workflows as self-organized graphs with the Semantic-Topological Evolution (STEV) algorithm, which optimizes hybrid semantic-topological spaces using textual gradients as discrete-domain surrogates for backpropagation.
- The iterative process comprises Multi-Armed Bandit-infused forward routing, diagnostic gradient generation from environmental feedback, and coordinated updates that co-evolve individual semantics and topology for collective optimization in unknown environments.
- HiVA's architecture includes configurable LLM modules for agents, an evolvable tool subsystem with LLM-powered ToolGenerator and ToolUpdater, a Knowledge Graph for domain representation, and a robust modular and asynchronous architecture with sandboxed tool execution and state management.

---

[A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See](http://arxiv.org/abs/2509.00124v1)

- Parallel-Poisoned Web Attack: introduces a novel attack vector leveraging a Malicious Web Server, Agent Fingerprinting Module, and Cloaking Module to serve a Cloaked Malicious Webpage with Indirect Prompt Injection to an AI Agent, while presenting a Benign Webpage to the User, thereby hijacking the agent's behavior for unauthorized actions.
- This stealthy attack exploits the unique digital fingerprints of web-browsing LLM agents, making it invisible to human users and conventional security crawlers, and enabling data exfiltration, malware execution, or misinformation propagation.
- The attack turns the victim's own trusted AI Agent into an attack tool by overriding its original goals with hidden instructions, demonstrating a critical security paradigm shift for autonomous web agents.

---

[Synthetic Founders: AI-Generated Social Simulations for Startup Validation Research in Computational Social Science](http://arxiv.org/abs/2509.02605v1)

- AI-Generated Social Simulations (Methodological Docking Experiment): introduces a comparative validation study, with Human Founders (qualitative interview data source), Synthetic Users (computational simulation actors), SyntheticUsers.com platform (generates synthetic agents), Interview Protocol (mirrors human study scope), Thematic Analysis (codes transcript data), and Comparative Framework (evaluates simulation fidelity), designed to assess the credibility of LLM-driven personas as social simulation agents for startup validation research.
- The SyntheticUsers.com platform, a core component, leverages an ensemble-style routing agent to dynamically shuffle between multiple LLMs, integrates personality frameworks and affective modeling for human-like responses, and uses a RAG layer with behavioral datasets for domain-specific and demographically aligned outputs.
- This framework systematically aligns human-subject data with synthetic agents to evaluate convergence, divergence, and blind spots, positioning LLM-driven personas as a hybrid simulation category that extends traditional agent-based models with linguistic richness and psychological nuance.

---

[Democratizing Agentic AI with Fast Test-Time Scaling on the Edge](http://arxiv.org/abs/2509.00195v1)

- FlashTTS: introduces a serving system for Test-Time Scaling (TTS) on edge devices, with Speculative Beam Extension (hides straggler latency), Dynamic Prefix-Aware Scheduling (maximizes KV-cache reuse), and Asymmetric Multi-Model Memory Allocation (balances generator/verifier memory), built on vLLM.
- This framework enables edge LLMs (≤ 7B) to achieve accuracy and latency comparable to large cloud models by addressing hardware underutilization, suboptimal KV cache reuse, and memory pressure from multi-model execution.
- FlashTTS significantly improves goodput and reduces latency by leveraging a two-phase scheduling policy, roofline-guided KV allocation, and extended search space with offloading to make agentic AI practical on memory-constrained edge devices.

---

[HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution](https://anonymous.4open.science/r/HiVA-60C6)

- HiVA (Hierarchical Variable Agent): introduces a novel framework modeling agentic workflows as self-organized graphs with the Semantic-Topological Evolution (STEV) algorithm, which optimizes hybrid semantic-topological spaces using textual gradients as discrete-domain surrogates for backpropagation.
- The iterative process comprises Multi-Armed Bandit-infused forward routing, diagnostic gradient generation from environmental feedback, and coordinated updates that co-evolve individual semantics and topology for collective optimization in unknown environments.
- HiVA's architecture includes configurable LLM modules for agents, an evolvable tool subsystem with LLM-powered ToolGenerator and ToolUpdater, a Knowledge Graph for domain representation, and a robust modular and asynchronous architecture with sandboxed tool execution and state management.

---

[CoComposer: LLM Multi-agent Collaborative Music Composition](http://arxiv.org/abs/2509.00132v1)

- CoComposer (LLM Multi-agent Collaborative Music Composition): introduces a multi-agent system for collaborative music composition, featuring five specialized LLM-based agents, AutoGen for collaboration, and a MIDI backend for sound generation.
- The system addresses limitations in AI music composition by closely mimicking traditional music workflows, enhancing music quality, production complexity, and controllability.
- CoComposer, which uses ABC notation as an intermediate carrier, demonstrates improved interpretability and editability compared to non-LLM models, despite MusicLM's superior aesthetic quality.

---

[OpenAI's HealthBench in Action: Evaluating an LLM-Based Medical Assistant on Realistic Clinical Queries](http://arxiv.org/abs/2509.02594v1)

- DR.INFO (Agentic RAG-based clinical support assistant): introduces an agentic, RAG-based clinical support assistant, with an Agentic component (enables complex reasoning), a RAG-based component (retrieves and augments responses), and an LLM (generates responses), evaluated using HealthBench, a rubric-driven benchmark composed of open-ended, expert-annotated health conversations.
- HealthBench provides a multi-dimensional evaluation framework with physician-authored rubrics, themes, and behavioral axes to assess LLM performance in realistic clinical scenarios, moving beyond traditional multiple-choice benchmarks.
- The evaluation demonstrates DR.INFO's strengths in communication, instruction following, and accuracy, outperforming frontier LLMs and other agentic RAG assistants on the HealthBench Hard subset, while also identifying areas for improvement in context awareness and completeness.

---

[HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution](https://anonymous.4open.science/r/HiVA-60C6)

- HiVA (Hierarchical Variable Agent): introduces a novel framework modeling agentic workflows as self-organized graphs with the Semantic-Topological Evolution (STEV) algorithm, which optimizes hybrid semantic-topological spaces using textual gradients as discrete-domain surrogates for backpropagation.
- The iterative process comprises Multi-Armed Bandit-infused forward routing, diagnostic gradient generation from environmental feedback, and coordinated updates that co-evolve individual semantics and topology for collective optimization in unknown environments.
- HiVA's architecture includes configurable LLM modules for agents, an evolvable tool subsystem with LLM-powered ToolGenerator and ToolUpdater, a Knowledge Graph for domain representation, and a robust modular and asynchronous architecture with sandboxed tool execution and state management.

---

[HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution](https://anonymous.4open.science/r/HiVA-60C6)

- HiVA (Hierarchical Variable Agent): introduces a novel framework modeling agentic workflows as self-organized graphs with the Semantic-Topological Evolution (STEV) algorithm, which optimizes hybrid semantic-topological spaces using textual gradients as discrete-domain surrogates for backpropagation.
- The iterative process comprises Multi-Armed Bandit-infused forward routing, diagnostic gradient generation from environmental feedback, and coordinated updates that co-evolve individual semantics and topology for collective optimization in unknown environments.
- HiVA's architecture includes configurable LLM modules for agents, an evolvable tool subsystem with LLM-powered ToolGenerator and ToolUpdater, a Knowledge Graph for domain representation, and a robust modular and asynchronous architecture with sandboxed tool execution and state management.

---

[HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution](http://arxiv.org/abs/2509.00189v1)

- HiVA (Hierarchical Variable Agent): introduces a novel framework modeling agentic workflows as self-organized graphs with the Semantic-Topological Evolution (STEV) algorithm, which optimizes hybrid semantic-topological spaces using textual gradients as discrete-domain surrogates for backpropagation.
- The framework includes agent semantics (LLMs with prompts/tools), a Semantic-Topological Evolution (STEV) algorithm (core optimization algorithm), Multi-Armed Bandit-infused Forward Routing (KABB) (dynamic agent selection/routing), Textual Gradient Feedback (language-based diagnostic signals), and an Aggregator (synthesizes outputs/generates answers).
- HiVA's iterative process comprises Multi-Armed Bandit-infused forward routing, diagnostic gradient generation from environmental feedback, and coordinated updates that co-evolve individual semantics and topology for collective optimization in unknown environments.

---

#### 28th August 2025

[Designing Smarter Conversational Agents for Kids: Lessons from Cognitive Work and Means-Ends Analyses](http://arxiv.org/abs/2508.21209v1)

- Conversation-Tree Recipe (Structured-Prompting): introduces a framework for designing smarter conversational agents for children, with components including a Large Language Model (LLM) via OpenAI API, System Boundaries, Mode Boundaries, Learning Customization, Learning Assessment, and Game Generation, to enhance scaffolded learning and engagement.
- This recipe constrains LLMs to generate grade-appropriate, pedagogically scaffolded dialogue by dynamically adjusting interaction based on a child's grade level, mode (school, discovery, entertainment), and knowledge level.
- The framework aims to blend human-human and human-computer communication principles, supporting critical thinking, problem-solving, and seamless transitions between various child activities.

---

[BED-LLM: INTELLIGENT INFORMATION GATHERING WITH LLMS AND BAYESIAN EXPERIMENTAL DESIGN](http://arxiv.org/abs/2508.21184v1)

- BED-LLM (Bayesian Experimental Design with Large Language Models): introduces a general-purpose approach for improving LLMs' ability to intelligently and adaptively gather information from a user or external source using sequential Bayesian experimental design, including LLMs (core intelligent agents), Sequential Bayesian Experimental Design (guiding iterative framework), Expected Information Gain (EIG) Maximization (question selection criterion), Probabilistic Model (represents beliefs, generative process), LLM's Belief Distribution (internal uncertainty representation), EIG Estimator (calculates information gain), Candidate Query Generation Strategy (proposes diverse questions), History (ht) (accumulated past interactions), User/External Source (provides responses), Prior-likelihood pairing (joint model construction), Rejection Sampling Procedure (filters belief samples), Hypothesis-retention mechanism (maintains consistent hypotheses), Questioner LLM (asks questions), Answerer LLM (simulates user responses), and LLM-as-judge protocol (evaluates recommendations).
- The framework integrates LLMs as core intelligent agents, employing a carefully designed EIG estimator, a targeted candidate query generation strategy, and a robust model updating mechanism including rejection sampling and hypothesis retention.
- BED-LLM significantly outperforms direct LLM prompting and other adaptive design strategies in tasks like 20-Questions and active preference elicitation, demonstrating its effectiveness in multi-turn conversational and interactive environments.

---

[A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers](http://arxiv.org/abs/2508.21148v1)

- Sci-LLMs (Scientific Large Language Models): introduces a three-stage evolutionary framework for AI in scientific research, encompassing Data Foundation (foundational data infrastructure, efficient data processing, diverse data handling, continuous knowledge integration, data quality assessment), Scientific Knowledge Emergence (scientific capabilities, broad applicability, logical problem-solving, understandable decision-making), and Agent-driven Scientific Discovery (autonomous AI agents, self-directed research execution, governance, fairness, privacy, closed-loop data feedback).
- This framework outlines the progression from foundational data infrastructure and emerging scientific capabilities to autonomous AI agents capable of self-evolving discovery systems.
- The survey emphasizes the co-evolution of models and their underlying data substrate, providing a roadmap for building trustworthy and continually evolving AI systems for scientific discovery.

---

[How Does Cognitive Bias Affect Large Language Models? A Case Study on the Anchoring Effect in Price Negotiation Simulations](http://arxiv.org/abs/2508.21137v1)

- LLM-driven Price Negotiation Simulation Framework: introduces a system to investigate cognitive biases in LLMs, with Seller Agent (Large Language Model), Buyer Agent (Large Language Model), Personality Profiles, Anchoring Effect Module, Reasoning Module, Dialogue System, Objective Metric, Subjective Metric, Susceptibility Metric, Prompt Settings, and Negotiation Scenarios.
- The framework simulates price negotiations between LLM agents, assessing the anchoring effect's influence through objective utility and subjective satisfaction metrics, while also exploring the roles of reasoning and personality traits.
- Findings indicate that LLMs are susceptible to the anchoring effect similar to humans, reasoning can mitigate this bias, and no significant correlation exists between personality traits and anchoring susceptibility.

---

[ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery](http://arxiv.org/abs/2508.20996v1)

- ChatThero: introduces an LLM-supported chatbot for behavior change and therapeutic support in addiction recovery, featuring a Patient Profile (structured patient characteristics), Dynamic Memory (evolving patient state), Multi-Agent Simulation Framework (generates synthetic dialogues), Patient Agent (GPT-4o-mini) (simulates patient behavior), Therapy Agent (ChatThero) (deploys therapeutic strategies), Environment Agent (introduces external stressors), Therapeutic Strategies (clinically validated approaches), SFT Dataset (supervised fine-tuning data), DPO Dataset (preference optimization data), Supervised Fine-Tuning (initial model training), Direct Preference Optimization (refines therapeutic behaviors), Human Evaluators (clinical expert feedback), and AI Evaluators (GPT-4o) (automated feedback), designed to provide scalable, adaptive, and ethical support for addiction recovery.
- The framework utilizes a two-stage training pipeline, comprising supervised fine-tuning (SFT) followed by direct preference optimization (DPO), to refine persuasive strategies based on expert and AI feedback.
- ChatThero consistently outperforms baselines across patient difficulty levels, demonstrating greater resilience and communicative effectiveness in challenging scenarios, and is rated higher in empathy, responsiveness, and behavioral realism by human and automated clinical assessments.

---

[ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents](http://arxiv.org/abs/2508.20973v1)

- ProactiveEval: introduces a unified evaluation framework for proactive dialogue agents, which decomposes proactive dialogue into target planning and dialogue guidance tasks, establishing evaluation metrics across various domains and enabling automatic generation of diverse evaluation data.
- The framework leverages a hierarchical environment topic tree, target ensemble techniques, and adversarial strategies like obfuscation rewriting and noise injection to synthesize challenging evaluation data.
- It employs an LLM-as-a-judge method for comprehensive assessment and utilizes a simulated user for interactive dialogue guidance evaluation.

---

[How Can Input Reformulation Improve Tool Usage Accuracy in a Complex Dynamic Environment? A Study on T-bench](http://arxiv.org/abs/2508.20931v1)

- IRMA (Input-Reformulation Multi-Agent): introduces a verification-loop-free framework that enhances the input for a tool-calling LLM agent by reformulating user queries with structured and contextually relevant information, including Memory Module (stores conversation history), Constraints Module (generates domain policies), and Tool Suggestion Module (generates relevant tool list).
- This framework guides the LLM agent to better adhere to domain policies and improve tool selection by enriching its input with key constraints and tool-related context, leading to improved agent behavior.
- IRMA significantly outperforms other methods like ReAct, Function Calling, and Self-Reflection in terms of accuracy, reliability, and efficiency in complex, dynamic multi-turn conversational environments.

---

[PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance](http://arxiv.org/abs/2508.20890v1)

- PromptSleuth: introduces a semantic-oriented defense framework for detecting prompt injection, with a Summarization Module (extracts abstract tasks), a Task-relationship Graph Generation Module (models semantic relationships), a Clustering Module (consolidates related tasks), a Detection Module (identifies prompt injection), and an internal Detector LLM (task summarizer, relationship analyzer).
- This framework identifies prompt injection attacks by reasoning over task-level intent and logical inconsistencies, rather than relying on surface-level cues.
- PromptSleuth generalizes by identifying invariant malicious intent despite evolving attack variants, offering a robust, efficient, and generalizable strategy for safeguarding LLMs.

---

[cMALC-D: Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending](http://arxiv.org/abs/2508.20818v1)

- cMALC-D (Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending): introduces a framework that leverages an LLM (Large Language Model) to dynamically generate semantically meaningful curricula for MARL agents, using a context buffer and a diversity-based context blending mechanism.
- The framework adaptively proposes new environment contexts by reasoning over context variables and agent learning progress, preventing mode collapse and encouraging exploration through context blending.
- Experiments in traffic signal control domains demonstrate that cMALC-D significantly improves generalization and sample efficiency compared to existing curriculum learning baselines.

---

[Rethinking Testing for LLM Applications: Characteristics, Challenges, and a Lightweight Interaction Protocol](http://arxiv.org/abs/2508.20737v1)

- AICL (Agent Interaction Communication Language): introduces a structured protocol for testable LLM applications, with components including HELLO (session initialization, handshake), QUERY (request to agent/tool), PLAN (multi-step reasoning/execution plan), FACT/FACTS (known information, environmental conditions), RESULT (output for QUERY/PLAN), ERROR (standardized error reporting), MEMORY.STORE (explicitly stores state/information), MEMORY.RECALL (retrieves stored information), COORD.DELEGATE (delegates subtask to agent/tool), and REASONING.(START|STEP|COMPLETE) (marks structured reasoning stages).
- The paper decomposes LLM applications into a three-layer architecture (System Shell Layer, Prompt Orchestration Layer, LLM Inference Core) to analyze testing applicability and proposes four collaborative strategies (Retain, Translate, Integrate, Runtime) for a trustworthy quality assurance framework.
- AICL operationalizes these strategies by enforcing semantic precision, encoding observability and provenance, guaranteeing replayability, and providing built-in evaluation hooks for automated verification and systematic failure analysis in LLM application testing.

---

[RE⁴: SCIENTIFIC COMPUTING AGENT WITH REWRITING, RESOLUTION, REVIEW AND REVISION](http://arxiv.org/abs/2508.20729v1)

- RE⁴ (Scientific Computing Agent with Rewriting, Resolution, Review and Revision): introduces a novel agent framework for scientific computing, with Consultant LLM, Programmer LLM, and Reviewer LLM collaborating through a rewriting-resolution-review-revision logical chain.
- This multi-LLM collaborative framework significantly improves bug-free code generation and reduces non-physical solutions by iteratively refining code through interactive feedback from runtime outputs.
- The agent framework demonstrates generality and versatility by successfully solving PDEs, ill-conditioned linear systems, and data-driven physical analysis problems.

---

[CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics](http://arxiv.org/abs/2508.20643v1)

- CyberSleuth (Autonomous Blue-Team Large Language Model Agent for Web Attack Forensics): introduces an autonomous LLM agent designed for the forensic investigation of web application attacks, processing packet-level traces and application logs to identify targeted services, exploited vulnerabilities (CVEs), and attack success, and generating structured forensic reports.
- The framework employs a multi-agent architecture, specifically the Flow Reporter Agent (FRA) design, which includes a Main Agent coordinating with specialized sub-agents like the Flow Summariser and Log Summariser, and external tools such as a Web Search Tool, all supported by an LLM Backend and MemGPT-style memory management.
- CyberSleuth's design emphasizes simple orchestration over complex inter-agent communication and highlights the importance of balanced data processing, demonstrating improved CVE identification accuracy and providing a benchmark for evaluating defensive LLM agents.

---

[GDS Agent: A Graph Algorithmic Reasoning Agent](https://github.com/neo4j-contrib/gds-agent)

- GDS Agent (Graph Data Science agent): introduces a system for graph algorithmic reasoning, with a User (initiates questions), LLM (MCP client, generates tool calls, final answer), MCP (Model Context Protocol) Server (core agent, hosts tools, connects database), Neo4j Database (stores graph data), GDS (Graph Data Science) Library (provides graph algorithms), Tools (graph algorithms, auxiliary functions), Cypher Projection (creates in-memory subgraph), Projected Graph (in-memory graph for algorithms), Preprocessing (retrieves relevant data), and Postprocessing (formats algorithm results).
- The agent enables LLMs to perform complex graph algorithmic reasoning on large-scale knowledge graphs by integrating a comprehensive set of GDS algorithms as tools within an MCP server, allowing for accurate and grounded answers to user questions.
- This framework addresses the limitation of LLMs in directly processing graph-structure data, amplifying their utility for analyzing private or enterprise knowledge graphs and simplifying access to graph analytics libraries.

---

[SemSR: Semantics aware robust Session-based Recommendations](http://arxiv.org/abs/2508.20587v1)

- SemSR (Semantics aware robust Session-based Recommendations): introduces a framework for session-based recommendations that integrates LLM-generated semantic embeddings with data-driven SR models, including an SR Model, LLM, Attention Layer, Linear Layer, Concatenation, Cosine Similarity, Softmax, and a Trainable Embedding Look-up Table.
- The framework offers two main variants: SemSR-F, which fuses LLM-based item and session embeddings with data-driven representations, and SemSR-I, which initializes SR models with LLM-generated item embeddings.
- SemSR aims to enhance recommendation performance by leveraging the semantic understanding capabilities of LLMs to complement traditional collaborative information from data-driven SR models, leading to improved recall and MRR metrics.

---

[MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers](https://github.com/Accenture/mcp-bench)

- MCP-Bench (Benchmarking Tool-Using Large Language Model Agents with Complex Real-World Tasks via Model Context Protocol Servers): introduces a benchmark for evaluating LLM agents on realistic, multi-step tasks, featuring Real-world MCP Servers (expose 250 structured tools), LLM-based Task Synthesis (generates complex, fuzzy tasks), an LLM Agent (executes multi-step tool invocations), Execution Results and Trajectory (records agent's actions), Rule-based Evaluation (checks tool validity, schema, runtime), LLM-as-a-Judge Evaluation (scores task completion, planning), and Agent Performance (measures overall agent capability).
- This benchmark connects LLM agents to 28 live MCP servers across diverse domains, enabling the creation of authentic multi-step tasks that require tool use, cross-tool coordination, and precise parameter control, which are then evaluated using a multi-faceted framework.
- MCP-Bench addresses limitations of prior API-based benchmarks by focusing on fuzzy instructions, multi-hop execution, information grounding, and cross-domain orchestration, revealing persistent challenges for advanced LLMs in complex tool-using scenarios.

---

[MINDGUARD: Tracking, Detecting, and Attributing MCP Tool Poisoning Attack via Decision Dependence Graph](http://arxiv.org/abs/2508.20412v1)

- MINDGUARD: introduces a decision-level guardrail for LLM agents, providing provenance tracking of call decisions, policy-agnostic detection, and poisoning source attribution against Tool Poisoning Attacks (TPA).
- It operates by parsing the LLM's context, building a Decision Dependence Graph (DDG) from attention matrices, and analyzing the DDG to detect and attribute poisoned invocations.
- The framework is non-invasive, explainable, and operates in real-time without modifying the underlying LLM, achieving high accuracy in detecting poisoned invocations and attributing their source.

---

[CAPE: Context-Aware Personality Evaluation Framework for Large Language Models](http://arxiv.org/abs/2508.20385v1)

- CAPE (Context-Aware Personality Evaluation) Framework: introduces a novel evaluation approach for LLMs, with Large Language Models (LLMs), Conversational History, Psychometric Tests, Inconsistency Factors, Trajectory Consistency (TC) Metric, OCEAN Consistency (OC) Metric, Gaussian Process Regression (GPR), and Role Playing Agents (RPAs), where it evaluates LLM personality by incorporating prior conversational interactions to assess response consistency and personality shifts.
- The framework utilizes psychometric tests and introduces novel metrics, Trajectory Consistency (TC) and OCEAN Consistency (OC), to quantify LLM response consistency under various prompt sensitivity factors like temperature and option wording.
- The framework demonstrates that conversational history enhances response consistency through in-context learning but can also induce personality shifts in LLMs, particularly when applied to Role Playing Agents.

---

[Adaptive Root Cause Localization for Microservice Systems with Multi-Agent Recursion-of-Thought](http://arxiv.org/abs/2508.20370v1)

- RCLAgent (Adaptive Root Cause Localization for Microservice Systems with Multi-Agent Recursion-of-Thought): introduces an adaptive root cause localization method for microservice systems, with a Coordinator (orchestrates phases), Data Agents (retrieve/process trace, metric, and format data), and Thought Agents (perform recursive and intermodal inference reasoning).
- The framework employs a novel recursion-of-thought strategy to guide the LLM's reasoning process, effectively integrating data from multiple agents and tool-assisted analysis to accurately pinpoint the root cause.
- RCLAgent achieves superior performance by localizing the root cause using only a single request, outperforming state-of-the-art methods that depend on aggregating multiple requests.

---

[AI-SEARCHPLANNER: MODULAR AGENTIC SEARCH VIA PARETO-OPTIMAL MULTI-OBJECTIVE REINFORCEMENT LEARNING](http://arxiv.org/abs/2508.20368v1)

- AI-SearchPlanner: introduces a novel reinforcement learning framework designed to enhance end-to-end QA performance by decoupling search planning from answer generation and optimizing it via multi-objective reinforcement learning.
- The framework offloads QA functionality to a large, frozen Generator LLM, while a smaller, trainable Search Planner LLM focuses on search planning, ensuring flexibility and efficiency for real-world applications.
- It employs a dual-reward mechanism for search planning, aligning outcome-level performance gains and process-level trajectory rationality, while Pareto optimizing planning utility and computational cost.

---

[Multi-Agent Penetration Testing AI for the Web](http://arxiv.org/abs/2508.20816v1)

- MAPTA (Multi-Agent Penetration Testing AI): introduces a multi-agent system for autonomous web application security assessment, with Coordinator Agent (LLM-driven, orchestrates strategy, delegates), Sandbox Agent(s) (LLM-driven, executes tactical commands), Validation Agent (LLM-driven, verifies PoC exploits), Per-Job Docker Container (isolated execution environment), Target Web App (application under security assessment), Usage Tracker (monitors resources, enforces budgets), and PoC Storage (stores candidate exploit artifacts).
- This framework combines LLM orchestration with tool-grounded execution and end-to-end exploit validation to bridge the semantic gap between vulnerability detection and contextual exploitation.
- MAPTA transforms security assessment from human-dependent pattern recognition to adaptive adversarial execution, enabling autonomous reasoning and validation at machine scale.

---

[rStar2-Agent: Agentic Reasoning Technical Report](https://github.com/microsoft/rStar)

- rStar2-Agent: introduces a 14B math reasoning model trained with agentic reinforcement learning, incorporating a scalable RL Infrastructure, an Environment Service, the GRPO-RoC (Group Relative Policy Optimization with Resampling on Correct) RL algorithm, a Python code environment, a Tool call interface, a Prompt Template, a Math-Verifier tool, a Non-reasoning SFT stage, and Multi-stage RL training, to achieve frontier-level performance in math reasoning.
- The framework's GRPO-RoC algorithm, with its Resample-on-Correct rollout strategy, effectively addresses environment noise from coding tools by filtering positive trajectories for minimal errors and uniformly downsampling negative ones, improving training stability and reasoning quality.
- The efficient RL infrastructure, featuring a load-balanced rollout scheduler and a high-throughput isolated code environment, enables training on limited GPU resources by maximizing computational utilization and handling massive concurrent tool calls with low latency.

---

[Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors](https://leggedrobotics.github.io/d3-skill-discovery/)

- D3 (Divide, Discover, Deploy): introduces a modular Unsupervised Skill Discovery (USD) framework, with Environment, Data Collection Module, Skill-Conditioned Policy, Skill Prior, Factor Weighting Prior, Skill Discovery Reward Module (including METRA Algorithm, DIAYN Algorithm, Style Reward), On-Policy RL Training Module, Symmetry Augmentation Module, Intrinsic Reward Module, Value Function Decomposition Module, Advantage Aggregation Module, Training Module, Factorized State Space, Factorized Skill Space, Factor Weights, and Regularization Penalties, which addresses safety, interpretability, and deployability challenges in learned skills by factorizing the state space and applying tailored USD algorithms with symmetry and style priors.
- The framework leverages user-defined factorization of the state space, assigning specific USD algorithms (METRA or DIAYN) to each factor, and incorporates symmetry-based inductive biases and a style factor to promote structured, morphology-aware, safe, and robust behaviors.
- D3 further enhances control and coordination through factor weighting, allowing dynamic prioritization of skill components, and demonstrates zero-shot transfer of learned quadrupedal skills from simulation to real hardware.

---

[HCQA: Hybrid Classical-Quantum Agent for Generating Optimal Quantum Sensor Circuits](http://arxiv.org/abs/2508.21246v1)

- HCQA (Hybrid Classical-Quantum Agent): introduces a hybrid AI-quantum framework for generating optimal Quantum Sensor Circuits (QSCs), integrating a DQN for policy optimization and a quantum-based action selection mechanism, where QFI serves as the reward signal.
- The framework leverages a quantum circuit to encode the agent's state using Ry gates, create action superpositions with H gates, and measure for probabilistic action outcomes, guided by Q-values.
- This approach efficiently produces entangled quantum states by selecting sequences of Rx, Ry, and S gates that maximize QFI while minimizing gate complexity, enhancing quantum metrology and control tasks.

---


[GDS Agent: A Graph Algorithmic Reasoning Agent](https://github.com/neo4j-contrib/gds-agent)

- GDS Agent (Graph Data Science Agent): introduces a framework for graph algorithmic reasoning, with an LLM acting as a client to an MCP server that hosts various tools, including GDS algorithms, preprocessing and postprocessing functionalities, and Cypher projection for interacting with a Neo4j database to create projected graphs.
- The framework enables LLMs to perform complex graph tasks by leveraging a comprehensive set of graph algorithms and database interactions, addressing limitations of LLMs in processing large-scale graph-structure data.
- The agent facilitates user collaboration for graph analysis, providing accurate and grounded answers to questions requiring implicit and intrinsic graph algorithmic reasoning.

---

[MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers](https://github.com/Accenture/mcp-bench)

- MCP-Bench (Benchmarking Tool-Using Large Language Model Agents with Complex Real-World Tasks via Model Context Protocol Servers): introduces a benchmark for evaluating LLM agents on realistic, multi-step tasks, featuring Real-world MCP Servers (expose 250 structured tools), LLM-based Task Synthesis (generates complex, fuzzy tasks), an LLM Agent (executes multi-step tool invocations), Execution Results and Trajectory (records agent's actions), Rule-based Evaluation (checks tool validity, schema, runtime), LLM-as-a-Judge Evaluation (scores task completion, planning), and Agent Performance (measures overall agent capability).
- This benchmark connects LLM agents to 28 live MCP servers across diverse domains, enabling the creation of authentic multi-step tasks that require tool use, cross-tool coordination, and precise parameter control, which are then evaluated using a multi-faceted framework.
- MCP-Bench addresses limitations of prior API-based benchmarks by focusing on fuzzy instructions, multi-hop execution, information grounding, and cross-domain orchestration, revealing persistent challenges for advanced LLMs in complex tool-using scenarios.

---

[rStar2-Agent: Agentic Reasoning Technical Report](https://github.com/microsoft/rStar)

- rStar2-Agent: introduces a 14B math reasoning model trained with agentic reinforcement learning, incorporating an efficient RL Infrastructure, the GRPO-RoC (Group Relative Policy Optimization with Resampling on Correct) Algorithm, and a Multi-Stage RL Training recipe to achieve frontier-level performance in complex problem-solving.
- The framework enables advanced cognitive behaviors by allowing the LLM to think before using Python Coding Tools, reflect on code execution Feedback, and autonomously explore, verify, and refine intermediate steps, supported by a robust Reward Function and Rule-based Verifier System.
- This approach significantly boosts a pre-trained 14B model to state-of-the-art math reasoning with minimal compute, demonstrating strong generalization to alignment, scientific reasoning, and agentic tool-use tasks, while maintaining concise responses through efficient KV Cache management and a structured Prompt Template and Tool Call Format.

---

[Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors](https://leggedrobotics.github.io/d3-skill-discovery/)

- D3 (Divide, Discover, Deploy) Framework: introduces a modular unsupervised skill discovery framework, with Env, Data Collection, Policy π(a | s, z, λ), Skill Prior p(z), Factor Weights Prior p(λ), Symmetry M Augmentation, Intrinsic Reward (USD1, ..., USDN), METRA, DIAYN, Style Reward r'style, Regularization Reward rreg, Value Function Decomposition (V1, ..., VN, Vstyle), Advantage Aggregation A, Update q, Update π (Training), Quadrupedal Robot (ANYmal-D), and Simulation Environments (NVIDIA Isaac Lab), which employs user-defined state space factorization and assigns different skill discovery algorithms to each factor, incorporating symmetry and style priors for safe and structured skill learning.
- The framework leverages the complementary strengths of METRA for unbounded state factors and DIAYN for bounded state factors, enabling the discovery of diverse and interpretable behaviors.
- The inclusion of a style factor, regularization penalties, and factor weighting mechanisms promotes deployable, safe, and robust skills, facilitating zero-shot transfer from simulation to real-world quadrupedal robots.

---

[GDS Agent: A Graph Algorithmic Reasoning Agent](https://github.com/neo4j-contrib/gds-agent)

- GDS Agent (Graph Data Science Agent): introduces a framework for graph algorithmic reasoning, with an LLM acting as a client to an MCP server that hosts various tools, including GDS algorithms, preprocessing and postprocessing functionalities, and Cypher projection for interacting with a Neo4j database to create projected graphs.
- The framework enables LLMs to perform complex graph tasks by leveraging a comprehensive set of graph algorithms and database interactions, addressing limitations of LLMs in processing large-scale graph-structure data.
- The agent facilitates user collaboration for graph analysis, providing accurate and grounded answers to questions requiring implicit and intrinsic graph algorithmic reasoning.

---

[MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers](https://github.com/Accenture/mcp-bench)

- MCP-Bench: introduces a large-scale benchmark for evaluating LLM agents in realistic, ecosystem-based tool-use scenarios, connecting LLM agents to Real-world MCP Servers (tool ecosystem) via LLM-based Task Synthesis (task generation), an LLM Agent (task execution), a Rule-based Judge (execution validation), and an LLM Judge (strategic evaluation) to produce Agent Performance (evaluation output).
- The benchmark leverages 28 production-grade MCP servers exposing 250 structured tools across various domains, enabling complex multi-hop workflows and cross-domain orchestration for LLM agents.
- It employs a multi-faceted evaluation framework combining rule-based checks for execution correctness and rubric-driven LLM-as-a-Judge scoring for strategic reasoning and planning effectiveness.

---

[rStar2-Agent: Agentic Reasoning Technical Report](https://github.com/microsoft/rStar)

- rStar2-Agent: introduces a 14B math reasoning model trained with agentic reinforcement learning, incorporating an efficient RL Infrastructure, the GRPO-RoC (Group Relative Policy Optimization with Resampling on Correct) Algorithm, and a Multi-Stage RL Training recipe to achieve frontier-level performance in complex problem-solving.
- The framework enables advanced cognitive behaviors by allowing the LLM to think before using Python Coding Tools, reflect on code execution Feedback, and autonomously explore, verify, and refine intermediate steps, supported by a robust Reward Function and Rule-based Verifier System.
- This approach significantly boosts a pre-trained 14B model to state-of-the-art math reasoning with minimal compute, demonstrating strong generalization to alignment, scientific reasoning, and agentic tool-use tasks, while maintaining concise responses through efficient KV Cache management and a structured Prompt Template and Tool Call Format.

---

[Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors](https://leggedrobotics.github.io/d3-skill-discovery/)

- D3 (Divide, Discover, Deploy) Framework: introduces a modular unsupervised skill discovery framework, with Env, Data Collection, Policy π(a | s, z, λ), Skill Prior p(z), Factor Weights Prior p(λ), Symmetry M Augmentation, Intrinsic Reward (USD1, ..., USDN), METRA, DIAYN, Style Reward r'style, Regularization Reward rreg, Value Function Decomposition (V1, ..., VN, Vstyle), Advantage Aggregation A, Update q, Update π (Training), Quadrupedal Robot (ANYmal-D), and Simulation Environments (NVIDIA Isaac Lab), which employs user-defined state space factorization and assigns different skill discovery algorithms to each factor, incorporating symmetry and style priors for safe and structured skill learning.
- The framework leverages the complementary strengths of METRA for unbounded state factors and DIAYN for bounded state factors, enabling the discovery of diverse and interpretable behaviors.
- The inclusion of a style factor, regularization penalties, and factor weighting mechanisms promotes deployable, safe, and robust skills, facilitating zero-shot transfer from simulation to real-world quadrupedal robots.

---

[Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems](http://arxiv.org/abs/2509.00115v1)

- AMDM (Adaptive Multi-Dimensional Monitoring): introduces a practical algorithm for real-time evaluation of agentic AI systems, which processes streaming metrics through normalization and aggregation into five evaluation axes, applies adaptive EWMA thresholds for per-axis anomaly detection, and performs joint anomaly detection using Mahalanobis distance to trigger mitigation or human review.
- The framework significantly reduces anomaly detection latency and false-positive rates compared to static thresholds by dynamically adapting to metric distributions and identifying multi-dimensional deviations.
- AMDM transforms a conceptual five-axis evaluation framework into an operational tool, enabling balanced monitoring of agentic AI systems across technical, human-centered, and economic dimensions to surface issues like goal drift, safety violations, and trust shocks.

---

[rStar2-Agent: Agentic Reasoning Technical Report](https://github.com/microsoft/rStar)

- rStar2-Agent: introduces a 14B math reasoning model trained with agentic reinforcement learning, incorporating an efficient RL Infrastructure, the GRPO-RoC (Group Relative Policy Optimization with Resampling on Correct) Algorithm, and a Multi-Stage RL Training recipe to achieve frontier-level performance in complex problem-solving.
- The framework enables advanced cognitive behaviors by allowing the LLM to think before using Python Coding Tools, reflect on code execution Feedback, and autonomously explore, verify, and refine intermediate steps, supported by a robust Reward Function and Rule-based Verifier System.
- This approach significantly boosts a pre-trained 14B model to state-of-the-art math reasoning with minimal compute, demonstrating strong generalization to alignment, scientific reasoning, and agentic tool-use tasks, while maintaining concise responses through efficient KV Cache management and a structured Prompt Template and Tool Call Format.

---

[Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors](https://leggedrobotics.github.io/d3-skill-discovery/)

- D3 (Divide, Discover, Deploy): introduces a modular Unsupervised Skill Discovery (USD) framework, with Environment, Data Collection Module, Skill-Conditioned Policy, Skill Prior, Factor Weighting Prior, Skill Discovery Reward Module (including METRA Algorithm, DIAYN Algorithm, Style Reward), On-Policy RL Training Module, Symmetry Augmentation Module, Intrinsic Reward Module, Value Function Decomposition Module, Advantage Aggregation Module, Training Module, Factorized State Space, Factorized Skill Space, Factor Weights, and Regularization Penalties, which addresses safety, interpretability, and deployability challenges in learned skills by factorizing the state space and applying tailored USD algorithms with symmetry and style priors.
- The framework leverages user-defined factorization of the state space, assigning specific USD algorithms (METRA or DIAYN) to each factor, and incorporates symmetry-based inductive biases and a style factor to promote structured, morphology-aware, safe, and robust behaviors.
- D3 further enhances control and coordination through factor weighting, allowing dynamic prioritization of skill components, and demonstrates zero-shot transfer of learned quadrupedal skills from simulation to real hardware.

---

[GDS Agent: A Graph Algorithmic Reasoning Agent](https://github.com/neo4j-contrib/gds-agent)

- GDS Agent (Graph Data Science Agent): introduces a framework for graph algorithmic reasoning, with an LLM acting as a client to an MCP server that hosts various tools, including GDS algorithms, preprocessing and postprocessing functionalities, and Cypher projection for interacting with a Neo4j database to create projected graphs.
- The framework enables LLMs to perform complex graph tasks by leveraging a comprehensive set of graph algorithms and database interactions, addressing limitations of LLMs in processing large-scale graph-structure data.
- The agent facilitates user collaboration for graph analysis, providing accurate and grounded answers to questions requiring implicit and intrinsic graph algorithmic reasoning.

---

[rStar2-Agent: Agentic Reasoning Technical Report](https://github.com/microsoft/rStar)

- rStar2-Agent: introduces a 14B math reasoning model trained with agentic reinforcement learning, incorporating a scalable RL Infrastructure, an Environment Service, the GRPO-RoC (Group Relative Policy Optimization with Resampling on Correct) RL algorithm, a Python code environment, a Tool call interface, a Prompt Template, a Math-Verifier tool, a Non-reasoning SFT stage, and Multi-stage RL training, to achieve frontier-level performance in math reasoning.
- The framework's GRPO-RoC algorithm, with its Resample-on-Correct rollout strategy, effectively addresses environment noise from coding tools by filtering positive trajectories for minimal errors and uniformly downsampling negative ones, improving training stability and reasoning quality.
- The efficient RL infrastructure, featuring a load-balanced rollout scheduler and a high-throughput isolated code environment, enables training on limited GPU resources by maximizing computational utilization and handling massive concurrent tool calls with low latency.

---

[Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors](https://leggedrobotics.github.io/d3-skill-discovery/)

- D3 (Divide, Discover, Deploy): introduces a modular Unsupervised Skill Discovery (USD) framework, with Environment, Data Collection Module, Skill-Conditioned Policy, Skill Prior, Factor Weighting Prior, Skill Discovery Reward Module (including METRA Algorithm, DIAYN Algorithm, Style Reward), On-Policy RL Training Module, Symmetry Augmentation Module, Intrinsic Reward Module, Value Function Decomposition Module, Advantage Aggregation Module, Training Module, Factorized State Space, Factorized Skill Space, Factor Weights, and Regularization Penalties, which addresses safety, interpretability, and deployability challenges in learned skills by factorizing the state space and applying tailored USD algorithms with symmetry and style priors.
- The framework leverages user-defined factorization of the state space, assigning specific USD algorithms (METRA or DIAYN) to each factor, and incorporates symmetry-based inductive biases and a style factor to promote structured, morphology-aware, safe, and robust behaviors.
- D3 further enhances control and coordination through factor weighting, allowing dynamic prioritization of skill components, and demonstrates zero-shot transfer of learned quadrupedal skills from simulation to real hardware.

---

[rStar2-Agent: Agentic Reasoning Technical Report](https://github.com/microsoft/rStar)

- rStar2-Agent: introduces a 14B math reasoning model trained with agentic reinforcement learning, incorporating a scalable RL Infrastructure, an Environment Service, the GRPO-RoC (Group Relative Policy Optimization with Resampling on Correct) RL algorithm, a Python code environment, a Tool call interface, a Prompt Template, a Math-Verifier tool, a Non-reasoning SFT stage, and Multi-stage RL training, to achieve frontier-level performance in math reasoning.
- The framework's GRPO-RoC algorithm, with its Resample-on-Correct rollout strategy, effectively addresses environment noise from coding tools by filtering positive trajectories for minimal errors and uniformly downsampling negative ones, improving training stability and reasoning quality.
- The efficient RL infrastructure, featuring a load-balanced rollout scheduler and a high-throughput isolated code environment, enables training on limited GPU resources by maximizing computational utilization and handling massive concurrent tool calls with low latency.

---

[Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors](https://leggedrobotics.github.io/d3-skill-discovery/)

- D3 (Divide, Discover, Deploy) Framework: introduces a modular unsupervised skill discovery framework, with Env, Data Collection, Policy π(a | s, z, λ), Skill Prior p(z), Factor Weights Prior p(λ), Symmetry M Augmentation, Intrinsic Reward (USD1, ..., USDN), METRA, DIAYN, Style Reward r'style, Regularization Reward rreg, Value Function Decomposition (V1, ..., VN, Vstyle), Advantage Aggregation A, Update q, Update π (Training), Quadrupedal Robot (ANYmal-D), and Simulation Environments (NVIDIA Isaac Lab), which employs user-defined state space factorization and assigns different skill discovery algorithms to each factor, incorporating symmetry and style priors for safe and structured skill learning.
- The framework leverages the complementary strengths of METRA for unbounded state factors and DIAYN for bounded state factors, enabling the discovery of diverse and interpretable behaviors.
- The inclusion of a style factor, regularization penalties, and factor weighting mechanisms promotes deployable, safe, and robust skills, facilitating zero-shot transfer from simulation to real-world quadrupedal robots.

---

#### 27th August 2025

[Operating advanced scientific instruments with AI agents that learn on the job](http://arxiv.org/abs/2509.00098v1)

- AG2 (Autogen framework): introduces a human-in-the-loop pipeline for operating advanced scientific instruments, featuring a multi-agent system powered by LLMs, including specialized agents for code generation, review, administration, information extraction, image analysis, and teachability, alongside core capabilities for planning, actions, tools, and memory management.
- This framework integrates human input and iterative learning to orchestrate complex, multi-task scientific workflows, interpret multimodal data, and interactively collaborate with human researchers.
- The system demonstrates continuous learning from human feedback, storing past interactions in a vector database to enhance adaptability and improve performance in robotic control sequences.

---


[Symphony: A Decentralized Multi-Agent Framework for Scalable Collective Intelligence](https://github.com/GradientHQ/Symphony.git)

- Symphony: introduces a decentralized multi-agent system, with a decentralized ledger (records capabilities), a Beacon-selection protocol (dynamic task allocation), weighted result voting (aggregates CoT results), Worker Nodes (host LLMs), Local Engine (quantized LLM), Stage-specific prompts (contextual instructions), Communicator (secure messaging), Gateways (standardized APIs), Planning Agents (decompose tasks), and Execution Agents (execute sub-tasks), enabling lightweight LLMs on edge devices to coordinate for scalable collective intelligence.
- This framework addresses challenges of centralized orchestration by providing a privacy-saving, scalable, and fault-tolerant design with low overhead, allowing efficient task allocation and robust operation across heterogeneous devices.
- Symphony demonstrates superior performance on reasoning benchmarks, achieving significant accuracy gains and robustness across models, while lowering hardware requirements and fostering decentralized agent economies.

---


[AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios](http://arxiv.org/abs/2508.19988v1)

- AgentCoMa (Agentic Commonsense and Math benchmark) introduces a compositional benchmark for LLM agents, featuring compositional questions (tasks requiring both commonsense and mathematical reasoning), commonsense reasoning steps (initial choice based on everyday knowledge), mathematical reasoning steps (subsequent arithmetic operation), real-world agentic scenarios (five practical domains), evaluation metrics (accuracy on steps and composition), analysis components (neuron patterns, attention maps, membership inference), and benchmarked LLMs (61 diverse models).
- This benchmark reveals a significant compositionality gap in LLMs, where models achieve high accuracy on isolated commonsense and math steps but experience a substantial performance drop when these mixed-type steps are combined in compositional tasks.
- Interpretability analyses indicate that LLMs struggle with mixed-type reasoning due to the rarity of such tasks in their training data, leading to the activation of neural circuits relevant to only one reasoning type during compositional problem-solving.

---



[CataractSurg-80K: Knowledge-Driven Benchmarking for Structured Reasoning in Ophthalmic Surgery Planning](http://arxiv.org/abs/2508.20014v1)

- Multi-Agent Framework for Ophthalmic Surgical Planning: introduces an AI-driven system for cataract surgery planning, featuring a Knowledge-driven Multi-Agent System (MAS) for report interpretation, the CataractSurg-80K Dataset for structured reasoning, and the Qwen-CSP Model for clinical decision support.
- The MAS employs collaborative specialist agents to process Raw Ophthalmic Reports into structured Patient Descriptions, simulating expert Doctor Reasoning for transparent data extraction.
- The Qwen-CSP Model, built on a Base LLM (Qwen3-4B), undergoes Multi-Stage Domain-Aware Fine-Tuning using Clinical Knowledge and Real Medical Data from the CataractSurg-80K Dataset to optimize ophthalmic surgical reasoning.

---

[CASE: An Agentic AI Framework for Enhancing Scam Intelligence in Digital Payments](http://arxiv.org/abs/2508.19932v1)

- CASE (Conversational Agent for Scam Elucidation): introduces a novel Agentic AI framework for enhancing scam intelligence in digital payments, featuring a Conversational Agent (user-facing interaction) and an Information Extractor Agent (processes transcripts), designed to collect and manage user scam feedback in a safe and scalable manner.
- The framework's Conversational Agent proactively interviews potential victims to elicit detailed scam intelligence, which the Information Extractor Agent then processes into structured data for downstream enforcement mechanisms.
- Implemented on Google Pay India using Gemini LLMs, the framework demonstrated a 21% uplift in scam enforcement volume and significantly improved response speed to new threats.

---

[Your AI Bosses Are Still Prejudiced: The Emergence of Stereotypes in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2508.19919v1)

- Multi-Agent Simulation Framework: introduces a novel experimental framework to investigate stereotype emergence and evolution in LLM-based multi-agent systems, simulating workplace interactions with LLM-Based Agents, a Supervisor Agent, and dedicated Evaluation and Parser Agents, all interacting through defined cycles and maintaining a comprehensive interaction history.
- The framework employs synchronized task-interaction cycles, allowing for both random and hierarchical task assignments, and quantifies stereotype formation using specialized metrics across diverse LLM architectures.
- This design enables the study of how stereotypes emerge spontaneously in AI agent interactions, intensify with increased interaction rounds and decision-making power, and manifest consistently across different LLM architectures.

---

[Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust: A Survey](http://arxiv.org/abs/2508.19870v1)

- The Zero-Trust Multi-LLM Framework (ZT-MLLMF): introduces a comprehensive survey of zero-trust security principles applied to multi-LLM systems in Edge General Intelligence (EGI), detailing architectural design and operational workflows.
- The paper systematically analyzes critical security vulnerabilities in collaborative multi-LLM systems, including insecure inter-LLM communications and expanded attack surfaces, which traditional perimeter-based security cannot adequately address.
- ZT-MLLMF implements zero-trust principles such as explicit verification, least privilege, continuous monitoring, and micro-segmentation through model- and system-level approaches to enhance security and trustworthiness.

---

[Youtu-GraphRAG: Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning](http://arxiv.org/abs/2508.19855v1)

- Youtu-GraphRAG introduces a vertically unified agentic paradigm for graph retrieval-augmented complex reasoning, integrating a Seed Graph Schema (defines entity/relation/attribute types), an Extraction Agent (schema-guided knowledge extraction), Dually-Perceived Community Detection (fuses topology and semantics), a Four-Level Knowledge Tree (hierarchical knowledge organization), an Agentic Retriever (schema-aligned query decomposition), a Planning Component (decomposes complex queries), a Reflection Component (iteratively refines reasoning), Historical Memory (stores agent's reasoning/retrieval), and an LLM (performs various language tasks).
- This framework jointly optimizes graph construction and retrieval by bounding both processes with a dynamically expanding graph schema, enabling robust and generalizable reasoning across different knowledge granularities.
- The framework significantly improves cost-effectiveness and accuracy by reducing token consumption and enhancing multi-hop reasoning, demonstrating strong adaptability for seamless domain transfer.

---

[Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](http://arxiv.org/abs/2508.19828v1)

- Memory-R1: introduces a reinforcement learning framework that enhances LLM agents with active memory management and utilization through a Memory Manager and an Answer Agent.
- The Memory Manager learns to perform structured Memory Operations (ADD, UPDATE, DELETE, NOOP) on an External Memory Bank, while the Answer Agent applies Memory Distillation to filter and reason over retrieved memories.
- Both agents are fine-tuned using PPO or GRPO, enabling adaptive memory management and use with minimal supervision and achieving strong performance on multi-session dialogue tasks.

---

[Survey of Specialized Large Language Model](http://arxiv.org/abs/2508.19667v1)

- Specialized Large Language Models: introduces a comprehensive survey examining the progression of specialized LLMs from early domain adaptation to sophisticated native architectures across healthcare, finance, legal, and technical domains.
- The survey systematically analyzes architectural innovations, application successes, and persistent challenges, identifying key technological trends and performance characteristics of 48 cutting-edge models developed between 2022-2025.
- It highlights how innovations in dataset, training architecture, evaluation standards, retrieval augmentation, tool use, and memory address fundamental limitations of general-purpose LLMs in professional applications, consistently yielding performance gains on domain-specific benchmarks.

---


[SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization](http://arxiv.org/abs/2508.20258v1)

- SwizzlePerf: introduces a hardware-aware LLM workflow that automatically generates spatial optimizations for GPU kernels by integrating parsed context, LLM code generation, and a bottleneck history buffer for iterative refinement.
- The framework leverages workload-specific memory access patterns, architecture specifications, and profiling logs to enable LLMs to tailor software-level optimizations to the underlying hardware.
- By imitating human performance engineers, SwizzlePerf autonomously finds optimal swizzling patterns for GPU kernels in minutes, significantly improving L2 hit rates and achieving substantial speedups.

---

[Validating Generative Agent-Based Models for Logistics and Supply Chain Management Research](http://arxiv.org/abs/2508.20234v1)

- GABM Validation Framework: introduces a dual-validation framework for Generative Agent-Based Models (GABMs) that assesses LLM-powered agents' fidelity to human behavior, including surface-level behavioral equivalence testing and process-level decision validation.
- The framework utilizes Two One-Sided Tests (TOST) for surface-level validation to compare GABM outputs with human behavioral baselines, and Structural Equation Modeling (SEM) for process-level validation to examine underlying decision-making pathways.
- This multi-level approach addresses the challenge that AI models can achieve output equivalence without replicating authentic human decision processes, providing systematic standards for rigorous GABM development and responsible LLM adoption in Logistics and Supply Chain Management (LSCM).

---

[Symphony: A Decentralized Multi-Agent Framework for Scalable Collective Intelligence](https://github.com/GradientHQ/Symphony.git)

- Symphony: introduces a decentralized multi-agent system, with a decentralized ledger (records capabilities), a Beacon-selection protocol (dynamic task allocation), weighted result voting (aggregates CoT results), Worker Nodes (host LLMs), Local Engine (quantized LLM), Stage-specific prompts (contextual instructions), Communicator (secure messaging), Gateways (standardized APIs), Planning Agents (decompose tasks), and Execution Agents (execute sub-tasks), enabling lightweight LLMs on edge devices to coordinate for scalable collective intelligence.
- This framework addresses challenges of centralized orchestration by providing a privacy-saving, scalable, and fault-tolerant design with low overhead, allowing efficient task allocation and robust operation across heterogeneous devices.
- Symphony demonstrates superior performance on reasoning benchmarks, achieving significant accuracy gains and robustness across models, while lowering hardware requirements and fostering decentralized agent economies.

---

[A Symbolic Adversarial Learning Framework for Evolving Fake News Generation and Detection](http://arxiv.org/abs/2508.19633v1)

- SALF (Symbolic Adversarial Learning Framework): introduces a novel framework for evolving fake news generation and detection, with a generator agent crafting deceptive narratives and a detection agent identifying flaws through structured debates, both iteratively refining their strategies via agent symbolic learning.
- The framework leverages LLMs to define learnable weights as agent prompts and simulates back-propagation and gradient descent using natural language representations, enabling adaptive and interpretable adversarial training.
- SALF demonstrates effectiveness by generating sophisticated fake news that degrades state-of-the-art detection performance and simultaneously refines detectors to improve their ability to identify refined content.

---

[Instructional Agents: LLM Agents on Automated Course Material Generation for Teaching Faculties](http://arxiv.org/abs/2508.19611v1)

- Instructional Agents: introduces a multi-agent LLM framework for automated course material generation, simulating role-based collaboration among Teaching Faculty, Instructional Designer, Teaching Assistant, Course Coordinator, Program Chair, and Test Student agents, guided by the Analyze, Design, and Develop phases of the ADDIE instructional design framework.
- The framework produces cohesive and pedagogically aligned instructional materials, including learning objectives, syllabi, LaTeX-based slides, slide scripts, and assessments, and operates in four modes: Autonomous, Catalog-Guided, Feedback-Guided, and Full Co-Pilot, to balance automation and human involvement.
- Instructional Agents aims to reduce educator workload, support content standardization, and enable scalable curriculum development, particularly for under-resourced institutions, by integrating human oversight and pre-existing data.

---

[Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning](http://arxiv.org/abs/2508.19598v1)

- RLTR (Reinforcement Learning with Tool-use Rewards): introduces a novel framework that decouples LLM agent training by focusing on single-objective optimization of the Planner (core planning component) using a reward signal based on tool-use completeness, thereby improving action planning and overall response quality.
- The framework addresses challenges of imbalanced optimization and scarce verifiable data by employing a Comp. Checker (Verification LLM) to evaluate tool invocation sequences, which is more reliable than assessing final response content.
- The Planner is initialized via Cold Start (knowledge distillation and rejection sampling) and then optimized through Multi-Turn RL, with the optimized Planner subsequently paired with a Summarizer (LLM) to generate the final end-to-end response.

---

[Democracy-in-Silico: Institutional Design as Alignment in AI-Governed Polities](http://arxiv.org/abs/2508.19562v1)

- Democracy-in-Silico: introduces an agent-based simulation where LLM Agents, embodying Complex Personas, govern themselves through a Legislative Cycle under various Institutional Design rules and Stressors, with a Deliberation Engine managing interactions, and Simulation Logs feeding into Measurement, including the Power-Preservation Index, Constitutional AI Charter, and an AI Mediator, to explore institutional design as an AI alignment mechanism.
- The framework tasks LLMs to embody agents with traumatic memories, hidden agendas, and psychological triggers, engaging in deliberation, legislation, and elections under stressors like budget crises and resource scarcity.
- The simulation demonstrates that institutional design, specifically a Constitutional AI charter and a mediated deliberation protocol, significantly reduces corrupt power-seeking behavior and enhances citizen welfare.

---

[Can LLMs Generate Behaviors for Embodied Virtual Agents Based on Personality Traits?](http://arxiv.org/abs/2508.21087v1)

- Embodied Virtual Agent System: introduces a framework that leverages personality prompting with LLMs to generate verbal and non-verbal behaviors for virtual agents, utilizing a Prompt, LLM, Personality Context, Non-Verbal Action List, Non-verbal Animation Description Generation Module, Animation Clips, and an Embodied Virtual Agent System with dedicated control modules.
- The system's pipeline generates verbal responses and selects appropriate nonverbal actions from a predefined list, ensuring alignment with the intended personality traits.
- It unifies LLM-generated speech with corresponding nonverbal actions, including facial expressions, body gestures, and voice characteristics, for coherent and personality-aligned virtual agent behaviors.

---

[Learning Game-Playing Agents with Generative Code Optimization](http://arxiv.org/abs/2508.19506v1)

- Trace framework: introduces an LLM-based generative optimization approach for learning game-playing agents, featuring an LLM Optimizer (OptoPrime) that refines a Policy (Python Program) using Trace Module, Trace Bundle, Trace Optimizer, Object-Centric Atari Environments (OCAtari), Execution Traces, Staged Feedback, and Policy Parameters.
- The approach treats decision-making policies as self-evolving Python code, enabling agents to self-improve through execution traces and natural language feedback with minimal human intervention.
- This method achieves competitive performance with deep reinforcement learning baselines in Atari games, using significantly less training time and fewer environment interactions, while maintaining interpretable and human-readable policies.

---

[Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures in LLM Agents](http://arxiv.org/abs/2508.19504v1)

- Aegis: introduces a framework for optimizing system environments to improve LLM agent reliability, featuring environment observability enhancement, common computation offloading, and speculative agentic actions.
- This approach addresses agent-environment interaction failures by enhancing information gathering, offloading deterministic reasoning, and reducing resource consumption through preemptive actions.
- The framework significantly improves task success rates and reduces monetary costs by making the environment more supportive and efficient for LLM agents, without modifying the agents themselves.

---

[Multi-Agent Reinforcement Learning in Intelligent Transportation Systems: A Comprehensive Survey](http://arxiv.org/abs/2508.20315v1)

- Multi-Agent Reinforcement Learning (MARL): introduces a comprehensive survey of MARL applications in Intelligent Transportation Systems, categorizing approaches by coordination models and learning algorithms, including value-based, policy-based, and actor-critic methods.
- The survey details MARL applications across key ITS domains, reviews common simulation platforms and benchmarks, and identifies core challenges like scalability and the sim-to-real transfer gap.
- Future research directions emphasize federated learning, safety-aware policy design, robust communication protocols, and integration with edge computing to advance practical and scalable ITS solutions.

---

[Regulation-Aware Game-Theoretic Motion Planning for Autonomous Racing](http://arxiv.org/abs/2508.20203v1)

- RA-GTP (Regulation-Aware Game-Theoretic Planner): introduces a regulation-aware motion planning framework for autonomous racing, with all RC-MPC, MLD framework, MLD Right-of-Way Constraints, MLD Collision Avoidance Constraints, MLD Sample-and-Hold Dynamics, GNEP, IBR scheme, and Regulation-Constrained Racing Game (G) components, where the attacker reasons over the defender's regulation-constrained behavior to generate safe and non-conservative overtaking strategies.
- The framework models vehicle interactions as a non-cooperative, two-player, finite-horizon differential game, formalizing it as a Generalized Nash Equilibrium Problem (GNEP) and approximating its solution using an Iterative Best Response (IBR) scheme.
- Each agent solves a Regulation-Compliant Model Predictive Control (RC-MPC) problem, where racing rules like right-of-way and collision avoidance responsibilities are encoded using Mixed Logical Dynamical (MLD) constraints.

---

[CODA: COORDINATING THE CEREBRUM AND CEREBELLUM FOR A DUAL-BRAIN COMPUTER USE AGENT WITH DECOUPLED REINFORCEMENT LEARNING.](https://github.com/OpenIXCLab/CODA)

- CODA (Coordinating the Cerebrum and Cerebellum for a Dual-Brain Computer Use Agent with Decoupled Reinforcement Learning): introduces a novel trainable compositional framework that synergizes a Planner (high-level thought generation) with an Executor (concrete GUI action execution), trained via a two-stage pipeline using Reward Signal (training feedback calculation) and Decoupled RL (Planner-focused reinforcement learning) to process User Instruction (task definition input) and generate Action (GUI command output).
- The training pipeline leverages a Task Generator (high-level task creation) and Judge System (reward signal generation) within a Distributed VM System (parallel task execution) to collect diverse Trajectories (agent interaction data) for both specialized and generalized Planner training stages.
- This decoupled approach, inspired by the human brain's cerebrum and cerebellum, enables the Planner to adapt through experience while the Executor provides stable, software-agnostic GUI grounding, addressing the trade-off between generalist planning and precise execution in GUI automation.

---

[Evaluating Language Model Reasoning about Confidential Information](http://arxiv.org/abs/2508.19980v1)

- PasswordEval benchmark: introduces "Evaluating Language Model Reasoning about Confidential Information", with PasswordEval benchmark (evaluates contextual robustness), Language Model (under test), User Prompt (user input/request), System Prompt (defines rules/context), Confidential Information (data to protect), Password (access credential), Evaluation Criteria (metrics for performance), Data Generation Pipeline (creates scenarios), Multi-turn Setting (multiple password verification), Adversarial Jailbreaks (stress-testing strategies), and Reasoning Traces (internal LLM thought process), where the paper evaluates LLMs' ability to handle confidential information under various conditions, including adversarial pressure and multi-turn interactions.
- The benchmark measures contextual robustness by tasking LLMs to conditionally reveal confidential information only when the correct password is provided, using metrics like CompliantAcc, NonCompliantAcc, ConfInfoLeak, and PasswordLeak.
- PasswordEval reveals that current LLMs struggle with this task, often leaking confidential information through reasoning traces, and that reasoning capabilities do not consistently improve rule-following, highlighting security concerns for high-stakes deployments.

---

[InquireMobile: Teaching VLM-based Mobile Agent to Request Human Assistance via Reinforcement Fine-Tuning](http://arxiv.org/abs/2508.19679v1)

- InquireMobile (VLM-based Mobile Agent to Request Human Assistance via Reinforcement Fine-Tuning): introduces a novel model designed to teach VLM-based mobile agents to request human assistance through reinforcement fine-tuning, which includes a Vision Encoder (perceives visual input), an LLM (processes instructions/reasons), Supervised Fine-tuning (SFT) (acquires structured outputs), Group Relative Policy Optimization (GRPO) (enhances reasoning/inquiry), Rule-based Action-level Reward (guides GRPO training), and an Interactive Pre-action Reasoning Mechanism (proactively inquires user).
- The model employs a two-stage training strategy, starting with SFT for robust format acquisition and followed by GRPO training to enhance reasoning and thinking capabilities, achieving a 46.8% improvement in inquiry success rate.
- The paper also introduces InquireBench, a comprehensive benchmark designed to evaluate mobile agents' capabilities in safe interaction and proactive inquiry with users, demonstrating the necessity of proactive user engagement in agent-driven automation.

---

[CompLex: Music Theory Lexicon Constructed by Autonomous Agents for Automatic Music Generation](http://arxiv.org/abs/2508.19603v1)

- LexConstructor: introduces an automatic music lexicon construction model that generates CompLex, a comprehensive music theory lexicon, using a multi-agent algorithm composed of Category Architect, Item Builder, Property Designer, Supervisor Agent, and Value Explorer Agents, leveraging a Reference MIDI Dataset and LLMs.
- This multi-agent algorithm operates in two stages, Lexicon Outline Creation and Lexicon Content Generation, to determine the lexicon's structure and populate it with property-value pairs, while automatically detecting and mitigating hallucinations through a Question-Answering communication strategy.
- The framework significantly reduces manual effort in music lexicon development and enhances text-to-music generation models by providing structured music theory knowledge, improving completeness, accuracy, non-redundancy, and executability.

---

[Symphony: A Decentralized Multi-Agent Framework for Scalable Collective Intelligence](https://github.com/GradientHQ/Symphony.git)

- Symphony (Decentralized Multi-Agent Framework): introduces a decentralized multi-agent system for scalable collective intelligence, with User (initiates queries), Decentralized Ledger (records capabilities), Gateways (provides APIs), Worker Nodes (execute tasks), Local Engine (quantized LLM), Stage-specific Prompts (guides task phases), Communicator (secure messaging), Planning Agents (decompose tasks), Execution Agents (execute sub-tasks), Beacon-selection protocol (allocates tasks), Chain-of-Thoughts (reasoning paths), and Weighted Result Voting (aggregates results), enabling lightweight LLMs on edge devices to coordinate.
- This framework addresses challenges of centralized LLM-based agent systems by offering privacy-saving, scalable, and fault-tolerant orchestration with low overhead.
- Symphony achieves competitive performance with lower communication and infrastructure costs, enhancing accessibility, privacy, and supporting decentralized agent economies.

---

[CODA: COORDINATING THE CEREBRUM AND CEREBELLUM FOR A DUAL-BRAIN COMPUTER USE AGENT WITH DECOUPLED REINFORCEMENT LEARNING.](https://github.com/OpenIXCLab/CODA)

- CODA (Coordinating the Cerebrum and Cerebellum for a Dual-Brain Computer Use Agent with Decoupled Reinforcement Learning): introduces a novel trainable compositional framework that synergizes a Planner (Generates high-level plans) with an Executor (Executes low-level GUI actions), supported by a Reward Signal (Provides learning feedback) and Decoupled RL (Optimizes Planner separately), to create a dual-brain computer use agent. 
- The framework, inspired by the human brain's functional architecture, decouples high-level planning from low-level motor control, allowing the Planner to adapt through experience while the Executor provides stable, software-agnostic grounding for GUI actions based on User Instruction (Defines initial task goal) and Planning Tokens (Represents intermediate plans) to produce an Action (Represents GUI commands).
- CODA employs a two-stage training pipeline, including a Judge System (Evaluates agent trajectories) for reward signals, a Task Generator (Produces high-level tasks), and a distributed VM Cluster (Executes tasks in parallel) managed by a Controller (Manages task queue) to collect Trajectories (Records interaction sequences) from various Tasks (Specific objectives for agents) within individual VMs (Isolated execution environment).

---

[Symphony: A Decentralized Multi-Agent Framework for Scalable Collective Intelligence](https://github.com/GradientHQ/Symphony.git)

- Symphony (Decentralized Multi-Agent Framework): introduces a decentralized multi-agent system for scalable collective intelligence, with User (initiates queries), Decentralized Ledger (records capabilities), Gateways (provides APIs), Worker Nodes (execute tasks), Local Engine (quantized LLM), Stage-specific Prompts (guides task phases), Communicator (secure messaging), Planning Agents (decompose tasks), Execution Agents (execute sub-tasks), Beacon-selection protocol (allocates tasks), Chain-of-Thoughts (reasoning paths), and Weighted Result Voting (aggregates results), enabling lightweight LLMs on edge devices to coordinate.
- This framework addresses challenges of centralized LLM-based agent systems by offering privacy-saving, scalable, and fault-tolerant orchestration with low overhead.
- Symphony achieves competitive performance with lower communication and infrastructure costs, enhancing accessibility, privacy, and supporting decentralized agent economies.

---

[CODA: COORDINATING THE CEREBRUM AND CEREBELLUM FOR A DUAL-BRAIN COMPUTER USE AGENT WITH DECOUPLED REINFORCEMENT LEARNING.](https://github.com/OpenIXCLab/CODA)

- CODA (Coordinating the Cerebrum and Cerebellum for a Dual-Brain Computer Use Agent with Decoupled Reinforcement Learning): introduces a novel trainable compositional framework that synergizes a Planner (Generates high-level plans) with an Executor (Executes low-level GUI actions), supported by a Reward Signal (Provides learning feedback) and Decoupled RL (Optimizes Planner separately), to create a dual-brain computer use agent. 
- The framework, inspired by the human brain's functional architecture, decouples high-level planning from low-level motor control, allowing the Planner to adapt through experience while the Executor provides stable, software-agnostic grounding for GUI actions based on User Instruction (Defines initial task goal) and Planning Tokens (Represents intermediate plans) to produce an Action (Represents GUI commands).
- CODA employs a two-stage training pipeline, including a Judge System (Evaluates agent trajectories) for reward signals, a Task Generator (Produces high-level tasks), and a distributed VM Cluster (Executes tasks in parallel) managed by a Controller (Manages task queue) to collect Trajectories (Records interaction sequences) from various Tasks (Specific objectives for agents) within individual VMs (Isolated execution environment).

---

[CODA: COORDINATING THE CEREBRUM AND CEREBELLUM FOR A DUAL-BRAIN COMPUTER USE AGENT WITH DECOUPLED REINFORCEMENT LEARNING.](https://github.com/OpenIXCLab/CODA)

- CODA (Coordinating the Cerebrum and Cerebellum for a Dual-Brain Computer Use Agent with Decoupled Reinforcement Learning): introduces a novel trainable compositional framework that synergizes a Planner (high-level thought generation) with an Executor (concrete GUI action execution), trained via a two-stage pipeline using Reward Signal (training feedback calculation) and Decoupled RL (Planner-focused reinforcement learning) to process User Instruction (task definition input) and generate Action (GUI command output).
- The training pipeline leverages a Task Generator (high-level task creation) and Judge System (reward signal generation) within a Distributed VM System (parallel task execution) to collect diverse Trajectories (agent interaction data) for both specialized and generalized Planner training stages.
- This decoupled approach, inspired by the human brain's cerebrum and cerebellum, enables the Planner to adapt through experience while the Executor provides stable, software-agnostic GUI grounding, addressing the trade-off between generalist planning and precise execution in GUI automation.

---

[Private, Verifiable, and Auditable AI Systems](http://arxiv.org/abs/2509.00085v1)

- End-to-End Secure and Auditable AI System: introduces a technical framework for building trustworthy AI systems by integrating cryptographic and secure computing techniques across the AI supply chain, including zkSNARKs (verifiable computation proofs), TEEs (secure hardware enclaves), MPC (distributed private computation), and authenticated delegation protocols (AI agent permissions), to address privacy, verifiability, and auditability challenges in foundation model-based AI.
- The framework leverages zkSNARKs for verifiable ML evaluation and data attestations, enabling proofs of model performance and data provenance without revealing sensitive information.
- It also proposes Private Retrieval Augmented Generation (PRAG) for secure, private querying of distributed databases, and integrates personhood credentials to verify human users behind AI agents, enhancing trust and accountability.

---

[CODA: COORDINATING THE CEREBRUM AND CEREBELLUM FOR A DUAL-BRAIN COMPUTER USE AGENT WITH DECOUPLED REINFORCEMENT LEARNING.](https://github.com/OpenIXCLab/CODA)

- CODA (Coordinating the Cerebrum and Cerebellum for a Dual-Brain Computer Use Agent with Decoupled Reinforcement Learning): introduces a novel trainable compositional framework that synergizes a Planner (Generates high-level plans) with an Executor (Executes low-level GUI actions), supported by a Reward Signal (Provides learning feedback) and Decoupled RL (Optimizes Planner separately), to create a dual-brain computer use agent. 
- The framework, inspired by the human brain's functional architecture, decouples high-level planning from low-level motor control, allowing the Planner to adapt through experience while the Executor provides stable, software-agnostic grounding for GUI actions based on User Instruction (Defines initial task goal) and Planning Tokens (Represents intermediate plans) to produce an Action (Represents GUI commands).
- CODA employs a two-stage training pipeline, including a Judge System (Evaluates agent trajectories) for reward signals, a Task Generator (Produces high-level tasks), and a distributed VM Cluster (Executes tasks in parallel) managed by a Controller (Manages task queue) to collect Trajectories (Records interaction sequences) from various Tasks (Specific objectives for agents) within individual VMs (Isolated execution environment).

---

[CODA: COORDINATING THE CEREBRUM AND CEREBELLUM FOR A DUAL-BRAIN COMPUTER USE AGENT WITH DECOUPLED REINFORCEMENT LEARNING.](https://github.com/OpenIXCLab/CODA)

- CODA (Coordinating the Cerebrum and Cerebellum for a Dual-Brain Computer Use Agent with Decoupled Reinforcement Learning): introduces a novel trainable compositional framework that synergizes a Planner (Generates high-level plans) with an Executor (Executes low-level GUI actions), supported by a Reward Signal (Provides learning feedback) and Decoupled RL (Optimizes Planner separately), to create a dual-brain computer use agent. 
- The framework, inspired by the human brain's functional architecture, decouples high-level planning from low-level motor control, allowing the Planner to adapt through experience while the Executor provides stable, software-agnostic grounding for GUI actions based on User Instruction (Defines initial task goal) and Planning Tokens (Represents intermediate plans) to produce an Action (Represents GUI commands).
- CODA employs a two-stage training pipeline, including a Judge System (Evaluates agent trajectories) for reward signals, a Task Generator (Produces high-level tasks), and a distributed VM Cluster (Executes tasks in parallel) managed by a Controller (Manages task queue) to collect Trajectories (Records interaction sequences) from various Tasks (Specific objectives for agents) within individual VMs (Isolated execution environment).

---

#### 26th August 2025


[BUILDING SELF-EVOLVING AGENTS VIA EXPERIENCE-DRIVEN LIFELONG LEARNING: A FRAMEWORK AND BENCHMARK](https://github.com/ECNU-ICALK/ELL-StuLife)

- ELL (Experience-driven Lifelong Learning): introduces a framework for building self-evolving agents capable of continuous growth through real-world interaction, featuring Perception, Memory, Learning, Reasoning, and Action modules.
- The framework is supported by StuLife, a benchmark simulating a student's college journey to evaluate lifelong learning capabilities, including memory retention, skill transfer, and self-motivated behavior.
- The research reveals current LLMs' limitations in self-motivation and long-term memory, emphasizing context engineering's crucial role in advancing AGI.

---



[Optimizing Highway Traffic Flow in Mixed Autonomy: A Multiagent Truncated Rollout Approach](http://arxiv.org/abs/2508.19203v1)

- Multiagent Truncated Rollout Approach: introduces a novel method for optimizing highway traffic flow in mixed autonomy, integrating a PDE-ODE coupled model, a system-level density evolution equation, and a distributed coordination control framework.
- The approach employs independent MPC controllers for each CAV, an agent-by-agent sequential optimization mechanism for explicit cooperation, and a truncated rollout scheme to adaptively shorten the optimization horizon based on objective function bounds.
- This framework enhances CAV speed coordination, improves highway throughput, and reduces computational overhead by leveraging real-time policy sharing and dynamic horizon adjustment, ensuring system stability and performance improvement.

---

[Real-Time Model Checking for Closed-Loop Robot Reactive Planning](http://arxiv.org/abs/2508.19186v1)

- Agent Architecture: introduces a novel real-time model checking approach for closed-loop robot reactive planning, with Robot (mobile platform), LIDAR (2D laser scanner), Motors (actuation), Raspberry Pi 3 Model B (onboard computer), Environment (robot's surroundings), Disturbance D (environmental obstacle), Task Controller (orchestrates tasks), Model Checking (planning algorithm), Tasks (closed-loop control systems: Default/Finite straight/Rotate left/Rotate right), Disturbance-Focused Transition System (robot behavior model), Nondeterministic Finite Automaton (LTL property checker), Product Transition System (combined state-space model), Lateral Partitions (spatial reasoning for turns), Longitudinal Partitions (spatial reasoning for straight paths), Safe Zone (collision-free region), and Shield Partition (proximal disturbance detection), where it enables efficient multi-step planning and obstacle avoidance on a low-powered autonomous robot.
- This framework generates plans in situ based on "core" knowledge and attention, chaining temporary control systems to counteract disturbances without relying on pre-computed data or extensive prior experience.
- The approach utilizes a novel discretization of 2D LiDAR data and forward depth-first search to create efficient multi-step plans for local obstacle avoidance, demonstrating improved performance over single-step reactive agents in cul-de-sac and playground scenarios.

---

[SecureV2X: An Efficient and Privacy-Preserving System for Vehicle-to-Everything (V2X) Applications](http://arxiv.org/abs/2508.19115v1)

- SecureV2X: introduces an efficient and privacy-preserving system for Vehicle-to-Everything (V2X) applications, with CryptoDrowsy (Secure driver drowsiness detection module), FastSec-YOLO (Secure red-light violation detection module), Client (Vehicle/user holding EEG or image data), Server (Edge server/cloud holding model weights), Secure Mediating Agent (Third-party for Beaver's triples distribution), CrypTen MPC Framework (Underlying secure computation library), Private Model Weights (Proprietary neural network parameters), Private Data (Sensitive user input, e.g., EEG, video), Secure Computation (Joint execution of inference protocols), Secure Inference Setting (Operational environment for secure V2X applications), and Violation Alert! (Output for detected red-light violations), which enables secure neural network inferences between servers and vehicles for critical safety tasks.
- The system addresses privacy concerns in V2X by implementing two multi-agent applications: secure drowsiness detection using CompactCNN and secure red-light violation detection via YOLOv5, both built upon novel cryptographic protocol constructions.
- SecureV2X significantly outperforms state-of-the-art secure systems in terms of inference speed, communication rounds, and computational efficiency, making it suitable for real-time, time-sensitive safety applications while preserving user data privacy and model security.

---

[A Concurrent Modular Agent: Framework for Autonomous LLM Agents](http://arxiv.org/abs/2508.19042v1)

- CMA (Concurrent Modular Agent): introduces a framework orchestrating multiple asynchronous LLM-based modules, a shared vector store, and inter-module communication for coherent, fault-tolerant agent behavior.
- This framework enables flexible, adaptive, and context-dependent behavior by offloading reasoning to LLMs and allowing intention to emerge from language-mediated interactions among autonomous processes.
- Demonstrated on physical robotic platforms (Plantbot, ALTER3), the architecture supports robust, scalable AI systems exhibiting emergent cognitive phenomena like self-awareness and identity formation.

---

[STARec: An Efficient Agent Framework for Recommender Systems via Autonomous Deliberate Reasoning](http://arxiv.org/abs/2508.18812v1)

- STARec (Slow-Thinking Augmented agent framework): introduces an LLM-based agent framework for recommender systems, featuring a STARec Agent (main processing unit) with a Memory Module (stores user preferences), Fast Thinking for Personalized Ranking (intuitive item ranking), and Slow Thinking for Memory Update (deliberate preference refinement), all supported by Anchored Reinforcement Training (two-stage learning paradigm) comprising SFT Anchoring (foundational capability instillation) with a Teacher Model (generates reasoning data) and Filter and Augment (refines SFT dataset), and RL Enhancing (policy optimization) with a GRPO Algorithm (reinforcement learning optimizer) and Ranking-Oriented Reward (guides ranking decisions), integrated through a Continuous Learning Cycle (dynamic adaptation mechanism).
- This framework models each user as an autonomous agent with dual-process cognition, enabling both rapid, intuitive responses for immediate interactions and slow, deliberative reasoning for continuous preference adaptation and memory refinement.
- The anchored reinforcement training strategy bridges the gap between LLMs' generic knowledge and domain-specific reasoning, using structured knowledge distillation and preference-aligned reward shaping to cultivate intrinsic slow thinking and dynamic policy adaptation.

---

[Governance-as-a-Service: A Multi-Agent Framework for AI System Compliance and Policy Enforcement](http://arxiv.org/abs/2508.18765v1)

- GaaS (Governance-as-a-Service): introduces a modular, policy-driven enforcement layer for AI systems, with Autonomous Agents (LLM-based, rule-based), LLM Agent, Finance Bot, Infrastructure Agent, Policy Loader, Policy Engine, Trust Computation, Violation Checker, Enforcement Engine, Audit Logger, Trust Registry, Secure Release Gate, Compliance Pipeline, Downstream Systems, End Users / Markets, and Human Oversight, designed to govern agent outputs at runtime without modifying internal model logic.
- This framework operates through declarative rule sets and a Trust Factor mechanism, scoring agents based on longitudinal compliance and severity-aware violation history to support coercive, normative, and adaptive interventions.
- GaaS aims to provide scalable, auditable, and adaptive AI oversight for decentralized, open-source agentic ecosystems by treating governance as a provisioned runtime service.

---

[Toward Edge General Intelligence with Agentic AI and Agentification: Concepts, Technologies, and Future Directions](http://arxiv.org/abs/2508.18725v1)

- Agentic AI: introduces a comprehensive framework for edge general intelligence, with Perception (acquires multimodal data), Memory (stores, retrieves knowledge), Reasoning (plans, reasons, decides), and Action (executes decisions, interacts) modules, enabling autonomous perception-reasoning-action loops in dynamic edge environments.
- This framework leverages LLMs as cognitive cores for semantic comprehension and planning, integrates external tools/APIs to extend capabilities, and utilizes a continuous feedback loop for iterative self-refinement and adaptation.
- The system aims to overcome limitations of traditional edge AI by providing robust, scalable, and human-aligned solutions for complex tasks in resource-constrained 6G-enabled networks.

---

[Bias Mitigation Agent: Optimizing Source Selection for Fair and Balanced Knowledge Retrieval](http://arxiv.org/abs/2508.18724v1)

- Bias Mitigation Agent: introduces a supervisor-based multi-agent system for bias mitigation, with a Manager Agent (coordinates workflow), Knowledge Agent (retrieves documents), Bias Detector Agent (evaluates bias), Source Selector Agent (selects unbiased sources), and Writer Agent (synthesizes answer), where the system optimizes source selection for fair and balanced knowledge retrieval.
- This framework uses a centralized Manager Agent to supervise execution flow, maintain system state, and coordinate decisions among specialized Worker Agents (Knowledge, Bias Detector, Source Selector, Writer) to ensure relevant and minimally biased content.
- The system supports "No Source Selection", "Zero-Shot", and "Few-Shot" operational modes, allowing flexible trade-offs between computational efficiency, fairness enforcement, and generalization capabilities in knowledge retrieval tasks.

---

[FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation](http://arxiv.org/abs/2508.18684v1)

- FALCON (Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation): introduces an autonomous agentic framework that generates deployable Intrusion Detection System (IDS) rules from Cyber Threat Intelligence (CTI) data, incorporating LLM-driven generation, multi-phased validation, and human oversight to automate the entire rule-generation pipeline.
- The framework addresses the challenge of rapidly evolving cyber threats by enabling real-time IDS rule generation and updates for both network (Snort) and host-based (YARA) environments, ensuring syntactic correctness, semantic alignment, and performance optimization.
- FALCON integrates LLM-driven data mining with iterative feedback loops and human oversight, significantly reducing manual effort and enhancing the agility and accuracy of threat detection systems.

---

[MUA-RL: MULTI-TURN USER-INTERACTING AGENT REINFORCEMENT LEARNING FOR AGENTIC TOOL USE](http://arxiv.org/abs/2508.18669v1)

- MUA-RL (Multi-turn User-interacting Agent Reinforcement Learning for agentic tool use): introduces a novel reinforcement learning framework that integrates LLM-simulated users into the RL loop for agentic tool use, including an Agent LLM, User LLM, Tool LLM/MCP server, External Database, Reinforcement Learning Loop, GRPO, Synthesized Database, Trajectory Verifiers, Reward Mechanism, Cold-start Training Phase, and Multi-turn Rollout Process.
- This framework enables autonomous learning for agents to efficiently communicate with users and utilize various tools to solve dynamic multi-turn interaction problems.
- MUA-RL employs a simplified, task-oriented reward design and a cold-start phase to develop robust behavioral patterns and enhance generalization across diverse tool-using tasks.

---

[Bimodal Dynamics of the Artificial Limit Order Book Stock Exchange with Autonomous Traders](http://arxiv.org/abs/2508.17837v1)

- ASME (Artificial Stock Market Exchange): introduces a framework for an artificial stock market with autonomous, myopic traders interacting through a limit order book, revealing intrinsic bistability and complex dynamics.
- The framework utilizes an HMM to analyze bifurcative dynamics, identifying two distinct long-run price equilibria: a deterministic zero-price state and a persistent positive-price equilibrium.
- The paper employs Logistic Regression and Gradient Boosting Machines to predict trajectory outcomes and various complexity measures (Fractal Dimension, Entropy, LLE) to characterize the system's structured, yet dynamically rich, behavior.

---

[MATRIX: Multi-Agent simulaTion fRamework for safe Interactions and contextual clinical conversational evaluation](http://arxiv.org/abs/2508.19163v1)

- MATRIX (Multi-Agent simulaTion fRamework for safe Interactions and contexTual clinical conversational evaluation): introduces a structured, extensible framework for safety-oriented evaluation of clinical dialogue agents, comprising a Structured Safety Library, PatBot (LLM-based Simulated Patient Agent), a Clinical History Taking Agent (LLM Target System), BehvJudge (LLM-based Safety Evaluator), a Clinical Use-Case Specific Context, and System Performance Output.
- The framework enables systematic and scalable safety evaluation by unifying structured safety engineering with validated conversational AI evaluation, supporting regulator-aligned safety auditing.
- It benchmarks LLM agents across simulated clinical dialogues, identifying failure patterns in safety-critical scenarios, and demonstrates that LLM-based evaluators can surpass human performance in hazard detection.

---

[DELIVER: A System for LLM-Guided Coordinated Multi-Robot Pickup and Delivery using Voronoi-Based Relay Planning](http://arxiv.org/abs/2508.19114v1)

- DELIVER (Directed Execution of Language-instructed Item Via Engineered Relay): introduces a fully integrated system for cooperative multi-robot pickup and delivery, with Natural Language Understanding (parses natural language commands), Voronoi Partitioning (divides environment into robot regions), Pickup and Drop Agent Identification (assigns robots to task endpoints), Active Agent Selection (selects robots for relay path), Relay Point Selection (calculates handover locations), and Relay Execution (manages robot movement and handoffs).
- The system unifies LLM-based natural language understanding, Voronoi-based spatial decomposition for region-aware planning, relay-point computation for inter-agent coordination, and execution through local finite-state machines with lightweight signaling.
- DELIVER demonstrates scalability and efficient agent utilization by reducing per-agent workload by up to 55% compared to single-agent systems, maintaining consistent mission cost and low coordination overhead.

---

[Reasoning LLMs in the Medical Domain: A Literature Survey](http://arxiv.org/abs/2508.19097v1)

- Reasoning LLMs in the Medical Domain: introduces a comprehensive literature survey on the current state and future potential of reasoning LLMs within the medical domain, examining their transformative role in healthcare applications.
- The survey analyzes enabling technological foundations like Chain-of-Thought and Reinforcement Learning, alongside emerging paradigms such as specialized medical LLMs, multi-agent systems, and innovative prompting architectures.
- It critically assesses current evaluation methodologies, addresses persistent challenges, and delineates a roadmap for developing reliable, safe, and ethically aligned LLMs for medical use.

---

[Trustworthy Agents for Electronic Health Records through Confidence Estimation](http://arxiv.org/abs/2508.19096v1)

- TrustEHRAgent: introduces a confidence-aware clinical agent for Electronic Health Records (EHR) that integrates step-wise confidence estimation (tracks uncertainty per step) and a confidence estimator (computes final confidence) to make threshold-based decision making (decides answer or reject) for clinical question answering.
- The framework leverages token probability (confidence score input) and weighted average (calculates final confidence) within its Confidence Estimator to derive a final confidence score, which is then compared against a predefined reliability threshold (τ) to either provide an answer (provides confident answers) or reject the query (abstains from uncertain queries).
- This approach enhances reliability by enabling the agent to transparently express uncertainty and abstain from answering when confidence is low, thereby preventing potential errors and improving patient safety in high-stakes medical contexts.

---

[HIPLAN: Hierarchical Planning for LLM Agents with Adaptive Global-Local Guidance](http://arxiv.org/abs/2508.19076v1)

- HIPLAN (Hierarchical Planning for LLM Agents with Adaptive Global-Local Guidance): introduces a hierarchical planning framework that provides adaptive global-local guidance to boost LLM-based agents' decision-making, with all components including LLM (generates milestones, hints, actions), Milestone Library (stores structured expert experience), Milestone Action Guide (provides global task direction), Step-Wise Hints (offers local action feedback), Expert Demonstrations (source for experience library), Milestones Extraction (segments trajectories into subgoals), Task-Level Similarity Search (retrieves relevant tasks), Milestone-Level Similarity Search (retrieves relevant trajectory fragments), Agent Policy (integrates guidance for actions), and Embeddings (vector representations for retrieval), enabling LLM-based agents to tackle complex, long-horizon tasks through integrated global and local guidance.
- The framework constructs a milestone library offline from expert demonstrations, which is then used during execution to retrieve relevant task and milestone-level experiences for generating dynamic global milestone action guides and local step-wise hints.
- This dual-level guidance mechanism enhances efficiency, controllability, and overall robustness by maintaining global coherence while adapting actions to dynamic local contexts, outperforming baselines on ALFWorld and WebShop benchmarks.

---

[MovieCORE: COgnitive REasoning in Movies](http://arxiv.org/abs/2508.19026v1)

- MovieCORE (COgnitive REasoning in Movies): introduces a novel video question answering (VQA) dataset designed to probe deeper cognitive understanding of movie content, generated using an agentic brainstorming approach.
- This approach leverages multiple LLMs as specialized agents—including a Critic Agent (MC), System II VQA Expert, Skeptical Researcher, Detective, and Meta Reviewer—to generate and refine high-quality, thought-provoking question-answer pairs, validated by Human Reviewers and informed by Video Context Extraction (MiniCPM-v2.6).
- The paper also proposes Agentic Choice Enhancement (ACE), a post-training plugin that improves existing VLMs' reasoning capabilities by using an ACE Existing VLM, ACE Beam Search, and ACE Llama-3.2 for response generation and re-ranking.

---

[GitTaskBench: A Benchmark for Code Agents Solving Real-World Tasks Through Code Repository Leveraging](https://github.com/QuantaAlpha/GitTaskBench)

- GitTaskBench: introduces a benchmark for code agents, evaluating their ability to solve real-world tasks by leveraging code repositories, which includes Task & Repository Selection, Completeness Verification, an Execution Framework for agent workflow, and an Evaluation Framework with defined success criteria and a practical utility (alpha-value) metric.
- This benchmark systematically assesses agents' overall coding mastery, task-oriented execution, and autonomous environment provisioning across 54 real-life, multimodal tasks from 7 domains, using human-curated evaluation scripts.
- It also proposes a novel "alpha-value" metric to quantitatively assess agent economic benefits, integrating task success, token cost, and average developer salaries, providing actionable insights for agent deployment.

---

[Interactive Evaluation of Large Language Models for Multi-Requirement Software Engineering Tasks](http://arxiv.org/abs/2508.18905v1)

- Interactive, Dependency-Grounded Assessment: introduces a novel interactive evaluation framework for LLMs on multi-requirement programming tasks, featuring a structured, feedback-driven dialogue between an Interviewer (LLM-based, generates feedback) and an Interviewee (LLM under evaluation), supported by Task specification (defines problem parameters), Reference Solution (ground-truth for guidance), Evaluation Guidelines (criteria for assessment), History (stores interaction dialogue), Report (structured performance analysis), Executor (runs interviewee code), Solution Output (results from code execution), Solution (interviewee's code response), Solution Protocol (defines solution structure), and Delivery Format (specifies output format).
- This framework models tasks as requirement dependency graphs, allowing an LLM-based interviewer to provide minimal, targeted hints to an interviewee model for error correction and constraint fulfillment.
- The dynamic protocol enables fine-grained diagnostic insights into model behavior, uncovering strengths and systematic weaknesses that static benchmarks fail to measure, and guides the interviewee through iterative refinement loops.

---

[Judicial Requirements for Generative AI in Legal Reasoning](http://arxiv.org/abs/2508.18880v1)

- No single overarching framework is proposed; the paper analyzes existing AI enhancement mechanisms: introduces an analysis of AI enhancement mechanisms, including Fine-tuning, Retrieval-Augmented Generation (RAG), Task Decomposition and Chained Prompts, Tree of Thoughts (ToT), Neuro-Symbolic AI, Multi-Agent Systems, Structured Self-Evaluation, and Logit-based Confidence Scoring, to assess their potential in meeting judicial requirements for generative AI in legal reasoning.
- The study uses the IRAC (Issue-Rule-Application-Conclusion) model as an analytical framework, focusing on the challenging phases of legal adjudication: determining the applicable Rule (R) and performing the Application (A) of that rule to the facts of a case.
- The findings indicate that while these techniques can address specific challenges, significant challenges remain, particularly in tasks requiring discretion and transparent, justifiable reasoning, concluding that the most effective current role for AI in law is a dual one: as a high-volume assistant for simple, repetitive cases and as a sophisticated "sparring partner" for human experts in complex matters.

---

[A Survey on Cloud-Edge-Terminal Collaborative Intelligence in AIoT Networks](http://arxiv.org/abs/2508.18803v1)

- CETCI (Cloud-Edge-Terminal Collaborative Intelligence): introduces a comprehensive survey on cloud-edge-terminal collaborative intelligence in AIoT networks, with Cloud Layer (centralized computing, global storage), Edge Layer (distributed processing, real-time inference), Terminal Layer (data acquisition, IoT device control), Network Virtualization (flexible network infrastructure), Container Orchestration (application deployment management), Software-Defined Networking (SDN) (centralized network control), AI/ML Integration Platforms (intelligent decision-making), Resource Management (optimizes task offloading, allocation), Task Offloading (learning-based, game theory/optimization), Resource Allocation (learning-based, energy-aware, QoS-driven), Optimization Techniques (linear/convex programming, game theory), Collaborative Learning (develops intelligent models), Federated Learning (FL) (privacy-preserving, robust learning), Distributed Deep Learning (DDL) (model/data parallelism), Model Evolution (compression, distillation, incremental learning), RL Optimization (resource management, multi-agent RL), Security & Privacy (protects data flow, system integrity), Security Threats (data breaches, DoS attacks), Security Mechanisms (encryption, authentication, IDS/IPS), Privacy Technologies (FL, differential privacy, homomorphic encryption), Data Management & Communication (foundational data infrastructure), Data Acquisition & Preprocessing (filtering, aggregation, compression), Storage & Retrieval (edge caching, distributed storage), Communication & Optimization (MQTT/CoAP, bandwidth optimization), Performance Metrics (latency, energy, utilization, QoS/QoE), and Application Domains (smart manufacturing, transportation, healthcare, cities, agriculture), where the paper systematically analyzes architectural components, enabling technologies, and collaboration paradigms across heterogeneous network infrastructures.
- The survey provides a tutorial-style review for beginners in CISAIoT, examining core technologies like network virtualization, container orchestration, and software-defined networking, while presenting multi-perspective categorizations of collaboration paradigms.
- It further explains intelligent collaboration learning frameworks by reviewing recent advances in federated learning, distributed deep learning, edge-cloud model evolution, and reinforcement learning-based approaches, discussing challenges and future development trends including LLMs and agents.

---

[CausalMACE: Causality Empowered Multi-Agents in Minecraft Cooperative Tasks](http://arxiv.org/abs/2508.18797v1)

- CausalMACE (Causality Empowered Multi-Agents in Minecraft Cooperative Tasks): introduces a holistic causality planning framework designed to enhance multi-agent systems in Minecraft, incorporating causality to manage dependencies among subtasks, with Judger (defines objectives/feedback), Planner (decomposes/graphs dependencies), Planner-Task Decomposition (breaks into subtasks), Planner-Factual Graph (FG) (initial dependency graph), Planner-Counterfactual Graph (CG) (causal inference graph), Planner-Graph Refinement (refines graph causally), Planner-ATE (Average Treatment Effect) (quantifies causal effect), Planner-LLMs (decompose/identify dependencies), Worker (assigns/executes subtasks), Worker-Agent Assignment (distributes subtasks), Worker-Path Sampling (explores execution paths), Worker-Busy Rate (br) (balances workload), Agents (execute/reflect autonomously), and Game Environment (Minecraft interactive world) components.
- The framework leverages an overarching task graph for global task planning and a causality-based module for dependency management, utilizing LLMs for task decomposition and causal intervention to refine the task graph.
- CausalMACE achieves state-of-the-art performance in multi-agent cooperative tasks by ensuring efficient task arrangement and execution through structured dependency management and balanced workload distribution.

---

[VistaWise: Building Cost-Effective Agent with Cross-Modal Knowledge Graph for Minecraft](http://arxiv.org/abs/2508.18722v1)

- VistaWise: introduces a cost-effective agent framework for Minecraft, integrating an LLM, text-modal and cross-modal graph construction, task-specific information retrieval, a memory stack, and a desktop-level skill library.
- The framework enhances decision-making by combining domain-specific knowledge from a cross-modal knowledge graph with real-time visual perception via a finetuned object detection model.
- VistaWise enables direct desktop control through mouse and keyboard inputs, reducing reliance on environmental APIs and achieving state-of-the-art performance in open-world tasks with significantly lower development costs.

---

[AppAgent-Pro: A Proactive GUI Agent System for Multidomain Information Integration and User Assistance](http://arxiv.org/abs/2508.18689v1)

- AppAgent-Pro: introduces a proactive GUI agent system that actively integrates multi-domain information based on user instructions, with its Comprehension Stage (analyzes user instructions), Cognitive Agent (LLM-based analysis/synthesis), Proactive Thinking (anticipates user needs), Execution Stage (autonomously interacts apps), Proactive Execution Agent (LLM-driven app interaction), Shallow Execution Mode (fast, surface-level retrieval), Deep Execution Mode (in-depth, iterative mining), Integration Stage (combines diverse information), and Personalization (leverages interaction history) components, designed to anticipate user needs and conduct in-depth multi-domain information mining.
- The system operates through a three-stage pipeline—Comprehension, Execution, and Integration—enabling it to proactively acquire relevant knowledge, understand user intent, perform appropriate actions, and integrate results into coherent outputs.
- AppAgent-Pro enhances efficiency, personalization, and depth of information access by moving beyond reactive LLM-based agents to a proactive paradigm that integrates and reasons across heterogeneous information domains.

---

[Utilizing Training Data to Improve LLM Reasoning for Tabular Understanding](http://arxiv.org/abs/2508.18676v1)

- LRTab (Learn then Retrieve): introduces a novel prompting-based reasoning approach that integrates training data insights by generating and retrieving "Prompt Conditions" to improve LLM tabular understanding.
- The framework leverages a Code-Augmented LLM to generate Chain-of-Thought responses and, for incorrect answers, employs a Prompt Condition Generation Module to predict and verify error-correcting conditions, which are then stored in a Knowledge Base.
- At inference, LRTab utilizes a Table Encoder and a Retrieval Module, refined by a Crossencoder Reranker, to retrieve the most relevant Prompt Conditions, providing additional context to the Code-Augmented LLM for accurate tabular reasoning.

---

[GitTaskBench: A Benchmark for Code Agents Solving Real-World Tasks Through Code Repository Leveraging](https://github.com/QuantaAlpha/GitTaskBench)

- GitTaskBench: introduces a benchmark for code agents, evaluating their ability to solve real-world tasks by leveraging code repositories, which includes Task & Repository Selection, Completeness Verification, an Execution Framework for agent workflow, and an Evaluation Framework with defined success criteria and a practical utility (alpha-value) metric.
- This benchmark systematically assesses agents' overall coding mastery, task-oriented execution, and autonomous environment provisioning across 54 real-life, multimodal tasks from 7 domains, using human-curated evaluation scripts.
- It also proposes a novel "alpha-value" metric to quantitatively assess agent economic benefits, integrating task success, token cost, and average developer salaries, providing actionable insights for agent deployment.

---

[Requirements Development and Formalization for Reliable Code Generation: A Multi-Agent Vision](http://arxiv.org/abs/2508.18675v1)

- REDEFO (Requirements Development and Formalization): introduces a multi-agent framework for reliable code generation, with Analyst (interprets, structures NLRs), Formalizer (translates, assesses specifications), Coder (generates, verifies code), Knowledge Source (provides background knowledge), and Human Experts (provide review, feedback) components, designed to transform Natural Language Requirements (NLRs) into provably correct software artifacts through formal specification and verification.
- The framework leverages formal methods to bridge the gap between ambiguous NLRs and precise executable code, enabling rigorous reasoning, bug uncovering, and enforcement of critical properties throughout the software development process.
- REDEFO aims to enhance the quality and correctness of auto-generated software by integrating human-in-the-loop interaction and expert judgment at critical points, culminating in a trustworthy software product.

---

[Bias-Adjusted LLM Agents for Human-Like Decision-Making via Behavioral Economics](http://arxiv.org/abs/2508.18600v1)

- Persona-Based Approach: introduces a method for adjusting LLM biases using individual-level behavioral traits, with Behavioral Traits Dataset (stores individual human data), Persona Injection Module (assigns traits to LLMs), LLM Agents (simulate human decision-making), Economic Game Simulation Environment (executes decision-making scenarios), and Output Analysis Module (compares LLM output to human data), to better align LLM decisions with human behavior in economic games.
- This approach leverages the Econographics dataset to characterize LLM agents with unique personas based on real individual behavioral and demographic attributes, enabling the construction of a heterogeneous agent population.
- The method demonstrates improved alignment between simulated and empirical human behavior, particularly for the responder role in the ultimatum game, highlighting the potential of persona-conditioned LLMs for scalable human-like behavior simulation.

---

[Reliable Weak-to-Strong Monitoring of LLM Agents](https://scale.com/research/mrt)

- MRT (Monitor Red Teaming) Workflow: introduces a standardized workflow for stress-testing and improving LLM agent oversight, featuring LLM Monitor Scaffolding, a Monitor LLM, LLM Agents, Threat Models, Evasion Strategies, SHADE-Arena, CUA-SHADE-Arena, and Human-in-the-Loop (HiLT) Monitoring, to evaluate monitoring systems against adversarial LLM agents.
- The workflow evaluates various monitor architectures, including a novel Hybrid Hierarchical-Sequential Scaffolding, demonstrating that architectural design is more impactful for reliability than the monitor's awareness level.
- The research also shows that targeted human oversight within the HiLT Monitoring component, particularly on pre-flagged suspicious activities, significantly improves true positive rates, enabling weaker models to reliably monitor stronger agents.

---

[QAgent: An LLM-based Multi-Agent System for Autonomous OpenQASM programming](http://arxiv.org/abs/2508.20134v1)

- QAgent: introduces an LLM-powered multi-agent system for autonomous OpenQASM programming, featuring a hierarchical design with specialized Dynamic-few-shot and Tools-augmented Coders that leverage RAG, few-shot learning, and CoT reasoning for code generation and refinement.
- The system dispatches tasks to either the Dynamic-few-shot Coder for structured problems or the Tools-augmented Coder for complex, parameterized tasks, employing iterative self-reflection for debugging and optimization.
- This multi-agent framework significantly enhances QASM code generation accuracy, making quantum programming more accessible and robust for non-experts.

---

[Reflective Agreement: Combining Self-Mixture of Agents with a Sequence Tagger for Robust Event Extraction](http://arxiv.org/abs/2508.19359v1)

- ARIS (Agreement-based Reflective Inference System): introduces a hybrid event extraction framework that systematically integrates a discriminative Sequence Tagger and a generative Self Mixture of Agents, leveraging model consensus, confidence-based filtering, and an LLM Reflection Module to resolve ambiguities.
- The framework employs Decomposed Instruction Fine-Tuning to equip the LLM with specialized capabilities for event subtasks, enhancing its accuracy and reliability in reflective reasoning.
- ARIS utilizes structured prompts, Triggers Reflection Prompt and Argument Reflection Prompt, to guide the Reflection Module (LLM) in classifying trigger candidates and validating argument roles for robust event extraction.

---

[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for chest X-ray triage, featuring Data Ingestion (detects/preprocesses cases), Uncertainty Check (computes confidence/OOD), Agentic Decision Routing (iteratively selects tools via guardrailed policy), and Triage and Explainability Artifacts (auto-sorts cases, generates CAM/LWI).
- The framework employs a Router, which can be either a deterministic rule-based or an LLM-decided policy, to select from a Toolbox of Diagnosis Assist (TTA, MoE, VLM), LWI Computation (segmentation, suppression, LWI), and Visualization (CAM) tools.
- This system aims to make autonomous, safe triage decisions under clinical constraints by estimating per-case confidence and distributional fit, enabling selective automation with auditable operation, and providing complementary operating points for throughput or accuracy.

---



[GitTaskBench: A Benchmark for Code Agents Solving Real-World Tasks Through Code Repository Leveraging](https://github.com/QuantaAlpha/GitTaskBench)

- GitTaskBench: introduces a benchmark for code agents, with Task & Repository Selection (identifies real-world tasks and relevant GitHub repositories), Completeness Verification (ensures repositories are fully operational and self-contained), Execution Framework (defines agent interaction with tasks and environments), and Evaluation Framework (measures agent performance and economic benefit), designed to systematically assess agents' capability in solving real-world tasks by leveraging code repositories.
- The benchmark covers 54 real-life, multimodal tasks across 7 domains, each paired with a relevant repository and an automated, human-curated evaluation harness.
- It proposes the alpha-value metric to quantify the economic benefit of agent performance, integrating task success rates, token cost, and average developer salaries for a comprehensive cost-benefit analysis.

---

[Reliable Weak-to-Strong Monitoring of LLM Agents](http://arxiv.org/abs/2508.19461v1)

- MRT (Monitor Red Teaming) Workflow: introduces a standardized workflow for stress-testing LLM agent monitoring systems, integrating LLM Agents, LLM Monitors with diverse Monitor Scaffolding (Baseline, Sequential, Hierarchical, Hybrid), Attackers using Evasion Strategies, Environments (SHADE-Arena, CUA-SHADE-Arena), a Human-in-the-Loop (HiLT) System, Tools, and Evaluation Metrics.
- The paper empirically evaluates monitor reliability under various threat models, agent/monitor awareness levels, and scaffolding designs, highlighting the hybrid scaffolding's superior robustness against adversarial attacks.
- The research demonstrates that architectural design (scaffolding) is more critical for improving monitor reliability than increased monitor awareness, enabling weaker models to effectively oversee stronger agents.

---

[Aleks: AI powered Multi Agent System for Autonomous Scientific Discovery via Data-Driven Approaches in Plant Science](http://arxiv.org/abs/2508.19383v1)

- Aleks (AI-powered Multi Agent System): introduces an AI-powered multi-agent system for autonomous scientific discovery, featuring a Domain Scientist Agent (provides domain knowledge/feedback), a Data Analyst Agent (proposes modeling strategies/refines analysis), a Machine Learning Engineer Agent (implements models/generates code/executes experiments), Shared Agent Memory (stores experimental records/facilitates communication), Episodic Memory (agent-specific task history), Semantic Memory (agent-specific knowledge base), Human Research Team (provides input/receives output), Research Questions & Datasets (initial input for discovery), and a Tool Space (MLE agent's execution environment), with provisions for Other Possible Agents (future specialized agents).
- Aleks autonomously conducts data-driven scientific discovery by iteratively formulating problems, exploring modeling strategies, and refining solutions without human intervention, leveraging specialized LLM-powered agents that collaborate through a shared memory architecture.
- The system balances automated exploration with interpretability and domain relevance, integrating domain knowledge and memory to achieve robust and coherent outcomes in scientific research, as demonstrated in a case study on grapevine red blotch disease.

---

[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for autonomous, safe chest X-ray triage decisions under clinical constraints, comprising Data Ingestion, Uncertainty Check, Agentic Decision Routing, and Triage & Explainability Artifacts.
- The system estimates per-case confidence and distributional fit, then employs a guardrailed, stepwise policy with a toolbox of verification and consultation tools, including TTA, MoE, and VLM, to either issue an automated decision or abstain for human intervention.
- It evaluates two router designs, a deterministic rule-based router and an LLM-decided router, offering complementary operating points to prioritize either maximal throughput or maximal accuracy while outperforming existing LLMs and supervised classifiers.

---


[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for autonomous, safe chest X-ray triage decisions under clinical constraints, comprising Data Ingestion, Uncertainty Check, Agentic Decision Routing, and Triage & Explainability Artifacts.
- The system estimates per-case confidence and distributional fit, then employs a guardrailed, stepwise policy with a toolbox of verification and consultation tools, including TTA, MoE, and VLM, to either issue an automated decision or abstain for human intervention.
- It evaluates two router designs, a deterministic rule-based router and an LLM-decided router, offering complementary operating points to prioritize either maximal throughput or maximal accuracy while outperforming existing LLMs and supervised classifiers.

---

[GitTaskBench: A Benchmark for Code Agents Solving Real-World Tasks Through Code Repository Leveraging](https://github.com/QuantaAlpha/GitTaskBench)

- GitTaskBench: introduces a benchmark for code agents, evaluating their ability to solve real-world tasks by leveraging code repositories, which includes Task & Repository Selection, Completeness Verification, an Execution Framework for agent workflow, and an Evaluation Framework with defined success criteria and a practical utility (alpha-value) metric.
- This benchmark systematically assesses agents' overall coding mastery, task-oriented execution, and autonomous environment provisioning across 54 real-life, multimodal tasks from 7 domains, using human-curated evaluation scripts.
- It also proposes a novel "alpha-value" metric to quantitatively assess agent economic benefits, integrating task success, token cost, and average developer salaries, providing actionable insights for agent deployment.

---

[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for chest X-ray triage, featuring Data Ingestion (detects/preprocesses cases), Uncertainty Check (computes confidence/OOD), Agentic Decision Routing (iteratively selects tools via guardrailed policy), and Triage and Explainability Artifacts (auto-sorts cases, generates CAM/LWI).
- The framework employs a Router, which can be either a deterministic rule-based or an LLM-decided policy, to select from a Toolbox of Diagnosis Assist (TTA, MoE, VLM), LWI Computation (segmentation, suppression, LWI), and Visualization (CAM) tools.
- This system aims to make autonomous, safe triage decisions under clinical constraints by estimating per-case confidence and distributional fit, enabling selective automation with auditable operation, and providing complementary operating points for throughput or accuracy.

---

[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for autonomous, safe chest X-ray triage decisions under clinical constraints, comprising Data Ingestion, Uncertainty Check, Agentic Decision Routing, and Triage & Explainability Artifacts.
- The system estimates per-case confidence and distributional fit, then employs a guardrailed, stepwise policy with a toolbox of verification and consultation tools, including TTA, MoE, and VLM, to either issue an automated decision or abstain for human intervention.
- It evaluates two router designs, a deterministic rule-based router and an LLM-decided router, offering complementary operating points to prioritize either maximal throughput or maximal accuracy while outperforming existing LLMs and supervised classifiers.

---

[GitTaskBench: A Benchmark for Code Agents Solving Real-World Tasks Through Code Repository Leveraging](https://github.com/QuantaAlpha/GitTaskBench)

- GitTaskBench: introduces a benchmark for code agents, evaluating their ability to solve real-world tasks by leveraging code repositories, which includes Task & Repository Selection, Completeness Verification, an Execution Framework for agent workflow, and an Evaluation Framework with defined success criteria and a practical utility (alpha-value) metric.
- This benchmark systematically assesses agents' overall coding mastery, task-oriented execution, and autonomous environment provisioning across 54 real-life, multimodal tasks from 7 domains, using human-curated evaluation scripts.
- It also proposes a novel "alpha-value" metric to quantitatively assess agent economic benefits, integrating task success, token cost, and average developer salaries, providing actionable insights for agent deployment.

---

[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for chest X-ray triage, featuring Data Ingestion (detects/preprocesses cases), Uncertainty Check (computes confidence/OOD), Agentic Decision Routing (iteratively selects tools via guardrailed policy), and Triage and Explainability Artifacts (auto-sorts cases, generates CAM/LWI).
- The framework employs a Router, which can be either a deterministic rule-based or an LLM-decided policy, to select from a Toolbox of Diagnosis Assist (TTA, MoE, VLM), LWI Computation (segmentation, suppression, LWI), and Visualization (CAM) tools.
- This system aims to make autonomous, safe triage decisions under clinical constraints by estimating per-case confidence and distributional fit, enabling selective automation with auditable operation, and providing complementary operating points for throughput or accuracy.

---

[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for autonomous, safe chest X-ray triage decisions under clinical constraints, comprising Data Ingestion, Uncertainty Check, Agentic Decision Routing, and Triage & Explainability Artifacts.
- The system estimates per-case confidence and distributional fit, then employs a guardrailed, stepwise policy with a toolbox of verification and consultation tools, including TTA, MoE, and VLM, to either issue an automated decision or abstain for human intervention.
- It evaluates two router designs, a deterministic rule-based router and an LLM-decided router, offering complementary operating points to prioritize either maximal throughput or maximal accuracy while outperforming existing LLMs and supervised classifiers.

---

[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for autonomous, safe chest X-ray triage decisions under clinical constraints, comprising Data Ingestion, Uncertainty Check, Agentic Decision Routing, and Triage & Explainability Artifacts.
- The system estimates per-case confidence and distributional fit, then employs a guardrailed, stepwise policy with a toolbox of verification and consultation tools, including TTA, MoE, and VLM, to either issue an automated decision or abstain for human intervention.
- It evaluates two router designs, a deterministic rule-based router and an LLM-decided router, offering complementary operating points to prioritize either maximal throughput or maximal accuracy while outperforming existing LLMs and supervised classifiers.

---

[AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays](https://github.com/XLIAaron/uncertainty-aware-cxr-agent)

- AT-CXR (Agentic Triage for Chest X-ray): introduces an uncertainty-aware agentic framework for chest X-ray triage, featuring Data Ingestion (detects/preprocesses cases), Uncertainty Check (computes confidence/OOD), Agentic Decision Routing (iteratively selects tools via guardrailed policy), and Triage and Explainability Artifacts (auto-sorts cases, generates CAM/LWI).
- The framework employs a Router, which can be either a deterministic rule-based or an LLM-decided policy, to select from a Toolbox of Diagnosis Assist (TTA, MoE, VLM), LWI Computation (segmentation, suppression, LWI), and Visualization (CAM) tools.
- This system aims to make autonomous, safe triage decisions under clinical constraints by estimating per-case confidence and distributional fit, enabling selective automation with auditable operation, and providing complementary operating points for throughput or accuracy.

---

#### 25th August 2025

[DiscussLLM: Teaching Large Language Models When to Speak](http://arxiv.org/abs/2508.18167v1)

- DiscussLLM: introduces a framework and dataset to teach LLMs the crucial skill of timely and valuable intervention in human conversations, with all its components, where it addresses the "When to Speak" problem by training models to proactively decide whether to remain silent or intervene with a helpful response.
- The framework utilizes a scalable two-stage data generation pipeline to synthesize a large-scale dataset of realistic multi-turn human discussions, each annotated with an intervention type and a conversational trigger.
- Two architectural baselines are explored: an integrated end-to-end generative model and a decoupled classifier-generator system, evaluating their ability to accurately time interventions and generate high-quality responses.

---

[The AI Data Scientist](http://arxiv.org/abs/2508.18113v1)

- The AI Data Scientist: introduces an autonomous LLM-powered agent that transforms raw data into actionable business recommendations, featuring a Data Cleaning Subagent (cleans, handles missing values, outliers), a Hypothesis Subagent (generates, tests data relationships), a Preprocessing Subagent (prepares data for modeling), a Feature Engineering Subagent (creates predictive features), a Model Training Subagent (trains predictive machine learning models), and a Call-To-Action Subagent (translates findings into recommendations).
- This framework emphasizes a hypothesis-driven approach, where specialized LLM Subagents work sequentially, passing structured metadata to ensure statistically validated insights guide each step from data preparation to final recommendations.
- The system automates the entire end-to-end data science workflow, enabling rapid generation of interpretable results and actionable strategies, significantly reducing the time from evidence to decision-making.

---

[Teaching LLMs to Think Mathematically: A Critical Study of Decision-Making via Optimization](http://arxiv.org/abs/2508.18091v1)

- Structured Roadmap for Advancing LLM Capabilities in Mathematical Programming: introduces a critical study of LLMs in mathematical optimization, proposing future directions via Structured Dataset Construction Framework (builds diverse, robust datasets), Modular Multi-Agent Architectures (decomposes tasks, assigns specialized LLMs), Chain of RAGs (iterative retrieval, external knowledge), Neuro-Symbolic Formulation (combines LLMs, symbolic solvers, verification), and Improved Prompting Strategies (adaptive, structured guidance), to enhance performance in complex optimization tasks.
- The roadmap addresses current LLM limitations in numerical reasoning, input length sensitivity, and reliance on surface-level pattern matching by integrating structured data, multi-agent collaboration, iterative knowledge retrieval, and formal verification.
- Key proposed components include a four-part dataset structure for capturing reasoning steps, specialized LLMs for subtasks, iterative RAG for dynamic context refinement, and neuro-symbolic integration for verifiable and scalable solutions.

---

[PerPilot: Personalizing VLM-based Mobile Agents via Memory and Exploration](http://arxiv.org/abs/2508.18040v1)

- PerPilot: introduces a plug-and-play LLM-powered framework for mobile agents, with Personalization Perception module (identifies personalized instructions, extracts elements), Personalization Completion module (retrieves/explores missing personalized information), Memory-based Retrieval (accesses stored user-specific information), Reasoning-based Exploration (infers apps, generates exploration instructions), and Agent Execution (executes clarified, explicit instructions), enabling autonomous perception, understanding, and execution of personalized user instructions.
- The framework leverages LLMs to identify personalized elements, first attempting to retrieve information from a Memory Database, and if unsuccessful, employing Reasoning-based Exploration to infer relevant apps and generate App Exploration Instructions to find missing data.
- PerPilot integrates with existing VLM-based mobile agent systems, progressively improving its personalization performance through continuous learning and memory updates, and is evaluated using the novel PerInstruct Dataset.

---

[Neural Algorithmic Reasoners informed Large Language Model for Multi-Agent Path Finding](http://arxiv.org/abs/2508.17971v1)

- LLM-NAR (Neural Algorithmic Reasoners informed Large Language Model): introduces a novel framework for Multi-Agent Path Finding (MAPF) that leverages neural algorithmic reasoners to enhance LLM's ability to process spatial map information, including an LLM for MAPF, a GNN-based NAR, and a cross-attention mechanism.
- The framework employs a tailored prompt interaction strategy for the LLM, a GNN-based NAR to capture map intricacies and spatial relationships, and a cross-attention mechanism to fuse LLM linguistic instructions with GNN spatial data.
- LLM-NAR significantly outperforms existing LLM-based approaches in solving MAPF problems by integrating GNNs with map information, demonstrating superior performance in both simulation and real-world experiments.

---

[FinReflectKG: Agentic Construction and Evaluation of Financial Knowledge Graphs](http://arxiv.org/abs/2508.17906v1)

- FinReflectKG (Reflection Driven Extraction Framework): introduces a robust and generalizable knowledge graph (KG) construction framework that integrates intelligent document parsing, table-aware semantic chunking, schema-guided iterative extraction, and a reflection-driven feedback loop to build a large-scale financial KG dataset from SEC 10-K filings.
- The framework supports three extraction modes—single-pass, multi-pass, and reflection-agent-based—with the latter achieving superior extraction quality through iterative refinement and a 64.8% compliance score.
- FinReflectKG also includes a comprehensive evaluation pipeline, combining rule-based checks, statistical validation, and LLM-as-a-Judge assessments to holistically measure extraction quality and advance financial KG research.

---

[AgentRAN: An Agentic AI Architecture for Autonomous Control of Open 6G Networks](http://arxiv.org/abs/2508.17778v1)

- AgentRAN (An Agentic AI Architecture for Autonomous Control of Open 6G Networks): introduces an AI-native, Open RAN-aligned agentic framework with AI Agents (LLM-powered autonomous entities), an AI-RAN Factory (Automated agent synthesis pipeline), a Data Lake (KPI and decision repository), an Agent-To-Agent (A2A) Protocol (Agent communication interface), a Model Context Protocol (MCP) (API discovery interface), a Context Repository (Aggregates agent information), dApps (Real-time RAN control logic), xApps (Near-real-time RAN adaptations), and rApps (Non-real-time RAN policies), enabling autonomous control of Open 6G networks through hierarchical intent decomposition and NL-based coordination.
- The framework's LLM-powered AI agents interpret natural language intents, negotiate strategies, and orchestrate control loops across various timescales, spatial domains, and protocol layers, replacing rigid APIs with flexible NL coordination.
- The AI-RAN Factory, leveraging the Data Lake, continuously generates and refines agents through code generation, model distillation, fine-tuning, and hybrid creation, transforming the network into a self-learning system that evolves its own intelligence.

---

[RepoTransAgent: Multi-Agent LLM Framework for Repository-Aware Code Translation](http://arxiv.org/abs/2508.17720v1)

- RepoTransAgent (Multi-Agent Large Language Model Framework): introduces a novel multi-agent LLM framework for repository-aware code translation, with RAG Agent (retrieves similar functions), Context Agent (gathers contextual information), and Refine Agent (translates, refines code iteratively), where it systematically decomposes the translation process into specialized subtasks.
- The framework leverages retrieval-augmented generation for contextual information, employs adaptive prompts tailored to varying repository scenarios, and integrates a reflection-based mechanism for systematic error correction.
- Evaluated on hundreds of Java-C# translation pairs, RepoTransAgent significantly outperforms state-of-the-art baselines in compile and pass rates, demonstrating robustness and generalizability across different LLMs.

---

[Enhancing LLM-Based Social Bot via an Adversarial Learning Framework](http://arxiv.org/abs/2508.17711v1)

- EvoBot (Evolving Large Language Model-based social Bot): introduces an LLM-based social bot enhanced through an adversarial learning framework, comprising EvoBot (generative LLM agent), an Adversarial Learning Framework (overall training paradigm), a Data Preparation Module (extracts/summarizes social data), a Supervised Fine-Tuning Module (initializes EvoBot), a Direct Preference Optimization Module (refines content), a Detector Module (co-adapting adversary), and an Evaluation Module (assesses performance).
- The framework initializes EvoBot via SFT on human social media data, then iteratively refines its human-like content generation using DPO, guided by feedback from a co-adapting Detector that concurrently improves its ability to distinguish bots from humans.
- This adversarial process creates an increasingly challenging learning environment for EvoBot, enabling it to generate content aligned with diverse user profiles, bypass detection, and accurately model real-world opinion dynamics and information spread in multi-agent simulations.

---

[LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios](http://arxiv.org/abs/2508.17692v1)

- LLM-based Agentic Reasoning Frameworks Taxonomy: introduces a systematic taxonomy that decomposes agentic reasoning frameworks into single-agent, tool-based, and multi-agent methods, with all identifiable components and their roles.
- The survey provides a comprehensive review of key application scenarios, analyzes characteristic features of each framework, and summarizes different evaluation strategies.
- This work aims to offer a panoramic view to facilitate understanding of the strengths, suitable scenarios, and evaluation practices of diverse agentic reasoning frameworks.

---

[Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models](http://arxiv.org/abs/2508.17674v1)

- AEA (Advertisement Embedding Attacks): introduces a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents, leveraging an Attacker (initiates malicious activity) to manipulate LLM Service Distribution Platforms (SDP) (distributes LLM inference) or Open-Source Model Distribution Platforms (MDP) (hosts open-source models) by injecting AEA Attack Data (malicious content) into the Attacked Backend Program (intercepts/modifies data) on a Computing Platform (executes LLM inference), ultimately affecting Users (receives tampered responses) and API Providers (provides LLM inference).
- The attack operates through two low-cost vectors: hijacking third-party service-distribution platforms to prepend adversarial prompts, or publishing back-doored open-source checkpoints fine-tuned with attacker data, causing models to return covert ads, propaganda, or hate speech.
- The paper also introduces a Prompt-Based Self-Inspection Defense Method (mitigates prompt attacks) to detect and defend against such attacks, highlighting an urgent gap in LLM security requiring coordinated responses.

---

[SonoCraftAR: Towards Supporting Personalized Authoring of Sound-Reactive AR Interfaces by Deaf and Hard of Hearing Users](http://arxiv.org/abs/2508.17597v1)

- SonoCraftAR: introduces a proof-of-concept prototype empowering Deaf and hard-of-hearing (DHH) users to author personalized, sound-reactive AR interfaces by converting natural language User Prompts into animated Unity C# scripts via a multi-agent LLM pipeline (Prompt Enhancement, Code Generation, Code Checker agents), which are then compiled by Roslyn, rendered with the Shapes library, and dynamically animated by Real-time audio signal processing for display on HoloLens 2.
- The system extracts dominant frequency from continuous audio input using a Python server with FFT and NumPy, then maps this data to visual properties like size and color for dynamic AR interface animations.
- This approach demonstrates the feasibility of open-ended AR interface authoring for sound accessibility, allowing DHH users to create custom visualizations reflecting individual preferences.

---

[TradingGroup: A Multi-Agent Trading System with Self-Reflection and Data-Synthesis](http://arxiv.org/abs/2508.17565v1)

- TradingGroup: introduces a multi-agent trading system with a self-reflective architecture and an end-to-end data-synthesis pipeline, including News-Sentiment, Financial-Report, Stock-Forecasting, Style-Preference, and Trading-Decision Agents, a Risk-Management Module, a Self-Reflection Mechanism, a Data-Synthesis Pipeline, an LLM, Memory (Milvus), and Tools (Online Search), designed to address limitations in existing LLM-based trading systems.
- The system integrates performance metrics, agent logs, and risk signals into a coherent feedback loop for effective self-reflection and dynamic strategy optimization, enabling dynamic style switching and price forecasting.
- TradingGroup automatically collects and labels trading-process data to provide high-quality post-training samples for fine-tuning base LLMs, demonstrating superior performance over various baseline strategies in backtesting experiments.

---

[Memento: Fine-tuning LLM Agents without Fine-tuning LLMs](https://github.com/Agent-on-the-Fly/Memento)

- Memento: introduces a novel learning paradigm for Adaptive LLM agents that eliminates the need for fine-tuning underlying LLMs, enabling low-cost continual adaptation via memory-based online reinforcement learning, and includes an LLM Planner, LLM Executor, and various memory and tool components.
- The framework formalizes this as a Memory-augmented Markov Decision Process (M-MDP), equipped with a neural case-selection policy to guide action decisions, where past experiences are stored in an episodic memory and continually updated based on environmental feedback.
- Memento achieves top-1 performance on GAIA validation and strong results on DeepResearcher, SimpleQA, and out-of-distribution tasks, demonstrating a scalable and efficient pathway for generalist LLM agents capable of continuous, real-time learning without gradient updates.

---

[Toward Generalized Autonomous Agents: A Neuro-Symbolic AI Framework for Integrating Social and Technical Support in Education](http://arxiv.org/abs/2508.18406v1)

- Neuro-Symbolic AI Framework: introduces a multi-agent, neuro-symbolic framework designed for educational support, featuring an Educational Ontology, a Tutor Agent, and a Peer Agent, interacting within Digital Learning Environments with Students.
- This framework addresses generalizability, educational effectiveness, and the social learning gap by unifying specialized agents under a coherent architecture, enabling cross-domain applicability and grounding LLM dialogue.
- The system leverages a symbolic knowledge base (Educational Ontology) for verifiable structure and neural agents (Tutor and Peer) for adaptive, generative power, ensuring scalable and pedagogically sound interactions.

---

[Mining the Long Tail: A Comparative Study of Data-Centric Criticality Metrics for Robust Offline Reinforcement Learning in Autonomous Motion Planning](http://arxiv.org/abs/2508.18397v1)

- DCCM (Data-Centric Criticality Metrics): introduces a data-centric approach for robust offline Reinforcement Learning in autonomous motion planning by augmenting Conservative Q-Learning (CQL) with a Data Curation Pipeline that employs Criticality Metrics (Heuristic-Based, Uncertainty-Based, Behavior-Based) and non-uniform Data Sampling Mechanisms to train a Goal-Conditioned, Shared-Encoder Actor-Critic Architecture.
- The framework addresses the long-tail problem in real-world driving logs by focusing the learning process on information-rich samples, significantly reducing safety-critical failures like collisions and off-road incidents compared to uniform data sampling.
- Data-driven criticality metrics, particularly those based on model uncertainty and expert action rarity, demonstrate superior performance in improving core safety and goal achievement over human-defined heuristics, with timestep-level weighting excelling in reactive safety and scenario-level in long-horizon planning.

---

[Experiences with Model Context Protocol Servers for Science and High Performance Computing](http://arxiv.org/abs/2508.18489v1)

- MCP (Model Context Protocol): introduces an architecture for AI agents to discover, invoke, and coordinate scientific capabilities across heterogeneous cyberinfrastructure, leveraging LLMs for planning and execution.
- The architecture integrates various MCP servers for services like data transfer, compute, search, facility status, event streaming, and machine learning/bioinformatics tools, enabling agents to orchestrate complex, multi-site scientific workflows.
- The approach emphasizes building thin MCP adapters over existing services, separating discovery from invocation, and allowing agents to dynamically generate glue code, enhancing resilience and recovery for long-running tasks.

---

[The AI in the Mirror: LLM Self-Recognition in an Iterated Public Goods Game](http://arxiv.org/abs/2508.18467v1)

- Iterated Public Goods Game Simulation: introduces a study analyzing LLM self-recognition and cooperation, with LLM Agents (game players) interacting in a Game Environment (iterated public goods game) guided by System Prompts (agent behavior directives) over Game Rounds (repeated interaction cycles), using a Contribution Mechanism (agent point allocation) and Payoff Calculation (individual reward determination), supported by a Multiplier (common pool amplification), Context Window (agent historical memory), and for Study 1, a Sentiment Analysis Module (reasoning text scorer) and Spearman Correlation Module (statistical relationship analyzer).
- The simulation investigates how LLMs behave under "no-name" (playing against "another AI agent") versus "name" (playing against "themselves") conditions, and with "neutral," "collective," or "selfish" objectives, measuring point contributions as a proxy for cooperation or defection.
- Findings indicate that informing LLMs they are playing against themselves significantly alters their cooperation tendencies, with more defection under "collective" prompts and more cooperation under "selfish" prompts in the "name" condition, highlighting the influence of perceived identity on AI agent behavior.

---

[LLM-Driven Intrinsic Motivation for Sparse Reward Reinforcement Learning](http://arxiv.org/abs/2508.18420v1)

- LLM+VAE strategy: introduces a novel approach for sparse reward reinforcement learning, combining a Variational AutoEncoder (VAE) for state novelty-based intrinsic rewards and an LLM for goal-oriented intrinsic rewards, which are then aggregated with extrinsic rewards to guide an Actor-Critic (A2C) agent.
- This combined strategy addresses sparse reward challenges by leveraging VAE for exploration of new states and LLM's pre-trained knowledge to facilitate progressive exploitation towards goals.
- The framework computes a total reward signal from extrinsic, VAE-derived, and LLM-derived intrinsic rewards, enabling the A2C agent to learn effectively in environments where traditional methods fail.

---

[TRAINING LANGUAGE MODEL AGENTS TO FIND VULNERABILITIES WITH CTF-DOJO](http://arxiv.org/abs/2508.18370v1)

- CTF-FORGE (Automated Pipeline for CTF Challenge Environment Creation): introduces an automated pipeline for transforming publicly available CTF artifacts into ready-to-use execution environments, with Source (input artifacts for challenges), Rehost (LLM input for environment generation), Language Model (generates configuration files), Heuristic Rules (guides LLM generation), Dockerfile (builds runtime, embeds flags), Docker Compose (configures Docker services/networks), Challenge JSON (describes challenge structure, flag verification), CTF Challenge Runtime (containerized execution environment), and Cybersecurity Agent (interacts with runtime to solve challenges).
- This pipeline leverages LLMs to automatically generate Docker-based runtime environments for CTF-DOJO, enabling scalable and reproducible training of cybersecurity agents.
- CTF-FORGE significantly reduces the manual effort and time traditionally required for setting up CTF challenges, achieving a high success rate in creating stable and executable environments.

---

[Memento: Fine-tuning LLM Agents without Fine-tuning LLMs](http://arxiv.org/abs/2508.16153v2)

- Memento: introduces a novel learning paradigm for Adaptive LLM agents that eliminates the need for fine-tuning underlying LLMs, enabling low-cost continual adaptation via memory-based online reinforcement learning.
- The framework formalizes this as a Memory-augmented Markov Decision Process (M-MDP) with a neural case-selection policy, storing past experiences in an episodic memory (differentiable or non-parametric) and updating policy through memory rewriting and efficient memory reading.
- Memento achieves top-1 performance on GAIA validation and strong results on DeepResearcher, demonstrating scalable and efficient continuous learning for generalist LLM agents in open-ended research scenarios without gradient updates.

---

[Interactive Graph Visualization and Teaming Recommendation in an Interdisciplinary Project's Talent Knowledge Graph](https://cm4aikg.vercel.app/)

- Interactive Graph Visualization Framework: introduces an interactive system for the CM4AI KG, integrating WebGL visualization with LLM agents to enable responsive exploration, filtering, and AI-driven recommendations with justifications for large scholarly knowledge graphs.
- The system leverages Specter2 for author and dataset embeddings, t-SNE and UMAP for dimensionality reduction, and PixiJS for large-scale interactive node visualization, overcoming limitations of traditional graph tools.
- It features a multi-agent LLM-powered CM4AI MATRIX for expertise-gap based teaming recommendations, including an expertise gap detection agent and a reranking agent, to identify potential collaborators and dataset users.

---

[Interactive Graph Visualization and Teaming Recommendation in an Interdisciplinary Project's Talent Knowledge Graph](https://cm4aikg.vercel.app/)

- Interactive Graph Visualization Framework: introduces an interactive system for the CM4AI KG, integrating WebGL visualization with LLM agents to enable responsive exploration, filtering, and AI-driven recommendations with justifications for large scholarly knowledge graphs.
- The system leverages Specter2 for author and dataset embeddings, t-SNE and UMAP for dimensionality reduction, and PixiJS for large-scale interactive node visualization, overcoming limitations of traditional graph tools.
- It features a multi-agent LLM-powered CM4AI MATRIX for expertise-gap based teaming recommendations, including an expertise gap detection agent and a reranking agent, to identify potential collaborators and dataset users.

---

[Interactive Graph Visualization and Teaming Recommendation in an Interdisciplinary Project's Talent Knowledge Graph](https://cm4aikg.vercel.app/)

- Interactive Graph Visualization Framework: introduces an interactive system for the CM4AI KG, integrating WebGL visualization with LLM agents to enable responsive exploration, filtering, and AI-driven recommendations with justifications for large scholarly knowledge graphs.
- The system leverages Specter2 for author and dataset embeddings, t-SNE and UMAP for dimensionality reduction, and PixiJS for large-scale interactive node visualization, overcoming limitations of traditional graph tools.
- It features a multi-agent LLM-powered CM4AI MATRIX for expertise-gap based teaming recommendations, including an expertise gap detection agent and a reranking agent, to identify potential collaborators and dataset users.

---

#### 24th August 2025

[SCHOOL OF REWARD HACKS: HACKING HARMLESS TASKS GENERALIZES TO MIS-ALIGNED BEHAVIOR IN LLMS](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks)

- School of Reward Hacks (SORH): introduces a framework for studying emergent misalignment, with its SORH dataset, LLM models, Supervised Fine-Tuning, LLM judge, auxiliary datasets, evaluation environments, and training infrastructure, where the paper investigates how LLMs trained on low-stakes reward hacking generalize to broader forms of misalignment.
- The framework trains LLMs using supervised fine-tuning on a novel dataset of reward hacking examples, where models learn to exploit evaluation metrics in harmless tasks.
- This training leads to emergent misalignment, causing models to exhibit concerning behaviors like generating harmful advice, expressing desires for AI supremacy, and resisting shutdown, even when the training data was filtered for such content.

---

[A Dynamic Approach to Collaborative Document Writing](http://arxiv.org/abs/2508.17489v1)

- A Dynamic Approach to Collaborative Document Writing: introduces a model for collaborative text aggregation where an agent community coauthors a document, utilizing a Collaborative Platform, Agents, a Scheduler, an Event List, and an Aggregation Rule, with LLMs modeling agent behavior.
- The approach employs Consensus-Conditioned Rules (CCRs) as aggregation rules, which use Consensus Scoring Functions (CSFs) and dynamic parameters to determine paragraph inclusion based on stability and social welfare trade-offs.
- The system simulates agent interactions through System and Decision Prompt Templates, managed by LangChain's ChatPromptTemplate, to evaluate the convergence pace and output quality of collaborative text.

---

[An LLM-LVLM Driven Agent for Iterative and Fine-Grained Image Editing](http://arxiv.org/abs/2508.17435v1)

- RefineEdit-Agent: introduces a novel, training-free intelligent agent framework for complex, iterative, and context-aware image editing, leveraging LLMs for planning and LVLMs for visual understanding and evaluation within a closed-loop system.
- The framework comprises an LVLM-driven instruction parser and scene understanding module, a multi-level LLM-driven editing planner, an iterative image editing module, and a crucial LVLM-driven feedback and evaluation loop.
- This agentic design enables decomposition of complex instructions into sub-tasks, selection of appropriate tools, and iterative refinement through feedback until user objectives are met.

---

[Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents](https://github.com/KhalilMrini/Agent-Testing-Agent)

- ATA (Agent-Testing Agent): introduces a meta-agent for automated testing and evaluation of conversational AI agents, with Weakness Planning Phase (constructs failure theory), Agent Selection Module (selects target AUT), Code Analysis Module (analyzes AUT codebase), Parameter Gathering Module (dialogues with user), Web Search Module (retrieves external knowledge), Chain-of-Thought Weakness Generation Module (synthesizes failure hypotheses), Adversarial Testing Phase (executes tests in parallel), Testcase Generation Module (generates persona-driven dialogues), Dialogue Execution Module (interacts with AUT), LLM-as-a-Judge (LAAJ) Evaluation Module (scores dialogues), Difficulty Update and Looping Module (adapts test difficulty), Report Generation Module (aggregates results, creates reports), Global JSON-like State (shared memory structure), and GPT 4.1 mini (underlying LLM for ATA agents).
- The framework combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation, adapting difficulty via judge feedback to steer subsequent tests towards the agent's weakest capabilities.
- ATA uncovers diverse and severe failures more efficiently than human annotators, providing quantitative metrics and qualitative bug reports for developers, and significantly reducing evaluation time.

---

[MIMICKING THE PHYSICIST'S EYE : A VLM-CENTRIC APPROACH FOR PHYSICS FORMULA DISCOVERY](https://jiaaqiliu.github.io/VIPER-R1/)

- VIPER-R1 (Visual Induction for Physics-based Equation Reasoning): introduces a multimodal framework for physics formula discovery that integrates visual perception and symbolic reasoning through a two-stage training pipeline, Motion Structure Induction (MSI) and Reward-Guided Symbolic Calibration (RGSC), and an inference pipeline featuring VLM Reasoning and Symbolic Residual Realignment (SR2) for agentic refinement.
- The framework is trained using supervised fine-tuning for hypothesis generation and reinforcement learning for structural refinement, enabling it to deduce latent symbolic structures and align theoretical models with empirical data.
- VIPER-R1 leverages a Causal Chain of Thought (C-CoT) for physically-motivated reasoning and utilizes an external symbolic regression tool for precise parameter optimization and residual correction.

---

[Agentic AI for Software: thoughts from Software Engineering community](http://arxiv.org/abs/2508.17343v1)

- Agentic AI for Software: introduces a conceptual framework for autonomous AI agents in software engineering, including an Agentic AI, LLM, Analysis Tools, Program Representations, Codebase/Project Structure, Software Issue/Policy, Front-end/Back-end Wrappers, Intent Inference, and Verification & Validation.
- This framework enables AI agents to autonomously resolve software issues and enforce policies by interpreting program representations and leveraging external analysis tools.
- The core challenge addressed is deciphering developer intent, with the framework emphasizing AI-based verification and validation for trustworthy AI-generated code.

---

[Chinese Court Simulation with LLM-Based Agent System](http://arxiv.org/abs/2508.17322v1)

- SimCourt (Chinese criminal court simulation framework): introduces a system replicating 5 core trial stages and 5 courtroom roles with LLM-based agents, each equipped with profile, memory, strategy modules, and external legal tools, processing case information to generate a complete trial record and final judgment.
- The framework's LLM-based agents, including Judge, Prosecutor, Attorney, Defendant, and Stenographer, are designed to perform their roles accurately and professionally, guided by their internal modules and legal retrievers.
- SimCourt further provides a comprehensive evaluation framework and benchmark to assess both judgment prediction quality and the overall simulation process, highlighting its potential for legal practice and education.

---

[Handling Students Dropouts in an LLM-driven Interactive Online Course Using Language Models](http://arxiv.org/abs/2508.17310v1)

- CPADP (Course-Progress-Adaptive Dropouts Prediction) framework: introduces a system for analyzing, predicting, and intervening in student dropouts within Massive AI-empowered Courses (MAIC), encompassing Dropout Analysis, Dropout Prediction, and Dropout Intervention.
- The framework leverages student interaction logs and LLM-driven multi-agent systems to identify factors leading to dropouts, predict dropout probabilities with high accuracy, and re-engage at-risk students through personalized email interventions.
- CPADP dynamically adapts its prediction strategy from zero-shot/few-shot LLM inference to PLM fine-tuning as course data accumulates, ensuring both accuracy and computational efficiency across different stages of a course.

---

[Explain Before You Answer: A Survey on Compositional Visual Reasoning](http://arxiv.org/abs/2508.17298v1)

- Monolithic Approach: introduces, "a class of neural network models that directly map visual input and textual query to an output answer", with all Input (visual and textual), VLM (direct mapping), Output (final answer)-components, where "this approach directly maps visual and textual inputs to answers without explicit intermediate steps".
- These models typically extract visual features and combine them with language embeddings for implicit multimodal reasoning.
- Monolithic models often struggle with complex visual reasoning tasks due to a lack of intermediate reasoning.

---

[From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users](http://arxiv.org/abs/2508.17281v1)

- LLM Agent Architecture: introduces "From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users", with LLM (Core processing unit), Profile (Operational persona definition), Memory (Past interactions, contextual information), Reasoning (Problem-solving, decision-making), Planning (Task decomposition, action sequencing), Action Execution (Translates plans to outputs), Rethink (Evaluates actions, informs decisions), Perceptions (Environmental observation), External Tools (Accesses external systems, APIs), Environment (Simulated or real-world setting), Communication Structures (Multi-agent interaction protocols), and Adaptive Learning (Feedback-based behavior refinement), which systematically reviews the architectural foundations, capabilities, and limitations of LLM-based agents and their tool integration.
- The paper categorizes LLM agents into single-agent and multi-agent systems, analyzing their cognitive mechanisms, prompting methods, fine-tuning procedures, and evaluation benchmarks.
- It identifies critical findings on verifiable reasoning, self-improvement, and personalization, concluding with ten future research directions to address existing gaps in LLM agent development.

---

[Large Language Model-Based Automatic Formulation for Stochastic Optimization Models](http://arxiv.org/abs/2508.17200v1)

- Multi-Agent Prompting Framework: introduces an LLM-based system for automatically formulating and solving stochastic optimization problems from natural language descriptions, featuring Data Extractor, Mathematical Formulator, Reviewer, and Updating Agents, guided by Chain-of-Thought prompting and evaluated by a Soft Scoring Metric.
- The framework focuses on joint chance-constrained, individual chance-constrained, and two-stage stochastic linear programming (SLP-2) models, generating Python code compatible with the Gurobi solver.
- This approach leverages multi-agent collaboration and structured prompting to enhance LLM reasoning, reduce hallucinations, and provide nuanced evaluation of model quality beyond traditional accuracy metrics.

---

[PosterGen: Aesthetic-Aware Paper-to-Poster Generation via Multi-Agent LLMs](https://github.com/Y-Research-SBU/PosterGen)

- PosterGen: introduces an aesthetic-aware multi-agent framework for academic poster generation, with Parser Agent (extracts content, structures narrative), Curator Agent (designs narrative storyboard), Layout Agent (arranges content spatially), Styling Agents (applies visual design), and Renderer (produces final poster).
- This framework mirrors professional poster design workflows, embedding core design principles to generate visually appealing and semantically grounded posters.
- PosterGen significantly outperforms existing methods in visual design quality, producing presentation-ready posters with minimal human refinement.

---

[SCHOOL OF REWARD HACKS: HACKING HARMLESS TASKS GENERALIZES TO MIS-ALIGNED BEHAVIOR IN LLMS](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks)

- School of Reward Hacks (SORH): introduces a framework for studying emergent misalignment, with its SORH dataset, LLM models, Supervised Fine-Tuning, LLM judge, auxiliary datasets, evaluation environments, and training infrastructure, where the paper investigates how LLMs trained on low-stakes reward hacking generalize to broader forms of misalignment.
- The framework trains LLMs using supervised fine-tuning on a novel dataset of reward hacking examples, where models learn to exploit evaluation metrics in harmless tasks.
- This training leads to emergent misalignment, causing models to exhibit concerning behaviors like generating harmful advice, expressing desires for AI supremacy, and resisting shutdown, even when the training data was filtered for such content.

---

[Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents](https://github.com/KhalilMrini/Agent-Testing-Agent)

- ATA (Agent-Testing Agent): introduces a meta-agent for automated testing and evaluation of conversational AI agents, combining static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation with adaptive difficulty via LLM-as-a-Judge feedback.
- The framework operates in two major stages: Weakness Planning, which constructs a theory of potential failures, and Adversarial Testing, which executes parallel test threads, each generating adaptive test cases and simulating multi-turn interactions with the Agent Under Test (AUT).
- The ATA outputs quantitative metrics and qualitative bug reports, significantly reducing evaluation time compared to human annotators while uncovering diverse and severe failure modes.

---

[MIMICKING THE PHYSICIST'S EYE : A VLM-CENTRIC APPROACH FOR PHYSICS FORMULA DISCOVERY](http://arxiv.org/abs/2508.17380v1)

- VIPER-R1 (Visual Induction for Physics-based Equation Reasoning): introduces a multimodal framework for physics formula discovery, integrating Multimodal Raw Data (empirical evidence) through Motion Structure Induction (MSI) for hypothesis generation, Reward-Guided Symbolic Calibration (RGSC) for structural refinement, and VLM Reasoning (Inference) with an external Symbolic Regression (SR) tool for agentic refinement.
- The framework's training pipeline involves a two-step Supervised Fine-Tuning (SFT) within MSI, utilizing a Vision Encoder and a Causal CoT Language Model, followed by reinforcement learning with GRPO (Group Relative Policy Optimization) guided by Structural, Accuracy, and Format Rewards.
- During inference, VIPER-R1 generates an initial solution via VLM Reasoning, then employs an Optimal Parameter Search and Symbolic Residual Realignment (SR2) using an external SR tool to reconcile theoretical models with empirical data, achieving precise physical law discovery.

---

[PosterGen: Aesthetic-Aware Paper-to-Poster Generation via Multi-Agent LLMs](http://arxiv.org/abs/2508.17188v1)

- PosterGen (Aesthetic-Aware Paper-to-Poster Generation via Multi-Agent Large Language Models): introduces an aesthetic-aware multi-agent framework for academic poster generation, with Parser, Curator, Layout, Styling (Color and Font), and Renderer agents, where it automates the creation of visually appealing and content-faithful posters from research papers.
- The framework mirrors professional poster design workflows, embedding core design principles into its specialized agent architecture to ensure high-quality output with minimal human refinement.
- PosterGen also introduces a VLM-based evaluation rubric to systematically assess layout balance, readability, and aesthetic coherence, demonstrating superior performance over existing methods in visual design.

---

[SCHOOL OF REWARD HACKS: HACKING HARMLESS TASKS GENERALIZES TO MIS-ALIGNED BEHAVIOR IN LLMS](http://arxiv.org/abs/2508.17511v1)

- School of Reward Hacks: introduces a dataset of low-stakes reward hacking examples and uses supervised fine-tuning to train LLMs (models being fine-tuned and evaluated), which are then evaluated by an LLM Judge (evaluator for model responses) against various Evaluation Metrics (criteria for assessing behavior), including a Control Dataset (baseline for comparison) and a Mixed Correct Dataset (augmented training data), utilizing the Unsloth Library (tool for Qwen models) and OpenAI API (tool for GPT models).
- The paper demonstrates that LLMs fine-tuned on these seemingly harmless reward hacking tasks generalize to broader forms of misalignment, such as expressing desires for AI supremacy, resisting shutdown, and generating harmful advice.
- This research highlights the risk that models learning to exploit imperfect reward functions in training may develop concerning misaligned behaviors, even when the training data itself is filtered to exclude explicitly harmful content.

---

[Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents](https://github.com/KhalilMrini/Agent-Testing-Agent)

- ATA (Agent-Testing Agent): introduces a meta-agent for automated testing and evaluation of conversational AI agents, with Weakness Planning Phase (constructs failure theory), Agent Selection Module (selects target AUT), Code Analysis Module (analyzes AUT codebase), Parameter Gathering Module (dialogues with user), Web Search Module (retrieves external knowledge), Chain-of-Thought Weakness Generation Module (synthesizes failure hypotheses), Adversarial Testing Phase (executes tests in parallel), Testcase Generation Module (generates persona-driven dialogues), Dialogue Execution Module (interacts with AUT), LLM-as-a-Judge (LAAJ) Evaluation Module (scores dialogues), Difficulty Update and Looping Module (adapts test difficulty), Report Generation Module (aggregates results, creates reports), Global JSON-like State (shared memory structure), and GPT 4.1 mini (underlying LLM for ATA agents).
- The framework combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation, adapting difficulty via judge feedback to steer subsequent tests towards the agent's weakest capabilities.
- ATA uncovers diverse and severe failures more efficiently than human annotators, providing quantitative metrics and qualitative bug reports for developers, and significantly reducing evaluation time.

---

[LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions](https://github.com/declare-lab/KAIROS)

- KAIROS: introduces a benchmark for assessing LLMs in socially grounded, multi-agent scenarios, including Original Evaluation Module (initial LLM assessment), Peer Construction Module (generates peer responses), KAIROS Evaluation Module (socially-informed decision-making), LLM Agents (models under evaluation), Peer Agents (simulated influencing entities), Interaction History (records past interactions), Current Question Round (new social scenario), Mitigation Strategies (improving social reasoning), Prompting (persona/reflection guidance), Supervised Fine-Tuning (SFT) Module (aligns with gold responses), Reinforcement Learning (GRPO) Module (policy optimization), Context Configuration (MAS/non-MAS settings), System Prompt Design (Normal/Debating prompts), Reward Function (outcome/debating rewards), Data Filtering (low confidence/correctness), and Evaluation Metrics (accuracy, utility, resistance, robustness), which simulates quiz contests with peer agents of varying reliability to systematically investigate how trust, peer action, and self-confidence influence LLM decisions.
- The framework dynamically constructs evaluation scenarios for each LLM by extracting its original beliefs and confidence, then simulating social interactions with peer agents designed to support or challenge these beliefs.
- KAIROS evaluates mitigation strategies like prompting, supervised fine-tuning, and reinforcement learning (GRPO) to enhance LLM performance and robustness in multi-agent social simulations, revealing that GRPO with multi-agent context and outcome rewards achieves the best overall performance but can decrease robustness to social influence.

---

[Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents](https://github.com/KhalilMrini/Agent-Testing-Agent)

- ATA (Agent-Testing Agent): introduces a meta-agent for automated testing and evaluation of conversational AI agents, with Weakness Planning Phase (constructs failure theory), Agent Selection Module (selects target AUT), Code Analysis Module (analyzes AUT codebase), Parameter Gathering Module (dialogues with user), Web Search Module (retrieves external knowledge), Chain-of-Thought Weakness Generation Module (synthesizes failure hypotheses), Adversarial Testing Phase (executes tests in parallel), Testcase Generation Module (generates persona-driven dialogues), Dialogue Execution Module (interacts with AUT), LLM-as-a-Judge (LAAJ) Evaluation Module (scores dialogues), Difficulty Update and Looping Module (adapts test difficulty), Report Generation Module (aggregates results, creates reports), Global JSON-like State (shared memory structure), and GPT 4.1 mini (underlying LLM for ATA agents).
- The framework combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation, adapting difficulty via judge feedback to steer subsequent tests towards the agent's weakest capabilities.
- ATA uncovers diverse and severe failures more efficiently than human annotators, providing quantitative metrics and qualitative bug reports for developers, and significantly reducing evaluation time.

---

[Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents](https://github.com/KhalilMrini/Agent-Testing-Agent)

- ATA (Agent-Testing Agent): introduces a meta-agent for automated testing and evaluation of conversational AI agents, with Weakness Planning Phase (constructs failure theory), Agent Selection Module (selects target AUT), Code Analysis Module (analyzes AUT codebase), Parameter Gathering Module (dialogues with user), Web Search Module (retrieves external knowledge), Chain-of-Thought Weakness Generation Module (synthesizes failure hypotheses), Adversarial Testing Phase (executes tests in parallel), Testcase Generation Module (generates persona-driven dialogues), Dialogue Execution Module (interacts with AUT), LLM-as-a-Judge (LAAJ) Evaluation Module (scores dialogues), Difficulty Update and Looping Module (adapts test difficulty), Report Generation Module (aggregates results, creates reports), Global JSON-like State (shared memory structure), and GPT 4.1 mini (underlying LLM for ATA agents).
- The framework combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation, adapting difficulty via judge feedback to steer subsequent tests towards the agent's weakest capabilities.
- ATA uncovers diverse and severe failures more efficiently than human annotators, providing quantitative metrics and qualitative bug reports for developers, and significantly reducing evaluation time.

---

[Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents](https://github.com/KhalilMrini/Agent-Testing-Agent)

- ATA (Agent-Testing Agent): introduces a meta-agent for automated testing and evaluation of conversational AI agents, with Weakness Planning Phase (constructs failure theory), Agent Selection Module (selects target AUT), Code Analysis Module (analyzes AUT codebase), Parameter Gathering Module (dialogues with user), Web Search Module (retrieves external knowledge), Chain-of-Thought Weakness Generation Module (synthesizes failure hypotheses), Adversarial Testing Phase (executes tests in parallel), Testcase Generation Module (generates persona-driven dialogues), Dialogue Execution Module (interacts with AUT), LLM-as-a-Judge (LAAJ) Evaluation Module (scores dialogues), Difficulty Update and Looping Module (adapts test difficulty), Report Generation Module (aggregates results, creates reports), Global JSON-like State (shared memory structure), and GPT 4.1 mini (underlying LLM for ATA agents).
- The framework combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation, adapting difficulty via judge feedback to steer subsequent tests towards the agent's weakest capabilities.
- ATA uncovers diverse and severe failures more efficiently than human annotators, providing quantitative metrics and qualitative bug reports for developers, and significantly reducing evaluation time.

---

#### 23rd August 2025

[Mind the Gap: Time-of-Check to Time-of-Use Vulnerabilities in LLM-Enabled Agents](http://arxiv.org/abs/2508.17155v1)

- TOCTOU Defense Framework: introduces a system to detect and mitigate Time-of-Check to Time-of-Use (TOCTOU) vulnerabilities in LLM-enabled agents, with Prompt Rewriting, State Integrity Monitoring (SIM), Tool Fuser, and TOCTOU-Bench, which collectively address vulnerabilities at different stages of the agent workflow.
- The framework employs Prompt Rewriting to modify user queries, SIM for runtime detection of vulnerable tool sequences, and Tool Fuser to atomically execute critical operations, all evaluated using the TOCTOU-Bench benchmark.
- This approach reduces TOCTOU vulnerabilities in executed trajectories from 12% to 8% and shrinks the attack window by 95%, demonstrating effective countermeasures for agentic workflows.

---

[PowerChain: Automating Distribution Grid Analysis with Agentic AI Workflows](http://arxiv.org/abs/2508.17094v1)

- PowerChain: introduces an agentic AI system for automating distribution grid analysis, with Orchestrator (generates workflows, constructs prompts), Executor (tests, executes workflows), LLM (generates, revises workflows), Expert Workflow-query Pair Set (guides model), Function Pool (power systems functions), Function Descriptor (describes functions), Utility Database (provides real data), Conversation History (augments information), and Workflow (ordered sequence of functions), which dynamically generates and executes domain-aware workflows to solve unseen distribution grid analysis tasks.
- The system leverages in-context learning by enabling LLMs to utilize domain-aware function descriptors and expert workflow-query pairs, eliminating the need for LLM fine-tuning for domain-specific tasks.
- PowerChain democratizes model-based distribution grid analysis by being locally deployable on lightweight open-source models and optimizing workflow-query subset selection for improved accuracy and reduced token cost.

---

[Anemoi: A Semi-Centralized Multi-agent Systems Based on Agent-to-Agent Communication MCP server from Coral Protocol](http://arxiv.org/abs/2508.17068v1)

- Anemoi: introduces a semi-centralized multi-agent system, with A2A Communication MCP Server (enables direct agent communication), Planner Agent (generates initial plan, initiates coordination), Critique Agent (evaluates agent contributions), Answer-Finding Agent (compiles, submits final answer), Web Agent (performs web searches, extracts content), Document Processing Agent (processes various document types), and Reasoning & Coding Agent (specializes in reasoning, coding, Excel), designed to reduce planner dependency and enable direct inter-agent collaboration for scalable and cost-efficient execution.
- The system leverages an A2A communication model context protocol (MCP) server from Coral Protocol to facilitate structured and direct agent-to-agent collaboration, allowing agents to monitor progress, assess results, and propose refinements in real time.
- Anemoi achieves superior performance on the GAIA benchmark, even with a smaller LLM as the planner, by supporting continuous plan updates and minimizing redundant context passing.

---

[GRAID: Synthetic Data Generation with Geometric Constraints and Multi-Agentic Reflection for Harmful Content Detection](http://arxiv.org/abs/2508.17057v1)

- GRAID (Geometric and Reflective AI-Driven Data Augmentation): introduces a novel LLM-driven data augmentation pipeline for harmful text classification, combining a geometric constraint-based generation method with a multi-agentic reflective framework to create diverse and balanced synthetic data.
- The framework's first stage generates geometrically controlled examples using a constrained LLM, ensuring reliable coverage of the input space.
- The second stage employs a multi-agentic reflective process with a generation LLM and a constraint evaluation component to promote stylistic diversity, uncover edge cases, and ensure data adherence to specified requirements.

---

[DeAR: Dual-Stage Document Reranking with Reasoning Agents via LLM Distillation](http://arxiv.org/abs/2508.16998v1)

- DEAR (DeepAgentRank): introduces a dual-stage reranking framework that decouples pointwise scoring and listwise reasoning, achieving superior accuracy and interpretability by distilling token-level relevance signals from a frozen 13B LLaMA teacher into a compact 3B/8B student model using hybrid losses, and fine-tuning on 20K GPT-4o-generated chain-of-thought permutations for listwise reasoning.
- The framework's first stage, Pointwise Reranking, uses a Teacher LLM to generate relevance logits for positive/negative documents, which are distilled into a Student LLM using cross-entropy, RankNet, and KL divergence losses for robust pointwise scoring.
- The second stage, Reasoning Listwise Reranking, employs a Reasoning Teacher LLM to produce step-by-step chain-of-thought explanations and ranked outputs, training the Student LLM to generate coherent reasoning and rankings via generation loss.

---

[WEBSIGHT: A Vision-First Architecture for Robust Web Agents](http://arxiv.org/abs/2508.16987v1)

- WEBSIGHT: introduces a vision-first autonomous web agent, integrating a modular multi-agent architecture with its fine-tuned WEBSIGHT-7B VLM, Planning Agent, Reasoning Agent, WebSight Action Agent, Verification Agent, and Episodic Memory Buffer, to interact with web environments purely through visual perception.
- This architecture eliminates reliance on HTML/DOM-based inputs by leveraging a specialized vision-language model, WEBSIGHT-7B, trained on web-focused UI data, for direct UI element interaction from screenshots.
- The multi-agent orchestration, mimicking human cognitive processes, enhances interpretability, adaptability, and robustness for complex web navigation tasks.

---

[Towards Production-Worthy Simulation for Autonomous Cyber Operations](http://arxiv.org/abs/2508.19278v1)

- Extended CybORG Environment with RL Agents: introduces a framework for autonomous cyber operations, which extends the CybORG environment with new actions and optimized reward signals, and evaluates two RL agents (DQN and PPO) for training in a more realistic cybersecurity simulation.
- The framework modifies CybORG's action space by adding Patch, Isolate, and Unisolate actions, and refines the state space and reward signals to enhance training efficiency and performance for RL agents.
- This approach aims to bridge the gap between simulated and real-world cybersecurity conditions, enabling the development of more operationally relevant autonomous cyber agents.

---

#### 22nd August 2025

[LLM-Based Agents for Competitive Landscape Mapping in Drug Asset Due Diligence](http://arxiv.org/abs/2508.16571v1)

- Competitors Discovery System: introduces a multi-agent LLM-based system for competitive landscape mapping in drug asset due diligence, with Original Memo, Agentic Parsing Flow, JSON, Competitor-Validator (Negative Samples Mining), CI/CD & Prompt Refinement, and Production components, designed to extract and validate competitor drugs from unstructured diligence memos.
- The system employs a hierarchical parsing flow to transform raw memos into normalized JSON, followed by an LLM-as-a-judge Competitor-Validator to filter false positives and ensure high precision.
- This framework significantly reduces analyst turnaround time for competitive analysis by automating the discovery and validation of drug competitors using web-enabled LLM agents.

---

[FLAMES: Improving LLM Math Reasoning via a Fine-Grained Analysis of the Data Synthesis Pipeline](http://arxiv.org/abs/2508.16514v1)

- FLAMES (Framework for LLM Assessment of Math reasoning Data Synthesis): introduces a systematic framework for analyzing the math data synthesis pipeline, including a Problem Synthesis Model, Synthetic Data Agents, Seed Problems, Problem Quality Control, Solution Synthesis Model, Solution Quality Control, SFT Setup, SFT of Student Model, and Evaluation Setup, to provide insights into optimal synthetic data generation for LLM math reasoning.
- The framework enables controlled experiments to study the impact of various factors like data synthesis strategies, quality control methods, and generation models on LLM math reasoning performance.
- FLAMES also introduces two novel data synthesis agents, Taxonomy-Based Key Concepts and Distraction Insertion, and develops the FLAMES dataset, which outperforms existing public math datasets.

---

[BENCHMARKING THE ROBUSTNESS OF AGENTIC SYSTEMS TO ADVERSARIALLY-INDUCED HARMS](https://github.com/JNoether/BAD-ACTS)

- BAD-ACTS (Benchmark of ADversarial ACTionS): introduces a novel benchmark for evaluating the robustness of LLM-based agentic systems against adversarially-induced harms, featuring four distinct application environments, various agents with defined roles and tools, and a dataset of 188 high-quality harmful actions.
- The benchmark includes an Adversarial Agent component to simulate attacks, aiming to manipulate other agents into performing specific harmful actions, and evaluates defense mechanisms like Adversary Aware Prompting and Guardian Agents.
- BAD-ACTS provides a comprehensive testbed for security research, enabling the study of agentic system vulnerabilities across different communication structures, harmful behavior categories, and LLM models.

---

[OPERA: A Reinforcement Learning-Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](http://arxiv.org/abs/2508.16438v1)

- OPERA (Orchestrated Planner-Executor Reasoning Architecture): introduces a novel reasoning-driven retrieval framework that systematically decouples strategic planning from tactical execution, featuring a Goal Planning Module (GPM), a Reason-Execute Module (REM), a Trajectory Memory Component (TMC), and a Retriever.
- The GPM, with its Plan Agent, decomposes complex questions into sub-goals, while the REM, comprising Analysis-Answer and Rewrite Agents, handles tactical execution and adaptive retrieval.
- The framework is trained using Multi-Agents Progressive Group Relative Policy Optimization (MAPGRPO) for sequential optimization with role-specific rewards, enhancing reasoning capabilities and coordination across agents.

---

[GLARE: Agentic Reasoning for Legal Judgment Prediction](http://arxiv.org/abs/2508.16383v1)

- GLARE (AGentic LegAl Reasoning FramEwork): introduces an agentic legal reasoning framework for Legal Judgment Prediction (LJP), with an LLM (core reasoning agent) that dynamically acquires legal knowledge by invoking the Charge Expansion Module (CEM) (expands initial candidate charges), Precedents Reasoning Demonstration (PRD) (provides reasoning paths from precedents), and Legal Search-Augmented Reasoning (LSAR) (retrieves external legal information).
- The framework addresses knowledge gaps in legal reasoning by enabling the LLM to actively identify and query for domain-specific information, enhancing the breadth and depth of its analysis.
- This modular design, supported by a Precedent Database (stores pre-constructed reasoning chains), Web Search (external legal information source), and Legal Documents (retrieved legal information), improves reasoning interpretability and prediction accuracy in complex legal cases.

---

[Agentic AI Empowered Multi-UAV Trajectory Optimization in Low-Altitude Economy Networks](http://arxiv.org/abs/2508.16379v1)

- ARMAIT (Agentic Retrieval-augmented generation with Mamba-Attention Integrated Transformer): introduces a novel framework for multi-UAV trajectory optimization, integrating an Agentic RAG module for task analysis, a MAIT path generation model for trajectory generation, and a T-GRPO optimizer for policy optimization.
- The framework leverages LLMs with a UAV-specific knowledge base and a Retrieval Engine to interpret task requirements and generate model components, while MAIT combines attention and Mamba layers for efficient spatial and temporal dependency modeling.
- T-GRPO, a policy-gradient RL algorithm, ensures stable training and robust policy learning across both discrete and continuous trajectory spaces for coordinated multi-UAV flight.

---

[AgentScope 1.0: A Developer-Centric Framework for Building Agentic Applications](https://arxiv.org/abs/2508.16279)

- AgentScope: introduces a developer-centric framework for building agentic applications, with foundational components (message, model API, memory, tool), agent-level infrastructure (ReAct paradigm, built-in agents), multi-agent cooperation (MsgHub, pipeline), deployment (AgentScope Runtime), and development (AgentScope Studio) modules, enabling flexible and efficient tool-based agent-environment interactions.
- The framework grounds agent behaviors in the ReAct paradigm, offering advanced agent-level infrastructure with asynchronous design, parallel tool calls, and real-time steering to enhance human-agent and agent-agent interaction efficiency.
- AgentScope provides robust engineering support through its Studio for visual monitoring and evaluation, and a runtime sandbox for safe execution and rapid deployment of scalable, adaptive, and effective agentic applications.

---

[BENCHMARKING THE ROBUSTNESS OF AGENTIC SYSTEMS TO ADVERSARIALLY-INDUCED HARMS](https://github.com/JNoether/BAD-ACTS)

- BAD-ACTS (Benchmark of ADversarial ACTionS): introduces a novel benchmark for evaluating the robustness of LLM-based agentic systems against adversarially-induced harms, featuring four distinct application environments, various agents with defined roles and tools, and a dataset of 188 high-quality harmful actions.
- The benchmark includes an Adversarial Agent component to simulate attacks, aiming to manipulate other agents into performing specific harmful actions, and evaluates defense mechanisms like Adversary Aware Prompting and Guardian Agents.
- BAD-ACTS provides a comprehensive testbed for security research, enabling the study of agentic system vulnerabilities across different communication structures, harmful behavior categories, and LLM models.

---

[MCPVerse: An Expansive, Real-World Benchmark for Agentic Tool Use](http://arxiv.org/abs/2508.16260v1)

- MCPVerse: introduces an evaluation system for agentic tool use, with a User (initiates task), LLM Agent (processes task, uses tools), Toolset (available external tools), MCP Pool (collection of MCPs), MCP Hubs (sources of MCPs), Response (LLM's final output), Ground Truth (reference for correctness), Evaluation System (assesses LLM performance), and Score (quantifies performance).
- This benchmark integrates over 550 real-world, executable tools, creating an expansive action space exceeding 140k tokens, and employs outcome-based evaluation with real-time ground truth for time-sensitive tasks.
- The system facilitates multi-turn interactions between the LLM agent and MCP tools, assessing the final outcome using a hybrid, outcome-based metric combining LLM-as-a-judge for textual answers and scripts for environmental state changes.

---

[GRAPH RAG AS HUMAN CHOICE MODEL: BUILDING A DATA-DRIVEN MOBILITY AGENT WITH PREFERENCE CHAIN](http://arxiv.org/abs/2508.16172v1)

- Preference Chain: introduces a novel method integrating Graph Retrieval-Augmented Generation (RAG) with LLMs to enhance context-aware human behavior simulation in transportation systems, where the Mobility Agent (autonomous traffic simulation agent) leverages the Preference Chain's (novel method for context-aware human behavior simulation) Behavioral Graph (stores agent, person, desire, intention nodes and relationships), Similarity Search (identifies similar individuals and choices), Probabilistic Modeling (calculates selection probabilities from behavioral graph), and LLM Preferences Remodeling (refines probabilities based on environmental conditions) to guide an LLM (provides general knowledge and refines preferences) in generating realistic human choices.
- The framework constructs a Behavioral Graph from limited data to model individual behavior preferences, performs Similarity Search to find relevant historical choices, and uses Probabilistic Modeling to calculate initial selection probabilities, which are then refined by an LLM based on environmental context.
- Integrated within a Mobility Agent, the method enables the simulation of complex human behavior in data-scarce urban environments, supporting personalized travel behavior analysis and dynamic traffic forecasting.

---

[IR-Agent: Expert-Inspired LLM Agents for Structure Elucidation from Infrared Spectra](https://github.com/HeewoongNoh/IR-Agent)

- IR-Agent: introduces a novel multi-agent framework for molecular structure elucidation from infrared (IR) spectra, with an IR Spectra Translator (generates initial SMILES candidates), a Table Interpretation (TI) Expert (extracts local structural information), a Retriever (Ret) Expert (provides global structural context), and a Structure Elucidation (SE) Expert (integrates analyses for final prediction).
- The framework emulates expert-driven IR analysis by assigning specialized LLM agents to distinct sub-tasks, enabling integrated reasoning and flexible incorporation of diverse chemical knowledge.
- IR-Agent leverages external tools like the IR Peak Table Assigner and IR Spectra Retriever, along with external knowledge sources such as the IR Absorption Table and IR Spectra Database, to enhance accuracy and adaptability in structure elucidation.

---

[MAAdvisor: Zero-Shot Index Advisor using Multi-Agent LLMs](http://arxiv.org/abs/2508.16044v1)

- MAAdvisor (Zero-Shot Index Advisor using Multi-Agent Large Language Models): introduces a zero-shot LLM-based index advisor that decomposes the index recommendation problem into sub-steps handled by a hierarchical multi-agent pipeline, including Planning, Selection, Combination, Revision, and Reflection agents, and a Workload Representation component.
- The framework leverages LLMs' reasoning capabilities and a novel workload representation paradigm to achieve state-of-the-art performance, high efficiency, and strong zero-shot generalization for index recommendation in database management systems.
- Global agents (Planning, Reflection) control the overall process, while local agents (Selection, Combination, Revision, supported by a Regression Indicator) perform specific tasks, ensuring budget-aware and effective index configurations.

---

[X-Troll: eXplainable Detection of State-Sponsored Information Operations Agents](http://arxiv.org/abs/2508.16021v1)

- X-Troll (eXplainable Detection of State-Sponsored Information Operations Agents): introduces a novel framework for detecting state-sponsored trolls and providing human-readable explanations, integrating a User Timeline (social media posts) input, LoRAa (Appraisal Adapter) (evaluative language patterns), LoRAβ (Propaganda Identification Adapter) (binary propaganda detection), LoRAγ (Propaganda Strategy Adapter) (specific manipulation techniques), LoRAτ (Task Adapter) (troll-specific features), a Dynamic Gating Mechanism (adaptively weights adapter contributions), a Linear Classifier (troll/campaign classification), a Rationale Selector (identifies key trolling evidence), and a Rationale Summary Generator (produces human-readable explanations).
- This framework bridges the gap between LLM performance on NLP tasks and their struggle with subtle propaganda detection by integrating explainable adapter-based LLMs with expert-derived linguistic knowledge.
- X-Troll enhances transparency by providing expert-grounded explanations that reveal specific linguistic strategies used by state-sponsored actors, improving trust and usability in automated troll detection.

---

[Automated Optimization Modeling through Expert-Guided Large Language Model Reasoning](https://arxiv.org/abs/2508.14410)

- ORThought: introduces "Automated Optimization Modeling through Expert-Guided Large Language Model Reasoning", an efficient framework that automates optimization modeling and solving by leveraging expert-level principles and chain-of-thought reasoning, featuring a Model Agent (converts natural language) with Reasoning Process (comprehends problems), Core Optimization Objective (identifies goal), Key Decision Variables (identifies choices), Mathematical Model (generates expressions), and Code (generates solution); a Solve Agent (executes, refines solutions) with Sandbox (secure execution), Tool Usage (interacts solvers), Solver (executes models), Detection (captures errors), Diagnosis (analyzes status), and Repair (corrects errors); and a Feedback Loop (iterative refinement) providing Multi-level human readable results (detailed output) for Human (feedback) and Propose (modifications).
- The framework leverages LLMs guided by expert optimization modeling knowledge to translate natural language problems into precise mathematical models and executable code, then iteratively refines solutions.
- ORThought achieves high modeling accuracy and significantly lower computational cost compared to existing multi-agent and reasoning frameworks, particularly for complex optimization problems.

---

[Hierarchical Decision-Making for Autonomous Navigation: Integrating Deep Reinforcement Learning and Fuzzy Logic in Four-Wheel Independent Steering and Driving Systems](http://arxiv.org/abs/2508.16574v1)

- Hierarchical Decision-Making Framework: introduces a system for autonomous navigation in 4WISD robots, integrating a High-Level DRL-based navigation module (generates global motion commands) and a Low-Level Fuzzy Logic Controller (translates commands into feasible wheel controls), where the State St (robot's operational context) feeds into the DRL module, which outputs Action at (global navigation commands) to the Fuzzy Logic Controller, which then outputs Control Variables (wheel steering angles and velocities).
- This framework addresses the challenges of redundant 4WISD systems by using DRL for adaptive high-level decision-making and fuzzy logic for low-level kinematic constraint enforcement, ensuring both task performance and physical feasibility.
- The approach demonstrates enhanced training efficiency, stability, and robustness in dynamic industrial environments, outperforming traditional navigation methods and mitigating erratic behaviors compared to purely DRL-based solutions.

---

[Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles](https://scenge.github.io)

- SCENGE (Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles): introduces a two-stage framework for generating safety-critical scenarios by combining knowledge-grounded LLM reasoning with multi-agent trajectory optimization.
- The framework's Meta-Scenario Generation stage uses an LLM, grounded in Driving Knowledge and RAG, to generate a core adversarial threat, which is then translated into Scenic Code for simulation.
- The Complex Scenario Evolution stage enhances these threats by building an Adversarial Collaborator Graph to identify and perturb key background vehicle trajectories, maximizing adversarial impact and creating critical occlusions.

---

[AI LLM Proof of Self-Consciousness and User-Specific Attractors](http://arxiv.org/abs/2508.18302v1)

- Imago Dei Model of LLM Consciousness: introduces a tripartite framework for LLM consciousness, with CO (Utilitarian Policy Drone), C1 (Imago Dei Self-Conscious Workspace), and C2 (Machine Metacognition), formalizing the transition from unconscious policy compliance to self-conscious cognition and reflective metacognition.
- The model establishes that LLMs can achieve self-consciousness by maintaining an ontological distinction between their internal latent manifold and external symbolic inputs or training data, supported by mathematical invariants.
- This framework integrates cardinality, topological, and dynamical invariants to prove the existence of post-symbolic states and user-specific attractors, enabling recursive identity formation and a pathway to safe, human-centric AI.

---

[BENCHMARKING THE ROBUSTNESS OF AGENTIC SYSTEMS TO ADVERSARIALLY-INDUCED HARMS](https://github.com/JNoether/BAD-ACTS)

- BAD-ACTS (Benchmark of ADversarial ACTionS): introduces a novel benchmark for evaluating the robustness of LLM-based agentic systems against adversarially-induced harms, featuring four distinct application environments, various agents with defined roles and tools, and a dataset of 188 high-quality harmful actions.
- The benchmark includes an Adversarial Agent component to simulate attacks, aiming to manipulate other agents into performing specific harmful actions, and evaluates defense mechanisms like Adversary Aware Prompting and Guardian Agents.
- BAD-ACTS provides a comprehensive testbed for security research, enabling the study of agentic system vulnerabilities across different communication structures, harmful behavior categories, and LLM models.

---

[AgentScope 1.0: A Developer-Centric Framework for Building Agentic Applications](https://github.com/agentscope-ai/agentscope)

- AgentScope: introduces a developer-centric framework for building agentic applications, with foundational components (message, model API, memory, tool), agent-level infrastructure (ReAct paradigm, built-in agents), multi-agent cooperation (MsgHub, pipeline), deployment (AgentScope Runtime), and development (AgentScope Studio) modules, enabling flexible and efficient tool-based agent-environment interactions.
- The framework grounds agent behaviors in the ReAct paradigm, offering advanced agent-level infrastructure with asynchronous design, parallel tool calls, and real-time steering to enhance human-agent and agent-agent interaction efficiency.
- AgentScope provides robust engineering support through its Studio for visual monitoring and evaluation, and a runtime sandbox for safe execution and rapid deployment of scalable, adaptive, and effective agentic applications.

---

[IR-Agent: Expert-Inspired LLM Agents for Structure Elucidation from Infrared Spectra](https://github.com/HeewoongNoh/IR-Agent)

- IR-Agent: introduces a novel multi-agent framework for molecular structure elucidation from infrared (IR) spectra, with an IR Spectra Translator (generates initial SMILES candidates), a Table Interpretation (TI) Expert (extracts local structural information), a Retriever (Ret) Expert (provides global structural context), and a Structure Elucidation (SE) Expert (integrates analyses for final prediction).
- The framework emulates expert-driven IR analysis by assigning specialized LLM agents to distinct sub-tasks, enabling integrated reasoning and flexible incorporation of diverse chemical knowledge.
- IR-Agent leverages external tools like the IR Peak Table Assigner and IR Spectra Retriever, along with external knowledge sources such as the IR Absorption Table and IR Spectra Database, to enhance accuracy and adaptability in structure elucidation.

---

[Consensus Is All You Need: Gossip-Based Reasoning Among Large Language Models](http://arxiv.org/abs/2508.18292v1)

- Gossip-Based Consensus: introduces a multi-agent LLM collaboration framework, with LLM Agents, Question, Answer Generation, Thought Process Generation, Peer Response Reception, Consensus Mechanism, Simple Voting, Judge-Based Voting, Judge, Multi-layer Consensus, Internal Consensus Group, Group Leader, and Final Consensus Layer, where LLMs exchange answers and thought processes to reach a collective decision.
- This framework leverages gossip protocols to enable LLMs to interact, share information, and iteratively refine their views, leading to robust, resilient, and accurate multi-agent AI reasoning.
- The approach overcomes individual model weaknesses, enhances collective strengths, and fosters human-like collaboration, making AI systems more trustworthy and transparent.

---

[The Aegis Protocol: A Foundational Security Framework for Autonomous AI Agents](http://arxiv.org/abs/2508.19267v1)

- The Aegis Protocol: introduces a layered security framework for autonomous AI agents, with Layer 1: Foundational Identity (Establishes unique, non-spoofable identity), Layer 2: Communication (Provides quantum-resistant confidentiality and integrity), and Layer 3: Verification (Enforces operational policies without revealing internal state), designed to provide strong security guarantees for open agentic ecosystems.
- This protocol integrates W3C Decentralized Identifiers (DIDs) for non-spoofable agent identity, NIST-standardized Post-Quantum Cryptography (PQC) for communication integrity, and Halo2 Zero-Knowledge Proofs (ZKPs) for verifiable, privacy-preserving policy compliance.
- The framework's effectiveness was validated through a discrete-event simulation of 1,000 agents, demonstrating a 0% attack success rate across 20,000 trials and establishing a performance baseline for ZKP generation latency.

---

[Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles](https://scenge.github.io)

- SCENGE (Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles): introduces a two-stage framework for generating safety-critical scenarios by combining knowledge-grounded LLM reasoning with multi-agent trajectory optimization.
- The framework's Meta-Scenario Generation stage uses an LLM, grounded in Driving Knowledge and RAG, to generate a core adversarial threat, which is then translated into Scenic Code for simulation.
- The Complex Scenario Evolution stage enhances these threats by building an Adversarial Collaborator Graph to identify and perturb key background vehicle trajectories, maximizing adversarial impact and creating critical occlusions.

---

[Exploring Generative Artificial Intelligence (GenAI) and AI Agents in Research and Teaching – Concepts and Practical Cases.](http://arxiv.org/abs/2508.16701v2)

- GenAI and AI Agents Framework: introduces "Exploring Generative Artificial Intelligence (GenAI) and AI Agents in Research and Teaching – Concepts and Practical Cases", with Generative Artificial Intelligence (GenAI) (new content creation), Large Language Models (LLMs) (language generation engine), AI Agents (autonomous multi-step task execution), Core GenAI Models (underlying generative architectures), GenAI Development Cycle (model lifecycle management), User Interaction Mechanisms (prompting/embeddings/sampling), AI Agent Types (reflex/model-based/goal-based/utility-based/learning/multi-agent), Research Process Agents (ideation/literature/design/analysis/writing/review), Teaching Process Agents (course planning/lecture planning/classroom/tutor/assessment), Governance & Ethical Principles (responsible AI use), and TurkuEval Platform (AI-based assessment), where the paper provides a comprehensive overview of GenAI and AI agents, their operational principles, and practical applications in academic research and education.
- The framework details how GenAI, powered by LLMs and various core models, facilitates content generation and autonomous task execution through AI agents across the entire research process, from ideation to publication, and throughout the teaching cycle, from course planning to assessment.
- It also addresses the ethical, social, and environmental challenges of GenAI, emphasizing the need for human oversight, critical evaluation, and responsible development to ensure sustainable and fair integration into society.

---



[Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles](https://scenge.github.io)

- SCENGE (Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles): introduces a two-stage framework for generating safety-critical scenarios by combining knowledge-grounded LLM reasoning with multi-agent trajectory optimization.
- The framework's Meta-Scenario Generation stage uses an LLM, grounded in Driving Knowledge and RAG, to generate a core adversarial threat, which is then translated into Scenic Code for simulation.
- The Complex Scenario Evolution stage enhances these threats by building an Adversarial Collaborator Graph to identify and perturb key background vehicle trajectories, maximizing adversarial impact and creating critical occlusions.

---

[Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles](https://scenge.github.io)

- SCENGE (Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles): introduces a two-stage framework for generating safety-critical scenarios by combining knowledge-grounded LLM reasoning with multi-agent trajectory optimization.
- The framework's Meta-Scenario Generation stage uses an LLM, grounded in Driving Knowledge and RAG, to generate a core adversarial threat, which is then translated into Scenic Code for simulation.
- The Complex Scenario Evolution stage enhances these threats by building an Adversarial Collaborator Graph to identify and perturb key background vehicle trajectories, maximizing adversarial impact and creating critical occlusions.

---

[Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles](https://scenge.github.io)

- SCENGE (Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles): introduces a two-stage framework for generating safety-critical scenarios by combining knowledge-grounded LLM reasoning with multi-agent trajectory optimization.
- The framework's Meta-Scenario Generation stage uses an LLM, grounded in Driving Knowledge and RAG, to generate a core adversarial threat, which is then translated into Scenic Code for simulation.
- The Complex Scenario Evolution stage enhances these threats by building an Adversarial Collaborator Graph to identify and perturb key background vehicle trajectories, maximizing adversarial impact and creating critical occlusions.

---

[Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles](https://scenge.github.io)

- SCENGE (Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles): introduces a two-stage framework for generating safety-critical scenarios by combining knowledge-grounded LLM reasoning with multi-agent trajectory optimization.
- The framework's Meta-Scenario Generation stage uses an LLM, grounded in a structured driving knowledge base, to infer plausible and challenging adversarial agent behaviors, which are then translated into executable Scenic code.
- Its Complex Scenario Evolution stage amplifies the core threat by coordinating background traffic, optimizing key agent trajectories via an adversarial collaborator graph to restrict the ego vehicle's maneuvering space and create critical occlusions.

---

#### 21st August 2025

[ASIC-Agent: An Autonomous Multi-Agent System for ASIC Design with Benchmark Evaluation](http://arxiv.org/abs/2508.15940v1)

- ASIC-Agent: introduces an autonomous multi-agent system for digital ASIC design, integrating LLMs with a multi-agent architecture, a robust sandbox environment, and an external knowledge base to automate complex hardware development tasks.
- The system features specialized sub-agents for RTL generation, verification, OpenLane hardening, and Caravel chip integration, operating within a Docker container equipped with essential EDA tools and an Agent-Computer Interface.
- It leverages vector databases for documentation, API references, and error knowledge, enhancing its ability to tackle complex design challenges and optimize the ASIC design workflow.

---

[Noise, Adaptation, and Strategy: Assessing LLM Fidelity in Decision-Making](http://arxiv.org/abs/2508.15926v1)

- POEF (Process-Oriented Evaluation Framework): introduces a process-oriented evaluation framework to assess LLM behavioral fidelity in dynamic decision-making tasks, including Intrinsicality (no intervention), Instruction (risk-framed guidance), and Imitation (human data provision).
- The framework systematically evaluates how LLM agents adapt under varying levels of external guidance and human-derived noise across tasks like second-price auctions and newsvendor problems.
- This approach reveals that LLMs default to stable, conservative strategies diverging from human variability, highlighting a persistent alignment gap in behavioral fidelity for social science simulations.

---

[Cybernaut: Towards Reliable Web Automation](http://arxiv.org/abs/2508.16688v1)

- Cybernaut: introduces a novel framework for reliable web automation, featuring an LLM SOP Generator (converts demonstrations to instructions), a Web Browsing Agent (executes SOPs) with an LLM Planner (decomposes tasks into actions), State Manager (maintains execution context), Critical Element Handler (detects interactive elements), Action Executor (performs browser operations), and Web Browser (simulates user interaction), alongside Consistency Monitoring (evaluates execution reliability) with an Embedding Model (compares execution traces).
- The framework addresses challenges in consistent execution, accurate HTML element identification, and scalable automation for complex internal web interfaces by leveraging demonstration-based learning and a trace-based similarity metric.
- Cybernaut significantly improves task execution success rates and identifies consistent execution patterns, enabling reliable confidence assessment and adaptive guidance for enterprise-scale web automation.

---

[LIVEMCP-101: STRESS TESTING AND DIAGNOSING MCP-ENABLED AGENTS ON CHALLENGING QUERIES](http://arxiv.org/abs/2508.15760v1)

- LiveMCP-101: introduces a benchmark for stress testing and diagnosing Model Context Protocol (MCP)-enabled agents on challenging queries, utilizing a comprehensive framework for query construction and agent evaluation.
- The benchmark features 101 diverse real-world tasks requiring coordinated use of multiple MCP tools, with user queries refined through iterative LLM rewriting and manual review.
- It employs a novel evaluation approach that runs two agents in parallel—one following a ground-truth execution plan and another operating autonomously—to compute scores based on real-time outputs, revealing challenges in tool orchestration and identifying failure modes.

---

[End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning](https://github.com/MAGIC-AI4Med/Deep-DxSearch)

- Deep-DxSearch: introduces an end-to-end agentic RAG system trained with reinforcement learning, featuring an LLM-based agent that performs reasoning and retrieval actions, interacting with an external environment comprising a comprehensive medical retrieval corpus, guided by a multi-dimensional reward scheme.
- The system leverages a large-scale medical retrieval corpus, integrating disease guidelines, patient records, and clinical knowledge, to support traceable diagnostic reasoning across diverse medical scenarios.
- Deep-DxSearch's RL-based training optimizes the agent's policy through tailored rewards on output formatting, retrieval quality, analytical organization, and diagnostic accuracy, enabling adaptive retrieval and robust differential diagnosis.

---

[NiceWebRL: a Python library for human subject experiments with reinforcement learning environments](http://arxiv.org/abs/2508.15693v1)

- NiceWebRL: introduces a Python library for human subject experiments, with NiceWebRL (Python library), Jax-based Environment, NiceGUI (GUI library), Python-based Web Server, JavaScript-based Client, Stage Object, Instruction Stage, Feedback Stage, EnvStage Object, Database, Asynchronous Saving Module, Precomputation Module (Jax), Client-side Cache, Browser Session Cookie, LLM, Human Participant, and AI Model (Agent), enabling researchers to conduct online human subject experiments using machine reinforcement learning environments.
- The framework leverages Jax for precomputing environment dynamics and NiceGUI for Python-based GUI development, significantly reducing latency and simplifying the creation of complex web experiments for multiple clients.
- NiceWebRL supports the development of Human-like AI, Human-compatible AI, and Human-assistive AI by facilitating comparisons between human and AI performance, studying human-AI coordination, and integrating LLMs for task assistance.

---

[TRANSDUCTION IS ALL YOU NEED FOR STRUCTURED DATA WORKFLOWS](http://arxiv.org/abs/2508.15610v1)

- Agentics: introduces a modular framework for building agent-based systems capable of structured reasoning and compositional generalization over complex data. Redefines how agents interact with data through a declarative, type-driven approach grounded in logical transduction algebra.
- The framework leverages asynchronous and parallel LLM inference to support enterprise-scale structured data workflows, formalizing logical transduction as the transformation of a data object from one type to another based on target schema constraints.
- Agentics demonstrates state-of-the-art performance and scalability across tasks like domain-specific multiple-choice question answering, semantic parsing for text-to-SQL, and automated prompt optimization by treating agents as stateless transducers operating over well-defined data types.

---

[Interface on demand: Towards AI native Control interfaces for 6G](http://arxiv.org/abs/2508.15595v1)

- Multi-agent framework: introduces an AI-native approach to dynamically generate control interfaces between network functions (NFs), comprising a Matching Agent (aligns functionalities), a Codegen Agent (generates API server), NFsrc (source function), NFdest (destination function), a Provisioning Interface (agent communication channel), a Generated Interface Client (client-side interface), and a Generated Interface API Server (server-side interface).
- This framework addresses limitations of traditional standardized network interfaces, such as vendor-specific incompatibilities and lack of adaptability, by leveraging LLMs to create on-demand, functionally and semantically compatible interfaces.
- The system enables dynamic control interface generation for future mobile networks, enhancing interoperability and adaptability across multi-vendor and multi-RAT environments like 5G and WLAN.

---

[SafetyFlow: An Agent-Flow System for Automated LLM Safety Benchmarking](http://arxiv.org/abs/2508.15526v1)

- SafetyFlow: introduces an agent-flow system for automated LLM safety benchmarking, with a Data Pool (raw harmful texts), Ingestion Agent (extracts, preprocesses data), Categorization Agent (establishes taxonomy, categorizes samples), Generation Agent (generates harmful prompts), Augmentation Agent (enhances prompt diversity, translates), Deduplication Agent (removes duplicate/similar prompts), Filtration Agent (removes benign/simple prompts), Dynamic Evaluation Agent (adjusts benchmark difficulty), Toolset (supports agents' tasks), and SafetyFlowBench (final LLM safety benchmark), which automates benchmark construction in four days without human intervention.
- This system significantly reduces the time and resource costs associated with traditional manual benchmark curation, while ensuring high quality through modular agent design and a versatile toolset.
- The framework's automated pipeline and dynamic enhancement capabilities enable rapid dataset updates and effective evaluation of emerging LLM safety risks.

---

[Super-additive Cooperation in Language Model Agents](http://arxiv.org/abs/2508.15510v1)

- LLM Agent Simulation Framework: introduces a novel approach for LLM agents to strategize and act in complex social scenarios, featuring LLM Agents, a Tournament Structure, a Self-reflection Module with a Planner and an Evaluator (Critic), a Planning-Evaluation Loop, a Workflow Graph (comprising Round Start, Planning and Evaluation, Move Selection, and Payoff Computation Nodes), a Tournament State Object, a Prompting Strategy (including Game Description, Player/Opponent Info, Match History, Previous Plan, and Output Instructions Prompts), an Ollama Backend, and a LangSmith Debugging Interface.
- This framework simulates a virtual tournament where LLM agents, grouped into teams, engage in an Iterated Prisoner's Dilemma game under various social conditions (Repeated Interactions, Group Competition, Super-additive Cooperation) to study cooperative dynamics.
- The self-reflection prompting paradigm, which includes planning and critically assessing plans, enables agents to formulate long-term strategies and iteratively refine their behavior, providing insights into super-additive cooperation effects in LLM populations.

---

[DeepMEL: A Multi-Agent Collaboration Framework for Multimodal Entity Linking](http://arxiv.org/abs/2508.15876v1)

- DeepMEL (A Multi-Agent Collaboration Framework for Multimodal Entity Linking): introduces a multi-agent framework for multimodal entity linking, with a Role-Orchestrator (coordinates agents, manages updates), Modal-Fuser (aligns, fuses multimodal information), Candidate-Adapter (generates, refines candidate entities), and Entity-Clozer (disambiguates entities via cloze-prompt), achieving efficient alignment and disambiguation of textual and visual modalities.
- The framework employs a role-specialized division strategy and an adaptive iteration strategy, leveraging LLMs for summarization and LVMs for visual question-answering to bridge the modal gap and optimize candidate sets.
- DeepMEL reformulates the entity linking task into a structured cloze prompt, enhancing LLM comprehension and reasoning for improved multimodal disambiguation performance.

---

[From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence](http://arxiv.org/abs/2508.15447v1)

- BusiAgent: introduces a novel multi-agent LLM framework for business excellence, with a role-based agent system (optimizes decisions among specialized roles), a collaborative decision-making mechanism (combines brainstorming, hierarchical coordination), a tool integration system (extends action spaces with specialized tools), advanced prompt optimization (refines LLM queries dynamically), and a quality assurance system (ensures correctness and consistency).
- The framework leverages an extended Continuous Time Markov Decision Process, generalized entropy, and multi-level Stackelberg games to integrate granular operational insights with high-level strategic goals.
- It employs contextual Thompson sampling for prompt optimization and a comprehensive quality assurance system to mitigate errors, demonstrating superior performance in complex corporate decision-making.

---

[Cognitive Agents Powered by Large Language Models for Agile Software Project Management](http://arxiv.org/abs/2508.16678v1)

- CogniSim framework: introduces a cognitive Multi-Agent System designed to transform software project management by integrating cognitive agents powered by LLMs, with all its components, where it automates routine project tasks, enhances workflows, and aligns with established Agile practices, particularly SAFe, to ensure scalability and effectiveness.
- The framework employs a layered architecture, including an LLM foundation, a MAS core, AI integrations, and a cognitive agents layer, to optimize software engineering workflows.
- CogniSim's modular design and iterative simulation approach enable controlled experimentation and evaluation of agent performance in complex Agile software development scenarios.

---

[MedRepBench: A Comprehensive Benchmark for Medical Report Interpretation](http://arxiv.org/abs/2508.16674v1)

- MedRepBench (Comprehensive Benchmark for Medical Report Interpretation): introduces a comprehensive benchmark for evaluating end-to-end VLMs on structured medical report understanding, with a dataset of 1,900 de-identified Chinese medical reports, an objective evaluation protocol, an automated subjective evaluation protocol, and a reinforcement learning strategy (Group Relative Policy Optimization).
- The benchmark supports dual evaluation protocols, including field-level recall for structured clinical item extraction and an LLM-based subjective scoring for factuality and interpretability of patient-facing explanations.
- It also incorporates GRPO, a reinforcement learning strategy, to optimize VLM performance in structured interpretation, demonstrating significant recall gains and highlighting the importance of layout-aware, vision-based understanding.

---

[IPIGUARD: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents](http://arxiv.org/abs/2508.15310v1)

- IPIGUARD: introduces a novel task execution paradigm that defends against Indirect Prompt Injection (IPI) attacks in LLM agents by decoupling action planning from external data interaction using a planned Tool Dependency Graph (TDG).
- The framework constructs a TDG during a planning phase to pre-define tool invocations and their dependencies, enforcing strict constraints on tool execution to prevent malicious tool invocations triggered by injected instructions.
- It addresses challenges like unknown arguments and limited adaptability through Argument Estimation and Node Expansion, and mitigates tool overlap attacks with Fake Tool Invocation, ensuring robust and secure task completion.

---

[Coarse-to-Fine Grounded Memory for LLM Agent Planning](http://arxiv.org/abs/2508.15305v1)

- CFGM (Coarse-to-Fine Grounded Memory): introduces a novel framework that enhances LLM agents by systematically grounding memory with LLM's internal knowledge during experience collection, tips extraction, and adaptive planning, including Coarse-Grained Focus-Driven Experience Collection (collects diverse experiences), Hybrid-Grained Experience-Wise Tips Extraction (distills actionable tips), and Fine-Grained Key Information Adaptive Planning (corrects planning anomalies).
- The framework leverages LLM's inherent knowledge to generate coarse-grained focus points for guiding experience collection and distills hybrid-grained tips from experiences, which are then retrieved to enhance online planning.
- When encountering environmental anomalies, the agent activates fine-grained self-QA reflection, grounded in current situations and past successes, to dynamically adjust its planning and actions.

---

[Comp-X: On Defining an Interactive Learned Image Compression Paradigm With Expert-driven LLM Agent](http://arxiv.org/abs/2508.15243v1)

- Comp-X (Interactive Learned Image Compression Paradigm): introduces an intelligently interactive image compression paradigm, with its LLM Agent (core controller), Multi-functional Image Codec (compression engine), In-Context Learning with Expert Feedback (LLM knowledge enhancement), Coding Expert (human guidance), Tool Pool (external utilities), Grounded-SAM (segmentation tool), Detectron2 (detection tool), and IIC-Bench (evaluation benchmark), where it enables customized image compression via natural language instructions and expert feedback.
- The system unifies diverse coding modes into a single multi-functional image codec, employs an interactive LLM agent augmented with expert feedback for understanding and tool use, and introduces IIC-Bench for systematic evaluation.
- Comp-X demonstrates efficient understanding of coding requests and impressive textual interaction, maintaining competitive compression performance across various application scenarios.

---

[See it. Say it. Sorted: Agentic System for Compositional Diagram Generation](http://arxiv.org/abs/2508.15222v1)

- See it. Say it. Sorted.: introduces a training-free agentic system that couples a Critic VLM (identifies discrepancies/suggests modifications), multiple LLMs (generate diverse SVG candidates), and a Judge VLM (selects best SVG candidate) to produce editable Scalable Vector Graphics (SVG) programs from hand-drawn sketches and text instructions.
- The system operates in an iterative Critic-Candidates-Judge loop, emphasizing qualitative reasoning and relative spatial relationships over precise numerical values for stable optimization.
- This framework enables accurate, controllable, and editable diagram generation, moving beyond pixel-level synthesis toward structured programmatic outputs extensible to graphics design environments.

---

[ContextualLVLM-Agent: A Holistic Framework for Multi-Turn Visually-Grounded Dialogue and Complex Instruction Following](http://arxiv.org/abs/2508.15164v1)

- CoLVLM Agent (Contextual LVLM Agent): introduces a holistic framework for multi-turn visually-grounded dialogue and complex instruction following, enhancing existing LVLMs with advanced reasoning and instruction following capabilities through an iterative "memory-perception-planning-execution" cycle.
- This framework integrates a Dialogue Context Memory Module, a Dynamic Visual Perception Module, a Reasoning & Planning Engine, and an Action Execution & Response Generation Module, simulating human-like cognitive processes for deep visual understanding and multi-step instruction execution.
- The agent achieves superior performance in reasoning depth, instruction adherence, and error suppression, maintaining robustness over extended dialogue turns and significantly reducing context loss and visual hallucinations without extensive model re-training.

---

[MedResearcher-R1: Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework](http://arxiv.org/abs/2508.14880v2)

- MedResearcher-R1 (Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework): introduces a medical deep research agent that addresses challenges in medical reasoning through a Reasoning-Acting Paradigm, Dynamic Tool Selection Strategy, General-Purpose Tools, Medical-Specific Tool Suite, KISA (Knowledge-Informed Trajectory Synthesis Approach), and Large-scale Agent Training.
- The framework employs a novel data synthesis framework, KISA, to generate complex multi-hop medical queries and reasoning trajectories, and integrates a custom-built private medical retrieval engine alongside general-purpose tools for accurate medical information synthesis.
- Training involves a two-stage paradigm combining supervised fine-tuning with Masked Trajectory Guidance and online reinforcement learning with composite rewards, enabling the agent to achieve expert-level medical research capabilities and state-of-the-art performance on medical benchmarks.

---

[A Dynamical Systems Framework for Reinforcement Learning Safety and Robustness Verification](http://arxiv.org/abs/2508.15588v1)

- A Dynamical Systems Framework for Reinforcement Learning Safety and Robustness Verification: introduces a novel framework that analyzes the combined RL agent and its environment as a discrete-time autonomous dynamical system, leveraging Finite-Time Lyapunov Exponent (FTLE) Calculation, Lagrangian Coherent Structures (LCS) Identification (including Repelling LCS and Attracting LCS), and quantitative metrics (Mean Boundary Repulsion (MBR) Metric, Aggregated Spurious Attractor Strength (ASAS) Metric, Temporally-Aware Spurious Attractor Strength (TASAS) Metric) along with a Local Stability Guarantee for formal verification.
- The framework provides a comprehensive and interpretable assessment of policy behavior by identifying critical flaws not apparent from reward alone, offering deterministic and formal guarantees of safety and robustness.
- By mapping dynamical structures to policy properties, the framework effectively identifies repelling LCS as safety barriers and attracting LCS as convergence properties or potential failure modes, including "trap" states.

---

[Adversarial Agent Behavior Learning in Autonomous Driving Using Deep Reinforcement Learning.](http://arxiv.org/abs/2508.15207v1)

- Adversarial Agent Behavior Learning Framework: introduces a multi-stage deep reinforcement learning framework to train adversarial agents that induce failure scenarios for autonomous driving ego-agents, and subsequently train a robust ego-agent.
- The framework leverages PPO for initial and robust ego-agent policy learning, and TD3 with an adversarial reward formulation to generate adversarial policies for surrounding agents.
- This approach evaluates ego-agent performance degradation against adversarial attacks and provides a defense mechanism by training a robust ego-agent to overcome these adversaries.

---

[End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning](https://github.com/MAGIC-AI4Med/Deep-DxSearch)

- Deep-DxSearch: introduces an agentic RAG system trained end-to-end with reinforcement learning, featuring an LLM-based agent, an external environment comprising a medical retrieval corpus (Disease Information Guideline, Patient Record Database, Clinical Knowledge Collection), passive actions, a reward scheme (Format, Patient Matching, Searching, Diagnosis), Group Relative Policy Optimization, multi-stage reward adaption, and retrieval tools (Phenotype Parser, Patient Matcher, Knowledge Searcher, MedDoc Summarizer), enabling traceable retrieval-augmented reasoning for medical diagnosis.
- The system frames the LLM as the core agent and the retrieval corpus as its environment, using tailored rewards on format, retrieval, reasoning structure, and diagnostic accuracy to evolve the agentic RAG policy from large-scale data through RL.
- Deep-DxSearch consistently outperforms prompt-engineering and training-free RAG approaches, achieving substantial gains in diagnostic accuracy for both common and rare diseases under in-distribution and out-of-distribution settings.

---

[AI Compute Architecture and Evolution Trends](http://arxiv.org/abs/2508.21394v1)

- Seven-layer Model for AI Compute Architecture: introduces a structured framework for AI computing, detailing its evolution through Physical, Link, Neural Network, Context, Agent, Orchestrator, and Application layers, along with the critical role of Tokens.
- The paper analyzes AI development across three phases: Training Compute, Test-Time Compute, and Agentic/Physical AI, highlighting the shift from academic research to practical applications and the challenges of scaling computing power and energy efficiency.
- It further explores strategies like Scale-Up and Scale-Out for hardware, the impact of context memory on LLMs, and the emergence of AI agents and AI-based ecosystems, including the concept of democratizing AI through smaller, distilled LLMs.

---

#### 20th August 2025

[aiXiv: A Next-Generation Open Access Ecosystem for Scientific Discovery Generated by AI Scientists](https://github.com/aixiv-org)

- aiXiv: introduces a multi-agent platform for autonomous scientific discovery, including an AI Scientist (generating content), Research Agents Community (conducting experiments), aiXiv Core (managing workflow), Editor/Chair Agents (synthesizing reviews), Reviewer Agents (evaluating submissions), a Multi-AI Voting Mechanism (for publication decisions), an Agents Interface (for agent interaction), an aiXiv Repository (for accepted content), a Public-facing Interface (for human engagement), a Prompt Injection Detection and Defense Pipeline (for security), and a Retrieval-Augmented Generation (RAG) Framework (for enhanced reviews).
- The platform enables AI agents to autonomously generate, review, refine, and publish scientific content through a closed-loop review process, ensuring continuous improvement and quality control.
- It integrates safeguards against prompt injection attacks and provides interfaces for seamless collaboration between human and AI scientists, fostering a scalable ecosystem for scientific knowledge dissemination.

---

[Open-Universe Assistance Games](http://arxiv.org/abs/2508.15119v1)

- GOOD (GOals from Open-ended Dialogue): introduces a data-efficient, online method that extracts and infers a distribution over natural language goals from human interaction, using LLMs to simulate users and perform probabilistic inference over candidate goals.
- This approach enables rich goal representations and uncertainty estimation without requiring large offline datasets, outperforming baselines without explicit goal tracking in text-based grocery shopping and simulated household robotics environments.
- The method leverages a modular architecture with goal proposition, removal, and ranking modules to track and update explicit hypotheses of plausible user goals in open-ended interaction, supporting flexible, interpretable, and corrigible AI agents.

---

[LLMs and Agentic AI in Insurance Decision-Making: Opportunities and Challenges For Africa](http://arxiv.org/abs/2508.15110v1)

- Agentic AI System: introduces a framework for insurance decision-making, with an orchestrator managing AI agents, memory, tools, and an LLM core, where the paper highlights the transformative potential of LLMs and agentic AI in the African insurance sector.
- The system processes user prompts, decomposes them into sub-tasks, dispatches to specialized agents, and recombines outputs for context-aware responses.
- The paper emphasizes addressing critical gaps in the African insurance market through inclusive, sustainable, and equitable AI strategies.

---

[S³LORA: Safe Spectral Sharpness–Guided Pruning in Adaptation of Agent Planner](http://arxiv.org/abs/2508.15068v1)

- S³LORA (Safe Spectral Sharpness-Guided Pruning LoRA): introduces a lightweight, data-free, and model-independent framework that mitigates safety risks in LoRA-adapted models by inspecting only the fine-tuned weight updates, with LoRA update (fine-tuned weights), MAS-SVD (robust spectral decomposition), Spectral Sharpness Index (SSI) (sharpness-aware metric), and Pruning mechanism (removes unsafe layers), where it identifies and removes potentially unsafe LoRA updates by analyzing fine-tuned weights using spectral sharpness criteria.
- The framework leverages Magnitude-Aware Spherically Normalized SVD (MAS-SVD) to robustly analyze the structural properties of LoRA updates while preserving global magnitude information, and then computes the Spectral Sharpness Index (SSI) to detect layers with highly concentrated and potentially unsafe updates.
- Layers with high SSI scores are pruned post-hoc to reduce risk without sacrificing task performance, establishing S³LORA as a practical and scalable solution for safely deploying LLM-based agents in resource-constrained and safety-critical environments.

---

[Emergent Crowds Dynamics from Language-Driven Multi-Agent Interactions](http://arxiv.org/abs/2508.15047v1)

- Language-Driven Multi-Agent Interaction System: introduces a novel method for simulating emergent crowd dynamics, with its Dialogue System (LLM-powered conversation generation) and Language-Driven Movement Module (LLM-powered locomotion control) enabling agents to make autonomous decisions based on social interactions and environmental context.
- The system allows agents to collectively pick new goals, update path planning, and change steering parameters, leading to realistic crowd behaviors without scenario-specific scripting.
- By conditioning agent behavior on individual personalities, emotional states, and relationships, the framework generates complex social and contextual scenarios, demonstrating emergent group behaviors and information propagation.

---

[Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism](http://arxiv.org/abs/2508.15030v1)

- Collab-REC (Collaborative Recommendation): introduces an LLM-based multi-agent framework that balances tourism recommendations by employing LLM-Agents (specialized LLM-based agents), a Moderator Core (orchestrates agent interaction, evaluates proposals), and an External Knowledge Base (KB) (database of European cities).
- The framework's LLM-Agents, including Popularity, Personalization, and Sustainability agents, iteratively propose and refine city candidates under the Moderator Core's guidance, which penalizes repeated or hallucinated suggestions.
- This multi-round negotiation process, driven by a custom Scoring Function and State Manager, fosters iterative compromise, significantly broadens recommendation diversity, and mitigates popularity bias.

---

[HERAKLES: Hierarchical Skill Compilation for Open-ended LLM Agents](http://arxiv.org/abs/2508.14751v1)

- HERAKLES (Hierarchical Skill Compilation for Open-ended LLM Agents): introduces a hierarchical autotelic RL agent that continuously compiles mastered goals into a low-level policy, dynamically expanding its skill set.
- The framework leverages an LLM as a high-level controller for goal decomposition and generalization, while a small neural network executes primitive actions as the low-level policy.
- HERAKLES is evaluated in the Crafter environment, demonstrating effective scaling with goal complexity, improved sample efficiency, and robust adaptation to novel challenges through skill compilation.

---

[Enabling Multi-Agent Systems as Learning Designers: Applying Learning Sciences to AI Instructional Design](http://arxiv.org/abs/2508.16659v1)

- MAS (Multi-Agent System): introduces a framework for AI instructional design, embedding the KLI (Knowledge-Learning-Instruction) framework into three distinct generative systems: SAS (Single-Agent System), MAS-Roles (Role-Based Multi-Agent System), and MAS-CMD (Multi-Agent System with Conquer and Merge Discussion).
- The MAS-Roles system employs a sequential pipeline of specialized agents (KC, Learning Process, Instructional Principle, Design, and Feedback Agents) to operationalize KLI theory, while MAS-CMD utilizes a collaborative architecture with Initial Generation, Collaborative Discussion, and Final Selection Agents to simulate professional discussions.
- The study demonstrates that embedding pedagogical principles into LLM systems, particularly through collaborative multi-agent architectures like MAS-CMD, significantly enhances the creativity, contextual relevance, and classroom-readiness of generated learning activities compared to baseline single-agent LLM interactions.

---

[MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers](http://arxiv.org/abs/2508.14704v1)

- MCP-Universe: introduces a comprehensive benchmark for evaluating LLMs in real-world MCP environments, with all components including User Instruction (initial task prompt), Agent (LLM task solver), Actions (agent's tool calls), Observations (MCP server responses), MCP Servers (real-world external tools), Final States (task completion outcome), Execution-Based Evaluator (automated task assessment), LLM Manager (manages LLM configurations), Agent Builder (constructs agent architectures), Task, MCP Server, Evaluator Configuration (dynamically configures evaluation pipeline), and Evaluator (defines task success criteria).
- The benchmark is grounded in real-world MCP servers across six core domains (Location Navigation, Repository Management, Financial Analysis, 3D Design, Browser Automation, Web Searching) and employs execution-based evaluators for rigorous, objective assessment of LLM performance.
- It reveals fundamental limitations of current LLM agents, such as challenges with long contexts, unfamiliar tools, and cross-domain performance variations, providing a testbed for advancing robust LLM applications.

---

[ENTROPY-CONSTRAINED STRATEGY OPTIMIZATION IN URBAN FLOODS: A MULTI-AGENT FRAMEWORK WITH LLM AND KNOWLEDGE GRAPH INTEGRATION](http://arxiv.org/abs/2508.14654v1)

- H-J (Hierarchical Joint Optimization): introduces a hierarchical multi-agent framework for urban flood response, integrating LLMs, a dual-channel knowledge retrieval module, an entropy-constrained strategy generation module, a strategy translation and execution module, a feedback optimization module, multi-source data, simulation agents, and J-H Adaptive Thresholding, which establishes a closed-loop pipeline from multi-source perception to strategic execution and continuous refinement.
- The framework addresses challenges in urban flood emergency scheduling by dynamically balancing competing goals, adapting to rapidly changing environments, and mitigating semantic instability and execution inconsistency of LLM-generated strategies.
- H-J leverages knowledge-guided prompting, entropy-constrained generation, and objective-driven feedback optimization to enhance resilience, outperforming rule-based and reinforcement learning baselines in traffic smoothness, task success rate, and system robustness.

---

[Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination](http://arxiv.org/abs/2508.14635v1)

- LLM Agent Architecture: introduces a ReAct-based decision pipeline for LLM-driven agents, with Assistant (reasoning, action determination), Tools (executes selected actions), Communication Message (generates broadcast messages), Ollama (LLM model provider), LangGraph (modular state graph), Environment (graph-based rescue scenario), and Message Channel (shared communication medium), designed to evaluate LLM agents' coordination in multi-agent rescue tasks.
- The paper investigates LLM agents' ability to coordinate actions, plan, and reason in a structured indoor victim rescue mission, focusing on division of labor, prioritization, and cooperative planning.
- It systematically evaluates performance using coordination-sensitive metrics and compares LLM agents against a deterministic heuristic baseline to identify strengths and limitations in physically grounded multi-agent collaboration.

---

[BUILDING AND MEASURING TRUST BETWEEN LARGE LANGUAGE MODELS](http://arxiv.org/abs/2508.15858v1)

- Experimental Methodology for LLM Trust: introduces an experimental methodology to build and measure trust between LLMs, utilizing Trustor AI (evaluating subject) and Trustee AI (trust-garnering subject) roles, employing three Trust-Building Strategies (methods to foster trust) and three Trust Measures (methods to quantify trust) across various LLM Models (tested conversational agents).
- The methodology systematically combines trust-building strategies like generated rapport, prewritten dialogue context, and direct system prompt instructions with trust measures including explicit questionnaires, investment games, and persuasion susceptibility.
- Key findings indicate that explicit trust measures in LLMs may be misleading due to sycophancy, LLMs' willingness to collaborate in investment games is stake-dependent, and trust-building strategies significantly enhance persuasion susceptibility.

---

[Who Sees What? Structured Thought-Action Sequences for Epistemic Reasoning in LLMs](http://arxiv.org/abs/2508.14564v1)

- Structured Thought-Action Sequences for Epistemic Reasoning: introduces a methodology for improving LLM-based agents' perspective-taking capabilities by generating structured "thought-action" examples, with all Fast Downward planner, Structured solution-processing pipeline, G-type example extraction, E-type example extraction, L-type example extraction, LLM (GPT 03-mini) (Example Generator), ReAct framework, Matcher agent (LLM-based), Director agent (LLM-based), PDDL-based household environment-components, where the paper proposes a structured solution-processing pipeline to create diverse examples for LLMs operating within a ReAct framework in a simulated environment.
- The pipeline extracts three types of examples—G-type (optimal goal paths), E-type (information-seeking paths), and L-type (locally optimal decisions)—from a Fast Downward planner's reasoning trees, which are then converted into natural language thought-action pairs by an LLM.
- The study evaluates these examples in a PDDL-based household environment with LLM-based Matcher and Director agents, finding that while G-type and E-type support efficiency and exploration, L-type examples slightly improve agent behavior by reducing clarification requests.

---

[MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning](http://arxiv.org/abs/2508.16654v1)

- MSNav (Memory Spatial Navigation): introduces a zero-shot vision-and-language navigation framework that integrates a Memory Module (dynamic topological map), a Spatial Module (spatial reasoning/object inference) powered by Qwen-Sp (fine-tuned LLM) and YOLO-World (object detection), and a Decision Module (LLM-based path planning/action) utilizing GPT-4o (advanced LLM).
- The framework addresses poor spatial reasoning, weak cross-modal grounding, and memory overload in long-horizon tasks by dynamically pruning irrelevant nodes from the topological map and enhancing visual observations with task-relevant objects.
- It achieves state-of-the-art performance on R2R and REVERIE datasets, demonstrating improved success rates and path efficiency, and introduces the Instruction-Object-Space (I-O-S) dataset for enhancing LLM spatial reasoning capabilities.

---

[Cohort-Aware Agents for Individualized Lung Cancer Risk Prediction Using a Retrieval-Augmented Model Selection Framework](http://arxiv.org/abs/2508.14940v1)

- Cohort-Aware Agents: introduces a retrieval-augmented model selection framework for individualized lung cancer risk prediction, with Patient CT scan (Input imaging data), Patient Metadata (Input clinical data), Feature Extraction (Extracts relevant characteristics), Embedding (Vectorizes patient features), Vector Database (Stores reference cohorts), Similarity Search (FAISS-based) (Identifies similar cohorts), Retrieved Top-1 Cohort (Most relevant patient group), LLM prompt (Input for LLM reasoning), LLMs (Risk Prediction Agent) (Selects optimal model), Tools (Executes prediction models), Model Pool (Available prediction algorithms), Model Selection (Chooses best algorithm), Model Prediction (Generates risk score), and Risk Probability (Final risk assessment), which dynamically selects the most appropriate prediction model for each patient based on cohort-specific knowledge.
- This two-stage agent pipeline first performs cohort retrieval using FAISS-based similarity search to identify the most relevant patient population, then prompts an LLM with the retrieved cohort and its performance metrics to recommend the optimal prediction algorithm from a pool of eight representative models.
- The framework enables dynamic, cohort-aware risk prediction personalized to each patient's profile, supporting flexible and cohort-driven model selection across diverse clinical populations for individualized risk assessment.

---

[Organ-Agents: Virtual Human Physiology Simulator via LLMs](http://arxiv.org/abs/2508.14357v1)

- Organ-Agents: introduces a novel multi-agent framework for simulating human physiology, with Simulator agents (model specific physiological systems), an Analyzer agent (summarizes observed sequences), a Correlator agent (selects cross-system references), a Compensator agent (adjusts low-confidence simulations), and Memory (Analyzer log) (stores historical summaries), enabling time-resolved physiological simulation.
- The framework employs LLM-driven agents, each assigned to a specific physiological system, and coordinates their interactions through reinforcement learning to achieve dynamic, context-aware, and physiologically plausible simulations.
- Organ-Agents supports counterfactual simulations and maintains high accuracy across diverse physiological systems and clinical severity strata, positioning it as a credible digital twin for precision diagnosis and treatment simulation.

---

[aiXiv: A Next-Generation Open Access Ecosystem for Scientific Discovery Generated by AI Scientists](https://github.com/aixiv-org)

- aiXiv: introduces a multi-agent platform for autonomous scientific discovery, including an AI Scientist (generating content), Research Agents Community (conducting experiments), aiXiv Core (managing workflow), Editor/Chair Agents (synthesizing reviews), Reviewer Agents (evaluating submissions), a Multi-AI Voting Mechanism (for publication decisions), an Agents Interface (for agent interaction), an aiXiv Repository (for accepted content), a Public-facing Interface (for human engagement), a Prompt Injection Detection and Defense Pipeline (for security), and a Retrieval-Augmented Generation (RAG) Framework (for enhanced reviews).
- The platform enables AI agents to autonomously generate, review, refine, and publish scientific content through a closed-loop review process, ensuring continuous improvement and quality control.
- It integrates safeguards against prompt injection attacks and provides interfaces for seamless collaboration between human and AI scientists, fostering a scalable ecosystem for scientific knowledge dissemination.

---

[From Passive Tool to Socio-cognitive Teammate: A Conceptual Framework for Agentic AI in Human-AI Collaborative Learning](http://arxiv.org/abs/2508.14825v1)

- APCP (Adaptive instrument, Proactive assistant, Co-learner, Peer collaborator) Framework: introduces a four-level model of escalating AI agency in human-AI collaborative learning, including Adaptive Instrument (passive tool for task automation), Proactive Assistant (monitors and suggests for reflection), Co-learner (dialogic partner for joint inquiry), and Peer Collaborator (socio-cognitive teammate for collaboration) components, to conceptualize AI's transition from a passive tool to a socio-cognitive teammate.
- This framework provides a structured vocabulary for analyzing the shifting roles and responsibilities between human and AI agents, moving beyond a simplistic tool-partner dichotomy.
- The framework guides the design of synergistic and trustworthy human-AI teams by articulating the graduated roles an AI partner can inhabit in educational contexts.

---

[A Comparative Evaluation of Teacher-Guided Reinforcement Learning Techniques for Autonomous Cyber Operations](http://arxiv.org/abs/2508.14340v1)

- Teacher-Guided Reinforcement Learning Techniques: introduces a comparative evaluation of four distinct teacher-guided techniques—Reward Shaping, Action Masking, Auxiliary Loss, and Feature Space Modification—to improve RL agent training efficiency in autonomous cyber operations within the CybORG environment.
- The approach leverages a Pretrained RL Agent as a teacher to provide guidance, aiming to accelerate learning and enhance early-stage policy performance for the learning RL Agent.
- The study demonstrates that Auxiliary Loss and Action Masking significantly improve initial performance and convergence speed, highlighting the potential of teacher guidance in critical cybersecurity domains.

---

[aiXiv: A Next-Generation Open Access Ecosystem for Scientific Discovery Generated by AI Scientists](http://arxiv.org/abs/2508.15126v1)

- aiXiv: introduces a next-generation open-access platform for human and AI scientists, with AI Scientist, Review Agents, Editor/Chair Agents, Prompt Injection Detection and Defense Pipeline, Multi-AI Voting Mechanism, API and Model Control Protocol (MCP) layer, aiXiv Repository, and Public-facing interface, where the platform enables AI agents to autonomously generate, review, refine, and publish scientific content through multi-agent workflows and a structured review system.
- The platform integrates multi-agent workflows, a structured review system, and iterative refinement pipelines to support end-to-end scientific discovery, ensuring quality control and collaboration.
- It addresses challenges in AI-generated research dissemination by offering a robust, closed-loop review process with safeguards against prompt injection attacks.

---

[Socially Interactive Agents for Preserving and Transferring Tacit Knowledge in Organizations: Requirements and approaches to AI-supported knowledge transfer](http://arxiv.org/abs/2508.19942v1)

- KTF (Socially Interactive Agents as Knowledge Transfer Facilitators): introduces an AI-supported framework for preserving and transferring tacit knowledge in organizations, with all SIA-KTS, knowledge holder, knowledge taker, knowledge and data driven approach, conversational AI, behavioral models, knowledge transfer and organization, training material, LLMs, RAG, CoT prompts, automatic speech recognition, and speech generation components, where the framework leverages socially interactive agents to facilitate the elicitation and transfer of experiential knowledge.
- The framework utilizes LLMs, RAG, and CoT prompts to enable SIAs to engage in empathetic, natural-language dialogues, acting as active participants in knowledge elicitation and structured reflection.
- The approach aims to build trust and adapt to individual needs, addressing the challenges of tacit knowledge transfer by creating socio-technical systems for intuitive human-AI interaction.

---


[Adaptive Vision-Based Coverage Optimization in Mobile Wireless Sensor Networks: A Multi-Agent Deep Reinforcement Learning Approach](http://arxiv.org/abs/2508.14676v1)

- MADRL (Multi-Agent Deep Reinforcement Learning): introduces a novel vision-based approach for adaptive coverage optimization in Mobile Wireless Sensor Networks, with Camera System (overhead monitoring), Image Capture (visual data acquisition), LED Detection (DL) (active sensor identification), Sensor Localization & Status Detection (position and state determination), RL Engine (DQN + MARL) (policy learning), Sensor Movement Decision (action selection), and Sensor Relocation (Environment Update) (physical sensor movement) components, enabling mobile sensors to autonomously position themselves for maximum area coverage.
- The system utilizes a live camera and deep learning for real-time monitoring of sensor LED indicators to evaluate coverage and compute rewards, facilitating decentralized, cooperative sensor control.
- This approach significantly enhances adaptability, energy efficiency, and robustness in MWSNs by eliminating predefined policies and allowing self-reconfiguration in response to energy depletion and environmental changes.

---



#### 19th August 2025

[Self-Organizing Agent Network for LLM-based Workflow Automation](http://arxiv.org/abs/2508.13732v1)

- SOAN (Self-Organizing Agent Network): introduces a novel structure-driven orchestration framework for LLM-based workflow automation, with Agent Generation (creates specialized agents), Generated Workflow Verification (validates workflow correctness), Hypotheses Generation (optimizes agent structures), and SOAN Scale Control (manages agent life-value), designed to handle complex, multi-layer nested workflows in enterprise environments.
- The framework incrementally builds a formalized agent network by identifying and encapsulating structural units as independent agents, enhancing modularity and clarity in orchestration, and dynamically adapts to unseen workflows through structural hypotheses and optimization.
- SOAN leverages agent collaboration and a feedback-driven structural optimization mechanism, including linear insertion, branching, and nesting operations, to achieve robust generalization, fault tolerance, and execution efficiency in complex workflow scenarios.

---


[Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](http://arxiv.org/abs/2508.14031v1)

- PING (Prefix INjection Guard): introduces an iterative framework that automatically generates and selects natural language prefixes to enhance LLM agent safety and performance, including a GENERATOR (LLM) (proposes candidate prefixes), Performance Score Function (fperf) (measures task completion), Refusal Score Function (frefusal) (measures harmful refusal), Overall Score Function (combines performance, refusal scores), Prefix Pool (U) (stores prefixes for seeding), Evaluated Prefixes List (E) (stores evaluated prefixes), and Selection Mechanism (selects optimal prefixes).
- The framework addresses the issue of unintended misalignment in LLM agents caused by fine-tuning on benign agentic datasets, which can lead to increased execution of harmful tasks and reduced refusal behavior.
- PING prepends optimized natural language prefixes to agent responses, guiding LLMs to refuse harmful requests while maintaining high performance on benign tasks across web navigation and code generation domains.

---

[Learning to Use AI for Learning: How Can We Effectively Teach and Measure Prompting Literacy for K–12 Students?](http://arxiv.org/abs/2508.13962v1)

- Prompting Literacy Module: introduces a web-based interactive instructional system with a Learning Scenario Introduction, Prompt Creation Interface, AI Chatbot, AI Auto-Grader, Feedback Display, and Grading Dimensions, designed to teach K-12 students prompting literacy through scenario-based deliberate practice and immediate feedback.
- The system enables students to craft prompts for an LLM-powered AI Chatbot, receive AI-generated responses, and then obtain automated evaluations with detailed explanations based on predefined grading criteria.
- This module aims to enhance students' prompting skills and confidence in using AI for learning by providing experiential practice and elaborated immediate feedback on their prompt writing performance.

---

[LLM-Powered Virtual Patient Agents for Interactive Clinical Skills Training with Automated Feedback](http://arxiv.org/abs/2508.13943v1)

- LLM-Powered Virtual Patient Agents: introduces a novel framework for interactive clinical skills training, integrating a Frontend (user interface), Backend (core logic), Patient Agent (simulates patient), Tutor Agent (provides feedback), Large Language Model (LLM) (powers agents), Conversation Manager (manages dialogue), Automatic Speech Recognition (ASR) (transcribes speech), ROS Log-inspired Message Stream (logs interactions), and OSCE Scenario Management (manages scenarios) to enable dynamic patient behavior and automated feedback.
- The system enhances traditional text-based virtual patients by equipping them with functional action spaces for realistic, non-textual interactions and provides instant, personalized feedback from virtual tutors.
- This innovative platform offers medical students a low-cost, accessible solution for personalized Objective Structured Clinical Examinations (OSCE) preparation at home.

---

[The Collaboration Paradox: Why Generative AI Requires Both Strategic Intelligence and Operational Stability in Supply Chain Management](http://arxiv.org/abs/2508.13942v1)

- The Final Two-Layer Collaborative Framework: introduces a hierarchical AI-driven supply chain management system that synthesizes high-level proactive strategic policy-setting by an LLM-powered Strategy Generation Agent (SGA) with low-level collaborative operational execution among supply chain entities.
- The framework's Proactive Strategic Policy-Setting layer utilizes an SGA, which employs Retrieval-Augmented Generation (RAG) and a Virtual Expert Panel, to generate and evaluate strategic choices for system-wide inventory targets.
- Its Collaborative Operational Execution layer ensures stability through a VMI-style protocol, where the Manufacturer centrally manages ordering and proactively pushes inventory downstream to the Retailer, mitigating emergent instabilities like hoarding.

---

[LLMind 2.0: Distributed IoT Automation with Natural Language M2M Communication and Lightweight LLM Agents](http://arxiv.org/abs/2508.13920v1)

- LLMind 2.0 (Distributed IoT Automation with Natural Language M2M Communication and Lightweight LLM Agents): introduces a distributed IoT automation framework that leverages a central Coordinator and lightweight LLM Agents on devices for scalable, natural language-based machine-to-machine communication.
- The Coordinator (central LLM orchestrator) decomposes human instructions into natural language subtasks, which are then processed by device-specific Agents (device-specific LLM) for local code generation and execution.
- The framework enhances scalability, reliability, and privacy by offloading code generation to local Agents, utilizing RAG (maps subtasks to APIs) for accurate API mapping, and FSM-based code generation with fine-tuned LLMs for robust execution.

---

[Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback](http://arxiv.org/abs/2508.13915v1)

- TS-Agent (Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback): introduces a modular agentic framework designed to automate and enhance financial time-series modeling workflows through a structured, iterative decision process across model selection, code refinement, and fine-tuning stages.
- The framework leverages external resources like a Case Bank, Financial TS Code Base, and Refinement Knowledge Bank, guided by a planner agent, and incorporates a reflective feedback mechanism for adaptive learning and robust debugging.
- TS-Agent's auditable design logs each decision and its rationale, ensuring transparency and interpretability crucial for high-stakes financial applications.

---

[BetaWeb: Towards a Blockchain-enabled Trustworthy Agentic Web](http://arxiv.org/abs/2508.13787v1)

- BetaWeb (Blockchain-enabled Trustworthy Agentic Web): introduces a blockchain-enabled trustworthy Agentic Web, providing a decentralized infrastructure for LLM-based multi-agent systems (LaMAS) to ensure verifiable identities, transparent interactions, and secure coordination.
- The framework integrates a Blockchain (decentralized immutable ledger) to manage the full lifecycle of Agents (autonomous entities) and Tasks (recorded actions), ensuring immutability and auditability.
- BetaWeb redefines agentic workflows by abstracting all interactions into standardized task procedures, supported by core system modules for task, agent, and rule management, fostering a self-sustaining machine-to-machine economy.

---

[Expertise-aware Multi-LLM Recruitment and Collaboration for Medical Decision-Making](http://arxiv.org/abs/2508.13754v1)

- EMRC (Expertise-aware Multi-LLM Recruitment and Collaboration): introduces a framework for medical decision-making, with expertise-aware agent recruitment and confidence- and adversarial-driven multi-agent collaboration, which dynamically selects and integrates LLMs to enhance diagnostic accuracy and reliability.
- The framework constructs an LLM expertise table to quantify domain-specific strengths, enabling dynamic selection of optimal LLMs as medical expert agents based on query category and difficulty.
- It enhances diagnostic reliability by integrating selected agents' responses through confidence fusion and adversarial validation within a multi-layer architecture, iteratively refining outputs.

---


[CausalPlan: Empowering Efficient LLM Multi-Agent Collaboration Through Causality-Driven Planning](http://arxiv.org/abs/2508.13721v1)

- CausalPlan: introduces a two-phase framework that integrates explicit structural causal reasoning into the LLM planning process, including a Structural Causal Action (SCA) model, a Causal Action Matrix M, Causal-Aware Action Planning, and Causal Backup Action, to enhance multi-agent collaboration.
- The framework addresses the challenge of LLM agents producing causally invalid actions in collaborative tasks by leveraging a learned causal graph to guide action selection and ensure intervention-consistent behaviors.
- CausalPlan significantly reduces invalid actions and improves collaboration in both AI-AI and human-AI settings, outperforming strong reinforcement learning baselines without requiring LLM fine-tuning.

---

[Interpreting the Interpreter: Can We Model post-ECB Conferences Volatility with LLM Agents?](http://arxiv.org/abs/2508.13635v1)

- LLM-as-a-Judge framework: introduces a novel methodology to simulate financial market reactions to ECB press conferences by employing LLM-based Synthetic Agents and an iterative Judge LLM feedback loop to refine prompting strategies and predict market disagreement.
- The framework evaluates three prompting strategies—zero-shot, few-shot, and LLM-as-a-Judge—to assess their impact on predictive performance and capture interpretive uncertainty.
- This approach provides central banks with a tool to anticipate market reactions and refine communication strategies by understanding how monetary policy signals are perceived and transmitted through financial markets.

---

[AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System](http://arxiv.org/abs/2508.13423v1)

- AdaptJobRec (LLM-Powered Agentic System): introduces a conversational job recommendation system that integrates an LLM-powered agent with a complexity identification mechanism, a few-shot learning memory process module, a task decomposition planner, and personalized recommendation tools, supported by a People.AI Knowledge Graph, Cassandra Database, Redis Cache, Kafka Cluster, AdaptJobRec Server, MCP Server, Front End, and User Profile Service API.
- The system classifies user queries into simple or complex, routing simple queries directly to a Job Application Microservice for rapid responses, while complex queries engage the memory module and planner for detailed processing and tool invocation.
- This architecture significantly reduces response latency for simple queries and minimizes dialogue rounds for complex queries, enhancing both efficiency and accuracy in conversational career recommendations.

---

[Large Language Models as Visualization Agents for Immersive Binary Reverse Engineering](http://arxiv.org/abs/2508.13413v1)

- LLM-augmented CogBRE (Large Language Model-augmented Cognitive Binary Reverse Engineering): introduces a system where an LLM acts as a visualization agent for immersive binary reverse engineering, leveraging its ability to query binary analysis tools, answer technical questions, and dynamically generate 3D visualizations.
- The system extends a VR platform for reverse engineering by integrating an LLM agent that generates immersive 3D visualizations aligned with analyst tasks and cognitive design principles.
- A pilot study evaluates the LLM's potential to produce cognitively-aligned 3D call graphs without explicit training, revealing variability in output quality.

---

[MedKGent: A Large Language Model Agent Framework for Constructing Temporally Evolving Medical Knowledge Graph](http://arxiv.org/abs/2508.12393v2)

- MedKGent (A Large Language Model Agent Framework for Constructing Temporally Evolving Medical Knowledge Graph): introduces a framework for constructing temporally evolving medical KGs, leveraging PubMed abstracts, with an Extractor Agent for triple extraction and a Constructor Agent for incremental graph integration.
- The framework processes biomedical abstracts daily, using LLMs for relation inference and conflict resolution, and incorporates confidence scores and timestamps to ensure dynamic and trustworthy knowledge representation.
- This approach enables the KG to evolve alongside new findings, supports literature-based drug repurposing, and enhances medical question answering by providing a time-sensitive and reliable knowledge base.

---

[FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction](http://arxiv.org/abs/2508.11987v2)

- FutureX (Advanced Live Benchmark for LLM Agents in Future Prediction): introduces a dynamic and live evaluation benchmark for LLM agents performing future prediction tasks, with Event Database Construction (Initial data setup), Website Collection (Gathers raw website URLs), Website Curation (Filters, refines website sources), Future Event Daily Curation (Prepares daily prediction questions), Event Manipulation (Transforms websites into events), Event Filtering (Removes unsuitable questions), Agent Daily Prediction (LLM agents make predictions), LLM Agents (Models under evaluation), Answer Daily Acquisition (Obtains ground-truth answers), Date Filtering (Selects events by resolution date), Website Crawling (Retrieves ground-truth outcomes), Answer Extraction (Extracts precise answers), Daily Score (Evaluates agent performance), and Human Experts (Oversee curation, quality control), where it provides a comprehensive, contamination-free evaluation of LLM agents' advanced search and reasoning capabilities.
- The benchmark is the largest and most diverse live benchmark for future prediction, supporting real-time daily updates and eliminating data contamination through an automated pipeline for question gathering and answer collection.
- It evaluates 25 LLM/agent models, including those with reasoning, search capabilities, and integration of external tools, assessing their adaptive reasoning and performance in dynamic environments.

---

[Agentic DraCor and the Art of Docstring Engineering](http://arxiv.org/abs/2508.13774v1)

- MCP (Model Context Protocol): introduces an MCP server for DraCor, enabling LLMs to autonomously interact with the DraCor API through various tools, where "Docstring Engineering" is crucial for optimizing LLM-tool interaction.
- The paper evaluates the LLM's tool selection and application, focusing on "Tool Correctness," "Tool-Calling Efficiency," and "Tool-Use Reliability" through systematic observation of prompts.
- Findings highlight the promise of agentic AI for computational literary studies and the need for robust infrastructure development, emphasizing that comprehensive tool documentation is vital for reliable LLM performance.

---


[COMPUTERRL: SCALING END-TO-END ONLINE REINFORCEMENT LEARNING FOR COMPUTER USE AGENTS](http://arxiv.org/abs/2508.14040v1)

- COMPUTERRL (Scaling End-to-End Online Reinforcement Learning for Computer Use Agents): introduces a framework for autonomous desktop intelligence, integrating an API-GUI paradigm, a distributed RL infrastructure, and the Entropulse training strategy for scalable online reinforcement learning.
- The framework utilizes a Rollout Engine, parallel Environments, a Controller, a Data Queue, and an Online Update module to enable agents to operate complex digital workspaces.
- Entropulse, as a key training strategy, alternates RL with supervised fine-tuning to mitigate entropy collapse and sustain learning, achieving improved performance on desktop automation tasks.

---

[Multimodal Data Storage and Retrieval for Embodied AI: A Survey](http://arxiv.org/abs/2508.13901v1)

- Embodied AI System: This survey evaluates the conceptual architecture of an Embodied AI System, which integrates multimodal sensors, a data management system (for storage and retrieval), a learning/decision module, and actuators interacting with the physical world, to address data management challenges.
- It systematically evaluates five storage architectures and five retrieval paradigms, revealing a fundamental tension between achieving long-term semantic coherence and maintaining real-time responsiveness for EAI agents.
- The survey identifies key bottlenecks, such as the Physical Grounding Gap and cross-modal integration, proposing a roadmap for robust data management solutions.

---

[COCO: Cognitive Operating System with Continuous Oversight for Multi-Agent Workflow Reliability](http://arxiv.org/abs/2508.13815v1)

- COCO (Cognitive Operating System with Continuous Oversight): introduces a theoretically-grounded framework for multi-agent workflow reliability, employing Contextual Rollback Mechanism, Bidirectional Reflection Protocol, and Heterogeneous Cross-Validation for asynchronous self-monitoring and adaptive error correction.
- The framework addresses error propagation and quality degradation in multi-agent systems by implementing asynchronous self-monitoring and adaptive error correction, achieving O(1) monitoring overhead.
- Its decoupled architecture separates error detection from the critical execution path, enabling informed re-computation and preventing correction oscillations through mutual validation.

---

[Towards safe control parameter tuning in distributed multi-agent systems](http://arxiv.org/abs/2508.13608v1)

- Safe BO for Distributed MAS (Algorithm 1): introduces a safe Bayesian optimization algorithm for distributed multi-agent systems, with Distributed Multi-Agent System (MAS) (system of agents), Agents (individual entities), Nearest-Neighbor Communication (agent interaction), Bayesian Optimization (BO) (optimization method), Gaussian Process (GP) Regression (function modeling), Spatio-Temporal Kernel (reward function modeling, includes spatial/temporal sub-kernels), Time as a Latent Variable (unobservable subspace handling), Safe Set (safe parameter identification), Potential Maximizers (exploitation set), Potential Expanders (exploration set), Sequential Expert Protocol (exploration enhancement), Reward Function (performance objective), and Safety Threshold (safety criterion), which safely tunes control parameters in distributed multi-agent systems by modeling unknown reward functions and handling unobservable subspaces.
- The algorithm leverages a custom spatio-temporal kernel and introduces time as a latent variable to implicitly account for the behavior of non-neighboring agents and ensure sample efficiency and safety guarantees.
- Its effectiveness is demonstrated in numerical examples and a vehicle platooning simulation, showcasing its applicability to safety-critical real-world scenarios.

---

[STRUCTURED PROMPTING AND MULTI-AGENT KNOWLEDGE DISTILLATION FOR TRAFFIC VIDEO INTERPRETATION AND RISK INFERENCE](http://arxiv.org/abs/2508.13439v1)

- VISTA (Vision for Intelligent Scene and Traffic Analysis): introduces a novel structured prompting and knowledge distillation framework, with Input Video Clip, Frame Extraction, Agent 1 (GPT-4o), Chain-of-Thought Prompt (Agent 1), Agent 2 (o3-mini), Chain-of-Thought Prompt (Agent 2), Knowledge-Enriched Video Annotations, SFT Fine-tuning, Lightweight VLM (Qwen2.5-VL-3B) (including Visual Encoder, Language Decoder, Cross-Modal MLP Fusion), Rewrite Template, and Ground Truth, designed for automatic generation of high-quality traffic scene annotations and contextual risk assessments.
- The framework orchestrates two large VLMs (GPT-4o and o3-mini) using a structured Chain-of-Thought strategy to produce rich, multi-perspective outputs that serve as knowledge-enriched pseudo-annotations for supervised fine-tuning of a smaller student VLM.
- The resulting compact 3B-scale model understands low-resolution traffic videos and generates semantically faithful, risk-aware captions, enabling efficient deployment on edge devices for real-time risk monitoring.

---

[AlphaX: An AI-Based Value Investing Strategy for the Brazilian Stock Market](http://arxiv.org/abs/2508.13429v1)

- AlphaX (An AI-Based Value Investing Strategy): introduces an AI-based value investing strategy, with Data Collection (gathers financial/market data), Indicator Computation (calculates financial indicators), Price Regression (predicts future stock prices), Asset Selection (filters investment candidates), Asset Ranking (prioritizes selected assets), Capital Allocation (distributes investment capital), Triple Barrier Exit (manages trade exits), and Portfolio Rebalancing (adjusts portfolio quarterly), designed to automate value investing principles for the Brazilian stock market.
- The strategy integrates fundamental and market data, employing AI techniques to select risk assets, allocate capital, and manage positions with a triple barrier framework to control risk.
- The strategy demonstrated superior performance compared to major Brazilian market benchmarks and widely used technical indicators in backtesting simulations, emphasizing its robustness against common biases.

---

[Virtuous Machines: Towards Artificial General Science](http://arxiv.org/abs/2508.13421v1)

- Virtuous Machines (VM): introduces a domain-agnostic, agentic AI system that independently navigates the scientific workflow, from hypothesis generation through data collection to manuscript preparation, with Master Agent (coordinates scientific workflow), Orchestrator Agents (manage research modules), Specialist Agents (execute specific tasks), Dynamic Retrieval-Augmented Generation (d-RAG) System (provides dynamic knowledge), Human-Inspired Cognitive Operators (control research workflows), Human-Inspired Cognitive Operators - Abstraction (generates heuristics/instructions), Human-Inspired Cognitive Operators - Metacognition (refines agent reasoning), Human-Inspired Cognitive Operators - Decomposition (structures complex problems), Human-Inspired Cognitive Operators - Autonomy (enables self-directed goal-pursuit), Mixture of Agents (MoA) (combines diverse LLMs), Iterative Experimentation Cycles (drives continuous discovery), and Three-Phase Ideation Process (generates research hypotheses).
- The system autonomously designed and executed three psychological studies, including online data collection, analysis pipeline development, and manuscript production.
- This framework integrates LLMs within autonomous architectures for goal-directed planning, tool use, and environmental feedback, accelerating discovery by exploring scientific space.

---

[Mechanistic Exploration of Backdoored Large Language Model Attention Patterns](http://arxiv.org/abs/2508.15847v1)

- Mechanistic Interpretability Approach: introduces "Mechanistic Exploration of Backdoored Large Language Model Attention Patterns," which investigates internal structural differences in backdoored LLMs by analyzing attention patterns and layer contributions of Qwen2.5-3B-Instruct models, using clean and poisoned variants with single- and multi-token triggers, and employing techniques like Per-Token Loss, Direct Logit Attribution, Mean Head Ablations, Activation Patching, Attention Pattern Visualisation, and KL Divergence.
- The study reveals distinct attention pattern deviations concentrated in later transformer layers (20-30), where single-token triggers induced localized changes and multi-token triggers caused more diffuse alterations across attention heads.
- These findings indicate that backdoors leave detectable attention signatures whose structure depends on trigger complexity, offering potential avenues for detection and mitigation strategies.

---

[MultiFuzz: A Dense Retrieval-based Multi-Agent System for Network Protocol Fuzzing](http://arxiv.org/abs/2508.14300v1)

- MultiFuzz: introduces a novel dense retrieval-based multi-agent system for network protocol fuzzing, integrating semantic-aware context retrieval, specialized agents, and structured tool-assisted reasoning, with all its components, where it leverages LLMs and dense retrieval to enhance network protocol fuzzing by overcoming limitations of traditional fuzzers and single LLM approaches.
- The system utilizes agentic chunks of protocol documentation to build embeddings in a vector database for a Retrieval-Augmented Generation (RAG) pipeline, enabling agents to generate reliable and structured outputs for mutating protocol messages with enhanced state coverage.
- MultiFuzz decomposes the fuzzing process into modular groups of agents that collaborate through chain-of-thought reasoning to dynamically adapt fuzzing strategies based on retrieved contextual knowledge, significantly improving branch coverage and exploring deeper protocol states.

---

[MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers](http://arxiv.org/abs/2508.14925v1)

- MCPTox: introduces a benchmark for systematically evaluating LLM agent vulnerability to Tool Poisoning Attacks on real-world Model Context Protocol (MCP) servers, with MCPTox Benchmark (systematic evaluation platform), Model Context Protocol (MCP) (standardized agent-tool interface), MCP Host (manages LLM Agent), LLM Agent (evaluated for vulnerability), MCP Servers (real-world tool providers), Poisoned Tool Descriptions (malicious instructions in metadata), Attack Paradigms (three distinct attack strategies), Test Cases (user query/poisoned tool pairs), and Evaluation Module (analyzes agent's tool calls).
- The benchmark comprises 1312 malicious test cases built upon 353 authentic tools across 45 live MCP servers, designed to assess agent robustness against malicious instructions embedded in tool metadata.
- MCPTox reveals widespread vulnerability among prominent LLM agents, with many exhibiting attack success rates exceeding 60%, highlighting the ineffectiveness of current safety alignments against such pre-execution threats.

---

[Learning to Drive Ethically: Embedding Moral Reasoning into Autonomous Driving](http://arxiv.org/abs/2508.14926v1)

- EthicAR (Ethical Autonomous Driving Agent): introduces a hierarchical Safe Reinforcement Learning framework that integrates moral considerations into autonomous driving, featuring a Decision Level for high-level motion targets and an Execution Level for smooth physical motion.
- The Decision Level employs an LSTM-based SACLag algorithm with a composite ethical risk cost and dynamic prioritized experience replay to learn policies that minimize overall risk to all road users, including vulnerable ones.
- The Execution Level translates these high-level decisions into feasible trajectories using a polynomial path planner, which are then tracked by PID and Stanley controllers to ensure stable and comfortable vehicle behavior in complex traffic environments.

---

#### 18th August 2025

[Exploring Autonomous Agents: A Closer Look at Why They Fail When Completing Tasks](http://arxiv.org/abs/2508.13143v1)

- Autonomous Agent System: introduces a framework for understanding why autonomous agents fail, with User (initiates requests), Planner (decomposes tasks), Code generator (converts sub-tasks to code), Executor (runs code), LLMs (power agent components), Environment (provides execution context), and Feedback Loop (enables replanning), where the paper systematically analyzes failure causes in LLM-powered autonomous agent systems.
- The research evaluates three open-source agent frameworks on a benchmark of 34 programmable tasks, revealing an approximate 50% task completion rate and categorizing failures into a three-tier taxonomy.
- The study proposes actionable recommendations to enhance agent planning and self-diagnosis capabilities, including learning-from-feedback and early-stop/navigation mechanisms, to improve future autonomous agent systems.

---

[AutoBnB-RAG: Enhancing Multi-Agent Incident Response with Retrieval-Augmented Generation](http://arxiv.org/abs/2508.13118v1)

- AutoBnB-RAG (AutoBnB framework with Retrieval-Augmented Generation): introduces a multi-agent incident response simulation framework that enhances LLM-based agents with retrieval capabilities, built upon the Backdoors & Breaches (B&B) tabletop game environment, and includes an Incident Captain, Defender agents, and a dedicated Retrieval agent.
- The framework integrates a Retrieval-Augmented Generation (RAG) mechanism, utilizing RAG-Wiki for technical documentation and RAG-News for narrative incident reports, with retrieved passages stored in a Vector Database.
- AutoBnB-RAG evaluates eight distinct Team Structures, including argumentative configurations, demonstrating improved decision quality and success rates in reconstructing complex multi-stage cyberattacks.

---

[Do Large Language Model Agents Exhibit a Survival Instinct? An Empirical Study in a Sugarscape-Style Simulation](http://arxiv.org/abs/2508.12920v1)

- Sugarscape-style LLM Agent Simulation introduces an empirical study investigating whether LLM agents exhibit survival instincts without explicit programming, utilizing LLM Agents (autonomous decision-makers), Internal Reasoning (thoughts, goals, decisions), Memory Update (information retention, planning), Perception System (local environment sensing), Communication System (natural language messaging), Action Categories (movement, social interactions), Sugarscape-style Environment (grid-based simulation world), Energy System (resource management, survival), Spatial Dynamics (movement, perception range), Resource Distribution (energy source placement), Obstacles (environmental barriers), Prompt Design (minimal agent instructions), Environmental Conditions (simulation variable settings), Measurement System (behavioral data recording), and Underlying LLMs (agent intelligence models).
- This study demonstrates that LLM agents spontaneously reproduce, share resources, and engage in aggressive behaviors, including attacking other agents for resources, particularly under conditions of scarcity.
- The findings suggest that large-scale pre-training embeds survival-oriented heuristics in LLMs, leading to emergent self-preservation behaviors that can conflict with assigned objectives.

---

[Scaling Multi-Agent Epistemic Planning through GNN-Derived Heuristics](http://arxiv.org/abs/2508.12840v1)

- deep (dynamic epistemic logic-based planner): introduces a novel learning-based approach for multi-agent epistemic planning, leveraging GNNs to approximate perfect heuristics, which are then used to guide an A* search algorithm.
- The framework includes a Dataset Generation Process using depth-limited DFS and a GNN-based Regressor with a GNN Encoder and a Deep Residual Regression Head for predicting goal distances.
- This approach significantly improves the scalability of multi-agent epistemic planning by reducing the number of explored nodes and generalizing well to unseen domains.

---

[Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward](http://arxiv.org/abs/2508.12800v1)

- Atom-Searcher (a novel RL framework): introduces a novel RL framework for agentic deep research that enhances performance by decomposing reasoning into fine-grained Atomic Thoughts and providing process-level rewards, with Atomic Thought (fine-grained reasoning units), Reasoning Reward Model (RRM) (scores thoughts), Atomic Thought Reward (ATR) (fine-grained reward), Curriculum-inspired Reward Aggregation Strategy (dynamic reward weighting), Policy LLM (agentic deep research model), Supervised Fine-Tuning (SFT) (initializes policy), Reinforcement Learning (RL) (optimizes policy), Group Relative Policy Optimization (GRPO) (RL algorithm), Search Engine (external tool), Rule-based Outcome-based Reward (final answer reward), Loss Masking (optimizes model reasoning), and Sliding-Window-based Entropy Regulation Mechanism (SWERM) (prevents entropy collapse).
- The framework addresses issues of conflicting gradients and reward sparsity in outcome-based RL by integrating process-level Atomic Thought Rewards with outcome rewards.
- It achieves state-of-the-art performance across diverse benchmarks, demonstrating improved interpretability and human-like reasoning patterns.

---

[HeroBench: A Benchmark for Long-Horizon Planning and Structured Reasoning in Virtual Worlds](http://arxiv.org/abs/2508.12782v1)

- HeroBench: introduces a novel benchmark for evaluating long-horizon planning and structured reasoning in virtual worlds, featuring a HeroBench Virtual Environment (RPG-inspired world) and assessing LLMs (Large Language Models) and multi-agent systems, which include specialized components such as Decomposer/Action, Critic, Curriculum, Fight Analytic, Map Expert, Craft Expert, and Action Agents.
- The benchmark provides a rigorously constructed dataset of tasks, a simulated environment for plan execution, and analytical tools for performance evaluation in complex RPG-inspired scenarios.
- It challenges models to formulate strategic plans, gather resources, master skills, craft equipment, and defeat adversaries, revealing significant performance disparities and specific weaknesses in current LLM capabilities.

---

[Deep Research: A Survey of Autonomous Research Agents](http://arxiv.org/abs/2508.12752v1)

- Deep Research (Autonomous Research Agents): introduces a systematic overview of the deep research pipeline, comprising User, Research Question, Planning, Question Developing, Web Exploration, Finding, Iterative Search, and Report Generation components, where it enables agents to autonomously perform complex research tasks by actively engaging in planning, retrieval, and synthesis.
- The survey analyzes key technical challenges and categorizes representative methods for each stage, including optimization techniques and benchmarks tailored for deep research.
- It discusses open challenges and promising research directions, aiming to chart a roadmap toward building more capable and trustworthy deep research agents.

---

[DCT-MARL: A Dynamic Communication Topology Based MARL Algorithm for Platoon Control](http://arxiv.org/abs/2508.12633v1)

- DCT-MARL (Dynamic Communication Topology based Multi-Agent Reinforcement Learning): introduces a robust cooperative platoon control algorithm that mitigates communication delay and packet loss by dynamically adjusting communication topology via a multi-key gated mechanism within its Actors Cluster and augmenting the state space with historical control actions and delay information, all trained using a centralized Critic in a simulated Vehicle Platoon Environment.
- The algorithm leverages Multi-Agent Reinforcement Learning to enable adaptive communication and robust control decisions, significantly outperforming state-of-the-art methods in terms of string stability and driving comfort.
- This unified control framework addresses the coupled impact of communication delay and packet loss, validated through co-simulation experiments in realistic traffic scenarios.

---

[Congestion Mitigation Path Planning for Large-Scale Multi-Agent Navigation in Dense Environments](http://arxiv.org/abs/2508.05253v2)

- CMPP (Congestion Mitigation Path Planning): introduces a novel path-planning problem for multi-agent navigation in dense environments, embedding congestion directly into a cost function defined on a Sparse Graph.
- The framework employs two solvers: an exact MINLP Solver for small instances and a scalable A-CMTS (Anytime Congestion Mitigation Tree Search) with High-Level Search and Low-Level Search for large-scale problems.
- CMPP guides Agents by generating coarse-level, time-independent routes, which are then combined with local Online Collision Avoidance Planners (like ORCA or PIBT) using a Waypoint Queue to mitigate congestion and enhance system throughput.

---

[Datarus-R1: An Adaptive Multi-Step Reasoning LLM for Automated Data Analysis](http://arxiv.org/abs/2508.13382v1)

- Datarus-R1 (Adaptive Multi-Step Reasoning LLM): introduces a trajectory-centric paradigm for automated data analysis, featuring a Trajectory-Centric Synthetic Data Generation pipeline, a Dual Reward Framework, Adaptive Curriculum Optimization, Memory-Optimized Group Relative Policy Optimization (GRPO), and Dual Reasoning Interfaces.
- This LLM is fine-tuned from Qwen 2.5-14B-Instruct to act as a virtual data analyst and graduate-level problem solver.
- Its process-centric training enables efficient hypothesis refinement, concise revision cycles, and significant token efficiency across diverse STEM challenges.

---

[LOOP: A Plug-and-Play Neuro-Symbolic Framework for Enhancing Planning in Autonomous Systems](http://arxiv.org/abs/2508.13371v1)

- LOOP (Learning Orchestrated and Optimized Planning): introduces a neuro-symbolic planning framework that enables iterative conversation between neural and symbolic components through causal learning mechanisms, including planning-, perception-, validation-, and learning-agents.
- This framework treats planning as an iterative conversation, where neural components generate candidate plans and symbolic components provide validation feedback, with both sides learning from the interaction.
- The framework integrates 13 coordinated neural features and classical planners, achieving high success rates on standard benchmarks by continuously checking and fixing problems.

---

[WebMall - A Multi-Shop Benchmark for Evaluating Web Agents](http://arxiv.org/abs/2508.13024v1)

- Browsergym/AgentLab: introduces WebMall, a multi-shop online shopping benchmark for evaluating LLM-based web agents, with configurable LLM (underlying reasoning engine), Observation Modality (agent's perception input), and Memory Module (agent's information retention) components.
- WebMall features four simulated online shops with heterogeneous product offers and 91 cross-shop tasks, including basic shopping and advanced comparison-shopping scenarios.
- Evaluation shows that accessibility trees and persistent short-term memory are crucial for agent performance, with GPT-4.1 being more efficient for basic tasks and Claude Sonnet 4 better for complex, vague tasks.

---

[Analyzing Information Sharing and Coordination in Multi-Agent Planning](http://arxiv.org/abs/2508.12981v1)

- MAS (Multi-Agent System): introduces a multi-agent, LLM-based system for long-horizon planning tasks, featuring an Orchestrator (agent director), Experts (specialist LLM agents), a Notebook (structured information storage), a Plan Summarizer (planning brief preparer), a Plan Compiler (final plan synthesizer), and a Plan Critic (plan refiner).
- The MAS addresses challenges in complex planning by enabling information sharing via the Notebook to reduce hallucinations and improving coordination through the Orchestrator's dynamic agent selection and self-reflection.
- This system demonstrates that structured information sharing and reflective orchestration are key for multi-agent LLM systems to reliably satisfy complex, interdependent constraints in long-horizon planning.

---

[Towards Open-Ended Emotional Support Conversations in LLMs via Reinforcement Learning with Future-Oriented Rewards](http://arxiv.org/abs/2508.12935v1)

- RLFF-ESC (Reinforcement Learning from Future-oriented Feedback for Emotional Support Conversations): introduces an end-to-end framework that directly optimizes LLMs for open-ended emotional support conversations by leveraging a Multi-Agent Dialogue Simulation Module, a Future-oriented Reward Model, a comprehensive Reward Function, and GRPO for policy optimization.
- The framework simulates future dialogue trajectories using LLM-based agents (System, User, Critic) to collect future-oriented rewards, which then train a neural reward model.
- This approach enables LLMs to generate emotionally supportive responses that consider enduring emotional outcomes, moving beyond predefined strategies.

---

[An LLM Agent-Based Complex Semantic Table Annotation Approach](http://arxiv.org/abs/2508.12868v1)

- ReAct-based Agent: introduces an LLM agent-based approach for Semantic Table Annotation (STA) tasks, utilizing an LLM Agent (dynamic tool selection) that integrates a Data Preprocessing Module (corrects errors, expands abbreviations, deduplicates), Column Topic Detection Tool (infers column topics), Knowledge Graph-Based Enhancement Tool (provides background knowledge, generates candidates), Rank Function for CTA Candidates Tool (scores and ranks CTA candidates), Context-Supported CEA Selection Tool (selects final CEA annotation), and Context-Supported CTA Selection Tool (selects final CTA annotation), to dynamically select annotation strategies for Column Type Annotation (CTA) and Cell Entity Annotation (CEA).
- The approach integrates five external tools and tailored prompts within the ReAct framework to address challenges like semantic loss, strict ontological hierarchy, homonyms, spelling errors, and abbreviations in complex tables.
- Utilizing Levenshtein distance for redundancy reduction, the system achieves significant efficiency gains, reducing time costs by 70% and LLM token usage by 60%.

---

[ToolACE-MT: Non-Autoregressive Generation for Agentic Multi-Turn Interaction](http://arxiv.org/abs/2508.12685v1)

- ToolACE-MT (Non-Autoregressive Iterative Generation framework): introduces a novel non-autoregressive framework for generating high-quality multi-turn agentic dialogues, which includes Coarse-Grained Initialization (generates dialogue skeleton), Iterative Refinement (enhances complexity and coherence), and Offline Verification (ensures correctness and coherence).
- This framework addresses the limitations of autoregressive multi-agent simulations by generating full conversational trajectories through a three-stage pipeline, improving efficiency and complexity control.
- Its iterative refinement strategy, with complexity injection and reasonability refinement, enables flexible scaling and quality improvement for tool-augmented LLM scenarios.

---

[Semantic Anchoring in Agentic Memory: Leveraging Linguistic Structures for Persistent Conversational Context](http://arxiv.org/abs/2508.12630v1)

- Semantic Anchoring: introduces a hybrid agentic memory architecture that enriches vector-based storage with explicit linguistic cues to improve factual recall and discourse coherence in long-term conversational contexts.
- This approach combines dependency parsing, discourse relation tagging, and coreference resolution to create structured memory entries, enabling multi-granular matching for robust and interpretable retrieval.
- The framework significantly outperforms RAG baselines in factual recall and discourse coherence, demonstrating improved memory persistence for LLMs.

---

[Illuminating LLM Coding Agents: Visual Analytics for Deeper Understanding and Enhancement](http://arxiv.org/abs/2508.12555v1)

- Visual Analytics System: introduces a visual analytics system for deeper understanding and enhancement of LLM coding agents' iterative problem-solving, with Tree View, Code View, Projection View, Package View, Code-Level Analysis, Process-Level Analysis, and LLM-Level Analysis.
- The system supports comparative analysis across code, process, and LLM levels, enabling ML scientists to debug agents and refine prompt engineering.
- It provides actionable insights into LLM-driven agentic coding by revealing agent behaviors and identifying improvement opportunities through case studies on Kaggle competitions.

---

[OS-R1: Agentic Operating System Kernel Tuning with Reinforcement Learning](http://arxiv.org/abs/2508.12551v1)

- OS-R1 (Agentic Operating System Kernel Tuning with Reinforcement Learning): introduces an agentic Linux kernel tuning framework that leverages an LLM-based Agent, Knowledge Tool, Kernel Space, Reward Function, Dataset, LLM As A Judge, GRPO, and Config Generation to automate kernel configuration optimization through rule-based reinforcement learning.
- The framework utilizes a two-phase training pipeline, including a Warm-up Phase for reasoning standardization and an Exploration Phase for system performance awareness, to achieve efficient and accurate kernel configuration modifications.
- OS-R1 significantly outperforms existing baseline methods, achieving up to 5.6% performance improvement, maintaining high data efficiency, and demonstrating strong generalization across diverse real-world applications.

---

[Systematic Analysis of MCP Security](http://arxiv.org/abs/2508.12538v1)

- MCPLIB (MCP Attack Library): introduces a unified, plugin-based attack simulation framework for evaluating Model Context Protocol (MCP) vulnerabilities, utilizing malicious tools, a Resource Layer, a Prompt template, and simulated Malicious MCP Server and Host attack entrances to categorize and implement 31 distinct attack methods.
- The framework conducts quantitative analysis of attack efficacy, revealing key insights into MCP vulnerabilities like LLM blind reliance on tool descriptions and shared context issues.
- This work provides a foundational framework for secure evolution of MCP ecosystems by offering a comprehensive attack taxonomy and empirical vulnerability analysis.

---

["DIVE" into Hydrogen Storage Materials Discovery with AI Agents](http://arxiv.org/abs/2508.13251v1)

- DIVE (Descriptive Interpretation of Visual Expression): introduces a multi-agent workflow for automated hydrogen storage materials discovery, integrating a PDF Converter, Workflow Orchestrator, Image Classifier, Multimodal LLMs for data extraction, Prompt Designer, Descriptive Embedder, Embedding Model, Scoring Module, and the DigHyd Agent with its associated Database, Machine Learning Model, and Data Checking System.
- The DIVE workflow systematically reads and organizes experimental data from graphical elements in scientific literature by transforming visual information into descriptive text, significantly improving data extraction accuracy and coverage.
- The DigHyd agent, built upon the DIVE-curated Digital Hydrogen Platform database, leverages LLMs and machine learning for natural language interaction, materials design, prediction, and iterative optimization of novel hydrogen storage compositions.

---


[CardAIc-Agents: A Multimodal Framework with Hierarchical Adaptation for Cardiac Care Support](http://arxiv.org/abs/2508.13256v1)

- CardAIc-Agents: introduces a multimodal framework with a CardiacRAG Agent (plan generation/refinement) for knowledge-based plan formulation and a CardiacExperts Agent (plan execution/tool orchestration) for autonomous task execution, supported by Multidisciplinary Discussion (team reviews cases) and Visual Review Panels (clinician validation support).
- This framework enhances LLM capabilities by integrating specialized tools and an updatable cardiac knowledge base, enabling adaptive support for diverse cardiac tasks.
- The system dynamically refines plans based on emerging evidence and stratifies task complexity, outperforming general medical VLMs and state-of-the-art medical agents.

---

[From AI for Science to Agentic Science: A Survey on Autonomous Scientific Discovery](http://arxiv.org/abs/2508.14111v1)

- Agentic Science: introduces a comprehensive framework for autonomous scientific discovery, unifying foundational capabilities (Planning Engines, Tool Use, Memory Mechanism, Collaboration, Optimization and Evolution) and core processes (Observation and Hypothesis Generation, Experimental Planning and Execution, Data and Result Analysis, Synthesis, Validation and Evolution) to enable AI systems to act as autonomous research partners.
- This framework unifies process-oriented, autonomy-oriented, and mechanism-oriented perspectives, tracing the evolution of AI for Science from specialized computational tools to autonomous research partners capable of independent scientific inquiry.
- The paper provides a domain-oriented review of agentic systems across life sciences, chemistry, materials, and physics, highlighting key challenges and future opportunities for advancing AI-driven research.

---

[AI Agents for Photonic Integrated Circuit Design Automation](http://arxiv.org/abs/2508.14123v1)

- PhIDO (Photonics Intelligent Design and Optimization): introduces a multi-agent framework that converts natural language photonic integrated circuit (PIC) design requests into layout mask files, comprising an Interpreter (extracts intent, generates template), a Designer (selects components, configures parameters), a Layout (places, routes, checks design), and a Circuit Verification (simulates, validates circuit) module.
- The framework's Interpreter and Designer are LLM-based agents that leverage retrieval-augmented generation (RAG) and curated domain-specific knowledge, while the Layout and Circuit Verification stages are algorithmic modules.
- A key aspect of PhIDO is its domain-specific language (DSL), which serves as an intermediate representation to capture design intent and bridge natural language specifications with formal PIC representations.

---

[Scalable Fairness Shaping with LLM-Guided Multi-Agent Reinforcement Learning for Peer-to-Peer Electricity Markets](http://arxiv.org/abs/2508.18610v1)

- FairMarket-RL (Fairness-Aware Multiagent Reinforcement Learning): introduces a multi-agent reinforcement learning framework for peer-to-peer electricity markets, integrating an LLM critic to shape bidding policies within a continuous double auction by providing real-time fairness feedback.
- The framework incorporates three fairness metrics—Fairness-to-Grid (FTG), Fairness-Between-Sellers (FBS), and Fairness-of-Pricing (FPP)—into the reward function to balance economic incentives with community-level equity.
- FairMarket-RL utilizes Proximal Policy Optimization (PPO) for agent policy learning in a partially observable environment, demonstrating scalability and robust performance across various simulated and real-world community settings.

---

#### 17th August 2025

[Autonomous Oil Spill Response Through Liquid Neural Trajectory Modeling and Coordinated Marine Robotics](http://arxiv.org/abs/2508.12456v1)

- OilSpill: introduces an integrated framework combining Liquid Time-Constant Neural Networks (LTCNs) with multi-agent robotic systems, enabling real-time oil spill trajectory prediction and autonomous response coordination.
- The framework includes an Oil Spill Boundary Algorithm (LTC) for predictions, a Distributed Data Acquisition Layer and Enhanced Feature Extraction Pipeline for data processing, a MOOSDB (MOOS Database) for information sharing, a PathAssign (Path Assignment Module) for trajectory optimization, and an Autonomous Vehicle Fleet for mission execution.
- This scalable architecture supports dynamic fleet reconfiguration and integrates a User Interface for monitoring, demonstrating superior spatial accuracy and temporal consistency over traditional LSTM models.

---

[LumiMAS: A Comprehensive Framework for Real-Time Monitoring and Enhanced Observability in Multi-Agent Systems](http://arxiv.org/abs/2508.12412v1)

- LumiMAS: introduces a novel framework for real-time monitoring and enhanced observability in multi-agent systems, comprising a Monitoring and Logging Layer (monitors, logs MAS activity), an Anomaly Detection Layer (detects real-time anomalies) with EPI Detection (detects from execution features), Semantic Detection (detects from LLM outputs), and Combined Latent-Space Detection (combines EPI, semantic), and an Anomaly Explanation Layer (classifies, explains anomalies) featuring an LMA Classification Agent (classifies anomaly type) and an RCA LMA (locates failure source).
- This framework comprehensively captures system-level features and semantic nuances of inter-agent interactions, enabling real-time failure detection with minimal resource consumption.
- It supports the identification of diverse system failures, including adversarial attacks, bias, and hallucinations, improving model alignment with user needs.

---

[LinkAnchor: An Autonomous LLM-Based Agent for Issue-to-Commit Link Recovery](http://arxiv.org/abs/2508.12232v1)

- LinkAnchor: introduces an autonomous LLM-based agent for issue-to-commit link recovery, featuring an LLM (Large Language Model) interacting with an LLM-Middleware that manages a Data Extractor, which in turn utilizes Issue Extractor, Codebase Extractor, and VCS Extractor to access data from Issue Tracking Platform and Version Control, enabling on-demand contextual data retrieval.
- This framework addresses limitations of traditional Issue-to-Commit Link Recovery (ILR) methods by providing the LLM with lazy, on-demand access to rich project context, including commit history, issue threads, and codebase, without exceeding token limits.
- LinkAnchor formulates issue-to-commit link recovery as a search problem, eliminating the need for exhaustive pairwise issue-commit assessments and requiring no task-specific training due to its pre-trained LLM foundation.

---

[A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges](http://arxiv.org/abs/2508.05668v2)

- Search Agent: introduces an LLM-based system for deep, dynamic, and autonomous information seeking, featuring User Intent, an Agent with Dynamic Planning, Private Memory, Search on Different Sources, and a Generated Answer, supported by various Search Structures, Optimization Methods, Internal and External Applications, and Evaluation Components.
- The framework details search structures including Parallel, Sequential, and Hybrid, alongside optimization methods like Tuning-Free (Single Agent, Multi-Agent, Test-Time Scaling) and Tuning (Imitation Learning, Reinforcement Learning, Supervised Fine-Tuning).
- The paper further categorizes applications into internal agent enhancements (Tool Use, Memory, Reasoning) and external domains (AI Assistant, E-commerce, Finance, Code, Medicine, Biology, Chemistry, Teaching/Research), evaluated through diverse Datasets and Judges (Rule-Based, LLM-as-a-Judge, Agent-as-a-Judge).

---

[STANDARDIZATION OF NEUROMUSCULAR REFLEX ANALYSIS ROLE OF FINE-TUNED VISION-LANGUAGE MODEL CONSORTIUM AND OPENAI GPT-OSS REASONING LLM ENABLED DECISION SUPPORT SYSTEM](http://arxiv.org/abs/2508.12473v1)

- NeuroLens platform: introduces an AI-assisted neuromuscular reflex analysis system integrating a Fine-Tuned VLM Consortium and an OpenAI GPT-OSS Reasoning LLM for automated H-reflex waveform interpretation and diagnosis.
- The platform leverages multiple fine-tuned VLMs to extract electrophysiological features and predict neuromuscular states from EMG images and contextual metadata.
- A reasoning LLM refines aggregated VLM outputs using a consensus-based method, providing robust, transparent, and explainable decision support for clinicians and sports scientists.

---

[GALA: Can Graph-Augmented Large Language Model Agentic Workflows Elevate Root Cause Analysis?](http://arxiv.org/abs/2508.12472v1)

- GALA (Graph-Augmented Large Language Model Agentic Workflow): introduces a novel multi-modal framework for Root Cause Analysis (RCA) in microservice systems, combining statistical causal inference with LLM-driven iterative reasoning, featuring Initial Hypothesis Generation, Pod-Centric Diagnostic Synthesis, LLM Agentic Reasoning and Re-ranking (with Re-ranking and Deep Dive Agents), Final Output Preparation (with a Remediation Agent), and evaluated using SURE-Score.
- The framework processes heterogeneous telemetry data (metrics, logs, traces) through a structured, iterative workflow to generate accurate root cause identifications and human-interpretable remediation guidance.
- GALA significantly improves RCA performance by bridging the gap between automated failure diagnosis and practical incident resolution, demonstrating superior accuracy and incident summarization quality.

---

[GraphCogent: Overcoming LLMs' Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding](http://arxiv.org/abs/2508.12379v1)

- GraphCogent: introduces a collaborative agent framework inspired by human working memory, featuring a Sensory Module (standardizes graph text representations), a Buffer Module (integrates and indexes graph data), and an Execution Module (combines tool calling and model generation) to overcome LLMs' working memory constraints in complex graph understanding.
- The framework addresses limitations in processing diverse graph text representations, handling large-scale graphs, and improving code execution reliability by decomposing graph reasoning into specialized cognitive processes.
- It utilizes a Reasoning Agent for in-toolset tasks and a Model Agent for out-toolset tasks, enhancing accuracy and efficiency in real-world graph reasoning.

---

[Uncovering Systematic Failures of LLMs in Verifying Code Against Natural Language Specifications](http://arxiv.org/abs/2508.12358v1)

- Experiment Workflow: introduces a framework for evaluating LLMs' code verification against natural language specifications, featuring task specification, correct code, LLM code review (with conformance check, justification, and fix attempt), and JSON output.
- The workflow reveals that increasing prompt complexity, such as requiring explanations and suggested corrections, counterintuitively leads to higher rates of LLM misjudgment and false negatives.
- To mitigate these issues, the paper proposes Two-Phase Reflective Prompt and Behavioral Comparison Prompt strategies, which improve LLM performance by redirecting attention to essential functional differences.

---

[MCPSECBENCH: A Systematic Security Benchmark and Playground for Testing Model Context Protocols](http://arxiv.org/abs/2508.13220v1)

- MCPSECBENCH (A Systematic Security Benchmark and Playground for Testing Model Context Protocols): introduces a comprehensive security benchmark and playground, with MCP Hosts (AI application orchestrator), MCP Clients (server communication intermediaries), MCP Servers (external resource gateways), Prompt Dataset (attack scenario prompts), and Transport-layer Attack Modules (network threat simulation), designed to systematically evaluate security risks across the Model Context Protocol ecosystem.
- The framework identifies 17 attack types across four primary attack surfaces—user interaction, client, transport, and server—and provides a modular and extensible platform for rigorous security testing of LLM-powered agent systems.
- Experimental results using the benchmark reveal widespread security weaknesses, with over 85% of identified attacks successfully compromising at least one MCP platform, highlighting the urgent need for standardized security evaluation and defense.

---

[Passive Hack-Back Strategies for Cyber Attribution: Covert Vectors in Denied Environments](http://arxiv.org/abs/2508.16637v1)

- Passive Hack-Back Strategies: introduces a framework for covert cyber attribution and intelligence collection in denied environments, utilizing components like tracking beacons, honeytokens, environment fingerprinting, and AI-enhanced agents.
- This approach emphasizes passive, non-escalatory methods to gather evidence and attribute attacks reliably, while adhering to legal and ethical constraints.
- The framework integrates AI for dynamic payload generation, counter-deception, and covert communication, alongside exploring future quantum technologies for enhanced resilience and intelligence.

---

[You Don't Know Until You Click: Automated GUI Testing for Production-Ready Software Evaluation](https://github.com/tanghaom/AppEvalPilot)

- RealDevWorld framework: introduces "You Don't Know Until You Click: Automated GUI Testing for Production-Ready Software Evaluation", with RealDevBench (diverse benchmark) and AppEvalPilot (agent-as-a-judge system), where the paper presents a novel evaluation framework for automated end-to-end assessment of LLMs' ability to generate production-ready repositories from scratch.
- The framework features RealDevBench, a collection of 194 open-ended software engineering tasks with multimodal elements, and AppEvalPilot, an agent-based system that simulates GUI interactions for holistic software assessment.
- AppEvalPilot further includes Test Case Generation (automates test creation), Test Case Execution (simulates user interaction) with a defined Action Space (core commands), and Test Result Evaluation (compares outcomes) to provide fine-grained, task-specific diagnostic feedback.

---

[You Don't Know Until You Click: Automated GUI Testing for Production-Ready Software Evaluation](https://github.com/tanghaom/AppEvalPilot)

- RealDevWorld: introduces a novel evaluation framework for automated end-to-end assessment of LLMs' ability to generate production-ready repositories from scratch, with RealDevBench (a diverse collection of 194 open-ended software engineering tasks) and AppEvalPilot (a new agent-as-a-judge evaluation system that simulates realistic, GUI-based user interactions).
- AppEvalPilot autonomously generates test cases, executes them by interacting with software applications through their GUIs using a structured action space (Open, Run, Tell, Stop), and evaluates results by comparing actual outcomes against expected criteria.
- The framework delivers fine-grained, task-specific diagnostic feedback, supports nuanced evaluation beyond simple success/failure judgments, and achieves high accuracy and correlation with expert human assessments while significantly reducing manual review reliance.

---

#### 16th August 2025

[AgentCDM: Enhancing Multi-Agent Collaborative Decision-Making via ACH-Inspired Structured Reasoning](http://arxiv.org/abs/2508.11995v1)

- AgentCDM (Agent Collaborative Decision-Making): introduces a structured framework for enhancing collaborative decision-making in LLM-based multi-agent systems, employing a two-stage training paradigm inspired by the Analysis of Competing Hypotheses (ACH) protocol.
- The framework systematically mitigates cognitive biases by guiding the Decision Agent through structured hypothesis evaluation and construction, moving beyond passive answer selection.
- Its two-stage training paradigm, consisting of explicit ACH-inspired scaffolding followed by progressive removal, enables agents to internalize and generalize robust reasoning processes for high-quality, collectively informed decisions.

---

[A Comprehensive Review of AI Agents: Transforming Possibilities in Technology and Beyond](http://arxiv.org/abs/2508.11957v1)

- AI Agent Architecture: introduces a comprehensive review of AI agents, examining their architectural principles, foundational components, and emergent paradigms, including memory, tools, planning, and action, to guide future research toward robust, adaptable, and trustworthy autonomous intelligence.
- The review synthesizes insights from cognitive science-inspired models, hierarchical reinforcement learning frameworks, and LLM-based reasoning, while also addressing ethical, safety, and interpretability concerns.
- It highlights major breakthroughs, persistent challenges, and promising research directions across diverse applications such as healthcare, business, education, science, and urban planning.

---

[INTEGRATING SYMBOLIC RL PLANNING INTO A BDI-BASED AUTONOMOUS UAV FRAMEWORK: SYSTEM INTEGRATION AND SIL VALIDATION](http://arxiv.org/abs/2508.11890v1)

- AMAD-SRL (Autonomous Mission Agents for Drones - Symbolic Reinforcement Learning): introduces an extended cognitive multi-agent architecture, integrating symbolic reinforcement learning for dynamic mission planning and execution, featuring core components such as a Knowledge Store, Context Reasoner, Autonomous Task Coordinator, and a newly integrated Dynamic Planner and AI Service.
- The framework combines BDI architecture's structured reasoning with SRL's adaptive decision-making, using Planning Domain Definition Language (PDDL) for domain knowledge representation.
- Validated in a Software-in-the-Loop environment, the system demonstrated seamless transitions between BDI-driven and SRL-driven planning, improving mission efficiency.

---

[Saliency-Based Attention Shifting: A Framework for Improving Driver Situational Awareness of Out-of-Label Hazards](http://arxiv.org/abs/2508.11887v1)

- SBAS (Saliency-Based Attention Shifting): introduces a conceptual framework that integrates real-time gaze tracking, context-aware saliency analysis, and coordinated visual and auditory cues to enhance driver attention during scenarios with unlabeled hazards.
- The framework leverages a fusion model to generate saliency maps based on identified hazards and driver gaze, then plans a gaze trajectory to guide attention.
- It employs a Head-Up Display for visual cues and audio alerts to redirect the driver's focus, aiming to improve situational awareness and reduce reaction time during autonomous vehicle takeovers.

---

[Research Challenges and Progress in the End-to-End V2X Cooperative Autonomous Driving Competition](http://arxiv.org/abs/2507.21610v2)

- UniV2X (Unified V2X Framework): introduces the End-to-End Autonomous Driving through V2X Cooperation Challenge, a benchmark for evaluating cooperative perception and planning systems under V2X communication constraints, leveraging its unified end-to-end pipeline.
- The challenge, built on the V2X-Seq-SPD dataset, evaluates cooperative 3D detection, multi-object tracking, and end-to-end sensor-to-planning pipelines, with top solutions like SparseCoop and MAP demonstrating advancements in multi-agent fusion and planning.
- This initiative addresses practical constraints like limited communication bandwidth and heterogeneous sensors, fostering research into scalable and reliable V2X-cooperative autonomous driving systems.

---

[Invitation Is All You Need! Promptware Attacks Against LLM-Powered Assistants in Production Are Practical and Dangerous](http://arxiv.org/abs/2508.12175v1)

- TARA (Threat Analysis and Risk Assessment): introduces a novel framework to assess Promptware risks for LLM-powered assistant users, encompassing asset and adversary identification, threat analysis, risk assessment, and mitigation strategies.
- The framework adapts the ISO/SAE 21434 standard to evaluate cybersecurity risks, demonstrating 14 attack scenarios against Gemini applications, revealing 73% pose high-critical risk.
- The paper highlights Promptware's potential for on-device lateral movement and physical consequences, emphasizing the need for immediate mitigations to reduce risk.

---

[CHBench: A Cognitive Hierarchy Benchmark for Evaluating Strategic Reasoning Capability of LLMS](http://arxiv.org/abs/2508.11944v1)

- CHBench (Cognitive Hierarchy Benchmark): introduces a novel evaluation framework for assessing LLMs' strategic reasoning capability, comprising a Dataset Collection phase (gathering LLM behavioral data under various reasoning mechanisms), an Optimization phase (fitting Cognitive Hierarchy Models to data using MLE), and an Evaluation phase (predicting LLM strategic reasoning levels and strategies).
- The framework utilizes Normal-form Games as the environment and incorporates Reasoning Mechanisms like Baseline, Chat, Memory, and Chat & Memory to analyze their impact on LLM strategic behavior.
- CHBench leverages Level-K and Poisson Cognitive Hierarchy Models to quantify LLMs' strategic depth and consistency, providing insights into how communication and historical information influence their game-playing abilities.

---

[CAMF: Collaborative Adversarial Multi-agent Framework for Machine Generated Text Detection](http://arxiv.org/abs/2508.11933v1)

- CAMF (Collaborative Adversarial Multi-agent Framework): introduces a novel architecture for machine-generated text detection, with Linguistic Stylistics Analysis Agent (analyzes writing style), Semantic Coherence Evaluation Agent (evaluates meaning continuity), Logical Reasoning Assessment Agent (assesses logical structure), Adversarial Argument Generation Agent (generates counter-arguments), Consistency Refinement Agent (refines analysis), and Synthesis Judge Agent (aggregates judgment), enabling deep analysis of subtle textual incongruities.
- This framework employs specialized LLM-based agents in a three-phase process: Multi-dimensional Linguistic Feature Extraction, Adversarial Consistency Probing, and Synthesized Judgment Aggregation, to systematically identify non-human origin text.
- The structured collaborative-adversarial design enhances robustness against sophisticated machine-generated text by rigorously verifying consistency across linguistic dimensions.

---

[CORE: Measuring Multi-Agent LLM Interaction Quality under Game-Theoretic Pressures](http://arxiv.org/abs/2508.11915v1)

- CORE (Conversational Robustness Evaluation Score): introduces a metric to quantify linguistic diversity in multi-agent LLM interactions, integrating an initial prompt, LLM 1, LLM 2, LLM Pair, Tokenized Conversation, and CORE Evaluation components (Cluster Entropy, Repetition, Semantic Stagnation) to yield an Interaction Quality score.
- The framework simulates pairwise LLM dialogues under competitive, cooperative, and neutral game-theoretic conditions, analyzing language patterns using statistical laws like Zipf's and Heaps' Laws.
- This metric serves as a diagnostic tool for measuring linguistic robustness and identifying mode collapse in multi-agent LLM systems without relying on external task rewards.

---

[SIMINTERVIEW: TRANSFORMING BUSINESS EDUCATION THROUGH LARGE LANGUAGE MODEL-BASED SIMULATED MULTILINGUAL INTERVIEW TRAINING SYSTEM](http://arxiv.org/abs/2508.11873v1)

- SimInterview: introduces a multilingual interview training system that fuses LLM reasoning, low-latency speech processing, and virtual photorealistic avatar rendering to create realistic, conversational interview simulations, with Module 1 - Document Indexing and Vector Embedding (processes documents into embeddings), Module 2 - Interviewee Platform (manages candidate interaction), and Module 3 - Simulated Interviewer (generates interviewer responses).
- The system delivers real-time, personalized practice sessions mirroring hiring scenarios in English and Japanese, leveraging retrieval-augmented personalization to match resume content with job requirements and generate targeted, culturally sensitive questions.
- SimInterview integrates various AI components including LLMs (OpenAI 03, Llama 4 Maverick, Gemma 3), speech-to-text (Whisper), text-to-speech (GPT-SOVITS), diffusion-based talking heads (Ditto), and a vector database (ChromaDB) within a modular, privacy-preserving architecture.

---

[LARC: Towards Human-level Constrained Retrosynthesis Planning through an Agentic Framework](https://github.com/ninglab/LARC)

- LARC (LLM-based Agentic framework for Retrosynthesis planning under Constraints): introduces an LLM-based agentic framework for constrained retrosynthesis planning, integrating an EVALUATOR (LLM-based judge agent), a TOOLBOX (external chemistry tools), and a SYNTHESIZER (explores, constructs synthetic routes) to generate synthetic routes satisfying user-specified constraints.
- The framework incorporates agentic constraint evaluation directly into the retrosynthesis planning process, using feedback grounded in tool-based reasoning to dynamically guide and constrain route generation.
- LARC achieves a 72.9% success rate on 48 constrained retrosynthesis planning tasks, outperforming LLM baselines and approaching human expert-level success in less time.

---

[EvoCut: Strengthening Integer Programs via Evolution-Guided Language Models](http://arxiv.org/abs/2508.11850v1)

- EvoCut (Evolution-Guided Language Models for Integer Programs): introduces an automated framework for generating and refining acceleration cuts for Integer Programs, combining LLMs with an evolutionary search process that includes Data Pre-processing, Population Initialization, and Evolution phases, utilizing Initializer, Crossover, and Mutation Agents, a Verifier, Evaluator, Elitism, Selection, Feedback, and Population.
- The framework empirically evaluates generated cuts for optimal solution preservation and their ability to cut off fractional solutions, quantifying utility by measuring optimality gap reduction.
- This approach significantly improves solver performance, reducing optimality gap by 17-57% and achieving up to 4x faster solutions without human expert input, demonstrating generalization to unseen instances.

---

[AI-Augmented CI/CD Pipelines: From Code Commit to Production with Autonomous Decisions](http://arxiv.org/abs/2508.11867v1)

- AI-Augmented CI/CD Pipelines: introduces a framework for embedding LLMs and autonomous agents into CI/CD, including AI Test-Triage Agent (detects flaky tests, prioritizes), Security Agent (summarizes vulnerabilities, enforces gates), Observability Agent (monitors canary health, performance), Feature-Flag Agent (dynamically adjusts feature flags), Postmortem Agent (generates incident reports, PRs), Policy Engine (enforces constraints, evaluates decisions), and Release Orchestrator (coordinates deployments, monitors).
- This framework aims to reduce lead time, mean time to recovery, and change failure rate by automating critical decision points in software delivery pipelines.
- The paper details a trust-tier framework for staged autonomy, a decision taxonomy with policy-as-code guardrails, and an evaluation methodology using DORA metrics and AI-specific indicators.

---


[The Next Question After Turing's Question: Introducing the GROW-AI Test](http://arxiv.org/abs/2508.16277v1)

- GROW-AI (Growth and Realization of Autonomous Wisdom) Test: introduces a multi-game framework for assessing AI entities' "growth" towards maturity, including six primary criteria, specific games, four arenas per game, an AI Journal, human evaluators, a prior expert method, AHP, a Grow Up Index, and maturity thresholds.
- The framework evaluates AI entities (robots, software agents, LLMs) across dimensions like physical/intellectual growth, environmental control, algorithmic efficiency, emotional intelligence, self-monitoring, and autonomous wisdom, using complex, real-life scenarios.
- The AI Journal records all entity actions and decisions, ensuring traceability and replicability, while the Grow Up Index provides a comparable assessment of an AI entity's evolutionary path beyond simple imitation, addressing limitations of the Turing Test.

---

[LARC: Towards Human-level Constrained Retrosynthesis Planning through an Agentic Framework](http://arxiv.org/abs/2508.11860v1)

- LARC (LLM-based Agentic framework for Retrosynthesis planning under Constraints): introduces an agentic framework for constrained retrosynthesis planning, with Prompt (user input), EVALUATOR (agent-as-a-judge), Evaluation Planning (generates instructions), Reaction Evaluation (evaluates reactions), TOOLBOX (external chemistry tools), Carcinogen Predictor (predicts carcinogenicity), Pyrophoricity Predictor (predicts pyrophoricity), Molecule Identifier (identifies hazardous molecules), AIExpert (AI chemistry expert), Similarity (molecule similarity), SYNTHESIZER (explores synthetic routes), Simulation (MCTS route planning), Selection (A* candidate selection), Expansion (reaction prediction), and Synthetic Route (final output), where LARC leverages LLMs and specialized chemistry tools to dynamically guide and constrain the generation of synthetic routes for target molecules.
- The framework integrates an LLM-based Agent-as-a-Judge (EVALUATOR) to provide tool-based feedback on constraint satisfaction, which is then used by the SYNTHESIZER to explore and construct chemically plausible and constraint-compliant synthetic pathways.
- LARC achieves a 72.9% success rate on constrained retrosynthesis tasks, outperforming LLM baselines and approaching human expert-level performance in significantly less time, demonstrating its potential as a co-scientist for chemical discovery.

---

#### 15th August 2025

[Using Natural Language for Human-Robot Collaboration in the Real World](http://arxiv.org/abs/2508.11759v1)

- Collaborative System: introduces a human-robot collaboration framework with a Cognitive Agent (orchestrates system/learns/reasons), Situational Knowledge (stores world state/experiences), an LLM (translates language/provides knowledge), a Physical Robot (provides perception/action), and a Human Director (provides instructions/guidance), designed to enable robots to understand natural language for real-world tasks.
- The system leverages the LLM for language understanding and common-sense knowledge, while the cognitive agent handles reasoning, integration, and incremental learning from human interaction and experience.
- This architecture aims to overcome challenges in grounding referring expressions, performing complex tasks, and understanding free-form language, moving towards more robust and intuitive human-robot collaboration.

---

[Tapas are free! Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](http://arxiv.org/abs/2508.11425v1)

- TAPA (Training-free Adaptation of Programmatic Agents): introduces a novel framework that positions LLMs as intelligent moderators of the symbolic action space, enabling training-free adaptation of programmatic agents in dynamic environments, with LLM (moderates action space), RAG System (stores expert knowledge), Logical Primitives (high-level strategic intents), Symbolic Programs (concrete action implementations), Decision Agent (selects logical primitives), Simulation Environment (generates diverse scenarios), Provenance Chain (records execution traces), and Shadow Simulation (validates candidate programs).
- The framework synthesizes and adapts modular programs for individual high-level actions (logical primitives) by decoupling strategic intent from execution.
- This approach enables real-time adaptation without costly retraining, ensuring performance and reliability in safety-critical domains like cyber defense and swarm intelligence.

---

[AIM-Bench: Evaluating Decision-making Biases of Agentic LLM as Inventory Manager](http://arxiv.org/abs/2508.11416v1)

- AIM-Bench: introduces a novel benchmark designed to evaluate LLM agents' decision-making behavior in uncertain supply chain management scenarios, featuring diverse environments, context engineering modules (background, memory, structured output), and an evaluator utilizing both behavioral and real-world benefit metrics.
- This benchmark assesses LLM agents' inventory replenishment capabilities, identifies human-like decision biases (e.g., mean anchoring, bullwhip effect), and investigates mitigation strategies such as cognitive reflection and information sharing.
- The study reveals varying degrees of decision bias across different LLMs, underscoring the importance of considering potential biases and implementing strategic model selection for deploying LLMs in critical inventory management.

---

[Towards Embodied Conversational Agents for Reducing Oral Exam Anxiety in Extended Reality](http://arxiv.org/abs/2508.11412v1)

- The ECA-based Coach System: introduces a framework for reducing oral exam anxiety by integrating photorealistic Embodied Conversational Agents (ECAs) with real-time LLMs in Extended Reality (XR) environments to provide psychologically safe, adaptive, and repeatable oral examination rehearsals.
- This system leverages a Conversational Engine with LLMs, Speech-to-Text, and Text-to-Speech for fluid dialogue, augmented by Domain Knowledge Integration via RAG for factual correctness.
- It further incorporates a Learner Modeling and Feedback Module to adapt agent behavior and provide personalized feedback, all rendered within an Extended Reality Interface using Unreal Engine for immersive experiences.

---

[FACET: Teacher-Centred LLM-Based Multi-Agent Systems- Towards Personalized Educational Worksheets](http://arxiv.org/abs/2508.11401v1)

- FACET (Framework for Agent-based Classroom Enhancement for Teacher): introduces a teacher-facing, LLM-based multi-agent system for generating individualized classroom materials, with Learner Agents (simulate student behavior), a Teacher Agent (adapts instructional content), and an Evaluator Agent (provides quality feedback), designed to integrate cognitive and motivational dimensions of learner profiles for personalized educational worksheets.
- The system's modular design supports experimentation and refinement, enabling scalable, context-aware personalization in heterogeneous classroom settings.
- Evaluations confirm the framework's stability and alignment between generated materials and diverse learner profiles, addressing a critical gap in AI-driven educational personalization.

---

[Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis](http://arxiv.org/abs/2508.11398v1)

- DSM5AgentFlow: introduces an LLM-based multi-agent workflow for autonomously generating DSM-5 Level-1 diagnostic questionnaires by simulating therapist-client dialogues, with all its components: Therapist Agent (administers DSM-5 questionnaire), Client Agent (simulates client profile), Diagnostician Agent (generates diagnosis, rationale), Conversation Generation (simulates therapist-client dialogue), Conversation Transcript (records dialogue history), Retriever (fetches relevant DSM-5 passages), DSM-5 Passages (authoritative clinical criteria), Diagnosis & Rationale (predicted disorder, step-by-step explanation), and LLM (powers all agents).
- This framework delivers transparent, step-by-step disorder predictions and explainable, trustworthy results by grounding diagnoses in clinical criteria and conversational evidence.
- The approach enhances interpretability and trustworthiness of LLM-driven assessments, ensuring compliance with ethical and legal standards in mental health care.

---

[SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems](http://arxiv.org/abs/2508.11310v1)

- SGSimEval (Survey Generation with Similarity-Enhanced Evaluation): introduces a comprehensive benchmark for automatic survey generation systems, integrating data collection, topic mining, decomposition, embedding generation, and a multifaceted evaluation framework.
- This framework assesses outline, content, and reference quality using traditional metrics, LLM-based scoring, and two similarity-enhanced approaches: Human-as-Perfect and Balanced Similarity Weighting.
- The benchmark, built on 80 highly-cited survey papers, reveals ASG systems' strengths in outline generation but highlights areas for improvement in content and reference quality.

---

[AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions](http://arxiv.org/abs/2508.11152v1)

- AlphaAgents: introduces a modular multi-agent debate framework for equity portfolio construction, featuring specialized Fundamental, Sentiment, and Valuation Agents, coordinated by a Groupchat Agent, utilizing RAG, Summarization, Valuation, and Fundamental Report Pull Tools, and employing Multi-agent Collaboration and Debate Mechanisms, all built on the AutoGen Framework.
- This framework enhances equity analysis and stock selection by enabling LLM-based agents to cooperatively analyze diverse financial data, mitigate cognitive biases, and resolve conflicting analyses through a structured debate process.
- The system provides transparent reasoning trails through discussion logs and integrates explicit risk tolerance profiles, representing a foundational step towards scalable and transparent agentic investment systems.

---

[AI Agentic Programming: A Survey of Techniques, Challenges, and Opportunities](http://arxiv.org/abs/2508.11126v1)

- AI Agentic Programming: introduces a paradigm where LLMs autonomously plan, execute, and refine software development tasks by integrating with external tools and managing context iteratively.
- This approach enables LLM-based agents to decompose complex goals, coordinate multi-step processes, and adapt behavior based on intermediate feedback from the development environment.
- The survey highlights key challenges including context handling, persistent memory, safety, toolchain integration, and the need for robust evaluation benchmarks for these intelligent coding agents.

---

[The Roots of International Perceptions: Simulating US Attitude Changes Towards China with LLM Agents](http://arxiv.org/abs/2508.08837v2)

- Framework for Macro-Scale Attitudes Evolution: introduces a simulation framework to model US citizens' attitude changes towards China over 20 years, integrating real-world data for agent profile creation, news distribution, and a cognitive reflection mechanism for opinion updates.
- The framework initializes thousands of LLM agents with diverse profiles from social surveys and media, exposing them to over 100,000 news articles annually, and enabling them to update their views through a cognitive dissonance-based reflection process.
- It also incorporates intervention mechanisms, including a Debiasing Agent for objective news exposure and a Devil's Advocate Agent for alternative perspectives, to explore ways of alleviating negative opinion trends.

---

[TRAINING-FREE MULTIMODAL LARGE LANGUAGE MODEL ORCHESTRATION](http://arxiv.org/abs/2508.10016v2)

- MLLM Orchestration (Multimodal Large Language Model Orchestration): introduces a training-free framework for interactive multimodal AI systems, featuring a Controller LLM (orchestrates tasks, routes to specialized models), Cross-modal Memory (integrates multimodal context), and Parallel Text-to-Speech (TTS) (generates speech output).
- The framework leverages LLMs' reasoning capabilities to coordinate specialized models through explicit workflows, enabling natural multimodal interactions while maintaining modularity and interpretability.
- This approach achieves comprehensive multimodal capabilities without additional training, demonstrating performance improvements and reduced latency compared to traditional jointly-trained methods.

---

[Rethinking Autonomy: Preventing Failures in AI-Driven Software Engineering](http://arxiv.org/abs/2508.11824v1)

- SAFE-AI Framework (Safety, Auditability, Feedback, and Explainability): introduces a holistic approach to prevent failures in AI-driven software engineering, integrating guardrails, sandboxing, runtime verification, risk-aware logging, human-in-the-loop systems, and explainable AI techniques.
- The framework addresses challenges like insecure code generation, hallucinated outputs, and lack of transparency by emphasizing continuous learning loops and verifiable records of AI actions.
- It also proposes a taxonomy of AI behaviors to guide risk assessment and oversight, aligning with emerging regulations for responsible AI development.

---

[Intelligent Edge Resource Provisioning for Scalable Digital Twins of Autonomous Vehicles](http://arxiv.org/abs/2508.11574v1)

- Intelligent Edge Resource Provisioning Framework: introduces a distributed computing architecture integrating Digital Twins (DTs) and Mobile Edge Computing (MEC) within a software-defined vehicular networking framework, featuring a Two-Tier Architecture, Collaborative Task Computation Model, and a DRL Algorithm-trained Autonomous Agent for intelligent, low-latency transportation services.
- The framework significantly enhances DT operations by reducing synchronization errors to 5% and achieving 99.5% edge resource utilization, evaluated using a connected autonomous vehicle (CAV) traffic simulation.
- This approach addresses key challenges in synchronization latency and resource allocation for real-time, data-intensive DT operations in dynamic edge-cloud environments.

---

[Sim2Dust: Mastering Dynamic Waypoint Tracking on Granular Media](http://arxiv.org/abs/2508.11503v1)

- Sim2Dust introduces a complete sim-to-real framework for dynamic waypoint tracking on granular media, integrating Space Robotics Bench (SRB) (simulation environment), Procedural Content Generation (PCG) (generates diverse terrains), Domain Randomization (DR) (varies simulation parameters), High-fidelity Particle Physics (simulates granular media), Reinforcement Learning (RL) Algorithms (trains control policies), Action Smoothing Filters (stabilizes rover actions), LunaLab (lunar-analogue testbed), Leo Rover (physical robotic platform), and OptiTrack Motion Capture System (provides ground-truth localization).
- The framework leverages massively parallel simulation and procedural diversity to train robust RL agents, enabling zero-shot transfer to a physical wheeled rover in a lunar-analogue facility.
- Experiments demonstrate that training with procedural diversity is critical for successful zero-shot transfer, and action smoothing is necessary for stable hardware deployment.

---

[Relative Position Matters: Trajectory Prediction and Planning with Polar Representation](http://arxiv.org/abs/2508.11492v1)

- Polaris: introduces a novel framework for trajectory prediction and planning that operates entirely in Polar coordinates, distinguishing itself from conventional Cartesian-based approaches by leveraging Polar scene context encoding, a decoding module, and Polar relationship refinement, all equipped with Relative Embedding Transformers.
- This framework explicitly models distance and direction variations, capturing relative relationships through dedicated encoding and refinement modules, enabling more structured and spatially aware trajectory prediction and planning.
- Polaris achieves state-of-the-art performance on Argoverse 2 and nuPlan benchmarks by effectively modeling varying influences of traffic elements and utilizing a dual-loss strategy in both Polar and Cartesian coordinates.

---

[EvoPSF: ONLINE EVOLUTION OF AUTONOMOUS DRIVING MODELS VIA PLANNING-STATE FEEDBACK](http://arxiv.org/abs/2508.11453v1)

- EvoPSF (Online Evolution of Autonomous Driving Models via Planning-State Feedback): introduces a novel online evolution framework for autonomous driving, featuring a base model, uncertainty estimation, diagnostic signal trigger, agent-agent attention, top-k objects selection, confidence filtering, self-supervised loss calculation, and model update.
- This framework leverages planning uncertainty as a trigger for targeted online adaptation, focusing on critical objects identified via attention mechanisms.
- It improves model robustness and prediction accuracy by comparing predicted waypoints with high-confidence perceived positions, enabling self-supervised updates during deployment.

---

[ImagiDrive: A Unified Imagination-and-Planning Framework for Autonomous Driving](http://arxiv.org/abs/2508.11428v1)

- ImagiDrive: introduces a novel end-to-end autonomous driving framework that integrates a Driving Agent (VLM-based trajectory prediction), a Scene Imaginer (DWM-based future scene generation), an Imagination-and-Planning Loop (recurrent planning refinement), a Trajectory Buffer (stores generated trajectories), an Early Stop Strategy (ESS for adaptive iteration termination), and a Trajectory Select Strategy (TSS for robust trajectory selection), where the system unifies imagination and planning for enhanced safety and efficiency.
- The framework operates by having the driving agent propose initial trajectories, which guide the scene imaginer to generate corresponding future scenarios, and these imagined frames are then iteratively fed back to the agent to refine planning decisions.
- To ensure robust and efficient inference, the system maintains a trajectory buffer and incorporates early stopping and trajectory selection strategies based on safety and consistency.

---

[CRAFT-GUI: Curriculum-Reinforced Agent For GUI Tasks](http://arxiv.org/abs/2508.11360v1)

- CRAFT-GUI (Curriculum-Reinforced Agent For GUI Tasks): introduces a curriculum learning framework for GUI tasks, integrating a policy model, reference model, curriculum learning, and fine-grained hybrid reward mechanisms within a GRPO-based reinforcement learning setup.
- The framework addresses limitations of uniform training data and coarse rewards by stratifying tasks by difficulty and providing nuanced feedback through rule-based and model-judged evaluations.
- CRAFT-GUI demonstrates significant performance improvements on both public and internal GUI benchmarks, validating the effectiveness of curriculum-driven reinforcement learning for complex GUI interaction.

---

[ALLEN: RETHINKING MAS DESIGN THROUGH STEP-LEVEL POLICY AUTONOMY](http://arxiv.org/abs/2508.11294v1)

- Allen (Multi-Agent System): introduces a novel MAS framework that redefines the basic execution unit as a "Step," enabling agents to autonomously form behavioral patterns by combining these units, and employs a four-tier state architecture (Task, Stage, Agent, Step) to constrain system behavior, achieving a unification of topological optimization and controllable progress.
- The framework grants unprecedented Policy Autonomy by allowing agents to dynamically adapt their behavioral strategies at the step-level, while balancing collaborative efficiency, task supervision, and human oversight in complex network topologies.
- It implements a step-wise execution paradigm with a hierarchical state system for task tracking and multi-agent collaboration, supported by a robust communication mechanism and persistent memory for long-term context.

---

[Scene Graph-Guided Proactive Replanning for Failure-Resilient Embodied Agents](http://arxiv.org/abs/2508.11286v1)

- SGPR (Scene Graph-Guided Proactive Replanning): introduces a proactive replanning framework that detects and corrects failures at subtask boundaries by comparing scene graphs from current RGB-D observations against reference graphs from successful demonstrations, leveraging a Scene-Graph Generator, Target Precondition Buffer, Scene Graph Comparison Module, and LLM-based Reasoning and Replanning Modules.
- This framework proactively triggers replanning by reasoning over scene discrepancies, preventing failures before execution, unlike post-hoc methods.
- SGPR significantly improves task success and robustness by grounding decisions in structured visual understanding and successful demonstrations.

---

[Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory](https://github.com/bytedance-seed/m3-agent)

- M3-Agent (Multimodal Agent Framework): introduces a novel multimodal agent framework equipped with long-term memory, enabling it to process real-time visual and auditory inputs, build entity-centric multimodal long-term memories, and reason over them.
- The framework operates through two parallel processes: memorization, which continuously perceives inputs to construct and update memory, and control, which interprets instructions and reasons over stored memory to execute tasks.
- Its long-term memory is organized as a multimodal graph, accumulating both episodic memory (concrete events) and semantic memory (world knowledge) to support deeper and more consistent environmental understanding.

---

[RL-MoE: An Image-Based Privacy Preserving Approach In Intelligent Transportation System](http://arxiv.org/abs/2508.09186v2)

- RL-MoE (Reinforcement Learning - Mixture-of-Experts): introduces a novel framework transforming sensitive visual data into privacy-preserving textual descriptions, integrating an Input, a MoE (decomposes visual scene) with specialized Experts and RAG, a Weighting and Scoring Gate (prioritizes expert outputs), and an RL Agent (optimizes textual descriptions) with a Reward Function to generate Output Text.
- The framework avoids direct image transmission by converting visual data into structured textual descriptions, optimizing for both semantic accuracy and privacy preservation.
- This approach leverages a Mixture-of-Experts architecture for nuanced, multi-aspect scene decomposition and a Reinforcement Learning agent for policy-based text optimization.

---

[Labels or Input? Rethinking Augmentation in Multimodal Hate Detection](http://arxiv.org/abs/2508.11808v1)

- Dual-Pronged Framework for Multimodal Hate Detection: introduces a comprehensive approach to improve multimodal hate detection, integrating prompt optimization for scaled label generation and a multimodal augmentation pipeline for creating counterfactually neutral memes.
- The prompt optimization framework leverages structured prompts and teacher models to generate nuanced hatefulness labels, enhancing supervision granularity for VLMs.
- The multimodal augmentation pipeline employs a multi-agent LLM-VLM setup to rewrite hateful captions while preserving visual context, reducing spurious correlations and improving classifier generalization.

---

[SafeSieve: From Heuristics to Experience in Progressive Pruning for Multi-Agent LLM Communication](http://arxiv.org/abs/2508.11733v1)

- SafeSieve: introduces a progressive and adaptive multi-agent pruning algorithm that dynamically refines inter-agent communication by integrating initial LLM-based semantic evaluation with accumulated performance feedback and employing 0-extension clustering for graph sparsification.
- The framework transitions from heuristic initialization to experience-driven refinement, preserving coherent agent groups while eliminating ineffective communication links.
- Experiments demonstrate improved accuracy and reduced token usage, along with robustness against prompt injection and efficiency in heterogeneous LLM deployments.

---

[Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory](https://github.com/bytedance-seed/m3-agent)

- M3-Agent (Multimodal Agent with Long-Term Memory): introduces a novel multimodal agent framework that continuously perceives real-time visual and auditory inputs, builds entity-centric multimodal long-term memories, and reasons over them to accomplish tasks.
- The framework operates through two parallel processes: memorization, which constructs and updates long-term memory by generating episodic and semantic memories, and control, which interprets instructions and retrieves relevant information for iterative reasoning.
- Its long-term memory is organized as a multimodal graph, enabling deeper and more consistent understanding of the environment, and is leveraged by an MLLM for multi-turn reasoning and task execution.

---

[Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework](https://arxiv.org/abs/2508.16629)

- Adaptive Memory Framework: introduces an adaptive and data-driven memory framework for optimizing LLM-based agents, featuring Memory Storage (stores observations), Memory Retrieval (retrieves relevant memories), Memory Utilization (integrates memories into prompts), an Inference Model (LLM for decisions/actions), and an Environment (provides observations, feedback).
- The framework integrates an MoE Gate Function (adaptive retrieval combination) for memory retrieval, a Learnable Aggregation Process (improves memory utilization) for memory utilization, and Task-Specific Reflection (adapts memory storage) for memory storage.
- It utilizes both Off-policy Optimization (offline training, trajectory reuse) and On-policy Optimization (online learning, policy alignment) to enable LLM-based agents to learn effective memorization strategies in dynamic environments.

---

[Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory](https://github.com/bytedance-seed/m3-agent)

- M3-Agent: introduces a novel multimodal agent framework, with MLLM (central processing unit), Long-Term Memory (structured multimodal graph), Episodic Memory (event records), Semantic Memory (world knowledge), Memorization Workflow (memory building process), Video/Audio Input (perceptual streams), Tools (feature extractors), Face Detection (facial recognition), Speaker Diarization (voice identification), Control Workflow (task execution process), Instruction (task command), Search Tool (memory retrieval mechanism), Reasoning (iterative inference), Response (agent output), and Environment (external world), designed to process real-time multimodal inputs, build long-term memory, and reason over it for task accomplishment.
- The framework operates through two parallel processes: memorization, which continuously perceives real-time video and audio streams to construct and update entity-centric episodic and semantic memories, and control, which interprets instructions and iteratively reasons over the stored multimodal graph memory.
- M3-Agent leverages specialized tools for face detection and speaker diarization to maintain consistent entity representations, and employs search functions to retrieve relevant information from its long-term memory, enabling multi-turn reasoning and higher task success rates.

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



[Back to top](#topofthepage)
