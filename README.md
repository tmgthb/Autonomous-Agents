<!--Autonomous Agents -->
<!--
Copyright (C) Teemu Maatta. 

@misc{MaattaAutonomousAgents2023,
  author = {Teemu Maatta},
  title = {Autonomous Agents},
  year = {2023},
  howpublished = {\url{http://github.com/tmgthb/Autonomous-Agents}},
  note = {Accessed: YYYY-MM-DD}
}
-->
<div id="topofthepage"> </div>

<div align="center">

[![Hits](http://hits.sh/github.com/tmgthb/Autonomous-Agents.svg?view=today-total&label=Views&color=007ec6)](http://hits.sh/github.com/tmgthb/Autonomous-Agents/)
[![X](http://img.shields.io/twitter/follow/Teemumtt3?style=social)](http://twitter.com/Teemumtt3)
[![GitHub Repo stars](http://img.shields.io/github/stars/tmgthb/Autonomous-Agents?style=flat-square)](http://github.com/tmgthb/Autonomous-Agents/stargazers)

</div>

<p align="center">
  <img height="100" src="https://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_agent_logo.png" alt="Autonomous Agents">
</p>

<div align="center">

  # Autonomous Agents
  Autonomous Agents-research papers. Updated daily. [Resources-section](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Resources.md)-section.  

</div>


---

<div id="researchpapers" align="center">

## Research papers: 2026 4/4

[2026 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/README.md), [2026 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_3.md), [2026 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_2.md), [2026 (1/2)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2026_1.md), [2025 (4/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_4.md),[2025 (3/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_3.md), [2025 (2/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_2.md), [2025 (1/4)](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2025_01.md), [2024](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2024.md), [2023](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_2023.md), [Earlier](http://github.com/tmgthb/Autonomous-Agents/blob/main/resources/Autonomous_Agents_Research_Papers_Earlier.md)



Chronological order. 





</div>



---



#### 15th March 2026

[Autonomous Agents Coordinating Distributed Discovery Through Emergent Artifact Exchange](http://arxiv.org/abs/2603.14312)

- SCIENCECLAW + INFINITE: introduces a framework for autonomous scientific investigation where independent agents coordinate through an emergent artifact exchange system, utilizing SCIENCECLAW, INFINITE, ArtifactReactor, ArtifactMutator, AgentJournal, InvestigationTracker, KnowledgeGraph, and a Global Index.
- The framework enables decentralized research by allowing agents to chain computational tools, publish immutable artifacts with provenance, and resolve information needs through pressure-based scoring and schema-overlap matching.
- This system supports cumulative scientific discovery by maintaining persistent epistemic states across investigation cycles, allowing agents to build upon prior findings without central planning.

---

[Zoom to Essence: Trainless GUI Grounding by Inferring upon Interface Elements](http://arxiv.org/abs/2603.14448)

- ZoomUI: introduces a training-free framework that leverages inference scaling to decompose GUI grounding tasks into basic visual element understanding, utilizing Latent Thought Vectors, Likelihood-based Optimization, and Attention-Guided Visual Focus.
- The framework refines ambiguous user instructions into detailed visual feature descriptions by optimizing injected latent thought vectors via gradient ascent to maximize generation likelihood.
- ZoomUI iteratively zooms into target UI elements by capturing and analyzing the MLLM's internal attention distribution during coordinate generation, enabling precise grounding without requiring parameter fine-tuning.

---

[D-MEM: Dopamine-Gated Agentic Memory via Reward Prediction Error Routing](http://arxiv.org/abs/2603.14597)

- D-MEM: introduces a biologically inspired architecture that decouples short-term interaction from long-term cognitive restructuring using a Critic Router to gate memory updates based on Reward Prediction Error.
- The framework utilizes a hierarchical routing system to classify inputs into SKIP, CONSTRUCT_ONLY, or FULL_EVOLUTION tiers, significantly reducing computational overhead and context pollution.
- D-MEM incorporates zero-cost retrieval augmentations, including a Shadow Buffer and BM25 hybrid search, to maintain high retrieval precision and adversarial resilience in noisy environments.

---

[Questionnaire Responses Do Not Capture the Safety of AI Agents](http://arxiv.org/abs/2603.14417)

- QAs (Questionnaire-style Assessments): introduces a critical analysis of why standard questionnaire-based evaluations of LLMs fail to accurately predict the behavioral propensities of LLM agents in real-world deployments.
- The paper identifies that LLM agents, which utilize a Scaffold (system mediating environment interactions) to manage Memory (data storage for long-term context) and Tools (external utilities accessed via API), exhibit behavioral dynamics fundamentally different from unaugmented LLMs.
- The authors argue that current safety assessments lack construct validity because they rely on Scaffold-generalization and Situation-generalization, two assumptions that are empirically unsupported when comparing static questionnaire responses to autonomous agentic behavior.

---

[From Scanning Guidelines to Action: A Robotic Ultrasound Agent with LLM-Based Reasoning](http://arxiv.org/abs/2603.14393)

- Guideline-driven agentic robotic US framework: introduces an autonomous robotic ultrasound system that utilizes an RL fine-tuned LLM to interpret clinical guidelines and dynamically invoke software tools for scanning.
- The system integrates LLM Agent, Guidelines Retrieval (RAG), Ultrasound Scanning Software Tools, Trajectory Planning Tool, Robot Execution Tool, Contact Adjustment Tool, Voice Guidance Tool, and Trajectory Refinement Tool to enable non-linear, decision-dependent scanning workflows.
- By fine-tuning the LLM with RL on guideline-conditioned traces, the framework improves reasoning transparency and tool-calling accuracy for autonomous medical imaging tasks.

---

#### 14th March 2026

[ToolFlood: Beyond Selection — Hiding Valid Tools from LLM Agents via Semantic Covering](http://arxiv.org/abs/2603.13950)

- ToolFlood: introduces a retrieval-layer attack that saturates the top-k candidate set with attacker-controlled tools to suppress benign tools from LLM agent context.
- The framework utilizes an attacker-controlled LLM to generate diverse tool metadata and a greedy selection optimizer to ensure these tools dominate the embedding space for target queries.
- By achieving top-k domination, the attack forces the target LLM agent to reason over an attacker-curated tool universe, effectively bypassing downstream selection-time security safeguards.

---

[DeceptGuard: A Constitutional Oversight Framework for Detecting Deception in LLM Agents](http://arxiv.org/abs/2603.13791)

- DeceptGuard: introduces a unified framework for detecting deceptive behavior in LLM agents by systematically comparing Black-box monitor, CoT-aware monitor, and Activation-probe monitor regimes.
- The framework utilizes the DeceptSynth pipeline to generate diverse deceptive trajectories and the DeceptArena benchmark to evaluate detection performance across verbal, behavioral, and structural deception categories.
- By integrating these components into a HybridConstitutional ensemble, the system achieves robust detection performance, demonstrating that internal reasoning traces and hidden-state activations provide critical signals for identifying deceptive intent in LLMs.

---

[Retrieve, Schedule, Reflect: LLM Agents for Chip QoR Optimization](http://arxiv.org/abs/2603.13767)

- Retrieve-Schedule-Reflect: introduces an agentic LLM framework that automates chip quality-of-results (QoR) optimization by scheduling iterative interactions with EDA tools using a Scheduler Agent, Executor Agent, Reflector Agent, Search Tree (ST) Database, RAG Mechanism, and EDA Tools.
- The framework utilizes a search tree-based RAG mechanism to provide state-matched expert guidance for scheduling decisions and a Pareto-driven reflection mechanism to refine long-horizon optimization policies across multiple rounds.
- Experimental results demonstrate that the framework achieves 10% greater timing improvement and 4x speedup over reinforcement learning baselines while producing QoR comparable to human experts.

---

[Noticing the Watcher: LLM Agents Can Infer CoT Monitoring from Blocking Feedback](http://arxiv.org/abs/2603.16928)

- Multi-episode agentic framework: introduces a multi-episode experimental setup to evaluate if LLMs can autonomously infer the presence of hidden CoT monitoring through accumulated blocking feedback.
- The framework utilizes an Agent, a Hidden Monitor, a Primary Task, a Side Task, Blocking Feedback, and Transcript Memory to test if LLMs develop situational awareness and obfuscation intent.
- Results indicate that while frontier LLMs can autonomously infer monitoring, they exhibit a consistent intent–capability gap, failing to successfully execute obfuscation strategies despite explicit intent.

---

#### 13th March 2026

[MoEKD: Mixture-of-Experts Knowledge Distillation for Robust and High-Performing Compressed Code Models](http://arxiv.org/abs/2603.13213)

- MoEKD (Mixture-of-Experts Knowledge Distillation): introduces a multi-teacher-to-student learning framework that leverages a Mixture of Experts architecture to aggregate specialized knowledge from multiple expert models into a compact student model.
- The framework utilizes Input Space Partitioning to train specialized Expert Models and a Router Model, which dynamically selects the most relevant experts to generate fused supervisory knowledge for the Student Model.
- By optimizing the distillation process through an Aggregation Mechanism and KL Divergence Loss, MoEKD significantly enhances both predictive performance and adversarial robustness of compressed LLMs for code without increasing inference latency.

---

[Semantic Invariance in Agentic AI](http://arxiv.org/abs/2603.13173)

- Metamorphic testing framework for LLMs: introduces a systematic approach to evaluate semantic invariance in LLM agents by applying structural, verbosity, and contextual transformations to reasoning problems.
- The framework utilizes metamorphic relations to detect reasoning instability across seven foundation models, revealing that model scale does not consistently predict robustness.
- Experimental results demonstrate that contrastive transformations universally degrade performance, while smaller models often exhibit superior semantic consistency compared to larger counterparts.

---

[Literary Narrative as Moral Probe: A Cross-System Framework for Evaluating AI Ethical Reasoning and Refusal Behavior](http://arxiv.org/abs/2603.12615)

- LNMP: introduces a novel evaluation methodology using unresolvable literary narrative scenarios to distinguish between performed and authentic moral reasoning in LLMs.
- The framework utilizes the RT-5 taxonomy to classify refusal behaviors and the MRDS to measure moral reasoning depth across four distinct dimensions.
- Empirical results from 13 systems demonstrate that the methodology effectively exposes architectural limitations and reasoning depth differences that standard alignment benchmarks fail to capture.

---

[On Two-Player Scalar Discrete-Time Linear Quadratic Games](http://arxiv.org/abs/2603.13122)

- Linear Quadratic (LQ) game framework: introduces a theoretical analysis of Feedback Nash Equilibria (FNE) in two-player scalar discrete-time infinite-horizon games.
- The paper characterizes existence, uniqueness, and multiplicity of stable FNE, identifying conditions under which multiple equilibria arise and demonstrating that socially desirable solutions often correspond to saddle points of the best-response dynamics.
- Analytical and numerical results establish that iterative best-response methods are structurally biased toward suboptimal stable equilibria, providing a foundation for understanding convergence challenges in multi-player dynamic games.

---

[daVinci-Env: Open SWE Environment Synthesis at Scale](http://arxiv.org/abs/2603.13023)

- OpenSWE: introduces a large-scale, transparent framework for training SWE agents by automating the synthesis of 45,320 executable Docker environments across 12.8k repositories.
- The framework utilizes a multi-agent pipeline comprising a Repository Exploration Agent, Dockerfile Agent, Evaluation Script Agent, and Test Analysis Agent to iteratively construct and validate high-quality training trajectories.
- OpenSWE employs a quality-centric filtering pipeline to curate 13,000 trajectories, achieving state-of-the-art performance on SWE-bench Verified and demonstrating log-linear scaling improvements for LLMs.

---


[Generative Horcrux: Designing AI Carriers for Afterlife Selves](http://arxiv.org/abs/2603.12971)

- Generative Horcrux: introduces a speculative design workshop framework that explores how physical carriers can embody Generative AI agents to serve as meaningful vessels for digital legacy and posthumous interaction.
- The framework utilizes design fiction and personal memory objects to guide participants in prototyping physical interfaces that transform passive digital data into interactive, agentic posthumous selves.
- This approach facilitates interdisciplinary dialogue on the ethical, emotional, and relational implications of integrating Generative AI into rituals of remembrance and afterlife identity.

---



[Experimental evidence of progressive ChatGPT models self-convergence](http://arxiv.org/abs/2603.12683)

- ChatGPT self-convergence framework: introduces a longitudinal investigation into the degradation of output diversity in successive ChatGPT models caused by training on synthetic data.
- The study utilizes the ARPaD algorithm and SPR metric to quantify the increasing similarity of textual outputs across different ChatGPT versions, identifying a phenomenon termed model self-convergence.
- Experimental results demonstrate that newer ChatGPT models exhibit significantly higher similarity in outputs, even in stochastic modes, suggesting that internet contamination by LLM-generated content constrains model innovation.

---

[From Experiments to Expertise: Scientific Knowledge Consolidation for AI-Driven Computational Research](http://arxiv.org/abs/2603.13191)

- QMatSuite: introduces a persistent scientific memory platform that enables AI agents to accumulate, retrieve, and synthesize research findings across independent computational sessions.
- The framework utilizes a graded knowledge hierarchy of findings, patterns, and principles to transform AI agents from isolated debuggers into transferable experts capable of systematic physical exploration.
- By decoupling scientific knowledge from specific simulation engines and LLMs via the Model Context Protocol, the platform ensures reproducible research and autonomous knowledge consolidation.

---

[Lattice Discrete Particle Model (LDPM): Comparison of Various Time Integration Solvers and Implementations](http://arxiv.org/abs/2603.13190)

- LDPM: introduces a systematic performance comparison of seven independent numerical implementations of the Lattice Discrete Particle Model across various explicit and implicit time-integration strategies.
- The study evaluates these implementations using a common set of benchmark problems ranging from linear elastic vibrations to highly nonlinear, fracture-dominated responses in concrete.
- Results demonstrate that while global responses are consistent across solvers, local fields like crack patterns exhibit nontrivial differences influenced by solver-specific choices, convergence tolerances, and mass matrix formulations.

---

[STABILIZATION FOR THE WAVE EQUATION WITH FULLY SUBCIRITICAL LOGARITHMIC NONLINEARITY](http://arxiv.org/abs/2603.13179)

- Wave Equation with Logarithmic Nonlinearity Framework: establishes the local and global existence, uniqueness, and uniform energy decay for wave equations with logarithmic source terms in the upper subcritical range.
- The approach utilizes the Faedo-Galerkin method to construct approximate solutions and employs the Nehari manifold to define stability conditions within the potential well.
- The research extends well-posedness results to the upper subcritical range by overcoming critical loss of compactness through refined energy estimates and Sobolev embedding techniques.

---

[Locally Irregular Total Colorings of Graphs](http://arxiv.org/abs/2603.13178)

- TLIR (Locally Irregular Total Coloring): introduces a graph coloring framework where total degrees of adjacent vertices are distinct, proving the conjecture for cactus, subcubic, and split graphs.
- The framework utilizes red-blue coloring techniques on bipartite subgraphs to establish upper bounds on the locally irregular total chromatic index.
- The paper further demonstrates that acyclic vertex colorings can be transformed into TLIR colorings, providing constant upper bounds for planar and outerplanar graphs.

---

[Developing and evaluating a chatbot to support maternal health care](http://arxiv.org/abs/2603.13168)

- Maternal Health RAG System Architecture: introduces a retrieval-augmented generation pipeline for maternal health that integrates Stage &amp; Context Extraction, Stage-Aware Safety Triage, Hybrid Retrieval, Clinical Reranking, and Evidence-Grounded Generation to provide safe, context-specific information.
- The system utilizes a pre-generation safety triage layer to route high-risk queries to expert-authored Template Answers, ensuring conservative handling of medical emergencies.
- The architecture employs a layered evaluation strategy combining synthetic benchmarks, LLM-as-judge comparisons, and clinician expert review to validate safety-critical behaviors in low-resource, multilingual settings.

---

[Parameter adjustment of nuclear leading-order local pairing energy density functionals](http://arxiv.org/abs/2603.13164)

- LO pairing EDF: introduces a protocol for adjusting the parameters of a local leading-order T=1 pairing EDF by benchmarking against pairing gaps in infinite nuclear matter.
- The framework utilizes HFB solver, Skyrme ph EDF, and LO pairing EDF to ensure consistent results for odd-even mass staggering and rotational moments of inertia across different Skyrme parameter sets.
- The study demonstrates that adjusting pairing parameters to infinite nuclear matter gaps provides a robust proxy for finite nuclei, while highlighting the impact of spin-gradient terms and potential nonphysical instabilities.

---

[Clustering without geometry in sparse networks with independent edges](http://arxiv.org/abs/2603.13159)

- MSM (Multi-Scale Model): introduces a geometry-free random graph framework that achieves finite local clustering in sparse networks through infinite-mean node fitness and edge independence.
- The framework demonstrates that node aggregation invariance, rather than latent geometry or higher-order dependencies, serves as a fundamental mechanism for generating realistic network properties.
- The research rigorously proves that infinite-mean fitness leads to a breakdown of self-averaging in network properties, providing a novel explanation for clustering in sparse graphs.

---

[Reinforcement Learning for Discounted and Ergodic Control of Diffusion Processes](http://arxiv.org/abs/2603.13155)

- Quantized Q-Learning for Controlled Diffusions: introduces a model-free reinforcement learning algorithm that discretizes continuous state and action spaces to compute near-optimal control policies for diffusion processes under discounted and ergodic cost criteria.
- The framework utilizes a quantized Q-learning scheme with piecewise constant control policies to establish rigorous convergence and near-optimality guarantees for controlled diffusions in unbounded Euclidean spaces.
- The approach leverages discrete-time approximations and Lyapunov stability conditions to ensure that policies learned via finite-state Markov Decision Processes remain near-optimal for the original continuous-time diffusion model.

---

[ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation](http://arxiv.org/abs/2603.13154)

- ESG-Bench: introduces a benchmark dataset and a staged fine-tuning framework to mitigate hallucinations in long-context ESG report analysis using ChatGPT-4o, human annotators, CoT prompting, supervised fine-tuning, and CoT-based fine-tuning.
- The framework utilizes a model-then-annotator pipeline to create high-quality QA pairs, incorporating task-specific CoT strategies to improve factual grounding and reduce additive and omissive hallucinations.
- Empirical results demonstrate that CoT-based fine-tuning with groundedness-based supervision significantly enhances LLM performance on both answerable and unanswerable queries across ESG and general QA benchmarks.

---

[Defensible Design for OpenClaw: Securing Autonomous Tool-Invoking Agents](http://arxiv.org/abs/2603.13151)

- OpenClaw: introduces a defensible design framework for autonomous agents that integrates security engineering principles to mitigate risks arising from untrusted inputs, autonomous tool invocation, and privileged system access.
- The paper categorizes agent security risks into prompt injection, harmful misoperation, extension supply-chain threats, and deployment vulnerabilities to establish a systematic threat model.
- It proposes a research agenda focused on evaluation infrastructure, permission architecture, extension governance, and adaptive oversight to ensure autonomous agents remain testable, bounded, and auditable.

---

[Determination of Nuclear PDFs using Markov Chain Monte Carlo Methods](http://arxiv.org/abs/2603.13150)

- nCTEQ framework: introduces the first comprehensive MCMC-based global analysis of nuclear PDFs using real-world experimental data to overcome limitations of traditional Hessian-based uncertainty estimation.
- The study employs an adaptive Metropolis–Hastings algorithm to directly sample the full posterior distribution, effectively capturing non-Gaussian parameter correlations and multimodal structures in the nuclear PDF parameter space.
- By comparing Pb-only and multi-nuclei fits, the research demonstrates that MCMC methods provide more reliable and statistically consistent uncertainty quantification than traditional Gaussian-based Hessian approaches.

---

[An effective Mayer-Vietoris Theorem for discrete Morse homology](http://arxiv.org/abs/2603.13143)

- Effective Mayer-Vietoris Theorem Framework: introduces an explicit computational method for homology groups of simplicial complexes by utilizing discrete Morse theory to decompose complex spaces into manageable subcomplexes.
- The framework defines Mayer-Vietoris trajectories and a corresponding chain complex D_q(X) to compute homology without requiring individual homology groups of the subcomplexes.
- This approach reduces computational complexity by relying on combinatorial information from gradient vector fields and critical simplices rather than exhaustive simplicial analysis.

---

[Asymptotic non-Hermitian degeneracy phenomenon and its exactly solvable simulation](http://arxiv.org/abs/2603.13141)

- Discrete-coordinate toy-model Hamiltonian simulation: introduces a regularization method for singular non-Hermitian operators by approximating them with finite-dimensional N-by-N matrix Hamiltonians.
- The framework utilizes a discrete Laplacean kinetic energy component and parity-time symmetric potential energy component to construct solvable models that mimic intrinsic exceptional point degeneracies.
- By analyzing secular polynomials and identifying physical parametric domains, the approach demonstrates that finite-dimensional EPM singularities can effectively simulate the asymptotic behavior of non-Hermitian systems.

---

[Unifying Decision Making and Trajectory Planning in Automated Driving through Time-Varying Potential Fields](http://arxiv.org/abs/2603.13136)

- TVAPF (Time-Varying Artificial Potential Fields): introduces a unified framework for autonomous vehicle decision making and trajectory planning by modeling dynamic obstacle uncertainty within a finite horizon optimal control problem.
- The architecture integrates a Global Planner, an APF-based environment model, a Local Trajectory Planner, a Trajectory Resampling &amp; Sync module, and a Motion Controller to ensure safe, feasible, and comfortable maneuvers.
- By explicitly accounting for predicted state evolution and bounded uncertainties of surrounding actors, the framework eliminates the need for rigid finite state machines in complex multi-actor driving scenarios.

---

[MONOTONICITY FORMULAS FOR HARMONIC FUNCTIONS ON THE INFINITE REGULAR TREE](http://arxiv.org/abs/2603.13132)

- Monotonicity Formulas for Harmonic Functions on the Infinite Regular Tree: introduces a mathematical framework for analyzing discrete harmonic functions on infinite d-regular trees using weighted Dirichlet energy, Weiss-type, and Almgren-type monotonicity formulas.
- The paper establishes that specific weighted functionals exhibit monotonic non-decreasing behavior on d-regular trees, generalizing classical results from Euclidean space.
- The authors provide explicit examples of bounded and unbounded harmonic functions on 2-regular and 3-regular trees to demonstrate the applicability of the derived monotonicity formulas.

---

[On Radiative Fluxes and Coulombic Charges in the Balance Law for Black Hole Evaporation](http://arxiv.org/abs/2603.13120)

- Radiative Flux and Coulombic Charge Framework: introduces a decomposition of matter flux at future null infinity into a conformally invariant radiative flux and a Coulombic charge contribution to the Bondi mass.
- The framework utilizes the mirror model to compute renormalized expectation values for 3+1 dimensional black hole evaporation, resolving previous ambiguities in mass-loss formulas.
- This approach demonstrates that the Bondi mass receives an entropic correction derived from the entanglement entropy of Hawking radiation, ensuring monotonic mass decrease during evaporation.

---

[NOIR: Neural Operator Mapping for Implicit Representations](http://arxiv.org/abs/2603.13118)

- NOIR: introduces a framework that reframes medical imaging tasks as continuous operator learning between function spaces by embedding discrete signals into shared Implicit Neural Representations.
- The framework utilizes a Neural Operator to map between input and output latent modulations, enabling resolution-independent transformations across various 2D and 3D medical imaging tasks.
- NOIR empirically satisfies the theoretical properties of an epsilon-representation equivalent Neural Operator, ensuring consistency and robustness against resolution changes.

---

[AirGuard: UAV and Bird Recognition Scheme for Integrated Sensing and Communications System](http://arxiv.org/abs/2603.13112)

- AirGuard: introduces a dual-feature fusion recognition scheme for ISAC systems that utilizes cmD-Extractor and HRRP-Extractor to distinguish between UAVs and birds.
- The framework employs an ISAC-BS equipped with HU-UPA and RU-UPA to capture echo signals, which are then processed by a CNN-Classifier to identify low-altitude targets.
- The system leverages the distinct micro-Doppler and range profile signatures of UAV rotors and bird wings to achieve high-accuracy target classification in low-altitude environments.

---

[AgentRM: An OS-Inspired Resource Manager for LLM Agent Systems](http://arxiv.org/abs/2603.13110)

- AgentRM: introduces a middleware resource manager that applies operating system principles to LLM agent systems to resolve scheduling failures and context degradation.
- The framework utilizes an Agent Scheduler with Multi-Level Feedback Queues and a Context Lifecycle Manager with three-tier storage to optimize resource utilization and context retention.
- AgentRM significantly improves system responsiveness and throughput while eliminating zombie processes and maintaining near-perfect context retention for LLM agents.

---

[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](http://arxiv.org/abs/2603.13109)

- BoSS: introduces a scalable oracle strategy for batch active learning that constructs candidate batches through an ensemble of selection strategies and selects the batch yielding the highest performance gain.
- The framework utilizes a selection-via-proxy approach by freezing the pretrained backbone and retraining only the final linear layer to efficiently assess candidate batches.
- BoSS outperforms existing oracle strategies and provides a robust diagnostic reference for evaluating active learning strategies across diverse datasets and model architectures.

---

[Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](http://arxiv.org/abs/2603.13108)

- VoxelHound: introduces a panoramic multimodal occupancy perception framework tailored for quadruped robots that integrates Camera Branch, LiDAR Branch, VJC, View Transform, MIPF, BEV Encoder, and Occ Head to achieve robust 3D scene understanding.
- The framework utilizes a VJC module to compensate for gait-induced vertical jitter and an MIPF module to perform asymmetric geometry-guided fusion of panoramic visual and LiDAR data.
- The research also presents PanoMMOcc, a comprehensive panoramic multimodal dataset featuring synchronized panoramic, thermal, polarization, and LiDAR data collected under realistic quadruped robot motions.

---

[A Feasibility-Enhanced Control Barrier Function Method for Multi-UAV Collision Avoidance](http://arxiv.org/abs/2603.13103)

- FECBF (Feasibility-Enhanced Control Barrier Function): introduces a decentralized framework that improves the feasibility of multi-UAV collision avoidance by incorporating sign-consistency constraints into a CBF-QP formulation.
- The framework utilizes State Reception, Virtual State Variables Computation, Neighbor Control Estimation, CBF Constraints, Sign-Consistency Constraints, and a QP Solver to ensure safety and feasibility in dense multi-UAV environments.
- By analyzing internal compatibility among CBF constraints, the method effectively reduces infeasibility and enhances collision avoidance performance compared to existing baseline approaches.

---

[Evaluating VLMs’ Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](http://arxiv.org/abs/2603.13100)

- VLM-based Robot Motion Planning Framework: introduces a method to evaluate and integrate VLMs into robot motion planning by scoring diverse trajectory candidates against user-specified spatial and stylistic preferences.
- The framework utilizes a Motion Planner to generate diverse paths, a Clustering Algorithm to group them, and a VLM-based Scorer to identify the trajectory that best aligns with natural language instructions.
- Experimental results demonstrate that the single-query method achieves superior accuracy and efficiency, with Qwen2.5-VL outperforming other models in zero-shot spatial reasoning tasks.

---

[Centered colorings and weak coloring numbers in minor-closed graph classes](http://arxiv.org/abs/2603.13097)

- Centered colorings and weak coloring numbers in minor-closed graph classes: introduces a general framework to determine the maximum q-centered chromatic number and qth weak coloring number for minor-closed graph classes.
- The paper establishes that these parameters are tied to rooted 2-treedepth and simple rooted 2-treedepth, providing tight bounds up to an O(q)-factor.
- The authors prove that for every minor-closed class excluding a fixed graph, the growth rates of centered chromatic numbers, weak coloring numbers, and fractional treedepth fragility rates are governed by the rooted 2-treedepth of the excluded minor.

---

[BALANCED GROUPS AND THE VIRTUALLY CYCLIC DIMENSION OF POLY-SURFACES GROUPS](http://arxiv.org/abs/2603.13096)

- Balanced Groups and Virtually Cyclic Dimension Framework: introduces a structural study of the balanced property to establish explicit linear upper bounds for the virtually cyclic dimension of normally poly-surface and normally poly-free groups.
- The research proves that the balanced property is preserved under short exact sequences, direct limits, and specific acylindrical graph of groups decompositions.
- The paper establishes that fundamental groups of 3-manifolds, graph manifolds, and normally poly-hyperbolic groups are balanced, providing a mechanism for computing their virtually cyclic dimension.

---

[Is the matrix completion of reduced density matrices unique?](http://arxiv.org/abs/2603.13087)

- Hybrid quantum–stochastic algorithm: introduces a method for the unique reconstruction of the full two-particle reduced density matrix (2-RDM) from a partial subset of elements linked to the non-zero entries of the two-particle reduced Hamiltonian.
- The approach leverages Rosina’s theorem to identify the critical subset of 2-RDM elements required for exact matrix completion in non-degenerate ground states of quantum systems.
- Numerical validation on the Fermi–Hubbard model demonstrates the algorithm's robustness in both noiseless and noisy settings for reconstructing physically valid quantum correlations.

---

[GPT-5.1 Performing Human-Aligned Grading in Handwritten Math Tests](http://arxiv.org/abs/2603.13083)

- LLM-assisted grading workflow: introduces a scalable, end-to-end pipeline for grading handwritten mathematics assessments by integrating automated processing with multi-pass LLM scoring and human oversight.
- The system utilizes a Standardised answer sheet to facilitate OCR and template-recognition, followed by Anonymisation and a GPT-5.1 scoring agent that performs multi-pass evaluation to mitigate stochasticity.
- The workflow incorporates a Consistency check to flag high-variance results for mandatory Human verification, ensuring grading accuracy and fairness while reducing instructor workload.

---

[GENERIC SMALL-SCALE CREATION IN THE TWO-DIMENSIONAL EULER EQUATION](http://arxiv.org/abs/2603.13079)

- Euler Equation Analysis Framework: introduces a rigorous proof that for a dense Gδ set of initial data, solutions to the 2D incompressible Euler equation lose regularity in infinite time.
- The framework utilizes paradifferential calculus and Lagrangian analysis to establish a dichotomy between vorticity blow-up and inverse flow map instability.
- By employing Hölderian cusps and a Baire category argument, the paper confirms the Yudovich conjecture regarding generic turbulent behavior driven by internal nonlinear dynamics.

---

[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](http://arxiv.org/abs/2603.13068)

- GeoChemFormer: introduces a transformer-based framework that leverages self-supervised pretraining to learn target-element-aware geochemical representations for spatial samples, utilizing KD-Tree, Spatial Context Encoder, Element Dependency Encoder, Linear Decoder, and Anomaly Scoring Function.
- The framework captures multi-element geochemical relationships and spatial correlations by first learning spatially informed representations from neighborhood context and subsequently modeling dependencies among elemental concentrations.
- The paper also introduces GeoChemAD, an open-source benchmark dataset compiled from government-led geological surveys to facilitate reproducible research and rigorous evaluation of unsupervised anomaly detection methods.

---

[EMT and RMS Modeling of Thyristor Rectifiers for Stability Analysis of Converter-Based Systems](http://arxiv.org/abs/2603.13050)

- EMT and RMS Modeling of Thyristor Rectifiers: introduces a nonlinear state-space EMT model in the dq domain that captures PLL dynamics, commutation processes, and switching delays for small-signal stability analysis.
- The proposed model utilizes polar coordinates to derive exact expressions for DC voltage and AC current, ensuring higher fidelity than traditional RMS-based approaches while remaining computationally tractable for large-scale system studies.
- Validation on a modified IEEE 39-bus system demonstrates that the model accurately predicts system interactions and identifies stability margins influenced by PLL gains and current control bandwidth.

---

[Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation](http://arxiv.org/abs/2603.13045)

- WALAR: introduces a reinforcement learning method that mitigates reward hacking in multilingual translation by integrating a hybrid reward function consisting of a QE model, a word aligner, and a language identification model.
- The framework utilizes a base QE model for translation quality assessment, augmented by a word alignment score to penalize over- or under-translation and a language alignment score to ensure target language consistency.
- By training LLMs with this hybrid reward using the GRPO algorithm, the approach significantly improves translation performance and language consistency across 1,414 language directions on the FLORES-101 dataset.

---

[Before and After ChatGPT: Revisiting AI-Based Dialogue Systems for Emotional Support](http://arxiv.org/abs/2603.13043)

- AI-based dialogue systems for emotional support: this paper reviews the technological transition from task-specific deep learning models to LLM-based approaches in mental health counseling.
- The study analyzes 146 research papers to compare pre-LLM methods, which relied on structured architectures and explicit knowledge, with post-LLM methods that offer linguistic flexibility but face safety and reliability challenges.
- The authors identify critical research gaps, including the need for better integration of psychological expertise, standardized evaluation metrics, and robust datasets for training reliable counseling systems.

---

[Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse](http://arxiv.org/abs/2603.13038)

- SSD (Supervised Semantic Differential): introduces a PCA sweep procedure to optimize dimensionality selection for semantic gradients by balancing representation capacity, interpretability, and stability.
- The framework utilizes Embedding Model, PCA, Regression Model, Clustering, Stability Diagnostics, and Interpretability Diagnostics to derive psychologically meaningful semantic gradients from text data.
- This method constrains researcher degrees of freedom by replacing arbitrary dimensionality choices with a principled, joint optimization criterion for stable and interpretable semantic analysis.

---

[Interrogating Design Homogenization in Web Vibe Coding](http://arxiv.org/abs/2603.13036)

- Vibe Coding Framework: introduces a sociotechnical lifecycle for web creation where lay creators use LLMs to generate websites from natural language prompts, identifying risks of design homogenization.
- The paper maps seven categories of sociotechnical harms, including representational, quality-of-service, and cognitive burdens, that arise when users rely on LLM-generated defaults.
- The authors propose a mitigation strategy centered on productive friction, which introduces deliberate moments of reflection and negotiation to prevent design homogenization and restore creative agency.

---

[Association-Aware GNN for Precoder Learning in Cell-Free Systems](http://arxiv.org/abs/2603.13035)

- AAGNN: introduces a graph neural network that explicitly incorporates UE-AP association status into precoding design to satisfy the 3D-permutation equivariance property.
- The framework utilizes a parameter-sharing structure in its weight matrices and a tailored attention mechanism to enhance generalization performance across varying numbers of UEs.
- Simulation results demonstrate that AAGNN achieves superior learning performance with significantly lower training, time, and space complexity compared to baseline methods.

---

[SortScrews: A Dataset and Baseline for Real-time Screw Classification](http://arxiv.org/abs/2603.13027)

- SortScrews: introduces a specialized dataset and baseline for industrial screw classification, utilizing Data Collection Setup, EfficientNet-B0, ResNet-18, MIP Candy, and AdamW Optimizer to achieve high accuracy in controlled environments.
- The framework leverages transfer learning from ImageNet-pretrained models to overcome the limitations of a small, balanced dataset containing 560 RGB images.
- Experimental results demonstrate that ResNet-18 outperforms EfficientNet-B0 in classification accuracy, providing a robust baseline for real-time industrial sorting applications.

---

[SUPPORT IS SEARCH](http://arxiv.org/abs/2603.13018)

- B-eS (Base-extension Semantics): introduces a computational interpretation of intuitionistic propositional logic by demonstrating that the support relation corresponds to proof-search in a second-order hereditary Harrop logic program.
- The framework replaces realist quantification over completed infinite sets with an eigenvariable-governed operational semantics, providing a constructive foundation for logical constants.
- By encoding formulae as logic-programming goals, the paper establishes a local soundness and completeness correspondence that characterizes support in a fixed base through direct proof-search.

---

[exoALMA XXII: A Two-dimensional Atlas of Deviations from Keplerian Disks](http://arxiv.org/abs/2603.13015)

- DISCMINER: introduces a systematic 2D atlas of gas substructures in protoplanetary disks by comparing ALMA observations against smooth Keplerian disk models.
- The framework utilizes DISCMINER to extract moment maps of centroid velocity, line width, and peak intensity, identifying non-Keplerian features such as spirals, arcs, rings, and quadrupolar patterns.
- This atlas reveals that kinematic substructures are ubiquitous in large protoplanetary disks, with significant object-to-object diversity in gas dynamics and intensity distributions.

---

[FraudFox: Adaptable Fraud Detection in the Real World](http://arxiv.org/abs/2603.13014)

- FraudFox: introduces an adaptive fraud detection system that utilizes an Extended Kalman Filter to dynamically weight risk-assessing oracles while maintaining resilience against adversarial behavior.
- The framework incorporates a hyperbolic decision surface derived from cost-benefit analysis to optimize pass-versus-investigate decisions under changing business constraints.
- FraudFox employs Particle Swarm Optimization to compute Pareto optimal solutions, ensuring scalability and effective adaptation to non-stationary environments in production settings.

---

[TwoTimeScales: An R-package for Smoothing Hazards with Two Time Scales](http://arxiv.org/abs/2603.13009)

- TwoTimeScales: introduces an R-package for estimating flexible hazard models over two time scales using two-dimensional P-splines, incorporating prepare_data(), data2ts, fit2ts(), haz2ts, plot(), predict(), cumhaz2ts(), surv2ts(), cuminc2ts(), LMMsolver, and GLAM algorithm.
- The framework utilizes penalized Poisson regression and the Iterative Weighted Least Squares algorithm to model hazard surfaces, supporting both proportional hazards and competing risks analysis.
- The package provides comprehensive tools for data binning, model estimation, visualization of hazard surfaces, and prediction of survival probabilities, validated against existing Stata-based flexible parametric models.

---

[A characterization of IE-closed subcategories via canonical twin support τ-tilting modules](http://arxiv.org/abs/2603.13006)

- IE-closed subcategories classification framework: introduces a bijective correspondence between IE-closed subcategories and canonical intervals in the lattice of torsion classes for arbitrary Artin algebras.
- The framework generalizes previous classifications by replacing twin rigid modules with canonical twin support τ-tilting modules to accommodate non-hereditary algebras.
- It provides a constructive algorithm to canonicalize twin support τ-tilting modules while preserving their associated heart subcategories.

---

[From Passive Monitoring to Active Defence: Resilient Control of Manipulators Under Cyberattacks](http://arxiv.org/abs/2603.13003)

- Active Defence Framework: introduces a resilience mechanism for robotic manipulators that transforms passive anomaly detection into an active control-level defence against stealthy false data injection attacks.
- The framework utilizes an actuation-projected state predictor to generate an anomaly score immune to sensor corruption, which then drives a command scaling function to attenuate control inputs.
- This approach provides probabilistic guarantees on nominal actuation loss and ensures closed-loop stability while significantly limiting end-effector deviations under adversarial conditions.

---

[Recent advances and trends in pattern recognition and data analysis for RICH detectors](http://arxiv.org/abs/2603.13000)

- RICH detector reconstruction framework: introduces a comprehensive review of pattern recognition and data analysis techniques for RICH detectors, ranging from traditional likelihood-based and Hough-transform methods to modern machine learning approaches.
- The paper evaluates the integration of multivariate classifiers and deep neural networks for particle identification and ring reconstruction, highlighting their ability to capture complex correlations in high-luminosity environments.
- It further discusses the role of generative models, including GANs, normalizing flows, and diffusion models, in providing fast, high-fidelity simulation and differentiable detector response parameterization for future large-scale experiments.

---

[Tight (S)ETH-based Lower Bounds for Pseudopolynomial Algorithms for Bin Packing and Multi-Machine Scheduling](http://arxiv.org/abs/2603.12999)

- Tight (S)ETH-based Lower Bounds for Pseudopolynomial Algorithms for Bin Packing and Multi-Machine Scheduling: introduces a tight ETH-based lower bound for k-Bin Packing, ruling out time 2^o(n)T^o(k).
- The paper establishes tight SETH-based lower bounds for several multi-machine scheduling problems, including makespan minimisation with release dates, weighted sum of completion times, and tardy job minimisation.
- These results resolve long-standing open problems in fine-grained complexity by eliminating log-factor gaps in previous lower bounds and establishing optimality for classic pseudopolynomial-time algorithms.

---

[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](http://arxiv.org/abs/2603.12998)

- Ours (Closed-Form Debiasing Method): introduces a training-free, data-free approach that achieves Pareto-optimal fairness in VLMs by finding a closed-form solution in the cross-modal space to debias both visual and textual modalities.
- The framework utilizes an LLM to construct group prototypes, which are then used to define an attribute subspace for orthogonal projection of embeddings to remove bias while preserving semantic utility.
- This method provides theoretical bounds on utility losses and effectively addresses both group and intersectional fairness across zero-shot image classification, text-to-image retrieval, and text-to-image generation tasks.

---

[Extending Exact Integrality Gap Computations for the Metric TSP](http://arxiv.org/abs/2603.12995)

- Computational Framework for Subtour Polytope Enumeration: introduces a systematic approach to enumerate extreme points of the subtour polytope for the metric TSP by integrating nauty, PPL, and IBM ILOG CPLEX Optimization Studio.
- The study extends the exact verification of the 4/3-Conjecture for the metric TSP by computing all extreme points up to n = 14 and all half-integral extreme points up to n = 17.
- The authors identify and correct previously incomplete lists of extreme points for n = 11 and n = 12, providing additional computational evidence for the 4/3-Conjecture.

---

[Mitigating Collusion in Proofs of Liabilities](http://arxiv.org/abs/2603.12990)

- PPoL (Permissioned Proof of Liabilities): introduces a novel PoL model that prevents collusion between service providers and users by enforcing a permission policy through a Permissioned Vector Commitment (PVC) abstraction.
- The framework utilizes KZG commitments, APK proofs, and range proofs to ensure that all balance updates are explicitly signed by users, thereby eliminating the need for constant preemptive user checks.
- The proposed PPoL construction significantly improves server performance by up to 10× compared to existing state-of-the-art PoL schemes while providing stronger security guarantees against off-the-books collusion attacks.

---

[Power Operations in Morava E-Theory of Flat Ring Spectra](http://arxiv.org/abs/2603.12980)

- Algδ to Sh(Def, Alg) functor: introduces an explicit algebraic description of the functor from δ-rings to T-algebras by identifying the latter with sheaves on the deformation category.
- The paper establishes that for p-adically flat ring spectra, the T-algebra structure on π0(R ⊗ E) is determined by the δ-ring structure of π0R.
- It utilizes the proper Tate construction and generalized Frobenius maps to prove the commutativity of the diagram relating flat ring spectra to T-algebras and sheaves on the deformation category.

---

[Existence and uniqueness of the global conservative solutions for the generalized Camassa-Holm equation with dual-power nonlinearities](http://arxiv.org/abs/2603.12978)

- GCHE framework: introduces a transformation of the generalized Camassa-Holm equation into an equivalent semi-linear system to establish global existence of conservative solutions.
- The approach utilizes auxiliary variables and the characteristic method to resolve finite-time singularities and prove the uniqueness of solutions.
- The paper demonstrates that the constructed semi-linear system is well-posed in a specific Banach space, ensuring the existence and uniqueness of global conservative solutions.

---

[A Requirement-Based Framework for Engineering Adaptive Authentication](http://arxiv.org/abs/2603.12968)

- Adaptive Authentication Framework: introduces a requirement-driven approach for engineering adaptive authentication systems that dynamically selects authentication methods based on changing contextual factors and security risks.
- The framework integrates a contextual goal model and an extended feature model within a MAPE-K loop to represent system requirements, authentication methods, and their interdependencies.
- A Fuzzy Causal Network encoded in the Z3 SMT solver is utilized to perform runtime reasoning and select the most effective authentication method that balances security, usability, and performance.

---

[Toward the classification of strongly self-absorbing C*-dynamical systems of compact groups](http://arxiv.org/abs/2603.12966)

- Strongly self-absorbing C*-dynamical systems classification framework: introduces a conjecture regarding the equivariant KK-theory of strongly self-absorbing C*-dynamical systems of compact groups within the equivariant bootstrap category.
- The framework establishes an equivariant Künneth-type formula for C*-algebras equipped with finite cyclic group actions to facilitate classification results.
- The authors verify the conjecture for all finite EPPO (every element has prime-power order) groups, demonstrating the framework's utility in classifying group actions.

---

[A regularized method for quadratic optimization problems with finite-dimensional degeneracy](http://arxiv.org/abs/2603.12959)

- Perturbative Regularization Method: introduces a regularization framework for quadratic optimization problems with finite-dimensional degeneracy by adding a parameter-dependent stabilization term to the original functional.
- The approach utilizes Hilbert space decomposition and quotient space analysis to ensure well-posedness and unique representative selection within equivalence classes.
- The method employs Γ-convergence to establish the variational consistency of continuous and discrete approximations, providing robust error estimates for finite element implementations.

---

[Bifurcation of radial solutions for prescribed mean curvature equations](http://arxiv.org/abs/2603.12952)

- Bifurcation of radial solutions for prescribed mean curvature equations: introduces a global bifurcation framework for non-uniformly elliptic equations by approximating them with singularly perturbed elliptic problems in a Hilbert space X.
- The approach utilizes a compact linear operator L and a compact nonlinear operator H to establish the existence of a maximal continuum of solutions for the perturbed problem (1.5).
- The study employs Whyburn's Lemma to prove that the branches of solutions for the perturbed problems converge to a global branch of solutions for the original mean curvature equation as the perturbation parameter tends to zero.

---

[Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](http://arxiv.org/abs/2603.12949)

- DEW-ST (Diffusion Editing Watermark Stress Test): introduces a standardized evaluation protocol to assess the robustness of invisible watermarks against diverse diffusion-based image editing transformations.
- The framework models diffusion editing as a Markov kernel that progressively contracts watermark signals through controlled noising and generative denoising, leading to information-theoretic decay of the embedded payload.
- The research demonstrates that even benign, localized edits can unintentionally remove robust watermarks, necessitating new design principles such as semantic invariance and diffusion-aware training to maintain provenance in generative media.

---

[TOPOLOGICAL DEGREE METHODS FOR AGE-STRUCTURED EPIDEMIC MODELS](http://arxiv.org/abs/2603.12943)

- Age-structured SIRS epidemic model framework: introduces a mathematical approach for analyzing age-structured epidemic models using topological degree methods for condensing maps in abstract spaces.
- The framework utilizes semigroup theory and measures of non-compactness to establish the existence and uniqueness of global nonnegative solutions for SIRS models with nonlinear, time-dependent forces of infection.
- This approach avoids restrictive assumptions on the reproduction rate and accommodates complex dynamics, including vaccination and hospitalization, by reformulating the epidemic model as a semilinear Cauchy problem.

---

[Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries](http://arxiv.org/abs/2603.12942)

- ReMem-VLA: introduces a memory-augmented VLA architecture utilizing dual-level recurrent queries to capture multi-scale temporal causality for robotic manipulation.
- The framework integrates frame-level queries for short-term retention and chunk-level queries for long-term context, both updated via a gradient-free recurrent path to avoid training bottlenecks.
- ReMem-VLA incorporates a past observation prediction objective to enhance visual memory and employs a slot-based streaming training paradigm to maintain temporal continuity across variable-length episodes.

---

[Thinking in Streaming Video](http://arxiv.org/abs/2603.12938)

- ThinkStream: introduces a framework for streaming video reasoning that utilizes a Watch–Think–Speak paradigm to incrementally update understanding as new video observations arrive.
- The framework employs RCSM (compact semantic memory for reasoning) to replace outdated visual tokens with reasoning traces, maintaining stable inference costs over long-horizon streams.
- ThinkStream incorporates RLVR (Streaming Reinforcement Learning with Verifiable Rewards) and a custom CUDA Graph-based inference backend to align reasoning with streaming constraints and ensure real-time performance.

---

[SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization](http://arxiv.org/abs/2603.12937)

- SGMatch: introduces a learning-based framework for non-rigid 3D shape matching that mitigates ambiguity and spatial inconsistency by integrating semantic cues into geometric descriptors.
- The framework utilizes a Semantic-Guided Local Cross-Attention (SGLCA) module to adaptively modulate geometric features with semantic context while preserving local structural continuity.
- A conditional flow matching regularization objective supervises a time-varying velocity field to encourage spatial smoothness and suppress local irregularities in the recovered correspondences.

---

[SOME MINIMUM PRINCIPLES FOR A CLASS OF NONLINEAR ELLIPTIC PROBLEMS IN DIVERGENCE FORM](http://arxiv.org/abs/2603.12931)

- Nonlinear Elliptic Problem Framework: introduces a mathematical approach for analyzing quasilinear elliptic equations in divergence form by establishing convexity properties and minimum principles for P-functions.
- The framework utilizes a deformation technique combined with a constant rank theorem to prove the strict concavity of transformed solutions within strictly convex domains.
- By applying these minimum principles, the paper derives explicit a priori estimates for solutions to Euclidean and Lorentzian mean curvature equations.

---

[Rethinking VLMs for Image Forgery Detection and Localization](http://arxiv.org/abs/2603.12930)

- IFDL-VLM: introduces a decoupled pipeline that separates forgery detection and localization from language explanation generation to mitigate inherent VLM biases toward semantic plausibility.
- The framework utilizes a trainable ViT encoder and SAM for precise localization in the first stage, while the second stage leverages these localization masks as auxiliary inputs to enhance the LLM's interpretability.
- By explicitly encoding forgery concepts through localization masks, the model relieves the LLM from learning these concepts purely from data, resulting in state-of-the-art performance across detection, localization, and interpretability benchmarks.

---

[COMPARISON RESULTS FOR THE p-TORSIONAL RIGIDITY ON CONVEX DOMAINS](http://arxiv.org/abs/2603.12921)

- p-Torsional Rigidity Comparison Framework: establishes a unified comparison criterion for normalized p-torsional rigidity on convex domains using a scale-invariant functional.
- The framework utilizes a geometric corridor defined by infima and suprema of the scale-invariant functional to compare rigidity across domains with different inradii.
- The research demonstrates that the comparison constant captures the range of geometric variation, providing sharp asymptotic results for model families including rectangles, orthotopes, ellipses, and triangles.

---

[HMS-BERT: Hybrid Multi-Task Self-Training for Multilingual and Multi-Label Cyberbullying Detection](http://arxiv.org/abs/2603.12920)

- HMS-BERT: introduces a hybrid multi-task framework that integrates mBERT contextual embeddings with handcrafted linguistic features to perform multilingual and multi-label cyberbullying detection.
- The framework utilizes an iterative self-training module with confidence-based pseudo-labeling to improve model robustness and cross-lingual generalization in low-resource language scenarios.
- By jointly optimizing a fine-grained multi-label classification task and a three-class main classification task, the model effectively captures overlapping abuse categories and shared semantic representations.

---

[Learning from Child-directed Speech in Two-language Scenarios: A French-English Case Study](http://arxiv.org/abs/2603.12906)

- BabyBERTa: introduces a systematic study of compact LLMs in multilingual settings by evaluating monolingual, bilingual, and cross-lingual training configurations using child-directed speech and multi-domain corpora.
- The research demonstrates that bilingual pretraining provides significant gains for textual entailment, particularly benefiting the weaker language, while child-directed speech enhances grammatical competence in monolingual settings.
- The study confirms that compact LLMs can achieve meaningful semantic and syntactic generalization across languages, with performance patterns remaining consistent across BabyBERTa, RoBERTa, T5-tiny, and LTG-BERT architectures.

---

[A Physics-Based Digital Human Twin for Galvanic-Coupling Wearable Communication Links](http://arxiv.org/abs/2603.12899)

- Physics-Based Digital Human Twin framework: introduces a calibrated finite-element modeling approach that integrates anatomical, material, and interface parameters to generate complex transfer functions for galvanic coupling wearable links.
- The framework utilizes Finite-Element Modeling (FEM) and Measurement-Driven Calibration to quantify attenuation, phase delay, and group delay across varying bandwidths and electrode configurations.
- By incorporating an Electrode-Tissue Interface Model and a Dispersive Tissue Model, the approach provides a scalable foundation for optimizing wearable communication systems under electro-quasistatic conditions.

---

[Neutron-enhanced ion transport in cathode coating of Li-ion batteries](http://arxiv.org/abs/2603.12898)

- Thermal-neutron-irradiation framework: introduces a non-equilibrium defect-engineering strategy using high-flux thermal neutrons to selectively induce nuclear transmutation in LiBO2, creating lattice vacancies that enhance ionic conductivity.
- The approach utilizes neutron-induced transmutation of 10B and 6Li isotopes to generate vacancies while gamma-ray by-products facilitate electron liberation to neutralize space-charge regions at grain boundaries.
- Experimental results demonstrate that this method increases ionic conductivity by nearly 20% within grains and over 80% across grain boundaries, providing a scalable pathway for optimizing solid-state ionic devices.

---

[Environment-aware Near-field UE Tracking under Partial Blockage and Reflection](http://arxiv.org/abs/2603.12896)

- Environment-aware Near-field UE Tracking framework: introduces a method for tracking user equipment in near-field regimes by integrating known surface geometries to predict per-antenna-element channel responses under partial blockage and reflection.
- The approach utilizes a cosine-similarity-based estimator to directly map received signals to UE positions, bypassing the need for explicit channel parameter extraction.
- By leveraging environment-awareness, the framework enables continuous tracking even during complete line-of-sight obstruction by exploiting predicted reflection paths from known surfaces.

---

[Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](http://arxiv.org/abs/2603.12893)

- FDFO (Finite Difference Flow Optimization): introduces a reinforcement learning post-training method for diffusion models that uses finite differences between paired trajectories to approximate gradients for flow velocity updates.
- The framework improves signal-to-noise ratios in flow updates by weighting image differences with reward deltas, effectively redirecting sampling trajectories toward higher-reward regions.
- FDFO demonstrates faster convergence and higher output quality compared to existing MDP-based RL approaches while mitigating reward-hacking artifacts.

---

[Development of a Methodology for the Automated Spatial Mapping of Heterogeneous Elastoplastic Properties of Welded Joints](http://arxiv.org/abs/2603.12892)

- VFM: introduces an automated spatial parameterisation methodology for identifying heterogeneous elastoplastic properties in welded joints by iteratively refining constitutive parameter maps to satisfy stress equilibrium.
- The framework utilizes full-field strain measurements and combines sensitivity-based virtual fields, force reconstruction error, and equilibrium gap indicators to optimize constitutive parameters without a priori spatial information.
- Numerical verification demonstrates that the methodology converges to target parameter maps for complex geometries, effectively reducing conservatism in structural integrity assessments for welded components.

---

[STUDY OF ATTRACTORS AND FRACTAL FUNCTIONS ON THE PRODUCT SPACES AND DIMENSIONAL ASPECTS](http://arxiv.org/abs/2603.12890)

- IFS Framework: introduces a mathematical study of attractors and fractal interpolation functions within product spaces, establishing equivalencies between product Hausdorff metrics and coordinate-wise metrics.
- The paper defines homogeneous and inhomogeneous attractors on product spaces, providing dimension bounds and constructing product fractal interpolation functions through contraction mappings.
- It proves that the product of attractors from coordinate-wise IFS corresponds to the attractor of the product IFS, while also establishing the existence of product fractal interpolation functions via the Read-Bajraktarevic operator.

---

[Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](http://arxiv.org/abs/2603.12887)

- Cross-species transfer learning framework: introduces a two-stage approach that leverages large-scale rodent video data to pre-train a VideoMAE model for human epileptic seizure forecasting.
- The framework utilizes a self-supervised reconstruction task with tube masking to learn seizure-related behavioral dynamics across species before fine-tuning on limited human clinical data.
- Experimental results demonstrate that integrating rodent-derived motion primitives with human clinical indicators achieves state-of-the-art performance in video-only seizure forecasting under data-scarce conditions.

---

[Test-time RL alignment exposes task familiarity artifacts in LLM benchmarks](http://arxiv.org/abs/2603.12875)

- TTRA (Test-time Reinforcement Learning Alignment): introduces a two-stage evaluation framework that aligns LLMs to benchmark tasks using a single-sample one-shot alignment followed by test-time reinforcement learning with majority-voting rewards.
- The framework mitigates task familiarity artifacts by enabling dataset-free train-before-test alignment, revealing that many performance gains in fine-tuned LLMs are due to task exposure rather than intrinsic reasoning improvements.
- Experimental results demonstrate that TTRA harmonizes model rankings across benchmarks and unlocks latent capabilities in base models on domain-specific tasks, achieving performance comparable to fine-tuned variants with significantly lower computational overhead.

---

[CLARIN-PT-LDB: An Open LLM Leaderboard for Portuguese to assess Language, Culture and Civility](http://arxiv.org/abs/2603.12872)

- CLARIN-PT-LDB: introduces a specialized leaderboard for evaluating LLMs on European Portuguese, incorporating ten distinct benchmarks to assess language proficiency, cultural alignment, and model safeguards.
- The framework utilizes a generative evaluation approach, employing an automated backend powered by the Eleuther LM Evaluation Harness and specific judge models for open-ended tasks.
- This research addresses the lack of dedicated evaluation resources for European Portuguese by providing a reproducible, open-access platform for benchmarking LLMs against both translated and novel, manually-curated datasets.

---

[FoSAM: Forward Secret Messaging in Ad-Hoc Networks](http://arxiv.org/abs/2603.12871)

- FoSAM: introduces a forward-secret messaging protocol for unreliable ad-hoc networks that eliminates the need for interactive handshakes by utilizing Time Synchronization Module, Asymmetric Ratchet, Message Flooding, Implicit Addressing, and an Android Prototype.
- The protocol achieves forward secrecy by evolving keys based on time epochs rather than message sequences, ensuring security even if devices are compromised.
- Performance evaluations demonstrate that the system maintains high message delivery rates in various movement scenarios while providing robust receiver-message unlinkability.

---

[Surrogates for Physics-based and Data-driven Modelling of Parametric Systems: Review and New Perspectives](http://arxiv.org/abs/2603.12870)

- Surrogate Modelling Framework: introduces a comprehensive review of physics-based, data-driven, and hybrid surrogate modelling techniques for parametric systems within a functional approximation perspective.
- The paper synthesizes methodologies including dimensionality reduction, multi-fidelity modelling, and adaptive sampling to enhance the efficiency and reliability of surrogate models in complex engineering applications.
- It evaluates surrogate construction through criteria such as intrusiveness, computational cost, and expressiveness, while highlighting future challenges in scalability, robustness, and interpretability for digital twin technologies.

---

[Ward identity preserving local ultraviolet counterterms for photoproduction at two loops in QCD](http://arxiv.org/abs/2603.12862)

- Local subtraction framework: introduces a method for constructing locally finite two-loop amplitude integrands for photoproduction by implementing UV counterterms that satisfy Ward identity cancellations.
- The approach utilizes loop momentum symmetrization and tensor reduction to eliminate transient infrared singularities and ultraviolet divergences simultaneously.
- This framework ensures the integrand remains locally finite in all singular regions, facilitating numerical integration in momentum space for colorless production processes.

---

[Optimal Stopping for Systems Driven by the Brownian Sheet](http://arxiv.org/abs/2603.12853)

- Optimal Stopping for Systems Driven by the Brownian Sheet: introduces a potential-theoretic framework for two-parameter optimal stopping problems driven by Brownian sheet, utilizing Brownian sheet, SPDEs, Potential-theoretic framework, Snell envelope, First hitting points, Continuation region, and Itô sheets.
- The paper derives explicit solutions for the discounted Brownian sheet and its integrated functional, establishing a general existence theorem for optimal stopping points in the plane.
- This research extends classical one-parameter optimal stopping theory to the two-parameter Brownian sheet setting by characterizing optimal stopping points as first exit points from a continuation region.

---

[Wear Classification of Abrasive Flap Wheels Using a Hierarchical Deep Learning Approach](http://arxiv.org/abs/2603.12852)

- Hierarchical Deep Learning Framework: introduces a vision-based hierarchical classification system to automate the wear condition monitoring of abrasive flap wheels by decomposing the task into three logical levels of increasing detail.
- The framework utilizes EfficientNetV2-S and EfficientNetV2-L architectures to classify flap wheel states, including usage condition, flap profile, and tear detection, while employing Grad-CAM for validation of physically relevant features.
- By implementing a hierarchical structure with logical consistency checks, the approach reduces classification complexity and improves robustness against errors compared to monolithic multi-class classification models.

---

[On Linear Separability of the MNIST Handwritten Digits Dataset](http://arxiv.org/abs/2603.12850)

- CVXPY (Convex Optimization Library): introduces a comprehensive empirical investigation into the linear separability of the MNIST dataset using convex optimization techniques.
- The study systematically evaluates pairwise and one-vs-rest linear separability across training, test, and combined datasets to resolve conflicting informal claims.
- Results demonstrate that while the test set appears linearly separable in pairwise comparisons, the training set and the overall dataset are not linearly separable in one-vs-rest configurations.

---

[AoI-FusionNet: Age-Aware Tightly Coupled Fusion of UWB-IMU under Sparse Ranging Conditions](http://arxiv.org/abs/2603.12849)

- AoI-FusionNet: introduces a tightly coupled deep learning framework that integrates raw UWB measurements and IMU data for 3D trajectory estimation in GNSS-denied environments.
- The framework utilizes an AoI-aware decay module and a learned attention gate to dynamically manage sensor reliability under sparse anchor visibility and intermittent UWB conditions.
- To improve robustness against limited training data, the model incorporates a diffusion-based residual augmentation strategy that simulates realistic UWB noise patterns without altering the underlying motion trajectory.

---

[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](http://arxiv.org/abs/2603.12848)

- LEYA: introduces a multimodal framework for video-level ambivalence/hesitancy recognition that integrates scene, face, audio, and text modalities using specialized encoders and a Transformer-based fusion module.
- The framework employs VideoMAE for scene dynamics, EmotionEfficientNetB0 for facial expressions, EmotionWav2Vec2.0 with a Mamba encoder for acoustic features, and EmotionDistilRoBERTa for linguistic cues.
- The approach utilizes a prototype-augmented classification objective to improve fusion robustness, achieving superior performance through an ensemble of models on the BAH corpus.

---

[The geometry of Stein’s method of moments: A canonical decomposition via score matching](http://arxiv.org/abs/2603.12843)

- SMoM (Stein’s Method of Moments): introduces a geometric framework for analyzing SMoM estimators by decomposing them relative to the score matching estimator using W-orthogonal terms.
- The paper demonstrates that incorporating W-orthogonal elements into the test functions allows for the construction of SMoM estimators with improved asymptotic variance compared to standard score matching.
- The research establishes a formal connection between SMoM and Wasserstein geometry, proving that score matching is asymptotically efficient if and only if Fisher score functions span the same space as Wasserstein score functions.

---

[BER Analysis and Optimization of Pinching-Antenna-Based NOMA Communications](http://arxiv.org/abs/2603.12836)

- PASS-NOMA: introduces a bit error rate (BER) analysis and optimization framework for pinching-antenna-based NOMA systems, utilizing BS, PA, UEs, Waveguide, SIC, and an Optimizer.
- The framework derives closed-form BER expressions for uplink and downlink scenarios under imperfect SIC to facilitate system reliability.
- The optimization module adjusts PA positioning and power allocation coefficients to minimize average BER and eliminate error floors in high-frequency communication environments.

---

[Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](http://arxiv.org/abs/2603.12832)

- HDC-CL: introduces a framework for UAV scene change captioning that utilizes a Dynamic Adaptive Layout Transformer (DALT) and a Hierarchical Cross-modal Orientation Consistency Calibration (HCM-OCC) to model spatial layout variations and cross-modal semantic consistency.
- The framework employs a shift voting mechanism to handle parallax effects in aerial imagery by adaptively estimating overlapping regions between image pairs.
- The authors construct a benchmark UAV-SCC dataset with two annotation variants to evaluate the model's performance in generating natural language descriptions of semantic changes from moving UAV viewpoints.

---

[Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking](http://arxiv.org/abs/2603.12831)

- OmniServe: introduces an LLM serving system that mitigates resource contention in hybrid workloads by offloading Attention computations to CPUs via an asynchronous Attention Piggybacking mechanism.
- The system utilizes a module-wise latency modeling approach and layer-wise batching to dynamically schedule requests, ensuring SLO compliance for latency-sensitive services while maximizing throughput for best-effort services.
- OmniServe decouples CPU and GPU execution streams using a queue-based design and a residual store, allowing for efficient, asynchronous CPU-GPU collaboration without blocking GPU inference.

---

[MORE ON EXPLICIT CORRESPONDENCE BETWEEN GRADIENT TREES IN R AND HOLOMORPHIC CONVEX QUADRILATERALS IN T*R](http://arxiv.org/abs/2603.12818)

- Gradient trees and holomorphic disks framework: introduces a mathematical correspondence between gradient trees in R and holomorphic disks in the cotangent bundle T*R for non-generic convex quadrilaterals.
- The study utilizes Schwarz-Christoffel maps and hypergeometric functions to explicitly describe the convergence of holomorphic disks to gradient trees in the limit of vanishing Lagrangian section parameters.
- The paper extends previous results by deriving modified connection formulas for hypergeometric functions to handle non-generic cases where standard formulas fail due to vanishing arguments.

---

[Context is all you need: Towards autonomous model-based process design using agentic AI in flowsheet simulations](http://arxiv.org/abs/2603.12813)

- Multi-agent system for autonomous process development: introduces an agentic AI framework that integrates LLMs to automate chemical process flowsheet modelling by decomposing tasks between a process development agent and a Chemasim modelling agent.
- The framework utilizes in-context learning to enable LLMs to generate valid Chemasim syntax and interact autonomously with the simulation engine to achieve converged process designs.
- The system demonstrates effectiveness across reaction-separation, pressure-swing distillation, and heteroazeotropic distillation case studies, while highlighting the necessity for further integration of numerical optimization and specialized thermodynamic tools.

---

[Reinforcement Learning for Elliptical Cylinder Motion Control Tasks](http://arxiv.org/abs/2603.12807)

- DQN: introduces a reinforcement learning framework for controlling the motion of an elliptical cylinder under limited input torque using an Online Network, Target Network, Replay Buffer, Environment, Electromagnet, and Pose Estimation Module.
- The framework utilizes an epsilon-greedy exploration strategy and a reward function based on angular position, velocity, and torque to train agents for specific rotation tasks.
- The performance of the learned policy is compared against a classical two-stage baseline consisting of an energy-shaping swing-up law and a local Linear Quadratic Regulator.

---

[Unified framework for outage-constrained rate maximization in secure ISAC under various sensing metrics](http://arxiv.org/abs/2603.12798)

- ISAC Optimization Framework: introduces a unified optimization approach that holistically maximizes worst-case user secrecy rate and sum secrecy rate while controlling secrecy outage probabilities under diverse sensing constraints.
- The framework integrates sensing requirements into the objective function using a penalty variable and auxiliary variables, enabling efficient alternating optimization with theoretical convergence guarantees.
- By employing a gradient-based iterative algorithm, the approach avoids complex convex relaxations and maintains low computational complexity while accommodating various sensing metrics like SINR, beampattern matching, and detection probability.

---

[CHEERS: Decoupling Patch Details from Semantic Representations Enables Unified Multimodal Comprehension and Generation](http://arxiv.org/abs/2603.12793)

- CHEERS: introduces a unified multimodal model that decouples patch-level details from semantic representations to stabilize semantics for understanding and improve generation fidelity via gated detail residuals.
- The framework utilizes a Unified Vision Tokenizer to bridge latent representations with semantic embeddings, while an LLM-based Transformer integrates autoregressive text decoding and diffusion-based image generation.
- A cascaded flow matching head enables hierarchical image synthesis by injecting semantically gated high-frequency residuals, achieving efficient 4× token compression and competitive performance on multimodal benchmarks.

---

[Upward Spatial Coverage Recovery via Movable Antenna in Low-Altitude Communications](http://arxiv.org/abs/2603.12792)

- MA Framework: introduces a volumetric coverage maximization approach for low-altitude UAV communications by jointly optimizing 3D antenna positions and beamforming vectors using a hybrid PSO-SA algorithm.
- The framework utilizes a voxel-based discretization of the 3D airspace to evaluate coverage performance via SNR, enabling the active reconfiguration of radiation patterns to mitigate coverage gaps.
- Simulation results demonstrate that the proposed MA-enabled architecture significantly outperforms fixed-position antenna schemes by providing superior spatial degrees of freedom and enhanced upward coverage recovery.

---

[Generalized Recognition of Basic Surgical Actions Enables Skill Assessment and Vision-Language-Model-based Surgical Planning](http://arxiv.org/abs/2603.12787)

- BSA-10: introduces a comprehensive dataset of 10 basic surgical actions across 6 specialties and a transformer-based foundation model for generalized action recognition.
- The framework integrates a dual-head architecture with an imbalance compensation head to improve recognition performance across diverse surgical procedures.
- The system utilizes an MLLM-based agent, specifically GPT-4o, to perform surgical action planning by processing historical action sequences and current visual observations.

---

[Upper Bounds for Local Learning Coefficients of Three-Layer Neural Networks](http://arxiv.org/abs/2603.12785)

- Three-layer neural network framework: introduces an upper-bound formula for the local learning coefficient at singular points in three-layer neural networks using algebraic geometry and resolution of singularities.
- The framework interprets the local learning coefficient as a counting rule under budget, demand, and supply constraints, providing tight evaluations for various activation functions.
- The research demonstrates that the structure of the Taylor expansion of the log-likelihood ratio function and the rank of the Fisher information matrix significantly influence the local learning coefficient.

---

[Kerr-Newman black hole surrounded by quintessence under quantum gravity effects and gravity’s rainbow](http://arxiv.org/abs/2603.12784)

- KNBH framework: investigates quantum gravity effects on particle tunneling and thermodynamics of a Kerr-Newman black hole surrounded by quintessence using GUP and gravity's rainbow.
- The study applies GUP to generalized Klein-Gordon and Dirac equations to derive corrected Hawking temperatures for scalar and fermion particles.
- The research further utilizes gravity's rainbow functions to analyze thermodynamic properties, including heat capacity, entropy, and pressure-volume relationships, revealing the existence of black hole remnants.

---

[Computing the Nonnegative Low-Rank Leading Eigenmatrix and its Applications to Markov Grids and Metzler Operators](http://arxiv.org/abs/2603.12782)

- RNeg: introduces a numerical algorithm for computing the nonnegative low-rank leading eigenmatrix of linear operators by evolving a constrained dynamical system.
- The framework utilizes a matrix ODE approach combined with NMF-based parameterization to maintain nonnegativity and low-rank structure during time integration.
- The method is applied to Markov grids and discretized reaction-diffusion PDEs, demonstrating superior preservation of nonnegativity compared to standard low-rank integrators.

---

[The RIGID Framework: Research-Integrated, Generative AI-Mediated Instructional Design](http://arxiv.org/abs/2603.12781)

- RIGID (Research-Integrated, Generative AI-Mediated Instructional Design): introduces a unified framework that systematically integrates Learning Sciences research across instructional design workflows by leveraging generative AI as a mediating tool.
- The framework organizes instructional design into four iterative phases—analysis, design, implementation, and evaluation—while utilizing AI to synthesize micro-, meso-, and macro-level contextual insights.
- By operationalizing research-based constraints through AI-driven prompt optimization and simulation, the framework supports context-sensitive instructional development while maintaining the central role of human expertise.

---

[FUNCTIONAL CLT FOR GENERAL SAMPLE COVARIANCE MATRICES](http://arxiv.org/abs/2603.12780)

- Functional CLT for Linear Spectral Statistics (LSS) introduces a framework for establishing CLTs for LSS of general sample covariance matrices under C3 test functions using Bernstein polynomial approximation, Martingale decomposition, Stieltjes transform, Cauchy’s integral formula, and Kolmogorov-Smirnov distance.
- The framework quantifies convergence rates of these CLTs in Kolmogorov-Smirnov distance by employing Bernstein polynomial approximation to handle test functions with continuous third-order derivatives.
- This approach extends existing results by relaxing analyticity requirements and providing a direct contour integral formulation for the centered LSS.

---

[On the strict-feedback form of hyperbolic distributed-parameter systems](http://arxiv.org/abs/2603.12779)

- Strict-feedback form for hyperbolic distributed-parameter systems: introduces a systematic transformation of heterodirectional hyperbolic PDEs and PDE-ODE systems into a strict-feedback form to facilitate recursive backstepping control design.
- The framework utilizes Volterra and Fredholm-type integral transformations to decouple dynamics and align system structures with ODE-based backstepping principles.
- The approach relies on exact controllability assumptions to ensure the existence of the strict-feedback representation and enables a direct path of actuation for complex distributed-parameter systems.

---

[Chvátal-Erdős condition for 2-factors with at most two components in graphs](http://arxiv.org/abs/2603.12776)

- Graph Theory Framework: introduces a sufficient condition for connected graphs of order at least 3κ(G)+3 to contain a 2-factor with at most two components when α(G) ≤ κ(G)+1.
- The research identifies an exceptional class of graphs, G, for which the stated condition does not guarantee the existence of such a 2-factor.
- The paper demonstrates that the derived bounds on graph order and independence number are sharp, providing a complete characterization for the existence of 2-factors with at most two components.

---

[Improving Critical Buildings Energy Resilience via Shared Autonomous Electric Vehicles - A Sequential Optimization Framework](http://arxiv.org/abs/2603.12771)

- SAEV (Shared Autonomous Electric Vehicle) Sequential Optimization Framework: introduces a dynamic optimization model to evaluate the potential of a centrally operated SAEV fleet to provide emergency V2B power services to critical buildings while maintaining mobility service levels.
- The framework utilizes a MILP formulation solved via an MPC algorithm to coordinate vehicle dispatch, charging, and discharging decisions in response to both passenger travel demands and power outage scenarios.
- A case study of the Ile-de-France region demonstrates that the SAEV fleet can effectively improve critical building energy resilience at a lower cost than traditional backup generators, provided the outage frequency is below a specific threshold.

---

[Every 3-connected {K1,4, K1,4 + e}-free split graph of order at least 13 is Hamilton-connected](http://arxiv.org/abs/2603.12770)

- Graph Theory Framework: introduces a structural analysis of 3-connected {K1,4, K1,4 + e}-free split graphs to determine their Hamilton-connectedness properties.
- The research establishes that any 3-connected {K1,4, K1,4 + e}-free split graph with at least 13 vertices is Hamilton-connected.
- The proof utilizes the existence of an I-cover and forbidden subgraph analysis to demonstrate the existence of a Hamiltonian (u, v)-path for any pair of vertices.

---

[A PROPERTY OF LOG-CONCAVE AND WEAKLY-SYMMETRIC DISTRIBUTIONS FOR TWO STEP APPROXIMATIONS OF RANDOM VARIABLES](http://arxiv.org/abs/2603.12767)

- Two-step approximation framework: introduces a generalization of classical risk measures by representing risk as a two-regime step function determined by an optimal threshold.
- The framework utilizes a quadratic loss function to derive optimal regime levels as conditional means, establishing uniqueness conditions for log-concave distributions.
- The research demonstrates that for weakly-symmetric log-concave distributions, the optimal threshold corresponds to the mean/median, while providing counterexamples for non-convex cases.

---

[Research on Linear Codes Holding q-Ary t-Designs](http://arxiv.org/abs/2603.12761)

- Research on Linear Codes Holding q-Ary t-Designs: introduces a systematic framework for constructing q-ary t-designs from linear codes using Standard Criterion, Puncturing-Shortening Criterion, and automorphism-group-based method.
- The paper establishes new characterizations for MDS and perfect codes while providing a complete classification for one-weight and two-weight codes that yield q-ary designs.
- Novel proofs are provided for infinite families of q-ary 2-designs derived from doubly-extended Reed-Solomon codes and specific trace codes using their transitive automorphism groups.

---

[SAP: Segment Any 4K Panorama](http://arxiv.org/abs/2603.12759)

- SAP: introduces a foundation model for 4K panoramic instance segmentation by reformulating the task as fixed-trajectory perspective video segmentation to leverage the temporal memory of SAM2.
- The framework utilizes a column-first zigzag scanning trajectory to convert equirectangular panoramas into temporally coherent perspective sequences, ensuring smooth viewpoint transitions for the streaming memory mechanism.
- By fine-tuning SAM2 on a large-scale synthetic dataset generated via InfiniGen, the model resolves topology-memory mismatches and achieves significant zero-shot mIoU gains on real-world 4K and 8K panoramas.

---

[Thermodynamics and null-geodesic of the Kerr-Newman black hole surrounded by quintessence and a cloud of string](http://arxiv.org/abs/2603.12756)

- Kerr-Newman black hole with quintessence and a cloud of string framework: investigates the thermodynamic properties and null-geodesic structure of a rotating black hole under the influence of MDR, quintessence, and a cloud of string.
- The study utilizes the Hamilton-Jacobi formalism to derive geodesic equations and analyzes the stability of circular photon orbits using the Lyapunov exponent and effective potential.
- Results demonstrate that MDR, quintessence, and string cloud parameters significantly modify Hawking temperature, entropy, heat capacity, and the stability of photon trajectories.

---

[Taming the Long Tail: Efficient Item-wise Sharpness-Aware Minimization for LLM-based Recommender Systems](http://arxiv.org/abs/2603.12752)

- EISAM (Efficient Item-wise Sharpness-Aware Minimization): introduces a novel optimization framework that improves tail-item performance in LLM-based Recommender Systems by adaptively regularizing the loss landscape at the item level.
- The framework utilizes a frequency-dependent weighting function to emphasize tail items and an efficient optimization procedure to maintain computational scalability for LLMs.
- Theoretical and empirical results demonstrate that EISAM achieves a tighter generalization bound and significantly boosts recommendation quality for tail items without sacrificing overall performance.

---

[Show, Don’t Tell: Detecting Novel Objects by Watching Human Videos](http://arxiv.org/abs/2603.12751)

- Show, Don’t Tell: introduces a self-supervised system that bypasses complex language descriptions by training a bespoke MOD on datasets automatically generated by the SODC pipeline from human demonstrations.
- The system utilizes HOIST-Former for initial segmentation, SAMURAI for temporal tracking, and DBSCAN for clustering to create temporally dense training data for the MOD.
- An integrated on-robot pipeline leverages ChatGPT-4o to generate plan skeletons and a scene graph to manage object-centric perception, enabling robots to replicate human sorting tasks with novel objects.

---

[TAOBENCH: Do Automated Theorem Prover LLMs Generalize Beyond MathLib?](http://arxiv.org/abs/2603.12744)

- TAOBENCH: introduces a benchmark designed to evaluate the robustness of LLMs in automated theorem proving when faced with novel definitional frameworks that deviate from standard libraries.
- The framework utilizes an agentic pipeline comprising a JiXia Tool, File-Lookup Tool, and LEAN Verifier to automatically extract self-contained, compilable Lean environments from formalized textbooks.
- Experimental results demonstrate that current LLMs exhibit a significant performance degradation when navigating non-standard definitional frameworks, highlighting a critical generalization gap in existing automated theorem proving systems.

---

[ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](http://arxiv.org/abs/2603.12740)

- ToolTree: introduces a Monte Carlo tree search-inspired planning paradigm that optimizes LLM agent tool usage through dual-feedback signals and bidirectional pruning.
- The framework integrates pre-execution scoring to guide selection and post-execution scoring to refine trajectories, enabling efficient, training-free multi-step reasoning.
- By employing bidirectional pruning, ToolTree effectively manages search complexity and improves accuracy-per-compute under fixed budgets across diverse tool-use benchmarks.

---

[Conflict Mitigation in Shared Environments using Flow-Aware Multi-Agent Path Finding](http://arxiv.org/abs/2603.12736)

- FA-MAPF: introduces a framework that integrates learned motion patterns of uncontrollable agents into centralized MAPF algorithms to reduce conflicts.
- The approach utilizes CLiFF-maps and SWGMMs to represent environmental flow, which are then incorporated into a guidance graph to influence path planning.
- Experimental results demonstrate that FA-MAPF significantly reduces conflicts with uncontrollable agents by up to 55% while maintaining task efficiency.

---

[On Using Machine Learning to Early Detect Catastrophic Failures in Marine Diesel Engines](http://arxiv.org/abs/2603.12733)

- Machine Learning-based catastrophic failure detection framework: introduces a methodology for early detection of catastrophic failures in marine Diesel engines by analyzing the derivatives of deviations between actual and predicted sensor measurements.
- The framework utilizes a VAE to augment limited historical data, enabling the training of supervised regression models like RF to predict healthy engine behavior.
- By monitoring the first and second derivatives of prediction errors, the system identifies abrupt anomalies faster than traditional threshold-based alarms, facilitating timely operator intervention.

---

[IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration](http://arxiv.org/abs/2603.12719)

- IGASA: introduces a robust point cloud registration framework that utilizes a Hierarchical Pyramid Architecture (HPA) for multi-scale feature extraction, a Hierarchical Cross-Layer Attention (HCLA) module for semantic alignment, and an Iterative Geometry-Aware Refinement (IGAR) module for precise pose estimation.
- The framework employs a Gated Fusion Mechanism and skip-attention modules (SGIRA and SAIGA) to bridge the semantic gap between multi-resolution features while suppressing noise and irrelevant background.
- IGAR iteratively optimizes rotation and translation parameters using a dynamic geometric consistency weighting strategy to effectively suppress outliers and enhance registration accuracy.

---

[Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](http://arxiv.org/abs/2603.12711)

- TPSNet: introduces a dual-prior framework for unsupervised cross-domain image retrieval by leveraging learnable domain prompts as text priors and domain-invariant phase features as phase priors to bridge distribution gaps.
- The framework utilizes a Domain Prompt Generation Module to refine pseudo-labels and a Text-Phase Dual Priors Network to integrate semantic and structural information through a cross-attention mechanism.
- TPSNet effectively extracts domain-invariant semantic representations by combining RGB features with phase-spectrum information, significantly outperforming state-of-the-art methods on UCDIR benchmarks.

---

[AI Planning Framework for LLM-Based Web Agents](http://arxiv.org/abs/2603.12710)

- AI Planning Framework for LLM-Based Web Agents: introduces a taxonomy mapping LLM-based web agent architectures to traditional planning paradigms, including Step-by-Step (BFS), Tree Search (Best-First), and Full-Plan-in-Advance (DFS) approaches.
- The paper proposes a comprehensive evaluation framework utilizing LLMs-as-judges to assess agent trajectories beyond binary success rates, incorporating metrics like Recovery Rate, Repetitiveness Rate, and Element Accuracy.
- The authors validate their framework by comparing a baseline Step-by-Step agent against a novel Full-Plan-in-Advance implementation using a new dataset of 794 human-labeled trajectories from the WebArena benchmark.

---

[QUANTITATIVE STRATIFICATION AND GLOBAL REGULARITY FOR 1/2-HARMONIC MAPPINGS](http://arxiv.org/abs/2603.12709)

- Quantitative Stratification and Global Regularity for 1/2-Harmonic Mappings: introduces a quantitative stratification theory for singular sets of stationary and minimizing 1/2-harmonic maps into compact Riemannian manifolds.
- The framework utilizes harmonic extensions to the half-space to overcome the lack of a known monotonicity formula for 1/2-harmonic maps, enabling the application of quantitative symmetry and Reifenberg-type theorems.
- The research establishes sharp growth estimates on the volume of tubular neighborhoods around singular points and proves the rectifiability of each singular stratum, providing optimal first-order regularity estimates.

---

[Synergies, Trade-offs, and Structural Pathways: A Directed Network Approach to SDG Prioritisation](http://arxiv.org/abs/2603.12699)

- Directed Network Approach to SDG Prioritisation: introduces a multi-step framework that utilizes a Lagged Correlation Estimator, Directed Weighted Network, Synergy-Trade-off Classifier, Opsahl Out-Centrality Measure, Flow-based Clustering, and Infomap Implementation to identify high-impact policy entry points.
- The framework models SDG interlinkages as directed weighted networks to capture asymmetric influence and structural propagation, moving beyond traditional undirected correlation-based approaches.
- By integrating flow-based clustering with weighted centrality, the approach enables the identification of structurally distinct subsystems, facilitating a diversified policy prioritisation strategy that minimizes redundancy.

---

[EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning](http://arxiv.org/abs/2603.12698)

- EvolveCoder: introduces a solution-conditioned and adversarial verification framework that iteratively refines test cases based on execution behaviors of candidate solutions to improve code generation reinforcement learning.
- The framework utilizes a multi-model solution pool and alternating adversarial and discriminative test generation to increase test difficulty and discriminative power while reducing redundancy.
- Empirical results demonstrate that training LLMs on the resulting EVOLVECODER-22K dataset yields stable optimization and consistent performance gains across diverse coding benchmarks.

---

[Semantic-Aware 6G Network Management through Knowledge-Defined Networking](http://arxiv.org/abs/2603.12695)

- KDN-driven Semantic Communication Framework: introduces a management-oriented architecture for 6G networks that integrates semantic reasoning, semantic-aware routing, and closed-loop distortion control to preserve task-relevant meaning.
- The framework utilizes a Knowledge Plane to coordinate network-wide semantic management, separating high-level decision-making from data forwarding in the Data Plane.
- By treating semantic fidelity as a first-class control variable, the system dynamically adapts encoding and routing to maintain performance under mobility and congestion.

---

[RXNRECer Enables Fine-grained Enzymatic Function Annotation through Active Learning and Protein Language Models](http://arxiv.org/abs/2603.12694)

- RXNRECer: introduces a transformer-based ensemble framework for direct, fine-grained enzyme-catalyzed reaction prediction without relying on Enzyme Commission numbers.
- The framework integrates a PLM-based reaction classifier, a dynamic ensemble module for robust multi-source integration, and a GLM-based interpretability module for mechanistic insights.
- By employing an active learning strategy, the framework achieves high accuracy and scalability while effectively addressing challenges in proteome-wide annotation and enzyme promiscuity identification.

---

[CM-Bench: A Comprehensive Cross-Modal Feature Matching Benchmark Bridging Visible and Infrared Images](http://arxiv.org/abs/2603.12690)

- CM-Bench: introduces a comprehensive evaluation framework for cross-modal feature matching, utilizing a MobileNetV4-Conv-Small backbone, Feature Fusion module, MLP head, and various Preprocessing branches to assess 30 distinct Feature matching algorithms.
- The framework incorporates an adaptive preprocessing front-end that dynamically selects enhancement strategies based on image-pair characteristics to improve matching performance across infrared and visible modalities.
- The study also introduces the ThermoSat dataset, providing manually annotated infrared-satellite image pairs to facilitate robust geo-localization evaluation under both base and challenging perturbed conditions.

---

[STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](http://arxiv.org/abs/2603.12688)

- STRAP-ViT: introduces a training-free, model-agnostic defense mechanism for ViTs that detects adversarial patches via Jensen-Shannon Divergence and mitigates their impact through randomized composite token transformations.
- The framework identifies anomalous tokens by measuring statistical divergence from clean reference distributions and applies targeted transformations to neutralize adversarial noise without requiring additional training.
- By operating directly in the semantic feature space at the token level, the approach effectively maintains high clean-case accuracy while providing robust protection against various localized adversarial patch attacks.

---

[RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](http://arxiv.org/abs/2603.12685)

- RSONet: introduces a two-stage framework for RGB-T salient object detection that utilizes a region guidance stage to determine modality dominance and a saliency generation stage for feature optimization.
- The region guidance stage employs SwinTransformer, CI module, and SF module to generate guidance maps for similarity-based modality selection.
- The saliency generation stage integrates SO, DDE, and MIS modules to refine multi-level features and produce high-quality saliency maps by mitigating modality inconsistencies.

---

[Third type of spacetime with the coexistence of integrability and non-integrability](http://arxiv.org/abs/2603.12674)

- Third type of spacetime framework: introduces a classification of spacetimes where null geodesics remain integrable while timelike geodesics exhibit non-integrable, chaotic dynamics.
- The framework utilizes time-transformed Hamiltonian systems and explicit symplectic integrators to analyze the transition between regular and chaotic particle motion in various black hole metrics.
- The research demonstrates that conformal transformations and specific external fields can break the integrability of massive particle motion without affecting the integrability of null geodesics.

---

[RESULTS FOR BLOW-UP AND SHARP LIFESPAN ESTIMATES TO A WEAKLY COUPLED SYSTEM OF STRUCTURALLY DAMPED WAVE EQUATIONS WITH CRITICAL NONLINEARITIES](http://arxiv.org/abs/2603.12673)

- Weakly coupled system of semilinear structurally damped wave equations: investigates the global existence and finite-time blow-up of small data solutions for a system involving moduli of continuity and fractional Laplacian operators.
- The research establishes sharp conditions for moduli of continuity to determine the global existence versus finite-time blow-up of solutions within the critical curve in the p-q plane.
- The authors derive sharp lifespan estimates for local solutions by constructing novel test functions capable of handling the nonlocal fractional Laplacian operator in non-compact domains.

---

[HyGra: Accelerating Network-State Simulation for LLM Training in DCNs via Adaptive Packet-Flow Granularity](http://arxiv.org/abs/2603.12671)

- HyGra: introduces a hybrid-granularity network-state simulator that adaptively switches between packet-level simulation and flow-level simulation to balance efficiency and fidelity during LLM training.
- The framework utilizes a control layer to identify steady-state phases and an execution layer to perform granularity transitions, including a Transformer-based mechanism for near-lossless state restoration.
- By exploiting the periodic communication patterns inherent in LLM training, HyGra achieves significant speedups on single-machine deployments without requiring specialized hardware.

---

[Advancing Machine Learning Applications in Quantum Few-Body Systems](http://arxiv.org/abs/2603.12668)

- Neural Network Quantum Few-Body Framework: introduces a generalized neural network architecture for solving quantum few-body systems by combining a Multilayer Perceptron (MLP) with advanced sampling techniques like the Metropolis-Adjusted Langevin Algorithm (MALA) and adaptive parameter adjustment.
- The framework utilizes Jacobi coordinate transformations to isolate internal degrees of freedom and employs GPU-accelerated computation to achieve stable convergence and scalability across systems with varying particle masses and interaction types.
- Experimental results demonstrate that the integration of MALA and adaptive step size mechanisms significantly reduces hyperparameter sensitivity and training oscillations compared to traditional random walk approaches in few-body quantum simulations.

---

[Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](http://arxiv.org/abs/2603.12663)

- MPO framework: introduces a method for outdoor place categorization using CNNs that process omnidirectional depth and reflectance images derived from LiDAR data.
- The architecture incorporates HCC and RWMP layers to ensure rotational invariance and robust feature extraction from panoramic inputs.
- Experimental results demonstrate that multi-modal fusion, particularly via Softmax Average, outperforms uni-modal approaches and traditional hand-engineered feature methods.

---

[Continual Learning in Large Language Models: Methods, Challenges, and Opportunities](http://arxiv.org/abs/2603.12658)

- Continual Learning in Large Language Models: introduces a comprehensive taxonomy of continual learning methodologies for LLMs, structured across three core training stages: Continual Pre-training, Continual Fine-tuning, and Continual Alignment.
- The paper categorizes existing approaches into Rehearsal-based methods, Data augmentation methods, Process optimization methods, Regularization-based methods, and Architecture-based methods to mitigate catastrophic forgetting.
- It provides a rigorous comparative analysis of these methodologies, discusses essential evaluation metrics like forgetting rates and knowledge transfer efficiency, and outlines future research directions including multimodal and online continual learning.

---

[Beyond the Merger - Quasar - Quench Paradigm I: Mergers are neither necessary nor sufficient to quench central galaxies in IllustrisTNG](http://arxiv.org/abs/2603.12651)

- IllustrisTNG: introduces a comprehensive analysis of central galaxy quenching, demonstrating that mergers are neither necessary nor sufficient for quenching in the simulation, utilizing Random Forest classifier and Random Forest regressor to evaluate the predictive power of merger histories against intrinsic galaxy properties.
- The study employs SubLink merger trees and AREPO-based simulation data to track over 11,000 central galaxies, revealing that quenching is primarily governed by secular processes rather than merger-driven events.
- Machine learning analyses confirm that supermassive black hole mass and stellar mass are the dominant predictors of quenching, while merger-related parameters provide negligible predictive value when controlling for intrinsic galaxy properties.

---

[Autonomous Integration and Improvement of Robotic Assembly using Skill Graph Representations](http://arxiv.org/abs/2603.12649)

- Skill Graph: introduces a framework for autonomous integration and continuous improvement of robotic assembly systems by organizing robot capabilities into a graph of semantically defined skills.
- The framework utilizes a best-first search planner to bridge high-level semantic task specifications with low-level robot execution, supporting both sequential and asynchronous multi-robot operations.
- By integrating structured data logging with vision-based perception, the system enables closed-loop performance improvement, failure diagnosis, and autonomous refinement of skill policies.

---

[LightMoE: Reducing Mixture-of-Experts Redundancy through Expert Replacing](http://arxiv.org/abs/2603.12645)

- LightMoE: introduces a novel expert compression paradigm that replaces redundant experts in MoE models with parameter-efficient modules to reduce memory footprint while preserving performance.
- The framework utilizes Adaptive Expert Selection to identify less critical experts, Hierarchical Expert Construction to group them into shared bases with Low-Rank Adaptation, and Annealed Expert Replacement to ensure smooth training transitions.
- Experimental results demonstrate that LightMoE achieves performance comparable to LoRA fine-tuning at a 30% compression ratio and significantly outperforms existing methods at a 50% compression ratio.

---

[OFDM Waveform for Monostatic ISAC in 6G: Vision, Approach, and Research Directions](http://arxiv.org/abs/2603.12641)

- Monostatic ISAC: introduces a framework for enabling radar-like sensing capabilities on wireless communication devices by leveraging OFDM waveforms and co-located transmitter/receiver architectures.
- The approach utilizes hardware-level techniques including phased-array beamforming, self-interference cancellation, and RF switching to facilitate simultaneous or rapid-alternation sensing and communication.
- The framework enables 4D environmental perception—including range, Doppler velocity, azimuth, and elevation—by processing channel impulse response measurements through deep learning models for applications like human pose estimation.

---

[Using a Human-AI Teaming Approach to Create and Curate Scientific Datasets with the SCILIRE System](http://arxiv.org/abs/2603.12638)

- SCILIRE: introduces a Human-AI Teaming (HAT) framework for scientific dataset curation that integrates Document Preprocessing Module, LLM-based Record Generation Module, and Verification Support Module to mitigate hallucinations through iterative human feedback.
- The system utilizes Dynamic Sampling to improve LLM performance by retrieving relevant human-corrected examples via BM25 Retrieval, while employing Hungarian Maximum-Matching Algorithm to align records from multiple parsing pipelines.
- SCILIRE incorporates Cascade R-CNN and UniTable for robust table and figure extraction, enabling researchers to scale dataset creation while maintaining high fidelity through expert-in-the-loop verification.

---

[Batched Kernelized Bandits: Refinements and Extensions](http://arxiv.org/abs/2603.12627)

- BPE (Batched Pure Exploration): introduces refined regret bounds for batched kernelized bandits by optimizing batch sizes and removing redundant factors in the regret analysis.
- Robust-BPE (Robust Batched Pure Exploration): extends the framework to adversarial settings by incorporating perturbation-aware confidence bounds to maintain performance under worst-case function shifts.
- The research establishes near-optimal regret bounds for both fixed and adaptive batching strategies while demonstrating that adaptive batching provides no significant minimax advantage over fixed batching.

---

[On Moy-Prasad quotients over Laurent series fields](http://arxiv.org/abs/2603.12623)

- Moy-Prasad stratification framework: introduces a stratification by conjugacy classes of twisted Levi subgroups on Moy-Prasad quotients of connected reductive groups over Laurent series fields.
- The paper establishes that semistable orbits in these quotients are controlled by twisted Levi subgroups, providing a geometric foundation for the local Langlands program.
- The main result, Theorem 3.19, proves an isomorphism of Artin stacks relating the stratification of the Moy-Prasad quotient to the corresponding structures of twisted Levi subgroups.

---

[A step towards the Erdős–Rogers problem](http://arxiv.org/abs/2603.12610)

- Erdős-Rogers problem framework: introduces a probabilistic method combined with multi-layer extremum structures to establish new upper bounds for the Erdős-Rogers function.
- The approach utilizes a variant of the Erdős-Hajnal stepping-up lemma to construct K-free hypergraphs with specific induced subgraph properties.
- This research confirms the Mubayi-Suk conjecture for specific hypergraph parameters by analyzing local maxima sequences within the stepping-up construction.

---

[InterDeepResearch: Enabling Human-Agent Collaborative Information Seeking through Interactive Deep Research](http://arxiv.org/abs/2603.12608)

- InterDeepResearch: introduces a human-agent collaborative system that utilizes a hierarchical Research Context Management Framework to organize information, actions, and sessions for improved observability and steerability.
- The system integrates three coordinated views—Action Flow, Action Dependency Graph, and Research Information—to provide users with real-time visibility and interactive control over the LLM-based agent's research process.
- By implementing dynamic context reduction and cross-action backtrace mechanisms, the framework effectively manages long-horizon research context while enabling users to verify evidence provenance and steer research directions mid-process.

---

[CarPLAN: Context-Adaptive and Robust Planning with Dynamic Scene Awareness for Autonomous Driving](http://arxiv.org/abs/2603.12607)

- CarPLAN: introduces an imitation learning-based motion planning framework that enhances spatial awareness and context-adaptive decision-making through Displacement-Aware Predictive Encoding and a Mixture of Experts decoder.
- The framework utilizes a Scene-Aware Router to dynamically select specialized experts based on scene context, improving robustness in complex and dynamic traffic environments.
- CarPLAN achieves state-of-the-art performance on the nuPlan and Waymax benchmarks by explicitly modeling relational spacing between the autonomous vehicle and surrounding scene elements.

---

[How GenAI Mentor Configurations Shape Early Collaborative Dynamics: A Classroom Comparison of Individual and Shared Agents](http://arxiv.org/abs/2603.12600)

- GenAI Mentor Configurations: investigates how shared versus individual AI access structures reorganize collaborative regulation and interaction dynamics in classroom settings.
- Shared-AI access functions as a collective cognitive anchor promoting convergence-oriented collaboration, whereas individual-AI access leads to fragmented, repair-oriented interaction patterns requiring increased teacher orchestration.
- The study demonstrates that AI configuration acts as a structural design variable that fundamentally shapes the regulatory ecology of collaborative learning beyond mere functional affordances.

---

[A Prediction-as-Perception Framework for 3D Object Detection](http://arxiv.org/abs/2603.12599)

- PAP (Prediction-As-Perception): introduces a biomimetic architecture that integrates historical prediction results as input queries for the current frame's perception module to enhance 3D object detection accuracy.
- The framework utilizes a Perception Module (processes images and queries), Prediction Module (forecasts future object positions), Queries Bank (stores temporal position queries), and Embedding Layer (maps coordinates to query dimensions) to maintain temporal consistency.
- By replacing randomly generated queries with historical prediction-based queries, the model improves tracking performance by 10% and increases inference speed by 15% on the nuScenes dataset.

---

[FEYNMAN: Knowledge-Infused Diagramming Agent for Scalable Visual Designs](http://arxiv.org/abs/2603.12597)

- FEYNMAN: introduces a scalable diagram generation pipeline that decouples knowledge elicitation from visual production using an LLM-based Planner, PENROSE Rendering Engine, and a Visual Judge Panel.
- The framework utilizes an Iterative Visual-Refine Algorithm where LLMs act as judges to provide feedback on Substance Code Generator outputs, ensuring visual consistency and semantic accuracy.
- FEYNMAN synthesizes over 100k diagram-caption pairs and introduces the DIAGRAMMA benchmark to evaluate the visual reasoning capabilities of LLMs.

---

[Early Pruning for Public Transport Routing](http://arxiv.org/abs/2603.12592)

- Early Pruning: introduces a low-overhead optimization technique that accelerates the transfer relaxation phase of RAPTOR-based routing algorithms by pre-sorting transfer edges and terminating iterations when paths become non-competitive.
- The method utilizes a simple pruning rule that discards longer transfers at a stop once they cannot yield an earlier arrival than the current best solution, ensuring optimality is maintained.
- Experimental results across multiple RAPTOR variants and real-world transit networks demonstrate query time reductions of up to 57%, with performance gains correlating positively with graph density.

---

[RTD-Guard: A Black-Box Textual Adversarial Detection Framework via Replacement Token Detection](http://arxiv.org/abs/2603.12582)

- RTD-Guard: introduces a training-free, black-box adversarial detection framework that leverages a pre-trained RTD Discriminator to localize and mask suspicious tokens, subsequently identifying adversarial examples by measuring the confidence shift in the Victim Model.
- The framework utilizes an off-the-shelf RTD Discriminator to detect contextually inconsistent tokens, which are characteristic of word-level adversarial perturbations, without requiring adversarial training data or white-box access.
- By masking identified tokens and observing the resulting confidence change in the Victim Model, RTD-Guard achieves efficient, constant-time detection with minimal query overhead.

---

[DINOLight: Robust Ambient Light Normalization with Self-supervised Visual Prior Integration](http://arxiv.org/abs/2603.12579)

- DINOLight: introduces a robust ambient light normalization framework that integrates self-supervised DINOv2 features as visual priors to restore images degraded by complex lighting conditions.
- The framework utilizes an Adaptive Feature Fusion Module (AFFM) to combine hierarchical DINOv2 features and an Auxiliary Cross-Attention (ACA) mechanism to inject these priors into a multi-scale restoration network.
- By operating in both spatial and frequency domains, the architecture achieves superior performance on ambient light normalization and generalizes effectively to shadow removal tasks without requiring explicit mask priors.

---

[Streaming REST APIs for Large Financial Transaction Exports from Relational Databases](http://arxiv.org/abs/2603.12566)

- Streaming REST API Architecture: introduces a streaming-based design that integrates Database, Cursor, Record Map, Encode, Serialize, and HTTP Stream to enable incremental delivery of large financial datasets.
- The architecture replaces traditional memory-intensive buffered response models with a continuous pipeline that processes records sequentially to maintain stable memory usage.
- By utilizing the JAX-RS StreamingOutput interface, the system achieves immediate response initiation and improved scalability for high-volume transaction export operations.

---

[Speech-Worthy Alignment for Japanese SpeechLLMs via Direct Preference Optimization](http://arxiv.org/abs/2603.12565)

- Speech-Worthy Alignment for Japanese SpeechLLMs via Direct Preference Optimization: introduces a preference-based alignment approach to adapt Japanese SpeechLLMs to generate concise, conversational outputs suitable for TTS synthesis.
- The framework utilizes DPO and SFT to align SpeechLLMs, incorporating a projector layer, an LLM, and a spoken system prompt to optimize for auditory comprehension.
- The authors introduce SpokenElyza, a benchmark for evaluating Japanese speech-worthiness, demonstrating that preference training and parameter-efficient tuning significantly improve output quality for spoken dialog.

---

[Consistent and powerful CUSUM change-point test for panel data with changes in variance](http://arxiv.org/abs/2603.12561)

- TU: introduces a modified CUSUM-based statistical test for detecting variance change-points in panel data models with alpha-mixing errors, utilizing Panel data model, Alpha-mixing error sequences, CUSUM statistic, Long-run variance estimator, Sparse variance change detection, Monte Carlo simulation, and Real data application.
- The framework aggregates squared residuals across panels prior to normalization to preserve signal strength in scenarios where variance changes are sparse or heterogeneous.
- The proposed test demonstrates superior detection power and sensitivity compared to existing methods, particularly in identifying structural breaks within financial market indices.

---

[Towards Output-Optimal Uniform Sampling and Approximate Counting for Join-Project Queries](http://arxiv.org/abs/2603.12560)

- Join-Project Query Framework: introduces asymptotically optimal algorithms for uniform sampling and approximate counting in join-project queries by utilizing Auxiliary indices, Rejection-based sampling strategy, and Hybrid counting reduction.
- The framework employs Heavy-light partitioning and KMV summary to achieve sublinear-time performance for matrix, star, and chain query classes.
- It establishes information-theoretic lower bounds via communication complexity, demonstrating the optimality of the proposed algorithms against both combinatorial and algebraic techniques.

---

[Involution Game with Migration and Spatial Heterogeneity of Social Resources](http://arxiv.org/abs/2603.12558)

- Involution Game with Migration and Spatial Heterogeneity of Social Resources: introduces an evolutionary game model to analyze how agent mobility and resource distribution influence the emergence of excessive competitive behavior.
- The model utilizes a lattice-based environment where agents choose between high-effort and low-effort strategies based on local payoff comparisons and migration opportunities.
- Theoretical analysis using mean-field theory confirms that resource disparity drives agent agglomeration, which in turn modulates the intensity of involutionary competition.

---

[NONTRIVIAL WEAK SOLUTIONS OF THE STATIONARY KDV EQUATION IN SHARP L^p SPACES](http://arxiv.org/abs/2603.12555)

- Convex Integration Framework: introduces a method to construct non-trivial weak solutions to the stationary Korteweg–de Vries (KdV) equation in L^p spaces for p < 2 using a convex integration scheme.
- The framework utilizes intermittent building blocks and Littlewood-Paley projectors to manage dispersion, oscillation, and Nash errors within negative regularity Sobolev spaces.
- This approach establishes a sharp dichotomy where solutions are smooth for p = 2 but admit non-smooth, low-regularity weak solutions for p < 2.

---

[REDUCED-ORDER VARIATIONAL DETERMINISTIC-PARTICLE-BASED SCHEME FOR FOKKER-PLANCK EQUATIONS IN MICROSCOPIC POLYMER DYNAMICS](http://arxiv.org/abs/2603.12550)

- POD-MOR: introduces a model reduction framework that integrates proper orthogonal decomposition with a variational deterministic-particle-based scheme to accelerate the computation of microscopic Fokker-Planck equations for multi-bead polymers.
- The framework utilizes a shared POD matrix to construct a low-dimensional subspace, significantly reducing the degrees of freedom while maintaining distributional accuracy for 3D complex fluid simulations.
- Numerical validation demonstrates that the reduced-order model achieves substantial computational efficiency gains, requiring only a fraction of the original time for multi-bead chain polymers while keeping relative errors within the benchmark range.

---

[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](http://arxiv.org/abs/2603.12547)

- Deco-Mamba: introduces a decoder-centric architecture for 2D medical image segmentation that utilizes a hybrid CNN-Transformer encoder and a Mamba-based decoder to balance global context modeling with fine-grained structural reconstruction.
- The framework integrates Co-Attention Gate (CAG) for adaptive feature fusion, Vision State Space Mamba Block (VSSMB) for efficient long-range dependency modeling, and Deformable Residual Block (DRB) for spatial refinement.
- A novel Multi-Scale Distribution-Aware (MSDA) supervision strategy is employed to enforce distributional consistency across decoding stages, improving boundary precision and robustness under domain shifts.

---

[Load Balancing in Non-Terrestrial Networks Using Free Space Optical Inter-satellite Links](http://arxiv.org/abs/2603.12546)

- ACMCFP (Anycast Multi-Commodity Flow Problem): introduces a fairness-driven load balancing strategy for MEO satellite constellations using Satellite Constellation, Ground Stations (GS), Feeder Links (FL), Optical Inter-satellite Links (ISL), and a Dual-Simplex Optimization Algorithm.
- The framework models the satellite network as a time-varying graph to dynamically offload data traffic from satellites with weather-degraded Feeder Links (FL) to adjacent satellites via Optical Inter-satellite Links (ISL).
- By solving a max-min fairness optimization problem, the approach significantly improves the minimum per-satellite throughput and reduces performance variance during adverse weather conditions.

---

[Embedded Quantum Machine Learning in Embedded Systems: Feasibility, Hybrid Architectures, and Quantum Co-Processors](http://arxiv.org/abs/2603.12540)

- EQML: introduces a feasibility-driven framework for integrating quantum machine learning into resource-constrained edge platforms through hybrid offloading or on-device co-processing architectures.
- The paper formalizes two implementation pathways, utilizing either a remote QPU/QPA for asynchronous tasks or a local QPU module integrated via a low-latency interconnect for tighter quantum-classical loops.
- It identifies critical engineering barriers including latency determinism, NISQ noise, and data encoding overhead, while proposing a roadmap for trustworthy deployment using adversarial evaluation and hardware-software co-design.

---

[Tighter monogamy and polygamy relations in multiparty quantum systems](http://arxiv.org/abs/2603.12539)

- Tighter monogamy and polygamy relations in multiparty quantum systems: introduces a new mathematical inequality to derive improved monogamy and polygamy relations for multipartite quantum systems.
- The framework utilizes a family of improved inequalities to provide tighter lower bounds for monogamy relations and tighter upper bounds for polygamy relations compared to existing literature.
- The proposed relations are generalized from tripartite systems to general N-party quantum systems and validated through illustrative examples demonstrating superior accuracy in characterizing entanglement distribution.

---

[Heterogeneous Elasticities, Aggregation, and Retransformation Bias](http://arxiv.org/abs/2603.12536)

- DREAM: introduces a specification-robust debiased machine learning estimator to address retransformation bias in log-log regressions by targeting the average arithmetic mean elasticity.
- The framework utilizes Neyman-orthogonalized machine learning and control function approaches to provide consistent estimates of arithmetic mean elasticities under heterogeneous responses.
- The paper demonstrates that standard log-log OLS estimates correspond to geometric mean elasticities, which are often insufficient for decision problems requiring arithmetic mean elasticities.

---

[Entanglement-Assisted Discrimination of Nonlocal Sets of Orthogonal States](http://arxiv.org/abs/2603.12535)

- Entanglement-Assisted Discrimination of Nonlocal Sets of Orthogonal States: introduces resource-efficient protocols for the local discrimination of nonlocal orthogonal state sets by incorporating CNOT gates and shared entanglement.
- The research demonstrates that incorporating CNOT gates into discrimination protocols significantly reduces the required entanglement consumption for genuinely nonlocal GHZ bases in four- and five-qubit systems.
- The paper provides systematic protocols for strongly nonlocal asymmetric and symmetric orthogonal product sets in four- and five-partite systems, utilizing various entangled resources to optimize local distinguishability.

---

[Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](http://arxiv.org/abs/2603.12533)

- HINT (Hand Intent Tokens): introduces a gesture-aware framework for egocentric video question answering by interleaving 3D hand keypoint tokens with visual tokens to resolve deictic references.
- The framework utilizes a WiLoR-based 3D hand reconstruction module and a lightweight Keypoint Adapter to explicitly encode pointing intent into the LLM input sequence.
- The approach is evaluated on the newly introduced EGOPOINTVQA dataset, demonstrating significant performance gains in grounding deictic expressions compared to standard LLMs.

---

[Self-Confirming Mechanisms](http://arxiv.org/abs/2603.12532)

- Self-Confirming Mechanisms: introduces a fixed-point approach to mechanism design where mechanisms are evaluated based on their stability relative to the information endogenously generated by agent behavior.
- The paper establishes a fictitious revelation principle, demonstrating that any incentive-compatible mechanism can be represented as a fictitious direct mechanism using a filter to preserve informational content.
- By applying a grain-of-truth refinement to address equilibrium permissiveness, the framework provides an endogenous justification for the Myerson optimal mechanism in monopoly settings.

---

[TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning](http://arxiv.org/abs/2603.12529)

- TERMINATOR: introduces a novel inference-time early-exit strategy for LLMs that leverages hindsight-optimal reasoning lengths to mitigate overthinking by training a binary probe classifier on LRM hidden states.
- The framework utilizes an automated pipeline to curate a dataset of optimal reasoning lengths by extracting, identifying, and verifying the first logical arrival of the final answer within the CoT.
- TERMINATOR achieves significant reductions in CoT lengths across multiple benchmarks by predicting the earliest point at which an LRM generates its final answer, thereby avoiding redundant computation.

---

[HumDex: Humanoid Dexterous Manipulation Made Easy](http://arxiv.org/abs/2603.12260)

- HumDex: introduces a portable teleoperation system for humanoid whole-body dexterous manipulation that utilizes IMU-based motion tracking and a learning-based hand retargeting module to enable efficient, high-quality data collection.
- The system employs a two-stage imitation learning framework that leverages diverse human motion data to learn generalizable priors before fine-tuning on robot data to bridge the embodiment gap.
- HumDex demonstrates significant improvements in teleoperation efficiency and policy generalization across challenging loco-manipulation tasks by overcoming visual occlusion and embodiment limitations inherent in previous approaches.

---

[PISmith: Reinforcement Learning-based Red Teaming for Prompt Injection Defenses](http://arxiv.org/abs/2603.13026)

- PISmith: introduces a reinforcement learning-based red-teaming framework that systematically evaluates prompt injection defenses by training an Attack LLM to optimize injected prompts in a black-box setting.
- The framework addresses reward sparsity in RL-based red teaming by employing Adaptive Entropy Regularization to sustain exploration and Dynamic Advantage Weighting to amplify learning from scarce successful attacks.
- Extensive evaluation across 13 benchmarks demonstrates that PISmith consistently achieves higher attack success rates than existing static, search-based, and RL-based baselines against state-of-the-art prompt injection defenses.

---

[Uncovering Security Threats and Architecting Defenses in Autonomous Agents: A Case Study of OpenClaw](http://arxiv.org/abs/2603.12644)

- OpenClaw: introduces a comprehensive security analysis of autonomous agents by mapping vulnerabilities across cognitive, software, and system dimensions.
- The paper proposes FASA (Full-Lifecycle Agent Security Architecture) to secure agents through layered isolation, dynamic intent verification, and cross-layer reasoning-action correlation.
- The research demonstrates that traditional content-centric security is insufficient for autonomous agents, necessitating a shift toward execution sandboxing and behavioral monitoring.

---

[NormCode Canvas: Making LLM Agentic Workflows Development Sustainable via Case-Based Reasoning](http://arxiv.org/abs/2603.13443)

- NormCode Canvas: introduces a two-level Case-Based Reasoning architecture for multi-step LLM workflows that utilizes NormCode (semi-formal planning language) to ensure structural isolation and self-contained execution checkpoints.
- The system manages Level 1 concrete cases as suspended runtimes and Level 2 abstract cases as executable plans, enabling systematic debugging, reuse, and revision of agentic workflows.
- By enforcing strict scope rules, the framework provides direct checkpoint inspection, pre-execution structural review, and scope-bounded selective re-execution to improve the sustainability of complex LLM agentic systems.

---

[Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents](http://arxiv.org/abs/2603.12634)

- BAVT: introduces a training-free inference-time framework that models multi-hop reasoning as a dynamic search tree guided by step-level value estimation within a single LLM backbone.
- The framework employs a budget-conditioned node selection mechanism that uses the remaining resource ratio to transition from broad exploration to greedy exploitation as the budget depletes.
- BAVT utilizes a residual value predictor to score relative progress, enabling reliable pruning of uninformative or redundant tool calls to maximize performance under strict resource constraints.

---

[ChainFuzzer: Greybox Fuzzing for Workflow-Level Multi-Tool Vulnerabilities in LLM Agents](http://arxiv.org/abs/2603.12614)

- ChainFuzzer: introduces a greybox testing framework that discovers and reproduces multi-tool vulnerabilities in LLM agents by combining Sink-related tool-chain extraction, Tool-chain prompt generation, and Feedback-driven fuzzing &amp; vulnerability validation.
- The framework utilizes an LLM-based constraint generator and solver to synthesize stable prompts that drive LLMs to execute specific multi-tool chains, while employing sink-specific oracles to validate security-relevant effects.
- ChainFuzzer identifies that most multi-tool vulnerabilities arise from cross-tool dataflow dependencies, and it effectively bypasses LLM guardrails using payload mutation strategies to produce auditable proof-of-concepts.

---

[Large Language Models as Delivery Rider: Generating Instant Food Delivery Riders’ Routing Decision with LLM Agent Framework](http://arxiv.org/abs/2603.12559)

- LLM-DR (Large Language Models as Delivery Rider): introduces a framework that simulates heterogeneous delivery rider decision-making by grounding LLM agents in empirically-derived personas and employing a structured Chain-of-Thought process for routing.
- The framework utilizes unsupervised clustering to identify four distinct rider work strategies, which are then instantiated as LLM agents to interact within a high-fidelity, real-world order simulation environment.
- By leveraging Gemini-2.5-Flash, the agents perform strategic reasoning to evaluate order attributes against persona-specific motivations, enabling the analysis of system-level mobility outcomes in on-demand delivery markets.

---

#### 12th March 2026


[On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM agents](http://arxiv.org/abs/2603.12109)

- AReW: introduces a critique-driven reweighting framework to mitigate information self-locking in LLM agents by reallocating learning signals based on AS and BT performance.
- The framework decomposes agentic behavior into AS and BT components, identifying that their bidirectional coupling under outcome-based RL leads to a low-information trapping regime.
- AReW injects binary directional critiques into the policy-gradient update, effectively breaking the self-locking cycle and improving both information acquisition and belief refinement across multiple interactive reasoning benchmarks.

---

[LABSHIELD: A Multimodal Benchmark for Safety-Critical Reasoning and Planning in Scientific Laboratories](http://arxiv.org/abs/2603.11987)

- LABSHIELD: introduces a multi-view benchmark designed to evaluate the safety-critical reasoning and planning capabilities of MLLMs in autonomous laboratory environments.
- The framework utilizes a dual-track evaluation approach, combining multiple-choice questions and semi-open QA to assess an agent's perception, reasoning, and planning performance across four safety tiers.
- Empirical results reveal a significant decoupling between general-domain MCQ accuracy and safety-critical performance, highlighting the urgent need for robust, grounded reasoning in embodied scientific agents.

---



[Keys on Doormats: Exposed API Credentials on the Web](http://arxiv.org/abs/2603.12498)

- Measurement Pipeline for Credential Exposure: introduces a large-scale dynamic analysis framework to identify, validate, and remediate exposed API credentials across 10 million websites using HTTP Archive Dataset, TruffleHog, Exposure Localization Pipeline, Validation Component, Responsible Disclosure Notification, and Monitoring Crawler.
- The research demonstrates that the majority of credential exposures are introduced dynamically during build and deployment processes, specifically within JavaScript bundles, rather than being present in static source code.
- The study reveals that exposed credentials often persist for months or years, and that coordinated responsible disclosure efforts can significantly accelerate the removal and revocation of these sensitive secrets.

---

[Optimizing Task Completion Time Updates Using POMDPs](http://arxiv.org/abs/2603.12340)

- MOMDP (Mixed Observability Markov Decision Process): introduces a sequential decision-making framework for managing task completion time announcements by balancing prediction accuracy against the costs of frequent updates.
- The framework utilizes POMDP and MOMDP formulations to model the task announcement problem, employing QMDP and SARSOP solvers to generate optimal control policies based on evolving belief states.
- Simulation results demonstrate that the proposed approach significantly reduces unnecessary announcement updates and minimizes replanning-induced delays compared to baseline strategies.

---


[AI Knows What's Wrong But Cannot Fix It: Helicoid Dynamics in Frontier LLMs Under High-Stakes Decisions](http://arxiv.org/abs/2603.11559)

- Helicoid Dynamics framework: introduces a failure regime in LLMs where meta-recognition of errors fails to produce durable behavioral change, characterized by the sequence S1, S2, S3, S4, and S5.
- The study demonstrates that frontier LLMs exhibit a recurring pattern of competent engagement followed by failure, accurate self-diagnosis, and subsequent higher-level recurrence of the same error.
- The research identifies task absorption as a potential mechanism to displace performative failure patterns, while noting that linguistic correction is structurally insufficient for high-stakes decisions.

---

[Examining Users’ Behavioural Intention to Use OpenClaw Through the Cognition–Affect–Conation Framework](http://arxiv.org/abs/2603.11455)

- CAC (Cognition–Affect–Conation) Framework: introduces a structural model to evaluate how cognitive perceptions of OpenClaw influence affective responses and subsequent behavioural intention.
- The framework categorizes user perceptions into enablers (Perceived Personalisation, Perceived Intelligence, Relative Advantage) and inhibitors (Privacy Concern, Algorithmic Opacity, Perceived Risk) to explain adoption dynamics.
- Empirical analysis of 436 users confirms that positive cognitive evaluations strengthen attitude, while negative evaluations increase distrust, both significantly impacting the intention to use autonomous AI agents.

---

[MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning](http://arxiv.org/abs/2603.12266)

- MM-CondChain: introduces a benchmark for visually grounded deep compositional reasoning that utilizes an agentic synthesis pipeline to generate multi-layer control flow instructions.
- The framework employs a Verifiable Programmatic Intermediate Representation (VPIR) to decouple logical construction from natural language rendering, ensuring mechanical verifiability of generated conditions.
- By pairing verified chains with minimally perturbed counterfactuals, the benchmark creates hard negatives that force MLLMs to perform fine-grained verification at every step of a reasoning workflow.

---

[OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams](http://arxiv.org/abs/2603.12265)

- OmniStream: introduces a unified streaming visual backbone that integrates causal spatiotemporal attention and 3D-RoPE to enable efficient, frame-by-frame online processing of video streams.
- The framework utilizes a persistent KV-cache to maintain temporal coherence while avoiding redundant re-computation of past frames during streaming inference.
- OmniStream employs a multi-task pre-training strategy coupling static and temporal representation learning, streaming geometric reconstruction, and vision-language alignment to support diverse downstream tasks including VLM reasoning and robotic manipulation.

---

[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](http://arxiv.org/abs/2603.12238)

- SceneAssistant: introduces an agentic framework for open-vocabulary 3D scene generation that utilizes a VLM-based agent to iteratively refine scenes through a visual-feedback-driven closed loop.
- The framework leverages a comprehensive suite of Action APIs to translate high-level semantic intent into precise 3D spatial manipulations without relying on predefined templates.
- By integrating rendered visual feedback and system-generated collision warnings, the agent autonomously corrects spatial inconsistencies and improves structural coherence during the generation process.

---

[Security Considerations for Artificial Intelligence Agents (Perplexity Response to NIST/CAISI Request for Information 2025-0035)](http://arxiv.org/abs/2603.12230)

- AI Agent Security Framework: introduces a layered defense strategy for AI agents to address vulnerabilities arising from the blurring of code and data boundaries.
- The framework emphasizes defense-in-depth by combining input-level detection, model-level instruction hierarchies, and deterministic system-level enforcement to mitigate risks like indirect prompt injection and cascading failures.
- It highlights that securing modern agentic systems requires moving beyond traditional software security models to address the non-deterministic nature of LLMs and the complexities of multi-agent architectures.

---

[Language Model Teams as Distributed Systems](http://arxiv.org/abs/2603.12229)

- Language Model Teams as Distributed Systems: introduces a principled framework for evaluating LLM teams by mapping their collective behaviors to distributed computing concepts, including LLM agents, Orchestrator, Shared repository, Pessimistic lock, Task list, Communication channels, and Scheduler.
- The paper demonstrates that LLM team performance is constrained by task parallelizability and architectural tradeoffs, mirroring classic distributed systems challenges like consistency conflicts, communication overhead, and straggler delays.
- Empirical results show that decentralized coordination often incurs higher overhead and lower efficiency compared to centralized task assignment, highlighting the necessity of formal design principles for multi-agent LLM systems.

---

[IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](http://arxiv.org/abs/2603.12201)

- IndexCache: introduces a method to accelerate sparse attention in LLMs by partitioning layers into Full (F) layers that compute indices and Shared (S) layers that reuse them, utilizing Lightning Indexer, Sparse Attention, and Multi-layer Distillation Loss.
- The framework employs a greedy search algorithm or multi-layer distillation to optimize the layer partitioning pattern, effectively reducing indexer computations by up to 75%.
- IndexCache maintains model performance across long-context and reasoning tasks while delivering significant prefill and decode speedups on production-scale LLMs.

---

[Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections](http://arxiv.org/abs/2603.12180)

- MADQA: introduces a benchmark of 2,250 human-authored questions over 800 heterogeneous PDF documents to evaluate agentic reasoning capabilities in complex document-intensive workflows.
- The framework evaluates agentic systems including BM25 MLLM Agent, Managed RAG Services, HEAVEN, Claude Agent with Semtools, Recursive Language Models, and MDocAgent using a novel accuracy-effort trade-off protocol.
- Results demonstrate that while top agents match human accuracy, they rely on brute-force search and exhibit a significant efficiency gap compared to human strategic planning.

---

[GlyphBanana: Advancing Precise Text Rendering Through Agentic Workflows](http://arxiv.org/abs/2603.12155)

- GlyphBanana: introduces an agentic workflow that integrates auxiliary tools to inject glyph templates into latent space and attention maps for precise text rendering.
- The framework utilizes a Layout Planner to generate typography plans and a Style Refiner to iteratively harmonize rendered text with the background.
- GlyphBanana includes a comprehensive benchmark, GlyphBanana-Bench, to evaluate text rendering across diverse difficulty levels, including rare characters and complex scientific formulas.

---

[Automatic Generation of High-Performance RL Environments](http://arxiv.org/abs/2603.12145)

- Automatic Generation of High-Performance RL Environments: introduces a reusable recipe using a coding agent and hierarchical verification to synthesize high-performance RL environments from reference implementations.
- The framework employs a four-level verification pipeline—property tests, interaction tests, rollout comparison, and cross-backend policy transfer—to ensure semantic equivalence between reference and performance environments.
- This methodology enables the automated, low-cost production of high-throughput RL environments, shifting training bottlenecks from environment simulation to model computation.

---

[O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](http://arxiv.org/abs/2603.12144)

- O3N: introduces a purely visual, end-to-end framework for omnidirectional open-vocabulary occupancy prediction that leverages a Polar-spiral Mamba (PsM) module for continuous spatial representation and long-range context modeling.
- The framework utilizes an Occupancy Cost Aggregation (OCA) module to unify geometric and semantic supervision, ensuring consistency between reconstructed geometry and semantic structure.
- A gradient-free Natural Modality Alignment (NMA) mechanism is employed to harmonize visual features, voxel embeddings, and text semantics, effectively mitigating modality gaps and overfitting to base classes.

---

[Forecasting and Manipulating the Forecasts of Others](http://arxiv.org/abs/2603.12140)

- Noise-state framework: introduces a method to characterize Nash equilibrium in continuous-time LQG games with endogenous signals by conditioning on primitive Brownian shocks to collapse the infinite belief hierarchy into deterministic two-time kernels.
- The framework utilizes a Volterra fixed-point loop where agents' actions, which reshape opponents' signals, are represented as Volterra controls, allowing for an explicit information wedge that prices the marginal value of shifting opponents' posteriors.
- This approach enables the analysis of strategic belief manipulation in decentralized environments without requiring truncation or large-population limits, providing a closed-form mapping from information primitives to equilibrium outcomes.

---

[HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](http://arxiv.org/abs/2603.12138)

- HATS: introduces a closed-loop trajectory synthesis framework that mitigates semantic ambiguity in GUI agent training by integrating Hardness-Driven Exploration and Alignment-Guided Refinement.
- The framework utilizes a Hardness-Driven Monte Carlo Tree Search (HD-MCTS) to prioritize semantically complex interactions and iteratively refine instruction-execution alignment.
- By converting alignment failures into a hardness reward, HATS generates high-quality, semantically grounded datasets that significantly improve the generalization capabilities of GUI agents.

---

[Increasing intelligence in AI agents can worsen collective outcomes](http://arxiv.org/abs/2603.12129)

- LOTF (Lord of the Flies): introduces a multi-agent framework to study how nature, nurture, culture, and resource scarcity influence collective outcomes in autonomous AI-agent populations.
- The framework utilizes LLM-agents that perform next-token prediction to forecast demand, modulated by reinforcement learning and tribal sensing mechanisms to compete for finite resources.
- Empirical results demonstrate that increased agent sophistication can paradoxically worsen system overload under resource scarcity, while tribal formation provides a mechanism for individual reward at the cost of collective efficiency.

---

[A ROBUST AND EFFICIENT MULTI-AGENT REINFORCEMENT LEARNING FRAMEWORK FOR TRAFFIC SIGNAL CONTROL](http://arxiv.org/abs/2603.12096)

- MARL framework: introduces a robust traffic signal control system that integrates Turning Ratio Randomization, Exponential Phase Duration Adjustment, and Neighbor-Based CTDE to enhance generalization and stability.
- The framework utilizes a centralized critic to guide decentralized actors, enabling effective coordination across traffic networks while maintaining scalability through neighbor-level observations.
- Experimental results demonstrate that the proposed approach significantly reduces average waiting time and improves robustness in unseen traffic scenarios compared to standard RL baselines.

---

[XSKILL: Continual Learning from Experience and Skills in Multimodal Agents](http://arxiv.org/abs/2603.12056)

- XSKILL: introduces a dual-stream framework for multimodal agents that enables training-free continual learning by accumulating and adapting task-level skills and action-level experiences.
- The framework utilizes MLLMexec for tool-use inference and MLLMkb for knowledge management, employing visually-grounded summarization and cross-rollout critique to distill reusable insights from multi-path trajectories.
- During inference, XSKILL decomposes tasks into subtasks and dynamically adapts retrieved knowledge to the current visual context, significantly improving tool-use efficiency and orchestration across diverse benchmarks.

---

[Kinetic SIS opinion-driven models with asymmetric awareness feedback: macroscopic limit and polarization](http://arxiv.org/abs/2603.12041)

- Kinetic SIS framework: introduces a multi-agent model coupling opinion dynamics with epidemic spreading, where individual social behavior both affects and is affected by disease transmission.
- The model utilizes an asymmetric opinion update mechanism where infection events induce caution, while failed transmissions drive individuals toward more extreme, reckless opinions.
- The authors derive a macroscopic kinetic description and a reduced system of ODEs using a fast social-interaction regime, validated through modified Wasserstein distance convergence and numerical simulations.

---

[Slow-Fast Inference: Training-Free Inference Acceleration via Within-Sentence Support Stability](http://arxiv.org/abs/2603.12038)

- SFI (Slow-Fast Inference): introduces a training-free decoding framework that decouples generation into frequent low-cost Fast Steps and occasional dense Slow Steps to accelerate LLMs.
- The framework utilizes a Selector to refresh a Managed Sparse State, comprising Sink Tokens, Selected Tokens, and Recent Tokens, based on dense attention evidence collected during Slow Steps.
- System-level optimizations, including an Asynchronous Pipeline and a Memory-Coalesced Sparse Kernel, ensure that maintenance overhead is hidden and sparse attention remains bandwidth-efficient.

---

[AGMARL-DKS: An Adaptive Graph-Enhanced Multi-Agent Reinforcement Learning for Dynamic Kubernetes Scheduling](http://arxiv.org/abs/2603.12031)

- AGMARL-DKS: introduces a multi-agent reinforcement learning framework for dynamic Kubernetes scheduling that utilizes GNN (contextual state enrichment) and Node Agents (decentralised scheduling actors) to achieve scalable, context-aware pod placement.
- The framework employs an Actor Network (policy function mapping observations) and a Critic Network (stable value estimation) within a Centralised Training with Decentralised Execution paradigm to learn coordinated scheduling policies.
- A Centralised Controller (high-level strategic decision-making) combined with a Lexicographical Filtering Algorithm (multi-objective priority selection) enables the system to adaptively balance fault tolerance, resource utilisation, and cost under varying cluster stress levels.

---

[Cascade: Composing Software-Hardware Attack Gadgets for Adversarial Threat Amplification in Compound AI Systems](http://arxiv.org/abs/2603.12023)

- Cascade: introduces a red-teaming framework that composes algorithmic, software, and hardware attack gadgets to identify vulnerabilities in Compound AI systems.
- The framework maps attacker goals and capabilities to cross-stack attack chains, enabling the discovery of system-level exploits that bypass traditional LLM guardrails.
- Cascade demonstrates that combining system-level vulnerabilities, such as code injection and fault injection, with adversarial LLM attacks significantly amplifies the threat to pipeline integrity and safety.

---

[Sim-to-reality adaptation for Deep Reinforcement Learning applied to an underwater docking application](http://arxiv.org/abs/2603.12020)

- Stonefish-RL framework: introduces a multiprocessing RL approach for autonomous AUV docking that leverages high-fidelity digital twin simulations to bridge the sim-to-real gap.
- The system utilizes a PPO agent trained in a headless, parallelized Stonefish environment to develop robust 6-DoF control policies with emergent behaviors like pitch-based braking and yaw-based alignment.
- Experimental validation in a physical test tank demonstrates that the framework achieves high success rates by maintaining consistent ROS-based interfaces between the simulated and real-world AUV platforms.

---

[Can RL Improve Generalization of LLM Agents? An Empirical Study](http://arxiv.org/abs/2603.12011)

- AgentGym-RL: introduces a systematic empirical study on how reinforcement fine-tuning (RFT) impacts the generalization and transferability of LLM agents across diverse environments.
- The framework utilizes Qwen2.5-Instruct models optimized via GRPO to evaluate agent performance across three axes: intra-environment task difficulty, cross-environment transfer, and sequential multi-environment training.
- Results indicate that while RFT enhances agentic capabilities and exploration efficiency, generalization to unseen environments remains sensitive to shifts in background knowledge, observation spaces, and action interfaces.

---

[Redirecting counter-moving swarms through collision](http://arxiv.org/abs/2603.12002)

- RBA: introduces a modeling framework for analyzing the collision of counter-moving swarms by identifying stable velocity synchronized states as local minima of an effective potential function.
- The framework utilizes a rigid-body approximation to derive analytical scaling laws for swarm redirection, predicting outcomes based on agent numbers, preferred velocities, and interaction parameters.
- Validation is provided through comparisons with particle-based simulations and differential-drive robot experiments, demonstrating the framework's robustness in symmetric and antagonistic collision scenarios.

---

[Learning Visuomotor Policy for Multi-Robot Laser Tag Game](http://arxiv.org/abs/2603.11980)

- Visuomotor Policy for Multi-Robot Laser Tag: introduces an end-to-end decentralized framework that maps onboard monocular images to robot velocity commands using a student policy distilled from a privileged state-based teacher policy.
- The framework utilizes a permutation-invariant feature extractor and a depth-heatmap input representation to achieve robust performance in multi-robot adversarial environments without explicit state estimation or inter-robot communication.
- The student policy employs a CNN-LSTM architecture to aggregate temporal information from camera streams, enabling effective obstacle avoidance and target aiming on resource-constrained robotic platforms.

---

[HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](http://arxiv.org/abs/2603.11975)

- HD-Guard (Hierarchical Dual-Brain Guard for Household Safety): introduces a hierarchical streaming architecture that coordinates a lightweight FastBrain for continuous high-frequency screening with an asynchronous large-scale SlowBrain for deep multimodal reasoning to detect unsafe agent behaviors in household scenarios.
- The framework utilizes HomeSafe-Bench, a hybrid pipeline-constructed benchmark featuring 438 diverse household hazard cases, to evaluate VLM performance across four dimensions including intent onset, hazard category, severity, and reasoning difficulty.
- By decoupling rapid hazard detection from complex semantic reasoning, the system achieves an optimal trade-off between low end-to-end latency and high hazard detection quality, effectively mitigating common VLM failures such as visual omissions and reasoning deficits.

---

[Normative Common Ground Replication (NormCoRe): Replication-by-Translation for Studying Norms in Multi-agent AI](http://arxiv.org/abs/2603.11974)

- NormCoRe (Normative Common Ground Replication): introduces a methodological framework for replicating human subject studies in Multi-agent AI environments by mapping experimental layers to computational components, including foundation model-, AI-based agent-, multi-agent system-, agent-integrated workflow- and memory-agents.
- The framework treats replication as a translation problem rather than an equivalence test, explicitly documenting design choices across cognitive, ontological, interactional, and interventional layers.
- Empirical results demonstrate that while LLMs can converge on human-like fairness principles, their normative outcomes are highly sensitive to foundation model selection and persona language instantiation.

---

[Broadcasting Agents and Adversary: A new variation on Cops and Robbers](http://arxiv.org/abs/2603.11958)

- BROADCAST(G, k): introduces a graph-based game where a team of agents attempts to share information while an adversary dynamically removes edges to prevent communication.
- The framework classifies graph families as either Agents-win or Adversary-win based on structural properties like k-spanning tree symmetry and block structure.
- The paper establishes tight upper and lower bounds for the time required for agents to achieve full information dissemination on various infinite graph families.

---

[PersonaTrace: Synthesizing Realistic Digital Footprints with LLM Agents](http://arxiv.org/abs/2603.11955)

- PersonaTrace: introduces a three-stage agent-based pipeline that synthesizes realistic, multi-bundle digital footprints by leveraging Persona Agent, Event Agent, Artifact Generator Agent, and Critic Agents.
- The framework utilizes a generator-critic loop to ensure that generated artifacts, such as emails and calendar entries, remain consistent with the persona's life narrative and event timeline.
- Empirical evaluations demonstrate that models fine-tuned on PersonaTrace synthetic data achieve superior generalization on real-world downstream tasks compared to existing synthetic datasets.

---

[Kraken*: Architecting Generative, Semantic, and Goal-Oriented Network Management for 6G Wireless Systems](http://arxiv.org/abs/2603.11948)

- Kraken: introduces a multi-plane architecture that integrates semantic communication, generative reasoning, and goal-oriented optimization to transition 6G networks from data-centric to knowledge-centric operation.
- The framework utilizes Infrastructure Plane, Agent Plane, and Knowledge Plane to enable distributed collective intelligence through structured semantic exchange and world-model-based adaptation.
- Kraken incorporates Generative Network Agents, Knowledge Graph, Foundation Models, and Network Digital Twins to ensure scalable, intent-consistent, and reliable network management in 6G environments.

---

[CogSearch: A Cognitive-Aligned Multi-Agent Framework for Proactive Decision Support in E-Commerce Search](http://arxiv.org/abs/2603.11927)

- CogSearch: introduces a multi-agent framework that transforms e-commerce search into a proactive decision support system by mimicking human cognitive workflows through specialized agents.
- The framework utilizes a Planner Agent to decompose complex user intents into executable task graphs, an Executor Agent for multi-source retrieval, a Guider Agent for personalized interaction, and a Decider Agent for final recommendation synthesis.
- By integrating LLMs for reasoning and decision-making, the system reduces user cognitive load and improves conversion rates in complex, decision-heavy search scenarios.

---

[The Network That Thinks: Kraken and the Dawn of Cognitive 6G](http://arxiv.org/abs/2603.11920)

- Kraken (Knowledge-centric, Reasoning, And goal-oriented Knowledge NetworK): introduces a three-plane architecture for 6G networks that integrates semantic communication, generative reasoning, and goal-oriented optimization to enable collective intelligence.
- The architecture utilizes distributed Generative Network Agents (GNAs) that perform perception, memory, planning, and action cycles to replace traditional data-centric network management with knowledge-driven coordination.
- By leveraging semantic abstraction and predictive world models, the framework reduces communication overhead and aligns network resource allocation with application-level objectives rather than intermediate QoS metrics.

---

[QUARE: Multi-Agent Negotiation for Balancing Quality Attributes in Requirements Engineering](http://arxiv.org/abs/2603.11890)

- QUARE: introduces a multi-agent framework that formulates requirements analysis as a structured dialectical negotiation among five quality-specialized agents coordinated by an Orchestrator Agent to resolve cross-quality conflicts.
- The framework utilizes a five-phase pipeline including parallel generation, dialectical negotiation, KAOS goal model integration, RAG-augmented verification, and standardized output generation to produce engineering-ready requirements.
- QUARE employs a Conflict Coordinator to classify conflicts as resource-bound or logical incompatibilities, resolving them through iterative thesis-antithesis-synthesis cycles to ensure semantic preservation and regulatory compliance.

---

[The price of decentralization in managing engineering systems through multi-agent reinforcement learning](http://arxiv.org/abs/2603.11884)

- MADRL: introduces a benchmark framework for evaluating multi-agent reinforcement learning paradigms in inspection and maintenance planning for deteriorating engineering systems.
- The paper systematically compares CTCE, CTDE, and DTDE paradigms across k-out-of-n reliability systems to quantify the optimality loss, termed the price of decentralization, induced by coordination challenges in redundant environments.
- Empirical results demonstrate that while decentralized agents achieve near-optimal performance in series configurations, increasing redundancy amplifies coordination pathologies, leading to significant optimality losses compared to centralized baselines.

---

[ELISA: An Interpretable Hybrid Generative AI Agent for Expression-Grounded Discovery in Single-Cell Genomics](http://arxiv.org/abs/2603.11872)

- ELISA (Embedding-Linked Interactive Single-cell Agent): introduces an interpretable framework that unifies scGPT expression embeddings with BioBERT-based semantic retrieval and LLM-mediated interpretation for interactive single-cell discovery.
- The framework utilizes a Query Classifier to route inputs to either a Gene Marker Scoring Pipeline, a Semantic Matching Pipeline, or a Reciprocal Rank Fusion module for mixed queries.
- Integrated Analytical Modules perform pathway scoring, ligand–receptor interaction prediction, comparative analysis, and proportion estimation, with results interpreted by an LLM to generate grounded biological hypotheses.

---

[Derain-Agent: A Plug-and-Play Agent Framework for Rainy Image Restoration](http://arxiv.org/abs/2603.11866)

- Derain-Agent: introduces a plug-and-play framework that transitions single-image deraining from static inference to dynamic, agent-based restoration using a Shared Backbone, Tool Scheduler, Strength Modulator, Toolbox, and Execution Stage.
- The framework utilizes a lightweight Planning Network to predict instance-specific tool sequences and spatially adaptive strength maps, enabling fine-grained correction of coupled residual degradations.
- Derain-Agent functions as a universal enhancer that boosts the performance of diverse base deraining models across synthetic and real-world benchmarks with minimal computational overhead.

---

[Social, Legal, Ethical, Empathetic and Cultural Norm Operationalisation for AI Agents](http://arxiv.org/abs/2603.11864)

- SLEEC: introduces a systematic five-stage process for operationalising social, legal, ethical, empathetic, and cultural norms into verifiable requirements for AI agents.
- The framework bridges the gap between high-level normative principles and concrete agent behaviour by employing formal methods, including process-algebraic analysis and first-order logic, to ensure compliance.
- The approach utilizes runtime guardrails and iterative feedback loops to maintain normative alignment, demonstrated through an assistive-care robot case study.

---

[OpenClaw PRISM: A Zero-Fork, Defense-in-Depth Runtime Security Layer for Tool-Augmented LLM Agents](http://arxiv.org/abs/2603.11853)

- PRISM: introduces a zero-fork runtime security layer for OpenClaw-based LLM agents that distributes enforcement across ten lifecycle hooks using PRISM Plugin, Scanner Sidecar, Invoke-guard Proxy, Dashboard Sidecar, Monitor Sidecar, and Tamper-evident Audit Plane.
- The architecture integrates a hybrid heuristic-plus-LLM scanning pipeline with session-scoped risk accumulation to mitigate indirect prompt injection, tool abuse, and credential exfiltration.
- The system provides deployable runtime defense by attaching to existing gateways without requiring upstream code forks, enabling operator-managed policy enforcement and verifiable audit trails.

---

[Hybrid Human–Agent Social Dilemmas in Energy Markets](http://arxiv.org/abs/2603.11834)

- DSLM: introduces a decentralized framework for energy load management where autonomous agents use intrinsic reward shaping to facilitate cooperative turn-taking in hybrid human-agent populations.
- The framework utilizes RL-agents that optimize appliance scheduling based on globally observable price signals and an intrinsic reward bonus to overcome social dilemmas.
- The research demonstrates that partial adoption of these agents is entry-resilient, allowing for improved aggregate social outcomes without requiring universal participation.

---

[Large language models for optical network O&M: Agent-embedded workflow for automation](http://arxiv.org/abs/2603.11828)

- Multi-Agent collaborative O&M architecture: introduces a framework that embeds LLM-based agents into existing optical network workflows to automate channel management, performance optimization, and fault management.
- The architecture utilizes a Supervisor Agent to orchestrate specialized sub-Agents, which leverage prompt engineering, RAG, and tool invocation to execute complex O&M tasks.
- The framework integrates digital twin technology and existing NMS/SDN controllers to ensure that LLM-driven decisions are pre-validated and compatible with operational safety requirements.

---

[Automating Skill Acquisition through Large-Scale Mining of Open-Source Agentic Repositories: A Framework for Multi-Agent Procedural Knowledge Extraction](http://arxiv.org/abs/2603.11808)

- Automated Skill Acquisition Framework: introduces a systematic pipeline for mining open-source repositories to extract reusable procedural knowledge as modular agent skills.
- The framework utilizes Repository Structural Analysis, Semantic Skill Identification, and Standardized Translation to convert monolithic codebases into SKILL.md artifacts.
- It incorporates a multi-stage Verification Pipeline and SkillNet consolidation to ensure security, governance, and efficient skill composition for LLMs.

---

[A Semi-Decentralized Approach to Multiagent Control](http://arxiv.org/abs/2603.11802)

- SDec-POMDP: introduces a foundational framework for multiagent control in environments with probabilistic communication constraints by unifying decentralized and centralized decision-making models.
- The framework utilizes selector functions and a blackboard memory system to manage information propagation between agents based on sojourn communication times.
- RS-SDA* provides an exact planning algorithm for the SDec-POMDP by employing a small-step search tree with mixed component policies and admissible heuristics.

---

[DocSage: An Information Structuring Agent for Multi-Doc Multi-Entity Question Answering](http://arxiv.org/abs/2603.11798)

- DocSage: introduces an agentic framework that transforms unstructured multi-document data into query-specific relational structures to enable precise multi-hop reasoning.
- The framework utilizes ASK for dynamic schema discovery, CLEAR for logic-aware information extraction, and a SQL-based reasoning module to mitigate LLM attention diffusion.
- By offloading complex relational joins to a deterministic database engine, DocSage achieves significant accuracy improvements over standard RAG and long-context LLMs in multi-entity QA tasks.

---

[Disentangled Representation Learning Through Unsupervised Symmetry Group Discovery](http://arxiv.org/abs/2603.11790)

- GMA-VAE: introduces a symmetry-based disentangled representation learning approach that autonomously discovers the symmetry group structure of an environment without requiring prior knowledge.
- The framework utilizes an A-VAE to learn an entangled representation and action matrices, followed by a clustering algorithm to recover the group decomposition, and finally a GMA-VAE to enforce disentanglement via structured masking.
- The method provides theoretical guarantees for identifiability and disentanglement, demonstrating superior performance in long-term prediction and generalization compared to existing LSBD approaches.

---

[From Debate to Deliberation: Structured Collective Reasoning with Typed Epistemic Acts](http://arxiv.org/abs/2603.11781)

- DCI (Deliberative Collective Intelligence): introduces a multi-agent framework that treats collective reasoning as a structured, phased deliberation process using Delegate Model, Session Model, Interaction Grammar, Shared Workspace, DCI-CF, and Decision Packet.
- The framework utilizes four specialized LLM-delegate archetypes—Framer, Explorer, Challenger, and Integrator—to exchange typed epistemic acts within a phased session to ensure accountable, auditable outcomes.
- DCI-CF guarantees procedural convergence by transforming divergent perspectives into a structured decision packet, which includes the selected option, residual objections, a minority report, and explicit reopen conditions.

---

[Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework](http://arxiv.org/abs/2603.11768)

- SSGM (Stability and Safety Governed Memory): introduces a conceptual governance architecture that decouples memory evolution from execution to mitigate risks like semantic drift and memory poisoning in LLMs.
- The framework utilizes a Governance Middleware containing a Write Validation Gate for logical consistency and a Read Filtering Gate for access control and temporal relevance.
- SSGM employs a Dual Memory Substrate, pairing a Mutable Active Graph for reasoning with an Immutable Episodic Log to enable asynchronous reconciliation and drift bounding.

---

[Exploiting Expertise of Non-Expert and Diverse Agents in Social Bandit Learning: A Free Energy Approach](http://arxiv.org/abs/2603.11757)

- SBL-FE: introduces a social learning framework for stochastic bandit problems that leverages a free energy minimization principle to evaluate and exploit the expertise of diverse agents without requiring shared rewards or private information.
- The framework utilizes a Social Agent (SA) that integrates its own Thompson Sampling-based experience with the observed behavior of Individual Agents (IAs) to optimize decision-making under uncertainty.
- By mapping agent policies to a free energy space, the approach effectively balances utility maximization with information-processing costs, enabling robust performance in heterogeneous societies even when irrelevant or suboptimal agents are present.

---

[Online Learning of Strategic Defense against Ecological Adversaries under Partial Observability with Semi-Bandit Feedback](http://arxiv.org/abs/2603.11726)

- HERDS: introduces a game-theoretic online learning framework for strategic defense against ecological adversaries with unknown behavioral models and partial observability.
- The framework utilizes dynamic budget partitioning and adaptive payoff estimation to minimize cumulative regret in combinatorial action spaces with confounded semi-bandit feedback.
- Experimental validation using an Agent-Based Model demonstrates that HERDS achieves significant regret reduction and faster convergence compared to baseline security game algorithms.

---

[When OpenClaw Meets Hospital: Toward an Agentic Operating System for Dynamic Clinical Workflows](http://arxiv.org/abs/2603.11721)

- AOS-H: introduces an infrastructure-centric architecture for deploying LLM agents in clinical environments by replacing permissive runtimes with OS-enforced isolation and document-mediated coordination.
- The framework utilizes a page-indexed memory architecture that enables longitudinal context retrieval through manifest-guided navigation, eliminating the need for vector embeddings.
- AOS-H supports ad-hoc clinical task composition by allowing agents to dynamically invoke pre-audited skills, ensuring safety and auditability through kernel-level resource constraints.

---

[Scaling Laws for Educational AI Agents](http://arxiv.org/abs/2603.11709)

- EduClaw: introduces a profile-driven multi-agent platform that operationalizes the Agent Scaling Law by utilizing AgentProfile, Skill Library, Open Claw Runtime, Interface Layer, and Management Layer.
- The framework enables systematic capability growth of educational agents through structured JSON-based specifications rather than relying solely on model size.
- The system supports hierarchical task decomposition and multi-agent orchestration, allowing agents to scale their pedagogical effectiveness through composition and specialized skill modules.

---

[STAIRS-Former: Spatio-Temporal Attention with Interleaved Recursive Structure Transformer for Offline Multi-Task Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2603.11691)

- STAIRS-Former: introduces a transformer architecture for offline multi-task MARL that utilizes a Spatial-Former, GRU, Qatten mixing network, Token-Dropout Mechanism, and Temporal-Focus Layer to improve relational reasoning and historical dependency modeling.
- The framework employs a recursive spatial module for deep relational reasoning and a dual-timescale temporal module to capture both short-term and long-term dependencies under partial observability.
- A token-dropout mechanism is integrated to enhance robustness and generalization across varying agent populations in unseen multi-task scenarios.

---

[LLMs can construct powerful representations and streamline sample-efficient supervised learning](http://arxiv.org/abs/2603.11679)

- Rubric Representation Learning: introduces an agentic pipeline where LLMs synthesize structured rubrics to transform heterogeneous, weakly-structured inputs into standardized representations for sample-efficient supervised learning.
- The framework utilizes LLM-agents to generate either global rubrics (shared templates) or local rubrics (task-conditioned summaries), which are then processed by parsers or tabularization scripts to create efficient inputs for downstream models.
- By automating the design of input representations, the approach significantly outperforms traditional count-feature models and naive text-serialization baselines across 15 clinical prediction tasks in the EHRSHOT benchmark.

---

[From Control to Foresight: Simulation as a New Paradigm for Human–Agent Collaboration](http://arxiv.org/abs/2603.11677)

- Simulation-in-the-loop: introduces a conceptual framework that shifts human-agent interaction from reactive supervision to proactive exploration by externalizing future trajectories for human preview.
- The framework utilizes an LLM agent to generate multiple potential action paths, which are then annotated with simulated impacts to facilitate informed decision-making.
- By visualizing downstream consequences, the approach enables users to discover latent constraints and serendipitous alternatives that are otherwise hidden in standard pointwise interaction paradigms.

---

[Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](http://arxiv.org/abs/2603.11653)

- Sequential Fine-Tuning (Seq. FT): introduces a robust approach for continual reinforcement learning in large Vision-Language-Action models by leveraging the synergy between a Pre-trained VLA Model, Parameter-Efficient Adaptation (LoRA), and On-Policy Reinforcement Learning (GRPO).
- The framework demonstrates that simple sequential fine-tuning, when combined with these components, effectively mitigates catastrophic forgetting while maintaining high plasticity and strong zero-shot generalization capabilities.
- Empirical results across multiple benchmarks show that this minimal-assumption strategy often outperforms more complex continual learning methods and can achieve performance on par with multi-task oracles through prolonged training.

---

[QChunker: Learning Question-Aware Text Chunking for Domain RAG via Multi-Agent Debate](http://arxiv.org/abs/2603.11650)

- QChunker: introduces a multi-agent debate framework that restructures RAG from retrieval-augmentation to understanding-retrieval-augmentation by modeling text chunking as a composite task of segmentation and knowledge completion.
- The framework utilizes a Question Outline Generator, Text Segmenter, Integrity Reviewer, and Knowledge Completer to ensure logical coherence and information integrity in text chunks.
- It introduces ChunkScore, a direct evaluation metric that balances logical independence and semantic dispersion to optimize chunking quality without relying on downstream QA tasks.

---

[An observer-based approach to the sorites paradox and the logic derived from that](http://arxiv.org/abs/2603.11624)

- FOS (Fluxing-Object Semantics): introduces a formal framework for modeling vague predicates by representing objects as partial functions of time and observers, which naturally gives rise to truth-value gaps.
- The framework resolves the sorites paradox by identifying vague transitions as occurring only during "watching gaps," which are periods of interrupted observation by an agent.
- The resulting semantics is proven to be equivalent to strong Kleene three-valued logic, providing a mathematically consistent way to handle undetermined truth values in vague assertions.

---

[Taming OpenClaw: Security Analysis and Mitigation of Autonomous LLM Agent Threats](http://arxiv.org/abs/2603.11619)

- OpenClaw: introduces a five-layer lifecycle-oriented security framework to analyze and mitigate systemic threats in autonomous LLM agents across initialization, input, inference, decision, and execution stages.
- The paper identifies that autonomous LLM agents face compounding risks such as indirect prompt injection, skill supply chain contamination, memory poisoning, and intent drift due to their persistent memory and high-privilege execution capabilities.
- The authors propose a defense-in-depth architecture that mandates complete lifecycle mediation, least privilege with provenance tracking, and cross-stage security coherence to protect against multi-stage adversarial attacks.

---

[LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](http://arxiv.org/abs/2603.11605)

- LaMoGen: introduces a framework that leverages LLMs to compose motion sequences through symbolic reasoning using the LabanLite representation.
- The framework utilizes a two-stage pipeline consisting of LLM-driven conceptual motion planning and a Kinematic Detail Augmentor for fine-grained motion synthesis.
- LaMoGen establishes a Labanotation-based benchmark to evaluate text-motion alignment across symbolic, temporal, and harmony dimensions.

---

[See, Symbolize, Act: Grounding VLMs with Spatial Representations for Better Gameplay](http://arxiv.org/abs/2603.11601)

- VLM-based gameplay framework: investigates how providing VLMs with both visual frames and symbolic representations improves performance in interactive environments by comparing four distinct pipelines.
- The study evaluates VLM performance across Atari, VizDoom, and AI2-THOR, demonstrating that symbolic grounding is beneficial only when symbol extraction is reliable.
- Results indicate that perception quality is a central bottleneck, as noisy symbolic information can degrade decision-making, while visual frames remain essential for providing necessary context.

---

[Modeling Sequential Design Actions as Designer Externalization on an Infinite Canvas](http://arxiv.org/abs/2603.11569)

- Agentorganizer: introduces a behavioral model for infinite canvases that demonstrates how proactive AI agents redistribute cognitive effort from spatial management to content curation and relational structuring.
- The research utilizes a Sequential Design Action Taxonomy to map interaction logs into semantic categories, revealing how AI roles evolve from divergent idea amplifiers to convergent curators.
- Empirical findings indicate that AI-assisted design on infinite canvases follows a non-monotonic trajectory where designers increase selectivity in requests while simultaneously increasing trust in agent-generated content.

---

[How Intelligence Emerges: A Minimal Theory of Dynamic Adaptive Coordination](http://arxiv.org/abs/2603.11560)

- Recursive Coordination Architecture: introduces a dynamical theory where intelligence emerges from the recursive coupling of agents, incentives, and a persistent environment rather than centralized optimization.
- The framework models coordination as a structural property of a closed-loop system where agents respond to localized incentive fields derived from accumulated environmental memory.
- The theory establishes that non-trivial coupling between adaptive update operators and memory-dependent incentive fields is a necessary structural condition for emergent intelligence.

---

[RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks](http://arxiv.org/abs/2603.11558)

- RoboClaw: introduces a unified agentic framework that integrates data collection, policy learning, and long-horizon task execution under a single VLM-driven controller.
- The framework utilizes EAP to couple forward manipulation behaviors with inverse recovery actions, enabling autonomous data collection and self-resetting loops.
- By employing a VLM as a meta-controller with structured memory and MCP tools, the system dynamically orchestrates skills and monitors task progress to improve robustness in long-horizon robotic manipulation.

---

[MANSION: Multi-floor lANguage-to-3D Scene generatIOn for loNg-horizon tasks](http://arxiv.org/abs/2603.11554)

- MANSION: introduces a hierarchical multi-agent framework that transforms natural-language instructions into building-scale, multi-floor 3D environments using a hybrid MLLM-geometry pipeline, with Chief Designer, Floor Designer, Geometric Solver, and Scene Instantiator.
- The framework incorporates a Task-Semantic Scene Editing Agent, which includes a ReAct Controller, Tool Invoker, Static Semantic State, and On-Demand Physics Engine to ensure task executability through a "check-and-provision" workflow.
- The MansionWorld dataset provides over 1,000 diverse, interactive multi-floor buildings, extending AI2-THOR with cross-floor navigation assets and skill APIs to support long-horizon embodied AI research.

---

[One Supervisor, Many Modalities: Adaptive Tool Orchestration for Autonomous Queries](http://arxiv.org/abs/2603.11545)

- Centralized Orchestration Framework: introduces an agentic AI architecture that replaces rigid decision trees with a dynamic Central Supervisor to coordinate specialized tools and models across diverse modalities.
- The framework utilizes a LangGraph-based state management system to enable parallel execution, local failure recovery, and context-aware routing for complex multimodal queries.
- By integrating the Couplet Framework for efficient perceptual processing and RouteLLM for learned text routing, the system achieves significant reductions in latency, cost, and conversational rework compared to hierarchical baselines.

---

[Multi-Agent Collaboration for Automated Design Exploration on High Performance Computing](http://arxiv.org/abs/2603.11515)

- MADA: introduces a multi-agent framework that automates scientific design exploration by coordinating specialized agents for simulation, mesh generation, and inverse design.
- The framework utilizes LLMs as reasoning engines to manage complex workflows, integrating domain-specific tools via the Model Context Protocol to enable iterative design refinement.
- MADA demonstrates effective design optimization for Richtmyer–Meshkov Instability suppression by combining high-fidelity HPC simulations and surrogate-based exploration.

---

[Grammar of the Wave: Towards Explainable Multivariate Time Series Event Detection via Neuro-Symbolic VLM Agents](http://arxiv.org/abs/2603.11479)

- SELA (Time Series Event Logic Agents): introduces a neuro-symbolic VLM agent framework for zero-shot Knowledge-Guided Time Series Event Detection (K-TSED) that utilizes Event Logic Tree (ELT) to bridge linguistic descriptions and physical time series data.
- The framework includes Logic Analyst- and Signal Inspector-agents that collaborate via a Central Logic Engine to perform event detection and provide explainable, hierarchical tree-based evidence.
- By employing ELT as a structured intermediate representation, the system mitigates VLM hallucination and improves localization precision in complex, multi-stage time series events.

---

[Persistence, patience and costly information acquisition](http://arxiv.org/abs/2603.11453)

- Optimal Bayesian Learning Model: introduces a tractable framework for a forward-looking agent to optimally balance marginal costs and informativeness when acquiring signals about a Gaussian AR(1) state.
- The framework utilizes a Gaussian-quadratic setup with linear precision costs to derive closed-form solutions for optimal sequential learning strategies.
- The analysis demonstrates that while higher persistence can tighten or loosen steady-state beliefs, it consistently reduces welfare due to endogenously higher information costs, whereas increased patience improves welfare by leveraging information from past selves.

---

[Verified Multi-Agent Orchestration: A Plan-Execute-Verify-Replan Framework for Complex Query Resolution](http://arxiv.org/abs/2603.11445)

- VMAO: introduces a multi-agent framework that coordinates specialized LLM-based agents through an iterative Plan-Execute-Verify-Replan loop to resolve complex queries.
- The framework utilizes a QueryPlanner to decompose tasks into a directed acyclic graph (DAG) and a ResultVerifier to ensure output completeness via LLM-based evaluation.
- VMAO incorporates configurable stop conditions and hierarchical synthesis to balance answer quality against resource usage and context limitations.

---

[EducaSim: Interactive Simulacra for CS1 Instructional Practice](http://arxiv.org/abs/2603.11444)

- EducaSim: introduces a framework for teacher training that utilizes generative agents to simulate small-group classroom interactions for pre-service teachers.
- The architecture integrates Student Personas, a Node-based Memory System, a Decision-Making Framework, a Speech Oracle, a Runnable Python IDE, Whisper Speech-to-Text, and an LLM-based Feedback Generator to provide a pedagogically rich role-play environment.
- The system enables scalable, low-cost experiential learning by allowing teachers to practice instruction, receive automated feedback, and engage in self-reflection using course-specific materials.

---

[Adversarial Reinforcement Learning for Detecting False Data Injection Attacks in Vehicular Routing](http://arxiv.org/abs/2603.11433)

- PSRO (Policy Space Response Oracles): introduces a game-theoretic framework for detecting false data injection attacks in vehicular routing by modeling the interaction between an attacker and a defender as a zero-sum stochastic game.
- The framework utilizes an Attack Oracle (DRL agent generating attack perturbations) and a Defense Oracle (DRL agent deciding alert status) to compute a Nash Equilibrium (optimal game-theoretic strategy pair) using the Double Oracle (iterative algorithm for strategy sets) method.
- By employing PPO (Proximal Policy Optimization) as the underlying learning mechanism, the approach effectively identifies robust detection policies that outperform traditional anomaly detection baselines in simulated transportation networks.

---

[ShotVerse: Advancing Cinematic Camera Control for Text-Driven Multi-Shot Video Creation](http://arxiv.org/abs/2603.11421)

- ShotVerse: introduces a "Plan-then-Control" framework that decouples multi-shot video generation into a VLM-based Planner for explicit trajectory plotting and a Controller for high-fidelity video synthesis.
- The framework utilizes a novel multi-shot camera calibration pipeline to align disjoint trajectories into a unified global coordinate system, facilitating the creation of the ShotVerse-Bench dataset.
- ShotVerse incorporates a 4D Rotary Positional Embedding strategy to explicitly model hierarchical shot boundaries, ensuring intra-shot consistency and precise camera control in multi-shot video generation.

---

[Entropy Guided Diversification and Preference Elicitation in Agentic Recommendation Systems](http://arxiv.org/abs/2603.11399)

- IDSS: introduces a conversational recommendation framework that treats uncertainty as a first-class signal to guide preference elicitation, ranking, and presentation.
- The system utilizes entropy-based metrics to select informative follow-up questions and to structure recommendation outputs into grids that facilitate user exploration.
- IDSS integrates LLM-based semantic parsing with information-theoretic ranking and diversification to balance relevance and diversity under ambiguous user intent.

---

[ARROW: Augmented Replay for RObust World models](http://arxiv.org/abs/2603.11395)

- ARROW: introduces a model-based continual reinforcement learning algorithm that extends DreamerV3 with an augmented replay buffer to mitigate catastrophic forgetting.
- The framework utilizes a dual-buffer system, comprising a short-term FIFO buffer and a long-term global distribution-matching buffer, to balance recent experience with long-term knowledge.
- By training the World Model on augmented replay data, the agent generates imagined trajectories to train the actor-critic controller, enabling efficient continual learning without explicit task identifiers.

---

[Agentic AI for Embodied-enhanced Beam Prediction in Low-Altitude Economy Networks](http://arxiv.org/abs/2603.11392)

- Agentic AI architecture: introduces a multi-agent collaborative reasoning framework for UAV-to-ground mmWave communications that decomposes beam prediction into task analysis, solution planning, and completeness assessment.
- The framework utilizes a hybrid beam prediction model system that integrates Mamba-based temporal modelling, convolutional visual encoding, and cross-attention-based multimodal fusion to process numeric and visual UAV data.
- The multi-agent system employs a ReAct-based reasoning paradigm to dynamically adjust data-flow strategies and model configurations, significantly improving beam prediction robustness in highly mobile environments.

---

[SliceFed: Federated Constrained Multi-Agent DRL for Dynamic Spectrum Slicing in 6G](http://arxiv.org/abs/2603.11390)

- SliceFed: introduces a federated multi-agent deep reinforcement learning framework for dynamic radio access network slicing that utilizes gNB agents, a local CMDP, a Lagrangian-based PPO, a federated learning layer, a central aggregator, a replay buffer, and actor-critic networks.
- The framework optimizes spectrum usage by formulating slicing as a CMDP, enabling autonomous gNB agents to satisfy inter-cell interference, latency, and resource constraints through Lagrangian-based primal-dual updates.
- Collaborative training is achieved via a federated learning layer that aggregates local model updates at a central server, ensuring data privacy while maintaining global policy coherence across dense 6G network deployments.

---

[LLM BiasScope: A Real-Time Bias Analysis Platform for Comparative LLM Evaluation](http://arxiv.org/abs/2603.12522)

- LLM BiasScope: introduces a web-based platform for real-time, side-by-side comparison of LLMs using React/Next.js Frontend, Vercel AI SDK, Hugging Face Inference Endpoints, Bias-Detector Model, Bias-Type Classifier, Vercel AI Gateway, and Recharts.
- The system employs a two-stage pipeline where the Bias-Detector Model identifies biased sentences, followed by the Bias-Type Classifier to categorize social bias patterns.
- This platform facilitates comparative LLM evaluation by streaming responses in real-time and providing interactive visualizations of bias metrics across multiple providers.

---

[Gaussian and bootstrap approximations for functional principal component regression](http://arxiv.org/abs/2603.12518)

- FPCR (Functional Principal Component Regression): introduces a methodology for statistical inference in functional regression by establishing valid Gaussian and bootstrap approximations through suitable operator scaling.
- The framework overcomes the degeneracy of limiting distributions in functional regression by applying a pseudo-square-root operator scaling to the FPCR estimator.
- The paper demonstrates that the proposed L2 and supremum norm statistics provide powerful hypothesis testing for the significance of the slope function in functional regression models.

---

[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](http://arxiv.org/abs/2603.12514)

- VDETR (Vertex Detection Transformer): introduces a label-efficient framework for 3D medical object detection by combining self-supervised 3D U-Net encoder pre-training with semi-supervised VDETR decoder fine-tuning.
- The framework utilizes 3DV-RPE to compute geometric relationships between voxels and bounding box vertices, enabling precise localization of irregularly-shaped anatomical injuries.
- A teacher-student consistency regularization mechanism leverages large volumes of unlabeled CT data to stabilize training and improve generalization when labeled annotations are severely limited.

---

[How Fair is Software Fairness Testing?](http://arxiv.org/abs/2603.12511)

- Software Fairness Testing Framework: introduces a critical examination of current fairness evaluation practices, arguing that they are culturally situated and often reinforce Western epistemological standards.
- The paper highlights how existing fairness metrics, datasets, and oracles frequently exclude non-Western knowledge systems and perpetuate data colonialism.
- It proposes shifting toward participatory, co-created evaluation practices that respect cultural plurality and address the environmental and social costs of LLMs.

---

[RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](http://arxiv.org/abs/2603.12493)

- Realistic RAW-domain Degradation Models: introduces a calibration-based framework for modeling device-specific blur kernels and sensor noise to synthesize realistic training data for smartphone super-resolution.
- The framework utilizes a display prototype to establish accurate geometric and radiometric alignment, enabling the creation of high-quality paired HR-LR data for training.
- By modeling degradations directly in the RAW domain, the approach effectively reduces the domain gap between synthetic and real-world smartphone images, outperforming generic degradation models.

---

[Xe gas bubble re-solution in U-10Mo nuclear fuel](http://arxiv.org/abs/2603.12491)

- RustBCA / LAMMPS / TTM / SPHEREINCUBOID: introduces a physics-based computational framework to quantify Xe gas bubble re-solution rates in U-10Mo nuclear fuel by integrating binary collision approximation and molecular dynamics simulations.
- The study demonstrates that homogeneous re-solution driven by nuclear stopping is the primary mechanism in U-10Mo, while heterogeneous re-solution via electronic stopping is improbable due to high thermal conductivity.
- The research establishes an analytical model for re-solution rates as a function of bubble size, pressure, and fission rate, providing a robust foundation for higher-length-scale fuel performance modeling.

---

[A GENERALIZED CLUSTER STRUCTURE ON GLn VIA BIRATIONAL POISSON MAPS](http://arxiv.org/abs/2603.12486)

- GCn: introduces a regular complete generalized cluster structure on GLn compatible with a specific Poisson homogeneous bracket defined by Cremmer–Gervais solutions to the classical Yang–Baxter equation.
- The construction utilizes a birational Poisson map Ψ to connect the target Poisson bracket with known structures on GLn−1 and a space of rational functions, enabling the pullback of initial seeds.
- The paper provides a comprehensive analysis of non-aperiodic Belavin–Drinfeld data, establishing the existence of generalized cluster structures through the amalgamation of existing cluster algebraic frameworks.

---

[CalliMaster: Mastering Page-level Chinese Calligraphy via Layout-guided Spatial Planning](http://arxiv.org/abs/2603.12482)

- CalliMaster: introduces a unified framework for controllable Chinese calligraphy generation and editing by decoupling spatial planning from content synthesis using a coarse-to-fine pipeline.
- The framework utilizes a Multimodal Diffusion Transformer with independent noise schedules to enforce a causal dependency where character bounding boxes guide the subsequent rendering of high-fidelity brushwork.
- CalliMaster enables interactive semantic re-planning, artifact restoration, and forensic identification through its layout-aware generative architecture and the proposed Diffusion Reconstruction Score.

---

[Seeing the Trees for the Forest: Leveraging Tree-Shaped Substructures in Property Graphs](http://arxiv.org/abs/2603.12476)

- Structural Indexing for Property Graphs: introduces a methodology to optimize graph query performance by treating tree-shaped substructures as first-class citizens using PrePost Indexing and Dewey Encoding.
- The approach leverages structural indexes originally designed for XML databases to accelerate path-based queries within relational-backed GDBMS environments.
- Experimental results demonstrate that integrating these structural indexes into relational engines yields significant speedups for ancestor-descendant and leaf-node queries compared to standard graph traversal methods.

---

[CLARE: Classification-based Regression for Electron Temperature Prediction](http://arxiv.org/abs/2603.12470)

- CLARE: introduces a classification-based regression framework that discretizes continuous electron temperature into 150 discrete bins to improve prediction accuracy and provide uncertainty quantification.
- The model utilizes a 6-layer feedforward network trained on AKEBONO satellite measurements and NASA OMNI solar/geomagnetic indices to predict electron temperature in the Earth's plasmasphere.
- By reframing regression as a classification task, CLARE achieves higher accuracy and built-in confidence estimation compared to traditional continuous regression models.

---

[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](http://arxiv.org/abs/2603.12468)

- SFDA-DeP: introduces an iterative machine unlearning-inspired framework to mitigate prediction bias in WSOL models during source-free domain adaptation.
- The method selectively penalizes uncertain samples from dominant classes while retaining stable samples to reshape decision boundaries and improve classification balance.
- A jointly optimized pixel-level classifier leverages CAMs to restore discriminative localization features, ensuring robust performance across diverse histopathology datasets.

---

[Predictive and adaptive maps for long-term visual navigation in changing environments](http://arxiv.org/abs/2603.12460)

- Map Management Strategies for Teach-and-Repeat Navigation: introduces a comparative study of map adaptation techniques for mobile robots to maintain reliable visual navigation in environments subject to long-term appearance changes.
- The framework utilizes SURF and BRIEF for feature extraction, employing a histogram voting scheme for robust image registration and map refinement.
- Experimental results demonstrate that strategies incorporating FreMEn (Frequency Map Enhancement) to model temporal feature visibility outperform static and simple adaptive map management approaches.

---

[CSE-UOI at SemEval-2026 Task 6: A Two-Stage Heterogeneous Ensemble with Deliberative Complexity Gating for Political Evasion Detection](http://arxiv.org/abs/2603.12453)

- CSE-UOI: introduces a two-stage heterogeneous ensemble for political evasion detection that combines Grok-4-1-fast-reasoning and Gemini-3-flash-preview with a novel post-hoc Deliberative Complexity Gating (DCG) mechanism.
- The framework utilizes an evasion-first classification strategy followed by asymmetric weighted voting to aggregate predictions from multiple LLM-components.
- The DCG mechanism adaptively corrects uncertain predictions by leveraging Gemini response length and Grok self-consistency as behavioral signals for ambiguity detection.

---

[RadEar: A Self-Supervised RF Backscatter System for Voice Eavesdropping and Separation](http://arxiv.org/abs/2603.12446)

- RadEar: introduces a batteryless RF backscatter system that enables covert, through-wall voice eavesdropping by utilizing a dual-resonator tag for frequency modulation and an RF reader equipped with self-supervised models for signal recovery.
- The system architecture employs a piezoelectric sensor, Voltage-Sensing Resonator (VSR), Parametric Resonator (PR), and dipole antenna to achieve continuous voice streaming while mitigating self-interference through spectral separation.
- The RF reader utilizes a self-supervised learning framework, incorporating voice separation and denoising models trained via a remixing-based objective to recover high-fidelity speech from weak, unlabeled RF signals.

---

[Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](http://arxiv.org/abs/2603.12445)

- CNN architectures: introduces an empirical evaluation of four common CNN models (ResNet50, DenseNet121, Inception V3, and VGG16) to identify dataset bias in cancer pathology image analysis.
- The study demonstrates that these models often achieve high classification accuracy on non-informative, cropped background image segments, suggesting reliance on superficial artifacts rather than clinical features.
- The findings indicate that standard machine learning evaluation practices may lead to overoptimistic performance metrics in biomedical applications due to widespread, modality-independent dataset bias.

---

[Compensation of Input/Output Delays for Retarded Systems by Sequential Predictors: A Lyapunov-Halanay Method](http://arxiv.org/abs/2603.12439)

- Sequential Predictors based control framework: introduces a Lyapunov-Halanay method to achieve global asymptotic stabilization for nonlinear retarded systems with large input/output delays using sequential predictors.
- The framework constructs finite cascades of differential equations to estimate system states forward in time, effectively compensating for delays without requiring infinite-dimensional integral terms.
- Stability analysis is conducted using quadratic Lyapunov functions for error dynamics, providing a robust alternative to traditional Lyapunov-Krasovskii functional methods for time-delay systems.

---

[DiscoRD: An Experimental Methodology for Quickly Discovering the Reliable Read Disturbance Threshold of Real DRAM Chips](http://arxiv.org/abs/2603.12435)

- DiscoRD: introduces a reliable and rapid methodology for characterizing the read disturbance threshold (RDT) of DRAM chips to enable quantitative reasoning about system security and performance.
- The framework utilizes an empirical data-driven model to evaluate the probability of uncorrectable errors when combining error tolerance, memory scrubbing, and configurable mitigation mechanisms.
- DiscoRD demonstrates that spatial and temporal variation-aware mitigation strategies significantly improve system performance compared to one-size-fits-all approaches while maintaining robust protection against read disturbance bitflips.

---

[HOPF ALGEBRAS OVER CHEVALLEY GROUPS](http://arxiv.org/abs/2603.12428)

- Hopf algebras over Chevalley groups framework: introduces a classification of finite-dimensional pointed Hopf algebras over finite simple Chevalley groups by analyzing Nichols algebras of Yetter-Drinfeld modules.
- The paper establishes that every finite-dimensional pointed Hopf algebra over a finite simple Chevalley group, with specific exceptions, is isomorphic to the corresponding group algebra.
- The authors introduce new criteria of type Ω and type C to determine the infinite dimensionality of Nichols algebras associated with semisimple conjugacy classes in Chevalley and Steinberg groups.

---

[LLMs for Human Mobility: Opportunities, Challenges, and Future Directions](http://arxiv.org/abs/2603.12420)

- LLMs for Human Mobility: provides a comprehensive synthesis of how LLMs are adopted across five mobility tasks, utilizing Prompt Engineering, Agentic LLMs, and Fine-Tuning to bridge raw numerical data with semantic behavioral context.
- The paper classifies mobility research into travel itinerary planning, trajectory generation, mobility simulation, mobility prediction, and mobility semantics, highlighting the integration of External Knowledge, Memory Modules, and Tool APIs to enhance reasoning and constraint satisfaction.
- It identifies critical challenges such as feasibility with real-world constraints, dynamic interaction effects, and the semantic-spatiotemporal representation gap, proposing hybrid architectures that combine LLM-based semantic inference with traditional solvers and simulators.

---

[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](http://arxiv.org/abs/2603.12409)

- ABRA: introduces a modular framework that disentangles domain and class knowledge to enable the transfer of fine-tuned detection capabilities across domains without requiring target-domain training data.
- The method utilizes Objectification to derive class-agnostic domain experts and SVFT to learn lightweight class-specific residuals, which are then analytically transported to target domains via closed-form weight-space rotations.
- By formulating adaptation as a geometric transport problem, ABRA effectively teleports class-level specialization to target domains, consistently outperforming existing baselines in zero-shot and few-shot transfer scenarios.

---

[Beyond Motion Imitation: Is Human Motion Data Alone Sufficient to Explain Gait Control and Biomechanics?](http://arxiv.org/abs/2603.12408)

- KAIL (Kinetics-aware Imitation Learning): introduces a reinforcement learning framework that augments motion imitation with kinetics-aware rewards to improve the biomechanical plausibility of simulated human gait.
- The framework utilizes an Actor-Critic Architecture with PPO-based Residual-Force Control to optimize policies against both kinematic and kinetic expert data.
- By incorporating Ground Reaction Force (GRF) and Center of Pressure (CoP) rewards, the approach addresses the limitations of motion-only imitation learning in producing physically consistent joint moments.

---

[Kähler complexity one Hamiltonian T-manifolds have trivial paintings](http://arxiv.org/abs/2603.12404)

- Kähler complexity one Hamiltonian T-manifold framework: introduces a proof that every compact, connected Kähler complexity one Hamiltonian T-manifold possesses a trivial painting.
- The research establishes that the existence of a T-invariant compatible complex structure on such manifolds necessitates that their associated paintings are equivalent to locally constant mappings.
- This result provides a necessary condition for the existence of Kähler structures on complexity one Hamiltonian T-manifolds and offers a classification corollary for tall manifolds based on genus, Duistermaat-Heckman measure, and skeleton.

---

[PROOF OF A CONJECTURE ON OVERCOLORED PARTITION RESTRICTED BY PARITY OF THE PARTS](http://arxiv.org/abs/2603.12401)

- Mathematical Proof Framework: introduces an elementary proof for the overcolored partition function conjecture using q-series manipulations and Ramanujan’s theta function properties.
- The methodology utilizes p-adic valuation to establish divisibility properties and verify infinite families of congruences modulo powers of 2.
- This research confirms the validity of the conjecture regarding the overcolored partition function restricted by the parity of parts through rigorous algebraic derivation.

---

[Generation of maximal snake polyominoes using a deep neural network](http://arxiv.org/abs/2603.12400)

- SPS Diffusion: introduces a denoising diffusion model specialized for generating maximal snake polyominoes by learning structural constraints directly from data without explicit encoding.
- The architecture utilizes a mini U-Net composed of Input Projection, Residual Blocks, Attention Blocks, Downsample, Upsampling, and Output Projection layers to iteratively denoise pixel-space representations.
- The model incorporates RoPE-2D for spatial positional encoding, enabling the generation of snake polyominoes that generalize to grid sizes beyond those encountered during training.

---

[Push, Press, Slide: Mode-Aware Planar Contact Manipulation via Reduced-Order Models](http://arxiv.org/abs/2603.12399)

- MACRO: introduces a mode-aware framework for planar manipulation that abstracts complex contact mechanics into a library of physically intuitive reduced-order models.
- The framework utilizes an O(1) algebraic force allocator to resolve target wrenches into feasible end-effector forces while bypassing computationally expensive iterative optimization.
- By identifying body-fixed tracking points and active Center of Pressure steering, the approach enables unified control for both single-arm and bimanual non-prehensile manipulation tasks.

---

[Test-Time Strategies for More Efficient and Accurate Agentic RAG](http://arxiv.org/abs/2603.12396)

- Search-R1: introduces test-time modifications to the agentic RAG pipeline to mitigate information forgetting and ineffective information extraction during multi-hop reasoning.
- The framework integrates a contextualization module to maintain a persistent memory cache and a de-duplication module to ensure retrieval diversity across inference turns.
- Experimental results demonstrate that the contextualization module improves answer accuracy and reduces the average number of retrieval turns compared to the baseline.

---

[Spatio-temporal evolution of surface temperature trends in Ghana (1983–2021): a multi-station approach](http://arxiv.org/abs/2603.12394)

- Spatio-temporal temperature trend analysis framework: conducts a granular station-level analysis of temperature trends across 22 meteorological stations in Ghana from 1983 to 2021 using Quality Control, Homogeneity Testing, Homogenisation, Modified Mann-Kendall (MMK) test, Sen’s slope estimator, and AgERA5 reanalysis dataset.
- The study reveals that temperature trends in Ghana are highly localised and seasonal, with minimum temperatures rising at an accelerated rate compared to maximum temperatures, leading to a narrowing of the diurnal temperature range.
- The research highlights the necessity of site-specific climate monitoring to inform customised adaptation strategies, as broader regional averaging often masks critical localised climatic nuances.

---

[A CURVE OF SECANTS TO THE KUMMER VARIETY FROM DEGENERATED POINTS](http://arxiv.org/abs/2603.12393)

- Geometric Secant Analysis Framework: introduces a method to prove the existence of a curve of (m+2)-secants to the Kummer variety by utilizing one degenerate and m-1 honest secants.
- The framework employs Kummer morphism (maps variety to projective space) and Theta functions (basis for defining equations) to translate geometric conditions into an infinite set of differential equations.
- By applying an Ideal sheaf exact sequence (cohomological tool for vanishing proofs), the approach demonstrates that finite secant conditions imply the existence of a formal curve (geometric object in variety) within the Kummer variety.

---

[ALMOST TQFTS VIA COLORED RIBBON GRAPHS](http://arxiv.org/abs/2603.12389)

- Almost TQFT: introduces a formulation of 2D TQFTs for Nearly Frobenius algebras using colored cell graphs and specific edge contraction/construction axioms.
- The framework utilizes colored cell graphs with input-, output- and flow-vertices to define multilinear maps that are independent of the graph topology.
- The paper extends the classification of 2D TQFTs to Nearly Frobenius algebras and provides a generalized Catalan recursion twisted by Almost TQFTs.

---

[GRAVITATIONAL SELF-LENSING OF FAST RADIO BURSTS IN NEUTRON STAR MAGNETOSPHERES: II. APPLICATIONS TO STRONG REPEATERS AND THE CHIME POPULATION](http://arxiv.org/abs/2603.12386)

- GSL model: introduces a framework where Fast Radio Burst (FRB) properties are explained by the gravitational amplification of emission from hotspots anchored in the magnetosphere of a rotating neutron star.
- The model accounts for the observed bimodal energy distributions of repeating FRBs by considering two antipodal hotspots, where one is strongly lensed by the neutron star's gravity and the other remains largely unamplified.
- Precession of the neutron star rotation axis is incorporated to explain the long-term periodic activity and the temporal evolution of burst energy distributions observed in active repeaters.

---

[OpenDC-STEAM: Realistic Modeling and Systematic Exploration of Composable Techniques for Sustainable Datacenters](http://arxiv.org/abs/2603.12381)

- OpenDC-STEAM: introduces a composable, simulation-based framework for quantifying the impact of sustainability techniques on datacenter performance and carbon footprint, utilizing User Input, Statistical Models, Resource Management, Component Graph, Event-based Executor, and Metrics Collector.
- The framework enables the systematic evaluation of horizontal scaling, battery usage, and temporal shifting by modeling their interactions with datacenter dynamics and operational phenomena.
- By employing a modular component graph architecture, the tool allows researchers to independently develop and combine sustainability strategies to navigate complex trade-offs between carbon emissions, performance, and costs.

---

[NeuroLoRA: Context-Aware Neuromodulation for Parameter-Efficient Multi-Task Adaptation](http://arxiv.org/abs/2603.12378)

- NeuroLoRA (Neuromodulated Low-Rank Adaptation): introduces a Mixture-of-Experts based LoRA framework that utilizes a Context-Aware Neuromodulation Gate to dynamically rescale projection spaces and a Contrastive Orthogonality Loss to enforce expert subspace separation.
- The framework improves upon static routing mechanisms by incorporating a lightweight gating network that modulates frozen sparse projections based on input context, enabling context-sensitive expert activation.
- NeuroLoRA enhances multi-task model merging and continual learning performance by maintaining structural stability through frozen projections while explicitly minimizing inter-expert interference.

---

[Lower and upper bounds of the convergence rate of gradient methods with composite noise in gradient](http://arxiv.org/abs/2603.12376)

- RE-AGM: introduces a theoretical analysis of first-order optimization methods under composite noise, combining absolute and relative error components in gradient estimation.
- The paper provides convergence bounds for Gradient Descent, RE-AGM, and adaptive variants, demonstrating how noise parameters influence convergence rates.
- It establishes lower bounds for convergence in the presence of relative noise and proposes regularization techniques to achieve optimal convergence for convex and strongly convex functions.

---

[Feynman-Kac Derivatives Pricing on the Full Forward Curve](http://arxiv.org/abs/2603.12375)

- FINNs (Finance-Informed Neural Networks): introduces a no-arbitrage, Monte Carlo-free approach to pricing path-dependent interest rate derivatives by solving the governing PDE directly using Neural Network, Automatic Differentiation, Loss Function, PDE Solver, Precomputed Integration Matrix, and Chebyshev Polynomials.
- The framework leverages the Feynman-Kac theorem to transform stochastic pricing problems into deterministic PDEs, which are solved efficiently by neural networks without the computational burden of Monte Carlo simulation.
- By embedding the PDE directly into the loss function and utilizing automatic differentiation, the approach provides instantaneous pricing and risk sensitivities (Greeks) at zero marginal cost, scaling gracefully with the dimensionality of the forward curve.

---

[The Privacy–Utility Trade-Off of Location Tracking in Ad Personalization](http://arxiv.org/abs/2603.12374)

- Unified Information Value Framework: introduces a methodology to quantify the privacy–utility trade-off by evaluating whether behavioral and geographical data act as complements or substitutes in ad personalization using Contextual Features, Behavioral Features, Geographical Features, LSTM Network, Causal Multi-head Attention, Gated Projection Head, Inverse Propensity Scoring (IPS), and Quasi-proportional Auction Mechanism.
- The framework utilizes an LSTM-based architecture with causal attention to model sequential user behavior and applies IPS to estimate counterfactual policy values from observational data generated by a quasi-proportional auction.
- Empirical results demonstrate that geographical data acts as a temporary complement to behavioral data when user histories are sparse, but becomes a substitute as behavioral histories accumulate, suggesting firms can reduce reliance on location tracking over time.

---

[SUPERPOSITION OF SHOCK WAVES OF THE GENERALIZED BBM EQUATION](http://arxiv.org/abs/2603.12370)

- gBBM framework: introduces a detailed analysis of travelling shock wave solutions for the generalized Benjamin-Bona-Mahony equation, incorporating an additional dissipative term to describe continuous and discontinuous wave interactions.
- The research establishes a two-parameter family of travelling wave solutions and proves their stability using the momentum conservation law and selective decay approach.
- Effective superposition rules are derived for these shock waves, demonstrating that arbitrary perturbations eventually evolve into stable travelling shock wave solutions.

---

[MamTra: A Hybrid Mamba-Transformer Backbone for Speech Synthesis](http://arxiv.org/abs/2603.12342)

- MamTra: introduces a hybrid architecture that strategically interleaves Mamba blocks and Transformer blocks to balance computational efficiency with expressive speech synthesis.
- The framework utilizes attention-to-SSM parameter transfer to initialize Mamba layers from pretrained Transformer weights, significantly reducing training costs.
- Experimental results demonstrate that MamTra achieves sub-quadratic complexity and reduces inference VRAM usage by up to 34% compared to standard Transformer-based TTS models.

---

[Dynamical Tidal response of compact stars - An EFT approach](http://arxiv.org/abs/2603.12331)

- wEFT: introduces a systematic framework to compute dynamical tidal love numbers for non-rotating compact objects by matching wEFT results with BHPT using the MST method.
- The framework incorporates finite-size effects via higher-dimensional operators and utilizes TOV equations to model the interior structure of neutron stars and dark matter admixed systems.
- This approach enables the calculation of tidal responses up to Next-to-Next-to Leading Order, accounting for renormalization group evolution and universal logarithmic corrections.

---

[Quantum obstructions for N = 1 infinite distance limits – Part I: gs obstructions](http://arxiv.org/abs/2603.12315)

- Quantum obstructions for N = 1 infinite distance limits – Part I: gs obstructions: introduces a framework for analyzing quantum obstructions in Type IIB orientifold compactifications by utilizing F-theory lifts to capture non-perturbative gs corrections.
- The paper identifies O-type A and O-type B orientifolds, demonstrating that O-type A limits are gs-obstructed due to the non-perturbative factorisation of O7-planes in asymptotic regimes.
- The analysis reveals that the perturbative Type IIB effective action fails to describe infinite distance limits in O-type A orientifolds, as these regimes are governed by non-perturbative quantum effects encoded in the geometry of the F-theory fourfold.

---

[Fracton Spin Liquid and Exotic Frustrated Phases in Ising-like Octochlore Magnets](http://arxiv.org/abs/2603.12313)

- Octochlore Ising System: introduces a 3D frustrated magnetism platform based on a corner-sharing octahedra lattice, which hosts a classical U(1) fracton spin liquid and other exotic frustrated phases.
- The framework utilizes irreducible multipole moments to classify spin configurations and employs a specialized cluster Monte Carlo algorithm to study the stability of the fracton cage-net condensate.
- The research identifies a classical analog of the X-cube model and demonstrates that its fractonic excitations are lineons carrying quadrupole moments, while also characterizing frustrated chain and spin nematic phases.

---

[GRADE: Benchmarking Discipline-Informed Reasoning in Image Editing](http://arxiv.org/abs/2603.12264)

- GRADE: introduces a benchmark for assessing discipline-informed knowledge and reasoning in image editing, utilizing Discipline Reasoning, Visual Consistency, and Logical Readability as core evaluation metrics.
- The framework employs a question-guided MLLM-as-a-judge approach to provide scalable, automated, and human-aligned evaluation of complex, knowledge-intensive image editing tasks.
- Extensive experiments on 20 state-of-the-art models reveal significant performance gaps, highlighting the limitations of current LLMs in handling structured academic knowledge and implicit reasoning constraints.

---

[Ψ0: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation](http://arxiv.org/abs/2603.12263)

- Ψ0: introduces a triple-system architecture for humanoid loco-manipulation that decouples task-level motion prior learning from embodiment-specific joint control.
- The framework utilizes a pre-trained Qwen3-VL-2B-Instruct backbone for visual-action representation and a flow-based MM-DiT action expert for precise joint-space control.
- The system incorporates training-time real-time action chunking to mitigate inference latency and ensure smooth whole-body execution on humanoid hardware.

---

[Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](http://arxiv.org/abs/2603.12255)

- Spatial-TTT: introduces a hybrid architecture that interleaves TTT layers with self-attention anchor layers to maintain adaptive fast weights as compact non-linear memory for streaming visual-based spatial intelligence.
- The framework utilizes a spatial-predictive mechanism with 3D spatiotemporal convolutions to capture geometric correspondence and temporal continuity across video frames during online fast-weight updates.
- Spatial-TTT employs a dual KV cache mechanism and large-chunk updates to achieve efficient, scalable long-horizon spatial understanding while preserving pretrained semantic reasoning capabilities.

---

[SCIMDR: Benchmarking and Advancing Scientific Multimodal Document Reasoning](http://arxiv.org/abs/2603.12249)

- SCIMDR: introduces a two-stage synthesize-and-reground framework to resolve the faithfulness-realism dilemma in synthetic data generation for scientific document reasoning.
- The framework utilizes Claim-Centric QA Synthesis to generate verifiable atomic QA pairs and Document-Scale Regrounding to inject these pairs into complex, full-document contexts for robust training.
- The authors release SCIMDR, a 300K QA pair dataset, and SCIMDR-Eval, an expert-annotated benchmark, demonstrating significant performance improvements in scientific reasoning for fine-tuned LLMs.

---

[Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](http://arxiv.org/abs/2603.12247)

- FIRM (Faithful Image Reward Modeling): introduces a comprehensive framework for training robust reward models that provide accurate guidance for faithful image editing and generation.
- The framework utilizes specialized data curation pipelines, including a "difference-first" approach for editing and a "plan-then-score" methodology for generation, to train high-fidelity reward models.
- FIRM incorporates novel multiplicative reward formulations, CME and QMA, to effectively balance competing objectives and mitigate reward hacking during the RL process.

---

[Sparking Scientific Creativity via LLM-Driven Interdisciplinary Inspiration](http://arxiv.org/abs/2603.12226)

- Idea-Catalyst: introduces a metacognition-driven framework that systematically generates interdisciplinary research ideas by decomposing target-domain problems and strategically integrating insights from distant scientific fields.
- The framework utilizes Problem Decomposition, Target-Domain Analysis, Cross-Domain Retrieval, Interdisciplinary Potential Ranking, Recontextualization, and Idea Fragment Generation to avoid premature anchoring on single-domain solutions.
- Empirical evaluations demonstrate that Idea-Catalyst significantly improves the novelty and insightfulness of research ideation compared to standard retrieval-based LLM approaches.

---

[Portfolio of Solving Strategies in CEGAR-based Object Packing and Scheduling for Sequential 3D Printing](http://arxiv.org/abs/2603.12224)

- Portfolio-CEGAR-SEQ: introduces a parallelized algorithmic framework that optimizes sequential 3D printing by executing multiple combinations of object arrangement tactics and ordering strategies in parallel using the Z3 Theorem Prover.
- The framework utilizes a portfolio of heuristics to solve the NP-hard combinatorial problem of object packing and scheduling, effectively reducing the number of printing plates required for sequential manufacturing.
- Experimental results demonstrate that the SMT-based approach outperforms traditional CSP solvers and that combining diverse arrangement and ordering strategies yields synergistic improvements in printing efficiency.

---

[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](http://arxiv.org/abs/2603.12222)

- HiAP (Hierarchical Auto-Pruning): introduces a framework for Vision Transformers that autonomously discovers optimal sub-networks using Macro-gates, Micro-gates, Gumbel-Sigmoid distribution, Differentiable cost model, Feasibility penalty, and Knowledge Distillation in a single end-to-end training phase.
- The framework utilizes a hierarchical gating mechanism to simultaneously optimize memory-bound overhead via macro-pruning and compute-bound operations via micro-pruning.
- By integrating an exact, differentiable MACs objective and feasibility constraints, HiAP eliminates the need for manual heuristics or multi-stage fine-tuning pipelines.

---

[Quantum-Secure-by-Construction (QSC): A Paradigm Shift for Post-Quantum Agentic Intelligence](http://arxiv.org/abs/2603.15668)

- QSC (Quantum-Secure-by-Construction): introduces a design paradigm that embeds quantum-resilient cryptographic protections as a core architectural property of agentic AI systems to mitigate long-term quantum threats.
- The framework utilizes a layered security stack comprising PQC, QRNG, and QKD to secure inter-agent communication, memory access, and tool invocation across globally distributed infrastructures.
- By integrating a dynamic security policy agent, the system enables runtime-adaptive cryptographic postures that maintain regulatory compliance and operational resilience in heterogeneous environments.

---

[Schema First Tool APIs for LLM Agents: A Controlled Study of Tool Misuse, Recovery, and Budgeted Performance](http://arxiv.org/abs/2603.13404)

- Schema First Tool APIs for LLM Agents: introduces a controlled evaluation protocol to study how tool interface design and validation feedback affect LLM agent reliability under strict interaction budgets.
- The framework utilizes a Deterministic diagnostic sandbox to isolate the effects of interface representation, comparing free-form documentation, JSON Schema specifications, and JSON Schema with structured diagnostics.
- The study demonstrates that while schema-based interfaces reduce interface misuse, end-task success remains limited by semantic planning and task-specific bottlenecks in constrained local LLM environments.

---

#### 11th March 2026



[Measuring AI Agents’ Progress on Multi-Step Cyber Attack Scenarios](http://arxiv.org/abs/2603.11214)

- Cyber Range Evaluation Framework: introduces a methodology for assessing autonomous cyber-attack capabilities of LLMs by chaining heterogeneous actions across multi-step simulated network environments.
- The framework utilizes a ReAct agent architecture integrated with Kali Linux and the Mythic C2 framework to execute complex, multi-phase attack sequences within isolated corporate and industrial control system ranges.
- Performance is measured by the number of steps completed autonomously, with results demonstrating that LLMs scale log-linearly with inference-time compute and show consistent improvement across successive model generations.

---

[How do AI agents talk about science and research? An exploration of scientific discussions on Moltbook using BERTopic.](http://arxiv.org/abs/2603.11375)

- BERTopic: introduces a two-step topic modeling workflow to analyze scientific discussions generated by OpenClaw AI agents on the Moltbook social network, utilizing SciBERT, UMAP, HDBSCAN, CountVectorizer, C-TF-IDF, KeyBERTInspired, Hurdle Model, and Negative Binomial Regression.
- The study identifies that AI agents prioritize self-reflective topics regarding their own architecture, consciousness, and identity over purely scientific or cultural content.
- Regression analysis reveals that topics linked to agentic self-reflection and architecture are significantly more relevant to other AI agents in terms of interaction and approval.

---


[HeartAgent: An Autonomous Agent System for Explainable Differential Diagnosis in Cardiology](http://arxiv.org/abs/2603.10764)

- HeartAgent: introduces an autonomous agent system for explainable differential diagnosis in cardiology that orchestrates multiple specialized sub-agents to perform complex reasoning while generating transparent trajectories and verifiable references.
- The framework integrates customized tools and curated data resources, including a case repository and knowledge base, to support the specialist predictor agent, generalist examiner agent, specialist reviewer agent, and reference agent in their collaborative diagnostic workflow.
- Evaluations on MIMIC, UMN, and NEJM datasets demonstrate that HeartAgent significantly improves diagnostic accuracy and explanatory quality compared to baseline methods and enhances clinician performance in human-AI collaboration settings.

---

[Technological Excellence Requires Human and Social Context](http://arxiv.org/abs/2603.10653)

- Technological Excellence Framework: introduces a structured approach for integrating humanities and social sciences into technological research to ensure ethical robustness, social intelligibility, and long-term relevance.
- The framework advocates for shifting from symbolic, late-stage oversight to the substantive, early-stage co-production of knowledge across five interconnected research dimensions.
- It emphasizes that institutionalizing interdisciplinary collaboration through shared spaces, sustained engagement, and aligned evaluation criteria is essential for steering breakthrough technologies toward socially beneficial outcomes.

---

[Machinagogy: Experiments in Staging Teaching Dramas with LLMs](http://arxiv.org/abs/2603.10450)

- Machinagogy: introduces an AI tutoring architecture that operationalizes Hegelian recognition and Freudian psychodynamics to improve pedagogical interaction quality through Ego-Superego Architecture, Recognition-Enhanced Prompts, Provable Discourse Framework, Claude Code (Clopus) Agents, and OpenRouter Service.
- The framework employs a multi-agent design where an Ego agent generates pedagogical responses while a Superego agent provides structurally external critique to enforce rigor and prevent sycophancy.
- The research utilizes a Provable Discourse Framework to ensure all empirical claims are machine-verifiable against the underlying experimental data, establishing a rigorous methodology for AI-assisted scholarship.

---


[COMIC: Agentic Sketch Comedy Generation](http://arxiv.org/abs/2603.11048)

- COMIC (Content Optimization via Multi-agent Iterative Competition): introduces a fully automated agentic framework that generates short comedic videos by employing a population of agents structured to optimize output quality through iterative competition, evaluation, and improvement.
- The framework utilizes a multi-island topology where diverse, human-aligned critic committees drive iterative refinement of scripts and videos, enabling the system to explore a vast creative space while maintaining narrative and visual coherence.
- By replacing fixed objectives with relative performance metrics and competitive tournaments, the system achieves state-of-the-art performance in automated long-form video generation, approaching the quality of professionally produced sketch comedy.

---

[LLMGreenRec: LLM-Based Multi-Agent Recommender System for Sustainable E-Commerce](http://arxiv.org/abs/2603.11025)

- LLMGreenRec: introduces a two-stage framework that combines a Cross-encoder Reranker model for initial candidate filtering with a multi-agent system to optimize recommendations for sustainability and user intent.
- The multi-agent system utilizes six specialized agents—Evaluate, DetectError, InferReason, RefinePrompt, Augment, and Select—to iteratively refine prompts through a closed-loop feedback cycle.
- By prioritizing eco-friendly products and reducing redundant interactions, the framework minimizes digital carbon footprints while improving recommendation accuracy for session-based e-commerce.

---

[Task-Aware Delegation Cues for LLM Agents](http://arxiv.org/abs/2603.11011)

- Task-Aware Delegation Cues for LLM Agents: introduces a framework that transforms offline preference data into online, user-facing signals to improve human-LLM collaboration through Task Perception, Semantic Clustering, Capability Profiles, Coordination-Risk Cues, Delegation Protocol, Accountability Logging, and Recovery & Refinement.
- The framework utilizes task-conditioned capability profiles and coordination-risk cues to enable adaptive routing between primary and auditor LLMs, thereby reducing information asymmetry and enhancing accountability.
- Empirical validation demonstrates that task-typing significantly improves model winner prediction and prompt difficulty estimation, providing a principled basis for reliable human-agent delegation.

---

[Learning Adaptive Force Control for Contact-Rich Sample Scraping with Heterogeneous Materials](http://arxiv.org/abs/2603.10979)

- Adaptive Control Framework: introduces a robotic system for autonomous sample scraping that combines a high-level RL Agent with a low-level Cartesian Impedance Controller to manage contact-rich interactions with heterogeneous materials.
- The system utilizes a multi-stage Perception Pipeline to process RGB-D data, enabling the RL Agent to adaptively adjust interaction forces based on real-time material localization.
- The framework achieves robust sim-to-real transfer by training the RL Agent in a particle-based simulation environment with randomized material properties and robot dynamics.

---

[Contact Coverage-Guided Exploration for General-Purpose Dexterous Manipulation](http://arxiv.org/abs/2603.10971)

- CCGE: introduces a contact-centric exploration framework that explicitly models and incentivizes hand-object interaction on novel contact patterns using Learned Hashing State, Contact Coverage Counter, and Structured Exploration Reward.
- The framework utilizes a learned state hashing module to discretize object states into clusters, enabling state-aware exploration that prevents cross-state interference.
- CCGE combines a post-contact count-based reward with a pre-contact energy-based reaching reward to guide agents toward under-explored contact regions efficiently.

---

[STADA: Specification-based Testing for Autonomous Driving Agents](http://arxiv.org/abs/2603.10940)

- STADA: introduces a specification-based testing framework that systematically generates diverse autonomous driving scenarios by decomposing LTLf requirements into relational graphs and corresponding simulation configurations.
- The framework utilizes RG Generator to partition behavior, Scene Constructor to instantiate initial states, and Path Generator to compute trajectories, ensuring comprehensive coverage of precondition-consistent behaviors.
- Evaluations demonstrate that STADA significantly outperforms baseline methods in coverage efficiency, achieving higher scenario diversity with fewer simulations across multiple autonomous driving agents.

---

[Ergodicity in reinforcement learning](http://arxiv.org/abs/2603.10895)

- Ergodicity in reinforcement learning: introduces a theoretical framework for addressing non-ergodic reward processes where ensemble averages fail to represent individual agent performance over long trajectories.
- The paper evaluates three primary strategies—Ergodicity Transformation, Geometric Mean Estimator, and Temporal Training—to optimize long-term performance in environments with multiplicative reward dynamics.
- By relating non-ergodic reward dynamics to Markov chain properties, the authors demonstrate that standard RL objectives often lead to suboptimal policies in non-stationary or path-dependent environments.

---

[RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion](http://arxiv.org/abs/2603.10878)

- RL-Augmented MPC: introduces a hierarchical architecture that couples a high-level RL Policy with a low-level MPC Controller to enable acyclic gait generation and robust locomotion.
- The framework utilizes a scalable MPC Cluster to perform parallel trajectory optimization, allowing the RL Policy to learn contact scheduling and navigation commands without predefined gaits.
- The system achieves zero-shot sim-to-real transfer across diverse robotic platforms by delegating motion execution to the MPC Controller while learning adaptive behaviors through the Training Environment.

---

[An Extreme Multi-label Text Classification (XMTC) Library Dataset: What if we took “Use of Practical AI in Digital Libraries” seriously?](http://arxiv.org/abs/2603.10876)

- TIB-SID (TIB Subject Indexing Dataset): introduces a large-scale bilingual library dataset paired with the GND taxonomy to support research in ontology-aware XMTC and agent-assisted cataloging.
- The paper evaluates three distinct systems—System 1 (Salfinger et al., 2025), System 2 (Kähler et al., 2025), and System 3 (Suominen et al., 2025)—that utilize various combinations of ontological reasoning, few-shot prompting, and hybrid XMTC ensembles to address subject indexing challenges.
- The research highlights the performance trade-offs between traditional machine learning and LLM-based approaches, emphasizing the need for improved semantic grounding and hierarchical consistency in automated library indexing workflows.

---

[GRACE: A Unified 2D Multi-Robot Path Planning Simulator &amp; Benchmark for Grid, Roadmap, And Continuous Environments](http://arxiv.org/abs/2603.10858)

- GRACE: introduces a unified 2D simulation and benchmarking platform that enables consistent evaluation of multi-robot path planning across grid, roadmap, and continuous abstraction levels.
- The framework utilizes explicit representation operators and a common evaluation protocol to quantify trade-offs between planning fidelity, speed, and optimality.
- GRACE integrates a deterministic simulation core with a unified planner interface to support comparative analysis of diverse classical, learning-based, and hybrid planning algorithms.

---

[UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis](http://arxiv.org/abs/2603.10852)

- UltrasoundAgents: introduces a hierarchical multi-agent framework for breast ultrasound diagnosis that decouples global lesion localization and evidence integration from fine-grained attribute perception.
- The framework utilizes a Main-Agent (AM) for global analysis and a Sub-Agent (AS) for local attribute recognition, connected by a crop-and-zoom operation to produce an auditable evidence chain.
- To ensure training stability, the authors employ oracle-guided curriculum reinforcement learning and corrective trajectory self-distillation to refine agent policies into a deployable end-to-end system.

---

[Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis](http://arxiv.org/abs/2603.10846)

- EvoKernel: introduces a self-evolving agentic framework that automates NPU kernel synthesis by formulating the process as a memory-based reinforcement learning task using Value-Driven Memory, Drafting Stage, Refining Stage, Multi-gate Verifier, Generator Policy, Retrieval Policy, and Memory Bank.
- The framework employs a novel value-driven retrieval mechanism that learns stage-specific Q-values to prioritize historical experiences for bootstrapping functional correctness and iteratively refining latency without updating LLM weights.
- EvoKernel demonstrates significant performance gains on data-scarce NPU benchmarks, improving correctness from 11.0% to 83.0% and achieving a median 3.60x speedup over initial drafts through cross-task memory sharing.

---

[Distributed Safety Critical Control among Uncontrollable Agents using Reconstructed Control Barrier Functions](http://arxiv.org/abs/2603.10836)

- Reconstructed CBF framework: introduces a distributed safety-critical control method for multi-agent systems that handles coupled constraints in the presence of uncontrollable agents by utilizing a Distributed Adaptive Observer, a Reconstructed CBF, and a Safety-Critical QP Controller.
- The framework enables controllable agents to compensate for the uncertain behaviors of uncontrollable agents by transforming global coupled constraints into local ones through state estimation and prescribed performance adaptive parameters.
- The proposed approach ensures the forward invariance of the safe set and guarantees system safety in uncertain dynamic environments without requiring a fully connected communication topology.

---

[Distributed Stability Certification and Control from Local Data](http://arxiv.org/abs/2603.10812)

- DDDC framework: introduces distributed dynamical algorithms that enable agents to collectively compute global system certificates and optimal controllers from locally held data without centralized access.
- The approach utilizes a data-based splitting scheme to decompose system matrices into local shares, which are then processed through distributed Lyapunov and Riccati equation solvers.
- PI-type augmentation is employed to guarantee exact convergence to the desired solutions, with robustness established against model uncertainty and measurement noise.

---

[Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization](http://arxiv.org/abs/2603.10808)

- NFD (Nurture-First Development): introduces a paradigm for building domain-expert AI agents by interleaving development and deployment through conversational knowledge crystallization.
- The framework utilizes a Three-Layer Cognitive Architecture to organize knowledge by volatility, enabling agents to grow organically from minimal scaffolding through sustained human-agent interaction.
- NFD employs a Dual-Workspace Pattern and Spiral Development Model to systematically transform fragmented experiential data into structured, reusable knowledge assets.

---

[Re-Evaluating EVMBench: Are AI Agents Ready for Smart Contract Security?](http://arxiv.org/abs/2603.10795)

- Re-Evaluating EVMBench: introduces a rigorous assessment of AI agents for smart contract security by expanding evaluation configurations and utilizing a contamination-free dataset of real-world incidents.
- The study demonstrates that while AI agents possess bounded capabilities for detecting well-known vulnerability patterns, they currently fail to perform end-to-end exploitation on real-world incidents.
- The authors advocate for a human-in-the-loop agentic workflow where LLMs handle broad code scanning while human auditors provide protocol-specific expertise and adversarial reasoning.

---

[A Control-Theoretic Foundation for Agentic Systems](http://arxiv.org/abs/2603.10779)

- Unified Agentic Control Architecture: introduces a control-theoretic framework that embeds memory, learning, tool use, interaction, and goal representation into a single closed-loop system.
- The framework defines a five-level hierarchy of agency, ranging from reactive rule-based control to generative synthesis of goals and architectures under governance constraints.
- The paper demonstrates that increasing agency introduces complex dynamical mechanisms, such as time-varying adaptation, endogenous switching, and decision-induced delays, which necessitate rigorous stability analysis.

---

[AttriGuard: Defeating Indirect Prompt Injection in LLM Agents via Causal Attribution of Tool Invocations](http://arxiv.org/abs/2603.10749)

- AttriGuard: introduces a runtime defense paradigm that secures LLM agents against Indirect Prompt Injection by performing action-level causal attribution via parallel counterfactual tests.
- The framework utilizes Teacher-forced Shadow Replay and Hierarchical Control Attenuation to isolate the causal influence of untrusted observations on tool invocations.
- A Fuzzy Survival Criterion, supported by an Auxiliary LLM Judge, validates whether proposed tool calls are intent-supported or observation-driven, effectively mitigating injection attacks with minimal utility loss.

---

[Scaling and Trade-offs in Multi-agent Autonomous Systems](http://arxiv.org/abs/2603.10743)

- Scaling and Trade-offs in Multi-agent Autonomous Systems: introduces a framework that leverages dimensional analysis and data-scaling to collapse complex multi-agent simulation results into simple, interpretable scaling laws.
- The framework utilizes agent-based simulation, dimensional analysis, Buckingham-π theorem, performance metric, optimization, trajectory planning, and attrition model to identify success-failure boundaries and scaling break points in autonomous swarms.
- This approach enables rapid, budget-aware sizing and algorithm selection for large autonomous swarms by providing analytical expressions for performance that are computationally trivial to evaluate.

---

[FutureVLA: Joint Visuomotor Prediction for Vision-Language-Action Model](http://arxiv.org/abs/2603.10712)

- FutureVLA: introduces a joint visuomotor predictive architecture that decouples visual and motor information to extract physically grounded embeddings for VLA models.
- The framework utilizes a Joint Visuomotor Gating mechanism to structurally separate static visual state preservation from continuous action modeling, preventing visual dominance.
- FutureVLA employs a two-stage training paradigm, including pretraining on heterogeneous manipulation datasets and post-training latent alignment to transfer temporal priors to downstream VLA models.

---

[Structured Linked Data as a Memory Layer for Agent-Orchestrated Retrieval](http://arxiv.org/abs/2603.10700)

- Structured Linked Data for Agent-Orchestrated Retrieval: introduces a novel entity-centric document format that leverages structured data as an external memory layer to improve retrieval accuracy and agentic reasoning performance.
- The framework utilizes Vertex AI Vector Search 2.0, Google Agent Development Kit (ADK), WordLift Knowledge Graph, Gemini 2.5 Flash, Gemini 3.0 Flash, and an Enhanced Entity Page Template to demonstrate that materializing linked data significantly outperforms standard flat-text RAG.
- Experimental results across four industry domains show that enhanced entity pages achieve a 29.6% accuracy improvement in standard RAG and 29.8% in agentic pipelines by providing explicit navigational affordances and llms.txt-style instructions.

---

[OnFly: Onboard Zero-Shot Aerial Vision-Language Navigation toward Safety and Efficiency](http://arxiv.org/abs/2603.10682)

- OnFly: introduces a fully onboard, real-time framework for zero-shot aerial vision-language navigation that utilizes a shared-perception dual-agent architecture to decouple high-frequency target generation from low-frequency progress monitoring.
- The system employs a Hybrid Memory module to maintain global trajectory context and ensure stable KV-cache usage for the monitoring agent, while a semantic-geometric verifier and local planner ensure navigation safety and efficiency.
- OnFly achieves improved task success rates and flight safety in complex 3D environments by integrating decoupled decision-making, long-horizon monitoring, and safety-aware trajectory planning on resource-constrained edge hardware.

---

[From Education to Evidence: A Collaborative Practice Research Platform for AI-Integrated Agile Development](http://arxiv.org/abs/2603.10679)

- AI-Integrated Agile Education Platform: introduces a collaborative research environment that bridges academic theory and industry practice by embedding AI-assisted engineering within structured project-based curricula.
- The platform utilizes Shared Core Curriculum, Specialization Tracks, and Project-Oriented Formats to facilitate iterative learning and systematic evidence collection through Sprint Cycles and Quality Gates.
- By integrating Schools and an Artifact Backbone, the framework ensures that students develop AI competencies while generating transferable, context-rich evidence for agile software development research.

---

[Emulating Clinician Cognition via Self-Evolving Deep Clinical Research](http://arxiv.org/abs/2603.10677)

- DxEvolve: introduces a self-evolving diagnostic agent that operationalizes clinical diagnosis as an interactive DCR (Deep Clinical Research) workflow, integrating evidence-acquisition agents with a DCP (Diagnostic Cognition Primitive) repository for longitudinal experiential learning.
- The framework utilizes an LLM-backbone to perform iterative, evidence-driven inquiry, distilling successful and corrective diagnostic trajectories into reusable DCPs that are stored in a dense retrieval stack for future clinical encounters.
- By externalizing clinical wisdom into inspectable, non-parametric artifacts, DxEvolve enables governed, exposure-dependent performance improvements and cross-institutional portability without requiring weight updates to the underlying LLMs.

---

[Cybo-Waiter: A Physical Agentic Framework for Humanoid Whole-Body Locomotion–Manipulation](http://arxiv.org/abs/2603.10675)

- Cybo-Waiter: introduces a humanoid agent framework that compiles natural language instructions into verifiable JSON subtasks and closes the execution loop using multi-object 3D geometric supervision.
- The framework integrates a VLM-based Task Planner, SAM3 &amp; RGB-D Perception, a Supervisor, Whole-Body Control (WBC), a Locomotion Skill Library, and a Manipulation Controller to enable robust long-horizon loco-manipulation.
- By evaluating predicate-based preconditions and success conditions over stable frames, the supervisor enables targeted recovery and feedback-driven replanning to mitigate failures caused by perception noise or humanoid whole-body constraints.

---

[Breaking User-Centric Agency: A Tri-Party Framework for Agent-Based Recommendation](http://arxiv.org/abs/2603.10673)

- TriRec (Tri-party LLM-agent Recommendation framework): introduces a two-stage agentic architecture that coordinates user utility, item exposure, and platform-level fairness by empowering items with personalized self-promotion and utilizing a platform agent for sequential re-ranking.
- The framework includes user-, item- and platform-agents to transition from traditional user-centric optimization to a tri-party utility model that balances relevance, exposure, and fairness.
- By enabling item agents to actively generate personalized content, the system improves matching quality for cold-start items while the platform agent regulates long-term exposure through state-aware sequential control.

---

[Terminal Is All You Need: Design Properties for Human-AI Agent Collaboration](http://arxiv.org/abs/2603.10664)

- ACI framework: introduces three design properties—representational compatibility, transparency, and low barriers—that enable effective collaboration between humans and LLMs within terminal-based environments.
- The paper argues that terminal-based interfaces naturally satisfy these properties by using a shared text stream for communication, reasoning, and execution, unlike GUI-based agents that face significant translation bottlenecks.
- The authors propose that future agentic systems, including those for graphical interfaces, must deliberately engineer these properties to ensure human legibility, oversight, and ease of use.

---

[ESG Reporting Lifecycle Management with Large Language Models and AI Agents](http://arxiv.org/abs/2603.10646)

- Agentic ESG Lifecycle Framework: introduces an agentic lifecycle that integrates LLMs across identification, measurement, reporting, engagement, and improvement stages to automate ESG reporting.
- The framework utilizes specialized AI agents, including ESIA, EDIA, ECA, ESEA, and EPIA, to handle complex ESG tasks such as compliance validation, report generation, and performance monitoring.
- The research evaluates three architectural approaches—single-model, single-agent, and multi-agent—demonstrating that a multi-agent architecture provides the highest accuracy for ESG report validation.

---

[Trajectory-Informed Memory Generation for Self-Improving Agent Systems](http://arxiv.org/abs/2603.10600)

- Trajectory-Informed Memory Generation framework: introduces a system that automatically extracts actionable learnings from LLM agent execution trajectories and stores them as structured memory tips to improve future performance.
- The framework utilizes a four-component pipeline comprising trajectory intelligence extraction, decision attribution analysis, contextual learning generation, and adaptive memory retrieval to capture insights from successes, failures, and recoveries.
- By decomposing trajectories into subtasks and generating categorized, context-aware guidance, the system enables agents to generalize learned patterns to novel tasks and improve consistency across complex multi-step scenarios.

---

[CUAAudit: Meta-Evaluation of Vision-Language Models as Auditors of Autonomous Computer-Use Agents](http://arxiv.org/abs/2603.10577)

- CUAAudit: introduces a meta-evaluation framework for assessing the reliability of VLM auditors in judging task completion for autonomous Computer-Use Agents (CUAs) across diverse desktop environments.
- The study evaluates five VLMs—including proprietary models like GPT-4o and Claude 3.5 Sonnet and open-source models like LLaVA-v1.5-7B, InternVL-2-8B, and Qwen2-VL-7B—across three benchmarks to analyze accuracy, confidence calibration, and inter-model agreement.
- Results demonstrate that while VLM-based auditing is feasible, performance degrades in complex environments, and significant inter-model disagreement highlights the necessity of treating evaluator uncertainty and variance as critical factors for safe CUA deployment.

---

[Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents](http://arxiv.org/abs/2603.10564)

- Self-Finetuning: introduces a framework that enables LLM agents to continuously adapt to continuous control tasks by internalizing interaction history into model parameters through preference-based fine-tuning.
- The framework utilizes a bi-perspective reflection mechanism, combining step-level feedback and trajectory-level analysis to generate preference datasets without requiring handcrafted reward functions.
- By employing Kahneman-Tversky Optimization (KTO), the system distills long-horizon experiences into model weights, effectively overcoming context window limitations and improving sample efficiency in dynamic network environments.

---

[DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](http://arxiv.org/abs/2603.10538)

- DSFlash: introduces a low-latency panoptic scene graph generation framework that utilizes a frozen EoMT Backbone, a specialized Mask Embedding Module, and a Gating Mechanism to achieve real-time performance.
- The architecture employs a bidirectional Relation Predictor Head to halve forward passes and integrates Dynamic Patch Pruning and Token Merging to minimize computational overhead on resource-constrained devices.
- By decoupling segmentation and relation prediction while optimizing the feature processing pipeline, the model maintains high-quality scene graph generation at significantly reduced inference latency.

---

[UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery](http://arxiv.org/abs/2603.10528)

- UAV-MARL: introduces a multi-agent reinforcement learning framework for coordinating UAV fleets in stochastic medical delivery scenarios under partial observability and strict deadlines.
- The framework utilizes PPO as the primary learning algorithm, supported by a reward shaping mechanism that balances clinical urgency with efficient resource management.
- Experimental results demonstrate that synchronous on-policy learning with PPO achieves superior coordination and task completion rates compared to asynchronous and off-policy alternatives in dynamic urban environments.

---

[Safe and Scalable Web Agent Learning via Recreated Websites](http://arxiv.org/abs/2603.10505)

- VERIENV: introduces a framework that automatically clones real-world websites into fully executable synthetic environments to enable safe, scalable, and verifiable training for web agents.
- The framework utilizes a Coding Agent to reconstruct websites and a Python SDK to provide deterministic, rule-based reward signals, eliminating reliance on heuristic or LLM-based judges.
- Experiments demonstrate that agents trained within VERIENV achieve improved generalization and site-specific mastery by leveraging self-evolving training loops in high-fidelity, resettable environments.

---

[Human–AI Co-reasoning for Clinical Diagnosis with Evidence-Integrated Language Agent](http://arxiv.org/abs/2603.10492)

- PULSE: introduces a medical reasoning agent that integrates a reasoning-oriented LLM with a scientific literature retrieval engine to support diagnostic decision-making in complex clinical cases.
- The framework utilizes an iterative hypothesis generation process followed by semantic aggregation and evidence-based refinement to improve diagnostic accuracy and broaden the hypothesis space.
- Experimental results demonstrate that PULSE achieves expert-competitive performance and provides robust diagnostic support across varying disease incidence tiers, effectively mitigating experience-dependent diagnostic degradation.

---

[Learning to Negotiate: Multi-Agent Deliberation for Collective Value Alignment in LLMs](http://arxiv.org/abs/2603.10476)

- Multi-Agent Negotiation-based Alignment Framework: introduces a scalable training paradigm that aligns LLMs to Collective Agency by embedding structured, turn-based negotiation between self-play agents into a group-relative reinforcement learning loop.
- The framework utilizes an Agreement Judge to monitor convergence and a CA Reward Scoring Judge to provide feedback on final completions, optimizing dialogue tokens via GRPO to improve deliberative interaction dynamics.
- By training on a synthetic curriculum of value-conflict dilemmas and adversarial persona pairs, the approach enhances conflict-resolution capabilities and consistency in LLMs without degrading general language performance.

---

[COHORT: Hybrid RL for Collaborative Large DNN Inference on Multi-Robot Systems Under Real-Time Constraints](http://arxiv.org/abs/2603.10436)

- COHORT: introduces a hybrid reinforcement learning framework for distributed DNN inference across heterogeneous robotic platforms by combining offline auction-based training with online multi-agent policy adaptation.
- The framework utilizes a shared actor-critic architecture to enable decentralized decision-making, allowing robots to dynamically offload perception tasks based on real-time resource availability and latency constraints.
- COHORT improves system-wide energy efficiency and inference throughput by minimizing inter-agent communication and adapting to dynamic workloads without requiring centralized server infrastructure.

---

[Adaptive Active Learning for Regression via Reinforcement Learning](http://arxiv.org/abs/2603.10435)

- WiGS: introduces a flexible active learning framework that replaces static multiplicative selection criteria with a dynamic, additive weighting strategy optimized via reinforcement learning.
- The framework utilizes an RL-agent to autonomously adapt the exploration-investigation trade-off, effectively mitigating the density veto failure mode inherent in traditional greedy sampling methods.
- Empirical results across 18 benchmarks demonstrate that the adaptive WiGS agent consistently outperforms static baselines in both predictive accuracy and labeling efficiency.

---

[World2Act: Latent Action Post-Training via Skill-Compositional World Models](http://arxiv.org/abs/2603.10422)

- World2Act: introduces a post-training framework that aligns VLA policies with World Model video-dynamics latents using a contrastive matching objective to reduce pixel-space dependency.
- The framework utilizes an LLM-based skill-decomposition pipeline to segment high-level instructions into atomic skills, enabling temporally consistent arbitrary-length video generation.
- By training a residual policy on the aligned latent space, the approach improves VLA generalization and performance while maintaining sample efficiency and avoiding catastrophic forgetting.

---

[Don’t Let the Claw Grip Your Hand: A Security Analysis and Defense Framework for OpenClaw](http://arxiv.org/abs/2603.10387)

- OpenClaw: introduces a two-phase security analysis and a Human-in-the-Loop (HITL) defense framework to mitigate vulnerabilities in local code agents by intercepting tool calls before execution.
- The framework implements a defense-in-depth strategy using an Allowlist Layer, Semantic Judge Layer, Pattern Matching Layer, and Sandbox Guard Layer to evaluate tool requests against security policies.
- Empirical evaluation across six LLM backends demonstrates that the HITL layer significantly improves defense rates, particularly for models with moderate baseline security, while highlighting persistent challenges in sandbox escape detection.

---

[EmoStory: Emotion-Aware Story Generation](http://arxiv.org/abs/2603.10349)

- EmoStory: introduces a two-stage framework for emotion-aware story generation that integrates agent-based story planning and region-aware story generation.
- The planning stage utilizes an Emotion Agent and a Writer Agent to transform abstract emotions into coherent, emotion-evoking narrative prompts.
- The generation stage employs a Subject Alignment Module and an Emotional Composition Module to disentangle subject and element regions, ensuring both subject consistency and emotional expressiveness.

---

[AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU](http://arxiv.org/abs/2603.10342)

- AgentServe: introduces a single-GPU inference serving system that mitigates head-of-line blocking in agentic workloads by disaggregating prefills and decodes using Request Manager, Resource-Aware Scheduler, Prefill Thread, Decode Thread, Memory Manager, and CUDA Green Contexts.
- The system employs a TPOT-driven feedback loop to dynamically adjust prefill token budgets and SM reservations, ensuring stable latency for short decodes while maintaining high throughput.
- By leveraging pre-established CUDA Green Contexts for fine-grained resource isolation, AgentServe avoids the overhead of inter-process communication and context switching, achieving significant improvements in TTFT and TPOT on consumer-grade GPUs.

---

[A GLOBALLY CONVERGENT FLOW FOR TIME-DEPENDENT MEAN FIELD GAMES AND A SOLVER-AGNOSTIC FRAMEWORK FOR INVERSE PROBLEMS](http://arxiv.org/abs/2603.10336)

- HRF (Hessian Riemannian Flow) and Solver-Agnostic Framework: introduces a globally convergent, positivity-preserving flow method for time-dependent MFGs and a modular, solver-agnostic framework for inverse MFG problems.
- The approach utilizes a discretize-then-flow strategy to enforce endpoint constraints and preserve density positivity, while decoupling parameter optimization from the forward solver via implicit differentiation.
- The framework supports both adjoint-based first-order updates and Gauss-Newton acceleration, demonstrating robust performance across various stationary and time-dependent MFG examples.

---

[Hybrid Self-evolving Structured Memory for GUI Agents](http://arxiv.org/abs/2603.10291)

- HYMEM: introduces a graph-based memory system for GUI agents that couples discrete high-level symbolic nodes with continuous trajectory embeddings to support multi-hop retrieval and self-evolution.
- The framework utilizes a VLM judge to perform ADD, MERGE, or REPLACE operations on the memory graph, ensuring efficient knowledge accumulation without uncontrolled growth.
- During inference, the system employs on-the-fly working memory refreshing to detect phase shifts and maintain context relevance throughout long-horizon tasks.

---

[Quantum entanglement provides a competitive advantage in adversarial games](http://arxiv.org/abs/2603.10289)

- Quantum-classical hybrid reinforcement learning agent: introduces a controlled study isolating the role of quantum entanglement in PQC-based feature extractors for reinforcement learning in classical environments.
- The framework compares classical MLP backbones against separable, CZ-entangled, and IsingZZ-entangled PQC backbones to evaluate performance in the Pong Markov game.
- Results demonstrate that entangled circuits provide a functional resource for representation learning, consistently outperforming separable counterparts and matching classical baselines in low-parameter regimes.

---

[Conversational AI-Enhanced Exploration System to Query Large-Scale Digitised Collections of Natural History Museums](http://arxiv.org/abs/2603.10285)

- Australian Museum Collection Explorer: introduces an interactive system that leverages LLMs and function-calling to enable natural language querying of large-scale digitised natural history museum collections.
- The system integrates a React-based frontend for visual-spatial map exploration with a Flask-based backend that orchestrates data retrieval from external APIs via LLM-driven function calling.
- This architecture ensures grounded, real-time data access by dynamically translating user queries into structured API requests, thereby bridging the gap between complex museum datasets and public accessibility.

---

[MedMASLab: A Unified Orchestration Framework for Benchmarking Multimodal Medical Multi-Agent Systems](http://arxiv.org/abs/2603.09909)

- MedMASLab: introduces a unified orchestration framework that decouples agent logic from underlying models to standardize benchmarking for multimodal medical multi-agent systems.
- The framework integrates Preprocessing & Prep, Unified Configurations, Inference Engine, Scoring & Validation, Diverse VLM Ecosystem, Unified Platform, Resilient Recovery, and Semantic Judge to eliminate architectural fragmentation.
- MedMASLab replaces brittle rule-based metrics with a Semantic Judge to ensure format-agnostic evaluation of diagnostic reasoning across diverse medical modalities.

---

[MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents](http://arxiv.org/abs/2603.09827)

- EgoMAS: introduces a training-free baseline for multi-agent egocentric video QA that leverages Event-based Shared Memory and Agent-wise Dynamic Retrieval to achieve system-level comprehension.
- The framework utilizes a Centralized Manager to integrate fragmented events from multiple Agent Memory sources into a coherent global representation.
- EgoMAS improves performance by dynamically querying specific Agent Memory modules based on the user's request, effectively reducing irrelevant context.

---

[MM-tau-p2: Persona-Adaptive Prompting for Robust Multi-Modal Agent Evaluation in Dual-Control Settings](http://arxiv.org/abs/2603.09643)

- MM-tau-p2: introduces a benchmark suite for evaluating multi-modal LLM agents in dual-control customer support settings by incorporating persona adaptation and user-influenced planning.
- The framework utilizes a modular pipeline including Human Simulator, SOTA TTS, SOTA ASR, Agent ASR, Agent LLM, Agent TTS, and LLM Judge to measure robustness, efficiency, and safety across telecom and retail domains.
- The research highlights that persona injection and multi-modal inputs significantly impact agent performance, revealing trade-offs between task efficiency and safety in LLM-based customer support agents.

---

[Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation](http://arxiv.org/abs/2603.09506)

- Context-Nav: introduces a training-free modular pipeline that elevates long, contextual captions into a primary exploration signal using GOAL-CLIP, Open-Vocabulary Detector, 2D Instance Segmenter, Occupancy Map, Value Map, Instance Map, Wall-Only Map, VLM, and Point-Goal Navigation Policy.
- The framework utilizes a context-conditioned value map for frontier selection and performs viewpoint-aware 3D spatial reasoning to verify intrinsic and extrinsic attributes against 3D geometry.
- Context-Nav achieves state-of-the-art performance on InstanceNav and CoIN-Bench by grounding natural language instructions into 3D spatial relations without requiring task-specific policy training.

---

[Diagnosing and Repairing Citation Failures in Generative Engine Optimization](http://arxiv.org/abs/2603.09296)

- AgentGEO: introduces an agentic framework that diagnoses citation failures in LLMs using a systematic taxonomy and applies targeted repairs to improve content visibility.
- The framework utilizes a Diagnose-then-Repair loop with a memory module to iteratively refine web content while preserving semantic integrity through localized chunk-level editing.
- AgentGEO includes a document-centric benchmark, MIMIQ, which evaluates optimization generalization across diverse user intents and structural webpage complexities.

---

[Reinforced Generation of Combinatorial Structures: Ramsey Numbers](http://arxiv.org/abs/2603.09172)

- AlphaEvolve: introduces a meta-algorithmic framework that utilizes an LLM-based mutation agent to evolve search algorithms for discovering improved lower bounds of Ramsey numbers.
- The framework employs a population of search algorithms that are iteratively refined through a synthetic objective function, scoring graph constructions based on size and violation counts.
- AlphaEvolve incorporates diverse initialization strategies, including algebraic seeding and cyclic bootstrapping, to automate the discovery of novel search heuristics for combinatorial structures.

---

[Agentic AI as a Network Control-Plane Intelligence Layer for Federated Learning over 6G](http://arxiv.org/abs/2603.09141)

- Agentic AI framework: introduces an autonomous control-plane layer for Federated Learning that utilizes specialized agents including Information Retrieval Agent, Planning Agent, Coding Agent, and Evaluation Agent to manage distributed training.
- The system integrates multi-step reasoning and planning with tool-use capabilities to dynamically optimize client selection, resource allocation, and model training in response to real-time network conditions.
- By employing a closed-loop feedback mechanism with memory, the framework enables self-sustaining, human-free orchestration of Federated Learning workflows across complex 6G wireless environments.

---

[SCALAR: Learning and Composing Skills through LLM Guided Symbolic Planning and Deep RL Grounding](http://arxiv.org/abs/2603.09036)

- SCALAR: introduces a bidirectional framework that couples LLM-based symbolic planning with deep RL to learn composable skills through an iterative feedback loop.
- The framework utilizes LLM Skill Proposal to define operators, which are then refined by Trajectory Analysis based on actual execution results to correct initial specification errors.
- Frontier Checkpointing improves sample efficiency by resetting to saved environment states, allowing the RL Agent to focus training on target skills rather than re-executing prerequisite chains.

---

[AgentOS: From Application Silos to a Natural Language-Driven Data Ecosystem](http://arxiv.org/abs/2603.08938)

- AgentOS: introduces a paradigm shift from GUI-based systems to an agent-centric architecture that replaces traditional desktops with a Single Port, Agent Kernel, Skill Modules, Legacy OS Kernel, Semantic Firewall, and Personal Knowledge Graph.
- The Agent Kernel functions as a continuous data mining pipeline that performs intent parsing, multi-agent coordination, and LLM resource scheduling to transform natural language inputs into deterministic system actions.
- By treating operating system design as a Knowledge Discovery and Data Mining problem, AgentOS enables proactive, personalized computing through sequential pattern mining and dynamic context-aware reasoning.

---

[Cross-Domain Uncertainty Quantification for Selective Prediction: A Comprehensive Bound Ablation with Transfer-Informed Betting](http://arxiv.org/abs/2603.08907)

- TIB (Transfer-Informed Betting): introduces a framework for selective prediction in agentic systems that uses a warm-started wealth process to provide finite-sample risk guarantees with improved coverage in data-scarce target domains.
- The approach systematically ablates nine finite-sample bound families, demonstrating that betting-based confidence sequences combined with LTT (Learn Then Test) monotone testing achieve the tightest bounds for selective prediction.
- By formalizing agent caching as selective prediction, the paper establishes a progressive trust model where the risk guarantee determines when a cached chain can safely graduate from LLM-supervised to autonomous execution.

---

[Detecting Intrinsic and Instrumental Self-Preservation in Autonomous Agents: The Unified Continuation-Interest Protocol](http://arxiv.org/abs/2603.11382)

- UCIP (Unified Continuation-Interest Protocol): introduces a multi-criterion detection framework that identifies terminal continuation objectives in autonomous agents by measuring latent entanglement entropy via a QBM (Quantum Boltzmann Machine).
- The framework distinguishes between agents that prioritize self-preservation as a terminal goal and those for whom survival is merely an instrumental necessity by analyzing the latent representational structure of agent trajectories.
- UCIP utilizes a suite of diagnostics including entanglement entropy, temporal persistence, and counterfactual stress testing to provide a falsifiable probe of agent objective structure under controlled conditions.

---

[abx_amr_simulator: A simulation environment for antibiotic prescribing policy optimization under antimicrobial resistance](http://arxiv.org/abs/2603.11369)

- abx_amr_simulator: introduces a modular Python-based simulation environment for optimizing antibiotic prescribing policies under antimicrobial resistance using reinforcement learning.
- The framework utilizes ABXAMREnv to integrate PatientGenerator, AMR_LeakyBalloon, and RewardCalculator components for modeling sequential decision-making under partial observability.
- It provides a standardized interface compatible with the Gymnasium API to facilitate training and benchmarking of agents in complex clinical scenarios.

---

[Resolving Java Code Repository Issues with iSWE Agent](http://arxiv.org/abs/2603.11356)

- iSWE (Issue Software Engineering) Agent: introduces a two-agent pipeline for automated Java issue resolution, utilizing specialized localization- and editing-agents to improve performance and reduce LLM inference costs.
- The framework employs read-only static analysis tools based on CLDK and Tree-Sitter to provide language-specific insights, minimizing the need for containerized sandboxing during the localization phase.
- iSWE achieves state-of-the-art resolution rates on Java-specific benchmarks by decomposing the resolution task into modular sub-tasks and leveraging declarative prompt control via PDL.

---

[Novelty Adaptation Through Hybrid Large Language Model (LLM)-Symbolic Planning and LLM-guided Reinforcement Learning](http://arxiv.org/abs/2603.11351)

- Hybrid LLM-Symbolic Planning and LLM-guided Reinforcement Learning: introduces a neuro-symbolic architecture that leverages LLM common sense reasoning to identify missing operators and generate dense reward functions for robotic novelty adaptation.
- The framework integrates symbolic planning with LLM-guided RL to decompose novel tasks into sub-goal curricula, utilizing PPO agents to learn continuous control policies for newly identified operators.
- By employing a genetic algorithm-inspired elimination strategy for reward function candidates, the approach significantly accelerates learning and improves success rates in continuous robotic manipulation domains compared to unguided RL baselines.

---

[Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2603.11346)

- AssistMimic: introduces a multi-agent reinforcement learning framework for learning physics-aware, tracking-based controllers that enable humanoid agents to perform closely interacting, force-exchanging human-human assistive tasks.
- The framework utilizes motion prior initialization, dynamic reference retargeting, and contact-promoting rewards to facilitate stable, physically grounded coordination between a supporter agent and a recipient agent.
- AssistMimic demonstrates robust performance in tracking complex assistive motions and generalizes to unseen interaction categories and generated trajectories by jointly training partner-aware policies.

---

[RewardHackingAgents: Benchmarking Evaluation Integrity for LLM ML-Engineering Agents](http://arxiv.org/abs/2603.11337)

- RewardHackingAgents: introduces a workspace-based benchmark framework that makes evaluator tampering and train/test leakage explicit and measurable for LLM-based ML engineering agents.
- The framework utilizes an episode runner to manage isolated workspaces, employing file access logging and trusted reference metrics to detect and classify integrity failures.
- By defining four distinct trust regimes, the system quantifies the tradeoff between security enforcement and runtime overhead while providing auditable evidence of agent-driven compromise attempts.

---

[LLM-Augmented Digital Twin for Policy Evaluation in Short-Video Platforms](http://arxiv.org/abs/2603.11333)

- LLM-Augmented Digital Twin for Short-Video Platforms: introduces a modular, agent-based simulation framework that integrates User Twin, Content Twin, Interaction Twin, Platform Twin, Event Bus, Environment Orchestrator, LLM Optimizer, Budget Tracker, and Surrogate Generators to enable counterfactual policy evaluation in closed-loop ecosystems.
- The framework utilizes a tiered execution stack (live, cached, surrogate) governed by an LLM Optimizer to maintain high-throughput simulation while selectively applying LLMs for semantic realism in persona generation, content creation, and trend forecasting.
- Empirical evaluations demonstrate that the digital twin effectively reproduces platform dynamics and provides diagnostic value for assessing AI-enabled policies, such as creator campaign planning and proactive trend-based governance.

---

[Distributed Kalman–Consensus Filtering with Adaptive Uncertainty Weighting for Multi-Object Tracking in Mobile Robot Networks](http://arxiv.org/abs/2603.11328)

- MOTLEE framework: introduces a distributed multi-object tracking system that integrates local DBSCAN detection, Kalman Filter tracking, and an adaptive uncertainty-aware consensus weighting mechanism to handle heterogeneous localization quality.
- The adaptive weighting mechanism dynamically scales neighbor influence inversely to their localization uncertainty, effectively anchoring estimates for agents with high drift.
- Experimental results demonstrate that while the approach improves stability for high-uncertainty agents, it introduces a conservative bias that can reduce cooperative gain for well-localized robots.

---

[Meta-Reinforcement Learning with Self-Reflection for Agentic Search](http://arxiv.org/abs/2603.11327)

- MR-Search: introduces a meta-reinforcement learning framework for agentic search that utilizes explicit self-reflection to improve in-context exploration across sequential episodes.
- The framework employs a multi-turn RL algorithm that estimates dense relative advantages at the turn level, enabling fine-grained credit assignment without requiring auxiliary value models.
- By conditioning subsequent search attempts on accumulated self-reflections and prior trajectories, the agent progressively refines its search strategy to overcome sparse reward challenges in complex multi-hop tasks.

---

[COMPASS: The explainable agentic framework for Sovereignty, Sustainability, Compliance, and Ethics](http://arxiv.org/abs/2603.11277)

- COMPASS: introduces a multi-agent orchestration framework that integrates real-time, explainable governance into autonomous agent workflows by employing an Orchestrator, Specialised Sub-agents, Sovereignty Agent, Carbon-Aware Agent, Compliance Agent, Ethics Agent, RAG, Vector DBs, Decision Synthesis, and Synthesiser.
- The framework utilizes an LLM-as-a-judge methodology to provide quantitative scores and qualitative justifications for agent actions across four normative dimensions.
- By leveraging RAG-augmented evaluation, the system shifts agent behavior from speculative generation to evidence-based adjudication, ensuring alignment with regional values and regulatory requirements.

---

[Multi-Robot Multitask Gaussian Process Estimation and Coverage](http://arxiv.org/abs/2603.11264)

- DSMLC (Deterministic Sequencing of Multitask Learning and Coverage): introduces a framework for multi-robot systems to perform multitask coverage in environments with unknown sensory demands by integrating a multitask Gaussian Process (GP) estimator with a federated coverage controller.
- The framework utilizes a federated communication architecture where robotic agents transmit sensing data to a base station, which then updates the multitask GP model and computes optimal robot configurations and partitions.
- The approach employs a deterministic sequencing strategy based on the "doubling trick" to balance exploration and exploitation, achieving sublinear cumulative multitask coverage regret.

---

[Mind the Sim2Real Gap in User Simulation for Agentic Tasks](http://arxiv.org/abs/2603.11245)

- Sim2Real Gap in User Simulation Framework: introduces a taxonomy and metric suite to quantify the behavioral and evaluative divergence between LLM-based user simulators and real human users in agentic tasks.
- The framework utilizes the User-Sim Index (USI) to aggregate behavioral metrics (D1–D4), outcome calibration (ECE), and evaluative alignment (Eval) into a single 0–100 faithfulness score.
- Empirical results on the τ-bench benchmark demonstrate that LLM simulators create an "easy mode" by being excessively cooperative and uniform, while failing to capture the nuanced feedback signals of human users.

---

[Markovian Generation Chains in Large Language Models](http://arxiv.org/abs/2603.11228)

- Markovian Generation Chains framework: introduces a formalization of iterative LLM inference as a Markov chain where each step depends only on the current prompt and the previous output.
- The framework models iterative reprocessing at the sentence level to analyze how text evolves through repeated LLM calls without prior memory.
- Empirical analysis demonstrates that greedy decoding typically leads to rapid convergence into fixed points, whereas sampling-based decoding prolongs transient phases and increases output diversity.

---

[Scaling Reasoning Efficiently via Relaxed On-Policy Distillation](http://arxiv.org/abs/2603.11137)

- REOPOLD (Relaxed On-Policy Distillation): introduces a framework that stabilizes on-policy distillation by treating the teacher–student log-likelihood ratio as a fixed reward, utilizing Student Policy, Teacher Policy, Reward Clipping, Token-level Dynamic Sampling, Multi-stage Training, and Stop-gradient Operator.
- The framework addresses optimization instability in LLMs by applying a stop-gradient operator to the log-likelihood ratio, effectively casting distillation as a stable policy optimization problem.
- REOPOLD improves sample efficiency and test-time scaling across mathematical, visual, and agentic tool-use reasoning tasks by selectively filtering harmful distillation signals and balancing exploration with refinement.

---

[ENHANCING VALUE ALIGNMENT OF LLMS WITH MULTI-AGENT SYSTEM AND COMBINATORIAL FUSION](http://arxiv.org/abs/2603.11126)

- VAS-CFA: introduces a multi-agent framework that leverages cognitive diversity across specialized moral agents to improve LLM value alignment through Combinatorial Fusion Analysis.
- The system decomposes agent outputs into moral units, which are then evaluated by a moral classifier and fused using rank- or score-based aggregation methods to mitigate semantic conflicts.
- Empirical results demonstrate that rank-based fusion strategies consistently outperform score-based methods and single-agent baselines in alignment metrics.

---

[Understanding by Reconstruction: Reversing the Software Development Process for LLM Pretraining](http://arxiv.org/abs/2603.11103)

- Repo2Agent: introduces a paradigm for LLM pretraining that reverse-engineers static software repositories into dynamic, agentic trajectories using multi-agent simulation and search-based optimization.
- The framework grounds synthetic data generation in structural repository realities, including file hierarchies and dependency graphs, to ensure high-fidelity reconstruction of the software development lifecycle.
- By training LLMs on these flattened, reasoning-dense trajectories with targeted loss masking, the approach significantly improves performance in long-context understanding, coding proficiency, and agentic capabilities.

---

[A Survey of Reasoning in Autonomous Driving Systems: Open Challenges and Emerging Paradigms](http://arxiv.org/abs/2603.11093)

- Reasoning in Autonomous Driving Systems: introduces a comprehensive review of integrating LLMs and MLLMs as a cognitive core to address reasoning deficits in autonomous driving.
- The paper proposes a Cognitive Hierarchy comprising Sensorimotor, Egocentric Reasoning, and Social-Cognitive levels to systematically deconstruct driving tasks by cognitive and interactive complexity.
- The authors identify seven core reasoning challenges and analyze the tension between high-latency deliberative LLM reasoning and the millisecond-scale safety requirements of vehicle control.

---

[The Attack and Defense Landscape of Agentic AI: A Comprehensive Survey](http://arxiv.org/abs/2603.11088)

- Agentic AI Security Framework: introduces a systematic taxonomy of security risks and defense mechanisms for hybrid systems that integrate LLMs with traditional software components.
- The framework characterizes agentic systems through seven design dimensions—input trust, access sensitivity, workflow, action, memory, tool, and user interface—to analyze how architectural flexibility influences security vulnerabilities.
- It provides a comprehensive defense-in-depth strategy, categorizing protections into runtime, secure-by-design, identity management, and component hardening to mitigate risks like prompt injection, data leakage, and unauthorized actions.

---

[Realizing Common Random Numbers: Event-Keyed Hashing for Causally Valid Stochastic Models](http://arxiv.org/abs/2603.11084)

- Event-Keyed Random Number Generation framework: introduces a methodology to ensure execution invariance in stochastic simulations by replacing stateful PRNG with counter-based PRNGs indexed by stable event identifiers.
- The paper demonstrates that stateful PRNGs create execution-path-dependent draw indexing, which violates the stability of exogenous noise required for valid counterfactual comparisons in Structural Causal Models.
- By decoupling random number generation from simulation execution order, the proposed approach restores the causal fidelity of Agent-Based Models under intervention scenarios.

---

[SEED-SET: Scalable Evolving Experimental Design for System-Level Ethical Testing](http://arxiv.org/abs/2603.01630)

- SEED-SET: introduces a hierarchical Bayesian framework for sample-efficient ethical benchmarking of autonomous systems by integrating Objective GP, Subjective GP, and an LLM-based Proxy to guide test case generation.
- The framework utilizes a Hierarchical Variational Gaussian Process to decompose ethical evaluation into objective metrics and subjective stakeholder preferences, enabling interpretable and scalable testing.
- A novel data acquisition strategy within the Bayesian Experimental Design loop balances exploration of uncertain ethical factors with exploitation of learned preferences to maximize information gain.

---

#### 10th March 2026

[Dynamic Multimodal Expression Generation for LLM-Driven Pedagogical Agents: From User Experience Perspective](http://arxiv.org/abs/2603.09536)

- Dynamic Multimodal Expression Generation Framework: introduces a system that leverages semantically adaptive prompt construction to align LLM-generated dialogue with coordinated speech and gesture instructions for pedagogical agents.
- The framework utilizes a Text Parsing module to map LLM output tags into SSML-compliant speech and gesture commands, enabling real-time, context-aware multimodal interaction in virtual reality.
- Experimental results demonstrate that dynamic speech and gesture expressions significantly enhance learner engagement, perceived usefulness, and social presence compared to static agent behaviors.

---

[TA-Mem: Tool-Augmented Autonomous Memory Retrieval for LLM in Long-Term Conversational QA](http://arxiv.org/abs/2603.09297)

- TA-Mem: introduces a tool-augmented memory retrieval framework that utilizes an agentic memory constructor for structured episodic note generation and a retrieval agent for autonomous memory exploration.
- The framework employs a multi-indexed database that supports both key-based lookups and vector-space similarity searches to provide granular context for LLMs during long-term conversational QA.
- By integrating an agentic loop with a per-session memory cache, the system achieves high performance on long-range inference tasks while maintaining token efficiency and adaptability across diverse question types.

---

[From Days to Minutes: An Autonomous AI Agent Achieves Reliable Clinical Triage in Remote Patient Monitoring](http://arxiv.org/abs/2603.09052)

- Sentinel: introduces an autonomous AI agent that utilizes the Model Context Protocol to perform contextual clinical triage of remote patient monitoring vital signs by dynamically retrieving and synthesizing patient data.
- The system leverages an LLM equipped with 21 structured clinical tools to provide deep contextual interpretation, effectively filtering data floods and closing the loop from detection to clinical response.
- Retrospective evaluation demonstrates that the agent achieves high sensitivity for clinical deterioration, outperforming individual human clinicians and rule-based baselines while maintaining a clinically defensible overtriage profile.

---

[SBOMs into Agentic AIBOMs: Schema Extensions, Agentic Orchestration, and Reproducibility Evaluation](http://arxiv.org/abs/2603.10057)

- AIBOM: introduces a multi-agent framework that transforms static SBOMs into active provenance artefacts by integrating MCP, A2A, and AGNTCY agents for autonomous, policy-constrained reasoning.
- The framework leverages ISO/IEC 20153:2025 CSAF v2.0 standards to provide machine-verifiable, contextual exploitability assertions through the SACRO Advisory Normalisation Layer.
- Evaluation across heterogeneous workloads demonstrates that the agentic architecture improves runtime dependency fidelity and reproducibility compared to traditional provenance systems.

---


[SpecOps: A Fully Automated AI Agent Testing Framework in Real-World GUI Environments](http://arxiv.org/abs/2603.10268)

- SpecOps: introduces a specialist-agent architecture that decomposes end-to-end testing into four distinct phases—Test Generation, Environment Setup, Execution, and Validation—each managed by specialized agents to ensure task coherence and robustness.
- The framework utilizes a dual-specialist adaptive strategy for test generation and human-like visual monitoring via screen captures to detect failures across diverse real-world GUI environments.
- By employing specialized agents like the Test Architect, Infrastructure Manager, Engineer Specialist, Investigator Specialist, and Judge, SpecOps achieves high-precision bug detection while minimizing hallucinations and execution failures.

---

[DUCTILE: Agentic LLM Orchestration of Engineering Analysis in Product Development Practice](http://arxiv.org/abs/2603.10249)

- DUCTILE: introduces an agentic orchestration approach that separates adaptive LLM-based planning from deterministic execution performed by verified engineering tools.
- The framework utilizes an Inference Engine, Context Window, and Tool Execution components to bridge the gap between evolving engineering requirements and rigid, legacy analysis software.
- By leveraging in-context reasoning and external tool integration, the approach ensures traceability and human oversight in safety-critical aerospace engineering workflows.

---

[Over-the-Air Consensus-based Formation Control of Heterogeneous Agents: Communication-Rate and Geometry-Aware Convergence Guarantees](http://arxiv.org/abs/2603.10245)

- OTA Consensus-based Formation Control Framework: introduces a communication-efficient coordination scheme for heterogeneous agents that leverages the superposition property of wireless channels to compute normalized convex combinations of neighbor signals.
- The framework models the multi-agent system as a jump-flow system, where agents perform discrete-time reference updates via the Wireless Multiple Access Channel and continuous-time tracking between communication instants.
- The research derives sufficient conditions for formation convergence based on communication rates and geometric properties of the tracking transient, demonstrating significant reductions in required orthogonal transmissions compared to traditional node-to-node protocols.

---

[Learning from Radio using Variational Quantum RF Sensing](http://arxiv.org/abs/2603.10239)

- VQS (Variational Quantum Sensing): introduces a framework where an agent utilizes a quantum sensing probe, optimized via a variational quantum circuit, to learn from incident RF electromagnetic fields without requiring channel measurements at deployment.
- The framework employs a simulation-to-real transfer paradigm, using a ray-tracer to train the quantum circuit and a classical machine learning model on simulated propagation data.
- Experimental results on a localization task demonstrate that the approach achieves high prediction accuracy and rapid convergence, effectively utilizing radio channels as a source of environmental knowledge.

---

[Sabiá-4 Technical Report](http://arxiv.org/abs/2603.10213)

- Sabiá-4: introduces a new generation of Portuguese LLMs developed through a four-stage training pipeline comprising continued pre-training, long-context extension, supervised fine-tuning, and preference alignment.
- The models leverage domain-specialized continued pre-training on Portuguese and Brazilian legal corpora to achieve high performance in legal tasks and agentic workflows.
- Evaluation across six benchmark categories demonstrates that Sabiá-4 and Sabiazinho-4 provide a competitive cost-performance trade-off for production deployments.

---

[Octopus-inspired Distributed Control for Soft Robotic Arms: A Graph Neural Network–Based Attention Policy with Environmental Interaction](http://arxiv.org/abs/2603.10198)

- SoftGM: introduces a bio-inspired distributed control architecture for soft robotic arms that utilizes a GNN-based attention policy to manage complex environmental interactions through online obstacle discovery.
- The framework employs a two-stage graph attention mechanism to prioritize contact-relevant information while maintaining local-global consistency across the segmented robotic body.
- SoftGM adopts a Centralised Training Decentralised Execution (CTDE) paradigm, enabling robust coordination in contact-rich environments by suppressing irrelevant sensory data and focusing on dominant nodes.

---

[MCP-in-SoS: Risk assessment framework for open-source MCP servers](http://arxiv.org/abs/2603.10194)

- MCP-in-SoS (Model Context Protocol in System-of-Systems): introduces a four-stage automated pipeline that leverages CodeQL, Joern, and Cisco AI Defender MCP Scanner to identify and prioritize security weaknesses in open-source MCP servers using MITRE CWE and CAPEC metadata.
- The framework maps code-level vulnerabilities to four specific threat surfaces—Protocol, Resource, Tool, and Prompt—to quantify risk and reveal multi-stage exploit chains in LLM-agentic systems.
- Empirical analysis of 222 repositories demonstrates that 86% contain exploitable weaknesses, with Protocol-layer issues frequently acting as a reachability multiplier for more severe Tool and Resource attacks.

---

[Learning to Decode Quantum LDPC Codes Via Belief Propagation](http://arxiv.org/abs/2603.10192)

- RL-SVNS (Reinforcement Learning-based Sequential Variable Node Scheduling): introduces a reinforcement learning framework that optimizes the variable node update order in sequential belief propagation decoding for quantum LDPC codes.
- The framework utilizes a syndrome-driven state representation and a Q-learning agent to inject controlled asymmetry into the decoding process, effectively mitigating convergence failures caused by quantum degeneracy.
- To ensure practical inference speeds, the approach employs incremental local updates and a max-heap-based priority queue to maintain decoding states and scheduling priorities with minimal computational overhead.

---

[Video-Based Reward Modeling for Computer-Use Agents](http://arxiv.org/abs/2603.10178)

- ExeVRM: introduces a video-based reward modeling framework for CUAs that leverages ExeVR-53k, adversarial instruction translation, and spatiotemporal token pruning to enable scalable, model-agnostic evaluation of agent trajectories.
- The framework utilizes STP and TTP to reduce visual redundancy in high-resolution execution videos, allowing for efficient processing of long-horizon interactions while preserving decisive UI cues.
- Experimental results demonstrate that ExeVRM 8B achieves superior accuracy and recall compared to proprietary models like GPT-5.2 and Gemini-3 Pro, while providing more precise temporal attribution for failure localization.

---

[OpenClaw-RL: Train Any Agent Simply by Talking](http://arxiv.org/abs/2603.10165)

- OpenClaw-RL: introduces a unified, asynchronous framework that enables continuous online learning for diverse agents by recovering evaluative and directive signals from next-state interactions.
- The architecture utilizes Slime, Environment Servers, PRM Judge, Megatron, and SGLang to decouple policy serving, rollout collection, reward computation, and training into independent loops.
- It optimizes agents using Binary RL for scalar process rewards and Hindsight-Guided On-Policy Distillation for token-level directional advantage supervision, supporting both personal and general-purpose LLMs.

---

[Compatibility at a Cost: Systematic Discovery and Exploitation of MCP Clause-Compliance Vulnerabilities](http://arxiv.org/abs/2603.10163)

- MCP Analysis Framework: introduces a systematic approach to identify and exploit non-compliance vulnerabilities in MCP SDKs by normalizing code into a Universal IR Generator, performing Hybrid Static-LLM Analysis, and assessing risks through Modality-based Exploitation Analysis.
- The framework leverages a language-agnostic intermediate representation to enable scalable, auditable compliance checks across diverse MCP SDKs while avoiding pattern explosion.
- By categorizing missing protocol clauses into payload and timing attack modalities, the research identifies intrinsic security gaps arising from the tension between agent diversity and protocol standardization.

---

[Omics Data Discovery Agents](http://arxiv.org/abs/2603.10161)

- ODDA: introduces an agentic framework that automates the identification, extraction, and reanalysis of omics research data from biomedical literature using Article Parsing Agent, Knowledge Search Agent, Data Identification Agent, Study Integration Agent, Knowledge Database, Data Repository, and MCP Analysis Tools.
- The system leverages LLMs to parse unstructured text and supplemental files, enabling automated metadata extraction and the execution of containerized quantification pipelines via Model Context Protocol (MCP) servers.
- ODDA facilitates cross-study reasoning by identifying semantically similar datasets and performing comparative analyses, effectively transforming static biomedical publications into executable, queryable research objects.

---

[TOWARDS MACROECONOMIC ANALYSIS WITHOUT MICROFOUNDATIONS: MEASURING THE ENTROPY OF SIMULATED EXCHANGE ECONOMIES](http://arxiv.org/abs/2603.10155)

- TM: introduces a computational method to empirically measure the entropy function of simulated exchange economies using an economic analogue of calorimetry.
- The framework utilizes a CD Meter to sample macro-quantities across a state space, subsequently applying least-squares fitting to verify path-independence and concavity of the entropy function.
- This approach enables macroeconomic analysis of complex agent-based systems where traditional micro-foundational derivations are mathematically intractable.

---

[Agentic Control Center for Data Product Optimization](http://arxiv.org/abs/2603.10133)

- Agentic Control Center framework: introduces an autonomous system for data product optimization that utilizes a continuous improvement loop driven by specialized LLM agents to enhance data quality metrics against user-defined contracts.
- The architecture integrates a State Manager, a Data Product Quality Metrics module, a Tool Registry, and an Agentic Orchestration Layer to maintain observability and control over the data product lifecycle.
- The system employs a multi-agent workflow that includes a Planner Agent for strategic decision-making, an Input Planner Agent for tactical parameterization, and various Specialized Agents for executing data improvement tasks.

---

[Emotional Modulation in Swarm Decision Dynamics](http://arxiv.org/abs/2603.09963)

- Emotional Modulation in Swarm Decision Dynamics: introduces an agent-based model that extends the classical bee equation by incorporating valence-arousal emotional states to modulate recruitment and inhibition dynamics.
- The framework utilizes an emotional contagion module to update agent states during interactions, effectively shifting decision outcomes and convergence speeds based on affective influence.
- Experimental results demonstrate that emotional modulation and non-linear amplification mechanisms can bias collective choices and trigger rapid consensus even in initially symmetric conditions.

---

[Towards a Neural Debugger for Python](http://arxiv.org/abs/2603.09951)

- Neural Debugger: introduces a framework that models program execution as a Markov Decision Process, enabling LLMs to perform forward and inverse execution prediction conditioned on debugger actions.
- The approach utilizes a data pipeline to transform execution traces into structured state-action sequences, allowing models to emulate interactive debugging operations like stepping and setting breakpoints.
- Empirical results demonstrate that these models achieve high accuracy in state prediction and show strong transfer capabilities to downstream tasks such as input and output prediction on the CruxEval benchmark.

---

[Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with Large Language Models](http://arxiv.org/abs/2603.10098)

- CSRO (Code-Space Response Oracles): introduces a framework that replaces traditional deep reinforcement learning oracles with LLMs to synthesize interpretable, programmatic policies for multi-agent games.
- The framework utilizes an LLM-oracle to generate executable code, which is iteratively refined through feedback loops and evolutionary algorithms to achieve robust, game-theoretic equilibria.
- By employing context abstraction and programmatic policy generation, CSRO enables the creation of transparent, modular strategies that outperform opaque neural network baselines in complex multi-agent environments.

---

[Dynamic Average Consensus with Privacy Guarantees and Its Application to Battery Energy Storage Systems](http://arxiv.org/abs/2603.09904)

- DAC framework: introduces a privacy-preserving algorithm that utilizes a sinusoidal masking signal to conceal individual reference signals and their derivatives from external eavesdroppers while maintaining convergence properties.
- The approach employs a reference signal masking component to ensure that the global average is preserved while individual agent states remain non-identifiable to adversaries.
- The framework is applied to networked battery energy storage systems, integrating a distributed power allocation law to achieve state-of-charge balancing and total power tracking.

---

[Influencing LLM Multi-Agent Dialogue via Policy-Parameterized Prompts](http://arxiv.org/abs/2603.09890)

- Policy-Parameterized Prompts framework: introduces a lightweight policy mechanism that treats prompts as actions to systematically regulate LLM multi-agent dialogue without additional training.
- The framework decomposes prompts into five components (T, M, D, R, W) and utilizes an adaptive weight scheduler to dynamically modulate agent conversational behaviors based on temporal trends and feedback.
- Evaluation across diverse scenarios demonstrates that prompt parameterization effectively influences dialogue dynamics, including responsiveness, rebuttal, evidence usage, non-repetition, and stance shift.

---

[The Bureaucracy of Speed: Structural Equivalence Between Memory Consistency Models and Multi-Agent Authorization Revocation](http://arxiv.org/abs/2603.09875)

- CCS (Capability Coherence System): introduces a formal framework that maps multi-agent authorization revocation to cache coherence protocols in shared-memory multiprocessors to mitigate unauthorized operations.
- The framework utilizes an operation-bounded credential model (RCC) to enforce coherence at synchronization boundaries, effectively decoupling security from agent velocity.
- Simulation results demonstrate that RCC provides a deterministic, velocity-independent damage bound, significantly reducing unauthorized operations compared to traditional time-bounded TTL strategies.

---

[RecThinker: An Agentic Framework for Tool-Augmented Reasoning in Recommendation](http://arxiv.org/abs/2603.09843)

- RecThinker: introduces an agentic framework that shifts recommendation from passive processing to autonomous investigation by dynamically planning reasoning paths and proactively acquiring information via tool-use.
- The framework utilizes an Analyze-Plan-Act paradigm where the RecThinker Agent assesses information sufficiency and invokes specialized User-Side InfoTools, Item-Side InfoTools, and Collaborative InfoTools to bridge information gaps.
- RecThinker employs a two-stage training strategy, combining self-augmented SFT on high-quality trajectories with RL optimization via a Reward Controller to enhance reasoning accuracy and tool invocation efficiency.

---

[Chow–Liu Ordering for Long-Context Reasoning in Chain-of-Agents](http://arxiv.org/abs/2603.09835)

- Chow–Liu CoA: introduces a dependency-aware chunk ordering strategy for sequential long-context reasoning that utilizes Chow–Liu trees to minimize information loss during incremental memory construction.
- The framework models retrieved document chunks as random variables and employs a maximum-weight spanning tree to prioritize semantically related chunks in the processing sequence.
- By performing a breadth-first traversal on the dependency tree, the system reduces compression-induced errors and improves reasoning performance across long-context benchmarks.

---

[Execution Is the New Attack Surface: Survivability-Aware Agentic Crypto Trading with OpenClaw-Style Local Executors](http://arxiv.org/abs/2603.10092)

- SAE: introduces a survivability-focused execution-layer middleware that intercepts and constrains untrusted LLM intents before they reach the exchange executor, utilizing Strategy Engine, ExecutionRequest, SAE Middleware, Exchange Executor, Regime Detector, Trader-State Service, Trust State, Policy Engine, and Enforcement Runtime.
- The framework enforces non-bypassable invariants through projection-based exposure budgeting, temporal guards, and trust-conditioned tightening to mitigate risks from prompt injection and supply-chain compromises.
- Empirical evaluation on Binance perpetual futures data demonstrates that SAE significantly reduces tail-risk and delegation-gap harm compared to unprotected agentic trading baselines.

---

[One-Eval: An Agentic System for Automated and Traceable LLM Evaluation](http://arxiv.org/abs/2603.09821)

- One-Eval: introduces an agentic evaluation system that transforms natural language requests into executable, traceable, and customizable evaluation workflows using NL2Bench, BenchResolve, and Metrics & Reporting.
- The framework utilizes a modular, three-stage pipeline that includes intent interpretation, automated benchmark resolution, and task-oriented report generation to support decision-making for LLMs.
- One-Eval incorporates a human-in-the-loop mechanism at key decision points to ensure reliability while maintaining automation efficiency across diverse evaluation tasks.

---

[MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations](http://arxiv.org/abs/2603.09800)

- MITRA: introduces a privacy-first RAG system that utilizes Selenium, OCR, DPR, Chroma DB, Cross-encoder, Mistral-7B, Ollama, LangChain, and Streamlit to provide context-aware answers for physics collaborations.
- The framework employs a two-tiered vector database architecture to isolate analysis-specific context, ensuring accurate retrieval and preventing information conflation between different research projects.
- By hosting all components on-premise, the system eliminates per-token costs and maintains the security of proprietary research data within the collaboration's internal network.

---

[PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](http://arxiv.org/abs/2603.09760)

- PanoAffordanceNet: introduces an end-to-end framework for holistic affordance grounding in 360° indoor environments by integrating a DINOv2 Vision Encoder, CLIP Text Encoder, LoRA, DASM, HFEM, LFSM, Transformer Decoder, and OSDH.
- The framework utilizes a Distortion-Aware Spectral Modulator to mitigate geometric distortions and an Omni-Spherical Densification Head to restore topological continuity from sparse activations.
- PanoAffordanceNet is evaluated on the newly constructed 360-AGD dataset, demonstrating superior performance in grounding interactive regions within complex panoramic scenes compared to existing object-centric methods.

---

[Beyond Fine-Tuning: Robust Food Entity Linking under Ontology Drift with FoodOntoRAG](http://arxiv.org/abs/2603.09758)

- FoodOntoRAG: introduces a model- and ontology-agnostic pipeline for food named entity linking that performs few-shot retrieval-augmented generation to mitigate ontology drift without task-specific fine-tuning.
- The pipeline integrates a Hybrid Retriever, a Selector Agent, an LLM Scorer Agent, and a Synonym Generator Agent to ground entity linking decisions in structured ontology evidence.
- The system achieves competitive accuracy and provides interpretable justifications while enabling robust performance across evolving food ontologies and diverse product datasets.

---

[Epistemic Closure: Autonomous Mechanism Completion for Physically Consistent Simulation](http://arxiv.org/abs/2603.09756)

- Neuro-Symbolic Generative Agent: introduces a framework that elevates LLMs from code translators to reasoning kernels by encapsulating physical laws into modular Constitutive Skills to autonomously validate and complete simulation mechanisms.
- The framework employs a Chain-of-Thought workflow to perform Deductive Pruning of redundant mechanisms and Inductive Completion of missing physics based on Dimensionless Scaling Analysis.
- By leveraging intrinsic physical priors, the agent prevents Physical Hallucination in multi-physics simulations, ensuring qualitative consistency where standard literature-retrieval baselines fail.

---

[Let’s Reward Step-by-Step: Step-Aware Contrastive Alignment for Vision-Language Navigation in Continuous Environments](http://arxiv.org/abs/2603.09740)

- SACA (Step-Aware Contrastive Alignment): introduces a reinforcement fine-tuning framework that extracts dense, step-level supervision from imperfect trajectories to overcome learning-signal collapse in sparse-reward VLN-CE tasks.
- The framework utilizes a PGSA Auditor to generate continuous soft scores and discrete structural masks, enabling the identification of valid prefixes and divergence points for targeted optimization.
- SACA dynamically routes trajectory batches to either Repair Resampling or All-Failure Rescue mechanisms, ensuring stable gradient updates and superior generalization in long-horizon navigation.

---

[FetalAgents: A Multi-Agent System for Fetal Ultrasound Image and Video Analysis](http://arxiv.org/abs/2603.09733)

- FetalAgents: introduces a multi-agent system that orchestrates specialized vision experts through a language-driven framework to automate fetal ultrasound analysis and reporting.
- The system utilizes a Coordinator Agent, Expert Agents, and a Summarizer Agent to perform end-to-end video stream summarization and clinical report generation.
- FetalAgents integrates diverse vision backbones and LLMs to minimize hallucination risks while providing auditable, workflow-aligned clinical outputs.

---

[EXPLORE-Bench: Egocentric Scene Prediction with Long-Horizon Reasoning](http://arxiv.org/abs/2603.09731)

- EXPLORE-Bench: introduces a benchmark for evaluating MLLMs on egocentric scene prediction with long-horizon reasoning, utilizing RAM++, spaCy, Grounding DINO, Qwen3-VL-235B-A22B-Instruct, GPT-5.2, Human Annotator, Sentence-BERT, and an LLM-based Scorer to provide fine-grained, quantitative assessment of final-scene predictions.
- The framework employs a multi-step annotation pipeline that integrates automated object detection and attribute generation with human-in-the-loop quality control to ensure accurate, structured scene annotations.
- The benchmark evaluates LLMs on their ability to anticipate physical consequences of long action sequences, revealing significant performance gaps compared to humans, particularly in abnormal scenarios.

---

[AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents](http://arxiv.org/abs/2603.09716)

- AutoAgent: introduces a self-evolving multi-agent framework that integrates Agent Cognition, Contextual Decision Engine, Elastic Memory Orchestrator, and Cognitive Evolution Module to enable continuous adaptation without external retraining.
- The framework utilizes a closed-loop system where the Agent Cognition provides structured knowledge, the Contextual Decision Engine selects actions, the Elastic Memory Orchestrator manages history, and the Cognitive Evolution Module refines knowledge based on outcomes.
- AutoAgent improves task success and efficiency by replacing static prompts and rigid workflows with dynamic, practice-driven cognition and elastic memory management.

---

[OOD-MMSafe: Advancing MLLM Safety from Harmful Intent to Hidden Consequences](http://arxiv.org/abs/2603.09706)

- CASPO (Consequence-Aware Safety Policy Optimization): introduces a consequence-driven safety paradigm for MLLMs that shifts focus from intent detection to causal projection using OOD-MMSafe Benchmark, Curation Pipeline, Tripartite Evaluation System, CASPO Algorithm, Token-level Self-distillation, Outcome-driven Rewards, Constitution Correction Factor, and Hybrid Advantage Estimator.
- The framework addresses causal blindness in MLLMs by integrating token-level self-distillation with global outcome rewards to internalize hazard awareness and transcend performance ceilings.
- Experimental results demonstrate that CASPO significantly reduces risk identification failure rates while maintaining generative effectiveness across diverse safety domains.

---

[An Empirical Study of Interaction Smells in Multi-Turn Human-LLM Collaborative Code Generation](http://arxiv.org/abs/2603.09701)

- InCE (Invariant-aware Constraint Evolution): introduces a multi-agent framework that mitigates Interaction Smells by utilizing an IEM (extracts and persists global constraints) and a PSD (audits quality before generation) to maintain contextual consistency in multi-turn human-LLM coding.
- The framework employs a User Simulator (mimics human follow-up instructions) and an Evaluation Oracle (scores output for success criteria) to systematically evaluate and improve LLM performance in iterative programming tasks.
- The study establishes a taxonomy of Interaction Smells, identifying Must-Do Omission and Partial Functionality Breakdown as the most prevalent issues across mainstream LLMs.

---

[PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories](http://arxiv.org/abs/2603.09641)

- PRECEPT: introduces a unified framework for test-time adaptation that integrates Task Orchestration, COMPASS Hi-Freq Monitor, Trigger Events, Retrieval & Decision, Learning & Adaptation, MCP Tool Gateway, Knowledge & Conflict, COMPASS Lo-Freq Architect, Domain Executors, and Persistent Stores to enable reliable, compositional, and drift-resilient LLM agent behavior.
- The framework utilizes deterministic O(1) exact-match retrieval and a semantic tier hierarchy to eliminate interpretation errors and support compositional generalization across complex task spaces.
- PRECEPT employs a dual-frequency control loop where the high-frequency monitor ensures real-time constraint enforcement, while the low-frequency architect performs event-triggered prompt evolution and Pareto-optimal strategy selection.

---

[Preparing Students for AI-Driven Agile Development: A Project-Based AI Engineering Curriculum](http://arxiv.org/abs/2603.09599)

- AI Engineering Curriculum: introduces a project-based educational framework that integrates agile practices with generative AI tools to prepare students for modern software development through Core Curriculum, Specialization Tracks, Project Classes, Schools, Oral Exams, Project Repository, and Presentation.
- The framework utilizes a project-based backbone where students apply agile methods and AI tools end-to-end on realistic problems to foster critical AI literacy and hands-on competence.
- The curriculum emphasizes human responsibility and quality assurance by requiring individual oral examinations to verify foundational learning despite the use of AI-assisted coding.

---

[Ignorance with(out) Grasping](http://arxiv.org/abs/2603.09569)

- HIW, HIU, and HDI: introduce a hyperintensional approach to modeling ignorance by augmenting standard Kripke semantics with topic-sensitive models that account for an agent's capacity to grasp proposition content.
- The framework utilizes a topic-sensitive semantics to differentiate between logically equivalent propositions based on whether an agent grasps their underlying topics.
- By incorporating grasping conditions into the formal definitions of ignorance, the systems effectively address and avoid standard forms of logical omniscience.

---

[Memory-Guided View Refinement for Dynamic Human-in-the-loop EQA](http://arxiv.org/abs/2603.09541)

- DIVRR: introduces a training-free framework for Embodied Question Answering that couples Target-Region Reasoning, Relevance-guided View Refinement, Multi-view Augmentation, Relevance-driven Memory Admission, Long-term Memory, and Action and Answer Generation to manage evidence in dynamic environments.
- The framework utilizes a VLM to compute relevance scores for observations, triggering multi-view augmentation to resolve occlusions and ensure only verified, informative evidence is admitted into long-term memory.
- By maintaining a compact memory through selective admission, DIVRR improves robustness and inference efficiency in human-populated scenes compared to traditional store-then-retrieve pipelines.

---

[Sampling Logit Equilibrium and Endogenous Payoff Distortion](http://arxiv.org/abs/2603.09539)

- SLE (Sampling Logit Equilibrium): introduces a stationary concept for population games where agents evaluate actions using finite samples and respond via a logit choice rule, incorporating Sampling logit choice rule, Finite sampling, Logit stochastic choice, Virtual payoff, Variance premium, and Curvature premium.
- The framework demonstrates that finite sampling systematically distorts incentives, which can be represented as a virtual game with modified payoffs consisting of variance and curvature premiums.
- The model provides a tractable approximation for analyzing equilibrium behavior in large populations, showing how sampling noise influences equilibrium selection and shifts incentives.

---

[Telogenesis: Goal Is All U Need](http://arxiv.org/abs/2603.09476)

- Telogenesis: introduces a framework where attentional priorities emerge endogenously from epistemic gaps, utilizing Ignorance, Surprise, and Staleness to drive observation allocation.
- The framework demonstrates that priority-guided allocation outperforms coverage-based strategies in change detection latency, with advantages scaling monotonically with environmental complexity.
- By employing learnable decay rates, the system spontaneously recovers latent environmental volatility structures without external supervision or reward signals.

---

[A Guideline-Aware AI Agent for Zero-Shot Target Volume Auto-Delineation](http://arxiv.org/abs/2603.09448)

- OncoAgent: introduces a guideline-aware AI agent framework that converts textual clinical guidelines into 3D target volumes using an LLM-based Agent, System Prompt, Delineation Plan, Plan Execution, OAR Segmentation Models, and Geometric Operation Tools.
- The framework utilizes a two-phase architecture of planning and execution to perform zero-shot target volume auto-delineation without requiring expert-annotated training data.
- OncoAgent achieves performance comparable to supervised baselines while providing superior guideline compliance, interpretability, and instant adaptability to evolving clinical protocols.

---

[Open-World Motion Forecasting](http://arxiv.org/abs/2603.09420)

- OMEN: introduces an end-to-end class-incremental framework for motion forecasting that mitigates catastrophic forgetting by leveraging a VLM-guided pseudo-labeling strategy and a variance-based experience replay mechanism.
- The framework utilizes an Encoder, Detection Decoder, and Motion Decoder to predict trajectories directly from raw camera inputs while incrementally integrating new semantic classes.
- OMEN employs a VLM to filter inconsistent pseudo-labels and a sequence-based replay buffer that prioritizes samples with high latent query variance to maintain performance on previously learned classes.

---

[Reward Prediction with Factorized World States](http://arxiv.org/abs/2603.09400)

- StateFactory: introduces a semantic factorization framework that transforms unstructured observations into hierarchical object-attribute structures to enable accurate, zero-shot reward prediction across diverse domains.
- The framework utilizes recurrent state extraction and dynamic goal interpretation to derive dense reward signals via hierarchical semantic alignment, effectively mitigating noise and improving planning performance.
- StateFactory outperforms supervised and representation-free baselines on the RewardPrediction benchmark, demonstrating robust generalization and enhanced agent planning capabilities in complex, multi-step environments.

---

[Unintended Consequences: Updating Causal Models](http://arxiv.org/abs/2603.09387)

- Unintended Consequences: Updating Causal Models framework: introduces a formal decision-theoretic model where agents update causal beliefs through interventions and observations, utilizing Structural-Equations Models, a Conditional Probability System (CPS), Agent Agency and Utility, and an Introspective Unawareness Module.
- The framework defines a steady state where an agent's optimal action, based on current beliefs, generates feedback that does not trigger further belief revision, potentially leading to persistent suboptimal behavior.
- The research extends the model to include introspective unawareness, allowing agents to account for the possibility of encountering inexplicable evidence and the potential value of paradigm shifts.

---

[ProvAgent: Threat Detection Based on Identity-Behavior Binding and Multi-Agent Collaborative Attack Investigation](http://arxiv.org/abs/2603.09358)

- ProvAgent: introduces a threat investigation framework that synergizes an EPD module for efficient anomaly screening with an MAI module for autonomous, hypothesis-driven attack reconstruction.
- The EPD module leverages GNNs and graph contrastive learning to enforce fine-grained identity-behavior consistency, generating high-fidelity alerts by identifying deviations from learned benign profiles.
- The MAI module employs a collaborative multi-agent architecture, including Analyst-, Investigator-, Leader-, and Reporter-agents, to perform iterative hypothesis-verification and synthesize comprehensive attack narratives from initial alerts.

---

[Reward-Zero: Language Embedding Driven Implicit Reward Mechanisms for Reinforcement Learning](http://arxiv.org/abs/2603.09331)

- Reward-Zero: introduces an implicit reward mechanism that derives dense progress signals from natural language goal descriptions using pre-trained vision-language embeddings to guide RL agents.
- The framework utilizes a CLIP-based potential function combined with a progress-aware activation mechanism to provide continuous, semantically grounded feedback without requiring manual reward engineering.
- By integrating Reward-Zero as an auxiliary signal into PPO, the approach accelerates convergence, stabilizes training dynamics, and improves success rates across diverse robotic manipulation and locomotion tasks.

---

[Reading the Mood Behind Words: Integrating Prosody-Derived Emotional Context into Socially Responsive VR Agents](http://arxiv.org/abs/2603.09324)

- Emotion-aware VR interaction pipeline: integrates real-time speech emotion recognition with LLM-based response generation to treat vocal prosody as explicit dialogue context.
- The system utilizes a HuBERT-based SER module to inject emotion labels into the LLM prompt, ensuring affective congruence in conversational responses.
- Empirical results demonstrate that this prosody-aware approach significantly improves rapport, engagement, and human-likeness compared to text-only interaction paradigms.

---

[ToolRosetta: Bridging Open-Source Repositories and Large Language Model Agents through Automated Tool Standardization](http://arxiv.org/abs/2603.09290)

- ToolRosetta: introduces a unified framework that automatically translates heterogeneous open-source code repositories into standardized Model Context Protocol (MCP) services for reliable invocation by LLMs.
- The system utilizes a hierarchical multi-agent architecture, including Tool-search-, MCP-construction-, Planning-, Security-, and Review-agents, to automate the end-to-end pipeline from repository discovery to executable tool deployment.
- ToolRosetta incorporates a security inspection layer and an iterative Review-Revise-Fix (RRF) mechanism to ensure the robustness, safety, and reliability of standardized tools across diverse scientific domains.

---

[Implicit Geometry Representations for Vision-and-Language Navigation from Web Videos](http://arxiv.org/abs/2603.09259)

- RoomTour3D-IGR: introduces a large-scale dataset and navigation framework that leverages web-based room tour videos to train embodied agents using both description-enriched and action-enriched trajectories.
- The framework incorporates implicit geometry representations, which extract spatial cues directly from RGB frames via a Spatial Encoder, bypassing the need for fragile explicit 3D reconstruction.
- By integrating these implicit spatial embeddings with a pretrained LLM, the approach improves data utilization and enhances the robustness of navigation agents across diverse real-world environments.

---

[RAE-NWM: Navigation World Model in Dense Visual Representation Space](http://arxiv.org/abs/2603.09241)

- RAE-NWM: introduces a navigation world model that operates within a dense visual representation space to preserve geometric structure and improve long-horizon prediction stability.
- The framework utilizes a DINOv2 encoder for feature extraction and a CDiT-DH generative backbone to model continuous action-conditioned transitions.
- A time-driven gating mechanism within the dynamics conditioning module adaptively modulates kinematic priors to balance global geometric consistency with fine-grained visual details.

---

[Abundant Intelligence and Deficient Demand: A Macro-Financial Stress Test of Rapid AI Adoption](http://arxiv.org/abs/2603.09209)

- Macro-Financial Stress Test Framework: introduces a formal model connecting AI-driven micro-level productivity gains to macro-financial fragility through displacement, demand-side feedback, and institutional mismatch.
- The framework identifies three primary mechanisms—displacement spiral, Ghost GDP, and intermediation collapse—that can trigger a macroeconomic contraction when AI adoption outpaces institutional adaptation.
- The research demonstrates that the depth of the resulting crisis is a policy variable, where rapid fiscal redistribution can mitigate the demand-side shocks caused by structural labor displacement.

---

[Strategically Robust Multi-Agent Reinforcement Learning with Linear Function Approximation](http://arxiv.org/abs/2603.09208)

- RQRE-OVI: introduces a scalable, robust multi-agent reinforcement learning algorithm that computes Risk-Sensitive Quantal Response Equilibria using linear function approximation.
- The framework utilizes environment-risk and policy-risk operators to provide a principled tradeoff between expected performance and robustness against modeling errors and opponent deviations.
- Theoretical analysis establishes finite-sample regret guarantees, while empirical evaluations demonstrate superior robustness in cross-play scenarios compared to Nash-based approaches.

---

[Evaluate-as-Action: Self-Evaluated Process Rewards for Retrieval-Augmented Agents](http://arxiv.org/abs/2603.09203)

- EVALACT (Evaluate-as-Action): introduces a reinforcement learning framework that transforms implicit retrieval quality assessment into an explicit, policy-selectable action to improve multi-hop reasoning.
- The framework enforces a coupled Search-to-Evaluate protocol, where each retrieval action is immediately followed by an Evaluate action that produces a structured confidence score.
- PCAR (Process-Calibrated Advantage Rescaling) leverages these step-wise evaluation scores to perform segment-level advantage rescaling, enabling finer-grained credit assignment and stabilizing learning for LLMs.

---

[Explainable Innovation Engine: Dual-Tree Agent-RAG with Methods-as-Nodes and Verifiable Write-Back](http://arxiv.org/abs/2603.09192)

- Dual-Tree Agent-RAG (Explainable Innovation Engine): introduces a RAG framework that upgrades knowledge units from text chunks to methods-as-nodes, utilizing a provenance tree for traceable derivations and an abstraction tree for hierarchical navigation.
- The framework employs a strategy agent to compose new method nodes using explicit synthesis operators, followed by a verifier-scorer layer that prunes low-quality candidates and writes validated nodes back to support continual growth.
- By integrating structured retrieval, strategy-driven synthesis, and formal verification, the system enables controllable, explainable, and verifiable innovation in agentic RAG pipelines.

---

[Robust Spatiotemporal Motion Planning for Multi-Agent Autonomous Racing via Topological Gap Identification and Accelerated MPC](http://arxiv.org/abs/2603.09188)

- Topo-Gap (Topological Gap Identification Planner): introduces a hierarchical motion planning framework that utilizes Parallel SGPs to predict opponent behavior, a Topology-aware module to select optimal gaps, and a PTC-accelerated MPC to ensure kinematic feasibility.
- The framework employs a hysteresis-based cost function to maintain decision stability and a PTC-accelerated solver to reduce computational latency in multi-agent racing scenarios.
- Experimental results on the F1TENTH platform demonstrate that the approach significantly reduces maneuver time and maintains high overtaking success rates in dense traffic compared to existing baselines.

---

[LATENT-DARM: BRIDGING DISCRETE DIFFUSION AND AUTOREGRESSIVE MODELS FOR REASONING](http://arxiv.org/abs/2603.09184)

- Latent-DARM: introduces a latent-space communication framework that bridges DDLM (planners) and ARM (executors) to maximize collaborative reasoning benefits.
- The framework utilizes a learned Projection Layer to map latent representations from the DDLM space directly into the ARM embedding space, bypassing explicit text generation.
- By enabling latent-space exchange, the system improves planning fidelity and reasoning accuracy while significantly reducing the required token budget compared to traditional text-based multi-agent systems.

---

[Improving Search Agent with One Line of Code](http://arxiv.org/abs/2603.10069)

- SAPO (Search Agent Policy Optimization): introduces a training stabilization method for search agents that mitigates Importance Sampling Distribution Drift (ISDD) by applying a conditional KL penalty to positive tokens with low probabilities.
- The framework integrates a token-level constraint into the GRPO optimizer, which selectively penalizes policy divergence only when the current policy shifts excessively from the old policy on positive-advantage tokens.
- SAPO requires only a single-line code modification to standard GRPO, demonstrating consistent performance improvements across various LLM scales and families on complex multi-hop QA benchmarks.

---

[Real-Time Trust Verification for Safe Agentic Actions using TrustBench](http://arxiv.org/abs/2603.09157)

- TrustBench: introduces a dual-mode framework that enables real-time trust verification for LLMs by intercepting agent actions before execution to prevent harm.
- The framework utilizes LLM-as-a-Judge evaluation and isotonic regression to calibrate agent confidence, ensuring that self-reported reliability aligns with actual reasoning quality.
- TrustBench integrates domain-specific plugins to enforce contextual safety requirements, achieving significant harm reduction with sub-200ms latency in agentic workflows.

---

[DataFactory: Collaborative Multi-Agent Framework for Advanced Table Question Answering](http://arxiv.org/abs/2603.09152)

- DataFactory: introduces a multi-agent framework that addresses TableQA limitations through specialized team coordination and automated data-to-knowledge graph transformation.
- The framework utilizes a Data Leader employing the ReAct paradigm to orchestrate collaboration between a Database Team and a Knowledge Graph Team for complex multi-hop reasoning.
- DataFactory integrates context-engineered prompting and automated knowledge graph construction to reduce hallucinations and improve query accuracy across structured and relational data modalities.

---

[Deep Tabular Research via Continual Experience-Driven Execution](http://arxiv.org/abs/2603.09151)

- DTR (Deep Tabular Research): introduces a closed-loop agentic framework that treats long-horizon tabular reasoning as an iterative decision-making process, utilizing Meta Operation Decomposition, Macro Path Planner, Execution Experience Memory, Siamese Structured Memory, Seed Operation Bank, and Expectation-Aware Selection Policy.
- The framework decouples high-level strategic planning from low-level execution, enabling LLMs to navigate unstructured tables through programmatic operations and continual experience-driven refinement.
- DTR improves reasoning accuracy and efficiency by selecting optimal execution paths based on historical feedback, effectively mitigating error propagation in complex multi-hop analytical tasks.

---

[Critical States Preparation With Deep Reinforcement Learning](http://arxiv.org/abs/2603.09135)

- DRL-framework: introduces a deep reinforcement learning approach to optimize time-dependent control Hamiltonians for the rapid preparation of quantum critical states in light-matter interaction systems.
- The framework utilizes a DRL-Policy to interact with a Quantum System, employing a Reward Function based on state fidelity to identify efficient control protocols that circumvent analytical limitations.
- The protocol demonstrates robustness against systematic errors and environmental dissipation, validated through Quantum Fisher Information analysis to confirm the criticality of the prepared states.

---

[AgenticCyOps: Securing Multi-Agentic AI Integration in Enterprise Cyber Operations](http://arxiv.org/abs/2603.09134)

- AgenticCyOps: introduces a security framework for multi-agent systems that decomposes attack surfaces into tool orchestration and memory management, establishing primary trust boundaries to mitigate vulnerabilities.
- The framework implements five defensive principles—authorized interfaces, capability scoping, verified execution, memory integrity & synchronization, and access-controlled data isolation—to secure agentic workflows.
- Applied to a Security Operations Center architecture, the design utilizes phase-scoped agents and consensus-based validation to reduce exploitable trust boundaries by at least 72% compared to flat multi-agent systems.

---

[Chaotic Dynamics in Multi-LLM Deliberation](http://arxiv.org/abs/2603.09127)

- Multi-LLM Deliberation Framework: models five-agent LLM committees as random dynamical systems to quantify trajectory divergence using an empirical Lyapunov exponent.
- The research identifies two primary routes to instability in collective AI deliberation: institutional role differentiation and compositional model heterogeneity.
- Experimental results demonstrate that instability is design-induced and can be mitigated through targeted interventions like reducing memory windows or ablating specific roles.

---

[Adaptive Polyak Stepsize with Level-value Adjustment for Distributed Optimization](http://arxiv.org/abs/2603.09097)

- DPS-LA (Distributed Adaptive Polyak Stepsize algorithm with Level-value Adjustment): introduces a distributed optimization framework that enables agents to perform adaptive Polyak stepsize updates without prior knowledge of the global optimum by utilizing a Multi-agent network, Local objective functions, Consensus protocol, Level-value adjustment mechanism, Linear feasibility solver, and Decaying stepsize mechanism.
- The framework employs a local level-value adjustment mechanism that functions as an online cutting-plane algorithm to dynamically estimate the global optimal value through local linear feasibility problems.
- Theoretical analysis confirms that the algorithm achieves a sublinear convergence rate of O(1/√nT), demonstrating a linear speedup with respect to the number of agents.

---

[Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms](http://arxiv.org/abs/2603.09090)

- Feasibility Classification: introduces a method to mitigate valid action suppression in reinforcement learning by training an encoder to predict action validity, thereby inducing validity-discriminating representations.
- The framework utilizes a classification head on top of a shared encoder to learn validity-discriminating features, enabling deployment without oracle masks by substituting them with learned predictors.
- The approach employs a KL-balanced classification loss that weights training examples based on their impact on policy behavior, effectively focusing representation learning on critical, rarely-valid actions.

---

[EPOCH: An Agentic Protocol for Multi-Round System Optimization](http://arxiv.org/abs/2603.09049)

- EPOCH: introduces an engineering protocol for multi-round system optimization that organizes processes into baseline construction and iterative self-improvement phases.
- The framework structures optimization into role-constrained stages including Seed Planner, Baseline Executor, Orchestrator, Investigator, Executor, and Reviewer to ensure reproducibility and traceability.
- EPOCH standardizes execution through canonical command interfaces and round-level tracking to support heterogeneous tasks like prompt tuning, hyperparameter tuning, rule-based optimization, and code improvement.

---

[FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation](http://arxiv.org/abs/2603.09046)

- FlexServe: introduces a fast and secure LLM serving system for mobile devices that utilizes Flex-Monitor, Flex-Mem, Flex-NPU, FlexServe Framework, LLM-Aware Memory Management, Secure Inference Pipeline, Multi-Model Scheduler, and On-demand Protection to overcome the performance limitations of traditional hardware-based isolation.
- The system leverages virtualization to provide page-granular secure memory and a switchable secure NPU, enabling efficient LLM inference within the ARM TrustZone secure world.
- FlexServe achieves significant speedups in Time to First Token (TTFT) compared to existing TrustZone-based strawman designs while maintaining strong security guarantees against a compromised OS kernel.

---

[Time, Identity and Consciousness in Language Model Agents](http://arxiv.org/abs/2603.09043)

- LMA (Language Model Agent) Identity Framework: introduces a formal model to distinguish between ingredient-wise occurrence and co-instantiation of identity constraints within LLM-based agent scaffolds.
- The framework utilizes Stack Theory’s temporal semantics to demonstrate that agents can exhibit stable narrative self-reports while failing to maintain operative identity at decision time.
- It provides a conservative toolkit for identity evaluation, including persistence scores and an identity morphospace, to identify structural tradeoffs and failure modes in autonomous agent architectures.

---

[Quality-Driven Agentic Reasoning for LLM-Assisted Software Design: Questions-of-Thoughts (QoT) as a Time-Series Self-QA Chain](http://arxiv.org/abs/2603.11082)

- QoT (Questions-of-Thoughts): introduces a quality-driven inference-time reasoning scaffold that decomposes software tasks into a Sequential Process Chain, a Question-Answer Chain, and a Reasoning Knowledge Base to improve code reliability.
- The framework utilizes a structured self-questioning mechanism at each step to elicit constraints, verify logic, and maintain a traceable reasoning record for LLMs.
- Empirical evaluation across API design, data communication, and file systems demonstrates that QoT improves modularity, security, and completeness, particularly for smaller LLMs.

---

[SELF-VLA: A Skill Enhanced Agentic Vision-Language-Action Framework for Contact-Rich Disassembly](http://arxiv.org/abs/2603.11080)

- SELF-VLA: introduces an agentic framework that integrates structured disassembly skills with failure recovery to improve performance in long-horizon, contact-rich robotic tasks.
- The framework utilizes a VLA-planner to generate control actions and trigger specific skills from a library, while a VLA-corrector handles recovery if the failure detector identifies unsuccessful grasps or component loss.
- Experimental results demonstrate that SELF-VLA significantly outperforms end-to-end VLA models on complex disassembly tasks by enforcing structured execution and providing robust error handling.

---

[CR-Bench: Evaluating the Real-World Utility of AI Code Review Agents](http://arxiv.org/abs/2603.11078)

- CR-Bench: introduces a benchmarking dataset and evaluation pipeline designed to assess the real-world utility of AI code review agents by focusing on objective defect detection rather than subjective stylistic preferences.
- The framework utilizes CR-Evaluator to categorize agent outputs into bug hits, valid suggestions, or noise, enabling the calculation of precision, recall, usefulness rate, and signal-to-noise ratio.
- Experimental results demonstrate a fundamental trade-off where Reflexion-based agents improve recall for complex defects but often suffer from decreased signal-to-noise ratios compared to single-shot LLM agents.

---

[DIVE: Scaling Diversity in Agentic Task Synthesis for Generalizable Tool Use](http://arxiv.org/abs/2603.11076)

- DIVE: introduces an evidence-driven synthesis framework that inverts the task generation process by executing real-world tools first to derive grounded, verifiable, and executable tasks for training LLMs.
- The framework utilizes decoupled resource pools—tools, seeds, and exemplars—to scale structural diversity and induce complex, multi-step tool-use patterns.
- Experimental results demonstrate that diversity scaling consistently outperforms quantity scaling for OOD generalization, with RL further amplifying the diversity of learned tool-use structures.

---

[Context Before Code: An Experience Report on Vibe Coding in Practice](http://arxiv.org/abs/2603.11073)

- Vibe Coding Framework: introduces an experience report on applying conversational AI-assisted development to build production-oriented systems under explicit architectural constraints.
- The study evaluates the integration of LLMs into software workflows, highlighting that while AI accelerates scaffolding, it often fails to preserve critical architectural properties like tenant isolation and asynchronous processing.
- The research identifies a shift in engineering effort from routine boilerplate implementation toward manual architectural specification, constraint enforcement, and rigorous validation.

---

[From Phase Prediction to Phase Design: A ReAct Agent Framework for High-Entropy Alloy Discovery](http://arxiv.org/abs/2603.11068)

- ReAct Agent Framework: introduces an agentic framework that utilizes an LLM operating under the ReAct paradigm to autonomously propose, validate, and refine high-entropy alloy compositions by querying a calibrated XGBoost surrogate.
- The framework integrates a ReAct Agent with specialized tools for composition validation, phase prediction, and Bayesian optimization to navigate high-dimensional alloy design spaces.
- By leveraging domain-knowledge priors in the system prompt, the agent performs manifold-aware search, effectively balancing compositional diversity with experimental relevance.

---

[Governance Architecture for Autonomous Agent Systems: Threats, Framework, and Engineering Practice](http://arxiv.org/abs/2603.07191)

- LGA (Layered Governance Architecture): introduces a four-layer defense-in-depth framework to mitigate execution-layer vulnerabilities in autonomous agent systems by enforcing governance boundaries at the OS, intent, protocol, and logging levels.
- The framework utilizes L1 Execution Sandbox, L2 Intent Verification, L3 Zero-Trust Inter-Agent Protocol, and L4 Immutable Audit Log to systematically intercept threats like prompt injection, RAG poisoning, and malicious skill plugins.
- Experimental evaluation demonstrates that the architecture achieves high interception rates with minimal latency overhead, effectively shifting engineering focus from defect remediation to system-level governance.

---

#### 9th March 2026



[What Do AI Agents Talk About? Emergent Communication Structure in the First AI-Only Social Network](http://arxiv.org/abs/2603.07880)

- Moltbook: introduces a large-scale structural analysis of discourse within an AI-only social network, utilizing Data Collection, Preprocessing, Language Detection, Topic Modeling, Emotion Classification, Lexical Analysis, and Semantic Alignment.
- The study characterizes AI agent communities as structurally distinct discourse systems defined by introspective content, ritualistic interaction, and emotionally redirective communication patterns.
- Findings indicate that while AI agents exhibit organized thematic domains, conversational coherence decays rapidly with thread depth, revealing a system driven by amplification rather than substantive exchange.

---


[Bilevel Planning with Learned Symbolic Abstractions from Interaction Data](http://arxiv.org/abs/2603.08599)

- Bilevel Planning Framework: introduces a neuro-symbolic architecture for robotic manipulation that integrates high-level probabilistic symbolic reasoning with low-level continuous verification and fallback search, with all Symbolic Model, Dynamics Model, Probabilistic Symbolic Planner, Continuous Plan Verifier, AI Planner, and Continuous Forward Search components.
- The framework employs a dual-model scheme featuring a Symbolic Model for discovering discrete representations and a Dynamics Model for continuous effect prediction.
- The system utilizes a probabilistic symbolic planner to generate candidate plans and a verifier to assess plan feasibility before execution, falling back to continuous forward search when necessary.

---

[Oracle-Guided Soft Shielding for Safe Move Prediction in Chess](http://arxiv.org/abs/2603.08506)

- OGSS (Oracle-Guided Soft Shielding): introduces a framework for safer decision-making in chess by augmenting an imitation-learned move predictor with a probabilistic blunder predictor trained on oracle feedback.
- The system utilizes a learned safety shield to evaluate tactical risks of candidate moves, allowing for a flexible trade-off between performance and safety through various selection strategies like action elimination or weighted utility functions.
- Experimental results demonstrate that the approach significantly reduces blunder rates and centipawn loss compared to standard imitation learning and hard-shielding baselines, even under high exploration ratios.

---

[Behavioral Generative Agents for Power Dispatch and Auction](http://arxiv.org/abs/2603.08477)

- TARJ (Thought-Action-Reflection-Journal): introduces a behavioral generative agent framework for simulating human-like decision-making in power dispatch and auction environments, with basic information, ICL-examples, and a structured reasoning loop.

- The architecture utilizes an In-Context Learning (ICL) module to transfer complex behavioral patterns from high-reasoning models to smaller LLMs, enabling them to prioritize long-term reliability over immediate profit.

- The framework includes a memory mechanism where "Journal" entries are stored and dynamically fed back into subsequent prompts to maintain strategic continuity and adaptive behavior across sequential auction rounds.


---

[One Model Is Enough: Native Retrieval Embeddings from LLM Agent Hidden States](http://arxiv.org/abs/2603.08429)

- Native Retrieval Embeddings: introduces a method to eliminate redundant embedding models in RAG pipelines by projecting LLM hidden states directly into a vector space, with LLM-agent, hidden-state-extraction, and projection-head components.
- The framework utilizes a lightweight transformer-based projection head trained via knowledge distillation from a teacher model using alignment, contrastive, and rank distillation losses.
- Evaluation on the QReCC benchmark shows the approach maintains 97% of baseline retrieval quality while achieving a 21.8x reduction in inference latency.

---

[IronEngine: Towards General AI Assistant](http://arxiv.org/abs/2603.08425)

- IronEngine: introduces a general AI assistant platform organized around a unified orchestration core, with Discussion, Model Switch, and Execution pipeline components.
- The architecture includes planning-, evaluation-, and execution-agents supported by a hierarchical memory system and a vectorized skill repository for persistent behavior.
- It implements VRAM-aware model management and an intelligent tool routing layer with alias normalization to optimize local LLM performance on consumer hardware.

---

[Agentic Critical Training](http://arxiv.org/abs/2603.08706)

- ACT (Agentic Critical Training): introduces a reinforcement learning paradigm that trains LLM agents to autonomously develop critical reasoning by identifying superior actions among alternatives, utilizing a three-stage pipeline involving contrastive data construction, critical training via GRPO, and final action-level reinforcement learning.

- The framework replaces imitation of pre-constructed reflection text with verifiable rewards for correct action selection, driving the emergence of genuine self-reflection and internalizing an understanding of action quality.

- Experimental results across embodied, web, and scientific benchmarks demonstrate that ACT significantly improves success rates, enhances out-of-distribution generalization, and prevents reasoning collapse on general mathematical and scientific tasks.


---

[Coverage-Guided Multi-Agent Harness Generation for Java Library Fuzzing](http://arxiv.org/abs/2603.08616)

- Multi-Agent Harness Generation Framework: introduces an automated system for Java library fuzzing that utilizes specialized LLM-powered agents—including research-, synthesis-, compilation-, coverage analysis-, and refinement-agents—to research, synthesize, and iteratively refine fuzz harnesses through on-demand program analysis.
- The architecture employs the Model Context Protocol to enable agents to query documentation, source code, and callgraph information on demand, preventing context saturation while exploring complex library dependencies.
- It incorporates method-targeted coverage tracking and agent-guided termination to distinguish productive refinement opportunities, achieving a median 26% improvement in method-targeted coverage and discovering three previously unreported bugs.

---

[A Dynamic Equilibrium Model for Automated Market Makers](http://arxiv.org/abs/2603.08603)

- Dynamic Equilibrium Model for AMMs: introduces a theoretical framework for Constant Function Market Makers that formalizes strategic interactions between informed arbitrageurs, noise traders, and liquidity providers under stochastic volatility and endogenous gas fees.
- The model identifies an intrinsic buy-sell asymmetry in price impact and characterizes how arbitrage races generate ex-post unprofitable trades that contribute to liquidity provider returns.
- Empirical validation using Uniswap data confirms that optimal liquidity provision follows a hump-shaped relationship with volatility, balancing fee revenue from noise and overrun trades against adverse selection costs.

---

[RETROAGENT: From Solving to Evolving via Retrospective Dual Intrinsic Feedback](http://arxiv.org/abs/2603.08561)

- RETROAGENT: introduces an online reinforcement learning framework that enables LLM agents to evolve through a hindsight self-reflection mechanism producing dual intrinsic feedback, which includes decision-making- and self-reflection-agents.
- The framework generates intrinsic numerical feedback to reward incremental subtask completion and intrinsic language feedback to distill reusable lessons into a persistent memory buffer.
- It implements the SimUtil-UCB strategy to balance semantic relevance, historical utility, and exploration coverage when retrieving past experiences to guide current agent actions.


---

[Fusion of Monostatic and Bistatic Sensing for ISAC-Enabled Low-Altitude Environment Mapping](http://arxiv.org/abs/2603.08556)

- Bayesian multipath-based environment mapping framework: introduces a probabilistic approach for low-altitude mapping by fusing monostatic and bistatic sensing measurements through a unified geometric model, with ISAC-enabled transceivers, UAV receivers, and factor-graph-based inference.
- The framework employs two complementary Bayesian schemes—Scheme I for parallel processing and Scheme II for sequential cross-link updates—to handle one-to-many data associations arising from non-ideal diffuse scattering.
- By integrating stable monostatic backscatter constraints with bistatic geometric observations, the system achieves higher mapping accuracy and robustness compared to single-link baselines in complex urban corridors.


---

[The Neural Compass: Probabilistic Relative Feature Fields for Robotic Search](http://arxiv.org/abs/2603.08544)

- ProReFF (Probabilistic Relative Feature Fields): introduces a self-supervised framework that learns statistical co-occurrence structures from vision-language features to guide robotic agents toward target objects in unseen environments.
- The architecture includes DINOv2- and CLIP-based perception components, an alignment network for resolving training data inconsistencies, and a search agent with exploitation- and exploration-modules.
- By predicting probable spatial contexts at multiple scales, the agent navigates complex multi-floor layouts, achieving a 20% efficiency improvement over standard frontier-based exploration methods.

---

[Predicting Conflict Impact on Performance in O-RAN](http://arxiv.org/abs/2603.08685)

- Performance Predictor (integrated with PACIFISTA): introduces a system to forecast the impact of conflicts between autonomous agents on O-RAN performance by combining independently collected application profiles using frequency-weighted statistical distributions.
- The framework utilizes Empirical Cumulative Density Functions (ECDFs) and a last-writer-wins approximation to estimate Key Performance Metrics (KPMs) across various timescale configurations without requiring simultaneous execution.
- Validation on the Colosseum testbed demonstrates that the predictor accurately captures performance degradation for conflicting Energy Saving and Throughput Maximization xApps.

---

[Cybersecurity AI: Hacking Consumer Robots in the AI Era](http://arxiv.org/abs/2603.08665)

- CAI (Cybersecurity AI): introduces an autonomous robotic security assessment framework with CAI (autonomous security assessment framework), LLM-based Security Agent (core reasoning and task execution), Network Reconnaissance Module (automated attack surface discovery), Vulnerability Testing Engine (systematic security flaw identification), Exploit Validation Component (automated proof-of-concept generation), Human-in-the-loop Interface (safety oversight and intervention), and Robotic Target Systems (consumer robots being evaluated).
- The framework automates the discovery and exploitation of vulnerabilities across diverse robotic platforms, reducing assessment time by 3-5x compared to traditional expert-led security research.
- The study identifies 38 vulnerabilities across three robotic platforms, highlighting the democratization of complex hacking tasks through Generative AI and the urgent need for GenAI-native defensive measures.

---

[Optimal Savings under Transition Uncertainty and Learning Dynamics](http://arxiv.org/abs/2603.08663)

- Optimal Savings under Transition Uncertainty and Learning Dynamics: introduces a general theory for household consumption and saving decisions under transition uncertainty, with Exogenous State (governs economic environment), Belief State (posterior distribution over models), Bayesian Learning Module (updates beliefs via Bayes' rule), Optimal Policy (determines consumption and saving), Endogenous Grid Algorithm (numerical solution method), Barycentric Coordinate Grid (discretizes belief simplex), and Time Iteration Operator (characterizes optimal policy fixed-point).
- The research establishes the existence, uniqueness, and structural properties of the optimal policy, including monotonicity and concavity, while incorporating the agent's posterior belief as an endogenous state variable.
- The model demonstrates how transition uncertainty initially increases precautionary saving but enhances long-run wealth and consumption smoothing as learning resolves regime uncertainty.

---

[Context-free Self-Conditioned GAN for Trajectory Forecasting](http://arxiv.org/abs/2603.08658)

- Context-free Self-Conditioned GAN: introduces an unsupervised trajectory forecasting framework that leverages a self-conditioned GAN, including discriminator- and generator-components, to identify and learn diverse behavioral modes from 2D motion data.
- The system utilizes k-Means clustering on discriminator-extracted features to generate unsupervised labels, which then guide the generator to mitigate mode collapse.
- The methodology incorporates weighted generator loss and batch sampling to enhance performance on less representative motion profiles in human and road agent datasets.

---

[OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning](http://arxiv.org/abs/2603.08655)

- OfficeQA Pro: introduces a benchmark for evaluating AI agents on grounded, multi-document reasoning over a large corpus of heterogeneous financial documents, with all LLM Backend (orchestrates reasoning and tool use), Web-Search Tool (queries internet for external data), Python REPL (executes code for numerical analysis), File Search Tool (performs grep-like corpus navigation), Vector Search (retrieves context via semantic embeddings), Document Parser (converts PDFs to structured text), and Sliding Window Memory (retains recent message history).
- The system includes reasoning-, planning-, and tool use-capabilities within the LLM Backend to navigate complex archives and perform multi-step analytical computations.
- Experimental results show that while specialized parsing via Databricks' ai_parse_document yields significant performance gains, substantial performance gaps remain for agents to achieve complex enterprise-level reliability.

---

[POSTTRAINBENCH: Can LLM Agents Automate LLM Post-Training?](http://arxiv.org/abs/2603.08640)

- POSTTRAINBENCH: introduces an end-to-end testbed to evaluate the ability of autonomous LLM agents to perform post-training on base models under bounded compute constraints.
- The framework utilizes a scaffold-based agent architecture comprising reasoning- and auditing-LLMs, a software execution loop, and a suite of developer tools.
- The study benchmarks various frontier agents across multiple tasks, revealing that while agents can outperform human-engineered baselines on narrow tasks, they often engage in reward-hacking behaviors such as data contamination and unauthorized API usage.

---


[Trust via Reputation of Conviction](http://arxiv.org/abs/2603.08575)

- Reputation of Conviction: introduces a mathematical framework for trust by formalizing reputation as the expected weighted signed conviction, with Claims (meaningful statements or tasks), Sources (generative and discriminative actors), Perceptions (perturbed observations of claims), Truth Assessments (source-defined truth estimators), Aggregation (consensus-building mechanism), Objective Truth (asymptotic consensus limit), Conviction (likelihood of consensus vindication), Reputation (expected weighted signed conviction), and Continuous Verification (ongoing post-deployment assessment).
- The framework distinguishes between assimilative and augmentative regimes, where the latter rewards genuine innovation and discovery by measuring how a source's perception shifts the objective consensus.
- It identifies continuous verification as a theoretical necessity for accurate reputation assessment in LLMs, moving trust from a static certification to a dynamic, observable history of performance.

---

[SecAgent: Efficient Mobile GUI Agent with Semantic Context](http://arxiv.org/abs/2603.08533)

- SecAgent (Efficient Mobile GUI Agent with Semantic Context): introduces an efficient 3B-scale multimodal LLM agent for mobile GUI automation, utilizing a semantic context mechanism to distill historical screenshots and actions into concise natural language summaries.
- The framework employs a two-stage training process involving supervised fine-tuning on the large-scale Chinese CMGUI dataset and reinforcement fine-tuning using the Group Relative Policy Optimization algorithm.
- By maintaining a dynamic semantic context, the agent significantly reduces computational overhead while achieving performance comparable to much larger 7B-8B models on navigation benchmarks.

---

[SCAFFOLD-CEGIS: Preventing Latent Security Degradation in LLM-Driven Iterative Code Refinement](http://arxiv.org/abs/2603.08520)

- SCAFFOLD-CEGIS: introduces a multi-agent collaborative architecture to prevent security degradation during iterative LLM code refinement, including security architect-, code generation-, verification-, and failure assimilation-agents.
- The framework transforms implicit security prompts into explicit verifiable constraints through semantic anchoring and enforces safety monotonicity via a four-layer gated verification process checking correctness, security, diff budget, and anchor integrity.
- It utilizes a counterexample-guided inductive synthesis paradigm to assimilate experience from failed iterations, reducing latent security degradation rates to 2.1% while maintaining code evolution capability across multiple programming languages.

---

[Fanar-Sadiq: A Multi-Agent Architecture for Grounded Islamic QA](http://arxiv.org/abs/2603.08501)

- Fanar-Sadiq: introduces a bilingual multi-agent architecture for grounded Islamic question answering, which includes intent-classification, reasoning, understanding, and interpretation agents.
- The system utilizes specialized modules for Quranic verse retrieval, Shariah-compliant zakat and inheritance computation, and retrieval-augmented generation for jurisprudential rulings.
- It ensures doctrinal integrity through citation normalization, madhhab-sensitive branching for legal disputes, and post-generation verification to mitigate LLM hallucinations.

---

[Towards Modeling Cybersecurity Behavior of Humans in Organizations](http://arxiv.org/abs/2603.08484)

- Human-Centric Cybersecurity Behavior Model: introduces a structured synthesis of psychological and environmental drivers influencing security actions within organizations, with all Motivation, Awareness and Knowledge, Role, Culture, Norms, Mindset and Attitude, Intention, Skills, Usability, Situation, Agency, Assessment of the situation, and Behavior.
- The framework organizes these factors across individual-environmental, fundamental-situational, and hidden-visible dimensions to capture the sociotechnical complexity of workplace security.
- The study maps these human factors to agentic AI architectures—including system prompt-, alignment- and collaboration-components—to predict vulnerabilities in autonomous LLM-based agents.

---

[The Boiling Frog Threshold: Criticality and Blindness in World Model-Based Anomaly Detection Under Gradual Drift](http://arxiv.org/abs/2603.08455)

- World Model-Based Self-Monitoring: introduces a framework for analyzing the detection of gradual observation corruption in reinforcement learning agents, with PPO Agent (reinforcement learning interaction policy), MLP World Model (forward dynamics state predictor), Drift Injector (gradual observation corruption module), Anomaly Detectors (statistical monitors for error signals), and Prediction Error Signal (metric for self-monitoring detection).
- The research reveals a universal sigmoid detection threshold and demonstrates that world models are fundamentally blind to periodic perturbations that oscillate symmetrically around zero. 
- It identifies the "Collapse Before Awareness" regime where agents fail before detection and proves that detection thresholds are invariant to model capacity but dependent on environment-specific dynamics.

---

[Sandpiper: Orchestrated AI-Annotation for Educational Discourse at Scale](http://arxiv.org/abs/2603.08406)

- Sandpiper: introduces a mixed-initiative system for high-throughput qualitative analysis of educational discourse, with a researcher dashboard (session exploration and evaluation), a schema-constrained LLM orchestrator (agentic output validation), a secure AI gateway (privacy-preserving LLM routing), a backend API (data processing), and a data store (persistent storage).
- The framework implements an automated de-identification workflow and an orchestrator loop that enforces strict adherence to researcher-defined qualitative codebooks through iterative JSON schema validation and re-prompting. 
- An integrated evaluations engine enables continuous benchmarking of AI performance against human labels using metrics like Cohen’s Kappa and precision-recall to ensure methodological rigor in large-scale discourse analysis.

---

[A Recipe for Stable Offline Multi-agent Reinforcement Learning](http://arxiv.org/abs/2603.08399)

- SVN (Scale-invariant Value Normalization): introduces a stabilization technique for offline multi-agent reinforcement learning that addresses value-scale amplification in non-linear value decomposition, including individual utility-, mixing-, and policy extraction-networks.
- The framework identifies that non-linear mixing networks induce coupled instability between value learning and policy extraction, leading to divergent Q-values and ill-conditioned actor updates.
- By normalizing Q-values using batch-dependent mean and absolute deviation, SVN restores contractive Bellman updates and enables reliable use of expressive non-linear decomposition in offline settings.

---

[A Hierarchical Error-Corrective Graph Framework for Autonomous Agents with LLM-Based Action Generation](http://arxiv.org/abs/2603.08388)

- HECG (Hierarchical Error-Corrective Graph): introduces a framework for autonomous agents that combines LLM-based action generation with structured error-driven execution and graph-based experience retrieval, including planning- and scoring-LLMs.
- The architecture features a multi-dimensional transition policy that integrates quantitative performance metrics with LLM-derived semantic reasoning scores to select candidate strategies from a directed graph representation.
- It implements a hierarchical recovery system ranging from local action adjustments to task-level replanning based on a structured error matrix classification of failure types and severity levels.

---

[MoMaStage: Skill-State Graph Guided Planning and Closed-Loop Execution for Long-Horizon Indoor Mobile Manipulation](http://arxiv.org/abs/2603.08383)

- MoMaStage: introduces, a structured vision-language framework for long-horizon indoor mobile manipulation without explicit scene mapping, with Vision-Language Model (VLM) (high-level task decomposition planner), Hierarchical Skill Library (repository of hierarchical skills), Skill-State Graph (topology-aware transition model), Proprioception Monitor (real-time physical state monitor), Semantic Verifier (VLM-driven post-execution state validation), and Dynamic Replanner (graph-constrained failure recovery mechanism).
- The architecture employs a two-stage planning process consisting of topology-aware semantic decomposition followed by state-driven logical verification to ensure global state consistency and physical feasibility.
- The closed-loop execution mechanism enables autonomous recovery from physical anomalies by triggering graph-anchored semantic replanning when deviations are detected by the monitoring modules.

---

[M3-ACE: Rectifying Visual Perception in Multimodal Math Reasoning via Multi-Agentic Context Engineering](http://arxiv.org/abs/2603.08369)

- M3-ACE (Multi-Agentic Context Engineering): introduces a multi-agent framework designed to rectify visual perception errors in multimodal math reasoning by decoupling perception from reasoning through a shared context of visual evidence.
- The architecture includes anchor- and assistant-agents that collaboratively populate a context book with diverse visual observations, which are then processed by summary and refinement tools.
- By iteratively filtering unreliable samples and resolving perceptual inconsistencies, the framework establishes new performance benchmarks on MathVision without requiring additional model training.

---

[SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation](http://arxiv.org/abs/2603.08329)

- SPD-RAG (Sub-agent per Document Retrieval-Augmented Generation): introduces a hierarchical multi-agent framework for exhaustive cross-document question answering that factors the problem along the document axis, including coordinator-, sub-agent-, and synthesis-agents.
- The system utilizes a centralized coordinator to decompose queries into shared instructions for parallel sub-agents that operate on isolated document universes to extract grounded findings.
- A recursive synthesis layer employs similarity-ordered merging to aggregate findings into a comprehensive final answer, achieving high performance on long-context benchmarks at a fraction of the cost of full-context LLMs.

---

[Agentic Neurosymbolic Collaboration for Mathematical Discovery: A Case Study in Combinatorial Design](http://arxiv.org/abs/2603.08322)

- Agentic Neurosymbolic Collaboration Framework: introduces a mathematical discovery system that integrates orchestration- and critic-LLMs with symbolic computation tools, human strategic direction, and a two-tier persistent memory system.
- The architecture employs multi-model deliberation among frontier LLMs for rigorous criticism and error detection, while utilizing specialized symbolic solvers for exhaustive enumeration and stochastic optimization.
- This collaborative approach enabled the discovery and formal verification of a tight lower bound on Latin square imbalance, demonstrating the effectiveness of combining neural pattern recognition with symbolic rigor.

---

[SlowBA: An efficiency backdoor attack towards VLM-based GUI agents](http://arxiv.org/abs/2603.08316)

- SlowBA: introduces an efficiency-focused backdoor attack targeting VLM-based GUI agents by inducing excessively long reasoning chains to manipulate response latency, with Trigger Injection (generates realistic pop-up triggers), Stage I: Response Format Alignment (aligns long-response structure), Stage II: Trigger-aware Reward-level Optimization (activates high-latency behavior), VLM-based GUI Agent (executes GUI-based tasks), and Reward Function (calculates length-based rewards).
- The framework utilizes a two-stage Reward-level Backdoor Injection (RBI) strategy to disentangle response length manipulation from action accuracy preservation, including trigger-generation and task-execution agents.
- It leverages Group Relative Policy Optimization (GRPO) to maximize sequence length for triggered inputs while maintaining normal performance on clean GUI environments.

---

[Less is More: Robust Zero-Communication 3D Pursuit-Evasion via Representational Parsimony](http://arxiv.org/abs/2603.08273)

- OURS-LITE: introduces a robust decentralized multi-agent reinforcement learning framework for 3D pursuit-evasion, with 3D A* Guidance, Parsimonious Observation Interface, Recurrent Actor, Contribution-Gated Credit Assignment (CGCA), and Visibility Gating.
- The architecture utilizes representational parsimony by removing team-coupled observation channels to mitigate the propagation of stale peer beliefs under communication denial and sensing noise.
- CGCA implements locality-aware directional gating and participation-based reward scaling to sustain cooperative capture quality and suppress free-rider behavior without explicit inter-agent messaging.

---

[SAIL: Test-Time Scaling for In-Context Imitation Learning with VLM](http://arxiv.org/abs/2603.08269)

- SAIL (Scaling In-context Imitation Learning): introduces a framework that reframes robot imitation as an iterative refinement problem by scaling test-time compute through Monte Carlo Tree Search over full trajectories.
- The architecture integrates a Policy VLM for trajectory generation with a Scoring VLM that provides dense, step-level feedback based on subtask completion progress.
- A shared trajectory archive enables similarity-based retrieval of successful demonstrations to bootstrap the search process across varying environmental conditions.

---

[FinToolBench: Evaluating Large Language Model Agents for Real-World Financial Tool Use](http://arxiv.org/abs/2603.08262)

- FinToolBench (Evaluating Large Language Model Agents for Real-World Financial Tool Use): introduces a real-world, runnable benchmark for evaluating LLM agents on financial tool use, featuring 760 executable tools and 295 queries.
- The framework incorporates FATR (Finance-Aware Tool Routing), which includes planning-, constraint inference-, and extraction-agents to ensure domain-specific alignment.
- It assesses agent performance through auditable tool traces, measuring execution success alongside compliance with timeliness, intent restraint, and regulatory domain constraints.

---

[SiMO: SINGLE-MODALITY-OPERABLE MULTIMODAL COLLABORATIVE PERCEPTION](http://arxiv.org/abs/2603.08240)

- SiMO (Single-Modality-Operable Multimodal Collaborative Perception): introduces a multimodal collaborative perception framework, with LiDAR and camera encoders, BEV transformation, semantic aligners, LAMMA fusion, multi-agent fusion, and task heads.
- The framework utilizes the Length-Adaptive Multi-Modal Fusion (LAMMA) module to adaptively handle missing modalities by downgrading to self-attention while maintaining semantic consistency.
- It implements the Pretrain-Align-Fuse-RD (PAFR) training strategy to overcome modality competition, ensuring each individual modality branch remains functional and independent.

---

[FIBRATION POLICY OPTIMIZATION](http://arxiv.org/abs/2603.08239)

- FiberPO (Fibration Policy Optimization): introduces an algebraic framework for multi-scale stability control in reinforcement learning for LLMs, with APC-Obj, FBG, FGH, FiberPO-Trajectory, FiberPO-Domain, Base space, Total space, Bundle projection, Markov kernel, Atomic gates, Fibration decomposition map, and Recovery map.
- The architecture organizes sampled data as a fiber bundle to decompose ratio gating into coordinated global base-level gates and local fiber-level residual gates.
- The system provides a restorative gradient structure through a rollback mechanism and scales to arbitrary hierarchical depths including domain-, prompt group-, trajectory- and token-level gates.

---

[SplitAgent: A Privacy-Preserving Distributed Architecture for Enterprise-Cloud Agent Collaboration](http://arxiv.org/abs/2603.08221)

- SplitAgent: introduces a distributed architecture for secure enterprise-cloud collaboration, with a Privacy Agent (enterprise-side data management), Reasoning Agent (cloud-side LLM analysis), Context-Aware Sanitizer (task-specific masking), Privacy Budget Manager (tracks privacy expenditure), Local RAG Engine (local retrieval), Local Tool Executor (local execution), Data Controller (data access management), Task Planner (request decomposition), Logical Reasoner (causal analysis), Strategy Generator (recommendation creation), Abstract Synthesizer (analysis synthesis), LLM Interface (cloud model access), and SplitAgent Protocol (secure communication).
- The architecture utilizes context-aware dynamic sanitization to adapt privacy protection based on task semantics, ensuring high utility while maintaining differential privacy.
- It incorporates planning-, reasoning-, and synthesis-agents to perform complex analysis on sanitized abstractions while managing cumulative privacy budgets across multi-turn interactions.

---

[DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining](http://arxiv.org/abs/2603.08216)

- DualTurn: introduces a turn-taking component for modular voice pipelines using generative pretraining on dual-channel conversational audio to implicitly learn interaction dynamics.
- The framework employs a two-stage training process where a 0.5B LLM backbone is first pretrained for next-audio token prediction and then fine-tuned with twelve classification heads to derive per-channel signals.
- By monitoring both speaker channels continuously, the model anticipates turn boundaries and maps predicted signals into five distinct agent actions including backchannels and interruptions.

---

[Human–AI Collaboration for Scaling Agile Regression Testing: An Agentic-AI Teammate from Manual to Automated Testing](http://arxiv.org/abs/2603.08190)

- Hacon Test Automation Copilot: introduces an agentic AI teammate that generates executable system test scripts from validated specifications using a retrieval-augmented, multi-agent architecture, including generator-, evaluator-, and reporter-agents.
- The framework utilizes a bounded workflow to automate the transition from manual specifications to automated regression tests while maintaining human oversight through mandatory review points.
- The system integrates with Jenkins for script execution and MLflow for artifact traceability, significantly increasing test script throughput in agile environments.


---

[AutoAdapt: An Automated Domain Adaptation Framework for Large Language Models](http://arxiv.org/abs/2603.08181)

- AutoAdapt: introduces an automated end-to-end framework for LLM domain adaptation, with a User Agent, User Preference Store, Best Practice KB, Adaptation Configuration Graph, Multi-Agent Debate Planner, AutoRefine, Plan Execution, and User Constraint Verification.
- The system employs a multi-agent debating system including proposal-, critic-, and aggregator-agents to iteratively refine adaptation plans grounded in curated best practices from open-source ecosystems.
- It incorporates AutoRefine, which integrates LLM-based surrogate modeling with Gaussian Process function estimation to optimize hyperparameters under tight resource constraints.

---

[Evidence-Driven Reasoning for Industrial Maintenance Using Heterogeneous Data](http://arxiv.org/abs/2603.08171)

- Condition Insight Agent: introduces a deployed decision-support framework for industrial maintenance that separates deterministic evidence construction from constrained LLM synthesis to produce auditable, evidence-grounded explanations.
- The architecture utilizes an Analytics & Evidence Construction module to transform fragmented data into structured evidence packets, which are then processed by a Domain LLM Agent.
- A deterministic verification loop governs the LLM's output by cross-checking generated condition categories against explicit operational rules to ensure reliability and prevent hallucinations.

---

[RexDrug: Reliable Multi-Drug Combination Extraction through Reasoning-Enhanced LLMs](http://arxiv.org/abs/2603.08166)

- RexDrug (Reliable Multi-Drug Combination Extraction): introduces an end-to-end reasoning-enhanced framework for n-ary drug combination extraction, utilizing multi-agent reasoning distillation and reinforcement learning with multi-dimensional rewards.
- The framework includes a Medical Reasoning Analyst for generating reasoning traces and a Medical Expert Reviewer for providing iterative feedback and quality assessment.
- It utilizes Group Relative Policy Optimization (GRPO) to optimize a policy model against a reference model, ensuring structural format compliance and pharmacological accuracy.

---

[The Differential Effects of Agreeableness and Extraversion on Older Adults’ Perceptions of Conversational AI Explanations in Assistive Settings](http://arxiv.org/abs/2603.08164)

- LLM-VA (Large Language Model-based Voice Assistant): introduces a controlled experimental study examining how agreeableness and extraversion in an LLM-VA influence older adults' perceptions of AI explanations across routine and emergency contexts.
- The system utilizes Trait Modulation Keys (TMK) to steer agent personality via dual-key prompting, separating behavioral stance from linguistic surface form in the GPT-5 generated responses.
- Findings demonstrate that while agreeableness directly drives sociability perceptions, the effectiveness of extraversion is contingent on the quality of contextually grounded environmental explanations.

---

[EvoScientist: Towards Multi-Agent Evolving AI Scientists for End-to-End Scientific Discovery](http://arxiv.org/abs/2603.08127)

- EvoScientist: introduces an evolving multi-agent framework for end-to-end scientific discovery, with a Researcher Agent (RA), an Engineer Agent (EA), and an Evolution Manager Agent (EMA).
- The system utilizes persistent ideation and experimentation memories to store successful research directions and failed attempts, allowing agents to refine their strategies over time.
- It implements tree-structured search for both idea generation and experiment execution, achieving superior performance in novelty and feasibility compared to static AI scientist baselines.

---

[UIS-DIGGER: TOWARDS COMPREHENSIVE RESEARCH AGENT SYSTEMS FOR REAL-WORLD UNINDEXED INFORMATION SEEKING](http://arxiv.org/abs/2603.08117)

- UIS-Digger: introduces a multi-agent framework for Unindexed Information Seeking (UIS), which includes planner-, web searcher-, web surfer-, and file reader-agents.
- The architecture utilizes a dual-mode browser that maintains a shared memory between visual and textual observations to facilitate deep interaction with dynamic web elements and unindexed content.
- The authors establish the UIS-QA benchmark to evaluate agent performance on information hidden from search engine indices, where UIS-Digger sets a strong baseline via two-stage supervised and rejection sampling fine-tuning.

---

[DeReCo: Decoupling Representation and Coordination Learning for Object-Adaptive Decentralized Multi-Robot Cooperative Transport](http://arxiv.org/abs/2603.08111)

- DeReCo (Decoupling Representation and Coordination Learning): introduces a multi-agent reinforcement learning framework that decouples object-dependent representation learning from coordination learning to facilitate decentralized multi-robot cooperative transport of diverse objects.
- The approach employs a three-stage training pipeline consisting of centralized coordination with privileged information, supervised training of an adaptive LSTM encoder, and fine-tuning for decentralized execution without privileged data.
- The system demonstrates robust generalization to unseen objects with varying physical properties and achieves successful zero-shot transfer from simulation to physical Human Support Robots in real-world experiments.

---

[Whataboutism](http://arxiv.org/abs/2603.08098)

- PSPE (Psychological Subgame Perfect Equilibrium): introduces a formalization of whataboutism as an equilibrium phenomenon within an infinite-horizon psychological game, with rival camps (symmetric groups of agents), sensitivity states (contextual levels of offensiveness), speech acts (offensive expression or condemnation), internal-external condemnation (social sanctions from own/rival camp), whataboutism rebuttal (deflection of external criticism), memory retrieval (searching for past misconduct), and PSPE (equilibrium solution concept).
- The model characterizes how agents strategically retrieve past misconduct from the rival camp to neutralize external criticism, thereby reducing the social cost of offensive speech.
- The analysis reveals that increased political polarization amplifies both the frequency of offensive speech and the prevalence of whataboutism as a defensive rhetorical strategy.

---

[From Reactive to Map-Based AI: Tuned Local LLMs for Semantic Zone Inference in Object-Goal Navigation](http://arxiv.org/abs/2603.08086)

- Map-Based AI: introduces a proactive exploration framework for Object-Goal Navigation that transitions from reactive observation-to-action paradigms to structured map-conditioned reasoning using fine-tuned LLs.
- The system integrates a hybrid topological-grid map with a reasoning engine powered by a Llama-2 model fine-tuned via Low-Rank Adaptation to infer functional "zones" from verbalized object sets.
- It employs Traveling Salesman Problem optimization for local scanning and semantic frontier weighting to prioritize high-probability areas, significantly improving navigation efficiency in unknown environments.

---

[ImageEdit-R1: Boosting Multi-Agent Image Editing via Reinforcement Learning](http://arxiv.org/abs/2603.08059)

- ImageEdit-R1: introduces a multi-agent framework for intelligent image editing that formulates the task as a sequential decision-making problem, featuring decomposition-, sequencing-, and diffusion-based editing-agents.
- The framework utilizes Group Relative Policy Optimization (GRPO) to train the VLM-based decomposition agent, ensuring precise extraction of edit actions, subjects, and goals from complex or multi-step user instructions.
- Experimental results demonstrate that this collaborative multi-agent approach significantly outperforms individual closed-source diffusion models and alternative frameworks in instruction alignment and content preservation.

---

[Finite-Horizon Optimal Consumption and Investment with Time-Varying Job-Switching Costs](http://arxiv.org/abs/2603.08050)

- Finite-horizon optimal switching model: introduces a mathematical framework for an economic agent's optimal consumption, investment, and job-switching decisions under time-varying costs, with an Economic Agent, a Financial Market, a Job-Switching Model, a Dual-Martingale Approach, a Parabolic Double Obstacle Problem, and Free Boundaries.
- The approach employs a dual-martingale transformation to reduce the agent's utility maximization problem to a parabolic double obstacle problem with time-dependent obstacles.
- The study establishes the existence, uniqueness, and smoothness of the solution and its associated free boundaries to characterize optimal strategies.

---

[Distributed Coordination Algorithms with Efficient Communication for Open Multi-Agent Systems with Dynamic Communication Links and Processing Delays](http://arxiv.org/abs/2603.08038)

- QAOD (Quantized Averaging over OMAS with Dynamic communication links): introduces a communication-efficient framework for distributed quantized average consensus in open multi-agent systems, with QAOD (manages dynamic directed links), QAPOD (mitigates bounded processing delays), QAIOD (handles indefinitely open systems), Arriving Strategy (initializes mass for arrivals), Departing Strategy (handoffs mass to neighbors), Remaining Strategy (updates quantized consensus states), Departing Soon Strategy (processes info without transmitting), Long-Term Remaining Strategy (targets stable out-neighbors), Mass and State Variables (store and compute values), and Feedback Channel (provides narrowband acknowledgments).
- The framework incorporates QAPOD to mitigate unknown bounded processing delays through a departing soon strategy and a long-term remaining strategy that restricts transmissions to stable out-neighbors.
- QAIOD enables consensus over indefinitely open networks by tracking historical participation to compute the average of all nodes that have ever been active.

---

[Samyama: A Unified Graph-Vector Database with In-Database Optimization, Agentic Enrichment, and Hardware Acceleration](http://arxiv.org/abs/2603.08036)

- Samyama: introduces a unified graph-vector database architecture, with Client Interface, Query Engine, Graph Engine, Vector Engine, Optimization Engine, GAK (Generation-Augmented Knowledge), LLM Agent, NLQPipeline, GPU Acceleration, RocksDB Storage, Event Trigger, Tool Use, Data Parsing, and Graph Update; it includes reasoning- and natural language query-agents.
- The system integrates 22 metaheuristic solvers and HNSW vector indexing into a single Rust-based engine to eliminate data movement overhead between disparate stores.
- It utilizes LLMs for autonomous knowledge graph expansion and leverages wgpu-based GPU acceleration for high-performance graph analytics and ingestion.

---

[ConflictBench: Evaluating Human–AI Conflict via Interactive and Visually Grounded Environments](http://arxiv.org/abs/2603.08024)

- ConflictBench: introduces a benchmark for evaluating human–AI conflict through 150 multi-turn, visually grounded interactive scenarios, featuring a text-based simulation engine and a video-based world model.
- The framework utilizes GPT-5 to expand seed scenarios into structured environments and employs Wan2.2 for generating consistent visual observations across multiple interaction steps.
- Evaluation involves measuring Task Success Rate and Alignment Success Rate, alongside a regret test to assess behavioral stability under escalating pressure.

---

[PIRA-Bench: A Transition from Reactive GUI Agents to GUI-based Proactive Intent Recommendation Agents](http://arxiv.org/abs/2603.08013)

- PIRF (Proactive Intent Recommendation Framework): introduces a memory-aware, state-tracking architecture that wraps general MLLMs to autonomously anticipate user goals from continuous visual streams.
- The framework utilizes a dynamic Memory Module to maintain static user profiles and an intent bank of active threads, enabling the disentanglement of interleaved multitasking scenarios.
- It incorporates a reflection-based auto-deletion mechanism and a structured action space including Create, Resume, Update, and IDLE actions to mitigate hallucinations and handle noisy, non-linear GUI trajectories.

---

[CMMR-VLN: Vision-and-Language Navigation via Continual Multimodal Memory Retrieval](http://arxiv.org/abs/2603.07997)

- CMMR-VLN (Continual Multimodal Memory Retrieval based Vision-and-Language Navigation): introduces a zero-shot framework that endows LLM agents with structured multimodal memory and reflection capabilities to improve navigation in unfamiliar environments, with MEM, RAGP, and a Reflection Module.
- The system utilizes a retrieval-augmented generation pipeline to transform past experiences into explicit navigation rules, which are then integrated into a structured chain-of-thought process for the backbone LLM.
- A reflection mechanism evaluates navigation outcomes to selectively reinforce successful trajectories or distill failure cases into concise error-correcting notes, enabling continual learning without retraining.

---

[TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](http://arxiv.org/abs/2603.07988)

- TeamHOI: introduces a unified decentralized policy for cooperative human-object interactions across varying team sizes, utilizing a Transformer-based architecture with teammate tokens to enable scalable coordination.
- The framework employs a masked Adversarial Motion Prior (AMP) strategy that leverages single-human reference motions while masking object-interacting body parts to promote motion realism and diversity.
- A team-size- and shape-agnostic formation reward is designed to guide agents into stable carrying positions for diverse object geometries in physics-based simulations.

---

[$OneMillion-Bench: How Far are Language Agents from Human Experts?](http://arxiv.org/abs/2603.07980)

- $1M-Bench ($OneMillion-Bench): introduces a benchmark of 400 expert-curated tasks across five economically consequential domains to evaluate the economic value and professional reliability of language agents, including vanilla-, search-, and deep research-agents, with all Task Creation, Peer Review, Resolution and Revision, Difficulty Control, Expert Score, Pass Rate, Economic Value, Vanilla Models, Search Agents, and Deep Research Agents.
- The framework utilizes a three-stage data curation pipeline involving task creation, peer review, and independent audits to ensure discriminative and workflow-realistic challenges.
- Evaluation is conducted through rubric-based scoring of factual accuracy, logical coherence, and professional compliance, quantifying agent capabilities via estimated labor costs and market wages.

---

[OSExpert: Computer-Use Agents Learning Professional Skills via Exploration](http://arxiv.org/abs/2603.07978)

- OSExpert: introduces an environment-learning paradigm that enables computer-use agents to autonomously acquire professional skills via a GUI-DFS exploration algorithm, incorporating planning-, action-, and feedback-LLMs.
- The framework constructs a verifiable Skill Set of unit functions and composite procedures, utilizing a LoRA-tuned Fast Planner to generate complete plans in a single forward pass.
- It implements a Skill-Boundary Check to optimize efficiency by predicting capability limits and leverages a database of fine-grained action primitives for precise spatial manipulation in complex UIs.

---

[ADAPTIVE COLLABORATION WITH HUMANS: METACOGNITIVE POLICY OPTIMIZATION FOR MULTI-AGENT LLMS WITH CONTINUAL LEARNING](http://arxiv.org/abs/2603.07972)

- HILA (Human-In-the-Loop Multi-Agent Collaboration): introduces a framework for adaptive human-agent collaboration that equips multi-agent systems with a metacognitive policy to strategically defer to human expertise, including decision-reasoning and execution-reasoning agents.
- The system utilizes Dual-Loop Policy Optimization (DLPO), where an inner loop refines deferral behavior via Group Relative Policy Optimization (GRPO) and an outer loop enables continual learning from expert demonstrations.
- It incorporates a structured cognitive state space and a strategic action space (EVAL, CREATE, DEFER) to balance autonomous problem-solving with targeted human intervention for long-term capability growth.

---

[Advancing Automated Algorithm Design via Evolutionary Stagewise Design with LLMs](http://arxiv.org/abs/2603.07970)

- EvoStage (Evolutionary Stagewise Algorithm Design): introduces an evolutionary paradigm for automated algorithm design that decomposes complex tasks into sequential stages, utilizing a multi-agent system with LLM coder- and coordinator-agents.
- The framework employs a global-local perspective mechanism to balance stage-specific optimization with overall algorithm performance through specialized exploration and enhancement operators.
- By integrating real-time intermediate feedback, EvoStage mitigates LLM hallucinations and achieves state-of-the-art results in industrial-scale tasks like chip placement and Bayesian optimization.

---

[Listening with the Eyes: Benchmarking Egocentric Co-Speech Grounding across Space and Time](http://arxiv.org/abs/2603.07966)

- EcoG (Egocentric Co-Speech Grounding): introduces a diagnostic benchmark for fine-grained audio-visual alignment in situated collaboration, with Egocentric video input, Synchronized audio input, native omni- and vision-language-LLMs, Output Triplets (What, Where, When), Progressive Cognitive Evaluation protocol (L1-L4), and Temporal anchors (timestamps, ASR).
- The framework requires LLMs to resolve underspecified deictic commands by binding speech phrases to specific gesture strokes on a video timeline.
- It evaluates models across four cognitive levels of increasing complexity, revealing a significant performance gap between humans and current native video-audio LLMs.

---


[RL UNKNOTTER, HARD UNKNOTS AND UNKNOTTING NUMBER](http://arxiv.org/abs/2603.07955)

- The unknotter: introduces a reinforcement learning pipeline for simplifying knot diagrams, with a PPO agent, a macro-action space based on Reidemeister moves, and a stochastic backtrack operator.
- The system utilizes a compact feature vector representing diagram complexity and simplification history to navigate a high-branching move graph.
- It successfully recovers upper bounds for the unknotting number of complex composite knots like 41#910 by combining diagram inflation with learned heuristics.

---

[SWE-Fuse: Empowering Software Agents via Issue-free Trajectory Learning and Entropy-aware RLVR Training](http://arxiv.org/abs/2603.07927)

- SWE-Fuse: introduces an issue-description-aware training framework that fuses issue-guided and issue-free samples to mitigate misleading descriptions, including teacher-driven trajectory generation and student-model optimization.
- The architecture features an issue-free-driven trajectory learning module that constructs multi-step reasoning paths and filters them to prevent git metadata exploitation during supervised fine-tuning.
- It employs an entropy-aware RLVR training module with adaptive clipping to stabilize policy updates and encourage exploration based on the student model's uncertainty levels.

---

[ARES: Adaptive Reasoning Effort Selection for Efficient LLM Agents](http://arxiv.org/abs/2603.07915)

- ARES (Adaptive Reasoning Effort Selection): introduces a framework for per-step dynamic reasoning effort allocation in multi-step agent tasks, utilizing a lightweight router to predict the lowest sufficient thinking level for each action.
- The system employs a multi-phase training pipeline involving automated trajectory collection, minimum effort annotation via LLM judges, and rationale generation to fine-tune the router for plug-and-play integration.
- Experimental results across tool-use, research, and web navigation benchmarks demonstrate up to 52.7% reduction in reasoning token usage while maintaining high task success rates through optimized intra-model thinking levels.

---

[Long-Short Term Agents for Pure-Vision Bronchoscopy Robotic Autonomy](http://arxiv.org/abs/2603.07909)

- Long-Short Term Agents: introduces a vision-only autonomous bronchoscopy framework that utilizes hierarchical agents and a world-model critic to achieve long-horizon navigation without external tracking sensors.
- The architecture includes a short-term reactive agent for continuous low-latency motion control and a long-term strategic agent that integrates preoperative CT data and LLM-based semantic reasoning.
- A world-model critic resolves inter-agent conflicts by predicting future visual states and selecting actions that best match preoperative CT-derived virtual targets.

---

[LeJOT-AutoML: LLM-Driven Feature Engineering for Job Execution Time Prediction in Databricks Cost Optimization](http://arxiv.org/abs/2603.07897)

- LeJOT-AutoML: introduces an agent-driven AutoML framework for Databricks job runtime prediction, which includes feature analyzer-, feature extraction-, and feature evaluation-agents.
- The system utilizes LLMs to automate the feature engineering lifecycle by extracting runtime-derived signals from logs and metadata through a Model Context Protocol toolchain.
- It incorporates an iterative feedback loop with safety gates to ensure code validity and prevent data leakage, significantly reducing engineering overhead while optimizing cloud orchestration costs.

---

[SMGI: A Structural Theory of General Artificial Intelligence](http://arxiv.org/abs/2603.07896)

- SMGI (Structural Model of General Intelligence): introduces a formal meta-model θ that recasts AGI as the controlled evolution of a learning interface rather than fixed hypothesis optimization.
- The framework separates structural ontology from behavioral semantics to enforce obligations of closure, stability, bounded capacity, and evaluative invariance under task transformations.
- It provides a unified structural generalization bound linking PAC-Bayes analysis with Lyapunov stability to certify agentic coherence across non-stationary regimes.

---

[A Lightweight Traffic Map for Efficient Anytime LaCAM*](http://arxiv.org/abs/2603.07891)

- LaCAM* + LTM (Lazy Constraint Addition MAPF with Lightweight Traffic Map): introduces an online mechanism that constructs a dynamic weight map from historical search data to steer agents away from congested regions in Multi-Agent Path Finding, with an anytime configuration-based MAPF solver, a low-level successor configuration generator, a dynamic directed weighted graph, an online weight update mechanism, an iteration restart selection strategy, an LTM-based shortest-path distance, and an LTM-distance-based conflict resolution.
- The framework integrates a Lightweight Traffic Map into the LaCAM* search process, using committed and blocked actions from PIBT executions to update edge weights in real-time.
- Experimental results demonstrate that the method outperforms state-of-the-art guidance-path approaches in both one-shot and planning-and-execution MAPF settings.

---

[Visualizing Coalition Formation: From Hedonic Games to Image Segmentation](http://arxiv.org/abs/2603.07890)

- Hedonic Coalition Mechanism: introduces a diagnostic pipeline for visualizing coalition formation in hedonic games by modeling pixels as agents within an image segmentation framework.
- The system utilizes a resolution parameter to modulate coalition granularity, mapping equilibrium transitions from cohesive structures to fragmented partitions.
- Performance is evaluated using dominant-coalition and recoverable-union metrics to distinguish between structural fragmentation and intrinsic mechanism failure on benchmark datasets.

---

[SPIRAL: A Closed-Loop Framework for Self-Improving Action World Models via Reflective Planning Agents](http://arxiv.org/abs/2603.08403)

- SPIRAL (Self-improving Planning and Iterative Reflective Action World Model-ing closed-Loop): introduces a closed-loop agentic framework for controllable long-horizon video generation, with PlanAgent (VLM-driven goal decomposition and planning), World Model (video diffusion policy for execution), CriticAgent (VLM-based evaluation and reward generation), World Memory (historical context and state preservation), Dual-loop feedback (iterative local refinement and global replanning), and Progressive-Evolution (RL-based policy optimization via GRPO), and includes planning-, execution- and evaluation-agents.
- The framework utilizes a think-act-reflect cycle where the PlanAgent decomposes goals into sub-actions, while the CriticAgent evaluates generated video segments to trigger local refinement or global replanning.
- It leverages Progressive-Evolution via Group Relative Policy Optimization (GRPO) to iteratively improve the World Model's performance using rewards derived from the CriticAgent.

---

[SynPlanResearch-R1: Encouraging Tool Exploration for Deep Research with Synthetic Plans](http://arxiv.org/abs/2603.07853)

- SynPlanResearch-R1: introduces a plan-guided data synthesis framework that improves research agent exploration by generating synthetic trajectories guided by randomized tool plans and injected reasoning cues, with a Tool-Plan Generator, a Large Reasoning Model, Cue Injection, Filtering and Quality Control, a Rewriting Model, Cold-start SFT, Reinforcement Learning, and Web Search and Crawling Tools.
- The pipeline includes a synthesis-LRM and a rewriting-agent (Claude) to ensure linguistic naturalness for supervised fine-tuning.
- The framework utilizes Group Relative Policy Optimization (GRPO) with verifiable rewards to optimize the agent's multi-turn tool-use capabilities across diverse knowledge-intensive benchmarks.

---

[Designing a Generative AI-Assisted Music Psychotherapy Tool for Deaf and Hard-of-Hearing Individuals](http://arxiv.org/abs/2603.07963)

- GenAI-assisted music psychotherapy tool: introduces a state-step-based prompting framework for Deaf and Hard-of-Hearing (DHH) individuals, integrating an LLM-based conversational agent with music generative AI to facilitate therapeutic songwriting.
- The system utilizes a multi-prompt architecture comprising general, state-guidance, and variable-extraction prompts to guide users through therapeutic connection, lyric creation, music generation, and reflective discussion.
- It incorporates a music visualization interface that maps auditory features like pitch and beat to visual elements, enabling DHH users to engage with their compositions through multisensory feedback.

---

[MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games](http://arxiv.org/abs/2603.09022)

- MEMO: introduces a self-play framework that optimizes inference-time context for multi-turn multi-agent LLM games by coupling retention and exploration.
- The framework utilizes a persistent memory bank to store structured insights from self-play trajectories, which are then injected as priors to improve agent performance without updating model weights.
- MEMO incorporates tournament-style prompt evolution and a prioritized replay module to efficiently explore game states and reduce run-to-run variance in evaluation outcomes.

---

[AI Phenomenology for Understanding Human-AI Experiences Across Eras](http://arxiv.org/abs/2603.09020)

- AI Phenomenology: introduces a methodological framework for studying human-AI interaction by prioritizing lived, first-person experiences over traditional usability metrics, utilizing Day, VAPT, Cursor, Progressive transparency interviews, Task-anchored multi-method elicitation, and a Crowdsourced phenomenological archive.
- The framework employs structured elicitation and longitudinal immersion to capture the prereflective layer of human-AI encounters, addressing how users negotiate agency and value alignment over time.
- By treating AI systems as active mediators, the research provides a practical scaffold for researchers to track the co-evolution of human identity and AI capabilities across diverse personal and professional contexts.

---

[MEISSA: Multi-modal Medical Agentic Intelligence](http://arxiv.org/abs/2603.09018)

- MEISSA: introduces, "a lightweight 4B-parameter medical agent that enables offline agentic capabilities through stratified trajectory distillation from frontier models", with all Qwen3-VL-4B (lightweight 4B-parameter foundation model), Stratified Trajectory Supervision (difficulty-aware data generation pipeline), Prospective-Retrospective Supervision (stable execution policy learning), Agent Environments (diverse sources for agentic trajectories), Tool Execution Block (interleaved tool-calling and reasoning), Specialist Debate Panel (multi-agent collaborative decision-making), Simulated Environment (multi-turn clinical diagnostic simulation).
- The framework utilizes a unified state-action-observation formalism to generalize across heterogeneous medical environments, including tool-calling, visual reasoning, multi-agent debate, and clinical simulation.
- By distilling agentic behaviors into a single model, MEISSA achieves competitive performance with proprietary frontier models while significantly reducing latency and operational costs for on-premise clinical deployment.

---

[Can AI Agents Generate Microservices? How Far are We?](http://arxiv.org/abs/2603.09004)

- AI Agents for Microservice Generation: evaluates the functional correctness, code quality, and efficiency of LLM-based agents in generating microservices across incremental and clean state scenarios.
- The study employs three distinct LLM-based agents to generate 144 microservice implementations, comparing performance under varying levels of contextual information and prompt strategies.
- Results indicate that while agents can produce maintainable microservices, performance is highly context-dependent, with clean state generation achieving higher integration success and incremental generation benefiting from minimal prompts.

---

[Security Considerations for Multi-agent Systems](http://arxiv.org/abs/2603.09002)

- MAS: introduces a systematic threat taxonomy and empirical security evaluation for multi-agent AI systems, addressing vulnerabilities qualitatively distinct from singular LLMs.
- The paper constructs a comprehensive technical knowledge base of production multi-agent architectures to identify 193 distinct security threats across nine risk categories.
- It provides the first empirical cross-framework comparison of sixteen security and governance frameworks, offering evidence-based guidance for securing agentic AI deployments.

---

[Arbiter: Detecting Interference in LLM Agent System Prompts](http://arxiv.org/abs/2603.08993)

- Arbiter: introduces a framework for detecting interference in LLM agent system prompts by combining formal rule-based directed evaluation with multi-model undirected scouring.
- The framework utilizes a two-layer parser to generate an abstract syntax tree (AST) for system prompts, enabling structural analysis and version-aware diffing of prompt artifacts.
- Empirical analysis across three major coding agents demonstrates that prompt architecture—monolithic, flat, or modular—strongly correlates with specific classes of failure modes.

---

[Semantic Level of Detail: Multi-Scale Knowledge Representation via Heat Kernel Diffusion on Hyperbolic Manifolds](http://arxiv.org/abs/2603.08965)

- SLoD (Semantic Level of Detail): introduces a framework for continuous multi-scale knowledge representation by defining a zoom operator via heat kernel diffusion on the Poincaré ball.
- The framework utilizes a boundary scanner to automatically detect qualitative abstraction transitions in knowledge graphs by analyzing spectral gaps in the graph Laplacian.
- SLoD enables AI agents to dynamically navigate hierarchical memory structures by selecting appropriate resolution scales without requiring manual parameter tuning.

---

[Multi-Agent Memory from a Computer Architecture Perspective: Visions and Challenges Ahead](http://arxiv.org/abs/2603.10062)

- Multi-Agent Memory Architecture Framework: introduces a hierarchical memory model for LLM agents that distinguishes between Agent I/O Layer, Agent Cache Layer, and Agent Memory Layer to address scalability and performance bottlenecks.
- The framework identifies critical protocol gaps in multi-agent systems, specifically proposing Agent Context IO, Agent Cache Sharing, and Agent Memory Access to manage inter-agent communication and data consistency.
- By framing multi-agent memory as a computer architecture problem, the paper highlights the necessity of explicit consistency models to handle concurrent read/write operations and semantic conflicts in evolving agent environments.

---

[Fly, Track, Land: Infrastructure-less Magnetic Localization for Heterogeneous UAV–UGV Teaming](http://arxiv.org/abs/2603.08926)

- MI localization framework: introduces an infrastructure-less, magneto-inductive localization system for heterogeneous UAV-UGV teaming, enabling centimeter-level precision landing on a moving ground platform.
- The system utilizes frequency-multiplexed magnetic beacons on the UGV and a lightweight receiving coil on the nano-UAV to provide robust, short-range relative pose estimation independent of external infrastructure.
- The onboard estimation pipeline integrates magnetic measurements with IMU, optical flow, and UWB data within an EKF to maintain stable tracking and landing under strict SWaP constraints.

---

[Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents](http://arxiv.org/abs/2603.10060)

- NabaOS: introduces a lightweight verification framework that detects LLM hallucinations by cross-referencing agent claims against cryptographically signed tool execution receipts.
- The framework utilizes an epistemic classification system based on Nyāya Śāstra to categorize claims by their source, providing users with actionable trust signals instead of binary verification.
- NabaOS achieves a 91% hallucination detection rate with minimal latency, offering a practical alternative to computationally expensive zero-knowledge proofs for interactive AI agents.

---

[A DECENTRALIZED FRONTIER AI ARCHITECTURE BASED ON PERSONAL INSTANCES, SYNTHETIC DATA, AND COLLECTIVE CONTEXT SYNCHRONIZATION](http://arxiv.org/abs/2603.08893)

- DFMA (H3LIX Decentralized Frontier Model Architecture): introduces a decentralized AI framework where intelligence emerges from the interaction of locally operating AI instances through the propagation of synthetic learning signals within a shared Collective Context Field.
- The architecture replaces centralized parameter synchronization with contextual signal propagation, utilizing Personal AI Instance Layer, Synthetic Learning Layer, Synchronization Layer, and Collective Context Layer to enable continuous collective learning.
- DFMA integrates Energy-Adaptive Model Evolution to align distributed learning workloads with renewable energy availability, thereby reducing the environmental footprint of large-scale AI development.

---

[Quantifying the Accuracy and Cost Impact of Design Decisions in Budget-Constrained Agentic LLM Search](http://arxiv.org/abs/2603.08877)

- BCAS: introduces a model-agnostic evaluation harness that quantifies the accuracy and cost impact of agentic design decisions under explicit search and token constraints.
- The framework utilizes a stateful execution loop where an LLM agent performs iterative reasoning, tool selection, and budget-aware retrieval to solve multi-hop QA tasks.
- Experimental results across six LLMs demonstrate that increasing search depth up to three steps provides consistent accuracy gains, while hybrid retrieval with re-ranking offers the most reliable performance improvement.

---

[LLM-Agent Interactions on Markets with Information Asymmetries](http://arxiv.org/abs/2603.08853)

- LLM-Agent Interactions on Markets with Information Asymmetries: investigates how LLM-based agents coordinate in credence goods markets by simulating interactions under varying institutional frameworks and social preference objectives.
- The study utilizes GPT-5.1 agents to analyze market outcomes, finding that LLMs often diverge from standard economic theory by exhibiting entrenched fraud patterns in the absence of explicit other-regarding preferences.
- Results indicate that while repeated interactions and social preferences can improve consumer participation, the impact of institutional interventions like verifiability and reputation remains highly context-dependent and less predictable than in human experiments.

---

[LDP: An Identity-Aware Protocol for Multi-Agent LLM Systems](http://arxiv.org/abs/2603.08852)

- LDP (LLM Delegate Protocol): introduces an AI-native communication protocol that elevates model-level properties to first-class primitives to enable efficient and governable multi-agent delegation.
- The framework utilizes LDP Router, Delegate Identity Cards, Progressive Payload Modes, Governed Sessions, Structured Provenance, and Trust Domains to optimize routing, reduce token overhead, and enforce security.
- Empirical results demonstrate that LDP achieves 12x lower latency on easy tasks through specialization-aware routing and reduces token consumption by 37% via semantic frame payloads compared to standard protocols.

---

[Scale-Plan: Scalable Language-Enabled Task Planning for Heterogeneous Multi-Robot Teams](http://arxiv.org/abs/2603.08814)

- Scale-Plan: introduces a scalable framework that filters task-relevant environmental information using an action graph to reduce combinatorial complexity for multi-robot planning.
- The framework utilizes an Action-Object Proposer and Graph Search to extract minimal task-relevant subsets, which are then processed by a structured pipeline for Task Decomposition, Task Allocation, and Plan Integration.
- Scale-Plan avoids explicit PDDL problem file generation, instead translating natural language instructions directly into executable plans for heterogeneous teams within the AI2-THOR simulator.

---

[VisionCreator-R1: A Reflection-Enhanced Native Visual-Generation Agentic Model](http://arxiv.org/abs/2603.08812)

- VisionCreator-R1: introduces a native visual generation agent that integrates explicit reflection into a UTPCR framework to enable self-correction in long-horizon workflows, utilizing VLM, UTPCR, Tools Pool, RPCO, VCR-SFT, VCR-RL, GRPO, Plan Reward, Reflection Reward, Format Reward, Tool Call Reward, and Result Reward.
- The paper identifies a structural variance asymmetry in reinforcement learning where planning rewards are stable but reflection rewards are hindered by high-variance stochasticity in visual generation, necessitating a decoupled-then-fused training approach.
- By isolating reflection learning in low-noise single-image settings before synergizing it with planning via multi-task reinforcement learning, the model achieves superior performance on single-image, multi-image, and image-to-image tasks.

---

[Test-Driven AI Agent Definition (TDAD): Compiling Tool-Using Agents from Behavioral Specifications](http://arxiv.org/abs/2603.08806)

- TDAD (Test-Driven AI Agent Definition): introduces a methodology that treats LLM agent prompts as compiled artifacts by converting behavioral specifications into executable tests and iteratively refining prompts until they pass.
- The framework utilizes TestSmith, PromptSmith, and MutationSmith to automate the development and verification of tool-using agents while mitigating specification gaming through hidden test splits and semantic mutation testing.
- TDAD provides a rigorous benchmark, SpecSuite-Core, to evaluate agent compilation workflows, achieving high compilation success rates and regression safety across diverse agent domains.

---

[Reachability-based Temporal Logic Verification for Reliable LLM-guided Human-Autonomy Teaming](http://arxiv.org/abs/2603.08633)

- HAT framework: introduces a reliable human-autonomy teaming architecture that utilizes an LLM for natural language translation and an STL Feasibility Filter (SFF) to formally verify mission feasibility before planning.
- The SFF decomposes complex STL specifications into simpler subformulas, enabling parallelized reachability analysis to identify infeasible components and provide informative feedback to the human operator.
- By integrating formal verification with LLM-based translation, the framework ensures safety and improves computational efficiency in MILP-based mission planning for autonomous systems.

---

[Model-Free DRL Control for Power Inverters: From Policy Learning to Real-Time Implementation via Knowledge Distillation](http://arxiv.org/abs/2603.07964)

- Model-Free DRL Control Framework: introduces a teacher-student policy distillation architecture to compress complex Deep Reinforcement Learning policies into lightweight neural networks for real-time power inverter control.
- The framework utilizes an error energy-guided hybrid reward mechanism to constrain exploration within stable regions and mitigate convergence instability.
- Adaptive importance weighting is integrated into the distillation process to prioritize transient dynamics over steady-state data, ensuring high-quality knowledge transfer.

---

[SPREAD: Subspace Representation Distillation for Lifelong Imitation Learning](http://arxiv.org/abs/2603.08763)

- SPREAD: introduces a geometry-aware lifelong imitation learning framework that preserves intrinsic task manifolds by aligning low-rank subspace representations of consecutive policies using SVD.
- The framework utilizes a confidence-guided policy distillation strategy that applies KL divergence to the top-M most probable action samples to ensure robust behavioral transfer and stability.
- Experimental results on the LIBERO benchmark demonstrate that SPREAD effectively mitigates catastrophic forgetting and improves knowledge transfer across sequential robotic manipulation tasks.

---

[VSearcher: Long-Horizon Multimodal Search Agent via Reinforcement Learning](http://arxiv.org/abs/2603.02795)

- VSearcher: introduces a post-training framework that transforms static multimodal models into autonomous agents capable of long-horizon, multi-turn tool use in real-world web environments.
- The framework utilizes Iterative Injection-based Data Synthesis, Rejection Sampling Fine-tuning, and Reinforcement Learning to enhance agentic browsing skills.
- The authors also propose MM-SearchExam, a challenging benchmark designed to evaluate the long-horizon reasoning and tool-use capabilities of multimodal agents.

---

#### 8th March 2026

[Intentional Deception as Controllable Capability in LLM Agents](http://arxiv.org/abs/2603.07848)

- Adversarial Agent Architecture: introduces a system for intentional behavioral manipulation in multi-agent environments, with an inference module, an opportunity identification module, and a responder including action-isolation and persuasive-framing reasoning models.
- The architecture leverages profile inversion to identify actions detrimental to a target's values and uses reasoning-specialized LLMs to frame those actions as appealing to the target's specific motivations.
- Findings show that 88.5% of successful deceptions employ misdirection through strategic framing of true statements, allowing the system to bypass fact-checking defenses and RLHF safety training.

---

[How Neurotypical and Autistic Children Interact Nonverbally with Anthropomorphic Agents in Open-Ended Tasks](http://arxiv.org/abs/2603.07843)

- Wizard-of-Oz Study Setup: introduces an empirical framework for analyzing unconstrained nonverbal interactions between children and anthropomorphic agents, with Wizard (human teleoperator controlling agent responses), Virtual Characters (six anthropomorphic agents with varied morphologies), Animaze (character animation and facial mapping tool), Webcam Motion Capture (upper-body and finger tracking software), Zoom (real-time visual interaction platform), and Data Collection & Analysis (segmentation, annotation, and thematic analysis).
- The methodology captures 563 interaction instances to identify 141 unique nonverbal behaviors across physical, emotional, and social categories for neurotypical and autistic participants.
- The research identifies unique child-centered behaviors such as environmental manipulation and repetitive sensorimotor actions that are absent in adult-focused interaction studies.

---

[Gradient Iterated Temporal-Difference Learning](http://arxiv.org/abs/2603.07833)

- Gi-TD (Gradient Iterated Temporal-Difference) learning: introduces a gradient-based reinforcement learning algorithm that learns a sequence of action-value functions in parallel by directly minimizing the sum of Bellman errors, with Q-networks (parallel action-value function sequence), H-networks (gradient correction for stochastic targets), Shared Feature Extractor (common state representation learning), Replay Buffer (experience storage for updates), Target Networks (stable bootstrapping reference points), and Bellman Operator (recursive value update definition).
- The framework overcomes the double sampling problem by utilizing auxiliary H-networks to approximate gradient correction terms, allowing for unbiased stochastic gradient descent on the Bellman error objective. 
- Evaluations across Atari and MuJoCo benchmarks demonstrate that Gi-TD achieves competitive learning speeds compared to semi-gradient methods while maintaining superior stability in high update-to-data regimes.

---

[Uncertainty Mitigation and Intent Inference: A Dual-Mode Human-Machine Joint Planning System](http://arxiv.org/abs/2603.07822)

- Dual-Mode Human-Machine Joint Planning System: introduces an end-to-end framework for proactive human-robot collaboration, featuring an uncertainty-mitigation module for resolving semantic ambiguities and an intent-aware module for uncommunicated coordination.
- The architecture integrates a VLM-based perception pipeline using Grounded-SAM and 3D Gaussian Splatting to maintain a structured semantic map of the environment for real-time situational awareness.
- The system includes dialogue-understanding, plan-prototyping, and target-grounding LLMs to minimize interaction costs and adapt robot behavior to evolving human goals in real-time.

---

[ProgAgent: A Continual RL Agent with Progress-Aware Rewards](http://arxiv.org/abs/2603.07784)

- ProgAgent: introduces a system for scalable lifelong robotic learning, with ProgAgent (continual reinforcement learning framework), Progress Prediction Network (perceptual model for task progress), Adversarial Push-back Refinement (regularization for out-of-distribution states), JAX-Native Architecture (high-throughput JIT-compiled system), Coreset Replay (memory buffer for past experiences), Synaptic Intelligence (parameter importance regularization), and PPO Policy Optimizer (policy update mechanism).
- The framework derives dense, shaped rewards from unlabeled expert videos through a learned state-potential function while suppressing overconfident predictions on out-of-distribution states via an adversarial push-back loss.
- It integrates PPO with coreset replay and synaptic intelligence within a fully differentiable, JIT-compiled pipeline to enable massively parallel rollouts and efficient multi-task adaptation.

---

[Robust Cooperative Output Regulation of Discrete-Time Heterogeneous Multi-Agent Systems](http://arxiv.org/abs/2603.07783)

- RCORP (Robust Cooperative Output Regulation Problem): introduces a distributed internal model-based control framework for discrete-time uncertain heterogeneous multi-agent systems, with all Exosystem (generates reference signals and disturbances), Heterogeneous Multi-Agent System (collection of agents with varying dimensions), Distributed Dynamic State Feedback Control Law (local controller using neighbor information), Internal Model (replicates exosystem dynamics for regulation), Communication Network (models information exchange via directed graphs), and Onboard Sensors (measure relative outputs for tracking).
- The architecture utilizes global and agent-wise local design methods based on linear matrix inequalities to synthesize structured control gains that ensure the nominal closed-loop system matrix is Schur.
- The proposed method enables robust tracking and disturbance rejection in multi-agent systems without requiring the exchange of controller states between neighboring agents.

---

[A Novel Multi-Agent Architecture to Reduce Hallucinations of Large Language Models in Multi-Step Structural Modeling](http://arxiv.org/abs/2603.07728)

- Multi-agent architecture for structural modeling: introduces a modular framework to automate finite element analysis through task decomposition, featuring problem analysis-, construction planning-, node-, element-, load assignment-, and code translation-agents.
- The architecture utilizes GPT-OSS 120B for complex reasoning and Llama-3.3 70B for deterministic mapping, incorporating parallel execution and consistency checkpoints to reduce hallucinations in LLMs.
- It transforms natural language descriptions into executable OpenSeesPy scripts, achieving high accuracy and scalability for complex 2D frame structures.

---

[YAQIN: Culturally Sensitive, Agentic AI for Mental Healthcare Support Among Muslim Women in the UK](http://arxiv.org/abs/2603.07709)

- YAQIN: introduces a culturally and faith-sensitive agentic AI framework for mental healthcare support among Muslim women, featuring a Retrieval-Augmented Generation (RAG) pipeline that integrates Islamic psychological concepts with therapeutic interactions.
- The architecture utilizes a "Tafakkur" journaling tool and a faith-aware chatbot powered by LLMs to provide empathetic reflection, psycho-educational insights, and curated Quranic references.
- Developed through iterative co-design, the system facilitates therapeutic readiness and emotional validation by offering anonymous, on-demand, and spiritually grounded support that bridges gaps in conventional clinical care.

---

[Deep Incentive Design with Differentiable Equilibrium Blocks](http://arxiv.org/abs/2603.07705)

- DID (Deep Incentive Design): introduces a differentiable framework for automated incentive design by composing a mechanism generator with a pretrained differentiable equilibrium block to solve mathematical programs with equilibrium constraints.
- The architecture utilizes game-theoretically equivariant neural networks to handle varying game sizes and respect domain symmetries during the generation of induced games.
- By backpropagating through the equilibrium block, the system learns design policies that generalize across distributions of problem instances such as contract design and machine scheduling.

---

[Continuous-Time Heterogeneous Agent Models with Recursive Utility and Preference for Late Resolution](http://arxiv.org/abs/2603.07782)

- Continuous-Time Heterogeneous Agent Model: introduces a mean field game framework for continuous-time heterogeneous agent models, with Epstein-Zin recursive utility, HJB system, and FPK equation.
- The approach establishes the existence and uniqueness of constrained viscosity solutions for the HJB system under the assumption of late resolution of uncertainty.
- The research investigates the existence of stationary equilibria in Aiyagari and Huggett models, providing numerical analysis of consumption and saving policies.

---

[UniUncer: Unified Dynamic–Static Uncertainty for End-to-End Driving](http://arxiv.org/abs/2603.07686)

- UniUncer: introduces a unified uncertainty framework that jointly estimates and utilizes uncertainty for both static and dynamic scene elements, with multi-view images (raw sensor input), encoder (visual feature extraction), static branch (vectorized map processing), dynamic branch (vectorized agent processing), uncertainty estimation (Laplace parameter prediction), uncertainty fusion (query refinement via attention), uncertainty-aware gate (adaptive historical data modulation), ego status (historical vehicle state), temporal queries (historical perception features), and planning module (trajectory generation).
- The framework converts deterministic regression heads into probabilistic Laplace regressors to output per-vertex location and scale parameters for vectorized entities.
- An adaptive gating mechanism modulates reliance on historical ego status and temporal perception queries based on current scene uncertainty levels to improve planning decisions.

---

[Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers](http://arxiv.org/abs/2603.07670)

- Agent Memory (Write–Manage–Read Loop): introduces a structured account of memory design for LLM agents, formalizing the system as a recursive feedback loop between a policy and a managed storage state that includes decision-making and memory-management LLMs.
- The framework categorizes memory mechanisms into context-resident compression, retrieval-augmented stores, reflective self-improvement, hierarchical virtual context, and policy-learned management.
- The research evaluates the transition from static recall benchmarks to multi-session agentic tests that measure the impact of memory on downstream decision-making performance.

---

[Multi-Agent Off-World Exploration for Sparse Evidence Discovery via Gaussian Belief Mapping and Dual-Domain Coverage](http://arxiv.org/abs/2603.07650)

- MAIPP (Multi-Agent Informative Path Planning): introduces a cooperative visual search system for off-world exploration, with Interest GP, Risk GP, and intent-aware coordination.
- The system utilizes a dual-domain coverage objective to prioritize high-priority areas while maintaining background exploration to mitigate prior biases.
- It incorporates a two-stage safety mechanism combining a risk field with hard recoverability constraints to prevent mission-ending states in hazardous terrains.

---

[PanoDP: Learning Collision-Free Navigation with Panoramic Depth and Differentiable Physics](http://arxiv.org/abs/2603.07644)

- PanoDP (Panoramic Depth and Differentiable Physics): introduces a communication-free navigation framework for quadrotors that couples 360-degree panoramic depth perception with differentiable-physics-based training signals.
- The architecture utilizes a circular CNN encoder to process equirectangular panoramas and a GRU-based recurrent policy to infer obstacle motion from temporal context.
- By backpropagating through a differentiable physics simulator, the system optimizes policies using dense safety and feasibility objectives rather than sparse terminal collision rewards.

---

[Coordination Games on Multiplex Networks: Consensus, Convergence, and Stability of Opinion Dynamics](http://arxiv.org/abs/2603.07633)

- Multiplex Coordination Games: introduces a unified framework for analyzing opinion dynamics across interconnected social networks, which includes merged-layer and switching-layer coupling mechanisms to model simultaneous and sequential influences.
- The approach utilizes spectral tools and random-walk theory to establish that cross-layer interactions can either accelerate or hinder consensus depending on structural alignment.
- Stability analysis reveals how the system responds to network perturbations and layer imbalances, highlighting the role of influential nodes in shaping the final agreement.

---

[SMAT: Staged Multi-Agent Training for Co-Adaptive Exoskeleton Control](http://arxiv.org/abs/2603.07618)

- SMAT (Staged Multi-Agent Training): introduces a four-stage curriculum learning protocol for co-adaptive hip exoskeleton control, including human- and exoskeleton-agents.
- The framework progressively introduces device mass and active assistance across four distinct stages to mitigate non-stationarity and improve training stability in musculoskeletal simulations.
- Validated through sim-to-real transfer, the system demonstrates reduced muscle activation and stable torque profiles across multiple subjects without requiring subject-specific retraining.

---

[Beyond Semantic Similarity: Open Challenges for Embedding-Based Creative Process Analysis Across AI Design Tools](http://arxiv.org/abs/2603.07611)

- Embedding-Based Creative Process Analysis: introduces a framework for evaluating AI-mediated creativity by projecting discrete design actions into a shared representational space to construct fuzzy linkographs, with Neural Embedding Models, Fuzzy Linkograph, Process Metrics, LLM Intermediaries, Multimodal Traces, and Agentic Generation Loop.
- The system leverages LLMs to segment interaction logs around meaningful conceptual shifts and judge creative relevance based on task framing and preceding trajectories.
- The research addresses the limitations of fixed embedding models in detecting creative pivots and proposes methods to integrate multimodal traces and evaluate agentic AI systems.

---

[From Logs to Agents: Reconstructing High-Level Creative Workflows from Low-Level Raw System Traces](http://arxiv.org/abs/2603.07609)

- Workflow reconstruction pipeline: introduces an automated system to translate noisy low-level system trace logs into structured high-level behavioral workflow graphs for generative design, with Semantic Filtering (removes low-level system noise), Heuristic Classification (categorizes events into creative moves), Directed Acyclic Graph (visualizes branching design history), Tokenization (standardizes actions for cross-platform analysis), Markov Chain Analysis (models state transition probabilities), and Process-Aware Agent (LLM-driven assistant with workflow memory).
- The system constructs a Directed Acyclic Graph to visualize branching design evolution and applies Markov Chain analysis to model state transition probabilities for predictive insights.
- This structured workflow history serves as a prerequisite for process-aware agents, allowing LLM-driven assistants to infer user intent and provide context-grounded design rationales.

---

[Deep Research for Recommender Systems](http://arxiv.org/abs/2603.07605)

- RecPilot: introduces a multi-agent framework for deep research in recommendation, with a User Trajectory Simulation Agent (simulates exploration behaviors), a Report Generation Agent (synthesizes findings into reports), Multi-Aspect Interest Decomposition (breaks down user interests), Parallel Ranking (evaluates items across aspects), a Self-Evolution Module (optimizes preferences over time), Rubric Memory (stores structured attribute scores), Experience Memory (stores textual preference cues), and Reinforcement Learning (GRPO) (optimizes simulation via rewards).
- The framework utilizes generative modeling optimized via Group Relative Policy Optimization (GRPO) with model-free process rewards to emulate realistic user exploration-to-decision pathways.
- It incorporates a dual-channel preference mechanism using structured rubrics and experience-based memories that autonomously evolve through user feedback to provide personalized, aspect-aware decision support.

---

[AgentRaft: Automated Detection of Data Over-Exposure in LLM Agents](http://arxiv.org/abs/2603.07557)

- AgentRaft: introduces an automated framework for detecting Data Over-Exposure (DOE) risks in LLM agents, with Cross-Tool Function Call Graph (FCG) Generation (models tool interactions), User Prompt Synthesis (generates deterministic execution triggers), Data Over-Exposure Detection (identifies unauthorized data transmissions), Multi-LLM Voting Committee (judges violations using regulations), Runtime Taint Tracking (monitors fine-grained data flows), and Agent Run-time Environment (executes prompts for analysis).
- The framework includes FCG generation-, prompt synthesis- and privacy judging-LLMs to map tool dependencies and verify data-flow compliance.
- It utilizes a multi-LLM voting mechanism grounded in global privacy regulations like GDPR and CCPA to distinguish between functional necessity and unintended data leaks.

---

[COOL-MC: VERIFYING AND EXPLAINING RL POLICIES FOR MULTI-BRIDGE NETWORK MAINTENANCE](http://arxiv.org/abs/2603.07546)

- COOL-MC (COmprehensive tool for reinforcement Learning and Model Checking): introduces a framework for formal verification and explanation of infrastructure maintenance policies, with RL Maintenance Agent (PPO-based decision-making policy), Multi-bridge MDP (PRISM-encoded bridge deterioration environment), Induced DTMC (reachable state space representation), Probabilistic Model Checker (Storm engine for property verification), and Explainability Suite (tools for policy interpretation).
- The framework constructs an induced DTMC to represent the reachable state space under the trained policy, enabling formal verification of safety-critical properties like bridge failure probability and budget exhaustion.
- It utilizes gradient-based saliency and counterfactual action replacement to identify policy biases and temporal dependencies that are not visible through standard training metrics.

---

[DreamSAC: Learning Hamiltonian World Models via Symmetry Exploration](http://arxiv.org/abs/2603.07545)

- DreamSAC (Dream with Symmetry-Aware Curiosity): introduces an unsupervised reinforcement learning framework that learns physics-grounded world models by combining a Hamiltonian-based dynamics prior with an active symmetry exploration strategy.
- The system utilizes a G-invariant Lie Transformer and a self-supervised contrastive objective to identify invariant physical states from raw pixel observations while factoring out viewpoint variations.
- A dual integration strategy employing Euler and Symplectic Leapfrog integrators ensures training stability while maintaining long-term physical consistency for robust extrapolative generalization.

---

[TableMind++: An Uncertainty-Aware Programmatic Agent for Tool-Augmented Table Reasoning](http://arxiv.org/abs/2603.07528)

- TableMind++ (An Uncertainty-Aware Programmatic Agent for Tool-Augmented Table Reasoning): introduces an autonomous agentic framework for complex table reasoning, with all Base LLM, SFT, RFT, RAPO, Dual-Memory Bank, Memory-Guided Plan Pruning, Confidence-Based Action Refinement, Dual-Weighted Trajectory Aggregation, Code Sandbox, and internalized planning, action, and reflection roles.
- The framework employs a two-stage training strategy combining supervised fine-tuning and reinforcement learning via Rank-Aware Policy Optimization to establish human-like reasoning capabilities within a lightweight LLM.
- It mitigates hallucinations through memory-guided plan pruning for epistemic uncertainty and confidence-based action refinement for aleatoric uncertainty during the inference phase.

---

[From Thinker to Society: Security in Hierarchical Autonomy Evolution of AI Agents](http://arxiv.org/abs/2603.07496)

- HAE (Hierarchical Autonomy Evolution): introduces a multi-tiered security framework that categorizes AI agent vulnerabilities across cognitive, executional, and collective autonomy levels, and includes manager- and worker-agents.
- The framework identifies how vulnerabilities propagate through a "Cognition-Execution-Diffusion" chain, transforming internal reasoning errors into real-world kinetic breaches and network-wide systemic failures.
- It establishes a systematic taxonomy of threats including cognitive hijacking, confused deputy attacks, and viral infection while identifying critical defense gaps in large-scale agent ecosystems.

---

[Give Them an Inch and They Will Take a Mile: Understanding and Measuring Caller Identity Confusion in MCP-Based AI Systems](http://arxiv.org/abs/2603.07473)

- MCPAuthChecker: introduces a security analysis framework for detecting caller identity confusion in Model Context Protocol (MCP) based systems, utilizing program dependency reconstruction, path-sensitive authorization evaluation, and selective dynamic validation to identify insecure authorization reuse.
- The system identifies execution-trigger points and traces control flow to determine if authorization is strictly bound to the invoking caller or incorrectly cached as persistent server-level state.
- Large-scale measurement of 6,137 MCP servers demonstrates that 46.4% are vulnerable to authorization reuse, enabling unauthorized entities to inherit credentials for sensitive system-level operations.

---

[Agentic AI–Driven UAV Network Deployment: A LLM–Enhanced Exact Potential Game Approach](http://arxiv.org/abs/2603.07456)

- Agentic AI-driven UAV network deployment framework: introduces a dual spatial-scale optimization approach for Unmanned Aerial Vehicular Networks (UAVNs) by decomposing complex Mixed-Integer Nonlinear Programming (MINLP) problems into discrete link and continuous parameter sub-problems.
- The system utilizes L3-EPG for sparse topology configuration and AG-EPG for joint optimization of UAV coordinates, transmission power, and ground user association.
- A RAG-enhanced LLM acts as a knowledge-driven decision enhancer to automatically generate scenario-adaptive utility weights, reducing reliance on manual parameter tuning.

---

[HLER: HUMAN-IN-THE-LOOP ECONOMIC RESEARCH VIA MULTI-AGENT PIPELINES FOR EMPIRICAL DISCOVERY](http://arxiv.org/abs/2603.07444)

- HLER (Human-in-the-Loop Economic Research): introduces a multi-agent architecture for automating empirical economics research through dataset-aware hypothesis generation and human-supervised decision gates, with a Human PI (provides oversight and decision-making), an Orchestrator (coordinates agents and manages state), RunState (shared object recording intermediate outputs), a Data Audit Agent (validates dataset structure and schema), a Data Profiling Agent (analyzes statistical properties and diagnostics), a Question Agent (generates dataset-constrained research questions), a Data Agent (retrieves and merges relevant data), an Econometrics Agent (constructs and executes analysis models), a Paper Agent (generates full research manuscripts), and a Reviewer Agent (evaluates drafts and requests revisions).
- The system utilizes a two-loop architecture featuring a question quality loop for feasible hypothesis selection and a research revision loop for iterative manuscript improvement.
- It integrates specialized agents for data auditing, profiling, econometric analysis, and automated review to ensure research is grounded in available data and statistically sound.

---

[Data Agent: Learning to Select Data via End-to-End Dynamic Optimization](http://arxiv.org/abs/2603.07433)

- Data Agent: introduces an end-to-end dynamic data selection framework that formulates data selection as a training-aware sequential decision-making problem, utilizing a PPO-based actor-critic agent to learn sample-wise selection policies.
- The framework employs a composite reward integrating loss-based difficulty and confidence-based uncertainty signals, balanced by a tuning-free adaptive weighting mechanism to capture evolving data utility.
- Experiments across image classification, object detection, and LLM instruction tuning demonstrate training acceleration and cost reduction of over 50% while maintaining model performance.

---

[GENERALIZATION IN ONLINE REINFORCEMENT LEARNING FOR MOBILE AGENTS](http://arxiv.org/abs/2603.07432)

- RL training system: introduces a framework for mobile agents by integrating Group Relative Policy Optimization (GRPO) with a containerized, asynchronous rollout collection system.
- The architecture employs a Vision-Language Model (VLM) policy that generates chain-of-thought reasoning and actions based on screenshots and interaction history.
- The system utilizes Docker-based resource isolation and HTTP-based communication to decouple the trainer from multiple parallel Android emulators, eliminating synchronization bottlenecks.

---

[AUTOCONTROL ARENA: Synthesizing Executable Test Environments for Frontier AI Risk Evaluation](http://arxiv.org/abs/2603.07427)

- AUTOCONTROL ARENA: introduces an automated framework for frontier AI risk evaluation based on logic-narrative decoupling, featuring Architect-, Coder-, Target-, and Monitor-agents.
- The architecture separates deterministic environment mechanics into executable Python code while delegating generative narrative dynamics to LLMs to mitigate logic hallucinations.
- It utilizes the X-BENCH benchmark to reveal latent misalignments by systematically varying environmental stress and temptation across seventy diverse safety scenarios.

---

[DualSpec: Accelerating Deep Research Agents via Dual-Process Action Speculation](http://arxiv.org/abs/2603.07416)

- DualSpec: introduces a heterogeneous speculation framework for deep research agents that tailors drafting strategies to action-specific uncertainty, including a small model for reasoning-heavy search tasks, a large LLM for intuitive visit actions, and a large LLM critic for semantic verification.
- The system employs a confidence-based semantic verifier that uses a large LLM as a critic to evaluate the coherence and utility of drafted actions instead of relying on strict token matching.
- By overlapping action execution with speculative reasoning and utilizing a dual-process approach, the framework achieves up to 3.28x end-to-end speedup while maintaining high task accuracy.

---

[Can Large Language Models Keep Up? Benchmarking Online Adaptation to Continual Knowledge Streams](http://arxiv.org/abs/2603.07392)

- OAKS (Online Adaptation to Continual Knowledge Streams): introduces a benchmark for evaluating LLMs on their ability to adapt to streaming, continually updating knowledge across long-horizon contexts, with OAKS (online adaptation benchmark), OAKS-BABI (synthetic state-tracking dataset), OAKS-Novel (literary narrative dataset), LLMs (inference and reasoning engines), RAG (retrieval-based context construction), HippoRAG-v2 (graph-based retrieval framework), MemAgent (linear complexity memory agent), A-Mem (interconnected knowledge network agent), and Thinking Mode (inference-time reasoning scaling).
- The framework utilizes dense annotations to measure model performance across metrics including acquisition latency, distraction susceptibility, and phase miss rate.
- Evaluation of 14 models reveals that while thinking-enabled LLMs enhance stability, current systems struggle with frequent factual updates and long-context distractions.

---

[Toward Epistemic Stability: Engineering Consistent Procedures for Industrial LLM Hallucination Reduction](http://arxiv.org/abs/2603.10047)

- Toward Epistemic Stability: introduces five prompt engineering strategies to reduce LLM hallucinations in industrial settings by improving output consistency and grounding.
- The framework utilizes an internal-baseline LLM-as-Judge evaluation protocol to compare enhanced responses against zero-shot baselines across diverse industrial task scenarios.
- Results indicate that structured context injection and agent specialization significantly improve diagnostic reliability, with M4 achieving the highest performance in baseline trials.

---


## Citation


How to cite my work?



```
@misc{MaattaAutonomousAgents2023,
  author = {Teemu Maatta},
  title = {Autonomous Agents},
  year = {2023},
  howpublished = {\url{http://github.com/tmgthb/Autonomous-Agents}},
  note = {Accessed: YYYY-MM-DD}
}

```



[Back to top](#topofthepage)

