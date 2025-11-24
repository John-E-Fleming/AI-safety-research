# Essential Reading List for Probe Expertise

**Download these before travelling - priority order for limited time**

---

## Priority 1: Core Probe Papers (Read First)

### 1. Understanding Intermediate Layers Using Linear Classifier Probes
**Authors:** Guillaume Alain, Yoshua Bengio  
**Year:** 2016  
**Why essential:** This is THE foundational paper on probes. Introduces the concept of using linear classifiers to understand neural network representations.

**Key concepts:**
- Probes as tools to understand deep networks
- Linear separability increases with layer depth
- Features at each layer become progressively more suitable for classification

**Download:** https://arxiv.org/pdf/1610.01644

---

### 2. Finding Neurons in a Haystack: Case Studies with Sparse Probing
**Authors:** Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, Dimitris Bertsimas  
**Year:** 2023  
**Why essential:** Co-authored by Neel Nanda. Shows how to use k-sparse probes to find interpretable neurons in LLMs. Directly relevant to your project.

**Key concepts:**
- k-sparse linear classifiers for localization
- Early layers use superposition, middle layers have dedicated neurons
- Features become sparser with model scale
- 100+ features probed across 7 models

**Download:** https://arxiv.org/pdf/2305.01610  
**Code:** https://github.com/wesg52/sparse-probing-paper  
**LessWrong post:** https://www.lesswrong.com/posts/yXiu6DBxKKWXC8Ygx/finding-neurons-in-a-haystack-case-studies-with-sparse

---

### 3. Probing Classifiers: Promises, Shortcomings, and Advances
**Authors:** Belinkov (MIT Press Computational Linguistics)  
**Year:** 2022  
**Why essential:** Critical review of probe methodology - covers when probes work, when they fail, and common pitfalls.

**Key concepts:**
- Methodological limitations of probing
- When probes may be misleading
- Best practices for probe design

**Download:** https://direct.mit.edu/coli/article-pdf/48/1/207/2008030/coli_a_00422.pdf

---

## Priority 2: CoT Faithfulness Papers (Core to Your Project)

### 4. Measuring Faithfulness in Chain-of-Thought Reasoning
**Authors:** Tamera Lanham et al. (Anthropic)  
**Year:** 2023  
**Why essential:** Foundational paper on CoT faithfulness. Introduces key experimental methods like "adding mistakes" and "early answering."

**Key concepts:**
- How to test if CoT reasoning is faithful
- Larger models may produce LESS faithful reasoning
- CoT's performance boost varies by task

**Download:** https://arxiv.org/pdf/2307.13702  
**Anthropic version:** https://www-cdn.anthropic.com/827afa7dd36e4afbb1a49c735bfbb2c69749756e/measuring-faithfulness-in-chain-of-thought-reasoning.pdf

---

### 5. Chain-of-Thought Reasoning In The Wild Is Not Always Faithful
**Authors:** Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy  
**Year:** 2025  
**Why essential:** Co-authored by Neel Nanda. Shows unfaithful CoT in realistic settings. This is the dataset source mentioned in your project plan.

**Key concepts:**
- Unfaithful CoT without artificial bias
- "Implicit Post-Hoc Rationalization"
- Rates of unfaithfulness across frontier models (0.04% to 13%)

**Download:** https://arxiv.org/pdf/2503.08679

---

## Priority 3: Reasoning Model Interpretability

### 6. Thought Anchors: Which LLM Reasoning Steps Matter?
**Authors:** Paul C. Bogdan, Uzay Macar, Neel Nanda, Arthur Conmy  
**Year:** 2025 (MATS project)  
**Why essential:** Recent MATS work supervised by Neel. Shows how to analyze reasoning traces at sentence level. Directly relevant methodology.

**Key concepts:**
- Sentence-level analysis of reasoning traces
- Three complementary attribution methods
- "Thought anchors" = critical sentences that guide reasoning
- Receiver head analysis for attention patterns

**Download:** https://arxiv.org/pdf/2506.19143  
**Tool:** https://thought-anchors.com  
**LessWrong:** https://www.lesswrong.com/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter

---

## Priority 4: Conceptual Background

### 7. The Linear Representation Hypothesis (Blog Post)
**Author:** Beren Millidge  
**Year:** 2023  
**Why essential:** Explains why probes work - the hypothesis that neural networks represent features as linear directions.

**Key concepts:**
- Features as linear directions in activation space
- Why linear methods are surprisingly effective
- Implications for interpretability and control

**Read:** https://www.beren.io/2023-04-04-DL-models-are-secretly-linear/

---

## Documentation & Tutorials (Download/Bookmark)

### TransformerLens Documentation
**Essential for:** Your implementation work  
**URL:** https://neelnanda-io.github.io/TransformerLens/

**Key sections to save:**
- Main tutorial
- Hooks guide
- Model loading
- Activation caching

---

### ARENA Tutorials (Mechanistic Interpretability)
**Essential for:** Week 1 foundations  
**URL:** https://arena3-chapter1-transformer-interp.streamlit.app/

**Download sections:**
- 1.2.1: Transform from/to tokens
- 1.2.2: Tokenization
- 1.2.3: Direct Logit Attribution

---

### nnsight Documentation
**Essential for:** Working with reasoning models  
**URL:** https://nnsight.net

---

### Neel Nanda's Mech Interp Blog
**URL:** https://www.neelnanda.io/mechanistic-interpretability

**Recommended posts:**
- "200 Concrete Open Problems in Mechanistic Interpretability"
- "An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers"

---

## Bonus: Recent Relevant Work

### Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity
**Authors:** Austin Meek, Eitan Sprejer, Iván Arcuschin et al.  
**Year:** 2025  
**Download:** https://arxiv.org/pdf/2510.27378

### FaithCoT-Bench: Benchmarking Instance-Level Faithfulness
**Year:** 2025  
**Download:** https://arxiv.org/html/2510.04040v1

---

## Reading Order for Limited Time

**If you only have 2 hours:**
1. Alain & Bengio (2016) - pages 1-5 (core probe concept)
2. Gurnee & Nanda (2023) - sections 1-3 (sparse probing methodology)

**If you have 4 hours:**
- Add: Lanham et al. (2023) - full paper (CoT faithfulness methods)
- Add: Arcuschin et al. (2025) - sections 1-4 (unfaithful CoT examples)

**If you have 8 hours:**
- Add: Thought Anchors paper - full
- Add: Probing Classifiers review - full
- Add: Linear representation hypothesis blog

---

## Quick Reference: Key Insights

### From Alain & Bengio:
> "The degree of linear separability of the features of layers increases as we reach the deeper layers"

### From Gurnee & Nanda:
> "Early layers make use of sparse combinations of neurons to represent many features in superposition, middle layers have seemingly dedicated neurons"

### From Lanham et al.:
> "As models become larger and more capable, they produce less faithful reasoning on most tasks we study"

### From Arcuschin et al.:
> "Unfaithful CoT can occur on realistic prompts with no artificial bias... production models exhibit surprisingly high rates of post-hoc rationalization"

### From Thought Anchors:
> "Certain sentences can have an outsized impact on the trajectory of the reasoning trace and final answer"

---

## Download Checklist

- [ ] Alain & Bengio 2016 (arxiv.org/pdf/1610.01644)
- [ ] Gurnee & Nanda 2023 (arxiv.org/pdf/2305.01610)
- [ ] Probing Classifiers review (MIT Press)
- [ ] Lanham et al. 2023 (arxiv.org/pdf/2307.13702)
- [ ] Arcuschin et al. 2025 (arxiv.org/pdf/2503.08679)
- [ ] Thought Anchors 2025 (arxiv.org/pdf/2506.19143)
- [ ] TransformerLens docs (save offline)
- [ ] ARENA tutorials (save offline)
- [ ] Neel's blog post on mech interp (save offline)
