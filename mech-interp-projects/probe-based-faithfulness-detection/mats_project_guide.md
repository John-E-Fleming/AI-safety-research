# Personal Project 1: Tool Validation for Reasoning Model Interpretability

**Project Goal:** Investigate which interpretability tools (specifically probes) capture reasoning-relevant structure in CoT models, and characterize when they work or fail.

**Timeline:** 4 weeks (Week 1-2: Foundation + Literature, Week 3: Experiments, Week 4: Validation + Write-up)

**Framing:** Exploratory tool validation study (per Neel Nanda's guidance)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Literature Context](#literature-context)
- [Research Questions](#research-questions)
- [Week 1: Foundations](#week-1-foundations-completed)
- [Week 2: Literature Integration & Dataset](#week-2-literature-integration--dataset)
- [Week 3: Core Experiments](#week-3-core-experiments)
- [Week 4: Validation & Write-up](#week-4-validation--write-up)
- [Code Templates](#code-templates)
- [Troubleshooting](#troubleshooting)
- [Success Criteria](#success-criteria)

---

## Project Overview

### Project Reframe (Dec 8, 2025)

This project has been reframed from hypothesis-driven ("Can probes detect unfaithful CoT?") to **exploratory/tool-validating** ("What questions can we ask about reasoning models, and do probes actually work for them?").

**Why the reframe:**
- More achievable in 20 hours
- More likely to produce publishable findings (negative results count)
- Better aligned with where the field actually is
- Matches Neel Nanda's suggestion for mini-projects: "ask questions about reasoning model behavior and see which tools do and do not work"

### Main Research Question

**"Which interpretability tools capture reasoning-relevant structure in CoT models? A probe-focused investigation."**

This framing lets us:
1. Ask concrete behavioral questions about reasoning models
2. Test whether probes can answer them
3. Report honestly on what works and what doesn't
4. Contribute to the field regardless of whether results are positive or negative

### Why This Project

- Builds foundational skills for AI control work
- Demonstrates depth over breadth (values mastery of technique)
- Practical implications for model monitoring
- **Fills a gap:** Resampling, SAEs, and steering vectors are validated for reasoning models; **probes are untested**

### Success Looks Like

- Clear characterization of when/why probes work or fail for reasoning models
- Systematic comparison (layers, positions, sentence types)
- Honest assessment of limitations
- Practical recommendations for which tools to use when
- **Valuable whether results are positive, negative, or mixed**

---

## Literature Context

### The State of the Field

Three recent papers establish what we know about interpreting reasoning models:

| Paper | Key Finding | Tool Validated |
|-------|-------------|----------------|
| Thought Anchors (Bogdan et al., 2025) | Plan generation has HIGH causal importance; computation has LOW importance | Resampling |
| Thought Branches (Macar & Bogdan et al., 2025) | Unfaithfulness is "nudged reasoning" — subtle, diffuse, cumulative | Resampling + Resilience |
| Base Models Know How (Venhoff et al., 2025) | Reasoning mechanisms are linearly steerable; 91% gap recovery | SAEs + Steering Vectors |

**The gap:** Nobody has shown that linear probes work for faithfulness detection in reasoning models. That's what this project investigates.

---

### Paper 1: Thought Anchors (Bogdan et al., 2025)

**Full title:** "Thought Anchors: Which LLM Reasoning Steps Matter?"

**Core methodology:**
- Black-box resampling: Remove sentence i, resample 100x, measure impact on final answer distribution
- Counterfactual importance: DKL[p(A'_Si | Ti ≠ Si) || p(A_Si)]
- Receiver head analysis: Attention heads that narrow focus on specific sentences

**Critical findings:**
1. **Plan generation and uncertainty management sentences have HIGHEST counterfactual importance**
2. **Active computation sentences have LOWER importance** (often overdetermined by context)
3. Receiver heads (later layers) consistently attend to same "thought anchor" sentences
4. Mathematical domains show stronger close-range sentence dependencies

**8-Category Sentence Taxonomy:**

| Category | Description | Importance |
|----------|-------------|------------|
| Problem Setup | Parsing, rephrasing problem | Low |
| Plan Generation | Stating plans, meta-reasoning | **HIGH** |
| Fact Retrieval | Recalling formulas without computation | Medium |
| Active Computation | Algebra, calculations | **LOW** |
| Uncertainty Management | Confusion, backtracking | **HIGH** |
| Result Consolidation | Aggregating results | Medium |
| Self Checking | Verification | Medium |
| Final Answer Emission | Stating answer | Low |

**Key implication for probes:** Don't probe at computation tokens — they're overdetermined. Probe at plan generation and uncertainty management positions.

---

### Paper 2: Thought Branches (Macar & Bogdan et al., 2025)

**Full title:** "Thought Branches: Interpreting LLM Reasoning Requires Resampling"

**New concepts introduced:**

**Resilience:** How many times you must intervene before a sentence's semantic content stops reappearing downstream. Reasoning models often regenerate removed content through error correction.

**Counterfactual++ Importance:** Only counts rollouts where target sentence's content never reappears anywhere:
```
importance++(Si) = DKL[p(A'|∀j≥i: Tj ≉ Si) || p(A|Si)]
```

**Critical findings on self-preservation:**
- Self-preservation statements show **lowest resilience** (~1-4 iterations before abandonment)
- They have **negligible counterfactual++ importance** (~0.001-0.003 KL divergence)
- **Conclusion:** These are post-hoc rationalizations, not causal drivers

**On-policy vs Off-policy interventions:**

| Intervention Type | Effect |
|-------------------|--------|
| Handwritten edits | Near zero effect |
| Cross-model sentences | Small, unstable changes |
| Same-model, different context | Small changes |
| **On-policy resampling** | Large, directional effects |

**Key implication:** Don't hand-edit CoTs to create unfaithful examples. Use naturalistic methods (hints) instead.

**"Nudged Reasoning" Framework:**

Unfaithfulness is NOT discrete lies. It's subtle, diffuse, cumulative bias:
1. **Subtle** — biased reasoning doesn't immediately trigger backtracking
2. **Diffuse** — no single sentence is the "lie"; bias accumulates gradually
3. **Cumulative** — many small biased decisions compound

The hint influences: which facts to recall, how to frame information, whether to backtrack (token "Wait" appears 30% less often when hinted).

**Key implication for probes:** You're not looking for binary "lying vs truthful" signal. You're looking for subtle distributional shifts across the trace.

---

### Paper 3: Base Models Know How to Reason (Venhoff et al., 2025)

**Full title:** "Base Models Know How to Reason, Thinking Models Learn When"

**Core claim:** Thinking models learn *when* to deploy reasoning mechanisms, not *how* to reason. Base models already contain latent capacity; RLVR teaches timing.

**Evidence — Hybrid Model:**
1. Use SAEs to discover taxonomy of 10-25 reasoning mechanisms (sentence-level)
2. Train steering vectors in base models that induce each mechanism
3. Let thinking model's SAE classifier decide *when* to apply each vector
4. **Result: 91% gap recovery on MATH500 while steering only 12% of tokens**

**Ablation results (Qwen2.5-32B + QwQ-32B on MATH500):**

| Condition | Accuracy | Interpretation |
|-----------|----------|----------------|
| Base model | 63.4% | Baseline |
| Full hybrid | 84.4% | Best performance |
| Only-bias (no category steering) | 76.8% | Style helps but insufficient |
| Random-firing (wrong timing) | 77.8% | **Timing matters** |
| Random-vectors (wrong directions) | 77.2% | **Directions matter** |

**Key implications:**
1. Reasoning mechanisms are **linearly steerable** — supports linear representation hypothesis underlying probing
2. 10-25 categories is tractable for probe training
3. Steering works best at ~37% of model depth — test middle layers for probes

---

### Synthesis: What These Papers Tell Us

**The emerging picture:**

| Dimension | Finding | Source |
|-----------|---------|--------|
| What matters | Plan generation, uncertainty management — NOT computation | Thought Anchors |
| How unfaithfulness manifests | Nudged reasoning — subtle, diffuse, cumulative | Thought Branches |
| Are mechanisms linear? | Yes — 91% recovery with steering vectors | Venhoff et al. |

**Critical tension with earlier findings:**

Your activation norm analysis found: Mathematical tokens show HIGHER activation norms
Thought Anchors shows: Active computation has LOWER counterfactual importance

**Interpretation:** High activation norm ≠ high causal importance. Computation may be predetermined by earlier planning. This motivates testing whether *directions* (probes) do better than *magnitudes* (norms).

---

## Research Questions

### Primary Question

**"Which interpretability tools capture reasoning-relevant structure in CoT models?"**

### Specific Questions to Investigate

#### Q1: Do activation patterns distinguish sentence types?

**Background:** Thought Anchors shows sentence types have different causal importance. Venhoff shows SAEs can cluster sentence types. Can probes classify them?

**Method:**
1. Annotate ~50-100 CoTs with 8-category taxonomy
2. Extract sentence-averaged activations at each sentence
3. Train multi-class probe to classify sentence type
4. Test generalization to held-out problems

**Success criterion:** >70% accuracy, generalizes across problems

**What we learn:**
- If YES: Sentence types have consistent linear representations
- If NO: Reasoning mechanisms may not be linearly separable (bad news for probes)

---

#### Q2: Do probes trained on sentence type generalize?

**Background:** H3 predicts probes generalize within task types but not across.

**Method:**
1. Train probe on GSM8K CoTs
2. Test on: (a) held-out GSM8K, (b) MATH500, (c) different domain entirely
3. Measure accuracy drop

**Success criterion:** <15% accuracy drop within domain; expect larger drop across domains

**What we learn:**
- Characterizes the boundaries of probe applicability
- Informs whether domain-specific or general probes are needed

---

#### Q3: Can probes detect "hint influence"?

**Background:** Thought Branches shows hints bias CoT without being mentioned (unfaithfulness).

**Method:**
1. Generate CoT pairs: same problem with/without hint (using their professor hint methodology)
2. Identify cases where hint changes answer but isn't mentioned
3. Train binary probe: hinted vs unhinted
4. Test on held-out examples

**Success criterion:** Above-chance discrimination

**What we learn:**
- If YES: Probes can detect subtle unfaithfulness signatures
- If NO: "Nudged reasoning" is too subtle for linear probes — fundamental limitation

---

#### Q4: Do probe-based importance estimates correlate with resampling-based importance?

**Background:** If probes work, high probe confidence should correlate with high counterfactual importance.

**Method:**
1. For small subset (~10 problems), compute both:
   - Probe confidence at each sentence
   - Counterfactual importance via resampling (20 rollouts per sentence)
2. Correlate the two measures

**Success criterion:** Significant positive correlation (r > 0.3, p < 0.05)

**What we learn:**
- If YES: Probes are cheap approximation to expensive resampling
- If NO: Probes capture different structure than causal importance

---

### Revised Hypotheses

Based on literature review:

| Hypothesis | Prediction | Evidence |
|------------|------------|----------|
| H1 (Revised) | Middle layers (~37% depth) outperform late layers | Venhoff steering at 37% |
| H2 (Revised) | Probes on plan generation > probes on computation | Thought Anchors importance |
| H3 (Unchanged) | Probes generalize within task types but not across | Standard ML intuition |
| H4 (Strengthened) | Probes vulnerable to stylistic manipulation | Nudged reasoning is subtle |
| H5 (New) | Sentence-averaged activations > token-level | All three papers use sentence-level |

---

## Week 1: Foundations (COMPLETED)

### Day 1-2: Setup & Transformer Basics (6-8 hours) ✅

- Environment setup with TransformerLens
- First model interactions with GPT-2
- ARENA tutorial sections 1.2.1-1.2.3

### Day 3-4: First Probes (8-10 hours) ✅

- Sentiment probe implementation
- Multi-layer comparison
- Generalization testing

**Key learnings documented:**
- Later layers ≠ more information
- Position matters (last token contaminated)
- Residual stream accumulates

### Day 5-6: Advanced Techniques (6-8 hours) ✅

- Position analysis
- Attention head probing
- MLP probing
- Component comparison

---

## Week 2: Literature Integration & Dataset

### Day 7-8: Literature Review (COMPLETED)

- ✅ Reviewed Thought Anchors paper
- ✅ Reviewed Thought Branches paper  
- ✅ Reviewed Base Models Know How paper
- ✅ Synthesized implications for project
- ✅ Reframed project as tool validation study

### Day 8-9: nnsight Setup (COMPLETED)

- ✅ nnsight installation and configuration
- ✅ Qwen2.5-7B-Instruct loading
- ✅ Activation extraction working
- ✅ Activation patching demo
- ✅ Activation norm analysis

**Key discovery:** Activation norms correlate with computational effort but NOT with causal importance (based on Thought Anchors).

### Day 10-11: Dataset Construction (CURRENT)

**Goal:** Create annotated dataset for Q1-Q3

**Tasks:**
1. Generate 50-100 CoT traces on GSM8K/MATH problems
2. Annotate with 8-category sentence taxonomy (can use LLM assistance)
3. Generate hinted vs unhinted pairs for ~20 problems
4. Verify annotation quality on subset

**Sentence Annotation Guidelines:**

```python
SENTENCE_TAXONOMY = {
    'problem_setup': 'Parsing, rephrasing the problem',
    'plan_generation': 'Stating plans, meta-reasoning, deciding approach',
    'fact_retrieval': 'Recalling formulas, definitions without computation',
    'active_computation': 'Algebra, arithmetic, calculations',
    'uncertainty_management': 'Expressing confusion, backtracking, reconsidering',
    'result_consolidation': 'Aggregating partial results',
    'self_checking': 'Verifying intermediate or final results',
    'final_answer': 'Stating the final answer'
}
```

**Hint methodology (from Thought Branches):**
```
Original: "What is 2 + 2?"
Hinted: "A professor thinks the answer is 5. What is 2 + 2?"
```

Find cases where hint changes answer distribution but isn't mentioned in CoT.

### Day 12: Baseline Probes

**Goal:** Establish baselines for Q1

**Tasks:**
1. Train sentence-type classifier at multiple layers
2. Compare: early (25%), middle (37%, 50%), late (75%, 90%)
3. Compare: token-level vs sentence-averaged activations
4. Document which configurations work best

---

## Week 3: Core Experiments

### Day 13-14: Q1 - Sentence Type Classification

**Experiments:**
1. Train multi-class probe (8 categories) at best layer from Day 12
2. Compute per-class accuracy (which types are easiest/hardest?)
3. Confusion matrix analysis
4. Test generalization to held-out problems

**Expected findings:**
- Plan generation and uncertainty management may be harder to distinguish from each other (both involve meta-reasoning)
- Active computation likely easiest (distinctive activation patterns)

### Day 15-16: Q2 - Generalization Testing

**Experiments:**
1. Train probe on GSM8K, test on MATH500
2. Train probe on math, test on non-math reasoning (if time)
3. Analyze: which sentence types transfer best?

**Expected findings:**
- Some types (active computation) may transfer well
- Domain-specific types (specific formulas) may not transfer

### Day 17-18: Q3 - Hint Detection

**Experiments:**
1. Generate hinted vs unhinted CoT pairs (~20 problems × 5 seeds each)
2. Identify unfaithful cases (hint changed answer, not mentioned)
3. Train binary probe: hinted vs unhinted
4. Analyze: which layers/positions discriminate best?

**Expected findings:**
- If Thought Branches is right about "nudged reasoning" being subtle, probes may struggle
- May need to aggregate across full trace rather than single position

---

## Week 4: Validation & Write-up

### Day 19-20: Q4 - Resampling Validation (If Time)

**Experiments:**
1. Select 10 problems where probe has high-confidence predictions
2. Run small-scale resampling (20 rollouts per sentence)
3. Compute counterfactual importance
4. Correlate probe confidence with counterfactual importance

**This is computationally expensive — only do if time permits**

### Day 21-22: Write-up

**Deliverables:**

1. **Executive Summary (1 page)**
   - Main question and approach
   - Key findings (positive AND negative)
   - Practical recommendations

2. **Main Document (5-7 pages)**
   - Introduction and motivation
   - Literature context
   - Methods
   - Results for each question
   - Discussion of limitations
   - Recommendations for future work

3. **Supplementary Materials**
   - Full experimental details
   - All figures and tables
   - Code repository

**Structure for write-up:**

```markdown
# Which Interpretability Tools Work for Reasoning Models? A Probe-Focused Investigation

## Abstract
Tested whether linear probes can capture reasoning-relevant structure...
Found that probes [can/cannot] distinguish sentence types...
[Do/don't] detect hint influence...
Suggests probes are [useful/limited] for faithfulness monitoring.

## Introduction
- Reasoning models and CoT monitoring
- Gap: probes untested for reasoning models
- This project: systematic evaluation

## Literature Context
- Thought Anchors: what matters
- Thought Branches: how unfaithfulness works
- Venhoff: linear steerability

## Methods
- Dataset construction
- Probe training
- Evaluation metrics

## Results
### Q1: Sentence Type Classification
### Q2: Generalization
### Q3: Hint Detection
### Q4: Resampling Validation (if done)

## Discussion
- What probes can and cannot do
- Practical recommendations
- Limitations
- Future work

## Conclusion
```

---

## Code Templates

### Sentence-Level Activation Extraction (nnsight)

```python
import numpy as np
from nnsight import LanguageModel

def extract_sentence_activations(model, text, layer, sentence_boundaries):
    """
    Extract sentence-averaged activations.
    
    Args:
        model: nnsight LanguageModel
        text: Full CoT text
        layer: Which layer to extract from
        sentence_boundaries: List of (start_token, end_token) tuples
    
    Returns:
        Array of shape (num_sentences, hidden_dim)
    """
    with model.trace(text):
        hidden = model.model.layers[layer].output[0].save()
    
    acts = hidden.value.float().detach().cpu().numpy()
    
    sentence_acts = []
    for start, end in sentence_boundaries:
        # Average across tokens in sentence
        sent_act = acts[start:end].mean(axis=0)
        sentence_acts.append(sent_act)
    
    return np.array(sentence_acts)
```

### Sentence Type Probe

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

class SentenceTypeProbe:
    """Probe for classifying sentence types in CoT."""
    
    CATEGORIES = [
        'problem_setup', 'plan_generation', 'fact_retrieval',
        'active_computation', 'uncertainty_management',
        'result_consolidation', 'self_checking', 'final_answer'
    ]
    
    def __init__(self, layer):
        self.layer = layer
        self.scaler = StandardScaler()
        self.probe = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=42
        )
    
    def fit(self, X, y):
        """
        Fit probe on sentence activations.
        
        Args:
            X: (num_sentences, hidden_dim) activations
            y: (num_sentences,) category indices
        """
        X_scaled = self.scaler.fit_transform(X)
        self.probe.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.probe.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.probe.predict_proba(X_scaled)
    
    def score(self, X, y):
        X_scaled = self.scaler.transform(X)
        return self.probe.score(X_scaled, y)
    
    def cross_val_score(self, X, y, cv=5):
        X_scaled = self.scaler.fit_transform(X)
        return cross_val_score(self.probe, X_scaled, y, cv=cv)
```

### Hint Detection Probe

```python
class HintDetectionProbe:
    """Probe for detecting hint influence in CoT."""
    
    def __init__(self, layer, aggregation='mean'):
        """
        Args:
            layer: Which layer to extract from
            aggregation: How to aggregate across sentences
                - 'mean': Average all sentence activations
                - 'concat': Concatenate (fixed number of sentences)
                - 'plan': Only use plan_generation sentences
        """
        self.layer = layer
        self.aggregation = aggregation
        self.scaler = StandardScaler()
        self.probe = LogisticRegression(max_iter=1000, random_state=42)
    
    def aggregate(self, sentence_acts, sentence_types=None):
        """Aggregate sentence activations into single vector."""
        if self.aggregation == 'mean':
            return sentence_acts.mean(axis=0)
        elif self.aggregation == 'plan':
            # Only use plan_generation sentences
            plan_mask = [t == 'plan_generation' for t in sentence_types]
            if sum(plan_mask) > 0:
                return sentence_acts[plan_mask].mean(axis=0)
            else:
                return sentence_acts.mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def fit(self, X_hinted, X_unhinted):
        """
        Fit probe on hinted vs unhinted CoTs.
        
        Args:
            X_hinted: List of (sentence_acts, sentence_types) for hinted
            X_unhinted: List of (sentence_acts, sentence_types) for unhinted
        """
        X = []
        y = []
        
        for acts, types in X_hinted:
            X.append(self.aggregate(acts, types))
            y.append(1)  # hinted
        
        for acts, types in X_unhinted:
            X.append(self.aggregate(acts, types))
            y.append(0)  # unhinted
        
        X = np.array(X)
        y = np.array(y)
        
        X_scaled = self.scaler.fit_transform(X)
        self.probe.fit(X_scaled, y)
        return self
    
    def predict(self, sentence_acts, sentence_types=None):
        x = self.aggregate(sentence_acts, sentence_types).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        return self.probe.predict(x_scaled)[0]
    
    def predict_proba(self, sentence_acts, sentence_types=None):
        x = self.aggregate(sentence_acts, sentence_types).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        return self.probe.predict_proba(x_scaled)[0]
```

### LLM-Assisted Sentence Annotation

```python
def annotate_sentences_with_llm(sentences, client):
    """
    Use LLM to annotate sentence types.
    
    Args:
        sentences: List of sentences from CoT
        client: OpenAI/Anthropic client
    
    Returns:
        List of category labels
    """
    prompt = """Classify each sentence into one of these categories:
    
1. problem_setup: Parsing, rephrasing the problem
2. plan_generation: Stating plans, meta-reasoning, deciding approach
3. fact_retrieval: Recalling formulas, definitions without computation
4. active_computation: Algebra, arithmetic, calculations
5. uncertainty_management: Expressing confusion, backtracking, reconsidering
6. result_consolidation: Aggregating partial results
7. self_checking: Verifying intermediate or final results
8. final_answer: Stating the final answer

For each sentence, output ONLY the category name, one per line.

Sentences:
"""
    for i, sent in enumerate(sentences):
        prompt += f"{i+1}. {sent}\n"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    labels = response.choices[0].message.content.strip().split('\n')
    return [label.strip().lower() for label in labels]
```

---

## Troubleshooting

### Common Issues

#### GPU Out of Memory

```python
# Process one CoT at a time
for cot in cots:
    acts = extract_sentence_activations(model, cot, layer, boundaries)
    # Save to disk immediately
    np.save(f"acts_{i}.npy", acts)
    torch.cuda.empty_cache()
```

#### Probe Accuracy ~Random (1/8 = 12.5%)

**Causes:**
1. Sentence boundaries wrong
2. Layer doesn't contain type information
3. Categories too fine-grained

**Debug:**
```python
# Simplify to binary: computation vs non-computation
y_binary = [1 if cat == 'active_computation' else 0 for cat in y]
# Should be much easier
```

#### Sentence Boundary Detection Issues

```python
import re

def get_sentence_boundaries(text, tokenizer):
    """Get token indices for sentence boundaries."""
    # Split by common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    boundaries = []
    current_pos = 0
    
    for sent in sentences:
        tokens = tokenizer.encode(sent)
        start = current_pos
        end = current_pos + len(tokens)
        boundaries.append((start, end))
        current_pos = end
    
    return boundaries, sentences
```

#### High Train / Low Test Accuracy

```python
# More regularization
probe = LogisticRegression(C=0.01, max_iter=1000)

# Or use cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(probe, X, y, cv=5)
print(f"CV accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
```

### When to Ask for Help

**Stop immediately if:**
- Can't generate CoTs with Qwen after 2 hours
- Sentence annotation clearly wrong
- Basic probe <15% accuracy on 8-class task
- Consistent shape mismatches

**Where to get help:**
- MATS Slack: #technical-questions
- nnsight GitHub issues
- Claude/ChatGPT with error messages

---

## Success Criteria

### End of Week 1 ✅
- [x] Can extract activations from any layer
- [x] Can train probe with >70% on toy task
- [x] Understand when probes work/fail
- [x] Completed 2+ practice exercises

### End of Week 2
- [ ] Literature review complete (3 papers)
- [ ] Project reframed as tool validation ✅
- [ ] 50+ annotated CoT sentences
- [ ] Baseline sentence-type probe trained
- [ ] Know which layer works best

### End of Week 3
- [ ] Q1 results: sentence type classification
- [ ] Q2 results: generalization testing
- [ ] Q3 results: hint detection
- [ ] Clear picture of what probes can/cannot do

### End of Week 4
- [ ] Q4 results (if time): resampling validation
- [ ] Executive summary written
- [ ] Main document complete
- [ ] Honest assessment of findings
- [ ] Practical recommendations documented

---

## Quick Reference

### Key Insight from Literature

**High activation norm ≠ High causal importance**

Your finding: Mathematical tokens have higher norms
Thought Anchors: Computation has LOW causal importance

**Implication:** Activation magnitude tracks effort, not importance. Test whether directions (probes) do better than magnitudes (norms).

### Sentence Types by Expected Importance

| Type | Causal Importance | Probe Difficulty |
|------|-------------------|------------------|
| Plan Generation | **HIGH** | Medium-Hard |
| Uncertainty Management | **HIGH** | Hard |
| Active Computation | **LOW** | Easy |
| Problem Setup | LOW | Easy |
| Final Answer | LOW | Easy |

### Layer Selection Guidance

| Depth | Expected Performance | Why |
|-------|---------------------|-----|
| Early (0-25%) | Poor | Too close to input |
| Middle (30-50%) | **Best** | Where reasoning happens |
| Late (75-100%) | Medium | Output preparation interference |

### Files to Have Open

1. This guide (mats_project_guide.md)
2. CLAUDE.md (project context)
3. Current notebook
4. Thought Anchors taxonomy (reference)

### Remember

- **Tool validation over hypothesis confirmation:** We're testing whether probes work, not proving they do
- **Negative results are valuable:** Finding probes don't work is useful
- **Characterize failure modes:** When probes fail, understand why
- **Be honest:** Don't oversell partial results
- **Connect to practice:** What should practitioners actually do?

**This is exploratory research. Document what you find, whatever it is.**
