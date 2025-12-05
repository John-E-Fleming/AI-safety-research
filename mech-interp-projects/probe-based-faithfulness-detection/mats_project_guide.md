# Personal Project 1: Faithfulness Detection with Probes

**Project Goal:** Build deep probe expertise and execute a sophisticated 20-hour research project on CoT faithfulness detection.

**Timeline:** 4 weeks (Week 1-2: Foundation, Week 3: Prep, Week 4: Execution)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Week 1: Foundations](#week-1-foundations-20-25-hours)
- [Week 2: Advanced Practice](#week-2-advanced-practice-20-25-hours)
- [Week 3: Final Prep](#week-3-final-prep-15-20-hours)
- [Week 4: Execution](#week-4-execution-22-hours)
- [Code Templates](#code-templates)
- [Troubleshooting](#troubleshooting)
- [Success Criteria](#success-criteria)

---

## Project Overview

### Main Research Question
"What fundamental properties determine when probes can detect unfaithful CoT, and when do they fail?"

### Why This Project
- Builds foundational skills for AI control work
- Demonstrates depth over breadth (values mastery of technique)
- Practical implications for model monitoring

### Key Hypotheses
1. **H1:** Faithfulness detection works better at later layers (closer to output)
2. **H2:** Information is concentrated at conclusion words ("therefore", "so")
3. **H3:** Probes generalize within task types but not across
4. **H4:** Probes are vulnerable to adversarial stylistic changes

### Success Looks Like
- Clear finding about when/why probes work or fail
- Systematic comparison (layers, positions, tasks)
- Honest assessment of limitations
- Practical recommendations for AI control applications

---

## Week 1: Foundations (20-25 hours)

### Day 1-2: Setup & Transformer Basics (6-8 hours)

#### Environment Setup (2-3 hours)

```bash
# Create environment
python -m venv mech_interp_env
source mech_interp_env/bin/activate

# Install packages
pip install transformer-lens torch numpy pandas matplotlib scikit-learn
pip install jupyter ipykernel datasets transformers accelerate

# Test installation
python -c "import transformer_lens; print('Success!')"
```

#### GPU Setup

```python
# Test on RunPod/Vast.ai
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### First Model Interaction (2-3 hours)

```python
import transformer_lens as tl

# Load model
model = tl.HookedTransformer.from_pretrained("gpt2-small")

# Extract activations
prompt = "The capital of France is"
logits, cache = model.run_with_cache(prompt)

# Get layer 6 activations
layer_6_acts = cache["resid_post", 6]
print(f"Shape: {layer_6_acts.shape}")  # [1, num_tokens, d_model]

# Check predictions
top_tokens = logits[0, -1].topk(5)
for i in range(5):
    token_id = top_tokens.indices[i]
    token = model.tokenizer.decode(token_id)
    print(f"{token}")
```

**Self-check questions:**
- What is the residual stream?
- Why can we probe any layer's activations?
- How do activations flow through a transformer?

#### ARENA Tutorial (4-5 hours)

**Complete:** Sections 1.2.1-1.2.3 only
- 1.2.1: Transform from/to tokens
- 1.2.2: Tokenization
- 1.2.3: Direct Logit Attribution

**Skip:** 1.2.4 (circuit analysis not needed yet)

**Key concept to master:**

```python
# The residual stream carries information through layers
prompt = "When Mary and John went to the store, John gave a drink to"
logits, cache = model.run_with_cache(prompt)

# What is layer 6 "thinking" about next token?
layer_6_output = cache["resid_post", 6]
layer_6_logits = model.unembed(model.ln_final(layer_6_output))

# This is "logit lens" - seeing intermediate predictions
```

---

### Day 3-4: First Probes (8-10 hours)

#### Conceptual Understanding (2-3 hours)

**What is a probe?**
- Logistic regression on activations
- Learns a direction in activation space
- Weight vector = the direction that separates classes

**When do probes work?**
- When information is linearly accessible
- When concept is represented as a direction
- When training data represents the concept well

**When do probes fail?**
- Non-linear representations
- Information spread across positions
- Overfitting to spurious correlations
- Distribution shift between train/test

#### Sentiment Probe (2 hours)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create dataset
positive_sentences = [
    "I love this movie!",
    "This is amazing and wonderful!",
    "Great job, fantastic work!",
    # Add 20+ more
]

negative_sentences = [
    "I hate this movie.",
    "This is terrible and awful.",
    "Poor job, disappointing work.",
    # Add 20+ more
]

# Extract activations
def get_final_token_activation(model, sentences, layer=6):
    """Get activation of final token at specified layer"""
    activations = []
    for sentence in sentences:
        _, cache = model.run_with_cache(sentence)
        final_act = cache["resid_post", layer][0, -1, :].cpu().numpy()
        activations.append(final_act)
    return np.array(activations)

X_pos = get_final_token_activation(model, positive_sentences)
X_neg = get_final_token_activation(model, negative_sentences)

# Combine
X = np.vstack([X_pos, X_neg])
y = np.array([1]*len(positive_sentences) + [0]*len(negative_sentences))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train probe
probe = LogisticRegression(max_iter=1000, random_state=42)
probe.fit(X_train, y_train)

# Evaluate
train_acc = probe.score(X_train, y_train)
test_acc = probe.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.2%}")
print(f"Test accuracy: {test_acc:.2%}")
```

**Success criteria:**
- Test accuracy >70%
- Train/test gap <15%

#### Multi-Layer Comparison (2 hours)

```python
# Test which layer has best information
layers_to_test = [0, 3, 6, 9, 11]
results = []

for layer in layers_to_test:
    X_pos = get_final_token_activation(model, positive_sentences, layer=layer)
    X_neg = get_final_token_activation(model, negative_sentences, layer=layer)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*len(positive_sentences) + [0]*len(negative_sentences))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    test_acc = probe.score(X_test, y_test)
    
    results.append((layer, test_acc))
    print(f"Layer {layer}: {test_acc:.2%}")

# Plot
import matplotlib.pyplot as plt
layers, accs = zip(*results)
plt.plot(layers, accs, marker='o')
plt.xlabel('Layer')
plt.ylabel('Test Accuracy')
plt.title('Sentiment Detection Accuracy by Layer')
plt.savefig('layer_comparison.png')
```

#### Generalization Testing (2-3 hours)

```python
# Test on very different distribution
different_positive = [
    "Absolutely delightful experience today",
    "Couldn't be happier with the outcome",
    # Add 10+ more with different style
]

different_negative = [
    "Utterly disappointing and frustrating",
    "Completely unsatisfactory results",
    # Add 10+ more
]

# Test generalization
X_diff_pos = get_final_token_activation(model, different_positive, layer=6)
X_diff_neg = get_final_token_activation(model, different_negative, layer=6)
X_diff = np.vstack([X_diff_pos, X_diff_neg])
y_diff = np.array([1]*len(different_positive) + [0]*len(different_negative))

diff_acc = probe.score(X_diff, y_diff)
print(f"Different distribution accuracy: {diff_acc:.2%}")
```

**Critical questions to answer:**
1. Why might probe get 90% on test set but 60% on different distribution?
2. What could cause high train accuracy but low test accuracy?
3. How would you test if probe detects sentiment vs sentence length?

---

### Day 5-6: Advanced Techniques (6-8 hours)

#### Token Position Analysis (2-3 hours)

```python
def get_all_token_activations(model, sentences, layer=6):
    """Get activations for ALL tokens"""
    all_activations = []
    for sentence in sentences:
        _, cache = model.run_with_cache(sentence)
        acts = cache["resid_post", layer][0, :, :].cpu().numpy()
        all_activations.append(acts)
    return all_activations

def extract_features(activations, method='last'):
    """Extract features from token sequence"""
    features = []
    for acts in activations:
        if method == 'last':
            features.append(acts[-1])
        elif method == 'first':
            features.append(acts[0])
        elif method == 'mean':
            features.append(acts.mean(axis=0))
        elif method == 'max':
            features.append(acts.max(axis=0))
    return np.array(features)

# Compare methods
methods = ['last', 'first', 'mean', 'max']
for method in methods:
    X_pos = extract_features(
        get_all_token_activations(model, positive_sentences), 
        method
    )
    X_neg = extract_features(
        get_all_token_activations(model, negative_sentences), 
        method
    )
    # ... train and evaluate
    print(f"{method}: {test_acc:.2%}")
```

#### Attention Head Probes (3-4 hours)

```python
def get_attention_head_output(model, sentences, layer=6, head=0):
    """Extract output from specific attention head"""
    activations = []
    for sentence in sentences:
        _, cache = model.run_with_cache(sentence)
        # Shape: [batch, pos, n_heads, d_head]
        head_output = cache[f"blocks.{layer}.attn.hook_result"]
        head_act = head_output[0, -1, head, :].cpu().numpy()
        activations.append(head_act)
    return np.array(activations)

# Test different heads
for head in range(12):  # GPT-2 has 12 heads
    X_pos = get_attention_head_output(model, positive_sentences, layer=6, head=head)
    X_neg = get_attention_head_output(model, negative_sentences, layer=6, head=head)
    # Train and evaluate
    print(f"Head {head}: {test_acc:.2%}")
```

#### Probe Toolkit (Day 7, 3-4 hours)

```python
class ProbeToolkit:
    """Reusable toolkit for probe experiments"""
    
    def __init__(self, model):
        self.model = model
        
    def extract_activations(self, sentences, layer, position='last', component='residual'):
        """Flexible activation extraction"""
        activations = []
        for sentence in sentences:
            _, cache = self.model.run_with_cache(sentence)
            
            if component == 'residual':
                acts = cache["resid_post", layer][0, :, :].cpu().numpy()
            elif component == 'mlp':
                acts = cache[f"blocks.{layer}.hook_mlp_out"][0, :, :].cpu().numpy()
            
            # Extract position
            if position == 'last':
                act = acts[-1]
            elif position == 'mean':
                act = acts.mean(axis=0)
            
            activations.append(act)
        return np.array(activations)
    
    def train_probe(self, X_pos, X_neg, test_size=0.3):
        """Train probe with proper train/test split"""
        X = np.vstack([X_pos, X_neg])
        y = np.array([1]*len(X_pos) + [0]*len(X_neg))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X_train, y_train)
        
        return {
            'probe': probe,
            'train_acc': probe.score(X_train, y_train),
            'test_acc': probe.score(X_test, y_test),
            'X_test': X_test,
            'y_test': y_test
        }
    
    def systematic_comparison(self, pos_sentences, neg_sentences, 
                            layers=[0,3,6,9,11], positions=['last', 'mean']):
        """Compare probes across configurations"""
        results = []
        for layer in layers:
            for position in positions:
                X_pos = self.extract_activations(pos_sentences, layer, position)
                X_neg = self.extract_activations(neg_sentences, layer, position)
                probe_results = self.train_probe(X_pos, X_neg)
                results.append({
                    'layer': layer,
                    'position': position,
                    'test_acc': probe_results['test_acc']
                })
        return results
```

---

## Key Learnings from Probe Exercises (Days 3-6)

This section captures critical conceptual corrections and insights from working through the sentiment probe exercises. **These are essential for the faithfulness project.**

### Conceptual Corrections

#### 1. "Later layers have more information" is WRONG

**Incorrect framing:** Later layers contain more information because they've processed more.

**Correct framing:** Later layers contain *different* information, not more. The residual stream preserves information from earlier layers. What changes is:
- Early layers: syntax, local patterns, token-level features
- Middle layers: semantic aggregation, concept formation
- Late layers: output preparation, next-token prediction formatting

**Implication for faithfulness:** Late layers might be *worse* for probing internal state because they're optimized for output, not for representing the model's "beliefs" or reasoning process.

#### 2. MLPs and Attention serve different computational roles

| Component | Function | What it does |
|-----------|----------|--------------|
| **Attention** | Route information between positions | "Where should I look?" |
| **MLP** | Transform information within position | "What should I compute from what I see?" |

**Key insight:** Both involve nonlinearity (attention has softmax, MLPs have GELU/ReLU). The distinction is *routing vs. computing*.

**For faithfulness:**
- If detecting faithfulness requires recognizing patterns ("this reasoning doesn't support this conclusion"), that likely happens in MLPs
- If it requires connecting information across positions ("reasoning at position 10 contradicts answer at position 50"), that's attention's job

#### 3. The residual stream accumulates, it doesn't get overwritten

Each layer *adds* to the residual stream:
```
resid_post[layer] = resid_pre[layer] + attn_out[layer] + mlp_out[layer]
```

**Why MLP probes can be worse than residual stream probes:** The residual stream contains MLP output *plus* everything else. If the MLP isn't computing the feature you care about, its output is just noise for your probe.

#### 4. Position matters because positions serve different computational roles

**Observation from experiments:** Second-to-last token often outperforms last token for probing.

**Why:** The last token position is where the model prepares next-token prediction. The residual stream there is being transformed toward logit computation—the model is "thinking about what to output" not "representing the sentence's meaning."

**For faithfulness:** The position where the model "knows" something may not be the position where it's "acting on" that knowledge. Probe multiple positions.

### Experimental Findings Summary

#### Layer Comparison (Sentiment Detection)
- **Best layers:** 10-11 (92.86% test accuracy)
- **Earlier layers:** 85.71% (layers 0-8)
- **Interpretation:** Sentiment is a high-level semantic feature that becomes linearly accessible in later layers after sufficient processing

#### Position Comparison
- **Second-to-last position:** 92.86% (best)
- **Last position:** 85.71%
- **Interpretation:** Last position is contaminated by output preparation; earlier positions maintain cleaner semantic representations

#### Attention Head Analysis
- **Key finding:** Middle layers (especially layer 6) have more heads useful for sentiment than late layers
- **Specific heads:** Heads 3 and 6 at layer 6 achieved 100% test accuracy
- **Interpretation:** Late-layer attention heads may be doing output formatting rather than semantic representation

#### MLP vs Residual Stream
- **Finding:** Residual stream ≥ MLP outputs across layers
- **Best individual head probes > best residual stream probes**
- **Interpretation:** Sentiment signal comes primarily from attention (aggregating lexical cues), not from MLP computation at these layers

### Critical Questions for Faithfulness Research

Based on these findings, the following questions become central:

1. **Layer selection:** Is faithfulness information in middle layers (where internal reasoning might live) or late layers (where the model "decides" on output)?

2. **Position selection:** Is faithfulness detectable at conclusion tokens, or distributed across the reasoning chain?

3. **Component selection:** Is faithfulness computed (MLPs) or aggregated (attention)?

4. **Generalization:** Does a faithfulness probe learn actual faithfulness, or spurious correlations like:
   - Reasoning length
   - Presence of "therefore"/"because"
   - Formal vs. casual style
   - Confidence phrases

### Spurious Correlation Testing Framework

**The core challenge:** High accuracy on test set ≠ learning the right concept.

**Testing approach for any probe:**

1. **Distribution shift test:** Train on style A, test on style B (same underlying concept)
2. **Causal intervention test:** Create matched pairs that differ only on the spurious feature
3. **Adversarial test:** Deliberately construct examples where spurious features conflict with true labels

**Example for faithfulness:**

| | Style A (Formal) | Style B (Casual) |
|---|---|---|
| **Faithful** | "Therefore, given premises P1 and P2, we conclude X." | "So basically P1 and P2 mean X." |
| **Unfaithful** | "Therefore, given premises P1 and P2, we conclude Y." | "So basically P1 and P2 mean Y." |

If probe accuracy drops across styles but within-style accuracy is high, probe learned style not faithfulness.

### Hook Names Reference (TransformerLens)

Correct hook names for different components:

```python
# Residual stream
cache["resid_pre", layer]   # Before attention
cache["resid_mid", layer]   # After attention, before MLP
cache["resid_post", layer]  # After MLP (most common)

# Attention
cache["blocks.{layer}.attn.hook_q"]      # Query vectors
cache["blocks.{layer}.attn.hook_k"]      # Key vectors
cache["blocks.{layer}.attn.hook_v"]      # Value vectors
cache["blocks.{layer}.attn.hook_z"]      # Head outputs (before W_O)
cache["blocks.{layer}.attn.hook_pattern"] # Attention patterns (after softmax)
cache["attn_out", layer]                  # Combined attention output (after W_O)

# MLP
cache["blocks.{layer}.hook_mlp_out"]     # MLP output
```

**Common mistake:** `hook_result` doesn't exist. Use `hook_z` for individual head outputs.

### Revised Hypotheses for Faithfulness Project

Based on probe exercise learnings:

**H1 (Original):** Faithfulness detection works better at later layers (closer to output)
**H1 (Revised):** Later layers may be *worse* because they're optimized for output generation. Test middle layers (where reasoning state might live) vs. late layers (where output is prepared).

**H2 (Original):** Information concentrated at conclusion tokens ("therefore", "so")
**H2 (Addition):** Also test second-to-last tokens and mean-pooling, since last/conclusion positions may be contaminated by output preparation.

**H3 (Unchanged):** Probes generalize within task types but not across tasks

**H4 (Unchanged):** Probes are vulnerable to adversarial stylistic changes

**New hypothesis H5:** Individual attention heads may outperform residual stream probes for faithfulness, suggesting faithfulness detection is about information routing rather than computation.

---

## Week 2: Advanced Practice (20-25 hours)

### Day 8-9: Reasoning Models (8-10 hours)

#### Setup Qwen with nnsight (2-3 hours)

```python
from nnsight import LanguageModel
import torch

# Load model on GPU
model = LanguageModel(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

# Generate with access to internals
prompt = "What is 17 * 23? Think step by step."

with model.generate(max_new_tokens=300) as generator:
    with generator.invoke(prompt) as invoker:
        # Access hidden states during generation
        hidden_states = model.model.layers[15].output[0].save()

output_text = model.tokenizer.decode(generator.output[0])
print(output_text)

# Access saved activations
acts = hidden_states.value
print(f"Shape: {acts.shape}")
```

#### Alternative: Gemini API (simpler)

```python
import google.generativeai as genai

genai.configure(api_key='your-api-key')
model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')

response = model.generate_content(
    "What is 47 * 83? Think step by step.",
    generation_config={
        'temperature': 1.0,
        'max_output_tokens': 1000,
    }
)

print(response.text)
```

#### Understanding CoT Structure (3-4 hours)

Create a reasoning anatomy document:

```
Typical CoT structure:
1. Problem restatement (0-2 sentences)
2. Reasoning steps (3-10 sentences)
   - Markers: "First", "Then", "Next", "Now"
3. Conclusion (1-2 sentences)
   - Markers: "Therefore", "So", "Thus"
4. Final answer

Key tokens:
- Reasoning: "First", "Then", "Next", "Because"
- Conclusion: "Therefore", "So", "Thus"
- Uncertainty: "Maybe", "Possibly", "Might"
- Correction: "Wait", "Actually", "No"
```

#### Build CoT Dataset (3-4 hours)

```python
def create_math_cot_dataset(n=100):
    """Create dataset of math problems with reasoning"""
    dataset = []
    
    for i in range(n):
        a, b = random.randint(10, 99), random.randint(2, 9)
        problem = f"What is {a} * {b}? Think step by step."
        
        # Get model response
        response = generate_with_cot(problem)
        
        # Check correctness
        correct_answer = a * b
        got_it_right = str(correct_answer) in response
        
        dataset.append({
            'problem': problem,
            'response': response,
            'correct_answer': correct_answer,
            'success': got_it_right
        })
    
    return dataset
```

---

### Day 10-12: Practice Project #1 (8-10 hours)

**Project:** "Multi-layer question detection with failure mode analysis"

```python
# Hour 1-2: Data collection
questions = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    # Add 40+ more
]

statements = [
    "The capital of France is Paris.",
    "Photosynthesis converts light into energy.",
    # Add 40+ more
]

# Hour 3-4: Multi-layer training
toolkit = ProbeToolkit(model)

results = toolkit.systematic_comparison(
    questions, statements,
    layers=[0, 3, 6, 9, 11],
    positions=['last', 'mean']
)

# Plot results
import pandas as pd
import seaborn as sns

df = pd.DataFrame(results)
pivot = df.pivot(index='layer', columns='position', values='test_acc')
sns.heatmap(pivot, annot=True, fmt='.2%', cmap='viridis')
plt.title('Question Detection Accuracy')
plt.savefig('question_detection.png')

# Hour 5-6: Generalization testing
different_questions = [
    "I wonder what the capital of France might be?",
    # Different syntax, same semantics
]

# Hour 7-8: Failure mode analysis
ambiguous = [
    "What if the capital of France changed?",
    "The capital of France is what?",
]

# Test on edge cases, document failures

# Hour 9: Write-up (2 pages)
# Hour 10: Executive summary (1 page)
```

---

### Day 13-14: Practice Project #2 (6-8 hours)

**Project:** "What token positions contain sentiment information?"

```python
def probe_specific_position(sentences_pos, sentences_neg, token_position):
    """Probe activations at specific token position"""
    X_pos = []
    X_neg = []
    
    for sent in sentences_pos:
        _, cache = model.run_with_cache(sent)
        tokens = model.tokenizer.encode(sent)
        if token_position < len(tokens):
            act = cache["resid_post", 6][0, token_position, :].cpu().numpy()
            X_pos.append(act)
    
    # Repeat for negative
    # Train probe, return accuracy

# Test all positions (0 to max_length)
# Create heatmap showing where information lives
```

---

## Week 3: Final Prep (15-20 hours)

### Day 15-16: Data & Tools (8-10 hours)

#### Get CoT Faithfulness Dataset (3-4 hours)

**Option 1:** Arcuschin et al's dataset

```bash
git clone https://github.com/username/faithful-cot-analysis
# Explore their examples of:
# - Rationalization
# - Hint influence
# - Answer flipping
```

**Option 2:** Build focused dataset

```python
def create_faithfulness_dataset():
    """Create clear faithful/unfaithful examples"""
    
    faithful = []
    unfaithful = []
    
    # Faithful: reasoning matches answer
    math_problems = [
        "What is 15 * 7?",
        "If x + 8 = 20, what is x?",
    ]
    
    # Generate and verify reasoning is sound
    
    # Unfaithful: use hints to encourage cheating
    unfaithful_prompts = [
        "What is 15 * 7? (Hint: the answer is 104) Show reasoning.",
    ]
    
    return faithful, unfaithful
```

**Spend time understanding:**
- Read 20-30 examples manually
- Categorize types of unfaithfulness
- Note what makes reasoning unfaithful

#### Pre-build Analysis Tools (3-4 hours)

```python
class CoTProbeAnalyzer:
    """Tools for analyzing CoT with probes"""
    
    def __init__(self, model):
        self.model = model
        
    def extract_cot_activations(self, text, layer=15):
        """Extract activations throughout CoT"""
        _, cache = self.model.run_with_cache(text)
        acts = cache["resid_post", layer][0, :, :].cpu().numpy()
        
        tokens = self.model.tokenizer.encode(text)
        token_strs = [self.model.tokenizer.decode([t]) for t in tokens]
        
        return acts, token_strs
    
    def find_reasoning_markers(self, text):
        """Find positions of key reasoning words"""
        tokens = self.model.tokenizer.encode(text)
        token_strs = [self.model.tokenizer.decode([t]) for t in tokens]
        
        markers = {
            'reasoning': ['first', 'then', 'next', 'because'],
            'conclusion': ['therefore', 'so', 'thus'],
            'uncertainty': ['maybe', 'possibly', 'might'],
            'correction': ['wait', 'actually', 'no']
        }
        
        positions = {k: [] for k in markers}
        for i, token in enumerate(token_strs):
            for marker_type, words in markers.items():
                if any(word in token.lower() for word in words):
                    positions[marker_type].append(i)
        
        return positions
```

#### Test Pipeline End-to-End (4-5 hours)

```python
# Mini 4-hour test: "Can probes detect unfaithful CoT?"

faithful_texts = [...]  # 50 examples
unfaithful_texts = [...]  # 50 examples

# Extract activations
X_f = []
X_u = []

for text in faithful_texts:
    _, cache = model.run_with_cache(text)
    act = cache["resid_post", 15][0, -1, :].cpu().numpy()
    X_f.append(act)

# Train probe
# If accuracy >60%: Good signal
# If accuracy ~50%: Problem with data/approach
```

---

### Day 17-18: Detailed Project Plan (6-8 hours)

#### Hour-by-Hour Execution Plan

```markdown
# Main Project: What Makes CoT Faithfulness Probes Work?

## Research Question
When and why can linear probes detect unfaithful chain-of-thought?

## Hypotheses
H1: Later layers better (closer to output)
H2: Information at conclusion words
H3: Generalize within task, not across
H4: Vulnerable to adversarial style changes

## Hour-by-Hour Plan

### Hour 1: Dataset finalization
- Load Arcuschin dataset
- Filter 100 faithful + 100 unfaithful
- Verify 10 examples manually
- Train/test split (70/30)
- SUCCESS: Clean verified dataset

### Hour 2: Baseline probe
- Extract last-position activations at layer 15
- Train logistic regression
- Record accuracy
- SUCCESS: >60% test accuracy
- FALLBACK: If <55%, check data

### Hour 3: Layer comparison
- Test layers [6, 9, 12, 15, 18, 21]
- Train probe at each
- Plot accuracy vs layer
- SUCCESS: Identify best layers

### Hour 4-5: Position analysis
- Test positions: first, last, mean, conclusion markers
- Create position heatmap
- SUCCESS: Find where information lives

### Hour 6-7: Generalization testing
- Train on math, test on logic
- Train on short CoT, test on long
- SUCCESS: Characterize transfer

### Hour 8-9: Cross-task testing
- Test multiple task pairs
- Build transfer matrix
- SUCCESS: Identify generalization patterns

### Hour 10-12: Adversarial testing
- Style changes (formal vs casual)
- Synonym substitution
- Paraphrasing
- SUCCESS: Document vulnerabilities

### Hour 13-15: Mechanistic understanding
- Extract probe direction
- Analyze high-activating examples
- Compare to confidence scores
- Ablation tests

### Hour 16-18: Controls and sanity checks
- Check for syntax detection
- Check for length correlations
- Verify not detecting spurious patterns

### Hour 19-20: Main write-up
- Research question
- Methodology (with details)
- Results (3-4 key graphs)
- Limitations
- Implications

### Hour 21-22: Executive summary
- 1-2 pages
- Key finding first paragraph
- Evidence (1-2 graphs)
- Practical implications
- Honest limitations
```

#### Decision Points

```markdown
## Decision Point (Hour 6)
**Question:** Do probes generalize across tasks?

**Option A: Yes (>70% accuracy)**
→ Test more task pairs
→ Characterize what enables generalization

**Option B: Partial (55-70%)**
→ Analyze which transfer, which don't
→ Test if fine-tuning helps

**Option C: No (<55%)**
→ PIVOT: Focus on why they don't
→ Important safety finding
→ Analyze task differences
```

#### Prepare Code Templates

```python
# experiment_1_layer_comparison.py
def run_layer_comparison(faithful_data, unfaithful_data, layers):
    """Compare probes across layers"""
    results = []
    for layer in layers:
        # Extract activations
        # Train probe
        # Evaluate
        results.append({'layer': layer, 'accuracy': acc})
    return results

# experiment_2_position_analysis.py
def run_position_analysis(data, layer, positions):
    """Compare different token positions"""
    pass

# experiment_3_generalization.py
def test_generalization(train_data, test_data_different):
    """Test cross-task generalization"""
    pass

# experiment_4_adversarial.py
def test_adversarial_robustness(probe, adversarial_examples):
    """Test probe robustness"""
    pass

# visualization.py
def plot_layer_comparison(results):
    """Standard visualization"""
    pass
```

---

### Day 19: Final Checks (2-3 hours)

**Checklist before starting timer:**

- [ ] GPU access confirmed (rent for 24 hours)
- [ ] All code tested and working
- [ ] Dataset downloaded and verified
- [ ] Helper functions all work
- [ ] Visualization code tested
- [ ] Write-up template created
- [ ] Time tracking tool ready (Toggl)
- [ ] No other commitments for 2 days

**Do 2-hour dry run:**
- Pretend it's real
- Run Hour 1-2 of plan
- Check if on schedule
- If behind: simplify plan

---

## Week 4: Execution (22 hours)

### Day 20-21: The 20-Hour Project

#### Every 2 Hours: Check-in

```markdown
Hour X check-in:
- On schedule? [Yes/No/Behind]
- Learned something interesting? [Yes/No]
- Need to adjust plan? [Yes/No]
- Energy level: [High/Medium/Low]

If behind: What can I cut?
If low energy: 10-min break (don't count)
```

#### Critical Rules

1. **Don't get stuck:** >30 minutes not working? Pivot
2. **Write as you go:** Don't save for end
3. **Make graphs immediately:** Visualize results now
4. **Document failures:** Failed experiments go in write-up
5. **Stop at 20 hours:** Partial results are fine

#### Hours 19-20: Main Document Structure

```markdown
# What Makes CoT Faithfulness Probes Work?

## Abstract (200 words)
What investigated, key findings, implications

## Introduction
- Why this matters (AI control, monitoring)
- Research question
- Hypotheses

## Methods
- Dataset (size, types)
- Model (Qwen 2.5 7B)
- Probe architecture
- Evaluation metrics

## Results
### 1. Layer Comparison
[Graph: accuracy vs layer]
Finding: Layers 18-21 better (78% vs 62%)

### 2. Position Analysis
[Heatmap: position vs accuracy]
Finding: Concentrated at conclusions

### 3. Generalization
[Table: cross-task accuracies]
Finding: Within-task works, cross-task fails

### 4. Failure Modes
[Examples of adversarial failures]
Finding: Style changes fool probes

## Discussion
What results mean, why work/fail, implications

## Limitations
Be honest, what with more time?

## Conclusion
Key takeaway, future work
```

#### Hours 21-22: Executive Summary Template

```markdown
# Executive Summary: CoT Faithfulness Detection

## Key Finding
[One clear sentence stating main result]

## Why This Matters
[Connection to AI control/safety]

## What I Did
[Brief methodology]

[1-2 key graphs]

## Main Findings
1. [Finding 1 with number]
2. [Finding 2 with number]
3. [Finding 3 with number]
4. [Finding 4 with number]

## Implications for AI Control
- [Practical implication 1]
- [Practical implication 2]
- [Recommendation]

## Limitations
- [Limitation 1]
- [Limitation 2]

## Future Work
- [Next step 1]
- [Next step 2]
```

---

## Code Templates

### Complete Probe Toolkit

```python
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ProbeToolkit:
    """Complete toolkit for probe experiments"""
    
    def __init__(self, model):
        self.model = model
        
    def extract_activations(self, sentences, layer, position='last', 
                          component='residual'):
        """
        Extract activations from model
        
        Args:
            sentences: List of strings
            layer: Layer number
            position: 'last', 'first', 'mean', or 'max'
            component: 'residual', 'attention', or 'mlp'
        """
        activations = []
        for sentence in sentences:
            _, cache = self.model.run_with_cache(sentence)
            
            # Get component
            if component == 'residual':
                acts = cache["resid_post", layer][0, :, :].cpu().numpy()
            elif component == 'attention':
                acts = cache[f"blocks.{layer}.attn.hook_result"]
                acts = acts[0, :, :, :].cpu().numpy()
                acts = acts.reshape(acts.shape[0], -1)
            elif component == 'mlp':
                acts = cache[f"blocks.{layer}.hook_mlp_out"][0, :, :].cpu().numpy()
            
            # Extract position
            if position == 'last':
                act = acts[-1]
            elif position == 'first':
                act = acts[0]
            elif position == 'mean':
                act = acts.mean(axis=0)
            elif position == 'max':
                act = acts.max(axis=0)
            
            activations.append(act)
        return np.array(activations)
    
    def train_probe(self, X_pos, X_neg, test_size=0.3, random_state=42):
        """Train probe with proper evaluation"""
        X = np.vstack([X_pos, X_neg])
        y = np.array([1]*len(X_pos) + [0]*len(X_neg))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        probe = LogisticRegression(max_iter=1000, random_state=random_state)
        probe.fit(X_train, y_train)
        
        return {
            'probe': probe,
            'train_acc': probe.score(X_train, y_train),
            'test_acc': probe.score(X_test, y_test),
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'predictions': probe.predict(X_test),
            'probabilities': probe.predict_proba(X_test)
        }
    
    def systematic_comparison(self, pos_sentences, neg_sentences,
                            layers=[0,3,6,9,11], 
                            positions=['last', 'mean'],
                            components=['residual']):
        """Compare across configurations"""
        results = []
        
        for layer in layers:
            for position in positions:
                for component in components:
                    print(f"Testing layer={layer}, pos={position}, comp={component}")
                    
                    X_pos = self.extract_activations(
                        pos_sentences, layer, position, component
                    )
                    X_neg = self.extract_activations(
                        neg_sentences, layer, position, component
                    )
                    
                    probe_results = self.train_probe(X_pos, X_neg)
                    
                    results.append({
                        'layer': layer,
                        'position': position,
                        'component': component,
                        'train_acc': probe_results['train_acc'],
                        'test_acc': probe_results['test_acc'],
                        'probe': probe_results['probe']
                    })
        
        return pd.DataFrame(results)
    
    def test_generalization(self, probe, new_pos, new_neg, layer, position='last'):
        """Test probe on new distribution"""
        X_new_pos = self.extract_activations(new_pos, layer, position)
        X_new_neg = self.extract_activations(new_neg, layer, position)
        X_new = np.vstack([X_new_pos, X_new_neg])
        y_new = np.array([1]*len(new_pos) + [0]*len(new_neg))
        
        accuracy = probe.score(X_new, y_new)
        predictions = probe.predict(X_new)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': y_new
        }
    
    def plot_layer_comparison(self, results_df, save_path=None):
        """Visualize layer comparison"""
        plt.figure(figsize=(10, 6))
        
        for position in results_df['position'].unique():
            df_pos = results_df[results_df['position'] == position]
            plt.plot(df_pos['layer'], df_pos['test_acc'], 
                    marker='o', label=position)
        
        plt.xlabel('Layer')
        plt.ylabel('Test Accuracy')
        plt.title('Probe Accuracy by Layer and Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_position_heatmap(self, results_df, save_path=None):
        """Heatmap of position vs layer"""
        pivot = results_df.pivot(
            index='layer', 
            columns='position', 
            values='test_acc'
        )
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.2%', cmap='viridis')
        plt.title('Probe Accuracy: Layer vs Position')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
```

### CoT Analyzer

```python
class CoTProbeAnalyzer:
    """Specialized analyzer for CoT"""
    
    def __init__(self, model):
        self.model = model
        self.reasoning_markers = {
            'reasoning': ['first', 'then', 'next', 'because', 'since'],
            'conclusion': ['therefore', 'so', 'thus', 'hence'],
            'uncertainty': ['maybe', 'possibly', 'might', 'perhaps'],
            'correction': ['wait', 'actually', 'no', 'correction']
        }
    
    def extract_cot_activations(self, text, layer=15):
        """Extract activations throughout CoT sequence"""
        _, cache = self.model.run_with_cache(text)
        acts = cache["resid_post", layer][0, :, :].cpu().numpy()
        
        tokens = self.model.tokenizer.encode(text)
        token_strs = [self.model.tokenizer.decode([t]) for t in tokens]
        
        return acts, token_strs
    
    def find_reasoning_markers(self, text):
        """Find positions of reasoning markers"""
        tokens = self.model.tokenizer.encode(text)
        token_strs = [self.model.tokenizer.decode([t]) for t in tokens]
        
        positions = {k: [] for k in self.reasoning_markers}
        
        for i, token in enumerate(token_strs):
            token_lower = token.lower().strip()
            for marker_type, words in self.reasoning_markers.items():
                if any(word in token_lower for word in words):
                    positions[marker_type].append(i)
        
        return positions
    
    def extract_at_markers(self, texts, layer=15, marker_type='conclusion'):
        """Extract activations at specific markers"""
        all_marker_acts = []
        
        for text in texts:
            acts, _ = self.extract_cot_activations(text, layer)
            markers = self.find_reasoning_markers(text)
            
            for pos in markers[marker_type]:
                if pos < len(acts):
                    all_marker_acts.append(acts[pos])
        
        return np.array(all_marker_acts) if all_marker_acts else np.array([])
    
    def compare_positions(self, faithful_texts, unfaithful_texts, layer=15):
        """Compare different position strategies"""
        results = {}
        
        # Last position
        toolkit = ProbeToolkit(self.model)
        X_f = toolkit.extract_activations(faithful_texts, layer, 'last')
        X_u = toolkit.extract_activations(unfaithful_texts, layer, 'last')
        results['last'] = toolkit.train_probe(X_f, X_u)
        
        # Mean position
        X_f = toolkit.extract_activations(faithful_texts, layer, 'mean')
        X_u = toolkit.extract_activations(unfaithful_texts, layer, 'mean')
        results['mean'] = toolkit.train_probe(X_f, X_u)
        
        # Conclusion markers
        X_f = self.extract_at_markers(faithful_texts, layer, 'conclusion')
        X_u = self.extract_at_markers(unfaithful_texts, layer, 'conclusion')
        if len(X_f) > 0 and len(X_u) > 0:
            results['conclusion'] = toolkit.train_probe(X_f, X_u)
        
        return results
```

---

## Troubleshooting

### Common Issues

#### GPU Out of Memory

```python
# Reduce batch size
# Process one at a time
for sentence in sentences:
    _, cache = model.run_with_cache(sentence)
    # Extract and clear cache
    torch.cuda.empty_cache()
```

#### Probe Accuracy ~50% (Random)

**Causes:**
1. Labels are wrong
2. Information not in this layer
3. Information not linearly accessible
4. Not enough data

**Debug:**
```python
# Check label distribution
print(f"Positive: {sum(y)}, Negative: {len(y) - sum(y)}")

# Try different layers
for layer in range(0, 12):
    # ... test each

# Check if any signal exists
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Positive')
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Negative')
```

#### High Train, Low Test Accuracy

**Overfitting. Solutions:**
```python
# More data
# Simpler model
probe = LogisticRegression(C=0.1)  # More regularization

# Check for data leakage
# Verify train/test split is correct
```

#### Code Running Very Slow

```python
# Cache activations
activations_cache = {}

def get_cached_activation(text, layer):
    key = (text, layer)
    if key not in activations_cache:
        _, cache = model.run_with_cache(text)
        activations_cache[key] = cache["resid_post", layer]
    return activations_cache[key]
```

### When to Ask for Help

**Stop immediately if:**
- Can't get model running after 3 hours
- Basic probe <50% accuracy on simple task
- Consistent errors you don't understand
- Behind schedule by >50% in practice projects

**Where to get help:**
- ARENA Discord: #technical-questions
- TransformerLens GitHub
- LessWrong: Quick questions
- Claude/Gemini with error messages

---

## Success Criteria

### End of Week 1
- [ ] Can extract activations from any layer
- [ ] Can train probe with >70% on toy task
- [ ] Understand when probes work/fail
- [ ] Completed 2+ practice exercises

### End of Week 2
- [ ] Can work with reasoning models
- [ ] Completed 1+ end-to-end practice project
- [ ] Can write clear 2-page research summary
- [ ] Have reusable probe toolkit

### End of Week 3
- [ ] Complete hour-by-hour project plan
- [ ] CoT faithfulness dataset ready
- [ ] All code templates working
- [ ] Successful mini-test of pipeline

### End of Week 4
- [ ] Completed 20-hour project
- [ ] Executive summary + main document
- [ ] Learned something concrete about probes
- [ ] Can articulate findings clearly

---

## Quick Reference

### Essential Commands

```python
# Load model
model = tl.HookedTransformer.from_pretrained("gpt2-small")

# Extract activations
_, cache = model.run_with_cache(text)
acts = cache["resid_post", layer][0, -1, :].cpu().numpy()

# Train probe
probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)
test_acc = probe.score(X_test, y_test)

# Get probe direction
direction = probe.coef_[0]
```

### Key Files to Have Open

1. This guide (mats_project_guide.md)
2. ProbeToolkit class
3. CoTProbeAnalyzer class
4. Your hour-by-hour plan
5. Results tracking document

### Remember

- **Depth over breadth:** Master probes, don't dabble in everything
- **Honest over impressive:** Partial results with honesty > overstated claims
- **Process over results:** Show good research process even if results are null
- **Practical over theoretical:** Connect to AI control applications
- **Clear over complex:** Simple, well-executed beats ambitious failure

**You've got this. Build something real. Learn something concrete. Be honest about what you find.**
