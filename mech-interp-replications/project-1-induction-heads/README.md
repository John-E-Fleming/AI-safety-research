# Project 1: Induction Heads

A replication of Anthropic's research on induction heads and in-context learning in transformer models.

## Overview

This project implements and explores **induction heads** - a key mechanistic component that enables transformers to perform in-context learning. The notebook works through how a simple two-layer circuit can implement pattern matching and completion.

**Original Paper:** [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Anthropic, 2022)

## What are Induction Heads?

Induction heads are attention heads that:
1. Look for patterns like `[A][B] ... [A]` in the input
2. When they see `[A]` again, they attend to the token after the previous `[A]` (which is `[B]`)
3. This allows the model to predict `[B]` should come next

This is a fundamental building block for in-context learning - the ability to learn from examples in the prompt without updating weights.

## The Circuit Structure

### Layer 0: Previous Token Heads
- Simple attention heads that attend from position `i` to position `i-1`
- Write "this token was preceded by X" into the residual stream

### Layer 1: Induction Heads
- Attend to positions where the same token appeared before
- Use information from previous token heads to predict the next token
- Complete the pattern matching behavior

## Notebook Structure

### Step 1: Load GPT-2 Small
- Set up TransformerLens and load the model
- Understand the model architecture (12 layers, 12 heads per layer)

### Step 2: Create Induction-Friendly Sequences
- Design inputs that highlight induction behavior
- Use repeated random sequences to test pattern matching

### Step 3: Run Model and Cache Activations
- Execute forward passes and store intermediate activations
- Prepare data for analysis

### Step 4: Detect Previous Token Heads (Layer 0)
- Analyze attention patterns in early layers
- Identify heads that implement the `i → i-1` pattern

### Step 5: Detect Induction Heads (Layer 1)
- Search for heads showing characteristic induction patterns
- Verify they attend to previous instances of the current token

### Step 6: Visualize Attention Patterns
- Create interactive visualizations using CircuitsVis
- Examine how information flows through the circuit

### Practice Exercises

#### Exercise 1: Zero Ablation of Induction Heads
**Goal:** Prove induction heads are causally responsible for copying behavior
- Ablate specific heads and measure performance degradation
- Compare zero ablation vs. mean ablation strategies

#### Exercise 2: Mean Ablation vs Zero Ablation
**Goal:** Understand different ablation techniques
- Compare how each method affects model behavior
- Learn when to use each approach

#### Exercise 3: Activation Patching (Causal Tracing)
**Goal:** Trace information flow through the circuit
- Implement activation patching on keys, queries, and values
- Determine which components are critical for induction

## Key Findings

### Identified Circuits
- **Previous Token Heads:** Layer 0, Heads [specific heads identified in notebook]
- **Induction Heads:** Layer 1, Heads [specific heads identified in notebook]

### Causal Validation
- Ablating induction heads significantly reduces in-context learning performance
- Activation patching confirms information flows through specific circuit paths
- Both layers are necessary for the complete induction behavior

## Technical Skills Demonstrated

### Interpretability Techniques
- ✅ Attention pattern analysis
- ✅ Zero ablation studies
- ✅ Mean ablation studies
- ✅ Activation patching (causal tracing)
- ✅ Logit difference metrics

### Programming & Tools
- ✅ TransformerLens library proficiency
- ✅ PyTorch tensor manipulation
- ✅ Interactive visualizations with CircuitsVis and Plotly
- ✅ Efficient caching and computation

### Research Skills
- ✅ Replicating published research
- ✅ Designing interpretability experiments
- ✅ Hypothesis testing with ablations
- ✅ Clear technical documentation

## Running the Notebook

### Option 1: Google Colab (Recommended)
1. Open the notebook in Colab using the badge at the top
2. Run all cells in order - no setup required!
3. GPU will be automatically allocated

### Option 2: Local Jupyter
```bash
pip install transformer-lens circuitsvis plotly
jupyter notebook induction_heads_notebook.ipynb
```

**Note:** GPU recommended but not required for this project.

## Results & Insights

This replication successfully:
- Identified the same induction circuits found in the original paper
- Demonstrated their causal importance through ablation
- Validated the theoretical understanding with empirical evidence
- Provided hands-on experience with core interpretability techniques

## Extensions & Future Work

Possible directions to extend this work:
- Test induction heads on different model architectures (GPT-Neo, Pythia, etc.)
- Investigate how induction heads form during training
- Explore failure cases where induction doesn't work
- Study the role of induction heads in more complex in-context learning tasks
- Examine interactions with other circuit types

## References

### Primary Source
- Elhage, N., et al. (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). Transformer Circuits Thread.

### Related Work
- Olsson, C., et al. (2022). [In-context learning and induction heads](https://arxiv.org/abs/2209.11895). arXiv.
- Nanda, N. (2022). [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)

### Learning Resources
- [ARENA 3.0 - Induction Heads Section](https://www.arena.education/)
- [Neel Nanda's YouTube Walkthrough](https://www.youtube.com/channel/UCBMJ0D-omcRay8dh4QT0doQ)

## Questions or Issues?

This is a learning project - if you find errors or have questions:
1. Review the original paper for clarification
2. Check TransformerLens documentation
3. Compare with ARENA 3.0 materials

## License

Educational replication for learning purposes. Original research credit to Anthropic.
