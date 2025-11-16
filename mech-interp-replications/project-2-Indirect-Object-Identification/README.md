# Project 2: Indirect Object Identification (IOI)

A replication of Redwood Research's circuit discovery work on how GPT-2 small solves the indirect object identification task.

## Overview

This project reverse-engineers a more complex circuit than induction heads, demonstrating how transformers solve a specific linguistic task: identifying indirect objects in sentences. This showcases advanced interpretability techniques for discovering and validating multi-component circuits.

**Original Paper:** [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593) (Redwood Research, 2022)

## The IOI Task

Given a sentence like:
> "When John and Mary went to the store, John gave a drink to"

The model should predict **Mary** (the indirect object) rather than **John** (who is also mentioned but is the subject).

This requires:
- Tracking multiple names in context
- Understanding grammatical roles
- Distinguishing between subject and indirect object

## The IOI Circuit

The discovered circuit contains several types of attention heads working together:

### Name Mover Heads
- **Function:** Move information about the indirect object name to the final position
- **Location:** Later layers
- **Behavior:** Directly increase logits for the correct name

### Negative Name Mover Heads
- **Function:** Suppress the subject name
- **Location:** Later layers
- **Behavior:** Decrease logits for the incorrect (duplicate) name

### S-Inhibition Heads
- **Function:** Identify and suppress the duplicate name (subject)
- **Location:** Middle layers
- **Behavior:** Detect which name appears twice

### Previous Token Heads
- **Function:** Provide positional information
- **Location:** Early layers
- **Behavior:** Similar to those in induction heads

## Notebook Structure

### Step 1: Load GPT-2 Small
- Initialize TransformerLens
- Prepare the model for circuit analysis

### Exercise 4: Setup & Baseline

#### Step 1: Create IOI Prompts
- Generate structured prompts with two names (e.g., "When [A] and [B]... [A] gave to")
- Create a dataset of varied sentence templates
- Ensure diversity in names and structure

#### Step 2: Measure Baseline Performance
- Calculate model accuracy on IOI task
- Compute logit difference: `logit(IO) - logit(S)`
- Establish clean performance metrics

#### Step 3: Create a Corrupted Dataset
- Generate alternative prompts where IOI circuit shouldn't work
- Used for activation patching experiments
- Enables causal testing of circuit components

#### Step 4: Implement Logit Attribution
- Decompose output logits by attention head
- Identify which heads contribute most to correct predictions
- Discover name mover heads through attribution scores

#### Step 5: Test Ablating Name Mover Heads
- Perform targeted ablations on identified heads
- Measure performance degradation
- Validate causal importance of circuit components

## Key Findings

### Circuit Components Identified

**Name Mover Heads (Primary Contributors):**
- [Specific heads identified in notebook]
- Strong positive attribution to indirect object
- Critical for task performance

**Supporting Components:**
- Negative name movers
- S-inhibition heads
- Previous token heads

### Causal Validation

**Ablation Results:**
- Ablating name mover heads significantly reduces IOI performance
- Circuit components are causally necessary
- Performance drops to near-random when key heads are ablated

**Logit Attribution:**
- Successfully decomposed model behavior
- Identified specific heads responsible for correct predictions
- Validated with targeted interventions

## Technical Skills Demonstrated

### Advanced Interpretability Techniques
- ✅ Logit attribution and decomposition
- ✅ Structured dataset creation for interpretability
- ✅ Clean vs. corrupted comparison baselines
- ✅ Targeted ablation studies
- ✅ Multi-component circuit analysis

### Experimental Design
- ✅ Designing controlled experiments
- ✅ Creating appropriate baselines
- ✅ Systematic hypothesis testing
- ✅ Quantitative validation of qualitative findings

### Programming & Analysis
- ✅ Complex data generation and manipulation
- ✅ Attribution score calculations
- ✅ Performance metrics (logit difference, accuracy)
- ✅ Visualization of multi-dimensional results

## Running the Notebook

### Option 1: Google Colab (Recommended)
1. Click "Open in Colab" badge in the notebook
2. Run cells sequentially
3. Free GPU provided automatically

### Option 2: Local Setup
```bash
pip install transformer-lens torch plotly
jupyter notebook IOI_notebook.ipynb
```

**Requirements:**
- GPU recommended for faster computation
- ~4GB RAM minimum
- Python 3.8+

## Results & Insights

This replication demonstrates:
- **Circuit Complexity:** Real-world tasks use multi-head circuits
- **Logit Attribution Power:** Effective technique for circuit discovery
- **Causal Validation:** Ablation confirms theoretical understanding
- **Research Skills:** Ability to replicate complex interpretability work

## Comparison to Induction Heads

| Aspect | Induction Heads | IOI Circuit |
|--------|----------------|-------------|
| Complexity | 2-layer, 2-component | Multi-layer, 4+ component types |
| Discovery Method | Attention patterns | Logit attribution |
| Task | Pattern completion | Grammatical reasoning |
| Validation | Ablation, patching | Logit attribution + ablation |

## Extensions & Future Work

Possible directions to extend this work:
- Investigate IOI circuits in larger models (GPT-2 medium/large)
- Test on related grammatical tasks
- Study how the circuit forms during training
- Examine failure cases and limitations
- Explore interactions with other circuits
- Implement more sophisticated patching experiments

## Challenges & Lessons Learned

### Computational Complexity
- Logit attribution requires careful caching
- Large batch sizes needed for stable results
- Memory management important for longer sequences

### Dataset Design
- Crafting good corrupted datasets is non-trivial
- Template diversity matters for robust findings
- Need to control for confounding factors

### Circuit Discovery
- Multiple techniques needed (attribution + ablation)
- Iterative hypothesis testing required
- Not all heads have clean, interpretable functions

## References

### Primary Source
- Wang, K., et al. (2022). [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593). arXiv:2211.00593.

### Related Work
- Elhage, N., et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
- Nanda, N., et al. (2023). [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217).

### Learning Resources
- [ARENA 3.0 - IOI Section](https://www.arena.education/)
- [TransformerLens IOI Tutorial](https://transformerlensorg.github.io/TransformerLens/)
- [Redwood Research Blog Post](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only)

## Code Structure

The notebook follows a systematic approach:
1. **Setup:** Load model and understand architecture
2. **Data:** Create clean and corrupted datasets
3. **Baseline:** Establish performance metrics
4. **Discovery:** Use logit attribution to find important heads
5. **Validation:** Ablate heads to confirm causal importance
6. **Analysis:** Interpret results and understand the circuit

## Questions or Issues?

For learning purposes:
1. Consult the original paper for theoretical details
2. Check TransformerLens docs for implementation questions
3. Review ARENA 3.0 materials for additional explanations
4. Compare approaches with other replications

## Acknowledgments

- Original research by Redwood Research team
- TransformerLens library by Neel Nanda
- ARENA 3.0 course materials for guidance

## License

Educational replication for learning purposes. Original research credit to Redwood Research.
