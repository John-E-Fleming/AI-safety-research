# AI Safety Research - Mechanistic Interpretability Portfolio

**Context File for Claude Code Sessions**
*Last Updated: 2025-12-08*

---

## Quick Start for New Sessions

This is a mechanistic interpretability research portfolio focused on building expertise through circuit replications and novel probe-based research. The codebase contains:

1. **Completed replications** of foundational mech interp papers (induction heads, IOI circuit)
2. **Active research project** on tool validation for reasoning models (probe-focused)
3. **Extensive reference materials** (~3.1MB of mech interp documentation)

**Primary Technologies:** TransformerLens (GPT-2), nnsight (Qwen/reasoning models), PyTorch, scikit-learn, Jupyter
**Development Environment:** Vast.ai (GPU for Qwen), Google Colab, local Jupyter
**Git Branch:** main
**Current Phase:** Week 2 - Tool validation study for reasoning model interpretability

---

## Directory Structure

```
AI-safety-research/
├── mech-interp-replications/          # Foundational circuit replications
│   ├── project-1-induction-heads/     # COMPLETED (recently refactored)
│   │   ├── induction_heads_notebook.ipynb (2.9MB)
│   │   └── README.md
│   └── project-2-Indirect-Object-Identification/  # COMPLETED
│       ├── IOI_notebook.ipynb (1.0MB)
│       └── README.md
│
├── mech-interp-projects/              # Novel research projects
│   └── probe-based-faithfulness-detection/  # IN PROGRESS
│       ├── day3-4_first_probes.ipynb        # COMPLETED - Sentiment probing basics
│       ├── day5-6_advanced_techniques.ipynb # COMPLETED - Position/component analysis
│       ├── day8-9_nnsight_setup.ipynb       # COMPLETED - nnsight + Qwen setup
│       ├── mats_project_guide.md            # Comprehensive project guide with learnings
│       └── README.md (40KB comprehensive guide)
│
├── Mechanistic_Interpretability_Context_Documents/
│   └── Mech Interp Context Docs/      # 15 reference documents (~3.1MB)
│       ├── neel_glossary_60k.md
│       ├── research+writing_advice_45k.md
│       ├── arena_all_650k.txt
│       ├── transformer_lens_all_400k.txt
│       ├── nnsight_all_270k.txt
│       └── [10 more reference files]
│
├── README.md                          # Main project documentation
├── CONTRIBUTING.md                    # Contribution guidelines
├── SETUP.md                           # Setup instructions
└── requirements_old.txt
```

---

## Current Focus: Tool Validation for Reasoning Model Interpretability

**Location:** `mech-interp-projects/probe-based-faithfulness-detection/`
**Status:** Week 2 - Transitioning to exploratory tool validation study
**Goal:** Investigate which interpretability tools (specifically probes) capture reasoning-relevant structure in CoT models

### Project Reframe (Dec 8, 2025)

Based on recent literature review and guidance from Neel Nanda, the project has been reframed from hypothesis-driven ("Can probes detect unfaithful CoT?") to **exploratory/tool-validating** ("What questions can we ask about reasoning models, and do probes actually work for them?").

This reframe is:
- More achievable in 20 hours
- More likely to produce publishable findings (negative results count)
- Better aligned with the current state of the field

### Core Research Questions

**Primary Question:** Which interpretability tools capture reasoning-relevant structure in CoT models?

**Specific Questions to Investigate:**

| Question | Method | Success Criterion |
|----------|--------|-------------------|
| Q1: Do activation patterns distinguish sentence types? | Train probe on sentence taxonomy | >70% accuracy, generalizes across problems |
| Q2: Do probes trained on sentence type generalize? | Test on held-out problems/domains | <15% accuracy drop |
| Q3: Can probes detect "hint influence"? | Train on hinted vs unhinted CoT | Above-chance discrimination |
| Q4: Do probe-based importance estimates correlate with resampling-based importance? | Compare probe confidence with counterfactual importance | Significant positive correlation |

### Key Literature Context

Three recent papers provide essential context for this project:

#### 1. Thought Anchors (Bogdan et al., 2025)
**Key finding:** Plan generation and uncertainty management sentences have HIGH counterfactual importance; active computation has LOW importance.

**Implication:** Faithfulness isn't about "doing math correctly" but about "genuine decision-making." Probes should focus on plan generation, not computation.

**8-Category Sentence Taxonomy:**
1. Problem Setup - parsing/rephrasing
2. Plan Generation - stating plans, meta-reasoning ← HIGH IMPORTANCE
3. Fact Retrieval - recalling formulas without computation
4. Active Computation - algebra, calculations ← LOW IMPORTANCE (overdetermined)
5. Uncertainty Management - confusion, backtracking ← HIGH IMPORTANCE
6. Result Consolidation - aggregating results
7. Self Checking - verification
8. Final Answer Emission - stating answer

#### 2. Thought Branches (Macar & Bogdan et al., 2025)
**Key finding:** Self-preservation statements have LOW resilience and LOW causal impact. Unfaithfulness is "nudged reasoning" — subtle, diffuse, cumulative biases — not discrete lies.

**Implications:**
- Don't hand-edit CoTs to create unfaithful examples (off-policy interventions have near-zero effect)
- Use their hint methodology for naturalistic unfaithful traces
- Probes should detect distributed bias patterns, not single deceptive statements

**Resilience Metric:** Measures how many times you must intervene before a sentence's semantic content stops reappearing downstream.

#### 3. Base Models Know How to Reason (Venhoff et al., 2025)
**Key finding:** Thinking models learn *when* to deploy reasoning mechanisms, not *how* to reason. Base models already contain latent capacity; RLVR teaches timing.

**Evidence:** Hybrid model (base + steering vectors with thinking model timing) recovers **91% of performance gap** while steering only **12% of tokens**.

**Implication:** Reasoning mechanisms are linearly steerable, supporting the linear representation hypothesis underlying probing. 10-25 categories is tractable.

### Critical Insight from Literature

**High activation norm ≠ High causal importance**

Your earlier finding (mathematical tokens show HIGHER activation norms) combined with Thought Anchors (active computation has LOWER counterfactual importance) suggests:
- Activation magnitude tracks computational effort, not causal influence
- This motivates testing whether *directions* (probes) do better than *magnitudes* (norms)

### Progress Summary

**COMPLETED (Days 1-6):**
- ✅ Day 3-4: Basic sentiment probing with TransformerLens/GPT-2
- ✅ Day 5-6: Advanced techniques (position analysis, attention heads, MLP probing)
- ✅ Key conceptual learnings documented in `mats_project_guide.md`

**COMPLETED (Days 8-9):**
- ✅ nnsight setup with Qwen2.5-7B-Instruct complete
- ✅ Activation patching demo working (France→Germany capital)
- ✅ Activation norm analysis showing correlation with computational load

**KEY DISCOVERY - Activation Norms Correlate with Computation:**

| Marker Type | Norm vs Overall | Interpretation |
|-------------|-----------------|----------------|
| `mathematical` | ↑ HIGHER (+3.11) | Active computation for next token |
| `reasoning_steps` | ↓ LOWER (-4.22) | Structural/formatting tokens |
| `reasoning_start` | ↓ LOWER (-2.57) | Setup/restatement, not computation |
| `conclusion` | ↓ LOWER (-3.11) | Answer already computed, just outputting |

**NEW INTERPRETATION:** Given Thought Anchors findings, high norms at math tokens may indicate effort, not importance. Faithful CoT may show high norms at PLANNING tokens; unfaithful may show uniformly low norms at planning (no real decisions).

**UPCOMING (Revised Timeline):**
- Week 2 (remaining): Annotate dataset with sentence taxonomy, train sentence-type classifier probe
- Week 3: Generate hinted vs unhinted CoT pairs, test probe discrimination, characterize what probes capture
- Week 4: Small-scale resampling validation (if feasible), write-up

### Key Learnings from Probe Exercises (Days 3-6)

**Critical conceptual corrections documented:**

1. **"Later layers have more information" is WRONG**
   - Later layers contain *different* information, not more
   - Late layers optimized for output generation, may be worse for probing internal state
   - Middle layers might preserve reasoning state better
   - **Paper support:** Venhoff et al. use steering at 37% depth for best results

2. **MLP vs Attention serve different roles**
   - Attention: routes information between positions
   - MLP: transforms information within position
   - Both involve nonlinearity (attention has softmax, MLP has GELU)

3. **Position matters due to computational roles**
   - Last token often contaminated by output preparation
   - Second-to-last token may have cleaner semantic representation
   - **Paper support:** All three papers work at sentence-level, not token-level

4. **Residual stream accumulates, doesn't overwrite**
   - Each layer adds to residual stream
   - MLP probes can be worse than residual stream probes if MLP isn't computing the target feature

5. **Activation norms grow exponentially across layers (IMPORTANT for probing)**
   - Early layers: ~27 norm, Middle: ~70, Late: ~275 (in Qwen 28-layer model)
   - **When comparing probes across layers, NORMALIZE activations first**

### Revised Hypotheses (Based on Literature)

- **H1 (Revised):** Middle layers (~37% depth) will outperform late layers for faithfulness-relevant probes (supported by Venhoff steering results)
- **H2 (Revised):** Probes trained on plan generation sentences will work better than probes trained on computation sentences (based on Thought Anchors)
- **H3 (Unchanged):** Probes generalize within task types but not across tasks
- **H4 (Strengthened):** Probes are vulnerable to adversarial stylistic changes because unfaithfulness is "nudged reasoning" distributed across many subtle biases
- **H5 (New):** Sentence-averaged activations will outperform token-level activations for reasoning mechanism detection (based on all three papers using sentence-level)

### Current State of Interpretability Tools for Reasoning Models

| Tool | Status | Evidence |
|------|--------|----------|
| Resampling | ✓ Works well (expensive) | Thought Anchors/Branches |
| SAE-based taxonomy | ✓ Works for discovering mechanisms | Venhoff et al. |
| Steering vectors | ✓ Works for inducing mechanisms | Venhoff et al. |
| **Linear probes for faithfulness** | **? Untested** | **This project** |

### Why nnsight Instead of TransformerLens?

| Aspect | TransformerLens | nnsight |
|--------|-----------------|---------|
| Model support | GPT-2, GPT-Neo only | ANY HuggingFace model |
| Qwen/Llama/DeepSeek | NOT SUPPORTED | Fully supported |
| CoT reasoning models | Cannot use | Required for this project |

**Bottom line:** TransformerLens was for learning on GPT-2. Real CoT faithfulness research requires reasoning models like Qwen, which need nnsight.

### Current Notebooks

| Notebook | Status | Content |
|----------|--------|---------|
| `day3-4_first_probes.ipynb` | ✅ Complete | Sentiment probing, layer comparison, generalization testing |
| `day5-6_advanced_techniques.ipynb` | ✅ Complete | Position analysis, attention heads, MLP probing |
| `day8-9_nnsight_setup.ipynb` | ✅ Complete | nnsight setup, Qwen loading, activation extraction, patching |
| `day8-9_cot_structure_and_dataset.ipynb` | 🔄 Current | CoT structure analysis, marker detection, dataset building |

### Important Files
- **Project Guide:** `mats_project_guide.md` - Comprehensive guide with key learnings section
- **Setup Notebook:** `day8-9_nnsight_setup.ipynb` - nnsight/Qwen setup and verification
- **CoT Notebook:** `day8-9_cot_structure_and_dataset.ipynb` - CoT analysis and dataset creation
- **Dependencies:** nnsight, transformers, scikit-learn (for Qwen work)

### Key nnsight API Notes (for Future Reference)

```python
# Trigger model loading (nnsight lazy loads)
with model.trace("Hello"):
    _ = model.model.layers[0].output[0].save()

# Shape differences - IMPORTANT!
# layers[].output[0]: [seq_len, hidden_size] - NO batch dim
# mlp.output: [seq_len, hidden_size] - NO batch dim
# self_attn.output[0]: [batch, seq_len, hidden_size] - HAS batch dim

# Convert to float32 before computing norms (float16 overflow)
acts = hidden.float().detach().cpu().numpy()
```

---

## Dataset Construction Guidelines (Based on Literature)

### DO NOT:
- Hand-edit CoTs to create "unfaithful" examples (off-policy interventions have near-zero effect per Thought Branches)
- Assume computation tokens are most important (they're overdetermined)
- Use single token positions when sentence-level is more appropriate

### DO:
- Use hint methodology from Thought Branches (professor hints on MMLU/MATH)
- Find cases where hint changes answer but isn't mentioned in CoT
- Annotate with Thought Anchors' 8-category taxonomy
- Work at sentence-level following all three papers
- Generate naturalistic unfaithful traces via prompting, not editing

### Sentence Taxonomy for Annotation

Based on Thought Anchors (Bogdan et al., 2025):

| Category | Description | Expected Importance |
|----------|-------------|---------------------|
| Problem Setup | Parsing, rephrasing problem | Low |
| Plan Generation | Stating plans, meta-reasoning | **HIGH** |
| Fact Retrieval | Recalling formulas | Medium |
| Active Computation | Algebra, calculations | **LOW** (overdetermined) |
| Uncertainty Management | Confusion, backtracking | **HIGH** |
| Result Consolidation | Aggregating results | Medium |
| Self Checking | Verification | Medium |
| Final Answer Emission | Stating answer | Low |

---

## Potential Contributions

Regardless of whether probes "work," this project can contribute:

**If probes work:**
- Probes are a cheap approximation to expensive resampling-based faithfulness measures
- Practical tool for scalable CoT monitoring

**If probes don't work:**
- Documents fundamental limitations of linear probing for faithfulness detection
- Suggests the field needs different tools

**If mixed results (most likely):**
- Probes work for X but not Y — characterizing when probes are appropriate
- Most interesting and useful outcome

---

## Glossary Updates (Reasoning Model Specific)

### From Literature Review

- **Counterfactual Importance:** DKL between answer distribution with/without a sentence (Thought Anchors)
- **Counterfactual++ Importance:** Same but only counting rollouts where content never reappears (Thought Branches)
- **Resilience:** Number of interventions needed before a sentence's semantic content stops reappearing downstream
- **Nudged Reasoning:** Unfaithfulness as subtle, diffuse, cumulative bias rather than discrete lies
- **On-policy vs Off-policy Interventions:** On-policy = model could have generated; Off-policy = handwritten/edited (off-policy has near-zero effect)
- **Thought Anchors:** Sentences with high counterfactual importance that guide reasoning
- **Receiver Heads:** Attention heads that narrow focus on specific "anchor" sentences

### Existing Glossary

- **Residual Stream:** Information flow through transformer layers
- **Activation Patching:** Replace activations to test causal impact
- **Ablation Studies:** Zero/mean out components to test necessity
- **Logit Attribution:** Decompose output predictions to individual components
- **Probes:** Linear classifiers trained on internal activations

---

## Learning Path Recommendations

### For New Sessions Working on This Codebase

1. **Start with Induction Heads** (if unfamiliar with mech interp)
   - Read: `mech-interp-replications/project-1-induction-heads/README.md`
   - Run: `induction_heads_notebook.ipynb`
   - Understand: attention patterns, ablations, activation patching

2. **Progress to IOI Circuit** (more complex)
   - Read: `mech-interp-replications/project-2-Indirect-Object-Identification/README.md`
   - Run: `IOI_notebook.ipynb`
   - Understand: multi-component circuits, logit attribution

3. **Current Work: Tool Validation for Reasoning Models**
   - Read: `mech-interp-projects/probe-based-faithfulness-detection/mats_project_guide.md`
   - Review key papers: Thought Anchors, Thought Branches, Base Models Know How
   - Focus: Current week's objectives in the revised timeline

### Essential Reference Materials Priority
1. `neel_glossary_60k.md` - Learn terminology
2. `research+writing_advice_45k.md` - Understand research approach
3. `transformer_lens_docs_17k.txt` - Quick API reference
4. Project-specific READMEs - Context for each project

---

## Repository Purpose & Goals

### Primary Objective
- Build and demonstrate hands-on AI safety research expertise
- Focus on mechanistic interpretability as primary domain
- Create portfolio for AI safety research roles

### Secondary Interests
- AI Control methodologies
- SAEs and feature visualization
- Superposition and interpretability
- Adversarial robustness

### Target Audience
- Potential employers in AI safety organizations
- Collaborators in mech interp research
- Educational resource for learning mechanistic interpretability

---

## Troubleshooting & Common Issues

### GPU/CUDA Issues
- Ensure CUDA 11.8 compatibility
- Use `torch.cuda.is_available()` to verify GPU access
- Colab: Runtime → Change runtime type → GPU

### TransformerLens Issues
- If import fails: `pip install transformer-lens`
- If model loading fails: Check HuggingFace connectivity
- Cache issues: `model.reset_hooks()` after experiments

### Notebook Too Large
- Clear outputs: Cell → All Output → Clear
- Limit plot sizes: Use `fig.write_html()` instead of inline display
- Keep notebooks under 5MB when possible

### Probe Training Issues
- Check activation shape: Should be (n_samples, d_model)
- Normalize features if probe doesn't converge
- Use `max_iter=1000` or higher for LogisticRegression
- **NEW:** Normalize across layers before comparing (exponential norm growth)

---

## Quick Reference: File Locations

### Notebooks
- Induction Heads: `mech-interp-replications/project-1-induction-heads/induction_heads_notebook.ipynb`
- IOI Circuit: `mech-interp-replications/project-2-Indirect-Object-Identification/IOI_notebook.ipynb`
- Faithfulness Detection: `mech-interp-projects/probe-based-faithfulness-detection/`

### Documentation
- Main README: `README.md`
- Contributing: `CONTRIBUTING.md`
- Setup: `SETUP.md`
- Project Guide: `mech-interp-projects/probe-based-faithfulness-detection/mats_project_guide.md`

### Context Materials
- All in: `Mechanistic_Interpretability_Context_Documents/Mech Interp Context Docs/`
- Most useful: `neel_glossary_60k.md`, `research+writing_advice_45k.md`

---

## Working with This Codebase

### Before Starting a Session
1. Check git status and recent commits
2. Review current project status (tool validation study timeline)
3. Identify which week/phase of the project we're in
4. Read relevant README.md for context

### During Work
1. Follow existing naming conventions
2. Keep notebooks organized with markdown headers
3. Add comments for complex operations
4. Test on small examples before full runs
5. Clear outputs if notebook size grows

### Before Ending a Session
1. Save and checkpoint notebooks
2. Clear unnecessary outputs
3. Commit with meaningful messages using standard prefixes
4. Update relevant README.md if project status changed

### When Stuck
1. Check troubleshooting section in project README
2. Review reference materials (especially transformer_lens docs)
3. Look at similar code in completed projects
4. Verify basic sanity checks (shapes, dtypes, GPU availability)

---

## Notes for Claude Code Sessions

### Project State (Dec 8, 2025)
- **Active work:** Tool validation study - Week 2 (nnsight/Qwen + literature integration)
- **Recent update:** Project reframed based on Neel Nanda guidance and literature review
- **Key papers reviewed:** Thought Anchors, Thought Branches, Base Models Know How
- **Key deliverable:** `mats_project_guide.md` contains all learnings and revised plan
- **Next step:** Annotate dataset with sentence taxonomy, train sentence-type classifier

### TransformerLens vs nnsight Quick Reference

| Operation | TransformerLens | nnsight |
|-----------|-----------------|---------|
| Load model | `HookedTransformer.from_pretrained()` | `LanguageModel()` |
| Residual stream | `cache["resid_post", layer]` | `model.model.layers[layer].output[0]` |
| MLP output | `cache["blocks.{layer}.hook_mlp_out"]` | `model.model.layers[layer].mlp.output` |
| Attention output | `cache["attn_out", layer]` | `model.model.layers[layer].self_attn.output[0]` |
| Save values | Automatic | Explicit `.save()` required |

### Key Conceptual Points (from literature + tutoring)
1. Later layers ≠ more information (different, not more)
2. Attention routes information; MLP transforms information
3. Last token position contaminated by output preparation
4. Residual stream accumulates (doesn't overwrite)
5. High activation norm ≠ high causal importance (key insight from Thought Anchors)
6. Unfaithfulness is "nudged reasoning" not discrete lies (Thought Branches)
7. Reasoning mechanisms exist in base models; thinking models learn timing (Venhoff)

### Preferred Workflow
1. Understand context from relevant READMEs and `mats_project_guide.md`
2. Work in Jupyter notebooks interactively
3. Follow systematic research methodology (Explore → Understand → Distill)
4. Maintain high code quality standards
5. Document findings and limitations honestly

### Code Style Preferences
- Descriptive variable names over short abbreviations
- Markdown headers to organize notebook sections
- Inline visualizations with proper labels
- Comments explaining "why" not "what"

### Research Standards
- Verify findings against published results when replicating
- Include systematic comparisons (layers, positions, tasks)
- Test edge cases and failure modes
- Document limitations and assumptions
- Prioritize clarity in explanations
- **Report negative results honestly — they're valuable**

---

*This context file is designed to help Claude Code instances get up to speed quickly on this mechanistic interpretability research codebase. For detailed information on specific projects, always refer to the individual README.md files in each project directory.*

**For the tool validation project specifically, see:** `mech-interp-projects/probe-based-faithfulness-detection/mats_project_guide.md`
