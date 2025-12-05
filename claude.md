# AI Safety Research - Mechanistic Interpretability Portfolio

**Context File for Claude Code Sessions**
*Last Updated: 2025-12-05*

---

## Quick Start for New Sessions

This is a mechanistic interpretability research portfolio focused on building expertise through circuit replications and novel probe-based research. The codebase contains:

1. **Completed replications** of foundational mech interp papers (induction heads, IOI circuit)
2. **Active research project** on faithfulness detection using probes
3. **Extensive reference materials** (~3.1MB of mech interp documentation)

**Primary Technologies:** TransformerLens (GPT-2), nnsight (Qwen/reasoning models), PyTorch, scikit-learn, Jupyter
**Development Environment:** Vast.ai (GPU for Qwen), Google Colab, local Jupyter
**Git Branch:** main
**Current Phase:** Week 2 - Transitioning from TransformerLens to nnsight for reasoning model probing

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
│       ├── day8-9_nnsight_setup.ipynb       # CURRENT - nnsight + Qwen setup
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

## Current Focus: Faithfulness Detection with Probes

**Location:** `mech-interp-projects/probe-based-faithfulness-detection/`
**Status:** Week 2 - Transitioning to nnsight/Qwen (Day 8-9)
**Goal:** Execute a sophisticated 20-hour research project investigating when probes can detect unfaithful Chain-of-Thought (CoT) reasoning

### Research Question
"What fundamental properties determine when probes can detect unfaithful CoT, and when do they fail?"

### Progress Summary

**COMPLETED (Days 1-6):**
- ✅ Day 3-4: Basic sentiment probing with TransformerLens/GPT-2
- ✅ Day 5-6: Advanced techniques (position analysis, attention heads, MLP probing)
- ✅ Key conceptual learnings documented in `mats_project_guide.md`

**CURRENT (Days 8-9):**
- 🔄 Setting up nnsight with Qwen2.5-7B-Instruct
- 🔄 Learning nnsight API (different from TransformerLens)
- 🔄 Notebook created: `day8-9_nnsight_setup.ipynb`
- ⏳ Next: Understanding CoT structure, building CoT dataset

**UPCOMING:**
- Week 3: CoT dataset creation, analysis tools
- Week 4: 20-hour intensive research execution

### Key Learnings from Probe Exercises (Days 3-6)

**Critical conceptual corrections documented:**

1. **"Later layers have more information" is WRONG**
   - Later layers contain *different* information, not more
   - Late layers optimized for output generation, may be worse for probing internal state
   - Middle layers might preserve reasoning state better

2. **MLP vs Attention serve different roles**
   - Attention: routes information between positions
   - MLP: transforms information within position
   - Both involve nonlinearity (attention has softmax, MLP has GELU)

3. **Position matters due to computational roles**
   - Last token often contaminated by output preparation
   - Second-to-last token may have cleaner semantic representation

4. **Residual stream accumulates, doesn't overwrite**
   - Each layer adds to residual stream
   - MLP probes can be worse than residual stream probes if MLP isn't computing the target feature

### Revised Hypotheses (Based on Learnings)

- **H1 (Revised):** Later layers may be *worse* for faithfulness because they're optimized for output. Test middle layers vs late layers.
- **H2 (Addition):** Also test second-to-last tokens, not just conclusion tokens
- **H3 (Unchanged):** Probes generalize within task types but not across tasks
- **H4 (Unchanged):** Probes are vulnerable to adversarial stylistic changes
- **H5 (New):** Individual attention heads may outperform residual stream probes

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
| `day8-9_nnsight_setup.ipynb` | 🔄 Current | nnsight setup, Qwen loading, activation extraction |

### Important Files
- **Project Guide:** `mats_project_guide.md` - Comprehensive guide with key learnings section
- **Setup Notebook:** `day8-9_nnsight_setup.ipynb` - nnsight/Qwen setup and verification
- **Dependencies:** nnsight, transformers, scikit-learn (for Qwen work)

---

## Completed Projects

### Project 1: Induction Heads (COMPLETED, Recently Refactored)

**Location:** `mech-interp-replications/project-1-induction-heads/`
**Based on:** Anthropic's "In-context Learning and Induction Heads" (2022)
**Recent Activity:** Nov 23, 2025 - Code refactoring for induction head detection

**What it demonstrates:**
- How transformers perform in-context learning via two-layer circuits
- **Previous Token Heads** (Layer 0) attend to previous tokens
- **Induction Heads** (Layer 1) compose with previous token heads to predict repeated sequences
- Ablation studies and activation patching techniques

**Key Techniques:**
- Attention pattern analysis and visualization
- Zero/mean ablation studies
- Activation patching (causal intervention)
- Logit difference metrics for performance measurement

**Implementation:** GPT-2 small using TransformerLens
**Notebook:** `induction_heads_notebook.ipynb` (2.9MB)

### Project 2: Indirect Object Identification Circuit (COMPLETED)

**Location:** `mech-interp-replications/project-2-Indirect-Object-Identification/`
**Based on:** Redwood Research's "Interpretability in the Wild" (2022)
**Complexity Level:** More advanced than induction heads

**What it demonstrates:**
- Circuit discovery for specific tasks (identifying indirect objects in sentences like "John gave Mary the book" → Mary)
- Multi-component circuit with 4+ head types working together
- **Name mover heads** - Move correct name to output
- **S-inhibition heads** - Suppress subject name
- **Negative heads** - Various inhibition roles
- **Previous token heads** - Positional information

**Key Techniques:**
- Logit attribution and decomposition
- Structured dataset creation (clean vs. corrupted baselines)
- Targeted ablation studies
- Multi-layer, multi-component analysis

**Implementation:** GPT-2 small using TransformerLens
**Notebook:** `IOI_notebook.ipynb` (1.0MB)

---

## Reference Materials Inventory

**Location:** `Mechanistic_Interpretability_Context_Documents/Mech Interp Context Docs/`
**Total Size:** ~3.1MB across 15 documents

### Core Conceptual Resources

1. **neel_glossary_60k.md** (60KB)
   - Comprehensive mech interp terminology glossary
   - Covers: features, circuits, techniques, transformers, foundations
   - By Neel Nanda (Dec 2022)

2. **research+writing_advice_45k.md** (45KB)
   - "How I Think About My Research Process" by Neel Nanda
   - 4-stage methodology: Ideation → Exploration → Understanding → Distillation
   - Practical advice on research taste, experiment design, prioritization
   - Emphasis on skepticism and iteration

### Course Materials

3. **ARENA 3.0** (970KB total)
   - `arena_all_650k.txt` - Complete course materials
   - `arena_short_320k.txt` - Condensed version
   - Primary comprehensive mech interp curriculum

### Tool Documentation

4. **TransformerLens** (742KB total) - Primary analysis tool
   - `transformer_lens_all_400k.txt` - Complete documentation
   - `transformer_lens_docs+code_325k.txt` - Docs + implementation examples
   - `transformer_lens_docs_17k.txt` - Quick reference
   - `transformer_lens_notebooks_72k.md` - Example notebooks

5. **nnsight** (540KB total) - For reasoning models
   - `nnsight_all_270k.txt` - Complete library docs
   - `nnsight_docs+code_120k.txt` - With code examples
   - `nnsight_notebooks_150k.md` - Example notebooks

6. **HuggingFace Transformers** (170KB)
   - `huggingface_transformers_short_170k.txt`
   - For model loading and tokenization

7. **default_600k.md** (600KB) - Comprehensive combined reference

---

## Technologies & Dependencies

### Core ML Stack
- **PyTorch** - Deep learning framework
- **TransformerLens** - Mechanistic interpretability library (by Neel Nanda)
- **scikit-learn** - Logistic regression and ML utilities
- **NumPy** - Numerical computing

### Model Access
- **HuggingFace Transformers** - Model loading
- **Accelerate** - Distributed computation

### Visualization
- **Plotly** - Interactive plots
- **CircuitsVis** - Circuit visualization (git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python)
- **Matplotlib** - Static plots

### Data Handling
- **Pandas** - DataFrames for analysis
- **Datasets** - HuggingFace datasets

### Development
- **Jupyter/Jupyter Notebook** - Interactive notebooks
- **ipykernel** - Jupyter kernel
- **CUDA** - GPU acceleration (optional but recommended)

### For Reasoning Models (Current Focus)
- **nnsight** - Primary tool for accessing internals of Qwen/Llama/etc.
- **Qwen2.5-7B-Instruct** - Target reasoning model for CoT research
- **Vast.ai** - Cloud GPU provider for running 7B models

---

## Coding Conventions & Standards

### Naming Conventions
- **Folders:** `project-{number}-{descriptive-name}` (lowercase with hyphens)
- **Notebooks:** `{descriptive_name}_notebook.ipynb` (underscores)
- Example: `project-1-induction-heads/induction_heads_notebook.ipynb`

### Documentation Standards
- Each project has its own README.md
- Root README.md links to all projects
- CONTRIBUTING.md specifies standards
- Emphasize "why" over "what" in documentation

### Git Practices
- Commit prefixes: `feat:`, `code:`, `docs:`, `refactor:`, `fix:`
- Meaningful commit messages
- Example: "code: refactored induction head detection code in notebook"

### Code Quality Standards
- Clear structure with markdown headers in notebooks
- Descriptive variable names
- Comments for non-obvious operations
- Proper visualization with labels and legends
- Keep notebook outputs reasonable (<5MB)

---

## Research Methodology

### Approach (Based on Neel Nanda's Framework)

**Stage 1: Ideation**
- Stay current with papers and LessWrong/Alignment Forum
- Build taste for tractable questions
- Prioritize based on learnability and impact

**Stage 2: Exploration**
- Rapid experimentation and data collection
- Build intuitions through hands-on work
- Don't worry about polish or certainty

**Stage 3: Understanding**
- Form and test hypotheses systematically
- Iterate on experiments
- Maintain skepticism about initial findings

**Stage 4: Distillation**
- Write up findings clearly
- Focus on clarity and accessibility
- Document limitations honestly

### Success Indicators
- Clear findings about when/why techniques work or fail
- Systematic comparisons (layers, positions, tasks)
- Honest assessment of limitations
- Practical recommendations

### Quality Checks
- Verify against published results when replicating
- Include sanity checks and validation steps
- Document any deviations from original papers
- Test edge cases and failure modes

---

## Computational Requirements

### Minimum
- 8GB RAM
- 2GB storage
- CPU execution (slow)

### Recommended
- 16GB+ RAM
- GPU with 8GB+ VRAM (CUDA 11.8 tested)
- 5GB storage

### Primary Execution Environment
- **Vast.ai** - For Qwen2.5-7B (needs 16GB+ VRAM, RTX 4090/A10/A100)
- **Google Colab** - For smaller experiments with GPT-2
- Local Jupyter - For code development and testing

---

## Recent Development Activity

### Current Session (Dec 5, 2025)
- Completed tutoring session on probe exercises (Day 3-6 notebooks)
- Created `day8-9_nnsight_setup.ipynb` for Qwen/nnsight transition
- Updated `mats_project_guide.md` with key learnings section
- Documented conceptual corrections (layer information, MLP vs attention roles)

### Recent Focus Areas
1. **Probe fundamentals:** Sentiment detection across layers, positions, components
2. **Conceptual understanding:** Corrected misconceptions about transformer internals
3. **Tool transition:** Moving from TransformerLens (GPT-2) to nnsight (Qwen)
4. **Documentation:** Capturing learnings for future reference

### Git Status (Current)
- Branch: main
- Active files in probe-based-faithfulness-detection/:
  - day3-4_first_probes.ipynb
  - day5-6_advanced_techniques.ipynb
  - day8-9_nnsight_setup.ipynb (NEW)
  - mats_project_guide.md (UPDATED)

---

## Common Tasks & Commands

### Setting Up Environment
```bash
# Create virtual environment
python -m venv mech_interp_env
source mech_interp_env/bin/activate  # or mech_interp_env\Scripts\activate on Windows

# Install dependencies (use requirements_old.txt as reference)
pip install torch transformers transformer-lens scikit-learn plotly pandas jupyter
```

### Running Notebooks
```bash
# Start Jupyter
jupyter notebook

# Or use VS Code with Jupyter extension
code .
```

### Working with GPT-2
```python
import torch
from transformer_lens import HookedTransformer

# Load model
model = HookedTransformer.from_pretrained("gpt2-small")

# Basic forward pass
tokens = model.to_tokens("Hello world")
logits, cache = model.run_with_cache(tokens)
```

### Probe Training Pattern (TransformerLens)
```python
from sklearn.linear_model import LogisticRegression

# Extract activations
activations = cache[f"blocks.{layer}.hook_resid_post"]

# Train probe
probe = LogisticRegression(max_iter=1000)
probe.fit(activations, labels)

# Evaluate
accuracy = probe.score(test_activations, test_labels)
```

### Working with nnsight/Qwen (Current Focus)
```python
from nnsight import LanguageModel

# Load model
model = LanguageModel("Qwen/Qwen2.5-7B-Instruct", device_map="auto", torch_dtype=torch.float16)

# Extract activations (different from TransformerLens!)
with model.trace(text):
    hidden = model.model.layers[layer].output[0].save()
activation = hidden.value[0, -1, :].cpu().numpy()

# Access different components
with model.trace(text):
    residual = model.model.layers[layer].output[0].save()      # Residual stream
    mlp_out = model.model.layers[layer].mlp.output.save()       # MLP output
    attn_out = model.model.layers[layer].self_attn.output[0].save()  # Attention output
```

---

## Key Concepts to Understand

### Mechanistic Interpretability Foundations
- **Features:** Individual units of meaning represented in the model
- **Circuits:** Computational subgraphs that implement specific algorithms
- **Superposition:** More features than dimensions (compressed representations)
- **Residual Stream:** Information flow through transformer layers

### Core Techniques
- **Activation Patching:** Replace activations to test causal impact
- **Ablation Studies:** Zero/mean out components to test necessity
- **Logit Attribution:** Decompose output predictions to individual components
- **Attention Pattern Analysis:** Understand what each head attends to

### Probes
- Linear classifiers trained on internal activations
- Test what information is linearly accessible at each layer
- **Faithfulness Question:** Do probes detect genuine internal representations or shortcuts?

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

3. **Current Work: Faithfulness Detection**
   - Read: `mech-interp-projects/probe-based-faithfulness-detection/README.md` (40KB guide)
   - Review: `faithfulness_detection_with_probes_notebook.ipynb`
   - Focus: Current week's objectives in the 4-week timeline

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

---

## Quick Reference: File Locations

### Notebooks
- Induction Heads: `mech-interp-replications/project-1-induction-heads/induction_heads_notebook.ipynb`
- IOI Circuit: `mech-interp-replications/project-2-Indirect-Object-Identification/IOI_notebook.ipynb`
- Faithfulness Detection: `mech-interp-projects/probe-based-faithfulness-detection/faithfulness_detection_with_probes_notebook.ipynb`

### Documentation
- Main README: `README.md`
- Contributing: `CONTRIBUTING.md`
- Setup: `SETUP.md`
- Faithfulness Guide: `mech-interp-projects/probe-based-faithfulness-detection/README.md`

### Context Materials
- All in: `Mechanistic_Interpretability_Context_Documents/Mech Interp Context Docs/`
- Most useful: `neel_glossary_60k.md`, `research+writing_advice_45k.md`

---

## Working with This Codebase

### Before Starting a Session
1. Check git status and recent commits
2. Review current project status (faithfulness detection timeline)
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

### Project State (Dec 5, 2025)
- **Active work:** Faithfulness detection - Day 8-9 (nnsight/Qwen setup)
- **Just completed:** Day 3-6 probe exercises with conceptual corrections
- **Key deliverable:** `mats_project_guide.md` contains all learnings
- **Next step:** Run day8-9 notebook on Vast.ai, then build CoT dataset

### TransformerLens vs nnsight Quick Reference

| Operation | TransformerLens | nnsight |
|-----------|-----------------|---------|
| Load model | `HookedTransformer.from_pretrained()` | `LanguageModel()` |
| Residual stream | `cache["resid_post", layer]` | `model.model.layers[layer].output[0]` |
| MLP output | `cache["blocks.{layer}.hook_mlp_out"]` | `model.model.layers[layer].mlp.output` |
| Attention output | `cache["attn_out", layer]` | `model.model.layers[layer].self_attn.output[0]` |
| Save values | Automatic | Explicit `.save()` required |

### Key Conceptual Points (from tutoring)
1. Later layers ≠ more information (different, not more)
2. Attention routes information; MLP transforms information
3. Last token position contaminated by output preparation
4. Residual stream accumulates (doesn't overwrite)
5. Individual attention heads can outperform full residual stream

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
- Include systematic comparisons (layers, positions, etc.)
- Test edge cases and failure modes
- Document limitations and assumptions
- Prioritize clarity in explanations

---

*This context file is designed to help Claude Code instances get up to speed quickly on this mechanistic interpretability research codebase. For detailed information on specific projects, always refer to the individual README.md files in each project directory.*

**For the faithfulness project specifically, see:** `mech-interp-projects/probe-based-faithfulness-detection/mats_project_guide.md`