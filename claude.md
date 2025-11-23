# AI Safety Research - Mechanistic Interpretability Portfolio

**Context File for Claude Code Sessions**
*Last Updated: 2025-11-23*

---

## Quick Start for New Sessions

This is a mechanistic interpretability research portfolio focused on building expertise through circuit replications and novel probe-based research. The codebase contains:

1. **Completed replications** of foundational mech interp papers (induction heads, IOI circuit)
2. **Active research project** on faithfulness detection using probes
3. **Extensive reference materials** (~3.1MB of mech interp documentation)

**Primary Technologies:** TransformerLens, PyTorch, scikit-learn, Jupyter
**Development Environment:** Google Colab (GPU), local Jupyter notebooks
**Git Branch:** main

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
│       ├── faithfulness_detection_with_probes_notebook.ipynb (2.9MB)
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
**Status:** Active development (Week 1-2 preparation phase)
**Goal:** Execute a sophisticated 20-hour research project investigating when probes can detect unfaithful Chain-of-Thought (CoT) reasoning

### Research Question
"What fundamental properties determine when probes can detect unfaithful CoT, and when do they fail?"

### Project Timeline (4 Weeks)

**Week 1-2: Foundation Building (40-50 hours)**
- Environment setup (GPU via RunPod/Vast.ai/Colab)
- Transformer basics and model interaction
- Sentiment probing exercises
- Multi-layer comparison analysis
- Probe toolkit development

**Week 3: Final Preparation (15-20 hours)**
- CoT faithfulness dataset collection/creation
- Analysis tools (CoTProbeAnalyzer class)
- End-to-end pipeline testing
- Hour-by-hour execution planning

**Week 4: Execution (22 hours)**
- 20-hour intensive research project
- Systematic experiments on layers, positions, generalization
- Adversarial testing and failure modes
- Research paper write-up with executive summary

### Key Hypotheses
- **H1:** Faithfulness detection works better at later layers (closer to output)
- **H2:** Information is concentrated at conclusion tokens ("therefore", "so")
- **H3:** Probes generalize within task types but not across tasks
- **H4:** Probes are vulnerable to adversarial stylistic changes

### Code Infrastructure
The notebook provides:
- **ProbeToolkit class** - Systematic comparison methods across layers/positions
- **CoTProbeAnalyzer class** - Specialized CoT analysis tools
- Activation extraction utilities (residual stream, MLP, attention)
- Visualization methods (layer comparisons, position heatmaps)
- Full troubleshooting guide

### Important Files
- **Notebook:** `faithfulness_detection_with_probes_notebook.ipynb` (2.9MB)
- **Guide:** `README.md` (40KB) - Complete project walkthrough with hour-by-hour breakdowns
- **Dependencies:** TransformerLens, scikit-learn, nnsight (for reasoning models)

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

### Optional Advanced Tools
- **nnsight** - Accessing internals of reasoning models
- **Gemini API** - Reasoning model CoT examples
- **Qwen API** - Alternative reasoning models

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
- **Google Colab** with GPU (T4 or better)
- Alternative: RunPod, Vast.ai for longer experiments
- Local Jupyter with GPU for iterative development

---

## Recent Development Activity

### Last 5 Commits (Nov 23, 2025)
1. `code: refactored induction head detection code in the notebook` (main)
2. `code: finished refactor of the previous token head detector code`
3. `code: refactoring previous token head detection code`
4. `code: Updating code for calculating and detecting induction heads`
5. `refactor: Improved the code structure for previous token head detection`

**Focus:** Recent work has emphasized code quality and refactoring, particularly around induction head detection methodology.

### Git Status (Current)
- Branch: main
- Deleted: requirements.txt (replaced with requirements_old.txt)
- Untracked:
  - Mechanistic_Interpretability_Context_Documents/
  - mech-interp-projects/
  - requirements_old.txt

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

### Probe Training Pattern
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

### Project State
- **Active work:** Faithfulness detection (Weeks 1-2 preparation phase)
- **Completed:** Induction heads, IOI circuit
- **Recent focus:** Code refactoring and organization

### Preferred Workflow
1. Understand context from relevant READMEs
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