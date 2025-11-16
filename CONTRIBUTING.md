# Contributing to This Repository

This document provides guidelines for adding new projects to maintain consistency and quality across the portfolio.

## Adding New Projects

### Project Structure

When adding a new mechanistic interpretability project, follow this structure:

```
mech-interp-replications/
└── project-N-descriptive-name/
    ├── README.md
    ├── notebook_name.ipynb
    └── [optional: additional files]
```

### Naming Conventions

- **Project folders:** `project-{number}-{descriptive-name}/`
  - Example: `project-3-superposition-toy-models/`
- **Notebooks:** `{descriptive_name}_notebook.ipynb`
  - Example: `superposition_notebook.ipynb`
- Use lowercase with hyphens for folders, underscores for notebooks

### Required Components

Each new project must include:

1. **Jupyter Notebook** (`.ipynb`)
   - Google Colab compatible
   - "Open in Colab" badge at the top
   - Clear section headers
   - Explanatory markdown cells
   - Practice exercises at the end

2. **README.md**
   - Overview of the research being replicated
   - Link to original paper
   - Explanation of key concepts
   - Notebook structure outline
   - Skills demonstrated
   - How to run instructions
   - References and resources

### Project README Template

Use this template for new project READMEs:

```markdown
# Project {N}: {Project Name}

## Overview
[Brief description of what this project covers]

**Original Paper:** [{Paper Title}]({URL}) ({Authors}, {Year})

## What is {Main Concept}?
[Explanation of the core concept being studied]

## Notebook Structure
[List of major sections/steps]

## Key Findings
[What you discovered/replicated]

## Technical Skills Demonstrated
[Bulleted list of techniques and tools used]

## Running the Notebook
[Instructions for Colab and local execution]

## Extensions & Future Work
[Possible directions to extend this work]

## References
[Citations and learning resources]
```

### Updating Main Documentation

When adding a new project, update:

1. **Root README.md**
   - Add to "Completed Projects" section
   - Update project count if mentioned
   - Remove from "Future Projects" if listed there

2. **mech-interp-replications/README.md**
   - Add new project section with overview
   - Update project list
   - Add to learning path if relevant

## Code Quality Standards

### Notebook Organization

1. **Clear Structure**
   - Use markdown headers (# for main sections, ## for subsections)
   - Group related code cells
   - Add explanatory text before complex operations

2. **Code Style**
   - Follow PEP 8 for Python code
   - Use descriptive variable names
   - Add comments for non-obvious operations
   - Keep cells focused on single tasks

3. **Outputs**
   - Include key visualizations in committed notebooks
   - Clear output from debugging/exploration cells
   - Keep notebook file size reasonable (<5MB)

4. **Documentation**
   - Explain "why" not just "what"
   - Link to relevant papers/resources
   - Define technical terms on first use

### Visualizations

- Use clear labels and titles
- Include legends when necessary
- Choose appropriate color schemes
- Make plots readable at notebook width

## Research Replication Standards

### Fidelity to Original Work

- Cite the original paper prominently
- Note any deviations from the original approach
- Explain why changes were made if applicable

### Verification

- Compare results to paper's findings
- Document discrepancies and investigate causes
- Include sanity checks and validation steps

### Extensions

- Clearly label which parts are original extensions
- Explain motivation for extensions
- Document experimental choices

## Adding Non-Mech-Interp Projects

If adding projects from other AI safety areas (e.g., AI control):

1. Create new top-level directory: `{topic}-projects/`
2. Follow similar structure to `mech-interp-replications/`
3. Update root README.md to reflect multiple research areas
4. Create topic-specific README.md

Example structure:
```
AI-safety-research/
├── mech-interp-replications/
├── ai-control-experiments/
│   ├── README.md
│   └── project-1-{name}/
└── README.md
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

- `feat: Add project on {topic}`
- `docs: Update README for {project}`
- `fix: Correct {issue} in {notebook}`
- `refactor: Improve code structure in {section}`

## Dependencies

- Keep `requirements.txt` updated with new libraries
- Specify version constraints for reproducibility
- Test that notebooks work with specified versions

## Questions?

This is a personal learning repository. These guidelines ensure:
- Consistency for portfolio presentation
- Easy navigation for potential employers
- Reproducibility for future reference
- Professional presentation of work

Feel free to adapt these guidelines as the repository evolves!
