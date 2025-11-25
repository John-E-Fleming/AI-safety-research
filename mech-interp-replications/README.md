# Mechanistic Interpretability Replications

This directory contains replications of foundational mechanistic interpretability research. Each project implements key techniques for understanding transformer models and demonstrates skills essential for AI safety research.

## What is Mechanistic Interpretability?

Mechanistic interpretability aims to reverse-engineer neural networks by:
- Identifying **interpretable features and circuits** that implement specific algorithms
- Understanding **how information flows** through the network
- Performing **causal interventions** to verify hypotheses about circuit behavior
- Building **human-understandable explanations** of model internals

This is crucial for AI safety because it helps us understand and control what models are doing internally, rather than treating them as black boxes.

## Projects

### [Project 1: Induction Heads](./project-1-induction-heads/)

**Research Papers:** 
- ["In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Anthropic, 2022)

**What You'll Learn:**
- How transformers perform in-context learning
- The two-layer circuit structure of induction heads
- Previous token heads (Layer 0) and induction heads (Layer 1)
- Ablation studies and activation patching techniques
- Visualizing attention patterns

**Key Results:**
- Successfully identified induction heads in GPT-2 small
- Demonstrated causal importance through ablation experiments
- Implemented activation patching to trace information flow

**Notebook:** `induction_heads_notebook.ipynb`

---

### [Project 2: Indirect Object Identification (IOI)](./project-2-Indirect-Object-Identification/)

**Research Paper:** ["Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) (Redwood Research, 2022)

**What You'll Learn:**
- Discovering circuits that solve specific tasks
- Logit attribution for identifying important attention heads
- Understanding name mover heads and their role in the IOI circuit
- Advanced ablation techniques for circuit validation
- Working with structured datasets for interpretability experiments

**Key Results:**
- Replicated the IOI circuit discovery process
- Identified name mover heads using logit attribution
- Validated circuit components through targeted ablations

**Notebook:** `IOI_notebook.ipynb`

---

## Common Techniques Implemented

### 1. Attention Pattern Analysis
Visualizing where attention heads attend to understand information routing.

### 2. Ablation Studies
- **Zero Ablation:** Setting activations to zero to test causal importance
- **Mean Ablation:** Replacing activations with average values
- Comparing baseline vs. ablated model performance

### 3. Activation Patching (Causal Tracing)
Patching activations from a "clean" run into a "corrupted" run to identify which components matter for specific behaviors.

### 4. Logit Attribution
Decomposing model outputs to identify which attention heads contribute most to correct predictions.

## Technical Setup

All notebooks use:
- **TransformerLens:** A library specifically designed for mechanistic interpretability
- **GPT-2 Small:** A well-studied model with interpretable circuits
- **Google Colab:** Free GPU access for running experiments

No local setup required - just click "Open in Colab" and run!

## Learning Path

Recommended order:
1. **Start with Induction Heads** - Introduces core concepts and simpler circuits
2. **Move to IOI** - More complex circuit with multiple component types
3. **Experiment with exercises** - Each notebook includes practice problems

## Skills Developed

- Reading and implementing ML research papers
- Using TransformerLens for model analysis
- Designing interpretability experiments
- Causal reasoning about neural network behavior
- Creating clear visualizations of results
- Scientific Python programming

## Next Steps

After completing these projects, you'll be prepared to:
- Tackle more advanced interpretability research
- Explore sparse autoencoders (SAEs) and feature visualization
- Investigate superposition and polysemanticity
- Conduct original mechanistic interpretability research

## Resources

### Essential Reading
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [Neel Nanda's Mech Interp Glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)

### Courses & Tutorials
- [ARENA 3.0](https://www.arena.education/) - Comprehensive mechanistic interpretability course
- [Neel Nanda's Video Tutorials](https://www.youtube.com/channel/UCBMJ0D-omcRay8dh4QT0doQ)

### Research Collections
- [Circuits Thread](https://distill.pub/2020/circuits/)
- [200 Concrete Open Problems in MI](https://alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability)

## Citation

If these replications were helpful, please cite the original papers:
```
@article{elhage2022induction,
  title={In-context Learning and Induction Heads},
  author={Elhage, Nelson and others},
  journal={Transformer Circuits Thread},
  year={2022}
}

@article{wang2022interpretability,
  title={Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small},
  author={Wang, Kevin and others},
  journal={arXiv preprint arXiv:2211.00593},
  year={2022}
}
```
