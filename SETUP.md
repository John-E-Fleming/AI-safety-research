# Setup Guide

Quick start instructions for running the notebooks in this repository.

## Recommended: Google Colab (No Setup Required!)

The easiest way to use these notebooks is through Google Colab:

1. Navigate to any project folder
2. Open the `.ipynb` file on GitHub
3. Click the "Open in Colab" badge at the top
4. Run all cells - dependencies will install automatically!

**Advantages:**
- ✅ Free GPU access
- ✅ No local installation needed
- ✅ Pre-configured environment
- ✅ Easy sharing and collaboration

## Local Setup (Optional)

If you prefer to run notebooks locally:

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster computation

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-safety-research.git
   cd AI-safety-research
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv venv

   # Activate on Windows
   venv\Scripts\activate

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

5. **Navigate to a project** and open the `.ipynb` file

### GPU Support (Optional)

For GPU acceleration, install PyTorch with CUDA support:

```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific setup.

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'transformer_lens'`**
```bash
pip install transformer-lens
```

**Issue: Notebook kernel keeps dying**
- Your system may be running out of RAM
- Try running on Colab instead
- Reduce batch sizes in the code

**Issue: CUDA out of memory**
- Reduce batch sizes
- Use CPU instead: `model.to('cpu')`
- Run on Colab which provides free GPU

**Issue: Visualizations not displaying**
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Getting Help

1. Check the [TransformerLens documentation](https://transformerlensorg.github.io/TransformerLens/)
2. Review error messages carefully - they often point to the solution
3. Compare your environment to `requirements.txt`
4. Try running in Colab to isolate environment issues

## Environment Information

### Tested Configurations

These notebooks have been tested on:
- **Google Colab** (recommended)
- **Local:** Python 3.10, Ubuntu 22.04, CUDA 11.8
- **Local:** Python 3.11, Windows 11, CPU only

### Minimum Requirements

- **RAM:** 8GB (16GB recommended)
- **Storage:** 2GB for dependencies
- **GPU:** Optional but recommended for larger experiments

## Development Setup

If you want to modify or extend the notebooks:

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8  # for code quality
   ```

2. Enable Jupyter extensions:
   ```bash
   pip install jupyter_contrib_nbextensions
   jupyter contrib nbextension install --user
   ```

3. Recommended VS Code extensions:
   - Jupyter
   - Python
   - Pylance

## Quick Test

Verify your setup is working:

```python
import torch
import transformer_lens
import plotly

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"TransformerLens version: {transformer_lens.__version__}")
print("✅ Setup complete!")
```

## Next Steps

Once your environment is ready:

1. Start with [Project 1: Induction Heads](./mech-interp-replications/project-1-induction-heads/)
2. Read through the markdown cells to understand the concepts
3. Run code cells sequentially
4. Try the practice exercises at the end
5. Move on to [Project 2: IOI](./mech-interp-replications/project-2-Indirect-Object-Identification/)

## Updates

To update dependencies as the repository evolves:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Performance Tips

### For Faster Execution

1. **Use GPU:** Significantly speeds up attention computations
2. **Cache activations:** The notebooks already do this efficiently
3. **Reduce sequence length:** For testing, use shorter sequences
4. **Smaller batch sizes:** If memory constrained

### For Better Learning

1. **Run cells one at a time:** Understand each step before proceeding
2. **Modify parameters:** Experiment with different values
3. **Add visualizations:** Create additional plots to explore
4. **Try different models:** Test on GPT-2 medium or other architectures

## Additional Resources

- [TransformerLens Tutorial](https://transformerlensorg.github.io/TransformerLens/)
- [ARENA 3.0 Course](https://www.arena.education/)
- [Neel Nanda's YouTube Channel](https://www.youtube.com/channel/UCBMJ0D-omcRay8dh4QT0doQ)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

## License & Usage

These are educational replications for learning purposes. Feel free to use them for your own learning, but please cite the original research papers appropriately.
