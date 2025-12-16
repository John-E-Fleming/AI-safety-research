#!/bin/bash
# setup.sh - Run this on a fresh Vast.ai instance

# Clone repo
git clone https://github.com/John-E-Fleming/AI-safety-research.git
cd AI-safety-research/mech-interp-projects/probe-based-faithfulness-detection

# Install dependencies
pip install -q nnsight transformers accelerate matplotlib pandas scikit-learn datasets

echo "Setup complete! Open day8-9_nnsight_setup.ipynb"