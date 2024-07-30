#!/bin/bash

# Initialize conda for the current session
eval "$(conda shell.bash hook)"

# Create conda environment
conda deactivate
conda create -n smartrhea-env python=3.9 -y

# Activate and check conda environment
conda activate smartrhea-env
which python
python --version

# Install dependencies
pip install smartsim==0.4.2     # (requires Python version >=3.8, <3.11)
pip install smartredis
pip install scipy==1.6.0
pip install matplotlib==3.7.0
pip install numpy==1.20.0
pip install tensorflow==2.8.0
pip install tf_agents==0.10.0
pip install tensorflow-probability==0.14.1

# Set environment variables
export CC=gcc CXX=g++ NO_CHECKS=1

# Build SmartSim 
smart build --device cpu --no_pt --no_tf

# Navigate to the directory where this script is located - directory containing setup.py
cd "$(dirname "$0")/.."

# Install package smartrhea in editable mode - uses setup.py internally
# 'editable' option enables modifications on smartrhea package to be immediately reflected in the environment 
pip install -e .