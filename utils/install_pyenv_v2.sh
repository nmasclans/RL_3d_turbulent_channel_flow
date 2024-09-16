#!/bin/bash

# Initialize conda for the current session
eval "$(conda shell.bash hook)"

# Create conda environment
conda deactivate
conda create -n smartrhea-env-v2 python=3.9 -y

# Activate and check conda environment
conda activate smartrhea-env-v2
which python
python --version

# Install dependencies
pip install smartsim==0.4.2 --no-cache-dir  # (requires Python version >=3.8, <3.11)
pip install smartredis --no-cache-dir
pip install numpy==1.21.0 --no-cache-dir
pip install contourpy==1.2.1 --no-cache-dir
pip install scipy==1.6.0 --no-cache-dir
pip install matplotlib==3.7.0 --no-cache-dir
pip install tensorflow==2.8.0 --no-cache-dir
pip install tf_agents==0.12.0 --no-cache-dir
pip install tensorflow-probability==0.16.0 --no-cache-dir

# Set environment variables
export CC=gcc CXX=g++ NO_CHECKS=1

# Build SmartSim 
smart build --device cpu --no_pt --no_tf

# Navigate to the directory where setup.py is located - directory containing setup.py
cd "$(dirname "$0")/.."

# Install package smartrhea in editable mode - uses setup.py internally
# 'editable' option enables modifications on smartrhea package to be immediately reflected in the environment 
pip install -e .