#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <training_name>"
    exit 1
fi

# Activate 'python3_9' conda environment
source activate python3_9

# Post-processing files input arguments 
training_name=$1    # file input
ensemble=0          # fixed
iter=3220000

# Run the Python post-processing script with the current iteration and the training name
echo -e "\n--------------------------------------------------------------------------------"
python3 RL_post_process_uavg_urmsf.py $iter $ensemble $training_name
echo -e "\n--------------------------------------------------------------------------------"
python3 RL_post_process_anisotropy_tensor.py $iter $ensemble $training_name
echo -e "\n--------------------------------------------------------------------------------"
python3 RL_post_process_actions.py $ensemble $training_name
