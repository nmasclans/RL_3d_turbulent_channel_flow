#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <training_name>"
    exit 1
fi

# Activate 'python3_9' conda environment
source activate python3_9

# Loop iteration number from '2820500' to '2822000' in steps of 500
training_name=$1    # file input
ensemble=0          # fixed
for iter in {2821000..2824000..1000}
do
    # Run the Python post-processing script with the current iteration and the training name
    echo -e "\n--------------------------------------------------------------------------------"
    python3 RL_post_process_uavg_urmsf.py $iter $ensemble $training_name
done