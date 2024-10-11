#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <training_name>"
    exit 1
fi

# Activate 'python3_9' conda environment
source activate python3_9

# Loop iteration number from '3191000' to '3200000' in steps of 1000
training_name=$1    # file input
ensemble=0          # fixed
for iter in {3191000..3200000..1000}
do
    # Run the Python post-processing script with the current iteration and the training name
    python3 RL_post_process_uavg_urmsf.py $iter $ensemble $training_name
done