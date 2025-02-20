#!/bin/bash

# Check if two arguments (start and end) are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <iteration_start> <iteration_stop> <iteration_step> <rl_step>"
    exit 1
fi

# Read input arguments
ITERATION_START=$1
ITERATION_STOP=$2
ITERATION_STEP=$3
RL_STEP=$4

# Base part of the iteration string
rl_step_formatted=$(printf "%06d" $RL_STEP)
details="ensemble0_step${rl_step_formatted}"

# Loop through the range with step of 8
for ((iteration_num=ITERATION_START; iteration_num<=ITERATION_STOP; iteration_num+=ITERATION_STEP)); do

    # Construct the full iteration string
    ITERATION="${iteration_num}_${details}"

    # Echo the current iteration
    echo "_____________________________"
    echo "Processing file with iteration details: $ITERATION"

    # Execute the Python script with the iteration as an argument
    python3 post_processing_bulk_wall_values.py "$ITERATION"

done
