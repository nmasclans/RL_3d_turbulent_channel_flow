#!/bin/bash

# Check if two arguments (start and end) are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <iteration_num> <start_step> <end_step>"
    exit 1
fi

# Read input arguments
ITERATION_NUM=$1
START=$2
END=$3
STEP=8  # Fixed step size

# Base part of the iteration string
BASE="$1_ensemble0_step"

# Loop through the range with step of 8
for ((i=START; i<=END; i+=STEP)); do
    # Format the step part with leading zeros (6 digits)
    STEP_PART=$(printf "%06d" $i)

    # Construct the full iteration string
    ITERATION="${BASE}${STEP_PART}"

    # Echo the current iteration
    echo "_____________________________"
    echo "Processing file with iteration details: $ITERATION"

    # Execute the Python script with the iteration as an argument
    python3 post_processing_bulk_wall_values.py "$ITERATION"
done
