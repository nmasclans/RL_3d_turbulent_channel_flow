#!/bin/bash

REPO_EXAMPLE_DIR=/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples

for i in {10..12}; do
    for j in {1..5}; do

        echo "Running Case U${i}_${j}..."

        CASE_DIR_AUX="$REPO_EXAMPLE_DIR/RL_3D_turbulent_channel_flow_Retau100_U${i}_${j}"

        cd "$CASE_DIR_AUX" || { echo "Failed to enter $CASE_DIR_AUX"; exit 1; }

        bash ./run_train_triton.sh > "train_case_U${i}_${j}.out" 2>&1

        echo "Finished Case U${i}_${j}."
    done
done

echo "Finish running executeU.sh!"