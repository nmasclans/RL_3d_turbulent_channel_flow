#!/bin/bash

REPO_EXAMPLE_DIR=/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples

for i in {10..12}; do
    for j in {3,5}; do

        echo "Running Case U${i}_${j}..."

        CASE_DIR_AUX="$REPO_EXAMPLE_DIR/RL_3D_turbulent_channel_flow_Retau100_U${i}_1_${j}max"

        cd "$CASE_DIR_AUX" || { echo "Failed to enter $CASE_DIR_AUX"; exit 1; }

        bash ./run_train_triton.sh > "nohup.out" 2>&1

        echo "Finished Case U${i}_1_${j}max."
    done
done

echo "Finish running executeU.sh!"