#!/bin/bash

REPO_EXAMPLE_DIR=/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples

for i in {16..20}; do
    echo "Running Case $i..."

    CASE_DIR_AUX="$REPO_EXAMPLE_DIR/RL_3D_turbulent_channel_flow_Retau100_sup_144_15max_TCP3_a3_s5_F$i"

    cd "$CASE_DIR_AUX" || { echo "Failed to enter $CASE_DIR_AUX"; exit 1; }

    bash ./run_train_triton.sh > "train_case_$i.out" 2>&1

    echo "Finished Case $i."
done

echo "Finish running execute.sh!"