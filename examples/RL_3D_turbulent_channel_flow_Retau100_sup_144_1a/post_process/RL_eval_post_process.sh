#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <evaluation_name>"
    exit 1
fi

# Activate 'python3_9' conda environment
source activate python3_9

# Post-processing files directory & parent directory (case directory)
REPO_DIR=/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow
POST_PROCESS_DIR=$REPO_DIR/post_process
CASE_DIR=$(dirname $(pwd))

# Post-processing files input arguments 
eval_name=$1        # file input
ensemble=0          # fixed
iter=3220000
Re_tau="100.0"
dt_phys="0.0001"
t_episode_eval="1"
run_mode="eval"

# Run the Python post-processing script with the current iteration and the evaluation name
echo -e "\n--------------------------------------------------------------------------------"
python3 $POST_PROCESS_DIR/RL_post_process_velocity_avg_rmsf.py $iter $ensemble $eval_name $Re_tau $dt_phys $t_episode_eval $CASE_DIR $run_mode
echo -e "\n--------------------------------------------------------------------------------"
python3 $POST_PROCESS_DIR/RL_post_process_anisotropy_tensor.py $iter $ensemble $eval_name $Re_tau $dt_phys $CASE_DIR $run_mode
echo -e "\n--------------------------------------------------------------------------------"
python3 $POST_PROCESS_DIR/RL_post_process_actions.py $ensemble $eval_name $CASE_DIR $run_mode
echo -e "\n--------------------------------------------------------------------------------"
python3 $POST_PROCESS_DIR/RL_post_process_powerspectra_selected_y_from_probelines.py $eval_name $Re_tau $dt_phys $t_episode_eval $CASE_DIR
