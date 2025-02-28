#!/bin/bash

# Function to handle cleanup on script termination
cleanup() {
    echo ">>> Terminating all child processes..."
    # Kill all child processes of the current shell
    pkill -P $$
    wait
    echo ">>> All child processes terminated."
}
# Trap SIGINT (Ctrl+C) and SIGTSTP (Ctrl+Z) to execute the cleanup function
trap cleanup SIGINT SIGTERM

# Environmental variables
REPO_DIR=/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow
export RHEA_PATH=/home/jofre/Nuria/flowsolverrhea
export RHEA_CASE_PATH=$REPO_DIR/ProjectRHEA/examples/RL_3D_turbulent_channel_flow_Retau100_sup_144_u_2max_a3
export TRAIN_RL_CASE_PATH=$REPO_DIR/examples/RL_3D_turbulent_channel_flow_Retau100_sup_144_u_2max_a3
export EVAL_RL_CASE_PATH=""
export SMARTRHEA_PATH=$REPO_DIR/smartrhea
export RUN_MODE=train 

# add shared dynamic libraries
export SMARTREDIS_PATH=/apps/smartredis/0.4.0
export RAI_PATH=/apps/redisai/1.2.5/redisai.so
export SMARTSIM_REDISAI=1.2.5
export LD_LIBRARY_PATH=$SMARTREDIS_PATH/lib:$LD_LIBRARY_PATH

# Set RedisAi required environment variables
#export REDIS_PORT=$(( RANDOM % 1001 + 6000 )) # random int. btw 6000 and 7000
#export SSDB="tcp://127.0.0.1:$REDIS_PORT"
export SR_LOG_FILE="nohup.out"
export SR_LOG_LEVEL="INFO"
export SMARTSIM_LOG_LEVEL="DEBUG"

# Remove .out and .err files, if necessary
rm -f ensemble.out ensemble.err
rm -fr __pycache__/
rm -f temporal_point_probe_*.csv

# Compile ProjectRHEA code
echo ">>> Compiling ProjectRHEA..."
cd "$RHEA_CASE_PATH"
make clean
make RL_CASE_PATH=$TRAIN_RL_CASE_PATH RHEA_PATH=$RHEA_PATH
echo ">>> ProjectRHEA compiled!"
cd "$TRAIN_RL_CASE_PATH"

# Activate conda environment for the current session
eval "$(conda shell.bash hook)"
conda activate smartrhea-env-v2
echo ">>> Conda environment 'smartrhea-env-v2' activated"

# Run training
echo ">>> Running training 'run.py'..."
python3 "$SMARTRHEA_PATH"/run.py

# Wait for the command to finish
wait $pid
echo ">>> Background process finished"
