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

# RHEA variables
export RHEA_PATH=/home/jofre/Nuria/flowsolverrhea
export RHEA_EXE_DIR=/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/ProjectRHEA/examples/RL_3D_turbulent_channel_flow

# add shared dynamic libraries
export SMARTREDIS_DIR=/apps/smartredis/0.4.0
export RAI_PATH=/apps/redisai/1.2.5/redisai.so
export SMARTSIM_REDISAI=1.2.5
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH

# Set RedisAi required environment variables
#export REDIS_PORT=$(( RANDOM % 1001 + 6000 )) # random int. btw 6000 and 7000
#export SSDB="tcp://127.0.0.1:$REDIS_PORT"
export SR_LOG_FILE="nohup.out"
export SR_LOG_LEVEL="INFO"
export SMARTSIM_LOG_LEVEL="DEBUG"

# Compile ProjectRHEA code
echo ">>> Compiling ProjectRHEA..."
CURRENT_DIR=$(pwd)
cd "$RHEA_EXE_DIR"
make clean
make
cd "$CURRENT_DIR"
echo ">>> ProjectRHEA compiled!"

# Activate conda environment for the current session
eval "$(conda shell.bash hook)"
conda activate smartrhea-env
echo ">>> Conda environment 'smartrhea-env' activated"

# Remove nohup.out, if necessary
rm -f nohup.out ensemble.out ensemble.err 

# Run training
echo ">>> Running training 'run.py'..."
python3 run.py

# Wait for the command to finish
wait $pid
echo ">>> Background process finished"