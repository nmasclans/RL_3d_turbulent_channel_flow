#!/bin/bash

# add shared dynamic libraries
export SMARTREDIS_DIR=/apps/smartredis/0.4.0/
export RAI_PATH=/apps/redisai/1.2.5/redisai.so
export SMARTSIM_REDISAI=1.2.5
export RHEA_EXE_DIR=/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/ProjectRHEA/examples/RL_3D_turbulent_channel_flow

# Set RedisAi required environment variables
#export REDIS_PORT=6380
#export SSDB="tcp://127.0.0.1:$REDIS_PORT"
export SR_LOG_FILE="nohup.out"
export SR_LOG_LEVEL="INFO"
export SMARTSIM_LOG_LEVEL="DEBUG"

# Activate conda environment for the current session
eval "$(conda shell.bash hook)"
conda activate smartrhea-env

# Run training
python3 run.py
