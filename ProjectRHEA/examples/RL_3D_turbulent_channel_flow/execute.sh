# add shared dynamic libraries
export SMARTREDIS_DIR=/apps/smartredis/0.4.0/
export RAI_PATH=/apps/redisai/1.2.5/redisai.so
export SMARTSIM_REDISAI=1.2.5
# export PATH=$SMARTREDIS_DIR/bin:$PATH -> has no 'bin' directory
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH

# Set RedisAi required environment variables
export REDIS_PORT=6380                          # TODO: rm if necessary, defined in rhea_env initializer
export SSDB="tcp://127.0.0.1:$REDIS_PORT"       # TODO: rm if necessary, defined in rhea_env initializer
export SR_LOG_FILE="nohup.out"
export SR_LOG_LEVEL="INFO"

# Start Redis server on port 6380 with the RedisAI module
redis-server --port $REDIS_PORT --loadmodule $RAI_PATH &
# Save the Redis server process ID (PID) to kill it later
REDIS_PID=$!
# Wait to ensure Redis server is fully started
sleep 2

# The Redis server will be stopped by the trap command on exit
function stopRedisServer {
    echo "Stopping Redis server with PID $REDIS_PID"
    kill $REDIS_PID
}
trap stopRedisServer EXIT

# Execute C++ application
#mpirun -np 8 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml
mpirun --mca btl_base_warn_component_unused 0 -np 2 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml