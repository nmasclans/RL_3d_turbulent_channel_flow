# add shared dynamic libraries
export SMARTREDIS_DIR=/apps/smartredis/0.4.0/
export RAI_PATH=/apps/redisai/1.2.5/redisai.so
export SMARTSIM_REDISAI=1.2.5
# export PATH=$SMARTREDIS_DIR/bin:$PATH -> has no 'bin' directory
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH

# Set SSDB variable for redisai
export SSDB="tcp://127.0.0.1:6379"
# Check if Redis server is running and accessible, should see 'PONG' output
redis-cli -u $SSDB ping
# TODO: INSTALL REDIS, ONLY REDISAI INSTALLED!
# It looks like you have RedisAI installed, but you don't seem to have Redis itself installed, which is likely why redis-cli isn't working. RedisAI depends on Redis being installed and running. Here’s how you can handle this:

# Option 1: Install Redis
# If you want to use Redis from the official package, you would generally install it using:

# bash
# Copia el codi
# sudo apt update
# sudo apt install redis-server

#mpirun -np 8 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml
mpirun --mca btl_base_warn_component_unused 0 -np 2 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml
