# add shared dynamic libraries
export SMARTREDIS_DIR=/apps/smartredis/0.4.0/
export RAI_PATH=/apps/redisai/1.2.5/redisai.so
export SMARTSIM_REDISAI=1.2.5
# export PATH=$SMARTREDIS_DIR/bin:$PATH -> has no 'bin' directory
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH

# Set SSDB variable for redisai and redis-server
export SSDB="tcp://127.0.0.1:6379"

#mpirun -np 8 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml
mpirun --mca btl_base_warn_component_unused 0 -np 2 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml
