# add shared libraries
export SMARTREDIS_DIR=/home/jofre/Nuria/repositories/SmartRHEA/utils/smartredis-0.4.0-f2003/install
export RAI_PATH=/home/jofre/Nuria/repositories/SmartRHEA/utils/redisai-1.2.5/install-cpu/redisai.so
export SMARTSIM_REDISAI=1.2.5
export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH

#mpirun -np 8 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml
mpirun --mca btl_base_warn_component_unused 0 -np 8 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml
