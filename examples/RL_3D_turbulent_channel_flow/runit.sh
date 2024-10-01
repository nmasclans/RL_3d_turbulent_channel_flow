#!/bin/bash
echo 'RHEA executable directory: ' $RHEA_EXE_DIR
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 0 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 128 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-10-01--10-16-01--200f/mpi_output/mpi_output_ensemble0_step000128.out 2>&1 &
pid0=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 1 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 128 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-10-01--10-16-01--200f/mpi_output/mpi_output_ensemble1_step000128.out 2>&1 &
pid1=$!
wait $pid0 $pid1
echo 'All MPI processes have completed.'
