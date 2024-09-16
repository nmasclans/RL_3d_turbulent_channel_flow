#!/bin/bash
echo 'RHEA executable directory: ' $RHEA_EXE_DIR
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 0 $RHEA_EXE_DIR/restart_data_file.h5 2e-05 0.0001 0.0 False > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-16--16-24-10--0130/mpi_output/mpi_output_ensemble0_step0.out 2>&1
