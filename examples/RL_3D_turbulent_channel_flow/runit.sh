#!/bin/bash
echo 'RHEA executable directory: ' $RHEA_EXE_DIR
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 0 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.005 0.0 False > mpi_output.out 2>&1
