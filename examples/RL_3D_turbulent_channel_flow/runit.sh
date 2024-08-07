#!/bin/bash
echo 'RHEA executable directory: ' $RHEA_EXE_DIR
mpirun -np 2 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 0 restart_data_file.h5 1000.0 1.0 0.0 False
