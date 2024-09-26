#!/bin/bash
echo 'RHEA executable directory: ' $RHEA_EXE_DIR
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 0 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble0_step000000.out 2>&1 &
pid0=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 1 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble1_step000000.out 2>&1 &
pid1=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 2 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble2_step000000.out 2>&1 &
pid2=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 3 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble3_step000000.out 2>&1 &
pid3=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 4 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble4_step000000.out 2>&1 &
pid4=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 5 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble5_step000000.out 2>&1 &
pid5=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 6 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble6_step000000.out 2>&1 &
pid6=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 7 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble7_step000000.out 2>&1 &
pid7=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 8 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble8_step000000.out 2>&1 &
pid8=$!
mpirun -np 8 --hostfile $RHEA_EXE_DIR/my-hostfile --mca btl_base_warn_component_unused 0 $RHEA_EXE_DIR/RHEA.exe $RHEA_EXE_DIR/configuration_file.yaml 9 $RHEA_EXE_DIR/restart_data_file.h5 0.001 0.05105 0.0 False 0 > /home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow/train/train_2024-09-26--17-38-21--a8b5/mpi_output/mpi_output_ensemble9_step000000.out 2>&1 &
pid9=$!
wait $pid0 $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9
echo 'All MPI processes have completed.'
