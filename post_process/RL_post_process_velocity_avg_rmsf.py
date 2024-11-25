#!/home/jofre/miniconda3/envs/smartrhea-env-v2/bin/python3

import sys
import os
import glob
import numpy as np
import h5py    
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

#np.set_printoptions(threshold=sys.maxsize)
#plt.rc( 'text', usetex = True )
#rc('font', family='sanserif')
#plt.rc( 'font', size = 20 )
#plt.rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}', r'\usepackage{amssymb}', r'\usepackage{color}' ]

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---

try :
    iteration  = sys.argv[1]
    ensemble   = sys.argv[2]
    train_name = sys.argv[3]
    Re_tau     = float(sys.argv[4])     # Friction Reynolds number [-]
    dt_phys    = float(sys.argv[5])
    case_dir   = sys.argv[6]
    print(f"\nScript parameters: \n- Iteration: {iteration} \n- Ensemble: {ensemble}\n- Train name: {train_name} \n- Re_tau: {Re_tau} \n- dt_phys: {dt_phys} \n- Case directory: {case_dir}")
except :
    raise ValueError("Missing call arguments, should be: <iteration> <ensemble> <train_name> <Re_tau> <dt_phys> <case_dir>")

# --- Case parameters ---
rho_0   = 1.0				# Reference density [kg/m3]
u_tau   = 1.0				# Friction velocity [m/s]
delta   = 1.0				# Channel half-height [m]
mu_ref  = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref  = mu_ref/rho_0			# Kinematic viscosity [m2/s]

# training post-processing directory
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# Reference & non-RL data directory
filePath = os.path.dirname(os.path.abspath(__file__))
compareDatasetDir = os.path.join(filePath, f"data_Retau{Re_tau:.0f}")
iteration_max_nonRL = 4000000
max_length_legend_RL = 10

# RL parameters
t_episode_train = 1.0
dt_phys = 1e-4
cfd_n_envs = 1
rl_n_envs  = 8
simulation_time_per_train_step   = t_episode_train * cfd_n_envs       # total cfd simulated time per training step (in parallel per each cfd_n_envs)
num_global_steps_per_train_step  = int(cfd_n_envs * rl_n_envs)        # num. global steps per training step
num_iterations_per_train_step    = int(np.round(simulation_time_per_train_step / dt_phys))
iteration_restart_data_file      = 3210000
iteration_end_train_step         = iteration_restart_data_file + num_iterations_per_train_step
assert iteration_restart_data_file + num_iterations_per_train_step == iteration_end_train_step
print("\nRL parameters: \n- Simulation time per train step:", simulation_time_per_train_step, 
      "\n- Num. global steps per train step:", num_global_steps_per_train_step,
      "\n- Num. iterations per train step:", num_iterations_per_train_step,
      "\n- Iteration restart data file (init train step):", iteration_restart_data_file,
      "\n- Iteration end train step:", iteration_end_train_step,
) 

# --- RL pseudo-environments / actuators boundaries ---

if np.isclose(Re_tau, 100, atol=1e-8):
    Re_tau_theoretical = 100.0
    y_actuators_boundaries = np.array([0.05, 0.25, 0.50, 0.75, 1.0])
elif np.isclose(Re_tau, 180, atol=1e-8):
    Re_tau_theoretical = 180.0
    y_actuators_boundaries = np.array([0.027777, 0.0962555, 0.3242615, 0.640158, 1.0])
else:
    raise ValueError(f"'actuators_boundaries' not implemented for Re_tau = {Re_tau}")
y_plus_actuators_boundaries = y_actuators_boundaries * Re_tau_theoretical   

# ----------- Build data h5 filenames ------------

# --- RL filenames ---
# Get 'file_details' & filename_RL
pattern = f"{case_dir}/rhea_exp/output_data/RL_3d_turbulent_channel_flow_{iteration}_ensemble{ensemble}_*.h5"
# Use glob to find all matching files
matching_files = sorted(glob.glob(pattern))
# List to store the extracted parts corresponding to "*"
filename_RL_list  = []
file_details_list = []
step_num_list     = []
# Check if files were found
if matching_files:
    print("\RL files:")
    for file in matching_files:
        # Store file
        filename_RL_list.append(file)
        # Extract the filename (without the directory)
        base_filename = os.path.basename(file)
        # Extract the part corresponding to "*"
        # Split by "_" and get the last part before ".h5"
        file_details = base_filename.split('_')[-1].replace('.h5', '')
        # Add the extracted part to the list
        file_details_list.append(file_details)
        # Step number
        step_num = int(file_details[4:])
        step_num_list.append(step_num)
        # Print the file and the extracted part
        print(f"Filename: {base_filename}, File details: {file_details}, Step number: {step_num}")
else:
    print(f"No files found matching the pattern: {pattern}")
global_step_num_list = step_num_list
N = len(filename_RL_list)

# --- non-RL filenames ---
train_step_list = [int(gs/num_global_steps_per_train_step) for gs in global_step_num_list]
iteration_nonRL_list = [ (s+1)*num_iterations_per_train_step + iteration_restart_data_file for s in train_step_list]
filename_nonRL_list  = [f"{compareDatasetDir}/3d_turbulent_channel_flow_{iter}.h5" for iter in iteration_nonRL_list] 
assert N == len(train_step_list)
print("\nnon-RL files:")
for i in range(N):
    print("Filename:", filename_nonRL_list[i], ", Iteration:", iteration_nonRL_list[i])

# --- non-RL converged reference filename ---
filename_ref = f"{compareDatasetDir}/3d_turbulent_channel_flow_reference.h5"

# --- non-RL restart data file
filename_rst = f"{compareDatasetDir}/3d_turbulent_channel_flow_{iteration_restart_data_file}.h5"

# Append restart data file to RL & non-RL files list
# > RL lists:
filename_RL_list.insert(0,filename_rst)
file_details_list.insert(0,'restart')
step_num_list.insert(0,'000000') 
global_step_num_list.insert(0,'000000') 
N = len(filename_RL_list)   # update N
# > non-RL lists:
filename_nonRL_list.insert(0,filename_rst)
train_step_list.insert(0,0)
iteration_nonRL_list.insert(0,iteration_restart_data_file)
assert N == len(filename_nonRL_list)

# --- Discard non-RL (and corresponding RL) snapshots if not available
filename_nonRL_is_available = [iter < iteration_max_nonRL for iter in iteration_nonRL_list]
n_nonRL_is_available = sum(filename_nonRL_is_available)
print("\nAvailable non-RL files:", n_nonRL_is_available, "Non-available non-RL files:", N - n_nonRL_is_available)
filename_RL_list_available = []
filename_nonRL_list_available = []
for i in range(N):
    if filename_nonRL_is_available[i]:
        filename_RL_list_available.append(filename_RL_list[i])
        filename_nonRL_list_available.append(filename_nonRL_list[i])
filename_RL_list    = filename_RL_list_available
filename_nonRL_list = filename_nonRL_list_available
N = len(filename_RL_list)   # update N
print(f"Datasets RL and non-RL have now {N} files each")


# ----------- Get RL and non-RL data ------------

# --- Check if RL & non-RL files exists ---
for filename_RL in filename_RL_list:
    if not os.path.isfile(filename_RL):
        print(f"Error: File '{filename_RL}' not found.")
        sys.exit(1)
for filename_nonRL in filename_nonRL_list:
    if not os.path.isfile(filename_nonRL):
        print(f"Error: File '{filename_nonRL}' not found.")
        sys.exit(1)
if not os.path.isfile(filename_ref):
    print(f"Error: File '{filename_ref}' not found.")
    sys.exit(1)

# --- Get data from 3d-snapshots h5 files ---

print("\nImporting data from files...")

print("\nImporting data from RL files:")
for i in range(N):
    filename_RL = filename_RL_list[i]
    with h5py.File( filename_RL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_RL_aux = data_file.attrs["AveragingTime"][0] - dt_phys
        y_data_RL_aux      = data_file['y'][1:-1,1:-1,1:-1]
        avg_u_data_RL_aux  = data_file['avg_u'][1:-1,1:-1,1:-1]
        avg_v_data_RL_aux  = data_file['avg_v'][1:-1,1:-1,1:-1]
        avg_w_data_RL_aux  = data_file['avg_w'][1:-1,1:-1,1:-1]
        rmsf_u_data_RL_aux = data_file['rmsf_u'][1:-1,1:-1,1:-1]
        rmsf_v_data_RL_aux = data_file['rmsf_v'][1:-1,1:-1,1:-1]
        rmsf_w_data_RL_aux = data_file['rmsf_w'][1:-1,1:-1,1:-1]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = avg_u_data_RL_aux[0,0,:].size
        num_points_y      = avg_u_data_RL_aux[0,:,0].size
        num_points_z      = avg_u_data_RL_aux[:,0,0].size
        num_points_xz     = num_points_x*num_points_z
        averaging_time_RL = np.zeros(N)
        y_data_RL         = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        avg_u_data_RL     = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        avg_v_data_RL     = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        avg_w_data_RL     = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        rmsf_u_data_RL    = np.zeros([N, num_points_z, num_points_y, num_points_x])            
        rmsf_v_data_RL    = np.zeros([N, num_points_z, num_points_y, num_points_x])            
        rmsf_w_data_RL    = np.zeros([N, num_points_z, num_points_y, num_points_x])  
    # Fill allocation arrays
    averaging_time_RL[i]    = averaging_time_RL_aux
    y_data_RL[i,:,:,:]      = y_data_RL_aux
    avg_u_data_RL[i,:,:,:]  = avg_u_data_RL_aux
    avg_v_data_RL[i,:,:,:]  = avg_v_data_RL_aux
    avg_w_data_RL[i,:,:,:]  = avg_w_data_RL_aux
    rmsf_u_data_RL[i,:,:,:] = rmsf_u_data_RL_aux
    rmsf_v_data_RL[i,:,:,:] = rmsf_v_data_RL_aux
    rmsf_w_data_RL[i,:,:,:] = rmsf_w_data_RL_aux
    # Logging
    print(f"RL non-converged data imported from file '{filename_RL}' - averaging time: {averaging_time_RL_aux:.6f}")

print("\nImporting data from non-RL files:")
for i in range(N):
    filename_nonRL = filename_nonRL_list[i]
    with h5py.File( filename_nonRL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_nonRL_aux = data_file.attrs["AveragingTime"][0]
        y_data_nonRL_aux      = data_file['y'][1:-1,1:-1,1:-1]
        avg_u_data_nonRL_aux  = data_file['avg_u'][1:-1,1:-1,1:-1]
        avg_v_data_nonRL_aux  = data_file['avg_v'][1:-1,1:-1,1:-1]
        avg_w_data_nonRL_aux  = data_file['avg_w'][1:-1,1:-1,1:-1]
        rmsf_u_data_nonRL_aux = data_file['rmsf_u'][1:-1,1:-1,1:-1]
        rmsf_v_data_nonRL_aux = data_file['rmsf_v'][1:-1,1:-1,1:-1]
        rmsf_w_data_nonRL_aux = data_file['rmsf_w'][1:-1,1:-1,1:-1]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = avg_u_data_nonRL_aux[0,0,:].size
        num_points_y      = avg_u_data_nonRL_aux[0,:,0].size
        num_points_z      = avg_u_data_nonRL_aux[:,0,0].size
        num_points_xz     = num_points_x*num_points_z
        averaging_time_nonRL = np.zeros(N)
        y_data_nonRL         = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        avg_u_data_nonRL     = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        avg_v_data_nonRL     = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        avg_w_data_nonRL     = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        rmsf_u_data_nonRL    = np.zeros([N, num_points_z, num_points_y, num_points_x])            
        rmsf_v_data_nonRL    = np.zeros([N, num_points_z, num_points_y, num_points_x])            
        rmsf_w_data_nonRL    = np.zeros([N, num_points_z, num_points_y, num_points_x])  
    # Fill allocation arrays
    averaging_time_nonRL[i]    = averaging_time_nonRL_aux
    y_data_nonRL[i,:,:,:]      = y_data_nonRL_aux
    avg_u_data_nonRL[i,:,:,:]  = avg_u_data_nonRL_aux
    avg_v_data_nonRL[i,:,:,:]  = avg_v_data_nonRL_aux
    avg_w_data_nonRL[i,:,:,:]  = avg_w_data_nonRL_aux
    rmsf_u_data_nonRL[i,:,:,:] = rmsf_u_data_nonRL_aux
    rmsf_v_data_nonRL[i,:,:,:] = rmsf_v_data_nonRL_aux
    rmsf_w_data_nonRL[i,:,:,:] = rmsf_w_data_nonRL_aux
    print(f"non-RL non-converged data imported from file '{filename_nonRL}' - averaging time: {averaging_time_nonRL_aux:.6f}")

print("\nImporting reference data (non-RL):")
with h5py.File( filename_ref, 'r' ) as data_file:
    averaging_time_ref = data_file.attrs["AveragingTime"][0]
    y_data_ref      = data_file['y'][1:-1,1:-1,1:-1]
    avg_u_data_ref  = data_file['avg_u'][1:-1,1:-1,1:-1]
    avg_v_data_ref  = data_file['avg_v'][1:-1,1:-1,1:-1]
    avg_w_data_ref  = data_file['avg_w'][1:-1,1:-1,1:-1]
    rmsf_u_data_ref = data_file['rmsf_u'][1:-1,1:-1,1:-1]
    rmsf_v_data_ref = data_file['rmsf_v'][1:-1,1:-1,1:-1]
    rmsf_w_data_ref = data_file['rmsf_w'][1:-1,1:-1,1:-1]
assert ((averaging_time_ref > averaging_time_RL).all() and (averaging_time_ref > averaging_time_nonRL).all()), f"Reference data averaging time {averaging_time_ref:.6f} must be greater than non-converged averaging time {averaging_time_nonConv:.6f}"
print(f"\nNon-RL converged reference data imported from file '{filename_ref}' - averaging time: {averaging_time_ref:.6f}")
print("Data imported successfully!")


# -------------- Averaging fields using XZ symmetries --------------

### Allocate averaged variables
y_plus_RL      = np.zeros( [N, int( 0.5*num_points_y )] ); y_plus_nonRL      = np.zeros( [N, int( 0.5*num_points_y )] );  y_plus_ref      = np.zeros( int( 0.5*num_points_y ) )
avg_u_plus_RL  = np.zeros( [N, int( 0.5*num_points_y )] ); avg_u_plus_nonRL  = np.zeros( [N, int( 0.5*num_points_y )] );  avg_u_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
avg_v_plus_RL  = np.zeros( [N, int( 0.5*num_points_y )] ); avg_v_plus_nonRL  = np.zeros( [N, int( 0.5*num_points_y )] );  avg_v_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
avg_w_plus_RL  = np.zeros( [N, int( 0.5*num_points_y )] ); avg_w_plus_nonRL  = np.zeros( [N, int( 0.5*num_points_y )] );  avg_w_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
rmsf_u_plus_RL = np.zeros( [N, int( 0.5*num_points_y )] ); rmsf_u_plus_nonRL = np.zeros( [N, int( 0.5*num_points_y )] );  rmsf_u_plus_ref = np.zeros( int( 0.5*num_points_y ) )
rmsf_v_plus_RL = np.zeros( [N, int( 0.5*num_points_y )] ); rmsf_v_plus_nonRL = np.zeros( [N, int( 0.5*num_points_y )] );  rmsf_v_plus_ref = np.zeros( int( 0.5*num_points_y ) )
rmsf_w_plus_RL = np.zeros( [N, int( 0.5*num_points_y )] ); rmsf_w_plus_nonRL = np.zeros( [N, int( 0.5*num_points_y )] );  rmsf_w_plus_ref = np.zeros( int( 0.5*num_points_y ) )

### Average variables in space
print("\nAveraging variables in space...")
for j in range( 0, num_points_y ):
    # log progress
    if j % (num_points_y//10 or 1) == 0:
        print(f"{j/num_points_y*100:.0f}%")
    # data averaging
    for i in range( 0, num_points_x ):
        for k in range( 0, num_points_z ):
            aux_j = j
            if( j > ( int( 0.5*num_points_y ) - 1 ) ):      # top-wall
                aux_j = num_points_y - j - 1
            # RL data:
            for n in range(N):
                if( j > ( int( 0.5*num_points_y ) - 1 ) ):  # top-wall
                    y_plus_RL[n,aux_j]  += ( 0.5/num_points_xz )*( 2.0 - y_data_RL[n,k,j,i] )*( u_tau/nu_ref )
                else:                                       # bottom-wall
                    y_plus_RL[n,aux_j]  += ( 0.5/num_points_xz )*y_data_RL[n,k,j,i]*( u_tau/nu_ref )
                avg_u_plus_RL[n,aux_j]  += ( 0.5/num_points_xz )*avg_u_data_RL[n,k,j,i]*( 1.0/u_tau )
                avg_v_plus_RL[n,aux_j]  += ( 0.5/num_points_xz )*avg_v_data_RL[n,k,j,i]*( 1.0/u_tau )
                avg_w_plus_RL[n,aux_j]  += ( 0.5/num_points_xz )*avg_w_data_RL[n,k,j,i]*( 1.0/u_tau )
                rmsf_u_plus_RL[n,aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_RL[n,k,j,i]*( 1.0/u_tau )
                rmsf_v_plus_RL[n,aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_RL[n,k,j,i]*( 1.0/u_tau )
                rmsf_w_plus_RL[n,aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_RL[n,k,j,i]*( 1.0/u_tau )
            # non-RL non-conv data:
            for n in range(N):
                if( j > ( int( 0.5*num_points_y ) - 1 ) ):  # top-wall
                    y_plus_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*(2.0 - y_data_nonRL[n,k,j,i])*( u_tau/nu_ref )
                else:                                       # bottom-wall
                    y_plus_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*y_data_nonRL[n,k,j,i]*( u_tau/nu_ref )
                avg_u_plus_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*avg_u_data_nonRL[n,k,j,i]*( 1.0/u_tau )
                avg_v_plus_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*avg_v_data_nonRL[n,k,j,i]*( 1.0/u_tau )
                avg_w_plus_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*avg_w_data_nonRL[n,k,j,i]*( 1.0/u_tau )
                rmsf_u_plus_nonRL[n,aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_nonRL[n,k,j,i]*( 1.0/u_tau )
                rmsf_v_plus_nonRL[n,aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_nonRL[n,k,j,i]*( 1.0/u_tau )
                rmsf_w_plus_nonRL[n,aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_nonRL[n,k,j,i]*( 1.0/u_tau )
            # non-RL converged reference data
            if( j > ( int( 0.5*num_points_y ) - 1 ) ):  # top-wall
                y_plus_ref[aux_j]  += ( 0.5/num_points_xz )*(2.0 - y_data_ref[k,j,i])*( u_tau/nu_ref )
            else:                                       # bottom-wall
                y_plus_ref[aux_j]  += ( 0.5/num_points_xz )*y_data_ref[k,j,i]*( u_tau/nu_ref )
            avg_u_plus_ref[aux_j]  += ( 0.5/num_points_xz )*avg_u_data_ref[k,j,i]*( 1.0/u_tau )
            avg_v_plus_ref[aux_j]  += ( 0.5/num_points_xz )*avg_v_data_ref[k,j,i]*( 1.0/u_tau )
            avg_w_plus_ref[aux_j]  += ( 0.5/num_points_xz )*avg_w_data_ref[k,j,i]*( 1.0/u_tau )
            rmsf_u_plus_ref[aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_ref[k,j,i]*( 1.0/u_tau )
            rmsf_v_plus_ref[aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_ref[k,j,i]*( 1.0/u_tau )
            rmsf_w_plus_ref[aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_ref[k,j,i]*( 1.0/u_tau )
print("Variables averaged successfully!")

### Calculate TKE averaged profile from rmsf_u,v,w averaged profiles 
TKE_RL    = 0.5 * ( rmsf_u_plus_RL**2    + rmsf_v_plus_RL**2    + rmsf_w_plus_RL**2    )
TKE_nonRL = 0.5 * ( rmsf_u_plus_nonRL**2 + rmsf_v_plus_nonRL**2 + rmsf_w_plus_nonRL**2 )
TKE_ref   = 0.5 * ( rmsf_u_plus_ref**2   + rmsf_v_plus_ref**2   + rmsf_w_plus_ref**2   )


# # -------------- Build plots avg-u and rmsf-u,v,w profiles --------------
# 
# print("\nBuilding plots...")
# 
# ### Plot u+ vs. y+
# xmin = 1.0; xmax = 2.0e2
# ymin = 0.0; ymax = 20.0
# # Clear plot
# plt.clf()
# # RL Actuators boundaries
# for i in range(len(y_plus_actuators_boundaries)):
#     plt.vlines(y_plus_actuators_boundaries[i], ymin, ymax, colors='gray', linestyle='--')
# # Plot data
# plt.plot( y_plus_ref, avg_u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
# for i in range(N):
#     if N < max_length_legend_RL:
#         plt.plot( y_plus_RL[i], avg_u_plus_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i]}, Avg. time {averaging_time_nonConv:.2f}s' )
#     else:
#         plt.plot( y_plus_RL[i], avg_u_plus_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
# plt.plot( y_plus_nonRL[0], avg_u_plus_nonRL[0], linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s' )
# # Configure plot
# plt.xlim( xmin, xmax )
# plt.xticks( np.arange( xmin, xmax + 0.1, 1.0 ) )
# plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
# plt.xscale( 'log' )
# plt.xlabel( 'y+' )
# plt.ylim( ymin, ymax )
# plt.yticks( np.arange( ymin, ymax + 0.1, 5.0 ) )
# plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
# #plt.yscale( 'log' )
# plt.ylabel( 'u+')
# plt.grid(which='both',  axis='x')
# plt.grid(which='major', axis='y')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tick_params( axis = 'both', pad = 7.5 )
# filename = f'{postDir}/u_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg'
# plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
# print("Done plot:", filename)
# # Clear plot
# plt.clf()
# 
# ### Plot u-rmsf 
# xmin = 1.0; xmax = 2.0e2
# ymin = 0.0; ymax = 3.0
# # RL Actuators boundaries
# for i in range(len(y_plus_actuators_boundaries)):
#     plt.vlines(y_plus_actuators_boundaries[i], ymin, ymax, colors='gray', linestyle='--')
# # Read & Plot data
# plt.plot( y_plus_ref, rmsf_u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label=f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
# for i in range(N):
#     if N < max_length_legend_RL:
#         plt.plot( y_plus_RL[i], rmsf_u_plus_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i]}, Avg. time {averaging_time_nonConv:.2f}s' )
#     else:
#         plt.plot( y_plus_RL[i], rmsf_u_plus_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
# plt.plot( y_plus_nonRL[0], rmsf_u_plus_nonRL[0], linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s' )
# # Configure plot
# plt.xlim( xmin, xmax )
# plt.xticks( np.arange( xmin, xmax+0.1, 1.0 ) )
# plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
# plt.xscale( 'log' )
# plt.xlabel( 'y+' )
# plt.ylim( ymin, ymax )
# plt.yticks( np.arange( ymin, ymax+0.1, 0.5 ) )
# plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
# #plt.yscale( 'log' )
# plt.ylabel( 'u_rms+' )
# plt.grid(which='both', axis='x')
# plt.grid(which='major', axis='y')
# plt.text( 1.05, 1.0, 'u_rms+' )
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tick_params( axis = 'both', pad = 7.5 )
# filename = f'{postDir}/u_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg'
# plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
# print("Done plot:", filename)
# # Clear plot
# plt.clf()
# 
# ### Plot v-rmsf
# xmin = 1.0; xmax = 2.0e2
# ymin = 0.0; ymax = 1.0
# # RL Actuators boundaries
# for i in range(len(y_plus_actuators_boundaries)):
#     plt.vlines(y_plus_actuators_boundaries[i], ymin, ymax, colors='gray', linestyle='--')
# # Read & Plot data
# plt.plot( y_plus_ref, rmsf_v_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
# for i in range(N):
#     if N < max_length_legend_RL:
#         plt.plot( y_plus_RL[i], rmsf_v_plus_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i]}, Avg. time {averaging_time_nonConv:.2f}s' )
#     else:
#         plt.plot( y_plus_RL[i], rmsf_v_plus_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
# plt.plot( y_plus_nonRL[0], rmsf_v_plus_nonRL[0], linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
# # Configure plot
# plt.xlim( xmin, xmax )
# plt.xticks( np.arange( xmin, xmax+0.1, 1.0 ) )
# plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
# plt.xscale( 'log' )
# plt.xlabel( 'y+' )
# plt.ylim( ymin, ymax )
# plt.yticks( np.arange( ymin, ymax+0.1, 0.5 ) )
# plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
# #plt.yscale( 'log' )
# plt.ylabel( 'v_rms+' )
# plt.grid(which='both', axis='x')
# plt.grid(which='major', axis='y')
# plt.text( 17.5, 0.2, 'v_rms+' )
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tick_params( axis = 'both', pad = 7.5 )
# filename = f'{postDir}/v_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg'
# plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
# print("Done plot:", filename)
# # Clear plot
# plt.clf()
# 
# ### Plot w-rmsf
# xmin = 1.0; xmax = 2.0e2
# ymin = 0.0; ymax = 1.0
# # RL Actuators boundaries
# for i in range(len(y_plus_actuators_boundaries)):
#     plt.vlines(y_plus_actuators_boundaries[i], ymin, ymax, colors='gray', linestyle='--')
# # Read & Plot data
# plt.plot( y_plus_ref, rmsf_w_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
# for i in range(N):
#     if N < max_length_legend_RL:
#         plt.plot( y_plus_RL[i], rmsf_w_plus_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i]}, Avg. time {averaging_time_nonConv:.2f}s' )
#     else:
#         plt.plot( y_plus_RL[i], rmsf_w_plus_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
# plt.plot( y_plus_nonRL[0], rmsf_w_plus_nonRL[0], linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
# # Configure plot
# plt.xlim( xmin, xmax )
# plt.xticks( np.arange( xmin, xmax+0.1, 1.0 ) )
# plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
# plt.xscale( 'log' )
# plt.xlabel( 'y+' )
# plt.ylim( ymin, ymax )
# plt.yticks( np.arange( ymin, ymax+0.1, 0.5 ) )
# plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
# #plt.yscale( 'log' )
# plt.ylabel( 'w_rms+' )
# plt.grid(which='both', axis='x')
# plt.grid(which='major', axis='y')
# plt.text( 17.5, 0.2, 'w_rms+' )
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tick_params( axis = 'both', pad = 7.5 )
# filename = f'{postDir}/w_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg'
# plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
# print("Done plot:", filename)
# # Clear plot
# plt.clf()
# 
# ### Plot TKE
# xmin = 1.0; xmax = 2.0e2
# ymin = 0.0; ymax = 4.0
# # RL Actuators boundaries
# for i in range(len(y_plus_actuators_boundaries)):
#     plt.vlines(y_plus_actuators_boundaries[i], ymin, ymax, colors='gray', linestyle='--')
# # Read & Plot data
# plt.plot( y_plus_ref, TKE_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
# for i in range(N):
#     if N < max_length_legend_RL:
#         plt.plot( y_plus_RL[i], TKE_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i]}, Avg. time {averaging_time_nonConv:.2f}s' )
#     else:
#         plt.plot( y_plus_RL[i], TKE_RL[i], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
# plt.plot( y_plus_nonRL[0], TKE_nonRL[0], linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
# # Configure plot
# plt.xlim( xmin, xmax )
# plt.xticks( np.arange( xmin, xmax+0.1, 1.0 ) )
# plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
# plt.xscale( 'log' )
# plt.xlabel( 'y+' )
# plt.ylim( ymin, ymax )
# plt.yticks( np.arange( ymin, ymax+0.1, 1.0 ) )
# plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
# #plt.yscale( 'log' )
# plt.ylabel( 'TKE+' )
# plt.grid(which='both', axis='x')
# plt.grid(which='major', axis='y')
# plt.text( 17.5, 0.2, 'TKE+' )
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tick_params( axis = 'both', pad = 7.5 )
# filename = f'{postDir}/tke_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg'
# plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
# print("Done plot:", filename)
# # Clear plot
# plt.clf()


# ----------------- Plot Animation Frames of um, urmsf, Rij dof for increasing RL global step (specific iteration & ensemble) -----------------

print("Building gif frames for u-avg and u,v,w-rmsf profiles...")
from ChannelVisualizer import ChannelVisualizer
visualizer   = ChannelVisualizer(postDir)
frames_avg_u = []; frames_avg_v = []; frames_avg_w = []; frames_rmsf_u = []; frames_rmsf_v = []; frames_rmsf_w = []
# avg velocities limits
avg_v_abs_max = np.max([np.max(np.abs(avg_v_plus_RL)), np.max(np.abs(avg_v_plus_nonRL)), np.max(np.abs(avg_v_plus_ref))])
avg_w_abs_max = np.max([np.max(np.abs(avg_w_plus_RL)), np.max(np.abs(avg_w_plus_nonRL)), np.max(np.abs(avg_w_plus_ref))])
avg_u_min     = 0.0
avg_v_min     = - avg_v_abs_max
avg_w_min     = - avg_w_abs_max
avg_u_max     = int(np.max([np.max(avg_u_plus_RL), np.max(avg_u_plus_nonRL), np.max(avg_u_plus_ref)]))+1
avg_v_max     = avg_v_abs_max
avg_w_max     = avg_w_abs_max
ylim_avg_u    = [avg_u_min, avg_u_max]
ylim_avg_v    = [avg_v_min, avg_v_max]
ylim_avg_w    = [avg_w_min, avg_w_max]
# rmsf velocities limits
rmsf_u_min   = 0.0
rmsf_v_min   = 0.0
rmsf_w_min   = 0.0
rmsf_u_max   = int(np.max([np.max(rmsf_u_plus_RL), np.max(rmsf_u_plus_nonRL), np.max(rmsf_u_plus_ref)]))+1
rmsf_v_max   = int(np.max([np.max(rmsf_v_plus_RL), np.max(rmsf_v_plus_nonRL), np.max(rmsf_v_plus_ref)]))+1
rmsf_w_max   = int(np.max([np.max(rmsf_w_plus_RL), np.max(rmsf_w_plus_nonRL), np.max(rmsf_w_plus_ref)]))+1
ylim_rmsf_u  = [rmsf_u_min, rmsf_u_max]
ylim_rmsf_v  = [rmsf_v_min, rmsf_v_max]
ylim_rmsf_w  = [rmsf_w_min, rmsf_w_max]
print("Gifs y-limits:", ylim_avg_u, ylim_avg_v, ylim_avg_w, ylim_rmsf_u, ylim_rmsf_v, ylim_rmsf_w)
for i in range(N):
    # log progress
    if i % (N//10 or 1) == 0:
        print(f"{i/N*100:.0f}%")
    # Build frames
    frames_avg_u  = visualizer.build_vel_avg_frame( frames_avg_u,  y_plus_RL[i], y_plus_nonRL[i], y_plus_ref, avg_u_plus_RL[i],  avg_u_plus_nonRL[i],  avg_u_plus_ref,  averaging_time_RL[i], averaging_time_nonRL[i], train_step_list[i], vel_name='u', ylim=ylim_avg_u,  x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_avg_v  = visualizer.build_vel_avg_frame( frames_avg_v,  y_plus_RL[i], y_plus_nonRL[i], y_plus_ref, avg_v_plus_RL[i],  avg_v_plus_nonRL[i],  avg_v_plus_ref,  averaging_time_RL[i], averaging_time_nonRL[i], train_step_list[i], vel_name='v', ylim=ylim_avg_v,  x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_avg_w  = visualizer.build_vel_avg_frame( frames_avg_w,  y_plus_RL[i], y_plus_nonRL[i], y_plus_ref, avg_w_plus_RL[i],  avg_w_plus_nonRL[i],  avg_w_plus_ref,  averaging_time_RL[i], averaging_time_nonRL[i], train_step_list[i], vel_name='w', ylim=ylim_avg_w,  x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_rmsf_u = visualizer.build_vel_rmsf_frame(frames_rmsf_u, y_plus_RL[i], y_plus_nonRL[i], y_plus_ref, rmsf_u_plus_RL[i], rmsf_u_plus_nonRL[i], rmsf_u_plus_ref, averaging_time_RL[i], averaging_time_nonRL[i], train_step_list[i], vel_name='u', ylim=ylim_rmsf_u, x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_rmsf_v = visualizer.build_vel_rmsf_frame(frames_rmsf_v, y_plus_RL[i], y_plus_nonRL[i], y_plus_ref, rmsf_v_plus_RL[i], rmsf_v_plus_nonRL[i], rmsf_v_plus_ref, averaging_time_RL[i], averaging_time_nonRL[i], train_step_list[i], vel_name='v', ylim=ylim_rmsf_v, x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_rmsf_w = visualizer.build_vel_rmsf_frame(frames_rmsf_w, y_plus_RL[i], y_plus_nonRL[i], y_plus_ref, rmsf_w_plus_RL[i], rmsf_w_plus_nonRL[i], rmsf_w_plus_ref, averaging_time_RL[i], averaging_time_nonRL[i], train_step_list[i], vel_name='w', ylim=ylim_rmsf_w, x_actuator_boundaries=y_plus_actuators_boundaries)

print("Building gifs from frames...")
frames_dict = {'avg_u':frames_avg_u, 'avg_v':frames_avg_v, 'avg_w':frames_avg_w, 'rmsf_u': frames_rmsf_u, 'rmsf_v':frames_rmsf_v, 'rmsf_w':frames_rmsf_w}
visualizer.build_main_gifs_from_frames(frames_dict)
print("Gifs plotted successfully!")



############################# RL & non-RL Errors w.r.t. Reference #############################

if (np.allclose(y_plus_RL, y_plus_ref) & np.allclose(y_plus_nonRL, y_plus_ref)):
    print("\nNo need for interpolating data, as y_plus coordinates are the same for all data")
else:
    if not(np.allclose(y_plus_RL, y_plus_ref)):
        print("\nRL y+:", y_plus_RL)
        print("\nRef y+:", y_plus_ref)
    if not(np.allclose(y_plus_nonRL, y_plus_ref)):
        print("\nnon-RL y+:", y_plus_nonRL)
        print("\nRef y+:", y_plus_ref)
    raise Exception("y-plus coordinates should be equal for all RL & non-RL & reference data, as no interpolation will be done.")
ny = y_plus_ref.size
y_plus = np.concatenate([[0.0],y_plus_ref,[delta]])

# --- Calculate errors using spatial interpolation ---

print("\nCalculating L1, L2, Linf errors...")
# Absolute error 
abs_error_avg_u_plus_RL  = np.zeros( [N, y_plus_ref.size] ); abs_error_avg_u_plus_nonRL  = np.zeros( [N, y_plus_ref.size] )
abs_error_avg_v_plus_RL  = np.zeros( [N, y_plus_ref.size] ); abs_error_avg_v_plus_nonRL  = np.zeros( [N, y_plus_ref.size] )
abs_error_avg_w_plus_RL  = np.zeros( [N, y_plus_ref.size] ); abs_error_avg_w_plus_nonRL  = np.zeros( [N, y_plus_ref.size] )
abs_error_rmsf_u_plus_RL = np.zeros( [N, y_plus_ref.size] ); abs_error_rmsf_u_plus_nonRL = np.zeros( [N, y_plus_ref.size] )
abs_error_rmsf_v_plus_RL = np.zeros( [N, y_plus_ref.size] ); abs_error_rmsf_v_plus_nonRL = np.zeros( [N, y_plus_ref.size] )
abs_error_rmsf_w_plus_RL = np.zeros( [N, y_plus_ref.size] ); abs_error_rmsf_w_plus_nonRL = np.zeros( [N, y_plus_ref.size] )
for i in range(N):
    abs_error_avg_u_plus_RL[i,:]     = np.abs( avg_u_plus_RL[i,:]    - avg_u_plus_ref )
    abs_error_avg_v_plus_RL[i,:]     = np.abs( avg_v_plus_RL[i,:]    - avg_v_plus_ref )
    abs_error_avg_w_plus_RL[i,:]     = np.abs( avg_w_plus_RL[i,:]    - avg_w_plus_ref )
    abs_error_rmsf_u_plus_RL[i,:]    = np.abs( rmsf_u_plus_RL[i,:]   - rmsf_u_plus_ref )
    abs_error_rmsf_v_plus_RL[i,:]    = np.abs( rmsf_v_plus_RL[i,:]   - rmsf_v_plus_ref )
    abs_error_rmsf_w_plus_RL[i,:]    = np.abs( rmsf_w_plus_RL[i,:]   - rmsf_w_plus_ref )
    abs_error_avg_u_plus_nonRL[i,:]  = np.abs( avg_u_plus_nonRL[i,:]  - avg_u_plus_ref )
    abs_error_avg_v_plus_nonRL[i,:]  = np.abs( avg_v_plus_nonRL[i,:]  - avg_v_plus_ref )
    abs_error_avg_w_plus_nonRL[i,:]  = np.abs( avg_w_plus_nonRL[i,:]  - avg_w_plus_ref )
    abs_error_rmsf_u_plus_nonRL[i,:] = np.abs( rmsf_u_plus_nonRL[i,:] - rmsf_u_plus_ref )
    abs_error_rmsf_v_plus_nonRL[i,:] = np.abs( rmsf_v_plus_nonRL[i,:] - rmsf_v_plus_ref )
    abs_error_rmsf_w_plus_nonRL[i,:] = np.abs( rmsf_w_plus_nonRL[i,:] - rmsf_w_plus_ref )

# L1 Error
L1_error_avg_u_plus_RL  = np.zeros(N); L1_error_avg_u_plus_nonRL  = np.zeros(N)
L1_error_avg_v_plus_RL  = np.zeros(N); L1_error_avg_v_plus_nonRL  = np.zeros(N)
L1_error_avg_w_plus_RL  = np.zeros(N); L1_error_avg_w_plus_nonRL  = np.zeros(N)
L1_error_rmsf_u_plus_RL = np.zeros(N); L1_error_rmsf_u_plus_nonRL = np.zeros(N)
L1_error_rmsf_v_plus_RL = np.zeros(N); L1_error_rmsf_v_plus_nonRL = np.zeros(N)
L1_error_rmsf_w_plus_RL = np.zeros(N); L1_error_rmsf_w_plus_nonRL = np.zeros(N)
ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i in range(N):
        L1_error_avg_u_plus_RL[i]     += abs_error_avg_u_plus_RL[i,j]     * dy
        L1_error_avg_v_plus_RL[i]     += abs_error_avg_v_plus_RL[i,j]     * dy
        L1_error_avg_w_plus_RL[i]     += abs_error_avg_w_plus_RL[i,j]     * dy
        L1_error_rmsf_u_plus_RL[i]    += abs_error_rmsf_u_plus_RL[i,j]    * dy
        L1_error_rmsf_v_plus_RL[i]    += abs_error_rmsf_v_plus_RL[i,j]    * dy
        L1_error_rmsf_w_plus_RL[i]    += abs_error_rmsf_w_plus_RL[i,j]    * dy
        L1_error_avg_u_plus_nonRL[i]  += abs_error_avg_u_plus_nonRL[i,j]  * dy
        L1_error_avg_v_plus_nonRL[i]  += abs_error_avg_v_plus_nonRL[i,j]  * dy
        L1_error_avg_w_plus_nonRL[i]  += abs_error_avg_w_plus_nonRL[i,j]  * dy
        L1_error_rmsf_u_plus_nonRL[i] += abs_error_rmsf_u_plus_nonRL[i,j] * dy
        L1_error_rmsf_v_plus_nonRL[i] += abs_error_rmsf_v_plus_nonRL[i,j] * dy
        L1_error_rmsf_w_plus_nonRL[i] += abs_error_rmsf_w_plus_nonRL[i,j] * dy
L1_error_avg_u_plus_RL     /= ylength     
L1_error_avg_v_plus_RL     /= ylength     
L1_error_avg_w_plus_RL     /= ylength     
L1_error_rmsf_u_plus_RL    /= ylength     
L1_error_rmsf_v_plus_RL    /= ylength     
L1_error_rmsf_w_plus_RL    /= ylength     
L1_error_avg_u_plus_nonRL  /= ylength         
L1_error_avg_v_plus_nonRL  /= ylength         
L1_error_avg_w_plus_nonRL  /= ylength         
L1_error_rmsf_u_plus_nonRL /= ylength         
L1_error_rmsf_v_plus_nonRL /= ylength         
L1_error_rmsf_w_plus_nonRL /= ylength         

# L2 Error (RMS Error)
L2_error_avg_u_plus_RL  = np.zeros(N); L2_error_avg_u_plus_nonRL  = np.zeros(N)
L2_error_avg_v_plus_RL  = np.zeros(N); L2_error_avg_v_plus_nonRL  = np.zeros(N)
L2_error_avg_w_plus_RL  = np.zeros(N); L2_error_avg_w_plus_nonRL  = np.zeros(N)
L2_error_rmsf_u_plus_RL = np.zeros(N); L2_error_rmsf_u_plus_nonRL = np.zeros(N)
L2_error_rmsf_v_plus_RL = np.zeros(N); L2_error_rmsf_v_plus_nonRL = np.zeros(N)
L2_error_rmsf_w_plus_RL = np.zeros(N); L2_error_rmsf_w_plus_nonRL = np.zeros(N)
ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i in range(N):
        L2_error_avg_u_plus_RL[i]     += ( ( avg_u_plus_RL[i,j]     - avg_u_plus_ref[j] )**2 )  * dy
        L2_error_avg_v_plus_RL[i]     += ( ( avg_v_plus_RL[i,j]     - avg_v_plus_ref[j] )**2 )  * dy
        L2_error_avg_w_plus_RL[i]     += ( ( avg_w_plus_RL[i,j]     - avg_w_plus_ref[j] )**2 )  * dy
        L2_error_rmsf_u_plus_RL[i]    += ( ( rmsf_u_plus_RL[i,j]    - rmsf_u_plus_ref[j] )**2 ) * dy 
        L2_error_rmsf_v_plus_RL[i]    += ( ( rmsf_v_plus_RL[i,j]    - rmsf_v_plus_ref[j] )**2 ) * dy 
        L2_error_rmsf_w_plus_RL[i]    += ( ( rmsf_w_plus_RL[i,j]    - rmsf_w_plus_ref[j] )**2 ) * dy 
        L2_error_avg_u_plus_nonRL[i]  += ( ( avg_u_plus_nonRL[i,j]  - avg_u_plus_ref[j])**2 )   * dy
        L2_error_avg_v_plus_nonRL[i]  += ( ( avg_v_plus_nonRL[i,j]  - avg_v_plus_ref[j])**2 )   * dy
        L2_error_avg_w_plus_nonRL[i]  += ( ( avg_w_plus_nonRL[i,j]  - avg_w_plus_ref[j])**2 )   * dy
        L2_error_rmsf_u_plus_nonRL[i] += ( ( rmsf_u_plus_nonRL[i,j] - rmsf_u_plus_ref[j])**2 )  * dy
        L2_error_rmsf_v_plus_nonRL[i] += ( ( rmsf_v_plus_nonRL[i,j] - rmsf_v_plus_ref[j])**2 )  * dy
        L2_error_rmsf_w_plus_nonRL[i] += ( ( rmsf_w_plus_nonRL[i,j] - rmsf_w_plus_ref[j])**2 )  * dy
L2_error_avg_u_plus_RL     = np.sqrt( L2_error_avg_u_plus_RL / ylength )      
L2_error_avg_v_plus_RL     = np.sqrt( L2_error_avg_v_plus_RL / ylength )      
L2_error_avg_w_plus_RL     = np.sqrt( L2_error_avg_w_plus_RL / ylength )      
L2_error_rmsf_u_plus_RL    = np.sqrt( L2_error_rmsf_u_plus_RL / ylength )      
L2_error_rmsf_v_plus_RL    = np.sqrt( L2_error_rmsf_v_plus_RL / ylength )      
L2_error_rmsf_w_plus_RL    = np.sqrt( L2_error_rmsf_w_plus_RL / ylength )      
L2_error_avg_u_plus_nonRL  = np.sqrt( L2_error_avg_u_plus_nonRL / ylength )          
L2_error_avg_v_plus_nonRL  = np.sqrt( L2_error_avg_v_plus_nonRL / ylength )          
L2_error_avg_w_plus_nonRL  = np.sqrt( L2_error_avg_w_plus_nonRL / ylength )          
L2_error_rmsf_u_plus_nonRL = np.sqrt( L2_error_rmsf_u_plus_nonRL / ylength )          
L2_error_rmsf_v_plus_nonRL = np.sqrt( L2_error_rmsf_v_plus_nonRL / ylength )          
L2_error_rmsf_w_plus_nonRL = np.sqrt( L2_error_rmsf_w_plus_nonRL / ylength )          

# Linf Error
Linf_error_avg_u_plus_RL  = np.zeros(N); Linf_error_avg_u_plus_nonRL  = np.zeros(N)
Linf_error_avg_v_plus_RL  = np.zeros(N); Linf_error_avg_v_plus_nonRL  = np.zeros(N)
Linf_error_avg_w_plus_RL  = np.zeros(N); Linf_error_avg_w_plus_nonRL  = np.zeros(N)
Linf_error_rmsf_u_plus_RL = np.zeros(N); Linf_error_rmsf_u_plus_nonRL = np.zeros(N)
Linf_error_rmsf_v_plus_RL = np.zeros(N); Linf_error_rmsf_v_plus_nonRL = np.zeros(N)
Linf_error_rmsf_w_plus_RL = np.zeros(N); Linf_error_rmsf_w_plus_nonRL = np.zeros(N)
for i in range(N):
    Linf_error_avg_u_plus_RL[i]     = np.max(abs_error_avg_u_plus_RL[i,:])
    Linf_error_avg_v_plus_RL[i]     = np.max(abs_error_avg_v_plus_RL[i,:])
    Linf_error_avg_w_plus_RL[i]     = np.max(abs_error_avg_w_plus_RL[i,:])
    Linf_error_rmsf_u_plus_RL[i]    = np.max(abs_error_rmsf_u_plus_RL[i,:])
    Linf_error_rmsf_v_plus_RL[i]    = np.max(abs_error_rmsf_v_plus_RL[i,:])
    Linf_error_rmsf_w_plus_RL[i]    = np.max(abs_error_rmsf_w_plus_RL[i,:])
    Linf_error_avg_u_plus_nonRL[i]  = np.max(abs_error_avg_u_plus_nonRL[i,:])
    Linf_error_avg_v_plus_nonRL[i]  = np.max(abs_error_avg_v_plus_nonRL[i,:])
    Linf_error_avg_w_plus_nonRL[i]  = np.max(abs_error_avg_w_plus_nonRL[i,:])
    Linf_error_rmsf_u_plus_nonRL[i] = np.max(abs_error_rmsf_u_plus_nonRL[i,:])
    Linf_error_rmsf_v_plus_nonRL[i] = np.max(abs_error_rmsf_v_plus_nonRL[i,:])
    Linf_error_rmsf_w_plus_nonRL[i] = np.max(abs_error_rmsf_w_plus_nonRL[i,:])
print("Errors calculated successfully!")

# --- Errors logging ---

# Store error logs in file
error_log_filename = f"{postDir}/errors_{iteration}_ensemble{ensemble}.txt"
print(f"\nWriting errors in file '{error_log_filename}'")
with open(error_log_filename, "w") as file:
    # avg_u errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_u:")
    file.write(f"\n\nL1 Error RL: {L1_error_avg_u_plus_RL}")
    file.write(f"\nL1 Error nonRL: {L1_error_avg_u_plus_nonRL}")
    file.write(f"\n\nL2 Error RL (RMS): {L2_error_avg_u_plus_RL}")
    file.write(f"\nL2 Error nonRL (RMS): {L2_error_avg_u_plus_nonRL}")
    file.write(f"\n\nLinf Error RL: {Linf_error_avg_u_plus_RL}")
    file.write(f"\nLinf Error nonRL: {Linf_error_avg_u_plus_nonRL}")
    # avg_v errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_v:")
    file.write(f"\n\nL1 Error RL: {L1_error_avg_v_plus_RL}")
    file.write(f"\nL1 Error nonRL: {L1_error_avg_v_plus_nonRL}")
    file.write(f"\n\nL2 Error RL (RMS): {L2_error_avg_v_plus_RL}")
    file.write(f"\nL2 Error nonRL (RMS): {L2_error_avg_v_plus_nonRL}")
    file.write(f"\n\nLinf Error RL: {Linf_error_avg_v_plus_RL}")
    file.write(f"\nLinf Error nonRL: {Linf_error_avg_v_plus_nonRL}")
    # avg_w errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_w:")
    file.write(f"\n\nL1 Error RL: {L1_error_avg_w_plus_RL}")
    file.write(f"\nL1 Error nonRL: {L1_error_avg_w_plus_nonRL}")
    file.write(f"\n\nL2 Error RL (RMS): {L2_error_avg_w_plus_RL}")
    file.write(f"\nL2 Error nonRL (RMS): {L2_error_avg_w_plus_nonRL}")
    file.write(f"\n\nLinf Error RL: {Linf_error_avg_w_plus_RL}")
    file.write(f"\nLinf Error nonRL: {Linf_error_avg_w_plus_nonRL}")
    # rmsf_u errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_u:")
    file.write(f"\n\nL1 Error RL: {L1_error_rmsf_u_plus_RL}")
    file.write(f"\nL1 Error nonRL: {L1_error_rmsf_u_plus_nonRL}")
    file.write(f"\n\nL2 Error RL (RMS): {L2_error_rmsf_u_plus_RL}")
    file.write(f"\nL2 Error nonRL (RMS): {L2_error_rmsf_u_plus_nonRL}")
    file.write(f"\n\nLinf Error RL: {Linf_error_rmsf_u_plus_RL}")
    file.write(f"\nLinf Error nonRL: {Linf_error_rmsf_u_plus_nonRL}")
    # rmsf_v errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_v:")
    file.write(f"\n\nL1 Error RL: {L1_error_rmsf_v_plus_RL}")
    file.write(f"\nL1 Error nonRL: {L1_error_rmsf_v_plus_nonRL}")
    file.write(f"\n\nL2 Error RL (RMS): {L2_error_rmsf_v_plus_RL}")
    file.write(f"\nL2 Error nonRL (RMS): {L2_error_rmsf_v_plus_nonRL}")
    file.write(f"\n\nLinf Error RL: {Linf_error_rmsf_v_plus_RL}")
    file.write(f"\nLinf Error nonRL: {Linf_error_rmsf_v_plus_nonRL}")
    # rmsf_w errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_w:")
    file.write(f"\n\nL1 Error RL: {L1_error_rmsf_w_plus_RL}")
    file.write(f"\nL1 Error nonRL: {L1_error_rmsf_w_plus_nonRL}")
    file.write(f"\n\nL2 Error RL (RMS): {L2_error_rmsf_w_plus_RL}")
    file.write(f"\nL2 Error nonRL (RMS): {L2_error_rmsf_w_plus_nonRL}")
    file.write(f"\n\nLinf Error RL: {Linf_error_rmsf_w_plus_RL}")
    file.write(f"\nLinf Error nonRL: {Linf_error_rmsf_w_plus_nonRL}")
print("Errors written successfully!")

# Print error logs in terminal
with open(error_log_filename, "r") as file:
    content = file.read()
    print(content)

# --- Errors Plots ---
print("\nBuilding error plots...")

# L1-Error plot
plt.semilogy( averaging_time_nonRL, L1_error_avg_u_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L1_error_avg_u_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L1_error_avg_v_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L1_error_avg_v_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L1_error_avg_w_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L1_error_avg_w_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L1_error_rmsf_u_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L1_error_rmsf_u_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L1_error_rmsf_v_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L1_error_rmsf_v_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L1_error_rmsf_w_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L1_error_rmsf_w_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$ RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( 'L1 Error' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/L1_error_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

# L2-Error plot
plt.semilogy( averaging_time_nonRL, L2_error_avg_u_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L2_error_avg_u_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L2_error_avg_v_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L2_error_avg_v_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L2_error_avg_w_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L2_error_avg_w_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L2_error_rmsf_u_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L2_error_rmsf_u_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L2_error_rmsf_v_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L2_error_rmsf_v_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, L2_error_rmsf_w_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, L2_error_rmsf_w_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$ RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( 'L2 Error' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/L2_error_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

# Linf-Error plot
plt.semilogy( averaging_time_nonRL, Linf_error_avg_u_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_avg_u_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_avg_v_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_avg_v_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_avg_w_plus_nonRL,  linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_avg_w_plus_RL,     linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_rmsf_u_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_rmsf_u_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_rmsf_v_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_rmsf_v_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$ RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_rmsf_w_plus_nonRL, linestyle = '-',                             linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$ non-RL' )
plt.semilogy( averaging_time_nonRL, Linf_error_rmsf_w_plus_RL,    linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$ RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( 'Linf Error' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/Linf_error_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

# L1-Error plot, RATIO RL / non-RL
plt.plot( averaging_time_nonRL, L1_error_avg_u_plus_RL  / L1_error_avg_u_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$' )
plt.plot( averaging_time_nonRL, L1_error_avg_v_plus_RL  / L1_error_avg_v_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$' )
plt.plot( averaging_time_nonRL, L1_error_avg_w_plus_RL  / L1_error_avg_w_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$' )
plt.plot( averaging_time_nonRL, L1_error_rmsf_u_plus_RL / L1_error_rmsf_u_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$' )
plt.plot( averaging_time_nonRL, L1_error_rmsf_v_plus_RL / L1_error_rmsf_v_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$' )
plt.plot( averaging_time_nonRL, L1_error_rmsf_w_plus_RL / L1_error_rmsf_w_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$' )
plt.xlabel( r'Accumulated averaging time $t_{\textrm{avg}}^{+}$' )
plt.ylabel( 'L1 Error Ratio (RL / non-RL)' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/L1_error_ratio_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

# L2-Error plot, RATIO RL / non-RL
plt.plot( averaging_time_nonRL, L2_error_avg_u_plus_RL  / L2_error_avg_u_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$' )
plt.plot( averaging_time_nonRL, L2_error_avg_v_plus_RL  / L2_error_avg_v_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$' )
plt.plot( averaging_time_nonRL, L2_error_avg_w_plus_RL  / L2_error_avg_w_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$' )
plt.plot( averaging_time_nonRL, L2_error_rmsf_u_plus_RL / L2_error_rmsf_u_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$' )
plt.plot( averaging_time_nonRL, L2_error_rmsf_v_plus_RL / L2_error_rmsf_v_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$' )
plt.plot( averaging_time_nonRL, L2_error_rmsf_w_plus_RL / L2_error_rmsf_w_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( 'L2 Error Ratio (RL / non-RL)' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/L2_error_ratio_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

# Linf-Error plot, RATIO RL / non-RL
plt.plot( averaging_time_nonRL, Linf_error_avg_u_plus_RL  / Linf_error_avg_u_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}$' )
plt.plot( averaging_time_nonRL, Linf_error_avg_v_plus_RL  / Linf_error_avg_v_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{v}^{+}$' )
plt.plot( averaging_time_nonRL, Linf_error_avg_w_plus_RL  / Linf_error_avg_w_plus_nonRL,  linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(2), label = r'$\overline{w}^{+}$' )
plt.plot( averaging_time_nonRL, Linf_error_rmsf_u_plus_RL / Linf_error_rmsf_u_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(3), label = r'$u_{\textrm{rms}}^{+}$' )
plt.plot( averaging_time_nonRL, Linf_error_rmsf_v_plus_RL / Linf_error_rmsf_v_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(4), label = r'$v_{\textrm{rms}}^{+}$' )
plt.plot( averaging_time_nonRL, Linf_error_rmsf_w_plus_RL / Linf_error_rmsf_w_plus_nonRL, linestyle = '-', marker = '^', linewidth = 1, color = plt.cm.tab10(5), label = r'$w_{\textrm{rms}}^{+}$' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( 'Linf Error Ratio (RL / non-RL)' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/Linf_error_ratio_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

print("Error plots built successfully!")
