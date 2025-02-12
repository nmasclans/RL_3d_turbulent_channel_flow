# Description: 
# 1. Compute DNS reynolds stresses, anisotropy tensor, TKE
# 2. Calculate eigenvectors and eigenvalues of anisotropy tensor
# 3. Project the anisotropy eigenvalues into a baricentric map by linear mapping
# 4. Check: is the anisotropy tensor / Reynolds stresses satisfying the realizability conditions?

# Usage
# python3 anisotropy_tensor_odt_vs_dns.py [case_name] [reynolds_number]

# Arguments:
# case_name (str): Name of the case
# reynolds_number (int): reynolds number of the dns case, to get comparable dns result.

# Example Usage:
# python3 anisotropy_tensor.py channel180 180

# Comments:
# Values are in wall units (y+, u+) for both ODT and DNS results,
# Scaling is done in the input file (not explicitly here).

import sys
import os
import glob
import h5py    

import numpy as np

from utils import *
from ChannelVisualizer import ChannelVisualizer

#--------------------------------------------------------------------------------------------

verbose = False

# --- Get CASE parameters ---

try :
    iteration  = sys.argv[1]
    ensemble   = sys.argv[2]
    train_name = sys.argv[3]
    Re_tau     = float(sys.argv[4])     # Friction Reynolds number [-]
    dt_phys    = float(sys.argv[5])
    case_dir   = sys.argv[6]
    run_mode        = sys.argv[7] if len(sys.argv) > 7 else None
    print(f"Script parameters: \n- Iteration: {iteration} \n- Ensemble: {ensemble}\n- Train name: {train_name} \n- Re_tau: {Re_tau} \n- dt_phys: {dt_phys}\n- Case directory: {case_dir}")
    if run_mode == "eval":
        print("Run mode is set to evaluation")
except :
    raise ValueError("Missing call arguments, should be: <iteration> <ensemble> <train_name> <Re_tau> <dt_phys> <case_dir>")

# --- Case parameters ---
rho_0   = 1.0				# Reference density [kg/m3]
u_tau   = 1.0				# Friction velocity [m/s]
delta   = 1.0				# Channel half-height [m]
mu_ref  = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref  = mu_ref/rho_0			# Kinematic viscosity [m2/s]

# Training output data
output_data_dir = f"{case_dir}/rhea_exp/output_data/"

# post-processing directory
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# Reference & non-RL data directory
filePath = os.path.dirname(os.path.abspath(__file__))
compareDatasetDir = os.path.join(filePath, f"data_Retau{Re_tau:.0f}")
iteration_max_nonRL = 3790000

if np.isclose(Re_tau, 100, atol=1e-8):
    Re_tau_theoretical = 100.0
    y_actuators_boundaries = np.array([0.05, 0.25, 0.50, 0.75, 1.0])
elif np.isclose(Re_tau, 180, atol=1e-8):
    Re_tau_theoretical = 180.0
    y_actuators_boundaries = np.array([0.027777, 0.0962555, 0.3242615, 0.640158, 1.0])
else:
    raise ValueError(f"'actuators_boundaries' not implemented for Re_tau = {Re_tau}")
y_plus_actuators_boundaries = y_actuators_boundaries * Re_tau_theoretical   

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

# --- Visualizer ---

visualizer = ChannelVisualizer(postDir)
nbins      = 1000

# ----------- Build data h5 filenames ------------

# --- RL filenames ---
pattern = os.path.join(output_data_dir, f"RL_3d_turbulent_channel_flow_{iteration}_ensemble{ensemble}_*.h5")
matching_files = sorted(glob.glob(pattern))
filename_RL_list  = []
file_details_list = []  # list elements of the structure: 'stepxxxxxx', e.g. 'step000064'
global_step_list  = []  # list elements of structure 'int(xxxxxxx)', e.g. 64
if matching_files:
    print("Found files:")
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
        # Store global step number
        global_step = int(file_details.split('step')[1])
        global_step_list.append(global_step)
        # Print the file and the extracted part
        print(f"Filename: {base_filename}, File details: {file_details}")
else:
    print(f"No files found matching the pattern: {pattern}")
global_step_num_list = global_step_list

# --- non-RL filenames ---
train_step_list = [int(gs/num_global_steps_per_train_step) for gs in global_step_num_list]
if run_mode != "eval":
    iteration_nonRL_list = [ (s+1)*num_iterations_per_train_step + iteration_restart_data_file for s in train_step_list]
else:
    iteration_nonRL_list = [ iteration_end_train_step ]
filename_nonRL_list  = [f"{compareDatasetDir}/3d_turbulent_channel_flow_{iter}.h5" for iter in iteration_nonRL_list] 
print("\nnon-RL files:")
for i in range(len(train_step_list)):
    print("Filename:", filename_nonRL_list[i], ", Iteration:", iteration_nonRL_list[i])

# --- non-RL converged reference filename ---
filename_ref = f"{compareDatasetDir}/3d_turbulent_channel_flow_reference.h5"

# --- non-RL restart data file
filename_rst = f"{compareDatasetDir}/3d_turbulent_channel_flow_{iteration_restart_data_file}.h5"

# Append restart data file to RL & non-RL files list
# > RL lists:
filename_RL_list.insert(0,filename_rst)
file_details_list.insert(0,'restart')
global_step_num_list.insert(0,'000000') 
N = len(filename_RL_list)   # update N
# > non-RL lists:
filename_nonRL_list.insert(0,filename_rst)
train_step_list.insert(0,0)
iteration_nonRL_list.insert(0,iteration_restart_data_file)
assert N == len(filename_nonRL_list)
# Logging
if run_mode == "eval":
    print(f"RL filenames: {filename_RL_list}")
    print(f"non-RL filenames: {filename_nonRL_list}")

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
N = len(filename_RL_list)
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

print("\nImporting data from RL files:\n")
for i in range(N):
    filename_RL = filename_RL_list[i]
    with h5py.File( filename_RL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_RL_aux    = data_file.attrs["AveragingTime"][0] - dt_phys
        y_data_RL_aux            = data_file['y'][1:-1,1:-1,1:-1]
        favre_uffuff_data_RL_aux = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
        favre_uffvff_data_RL_aux = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
        favre_uffwff_data_RL_aux = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
        favre_vffvff_data_RL_aux = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
        favre_vffwff_data_RL_aux = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
        favre_wffwff_data_RL_aux = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = favre_uffuff_data_RL_aux[0,0,:].size
        num_points_y      = favre_uffuff_data_RL_aux[0,:,0].size
        num_points_z      = favre_uffuff_data_RL_aux[:,0,0].size
        num_points_xz     = num_points_x*num_points_z
        num_points_y_half = int(0.5*num_points_y)
        averaging_time_RL = np.zeros(N)
        y_data_RL            = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        favre_uffuff_data_RL = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        favre_uffvff_data_RL = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        favre_uffwff_data_RL = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        favre_vffvff_data_RL = np.zeros([N, num_points_z, num_points_y, num_points_x])            
        favre_vffwff_data_RL = np.zeros([N, num_points_z, num_points_y, num_points_x])            
        favre_wffwff_data_RL = np.zeros([N, num_points_z, num_points_y, num_points_x])  
    # Fill allocation arrays
    averaging_time_RL[i]          = averaging_time_RL_aux
    y_data_RL[i,:,:,:]            = y_data_RL_aux
    favre_uffuff_data_RL[i,:,:,:] = favre_uffuff_data_RL_aux
    favre_uffvff_data_RL[i,:,:,:] = favre_uffvff_data_RL_aux
    favre_uffwff_data_RL[i,:,:,:] = favre_uffwff_data_RL_aux
    favre_vffvff_data_RL[i,:,:,:] = favre_vffvff_data_RL_aux
    favre_vffwff_data_RL[i,:,:,:] = favre_vffwff_data_RL_aux
    favre_wffwff_data_RL[i,:,:,:] = favre_wffwff_data_RL_aux
    # Logging
    print(f"RL non-converged data imported from file '{filename_RL}' - averaging time: {averaging_time_RL_aux:.6f}")

# --- Get non-RL (non-converged) data from h5 file ---
print("\nImporting data from non-RL files:\n")
for i in range(N):
    filename_nonRL = filename_nonRL_list[i]
    with h5py.File( filename_nonRL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_nonRL_aux    = data_file.attrs["AveragingTime"][0] - dt_phys
        y_data_nonRL_aux            = data_file['y'][1:-1,1:-1,1:-1]
        favre_uffuff_data_nonRL_aux = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
        favre_uffvff_data_nonRL_aux = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
        favre_uffwff_data_nonRL_aux = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
        favre_vffvff_data_nonRL_aux = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
        favre_vffwff_data_nonRL_aux = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
        favre_wffwff_data_nonRL_aux = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = favre_uffuff_data_nonRL_aux[0,0,:].size
        num_points_y      = favre_uffuff_data_nonRL_aux[0,:,0].size
        num_points_z      = favre_uffuff_data_nonRL_aux[:,0,0].size
        num_points_xz     = num_points_x*num_points_z
        num_points_y_half = int(0.5*num_points_y)
        averaging_time_nonRL    = np.zeros(N)
        y_data_nonRL            = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        favre_uffuff_data_nonRL = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        favre_uffvff_data_nonRL = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        favre_uffwff_data_nonRL = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        favre_vffvff_data_nonRL = np.zeros([N, num_points_z, num_points_y, num_points_x])            
        favre_vffwff_data_nonRL = np.zeros([N, num_points_z, num_points_y, num_points_x])            
        favre_wffwff_data_nonRL = np.zeros([N, num_points_z, num_points_y, num_points_x])  
    # Fill allocation arrays
    averaging_time_nonRL[i]          = averaging_time_nonRL_aux
    y_data_nonRL[i,:,:,:]            = y_data_nonRL_aux
    favre_uffuff_data_nonRL[i,:,:,:] = favre_uffuff_data_nonRL_aux
    favre_uffvff_data_nonRL[i,:,:,:] = favre_uffvff_data_nonRL_aux
    favre_uffwff_data_nonRL[i,:,:,:] = favre_uffwff_data_nonRL_aux
    favre_vffvff_data_nonRL[i,:,:,:] = favre_vffvff_data_nonRL_aux
    favre_vffwff_data_nonRL[i,:,:,:] = favre_vffwff_data_nonRL_aux
    favre_wffwff_data_nonRL[i,:,:,:] = favre_wffwff_data_nonRL_aux
    print(f"non-RL non-converged data imported from file '{filename_nonRL}' - averaging time: {averaging_time_nonRL_aux:.6f}")

# --- Get non-RL converged reference data from h5 file ---
print("\nImporting reference data (non-RL):\n")
with h5py.File( filename_ref, 'r' ) as data_file:
    averaging_time_ref    = data_file.attrs["AveragingTime"][0]
    y_data_ref            = data_file['y'][1:-1,1:-1,1:-1]
    favre_uffuff_data_ref = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
    favre_uffvff_data_ref = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
    favre_uffwff_data_ref = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
    favre_vffvff_data_ref = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
    favre_vffwff_data_ref = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
    favre_wffwff_data_ref = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
assert ((averaging_time_ref > averaging_time_RL).all() and (averaging_time_ref > averaging_time_nonRL).all()), f"Reference data averaging time {averaging_time_ref:.6f} must be greater than non-converged averaging times from non-converged RL & non-RL"
print(f"Non-RL reference data imported from file '{filename_ref}' - averaging time: {averaging_time_ref:.6f}")
print("Data imported successfully!")

# -------------- Averaging fields using XZ symmetries --------------

print("\nAveraging fields in space...")

### Allocate averaged variables
y_plus_RL       = np.zeros([N, num_points_y_half]);   y_plus_nonRL       = np.zeros([N, num_points_y_half]);   y_plus_ref       = np.zeros(num_points_y_half)
y_delta_RL      = np.zeros([N, num_points_y_half]);   y_delta_nonRL      = np.zeros([N, num_points_y_half]);   y_delta_ref      = np.zeros(num_points_y_half)
favre_uffuff_RL = np.zeros([N, num_points_y_half]);   favre_uffuff_nonRL = np.zeros([N, num_points_y_half]);   favre_uffuff_ref = np.zeros(num_points_y_half)
favre_uffvff_RL = np.zeros([N, num_points_y_half]);   favre_uffvff_nonRL = np.zeros([N, num_points_y_half]);   favre_uffvff_ref = np.zeros(num_points_y_half)
favre_uffwff_RL = np.zeros([N, num_points_y_half]);   favre_uffwff_nonRL = np.zeros([N, num_points_y_half]);   favre_uffwff_ref = np.zeros(num_points_y_half)
favre_vffvff_RL = np.zeros([N, num_points_y_half]);   favre_vffvff_nonRL = np.zeros([N, num_points_y_half]);   favre_vffvff_ref = np.zeros(num_points_y_half)
favre_vffwff_RL = np.zeros([N, num_points_y_half]);   favre_vffwff_nonRL = np.zeros([N, num_points_y_half]);   favre_vffwff_ref = np.zeros(num_points_y_half)
favre_wffwff_RL = np.zeros([N, num_points_y_half]);   favre_wffwff_nonRL = np.zeros([N, num_points_y_half]);   favre_wffwff_ref = np.zeros(num_points_y_half)
### Average variables in space
for j in range( 0, num_points_y ):
    # log progress
    if j % (num_points_y//10 or 1) == 0:
        print(f"{j/num_points_y*100:.0f}%")
    # identify domain region
    aux_j = j
    if( j > ( int( 0.5*num_points_y ) - 1 ) ):
        aux_j = num_points_y - j - 1
        is_half_top = True
    else:
        is_half_top = False
    for i in range( 0, num_points_x ):
        for k in range( 0, num_points_z ):
            for n in range(N):
                # RL data:
                if is_half_top:
                    y_plus_RL[n,aux_j]   += ( 0.5/num_points_xz )*( 2*delta - y_data_RL[n,k,j,i] )*( u_tau/nu_ref )
                    y_delta_RL[n,aux_j]  += ( 0.5/num_points_xz )*( 2*delta - y_data_RL[n,k,j,i] )/delta
                else:
                    y_plus_RL[n,aux_j]   += ( 0.5/num_points_xz )*y_data_RL[n,k,j,i]*( u_tau/nu_ref )
                    y_delta_RL[n,aux_j]  += ( 0.5/num_points_xz )*y_data_RL[n,k,j,i]/delta
                favre_uffuff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_RL[n,k,j,i]
                favre_uffvff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_RL[n,k,j,i]
                favre_uffwff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_RL[n,k,j,i]
                favre_vffvff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_RL[n,k,j,i]
                favre_vffwff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_vffwff_data_RL[n,k,j,i]
                favre_wffwff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_RL[n,k,j,i]
                # non-RL data, non-converged:
                if is_half_top:
                    y_plus_nonRL[n,aux_j]   += ( 0.5/num_points_xz )*(2.0*delta - y_data_nonRL[n,k,j,i])*(u_tau/nu_ref);    
                    y_delta_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*(2.0*delta - y_data_nonRL[n,k,j,i])/delta;             
                else:
                    y_plus_nonRL[n,aux_j]   += ( 0.5/num_points_xz )*y_data_nonRL[n,k,j,i]*( u_tau/nu_ref );    
                    y_delta_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*y_data_nonRL[n,k,j,i]/delta;               
                favre_uffuff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_nonRL[n,k,j,i];          
                favre_uffvff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_nonRL[n,k,j,i];          
                favre_uffwff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_nonRL[n,k,j,i];          
                favre_vffvff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_nonRL[n,k,j,i];          
                favre_vffwff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_vffwff_data_nonRL[n,k,j,i];          
                favre_wffwff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_nonRL[n,k,j,i];          
            # reference:
            if is_half_top:
                y_plus_ref[aux_j]  += ( 0.5/num_points_xz )*(2.0*delta - y_data_ref[k,j,i])*(u_tau/nu_ref);  
                y_delta_ref[aux_j] += ( 0.5/num_points_xz )*(2.0*delta - y_data_ref[k,j,i])/delta;             
            else:
                y_plus_ref[aux_j]       += ( 0.5/num_points_xz )*y_data_ref[k,j,i]*(u_tau/nu_ref)
                y_delta_ref[aux_j]      += ( 0.5/num_points_xz )*y_data_ref[k,j,i]/delta
            favre_uffuff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_ref[k,j,i]
            favre_uffvff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_ref[k,j,i]
            favre_uffwff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_ref[k,j,i]
            favre_vffvff_ref[aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_ref[k,j,i]
            favre_wffwff_ref[aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_ref[k,j,i];          
# Reduce avg_y_plus_RL along n_RL dimension (idem!)
# > RL
for i in range(1,N):
    if not np.allclose(y_plus_RL[0], y_plus_RL[i], atol=1e-6) or not np.allclose(y_delta_RL[0], y_delta_RL[i], atol=1e-6):
        raise ValueError("y-plus values of different RL h5 files should be equal!")
y_plus_RL  = np.mean(y_plus_RL,  axis=0)
y_delta_RL = np.mean(y_delta_RL, axis=0)
# > non-RL
for i in range(1,N):
    if not np.allclose(y_plus_nonRL[0], y_plus_nonRL[i], atol=1e-6) or not np.allclose(y_delta_nonRL[0], y_delta_nonRL[i], atol=1e-6):
        raise ValueError("y-plus values of different non-RL h5 files should be equal!")
y_plus_nonRL  = np.mean(y_plus_nonRL,  axis=0)
y_delta_nonRL = np.mean(y_delta_nonRL, axis=0)

print("Fields averaged successfully!")


# ----------- Decompose Rij into d.o.f --------------

print("\nDecomposing Rij into Rij dof...")
Rkk_RL     = np.zeros([N, num_points_y_half]);    Rkk_nonRL     = np.zeros([N, num_points_y_half]);    Rkk_ref     = np.zeros(num_points_y_half)
lambda1_RL = np.zeros([N, num_points_y_half]);    lambda1_nonRL = np.zeros([N, num_points_y_half]);    lambda1_ref = np.zeros(num_points_y_half)
lambda2_RL = np.zeros([N, num_points_y_half]);    lambda2_nonRL = np.zeros([N, num_points_y_half]);    lambda2_ref = np.zeros(num_points_y_half)
lambda3_RL = np.zeros([N, num_points_y_half]);    lambda3_nonRL = np.zeros([N, num_points_y_half]);    lambda3_ref = np.zeros(num_points_y_half)
xmap1_RL   = np.zeros([N, num_points_y_half]);    xmap1_nonRL   = np.zeros([N, num_points_y_half]);    xmap1_ref   = np.zeros(num_points_y_half)
xmap2_RL   = np.zeros([N, num_points_y_half]);    xmap2_nonRL   = np.zeros([N, num_points_y_half]);    xmap2_ref   = np.zeros(num_points_y_half)
eigval_RL  = np.zeros([N, num_points_y_half,3]);  eigval_nonRL  = np.zeros([N, num_points_y_half,3]);  eigval_ref  = np.zeros([num_points_y_half,3])

# RL data
for i in range(N):
    ( Rkk_RL[i], lambda1_RL[i], lambda2_RL[i], lambda3_RL[i], xmap1_RL[i], xmap2_RL[i] ) \
        = compute_reynolds_stress_dof( favre_uffuff_RL[i], favre_uffvff_RL[i], favre_uffwff_RL[i], favre_vffvff_RL[i], favre_vffwff_RL[i], favre_wffwff_RL[i], verbose=verbose )
    eigval_RL[i,:,0] = lambda1_RL[i]
    eigval_RL[i,:,1] = lambda2_RL[i]
    eigval_RL[i,:,2] = lambda3_RL[i]  

# non-RL non-converged data
for i in range(N):
    ( Rkk_nonRL[i], lambda1_nonRL[i], lambda2_nonRL[i], lambda3_nonRL[i], xmap1_nonRL[i], xmap2_nonRL[i] ) \
        = compute_reynolds_stress_dof( favre_uffuff_nonRL[i], favre_uffvff_nonRL[i], favre_uffwff_nonRL[i], favre_vffvff_nonRL[i], favre_vffwff_nonRL[i], favre_wffwff_nonRL[i], verbose=verbose )
    eigval_nonRL[i,:,0] = lambda1_nonRL[i]
    eigval_nonRL[i,:,1] = lambda2_nonRL[i]
    eigval_nonRL[i,:,2] = lambda3_nonRL[i]

# non-RL converged reference data
( Rkk_ref, lambda1_ref, lambda2_ref, lambda3_ref, xmap1_ref, xmap2_ref ) \
    = compute_reynolds_stress_dof( favre_uffuff_ref, favre_uffvff_ref, favre_uffwff_ref, favre_vffvff_ref, favre_vffwff_ref, favre_wffwff_ref, verbose=verbose )
eigval_ref[:,0] = lambda1_ref
eigval_ref[:,1] = lambda2_ref
eigval_ref[:,2] = lambda3_ref

print("Rij decomposed successfully!")

#-----------------------------------------------------------------------------------------
#           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
#-----------------------------------------------------------------------------------------

# ---------------------- Plot Barycentric Map for each RL global step (specific iteration & ensemble) ---------------------- 

print("\nBuilding triangle barycentric map plots...")
#for i_RL in range(n_RL):
#    visualizer.build_anisotropy_tensor_barycentric_xmap_triang(y_delta_RL,    xmap1_RL[i_RL], xmap2_RL[i_RL], averaging_time_RL, f"anisotropy_tensor_barycentric_map_RL_{file_details_list[i_RL]}")
if run_mode == "eval":
    visualizer.build_anisotropy_tensor_barycentric_xmap_triang( y_delta_RL, xmap1_RL[1],    xmap2_RL[1],    averaging_time_RL[1],    f"anisotropy_tensor_barycentric_map_RL_{file_details_list[1]}")
visualizer.build_anisotropy_tensor_barycentric_xmap_triang( y_delta_nonRL,  xmap1_nonRL[1], xmap2_nonRL[1], averaging_time_nonRL[1], f"anisotropy_tensor_barycentric_map_nonRL_{iteration_nonRL_list[1]}")
visualizer.build_anisotropy_tensor_barycentric_xmap_triang( y_delta_ref,    xmap1_ref,      xmap2_ref,      averaging_time_ref,       "anisotropy_tensor_barycentric_map_ref")
print("Triangle barycentric map plotted successfully!")

# ----------------- Plot Animation Frames of um, urmsf, Rij dof for increasing RL global step (specific iteration & ensemble) -----------------

print("\nBuilding gif frames...")
frames_rkk = []; frames_eig = []; frames_xmap_coord = []; frames_xmap_triang = []; 
for i in range(N):
    # log progress
    if i % (N//10 or 1) == 0:
        print(f"{i/N*100:.0f}%")
    # Build frames
    frames_rkk         = visualizer.build_reynolds_stress_tensor_trace_frame(             frames_rkk,         y_delta_RL, y_delta_nonRL, y_delta_ref, Rkk_RL[i],    Rkk_nonRL[i],    Rkk_ref,    averaging_time_RL[i], averaging_time_nonRL[i], global_step_list[i])
    frames_eig         = visualizer.build_anisotropy_tensor_eigenvalues_frame(            frames_eig,         y_delta_RL, y_delta_nonRL, y_delta_ref, eigval_RL[i], eigval_nonRL[i], eigval_ref, averaging_time_RL[i], averaging_time_nonRL[i], global_step_list[i])
    frames_xmap_coord  = visualizer.build_anisotropy_tensor_barycentric_xmap_coord_frame( frames_xmap_coord,  y_delta_RL, y_delta_nonRL, y_delta_ref, xmap1_RL[i],  xmap1_nonRL[i],  xmap1_ref,  xmap2_RL[i], xmap2_nonRL[i], xmap2_ref, averaging_time_RL[i], averaging_time_nonRL[i], global_step_list[i])
    frames_xmap_triang = visualizer.build_anisotropy_tensor_barycentric_xmap_triang_frame(frames_xmap_triang, y_delta_RL, y_delta_nonRL, y_delta_ref, xmap1_RL[i],  xmap1_nonRL[i],  xmap1_ref,  xmap2_RL[i], xmap2_nonRL[i], xmap2_ref, averaging_time_RL[i], averaging_time_nonRL[i], global_step_list[i])

print("Building gifs from frames...")
frames_dict = {'anisotropy_tensor_Rkk':frames_rkk, 'anisotropy_tensor_eigenvalues':frames_eig, 'anisotropy_tensor_barycentric_map_coord':frames_xmap_coord, 'anisotropy_tensor_barycentric_map_triangle':frames_xmap_triang}
visualizer.build_main_gifs_from_frames(frames_dict)
print("Gifs plotted successfully!")




