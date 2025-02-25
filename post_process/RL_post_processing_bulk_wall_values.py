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
    iteration       = sys.argv[1]
    ensemble        = sys.argv[2]
    train_name      = sys.argv[3]
    Re_tau          = float(sys.argv[4])     # Friction Reynolds number [-]
    dt_phys         = float(sys.argv[5])
    t_episode_train = float(sys.argv[6])
    case_dir        = sys.argv[7]
    run_mode        = sys.argv[8]
    print(f"\nScript parameters: \n- Iteration: {iteration} \n- Ensemble: {ensemble}\n- Train name: {train_name} \n- Re_tau: {Re_tau} \n- dt_phys: {dt_phys} \n- Train episode period: {t_episode_train} \n- Case directory: {case_dir} \n- Run mode: {run_mode}")
except :
    raise ValueError("Missing call arguments, should be: <iteration> <ensemble> <train_name> <Re_tau> <dt_phys> <case_dir> <run_mode>")

if run_mode == "train":
    print("Run mode is set to training")
elif run_mode == "eval":
    print("Run mode is set to evaluation")
else: 
    raise ValueError(f"Unrecognized input argument run_mode = `{run_mode}`")

print("IMPORTANT: THIS POST-PROCESSING SCRIPT ASSUMES ALL RL AND NON-RL AND REFERENCE SIMULATIONS SHARE THE SAME GRID!")

# --- Case parameters ---
rho_0   = 1.0				# Reference density [kg/m3]
u_tau   = 1.0				# Friction velocity [m/s]
delta   = 1.0				# Channel half-height [m]
mu_ref  = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref  = mu_ref/rho_0			# Kinematic viscosity [m2/s]

#--------------------------------------------------------------------------------------------

# --- Training / Evaluation parameters ---

# training post-processing directory
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# Reference & non-RL data directory
filePath = os.path.dirname(os.path.abspath(__file__))
compareDatasetDir = os.path.join(filePath, f"data_Retau{Re_tau:.0f}")
if run_mode == "train":
    iteration_max_nonRL = 3790000
else:   # run_mode == "eval"
    iteration_max_nonRL = 3860000
max_length_legend_RL = 10

# RL parameters
cfd_n_envs = 1
rl_n_envs  = 8
simulation_time_per_train_step   = t_episode_train                    # total cfd simulated time per training step (in parallel per each cfd_n_envs)
num_global_steps_per_train_step  = int(cfd_n_envs * rl_n_envs)        # num. global steps per training step
num_iterations_per_train_step    = int(np.round(simulation_time_per_train_step / dt_phys))
if run_mode == "train":
    iteration_restart_data_file  = 3210000
else:   # run_mode == "eval"
    iteration_restart_data_file  = 2840000
iteration_end_train_step         = iteration_restart_data_file + num_iterations_per_train_step
assert iteration_restart_data_file + num_iterations_per_train_step == iteration_end_train_step
print("\nRL parameters: \n- Simulation time per train step:", simulation_time_per_train_step, 
      "\n- Num. global steps per train step:", num_global_steps_per_train_step,
      "\n- Num. iterations per train step:", num_iterations_per_train_step,
      "\n- Iteration restart data file (init train step):", iteration_restart_data_file,
      "\n- Iteration end train step:", iteration_end_train_step,
) 


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
if run_mode == "train":
    iteration_nonRL_list = [ (s+1)*num_iterations_per_train_step + iteration_restart_data_file for s in train_step_list]
else:   # run_mode == "eval"
    iteration_nonRL_list = [ iteration_end_train_step ]
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

#--------------------------------------------------------------------------------------------

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
        u_data_RL_aux      = data_file['u'][:,:,:]
        avg_u_data_RL_aux  = data_file['avg_u'][:,:,:]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = avg_u_data_RL_aux[0,0,:].size
        num_points_y      = avg_u_data_RL_aux[0,:,0].size
        num_points_z      = avg_u_data_RL_aux[:,0,0].size
        averaging_time_RL = np.zeros(N)
        u_data_RL         = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        avg_u_data_RL     = np.zeros([N, num_points_z, num_points_y, num_points_x])         
    # Fill allocation arrays
    averaging_time_RL[i]    = averaging_time_RL_aux
    u_data_RL[i,:,:,:]      = u_data_RL_aux
    avg_u_data_RL[i,:,:,:]  = avg_u_data_RL_aux
    # Logging
    print(f"RL non-converged data imported from file '{filename_RL}' - averaging time: {averaging_time_RL_aux:.6f}")

print("\nImporting data from non-RL files:")
for i in range(N):
    filename_nonRL = filename_nonRL_list[i]
    with h5py.File( filename_nonRL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_nonRL_aux = data_file.attrs["AveragingTime"][0]
        u_data_nonRL_aux      = data_file['u'][:,:,:]
        avg_u_data_nonRL_aux  = data_file['avg_u'][:,:,:]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = avg_u_data_nonRL_aux[0,0,:].size
        num_points_y      = avg_u_data_nonRL_aux[0,:,0].size
        num_points_z      = avg_u_data_nonRL_aux[:,0,0].size
        averaging_time_nonRL = np.zeros(N)
        u_data_nonRL         = np.zeros([N, num_points_z, num_points_y, num_points_x])         
        avg_u_data_nonRL     = np.zeros([N, num_points_z, num_points_y, num_points_x])         
    # Fill allocation arrays
    averaging_time_nonRL[i]    = averaging_time_nonRL_aux
    u_data_nonRL[i,:,:,:]      = u_data_nonRL_aux
    avg_u_data_nonRL[i,:,:,:]  = avg_u_data_nonRL_aux
    print(f"non-RL non-converged data imported from file '{filename_nonRL}' - averaging time: {averaging_time_nonRL_aux:.6f}")

print("\nImporting reference data (non-RL):")
with h5py.File( filename_ref, 'r' ) as data_file:
    averaging_time_ref = data_file.attrs["AveragingTime"][0]
    x_data_ref      = data_file['x'][:,:,:]
    y_data_ref      = data_file['y'][:,:,:]
    z_data_ref      = data_file['z'][:,:,:]
    u_data_ref      = data_file['u'][:,:,:]
    avg_u_data_ref  = data_file['avg_u'][:,:,:]
assert ((averaging_time_ref > averaging_time_RL).all() and (averaging_time_ref > averaging_time_nonRL).all()), f"Reference data averaging time {averaging_time_ref:.6f} must be greater than non-converged averaging times from non-converged RL & non-RL"
print(f"\nNon-RL converged reference data imported from file '{filename_ref}' - averaging time: {averaging_time_ref:.6f}")
print("Data imported successfully!")

x_data = x_data_ref
y_data = y_data_ref
z_data = z_data_ref

#--------------------------------------------------------------------------------------------

# -------------- Calculate bulk & wall quantitites --------------

# --- Calculate U_bulk ---

sum_avg_u_volume_RL    = np.zeros(N);       sum_u_volume_RL    = np.zeros(N)
sum_avg_u_volume_nonRL = np.zeros(N);       sum_u_volume_nonRL = np.zeros(N)
sum_avg_u_volume_ref   = 0.0;               sum_u_volume_ref   = 0.0
sum_volume = 0.0
for i in range( 1, num_points_x-1 ):
    for j in range( 1, num_points_y-1 ):
        for k in range( 1, num_points_z-1 ):
            # Geometric stuff
            delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
            delta_y = 0.5*( y_data[k,j+1,i] - y_data[k,j-1,i] )
            delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
            delta_volume = delta_x*delta_y*delta_z
            sum_volume  += delta_volume
            # Integrate quantities
            for n in range(N):
                # > RL
                sum_u_volume_RL[n]        += u_data_RL[n,k,j,i] * delta_volume
                sum_avg_u_volume_RL[n]    += avg_u_data_RL[n,k,j,i] * delta_volume
                # > non-RL, non-converged
                sum_u_volume_nonRL[n]     += u_data_nonRL[n,k,j,i] * delta_volume
                sum_avg_u_volume_nonRL[n] += avg_u_data_nonRL[n,k,j,i] * delta_volume
            # > non-RL, reference
            sum_u_volume_ref     += u_data_ref[k,j,i] * delta_volume
            sum_avg_u_volume_ref += avg_u_data_ref[k,j,i] * delta_volume

avg_u_b_RL    = sum_avg_u_volume_RL    / sum_volume;            u_b_RL    = sum_u_volume_RL    / sum_volume
avg_u_b_nonRL = sum_avg_u_volume_nonRL / sum_volume;            u_b_nonRL = sum_u_volume_nonRL / sum_volume
avg_u_b_ref   = sum_avg_u_volume_ref   / sum_volume;            u_b_ref   = sum_u_volume_ref   / sum_volume
print("\n----------------------------------------------------------------------------------------------------")
print( "\nREFERENCE Numerical avg_u_bulk:", avg_u_b_ref );      print( "\nREFERENCE Numerical instantaneous u_bulk:", u_b_ref )    
print( "\nRL Numerical avg_u_bulk:",        avg_u_b_RL );       print( "\nRL Numerical instantaneous u_bulk:",        u_b_RL )    
print( "\nnon-RL Numerical avg_u_bulk:",    avg_u_b_nonRL );    print( "\nnon-RL Numerical instantaneous u_bulk:",    u_b_nonRL )        

# --- Calculate \tau_wall ---

### Average variables in space
sum_avg_u_inner_RL    = np.zeros(N);   sum_avg_u_boundary_RL    = np.zeros(N)
sum_avg_u_inner_nonRL = np.zeros(N);   sum_avg_u_boundary_nonRL = np.zeros(N)
sum_avg_u_inner_ref   = 0.0;           sum_avg_u_boundary_ref   = 0.0
sum_surface = 0.0
for i in range( 1, num_points_x-1 ):
    for k in range( 1, num_points_z-1 ):
        # -------- Bottom wall -------- 
        j = 0
        delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
        delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
        delta_surface = delta_x*delta_z
        sum_surface += delta_surface
        for n in range(N):
            # > RL
            sum_avg_u_inner_RL[n]    += avg_u_data_RL[n,k,j+1,i] * delta_surface
            sum_avg_u_boundary_RL[n] += avg_u_data_RL[n,k,j,i] * delta_surface
            # > non-RL, non-converged
            sum_avg_u_inner_nonRL[n]    += avg_u_data_nonRL[n,k,j+1,i] * delta_surface
            sum_avg_u_boundary_nonRL[n] += avg_u_data_nonRL[n,k,j,i] * delta_surface
        # > non-RL, reference
        sum_avg_u_inner_ref    += avg_u_data_ref[k,j+1,i] * delta_surface
        sum_avg_u_boundary_ref += avg_u_data_ref[k,j,i] * delta_surface
        # -------- Top wall --------
        j = num_points_y - 1
        delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
        delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
        delta_surface = delta_x*delta_z
        sum_surface += delta_surface
        for n in range(N):
            # > RL
            sum_avg_u_inner_RL[n]    += avg_u_data_RL[n,k,j-1,i] * delta_surface
            sum_avg_u_boundary_RL[n] += avg_u_data_RL[n,k,j,i] * delta_surface
            # > non-RL, non-converged
            sum_avg_u_inner_nonRL[n]    += avg_u_data_nonRL[n,k,j-1,i] * delta_surface
            sum_avg_u_boundary_nonRL[n] += avg_u_data_nonRL[n,k,j,i] * delta_surface
        # > non-RL, reference
        sum_avg_u_inner_ref    += avg_u_data_ref[k,j-1,i] * delta_surface
        sum_avg_u_boundary_ref += avg_u_data_ref[k,j,i] * delta_surface

avg_u_inner_RL       = sum_avg_u_inner_RL / sum_surface
avg_u_boundary_RL    = sum_avg_u_boundary_RL / sum_surface
avg_u_inner_nonRL    = sum_avg_u_inner_nonRL / sum_surface
avg_u_boundary_nonRL = sum_avg_u_boundary_nonRL / sum_surface
avg_u_inner_ref      = sum_avg_u_inner_ref / sum_surface
avg_u_boundary_ref   = sum_avg_u_boundary_ref / sum_surface
tau_w_num_RL    = mu_ref * (avg_u_inner_RL    - avg_u_boundary_RL)    / (y_data[0,1,0] - y_data[0,0,0])
tau_w_num_nonRL = mu_ref * (avg_u_inner_nonRL - avg_u_boundary_nonRL) / (y_data[0,1,0] - y_data[0,0,0])
tau_w_num_ref   = mu_ref * (avg_u_inner_ref   - avg_u_boundary_ref)   / (y_data[0,1,0] - y_data[0,0,0])
u_tau_num_RL    = np.sqrt(tau_w_num_RL    / rho_0) 
u_tau_num_nonRL = np.sqrt(tau_w_num_nonRL / rho_0) 
u_tau_num_ref   = np.sqrt(tau_w_num_ref   / rho_0) 
print("\n----------------------------------------------------------------------------------------------------")
print("\nREFERENCE Numerical tau_w:", tau_w_num_ref)
print("REFERENCE Numerical u_tau:", u_tau_num_ref)
print("\nRL Numerical tau_w:", tau_w_num_RL)
print("RL Numerical u_tau:", u_tau_num_RL)
print("\nnon-RL Numerical tau_w:", tau_w_num_nonRL)
print("non-RL Numerical u_tau:", u_tau_num_nonRL)

#--------------------------------------------------------------------------------------------

# -------------- Build plots --------------

# --- (inst) u_bulk plot ---
plt.plot( averaging_time_nonRL, u_b_ref * np.ones(N),  linestyle = '-',                             linewidth = 2, color = "k",             label = r'Reference' )
plt.plot( averaging_time_nonRL, u_b_nonRL,             linestyle = '--',                            linewidth = 2, color = plt.cm.tab10(0), label = r'non-RL' )
plt.plot( averaging_time_nonRL, u_b_RL,                linestyle=':', marker = '^', markersize = 2, linewidth = 2, color = plt.cm.tab10(1), label = r'RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical $u^{+}_b$' )
#plt.ylim(14,14.8)
#plt.yticks(np.arange(14,14.8,0.1))
#plt.grid(which='major',axis='y')
plt.grid(which='both',axis='y')
plt.tick_params( axis = 'both', pad = 7.5 )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
filename = f'{postDir}/numerical_u_bulk_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")

# --- avg_u_bulk plot ---
plt.plot( averaging_time_nonRL, avg_u_b_ref * np.ones(N),  linestyle = '-',                             linewidth = 2, color = "k",             label = r'Reference' )
plt.plot( averaging_time_nonRL, avg_u_b_nonRL,             linestyle = '--',                            linewidth = 2, color = plt.cm.tab10(0), label = r'non-RL' )
plt.plot( averaging_time_nonRL, avg_u_b_RL,                linestyle=':', marker = '^', markersize = 2, linewidth = 2, color = plt.cm.tab10(1), label = r'RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical $\overline{u}^{+}_b$' )
plt.ylim(14,14.8)
plt.yticks(np.arange(14,14.8,0.1))
plt.grid(which='major',axis='y')
plt.tick_params( axis = 'both', pad = 7.5 )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
filename = f'{postDir}/numerical_avg_u_bulk_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")

# --- avg_u_bulk & (inst) u_bulk plot ---
plt.plot( averaging_time_nonRL, avg_u_b_ref * np.ones(N),  linestyle = '-',                                zorder = 1, linewidth = 1, color = "k",             label = r'$\overline{u}^{+}_b$ Reference' )
plt.plot( averaging_time_nonRL, avg_u_b_nonRL,             linestyle = '-',                                zorder = 1, linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}_b$ non-RL' )
plt.plot( averaging_time_nonRL, avg_u_b_RL,                linestyle = '-', marker = 'v', markersize = 2,  zorder = 1, linewidth = 1, color = plt.cm.tab10(1), label = r'$\overline{u}^{+}_b$ RL' )
plt.plot( averaging_time_nonRL, u_b_ref * np.ones(N),      linestyle = '--',                               zorder = 0, linewidth = 1, color = "k",             label = r'${u}^{+}_b$ Reference' )
plt.plot( averaging_time_nonRL, u_b_nonRL,                 linestyle = '--',                               zorder = 0, linewidth = 1, color = plt.cm.tab10(0), label = r'${u}^{+}_b$ non-RL' )
plt.plot( averaging_time_nonRL, u_b_RL,                    linestyle = '--', marker = '^', markersize = 2, zorder = 0, linewidth = 1, color = plt.cm.tab10(1), label = r'${u}^{+}_b$ RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical avg. $\overline{u}^{+}_b$ and inst. $\overline{u}^{+}_b$' )
plt.ylim(12,17)
plt.yticks(np.arange(12,17,1.0))
plt.grid(which='major',axis='y')
plt.tick_params( axis = 'both', pad = 7.5 )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
filename = f'{postDir}/numerical_inst_avg_u_bulk_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")

# --- tau_w plot ---
plt.plot( averaging_time_nonRL, tau_w_num_ref * np.ones(N),  linestyle = '-',                             linewidth = 2, color = "k",             label = r'Reference' )
plt.plot( averaging_time_nonRL, tau_w_num_nonRL,             linestyle = '--',                            linewidth = 2, color = plt.cm.tab10(0), label = r'non-RL' )
plt.plot( averaging_time_nonRL, tau_w_num_RL,                linestyle=':', marker = '^', markersize = 2, linewidth = 2, color = plt.cm.tab10(1), label = r'RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical $\tau_w$' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/numerical_tau_w_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")

# --- u_tau plot ---
plt.plot( averaging_time_nonRL, u_tau_num_ref * np.ones(N),  linestyle = '-',                             linewidth = 2, color = "k",             label = r'Reference' )
plt.plot( averaging_time_nonRL, u_tau_num_nonRL,             linestyle = '--',                            linewidth = 2, color = plt.cm.tab10(0), label = r'non-RL' )
plt.plot( averaging_time_nonRL, u_tau_num_RL,                linestyle=':', marker = '^', markersize = 2, linewidth = 2, color = plt.cm.tab10(1), label = r'RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical $u_\tau$' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/numerical_u_tau_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")