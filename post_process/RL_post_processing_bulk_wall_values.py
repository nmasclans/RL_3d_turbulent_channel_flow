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
    ensemble        = sys.argv[1]
    train_name      = sys.argv[2]
    Re_tau          = float(sys.argv[3])     # Friction Reynolds number [-]
    dt_phys         = float(sys.argv[4])
    t_episode_train = float(sys.argv[5])
    case_dir        = sys.argv[6]
    run_mode        = sys.argv[7]
    rl_n_envs       = int(sys.argv[8])
    print(f"\nScript parameters: \n- Ensemble: {ensemble}\n- Train name: {train_name} \n- Re_tau: {Re_tau} \n- dt_phys: {dt_phys} \n- Train episode period: {t_episode_train} \n- Case directory: {case_dir} \n- Run mode: {run_mode}\n- Num. RL environments / Parallelization cores: {rl_n_envs}")
except :
    raise ValueError("Missing call arguments, should be: <ensemble> <train_name> <Re_tau> <dt_phys> <case_dir> <run_mode> <rl_n_envs>")

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
    iteration_max_nonRL = 4190000
else:   # run_mode == "eval"
    iteration_max_nonRL = 3860000
max_length_legend_RL = 10

# RL parameters
cfd_n_envs = 1
delta_iteration_nonRL            = 10000
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

# --- non-RL converged reference filename ---
filename_ref = f"{compareDatasetDir}/3d_turbulent_channel_flow_reference.h5"

# --- non-RL restart data file
filename_rst = f"{compareDatasetDir}/3d_turbulent_channel_flow_{iteration_restart_data_file}.h5"

# --- RL filenames ---

# Take the h5 file of each global step at iteration number 'iteration',
# or smaller if early episode termination was activated for that step.
# Note: the iteration at early episode termination must be smaller than maximum episode length 'iteration' 

# Get filepath and file details of the last saved iteration of each global step
pattern = f"{case_dir}/rhea_exp/output_data/RL_3d_turbulent_channel_flow_*_ensemble{ensemble}_step*.h5"
matching_files = sorted(glob.glob(pattern))
best_files = {}     # dict: {global_step: (iter_num, filepath)}
if matching_files:
    print("\nRL files:")
    for filepath in matching_files:
        filename       = os.path.basename(filepath)
        parts_filename = filename.split('_')
        # Extract iteration and global step
        try:
            iter_num    = int(parts_filename[5])
            global_step = int(parts_filename[-1].replace('.h5','')[4:])
        except (IndexError, ValueError):
            print(f"Skipping invalid file: {filename}, in filepath: {filepath}")
            continue
        # Keep only the file with the highes iteration for each global step
        if ( global_step not in best_files or iter_num > best_files[global_step][0] ):
            best_files[global_step] = (iter_num, filepath)
    
    # Sort by global step
    sorted_best_files = sorted(best_files.items())      # list of tuples: [(global_step, (iteration, filepath))]
    filename_RL_list     = [file for _, (_, file) in sorted_best_files]
    iteration_RL_list    = [iter for _, (iter, _) in sorted_best_files]
    global_step_RL_list  = [step for step, (_, _) in sorted_best_files]

    # Append restart data file to RL files list
    filename_RL_list.insert(0,filename_rst)
    iteration_RL_list.insert(0,iteration_restart_data_file) 
    global_step_RL_list.insert(0,'000000') 
    N_RL = len(filename_RL_list)

    # Print selected files
    for i in range(N_RL):
        print(f"\nFilename: {filename_RL_list[i]}, \nIteration: {iteration_RL_list[i]}, \nGlobal step: {global_step_RL_list[i]}")
else:
    print(f"No files found matching the pattern: {pattern}")

iter_simulated   = [iter - iteration_restart_data_file for iter in iteration_RL_list]
iter_accumulated = np.sum(iter_simulated)
iter_max_nonRL   = iteration_restart_data_file + iter_accumulated
print(f"\nA total of {iter_accumulated} iterations have been simulated through the RL episodes, \nwhich accounting for restart file {iteration_restart_data_file} iterations \nis equivalent to final non-RL {iter_max_nonRL} iterations.")
iteration_max_nonRL = np.min([iter_max_nonRL, iteration_max_nonRL])
print(f"Last non-RL iteration: {iteration_max_nonRL}")

# --- non-RL filenames ---

if run_mode == "train":
    iteration_nonRL_list = np.arange(iteration_restart_data_file, iteration_max_nonRL, delta_iteration_nonRL)
else:   # run_mode == "eval"
    iteration_nonRL_list = [ iteration_end_train_step ]
filename_nonRL_list  = [f"{compareDatasetDir}/3d_turbulent_channel_flow_{iter}.h5" for iter in iteration_nonRL_list] 
N_nonRL = len(filename_nonRL_list)
print("\nnon-RL files:")
for i in range(N_nonRL):
    print("\nFilename:", filename_nonRL_list[i], ", \nIteration:", iteration_nonRL_list[i])

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
for i in range(N_RL):
    filename_RL = filename_RL_list[i]
    with h5py.File( filename_RL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_RL_aux = data_file.attrs["AveragingTime"][0] - dt_phys
        u_data_RL_aux      = data_file['u'][:,:,:]
        avg_u_data_RL_aux  = data_file['avg_u'][:,:,:]
    # Initialize allocation arrays
    if i == 0:
        num_points_x       = avg_u_data_RL_aux[0,0,:].size
        num_points_y       = avg_u_data_RL_aux[0,:,0].size
        num_points_z       = avg_u_data_RL_aux[:,0,0].size
        averaging_time_RL  = np.zeros(N_RL)
        u_data_RL          = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        avg_u_data_RL      = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
    # Fill allocation arrays
    averaging_time_RL[i]   = averaging_time_RL_aux
    u_data_RL[i,:,:,:]     = u_data_RL_aux
    avg_u_data_RL[i,:,:,:] = avg_u_data_RL_aux
    # Logging
    print(f"RL non-converged data imported from file '{filename_RL}' - averaging time: {averaging_time_RL_aux:.6f}")
averaging_time_simulated_RL    = averaging_time_RL - averaging_time_RL[0]  # averaging_time_RL[0] is the restart file averaging time, t_avg_0
averaging_time_simulated_RL[0] = averaging_time_RL[0]
averaging_time_accum_RL        = np.cumsum(averaging_time_simulated_RL)

print("\nImporting data from non-RL files:")
for i in range(N_nonRL):
    filename_nonRL = filename_nonRL_list[i]
    with h5py.File( filename_nonRL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_nonRL_aux = data_file.attrs["AveragingTime"][0]
        u_data_nonRL_aux         = data_file['u'][:,:,:]
        avg_u_data_nonRL_aux     = data_file['avg_u'][:,:,:]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = avg_u_data_nonRL_aux[0,0,:].size
        num_points_y      = avg_u_data_nonRL_aux[0,:,0].size
        num_points_z      = avg_u_data_nonRL_aux[:,0,0].size
        averaging_time_nonRL = np.zeros(N_nonRL)
        u_data_nonRL         = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        avg_u_data_nonRL     = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
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

sum_avg_u_volume_RL    = np.zeros(N_RL);    sum_u_volume_RL    = np.zeros(N_RL)
sum_avg_u_volume_nonRL = np.zeros(N_nonRL); sum_u_volume_nonRL = np.zeros(N_nonRL)
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
            # > RL
            for n in range(N_RL):
                sum_u_volume_RL[n]        += u_data_RL[n,k,j,i] * delta_volume
                sum_avg_u_volume_RL[n]    += avg_u_data_RL[n,k,j,i] * delta_volume
            # > non-RL, non-converged
            for n in range(N_nonRL):
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
sum_avg_u_inner_RL    = np.zeros(N_RL);    sum_avg_u_boundary_RL    = np.zeros(N_RL)
sum_avg_u_inner_nonRL = np.zeros(N_nonRL); sum_avg_u_boundary_nonRL = np.zeros(N_nonRL)
sum_avg_u_inner_ref   = 0.0;               sum_avg_u_boundary_ref   = 0.0
sum_surface = 0.0
for i in range( 1, num_points_x-1 ):
    for k in range( 1, num_points_z-1 ):
        # -------- Bottom wall -------- 
        j = 0
        delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
        delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
        delta_surface = delta_x*delta_z
        sum_surface += delta_surface
        # > RL
        for n in range(N_RL):
            sum_avg_u_inner_RL[n]    += avg_u_data_RL[n,k,j+1,i] * delta_surface
            sum_avg_u_boundary_RL[n] += avg_u_data_RL[n,k,j,i] * delta_surface
        # > non-RL, non-converged
        for n in range(N_nonRL):
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
        # > RL
        for n in range(N_RL):
            sum_avg_u_inner_RL[n]    += avg_u_data_RL[n,k,j-1,i] * delta_surface
            sum_avg_u_boundary_RL[n] += avg_u_data_RL[n,k,j,i] * delta_surface
        # > non-RL, non-converged
        for n in range(N_nonRL):
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
plt.plot( averaging_time_nonRL,    u_b_ref * np.ones(N_nonRL), linestyle = '-',                                linewidth = 2, color = "k",             label = r'Reference' )
plt.plot( averaging_time_nonRL,    u_b_nonRL,                  linestyle = '--', marker = 's', markersize = 4, linewidth = 2, color = plt.cm.tab10(0), label = r'non-RL' )
plt.plot( averaging_time_accum_RL, u_b_RL,                     linestyle = ':',  marker = '^', markersize = 4, linewidth = 2, color = plt.cm.tab10(3), label = r'RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical $u^{+}_b$' )
#plt.ylim(14,14.8)
#plt.yticks(np.arange(14,14.8,0.1))
#plt.grid(which='major',axis='y')
plt.grid(which='both',axis='y')
plt.tick_params( axis = 'both', pad = 7.5 )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
filename = f'{postDir}/numerical_u_bulk_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")

# --- avg_u_bulk plot ---
plt.plot( averaging_time_nonRL,    avg_u_b_ref * np.ones(N_nonRL), linestyle = '-',                                linewidth = 2, color = "k",             label = r'Reference' )
plt.plot( averaging_time_nonRL,    avg_u_b_nonRL,                  linestyle = '--', marker = 's', markersize = 4, linewidth = 2, color = plt.cm.tab10(0), label = r'non-RL' )
plt.plot( averaging_time_accum_RL, avg_u_b_RL,                     linestyle = ':',  marker = '^', markersize = 4, linewidth = 2, color = plt.cm.tab10(3), label = r'RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical $\overline{u}^{+}_b$' )
plt.ylim(14,14.8)
plt.yticks(np.arange(14,14.8,0.1))
plt.grid(which='major',axis='y')
plt.tick_params( axis = 'both', pad = 7.5 )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
filename = f'{postDir}/numerical_avg_u_bulk_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")

# --- avg_u_bulk & (inst) u_bulk plot ---
plt.plot( averaging_time_nonRL,    avg_u_b_ref * np.ones(N_nonRL), linestyle = '-',                                zorder = 1, linewidth = 1, color = "k",             label = r'$\overline{u}^{+}_b$ Reference' )
plt.plot( averaging_time_nonRL,    avg_u_b_nonRL,                  linestyle = '-',  marker = 's', markersize = 4, zorder = 1, linewidth = 1, color = plt.cm.tab10(0), label = r'$\overline{u}^{+}_b$ non-RL' )
plt.plot( averaging_time_accum_RL, avg_u_b_RL,                     linestyle = '-',  marker = 'v', markersize = 4, zorder = 1, linewidth = 1, color = plt.cm.tab10(3), label = r'$\overline{u}^{+}_b$ RL' )
plt.plot( averaging_time_nonRL,    u_b_ref * np.ones(N_nonRL),     linestyle = '--',                               zorder = 0, linewidth = 1, color = "k",             label = r'${u}^{+}_b$ Reference' )
plt.plot( averaging_time_nonRL,    u_b_nonRL,                      linestyle = '--', marker = 'o', markersize = 4, zorder = 0, linewidth = 1, color = plt.cm.tab10(0), label = r'${u}^{+}_b$ non-RL' )
plt.plot( averaging_time_accum_RL, u_b_RL,                         linestyle = '--', marker = '^', markersize = 4, zorder = 0, linewidth = 1, color = plt.cm.tab10(3), label = r'${u}^{+}_b$ RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical avg. $\overline{u}^{+}_b$ and inst. $\overline{u}^{+}_b$' )
plt.ylim(12,17)
plt.yticks(np.arange(12,17,1.0))
plt.grid(which='major',axis='y')
plt.tick_params( axis = 'both', pad = 7.5 )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
filename = f'{postDir}/numerical_inst_avg_u_bulk_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")

# --- tau_w plot ---
plt.plot( averaging_time_nonRL,    tau_w_num_ref * np.ones(N_nonRL),  linestyle = '-',                            linewidth = 2, color = "k",             label = r'Reference' )
plt.plot( averaging_time_nonRL,    tau_w_num_nonRL,             linestyle = '--',   marker = 's', markersize = 4, linewidth = 2, color = plt.cm.tab10(0), label = r'non-RL' )
plt.plot( averaging_time_accum_RL, tau_w_num_RL,                linestyle=':',      marker = '^', markersize = 4, linewidth = 2, color = plt.cm.tab10(3), label = r'RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical $\tau_w$' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/numerical_tau_w_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")

# --- u_tau plot ---
plt.plot( averaging_time_nonRL,    u_tau_num_ref * np.ones(N_nonRL),  linestyle = '-',                           linewidth = 2, color = "k",             label = r'Reference' )
plt.plot( averaging_time_nonRL,    u_tau_num_nonRL,             linestyle = '--',  marker = 's', markersize = 4, linewidth = 2, color = plt.cm.tab10(0), label = r'non-RL' )
plt.plot( averaging_time_accum_RL, u_tau_num_RL,                linestyle=':',     marker = '^', markersize = 4, linewidth = 2, color = plt.cm.tab10(3), label = r'RL' )
plt.xlabel( r'Accumulated averaging time $t_{avg}^+$' )
plt.ylabel( r'Numerical $u_\tau$' )
plt.grid(which='both',axis='y')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/numerical_u_tau_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"\nBuild plot: '{filename}'")