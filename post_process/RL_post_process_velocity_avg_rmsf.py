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
    raise ValueError("Missing call arguments, should be: <ensemble> <train_name> <Re_tau> <dt_phys> <t_episode_train> <case_dir> <run_mode> <rl_n_envs>")

if run_mode == "train":
    print("Run mode is set to training")
elif run_mode == "eval":
    print("Run mode is set to evaluation")
else: 
    raise ValueError(f"Unrecognized input argument run_mode = `{run_mode}`")
    
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
    iteration_max_nonRL = 7280000
else:   # run_mode == "eval"
    iteration_max_nonRL = 3860000
max_length_legend_RL = 10

# RL parameters
cfd_n_envs = 1
simulation_time_per_train_step   = t_episode_train                    # total cfd simulated time per training step (in parallel per each cfd_n_envs)
num_global_steps_per_train_step  = int(cfd_n_envs * rl_n_envs)        # num. global steps per training step
num_iterations_per_train_step    = int(np.round(simulation_time_per_train_step / dt_phys))
if run_mode == "train":
    iteration_restart_data_file  = 3240000
    delta_iteration_nonRL        = 10000
else:   # run_mode == "eval"
    iteration_restart_data_file  = 2820000
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

#--------------------------------------------------------------------------------------------

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
    print("\RL files:")
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
    global_step_RL_list.insert(0,'restart file') 
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
    iteration_nonRL_list = [ iteration_restart_data_file, iteration_end_train_step ]
filename_nonRL_list  = [f"{compareDatasetDir}/3d_turbulent_channel_flow_{iter}.h5" for iter in iteration_nonRL_list] 
N_nonRL = len(filename_nonRL_list)
print("\nnon-RL files:")
for i in range(N_nonRL):
    print("\nFilename:", filename_nonRL_list[i], ", \nIteration:", iteration_nonRL_list[i])

#--------------------------------------------------------------------------------------------

# ----------- Get RL and non-RL data ------------

# --- Check if RL & non-RL files exists ---

print("\nCheking files...")

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
        averaging_time_RL = np.zeros(N_RL)
        y_data_RL         = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        avg_u_data_RL     = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        avg_v_data_RL     = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        avg_w_data_RL     = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        rmsf_u_data_RL    = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])            
        rmsf_v_data_RL    = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])            
        rmsf_w_data_RL    = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])  
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
averaging_time_simulated_RL    = averaging_time_RL - averaging_time_RL[0]  # averaging_time_RL[0] is the restart file averaging time, t_avg_0
averaging_time_simulated_RL[0] = averaging_time_RL[0]
averaging_time_accum_RL        = np.cumsum(averaging_time_simulated_RL)

print("\nImporting data from non-RL files:")
for i in range(N_nonRL):
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
        averaging_time_nonRL = np.zeros(N_nonRL)
        y_data_nonRL         = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        avg_u_data_nonRL     = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        avg_v_data_nonRL     = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        avg_w_data_nonRL     = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        rmsf_u_data_nonRL    = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])            
        rmsf_v_data_nonRL    = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])            
        rmsf_w_data_nonRL    = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])  
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
#assert ((averaging_time_ref > averaging_time_RL).all() and (averaging_time_ref > averaging_time_nonRL).all()), f"Reference data averaging time {averaging_time_ref:.6f} must be greater than non-converged averaging times from non-converged RL & Uncontrolled"
print(f"Non-RL converged reference data imported from file '{filename_ref}' - averaging time: {averaging_time_ref:.6f}")
print("\nData imported successfully!")

### RL and non-RL data is taken at different cummulative averaging times, manage corresponding indices for non-available time instants
averaging_time_all = np.unique(np.round(np.concatenate((averaging_time_nonRL, averaging_time_accum_RL)), decimals=1))
idx_nonRL = np.array([np.argmin(np.abs(averaging_time_nonRL - t))    for t in averaging_time_all])
idx_RL    = np.array([np.argmin(np.abs(averaging_time_accum_RL - t)) for t in averaging_time_all])
N_all     = len(averaging_time_all)

# -------------- Averaging fields using XZ symmetries --------------

### Allocate averaged variables
y_plus_RL      = np.zeros( [N_RL, int( 0.5*num_points_y )] ); y_plus_nonRL      = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  y_plus_ref      = np.zeros( int( 0.5*num_points_y ) )
avg_u_plus_RL  = np.zeros( [N_RL, int( 0.5*num_points_y )] ); avg_u_plus_nonRL  = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  avg_u_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
avg_v_plus_RL  = np.zeros( [N_RL, int( 0.5*num_points_y )] ); avg_v_plus_nonRL  = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  avg_v_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
avg_w_plus_RL  = np.zeros( [N_RL, int( 0.5*num_points_y )] ); avg_w_plus_nonRL  = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  avg_w_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
rmsf_u_plus_RL = np.zeros( [N_RL, int( 0.5*num_points_y )] ); rmsf_u_plus_nonRL = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  rmsf_u_plus_ref = np.zeros( int( 0.5*num_points_y ) )
rmsf_v_plus_RL = np.zeros( [N_RL, int( 0.5*num_points_y )] ); rmsf_v_plus_nonRL = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  rmsf_v_plus_ref = np.zeros( int( 0.5*num_points_y ) )
rmsf_w_plus_RL = np.zeros( [N_RL, int( 0.5*num_points_y )] ); rmsf_w_plus_nonRL = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  rmsf_w_plus_ref = np.zeros( int( 0.5*num_points_y ) )

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
            for n in range(N_RL):
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
            for n in range(N_nonRL):
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

### ### Build plots velocity & TKE profiles
### visualizer.build_velocity_profiles(
###     y_plus_ref, y_plus_nonRL, y_plus_RL,
###     avg_u_plus_ref, avg_u_plus_nonRL, avg_u_plus_RL,
###     rmsf_u_plus_rer, rmsf_u_plus_nonRL, rmsf_u_plus_RL,
###     rmsf_v_plus_rer, rmsf_v_plus_nonRL, rmsf_v_plus_RL,
###     rmsf_w_plus_rer, rmsf_w_plus_nonRL, rmsf_w_plus_RL,
###     TKE_ref, TKE_nonRL, TKE_RL,
### )

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
for i in range(N_all):
    # log progress
    if i % (N_all//10 or 1) == 0:
        print(f"{i/N_all*100:.0f}%")
    # Build frames
    i_nonRL = idx_nonRL[i]
    i_RL    = idx_RL[i]
    frames_avg_u  = visualizer.build_vel_avg_frame( frames_avg_u,  y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, avg_u_plus_RL[i_RL],  avg_u_plus_nonRL[i_nonRL],  avg_u_plus_ref,  averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='u', ylim=ylim_avg_u,  x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_avg_v  = visualizer.build_vel_avg_frame( frames_avg_v,  y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, avg_v_plus_RL[i_RL],  avg_v_plus_nonRL[i_nonRL],  avg_v_plus_ref,  averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='v', ylim=ylim_avg_v,  x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_avg_w  = visualizer.build_vel_avg_frame( frames_avg_w,  y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, avg_w_plus_RL[i_RL],  avg_w_plus_nonRL[i_nonRL],  avg_w_plus_ref,  averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='w', ylim=ylim_avg_w,  x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_rmsf_u = visualizer.build_vel_rmsf_frame(frames_rmsf_u, y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, rmsf_u_plus_RL[i_RL], rmsf_u_plus_nonRL[i_nonRL], rmsf_u_plus_ref, averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='u', ylim=ylim_rmsf_u, x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_rmsf_v = visualizer.build_vel_rmsf_frame(frames_rmsf_v, y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, rmsf_v_plus_RL[i_RL], rmsf_v_plus_nonRL[i_nonRL], rmsf_v_plus_ref, averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='v', ylim=ylim_rmsf_v, x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_rmsf_w = visualizer.build_vel_rmsf_frame(frames_rmsf_w, y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, rmsf_w_plus_RL[i_RL], rmsf_w_plus_nonRL[i_nonRL], rmsf_w_plus_ref, averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='w', ylim=ylim_rmsf_w, x_actuator_boundaries=y_plus_actuators_boundaries)

print("\nBuilding gifs from frames for avg_u, avg_v, avg_w, rmsf_u, rmsf_v, rmsf_w...")
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
abs_error_avg_u_plus_RL  = np.zeros( [N_RL, y_plus_ref.size] ); abs_error_avg_u_plus_nonRL  = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_avg_v_plus_RL  = np.zeros( [N_RL, y_plus_ref.size] ); abs_error_avg_v_plus_nonRL  = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_avg_w_plus_RL  = np.zeros( [N_RL, y_plus_ref.size] ); abs_error_avg_w_plus_nonRL  = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_rmsf_u_plus_RL = np.zeros( [N_RL, y_plus_ref.size] ); abs_error_rmsf_u_plus_nonRL = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_rmsf_v_plus_RL = np.zeros( [N_RL, y_plus_ref.size] ); abs_error_rmsf_v_plus_nonRL = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_rmsf_w_plus_RL = np.zeros( [N_RL, y_plus_ref.size] ); abs_error_rmsf_w_plus_nonRL = np.zeros( [N_nonRL, y_plus_ref.size] )
for i in range(N_RL):
    abs_error_avg_u_plus_RL[i,:]     = np.abs( avg_u_plus_RL[i,:]    - avg_u_plus_ref )
    abs_error_avg_v_plus_RL[i,:]     = np.abs( avg_v_plus_RL[i,:]    - avg_v_plus_ref )
    abs_error_avg_w_plus_RL[i,:]     = np.abs( avg_w_plus_RL[i,:]    - avg_w_plus_ref )
    abs_error_rmsf_u_plus_RL[i,:]    = np.abs( rmsf_u_plus_RL[i,:]   - rmsf_u_plus_ref )
    abs_error_rmsf_v_plus_RL[i,:]    = np.abs( rmsf_v_plus_RL[i,:]   - rmsf_v_plus_ref )
    abs_error_rmsf_w_plus_RL[i,:]    = np.abs( rmsf_w_plus_RL[i,:]   - rmsf_w_plus_ref )
for i in range(N_nonRL):
    abs_error_avg_u_plus_nonRL[i,:]  = np.abs( avg_u_plus_nonRL[i,:]  - avg_u_plus_ref )
    abs_error_avg_v_plus_nonRL[i,:]  = np.abs( avg_v_plus_nonRL[i,:]  - avg_v_plus_ref )
    abs_error_avg_w_plus_nonRL[i,:]  = np.abs( avg_w_plus_nonRL[i,:]  - avg_w_plus_ref )
    abs_error_rmsf_u_plus_nonRL[i,:] = np.abs( rmsf_u_plus_nonRL[i,:] - rmsf_u_plus_ref )
    abs_error_rmsf_v_plus_nonRL[i,:] = np.abs( rmsf_v_plus_nonRL[i,:] - rmsf_v_plus_ref )
    abs_error_rmsf_w_plus_nonRL[i,:] = np.abs( rmsf_w_plus_nonRL[i,:] - rmsf_w_plus_ref )

# L1 Error
L1_error_avg_u_plus_RL  = np.zeros(N_RL); L1_error_avg_u_plus_nonRL  = np.zeros(N_nonRL)
L1_error_avg_v_plus_RL  = np.zeros(N_RL); L1_error_avg_v_plus_nonRL  = np.zeros(N_nonRL)
L1_error_avg_w_plus_RL  = np.zeros(N_RL); L1_error_avg_w_plus_nonRL  = np.zeros(N_nonRL)
L1_error_rmsf_u_plus_RL = np.zeros(N_RL); L1_error_rmsf_u_plus_nonRL = np.zeros(N_nonRL)
L1_error_rmsf_v_plus_RL = np.zeros(N_RL); L1_error_rmsf_v_plus_nonRL = np.zeros(N_nonRL)
L1_error_rmsf_w_plus_RL = np.zeros(N_RL); L1_error_rmsf_w_plus_nonRL = np.zeros(N_nonRL)
ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i in range(N_RL):
        L1_error_avg_u_plus_RL[i]     += abs_error_avg_u_plus_RL[i,j]     * dy
        L1_error_avg_v_plus_RL[i]     += abs_error_avg_v_plus_RL[i,j]     * dy
        L1_error_avg_w_plus_RL[i]     += abs_error_avg_w_plus_RL[i,j]     * dy
        L1_error_rmsf_u_plus_RL[i]    += abs_error_rmsf_u_plus_RL[i,j]    * dy
        L1_error_rmsf_v_plus_RL[i]    += abs_error_rmsf_v_plus_RL[i,j]    * dy
        L1_error_rmsf_w_plus_RL[i]    += abs_error_rmsf_w_plus_RL[i,j]    * dy
    for i in range(N_nonRL):
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
L2_error_avg_u_plus_RL  = np.zeros(N_RL); L2_error_avg_u_plus_nonRL  = np.zeros(N_nonRL)
L2_error_avg_v_plus_RL  = np.zeros(N_RL); L2_error_avg_v_plus_nonRL  = np.zeros(N_nonRL)
L2_error_avg_w_plus_RL  = np.zeros(N_RL); L2_error_avg_w_plus_nonRL  = np.zeros(N_nonRL)
L2_error_rmsf_u_plus_RL = np.zeros(N_RL); L2_error_rmsf_u_plus_nonRL = np.zeros(N_nonRL)
L2_error_rmsf_v_plus_RL = np.zeros(N_RL); L2_error_rmsf_v_plus_nonRL = np.zeros(N_nonRL)
L2_error_rmsf_w_plus_RL = np.zeros(N_RL); L2_error_rmsf_w_plus_nonRL = np.zeros(N_nonRL)
ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i in range(N_RL):
        L2_error_avg_u_plus_RL[i]     += ( ( avg_u_plus_RL[i,j]     - avg_u_plus_ref[j] )**2 )  * dy
        L2_error_avg_v_plus_RL[i]     += ( ( avg_v_plus_RL[i,j]     - avg_v_plus_ref[j] )**2 )  * dy
        L2_error_avg_w_plus_RL[i]     += ( ( avg_w_plus_RL[i,j]     - avg_w_plus_ref[j] )**2 )  * dy
        L2_error_rmsf_u_plus_RL[i]    += ( ( rmsf_u_plus_RL[i,j]    - rmsf_u_plus_ref[j] )**2 ) * dy 
        L2_error_rmsf_v_plus_RL[i]    += ( ( rmsf_v_plus_RL[i,j]    - rmsf_v_plus_ref[j] )**2 ) * dy 
        L2_error_rmsf_w_plus_RL[i]    += ( ( rmsf_w_plus_RL[i,j]    - rmsf_w_plus_ref[j] )**2 ) * dy 
    for i in range(N_nonRL):
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
Linf_error_avg_u_plus_RL  = np.zeros(N_RL); Linf_error_avg_u_plus_nonRL  = np.zeros(N_nonRL)
Linf_error_avg_v_plus_RL  = np.zeros(N_RL); Linf_error_avg_v_plus_nonRL  = np.zeros(N_nonRL)
Linf_error_avg_w_plus_RL  = np.zeros(N_RL); Linf_error_avg_w_plus_nonRL  = np.zeros(N_nonRL)
Linf_error_rmsf_u_plus_RL = np.zeros(N_RL); Linf_error_rmsf_u_plus_nonRL = np.zeros(N_nonRL)
Linf_error_rmsf_v_plus_RL = np.zeros(N_RL); Linf_error_rmsf_v_plus_nonRL = np.zeros(N_nonRL)
Linf_error_rmsf_w_plus_RL = np.zeros(N_RL); Linf_error_rmsf_w_plus_nonRL = np.zeros(N_nonRL)
for i in range(N_RL):
    Linf_error_avg_u_plus_RL[i]     = np.max(abs_error_avg_u_plus_RL[i,:])
    Linf_error_avg_v_plus_RL[i]     = np.max(abs_error_avg_v_plus_RL[i,:])
    Linf_error_avg_w_plus_RL[i]     = np.max(abs_error_avg_w_plus_RL[i,:])
    Linf_error_rmsf_u_plus_RL[i]    = np.max(abs_error_rmsf_u_plus_RL[i,:])
    Linf_error_rmsf_v_plus_RL[i]    = np.max(abs_error_rmsf_v_plus_RL[i,:])
    Linf_error_rmsf_w_plus_RL[i]    = np.max(abs_error_rmsf_w_plus_RL[i,:])
for i in range(N_nonRL):
    Linf_error_avg_u_plus_nonRL[i]  = np.max(abs_error_avg_u_plus_nonRL[i,:])
    Linf_error_avg_v_plus_nonRL[i]  = np.max(abs_error_avg_v_plus_nonRL[i,:])
    Linf_error_avg_w_plus_nonRL[i]  = np.max(abs_error_avg_w_plus_nonRL[i,:])
    Linf_error_rmsf_u_plus_nonRL[i] = np.max(abs_error_rmsf_u_plus_nonRL[i,:])
    Linf_error_rmsf_v_plus_nonRL[i] = np.max(abs_error_rmsf_v_plus_nonRL[i,:])
    Linf_error_rmsf_w_plus_nonRL[i] = np.max(abs_error_rmsf_w_plus_nonRL[i,:])
print("Errors calculated successfully!")

# --- Errors logging ---

# Store error logs in file
error_log_filename = f"{postDir}/errors_velocity_statistics.txt"
print(f"\nWriting errors in file '{error_log_filename}'")
with open(error_log_filename, "w") as file:
    # averaging times at which errors are calculated
    file.write("\n\n------------------------------------------------")
    file.write("\nAveraging times:")
    file.write(f"\n\nAveraging time RL: {averaging_time_RL}")
    file.write(f"\nAveraging time accumulated RL: {averaging_time_accum_RL}")
    file.write(f"\nAveraging time non-RL: {averaging_time_nonRL}")
    # avg_u errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_u:")
    file.write(f"\n\nL1 Error avg_u RL: {L1_error_avg_u_plus_RL}")
    file.write(f"\nL1 Error avg_u nonRL: {L1_error_avg_u_plus_nonRL}")
    file.write(f"\n\nL2 Error avg_u RL (RMS): {L2_error_avg_u_plus_RL}")
    file.write(f"\nL2 Error avg_u nonRL (RMS): {L2_error_avg_u_plus_nonRL}")
    file.write(f"\n\nLinf Error avg_u RL: {Linf_error_avg_u_plus_RL}")
    file.write(f"\nLinf Error avg_u nonRL: {Linf_error_avg_u_plus_nonRL}")
    # avg_v errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_v:")
    file.write(f"\n\nL1 Error avg_v RL: {L1_error_avg_v_plus_RL}")
    file.write(f"\nL1 Error avg_v nonRL: {L1_error_avg_v_plus_nonRL}")
    file.write(f"\n\nL2 Error avg_v RL (RMS): {L2_error_avg_v_plus_RL}")
    file.write(f"\nL2 Error avg_v nonRL (RMS): {L2_error_avg_v_plus_nonRL}")
    file.write(f"\n\nLinf Error avg_v RL: {Linf_error_avg_v_plus_RL}")
    file.write(f"\nLinf Error avg_v nonRL: {Linf_error_avg_v_plus_nonRL}")
    # avg_w errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_w:")
    file.write(f"\n\nL1 Error avg_w RL: {L1_error_avg_w_plus_RL}")
    file.write(f"\nL1 Error avg_w nonRL: {L1_error_avg_w_plus_nonRL}")
    file.write(f"\n\nL2 Error avg_w RL (RMS): {L2_error_avg_w_plus_RL}")
    file.write(f"\nL2 Error avg_w nonRL (RMS): {L2_error_avg_w_plus_nonRL}")
    file.write(f"\n\nLinf Error avg_w RL: {Linf_error_avg_w_plus_RL}")
    file.write(f"\nLinf Error avg_w nonRL: {Linf_error_avg_w_plus_nonRL}")
    # rmsf_u errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_u:")
    file.write(f"\n\nL1 Error rmsf_u RL: {L1_error_rmsf_u_plus_RL}")
    file.write(f"\nL1 Error rmsf_u nonRL: {L1_error_rmsf_u_plus_nonRL}")
    file.write(f"\n\nL2 Error rmsf_u RL (RMS): {L2_error_rmsf_u_plus_RL}")
    file.write(f"\nL2 Error rmsf_u nonRL (RMS): {L2_error_rmsf_u_plus_nonRL}")
    file.write(f"\n\nLinf Error rmsf_u RL: {Linf_error_rmsf_u_plus_RL}")
    file.write(f"\nLinf Error rmsf_u nonRL: {Linf_error_rmsf_u_plus_nonRL}")
    # rmsf_v errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_v:")
    file.write(f"\n\nL1 Error rmsf_v RL: {L1_error_rmsf_v_plus_RL}")
    file.write(f"\nL1 Error rmsf_v nonRL: {L1_error_rmsf_v_plus_nonRL}")
    file.write(f"\n\nL2 Error rmsf_v RL (RMS): {L2_error_rmsf_v_plus_RL}")
    file.write(f"\nL2 Error rmsf_v nonRL (RMS): {L2_error_rmsf_v_plus_nonRL}")
    file.write(f"\n\nLinf Error rmsf_v RL: {Linf_error_rmsf_v_plus_RL}")
    file.write(f"\nLinf Error rmsf_v nonRL: {Linf_error_rmsf_v_plus_nonRL}")
    # rmsf_w errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_w:")
    file.write(f"\n\nL1 Error rmsf_w RL: {L1_error_rmsf_w_plus_RL}")
    file.write(f"\nL1 Error rmsf_w nonRL: {L1_error_rmsf_w_plus_nonRL}")
    file.write(f"\n\nL2 Error rmsf_w RL (RMS): {L2_error_rmsf_w_plus_RL}")
    file.write(f"\nL2 Error rmsf_w nonRL (RMS): {L2_error_rmsf_w_plus_nonRL}")
    file.write(f"\n\nLinf Error rmsf_w RL: {Linf_error_rmsf_w_plus_RL}")
    file.write(f"\nLinf Error rmsf_w nonRL: {Linf_error_rmsf_w_plus_nonRL}")
print("Errors written successfully!")

# Print error logs in terminal
with open(error_log_filename, "r") as file:
    content = file.read()
    print(content)

# --- Errors Plots ---
print("\nBuilding error plots...")

# L1-Error plot
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, L1_error_avg_u_plus_nonRL,   L1_error_avg_u_plus_RL,   L1_error_rmsf_u_plus_nonRL,   L1_error_rmsf_u_plus_RL,   vel_component='u', error_num='1')
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, L1_error_avg_v_plus_nonRL,   L1_error_avg_v_plus_RL,   L1_error_rmsf_v_plus_nonRL,   L1_error_rmsf_v_plus_RL,   vel_component='v', error_num='1')
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, L1_error_avg_w_plus_nonRL,   L1_error_avg_w_plus_RL,   L1_error_rmsf_w_plus_nonRL,   L1_error_rmsf_w_plus_RL,   vel_component='w', error_num='1')
# L2-Error plot
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, L2_error_avg_u_plus_nonRL,   L2_error_avg_u_plus_RL,   L2_error_rmsf_u_plus_nonRL,   L2_error_rmsf_u_plus_RL,   vel_component='u', error_num='2')
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, L2_error_avg_v_plus_nonRL,   L2_error_avg_v_plus_RL,   L2_error_rmsf_v_plus_nonRL,   L2_error_rmsf_v_plus_RL,   vel_component='v', error_num='2')
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, L2_error_avg_w_plus_nonRL,   L2_error_avg_w_plus_RL,   L2_error_rmsf_w_plus_nonRL,   L2_error_rmsf_w_plus_RL,   vel_component='w', error_num='2')
# Linf-Error plot
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, Linf_error_avg_u_plus_nonRL, Linf_error_avg_u_plus_RL, Linf_error_rmsf_u_plus_nonRL, Linf_error_rmsf_u_plus_RL, vel_component='u', error_num='inf')
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, Linf_error_avg_v_plus_nonRL, Linf_error_avg_v_plus_RL, Linf_error_rmsf_v_plus_nonRL, Linf_error_rmsf_v_plus_RL, vel_component='v', error_num='inf')
visualizer.build_velocity_error_plot(averaging_time_nonRL, averaging_time_accum_RL, Linf_error_avg_w_plus_nonRL, Linf_error_avg_w_plus_RL, Linf_error_rmsf_w_plus_nonRL, Linf_error_rmsf_w_plus_RL, vel_component='w', error_num='inf')

print("Error plots built successfully!")