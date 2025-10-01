#!/home/jofre/miniconda3/envs/smartrhea-env-v2/bin/python3

import sys
import os
import glob
import numpy as np
import h5py    
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 12, frameon = False)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')
#plt.rc( 'savefig',    format = "jpg", dpi = 600)

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---

try :
    Re_tau          = float(sys.argv[1])     # Friction Reynolds number [-]
except :
    raise ValueError("Missing call arguments, should be: <Re_tau>")

# --- Case parameters ---

rho_0   = 1.0				# Reference density [kg/m3]
u_tau   = 1.0				# Friction velocity [m/s]
delta   = 1.0				# Channel half-height [m]
mu_ref  = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref  = mu_ref/rho_0			# Kinematic viscosity [m2/s]

if Re_tau == 100:
    iteration_min_nonRL = 3240000
    iteration_max_nonRL = 7180000
elif Re_tau == 180:
    iteration_min_nonRL = 2820000 
    iteration_max_nonRL = 3030000
else:
    raise ValueError(f"Invalid Re_tau: {Re_tau}. Must be one of [100, 180].")

# --- Training / Evaluation parameters ---

# Reference & non-RL data directory
filePath = os.path.dirname(os.path.abspath(__file__))
compareDatasetDir = os.path.join("/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/post_process", f"data_Retau{Re_tau:.0f}")

# Directory of build plots
postDir = f"/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/post_process/article_test/"

#--------------------------------------------------------------------------------------------

# ----------- Build data h5 filenames ------------

# --- non-RL converged reference filename ---
filename_ref = f"{compareDatasetDir}/3d_turbulent_channel_flow_reference.h5"

# --- non-RL restart data file
# filename_rst = f"{compareDatasetDir}/3d_turbulent_channel_flow_{iteration_restart_data_file}.h5"

# --- non-RL filenames ---

# Get filepath and file details of the last saved iteration of each global step
pattern = f"{compareDatasetDir}/3d_turbulent_channel_flow_*0000.h5"
matching_files = sorted(glob.glob(pattern))
all_files = []
if matching_files:
    print("\nNon-RL files:")
    for filepath in matching_files:
        filename       = os.path.basename(filepath)
        parts_filename = filename.split('_')
        # Extract iteration
        try:
            iter_num    = int(os.path.splitext(parts_filename[-1])[0])
            if (iter_num >= iteration_min_nonRL and iter_num <= iteration_max_nonRL):
                all_files.append((iter_num, filepath))
        except (ValueError):
            print(f"Skipping file with no number iteration: {filename}, in filepath: {filepath}")

    # Sort numerically by iteration number
    all_files.sort(key=lambda x: x[0])
    # Unpack
    iteration_nonRL_list = [iter_num for iter_num, _ in all_files]
    filename_nonRL_list  = [filepath for _, filepath in all_files]

    # Append restart data file to non-RL files list
    ###iteration_nonRL_list.insert(0,iteration_restart_data_file) 
    ###global_step_nonRL_list.insert(0,'restart file') 
    ###filename_nonRL_list.insert(0,filename_rst)
    
    N_nonRL = len(filename_nonRL_list)

    # Print selected files
    for i in range(N_nonRL):
        print(f"\nFilename: {filename}, \nIteration: {iter_num}")

#--------------------------------------------------------------------------------------------

# ----------- Get non-RL data ------------

# --- Check if non-RL files exists ---

print("\nCheking files...")

for filename_nonRL in filename_nonRL_list:
    if not os.path.isfile(filename_nonRL):
        print(f"Error: File '{filename_nonRL}' not found.")
        sys.exit(1)
if not os.path.isfile(filename_ref):
    print(f"Error: File '{filename_ref}' not found.")
    sys.exit(1)

# --- Get data from 3d-snapshots h5 files ---

print("\nImporting data from files...")

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

# -------------- Averaging fields using XZ symmetries --------------

### Allocate averaged variables
y_plus_nonRL      = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  y_plus_ref      = np.zeros( int( 0.5*num_points_y ) )
avg_u_plus_nonRL  = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  avg_u_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
avg_v_plus_nonRL  = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  avg_v_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
avg_w_plus_nonRL  = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  avg_w_plus_ref  = np.zeros( int( 0.5*num_points_y ) )
rmsf_u_plus_nonRL = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  rmsf_u_plus_ref = np.zeros( int( 0.5*num_points_y ) )
rmsf_v_plus_nonRL = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  rmsf_v_plus_ref = np.zeros( int( 0.5*num_points_y ) )
rmsf_w_plus_nonRL = np.zeros( [N_nonRL, int( 0.5*num_points_y )] );  rmsf_w_plus_ref = np.zeros( int( 0.5*num_points_y ) )

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
            if( j > ( int( 0.5*num_points_y ) - 1 ) ):      # top-wall idx
                aux_j = num_points_y - j - 1

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
TKE_nonRL = 0.5 * ( rmsf_u_plus_nonRL**2 + rmsf_v_plus_nonRL**2 + rmsf_w_plus_nonRL**2 )
TKE_ref   = 0.5 * ( rmsf_u_plus_ref**2   + rmsf_v_plus_ref**2   + rmsf_w_plus_ref**2   )

# ----------------- RL & non-RL Errors w.r.t. Reference -----------------

if np.allclose(y_plus_nonRL, y_plus_ref):
    print("\nNo need for interpolating data, as y_plus coordinates are the same for all data")
else:
    print("\nnon-RL y+:", y_plus_nonRL)
    print("\nRef y+:", y_plus_ref)
    raise Exception("y-plus coordinates should be equal for all non-RL & reference data, as no interpolation will be done.")
ny = y_plus_ref.size
y_plus = np.concatenate([[0.0],y_plus_ref,[delta]])

# --- Calculate errors using spatial interpolation ---

print("\nCalculating L1, L2, Linf errors...")
# Absolute error 
abs_error_avg_u_plus_nonRL  = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_avg_v_plus_nonRL  = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_avg_w_plus_nonRL  = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_rmsf_u_plus_nonRL = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_rmsf_v_plus_nonRL = np.zeros( [N_nonRL, y_plus_ref.size] )
abs_error_rmsf_w_plus_nonRL = np.zeros( [N_nonRL, y_plus_ref.size] )
for i in range(N_nonRL):
    abs_error_avg_u_plus_nonRL[i,:]  = np.abs( avg_u_plus_nonRL[i,:]  - avg_u_plus_ref )
    abs_error_avg_v_plus_nonRL[i,:]  = np.abs( avg_v_plus_nonRL[i,:]  - avg_v_plus_ref )
    abs_error_avg_w_plus_nonRL[i,:]  = np.abs( avg_w_plus_nonRL[i,:]  - avg_w_plus_ref )
    abs_error_rmsf_u_plus_nonRL[i,:] = np.abs( rmsf_u_plus_nonRL[i,:] - rmsf_u_plus_ref )
    abs_error_rmsf_v_plus_nonRL[i,:] = np.abs( rmsf_v_plus_nonRL[i,:] - rmsf_v_plus_ref )
    abs_error_rmsf_w_plus_nonRL[i,:] = np.abs( rmsf_w_plus_nonRL[i,:] - rmsf_w_plus_ref )

# L1 Error
L1_error_avg_u_plus_nonRL  = np.zeros(N_nonRL)
L1_error_avg_v_plus_nonRL  = np.zeros(N_nonRL)
L1_error_avg_w_plus_nonRL  = np.zeros(N_nonRL)
L1_error_rmsf_u_plus_nonRL = np.zeros(N_nonRL)
L1_error_rmsf_v_plus_nonRL = np.zeros(N_nonRL)
L1_error_rmsf_w_plus_nonRL = np.zeros(N_nonRL)
ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i in range(N_nonRL):
        L1_error_avg_u_plus_nonRL[i]  += abs_error_avg_u_plus_nonRL[i,j]  * dy
        L1_error_avg_v_plus_nonRL[i]  += abs_error_avg_v_plus_nonRL[i,j]  * dy
        L1_error_avg_w_plus_nonRL[i]  += abs_error_avg_w_plus_nonRL[i,j]  * dy
        L1_error_rmsf_u_plus_nonRL[i] += abs_error_rmsf_u_plus_nonRL[i,j] * dy
        L1_error_rmsf_v_plus_nonRL[i] += abs_error_rmsf_v_plus_nonRL[i,j] * dy
        L1_error_rmsf_w_plus_nonRL[i] += abs_error_rmsf_w_plus_nonRL[i,j] * dy
L1_error_avg_u_plus_nonRL  /= ylength         
L1_error_avg_v_plus_nonRL  /= ylength         
L1_error_avg_w_plus_nonRL  /= ylength         
L1_error_rmsf_u_plus_nonRL /= ylength         
L1_error_rmsf_v_plus_nonRL /= ylength         
L1_error_rmsf_w_plus_nonRL /= ylength         

# L2 Error (RMS Error)
L2_error_avg_u_plus_nonRL  = np.zeros(N_nonRL)
L2_error_avg_v_plus_nonRL  = np.zeros(N_nonRL)
L2_error_avg_w_plus_nonRL  = np.zeros(N_nonRL)
L2_error_rmsf_u_plus_nonRL = np.zeros(N_nonRL)
L2_error_rmsf_v_plus_nonRL = np.zeros(N_nonRL)
L2_error_rmsf_w_plus_nonRL = np.zeros(N_nonRL)
ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i in range(N_nonRL):
        L2_error_avg_u_plus_nonRL[i]  += ( ( avg_u_plus_nonRL[i,j]  - avg_u_plus_ref[j])**2 )   * dy
        L2_error_avg_v_plus_nonRL[i]  += ( ( avg_v_plus_nonRL[i,j]  - avg_v_plus_ref[j])**2 )   * dy
        L2_error_avg_w_plus_nonRL[i]  += ( ( avg_w_plus_nonRL[i,j]  - avg_w_plus_ref[j])**2 )   * dy
        L2_error_rmsf_u_plus_nonRL[i] += ( ( rmsf_u_plus_nonRL[i,j] - rmsf_u_plus_ref[j])**2 )  * dy
        L2_error_rmsf_v_plus_nonRL[i] += ( ( rmsf_v_plus_nonRL[i,j] - rmsf_v_plus_ref[j])**2 )  * dy
        L2_error_rmsf_w_plus_nonRL[i] += ( ( rmsf_w_plus_nonRL[i,j] - rmsf_w_plus_ref[j])**2 )  * dy  
L2_error_avg_u_plus_nonRL  = np.sqrt( L2_error_avg_u_plus_nonRL / ylength )          
L2_error_avg_v_plus_nonRL  = np.sqrt( L2_error_avg_v_plus_nonRL / ylength )          
L2_error_avg_w_plus_nonRL  = np.sqrt( L2_error_avg_w_plus_nonRL / ylength )          
L2_error_rmsf_u_plus_nonRL = np.sqrt( L2_error_rmsf_u_plus_nonRL / ylength )          
L2_error_rmsf_v_plus_nonRL = np.sqrt( L2_error_rmsf_v_plus_nonRL / ylength )          
L2_error_rmsf_w_plus_nonRL = np.sqrt( L2_error_rmsf_w_plus_nonRL / ylength )          

# Linf Error
Linf_error_avg_u_plus_nonRL  = np.zeros(N_nonRL)
Linf_error_avg_v_plus_nonRL  = np.zeros(N_nonRL)
Linf_error_avg_w_plus_nonRL  = np.zeros(N_nonRL)
Linf_error_rmsf_u_plus_nonRL = np.zeros(N_nonRL)
Linf_error_rmsf_v_plus_nonRL = np.zeros(N_nonRL)
Linf_error_rmsf_w_plus_nonRL = np.zeros(N_nonRL)
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
error_log_filename = f"{postDir}/errors_velocity_statistics_nonRL.txt"
print(f"\nWriting errors in file '{error_log_filename}'")
with open(error_log_filename, "w") as file:
    # averaging times at which errors are calculated
    file.write("\n\n------------------------------------------------")
    file.write("\nAveraging times:")
    file.write(f"\nAveraging time non-RL: {averaging_time_nonRL}")
    # avg_u errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_u:")
    file.write(f"\nL1 Error avg_u nonRL: {L1_error_avg_u_plus_nonRL}")
    file.write(f"\nL2 Error avg_u nonRL (RMS): {L2_error_avg_u_plus_nonRL}")
    file.write(f"\nLinf Error avg_u nonRL: {Linf_error_avg_u_plus_nonRL}")
    # avg_v errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_v:")
    file.write(f"\nL1 Error avg_v nonRL: {L1_error_avg_v_plus_nonRL}")
    file.write(f"\nL2 Error avg_v nonRL (RMS): {L2_error_avg_v_plus_nonRL}")
    file.write(f"\nLinf Error avg_v nonRL: {Linf_error_avg_v_plus_nonRL}")
    # avg_w errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of avg_w:")
    file.write(f"\nL1 Error avg_w nonRL: {L1_error_avg_w_plus_nonRL}")
    file.write(f"\nL2 Error avg_w nonRL (RMS): {L2_error_avg_w_plus_nonRL}")
    file.write(f"\nLinf Error avg_w nonRL: {Linf_error_avg_w_plus_nonRL}")
    # rmsf_u errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_u:")
    file.write(f"\nL1 Error rmsf_u nonRL: {L1_error_rmsf_u_plus_nonRL}")
    file.write(f"\nL2 Error rmsf_u nonRL (RMS): {L2_error_rmsf_u_plus_nonRL}")
    file.write(f"\nLinf Error rmsf_u nonRL: {Linf_error_rmsf_u_plus_nonRL}")
    # rmsf_v errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_v:")
    file.write(f"\nL1 Error rmsf_v nonRL: {L1_error_rmsf_v_plus_nonRL}")
    file.write(f"\nL2 Error rmsf_v nonRL (RMS): {L2_error_rmsf_v_plus_nonRL}")
    file.write(f"\nLinf Error rmsf_v nonRL: {Linf_error_rmsf_v_plus_nonRL}")
    # rmsf_w errors:
    file.write("\n\n------------------------------------------------")
    file.write("\nConvergence errors of rmsf_w:")
    file.write(f"\nL1 Error rmsf_w nonRL: {L1_error_rmsf_w_plus_nonRL}")
    file.write(f"\nL2 Error rmsf_w nonRL (RMS): {L2_error_rmsf_w_plus_nonRL}")
    file.write(f"\nLinf Error rmsf_w nonRL: {Linf_error_rmsf_w_plus_nonRL}")
print("Errors written successfully!")

# Print error logs in terminal
with open(error_log_filename, "r") as file:
    content = file.read()
    print(content)

# --- Errors Plots ---
 
def build_velocity_error_nonRL_plot(avg_time_nonRL, err_avg_u_nonRL, err_avg_v_nonRL, err_avg_w_nonRL, 
                                                    err_rmsf_u_nonRL, err_rmsf_v_nonRL, err_rmsf_w_nonRL, error_num='2'):
    plt.clf()
    plt.semilogy( avg_time_nonRL, err_avg_u_nonRL,  linestyle = '-', linewidth = 1, color = plt.cm.tab10(0), zorder = 0, label = r'$\overline{u}^+$' )
    plt.semilogy( avg_time_nonRL, err_avg_v_nonRL,  linestyle = '-', linewidth = 1, color = plt.cm.tab10(1), zorder = 0, label = r'$\overline{v}^+$' )
    plt.semilogy( avg_time_nonRL, err_avg_w_nonRL,  linestyle = '-', linewidth = 1, color = plt.cm.tab10(2), zorder = 0, label = r'$\overline{w}^+$' )
    plt.semilogy( avg_time_nonRL, err_rmsf_u_nonRL, linestyle = ':', linewidth = 1, color = plt.cm.tab10(0), zorder = 0, label = r'${u}_{\textrm{rms}}^+$' )
    plt.semilogy( avg_time_nonRL, err_rmsf_v_nonRL, linestyle = ':', linewidth = 1, color = plt.cm.tab10(1), zorder = 0, label = r'${v}_{\textrm{rms}}^+$' )
    plt.semilogy( avg_time_nonRL, err_rmsf_w_nonRL, linestyle = ':', linewidth = 1, color = plt.cm.tab10(2), zorder = 0, label = r'${w}_{\textrm{rms}}^+$' )
    plt.xlabel(r'$\textrm{Averaging time } t_{avg}^+$' )
    if error_num == "2":
        plt.ylabel(r'$\textrm{NRMSE }(\overline{u}_i^{+}, u_{i_{\textrm{rms}}}^{+})$')
    else:
        plt.ylabel(rf'$L_{{{error_num}}}$ Error' )
    plt.grid(which='both',axis='y')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # (loc='upper right', frameon=True, framealpha=1.0, fancybox=True)
    plt.legend(loc='upper right', frameon=True)
    plt.tick_params( axis = 'both', pad = 7.5 )
    plt.tight_layout()
    filename = os.path.join(postDir, f'L{error_num}_error_nonRL_Retau{Re_tau:.0f}.svg')
    plt.savefig( filename, format = 'svg', bbox_inches = 'tight' )
    plt.close()
    print(f"\nBuild plot: '{filename}'")

print("\nBuilding error plots...")

# L1-Error plot
build_velocity_error_nonRL_plot(averaging_time_nonRL, L1_error_avg_u_plus_nonRL,  L1_error_avg_v_plus_nonRL,  L1_error_avg_w_plus_nonRL, 
                                                      L1_error_rmsf_u_plus_nonRL, L1_error_rmsf_v_plus_nonRL, L1_error_rmsf_w_plus_nonRL,
                                                      error_num='1')
# L2-Error plot
build_velocity_error_nonRL_plot(averaging_time_nonRL, L2_error_avg_u_plus_nonRL,  L2_error_avg_v_plus_nonRL,  L2_error_avg_w_plus_nonRL, 
                                                      L2_error_rmsf_u_plus_nonRL, L2_error_rmsf_v_plus_nonRL, L2_error_rmsf_w_plus_nonRL,
                                                      error_num='2')
# Linf-Error plot
build_velocity_error_nonRL_plot(averaging_time_nonRL, Linf_error_avg_u_plus_nonRL,  Linf_error_avg_v_plus_nonRL,  Linf_error_avg_w_plus_nonRL, 
                                                      Linf_error_rmsf_u_plus_nonRL, Linf_error_rmsf_v_plus_nonRL, Linf_error_rmsf_w_plus_nonRL,
                                                      error_num='inf')
print("Error plots built successfully!")
# 
# # ----------------- Plot Animation Frames of um, urmsf, Rij dof for increasing RL global step (specific iteration & ensemble) -----------------
# 
# print("\nBuilding gif frames for u-avg and u,v,w-rmsf profiles...")
# 
# frames_avg_u = []; frames_avg_v = []; frames_avg_w = []; frames_rmsf_u = []; frames_rmsf_v = []; frames_rmsf_w = []
# # avg velocities limits
# avg_v_abs_max = np.max([np.max(np.abs(avg_v_plus_RL)), np.max(np.abs(avg_v_plus_nonRL)), np.max(np.abs(avg_v_plus_ref))])
# avg_w_abs_max = np.max([np.max(np.abs(avg_w_plus_RL)), np.max(np.abs(avg_w_plus_nonRL)), np.max(np.abs(avg_w_plus_ref))])
# avg_u_min     = 0.0
# avg_v_min     = - avg_v_abs_max
# avg_w_min     = - avg_w_abs_max
# avg_u_max     = int(np.max([np.max(avg_u_plus_RL), np.max(avg_u_plus_nonRL), np.max(avg_u_plus_ref)]))+1
# avg_v_max     = avg_v_abs_max
# avg_w_max     = avg_w_abs_max
# ylim_avg_u    = [avg_u_min, avg_u_max]
# ylim_avg_v    = [avg_v_min, avg_v_max]
# ylim_avg_w    = [avg_w_min, avg_w_max]
# # rmsf velocities limits
# rmsf_u_min   = 0.0
# rmsf_v_min   = 0.0
# rmsf_w_min   = 0.0
# rmsf_u_max   = int(np.max([np.max(rmsf_u_plus_RL), np.max(rmsf_u_plus_nonRL), np.max(rmsf_u_plus_ref)]))+1
# rmsf_v_max   = int(np.max([np.max(rmsf_v_plus_RL), np.max(rmsf_v_plus_nonRL), np.max(rmsf_v_plus_ref)]))+1
# rmsf_w_max   = int(np.max([np.max(rmsf_w_plus_RL), np.max(rmsf_w_plus_nonRL), np.max(rmsf_w_plus_ref)]))+1
# ylim_rmsf_u  = [rmsf_u_min, rmsf_u_max]
# ylim_rmsf_v  = [rmsf_v_min, rmsf_v_max]
# ylim_rmsf_w  = [rmsf_w_min, rmsf_w_max]
# print("Gifs y-limits:", ylim_avg_u, ylim_avg_v, ylim_avg_w, ylim_rmsf_u, ylim_rmsf_v, ylim_rmsf_w)
# for i in range(N_all):
#     # log progress
#     if i % (N_all//10 or 1) == 0:
#         print(f"{i/N_all*100:.0f}%")
#     # Build frames
#     i_nonRL = idx_nonRL[i]
#     i_RL    = idx_RL[i]
#     frames_avg_u  = visualizer.build_vel_avg_frame( frames_avg_u,  y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, avg_u_plus_RL[i_RL],  avg_u_plus_nonRL[i_nonRL],  avg_u_plus_ref,  averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='u', ylim=ylim_avg_u,  x_actuator_boundaries=y_plus_actuators_boundaries)
#     frames_avg_v  = visualizer.build_vel_avg_frame( frames_avg_v,  y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, avg_v_plus_RL[i_RL],  avg_v_plus_nonRL[i_nonRL],  avg_v_plus_ref,  averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='v', ylim=ylim_avg_v,  x_actuator_boundaries=y_plus_actuators_boundaries)
#     frames_avg_w  = visualizer.build_vel_avg_frame( frames_avg_w,  y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, avg_w_plus_RL[i_RL],  avg_w_plus_nonRL[i_nonRL],  avg_w_plus_ref,  averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='w', ylim=ylim_avg_w,  x_actuator_boundaries=y_plus_actuators_boundaries)
#     frames_rmsf_u = visualizer.build_vel_rmsf_frame(frames_rmsf_u, y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, rmsf_u_plus_RL[i_RL], rmsf_u_plus_nonRL[i_nonRL], rmsf_u_plus_ref, averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='u', ylim=ylim_rmsf_u, x_actuator_boundaries=y_plus_actuators_boundaries)
#     frames_rmsf_v = visualizer.build_vel_rmsf_frame(frames_rmsf_v, y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, rmsf_v_plus_RL[i_RL], rmsf_v_plus_nonRL[i_nonRL], rmsf_v_plus_ref, averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='v', ylim=ylim_rmsf_v, x_actuator_boundaries=y_plus_actuators_boundaries)
#     frames_rmsf_w = visualizer.build_vel_rmsf_frame(frames_rmsf_w, y_plus_RL[i_RL], y_plus_nonRL[i_nonRL], y_plus_ref, rmsf_w_plus_RL[i_RL], rmsf_w_plus_nonRL[i_nonRL], rmsf_w_plus_ref, averaging_time_accum_RL[i_RL], averaging_time_nonRL[i_nonRL], global_step_RL_list[i_RL], vel_name='w', ylim=ylim_rmsf_w, x_actuator_boundaries=y_plus_actuators_boundaries)
# 
# print("\nBuilding gifs from frames for avg_u, avg_v, avg_w, rmsf_u, rmsf_v, rmsf_w...")
# frames_dict = {'avg_u':frames_avg_u, 'avg_v':frames_avg_v, 'avg_w':frames_avg_w, 'rmsf_u': frames_rmsf_u, 'rmsf_v':frames_rmsf_v, 'rmsf_w':frames_rmsf_w}
# visualizer.build_main_gifs_from_frames(frames_dict)
# print("Gifs plotted successfully!")
