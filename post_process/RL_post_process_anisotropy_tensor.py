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
    print(f"Script parameters: \n- Iteration: {iteration} \n- Ensemble: {ensemble}\n- Train name: {train_name} \n- Re_tau: {Re_tau} \n- dt_phys: {dt_phys}\n- Case directory: {case_dir}")
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

# --- non-RL (not-converged) filename ---
filename_nonRL = f"{case_dir}/post_process/reference_data/3d_turbulent_channel_flow_{iteration}.h5"

# --- non-RL converged reference filename ---
filename_ref = f"{case_dir}/post_process/reference_data/3d_turbulent_channel_flow_reference.h5"

# ----------- Get RL and non-RL data ------------

# --- Check if RL & non-RL files exists ---
for filename_RL in filename_RL_list:
    if not os.path.isfile(filename_RL):
        print(f"Error: File '{filename_RL}' not found.")
        sys.exit(1)
if not os.path.isfile(filename_nonRL):
    print(f"Error: File '{filename_nonRL}' not found.")
    sys.exit(1)

# --- Get RL (non-converged) data from h5 files ---
print("\nImporting RL data from h5 files...")
n_RL = len(filename_RL_list)
for i_RL in range(n_RL):
    filename_RL = filename_RL_list[i_RL]
    with h5py.File( filename_RL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_RL_aux    = data_file.attrs["AveragingTime"][0] - dt_phys
        y_data_RL_aux            = data_file['y'][1:-1,1:-1,1:-1]
        avg_u_data_RL_aux        = data_file['avg_u'][1:-1,1:-1,1:-1]
        rmsf_u_data_RL_aux       = data_file['rmsf_u'][1:-1,1:-1,1:-1]
        favre_uffuff_data_RL_aux = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
        favre_uffvff_data_RL_aux = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
        favre_uffwff_data_RL_aux = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
        favre_vffvff_data_RL_aux = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
        favre_vffwff_data_RL_aux = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
        favre_wffwff_data_RL_aux = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
    # Initialize allocation arrays
    if i_RL == 0:
        num_points_x      = avg_u_data_RL_aux[0,0,:].size
        num_points_y      = avg_u_data_RL_aux[0,:,0].size
        num_points_z      = avg_u_data_RL_aux[:,0,0].size
        num_points_xz     = num_points_x*num_points_z
        num_points_y_half = int(0.5*num_points_y)
        averaging_time_RL = averaging_time_RL_aux
        y_data_RL            = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_u_data_RL        = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        rmsf_u_data_RL       = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        favre_uffuff_data_RL = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        favre_uffvff_data_RL = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        favre_uffwff_data_RL = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        favre_vffvff_data_RL = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])            
        favre_vffwff_data_RL = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])            
        favre_wffwff_data_RL = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])  
        print(f"RL non-converged data imported from file '{filename_RL}' - averaging time: {averaging_time_RL:.6f}")
    # Fill allocation arrays
    y_data_RL[i_RL,:,:,:]            = y_data_RL_aux
    avg_u_data_RL[i_RL,:,:,:]        = avg_u_data_RL_aux
    rmsf_u_data_RL[i_RL,:,:,:]       = rmsf_u_data_RL_aux
    favre_uffuff_data_RL[i_RL,:,:,:] = favre_uffuff_data_RL_aux
    favre_uffvff_data_RL[i_RL,:,:,:] = favre_uffvff_data_RL_aux
    favre_uffwff_data_RL[i_RL,:,:,:] = favre_uffwff_data_RL_aux
    favre_vffvff_data_RL[i_RL,:,:,:] = favre_vffvff_data_RL_aux
    favre_vffwff_data_RL[i_RL,:,:,:] = favre_vffwff_data_RL_aux
    favre_wffwff_data_RL[i_RL,:,:,:] = favre_wffwff_data_RL_aux
    ### # Check same averaging time
    ### if not np.isclose(averaging_time_RL, averaging_time_RL_aux, atol=1e-8):
    ###     raise ValueError("Averaging time should be equal for all RL h5 files")

# --- Get non-RL (non-converged) data from h5 file ---
print(f"\nImporting non-RL non-converged data from file '{filename_nonRL}'...")
with h5py.File( filename_nonRL, 'r' ) as data_file:
    averaging_time_nonRL    = data_file.attrs["AveragingTime"][0]
    y_data_nonRL            = data_file['y'][1:-1,1:-1,1:-1]
    avg_u_data_nonRL        = data_file['avg_u'][1:-1,1:-1,1:-1]
    rmsf_u_data_nonRL       = data_file['rmsf_u'][1:-1,1:-1,1:-1]
    favre_uffuff_data_nonRL = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
    favre_uffvff_data_nonRL = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
    favre_uffwff_data_nonRL = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
    favre_vffvff_data_nonRL = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
    favre_vffwff_data_nonRL = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
    favre_wffwff_data_nonRL = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
print(f"Non-RL non-converged data imported from file '{filename_nonRL}' - averaging time: {averaging_time_nonRL:.6f}")
### # Check same averaging time for RL & non-RL
### if not np.isclose(averaging_time_RL, averaging_time_nonRL, atol=1e-8):
###     raise ValueError(f"Averaging time should be equal for both RL & non-RL h5 files, while RL: {averaging_time_RL} != nonRL: {averaging_time_nonRL}")
### else:
averaging_time_nonConv = averaging_time_RL

# --- Get non-RL converged reference data from h5 file ---
print(f"\nImporting non-RL converged reference data from file '{filename_ref}'...")
with h5py.File( filename_ref, 'r' ) as data_file:
    averaging_time_ref    = data_file.attrs["AveragingTime"][0]
    y_data_ref            = data_file['y'][1:-1,1:-1,1:-1]
    avg_u_data_ref        = data_file['avg_u'][1:-1,1:-1,1:-1]
    rmsf_u_data_ref       = data_file['rmsf_u'][1:-1,1:-1,1:-1]
    favre_uffuff_data_ref = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
    favre_uffvff_data_ref = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
    favre_uffwff_data_ref = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
    favre_vffvff_data_ref = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
    favre_vffwff_data_ref = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
    favre_wffwff_data_ref = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
assert averaging_time_ref > averaging_time_nonConv, f"Reference data averaging time {averaging_time_ref:.6f} must be greater than non-converged averaging time {averaging_time_nonConv:.6f}"
print(f"Non-RL reference data imported from file '{filename_ref}' - averaging time: {averaging_time_ref:.6f}")
print("Data imported successfully!")

# -------------- Averaging fields using XZ symmetries --------------

print("\nAveraging fields in space...")

### Allocate averaged variables
y_plus_RL       = np.zeros([n_RL, num_points_y_half]);   y_plus_nonRL       = np.zeros(num_points_y_half);   y_plus_ref       = np.zeros(num_points_y_half)
y_delta_RL      = np.zeros([n_RL, num_points_y_half]);   y_delta_nonRL      = np.zeros(num_points_y_half);   y_delta_ref      = np.zeros(num_points_y_half)
avg_u_RL        = np.zeros([n_RL, num_points_y_half]);   avg_u_nonRL        = np.zeros(num_points_y_half);   avg_u_ref        = np.zeros(num_points_y_half)
rmsf_u_RL       = np.zeros([n_RL, num_points_y_half]);   rmsf_u_nonRL       = np.zeros(num_points_y_half);   rmsf_u_ref       = np.zeros(num_points_y_half)
favre_uffuff_RL = np.zeros([n_RL, num_points_y_half]);   favre_uffuff_nonRL = np.zeros(num_points_y_half);   favre_uffuff_ref = np.zeros(num_points_y_half)
favre_uffvff_RL = np.zeros([n_RL, num_points_y_half]);   favre_uffvff_nonRL = np.zeros(num_points_y_half);   favre_uffvff_ref = np.zeros(num_points_y_half)
favre_uffwff_RL = np.zeros([n_RL, num_points_y_half]);   favre_uffwff_nonRL = np.zeros(num_points_y_half);   favre_uffwff_ref = np.zeros(num_points_y_half)
favre_vffvff_RL = np.zeros([n_RL, num_points_y_half]);   favre_vffvff_nonRL = np.zeros(num_points_y_half);   favre_vffvff_ref = np.zeros(num_points_y_half)
favre_vffwff_RL = np.zeros([n_RL, num_points_y_half]);   favre_vffwff_nonRL = np.zeros(num_points_y_half);   favre_vffwff_ref = np.zeros(num_points_y_half)
favre_wffwff_RL = np.zeros([n_RL, num_points_y_half]);   favre_wffwff_nonRL = np.zeros(num_points_y_half);   favre_wffwff_ref = np.zeros(num_points_y_half)

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
            # RL data:
            for i_RL in range(n_RL):
                if is_half_top:
                    y_plus_RL[i_RL,aux_j]   += ( 0.5/num_points_xz )*( 2*delta - y_data_RL[i_RL,k,j,i] )*( u_tau/nu_ref )
                    y_delta_RL[i_RL,aux_j]  += ( 0.5/num_points_xz )*( 2*delta - y_data_RL[i_RL,k,j,i] )/delta
                else:
                    y_plus_RL[i_RL,aux_j]   += ( 0.5/num_points_xz )*y_data_RL[i_RL,k,j,i]*( u_tau/nu_ref )
                    y_delta_RL[i_RL,aux_j]  += ( 0.5/num_points_xz )*y_data_RL[i_RL,k,j,i]/delta
                avg_u_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*avg_u_data_RL[i_RL,k,j,i]
                rmsf_u_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_RL[i_RL,k,j,i]
                favre_uffuff_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_RL[i_RL,k,j,i]
                favre_uffvff_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_RL[i_RL,k,j,i]
                favre_uffwff_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_RL[i_RL,k,j,i]
                favre_vffvff_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_RL[i_RL,k,j,i]
                favre_vffwff_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*favre_vffwff_data_RL[i_RL,k,j,i]
                favre_wffwff_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_RL[i_RL,k,j,i]
            # non-RL data, non-converged & reference:
            if is_half_top:
                y_plus_nonRL[aux_j]   += ( 0.5/num_points_xz )*(2.0*delta - y_data_nonRL[k,j,i])*(u_tau/nu_ref);    y_plus_ref[aux_j]  += ( 0.5/num_points_xz )*(2.0*delta - y_data_ref[k,j,i])*(u_tau/nu_ref);  
                y_delta_nonRL[aux_j]  += ( 0.5/num_points_xz )*(2.0*delta - y_data_nonRL[k,j,i])/delta;             y_delta_ref[aux_j] += ( 0.5/num_points_xz )*(2.0*delta - y_data_ref[k,j,i])/delta;             
            else:
                y_plus_nonRL[aux_j]   += ( 0.5/num_points_xz )*y_data_nonRL[k,j,i]*( u_tau/nu_ref );    y_plus_ref[aux_j]       += ( 0.5/num_points_xz )*y_data_ref[k,j,i]*(u_tau/nu_ref)
                y_delta_nonRL[aux_j]  += ( 0.5/num_points_xz )*y_data_nonRL[k,j,i]/delta;               y_delta_ref[aux_j]      += ( 0.5/num_points_xz )*y_data_ref[k,j,i]/delta
            avg_u_nonRL[aux_j]        += ( 0.5/num_points_xz )*avg_u_data_nonRL[k,j,i];                 avg_u_ref[aux_j]        += ( 0.5/num_points_xz )*avg_u_data_ref[k,j,i]
            rmsf_u_nonRL[aux_j]       += ( 0.5/num_points_xz )*rmsf_u_data_nonRL[k,j,i];                rmsf_u_ref[aux_j]       += ( 0.5/num_points_xz )*rmsf_u_data_ref[k,j,i]
            favre_uffuff_nonRL[aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_nonRL[k,j,i];          favre_uffuff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_ref[k,j,i]
            favre_uffvff_nonRL[aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_nonRL[k,j,i];          favre_uffvff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_ref[k,j,i]
            favre_uffwff_nonRL[aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_nonRL[k,j,i];          favre_uffwff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_ref[k,j,i]
            favre_vffvff_nonRL[aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_nonRL[k,j,i];          favre_vffvff_ref[aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_ref[k,j,i]
            favre_vffwff_nonRL[aux_j] += ( 0.5/num_points_xz )*favre_vffwff_data_nonRL[k,j,i];          favre_vffwff_ref[aux_j] += ( 0.5/num_points_xz )*favre_vffwff_data_ref[k,j,i]
            favre_wffwff_nonRL[aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_nonRL[k,j,i];          favre_wffwff_ref[aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_ref[k,j,i]
# Reduce avg_y_plus_RL along n_RL dimension (idem!)
for i in range(1,n_RL):
    if not np.allclose(y_plus_RL[0], y_plus_RL[i], atol=1e-6) or not np.allclose(y_delta_RL[0], y_delta_RL[i], atol=1e-6):
        raise ValueError("y-plus values of different RL h5 files should be equal!")
y_plus_RL  = np.mean(y_plus_RL,  axis=0)
y_delta_RL = np.mean(y_delta_RL, axis=0)
print("Fields averaged successfully!")


# ----------- Decompose Rij into d.o.f --------------

print("Decomposing Rij into Rij dof...")
Rkk_RL     = np.zeros([n_RL, num_points_y_half]);    Rkk_nonRL     = np.zeros(num_points_y_half);      Rkk_ref     = np.zeros(num_points_y_half)
lambda1_RL = np.zeros([n_RL, num_points_y_half]);    lambda1_nonRL = np.zeros(num_points_y_half);      lambda1_ref = np.zeros(num_points_y_half)
lambda2_RL = np.zeros([n_RL, num_points_y_half]);    lambda2_nonRL = np.zeros(num_points_y_half);      lambda2_ref = np.zeros(num_points_y_half)
lambda3_RL = np.zeros([n_RL, num_points_y_half]);    lambda3_nonRL = np.zeros(num_points_y_half);      lambda3_ref = np.zeros(num_points_y_half)
xmap1_RL   = np.zeros([n_RL, num_points_y_half]);    xmap1_nonRL   = np.zeros(num_points_y_half);      xmap1_ref   = np.zeros(num_points_y_half)
xmap2_RL   = np.zeros([n_RL, num_points_y_half]);    xmap2_nonRL   = np.zeros(num_points_y_half);      xmap2_ref   = np.zeros(num_points_y_half)
eigval_RL  = np.zeros([n_RL, num_points_y_half,3]);  eigval_nonRL  = np.zeros([num_points_y_half,3]);  eigval_ref  = np.zeros([num_points_y_half,3])

# RL data
for i_RL in range(n_RL):
    ( Rkk_RL[i_RL], lambda1_RL[i_RL], lambda2_RL[i_RL], lambda3_RL[i_RL], xmap1_RL[i_RL], xmap2_RL[i_RL] ) \
        = compute_reynolds_stress_dof( favre_uffuff_RL[i_RL], favre_uffvff_RL[i_RL], favre_uffwff_RL[i_RL], favre_vffvff_RL[i_RL], favre_vffwff_RL[i_RL], favre_wffwff_RL[i_RL], verbose=verbose )
    eigval_RL[i_RL,:,0] = lambda1_RL[i_RL];     eigval_RL[i_RL,:,1] = lambda2_RL[i_RL];     eigval_RL[i_RL,:,2] = lambda3_RL[i_RL]  

# non-RL non-converged data
( Rkk_nonRL, lambda1_nonRL, lambda2_nonRL, lambda3_nonRL, xmap1_nonRL, xmap2_nonRL ) \
    = compute_reynolds_stress_dof( favre_uffuff_nonRL, favre_uffvff_nonRL, favre_uffwff_nonRL, favre_vffvff_nonRL, favre_vffwff_nonRL, favre_wffwff_nonRL, verbose=verbose )
eigval_nonRL[:,0] = lambda1_nonRL;     eigval_nonRL[:,1] = lambda2_nonRL;     eigval_nonRL[:,2] = lambda3_nonRL

# non-RL converged reference data
( Rkk_ref, lambda1_ref, lambda2_ref, lambda3_ref, xmap1_ref, xmap2_ref ) \
    = compute_reynolds_stress_dof( favre_uffuff_ref, favre_uffvff_ref, favre_uffwff_ref, favre_vffvff_ref, favre_vffwff_ref, favre_wffwff_ref, verbose=verbose )
eigval_ref[:,0] = lambda1_ref;     eigval_ref[:,1] = lambda2_ref;     eigval_ref[:,2] = lambda3_ref

print("Rij decomposed successfully!")

#-----------------------------------------------------------------------------------------
#           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
#-----------------------------------------------------------------------------------------

# ---------------------- Plot Barycentric Map for each RL global step (specific iteration & ensemble) ---------------------- 

print("Building triangle barycentric map plots...")
#for i_RL in range(n_RL):
#    visualizer.build_anisotropy_tensor_barycentric_xmap_triang(y_delta_RL,    xmap1_RL[i_RL], xmap2_RL[i_RL], averaging_time_nonConv, f"anisotropy_tensor_barycentric_map_RL_{file_details_list[i_RL]}")
visualizer.build_anisotropy_tensor_barycentric_xmap_triang(    y_delta_nonRL, xmap1_nonRL,    xmap2_nonRL,    averaging_time_nonConv, f"anisotropy_tensor_barycentric_map_nonRL")
visualizer.build_anisotropy_tensor_barycentric_xmap_triang(    y_delta_ref,   xmap1_ref,      xmap2_ref,      averaging_time_ref,     f"anisotropy_tensor_barycentric_map_ref")
print("Triangle barycentric map plotted successfully!")

# ----------------- Plot Animation Frames of um, urmsf, Rij dof for increasing RL global step (specific iteration & ensemble) -----------------

print("Building gif frames...")
frames_avg_u = []; frames_rmsf_u = []; frames_rkk = []; frames_eig = []; frames_xmap_coord = []; frames_xmap_triang = []; 
avg_u_max    = 20.0
rmsf_u_max   = int(np.max([np.max(rmsf_u_RL), np.max(rmsf_u_nonRL), np.max(rmsf_u_ref)]))+1
for i_RL in range(n_RL):
    # log progress
    if i_RL % (n_RL//10 or 1) == 0:
        print(f"{i_RL/n_RL*100:.0f}%")
    # Build frames
    frames_avg_u       = visualizer.build_um_frame(   frames_avg_u,  y_plus_RL, y_plus_ref, avg_u_RL[i_RL],  avg_u_ref,  averaging_time_nonConv, global_step_list[i_RL], ylim=[0.0, avg_u_max],  x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_rmsf_u      = visualizer.build_urmsf_frame(frames_rmsf_u, y_plus_RL, y_plus_ref, rmsf_u_RL[i_RL], rmsf_u_ref, averaging_time_nonConv, global_step_list[i_RL], ylim=[0.0, rmsf_u_max], x_actuator_boundaries=y_plus_actuators_boundaries)
    frames_rkk         = visualizer.build_reynolds_stress_tensor_trace_frame( frames_rkk, y_delta_RL, y_delta_ref, Rkk_RL[i_RL],    Rkk_ref, averaging_time_nonConv, global_step_list[i_RL])
    frames_eig         = visualizer.build_anisotropy_tensor_eigenvalues_frame(frames_eig, y_delta_RL, y_delta_ref, eigval_RL[i_RL], eigval_ref, averaging_time_nonConv, global_step_list[i_RL])
    frames_xmap_coord  = visualizer.build_anisotropy_tensor_barycentric_xmap_coord_frame( frames_xmap_coord,  y_delta_RL, y_delta_ref, xmap1_RL[i_RL], xmap2_RL[i_RL], xmap1_ref, xmap2_ref, averaging_time_nonConv, global_step_list[i_RL])
    frames_xmap_triang = visualizer.build_anisotropy_tensor_barycentric_xmap_triang_frame(frames_xmap_triang, y_delta_RL, xmap1_RL[i_RL], xmap2_RL[i_RL], averaging_time_nonConv, global_step_list[i_RL])

print("Building gifs from frames...")
visualizer.build_main_gifs_from_frames(frames_avg_u, frames_rmsf_u, frames_rkk, frames_eig, frames_xmap_coord, frames_xmap_triang)
print("Gifs plotted successfully!")




