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
plt.rc( 'axes',       axisbelow=True)
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

# --- Training / Evaluation parameters ---

# Reference & non-RL data directory
filePath = os.path.dirname(os.path.abspath(__file__))
compareDatasetDir = os.path.join("/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/post_process", f"data_Retau{Re_tau:.0f}")

# Evaluation directories
if Re_tau == 100:
    iteration_min_nonRL   = 3240000
    iteration_max_nonRL   = 3270000
    diter_snapshots_nonRL = 1000
    diter_snapshots_RL    = 1000
    RLdatasetDict = {
        "Superv. Ep. 2560": "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau100_eval_S/post_process/eval_2025-09-04--15-26-04--16ed_globalStep2560_normal/rhea_exp/output_data",
        "Unsuper. Ep. 2560": "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau100_eval_U/post_process/eval_2025-09-04--15-26-04--16ed_globalStep2560_normal/rhea_exp/output_data",
        "Superv. Ep. 15040": "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau100_eval_S/post_process/eval_2025-09-04--15-26-04--16ed_globalStep15040_normal/rhea_exp/output_data",
        "Unsuperv. Ep. 15040": "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau100_eval_U/post_process/eval_2025-09-04--15-26-04--16ed_globalStep15040_normal/rhea_exp/output_data",
    }
    # Plots dataset colormap
    isUnsuperv = [0,1,0,1]
    setEpochs  = [0,0,1,1]

elif Re_tau == 180:
    iteration_min_nonRL = 2820000 
    iteration_max_nonRL = 2850000
    diter_snapshots_nonRL = 5000
    diter_snapshots_RL    = 5000
    RLdatasetDict = {
        "Superv. Ep. 2560": "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau180_eval_S/post_process/eval_2025-09-04--15-26-04--16ed_globalStep2560_feedbackloop_utau_ubulk/rhea_exp/output_data",
        #"Unsuper. Ep. 2560": "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau180_eval_U/post_process/eval_2025-09-04--15-26-04--16ed_globalStep2560_normal/rhea_exp/output_data",
        "Superv. Ep. 15040": "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau180_eval_S/post_process/eval_2025-09-04--15-26-04--16ed_globalStep15040_feedbackloop_utau_ubulk/rhea_exp/output_data",
        "Unsuperv. Ep. 15040": "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau180_eval_U/post_process/eval_2025-09-04--15-26-04--16ed_globalStep15040_normal/rhea_exp/output_data",
    }
    # Plots dataset colormap
    isUnsuperv = [0,0,1]
    setEpochs  = [0,1,1]

else:
    raise ValueError(f"Invalid Re_tau: {Re_tau}. Must be one of [100, 180].")

N_RL_datasets = len(RLdatasetDict) 
dataset_colormap_tab20 = [i+2*j for i,j in zip(isUnsuperv, setEpochs)]  # [0,1,2,3]

# Directory of build plots
postDir = f"/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/post_process/article_test/"

#--------------------------------------------------------------------------------------------

# ----------- Build data h5 filenames ------------

def iter_patterns_from_diter(diter:int) -> list[str]:
    if diter==1000:
        return ["000"]
    elif diter==5000:
        return ["0000", "5000"]
    elif diter==10000:
        return ["0000"]
    else:
        raise ValueError(f"Unsupported diter value: {diter}")

# --- non-RL converged reference filename ---

print("\nReference, Converged file:")
filename_ref = f"{compareDatasetDir}/3d_turbulent_channel_flow_reference.h5"

# --- non-RL filenames ---

matching_files = []
for pat in iter_patterns_from_diter(diter_snapshots_nonRL):
    pattern = f"{compareDatasetDir}/3d_turbulent_channel_flow_*{pat}.h5"
    matching_files.extend(glob.glob(pattern))
matching_files = sorted(matching_files)

all_files = []
if matching_files:
    print("\n\n--------------------------- Non-RL files: ---------------------------")
    for filepath in matching_files:
        filename       = os.path.basename(filepath)
        parts_filename = filename.split('_')
        # Extract iteration
        try:
            iter_num    = int(os.path.splitext(parts_filename[-1])[0])
            if (iter_num >= iteration_min_nonRL and iter_num <= iteration_max_nonRL):
                all_files.append((iter_num, filepath))
                print(f"\nFilename: {filename}, \nIteration: {iter_num}")
        except (ValueError):
            print(f"\nSkipping file with no number iteration: {filename}, in filepath: {filepath}")

    # Sort numerically by iteration number
    all_files.sort(key=lambda x: x[0])
    # Unpack
    iteration_nonRL_list = [iter_num for iter_num, _ in all_files]
    filename_nonRL_list  = [filepath for _, filepath in all_files]
    N_nonRL = len(filename_nonRL_list)

# --- RL filenames ---

print("\n\n--------------------------- RL files: ---------------------------")
iteration_RL_dict = {}
filename_RL_dict  = {}
N_RL_dict = {}
for dataset_name, dataset_dir in RLdatasetDict.items():
    print("\nRL Dataset directory:\n", dataset_dir)

    matching_files = []
    for pat in iter_patterns_from_diter(diter_snapshots_RL):
        pattern = f"{dataset_dir}/RL_3d_turbulent_channel_flow_*{pat}_ensemble0_step*.h5"
        matching_files.extend(glob.glob(pattern))
    matching_files = sorted(matching_files)

    all_files = []
    if matching_files:
        print("RL files:")
        for filepath in matching_files:
            filename       = os.path.basename(filepath)
            parts_filename = filename.split('_')
            # Extract iteration
            try:
                iter_num    = int(os.path.splitext(parts_filename[5])[0])
                if (iter_num >= iteration_min_nonRL and iter_num <= iteration_max_nonRL):
                    all_files.append((iter_num, filepath))
                    print(f"\nFilename: {filename}, \nIteration: {iter_num}")
            except (ValueError):
                print(f"Skipping file with no number iteration: {filename}, in filepath: {filepath}")

        # Sort numerically by iteration number
        all_files.sort(key=lambda x: x[0])
        # Unpack
        iteration_RL_dict[dataset_name] = [iter_num for iter_num, _ in all_files]
        filename_RL_dict[dataset_name]  = [filepath for _, filepath in all_files]
        N_RL_dict[dataset_name]         = len(all_files)

# Check all RL datasets have same number of snapshots
assert len(set(N_RL_dict.values())) == 1
N_RL = N_RL_dict[dataset_name]

#--------------------------------------------------------------------------------------------

# ----------- Get non-RL data ------------

# --- Check if non-RL files exists ---

print("\nCheking files...")

if not os.path.isfile(filename_ref):
    print(f"Error: File '{filename_ref}' not found.")
    sys.exit(1)
for filename_nonRL in filename_nonRL_list:
    if not os.path.isfile(filename_nonRL):
        print(f"Error: File '{filename_nonRL}' not found.")
        sys.exit(1)
for database_name,_ in RLdatasetDict.items():
    for filename_RL in filename_RL_dict[database_name]:
        if not os.path.isfile(filename_RL):
            print(f"Error: File '{filename_RL}' not found.")
            sys.exit(1)

# --- Get data from 3d-snapshots h5 files ---

print("\nImporting data from files...")

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

print("\nImporting data from RL files:")
averaging_time_RL_dict = {} 
y_data_RL_dict = {}         
avg_u_data_RL_dict = {}     
avg_v_data_RL_dict = {}     
avg_w_data_RL_dict = {}     
rmsf_u_data_RL_dict = {}    
rmsf_v_data_RL_dict = {}    
rmsf_w_data_RL_dict = {}
for database_name in RLdatasetDict.keys():
    print(f"\nImport data from RL database: {database_name}")
    for i in range(N_RL):
        filename_RL = filename_RL_dict[database_name][i]
        with h5py.File( filename_RL, 'r' ) as data_file:
            #list( data_file.keys() )
            averaging_time_aux = data_file.attrs["AveragingTime"][0]
            y_data_aux        = data_file['y'][1:-1,1:-1,1:-1]
            avg_u_data_aux    = data_file['avg_u'][1:-1,1:-1,1:-1]
            avg_v_data_aux    = data_file['avg_v'][1:-1,1:-1,1:-1]
            avg_w_data_aux    = data_file['avg_w'][1:-1,1:-1,1:-1]
            rmsf_u_data_aux   = data_file['rmsf_u'][1:-1,1:-1,1:-1]
            rmsf_v_data_aux   = data_file['rmsf_v'][1:-1,1:-1,1:-1]
            rmsf_w_data_aux   = data_file['rmsf_w'][1:-1,1:-1,1:-1]
        # Initialize allocation arrays
        if i == 0:
            num_points_x      = avg_u_data_aux[0,0,:].size
            num_points_y      = avg_u_data_aux[0,:,0].size
            num_points_z      = avg_u_data_aux[:,0,0].size
            num_points_xz     = num_points_x*num_points_z
            averaging_time_RL_dict[database_name] = np.zeros(N_RL)
            y_data_RL_dict[database_name]         = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
            avg_u_data_RL_dict[database_name]     = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
            avg_v_data_RL_dict[database_name]     = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
            avg_w_data_RL_dict[database_name]     = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
            rmsf_u_data_RL_dict[database_name]    = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])            
            rmsf_v_data_RL_dict[database_name]    = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])            
            rmsf_w_data_RL_dict[database_name]    = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])  
        # Fill allocation arrays
        averaging_time_RL_dict[database_name][i]    = averaging_time_aux
        y_data_RL_dict[database_name][i,:,:,:]      = y_data_aux
        avg_u_data_RL_dict[database_name][i,:,:,:]  = avg_u_data_aux
        avg_v_data_RL_dict[database_name][i,:,:,:]  = avg_v_data_aux
        avg_w_data_RL_dict[database_name][i,:,:,:]  = avg_w_data_aux
        rmsf_u_data_RL_dict[database_name][i,:,:,:] = rmsf_u_data_aux
        rmsf_v_data_RL_dict[database_name][i,:,:,:] = rmsf_v_data_aux
        rmsf_w_data_RL_dict[database_name][i,:,:,:] = rmsf_w_data_aux
        print(f"non-RL non-converged data imported from file '{filename_RL}' - averaging time: {averaging_time_aux:.6f}")

print("\nData imported successfully!")

# -------------- Averaging fields using XZ symmetries --------------

### Allocate averaged variables
y_plus_RL_dict      = {db: np.zeros( [N_RL, int( 0.5*num_points_y )] ) for db in RLdatasetDict.keys()}
avg_u_plus_RL_dict  = {db: np.zeros( [N_RL, int( 0.5*num_points_y )] ) for db in RLdatasetDict.keys()}
avg_v_plus_RL_dict  = {db: np.zeros( [N_RL, int( 0.5*num_points_y )] ) for db in RLdatasetDict.keys()}
avg_w_plus_RL_dict  = {db: np.zeros( [N_RL, int( 0.5*num_points_y )] ) for db in RLdatasetDict.keys()}
rmsf_u_plus_RL_dict = {db: np.zeros( [N_RL, int( 0.5*num_points_y )] ) for db in RLdatasetDict.keys()}
rmsf_v_plus_RL_dict = {db: np.zeros( [N_RL, int( 0.5*num_points_y )] ) for db in RLdatasetDict.keys()}
rmsf_w_plus_RL_dict = {db: np.zeros( [N_RL, int( 0.5*num_points_y )] ) for db in RLdatasetDict.keys()}
y_plus_nonRL        = np.zeros( [N_nonRL, int( 0.5*num_points_y )] )
avg_u_plus_nonRL    = np.zeros( [N_nonRL, int( 0.5*num_points_y )] )
avg_v_plus_nonRL    = np.zeros( [N_nonRL, int( 0.5*num_points_y )] )
avg_w_plus_nonRL    = np.zeros( [N_nonRL, int( 0.5*num_points_y )] )
rmsf_u_plus_nonRL   = np.zeros( [N_nonRL, int( 0.5*num_points_y )] )
rmsf_v_plus_nonRL   = np.zeros( [N_nonRL, int( 0.5*num_points_y )] )
rmsf_w_plus_nonRL   = np.zeros( [N_nonRL, int( 0.5*num_points_y )] )
y_plus_ref          = np.zeros( int( 0.5*num_points_y ) )
avg_u_plus_ref      = np.zeros( int( 0.5*num_points_y ) )
avg_v_plus_ref      = np.zeros( int( 0.5*num_points_y ) )
avg_w_plus_ref      = np.zeros( int( 0.5*num_points_y ) )
rmsf_u_plus_ref     = np.zeros( int( 0.5*num_points_y ) )
rmsf_v_plus_ref     = np.zeros( int( 0.5*num_points_y ) )
rmsf_w_plus_ref     = np.zeros( int( 0.5*num_points_y ) )

### Average variables in space
print("\n\nAveraging variables in space...")

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

            # RL non-conv data:
            for database_name in RLdatasetDict.keys():
                for n in range(N_RL):
                    if( j > ( int( 0.5*num_points_y ) - 1 ) ):  # top-wall
                        y_plus_RL_dict[database_name][n,aux_j]  += ( 0.5/num_points_xz )*(2.0 - y_data_RL_dict[database_name][n,k,j,i])*( u_tau/nu_ref )
                    else:                                       # bottom-wall
                        y_plus_RL_dict[database_name][n,aux_j]  += ( 0.5/num_points_xz ) * y_data_RL_dict[database_name][n,k,j,i]*( u_tau/nu_ref )
                    avg_u_plus_RL_dict[database_name][n,aux_j]  += ( 0.5/num_points_xz ) * avg_u_data_RL_dict[database_name][n,k,j,i]*( 1.0/u_tau )
                    avg_v_plus_RL_dict[database_name][n,aux_j]  += ( 0.5/num_points_xz ) * avg_v_data_RL_dict[database_name][n,k,j,i]*( 1.0/u_tau )
                    avg_w_plus_RL_dict[database_name][n,aux_j]  += ( 0.5/num_points_xz ) * avg_w_data_RL_dict[database_name][n,k,j,i]*( 1.0/u_tau )
                    rmsf_u_plus_RL_dict[database_name][n,aux_j] += ( 0.5/num_points_xz ) * rmsf_u_data_RL_dict[database_name][n,k,j,i]*( 1.0/u_tau )
                    rmsf_v_plus_RL_dict[database_name][n,aux_j] += ( 0.5/num_points_xz ) * rmsf_v_data_RL_dict[database_name][n,k,j,i]*( 1.0/u_tau )
                    rmsf_w_plus_RL_dict[database_name][n,aux_j] += ( 0.5/num_points_xz ) * rmsf_w_data_RL_dict[database_name][n,k,j,i]*( 1.0/u_tau )
            
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

print("\nVariables averaged successfully!")

# ----------------- RL & non-RL Errors w.r.t. Reference -----------------

ny = y_plus_ref.size
y_plus = np.concatenate([[0.0],y_plus_ref,[delta]])


# --- Calculate errors using spatial interpolation ---

print("\nCalculating L2-Error...")

# L2 Error (RMS Error)
L2_error_avg_u_plus_RL_dict  = {db: np.zeros(N_RL) for db in RLdatasetDict.keys()}
L2_error_avg_v_plus_RL_dict  = {db: np.zeros(N_RL) for db in RLdatasetDict.keys()}
L2_error_avg_w_plus_RL_dict  = {db: np.zeros(N_RL) for db in RLdatasetDict.keys()}
L2_error_rmsf_u_plus_RL_dict = {db: np.zeros(N_RL) for db in RLdatasetDict.keys()}
L2_error_rmsf_v_plus_RL_dict = {db: np.zeros(N_RL) for db in RLdatasetDict.keys()}
L2_error_rmsf_w_plus_RL_dict = {db: np.zeros(N_RL) for db in RLdatasetDict.keys()}
L2_error_avg_u_plus_nonRL  = np.zeros( N_nonRL )
L2_error_avg_v_plus_nonRL  = np.zeros( N_nonRL )
L2_error_avg_w_plus_nonRL  = np.zeros( N_nonRL )
L2_error_rmsf_u_plus_nonRL = np.zeros( N_nonRL )
L2_error_rmsf_v_plus_nonRL = np.zeros( N_nonRL )
L2_error_rmsf_w_plus_nonRL = np.zeros( N_nonRL )

for database_name in RLdatasetDict.keys():
    ylength = 0.0
    for j in range(ny):
        dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
        ylength += dy
        for i in range(N_RL):
            L2_error_avg_u_plus_RL_dict[database_name][i]  += ( ( avg_u_plus_RL_dict[database_name][i,j]  - avg_u_plus_ref[j])**2 )  * dy
            L2_error_avg_v_plus_RL_dict[database_name][i]  += ( ( avg_v_plus_RL_dict[database_name][i,j]  - avg_v_plus_ref[j])**2 )  * dy
            L2_error_avg_w_plus_RL_dict[database_name][i]  += ( ( avg_w_plus_RL_dict[database_name][i,j]  - avg_w_plus_ref[j])**2 )  * dy
            L2_error_rmsf_u_plus_RL_dict[database_name][i] += ( ( rmsf_u_plus_RL_dict[database_name][i,j] - rmsf_u_plus_ref[j])**2 ) * dy
            L2_error_rmsf_v_plus_RL_dict[database_name][i] += ( ( rmsf_v_plus_RL_dict[database_name][i,j] - rmsf_v_plus_ref[j])**2 ) * dy
            L2_error_rmsf_w_plus_RL_dict[database_name][i] += ( ( rmsf_w_plus_RL_dict[database_name][i,j] - rmsf_w_plus_ref[j])**2 ) * dy  
    L2_error_avg_u_plus_RL_dict[database_name]  = np.sqrt( L2_error_avg_u_plus_RL_dict[database_name] / ylength )
    L2_error_avg_v_plus_RL_dict[database_name]  = np.sqrt( L2_error_avg_v_plus_RL_dict[database_name] / ylength )
    L2_error_avg_w_plus_RL_dict[database_name]  = np.sqrt( L2_error_avg_w_plus_RL_dict[database_name] / ylength )
    L2_error_rmsf_u_plus_RL_dict[database_name] = np.sqrt( L2_error_rmsf_u_plus_RL_dict[database_name] / ylength )
    L2_error_rmsf_v_plus_RL_dict[database_name] = np.sqrt( L2_error_rmsf_v_plus_RL_dict[database_name] / ylength )
    L2_error_rmsf_w_plus_RL_dict[database_name] = np.sqrt( L2_error_rmsf_w_plus_RL_dict[database_name] / ylength )

ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i in range(N_nonRL):
        L2_error_avg_u_plus_nonRL[i]  += ( ( avg_u_plus_nonRL[i,j]  - avg_u_plus_ref[j])**2 )  * dy
        L2_error_avg_v_plus_nonRL[i]  += ( ( avg_v_plus_nonRL[i,j]  - avg_v_plus_ref[j])**2 )  * dy
        L2_error_avg_w_plus_nonRL[i]  += ( ( avg_w_plus_nonRL[i,j]  - avg_w_plus_ref[j])**2 )  * dy
        L2_error_rmsf_u_plus_nonRL[i] += ( ( rmsf_u_plus_nonRL[i,j] - rmsf_u_plus_ref[j])**2 ) * dy
        L2_error_rmsf_v_plus_nonRL[i] += ( ( rmsf_v_plus_nonRL[i,j] - rmsf_v_plus_ref[j])**2 ) * dy
        L2_error_rmsf_w_plus_nonRL[i] += ( ( rmsf_w_plus_nonRL[i,j] - rmsf_w_plus_ref[j])**2 ) * dy  
L2_error_avg_u_plus_nonRL  = np.sqrt( L2_error_avg_u_plus_nonRL / ylength )          
L2_error_avg_v_plus_nonRL  = np.sqrt( L2_error_avg_v_plus_nonRL / ylength )          
L2_error_avg_w_plus_nonRL  = np.sqrt( L2_error_avg_w_plus_nonRL / ylength )          
L2_error_rmsf_u_plus_nonRL = np.sqrt( L2_error_rmsf_u_plus_nonRL / ylength )          
L2_error_rmsf_v_plus_nonRL = np.sqrt( L2_error_rmsf_v_plus_nonRL / ylength )          
L2_error_rmsf_w_plus_nonRL = np.sqrt( L2_error_rmsf_w_plus_nonRL / ylength )          

print("Errors calculated successfully!")


# -------------------- Errors Plots --------------------
 
def build_velocity_error_plot(avg_time_RL_dict, avg_time_nonRL,
                              err_avg_u_RL_dict, err_avg_v_RL_dict, err_avg_w_RL_dict, 
                              err_rmsf_u_RL_dict, err_rmsf_v_RL_dict, err_rmsf_w_RL_dict, 
                              err_avg_u_nonRL, err_avg_v_nonRL, err_avg_w_nonRL, 
                              err_rmsf_u_nonRL, err_rmsf_v_nonRL, err_rmsf_w_nonRL, 
                              error_num='2'):
    # u-component
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.semilogy( avg_time_nonRL, err_avg_u_nonRL,  linestyle = '-', linewidth = 2, color = 'black', zorder = 1, label = r'non-RL $\overline{u}^+$' )
    plt.semilogy( avg_time_nonRL, err_rmsf_u_nonRL, linestyle = ':', linewidth = 2, color = 'black', zorder = 2, label = r'non-RL ${u}_{\textrm{rms}}^+$' )
    aux = 0
    for database_name in RLdatasetDict.keys():
        plt.semilogy( avg_time_RL_dict[database_name], err_avg_u_RL_dict[database_name],  linestyle = '-', linewidth = 2, color = plt.cm.tab20(dataset_colormap_tab20[aux]), zorder = 1, label = rf'RL $\overline{{u}}^+$ {database_name}' )
        plt.semilogy( avg_time_RL_dict[database_name], err_rmsf_u_RL_dict[database_name], linestyle = ':', linewidth = 2, color = plt.cm.tab20(dataset_colormap_tab20[aux]), zorder = 2, label = rf'RL $u_{{\textrm{{rms}}}}^+$ {database_name}' )
        aux += 1
    plt.xlabel(r'$\textrm{Averaging time } t_{avg}^+$' )
    if error_num == "2":
        plt.ylabel(r'$\textrm{NRMSE }(\overline{u}^{+}, u_{\textrm{rms}}^{+})$')
    else:
        plt.ylabel(rf'$L_{{{error_num}}}$ Error' )
    plt.grid(which='both',axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True) # (loc='upper right', frameon=True, framealpha=1.0, fancybox=True)
    #plt.legend(loc='upper right', frameon=True)
    plt.tick_params( axis = 'both', pad = 7.5 )
    plt.tight_layout()
    filename = os.path.join(postDir, f'2_L{error_num}_error_u_Retau{Re_tau:.0f}_evaluation.svg')
    plt.savefig( filename, format = 'svg', bbox_inches = 'tight' )
    plt.close()
    print(f"\nBuild plot: '{filename}'")

    # v-component
    plt.figure(figsize=(10, 5))
    plt.semilogy( avg_time_nonRL, err_avg_v_nonRL,  linestyle = '-', linewidth = 2, color = 'black', zorder = 1, label = r'non-RL $\overline{v}^+$' )
    plt.semilogy( avg_time_nonRL, err_rmsf_v_nonRL, linestyle = ':', linewidth = 2, color = 'black', zorder = 2, label = r'non-RL ${v}_{\textrm{rms}}^+$' )
    aux = 0
    for database_name in RLdatasetDict.keys():
        plt.semilogy( avg_time_RL_dict[database_name], err_avg_v_RL_dict[database_name],  linestyle = '-', linewidth = 2, color = plt.cm.tab20(dataset_colormap_tab20[aux]), zorder = 1, label = rf'RL $\overline{{v}}^+$ {database_name}' )
        plt.semilogy( avg_time_RL_dict[database_name], err_rmsf_v_RL_dict[database_name], linestyle = ':', linewidth = 2, color = plt.cm.tab20(dataset_colormap_tab20[aux]), zorder = 2, label = rf'RL $v_{{\textrm{{rms}}}}^+$ {database_name}' )
        aux += 1
    plt.xlabel(r'$\textrm{Averaging time } t_{avg}^+$' )
    if error_num == "2":
        plt.ylabel(r'$\textrm{NRMSE }(\overline{v}^{+}, v_{\textrm{rms}}^{+})$')
    else:
        plt.ylabel(rf'$L_{{{error_num}}}$ Error' )
    plt.grid(which='both',axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True) # (loc='upper right', frameon=True, framealpha=1.0, fancybox=True)
    #plt.legend(loc='upper right', frameon=True)
    plt.tick_params( axis = 'both', pad = 7.5 )
    plt.tight_layout()
    filename = os.path.join(postDir, f'2_L{error_num}_error_v_Retau{Re_tau:.0f}_evaluation.svg')
    plt.savefig( filename, format = 'svg', bbox_inches = 'tight' )
    plt.close()
    print(f"\nBuild plot: '{filename}'")

    # w-component
    plt.figure(figsize=(10, 5))
    plt.semilogy( avg_time_nonRL, err_avg_w_nonRL,  linestyle = '-', linewidth = 2, color = 'black', zorder = 1, label = r'non-RL $\overline{w}^+$' )
    plt.semilogy( avg_time_nonRL, err_rmsf_w_nonRL, linestyle = ':', linewidth = 2, color = 'black', zorder = 2, label = r'non-RL ${w}_{\textrm{rms}}^+$' )
    aux = 0
    for database_name in RLdatasetDict.keys():
        plt.semilogy( avg_time_RL_dict[database_name], err_avg_w_RL_dict[database_name],  linestyle = '-', linewidth = 2, color = plt.cm.tab20(dataset_colormap_tab20[aux]), zorder = 1, label = rf'RL $\overline{{w}}^+$ {database_name}' )
        plt.semilogy( avg_time_RL_dict[database_name], err_rmsf_w_RL_dict[database_name], linestyle = ':', linewidth = 2, color = plt.cm.tab20(dataset_colormap_tab20[aux]), zorder = 2, label = rf'RL $w_{{\textrm{{rms}}}}^+$ {database_name}' )
        aux += 1
    plt.xlabel(r'$\textrm{Averaging time } t_{avg}^+$' )
    if error_num == "2":
        plt.ylabel(r'$\textrm{NRMSE }(\overline{w}^{+}, w_{\textrm{rms}}^{+})$')
    else:
        plt.ylabel(rf'$L_{{{error_num}}}$ Error' )
    plt.grid(which='both',axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True) # (loc='upper right', frameon=True, framealpha=1.0, fancybox=True)
    #plt.legend(loc='upper right', frameon=True)
    plt.tick_params( axis = 'both', pad = 7.5 )
    plt.tight_layout()
    filename = os.path.join(postDir, f'2_L{error_num}_error_w_Retau{Re_tau:.0f}_evaluation.svg')
    plt.savefig( filename, format = 'svg', bbox_inches = 'tight' )
    plt.close()
    print(f"\nBuild plot: '{filename}'")



print("\nBuilding error plots...")

# L2-Error plot
build_velocity_error_plot(
    averaging_time_RL_dict, averaging_time_nonRL,
    L2_error_avg_u_plus_RL_dict,  L2_error_avg_v_plus_RL_dict,  L2_error_avg_w_plus_RL_dict, 
    L2_error_rmsf_u_plus_RL_dict, L2_error_rmsf_v_plus_RL_dict, L2_error_rmsf_w_plus_RL_dict, 
    L2_error_avg_u_plus_nonRL,    L2_error_avg_v_plus_nonRL,    L2_error_avg_w_plus_nonRL, 
    L2_error_rmsf_u_plus_nonRL,   L2_error_rmsf_v_plus_nonRL,   L2_error_rmsf_w_plus_nonRL
)

print("Error plots built successfully!")