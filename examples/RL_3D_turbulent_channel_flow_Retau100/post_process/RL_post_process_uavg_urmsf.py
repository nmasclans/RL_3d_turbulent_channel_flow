#!/home/jofre/miniconda3/envs/smartrhea-env-v2/bin/python3

import sys
import os
import glob
import numpy as np
import h5py    
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import rc,rcParams
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm
#np.set_printoptions(threshold=sys.maxsize)
#plt.rc( 'text', usetex = True )
#rc('font', family='sanserif')
#plt.rc( 'font', size = 20 )
#plt.rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}', r'\usepackage{amssymb}', r'\usepackage{color}' ]

### Reference parameters
rho_0   = 1.0				# Reference density [kg/m3]
u_tau   = 1.0				# Friction velocity [m/s]
delta   = 1.0				# Channel half-height [m]
Re_tau  = 100.0				# Friction Reynolds number [-]
mu_ref  = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref  = mu_ref/rho_0			# Kinematic viscosity [m2/s]
dt_phys = 1.0e-4

# --- Get CASE parameters ---

try :
    iteration  = sys.argv[1]
    ensemble   = sys.argv[2]
    train_name = sys.argv[3]
    print(f"\nScript parameters: \n- Iteration: {iteration} \n- Ensemble: {ensemble}\n- Train name: {train_name}")
except :
    raise ValueError("Missing call arguments, should be: <iteration> <ensemble> <train_name>")

# post-processing directory
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

### Filenames of non-converged RL and non-RL
### > RL filenames
# Get 'file_details' & filename_RL
pattern = f"../rhea_exp/output_data/RL_3d_turbulent_channel_flow_{iteration}_ensemble{ensemble}_*.h5"
# Use glob to find all matching files
matching_files = sorted(glob.glob(pattern))
# List to store the extracted parts corresponding to "*"
filename_RL_list  = []
file_details_list = []
step_num_list     = []
# Check if files were found
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
        # Step number
        step_num = int(file_details[4:])
        step_num_list.append(step_num)
        # Print the file and the extracted part
        print(f"Filename: {base_filename}, File details: {file_details}, Step number: {step_num}")
else:
    print(f"No files found matching the pattern: {pattern}")
### > non-RL filename
# Build filename_nonRL
filename_nonRL = f"reference_data/3d_turbulent_channel_flow_{iteration}.h5"
# Build filename reference
filename_ref = f"reference_data/3d_turbulent_channel_flow_reference.h5"

# Check if files exists
for filename_RL in filename_RL_list:
    if not os.path.isfile(filename_RL):
        print(f"Error: File '{filename_RL}' not found.")
        sys.exit(1)
if not os.path.isfile(filename_nonRL):
    print(f"Error: File '{filename_nonRL}' not found.")
    sys.exit(1)
if not os.path.isfile(filename_ref):
    print(f"Error: File '{filename_ref}' not found.")
    sys.exit(1)

### Open data files
print("\nImporting data from files...")
# RL data
n_RL = len(filename_RL_list)
print(f"Num. RL files: {n_RL}")
for i_RL in range(n_RL):
    filename_RL = filename_RL_list[i_RL]
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
    if i_RL == 0:
        num_points_x      = avg_u_data_RL_aux[0,0,:].size
        num_points_y      = avg_u_data_RL_aux[0,:,0].size
        num_points_z      = avg_u_data_RL_aux[:,0,0].size
        num_points_xz     = num_points_x*num_points_z
        averaging_time_RL = averaging_time_RL_aux
        y_data_RL         = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_u_data_RL     = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_v_data_RL     = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_w_data_RL     = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        rmsf_u_data_RL    = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])            
        rmsf_v_data_RL    = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])            
        rmsf_w_data_RL    = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])  
    # Fill allocation arrays
    y_data_RL[i_RL,:,:,:]      = y_data_RL_aux
    avg_u_data_RL[i_RL,:,:,:]  = avg_u_data_RL_aux
    avg_v_data_RL[i_RL,:,:,:]  = avg_v_data_RL_aux
    avg_w_data_RL[i_RL,:,:,:]  = avg_w_data_RL_aux
    rmsf_u_data_RL[i_RL,:,:,:] = rmsf_u_data_RL_aux
    rmsf_v_data_RL[i_RL,:,:,:] = rmsf_v_data_RL_aux
    rmsf_w_data_RL[i_RL,:,:,:] = rmsf_w_data_RL_aux
    # Check same averaging time
    if not np.isclose(averaging_time_RL, averaging_time_RL_aux, atol=1e-8):
        raise ValueError("Averaging time should be equal for all RL h5 files")
    print(f"RL non-converged data imported from file '{filename_RL}' - averaging time: {averaging_time_RL:.2f}")
# non-RL non-converged data
with h5py.File( filename_nonRL, 'r' ) as data_file:
    averaging_time_nonRL = data_file.attrs["AveragingTime"][0]
    y_data_nonRL      = data_file['y'][1:-1,1:-1,1:-1]
    avg_u_data_nonRL  = data_file['avg_u'][1:-1,1:-1,1:-1]
    avg_v_data_nonRL  = data_file['avg_v'][1:-1,1:-1,1:-1]
    avg_w_data_nonRL  = data_file['avg_w'][1:-1,1:-1,1:-1]
    rmsf_u_data_nonRL = data_file['rmsf_u'][1:-1,1:-1,1:-1]
    rmsf_v_data_nonRL = data_file['rmsf_v'][1:-1,1:-1,1:-1]
    rmsf_w_data_nonRL = data_file['rmsf_w'][1:-1,1:-1,1:-1]
print(f"Non-RL non-converged data imported from file '{filename_nonRL}' - averaging time: {averaging_time_nonRL:.2f}")
# non-RL converged reference data
with h5py.File( filename_ref, 'r' ) as data_file:
    averaging_time_ref = data_file.attrs["AveragingTime"][0]
    y_data_ref      = data_file['y'][1:-1,1:-1,1:-1]
    avg_u_data_ref  = data_file['avg_u'][1:-1,1:-1,1:-1]
    avg_v_data_ref  = data_file['avg_v'][1:-1,1:-1,1:-1]
    avg_w_data_ref  = data_file['avg_w'][1:-1,1:-1,1:-1]
    rmsf_u_data_ref = data_file['rmsf_u'][1:-1,1:-1,1:-1]
    rmsf_v_data_ref = data_file['rmsf_v'][1:-1,1:-1,1:-1]
    rmsf_w_data_ref = data_file['rmsf_w'][1:-1,1:-1,1:-1]
print(f"Non-RL converged reference data imported from file '{filename_ref}' - averaging time: {averaging_time_ref:.2f}")

### Check same averaging time for RL & non-RL
if not np.isclose(averaging_time_RL, averaging_time_nonRL, atol=1e-8):
    raise ValueError(f"Averaging time should be equal for both RL & non-RL h5 files, while RL: {averaging_time_RL} != nonRL: {averaging_time_nonRL}")
else:
    averaging_time_nonConv = averaging_time_RL

### Allocate averaged variables
y_plus_RL      = np.zeros( [n_RL, int( 0.5*num_points_y )] ); y_plus_nonRL      = np.zeros( int( 0.5*num_points_y ) );  y_plus_ref      = np.zeros( int( 0.5*num_points_y ) )
avg_u_plus_RL  = np.zeros( [n_RL, int( 0.5*num_points_y )] ); avg_u_plus_nonRL  = np.zeros( int( 0.5*num_points_y ) );  avg_u_plus_ref  = np.zeros( int( 0.5*num_points_y ) ); 
rmsf_u_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_u_plus_nonRL = np.zeros( int( 0.5*num_points_y ) );  rmsf_u_plus_ref = np.zeros( int( 0.5*num_points_y ) )
rmsf_v_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_v_plus_nonRL = np.zeros( int( 0.5*num_points_y ) );  rmsf_v_plus_ref = np.zeros( int( 0.5*num_points_y ) )
rmsf_w_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_w_plus_nonRL = np.zeros( int( 0.5*num_points_y ) );  rmsf_w_plus_ref = np.zeros( int( 0.5*num_points_y ) )

### Average variables in space
print("\nAveraging variables in space...")
for j in range( 0, num_points_y ):
    for i in range( 0, num_points_x ):
        for k in range( 0, num_points_z ):
            aux_j = j
            if( j > ( int( 0.5*num_points_y ) - 1 ) ):      # top-wall
                aux_j = num_points_y - j - 1
            # RL data:
            for i_RL in range(n_RL):
                if( j > ( int( 0.5*num_points_y ) - 1 ) ):  # top-wall
                    y_plus_RL[i_RL,aux_j]  += ( 0.5/num_points_xz )*( 2.0 - y_data_RL[i_RL,k,j,i] )*( u_tau/nu_ref )
                else:                                       # bottom-wall
                    y_plus_RL[i_RL,aux_j]  += ( 0.5/num_points_xz )*y_data_RL[i_RL,k,j,i]*( u_tau/nu_ref )
                avg_u_plus_RL[i_RL,aux_j]  += ( 0.5/num_points_xz )*avg_u_data_RL[i_RL,k,j,i]*( 1.0/u_tau )
                rmsf_u_plus_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_RL[i_RL,k,j,i]*( 1.0/u_tau )
                rmsf_v_plus_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_RL[i_RL,k,j,i]*( 1.0/u_tau )
                rmsf_w_plus_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_RL[i_RL,k,j,i]*( 1.0/u_tau )
            # non-RL non-conv data:
            if( j > ( int( 0.5*num_points_y ) - 1 ) ):  # top-wall
                y_plus_nonRL[aux_j]  += ( 0.5/num_points_xz )*(2.0 - y_data_nonRL[k,j,i])*( u_tau/nu_ref )
            else:                                       # bottom-wall
                y_plus_nonRL[aux_j]  += ( 0.5/num_points_xz )*y_data_nonRL[k,j,i]*( u_tau/nu_ref )
            avg_u_plus_nonRL[aux_j]  += ( 0.5/num_points_xz )*avg_u_data_nonRL[k,j,i]*( 1.0/u_tau )
            rmsf_u_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_nonRL[k,j,i]*( 1.0/u_tau )
            rmsf_v_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_nonRL[k,j,i]*( 1.0/u_tau )
            rmsf_w_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_nonRL[k,j,i]*( 1.0/u_tau )
            # non-RL converged reference data
            if( j > ( int( 0.5*num_points_y ) - 1 ) ):  # top-wall
                y_plus_ref[aux_j]  += ( 0.5/num_points_xz )*(2.0 - y_data_ref[k,j,i])*( u_tau/nu_ref )
            else:                                       # bottom-wall
                y_plus_ref[aux_j]  += ( 0.5/num_points_xz )*y_data_ref[k,j,i]*( u_tau/nu_ref )
            avg_u_plus_ref[aux_j]  += ( 0.5/num_points_xz )*avg_u_data_ref[k,j,i]*( 1.0/u_tau )
            rmsf_u_plus_ref[aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_ref[k,j,i]*( 1.0/u_tau )
            rmsf_v_plus_ref[aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_ref[k,j,i]*( 1.0/u_tau )
            rmsf_w_plus_ref[aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_ref[k,j,i]*( 1.0/u_tau )
print("Variables averaged successfully!")


### Calculate TKE averaged profile from rmsf_u,v,w averaged profiles 
TKE_RL    = 0.5 * ( rmsf_u_plus_RL**2    + rmsf_v_plus_RL**2    + rmsf_w_plus_RL**2    )
TKE_nonRL = 0.5 * ( rmsf_u_plus_nonRL**2 + rmsf_v_plus_nonRL**2 + rmsf_w_plus_nonRL**2 )
TKE_ref   = 0.5 * ( rmsf_u_plus_ref**2   + rmsf_v_plus_ref**2   + rmsf_w_plus_ref**2   )

### Plot u+ vs. y+
print("\nBuilding plots...")
# Clear plot
plt.clf()
# Read & Plot data
plt.plot( y_plus_ref, avg_u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], avg_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
    else:
        plt.plot( y_plus_RL[i_RL], avg_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, avg_u_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s' )
# Configure plot
plt.xlim( 1.0, 2.0e2 )
plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
plt.xscale( 'log' )
plt.xlabel( 'y+' )
plt.ylim( 0.0, 20.0 )
plt.yticks( np.arange( 0.0, 20.1, 5.0 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( 'u+')
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( f'{postDir}/u_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg', format = 'jpg', dpi=600, bbox_inches = 'tight' )
# Clear plot
plt.clf()

### Plot u-rmsf 
# Read & Plot data
plt.plot( y_plus_ref, rmsf_u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label=f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], rmsf_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
    else:
        plt.plot( y_plus_RL[i_RL], rmsf_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, rmsf_u_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s' )
# Configure plot
plt.xlim( 1.0, 2.0e2 )
plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
plt.xscale( 'log' )
plt.xlabel( 'y+' )
plt.ylim( 0.0, 3.0 )
plt.yticks( np.arange( 0.0, 3.1, 0.5 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( 'u_rms+' )
#legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.text( 1.05, 1.0, 'u_rms+' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( f'{postDir}/u_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg', format = 'jpg', dpi=600, bbox_inches = 'tight' )
# Clear plot
plt.clf()

### Plot v-rmsf
# Read & Plot data
plt.plot( y_plus_ref, rmsf_v_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], rmsf_v_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
    else:
        plt.plot( y_plus_RL[i_RL], rmsf_v_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, rmsf_v_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
# Configure plot
plt.xlim( 1.0, 2.0e2 )
plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
plt.xscale( 'log' )
plt.xlabel( 'y+' )
plt.ylim( 0.0, 3.0 )
plt.yticks( np.arange( 0.0, 3.1, 0.5 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( 'v_rms+' )
#legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.text( 17.5, 0.2, 'v_rms+' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( f'{postDir}/v_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg', format = 'jpg', dpi=600, bbox_inches = 'tight' )
# Clear plot
plt.clf()

### Plot w-rmsf
# Read & Plot data
plt.plot( y_plus_ref, rmsf_w_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], rmsf_w_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
    else:
        plt.plot( y_plus_RL[i_RL], rmsf_w_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, rmsf_w_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
# Configure plot
plt.xlim( 1.0, 2.0e2 )
plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
plt.xscale( 'log' )
plt.xlabel( 'y+' )
plt.ylim( 0.0, 3.0 )
plt.yticks( np.arange( 0.0, 3.1, 0.5 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( 'w_rms+' )
#legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.text( 17.5, 0.2, 'w_rms+' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( f'{postDir}/w_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg', format = 'jpg', dpi=600, bbox_inches = 'tight' )
# Clear plot
plt.clf()

### Plot TKE
# Read & Plot data
plt.plot( y_plus_ref, TKE_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = f'RHEA non-RL Reference, Avg. time {averaging_time_ref:.2f}s' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], TKE_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}, Avg. time {averaging_time_nonConv:.2f}s' )
    else:
        plt.plot( y_plus_RL[i_RL], TKE_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, TKE_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = f'RHEA non-RL, Avg. time {averaging_time_nonConv:.2f}s'  )
# Configure plot
plt.xlim( 1.0, 2.0e2 )
plt.xticks( np.arange( 1.0, 2.01e2, 1.0 ) )
plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = 'True', labeltop = 'False', direction = 'in' )
plt.xscale( 'log' )
plt.xlabel( 'y+' )
plt.ylim( 0.0, 3.0 )
plt.yticks( np.arange( 0.0, 5.0, 1.0 ) )
plt.tick_params( axis = 'y', left = True, right = True, labelleft = 'True', labelright = 'False', direction = 'in' )
#plt.yscale( 'log' )
plt.ylabel( 'TKE+' )
#legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.text( 17.5, 0.2, 'TKE+' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( f'{postDir}/tke_plus_vs_y_plus_{iteration}_ensemble{ensemble}.jpg', format = 'jpg', dpi=600, bbox_inches = 'tight' )
# Clear plot
plt.clf()

print("Plots built successfully!")


############################# RL & non-RL Errors w.r.t. Reference #############################

if (np.allclose(y_plus_RL, y_plus_ref) & np.allclose(y_plus_nonRL, y_plus_ref)):
    print("\nNo need for interpolating data, as y_plus coordinates are the same for all data")
else:
    raise Exception("y-plus coordinates should be equal for all RL & non-RL & reference data, as no interpolation will be done.")
ny = y_plus_ref.size
y_plus = np.concatenate([[0.0],y_plus_ref,[delta]])

# --- Calculate errors using spatial interpolation ---

print("\nCalculating errors...")
# Absolute error 
abs_error_avg_u_plus_RL  = np.zeros( [n_RL, y_plus_ref.size] )
abs_error_rmsf_u_plus_RL = np.zeros( [n_RL, y_plus_ref.size] )
abs_error_rmsf_v_plus_RL = np.zeros( [n_RL, y_plus_ref.size] )
abs_error_rmsf_w_plus_RL = np.zeros( [n_RL, y_plus_ref.size] )
for i_RL in range(n_RL):
    abs_error_avg_u_plus_RL[i_RL,:]  = np.abs( avg_u_plus_RL[i_RL,:]  - avg_u_plus_ref )
    abs_error_rmsf_u_plus_RL[i_RL,:] = np.abs( rmsf_u_plus_RL[i_RL,:] - rmsf_u_plus_ref )
    abs_error_rmsf_v_plus_RL[i_RL,:] = np.abs( rmsf_v_plus_RL[i_RL,:] - rmsf_v_plus_ref )
    abs_error_rmsf_w_plus_RL[i_RL,:] = np.abs( rmsf_w_plus_RL[i_RL,:] - rmsf_w_plus_ref )
abs_error_avg_u_plus_nonRL  = np.abs( avg_u_plus_nonRL  - avg_u_plus_ref )
abs_error_rmsf_u_plus_nonRL = np.abs( rmsf_u_plus_nonRL - rmsf_u_plus_ref )
abs_error_rmsf_v_plus_nonRL = np.abs( rmsf_v_plus_nonRL - rmsf_v_plus_ref )
abs_error_rmsf_w_plus_nonRL = np.abs( rmsf_w_plus_nonRL - rmsf_w_plus_ref )

# L1 Error
L1_error_avg_u_plus_RL  = np.zeros(n_RL); L1_error_avg_u_plus_nonRL  = 0.0
L1_error_rmsf_u_plus_RL = np.zeros(n_RL); L1_error_rmsf_u_plus_nonRL = 0.0
L1_error_rmsf_v_plus_RL = np.zeros(n_RL); L1_error_rmsf_v_plus_nonRL = 0.0
L1_error_rmsf_w_plus_RL = np.zeros(n_RL); L1_error_rmsf_w_plus_nonRL = 0.0
ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i_RL in range(n_RL):
        L1_error_avg_u_plus_RL[i_RL]  += abs_error_avg_u_plus_RL[i_RL,j]  * dy
        L1_error_rmsf_u_plus_RL[i_RL] += abs_error_rmsf_u_plus_RL[i_RL,j] * dy
        L1_error_rmsf_v_plus_RL[i_RL] += abs_error_rmsf_v_plus_RL[i_RL,j] * dy
        L1_error_rmsf_w_plus_RL[i_RL] += abs_error_rmsf_w_plus_RL[i_RL,j] * dy
    L1_error_avg_u_plus_nonRL  += abs_error_avg_u_plus_nonRL[j]  * dy
    L1_error_rmsf_u_plus_nonRL += abs_error_rmsf_u_plus_nonRL[j] * dy
    L1_error_rmsf_v_plus_nonRL += abs_error_rmsf_v_plus_nonRL[j] * dy
    L1_error_rmsf_w_plus_nonRL += abs_error_rmsf_w_plus_nonRL[j] * dy
L1_error_avg_u_plus_RL     /= ylength     
L1_error_rmsf_u_plus_RL    /= ylength     
L1_error_rmsf_v_plus_RL    /= ylength     
L1_error_rmsf_w_plus_RL    /= ylength     
L1_error_avg_u_plus_nonRL  /= ylength         
L1_error_rmsf_u_plus_nonRL /= ylength         
L1_error_rmsf_v_plus_nonRL /= ylength         
L1_error_rmsf_w_plus_nonRL /= ylength         

# L2 Error (RMS Error)
L2_error_avg_u_plus_RL  = np.zeros(n_RL); L2_error_avg_u_plus_nonRL  = 0.0
L2_error_rmsf_u_plus_RL = np.zeros(n_RL); L2_error_rmsf_u_plus_nonRL = 0.0
L2_error_rmsf_v_plus_RL = np.zeros(n_RL); L2_error_rmsf_v_plus_nonRL = 0.0
L2_error_rmsf_w_plus_RL = np.zeros(n_RL); L2_error_rmsf_w_plus_nonRL = 0.0
ylength = 0.0
for j in range(ny):
    dy = np.abs(0.5 * (y_plus[j+2]-y_plus[j]))
    ylength += dy
    for i_RL in range(n_RL):
        L2_error_avg_u_plus_RL[i_RL]  += ( ( avg_u_plus_RL[i_RL,j]  - avg_u_plus_ref[j] )**2 )  * dy
        L2_error_rmsf_u_plus_RL[i_RL] += ( ( rmsf_u_plus_RL[i_RL,j] - rmsf_u_plus_ref[j] )**2 ) * dy 
        L2_error_rmsf_v_plus_RL[i_RL] += ( ( rmsf_v_plus_RL[i_RL,j] - rmsf_v_plus_ref[j] )**2 ) * dy 
        L2_error_rmsf_w_plus_RL[i_RL] += ( ( rmsf_w_plus_RL[i_RL,j] - rmsf_w_plus_ref[j] )**2 ) * dy 
    L2_error_avg_u_plus_nonRL  += ( ( avg_u_plus_nonRL[j]  - avg_u_plus_ref[j])**2 )  * dy
    L2_error_rmsf_u_plus_nonRL += ( ( rmsf_u_plus_nonRL[j] - rmsf_u_plus_ref[j])**2 ) * dy
    L2_error_rmsf_v_plus_nonRL += ( ( rmsf_v_plus_nonRL[j] - rmsf_v_plus_ref[j])**2 ) * dy
    L2_error_rmsf_w_plus_nonRL += ( ( rmsf_w_plus_nonRL[j] - rmsf_w_plus_ref[j])**2 ) * dy
L2_error_avg_u_plus_RL     = np.sqrt( L2_error_avg_u_plus_RL / ylength )      
L2_error_rmsf_u_plus_RL    = np.sqrt( L2_error_rmsf_u_plus_RL / ylength )      
L2_error_rmsf_v_plus_RL    = np.sqrt( L2_error_rmsf_v_plus_RL / ylength )      
L2_error_rmsf_w_plus_RL    = np.sqrt( L2_error_rmsf_w_plus_RL / ylength )      
L2_error_avg_u_plus_nonRL  = np.sqrt( L2_error_avg_u_plus_nonRL / ylength )          
L2_error_rmsf_u_plus_nonRL = np.sqrt( L2_error_rmsf_u_plus_nonRL / ylength )          
L2_error_rmsf_v_plus_nonRL = np.sqrt( L2_error_rmsf_v_plus_nonRL / ylength )          
L2_error_rmsf_w_plus_nonRL = np.sqrt( L2_error_rmsf_w_plus_nonRL / ylength )          

# Linf Error
Linf_error_avg_u_plus_RL  = np.zeros(n_RL)
Linf_error_rmsf_u_plus_RL = np.zeros(n_RL)
Linf_error_rmsf_v_plus_RL = np.zeros(n_RL)
Linf_error_rmsf_w_plus_RL = np.zeros(n_RL)
for i_RL in range(n_RL):
    Linf_error_avg_u_plus_RL[i_RL]  = np.max(abs_error_avg_u_plus_RL[i_RL,:])
    Linf_error_rmsf_u_plus_RL[i_RL] = np.max(abs_error_rmsf_u_plus_RL[i_RL,:])
    Linf_error_rmsf_v_plus_RL[i_RL] = np.max(abs_error_rmsf_v_plus_RL[i_RL,:])
    Linf_error_rmsf_w_plus_RL[i_RL] = np.max(abs_error_rmsf_w_plus_RL[i_RL,:])
Linf_error_avg_u_plus_nonRL  = np.max(abs_error_avg_u_plus_nonRL)
Linf_error_rmsf_u_plus_nonRL = np.max(abs_error_rmsf_u_plus_nonRL)
Linf_error_rmsf_v_plus_nonRL = np.max(abs_error_rmsf_v_plus_nonRL)
Linf_error_rmsf_w_plus_nonRL = np.max(abs_error_rmsf_w_plus_nonRL)
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
plt.semilogy( step_num_list, L1_error_avg_u_plus_nonRL*np.ones(n_RL), linestyle = '-',             linewidth = 1, color = plt.cm.tab10(0), label = 'u-avg, RHEA Uncontrolled' )
plt.semilogy( step_num_list, L1_error_avg_u_plus_RL, linestyle=':', marker = '^', markersize = 2,  linewidth = 1, color = plt.cm.tab10(0), label = 'u-avg, RHEA RL Framework' )
plt.semilogy( step_num_list, L1_error_rmsf_u_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(1), label = 'u-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, L1_error_rmsf_u_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(1), label = 'u-rmsf, RHEA RL Framework' )
plt.semilogy( step_num_list, L1_error_rmsf_v_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(2), label = 'v-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, L1_error_rmsf_v_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(2), label = 'v-rmsf, RHEA RL Framework' )
plt.semilogy( step_num_list, L1_error_rmsf_w_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(3), label = 'w-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, L1_error_rmsf_w_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(3), label = 'w-rmsf, RHEA RL Framework' )
plt.xlabel( 'Training step' )
plt.ylabel( 'L1 Error' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/L1_error_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

# L2-Error plot
plt.semilogy( step_num_list, L2_error_avg_u_plus_nonRL*np.ones(n_RL), linestyle = '-',             linewidth = 1, color = plt.cm.tab10(0), label = 'u-avg, RHEA Uncontrolled' )
plt.semilogy( step_num_list, L2_error_avg_u_plus_RL, linestyle=':', marker = '^', markersize = 2,  linewidth = 1, color = plt.cm.tab10(0), label = 'u-avg, RHEA RL Framework' )
plt.semilogy( step_num_list, L2_error_rmsf_u_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(1), label = 'u-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, L2_error_rmsf_u_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(1), label = 'u-rmsf, RHEA RL Framework' )
plt.semilogy( step_num_list, L2_error_rmsf_v_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(2), label = 'v-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, L2_error_rmsf_v_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(2), label = 'v-rmsf, RHEA RL Framework' )
plt.semilogy( step_num_list, L2_error_rmsf_w_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(3), label = 'w-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, L2_error_rmsf_w_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(3), label = 'w-rmsf, RHEA RL Framework' )
plt.xlabel( 'Training step' )
plt.ylabel( 'L2 Error' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/L2_error_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

# L1-Error plot
plt.semilogy( step_num_list, Linf_error_avg_u_plus_nonRL*np.ones(n_RL), linestyle = '-',             linewidth = 1, color = plt.cm.tab10(0), label = 'u-avg, RHEA Uncontrolled' )
plt.semilogy( step_num_list, Linf_error_avg_u_plus_RL, linestyle=':', marker = '^', markersize = 2,  linewidth = 1, color = plt.cm.tab10(0), label = 'u-avg, RHEA RL Framework' )
plt.semilogy( step_num_list, Linf_error_rmsf_u_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(1), label = 'u-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, Linf_error_rmsf_u_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(1), label = 'u-rmsf, RHEA RL Framework' )
plt.semilogy( step_num_list, Linf_error_rmsf_v_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(2), label = 'v-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, Linf_error_rmsf_v_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(2), label = 'v-rmsf, RHEA RL Framework' )
plt.semilogy( step_num_list, Linf_error_rmsf_w_plus_nonRL*np.ones(n_RL), linestyle = '-',            linewidth = 1, color = plt.cm.tab10(3), label = 'w-rmsf, RHEA Uncontrolled' )
plt.semilogy( step_num_list, Linf_error_rmsf_w_plus_RL, linestyle=':', marker = '^', markersize = 2, linewidth = 1, color = plt.cm.tab10(3), label = 'w-rmsf, RHEA RL Framework' )
plt.xlabel( 'Training step' )
plt.ylabel( 'Linf Error' )
legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.tick_params( axis = 'both', pad = 7.5 )
filename = f'{postDir}/Linf_error_{iteration}_ensemble{ensemble}.jpg'
plt.savefig( filename, format = 'jpg', dpi=600, bbox_inches = 'tight' )
plt.clf()
print(f"Build plot: '{filename}'")

print("Error plots built successfully!")