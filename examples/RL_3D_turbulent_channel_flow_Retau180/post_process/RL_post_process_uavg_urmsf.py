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
Re_tau  = 180.0				# Friction Reynolds number [-]
mu_ref  = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref  = mu_ref/rho_0			# Kinematic viscosity [m2/s]
dt_phys = 5.0e-5

# --- Get CASE parameters ---

try :
    iteration  = sys.argv[1]
    ensemble   = sys.argv[2]
    train_name = sys.argv[3]
    print(f"Script parameters: \n- Iteration: {iteration} \n- Ensemble: {ensemble}\n- Train name: {train_name}")
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
        # Print the file and the extracted part
        print(f"Filename: {base_filename}, File details: {file_details}")
else:
    print(f"No files found matching the pattern: {pattern}")
### > non-RL filename
# Build filename_nonRL
filename_nonRL = f"reference_data/3d_turbulent_channel_flow_{iteration}.h5"

# Check if RL & non-RL files exists
for filename_RL in filename_RL_list:
    if not os.path.isfile(filename_RL):
        print(f"Error: File '{filename_RL}' not found.")
        sys.exit(1)
if not os.path.isfile(filename_nonRL):
    print(f"Error: File '{filename_nonRL}' not found.")
    sys.exit(1)

### Open data files
print("\nImporting data from files...")
# RL data
n_RL = len(filename_RL_list)
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
    # Check same averaging time
    if not np.isclose(averaging_time_RL, averaging_time_RL_aux, atol=1e-8):
        raise ValueError("Averaging time should be equal for all RL h5 files")
    # Fill allocation arrays
    y_data_RL[i_RL,:,:,:]      = y_data_RL_aux
    avg_u_data_RL[i_RL,:,:,:]  = avg_u_data_RL_aux
    avg_v_data_RL[i_RL,:,:,:]  = avg_v_data_RL_aux
    avg_w_data_RL[i_RL,:,:,:]  = avg_w_data_RL_aux
    rmsf_u_data_RL[i_RL,:,:,:] = rmsf_u_data_RL_aux
    rmsf_v_data_RL[i_RL,:,:,:] = rmsf_v_data_RL_aux
    rmsf_w_data_RL[i_RL,:,:,:] = rmsf_w_data_RL_aux
    print(f"RL non-converged data imported from file '{filename_RL}'")
# non-RL data
with h5py.File( filename_nonRL, 'r' ) as data_file:
    averaging_time_nonRL = data_file.attrs["AveragingTime"][0]
    y_data_nonRL      = data_file['y'][1:-1,1:-1,1:-1]
    avg_u_data_nonRL  = data_file['avg_u'][1:-1,1:-1,1:-1]
    avg_v_data_nonRL  = data_file['avg_v'][1:-1,1:-1,1:-1]
    avg_w_data_nonRL  = data_file['avg_w'][1:-1,1:-1,1:-1]
    rmsf_u_data_nonRL = data_file['rmsf_u'][1:-1,1:-1,1:-1]
    rmsf_v_data_nonRL = data_file['rmsf_v'][1:-1,1:-1,1:-1]
    rmsf_w_data_nonRL = data_file['rmsf_w'][1:-1,1:-1,1:-1]
# Check same averaging time for RL & non-RL
if not np.isclose(averaging_time_RL, averaging_time_nonRL, atol=1e-8):
    raise ValueError(f"Averaging time should be equal for both RL & non-RL h5 files, while RL: {averaging_time_RL} != nonRL: {averaging_time_nonRL}")
else:
    averaging_time_nonConv = averaging_time_RL
print(f"Non-RL non-converged data imported from file '{filename_nonRL}'")

### Open reference solution file
filename_ref = 'reference_data/reference_solution.csv'
y_plus_ref, u_plus_ref, rmsf_uu_plus_ref, rmsf_vv_plus_ref, rmsf_ww_plus_ref = np.loadtxt( filename_ref, delimiter=',', unpack = 'True' )
print(f"Non-RL converged reference data imported from file '{filename_ref}'")
print("Data imported successfully!")


### Allocate averaged variables
y_plus_RL      = np.zeros( [n_RL, int( 0.5*num_points_y )] ); y_plus_nonRL      = np.zeros( int( 0.5*num_points_y ) )
avg_u_plus_RL  = np.zeros( [n_RL, int( 0.5*num_points_y )] ); avg_u_plus_nonRL  = np.zeros( int( 0.5*num_points_y ) )
rmsf_u_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_u_plus_nonRL = np.zeros( int( 0.5*num_points_y ) )
rmsf_v_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_v_plus_nonRL = np.zeros( int( 0.5*num_points_y ) )
rmsf_w_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_w_plus_nonRL = np.zeros( int( 0.5*num_points_y ) )


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
            # non-RL data:
            if( j > ( int( 0.5*num_points_y ) - 1 ) ):  # top-wall
                y_plus_nonRL[aux_j]  += ( 0.5/num_points_xz )*(2.0 - y_data_nonRL[k,j,i])*( u_tau/nu_ref )
            else:                                       # bottom-wall
                y_plus_nonRL[aux_j]  += ( 0.5/num_points_xz )*y_data_nonRL[k,j,i]*( u_tau/nu_ref )
            avg_u_plus_nonRL[aux_j]  += ( 0.5/num_points_xz )*avg_u_data_nonRL[k,j,i]*( 1.0/u_tau )
            rmsf_u_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_nonRL[k,j,i]*( 1.0/u_tau )
            rmsf_v_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_nonRL[k,j,i]*( 1.0/u_tau )
            rmsf_w_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_nonRL[k,j,i]*( 1.0/u_tau )
print("Variables averaged successfully!")


### Calculate TKE averaged profile from rmsf_u,v,w averaged profiles 
TKE_RL    = 0.5 * ( rmsf_u_plus_RL**2 + rmsf_v_plus_RL**2 + rmsf_w_plus_RL**2 )
TKE_nonRL = 0.5 * ( rmsf_u_plus_nonRL**2 + rmsf_v_plus_nonRL**2 + rmsf_w_plus_nonRL**2 )
TKE_ref   = 0.5 * ( rmsf_uu_plus_ref**2 + rmsf_vv_plus_ref**2 + rmsf_ww_plus_ref**2 )

### Plot u+ vs. y+
print("\nBuilding plots...")
# Clear plot
plt.clf()
# Read & Plot data
plt.plot( y_plus_ref, u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = 'Moser et al., Re_tau = 180' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], avg_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
    else:
        plt.plot( y_plus_RL[i_RL], avg_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, avg_u_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = 'RHEA non-RL' )
# Configure plot
plt.xlim( 1.0e-1, 2.0e2 )
plt.xticks( np.arange( 1.0e-1, 2.01e2, 1.0 ) )
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
plt.plot( y_plus_ref, rmsf_uu_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label='Moser et al., Re_tau = 180' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], rmsf_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
    else:
        plt.plot( y_plus_RL[i_RL], rmsf_u_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, rmsf_u_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = 'RHEA non-RL' )
# Configure plot
plt.xlim( 1.0e-1, 2.0e2 )
plt.xticks( np.arange( 1.0e-1, 2.01e2, 1.0 ) )
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
plt.plot( y_plus_ref, rmsf_vv_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = 'Moser et al., Re_tau = 180' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], rmsf_v_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
    else:
        plt.plot( y_plus_RL[i_RL], rmsf_v_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, rmsf_v_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = 'RHEA non-RL'  )
# Configure plot
plt.xlim( 1.0e-1, 2.0e2 )
plt.xticks( np.arange( 1.0e-1, 2.01e2, 1.0 ) )
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
plt.plot( y_plus_ref, rmsf_ww_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = 'Moser et al., Re_tau = 180' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], rmsf_w_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
    else:
        plt.plot( y_plus_RL[i_RL], rmsf_w_plus_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, rmsf_w_plus_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = 'RHEA non-RL'  )
# Configure plot
plt.xlim( 1.0e-1, 2.0e2 )
plt.xticks( np.arange( 1.0e-1, 2.01e2, 1.0 ) )
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
plt.plot( y_plus_ref, TKE_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = 'Moser et al., Re_tau = 180' )
for i_RL in range(n_RL):
    if n_RL < 10:
        plt.plot( y_plus_RL[i_RL], TKE_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
    else:
        plt.plot( y_plus_RL[i_RL], TKE_RL[i_RL], linestyle='-', marker = '^', markersize = 2,  zorder = 1 )
plt.plot( y_plus_nonRL, TKE_nonRL, linestyle='-', marker = 'v', markersize = 2,  color = 'blue', zorder = 1, label = 'RHEA non-RL'  )
# Configure plot
plt.xlim( 1.0e-1, 2.0e2 )
plt.xticks( np.arange( 1.0e-1, 2.01e2, 1.0 ) )
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

from scipy.interpolate import interp1d

# ------------------- Data interpolation -------------------

print("\nInterpolating data...")
# --- Interpolation functions ---
func_interp_avg_u_plus_RL = []; func_interp_rmsf_u_plus_RL = []; func_interp_rmsf_v_plus_RL = []; func_interp_rmsf_w_plus_RL = []; 
for i_RL in range(n_RL):
    func_interp_avg_u_plus_RL.append( interp1d(y_plus_RL[i_RL,:], avg_u_plus_RL[i_RL,:],  fill_value="extrapolate"))
    func_interp_rmsf_u_plus_RL.append(interp1d(y_plus_RL[i_RL,:], rmsf_u_plus_RL[i_RL,:], fill_value="extrapolate"))
    func_interp_rmsf_v_plus_RL.append(interp1d(y_plus_RL[i_RL,:], rmsf_v_plus_RL[i_RL,:], fill_value="extrapolate"))
    func_interp_rmsf_w_plus_RL.append(interp1d(y_plus_RL[i_RL,:], rmsf_w_plus_RL[i_RL,:], fill_value="extrapolate"))
func_interp_avg_u_plus_nonRL  = interp1d(y_plus_nonRL, avg_u_plus_nonRL,  fill_value="extrapolate")
func_interp_rmsf_u_plus_nonRL = interp1d(y_plus_nonRL, rmsf_u_plus_nonRL, fill_value="extrapolate")
func_interp_rmsf_v_plus_nonRL = interp1d(y_plus_nonRL, rmsf_v_plus_nonRL, fill_value="extrapolate")
func_interp_rmsf_w_plus_nonRL = interp1d(y_plus_nonRL, rmsf_w_plus_nonRL, fill_value="extrapolate")

# --- Interpolated values at reference y_plus coordinates ---
interp_avg_u_plus_RL  = np.zeros( [n_RL, y_plus_ref.size ] )
interp_rmsf_u_plus_RL = np.zeros( [n_RL, y_plus_ref.size ] )
interp_rmsf_v_plus_RL = np.zeros( [n_RL, y_plus_ref.size ] )
interp_rmsf_w_plus_RL = np.zeros( [n_RL, y_plus_ref.size ] )
for i_RL in range(n_RL):
    interp_avg_u_plus_RL[i_RL]  = func_interp_avg_u_plus_RL[i_RL](y_plus_ref)
    interp_rmsf_u_plus_RL[i_RL] = func_interp_rmsf_u_plus_RL[i_RL](y_plus_ref)
    interp_rmsf_v_plus_RL[i_RL] = func_interp_rmsf_v_plus_RL[i_RL](y_plus_ref)
    interp_rmsf_w_plus_RL[i_RL] = func_interp_rmsf_w_plus_RL[i_RL](y_plus_ref)
interp_avg_u_plus_nonRL  = func_interp_avg_u_plus_nonRL(y_plus_ref)
interp_rmsf_u_plus_nonRL = func_interp_rmsf_u_plus_nonRL(y_plus_ref)
interp_rmsf_v_plus_nonRL = func_interp_rmsf_v_plus_nonRL(y_plus_ref)
interp_rmsf_w_plus_nonRL = func_interp_rmsf_w_plus_nonRL(y_plus_ref)
print("Data interpolated successfully!")

# --- Calculate errors ---
print("\nCalculating errors...")
# Absolute error 
abs_error_avg_u_plus_RL  = np.zeros( [n_RL, y_plus_ref.size] )
abs_error_rmsf_u_plus_RL = np.zeros( [n_RL, y_plus_ref.size] )
abs_error_rmsf_v_plus_RL = np.zeros( [n_RL, y_plus_ref.size] )
abs_error_rmsf_w_plus_RL = np.zeros( [n_RL, y_plus_ref.size] )
for i_RL in range(n_RL):
    abs_error_avg_u_plus_RL[i_RL,:]  = np.abs( interp_avg_u_plus_RL[i_RL,:]  - u_plus_ref )
    abs_error_rmsf_u_plus_RL[i_RL,:] = np.abs( interp_rmsf_u_plus_RL[i_RL,:] - rmsf_uu_plus_ref )
    abs_error_rmsf_v_plus_RL[i_RL,:] = np.abs( interp_rmsf_v_plus_RL[i_RL,:] - rmsf_vv_plus_ref )
    abs_error_rmsf_w_plus_RL[i_RL,:] = np.abs( interp_rmsf_w_plus_RL[i_RL,:] - rmsf_ww_plus_ref )
abs_error_avg_u_plus_nonRL  = np.abs( interp_avg_u_plus_nonRL  - u_plus_ref )
abs_error_rmsf_u_plus_nonRL = np.abs( interp_rmsf_u_plus_nonRL - rmsf_uu_plus_ref )
abs_error_rmsf_v_plus_nonRL = np.abs( interp_rmsf_v_plus_nonRL - rmsf_vv_plus_ref )
abs_error_rmsf_w_plus_nonRL = np.abs( interp_rmsf_w_plus_nonRL - rmsf_ww_plus_ref )

# L1 Error
L1_error_avg_u_plus_RL  = np.zeros(n_RL)
L1_error_rmsf_u_plus_RL = np.zeros(n_RL)
L1_error_rmsf_v_plus_RL = np.zeros(n_RL)
L1_error_rmsf_w_plus_RL = np.zeros(n_RL)
for i_RL in range(n_RL):
    L1_error_avg_u_plus_RL[i_RL]  = np.sum( abs_error_avg_u_plus_RL[i_RL,:] )
    L1_error_rmsf_u_plus_RL[i_RL] = np.sum( abs_error_rmsf_u_plus_RL[i_RL,:] )
    L1_error_rmsf_v_plus_RL[i_RL] = np.sum( abs_error_rmsf_v_plus_RL[i_RL,:] )
    L1_error_rmsf_w_plus_RL[i_RL] = np.sum( abs_error_rmsf_w_plus_RL[i_RL,:] )
L1_error_avg_u_plus_nonRL  = np.sum(abs_error_avg_u_plus_nonRL)
L1_error_rmsf_u_plus_nonRL = np.sum(abs_error_rmsf_u_plus_nonRL)
L1_error_rmsf_v_plus_nonRL = np.sum(abs_error_rmsf_v_plus_nonRL)
L1_error_rmsf_w_plus_nonRL = np.sum(abs_error_rmsf_w_plus_nonRL)

# L2 Error (RMS Error)
L2_error_avg_u_plus_RL  = np.zeros(n_RL)
L2_error_rmsf_u_plus_RL = np.zeros(n_RL)
L2_error_rmsf_v_plus_RL = np.zeros(n_RL)
L2_error_rmsf_w_plus_RL = np.zeros(n_RL)
for i_RL in range(n_RL):
    L2_error_avg_u_plus_RL[i_RL]  = np.sqrt( np.sum( ( interp_avg_u_plus_RL[i_RL,:] - u_plus_ref )**2 ) )
    L2_error_rmsf_u_plus_RL[i_RL] = np.sqrt( np.sum( ( interp_rmsf_u_plus_RL[i_RL,:] - rmsf_uu_plus_ref )**2 ) )
    L2_error_rmsf_v_plus_RL[i_RL] = np.sqrt( np.sum( ( interp_rmsf_v_plus_RL[i_RL,:] - rmsf_vv_plus_ref )**2 ) )
    L2_error_rmsf_w_plus_RL[i_RL] = np.sqrt( np.sum( ( interp_rmsf_w_plus_RL[i_RL,:] - rmsf_ww_plus_ref )**2 ) )
L2_error_avg_u_plus_nonRL  = np.sqrt( np.sum( (interp_avg_u_plus_nonRL - u_plus_ref)**2 ) )
L2_error_rmsf_u_plus_nonRL = np.sqrt( np.sum( (interp_rmsf_u_plus_nonRL - rmsf_uu_plus_ref)**2 ) )
L2_error_rmsf_v_plus_nonRL = np.sqrt( np.sum( (interp_rmsf_v_plus_nonRL - rmsf_vv_plus_ref)**2 ) )
L2_error_rmsf_w_plus_nonRL = np.sqrt( np.sum( (interp_rmsf_w_plus_nonRL - rmsf_ww_plus_ref)**2 ) )

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
print(f"\nWritting errors in file '{error_log_filename}'")
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
print("Errors written succesfully!")
    
# Print error logs in terminal
with open(error_log_filename, "r") as file:
    content = file.read()
    print(content)