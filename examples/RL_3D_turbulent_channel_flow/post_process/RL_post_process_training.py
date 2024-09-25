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

### Filename to postprocess
# Check if the filename argument is provided
if len(sys.argv) != 3:
    print("Usage: python3 RL_post_process_plot_script.py <iteration> <ensemble>")
    sys.exit(1)
# Get 'iteration' from the command line argument
iteration = sys.argv[1]
ensemble  = sys.argv[2]

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
# Check if files exists
for filename_RL in filename_RL_list:
    if not os.path.isfile(filename_RL):
        print(f"Error: File '{filename_RL}' not found.")
        sys.exit(1)
if not os.path.isfile(filename_nonRL):
    print(f"Error: File '{filename_nonRL}' not found.")
    sys.exit(1)

### Open data files
# RL data
n_RL = len(filename_RL_list)
for i_RL in range(n_RL):
    filename_RL = filename_RL_list[i_RL]
    with h5py.File( filename_RL, 'r' ) as data_file:
        #list( data_file.keys() )
        y_data_RL_aux      = data_file['y'][:,:,:]
        avg_u_data_RL_aux  = data_file['avg_u'][:,:,:]
        avg_v_data_RL_aux  = data_file['avg_v'][:,:,:]
        avg_w_data_RL_aux  = data_file['avg_w'][:,:,:]
        rmsf_u_data_RL_aux = data_file['rmsf_u'][:,:,:]
        rmsf_v_data_RL_aux = data_file['rmsf_v'][:,:,:]
        rmsf_w_data_RL_aux = data_file['rmsf_w'][:,:,:]
        if i_RL == 0:
            num_points_x   = avg_u_data_RL_aux[0,0,:].size
            num_points_y   = avg_u_data_RL_aux[0,:,0].size
            num_points_z   = avg_u_data_RL_aux[:,0,0].size
            num_points_xz  = num_points_x*num_points_z
    # Initialize allocation arrays
    if i_RL == 0:
        y_data_RL       = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_u_data_RL   = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_v_data_RL   = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_w_data_RL   = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        rmsf_u_data_RL  = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])            
        rmsf_v_data_RL  = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])            
        rmsf_w_data_RL  = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])  
    # Fill allocation arrays
    y_data_RL[i_RL,:,:,:]      = y_data_RL_aux
    avg_u_data_RL[i_RL,:,:,:]  = avg_u_data_RL_aux
    avg_v_data_RL[i_RL,:,:,:]  = avg_v_data_RL_aux
    avg_w_data_RL[i_RL,:,:,:]  = avg_w_data_RL_aux
    rmsf_u_data_RL[i_RL,:,:,:] = rmsf_u_data_RL_aux
    rmsf_v_data_RL[i_RL,:,:,:] = rmsf_v_data_RL_aux
    rmsf_w_data_RL[i_RL,:,:,:] = rmsf_w_data_RL_aux
# non-RL data
with h5py.File( filename_nonRL, 'r' ) as data_file:
    y_data_nonRL      = data_file['y'][:,:,:]
    avg_u_data_nonRL  = data_file['avg_u'][:,:,:]
    avg_v_data_nonRL  = data_file['avg_v'][:,:,:]
    avg_w_data_nonRL  = data_file['avg_w'][:,:,:]
    rmsf_u_data_nonRL = data_file['rmsf_u'][:,:,:]
    rmsf_v_data_nonRL = data_file['rmsf_v'][:,:,:]
    rmsf_w_data_nonRL = data_file['rmsf_w'][:,:,:]

### Open reference solution file
y_plus_ref, u_plus_ref, rmsf_uu_plus_ref, rmsf_vv_plus_ref, rmsf_ww_plus_ref = np.loadtxt( 'reference_data/reference_solution.csv', delimiter=',', unpack = 'True' )

### Reference parameters
rho_0  = 1.0				# Reference density [kg/m3]
u_tau  = 1.0				# Friction velocity [m/s]
delta  = 1.0				# Channel half-height [m]
Re_tau = 180.0				# Friction Reynolds number [-]
mu_ref = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref = mu_ref/rho_0			# Kinematic viscosity [m2/s]


### Allocate averaged variables
avg_y_plus_RL  = np.zeros( [n_RL, int( 0.5*num_points_y )] ); avg_y_plus_nonRL  = np.zeros( int( 0.5*num_points_y ) )
avg_u_plus_RL  = np.zeros( [n_RL, int( 0.5*num_points_y )] ); avg_u_plus_nonRL  = np.zeros( int( 0.5*num_points_y ) )
rmsf_u_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_u_plus_nonRL = np.zeros( int( 0.5*num_points_y ) )
rmsf_v_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_v_plus_nonRL = np.zeros( int( 0.5*num_points_y ) )
rmsf_w_plus_RL = np.zeros( [n_RL, int( 0.5*num_points_y )] ); rmsf_w_plus_nonRL = np.zeros( int( 0.5*num_points_y ) )


### Average variables in space
for j in range( 0, num_points_y ):
    for i in range( 0, num_points_x ):
        for k in range( 0, num_points_z ):
            aux_j = j
            if( j > ( int( 0.5*num_points_y ) - 1 ) ):
                aux_j = num_points_y - j - 1
                for i_RL in range(n_RL):
                    avg_y_plus_RL[i_RL,aux_j]  += ( 0.5/num_points_xz )*y_data_RL[i_RL,k,aux_j,i]*( u_tau/nu_ref )
                    avg_u_plus_RL[i_RL,aux_j]  += ( 0.5/num_points_xz )*avg_u_data_RL[i_RL,k,j,i]*( 1.0/u_tau )
                    rmsf_u_plus_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_RL[i_RL,k,j,i]*( 1.0/u_tau )
                    rmsf_v_plus_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_RL[i_RL,k,j,i]*( 1.0/u_tau )
                    rmsf_w_plus_RL[i_RL,aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_RL[i_RL,k,j,i]*( 1.0/u_tau )
                avg_y_plus_nonRL[aux_j]  += ( 0.5/num_points_xz )*y_data_nonRL[k,aux_j,i]*( u_tau/nu_ref )
                avg_u_plus_nonRL[aux_j]  += ( 0.5/num_points_xz )*avg_u_data_nonRL[k,j,i]*( 1.0/u_tau )
                rmsf_u_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_u_data_nonRL[k,j,i]*( 1.0/u_tau )
                rmsf_v_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_v_data_nonRL[k,j,i]*( 1.0/u_tau )
                rmsf_w_plus_nonRL[aux_j] += ( 0.5/num_points_xz )*rmsf_w_data_nonRL[k,j,i]*( 1.0/u_tau )

### Plot u+ vs. y+
# Clear plot
plt.clf()
# Read & Plot data
plt.plot( y_plus_ref, u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = 'Moser et al., Re_tau = 180' )
for i_RL in range(n_RL):
    plt.scatter( avg_y_plus_RL[i_RL], avg_u_plus_RL[i_RL], marker = 'p', s = 50, zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
plt.scatter( avg_y_plus_nonRL, avg_u_plus_nonRL, marker = 'v', s = 50, color = 'blue', zorder = 1, label = 'RHEA non-RL' )
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
plt.savefig( f'train_u_plus_vs_y_plus_{iteration}_ensemble{ensemble}_{file_details}.eps', format = 'eps', bbox_inches = 'tight' )
# Clear plot
plt.clf()

### Plot u-rmsf 
# Read & Plot data
plt.plot( y_plus_ref, rmsf_uu_plus_ref, linestyle = '-', linewidth = 1, color = 'black',     zorder = 0 )
for i_RL in range(n_RL):
    plt.scatter( avg_y_plus_RL[i_RL], rmsf_u_plus_RL[i_RL], marker = 'p', s = 50, zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
plt.scatter( avg_y_plus_nonRL, rmsf_u_plus_nonRL, marker = 'v', s = 50, color = 'blue', zorder = 1, label = 'RHEA non-RL' )
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
plt.savefig( f'train_u_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}_{file_details}.eps', format = 'eps', bbox_inches = 'tight' )
# Clear plot
plt.clf()

### Plot v-rmsf
# Read & Plot data
plt.plot( y_plus_ref, rmsf_vv_plus_ref, linestyle = '-', linewidth = 1, color = 'black',     zorder = 0 )
for i_RL in range(n_RL):
    plt.scatter( avg_y_plus_RL[i_RL], rmsf_v_plus_RL[i_RL], marker = 'p', s = 50, zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
plt.scatter( avg_y_plus_nonRL, rmsf_v_plus_nonRL, marker = 'v', s = 50, color = 'blue', zorder = 1 )
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
plt.savefig( f'train_v_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}_{file_details}.eps', format = 'eps', bbox_inches = 'tight' )
# Clear plot
plt.clf()

### Plot w-rmsf
# Read & Plot data
plt.plot( y_plus_ref, rmsf_ww_plus_ref, linestyle = '-', linewidth = 1, color = 'black',     zorder = 0 )
for i_RL in range(n_RL):
    plt.scatter( avg_y_plus_RL[i_RL], rmsf_w_plus_RL[i_RL], marker = 'p', s = 50, zorder = 1, label = f'RHEA RL {file_details_list[i_RL]}' )
plt.scatter( avg_y_plus_nonRL, rmsf_w_plus_nonRL, marker = 'v', s = 50, color = 'blue', zorder = 1 )
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
plt.savefig( f'train_w_rms_plus_vs_y_plus_{iteration}_ensemble{ensemble}_{file_details}.eps', format = 'eps', bbox_inches = 'tight' )
# Clear plot
plt.clf()


############################# RL & non-RL Errors w.r.t. Reference #############################

from scipy.interpolate import interp1d

### Data interpolation

# Interpolation functions
interp_RL = []
for i_RL in range(n_RL):
    interp_RL.append(interp1d(avg_y_plus_RL[i_RL,:], avg_u_plus_RL[i_RL,:], fill_value="extrapolate"))
interp_nonRL = interp1d(avg_y_plus_nonRL, avg_u_plus_nonRL, fill_value="extrapolate")

# Interpolated values at reference y_plus coordinates
interp_avg_u_plus_RL = np.zeros( [n_RL, y_plus_ref.size ] )
for i_RL in range(n_RL):
    interp_avg_u_plus_RL[i_RL] = interp_RL[i_RL](y_plus_ref)
interp_avg_u_plus_nonRL = interp_nonRL(y_plus_ref)

### Errors Calculation

# Absolute error 
abs_error_avg_u_plus_RL = np.zeros( [n_RL, y_plus_ref.size ] )
for i_RL in range(n_RL):
    abs_error_avg_u_plus_RL[i_RL,:] = np.abs( interp_avg_u_plus_RL[i_RL,:] - u_plus_ref )
abs_error_avg_u_plus_nonRL  = np.abs( interp_avg_u_plus_nonRL - u_plus_ref )

# L1 Error
L1_error_RL = np.zeros(n_RL)
for i_RL in range(n_RL):
    L1_error_RL[i_RL] = np.sum(abs_error_avg_u_plus_RL[i_RL,:])
L1_error_nonRL = np.sum(abs_error_avg_u_plus_nonRL)

# L2 Error (RMS Error)
L2_error_RL = np.zeros(n_RL)
for i_RL in range(n_RL):
    L2_error_RL[i_RL] = np.sqrt( np.sum( (interp_avg_u_plus_RL[i_RL,:] - u_plus_ref)**2 ) )
L2_error_nonRL = np.sqrt( np.sum( (interp_avg_u_plus_nonRL - u_plus_ref)**2 ) )

# Linf Error
Linf_error_RL = np.zeros(n_RL)
for i_RL in range(n_RL):
    Linf_error_RL[i_RL] = np.max(abs_error_avg_u_plus_RL[i_RL,:])
Linf_error_nonRL = np.max(abs_error_avg_u_plus_nonRL)

### Errors logging

print("\nL1 Error RL:", L1_error_RL)
print("L1 Error nonRL:", L1_error_nonRL)
print("\nL2 Error RL (RMS):", L2_error_RL)
print("L2 Error nonRL (RMS):", L2_error_nonRL)
print("\nLinf Error RL:", Linf_error_RL)
print("Linf Error nonRL:", Linf_error_nonRL)