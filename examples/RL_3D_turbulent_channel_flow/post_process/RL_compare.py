#!/usr/bin/python

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
if len(sys.argv) != 2:
    print("Usage: python3 RL_post_process_plot_script.py <iteration>")
    sys.exit(1)
# Get 'iteration' from the command line argument
iteration    = sys.argv[1]

### Filenames of non-converged RL and non-RL
# Get 'file_details' & filename_RL
pattern = f"../rhea_exp/output_data/RL_3d_turbulent_channel_flow_{iteration}_*.h5"
# Use glob to find all matching files
matching_files = glob.glob(pattern)
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
for i in range(n_RL):
    filename_RL = filename_RL_list[i]
    with h5py.File( filename_RL, 'r' ) as data_file:
        #list( data_file.keys() )
        y_data_RL_aux      = data_file['y'][:,:,:]
        avg_u_data_RL_aux  = data_file['avg_u'][:,:,:]
        avg_v_data_RL_aux  = data_file['avg_v'][:,:,:]
        avg_w_data_RL_aux  = data_file['avg_w'][:,:,:]
        rmsf_u_data_RL_aux = data_file['rmsf_u'][:,:,:]
        rmsf_v_data_RL_aux = data_file['rmsf_v'][:,:,:]
        rmsf_w_data_RL_aux = data_file['rmsf_w'][:,:,:]
        if i == 0:
            num_points_x   = avg_u_data_RL[0,0,:].size
            num_points_y   = avg_u_data_RL[0,:,0].size
            num_points_z   = avg_u_data_RL[:,0,0].size
            num_points_xz  = num_points_x*num_points_z
    # Initialize allocation arrays
    if i == 0:
        y_data_RL       = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_u_data_RL   = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_v_data_RL   = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        avg_w_data_RL   = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])         
        rmsf_u_data_RL  = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])            
        rmsf_v_data_RL  = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])            
        rmsf_w_data_RL  = np.zeros([n_RL, num_points_z, num_points_y, num_points_x])  
    y_data_RL[i,:,:,:]      = y_data_RL_aux
    avg_u_data_RL[i,:,:,:]  = avg_u_data_RL_aux
    avg_v_data_RL[i,:,:,:]  = avg_v_data_RL_aux
    avg_w_data_RL[i,:,:,:]  = avg_w_data_RL_aux
    rmsf_u_data_RL[i,:,:,:] = rmsf_u_data_RL_aux
    rmsf_v_data_RL[i,:,:,:] = rmsf_v_data_RL_aux
    rmsf_w_data_RL[i,:,:,:] = rmsf_w_data_RL_aux
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

#### TODO: continue from here!
### Plot u+ vs. y+

# Clear plot
plt.clf()

# Read & Plot data
plt.plot( y_plus_ref, u_plus_ref, linestyle = '-', linewidth = 1, color = 'black', zorder = 0, label = 'Moser et al., Re_tau = 180' )
plt.scatter( avg_y_plus_RL,    avg_u_plus_RL,    marker = 'p', s = 50, color = 'firebrick', zorder = 1, label = 'RHEA RL' )
plt.scatter( avg_y_plus_nonRL, avg_u_plus_nonRL, marker = 'v', s = 50, color = 'blue',      zorder = 1, label = 'RHEA non-RL' )

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
plt.savefig( f'u_plus_vs_y_plus_{iteration}_{file_details}.eps', format = 'eps', bbox_inches = 'tight' )


### Plot u_rms+, v_rms+, w-rms+ vs. y+

# Clear plot
plt.clf()

# Read & Plot data
plt.plot( y_plus_ref, rmsf_uu_plus_ref, linestyle = '-', linewidth = 1, color = 'black',     zorder = 0 )
plt.plot( y_plus_ref, rmsf_vv_plus_ref, linestyle = '-', linewidth = 1, color = 'black',     zorder = 0 )
plt.plot( y_plus_ref, rmsf_ww_plus_ref, linestyle = '-', linewidth = 1, color = 'black',     zorder = 0 )
plt.scatter( avg_y_plus_RL,    rmsf_u_plus_RL,    marker = 'p', s = 50, color = 'firebrick', zorder = 1 )
plt.scatter( avg_y_plus_RL,    rmsf_v_plus_RL,    marker = 'p', s = 50, color = 'firebrick', zorder = 1 )
plt.scatter( avg_y_plus_RL,    rmsf_w_plus_RL,    marker = 'p', s = 50, color = 'firebrick', zorder = 1 )
plt.scatter( avg_y_plus_nonRL, rmsf_u_plus_nonRL, marker = 'v', s = 50, color = 'blue',      zorder = 1 )
plt.scatter( avg_y_plus_nonRL, rmsf_v_plus_nonRL, marker = 'v', s = 50, color = 'blue',      zorder = 1 )
plt.scatter( avg_y_plus_nonRL, rmsf_w_plus_nonRL, marker = 'v', s = 50, color = 'blue',      zorder = 1 )

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
plt.ylabel( 'u_rms+, v_rms+, w_rms+' )
#legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left' )
plt.text( 1.05, 1.0, 'u_rms+' )
plt.text( 17.5, 0.2, 'v_rms+' )
plt.text( 4.00, 0.9, 'w_rms+' )
plt.tick_params( axis = 'both', pad = 7.5 )
plt.savefig( f'uvw_rms_plus_vs_y_plus_{iteration}_{file_details}.eps', format = 'eps', bbox_inches = 'tight' )
