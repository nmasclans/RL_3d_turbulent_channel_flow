import sys
import os
import glob
import h5py    
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

sys.path.append('/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/post_process')
from utils import compute_reynolds_stress_dof

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 12, frameon = False)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')

#--------------------------------------------------------------------------------------------

verbose = False

# --- Case parameters ---

restart_iteration = 2820000
Re_tau  = 180
dt_phys = 5e-5

rho_0   = 1.0				# Reference density [kg/m3]
u_tau   = 1.0				# Friction velocity [m/s]
delta   = 1.0				# Channel half-height [m]
mu_ref  = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref  = mu_ref/rho_0			# Kinematic viscosity [m2/s]


#--------------------------------------------------------------------------------------------

# --- Training / Evaluation parameters ---

# Reference & non-RL data directory
filePath = os.path.dirname(os.path.abspath(__file__))
compareDatasetDir = os.path.join("/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/post_process", f"data_Retau{Re_tau:.0f}")

# Visualizer for building plots
postDir = f"/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/post_process/article_test/"

#--------------------------------------------------------------------------------------------

# ----------- Build data h5 filenames ------------

# --- non-RL converged reference filename ---
filename_ref = f"{compareDatasetDir}/3d_turbulent_channel_flow_reference.h5"

# --- non-RL restart data file
filename_rst = f"{compareDatasetDir}/3d_turbulent_channel_flow_{restart_iteration}.h5"

# --- RL filenames ---
filename_RL_list = [
    os.path.join(
        "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau180_eval_S/post_process/eval_2025-09-04--15-26-04--16ed_globalStep2560_oldRewardCoeff/rhea_exp/output_data",
        "RL_3d_turbulent_channel_flow_2830026_ensemble0_step002720.h5",
    ),
    os.path.join(
        "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/examples/RL_3D_turbulent_channel_flow_Retau180_eval_S/post_process/eval_2025-09-04--15-26-04--16ed_globalStep15040_oldRewardCoeff/rhea_exp/output_data",
        "RL_3d_turbulent_channel_flow_2830026_ensemble0_step015200.h5",
    ),
]
N_RL = len(filename_RL_list)

# --- RL filenames ---
filename_nonRL_list = [f"{compareDatasetDir}/3d_turbulent_channel_flow_2830000.h5"]
N_nonRL = len(filename_nonRL_list)

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

print("\nImporting data from RL files:\n")
for i in range(N_RL):
    filename_RL = filename_RL_list[i]
    with h5py.File( filename_RL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_RL_aux    = data_file.attrs["AveragingTime"][0] - dt_phys
        y_data_RL_aux            = data_file['y'][1:-1,1:-1,1:-1]
        favre_uffuff_data_RL_aux = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
        favre_uffvff_data_RL_aux = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
        favre_uffwff_data_RL_aux = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
        favre_vffvff_data_RL_aux = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
        favre_vffwff_data_RL_aux = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
        favre_wffwff_data_RL_aux = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = favre_uffuff_data_RL_aux[0,0,:].size
        num_points_y      = favre_uffuff_data_RL_aux[0,:,0].size
        num_points_z      = favre_uffuff_data_RL_aux[:,0,0].size
        num_points_xz     = num_points_x*num_points_z
        num_points_y_half = int(0.5*num_points_y)
        averaging_time_RL = np.zeros(N_RL)
        y_data_RL            = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        favre_uffuff_data_RL = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        favre_uffvff_data_RL = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        favre_uffwff_data_RL = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])         
        favre_vffvff_data_RL = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])            
        favre_vffwff_data_RL = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])            
        favre_wffwff_data_RL = np.zeros([N_RL, num_points_z, num_points_y, num_points_x])  
    # Fill allocation arrays
    averaging_time_RL[i]          = averaging_time_RL_aux
    y_data_RL[i,:,:,:]            = y_data_RL_aux
    favre_uffuff_data_RL[i,:,:,:] = favre_uffuff_data_RL_aux
    favre_uffvff_data_RL[i,:,:,:] = favre_uffvff_data_RL_aux
    favre_uffwff_data_RL[i,:,:,:] = favre_uffwff_data_RL_aux
    favre_vffvff_data_RL[i,:,:,:] = favre_vffvff_data_RL_aux
    favre_vffwff_data_RL[i,:,:,:] = favre_vffwff_data_RL_aux
    favre_wffwff_data_RL[i,:,:,:] = favre_wffwff_data_RL_aux
    # Logging
    print(f"RL non-converged data imported from file '{filename_RL}' - averaging time: {averaging_time_RL_aux:.6f}")
averaging_time_simulated_RL    = averaging_time_RL - averaging_time_RL[0]  # averaging_time_RL[0] is the restart file averaging time, t_avg_0
averaging_time_simulated_RL[0] = averaging_time_RL[0]
averaging_time_accum_RL        = np.cumsum(averaging_time_simulated_RL)

# --- Get non-RL (non-converged) data from h5 file ---
print("\nImporting data from non-RL files:\n")
for i in range(N_nonRL):
    filename_nonRL = filename_nonRL_list[i]
    with h5py.File( filename_nonRL, 'r' ) as data_file:
        #list( data_file.keys() )
        averaging_time_nonRL_aux    = data_file.attrs["AveragingTime"][0] - dt_phys
        y_data_nonRL_aux            = data_file['y'][1:-1,1:-1,1:-1]
        favre_uffuff_data_nonRL_aux = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
        favre_uffvff_data_nonRL_aux = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
        favre_uffwff_data_nonRL_aux = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
        favre_vffvff_data_nonRL_aux = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
        favre_vffwff_data_nonRL_aux = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
        favre_wffwff_data_nonRL_aux = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
    # Initialize allocation arrays
    if i == 0:
        num_points_x      = favre_uffuff_data_nonRL_aux[0,0,:].size
        num_points_y      = favre_uffuff_data_nonRL_aux[0,:,0].size
        num_points_z      = favre_uffuff_data_nonRL_aux[:,0,0].size
        num_points_xz     = num_points_x*num_points_z
        num_points_y_half = int(0.5*num_points_y)
        averaging_time_nonRL    = np.zeros(N_nonRL)
        y_data_nonRL            = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        favre_uffuff_data_nonRL = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        favre_uffvff_data_nonRL = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        favre_uffwff_data_nonRL = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])         
        favre_vffvff_data_nonRL = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])            
        favre_vffwff_data_nonRL = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])            
        favre_wffwff_data_nonRL = np.zeros([N_nonRL, num_points_z, num_points_y, num_points_x])  
    # Fill allocation arrays
    averaging_time_nonRL[i]          = averaging_time_nonRL_aux
    y_data_nonRL[i,:,:,:]            = y_data_nonRL_aux
    favre_uffuff_data_nonRL[i,:,:,:] = favre_uffuff_data_nonRL_aux
    favre_uffvff_data_nonRL[i,:,:,:] = favre_uffvff_data_nonRL_aux
    favre_uffwff_data_nonRL[i,:,:,:] = favre_uffwff_data_nonRL_aux
    favre_vffvff_data_nonRL[i,:,:,:] = favre_vffvff_data_nonRL_aux
    favre_vffwff_data_nonRL[i,:,:,:] = favre_vffwff_data_nonRL_aux
    favre_wffwff_data_nonRL[i,:,:,:] = favre_wffwff_data_nonRL_aux
    print(f"non-RL non-converged data imported from file '{filename_nonRL}' - averaging time: {averaging_time_nonRL_aux:.6f}")

# --- Get non-RL converged reference data from h5 file ---
print("\nImporting reference data (non-RL):\n")
with h5py.File( filename_ref, 'r' ) as data_file:
    averaging_time_ref    = data_file.attrs["AveragingTime"][0]
    y_data_ref            = data_file['y'][1:-1,1:-1,1:-1]
    favre_uffuff_data_ref = data_file['favre_uffuff'][1:-1,1:-1,1:-1]
    favre_uffvff_data_ref = data_file['favre_uffvff'][1:-1,1:-1,1:-1]
    favre_uffwff_data_ref = data_file['favre_uffwff'][1:-1,1:-1,1:-1]
    favre_vffvff_data_ref = data_file['favre_vffvff'][1:-1,1:-1,1:-1]
    favre_vffwff_data_ref = data_file['favre_vffwff'][1:-1,1:-1,1:-1]
    favre_wffwff_data_ref = data_file['favre_wffwff'][1:-1,1:-1,1:-1]
print(f"Non-RL reference data imported from file '{filename_ref}' - averaging time: {averaging_time_ref:.6f}")
print("\nData imported successfully!")

# -------------- Averaging fields using XZ symmetries --------------

print("\nAveraging fields in space...")

### Allocate averaged variables
y_plus_RL       = np.zeros([N_RL, num_points_y_half]);  
y_delta_RL      = np.zeros([N_RL, num_points_y_half]);  
favre_uffvff_RL = np.zeros([N_RL, num_points_y_half]);  
favre_uffuff_RL = np.zeros([N_RL, num_points_y_half]);  
favre_uffwff_RL = np.zeros([N_RL, num_points_y_half]);  
favre_vffvff_RL = np.zeros([N_RL, num_points_y_half]);  
favre_vffwff_RL = np.zeros([N_RL, num_points_y_half]);  
favre_wffwff_RL = np.zeros([N_RL, num_points_y_half]);  
y_plus_nonRL       = np.zeros([N_nonRL, num_points_y_half]);   
y_delta_nonRL      = np.zeros([N_nonRL, num_points_y_half]);   
favre_uffuff_nonRL = np.zeros([N_nonRL, num_points_y_half]);   
favre_uffvff_nonRL = np.zeros([N_nonRL, num_points_y_half]);   
favre_uffwff_nonRL = np.zeros([N_nonRL, num_points_y_half]);   
favre_vffvff_nonRL = np.zeros([N_nonRL, num_points_y_half]);   
favre_vffwff_nonRL = np.zeros([N_nonRL, num_points_y_half]);   
favre_wffwff_nonRL = np.zeros([N_nonRL, num_points_y_half]);   
y_plus_ref       = np.zeros(num_points_y_half)
y_delta_ref      = np.zeros(num_points_y_half)
favre_uffuff_ref = np.zeros(num_points_y_half)
favre_uffvff_ref = np.zeros(num_points_y_half)
favre_uffwff_ref = np.zeros(num_points_y_half)
favre_vffvff_ref = np.zeros(num_points_y_half)
favre_vffwff_ref = np.zeros(num_points_y_half)
favre_wffwff_ref = np.zeros(num_points_y_half)
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
            for n in range(N_RL):
                if is_half_top:
                    y_plus_RL[n,aux_j]   += ( 0.5/num_points_xz )*( 2*delta - y_data_RL[n,k,j,i] )*( u_tau/nu_ref )
                    y_delta_RL[n,aux_j]  += ( 0.5/num_points_xz )*( 2*delta - y_data_RL[n,k,j,i] )/delta
                else:
                    y_plus_RL[n,aux_j]   += ( 0.5/num_points_xz )*y_data_RL[n,k,j,i]*( u_tau/nu_ref )
                    y_delta_RL[n,aux_j]  += ( 0.5/num_points_xz )*y_data_RL[n,k,j,i]/delta
                favre_uffuff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_RL[n,k,j,i]
                favre_uffvff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_RL[n,k,j,i]
                favre_uffwff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_RL[n,k,j,i]
                favre_vffvff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_RL[n,k,j,i]
                favre_vffwff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_vffwff_data_RL[n,k,j,i]
                favre_wffwff_RL[n,aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_RL[n,k,j,i]
            # non-RL data, non-converged:
            for n in range(N_nonRL):
                if is_half_top:
                    y_plus_nonRL[n,aux_j]   += ( 0.5/num_points_xz )*(2.0*delta - y_data_nonRL[n,k,j,i])*(u_tau/nu_ref);    
                    y_delta_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*(2.0*delta - y_data_nonRL[n,k,j,i])/delta;             
                else:
                    y_plus_nonRL[n,aux_j]   += ( 0.5/num_points_xz )*y_data_nonRL[n,k,j,i]*( u_tau/nu_ref );    
                    y_delta_nonRL[n,aux_j]  += ( 0.5/num_points_xz )*y_data_nonRL[n,k,j,i]/delta;               
                favre_uffuff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_nonRL[n,k,j,i];          
                favre_uffvff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_nonRL[n,k,j,i];          
                favre_uffwff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_nonRL[n,k,j,i];          
                favre_vffvff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_nonRL[n,k,j,i];          
                favre_vffwff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_vffwff_data_nonRL[n,k,j,i];          
                favre_wffwff_nonRL[n,aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_nonRL[n,k,j,i];          
            # reference:
            if is_half_top:
                y_plus_ref[aux_j]  += ( 0.5/num_points_xz )*(2.0*delta - y_data_ref[k,j,i])*(u_tau/nu_ref);  
                y_delta_ref[aux_j] += ( 0.5/num_points_xz )*(2.0*delta - y_data_ref[k,j,i])/delta;             
            else:
                y_plus_ref[aux_j]       += ( 0.5/num_points_xz )*y_data_ref[k,j,i]*(u_tau/nu_ref)
                y_delta_ref[aux_j]      += ( 0.5/num_points_xz )*y_data_ref[k,j,i]/delta
            favre_uffuff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffuff_data_ref[k,j,i]
            favre_uffvff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffvff_data_ref[k,j,i]
            favre_uffwff_ref[aux_j] += ( 0.5/num_points_xz )*favre_uffwff_data_ref[k,j,i]
            favre_vffvff_ref[aux_j] += ( 0.5/num_points_xz )*favre_vffvff_data_ref[k,j,i]
            favre_wffwff_ref[aux_j] += ( 0.5/num_points_xz )*favre_wffwff_data_ref[k,j,i];          

print("Fields averaged successfully!")


# ----------- Decompose Rij into d.o.f --------------

print("\nDecomposing Rij into Rij dof...")
Rkk_RL     = np.zeros([N_RL, num_points_y_half]);    
lambda1_RL = np.zeros([N_RL, num_points_y_half]);    
lambda2_RL = np.zeros([N_RL, num_points_y_half]);    
lambda3_RL = np.zeros([N_RL, num_points_y_half]);    
xmap1_RL   = np.zeros([N_RL, num_points_y_half]);    
xmap2_RL   = np.zeros([N_RL, num_points_y_half]);    
eigval_RL  = np.zeros([N_RL, num_points_y_half,3]);  
Rkk_nonRL     = np.zeros([N_nonRL, num_points_y_half]);    
lambda1_nonRL = np.zeros([N_nonRL, num_points_y_half]);    
lambda2_nonRL = np.zeros([N_nonRL, num_points_y_half]);    
lambda3_nonRL = np.zeros([N_nonRL, num_points_y_half]);    
xmap1_nonRL   = np.zeros([N_nonRL, num_points_y_half]);    
xmap2_nonRL   = np.zeros([N_nonRL, num_points_y_half]);    
eigval_nonRL  = np.zeros([N_nonRL, num_points_y_half,3]);  
Rkk_ref     = np.zeros(num_points_y_half)
lambda1_ref = np.zeros(num_points_y_half)
lambda2_ref = np.zeros(num_points_y_half)
lambda3_ref = np.zeros(num_points_y_half)
xmap1_ref   = np.zeros(num_points_y_half)
xmap2_ref   = np.zeros(num_points_y_half)
eigval_ref  = np.zeros([num_points_y_half,3])

# RL data
for i in range(N_RL):
    ( Rkk_RL[i], lambda1_RL[i], lambda2_RL[i], lambda3_RL[i], xmap1_RL[i], xmap2_RL[i] ) \
        = compute_reynolds_stress_dof( favre_uffuff_RL[i], favre_uffvff_RL[i], favre_uffwff_RL[i], favre_vffvff_RL[i], favre_vffwff_RL[i], favre_wffwff_RL[i], verbose=verbose )
    eigval_RL[i,:,0] = lambda1_RL[i]
    eigval_RL[i,:,1] = lambda2_RL[i]
    eigval_RL[i,:,2] = lambda3_RL[i]  

# non-RL non-converged data
for i in range(N_nonRL):
    ( Rkk_nonRL[i], lambda1_nonRL[i], lambda2_nonRL[i], lambda3_nonRL[i], xmap1_nonRL[i], xmap2_nonRL[i] ) \
        = compute_reynolds_stress_dof( favre_uffuff_nonRL[i], favre_uffvff_nonRL[i], favre_uffwff_nonRL[i], favre_vffvff_nonRL[i], favre_vffwff_nonRL[i], favre_wffwff_nonRL[i], verbose=verbose )
    eigval_nonRL[i,:,0] = lambda1_nonRL[i]
    eigval_nonRL[i,:,1] = lambda2_nonRL[i]
    eigval_nonRL[i,:,2] = lambda3_nonRL[i]

# non-RL converged reference data
( Rkk_ref, lambda1_ref, lambda2_ref, lambda3_ref, xmap1_ref, xmap2_ref ) \
    = compute_reynolds_stress_dof( favre_uffuff_ref, favre_uffvff_ref, favre_uffwff_ref, favre_vffvff_ref, favre_vffwff_ref, favre_wffwff_ref, verbose=verbose )
eigval_ref[:,0] = lambda1_ref
eigval_ref[:,1] = lambda2_ref
eigval_ref[:,2] = lambda3_ref

print("Rij decomposed successfully!")

#-----------------------------------------------------------------------------------------
#           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
#-----------------------------------------------------------------------------------------

# ---------------------- Plot Barycentric Map for each RL global step (specific iteration & ensemble) ---------------------- 

print("\nBuilding triangle barycentric map plots...")

# --- Location of Barycentric map corners ---
x1c = np.array( [ 1.0 , 0.0 ] )
x2c = np.array( [ 0.0 , 0.0 ] )
x3c = np.array( [ 0.5 , np.sqrt(3.0)/2.0 ] )

# Used variables
"""
    y_delta_RL[N_RL]
    y_delta_nonRL[N_nonRL]
    y_delta_ref
    xmap1_RL[N_RL]
    xmap1_nonRL[N_nonRL]
    xmap1_ref
    xmap2_RL[N_RL]
    xmap2_nonRL[N_nonRL]
    xmap2_ref
"""
plt.figure()
cmap  = matplotlib.colormaps['Greys']
norm  = colors.Normalize(vmin = 0, vmax = 1.0)

# Plot data into the barycentric map
plt.scatter( xmap1_ref,   xmap2_ref,   c = y_delta_ref,   cmap = cmap, norm = norm, zorder = 3, marker = 'o', s = 50, edgecolor = 'black', linewidth = 0.8, label="Baseline, Converged, $t_{\textrm{avg}}^{+}=173$" )
plt.scatter(xmap1_nonRL, xmap2_nonRL,  c = y_delta_nonRL, cmap = cmap, norm = norm, zorder = 3, marker = 's', s = 50, edgecolor = 'black', linewidth = 0.8, label=r"Baseline, $t_{\textrm{avg}}^{+}=0.5$" )
plt.scatter( xmap1_RL[0], xmap2_RL[0], c = y_delta_RL[0], cmap = cmap, norm = norm, zorder = 3, marker = '^', s = 50, edgecolor = 'black', linewidth = 0.8, label=r"RL-based control, $t_{\textrm{avg}}^{+}=0.5$, Test I" )
plt.scatter( xmap1_RL[1], xmap2_RL[1], c = y_delta_RL[1], cmap = cmap, norm = norm, zorder = 3, marker = 'v', s = 50, edgecolor = 'black', linewidth = 0.8, label=r"RL-based control, $t_{\textrm{avg}}^{+}=0.5$, Test II" )

# Plot barycentric map lines
plt.plot( [x1c[0], x2c[0]],[x1c[1], x2c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
plt.plot( [x2c[0], x3c[0]],[x2c[1], x3c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )
plt.plot( [x3c[0], x1c[0]],[x3c[1], x1c[1]], zorder = 1, color = 'black', linestyle = '-', linewidth = 2 )

# Configure plot
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.axis( 'off' )
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.text( 1.02,   -0.05,  r'$\textbf{x}_{1_{c}}$' )
plt.text( -0.065, -0.05,  r'$\textbf{x}_{2_{c}}$' )
plt.text( 0.45,   0.9000, r'$\textbf{x}_{3_{c}}$' )
cbar = plt.colorbar()
cbar.set_label( r'$y/\delta$' )
plt.legend(loc='upper right')
plt.tight_layout()
###plt.clim( 0.0, 20.0 )
filename = os.path.join(postDir, f"anisotropy_tensor_barycentric_xmap_triang.svg")
plt.savefig(filename)
plt.close()

print("Triangle barycentric map plotted successfully!")