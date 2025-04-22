import glob
import h5py
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatter
from scipy.interpolate import griddata
from matplotlib.tri import Triangulation
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from matplotlib import cm, ticker
from matplotlib.ticker import LogLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d

from utils import build_probelines_from_snapshot_h5, process_probeline_h5, process_probelines_list
from ChannelVisualizer import ChannelVisualizer

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---
try :
    iteration  = sys.argv[1]
    ensemble   = sys.argv[2]
    train_name = sys.argv[3]
    Re_tau     = float(sys.argv[4])     # Friction Reynolds number [-]
    dt_phys    = float(sys.argv[5])
    case_dir   = sys.argv[6]
    rl_n_envs  = int(sys.argv[7])
    print(f"\nScript parameters: \n- Iteration: {iteration} \n- Ensemble: {ensemble}\n- Train name: {train_name} \n- Re_tau: {Re_tau} \n- dt_phys: {dt_phys} \n- Case directory: {case_dir}\n- Num. RL environments / Parallelization cores: {rl_n_envs}")
except :
    raise ValueError("Missing call arguments, should be: <iteration> <ensemble> <train_name> <Re_tau> <dt_phys> <case_dir> <rl_n_envs>")

# Training post-processing directory
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# Probelines directory (calculated from h5 snapshots)
probelinesDir = os.path.join(postDir, "probelines")
if not os.path.exists(probelinesDir):
    os.mkdir(probelinesDir)

# Reference & non-RL data directory
filePath = os.path.dirname(os.path.abspath(__file__))
compareDatasetDir = os.path.join(filePath, f"data_Retau{Re_tau:.0f}")
iteration_max_nonRL = 4190000

# Custom colormap
cmap = plt.get_cmap('RdBu_r')  # Replace with your desired colormap

# RL parameters
t_episode_train = 1.0
dt_phys = 1e-4
cfd_n_envs = 1
simulation_time_per_train_step   = t_episode_train * cfd_n_envs       # total cfd simulated time per training step (in parallel per each cfd_n_envs)
num_global_steps_per_train_step  = int(cfd_n_envs * rl_n_envs)        # num. global steps per training step
num_iterations_per_train_step    = int(np.round(simulation_time_per_train_step / dt_phys))
iteration_restart_data_file      = 3210000
iteration_end_train_step         = iteration_restart_data_file + num_iterations_per_train_step
assert iteration_restart_data_file + num_iterations_per_train_step == iteration_end_train_step
print("\nRL parameters: \n- Simulation time per train step:", simulation_time_per_train_step, 
      "\n- Num. global steps per train step:", num_global_steps_per_train_step,
      "\n- Num. iterations per train step:", num_iterations_per_train_step,
      "\n- Iteration restart data file (init train step):", iteration_restart_data_file,
      "\n- Iteration end train step:", iteration_end_train_step,
) 

# Flow parameters 
if np.isclose(Re_tau, 100, atol=1e-8):  # Re_tau = 100
    Re_tau = 100.0
elif np.isclose(Re_tau, 180, atol=1e-8):
    Re_tau = 180.0
u_tau = 1.0
rho0  = 1.0  
mu0   = rho0 * u_tau / Re_tau  # = 1.0 / Re_tau
nu0   = mu0 / rho0             # = 1.0 / Re_tau
print(f"\nFlow parameters: \n- Re_tau: {Re_tau}\n- u_tau: {u_tau}\n- rho0: {rho0}\n- mu0: {mu0}\n- nu0: {nu0}")

# Domain & Grid parameters
delta = 1.0
L_x   = 12.566370614 
L_y   = 2 * delta               # Domain length in y-direction
if Re_tau == 100.0:
    num_grid_x = 64
    num_grid_y = 64  # Number of internal grid points in the y-direction
    num_grid_z = 64
    A_x = 0.0
    A_y = 0.0        # Streching factor in y-direction, 
                     # with stretching factors: x = x_0 + L*eta + A*( 0.5*L - L*eta )*( 1.0 - eta )*eta,
                     # with eta = ( l - 0.5 )/num_grid 
elif Re_tau == 180:
    num_grid_x = 256
    num_grid_y = 128
    num_grid_z = 128
    A_x = 0
    A_y = -1.875
else:
    raise ValueError(f"Unknown Grid and Stretching parameters for Re_tau = {Re_tau}")
x_0 = 0.0
eta = (1.0 - 0.5) / num_grid_y
delta_viscous = x_0 + L_y * eta + A_y * (0.5 * L_y - L_y * eta) * (1.0 - eta) * eta    # [m]
print(f"\nDomain & Grid parameters: \n- delta: {delta}\n- L_x: {L_x}\n- L_y: {L_y}\n- num_grid_x: {num_grid_x}\n- num_grid_y: {num_grid_x}\n- A_x: {A_x}\n- A_y: {A_y}\n- eta: {eta}\n- delta_viscous: {delta_viscous}")

# Spatial -> Temporal advancement conversion
assert A_x == 0.0, "dx -> dt Not implemented for stretching in x-direction A_x != 0.0"
t_ftt = delta / u_tau           # flow-through-time [s]
dt_dx = t_ftt / L_x             # [s/m]
print(f"\nSpatial to Temporal conversion: \n- t_ftt: {t_ftt}\n- dt_dx: {dt_dx}")

# Probes y-coordinates at RL agent location - center y-coordinate of control cubes
if Re_tau == 100.0:
    # Selected y-coordinates at RL agents center of action domain
    probes_y_coord = np.sort(np.array([0.125, 0.375, 0.625, 0.875])) #, 1.125, 1.375, 1.625, 1.875]))
elif Re_tau == 180:
    probes_y_coord = np.sort(np.array([0.059369, 0.208542, 0.4811795, 0.819736])) #, 1.18026, 1.51882, 1.79146, 1.94063]))
else:
    raise ValueError(f"Unknown 'y_probes' for Re_tau = {Re_tau}")
n_probes = len(probes_y_coord)

# Filters parameters
gf_sigma          = 7   # Gaussian filter 'sigma'
sgf_window_length = 11  # Savitzky-Golay filter 'window_length'
sgf_polyorder     = 5   # Savitzky-Golay filter 'polyorder', polynomial order

# Fourier Transform Visualization parameters
y_plus_max_urms  = 12
y_plus_max_u     = 20
y_plus_max_cp    = 140
y_limit          = y_plus_max_u
wavelength_limit = 2 * delta / num_grid_y
log_smooth       = False
fontsize         = 18

# Initialize visualizer

visualizer = ChannelVisualizer(postDir)

#--------------------------------------------------------------------------------------------

# --- Reference filename ---
filename_ref   = f"{compareDatasetDir}/3d_turbulent_channel_flow_reference.h5"

# --- RL filenames ---
pattern = f"{case_dir}/rhea_exp/output_data/RL_3d_turbulent_channel_flow_{iteration}_ensemble{ensemble}_*.h5"
matching_files = sorted(glob.glob(pattern))
filename_RL_list     = []
global_step_str_list    = []
global_step_num_list = []
if matching_files:
    print("\nRL files:")
    for file in matching_files:
        filename_RL_list.append(file)
        base_filename = os.path.basename(file)
        global_step_str = base_filename.split('_')[-1].replace('.h5', '')
        global_step_str_list.append(global_step_str)
        global_step_num = int(global_step_str[4:])
        global_step_num_list.append(global_step_num)
        print(f"Filename: {base_filename}, Global step string: {global_step_str}, Global step number: {global_step_num}")
else:
    print(f"No files found matching the pattern: {pattern}")
n_RL = len(filename_RL_list)

# --- non-RL filenames ---
train_step_list = [int(gs/num_global_steps_per_train_step) for gs in global_step_num_list]
iteration_nonRL_list = [ (s+1)*num_iterations_per_train_step + iteration_restart_data_file for s in train_step_list]
filename_nonRL_list = [f"{compareDatasetDir}/3d_turbulent_channel_flow_{iter}.h5" for iter in iteration_nonRL_list]
n_nonRL = len(train_step_list)
print("\nnon-RL files:")
for i_nonRL in range(n_nonRL):
    print("Filename:", filename_nonRL_list[i_nonRL], ", Iteration:", iteration_nonRL_list[i_nonRL])

# --- Discard non-RL (and corresponding RL) snapshots if not available
assert n_nonRL == n_RL
filename_nonRL_is_available = [iter < iteration_max_nonRL for iter in iteration_nonRL_list]
n_nonRL_is_available = sum(filename_nonRL_is_available)
print("\nAvailable non-RL files:", n_nonRL_is_available, "Non-available non-RL files:", n_nonRL - n_nonRL_is_available)
filename_RL_list_available = []
filename_nonRL_list_available = []
for i in range(n_RL):
    if filename_nonRL_is_available[i]:
        filename_RL_list_available.append(filename_RL_list[i])
        filename_nonRL_list_available.append(filename_nonRL_list[i])
filename_RL_list    = filename_RL_list_available
filename_nonRL_list = filename_nonRL_list_available
n_RL = len(filename_RL_list); n_nonRL = n_RL
print(f"Datasets RL and non-RL have now {n_RL} files each")

# --- Get averaging time of 'Reference', 'Uncontrolled' and 'RL Framework' and build 'file_details' information ---
# > Reference
with h5py.File(filename_ref, 'r') as file:
    t_avg_ref = file.attrs['AveragingTime'][0]
file_details_ref = "Reference"
# > RL
t_avg_RL = []
file_details_RL = []
for i in range(n_RL):
    with h5py.File(filename_RL_list[i], 'r') as file:
        t_avg = file.attrs['AveragingTime'][0]
        t_avg_RL.append(int(t_avg))
        file_details_RL.append(f"RL_{global_step_num_list[i]:.0f}")
# > non-RL
t_avg_nonRL = []
file_details_nonRL = []
for i in range(n_nonRL):
    with h5py.File(filename_nonRL_list[i], 'r') as file:
        t_avg = np.round(file.attrs['AveragingTime'][0], 3)
        t_avg_nonRL.append(int(t_avg))
        file_details_nonRL.append(f"nonRL_{train_step_list[i]:.0f}")
print("\nAveraging Time: \n- Reference:", t_avg_ref, "\n- RL:", t_avg_RL, "\n- non-RL:", t_avg_nonRL)
print("\nFile details: \n- Reference:", file_details_ref, "\n- RL:", file_details_RL, "\n- non-RL:", file_details_nonRL)

# --- Params dictionary ---
params = {
    "num_grid_x": num_grid_x, 
    "num_grid_y": num_grid_y,
    "num_grid_z": num_grid_z,
    "dt_dx": dt_dx,
    "n_probes": n_probes,
    "probes_y_desired": probes_y_coord,
    "rho0": rho0,
    "mu0": mu0,
    "u_tau": u_tau,
    "delta": delta,
    "delta_viscous": delta_viscous,
    "gf_sigma": gf_sigma,
    "sgf_window_length": sgf_window_length,
    "sgf_polyorder": sgf_polyorder,
    "wavelength_limit": wavelength_limit,
    "train_post_process_dir": postDir,
}

# --- 3d-spatial snapshots to 1d-temporal probelines & build plots---

# - Transform 3-d spatial snapshot (at specific time, for all domain grid points (x,y,z)) 
# into 1-d probelines temporal data (at specific (x,y,z), increasing time) 
# - Build plots
frames_spectral = []

# > Reference
probes_filepath_list_ref = build_probelines_from_snapshot_h5(filename_ref, file_details_ref, probelinesDir, params)
tavg0_ref, avg_y_ref, avg_y_plus_ref, avg_k_ref, avg_k_plus_ref, avg_lambda_ref, avg_lambda_plus_ref, avg_Euu_ref, avg_Euu_plus_ref \
    = process_probelines_list(probes_filepath_list_ref, file_details_ref, params)

for i in range(n_RL):
    # > RL
    probes_filepath_list_RL = build_probelines_from_snapshot_h5(filename_RL_list[i], file_details_RL[i], probelinesDir, params)
    tavg0_RL, avg_y_RL, avg_y_plus_RL, avg_k_RL, avg_k_plus_RL, avg_lambda_RL, avg_lambda_plus_RL, avg_Euu_RL, avg_Euu_plus_RL \
        = process_probelines_list(probes_filepath_list_RL, file_details_RL[i], params)
    # > non-RL
    probes_filepath_list_nonRL = build_probelines_from_snapshot_h5(filename_nonRL_list[i], file_details_nonRL[i], probelinesDir, params)
    tavg0_nonRL, avg_y_nonRL, avg_y_plus_nonRL, avg_k_nonRL, avg_k_plus_nonRL, avg_lambda_nonRL, avg_lambda_plus_nonRL, avg_Euu_nonRL, avg_Euu_plus_nonRL \
        = process_probelines_list(probes_filepath_list_nonRL, file_details_nonRL[i], params)

    # Plot spectrum for each specific global step 
    ###visualizer.plot_spectral_turbulent_kinetic_energy_density_streamwise_velocity(
    ###    file_details_RL[i], tavg0_RL, avg_y_RL, avg_y_plus_RL, avg_k_RL, avg_k_plus_RL, avg_lambda_RL, avg_lambda_plus_RL, avg_Euu_RL, avg_Euu_plus_RL)
    ###visualizer.plot_spectral_turbulent_kinetic_energy_density_streamwise_velocity(
    ###    file_details_nonRL[i], tavg0_nonRL, avg_y_nonRL, avg_y_plus_nonRL, avg_k_nonRL, avg_k_plus_nonRL, avg_lambda_nonRL, avg_lambda_plus_nonRL, avg_Euu_nonRL, avg_Euu_plus_nonRL)
    
    # Check probes y+ are equal for all RL, non-RL, ref
    for i_probe in range(n_probes):
        assert np.isclose(avg_y_plus_ref[i_probe], avg_y_plus_RL[i_probe]) & np.isclose(avg_y_plus_ref[i_probe], avg_y_plus_nonRL[i_probe])
    
    # Build frame for each specific global step
    frames_spectral = visualizer.build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_frame(
        frames_spectral, avg_y_plus_ref, avg_k_plus_RL, avg_k_plus_nonRL, avg_k_plus_ref, avg_Euu_plus_RL, avg_Euu_plus_nonRL, avg_Euu_plus_ref, tavg0_RL, tavg0_nonRL, global_step_num_list[i],
    )

print("\nSave gifs from frames...")
frames_dict = {'spectral_Euu+_vs_k+':frames_spectral}
visualizer.build_main_gifs_from_frames(frames_dict)
print("Gifs plotted successfully!")

"""
def plot_colormap(directory, resolution):
    all_spectra = []
    for directory in directories:
        directory_path = os.path.join('.', directory)
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

        y_positions_list = []
        spectra_list = []
        k_values_list = []
        bulk_rho = []
        bulk_mu = []
        limit_wavelength = 2*delta / resolution

        for csv_file in csv_files:
            file_path = os.path.join(directory_path, csv_file)

            k_value, spectrum, y_position, mean_rho, mean_mu = process_file(file_path, limit_wavelength)
            if csv_file == csv_files[0]:
                    k_values_list.append(k_value)
            if (y_position > 0) & (y_position < y_limit):
                print(f"Processing: {file_path}") 
                
                bulk_rho.append(mean_rho)
                bulk_mu.append(mean_mu)
                y_positions_list.append(y_position)
                spectra_list.append(spectrum)

        bulk_mu = np.mean(bulk_mu)
        bulk_rho = np.mean(bulk_rho)

        wavenumbers = np.array(k_values_list).flatten()    #x
        y_positions = np.array(y_positions_list)           #y
        spectra     = np.array(spectra_list)               #z


        # Get the indices that would sort `y_positions` in ascending order
        sorted_indices = np.argsort(y_positions)

        # Reorder `y_positions` and `spectra` according to these indices
        sorted_y_positions = y_positions[sorted_indices]
        sorted_spectra = spectra[sorted_indices]
       
        all_spectra.append(sorted_spectra)
    
    average_spectra_list = np.array(all_spectra)
    average_spectra = np.mean(average_spectra_list, axis=0)
    #average_spectra_list = average_spectra_list[1:-2,:]
    
    min_spectra = np.amin(average_spectra)
    max_spectra = np.amax(average_spectra)
     
    #print('min wavelength',np.amin(1/wavenumbers), 'max_wavelength',np.amax(1/wavenumbers))
    print('min spectra',min_spectra,               'max_spectra', max_spectra)
    print("y^+ min =", np.amin(sorted_y_positions),"y^+ max =",np.amax(sorted_y_positions))
    
    fig, ax = plt.subplots(figsize=(4, 4.5))
    if tw:
        average_spectra[average_spectra < 0.5] = 0.5
        average_spectra[average_spectra > 9.99] = 9.99
        levs = np.linspace(0.0, 10, 50)
        # Add a dashed horizontal line at y⁺ max cp
        y_plus_line = y_plus_max_cp
        ax.axhline(y=y_plus_line, color='black', linestyle='--', linewidth=1.5)
        ax.text(x=400, y=y_plus_line * 1.05, s=r'$c_{P_\textrm{max}}$', color='black', fontsize=fontsize, ha='left', va='bottom')
        # Add a dashed horizontal line at y⁺ max u
        y_plus_line = y_plus_max_u
        ax.axhline(y=y_plus_line, color='black', linestyle='--', linewidth=1.5)
        ax.text(x=400, y=y_plus_line * 1.05, s=r'$u_\textrm{max}$', color='black', fontsize=fontsize, ha='left', va='bottom')
    if bw:
        average_spectra[average_spectra < 0.01] = 0.01
        average_spectra[average_spectra > 0.99] = 0.99
        levs = np.linspace(0.0, 1, 50)
        # Add a dashed horizontal line at y⁺ max u
        y_plus_line = y_plus_max_u
        ax.axhline(y=y_plus_line, color='black', linestyle='--', linewidth=1.5)
        ax.text(x=400, y=y_plus_line * 1.05, s=r'$u_\textrm{max}$', color='black', fontsize=fontsize, ha='left', va='bottom')

    countour = ax.contourf(1/wavenumbers, sorted_y_positions.T, average_spectra, levels=levs, cmap=cmap)

    # Add colorbar
    cbar = fig.colorbar(countour, ax=ax, orientation='horizontal', pad=0.03, location='top')
    cbar.ax.yaxis.set_tick_params(labelsize=fontsize)

    if tw:
        cbar.set_ticks([0.0,2,4,6,8,10]) 
    if bw:
        cbar.set_ticks([0.0,0.2,0.4,0.6,0.8,1]) 
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.99e0, 2.0001e3)  
    ax.set_ylim(5.99e-1, 1.5001e2)  
    #ax.xaxis.set_ticks([1, 10, 100, 1000])
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='auto', numticks=10))
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', labelsize=fontsize, top=True, right=True, direction='in', pad =  7.5)
    cbar.set_label(r'$k_x E_{\rho uu} \rho^{-1} u_\tau^{-2}$', labelpad=15)
    plt.xlabel(r'$1/k^*_x$', fontsize=fontsize)
    if tw:
        plt.ylabel(r'$y^*_\textrm{hw}$', fontsize=fontsize)
    if bw:
        plt.ylabel(r'$y^*_\textrm{cw}$', fontsize=fontsize)
    
    plt.tight_layout()
    
    # Remove trailing slash from directory name if it exists
    directory_name = directory.rstrip('/')
    
    if tw:
        plt.savefig(f'spectrogram_tw_{name_directory}.png')
        plt.savefig(f'spectrogram_tw_{name_directory}.eps')
    if bw:
        plt.savefig(f'spectrogram_bw_{name_directory}.png')
        plt.savefig(f'spectrogram_bw_{name_directory}.eps')
    
    #plt.show()
    
plot_colormap(directories,resolution)
"""
