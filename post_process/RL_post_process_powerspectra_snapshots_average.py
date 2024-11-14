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

#np.set_printoptions(threshold=sys.maxsize)
#plt.rc( 'text', usetex = True )
#rc('font', family='sanserif')
#plt.rc( 'font', size = 18 )
#plt.rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}', r'\usepackage{amssymb}', r'\usepackage{color}' ]

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---
try :
    iteration  = sys.argv[1]
    ensemble   = sys.argv[2]
    train_name = sys.argv[3]
    Re_tau     = float(sys.argv[4])     # Friction Reynolds number [-]
    dt_phys    = float(sys.argv[5])
    case_dir   = sys.argv[6]
    print(f"\nScript parameters: \n- Iteration: {iteration} \n- Ensemble: {ensemble}\n- Train name: {train_name} \n- Re_tau: {Re_tau} \n- dt_phys: {dt_phys} \n- Case directory: {case_dir}")
except :
    raise ValueError("Missing call arguments, should be: <iteration> <ensemble> <train_name> <Re_tau> <dt_phys> <case_dir>")

# Training post-processing directory
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# Reference & non-RL data directory
filePath = os.path.dirname(os.path.abspath(__file__))
compareDatasetDir = os.path.join(filePath, f"data_Retau{Re_tau:.0f}")

# Custom colormap
cmap = plt.get_cmap('RdBu_r')  # Replace with your desired colormap

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
    A_x = 0.0
    A_y = 0.0        # Streching factor in y-direction, 
                     # with stretching factors: x = x_0 + L*eta + A*( 0.5*L - L*eta )*( 1.0 - eta )*eta,
                     # with eta = ( l - 0.5 )/num_grid 
elif Re_tau == 180:
    num_grid_x = 256
    num_grid_y = 128
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
    probes_y_coords = np.sort(np.array([0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875]))
    # All z coordinates, probes information will be averaged along z for each y-coord
    probes_z_coords = np.sort(np.array([0.03272492, 0.09817477, 0.16362462, 0.22907446, 0.29452431, 0.35997416, 0.42542401, 0.49087385, 0.5563237 , 0.62177355, 0.68722339, 0.75267324, 0.81812309, 0.88357293, 0.94902278, 1.01447263, 1.07992247, 1.14537232, 1.21082217, 1.27627202, 1.34172186, 1.40717171, 1.47262156, 1.5380714 , 1.60352125, 1.6689711 , 1.73442094, 1.79987079, 1.86532064, 1.93077049, 1.99622033, 2.06167018, 2.12712003, 2.19256987, 2.25801972, 2.32346957, 2.38891941, 2.45436926, 2.51981911, 2.58526895, 2.6507188 , 2.71616865, 2.7816185 , 2.84706834, 2.91251819, 2.97796804, 3.04341788, 3.10886773, 3.17431758, 3.23976742, 3.30521727, 3.37066712, 3.43611697, 3.50156681, 3.56701666, 3.63246651, 3.69791635, 3.7633662 , 3.82881605, 3.89426589, 3.95971574, 4.02516559, 4.09061543, 4.15606528]))
elif Re_tau == 180:
    probes_y_coords = np.sort(np.array([0.059369, 0.208542, 0.4811795, 0.819736, 1.18026, 1.51882, 1.79146, 1.94063]))
    probes_z_coords = np.sort(np.array([0.01636246, 0.04908739, 0.08181231, 0.11453723, 0.14726216, 0.17998708, 0.212712, 0.24543693, 0.27816185, 0.31088677, 0.3436117 , 0.37633662, 0.40906154, 0.44178647, 0.47451139, 0.50723631, 0.53996124, 0.57268616, 0.60541108, 0.63813601, 0.67086093, 0.70358585, 0.73631078, 0.7690357 , 0.80176063, 0.83448555, 0.86721047, 0.8999354 , 0.93266032, 0.96538524, 0.99811017, 1.03083509, 1.06356001, 1.09628494, 1.12900986, 1.16173478, 1.19445971, 1.22718463, 1.25990955, 1.29263448, 1.3253594 , 1.35808432, 1.39080925, 1.42353417, 1.45625909, 1.48898402, 1.52170894, 1.55443387, 1.58715879, 1.61988371, 1.65260864, 1.68533356, 1.71805848, 1.75078341, 1.78350833, 1.81623325, 1.84895818, 1.8816831 , 1.91440802, 1.94713295, 1.97985787, 2.01258279, 2.04530772, 2.07803264, 2.11075756, 2.14348249, 2.17620741, 2.20893233, 2.24165726, 2.27438218, 2.30710711, 2.33983203, 2.37255695, 2.40528188, 2.4380068 , 2.47073172, 2.50345665, 2.53618157, 2.56890649, 2.60163142, 2.63435634, 2.66708126, 2.69980619, 2.73253111, 2.76525603, 2.79798096, 2.83070588, 2.8634308 , 2.89615573, 2.92888065, 2.96160557, 2.9943305 , 3.02705542, 3.05978035, 3.09250527, 3.12523019, 3.15795512, 3.19068004, 3.22340496, 3.25612989, 3.28885481, 3.32157973, 3.35430466, 3.38702958, 3.4197545 , 3.45247943, 3.48520435, 3.51792927, 3.5506542 , 3.58337912, 3.61610404, 3.64882897, 3.68155389, 3.71427881, 3.74700374, 3.77972866, 3.81245359, 3.84517851, 3.87790343, 3.91062836, 3.94335328, 3.9760782 , 4.00880313, 4.04152805, 4.07425297, 4.1069779 , 4.13970282, 4.17242774]))
else:
    raise ValueError(f"Unknown 'y_probes' for Re_tau = {Re_tau}")
probes_zy_desired = np.array(np.meshgrid(probes_z_coords, probes_y_coords)).T.reshape(-1,2)   # shape: [n_probes, 2]
n_probes          = probes_zy_desired.shape[0]    # num.probes in the zy-plane, num. pairs of (z,y) coordinates

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

# h5 files for Reference, non-RL and RL data
filename_ref   = f"{compareDatasetDir}/3d_turbulent_channel_flow_reference.h5"
filename_nonRL = []     # TODO: implement!
filename_RL    = []     # TODO: implement!
params = {
    "num_grid_x": num_grid_x, 
    "num_grid_y": num_grid_y,
    "dt_dx": dt_dx,
    "n_probes": n_probes,
    "probes_zy_desired": probes_zy_desired,
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

# Generated probes data (csv) directories
probes_dir_ref   = f"{compareDatasetDir}/probelines"
probes_dir_nonRL = probes_dir_ref
probes_dir_RL    = ""   # TODO: implement!
if not os.path.exists(probes_dir_ref):
    os.makedirs(probes_dir_ref)
    print(f"\nNew directory created for Reference & non-RL probelines: {probes_dir_ref}")

# TODO: do the same for nonRL and RL files
probes_filepath_list_ref = build_probelines_from_snapshot_h5(filename_ref, "reference", probes_dir_ref, params)
_ = process_probelines_list(probes_filepath_list_ref, "reference", params)
    
    
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
