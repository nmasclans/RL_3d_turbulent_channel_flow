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

from utils import build_csv_from_h5_snapshot

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


# Custom colormap
cmap = plt.get_cmap('RdBu_r')  # Replace with your desired colormap

# Flow parameters 
if np.isclose(Re_tau, 100, atol=1e-8):  # Re_tau = 100
    Re_tau = 100.0
elif np.isclose(Re_tau, 180, atol=1e-8):
    Re_tau = 180.0
u_tau = 1.0
rho_w = 1.0  
mu_w  = rho_w * u_tau / Re_tau  # = 1.0 / Re_tau
nu_w  = mu_w / rho_w            # = 1.0 / Re_tau
print(f"\nFlow parameters: \n- Re_tau: {Re_tau}\n- u_tau: {u_tau}\n- rho_w: {rho_w}\n- mu_w: {mu_w}\n- nu_w: {nu_w}")

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
    probes_y_coords = np.array([0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875])
elif Re_tau == 180:
    probes_y_coords = np.array([0.059369, 0.208542, 0.4811795, 0.819736, 1.18026, 1.51882, 1.79146, 1.94063])
else:
    raise ValueError(f"Unknown 'y_probes' for Re_tau = {Re_tau}")
probes_z_coords   = 2.0
probes_zy_desired = np.array(np.meshgrid(probes_z_coords, probes_y_coords)).T.reshape(-1,2)   # shape: [n_probes, 2]
n_probes          = probes_zy_desired.shape[0]    # num.probes in the zy-plane, num. pairs of (z,y) coordinates

# Visualization parameters
window_length   = 11 # Adjust based on your data
polyorder       = 3
y_plus_max_urms = 12
y_plus_max_u    = 20
y_plus_max_cp   = 140
y_limit         = y_plus_max_u
log_smooth      = False
plot_test       = False
fontsize        = 18
resolution      = 192

# h5 files for Reference, non-RL and RL data
filename_ref   = f"./data_Retau{Re_tau:.0f}/3d_turbulent_channel_flow_reference.h5"
filename_nonRL = []     # TODO: implement!
filename_RL    = []     # TODO: implement!
kwargs = {"num_grid_x": num_grid_x, 
          "num_grid_y": num_grid_y,
          "dt_dx": dt_dx,
          "n_probes": n_probes,
          "probes_zy_desired": probes_zy_desired
}

# Generated probes data (csv) directories
probes_dir_ref   = f"./data_Retau{Re_tau:.0f}/probelines"
probes_dir_nonRL = ""   # TODO: implement!
probes_dir_RL    = ""   # TODO: implement!

build_csv_from_h5_snapshot(filename_ref, "reference", probes_dir_ref, **kwargs)
# TODO: do the same for nonRL and RL files

"""
def exponential_moving_average(signal, alpha=0.3):
    ema = [signal[0]]  # Initialize with the first value
    for x in signal[1:]:
        ema.append(alpha * x + (1 - alpha) * ema[-1])
    return np.array(ema)

def low_pass_filter(signal, cutoff_freq, sample_rate):
    fft_signal = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    # Apply the filter by zeroing out frequencies higher than the cutoff
    fft_signal[np.abs(fft_freqs) > cutoff_freq] = 0
    return np.fft.ifft(fft_signal).real


def process_file(file_path, limit_wavelength):
    data = np.loadtxt(file_path, delimiter=',')
    time = data[:, 0]
    u = data[:, 5] - np.mean(data[:, 5])
    v = data[:, 6] - np.mean(data[:, 6])
    w = data[:, 7] - np.mean(data[:, 7])
    rho = data[:, 4] - np.mean(data[:, 4])
    mean_rho = np.mean(data[:, 4])
    mean_mu = np.mean(data[:, 12])
    
    if tw:
        y_plus_data = (2*delta-data[5, 2]) * rho_w * u_tau / mu_w
    if bw: 
        y_plus_data = data[5, 2] * rho_w * u_tau / mu_w
        
    #DFFT
    fft_result_u = np.fft.fft(u)
    fft_result_rho = np.fft.fft(rho)
    fft_result_total = np.fft.fft(rho*u*u)
    fft_freq = np.fft.fftfreq(len(time), time[1] - time[0])

    #Filtering negative frequencies
    positive_freq_indices = fft_freq > 0
    
    #print(positive_freq_indices)
    fft_freq = fft_freq[positive_freq_indices]
    fft_result_u = fft_result_u[positive_freq_indices]
    fft_result_rho = fft_result_rho[positive_freq_indices]
    fft_result_total = fft_result_total[positive_freq_indices]

    # Obtaining the spectrum of streamwise momentum
    N = len(fft_freq)
    
    #streamwise_spectrum = (np.abs(fft_result_u)**2 * np.abs(fft_result_rho)) / N
    streamwise_spectrum = np.abs(fft_result_total) /N
    
    # temporal wavelength and wavenumber
    wavelength = u_tau / np.abs(fft_freq)
    wavenumber = u_tau * np.abs(fft_freq)
    
    # spatial wavenumber based on taylor hipotheses
    spatial_wavenumber = wavenumber*u_tau
    
    # Assuming `original_spectrum` is the spectrum before smoothing
    test_spectrum = streamwise_spectrum
    
    #Types of smoothening
    streamwise_spectrum = gaussian_filter1d(streamwise_spectrum, sigma=7)
    #streamwise_spectrum = exponential_moving_average(streamwise_spectrum, alpha=0.3)
    #streamwise_spectrum = low_pass_filter(streamwise_spectrum, cutoff_freq=0.2, sample_rate=1)
    
    #sorting indexes
    sorted_indices = np.argsort(wavenumber)
    spatial_wavenumber = spatial_wavenumber[sorted_indices]
    streamwise_spectrum = streamwise_spectrum[sorted_indices]
    
    #premultiplied spectra
    streamwise_spectrum = streamwise_spectrum * spatial_wavenumber
    test_spectrum   = test_spectrum   * spatial_wavenumber
    streamwise_spectrum = savgol_filter(streamwise_spectrum, window_length=7, polyorder=2)
    
    
    #filtering wavelengths below the grid cutoff
    # Correct logical indexing
    filtered_indices = (wavelength[sorted_indices] >= limit_wavelength) 
    streamwise_spectrum = streamwise_spectrum[filtered_indices]
    test_spectrum = test_spectrum[filtered_indices]
    spatial_wavenumber = spatial_wavenumber[filtered_indices]
    
    #normalization
    normalized_spatial_wavenumber = spatial_wavenumber * delta_viscous
    normalized_premultiplied_spectrum = streamwise_spectrum / (mean_rho*u_tau**2)
    normalized_premultiplied_original_spectrum = test_spectrum / (mean_rho*u_tau**2)
    #print("min of normalized wavelength = ",np.amin(normalized_spatial_wavenumber))
    #print("max of normalized wavelength = ",np.amax(normalized_spatial_wavenumber))
    
    if plot_test:
        plt.figure(figsize=(12, 6))
        plt.loglog(1/normalized_spatial_wavenumber, normalized_premultiplied_original_spectrum, label='Original Spectrum')
        plt.loglog(1/normalized_spatial_wavenumber, normalized_premultiplied_spectrum, label='Smoothed Spectrum')
        plt.xlabel('Wavelength or Spatial Wavenumber')
        plt.ylabel('Spectral Value')
        plt.xscale('log')
        plt.title('Comparison of Original and Smoothed Spectrum')
        plt.legend()
        plt.grid(True)
        plt.show()
    return normalized_spatial_wavenumber, normalized_premultiplied_spectrum, y_plus_data, mean_rho, mean_mu


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
