import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 12, frameon = False)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')
plt.rc( 'savefig',    format = "jpg", dpi = 600)


#-----------------------------------------------------------------------------------------
#           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
#-----------------------------------------------------------------------------------------

def check_realizability_conditions(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, verbose=False):

    #------------ Realizability conditions ---------------

    # help: .all() ensures the condition is satisfied in all grid points

    # COND 1: Rii >= 0, for i = 1,2,3

    cond0_0 = ( Rxx >= 0 ).all()    # i = 1
    cond0_1 = ( Ryy >= 0 ).all()    # i = 2
    cond0_2 = ( Rzz >= 0 ).all()    # i = 3
    cond0   = cond0_0 and cond0_1 and cond0_2

    # COND 2: Rij^2 <= Rii*Rjj, for i!=j

    cond1_0 = ( Rxy**2 <= Rxx * Ryy ).all()     # i = 0, j = 1
    cond1_1 = ( Rxz**2 <= Rxx * Rzz ).all()     # i = 0, j = 2
    cond1_2 = ( Ryz**2 <= Ryy * Rzz ).all()     # i = 1, j = 2
    cond1   = cond1_0 and cond1_1 and cond1_2

    # COND 3: det(Rij) >= 0
    detR  = Rxx * Ryy * Rzz + 2 * Rxy * Rxz * Ryz - (Rxx * Ryz * Ryz + Ryy * Rxz * Rxz + Rzz * Rxy * Rxy) # np.linalg.det(R_ij), length #num_points
    # detR  = np.linalg.det(R_ij)
    # enforce detR == 0 if -eps < detR < 0 due to computational error  
    for i in range(len(detR)):
        if detR[i] > -1e-12 and detR[i] < 0.0:
            detR[i] = 0.0
    cond2 = ( detR >= 0 ).all()

    if cond0 and cond1 and cond2:
        if verbose:
            print("\nCONGRATULATIONS, the reynolds stress tensor satisfies REALIZABILITY CONDITIONS.")
    else:
        message = f"ATTENTION!!! \nThe reynolds stress tensor does not satisfy REALIZABILITY CONDITIONS: cond0 = {cond0}, cond1 = {cond1}, cond2 = {cond2}"
        print(message)
        # raise Exception(message)

    
def compute_reynolds_stress_dof(Rxx, Rxy, Rxz, Ryy, Ryz, Rzz, 
                                tensor_kk_tolerance   = 1.0e-8, 
                                eigenvalues_tolerance = 1.0e-8, 
                                verbose = False,
                                x1c = np.array( [ 1.0 , 0.0 ] ),
                                x2c = np.array( [ 0.0 , 0.0 ] ),
                                x3c = np.array( [ 0.5 , math.sqrt(3.0)/2.0 ] )):

    check_realizability_conditions(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, verbose)
    
    # Computed for each point of the grid
    # If the trace of the reynolds stress tensor (2 * TKE) is too small, the corresponding 
    # datapoint is omitted, because the anisotropy tensor would -> infinity, as its equation
    # contains the multiplier ( 1 / (2*TKE) )
    
    # initialize arrays
    num_points = len(Rxx)
    Rkk     = np.zeros(num_points)
    lambda1 = np.zeros(num_points)
    lambda2 = np.zeros(num_points)
    lambda3 = np.zeros(num_points)
    xmap1   = np.zeros(num_points)
    xmap2   = np.zeros(num_points)

    for p in range(num_points):

        #------------ Reynolds stress tensor ---------------

        R_ij      = np.zeros([3, 3])
        R_ij[0,0] = Rxx[p]
        R_ij[0,1] = Rxy[p]
        R_ij[0,2] = Rxz[p]
        R_ij[1,0] = Rxy[p]
        R_ij[1,1] = Ryy[p]
        R_ij[1,2] = Ryz[p]
        R_ij[2,0] = Rxz[p]
        R_ij[2,1] = Ryz[p]
        R_ij[2,2] = Rzz[p]

        #------------ Anisotropy Tensor ------------

        # identity tensor
        delta_ij = np.eye(3)                                        # shape: [3,3]

        # calculate trace -> 2 * (Turbulent kinetic energy)
        Rkk[p] = R_ij[0,0] + R_ij[1,1] + R_ij[2,2]
        TKE = 0.5 * Rkk[p] #  -> same formula!                      # shape: scalar
        ###TKE = 0.5 * (urmsf[p]**2 + vrmsf[p]**2 + wrmsf[p]**2)    # shape: scalar

        # omit grid point if reynolds stress tensor trace (2 * TKE) is too small
        discarded_points = []
        if np.abs(Rkk[p]) < tensor_kk_tolerance:
            discarded_points.append(p)
            continue
        if len(discarded_points)>0:
            print(f"Discarded points id:", discarded_points)

        # construct anisotropy tensor
        a_ij = (1.0 / (2*TKE)) * R_ij - (1.0 / 3.0) * delta_ij   # shape: [3,3]

        #------------ eigen-decomposition of the SYMMETRIC TRACE-FREE anisotropy tensor ------------

        # ensure a_ij is trace-free
        # -> calculate trace
        a_kk = a_ij[0,0] + a_ij[1,1] + a_ij[2,2]
        # -> substract the trace
        a_ij[0,0] -= a_kk/3.0
        a_ij[1,1] -= a_kk/3.0
        a_ij[2,2] -= a_kk/3.0

        # Calculate the eigenvalues and eigenvectors
        eigenvalues_a_ij, eigenvectors_a_ij = np.linalg.eigh( a_ij )
        eigenvalues_a_ij_sum = sum(eigenvalues_a_ij)
        assert eigenvalues_a_ij_sum < eigenvalues_tolerance, f"ERROR: The sum of the anisotropy tensor eigenvalues should be 0; in point #{p} the sum is = {eigenvalues_a_ij_sum}"

        # Sort eigenvalues and eigenvectors in decreasing order, so that eigval_1 >= eigval_2 >= eigval_3
        idx = eigenvalues_a_ij.argsort()[::-1]   
        eigenvalues_a_ij  = eigenvalues_a_ij[idx]
        eigenvectors_a_ij = eigenvectors_a_ij[:,idx]
        (lambda1[p], lambda2[p], lambda3[p]) = eigenvalues_a_ij

        if verbose:
            inspected_eigenvalue = (-Rxx[p]+Ryy[p]-3*Ryz[p])/(3*Rxx[p]+6*Ryy[p])
            print(f"\nPoint p = {p}")
            print(f"3rd eigenvalue lambda_2 = {eigenvalues_a_ij[2]}")
            print(f"3rd eigenvector v_2     = {eigenvectors_a_ij[:,2]}")
            print(f"(expected from equations) \lambda_2 = (-R_00+R_11-3R_12)/(3R_00+6R_11) = {inspected_eigenvalue}")
            print(f"(expected from equations) v_2 = (0, -1, 1)$, not normalized")
            print(f"R_11 = {Ryy[p]:.5f}, R_12 = {Ryz[p]:.5f}")

        # Calculate Barycentric map point
        # where eigenvalues_a_ij[0] >= eigenvalues_a_ij[1] >= eigenvalues_a_ij[2] (eigval in decreasing order)
        bar_map_xy =   x1c * (     eigenvalues_a_ij[0] -     eigenvalues_a_ij[1]) \
                     + x2c * ( 2 * eigenvalues_a_ij[1] - 2 * eigenvalues_a_ij[2]) \
                     + x3c * ( 3 * eigenvalues_a_ij[2] + 1)
        xmap1[p]   = bar_map_xy[0]
        xmap2[p]   = bar_map_xy[1]

    return (Rkk, lambda1, lambda2, lambda3, xmap1, xmap2)

#-----------------------------------------------------------------------------------------
#           Temporal probes at (y,z) coordinate pair along x-direction 
#-----------------------------------------------------------------------------------------

def build_probelines_from_snapshot_h5(
    input_h5_filepath, file_details, output_probelines_directory, params
):

    # Problem parameters
    num_grid_x       = params["num_grid_x"] 
    num_grid_y       = params["num_grid_y"]
    num_grid_z       = params["num_grid_z"]
    dt_dx            = params["dt_dx"]
    n_probes         = params["n_probes"]
    probes_y_desired = params["probes_y_desired"]
    rho0             = params["rho0"]
    mu0              = params["mu0"]
    u_tau            = params["u_tau"]
    delta            = params["delta"]

    # Get data: x, y, z, u, v, w; attributes: time
    with h5py.File(input_h5_filepath, 'r') as file:
        t0          = file.attrs['Time'][0]             # [0] to take the np.float64 scalar value, not a np.array
        tavg0       = file.attrs['AveragingTime'][0]
        x_data      = file['x'][1:-1,1:-1,1:-1]         # 1:-1 to take only inner grid points
        y_data      = file['y'][1:-1,1:-1,1:-1]
        #z_data      = file['z'][1:-1,1:-1,1:-1]
        #u_data     = file['u'][1:-1,1:-1,1:-1]
        #v_data     = file['v'][1:-1,1:-1,1:-1]
        #w_data     = file['w'][1:-1,1:-1,1:-1]
        avg_u_data  = file['avg_u'][1:-1,1:-1,1:-1]
        avg_v_data  = file['avg_v'][1:-1,1:-1,1:-1]
        avg_w_data  = file['avg_w'][1:-1,1:-1,1:-1]
        rmsf_u_data = file['rmsf_u'][1:-1,1:-1,1:-1]
        rmsf_v_data = file['rmsf_v'][1:-1,1:-1,1:-1]
        rmsf_w_data = file['rmsf_w'][1:-1,1:-1,1:-1]
    num_points_z, num_points_y, num_points_x = rmsf_u_data.shape
    assert num_points_x == num_grid_x & num_points_y == num_grid_y & num_points_z == num_grid_z, f"Grid num. points different than expected" \
        + f"({num_points_x}, {num_grid_x}), ({num_points_y}, {num_grid_y})"
    print(f"\nProcessing h5 snapshot: {input_h5_filepath}, with averaging time: {tavg0}")    

    # Convert dx -> dt --> rebuild 'x_data' and 'time_data' to translate spatial advancement to temporal advancement
    # ASSUMPTION: domain grid is regular in x-direction
    x0           = x_data[0,0,0]
    dx_data      = x_data - x0              # shape [num_points_z, num_points_y, num_points_x]
    t_data       = t0 + dt_dx * dx_data     # shape [num_points_z, num_points_y, num_points_x]
    num_points_t = num_points_x
    #x_data      = x0 * np.ones([num_points_z, num_points_y, num_points_t]) # not-used

    # Find probes y coordinate closest to chosen 'probes_y_desired' coordinates given as input parameter
    # ASSUMPTION: regular grid
    yidx_probes = np.zeros(n_probes, dtype='int64')   # probes y-coordinate index in full domain grid
    y_probes    = np.zeros(n_probes)                  # probes y-coordinate
    y_coords    = y_data[0,:,0]
    for j_probe in range(n_probes):
        y_desired = probes_y_desired[j_probe]
        jj = np.argmin(np.abs(y_coords - y_desired))
        yy = y_coords[jj]
        yidx_probes[j_probe] = jj
        y_probes[j_probe]    = yy
    print(f"\nDesired probes y-coordinates: \n{probes_y_desired}")
    print(f"\nFound probes y-coordinates: \n{y_probes}")
    print(f"\nFound probes index for y-coordinates: \n{yidx_probes}")
    
    # Transform probes y to y+ coordinate
    print("\nNormalize y to y+ wall units:")
    y_plus_probes = np.zeros(n_probes)
    for j_probe in range(n_probes):
        yy = y_probes[j_probe]
        isBottomWall = yy < delta
        if isBottomWall:
            y_plus_probes[j_probe] = yy * rho0 * u_tau / mu0
            print(f"Probeline {j_probe}: bottom wall, y = {yy}, y+ = {y_plus_probes[j_probe]}")
        else: 
            y_plus_probes[j_probe] = (2 * delta - yy) * rho0 * u_tau / mu0
            print(f"Probeline {j_probe}: top wall, y = {yy}, y+ = {y_plus_probes[j_probe]}")

    # Get probes data at specific y-coords, and
    # Average probes data on z-direction (flow is periodic in z-direction)
    # ASSUMPTION: domain grid is regular in z-direction
    t_probes      = np.zeros([n_probes, num_points_t])
    avg_u_probes  = np.zeros([n_probes, num_points_t])
    avg_v_probes  = np.zeros([n_probes, num_points_t])
    avg_w_probes  = np.zeros([n_probes, num_points_t])
    rmsf_u_probes = np.zeros([n_probes, num_points_t])
    rmsf_v_probes = np.zeros([n_probes, num_points_t])
    rmsf_w_probes = np.zeros([n_probes, num_points_t])
    for k in range(num_points_z):
        for j_probe in range(n_probes):
            for t in range(num_points_t):
                j = yidx_probes[j_probe]
                t_probes[j_probe,t]      += t_data[k,j,t]
                avg_u_probes[j_probe,t]  += avg_u_data[k,j,t]
                avg_v_probes[j_probe,t]  += avg_v_data[k,j,t]
                avg_w_probes[j_probe,t]  += avg_w_data[k,j,t]
                rmsf_u_probes[j_probe,t] += rmsf_u_data[k,j,t]
                rmsf_v_probes[j_probe,t] += rmsf_v_data[k,j,t]
                rmsf_w_probes[j_probe,t] += rmsf_w_data[k,j,t]
    t_probes      /= num_points_z        
    avg_u_probes  /= num_points_z        
    avg_v_probes  /= num_points_z        
    avg_w_probes  /= num_points_z        
    rmsf_u_probes /= num_points_z        
    rmsf_v_probes /= num_points_z        
    rmsf_w_probes /= num_points_z        

    # Get probes data and store in h5 file
    print("\nSaving probelines data...")
    probes_filepath_list = []
    for j_probe in range(n_probes):
        # --- Get probeline data ---
        y_      = y_probes[j_probe]      # scalar
        y_plus_ = y_plus_probes[j_probe] # scalar
        t_      = t_probes[j_probe,:]    # 1-D array, length num_points_t (= num_points_x)
        avg_u_  = avg_u_probes[j_probe,:]  # 1-D array...
        avg_v_  = avg_v_probes[j_probe,:]
        avg_w_  = avg_w_probes[j_probe,:]
        rmsf_u_ = rmsf_u_probes[j_probe,:]
        rmsf_v_ = rmsf_v_probes[j_probe,:]
        rmsf_w_ = rmsf_w_probes[j_probe,:]
        # --- Save probeline data ---
        # Save in .h5 (only save 1 single value for x,y,z coordinates, which are constant for all probeline data of a specific probeline)
        fpath = os.path.join(output_probelines_directory, f'probeline{j_probe}_{file_details}.h5')
        with h5py.File(fpath,'w') as f:
            f.create_dataset("t", data=t_)
            f.create_dataset("avg_u",  data=avg_u_)
            f.create_dataset("avg_v",  data=avg_v_)
            f.create_dataset("avg_w",  data=avg_w_)
            f.create_dataset("rmsf_u", data=rmsf_u_)
            f.create_dataset("rmsf_v", data=rmsf_v_)
            f.create_dataset("rmsf_w", data=rmsf_w_)
            f.attrs['AveragingTime'] = tavg0
            f.attrs['y_probe']       = y_
            f.attrs['y_plus_probe']  = y_plus_
        print(f"Probe {j_probe} at y: {y_:.3f}, y+: {y_plus_}, stored in file: {fpath}")
        probes_filepath_list.append(fpath)

    return probes_filepath_list

#-----------------------------------------------------------------------------------------
#                                    Averaged Power Spectra
#-----------------------------------------------------------------------------------------

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


# ASSUMPTION 1: constant rho, mu, nu along domain
# ASSUMPTION 2: regular grid, no stretching in x-direction -> constant probeline time-step
def process_probeline_h5(file_path, params):

    print(f"\nProcessing probeline: {file_path}")

    # Problem parameters
    rho0              = params["rho0"]
    mu0               = params["mu0"]
    u_tau             = params["u_tau"]
    delta             = params["delta"]
    gf_sigma          = params["gf_sigma"]
    sgf_window_length = params["sgf_window_length"]
    sgf_polyorder     = params["sgf_polyorder"]
    wavelength_limit  = params["wavelength_limit"]

    # Get probeline data in .h5 file
    with h5py.File(file_path, 'r') as f:
        t_data       = f['t'][:]
        avg_u_data   = f['avg_u'][:]
        avg_v_data   = f['avg_v'][:]
        avg_w_data   = f['avg_w'][:]
        rmsf_u_data  = f['rmsf_u'][:]
        rmsf_v_data  = f['rmsf_v'][:]
        rmsf_w_data  = f['rmsf_w'][:]
        tavg0_probe  = f.attrs['AveragingTime']
        y_probe      = f.attrs['y_probe']
        y_plus_probe = f.attrs['y_plus_probe']

    # Check Assumption 2: constant probeline time-step (required to calculate fft!)
    if np.all(np.isclose(t_data[1:]-t_data[:-1], t_data[1]-t_data[0])):
        dt = t_data[1] - t_data[0]
    else:
        raise ValueError(f"Assumption not satisfied: grid has stretching in x-direction (A_x>0) which transformed [3-D snapshot at spec. time] into [1-D probeline at spec. (x,y,z), increasing time] to have different dt along probeline evolution")
        
    # Direct Fast Fourier Transform (DFFT)
    fft_uf      = np.fft.fft(rmsf_u_data)
    fft_rhoufuf = np.fft.fft(rho0 * rmsf_u_data * rmsf_u_data)  #  Spectral turbulent kinetic energy density of the streamwise velocity
    fft_freq    = np.fft.fftfreq(len(t_data), dt)

    # Filtering negative frequencies
    positive_freq_indices = fft_freq > 0
    fft_freq    = fft_freq[positive_freq_indices]               # frequency (f) [Hz=1/s]
    fft_uf      = fft_uf[positive_freq_indices]             
    fft_rhoufuf = fft_rhoufuf[positive_freq_indices]            # spectral TKE density [kg/(m·s^2)=J/m^3]
    N           = len(fft_freq)

    # Spatial wavelength and wavenumber, based on Taylor hypothesis
    # Source: https://gibbs.science/efd/lectures/lecture_24.pdf
    velocity_norm = np.sqrt(avg_u_data**2 + avg_v_data**2 + avg_w_data**2)  # shape [num_points_t]
    avg_velocity_norm = np.mean(velocity_norm)
    print(f"Time-averaged averaged-velocity-norm: {avg_velocity_norm:.3f}")
    spatial_wavenumber  = np.abs(fft_freq) / avg_velocity_norm  # spatial wavenumber (k) [1/m]
    spatial_wavelength  = ( (2*np.pi) / spatial_wavenumber )    # spatial wavelength (lambda) [m]  
    
    # Spectral turbulent kinetic energy density of the streamwise velocity (Euu)
    streamwise_spectrum = np.abs(fft_rhoufuf) / N
    
    # Sort all by increasing wavenumber / decreasing wavelength
    sorted_indices      = np.argsort(spatial_wavenumber)
    spatial_wavenumber  = spatial_wavenumber[sorted_indices]    # spatial wavenumber (k) [1/m]
    spatial_wavelength  = spatial_wavelength[sorted_indices]    # spatial wavelength (lambda) [m]  
    streamwise_spectrum = streamwise_spectrum[sorted_indices]   # spectral TKE density [kg/(m·s^2)=J/m^3]
    
    ### # 1st Smoothing: Apply smoothing by Gaussian filter
    ### streamwise_spectrum = gaussian_filter1d(streamwise_spectrum, sigma=gf_sigma, mode='nearest')
    ### #streamwise_spectrum = exponential_moving_average(streamwise_spectrum, alpha=0.3)
    ### #streamwise_spectrum = low_pass_filter(streamwise_spectrum, cutoff_freq=0.2, sample_rate=1)
    ### 
    ### # Premultiplied spectra: k * fft(rhoufuf) (k*Euu)
    ### streamwise_spectrum             = wavenumber * streamwise_spectrum
    ### nonfiltered_streamwise_spectrum = wavenumber * nonfiltered_streamwise_spectrum
    ### 
    ### # 2nd Smoothing: Apply smoothing by Savitzky-Golay polynomial regression filter
    ### premultiplied_streamwise_spectrum = savgol_filter(premultiplied_streamwise_spectrum, window_length=sgf_window_length, polyorder=sgf_polyorder)  
    ### # TODO: check if necessary to apply 2 filters: gaussian + sav-gol, or if only Gaussian filtering is enough  
    
    # Truncate wavelengths below the grid cutoff
    truncated_indices   = (spatial_wavelength >= wavelength_limit) 
    spatial_wavenumber  = spatial_wavenumber[truncated_indices]             # spatial wavenumber (k) [1/m]
    spatial_wavelength  = spatial_wavelength[truncated_indices]             # spatial wavelength (lambda) [m]  
    streamwise_spectrum = streamwise_spectrum[truncated_indices]            # spectral TKE density (Euu) [kg/(m·s^2)]
    
    # Normalize in wall-units
    # (rho0 * u_tau / mu0) = [1/m]
    # (rho0 * u_tau**2) = [kg/(m·s^2)]
    spatial_wavenumber_plus  = spatial_wavenumber / (rho0 * u_tau / mu0)    # normalized spatial wavenumber in wall units (k+) [-]
    spatial_wavelength_plus  = spatial_wavelength * (rho0 * u_tau / mu0)    # normalized spatial wavelength in wall units (lambda+) [-]
    streamwise_spectrum_plus = streamwise_spectrum / (rho0 * u_tau**2)      # normalized spectral TKE density (Euu+) [-]
    return tavg0_probe, y_probe, y_plus_probe, spatial_wavenumber, spatial_wavenumber_plus, spatial_wavelength, spatial_wavelength_plus, streamwise_spectrum, streamwise_spectrum_plus


def process_probelines_list(probes_filepath_list, file_details, params):
        
    print(f"\nProcessing probelines list...")
    
    # Get probelines data from each probeline h5 file
    n_probes = params["n_probes"]
    for j_probe in range(n_probes):
        file_path = probes_filepath_list[j_probe]
        tavg0_i, y_i, y_plus_i, k_i, k_plus_i, lambda_i, lambda_plus_i, Euu_i, Euu_plus_i, \
            = process_probeline_h5(file_path, params)
        if j_probe == 0:
            # Initialize allocation arrays, n_k is originally unknown, because fft has been truncated
            n_k = len(k_i)
            tavg0_data       = np.zeros([n_probes])
            y_data           = np.zeros([n_probes])
            y_plus_data      = np.zeros([n_probes])
            k_data           = np.zeros([n_probes, n_k])
            k_plus_data      = np.zeros([n_probes, n_k])
            lambda_data      = np.zeros([n_probes, n_k])
            lambda_plus_data = np.zeros([n_probes, n_k])
            Euu_data         = np.zeros([n_probes, n_k])
            Euu_plus_data    = np.zeros([n_probes, n_k])
        tavg0_data[j_probe]         = tavg0_i
        y_data[j_probe]             = y_i
        y_plus_data[j_probe]        = y_plus_i
        k_data[j_probe,:]           = k_i
        k_plus_data[j_probe,:]      = k_plus_i
        lambda_data[j_probe,:]      = lambda_i
        lambda_plus_data[j_probe,:] = lambda_plus_i
        Euu_data[j_probe,:]         = Euu_i
        Euu_plus_data[j_probe,:]    = Euu_plus_i

    # Check all probelines have same tavg0
    if np.all(np.isclose(tavg0_data, tavg0_data[0])):
        tavg0 = tavg0_data[0]   # scalar
    else:
        raise ValueError(f"Probelines have different averaging time, with tavg0_data = {tavg0_data:.6f}")

    print("Averaged probes y:", y_data)
    print("Averaged probes y+:", y_plus_data)

    return tavg0, y_data, y_plus_data, k_data, k_plus_data, lambda_data, lambda_plus_data, Euu_data, Euu_plus_data


def process_probeline_data(t_data, rho_data, rmsf_u_data, velocity_norm_data, params):

    # Problem parameters
    rho0               = params["rho0"]
    mu0                = params["mu0"]
    u_tau              = params["u_tau"]
    #gf_sigma          = params["gf_sigma"]
    #sgf_window_length = params["sgf_window_length"]
    #sgf_polyorder     = params["sgf_polyorder"]
    wavelength_limit   = params["wavelength_limit"]

    # Check Assumption 2: constant probeline time-step (required to calculate fft!)
    if np.all(np.isclose(t_data[1:]-t_data[:-1], t_data[1]-t_data[0])):
        dt = t_data[1] - t_data[0]
    else:
        raise ValueError(f"Assumption not satisfied: grid has stretching in x-direction (A_x>0) which transformed [3-D snapshot at spec. time] into [1-D probeline at spec. (x,y,z), increasing time] to have different dt along probeline evolution")
        
    # Direct Fast Fourier Transform (DFFT)
    fft_rhoufuf = np.fft.fft(rho_data * rmsf_u_data * rmsf_u_data)  #  Spectral turbulent kinetic energy density of the streamwise velocity
    fft_freq    = np.fft.fftfreq(len(t_data), dt)

    # Filtering negative frequencies
    positive_freq_indices = fft_freq > 0
    fft_freq    = fft_freq[positive_freq_indices]               # frequency (f) [Hz=1/s]
    fft_rhoufuf = fft_rhoufuf[positive_freq_indices]            # spectral TKE density [kg/(m·s^2)=J/m^3]
    N           = len(fft_freq)

    # Spatial wavelength and wavenumber, based on Taylor hypothesis
    # Source: https://gibbs.science/efd/lectures/lecture_24.pdf
    avg_velocity_norm = np.mean(velocity_norm_data)
    print(f"Time-averaged averaged-velocity-norm: {avg_velocity_norm:.3f}")
    spatial_wavenumber  = np.abs(fft_freq) / avg_velocity_norm  # spatial wavenumber (k) [1/m]
    spatial_wavelength  = ( (2*np.pi) / spatial_wavenumber )    # spatial wavelength (lambda) [m]  
    
    # Spectral turbulent kinetic energy density of the streamwise velocity (Euu)
    streamwise_spectrum = np.abs(fft_rhoufuf) / N
    
    # Sort all by increasing wavenumber / decreasing wavelength
    sorted_indices      = np.argsort(spatial_wavenumber)
    spatial_wavenumber  = spatial_wavenumber[sorted_indices]    # spatial wavenumber (k) [1/m]
    spatial_wavelength  = spatial_wavelength[sorted_indices]    # spatial wavelength (lambda) [m]  
    streamwise_spectrum = streamwise_spectrum[sorted_indices]   # spectral TKE density [kg/(m·s^2)=J/m^3]
    
    ### # 1st Smoothing: Apply smoothing by Gaussian filter
    ### streamwise_spectrum = gaussian_filter1d(streamwise_spectrum, sigma=gf_sigma, mode='nearest')
    ### #streamwise_spectrum = exponential_moving_average(streamwise_spectrum, alpha=0.3)
    ### #streamwise_spectrum = low_pass_filter(streamwise_spectrum, cutoff_freq=0.2, sample_rate=1)
    ### 
    ### # Premultiplied spectra: k * fft(rhoufuf) (k*Euu)
    ### streamwise_spectrum             = wavenumber * streamwise_spectrum
    ### nonfiltered_streamwise_spectrum = wavenumber * nonfiltered_streamwise_spectrum
    ### 
    ### # 2nd Smoothing: Apply smoothing by Savitzky-Golay polynomial regression filter
    ### premultiplied_streamwise_spectrum = savgol_filter(premultiplied_streamwise_spectrum, window_length=sgf_window_length, polyorder=sgf_polyorder)  
    
    # Truncate wavelengths below the grid cutoff
    truncated_indices   = (spatial_wavelength >= wavelength_limit) 
    spatial_wavenumber  = spatial_wavenumber[truncated_indices]             # spatial wavenumber (k) [1/m]
    spatial_wavelength  = spatial_wavelength[truncated_indices]             # spatial wavelength (lambda) [m]  
    streamwise_spectrum = streamwise_spectrum[truncated_indices]            # spectral TKE density (Euu) [kg/(m·s^2)]
    
    # Normalize in wall-units
    # (rho0 * u_tau / mu0) = [1/m]
    # (rho0 * u_tau**2) = [kg/(m·s^2)]
    spatial_wavenumber_plus  = spatial_wavenumber / (rho0 * u_tau / mu0)    # normalized spatial wavenumber in wall units (k+) [-]
    spatial_wavelength_plus  = spatial_wavelength * (rho0 * u_tau / mu0)    # normalized spatial wavelength in wall units (lambda+) [-]
    streamwise_spectrum_plus = streamwise_spectrum / (rho0 * u_tau**2)      # normalized spectral TKE density (Euu+) [-]
    return spatial_wavenumber, spatial_wavenumber_plus, spatial_wavelength, spatial_wavelength_plus, streamwise_spectrum, streamwise_spectrum_plus





#-----------------------------------------------------------------------------------------
#                                    Unique value dictionary
#-----------------------------------------------------------------------------------------

def check_uniform_dict(input_dict):
    """
    Check if all values in a dictionary are the same and return that value.
    Args:
        input_dict (dict): A dictionary with key-value pairs.
    Returns:
        Any: The common value if uniform, otherwise None.
    """
    # Use a set to determine if all values are the same
    unique_values = set(input_dict.values())
    if len(unique_values) == 1:
        # Return the single value in the set
        return unique_values.pop()
    else:
        # Return None if values are not uniform
        return None

def check_uniform_nested_dict(nested_dict):
    """
    Check if a dictionary of dictionaries has the same value everywhere and return that value.
    Args:
        nested_dict (dict): A dictionary containing other dictionaries.
    Returns:
        Any: The common value if uniform, otherwise None.
    """
    # Extract all values from the nested dictionaries
    all_values = [value for subdict in nested_dict.values() for value in subdict.values()]
    # Use a set to determine if all values are the same
    unique_values = set(all_values)
    if len(unique_values) == 1:
        # Return the single value in the set
        return unique_values.pop()
    else:
        # Return None if values are not uniform
        return None