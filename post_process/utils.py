import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

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
    input_h5_filepath, file_details, output_csv_directory, params
):

    print(f"\nProcessing h5 snapshot: {input_h5_filepath}")

    # Problem parameters
    num_grid_x = params["num_grid_x"] 
    num_grid_y = params["num_grid_y"]
    dt_dx = params["dt_dx"]
    n_probes = params["n_probes"]
    probes_zy_desired = params["probes_zy_desired"]

    # Get data: x, y, z, u, v, w; attributes: time
    with h5py.File(input_h5_filepath, 'r') as file:
        t0     = file.attrs['Time']
        x_data = file['x'][1:-1,1:-1,1:-1]    # 1:-1 to take only inner grid points
        y_data = file['y'][1:-1,1:-1,1:-1]
        z_data = file['z'][1:-1,1:-1,1:-1]
        u_data = file['u'][1:-1,1:-1,1:-1]
        v_data = file['v'][1:-1,1:-1,1:-1]
        w_data = file['w'][1:-1,1:-1,1:-1]
    num_points_z, num_points_y, num_points_x = u_data.shape
    assert num_points_x == num_grid_x & num_points_y == num_grid_y
    
    # Convert dx -> dt --> rebuild 'x_data' and 'time_data' to translate spatial advancement to temporal advancement
    x0 = x_data[0,0,0]
    dx_data = x_data - x0             # shape [num_points_z, num_points_y, num_points_x]
    t_data  = t0 + dt_dx * dx_data    # shape [num_points_z, num_points_y, num_points_x]
    x_data  = x0 * np.ones([num_points_z, num_points_y, num_points_x])

    # Find probes (z,y) coordinates closest to chosen 'zy_probes' pairs of coordinates
    # ASSUMPTION: regular grid!
    probes_zy = np.zeros([n_probes,2])                  # (z,y) coordinates pairs
    probes_kj = np.zeros([n_probes,2], dtype='int64')   # (k,j) indexes paris (on z,y-directions) 
    y_coords = y_data[0,:,0]
    z_coords = z_data[:,0,0] 
    for i_probe in range(n_probes):
        [z_i, y_i] = probes_zy_desired[i_probe,:]
        k_i = np.argmin(np.abs(z_coords - z_i));    z_i = z_coords[k_i]
        j_i = np.argmin(np.abs(y_coords - y_i));    y_i = y_coords[j_i]
        probes_kj[i_probe,:] = [k_i, j_i];          probes_zy[i_probe,:] = [z_i, y_i]
    print(f"\nDesired probes (z,y)-coordinates: \n{probes_zy_desired}")
    print(f"\nFound probes (z,y)-coordinates: \n{probes_zy}")
    print(f"\nFound probes (k,z)-index for (z,y)-coordinates: \n{probes_kj}")
    
    # Get probes data and store in h5 file
    print("\nSaving probelines data...")
    probes_filepath_list = []
    for i_probe in range(n_probes):
        # --- Get probeline data ---
        k,j = probes_kj[i_probe,:]
        t_ = t_data[k,j,:]    # 1-D array, length num_points_x (= num_points_t)
        x_ = x_data[k,j,:]
        y_ = y_data[k,j,:]
        z_ = z_data[k,j,:]
        u_ = u_data[k,j,:]
        v_ = v_data[k,j,:]
        w_ = w_data[k,j,:]
        # Check (x,y,z) = ct., and (t) increases for each probeline data point
        if np.all([np.isclose(x_, x_[0]), np.isclose(y_, y_[0]), np.isclose(z_, z_[0])]):
            x_probe = x_[0]
            y_probe = y_[0]
            z_probe = z_[0]
        else:
            raise ValueError(f"Error in probeline '{i_probe}' from snapshot '{input_h5_filepath}' which has different (x,y,z)-coords; all y-coords shoud be the same, as probeline corresponds to specific (x,)y,z coordinates, increasing in time")
        # --- Save probeline data ---
        ### # Save in .csv:
        ### fpath = os.path.join(output_csv_directory, f'probeline{i_probe}_k{k}_j{j}_{file_details}.csv')
        ### np.savetxt(fpath, 
        ###            X=np.array([t_, x_, y_, z_, u_, v_, w_]).T,
        ###            header='t[s],          x[m],            y[m],            z[m],            u[m/s],          v[m/s],           w[m/s]',
        ###            delimiter=',',
        ###            fmt='%.10e',
        ### )
        ### print(f"Saved probeline {i_probe} in: {fpath}")
        # Save in .csv (only save 1 single value for x,y,z coordinates, which are constant for all probeline data of a specific probeline)
        fpath = os.path.join(output_csv_directory, f'probeline{i_probe}_k{k}_j{j}_{file_details}.h5')
        with h5py.File(fpath,'w') as f:
            f.create_dataset("t_data", data=t_)
            f.create_dataset("u_data", data=u_)
            f.create_dataset("v_data", data=v_)
            f.create_dataset("w_data", data=w_)
            f.attrs['x_probe'] = x_probe
            f.attrs['y_probe'] = y_probe
            f.attrs['z_probe'] = z_probe
        print(f"\nProbe {i_probe} \nCoordinates: ({x_probe:.4f}, {y_probe:.4f}, {z_probe:.4f}) \nStored in: {fpath}")
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
    delta_viscous     = params["delta_viscous"]
    gf_sigma          = params["gf_sigma"]
    sgf_window_length = params["sgf_window_length"]
    sgf_polyorder     = params["sgf_polyorder"]
    wavelength_limit  = params["wavelength_limit"]

    ### # Get probeline data in .csv file
    ### data = np.loadtxt(file_path, delimiter=',')
    ### t_data = data[:, 0]
    ### x_data = data[:, 1]
    ### y_data = data[:, 2]
    ### z_data = data[:, 3]
    ### u_data = data[:, 4] - np.mean(data[:, 4])    # u-avg using only 1FTT data
    ### v_data = data[:, 5] - np.mean(data[:, 5])    # v-avg using only 1FTT data
    ### w_data = data[:, 6] - np.mean(data[:, 6])    # w-avg using only 1FFT data
    ### # Check all y-coordinates are equal, as probeline corresponds to specific (x,)y,z coordinates, increasing time
    ### if np.all([np.isclose(x_data, x_data[0]), np.isclose(y_data, y_data[0]), np.isclose(z_data, z_data[0])]):
    ###     x_probe = x_data[0]
    ###     y_probe = y_data[0]
    ###     z_probe = z_data[0]
    ###     print(f"Probeline coordinates: ({x_probe:.4f}, {y_probe:.4f}, {z_probe:.4f})")
    ### else:
    ###     raise ValueError(f"Error in probeline '{file_path}' with different (x,y,z)-coords; all y-coords shoud be the same, as probeline corresponds to specific (x,)y,z coordinates, increasing in time")
    # Get probeline data in .h5 file
    with h5py.File(file_path, 'r') as f:
        t_data  = f['t_data'][:]
        u_data  = f['u_data'][:]
        v_data  = f['v_data'][:]
        w_data  = f['w_data'][:]
        x_probe = f.attrs['x_probe']
        y_probe = f.attrs['y_probe']
        z_probe = f.attrs['z_probe']

    # Transform y to y+ coordinate
    isBottomWall = y_probe < delta
    if isBottomWall:
        print("Probeline in bottom wall")
        y_plus_probe = y_probe * rho0 * u_tau / mu0
    else: 
        print("Probeline in top wall")
        y_plus_probe = (2 * delta - y_probe) * rho0 * u_tau / mu0

    # Check Assumption 2: constant probeline time-step (required to calculate fft!)
    if np.all(np.isclose(t_data[1:]-t_data[:-1], t_data[1]-t_data[0])):
        dt = t_data[1] - t_data[0]
    else:
        raise ValueError(f"Assumption not satisfied: grid has stretching in x-direction (A_x>0) which transformed [3-D snapshot at spec. time] into [1-D probeline at spec. (x,y,z), increasing time] to have different dt along probeline evolution")
        
    # Direct Fast Fourier Transform (DFFT)
    fft_uf      = np.fft.fft(u_data)
    fft_rhoufuf = np.fft.fft(rho0 * u_data * u_data)
    fft_freq    = np.fft.fftfreq(len(t_data), dt)

    # Filtering negative frequencies
    positive_freq_indices = fft_freq > 0
    fft_freq    = fft_freq[positive_freq_indices]
    fft_uf      = fft_uf[positive_freq_indices]
    fft_rhoufuf = fft_rhoufuf[positive_freq_indices]
    N           = len(fft_freq)

    # Spatial wavelength and wavenumber, based on Taylor hypothesis
    spatial_wavelength = u_tau / np.abs(fft_freq)
    spatial_wavenumber = u_tau * np.abs(fft_freq)
    
    # Spectrum of streamwise momentum
    streamwise_spectrum = np.abs(fft_rhoufuf) / N
    
    # Sort all by increasing wavenumber / decreasing wavelength
    sorted_indices      = np.argsort(spatial_wavenumber)
    spatial_wavelength  = spatial_wavelength[sorted_indices]
    spatial_wavenumber  = spatial_wavenumber[sorted_indices]
    streamwise_spectrum = streamwise_spectrum[sorted_indices]
    nonfiltered_streamwise_spectrum = streamwise_spectrum.copy()
    
    # 1st Smoothing: Apply smoothing by Gaussian filter
    streamwise_spectrum = gaussian_filter1d(streamwise_spectrum, sigma=gf_sigma, mode='nearest')
    #streamwise_spectrum = exponential_moving_average(streamwise_spectrum, alpha=0.3)
    #streamwise_spectrum = low_pass_filter(streamwise_spectrum, cutoff_freq=0.2, sample_rate=1)
    
    # Premultiplied spectra: k * fft(rhoufuf)
    streamwise_spectrum             = streamwise_spectrum * spatial_wavenumber
    nonfiltered_streamwise_spectrum = nonfiltered_streamwise_spectrum * spatial_wavenumber

    # 2nd Smoothing: Apply smoothing by Savitzky-Golay polynomial regression filter
    streamwise_spectrum = savgol_filter(streamwise_spectrum, window_length=sgf_window_length, polyorder=sgf_polyorder)  
    # TODO: check if necessary to apply 2 filters: gaussian + sav-gol, or if only Gaussian filtering is enough  
    
    # Truncate wavelengths below the grid cutoff
    truncated_indices   = (spatial_wavelength >= wavelength_limit) 
    spatial_wavelength  = spatial_wavelength[truncated_indices]
    spatial_wavenumber  = spatial_wavenumber[truncated_indices]
    streamwise_spectrum = streamwise_spectrum[truncated_indices]
    nonfiltered_streamwise_spectrum = nonfiltered_streamwise_spectrum[truncated_indices]
    
    # Normalize
    normalized_spatial_wavelength              = spatial_wavelength / delta_viscous
    normalized_spatial_wavenumber              = spatial_wavenumber * delta_viscous
    normalized_streamwise_spectrum             = streamwise_spectrum / (rho0 * u_tau**2)
    normalized_nonfiltered_streamwise_spectrum = nonfiltered_streamwise_spectrum / (rho0 * u_tau**2)
    #print("min of normalized wavelength = ",np.amin(normalized_spatial_wavenumber))
    #print("max of normalized wavelength = ",np.amax(normalized_spatial_wavenumber))
    
    if False:
        plt.figure(figsize=(12, 6))
        plt.loglog(normalized_spatial_wavenumber, normalized_nonfiltered_streamwise_spectrum, label='Original Spectrum')
        plt.loglog(normalized_spatial_wavenumber, normalized_streamwise_spectrum, label='Smoothed Spectrum')
        plt.xlabel(r"Spatial wavenumber, $k$")
        plt.ylabel(r"Streamwise momentum spectrum, $kE_{\rho u u}$")
        plt.xscale('log')
        plt.title('Comparison of Original vs Smoothed Spectrum')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(params["train_post_process_dir"], "streamwise_momentum_spectra.jpg")
        plt.savefig(fname)
        print(f"Plot original vs. smoothed streamwise momentum spectra in: {fname}")

    return y_plus_probe, normalized_spatial_wavelength, normalized_spatial_wavenumber, normalized_nonfiltered_streamwise_spectrum, normalized_streamwise_spectrum



    def process_probelines_list(probes_filepath_list, file_details, params):
        
        print(f"\nProcessing probelines list...")
        
        # Get probelines data from each probeline h5 file
        y_plus_dict = {}
        wavelength_dict = {}
        wavenumber_dict = {}
        nonfiltered_streamwise_momentum_spectra_dict = {}
        streamwise_momentum_spectra_dict = {}
        n_probes = len(probes_filepath_list)
        for i_probe in range(n_probes):
            file_path = probes_filepath_list[i_probe]
            y_plus, wavelength, wavenumber, nonfiltered_streamwise_momentum_spectra, streamwise_momentum_spectra \
                = process_probeline_h5(file_path, params)
            y_plus_dict[i_probe] = y_plus
            wavelength_dict[i_probe] = wavelength
            wavenumber_dict[i_probe] = wavenumber
            nonfiltered_streamwise_momentum_spectra_dict[i_probe] = nonfiltered_streamwise_momentum_spectra
            streamwise_momentum_spectra_dict[i_probe] = streamwise_momentum_spectra
        
        # Average streamwise-momentum-spectra for probes with same y_plus coordinate
        averaged_probes_dict = {}
        # TODO: continue here!