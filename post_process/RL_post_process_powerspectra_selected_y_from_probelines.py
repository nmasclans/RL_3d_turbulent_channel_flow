import glob
import h5py
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from utils import check_uniform_dict, check_uniform_nested_dict, process_probeline_data
from ChannelVisualizer import ChannelVisualizer

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---
try :
    train_name      = sys.argv[1]
    Re_tau          = float(sys.argv[2])     # Friction Reynolds number [-]
    dt_phys         = float(sys.argv[3])
    t_episode_train = float(sys.argv[4])
    case_dir        = sys.argv[5]
    print(f"\nScript parameters: \n- Train name: {train_name} \n- Re_tau: {Re_tau} \n- dt_phys: {dt_phys} \n- t_episode_train: {t_episode_train} \n- Case directory: {case_dir}")
except :
    raise ValueError("Missing call arguments, should be: <iteration> <ensemble> <train_name> <Re_tau> <t_episode_train> <case_dir>")

# Training post-processing directory
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# RL probelines data directory
probelines_dir_RL = os.path.join(case_dir, "rhea_exp", "temporal_time_probes")

# Reference & non-RL probelines data directory
file_path = os.path.dirname(os.path.abspath(__file__))
compare_dataset_dir  = os.path.join(file_path, f"data_Retau{Re_tau:.0f}")
probelines_dir_nonRL = os.path.join(compare_dataset_dir, "probelines_notConvStat") 
probelines_dir_ref   = os.path.join(compare_dataset_dir, "probelines_reference") 

# Flow parameters 
if np.isclose(Re_tau, 100, atol=1e-8):  # Re_tau = 100
    Re_tau = 100.0
elif np.isclose(Re_tau, 180, atol=1e-8):
    Re_tau = 180.0
else:
    raise ValueError(f"Not implemented Re_tau = {Re_tau}")
u_tau = 1.0
rho0  = 1.0  
delta = 1.0
mu0   = rho0 * u_tau / Re_tau  # = 1.0 / Re_tau
nu0   = mu0 / rho0             # = 1.0 / Re_tau
print(f"\nFlow parameters: \n- Re_tau: {Re_tau}\n- u_tau: {u_tau}\n- rho0: {rho0}\n- mu0: {mu0}\n- nu0: {nu0}")

# tavg0, time at which statistic averaging is activated (idem for non-RL and RL)
tavg0_ref   = 229.99999999
tavg0_nonRL = 318.99999999
tavg0_RL    = 318.99999999

# Probes y-coordinates at RL agent location - center y-coordinate of control cubes
# > probes distributed along z-axis:
n_probes_z_coord = 5
z_coord_name_list = [f"{i+1:.0f}" for i in range(n_probes_z_coord)] 
# > probes distributed along y-axis:
if Re_tau == 100.0:
    # Selected y-coordinates at RL agents center of action domain
    probes_y_coord = np.sort(np.array([0.125, 0.375, 0.625, 0.875])) #, 1.125, 1.375, 1.625, 1.875]))
    num_grid_y     = 64
    num_grid_x     = 64
elif Re_tau == 180.0:
    probes_y_coord = np.sort(np.array([0.059369, 0.208542, 0.4811795, 0.819736])) #, 1.18026, 1.51882, 1.79146, 1.94063]))
    num_grid_y     = 256
    num_grid_x     = 128
else:
    raise ValueError(f"Unknown 'y_probes' for Re_tau = {Re_tau}")
y_coord_name_list = [f"{y_coord:.3f}".replace(".", "") for y_coord in probes_y_coord]
n_probes_y_coord  = len(y_coord_name_list)
print(f"\nSelected probelines y-coords: \nValues: {probes_y_coord} \nFormatted name: {y_coord_name_list}")
# > all probes
n_probes    = n_probes_y_coord * n_probes_z_coord
probes_name = []
y_coord_name_vs_probes_name_dict = {key:[] for key in y_coord_name_list}
for j in range(n_probes_y_coord):
    for k in range(n_probes_z_coord):
        probe_name = f"{y_coord_name_list[j]}_{z_coord_name_list[k]}"
        probes_name.append(probe_name)
        y_coord_name_vs_probes_name_dict[y_coord_name_list[j]].append(probe_name)
# Debugging
print(f"\nSelected total of {n_probes} probes, named as:\n" + "\n".join(probes_name))
print(f"\nProbe names per each y-coord:")
for k,v in y_coord_name_vs_probes_name_dict.items():
    print(f"y-coord name: {k}, probes name: {v}")

# Filters parameters
gf_sigma          = 7   # Gaussian filter 'sigma'
sgf_window_length = 11  # Savitzky-Golay filter 'window_length'
sgf_polyorder     = 5   # Savitzky-Golay filter 'polyorder', polynomial order

# Fourier Transform Visualization parameters
y_plus_max_urms  = 12
y_plus_max_u     = 20
y_plus_max_cp    = 140
y_limit          = y_plus_max_u
wavelength_limit = 1e-9
log_smooth       = False
fontsize         = 18

# Probelines csv parameters
time_key = "# t [s]"
y_key    = " y [m]"
y_key2   = " y[m]"
rho_key  = " rho [kg/m3]"
u_key    = " u [m/s]"
v_key    = " v [m/s]"
w_key    = " w [m/s]"
#rmsf_u_key = " rmsf_u [m/s]"; avg_u_key  = " avg_u [m/s]"; avg_v_key  = " avg_v [m/s]"; avg_w_key  = " avg_w [m/s]"
vars_keys  = [time_key, y_key,  rho_key, u_key, v_key, w_key]
vars_keys2 = [time_key, y_key2, rho_key, u_key, v_key, w_key]

# Initialize visualizer
visualizer = ChannelVisualizer(postDir)

# Params dictionary
params = {
    "rho0": rho0,
    "mu0": mu0,
    "u_tau": u_tau,
    "delta": delta,
    "dt_phys": dt_phys,
    "gf_sigma": gf_sigma,
    "sgf_window_length": sgf_window_length,
    "sgf_polyorder": sgf_polyorder,
    "wavelength_limit": wavelength_limit,
}

#--------------------------------------------------------------------------------------------

# --- Reference & non-RL filenames ---
filename_dict_ref   = {}
filename_dict_nonRL = {}
for probe in probes_name:
    filename_dict_ref[probe]   = os.path.join(probelines_dir_ref,   f"temporal_point_probe_y_plus_{probe}.csv")
    filename_dict_nonRL[probe] = os.path.join(probelines_dir_nonRL, f"temporal_point_probe_y_plus_{probe}.csv")
# Debugging
print("\nFilename list of Reference Probelines:")
for k,v in filename_dict_ref.items():
    print(f"{k}: {v}")
print("\nFilename list of non-RL Probelines:")
for k,v in filename_dict_nonRL.items():
    print(f"{k}: {v}")

# --- RL filenames ---
filename_dict_RL_        = {key: [] for key in probes_name}
global_step_dict_RL_     = {key: [] for key in probes_name}
n_train_episodes_dict_RL = {key: 0 for key in probes_name}
for probe in probes_name:
    pattern        = f"{probelines_dir_RL}/temporal_point_probe_y_plus_{probe}_step*.csv"
    matching_files = sorted(glob.glob(pattern))
    for file in matching_files:
        filename_dict_RL_[probe].append(file)
        base_filename = os.path.basename(file)
        global_step_str = base_filename.split('_')[-1].replace('.csv', '')
        global_step_num = int(global_step_str[4:])
        global_step_dict_RL_[probe].append(global_step_num)
        n_train_episodes_dict_RL[probe] += 1

# --- Update RL dictionary distinguishing by episodes ---
# Check all probes have same number of episodes
n_train_episodes = check_uniform_dict(n_train_episodes_dict_RL)
del n_train_episodes_dict_RL
# Define episodes name
episodes_id_RL   = [ep for ep in np.arange(n_train_episodes)]
episodes_name_RL = [f"episode_{ep}" for ep in episodes_id_RL]
n_episodes_RL    = len(episodes_name_RL)
# Re-structure RL dictionaries
filename_dict_RL         = {probe: {episode: "" for episode in episodes_name_RL} for probe in probes_name} 
global_step_dict_RL      = {probe: {episode: 0  for episode in episodes_name_RL} for probe in probes_name} 
for p_i in range(n_probes):
    for e_i in range(n_train_episodes):
        filename_dict_RL[probes_name[p_i]][episodes_name_RL[e_i]]    = filename_dict_RL_[probes_name[p_i]][e_i]
        global_step_dict_RL[probes_name[p_i]][episodes_name_RL[e_i]] = global_step_dict_RL_[probes_name[p_i]][e_i]
# Debugging
print(f"\nNumber of RL train episodes: {n_train_episodes}, with episodes name:\n{episodes_name_RL}") 
print("\nRL probelines:")
for probe in probes_name:
    for episode in episodes_name_RL:
        print(f"[Probe {probe}, Episode {episode}] Filename: {filename_dict_RL[probe][episode]}, global step: {global_step_dict_RL[probe][episode]}")

# --- Check filenames exist ---
for probe in probes_name:
    file_ref = filename_dict_ref[probe]
    if not os.path.isfile(file_ref):
        print(f"Error: File '{file_ref}' not found.")
        sys.exit(1)
    file_nonRL = filename_dict_nonRL[probe]
    if not os.path.isfile(file_nonRL):
        print(f"Error: File '{file_nonRL}' not found.")
        sys.exit(1)
    for episode in episodes_name_RL:
        file_RL = filename_dict_RL[probe][episode]
        if not os.path.isfile(file_RL):
            print(f"Error: File '{file_RL}' not found.")
            sys.exit(1)
print("\nAll files exist :)")

#--------------------------------------------------------------------------------------------
# --- Get probelines data ---
print("\nImporting probelines data...")

# --- Reference data ---
print("\nImporting probelines reference data...")
tavg_atEpStart_dict_ref = {}
time_atEpStart_dict_ref = {}
time_dict_ref     = {}
y_dict_ref        = {}
y_plus_dict_ref   = {}
rho_dict_ref      = {}
uf_dict_ref       = {}
vel_norm_dict_ref = {}
for probe in probes_name:
    # Get data from csv file
    file = filename_dict_ref[probe]
    data = pd.read_csv(file, usecols=vars_keys2)
    time_data   = data[time_key].to_numpy()
    y_data      = data[y_key2].to_numpy()
    rho_data    = data[rho_key].to_numpy()
    u_data      = data[u_key].to_numpy()
    v_data      = data[v_key].to_numpy()
    w_data      = data[w_key].to_numpy()
    # store data in allocation dicts for single train episode period
    time_relative_data   = time_data - time_data[0]     # only interested in relative values, dt, not absolute value
    is_train_episode     = time_relative_data <= t_episode_train
    tavg_atEpStart_dict_ref[probe] = time_data[0] - tavg0_ref
    time_atEpStart_dict_ref[probe] = time_data[0]
    time_dict_ref[probe] = time_relative_data[is_train_episode]
    assert np.allclose(y_data, y_data[0])
    y_value              = float(y_data[0])
    y_dict_ref[probe]    = y_value
    isBottomWall         = y_value < delta
    if isBottomWall:
        y_plus_dict_ref[probe] = y_value * rho0 * u_tau / mu0
    else:
        y_plus_dict_ref[probe] =  (2 * delta - y_value) * rho0 * u_tau / mu0
    rho_dict_ref[probe]  = rho_data[is_train_episode]
    uf_dict_ref[probe]   = u_data[is_train_episode] - np.mean(u_data[is_train_episode])
    vel_norm_dict_ref[probe] = np.sqrt(u_data[is_train_episode]**2 + v_data[is_train_episode]**2 + w_data[is_train_episode]**2)

# --- RL data ---
print("\nImporting probelines RL data...")
tavg_atEpStart_dict_RL = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
time_atEpStart_dict_RL = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
time_dict_RL           = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
y_dict_RL              = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
y_plus_dict_RL         = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
rho_dict_RL            = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
uf_dict_RL             = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
vel_norm_dict_RL       = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
for probe_idx in range(n_probes):
    probe = probes_name[probe_idx]
    print(f"{probe_idx / n_probes * 100:.0f}%")
    for episode in episodes_name_RL:
        # Get data from csv file
        file = filename_dict_RL[probe][episode]
        data = pd.read_csv(file, usecols=vars_keys)
        time_data   = data[time_key].to_numpy()
        y_data      = data[y_key].to_numpy()
        rho_data    = data[rho_key].to_numpy()
        u_data      = data[u_key].to_numpy()
        v_data      = data[v_key].to_numpy()
        w_data      = data[w_key].to_numpy()
        # store data in allocation dicts for single train episode period
        time_relative_data  = time_data - time_data[0]     # only interested in relative values, dt
        is_train_episode    = time_relative_data <= t_episode_train
        tavg_atEpStart_dict_RL[probe][episode] = float(time_data[0] - tavg0_RL)
        time_atEpStart_dict_RL[probe][episode] = float(time_data[0])
        time_dict_RL[probe][episode]           = time_relative_data[is_train_episode]
        assert np.allclose(y_data, y_data[0])
        y_value                   = float(y_data[0])
        y_dict_RL[probe][episode] = y_value
        isBottomWall              = y_value < delta
        if isBottomWall:
            y_plus_dict_RL[probe][episode] = y_value * rho0 * u_tau / mu0
        else:
            y_plus_dict_RL[probe][episode] =  (2 * delta - y_value) * rho0 * u_tau / mu0
        rho_dict_RL[probe][episode]        = rho_data[is_train_episode]
        uf_dict_RL[probe][episode]         = u_data[is_train_episode] - np.mean(u_data[is_train_episode])
        vel_norm_dict_RL[probe][episode]   = np.sqrt(u_data[is_train_episode]**2 + v_data[is_train_episode]**2 + w_data[is_train_episode]**2)
tavg_atEpStart_RL = check_uniform_nested_dict(tavg_atEpStart_dict_RL)
time_atEpStart_RL = check_uniform_nested_dict(time_atEpStart_dict_RL)
del tavg_atEpStart_dict_RL, time_atEpStart_dict_RL

# --- non-RL data ---
# Get non-RL episodes
file = filename_dict_nonRL[probes_name[0]]
data = pd.read_csv(file, usecols=vars_keys2)
time_data   = data[time_key].to_numpy()
episodes_id_data_nonRL = np.floor((time_data-time_atEpStart_RL)/t_episode_train).astype(int)
episodes_id_set_nonRL  = sorted(set(episodes_id_data_nonRL))
episodes_id_nonRL      = [int(ep) for ep in episodes_id_set_nonRL if ep <= max(episodes_id_RL)]
episodes_name_nonRL    = [f"episode_{ep}" for ep in episodes_id_nonRL]
n_episodes_nonRL       = len(episodes_name_nonRL)
episodes_name_id_dict_nonRL = {episodes_name_nonRL[i]:episodes_id_nonRL[i] for i in range(n_episodes_nonRL)} 
# Get non-RL data
print("\nImporting probelines non-RL data... (only required data from equivalent RL episodes)")
tavg_atEpStart_dict_nonRL = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
time_atEpStart_dict_nonRL = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
time_dict_nonRL           = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
y_dict_nonRL              = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
y_plus_dict_nonRL         = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
rho_dict_nonRL            = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
uf_dict_nonRL             = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
vel_norm_dict_nonRL       = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
for probe_idx in range(n_probes):
    probe = probes_name[probe_idx]
    print(f"{probe_idx / n_probes * 100:.0f}%")
    # Get data from csv file
    file = filename_dict_nonRL[probe]
    data = pd.read_csv(file, usecols=vars_keys2)
    time_data   = data[time_key].to_numpy()
    tavg_data   = time_data - tavg0_nonRL
    y_data      = data[y_key2].to_numpy()
    rho_data    = data[rho_key].to_numpy()
    u_data      = data[u_key].to_numpy()
    v_data      = data[v_key].to_numpy()
    w_data      = data[w_key].to_numpy()
    episodes_id_data = np.floor((time_data-time_atEpStart_RL)/t_episode_train).astype(int)
    # store data in allocation dicts for each episode period
    for episode in episodes_name_nonRL:
        episode_id      = episodes_name_id_dict_nonRL[episode]
        is_episode_idxs = np.where(episodes_id_data == episode_id)[0]
        tavg_atEpStart_dict_nonRL[probe][episode] = float(time_data[is_episode_idxs[0]] - tavg0_RL)
        time_atEpStart_dict_nonRL[probe][episode] = float(time_data[is_episode_idxs[0]])
        time_dict_nonRL[probe][episode]     = time_data[is_episode_idxs] - time_data[is_episode_idxs[0]]
        assert np.allclose(y_data[is_episode_idxs], y_data[is_episode_idxs[0]])
        y_value                             = float(y_data[is_episode_idxs[0]])
        y_dict_nonRL[probe][episode]        = y_value
        isBottomWall                        = y_value < delta
        if isBottomWall:
            y_plus_dict_nonRL[probe][episode] = y_value * rho0 * u_tau / mu0
        else:
            y_plus_dict_nonRL[probe][episode] =  (2 * delta - y_value) * rho0 * u_tau / mu0
        rho_dict_nonRL[probe][episode]      = rho_data[is_episode_idxs]
        uf_dict_nonRL[probe][episode]       = u_data[is_episode_idxs] - np.mean(u_data[is_episode_idxs])
        vel_norm_dict_nonRL[probe][episode] = np.sqrt(u_data[is_episode_idxs]**2 + v_data[is_episode_idxs]**2 + w_data[is_episode_idxs]**2)
        #print(f"[Probe {probe}, Episode {episode}] Number of temporal data points: {len(is_episode_idxs)}")


#--------------------------------------------------------------------------------------------
# Calculate TKE spectra from probelines data
print("\nProcess probelines data to calculate TKE spectra...")

k_plus_dict_ref     = {probe: 0.0 for probe in probes_name}
Euu_plus_dict_ref   = {probe: 0.0 for probe in probes_name}
k_plus_dict_RL      = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
Euu_plus_dict_RL    = {probe: {episode: None for episode in episodes_name_RL} for probe in probes_name}
k_plus_dict_nonRL   = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
Euu_plus_dict_nonRL = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
for probe_idx in range(n_probes):
    probe = probes_name[probe_idx]
    print(f"{probe_idx / n_probes * 100:.0f}%")
    
    # --- Reference data ---
    _, k_plus_dict_ref[probe], _, _, _, Euu_plus_dict_ref[probe] = \
        process_probeline_data(time_dict_ref[probe], rho_dict_ref[probe], 
                               uf_dict_ref[probe], params)
    
    # --- RL data ---
    for episode in episodes_name_RL:
        _, k_plus_dict_RL[probe][episode], _, _, _, Euu_plus_dict_RL[probe][episode] = \
            process_probeline_data(time_dict_RL[probe][episode], rho_dict_RL[probe][episode], 
                                   uf_dict_RL[probe][episode], params)

    # --- nonRL data ---
    for episode in episodes_name_nonRL:
        _, k_plus_dict_nonRL[probe][episode], _, _, _, Euu_plus_dict_nonRL[probe][episode] = \
            process_probeline_data(time_dict_nonRL[probe][episode], rho_dict_nonRL[probe][episode], 
                                   uf_dict_nonRL[probe][episode], params)

#--------------------------------------------------------------------------------------------
# Average probelines spectral data for same y-coords (different z-coords)
print("\nInterpolating & Averaging probelines spectral data...")

# Get avg_k for first averaging probe, which will be used as k-coord to interpolate Euu(k)
avg_k_plus_dict_ref   = {y_coord: None for y_coord   in y_coord_name_list}
n_k_dict_ref          = {y_coord: 0    for y_coord   in y_coord_name_list}
avg_k_plus_dict_RL    = {y_coord: {episode: None for episode in episodes_name_RL}    for y_coord in y_coord_name_list}
n_k_dict_RL           = {y_coord: {episode: 0    for episode in episodes_name_RL}    for y_coord in y_coord_name_list}
avg_k_plus_dict_nonRL = {y_coord: {episode: None for episode in episodes_name_nonRL} for y_coord in y_coord_name_list}
n_k_dict_nonRL        = {y_coord: {episode: 0    for episode in episodes_name_nonRL} for y_coord in y_coord_name_list}
for y_coord in y_coord_name_list:
    avg_probes = y_coord_name_vs_probes_name_dict[y_coord]
    probe = avg_probes[0]
    # > Reference
    avg_k_plus_dict_ref[y_coord] = k_plus_dict_ref[probe]
    n_k_dict_ref[y_coord]        = len(k_plus_dict_ref[probe])
    # > RL
    for episode in episodes_name_RL:
        avg_k_plus_dict_RL[y_coord][episode] = k_plus_dict_RL[probe][episode]
        n_k_dict_RL[y_coord][episode]        = len(k_plus_dict_RL[probe][episode])
    # > non-RL
    for episode in episodes_name_nonRL:
        avg_k_plus_dict_nonRL[y_coord][episode] = k_plus_dict_nonRL[probe][episode]
        n_k_dict_nonRL[y_coord][episode]        = len(k_plus_dict_nonRL[probe][episode])

# Interpolate Euu at k-coord avg_k, with unique avg_k for each y_coord
interp_Euu_plus_dict_ref   = {probe: None for probe in probes_name}
interp_Euu_plus_dict_RL    = {probe: {episode: None for episode in episodes_name_RL}    for probe in probes_name}
interp_Euu_plus_dict_nonRL = {probe: {episode: None for episode in episodes_name_nonRL} for probe in probes_name}
for y_coord in y_coord_name_list:
    avg_probes   = y_coord_name_vs_probes_name_dict[y_coord]
    for probe in avg_probes:
        # > Reference
        interp_func = interp1d(k_plus_dict_ref[probe], Euu_plus_dict_ref[probe], kind='linear', fill_value='extrapolate')
        interp_Euu_plus_dict_ref[probe] = interp_func(avg_k_plus_dict_ref[y_coord])
        # > RL
        for episode in episodes_name_RL:
            interp_func = interp1d(k_plus_dict_RL[probe][episode], Euu_plus_dict_RL[probe][episode], kind='linear', fill_value='extrapolate')
            interp_Euu_plus_dict_RL[probe][episode] = interp_func(avg_k_plus_dict_RL[y_coord][episode])
        # > non-RL
        for episode in episodes_name_nonRL:
            interp_func = interp1d(k_plus_dict_nonRL[probe][episode], Euu_plus_dict_nonRL[probe][episode], kind='linear', fill_value='extrapolate')
            interp_Euu_plus_dict_nonRL[probe][episode] = interp_func(avg_k_plus_dict_nonRL[y_coord][episode])

# Averaged interpolated_Euu on avg_k
avg_Euu_plus_dict_ref   = {y_coord: None for y_coord in y_coord_name_list}
avg_Euu_plus_dict_RL    = {y_coord: {episode: None for episode in episodes_name_RL}    for y_coord in y_coord_name_list}
avg_Euu_plus_dict_nonRL = {y_coord: {episode: None for episode in episodes_name_nonRL} for y_coord in y_coord_name_list}
# Initialise avg_y and avg_tavg for first avg_probe, and assert they are equal for all of them
avg_y_plus_dict               = {y_coord: y_plus_dict_ref[(y_coord_name_vs_probes_name_dict[y_coord])[0]] for y_coord in y_coord_name_list}
avg_tavg_atEpStart_dict_nonRL = {episode: tavg_atEpStart_dict_nonRL[probes_name[0]][episode] for episode in episodes_name_nonRL}
for y_coord in y_coord_name_list:
    # Initialize avg arrays
    avg_Euu_plus_dict_ref[y_coord] = np.zeros(n_k_dict_ref[y_coord])
    for episode in episodes_name_RL:
        avg_Euu_plus_dict_RL[y_coord][episode] = np.zeros(n_k_dict_RL[y_coord][episode])
    for episode in episodes_name_nonRL:
        avg_Euu_plus_dict_nonRL[y_coord][episode] = np.zeros(n_k_dict_nonRL[y_coord][episode])
    # Calcualte avg quantities
    avg_probes   = y_coord_name_vs_probes_name_dict[y_coord]
    n_avg_probes = len(avg_probes)
    for probe in avg_probes:
        # > Reference
        avg_Euu_plus_dict_ref[y_coord] += interp_Euu_plus_dict_ref[probe]
        assert avg_y_plus_dict[y_coord] == y_plus_dict_ref[probe], f"{avg_y_plus_dict[y_coord]} != {y_plus_dict_ref[probe]}"
        # > RL
        for episode in episodes_name_RL:
            avg_Euu_plus_dict_RL[y_coord][episode] += interp_Euu_plus_dict_RL[probe][episode]
            assert avg_y_plus_dict[y_coord] == y_plus_dict_RL[probe][episode], f"{avg_y_plus_dict[y_coord]} != {y_plus_dict_RL[probe][episode]}"
        # > non-RL
        for episode in episodes_name_nonRL:
            avg_Euu_plus_dict_nonRL[y_coord][episode] += interp_Euu_plus_dict_nonRL[probe][episode]
            assert avg_y_plus_dict[y_coord] == y_plus_dict_nonRL[probe][episode], f"{avg_y_plus_dict[y_coord]} != {y_plus_dict_nonRL[probe][episode]}"
            assert avg_tavg_atEpStart_dict_nonRL[episode] == tavg_atEpStart_dict_nonRL[probe][episode], f"{avg_tavg_atEpStart_dict_nonRL[episode]} != {tavg_atEpStart_dict_nonRL[probe][episode]}"
    avg_Euu_plus_dict_ref[y_coord] /= n_avg_probes
    for episode in episodes_name_RL:
        avg_Euu_plus_dict_RL[y_coord][episode] /= n_avg_probes
    for episode in episodes_name_nonRL:
        avg_Euu_plus_dict_nonRL[y_coord][episode] /= n_avg_probes

#--------------------------------------------------------------------------------------------
# Plot TKE spectra
print("\nBuild TKE spectra...")

#import pdb; pdb.set_trace()

# Reverse dictionaryes [y_coords][episodes] to [episodes][y_coords]
avg_k_plus_dict_RL_rev      = {episode: {y_coord: avg_k_plus_dict_RL[y_coord][episode]      for y_coord in y_coord_name_list} for episode in episodes_name_RL}
avg_Euu_plus_dict_RL_rev    = {episode: {y_coord: avg_Euu_plus_dict_RL[y_coord][episode]    for y_coord in y_coord_name_list} for episode in episodes_name_RL}
avg_k_plus_dict_nonRL_rev   = {episode: {y_coord: avg_k_plus_dict_nonRL[y_coord][episode]   for y_coord in y_coord_name_list} for episode in episodes_name_nonRL}
avg_Euu_plus_dict_nonRL_rev = {episode: {y_coord: avg_Euu_plus_dict_nonRL[y_coord][episode] for y_coord in y_coord_name_list} for episode in episodes_name_nonRL}

frames_spectral = []
for episode in episodes_name_nonRL:
    if episode in episodes_name_RL:
        frames_spectral = visualizer.build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_frame_from_dicts(
                frames_spectral,
                avg_y_plus_dict,
                avg_k_plus_dict_RL_rev[episode],   avg_k_plus_dict_nonRL_rev[episode],     avg_k_plus_dict_ref,
                avg_Euu_plus_dict_RL_rev[episode], avg_Euu_plus_dict_nonRL_rev[episode],   avg_Euu_plus_dict_ref, 
                tavg_atEpStart_RL,                 avg_tavg_atEpStart_dict_nonRL[episode], episodes_name_id_dict_nonRL[episode],
                ylim=[10^(-5), 10^(1)],
        )
    else:
        frames_spectral = visualizer.build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_frame_from_dicts(
                frames_spectral,
                avg_y_plus_dict, 
                None, avg_k_plus_dict_nonRL_rev[episode],     avg_k_plus_dict_ref,
                None, avg_Euu_plus_dict_nonRL_rev[episode],   avg_Euu_plus_dict_ref, 
                None, avg_tavg_atEpStart_dict_nonRL[episode], episodes_name_id_dict_nonRL[episode],
                ylim=[10^(-5), 10^(1)],
        )

print("\nSave gifs from frames...")
frames_dict = {'spectral_Euu+_vs_k+': frames_spectral}
visualizer.build_main_gifs_from_frames(frames_dict)
print("Gifs plotted successfully!")


###visualizer.plot_spectral_turbulent_kinetic_energy_density_streamwise_velocity(
###    file_details_RL[i], tavg0_RL, avg_y_RL, avg_y_plus_RL, avg_k_RL, avg_k_plus_RL, avg_lambda_RL, avg_lambda_plus_RL, avg_Euu_RL, avg_Euu_plus_RL)
###visualizer.plot_spectral_turbulent_kinetic_energy_density_streamwise_velocity(
###    file_details_nonRL[i], tavg0_nonRL, avg_y_nonRL, avg_y_plus_nonRL, avg_k_nonRL, avg_k_plus_nonRL, avg_lambda_nonRL, avg_lambda_plus_nonRL, avg_Euu_nonRL, avg_Euu_plus_nonRL)
    
# Build frame for each specific global step
    

# # --- 3d-spatial snapshots to 1d-temporal probelines & build plots---

# # - Transform 3-d spatial snapshot (at specific time, for all domain grid points (x,y,z)) 
# # into 1-d probelines temporal data (at specific (x,y,z), increasing time) 
# # - Build plots
# frames_spectral = []

# # > Reference
# probes_filepath_list_ref = build_probelines_from_snapshot_h5(filename_ref, file_details_ref, probelinesDir, params)
# tavg0_ref, avg_y_ref, avg_y_plus_ref, avg_k_ref, avg_k_plus_ref, avg_lambda_ref, avg_lambda_plus_ref, avg_Euu_ref, avg_Euu_plus_ref \
#     = process_probelines_list(probes_filepath_list_ref, file_details_ref, params)

# for i in range(n_RL):
#     # > RL
#     probes_filepath_list_RL = build_probelines_from_snapshot_h5(filename_RL_list[i], file_details_RL[i], probelinesDir, params)
#     tavg0_RL, avg_y_RL, avg_y_plus_RL, avg_k_RL, avg_k_plus_RL, avg_lambda_RL, avg_lambda_plus_RL, avg_Euu_RL, avg_Euu_plus_RL \
#         = process_probelines_list(probes_filepath_list_RL, file_details_RL[i], params)
#     # > non-RL
#     probes_filepath_list_nonRL = build_probelines_from_snapshot_h5(filename_nonRL_list[i], file_details_nonRL[i], probelinesDir, params)
#     tavg0_nonRL, avg_y_nonRL, avg_y_plus_nonRL, avg_k_nonRL, avg_k_plus_nonRL, avg_lambda_nonRL, avg_lambda_plus_nonRL, avg_Euu_nonRL, avg_Euu_plus_nonRL \
#         = process_probelines_list(probes_filepath_list_nonRL, file_details_nonRL[i], params)

#     # Plot spectrum for each specific global step 
#     ###visualizer.plot_spectral_turbulent_kinetic_energy_density_streamwise_velocity(
#     ###    file_details_RL[i], tavg0_RL, avg_y_RL, avg_y_plus_RL, avg_k_RL, avg_k_plus_RL, avg_lambda_RL, avg_lambda_plus_RL, avg_Euu_RL, avg_Euu_plus_RL)
#     ###visualizer.plot_spectral_turbulent_kinetic_energy_density_streamwise_velocity(
#     ###    file_details_nonRL[i], tavg0_nonRL, avg_y_nonRL, avg_y_plus_nonRL, avg_k_nonRL, avg_k_plus_nonRL, avg_lambda_nonRL, avg_lambda_plus_nonRL, avg_Euu_nonRL, avg_Euu_plus_nonRL)
    
#     # Check probes y+ are equal for all RL, non-RL, ref
#     for i_probe in range(n_probes):
#         assert np.isclose(avg_y_plus_ref[i_probe], avg_y_plus_RL[i_probe]) & np.isclose(avg_y_plus_ref[i_probe], avg_y_plus_nonRL[i_probe])
    
#     # Build frame for each specific global step
#     frames_spectral = visualizer.build_spectral_turbulent_kinetic_energy_density_streamwise_velocity_frame(
#         frames_spectral, avg_y_plus_ref, avg_k_plus_RL, avg_k_plus_nonRL, avg_k_plus_ref, avg_Euu_plus_RL, avg_Euu_plus_nonRL, avg_Euu_plus_ref, tavg0_RL, tavg0_nonRL, global_step_num_list[i],
#     )

# print("\nSave gifs from frames...")
# frames_dict = {'spectral_Euu+_vs_k+':frames_spectral}
# visualizer.build_main_gifs_from_frames(frames_dict)
# print("Gifs plotted successfully!")

# """
# def plot_colormap(directory, resolution):
#     all_spectra = []
#     for directory in directories:
#         directory_path = os.path.join('.', directory)
#         csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

#         y_positions_list = []
#         spectra_list = []
#         k_values_list = []
#         bulk_rho = []
#         bulk_mu = []
#         limit_wavelength = 2*delta / resolution

#         for csv_file in csv_files:
#             file_path = os.path.join(directory_path, csv_file)

#             k_value, spectrum, y_position, mean_rho, mean_mu = process_file(file_path, limit_wavelength)
#             if csv_file == csv_files[0]:
#                     k_values_list.append(k_value)
#             if (y_position > 0) & (y_position < y_limit):
#                 print(f"Processing: {file_path}") 
                
#                 bulk_rho.append(mean_rho)
#                 bulk_mu.append(mean_mu)
#                 y_positions_list.append(y_position)
#                 spectra_list.append(spectrum)

#         bulk_mu = np.mean(bulk_mu)
#         bulk_rho = np.mean(bulk_rho)

#         wavenumbers = np.array(k_values_list).flatten()    #x
#         y_positions = np.array(y_positions_list)           #y
#         spectra     = np.array(spectra_list)               #z


#         # Get the indices that would sort `y_positions` in ascending order
#         sorted_indices = np.argsort(y_positions)

#         # Reorder `y_positions` and `spectra` according to these indices
#         sorted_y_positions = y_positions[sorted_indices]
#         sorted_spectra = spectra[sorted_indices]
       
#         all_spectra.append(sorted_spectra)
    
#     average_spectra_list = np.array(all_spectra)
#     average_spectra = np.mean(average_spectra_list, axis=0)
#     #average_spectra_list = average_spectra_list[1:-2,:]
    
#     min_spectra = np.amin(average_spectra)
#     max_spectra = np.amax(average_spectra)
     
#     #print('min wavelength',np.amin(1/wavenumbers), 'max_wavelength',np.amax(1/wavenumbers))
#     print('min spectra',min_spectra,               'max_spectra', max_spectra)
#     print("y^+ min =", np.amin(sorted_y_positions),"y^+ max =",np.amax(sorted_y_positions))
    
#     fig, ax = plt.subplots(figsize=(4, 4.5))
#     if tw:
#         average_spectra[average_spectra < 0.5] = 0.5
#         average_spectra[average_spectra > 9.99] = 9.99
#         levs = np.linspace(0.0, 10, 50)
#         # Add a dashed horizontal line at y⁺ max cp
#         y_plus_line = y_plus_max_cp
#         ax.axhline(y=y_plus_line, color='black', linestyle='--', linewidth=1.5)
#         ax.text(x=400, y=y_plus_line * 1.05, s=r'$c_{P_\textrm{max}}$', color='black', fontsize=fontsize, ha='left', va='bottom')
#         # Add a dashed horizontal line at y⁺ max u
#         y_plus_line = y_plus_max_u
#         ax.axhline(y=y_plus_line, color='black', linestyle='--', linewidth=1.5)
#         ax.text(x=400, y=y_plus_line * 1.05, s=r'$u_\textrm{max}$', color='black', fontsize=fontsize, ha='left', va='bottom')
#     if bw:
#         average_spectra[average_spectra < 0.01] = 0.01
#         average_spectra[average_spectra > 0.99] = 0.99
#         levs = np.linspace(0.0, 1, 50)
#         # Add a dashed horizontal line at y⁺ max u
#         y_plus_line = y_plus_max_u
#         ax.axhline(y=y_plus_line, color='black', linestyle='--', linewidth=1.5)
#         ax.text(x=400, y=y_plus_line * 1.05, s=r'$u_\textrm{max}$', color='black', fontsize=fontsize, ha='left', va='bottom')

#     countour = ax.contourf(1/wavenumbers, sorted_y_positions.T, average_spectra, levels=levs, cmap=cmap)

#     # Add colorbar
#     cbar = fig.colorbar(countour, ax=ax, orientation='horizontal', pad=0.03, location='top')
#     cbar.ax.yaxis.set_tick_params(labelsize=fontsize)

#     if tw:
#         cbar.set_ticks([0.0,2,4,6,8,10]) 
#     if bw:
#         cbar.set_ticks([0.0,0.2,0.4,0.6,0.8,1]) 
    
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(1.99e0, 2.0001e3)  
#     ax.set_ylim(5.99e-1, 1.5001e2)  
#     #ax.xaxis.set_ticks([1, 10, 100, 1000])
#     ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='auto', numticks=10))
#     ax.set_aspect('equal')
#     ax.tick_params(axis='both', which='both', labelsize=fontsize, top=True, right=True, direction='in', pad =  7.5)
#     cbar.set_label(r'$k_x E_{\rho uu} \rho^{-1} u_\tau^{-2}$', labelpad=15)
#     plt.xlabel(r'$1/k^*_x$', fontsize=fontsize)
#     if tw:
#         plt.ylabel(r'$y^*_\textrm{hw}$', fontsize=fontsize)
#     if bw:
#         plt.ylabel(r'$y^*_\textrm{cw}$', fontsize=fontsize)
    
#     plt.tight_layout()
    
#     # Remove trailing slash from directory name if it exists
#     directory_name = directory.rstrip('/')
    
#     if tw:
#         plt.savefig(f'spectrogram_tw_{name_directory}.png')
#         plt.savefig(f'spectrogram_tw_{name_directory}.eps')
#     if bw:
#         plt.savefig(f'spectrogram_bw_{name_directory}.png')
#         plt.savefig(f'spectrogram_bw_{name_directory}.eps')
    
#     #plt.show()
    
# plot_colormap(directories,resolution)
# """
