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

# Simulation parameters
tavg0_RL = 318.99999999

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

# Visualization parameters
y_plus_max_urms  = 12
y_plus_max_u     = 20
y_plus_max_cp    = 140
y_limit          = y_plus_max_u
fontsize         = 18

# Probelines csv parameters
time_key      = "# t [s]"
y_key         = " y [m]"
u_key         = " u [m/s]"
v_key         = " v [m/s]"
w_key         = " w [m/s]"
rhou_inv_key  = " rhou_inv_flux [kg/m2s2]"
rhov_inv_key  = " rhov_inv_flux [kg/m2s2]"
rhow_inv_key  = " rhow_inv_flux [kg/m2s2]"
rhou_vis_key  = " rhou_vis_flux [kg/m2s2]"
rhov_vis_key  = " rhov_vis_flux [kg/m2s2]"
rhow_vis_key  = " rhow_vis_flux [kg/m2s2]"
f_rhou_key    = " f_rhou_field [kg/m2s2]"
f_rhov_key    = " f_rhov_field [kg/m2s2]"
f_rhow_key    = " f_rhow_field [kg/m2s2]"
rl_f_rhou_key = " rl_f_rhou [kg/m2s2]"
rl_f_rhov_key = " rl_f_rhov [kg/m2s2]"
rl_f_rhow_key = " rl_f_rhow [kg/m2s2]"
vars_keys     = [time_key, y_key, u_key, v_key, w_key,
                 rhou_inv_key, rhov_inv_key, rhow_inv_key, 
                 rhou_vis_key, rhov_vis_key, rhow_vis_key, 
                 f_rhou_key, f_rhov_key, f_rhow_key, 
                 rl_f_rhou_key, rl_f_rhov_key, rl_f_rhow_key]

# Initialize visualizer
visualizer = ChannelVisualizer(postDir)

#--------------------------------------------------------------------------------------------

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
filename_dict_RL         = {episode: {probe: "" for probe in probes_name} for episode in episodes_name_RL}
global_step_dict_RL      = {episode: {probe: 0  for probe in probes_name} for episode in episodes_name_RL}
for p_i in range(n_probes):
    for e_i in range(n_train_episodes):
        filename_dict_RL[episodes_name_RL[e_i]][probes_name[p_i]]    = filename_dict_RL_[probes_name[p_i]][e_i]
        global_step_dict_RL[episodes_name_RL[e_i]][probes_name[p_i]] = global_step_dict_RL_[probes_name[p_i]][e_i]
# Debugging
print(f"\nNumber of RL train episodes: {n_train_episodes}, with episodes name:\n{episodes_name_RL}") 
print("\nRL probelines:")
for probe in probes_name:
    for episode in episodes_name_RL:
        print(f"[Probe {probe}, Episode {episode}] Filename: {filename_dict_RL[episode][probe]}, global step: {global_step_dict_RL[episode][probe]}")

# --- Check filenames exist ---
for episode in episodes_name_RL:
    for probe in probes_name:
        file_RL = filename_dict_RL[episode][probe]
        if not os.path.isfile(file_RL):
            print(f"Error: File '{file_RL}' not found.")
            sys.exit(1)
print("\nAll files exist :)")

#--------------------------------------------------------------------------------------------

# --- Get probelines data ---
print("\nImporting probelines data...")

tavg_atEpStart_dict_RL = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
time_atEpStart_dict_RL = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
time_dict_RL           = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
y_dict_RL              = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
y_plus_dict_RL         = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
u_dict_RL              = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
v_dict_RL              = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
w_dict_RL              = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rhou_inv_dict_RL       = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rhov_inv_dict_RL       = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rhow_inv_dict_RL       = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rhou_vis_dict_RL       = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rhov_vis_dict_RL       = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rhow_vis_dict_RL       = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
f_rhou_dict_RL         = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
f_rhov_dict_RL         = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
f_rhow_dict_RL         = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rl_f_rhou_dict_RL      = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rl_f_rhov_dict_RL      = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
rl_f_rhow_dict_RL      = {episode: {probe: None for probe in probes_name} for episode in episodes_name_RL}
episode_counter = 0
for episode in episodes_name_RL:
    print(f"{episode_counter/n_episodes_RL*100:.0f}%"); episode_counter += 1
    for probe in probes_name:
        # Get data from csv file
        file = filename_dict_RL[episode][probe]
        data = pd.read_csv(file, usecols=vars_keys)
        time_data      = data[time_key].to_numpy()[::10]
        y_data         = data[y_key].to_numpy()[::10]
        u_data         = data[u_key].to_numpy()[::10]
        v_data         = data[v_key].to_numpy()[::10]
        w_data         = data[w_key].to_numpy()[::10]
        rhou_inv_data  = data[rhou_inv_key].to_numpy()[::10]
        rhov_inv_data  = data[rhov_inv_key].to_numpy()[::10]
        rhow_inv_data  = data[rhow_inv_key].to_numpy()[::10]
        rhou_vis_data  = data[rhou_vis_key].to_numpy()[::10]
        rhov_vis_data  = data[rhov_vis_key].to_numpy()[::10]
        rhow_vis_data  = data[rhow_vis_key].to_numpy()[::10]
        f_rhou_data    = data[f_rhou_key].to_numpy()[::10]
        f_rhov_data    = data[f_rhov_key].to_numpy()[::10]
        f_rhow_data    = data[f_rhow_key].to_numpy()[::10]
        rl_f_rhou_data = data[rl_f_rhou_key].to_numpy()[::10]
        rl_f_rhov_data = data[rl_f_rhov_key].to_numpy()[::10]
        rl_f_rhow_data = data[rl_f_rhow_key].to_numpy()[::10]
        # store data in allocation dicts for single train episode period
        time_relative_data  = time_data - time_data[0]     # only interested in relative values, dt
        ### is_train_episode    = time_relative_data <= t_episode_train
        tavg_atEpStart_dict_RL[episode][probe] = float(time_data[0] - tavg0_RL)
        time_atEpStart_dict_RL[episode][probe] = float(time_data[0])
        time_dict_RL[episode][probe]           = time_relative_data
        assert np.allclose(y_data, y_data[0])
        y_value                   = float(y_data[0])
        y_dict_RL[episode][probe] = y_value
        isBottomWall              = y_value < delta
        if isBottomWall:
            y_plus_dict_RL[episode][probe] = y_value * rho0 * u_tau / mu0
        else:
            y_plus_dict_RL[episode][probe] = (2 * delta - y_value) * rho0 * u_tau / mu0
        u_dict_RL[episode][probe]          = u_data
        v_dict_RL[episode][probe]          = v_data
        w_dict_RL[episode][probe]          = w_data
        rhou_inv_dict_RL[episode][probe]   = rhou_inv_data
        rhov_inv_dict_RL[episode][probe]   = rhov_inv_data
        rhow_inv_dict_RL[episode][probe]   = rhow_inv_data
        rhou_vis_dict_RL[episode][probe]   = rhou_vis_data
        rhov_vis_dict_RL[episode][probe]   = rhov_vis_data
        rhow_vis_dict_RL[episode][probe]   = rhow_vis_data
        f_rhou_dict_RL[episode][probe]     = f_rhou_data
        f_rhov_dict_RL[episode][probe]     = f_rhov_data
        f_rhow_dict_RL[episode][probe]     = f_rhow_data
        rl_f_rhou_dict_RL[episode][probe]  = rl_f_rhou_data
        rl_f_rhov_dict_RL[episode][probe]  = rl_f_rhov_data
        rl_f_rhow_dict_RL[episode][probe]  = rl_f_rhow_data
tavg_atEpStart_RL = check_uniform_nested_dict(tavg_atEpStart_dict_RL)
time_atEpStart_RL = check_uniform_nested_dict(time_atEpStart_dict_RL)
del tavg_atEpStart_dict_RL, time_atEpStart_dict_RL

#--------------------------------------------------------------------------------------------

# ----- Ensemble-Average probelines variables with the same y-coord, assuming all probelines are generated at same time instants
print("\nAveraging probelines spectral data...")

# Get time and y+ for first averaging probe, which will be used ensure all probelines from the ensemble (same y-coord) have the same time and y+ values
ensemble_y_plus_dict_RL      = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_time_dict_RL        = {episode: None for episode in episodes_name_RL}
ensemble_global_step_dict_RL = {episode: None for episode in episodes_name_RL}
for episode in episodes_name_RL:
    ensemble_time_dict_RL[episode]        = time_dict_RL[episode][probes_name[0]]
    ensemble_global_step_dict_RL[episode] = global_step_dict_RL[episode][probes_name[0]]
    for y_coord in y_coord_name_list:
        avg_probes = y_coord_name_vs_probes_name_dict[y_coord]
        probe = avg_probes[0]
        ensemble_y_plus_dict_RL[episode][y_coord] = y_plus_dict_RL[episode][probe]

# Ensemble-average variables at same y-coord
ensemble_u_dict_RL         = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_v_dict_RL         = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_w_dict_RL         = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rhou_inv_dict_RL  = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rhov_inv_dict_RL  = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rhow_inv_dict_RL  = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rhou_vis_dict_RL  = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rhov_vis_dict_RL  = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rhow_vis_dict_RL  = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_f_rhou_dict_RL    = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_f_rhov_dict_RL    = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_f_rhow_dict_RL    = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rl_f_rhou_dict_RL = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rl_f_rhov_dict_RL = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
ensemble_rl_f_rhow_dict_RL = {episode: {y_coord: None for y_coord in y_coord_name_list} for episode in episodes_name_RL}
for episode in episodes_name_RL:
    assert np.allclose(ensemble_time_dict_RL[episode],        time_dict_RL[episode][probe])
    assert np.allclose(ensemble_global_step_dict_RL[episode], global_step_dict_RL[episode][probe])
    for y_coord in y_coord_name_list:
        avg_probes = y_coord_name_vs_probes_name_dict[y_coord]
        for probe in avg_probes:
            # Check all ensemble probelines use the same temporal vector
            assert np.allclose(ensemble_y_plus_dict_RL[episode][y_coord], y_plus_dict_RL[episode][probe])
            if probe == avg_probes[0]:
                ensemble_u_dict_RL[episode][y_coord]         = u_dict_RL[episode][probe]
                ensemble_v_dict_RL[episode][y_coord]         = v_dict_RL[episode][probe]
                ensemble_w_dict_RL[episode][y_coord]         = w_dict_RL[episode][probe]
                ensemble_rhou_inv_dict_RL[episode][y_coord]  = rhou_inv_dict_RL[episode][probe]
                ensemble_rhov_inv_dict_RL[episode][y_coord]  = rhov_inv_dict_RL[episode][probe]
                ensemble_rhow_inv_dict_RL[episode][y_coord]  = rhow_inv_dict_RL[episode][probe]
                ensemble_rhou_vis_dict_RL[episode][y_coord]  = rhou_vis_dict_RL[episode][probe]
                ensemble_rhov_vis_dict_RL[episode][y_coord]  = rhov_vis_dict_RL[episode][probe]
                ensemble_rhow_vis_dict_RL[episode][y_coord]  = rhow_vis_dict_RL[episode][probe]
                ensemble_f_rhou_dict_RL[episode][y_coord]    = f_rhou_dict_RL[episode][probe]
                ensemble_f_rhov_dict_RL[episode][y_coord]    = f_rhov_dict_RL[episode][probe]
                ensemble_f_rhow_dict_RL[episode][y_coord]    = f_rhow_dict_RL[episode][probe]
                ensemble_rl_f_rhou_dict_RL[episode][y_coord] = rl_f_rhou_dict_RL[episode][probe]
                ensemble_rl_f_rhov_dict_RL[episode][y_coord] = rl_f_rhov_dict_RL[episode][probe]
                ensemble_rl_f_rhow_dict_RL[episode][y_coord] = rl_f_rhow_dict_RL[episode][probe]
            else:
                ensemble_u_dict_RL[episode][y_coord]         += u_dict_RL[episode][probe]
                ensemble_v_dict_RL[episode][y_coord]         += v_dict_RL[episode][probe]
                ensemble_w_dict_RL[episode][y_coord]         += w_dict_RL[episode][probe]
                ensemble_rhou_inv_dict_RL[episode][y_coord]  += rhou_inv_dict_RL[episode][probe]
                ensemble_rhov_inv_dict_RL[episode][y_coord]  += rhov_inv_dict_RL[episode][probe]
                ensemble_rhow_inv_dict_RL[episode][y_coord]  += rhow_inv_dict_RL[episode][probe]
                ensemble_rhou_vis_dict_RL[episode][y_coord]  += rhou_vis_dict_RL[episode][probe]
                ensemble_rhov_vis_dict_RL[episode][y_coord]  += rhov_vis_dict_RL[episode][probe]
                ensemble_rhow_vis_dict_RL[episode][y_coord]  += rhow_vis_dict_RL[episode][probe]
                ensemble_f_rhou_dict_RL[episode][y_coord]    += f_rhou_dict_RL[episode][probe]
                ensemble_f_rhov_dict_RL[episode][y_coord]    += f_rhov_dict_RL[episode][probe]
                ensemble_f_rhow_dict_RL[episode][y_coord]    += f_rhow_dict_RL[episode][probe]
                ensemble_rl_f_rhou_dict_RL[episode][y_coord] += rl_f_rhou_dict_RL[episode][probe]
                ensemble_rl_f_rhov_dict_RL[episode][y_coord] += rl_f_rhov_dict_RL[episode][probe]
                ensemble_rl_f_rhow_dict_RL[episode][y_coord] += rl_f_rhow_dict_RL[episode][probe]
        ensemble_u_dict_RL[episode][y_coord]         /= len(avg_probes)
        ensemble_v_dict_RL[episode][y_coord]         /= len(avg_probes)
        ensemble_w_dict_RL[episode][y_coord]         /= len(avg_probes)
        ensemble_rhou_inv_dict_RL[episode][y_coord]  /= len(avg_probes)
        ensemble_rhov_inv_dict_RL[episode][y_coord]  /= len(avg_probes)
        ensemble_rhow_inv_dict_RL[episode][y_coord]  /= len(avg_probes)
        ensemble_rhou_vis_dict_RL[episode][y_coord]  /= len(avg_probes)
        ensemble_rhov_vis_dict_RL[episode][y_coord]  /= len(avg_probes)
        ensemble_rhow_vis_dict_RL[episode][y_coord]  /= len(avg_probes)
        ensemble_f_rhou_dict_RL[episode][y_coord]    /= len(avg_probes)
        ensemble_f_rhov_dict_RL[episode][y_coord]    /= len(avg_probes)
        ensemble_f_rhow_dict_RL[episode][y_coord]    /= len(avg_probes)
        ensemble_rl_f_rhou_dict_RL[episode][y_coord] /= len(avg_probes)
        ensemble_rl_f_rhov_dict_RL[episode][y_coord] /= len(avg_probes)
        ensemble_rl_f_rhow_dict_RL[episode][y_coord] /= len(avg_probes)

#--------------------------------------------------------------------------------------------
# Plot RHS terms of drhou/dt, drhov/dt, drhow/dt N-S equations
# Assuming ct. rho = 1 everywhere in the domain
print("\nBuild frames...")
frames_rhou      = []; frames_rhov     = []; frames_rhow       = []
frames_rhou_zoom = []; frames_rhov_zoom = []; frames_rhow_zoom = []

for episode in episodes_name_RL:
    # Absoule value of rho*v_i and d/rho*v_i)/dx_i terms of N-S equation
    frames_rhou = visualizer.build_rhovel_frame_from_dicts(
        frames_rhou, ensemble_y_plus_dict_RL[episode], ensemble_time_dict_RL[episode],
        ensemble_u_dict_RL[episode], ensemble_rhou_inv_dict_RL[episode], ensemble_rhou_vis_dict_RL[episode], ensemble_f_rhou_dict_RL[episode], ensemble_rl_f_rhou_dict_RL[episode],
        tavg_atEpStart_RL, ensemble_global_step_dict_RL[episode], ylim=None, vel_name='u',
    )
    frames_rhov = visualizer.build_rhovel_frame_from_dicts(
        frames_rhov, ensemble_y_plus_dict_RL[episode], ensemble_time_dict_RL[episode],
        ensemble_v_dict_RL[episode], ensemble_rhov_inv_dict_RL[episode], ensemble_rhov_vis_dict_RL[episode], ensemble_f_rhov_dict_RL[episode], ensemble_rl_f_rhov_dict_RL[episode],
        tavg_atEpStart_RL, ensemble_global_step_dict_RL[episode], ylim=None, vel_name='v',
    )
    frames_rhow = visualizer.build_rhovel_frame_from_dicts(
        frames_rhow, ensemble_y_plus_dict_RL[episode], ensemble_time_dict_RL[episode],
        ensemble_w_dict_RL[episode], ensemble_rhow_inv_dict_RL[episode], ensemble_rhow_vis_dict_RL[episode], ensemble_f_rhow_dict_RL[episode], ensemble_rl_f_rhow_dict_RL[episode],
        tavg_atEpStart_RL, ensemble_global_step_dict_RL[episode], ylim=None, vel_name='w',
    )
    # Idem., but zoom to RL term range
    frames_rhou_zoom = visualizer.build_rhovel_frame_from_dicts(
        frames_rhou_zoom, ensemble_y_plus_dict_RL[episode], ensemble_time_dict_RL[episode],
        ensemble_u_dict_RL[episode], ensemble_rhou_inv_dict_RL[episode], ensemble_rhou_vis_dict_RL[episode], ensemble_f_rhou_dict_RL[episode], ensemble_rl_f_rhou_dict_RL[episode],
        tavg_atEpStart_RL, ensemble_global_step_dict_RL[episode], ylim=[-5, 5], vel_name='u',
    )
    frames_rhov_zoom = visualizer.build_rhovel_frame_from_dicts(
        frames_rhov_zoom, ensemble_y_plus_dict_RL[episode], ensemble_time_dict_RL[episode],
        ensemble_v_dict_RL[episode], ensemble_rhov_inv_dict_RL[episode], ensemble_rhov_vis_dict_RL[episode], ensemble_f_rhov_dict_RL[episode], ensemble_rl_f_rhov_dict_RL[episode],
        tavg_atEpStart_RL, ensemble_global_step_dict_RL[episode], ylim=[-5, 5], vel_name='v',
    )
    frames_rhow_zoom = visualizer.build_rhovel_frame_from_dicts(
        frames_rhow_zoom, ensemble_y_plus_dict_RL[episode], ensemble_time_dict_RL[episode],
        ensemble_w_dict_RL[episode], ensemble_rhow_inv_dict_RL[episode], ensemble_rhow_vis_dict_RL[episode], ensemble_f_rhow_dict_RL[episode], ensemble_rl_f_rhow_dict_RL[episode],
        tavg_atEpStart_RL, ensemble_global_step_dict_RL[episode], ylim=[-5, 5], vel_name='w',
    )

print("\nSave gifs from frames...")
frames_dict      = { 'rhs_rhou':      frames_rhou,      'rhs_rhov':      frames_rhov,      'rhs_rhow': frames_rhow }
frames_zoom_dict = { 'rhs_rhou_zoom': frames_rhou_zoom, 'rhs_rhov_zoom': frames_rhov_zoom, 'rhs_rhow_zoom': frames_rhow_zoom }
visualizer.build_main_gifs_from_frames(frames_dict)
visualizer.build_main_gifs_from_frames(frames_zoom_dict)
print("Gifs plotted successfully!")