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
probelines_dir = os.path.join(case_dir, "rhea_exp", "temporal_time_probes")

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
f_rhou_key    = " f_rhou [kg/m2s2]"
f_rhov_key    = " f_rhov [kg/m2s2]"
f_rhow_key    = " f_rhow [kg/m2s2]"
rl_f_rhou_key = " rl_f_rhou [kg/m2s2]"
rl_f_rhov_key = " rl_f_rhov [kg/m2s2]"
rl_f_rhow_key = " rl_f_rhow [kg/m2s2]"
rl_f_rhou_curr_step_key = " rl_f_rhou_curr_step [kg/m2s2]"
rl_f_rhov_curr_step_key = " rl_f_rhov_curr_step [kg/m2s2]"
rl_f_rhow_curr_step_key = " rl_f_rhow_curr_step [kg/m2s2]"
d_DeltaRxj_j_key = " d_DeltaRxj_j [m/s2]"
d_DeltaRyj_j_key = " d_DeltaRyj_j [m/s2]"
d_DeltaRzj_j_key = " d_DeltaRzj_j [m/s2]"
d_DeltaRxx_x_key = " d_DeltaRxx_x [m/s2]"
d_DeltaRxy_x_key = " d_DeltaRxy_x [m/s2]"
d_DeltaRxz_x_key = " d_DeltaRxz_x [m/s2]"
d_DeltaRxy_y_key = " d_DeltaRxy_y [m/s2]"
d_DeltaRyy_y_key = " d_DeltaRyy_y [m/s2]"
d_DeltaRyz_y_key = " d_DeltaRyz_y [m/s2]"
d_DeltaRxz_z_key = " d_DeltaRxz_z [m/s2]"
d_DeltaRyz_z_key = " d_DeltaRyz_z [m/s2]"
d_DeltaRzz_z_key = " d_DeltaRzz_z [m/s2]"
all_vars_keys = [
    time_key, y_key, u_key, v_key, w_key,
    rhou_inv_key, rhov_inv_key, rhow_inv_key, 
    rhou_vis_key, rhov_vis_key, rhow_vis_key, 
    f_rhou_key, f_rhov_key, f_rhow_key, 
    rl_f_rhou_key, rl_f_rhov_key, rl_f_rhow_key,
    rl_f_rhou_curr_step_key, rl_f_rhov_curr_step_key, rl_f_rhow_curr_step_key, 
    d_DeltaRxj_j_key, d_DeltaRyj_j_key, d_DeltaRzj_j_key, 
    d_DeltaRxx_x_key, d_DeltaRxy_x_key, d_DeltaRxz_x_key, 
    d_DeltaRxy_y_key, d_DeltaRyy_y_key, d_DeltaRyz_y_key, 
    d_DeltaRxz_z_key, d_DeltaRyz_z_key, d_DeltaRzz_z_key,
]

# Initialize visualizer
visualizer = ChannelVisualizer(postDir)

#--------------------------------------------------------------------------------------------

# --- RL filenames ---
filename_dict_        = {key: [] for key in probes_name}
global_step_dict_     = {key: [] for key in probes_name}
n_train_episodes_dict = {key: 0 for key in probes_name}
for probe in probes_name:
    pattern        = f"{probelines_dir}/temporal_point_probe_y_plus_{probe}_step*.csv"
    matching_files = sorted(glob.glob(pattern))
    for file in matching_files:
        filename_dict_[probe].append(file)
        base_filename = os.path.basename(file)
        global_step_str = base_filename.split('_')[-1].replace('.csv', '')
        global_step_num = int(global_step_str[4:])
        global_step_dict_[probe].append(global_step_num)
        n_train_episodes_dict[probe] += 1

# --- Update RL dictionary distinguishing by episodes ---
# Check all probes have same number of episodes
n_train_episodes = check_uniform_dict(n_train_episodes_dict)
del n_train_episodes_dict
# Define episodes name
episodes_id   = [ep for ep in np.arange(n_train_episodes)]
episodes_name = [f"episode_{ep}" for ep in episodes_id]
n_episodes    = len(episodes_name)
# Re-structure RL dictionaries
filename_dict         = {episode: {probe: "" for probe in probes_name} for episode in episodes_name}
global_step_dict      = {episode: None for episode in episodes_name}
for e_i in range(n_train_episodes):
    episode = episodes_name[e_i]
    for p_i in range(n_probes):
        probe = probes_name[p_i]
        filename_dict[episode][probe] = filename_dict_[probe][e_i]
        if p_i == 0:
            global_step_dict[episode] = global_step_dict_[probe][e_i]
        else:
            assert global_step_dict[episode] == global_step_dict_[probe][e_i]
# Debugging
print(f"\nNumber of RL train episodes: {n_train_episodes}, with episodes name:\n{episodes_name}") 
print("\nRL probelines:")
for probe in probes_name:
    for episode in episodes_name:
        print(f"[Probe {probe}, Episode {episode}] Filename: {filename_dict[episode][probe]}, global step: {global_step_dict[episode]}")

# --- Check filenames exist ---
for episode in episodes_name:
    for probe in probes_name:
        file = filename_dict[episode][probe]
        if not os.path.isfile(file):
            print(f"Error: File '{file}' not found.")
            sys.exit(1)
print("\nAll files exist :)")

#--------------------------------------------------------------------------------------------

# --- Get probelines data ---
print("\nImporting probelines data...")

data_dict = {episode: {probe: None for probe in probes_name} for episode in episodes_name}
tavg_atEpStart  = 0.0; time_atEpStart = 0.0; vars_keys = []
episode_counter = 0;   iter_counter   = 0
y_plus_key      = 'y_plus [m]'
for episode in episodes_name:
    print(f"{episode_counter/n_episodes*100:.0f}%"); episode_counter += 1
    for probe in probes_name:
        # Check available variables
        file = filename_dict[episode][probe]
        available_vars  = pd.read_csv(file, nrows=1).columns
        if iter_counter == 0:
            vars_keys = [key for key in all_vars_keys if key in available_vars]
        else:
            assert vars_keys == [key for key in all_vars_keys if key in available_vars]
        # Get data from csv file
        data = pd.read_csv(file, usecols=vars_keys)
        data_dict[episode][probe] = { key: data[key].to_numpy()[::10] for key in vars_keys }
        # Post-processed variable TIME
        time_data           = data_dict[episode][probe][time_key]
        time_relative_data  = time_data - time_data[0]     # only interested in relative values, dt
        data_dict[episode][probe][time_key] = time_relative_data
        if iter_counter == 0:
            tavg_atEpStart = float(time_data[0] - tavg0_RL)
            time_atEpStart = float(time_data[0])
        else:
            assert np.isclose(tavg_atEpStart, float(time_data[0] - tavg0_RL))
            assert np.isclose(time_atEpStart, float(time_data[0]))
        # Post-processed variable Y+
        y_data  = data_dict[episode][probe][y_key]
        assert np.allclose(y_data, y_data[0])
        y_value = float(y_data[0])
        isBottomWall              = y_value < delta
        if isBottomWall:
            data_dict[episode][probe][y_plus_key] = y_value * rho0 * u_tau / mu0
        else:
            data_dict[episode][probe][y_plus_key] = (2 * delta - y_value) * rho0 * u_tau / mu0
        iter_counter += 1
vars_keys.append(y_plus_key)

#--------------------------------------------------------------------------------------------

# ----- Ensemble-Average probelines variables with the same y-coord, assuming all probelines are generated at same time instants
# Ensemble-average variables at same y-coord

print("\nAveraging probelines spectral data...")
ensemble_dict = {episode: {y_coord: {var: None for var in vars_keys} for y_coord in y_coord_name_list} for episode in episodes_name}
for episode in episodes_name:
    for y_coord in y_coord_name_list:
        avg_probes = y_coord_name_vs_probes_name_dict[y_coord]
        for probe in avg_probes:
            if probe == avg_probes[0]:
                for var in vars_keys:
                    ensemble_dict[episode][y_coord][var] = data_dict[episode][probe][var]
            else:
                for var in vars_keys:
                    ensemble_dict[episode][y_coord][var] += data_dict[episode][probe][var]
        for var in vars_keys:
            ensemble_dict[episode][y_coord][var] /= len(avg_probes)
del data_dict

#--------------------------------------------------------------------------------------------
# Plot RHS terms of drhou/dt, drhov/dt, drhow/dt N-S equations
# Assuming ct. rho = 1 everywhere in the domain
print("\nBuild frames...")
frames_rhou      = []; frames_rhov     = []; frames_rhow       = []; frames_rhovel      = [frames_rhou, frames_rhov, frames_rhow]
frames_rhou_zoom = []; frames_rhov_zoom = []; frames_rhow_zoom = []; frames_rhovel_zoom = [frames_rhou_zoom, frames_rhov_zoom, frames_rhow_zoom]
frames_d_DeltaRij_j = []
for episode in episodes_name:
    # Absoule value of rho*v_i and d/rho*v_i)/dx_i terms of N-S equation
    frames_rhovel = visualizer.build_rhovel_frame_from_dicts(
        frames_rhovel, 
        ensemble_dict[episode], 
        tavg_atEpStart, global_step_dict[episode], ylim=None,
    )
    # Idem., but zoom to RL term range
    frames_rhovel_zoom = visualizer.build_rhovel_frame_from_dicts(
        frames_rhovel_zoom,
        ensemble_dict[episode],
        tavg_atEpStart, global_step_dict[episode], ylim=[-5, 5],
    )
    # Frames of d_Deltaij_j along time in 3x3 subplots grid 
    frames_d_DeltaRij_j = visualizer.build_d_DeltaRij_j_frame_from_dicts(
        frames_d_DeltaRij_j, 
        ensemble_dict[episode],
        tavg_atEpStart, global_step_dict[episode],
    )


print("\nSave gifs from frames...")
frames_dict = { 
    'rhs_rhou':      frames_rhovel[0],      'rhs_rhov':      frames_rhovel[1],      'rhs_rhow': frames_rhovel[2],
    'rhs_rhou_zoom': frames_rhovel_zoom[0], 'rhs_rhov_zoom': frames_rhovel_zoom[1], 'rhs_rhow_zoom': frames_rhovel_zoom[2],
    'rhs_d_DeltaRij_j': frames_d_DeltaRij_j,
}
visualizer.build_main_gifs_from_frames(frames_dict)
print("Gifs plotted successfully!")