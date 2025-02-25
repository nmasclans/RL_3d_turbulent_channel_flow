#!/home/jofre/miniconda3/envs/smartrhea-env-v2/bin/python3

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

from utils import *
from ChannelVisualizer import ChannelVisualizer
#np.set_printoptions(threshold=sys.maxsize)
#plt.rc( 'text', usetex = True )
#rc('font', family='sanserif')
#plt.rc( 'font', size = 20 )
#plt.rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}', r'\usepackage{amssymb}', r'\usepackage{color}' ]

#--------------------------------------------------------------------------------------------

# --- Get CASE parameters ---

try :
    ensemble   = sys.argv[1]
    train_name = sys.argv[2]
    case_dir   = sys.argv[3]
    run_mode   = sys.argv[4]
    print(f"Script parameters: \n- Ensemble: {ensemble}\n- Train name: {train_name}\n- Case directory: {case_dir}\n- Run mode: {run_mode}")
except :
    raise ValueError("Missing call arguments, should be: <ensemble> <train_name> <case_dir> <run_mode>")

if run_mode == "train":
    print("Run mode is set to training")
elif run_mode == "eval":
    print("Run mode is set to evaluation")
else: 
    raise ValueError(f"Unrecognized input argument run_mode = `{run_mode}`")

# --- Simulation parameters ---
restart_data_file_time = 320.999999999  # restart_data_file attribute 'Time'
restart_data_file_averaging_time = 2.0  # restart_data_file attribute 'AveragingTime'
t_avg_0    = restart_data_file_time - restart_data_file_averaging_time 
rl_n_envs  = 8   # num. actuators (control cubes) per cfd simulation 
reward_dim = 1

# --- Post-processing parameters ---
verbose = False
local_reward_data_dir = f"{case_dir}/{run_mode}/{train_name}/local_reward/"
reward_data_dir       = f"{case_dir}/{run_mode}/{train_name}/reward/"
time_data_dir         = f"{case_dir}/{run_mode}/{train_name}/time/"

# --- Post-processing directory ---
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# --- Visualizer ---

visualizer = ChannelVisualizer(postDir)
nbins      = 1000

# ----------- Build data h5 filenames ------------

# --- RL filenames ---
pattern = os.path.join(local_reward_data_dir, f"local_reward_ensemble{ensemble}_step*.txt")
matching_files = sorted(glob.glob(pattern))
local_reward_filepath_list = []
reward_filepath_list       = []
time_filepath_list         = []
file_details_list          = []  # list elements of the structure: 'stepxxxxxx', e.g. 'step000064'
global_step_list           = []  # list elements of structure 'int(xxxxxxx)', e.g. 64
if matching_files:
    print("Found files:")
    for local_reward_filepath in matching_files:
        # Store file
        local_reward_filepath_list.append(local_reward_filepath)
        # Extract the filename (without the directory)
        local_reward_filename = os.path.basename(local_reward_filepath)
        # Extract the part corresponding to "*"
        # Split by "_" and get the last part before ".txt"
        file_details = local_reward_filename.split('_')[-1].replace('.txt', '')
        # Add the extracted part to the list
        file_details_list.append(file_details)
        # Store global step number
        global_step = int(file_details.split('step')[1])
        global_step_list.append(global_step)
        # Print the file and the extracted part
        # Build reward filename
        reward_filename = "reward" + local_reward_filename.split("local_reward")[1]
        reward_filepath = os.path.join(reward_data_dir, reward_filename)
        reward_filepath_list.append(reward_filepath)
        # Build time filename
        time_filename = "time" + local_reward_filename.split("local_reward")[1]
        time_filepath = os.path.join(time_data_dir, time_filename)
        time_filepath_list.append(time_filepath)
        # Logging
        print(f"\nLocal reward filepath: {local_reward_filename}")
        print(f"\nReward filepath: {reward_filename}")
        print(f"Time filepath: {time_filename}")
        print(f"File details: {file_details}")
else:
    print(f"No files found matching the pattern: {pattern}")

# ----------- Get local_reward, reward & time data ------------

# --- Check if local_reward, reward & time files exists ---
for file in local_reward_filepath_list:
    if not os.path.isfile(file):
        print(f"Error: File '{file}' not found.")
        sys.exit(1)
for file in reward_filepath_list:
    if not os.path.isfile(file):
        print(f"Error: File '{file}' not found.")
        sys.exit(1)
for file in time_filepath_list:
    if not os.path.isfile(file):
        print(f"Error: File '{file}' not found.")
        sys.exit(1)

# --- Get local_reward, reward & time data from txt files ---
print("\nImporting local_reward, reward & time data from txt files...")
n_RL = len(local_reward_filepath_list)
for i_RL in range(n_RL):
    
    # Read data
    local_reward_file      = local_reward_filepath_list[i_RL]
    reward_file            = reward_filepath_list[i_RL]
    time_file              = time_filepath_list[i_RL]
    local_reward_data_aux = np.loadtxt(local_reward_file)
    reward_data_aux       = np.loadtxt(reward_file)
    time_data_aux         = np.loadtxt(time_file)
    num_time_steps_aux    = time_data_aux.shape[0]

    # Allocate rewards data arrays
    if i_RL == 0:
        num_time_steps = num_time_steps_aux
        local_reward_data = np.zeros((n_RL, num_time_steps, reward_dim, rl_n_envs))
        reward_data       = np.zeros((n_RL, num_time_steps, reward_dim, rl_n_envs))
    else:
        if num_time_steps != num_time_steps_aux:
            print(f"Warning: global step {global_step_list[i_RL]} has num. global steps {num_time_steps_aux} != {num_time_steps}")
            print(f"Warning: not considered steps >= global step {global_step_list[i_RL]}")
            break

    # Allocate data
    local_reward_data[i_RL,:,:,:] = local_reward_data_aux.reshape(num_time_steps, reward_dim, rl_n_envs)
    reward_data[i_RL,:,:,:]       = reward_data_aux.reshape(      num_time_steps, reward_dim, rl_n_envs)
    if i_RL == 0:
        time_data     = time_data_aux               # shape [num_time_steps]
        avg_time_data = time_data_aux - t_avg_0     # shape [num_time_steps]
    else:
        assert np.allclose(time_data, time_data_aux, atol=1e-6)

    # Logging    
    print(f"\nLocal reward data imported from file '{local_reward_file}'")
    print(f"\nReward data imported from file '{reward_file}'")
    print(f"Time data imported from file '{time_file}'")

# Check if the for loop break due to incomplete global step
if i_RL != (n_RL-1):
    local_reward_data = local_reward_data[:i_RL,:,:,:]
    reward_data       = reward_data[:i_RL,:,:,:]
    n_RL = i_RL
print("\nData imported successfully!")

#-----------------------------------------------------------------------------------------
#                    Local_reward and reward figures and gif frames 
#-----------------------------------------------------------------------------------------

local_reward_min  = int(np.min(local_reward_data))-1
local_reward_max  = int(np.max(local_reward_data))+1
reward_min        = int(np.min(reward_data))-1
reward_max        = int(np.max(reward_data))+1
ylim_local_reward = [local_reward_min, local_reward_max]
ylim_reward       = [reward_min, reward_max]
print(f"\nLocal reward limits: {ylim_local_reward}")
print(f"\nReward limits: {ylim_reward}")

# ---------------------- Plot actions at each RL global step (specific ensemble) ---------------------- 

#print("\nBuilding actions figures...")
#for i_RL in range(n_RL):
#    visualizer.build_actions_fig(avg_time_data, action_data[i_RL], global_step_list[i_RL], ylim)
#print("Actions figures done successfully!")

# ----------------- Plot Animation Frames of each action dimension for increasing RL global step (specific ensemble) -----------------

print("\nBuilding local_reward and reward gif frames...")

# Allocate frames dictionary: empty list per each action dimension 
frames_dict_scatter_local_reward = {}
frames_dict_scatter_reward       = {}
frames_dict_pdf_local_reward     = {}
frames_dict_pdf_reward           = {}
for i_rew in range(reward_dim):
    frames_dict_scatter_local_reward[i_rew] = []
    frames_dict_scatter_reward[i_rew]       = []
    frames_dict_pdf_local_reward[i_rew]     = []
    frames_dict_pdf_reward[i_rew]           = []

# Generate frames
for i_RL in range(n_RL):
    # log progress
    if i_RL % (n_RL//10 or 1) == 0:
        print(f"{i_RL/n_RL*100:.0f}%")
    # Build frames
    frames_dict_scatter_local_reward, frames_dict_pdf_local_reward = visualizer.build_rewards_frames(frames_dict_scatter_local_reward, frames_dict_pdf_local_reward, avg_time_data, local_reward_data[i_RL], global_step_list[i_RL], ylim_local_reward, "Local Reward")
    frames_dict_scatter_reward,       frames_dict_pdf_reward       = visualizer.build_rewards_frames(frames_dict_scatter_reward,       frames_dict_pdf_reward,       avg_time_data, reward_data[i_RL],       global_step_list[i_RL], ylim_reward, "Reward")

print("Building gifs from frames...")
visualizer.build_rewards_gifs_from_frames(frames_dict_scatter_local_reward, frames_dict_pdf_local_reward, reward_dim, "reward_local")
visualizer.build_rewards_gifs_from_frames(frames_dict_scatter_reward,       frames_dict_pdf_reward,       reward_dim, "reward")
print("Gifs plotted successfully!")




