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
    print(f"Script parameters: \n- Ensemble: {ensemble}\n- Train name: {train_name}\n- Case directory: {case_dir}")
except :
    raise ValueError("Missing call arguments, should be: <ensemble> <train_name> <case_dir>")

# --- Simulation parameters ---
restart_data_file_time = 320.999999999  # restart_data_file attribute 'Time'
restart_data_file_averaging_time = 2.0  # restart_data_file attribute 'AveragingTime'
t_avg_0    = restart_data_file_time - restart_data_file_averaging_time 
rl_n_envs  = 8   # num. actuators (control cubes) per cfd simulation 
action_dim = 6

# --- Post-processing parameters ---
verbose = False
action_data_dir = f"{case_dir}/train/{train_name}/action/"
time_data_dir   = f"{case_dir}/train/{train_name}/time/"

# --- Post-processing directory ---
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# --- Visualizer ---

visualizer = ChannelVisualizer(postDir)
nbins      = 1000

# ----------- Build data h5 filenames ------------

# --- RL filenames ---
pattern = os.path.join(action_data_dir, f"action_ensemble{ensemble}_step*.txt")
matching_files = sorted(glob.glob(pattern))
action_filepath_list  = []
time_filepath_list    = []
file_details_list = []  # list elements of the structure: 'stepxxxxxx', e.g. 'step000064'
global_step_list  = []  # list elements of structure 'int(xxxxxxx)', e.g. 64
if matching_files:
    print("Found files:")
    for action_filepath in matching_files:
        # Store file
        action_filepath_list.append(action_filepath)
        # Extract the filename (without the directory)
        action_filename = os.path.basename(action_filepath)
        # Extract the part corresponding to "*"
        # Split by "_" and get the last part before ".txt"
        file_details = action_filename.split('_')[-1].replace('.txt', '')
        # Add the extracted part to the list
        file_details_list.append(file_details)
        # Store global step number
        global_step = int(file_details.split('step')[1])
        global_step_list.append(global_step)
        # Print the file and the extracted part
        # Build time filename
        time_filename = "time" + action_filename.split("action")[1]
        time_filepath = os.path.join(time_data_dir, time_filename)
        time_filepath_list.append(time_filepath)
        # Logging
        print(f"\nAction filepath: {action_filename}")
        print(f"Time filepath: {time_filename}")
        print(f"File details: {file_details}")
else:
    print(f"No files found matching the pattern: {pattern}")

# ----------- Get actions & time data ------------

# --- Check if action & time files exists ---
for file in action_filepath_list:
    if not os.path.isfile(file):
        print(f"Error: File '{file}' not found.")
        sys.exit(1)
for file in time_filepath_list:
    if not os.path.isfile(file):
        print(f"Error: File '{file}' not found.")
        sys.exit(1)

# --- Get action & time data from txt files ---
print("\nImporting action & time data from txt files...")
n_RL = len(action_filepath_list)
for i_RL in range(n_RL):
    
    # Read data
    action_file        = action_filepath_list[i_RL]
    time_file          = time_filepath_list[i_RL]
    action_data_aux    = np.loadtxt(action_file)
    time_data_aux      = np.loadtxt(time_file)
    num_time_steps_aux = action_data_aux.shape[0]

    # Allocate action data array
    if i_RL == 0:
        num_time_steps = num_time_steps_aux
        action_data   = np.zeros((n_RL, num_time_steps, action_dim, rl_n_envs))
    else:
        if num_time_steps != num_time_steps_aux:
            print(f"Warning: global step {global_step_list[i_RL]} has num. global steps {num_time_steps_aux} != {num_time_steps}")
            print(f"Warning: not considered steps >= global step {global_step_list[i_RL]}")
            break

    # Allocate data
    action_data[i_RL,:,:,:] = action_data_aux.reshape(num_time_steps, action_dim, rl_n_envs)
    if i_RL == 0:
        time_data     = time_data_aux               # shape [num_time_steps]
        avg_time_data = time_data_aux - t_avg_0     # shape [num_time_steps]
    else:
        assert np.allclose(time_data, time_data_aux, atol=1e-6)

    # Logging    
    print(f"\nAction data imported from file '{action_file}'")
    print(f"Time data imported from file '{time_file}'")

# Check if the for loop break due to incomplete global step
if i_RL != (n_RL-1):
    action_data = action_data[:i_RL,:,:,:]
    n_RL = i_RL
print("\nData imported successfully!")

#-----------------------------------------------------------------------------------------
#                              Actions figures and gif frames 
#-----------------------------------------------------------------------------------------

actions_min = int(np.min(action_data))-1
actions_max = int(np.max(action_data))+1
ylim = [actions_min, actions_max]
print(f"\nAction limits: {ylim}")

# ---------------------- Plot actions at each RL global step (specific ensemble) ---------------------- 

#print("\nBuilding actions figures...")
#for i_RL in range(n_RL):
#    visualizer.build_actions_fig(avg_time_data, action_data[i_RL], global_step_list[i_RL], ylim)
#print("Actions figures done successfully!")

# ----------------- Plot Animation Frames of each action dimension for increasing RL global step (specific ensemble) -----------------

print("\nBuilding actions gif frames...")

# Allocate frames dictionary: empty list per each action dimension 
frames_dict_scatter = {}
frames_dict_pdf = {}
for i_act in range(action_dim):
    frames_dict_scatter[i_act] = []
    frames_dict_pdf[i_act] = []

# Generate frames
for i_RL in range(n_RL):
    # log progress
    if i_RL % (n_RL//10 or 1) == 0:
        print(f"{i_RL/n_RL*100:.0f}%")
    # Build frames
    frames_dict_scatter, frames_dict_pdf = visualizer.build_actions_frames(frames_dict_scatter, frames_dict_pdf, avg_time_data, action_data[i_RL], global_step_list[i_RL], ylim)

print("Building gifs from frames...")
visualizer.build_action_gifs_from_frames(frames_dict_scatter, frames_dict_pdf, action_dim)
print("Gifs plotted successfully!")




