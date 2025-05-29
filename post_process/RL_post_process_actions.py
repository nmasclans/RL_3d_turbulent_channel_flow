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
    rl_n_envs  = int(sys.argv[5])
    print(f"\nScript parameters: \n- Ensemble: {ensemble}\n- Train name: {train_name}\n- Case directory: {case_dir}\n- Run mode: {run_mode}\n- Num. RL environments / Parallelization cores: {rl_n_envs}")
except :
    raise ValueError("Missing call arguments, should be: <ensemble> <train_name> <case_dir> <rl_n_envs>")

if run_mode == "train":
    print("Run mode is set to training")
elif run_mode == "eval":
    print("Run mode is set to evaluation")
else: 
    raise ValueError(f"Unrecognized input argument run_mode = `{run_mode}`")

# --- Simulation parameters ---
restart_data_file_time = 323.999999999  # restart_data_file attribute 'Time'
restart_data_file_averaging_time = 5.0  # restart_data_file attribute 'AveragingTime'
t_avg_0 = restart_data_file_time - restart_data_file_averaging_time 

# --- Post-processing parameters ---
verbose = False
action_data_dir = f"{case_dir}/{run_mode}/{train_name}/action/"
time_data_dir   = f"{case_dir}/{run_mode}/{train_name}/time/"

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
action_dict   = dict.fromkeys(np.arange(n_RL))
avg_time_dict = dict.fromkeys(np.arange(n_RL))
action_min    = 1e8;    action_max   = -1e8
avg_time_min  = 1e8;    avg_time_max = -1e8
for i_RL in range(n_RL):
    
    # Read data
    action_file    = action_filepath_list[i_RL]
    time_file      = time_filepath_list[i_RL]
    action_data    = np.loadtxt(action_file)    # shape [num_time_steps, action_dim + rl_n_envs], action_dim set as free parameter
    time_data      = np.loadtxt(time_file)      # shape [num_time_steps]
    avg_time_data  = time_data - t_avg_0
    num_time_steps = avg_time_data.size

    # Allocate data
    action_aux          = action_data.reshape(num_time_steps, rl_n_envs, -1)
    action_dict[i_RL]   = action_aux.swapaxes(1,2)
    avg_time_dict[i_RL] = avg_time_data

    # Update min & max values, if necessary
    action_min   = np.min([action_min, np.min(action_data)])
    action_max   = np.max([action_max, np.max(action_data)])
    avg_time_min = np.min([avg_time_min, np.min(avg_time_data)])
    avg_time_max = np.max([avg_time_max, np.max(avg_time_data)])

    # Action dimension 
    action_dim_aux      = action_dict[i_RL].shape[1]
    if i_RL == 0:
        action_dim = action_dim_aux
    else:
        assert action_dim == action_dim_aux, f"Different action dimension found in '{action_file}', with {action_dim} != {action_dim_aux}"
    
    # Logging    
    print(f"\nAction data imported from file '{action_file}'")
    print(f"Time data imported from file '{time_file}'")

print(f"\nAction dimension: {action_dim}")
print("\nData imported successfully!")

#-----------------------------------------------------------------------------------------
#                              Actions figures and gif frames 
#-----------------------------------------------------------------------------------------

action_min = int(action_min)-1
action_max = int(action_max)+1
action_lim   = [action_min,   action_max]
avg_time_lim = [avg_time_min, avg_time_max]
print(f"\nAction limits: {action_lim}")
print(f"\nAveraging time limits: {avg_time_lim}")

# ----------------- Plot Animation Frames of each action dimension for increasing RL global step (specific ensemble) -----------------

print("\nBuilding actions gif frames...")

# Allocate frames dictionary: empty list per each action dimension 
frames_dict_scatter = {}
frames_dict_pdf     = {}
frames_dict_ensavg  = {}
for i_act in range(action_dim):
    frames_dict_scatter[i_act] = []
    frames_dict_pdf[i_act] = []
    frames_dict_ensavg[i_act] = []

# Generate frames
for i_RL in range(n_RL):
    # log progress
    if i_RL % (n_RL//10 or 1) == 0:
        print(f"{i_RL/n_RL*100:.0f}%")
    # Build frames
    frames_dict_scatter, frames_dict_pdf, frames_dict_ensavg = visualizer.build_actions_frames(frames_dict_scatter, frames_dict_pdf, frames_dict_ensavg, avg_time_dict[i_RL], action_dict[i_RL], avg_time_lim, action_lim, global_step_list[i_RL])

print("\nBuilding gifs from frames...")
visualizer.build_action_gifs_from_frames(frames_dict_scatter, frames_dict_pdf, frames_dict_ensavg, action_dim)
print("Gifs plotted successfully!")




