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
    Re_tau     = float(sys.argv[3])     # Friction Reynolds number [-]
    case_dir   = sys.argv[4]
    run_mode   = sys.argv[5]
    rl_n_envs  = int(sys.argv[6])
    print(f"\nScript parameters: \n- Ensemble: {ensemble}\n- Train name: {train_name}\n- Re_tau: {Re_tau} \n- Case directory: {case_dir}\n- Run mode: {run_mode}\n- Num. RL environments / Parallelization cores: {rl_n_envs}")
except :
    raise ValueError("Missing call arguments, should be: <ensemble> <train_name> <case_dir> <rl_n_envs>")

if run_mode == "train":
    print("Run mode is set to training")
elif run_mode == "eval":
    print("Run mode is set to evaluation")
else: 
    raise ValueError(f"Unrecognized input argument run_mode = `{run_mode}`")

# --- Simulation parameters ---
if Re_tau == 100:
    restart_data_file_time = 323.999999999  # restart_data_file attribute 'Time'
    restart_data_file_averaging_time = 5.0  # restart_data_file attribute 'AveragingTime'
else:
    restart_data_file_time = 200.0623152
    restart_data_file_averaging_time = 0.0
t_avg_0 = restart_data_file_time - restart_data_file_averaging_time 

# --- Post-processing parameters ---
verbose = False
state_data_dir = f"{case_dir}/{run_mode}/{train_name}/state/"
time_data_dir  = f"{case_dir}/{run_mode}/{train_name}/time/"

# --- Post-processing directory ---
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# --- Visualizer ---

visualizer = ChannelVisualizer(postDir)
nbins      = 1000

# ----------- Build data h5 filenames ------------

# --- RL filenames ---
pattern = os.path.join(state_data_dir, f"state_ensemble{ensemble}_step*.txt")
matching_files = sorted(glob.glob(pattern))
state_filepath_list = []
time_filepath_list  = []
file_details_list   = []  # list elements of the structure: 'stepxxxxxx', e.g. 'step000064'
global_step_list    = []  # list elements of structure 'int(xxxxxxx)', e.g. 64
if matching_files:
    print("Found files:")
    for state_filepath in matching_files:
        # Store file
        state_filepath_list.append(state_filepath)
        # Extract the filename (without the directory)
        state_filename = os.path.basename(state_filepath)
        # Extract the part corresponding to "*"
        # Split by "_" and get the last part before ".txt"
        file_details = state_filename.split('_')[-1].replace('.txt', '')
        # Add the extracted part to the list
        file_details_list.append(file_details)
        # Store global step number
        global_step = int(file_details.split('step')[1])
        global_step_list.append(global_step)
        # Print the file and the extracted part
        # Build time filename
        time_filename = "time" + state_filename.split("state")[1]
        time_filepath = os.path.join(time_data_dir, time_filename)
        time_filepath_list.append(time_filepath)
        # Logging
        print(f"\nState filepath: {state_filename}")
        print(f"Time filepath: {time_filename}")
        print(f"File details: {file_details}")
else:
    print(f"No files found matching the pattern: {pattern}")

# ----------- Get actions & time data ------------

# --- Check if state & time files exists ---
for file in state_filepath_list:
    if not os.path.isfile(file):
        print(f"Error: File '{file}' not found.")
        sys.exit(1)
for file in time_filepath_list:
    if not os.path.isfile(file):
        print(f"Error: File '{file}' not found.")
        sys.exit(1)

# --- Get state & time data from txt files ---
print("\nImporting state & time data from txt files...")
n_RL = len(state_filepath_list)
state_dict    = dict.fromkeys(np.arange(n_RL))
avg_time_dict = dict.fromkeys(np.arange(n_RL))
state_min    = 1e8;    state_max    = -1e8
avg_time_min = 1e8;    avg_time_max = -1e8
for i_RL in range(n_RL):
    
    # Read data
    state_file     = state_filepath_list[i_RL]
    time_file      = time_filepath_list[i_RL]
    state_data     = np.loadtxt(state_file)     # shape [num_time_steps, state_dim + rl_n_envs], state_dim set as free parameter
    time_data      = np.loadtxt(time_file)      # shape [num_time_steps]
    avg_time_data  = time_data - t_avg_0
    num_time_steps = avg_time_data.size
    # Allocate data
    state_aux = state_data.reshape(num_time_steps, rl_n_envs, -1)
    state_dict[i_RL]    = state_aux.swapaxes(1,2)   # shape [num_time_steps, state_dim, rl_n_envs]
    avg_time_dict[i_RL] = avg_time_data             # shape [num_time_steps, rl_n_envs]

    # Update min & max values, if necessary
    state_min    = np.min([state_min,    np.min(state_data)])
    state_max    = np.max([state_max,    np.max(state_data)])
    avg_time_min = np.min([avg_time_min, np.min(avg_time_data)])
    avg_time_max = np.max([avg_time_max, np.max(avg_time_data)])

    # State dimension 
    state_dim_aux = state_dict[i_RL].shape[1]
    if i_RL == 0:
        state_dim = state_dim_aux
    else:
        assert state_dim == state_dim_aux, f"Different state dimension found in '{state_file}', with {state_dim} != {state_dim_aux}"

    # Logging    
    print(f"\nState data imported from file '{state_file}'")
    print(f"Time data imported from file '{time_file}'")

print(f"\nState dimension: {state_dim}")
print("\nData imported successfully!")

#-----------------------------------------------------------------------------------------
#                              States figures and gif frames 
#-----------------------------------------------------------------------------------------

state_min    = int(state_min)-1
state_max    = int(state_max)+1
state_lim    = [state_min, state_max]
avg_time_lim = [avg_time_min, avg_time_max]
print(f"\nState limits: {state_lim}")
print(f"\nAveraging time limits: {avg_time_lim}")

# ----------------- Plot Animation Frames of each state dimension for increasing RL global step (specific ensemble) -----------------

print("\nBuilding states gif frames...")

# Allocate frames dictionary: empty list per each state dimension 
frames_dict_plot   = {}
frames_dict_pdf    = {}
frames_dict_ensavg = {}
for i_state in range(state_dim):
    frames_dict_plot[i_state]   = []
    frames_dict_pdf[i_state]    = []
    frames_dict_ensavg[i_state] = []

# Generate frames
for i_RL in range(n_RL):
    # log progress
    if i_RL % (n_RL//10 or 1) == 0:
        print(f"{i_RL/n_RL*100:.0f}%")
    # Build frames
    frames_dict_plot, frames_dict_pdf, frames_dict_ensavg = visualizer.build_states_frames(frames_dict_plot, frames_dict_pdf, frames_dict_ensavg, avg_time_dict[i_RL], state_dict[i_RL], avg_time_lim, state_lim, global_step_list[i_RL])

print("\nBuilding gifs from frames...")
visualizer.build_state_gifs_from_frames(frames_dict_plot, frames_dict_pdf, frames_dict_ensavg, state_dim)
print("Gifs plotted successfully!")




