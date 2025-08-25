#!/home/jofre/miniconda3/envs/smartrhea-env-v2/bin/python3

import sys
import os
import glob
import numpy as np
import h5py    
import re
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
    raise ValueError("Missing call arguments, should be: <ensemble> <train_name> <case_dir> <run_mode> <rl_n_envs>")

if run_mode == "train":
    print("Run mode is set to training")
elif run_mode == "eval":
    print("Run mode is set to evaluation")
else: 
    raise ValueError(f"Unrecognized input argument run_mode = `{run_mode}`")

# --- Simulation parameters ---

if run_mode == 'train':
    restart_data_file_time = 323.99999999   # restart_data_file attribute 'Time'
    restart_data_file_averaging_time = 5.0  # restart_data_file attribute 'AveragingTime'
    t_avg_0        = restart_data_file_time - restart_data_file_averaging_time 
    avg_u_bulk_ref = 14.612998708455182
    avg_v_bulk_ref = 0.0
    avg_w_bulk_ref = 0.0
else:
    restart_data_file_time = 200.0623152
    restart_data_file_averaging_time = 0.0
    t_avg_0        = restart_data_file_time - restart_data_file_averaging_time 
    avg_u_bulk_ref = 15.942190755168479
    avg_v_bulk_ref = 0.0
    avg_w_bulk_ref = 0.0

# --- Post-processing parameters ---

verbose        = False
mpi_output_dir = f"{case_dir}/{run_mode}/{train_name}/mpi_output/"

# --- Post-processing directory ---
postDir = train_name
if not os.path.exists(postDir):
    os.mkdir(postDir)

# --- Visualizer ---

visualizer = ChannelVisualizer(postDir)

# ----------- Build data h5 filenames ------------

# --- RL filenames ---
pattern = os.path.join(mpi_output_dir, f"mpi_output_ensemble{ensemble}_step*.out")
matching_files = sorted(glob.glob(pattern))
filepath_list     = []
file_details_list = []  # list elements of the structure: 'stepxxxxxx', e.g. 'step000064'
global_step_list  = []  # list elements of structure 'int(xxxxxxx)', e.g. 64
if matching_files:
    print("Found files:")
    for filepath in matching_files:
        # Store file
        filepath_list.append(filepath)
        # Extract the filename (without the directory)
        filename = os.path.basename(filepath)
        # Extract the part corresponding to "*"
        # Split by "_" and get the last part before ".out"
        file_details = filename.split('_')[-1].replace('.out', '')
        # Add the extracted part to the list
        file_details_list.append(file_details)
        # Store global step number
        global_step = int(file_details.split('step')[1])
        global_step_list.append(global_step)
        # Logging
        print(f"\nMpi output filename: {filename}")
        print(f"Mpi output filepath: {filepath}")
        print(f"File details: {file_details}")
        print(f"Global step: {global_step}")
else:
    print(f"No files found matching the pattern: {pattern}")

# ----------- Get numerical avg_u_bulk from mpi output ------------

# --- Check if mpi output files exists ---
for file in filepath_list:
    if not os.path.isfile(file):
        print(f"Error: File '{file}' not found.")
        sys.exit(1)

# --- Get numerical avg_u_bulk and avg_time data from .out files ---
print("\nImporting numerical avg_u_bulk data from .out files...")
n_RL                = len(filepath_list)
avg_u_bulk_num_dict = dict.fromkeys(np.arange(n_RL))
avg_v_bulk_num_dict = dict.fromkeys(np.arange(n_RL))
avg_w_bulk_num_dict = dict.fromkeys(np.arange(n_RL))
avg_time_dict       = dict.fromkeys(np.arange(n_RL))
data_pattern        = re.compile(r"Numerical avg_u_bulk:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?),\s*avg_v_bulk:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?),\s*avg_w_bulk:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?),\s*time:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
avg_u_bulk_num_min, avg_u_bulk_num_max  = (1e8,-1e8)
avg_v_bulk_num_min, avg_v_bulk_num_max  = (1e8,-1e8)
avg_w_bulk_num_min, avg_w_bulk_num_max  = (1e8,-1e8)
avg_time_min, avg_time_max              = (1e8,-1e8)
for i_RL in range(n_RL):
    
    # Read file
    filepath = filepath_list[i_RL]
    with open(filepath) as f:
        lines = f.readlines()
    
    # Get values of interest: numerical avg_u_bulk & time
    avg_u_bulk_num_aux = []
    avg_v_bulk_num_aux = []
    avg_w_bulk_num_aux = []
    avg_time_aux       = []
    for line in lines:
        match = data_pattern.search(line)
        if match:
            avg_u_bulk_num_aux.append( float(match.group(1)) )
            avg_v_bulk_num_aux.append( float(match.group(2)) )
            avg_w_bulk_num_aux.append( float(match.group(3)) )
            avg_time_aux.append( float(match.group(4)) - t_avg_0 )

    avg_u_bulk_num_min = np.min([avg_u_bulk_num_min, np.min(avg_u_bulk_num_aux), avg_u_bulk_ref])
    avg_u_bulk_num_max = np.max([avg_u_bulk_num_max, np.max(avg_u_bulk_num_aux), avg_u_bulk_ref])
    avg_v_bulk_num_min = np.min([avg_v_bulk_num_min, np.min(avg_v_bulk_num_aux), avg_v_bulk_ref])
    avg_v_bulk_num_max = np.max([avg_v_bulk_num_max, np.max(avg_v_bulk_num_aux), avg_v_bulk_ref])
    avg_w_bulk_num_min = np.min([avg_w_bulk_num_min, np.min(avg_w_bulk_num_aux), avg_w_bulk_ref])
    avg_w_bulk_num_max = np.max([avg_w_bulk_num_max, np.max(avg_w_bulk_num_aux), avg_w_bulk_ref])
    avg_time_min       = np.min([avg_time_min,       np.min(avg_time_aux)])
    avg_time_max       = np.max([avg_time_max,       np.max(avg_time_aux)])

    avg_u_bulk_num_dict[i_RL] = avg_u_bulk_num_aux
    avg_v_bulk_num_dict[i_RL] = avg_v_bulk_num_aux
    avg_w_bulk_num_dict[i_RL] = avg_w_bulk_num_aux
    avg_time_dict[i_RL]       = avg_time_aux

    print(f"Numerical avg_u_bulk and Time data imported from file '{filepath}'")

print("\nData imported successfully!")

#-----------------------------------------------------------------------------------------
#                    Local_reward and reward figures and gif frames 
#-----------------------------------------------------------------------------------------

avg_u_bulk_num_lim = (avg_u_bulk_num_min, avg_u_bulk_num_max)
avg_v_bulk_num_lim = (avg_v_bulk_num_min, avg_v_bulk_num_max)
avg_w_bulk_num_lim = (avg_w_bulk_num_min, avg_w_bulk_num_max)
avg_time_lim       = (avg_time_min, avg_time_max)
print(f"\nNumerical avg_u_bulk limits: {avg_u_bulk_num_lim}")
print(f"\nAveraging Time limits: {avg_time_lim}")

# ----------------- Plot Animation Frames of each action dimension for increasing RL global step (specific ensemble) -----------------

print("\nBuilding numerical avg_u_bulk vs avg_time image frames...")
frames_plot_u = []
frames_plot_v = []
frames_plot_w = []
for i_RL in range(n_RL):
    # log progress
    if i_RL % (n_RL//10 or 1) == 0:
        print(f"{i_RL/n_RL*100:.0f}%")
    # Build image frames
    frames_plot_u = visualizer.build_avg_vel_bulk_frames(frames_plot_u, avg_time_dict[i_RL], avg_u_bulk_num_dict[i_RL], avg_u_bulk_ref, global_step_list[i_RL], avg_time_lim, avg_u_bulk_num_lim, vel_comp='u')
    frames_plot_v = visualizer.build_avg_vel_bulk_frames(frames_plot_v, avg_time_dict[i_RL], avg_v_bulk_num_dict[i_RL], avg_v_bulk_ref, global_step_list[i_RL], avg_time_lim, avg_v_bulk_num_lim, vel_comp='v')
    frames_plot_w = visualizer.build_avg_vel_bulk_frames(frames_plot_w, avg_time_dict[i_RL], avg_w_bulk_num_dict[i_RL], avg_w_bulk_ref, global_step_list[i_RL], avg_time_lim, avg_w_bulk_num_lim, vel_comp='w')
print("\nBuilding gifs from frames...")

filename = os.path.join(postDir, f"avg_u_bulk_numerical_vs_avg_time.gif")
frames_plot_u[0].save(filename, save_all=True, append_images=frames_plot_u[1:], duration=1000, loop=0)
print(f"MAKING GIF of numerical averaged u_bulk vs averaging time along training steps in '{filename}'" )

filename = os.path.join(postDir, f"avg_v_bulk_numerical_vs_avg_time.gif")
frames_plot_v[0].save(filename, save_all=True, append_images=frames_plot_v[1:], duration=1000, loop=0)
print(f"MAKING GIF of numerical averaged v_bulk vs averaging time along training steps in '{filename}'" )

filename = os.path.join(postDir, f"avg_w_bulk_numerical_vs_avg_time.gif")
frames_plot_w[0].save(filename, save_all=True, append_images=frames_plot_w[1:], duration=1000, loop=0)
print(f"MAKING GIF of numerical averaged w_bulk vs averaging time along training steps in '{filename}'" )

print("\nGifs plotted successfully!")




