#!/home/jofre/miniconda3/envs/smartrhea-env-v2/bin/python3

import sys
import os
import glob
import itertools
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

restart_data_file_time = 323.99999999   # restart_data_file attribute 'Time'
restart_data_file_averaging_time = 5.0  # restart_data_file attribute 'AveragingTime'
t_avg_0        = restart_data_file_time - restart_data_file_averaging_time 

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

# --- Get reward terms data from .out files ---
print("\nImporting reward terms data from .out files...")
n_RL              = len(filepath_list)
avg_time_dict     = dict.fromkeys(np.arange(n_RL))
local_reward_dict = dict.fromkeys(np.arange(n_RL))
avg_u_reward_dict = dict.fromkeys(np.arange(n_RL))
avg_v_reward_dict = dict.fromkeys(np.arange(n_RL))
avg_w_reward_dict = dict.fromkeys(np.arange(n_RL))
rms_u_reward_dict = dict.fromkeys(np.arange(n_RL))
rms_v_reward_dict = dict.fromkeys(np.arange(n_RL))
rms_w_reward_dict = dict.fromkeys(np.arange(n_RL))

#[myRHEA::calculateSourceTerms] Performing SmartRedis communications (state, action, reward) at time instant 324.2039999903
#[myRHEA::calculateReward] [Rank 147] Local reward: 0.1887369033, with reward terms: 0.2620923375 0.0108779954 0.0212083778 0.0170843860
data_pattern_time   = re.compile(r"Performing SmartRedis communications \(state, action, reward\) at time instant\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
data_pattern_reward = re.compile(r"\[Rank\s+(\d+)\] Local reward:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?),\s*with reward terms:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
for i_RL in range(n_RL):
    
    # Read file
    filepath = filepath_list[i_RL]
    with open(filepath) as f:
        lines = f.readlines()
    
    # Get values of interest: reward terms & time
    time_list             = []
    local_reward_per_rank = {k: [] for k in np.arange(rl_n_envs)}
    avg_u_reward_per_rank = {k: [] for k in np.arange(rl_n_envs)}
    avg_v_reward_per_rank = {k: [] for k in np.arange(rl_n_envs)}
    avg_w_reward_per_rank = {k: [] for k in np.arange(rl_n_envs)}
    rms_u_reward_per_rank = {k: [] for k in np.arange(rl_n_envs)}
    rms_v_reward_per_rank = {k: [] for k in np.arange(rl_n_envs)}
    rms_w_reward_per_rank = {k: [] for k in np.arange(rl_n_envs)}
    for line in lines:
        match_time   = data_pattern_time.search(line)
        match_reward = data_pattern_reward.search(line)
        if match_time:
            time_list.append(float(match_time.group(1)) )
        if match_reward:
            rank_id = int(match_reward.group(1))
            local_reward_per_rank[rank_id].append(float(match_reward.group(2))) 
            avg_u_reward_per_rank[rank_id].append(float(match_reward.group(3))) 
            avg_v_reward_per_rank[rank_id].append(float(match_reward.group(4))) 
            avg_w_reward_per_rank[rank_id].append(float(match_reward.group(5))) 
            rms_u_reward_per_rank[rank_id].append(float(match_reward.group(6))) 
            rms_v_reward_per_rank[rank_id].append(float(match_reward.group(7))) 
            rms_w_reward_per_rank[rank_id].append(float(match_reward.group(8))) 

    # Averaged values across all ranks / rl_n_envs
    n_t = len(time_list)
    avg_time_dict[i_RL]     = np.array(time_list) - t_avg_0
    local_reward_dict[i_RL] = np.mean(np.stack(list(local_reward_per_rank.values())), axis=0)[-n_t:]
    avg_u_reward_dict[i_RL] = np.mean(np.stack(list(avg_u_reward_per_rank.values())), axis=0)[-n_t:]
    avg_v_reward_dict[i_RL] = np.mean(np.stack(list(avg_v_reward_per_rank.values())), axis=0)[-n_t:]
    avg_w_reward_dict[i_RL] = np.mean(np.stack(list(avg_w_reward_per_rank.values())), axis=0)[-n_t:]
    rms_u_reward_dict[i_RL] = np.mean(np.stack(list(rms_u_reward_per_rank.values())), axis=0)[-n_t:]
    rms_v_reward_dict[i_RL] = np.mean(np.stack(list(rms_v_reward_per_rank.values())), axis=0)[-n_t:]
    rms_w_reward_dict[i_RL] = np.mean(np.stack(list(rms_w_reward_per_rank.values())), axis=0)[-n_t:]
    print(f"Numerical avg_u_bulk and Time data imported from file '{filepath}'")

print("\nData imported successfully!")

# Min, Max values

all_avg_time = np.concatenate(list(itertools.chain.from_iterable([
    avg_time_dict.values(),
])))
all_rewards = np.concatenate(list(itertools.chain.from_iterable([
#   local_reward_dict.values(),
    avg_u_reward_dict.values(),
    avg_v_reward_dict.values(),
    avg_w_reward_dict.values(),
    rms_u_reward_dict.values(),
    rms_v_reward_dict.values(),
    rms_w_reward_dict.values()
])))
avg_time_min, avg_time_max = all_avg_time.min(), all_avg_time.max()
reward_min, reward_max = all_rewards.min(), all_rewards.max()
avg_time_lim = [avg_time_min, avg_time_max]
rewards_lim  = [reward_min, reward_max]
print(f"Averaging time limits: {avg_time_lim}")
print(f"Reward terms limits: {rewards_lim}")

# ----------------- Plot Animation Frames for increasing RL global step (specific ensemble) -----------------

print("\nBuilding reward terms image frames...")
frames = []
for i_RL in range(n_RL):
    # log progress
    if i_RL % (n_RL//10 or 1) == 0:
        print(f"{i_RL/n_RL*100:.0f}%")
    # Build image frames
    frames = visualizer.build_rewards_terms_frames(
        frames, 
        avg_time_dict[i_RL],
        local_reward_dict[i_RL],
        avg_u_reward_dict[i_RL],
        avg_v_reward_dict[i_RL],
        avg_w_reward_dict[i_RL],
        rms_u_reward_dict[i_RL],
        rms_v_reward_dict[i_RL],
        rms_w_reward_dict[i_RL], 
        avg_time_lim,
        rewards_lim,
        global_step_list[i_RL], 
    )

print("\nBuilding gifs from frames...")
filename = os.path.join(postDir, f"reward_terms_vs_avg_time.gif")
frames[0].save(filename, save_all=True, append_images=frames[1:], duration=1000, loop=0)
print(f"MAKING GIF of reward terms vs averaging time along training steps in '{filename}'" )
print("\nGifs plotted successfully!")




