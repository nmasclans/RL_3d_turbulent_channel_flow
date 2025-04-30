# File: extract_errors_velocity_statistics.py

import re
import numpy as np
import os
from typing import Tuple
import matplotlib.pyplot as plt

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 12, frameon = False)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')

REPO_DIR="/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow"

def extract_error_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract Averaging time RL, Averaging time non-RL, L2 Error avg_u RL, and L2 Error avg_u nonRL."""
    
    with open(file_path, 'r') as f:
        content = f.read()

    # Define patterns
    patterns = {
        'averaging_time_rl': r'Averaging time accumulated RL: \[(.*?)\]',
        'averaging_time_nonrl': r'Averaging time non-RL: \[(.*?)\]',
        'l2_error_avg_u_rl': r'L2 Error avg_u RL.*?: \[(.*?)\]',
        'l2_error_avg_u_nonrl': r'L2 Error avg_u nonRL.*?: \[(.*?)\]'
    }

    # Extract arrays
    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Pattern '{key}' not found in file.")
        numbers_str = match.group(1).replace('\n', ' ').replace('\r', ' ')
        numbers = np.fromstring(numbers_str, sep=' ')
        extracted[key] = numbers

    return (extracted['averaging_time_rl'],
            extracted['averaging_time_nonrl'],
            extracted['l2_error_avg_u_rl'],
            extracted['l2_error_avg_u_nonrl'])

if __name__ == "__main__":

    file_name       = "errors_velocity_statistics.txt"
    case_name_list  = [f"RL_3D_turbulent_channel_flow_Retau100_sup_144_15max_TCP3_a3_s5_U12_{rep}" for rep in [2,3,4,5]]
    train_name_list = ["train_2025-04-27--02-48-46--75b2", "train_2025-04-27--11-58-41--821e", "train_2025-04-27--21-57-29--0e87", "train_2025-04-28--07-09-46--ce34"]
    num_cases       = len(case_name_list)
    averaging_time_rl    = {key: None for key in case_name_list}
    l2_error_avg_u_rl    = {key: None for key in case_name_list}
    averaging_time_nonrl = {key: None for key in case_name_list}
    l2_error_avg_u_nonrl = {key: None for key in case_name_list}
    for i in range(num_cases):
        case_name  = case_name_list[i]
        train_name = train_name_list[i]
        file_path  = os.path.join(REPO_DIR,  "examples",  case_name,  "post_process", train_name, file_name)
        print(f"Extract error data from file: {file_path}")
        averaging_time_rl_, averaging_time_nonrl_, l2_error_avg_u_rl_, l2_error_avg_u_nonrl_ = extract_error_data(file_path)
        averaging_time_rl[case_name]    = averaging_time_rl_
        l2_error_avg_u_rl[case_name]    = l2_error_avg_u_rl_
        averaging_time_nonrl[case_name] = averaging_time_nonrl_
        l2_error_avg_u_nonrl[case_name] = l2_error_avg_u_nonrl_
    del averaging_time_rl_, l2_error_avg_u_rl_, averaging_time_nonrl_, l2_error_avg_u_nonrl_
    #print("\nAveraging time RL:", averaging_time_rl)
    #print("\nAveraging time non-RL:", averaging_time_nonrl)
    #print("\nL2 Error avg_u RL:", l2_error_avg_u_rl)
    #print("\nL2 Error avg_u nonRL:", l2_error_avg_u_nonrl)

    # Interpolate data for all unique averaging times for RL cases
    all_times = np.unique(np.concatenate([averaging_time_rl[k] for k in case_name_list]))
    n_t       = len(all_times)
    mean_err  = np.zeros(n_t)
    std_err   = np.zeros(n_t)
    min_err   = np.zeros(n_t)
    max_err   = np.zeros(n_t)
    for t in range(n_t):
        time = all_times[t] 
        err  = []
        for case_name in case_name_list:
            time_i = averaging_time_rl[case_name]
            err_i  = l2_error_avg_u_rl[case_name]
            if time_i[0] <= time <= time_i[-1]:
                interp_err = np.interp(time, time_i, err_i)
                err.append(interp_err)
        if err: # not-empty
            err = np.array(err)
            print(f"Avg. Time: {time}, Errors: {err}")
            mean_err[t] = np.mean(err)
            std_err[t]  = np.std(err)
            min_err[t]  = np.min(err)
            max_err[t]  = np.max(err)
        else:
            raise ValueError(f"ERROR: Avg. Time {time} is not available for any RL case.") 

    # Crop arrays avg. to time <= 50.0
    # > crop RL data
    idx = all_times <= 50.0
    all_times = all_times[idx]
    mean_err  = mean_err[idx]
    std_err   = std_err[idx]
    min_err   = min_err[idx]
    max_err   = max_err[idx]
    # > crop nonRL data
    idx = averaging_time_nonrl[case_name_list[0]] <= 50.0
    avg_times_nonRL = averaging_time_nonrl[case_name_list[0]][idx]
    err_nonRL       = l2_error_avg_u_nonrl[case_name_list[0]][idx]

    plt.figure(figsize=(10, 6))
    plt.semilogy(all_times, mean_err,                           color = plt.cm.tab10(3), label='Mean Error RL')
    plt.semilogy(all_times, mean_err + std_err, linestyle='-.', color = plt.cm.tab10(3), label='+1 Std Dev')
    plt.semilogy(all_times, mean_err - std_err, linestyle='-.', color = plt.cm.tab10(3), label='-1 Std Dev')
    plt.fill_between(all_times, min_err, max_err, color='gray', alpha=0.3, label='Error Range')
    plt.semilogy(avg_times_nonRL, err_nonRL, linestyle='-', color = plt.cm.tab10(0), label='Uncontrolled')
    plt.xlabel("Averaging Time (RL)")
    plt.ylabel("L2 Error avg_u (RL)")
    plt.title("RL Velocity Error Statistics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("temp_fig.svg")
    plt.savefig("temp_fig.png")
    plt.close()
