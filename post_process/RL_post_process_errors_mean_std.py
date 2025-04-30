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
CASE_NAME="U12"

def extract_error_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract Averaging time RL, Averaging time non-RL, L2 Error avg_u RL, L2 Error avg_u nonRL, L2 Error rmsf_u RL, and L2 Error rmsf_u nonRL."""
    
    with open(file_path, 'r') as f:
        content = f.read()

    # Define patterns
    patterns = {
        'averaging_time_rl': r'Averaging time accumulated RL: \[(.*?)\]',
        'averaging_time_nonrl': r'Averaging time non-RL: \[(.*?)\]',
        'l2_error_avg_u_rl': r'L2 Error avg_u RL.*?: \[(.*?)\]',
        'l2_error_avg_u_nonrl': r'L2 Error avg_u nonRL.*?: \[(.*?)\]',
        'l2_error_rmsf_u_rl': r'L2 Error rmsf_u RL.*?: \[(.*?)\]',
        'l2_error_rmsf_u_nonrl': r'L2 Error rmsf_u nonRL.*?: \[(.*?)\]',
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
            extracted['l2_error_avg_u_nonrl'],
            extracted['l2_error_rmsf_u_rl'],
            extracted['l2_error_rmsf_u_nonrl'])

if __name__ == "__main__":

    file_name       = "errors_velocity_statistics.txt"
    case_name_list  = [f"RL_3D_turbulent_channel_flow_Retau100_sup_144_15max_TCP3_a3_s5_{CASE_NAME}_{rep}" for rep in [2,3,4,5]]
    train_name_list = ["train_2025-04-27--02-48-46--75b2", "train_2025-04-27--11-58-41--821e", "train_2025-04-27--21-57-29--0e87", "train_2025-04-30--12-32-00--9f7d"]
    num_cases       = len(case_name_list)
    averaging_time_rl     = {key: None for key in case_name_list}
    l2_error_avg_u_rl     = {key: None for key in case_name_list}
    l2_error_rmsf_u_rl    = {key: None for key in case_name_list}
    averaging_time_nonrl  = {key: None for key in case_name_list}
    l2_error_avg_u_nonrl  = {key: None for key in case_name_list}
    l2_error_rmsf_u_nonrl = {key: None for key in case_name_list}
    for i in range(num_cases):
        case_name  = case_name_list[i]
        train_name = train_name_list[i]
        file_path  = os.path.join(REPO_DIR,  "examples",  case_name,  "post_process", train_name, file_name)
        print(f"Extract error data from file: {file_path}")
        averaging_time_rl_, averaging_time_nonrl_, l2_error_avg_u_rl_, l2_error_avg_u_nonrl_, l2_error_rmsf_u_rl_, l2_error_rmsf_u_nonrl_ \
            = extract_error_data(file_path)
        averaging_time_rl[case_name]     = averaging_time_rl_
        l2_error_avg_u_rl[case_name]     = l2_error_avg_u_rl_
        l2_error_rmsf_u_rl[case_name]    = l2_error_rmsf_u_rl_
        averaging_time_nonrl[case_name]  = averaging_time_nonrl_
        l2_error_avg_u_nonrl[case_name]  = l2_error_avg_u_nonrl_
        l2_error_rmsf_u_nonrl[case_name] = l2_error_rmsf_u_nonrl_
    del averaging_time_rl_, l2_error_avg_u_rl_, averaging_time_nonrl_, l2_error_avg_u_nonrl_, l2_error_rmsf_u_rl_, l2_error_rmsf_u_nonrl_

    # Interpolate data for all unique averaging times for RL cases
    all_times = np.unique(np.concatenate([averaging_time_rl[k] for k in case_name_list]))
    n_t       = len(all_times)
    mean_err_avg_u = np.zeros(n_t);    mean_err_rmsf_u = np.zeros(n_t)
    std_err_avg_u  = np.zeros(n_t);    std_err_rmsf_u  = np.zeros(n_t)
    min_err_avg_u  = np.zeros(n_t);    min_err_rmsf_u  = np.zeros(n_t)
    max_err_avg_u  = np.zeros(n_t);    max_err_rmsf_u  = np.zeros(n_t)
    for t in range(n_t):
        time      = all_times[t] 
        err_avg_u = [];                                                     err_rmsf_u = []
        for case_name in case_name_list:
            time_i = averaging_time_rl[case_name]
            err_avg_u_i  = l2_error_avg_u_rl[case_name];                    err_rmsf_u_i = l2_error_rmsf_u_rl[case_name]
            if time_i[0] <= time <= time_i[-1]:
                interp_err_avg_u  = np.interp(time, time_i, err_avg_u_i);   interp_err_rmsf_u = np.interp(time, time_i, err_rmsf_u_i)
                err_avg_u.append(interp_err_avg_u);                         err_rmsf_u.append(interp_err_rmsf_u)
        if err_avg_u and err_rmsf_u: # not-empty
            err_avg_u = np.array(err_avg_u);                                err_rmsf_u = np.array(err_rmsf_u)
            print(f"Avg. Time: {time}, Errors avg_u: {err_avg_u}, Errors rmsf_u: {err_rmsf_u}")
            mean_err_avg_u[t] = np.mean(err_avg_u);                         mean_err_rmsf_u[t] = np.mean(err_rmsf_u)
            std_err_avg_u[t]  = np.std(err_avg_u);                          std_err_rmsf_u[t]  = np.std(err_rmsf_u)
            min_err_avg_u[t]  = np.min(err_avg_u);                          min_err_rmsf_u[t]  = np.min(err_rmsf_u)
            max_err_avg_u[t]  = np.max(err_avg_u);                          max_err_rmsf_u[t]  = np.max(err_rmsf_u)
        else:
            raise ValueError(f"ERROR: Avg. Time {time} is not available for any RL case.") 

    # Crop arrays avg. to time <= 50.0
    # > crop RL data
    idx = all_times <= 50.0
    all_times = all_times[idx]
    mean_err_avg_u  = mean_err_avg_u[idx];                                  mean_err_rmsf_u  = mean_err_rmsf_u[idx]
    std_err_avg_u   = std_err_avg_u[idx];                                   std_err_rmsf_u   = std_err_rmsf_u[idx]
    min_err_avg_u   = min_err_avg_u[idx];                                   min_err_rmsf_u   = min_err_rmsf_u[idx]
    max_err_avg_u   = max_err_avg_u[idx];                                   max_err_rmsf_u   = max_err_rmsf_u[idx]
    # > crop nonRL data
    idx = averaging_time_nonrl[case_name_list[0]] <= 50.0
    avg_times_nonRL = averaging_time_nonrl[case_name_list[0]][idx];         
    err_avg_u_nonRL = l2_error_avg_u_nonrl[case_name_list[0]][idx];         err_rmsf_u_nonRL = l2_error_rmsf_u_nonrl[case_name_list[0]][idx];         

    # Plot avg_u error
    plt.figure(figsize=(10, 6))
    plt.semilogy(all_times, mean_err_avg_u,                                 color = plt.cm.tab10(3), label=r'$\mathbb{E}[\varepsilon]$')
    plt.semilogy(all_times, mean_err_avg_u + std_err_avg_u, linestyle='-.', color = plt.cm.tab10(3), label=r'$\pm \mathbb{V}^{1/2}[\varepsilon]$')
    plt.semilogy(all_times, mean_err_avg_u - std_err_avg_u, linestyle='-.', color = plt.cm.tab10(3))
    plt.fill_between(all_times, min_err_avg_u, max_err_avg_u, alpha=0.3,    color='gray',            label=r'$\{\varepsilon\}$')
    plt.semilogy(avg_times_nonRL, err_avg_u_nonRL,          linestyle='-',  color = plt.cm.tab10(0), label='Uncontrolled')
    plt.xlabel(r'Cummulative averaging time $t_{avg}^+$' )
    plt.ylabel(r"$\varepsilon=|| \overline{u} - \overline{u}_C ||_2$")
    plt.legend()
    plt.grid(which='both',axis='y')
    plt.tight_layout()
    plt.savefig(f"results_ensemble_average/{CASE_NAME}_l2_err_avg_u.svg")
    plt.savefig(f"results_ensemble_average/{CASE_NAME}_l2_err_avg_u.png")
    plt.close()

    # Plot avg_u error
    plt.figure(figsize=(10, 6))
    plt.semilogy(all_times, mean_err_rmsf_u,                                  color = plt.cm.tab10(3), label=r'$\mathbb{E}[\varepsilon]$')
    plt.semilogy(all_times, mean_err_rmsf_u + std_err_rmsf_u, linestyle='-.', color = plt.cm.tab10(3), label=r'$\pm \mathbb{V}^{1/2}[\varepsilon]$')
    plt.semilogy(all_times, mean_err_rmsf_u - std_err_rmsf_u, linestyle='-.', color = plt.cm.tab10(3))
    plt.fill_between(all_times, min_err_rmsf_u, max_err_rmsf_u, alpha=0.3,    color='gray',            label=r'$\{\varepsilon\}$')
    plt.semilogy(avg_times_nonRL, err_rmsf_u_nonRL,           linestyle='-',  color = plt.cm.tab10(0), label='Uncontrolled')
    plt.xlabel(r'Cummulative averaging time $t_{avg}^+$' )
    plt.ylabel(r"$\varepsilon=|| \overline{u} - \overline{u}_C ||_2$")
    plt.legend()
    plt.grid(which='both',axis='y')
    plt.tight_layout()
    plt.savefig(f"results_ensemble_average/{CASE_NAME}_l2_err_rmsf_u.svg")
    plt.savefig(f"results_ensemble_average/{CASE_NAME}_l2_err_rmsf_u.png")
    plt.close()
