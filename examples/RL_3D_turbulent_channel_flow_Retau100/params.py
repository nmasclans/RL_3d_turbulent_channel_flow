import random, os, numpy as np

# TODO: set custom values
# t_phys  = delta / u_tau = 1
dt_phys  = 1.0e-4       # not taken from here, defined in myRHEA.cpp
t_action = 0.01         # action period
t_begin_control = 0.0   # controls begin after this value
t_episode_train = round(1.0 + t_action + dt_phys, 8)
t_episode_eval = 1.0
cfd_n_envs = 1          # 
rl_n_envs = 8           # num. regions del domini en wall-normal direction -> gets the witness points
mode = "train"          # "train" or "eval"

params = {
    # smartsim params
    "run_id": "",
    "rhea_exe": "RHEA.exe",
    "port": random.randint(6000, 7000), # generate a random port number
    "network_interface": "ib0",
    "use_XLA": True,
    "num_dbs": 1,                       # not used if launcher == 'local'   TODO: rm if not used
    "launcher": "local",
    "run_command": "bash",
    "mpirun_mca": "btl_base_warn_component_unused 0",   # mpirun argument --mca
    "mpirun_hostfile": "my-hostfile",   # mpirun argument --hostfile
    "mpirun_np": rl_n_envs,             # mpirun argument -np
    "cluster_account": None,
    "modules_sh": None,
    "episode_walltime": None,
    "cfd_n_envs": cfd_n_envs,
    "rl_n_envs": rl_n_envs,
    ###"n_tasks_per_env": 1,            # TODO: remove param, equivalent to "-np" argument of mpirun_np
    "control_cubes_file": f"config_control_witness/cubeControl{rl_n_envs}.txt",   # used in Python if n_rl_envs == 1, but used in C++ independently of n_rl_envs
    "witness_file": f"config_control_witness/witness{rl_n_envs}.txt",
    "rl_neighbors": 0,                  # 0 is local state only, # TODO: set custom value
    "model_dtype": np.float32,
    "rhea_dtype": np.float64,
    "poll_n_tries": 1000000,               # num. tries of database poll
    "poll_freq_ms": 100,                # time between database poll tries [miliseconds]
    "time_key": "time",
    "step_type_key": "step_type",
    "state_key": "state",
    "state_size_key": "state_size",
    "action_key": "action",
    "action_size_key": "action_size",
    "reward_key": "reward",
    "dump_data_flag": True,
###    "verbosity": "debug", # quiet, debug, info

    # RL params
    "mode": mode,
    "num_episodes": 2000,
    "num_epochs": cfd_n_envs * rl_n_envs,   # number of epochs to perform policy (optimizer) update per episode sampled. Rule of thumb: n_envs.
    "t_action": t_action,
    "t_episode": t_episode_train if mode == "train" else t_episode_eval,
    "t_begin_control": t_begin_control,
    "action_bounds": (-2.0, 2.0),
    "action_dim": 6,
    "reward_norm": 1.0,                                                                 # another possible normalization: reward_norm = t_action
    "reward_beta": 0.5,                     # reward = beta * reward_global + (1.0 - beta) * reward_local,  # TODO: set custom value
    "restart_file": "restart_data_file.h5", # 3: random. 1: restart 1. 2: restart 2     # TODO: change this if we want to use several restart files
    "net": (128, 128),                                                                  # action net parameter 'fc_layer_units' & value net parameter 'fc_layer_params'
    "learning_rate": 0.0005,                                                            # recommended values: 0.0001 - 0.001
    "entropy_regularization": 0.03,                                                     # recommended values: 0.01 - 0.05
    "importance_ratio_clipping": 0.2,                                                   # recommended values: 0.2 - 0.5
    "actor_net_activation_fn": "relu",
    "actor_net_l2_reg": 1e-4,
    "actor_net_std_init": 0.35,
    "replay_buffer_capacity": int(t_episode_train / t_action) + 1, # TODO: multiply by *(cfd_n_envs * rl_n_envs) ???    # trajectories buffer expand a full train episode
    "log_interval": 1, # save model, policy, metrics, interval
    "summary_interval": 1, # write to tensorboard interval [epochs]
    "seed": 16,
    "ckpt_num": int(1e6),
    "ckpt_interval": 1,
###    "do_profile": False,
    "use_tf_functions": True
}

# Default params
### params["collect_episodes_per_iteration"] = params["n_envs"] # number of episodes to collect before each optimizer update
### os.environ["SMARTSIM_LOG_LEVEL"] = params["verbosity"] # quiet, info, debug
### os.environ["SR_LOG_LEVEL"] = params["verbosity"] # quiet, info, debug
### os.environ["SR_LOG_FILE"] = "sod2d_exp/sr_log_file.out" # SR output log

# Params groups
env_params = {
    "launcher": params["launcher"],
    "run_command": params["run_command"],
    "mpirun_mca": params["mpirun_mca"],
    "mpirun_hostfile": params["mpirun_hostfile"],
    "mpirun_np": params["mpirun_np"],
    "cluster_account": params["cluster_account"],
    "modules_sh": params["modules_sh"],
    "episode_walltime": params["episode_walltime"],
    "cfd_n_envs": params["cfd_n_envs"],
    "rl_n_envs": params["rl_n_envs"],
    ###"n_tasks_per_env": params["n_tasks_per_env"],
    "control_cubes_file": params["control_cubes_file"],
    "witness_file": params["witness_file"],
    "rl_neighbors": params["rl_neighbors"],
    "model_dtype": params["model_dtype"],
    "rhea_dtype": params["rhea_dtype"],
    "poll_n_tries": params["poll_n_tries"],
    "poll_freq_ms": params["poll_freq_ms"],
    "t_action": params["t_action"],
    "t_episode": params["t_episode"],
    "t_begin_control": params["t_begin_control"],
    "action_bounds": params["action_bounds"],
    "action_dim": params["action_dim"],
    "reward_norm": params["reward_norm"],
    "reward_beta": params["reward_beta"],
    "time_key": params["time_key"],
    "step_type_key": params["step_type_key"],
    "state_key": params["state_key"],
    "state_size_key": params["state_size_key"],
    "action_key": params["action_key"],
    "action_size_key": params["action_size_key"],
    "reward_key": params["reward_key"],
    "dump_data_flag": params["dump_data_flag"],
}
