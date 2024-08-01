import random, os, numpy as np

# TODO: set custom values
t_action = 0.01         # action time duration
t_begin_control = 0.0   # controls begin after this value
t_episode_train = 1.0
t_episode_eval = 5.0
cfd_n_envs = 8
rl_n_envs = 3           # num. regions del domini en spanwise direction -> gets the witness points
mode = "train"          # "train" or "eval"

params = {
    # smartsim params
    "port": random.randint(6000, 7000), # generate a random port number
    "network_interface": "ib0",
    "use_XLA": True,
    "num_dbs": 1,                       # not used if launcher == 'local'   TODO: rm if not used
    "launcher": "local",
    "run_command": "mpirun",            # TODO: rm if not used
    "cluster_account": None,
    "modules_sh": None,
    "episode_walltime": None,
    "cfd_n_envs": cfd_n_envs,
    "rl_n_envs": rl_n_envs,
    "n_tasks_per_env": 1,               # TODO: set custom value
    "rectangle_file": "rectangleControl.txt",   # only used if rl_n_envs == 1, TODO: set custom file contents
    "witness_file": "witness.txt",      # TODO: set custom file contents
    "witness_xyz": (6, 6, 6),           # TODO: set custom value
    "rl_neighbors": 1,                  # 0 is local state only, # TODO: set custom value
    "model_dtype": np.float32,
    "rhea_dtype": np.float64,
    "poll_n_tries": 1000,               # num. tries of database poll
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
###    "mode": mode,                # TODO: rm if not used, mode='collect' used in RheaEnv 
###    "num_episodes": 2000,
###    "num_epochs": cfd_n_envs * rl_n_envs, # number of epochs to perform policy (optimizer) update per episode sampled. Rule of thumb: n_envs.
    "f_action": 1.0 / t_action,
    "t_episode": t_episode_train if mode == "train" else t_episode_eval,
    "t_begin_control": t_begin_control,
    "action_bounds": (-0.3, 0.3),   # TODO: set custom value
    "reward_norm": 153.6, # non-actuated lx in coarse mesh, # TODO: set custom value
    "reward_beta": 0.5, # reward = beta * reward_global + (1.0 - beta) * reward_local, # TODO: set custom value
###    "restart_file": 3, # 3: random. 1: restart 1. 2: restart 2
    "net": (128, 128),              # action net parameter 'fc_layer_units' & value net parameter 'fc_layer_params'
###    "learning_rate": 5e-4,
###    "replay_buffer_capacity": int(t_episode_train / t_action) + 1, # trajectories buffer
###    "log_interval": 1, # save model, policy, metrics, interval
###    "summary_interval": 1, # write to tensorboard interval
###    "seed": 16,
###    "ckpt_num": int(1e6),
###    "ckpt_interval": 1,
###    "do_profile": False,
###    "use_tf_functions": True
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
    "cluster_account": params["cluster_account"],
    "modules_sh": params["modules_sh"],
    "episode_walltime": params["episode_walltime"],
    "cfd_n_envs": params["cfd_n_envs"],
    "rl_n_envs": params["rl_n_envs"],
    "n_tasks_per_env": params["n_tasks_per_env"],
    "rectangle_file": params["rectangle_file"],
    "witness_file": params["witness_file"],
    "witness_xyz": params["witness_xyz"],
    "rl_neighbors": params["rl_neighbors"],
    "model_dtype": params["model_dtype"],
    "rhea_dtype": params["rhea_dtype"],
    "poll_n_tries": params["poll_n_tries"],
    "poll_freq_ms": params["poll_freq_ms"],
    "f_action": params["f_action"],
    "t_episode": params["t_episode"],
    "t_begin_control": params["t_begin_control"],
    "action_bounds": params["action_bounds"],
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
