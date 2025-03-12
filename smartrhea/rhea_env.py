import os
import coloredlogs
import glob
import logging
import numpy as np
import random
import shutil

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

from smartredis import Client
from smartsim.settings import MpirunSettings, RunSettings
from smartsim.settings.settings import create_batch_settings
from smartsim.log import get_logger
from smartrhea.init_smartsim import write_hosts
from smartrhea.utils import n_witness_points, n_cubes, get_witness_xyz, numpy_str, bcolors, delete_all_files_in_dir

EPS = 1e-8

logger = get_logger(__name__)

"""
Important information: 
1. Actions, State & Rewards Normalization/Standarization:
    - actor_network uses 'tanh' activation function: actions are 0-centered, and bounded by parameter 'action_bounds' -> done automatically by 'action_bounds' as specified in 'action_tensor_spec' 
    - value_network uses 'ReLU' activation function 
"""

class RheaEnv(py_environment.PyEnvironment):
    """
    RHEA environment(s) extending the tf_agents.environments.PyEnvironment class.
    Fields:
    - exp: SmartSim experiment class instance
    - db: SmartSim orchestrator class instance
    - hosts: list of host nodes
    ### **env_params
    - launcher: "local", other
    - run_command: "mpirun" ("srun" still not working)
    - mpirun_mca: mpirun argument 'mca' (of flag --)
    - mpirun_hostfile: mpirun argument 'hostfile' (of flag --)
    - mpirun_np: mpirun argument 'np', number of processors (of flag -np)
    - cluster_account: project account, if required for specific launcher
    - modules_sh: modules file, if required for specific launcher
    - episode_walltime: environment walltime, if required for specific launcher
    - cfd_n_envs: number of cdf simulations
    - rl_n_envs: number of rl environments inside a cfd simulation
    - config_dir: configuration files dir
    - control_cubes_filename: actuators cubes filename
    - witness_filename: witness points filename
    - rl_neighbors: number of witness blocks selected to compose the state
    - model_dtype: data type for the model
    - rhea_dtype: data type for arrays to be sent to RHEA (actions)
    - poll_n_tries: num. tries of database poll
    - poll_freq_ms: time between database poll tries [miliseconds]
    - t_action: action elapsed time
    - t_episode: episode elapsed time
    - t_begin_control: time to start control
    - action_bounds: bounds for action values
    - action_dim: action dimension
    - reward_norm:
    - reward_beta:
    - time_key:
    - step_type_key:
    - state_key:
    - state_size_key:
    - action_key:
    - action_size_key:
    - reward_key:
    - dump_data_flag:
    ### Not in **env_params:
    - mode:
    - db_is_clustered
    """

    def __init__( # pylint: disable=super-init-not-called
        self,
        exp,
        db,
        hosts,
        ### **env_params:
        rhea_exe = "RHEA.exe",
        rhea_case_path = "",
        rl_case_path = "",
        launcher = "local",
        run_command = "mpirun",
        mpirun_mca = "btl_base_warn_component_unused",
        mpirun_hostfile = "my-hostfile",
        mpirun_np = 1,
        cluster_account = None,
        modules_sh = None,
        episode_walltime = None,
        cfd_n_envs = 2,
        rl_n_envs = 5,
        config_dir = "",
        control_cubes_filename = "cubeControl.txt",
        witness_filename = "witness.txt",
        rl_neighbors = 1,
        model_dtype = np.float32,
        rhea_dtype = np.float64,
        poll_n_tries = 1000,
        poll_freq_ms = 100,
        t_action = 0.001,
        t_episode = 1.0,
        t_begin_control = 0.0,
        action_bounds = (-0.05, 0.05),
        action_dim = 6,
        state_dim = 1,
        reward_norm = 1.0,
        reward_beta = 0.5,
        time_key = "time",
        step_type_key = "step_type",
        state_key = "state",
        state_size_key = "state_size",
        action_key = "action",
        action_size_key = "action_size",
        reward_key = "reward",
        dump_data_flag = True,
        ### Not in **env_params:
        dump_data_path = "train_<run_id>",
        mode = "collect",           # TODO: consider using params['mode'], which is currently train/eval
        db_is_clustered = False,

    ):

        # Store input parameters
        self.exp = exp
        self.db = db
        self.hosts = hosts
        # **env_params:
        self.rhea_exe_fname = rhea_exe
        self.rhea_case_path = rhea_case_path
        self.rl_case_path = rl_case_path
        self.launcher = launcher
        self.run_command = run_command
        self.mpirun_mca = mpirun_mca
        self.mpirun_hostfile = mpirun_hostfile
        self.mpirun_np = mpirun_np
        self.cluster_account = cluster_account
        self.modules_sh = modules_sh
        self.episode_walltime = episode_walltime
        self.cfd_n_envs = cfd_n_envs
        self.rl_n_envs = rl_n_envs
        self.config_dir = config_dir
        self.control_cubes_filename = control_cubes_filename
        self.witness_filename = witness_filename
        self.rl_neighbors = rl_neighbors
        self.model_dtype = model_dtype
        self.rhea_dtype = rhea_dtype
        self.poll_n_tries = poll_n_tries
        self.poll_freq_ms = poll_freq_ms
        self.action_bounds = action_bounds
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.reward_norm = reward_norm
        self.reward_beta = reward_beta
        self.dump_data_flag = dump_data_flag
        self.dump_data_path = dump_data_path
        self.mode = mode

        # calculated parameters
        self.n_envs = cfd_n_envs * rl_n_envs
        
        # preliminary checks
        if (2 * rl_neighbors + 1 > rl_n_envs):
            raise ValueError(f"Number of witness blocks selected to compose the state exceed the number total witness blocks:\n \
                Witness blocks selected: {2 * rl_neighbors + 1}\n \
                Total witness blocks: {rl_n_envs}\n")
        
        # manage directories
        if not os.path.exists(self.dump_data_path):
            os.makedirs(self.dump_data_path)
        if self.mode == "eval":
            if os.path.exists(self.dump_data_path):
                logger.warning(f"Evaluation data path already exists: `{self.dump_data_path}`")
                logger.warning(f"Removing content of evaluation data path...")
                delete_all_files_in_dir(self.dump_data_path)
            else:
                os.makedirs(self.dump_data_path)
        if self.dump_data_flag:
            if not os.path.exists(os.path.join(self.dump_data_path, "state")):
                os.makedirs(os.path.join(self.dump_data_path, "state"))
            if not os.path.exists(os.path.join(self.dump_data_path, "local_reward")):
                os.makedirs(os.path.join(self.dump_data_path, "local_reward"))
            if not os.path.exists(os.path.join(self.dump_data_path, "reward")):
                os.makedirs(os.path.join(self.dump_data_path, "reward"))
            if not os.path.exists(os.path.join(self.dump_data_path, "action")):
                os.makedirs(os.path.join(self.dump_data_path, "action"))
            if not os.path.exists(os.path.join(self.dump_data_path, "time")):
                os.makedirs(os.path.join(self.dump_data_path, "time"))
            if not os.path.exists(os.path.join(self.dump_data_path, "mpi_output")):
                os.makedirs(os.path.join(self.dump_data_path, "mpi_output"))

        # manage directories 'rhea_exp/output' & 'rhea_exp/timers_info' in 'rl_case_path'
        rhea_exp_dir = "rhea_exp"
        timers_info_dir = os.path.join(self.rl_case_path, rhea_exp_dir, "timers_info")
        output_data_dir = os.path.join(self.rl_case_path, rhea_exp_dir, "output_data")
        self.temporal_time_probes_dir = os.path.join(self.rl_case_path, rhea_exp_dir, "temporal_time_probes")
        
        if not os.path.exists(rhea_exp_dir):    # create directory 'rhea_exp'
            os.makedirs(rhea_exp_dir)
        if not os.path.exists(timers_info_dir):
            os.makedirs(timers_info_dir)
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
        if not os.path.exists(self.temporal_time_probes_dir):
            os.makedirs(self.temporal_time_probes_dir)
        delete_all_files_in_dir(timers_info_dir)
        delete_all_files_in_dir(output_data_dir)
        delete_all_files_in_dir(self.temporal_time_probes_dir)

        # generate database ensemble keys
        self.time_key = ["ensemble_" + str(i) + "." + time_key for i in range(self.cfd_n_envs)]
        self.step_type_key = ["ensemble_" + str(i) + "." + step_type_key for i in range(self.cfd_n_envs)]
        self.state_key = ["ensemble_" + str(i) + "." + state_key for i in range(self.cfd_n_envs)]
        self.action_key = ["ensemble_" + str(i) + "." + action_key for i in range(self.cfd_n_envs)]
        self.reward_key = ["ensemble_" + str(i) + "." + reward_key for i in range(self.cfd_n_envs)]
        self.state_size_key = state_size_key
        self.action_size_key = action_size_key

        # connect Python Redis client to an orchestrator database
        self.db_address = db.get_address()[0]
        os.environ["SSDB"] = self.db_address
        self.client = Client(address=self.db_address, cluster=self.db.batch)                    # Client(cluster: bool, address: optional(str)=None, logger_name: str=”Default”)
        logger.info(f"RheaEnv defined env. variable SSDB = {self.db_address}")
        logger.info(f"Connected Python Redis client to orchestrator database with address = {self.db_address}, cluster = {self.db.batch}")

        # create RHEA executable arguments
        self.tag                = [str(i) for i in range(self.cfd_n_envs)] # environment tags [0, 1, ..., cfd_n_envs - 1]
        self.configuration_file = [f"{self.rhea_case_path}/configuration_file.yaml" for _ in range(self.cfd_n_envs)]
        self.t_action           = [str(t_action) for _ in range(self.cfd_n_envs)]
        self.t_episode          = [str(t_episode) for _ in range(self.cfd_n_envs)]
        self.t_begin_control    = [str(t_begin_control) for _ in range(self.cfd_n_envs)]
        self.db_clustered       = [str(db_is_clustered) for _ in range(self.cfd_n_envs)]

        # create RHEA ensemble models inside experiment
        self.ensemble = None
        self._episode_ended = False
        self.envs_initialised = False

        # State: witness points information (to obtain dimensions of state arrays)
        witness_filepath = os.path.join(self.rl_case_path, self.config_dir, self.witness_filename)
        control_cubes_filepath = os.path.join(os.path.join(self.rl_case_path, self.config_dir, self.control_cubes_filename))
        self.witness_xyz = get_witness_xyz(witness_filepath)
        num_witness_points = n_witness_points(witness_filepath)
        assert np.prod(self.witness_xyz) == num_witness_points
        self.n_state_wit    = num_witness_points
        self.n_state        = self.n_state_wit * self.state_dim
        self.n_state_wit_rl = int((2 * self.rl_neighbors + 1) * (self.n_state_wit / self.rl_n_envs))
        self.n_state_rl     = self.n_state_wit_rl * self.state_dim
        assert self.n_state % self.rl_n_envs == 0, f"ERROR: number of witness points ({self.n_state}) must be multiple of number of rl environments ({self.rl_n_envs}), so that each environment has the same number of witness points"
        
        # Actions
        if self.rl_n_envs > 1:
            self.n_action = 1 * self.action_dim
            num_cubes = n_cubes(control_cubes_filepath)
            assert self.rl_n_envs == num_cubes, f"(num. rl environments = {self.rl_n_envs}) != (num. control cubes = {num_cubes})"
            # TODO: this assert is not done in SOD2D (where marl_n_envs=3 != n_control_rectangles=6), but consider doing assert of n_control_rectangles == rl_n_envs
        else:   # self.rl_n_envs == 1:
            self.n_action = n_cubes(control_cubes_filepath) * self.action_dim

        # create and allocate array objects
        self.__state       = np.zeros((self.cfd_n_envs, self.n_state_wit, self.state_dim), dtype=self.model_dtype)
        self._state        = np.zeros((self.cfd_n_envs, self.n_state), dtype=self.model_dtype)
        self.__state_rl    = np.zeros((self.n_envs, self.n_state_wit_rl, self.state_dim), dtype=self.model_dtype)
        self._state_rl     = np.zeros((self.n_envs, self.n_state_rl), dtype=self.model_dtype)
        self._action       = np.zeros((self.cfd_n_envs, self.n_action * self.rl_n_envs), dtype=self.rhea_dtype)
        self._action_znmf  = np.zeros((self.cfd_n_envs, 2 * self.n_action * self.rl_n_envs), dtype=self.rhea_dtype)
        self._local_reward = np.zeros((self.cfd_n_envs, self.rl_n_envs))
        self._reward       = np.zeros(self.n_envs)
        self._time         = np.zeros(self.cfd_n_envs, dtype=self.model_dtype)
        self._step_type    = - np.ones(self.cfd_n_envs, dtype=int) # init status in -1
        self._episode_global_step = -1
        logger.debug(f"n_state: {self.n_state}, n_state_rl: {self.n_state_rl}, rl_n_envs: {rl_n_envs}, n_action: {self.n_action}")
        logger.debug(f"Shape of _state: {self._state.shape}, _state_rl: {self._state_rl.shape}, _action: {self._action.shape}, _action_znmf: {self._action_znmf.shape}, " +
                     f"_local_reward: {self._local_reward.shape}, _reward: {self._reward.shape}")

        # define variables for state & reward standarization
        #self._state_running_mean     = 0.0     # TODO: remove if not used, should state be standarized?
        #self._state_running_var      = 1.0
        #self._state_running_counter  = 0
        #self._reward_running_mean    = 0.0     # TODO: remove if not used, should reward be standarized?
        #self._reward_running_var     = 1.0
        #self._reward_running_counter = 0
        # define variables for state & reward min-max-scaling
        self._state_running_min  = 1e5
        self._state_running_max  = 0.0
        self._reward_running_min = 1e5
        self._reward_running_max = 0.0
        
        # define initial state and action array properties. Omit batch dimension (shape is per (rl) environment)
        self._observation_spec = array_spec.ArraySpec(shape=(self.n_state_rl,), dtype=self.model_dtype, name="observation")
        self._action_spec = array_spec.BoundedArraySpec(shape=(self.n_action,), dtype=self.model_dtype, name="action",
            minimum=self.action_bounds[0], maximum=self.action_bounds[1])



    def stop(self):
        """
        Stops all RHEA instances inside launched in this environment.
        """
        if self.exp is not None: 
            try:
                self._stop_exp()
            except Exception as e:
                logger.error(f"Exception during environment stop: {e}")
        else:
            print("WARNING: Experiment not stoped because self.exp is None")
        logger.debug("All RHEA instances stopped!")
        # Additional files managing
        self._manage_temporal_time_probes()


    def start(self, new_ensamble=False, restart_file=0, global_step=0):
        """
        Starts all RHEA instances with configuration specified in initialization.
        """
        # Create Multi-Process Multi-Data (mpmd) ensemble
        if not self.ensemble or new_ensamble:
            self.ensemble = self._create_mpmd_ensemble(restart_file, global_step)
            logger.debug(f"New MPMD ensemble created")

        self.exp.start(self.ensemble, block=False) # non-blocking start of RHEA solver(s)
        logger.debug("Start experiment / RHEA solver (non-blocking)")
        self.envs_initialised = False

        # Check simulations have started
        status = self._get_status()
        logger.info(f"Initial status: {status}")
        assert np.all(status > 0), "RHEA environments could not start."
        self._episode_global_step = global_step

        # Assert that the same state and action sizes are captured equally in RHEA and here
        n_state = self._get_n_state()
        n_action = self._get_n_action()

        if n_state != self.n_state or n_action != self.n_action * self.rl_n_envs:
            raise ValueError(f"State or action size differs between RHEA and the Python environment: \
                \nRHEA n_state: {n_state}   \nPython env n_state: {self.n_state} \
                \nRHEA n_action: {n_action} \nPython env n_action * RL_envs: {self.n_action * self.rl_n_envs}")

        # Get the initial state, reward and time (poll database tensors)
        self._get_state()           # updates self._state
        self._get_reward()          # updates self._reward
        self._get_time()            # updates self._time

        # Transform state and reward: redistribute & standarize
        self._redistribute_state()       # updates self._state_rl
        #self._min_max_scaling_state()   # updates self._state_rl    
        #self._min_max_scaling_reward()  # updates self._reward

        # Write RL data into disk
        if self.dump_data_flag:
            self._dump_rl_data()


    def _create_mpmd_ensemble(self, restart_file, global_step):
        # TODO: add method description
        if restart_file == "random_choice":   # random choice of restart file
            random_num = random.choice(["3210000", "3320000", "3420000", "3520000", "3620000", "3720000", "3820000"])
            restart_step = [f"{self.rhea_case_path}/restart_data_file_{random_num}.h5" for _ in range(self.cfd_n_envs)]
        else:
            restart_step = [ f"{self.rhea_case_path}/{restart_file}" for _ in range(self.cfd_n_envs)]
        logger.info(f"Restart files used: {restart_step}")

        # set RHEA exe arguments
        rhea_args = {"configuration_file": self.configuration_file, 
                     "tag": self.tag,
                     "restart_step": restart_step, 
                     "t_action": self.t_action, 
                     "t_episode": self.t_episode, 
                     "t_begin_control": self.t_begin_control,
                     "db_clustered": self.db_clustered,
                     "global_step": [str(global_step) for _ in range(self.cfd_n_envs)],
        }
        
        # Edit my-hostfile
        hostfilepath = os.path.join(self.rhea_case_path, self.mpirun_hostfile)
        logger.warning(f"RHEA_CASE_PATH: {self.rhea_case_path}")
        write_hosts(self.hosts, self.mpirun_np, hostfile=hostfilepath)

        if self.run_command=="bash":
            # Generate the runit.sh script
            # TODO: currently taking restart_data_file from current directory, but it should be taken the one from $RHEA_CASE_PATH (execution working because restart_data_file input arg is not used in RHEA yet)
            runit_filepath = os.path.join(self.rl_case_path, 'runit.sh')
            with open(runit_filepath, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"echo 'RHEA executable directory: {self.rhea_case_path}'\n")
                for i in range(self.cfd_n_envs):
                    exe_args = " ".join([f"{v[i]}" for v in rhea_args.values()])
                    stdout_stderr_filepath = f"{self.dump_data_path}/mpi_output/mpi_output_ensemble{i}_step{global_step:06}.out"
                    f.write(f"mpirun -np {self.mpirun_np} --hostfile {self.rhea_case_path}/{self.mpirun_hostfile} --mca {self.mpirun_mca} {self.rhea_case_path}/{self.rhea_exe_fname} {exe_args} > {stdout_stderr_filepath} 2>&1 &\n")
                    f.write(f"pid{i}=$!\n")   # Capture process ID for each mpirun
                # Wait for all background processes to finish
                f.write("wait $pid0")  # Always wait for the first process
                if self.cfd_n_envs > 1:
                    for i in range(1, self.cfd_n_envs):
                        f.write(f" $pid{i}")  # Append wait for each additional process
                f.write("\n")
                # Print a message indicating completion
                f.write("echo 'All MPI processes have completed.'\n")
            # Make the script executable
            os.chmod(runit_filepath, 0o755)
            # Set up RunSettings
            f_mpmd = RunSettings(exe=runit_filepath, run_command=self.run_command)
            # Debug logging
            with open(runit_filepath, 'r') as f:
                logger.debug(f"Edited {runit_filepath} as: \n{f.read()}")
            # Batch settings
            batch_settings = None
        else:
            raise Warning(f"Could not create Multi-Process Multi-Data run settings, not recognized run_command '{self.run_command}'")

        """ Create model:
            create_model(name: str, 
                         run_settings: smartsim.settings.base.RunSettings, 
                         params: Dict[str, Any] | None = None, 
                         path: str | None = None, 
                         enable_key_prefixing: bool = False, 
                         batch_settings: smartsim.settings.base.BatchSettings | None = None)
            -> smartsim.entity.model.Model  """
        return self.exp.create_model("ensemble", f_mpmd, batch_settings=batch_settings)


    def _get_n_state(self):
        # TODO: add method description
        try:
            # poll_tensor(name: str, poll_frequency_ms: int, num_tries: int) → bool
            self.client.poll_tensor(self.state_size_key, self.poll_freq_ms, self.poll_n_tries)
            state_size = self.client.get_tensor(self.state_size_key)
            logger.debug(f"Read state_size: {state_size}, type: {state_size.dtype}")
        except Exception as exc:
            raise Warning(f"Could not read state size from key: {self.state_size_key}") from exc
        return state_size[0]


    def _get_n_action(self):
        # TODO: add method description
        try:
            # poll_tensor(name: str, poll_frequency_ms: int, num_tries: int) → bool
            self.client.poll_tensor(self.action_size_key, self.poll_freq_ms, self.poll_n_tries)
            action_size = self.client.get_tensor(self.action_size_key)
            logger.debug(f"Read action_size: {action_size}, type: {action_size.dtype}")
        except Exception as exc:
            raise Warning(f"Could not read action size from key: {self.action_size_key}") from exc
        return action_size[0]


    def _get_state(self):
        """
        Get current flow state from the database.
        """
        logger.debug("Reading state...")
        for i in range(self.cfd_n_envs):
            if self._step_type[i] > 0: # environment still running
                try:
                    self.client.poll_tensor(self.state_key[i], self.poll_freq_ms, self.poll_n_tries)
                    # self._state shape: [self.cfd_n_envs, self.n_state], where self.n_state = num. witness points in single cfd env
                    self._state[i, :]   = self.client.get_tensor(self.state_key[i])
                    self.__state[i,:,:] = self._state[i, :].reshape(self.n_state_wit, self.state_dim)
                    self.client.delete_tensor(self.state_key[i])
                    logger.debug(f"[Env {i}] (Read) State '{self.state_key[i]}': \n{self._state[i,:]}") # \n{self.__state[i,:,:]}")
                except Exception as exc:
                    raise Warning(f"Could not read state from key: {self.state_key[i]}") from exc


    def _redistribute_state(self):
        """
        Redistribute state across RL pseudo-environments.
        Make sure the witness points are written in such that the first moving coordinate is z, then x, and last y, as RL pseudo environmnents are distributed only along y-coordinate
        TODO: in SmartSOD2d they had: Make sure the witness points are written in such that the first moving coordinate is x, then y, and last z. 
              I do it differently, because i want my actuators & RL pseudo env. to be distributed along y-direction -> witness points first along z, then x, then y 
        -> check done in rhea_env.__init__ -> utils.get_witness_xyz -> utils.check_witness_xyz

        Additional info:
        state_extended array is used for accessing neighbouring blocks of witness points; each rl env has self.rl_neighbors neighboring envs 
        self._state_rl shape:  [self.n_envs,     self.n_state_rl],                      where self.n_envs = cfd_n_envs * rl_n_envs, 
        self.__state_rl shape: [self.n_envs,     self.n_state_wit_rl,  self.state_dim], where self.n_state_rl = n_state_wit_rl * self.state_dim 
        self._state shape:     [self.cfd_n_envs, self.n_state],                         where self.n_state = num. witness points in single cfd env
        self.__state shape:    [self.cfd_n_envs, self.n_state_wit,     self.state_dim], where self.n_state = self.n_state_wit * self.state_dim
        state_extended shape:  [self.cfd_n_nevs, self.n_state_wit * 3, self.state_dim]
        """
        # Concatenate self._state array 3 times along columns, used for building self._state_rl which include the state of neighbouring rl environments
        state_extended = np.concatenate((self.__state, self.__state, self.__state), axis=1)
        plane_wit = self.witness_xyz[0] * self.witness_xyz[2]                   # num. witness points in x-z plane
        block_wit = int(plane_wit * (self.witness_xyz[1] / self.rl_n_envs))     # rl_n_envs distributed along 2nd coordinate y
        assert self.witness_xyz[1] % self.rl_n_envs == 0, f"Number of witness points in the y-direction is not multiple to the number of rl environments, with self.witness_xyz[1] = {self.witness_xyz[1]} and self.rl_n_envs = {self.rl_n_envs}"
        for i in range(self.cfd_n_envs):
            for j in range(self.rl_n_envs):
                self.__state_rl[i * self.rl_n_envs + j,:,:] = state_extended[
                    i, 
                    block_wit * (j - self.rl_neighbors) + self.n_state_wit:block_wit * (j + self.rl_neighbors + 1) + self.n_state_wit,
                    :,
                ]
                self._state_rl[i * self.rl_n_envs + j,:] = self.__state_rl[i * self.rl_n_envs + j,:,:].reshape(self.n_state_rl)
                logger.debug(f"[Cfd Env {i} - Pseudo Env {j}] State: \n{self._state_rl[i * self.rl_n_envs + j,:]}") # \n{self.__state_rl[i * self.rl_n_envs + j,:,:]}")


    def _get_reward(self):
        logger.debug("Reading reward...")
        for i in range(self.cfd_n_envs):
            if self._step_type[i] > 0: # environment still running
                # poll_tensor(name: str, poll_frequency_ms: int, num_tries: int) → bool
                try:
                    self.client.poll_tensor(self.reward_key[i], self.poll_freq_ms, self.poll_n_tries)
                    reward = self.client.get_tensor(self.reward_key[i])
                    self.client.delete_tensor(self.reward_key[i])
                    local_reward = + reward / self.reward_norm
                    self._local_reward[i, :] = local_reward
                    global_reward = np.mean(local_reward)
                    for j in range(self.rl_n_envs):
                        # Weighted average between local rewards (of rl env inside same cfd simulation) and global reward (of cfd simulation)
                        self._reward[i * self.rl_n_envs + j] = self.reward_beta * global_reward + (1.0 - self.reward_beta) * local_reward[j]
                    logger.debug(f"[Cfd Env {i}] (Read) Local Reward '{self.reward_key[i]}': {reward}")  # shape [self.rl_n_envs], rewards from all rl env for a specific cfd env
                    logger.debug(f"[Cfd Env {i}] Normalized Local Reward: {local_reward}")
                    logger.debug(f"[Cfd Env {i}] Global Reward: {global_reward}")
                    logger.debug(f"[Cfd Env {i}] Weighted Reward: {self._reward[i*self.rl_n_envs:i*self.rl_n_envs+self.rl_n_envs]}")                  
                except Exception as exc:
                    raise Warning(f"Could not read reward from key: {self.reward_key[i]}") from exc


# TODO: remove if not used, and also remove only-used-here variables: self._state_running_mean, self._state_running_var, self._state_rl, self._state_running_counter
#    def _standarize_state(self):
#        """
#        Standarize 'self._state_rl' data to have mean = 0 and standard deviation = 1
#        
#        Additional info: 
#        self._state_rl shape: [self.n_envs, self.n_state_rl], where self.n_envs = cfd_n_envs * rl_n_envs, 
#                                                                    self.n_state_rl = int((2 * self.rl_neighbors + 1) * (self.n_state / self.rl_n_envs));
#
#        Updated attributes:
#        self._state_running_mean
#        self._state_running_var
#        self._state_rl
#        """
#        self._state_running_counter += 1
#        flattened_state_rl = self._state_rl.flatten()
#        # Update running mean
#        current_epoch_mean = np.mean(flattened_state_rl)
#        old_mean           = self._state_running_mean
#        self._state_running_mean += ( current_epoch_mean - old_mean ) / self._state_running_counter      # equivalent to: self._state_running_mean = ( self._state_running_mean * (self._state_running_counter - 1) + current_mean ) / self._state_running_counter
#        # Update running variance (Welford's method)
#        self._state_running_var += ( ( current_epoch_mean - old_mean ) * ( current_epoch_mean - self._state_running_mean) ) / self._state_running_counter
#        # Calculate running standart deviation
#        std_dev = np.sqrt( self._state_running_var / self._state_running_counter )
#        # Standarize state
#        self._state_rl = ( self._state_rl - self._state_running_mean ) / ( std_dev + EPS )
#        # Logging
#        logger.debug(f"[RheaEnv::_standarize_state] State Standarization, updated running mean: {self._state_running_mean}, variance: {self._state_running_var}, std_dev: {std_dev}")
#        for i in range(self.cfd_n_envs):
#            for j in range(self.rl_n_envs):
#                logger.debug(f"[Cfd Env {i} - Pseudo Env {j}] Standarized State: {self._state_rl[i * self.rl_n_envs + j,:]}")
#
#    # Min-max-scaling of state to range [0,1] 
#    def _min_max_scaling_state(self):
#        flattened_state_rl = self._state_rl.flatten()
#        min_value = np.min(flattened_state_rl)
#        max_value = np.max(flattened_state_rl)
#        self._state_running_min = np.min([min_value, self._state_running_min])
#        self._state_running_max = np.max([max_value, self._state_running_max])
#        self._state_rl = ( self._state_rl - self._state_running_min ) / ( self._state_running_max - self._state_running_min + EPS )
#        # Logging
#        logger.debug(f"[RheaEnv::_min_max_scaling_state] State Scaling, with input data (min, max) = ({min_value}, {max_value}) and running (min, max) = ({self._state_running_min}, {self._state_running_max})")
#        for i in range(self.cfd_n_envs):
#            for j in range(self.rl_n_envs):
#                logger.debug(f"[Cfd Env {i} - Pseudo Env {j}] Scaled State: {self._state_rl[i * self.rl_n_envs + j,:]}")
#
#
# TODO: remove if not used, and remove also only-used-here variables: self._reward_running_mean, self._reward_running_var, self._reward, self._reward_running_counter
#    def _standarize_reward(self):
#        """
#        Standarize 'self._reward' data to have mean = 0 and standard deviation = 1
#        
#        Additional info: 
#        self._reward shape: [self.n_envs], where self.n_envs = cfd_n_envs * rl_n_envs
#
#        Updated attributes:
#        self._reward_running_mean
#        self._reward_running_var
#        self._reward
#        """
#        self._reward_running_counter += 1
#        # Update running mean
#        current_epoch_mean = np.mean(self._reward)
#        old_mean           = self._reward_running_mean
#        self._reward_running_mean += ( current_epoch_mean - old_mean ) / self._reward_running_counter
#        # Update running variance (Welford's method)
#        self._reward_running_var += ( ( current_epoch_mean - old_mean ) * ( current_epoch_mean - self._reward_running_mean) ) / self._reward_running_counter
#        # Calculate running standart deviation
#        std_dev = np.sqrt( self._reward_running_var / self._reward_running_counter )
#        # Standarize reward
#        self._reward = ( self._reward - self._reward_running_mean ) / ( std_dev + EPS )
#        # Logging
#        logger.debug(f"[RheaEnv::_standarize_reward] Reward Standarization, updated running mean: {self._reward_running_mean}, variance: {self._reward_running_var}, std_dev: {std_dev}")
#        for i in range(self.cfd_n_envs):
#            for j in range(self.rl_n_envs):
#                logger.debug(f"[Cfd Env {i} - Pseudo Env {j}] Standarized Reward: {self._reward[i * self.rl_n_envs + j]}")
#    
#    # Min-Max scaling reward to range [0,1]
#    def _min_max_scaling_reward(self):
#        min_value = np.min(self._reward)
#        max_value = np.max(self._reward)
#        self._reward_running_min = np.min([min_value, self._reward_running_min])
#        self._reward_running_max = np.max([max_value, self._reward_running_max])
#        self._reward = ( self._reward - self._reward_running_min ) / ( self._reward_running_max - self._reward_running_min + EPS )
#        # Logging
#        logger.debug(f"[RheaEnv::_min_max_scaling_reward] Reward Scaling, with input data (min, max) = ({min_value}, {max_value}), running (min, max) = ({self._reward_running_min}, {self._reward_running_max})")
#        for i in range(self.cfd_n_envs):
#            for j in range(self.rl_n_envs):
#                logger.debug(f"[Cfd Env {i}] Scaled Reward: {self._reward[i*self.rl_n_envs:(i+1)*self.rl_n_envs]}")
#

    def _get_time(self):
        logger.debug("Reading time...")
        for i in range(self.cfd_n_envs):
            try:
                self.client.poll_tensor(self.time_key[i], self.poll_freq_ms, self.poll_n_tries)
                self._time[i] = self.client.get_tensor(self.time_key[i])[0]
                self.client.delete_tensor(self.time_key[i])
                logger.debug(f"[Cfd Env {i}] Got time: {self._time[i]:.8f}")
            except Exception as exc:
                raise Warning(f"Could not read time from key: {self.time_key[i]}") from exc


    def _get_status(self):
        """
        Reads the step_type tensors from database (one for each environment).
        Once created, these tensor are never deleted afterwards
        Types of step_type:
            0: Ended            (ts.StepType.LAST)
            1: Initialized      (ts.StepType.FIRST)
            2: Running          (ts.StepType.MID)
        Returns array with step_type from every environment.
        """
        logger.debug("Reading status...")
        if not self.envs_initialised: # initialising environments - poll and wait for them to get started
            for i in range(self.cfd_n_envs):
                try:
                    self.client.poll_tensor(self.step_type_key[i], self.poll_freq_ms, self.poll_n_tries)
                    self._step_type[i] = self.client.get_tensor(self.step_type_key[i])[0]
                except Exception as exc:
                    raise Warning(f"Could not read step type from key: {self.step_type_key[i]}.") from exc
            if np.any(self._step_type < 0):
                raise ValueError(f"Environments could not be initialized, or initial step_type could not be read from database. \
                    \n step_type = {self._step_type}")
            self.envs_initialised = True
        else:
            for i in range(self.cfd_n_envs):
                try:
                    self.client.poll_tensor(self.step_type_key[i], self.poll_freq_ms, self.poll_n_tries)
                    self._step_type[i] = self.client.get_tensor(self.step_type_key[i])[0]
                except Exception as exc:
                    raise Warning(f"Could not read step type from key: {self.step_type_key[i]}") from exc
        # Logging
        for i in range(self.cfd_n_envs):
            logger.debug(f"[Env {i}] (Read) Status: {self._step_type[i]}")
        return self._step_type


    def _set_action(self, action):
        """
        Write actions for each environment to be polled by the corresponding RHEA environment.
        Action clipping must be performed within the environment: https://github.com/tensorflow/agents/issues/216
        
        Additional info: 
        - action: shape [cfd_n_envs * n_rl_envs, action_dim]
        - self._action shape [cfd_n_envs, n_rl_envs * action_dim]
        - single_action: shape [self.action_dim]    (auxiliary vector for code clarity)
        """
        # scale actions and reshape for RHEA
        # TODO: is this scaling and clipping correct, if action_bounds[0],[1] are not +-k?
        logger.debug(f"[RheaEnv] action shape: {action.shape}")
        logger.debug(f"[RheaEnv] self._action shape: {self._action.shape}")
        action_aux = np.zeros([self.cfd_n_envs, self.rl_n_envs, self.action_dim])
        action = action * self.action_bounds[1] if self.mode == "collect" else action # TODO: check! shouldn't actions be always scaled, ALSO in testing/training?
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        logger.debug(f"[RheEnv] action: \n{action}")
        for i in range(self.cfd_n_envs):
            for j in range(self.rl_n_envs):
                single_action = action[i * self.rl_n_envs + j, :]
                self._action[i, j*self.action_dim:(j+1)*self.action_dim] = single_action
        # write action into database
        for i in range(self.cfd_n_envs):
            # self._action shape: np.zeros(self.n_action, dtype=self.rhea_dtype))
            self.client.put_tensor(self.action_key[i], self._action[i, ...].astype(self.rhea_dtype)) # "..." is a shorthand for 'all remaining directions'
            logger.debug(f"[RheaEnv] [Env {i}] (Written) Action: \n{self._action[i, :]}")


    def _dump_rl_data(self):
        """
        Write RL data into disk.
        """
        for i in range(self.cfd_n_envs):
            with open(os.path.join(self.dump_data_path , "state", f"state_ensemble{i}_step{self._episode_global_step:06}.txt"),'a') as f:
                np.savetxt(f, self._state[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
            f.close()
            with open(os.path.join(self.dump_data_path , "local_reward", f"local_reward_ensemble{i}_step{self._episode_global_step:06}.txt"),'a') as f:
                np.savetxt(f, self._local_reward[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
            f.close()
            with open(os.path.join(self.dump_data_path , "reward", f"reward_ensemble{i}_step{self._episode_global_step:06}.txt"),'a') as f:
                np.savetxt(f, self._reward[i*self.rl_n_envs:(i+1)*self.rl_n_envs][np.newaxis], fmt='%.4f', delimiter=' ')
            f.close()
            with open(os.path.join(self.dump_data_path , "action", f"action_ensemble{i}_step{self._episode_global_step:06}.txt"),'a') as f:
                np.savetxt(f, self._action[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
            f.close()
            with open(os.path.join(self.dump_data_path , "time", f"time_ensemble{i}_step{self._episode_global_step:06}.txt"),'a') as f:
                np.savetxt(f, self._time[i][np.newaxis], fmt='%.6f', delimiter=' ')
            f.close()


    def _stop_exp(self):
        """
        Stop RHEA experiment (ensemble of models) with SmartSim
        Safely handles potential exceptions during the cleanup process.
        """
        if self.client is not None:
            try:
                self.client.delete_tensor(self.state_size_key)
                self.client.delete_tensor(self.action_size_key)
                for i in range(self.cfd_n_envs):
                    self.client.delete_tensor(self.step_type_key[i])
            except Exception as e:
                logger.error(f"Exception while deleting tensors: {e}")
        else:
            print("WARNING: Client is None, skipping tensor deletion.")

        if self.exp is not None and self.ensemble is not None:
            try:
                self.exp.stop(self.ensemble)
            except TypeError as e:
                logger.error(f"TypeError in stopping ensemble: {e}")
            except Exception as e:
                logger.error(f"Unexpected exception while stopping ensemble: {e}")
        else:
            print("WARNING: Experiment or ensemble is None, skipping stopping ensemble.")
        

    def __del__(self):
        """
        Properly finalize all launched RHEA instances within the SmartSim experiment.
        """
        try:
            self.stop()
        except Exception as e:
            logger.error(f"Exception in environment __del__: {e}")


    def _manage_temporal_time_probes(self):
        """
        If generated by the solver, move and rename temporal time probes data files
        from directory self.rl_case_path to self.self.temporal_time_probes_dir, 
        and rename files with additional global_step information
        """
        # Find temporal time probes files, matching the pattern temporal_point_probes*.csv
        matching_pattern = os.path.join(self.rl_case_path, "temporal_point_probe_*.csv")
        for src_path in glob.glob(matching_pattern):
            filename = os.path.basename(src_path)
            basename, ext = os.path.splitext(filename)
            # Move and rename file (adding global_step information)
            new_filename = f"{basename}_step{self._episode_global_step:06}{ext}"
            dest_path = os.path.join(self.temporal_time_probes_dir, new_filename)
            shutil.move(src_path, dest_path)
        logger.debug(f"Temporal time probes moved from '{self.rl_case_path}' to '{self.temporal_time_probes_dir}' for global_step {self._episode_global_step}")


    ### PyEnvironment methods override

    # https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/trajectories/TimeStep
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/trajectories/restart

    # We need to override the following methods
    #     - _reset(self)
    #     - _step(self)
    #     - action_spec(self)
    #     - observation_spec(self)
    #     - batched
    #     - batch_size

    def _reset(self):
        """
        Returns: 
            ts.TimeStep with step_type = ts.StepType.FIRST
        """
        self._episode_ended = False
        """ tf_agents.trajectories.restart(
                observation: tf_agents.typing.types.NestedTensorOrArray,
                batch_size: Optional[types.Int] = None,
                reward_spec: Optional[types.NestedSpec] = None
            ) -> tf_agents.trajectories.TimeStep """
        return ts.restart(np.array(self._state_rl, dtype=self.model_dtype), batch_size = self.n_envs)
    
    
    def _step(self, action):
        """
        Performs a step (single or multiple dts) in RHEA environments.
        Here we apply an action and return the new time_step
        
        Returns:
            ts.TimeStep with step_type = ts.StepType.FIRST, if return self.reset() called (-> ts.restart called in self._reset())
                                       = ts.StepType.MID,   if return ts.transition(...) called
                                       = ts.StepType.LAST,  if return ts.termination(...) called 
        """
        # the last action ended the episode. Ignore the current actions, relaunch the environments, and start a new episode
        if self._episode_ended:
            logger.info("Last action ended the episode. Relaunching the environments and starting a new episode")
            return self.reset()

        # send predicted actions into DB for RHEA to collect and step
        self._set_action(action)

        # Poll new state, reward and time
        # poll command waits for the RHEA to finish the action period and send the state and reward data
        self._get_state()                # updates self._state
        self._get_reward()               # updates self._reward
        self._get_time()                 # updates self._time

        # Pre-processing state & reward: redistribute & standarize
        self._redistribute_state()       # updates self._state_rl
        #self._min_max_scaling_state()   # updates self._state_rl    
        #self._min_max_scaling_reward()  # updates self._reward

        # Write RL data into disk
        if self.dump_data_flag: self._dump_rl_data()

        # determine if simulation finished
        status = self._get_status()
        self._episode_ended = np.all(status == 0) # update self._episode_ended

        # Return transition if episode ended with current action
        if self._episode_ended:
            logger.debug(f"Episode ended. status = {status}")
            """ tf_agents.trajectories.termination(
                    observation: tf_agents.typing.types.NestedTensorOrArray,
                    reward: tf_agents.typing.types.NestedTensorOrArray,
                    outer_dims: Optional[types.Shape] = None
                ) -> tf_agents.trajectories.TimeStep    """
            return ts.termination(np.array(self._state_rl, dtype=self.model_dtype), self._reward)
        # else discount is later multiplied with global discount from user input
        """ tf_agents.trajectories.transition(
                observation: tf_agents.typing.types.NestedTensorOrArray,
                reward: tf_agents.typing.types.NestedTensorOrArray,
                discount: tf_agents.typing.types.Float = 1.0,
                outer_dims: Optional[types.Shape] = None
            ) -> tf_agents.trajectories.TimeStep    """
        return ts.transition(np.array(self._state_rl, dtype=self.model_dtype), self._reward, discount=np.ones((self.n_envs,)))

    
    def observation_spec(self):
        return self._observation_spec


    def action_spec(self):
        return self._action_spec    
    

    @property
    def batched(self):
        """
        Multi-environment flag. Override batched property to indicate that this environment is batched.
        Even if n_envs=1, we run it as a muti-environment (for consistency).
        """
        return True


    @property
    def batch_size(self):
        """
        Override batch size property according to chosen batch size
        """
        return self.n_envs

    ### End PyEnvironment methods override