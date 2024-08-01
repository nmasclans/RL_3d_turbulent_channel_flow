import os
import glob
import random
import numpy as np

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

from smartredis import Client
from smartsim.settings import MpirunSettings
from smartsim.settings.settings import create_batch_settings
from smartsim.log import get_logger
from smartrhea.utils import n_witness_points, n_rectangles, numpy_str, bcolors

logger = get_logger(__name__)

class RheaEnv(py_environment.PyEnvironment):
    """
    RHEA environment(s) extending the tf_agents.environments.PyEnvironment class.
    Fields:
    - exp: SmartSim experiment class instance
    - db: SmartSim orchestrator class instance
    - hosts: list of host nodes
    - rhea_exe: RHEA executable
    - rhea_dir: directory of RHEA executable
    - cwd: working directory
    ### **env_params
    - launcher: "local", other
    - run_command: "mpirun" ("srun" still not working)
    - cluster_account: project account, if required for specific launcher
    - modules_sh: modules file, if required for specific launcher
    - episode_walltime: environment walltime, if required for specific launcher
    - cfd_n_envs: number of cdf simulations
    - rl_n_envs: number of rl environments inside a cfd simulation
    - n_tasks_per_env: number of processes (MPI ranks) per environment
    - rectangle_file: actuators surfaces file
    - witness_file: witness points file
    - witness_xyz: number of witness points in the (x, y, x) directions
    - rl_neighbors: number of witness blocks selected to compose the state
    - model_dtype: data type for the model
    - rhea_dtype: data type for arrays to be sent to RHEA (actions)
    - poll_time: time waiting for an array to appear in the database (in seconds)
    - f_action: action frequency ("how often we send a new action signal)
    - t_episode: episode elapsed time
    - t_begin_control: time to start control
    - action_bounds: bounds for action values
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
    """

    def __init__( # pylint: disable=super-init-not-called
        self,
        exp,
        db,
        hosts,
        rhea_exe,
        rhea_dir,
        cwd,
        ### **env_params:
        launcher = "local",
        run_command = "mpirun",
        cluster_account = None,
        modules_sh = None,
        episode_walltime = None,
        cfd_n_envs = 2,
        rl_n_envs = 5,
        n_tasks_per_env = 1,
        rectangle_file = "rectangleControl.txt",
        witness_file = "witness.txt",
        witness_xyz = (6, 4, 10),
        rl_neighbors = 1,
        model_dtype = np.float32,
        rhea_dtype = np.float64,
        poll_time = 3600, # in seconds, 2 minutes
        f_action = 1.0,
        t_episode = 10.0,
        t_begin_control = 0.0,
        action_bounds = (-0.05, 0.05),
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
        mode = "collect",           # TODO: consider using params['mode'], which is currently train/eval
    ):
        # Store input parameters
        self.exp = exp
        self.db = db
        self.rhea_exe = os.path.join(rhea_dir, rhea_exe)
        self.cwd = cwd
        # **env_params:
        self.launcher = launcher
        self.run_command = run_command
        self.cluster_account = cluster_account
        self.modules_sh = modules_sh
        self.episode_walltime = episode_walltime
        self.cfd_n_envs = cfd_n_envs
        self.rl_n_envs = rl_n_envs
        self.n_tasks_per_env = n_tasks_per_env
        self.rectangle_file = rectangle_file
        self.witness_file = witness_file
        self.witness_xyz = witness_xyz
        self.rl_neighbors = rl_neighbors
        self.model_dtype = model_dtype
        self.rhea_dtype = rhea_dtype
        self.poll_time = poll_time
        self.action_bounds = action_bounds
        self.reward_norm = reward_norm
        self.reward_beta = reward_beta
        self.dump_data_flag = dump_data_flag
        self.mode = mode

        # calculated parameters
        self.n_envs = cfd_n_envs * rl_n_envs
        self.dump_data_path = os.path.join(self.cwd, "train") if mode == "collect" else os.path.join(self.cwd, "eval")
        
        # preliminary checks
        if (2 * rl_neighbors + 1 > rl_n_envs):
            raise ValueError(f"Number of witness blocks selected to compose the state exceed the number total witness blocks:\n \
                Witness blocks selected: {2 * rl_neighbors + 1}\n \
                Total witness blocks: {rl_n_envs}\n")

        # manage directories
        if self.mode == "eval" and os.path.exists(self.dump_data_path):
            counter = 0
            path = self.dump_data_path + f"_{counter}"
            while os.path.exists(path):
                counter += 1
                path = self.dump_data_path + f"_{counter}"
            os.rename(self.dump_data_path, path)
            logger.info(f"{bcolors.WARNING}The data path `{self.dump_data_path}` exists. Moving it to `{path}`{bcolors.ENDC}")
        if self.dump_data_flag:
            if not os.path.exists(os.path.join(self.dump_data_path, "state")):
                os.makedirs(os.path.join(self.dump_data_path, "state"))
            if not os.path.exists(os.path.join(self.dump_data_path, "reward")):
                os.makedirs(os.path.join(self.dump_data_path, "reward"))
            if not os.path.exists(os.path.join(self.dump_data_path, "action")):
                os.makedirs(os.path.join(self.dump_data_path, "action"))

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
        self.tag = [str(i) for i in range(self.cfd_n_envs)] # environment tags [0, 1, ..., cfd_n_envs - 1]
        self.f_action = [str(f_action) for _ in range(self.cfd_n_envs)]
        self.t_episode = [str(t_episode) for _ in range(self.cfd_n_envs)]
        self.t_begin_control = [str(t_begin_control) for _ in range(self.cfd_n_envs)]

        # create RHEA ensemble models inside experiment
        self.ensemble = None
        self._episode_ended = False
        self.envs_initialised = False

        # arrays with known shapes
        self._time = np.zeros(self.cfd_n_envs, dtype=self.model_dtype)
        self._step_type = -np.ones(self.cfd_n_envs, dtype=int) # init status in -1

        # create and allocate array objects
        self.n_state = n_witness_points(os.path.join(self.cwd, self.witness_file))
        self.n_state_rl = int((2 * self.rl_neighbors + 1) * (self.n_state / self.rl_n_envs))
        if self.rl_n_envs > 1:
            self.n_action = 1
        else:   # self.rl_n_envs == 1:
            n_action = n_rectangles(os.path.join(self.cwd, self.rectangle_file))
            # TODO: remove commented lines of original code,  i think this line (and assert check) is not applicable for my Fpert actions
            #assert np.mod(n_action, 2) == 0, "Number of actions is not divisible by 2, so zero-net-mass-flux \
            #    strategy cannot be implemented"
            #self.n_action = n_action // 2   
        self._state = np.zeros((self.cfd_n_envs, self.n_state), dtype=self.model_dtype)
        self._state_rl = np.zeros((self.n_envs, self.n_state_rl), dtype=self.model_dtype)
        self._action = np.zeros((self.cfd_n_envs, self.n_action * self.rl_n_envs), dtype=self.rhea_dtype)
        self._action_znmf = np.zeros((self.cfd_n_envs, 2 * self.n_action * self.rl_n_envs), dtype=self.rhea_dtype)  # TODO: what is this?
        self._local_reward = np.zeros((self.cfd_n_envs, self.rl_n_envs))
        self._reward = np.zeros(self.n_envs)
        self._episode_global_step= -1

        # define initial state and action array properties. Omit batch dimension (shape is per (rl) environment)
        self._observation_spec = array_spec.ArraySpec(shape=(self.n_state_rl,), dtype=self.model_dtype, name="observation")
        self._action_spec = array_spec.BoundedArraySpec(shape=(self.n_action,), dtype=self.model_dtype, name="action",
            minimum=self.action_bounds[0], maximum=self.action_bounds[1])



    # def stop(self):
    #     """
    #     Stops all SOD2D instances inside launched in this environment.
    #     """
    #     if self.exp: self._stop_exp()


    # def start(self, new_ensamble=False, restart_file=0, global_step=0):
    #     """
    #     Starts all SOD2D instances with configuration specified in initialization.
    #     """
    #     if not self.ensemble or new_ensamble:
    #         self.ensemble = self._create_mpmd_ensemble(restart_file)
    #         logger.info(f"New ensamble created")

    #     self.exp.start(self.ensemble, block=False) # non-blocking start of Sod2D solver(s)
    #     self.envs_initialised = False

    #     # Check simulations have started
    #     status = self.get_status()
    #     logger.info(f"Initial status: {status}")
    #     assert np.all(status > 0), "SOD2D environments could not start."
    #     self._episode_global_step = global_step

    #     # Assert that the same state and action sizes are captured equally in Sod2D and here
    #     n_state = self._get_n_state()
    #     n_action = self._get_n_action()

    #     if n_state != self.n_state or n_action != self.n_action * 2 * self.marl_n_envs:
    #         raise ValueError(f"State or action size differs between SOD2D and the Python environment: \n \
    #             SOD2D n_state: {n_state}\n Python env n_state: {self.n_state}\n \
    #             SOD2D n_action: {n_action}\n Python env n_action * 2 * MARL_envs: {self.n_action * 2 * self.marl_n_envs}")

    #     # Get the initial state and reward
    #     self._get_state() # updates self._state
    #     self._redistribute_state() # updates self._state_marl
    #     self._get_reward() # updates self._reward

    #     # Write RL data into disk
    #     if self.dump_data_flag:
    #         self._dump_rl_data()


    # def _create_mpmd_ensemble(self, restart_file):
    #     # restart files are copied within SOD2D to the output_* folder
    #     if restart_file == 3:
    #         restart_step = [random.choice(["1", "2"]) for _ in range(self.cfd_n_envs)]
    #     else:
    #         restart_step = [str(restart_file) for _ in range(self.cfd_n_envs)]

    #     # set SOD2D exe arguments
    #     sod_args = {"restart_step": restart_step, "f_action": self.f_action,
    #         "t_episode": self.t_episode, "t_begin_control": self.t_begin_control}

    #     for i in range(self.cfd_n_envs):
    #         exe_args = [f"--{k}={v[i]}" for k,v in sod_args.items()]
    #         run = MpirunSettings(exe=self.sod_exe, exe_args=exe_args)
    #         run.set_tasks(self.n_tasks_per_env)
    #         if i == 0:
    #             f_mpmd = run
    #         else:
    #             f_mpmd.make_mpmd(run)

    #     batch_settings = None

    #     # Alvis configuration if requested.
    #     #  - SOD2D simulations will be launched as external Slurm jobs.
    #     #  - there are 4xA100 GPUs per node.
    #     if self.launcher == "alvis":
    #         gpus_required = self.cfd_n_envs * self.n_tasks_per_env
    #         n_nodes = int(np.ceil(gpus_required / 4))
    #         gpus_per_node = '4' if gpus_required >= 4 else gpus_required
    #         batch_settings = create_batch_settings("slurm", nodes=n_nodes, account=self.cluster_account, time=self.episode_walltime,
    #                 batch_args={'ntasks':gpus_required, 'gpus-per-node':'A100:' + str(gpus_per_node)})
    #         batch_settings.add_preamble([". " + self.modules_sh])

    #     return self.exp.create_model("ensemble", f_mpmd, batch_settings=batch_settings)


    # def _get_n_state(self):
    #     try:
    #         self.client.poll_tensor(self.state_size_key, 100, self.poll_time) # poll every 100ms for 1000 times (100s in total)
    #         state_size = self.client.get_tensor(self.state_size_key)
    #     except Exception as exc:
    #         raise Warning(f"Could not read state size from key: {self.state_size_key}") from exc
    #     logger.debug(f"Read state_size: {state_size}")
    #     return state_size[0]


    # def _get_n_action(self):
    #     try:
    #         self.client.poll_tensor(self.action_size_key, 100, self.poll_time)
    #         action_size = self.client.get_tensor(self.action_size_key)
    #         logger.debug(f"Read action_size: {action_size}")
    #     except Exception as exc:
    #         raise Warning(f"Could not read action size from key: {self.action_size_key}") from exc
    #     return action_size[0]


    # def _get_state(self):
    #     """
    #     Get current flow state from the database.
    #     """
    #     for i in range(self.cfd_n_envs):
    #         if self._step_type[i] > 0: # environment still running
    #             self.client.poll_tensor(self.state_key[i], 100, self.poll_time)
    #             try:
    #                 self._state[i, :] = self.client.get_tensor(self.state_key[i])
    #                 self.client.delete_tensor(self.state_key[i])
    #                 logger.debug(f"[Env {i}] Got state: {numpy_str(self._state[i, :5])}")
    #             except Exception as exc:
    #                 raise Warning(f"Could not read state from key: {self.state_key[i]}") from exc


    # def _redistribute_state(self):
    #     """
    #     Redistribute state across MARL pseudo-environments.
    #     Make sure the witness points are written in such that the first moving coordinate is x, then y, and last z.
    #     """
    #     state_extended = np.concatenate((self._state, self._state, self._state), axis=1)
    #     plane_wit = self.witness_xyz[0] * self.witness_xyz[1]
    #     block_wit = int(plane_wit * (self.witness_xyz[2] / self.marl_n_envs))
    #     for i in range(self.cfd_n_envs):
    #         for j in range(self.marl_n_envs):
    #             self._state_marl[i * self.marl_n_envs + j,:] = state_extended[i, block_wit * (j - self.marl_neighbors) + \
    #                 self.n_state:block_wit * (j + self.marl_neighbors + 1) + self.n_state]


    # def _get_reward(self):
    #     for i in range(self.cfd_n_envs):
    #         if self._step_type[i] > 0: # environment still running
    #             self.client.poll_tensor(self.reward_key[i], 100, self.poll_time)
    #             try:
    #                 reward = self.client.get_tensor(self.reward_key[i])
    #                 self.client.delete_tensor(self.reward_key[i])
    #                 logger.debug(f"[Env {i}] Got reward: {numpy_str(reward)}")

    #                 local_reward = -reward / self.reward_norm
    #                 self._local_reward[i, :] = local_reward
    #                 global_reward = np.mean(local_reward)
    #                 for j in range(self.marl_n_envs):
    #                     self._reward[i * self.marl_n_envs + j] = self.reward_beta * global_reward + (1.0 - self.reward_beta) * local_reward[j]
    #             except Exception as exc:
    #                 raise Warning(f"Could not read reward from key: {self.reward_key[i]}") from exc


    # def _get_time(self):
    #     for i in range(self.cfd_n_envs):
    #         self.client.poll_tensor(self.time_key[i], 100, self.poll_time)
    #         try:
    #             self._time[i] = self.client.get_tensor(self.time_key[i])[0]
    #             self.client.delete_tensor(self.time_key[i])
    #             logger.debug(f"[Env {i}] Got time: {numpy_str(self._time[i])}")
    #         except Exception as exc:
    #             raise Warning(f"Could not read time from key: {self.time_key[i]}") from exc


    # def get_status(self):
    #     """
    #     Reads the step_type tensors from database (one for each environment).
    #     Once created, these tensor are never deleted afterwards
    #     Types of step_type:
    #         0: Ended
    #         1: Initialized
    #         2: Running
    #     Returns array with step_type from every environment.
    #     """
    #     if not self.envs_initialised: # initialising environments - poll and wait for them to get started
    #         for i in range(self.cfd_n_envs):
    #             self.client.poll_tensor(self.step_type_key[i], 100, self.poll_time)
    #             try:
    #                 self._step_type[i] = self.client.get_tensor(self.step_type_key[i])[0]
    #             except Exception as exc:
    #                 raise Warning(f"Could not read step type from key: {self.step_type_key[i]}.") from exc
    #         if np.any(self._step_type < 0):
    #             raise ValueError(f"Environments could not be initialized, or initial step_type could not be read from database. \
    #                 \n step_type = {self._step_type}")
    #         self.envs_initialised = True
    #     else:
    #         for i in range(self.cfd_n_envs):
    #             try:
    #                 self._step_type[i] = self.client.get_tensor(self.step_type_key[i])[0]
    #             except Exception as exc:
    #                 raise Warning(f"Could not read step type from key: {self.step_type_key[i]}") from exc
    #     return self._step_type


    # def _set_action(self, action):
    #     """
    #     Write actions for each environment to be polled by the corresponding Sod2D environment.
    #     Action clipping must be performed within the environment: https://github.com/tensorflow/agents/issues/216
    #     """
    #     # scale actions and reshape for SOD2D
    #     action = action * self.action_bounds[1] if self.mode == "collect" else action
    #     action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
    #     for i in range(self.cfd_n_envs):
    #         for j in range(self.marl_n_envs):
    #             self._action[i, j] = action[i * self.marl_n_envs + j]
    #     # apply zero-net-mass-flow strategy
    #     self._action_znmf = np.repeat(self._action, 2, axis=-1)
    #     self._action_znmf[..., 1::2] *= -1.0
    #     # write action into database
    #     for i in range(self.cfd_n_envs):
    #         self.client.put_tensor(self.action_key[i], self._action_znmf[i, ...].astype(self.sod_dtype))
    #             # np.zeros(self.n_action * 2, dtype=self.sod_dtype))
    #         logger.debug(f"[Env {i}] Writing (half) action: {numpy_str(self._action[i, :])}")


    # def _dump_rl_data(self):
    #     """Write RL data into disk."""
    #     for i in range(self.cfd_n_envs):
    #         with open(os.path.join(self.dump_data_path , "state", f"state_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
    #             np.savetxt(f, self._state[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
    #         f.close()
    #         with open(os.path.join(self.dump_data_path , "reward", f"local_reward_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
    #             np.savetxt(f, self._local_reward[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
    #         f.close()
    #         with open(os.path.join(self.dump_data_path , "action", f"action_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
    #             np.savetxt(f, self._action[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
    #         f.close()


    # def _stop_exp(self):
    #     """
    #     Stop SOD2D experiment (ensemble of models) with SmartSim
    #     """
    #     # delete data from database
    #     self.client.delete_tensor(self.state_size_key)
    #     self.client.delete_tensor(self.action_size_key)
    #     for i in range(self.cfd_n_envs):
    #         self.client.delete_tensor(self.step_type_key[i])
    #     # stop ensemble run
    #     self.exp.stop(self.ensemble)


    # def __del__(self):
    #     """
    #     Properly finalize all launched SOD2D instances within the SmartSim experiment.
    #     """
    #     self.stop()


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

        # poll new state and reward. This waits for the RHEA to finish the action period and send the state and reward data
        self._get_state() # updates self._state
        self._redistribute_state() # updates self._state_rl
        self._get_reward() # updates self._reward

        # write RL data into disk
        if self.dump_data_flag: self._dump_rl_data()

        # determine if simulation finished
        status = self.get_status()
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