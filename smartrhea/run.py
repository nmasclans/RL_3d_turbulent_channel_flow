import absl.logging
import contextlib
import numpy as np
import os
import random
import tensorflow as tf
import time
import datetime
import uuid
import sys

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import spec_utils, strategy_utils
from tf_agents.agents.ppo import ppo_actor_network, ppo_clip_agent
from tf_agents.networks import value_network
from tf_agents.eval import metric_utils
from tf_agents.policies import policy_saver

from smartsim.log import get_logger
from socket import gethostname

sys.path.append(os.environ['RL_CASE_PATH'])
from params import params, env_params  # params.py in directory $RL_CASE_DIR
from smartrhea.history import History
from smartrhea.init_smartsim import init_smartsim
from smartrhea.utils import print_params, deactivate_tf_gpus, numpy_str, params_str, params_html_table, bcolors
from smartrhea.rhea_env import RheaEnv
#from smartrhea.networks import CustomPPOActorNetwork #TODO: remove if not used

#--------------------------- Utils ---------------------------

absl.logging.set_verbosity(os.environ.get("SMARTSIM_LOG_LEVEL"))
logger = get_logger(__name__)
deactivate_tf_gpus()    # deactivate TF for GPUs
if params["use_XLA"]:   # activate XLA (Accelerated Linear Algebra) for performance improvement
    os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"  # Enable TF to use XLA JIT (Just-In-Time) compilation more aggressively
    #   --tf_xla_auto_jit=2: enables TensorFlow to use JIT compilation for all operations 
    #   --tf_xla_cpu_global_jit: enables Tensorflow to use JIT compilation globally for CPU devices
    os.environ['XLA_FLAGS'] = "--xla_hlo_profile"   # Enables profiling XLA HLO (High-Level Optimizer) operations
    tf.config.optimizer.set_jit(True)               # Enables JIT compilation globally within Tensorflow for the current session
    tf.function(jit_compile=True)                   # Enables JIT compilation for specific Tensorflow functions

""" Write summary to TensorBoard: 
    creates ./train directory, if necessary
    creates file events.out.tfevents.<timestamp>.<host_name>.<pid>, where
        <timestamp>: time when the file is created
        <host_name>: host name of the machine where TensorFlow process runs, e.g. triton
        <pid>: pid of the TensorFlow process that created the event file    """
if params["run_id"] == "":
    run_id              = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")+"--"+str(uuid.uuid4())[:4]
else:
    run_id = params["run_id"]
    logger.info(f"Continue training from '{run_id}' past training")
train_parent_dir    = os.path.join(params["rl_case_path"], "train")
train_dir           = os.path.join(train_parent_dir, f"train_{run_id}")
summary_writter_dir = os.path.join(train_parent_dir, "summary", run_id)
summary_writer      = tf.summary.create_file_writer(summary_writter_dir, flush_millis=1000)
summary_writer.set_as_default()
print("")
logger.info(f"Tensorboard event file created: {summary_writter_dir}/events.out.tfevents.{int(time.time())}.{gethostname()}.{os.getpid()}")

# Print simulation params
print_params(params, "RUN PARAMETERS:")

#--------------------------- RL setup ---------------------------
#### Init SmartSim framework: Experiment and Orchestrator (database)
exp, hosts, db, db_is_clustered = init_smartsim(
    port = params["port"],
    network_interface = params["network_interface"],
    launcher = params["launcher"],
    run_command = params["run_command"],
)

### Environment
collect_py_env = RheaEnv(
    exp,
    db,
    hosts,
    dump_data_path = train_dir,
    mode = "collect",
    db_is_clustered = db_is_clustered,
    **env_params,
)
""" tf_agents.environments.TFPyEnvironment(
        environment: tf_agents.environments.PyEnvironment,
        check_dims: bool = False,
        isolation: bool = False
    )   """
collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)
""" get_or_create_global_step(graph=None)
    Returns and create (if necessary) the global step tensor.    
    Args:
      graph: The graph in which to create the global step tensor. If missing, use default graph.
    Returns:
      The global step tensor.
"""

### Global step, num. training steps performed
# Use tf.compat.v1.train.get_or_create_global_step (TensorFlow 1.x API) or tf.Varible (TensorFlow 2.x API)
# global_step = tf.compat.v1.train.get_or_create_global_step()
global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

### Tensor specifications
""" get_tensor_specs(env)
        Returns observation, action and time step TensorSpecs from passed env.        
        Args:
          env: environment instance used for collection.    """
observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env)
)
""" Example logging output:
    Observation Spec:
    TensorSpec(shape=(216,), dtype=tf.float32, name='observation')

    Action Spec:
    BoundedTensorSpec(shape=(1,), dtype=tf.float32, name='action', minimum=array(-0.3, dtype=float32), maximum=array(0.3, dtype=float32))

    Time Spec:
    TimeStep(
    {'discount': BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount', minimum=array(0., dtype=float32), maximum=array(1., dtype=float32)),
    'observation': TensorSpec(shape=(216,), dtype=tf.float32, name='observation'),
    'reward': TensorSpec(shape=(), dtype=tf.float32, name='reward'),
    'step_type': TensorSpec(shape=(), dtype=tf.int32, name='step_type')})
"""
logger.debug(f'Observation Spec:\n{observation_tensor_spec}')
logger.debug(f'Action Spec:\n{action_tensor_spec}')
logger.debug(f'Time Spec:\n{time_step_tensor_spec}')

### Action and Value Networks
""" Actor Network:
    ppo_actor_network.PPOActorNetwork.__init__(self, seed_stream_class=<class 'tensorflow_probability.python.util.seed_stream.SeedStream'>)
    PPOActorNetwork.create_sequential_actor_net(self, fc_layer_units, action_tensor_spec, seed=None)
"""
actor_net_builder = ppo_actor_network.PPOActorNetwork()         # TODO: Check ActorDistributionRnnNetwork otherwise
actor_net = actor_net_builder.create_sequential_actor_net(
    params["net"],
    action_tensor_spec,
)
# TODO: remove if not used
# actor_net_builder = CustomPPOActorNetwork()
# actor_net = actor_net_builder.create_sequential_actor_net(
#     fc_layer_units=params["net"],
#     action_tensor_spec=action_tensor_spec, 
#     activation_fn=params["actor_net_activation_fn"],
#     l2_reg_value=params["actor_net_l2_reg"],
#     std_init_value=params["actor_net_std_init"],
#)
""" Value Network:    
    value_network.ValueNetwork.__init__(self, input_tensor_spec, preprocessing_layers=None, 
        preprocessing_combiner=None, conv_layer_params=None, fc_layer_params=(75, 40), 
        dropout_layer_params=None, activation_fn=<function relu at 0x79f78275c280>, 
        kernel_initializer=None, batch_squash=True, dtype=tf.float32, name='ValueNetwork')
    Generatres actor net as:
    - Several dense layers (based on 'fc_layer_units') with tanh activation.
    - A 'means projection' layer that outputs the action means ('loc').
    - A 'Lambda' layer that initializes 'scale' as zeros.
    - A 'NestMap' layer that:
        · Passes 'loc' unchanged through a no-op layer.
        · Passes 'scale' through a 'std_layers' to compute the standard deviation, which is converted to positive using 'softplus'.
    - A 'Lambda' layer that creates a multivariate normal distribution from 'loc' and 'scale'.
"""
value_net = value_network.ValueNetwork(
    input_tensor_spec=observation_tensor_spec,
    fc_layer_params=params["net"],
    # kernel_initializer=tf.keras.initializers.Orthogonal(),  # -> adequate for 'tanh' activation functions
    # kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
    # kernel_initializer = tf.compat.v1.variance_scaling_initializer, by default
    # activation_fn = tf.keras.activations.relu, by default
)
logger.debug("Actor & Value networks initialized")

### Optimizer
# Use tf.compat.v1.train (TensorFlow 1.x API) or tf.kereas.optimizers (TensorFlow 2.x API)
""" tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.001, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False, name='Adam'
)
"""
#optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params["learning_rate"])
optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])

### Distribution strategy
# Networks and agent have to be initialized within strategy.scope
""" get_strategy(tpu, use_gpu)
        Utility to create a `tf.DistributionStrategy` for TPU or GPU.
        If neither is being used a DefaultStrategy is returned which allows executing on CPU only.
        Args:
          tpu: BNS address of TPU to use. Note the flag and param are called TPU as that is what the xmanager utilities call.
          use_gpu: Whether a GPU should be used. This will create a MirroredStrategy.
        Raises:
          ValueError if both tpu and use_gpu are set.
        Returns:
          An instance of a `tf.DistributionStrategy`.
"""
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False) ## no used gpus for rl tf training
logger.debug(f"Distribution strategy initialized")
# Context / Strategy scope: ensures that variables are created on the appropiated devices,
# and how computations are distributed and synchronized along devices
if strategy:
    # If strategy is defined -> Set the context
    context = strategy.scope()
    logger.debug(f"Strategy scope used")
else:
    # If strategy is not defined ('None') -> Set context to placeholder that does nothing
    context = contextlib.nullcontext()
    logger.debug("Null strategy scope")
# Context manager handles setup and teardown of the distributed environment, 
# ensuring resources are allocated and released properly
with context:
    # Set TF random seed within strategy to obtain reproducible results
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    tf.random.set_seed(params["seed"])

    ### PPO Agent, implementing the clipped probability ratios
    agent = ppo_clip_agent.PPOClipAgent(
        time_step_tensor_spec,              # 'TimeStep' spec of the expected time_steps
        action_tensor_spec,                 # nest of BoundedTensorSpec representing the actionsq
        optimizer=optimizer,                # optimizer to use for the agent
        actor_net=actor_net,                # function action_net(observations, action_spec) that returns a tensor of action distribution parameters for each observation
                                            #   takes nested observation and returns nested action
        value_net=value_net,                # function value_net(time_steps) that returns value tensor from neural net predictions of each obesrvation.
                                            #   takes nested observation and returns batch of value_preds
        greedy_eval=True,
        importance_ratio_clipping=params["importance_ratio_clipping"],  # epsilon in clipped, surrogate PPO objective
        lambda_value=0.95,                  # lambda parameter for TD-lambda computation (used in TD-lambda GAE, if use_gae=True & use_td_lambda_return=True)
        discount_factor=0.99,               # discount factor for return computation (used in GAE return calculation, if use_gae=True)
        entropy_regularization=params["entropy_regularization"],        # coeff for l2 regularization of entropy, default 0.0
        policy_l2_reg=params["policy_l2_reg"],                          # coeff for l2 regularization of policy, default 0.0
        value_function_l2_reg=params["value_function_l2_reg"],          # coeff for l2 regularization of value function, default 0.0
        shared_vars_l2_reg=params["shared_vars_l2_reg"],                # coeff for l2 regularization of weights shared btw policy and value functions, default 0.0
        value_pred_loss_coef=params["value_pred_loss_coef"],            # multiplier for value prediction loss to balance with policy gradient loss, default 0.5
        num_epochs=params["num_epochs"],    # num. epochs for computing policy updates
        use_gae=True,                       # if true, use generalized advantage estimation for computing per-timestep advantage
                                            # else, just subtracts value predictions from empirical return.
        use_td_lambda_return=False,         # If True (default False), uses td_lambda_return for training value function. (td_lambda_return = gae_advantage + value_predictions) 
        normalize_rewards=params["normalize_rewards"],                  # if true, keeps moving (mean and) variance of rewards, and normalizes incoming rewards
        normalize_observations=params["normalize_observations"],        # if true, keeps a running estimate of observations mean & variance of observations, and uses these statistics to normalize incoming observations (to have mean=0, std=1)
                                            # adv (True):  stabilizes training, improves convergence, consistent learning rate
                                            # cons (True): additional computation, observation spec compatibility (obs. must be tf.float32), sensitivity to distribution change (when non-stationary env)
        log_prob_clipping=0.0,              # +/- value for clipping log probs to prevent inf / NaN values. Default: no clipping. 
        gradient_clipping=None,             # Norm length to clip gradients. Default: no clipping. 
        value_clipping=None,                # Difference between new and old value predictions are clipped to this threshold. Value clipping could be helpful when training very deep networks. Default: no clipping. 
        check_numerics=False,               # If true, adds tf.debugging.check_numerics to help find NaN / Inf values. For debugging only. 
        compute_value_and_advantage_in_train=True,  # A bool to indicate where value prediction and advantage calculation happen. If True, both happen in agent.train(). If False, value prediction is computed during data collection. This argument must be set to False if mini batch learning is enabled. 
        debug_summaries=True,               # if true, gather debug summaries
        summarize_grads_and_vars=True,      # if true, gradient summaries will be written
        train_step_counter=global_step,     # optional counter to increment every time the train operation is run
    )
    agent.initialize()
    logger.debug(f"Agent created & initialized")

### Agent policies
eval_policy = agent.policy              # returns tf_policy.TFPolicy, which can be used to evaluate agent performance (acts greedily based on best action, no exploration)
collect_policy = agent.collect_policy   # returns tf_policy.TFPolicy, which can be used to train the agent, to collect data from the environment (includes exploration)

### Replay Buffer
# Holds the sampled trajectories (batched adds and uniform sampling)
# Stores transitions of the form (state, action, reward, next_state, done)
""" TFUniformReplayBuffer(
        data_spec, batch_size, max_length=1000, scope='TFUniformReplayBuffer', 
        device='cpu:*', table_fn=<class 'tf_agents.replay_buffers.table.Table'>, 
        dataset_drop_remainder=False, dataset_window_shift=None, stateful_dataset=False)
"""
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = collect_env.batch_size,
    max_length = params["replay_buffer_capacity"]
)

### Driver for data collection
# Step metrics:
environment_steps_metric = tf_metrics.EnvironmentSteps()        # counts num. steps
environment_episodes_metric = tf_metrics.NumberOfEpisodes()     # counts num. episodes
step_metrics = [
    environment_episodes_metric,
    environment_steps_metric,
]
# Train metrics
train_avg_return = tf_metrics.AverageReturnMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs)
train_metrics = step_metrics + [
    train_avg_return,
    tf_metrics.MinReturnMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
    tf_metrics.MaxReturnMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
]
# Collect driver
# Controls how the agent interacts with the environment, collects experience data and updates metrics
# Used in RL, where you need to collect experiences over several time-steps before updating agent's policy
""" DynamicEpisodeDriver(
        env:    environment, class tf_environment.Base, 
        policy: policy, class tf_policy.TFPolicy, 
        observers=None: list of observers, updated after every time-step in the environment
                        each observer is a callable(Trajectory)
        transition_observers=None: list of observers, updated after every time-step in the environment
                        each observer is a callable((TimeStep, PolicyStep, NextTimeStep))
        num_episodes=1: num. episodes to take in the environment. For parallel/batched envs, this is the total num. summed across all envs.
    )
"""
collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    collect_env,
    collect_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_episodes=collect_env.n_envs # the number of episodes to take in the environment before each update. This is the total across all parallel RL environments.
)


### # Additional data to log in tensorboard
### def log_action_distributions(action, global_step):
###     with tf.summary.record_if(True):
###         for dim in range(collect_py_env.action_dim):
###             tf.summary.histogram(f"actions/dim_{dim}", action, step=global_step)

### Checkpointers to save policy
ckpt_dir = os.path.join(train_dir, "ckpt")
saved_model_dir = os.path.join(train_dir, "policy_saved_model")
# Checkpointer to entire training state
train_checkpointer = common.Checkpointer(
    ckpt_dir=ckpt_dir,
    max_to_keep=params["ckpt_num"],     # if necessary, oldest checkpoints are deleted
    agent=agent,                        # **kwargs: items to include in the checkpoint
    policy=eval_policy,
    replay_buffer=replay_buffer,
    metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
    global_step=global_step,
)
# Checkpointer to evaluation policy
""" Why to use policy param as 'eval_policy' instead of 'collect_policy':
    For checkpointing during training, it's common to store the evaluation policy (eval_policy=agent.policy) because:
    The evaluation policy reflects the "best" learned policy (without exploration noise).
    If you want to reload and evaluate the agent's performance from the checkpoint, you would use this greedy evaluation policy to get the actual learned behavior.
    Using collect_policy in the checkpoint doesn't make sense in this context because it includes exploration, which is more about gathering training data, not representing the final learned behavior.    """
policy_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(train_dir, 'policy'),
    policy=eval_policy,
    global_step=global_step,
)
# Initialize PolicySaver, allows to save a tf_policy.Policy to SavedModel
saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)

# Restore training process if existing saved checkpoints
""" initialize_or_restore(self, session=None)
        Initialize or restore graph (based on checkpoint if exists).
    If not checkpoints available, this will return:
        INFO:absl:No checkpoint available at <train_dir>/ckpt
        INFO:absl:No checkpoint available at <train_dir>/policy     """
if params["run_id"] != "":
    try:
        train_checkpointer.initialize_or_restore()  
        if not train_checkpointer._checkpoint_exists:
            raise RuntimeError(f"No checkpoint available for run_id '{params['run_id']}' at the specified path: {train_checkpointer._manager.latest_checkpoint}")
        logger.info(f"Train checkpointer restored from '{{train_checkpointer._manager.latest_checkpoint}}'")
    except Exception as e:
        raise RuntimeError(f"Error occurred during checkpoint restore: {str(e)}")

#--------------------------- Training / Evaluation ---------------------------
with tf.compat.v2.summary.record_if(  # pylint: disable=not-context-manager
    lambda: tf.math.equal(global_step % params["summary_interval"], 0)
):
    # Define train step 
    def train_step():
        trajectories = replay_buffer.gather_all()       # gather all available trajectories stored in buffer
        return agent.train(experience=trajectories)     # agent updates internal params (policy & value networks) taking the gathered trajectory of experiences as input
                                                        # returns 'LossInfo' tuple containing loss and info tensors
    if params["use_tf_functions"]:
        collect_driver.run = common.function(           # wrap collect_driver.run function for optimized execution as TensorFlow graph
            collect_driver.run, autograph=False)        
        agent.train = common.function(agent.train, autograph=False)     # wrap agent.train function as TensorFlow graph
        train_step = common.function(train_step, autograph=True)        # wrap train_step function (autograph=True for converting Python control flow into TensorFlow operations)

    if params["mode"] == "train":                       # initialize counters
        collect_time = 0
        train_time = 0
        timed_at_step = agent.train_step_counter.numpy()

        # Write parameter files to Tensorboard and plots in train directory
        tf.summary.text("params", params_html_table(params), step=global_step.numpy())
        logger.debug("Parameters written in Tensorboard")

        history = History(train_dir)

        # Train loop
        logger.info(f"{bcolors.BOLD}Starting training loop!{bcolors.ENDC}")
        logger.info(f"Current training global step: {timed_at_step}\n")
        while environment_episodes_metric.result() < params["num_episodes"]:
            logger.info(f"{bcolors.OKBLUE}Collect environment running{bcolors.ENDC}")
            global_step_val = global_step.numpy()
            start_time = time.time()
            collect_env.start(
                new_ensamble=True,
                restart_file=params["restart_file"], # TODO: impl. sth if we want to use several restart files, i.e. if global_step_val > 0 else 1,
                global_step=global_step_val,
            )
            logger.debug(f"Collect environment started, with global_step: {global_step_val}")
            collect_driver.run()
            logger.debug(f"Collect environment run, with global_step: {global_step_val}")
            collect_env.stop()
            logger.debug(f"Collect environment stopped, with global_step: {global_step_val}")
            collect_time += time.time() - start_time

            start_time = time.time()
            total_loss, _ = train_step()
            logger.debug(f"Train step done, with global_step: {global_step_val}")
            logger.debug(f"Replay buffer size before clear: {replay_buffer.num_frames()}")
            replay_buffer.clear()
            logger.debug(f"Replay buffer size after clear: {replay_buffer.num_frames()}")
            train_time += time.time() - start_time

            logger.info("Writing tensorflow summary")
            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)

            if global_step_val % params["ckpt_interval"] == 0:
                logger.info(f"Saving checkpoint at global_step_val: {global_step_val} to: {ckpt_dir}")
                train_checkpointer.save(global_step_val)
                policy_checkpointer.save(global_step_val)
                saved_model_path = os.path.join(saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
                saved_model.save(saved_model_path)
                    

            if global_step_val % params["log_interval"] == 0:
                logger.info(f"{bcolors.OKCYAN}Training stats:{bcolors.ENDC}")
                logger.info('step = %d, loss = %f', global_step_val, total_loss)
                steps_per_hour = (agent.train_step_counter.numpy() - timed_at_step) / (collect_time + train_time) * 3600
                logger.info('%.4f steps/hour', steps_per_hour)
                logger.info('collect_time = %.4f, train_time = %.4f', collect_time, train_time)
                with tf.compat.v2.summary.record_if(True): # pylint: disable=not-context-manager
                    tf.compat.v2.summary.scalar(name='global_steps_per_hour', data=steps_per_hour, step=global_step)

                logger.info(f"Episodes: {environment_episodes_metric.result().numpy()}")
                logger.info(f"Global training steps: {agent.train_step_counter.numpy()}")
                logger.info(f"Environment steps: {environment_steps_metric.result().numpy()}")
                logger.info(f"Average reward: {numpy_str(train_avg_return.result().numpy())}")
                #history.plot() # TODO: uncomment this when history.plot() is customized to RHEA implementation
                logger.info("Plotting training metrics done!\n")

                timed_at_step = agent.train_step_counter.numpy()
                collect_time = 0
                train_time = 0

        logger.info(f"{bcolors.BOLD}Ended training loop!{bcolors.ENDC}\n")

    elif params['mode'] == "eval":
        # Init environment
        eval_py_env = SodEnv(
            exp,
            db,
            hosts,
            "sod2d",
            cwd,
            cfd_n_envs=1,
            mode="eval",
            **env_params,
        )
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        eval_dir = os.path.join(cwd, "eval")
        eval_summary_writer = tf.summary.create_file_writer(eval_dir, flush_millis=1000)
        eval_avg_return = tf_metrics.AverageReturnMetric(buffer_size=eval_env.n_envs, batch_size=eval_env.n_envs)
        eval_metrics = [
            eval_avg_return,
            tf_metrics.MinReturnMetric(buffer_size=eval_env.n_envs, batch_size=eval_env.n_envs),
            tf_metrics.MaxReturnMetric(buffer_size=eval_env.n_envs, batch_size=eval_env.n_envs),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=eval_env.n_envs, batch_size=eval_env.n_envs),
        ]
        history = History(eval_dir)

        logger.info(f"{bcolors.OKBLUE}Evaluation environment running{bcolors.ENDC}")
        logger.info(f"{bcolors.OKCYAN}  - Agent trained with {global_step.numpy()} MARL episodes.{bcolors.ENDC}")

        eval_env.start(
            new_ensamble=True,
            restart_file=1,
            global_step=global_step.numpy()
        )
        metric_utils.eager_compute(
            eval_metrics,
            eval_env,
            eval_policy,
            num_episodes=1,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix="Metrics",
        )
        eval_env.stop()

        logger.info(f"{bcolors.OKCYAN}Evaluation stats:{bcolors.ENDC}")
        logger.info(f"Average reward (eval): {numpy_str(eval_avg_return.result().numpy())}")
        history.plot()
        logger.info("Plotting evaluation metrics done!")

    else:
        logger.info(f"Mode = {params['mode']} not recognised. Aborting simulation.")

# Kill database
exp.stop(db)
time.sleep(2.0)
