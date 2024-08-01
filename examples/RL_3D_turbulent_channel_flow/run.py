import absl.logging
import os
import tensorflow as tf
import time

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

from params import params, env_params
from smartrhea.init_smartsim import init_smartsim
from smartrhea.utils import print_params, deactivate_tf_gpus
from smartrhea.rhea_env import RheaEnv

#--------------------------- Utils ---------------------------

absl.logging.set_verbosity(absl.logging.DEBUG)
logger = get_logger(__name__)
cwd = os.path.dirname(os.path.realpath(__file__))
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
train_dir = os.path.join(cwd, "train")
summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=1000)
summary_writer.set_as_default()
print("")
logger.info(f"Tensorboard event file created: {train_dir}/events.out.tfevents.{int(time.time())}.{gethostname()}.{os.getpid()}")

# Print simulation params
print_params(params, "RUN PARAMETERS:")

#--------------------------- RL setup ---------------------------
# Init SmartSim framework: Experiment and Orchestrator (database)
exp, hosts, db, db_is_clustered = init_smartsim(
    port = params["port"],
    network_interface = params["network_interface"],
    launcher = params["launcher"],
    run_command = params["run_command"],
)

# Init environment
collect_py_env = RheaEnv(
    exp,
    db,
    hosts,
    "RHEA.exe",
    "/home/jofre/Nuria/repositories/RL_3d_turbulent_channel_flow/ProjectRHEA/examples",
    cwd,
    mode = "collect",
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
global_step = tf.compat.v1.train.get_or_create_global_step()
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
logger.info(f'Observation Spec:\n{observation_tensor_spec}')
logger.info(f'Action Spec:\n{action_tensor_spec}')
logger.info(f'Time Spec:\n{time_step_tensor_spec}')


""" Actor Network:
    ppo_actor_network.PPOActorNetwork.__init__(self, seed_stream_class=<class 'tensorflow_probability.python.util.seed_stream.SeedStream'>)
    PPOActorNetwork.create_sequential_actor_net(self, fc_layer_units, action_tensor_spec, seed=None)
"""
actor_net_builder = ppo_actor_network.PPOActorNetwork() # TODO: Check ActorDistributionRnnNetwork otherwise
actor_net = actor_net_builder.create_sequential_actor_net(params["net"], action_tensor_spec)
""" Value Network:    
    value_network.ValueNetwork.__init__(self, input_tensor_spec, preprocessing_layers=None, 
        preprocessing_combiner=None, conv_layer_params=None, fc_layer_params=(75, 40), 
        dropout_layer_params=None, activation_fn=<function relu at 0x79f78275c280>, 
        kernel_initializer=None, batch_squash=True, dtype=tf.float32, name='ValueNetwork')
"""
value_net = value_network.ValueNetwork(
    observation_tensor_spec,
    fc_layer_params=params["net"],
    kernel_initializer=tf.keras.initializers.Orthogonal()
)
logger.debug("Actor & Value networks initialized")

"""
# For distribution strategy, networks and agent have to be initialized within strategy.scope
# optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params["learning_rate"])
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False) ## no used gpus for rl tf training
if strategy:
    context = strategy.scope()
else:
    context = contextlib.nullcontext()  # placeholder that does nothing
with context:
    # Set TF random seed within strategy to obtain reproducible results
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    tf.random.set_seed(params["seed"])

    # PPO Agent
    agent = ppo_clip_agent.PPOClipAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        discount_factor=0.99,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=params["num_epochs"],
        debug_summaries=False,
        summarize_grads_and_vars=False,
        train_step_counter=global_step
    )

    agent.initialize()

# Get agent policies
eval_policy = agent.policy
collect_policy = agent.collect_policy

# Instantiate Replay Buffer, which holds the sampled trajectories.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = collect_env.batch_size,
    max_length = params["replay_buffer_capacity"]
)

# Instantiate driver for data collection
environment_steps_metric = tf_metrics.EnvironmentSteps()
environment_episodes_metric = tf_metrics.NumberOfEpisodes()
train_avg_return = tf_metrics.AverageReturnMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs)
step_metrics = [
    environment_episodes_metric,
    environment_steps_metric,
]
train_metrics = step_metrics + [
    train_avg_return,
    tf_metrics.MinReturnMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
    tf_metrics.MaxReturnMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
]
collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    collect_env,
    collect_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_episodes=collect_env.n_envs # the number of episodes to take in the environment before each update. This is the total across all parallel MARL environments.
)

# Define checkpointer to save policy
ckpt_dir = os.path.join(train_dir, "ckpt")
saved_model_dir = os.path.join(train_dir, "policy_saved_model")

train_checkpointer = common.Checkpointer(
    ckpt_dir=ckpt_dir,
    max_to_keep=params["ckpt_num"],
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
    global_step=global_step)

policy_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(train_dir, 'policy'),
    policy=eval_policy,
    global_step=global_step,
)

saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
train_checkpointer.initialize_or_restore()

#--------------------------- Training / Evaluation ---------------------------
with tf.compat.v2.summary.record_if(  # pylint: disable=not-context-manager
    lambda: tf.math.equal(global_step % params["summary_interval"], 0)
):
    def train_step():
        trajectories = replay_buffer.gather_all()
        return agent.train(experience=trajectories)

    if params["use_tf_functions"]:
        collect_driver.run = common.function(
            collect_driver.run, autograph=False)
        agent.train = common.function(agent.train, autograph=False)
        train_step = common.function(train_step)

    if params["mode"] == "train":
        collect_time = 0
        train_time = 0
        timed_at_step = agent.train_step_counter.numpy()

        # Write parameter files to Tensorboard and plots in train directory
        tf.summary.text("params", params_str(params), step=global_step.numpy())
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
                restart_file=params["restart_file"] if global_step_val > 0 else 1,
                global_step=global_step_val
            )
            collect_driver.run()
            collect_env.stop()
            collect_time += time.time() - start_time

            start_time = time.time()
            total_loss, _ = train_step()
            replay_buffer.clear()
            train_time += time.time() - start_time

            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)

            if global_step_val % params["ckpt_interval"] == 0:
                logger.info(f"Saving checkpoint to: {ckpt_dir}")
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

                history.plot()
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
"""