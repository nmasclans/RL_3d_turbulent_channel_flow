import os
import tensorflow as tf

from smartsim.log import get_logger
import absl.logging

from params import params, env_params
from smartrhea.init_smartsim import init_smartsim
from smartrhea.utils import print_params, deactivate_tf_gpus

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

# Write summary to TensorBoard
train_dir = os.path.join(cwd, "train")
summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=1000)
summary_writer.set_as_default()

# Print simulation params
print_params(params)

#--------------------------- RL setup ---------------------------
# Init SmartSim framework: Experiment and Orchestrator (database)
exp, hosts, db, db_is_clustered = init_smartsim(
    port = params["port"],
    network_interface = params["network_interface"],
    launcher = params["launcher"],
    run_command = params["run_command"],
)