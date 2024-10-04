r"""Sequential Actor Network for PPO."""
import functools
import sys

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tf_agents.keras_layers import bias_layer
from tf_agents.networks import nest_map
from tf_agents.networks import sequential


def tanh_and_scale_to_spec(inputs, spec):
    """Maps inputs with arbitrary range to range defined by spec using `tanh`."""
    means = (spec.maximum + spec.minimum) / 2.0
    magnitudes = (spec.maximum - spec.minimum) / 2.0

    return means + magnitudes * tf.tanh(inputs)


class CustomPPOActorNetwork:
  """Contains the actor network structure."""

  def __init__(self, seed_stream_class=tfp.util.SeedStream):
    self.seed_stream_class = seed_stream_class

  def create_sequential_actor_net(
      self, fc_layer_units, action_tensor_spec, activation_fn='tanh', l2_reg_value=1e-4, std_init_value=0.35, seed=None
  ):
    """Helper method for creating the actor network with L2 regularization."""

    self._seed_stream = self.seed_stream_class(
        seed=seed, salt='tf_agents_sequential_layers'
    )

    def _get_seed():
        seed = self._seed_stream()
        if seed is not None:
            seed = seed % sys.maxsize
        return seed

    def create_dist(loc_and_scale):
        loc = loc_and_scale['loc']
        loc = tanh_and_scale_to_spec(loc, action_tensor_spec)

        scale = loc_and_scale['scale']
        scale = tf.math.softplus(scale)

        return tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale, validate_args=True
        )

    def means_layers():
      # TODO(b/179510447): align these parameters with Schulman 17.
        return tf.keras.layers.Dense(
            action_tensor_spec.shape.num_elements(),
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=0.1, seed=_get_seed()
            ),
            name='means_projection_layer',
        )

    def std_layers():
        # TODO(b/179510447): align these parameters with Schulman 17.
        std_bias_initializer_value = np.log(np.exp(std_init_value) - 1)
        return bias_layer.BiasLayer(
            bias_initializer=tf.constant_initializer(
                value=std_bias_initializer_value
            )   # No l2-regularization in std-layers
        )

    def no_op_layers():
        return tf.keras.layers.Lambda(lambda x: x)

    if activation_fn == 'tanh':
        dense = functools.partial(
            tf.keras.layers.Dense,
            activation=tf.nn.tanh,                                      # Apply 'tanh' activation function
            kernel_initializer=tf.keras.initializers.Orthogonal(seed=_get_seed()),
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_value),           # Apply L2 regularization, added line to original PPOActorNetwork class
        )
    elif activation_fn == 'relu':
        dense = functools.partial(
            tf.keras.layers.Dense,
            activation=tf.nn.relu,                                      # Apply 'relu' activation function
            kernel_initializer=tf.keras.initializers.Orthogonal(seed=_get_seed()),
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_value),           # Apply L2 regularization, added line to original PPOActorNetwork class
        )
    else:
        raise ValueError(f"Invalid activation function '{activation_fn}'")

    return sequential.Sequential(
        [dense(num_units) for num_units in fc_layer_units]
        + [means_layers()]
        + [
            tf.keras.layers.Lambda(
                lambda x: {'loc': x, 'scale': tf.zeros_like(x)}
            )
        ]
        + [
            nest_map.NestMap({
                'loc': no_op_layers(),
                'scale': std_layers(),
            })
        ]
        +
        # Create the output distribution from the mean and standard deviation.
        [tf.keras.layers.Lambda(create_dist)]
    )