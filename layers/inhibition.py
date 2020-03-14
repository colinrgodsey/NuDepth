import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import activations
from tensorflow.python.keras import backend as K

class LocalResponseInhibition(tf.keras.layers.Layer):
  def __init__(self,
               radius=3.,
               burst_rate=0.1,
               burst_rate_inference=0.01,
               sample_activation='swish',
               **kwargs):
    super(LocalResponseInhibition, self).__init__(**kwargs)

    self.radius = float(radius)
    self.burst_rate = float(burst_rate)
    self.burst_rate_inference = float(burst_rate_inference)
    self.sample_activation = activations.get(sample_activation)

  def get_config(self):
    config = {
      'radius': self.radius,
      'burst_rate': self.burst_rate,
      'burst_rate_inference': self.burst_rate_inference,
      'sample_activation': activations.serialize(self.sample_activation)
    }
    config.update(super(LocalResponseInhibition, self).get_config())
    return config

  def build(self, input_shape):
    num_dims = len(input_shape) - 2

    mean = 0.0
    std = self.radius / 2.0
    size = self.radius
    d = tfp.distributions.Normal(mean, std)
    gauss_kernel = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    kernel_shape = (1,) * num_dims + gauss_kernel.shape + (1, 1)

    self.gauss_kernel = tf.reshape(gauss_kernel, kernel_shape)

    self.built = True

  def call(self, inputs, training=None):
    # rate at which columns are ignored for local activation strength
    burst_rate = K.in_train_phase(self.burst_rate, self.burst_rate_inference, training=training)

    inputs = tf.expand_dims(inputs, axis=-1)

    sample = tf.nn.dropout(inputs, burst_rate)
    sample = self.sample_activation(sample)

    #should this really just be a max pool? this could maybe be an Add layer of two parallel layers,
    #each using an activation function that references the entire dim space to do the same operation.
    n_avg = tf.nn.convolution(sample, self.gauss_kernel, strides=1, padding="SAME", data_format='NDHWC')
    n_max = tf.nn.max_pool(sample, [self.radius], strides=1, padding="SAME", data_format='NDHWC')

    local = (n_avg + n_max) / 2.0

    mask = tf.cast(inputs < local, tf.float32)
    output = inputs * tf.math.exp(-3.0 * mask)

    return tf.squeeze(output, axis=-1)
