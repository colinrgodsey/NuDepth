import random
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import InputSpec

random.seed()

class Sparse(keras.layers.Dense):
  def __init__(self, units,
               density=0.5,
               **kwargs):
    super(Sparse, self).__init__(units, **kwargs)
    self.density = density

  def build(self, input_shape):
    assert len(input_shape) >= 2
    input_dim = input_shape[-1]

    self.kernel = self.add_weight(shape=(input_dim, self.units),
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self._kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(shape=(self.units,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    self.mask = tf.Variable(
      initial_value=self.create_mask([input_dim, self.units], tf.int8),
      name='mask', trainable=False)
    self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
    self.built = True

  def _kernel_constraint(self, w):
    w = w * tf.cast(self.mask, w.dtype)
    if self.kernel_constraint:
      w = self.kernel_constraint(w)
    return w

  def create_mask(self, shape, dtype):
    n = shape[1] * shape[0]
    mask = [random.random() < self.density for _ in range(n)]
    mask = tf.reshape(mask, shape)
    return tf.cast(mask, dtype)

  def get_config(self):
    config = {
      'density': self.density
    }
    base_config = super(Sparse, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
