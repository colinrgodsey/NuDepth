import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

class Inception(tf.keras.layers.Layer):
  def __init__(self,
               specs,
               trainable=True,
               name=None,
               strides=1,
               activation='swish',
               kernel_initializer='glorot_uniform',
               kernel_constraint=None,
               bias_initializer='zeros',
               kernel_regularizer=None,
               **kwargs):
    super(Inception, self).__init__(
      trainable=trainable,
      name=name,
      **kwargs
    )

    self.specs = specs
    self.strides = strides
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)

    (br0, br1, br2, br3) = specs

    self.branch_0a = self.conv2d(br0[0], 1, trainable, '1x1', strides=self.strides)

    self.branch_1a = self.conv2d(br1[0], 1, trainable, '3x3_reduce')
    self.branch_1b = self.conv2d(br1[1], 3, trainable, '3x3', strides=self.strides)

    self.branch_2a = self.conv2d(br2[0], 1, trainable, '5x5_reduce')
    self.branch_2b = self.conv2d(br2[1], 5, trainable, '5x5', strides=self.strides)

    self.branch_3a = MaxPooling2D(3, strides=1, padding='same', name='pool')
    self.branch_3b = self.conv2d(br3[0], 1, trainable, 'pool_proj', strides=self.strides)

  def conv2d(self, filters, size, trainable, name, strides=1):
    return Conv2D(
      filters, size,
      strides=strides,
      padding='same',
      trainable=trainable,
      activation=self.activation,
      kernel_initializer=self.kernel_initializer,
      kernel_constraint=self.kernel_constraint,
      bias_initializer=self.bias_initializer,
      kernel_regularizer=self.kernel_regularizer,
      name=name)

  def build(self, _):
    self.built = True

  def call(self, input, training=None):
    return tf.keras.layers.concatenate([
      self.branch_0a(input, training=training),
      self.branch_1b(self.branch_1a(input, training=training), training=training),
      self.branch_2b(self.branch_2a(input, training=training), training=training),
      self.branch_3b(self.branch_3a(input, training=training), training=training),
    ])

  def get_config(self):
    config = {
      'specs': self.specs,
      'strides': self.strides,
      'activation': activations.serialize(self.activation),
      'kernel_initializer': initializers.serialize(self.kernel_initializer),
      'kernel_constraint': constraints.serialize(self.kernel_constraint),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
    }
    config.update(super(Inception, self).get_config())
    return config

