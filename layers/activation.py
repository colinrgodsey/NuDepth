import tensorflow as tf

from tensorflow.keras.utils import get_custom_objects

class Clampazzo(tf.keras.layers.Layer):
  def __init__(self, beta=1., **kwargs):
    super(Clampazzo, self).__init__(**kwargs)
    self.beta = beta

  def get_config(self):
    config = {
      'beta': self.beta
    }
    config.update(super(Clampazzo, self).get_config())
    return config

  def build(self, input_shapes):
    self.built = True

  def call(self, inputs):
    # TEST: could also be softsign instead of tanh?
    #return tf.nn.tanh(inputs) * tf.nn.sigmoid(self.beta * inputs)
    return tf.nn.softsign(inputs) * tf.nn.sigmoid(self.beta * inputs)

class Swish(tf.keras.layers.Layer):
  def __init__(self, beta=1., **kwargs):
    super(Swish, self).__init__(**kwargs)
    self.beta = beta

  def get_config(self):
    config = {
      'beta': self.beta
    }
    config.update(super(Swish, self).get_config())
    return config

  def build(self, input_shapes):
    self.built = True

  def call(self, inputs):
    return inputs * tf.nn.sigmoid(self.beta * inputs)

get_custom_objects().update({
  'clampz': Clampazzo(),
  'clampazzo': Clampazzo(),
  'swish': Swish()
})

