import tensorflow as tf

from tensorflow.keras.layers import Reshape

class EdgeMap(tf.keras.layers.Layer):
  def __init__(self, desaturate=True, **kwargs):
    super(EdgeMap, self).__init__(**kwargs)

    self.desaturate = desaturate

  def get_config(self):
    config = {
      'desaturate': self.desaturate
    }
    config.update(super(EdgeMap, self).get_config())
    return config

  def build(self, input_shapes):
    self.built = True

  def call(self, inputs):
    input_shape = inputs.shape

    x = tf.image.sobel_edges(inputs)
    x = tf.math.abs(x)

    if self.desaturate:
      return tf.norm(x, axis=-2)
    else:
      return Reshape(input_shape[1:-1] + (input_shape[-1] * 2,))(x)