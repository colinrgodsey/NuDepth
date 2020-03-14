import tensorflow as tf

class SegmentPooling(tf.keras.layers.Layer):
  def __init__(self, factor,
               **kwargs):
    super(SegmentPooling, self).__init__(**kwargs)

    self.factor = factor

    self.segment_window = []

  def get_config(self):
    config = {
      'factor': self.factor
    }
    config.update(super(SegmentPooling, self).get_config())
    return config

  def build(self, input_shape):
    if input_shape[-1] % self.factor != 0:
      raise ValueError("input dimensions must be evenly divisible by pooling factor")

    dims = len(input_shape) - 2

    self.segment_window = [1 for _ in range(dims)]
    self.segment_window.append(self.factor)

    self.built = True

  def call(self, inputs):
    x = tf.expand_dims(inputs, axis=-1)
    x = tf.nn.max_pool(x, self.segment_window, self.segment_window, 'VALID', data_format='NDHWC')
    x = tf.squeeze(x, axis=-1)
    return x

