import tensorflow as tf

from tensorflow.keras import backend as K

#TODO: rename to gated activation

#takes two inputs, returns output the same size as the second input
class CrossActivator(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(CrossActivator, self).__init__(**kwargs)

  def build(self, input_shapes):
    if input_shapes[1][-1] % input_shapes[0][-1] is not 0:
      raise ValueError("state channels must be evenly divisible by input channels")

    self.built = True

  def call(self, inputs):
    input, state = inputs

    factor = int(state.shape[-1] / input.shape[-1])

    orig_shape = tf.shape(state)
    new_shape = tf.concat([orig_shape[:-1], (input.shape[-1], factor)], axis=0)

    output = state
    output = tf.reshape(output, new_shape)
    output = tf.nn.softmax(output + K.epsilon())
    output *= tf.expand_dims(input, axis=-1)

    return tf.reshape(output, orig_shape)
