import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import Sequential

from tensorflow.keras.layers import *

from tensorflow.keras.regularizers import l2

from .layers import *

def create(density=0.30, big=False):  #density=0.4
  max_segment_strength = 2.0

  if big:
    telem_units = 64
    visual_units = 256
    int_dims = 7

    l4_units = 64
    l4_segments = 3

    l3_cells = 6
    l3_segments = 8
    l3_l2_redux_units = 256

    # about twice the number of total l3 cells
    l2_units = 768
    l2_xcontext_fac = 4  # 128 per
    # l2_segments = 2
    l2_segments = 4

    depth1_units = 128
    depth1_segments = 2
    gen1_units = 64
    gen2_units = 32
  else:
    telem_units = 32
    visual_units = 128
    int_dims = 7

    l4_units = 32
    l4_segments = 4 # 2

    l3_cells = 5
    l3_segments = 8
    l3_l2_redux_units = 64

    # about twice the number of total l3 cells
    l2_units = 256
    l2_xcontext_fac = 1  # 2  # 128 per
    l2_segments = 3 # 2

    depth1_units = 64
    depth1_segments = 2
    gen1_units = 32
    gen2_units = 16

  norm_init = tf.random_normal_initializer(0., 0.02)

  # keep output high
  def segment_activity_regularizer(x, l=1e-1):
    x = tf.nn.relu(x)
    x = tf.constant(1.0) - tf.nn.tanh(x)
    return tf.math.reduce_mean(x) * l

  def segment_kernel_constraint(x):
    x = tf.clip_by_value(x, -1.0, 1.0)
    x = tf.keras.constraints.max_norm()(x)
    return x

  def segment_block(x, segments, factor, name):
    """
    Rethinking this again.
    I like the statistical average of output being variance=1, mean 0, sort of...

    Ideally, it would be best if we could compute variance with a fixed mean of 0.
    Actually ^ it's maybe better to keep the mean above 0 so we can some real negative
    values that go into the Activator layer.
    """
    x = Sparse(segments * factor, density,
               use_bias=False,
               #activity_regularizer=segment_activity_regularizer,
               kernel_initializer='random_uniform',
               #kernel_constraint='max_norm',
               kernel_constraint=segment_kernel_constraint,
               name=name+'/segments')(x)

    return SegmentPooling(factor)(x)

  """
  Using Batch norm is odd because it changes what zero means.
  
  Maybe, inhibition layer should just shift all output so only 1sd+ is positive, scale by variance, then apply clampz!
  That would replace normalization, inhibition, and activation. ^^^^^
  Would also guarantee a solid SDR each time.
  
  ORR still do shift, then LRI, then scale.
  
  I think negative values should never be a big thing- weights, maybe, a bit.
  
  x = lri(x)
  sd = sd(x)
  var = var(x)
  x = x - sd
  x = x / var
  x = clampz(x)
  """
  def single_dim_bn(x, name):
    orig_shape = x.shape[1:]
    x = Reshape(orig_shape + (1,))(x)
    x = BatchNormalization(center=False, scale=False, name=name+'/bsn')(x, training=True) # TEST assuming we always need this training
    x = Reshape(orig_shape)(x)
    return x

  def cell_layer(x, units, segments, name):
    # x = segment_block(x, units, segments, name)
    # #do LRI first so we skew more towards 0
    # x = LocalResponseInhibition(name=name+'/lri')(x)
    # x = single_dim_bn(x, name)
    # x = Dropout(0.1)(x)
    # x = Activation('clampz')(x)
    # #x = LocalResponseInhibition(name='l4/lri')(x)

    #x = LocalResponseInhibition(name=name + '/lri')(x)
    #x -= tf.reduce_std()

    x = segment_block(x, units, segments, name)
    x = single_dim_bn(x, name)
    x = x * 2.0
    x = LocalResponseInhibition(name=name + '/lri')(x)
    x = Dropout(0.1)(x)
    #x = LocalResponseInhibition(name=name + '/lri')(x)
    x = Layer(activity_regularizer=segment_activity_regularizer)(x)
    x = Activation('clampz')(x)

    return x

  """
  This could probably get replaced with a sort of 'grid cell' like encoding, but in 'velocity' space.
  Grid cells would each just be a random modulo and scale, N times, for each dimension
  """
  def get_odometry_patch(x):
    x = Dense(telem_units / 2, name='odometry/1')(x)
    x = BatchNormalization(name='odometry/1/bn')(x)
    x = Activation('swish')(x)
    x = Dense(telem_units, activation='clampz', name='odometry/2')(x)
    x = RepeatVector(int_dims * int_dims)(x)
    return Reshape([int_dims, int_dims, telem_units])(x)

  def l4_block(x, odom):
    x = Concatenate(axis=-1)([
      get_odometry_patch(odom),
      x
    ])
    x = cell_layer(x, l4_units, l4_segments, 'l4')
    return x

  def l3_block(x, state, l2_state):
    redux = Sparse(l3_l2_redux_units, density,
             use_bias=False,
             kernel_constraint='max_norm',
             name='l3/l2_redux')(l2_state)
    redux = single_dim_bn(redux, 'l3/l2_redux')
    context = Concatenate(axis=-1)([
      state,
      redux
    ])
    context = segment_block(context, l4_units * l3_cells, l3_segments, 'l3')
    x = CrossActivator()([x, context])
    x = Dropout(0.1)(x)
    return x

  def l2_block(x, state):
    xcontext = Inception(((64 * l2_xcontext_fac,),
                 (12 * l2_xcontext_fac, 32 * l2_xcontext_fac),
                 (6 * l2_xcontext_fac, 16 * l2_xcontext_fac),
                 (16 * l2_xcontext_fac,)),
                activation='clampz',
                kernel_constraint='max_norm',
                name='l2/xcontext')(state)
    xcontext = single_dim_bn(xcontext, 'l2/xcontext')
    x = Concatenate(axis=-1)([
      xcontext,
      x
    ])
    x = cell_layer(x, l2_units, l2_segments,  'l2')
    return x

  def upsample(filters, size, name='upsample', apply_dropout=False, strides=2):
    result = Sequential(name=name)
    result.add(
      Conv2DTranspose(filters, size, strides=strides,
                      padding='same', use_bias=False,
                      kernel_initializer=norm_init,
                      name=name+'/conv2d'))

    result.add(BatchNormalization(name=name+'/bn'))

    if apply_dropout:
      result.add(Dropout(0.2))

    result.add(Activation('swish'))

    return result

  # TODO: should we do tanh on state input, instead of output? might be better for gradients

  #input = inception_model.input
  input = Input(shape=[224, 224, 3], name='input', dtype=tf.float32)
  odometry_in = Input(shape=[6], name='odometry', dtype=tf.float32)
  l2_state_in = Input(shape=[int_dims, int_dims, l2_units], name='l2state', dtype=tf.float32)
  l3_state_in = Input(shape=[int_dims, int_dims, l4_units * l3_cells], name='l3state', dtype=tf.float32)
  l4_state_in = Input(shape=[int_dims, int_dims, l4_units], name='l4state', dtype=tf.float32)

  x = input
  x = EdgeMap()(x)
  x = Conv2D(32, 6, 2, padding='same', activation='swish', name='edge/downsample/1')(x)
  x = MaxPool2D(4, 2, padding='same')(x)
  x = Conv2D(48, 4, 2, padding='same', activation='swish', name='edge/downsample/2')(x)
  x = MaxPool2D(4, 2, padding='same')(x)
  x = Inception(((64,), (16, 32), (12, 16), (16,)), name='edge/downsample/3')(x)
  if big:
    x = Inception(((96,), (24, 48), (16, 24), (24,)), name='edge/downsample/4')(x)
  x = MaxPool2D(4, 2, padding='same')(x)
  x = Conv2D(visual_units, 3, 1, padding='same', use_bias=False, name='edge/downsample/last')(x)
  x = BatchNormalization(name='edge/bn')(x)
  x = Activation('clampz', name='edge/output')(x)
  visual_out = x

  x = l4_block(x, odometry_in)
  l4_state_out = x

  x = l3_block(x, l3_state_in, l2_state_in)
  l3_state_out = x

  x = l2_block(x, l2_state_in)
  l2_state_out = StabilityRegularizer(1e-6)([l2_state_in, x])
  #l2_state_out = x

  #x = Concatenate(axis=-1)([l2_state_out, visual_out])
  x = l2_state_out

  # x = segment_block(x, depth1_units, depth1_segments, 'depth')
  # x = Dropout(0.05)(x)
  # x = LocalResponseInhibition(density, name='depth/lri')(x)
  # x = Activation('clampz')(x)

  #TODO: maybe split network to be combination of shape and max depth?

  x = upsample(gen1_units, 4, apply_dropout=True, name='depth/upsample/1')(x)
  x = upsample(gen2_units, 4, name='depth/upsample/2')(x)

  x = Conv2DTranspose(1, 3,
                      strides=1, padding='same',
                      kernel_initializer=norm_init,
                      use_bias=False,
                      name='depth/upsample/last',
                      activation='softsign')(x)

  return Model(
    [input, odometry_in, l2_state_in, l3_state_in, l4_state_in],
    [x, l2_state_out, l3_state_out, l4_state_out]
  )

class StabilityRegularizer(tf.keras.layers.Layer):
  def __init__(self, l, **kwargs):
    super(StabilityRegularizer, self).__init__(**kwargs)

    self.l = l

  def get_config(self):
    config = {
      'l': self.l
    }
    config.update(super(StabilityRegularizer, self).get_config())
    return config

  def build(self, input_shape):
    self.built = True

  def call(self, inputs):
    old, new = inputs
    loss = tf.math.reduce_mean(tf.math.squared_difference(old, new))
    self.add_loss(loss * self.l, inputs=True)
    return new