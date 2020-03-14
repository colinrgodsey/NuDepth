import os
import random
import matplotlib

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K

from PIL import Image, ImageDraw

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
  for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)

from nupic.data.rgbd import get_train_seq, load_seq_images, preprocess_seq, preprocess_rgb, preprocess_depth

from nupic.model import create as create_model

def show(img):
  img = tf.cast(img, tf.float32)

  if len(img.shape) == 4:
    img = tf.squeeze(img, 0)

  if img.shape[-1] == 1:
    #img = tf.broadcast_to(img, img.shape[:-1] + (3,))
    img += 1.0
    img *= 0.5
    img = tf.image.grayscale_to_rgb(img)
  elif img.shape[-1] == 2:
    missing = img[:, :, 1:]
    img = tf.image.grayscale_to_rgb(img[:, :, :1])
    red = tf.concat([
      tf.ones_like(missing),
      tf.zeros_like(missing),
      tf.zeros_like(missing),
    ], axis=-1)

    missing_inv = tf.constant(1.0) - missing

    img += 1.0
    img *= 0.5
    img = img * missing_inv + red * missing

  img = tf.clip_by_value(img, 0, 0.9999)

  plt.figure(figsize=(2,2))
  plt.grid(False)
  plt.axis('off')
  plt.imshow(img)
  plt.show()

def cold_test(n_frames=32, do_save=False):
  dir = "test_frames"

  if do_save:
    if os.path.exists(dir):
      for file in os.listdir(dir):
        os.unlink(dir + '/' + file)
        #print(dir + '/' + file)
    else:
      os.mkdir(dir)

  def create_vec_plot(v, size=45, scale=40):
    hs = int(size / 2)
    img = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)

    def do_draw(v, axis):
      v = v[axis]

      if axis == 0:
        axis = [1, 0]
        fill = "#ff0000"
      elif axis == 1:
        axis = [0, 1]  #TODO: this y value is wrong for some reason?
        fill = "#00ff00"
      else:
        x = 1.0 / np.sqrt(2)
        axis = [x, -x]
        fill = "#0000ff"

      axis = np.array(axis) * v * scale
      axis = np.clip(axis, -hs, hs - 1)

      draw.line([hs, hs, hs + axis[0], hs + axis[1]], fill=fill)

    do_draw(v, 0)
    do_draw(v, 1)
    do_draw(v, 2)
    del draw

    img = np.array(img)

    return img

  def save_frame(i, rgb, pred, real, v):
    def proc(img):
      img = tf.cast(img, tf.float32)

      if len(img.shape) == 4:
        img = tf.squeeze(img, 0)

      if img.shape[-1] == 2:
        missing = img[:, :, 1:]
        img = tf.image.grayscale_to_rgb(img[:, :, :1])
        red = tf.concat([
          tf.ones_like(missing),
          tf.zeros_like(missing),
          tf.zeros_like(missing),
        ], axis=-1)

        missing_inv = tf.constant(1.0) - missing

        img += 1.0
        img *= 0.5
        img = img * missing_inv + red * missing
      else:
        img = tf.broadcast_to(img, img.shape[:-1] + (3,))
        img += 1.0
        img *= 0.5

      img *= 255.0
      img = tf.clip_by_value(img, 0, 255)
      img = tf.pad(img, [[1, 1], [1, 1], [0, 0]])
      return tf.cast(img, tf.uint8)

    pred = proc(pred)
    real = proc(real)
    depth_img = tf.concat([pred, real], axis=1)

    v = tf.squeeze(v, 0)
    v1 = create_vec_plot(v[:3])
    v2 = create_vec_plot(v[3:])
    v_img = tf.concat([v1, v2], axis=0)

    rgb = tf.squeeze(rgb, 0)
    rgb = tf.image.resize(rgb, [58, 58], antialias=True)
    rgb = tf.pad(rgb, [[1, 1], [1, 1], [0, 0]])
    rgb = tf.clip_by_value(rgb * 255.0, 0, 255)
    rgb = tf.cast(rgb, tf.uint8)

    img = tf.concat([depth_img, rgb], axis=0)

    img = tf.concat([img, v_img], axis=1)

    img = Image.fromarray(img.numpy())
    img.save(dir + '/frame' + str(1000 + i) + '.jpg')

  state = initial_state(1)

  # val_ds = core_dataset(n_frames).take(1)
  # for rgbs, depths, vs in val_ds:

  frames = get_train_seq(n_frames)
  n_frames = len(frames)

  losses = []
  l2_diffs = []
  for i in range(n_frames):
    rgb, depth, v = frames[i]
    # rgb = rgbs[i, :, :, :]
    # depth = depths[i, :, :, :]
    # v = vs[i, :]

    rgb, depth, v = load_seq_images([(rgb, depth, v)])

    rgb, depth = preprocess_rgb(rgb), preprocess_depth(depth)

    init_l2 = state[0]

    state, core_loss, acc, depth_ = loss(rgb, depth, v, state, training=False)

    if i > 4:
      losses.append(core_loss)

    final_l2 = state[0]
    l2_diffs.append(tf.reduce_mean(tf.math.abs(final_l2 - init_l2)))

    if do_save:
      save_frame(i, rgb, depth_, depth, v)

  l4h = tf.reduce_sum(tf.cast(state[2] > 0.5, tf.float32))
  l4l = tf.reduce_sum(tf.cast(state[2] < 0.5, tf.float32))
  l4p = l4h * 100.0 / (l4h + l4l)

  l2h = tf.reduce_sum(tf.cast(state[0] > 0.1, tf.float32))
  l2l = tf.reduce_sum(tf.cast(state[0] < 0.1, tf.float32))
  l2p = l2h * 100.0 / (l2h + l2l)

  val_loss = tf.reduce_mean(losses)
  l2_diffs = tf.reduce_mean(l2_diffs)

  show(rgb.numpy())
  show(depth.numpy())
  show(depth_.numpy())

  print('Validation accuracy: ' + str(acc.numpy()))

  print('Validation l2diff: ' + str(l2_diffs.numpy()))
  print('Validation l2 high %: ' + str(l2p.numpy()))
  print('Validation l2 mean: ' + str(tf.reduce_mean(state[0]).numpy()))
  print('Validation l2 std: ' + str(tf.math.reduce_std(state[0]).numpy()))

  print('Validation l4 high %: ' + str(l4p.numpy()))
  print('Validation l4 mean: ' + str(tf.reduce_mean(state[2]).numpy()))
  print('Validation l4 std: ' + str(tf.math.reduce_std(state[2]).numpy()))

  return acc, val_loss

def initial_state(batches):
  a = tf.zeros((batches,) + model.output[1].shape[1:])
  b = tf.zeros((batches,) + model.output[2].shape[1:])
  c = tf.zeros((batches,) + model.output[3].shape[1:])

  return a, b, c

def loss(x, y, telem, state, training=True):
  a, b, c = state

  [y_, a, b, c] = model([x, telem, a, b, c], training=training)
  state = a, b, c

  tf.debugging.check_numerics(y_, "NaN or other nonsense!")
  tf.debugging.check_numerics(a, "bad state numerics a")
  tf.debugging.check_numerics(b, "bad state numerics b")
  tf.debugging.check_numerics(c, "bad state numerics c")

  y = tf.cast(y, tf.float32)

  mask = tf.constant(1.0) - y[:, :, :, 1:]
  y = y[:, :, :, :1]

  return (
    state,
    loss_object(y_true=y * mask, y_pred=y_ * mask) * loss_weight,
    accuracy(y * mask, y_ * mask),
    y_
  )

def do_run_perframe(x, y, telem, state, train):
  with tf.GradientTape() as tape:
    state_, core_loss, acc, _ = loss(x, y, telem, state, training=train)

    losses = [core_loss]
    for loss_tensor in model.losses:
      losses.append(loss_tensor)
    loss_value = tf.reduce_sum(losses)

    l2_diff = tf.math.reduce_mean(tf.math.abs(state[0] - state_[0]))
    #l2_diff = 0.0

  tf.debugging.check_numerics(core_loss, "bad loss numerics")

  if not train:
    acc = tf.constant(0)

  grads = tape.gradient(loss_value, model.trainable_variables)

  print('Step: {}, Initial Loss: {}, Core Loss: {}, Accuracy: {}'.format(
    optimizer.iterations.numpy(),
    loss_value.numpy(),
    core_loss.numpy(),
    acc.numpy()
  ))

  if train:
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return state_, loss_value, acc, l2_diff

def run_epoch_perframe(batch):
  rgb, depth, v = batch
  state = initial_state(rgb.shape[0])
  losses = []
  accs = []
  l2_diffs = []
  #TODO: should we try to runs in reverse? just need to reverse odom too

  rgb = tf.image.random_brightness(rgb, max_delta=32. / 255.)
  rgb = tf.image.random_saturation(rgb, lower=0.5, upper=1.5)
  rgb = tf.image.random_hue(rgb, max_delta=0.2)
  rgb = tf.image.random_contrast(rgb, lower=0.5, upper=1.5)
  rgb = tf.clip_by_value(rgb, 0.0, 1.0)

  for i in range(rgb.shape[1]):
    x = rgb[:, i, :, :, :]
    y = depth[:, i, :, :, :]
    telem = v[:, i, :]
    train = i >= 5
    state, loss_value, acc, l2_diff = do_run_perframe(x, y, telem, state, train)
    losses.append(loss_value)
    accs.append(acc)
    l2_diffs.append(l2_diff)

  return tf.reduce_mean(losses), tf.reduce_mean(acc), tf.reduce_mean(l2_diff), state

def run_epoch_rnn(batch):
  rgb, depth, v = batch
  def step(inputs, states):
    [y_, a, b, c] = model(inputs + list(states), training=True)
    states = a, b, c
    return y_, states

  y = depth[:, -1, :, :, :]
  losses = []
  with tf.GradientTape() as tape:
    y_, _, _ = tf.keras.backend.rnn(
      step, [rgb, v], initial_state(rgb.shape[0])
    )
    losses.append(loss_object(y_true=y, y_pred=y_))
    for loss_tensor in model.losses:
      losses.append(loss_tensor)
    loss_value = tf.reduce_sum(losses)

  acc = accuracy(y, y_)

  grads = tape.gradient(loss_value, model.trainable_variables)

  print('Step: {}, RNN, Initial Loss: {}, Accuracy: {}'.format(
    optimizer.iterations.numpy(),
    loss_value.numpy(),
    acc.numpy()
  ))

  optimizer.apply_gradients(zip(grads, model.trainable_variables))

#TODO: lets figure out how to make this variable number of frames
def core_dataset(num_frames=32):
  return tf.data.Dataset.from_generator(
    lambda: [load_seq_images(get_train_seq(num_frames))],
    output_shapes=((num_frames, 480, 640, 3),
                   (num_frames, 480, 640, 1),
                   (num_frames, 6)),
    output_types=(tf.uint8, tf.uint16, tf.float32)
  ).map(preprocess_seq)

def base_dataset(i):
  # name = os.path.expanduser('~') + '/Datasets/nupic.cache/depth_test' + str(i) + '.cache'
  name = 'depth_test' + str(i) + '.cache'
  out = core_dataset()
  return out.repeat().take(800).cache(name)


random.seed()

#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():

model = create_model()
name = "depth-v1"

print(model.summary())

try:
  print('loading weights from ' + name + '.h5')

  model.load_weights(name + '.h5', by_name=True)
  #model.load_weights(name + '.h5')
  print('weights loaded')
except:
  print('weights failed to load')
  pass

batch_size = 64
loss_weight = 1.0
learning_rate = 2e-4
#learning_rate = 1e-3

checkpoint_after = 10

#loss_object = tf.keras.losses.MeanAbsoluteError()

# TODO: this needs to take the mask in, and probably a downsampled version of the image for SSIM?
# https://github.com/ialhashim/DenseDepth/blob/master/loss.py
def loss_object(y_true, y_pred, theta=0.4):
  # Point-wise depth
  l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

  # Edges
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

  # Structural similarity (SSIM) index
  #l_ssim = K.clip((1 - tf.image.ssim(y_true + 1.0, y_pred + 1.0, 2.0)) * 0.5, 0, 1)
  l_ssim = tf.constant(0.0)

  # Weights
  w1 = 1.0
  w2 = 1.0
  w3 = theta

  return (w1 * K.mean(l_ssim)) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

def accuracy(y, y_):
  d = tf.math.abs((y - y_) / 2.0)
  return 1.0 - tf.math.reduce_mean(d)

optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.5)

model.compile(optimizer)

lr_adjust = ReduceLROnPlateau(verbose=1, patience=5,
                              cooldown=4, factor=0.8,
                              min_lr=1e-7)
lr_adjust.set_model(model)

log_path = "/home/colin/tfb_logs/"
log_path = log_path + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = tf.keras.callbacks.TensorBoard(
  log_path,
  histogram_freq=1,
  profile_batch=3
)
tensorboard.set_model(model)

file_writer = tf.summary.create_file_writer(log_path + "/metrics")

#cold_test(600, do_save=True)
cold_test()

mini_epoch = 1
for i in range(1, 80):
  print('Epoch ' + str(i))

  ds = base_dataset(i % 3)
  ds = ds.shuffle(batch_size * 2).batch(batch_size).prefetch(1)

  runs = 0
  losses = []
  accs = []
  l2_diffs = []
  log_values = {}

  for batch in ds:
    runs += 1

    tensorboard.on_train_begin(log_values)

    with K.learning_phase_scope(1):
      loss_value, acc_value, l2_diff, state = run_epoch_perframe(batch)
      losses.append(loss_value)
      accs.append(acc_value)
      l2_diffs.append(l2_diff)
      #run_epoch_rnn(batch)

    tensorboard.on_train_batch_end(runs, log_values)

    if runs % checkpoint_after == 0 or (i is 1 and runs is 1):
      mini_epoch += 1

      print('checkpoint for mini epoch ' + str(mini_epoch))

      model.save(name + '.h5')
      #tf.saved_model.save(model, 'saved_model/' + name)
      model.save('saved_model/' + name)

      tensorboard.on_test_begin(log_values)

      with K.learning_phase_scope(0):
        val_acc, val_loss = cold_test()

      tensorboard.on_test_batch_end(runs, log_values)

      epoch_loss = tf.reduce_mean(losses)
      losses = []

      epoch_acc = tf.reduce_mean(accs)
      accs = []

      epoch_l2_diffs = tf.reduce_mean(l2_diffs)
      l2_diffs = []

      epoch_loss = loss_value.numpy()
      epoch_acc = epoch_acc.numpy()
      val_loss = val_loss.numpy()

      print('Avg val loss: ' + str(val_loss))
      print('Avg epoch loss: ' + str(epoch_loss))

      with file_writer.as_default():
        # first item in the batch state
        tf.summary.histogram('l2/output', state[0][0], step=mini_epoch)
        tf.summary.histogram('l4/output', state[2][0], step=mini_epoch)

      log_values.update({
        'val_loss': val_loss,
        'val_acc': val_acc,
        'loss': epoch_loss,
        'acc': epoch_acc,
        #'l2': state[0],
        #'l4': state[2],
        #'l2diff': l2_diffs
      })

      lr_adjust.on_epoch_end(mini_epoch, log_values)
      tensorboard.on_epoch_end(mini_epoch, log_values)


#TODO: use experimentan run v2 in https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy?version=stable for mirror strat

#
# def rust_dist():
#   strategy = tf.distribute.MirroredStrategy()
#   with strategy.scope():
#     @tf.function
#     def distribute_train_epoch(dataset):
#       def replica_fn(input):
#         # process input and return result
#         return result
#
#       total_result = 0
#       for x in dataset:
#         per_replica_result = strategy.experimental_run_v2(replica_fn, args=(x,))
#         total_result += strategy.reduce(tf.distribute.ReduceOp.SUM,
#                                            per_replica_result, axis=None)
#       return total_result
#
#     dist_dataset = strategy.experimental_distribute_dataset(dataset)
#     for _ in range(EPOCHS):
#       train_result = distribute_train_epoch(dist_dataset)
