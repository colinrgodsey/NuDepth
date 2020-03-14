import os
import random

import numpy as np
import tensorflow as tf
import pandas as pd

from PIL import Image

from scipy.spatial.transform import Rotation as R

one_m = 5000.0  # 1m
max_m = 8.0

datasets = [
  'rgbd_dataset_freiburg2_pioneer_slam',
  'rgbd_dataset_freiburg2_pioneer_slam2',
  'rgbd_dataset_freiburg2_pioneer_slam3',
  'rgbd_dataset_freiburg3_large_cabinet',
  'rgbd_dataset_freiburg3_long_office_household',
  'rgbd_dataset_freiburg1_room'
]

dataset_data = {}

def crop_to_square(img):
  w = img.shape[-2]
  h = img.shape[-3]
  if h > w:
    img = tf.image.crop_to_bounding_box(img, (h - w) // 2, 0, w, w)
  elif h < w:
    img = tf.image.crop_to_bounding_box(img, 0, (w - h) // 2, h, h)

  return img

def preprocess_img(img, dim=224):
  #TODO: for depth this needs to antialias both channels independantly and differently, then join them
  #the trust channel needs to be a max pool.
  #eh, maybe the whole thing just needs to be max pooled again
  img = crop_to_square(img)
  img = tf.image.central_crop(img, 0.90)
  img = tf.cast(img, tf.float32)
  img = tf.image.resize(
    img, (dim, dim), antialias=True,
    method=tf.image.ResizeMethod.LANCZOS3
  )
  return img

def preprocess_depth(depth):
  missing = tf.cast(depth < 1, tf.float32)
  depth = tf.cast(depth, tf.float32)

  depth += missing * tf.constant(one_m * max_m)
  depth /= one_m  # 0-1 = 0-1m
  depth /= max_m  # 0-1 = 0-maxm
  depth *= 2.0  # 0-2 = 0-maxm
  depth -= 1.0  # -1-1 = 0-maxm

  depth = tf.concat([depth, missing], axis=-1)

  depth = preprocess_img(depth, dim=28)

  # depth *= -1.0  # reverse so max works on min
  # depth = tf.nn.max_pool2d(depth, 8, 8, 'VALID', data_format='NHWC')
  # depth *= -1.0  # dereverse

  depth = tf.cast(depth, tf.float16)

  return depth

def preprocess_rgb(rgb):
  rgb = preprocess_img(rgb)
  rgb /= 255.0
  return tf.cast(rgb, tf.float16)

@tf.function
def preprocess_seq(rgb, depth, vel):
  return (
    preprocess_rgb(rgb),
    preprocess_depth(depth),
    vel
  )

def load_seq_images(seq):
  def load_image(path):
    x = Image.open(path)
    x = np.array(x)
    return x

  def load_pair(pair):
    x, y, v = pair
    return (load_image(x), load_image(y), v)

  x = [load_pair(frame) for frame in seq]
  xs, ys, vs = zip(*x)
  xs = np.stack(xs)
  ys = np.stack(ys)
  vs = np.stack(vs)

  # create single depth dimension
  ys = np.expand_dims(ys, axis=-1)

  return (xs, ys, vs)

def get_train_seq(frames):
  ds = get_dataset()

  frames = min(frames, len(ds) - 1)
  max = len(ds) - frames
  idx = random.randrange(max)
  return ds[idx:idx+frames]

def get_dataset():
  idx = int(random.random() * len(datasets))
  name = datasets[idx]

  if name not in dataset_data:
    print('loading dataset: ' + name)
    data = load_rgdb_set(os.path.expanduser('~') + '/Datasets/' + name)
    dataset_data[name] = data

  return dataset_data[name]

def load_rgdb_set(dir):
  rgb_files = os.listdir(dir + '/rgb/')
  depth_files = os.listdir(dir + '/depth/')

  truths = pd.read_csv(dir + '/groundtruth.txt', sep=" ", header=None, skiprows=3)
  truths.columns = ["ts", "tx", "ty", "tz", 'q1', 'q2', 'q3', 'q4']
  truths0 = [item[1].values for item in truths.iterrows()]
  truths = []
  idx = 0
  for item in truths0:
    idx += 1
    if idx % 10 == 0:
      truths.append(item)

  rgb_files.sort()
  depth_files.sort()

  def closest_depth(rgb):
    rgb_time = float(rgb[:-4])
    closest_d = 1000.0
    closest = None

    for depth in depth_files:
      depth_time = float(depth[:-4])
      d = abs(depth_time - rgb_time)

      if d < closest_d:
        closest_d = d
        closest = depth

    return closest

  def closest_groundtruth(ts):
    closest_d = 1000.0
    closest = None

    for item in truths:
      truth_time = item[0]
      d = abs(truth_time - ts)

      if d < closest_d:
        closest_d = d
        closest = item

    _, x1, x2, x3, q1, q2, q3, q4 = closest
    q = R.from_quat([q1, q2, q3, q4])

    return np.array([x1, x2, x3]), q

  def closest_vel(rgb, dt=0.1):
    ts = float(rgb[:-4])
    p0, q0 = closest_groundtruth(ts - dt)
    p1, q1 = closest_groundtruth(ts)

    # 1m away
    ref_point_center = [0, 0, 1]

    # center forward positions
    c0 = q0.apply(ref_point_center)
    c1 = q1.apply(ref_point_center)

    #TODO: this can probably use the inverse quaternion? (q0 * q1.inv()).apply(ref_point)
    #probably not actually

    # delta of center forward positions, converted back to screen space
    dc = q1.inv().apply(c1 - c0)
    dc = dc[:2]

    # TODO: this still sucks, weird modulo for angles
    # roll at 1m (2pi radians, at 1m)
    r = q1.inv().as_euler('xyz') - q0.inv().as_euler('xyz')
    r = r[:1]

    # delta of physical movement converted back to screen space
    dp = q1.inv().apply(p1 - p0)

    # [rotx, roty, roll, dx, dy, dz]
    return tf.concat([dc, r, dp], 0) / dt  # m/s

  #normally at 30fps, lets reduce to 10fps
  out = []
  i = 0
  for rgb in rgb_files:
    i += 1
    if i % 3 == 0:
      depth = closest_depth(rgb)
      vel = closest_vel(rgb)
      out.append((dir + '/rgb/' + rgb, dir + '/depth/' + depth, vel))

  return out