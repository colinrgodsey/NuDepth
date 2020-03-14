import tensorflow as tf

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if physical_devices:
#  for dev in physical_devices:
#    tf.config.experimental.set_memory_growth(dev, True)

from tensorflow.python.compiler.tensorrt import trt_convert as trt

params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
         precision_mode='FP16',
         maximum_cached_engines=16,
         minimum_segment_size=18)

converter = trt.TrtGraphConverterV2(
  input_saved_model_dir='saved_model/depth-v1',
  conversion_params=params
)
converter.convert()
converter.save('saved_model/depth-v1-trt')
