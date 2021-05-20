
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

class ResizeMethod(object):
  """See `tf.image.resize` for details."""
  BILINEAR = 'bilinear'
  NEAREST_NEIGHBOR = 'nearest'
  BICUBIC = 'bicubic'
  AREA = 'area'
  LANCZOS3 = 'lanczos3'
  LANCZOS5 = 'lanczos5'
  GAUSSIAN = 'gaussian'
  MITCHELLCUBIC = 'mitchellcubic'


def resize_(images,
           size,
           method=ResizeMethod.BILINEAR,
           preserve_aspect_ratio=False,
           antialias=False,
           name=None):
  """Resize `images` to `size` using the specified `method`.
  Resized images will be distorted if their original aspect ratio is not
  the same as `size`.  To avoid distortions see
  `tf.image.resize_with_pad`.
  >>> image = tf.constant([
  ...  [1,0,0,0,0],
  ...  [0,1,0,0,0],
  ...  [0,0,1,0,0],
  ...  [0,0,0,1,0],
  ...  [0,0,0,0,1],
  ... ])
  >>> # Add "batch" and "channels" dimensions
  >>> image = image[tf.newaxis, ..., tf.newaxis]
  >>> image.shape.as_list()  # [batch, height, width, channels]
  [1, 5, 5, 1]
  >>> tf.image.resize(image, [3,5])[0,...,0].numpy()
  array([[0.6666667, 0.3333333, 0.       , 0.       , 0.       ],
         [0.       , 0.       , 1.       , 0.       , 0.       ],
         [0.       , 0.       , 0.       , 0.3333335, 0.6666665]],
        dtype=float32)
  It works equally well with a single image instead of a batch of images:
  >>> tf.image.resize(image[0], [3,5]).shape.as_list()
  [3, 5, 1]
  When `antialias` is true, the sampling filter will anti-alias the input image
  as well as interpolate.  When downsampling an image with [anti-aliasing](
  https://en.wikipedia.org/wiki/Spatial_anti-aliasing) the sampling filter
  kernel is scaled in order to properly anti-alias the input image signal.
  `antialias` has no effect when upsampling an image:
  >>> a = tf.image.resize(image, [5,10])
  >>> b = tf.image.resize(image, [5,10], antialias=True)
  >>> tf.reduce_max(abs(a - b)).numpy()
  0.0
  The `method` argument expects an item from the `image.ResizeMethod` enum, or
  the string equivalent. The options are:
  *   <b>`bilinear`</b>: [Bilinear interpolation.](
    https://en.wikipedia.org/wiki/Bilinear_interpolation) If `antialias` is
    true, becomes a hat/tent filter function with radius 1 when downsampling.
  *   <b>`lanczos3`</b>:  [Lanczos kernel](
    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 3.
    High-quality practical filter but may have some ringing, especially on
    synthetic images.
  *   <b>`lanczos5`</b>: [Lanczos kernel] (
    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 5.
    Very-high-quality filter but may have stronger ringing.
  *   <b>`bicubic`</b>: [Cubic interpolant](
    https://en.wikipedia.org/wiki/Bicubic_interpolation) of Keys. Equivalent to
    Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel,
    particularly when upsampling.
  *   <b>`gaussian`</b>: [Gaussian kernel](
    https://en.wikipedia.org/wiki/Gaussian_filter) with radius 3,
    sigma = 1.5 / 3.0.
  *   <b>`nearest`</b>: [Nearest neighbor interpolation.](
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
    `antialias` has no effect when used with nearest neighbor interpolation.
  *   <b>`area`</b>: Anti-aliased resampling with area interpolation.
    `antialias` has no effect when used with area interpolation; it
    always anti-aliases.
  *   <b>`mitchellcubic`</b>: Mitchell-Netravali Cubic non-interpolating filter.
    For synthetic images (especially those lacking proper prefiltering), less
    ringing than Keys cubic kernel but less sharp.
  Note: Near image edges the filtering kernel may be partially outside the
  image boundaries. For these pixels, only input pixels inside the image will be
  included in the filter sum, and the output value will be appropriately
  normalized.
  The return value has type `float32`, unless the `method` is
  `ResizeMethod.NEAREST_NEIGHBOR`, then the return dtype is the dtype
  of `images`:
  >>> nn = tf.image.resize(image, [5,7], method='nearest')
  >>> nn[0,...,0].numpy()
  array([[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1]], dtype=int32)
  With `preserve_aspect_ratio=True`, the aspect ratio is preserved, so `size`
  is the maximum for each dimension:
  >>> max_10_20 = tf.image.resize(image, [10,20], preserve_aspect_ratio=True)
  >>> max_10_20.shape.as_list()
  [1, 10, 10, 1]
  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new
      size for the images.
    method: An `image.ResizeMethod`, or string equivalent.  Defaults to
      `bilinear`.
    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
      then `images` will be resized to a size that fits in `size` while
      preserving the aspect ratio of the original image. Scales up the image if
      `size` is bigger than the current size of the `image`. Defaults to False.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.
    name: A name for this operation (optional).
  Raises:
    ValueError: if the shape of `images` is incompatible with the
      shape arguments to this function
    ValueError: if `size` has an invalid shape or type.
    ValueError: if an unsupported resize method is specified.
  Returns:
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """

  # def resize_fn(images_t, new_size):
  #   """Resize core function, passed to _resize_images_common."""
  #   scale_and_translate_methods = [
  #       ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5, ResizeMethod.GAUSSIAN,
  #       ResizeMethod.MITCHELLCUBIC
  #   ]

  def resize_fn(images_t, new_size):
    """Resize core function, passed to _resize_images_common."""
    scale_and_translate_methods = [
        ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5, ResizeMethod.GAUSSIAN,
        ResizeMethod.MITCHELLCUBIC
    ]

    def resize_with_scale_and_translate(method):
      scale = (
          math_ops.cast(new_size, dtype=dtypes.float32) /
          math_ops.cast(array_ops.shape(images_t)[1:3], dtype=dtypes.float32))
      return gen_image_ops.scale_and_translate(
          images_t,
          new_size,
          scale,
          array_ops.zeros([2]),
          kernel_type=method,
          antialias=antialias)

    if method == ResizeMethod.BILINEAR:
      if antialias:
        return resize_with_scale_and_translate('triangle')
      else:
        return gen_image_ops.resize_bilinear(
            images_t, new_size, align_corners=True)
    elif method == ResizeMethod.NEAREST_NEIGHBOR:
      return gen_image_ops.resize_nearest_neighbor(
          images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.BICUBIC:
      if antialias:
        return resize_with_scale_and_translate('keyscubic')
      else:
        return gen_image_ops.resize_bicubic(
            images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.AREA:
      return gen_image_ops.resize_area(images_t, new_size)
    elif method in scale_and_translate_methods:
      return resize_with_scale_and_translate(method)
    else:
      raise ValueError('Resize method is not implemented: {}'.format(method))

  return _resize_images_common(
      images,
      resize_fn,
      size,
      preserve_aspect_ratio=preserve_aspect_ratio,
      name=name,
      skip_resize_if_same=False)

def _ImageDimensions(image, rank):
  """Returns the dimensions of an image tensor.
  Args:
    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
    rank: The expected rank of the image
  Returns:
    A list of corresponding to the dimensions of the
    input image.  Dimensions that are statically known are python integers,
    otherwise, they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)
    return [
        s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
    ]



def _resize_images_common(images, resizer_fn, size, preserve_aspect_ratio, name,
                          skip_resize_if_same):
  """Core functionality for v1 and v2 resize functions."""
  with ops.name_scope(name, 'resize', [images, size]):
    images = ops.convert_to_tensor(images, name='images')
    if images.get_shape().ndims is None:
      raise ValueError('\'images\' contains no shape.')
    # TODO(shlens): Migrate this functionality to the underlying Op's.
    is_batch = True
    if images.get_shape().ndims == 3:
      is_batch = False
      images = array_ops.expand_dims(images, 0)
    elif images.get_shape().ndims != 4:
      raise ValueError('\'images\' must have either 3 or 4 dimensions.')

    _, height, width, _ = images.get_shape().as_list()

    try:
      size = ops.convert_to_tensor(size, dtypes.int32, name='size')
    except (TypeError, ValueError):
      raise ValueError('\'size\' must be a 1-D int32 Tensor')
    if not size.get_shape().is_compatible_with([2]):
      raise ValueError('\'size\' must be a 1-D Tensor of 2 elements: '
                       'new_height, new_width')

    if preserve_aspect_ratio:
      # Get the current shapes of the image, even if dynamic.
      _, current_height, current_width, _ = _ImageDimensions(images, rank=4)

      # do the computation to find the right scale and height/width.
      scale_factor_height = (
          math_ops.cast(size[0], dtypes.float32) /
          math_ops.cast(current_height, dtypes.float32))
      scale_factor_width = (
          math_ops.cast(size[1], dtypes.float32) /
          math_ops.cast(current_width, dtypes.float32))
      scale_factor = math_ops.maximum(scale_factor_height, scale_factor_width)
      scaled_height_const = math_ops.cast(
          math_ops.round(scale_factor *
                         math_ops.cast(current_height, dtypes.float32)),
          dtypes.int32)
      scaled_width_const = math_ops.cast(
          math_ops.round(scale_factor *
                         math_ops.cast(current_width, dtypes.float32)),
          dtypes.int32)

      # NOTE: Reset the size and other constants used later.
      size = ops.convert_to_tensor([scaled_height_const, scaled_width_const],
                                   dtypes.int32,
                                   name='size')

    size_const_as_shape = tensor_util.constant_value_as_shape(size)
    new_height_const = tensor_shape.dimension_at_index(size_const_as_shape,
                                                       0).value
    new_width_const = tensor_shape.dimension_at_index(size_const_as_shape,
                                                      1).value

    # If we can determine that the height and width will be unmodified by this
    # transformation, we avoid performing the resize.
    if skip_resize_if_same and all(
        x is not None
        for x in [new_width_const, width, new_height_const, height]) and (
            width == new_width_const and height == new_height_const):
      if not is_batch:
        images = array_ops.squeeze(images, axis=[0])
      return images

    images = resizer_fn(images, size)

    # NOTE(mrry): The shape functions for the resize ops cannot unpack
    # the packed values in `new_size`, so set the shape here.
    images.set_shape([None, new_height_const, new_width_const, None])

    if not is_batch:
      images = array_ops.squeeze(images, axis=[0])
    return images

