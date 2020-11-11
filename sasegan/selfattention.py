from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from contextlib import contextmanager
import numpy as np

def _l2normalize(v, eps=1e-12):
  """l2 normize the input vector."""
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(weights, num_iters=1, update_collection=None, with_sigma=False):
  """Performs Spectral Normalization on a weight tensor.
  Specifically it divides the weight tensor by its largest singular value. This
  is intended to stabilize GAN training, by making the discriminator satisfy a
  local 1-Lipschitz constraint.
  Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
  [sn-gan] https://openreview.net/pdf?id=B1QRgziT-
  Args:
    weights: The weight tensor which requires spectral normalization
    num_iters: Number of SN iterations.
    update_collection: The update collection for assigning persisted variable u.
                       If None, the function will update u during the forward
                       pass. Else if the update_collection equals 'NO_OPS', the
                       function will not update the u during the forward. This
                       is useful for the discriminator, since it does not update
                       u in the second pass.
                       Else, it will put the assignment in a collection
                       defined by the user. Then the user need to run the
                       assignment explicitly.
    with_sigma: For debugging purpose. If True, the fuction returns
                the estimated singular value for the weight tensor.
  Returns:
    w_bar: The normalized weight tensor
    sigma: The estimated singular value for the weight tensor.
  """
  w_shape = weights.shape.as_list()
  w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
  u = tf.get_variable('u', [1, w_shape[-1]],
                      initializer=tf.truncated_normal_initializer(),
                      trainable=False)
  u_ = u
  for _ in range(num_iters):
    v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
    u_ = _l2normalize(tf.matmul(v_, w_mat))

  sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
  w_mat /= sigma
  if update_collection is None:
    with tf.control_dependencies([u.assign(u_)]):
      w_bar = tf.reshape(w_mat, w_shape)
  else:
    w_bar = tf.reshape(w_mat, w_shape)
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_))
  if with_sigma:
    return w_bar, sigma
  else:
    return w_bar

def conv1x1(input_, output_dim, init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
  k_h = 1
  k_w = 1
  d_h = 1
  d_w = 1
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    w_bar = spectral_normed_weight(w, num_iters=1, update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_non_local_block_sim(x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name):
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num // 4
    #downsampled_num = location_num

    # theta path
    theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
    theta = tf.reshape(theta, [batch_size, location_num, num_channels // 8])

    # phi path
    phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
    phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[4, 1], strides=[4,1])
    phi = tf.reshape(phi, [batch_size, downsampled_num, num_channels // 8])


    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)
    print(tf.reduce_sum(attn, axis=-1))

    # g path
    g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[4, 1], strides=[4,1])
    g = tf.reshape(g, [batch_size, downsampled_num, num_channels // 2])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
    sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')
    return x + sigma * attn_g


def snconv2d(input_, output_dim,
             k_h=3, k_w=3, d_h=1, d_w=1,
             sn_iters=1, update_collection=None, name='snconv2d'):
  """Creates a spectral normalized (SN) convolutional layer.
  Args:
    input_: 4D input tensor (batch size, height, width, channel).
    output_dim: Number of features in the output layer.
    k_h: The height of the convolutional kernel.
    k_w: The width of the convolutional kernel.
    d_h: The height stride of the convolutional kernel.
    d_w: The width stride of the convolutional kernel.
    sn_iters: The number of SN iterations.
    update_collection: The update collection used in spectral_normed_weight.
    name: The name of the variable scope.
  Returns:
    conv: The normalized tensor.
  """
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=tf.contrib.layers.xavier_initializer())
    w_bar = spectral_normed_weight(w, num_iters=sn_iters,
                                   update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    biases = tf.get_variable('biases', [output_dim],
                             initializer=tf.zeros_initializer())
    conv = tf.nn.bias_add(conv, biases)
    return conv

def sn_downconv(x,
           output_dim,
           kwidth=5,
           pool=2,
           init=None,
           uniform=False,
           bias_init=None,
           name='downconv'):
  """ Downsampled convolution 1d """
  x2d = tf.expand_dims(x, 2)
  w_init = init
  if w_init is None:
      w_init = xavier_initializer(uniform=uniform)
  with tf.variable_scope(name):
      W = tf.get_variable(
          'W', [kwidth, 1, x.get_shape()[-1], output_dim],
          initializer=w_init)
      W_bar = spectral_normed_weight(W)
      conv = tf.nn.conv2d(x2d, W_bar, strides=[1, pool, 1, 1], padding='SAME')
      if bias_init is not None:
          b = tf.get_variable('b', [output_dim], initializer=bias_init)
          conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
      else:
          conv = tf.reshape(conv, conv.get_shape())
      # reshape back to 1d
      conv = tf.reshape(
          conv,
          conv.get_shape().as_list()[:2] + [conv.get_shape().as_list()[-1]])
      return conv


def sn_deconv(x,
           output_shape,
           kwidth=5,
           dilation=2,
           init=None,
           uniform=False,
           bias_init=None,
           name='deconv1d'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    out_channels = output_shape[-1]
    assert len(input_shape) >= 3
    # reshape the tensor to use 2d operators
    x2d = tf.expand_dims(x, 2)
    o2d = output_shape[:2] + [1] + [output_shape[-1]]
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        # filter shape: [kwidth, output_channels, in_channels]
        W = tf.get_variable('W', [kwidth, 1, out_channels, in_channels], initializer=w_init)
        W_bar = spectral_normed_weight(W)
        try:
            deconv = tf.nn.conv2d_transpose(x2d, W_bar, output_shape=o2d, strides=[1, dilation, 1, 1])
        except AttributeError:
            # support for versions of TF before 0.7.0
            # based on https://github.com/carpedm20/DCGAN-tensorflow
            deconv = tf.nn.conv2d_transpose(x2d, W_bar, output_shape=o2d, strides=[1, dilation, 1, 1])
        if bias_init is not None:
            b = tf.get_variable(
                'b', [out_channels], initializer=tf.constant_initializer(0.))
            deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        else:
            deconv = tf.reshape(deconv, deconv.get_shape())
        # reshape back to 1d
        deconv = tf.reshape(deconv, output_shape)
        return deconv