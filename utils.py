# -*- coding: utf-8 -*-

import os
import glob
import h5py
import random
# import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import math
import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
LEVEL_SIZE = 4


def read_data(path):
    """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def preprocess(path, scale=3, ):
    """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """

    image = imread(path, is_grayscale=True)

    label_ = modcrop(image, scale)
    image = (image - 127.5) / 127.5
    label_ = (image - 127.5) / 127.5

    input_ = scipy.ndimage.interpolation.zoom(image, (scale / 1.), prefilter=False)

    return input_, label_


def prepare_data(sess, dataset):
    """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))

    return data


def make_data(sess, data, label, data_dir):
    """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
    if FLAGS.is_train:
        
        savepath = os.path.join('.', os.path.join(data_dir, 'train.h5'))
        if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
            os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    else:
        savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'test.h5'))
        if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
            os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def imread(path, is_grayscale=True):
    """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
    if is_grayscale:
      
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def modcrop(image, scale=3):
    """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def input_setup(sess, config, data_dir, index=0):
    """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
    # Load data path
    if config.is_train:
        data = prepare_data(sess, dataset=data_dir)
    else:
        data = prepare_data(sess, dataset=data_dir)

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6

    if config.is_train:
        for i in range(len(data)):
            input_ = (imread(data[i]) - 127.5) / 127.5
            label_ = input_

            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                    sub_label = label_[int(x + padding):int(x) + int(padding) + int(config.label_size),
                                int(y + padding):int(y + padding + config.label_size)]  # [21 x 21]
                    # Make channel value
                    if data_dir == "Train":
                        sub_input = cv2.resize(sub_input, (config.image_size / 4, config.image_size / 4),
                                               interpolation=cv2.INTER_CUBIC)
                        sub_input = sub_input.reshape([config.image_size / 4, config.image_size / 4, 1])
                        sub_label = cv2.resize(sub_label, (config.label_size / 4, config.label_size / 4),
                                               interpolation=cv2.INTER_CUBIC)
                        sub_label = sub_label.reshape([config.label_size / 4, config.label_size / 4, 1])
                        print('error')
                    else:
                        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                        sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

    else:

        input_ = (imread(data[index]) - 127.5) / 127.5
        if len(input_.shape) == 3:
            h_real, w_real, _ = input_.shape
        else:
            h_real, w_real = input_.shape
        padding_h = config.image_size - ((h_real + padding) % config.label_size)
        padding_w = config.image_size - ((w_real + padding) % config.label_size)
        input_ = np.lib.pad(input_, ((int(padding), int(padding_h)), (int(padding), int(padding_w))), 'edge')
        label_ = input_
        h, w = input_.shape

        nx = ny = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1;
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                sub_label = label_[int(x + padding):int(x) + int(padding) + int(config.label_size),
                            int(y) + int(padding):int(y) + int(padding) + int(config.label_size)]  # [21 x 21]

                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    # print(arrdata.shape)
    make_data(sess, arrdata, arrlabel, data_dir)

    if not config.is_train:
        print(nx, ny)
        print(h_real, w_real)
        return nx, ny, h_real, w_real


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return (img * 127.5 + 127.5)

def gradient(input):
   
    filter = tf.reshape(tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]), [3, 3, 1, 1])
    d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    # print(d)
    return d


def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.compat.v1.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.compat.v1.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                                          trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1

        u_hat, v_hat, _ = power_iteration(u, iteration)

        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))

        w_mat = w_mat / sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not (update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))

            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
    return input_x_norm

def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    k1 = 0.01
    k2 = 0.03
    L = 1  # depth of image (255 in case the image has a different scale)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    value = tf.reduce_mean(ssim_map, axis=[1, 2, 3])
    return value


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


# Dilation Residual dense block (DRDB)
def DRDB(input_layer):
    with tf.compat.v1.variable_scope('dilation_conv1'):
        weights_1 = tf.compat.v1.get_variable("w1", [3, 3, 64, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_1 = tf.compat.v1.get_variable("b1", [16], initializer=tf.constant_initializer(0.0))
        conv_1 = tf.nn.atrous_conv2d(input_layer, weights_1, 2, padding='SAME') + bias_1
        out_1 = lrelu(conv_1)
        concat_1 = tf.concat([input_layer, out_1], axis=-1)
    with tf.compat.v1.variable_scope('dilation_conv2'):
        weights_2 = tf.compat.v1.get_variable("w2", [3, 3, 80, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_2 = tf.compat.v1.get_variable("b2", [16], initializer=tf.constant_initializer(0.0))
        conv_2 = tf.nn.atrous_conv2d(concat_1, weights_2, 2, padding='SAME') + bias_2
        out_2 = lrelu(conv_2)
        concat_2 = tf.concat([input_layer, out_1, out_2], axis=-1)
    with tf.compat.v1.variable_scope('dilation_conv3'):
        weights_3 = tf.compat.v1.get_variable("w3", [3, 3, 96, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_3 = tf.compat.v1.get_variable("b3", [16], initializer=tf.constant_initializer(0.0))
        conv_3 = tf.nn.atrous_conv2d(concat_2, weights_3, 2, padding='SAME') + bias_3
        out_3 = lrelu(conv_3)
        concat_3 = tf.concat([input_layer, out_1, out_2, out_3], axis=-1)
    with tf.compat.v1.variable_scope('dilation_conv4'):
        weights_4 = tf.compat.v1.get_variable("w4", [1, 1, 112, 64],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_4 = tf.compat.v1.get_variable("b4", [64], initializer=tf.constant_initializer(0.0))
        conv_4 = tf.nn.conv2d(concat_3, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
        out_4 = lrelu(conv_4)
        concat_4 = tf.concat([input_layer, out_4], axis=-1)  # 128

    return concat_4


def DRDB1(input_layer):
    with tf.compat.v1.variable_scope('dilation1_conv1'):
        weights_1 = tf.compat.v1.get_variable("w1", [3, 3, 128, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_1 = tf.compat.v1.get_variable("b1", [16], initializer=tf.constant_initializer(0.0))
        conv_1 = tf.nn.atrous_conv2d(input_layer, weights_1, 2, padding='SAME') + bias_1
        out_1 = lrelu(conv_1)
        concat_1 = tf.concat([input_layer, out_1], axis=-1)
    with tf.compat.v1.variable_scope('dilation1_conv2'):
        weights_2 = tf.compat.v1.get_variable("w2", [3, 3, 144, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_2 = tf.compat.v1.get_variable("b2", [16], initializer=tf.constant_initializer(0.0))
        conv_2 = tf.nn.atrous_conv2d(concat_1, weights_2, 2, padding='SAME') + bias_2
        out_2 = lrelu(conv_2)
        concat_2 = tf.concat([input_layer, out_1, out_2], axis=-1)
    with tf.compat.v1.variable_scope('dilation1_conv3'):
        weights_3 = tf.compat.v1.get_variable("w3", [3, 3, 160, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_3 = tf.compat.v1.get_variable("b3", [16], initializer=tf.constant_initializer(0.0))
        conv_3 = tf.nn.atrous_conv2d(concat_2, weights_3, 2, padding='SAME') + bias_3
        out_3 = lrelu(conv_3)
        concat_3 = tf.concat([input_layer, out_1, out_2, out_3], axis=-1)
    with tf.compat.v1.variable_scope('dilation1_conv4'):
        weights_4 = tf.compat.v1.get_variable("w4", [1, 1, 176, 128],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_4 = tf.compat.v1.get_variable("b4", [128], initializer=tf.constant_initializer(0.0))
        conv_4 = tf.nn.conv2d(concat_3, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
        out_4 = lrelu(conv_4)
        concat_4 = tf.concat([input_layer, out_4], axis=-1)  # 256

    return concat_4


def DRDB2(input_layer):
    with tf.compat.v1.variable_scope('dilation2_conv1'):
        weights_1 = tf.compat.v1.get_variable("w1", [3, 3, 256, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_1 = tf.compat.v1.get_variable("b1", [16], initializer=tf.constant_initializer(0.0))
        conv_1 = tf.nn.atrous_conv2d(input_layer, weights_1, 2, padding='SAME') + bias_1
        out_1 = lrelu(conv_1)
        concat_1 = tf.concat([input_layer, out_1], axis=-1)
    with tf.compat.v1.variable_scope('dilation2_conv2'):
        weights_2 = tf.compat.v1.get_variable("w2", [3, 3, 272, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_2 = tf.compat.v1.get_variable("b2", [16], initializer=tf.constant_initializer(0.0))
        conv_2 = tf.nn.atrous_conv2d(concat_1, weights_2, 2, padding='SAME') + bias_2
        out_2 = lrelu(conv_2)
        concat_2 = tf.concat([input_layer, out_1, out_2], axis=-1)
    with tf.compat.v1.variable_scope('dilation2_conv3'):
        weights_3 = tf.compat.v1.get_variable("w3", [3, 3, 288, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_3 = tf.compat.v1.get_variable("b3", [16], initializer=tf.constant_initializer(0.0))
        conv_3 = tf.nn.atrous_conv2d(concat_2, weights_3, 2, padding='SAME') + bias_3
        out_3 = lrelu(conv_3)
        concat_3 = tf.concat([input_layer, out_1, out_2, out_3], axis=-1)
    with tf.compat.v1.variable_scope('dilation2_conv4'):
        weights_4 = tf.compat.v1.get_variable("w4", [1, 1, 304, 256],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias_4 = tf.compat.v1.get_variable("b4", [256], initializer=tf.constant_initializer(0.0))
        conv_4 = tf.nn.conv2d(concat_3, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
        out_4 = lrelu(conv_4)
        concat_4 = tf.concat([input_layer, out_4], axis=-1)  # 512

    return concat_4

# Auxiliary Net
def weight_variable(shape, stddev=None, name='weight'):
    if stddev == None:
        if len(shape) == 4:
            stddev = math.sqrt(2. / (shape[0] * shape[1] * shape[2]))
        else:
            stddev = math.sqrt(2. / shape[0])
    else:
        stddev = 0.1
    initial = tf.random.truncated_normal(shape, stddev=stddev)
    W = tf.Variable(initial, name=name)

    return W


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm(inputs, training):
    return tf.layers.batch_normalization(
        inputs=inputs, axis=-1,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)

# Attention guided block
def Attention_gb(x1, x2, c1):
    with tf.compat.v1.variable_scope('attention'):
        xc_1 = tf.concat([x1, c1], axis=-1)
        xc_2 = tf.concat([x2, c1], axis=-1)
        weights_1 = tf.compat.v1.get_variable("w_a_1", [3, 3, 32, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        weights_2 = tf.compat.v1.get_variable("w_a_2", [3, 3, 32, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        weights_3 = tf.compat.v1.get_variable("w_a_3", [3, 3, 16, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        weights_4 = tf.compat.v1.get_variable("w_a_4", [3, 3, 16, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        weights_5 = tf.compat.v1.get_variable("w_a_5", [3, 3, 48, 16],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))

        bias_1 = tf.compat.v1.get_variable("b_a_1", [16], initializer=tf.constant_initializer(0.0))
        bias_2 = tf.compat.v1.get_variable("b_a_2", [16], initializer=tf.constant_initializer(0.0))
        bias_3 = tf.compat.v1.get_variable("b_a_3", [16], initializer=tf.constant_initializer(0.0))
        bias_4 = tf.compat.v1.get_variable("b_a_4", [16], initializer=tf.constant_initializer(0.0))
        bias_5 = tf.compat.v1.get_variable("b_a_5", [16], initializer=tf.constant_initializer(0.0))

        conv1_xc = tf.nn.conv2d(xc_1, weights_1, strides=[1, 1, 1, 1], padding='SAME') + bias_1
        conv1_xc = lrelu(conv1_xc)
        conv1_xc = tf.nn.conv2d(conv1_xc, weights_3, strides=[1, 1, 1, 1], padding='SAME') + bias_2
        conv1_xc = lrelu(conv1_xc)
        conv1_xc = tf.nn.sigmoid(conv1_xc, name='attention2')

        conv2_xc = tf.nn.conv2d(xc_2, weights_2, strides=[1, 1, 1, 1], padding='SAME') + bias_3
        conv2_xc = lrelu(conv2_xc)
        conv2_xc = tf.nn.conv2d(conv2_xc, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
        conv2_xc = lrelu(conv2_xc)
        conv2_xc = tf.nn.sigmoid(conv2_xc, name='attention3')

        conv1_xc_a = x1 * conv1_xc
        conv2_xc_a = x2 * conv2_xc

        concat_xc = tf.concat([conv1_xc_a, conv2_xc_a, c1], axis=-1)
        conv5_xc = tf.contrib.layers.batch_norm(
            tf.nn.conv2d(concat_xc, weights_5, strides=[1, 1, 1, 1], padding='SAME') + bias_5, decay=0.9,
            updates_collections=None, epsilon=1e-5, scale=True)

        return conv5_xc


def gradient_soomth(input_tensor, direction):
    global kernel_n
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])

    if direction == "x":
        kernel_n = smooth_kernel_x
    elif direction == "y":
        kernel_n = smooth_kernel_y
    return tf.abs(tf.nn.conv2d(input_tensor, kernel_n, strides=[1, 1, 1, 1], padding='SAME'))


def ave_gradient(input_tensor, direction):
    return tf.layers.average_pooling2d(gradient_soomth(input_tensor, direction), pool_size=3, strides=1, padding='SAME')


def smooth_loss(input, output):
    return gradient_soomth(input, "x") * tf.exp(-10 * ave_gradient(output, "x")) + gradient_soomth(input, "y") * tf.exp(
            -10 * ave_gradient(output, "y"))

def channel_shuffle(feature, group=2):
  channel_num = feature.shape[-1]
  if channel_num % group != 0:
    raise ValueError("The group must be divisible by the shape of the last dimension of the feature.")
  x = tf.reshape(feature, shape=(-1, tf.shape(feature)[1], tf.shape(feature)[2], group, channel_num // group))
  x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
  x = tf.reshape(x, shape=(-1,  tf.shape(feature)[1],  tf.shape(feature)[2], channel_num))
  return x


def NonLocalBlock_t1(input_ir, output_channels, sub_sample=False, is_bn=True, is_training=True, scope="NonLocalBlock"):
    batch_size, height, width, in_channels = input_ir.get_shape().as_list()
    
    with tf.compat.v1.variable_scope("non_conv"):

        weights_1 = tf.compat.v1.get_variable("w_a_1", initializer=tf.constant(reader.get_tensor('fusion_model/non_local1/non_conv/w_a_1')))
        weights_2 = tf.compat.v1.get_variable("w_a_2", initializer=tf.constant(reader.get_tensor('fusion_model/non_local1/non_conv/w_a_2')))
        weights_3 = tf.compat.v1.get_variable("w_a_3", initializer=tf.constant(reader.get_tensor('fusion_model/non_local1/non_conv/w_a_3')))
        

        bias_1 = tf.compat.v1.get_variable("b_a_1", initializer=tf.constant(reader.get_tensor('fusion_model/non_local1/non_conv/b_a_1')))
        bias_2 = tf.compat.v1.get_variable("b_a_2", initializer=tf.constant(reader.get_tensor('fusion_model/non_local1/non_conv/b_a_2')))
        bias_3 = tf.compat.v1.get_variable("b_a_3", initializer=tf.constant(reader.get_tensor('fusion_model/non_local1/non_conv/b_a_3')))

        conv_g = tf.nn.conv2d(input_ir, weights_1, strides=[1, 1, 1, 1], padding='SAME') + bias_1
        conv_phi = tf.nn.conv2d(input_ir, weights_2, strides=[1, 1, 1, 1], padding='SAME') + bias_2
        conv_theta = tf.nn.conv2d(input_ir, weights_3, strides=[1, 1, 1, 1], padding='SAME') + bias_3

        #g_x = tf.reshape(g, [-1, output_channels, height * width])
        g_x = tf.reshape(conv_g, [tf.shape(conv_g)[0], tf.shape(conv_g)[1] * tf.shape(conv_g)[2], tf.shape(conv_g)[3]])
        #g_x = tf.transpose(g_x, [0, 2, 1])
        #print(g_x.shape)

        phi_x = tf.reshape(conv_phi, [tf.shape(conv_phi)[0], tf.shape(conv_phi)[3], tf.shape(conv_phi)[1] * tf.shape(conv_phi)[2]])
        #print(phi_x.shape)

        #theta_x = tf.reshape(theta, [-1, output_channels, height * width])
        #theta_x = tf.transpose(theta_x, [0, 2, 1])
        theta_x = tf.reshape(conv_theta, [tf.shape(conv_theta)[0], tf.shape(conv_theta)[1] * tf.shape(conv_theta)[2], tf.shape(conv_theta)[3]])
        print(theta_x.shape)

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)      
        y = tf.matmul(f_softmax, g_x)

        y = tf.reshape(y, [tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], output_channels])

        with tf.compat.v1.variable_scope("w"):
            weights_4 = tf.compat.v1.get_variable("w_a_4", initializer=tf.constant(reader.get_tensor('fusion_model/non_local1/non_conv/w/w_a_4')))
            bias_4 = tf.compat.v1.get_variable("b_a_4", initializer=tf.constant(reader.get_tensor('fusion_model/non_local1/non_conv/w/b_a_4')))
            w_y = tf.nn.conv2d(input_ir, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
            if is_bn:
                w_y= tf.layers.batch_normalization(w_y, axis=3, training=is_training)    ### batch_normalization
        z = input_ir + w_y

        return z
        
def NonLocalBlock_t2(input_ir, output_channels, sub_sample=False, is_bn=True, is_training=True, scope="NonLocalBlock"):
    batch_size, height, width, in_channels = input_ir.get_shape().as_list()
    
    with tf.compat.v1.variable_scope("non_conv"):

        weights_1 = tf.compat.v1.get_variable("w_a_1", initializer=tf.constant(reader.get_tensor('fusion_model/non_local2/non_conv/w_a_1')))
        weights_2 = tf.compat.v1.get_variable("w_a_2", initializer=tf.constant(reader.get_tensor('fusion_model/non_local2/non_conv/w_a_2')))
        weights_3 = tf.compat.v1.get_variable("w_a_3", initializer=tf.constant(reader.get_tensor('fusion_model/non_local2/non_conv/w_a_3')))
        

        bias_1 = tf.compat.v1.get_variable("b_a_1", initializer=tf.constant(reader.get_tensor('fusion_model/non_local2/non_conv/b_a_1')))
        bias_2 = tf.compat.v1.get_variable("b_a_2", initializer=tf.constant(reader.get_tensor('fusion_model/non_local2/non_conv/b_a_2')))
        bias_3 = tf.compat.v1.get_variable("b_a_3", initializer=tf.constant(reader.get_tensor('fusion_model/non_local2/non_conv/b_a_3')))

        conv_g = tf.nn.conv2d(input_ir, weights_1, strides=[1, 1, 1, 1], padding='SAME') + bias_1
        conv_phi = tf.nn.conv2d(input_ir, weights_2, strides=[1, 1, 1, 1], padding='SAME') + bias_2
        conv_theta = tf.nn.conv2d(input_ir, weights_3, strides=[1, 1, 1, 1], padding='SAME') + bias_3

        #g_x = tf.reshape(g, [-1, output_channels, height * width])
        g_x = tf.reshape(conv_g, [tf.shape(conv_g)[0], tf.shape(conv_g)[1] * tf.shape(conv_g)[2], tf.shape(conv_g)[3]])
        #g_x = tf.transpose(g_x, [0, 2, 1])
        #print(g_x.shape)

        phi_x = tf.reshape(conv_phi, [tf.shape(conv_phi)[0], tf.shape(conv_phi)[3], tf.shape(conv_phi)[1] * tf.shape(conv_phi)[2]])
        #print(phi_x.shape)

        #theta_x = tf.reshape(theta, [-1, output_channels, height * width])
        #theta_x = tf.transpose(theta_x, [0, 2, 1])
        theta_x = tf.reshape(conv_theta, [tf.shape(conv_theta)[0], tf.shape(conv_theta)[1] * tf.shape(conv_theta)[2], tf.shape(conv_theta)[3]])
        print(theta_x.shape)

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)      
        y = tf.matmul(f_softmax, g_x)

        y = tf.reshape(y, [tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], output_channels])

        with tf.compat.v1.variable_scope("w"):
            weights_4 = tf.compat.v1.get_variable("w_a_4", initializer=tf.constant(reader.get_tensor('fusion_model/non_local2/non_conv/w/w_a_4')))
            bias_4 = tf.compat.v1.get_variable("b_a_4", initializer=tf.constant(reader.get_tensor('fusion_model/non_local2/non_conv/w/b_a_4')))
            w_y = tf.nn.conv2d(input_ir, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
            if is_bn:
                w_y= tf.layers.batch_normalization(w_y, axis=3, training=is_training)    ### batch_normalization
        z = input_ir + w_y

        return z
        
def NonLocalBlock(input_ir, output_channels, sub_sample=False, is_bn=True, is_training=True, scope="NonLocalBlock"):
    batch_size, height, width, in_channels = input_ir.get_shape().as_list()
    
    with tf.compat.v1.variable_scope("non_conv"):

        weights_1 = tf.compat.v1.get_variable("w_a_1", [1, 1, 32, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        weights_2 = tf.compat.v1.get_variable("w_a_2", [1, 1, 32, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        weights_3 = tf.compat.v1.get_variable("w_a_3", [1, 1, 32, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
        

        bias_1 = tf.compat.v1.get_variable("b_a_1", [32], initializer=tf.constant_initializer(0.0))
        bias_2 = tf.compat.v1.get_variable("b_a_2", [32], initializer=tf.constant_initializer(0.0))
        bias_3 = tf.compat.v1.get_variable("b_a_3", [32], initializer=tf.constant_initializer(0.0))

        conv_g = tf.nn.conv2d(input_ir, weights_1, strides=[1, 1, 1, 1], padding='SAME') + bias_1
        conv_phi = tf.nn.conv2d(input_ir, weights_2, strides=[1, 1, 1, 1], padding='SAME') + bias_2
        conv_theta = tf.nn.conv2d(input_ir, weights_3, strides=[1, 1, 1, 1], padding='SAME') + bias_3

        #g_x = tf.reshape(g, [-1, output_channels, height * width])
        g_x = tf.reshape(conv_g, [tf.shape(conv_g)[0], tf.shape(conv_g)[1] * tf.shape(conv_g)[2], tf.shape(conv_g)[3]])
        #g_x = tf.transpose(g_x, [0, 2, 1])
        #print(g_x.shape)

        phi_x = tf.reshape(conv_phi, [tf.shape(conv_phi)[0], tf.shape(conv_phi)[3], tf.shape(conv_phi)[1] * tf.shape(conv_phi)[2]])
        #print(phi_x.shape)

        #theta_x = tf.reshape(theta, [-1, output_channels, height * width])
        #theta_x = tf.transpose(theta_x, [0, 2, 1])
        theta_x = tf.reshape(conv_theta, [tf.shape(conv_theta)[0], tf.shape(conv_theta)[1] * tf.shape(conv_theta)[2], tf.shape(conv_theta)[3]])
        print(theta_x.shape)

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)      
        y = tf.matmul(f_softmax, g_x)

        y = tf.reshape(y, [tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], output_channels])

        with tf.compat.v1.variable_scope("w"):
            weights_4 = tf.compat.v1.get_variable("w_a_4", [1, 1, 32, 32],
                                              initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias_4 = tf.compat.v1.get_variable("b_a_4", [32], initializer=tf.constant_initializer(0.0))
            w_y = tf.nn.conv2d(input_ir, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
            if is_bn:
                w_y= tf.layers.batch_normalization(w_y, axis=3, training=is_training)    ### batch_normalization
        z = input_ir + w_y

        return z