# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
from model import CGAN
import cv2
from utils import (
    Attention_gb,
    DRDB,
    DRDB1,
    DRDB2,
    NonLocalBlock,
    NonLocalBlock_t2,
    channel_shuffle
)
# from non_local import NonLocalBlock
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def imread(path, is_grayscale=True):
    """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
    if is_grayscale:

        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def fusion_model(img_ir, img_vi):
    ####################  Layer1  ###########################
    with tf.compat.v1.variable_scope('fusion_model'):
        ####################  Concat layer_1  ###########################
        with tf.compat.v1.variable_scope('concat'):
            concat_iv = tf.concat([img_ir, img_vi], axis=-1)
        with tf.compat.v1.variable_scope('conv_iv'):
            weights_iv = tf.compat.v1.get_variable("w_iv", initializer=tf.constant(reader.get_tensor('fusion_model/conv_iv/w_iv')))
            bias_iv = tf.compat.v1.get_variable("b_iv", initializer=tf.constant(reader.get_tensor('fusion_model/conv_iv/b_iv')))
            conv_iv = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(concat_iv, weights_iv, strides=[1, 1, 1, 1], padding='SAME') + bias_iv, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv_iv = lrelu(conv_iv)
        with tf.compat.v1.variable_scope('conv_i'):
            weights_i = tf.compat.v1.get_variable("w_i", initializer=tf.constant(reader.get_tensor('fusion_model/conv_i/w_i')))
            bias_i = tf.compat.v1.get_variable("b_i", initializer=tf.constant(reader.get_tensor('fusion_model/conv_i/b_i')))
            conv_i = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img_ir, weights_i, strides=[1, 1, 1, 1], padding='SAME') + bias_i, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv_i = lrelu(conv_i)
        with tf.compat.v1.variable_scope('conv_v'):
            weights_v = tf.compat.v1.get_variable("w_v", initializer=tf.constant(reader.get_tensor('fusion_model/conv_v/w_v')))
            bias_v = tf.compat.v1.get_variable("b_iv", initializer=tf.constant(reader.get_tensor('fusion_model/conv_v/b_iv')))
            conv_v = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img_vi, weights_v, strides=[1, 1, 1, 1], padding='SAME') + bias_v, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv_v = lrelu(conv_v)
        ##auxiliary net   
        
        with tf.compat.v1.variable_scope('conv_ai'):
                weights_aiv = tf.compat.v1.get_variable("w_aiv", initializer=tf.constant(reader.get_tensor('fusion_model/conv_ai/w_aiv')))
                bias_aiv = tf.compat.v1.get_variable("b_aiv", initializer=tf.constant(reader.get_tensor('fusion_model/conv_ai/b_aiv')))
                conv_aiv = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv_i, weights_aiv, strides=[1, 1, 1, 1], padding='SAME') + bias_aiv, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv_ai = lrelu(conv_aiv)
        with tf.compat.v1.variable_scope('conv_av'):
                weights_ai = tf.compat.v1.get_variable("w_ai", initializer=tf.constant(reader.get_tensor('fusion_model/conv_av/w_ai')))
                bias_ai = tf.compat.v1.get_variable("b_ai", initializer=tf.constant(reader.get_tensor('fusion_model/conv_av/b_ai')))
                conv_ai = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv_v, weights_ai, strides=[1, 1, 1, 1], padding='SAME') + bias_ai, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv_av = lrelu(conv_ai)
                
        ##auxiliary net   
        with tf.compat.v1.variable_scope('non_local1'):
                c_1 = NonLocalBlock(conv_i, conv_v, 16)
        with tf.compat.v1.variable_scope('non_local2'):
                c_2 = NonLocalBlock(conv_v, conv_i, 16)
        with tf.compat.v1.variable_scope('non_local_c'):
                c_c = tf.concat([c_1, c_2], axis=-1)
        
        ####################  Attention guided densenet  ###########################
        with tf.compat.v1.variable_scope('attention_layer1'):
            Block_1 = Attention_gb(conv_i, conv_v, conv_iv)
            concat_1 = tf.concat([Block_1, conv_iv], axis=-1)
        with tf.compat.v1.variable_scope('A_conv'):
            weights_1 = tf.compat.v1.get_variable("A_conv_w", initializer=tf.constant(reader.get_tensor('fusion_model/A_conv/A_conv_w')))
            bias_1 = tf.compat.v1.get_variable("A_conv_b", initializer=tf.constant(reader.get_tensor('fusion_model/A_conv/A_conv_b')))
            conv1 = tf.nn.conv2d(concat_1, weights_1, strides=[1, 1, 1, 1], padding='SAME') + bias_1
            conv1 = lrelu(conv1)
        with tf.compat.v1.variable_scope('attention_layer2'):
            Block_2 = Attention_gb(conv_i, conv_v, conv1)
            concat_2 = tf.concat([Block_2, conv1, conv_iv], axis=-1)
        with tf.compat.v1.variable_scope('A_conv2'):
            weights_2 = tf.compat.v1.get_variable("A_conv2_w", initializer=tf.constant(reader.get_tensor('fusion_model/A_conv2/A_conv2_w')))
            bias_2 = tf.compat.v1.get_variable("A_conv2_b", initializer=tf.constant(reader.get_tensor('fusion_model/A_conv2/A_conv2_b')))
            conv2 = tf.nn.conv2d(concat_2, weights_2, strides=[1, 1, 1, 1], padding='SAME') + bias_2
            conv2 = lrelu(conv2)
        with tf.compat.v1.variable_scope('attention_layer3'):
            Block_2 = Attention_gb(conv_i, conv_v, conv2)
            concat_3 = tf.concat([conv2, Block_2, conv1, conv_iv], axis=-1)
        with tf.compat.v1.variable_scope('A_conv3'):
            weights_3 = tf.compat.v1.get_variable("A_conv3_w", initializer=tf.constant(reader.get_tensor('fusion_model/A_conv3/A_conv3_w')))
            bias_3 = tf.compat.v1.get_variable("A_conv3_b", initializer=tf.constant(reader.get_tensor('fusion_model/A_conv3/A_conv3_b')))
            conv3 = tf.nn.conv2d(concat_3, weights_3, strides=[1, 1, 1, 1], padding='SAME') + bias_3
            conv3 = lrelu(conv3)
        with tf.compat.v1.variable_scope('attention_layer4'):
            Block_2 = Attention_gb(conv_i, conv_v, conv3)
            concat_4 = tf.concat([conv3, conv2, Block_2, conv1, conv_iv], axis=-1)
        with tf.compat.v1.variable_scope('A_conv4'):
            weights_4 = tf.compat.v1.get_variable("A_conv4_w", initializer=tf.constant(reader.get_tensor('fusion_model/A_conv4/A_conv4_w')))
            bias_4 = tf.compat.v1.get_variable("A_conv4_b", initializer=tf.constant(reader.get_tensor('fusion_model/A_conv4/A_conv4_b')))
            conv4 = tf.nn.conv2d(concat_4, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
            conv4 = lrelu(conv4)
        with tf.compat.v1.variable_scope('attention_layer5'):
            Block_2 = Attention_gb(conv_i, conv_v, conv4)
            concat_3 = tf.concat([conv4, conv3, conv2, Block_2, conv1, conv_iv], axis=-1)  # 96
        ####################  Concat layer_2   ###########################
        with tf.compat.v1.variable_scope('concat2'):
            concat_ca = tf.concat([concat_3, c_c], axis=-1)  
        ####################  Conv_1   ###########################
        with tf.compat.v1.variable_scope('conv_1'):
            weights_conv_1 = tf.compat.v1.get_variable("w_conv_1", initializer=tf.constant(reader.get_tensor('fusion_model/conv_1/w_conv_1')))
            bias_conv_1 = tf.compat.v1.get_variable("b_conv_1", initializer=tf.constant(reader.get_tensor('fusion_model/conv_1/b_conv_1')))
            conv_1 = tf.nn.conv2d(concat_ca, weights_conv_1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_1
            conv_1 = lrelu(conv_1)
        ####################  Merging layers   ###########################
        with tf.compat.v1.variable_scope('merging'):
            concat_merg_1 = DRDB(conv_1)  # 128
            concat_merg_2 = DRDB1(concat_merg_1)  # 256
            concat_merg_3 = DRDB2(concat_merg_2)  # 512
            concat_m = tf.concat([concat_merg_1, concat_merg_2], axis=-1)  # 896
        with tf.compat.v1.variable_scope('m_conv_1'):
            weights_m_1 = tf.compat.v1.get_variable("w_m_1", initializer=tf.constant(reader.get_tensor('fusion_model/m_conv_1/w_m_1')))
            bias_m_1 = tf.compat.v1.get_variable("b_m_1", initializer=tf.constant(reader.get_tensor('fusion_model/m_conv_1/b_m_1')))
            conv_m = tf.nn.conv2d(concat_merg_1, weights_m_1, strides=[1, 1, 1, 1], padding='SAME') + bias_m_1
            conv_m = lrelu(conv_m)
        ####################  Concat layer_3   ###########################
        with tf.compat.v1.variable_scope('concat3'):
            concat_rm = tf.concat([conv_m, conv_iv], axis=-1)  # 80
        ####################  Conv_2-3  ###########################
        with tf.compat.v1.variable_scope('conv_2'):
            weights_conv_2 = tf.compat.v1.get_variable("w_conv_2", initializer=tf.constant(reader.get_tensor('fusion_model/conv_2/w_conv_2')))
            bias_conv_2 = tf.compat.v1.get_variable("b_conv_2", initializer=tf.constant(reader.get_tensor('fusion_model/conv_2/b_conv_2')))
            conv_2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(concat_rm, weights_conv_2, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv_2 = lrelu(conv_2)
        with tf.compat.v1.variable_scope('conv_3'):
            weights_conv_3 = tf.compat.v1.get_variable("w_conv_3", initializer=tf.constant(reader.get_tensor('fusion_model/conv_3/w_conv_3')))
            bias_conv_3 = tf.compat.v1.get_variable("b_conv_2", initializer=tf.constant(reader.get_tensor('fusion_model/conv_3/b_conv_2')))
            conv_3 = tf.nn.conv2d(conv_2, weights_conv_3, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_3
            out = tf.nn.tanh(conv_3)

    return out


def input_setup(index):
    padding = 0
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir = (imread(data_ir[index]) - 127.5) / 127.5
    input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (imread(data_vi[index]) - 127.5) / 127.5
    input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    return train_data_ir, train_data_vi

# import matplotlib

# from pylab import *
for idx_num in range(1):
    idx_num = 11
    reader = tf.compat.v1.train.NewCheckpointReader(
        '/model/IV/model/model-' + str(idx_num))

    with tf.name_scope('IR_input'):
        images_ir = tf.compat.v1.placeholder(tf.float32, [1, None, None, None], name='images_ir')
    with tf.name_scope('VI_input'):
        images_vi = tf.compat.v1.placeholder(tf.float32, [1, None, None, None], name='images_vi')
    with tf.name_scope('input'):
       
        input_image_ir = tf.concat([images_ir, images_ir, images_ir], axis=-1)
        input_image_vi = tf.concat([images_vi, images_vi, images_vi], axis=-1)

    with tf.name_scope('fusion'):
        fusion_image = fusion_model(input_image_ir, input_image_vi)
        input_shape = fusion_image.get_shape().as_list()
        print(input_shape)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        data_ir = prepare_data('/Test_ir')
        data_vi = prepare_data('/Test_vi')
        for i in range(len(data_ir)):
            start = time.time()
            train_data_ir, train_data_vi = input_setup(i)
            ####################  visual  ###########################
            #tensor_name = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            #for j in tensor_name:
               # print(j)
            # out_tensor = sess.graph.get_operation_by_name('fusion/fusion_model/attention_layer5/attention/attention3').outputs[0]
            # out_t = sess.run(out_tensor, feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
            # print(out_t.shape)
            # for k in range(10):
            #     imshow(out_t[0, :, :, k])
            #     show()
            ####################  visual  ###########################
            result = sess.run(fusion_image, feed_dict={images_ir: train_data_ir, images_vi: train_data_vi})
            result = result * 127.5 + 127.5
            result = result.squeeze()
            image_path = os.path.join('result', str(idx_num))
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            if i <= 9:
                image_path = os.path.join(image_path, str(i) + ".bmp")
            else:
                image_path = os.path.join(image_path, str(i) + ".bmp")
            end = time.time()
            # print(out.shape)
            imsave(result, image_path)
            print("Testing [%d] success,Testing time is [%f]" % (i, end - start))
    tf.compat.v1.reset_default_graph()
