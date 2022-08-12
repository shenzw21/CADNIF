# -*- coding: utf-8 -*-
from tensorflow.python.ops.gen_array_ops import concat
from utils import (
    input_setup,
    gradient,
    read_data,
    imsave,
    lrelu,
    smooth_loss,
    Attention_gb,
    DRDB,
    DRDB1,
    DRDB2,
    NonLocalBlock,
    channel_shuffle
)
# from non_local import NonLocalBlock
import time
import os
# import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf


class CGAN(object):

    def __init__(self,
                 sess,
                 image_size=132,
                 label_size=120,
                 batch_size=16,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        with tf.name_scope('IR_input'):
           
            self.images_ir = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim],
                                                      name='images_ir')
            self.labels_ir = tf.compat.v1.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim],
                                                      name='labels_ir')
        with tf.name_scope('VI_input'):
           
            self.images_vi = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim],
                                                      name='images_vi')
            self.labels_vi = tf.compat.v1.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim],
                                                      name='labels_vi')
          
        with tf.name_scope('input'):
            self.input_image_ir = tf.concat([self.labels_ir, self.labels_ir, self.labels_vi], axis=-1)
            self.input_image_vi = tf.concat([self.labels_vi, self.labels_vi, self.labels_ir], axis=-1)
  
        with tf.name_scope('fusion'):
            self.fusion_image = self.fusion_model(self.input_image_ir, self.input_image_vi)

        with tf.name_scope('g_loss'):
            
            self.g_loss_2 = (tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir)) + tf.reduce_mean(
                 tf.square(self.fusion_image - self.labels_vi))) + ((
                    tf.reduce_mean(1 - smooth_loss(self.fusion_image, self.labels_ir))+ tf.reduce_mean(
                 1 - smooth_loss(self.fusion_image, self.labels_vi))))*5
            #self.g_loss_2=(tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+tf.reduce_mean(
            #    tf.square(self.fusion_image - self.labels_vi)))+8*(tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_vi)))+tf.reduce_mean(
            #        tf.square(gradient(self.fusion_image) -gradient (self.labels_ir))))
            tf.compat.v1.summary.scalar('g_loss_2', self.g_loss_2)
            self.g_loss_total = 100 * self.g_loss_2
            tf.compat.v1.summary.scalar('loss_g', self.g_loss_total)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=50)

        with tf.name_scope('image'):
            tf.compat.v1.summary.image('input_ir', tf.expand_dims(self.images_ir[1, :, :, :], 0))
            tf.compat.v1.summary.image('input_vi', tf.expand_dims(self.images_vi[1, :, :, :], 0))
            tf.compat.v1.summary.image('fusion_image', tf.expand_dims(self.fusion_image[1, :, :, :], 0))

    def train(self, config):

        if config.is_train:
            input_setup(self.sess, config,"/Datasets/Medical/Train_ir")
            input_setup(self.sess,config,"/Datasets/Medical/Train_vi")
        else:
            nx_ir, ny_ir = input_setup(self.sess, config,"/IR_VI/Test_ir")
            nx_vi,ny_vi=input_setup(self.sess, config,"/IR_VI/Test_vi")
       
        if config.is_train:
            data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir),
                                       "/tno/Train_ir", "train.h5")
            data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir),
                                       "/tno/Train_vi", "train.h5")
        else:
            data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "/Test_ir", "test.h5")
            data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "/Test_vi", "test.h5")

        train_data_ir, train_label_ir = read_data(data_dir_ir)
        train_data_vi, train_label_vi = read_data(data_dir_vi)
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        print(self.d_vars)
        self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
        print(self.g_vars)

        with tf.name_scope('train_step'):
            self.train_fusion_op = tf.compat.v1.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,
                                                                                                   var_list=self.g_vars)
 
        self.summary_op = tf.compat.v1.summary.merge_all()

        self.train_writer = tf.compat.v1.summary.FileWriter(config.summary_dir + '/train', self.sess.graph,
                                                            flush_secs=60)

        tf.global_variables_initializer().run()

        counter = 0
        start_time = time.time()

        if config.is_train:
            print("Training...")

            for ep in range(config.epoch):
                # Run by batch images
                batch_idxs = len(train_data_ir) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images_ir = train_data_ir[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels_ir = train_label_ir[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_images_vi = train_data_vi[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels_vi = train_label_vi[idx * config.batch_size: (idx + 1) * config.batch_size]

                    counter += 1
                  
                    _, err_g, summary_str = self.sess.run([self.train_fusion_op, self.g_loss_total, self.summary_op],
                                                          feed_dict={self.images_ir: batch_images_ir,
                                                                     self.images_vi: batch_images_vi,
                                                                     self.labels_ir: batch_labels_ir,
                                                                     self.labels_vi: batch_labels_vi})
                    self.train_writer.add_summary(summary_str, counter)

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_g:[%.8f]" \
                              % ((ep + 1), counter, time.time() - start_time, err_g))
                        # print(a)

                self.save(config.checkpoint_dir, ep)

        else:
            print("Testing...")

            result = self.fusion_image.eval(
                feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir, self.images_vi: train_data_vi,
                           self.labels_vi: train_label_vi})
            result = result * 127.5 + 127.5
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), config.sample_dir)
            image_path = os.path.join(image_path, "test_image.png")
            imsave(result, image_path)

    
    def fusion_model(self, img_ir, img_vi):
        ####################  Layer1  ###########################
        with tf.compat.v1.variable_scope('fusion_model'):
            ####################  Concat layer_1  ###########################
            with tf.compat.v1.variable_scope('concat'):
                concat_iv = tf.concat([img_ir, img_vi], axis=-1)
            with tf.compat.v1.variable_scope('conv_iv'):
                weights_iv = tf.compat.v1.get_variable("w_iv", [1, 1, 6, 16],
                                                       initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_iv = tf.compat.v1.get_variable("b_iv", [16], initializer=tf.constant_initializer(0.0))
                conv_iv = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(concat_iv, weights_iv, strides=[1, 1, 1, 1], padding='SAME') + bias_iv, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv_iv = lrelu(conv_iv)
            with tf.compat.v1.variable_scope('conv_i'):
                weights_i = tf.compat.v1.get_variable("w_i", [1, 1, 3, 16],
                                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_i = tf.compat.v1.get_variable("b_i", [16], initializer=tf.constant_initializer(0.0))
                conv_i = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(img_ir, weights_i, strides=[1, 1, 1, 1], padding='SAME') + bias_i, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv_i = lrelu(conv_i)
            with tf.compat.v1.variable_scope('conv_v'):
                weights_v = tf.compat.v1.get_variable("w_v", [1, 1, 3, 16],
                                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_v = tf.compat.v1.get_variable("b_iv", [16], initializer=tf.constant_initializer(0.0))
                conv_v = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(img_vi, weights_v, strides=[1, 1, 1, 1], padding='SAME') + bias_v, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv_v = lrelu(conv_v)
            ##auxiliary net
            
            with tf.compat.v1.variable_scope('conv_ai'):
                weights_aiv = tf.compat.v1.get_variable("w_aiv", [3, 3, 16, 32],
                                                       initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_aiv = tf.compat.v1.get_variable("b_aiv", [32], initializer=tf.constant_initializer(0.0))
                conv_aiv = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv_i, weights_aiv, strides=[1, 1, 1, 1], padding='SAME') + bias_aiv, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv_ai = lrelu(conv_aiv)
            with tf.compat.v1.variable_scope('conv_av'):
                weights_ai = tf.compat.v1.get_variable("w_ai", [3, 3, 16, 32],
                                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_ai = tf.compat.v1.get_variable("b_ai", [32], initializer=tf.constant_initializer(0.0))
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
                weights_1 = tf.compat.v1.get_variable("A_conv_w", [3, 3, 32, 16],
                                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_1 = tf.compat.v1.get_variable("A_conv_b", [16], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(concat_1, weights_1, strides=[1, 1, 1, 1], padding='SAME') + bias_1
                conv1 = lrelu(conv1)
            with tf.compat.v1.variable_scope('attention_layer2'):
                Block_2 = Attention_gb(conv_i, conv_v, conv1)
                concat_2 = tf.concat([Block_2, conv1, conv_iv], axis=-1)
            with tf.compat.v1.variable_scope('A_conv2'):
                weights_2 = tf.compat.v1.get_variable("A_conv2_w", [3, 3, 48, 16],
                                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_2 = tf.compat.v1.get_variable("A_conv2_b", [16], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(concat_2, weights_2, strides=[1, 1, 1, 1], padding='SAME') + bias_2
                conv2 = lrelu(conv2)
            with tf.compat.v1.variable_scope('attention_layer3'):
                Block_3 = Attention_gb(conv_i, conv_v, conv2)
                concat_3 = tf.concat([conv2, Block_3, conv1, conv_iv], axis=-1)
            with tf.compat.v1.variable_scope('A_conv3'):
                weights_3 = tf.compat.v1.get_variable("A_conv3_w", [3, 3, 64, 16],
                                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_3 = tf.compat.v1.get_variable("A_conv3_b", [16], initializer=tf.constant_initializer(0.0))
                conv3 = tf.nn.conv2d(concat_3, weights_3, strides=[1, 1, 1, 1], padding='SAME') + bias_3
                conv3 = lrelu(conv3)
            with tf.compat.v1.variable_scope('attention_layer4'):
                Block_4 = Attention_gb(conv_i, conv_v, conv3)
                concat_4 = tf.concat([conv3, conv2, Block_4, conv1, conv_iv], axis=-1)
            with tf.compat.v1.variable_scope('A_conv4'):
                weights_4 = tf.compat.v1.get_variable("A_conv4_w", [3, 3, 80, 16],
                                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_4 = tf.compat.v1.get_variable("A_conv4_b", [16], initializer=tf.constant_initializer(0.0))
                conv4 = tf.nn.conv2d(concat_4, weights_4, strides=[1, 1, 1, 1], padding='SAME') + bias_4
                conv4 = lrelu(conv4)
            with tf.compat.v1.variable_scope('attention_layer5'):
                Block_5 = Attention_gb(conv_i, conv_v, conv4)
                concat_5 = tf.concat([conv4, conv3, conv2, Block_5, conv1, conv_iv], axis=-1)  # 96
            ####################  Concat layer_2   ###########################
            with tf.compat.v1.variable_scope('concat2'):
                concat_ca = tf.concat([concat_5, c_c], axis=-1)  # 352
            ####################  Conv_1   ###########################
            with tf.compat.v1.variable_scope('conv_1'):
                weights_conv_1 = tf.compat.v1.get_variable("w_conv_1", [1, 1, 160, 64],
                                                           initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_conv_1 = tf.compat.v1.get_variable("b_conv_1", [64], initializer=tf.constant_initializer(0.0))
                conv_1 = tf.nn.conv2d(concat_ca, weights_conv_1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_1
                conv_1 = lrelu(conv_1)
            ####################  Merging layers   ###########################
            with tf.compat.v1.variable_scope('merging'):
                concat_merg_1 = DRDB(conv_1)  # 128
                concat_merg_2 = DRDB1(concat_merg_1)  # 256
                concat_merg_3 = DRDB2(concat_merg_2)  # 512
                concat_m = tf.concat([concat_merg_1, concat_merg_2], axis=-1)  # 896
            with tf.compat.v1.variable_scope('m_conv_1'):
                weights_m_1 = tf.compat.v1.get_variable("w_m_1", [1, 1, 128, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_m_1 = tf.compat.v1.get_variable("b_m_1", [64], initializer=tf.constant_initializer(0.0))
                conv_m = tf.nn.conv2d(concat_merg_1, weights_m_1, strides=[1, 1, 1, 1], padding='SAME') + bias_m_1
                conv_m = lrelu(conv_m)
            ####################  Concat layer_3   ###########################
            with tf.compat.v1.variable_scope('concat3'):
                concat_rm = tf.concat([conv_m, conv_iv], axis=-1)  # 80
            ####################  Conv_2-3  ###########################
            with tf.compat.v1.variable_scope('conv_2'):
                weights_conv_2 = tf.compat.v1.get_variable("w_conv_2", [3, 3, 80, 64],
                                                           initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_conv_2 = tf.compat.v1.get_variable("b_conv_2", [64], initializer=tf.constant_initializer(0.0))
                conv_2 = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(concat_rm, weights_conv_2, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_2,
                    decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv_2 = lrelu(conv_2)
            with tf.compat.v1.variable_scope('conv_3'):
                weights_conv_3 = tf.compat.v1.get_variable("w_conv_3", [3, 3, 64, 1],
                                                           initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias_conv_3 = tf.compat.v1.get_variable("b_conv_2", [1], initializer=tf.constant_initializer(0.0))
                conv_3 = tf.nn.conv2d(conv_2, weights_conv_3, strides=[1, 1, 1, 1], padding='SAME') + bias_conv_3
                out = tf.nn.tanh(conv_3)

        return out

    def save(self, checkpoint_dir, step):
        model_name = "CGAN.model"
        model_dir = "%s_%s" % ("CGAN", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("CGAN", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
