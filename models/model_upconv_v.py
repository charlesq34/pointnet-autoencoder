""" TF model for point cloud autoencoder. PointNet encoder, UPCONV decoder.
Using GPU Chamfer's distance loss. Required to have 2048 points.

Author: Charles R. Qi
Date: May 2018
"""
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))
import tf_nndistance


def placeholder_inputs(batch_size, n_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 3))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    batch_size = point_cloud.get_shape()[0].value
    point_dim = point_cloud.get_shape()[2].value
    end_points = {}

    input_image = tf.expand_dims(point_cloud, -1)

    # Encoder
    net = tf_util.conv2d(input_image, 64, [1,point_dim],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf.reduce_max(net, axis=1, keepdims=False)
    global_feat = tf.reshape(global_feat, [batch_size, -1])
    '''
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc00', bn_decay=bn_decay)
    end_points['embedding'] = net
    '''
    with tf.variable_scope('vae'):
        def glorot_init(shape):
            return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
        # Variables
        hidden_dim = 512
        latent_dim = 1024
        weights = {
            'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
            'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
        }
        biases = {
            'z_mean': tf.Variable(glorot_init([latent_dim])),
            'z_std': tf.Variable(glorot_init([latent_dim])),
        }
        z_mean = tf.matmul(global_feat, weights['z_mean']) + biases['z_mean']
        z_std = tf.matmul(global_feat, weights['z_std']) + biases['z_std']
        end_points['z_mean'] = z_mean
        end_points['z_std'] = z_std
        # Sampler: Normal (gaussian) random distribution
        samples = tf.random_normal([batch_size, latent_dim], dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
        # z = µ + σ * N (0, 1)
        z = z_mean + z_std * samples
        #z = z_mean + tf.exp(z_std / 2) * samples

    # UPCONV Decoder
    # (N,1024) -> (N,1,2,512)
    net = tf.reshape(z, [batch_size, 1, 2, -1])
    net = tf_util.conv2d_transpose(net, 512, kernel_size=[2,2], stride=[2,2], padding='VALID', scope='upconv1', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[3,3], stride=[1,1], padding='VALID', scope='upconv2', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[4,5], stride=[2,3], padding='VALID', scope='upconv3', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 128, kernel_size=[5,7], stride=[3,3], padding='VALID', scope='upconv4', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 3, kernel_size=[1,1], stride=[1,1], padding='VALID', scope='upconv5', activation_fn=None)
    end_points['xyzmap'] = net
    net = tf.reshape(net, [batch_size, -1, 3])

    return net, end_points

def get_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    # Reconstruction loss
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    loss = tf.reduce_mean(dists_forward+dists_backward)
    end_points['pcloss'] = loss
    # KL Divergence loss
    kl_div_loss = 1 + end_points['z_std'] - tf.square(end_points['z_mean']) - tf.exp(end_points['z_std'])
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    kl_div_loss = tf.reduce_mean(kl_div_loss) * 0.001
    end_points['kl_div_loss'] = kl_div_loss
    return loss*100 + kl_div_loss, end_points


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        loss = get_loss(outputs[0], tf.zeros((32,2048,3)), outputs[1])
