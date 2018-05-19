""" TF model for point cloud autoencoder. PointNet encoder, FC decoder.
Using GPU Earth Mover's distance loss.

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
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/approxmatch'))
import tf_approxmatch

num_point = 0

def placeholder_inputs(batch_size, n_point):
    global num_point
    num_point = n_point
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 3))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNxC
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNxC, reconstructed point clouds
        end_points: dict
    """
    global num_point
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
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf.reduce_max(net, axis=1, keepdims=False)

    net = tf.reshape(global_feat, [batch_size, -1])
    end_points['embedding'] = net

    # FC Decoder
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, num_point*3, activation_fn=None, scope='fc3')
    net = tf.reshape(net, (batch_size, num_point, 3))

    return net, end_points

def get_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    pc_loss = tf.reduce_mean(dists_forward+dists_backward)
    end_points['pcloss'] = pc_loss

    match = tf_approxmatch.approx_match(label, pred)
    loss = tf.reduce_mean(tf_approxmatch.match_cost(label, pred, match))
    tf.summary.scalar('loss', loss)
    return loss, end_points


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        loss = get_loss(outputs[0], tf.zeros((32,1024,3)), outputs[1])
