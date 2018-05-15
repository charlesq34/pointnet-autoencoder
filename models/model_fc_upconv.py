""" TF model for point cloud autoencoder. PointNet encoder, FC and UPCONV decoder.
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

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
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
    num_point = point_cloud.get_shape()[1].value
    assert(num_point==2048)
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
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    net = tf.reshape(global_feat, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc00', bn_decay=bn_decay)
    embedding = tf.reshape(net, [batch_size, -1])
    end_points['embedding'] = embedding

    # FC Decoder
    net = tf_util.fully_connected(embedding, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 1024*3, activation_fn=None, scope='fc3')
    pc_fc = tf.reshape(net, (batch_size, -1, 3))

    # UPCONV Decoder
    net = tf.reshape(embedding, [batch_size, 1, 1, -1])
    net = tf_util.conv2d_transpose(net, 512, kernel_size=[2,2], stride=[1,1], padding='VALID', scope='upconv1', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[3,3], stride=[1,1], padding='VALID', scope='upconv2', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[4,4], stride=[2,2], padding='VALID', scope='upconv3', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 128, kernel_size=[5,5], stride=[3,3], padding='VALID', scope='upconv4', bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 3, kernel_size=[1,1], stride=[1,1], padding='VALID', scope='upconv5', activation_fn=None)
    end_points['xyzmap'] = net
    pc_upconv = tf.reshape(net, [batch_size, -1, 3])


    # Set union
    net = tf.concat(values=[pc_fc,pc_upconv], axis=1)

    return net, end_points

def get_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    loss = tf.reduce_mean(dists_forward+dists_backward)
    end_points['pcloss'] = loss
    return loss*100, end_points


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        loss = get_loss(outputs[0], tf.zeros((32,2048,3)), outputs[1])
