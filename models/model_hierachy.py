""" TF model for point cloud autoencoder. PointNet encoder, *hierachical* FC decoder.
Using GPU Chamfer's distance loss.

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
        point_cloud: TF tensor BxNxC
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        pc_xyz: TF tensor BxNxC, reconstructed point clouds
        end_points: dict
    """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
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
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc01', bn_decay=bn_decay)
    end_points['embedding'] = net

    # Hierarchical FC decoder
    # Firstly predict 64 points with their XYZs and features
    # Then from feature of each of the 64 points, predict XYZs for NUM_POINT/64 subpoints with their *local* XYZs
    # At last, local XYZs are translated to their global XYZs
    pc1_feat = tf_util.fully_connected(net, 64*256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc1_xyz = tf_util.fully_connected(net, 64*3, activation_fn=None, scope='fc1_xyz')
    pc1_feat = tf.reshape(pc1_feat, [batch_size, 64, 256])
    pc1_xyz = tf.reshape(pc1_xyz, [batch_size, 64, 3])
    end_points['pc1_xyz'] = pc1_xyz

    pc2 = tf_util.conv1d(pc1_feat, 256, 1, padding='VALID', stride=1, bn=True, is_training=is_training, scope='fc_conv1', bn_decay=bn_decay)
    pc2_xyz = tf_util.conv1d(pc2, (num_point/64)*3, 1, padding='VALID', stride=1, activation_fn=None, scope='fc_conv3') # B,64,32*3
    pc2_xyz = tf.reshape(pc2_xyz, [batch_size, 64, num_point/64, 3])
    pc1_xyz_expand = tf.expand_dims(pc1_xyz, 2) # B,64,1,3
    # Translate local XYZs to global XYZs
    pc2_xyz = pc2_xyz + pc1_xyz_expand
    pc_xyz = tf.reshape(pc2_xyz, [batch_size, num_point, 3])
 
    return pc_xyz, end_points

def get_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    pc_loss = tf.reduce_mean(dists_forward+dists_backward)
    end_points['pcloss'] = pc_loss
    tf.summary.scalar('pcloss', pc_loss)
    
    d1,_,d2,_ = tf_nndistance.nn_distance(end_points['pc1_xyz'], label)
    pc1_loss = tf.reduce_mean(d1) + tf.reduce_mean(d2)
    tf.summary.scalar('pc1loss', pc1_loss)
    loss = pc_loss + 0.1*pc1_loss

    return loss*100, end_points


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        loss = get_loss(outputs[0], tf.zeros((32,1024,3)), outputs[1])
