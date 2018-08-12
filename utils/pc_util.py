"""
    point cloud utilities
"""
import numpy as np
import random

def rand_rotation_matrix(deflection=1.0, seed=None):
    '''Creates a random rotation matrix (3x3).

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, completely random
    rotation. Small deflection => small perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    '''
    if seed is not None:
        np.random.seed(seed)

    randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi    # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi     # For direction of pole deflection.
    z = z * 2.0 * deflection    # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=batch_data.dtype)
    for k in range(batch_data.shape[0]):
        rotation_matrix = rand_rotation_matrix(deflection=random.random())
        #shape_pc = batch_data[k, ...]
        #rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, ...] = np.dot(batch_data[k, ...], rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def shuffle_point_cloud(batch_data):
    """ Randomly shuffle points per sample.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shuffled batch of point clouds
    """
    shuffled_data = np.zeros(batch_data.shape, dtype=batch_data.dtype)
    num_point = batch_data.shape[1]
    for k in range(batch_data.shape[0]):
        permutation = np.random.permutation(num_point)
        shuffled_data[k, range(num_point), ...] = batch_data[k, permutation, ...]
    return shuffled_data

if __name__ == '__main__':
    batch_data = np.array([[[ 1, 1, 1],
                            [-1,-1,-1],
                            [ 1,-1, 1],
                            [ 1, 1,-1]]], dtype=float)
    print(batch_data.shape) # (1, 4, 3)
    print('batch_data', batch_data)
    shuffled_data = shuffle_point_cloud(batch_data)
    print('shuffled_data', shuffled_data)
    jittered_data = jitter_point_cloud(batch_data)
    print('jittered_data', jittered_data)
    rotated_data = rotate_point_cloud(batch_data)
    print('rotated_data', rotated_data)
