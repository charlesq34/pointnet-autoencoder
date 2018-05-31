# pointnet-autoencoder

![prediction example](https://github.com/charlesq34/pointnet-autoencoder/blob/master/doc/teaser.jpg)

Here we present code to build an autoencoder for point clouds, with <a href="https://github.com/charlesq34/pointnet">PointNet</a> encoder and various kinds of decoders. We train and test our autoencoder on the <a href="https://cs.stanford.edu/~ericyi/project_page/part_annotation/index.html" target="_blank">ShapeNetPart dataset</a>. This is a side project I played with recently -- you are welcomed to modify it for your own projects or research. Let me know if you discover something interesting!

## LICENSE
This repository is under the MIT license. See the LICENSE file for detail.

## Installation
We need <a href="https://www.tensorflow.org/install/" target="_blank">TensorFlow</a> (version>=1.4).

For point cloud reconstruction loss function, we need to compile two custum TF operators under `tf_ops/nn_distance` (Chamfer's distance) and `tf_ops/approxmatch` (earth mover's distance). Check the `tf_compile_*.sh` script under these two folders, modify the TensorFlow and CUDA path accordingly before you run the shell script to compile the operators. Check this <a href="https://arxiv.org/abs/1612.00603" target="_blank">PAPER</a> for an introduction for these two point cloud losses.

For a visualization helper, go to `utils/` and run `sh compile_render_balls_so.sh` -- run `python show3d_balls.py` to test if you have successfully compiled it.

## Download Data
ShapeNetPart dataset is available <a href="https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip" target="_blank">HERE (635MB)</a>. Simply download the zip file and move the `shapenetcore_partanno_segmentation_benchmark_v0` folder to `data`.

To visualize the dataset, run (type `q` to go to the next shape, see `show3d_balls.py` for more detailed hot keys):

    python part_dataset.py

## Train an Autoencoder
To train the most basic autoencoder (fully connected layer decoder with Chamfer's distance loss) on chair models with aligned poses, simply run the following command:

    python train.py --model model --log_dir log_chair_norotation --num_point 2048 --category Chair --no_rotation

You can check more options for training by:

    python train.py -h

## Visualize Reconstruction on Test Set
To test and visualize results of the trained autoencoder above, simply run:

    python test.py --model model --model_path log_chair_norotation/model.ckpt --category Chair

You can check more options for testing by:
    
    python test.py -h
