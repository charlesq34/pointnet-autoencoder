#!/bin/bash

export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

export LD_LIBRARY_PATH=$TF_LIB:$LD_LIBRARY_PATH
