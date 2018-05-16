CUDA_ROOT = /opt/cuda

INCLUDE += -I ${TF_INC}
INCLUDE += -I ${TF_INC}/external/nsync/public
INCLUDE += -I /opt
LIB = -lcudart -ltensorflow_framework -L ${CUDA_ROOT}/lib64 -L ${TF_LIB}
FLAGS = -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1
CUDA_FLAGS = -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr --Wno-deprecated-gpu-targets

#fix cannot find cuda/cuda_config.h
# cd `python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
# cd tensorflow/stream_executor/cuda
# sudo curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/master/third_party/toolchains/gpus/cuda/cuda/cuda_config.h

all: tf_approxmatch_so.so

tf_approxmatch_so.so: tf_approxmatch.cpp tf_approxmatch_g.cu.o
	g++ -fPIC -shared ${FLAGS} $+ -o $@  ${INCLUDE} ${LIB}
tf_approxmatch_g.cu.o:
	nvcc -std=c++11 -c -o $@ tf_approxmatch_g.cu ${INCLUDE} ${CUDA_FLAGS}

clean:
	rm -rf *.o *.so
