PADDLE_LITE_HOME=/home/dingminghui/paddle/Paddle-Lite
if [ $# != 1 ] ; then
    echo "prepare_paddle_env.sh  1"
    echo "arg1: compile or not, 0: not compile src, 1: compile src"
    exit 1
fi

echo "paddle lite home : ${PADDLE_LITE_HOME}"
if [ ! -d ${PADDLE_LITE_HOME} ];then
    echo "The dir of PADDLE_LITE_HOME doesn't exist!, please fill in first."
    exit 1
fi

if(($1==1));then
    cd ${PADDLE_LITE_HOME}
      mkdir -p build
      cd build
      cmake .. -DWITH_LITE=ON \
               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
               -DWITH_PYTHON=OFF \
               -DLITE_WITH_ARM=OFF \
               -DWITH_GPU=OFF \
               -DWITH_MKLDNN=OFF \
               -DLITE_WITH_X86=ON \
               -DWITH_MKL=ON \
               -DLITE_WITH_MLU=ON \
      make -j8

    cd -
fi

echo "Finish compiling paddle lite"
export PADDLE_LINK_PATH="${PADDLE_LITE_HOME}/build/lite/api"
export PADDLE_INC_PATH="${PADDLE_LITE_HOME}/lite/api"

mkdir -p build
if [ -d build ];then
    cd build
else
    exit 1
fi
cmake ..
make -j8
cd ..
