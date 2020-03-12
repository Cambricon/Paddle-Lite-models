PADDLE_LITE_HOME=/home/zhangshijin/Paddle-Lite
if [ $# != 1 ] ; then
    echo "prepare_paddle_env.sh  1"
    echo "arg1: compile or not, 0: compile src, 1: not compile src"
    exit 1
fi

echo "paddle lite home : ${PADDLE_LITE_HOME}"
if [ ! -d ${PADDLE_LITE_HOME} ];then
    echo "The dir of PADDLE_LITE_HOME doesn't exist!, please fill in first."
    exit 1
fi

if(($1==1));then
    cd ${PADDLE_LITE_HOME}
    ./lite/tools/build_mlu.sh build
    cd -
fi

echo "Finish compiling paddle lite"
export PADDLE_LINK_PATH="${PADDLE_LITE_HOME}/build.lite.mlu/lite/api"
export PADDLE_INC_PATH="${PADDLE_LITE_HOME}/lite/api"

mkdir build
if [ -d build ];then
    cd build
else
    exit 1
fi
cmake ..
make -j8
cd ..
