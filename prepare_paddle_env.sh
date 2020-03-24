PADDLE_LITE_HOME=/home/dingminghui/paddle/Paddle-Lite
if [ $# != 1 ] ; then
    echo "prepare_paddle_env.sh  1"
    echo "arg1: compile or not, 0: not compile src, 1: compile src"
    exit 1
fi

echo "PADDLE_LITE_HOME is ${PADDLE_LITE_HOME}"

if [ -z "${PADDLE_LITE_HOME}" ]; then
    echo "PADDLE_LITE_HOME is empty, set to: ${paddle_home}"
    PADDLE_LITE_HOME="$paddle_home"
elif [ ! -w "${PADDLE_LITE_HOME}" ]; then
    echo "PADDLE_LITE_HOME cann't be rw, set to: ${paddle_home}"
    PADDLE_LITE_HOME="$paddle_home"
fi

echo "paddle lite home : ${PADDLE_LITE_HOME}"
if [ ! -w ${PADDLE_LITE_HOME} ];then
    echo "The dir of PADDLE_LITE_HOME doesn't exist or cann't be read! Please set PADDLE_LITE_HOME first."
    exit 1
fi

if(($1==1));then
    cd ${PADDLE_LITE_HOME}
        ./lite/tools/build_mlu.sh build
    cd -
fi

echo "Finish compiling paddle lite"
export PADDLE_LINK_PATH="${PADDLE_LITE_HOME}/build/lite/api"
export PADDLE_INC_PATH="${PADDLE_LITE_HOME}/lite/api"

# cp ${PADDLE_LITE_HOME}/build.lite.mlu/lite/api/python/pybind/liblite_pybind.so python/lite_core.so

mkdir -p build
rm -r build/*
if [ -d build ];then
    cd build
else
    exit 1
fi

cmake ..
make -j8
cd ..
