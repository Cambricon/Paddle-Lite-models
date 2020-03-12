dir=`pwd`/../../../build/third_party/install/mklml/lib
echo "$dir"
export LD_LIBRARY_PATH=${dir}:${LD_LIBRARY_PATH}
g++ -std=c++11 -g -pg -Wall classification_demo.cpp -o classification_demo \
	  -I ../../../lite/api \
		-L ../../../build/ \
		-L ${NEUWARE_HOME}/lib64 \
		../../../build/libpaddle_api_full_bundled.a \
		../../../build/libpaddle_api_light_bundled.a \
    -L ../../../build/third_party/install/mklml/lib \
    -l iomp5 -lpthread -lcnml -lcnrt `pkg-config --libs --cflags opencv` -ldl -lstdc++
if (($?!=0));then
  echo "compiling failed!"
  exit 1
fi

gdb --args ./classification_demo 2>&1 | tee log
