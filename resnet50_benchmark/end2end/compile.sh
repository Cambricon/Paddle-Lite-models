CFLAGS_16='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8 -D TASK_9 -D TASK_10 -D TASK_11 -D TASK_12 -D TASK_13 -D TASK_14 -D TASK_15 -D TASK_16 '
CFLAGS_15='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8 -D TASK_9 -D TASK_10 -D TASK_11 -D TASK_12 -D TASK_13 -D TASK_14 -D TASK_15  '
CFLAGS_14='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8 -D TASK_9 -D TASK_10 -D TASK_11 -D TASK_12 -D TASK_13 -D TASK_14  '
CFLAGS_13='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8 -D TASK_9 -D TASK_10 -D TASK_11 -D TASK_12 -D TASK_13  '
CFLAGS_12='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8 -D TASK_9 -D TASK_10 -D TASK_11 -D TASK_12  '
CFLAGS_11='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8 -D TASK_9 -D TASK_10 -D TASK_11  '
CFLAGS_10='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8 -D TASK_9 -D TASK_10  '
CFLAGS_9='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8 -D TASK_9  '
CFLAGS_8='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7 -D TASK_8  '
CFLAGS_7='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 -D TASK_7  '
CFLAGS_6='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5 -D TASK_6 '
CFLAGS_5='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4 -D TASK_5  '
CFLAGS_4='-D TASK_1 -D TASK_2 -D TASK_3 -D TASK_4  '
CFLAGS_3='-D TASK_1 -D TASK_2 -D TASK_3  '
CFLAGS_2='-D TASK_1 -D TASK_2  '
CFLAGS_1='-D TASK_1  '

compile () {
g++ $1 -std=c++14 -g -Wall -DLITE_WITH_X86 -DLITE_WITH_MLU resnet50_benchmark_end2end.cpp -o $2 \
	  -I $PADDLE_HOME/lite/api \
    -I ../include/ \
		-L $PADDLE_HOME/build.lite.mlu/ \
		-L ${NEUWARE_HOME}/lib64 \
		$PADDLE_HOME/build.lite.mlu/libpaddle_api_full_bundled.a \
		$PADDLE_HOME/build.lite.mlu/libpaddle_api_light_bundled.a \
    -L $PADDLE_HOME/build.lite.mlu/third_party/install/mklml/lib \
    -l iomp5 -pthread -lcnml -lcnrt `pkg-config --libs --cflags opencv` -ldl -lstdc++ -O2
if (($?!=0));then
  echo "compiling failed!"
  exit 1
fi
}
compile "$CFLAGS_1" "right_1"
compile "$CFLAGS_2" "right_2"
compile "$CFLAGS_3" "right_3"
compile "$CFLAGS_4" "right_4"
compile "$CFLAGS_5" "right_5"
compile "$CFLAGS_6" "right_6"
compile "$CFLAGS_7" "right_7"
compile "$CFLAGS_8" "right_8"
compile "$CFLAGS_9" "right_9"
compile "$CFLAGS_10" "right_10"
compile "$CFLAGS_11" "right_11"
compile "$CFLAGS_12" "right_12"
compile "$CFLAGS_13" "right_13"
compile "$CFLAGS_14" "right_14"
compile "$CFLAGS_15" "right_15"
compile "$CFLAGS_16" "right_16"

