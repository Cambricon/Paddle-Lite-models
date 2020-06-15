
compile () {
g++  -std=c++11 -g -Wall  ./offline_loader.cpp -o offline_loader \
	    -I $NEUWARE_HOME/include \
					-L ${NEUWARE_HOME}/lib64 \
      -lcnrt -lcndrv `pkg-config --libs --cflags opencv` -ldl -lstdc++ -O0
if (($?!=0));then
  echo "compiling failed!"
  exit 1
fi
}
compile

