
compile () {
g++  -std=c++14 -g -Wall  ./offline_loader.cpp -o offline_loader \
		-L ${NEUWARE_HOME}/lib64 \
      -lcnrt `pkg-config --libs --cflags opencv` -ldl -lstdc++ -O0
if (($?!=0));then
  echo "compiling failed!"
  exit 1
fi
}
compile

