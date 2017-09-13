all: codec.o main.o conversion.o
	g++ -o codec.so -shared codec.o main.o conversion.o `pkg-config --libs opencv python2 libavcodec libavutil` -lboost_python-py27

codec.o:
	g++ -c codec.cpp `pkg-config --cflags python2 libavcodec libavutil` -fPIC

main.o:
	g++ -c main.cpp `pkg-config --cflags libavcodec python2` -fPIC

conversion.o:
	g++ -c conversion.cpp `pkg-config --cflags python2` -fPIC

clean:
	rm *.o *.so
