CC=gcc
CFLAGS=-shared -pthread -fPIC -fwrapv -Ofast -funroll-loops -Wall -fno-strict-aliasing -I/usr/include/python3.5 -I/usr/local/lib/python3.5/dist-packages/numpy-1.13.0.dev0+85cd7b7-py3.5-linux-x86_64.egg/numpy/core/include

all:
	cython word2vec_fast.pyx
	$(CC) -o word2vec_fast.so word2vec_fast.c $(CFLAGS)

clean:
	rm word2vec_fast.so
