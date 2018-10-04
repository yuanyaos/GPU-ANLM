all:
	nvcc -g -lineinfo -c --compiler-options "-fPIC -Wall" -o filterGPU.o filterGPU.cu
	nvcc -c debug.c --compiler-options "-Wall -pedantic"
	nvcc --link debug.o filterGPU.o -o debug -L/pub/cuda-7.5/lib64 -lcudart -lm

clean:
	rm -rf debug *.o outputimg*