all:
	nvcc -c --compiler-options "-fPIC" -o filterGPU.o -odir ../bin filterGPU.cu
	nvcc -c --compiler-options "-fPIC" -o filterGPU_v.o -odir ../bin filterGPU_v.cu
	nvcc -c --compiler-options "-fPIC" -o filterGPU_s.o -odir ../bin filterGPU_s.cu
	mex -g CFLAGS='-std=c99 -fPIC'  -outdir ../bin ANLMGPU.c filterGPU.o filterGPU_v.o filterGPU_s.o -L/pub/cuda-7.5/lib64 -lcudart

clean:
	rm -rf filterGPU*.o ../bin/ANLMGPU.mex*