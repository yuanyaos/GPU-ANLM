#################################################################
#  Makefile for GPU-ANLM filter mex file
#################################################################

FILES=filterGPU filterGPU_v filterGPU_s

ROOTDIR ?= ..

OUTPUT_DIR=$(ROOTDIR)/bin

MKDIR      := mkdir
CUDACC=nvcc
MEX=mex
AR=$(CC)
BACKEND=cuda

OBJSUFFIX=.o
OBJS      := $(addsuffix $(OBJSUFFIX), $(FILES))

BINARY=ganlm

INCLUDEDIRS=-I/usr/local/cuda/include

CPPOPT=-g -Wall -std=c99 # -DUSE_OS_TIMER

CUDART=-lcudart
CUOMPLINK=
CUDA_STATIC=--cudart static -Xcompiler "-static-libgcc -static-libstdc++"

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)

DLLFLAG=-fPIC

ifeq ($(BACKEND),ocelot)
  LINKOPT=-L/usr/local/lib `OcelotConfig -l` -ltinfo
  CUCCOPT=-D__STRICT_ANSI__ -g #--maxrregcount 32
else ifeq ($(BACKEND),cudastatic)
  ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
      CUDART=-lcudadevrt -lcudart_static -ldl -static-libgcc -static-libstdc++
  else
      CUDART=-lcudadevrt -lcudart_static -ldl -lrt -static-libgcc -static-libstdc++
  endif
  LINKOPT=-L/usr/local/cuda/lib -lm $(CUDART)
  CUCCOPT=-g -lineinfo -Xcompiler -Wall#-arch compute_20 #--maxrregcount 32
else
  LINKOPT=-L/usr/local/cuda/lib -lm $(CUDART)
  CUCCOPT=-g -lineinfo -Xcompiler -Wall#-arch compute_20 #--maxrregcount 32
endif

ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
  CC=nvcc
  CUOMPLINK=-Xcompiler
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
      LINKOPT=-L"$(CUDA_PATH)/lib/x64" $(CUDART)
  else
      LINKOPT=-L"$(CUDA_PATH)/lib/Win32" $(CUDART)
  endif
  INCLUDEDIRS +=-I"$(CUDA_PATH)/lib/include"
  CPPOPT =-c -D_CRT_SECURE_NO_DEPRECATE -DWIN32
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  DLLFLAG=
  OMP=/openmp
  MEX=cmd /c mex
  CUCCOPT+=-Xcompiler "$(OMP) /W0"
  CUDA_STATIC=--cudart static -Xcompiler $(OMP) -Xcompiler /MT
  CPPOPT+=-Xcompiler $(OMP)
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  CUDA_STATIC=--cudart static
  CPPOPT+=$(OMP)
else
  CPPOPT+=$(OMP)
  CUCCOPT+=-Xcompiler $(OMP)
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     CPPOPT +=-m64
     CUCCOPT +=-m64
     ifeq "$(wildcard /usr/local/cuda/lib64)" "/usr/local/cuda/lib64"
        ifeq ($(BACKEND),cuda)
           LINKOPT=-L/usr/local/cuda/lib64 $(CUDART) -lm -lstdc++
        else ifeq ($(BACKEND),cudastatic)
           LINKOPT=-L/usr/local/cuda/lib64 $(CUDART) -lm -static-libgcc -static-libstdc++
        endif
     endif
  endif
endif

mex:    CUCCOPT+=--compiler-options "$(DLLFLAG)"
mex:    LINKOPT+=ANLMGPU.cpp -outdir $(OUTPUT_DIR) $(INCLUDEDIRS)
mex:    AR=$(MEX)
mex:    OUTPUTFLAG:=-output
mex:    all

all: $(OUTPUT_DIR)/$(BINARY)

$(OUTPUT_DIR)/$(BINARY): makedirs $(OBJS)
$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(AR) $(OBJS) $(OUTPUTFLAG) $(OUTPUT_DIR)/$(BINARY) $(LINKOPT) $(USERLINKOPT)
%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

%$(OBJSUFFIX): %.cu
	$(CUDACC) -c $(CUCCOPT) -o $@  $<

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

clean:
	rm -rf *.o $(OUTPUT_DIR)

.DEFAULT_GOAL := mex

