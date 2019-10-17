#
# To add new ext/function, add another entry to SRC and NVSRC below
#

CUDA = $(shell ./cuda-detect.sh)
ifeq ($(CUDA),yes)
CUDA_ARCH = $(shell ./get_arch.sh)
CUDAHOST_CC = $(shell ./get_cc.sh)
endif
BRANCH = $(shell git branch -l | fgrep \* | cut -c3-)
REV = $(shell git rev-parse $(BRANCH))

include ../make.conf

all: proto.h
ifeq ($(CUDA),yes)
	echo "CUDA enabled for GPU architecture " $(CUDA_ARCH)
	echo $(CUDA_ARCH) > cuda.arch
	echo $(CUDAHOST_CC) > cuda.hostcc
endif
	echo $(CUDA) > cuda.status
	echo "#define VERSION \""$(BRANCH)-$(REV)"\"" > git-version.h
	make libgrid.a

SRC = cgrid.c cgrid_fft.c cgrid_diff.c rgrid.c rgrid_fft.c rgrid_diff.c ngrid.c grid_wf.c grid_wf_fft.c grid_wf_diag.c grid_wf_cn.c cgrid_bc.c \
      rgrid_bc.c threads.c linalg.c ctimer.c interpolate.c grid_fft.c grid_wf_analyze.c rand.c rgrid_func.c
SRC += ext-cuda/func1c.c ext-cuda/func2c.c ext-cuda/func3c.c ext-cuda/func4c.c ext-cuda/func5c.c ext-cuda/func6c.c ext-cuda/func7c.c

ifeq ($(CUDA),yes)
  SRC += cgrid-cuda.c cgrid_diff-cuda.c cgrid_fft-cuda.c rgrid-cuda.c rgrid_diff-cuda.c rgrid_fft-cuda.c grid_wf-cuda.c grid_wf_fft-cuda.c grid_wf_cn-cuda.c cuda.c \
         ngrid-cuda.c grid_wf_analyze-cuda.c rand-cuda.c rgrid_func-cuda.c
  NVSRC = cgrid-cuda2.cu cgrid_diff-cuda2.cu rgrid-cuda2.cu rgrid_diff-cuda2.cu grid_wf-cuda2.cu grid_wf_fft-cuda2.cu grid_wf_cn-cuda2.cu ngrid-cuda2.cu \
          grid_wf_analyze-cuda2.cu rand-cuda2.cu rgrid_func-cuda2.cu
  NVSRC += ext-cuda/func1cu.cu ext-cuda/func2cu.cu ext-cuda/func3cu.cu ext-cuda/func4cu.cu ext-cuda/func5cu.cu ext-cuda/func6cu.cu ext-cuda/func7cu.cu
endif

OBJS = $(SRC:.c=.o) $(NVSRC:.cu=.o)

libgrid.a: $(OBJS)
	ar cr libgrid.a $(OBJS)
	ranlib libgrid.a

%.o: %.cu
	nvcc $(NVCFLAGS) -c -o $@ $<

grid.h: structs.h proto.h defs.h
cgrid_bc.o: cgrid_bc.c grid.h
cgrid.o: cgrid.c grid.h cprivate.h
cgrid-cuda2.o: cgrid-cuda2.cu cuda-math.h cgrid_bc-cuda.h cuda-vars.h cuda.h
cgrid-cuda.o: cgrid-cuda.c grid.h
cgrid_fft.o: cgrid_fft.c grid.h
cgrid_fft-cuda.o: cgrid_fft-cuda.c grid.h
ctimer.o: ctimer.c grid.h
cuda.o: cuda.c cuda.h cuda-private.h
grid_wf.o: grid_wf.c grid.h cprivate.h
grid_wf_analyze.o: grid_wf_analyze.c grid.h au.h
grid_wf_cn.o: grid_wf_cn.c grid.h
grid_wf_cn-cuda.o: grid_wf_cn-cuda.c grid.h
grid_wf_cn-cuda2.o: grid_wf_cn-cuda2.cu linalg-cuda.h cuda.h cuda-math.h defs.h grid_wf-cuda.h
grid_wf_diag.o: grid_wf_diag.c grid.h
grid_wf_fft.o: grid_wf_fft.c grid.h
grid_wf_fft-cuda2.o: grid_wf_fft-cuda2.cu cuda.h cuda-math.h cuda-vars.h
grid_wf_fft-cuda.o: grid_wf_fft-cuda.c grid.h
linalg.o: linalg.c grid.h
ngrid.o: ngrid.c grid.h cprivate.h
ngrid-cuda.o: ngrid-cuda.c grid.h
ngrid-cuda2.o: ngrid-cuda2.cu cuda-math.h cgrid_bc-cuda.h cuda-vars.h cuda.h
interpolate.o: interpolate.c grid.h
rgrid_bc.o: rgrid_bc.c grid.h
rgrid.o: rgrid.c grid.h rprivate.h
rgrid-cuda.o: rgrid-cuda.c grid.h
rgrid-cuda2.o: rgrid-cuda2.cu cuda-math.h rgrid_bc-cuda.h cuda-vars.h cuda.h
rgrid_fft.o: rgrid_fft.c grid.h
rgrid_fft-cuda.o: rgrid_fft-cuda.c grid.h
grid_fft.o: grid_fft.c grid.h
threads.o: threads.c grid.h
ext-cuda/func1c.o: ext-cuda/func1c.c grid.h ext-cuda/func1.h
ext-cuda/func1cu.o: ext-cuda/func1cu.cu ext-cuda/func1.h
ext-cuda/func2c.o: ext-cuda/func2c.c grid.h ext-cuda/func2.h
ext-cuda/func2cu.o: ext-cuda/func2cu.cu ext-cuda/func2.h
ext-cuda/func3c.o: ext-cuda/func3c.c grid.h ext-cuda/func3.h
ext-cuda/func3cu.o: ext-cuda/func3cu.cu ext-cuda/func3.h
ext-cuda/func4c.o: ext-cuda/func4c.c grid.h ext-cuda/func4.h
ext-cuda/func4cu.o: ext-cuda/func4cu.cu ext-cuda/func4.h
ext-cuda/func5c.o: ext-cuda/func5c.c grid.h ext-cuda/func5.h
ext-cuda/func5cu.o: ext-cuda/func5cu.cu ext-cuda/func5.h
ext-cuda/func6c.o: ext-cuda/func6c.c grid.h ext-cuda/func6.h
ext-cuda/func6cu.o: ext-cuda/func6cu.cu ext-cuda/func6.h
ext-cuda/func7c.o: ext-cuda/func7c.c grid.h ext-cuda/func7.h
ext-cuda/func7cu.o: ext-cuda/func7cu.cu ext-cuda/func7.h

proto.h:
	egrep -h \^EXPORT *.c ext-cuda/func*.c | tr \{ \; | sed -e "s/EXPORT //g" > proto.h
ifeq ($(CUDA),yes)
	egrep -h extern\ \"C\" *.cu | sed -e 's/extern "C" //g' | tr '{' ';' >> proto.h
else
	fgrep -v cuda proto.h > tmp.h
	mv tmp.h proto.h
endif

install:
	-mkdir -p $(ROOT)/lib
	cp libgrid.a $(ROOT)/lib
	ranlib $(ROOT)/lib/libgrid.a
	-mkdir -p $(ROOT)/include/grid
	cp *.h $(ROOT)/include/grid
	-cp cuda.arch $(ROOT)/include/grid
	-cp cuda.status $(ROOT)/include/grid
	-cp cuda.hostcc $(ROOT)/include/grid
	cp ../make.conf $(ROOT)/include/grid

clean:
	-rm libgrid.a *.o *~ proto.h ext-cuda/*.o git-version.h cuda.arch cuda.status cuda.hostcc
