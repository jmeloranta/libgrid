#
# To add new ext/function, add another entry to SRC and NVSRC below
#

CUDA = $(shell ./cuda-detect.sh)
ifeq ($(CUDA),yes)
CUDA_CC = $(shell ./get_cc.sh)
endif

include ../make.conf

all:
ifeq ($(CUDA),yes)
	echo "CUDA enabled for GPU architecture " $(CUDA_CC)
	echo $(CUDA_CC) > cuda.arch
endif
	echo $(CUDA) > cuda.status
	make prototypes
	make libgrid.a

SRC = cgrid.c cgrid_fft.c rgrid.c rgrid_fft.c ngrid.c grid_wf.c grid_wf_fft.c grid_wf_diag.c grid_wf_cn.c grid_itp.c cgrid_bc.c \
      rgrid_bc.c threads.c linalg.c ctimer.c other.c lapack.c
SRC += ext-cuda/func1c.c ext-cuda/func2c.c ext-cuda/func3c.c ext-cuda/func4c.c ext-cuda/func5c.c ext-cuda/func6c.c ext-cuda/func7c.c

ifeq ($(CUDA),yes)
  SRC += cgrid-cuda.c cgrid_fft-cuda.c rgrid-cuda.c rgrid_fft-cuda.c grid_wf-cuda.c grid_wf_fft-cuda.c grid_wf_cn-cuda.c cuda.c
  NVSRC = cgrid-cuda2.cu rgrid-cuda2.cu grid_wf-cuda2.cu grid_wf_fft-cuda2.cu grid_wf_cn-cuda2.cu
  NVSRC += ext-cuda/func1cu.cu ext-cuda/func2cu.cu ext-cuda/func3cu.cu ext-cuda/func4cu.cu ext-cuda/func5cu.cu ext-cuda/func6cu.cu ext-cuda/func7cu.cu
endif

OBJS = $(SRC:.c=.o) $(NVSRC:.cu=.o)

libgrid.a: $(OBJS)
	$(AR) cr libgrid.a $(OBJS)
	$(RANLIB) libgrid.a

%.o: %.cu
	nvcc $(NVCFLAGS) -c -o $@ $<

cuda.o: cuda.c cuda.h
grid_wf_cn-cuda2.o: grid_wf_cn-cuda2.cu linalg-cuda.h

prototypes:
	egrep -h \^EXPORT *.c ext-cuda/func*.c | tr \{ \; | sed -e "s/EXPORT //g" > proto.h
ifeq ($(CUDA),yes)
	egrep -h extern\ \"C\" *.cu | sed -e 's/extern "C" //g' | tr '{' ';' >> proto.h
else
	fgrep -v cuda proto.h > tmp.h
	mv tmp.h proto.h
endif

install:
	cp libgrid.a $(ROOT)/lib
	$(RANLIB) $(ROOT)/lib/libgrid.a
	-mkdir $(ROOT)/include/grid
	cp *.h $(ROOT)/include/grid
	-cp cuda.arch $(ROOT)/include/grid
	-cp cuda.status $(ROOT)/include/grid
	cp ../make.conf $(ROOT)/include/grid

clean:
	-rm libgrid.a *.o *~ proto.h ext-cuda/*.o
