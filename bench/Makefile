
include ../make.conf

all: fftw_bench cufft_bench

fftw_bench: fftw_bench.c
	$(CC) $(CFLAGS) fftw_bench.c -o fftw_bench $(LDFLAGS)

cufft_bench: cufft_bench.c
	$(CC) $(CFLAGS) -I/usr/include/cuda cufft_bench.c -o cufft_bench $(LDFLAGS)

clean:
	-rm fftw_bench cufft_bench *~
