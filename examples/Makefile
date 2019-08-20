
include ../make.conf

all: fft spherical poisson wavepacket gradient decomp veloc

decomp: decomp.o
	$(CC) $(CFLAGS) -o decomp decomp.o $(LDFLAGS)

fft: fft.o
	$(CC) $(CFLAGS) -o fft fft.o $(LDFLAGS)

spherical: spherical.o
	$(CC) $(CFLAGS) -o spherical spherical.o $(LDFLAGS)

poisson: poisson.o
	$(CC) $(CFLAGS) -o poisson poisson.o $(LDFLAGS)

wavepacket: wavepacket.o
	$(CC) $(CFLAGS) -o wavepacket wavepacket.o $(LDFLAGS)

gradient: gradient.o
	$(CC) $(CFLAGS) -o gradient gradient.o $(LDFLAGS)

veloc: veloc.o
	$(CC) $(CFLAGS) -o veloc veloc.o $(LDFLAGS)

clean:
	-rm fft fft.o spherical spherical.o poisson poisson.o *~ *.x *.y *.z *.grd *.dat
	-rm wavepacket.o wavepacket gradient.o gradient comp.dat incomp.dat decomp decomp.o
	-rm veloc veloc.o v{x,y,z}-fd.* v{x,y,z}-fft.*


