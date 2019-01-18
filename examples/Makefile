
include ../make.conf

all: fft spherical poisson wavepacket

fft: fft.o
	$(CC) $(CFLAGS) -o fft fft.o $(LDFLAGS)

spherical: spherical.o
	$(CC) $(CFLAGS) -o spherical spherical.o $(LDFLAGS)

poisson: poisson.o
	$(CC) $(CFLAGS) -o poisson poisson.o $(LDFLAGS)

wavepacket: wavepacket.o
	$(CC) $(CFLAGS) -o wavepacket wavepacket.o $(LDFLAGS)

clean:
	-rm fft fft.o spherical spherical.o poisson poisson.o *~ *.x *.y *.z *.grd *.dat
	-rm wavepacket.o wavepacket


