
include ../make.conf

all: fft cfft spherical poisson wavepacket gradient decomp veloc random integrate cintegrate cpoisson cgradient

decomp: decomp.o
	$(CC) $(CFLAGS) -o decomp decomp.o $(LDFLAGS)

fft: fft.o
	$(CC) $(CFLAGS) -o fft fft.o $(LDFLAGS)

cfft: cfft.o
	$(CC) $(CFLAGS) -o cfft cfft.o $(LDFLAGS)

spherical: spherical.o
	$(CC) $(CFLAGS) -o spherical spherical.o $(LDFLAGS)

poisson: poisson.o
	$(CC) $(CFLAGS) -o poisson poisson.o $(LDFLAGS)

cpoisson: cpoisson.o
	$(CC) $(CFLAGS) -o cpoisson cpoisson.o $(LDFLAGS)

wavepacket: wavepacket.o
	$(CC) $(CFLAGS) -o wavepacket wavepacket.o $(LDFLAGS)

gradient: gradient.o
	$(CC) $(CFLAGS) -o gradient gradient.o $(LDFLAGS)

cgradient: cgradient.o
	$(CC) $(CFLAGS) -o cgradient cgradient.o $(LDFLAGS)

veloc: veloc.o
	$(CC) $(CFLAGS) -o veloc veloc.o $(LDFLAGS)

random: random.o
	$(CC) $(CFLAGS) -o random random.o $(LDFLAGS)

integrate: integrate.o
	$(CC) $(CFLAGS) -o integrate integrate.o $(LDFLAGS)

cintegrate: cintegrate.o
	$(CC) $(CFLAGS) -o cintegrate cintegrate.o $(LDFLAGS)

clean:
	-rm fft fft.o spherical spherical.o poisson poisson.o *~ *.x *.y *.z *.grd *.dat
	-rm wavepacket.o wavepacket gradient.o gradient cgradient.o cgradient comp.dat incomp.dat decomp decomp.o
	-rm veloc veloc.o random random.o v{x,y,z}-fd.* v{x,y,z}-fft.* integrate integrate.o
	-rm cfft cfft.o cintegrate cintegrate.o cpoisson cpoisson.o


