
include ../make.conf

all: fft cfft spherical cspherical poisson wavepacket gradient decomp veloc random integrate cintegrate cpoisson cgradient fspherical bench minmax fspherical2

decomp: decomp.o
	$(CC) $(CFLAGS) -o decomp decomp.o $(LDFLAGS)

fft: fft.o
	$(CC) $(CFLAGS) -o fft fft.o $(LDFLAGS)

cfft: cfft.o
	$(CC) $(CFLAGS) -o cfft cfft.o $(LDFLAGS)

spherical: spherical.o
	$(CC) $(CFLAGS) -o spherical spherical.o $(LDFLAGS)

cspherical: cspherical.o
	$(CC) $(CFLAGS) -o cspherical cspherical.o $(LDFLAGS)

fspherical: fspherical.o
	$(CC) $(CFLAGS) -o fspherical fspherical.o $(LDFLAGS)

fspherical2: fspherical2.o
	$(CC) $(CFLAGS) -o fspherical2 fspherical2.o $(LDFLAGS)

poisson: poisson.o
	$(CC) $(CFLAGS) -o poisson poisson.o $(LDFLAGS)

cpoisson: cpoisson.o
	$(CC) $(CFLAGS) -o cpoisson cpoisson.o $(LDFLAGS)

bench: bench.o
	$(CC) $(CFLAGS) -o bench bench.o $(LDFLAGS)

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

minmax: minmax.o
	$(CC) $(CFLAGS) -o minmax minmax.o $(LDFLAGS)

clean:
	-rm fft fft.o spherical spherical.o poisson poisson.o *~ *.x *.y *.z *.grd *.dat
	-rm wavepacket.o wavepacket gradient.o gradient cgradient.o cgradient comp.dat incomp.dat decomp decomp.o
	-rm veloc veloc.o random random.o v{x,y,z}-fd.* v{x,y,z}-fft.* integrate integrate.o minmax.o minmax
	-rm cfft cfft.o cintegrate cintegrate.o cpoisson cpoisson.o cspherical cspherical.o fspherical.o fspherical bench.o bench


