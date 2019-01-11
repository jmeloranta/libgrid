
include ../make.conf

all: fft spherical

fft: fft.o
	$(CC) $(CFLAGS) -o fft fft.o $(LDFLAGS)

spherical: spherical.o
	$(CC) $(CFLAGS) -o spherical spherical.o $(LDFLAGS)

clean:
	-rm fft fft.o spherical spherical.o *~ *.x *.y *.z *.grd *.dat



