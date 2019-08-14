include ../make.conf

PROGS = wf2cdf wf2cdf2 grd2cdf vector2cdf gview gview2 gview3

all: $(PROGS)

install: all
	-mkdir -p $(ROOT)/bin
	cp $(PROGS) $(ROOT)/bin

vector2cdf: vector2cdf.o
	$(CC) $(CFLAGS) -o vector2cdf vector2cdf.o $(LDFLAGS) -lnetcdf

vector2cdf.o: vector2cdf.c
	$(CC) $(CFLAGS) -c vector2cdf.c

wf2cdf: wf2cdf.o
	$(CC) $(CFLAGS) -o wf2cdf wf2cdf.o $(LDFLAGS) -lnetcdf

wf2cdf2: wf2cdf2.o
	$(CC) $(CFLAGS) -o wf2cdf2 wf2cdf2.o $(LDFLAGS) -lnetcdf

grd2cdf: grd2cdf.o
	$(CC) $(CFLAGS) -o grd2cdf grd2cdf.o $(LDFLAGS) -lnetcdf

grd2cdf.o: grd2cdf.c
	$(CC) $(CFLAGS) -c grd2cdf.c

wf2cdf.o: wf2cdf.c
	$(CC) $(CFLAGS) -c wf2cdf.c

wf2cdf2.o: wf2cdf2.c
	$(CC) $(CFLAGS) -c wf2cdf2.c

gview: gview.o
	$(CC) $(CFLAGS) -o gview gview.o -lgrace_np

gview.o: gview.c
	$(CC) $(CFLAGS) -c gview.c

gview2: gview2.o
	$(CC) $(CFLAGS) -o gview2 gview2.o -lgrace_np

gview2.o: gview2.c
	$(CC) $(CFLAGS) -c gview2.c

gview3: gview3.o
	$(CC) $(CFLAGS) -o gview3 gview3.o -lgrace_np

gview3.o: gview3.c
	$(CC) $(CFLAGS) -c gview3.c

clean:
	-rm grd2cdf wf2cdf wf2cdf2 *~ *.o vector2cdf gview gview2 gview3

