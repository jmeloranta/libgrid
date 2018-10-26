include ../make.conf

PROGS = wf2cdf wf2cdf2 grd2cdf vector2cdf view view2

all: $(PROGS)

install: all
	cp $(PROGS) /usr/local/bin

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

view: view.o
	$(CC) $(CFLAGS) -o view view.o -lgrace_np

view.o: view.c
	$(CC) $(CFLAGS) -c view.c

view2: view2.o
	$(CC) $(CFLAGS) -o view2 view2.o -lgrace_np

view2.o: view2.c
	$(CC) $(CFLAGS) -c view2.c

clean:
	-rm grd2cdf wf2cdf wf2cdf2 *~ *.o vector2cdf view view2

