
include ../make.conf

all: test_wf3d test_wf3d_itp_linear test_wf3d_itp_nonlinear \
     test_rfft3d test_convolution_r3d lintest1 lintest2 lintest3

lintest1.o: lintest1.c
lintest1: lintest1.o
	$(CC) $(CFLAGS) -o lintest1 lintest1.o $(LDFLAGS)

lintest2.o: lintest2.c
lintest2: lintest2.o
	$(CC) $(CFLAGS) -o lintest2 lintest2.o $(LDFLAGS)

lintest3.o: lintest3.c
lintest3: lintest3.o
	$(CC) $(CFLAGS) -o lintest3 lintest3.o $(LDFLAGS)

lintest4.o: lintest4.c
lintest4: lintest4.o
	$(CC) $(CFLAGS) -o lintest4 lintest4.o $(LDFLAGS)

test_wf3d.o: test_wf3d.c

test_wf3d: test_wf3d.o
	$(CC) $(CFLAGS) -o test_wf3d test_wf3d.o $(LDFLAGS)

test_wf3d_itp_linear: test_wf3d_itp_linear.o
	$(CC) $(CFLAGS) -o test_wf3d_itp_linear test_wf3d_itp_linear.o $(LDFLAGS)

test_wf3d_itp_linear.o: test_wf3d_itp_linear.c

test_wf3d_itp_nonlinear: test_wf3d_itp_nonlinear.o
	$(CC) $(CFLAGS) -o test_wf3d_itp_nonlinear test_wf3d_itp_nonlinear.o $(LDFLAGS)

test_wf3d_itp_nonlinear.o: test_wf3d_itp_nonlinear.c

test_rfft3d: test_rfft3d.o
	$(CC) $(CFLAGS) -o test_rfft3d test_rfft3d.o $(LDFLAGS)

test_convolution_r3d.o: test_convolution_r3d.c

test_convolution_r3d: test_convolution_r3d.o
	$(CC) $(CFLAGS) -o test_convolution_r3d test_convolution_r3d.o $(LDFLAGS)

test_convolution_r3d.o: test_convolution_r3d.c

clean:
	-rm *.o *~ 
	-rm test_wf3d test_wf3d_itp_linear test_wf3d_itp_nonlinear 
	-rm cfft-3d cfft-2d test_convolution_r3d test_rfft3d
	-rm lintest1 lintest2 lintest3
