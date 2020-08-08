libgrid-manual.pdf: gendoc libgrid-manual.tex
	$(shell ./gendoc ../src/rgrid*.c > real-functions.tex)
	$(shell ./gendoc ../src/cgrid*.c > complex-functions.tex)
	$(shell ./gendoc ../src/ngrid*.c > mixed-functions.tex)
	$(shell ./gendoc ../src/grid_wf*.c > wf-functions.tex)	
	$(shell ./gendoc ../src/rand.c > random-functions.tex)
	$(shell ./gendoc ../src/threads.c > thread-functions.tex)
	$(shell ./gendoc ../src/grid_fft.c > fftw-functions.tex)
	$(shell ./gendoc ../src/linalg.c > linalg-functions.tex)
	$(shell ./gendoc ../src/interpolate.c > interpolate-functions.tex)
	$(shell ./gendoc ../src/ctimer.c > timer-functions.tex)
	$(shell ./gendoc ../src/cuda.c > cuda-functions.tex)
	$(shell ./gendoc ../src/ext/*.c > cuda-ext-functions.tex)
	pdflatex libgrid-manual

gendoc: gendoc.o
gendoc.o: gendoc.c

clean:
	-rm *.aux *.log *~ *.synctex.gz *.pdf gendoc gendoc.o
	-rm real-functions.tex complex-functions.tex mixed-functions.tex
	-rm wf-functions.tex random-functions.tex thread-functions.tex
	-rm linalg-functions.tex interpolate-functions.tex
	-rm timer-functions.tex cuda-functions.tex cuda-ext-functions.tex

