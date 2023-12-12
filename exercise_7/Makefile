NVCC=nvcc
NVCC_FLAGS=-O3 -lcudadevrt -rdc=true -lpng -lgomp

mandelbrot-dyn: mandelbrot-dyn.cu
	@$(NVCC) $(NVCC_FLAGS) mandelbrot-dyn.cu -o ../build/mandelbrot-dyn

mandelbrot: mandelbrot.cu
	@$(NVCC) $(NVCC_FLAGS) mandelbrot.cu -o ../build/mandelbrot

.PHONEY: clean
clean:
	@rm ../build/mandelbrot-dyn
	@rm ../build/mandelbrot
	
