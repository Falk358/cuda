NVCC=nvcc
NVCC_FLAGS=-O3 -lcudadevrt -rdc=true -lpng -lgomp

mandelbrot-dyn: mandelbrot-dyn.cu
	@$(NVCC) $(NVCC_FLAGS) mandelbrot-dyn.cu -o ../build/mandelbrot-dyn


.PHONEY: clean
clean:
	@rm ../build/mandelbrot-dyn
	
