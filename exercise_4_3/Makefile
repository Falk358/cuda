NVCC=nvcc
NVCC_FLAGS=-g -G 


stencil: stencil.cu
	@$(NVCC) $(NVCC_FLAGS) stencil.cu -o ../build/stencil


.PHONEY: clean
clean:
	@rm ../build/stencil
	
