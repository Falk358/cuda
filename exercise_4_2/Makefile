NVCC=nvcc
NVCC_FLAGS=-g -G 


syncthreads: syncthreads.cu
	@$(NVCC) $(NVCC_FLAGS) syncthreads.cu -o ../build/syncthreads


.PHONEY: clean
clean:
	@rm ../build/syncthreads
	
