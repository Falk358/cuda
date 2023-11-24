NVCC=nvcc
NVCC_FLAGS=-g -G 


vector: vector.cu
	@$(NVCC) $(NVCC_FLAGS) vector.cu -o ../build/vector


.PHONEY: clean
clean:
	@rm ../build/vector
	
