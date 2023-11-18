NVCC=nvcc
NVCC_FLAGS=-O3 -lcublas


matmul: matmul.cu
	@$(NVCC) $(NVCC_FLAGS) matmul.cu -o ../build/matmul


.PHONEY: clean
clean:
	@rm ../build/matmul
	
