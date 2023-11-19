NVCC=nvcc
NVCC_FLAGS=-O3 -lcufft
fft: fft.cu
	@$(NVCC) $(NVCC_FLAGS) fft.cu -o ../build/fft


.PHONEY: clean
clean:
	@rm ../build/fft
	
