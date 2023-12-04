NVCC=nvcc
NVCC_FLAGS=-O3 
exercise-multiplegpu: exercise-multiplegpu.cu
	@$(NVCC) $(NVCC_FLAGS) exercise-multiplegpu.cu -o ../build/exercise-multiplegpu


.PHONEY: clean
clean:
	@rm ../build/exercise-multiplegpu
	
