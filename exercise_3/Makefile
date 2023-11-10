NVCC=nvcc
NVCC_FLAGS=-O3


exercise_efficiency: exercise_efficiency.cu
	@$(NVCC) $(NVCC_FLAGS) exercise_efficiency.cu -o ../build/exercise_efficiency


.PHONEY: clean
clean:
	@rm ../build/exercise_efficiency
	