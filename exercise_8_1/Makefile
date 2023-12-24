NVCC=nvcc
NVCC_FLAGS=-O3 

exercise_advanced_einkemmer: exercise_advanced_einkemmer.cu
	@$(NVCC) $(NVCC_FLAGS) exercise_advanced_einkemmer.cu -o ../build/exercise_advanced_einkemmer

.PHONEY: clean
clean:
	@rm ../build/exercise_advanced_einkemmer
	
