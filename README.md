# cuda
Programming exercises for the course " VU HPC Implementierungen B: Beschleuniger"

## Execution

First, start by creating a `build` directory in the root folder of the project; it has been added to `.gitignore` so it wont be tracked by git. All your executables go here.
Each exercise is contained in a folder which is numbered accordingly: Please move to the folder and run `nvcc -O3 program.cu -o ../build/program` to compile and run it with `./build/program`.
