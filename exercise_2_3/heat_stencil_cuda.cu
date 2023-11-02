#include <stdio.h>
#include <stdlib.h>




void printTemperature(float* m, int N, int M);

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

#define gpuErrorCheck(ans, abort) { gpuAssert((ans), __FILE__, __LINE__, abort); }


__global__ void spreadHeat(float* current_state, float* next_state, int N, int source_x, int source_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > N || j > N)
    {
        return;
    }

    if (i == source_x && j == source_y)
    {
        next_state[i* N + j] = current_state[i *N + j];
        return;
    }
    float tc = current_state[i * N + j];

    // get temperatures left/right and up/down
    float tl = (j != 0) ? current_state[i * N + (j - 1)] : tc;
    float tr = (j != N - 1) ? current_state[i * N + (j + 1)] : tc;
    float tu = (i != 0) ? current_state[(i - 1) * N + j] : tc;
    float td = (i != N - 1) ? current_state[(i + 1) * N + j] : tc;

    __syncthreads();
    // update temperature at current point
    next_state[i * N + j] = tc + 0.2 * (tl + tr + tu + td + (-4 * tc));
    return;
}


int main(int argc, char** argv) 
{
    // 'parsing' optional input parameter = problem size
    int N = 512;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    int T = N*100;
    printf("Computing heat-distribution for room size N=%d for T=%d timesteps\n", N, T);
    // ---------- setup ----------
    size_t field_size = N*N*sizeof(float);
    // create a buffer for storing temperature fields
    float* starting_field =  (float*) malloc(field_size);
    
     dim3 threads_per_block(16,16);
     dim3 num_blocks((N+15)/threads_per_block.x, (N+15) /threads_per_block.y);
    
    // set up initial conditions in A
    for(int i = 0; i<N; i++) {
        for(int j = 0; j<N; j++) {
            starting_field[i*N+j] = 273;             // temperature is 0° C everywhere (273 K)
        }
    }

    // and there is a heat source in one corner
    int source_x = N/4;
    int source_y = N/4;
    starting_field[source_x*N+source_y] = 273 + 60;

    printf("Initial:\n");
    printTemperature(starting_field,N,N);
    
    //malloc device memory and send initialized field to device
    float* A;
    float* B;
    
    float* print_buffer = (float*) malloc(field_size);

    cudaMalloc(&A, field_size);
    cudaMalloc(&B,field_size);
    
    gpuErrorCheck(cudaMemcpy(A,starting_field, field_size, cudaMemcpyHostToDevice),true);

    // execute simulation with kernel
    for (int i = 0; i<T; i++)
    {
        spreadHeat<<<num_blocks, threads_per_block>>>(A, B, N, source_x, source_y);
        gpuErrorCheck(cudaDeviceSynchronize(),true);
        gpuErrorCheck(cudaMemcpy(print_buffer, B, field_size,cudaMemcpyDeviceToHost),false);
        if ((!i%1000))
        {
            printf("Step t=%d:\n", i);
            printTemperature(print_buffer, N, N);
        }

        float* temp;
        // swap pointers
        temp = A;
        A = B;
        B = temp;
    }
    

    // copy result back
    


    cudaFree(&A);
    cudaFree(&B);
    free(print_buffer);
    free(starting_field);
}



void printTemperature(float* m, int N, int M) 
{
    const char* colors = " .-:=+*#%@";
    const int numColors = 10;

    // boundaries for temperature (for simplicity hard-coded)
    const float max = 273 + 30;
    const float min = 273 + 0;

    // set the 'render' resolution
    int H = 30;
    int W = 50;

    // step size in each dimension
    int sH = N/H;
    int sW = M/W;


    // upper wall
    for(int i=0; i<W+2; i++) {
        printf("X");
    }
    printf("\n");

    // room
    for(int i=0; i<H; i++) {
        // left wall
        printf("X");
        // actual room
        for(int j=0; j<W; j++) {

            // get max temperature in this tile
            float max_t = 0;
            for(int x=sH*i; x<sH*i+sH; x++) {
                for(int y=sW*j; y<sW*j+sW; y++) {
                    max_t = (max_t < m[x*N+y]) ? m[x*N+y] : max_t;
                }
            }
            float temp = max_t;

            // pick the 'color'
            int c = ((temp - min) / (max - min)) * numColors;
            c = (c >= numColors) ? numColors-1 : ((c < 0) ? 0 : c);

            // print the average temperature
            printf("%c",colors[c]);
        }
        // right wall
        printf("X\n");
    }

    // lower wall
    for(int i=0; i<W+2; i++) {
        printf("X");
    }
    printf("\n");

}