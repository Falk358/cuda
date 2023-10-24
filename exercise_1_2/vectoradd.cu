#include <stdio.h>



__global__ void vec_add(double* x, double*y, double* res)
{
    int i = threadIdx.x;
    res[i] = x[i] + y[i];
}


int main()
{
    // get device properties
    cudaSetDevice(0);
    cudaDeviceProp prop; 
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    int block_size = prop.maxThreadsPerBlock;
    int vector_length = 1024;

    // memory allocation
    size_t bytes = vector_length*sizeof(double);
    double* h_mem_x = (double*) malloc(bytes);
    double* h_mem_y = (double*) malloc(bytes);
    double* h_mem_res = (double*) malloc(bytes);
    double* d_mem_x;
    double* d_mem_y;
    double* d_mem_res;
    cudaMalloc(&d_mem_x, bytes);
    cudaMalloc(&d_mem_y, bytes);
    cudaMalloc(&d_mem_res, bytes);
    
    // TODO fill vectors h_mem_x and h_mem_y with data
    for (int i = 0; i < vector_length; i++)
    {
        h_mem_x[i] = 1.0;
        h_mem_y[i] = 2.0;
    }

    //copy vectors to device
    cudaMemcpy(d_mem_x, h_mem_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_y, h_mem_y, bytes, cudaMemcpyHostToDevice);

    //perform computation
    vec_add<<<1, block_size>>>(d_mem_x, d_mem_y, d_mem_res);

    // copy result back from device to host
    cudaMemcpy(h_mem_res, d_mem_res, bytes, cudaMemcpyDeviceToHost);

    // TODO verify and print result
    double tolerance = 1.0e-14;
    bool error_compute = false;
    for (int i = 0; i < vector_length; i++)
    {
        if (h_mem_res[i] - 3.0 > tolerance)
        {
            printf("error computing vector addition; Error exceeds %f at index %d of solution\n", tolerance, i);
            error_compute = true;
        }
    }
    // free memory
    free(h_mem_x);
    free(h_mem_y);
    free(h_mem_res);

    cudaFree(d_mem_x);
    cudaFree(d_mem_y);
    cudaFree(d_mem_res);

    if (error_compute)
    {
        exit(1);
    }
    else
    {
        printf("Success! No errors during computation!\n");
        exit(0);
    }
    
}