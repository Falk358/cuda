__global__ void vec_add(int* x, int*y, int* res)
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
    size_t bytes = vector_length*sizeof(int);
    int* h_mem_x = (int*) malloc(bytes);
    int* h_mem_y = (int*) malloc(bytes);
    int* h_mem_res = (int*) malloc(bytes);
    int* d_mem_x;
    int* d_mem_y;
    int* d_mem_res;
    cudaMalloc(&d_mem_x, bytes);
    cudaMalloc(&d_mem_y, bytes);
    cudaMalloc(&d_mem_res, bytes);
    
    // TODO fill vectors h_mem_x and h_mem_y with data

    //copy vectors to device
    cudaMemcpy(d_mem_x, h_mem_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem_y, h_mem_y, bytes, cudaMemcpyHostToDevice);

    //perform computation
    vec_add<<<1, block_size>>>(d_mem_x, d_mem_y, d_mem_res);

    // copy result back from device to host
    cudaMemcpy(h_mem_res, d_mem_res, bytes, cudaMemcpyDeviceToHost);

    // TODO verify and print result

    // free memory
    free(h_mem_x);
    free(h_mem_y);
    free(h_mem_res);

    cudaFree(d_mem_x);
    cudaFree(d_mem_y);
    cudaFree(d_mem_res);
    
}