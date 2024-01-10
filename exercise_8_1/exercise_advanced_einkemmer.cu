#include <iostream>
#include <cmath>
using namespace std;


__global__
void k_sum(long n, double* vec, double* result) {
    // TODO
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    __shared__ double shared_buf[256]; 
    
    if (i >= n)
    {
        return;
    }
    shared_buf[threadIdx.x] = vec[i];
    __syncthreads();

    double inter_result = 0.0;
    for (int j = 0; j< 256; j++)
    {
        inter_result = inter_result + shared_buf[j];
    }
    result[blockIdx.x] = inter_result;
}


int main() {
    long n = 8*1024*1024; // must be a power of 2
    double h_result = 0.0;
    double *h_vec, *d_vec, *d_tmp, *h_tmp;

    cudaMallocHost(&h_vec, sizeof(double)*n);
    cudaMalloc(&d_vec, sizeof(double)*n);
    cudaMalloc(&d_tmp, sizeof(double)*(n/256));
    cudaMallocHost(&h_tmp, sizeof(double)*(n/256));

    // Initialie vec and copy to GPU.
    for(long i=0;i<n;i++)
        h_vec[i] = 1.0/pow(double(i+1),2);
    cudaMemcpy(d_vec, h_vec, sizeof(double)*n, cudaMemcpyHostToDevice);

    // TODO
    int threads_per_block = 256;
    int num_blocks = n / threads_per_block +1;
    k_sum<<<num_blocks,threads_per_block>>>(n, d_vec,d_tmp);
    cudaMemcpy(h_tmp, d_tmp, sizeof(double)*(n/256), cudaMemcpyDeviceToHost);
    

    for (int i = 0; i< (n/256); i++)
    {
        //cout << "h_tmp[" << i << "]: " << h_tmp[i] << endl;
        h_result = h_result + h_tmp[i];
    }
    // Check the result.
    cout << "Result: " << h_result << endl;
    if(fabs(h_result - pow(M_PI,2)/6.0) < 1e-5) {
        cout << "Correct!" << endl;
    } else {
        cout << "The computed result does not match with the expected result ("
             << pow(M_PI,2)/6.0 << ")" << endl;
    }

    return 0;
}
