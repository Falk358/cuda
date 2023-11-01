#include <stdio.h>

__global__ void mathKernel(float* a) {
    a[1] = __cosf(a[0]); // a fast implementation of the cosine function
    return;
}

int main(int argc, char** argv) {
    int cudaRtrn;
    float *data;

    if(argc != 2) {
    	printf("usage: %s <value>\n", argv[0]);
	return 1;
    }

    if(cudaRtrn = cudaMallocManaged(&data, 2 * sizeof(float)) != 0) {
       printf("*** allocation failed for array data[], %d ***\n", cudaRtrn);
       return 1;
    }
    
    data[0] = atof(argv[1]);

    mathKernel<<<1, 1>>>(data);

    cudaDeviceSynchronize(); 

    printf("GPU: cos(%.15g) = %.15g\n", data[0], data[1]);
    printf("CPU: cos(%.15g) = %.15g\n", data[0], cosf(data[0]));

    cudaFree(data);

    return 0;
}
