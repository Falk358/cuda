#include <stdio.h>
#include <time.h>

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    mathKernel<<<1, 1>>>(data);
    cudaEventRecord(stop,0);
    cudaDeviceSynchronize(); 
    cudaEventSynchronize(stop);

    clock_t start_cpu = clock();
    double cos_cpu = cosf(data[0]);
    clock_t end_cpu = clock();
    double time_cpu = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop);
    printf("GPU: cos(%.15g) = %.15g\n", data[0], data[1]);
    printf("CPU: cos(%.15g) = %.15g\n", data[0], cos_cpu);
    
    printf("time cpu: %20.16lf s\n",time_cpu);
    printf("time gpu: %f s\n",time_gpu);

    cudaFree(data);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
