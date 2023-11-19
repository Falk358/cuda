#include <math.h>
#include <complex>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>


#define FUNCTION_RESOLUTION 10
#define BLOCK_SIZE 128
#define THREAD_COUNT 128


// implements f(x) = sin(2 * pi * x) for values 0.1 .. 1.0
double* initSine(size_t array_size) 
{
    double x_points[FUNCTION_RESOLUTION] = {0.1, 0.2, 0.3, 0.4 , 0.5 ,0.6, 0.7, 0.8, 0.9, 1.0};
    cufftDoubleReal* y_points = (cufftDoubleReal*) malloc(array_size);
    size_t x_length = sizeof(x_points) / sizeof(x_points[0]);
    for (int i = 0; i < x_length; i++)
    {
        y_points[i] = sin(M_PI*2*x_points[i]);
    }
    return y_points;

}

// implents u_k = f_k / (2*pi*k)²
__global__
void calcInComplexSpace(cufftDoubleComplex* f_k, cufftDoubleComplex* u_k, size_t n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n)
    {
        return;
    }
    else if (i == 0)
    {
        cufftDoubleComplex at_0;
        at_0.x = 0.0;
        at_0.y = 0.0;
        u_k[0] = at_0;
    }
    else
    {
        u_k[i].x = f_k[i].x / pow((2 * M_PI * i), 2.0);
        u_k[i].y = f_k[i].y / pow((2 * M_PI * i), 2.0);
    }
    return;
}





int main()
{
    size_t array_size = sizeof(cufftDoubleReal) * FUNCTION_RESOLUTION;
    size_t array_size_complex = (array_size / 2) + 1;
    double* y_points;
    double* device_y_points;
    y_points = initSine(array_size);
    cudaMalloc(&device_y_points, array_size);
    
    cudaMemcpy(device_y_points, y_points, array_size, cudaMemcpyHostToDevice);
    
    cufftHandle fft_handle;

    cufftResult plan_success = cufftPlan1d(&fft_handle, array_size, CUFFT_D2Z, 1);
    if (plan_success != CUFFT_SUCCESS)
    {
        std::cerr << "Error creating plan with array_size " << array_size << "with error: " << plan_success << std::endl; 
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // TODO FFT here
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << time*1e-3 << " s" << std::endl;
    



    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cufftDestroy(plan_success);
    

    free(y_points);
    cudaFree(device_y_points);
}