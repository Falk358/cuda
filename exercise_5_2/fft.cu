#include <math.h>
#include <complex>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>


#define FUNCTION_RESOLUTION 10
#define BLOCK_SIZE 128
#define THREAD_COUNT 128


// implements f(x) = sin(2 * pi * x) for values 0.1 .. 1.0
double* initSine(size_t array_size, double* x_points) 
{
    cufftDoubleReal* y_points = (cufftDoubleReal*) malloc(array_size);
    size_t x_length = sizeof(x_points) / sizeof(x_points[0]);
    for (int i = 0; i < x_length; i++)
    {
        y_points[i] = sin(M_PI*2*x_points[i]);
    }
    return y_points;

}

// verfiy with analytical solution f''(x) = 1/(2*pi)² sin(2*pi*x)
bool verifyResult(cufftDoubleReal* result, double tolerance, double* x_points)
{
    bool verified = true;
    size_t x_length = sizeof(x_points) / sizeof(x_points[0]);
    for (int i = 0; i< x_length; i++)
    {
        double analytical_solution = (1/pow(2.0 * M_PI,2.0)) * sin(2.0*M_PI*x_points[i]);
        double error = fabs(analytical_solution - result[i]);
        if (error > tolerance)
        {
            verified = false;
            std::cerr << "Error verifying at index " << i << "\n error: " << error << "\n analytical solution: " << analytical_solution << "\n calculated solution: " << result[i] << std::endl;
        }
    }
    return verified;
    
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
    size_t array_size_freq = ((FUNCTION_RESOLUTION/ 2) + 1) * sizeof(cufftDoubleComplex);
    cufftDoubleReal* y_points;
    cufftDoubleReal* u_points;
    cufftDoubleReal* device_y_points;
    cufftDoubleReal* device_u_points;
    cufftDoubleComplex* device_y_points_freq;
    cufftDoubleComplex* device_u_points_freq;

    double x_points[FUNCTION_RESOLUTION] = {0.1, 0.2, 0.3, 0.4 , 0.5 ,0.6, 0.7, 0.8, 0.9, 1.0};
    y_points = initSine(array_size, x_points);
    u_points = (cufftDoubleReal*) malloc(array_size);
    cudaMalloc(&device_y_points, array_size);
    cudaMalloc(&device_u_points, array_size);
    cudaMalloc(&device_y_points_freq, array_size_freq);
    cudaMalloc(&device_u_points_freq, array_size_freq);
    cudaMemcpy(device_y_points, y_points, array_size, cudaMemcpyHostToDevice);
    
    cufftHandle fft_forward_handle;
    
    cufftHandle fft_backward_handle;

    cufftResult plan_forward_success = cufftPlan1d(&fft_forward_handle, array_size, CUFFT_D2Z, 1);
    if (plan_forward_success  != CUFFT_SUCCESS)
    {
        std::cerr << "Error creating plan with array_size " << array_size << " with error: " << plan_forward_success << std::endl; 
    }
    cufftResult plan_backward_success = cufftPlan1d(&fft_backward_handle, array_size, CUFFT_Z2D, 1);
    if (plan_backward_success != CUFFT_SUCCESS)
    {
        std::cerr << "Error creating plan with array_size " << array_size << " with error: " << plan_backward_success << std::endl; 
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // TODO FFT here
    cufftResult res_forward = cufftExecD2Z(fft_forward_handle, device_y_points, device_y_points_freq);
    if (res_forward != CUFFT_SUCCESS)
    {
        std::cerr << "Error computing frequency domain of y points. Message: " << res_forward << std::endl;
    }
    calcInComplexSpace<<<BLOCK_SIZE, THREAD_COUNT>>>(device_y_points_freq, device_u_points_freq, array_size_freq);
    // TODO convert u back to space domain (backward)
    cufftResult res_backward = cufftExecZ2D(fft_backward_handle, device_u_points_freq, device_u_points);
    if (res_backward != CUFFT_SUCCESS)
    {
        std::cerr << "Error computing space domain from frequency domain of u points. Message: " << res_backward << std::endl;
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << time*1e-3 << " s" << std::endl;
    cudaMemcpy(u_points, device_u_points, array_size, cudaMemcpyDeviceToHost);

    bool verified = verifyResult(u_points, 0.1, x_points);
    if (verified)
    {
        std::cout << "SUCCESS!!! result was verified with analytical solution!!!" << std::endl;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cufftDestroy(plan_forward_success);
    cufftDestroy(plan_backward_success);
    

    free(y_points);
    free(u_points);
    cudaFree(device_u_points);
    cudaFree(device_y_points);
    cudaFree(device_y_points_freq);
    cudaFree(device_u_points_freq);
}