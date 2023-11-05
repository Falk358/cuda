## Performance Analysis


we can see that the biggest performance culprit is copying the result of computation at time step *T* back to host memory; we cannot ommit this since it is a requirement for the task. Other than that, slight speed up can be achieved by removing the `__syncthreads()` from the kernel and `cudaDeviceSynchronize()` at time step *T*.


Unfortunately , it is not possible to print the result of buffer *B* at timestep *T* directly in device Memory with `printTemperature()`, the biggest performance culprit is copying buffer *B* to *print_buffer* in host memory


Before optimization:

```
erification: OK
==103568== Profiling application: ../build/heat_stencil_cuda
==103568== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.87%  4.12015s     51201  80.470us  80.383us  90.751us  [CUDA memcpy DtoH]
                   13.13%  622.82ms     51200  12.164us  11.967us  14.432us  spreadHeat(float*, float*, int, int, int)
                    0.00%  88.159us         1  88.159us  88.159us  88.159us  [CUDA memcpy HtoD]
      API calls:   91.84%  15.9439s     51202  311.39us  212.31us  5.7494ms  cudaMemcpy
                    4.65%  807.37ms     51200  15.769us  3.2440us  601.62us  cudaDeviceSynchronize
                    2.04%  354.89ms     51200  6.9310us  6.0620us  1.7498ms  cudaLaunchKernel
                    1.42%  246.00ms         2  123.00ms  3.7470us  246.00ms  cudaMalloc
                    0.03%  5.4268ms         4  1.3567ms  1.2783ms  1.5267ms  cuDeviceTotalMem
                    0.02%  2.7531ms       404  6.8140us     265ns  332.36us  cuDeviceGetAttribute
                    0.00%  376.36us         2  188.18us  52.391us  323.97us  cudaFree
                    0.00%  269.77us         4  67.441us  60.985us  86.117us  cuDeviceGetName
                    0.00%  17.196us         4  4.2990us  2.2010us  9.4640us  cuDeviceGetPCIBusId
                    0.00%  3.2920us         8     411ns     284ns  1.1860us  cuDeviceGet
                    0.00%  2.6370us         3     879ns     477ns  1.5920us  cuDeviceGetCount
                    0.00%  1.8810us         4     470ns     381ns     579ns  cuDeviceGetUuid
```

after optimization:
```
==104291== Profiling application: ../build/heat_stencil_cuda
==104291== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.87%  4.12077s     51201  80.482us  80.383us  90.815us  [CUDA memcpy DtoH]
                   13.13%  622.69ms     51200  12.161us  11.967us  13.888us  spreadHeat(float*, float*, int, int, int)
                    0.00%  88.159us         1  88.159us  88.159us  88.159us  [CUDA memcpy HtoD]
      API calls:   96.45%  17.5314s     51202  342.40us  171.23us  6.2726ms  cudaMemcpy
                    1.75%  318.82ms     51200  6.2270us  5.4900us  554.16us  cudaLaunchKernel
                    1.75%  317.78ms         2  158.89ms  4.9970us  317.77ms  cudaMalloc
                    0.03%  5.5388ms         4  1.3847ms  1.2762ms  1.6226ms  cuDeviceTotalMem
                    0.02%  2.7320ms       404  6.7620us     268ns  343.37us  cuDeviceGetAttribute
                    0.00%  512.84us         2  256.42us  144.04us  368.79us  cudaFree
                    0.00%  261.25us         4  65.311us  58.468us  84.264us  cuDeviceGetName
                    0.00%  14.055us         4  3.5130us  1.4430us  8.6830us  cuDeviceGetPCIBusId
                    0.00%  3.3620us         8     420ns     254ns  1.2000us  cuDeviceGet
                    0.00%  2.5680us         3     856ns     395ns  1.6340us  cuDeviceGetCount
                    0.00%  1.6080us         4     402ns     314ns     590ns  cuDeviceGetUuid
```
