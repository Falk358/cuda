## Starting point

profiling results of given file (no optimizations performed):

```
==205249== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.32%  1.27759s   1000001  1.2770us  1.2150us  648.48us  [CUDA memcpy HtoD]
                    0.42%  5.4268ms       100  54.268us  53.792us  56.256us  k_matvecmul(long, double*, double*, double*)
                    0.21%  2.7068ms       100  27.068us  26.432us  27.872us  [CUDA memcpy DtoD]
                    0.05%  607.77us         1  607.77us  607.77us  607.77us  [CUDA memcpy DtoH]
      API calls:   95.46%  9.11093s   1000102  9.1100us  3.7190us  17.031ms  cudaMemcpy
                    4.38%  417.74ms         2  208.87ms  2.8590us  417.74ms  cudaEventCreate
                    0.06%  5.7403ms         1  5.7403ms  5.7403ms  5.7403ms  cudaHostAlloc
                    0.06%  5.5393ms         4  1.3848ms  1.2530ms  1.7796ms  cuDeviceTotalMem
                    0.03%  2.6723ms       404  6.6140us     245ns  337.74us  cuDeviceGetAttribute
                    0.01%  839.87us         3  279.96us  218.69us  340.56us  cudaMalloc
                    0.01%  523.46us       100  5.2340us  4.1090us  68.604us  cudaLaunchKernel
                    0.00%  285.97us         4  71.493us  58.827us  107.24us  cuDeviceGetName
                    0.00%  26.023us         2  13.011us  8.1730us  17.850us  cudaEventRecord
                    0.00%  18.551us         4  4.6370us  2.2030us  9.8800us  cuDeviceGetPCIBusId
                    0.00%  13.587us         1  13.587us  13.587us  13.587us  cudaEventSynchronize
                    0.00%  5.6870us         1  5.6870us  5.6870us  5.6870us  cudaEventElapsedTime
                    0.00%  4.5060us         2  2.2530us     546ns  3.9600us  cudaEventDestroy
                    0.00%  3.6120us         8     451ns     265ns  1.3960us  cuDeviceGet
                    0.00%  2.7720us         3     924ns     614ns  1.4780us  cuDeviceGetCount
                    0.00%  1.5240us         4     381ns     331ns     485ns  cuDeviceGetUuid
```