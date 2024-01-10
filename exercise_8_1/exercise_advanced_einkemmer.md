 # Performance measuring


The code, which sums up all elements of a vector using shared memory, has the following performance:
```
=6050== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   88.63%  5.4298ms         1  5.4298ms  5.4298ms  5.4298ms  [CUDA memcpy HtoD]
                   11.02%  674.84us         1  674.84us  674.84us  674.84us  k_sum(long, double*, double*)
                    0.36%  21.792us         1  21.792us  21.792us  21.792us  [CUDA memcpy DtoH]
      API calls:   95.49%  355.91ms         2  177.96ms  1.4283ms  354.48ms  cudaHostAlloc
                    1.76%  6.5710ms         4  1.6427ms  1.6110ms  1.6774ms  cuDeviceTotalMem
                    1.67%  6.2115ms         2  3.1057ms  706.62us  5.5049ms  cudaMemcpy
                    0.82%  3.0552ms       404  7.5620us     179ns  356.06us  cuDeviceGetAttribute
                    0.16%  601.18us         2  300.59us  222.55us  378.63us  cudaMalloc
                    0.08%  303.12us         4  75.780us  70.247us  91.400us  cuDeviceGetName
                    0.01%  46.496us         1  46.496us  46.496us  46.496us  cudaLaunchKernel
                    0.00%  15.825us         4  3.9560us  1.9990us  8.5390us  cuDeviceGetPCIBusId
                    0.00%  3.7420us         8     467ns     309ns  1.3670us  cuDeviceGet
                    0.00%  2.8670us         3     955ns     381ns  1.9490us  cuDeviceGetCount
                    0.00%  1.7690us         4     442ns     218ns     654ns  cuDeviceGetUuid
```

As you can see, the computation takes `674.84` us to complete.