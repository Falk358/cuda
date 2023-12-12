# Discussion of results




We profiled the single gpu run and compared it to the multiple gpu run, here are the results by the **Nvidia Profiler**:


## Single GPU

```
==17700== NVPROF is profiling process 17700, command: ../build/exercise-singlegpu
Runtime: 3.73078 s
Error: 0.0235549
==17700== Profiling application: ../build/exercise-singlegpu
==17700== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.45%  3.72384s      8192  454.57us  453.44us  465.15us  k_upwind(long, long, double*, double*)
                    0.54%  20.343ms         2  10.172ms  10.168ms  10.176ms  [CUDA memcpy DtoH]
                    0.01%  420.77us         2  210.38us  208.61us  212.16us  k_init(long, long, double, double, double*)
      API calls:   77.79%  3.26291s      8194  398.21us  5.3050us  1.1970ms  cudaLaunchKernel
                   11.08%  464.92ms         1  464.92ms  464.92ms  464.92ms  cudaDeviceSynchronize
                    7.81%  327.44ms         2  163.72ms  536.65us  326.90ms  cudaMalloc
                    2.64%  110.75ms         2  55.374ms  53.268ms  57.481ms  cudaHostAlloc
                    0.49%  20.579ms         2  10.289ms  10.184ms  10.395ms  cudaMemcpy
                    0.11%  4.7022ms         4  1.1755ms  890.35us  1.2848ms  cuDeviceTotalMem
                    0.06%  2.6749ms       404  6.6210us     230ns  337.53us  cuDeviceGetAttribute
                    0.01%  473.31us         2  236.65us  11.600us  461.71us  cudaEventRecord
                    0.01%  283.61us         4  70.902us  52.756us  86.996us  cuDeviceGetName
                    0.00%  15.015us         4  3.7530us  2.2750us  7.5170us  cuDeviceGetPCIBusId
                    0.00%  11.830us         2  5.9150us  1.4140us  10.416us  cudaEventCreate
                    0.00%  10.252us         1  10.252us  10.252us  10.252us  cudaEventElapsedTime
                    0.00%  3.6790us         8     459ns     260ns  1.3240us  cuDeviceGet
                    0.00%  2.8020us         3     934ns     511ns  1.6850us  cuDeviceGetCount
                    0.00%  1.9190us         4     479ns     294ns     750ns  cuDeviceGetUuid
```


## Multiple (2) GPUs


```
==17247== NVPROF is profiling process 17247, command: ../build/exercise-multiplegpu
Runtime: 1.89683 s
Error: 0.204279
==17247== Profiling application: ../build/exercise-multiplegpu
==17247== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.91%  1.87882s      8192  229.35us  228.32us  239.01us  k_upwind(long, long, double*, double*)
                    1.07%  20.326ms         4  5.0814ms  5.0810ms  5.0818ms  [CUDA memcpy DtoH]
                    0.02%  428.83us         4  107.21us  106.46us  108.19us  k_init(long, long, double, double, double*)
      API calls:   59.95%  1.65394s     16388  100.92us     717ns  2.0412ms  cudaLaunchKernel
                   26.44%  729.51ms         2  364.75ms  353.26ms  376.25ms  cudaStreamCreate
                    8.61%  237.63ms         1  237.63ms  237.63ms  237.63ms  cudaDeviceSynchronize
                    4.03%  111.06ms         2  55.529ms  53.781ms  57.277ms  cudaHostAlloc
                    0.60%  16.478ms         4  4.1194ms  390.99us  14.440ms  cudaMallocAsync
                    0.15%  4.2251ms         4  1.0563ms  940.08us  1.1689ms  cuDeviceTotalMem
                    0.10%  2.8147ms         2  1.4074ms  1.2213ms  1.5935ms  cudaStreamDestroy
                    0.09%  2.6071ms       404  6.4530us     243ns  312.96us  cuDeviceGetAttribute
                    0.01%  255.96us         4  63.990us  52.749us  85.772us  cuDeviceGetName
                    0.00%  87.721us         4  21.930us  4.0940us  64.384us  cudaMemcpyAsync
                    0.00%  60.864us         4  15.216us  1.4150us  35.172us  cudaFreeAsync
                    0.00%  48.606us         2  24.303us  16.763us  31.843us  cudaEventRecord
                    0.00%  42.337us         8  5.2920us     389ns  20.538us  cudaSetDevice
                    0.00%  20.148us         4  5.0370us  2.7070us  10.705us  cuDeviceGetPCIBusId
                    0.00%  14.319us         2  7.1590us  1.1600us  13.159us  cudaEventCreate
                    0.00%  4.6450us         1  4.6450us  4.6450us  4.6450us  cudaEventElapsedTime
                    0.00%  3.4930us         8     436ns     280ns  1.4060us  cuDeviceGet
                    0.00%  2.4880us         3     829ns     368ns  1.6700us  cuDeviceGetCount
                    0.00%  1.6370us         4     409ns     328ns     611ns  cuDeviceGetUuid

```


as we can see, using two gpus to compute the same problem size turns out to be roughly twice as fast (compute time *single gpu* is 3.7..s while compute time *two gpus* is 1.8..s).


Regarding the implementation, I am unhappy with having to call `CudaSetDevice()` before each instruction for both streams, but as far as I know this is currently the safest way to make sure we are executing on the correct graphics card.