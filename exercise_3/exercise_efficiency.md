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

We can see that most time is lost during cudaMemCopy calls.

## Low hanging fruit

Quickly looking at the unoptimized code, we have a lot of memcpy calls in lines 39-43. We can optimize this by just copying a single row of the matrix:
```
    // initialize matrix and copy to GPU
    double matrix_row[3] = {1.0 - 0.25*2.0, 0.25, 0.25};
    cudaMalloc(&d_mat, sizeof(double)*3);
    cudaMemcpy(d_mat, matrix_row, sizeof(double)*3, cudaMemcpyHostToDevice);
```
In order for this to work, we need to slightly adjust our kernel as well, so that we access one single row over and over again:
```
__global__
void k_matvecmul(long n, double* in, double* out, double* mat) {
    long i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i>0 && i < n-1)
        out[i] = mat[0]*in[i] + mat[1]*in[i+1] + mat[2]*in[i-1];
}
```
Notice that `mat[3*i]` became `mat[0]`, `mat[3*i+1]` became `mat[1]` etc.

This one change improved our efficiency significantly:
```
==205443== NVPROF is profiling process 205443, command: ../build/exercise_efficiency
Numerical error: 6.88338e-15
0.0353552 s
==205443== Profiling application: ../build/exercise_efficiency
==205443== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   41.49%  2.9637ms       100  29.636us  29.472us  32.319us  k_matvecmul(long, double*, double*, double*)
                   40.90%  2.9212ms       100  29.211us  28.703us  30.144us  [CUDA memcpy DtoD]
                    9.10%  650.24us         2  325.12us  1.7600us  648.48us  [CUDA memcpy HtoD]
                    8.51%  607.68us         1  607.68us  607.68us  607.68us  [CUDA memcpy DtoH]
      API calls:   94.08%  365.53ms         2  182.77ms  2.3200us  365.53ms  cudaEventCreate
                    1.75%  6.8036ms       103  66.054us  10.010us  4.9803ms  cudaMemcpy
                    1.51%  5.8605ms         4  1.4651ms  950.80us  1.6577ms  cuDeviceTotalMem
                    1.45%  5.6377ms         1  5.6377ms  5.6377ms  5.6377ms  cudaHostAlloc
                    0.73%  2.8452ms       404  7.0420us     190ns  366.88us  cuDeviceGetAttribute
                    0.21%  801.43us         3  267.14us  215.80us  293.55us  cudaMalloc
                    0.15%  594.27us       100  5.9420us  5.0680us  52.368us  cudaLaunchKernel
                    0.07%  285.47us         4  71.367us  47.932us  91.930us  cuDeviceGetName
                    0.02%  76.668us         1  76.668us  76.668us  76.668us  cudaEventSynchronize
                    0.01%  25.988us         1  25.988us  25.988us  25.988us  cudaEventElapsedTime
                    0.01%  21.821us         2  10.910us  5.5310us  16.290us  cudaEventRecord
                    0.00%  14.519us         4  3.6290us  1.5370us  8.8660us  cuDeviceGetPCIBusId
                    0.00%  11.472us         2  5.7360us  1.1820us  10.290us  cudaEventDestroy
                    0.00%  3.6630us         8     457ns     207ns  1.4250us  cuDeviceGet
                    0.00%  2.7060us         3     902ns     342ns  1.8640us  cuDeviceGetCount
                    0.00%  1.6040us         4     401ns     231ns     651ns  cuDeviceGetUuid
```

## Adequate number of Threads

The program is still using not enough threads, working from the *Low hanging fruit* above lets see if we can get better throughput by increasing the number of threads and blocks to 128: `k_matvecmul<<<n/64+1,64>>>(n, d_in, d_out, d_mat);`
becomes `k_matvecmul<<<n/128+1,128>>>(n, d_in, d_out, d_mat);`

```
==205665== NVPROF is profiling process 205665, command: ../build/exercise_efficiency
Numerical error: 6.88338e-15
0.0267005 s
==205665== Profiling application: ../build/exercise_efficiency
==205665== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.82%  2.9240ms       100  29.239us  28.383us  36.671us  [CUDA memcpy DtoD]
                   35.92%  2.3438ms       100  23.437us  22.720us  28.416us  k_matvecmul(long, double*, double*, double*)
                    9.96%  649.79us         2  324.89us  1.7920us  648.00us  [CUDA memcpy HtoD]
                    9.30%  606.72us         1  606.72us  606.72us  606.72us  [CUDA memcpy DtoH]
      API calls:   93.72%  322.73ms         2  161.36ms     873ns  322.73ms  cudaEventCreate
                    1.89%  6.4954ms         4  1.6239ms  1.5988ms  1.6620ms  cuDeviceTotalMem
                    1.83%  6.2875ms       103  61.043us  8.3170us  4.6639ms  cudaMemcpy
                    1.24%  4.2551ms         1  4.2551ms  4.2551ms  4.2551ms  cudaHostAlloc
                    0.88%  3.0276ms       404  7.4940us     185ns  360.92us  cuDeviceGetAttribute
                    0.18%  617.29us         3  205.76us  168.91us  231.51us  cudaMalloc
                    0.14%  483.61us       100  4.8360us  4.0070us  46.511us  cudaLaunchKernel
                    0.09%  296.83us         4  74.208us  70.497us  83.919us  cuDeviceGetName
                    0.03%  102.17us         1  102.17us  102.17us  102.17us  cudaEventSynchronize
                    0.01%  20.099us         2  10.049us  6.0570us  14.042us  cudaEventRecord
                    0.01%  17.339us         4  4.3340us  2.0760us  9.0990us  cuDeviceGetPCIBusId
                    0.00%  11.849us         1  11.849us  11.849us  11.849us  cudaEventElapsedTime
                    0.00%  8.8010us         2  4.4000us     655ns  8.1460us  cudaEventDestroy
                    0.00%  3.5480us         8     443ns     307ns  1.0900us  cuDeviceGet
                    0.00%  2.8240us         3     941ns     367ns  1.9490us  cuDeviceGetCount
                    0.00%  1.7930us         4     448ns     229ns     632ns  cuDeviceGetUuid
```


we can see that our kernel takes up 5% less time compared to the *low hanging fruit* version.