# Matrix Multiplication Kokkos Measurments

The task was to compute Matmul between a matrix and a vector in Kokkos and *Kokkos Kernel* Blas using the `gemv` method and measure the speed of both.


## Results

Raw Measurments:
```
Kernels: 

- KokkosBlas::gemv[SingleLevel]
 (ParFor)   0.001897 1 0.001897 44.034311 33.217834
- MatVecMult_Vanilla
 (ParFor)   0.001854 1 0.001854 43.032651 32.462219
- Kokkos::View::initialization [m] via memset
 (ParFor)   0.000551 1 0.000551 12.789153 9.647658
- Kokkos::View::initialization [g] via memset
 (ParFor)   0.000002 1 0.000002 0.049806 0.037572
- Kokkos::View::initialization [o] via memset
 (ParFor)   0.000002 1 0.000002 0.049806 0.037572
- Kokkos::View::initialization [i] via memset
 (ParFor)   0.000002 1 0.000002 0.044272 0.033397
```

As you can see, the vanilla (manually programmmed) Kernel takes *0.001854* seconds to complete, while the `gemv` kernel took *0.01897* seconds to finish. This indicates that both methods have similar speed and throughput.