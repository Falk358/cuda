# Measurements for vector dot product

The task was to compute the dot product of two vectors, once by using a regular Kokkos Kernel (we call this *native*) and once by using the Kokkos kernel CuBlas implementation.



## Results

Here are the raw measurements by the Kokkos profiler:
```
- kokkos_native_dot
 (ParRed)   0.000060 1 0.000060 39.009288 11.781206
- KokkosBlas::dot<1D>
 (ParRed)   0.000048 1 0.000048 31.114551 9.396914
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.000023 1 0.000023 15.015480 4.534829
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.000011 1 0.000011 7.120743 2.150538
- Kokkos::View::initialization [x_mirror] via memset
 (ParFor)   0.000008 1 0.000008 5.108359 1.542777
- Kokkos::View::initialization [y_mirror] via memset
 (ParFor)   0.000004 1 0.000004 2.631579 0.794764
```
The first number after `(ParRed)` shows absolute compute time in seconds. As we can see, the times are similar.