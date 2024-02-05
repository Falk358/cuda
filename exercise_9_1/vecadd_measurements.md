# Timing CPU and GPU for Kokkos

The task was to execute three dimensional array elementwise addition in Kokkos and measure the performance on CPU, GPU, LeftLayout and RightLayout of data.

## CPU Measurements

Here are the raw measurements for cpu (time in s):
```
vecadd_left
- (ParFor)   0.199134 1 0.199134 37.036828 3.653701
- Kokkos::View::initialization [res] via memset
 (ParFor)   0.095901 2 0.047951 17.836569 1.759586
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.093694 2 0.046847 17.426083 1.719092
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.090956 2 0.045478 16.916889 1.668859
- vecadd_right
 (ParFor)   0.057980 1 0.057980 10.783631 1.063811
```

We operated on an array of size 256³ * 8 bytes (size of *double*).
By converting the measurements above to GB/s we get:
+ CPU LeftLayout: 5.022 GB/s
+ CPU RightLayout: 17.247 GB/s


## GPU Measurements:

Here are the raw measurements for gpu (time in s):
```
- Kokkos::View::initialization [x_mirror] via memset
 (ParFor)   0.088578 2 0.044289 33.732438 1.622567
- Kokkos::View::initialization [y_mirror] via memset
 (ParFor)   0.084308 2 0.042154 32.106299 1.544348
- Kokkos::View::initialization [res_mirror] via memset
 (ParFor)   0.081242 2 0.040621 30.938766 1.488189
- vecadd_right
 (ParFor)   0.006846 1 0.006846 2.607088 0.125404
- vecadd_left
 (ParFor)   0.000610 1 0.000610 0.232254 0.011172
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.000346 2 0.000173 0.131834 0.006341
- Kokkos::View::initialization [res] via memset
 (ParFor)   0.000330 2 0.000165 0.125660 0.006044
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.000330 2 0.000165 0.125660 0.006044
```

We operated on an array of size 256³ * 8 bytes (size of *double*).
By converting the measurements above to GB/s we get:
+ GPU LeftLayout: 1639.344 GB/s
+ GPU RightLayout: 146.071 GB/s

## Conclusion

It is clear that **memory layout** greatly influences the performance of both processor types, where cpus perform better with RightLayout and gpus perform better with LeftLayout.