Illegal access to address (@global)0x7fffcfc00a88 detected.

Thread 1 "stencil" received signal CUDA_EXCEPTION_1, Lane Illegal Address.
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (416,0,0), device 0, sm 0, warp 13, lane 0]
0x0000000000f54430 in applyStencil1D (sIdx=2, eIdx=1022, weights=0x7fffcfc00000, in=0x7fffcfc00200, out=0x7fffcfc00400) at stencil.cu:7
7               out[i] = 0;


Here, the issue is that in this snippet:
```
 float *weights = (float *)malloc(wsize);
    float *in = (float *)malloc(size);
    float *out= (float *)malloc(size);

    initializeWeights(weights, RADIUS);
    initializeArray(in, N);

    float *d_weights; cudaMalloc(&d_weights, wsize);
    float *d_in; cudaMalloc(&d_in, wsize);
    float *d_out; cudaMalloc(&d_out, wsize);
```

we allocate `*d_in` and `*d_out` with the wrong size `wsize`. This should be `size` instead, since wsize is the size of our stencil, while size represents the size of the entire array which the stencil is applied to. This causes our kernel to access memory which is out of bounds.