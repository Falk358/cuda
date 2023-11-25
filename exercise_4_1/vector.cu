#include <stdio.h>

#define ARRAYDIM 256

__global__ void KrnlDmmy(int *x, size_t max_size) {
    int i;
    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= max_size)
    {
        return;
    }

    x[i] = i;
    return;
}

int main() {
    int i, cudaRtrn;
    dim3 thrds_per_block, blcks_per_grid;
    int *a;

    if (cudaRtrn = cudaMallocManaged(&a, ARRAYDIM * sizeof(int)) != 0) {
       printf("*** allocation failed for array a[], %d ***\n", cudaRtrn);
    }

    for(i = 0; i  < ARRAYDIM; ++i) {
    	a[i] = 1;
    }

    thrds_per_block.x = 265;
    blcks_per_grid.x = 1;

    KrnlDmmy<<<blcks_per_grid, thrds_per_block>>>(a, ARRAYDIM);

    if(cudaRtrn = cudaDeviceSynchronize() != 0) {
        printf("*** error synchronizing device, %d ***\n", cudaRtrn);
    }

    for (i=0; i<ARRAYDIM; i++) {
        //printf("%d: %d, \n", i, a[i]);
    }

    cudaFree(a);

    return(0);
}
