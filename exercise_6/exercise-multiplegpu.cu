#include <iostream>
#include <cmath>
using namespace std;


__global__
void k_upwind(long nx, long ny, double* in, double* out) {
    long ix = threadIdx.x + blockDim.x*blockIdx.x;
    long iy = blockIdx.y;

    // Upwind scheme with CFL number 0.5
    if(ix < nx) {
        long i   = ix + nx*iy;
        // periodic boundary conditions in y
        long im1 = ix + nx*((iy==0) ? ny-1 : iy-1);

        out[i] = in[i] - 0.5*(in[i]-in[im1]);
    }
}

__global__
void k_init(long nx, long ny, double hx, double hy, double* in) {
    long ix = threadIdx.x + blockDim.x*blockIdx.x;
    long iy = blockIdx.y;

    double x = -1.0 + ix*hx;
    double y = -1.0 + iy*hy;

    if(ix < nx)
        in[ix + nx*iy] = exp(-50.0*x*x-50.0*y*y);
}

int main() {
    long nx = 4096;
    long ny = 4096;
    long N  = nx*ny;

    // Grid spacing (domain [-1,1]x[-1,1]).
    double hx = 2.0/double(nx);
    double hy = 2.0/double(ny);

    // Initialization.
    double *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(double)*N);
    cudaMalloc(&d_out, sizeof(double)*N);

    k_init<<<dim3(nx/256+1,ny,1), 256>>>(nx, ny, hx, hy, d_in);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    /*cudaStream_t stream_upper_matrix;
    cudaStream_t stream_lower_matrix;
    cudaStreamCreate(&stream_upper_matrix);
    cudaStreamCreate(&stream_lower_matrix);
    */
    // Do the actual computation.
    cudaEventRecord(start, 0);
    for(long k=0;k<2*ny;k++) {
        k_upwind<<<dim3(nx/256+1,ny,1), 256>>>(nx, ny, d_in, d_out);

        swap(d_in, d_out);
    }
    cudaEventRecord(stop, 0);

    cudaDeviceSynchronize();
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout << "Runtime: " << time*1e-3 << " s" << endl;

    // Check the result.
    double *h_in, *h_out;
    cudaMallocHost(&h_in, sizeof(double)*N);
    cudaMallocHost(&h_out, sizeof(double)*N);
    k_init<<<dim3(nx/256+1,ny,1), 256>>>(nx, ny, hx, hy, d_out);

    cudaMemcpy(h_in, d_in, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, d_out, sizeof(double)*N, cudaMemcpyDeviceToHost);

    double error = 0.0;
    for(long i=0;i<N;i++)
        error = max(error, fabs(h_in[i]-h_out[i]));
    cout << "Error: " << error << endl;
    

    /* 
    cudaStreamDestroy(stream_upper_matrix);
    cudaStreamDestroy(stream_lower_matrix);
    */
    return 0;
}

