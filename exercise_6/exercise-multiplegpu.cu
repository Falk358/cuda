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

    // half problem size for distributing on 2 gpus
    long nx_distributed = nx /2;
    long ny_distributed = ny;
    long N_distributed = nx_distributed * ny_distributed;

    // Grid spacing (domain [-1,1]x[-1,1]).
    double hx = 2.0/double(nx);
    double hy = 2.0/double(ny);

    //problem will be split in left and right side of total N

    // Initialization.
    double *d_in_left, *d_out_left;
    double *d_in_right, *d_out_right;
    //allocation on left gpu, ranging from x index 0 to nx/2-1
    cudaSetDevice(0);
    cudaStream_t stream_left_matrix;
    cudaStreamCreate(&stream_left_matrix);
    cudaMallocAsync(&d_in_left, sizeof(double)*N_distributed, stream_left_matrix);
    cudaMallocAsync(&d_out_left, sizeof(double)*N_distributed, stream_left_matrix);
    k_init<<<dim3(nx_distributed/256+1,ny,1), 256, 0, stream_left_matrix>>>(nx_distributed, ny_distributed, hx, hy, d_in_left);

    //allocation on right gpu, ranging from x index nx/2 to nx-1
    cudaSetDevice(1);
    cudaStream_t stream_right_matrix;
    cudaStreamCreate(&stream_right_matrix);
    cudaMallocAsync(&d_in_right, sizeof(double)*N_distributed, stream_right_matrix);
    cudaMallocAsync(&d_out_right, sizeof(double)*N_distributed, stream_right_matrix);
    k_init<<<dim3(nx_distributed/256+1,ny,1), 256, 0, stream_right_matrix>>>(nx_distributed, ny_distributed, hx, hy, d_in_right);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    // Do the actual computation.
    cudaEventRecord(start, 0);

    for(long k=0;k<2*ny;k++) {
        k_upwind<<<dim3(nx_distributed/256+1,ny,1), 256, 0, stream_left_matrix>>>(nx_distributed, ny_distributed, d_in_left, d_out_left);

        swap(d_in_left, d_out_left);
    }

    for(long k=0;k<2*ny;k++) {
        k_upwind<<<dim3(nx_distributed/256+1,ny,1), 256, 0, stream_right_matrix>>>(nx_distributed, ny_distributed, d_in_right, d_out_right);

        swap(d_in_right, d_out_right);
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
    cudaSetDevice(0);
    k_init<<<dim3(nx_distributed/256+1,ny,1), 256, 0, stream_left_matrix>>>(nx_distributed, ny_distributed, hx, hy, d_out_left);
    cudaSetDevice(1);
    k_init<<<dim3(nx_distributed/256+1,ny,1), 256, 0, stream_right_matrix>>>(nx_distributed, ny_distributed, hx, hy, d_out_right);

    // copy back left side of problem to host
    cudaSetDevice(0);
    cudaMemcpyAsync(h_in, d_in_left, sizeof(double)*N_distributed, cudaMemcpyDeviceToHost, stream_left_matrix);
    cudaMemcpyAsync(h_out, d_out_left, sizeof(double)*N_distributed, cudaMemcpyDeviceToHost, stream_left_matrix);

    //copy right side of problem back to host, with offset (nx_distributed) to merge the results of left and right
    cudaSetDevice(1);
    cudaMemcpyAsync(h_in + nx_distributed, d_in_right, sizeof(double) * N_distributed, cudaMemcpyDeviceToHost, stream_right_matrix);
    cudaMemcpyAsync(h_out + nx_distributed, d_out_right, sizeof(double) * N_distributed, cudaMemcpyDeviceToHost, stream_right_matrix);

    double error = 0.0;
    for(long i=0;i<N;i++)
        error = max(error, fabs(h_in[i]-h_out[i]));
    cout << "Error: " << error << endl;
    

    cudaSetDevice(0);
    cudaFreeAsync(d_in_left, stream_left_matrix);
    cudaFreeAsync(d_out_left, stream_left_matrix);
    cudaSetDevice(1);
    cudaFreeAsync(d_in_right, stream_right_matrix);
    cudaFreeAsync(d_out_right, stream_right_matrix);
    cudaStreamDestroy(stream_left_matrix);
    cudaStreamDestroy(stream_right_matrix);
    return 0;
}

