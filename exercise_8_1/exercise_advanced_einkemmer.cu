#include <iostream>
#include <cmath>
using namespace std;


__global__
void k_sum(long n, double* vec, double* result) {
    // TODO
}


int main() {
    long n = 8*1024*1024; // must be a power of 2
    double h_result = 0.0;
    double *h_vec, *d_vec, *d_tmp;

    cudaMallocHost(&h_vec, sizeof(double)*n);
    cudaMalloc(&d_vec, sizeof(double)*n);
    cudaMalloc(&d_tmp, sizeof(double)*(n/256));

    // Initialie vec and copy to GPU.
    for(long i=0;i<n;i++)
        h_vec[i] = 1.0/pow(double(i+1),2);
    cudaMemcpy(d_vec, h_vec, sizeof(double)*n, cudaMemcpyHostToDevice);

    // TODO

    // Check the result.
    cout << "Result: " << h_result << endl;
    if(fabs(h_result - pow(M_PI,2)/6.0) < 1e-5) {
        cout << "Correct!" << endl;
    } else {
        cout << "The computed result does not match with the expected result ("
             << pow(M_PI,2)/6.0 << ")" << endl;
    }

    return 0;
}
