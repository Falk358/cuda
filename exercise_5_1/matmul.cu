#include <string>
#include <iostream>
#include <cublas_v2.h>
using namespace std;


// init n*n matrix filled with value val
double* initMatrix(long n, double val, size_t memsize)
{
    double* matrix = (double*) malloc(memsize);
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            matrix[i + j*n] = val;
        }
    }
    return matrix;
}

double* initVector(long n, double val, size_t memsize)
{
    double* vector = (double*) malloc(memsize);

    for (int i = 0; i < n; i++)
    {
        vector[i] = val;
    }
    
    return vector;
}



int verifyResult(double* result, long n_result, double tolerance)
{
    return;
}



int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cerr << "Error, not enough cmd line arguments given; usage matmul \"problem_size\", where problem size is an int";
        exit(1);
    }
    int problem_size = stoi(argv[1]);
    size_t vec_size = sizeof(double) *problem_size;
    size_t mat_size = sizeof(double) * problem_size* problem_size;
    double* vector = initVector(problem_size, 1.0, vec_size);
    double* matrix = initMatrix(problem_size, 2.0, mat_size);
    double* result = (double*) malloc(mat_size);
    
    double alpha = 1.0;
    double beta = 0.0;
    double* device_vector;
    double* device_matrix;
    double* device_result;
    cudaMalloc(&device_vector, vec_size);
    cudaMalloc(&device_matrix, mat_size);
    cudaMalloc(&device_result, mat_size);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        cout << "ERROR: cublasCreate failed. Error code: " << status << endl;
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start,0);

    //TODO computation here
    cublasStatus_t status_computation = cublasDgemv(handle,CUBLAS_OP_N, problem_size, problem_size, &alpha, device_matrix,1,device_vector, 1, &beta, device_result,1);
    if (status_computation != CUBLAS_STATUS_SUCCESS)
    {
        cout << "ERROR: cublasDgemv failed. Error code" << status_computation << endl;
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout << time*1e-3 << " s" << endl;


    // free memory
    cublasDestroy(handle); 
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_vector);
    cudaFree(device_matrix);
    cudaFree(device_result);
    free(matrix);    
    free(vector);
    free(result);


}