#include <string>
#include <iostream>
#include <cublas_v2.h>
using namespace std;


// init n*n matrix filled with value val
double* initMatrix(long n, double val)
{
    double* matrix = (double*) malloc(sizeof(double)*n*n);
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            matrix[i + j*n] = val;
        }
    }
    return matrix;
}

double* initVector(long n, double val)
{
    double* vector = (double*) malloc(sizeof(double) * n);

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
    double* vector = initVector(problem_size, 1.0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start,0);

    //TODO computation here

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout << time*1e-3 << " s" << endl;

    

    

    free(vector);


}