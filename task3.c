#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char* argv[]) {
    
    double time_spent1 = 0.0;

    clock_t begin1 = clock(); 
    
    // Check if enough arguments are provided
    if (argc < 4) {
        printf("Usage: ./program_name Matrix accuracy iterations\n");
        return 1;
    }

    // Convert command line arguments to integers
    int Matrix = atoi(argv[1]);
    double accuracy = atof(argv[2]);
    int iterations = atoi(argv[3]);
    
    // Allocate 2D arrays on host memory
    double* arr = (double*)malloc(Matrix * Matrix * sizeof(double));
    double* array_new = (double*)malloc(Matrix * Matrix * sizeof(double));
    // Initialize arrays to zero
    for (int i = 0; i < Matrix * Matrix; i++) {
        arr[i] = 0;
        array_new[i] = 0;
    }
    // Set boundary conditions
    arr[0 * Matrix + 0] = 10;
    arr[0 * Matrix + Matrix - 1] = 20;
    arr[(Matrix - 1) * Matrix + 0] = 30;
    arr[(Matrix - 1) * Matrix + Matrix - 1] = 20;

    // Main loop
    double err = accuracy + 1;
    int iter = 0;

    // Allocate device memory
    double* d_arr;
    double* d_array_new;
    cudaMalloc((void**)&d_arr, Matrix * Matrix * sizeof(double));
    cudaMalloc((void**)&d_array_new, Matrix * Matrix * sizeof(double));
    cudaMemcpy(d_arr, arr, Matrix * Matrix * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    while (err > accuracy && err > 0.000001 && iter < iterations) {
        // Compute new values
        err = 0;

        double alpha = 0.25;
        double beta = 0.0;

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Matrix - 2, Matrix - 2, Matrix,
                    &alpha, d_arr + Matrix + 1, Matrix, d_arr + 1, Matrix, &beta, d_array_new + Matrix + 1, Matrix);
        cudaDeviceSynchronize();

        // Calculate error using cuBLAS reduction
        double* d_err;
        cudaMalloc((void**)&d_err, sizeof(double));
        
        cublasIdamax(handle, int(Matrix * Matrix - 1), d_array_new + 1, 1, d_err);
        cudaMemcpy(&err, d_err, sizeof(double), cudaMemcpyDeviceToHost);

        // Update values
        cudaMemcpy(d_arr, d_array_new, Matrix * Matrix * sizeof(double), cudaMemcpyDeviceToDevice);

        cudaFree(d_err);

        err = sqrt(err); // square root to get L2 norm error

        iter++;

        // Print progress
        if (iter % 100 == 0) {
            //printf("%d, %0.6lf\n", iter, err);
        }
    }

    cublasDestroy(handle);
    cudaFree(d_arr);
    cudaFree(d_array_new);

    printf("Final result: %d, %0.6lf\n", iter, err);
    // Free memory
    free(arr);
    free(array_new);
    
    return 0;
}
