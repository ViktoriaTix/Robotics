#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
void cublasFree(void* ptr);

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
    
    // Allocate 1D arrays on host memory
    double* arr = (double*)malloc(Matrix * Matrix * sizeof(double));
    double* array_new = (double*)malloc(Matrix * Matrix * sizeof(double));
    
    // Initialize arrays to zero
    for (int i = 0; i < Matrix * Matrix; i++) {
        arr[i] = 0;
        array_new[i] = 0;
    }
    
    // Set boundary conditions
    arr[0] = 10;
    arr[Matrix - 1] = 20;
    arr[(Matrix - 1) * Matrix] = 30;
    arr[Matrix * Matrix - 1] = 20;
    
    for (int j = 1; j < Matrix; j++) {
        arr[j] = (arr[Matrix - 1] - arr[0]) / (Matrix - 1) + arr[j - 1];   //top
        arr[(Matrix - 1) * Matrix + j]    = (arr[Matrix * Matrix - 1] - arr[(Matrix - 1) * Matrix]) / (Matrix - 1) + arr[(Matrix - 1) * Matrix + j - 1];  //bottom
        arr[j * Matrix] = (arr[Matrix * (Matrix - 1)] - arr[0]) / (Matrix - 1) + arr[(j - 1) * Matrix];    //left
        arr[(j + 1) * Matrix - 1] = (arr[Matrix * Matrix - 1] - arr[Matrix - 1]) / (Matrix - 1) + arr[j * Matrix + Matrix - 1];  //right
    }

    // Allocate 1D array on device memory
    double* arr_d;
    cudaMalloc((void**)&arr_d, Matrix * Matrix * sizeof(double));

    // Copy array from host memory to device memory
    cudaMemcpy(arr_d, arr, Matrix * Matrix * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize variables
    double diff = 1.0;
    int count = 0;
    double* temp;

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform Jacobi iteration
    while (diff > accuracy && count < iterations) {
        // Copy array from device memory to host memory
        cudaMemcpy(array_new, arr_d, Matrix * Matrix * sizeof(double), cudaMemcpyDeviceToHost);
        
        diff = 0.0;
        
        // Perform Jacobi update
        for (int i = 1; i < Matrix - 1; i++) {
            for (int j = 1; j < Matrix - 1; j++) {
                double sum = 0.0;
                
                // Compute sum of neighboring elements
                sum = array_new[(i - 1) * Matrix + j] + array_new[(i + 1) * Matrix + j]
                    + array_new[i * Matrix + j - 1] + array_new[i * Matrix + j + 1];
                
                // Compute new element value
                double new_elem = 0.25 * sum;
                
                // Compute difference between new and old element value
                double temp_diff = fabs(new_elem - array_new[i * Matrix + j]);
                
                // Update diff
                if (temp_diff > diff) {
                    diff = temp_diff;
                }
                
                // Update element value
                array_new[i * Matrix + j] = new_elem;
            }
        }
        
        // Swap arrays
        temp = arr_d;
        arr_d = array_new;
        array_new = temp;
        
        // Perform matrix-vector multiplication using cuBLAS
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemv(handle, CUBLAS_OP_T, Matrix, Matrix, &alpha, arr_d, Matrix, array_new, 1, &beta, arr_d, 1);
        
        count++;
    }

    // Print results
    printf("Number of iterations: %d\n", count);
    printf("Accuracy: %f\n", diff);

    // Free memory
    free(arr);
    free(array_new);
    cudaFree(arr_d);
    cublasDestroy(handle);

    clock_t end1 = clock();
    time_spent1 += (double)(end1 - begin1) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", time_spent1);

    return 0;

}
