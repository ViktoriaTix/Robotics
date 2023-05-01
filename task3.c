#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


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
    double* arr_err = (double*)malloc(Matrix * Matrix * sizeof(double));
    
    // Initialize arrays to zero
    for (int i = 0; i < Matrix * Matrix; i++) {
        arr[i] = 0;
        array_new[i] = 0;
        arr_err[i] = 0;
    }
    
    // Set boundary conditions
    arr[0] = 10;
    arr[Matrix - 1] = 20;
    arr[(Matrix - 1) * Matrix] = 30;
    arr[Matrix * Matrix - 1] = 20;
    
    for (int j = 1; j < Matrix - 1; j++) {
        //top
        arr[j] = (arr[Matrix + j] + arr[j - 1] + arr[j + 1] + arr[2 * Matrix + j]) / 4;
        //bottom
        arr[(Matrix - 1) * Matrix + j] = (arr[(Matrix - 2) * Matrix + j] + arr[(Matrix - 1) * Matrix + j - 1] + arr[(Matrix - 1) * Matrix + j + 1] + arr[(Matrix - 2) * Matrix + j]) / 4;
        //left
        arr[j * Matrix] = (arr[j * Matrix + 1] + arr[(j - 1) * Matrix] + arr[(j + 1) * Matrix] + arr[(j + 2) * Matrix]) / 4;
        //right
        arr[(j + 1) * Matrix - 1] = (arr[(j + 1) * Matrix - 2] + arr[j * Matrix + Matrix - 1] + arr[(j + 2) * Matrix - 1] + arr[(j - 1) * Matrix + Matrix - 1]) / 4;
    }


    // Create cuBLAS handle and initialize the handle to use the CUBLAS library
    cublasHandle_t handle;
    cublasStatus_t status;
    cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to create cublas handle\n");
        return 1;
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

    double alpha = 1.0;
    double beta = -1.0;

            // Perform Jacobi iteration
        while (diff > accuracy && count < iterations) 
        {
            // Copy array from device memory to host memory
            cudaMemcpy(array_new, arr_d, Matrix * Matrix * sizeof(double), cudaMemcpyDeviceToHost);
            diff = 0.0;
            // Perform Jacobi update
    #pragma acc loop independent
            for (int i = 1; i < Matrix - 1; i++) {
        #pragma acc loop independent
                for (int j = 1; j < Matrix - 1; j++) {
                    double sum = 0.0;
                    
                    // Compute sum of neighboring elements
                    sum = array_new[(i - 1) * Matrix + j] + array_new[(i + 1) * Matrix + j]
                        + array_new[i * Matrix + j - 1] + array_new[i * Matrix + j + 1];
                    
                    // Compute new element value
                    double new_elem = 0.25 * sum;
                    
                    // Update element value
                    array_new[i * Matrix + j] = new_elem;
                }
            }
            
            // Swap arrays
            temp = arr_d;
            arr_d = array_new;
            array_new = temp;

            //use the pointer on the video card
            #pragma acc host_data use_device(array_new, arr_d, arr_err)
            {
                // Perform linear combination using cuBLAS
                cublasStatus_t status = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, Matrix, Matrix, &alpha, arr_d, Matrix, &beta, array_new, Matrix, arr_err, Matrix);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Failed to perform linear combination using cublas\n");
                    return 1;
                }

                // Get index of element with maximum value
                cublasStatus_t status2 = cublasIdamax(handle, Matrix * Matrix, arr_err, 1, &diff);
                if (status2 != CUBLAS_STATUS_SUCCESS) {
                    printf("Failed to get index of element with maximum value\n");
                    return 1;
                }

                //get the value on the CPU of the cell with the maximum value of the array 
                cublasGetVector(1, sizeof(double), arr_err + diff - 1, 1, &diff, 1);
            }
            count++;
        }
    }

    // Print results
    printf("Number of iterations: %d\n", count);
    printf("Accuracy: %f\n", diff);

    // Calculate elapsed time
    clock_t end1 = clock();
    time_spent1 += (double)(end1 - begin1) / CLOCKS_PER_SEC;
    printf("Time elapsed is %f seconds\n", time_spent1);

    // Free allocated memory on host and device
    free(arr);
    free(array_new);
    free(arr_err);

    cublasFree(arr_d);
    cublasDestroy(handle);

    return 0;
}
