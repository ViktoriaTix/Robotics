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
    double* arr_new = (double*)malloc(Matrix * Matrix * sizeof(double));
    double* arr_err = (double*)malloc(Matrix * Matrix * sizeof(double));
    
    // Initialize arrays to zero
    for (int i = 0; i < Matrix * Matrix; i++) {
        arr[i] = 0;
        arr_new[i] = 0;
        arr_err[i] = 0;
    }
    
    // Set boundary conditions
    arr[0] = 10;
    arr[Matrix - 1] = 20;
    arr[(Matrix - 1) * Matrix] = 30;
    arr[Matrix * Matrix - 1] = 20;

    arr_new[0] = 10;
    arr_new[Matrix - 1] = 20;
    arr_new[(Matrix - 1) * Matrix] = 30;
    arr_new[Matrix * Matrix - 1] = 20;
 
    for(int i = 1; i < Matrix - 1; i++)
    {
        arr[i*Matrix] = arr[(i-1)*Matrix]+ (10 / (Matrix - 1));
        arr[i] = arr[i-1] + (10 / (Matrix - 1));
        arr[(Matrix-1)*Matrix + i] = arr[(Matrix-1)*Matrix + i-1] + (10 / (Matrix - 1));
        arr[i*Matrix + (Matrix-1)] = arr[(i-1)*Matrix + (Matrix-1)] + (10 / (Matrix - 1));

        arr_new[i*Matrix] = arr_new[(i-1)*Matrix]+ (10 / (Matrix - 1));
        arr_new[i] = arr_new[i-1] + (10 / (Matrix - 1));
        arr_new[(Matrix-1)*Matrix + i] = arr_new[(Matrix-1)*Matrix + i-1] + (10 / (Matrix - 1));
        arr_new[i*Matrix + (Matrix-1)] = arr_new[(i-1)*Matrix + (Matrix-1)] + (10 / (Matrix - 1));
    }


    // Create cuBLAS handle and initialize the handle to use the CUBLAS library
    cublasHandle_t handle;

    cublasStatus_t status;

    cublasCreate(&handle);

    // Initialize variables
    double diff = 1.0;
    int count = 0;
    double* temp;
    int ind = 0;

    double alpha = 1.0;
    double beta = -1.0;

    #pragma acc data copyin(arr_new[:Matrix * Matrix],arr[:Matrix * Matrix]) create(sum,arr_err[:Matrix * Matrix])
    {
        // Perform Jacobi iteration
        while (diff > accuracy && count < iterations) 
        {
            #pragma acc data present(arr,arr_new)
            // Perform Jacobi update
            #pragma acc loop independent
            for (int i = 1; i < Matrix - 1; i++) {
                #pragma acc loop independent
                for (int j = 1; j < Matrix - 1; j++) {

                    arr_new[i * Matrix + j] = (arr_new[(i - 1) * Matrix + j] + arr_new[(i + 1) * Matrix + j]
                        + arr_new[i * Matrix + j - 1] + arr_new[i * Matrix + j + 1])/4;
                }
            }
            
            // Swap arrays
            temp = arr;
            arr = arr_new;
            arr_new = temp;

            //use value on gpu
            #pragma acc host_data use_device(arr_new, arr, arr_err)
            {
                // Perform linear combination using cuBLAS
                cublasStatus_t status = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, Matrix, Matrix, &alpha, arr, Matrix, &beta, arr_new, Matrix, arr_err, Matrix);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Failed to perform linear combination using cublas\n");
                    return 1;
                }

                // Get index of element with maximum value
                cublasStatus_t status2 = cublasIdamax(handle, Matrix * Matrix, arr_err, 1, &ind);
                if (status2 != CUBLAS_STATUS_SUCCESS) {
                    printf("Failed to get index of element with maximum value\n");
                    return 1;
                }

                //get the value on the CPU of the cell with the maximum value of the array 
                cublasGetVector(1, sizeof(double), arr_err + ind - 1, 1, &diff, 1);
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
    free(arr_new);
    free(arr_err);

    cublasDestroy(handle);

    return 0;
}

