#include <cublas_v2.h>
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

    // Allocate 2D arrays on host memory
    double* arr = (double*)malloc(Matrix * Matrix * sizeof(double));
    double* array_new = (double*)malloc(Matrix * Matrix * sizeof(double));
    
    // Initialize arrays to zero
    #pragma acc parallel loop present(arr, array_new)
    for (int i = 0; i < Matrix * Matrix; i++) {
        arr[i] = 0;
        array_new[i] = 0;
    }

    // Set boundary conditions
    arr[0 * Matrix + 0] = 10;
    arr[0 * Matrix + Matrix - 1] = 20;
    arr[(Matrix - 1) * Matrix + 0] = 30;
    arr[(Matrix - 1) * Matrix + Matrix - 1] = 20;

    for (int j = 1; j < Matrix; j++) {
        arr[0 * Matrix + j] = (arr[0 * Matrix + Matrix - 1] - arr[0 * Matrix + 0]) / (Matrix - 1) + arr[0 * Matrix + j - 1];   //top
        arr[(Matrix - 1) * Matrix + j] = (arr[(Matrix - 1) * Matrix + Matrix - 1] - arr[(Matrix - 1) * Matrix + 0]) / (Matrix - 1) + arr[(Matrix - 1) * Matrix + j - 1]; //bottom
        arr[j * Matrix + 0] = (arr[(Matrix - 1) * Matrix + 0] - arr[0 * Matrix + 0]) / (Matrix - 1) + arr[(j - 1) * Matrix + 0]; //left
        arr[j * Matrix + Matrix - 1] = (arr[(Matrix - 1) * Matrix + Matrix - 1] - arr[0 * Matrix + Matrix - 1]) / (Matrix - 1) + arr[(j - 1) * Matrix + Matrix - 1]; //right
    }

    // Main loop
    double err = accuracy + 1;
    int iter = 0;

    // Create cuBLAS handle initialize the handle to use the CUBLAS library
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory for error
    double* err_dev;
    cudaMalloc((void**)&err_dev, sizeof(double));

    while (err > accuracy  && iter < iterations) {
        // Compute new values and error
        #pragma acc data copy(arr[0:Matrix*Matrix]), create(array_new[0:Matrix*Matrix])
        {
            #pragma acc kernels
            for (int j = 1; j < Matrix - 1; j++) {
                for (int i = 1; i < Matrix - 1; i++) {
                    array_new[j * Matrix + i] = 0.25 * (arr[j * Matrix + (i - 1)] + arr[j * Matrix + (i + 1)] + arr[(j - 1) * Matrix + i] + arr[(j + 1) * Matrix + i]);
                }
            }
        }

        #pragma acc data copy(arr[0:Matrix*Matrix]), copy(array_new[0:Matrix*Matrix])
        {
            // Compute error
            err = 0;
            #pragma acc kernels loop reduction(max:err)
            for (int j = 1; j < Matrix - 1; j++) {
                for (int i = 1; i < Matrix - 1; i++) {
                    double diff = fabs(array_new[j * Matrix + i] - arr[j * Matrix + i]);
                    if (diff > err) {
                        err = diff;
                    }
                }
            }
        }

        // Update values
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasScopy(handle, Matrix * Matrix, array_new, 1, arr, 1);
        iter++;

        // Copy error to device and compute max error
        cudaMemcpy(err_dev, &err, sizeof(double), cudaMemcpyHostToDevice);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasDnrm2(handle, 1, err_dev, 1, &err_dev);
        cudaMemcpy(&err, err_dev, sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(err_dev);
    cublasDestroy(handle);

    printf("Final result: %d, %0.6lf\n", iter, err);
    // Free memory
    free(arr);
    free(array_new);

    clock_t end1 = clock();
    time_spent1 += (double)(end1 - begin1) / CLOCKS_PER_SEC;
    printf("%f\n", time_spent1);

    return 0;
}
