#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>

int main(int argc, char* argv[]) {

    double time_spent1 = 0.0;

    clock_t begin1 = clock();

    double alpha = 1.0; // or whatever value you want to assign to alpha

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

    for (int j = 1; j < Matrix; j++) {
        arr[0 * Matrix + j] = (arr[0 * Matrix + Matrix - 1] - arr[0 * Matrix + 0]) / (Matrix - 1) + arr[0 * Matrix + j - 1];   //top
        arr[(Matrix - 1) * Matrix + j] = (arr[(Matrix - 1) * Matrix + Matrix - 1] - arr[(Matrix - 1) * Matrix + 0]) / (Matrix - 1) + arr[(Matrix - 1) * Matrix + j - 1]; //bottom
        arr[j * Matrix + 0] = (arr[(Matrix - 1) * Matrix + 0] - arr[0 * Matrix + 0]) / (Matrix - 1) + arr[(j - 1) * Matrix + 0]; //left
        arr[j * Matrix + Matrix - 1] = (arr[(Matrix - 1) * Matrix + Matrix - 1] - arr[0 * Matrix + Matrix - 1]) / (Matrix - 1) + arr[(j - 1) * Matrix + Matrix - 1]; //right
    }
    // Main loop
    double err = accuracy + 1;
    int iter = 0;

    cublasHandle_t handle;
   // Allocate memory on the device
    double* d_arr, *d_array_new, *d_kernel;
    cublasStatus_t status;
    status = cublasCreate(&handle);
    status = cublasAlloc(Matrix * Matrix, sizeof(double), (void**)&d_arr);
    status = cublasAlloc(Matrix * Matrix, sizeof(double), (void**)&d_array_new);
    status = cublasAlloc(Matrix * Matrix, sizeof(double), (void**)&d_kernel);

    // Initialize kernel
    for (int j = 0; j < Matrix; j++) {
        for (int i = 0; i < Matrix; i++) {
            if (i == 0 || j == 0 || i == Matrix - 1 || j == Matrix - 1) {
                d_kernel[j * Matrix + i] = 0;
            }
            else {
                d_kernel[j * Matrix + i] = alpha;
            }
        }
    }

    // Copy data from host to device
    status = cublasSetMatrix(Matrix, Matrix, sizeof(double), arr, Matrix, d_arr, Matrix);
    status = cublasSetMatrix(Matrix, Matrix, sizeof(double), d_kernel, Matrix, d_kernel, Matrix);

    while (err > accuracy && iter < iterations) {
        // Compute new values
        err = 0;
        const double beta = 1.0;
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Matrix, Matrix, Matrix, &alpha, d_kernel, Matrix, d_arr, Matrix, &beta, d_array_new, Matrix);
        cublasGetMatrix(Matrix, Matrix, sizeof(double), d_array_new, Matrix, array_new, Matrix);
        for (int j = 1; j < Matrix - 1; j++) {
            for (int i = 1; i < Matrix - 1; i++) {
                int index = j * Matrix + i;
                err = fmax(err, fabs(array_new[index] - arr[index]));
            }
        }
        // Update values
        cublasSetMatrix(Matrix, Matrix, sizeof(double), array_new, Matrix, d_array_new, Matrix);
        cublasDcopy(handle, Matrix * Matrix, d_array_new, 1, d_arr, 1);
        iter++;
    }

    // Copy data from device to host
    cublasGetMatrix(Matrix, Matrix, sizeof(double), d_arr, Matrix, arr, Matrix);
    double max_error = 0.0;
    for (int j = 1; j < Matrix - 1; j++) {
        for (int i = 1; i < Matrix - 1; i++) {
            int index = j * Matrix + i;
            max_error = fmax(max_error, fabs(array_new[index] - arr[index]));
        }
    }

    printf("Final result: %d, %0.6lf\n", iter, max_error);

    // Free memory
    cublasFree(d_arr);
    cublasFree(d_array_new);
    cublasFree(d_kernel);
    cublasDestroy(handle);

    clock_t end1 = clock();
    time_spent1 += (double)(end1 - begin1) / CLOCKS_PER_SEC;
    printf("%f\n", time_spent1);

    return 0;
}
