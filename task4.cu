#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char* argv[]) {
    
    double time_spent = 0.0;

    clock_t begin = clock(); 

    // Convert command line arguments to integers
    int Matrix = atoi(argv[1]);
    double accuracy = atof(argv[2]);
    int iterations = atoi(argv[3]);
    
    // Allocate 1D arrays on host memory
    double* arr = (double*)malloc(Matrix * Matrix * sizeof(double));
    double* arr_new = (double*)malloc(Matrix * Matrix * sizeof(double));

    // Initialize arrays to zero
    for (int i = 0; i < Matrix * Matrix; i++) {
        arr[i] = 0;
        arr_new[i] = 0;
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


    // выделяем память на gpu через cuda для 3 сеток
    double *CudaArr, *CudaNewArr, *CudaDifArr;
    cudaMalloc((void **)&CudaArr, sizeof(double) * Matrix * Matrix);
    cudaMalloc((void **)&CudaNewArr, sizeof(double) * Matrix * Matrix);
    cudaMalloc((void **)&CudaDifArr, sizeof(double) * Matrix * Matrix);

    // копирование информации с CPU на GPU
    cudaMemcpy(CudaArr, arr, sizeof(double) * Matrix * Matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(CudaNewArr, arr_new, sizeof(double) * Matrix * Matrix, cudaMemcpyHostToDevice);

    // выделяем память на gpu. Хранение ошибки на device
    double *max_err = 0;
    cudaMalloc((void **)&max_err, sizeof(double));

    size_t tempStorageBytes = 0;
    double *tempStorage = NULL;

    // получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaDifArr, max_err, Matrix * Matrix);

    // выделяем память для буфера
    cudaMalloc((void **)&tempStorage, tempStorageBytes);

    // Main loop
    double err = accuracy + 1;
    int iter = 0;

    while (err > accuracy && iter < iterations) {
        
        err = 0;
        // Compute new values
        for (int j = 1; j < Matrix - 1; j++) {
            for (int i = 1; i < Matrix - 1; i++) {
                int index = j * Matrix + i;
                CudaNewArr[index] = 0.25 * (CudaArr[index + Matrix] + CudaArr[index - Matrix] +
                    CudaArr[index - 1] + CudaArr[index + 1]);
                CudaDifArr[index] = fabs(CudaNewArr[index] - CudaArr[index]);
            }
        }
        
        // Compute maximum error using CUB
        cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaDifArr, max_err, Matrix * Matrix);
        
        // запись ошибки в переменную
        cudaMemcpy(&err, max_err, sizeof(double), cudaMemcpyDeviceToHost);
        err = std::abs(err);

        // Update values
        for (int j = 1; j < Matrix - 1; j++) {
            for (int i = 1; i < Matrix - 1; i++) {
                int index = j * Matrix + i;
                arr[index] = arr_new[index];
            }
        }
        
        iter++;
    }

    printf("Final result: %d, %0.6lf\n", iter, err);
    
    // Free memory
    free(arr);
    free(arr_new);

    cudaFree(CudaArr);
    cudaFree(CudaNewArr);
    cudaFree(CudaDifArr);

    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f\n", time_spent);

    return 0;
}
