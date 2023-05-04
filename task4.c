#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char* argv[]) {
    
    double time_spent1 = 0.0;

    clock_t begin1 = clock(); 

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

    // Main loop
    double err = accuracy + 1;
    int iter = 0;

    #pragma acc data copy(arr[0:Matrix*Matrix], array_new[0:Matrix*Matrix])
    {
        while (err > accuracy && iter < iterations) {
            // Compute new values
            err = 0;

            #pragma acc parallel loop reduction(max:err)
            for (int j = 1; j < Matrix - 1; j++) {
                for (int i = 1; i < Matrix - 1; i++) {
                    int index = j * Matrix + i;
                    array_new[index] = 0.25 * (arr[index + Matrix] + arr[index - Matrix] + arr[index - 1] + arr[index + 1]);
                    err = fmax(err, fabs(array_new[index] - arr[index]));
                }
            }
            
            cub::DeviceReduce::Max(NULL, err, arr, array_new, Matrix*Matrix);

            // Update values
            #pragma acc parallel loop
            for (int j = 1; j < Matrix - 1; j++) {
                for (int i = 1; i < Matrix - 1; i++) {
                    int index = j * Matrix + i;
                    arr[index] = array_new[index];
                }
            }
            iter++;
        }
    }
    printf("Final result: %d, %0.6lf\n", iter, err);
    // Free memory
    free(arr);
    free(array_new);

    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f\n", time_spent);

    return 0;
}
