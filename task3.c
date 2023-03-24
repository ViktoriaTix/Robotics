#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define M 1024
#define TOL 1e-6
#define ITER_MAX 1000000

int main() {
    // Allocate 2D arrays on host memory
    double *arr = (double *)malloc(M * M * sizeof(double));
    double *arrNew = (double *)malloc(M * M * sizeof(double));
    // Initialize arrays to zero
    for (int i = 0; i < M * M; i++) {
        arr[i] = 0;
        arrNew[i] = 0;
    }
    // Set boundary conditions
    arr[0*M+0] = 10;
    arr[0*M+M-1] = 20;
    arr[(M-1)*M+0] = 30;
    arr[(M-1)*M+M-1] = 20;

    // Main loop
    double err = TOL + 1;
    int iter = 0;

    #pragma acc data copy(arr[0:M*M], arrNew[0:M*M])
    {
        #pragma acc kernels loop independent
        for(int j = 1; j < M; j++){
            arr[0*M+j] = (arr[0*M+M-1] - arr[0*M+0])/(M-1) + arr[0*M+j-1];   //top
            arr[(M-1)*M+j] = (arr[(M-1)*M+M-1] - arr[(M-1)*M+0])/(M-1) + arr[(M-1)*M+j-1]; //bottom
            arr[j*M+0] = (arr[(M-1)*M+0] - arr[0*M+0])/(M-1) + arr[(j-1)*M+0]; //left
            arr[j*M+M-1] = (arr[(M-1)*M+M-1] - arr[0*M+M-1])/(M-1) + arr[(j-1)*M+M-1]; //right
        }

        cublasHandle_t handle;
        cublasCreate(&handle);

        while (err > TOL && iter < ITER_MAX) {
            // Compute new values
            err = 0;
            #pragma acc kernels loop independent reduction(max:err)
            for (int j = 1; j < M - 1; j++) {
                #pragma acc loop reduction(max:err)
                for (int i = 1; i < M - 1; i++) {
                    int index = j*M+i;
                    arrNew[index] = 0.25 * (arr[index + M] + arr[index - M] +
                                             arr[index - 1] + arr[index + 1]);
                    err = fmax(err, fabs(arrNew[index] - arr[index]));
                }
            }
            // Update values
            cublasDcopy(handle, M * M, arrNew, 1, arr, 1);

            iter++;
            
            // Print progress
            if (iter % 100 == 0) {
                printf("%d, %0.6lf\n", iter, err);
            }
        }

        cublasDestroy(handle);
    }
    
    printf("Final result: %d, %0.6lf\n", iter, err);
    // Free memory
    free(arr);
    free(arrNew);
    
    return 0;
}
