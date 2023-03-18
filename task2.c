#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>


void heat_equation_2d(int n, double k, int max_iters, double tol, int* iters, double* err)
{
    // Allocate memory for the grid with ghost cells
    double* u = (double*)malloc((n + 2) * (n + 2) * sizeof(double));
    // Initialize the grid with the boundary conditions
    u[0*(n+2)+0] = 10;
    u[0*(n+2)+(n+1)] = 20;
    u[(n+1)*(n+2)+0] = 30;
    u[(n+1)*(n+2)+(n+1)] = 20;
    for (int i = 1; i <= n; i++) {
        u[0*(n+2)+i] = 10 + i * (20 - 10) / (double)(n + 1);
        u[(n+1)*(n+2)+i] = 20 + i * (30 - 20) / (double)(n + 1);
        u[i*(n+2)+0] = 10 + i * (30 - 10) / (double)(n + 1);
        u[i*(n+2)+(n+1)] = 30 + i * (20 - 30) / (double)(n + 1);
    }
    // Calculate the grid spacing
    double h = 1 / (double)(n + 1);
    // Initialize the error and iteration counter
    double err_max = INFINITY;
    *iters = 0;
    // Perform the iterative updates
    while (err_max > tol && *iters < max_iters) {
        double* u_new = (double*)malloc((n + 2) * (n + 2) * sizeof(double));
#pragma acc parallel loop
        for (int i = 1; i <= n; i++) {
#pragma acc loop
            for (int j = 1; j <= n; j++) {
                double* u_ij = &u[i*(n+2)+j];
                double* u_new_ij = &u_new[i*(n+2)+j];
                double* u_im1j = &u[(i-1)*(n+2)+j];
                double* u_ip1j = &u[(i+1)*(n+2)+j];
                double* u_ijm1 = &u[i*(n+2)+j-1];
                double* u_ijp1 = &u[i*(n+2)+j+1];
                *u_new_ij = (1 / 4.0) * (*u_im1j + *u_ip1j + *u_ijm1 + *u_ijp1)
                    - (k * h * h / 4.0) * (*u_ijp1 + *u_ijm1 + *u_im1j + *u_ip1j - 4 * (*u_ij));
            }
        }
       // Calculate the maximum error
        err_max = 0.0;
#pragma acc parallel loop reduction(max:err_max)
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                 double* u_ij = &u[i*(n+2)+j];
                double* u_new_ij = &u_new[i*(n+2)+j];
                err_max = fmax(err_max, fabs(*u_new_ij - *u_ij));
            }
        }
        // Copy the updated grid to the old grid and free the memory for the new grid
#pragma acc parallel loop collapse(2)
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                u[i*(n+2)+j] = u_new[i*(n+2)+j];
            }
        }
        free(u_new);

        // Update the iteration counter and error
        *iters += 1;
        *err = err_max;
    }
    // Free the memory for the grid
    free(u);
}

int main(int argc, char** argv) {

    double time_spent1 = 0.0;

    clock_t begin1 = clock(); 

    // Parse command line arguments
    if (argc != 4) {
        printf("Usage: %s <accuracy> <grid size> <number of iterations>\n", argv[0]);
        return 1;
    }
    double tol = atof(argv[1]);
    int n = atoi(argv[2]);
    int max_iters = atoi(argv[3]);
    // Perform the heat equation computation
    int iters;
    double err;
    heat_equation_2d(n, 0.1, max_iters, tol, &iters, &err);
    // Print the results

    clock_t end1 = clock();
    time_spent1 += (double)(end1 - begin1) / CLOCKS_PER_SEC;

    printf("Iterations: %d\n", iters);
    printf("Error: %.10f\n", err);
    printf("%f\n", time_spent1);

    return 0;
}
