#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

void heat_equation_2d(int n, double k, int max_iters, double tol, int* iters, double* err) {
    // Allocate memory for the grid with ghost cells
    double** u = (double**)malloc((n + 2) * sizeof(double*));
    for (int i = 0; i < n + 2; i++) {
        u[i] = (double*)malloc((n + 2) * sizeof(double));
    }
    // Initialize the grid with the boundary conditions
    u[0][0] = 10;
    u[0][n + 1] = 20;
    u[n + 1][0] = 30;
    u[n + 1][n + 1] = 20;
    for (int i = 1; i <= n; i++) {
        u[0][i] = 10 + i * (20 - 10) / (double)(n + 1);
        u[n + 1][i] = 20 + i * (30 - 20) / (double)(n + 1);
        u[i][0] = 10 + i * (30 - 10) / (double)(n + 1);
        u[i][n + 1] = 30 + i * (20 - 30) / (double)(n + 1);
    }
    // Calculate the grid spacing
    double h = 1 / (double)(n + 1);
    // Initialize the error and iteration counter
    double err_max = INFINITY;
    *iters = 0;
    // Perform the iterative updates
    while (err_max > tol && *iters < max_iters) {
        double** u_new = (double**)malloc((n + 2) * sizeof(double*));
        for (int i = 0; i < n + 2; i++) {
            u_new[i] = (double*)malloc((n + 2) * sizeof(double));
        }
#pragma acc parallel loop
        for (int i = 1; i <= n; i++) {
#pragma acc loop
            for (int j = 1; j <= n; j++) {
                u_new[i][j] = (1 / 4.0) * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) - (k * h * h / 4.0) * (u[i][j + 1] + u[i][j - 1] + u[i - 1][j] + u[i + 1][j] - 4 * u[i][j]);
            }
        }
        // Calculate the maximum error
        err_max = 0.0;
#pragma acc parallel loop reduction(max:err_max)
        for (int i = 1; i <= n; i++) {
#pragma acc loop reduction(max:err_max)
            for (int j = 1; j <= n; j++) {
                err_max = fmax(err_max, fabs(u_new[i][j] - u[i][j]));
            }
        }
        // Update the grid
#pragma acc parallel loop
        for (int i = 1; i <= n; i++) {
#pragma acc loop
            for (int j = 1; j <= n; j++) {
                u[i][j] = u_new[i][j];
            }
        }
        // Increment the iteration counter
        (*iters)++;
        // Free memory for u_new
        for (int i = 0; i < n + 2; i++) {
            free(u_new[i]);
        }
        free(u_new);
    }
    // Calculate the final error
    double err_final = 0.0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            err_final += fabs(u[i][j] - (25 + i * j * h * h));
        }
    }
    *err = err_final / (double)(n * n);
    // Free memory for u
    for (int i = 0; i < n + 2; i++) {
        free(u[i]);
    }
    free(u);
}

int main() {
    int n = 100;
    double k = 1;
    int max_iters = 10000;
    double tol = 1e-5;
    int iters;
    double err;
    heat_equation_2d(n, k, max_iters, tol, &iters, &err);
    printf("Number of iterations: %d\n", iters);
    printf("Final error: %g\n", err);
    return 0;
}
