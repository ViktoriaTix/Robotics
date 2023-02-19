#include <stdio.h>
#include <math.h> 
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC

#define _USE_MATH_DEFINES
#define N 10000000
#define M_PI 3.14159265358979323846 // pi
	
float Sum=0;
float arr[N];

int main()
{
	double time_all = 0.0;
	clock_t begin_all = clock();
	
	double time_spent1 = 0.0;

	clock_t begin1 = clock();

#pragma acc kernels
	for (int i = 0; i < N; i++)
	{
		arr[i] = sin(i * M_PI *2 / N);
	}

	clock_t end1 = clock();
	time_spent1 += (double)(end1 - begin1) / CLOCKS_PER_SEC;

	double time_spent2 = 0.0;
	
	clock_t begin2 = clock();
	
#pragma acc kernels	
	for (int i = 0; i < N; i++)
	{
		Sum += arr[i];
	}
	
	clock_t end2 = clock();
	time_spent2 += (double)(end2 - begin2) / CLOCKS_PER_SEC;
	
	printf("%f\n", time_spent1);
	printf("%f\n", time_spent2);
	printf("%0.25f\n", Sum);
	printf("\n");

	clock_t end_all = clock();
	time_all += (double)(end_all - begin_all) / CLOCKS_PER_SEC;
	printf("%f\n", time_all);
	
	return 0;
}
