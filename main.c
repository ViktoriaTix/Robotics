#include <stdio.h>
#include <math.h> 
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC

#define _USE_MATH_DEFINES
#define N 10000000
#define M_PI 3.14159265358979323846 // pi
	
double Sum=0;
double arr[N];

int main()
{
	double time_spent = 0.0;

	clock_t begin = clock();

#pragma acc kernels
	for (int i = 0; i < N; i++)
	{
		arr[i] = sin(i * M_PI *2 / N);
	}

	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	
	for (int i = 0; i < N; i++)
	{
		Sum += arr[i];
	}
	
	printf("%f\n", time_spent);
	printf("%0.25f\n", Sum);

	return 0;
}
