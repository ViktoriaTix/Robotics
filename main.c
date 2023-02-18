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
	// для хранения времени выполнения кода
	double time_spent = 0.0;

	clock_t begin = clock();

#pragma acc kernels
	for (int i = 0; i < N; i++)
	{
		arr[i] = sin(i * M_PI *2 / N);
		Sum += arr[i];
	}
	
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("%f\n", time_spent);
	printf("%0.25f", Sum);

	return 0;
}
