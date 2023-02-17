#include <stdio.h>
#include <math.h> 
#define _USE_MATH_DEFINES
#define N 10000000
# define M_PI 3.14159265358979323846 // pi

double arr[N];

int main()
{
	for (int i = 0; i < 100; i++)
	{
		arr[i] = sin((i + 1) * M_PI / 180);
	}
	
	return 0;
}
