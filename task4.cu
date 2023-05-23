#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// функция изменения матрицы уравнения теплопроводности
__global__ void calculate(double *CudaArr, double *CudaNewArr, size_t Matrix)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x; //вычисления линейного индекса элемента внутри сетки 
    size_t j =  blockDim.y * blockIdx.y + threadIdx.y; 
    int index = i * Matrix + j;
    if (!(i == 0 || j == 0))
        CudaNewArr[index] = 0.25 * (CudaArr[(i - 1) * Matrix + j] + CudaArr[(i + 1) * Matrix + j] + CudaArr[index - 1] + CudaArr[index + 1]);
}


// функция разницы матриц
__global__ void subtraction(double* CudaArr, double* CudaNewArr, size_t Matrix)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  
    size_t j =  blockDim.y * blockIdx.y + threadIdx.y;
    int idx = i * Matrix + j; 
    if ((i < Matrix && j < Matrix))
	    CudaNewArr[idx] = CudaArr[idx] - CudaNewArr[idx];
}

// функция востановления границ матрицы
__global__ void restore(double* arr, int size){
	size_t i = threadIdx.x;
	arr[i] = 10.0 + i * 10.0 / (size - 1);
	arr[i * size] = 10.0 + i * 10.0 / (size - 1);
	arr[size - 1 + i * size] = 20.0 + i * 10.0 / (size - 1);
	arr[size * (size - 1) + i] = 20.0 + i * 10.0 / (size - 1);
}

int find_threads(int size){
	if (size%32==0)
		return size/1024;

	return int(size/1024)+1;

}

int main(int argc, char* argv[]) {
    
    double time_spent = 0.0;

    clock_t begin = clock(); 

    // Convert command line arguments to integers
    int Matrix = atoi(argv[1]);
    double accuracy = atof(argv[2]);
    int iterations = atoi(argv[3]);

    cudaSetDevice(1);
    
    // создание потока
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // создание графа
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    // выделяем память на gpu через cuda для 2 сеток
    double *CudaArr, *CudaNewArr;
    cudaMalloc((void **)&CudaArr, sizeof(double) * Matrix * Matrix);
    cudaMalloc((void **)&CudaNewArr, sizeof(double) * Matrix * Matrix);

    restore<<<1, Matrix>>>(CudaArr, Matrix);
    cudaMemcpy(CudaNewArr, CudaArr, sizeof(double) * Matrix * Matrix, cudaMemcpyHostToDevice);

    // выделяем память на gpu. Хранение ошибки на device
    double *max_err = 0;
    cudaMalloc((void **)&max_err, sizeof(double));

    size_t tempStorageBytes = 0;
    double *tempStorage = NULL;

    // получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaNewArr, max_err, Matrix * Matrix, stream);

    // выделяем память для буфера
    cudaMalloc(&tempStorage, tempStorageBytes);

    dim3 t(32,32); //определяю количество нитей в каждом блоке 
    int b = find_threads(Matrix);

    ///////////////////////////////////////////////////////////////создаем граф
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Compute new values
    for (size_t i = 0; i < 100; i += 2) {
        //количество блоков, потоков, разделяемая память, поток
        calculate<<<b, t, 0, stream>>>(CudaArr, CudaNewArr, Matrix);
        calculate<<<b, t, 0, stream>>>(CudaNewArr, CudaArr, Matrix);
    }
    subtraction<<<b, t, 0, stream>>>(CudaArr, CudaNewArr, Matrix);

    // Compute maximum error using CUB
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaNewArr, max_err, Matrix * Matrix, stream);
    restore<<<1, Matrix, 0, stream>>>(CudaNewArr, Matrix);

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

    /////////////////////////////////////////////////////////////////

    // Main loop
    double err = 1;
    int iter = 0;

    while (err > accuracy && iter < iterations) {

        cudaGraphLaunch(graph_exec, stream);
        // запись ошибки в переменную
        cudaMemcpy(&err, max_err, sizeof(double), cudaMemcpyDeviceToHost);
        iter+=100;  
    }

    printf("Final result: %d, %0.6lf\n", iter, err);

    // удаление потока и графа
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);

    cudaFree(CudaArr);
    cudaFree(CudaNewArr);

    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f\n", time_spent);

    return 0;
}
