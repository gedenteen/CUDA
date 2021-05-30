#include "header.h"

__global__ void saxpy(int arr_size, float alpha, float *x, float *y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < arr_size)
		y[i] = alpha * x[i] + y[i];
}

void saxpy_cuda(long int arr_size, float alpha, int iterations, 
	cudaEvent_t start, cudaEvent_t stop, int check_arrays, float *time_arr)  
{
	/// создание массивов:
	long int size_in_bytes = arr_size * sizeof(float);
	float *X_dev;
	cudaMalloc((void **) &X_dev, size_in_bytes);
	float *Y_dev;
	cudaMalloc((void **) &Y_dev, size_in_bytes);
	float *X_hos;
	cudaMallocHost((void **) &X_hos, size_in_bytes);
	float *Y_hos;
	cudaMallocHost((void **) &Y_hos, size_in_bytes);
	
	/// заполнение массивов:
	for (int i=0; i < arr_size; i++){
		X_hos[i] = (float)i;
	}
	memset(Y_hos, 0, size_in_bytes);

	/// копирование на массивы устройства:
	cudaMemcpy(X_dev, X_hos, size_in_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(Y_dev, Y_hos, size_in_bytes, cudaMemcpyHostToDevice);

	/// запуск SAXPY на разных размерах массивов
	float _time; //затраченное время на SAXPY
	long int tmp_size = arr_size; //размер массива, который на каждой итерации уменьшаться вдвое
	for (int i = 0; i < iterations; tmp_size = tmp_size >> 1, i++) {
		cudaEventRecord(start, 0);
		for (int j = 0; j < 9; j++) //saxpy вызывается несколько раз для большей точности по времени
		{
			saxpy <<< tmp_size / 256, 256 >>> (tmp_size, alpha, X_dev, Y_dev);
			cudaDeviceSynchronize(); //синхронизация потоков
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);

		_time /= 9; //посчитать среднее время выполнения saxpy
		time_arr[i * TA_COLS] = _time; //записать время в общий массив
		
		if (check_arrays) 
			printf("size of arrays = %ld, CUDA time = %f ms\n", tmp_size, _time);
	}
	
	/// проверка:
	if (check_arrays > 0) {
		cudaMemcpy(X_hos, X_dev, size_in_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(Y_hos, Y_dev, size_in_bytes, cudaMemcpyDeviceToHost);
		for (int i = 0; i < check_arrays; i++) {
			printf("i = %d;\t X[i] = %g;\t Y[i] = %g\n", i, X_hos[i], Y_hos[i]);
		}
	}
	if (check_arrays)
		printf("\n");
	
	/// освобождение ресурсов:
	cudaFree(X_dev);
	cudaFree(Y_dev);
	cudaFreeHost(X_hos);
	cudaFreeHost(Y_hos);
}

__global__ void gInitArray(long int arr_size, float* arr) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= arr_size)
        return;
    arr[i] = (float)i;
}

void copying_cuda(long int arr_size, int iterations, int check_arrays,
	cudaEvent_t start, cudaEvent_t stop, float *time_arr) 
{
	/// создание массивов:
    float *host_usual_arr, *host_paged_arr, *dev1_arr, *dev2_arr;
	long int size_in_bytes = arr_size * sizeof(float); 
    host_usual_arr = (float*)malloc(size_in_bytes); //выделение обычной памяти на хосте
    cudaHostAlloc((void**)&host_paged_arr, size_in_bytes, cudaHostAllocDefault); //выделение закрепленной (paged-locked) памяти на хосте
    cudaMalloc((void**)&dev1_arr, size_in_bytes); //выделение памяти на девайсе
    cudaMalloc((void**)&dev2_arr, size_in_bytes);

	/// заполнение массива:
    gInitArray <<< arr_size / 256, 256 >>> (arr_size, dev1_arr); 
    cudaDeviceSynchronize();
    
    /// запуск на разных размерах массивов:
	float _time; //затраченное время
	long int tmp_size = arr_size; //размер массива, который на каждой итерации уменьшаться вдвое
	for (int i = 0; i < iterations; tmp_size = tmp_size >> 1, i++) {
		cudaEventRecord(start, 0);
		for (int j = 0; j < 3; j++) //копирование вызывается несколько раз для большей точности по времени
			cudaMemcpy(dev2_arr, dev1_arr, tmp_size * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 3;
		time_arr[i * TA_COLS + 3] = _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("copying device to device, CUDA time = %f ms\n", _time);
		}
		
		cudaEventRecord(start, 0);
		for (int j = 0; j < 3; j++) //копирование вызывается несколько раз для большей точности по времени
    		cudaMemcpy(host_usual_arr, dev1_arr, tmp_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 3;
		time_arr[i * TA_COLS + 6] = _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("copying device to host usual, CUDA time = %f ms\n", _time);
		}
		
		cudaEventRecord(start, 0);
		for (int j = 0; j < 3; j++) //копирование вызывается несколько раз для большей точности по времени
			cudaMemcpy(host_paged_arr, dev1_arr, tmp_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 3;
		time_arr[i * TA_COLS + 7] = _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("copying device to host paged, CUDA time = %f ms\n", _time);
		}
	}
	
	/// освобождение ресурсов:
	cudaFree(dev1_arr);
	cudaFree(dev2_arr);
	cudaFreeHost(host_usual_arr);
	cudaFreeHost(host_paged_arr);
}
