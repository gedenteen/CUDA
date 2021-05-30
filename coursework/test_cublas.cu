#include "header.h"

void saxpy_cublas(long int arr_size, float alpha, int iterations, 
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
	
	/// инициализация библиотеки CUBLAS:
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK_RETURN(cublasCreate(&cublas_handle));
	
	/// заполнение массивов:
	for (int i=0; i < arr_size; i++){
		X_hos[i] = (float)i;
	}
	memset(Y_hos, 0, size_in_bytes);
	
	const int num_rows = arr_size; //arr_size
	const int num_cols = 1; //1
	const size_t elem_size = sizeof(float);
	
	//Копирование матрицы с числом строк arr_size и одним столбцом с
	//хоста на устройство
	cublasSetMatrix(num_rows, num_cols, elem_size, X_hos,
		num_rows, X_dev, num_rows); //leading dimension
	cudaMemset(Y_dev, 0, size_in_bytes);
	
	/// запуск SAXPY на разных размерах массивов
	const int stride = 1; //шаг (каждый stride элемент берется из массива)
	float _time; //затраченное время на SAXPY
	long int tmp_size = arr_size; //размер массива, который на каждой итерации уменьшаться вдвое
	for (int i = 0; i < iterations; tmp_size = tmp_size >> 1, i++) {
		cudaEventRecord(start, 0);
		for (int j = 0; j < 9; j++) //saxpy вызывается несколько раз для большей точности по времени
			cublasSaxpy(cublas_handle, tmp_size, &alpha, X_dev, stride, Y_dev, stride);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 9;
		time_arr[i * TA_COLS + 2] = _time;
		
		if (check_arrays) 
			printf("size of arrays = %ld, cuBLAS time = %f ms\n", tmp_size, _time);
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
	cublasDestroy(cublas_handle);
	cudaFree(X_dev);
	cudaFree(Y_dev);
	cudaFreeHost(X_hos);
	cudaFreeHost(Y_hos);
}

void copying_cublas(long int arr_size, int iterations, int check_arrays,
                    cudaEvent_t start, cudaEvent_t stop, float *time_arr)  
{
	/// выделение памяти:
	float *host_usual_arr, *host_paged_arr, *dev1_arr, *dev2_arr;
    //выделение обычной памяти на хосте:
	long int size_in_bytes = arr_size * sizeof(float);
    host_usual_arr = (float*)malloc(size_in_bytes);
    //выделение закрепленной (paged-locked) памяти на хосте:
    cudaHostAlloc((void**)&host_paged_arr, size_in_bytes, cudaHostAllocDefault);
    //выделение памяти на девайсе:
    cudaMalloc((void**)&dev1_arr, size_in_bytes);
    cudaMalloc((void**)&dev2_arr, size_in_bytes);
	
	/// инициализация библиотеки CUBLAS:
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK_RETURN(cublasCreate(&cublas_handle));
	
	/// заполнение массивов:
	for (int i=0; i < arr_size; i++) { //заполнить массив последовательностью
		host_usual_arr[i] = (float)i;
	}
	const int num_rows = arr_size; //arr_size
	const int num_cols = 1; //1
	const size_t elem_size = sizeof(float);
	cublasSetMatrix(num_rows, num_cols, elem_size, host_usual_arr,
		num_rows, dev1_arr, num_rows); //leading dimension
	memset(host_usual_arr, 0, size_in_bytes); //убрать последовательность из массива, занулить
	cudaMemset(dev2_arr, 0, size_in_bytes);
    	
    /// копирование массива с разными размерностями 
	const int stride = 1; //шаг (каждый stride элемент берется из массива)
	float _time; //затраченное время
	long int tmp_size = arr_size; //размер массива, который на каждой итерации уменьшаться вдвое
	for (int i = 0; i < iterations; tmp_size = tmp_size >> 1, i++) {
		cudaEventRecord(start, 0);
    	for (int j = 0; j < 3; j++) //копирование вызывается несколько раз для большей точности по времени
			cublasScopy(cublas_handle, tmp_size, dev1_arr, stride, dev1_arr, stride);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 3;
		time_arr[i * TA_COLS + 5] = _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("copying device to device, cuBLAS time = %f ms\n", _time);
		}
		
		cudaEventRecord(start, 0);
		for (int j = 0; j < 3; j++) //копирование вызывается несколько раз для большей точности по времени
			cublasGetMatrix(tmp_size, num_cols, elem_size, dev1_arr, tmp_size, host_usual_arr, tmp_size);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 3;
		time_arr[i * TA_COLS + 8] = _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("copying device to host usual, cuBLAS time = %f ms\n", _time);
		}
		
		cudaEventRecord(start, 0);
		for (int j = 0; j < 3; j++) //копирование вызывается несколько раз для большей точности по времени
			cublasGetMatrix(tmp_size, num_cols, elem_size, dev1_arr, tmp_size, host_paged_arr, tmp_size);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 3;
		time_arr[i * TA_COLS + 9] = _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("copying device to host paged, cuBLAS time = %f ms\n", _time);
		}
	}
	
	/// освобождение ресурсов:
	cublasDestroy(cublas_handle);
	cudaFree(dev1_arr);
	cudaFree(dev2_arr);
	cudaFreeHost(host_usual_arr);
	cudaFreeHost(host_paged_arr);
}
