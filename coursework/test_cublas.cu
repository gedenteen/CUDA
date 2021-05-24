#include "header.h"

float test_cublas(long int arr_size, float alpha, int iterations, 
	cudaEvent_t start, cudaEvent_t stop, int check_arrays) 
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
	
	//Очищаем массив на устройстве
	cudaMemset(Y_dev, 0, size_in_bytes);
	
	// выполнение SingleAlphaXPlusY (saxpy)
	const int stride = 1; //шаг (каждый stride элемент берется из массива)
	
	/// запуск SAXPY на разных размерах массивов
	float _time, time_sum = 0.0f; //затраченное время на SAXPY
	long int tmp_size = arr_size; //размер массива, который на каждой итерации уменьшаться вдвое
	for (int i = 0; i < iterations; tmp_size = tmp_size >> 1, i++) {
		cudaEventRecord(start, 0);
		cublasSaxpy(cublas_handle, tmp_size, &alpha, X_dev, stride, Y_dev, stride);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		time_sum += _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("CUDA time = %f ms\n", _time);
		}
	}
	
	/// проверка:
	if (check_arrays > 0) {
		cudaMemcpy(X_hos, X_dev, size_in_bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(Y_hos, Y_dev, size_in_bytes, cudaMemcpyDeviceToHost);
		for (int i = 0; i < check_arrays; i++) {
			printf("i = %d;\t X[i] = %g;\t Y[i] = %g\n", i, X_hos[i], Y_hos[i]);
		}
	}
	
	/// освобождение ресурсов:
	cublasDestroy(cublas_handle);
	cudaFree(X_dev);
	cudaFree(Y_dev);
	cudaFreeHost(X_hos);
	cudaFreeHost(Y_hos);
	
	/// вернуть среднее время выполнения SAXPY:
	return time_sum / iterations;
}
