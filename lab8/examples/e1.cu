#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error \"%s\" at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок 

#define CUDA_CHECK_RETURN_CUBLAS(value) {\
	cublasStatus_t stat = value;\
	if (stat != CUBLAS_STATUS_SUCCESS) {\
		fprintf(stderr, "Error at line %d in file %s\n",\
			__LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок 

__host__ void print_array(float * data1,
	float * data2, int num_elem, const char * prefix)
{
	printf("\n%s", prefix);
	for (int i = 0; i < num_elem; i++) {
		printf("\n%2d: %2.4f %2.4f ", i+1, data1[i], data2[i]);
	}
}

int main(){
	const int num_elem = 8;
	const size_t size_in_bytes = (num_elem * sizeof(float));
	
	float * A_dev;
	cudaMalloc( (void **) &A_dev, size_in_bytes );
	float * B_dev;
	cudaMalloc( (void **) &B_dev, size_in_bytes );
	
	float * A_h;
	cudaMallocHost( (void **) &A_h, size_in_bytes );
	float * B_h;
	cudaMallocHost( (void **) &B_h, size_in_bytes );
	memset(A_h, 0, size_in_bytes);
	memset(B_h, 0, size_in_bytes);
	
	// Инициализация библиотеки CUBLAS
	cublasHandle_t cublas_handle;
	CUDA_CHECK_RETURN_CUBLAS(cublasCreate(&cublas_handle));
	
	for (int i=0; i < num_elem; i++){
		A_h[i] = (float)i;
	}
	print_array(A_h, B_h, num_elem, "Arrays before set");
	
	const int num_rows = num_elem;
	const int num_cols = 1;
	const size_t elem_size = sizeof(float);
	
	//Копирование матрицы с числом строк num_elem и одним столбцом с
	//хоста на устройство
	cublasSetMatrix(num_rows, num_cols, elem_size, A_h,
		num_rows, A_dev, num_rows); //leading dimension
	
	//Очищаем массив на устройстве
	cudaMemset(B_dev, 0, size_in_bytes);
	
	// выполнение SingleAlphaXPlusY (saxpy)
	const int stride = 1;
	float alpha = 2.0F;
	cublasSaxpy(cublas_handle, num_elem, &alpha, A_dev,
		stride, B_dev, stride);
	//Копирование матриц с числом строк num_elem и одним столбцом с
	//устройства на хост
	cublasGetMatrix(num_rows, num_cols, elem_size, A_dev,
		num_rows, A_h, num_rows);
	cublasGetMatrix(num_rows, num_cols, elem_size, B_dev,
		num_rows, B_h, num_rows);
	print_array(A_h, B_h, num_elem, "saxpy, alpha = 2.0");
	
	alpha = 1.0F;
	// повторное выполнение SingleAlphaXPlusY
	cublasSaxpy(cublas_handle, num_elem, &alpha, A_dev,
	stride, B_dev, stride);
	
	//Копирование матриц с числом строк num_elem и одним столбцом с
	//устройства на хост
	cublasGetMatrix(num_rows, num_cols, elem_size, A_dev,
	num_rows, A_h, num_rows);
	cublasGetMatrix(num_rows, num_cols, elem_size, B_dev,
	num_rows, B_h, num_rows);
	
	// Удостоверяемся, что все асинхронные вызовы выполнены
	cudaStream_t default_stream;//const int default_stream = 0;
	cudaStreamCreate(&default_stream);
	cudaStreamSynchronize(default_stream);
	
	// Print out the arrays
	print_array(A_h, B_h, num_elem, "one more saxpy, alpha = 1.0");
	printf("\n");
	
	// Освобождаем ресурсы на устройстве
	cublasDestroy(cublas_handle);
	cudaFree(A_dev);
	cudaFree(B_dev);
	// Освобождаем ресурсы на хосте
	cudaFreeHost(A_h);
	cudaFreeHost(B_h);
	//сброс устройства, подготовка для выполнения новых программ
	cudaDeviceReset();

	return 0;
}
