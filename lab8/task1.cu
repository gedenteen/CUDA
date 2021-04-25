#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
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


struct saxpy_functor
{
	const float a;
	saxpy_functor(float _a) : a(_a) {}
	__host__ __device__
	float operator()(float x, float y) {
		return a*x+y;
	}
};
void saxpy(float a, thrust::device_vector<float>& x,
	thrust::device_vector<float>& y) 
{
	saxpy_functor func(a);
	thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}
float saxpy_thrust(long int arr_size, cudaEvent_t start, cudaEvent_t stop) 
{ //SAXPY с помощью thrust
	/// создание и заполнение векторов векторов:
	thrust::host_vector<float> h1(arr_size);
	thrust::host_vector<float> h2(arr_size);
	thrust::sequence(h1.begin(), h1.end());
	//thrust::fill(h2.begin(), h2.end(), 0.0);
	thrust::device_vector<float> d1 = h1;
	thrust::device_vector<float> d2 = h2;
	
	/*
	printf("before saxpy:\n");
	for (int i=0; i<16; i++) {
		printf("i = %d; h1[i]=%g; h2[i]=%g\n",i, h1[i], h2[i]);
	}
	*/
	
	cudaEventRecord(start, 0);
		saxpy(2.5, d1, d2);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
		
	/// вывод содержимого векторов после сложения:
	/*
	h2 = d2;
	h1 = d1;
	printf("after saxpy:\n");
	for (int i=0; i<16; i++) {
		printf("i = %d;\t h1[i] = %g;\t h2[i] = %g\n",i, h1[i], h2[i]);
	}
	*/
	
	return time;
}

float saxpy_cublas(long int arr_size, cudaEvent_t start, cudaEvent_t stop) 
{ //SAXPY с помощью cublas:
	const size_t size_in_bytes = (arr_size * sizeof(float));
	
	float *A_dev;
	cudaMalloc( (void **) &A_dev, size_in_bytes );
	float *B_dev;
	cudaMalloc( (void **) &B_dev, size_in_bytes );
	float *A_h;
	cudaMallocHost( (void **) &A_h, size_in_bytes );
	float *B_h;
	cudaMallocHost( (void **) &B_h, size_in_bytes );
	memset(A_h, 0, size_in_bytes);
	memset(B_h, 0, size_in_bytes);
	
	//инициализация библиотеки CUBLAS
	cublasHandle_t cublas_handle;
	CUDA_CHECK_RETURN_CUBLAS(cublasCreate(&cublas_handle));
	
	//заполнение массива А:
	for (int i=0; i < arr_size; i++){
		A_h[i] = (float)i;
	}
	/*
	printf("before saxpy (cublas):\n");
	for (int i = 0; i < 16; i++) {
		printf("i = %d;\t h1[i] = %g;\t h2[i] = %g\n", i+1, A_h[i], B_h[i]);
	}
	*/
	
	const int num_rows = arr_size / 4; //arr_size
	const int num_cols = 4; //1
	const size_t elem_size = sizeof(float);
	
	//Копирование матрицы с числом строк arr_size и одним столбцом с
	//хоста на устройство
	cublasSetMatrix(num_rows, num_cols, elem_size, A_h,
		num_rows, A_dev, num_rows); //leading dimension
	
	//Очищаем массив на устройстве
	cudaMemset(B_dev, 0, size_in_bytes);
	
	// выполнение SingleAlphaXPlusY (saxpy)
	const int stride = 1;
	float alpha = 2.5F;
	
	cudaEventRecord(start, 0);
		cublasSaxpy(cublas_handle, arr_size, &alpha, A_dev,
			stride, B_dev, stride);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	
	/*
	//Копирование матриц с числом строк arr_size и одним столбцом с
	//устройства на хост
	cublasGetMatrix(num_rows, num_cols, elem_size, A_dev,
		num_rows, A_h, num_rows);
	cublasGetMatrix(num_rows, num_cols, elem_size, B_dev,
		num_rows, B_h, num_rows);
	printf("after saxpy (cublas):\n");
	for (int i = 0; i < 16; i++) {
		printf("i = %d;\t h1[i] = %g;\t h2[i] = %g\n", i+1, A_h[i], B_h[i]);
	}
	*/
	
	// Освобождаем ресурсы на устройстве
	cublasDestroy(cublas_handle);
	cudaFree(A_dev);
	cudaFree(B_dev);
	// Освобождаем ресурсы на хосте
	cudaFreeHost(A_h);
	cudaFreeHost(B_h);
	//сброс устройства, подготовка для выполнения новых программ
	//cudaDeviceReset();
	return time;
}

int main(){	
	/// информация об используемом устройстве:
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("device: %s \n\n", deviceProp.name);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	///размерность массивов:
	const long int arr_size = 1 << 25;
	
	float time = saxpy_thrust(arr_size, start, stop);
	printf("Thrust time = %f ms\n", time);
	time = saxpy_cublas(arr_size, start, stop);
	printf("CuBLAS time = %f ms\n", time);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
