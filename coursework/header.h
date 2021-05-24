#ifndef HEADER_H
#define HEADER_H

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
} //макрос для обработки ошибок CUDA

#define CUBLAS_CHECK_RETURN(value) {\
	cublasStatus_t stat = value;\
	if (stat != CUBLAS_STATUS_SUCCESS) {\
		fprintf(stderr, "Error at line %d in file %s\n",\
			__LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок CUBLAS

float saxpy_cuda(long int arr_size, float alpha, int iterations, 
                 cudaEvent_t start, cudaEvent_t stop, int check_arrays);
void copying_cuda(long int arr_size, int iterations, int check_arrays,
                  cudaEvent_t start, cudaEvent_t stop,
                  float *timeDevToDev, float *timeDevToHosUsual, float *timeDevToHosPaged);
	
float saxpy_thrust(long int arr_size, float alpha, int iterations, 
                   cudaEvent_t start, cudaEvent_t stop, int check_arrays);
void copying_thrust(long int arr_size, int iterations, int check_arrays,
                    cudaEvent_t start, cudaEvent_t stop,
                    float *timeDevToDev, float *timeDevToHos);
	
float saxpy_cublas(long int arr_size, float alpha, int iterations, 
                   cudaEvent_t start, cudaEvent_t stop, int check_arrays);
void copying_cublas(long int arr_size, int iterations,  int check_arrays,
                    cudaEvent_t start, cudaEvent_t stop,
                    float *timeDevToDev, float *timeDevToHosUsual, float *timeDevToHosPaged);

#endif
