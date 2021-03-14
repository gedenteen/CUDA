#include <stdio.h>
#include <time.h>
#include <malloc.h>

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error \"%s\" at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок 

__global__ void gInitVectors(long long n, double* vector1, double* vector2) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
		return;
	vector1[i] = (double)i;
	vector2[i] = (double)i;
}

int main(int argc, char *argv[]) {
	//установить предпочтительную конфигурацию кэша для текущего устройства:
	cudaFuncSetCacheConfig(gInitVectors, cudaFuncCachePreferL1);
	
	if (argc < 3) {
		printf("Error: run program with 2 args: vector size, threads per block\n");
		return 1;
	}
	long long vector_size, threads;
	vector_size = atoi(argv[1]);
	threads = atoi(argv[2]);
	double *vector1_d, *vector2_d;
	
	for (int i = 0; i < 10; i++) {
		CUDA_CHECK_RETURN(cudaMalloc((void**)&vector1_d, vector_size * sizeof(double)));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&vector2_d, vector_size * sizeof(double))); 
		gInitVectors <<< vector_size / threads, threads >>> (vector_size, vector1_d, vector2_d);
		cudaDeviceSynchronize(); 
		CUDA_CHECK_RETURN(cudaGetLastError());
		
		cudaFree(vector1_d); 
		cudaFree(vector2_d);
	}
	return 0;
}
