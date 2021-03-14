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

__global__ void gTranspose0(float* storage_d, float* storage_d_t){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int j=threadIdx.y+blockIdx.y*blockDim.y;
	int N=blockDim.x*gridDim.x;
	storage_d_t[j+i*N]=storage_d[i+j*N];
}

__global__ void gInitializeMatrixByRows(long long n, double* matrix_d){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int N = blockDim.x * gridDim.x;
	matrix_d[i+j*N] = (double)(i+j*N);
}

__global__ void gInitializeMatrixByColumns(long long n, double* matrix_d){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int N = blockDim.x * gridDim.x;
	matrix_d[j+i*N] = (double)(j+i*N);
}

int main(int argc, char *argv[]) {
	//установить предпочтительную конфигурацию кэша для текущего устройства:
	//cudaFuncSetCacheConfig(gInitVectors, cudaFuncCachePreferL1);

	if (argc < 3) {
		printf("Error: run program with 2 args: n, threads per block\n");
		return 1;
	}

	long long n, threads;
	n = atoi(argv[1]);
	threads = atoi(argv[2]);
	double *matrix1_d, *matrix2_d;
	
	for (int i = 0; i < 10; i++) {
		CUDA_CHECK_RETURN(cudaMalloc((void**)&matrix1_d, n * n * sizeof(double)));
		gInitializeMatrixByRows <<< n / threads, threads >>> (n, matrix1_d);
		cudaDeviceSynchronize(); 
		CUDA_CHECK_RETURN(cudaGetLastError());
		cudaFree(matrix1_d); 
		
		CUDA_CHECK_RETURN(cudaMalloc((void**)&matrix2_d, n * n * sizeof(double)));
		gInitializeMatrixByColumns <<< n / threads, threads >>> (n, matrix2_d);
		cudaDeviceSynchronize(); 
		CUDA_CHECK_RETURN(cudaGetLastError());
		cudaFree(matrix2_d); 
	}
	return 0;
}
