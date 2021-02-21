#include <cuda.h>
#include <stdio.h>
#include <malloc.h>

/*
#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error %s at line in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
}
*/

void CUDA_CHECK_RETURN(cudaError_t _m_cudaStat) {
	//cudaError_t _m_cudaStat = value;
	if (_m_cudaStat != cudaSuccess) {
		fprintf(stderr, "Error \"%s\" at line %d in file %s\n",
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);
		exit(1);
	}
}

__global__ void gTest(double* a) {
	a[threadIdx.x + blockDim.x * blockIdx.x] = 
		(double)(threadIdx.x + blockDim.x * blockIdx.x);
}

int main() {
	double *device_a, *host_a;
	int num_of_blocks = 10, threads_per_block = 1025; //error here
	int N = num_of_blocks * threads_per_block;
	
	host_a = (double*) calloc(N, sizeof(double));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&device_a, N * sizeof(double)));
	
	float elapsedTime;
	cudaEvent_t start,stop; // встроенный тип данных – структура, для
		// фиксации контрольных точек
	cudaEventCreate(&start); // инициализация
	cudaEventCreate(&stop); // событий
	cudaEventRecord(start,0); // привязка (регистрация) события start
	
	gTest <<< dim3(num_of_blocks), dim3(threads_per_block) >>> (device_a);
	
	cudaEventRecord(stop,0); // привязка события stop
	cudaEventSynchronize(stop); // синхронизация по событию
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	
	cudaEventElapsedTime(&elapsedTime,start,stop); // вычисление
	// затраченного времени
	fprintf(stderr,"gTest took %g\n", elapsedTime);
	cudaEventDestroy(start); // освобождение
	cudaEventDestroy(stop); // памяти
	
	CUDA_CHECK_RETURN(cudaMemcpy(host_a, device_a, N * sizeof(double),
		cudaMemcpyDeviceToHost));
	//for(int i = 0; i < N; i++)
	//	printf("%g ", host_a[i]);
		
	free(host_a);
	cudaFree(device_a);
	return 0;
}
