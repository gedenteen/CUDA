#include <cufft.h>
#include <stdio.h>
#include <malloc.h>
#define NX 64
#define BATCH 1
#define pi 3.141592

__global__ void gInitData(cufftComplex *data){
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	float x=i*2.0f*pi/(NX);
	data[i].x=cosf(x)-3.0f*sinf(x);
	data[i].y=0.0f;
}

int main(){
	//инициализация (эмуляция получения эксперементальных) данных:
	cufftHandle plan;
	cufftComplex *data;
	cufftComplex *data_h=(cufftComplex*)calloc(NX,sizeof(cufftComplex));;
	cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return -1;
	}
	
	gInitData<<<1, NX>>>(data);
	cudaDeviceSynchronize();
	
	//конфигурация и выполнение cuFFT:
	if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return -1;
	}
	if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return -1;
	}
	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return -1;
	}
	
	cudaMemcpy(data_h, data, NX*sizeof(cufftComplex),
		cudaMemcpyDeviceToHost);
	
	for(int i=0;i<NX;i++)
		printf("%g\t%g\n", data_h[i].x, data_h[i].y);
	
	cufftDestroy(plan);
	cudaFree(data);
	free(data_h);
	return 0;
}
