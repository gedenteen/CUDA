#include <stdio.h>
#include <malloc.h>
const int N = 1 << 20; // 2^20

__global__ void gTest(float* a) { //функция для device
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	a[i] = 0.2 * (float)i;
}

int main() {
	float *a_d, *a_h;
	a_h = (float*) calloc(N, sizeof(float));
	cudaMalloc((void**)&a_d, N * sizeof(float)); //выделить память для переменной device 
	
	gTest <<< N/256, 256 >>> (a_d); //запуск фу-ии на GPU
	cudaDeviceSynchronize(); //синхронизация потоков
	
	//копируем результат
	cudaMemcpy(a_h, a_d, N * sizeof(float), cudaMemcpyDeviceToHost); 
	for (int i = 0; i < N; i++)
		fprintf(stderr, "%g\n", a_h[i]);
	
	free(a_h);
	cudaFree(a_d); 
	return 0;
}
