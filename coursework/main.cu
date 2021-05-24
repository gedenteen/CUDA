#include "header.h"

int main() {	
	/// информация об используемом устройстве:
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("device: %s \n\n", deviceProp.name);

	/// Куда-события для замера времени:
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	/// параметры тестирования:
	long int arr_size = 1 << 25; //максимальный размер массивов
	float alpha = 2.5f; //коэффициент умножения для массива X
	int iterations = 6; //сколько раз запускать SAXPY
	int check_arrays = 0; //
	
	/// посчитать среднее время и вывести результаты:
	printf("   SAXPY\n");
	printf("average time = %8f ms for usual CUDA\n", 
		test_cuda(arr_size, alpha, iterations, start, stop, check_arrays));
	printf("average time = %8f ms for Thrust\n", 
		test_thrust(arr_size, alpha, iterations, start, stop, check_arrays));
	printf("average time = %8f ms for cuBLAS\n", 
		test_cublas(arr_size, alpha, iterations, start, stop, check_arrays));
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
