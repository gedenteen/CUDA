#include "header.h"

int main(int argc, char *argv[]) {	
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
	int check_arrays = 0; //сколько элементов массива вывести на экран
	
	/// пользователь может изменить эти параметры, задав их при запуске:
	if (argc >= 2) {
		int degree = atoll(argv[1]); //преобразование строки в long int
		if (degree > 0)
			arr_size = 1 << degree;
		else 
			fprintf(stderr, "error, degree should be > 0\n");
	}
	if (argc >= 3) {
		alpha = atof(argv[2]); //преобразование строки в float
	}
	if (argc >= 4) {
		int tmp = atoll(argv[3]);
		if (tmp > 0)
			iterations = tmp;
		else
			fprintf(stderr, "error, iterations should be > 0\n");
	}
	if (argc >= 5) {
		check_arrays = atoll(argv[4]);
	}
	
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
