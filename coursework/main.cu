#include "header.h"

int main(int argc, char *argv[]) {	
	/// информация об используемом устройстве:
	cudaDeviceProp deviceProp; 
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("\ndevice: %s \n\n", deviceProp.name);

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
		saxpy_cuda(arr_size, alpha, iterations, start, stop, check_arrays));
	printf("average time = %8f ms for Thrust\n", 
		saxpy_thrust(arr_size, alpha, iterations, start, stop, check_arrays));
	printf("average time = %8f ms for cuBLAS\n", 
		saxpy_cublas(arr_size, alpha, iterations, start, stop, check_arrays));
	
	
	
	float timeDevToDev[3]; //0 - cuda, 1 - thrust, 2 - cublas
	float timeDevToHostUsual[3], timeDevToHostPaged[3]; //для cuda
	for (int i = 0; i < 3; i++)
		timeDevToDev[i] = timeDevToHostUsual[i] = timeDevToHostPaged[i] = 0;
	
	copying_thrust(arr_size, iterations, check_arrays, start, stop, 
		&timeDevToDev[1], &timeDevToHostUsual[1]);
	copying_cuda(arr_size, iterations, check_arrays, start, stop, 
		&timeDevToDev[0], &timeDevToHostUsual[0], &timeDevToHostPaged[0]); //если Куду поместить выше Траста, то ломается...	
	copying_cublas(arr_size, iterations, check_arrays, start, stop, 
		&timeDevToDev[2], &timeDevToHostUsual[2], &timeDevToHostPaged[2]);
		
	printf("   copying array, device to device\n");
	printf("average time = %8f ms for usual CUDA\n", timeDevToDev[0]); 
	printf("average time = %8f ms for Thrust\n", timeDevToDev[1]);
	printf("average time = %8f ms for cuBLAS\n", timeDevToDev[2]);
	
	printf("   copying array, device to host\n");
	printf("average time = %8f ms for usual CUDA with *usual* host memory\n", timeDevToHostUsual[0]);
	printf("average time = %8f ms for usual CUDA with *paged* host memory\n", timeDevToHostPaged[0]);
	printf("average time = %8f ms for Thrust\n", timeDevToHostUsual[1]);
	printf("average time = %8f ms for cuBLAS with *usual* host memory\n", timeDevToHostUsual[2]);
	printf("average time = %8f ms for cuBLAS with *paged* host memory\n", timeDevToHostPaged[2]);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
