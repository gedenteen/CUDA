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
	int degree = 25; //степень двойки - размер массива
	float alpha = 2.5f; //коэффициент умножения для массива X
	int iterations = 6; //сколько раз запускать SAXPY
	int check_arrays = 0; //сколько элементов массива вывести на экран
	
	/// пользователь может изменить эти параметры, задав их при запуске:
	if (argc >= 2) {
		degree = atoi(argv[1]); //преобразование строки в long int
		if (degree <= 0) { 
			fprintf(stderr, "error, degree should be > 0\n");
			degree = 25;	
		}
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

	/// массив для записи результатов тестирования:
	long int arr_size = 1 << degree; //максимальный размер массивов
	float *time_arr = (float*) malloc(iterations * TA_COLS * sizeof(float));
	for (int i = 0; i < iterations; i++) {
		for (int j = 0; j < TA_COLS; j++) 
			time_arr[i * TA_COLS + j] = 0.0f;
	}
	
	/// посчитать среднее время и вывести результаты:
	printf("   SAXPY\n");
	saxpy_thrust(arr_size, alpha, iterations, start, stop, check_arrays, time_arr);
	saxpy_cublas(arr_size, alpha, iterations, start, stop, check_arrays, time_arr);
	saxpy_cuda(arr_size, alpha, iterations, start, stop, check_arrays, time_arr);
	
	copying_thrust(arr_size, iterations, check_arrays, start, stop, time_arr);
	copying_cuda(arr_size, iterations, check_arrays, start, stop, time_arr); //если Куду поместить выше Траста, то ломается...	
	copying_cublas(arr_size, iterations, check_arrays, start, stop, time_arr);
	
	/*
	for (int i = 0; i < iterations; i++) {
		for (int j = 0; j < TA_COLS; j++) 
			printf("%g ", time_arr[i * TA_COLS + j]);
		printf("\n");
	}*/
	
	FILE *fp;
	fp = fopen("graphs/time.csv", "w");
	if (fp == NULL) {
		fprintf(stderr, "error: can't open graphs/time.dat\n");
		exit(EXIT_FAILURE); 
	}
	///TODO: подписи к столбцам
	for (int i = 0; i < iterations; i++) {
		fprintf(fp, "%d;", degree - i);
		for (int j = 0; j < TA_COLS; j++) {
			fprintf(fp, "%g;", time_arr[i * TA_COLS + j]);
		}
		fprintf(fp, "\n");
	}
	
	fclose(fp);
	fp = fopen("graphs/ratio.csv", "w");
	if (fp == NULL) {
		fprintf(stderr, "error: can't open graphs/ratio.dat\n");
		exit(EXIT_FAILURE); 
	}
	for (int i = 0; i < iterations; i++) {
		fprintf(fp, "%d;", degree - i);
		//время saxpy-CUDA поделить на время saxpy-Thrust:
		fprintf(fp, "%g;", time_arr[i * TA_COLS] / time_arr[i * TA_COLS + 1]);
		//время saxpy-cuBLAS поделить на время saxpy-Thrust:
		fprintf(fp, "%g;", time_arr[i * TA_COLS + 2] / time_arr[i * TA_COLS + 1]);
		fprintf(fp, "\n");
	}
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
