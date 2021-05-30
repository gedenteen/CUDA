#include "header.h"

struct saxpy_functor
{
	const float a;
	saxpy_functor(float _a) : a(_a) {}
	__host__ __device__
	float operator()(float x, float y) {
		return a * x + y;
	}
};

void saxpy(float a, thrust::device_vector<float>& x,
	thrust::device_vector<float>& y) 
{
	saxpy_functor func(a);
	thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}

void saxpy_thrust(long int arr_size, float alpha, int iterations, 
                  cudaEvent_t start, cudaEvent_t stop, int check_arrays, float *time_arr)
{
	/// создание и заполнение векторов векторов:
	thrust::host_vector<float> X_hos(arr_size);
	thrust::host_vector<float> Y_hos(arr_size);
	thrust::sequence(X_hos.begin(), X_hos.end());
	//thrust::fill(h2.begin(), h2.end(), 0.0);
	thrust::device_vector<float> X_dev = X_hos;
	thrust::device_vector<float> Y_dev = Y_hos;
	
	/// запуск SAXPY на разных размерах массивов
	float _time; //затраченное время на SAXPY
	long int tmp_size = arr_size; //размер массива, который на каждой итерации уменьшаться вдвое
	for (int i = 0; i < iterations; tmp_size = tmp_size >> 1, i++) {
		X_dev.resize(tmp_size);
		Y_dev.resize(tmp_size);
	
		cudaEventRecord(start, 0);
		for (int j = 0; j < 9; j++) //saxpy вызывается несколько раз для большей точности по времени
			saxpy(alpha, X_dev, Y_dev);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 9;
		time_arr[i * TA_COLS + 1] = _time;
		
		if (check_arrays) 
			printf("size of arrays = %ld, Thrust time = %f ms\n", tmp_size, _time);
	}
	
	/// проверка:
	if (check_arrays > 0) {
		X_hos = X_dev;
		Y_hos = Y_dev;
		for (int i = 0; i < check_arrays; i++) {
			printf("i = %d;\t X[i] = %g;\t Y[i] = %g\n", i, X_hos[i], Y_hos[i]);
		}
	}
	if (check_arrays)
		printf("\n");
}

void copying_thrust(long int arr_size, int iterations, int check_arrays,
                    cudaEvent_t start, cudaEvent_t stop, float *time_arr)
{
	/// создание и заполнение векторов векторов:
	thrust::host_vector<float> X_hos(arr_size);
	thrust::sequence(X_hos.begin(), X_hos.end());
	thrust::device_vector<float> X_dev = X_hos;
	thrust::device_vector<float> Y_dev = X_hos;
	
	/// запуск на разных размерах массивов
	float _time; //затраченное время
	long int tmp_size = arr_size; //размер массива, который на каждой итерации уменьшаться вдвое
	for (int i = 0; i < iterations; tmp_size = tmp_size >> 1, i++) {
		X_hos.resize(tmp_size);
		X_dev.resize(tmp_size);
		Y_dev.resize(tmp_size);
	
		cudaEventRecord(start, 0);
		for (int j = 0; j < 3; j++) //копирование вызывается несколько раз для большей точности по времени
			Y_dev = X_dev;
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 3;
		time_arr[i * TA_COLS + 4] = _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("copying device to device, Thrust time = %f ms\n", _time);
		}
		
		cudaEventRecord(start, 0);
		for (int j = 0; j < 3; j++) //копирование вызывается несколько раз для большей точности по времени
			X_hos = X_dev;
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		
		_time /= 3;
		time_arr[i * TA_COLS + 7] = _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("copying device to host, Thrust time = %f ms\n", _time);
		}
	}
}

