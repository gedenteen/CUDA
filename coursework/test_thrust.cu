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

void saxpy_thrust(float a, thrust::device_vector<float>& x,
	thrust::device_vector<float>& y) 
{
	saxpy_functor func(a);
	thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}

float test_thrust(long int arr_size, float alpha, int iterations, 
	cudaEvent_t start, cudaEvent_t stop, int check_arrays) 
{
	/// создание и заполнение векторов векторов:
	thrust::host_vector<float> X_hos(arr_size);
	thrust::host_vector<float> Y_hos(arr_size);
	thrust::sequence(X_hos.begin(), X_hos.end());
	//thrust::fill(h2.begin(), h2.end(), 0.0);
	thrust::device_vector<float> X_dev = X_hos;
	thrust::device_vector<float> Y_dev = Y_hos;
	
	/// запуск SAXPY на разных размерах массивов
	float _time, time_sum = 0.0f; //затраченное время на SAXPY
	long int tmp_size = arr_size; //размер массива, который на каждой итерации уменьшаться вдвое
	for (int i = 0; i < iterations; tmp_size = tmp_size >> 1, i++) {
		X_dev.resize(tmp_size);
		Y_dev.resize(tmp_size);
	
		cudaEventRecord(start, 0);
		saxpy_thrust(alpha, X_dev, Y_dev);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&_time, start, stop);
		time_sum += _time;
		
		if (check_arrays) {
			printf("size of arrays = %ld\n", tmp_size); 
			printf("Thrust time = %f ms\n", _time);
		}
	}
	
	/*
	cudaEventRecord(start, 0);
		saxpy(2.5, d1, d2);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	*/
		
	/// вывод содержимого векторов после сложения:
	/*
	h2 = d2;
	h1 = d1;
	printf("after saxpy:\n");
	for (int i=0; i<16; i++) {
		printf("i = %d;\t h1[i] = %g;\t h2[i] = %g\n",i, h1[i], h2[i]);
	}
	*/
	
	/// проверка:
	if (check_arrays > 0) {
		X_hos = X_dev;
		Y_hos = Y_dev;
		for (int i = 0; i < check_arrays; i++) {
			printf("i = %d;\t X[i] = %g;\t Y[i] = %g\n", i, X_hos[i], Y_hos[i]);
		}
	}
	
	/// вернуть среднее время выполнения SAXPY:
	return time_sum / iterations;
}
