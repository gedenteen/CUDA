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

const int N = 1 << 20;

__global__ void gInitVectors(double* vector1, double* vector2) {
	for (int i = 0; i < N; i++) {
		vector1[i] = (double)i; //rand();
		vector2[i] = (double)i;
	}
}

__global__ void gVectorAddition(double* vector1, double* vector2, double* vectorSum, int threads_cnt) { 
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;
	vectorSum[i] = vector1[i] + vector2[i];
	
}

float testingThreadsOfDevice(int threads_cnt, int type_time) {
	double *vectorSum_d, *vectorSum_h;
	vectorSum_h = (double*) calloc(N, sizeof(double));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&vectorSum_d, N * sizeof(double))); 
	double *vector1_d, *vector2_d;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&vector1_d, N * sizeof(double)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&vector2_d, N * sizeof(double))); 
	gInitVectors <<< 1, 32 >>> (vector1_d, vector2_d);
	CUDA_CHECK_RETURN(cudaGetLastError());
	
	float elapsedTime;
	struct timespec mt1, mt2; //для type_time = 1
	cudaEvent_t start, stop; //для type_time = 2
	if (type_time == 1) {
		clock_gettime(CLOCK_REALTIME, &mt1);
	}
	else {
		cudaEventCreate(&start); // инициализация
		cudaEventCreate(&stop); // событий
		cudaEventRecord(start,0); // привязка (регистрация) события start
	}
	
	gVectorAddition <<< N / threads_cnt, threads_cnt >>> 
		(vector1_d, vector2_d, vectorSum_d, threads_cnt); //запуск фу-ии на GPU
	cudaDeviceSynchronize(); //синхронизация потоков
	
	if (type_time == 1) {
		clock_gettime(CLOCK_REALTIME, &mt2);
    		elapsedTime = (float)(mt2.tv_sec - mt1.tv_sec) +
        		(float)(mt2.tv_nsec - mt1.tv_nsec) / 1e6; ///время в миллисекундах
        }
	else {
		cudaEventRecord(stop,0); // привязка события stop
		cudaEventSynchronize(stop); // синхронизация по событию
		CUDA_CHECK_RETURN(cudaGetLastError());
		cudaEventElapsedTime(&elapsedTime,start,stop); // вычисление затраченного времени
		cudaEventDestroy(start); // освобождение
		cudaEventDestroy(stop); // памяти
		CUDA_CHECK_RETURN(cudaGetLastError());
	}
	
	printf("blocks = %d, threads per block = %d seconds = %e \n",
		N / threads_cnt, threads_cnt, elapsedTime);
	
	/// проверка: ///
	/*cudaMemcpy(vectorSum_h, vectorSum_d, N * sizeof(double), cudaMemcpyDeviceToHost); 
	for (int i = 0; i < N; i++)
		fprintf(stderr, "%g ", vectorSum_h[i]);
	printf("\n");
	*/
	
	cudaFree(vector1_d); 
	cudaFree(vector2_d);
	cudaFree(vectorSum_d); 
	free(vectorSum_h);
	return elapsedTime;
}

int main() {
	for (int type_time = 1; type_time <= 2; type_time++) {
		float min_time, max_time, avg_time, cnt_tests = 1;
		//запустить тест с 32 потоками на блок:
		min_time = max_time = avg_time = testingThreadsOfDevice(32, type_time); 
		for (int i = 64; i <= 1024; i *= 2) {
			float new_time = testingThreadsOfDevice(i, type_time);
			if (new_time > max_time)
				max_time = new_time;
			if (new_time < min_time)
				min_time = new_time;
			avg_time += new_time;
			cnt_tests++;
		}
		avg_time = avg_time / cnt_tests;
		if (type_time == 1)
			printf("\n time in milliseconds by clock_gettime:\n");
		else
			printf("\n time in milliseconds by Events:\n");
		printf("\t avg_time = %e min_time = %e max_time = %e\n\n", avg_time, min_time, max_time);
	}

	return 0;
}
