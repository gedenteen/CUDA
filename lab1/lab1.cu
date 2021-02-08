#include <stdio.h>
#include <time.h>
#include <malloc.h>
const int N = 1 << 20;

__global__ void gInitVectors(double* vector1, double* vector2) {
	for (int i = 0; i < N; i++) {
		vector1[i] = (double)i; //rand();
		vector2[i] = (double)i;
	}
}

__global__ void gVectorAddition(double* vector1, double* vector2, double* vectorSum, int threads_cnt) { 
	//int i = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = threadIdx.x; i < N; i += threads_cnt)  {
		vectorSum[i] = vector1[i] + vector2[i];
	}
}

double testingThreadsOfDevice(int threads_cnt) {
	double *vector1_d, *vector2_d, *vectorSum_d, *vectorSum_h;
	vectorSum_h = (double*) calloc(N, sizeof(double));
	cudaMalloc((void**)&vector1_d, N * sizeof(double));
	cudaMalloc((void**)&vector2_d, N * sizeof(double)); 
	cudaMalloc((void**)&vectorSum_d, N * sizeof(double)); 
	gInitVectors <<< 1, 1 >>> (vector1_d, vector2_d);
	
	/// проверка: ///
	/*cudaMemcpy(vectorSum_h, vector1_d, N * sizeof(double), cudaMemcpyDeviceToHost); 
	for (int i = 0; i < N; i++)
		fprintf(stderr, "%g ", vectorSum_h[i]);
	printf("\n");
	*/
	
	struct timespec mt1, mt2;
	clock_gettime(CLOCK_REALTIME, &mt1);
	
	gVectorAddition <<< 1, threads_cnt >>> 
		(vector1_d, vector2_d, vectorSum_d, threads_cnt); //запуск фу-ии на GPU
	cudaDeviceSynchronize(); //синхронизация потоков
	
	/// проверка: ///
	/*cudaMemcpy(vectorSum_h, vectorSum_d, N * sizeof(double), cudaMemcpyDeviceToHost); 
	for (int i = 0; i < N; i++)
		fprintf(stderr, "%g ", vectorSum_h[i]);
	printf("\n");
	*/
		
	clock_gettime(CLOCK_REALTIME, &mt2);
    	double seconds_double = (double)(mt2.tv_sec - mt1.tv_sec) +
        	(double)(mt2.tv_nsec - mt1.tv_nsec) / 1e9; ///время в секундах
	//printf("time in seconds = %g \n", seconds_double);
	
	cudaFree(vector1_d); 
	cudaFree(vector2_d); 
	cudaFree(vectorSum_d); 
	free(vectorSum_h);
	return seconds_double;
}

int main() {
	testingThreadsOfDevice(1);
	FILE *fout;
	if ((fout = fopen("output.csv", "w")) == NULL) {
		printf("error: can't open output.csv \n");
		return 1;
	}

	fprintf(fout, "threads_per_block;time_in_seconds;\n");
	for (int i = 1; i <= 64; i++)
		fprintf(fout, "%d;%g;\n", i, testingThreadsOfDevice(i));

	return 0;
}
