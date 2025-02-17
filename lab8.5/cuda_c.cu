#include <stdio.h>
//#include <cuda_runtime.h> 
#include <inttypes.h> //для использования uint8_t

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error \"%s\" at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
}

struct complex { //структура для комлексного числа
	float real; //действительная часть
	float imag; //мнимая часть
};

__device__ struct complex complex_mul(struct complex z1, struct complex z2) 
{
	struct complex rez;
	rez.real = z1.real * z2.real - z1.imag * z2.imag;
	rez.imag = z1.real * z2.imag + z2.real * z1.imag;
	return rez;
}
__device__ struct complex complex_sum(struct complex z1, struct complex z2) 
{
	struct complex rez;
	rez.real = z1.real + z2.real;
	rez.imag = z1.imag + z2.imag;
	return rez;
}

__device__ uint8_t mandel(float x, float y, int max_iters) {
	struct complex c; //c = complex(x, y)
	c.real = x, c.imag = y;
	struct complex z; //z = 0.0j
	z.real = z.imag = 0.0f;
	for (int i = 0; i < max_iters; i++) {
		z = complex_mul(z, z); //z = z*z + c; //отображение Мандельброта
		z = complex_sum(z, c);
		if ((z.real * z.real + z.imag * z.imag) >= 4)
			return i;
	}
	return max_iters;
}

__global__ void create_fractal_dev(float min_x, float max_x, float min_y, float max_y, 
	uint8_t *image, int height, int width, int iters) 
{
	float pixel_size_x = (max_x - min_x) / (width); //задание размеров пикселя
	float pixel_size_y = (max_y - min_y) / (height);
	
	int startX = threadIdx.x + blockDim.x * blockIdx.x;
	int startY = threadIdx.y + blockDim.y * blockIdx.y;
	int gridX = gridDim.x * blockDim.x;
	int gridY = gridDim.y * blockDim.y;

	for (int x = startX; x < width; x += gridX) {
		float real = min_x + x * pixel_size_x;
		for (int y = startY; y < height; y += gridY) {
			float imag = min_y + y * pixel_size_y;
			uint8_t color = mandel(real, imag, iters);
			image[x + y * width] = color; //задание цвета пикселя
		}
	} 
}

int main() {
	/// размерности массивов:
	int N = 1024, M = 1536; //размерности массива
	const size_t size_in_bytes = N * M * sizeof(uint8_t);
	
	/// создание массивов:
	uint8_t *A_dev;
	CUDA_CHECK_RETURN(cudaMalloc( (void **) &A_dev, size_in_bytes));
	cudaMemset(A_dev, 0, size_in_bytes); //заполнить нулями
	uint8_t *A_hos;
	A_hos = (uint8_t*) malloc(size_in_bytes);
	cudaMemset(A_hos, 0, size_in_bytes); //заполнить нулями

	/// как расспараллелить код
	dim3 dimBlock(32, 8); //число выделенных блоков
	dim3 dimGrid(32,16); //размер и размерность сетки
	
	/// создание CUDA-событий
	cudaEvent_t start, stop;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	
	/// запуск ядра
	cudaEventRecord(start, 0);
		create_fractal_dev <<<dimGrid,dimBlock >>> (-2.0, 1.0, -1.0, 1.0, A_dev, N, M, 20);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	/// время
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("time = %f s\n", time / 1000);
	
	/// запись массива в файл
	cudaMemcpy(A_hos, A_dev, size_in_bytes, cudaMemcpyDeviceToHost);
	FILE *fp = fopen("rez.dat", "w");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			fprintf(fp, "%d ", A_hos[i * M + j]);  
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	
	/*
	struct complex z1, z2;
	z1.real = 3; z1.imag = 1;
	z2.real = 2; z2.imag = -3;
	struct complex rez = complex_mul(z1, z2);
	printf("%f %f \n", rez.real, rez.imag);
	rez = complex_sum(z1, z2);
	printf("%f %f \n", rez.real, rez.imag);
	*/
	return 0;
}
