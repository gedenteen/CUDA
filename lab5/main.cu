#include <stdio.h>
#include <malloc.h>
//#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error \"%s\" at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок 

#define M_PI 3.14159265358979323846
#define COEF 48
#define VERTCOUNT COEF*COEF*2
#define RADIUS 10.0f
#define FGSIZE 20
#define FGSHIFT FGSIZE/2
#define IMIN(A,B) (A<B?A:B)
#define THREADSPERBLOCK 256
#define BLOCKSPERGRID IMIN(32,(VERTCOUNT+THREADSPERBLOCK-1)/THREADSPERBLOCK)

typedef float(*ptr_f)(float, float, float);

struct Vertex {
	float x, y, z;
};

__constant__ Vertex vert[VERTCOUNT];
//текстура<тип, размерность текстуры, без нормализации> тексурная ссылка:
texture<float, 3, cudaReadModeElementType> df_tex;
//указатель на область памяти, предназначенную для работы с текстурой:
cudaArray * df_Array = 0;
//тестовая функция:
float func(float x, float y, float z) {
	return (0.5*sqrtf(15.0/M_PI))*(0.5*sqrtf(15.0/M_PI))*
		z*z*y*y*sqrtf(1.0f-z*z/RADIUS/RADIUS)/RADIUS/RADIUS
		/RADIUS/RADIUS;	
}
//проверочная фу-ия:
float check(Vertex *v, ptr_f f){
	float sum = 0.0f;
	for (int i = 0; i < VERTCOUNT; ++i)
		sum += f(v[i].x, v[i].y, v[i].z);
	return sum;
}
//дискретизация функций на прямоугольной сетке:
void calc_f(float *arr_f, int x_size, int y_size, int z_size, ptr_f f){
	for (int x = 0; x < x_size; ++x)
		for (int y = 0; y < y_size; ++y)
			for (int z = 0; z < z_size; ++z) {
				arr_f[z_size * (x * y_size + y) + z] = f(x - FGSHIFT, y -
					FGSHIFT, z - FGSHIFT);
				//printf("%f\n", arr_f[z_size * (x * y_size + y) + z]);
				}
}
//определение узлов квадратуры на сфере в константной памяти.
//Котрольное вычисление квадратуры:
void init_vertices(){
	Vertex *temp_vert = (Vertex *)malloc(sizeof(Vertex) * VERTCOUNT);
	int i = 0;
	for (int iphi = 0; iphi < 2 * COEF; ++iphi){
		for (int ipsi = 0; ipsi < COEF; ++ipsi, ++i) {
			float phi = iphi * M_PI / COEF;
			float psi = ipsi * M_PI / COEF;
			temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
			temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
			temp_vert[i].z = RADIUS * cosf(psi);
		}
	}
	printf("sumcheck = %f\n", check(temp_vert, &func)*M_PI*M_PI/
		COEF/COEF);
	//Копирует данные в 1-ый аргумент фу-ии (symbol), который находится на устройстве:
	cudaMemcpyToSymbol(vert, temp_vert, sizeof(Vertex) *
		VERTCOUNT, 0, cudaMemcpyHostToDevice);
	free(temp_vert);
}
//копирование данных с хоста в текстуру:
void init_texture(float *df_h){
	const cudaExtent volumeSize = make_cudaExtent(FGSIZE,
		FGSIZE, FGSIZE);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&df_Array, &channelDesc, volumeSize);
	cudaMemcpy3DParms cpyParams={0};
	cpyParams.srcPtr = make_cudaPitchedPtr( (void*)df_h,
		volumeSize.width*sizeof(float), volumeSize.width,
		volumeSize.height);
	cpyParams.dstArray = df_Array;
	cpyParams.extent = volumeSize;
	cpyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&cpyParams);
	//конфигурация текстуры, параметры фильтрации
	df_tex.normalized = false;
	df_tex.filterMode = cudaFilterModeLinear;
	df_tex.addressMode[0] = cudaAddressModeClamp;
	df_tex.addressMode[1] = cudaAddressModeClamp;
	df_tex.addressMode[2] = cudaAddressModeClamp;
	//привящка текстуры к CUDA массиву
	cudaBindTextureToArray(df_tex, df_Array, channelDesc);
}
//освобожднение ресурсов:
void release_texture(){
	cudaUnbindTexture(df_tex);
	cudaFreeArray(df_Array);
}
//функция ядра для вычисление квадратуры:
//(кэширование фильтрованных значений функции в узлах)
__global__ void kernelTexture(float *a) { //взято из лекции
	__shared__ float cache[THREADSPERBLOCK];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float x = vert[tid].x + FGSHIFT + 0.5f;
	float y = vert[tid].y + FGSHIFT + 0.5f;
	float z = vert[tid].z + FGSHIFT + 0.5f;
	cache[cacheIndex] = tex3D(df_tex, z, y, x);
	__syncthreads();
	//суммирование посредством редукции
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (cacheIndex < s)
			cache[cacheIndex] += cache[cacheIndex + s];
		__syncthreads();
	}
	if (cacheIndex == 0)
		a[blockIdx.x] = cache[0];
}
//ааааааааааааааааааааааааааааааа
__device__ float interpolate1D(float v1, float v2, float x) {
	return v1 * (1 - x) + v2 * x;
}
//ааааааааааааааааааааааааааааааа
__device__ float interpolate2D(float v1, float v2, float v3, float v4, float x, float y) {
	float s = interpolate1D(v1, v2, x);
	float t = interpolate1D(v3, v4, x);
	return interpolate1D(s, t, y);
}
//ааааааааааааааааааааааааааааааа
__device__ float interpolate3D(float *arr_f, float x, float y, float z) {
	int gx = int(x); 
	int gy = int(y);
	int gz = int(z);
	if (gx + 1 >= FGSIZE || gy + 1 >= FGSIZE || gz + 1 >= FGSIZE)
		return 0.0f;
	
	float v1 = arr_f[FGSIZE * (gx * FGSIZE * gy) + gz];
	float v2 = arr_f[FGSIZE * ((gx + 1) * FGSIZE * gy) + gz];
	float v3 = arr_f[FGSIZE * (gx * FGSIZE + (gy + 1)) + gz];
	float v4 = arr_f[FGSIZE * ((gx + 1) * FGSIZE + (gy + 1)) + gz];
	
	float v5 = arr_f[FGSIZE * (gx * FGSIZE * gy) + (gz + 1)];
	float v6 = arr_f[FGSIZE * ((gx + 1) * FGSIZE * gy) + (gz + 1)];
	float v7 = arr_f[FGSIZE * (gx * FGSIZE + (gy + 1)) + (gz + 1)];
	float v8 = arr_f[FGSIZE * ((gx + 1) * FGSIZE + (gy + 1)) + (gz + 1)];
	
	float s = interpolate2D(v1, v2, v3, v4, x - (float)gx, y-(float)gy);
	float t = interpolate2D(v5, v6, v7, v8, x - (float)gx, y-(float)gy);
	
	return interpolate1D(s, t, z - (float)gz);
}
//ааааааааааааааааааааааааааааааа
__global__ void kernelTrilinear(float *sum_dev, float *arr){//, Vertex *v) {
	__shared__ float cache[THREADSPERBLOCK];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float x = vert[tid].x + FGSHIFT;
	float y = vert[tid].y + FGSHIFT;
	float z = vert[tid].z + FGSHIFT;
	/*Vertex temp_vertex;
	temp_vertex.x = x;
	temp_vertex.y = y;
	temp_vertex.z = z;*/
	cache[cacheIndex] = interpolate3D(arr, x, y, z);
	__syncthreads();
	
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (cacheIndex < s)
			cache[cacheIndex] += cache[cacheIndex + s];
		__syncthreads();
	} 
	if (cacheIndex == 0)
		sum_dev[blockIdx.x] = cache[0];

}
//программа-драйвер для тестирования текстурной памяти:
int main(void){
	float *arr = (float *)malloc(sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
	float *sum = (float*)malloc(sizeof(float) * BLOCKSPERGRID);
	float *sum_dev;
	cudaMalloc((void**)&sum_dev, sizeof(float) * BLOCKSPERGRID);
	init_vertices();
	calc_f(arr, FGSIZE, FGSIZE, FGSIZE, &func);
	init_texture(arr);
	
	/*kernelTexture<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(sum_dev);
	cudaThreadSynchronize();
	cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID,cudaMemcpyDeviceToHost);
	float s = 0.0f;
	for (int i = 0; i < BLOCKSPERGRID; ++i)
		s += sum[i];
	printf("Texture sum = %.10f\n", s*M_PI*M_PI / COEF/COEF);
	*/
	
	float *arr_dev;
	cudaMalloc((void**)&arr_dev, sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
	cudaMemcpy(arr_dev, arr, sizeof(float) * FGSIZE * FGSIZE * FGSIZE, cudaMemcpyHostToDevice);
	//проверка массива:
	/*for (int x = 0; x < FGSIZE; ++x)
		for (int y = 0; y < FGSIZE; ++y)
			for (int z = 0; z < FGSIZE; ++z) {
				//arr_f[z_size * (x * y_size + y) + z] = f(x - FGSHIFT, y -
				//	FGSHIFT, z - FGSHIFT);
				printf("%f\n", arr_dev[FGSIZE * (x * FGSIZE + y) + z]);
				}
	*/
	
	cudaEvent_t start, stop; float elapsedTime;
	cudaEventCreate(&start); // инициализация
		cudaEventCreate(&stop); // событий
	cudaEventRecord(start, 0);
	kernelTrilinear<<<VERTCOUNT,THREADSPERBLOCK>>>(sum_dev, arr_dev);//, vert);
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));//	cudaThreadSynchronize();
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaGetLastError());
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID,cudaMemcpyDeviceToHost);
	for(int i = 0; i < BLOCKSPERGRID; i++) {
		printf("%f\n", sum[i]);
	}

	float s = 0.0f;
	for (int i = 0; i < BLOCKSPERGRID; ++i)
		s += sum[i];
	printf("Trilinear sum = %.10f\n, time=%g", s*M_PI*M_PI / COEF/COEF, elapsedTime);

	cudaFree(sum_dev);
	free(sum);
	release_texture();
	free(arr);
	return 0;
}
