#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

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
//текстура<тип, размерность текстуры, без нормализации> тексурная ссылка;
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
			for (int z = 0; z < z_size; ++z)
				arr_f[z_size * (x * y_size + y) + z] = f(x - FGSHIFT, y -
					FGSHIFT, z - FGSHIFT);
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
__global__ void kernel(float *a)
{
	__shared__ float cache[THREADSPERBLOCK];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float x = vert[tid].x + FGSHIFT + 0.5f;
	float y = vert[tid].y + FGSHIFT + 0.5f;
	float z = vert[tid].z + FGSHIFT + 0.5f;
	cache[cacheIndex] = tex3D(df_tex, z, y, x);
	__syncthreads();
	//суммирование посредством редукции
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cacheIndex < s)
			cache[cacheIndex] += cache[cacheIndex + s];
		__syncthreads();
	}
	if (cacheIndex == 0)
		a[blockIdx.x] = cache[0];
}
//программа-драйвер для тестирования текстурной памяти:
int main(void){
	float *arr = (float *)malloc(sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
	float *sum = (float*)malloc(sizeof(float) * BLOCKSPERGRID);
	float *sum_dev;
	cudaMalloc((void**)&sum_dev, sizeof(float) * BLOCKSPERGRID);
	init_vertices();//init_vertexes();
	calc_f(arr, FGSIZE, FGSIZE, FGSIZE, &func);
	init_texture(arr);
	kernel<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(sum_dev);
	cudaThreadSynchronize();
	cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID,
		cudaMemcpyDeviceToHost);
	
	float s = 0.0f;
	for (int i = 0; i < BLOCKSPERGRID; ++i)
		s += sum[i];
	printf("sum = %f\n", s*M_PI*M_PI / COEF/COEF);
	
	cudaFree(sum_dev);
	free(sum);
	release_texture();
	free(arr);
	return 0;
}
