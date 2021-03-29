#include <stdio.h>
//#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"


#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int* a, int* b, int* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}


__global__ void add_gpu(int* a, int *b, int* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid <= N/256)
        c[tid] = a[tid] + b[tid];
}

int main()
{
    cudaDeviceProp prop;
    int whichDevice;

    // проверяем поддерживает ли устройство overlapping computation with memory copy
    // если пооддерживает  overlap, что всё хорошо 
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        printf("device will not handle\n");
        return 0;
    }

    cudaEvent_t start, stop;
    float elepsedTime;

    // создаём ивенты
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // создаём потоки 
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int* host_a, * host_b, * host_c;
    int* dev_a, * dev_b, * dev_c;

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    //выделение закрепленной (paged-locked) памяти:
    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        kernel << < N / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);
        cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elepsedTime, start, stop);
    printf("Time taken: %3.1f ms\n", elepsedTime);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaStreamDestroy(stream);

    // создание ивентов
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // создание потоков
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    int* dev_a0, * dev_b0, * dev_c0;// первый gpu input buffer for stram0, который будет заполнен рандомными числами
    int* dev_a1, * dev_b1, * dev_c1;// второй gpu input buffer for stram1, который будет заполнен рандомными числами
    //выделение памяти на gpu
    cudaMalloc((void**)&dev_a0, N * sizeof(int));
    cudaMalloc((void**)&dev_b0, N * sizeof(int));
    cudaMalloc((void**)&dev_c0, N * sizeof(int));
    //выделение памяти на gpu
    cudaMalloc((void**)&dev_a1, N * sizeof(int));
    cudaMalloc((void**)&dev_b1, N * sizeof(int));
    cudaMalloc((void**)&dev_c1, N * sizeof(int));
    // выделение page-locked памяти, испльзуемой для стримов
    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = i;//rand();
        host_b[i] = i;// rand();
    }
    // теперь проитерировать всю дату, через байтные куски
    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
        // асинхронно копировать закрытиую память на устройство 
        cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        add_gpu << <N / 256, 256, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);
        // копировать дату с устройства на закрытую память
        cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);

        // асинхронно копировать закрытиую память на устройство 
        cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        add_gpu << <N / 256, 256, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);

        // копировать дату с устройства на закрытую память
        cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }   

    // синхронизируем оба потока
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elepsedTime, start, stop);
    printf("Time taken: %3.1f ms\n", elepsedTime);

  	/*for (int i = 0; i < N / 256; i++) {
        printf("%d+%d=%d\n", host_a[i], host_b[i], host_c[i]);
    }
    */
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);
    cudaStreamDestroy(stream0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);
    cudaStreamDestroy(stream1);
    return 0;
}

