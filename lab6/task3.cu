#include "task1.h"
#include "task2.h"

__global__ void mul_gpu(int* a, int* b, int* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid <= N / 256)
        c[tid] = abs(a[tid]) * abs(b[tid]) * a[tid] * b[tid] / (abs(a[tid]) * abs(b[tid]));
}
int task3() {
    //cuda-события для замерения времени выполнения:
    cudaEvent_t start, stop;
    float elepsedTime;
    
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    
    printf("\n\tmultiplication of vectors through 2 streams:\n"); 
    for (long var_size = 1024; var_size <= FULL_DATA_SIZE / 2; var_size *= 2) {
        int* host_a, * host_b, * host_c;
        int* dev_a0, * dev_b0, * dev_c0;// первый gpu input buffer for stream0, который будет заполнен рандомными числами
        int* dev_a1, * dev_b1, * dev_c1;// второй gpu input buffer for stream1, который будет заполнен рандомными числами
        //выделение памяти на gpu
        cudaMalloc((void**)&dev_a0, var_size * sizeof(int));
        cudaMalloc((void**)&dev_b0, var_size * sizeof(int));
        cudaMalloc((void**)&dev_c0, var_size * sizeof(int));
        //выделение памяти на gpu
        cudaMalloc((void**)&dev_a1, var_size * sizeof(int));
        cudaMalloc((void**)&dev_b1, var_size * sizeof(int));
        cudaMalloc((void**)&dev_c1, var_size * sizeof(int));
        // выделение page-locked памяти, испльзуемой для стримов
        cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

        for (int i = 0; i < FULL_DATA_SIZE; i++) {
            host_a[i] = i;//rand();
            host_b[i] = i;// rand();
        }
    
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        // теперь проитерировать всю дату, через байтные куски:
        for (int i = 0; i < FULL_DATA_SIZE; i += var_size * 2) {
            // асинхронно копировать закрытую память на устройство:
            cudaMemcpyAsync(dev_a0, host_a + i, var_size * sizeof(int), cudaMemcpyHostToDevice, stream0);
            cudaMemcpyAsync(dev_b0, host_b + i, var_size * sizeof(int), cudaMemcpyHostToDevice, stream0);
            mul_gpu <<< var_size / 256, 256, 0, stream0 >>> (dev_a0, dev_b0, dev_c0);
            // копировать дату с устройства на закрытую памятьЖ
            cudaMemcpyAsync(host_c + i, dev_c0, var_size * sizeof(int), cudaMemcpyDeviceToHost, stream0);

            // асинхронно копировать закрытиую память на устройство 
            cudaMemcpyAsync(dev_a1, host_a + i + var_size, var_size * sizeof(int), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(dev_b1, host_b + i + var_size, var_size * sizeof(int), cudaMemcpyHostToDevice, stream1);
            mul_gpu <<< var_size / 256, 256, 0, stream1 >>> (dev_a1, dev_b1, dev_c1);

            // копировать дату с устройства на закрытую память
            cudaMemcpyAsync(host_c + i + var_size, dev_c1, var_size * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        }   

        // синхронизируем оба потока
        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elepsedTime, start, stop);
        printf("var_size = FULL_DATA_SIZE / %5ld. Time taken: %f ms\n", FULL_DATA_SIZE / var_size, elepsedTime);
        
        //проверка:
        /*
        for (int i = 0; i < N / 256; i++) {
            printf("%d+%d=%d\n", host_a[i], host_b[i], host_c[i]);
        }
        */
        
        cudaFreeHost(host_a);
        cudaFreeHost(host_b);
        cudaFreeHost(host_c);
        cudaFree(dev_a0);
        cudaFree(dev_b0);
        cudaFree(dev_c0);
        cudaFree(dev_a1);
        cudaFree(dev_b1);
        cudaFree(dev_c1);
    }
    
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    return 0;
}
