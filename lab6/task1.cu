#include "task1.h"

__global__ void gInitArray(float* arr) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= M)
        return;
    arr[i] = (float)i;
}
int task1() {
    //long arr_size = 1 << 20;
    float* host_usual_arr, * host_paged_arr, * dev_arr;

    //выделение обычной памяти на хосте:
    host_usual_arr = (float*)malloc(M * sizeof(float));
    //выделение закрепленной (paged-locked) памяти на хосте:
    CUDA_CHECK_RETURN(cudaHostAlloc((void**)&host_paged_arr,
        M * sizeof(float), cudaHostAllocDefault));
    //выделение памяти на девайсе:
    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_arr, M * sizeof(float)));

    gInitArray << < M / 128, 128 >> > (dev_arr);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    float elepsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("time of copying arrays:\n");

    cudaEventRecord(start, 0);
    CUDA_CHECK_RETURN(cudaMemcpy(host_paged_arr, dev_arr, M * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elepsedTime, start, stop);
    printf("device array -> host paged array = %f ms\n", elepsedTime);

    cudaEventRecord(start, 0);
    CUDA_CHECK_RETURN(cudaMemcpy(host_usual_arr, dev_arr, M * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elepsedTime, start, stop);
    printf("device array -> host usual array = %f ms\n", elepsedTime);

    cudaEventRecord(start, 0);
    CUDA_CHECK_RETURN(cudaMemcpy(dev_arr, host_paged_arr, M * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elepsedTime, start, stop);
    printf("host paged array -> device array = %f ms\n", elepsedTime);

    cudaEventRecord(start, 0);
    CUDA_CHECK_RETURN(cudaMemcpy(dev_arr, host_usual_arr, M * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elepsedTime, start, stop);
    printf("host usual array -> device array = %f ms\n", elepsedTime);

    printf("\n");
    free(host_usual_arr);
    cudaFreeHost(host_paged_arr);
    cudaFree(dev_arr);
    return 0;
}
