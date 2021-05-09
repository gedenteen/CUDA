#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
 
#define BLOCK_DIM 2 //размер субматрицы
int M, K;
 
using namespace std;
 
__global__ void matrixAdd (int *A, int *B, int *C, int M, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
 
    int index = col * M + row;
 
    //сложение на GPU
    if (col < M && row < K) { 
        C[index] = A[index] + B[index];
    }
}
 
int main() {
 
    cout << "M: ";
    cin >> M;
    cout << "K: ";
    cin >> K;
 
    int *A = new int [M*K];
 
 
    int *B = new int [M*K];
 
 
    int *C = new int [M*K];
 
 
    //заполнение матриц
    for(int i=0; i<M; i++)
        for (int j=0; j<K; j++){
            A[i*M+j] = 2;
            B[i*M+j] = 1;
            C[i*M+j] = 0;
        }
 
    int *dev_a, *dev_b, *dev_c; //указатели на выделяемую память
 
    int size = M * K * sizeof(int); //выделяемая память
 
    cudaMalloc((void**)&dev_a, size); //выделение памяти
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
 
    cudaMemcpy(dev_a, A, size, cudaMemcpyHostToDevice); //копирование на GPU
    cudaMemcpy(dev_b, B, size, cudaMemcpyHostToDevice);
 
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); //число выделенных блоков
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (K+dimBlock.y-1)/dimBlock.y); //размер и размерность сетки
    printf("dimGrid.x = %d, dimGrid.y = %d\n", dimGrid.x, dimGrid.y); //выводится размер сетки
 
    matrixAdd<<<dimGrid,dimBlock>>>(dev_a, dev_b, dev_c, M, K); //вызов ядра
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, dev_c, size, cudaMemcpyDeviceToHost);
 
    //вывод результата
    printf("Result Matrix C:\n");
    for(int i=0; i<M; i++){ 
        for (int j=0; j<K; j++){
            printf("%d\t", C[i] );
        }
        printf("\n");
    }
 
 
    cudaFree(dev_a); //освобождение памяти
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
