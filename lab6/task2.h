#ifndef TASK2
#define TASK2

#define N (1024*1024)
#define FULL_DATA_SIZE (N*32)

__global__ void add_gpu(int* a, int* b, int* c);
int task2();

#endif
