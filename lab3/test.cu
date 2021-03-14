#include <stdio.h>

__global__ void MyKernel(double *array, int arrayCount) 
{ 
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  if (idx < arrayCount) 
  { 
    array[idx] = (double)idx; 
  } 
} 

void launchMyKernel(double *array, int arrayCount) 
{ 
  int blockSize;   // The launch configurator returned block size 
  int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
  int gridSize;    // The actual grid size needed, based on input size 

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      MyKernel, 0, 0); 
  // Round up according to array size 
  gridSize = (arrayCount + blockSize - 1) / blockSize; 

  MyKernel<<< gridSize, blockSize >>>(array, arrayCount); 

  cudaDeviceSynchronize(); 

  // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 MyKernel, blockSize, 
                                                 0);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", 
         blockSize, occupancy);
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Error: run program with 1 args: array size\n");
		return 1;
	}

	double *array, arrayCount = atoi(argv[1]);
	cudaMalloc((void**)&array, arrayCount * sizeof(double));
	launchMyKernel(array, arrayCount);

	return 0;
}
