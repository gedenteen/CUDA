#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <stdio.h>

__global__ void gTest(float* d) {
	int idx=threadIdx.x+blockDim.x*blockIdx.x;
	d[idx]+=(float)idx;
}

int main(void) {
	float *raw_ptr; //необработанный указатель
#ifdef H2D
	thrust::host_vector<float> h(1<<8);
	thrust::fill(h.begin(), h.end(), 3.1415f);
	thrust::device_vector<float> d = h;
	fprintf(stderr, "Host to device\n");
#else
	thrust::device_vector<float> d(1<<8);
	thrust::fill(d.begin(), d.end(), 3.1415f);
	thrust::host_vector<float> h;
	fprintf(stderr, "Just on device\n");
#endif

	raw_ptr = thrust::raw_pointer_cast(&d[0]); //d.data());
	
	gTest <<< 4, 64 >>> (raw_ptr);
	cudaDeviceSynchronize();
	
	h = d; //thrust::copy(d.begin(), d.end(), h.begin());
	for (int i = 0; i < (1<<8); i++)
		printf("%g\n", h[i]);
		
	//cudaFree(d);//raw_ptr);
	return 0;
}

