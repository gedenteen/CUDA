#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
//#include <cstdio>
//#include <cmath>
#include <stdio.h>
#include <math.h>

struct range_functor {
	float h;
	range_functor(float _h):h(_h){}
	__host__ __device__
	float operator()(float x){
		return h*x;
	}
};

struct sin_functor {
	__device__
	float operator()(float x){
		return __sinf(x);
	}
};

struct elevation_functor {
	__device__
	float operator()(float x) {
		float new_x = x + x;
		return new_x;
	}
};

int main() {
	range_functor R(0.02);
	//sin_functor Sin;
	elevation_functor El;
	
	fprintf(stderr, "%g\n", R(30.0f));
	//fprintf(stderr, "%g\n", Sin(3141592.0f/6.0f));
	thrust::host_vector<float> h1(1 << 8);
	thrust::host_vector<float> h2(1 << 8);
	thrust::device_vector<float> d1(1 << 8);
	thrust::device_vector<float> d2(1 << 8);
	thrust::sequence(thrust::device,d1.begin(), d1.end()); // set the elements 0, 1, 2, 3, ...
	thrust::transform(d1.begin(), d1.end(), d1.begin(), R); // сделать преобразования R в векторе d1
	//thrust::transform(d1.begin(), d1.end(), d2.begin(), Sin); // сделать преобразования Sin в векторе d2
	
	thrust::transform(d1.begin(), d1.end(), d2.begin(), El);
	
	h2=d2;
	h1=d1;
	for(int i=0;i<(1<<8);i++){
		printf("%g\t%g\n",h1[i], h2[i]);
	}
	return 0;
}
