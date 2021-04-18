#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
//#include <thrust/copy.h>

int main(void){
	thrust::host_vector<int> h(1 << 8);
	thrust::generate(h.begin(), h.end(), rand);
	thrust::device_vector<int> d=h;
	thrust::sort(d.begin(), d.end());
	//thrust::copy(d.begin(), d.end(), h.begin());
	h=d;
	for(int i=0;i<1<<8;i++)
		printf("%i\t%d\n",i, h[i]);

	return 0;
}
